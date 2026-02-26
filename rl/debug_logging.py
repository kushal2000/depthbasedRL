"""Debug logging for diagnosing multi-GPU SIGSEGV crashes.

Logs GPU memory, NCCL operation health, tensor stats, and installs
signal handlers to dump state on crash.
"""

import atexit
import datetime
import faulthandler
import gc
import logging
import os
import signal
import sys
import threading
import time
import traceback

import torch
import torch.distributed as dist


def setup_debug_logger(rank, log_dir="debug_logs"):
    """Create a per-rank file logger that flushes every write."""
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"rank_{rank}.log")

    logger = logging.getLogger(f"debug_rank_{rank}")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    handler = logging.FileHandler(log_path, mode="w")
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        f"[Rank {rank}] %(asctime)s.%(msecs)03d %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # Also log to stderr so it appears in .err files
    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setLevel(logging.WARNING)
    stderr_handler.setFormatter(formatter)
    logger.addHandler(stderr_handler)

    return logger


def enable_faulthandler(rank, log_dir="debug_logs"):
    """Enable faulthandler to dump Python traceback on SIGSEGV/SIGABRT/SIGFPE.

    This is the single most important thing — it will give us a Python-level
    traceback even when a C extension segfaults.
    """
    os.makedirs(log_dir, exist_ok=True)
    fault_path = os.path.join(log_dir, f"fault_rank_{rank}.log")
    fault_file = open(fault_path, "w")
    faulthandler.enable(file=fault_file, all_threads=True)
    # Also enable on stderr
    faulthandler.enable(file=sys.stderr, all_threads=True)

    # Dump traceback on SIGUSR1 (useful for manually inspecting hung processes)
    if hasattr(faulthandler, "register"):
        faulthandler.register(signal.SIGUSR1, file=fault_file, all_threads=True)

    return fault_file


def log_gpu_memory(logger, rank, tag=""):
    """Log detailed GPU memory stats for the current device."""
    if not torch.cuda.is_available():
        return

    device = torch.device(f"cuda:{rank}")
    allocated = torch.cuda.memory_allocated(device) / (1024 ** 3)
    reserved = torch.cuda.memory_reserved(device) / (1024 ** 3)
    max_allocated = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
    max_reserved = torch.cuda.max_memory_reserved(device) / (1024 ** 3)

    try:
        free, total = torch.cuda.mem_get_info(device)
        free_gb = free / (1024 ** 3)
        total_gb = total / (1024 ** 3)
        used_gb = total_gb - free_gb
        pct = (used_gb / total_gb) * 100
        driver_info = f" | driver: {used_gb:.2f}/{total_gb:.2f} GB ({pct:.1f}%)"
    except Exception:
        driver_info = ""

    msg = (
        f"[GPU MEM {tag}] "
        f"alloc={allocated:.2f} GB, reserved={reserved:.2f} GB, "
        f"max_alloc={max_allocated:.2f} GB, max_reserved={max_reserved:.2f} GB"
        f"{driver_info}"
    )
    logger.info(msg)

    # Warn if memory pressure is high
    if allocated > 0.85 * max_reserved and max_reserved > 0:
        logger.warning(f"HIGH MEMORY PRESSURE: allocated is {allocated/max_reserved*100:.0f}% of max reserved")

    return {
        "allocated_gb": allocated,
        "reserved_gb": reserved,
        "max_allocated_gb": max_allocated,
        "max_reserved_gb": max_reserved,
    }


def check_tensor_health(tensor, name, logger):
    """Check a tensor for NaN/Inf and log stats."""
    if tensor is None:
        return True

    has_nan = torch.isnan(tensor).any().item()
    has_inf = torch.isinf(tensor).any().item()

    if has_nan or has_inf:
        logger.error(
            f"UNHEALTHY TENSOR '{name}': nan={has_nan}, inf={has_inf}, "
            f"shape={tuple(tensor.shape)}, dtype={tensor.dtype}, device={tensor.device}"
        )
        return False
    return True


def log_tensor_stats(tensor, name, logger):
    """Log min/max/mean/std of a tensor."""
    if tensor is None or tensor.numel() == 0:
        return
    with torch.no_grad():
        t = tensor.float()
        logger.debug(
            f"[TENSOR {name}] shape={tuple(tensor.shape)}, "
            f"min={t.min().item():.6f}, max={t.max().item():.6f}, "
            f"mean={t.mean().item():.6f}, std={t.std().item():.6f}"
        )


def timed_dist_op(op_fn, op_name, logger, warn_threshold_s=30.0):
    """Execute a distributed op with timing and timeout warning.

    Returns the result of op_fn().
    """
    logger.debug(f"[DIST] Starting {op_name}")
    start = time.monotonic()
    result = op_fn()
    elapsed = time.monotonic() - start
    level = logging.WARNING if elapsed > warn_threshold_s else logging.DEBUG
    logger.log(level, f"[DIST] {op_name} took {elapsed:.3f}s")
    return result


class PeriodicMemoryMonitor:
    """Background thread that logs GPU memory at fixed intervals."""

    def __init__(self, logger, rank, interval_s=60):
        self.logger = logger
        self.rank = rank
        self.interval_s = interval_s
        self._stop = threading.Event()
        self._thread = None

    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)

    def _run(self):
        while not self._stop.wait(self.interval_s):
            try:
                log_gpu_memory(self.logger, self.rank, tag="periodic")
            except Exception as e:
                self.logger.error(f"Memory monitor error: {e}")


class DistributedDebugger:
    """Wraps all distributed operations with logging and health checks.

    Usage:
        dbg = DistributedDebugger(rank, world_size, log_dir="debug_logs")
        # ... in training loop ...
        dbg.log_epoch_start(epoch)
        dbg.log_gpu_memory("before_forward")
        dbg.all_reduce_with_check(tensor, "gradients")
        dbg.broadcast_with_check(tensor, src=0, name="params")
        dbg.log_epoch_end(epoch)
    """

    def __init__(self, rank, world_size, log_dir="debug_logs", memory_interval_s=60):
        self.rank = rank
        self.world_size = world_size
        self.log_dir = log_dir

        self.logger = setup_debug_logger(rank, log_dir)
        self.fault_file = enable_faulthandler(rank, log_dir)

        self.logger.info(f"=== Debug logging initialized ===")
        self.logger.info(f"rank={rank}, world_size={world_size}, pid={os.getpid()}")
        self.logger.info(f"hostname={os.uname().nodename}")
        self.logger.info(f"CUDA device: {torch.cuda.get_device_name(rank)}")
        self.logger.info(f"PyTorch version: {torch.__version__}")
        self.logger.info(f"CUDA version: {torch.version.cuda}")

        # Set NCCL debug env vars (these take effect for future NCCL calls)
        # WARN level gives useful info without being too noisy
        if "NCCL_DEBUG" not in os.environ:
            os.environ["NCCL_DEBUG"] = "WARN"
        self.logger.info(f"NCCL_DEBUG={os.environ.get('NCCL_DEBUG', 'unset')}")
        self.logger.info(f"NCCL_TIMEOUT (env): {os.environ.get('NCCL_TIMEOUT', 'unset')}")

        # Log initial memory
        log_gpu_memory(self.logger, rank, tag="init")

        # Start background memory monitor
        self.mem_monitor = PeriodicMemoryMonitor(self.logger, rank, interval_s=memory_interval_s)
        self.mem_monitor.start()

        # Epoch tracking
        self.epoch_start_time = None
        self._dist_op_count = 0

        # Register cleanup
        atexit.register(self._cleanup)

    def _cleanup(self):
        self.logger.info("=== Shutting down debug logger ===")
        self.mem_monitor.stop()
        if self.fault_file:
            self.fault_file.close()

    def log_epoch_start(self, epoch):
        self.epoch_start_time = time.monotonic()
        self._dist_op_count = 0
        self.logger.info(f"{'='*60}")
        self.logger.info(f"EPOCH {epoch} START")
        log_gpu_memory(self.logger, self.rank, tag=f"epoch_{epoch}_start")
        # Reset peak memory stats each epoch to track per-epoch growth
        torch.cuda.reset_peak_memory_stats(self.rank)

    def log_epoch_end(self, epoch):
        elapsed = time.monotonic() - self.epoch_start_time if self.epoch_start_time else 0
        self.logger.info(f"EPOCH {epoch} END (took {elapsed:.2f}s, {self._dist_op_count} dist ops)")
        mem = log_gpu_memory(self.logger, self.rank, tag=f"epoch_{epoch}_end")
        # Force a garbage collection to see true memory usage
        gc.collect()
        torch.cuda.empty_cache()
        log_gpu_memory(self.logger, self.rank, tag=f"epoch_{epoch}_post_gc")

    def log_phase(self, phase_name):
        """Log a named phase within an epoch (e.g., 'play_steps', 'calc_gradients')."""
        self.logger.debug(f"--- {phase_name} ---")
        log_gpu_memory(self.logger, self.rank, tag=phase_name)

    def all_reduce_with_check(self, tensor, name, op=dist.ReduceOp.SUM):
        """All-reduce with health checking and timing."""
        self._dist_op_count += 1

        # Pre-check: zero out NaN/Inf so all ranks still participate in the
        # all_reduce (keeping NCCL in sync).  The zeroed gradients effectively
        # skip this rank's contribution for the bad minibatch.
        healthy = check_tensor_health(tensor, f"{name}_pre_allreduce", self.logger)
        if not healthy:
            self.logger.error(f"NaN/Inf in '{name}' pre-allreduce — zeroing tensor to keep NCCL in sync")
            tensor.zero_()

        self.logger.debug(
            f"[DIST] all_reduce '{name}': shape={tuple(tensor.shape)}, "
            f"dtype={tensor.dtype}, numel={tensor.numel()}"
        )

        start = time.monotonic()
        dist.all_reduce(tensor, op=op)
        elapsed = time.monotonic() - start

        if elapsed > 30.0:
            self.logger.warning(f"[DIST] all_reduce '{name}' SLOW: {elapsed:.3f}s")
        else:
            self.logger.debug(f"[DIST] all_reduce '{name}' took {elapsed:.3f}s")

        # Post-check
        check_tensor_health(tensor, f"{name}_post_allreduce", self.logger)

    def broadcast_with_check(self, tensor, src, name):
        """Broadcast with health checking and timing."""
        self._dist_op_count += 1

        check_tensor_health(tensor, f"{name}_pre_broadcast", self.logger)

        self.logger.debug(
            f"[DIST] broadcast '{name}': shape={tuple(tensor.shape)}, "
            f"src={src}, numel={tensor.numel()}"
        )

        start = time.monotonic()
        dist.broadcast(tensor, src)
        elapsed = time.monotonic() - start

        if elapsed > 30.0:
            self.logger.warning(f"[DIST] broadcast '{name}' SLOW: {elapsed:.3f}s")
        else:
            self.logger.debug(f"[DIST] broadcast '{name}' took {elapsed:.3f}s")

        check_tensor_health(tensor, f"{name}_post_broadcast", self.logger)

    def log_grad_stats(self, model, model_name="model"):
        """Log gradient statistics for a model."""
        total_norm = 0.0
        num_params_with_grad = 0
        max_grad = 0.0
        has_nan = False
        has_inf = False

        for name, p in model.named_parameters():
            if p.grad is not None:
                num_params_with_grad += 1
                grad_norm = p.grad.data.norm(2).item()
                total_norm += grad_norm ** 2
                max_grad = max(max_grad, grad_norm)
                if torch.isnan(p.grad).any():
                    has_nan = True
                    self.logger.error(f"NaN gradient in {model_name}.{name}")
                if torch.isinf(p.grad).any():
                    has_inf = True
                    self.logger.error(f"Inf gradient in {model_name}.{name}")

        total_norm = total_norm ** 0.5
        self.logger.debug(
            f"[GRADS {model_name}] total_norm={total_norm:.6f}, "
            f"max_single={max_grad:.6f}, params_w_grad={num_params_with_grad}, "
            f"nan={has_nan}, inf={has_inf}"
        )

        if has_nan or has_inf:
            self.logger.error(f"CORRUPT GRADIENTS in {model_name}!")
