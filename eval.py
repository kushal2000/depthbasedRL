#!/usr/bin/env python3
"""Minimal evaluation script for IsaacGym and Isaac Lab.

Usage:
    # IsaacGym (default MLP):
    python eval.py --sim isaacgym --checkpoint model.pth

    # IsaacGym (LSTM asymmetric, matching pretrained checkpoint):
    python eval.py --sim isaacgym --checkpoint pretrained_policy/model.pth \
        --task SimToolRealLSTMAsymmetric --train SimToolRealLSTMAsymmetricPPO \
        --override task.env.dofSpeedScale=1.5 task.env.useActionDelay=False

    # Isaac Lab:
    python eval.py --sim isaaclab --checkpoint model.pth

    # Options:
    python eval.py --sim isaacgym --checkpoint model.pth --num_envs 1024 --num_episodes 200 --gpu 0
"""

import argparse
import os
import sys


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate a trained policy")
    p.add_argument("--sim", required=True, choices=["isaacgym", "isaaclab"])
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--num_envs", type=int, default=1024)
    p.add_argument("--num_episodes", type=int, default=200,
                   help="Minimum episodes to collect before reporting")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--deterministic", action="store_true",
                   help="Use mean action (no sampling)")
    # IsaacGym Hydra overrides
    p.add_argument("--task", type=str, default=None,
                   help="Hydra task config name (e.g. SimToolRealLSTMAsymmetric)")
    p.add_argument("--train", type=str, default=None,
                   help="Hydra train config name (e.g. SimToolRealLSTMAsymmetricPPO)")
    p.add_argument("--override", nargs="*", default=[],
                   help="Extra Hydra overrides (e.g. task.env.dofSpeedScale=1.5)")
    return p.parse_args()


def eval_isaacgym(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    from isaacgym import gymapi, gymtorch, gymutil  # noqa: F401
    import torch
    import hydra
    from omegaconf import OmegaConf, open_dict

    from envs.isaacgym.reformat import omegaconf_to_dict
    from envs.isaacgym.utils import set_np_formatting, set_seed
    from envs.isaacgym import isaacgym_task_map
    from rl.agent import SAPGAgent
    from rl.vec_env import IsaacVecEnv

    overrides = [
        "headless=True",
        f"task.env.numEnvs={args.num_envs}",
    ]
    if args.task:
        overrides.append(f"task={args.task}")
    if args.train:
        overrides.append(f"train={args.train}")
    overrides.extend(args.override)

    with hydra.initialize(version_base="1.1", config_path="./envs/isaacgym/cfg"):
        cfg = hydra.compose(config_name="config", overrides=overrides)

    set_np_formatting()
    cfg.seed = set_seed(cfg.seed)

    task_config = omegaconf_to_dict(cfg.task)
    raw_env = isaacgym_task_map[cfg.task_name](
        cfg=task_config,
        rl_device="cuda:0",
        sim_device="cuda:0",
        graphics_device_id=0,
        headless=True,
        virtual_screen_capture=False,
        force_render=True,
    )
    vec_env = IsaacVecEnv(raw_env)

    train_cfg = omegaconf_to_dict(cfg.train)
    config = train_cfg['params']['config']
    config['device'] = 'cuda:0'
    config['multi_gpu'] = False
    config['num_actors'] = args.num_envs
    config['population_based_training'] = False
    config['pbt_idx'] = None
    config['full_experiment_name'] = 'eval'
    # Ensure block size divides num_envs
    block_size = config.get('expl_coef_block_size', 4096)
    if args.num_envs % block_size != 0:
        config['expl_coef_block_size'] = args.num_envs

    agent = SAPGAgent(vec_env, config, writer=None)
    agent.restore(args.checkpoint, set_epoch=False)

    return agent, vec_env


def eval_isaaclab(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ.pop("DISPLAY", None)

    sys.argv.extend([
        f"--/renderer/activeGpu={args.gpu}",
        "--/renderer/multiGpu/enabled=false",
        "--/crashreporter/enabled=false",
    ])

    from isaaclab.app import AppLauncher
    app_launcher = AppLauncher(headless=True, enable_cameras=False)

    import gymnasium
    import envs.isaaclab  # noqa: F401
    from envs.isaaclab.sim_tool_real_cfg import SimToolRealEnvCfg
    from rl.agent import SAPGAgent
    from rl.vec_env import IsaacLabVecEnv

    env_cfg = SimToolRealEnvCfg()
    env_cfg.sim.device = "cuda:0"
    env_cfg.scene.num_envs = args.num_envs

    env = gymnasium.make("SimToolReal-Direct-v0", cfg=env_cfg)
    vec_env = IsaacLabVecEnv(env)

    config = {
        'name': 'eval',
        'device': 'cuda:0',
        'multi_gpu': False,
        'ppo': True,
        'mixed_precision': False,
        'normalize_input': True,
        'normalize_value': True,
        'normalize_advantage': True,
        'reward_shaper': {'scale_value': 0.01},
        'num_actors': args.num_envs,
        'gamma': 0.99,
        'tau': 0.95,
        'learning_rate': 1e-4,
        'lr_schedule': 'adaptive',
        'schedule_type': 'standard',
        'kl_threshold': 0.016,
        'score_to_win': 1_000_000,
        'max_epochs': 1,
        'max_frames': 1,
        'save_best_after': 999999,
        'save_frequency': 999999,
        'print_stats': False,
        'grad_norm': 1.0,
        'entropy_coef': 0.0,
        'truncate_grads': True,
        'e_clip': 0.1,
        'minibatch_size': min(32768, args.num_envs * 16),
        'mini_epochs': 2,
        'critic_coef': 4.0,
        'clip_value': True,
        'horizon_length': 16,
        'seq_length': 16,
        'bounds_loss_coef': 0.0001,
        'expl_type': 'none',
        'central_value_config': {
            'minibatch_size': min(32768, args.num_envs * 16),
            'mini_epochs': 2,
            'learning_rate': 1e-4,
            'kl_threshold': 0.016,
            'clip_value': True,
            'normalize_input': True,
            'truncate_grads': True,
        },
        'population_based_training': False,
        'pbt_idx': None,
        'full_experiment_name': 'eval',
        'value_bootstrap': True,
    }

    agent = SAPGAgent(vec_env, config, writer=None)
    agent.restore(args.checkpoint, set_epoch=False)

    return agent, vec_env


def run_eval(agent, vec_env, num_episodes, deterministic=False):
    """Collect episodes and report stats."""
    import torch
    import numpy as np
    obs = vec_env.reset()
    agent.obs = obs

    num_envs = obs['obs'].shape[0]
    episode_rewards = []
    episode_lengths = []
    running_rewards = torch.zeros(num_envs, device=obs['obs'].device)
    running_lengths = torch.zeros(num_envs, device=obs['obs'].device)

    # Initialize RNN states (works for both MLP and LSTM)
    rnn_states = agent.model.get_default_rnn_state()
    rnn_states = [s.to(obs['obs'].device) for s in rnn_states]

    # SAPG appends exploration coef embedding to obs during training;
    # replicate here so the network input dimensions match
    has_sapg = agent.intr_reward_coef_embd is not None
    if has_sapg:
        sapg_embd = agent.intr_reward_coef_embd

    def _append_sapg(obs_dict):
        if has_sapg:
            obs_dict = dict(obs_dict)  # shallow copy
            obs_dict['obs'] = torch.cat([obs_dict['obs'], sapg_embd], dim=1)
            if 'states' in obs_dict:
                obs_dict['states'] = torch.cat([obs_dict['states'], sapg_embd], dim=1)
        return obs_dict

    print(f"Collecting {num_episodes} episodes across {num_envs} envs...")
    step = 0
    while len(episode_rewards) < num_episodes:
        res = agent.get_action_values(_append_sapg(obs), rnn_states)
        actions = res['mus'] if deterministic else res['actions']
        rnn_states = res['rnn_states']

        obs, rewards, dones, infos = vec_env.step(actions)

        running_rewards += rewards
        running_lengths += 1

        done_indices = dones.nonzero(as_tuple=False).squeeze(-1)
        if done_indices.numel() > 0:
            for idx in done_indices:
                episode_rewards.append(running_rewards[idx].item())
                episode_lengths.append(running_lengths[idx].item())
            running_rewards[done_indices] = 0
            running_lengths[done_indices] = 0

            # Reset RNN states for done envs
            if rnn_states is not None and isinstance(rnn_states, (list, tuple)):
                for s in rnn_states:
                    s[:, done_indices, :] = 0

        step += 1
        if step % 100 == 0:
            print(f"  step {step}, episodes collected: {len(episode_rewards)}/{num_episodes}")

    episode_rewards = np.array(episode_rewards[:num_episodes])
    episode_lengths = np.array(episode_lengths[:num_episodes])

    print(f"\n{'='*50}")
    print(f"Evaluation Results ({num_episodes} episodes)")
    print(f"{'='*50}")
    print(f"  Mean reward:   {episode_rewards.mean():.2f} +/- {episode_rewards.std():.2f}")
    print(f"  Median reward: {np.median(episode_rewards):.2f}")
    print(f"  Min / Max:     {episode_rewards.min():.2f} / {episode_rewards.max():.2f}")
    print(f"  Mean length:   {episode_lengths.mean():.1f}")
    print(f"{'='*50}")

    return episode_rewards, episode_lengths


def main():
    args = parse_args()

    if args.sim == "isaacgym":
        agent, vec_env = eval_isaacgym(args)
    else:
        agent, vec_env = eval_isaaclab(args)

    run_eval(agent, vec_env, args.num_episodes, args.deterministic)


if __name__ == "__main__":
    main()
