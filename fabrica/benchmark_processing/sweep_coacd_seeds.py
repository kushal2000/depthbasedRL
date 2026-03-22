#!/usr/bin/env python3
"""Sweep CoACD seeds for body (part 1) until all car insertions pass validation."""

import subprocess
import sys

MAX_POS = 3.0   # mm
MAX_KP = 5.0    # mm
PARTS = ["0", "1", "2", "3", "4", "5"]
ASSEMBLY = "car"
BODY_PART = "1"

for seed in range(0, 50):
    sep = "=" * 60
    print(f"\n{sep}")
    print(f"SEED {seed}: Running CoACD for body (part {BODY_PART})...")
    print(sep)

    # Step 4: CoACD for body with this seed
    r = subprocess.run([
        sys.executable, "fabrica/benchmark_processing/step4_run_coacd.py",
        "--assembly", ASSEMBLY, "--part", BODY_PART,
        "--preprocess-mode", "off", "--seed", str(seed),
    ], capture_output=True, text=True)
    for line in r.stdout.split("\n"):
        if "Convex Hulls" in line or "overshoot" in line:
            print(f"  {line.strip()}")

    # Step 5: Regenerate scene URDFs
    subprocess.run([
        sys.executable, "fabrica/benchmark_processing/step5_generate_table_urdfs.py",
        "--assembly", ASSEMBLY,
    ], capture_output=True, text=True)

    # Step 6: Validate all parts
    ts = f"seed_{seed}"
    all_pass = True
    for part in PARTS:
        r = subprocess.run([
            sys.executable, "fabrica/benchmark_processing/step6_validate_insertions.py",
            "--assembly", ASSEMBLY, "--part", part,
            "--method", "coacd", "--timestamp", ts,
        ], capture_output=True, text=True)

        # Parse final pos and kp
        final_pos = final_kp = None
        for line in r.stdout.split("\n"):
            if "Final:" in line:
                val = float(line.split("Final:")[1].strip().split()[0])
                if final_pos is None:
                    final_pos = val
                else:
                    final_kp = val

        if final_pos is None or final_kp is None:
            print(f"  Part {part}: PARSE ERROR")
            print(r.stdout[-500:] if r.stdout else "no stdout")
            print(r.stderr[-500:] if r.stderr else "no stderr")
            all_pass = False
            continue

        passed = final_pos < MAX_POS and final_kp < MAX_KP
        status = "PASS" if passed else "FAIL"
        print(f"  Part {part}: pos={final_pos:.1f}mm kp={final_kp:.1f}mm {status}")
        if not passed:
            all_pass = False

    if all_pass:
        print(f"\n*** ALL PARTS PASS with seed {seed}! ***")
        break
else:
    print("\nNo seed in range 0-49 passed all parts.")
