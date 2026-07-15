#!/usr/bin/env python3
"""
optimize_pid_6joint.py
======================
Uses Optuna to find the best PID gains for all 6 joints of the Baloo
hugger robot independently.

WHAT IS TUNED (24 parameters total)
-------------------------------------
For each of the 6 joints: kp, ki, kd, threshold
  left_j0_kp,  left_j0_ki,  left_j0_kd,  left_j0_threshold
  left_j1_kp,  left_j1_ki,  left_j1_kd,  left_j1_threshold
  left_j2_kp,  left_j2_ki,  left_j2_kd,  left_j2_threshold
  right_j0_kp, right_j0_ki, right_j0_kd, right_j0_threshold
  right_j1_kp, right_j1_ki, right_j1_kd, right_j1_threshold
  right_j2_kp, right_j2_ki, right_j2_kd, right_j2_threshold

WHAT IS FIXED (6 parameters — 99th percentile safety ceilings)
----------------------------------------------------------------
  left_j0  correction_max = 299.3 kPa
  left_j1  correction_max = 299.9 kPa
  left_j2  correction_max = 163.0 kPa
  right_j0 correction_max = 299.4 kPa
  right_j1 correction_max = 276.2 kPa
  right_j2 correction_max = 187.1 kPa

USAGE
-----
    python optimize_pid_6joint.py                        # defaults
    python optimize_pid_6joint.py --n_trials 100 --n_eval 30
    python optimize_pid_6joint.py --perturbation 100     # 100N perturbation
"""

import os
import sys
import json
import argparse
import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)

sys.path.append('/home/cameronc/baloo_ws/src/baloo-gym/src')
sys.path.append('/home/cameronc/baloo_ws/src')
from evaluate_pid_1000 import run_single_trial


# ─────────────────────────────────────────────────────────────────────────────
# Joint names — must match keys used in PIDHugger.per_joint_pid_params
# ─────────────────────────────────────────────────────────────────────────────
JOINT_NAMES = [
    'left_j0', 'left_j1', 'left_j2',
    'right_j0', 'right_j1', 'right_j2',
]

# ─────────────────────────────────────────────────────────────────────────────
# Fixed correction_max per joint — Optuna never touches these
# Values are 99th-percentile max pressures from your evaluation data (kPa)
# ─────────────────────────────────────────────────────────────────────────────
CORRECTION_MAX = {
    'left_j0':  149.3,
    'left_j1':  149.9,
    'left_j2':  13.0,
    'right_j0': 149.4,
    'right_j1': 126.2,
    'right_j2': 37.1,
}

# ─────────────────────────────────────────────────────────────────────────────
# Search space — same ranges for every joint
# Optuna picks kp, ki, kd, threshold independently per joint (24 params total)
# ─────────────────────────────────────────────────────────────────────────────
PARAM_RANGES = {
    'kp':        (10.0,  200.0),
    'ki':        ( 0.0,   20.0),
    'kd':        ( 0.0,   10.0),
    'threshold': ( 0.001,  0.1),
}

# ─────────────────────────────────────────────────────────────────────────────
# Baseline gains (your current known-good single-PID values broadcast to all
# joints) — used as trial 0 so you see the baseline before Optuna explores
# ─────────────────────────────────────────────────────────────────────────────
BASELINE_GAINS = {
    joint: {
        'kp': 105.123335, 'ki': 12.854383,
        'kd': 4.886744,   'threshold': 0.092517,
    }
    for joint in JOINT_NAMES
}


# ─────────────────────────────────────────────────────────────────────────────
# Build per-joint pid config from a flat Optuna params dict
# ─────────────────────────────────────────────────────────────────────────────
def build_per_joint_config(optuna_params: dict) -> dict:
    """
    Converts Optuna's flat param dict like:
        {'left_j0_kp': 97.4, 'left_j0_ki': 3.1, ..., 'right_j2_threshold': 0.05}
    into the nested per-joint dict PIDHugger expects:
        {'left_j0': {'kp': 97.4, 'ki': 3.1, ..., 'correction_max': 299.3}, ...}
    correction_max is always injected from CORRECTION_MAX — never from Optuna.
    """
    config = {}
    for joint in JOINT_NAMES:
        config[joint] = {
            param: optuna_params[f'{joint}_{param}']
            for param in PARAM_RANGES
        }
        config[joint]['correction_max'] = CORRECTION_MAX[joint]
    return config


# ─────────────────────────────────────────────────────────────────────────────
# Objective function
# ─────────────────────────────────────────────────────────────────────────────
def make_objective(n_eval: int, perturbation: float):
    def objective(trial: optuna.Trial) -> float:

        # ── 1. Ask Optuna for 24 values (4 params × 6 joints) ────────────
        flat_params = {}
        for joint in JOINT_NAMES:
            for param, (lo, hi) in PARAM_RANGES.items():
                flat_params[f'{joint}_{param}'] = trial.suggest_float(
                    f'{joint}_{param}', lo, hi
                )

        per_joint_config = build_per_joint_config(flat_params)

        # ── 2. Run n_eval headless episodes ──────────────────────────────
        successes = 0
        halfway   = n_eval // 2

        for episode in range(n_eval):
            try:
                result, _, _ = run_single_trial(
                    perturbation_magnitude=perturbation,
                    render_mode=None,
                    record_all_steps=False,
                    per_joint_pid_params=per_joint_config,  # 6-joint config
                )
                if result['success']:
                    successes += 1
            except Exception as e:
                print(f"  [warn] episode {episode} raised: {e}")

            # Halfway pruning — abandon clearly bad candidates early
            if episode == halfway - 1:
                interim_rate = successes / halfway
                trial.report(interim_rate, step=episode)
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()

        # ── 3. Return success rate ────────────────────────────────────────
        return successes / n_eval

    return objective


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Optimise Baloo 6-joint PID gains with Optuna")
    parser.add_argument(
        '--n_trials', type=int, default=50,
        help='Number of PID candidates Optuna will try (default 50)')
    parser.add_argument(
        '--n_eval', type=int, default=20,
        help='Simulation episodes per candidate (default 20)')
    parser.add_argument(
        '--perturbation', type=float, default=0.0,
        help='Perturbation magnitude in Newtons (default 0)')
    parser.add_argument(
        '--output', type=str, default='best_pid_6joint_gains.json',
        help='Where to save best gains (default: best_pid_6joint_gains.json)')
    args = parser.parse_args()

    print("=" * 65)
    print("  Baloo 6-Joint PID Optimiser — powered by Optuna")
    print("=" * 65)
    print(f"  Joints tuned       : {', '.join(JOINT_NAMES)}")
    print(f"  Parameters / joint : kp, ki, kd, threshold  (4 × 6 = 24)")
    print(f"  Fixed / joint      : correction_max  (6 values, never tuned)")
    print(f"  PID candidates     : {args.n_trials}")
    print(f"  Episodes/candidate : {args.n_eval}")
    print(f"  Total sim episodes : {args.n_trials * args.n_eval:,}")
    print(f"  Perturbation       : {args.perturbation} N")
    print("=" * 65)
    print()
    print("  Fixed correction_max values (kPa):")
    for joint, val in CORRECTION_MAX.items():
        print(f"    {joint:<12} = {val}")
    print()

    # ── Create study ─────────────────────────────────────────────────────
    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=10,
        ),
    )

    # ── Seed trial 0 with your current best single-PID gains ─────────────
    # This lets you immediately see how the baseline performs before
    # Optuna starts exploring the 24-dimensional space.
    baseline_flat = {}
    for joint in JOINT_NAMES:
        for param in PARAM_RANGES:
            baseline_flat[f'{joint}_{param}'] = BASELINE_GAINS[joint][param]
    study.enqueue_trial(baseline_flat)

    # ── Progress callback ─────────────────────────────────────────────────
    def progress_callback(study, trial):
        best   = study.best_value * 100
        status = ("PRUNED" if trial.value is None
                  else f"{trial.value * 100:.1f}%")
        print(f"  Trial {trial.number:>3d}/{args.n_trials}  "
              f"this={status:<8}  best={best:.1f}%")

    # ── Run ───────────────────────────────────────────────────────────────
    objective = make_objective(args.n_eval, args.perturbation)
    study.optimize(objective, n_trials=args.n_trials,
                   callbacks=[progress_callback])

    # ── Results ───────────────────────────────────────────────────────────
    best_flat  = study.best_params
    best_rate  = study.best_value * 100
    best_config = build_per_joint_config(best_flat)

    print()
    print("=" * 65)
    print("  OPTIMISATION COMPLETE")
    print("=" * 65)
    print(f"  Best success rate : {best_rate:.1f}%")
    print()
    print("  Best gains per joint:")
    for joint in JOINT_NAMES:
        cfg = best_config[joint]
        print(f"    {joint:<12}  kp={cfg['kp']:.3f}  ki={cfg['ki']:.3f}  "
              f"kd={cfg['kd']:.3f}  threshold={cfg['threshold']:.4f}  "
              f"correction_max={cfg['correction_max']}")
    print()

    # ── Save ──────────────────────────────────────────────────────────────
    output_data = {
        'best_success_rate_pct': round(best_rate, 2),
        'perturbation_N':        args.perturbation,
        'n_trials':              args.n_trials,
        'n_eval_per_trial':      args.n_eval,
        'fixed_correction_max':  CORRECTION_MAX,
        'best_per_joint_config': best_config,
        'baseline_per_joint_config': {
            joint: {**BASELINE_GAINS[joint],
                    'correction_max': CORRECTION_MAX[joint]}
            for joint in JOINT_NAMES
        },
    }
    with open(args.output, 'w') as f:
        import json
        json.dump(output_data, f, indent=2)

    print(f"  Results saved to  : {args.output}")
    print()
    print("  To use these gains, pass per_joint_pid_params into")
    print("  OpenLoopHuggerPolicy:")
    print()
    print(f"    per_joint_pid_params = {best_config}")


if __name__ == '__main__':
    main()
