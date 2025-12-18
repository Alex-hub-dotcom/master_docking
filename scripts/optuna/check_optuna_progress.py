#!/usr/bin/env python3
"""
Check Optuna study progress and best results.

Usage:
    python check_optuna_progress.py
"""

import os
import sys

try:
    import optuna
    from optuna.trial import TrialState
except ImportError:
    print("ERROR: optuna not installed")
    sys.exit(1)

STORAGE_PATH = "/home/schux00/optuna/teko_study.db"
STUDY_NAME = "teko_ppo_optimization_v1"

def main():
    if not os.path.exists(STORAGE_PATH):
        print(f"❌ Study database not found: {STORAGE_PATH}")
        print("   Run create_optuna_study.sh first.")
        return
    
    storage = f"sqlite:///{STORAGE_PATH}"
    
    try:
        study = optuna.load_study(study_name=STUDY_NAME, storage=storage)
    except Exception as e:
        print(f"❌ Failed to load study: {e}")
        return
    
    trials = study.trials
    
    complete = [t for t in trials if t.state == TrialState.COMPLETE]
    running = [t for t in trials if t.state == TrialState.RUNNING]
    pruned = [t for t in trials if t.state == TrialState.PRUNED]
    failed = [t for t in trials if t.state == TrialState.FAIL]
    
    print("=" * 60)
    print(f"OPTUNA STUDY: {STUDY_NAME}")
    print("=" * 60)
    print(f"Total trials:    {len(trials)}")
    print(f"  ✅ Complete:   {len(complete)}")
    print(f"  🔄 Running:    {len(running)}")
    print(f"  ✂️  Pruned:     {len(pruned)}")
    print(f"  ❌ Failed:     {len(failed)}")
    print()
    
    if complete:
        print("=" * 60)
        print("BEST TRIAL")
        print("=" * 60)
        best = study.best_trial
        print(f"Trial #{best.number}")
        print(f"Value (SSR + stage bonus): {best.value:.4f}")
        print()
        print("Hyperparameters:")
        for k, v in best.params.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.6f}")
            else:
                print(f"  {k}: {v}")
        print()
        
        print("=" * 60)
        print("TOP 5 TRIALS")
        print("=" * 60)
        sorted_trials = sorted(complete, key=lambda t: t.value if t.value else 0, reverse=True)
        for i, t in enumerate(sorted_trials[:5]):
            print(f"#{i+1} Trial {t.number}: {t.value:.4f}")
            print(f"    entropy={t.params.get('entropy_coef', '?'):.4f}, "
                  f"gae={t.params.get('gae_lambda', '?'):.3f}, "
                  f"clip={t.params.get('clip_ratio', '?')}, "
                  f"epochs={t.params.get('epochs', '?')}")
        print()
    
    if running:
        print("=" * 60)
        print("RUNNING TRIALS")
        print("=" * 60)
        for t in running:
            print(f"  Trial #{t.number}")
        print()
    
    print("=" * 60)
    print("COMMANDS")
    print("=" * 60)
    print("Launch more workers:")
    print("  ./launch_optuna_workers.sh 4")
    print()
    print("View worker logs:")
    print("  tail -f /home/schux00/logs/optuna_*.out")
    print()
    print("Cancel all jobs:")
    print("  scancel -u schux00")
    print()


if __name__ == "__main__":
    main()