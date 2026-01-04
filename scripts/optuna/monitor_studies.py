#!/usr/bin/env python3
# =============================================================================
# TEKO Optuna Study Monitor
# =============================================================================
# Shows progress of vision and state-based training studies.
#
# Usage:
#   python monitor_studies.py           # Show current status
#   python monitor_studies.py --watch   # Continuous monitoring
#   python monitor_studies.py --pareto  # Show Pareto front
#
# Author: Alexandre Schleier Neves da Silva
# =============================================================================

import argparse
import sqlite3
import time
import os
from datetime import datetime, timedelta


VISION_DB = "/home/schux00/optuna/teko_vision_v6.db"
STATE_DB = "/home/schux00/optuna/teko_state_v6.db"


def get_study_stats(db_path: str) -> dict:
    """Get statistics from Optuna study database."""
    if not os.path.exists(db_path):
        return {"exists": False}
    
    try:
        conn = sqlite3.connect(db_path, timeout=10)
        cur = conn.cursor()
        
        # Count trials by state
        cur.execute("SELECT state, COUNT(*) FROM trials GROUP BY state")
        states = dict(cur.fetchall())
        
        # State codes: 0=RUNNING, 1=COMPLETE, 2=PRUNED, 3=FAIL, 4=WAITING
        state_names = {0: "running", 1: "complete", 2: "pruned", 3: "failed", 4: "waiting"}
        state_counts = {name: states.get(code, 0) for code, name in state_names.items()}
        
        # Get best trials (Pareto front approximation)
        cur.execute("""
            SELECT trial_id, value_0, value_1 
            FROM trial_values 
            WHERE objective = 0
        """)
        values_0 = {row[0]: row[1] for row in cur.fetchall()}
        
        cur.execute("""
            SELECT trial_id, value_0, value_1 
            FROM trial_values 
            WHERE objective = 1
        """)
        values_1 = {row[0]: row[1] for row in cur.fetchall()}
        
        # Combine objectives
        best_ssr = 0.0
        best_stage = 0
        pareto_front = []
        
        for tid in values_0:
            if tid in values_1:
                ssr = values_0[tid]
                stage = values_1[tid]
                if ssr is not None and stage is not None:
                    best_ssr = max(best_ssr, ssr)
                    best_stage = max(best_stage, int(stage))
                    pareto_front.append((ssr, stage, tid))
        
        # Get recent trial info
        cur.execute("""
            SELECT datetime_start, datetime_complete 
            FROM trials 
            WHERE state = 1 
            ORDER BY datetime_complete DESC 
            LIMIT 1
        """)
        last_complete = cur.fetchone()
        
        conn.close()
        
        return {
            "exists": True,
            "total": sum(state_counts.values()),
            **state_counts,
            "best_ssr": best_ssr,
            "best_stage": best_stage,
            "pareto_front": sorted(pareto_front, key=lambda x: -x[0])[:10],
            "last_complete": last_complete,
        }
        
    except Exception as e:
        return {"exists": True, "error": str(e)}


def print_study_status(name: str, stats: dict):
    """Print formatted study status."""
    print(f"\n{'=' * 50}")
    print(f"  {name}")
    print(f"{'=' * 50}")
    
    if not stats.get("exists"):
        print("  Database not found")
        return
    
    if "error" in stats:
        print(f"  Error: {stats['error']}")
        return
    
    print(f"  Total trials: {stats['total']}")
    print(f"  ├─ Complete: {stats['complete']}")
    print(f"  ├─ Running:  {stats['running']}")
    print(f"  ├─ Pruned:   {stats['pruned']}")
    print(f"  └─ Failed:   {stats['failed']}")
    print()
    print(f"  Best SSR:   {stats['best_ssr']:.1%}")
    print(f"  Max Stage:  S{stats['best_stage']}")
    
    if stats.get("pareto_front"):
        print()
        print("  Top trials (Pareto front):")
        for i, (ssr, stage, tid) in enumerate(stats["pareto_front"][:5]):
            print(f"    {i+1}. Trial {tid}: SSR={ssr:.1%}, Stage={int(stage)}")


def print_combined_status():
    """Print status of both studies."""
    print("\n" + "=" * 60)
    print(f"  TEKO Training Monitor - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    vision_stats = get_study_stats(VISION_DB)
    state_stats = get_study_stats(STATE_DB)
    
    print_study_status("VISION (6 workers)", vision_stats)
    print_study_status("STATE-BASED (2 workers)", state_stats)
    
    # Summary comparison
    print("\n" + "-" * 50)
    print("  COMPARISON")
    print("-" * 50)
    
    if vision_stats.get("exists") and state_stats.get("exists"):
        v_ssr = vision_stats.get("best_ssr", 0)
        s_ssr = state_stats.get("best_ssr", 0)
        v_stage = vision_stats.get("best_stage", 0)
        s_stage = state_stats.get("best_stage", 0)
        
        print(f"  {'Metric':<15} {'Vision':>12} {'State':>12}")
        print(f"  {'-'*15} {'-'*12} {'-'*12}")
        print(f"  {'Best SSR':<15} {v_ssr:>11.1%} {s_ssr:>11.1%}")
        print(f"  {'Max Stage':<15} {'S' + str(v_stage):>12} {'S' + str(s_stage):>12}")
        print(f"  {'Trials':<15} {vision_stats.get('complete', 0):>12} {state_stats.get('complete', 0):>12}")
        
        if s_stage > 0 and v_stage < s_stage:
            print("\n  ⚠️  State-based is ahead - possible vision bottleneck")
        elif v_stage > 0 and s_stage < v_stage:
            print("\n  ✓  Vision is learning well!")
        elif v_stage == 0 and s_stage == 0:
            print("\n  ⏳ Both starting...")


def main():
    parser = argparse.ArgumentParser(description="Monitor TEKO Optuna studies")
    parser.add_argument("--watch", action="store_true", help="Continuous monitoring")
    parser.add_argument("--interval", type=int, default=60, help="Watch interval (seconds)")
    parser.add_argument("--pareto", action="store_true", help="Show full Pareto fronts")
    args = parser.parse_args()
    
    if args.watch:
        try:
            while True:
                os.system("clear")
                print_combined_status()
                print(f"\n  Refreshing every {args.interval}s... (Ctrl+C to stop)")
                time.sleep(args.interval)
        except KeyboardInterrupt:
            print("\n\nStopped.")
    else:
        print_combined_status()
        print()


if __name__ == "__main__":
    main()