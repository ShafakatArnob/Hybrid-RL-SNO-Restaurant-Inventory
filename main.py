"""
main.py
==============
Master runner for the complete research pipeline.

Usage
-----
    python main.py              # full pipeline (all 5 restaurants, ~15-20 min)
    python main.py --quick      # fast demo (~3 min, Kachchi Bhai only for S2)
    python main.py --s1only     # Scenario 1 only
    python main.py --s2only     # Scenario 2 only
    python main.py --tests      # run 29 unit tests
    python main.py --figs       # regenerate figures from saved results
"""

import sys, time, json, warnings
from pathlib import Path


def _apply_quick_mode():
    import config
    config.TRAIN_CONFIG['ql_episodes']   = 200
    config.TRAIN_CONFIG['dqn_episodes']  = 150
    config.TRAIN_CONFIG['ppo_episodes']  = 100
    config.TRAIN_CONFIG['eval_episodes'] = 30
    config._QUICK_S2_RESTAURANTS = [config.KACCHI_BHAI]
    print("  [QUICK MODE] Reduced episodes | Scenario 2: Kachchi Bhai only.")


def _print_banner():
    print("""
╔══════════════════════════════════════════════════════════════════════════╗
║  DASE4174 FINAL YEAR PROJECT — HYBRID RL + SNO FRAMEWORK  v5             ║
║  Reinforcement Learning with Stochastic Network Optimisation             ║
║  for Restaurant Inventory Management                                     ║
║                                                                          ║
║  Author    : Mursalin Mahdi Islam  (3035962104)                          ║
║  Supervisor: Dr. W.J. Huang  |  HKU Department of DASE                   ║
╚══════════════════════════════════════════════════════════════════════════╝""")


def main():
    args      = set(sys.argv[1:])
    quick     = '--quick'  in args
    s1_only   = '--s1only' in args
    s2_only   = '--s2only' in args
    figs_only = '--figs'   in args
    run_tests = '--tests'  in args

    _print_banner()

    if run_tests:
        import subprocess
        r = subprocess.run([sys.executable, '-m', 'pytest', 'tests/', '-v', '--tb=short'])
        sys.exit(r.returncode)

    if quick:
        _apply_quick_mode()

    warnings.filterwarnings('ignore', category=UserWarning, module='data_loader')
    warnings.filterwarnings('ignore', category=FutureWarning)

    from scenario1_generic     import run_scenario1
    from scenario2_kacchi_bhai import run_scenario2
    from visualisations        import generate_all_figures
    from results_summary       import generate_full_summary
    import config

    tc       = config.TRAIN_CONFIG
    s2_rests = getattr(config, '_QUICK_S2_RESTAURANTS', None)

    RESULTS_DIR = Path('./results')
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    if figs_only:
        print("\n  [FIGS ONLY] Regenerating from saved results...")
        s1_f = RESULTS_DIR / 'scenario1' / 'scenario1_detailed_metrics.json'
        s1_c = RESULTS_DIR / 'scenario1' / 'scenario1_reward_curves.json'
        if not s1_f.exists():
            print("  ERROR: Run full pipeline first."); sys.exit(1)
        with open(s1_f) as f: s1_m = json.load(f)
        with open(s1_c) as f: s1_curves = json.load(f)
        s1_r = {k: {'mean_metrics': v} for k, v in s1_m.items()}
        s2_a = {}
        for cfg in config.ALL_RESTAURANTS:
            rid = cfg['restaurant_id']
            mf  = RESULTS_DIR / 'scenario2' / rid / f'{rid}_detailed_metrics.json'
            cf  = RESULTS_DIR / 'scenario2' / rid / f'{rid}_reward_curves.json'
            if mf.exists():
                with open(mf) as f: m = json.load(f)
                crv = json.load(open(cf)) if cf.exists() else {}
                s2_a[rid] = {'config': cfg,
                             'results': {k: {'mean_metrics': v} for k,v in m.items()},
                             'reward_curves': crv, 'demand_mode': 'saved'}
        generate_all_figures(s1_r, s1_curves, s2_a)
        return

    s1_data = s2_data = None

    if not s2_only:
        print("\n" + "─"*70)
        print("  STEP 0a: EDA — SCENARIO 1 PUBLIC DATASETS")
        print("─"*70)
        from eda_scenario1 import run_s1_eda
        run_s1_eda()

    if not s1_only:
        print("\n" + "─"*70)
        print("  STEP 0b: EDA — SCENARIO 2 DATASETS")
        print("─"*70)
        from eda_scenario2 import run_s2_eda
        run_s2_eda()

    if not s2_only:
        print("\n" + "─"*70)
        print("  STEP 1-2: SCENARIO 1 — GENERIC RESTAURANT FRAMEWORK")
        print("─"*70)
        ve = max(1, tc['ql_episodes'] // 5)
        s1_data = run_scenario1(train_cfg=tc, verbose_every=ve)

    if not s1_only:
        n_r = len(s2_rests) if s2_rests else 5
        print("\n" + "─"*70)
        print(f"  STEP 3: SCENARIO 2 — {n_r} BANGLADESHI RESTAURANT(S)")
        print("─"*70)
        ve = max(1, tc['ql_episodes'] // 5)
        s2_data = run_scenario2(train_cfg=tc, restaurants=s2_rests,
                                verbose_every=ve)

    if s1_data and s2_data:
        print("\n" + "─"*70)
        print("  STEP 4: GENERATING ALL FIGURES")
        print("─"*70)
        generate_all_figures(s1_data['results'], s1_data['reward_curves'], s2_data)

    if s1_data and s2_data:
        print("\n" + "─"*70)
        print("  STEP 5: RESULTS SUMMARY & KEY FINDINGS")
        print("─"*70)
        generate_full_summary(s1_data['results'], s2_data, out_dir=RESULTS_DIR)

    m, s = divmod(int(time.time() - t_start), 60)
    print(f"\n{'='*70}")
    print(f"  PIPELINE COMPLETE  |  Total runtime: {m}m {s}s")
    print(f"  All outputs saved in ./results/ and ./data/")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
