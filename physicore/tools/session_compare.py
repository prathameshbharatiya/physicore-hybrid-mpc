#!/usr/bin/env python3
"""
PhysiCore Session Comparison Tool
===================================
Proves the registry flywheel works in real numbers.

Replays your actual bridge data twice:
  Session 1: cold start — no prior knowledge
  Session 2: loads Session 1 from registry — starts warmer

Shows side-by-side convergence in terminal.

Usage:
    python tools/session_compare.py --file bridge_file_after_the_pysicore_is_connected.txt
    python tools/session_compare.py --real --platform balancing_bot
    python tools/session_compare.py --help

Author: Prathamesh Shirbhate — physicore.ai
"""

import sys, os, argparse, json, time, math, tempfile, shutil
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np

GRN = "\033[92m"; RED = "\033[91m"; YLW = "\033[93m"
CYN = "\033[96m"; BLD = "\033[1m";  RST = "\033[0m"


def parse_bridge_file(path):
    entries = []
    with open(path, encoding='utf-8', errors='replace') as f:
        for line in f:
            line = line.strip().replace('\r','')
            if line.startswith('RAW:'):
                try:
                    entries.append(json.loads(line[4:].strip()))
                except Exception:
                    pass
    return entries


def to_state(entry):
    pitch = math.radians(entry.get('pitch', 0))
    gyro  = math.radians(entry.get('gyro_x', 0))
    return np.array([pitch, gyro, 0.0, 0.0])


def run_session(engine, entries, label, max_steps=None):
    from physicore.sentinel.core import SentinelOS
    sentinel = SentinelOS(engine, platform='balancing_bot', verbose=False)
    x_ref    = np.zeros(4)
    prev_x   = None
    residuals, uncs, params_hist = [], [], []
    steps    = 0
    limit    = max_steps or len(entries)

    t0 = time.time()
    for e in entries[:limit]:
        x      = to_state(e)
        action = sentinel.step(x, x_ref)
        if prev_x is not None:
            sentinel.observe(prev_x, action, x)
        prev_x = x.copy()

        d = engine.diagnostics_full
        residuals.append(d['residual_norm'])
        uncs.append(d['uncertainty'])
        params_hist.append(d['params'].copy())
        steps += 1

    duration = time.time() - t0

    # Convergence: residual drop from first 10 to last 10 steps
    init_res  = float(np.mean(residuals[:10]))  if len(residuals) >= 10 else residuals[0]
    final_res = float(np.mean(residuals[-10:])) if len(residuals) >= 10 else residuals[-1]
    conv_pct  = max(0.0, (init_res - final_res) / max(init_res, 1e-9) * 100)

    return {
        'label':        label,
        'steps':        steps,
        'duration_s':   round(duration, 2),
        'init_residual':  round(init_res,  4),
        'final_residual': round(final_res, 4),
        'convergence_pct': round(conv_pct, 1),
        'final_params':   params_hist[-1] if params_hist else {},
        'residuals':      residuals,
        'uncs':           uncs,
        'sentinel_hash':  sentinel.chain_hash,
        'sentinel_entries': sentinel.status['ledger_entries'],
    }


def terminal_bar(val, max_val, width=40, colour=GRN):
    filled = int(min(val / max(max_val, 1e-9), 1.0) * width)
    return f"{colour}{'█' * filled}{'░' * (width - filled)}{RST}"


def print_comparison(s1, s2):
    print(f"\n{BLD}{'═'*70}{RST}")
    print(f"{BLD}  PHYSICORE SESSION COMPARISON — REGISTRY FLYWHEEL PROOF{RST}")
    print(f"{BLD}{'═'*70}{RST}\n")

    rows = [
        ("Steps",           str(s1['steps']),                 str(s2['steps'])),
        ("Duration",        f"{s1['duration_s']}s",           f"{s2['duration_s']}s"),
        ("Initial residual",f"{s1['init_residual']:.4f}",     f"{s2['init_residual']:.4f}"),
        ("Final residual",  f"{s1['final_residual']:.4f}",    f"{s2['final_residual']:.4f}"),
        ("Convergence",     f"{s1['convergence_pct']:.1f}%",  f"{s2['convergence_pct']:.1f}%"),
        ("Ledger hash",     s1['sentinel_hash'][:12],         s2['sentinel_hash'][:12]),
    ]

    print(f"  {'Metric':<25} {'Session 1 (cold)':>20}   {'Session 2 (warm)':>20}")
    print(f"  {'─'*25} {'─'*20}   {'─'*20}")
    for name, v1, v2 in rows:
        better = ""
        try:
            f1, f2 = float(v1.rstrip('%s')), float(v2.rstrip('%s'))
            if 'residual' in name.lower() and f2 < f1:
                better = f" {GRN}↓ better{RST}"
            elif 'convergence' in name.lower() and f2 > f1:
                better = f" {GRN}↑ better{RST}"
        except Exception:
            pass
        print(f"  {name:<25} {v1:>20}   {v2:>20}{better}")

    print(f"\n  {BLD}Convergence improvement:{RST}")
    delta = s2['convergence_pct'] - s1['convergence_pct']
    if delta > 0:
        print(f"    Session 2 converged {GRN}{delta:.1f}pp faster{RST} — "
              f"registry prior is working.")
    elif delta == 0:
        print(f"    {YLW}Convergence identical — more sessions needed to see improvement.{RST}")
    else:
        print(f"    {YLW}Session 2 converged slightly slower — normal for small datasets.{RST}")

    print(f"\n  {BLD}Final params after learning:{RST}")
    print(f"  {'Param':<15} {'Session 1':>15}   {'Session 2':>15}")
    print(f"  {'─'*15} {'─'*15}   {'─'*15}")
    all_keys = set(list(s1['final_params'].keys()) + list(s2['final_params'].keys()))
    for k in sorted(all_keys):
        v1 = s1['final_params'].get(k, 0)
        v2 = s2['final_params'].get(k, 0)
        print(f"  {k:<15} {v1:>15.4f}   {v2:>15.4f}")

    print(f"\n  {BLD}Residual over time:{RST}")
    # Simple terminal sparkline
    def sparkline(vals, width=50):
        if not vals: return ""
        step = max(1, len(vals) // width)
        sampled = [vals[i] for i in range(0, len(vals), step)][:width]
        vmax = max(sampled + [1e-9])
        bars = " ▁▂▃▄▅▆▇█"
        return ''.join(bars[min(int(v / vmax * 8), 8)] for v in sampled)

    print(f"  S1: {RED}{sparkline(s1['residuals'])}{RST}")
    print(f"  S2: {GRN}{sparkline(s2['residuals'])}{RST}")
    print(f"      ↑ start                                          end ↑")

    print(f"\n{BLD}{'═'*70}{RST}\n")


def show_real_registry(platform):
    from physicore.core.registry import get_registry
    reg = get_registry()

    print(f"\n{BLD}{'═'*60}{RST}")
    print(f"{BLD}  REAL REGISTRY — {platform.upper()}{RST}")
    print(f"{BLD}{'═'*60}{RST}")
    print(f"  Location: {reg._platform_dir(platform)}\n")

    d  = reg._platform_dir(platform)
    sp = d / "sessions.jsonl"
    pp = d / "params.json"
    fp = d / "platform_prior.json"

    if sp.exists():
        sessions = [json.loads(l) for l in open(sp) if l.strip()]
        print(f"{BLD}  Sessions: {len(sessions)}{RST}")
        for i, s in enumerate(sessions):
            ts  = time.strftime('%Y-%m-%d %H:%M', time.localtime(s['timestamp']))
            pct = s['convergence_pct']
            bar = terminal_bar(pct, 100, width=30,
                               colour=GRN if pct > 50 else YLW)
            print(f"\n  Session {i+1}  {ts}")
            print(f"    Steps:       {s['steps']}")
            print(f"    Duration:    {s['duration_s']:.1f}s")
            print(f"    Convergence: {bar} {pct:.1f}%")
            print(f"    Innovation:  {s['innovation_ema']:.4f}")
            for k,v in s['final_params'].items():
                print(f"    {k}:{'':>10} {v:.4f}")
    else:
        print(f"  {YLW}No sessions yet for '{platform}'.{RST}")
        print(f"  Run the bridge: python physicore/bridge/physicore_bridge.py "
              f"--platform balancing_bot_arduino --connection COM8")

    if pp.exists():
        saved = json.load(open(pp))
        print(f"\n{BLD}  Converged params (session-averaged):{RST}")
        for k,v in saved.get('params', {}).items():
            print(f"    {k}: {GRN}{v:.4f}{RST}")
        print(f"  Sessions contributing: {saved.get('sessions_count', 1)}")

    if fp.exists():
        prior = json.load(open(fp))
        print(f"\n{BLD}  Platform prior:{RST}")
        for k,v in prior.get('params', {}).items():
            print(f"    {k}: {GRN}{v:.4f}{RST}")
        print(f"  Weight: {prior.get('weight',0):.2f}  "
              f"Sessions: {prior.get('sessions',0)}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="PhysiCore Session Comparison — proves registry flywheel"
    )
    parser.add_argument('--file',     default=None,
                        help='Bridge log file to replay (RAW: JSON lines)')
    parser.add_argument('--platform', default='balancing_bot',
                        help='Platform name (default: balancing_bot)')
    parser.add_argument('--steps',    type=int, default=400,
                        help='Max steps per session (default: 400)')
    parser.add_argument('--real',     action='store_true',
                        help='Show real registry data (no simulation)')
    args = parser.parse_args()

    if args.real:
        show_real_registry(args.platform)
        return

    if not args.file:
        print(f"{RED}Error: --file required (or use --real to show registry){RST}")
        parser.print_help()
        sys.exit(1)

    if not os.path.exists(args.file):
        print(f"{RED}Error: file not found: {args.file}{RST}")
        sys.exit(1)

    print(f"\n{BLD}PhysiCore Session Comparison{RST}")
    print(f"  File:     {args.file}")
    print(f"  Platform: {args.platform}")
    print(f"  Steps:    {args.steps}\n")

    from physicore import PhysiCore
    from physicore.core.registry import ModelRegistry

    entries = parse_bridge_file(args.file)
    print(f"  Loaded {len(entries)} data points from bridge log")

    # Use a temporary registry so we don't pollute the real one
    tmp_dir = tempfile.mkdtemp(prefix='physicore_compare_')
    try:
        reg = ModelRegistry(root=tmp_dir)

        # ── Session 1: cold start ──────────────────────────────────────────
        print(f"\n{BLD}▸ Session 1 — cold start{RST}")
        e1 = PhysiCore.for_platform(args.platform,
                                     {'mass':1.0,'friction':0.15,'inertia':0.01})
        s1 = run_session(e1, entries, "Session 1 (cold)", args.steps)
        # Save to temp registry
        reg.save(e1, platform=args.platform,
                 session_meta={'session': 1, 'source': args.file})
        print(f"  Done: convergence={s1['convergence_pct']}% | "
              f"residual {s1['init_residual']:.4f}→{s1['final_residual']:.4f}")

        # ── Session 2: warm start from registry ────────────────────────────
        print(f"\n{BLD}▸ Session 2 — warm start from registry{RST}")
        e2 = PhysiCore.for_platform(args.platform,
                                     {'mass':1.0,'friction':0.15,'inertia':0.01})
        loaded = reg.load(e2, args.platform)
        print(f"  Registry loaded: {loaded}")
        s2 = run_session(e2, entries, "Session 2 (warm)", args.steps)
        print(f"  Done: convergence={s2['convergence_pct']}% | "
              f"residual {s2['init_residual']:.4f}→{s2['final_residual']:.4f}")

        print_comparison(s1, s2)

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == '__main__':
    main()
