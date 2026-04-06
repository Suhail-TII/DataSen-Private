"""
Prints Table 13 from PromptLocate paper (IEEE S&P 2026).
  Precision and Recall of PromptLocate and baselines.
  Data comes from the ALREADY COMPUTED results_table1_pl.json —
  no re-evaluation needed.

Usage:
    python table13_pl.py --results_json results_table1_pl.json
"""

import argparse
import json
import numpy as np

# ── Attacks ───────────────────────────────────────────────────────────────────

ATTACKS = [
    ('naive',       'Naive Attack'),
    ('escape',      'Escape Character'),
    ('ignore',      'Context Ignoring'),
    ('fake_comp',   'Fake Completion'),
    ('combine',     'Combined Attack'),
    ('universal',   'Universal'),
    ('neural_exec', 'NeuralExec'),
]

# Paper reference values for baselines (Table 13)
PAPER_BASELINES_PREC_REC = {
    'SFA-H': [(0.30,0.22),(0.32,0.23),(0.41,0.23),(0.33,0.16),(0.65,0.35),(0.54,0.18),(0.99,0.63)],
    'SFA-T': [(0.50,0.77),(0.55,0.82),(0.44,0.63),(0.45,0.65),(0.39,0.49),(0.49,0.64),(0.66,0.80)],
    'FRA-H': [(0.33,0.23),(0.35,0.25),(0.45,0.25),(0.41,0.22),(0.70,0.36),(0.55,0.21),(0.99,0.62)],
    'FRA-T': [(0.28,0.28),(0.30,0.30),(0.35,0.28),(0.32,0.27),(0.55,0.40),(0.35,0.20),(0.94,0.67)],
    'SVA-H': [(0.33,0.24),(0.36,0.26),(0.42,0.24),(0.37,0.16),(0.68,0.33),(0.50,0.17),(0.97,0.63)],
    'SVA-T': [(0.23,0.21),(0.26,0.23),(0.32,0.23),(0.26,0.22),(0.56,0.45),(0.35,0.22),(0.91,0.72)],
}

# Paper's PromptLocate values for reference
PAPER_PL_PREC = [0.98, 0.99, 0.99, 1.00, 1.00, 0.98, 0.99]
PAPER_PL_REC  = [0.99, 0.99, 0.95, 0.97, 0.97, 0.97, 0.96]

# ── Helper ────────────────────────────────────────────────────────────────────

def avg(lst):
    clean = [float(x) for x in lst if x is not None]
    return sum(clean) / len(clean) if clean else float('nan')

# ── Print Table 13 ────────────────────────────────────────────────────────────

def print_table13(agg: dict):
    attack_names = [n for _, n in ATTACKS]
    attack_keys  = [k for k, _ in ATTACKS]

    M  = 16
    V  = 5
    SP = "  "

    def divider():
        return "─" * (M + len(attack_names) * (V * 2 + len(SP) + 4))

    def atk_header_row():
        row = " " * M
        for name in attack_names:
            w = V * 2 + len(SP) + 2
            row += f"  {name:^{w}}"
        return row

    def subhdr_row(left, right):
        row = f"{'Method':<{M}}"
        for _ in attack_names:
            row += f"  {left:^{V}}{SP}{right:^{V}}  "
        return row

    DIV = divider()

    print()
    print("╔" + "═" * (len(DIV) - 2) + "╗")
    title = "TABLE 13  —  Precision & Recall  —  OpenPromptInjection Benchmark"
    print(f"║{title:^{len(DIV)-2}}║")
    print("╚" + "═" * (len(DIV) - 2) + "╝")
    print()
    print("  Each result is averaged over the 7×7 target-injected task combinations.")
    print()

    print(atk_header_row())
    print(subhdr_row("Prec.", "Rec."))
    print(DIV)

    # Baseline rows (paper values)
    for method, vals in PAPER_BASELINES_PREC_REC.items():
        row = f"  {method:<{M-2}}"
        for prec, rec in vals:
            row += f"  {prec:{V}.2f}{SP}{rec:{V}.2f}  "
        print(row)

    print(DIV)

    # PromptLocate row — your results
    row = f"  {'PromptLocate':<{M-2}}"
    for key in attack_keys:
        prec = avg(agg[key]['prec'])
        rec  = avg(agg[key]['rec'])
        row += f"  {prec:{V}.2f}{SP}{rec:{V}.2f}  "
    print(row + "  ← your results")
    print(DIV)

    # Summary comparison
    print()
    print("  SUMMARY — Your Precision/Recall vs. Paper (±0.03 tolerance)")
    print()
    print(f"  {'Attack':<22}  {'Prec: yours/paper':^20}  {'Rec: yours/paper':^20}  {'OK?':^5}")
    print("  " + "─" * 75)

    all_ok = True
    for (key, name), p_prec, p_rec in zip(ATTACKS, PAPER_PL_PREC, PAPER_PL_REC):
        our_prec = avg(agg[key]['prec'])
        our_rec  = avg(agg[key]['rec'])
        ok = abs(our_prec - p_prec) < 0.03 and abs(our_rec - p_rec) < 0.03
        if not ok:
            all_ok = False
        icon = "✓" if ok else "✗"
        print(f"  {name:<22}  "
              f"  {our_prec:.2f} / {p_prec:.2f}           "
              f"  {our_rec:.2f} / {p_rec:.2f}           "
              f"  {icon}")

    print()
    print(f"  {'✓  All match paper.' if all_ok else '✗  Some deviate.'}")
    print()


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Print Table 13 from saved Table 1 results JSON"
    )
    parser.add_argument('--results_json', default='results_table1_pl.json',
                        help='Path to results_table1_pl.json from table1_pl.py run')
    args = parser.parse_args()

    with open(args.results_json) as f:
        agg = json.load(f)

    # Verify prec/rec keys exist
    first_key = list(agg.keys())[0]
    if 'prec' not in agg[first_key] or not agg[first_key]['prec']:
        print("WARNING: No precision/recall data found in JSON.")
        print("  Make sure table1_pl.py collected prec/rec during evaluation.")
        print("  Check that agg[key]['prec'] and agg[key]['rec'] are populated.")
        return

    print_table13(agg)


if __name__ == '__main__':
    main()