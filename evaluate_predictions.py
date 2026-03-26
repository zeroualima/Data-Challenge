"""
evaluate_predictions.py
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def diagnose_coverage(df, predictions, min_dist=3):
    eval_pairs = set(zip(df['airport'].astype(str), df['airport_alert_id'].astype(str)))
    pred_pairs = set(zip(predictions['airport'].astype(str), predictions['airport_alert_id'].astype(str)))
    missing    = eval_pairs - pred_pairs

    print(f"\n{'─'*55}")
    print(f"  Coverage diagnosis")
    print(f"{'─'*55}")
    print(f"  Alerts in eval    : {len(eval_pairs)}")
    print(f"  Alerts with preds : {len(pred_pairs)}")
    print(f"  Missing           : {len(missing)}")

    if not missing:
        print("  ✓ Full coverage.")
        return [], [], []

    outer_only, single_strike, inner_missing = [], [], []
    for airport, alert_id in missing:
        mask     = (df['airport'].astype(str) == airport) & (df['airport_alert_id'].astype(str) == alert_id)
        n_inner  = (df[mask]['dist'] <= 20).sum()
        if   n_inner == 0: outer_only.append((airport, alert_id))
        elif n_inner == 1: single_strike.append((airport, alert_id))
        else:              inner_missing.append((airport, alert_id))

    print(f"\n  Of the {len(missing)} missing alerts:")
    print(f"    {len(outer_only):3d}  outer-only   (dist>20 km — unpredictable, expected)")
    print(f"    {len(single_strike):3d}  single inner strike (n=1 — features impossible)")
    print(f"    {len(inner_missing):3d}  ≥2 inner strikes — SHOULD HAVE PREDICTIONS ⚠️")

    if inner_missing:
        print(f"\n  Alerts with ≥2 inner strikes but no prediction:")
        for ap, aid in inner_missing[:10]:
            n = ((df['airport'].astype(str)==ap) & (df['airport_alert_id'].astype(str)==aid) & (df['dist']<=20)).sum()
            print(f"    {ap:20s}  id={aid}  inner_n={n}")
        if len(inner_missing) > 10:
            print(f"    ... and {len(inner_missing)-10} more")

    # Dangerous strikes in uncovered alerts — these are the unavoidable R floor
    danger_uncovered = 0
    tot = (df['dist'] < min_dist).sum()
    for airport, alert_id in missing:
        mask = (df['airport'].astype(str)==airport) & (df['airport_alert_id'].astype(str)==alert_id)
        danger_uncovered += (df[mask]['dist'] < min_dist).sum()

    print(f"\n  Dangerous strikes in uncovered alerts : {danger_uncovered}/{tot} = {danger_uncovered/tot:.1%}")
    print(f"  → R floor from uncovered alerts alone  : {danger_uncovered/tot:.4f}")
    print(f"{'─'*55}")
    return outer_only, single_strike, inner_missing


def evaluate(data_path, preds_path, output_fig=None):
    MAX_GAP_MINUTES = 30
    min_dist        = 3
    ACCEPTABLE_RISK = 0.02

    # ── Load full labelled dataset ────────────────────────────────────────────
    df = pd.read_csv(data_path)
    df.columns = df.columns.str.strip()
    df = df[df['airport_alert_id'].notna()].copy()

    print(f"Full dataset  : {len(df):,} rows")
    print(f"Airports      : {df['airport'].nunique()}")
    print(f"Alerts        : {df['airport_alert_id'].nunique()}")
    tot_lightnings = len(df[df['dist'] < min_dist])
    print(f"Lightning ≤ {min_dist} km : {tot_lightnings}")

    # ── Load predictions ──────────────────────────────────────────────────────
    predictions = pd.read_csv(preds_path)
    predictions = predictions[predictions['airport_alert_id'] != -1.0]
    predictions['predicted_date_end_alert'] = pd.to_datetime(
        predictions['predicted_date_end_alert'], utc=True, format='mixed')
    predictions['prediction_date'] = pd.to_datetime(
        predictions['prediction_date'], utc=True, format='mixed')
    aid_dtype = df['airport_alert_id'].dtype
    predictions['airport_alert_id'] = predictions['airport_alert_id'].astype(aid_dtype)

    print(f"\nPredictions   : {len(predictions):,} rows")
    print(f"Alerts covered: {predictions['airport_alert_id'].nunique()}")
    print(f"Confidence levels: {sorted(predictions['confidence'].unique())}")

    # Keep only the most recent prediction per (alert, confidence level)
    predictions = (predictions
                   .sort_values('prediction_date')
                   .groupby(['airport', 'airport_alert_id', 'confidence'], as_index=False)
                   .last())
    print(f"After keeping latest per (alert, ε): {len(predictions):,} rows")

    # ── Coverage diagnosis ────────────────────────────────────────────────────
    diagnose_coverage(df, predictions, min_dist)

    # Dangerous strikes in covered alerts (for fair comparison)
    pred_pairs = set(zip(predictions['airport'].astype(str),
                         predictions['airport_alert_id'].astype(str)))
    covered_mask = df.apply(
        lambda r: (str(r['airport']), str(r['airport_alert_id'])) in pred_pairs, axis=1)
    tot_covered  = len(df[covered_mask & (df['dist'] < min_dist)])
    print(f"\n  Dangerous strikes in covered alerts : {tot_covered}/{tot_lightnings}")
    uncovered_floor = (tot_lightnings - tot_covered) / tot_lightnings
    print(f"  R floor from uncovered alerts       : {uncovered_floor:.4f}")

    # ── Theta sweep ───────────────────────────────────────────────────────────
    alerts_grp = df.groupby(['airport', 'airport_alert_id'])
    thetas     = [i / 20 for i in range(20)]
    results    = {}

    for theta in thetas:
        pred_t = predictions[predictions['confidence'] >= theta]
        if pred_t.empty:
            results[theta] = (0, tot_lightnings)
            continue

        pred_min = pred_t.groupby(['airport', 'airport_alert_id'])['predicted_date_end_alert'].min()
        gain, missed = 0, 0
        for (airport, alert_id), end_pred in pred_min.items():
            try:   lightnings = alerts_grp.get_group((airport, alert_id))
            except KeyError: continue
            dates        = pd.to_datetime(lightnings['date'], utc=True)
            end_baseline = dates.max() + pd.Timedelta(minutes=MAX_GAP_MINUTES)
            gain        += (end_baseline - end_pred).total_seconds()
            dangerous    = dates[lightnings['dist'] < min_dist]
            missed      += int((dangerous > end_pred).sum())
        results[theta] = (gain, missed)

    # ── Pareto curves ─────────────────────────────────────────────────────────
    gains      = [results[t][0] / 3600              for t in thetas]
    risk_jury  = [results[t][1] / tot_lightnings    for t in thetas]
    risk_fair  = [results[t][1] / max(tot_covered,1) for t in thetas]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, risk_vals, label, color in [
        (axes[0], risk_jury, f'Jury metric (denom={tot_lightnings} all alerts)', 'steelblue'),
        (axes[1], risk_fair, f'Fair metric (denom={tot_covered} covered alerts)', 'darkorange'),
    ]:
        ax.plot(risk_vals, gains, marker='*', markersize=9, color=color)
        for t, g, m in zip(thetas, gains, risk_vals):
            ax.annotate(f'{t:.2f}', (m, g), fontsize=7,
                        xytext=(4, 2), textcoords='offset points')
        ax.axvline(ACCEPTABLE_RISK, color='red', ls='--', lw=1.2,
                   label=f'R_accept={ACCEPTABLE_RISK}')
        if label.startswith('Jury') and uncovered_floor > 0:
            ax.axvline(uncovered_floor, color='grey', ls=':', lw=1.2,
                       label=f'Uncovered floor={uncovered_floor:.3f}')
        ax.set_xlabel('Risk R')
        ax.set_ylabel('Gain (hours)')
        ax.set_title(label)
        ax.legend(fontsize=8)

    plt.suptitle('Gain vs Risk — θ sweep', fontsize=13, fontweight='bold')
    plt.tight_layout()
    if output_fig:
        plt.savefig(output_fig, dpi=130)
        print(f"\nCurve saved → {output_fig}")
    plt.show()

    # ── Best theta ────────────────────────────────────────────────────────────
    for risk_fn, denom_label, denom in [
        (lambda m: m/tot_lightnings,        "JURY   ", tot_lightnings),
        (lambda m: m/max(tot_covered,1),    "FAIR   ", tot_covered),
    ]:
        candidates = [(results[t][0], t, results[t][1])
                      for t in thetas if risk_fn(results[t][1]) < ACCEPTABLE_RISK]
        print(f"\n{'='*55}")
        if not candidates:
            best = min(thetas, key=lambda t: risk_fn(results[t][1]))
            g, m = results[best]
            print(f"  {denom_label}: ⚠️  No θ satisfies R < {ACCEPTABLE_RISK}")
            print(f"  Best: θ={best:.2f}  gain={g/3600:.1f}h  R={risk_fn(m):.4f}")
        else:
            gain_h, theta, miss = max(candidates)
            print(f"  {denom_label}: θ={theta}  gain={int(gain_h/3600)}h  R={risk_fn(miss):.4f}")
        print(f"{'='*55}")

    # ── Per-airport at best θ (jury metric) ───────────────────────────────────
    jury_candidates = [(results[t][0], t) for t in thetas
                       if results[t][1]/tot_lightnings < ACCEPTABLE_RISK]
    if jury_candidates:
        _, best_theta = max(jury_candidates)
    else:
        best_theta = min(thetas, key=lambda t: results[t][1])

    print(f"\nPer-airport at θ={best_theta}:")
    pred_min = (predictions[predictions['confidence'] >= best_theta]
                .groupby(['airport', 'airport_alert_id'])['predicted_date_end_alert'].min())

    per_ap = {}
    for (airport, alert_id), end_pred in pred_min.items():
        try:   lts = alerts_grp.get_group((airport, alert_id))
        except KeyError: continue
        dates = pd.to_datetime(lts['date'], utc=True)
        g_sec = (dates.max() + pd.Timedelta(minutes=MAX_GAP_MINUTES) - end_pred).total_seconds()
        m     = int((dates[lts['dist'] < min_dist] > end_pred).sum())
        if airport not in per_ap:
            per_ap[airport] = {'gain_h': 0., 'missed': 0, 'alerts': 0}
        per_ap[airport]['gain_h']  += g_sec / 3600
        per_ap[airport]['missed']  += m
        per_ap[airport]['alerts']  += 1

    print(f"\n  {'Airport':<22} {'Alerts':>7} {'Gain (h)':>9} {'Missed':>8}")
    print(f"  {'-'*50}")
    for ap, v in sorted(per_ap.items()):
        print(f"  {ap:<22} {v['alerts']:>7} {v['gain_h']:>9.1f} {v['missed']:>8}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data',   default='./dataset_test/segment_alerts_all_airports_eval.csv')
    parser.add_argument('--preds',  default='./dataset_test/predictions.csv')
    parser.add_argument('--output', default='./figures/evaluation_curve.png')
    args = parser.parse_args()
    evaluate(args.data, args.preds, args.output)