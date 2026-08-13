from __future__ import annotations

import gc
import json
import math
import sys
from dataclasses import asdict
from datetime import date
from pathlib import Path
from statistics import median

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from atlas_quant.backtest.corporate_actions import compute_economic_return
from atlas_quant.cli.filing_momentum import (
    _build_run_inputs,
    _build_trading_calendar,
    _feature_cache_identity,
    _load_manifest,
    _periods_from_args,
    _run_validation,
    load_normalized_bundle,
)
from atlas_quant.strategies.filing_momentum_ml.config import FilingMomentumMLConfig
from atlas_quant.strategies.filing_momentum_ml.feature_domain import FEATURE_NAMES
from atlas_quant.strategies.filing_momentum_ml.forward_return import build_forward_return_outcome
from atlas_quant.strategies.filing_momentum_ml.labeling import assign_quarterly_labels
from atlas_quant.strategies.filing_momentum_ml.performance_analysis import (
    analyze_backtest_result,
    performance_analysis_to_dict,
)
from atlas_quant.strategies.filing_momentum_ml.production.feature_label_build import build_production_features
from atlas_quant.strategies.filing_momentum_ml.production.orchestration import run_filing_momentum_production_backtest
from atlas_quant.strategies.filing_momentum_ml.sector_encoding import SectorEncoder


START = "2015-03-31"
END = "2026-06-30"
BENCHMARK = "SPY"
PRICE_FEATURES = ("price_mom_3m", "price_mom_6m", "price_mom_12m", "vol_20d", "vol_63d", "vol_ratio")


class Args:
    def __init__(self, raw_root: Path, manifest: Path, checkpoint_root: Path | None):
        self.raw_root = raw_root
        self.manifest = manifest
        self.start_quarter = START
        self.end_quarter = END
        self.benchmark = BENCHMARK
        self.earnings_lag_days = None
        self.dry_run = checkpoint_root is None
        self.checkpoint_root = checkpoint_root
        self.source_report_html = None


def finite_diff(a: float, b: float, tol: float = 1e-12) -> bool:
    if math.isnan(a) and math.isnan(b):
        return False
    if math.isnan(a) != math.isnan(b):
        return True
    return abs(a - b) > tol


def pct(values: list[float], q: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    idx = min(len(values) - 1, max(0, round((len(values) - 1) * q)))
    return values[idx]


def summarize_feature_diffs(old: dict, new: dict) -> dict:
    out = {}
    keys = sorted(set(old) & set(new))
    for feature in PRICE_FEATURES:
        abs_changes = []
        rel_changes = []
        changed = 0
        suspicious = []
        for key in keys:
            a = old[key].get(feature, float("nan"))
            b = new[key].get(feature, float("nan"))
            if finite_diff(a, b):
                changed += 1
                if math.isfinite(a) and math.isfinite(b):
                    d = abs(b - a)
                    abs_changes.append(d)
                    if abs(a) > 1e-12:
                        rel_changes.append(d / abs(a))
                    if d > 1.0 and feature.startswith("price_mom"):
                        suspicious.append({"key": key, "legacy": a, "clean": b, "abs_change": d})
        out[feature] = {
            "matched_rows": len(keys),
            "changed_rows": changed,
            "percentage_changed": changed / len(keys) if keys else None,
            "median_absolute_change": median(abs_changes) if abs_changes else 0.0,
            "p95_absolute_change": pct(abs_changes, 0.95),
            "max_absolute_change": max(abs_changes) if abs_changes else 0.0,
            "median_relative_change": median(rel_changes) if rel_changes else None,
            "largest_suspicious": sorted(suspicious, key=lambda r: r["abs_change"], reverse=True)[:10],
        }
    any_changed = sum(1 for key in keys if any(finite_diff(old[key].get(f, float("nan")), new[key].get(f, float("nan"))) for f in PRICE_FEATURES))
    out["_summary"] = {
        "matched_feature_observations": len(keys),
        "feature_observations_changed_any_price_feature": any_changed,
        "percentage_changed_any_price_feature": any_changed / len(keys) if keys else None,
    }
    return out


def summarize_label_diffs(old: dict, new: dict) -> dict:
    keys = sorted(set(old) & set(new))
    ret_changed = entry_changed = exit_changed = cash_changed = label_changed = 0
    outliers = []
    for key in keys:
        a = old[key]
        b = new[key]
        if a["label"] != b["label"]:
            label_changed += 1
        if (a["raw_return"] is None) != (b["raw_return"] is None) or (
            a["raw_return"] is not None and abs(a["raw_return"] - b["raw_return"]) > 1e-12
        ):
            ret_changed += 1
            if a["raw_return"] is not None and b["raw_return"] is not None:
                outliers.append({"key": key, "legacy": a["raw_return"], "clean": b["raw_return"], "delta": b["raw_return"] - a["raw_return"]})
        if a["entry_price"] != b["entry_price"]:
            entry_changed += 1
        if a["exit_price"] != b["exit_price"]:
            exit_changed += 1
        if a["dividend_cash"] != b["dividend_cash"] or a["dividend_count"] != b["dividend_count"] or a["split_count"] != b["split_count"]:
            cash_changed += 1
    return {
        "matched_label_rows": len(keys),
        "labels_changed": label_changed,
        "return_value_changed": ret_changed,
        "entry_endpoint_prices_changed": entry_changed,
        "exit_endpoint_prices_changed": exit_changed,
        "corporate_action_cash_flows_changed": cash_changed,
        "material_outliers": sorted(outliers, key=lambda r: abs(r["delta"]), reverse=True)[:20],
    }


def selected_stock_set(q) -> set[str]:
    if not q.strategy_result:
        return set()
    return {r.instrument_id.symbol for r in q.strategy_result.recommendations if r.kind.value == "primary"}


def summarize_selection(old_bt, new_bt) -> dict:
    old_by_q = {q.period.quarter_end.isoformat(): q for q in old_bt.quarter_results}
    new_by_q = {q.period.quarter_end.isoformat(): q for q in new_bt.quarter_results}
    rows = []
    changed_rankings = changed_selections = model_changed = 0
    overlaps = []
    for qid in sorted(set(old_by_q) & set(new_by_q)):
        a = old_by_q[qid]
        b = new_by_q[qid]
        old_sel = selected_stock_set(a)
        new_sel = selected_stock_set(b)
        denom = max(len(old_sel), len(new_sel), 1)
        overlap = len(old_sel & new_sel) / denom
        overlaps.append(overlap)
        if old_sel != new_sel:
            changed_selections += 1
        old_rank = [
            c.instrument_id.symbol
            for c in sorted(a.scoring_result.scored_candidates, key=lambda c: (-c.score, c.instrument_id.symbol))
        ] if a.scoring_result else []
        new_rank = [
            c.instrument_id.symbol
            for c in sorted(b.scoring_result.scored_candidates, key=lambda c: (-c.score, c.instrument_id.symbol))
        ] if b.scoring_result else []
        if old_rank != new_rank:
            changed_rankings += 1
        old_model = a.model_identity.identity() if a.model_identity else None
        new_model = b.model_identity.identity() if b.model_identity else None
        if old_model != new_model:
            model_changed += 1
        rows.append({
            "quarter": qid,
            "legacy_return": a.period_return,
            "clean_return": b.period_return,
            "selection_overlap": overlap,
            "legacy_selected": sorted(old_sel),
            "clean_selected": sorted(new_sel),
            "model_identity_changed": old_model != new_model,
            "candidate_ranking_changed": old_rank != new_rank,
            "selection_changed": old_sel != new_sel,
            "dividend_position_count": sum(1 for p in b.positions if p.dividend_count),
            "split_position_count": sum(1 for p in b.positions if p.split_count),
        })
    return {
        "quarters_compared": len(rows),
        "fitted_model_identities_changed": model_changed,
        "quarters_with_changed_candidate_rankings": changed_rankings,
        "quarters_with_changed_final_selections": changed_selections,
        "average_selected_stock_overlap": sum(overlaps) / len(overlaps) if overlaps else None,
        "minimum_quarter_portfolio_overlap": min(overlaps) if overlaps else None,
        "maximum_quarter_portfolio_overlap": max(overlaps) if overlaps else None,
        "quarter_rows": rows,
        "largest_positive_quarter_changes": sorted(rows, key=lambda r: (r["clean_return"] or 0) - (r["legacy_return"] or 0), reverse=True)[:5],
        "largest_negative_quarter_changes": sorted(rows, key=lambda r: (r["clean_return"] or 0) - (r["legacy_return"] or 0))[:5],
    }


def no_dividend_strategy_total_return(bt) -> dict:
    growth = 1.0
    div_growth = 1.0
    periods_with_divs = 0
    weighted_div_cash_return = 0.0
    for q in bt.quarter_results:
        if q.period_return is None:
            continue
        no_div_period = 0.0
        has_div = False
        for p in q.positions:
            if p.entry_resolved is None or p.exit_resolved is None or p.entry_resolved.price is None or p.exit_resolved.price is None:
                continue
            actions = []
            # The PositionOutcome does not retain action records; infer no-dividend delta from realized fields.
            if p.raw_return is None or p.capped_return is None:
                continue
            div_component = (p.dividend_cash / p.entry_resolved.price) if p.entry_resolved.price else 0.0
            raw_no_div = p.raw_return - div_component
            capped_no_div = max(-0.50, min(0.50, raw_no_div))
            no_div_period += p.target_weight * capped_no_div
            if p.dividend_count:
                has_div = True
                weighted_div_cash_return += p.target_weight * div_component
        if has_div:
            periods_with_divs += 1
        growth *= 1 + q.period_return
        div_growth *= 1 + no_div_period
    return {
        "strategy_holding_periods_containing_dividends": sum(1 for q in bt.quarter_results for p in q.positions if p.dividend_count),
        "quarters_containing_strategy_dividends": periods_with_divs,
        "aggregate_position_dividend_cash_per_share": sum(p.dividend_cash for q in bt.quarter_results for p in q.positions),
        "approx_weighted_dividend_cash_return_before_compounding": weighted_div_cash_return,
        "total_return_with_dividends": growth - 1.0,
        "approx_total_return_without_dividends": div_growth - 1.0,
        "approx_total_return_delta_from_dividends": (growth - 1.0) - (div_growth - 1.0),
    }


def raw_counts(raw_root: Path) -> dict:
    counts = {}
    for name in ("filings", "prices", "corporate_actions", "sic_history", "universe"):
        path = raw_root / f"{name}.json"
        if not path.exists():
            counts[name] = None
            continue
        data = json.loads(path.read_text())
        counts[name] = len(data)
        del data
        gc.collect()
    return counts


def build_reduced(dataset_name: str, raw_root: Path, manifest_path: Path, checkpoint_root: Path | None) -> dict:
    args = Args(raw_root, manifest_path, checkpoint_root)
    bundle = load_normalized_bundle(raw_root)
    calendar = _build_trading_calendar(bundle)
    periods = _periods_from_args(args, required=True)
    validation = _run_validation(bundle, calendar, periods)
    config = FilingMomentumMLConfig()
    targets = [(iid, p.quarter_end, p.entry_timestamp) for p in periods for iid in bundle.universe]
    feature_result = build_production_features(
        config=config,
        calendar=calendar,
        sector_encoder=SectorEncoder(),
        targets=targets,
        filings_by_instrument=bundle.filings_by_instrument,
        prices_by_instrument=bundle.prices_by_instrument,
        sector_by_instrument=bundle.sector_by_instrument,
        cache_identity=_feature_cache_identity(config, periods, bundle),
        cache_root=None,
        mode="training",
    )
    observations = feature_result.feature_pipeline_result.observations
    feature_map = {
        f"{obs.strategy_cohort_end.isoformat()}|{obs.instrument_id.symbol}": {name: obs.features.get(name, float("nan")) for name in PRICE_FEATURES}
        for obs in observations
    }
    obs_by_q = {}
    for obs in observations:
        obs_by_q.setdefault(obs.strategy_cohort_end, []).append(obs)
    label_map = {}
    for period in periods:
        outcomes = [
            build_forward_return_outcome(
                obs.instrument_id,
                period.quarter_end,
                obs.feature_timestamp,
                period.exit_timestamp,
                bundle.prices_by_instrument.get(obs.instrument_id, ()),
                period.exit_timestamp,
                bundle.corporate_actions_by_instrument.get(obs.instrument_id, ()),
            )
            for obs in obs_by_q.get(period.quarter_end, ())
        ]
        labeling = assign_quarterly_labels(outcomes, period.quarter_end, n_winners=config.n_winners)
        label_by_id = {a.instrument_id: a.label for a in labeling.assignments}
        for o in outcomes:
            label_map[f"{period.quarter_end.isoformat()}|{o.instrument_id.symbol}"] = {
                "label": label_by_id.get(o.instrument_id),
                "entry_price": o.entry_price,
                "exit_price": o.exit_price,
                "raw_return": o.raw_return,
                "clipped_return": o.clipped_return,
                "dividend_cash": o.dividend_cash,
                "dividend_count": o.dividend_count,
                "split_count": o.split_count,
            }
    run_inputs = _build_run_inputs(args, bundle, calendar, periods, with_report=False)
    result = run_filing_momentum_production_backtest(run_inputs)
    perf = result.performance_analysis or analyze_backtest_result(result.backtest_result)
    bt = result.backtest_result
    reduced = {
        "dataset_name": dataset_name,
        "manifest": _load_manifest(manifest_path).to_dict(),
        "validation_counts": validation.counts_by_severity(),
        "validation_first_errors": [
            {"category": i.category, "subject": i.subject, "message": i.message}
            for i in validation.issues
            if i.severity.value in {"fatal", "error"}
        ][:10],
        "feature_observation_count": len(feature_map),
        "feature_map": feature_map,
        "label_map": label_map,
        "run_state": result.state.value,
        "run_identity": result.run_identity,
        "manifest_identity": result.manifest_identity,
        "performance": performance_analysis_to_dict(perf),
        "backtest_result": bt,
        "backtest_summary": bt.to_dict(),
        "corporate_action_impact": no_dividend_strategy_total_return(bt),
    }
    del bundle, calendar, periods, feature_result, observations, obs_by_q, result, perf
    gc.collect()
    return reduced


def strip_for_json(reduced: dict) -> dict:
    out = dict(reduced)
    out.pop("feature_map", None)
    out.pop("label_map", None)
    out.pop("backtest_result", None)
    return out


def main() -> int:
    legacy_raw = REPO_ROOT / "data/raw/filing_momentum_ml"
    legacy_manifest = REPO_ROOT / "data/manifests/filing_momentum_ml/data_manifest.json"
    clean_raw = REPO_ROOT / "data/raw/filing_momentum_ml_split_adj_div_unadj_20260813"
    clean_manifest = REPO_ROOT / "data/manifests/filing_momentum_ml/data_manifest_split_adj_div_unadj_20260813.json"
    out_path = REPO_ROOT / "research/strategies/filing_momentum_ml/outputs/clean_reacquisition_analysis.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    legacy = build_reduced("legacy_corrected_baseline_dataset", legacy_raw, legacy_manifest, REPO_ROOT / "data/manifests/filing_momentum_ml_legacy_compare_20260813")
    clean = build_reduced("clean_split_adjusted_dividend_unadjusted_20260813", clean_raw, clean_manifest, REPO_ROOT / "data/manifests/filing_momentum_ml_clean_compare_20260813")

    report = {
        "window": {"start": START, "end": END, "benchmark": BENCHMARK},
        "legacy_raw_root": str(legacy_raw),
        "clean_raw_root": str(clean_raw),
        "legacy_counts": raw_counts(legacy_raw),
        "clean_counts": raw_counts(clean_raw),
        "legacy": strip_for_json(legacy),
        "clean": strip_for_json(clean),
        "feature_comparison": summarize_feature_diffs(legacy["feature_map"], clean["feature_map"]),
        "label_comparison": summarize_label_diffs(legacy["label_map"], clean["label_map"]),
        "selection_comparison": summarize_selection(legacy["backtest_result"], clean["backtest_result"]),
    }
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True, default=str))
    print(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
