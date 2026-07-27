"""The canonical regime evaluator — one implementation shared by every execution context.

:class:`RegimeEvaluator` is the single place Markov + HMM classification
and gate combination happen. ``evaluate_one`` and ``evaluate_batch`` are
the "direct in-process" and "batch in-process" adapters this stage
implements — both call exactly the same component-evaluation methods, so
there is no possibility of live/backtest/batch math drifting apart the
way the legacy prototype's separate worker scripts could. A future
subprocess adapter (only if ever needed for a different HMM dependency
boundary) would serialize the same typed requests/results through this
same evaluator, never reimplementing the math.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Sequence

from atlas_quant.data.records import DailyPriceObservation
from atlas_quant.domain.audit import AuditRecord, AuditTrail
from atlas_quant.domain.identifiers import InstrumentId
from atlas_quant.strategies.filing_momentum_ml.regime_config import RegimeConfig
from atlas_quant.strategies.filing_momentum_ml.regime_domain import (
    ComponentAvailability,
    ComponentClassification,
    RegimeClassification,
    RegimeResult,
    WindowClassification,
)
from atlas_quant.strategies.filing_momentum_ml.regime_hmm import (
    HMMFitter,
    build_weekly_observations,
    map_bear_state,
)
from atlas_quant.strategies.filing_momentum_ml.regime_markov import (
    annualized_volatility,
    confirmed_bear,
    daily_returns,
    label_window,
    multi_window_vote,
    window_threshold,
)


@dataclass(frozen=True, slots=True)
class RegimeEvaluationRequest:
    """One instrument's inputs for one regime evaluation — the unit ``evaluate_batch`` iterates over."""

    instrument_id: InstrumentId
    prices: tuple[DailyPriceObservation, ...]
    evaluation_timestamp: datetime
    data_cutoff: datetime


class RegimeEvaluator:
    """The canonical regime subsystem, report §5b."""

    def __init__(self, config: RegimeConfig) -> None:
        self.config = config

    def evaluate_markov_component(
        self,
        instrument_id: InstrumentId,
        prices: Sequence[DailyPriceObservation],
        evaluation_timestamp: datetime,
        data_cutoff: datetime,
    ) -> ComponentClassification:
        """Report §5b.1: multi-window vol-adjusted Markov classification."""
        cfg = self.config
        audit = AuditTrail()

        eligible = sorted(
            (p for p in prices if p.trading_date <= data_cutoff.date()),
            key=lambda p: p.trading_date,
        )
        closes = [p.close for p in eligible]
        provenance = tuple(p.provenance for p in eligible[-1:])

        if not closes:
            return ComponentClassification(
                component="markov",
                instrument_id=instrument_id,
                evaluation_timestamp=evaluation_timestamp,
                data_cutoff=data_cutoff,
                classification=RegimeClassification.UNKNOWN,
                is_bear=False,
                availability=ComponentAvailability.MISSING_PRICES,
                confidence=None,
                observation_count=0,
                required_observation_count=max(cfg.markov_windows) + cfg.markov_persistence,
                windows=(),
                config_identity=cfg.identity(),
                provenance=(),
                audit_trail=audit.append(
                    AuditRecord(
                        stage="markov",
                        message="no eligible price observations as of data_cutoff",
                        timestamp=evaluation_timestamp,
                    )
                ),
            )

        returns = daily_returns(closes)
        vol_window = returns[-cfg.markov_volatility_lookback_days :]
        ann_vol = (
            annualized_volatility(vol_window, cfg.markov_annualization_factor)
            if len(vol_window) >= cfg.markov_volatility_lookback_days
            else float("nan")
        )

        windows: list[WindowClassification] = []
        confirmations: list[bool] = []
        for w in cfg.markov_windows:
            required = w + cfg.markov_persistence
            if len(closes) < required or len(vol_window) < cfg.markov_volatility_lookback_days:
                windows.append(
                    WindowClassification(
                        window_days=w,
                        threshold=float("nan"),
                        annualized_volatility=ann_vol,
                        classification=RegimeClassification.UNKNOWN,
                        persistence_count=0,
                        persistence_required=cfg.markov_persistence,
                        bear_confirmed=False,
                        reason="insufficient price history for this window",
                    )
                )
                confirmations.append(False)
                continue

            threshold = window_threshold(
                ann_vol,
                w,
                multiplier=cfg.markov_threshold_multiplier,
                floor=cfg.markov_threshold_floor,
                annualization_factor=cfg.markov_annualization_factor,
            )
            labels = label_window(closes, w, threshold)
            confirmed, persistence_count = confirmed_bear(labels, cfg.markov_persistence)
            windows.append(
                WindowClassification(
                    window_days=w,
                    threshold=threshold,
                    annualized_volatility=ann_vol,
                    classification=labels[-1],
                    persistence_count=persistence_count,
                    persistence_required=cfg.markov_persistence,
                    bear_confirmed=confirmed,
                )
            )
            confirmations.append(confirmed)

        available_windows = [w for w in windows if w.reason is None]
        if not available_windows:
            return ComponentClassification(
                component="markov",
                instrument_id=instrument_id,
                evaluation_timestamp=evaluation_timestamp,
                data_cutoff=data_cutoff,
                classification=RegimeClassification.UNKNOWN,
                is_bear=False,
                availability=ComponentAvailability.INSUFFICIENT_HISTORY,
                confidence=None,
                observation_count=len(closes),
                required_observation_count=max(cfg.markov_windows) + cfg.markov_persistence,
                windows=tuple(windows),
                config_identity=cfg.identity(),
                provenance=provenance,
                audit_trail=audit.append(
                    AuditRecord(
                        stage="markov",
                        message="no window had sufficient history",
                        timestamp=evaluation_timestamp,
                    )
                ),
            )

        is_bear = multi_window_vote(confirmations, cfg.markov_min_window_agreement)
        classification = (
            RegimeClassification.BEAR if is_bear else available_windows[-1].classification
        )
        audit = audit.append(
            AuditRecord(
                stage="markov",
                message=f"{sum(confirmations)} of {len(windows)} window(s) confirmed Bear",
                timestamp=evaluation_timestamp,
                data={"window_days": [w.window_days for w in windows], "confirmations": confirmations},
            )
        )
        return ComponentClassification(
            component="markov",
            instrument_id=instrument_id,
            evaluation_timestamp=evaluation_timestamp,
            data_cutoff=data_cutoff,
            classification=classification,
            is_bear=is_bear,
            availability=ComponentAvailability.OK,
            confidence=None,
            observation_count=len(closes),
            required_observation_count=max(cfg.markov_windows) + cfg.markov_persistence,
            windows=tuple(windows),
            config_identity=cfg.identity(),
            provenance=provenance,
            audit_trail=audit,
        )

    def evaluate_hmm_component(
        self,
        instrument_id: InstrumentId,
        prices: Sequence[DailyPriceObservation],
        evaluation_timestamp: datetime,
        data_cutoff: datetime,
        fitter: HMMFitter,
    ) -> ComponentClassification:
        """Report §5b.2: 3-state Gaussian HMM on weekly (return, 4-week vol) observations."""
        cfg = self.config
        weekly = build_weekly_observations(
            prices, data_cutoff.date(), volatility_window_weeks=cfg.hmm_volatility_window_weeks
        )
        usable = [w for w in weekly if w.weekly_return is not None and w.rolling_volatility_4w is not None]
        provenance = tuple(p.provenance for p in sorted(prices, key=lambda p: p.trading_date)[-1:])

        if len(usable) < cfg.hmm_min_weekly_observations:
            return ComponentClassification(
                component="hmm",
                instrument_id=instrument_id,
                evaluation_timestamp=evaluation_timestamp,
                data_cutoff=data_cutoff,
                classification=RegimeClassification.UNKNOWN,
                is_bear=False,
                availability=ComponentAvailability.INSUFFICIENT_HISTORY,
                confidence=None,
                observation_count=len(usable),
                required_observation_count=cfg.hmm_min_weekly_observations,
                windows=(),
                config_identity=cfg.identity(),
                provenance=provenance,
                audit_trail=AuditTrail().append(
                    AuditRecord(
                        stage="hmm",
                        message=(
                            f"only {len(usable)} usable weekly observation(s), "
                            f"need {cfg.hmm_min_weekly_observations}"
                        ),
                        timestamp=evaluation_timestamp,
                    )
                ),
            )

        observations = [(w.weekly_return, w.rolling_volatility_4w) for w in usable]
        fit_result = fitter.fit_predict(
            observations,
            n_states=cfg.hmm_state_count,
            covariance_type=cfg.hmm_covariance_type,
            n_iter=cfg.hmm_n_iter,
            random_state=cfg.hmm_random_state,
        )

        if fit_result.error is not None or not fit_result.predicted_states:
            return ComponentClassification(
                component="hmm",
                instrument_id=instrument_id,
                evaluation_timestamp=evaluation_timestamp,
                data_cutoff=data_cutoff,
                classification=RegimeClassification.UNKNOWN,
                is_bear=False,
                availability=ComponentAvailability.NUMERICAL_FIT_FAILURE,
                confidence=None,
                observation_count=len(usable),
                required_observation_count=cfg.hmm_min_weekly_observations,
                windows=(),
                config_identity=cfg.identity(),
                provenance=provenance,
                audit_trail=AuditTrail().append(
                    AuditRecord(
                        stage="hmm",
                        message=f"HMM fit failed: {fit_result.error or 'no predicted states'}",
                        timestamp=evaluation_timestamp,
                    )
                ),
            )

        bear_map = map_bear_state(fit_result.state_means)
        current_state = fit_result.predicted_states[-1]
        state_label = bear_map.get(current_state)
        classification = {
            "bear": RegimeClassification.BEAR,
            "sideways": RegimeClassification.NEUTRAL,
            "bull": RegimeClassification.BULL,
        }.get(state_label, RegimeClassification.UNKNOWN)

        audit = AuditTrail().append(
            AuditRecord(
                stage="hmm",
                message=f"current state {current_state} mapped to {state_label!r}",
                timestamp=evaluation_timestamp,
                data={"converged": fit_result.converged, "state_means": list(fit_result.state_means)},
            )
        )
        return ComponentClassification(
            component="hmm",
            instrument_id=instrument_id,
            evaluation_timestamp=evaluation_timestamp,
            data_cutoff=data_cutoff,
            classification=classification,
            is_bear=classification == RegimeClassification.BEAR,
            availability=ComponentAvailability.OK,
            confidence=None,
            observation_count=len(usable),
            required_observation_count=cfg.hmm_min_weekly_observations,
            windows=(),
            config_identity=cfg.identity(),
            provenance=provenance,
            audit_trail=audit,
        )

    def combine(
        self,
        markov: ComponentClassification,
        hmm: ComponentClassification,
        instrument_id: InstrumentId,
        evaluation_timestamp: datetime,
        data_cutoff: datetime,
    ) -> RegimeResult:
        """Report §5.1's gate-mode truth table, generalized from the report's
        documented "both" behavior (§5b.2: an unavailable component can never
        satisfy a required confirmation)."""
        mode = self.config.gate_mode
        warnings: list[str] = []
        if markov.availability != ComponentAvailability.OK:
            warnings.append(f"markov component unavailable: {markov.availability.value}")
        if hmm.availability != ComponentAvailability.OK:
            warnings.append(f"hmm component unavailable: {hmm.availability.value}")

        if mode == "none":
            is_blocked, reason = False, None
        elif mode == "markov":
            is_blocked = markov.is_bear
            reason = "markov confirmed Bear" if is_blocked else None
        elif mode == "hmm":
            is_blocked = hmm.is_bear
            reason = "hmm confirmed Bear" if is_blocked else None
        elif mode == "either":
            is_blocked = markov.is_bear or hmm.is_bear
            reason = (
                "markov and hmm both confirmed Bear"
                if markov.is_bear and hmm.is_bear
                else "markov confirmed Bear"
                if markov.is_bear
                else "hmm confirmed Bear"
                if hmm.is_bear
                else None
            )
        else:  # "both"
            is_blocked = markov.is_bear and hmm.is_bear
            reason = "both markov and hmm confirmed Bear" if is_blocked else None

        audit = AuditTrail().append(
            AuditRecord(
                stage="gate",
                message=f"gate_mode={mode!r} -> is_blocked={is_blocked}",
                timestamp=evaluation_timestamp,
                data={"markov_is_bear": markov.is_bear, "hmm_is_bear": hmm.is_bear},
            )
        )
        return RegimeResult(
            instrument_id=instrument_id,
            evaluation_timestamp=evaluation_timestamp,
            data_cutoff=data_cutoff,
            markov=markov,
            hmm=hmm,
            gate_mode=mode,
            is_blocked=is_blocked,
            block_reason=reason,
            warnings=tuple(warnings),
            config_identity=self.config.identity(),
            provenance=markov.provenance + hmm.provenance,
            audit_trail=audit,
        )

    def evaluate_one(
        self,
        instrument_id: InstrumentId,
        prices: Sequence[DailyPriceObservation],
        evaluation_timestamp: datetime,
        data_cutoff: datetime,
        fitter: HMMFitter,
    ) -> RegimeResult:
        """Direct in-process adapter: evaluate one instrument, right now."""
        markov = self.evaluate_markov_component(instrument_id, prices, evaluation_timestamp, data_cutoff)
        hmm = self.evaluate_hmm_component(instrument_id, prices, evaluation_timestamp, data_cutoff, fitter)
        return self.combine(markov, hmm, instrument_id, evaluation_timestamp, data_cutoff)

    def evaluate_batch(
        self, requests: Sequence[RegimeEvaluationRequest], fitter: HMMFitter
    ) -> tuple[RegimeResult, ...]:
        """Batch in-process adapter: evaluate many instruments via the same
        per-instrument methods ``evaluate_one`` uses. Preserves ``requests``' input
        order; never reimplements Markov/HMM math itself."""
        return tuple(
            self.evaluate_one(
                request.instrument_id, request.prices, request.evaluation_timestamp, request.data_cutoff, fitter
            )
            for request in requests
        )
