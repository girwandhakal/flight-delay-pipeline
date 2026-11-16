---
goal: Add two-hour airport delay history and federal holiday predictors
version: 1
date_created: 2026-11-15
last_updated: 2026-11-15
owner: flight-delay-pipeline
status: In progress
tags: [feature, bts, calendar, chronological-validation]
---

# Introduction

![Status: In progress](https://img.shields.io/badge/status-In_progress-yellow)

Extend all three existing feature sets with six recent airport operational measurements and three federal holiday flags. Preserve the existing prediction cutoff, target cohort, chronological training, algorithms and explicit weather test-scoring commands.

## 1. Requirements & Constraints

- **REQ-001**: For prediction cutoff T (scheduled departure UTC), aggregate origin departures and destination arrivals with actual event times in the half-open interval (T - 120 minutes, T]. Include all reporting airlines, carry history across midnight/month/split boundaries, and exclude the predicted flight by stable flight_id even if it departed early.
- **REQ-002**: Add exactly origin_departure_delay_rate_2h, origin_departure_mean_delay_minutes_2h, origin_departure_count_2h, destination_arrival_delay_rate_2h, destination_arrival_mean_delay_minutes_2h, destination_arrival_count_2h. Rate is the fraction with gate delay >=15 minutes. Mean is sum(max(delay_minutes, 0))/count across all included events. Rates are fractions, not percentages. Zero events means count=0 and null rate/mean; invalid prediction cutoff means all six null. Saved model preprocessing fills null measurements with zero and keeps counts so zero evidence is distinguishable from observed on-time traffic.
- **REQ-003**: Assume operational events become available at their exact occurrence; no BTS publication lag. Build events directly from local Bronze CSVs, independently of the label-filtered Silver cohort. Require a valid actual gate clock and finite delay; departures do not require a valid arrival label and include diverted departures. Arrival events require a confirmed non-diverted arrival. Reconstruct actual departure UTC as scheduled departure UTC + DepDelay; actual arrival UTC as scheduled departure UTC + CRSElapsedTime + ArrDelay. Validate clock consistency against airport local time; reject missing/invalid/inconsistent events. No final delay-cause fields or current-flight actual predictors enter model inputs.
- **REQ-004**: Reuse historical airport reference intervals and the existing local_cutoff DST behavior. Unknown/ambiguous/nonexistent scheduled departures remain unmatched; no guessed clocks. Validate shipped reference checksums before using them. Invalid events must never change target cohort, labels or splits.
- **REQ-005**: Add exactly is_federal_holiday, is_day_before_federal_holiday, is_day_after_federal_holiday using FlightDate, the origin-local scheduled service date. Holiday dates are the union of actual and observed recurring US federal holidays from the installed pandas USFederalHolidayCalendar; include surrounding years for New Year/weekend observance and adjacency. Flags are independent and may overlap. No school breaks, special events, traffic-density, tail rotations or backlog predictors.
- **REQ-006**: Put rolling_window_minutes=120 in config.json under bts_features. Record feature version, policy, source-file fingerprints, reference checksums and configured splits in Gold _features.json. The normal destination and origin workflows must rebuild stale BTS features and invalidate stale weather parents through existing fingerprints. Old saved models retain their saved preprocessing; frozen artifact checks remain intact.
- **REQ-007**: Extend shared model input declarations and persisted preprocessing for all feature sets and both existing algorithms. No new dependencies, algorithm candidates, threshold rules, random splits or test-based selection. Update affected synthetic fixtures, README migration instructions and local changelog.
- **CON-001**: Do not run full-data jobs, downloads, real model training or real test scoring for implementation verification. Preserve unrelated working changes. Do not commit, stage or push; AGENTS.md stays untracked and docs stays ignored.
- **CON-002**: Restrict raw-data reads to required schedule/event fields; deduplicate stable event keys and compute airport histories using bounded Spark windows rather than all-pairs flight joins. Keep comments short, natural, one line and limited to non-obvious reasoning.

## 2. Implementation Steps

### Implementation Phase 1

- **GOAL-001**: Implement the nine-feature contract and dataset freshness integration.

| Task | Description | Completed | Date |
|------|-------------|-----------|------|
| TASK-001 | In scripts/bts/recent_delays.py implement raw event projection, verified historical UTC conversion, bounded rolling aggregation, holiday flags and feature-policy/freshness helpers; in config.json add the required window. | Yes | 2026-11-15 |
| TASK-002 | Depends on TASK-001. In scripts/bts/build_features.py enrich the existing cohort before writing chronological splits, persist reusable computations on disk, write _features.json only after all splits succeed, and keep current result-publication behavior. In pipeline/main.py replace existence-only BTS reuse checks with freshness checks. | Yes | 2026-11-15 |
| TASK-003 | Depends on TASK-001. In scripts/bts/train.py extend shared input lists and saved SQL preprocessing with the nine predictors and structural missing-value filling, preserving weather vector ordering and existing fold isolation. | Yes | 2026-11-15 |

### Implementation Phase 2

- **GOAL-002**: Verify implementation, complete two independent reviews and document actual outcomes.

| Task | Description | Completed | Date |
|------|-------------|-----------|------|
| TASK-004 | Depends on TASK-002 and TASK-003. Add tests/test_recent_delays.py and update tests/test_build_features.py, tests/test_bts_train.py, tests/test_bts_evaluate.py, tests/test_origin_weather.py and tests/test_destination_weather.py for boundaries, self-exclusion, raw event eligibility, clock conversion, holiday rules, cohort preservation, saved preprocessing and stale dataset rebuilding. Run focused tests followed by the full synthetic suite under Python 3.11/Java 17/PySpark 3.5.7. | | |
| TASK-005 | Depends on TASK-002 and TASK-003. Assign exactly two review sub-agents: one reviews correctness, simplicity and natural one-line comments; the other audits every requirement against the final diff and rejects extra scope. Resolve actionable findings and rerun affected checks. | Yes | 2026-11-15 |
| TASK-006 | Depends on TASK-004 and TASK-005. Update README.md with definitions, assumptions, normal-command rebuilding and runtime implications; update docs/changelog.md with actual Chicago recording time, references, checks, review outcomes and limitations. Mark this plan Completed only when all tasks are done. | | |

## 3. Alternatives

- **ALT-001**: Since-midnight history was rejected because midnight resets ongoing disruption and older same-day traffic dilutes recent changes.
- **ALT-002**: Tuning multiple window lengths was rejected because the user explicitly selected two hours. No claim that two hours is optimal will be made.
- **ALT-003**: Aggregating only Silver flights was rejected because final arrival-label/cohort filtering would discard otherwise observable departures. A bounded event/query timeline avoids a large inequality join.

## 4. Dependencies

- **DEP-001**: Existing Python 3.11, Java 17, PySpark 3.5.7 and pandas 2.2-compatible environment; no installation or dependency changes planned.
- **DEP-002**: Existing local Bronze BTS CSVs and versioned airport/time-zone references; synthetic tests provide isolated equivalents. TASK-002 and TASK-003 depend on TASK-001; phase 2 follows phase 1.

## 5. Files

- **FILE-001**: scripts/bts/recent_delays.py (new), scripts/bts/build_features.py, scripts/bts/train.py, pipeline/main.py and config.json implement acquisition-independent enrichment and reuse validation.
- **FILE-002**: tests/test_recent_delays.py (new) and affected tests named in TASK-004 validate behavior without real datasets.
- **FILE-003**: README.md, this plan and ignored docs/changelog.md describe the exact implementation and verification.

## 6. Testing

- **TEST-001**: Hand-calculated rolling fixtures cover the excluded lower bound, included upper bound, future events, early-departure self-exclusion, separate airports/roles, negative delay clipping, missing/invalid cutoff, zero history, midnight/month/split continuity and invariance to events after T.
- **TEST-002**: Raw event fixtures cover missing arrival labels, diverted departures, rejected diverted arrivals, missing/non-finite/invalid clocks, clock mismatch, duplicate keys, timezone differences, 2400 and DST gaps/repeats.
- **TEST-003**: Calendar fixtures cover recurring federal holidays, adjacent days, ordinary dates, actual versus observed weekend holidays, Juneteenth, Thanksgiving and year-boundary observance.
- **TEST-004**: Integration fixtures verify unchanged IDs/labels/splits, nine columns in all variants, structural filling and persisted model reload parity, stale manifests/config/source/reference detection, bootstrap failure propagation and no incidental result publication. Full synthetic suite and whitespace/syntax checks must pass.

## 7. Risks & Assumptions

- **ASSUMPTION-001**: BTS event records simulate immediately observable operational events, as explicitly authorized. Final records do not provide a complete live backlog or cancellation-announcement timeline.
- **ASSUMPTION-002**: Reference historical time zones and schedule elapsed times support UTC reconstruction; ambiguous or inconsistent records are omitted rather than guessed. Holiday flags are calendar descriptors, not a claim that every federal holiday causes more travel.
- **RISK-001**: Full-data rolling sorts and raw CSV parsing add runtime and disk use; synthetic verification does not measure production runtime or predictive benefit. Histories near acquisition start have limited coverage; counts describe only available valid completed events.
- **RISK-002**: Rebuilding BTS Gold invalidates weather parents and may prevent frozen historical runs from passing existing mutable-data fingerprints; never silently rescore or retrain old runs. Existing artifacts are not regenerated during this coding task.

## 8. Related Specifications / Further Reading

- [Repository instructions](../AGENTS.md), [README](../README.md), [configuration](../config.json).
- [Spark 3.5.7 window implementation](https://github.com/apache/spark/blob/v3.5.7/python/pyspark/sql/window.py).
- [pandas 2.2.3 holiday rules](https://github.com/pandas-dev/pandas/blob/v2.2.3/pandas/tseries/holiday.py).
