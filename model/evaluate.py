"""Everything the models get wrong, measured on cohorts they never saw.

Run as `python -m model.evaluate`. All numbers come from graduates of 2025 and
2026; the models were fitted on 2022 and earlier, with 2023-24 held back for
early stopping. The split is by graduation year rather than at random because
classmates share courses, employers and a job market, so a random split would put
near-duplicates on both sides and quietly inflate everything below.
"""

from __future__ import annotations

import random

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, mean_absolute_error, roc_auc_score

from . import datasets, skills
from . import state as state_module
from .planner import Planner
from .train_next_step import NextStepModel, ranking_report
from .train_outcome import OutcomeModel

BACKTEST_ALUMNI = 546
BACKTEST_GOALS_PER_ALUMNUS = 3
BACKTEST_SEED = 23
BACKTEST_PROGRESS = 0.6
BOOTSTRAP_DRAWS = 2000
TOP_K = 3


def outcome_metrics(model: OutcomeModel, frame: pd.DataFrame) -> None:
    test = frame[frame["split"] == "test"]
    print(f"\n=== Model B: outcome distribution ({len(test):,} held-out snapshots) ===")

    for head, target in (("family", "outcome_family"), ("industry", "outcome_industry")):
        classifier = getattr(model, head)
        matrix = test[model.features]
        predicted = classifier.predict(matrix)
        truth = test[target]
        probabilities = classifier.predict_proba(matrix)
        classes = list(classifier.classes_)
        order = np.argsort(-probabilities, axis=1)
        positions = np.array(
            [classes.index(value) if value in classes else -1 for value in truth]
        )
        top3 = float(np.mean([positions[i] in order[i, :3] for i in range(len(truth))]))
        majority = truth.value_counts(normalize=True).iloc[0]
        print(
            f"  {head:9s} top-1 {float((predicted == truth).mean()):.3f} "
            f"(majority class {majority:.3f})  top-3 {top3:.3f}  "
            f"macro-F1 {f1_score(truth, predicted, average='macro', zero_division=0):.3f}"
        )

    print("\n  Family accuracy by how far through the degree the snapshot is.")
    print("  A freshman's eventual job family is close to unknowable; the point of")
    print("  this table is to show where the model is worth listening to.")
    buckets = pd.cut(
        test["progress"],
        [-0.01, 0.25, 0.5, 0.75, 1.01],
        labels=["first quarter", "second quarter", "third quarter", "final quarter"],
    )
    predicted = model.family.predict(test[model.features])
    correct = pd.Series(predicted == test["outcome_family"].to_numpy(), index=test.index)
    for bucket, group in correct.groupby(buckets, observed=True):
        print(f"    {str(bucket):16s} top-1 {group.mean():.3f}  (n={len(group)})")


def salary_calibration(model: OutcomeModel, frame: pd.DataFrame) -> None:
    paid = frame.dropna(subset=["salary_real"])
    test = paid[paid["split"] == "test"]
    quantiles = model.salary_quantiles(test)
    truth = test["salary_real"].to_numpy()
    low = quantiles[0.1].to_numpy()
    median = quantiles[0.5].to_numpy()
    high = quantiles[0.9].to_numpy()

    print(f"\n=== Salary, in 2026 dollars ({len(test):,} held-out snapshots) ===")
    baseline = np.full_like(truth, float(paid[paid["split"] == "train"]["salary_real"].median()))
    print(f"  MAE on the median          ${mean_absolute_error(truth, median):,.0f}")
    print(f"  MAE guessing the median    ${mean_absolute_error(truth, baseline):,.0f}")
    print(f"  p10-p90 band covers        {float(np.mean((truth >= low) & (truth <= high))):.1%} (nominal 80%)")
    print(f"  below p10                  {float(np.mean(truth < low)):.1%} (nominal 10%)")
    print(f"  above p90                  {float(np.mean(truth > high)):.1%} (nominal 10%)")

    print("\n  Coverage across the range, to check the band is not just wide on average.")
    deciles = pd.qcut(median, 4, labels=["lowest", "low", "high", "highest"])
    inside = pd.Series((truth >= low) & (truth <= high))
    for bucket, group in inside.groupby(pd.Series(deciles), observed=True):
        print(f"    predicted {str(bucket):8s} coverage {group.mean():.1%}  (n={len(group)})")


def next_step_metrics(model: NextStepModel, frame: pd.DataFrame, limit: int) -> None:
    test = frame[frame["split"] == "test"]
    scores = model.score_frame(test)
    print(f"\n=== Model A: next-step ranker ({len(test):,} held-out pairs) ===")
    print(f"  pairwise ROC-AUC {roc_auc_score(test['label'], scores):.3f}")
    ranking_report(model, limit=limit)


def _backtest_snapshot(timeline) -> tuple[state_module.State, set[str]] | None:
    """The decision point closest to `BACKTEST_PROGRESS` through the degree."""
    steps = [
        (snapshot, set(taken))
        for snapshot, taken in datasets.walk(timeline)
        if taken
    ]
    if not steps:
        return None
    target = BACKTEST_PROGRESS * (len(steps) - 1)
    return steps[int(round(target))]


def planner_backtest(
    planner: Planner,
    alumni_limit: int = BACKTEST_ALUMNI,
    goals_per_alumnus: int = BACKTEST_GOALS_PER_ALUMNUS,
    seed: int = BACKTEST_SEED,
) -> None:
    """Did the alumni who happened to take the recommended step do better?

    The goal is assigned, not read off the person's own outcome. Using their real
    outcome as the goal would make the test circular -- they reached it by
    definition. So each held-out graduate is scored against job families picked at
    random, the planner is asked what it would have told them at one decision
    point, and the question becomes whether the graduates who happened to take one
    of its top three suggestions reached that assigned family more often than the
    ones who did not.

    This is observational and not a trial. Students who take a certification may
    differ from those who do not in ways the state does not capture, so the gap
    below is an association, not a measured effect of following the advice.
    """
    rng = random.Random(seed)
    families = list(skills.job_families())
    timelines = datasets.timelines()
    outcomes = datasets.outcomes()

    test_ids = [
        campus_id
        for campus_id, timeline in timelines.items()
        if timeline.is_alumnus
        and datasets.split_label(timeline.graduation_year) == "test"
        and campus_id in outcomes
    ]
    rng.shuffle(test_ids)
    test_ids = test_ids[:alumni_limit]

    records: list[dict] = []
    for campus_id in test_ids:
        snapshot = _backtest_snapshot(timelines[campus_id])
        if snapshot is None:
            continue
        state, taken = snapshot
        actual_family = outcomes[campus_id]["outcome_family"]
        actual_salary = outcomes[campus_id]["salary_real"]

        for family in rng.sample(families, goals_per_alumnus):
            goal = state_module.Goal(job_family=family)
            chosen, _, _ = planner.rank(state, goal, width=TOP_K + 3)
            if not chosen:
                continue
            recommended = [action for _, _, action in chosen]
            # Split out the steps that are actually a choice. Course enrolment is
            # close to deterministic -- everyone takes courses every term, and the
            # curriculum decides which -- so counting a course as "followed the
            # advice" swamps the comparison with people who simply enrolled.
            levers = [
                action.action_id for action in recommended if not action.is_course
            ][:TOP_K]
            records.append(
                {
                    "goal": family,
                    "followed": bool(
                        taken & {action.action_id for action in recommended[:TOP_K]}
                    ),
                    "followed_lever": bool(taken & set(levers)),
                    "reached": actual_family == family,
                    "salary": actual_salary,
                }
            )

    if not records:
        print("\n=== Planner backtest: no usable decision points ===")
        return

    frame = pd.DataFrame(records)
    print(
        f"\n=== Planner backtest ({len(test_ids)} held-out graduates, "
        f"{len(frame)} assigned-goal decisions) ==="
    )

    generator = np.random.default_rng(seed)
    for column, label in (
        ("followed", f"any of the top {TOP_K}"),
        ("followed_lever", f"top {TOP_K} excluding courses"),
    ):
        followed = frame[frame[column]]
        ignored = frame[~frame[column]]
        print(f"\n  Recommendation counted as followed: {label}")
        print(
            f"    followed in {len(followed)} of {len(frame)} decisions "
            f"({len(followed) / len(frame):.1%})"
        )
        if followed.empty or ignored.empty:
            print("    not enough of both groups to compare")
            continue

        left_values = followed["reached"].to_numpy()
        right_values = ignored["reached"].to_numpy()
        reached_followed = float(left_values.mean())
        reached_ignored = float(right_values.mean())
        lift = (
            reached_followed / reached_ignored if reached_ignored > 0 else float("inf")
        )

        # Quote an interval, not just the ratio. At these group sizes the point
        # estimate moves a long way between samples, and a ratio whose interval
        # spans 1.0 is not evidence of anything.
        draws = []
        for _ in range(BOOTSTRAP_DRAWS):
            left = generator.choice(left_values, len(left_values), replace=True).mean()
            right = generator.choice(
                right_values, len(right_values), replace=True
            ).mean()
            if right > 0:
                draws.append(left / right)
        low, high = (
            np.percentile(draws, [2.5, 97.5]) if draws else (float("nan"),) * 2
        )
        print(f"    reached the assigned family, followed  {reached_followed:.1%}")
        print(f"    reached the assigned family, did not   {reached_ignored:.1%}")
        print(f"    ratio  {lift:.2f}x  (95% CI {low:.2f}-{high:.2f})")
        if low <= 1.0 <= high:
            print("    interval spans 1.0, so this is not a detectable difference")

        for name, group in (("followed", followed), ("did not", ignored)):
            salaries = group["salary"].dropna()
            if len(salaries):
                print(
                    f"    median first salary, {name:8s} "
                    f"${salaries.median():,.0f}  (n={len(salaries)})"
                )


def main(ranking_limit: int = 1500) -> None:
    outcome_model = OutcomeModel.load()
    next_step_model = NextStepModel.load()
    outcome_frame = datasets.outcome_dataset()
    next_step_frame = datasets.next_step_dataset()

    print("Held-out cohorts: graduated 2025-2026. Trained on 2022 and earlier.")
    outcome_metrics(outcome_model, outcome_frame)
    salary_calibration(outcome_model, outcome_frame)
    next_step_metrics(next_step_model, next_step_frame, ranking_limit)
    planner_backtest(Planner(next_step_model, outcome_model, explain=False))


if __name__ == "__main__":
    main()
