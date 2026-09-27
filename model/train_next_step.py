"""Model A: which opportunity would this student plausibly take next?

Framed as a relevance ranker over (state, action) pairs rather than a softmax
over a few hundred action classes. Two reasons that matters:

* the score depends on interaction features -- how much of the skill gap this
  action closes for *this* student -- so the model cannot fall back on
  memorising which opportunities are popular overall;
* it generalises to a role-and-industry combination nobody in the training data
  happened to hold, because such a pair is described by its features rather than
  by a class index.

Positives are what a person actually did next. Negatives are sampled from the
opportunities that were feasible at that same moment and not taken, which makes
the task "pick the right one out of the real options" instead of "pick the one
that is not absurd".
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from . import actions as action_space
from . import data, datasets
from . import state as state_module

MODEL_PATH = data.ARTIFACT_DIR / "next_step_model.pkl"
RANKING_GROUPS = 500
RANKING_SEED = 5

PARAMS = dict(
    objective="binary",
    learning_rate=0.05,
    num_leaves=63,
    min_child_samples=50,
    subsample=0.9,
    subsample_freq=1,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    n_estimators=1200,
    verbose=-1,
)


@dataclass
class NextStepModel:
    booster: lgb.LGBMClassifier
    features: list[str]
    categories: dict[str, list]

    def score_frame(self, frame: pd.DataFrame) -> np.ndarray:
        aligned = datasets.align_categories(frame, self.categories)
        return self.booster.predict_proba(aligned[self.features])[:, 1]

    def pair_frame(
        self,
        state: state_module.State,
        candidates: list[action_space.Action],
        goal: state_module.Goal | None,
    ) -> pd.DataFrame:
        base = state_module.state_features(state)
        base.update(state_module.goal_features(state, goal))
        rows = []
        for action in candidates:
            row = dict(base)
            row.update(state_module.action_features(state, action, goal))
            rows.append(row)
        return datasets.to_frame(rows)

    def score_actions(
        self,
        state: state_module.State,
        candidates: list[action_space.Action],
        goal: state_module.Goal | None = None,
    ) -> np.ndarray:
        """Feasibility of each candidate as the student's next step."""
        if not candidates:
            return np.zeros(0)
        return self.score_frame(self.pair_frame(state, candidates, goal))

    def explainer(self):
        """A SHAP explainer over the booster, built once and reused.

        Constructing it costs more than using it, so it is cached on the
        instance rather than rebuilt per recommendation.
        """
        if getattr(self, "_explainer", None) is None:
            import shap

            self._explainer = shap.TreeExplainer(self.booster)
        return self._explainer

    def explain(
        self,
        state: state_module.State,
        candidates: list[action_space.Action],
        goal: state_module.Goal | None = None,
        top: int = 4,
    ) -> list[list[tuple[str, float]]]:
        """Per candidate, the features that pushed its score up, largest first."""
        if not candidates:
            return []
        frame = datasets.align_categories(
            self.pair_frame(state, candidates, goal), self.categories
        )[self.features]
        values = np.asarray(self.explainer().shap_values(frame))
        if values.ndim == 3:
            values = values[:, :, -1]
        results = []
        for row in values:
            order = np.argsort(-row)
            results.append(
                [
                    (self.features[index], float(row[index]))
                    for index in order[:top]
                    if row[index] > 0
                ]
            )
        return results

    def save(self, path=MODEL_PATH) -> None:
        # Store the parts, not the instance. Pickling `self` records the class by
        # the module it was defined in, which is `__main__` when training is run
        # as a script, and then nothing else can load it.
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as handle:
            pickle.dump(
                {
                    "booster": self.booster,
                    "features": self.features,
                    "categories": self.categories,
                },
                handle,
            )

    @staticmethod
    def load(path=MODEL_PATH) -> "NextStepModel":
        with open(path, "rb") as handle:
            parts = pickle.load(handle)
        return NextStepModel(**parts)


def train(rebuild: bool = False) -> NextStepModel:
    frame = datasets.next_step_dataset(rebuild=rebuild)
    features = datasets.feature_columns(frame)
    categories = datasets.category_levels(frame)

    train_rows = frame[frame["split"] == "train"]
    validation_rows = frame[frame["split"] == "validation"]
    test_rows = frame[frame["split"] == "test"]
    print(
        f"pairs: {len(train_rows):,} train / {len(validation_rows):,} validation / "
        f"{len(test_rows):,} test, {len(features)} features"
    )

    booster = lgb.LGBMClassifier(**PARAMS)
    booster.fit(
        train_rows[features],
        train_rows["label"],
        eval_X=validation_rows[features],
        eval_y=validation_rows["label"],
        eval_metric="auc",
        callbacks=[lgb.early_stopping(80, verbose=False)],
    )

    model = NextStepModel(booster=booster, features=features, categories=categories)

    scores = model.score_frame(test_rows)
    truth = test_rows["label"].to_numpy()
    print("\n-- held-out cohorts (graduated 2025-2026) --")
    print(f"pairwise ROC-AUC     {roc_auc_score(truth, scores):.3f}")
    print(f"average precision    {average_precision_score(truth, scores):.3f}")
    print(f"best iteration       {booster.best_iteration_}")

    ranking_report(model)
    model.save()
    print(f"\nsaved {MODEL_PATH.relative_to(data.ROOT_DIR)}")
    return model


def _popularity_scores(candidates: list[action_space.Action]) -> np.ndarray:
    return np.array([action.observations for action in candidates], dtype=float)


def _goal_matched_popularity(
    candidates: list[action_space.Action], goal: state_module.Goal | None
) -> np.ndarray:
    """Stand-in for the previous heuristic recommender.

    That approach found the alumni whose outcome best matched the stated goal and
    then surfaced whatever those alumni did most often. Scoring by overall
    frequency, doubled when the opportunity points at the goal's family or
    industry, reproduces its ranking without needing the database.
    """
    scores = np.log1p(_popularity_scores(candidates))
    if goal is None:
        return scores
    weights = np.ones(len(candidates))
    for index, action in enumerate(candidates):
        if action.job_family == goal.job_family:
            weights[index] *= 2.0
        if goal.industry and action.industry == goal.industry:
            weights[index] *= 1.5
    return scores * weights


def _ranking_groups(limit: int, seed: int):
    """Held-out decision points, each with the full set of feasible options.

    Evaluating on the five sampled negatives would flatter the model. What
    matters is whether the opportunity a student actually took surfaces near the
    top of every option open to them, so the whole candidate set is rebuilt here.
    """
    rng = np.random.default_rng(seed)
    catalog = action_space.by_id()
    pre_graduation = [
        action for action in action_space.action_space() if not action.is_role
    ]

    timelines = datasets.timelines()
    test_ids = [
        campus_id
        for campus_id, timeline in timelines.items()
        if timeline.is_alumnus
        and datasets.split_label(timeline.graduation_year) == "test"
    ]
    rng.shuffle(test_ids)

    produced = 0
    for campus_id in test_ids:
        timeline = timelines[campus_id]
        goal = datasets.goal_for(campus_id)
        for snapshot, taken_ids in datasets.walk(timeline):
            relevant = {
                action_id
                for action_id in taken_ids
                if action_id in catalog and not catalog[action_id].is_role
            }
            if not relevant:
                continue
            candidates = [
                action
                for action in pre_graduation
                if action_space.feasible(action, snapshot)
            ]
            if len(candidates) < 10:
                continue
            yield snapshot, goal, candidates, relevant
            produced += 1
            if produced >= limit:
                return


KINDS = ("Course", "Internship/Co-op", "Certification", "Campus activity")


def _kind_of(action: action_space.Action) -> str:
    if action.is_course:
        return "Course"
    if action.action_type in ("Internship", "Co-op"):
        return "Internship/Co-op"
    if action.action_type == "Certification":
        return "Certification"
    return "Campus activity"


def _rank_of(ranked: list[str], target: str) -> int | None:
    try:
        return ranked.index(target)
    except ValueError:
        return None


def ranking_report(model: NextStepModel, limit: int = RANKING_GROUPS) -> dict:
    """Score every feasible option at held-out decision points.

    Metrics are per *positive*, not per decision point. Averaging over decision
    points and taking the best-ranked hit in each looks far better than the model
    deserves, because almost every term contains several courses and one of them
    is usually easy to call.

    Internships are additionally scored at the archetype level. There are 265
    internship actions because each of 20 role archetypes is crossed with 16
    industries, and picking the exact employer industry is close to a coin flip
    -- the advice a student can act on is "do a cybersecurity internship", so
    that is measured separately from which industry it happens to be in.
    """
    scorers = {
        "model": lambda snapshot, candidates, goal: model.score_actions(
            snapshot, candidates, goal
        ),
        "popularity": lambda snapshot, candidates, goal: _popularity_scores(candidates),
        "goal-matched popularity": lambda snapshot, candidates, goal: (
            _goal_matched_popularity(candidates, goal)
        ),
    }
    per_kind: dict[str, dict[str, dict[str, list]]] = {
        name: {kind: {"r3": [], "r5": [], "mrr": []} for kind in KINDS}
        for name in scorers
    }
    archetype: dict[str, dict[str, list]] = {
        name: {"r3": [], "mrr": [], "options": []} for name in scorers
    }
    candidate_counts = []
    unreachable = 0
    total_positives = 0

    for snapshot, goal, candidates, relevant in _ranking_groups(limit, RANKING_SEED):
        candidate_counts.append(len(candidates))
        catalog = {action.action_id: action for action in candidates}

        for name, scorer in scorers.items():
            scores = scorer(snapshot, candidates, goal)
            ranked = [candidates[index].action_id for index in np.argsort(-scores)]

            # Collapse to role archetypes, taking each archetype's best score.
            best_by_position: dict[str, float] = {}
            for action, score in zip(candidates, scores):
                if _kind_of(action) != "Internship/Co-op":
                    continue
                key = action.position
                best_by_position[key] = max(best_by_position.get(key, -np.inf), score)
            archetype_ranking = sorted(
                best_by_position, key=lambda key: -best_by_position[key]
            )

            for action_id in relevant:
                action = catalog.get(action_id)
                if action is None:
                    if name == "model":
                        unreachable += 1
                        total_positives += 1
                    continue
                if name == "model":
                    total_positives += 1
                position = _rank_of(ranked, action_id)
                bucket = per_kind[name][_kind_of(action)]
                bucket["r3"].append(float(position is not None and position < 3))
                bucket["r5"].append(float(position is not None and position < 5))
                bucket["mrr"].append(0.0 if position is None else 1.0 / (position + 1))

                if _kind_of(action) == "Internship/Co-op" and archetype_ranking:
                    place = _rank_of(archetype_ranking, action.position)
                    archetype[name]["r3"].append(float(place is not None and place < 3))
                    archetype[name]["mrr"].append(
                        0.0 if place is None else 1.0 / (place + 1)
                    )
                    archetype[name]["options"].append(len(archetype_ranking))

    print(
        f"\n-- ranking the full feasible set: {len(candidate_counts)} held-out "
        f"decision points, median {int(np.median(candidate_counts))} options each --"
    )
    print(
        f"   {total_positives} opportunities actually taken; {unreachable} "
        f"({unreachable / max(total_positives, 1):.1%}) fell outside the feasible "
        f"set and are counted as misses"
    )

    summary: dict[str, dict] = {}
    header = f"{'':24s} {'n':>5s} {'r@3':>6s} {'r@5':>6s} {'MRR':>6s}"
    for kind in KINDS:
        print(f"\n   {kind}")
        print(header)
        for name in scorers:
            bucket = per_kind[name][kind]
            if not bucket["mrr"]:
                continue
            summary.setdefault(name, {})[kind] = {
                key: float(np.mean(values)) for key, values in bucket.items()
            }
            print(
                f"   {name:21s} {len(bucket['mrr']):5d} "
                f"{np.mean(bucket['r3']):6.3f} {np.mean(bucket['r5']):6.3f} "
                f"{np.mean(bucket['mrr']):6.3f}"
            )

    if archetype["model"]["mrr"]:
        options = int(np.median(archetype["model"]["options"]))
        print(
            f"\n   Internship role archetype only, ignoring which industry "
            f"({options} archetypes open)"
        )
        print(f"{'':24s} {'n':>5s} {'r@3':>6s} {'':>6s} {'MRR':>6s}")
        for name in scorers:
            bucket = archetype[name]
            summary.setdefault(name, {})["archetype"] = {
                "r3": float(np.mean(bucket["r3"])),
                "mrr": float(np.mean(bucket["mrr"])),
            }
            print(
                f"   {name:21s} {len(bucket['mrr']):5d} "
                f"{np.mean(bucket['r3']):6.3f} {'':6s} {np.mean(bucket['mrr']):6.3f}"
            )
    return summary


if __name__ == "__main__":
    train()
