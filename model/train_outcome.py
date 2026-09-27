"""Model B: where does a student end up, and what will they earn?

This is deliberately *not* conditioned on a goal. It learns the whole outcome
distribution from a state, and the planner then reads the student's particular
goal off that distribution. Training against (state, goal) pairs instead would
mean inventing goals for people who never stated one, and would tie the model to
one notion of what a goal is.

Three heads share the same features:
  family    - which job family the person eventually works in, plus the honest
              non-employment outcomes (continuing education, still seeking,
              military), because a model that has never seen a path that failed
              cannot warn anyone off one
  industry  - which industry employs them
  salary    - quantiles of first-job salary in 2026 dollars, so the output is a
              range with real uncertainty rather than a single fabricated number
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.metrics import f1_score, mean_absolute_error

from . import data, datasets
from . import state as state_module

MODEL_PATH = data.ARTIFACT_DIR / "outcome_model.pkl"
QUANTILES = (0.1, 0.5, 0.9)

# The spread between the 10th and 90th percentile of a normal distribution is
# this many standard deviations, used to turn the quantile fit into a tail
# probability for "will I clear my target salary?".
P10_TO_P90_SIGMAS = 2.0 * norm.ppf(0.9)

CLASSIFIER_PARAMS = dict(
    objective="multiclass",
    learning_rate=0.05,
    num_leaves=31,
    min_child_samples=40,
    subsample=0.9,
    subsample_freq=1,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    n_estimators=800,
    verbose=-1,
)

REGRESSOR_PARAMS = dict(
    objective="quantile",
    learning_rate=0.05,
    num_leaves=31,
    min_child_samples=40,
    subsample=0.9,
    subsample_freq=1,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    n_estimators=800,
    verbose=-1,
)

GOAL_WEIGHTS = {"family": 0.45, "industry": 0.25, "salary": 0.30}


@dataclass
class OutcomeModel:
    family: lgb.LGBMClassifier
    industry: lgb.LGBMClassifier
    salary: dict[float, lgb.LGBMRegressor]
    features: list[str]
    categories: dict[str, list]

    def _matrix(self, frame: pd.DataFrame) -> pd.DataFrame:
        aligned = datasets.align_categories(frame, self.categories)
        return aligned[self.features]

    def family_distribution(self, frame: pd.DataFrame) -> pd.DataFrame:
        matrix = self._matrix(frame)
        return pd.DataFrame(
            self.family.predict_proba(matrix), columns=list(self.family.classes_)
        )

    def industry_distribution(self, frame: pd.DataFrame) -> pd.DataFrame:
        matrix = self._matrix(frame)
        return pd.DataFrame(
            self.industry.predict_proba(matrix), columns=list(self.industry.classes_)
        )

    def salary_quantiles(self, frame: pd.DataFrame) -> pd.DataFrame:
        matrix = self._matrix(frame)
        return pd.DataFrame(
            {
                quantile: np.exp(model.predict(matrix))
                for quantile, model in sorted(self.salary.items())
            }
        )

    def probability_salary_at_least(
        self, frame: pd.DataFrame, target_usd: float
    ) -> np.ndarray:
        """Turn the quantile fit into P(salary >= target).

        The three quantiles are summarised as a lognormal: the median sets the
        centre and the 10-90 spread sets the width. That is an approximation, but
        it is an approximation of a measured spread rather than an assumed one.
        """
        quantiles = self.salary_quantiles(frame)
        median = np.log(np.clip(quantiles[0.5].to_numpy(), 1.0, None))
        low = np.log(np.clip(quantiles[0.1].to_numpy(), 1.0, None))
        high = np.log(np.clip(quantiles[0.9].to_numpy(), 1.0, None))
        sigma = np.clip((high - low) / P10_TO_P90_SIGMAS, 1e-3, None)
        return 1.0 - norm.cdf((np.log(max(target_usd, 1.0)) - median) / sigma)

    def goal_score(self, frame: pd.DataFrame, goal: state_module.Goal) -> np.ndarray:
        """How close a state is to producing the student's stated goal.

        A weighted blend of the three heads. Whichever parts of the goal the
        student left unspecified drop out and the remaining weights are
        renormalised, so a goal of "any job paying 120k" scores sensibly.
        """
        total = np.zeros(len(frame))
        weight_used = 0.0

        families = self.family_distribution(frame)
        if goal.job_family in families.columns:
            total += GOAL_WEIGHTS["family"] * families[goal.job_family].to_numpy()
            weight_used += GOAL_WEIGHTS["family"]

        if goal.industry:
            industries = self.industry_distribution(frame)
            if goal.industry in industries.columns:
                total += GOAL_WEIGHTS["industry"] * industries[goal.industry].to_numpy()
                weight_used += GOAL_WEIGHTS["industry"]

        if goal.salary_usd:
            total += GOAL_WEIGHTS["salary"] * self.probability_salary_at_least(
                frame, float(goal.salary_usd)
            )
            weight_used += GOAL_WEIGHTS["salary"]

        if weight_used <= 0:
            return total
        return total / weight_used

    def score_state(self, state: state_module.State, goal: state_module.Goal) -> float:
        frame = datasets.to_frame([state_module.state_features(state)])
        return float(self.goal_score(frame, goal)[0])

    def save(self, path=MODEL_PATH) -> None:
        # Store the parts, not the instance. Pickling `self` records the class by
        # the module it was defined in, which is `__main__` when training is run
        # as a script, and then nothing else can load it.
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as handle:
            pickle.dump(
                {
                    "family": self.family,
                    "industry": self.industry,
                    "salary": self.salary,
                    "features": self.features,
                    "categories": self.categories,
                },
                handle,
            )

    @staticmethod
    def load(path=MODEL_PATH) -> "OutcomeModel":
        with open(path, "rb") as handle:
            parts = pickle.load(handle)
        return OutcomeModel(**parts)


def _fit_classifier(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    features: list[str],
    target: str,
) -> lgb.LGBMClassifier:
    model = lgb.LGBMClassifier(**CLASSIFIER_PARAMS)
    model.fit(
        train[features],
        train[target],
        eval_set=[(validation[features], validation[target])],
        eval_metric="multi_logloss",
        callbacks=[lgb.early_stopping(60, verbose=False)],
    )
    return model


def train(rebuild: bool = False) -> OutcomeModel:
    frame = datasets.outcome_dataset(rebuild=rebuild)
    features = datasets.feature_columns(frame)
    categories = datasets.category_levels(frame)

    train_rows = frame[frame["split"] == "train"]
    validation_rows = frame[frame["split"] == "validation"]
    test_rows = frame[frame["split"] == "test"]
    print(
        f"snapshots: {len(train_rows)} train / {len(validation_rows)} validation / "
        f"{len(test_rows)} test, {len(features)} features"
    )

    family = _fit_classifier(train_rows, validation_rows, features, "outcome_family")
    industry = _fit_classifier(train_rows, validation_rows, features, "outcome_industry")

    paid = frame.dropna(subset=["salary_real"]).copy()
    paid["log_salary"] = np.log(paid["salary_real"])
    paid_train = paid[paid["split"] == "train"]
    paid_validation = paid[paid["split"] == "validation"]
    paid_test = paid[paid["split"] == "test"]

    salary_models: dict[float, lgb.LGBMRegressor] = {}
    for quantile in QUANTILES:
        regressor = lgb.LGBMRegressor(alpha=quantile, **REGRESSOR_PARAMS)
        regressor.fit(
            paid_train[features],
            paid_train["log_salary"],
            eval_set=[(paid_validation[features], paid_validation["log_salary"])],
            callbacks=[lgb.early_stopping(60, verbose=False)],
        )
        salary_models[quantile] = regressor

    model = OutcomeModel(
        family=family,
        industry=industry,
        salary=salary_models,
        features=features,
        categories=categories,
    )

    _report(model, test_rows, paid_test)
    model.save()
    print(f"\nsaved {MODEL_PATH.relative_to(data.ROOT_DIR)}")
    return model


def _report(model: OutcomeModel, test_rows: pd.DataFrame, paid_test: pd.DataFrame) -> None:
    print("\n-- held-out cohorts (graduated 2025-2026) --")

    for head, target in (("family", "outcome_family"), ("industry", "outcome_industry")):
        classifier = getattr(model, head)
        predicted = classifier.predict(test_rows[model.features])
        truth = test_rows[target]
        accuracy = float((predicted == truth).mean())
        macro = f1_score(truth, predicted, average="macro", zero_division=0)
        probabilities = classifier.predict_proba(test_rows[model.features])
        classes = list(classifier.classes_)
        ranks = np.argsort(-probabilities, axis=1)
        truth_positions = np.array(
            [classes.index(value) if value in classes else -1 for value in truth]
        )
        top3 = float(
            np.mean([truth_positions[i] in ranks[i, :3] for i in range(len(truth))])
        )
        print(
            f"{head:9s} top-1 {accuracy:.3f}  top-3 {top3:.3f}  "
            f"macro-F1 {macro:.3f}  ({len(classes)} classes)"
        )

    quantiles = model.salary_quantiles(paid_test)
    truth = paid_test["salary_real"].to_numpy()
    median = quantiles[0.5].to_numpy()
    coverage = float(
        np.mean((truth >= quantiles[0.1].to_numpy()) & (truth <= quantiles[0.9].to_numpy()))
    )
    baseline = np.full_like(truth, float(paid_test["salary_real"].median()))
    print(
        f"salary    MAE ${mean_absolute_error(truth, median):,.0f} "
        f"(median-guess baseline ${mean_absolute_error(truth, baseline):,.0f})"
    )
    print(
        f"          p10-p90 band covers {coverage:.1%} of held-out salaries "
        f"(nominal 80%)"
    )

    # An overall average hides the shape of the problem. A freshman's eventual
    # job family is close to unknowable; a senior who has done two internships is
    # not. Reporting by degree stage says where the model is worth trusting.
    print("\n-- family accuracy by how far through the degree the snapshot is --")
    majority = test_rows["outcome_family"].value_counts(normalize=True).iloc[0]
    buckets = pd.cut(
        test_rows["progress"], [-0.01, 0.25, 0.5, 0.75, 1.01],
        labels=["first quarter", "second quarter", "third quarter", "final quarter"],
    )
    predicted = model.family.predict(test_rows[model.features])
    correct = pd.Series(predicted == test_rows["outcome_family"].to_numpy())
    correct.index = test_rows.index
    for bucket, group in correct.groupby(buckets, observed=True):
        print(f"  {str(bucket):16s} top-1 {group.mean():.3f}  (n={len(group)})")
    print(f"  {'majority class':16s} top-1 {majority:.3f}")


if __name__ == "__main__":
    train()
