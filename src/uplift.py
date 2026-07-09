"""
Part C — uplift modeling: estimating the per-customer treatment effect (CATE) to
decide who to target, with T/S-learners, Qini/uplift metrics, and the four segments
"""

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklift.models import SoloModel, TwoModels
from sklift.metrics import qini_auc_score, uplift_auc_score, uplift_curve

from src import data_prep as dp
from src import ab_test as ab


# Train/test split stratified by treatment to preserve the group ratio
def split_train_test(df, seed: int = 42, test_size: float = 0.3):
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=seed,
                                         stratify=df["treatment"])
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def _xyt(df, outcome):
    return dp.encode_features(df).values, df[outcome].values, df["treatment"].values


# T-learner (two models): separate outcome models for treated and control
def fit_t_learner(train_df, outcome, base=None):
    base = base or LogisticRegression(max_iter=2000)
    model = TwoModels(clone(base), clone(base), method="vanilla")
    X, y, t = _xyt(train_df, outcome)
    model.fit(X, y, t)
    return model


# S-learner (single model with treatment as a feature)
def fit_s_learner(train_df, outcome, base=None):
    base = base or LogisticRegression(max_iter=2000)
    model = SoloModel(base)
    X, y, t = _xyt(train_df, outcome)
    model.fit(X, y, t)
    return model


# Predicted per-customer uplift on the test set
def predict_uplift(model, test_df, outcome):
    X, _, _ = _xyt(test_df, outcome)
    return model.predict(X)


# Recover the control- and treated-response probabilities from a fitted T-learner
def t_learner_p0_p1(t_model, test_df, outcome):
    X, _, _ = _xyt(test_df, outcome)
    p1 = t_model.estimator_trmnt.predict_proba(X)[:, 1]
    p0 = t_model.estimator_ctrl.predict_proba(X)[:, 1]
    return p0, p1


# Qini AUC and uplift AUC for a predicted-uplift ranking
def evaluate(test_df, uplift, outcome) -> dict:
    _, y, t = _xyt(test_df, outcome)
    return {"qini_auc": qini_auc_score(y, uplift, t),
            "uplift_auc": uplift_auc_score(y, uplift, t)}


# Share of total incremental outcome captured by targeting the top-ranked customers
def capture_fractions(test_df, score, outcome, fracs=(0.1, 0.2, 0.3, 0.4, 0.5)) -> dict:
    _, y, t = _xyt(test_df, outcome)
    x_uc, y_uc = uplift_curve(y, score, t)
    total = y_uc[-1]
    return {f: (y_uc[int(f * len(x_uc))] / total if total else np.nan) for f in fracs}


"""
The two population cut points that define the four segments.
`tau` is the median of the positive predicted uplifts (the persuadable cutoff);
`p0_median` is the median control-response probability (splits sure-things from
lost-causes among the low-uplift group). Exposing them lets a single customer be
labelled with the exact same rule used across the whole population.
"""
def segment_thresholds(p0, p1):
    uplift = p1 - p0
    positive = uplift[uplift > 0]
    tau = float(np.quantile(positive, 0.5)) if len(positive) else 0.0
    return tau, float(np.median(p0))


# Assign each customer to a segment given fixed population thresholds (vectorised)
def classify_segment(p0, p1, tau, p0_median):
    uplift = p1 - p0
    return np.where(uplift < 0, "sleeping_dog",
           np.where(uplift >= tau, "persuadable",
           np.where(p0 >= p0_median, "sure_thing", "lost_cause")))


# Split customers into persuadables / sure-things / lost-causes / sleeping-dogs
def segment_customers(p0, p1):
    tau, p0_median = segment_thresholds(p0, p1)
    return classify_segment(p0, p1, tau, p0_median)


# Segment label for one customer, using thresholds from segment_thresholds()
def segment_for(p0: float, p1: float, tau: float, p0_median: float) -> str:
    return str(classify_segment(np.array([p0]), np.array([p1]), tau, p0_median)[0])


"""
Predict (p0, p1, uplift) for a single hypothetical customer.

`customer` holds the raw columns encode_features expects (recency, history, mens,
womens, newbie, history_segment, zip_code, channel). A one-row frame is one-hot
encoded and realigned to `feature_columns` (the training matrix's columns) so the
dummy columns line up with what the T-learner was fit on. Uses the *same* fitted
model that powers the capture/profit charts — no retraining.
"""

def predict_customer(t_model, customer: dict, feature_columns) -> tuple:
    row = pd.DataFrame([customer])
    X = dp.encode_features(row).reindex(columns=feature_columns, fill_value=0.0).values
    p1 = float(t_model.estimator_trmnt.predict_proba(X)[:, 1][0])
    p0 = float(t_model.estimator_ctrl.predict_proba(X)[:, 1][0])
    return p0, p1, p1 - p0


"""
Validate the predicted sleeping dogs on held-out data via a two-proportion z-test.

Claims a real negative-uplift effect only if the treated rate is significantly lower.
"""

def sleeping_dog_holdout(test_df, uplift, outcome) -> dict:
    mask = uplift < 0
    sub = test_df[mask]
    t = sub["treatment"].values
    y = sub[outcome].values
    n_t, n_c = int((t == 1).sum()), int((t == 0).sum())
    res = ab.two_proportion_ztest(int(y[t == 1].sum()), n_t, int(y[t == 0].sum()), n_c)
    confirmed = bool(res.abs_lift < 0 and res.p_value < 0.05)
    return {
        "n_flagged": int(mask.sum()), "n_treated": n_t, "n_control": n_c,
        "treated_rate": res.p_treat, "control_rate": res.p_control,
        "lift": res.abs_lift, "p_value": res.p_value, "confirmed": confirmed,
    }


# Realized profit at each targeting fraction for one set of customers
def _profit_at_fracs(score, treatment, spend, fracs, margin, cost):
    order = np.argsort(-score)
    t = treatment[order]
    sp = spend[order]
    n = len(sp)
    profits = []
    for f in fracs:
        k = max(2, int(f * n))
        st = sp[:k][t[:k] == 1]
        sc = sp[:k][t[:k] == 0]
        inc_spend = (st.mean() - sc.mean()) if len(st) and len(sc) else 0.0
        profits.append((margin * inc_spend - cost) * k)
    return np.array(profits)


"""
Profit vs fraction targeted, ranking customers by `score`.

Spend is heavily zero-inflated and whale-dominated, so each point is a noisy difference of
subsample means. With `n_boot` set, bootstrap the evaluation (no model refit) and return a
median line with a 5-95% band.

Returns `(fracs, profits)` when `n_boot` is None, else `(fracs, median, lower, upper)`.
"""

def targeting_profit_curve(test_df, score, margin: float = 0.30, cost: float = 0.10,
                           n_points: int = 50, n_boot: int = None, seed: int = 0):
    score = np.asarray(score)
    treatment = test_df["treatment"].values
    spend = test_df["spend"].values
    fracs = np.linspace(0.02, 1.0, n_points)

    profits = _profit_at_fracs(score, treatment, spend, fracs, margin, cost)
    if n_boot is None:
        return fracs, profits

    rng = np.random.default_rng(seed)
    n = len(test_df)
    curves = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)                       # resampling customers with replacement
        curves.append(_profit_at_fracs(score[idx], treatment[idx], spend[idx], fracs, margin, cost))
    curves = np.array(curves)
    median = np.percentile(curves, 50, axis=0)
    lower = np.percentile(curves, 5, axis=0)
    upper = np.percentile(curves, 95, axis=0)
    return fracs, median, lower, upper
