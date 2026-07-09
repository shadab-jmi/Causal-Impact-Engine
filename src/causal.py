"""
Part B — recovering a causal effect from biased observational data via
propensity-score matching (PSM), inverse-propensity weighting (IPW), and DoWhy
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors

from src import data_prep as dp

"""
Build an observational subsample where treatment correlates with `history`.

Treated customers are kept more often when history is high, control customers when
history is low, so a naive comparison overstates the offer's effect.
"""
def inject_selection_bias(df: pd.DataFrame, seed: int = 42,
                          lo: float = 0.15, hi: float = 0.85) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    hist_rank = df["history"].rank(pct=True).values
    keep_prob = np.where(
        df["treatment"].values == 1,
        lo + (hi - lo) * hist_rank,   # treated: rises with history
        hi - (hi - lo) * hist_rank,   # control: falls with history
    )
    keep = rng.random(len(df)) < keep_prob
    return df[keep].reset_index(drop=True)


# Plain treated-minus-control difference in means, with a normal-approx 95% CI
def naive_effect(df: pd.DataFrame, outcome: str) -> dict:
    t = df.loc[df.treatment == 1, outcome].values
    c = df.loc[df.treatment == 0, outcome].values
    diff = t.mean() - c.mean()
    se = np.sqrt(t.var(ddof=1) / len(t) + c.var(ddof=1) / len(c))
    return {"estimate": diff, "se": se, "ci_low": diff - 1.96 * se, "ci_high": diff + 1.96 * se}


"""
Estimate e(x) = P(treated | features) with logistic regression. Returns (scores, model).
Scores are clipped away from 0/1 so IPW weights stay finite.
"""
def fit_propensity(df: pd.DataFrame, clip: tuple = (0.02, 0.98)):
    X = dp.encode_features(df)
    model = LogisticRegression(max_iter=2000)
    model.fit(X, df["treatment"])
    ps = model.predict_proba(X)[:, 1]
    if clip is not None:
        ps = np.clip(ps, clip[0], clip[1])
    return ps, model


# Inverse-propensity-weighted ATE (normalized / Hajek estimator)
def ipw_ate(df: pd.DataFrame, outcome: str, ps=None) -> float:
    if ps is None:
        ps, _ = fit_propensity(df)
    T = df["treatment"].values
    y = df[outcome].values
    w_treat = 1.0 / ps[T == 1]
    w_ctrl = 1.0 / (1.0 - ps[T == 0])
    return np.average(y[T == 1], weights=w_treat) - np.average(y[T == 0], weights=w_ctrl)


"""
Propensity-score matching (ATT): nearest control per treated unit on the logit scale.

Uses a caliper of `caliper_sd` * SD(logit); treated units with no control inside the
caliper are dropped.
"""
def psm_att(df: pd.DataFrame, outcome: str, ps=None, caliper_sd: float = 0.2) -> dict:
    if ps is None:
        ps, _ = fit_propensity(df)
    logit = np.log(ps / (1 - ps))
    caliper = caliper_sd * logit.std()

    T = df["treatment"].values
    y = df[outcome].values
    treated_pos = np.where(T == 1)[0]
    control_pos = np.where(T == 0)[0]

    nn = NearestNeighbors(n_neighbors=1).fit(logit[control_pos].reshape(-1, 1))
    dist, idx = nn.kneighbors(logit[treated_pos].reshape(-1, 1))
    ok = dist.ravel() <= caliper

    treated_idx = treated_pos[ok]
    control_idx = control_pos[idx.ravel()[ok]]
    att = (y[treated_idx] - y[control_idx]).mean()
    return {
        "att": att,
        "matched_fraction": ok.mean(),
        "n_matched": int(ok.sum()),
        "treated_idx": treated_idx,
        "control_idx": control_idx,
    }


# Rebuild the matched dataset (matched treated + their controls) for a balance re-check
def matched_sample(df: pd.DataFrame, psm_result: dict) -> pd.DataFrame:
    rows = np.concatenate([psm_result["treated_idx"], psm_result["control_idx"]])
    return df.iloc[rows].reset_index(drop=True)


# Resample rows with replacement, refit propensity, and recompute all three estimators.
def bootstrap_estimates(df: pd.DataFrame, outcome: str, n_boot: int = 200, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    n = len(df)
    naive, psm, ipw = [], [], []
    for _ in range(n_boot):
        s = df.iloc[rng.integers(0, n, n)].reset_index(drop=True)
        ps, _ = fit_propensity(s)
        naive.append(naive_effect(s, outcome)["estimate"])
        ipw.append(ipw_ate(s, outcome, ps))
        psm.append(psm_att(s, outcome, ps)["att"])
    return {"naive": np.array(naive), "psm": np.array(psm), "ipw": np.array(ipw)}


# Run DoWhy's identify -> estimate -> refute workflow (placebo, random common cause, subset)
def dowhy_analysis(df: pd.DataFrame, outcome: str,
                   method: str = "backdoor.linear_regression",
                   n_sim: int = 100, seed: int = 0) -> dict:
    import logging
    logging.getLogger("dowhy").setLevel(logging.ERROR)
    from dowhy import CausalModel

    X = dp.encode_features(df)
    mdf = X.copy()
    mdf["treatment"] = df["treatment"].values
    mdf[outcome] = df[outcome].values
    confounders = list(X.columns)

    model = CausalModel(data=mdf, treatment="treatment", outcome=outcome, common_causes=confounders)
    estimand = model.identify_effect(proceed_when_unidentifiable=True)
    estimate = model.estimate_effect(estimand, method_name=method, target_units="ate")

    placebo = model.refute_estimate(estimand, estimate, method_name="placebo_treatment_refuter",
                                    placebo_type="permute", num_simulations=n_sim)
    random_cc = model.refute_estimate(estimand, estimate, method_name="random_common_cause",
                                      num_simulations=10)
    subset = model.refute_estimate(estimand, estimate, method_name="data_subset_refuter",
                                   subset_fraction=0.8, num_simulations=10)
    return {
        "estimate": float(estimate.value),
        "placebo_effect": float(placebo.new_effect),
        "placebo_p": placebo.refutation_result.get("p_value"),
        "random_cc_effect": float(random_cc.new_effect),
        "subset_effect": float(subset.new_effect),
    }


# Draw the assumed causal DAG: confounders X -> T, X -> Y, and the T -> Y effect
def draw_causal_dag(save_path=None):
    import matplotlib.pyplot as plt
    import networkx as nx

    X_LABEL = "Confounders X\n(recency, history,\nzip, channel, ...)"
    T_LABEL = "Treatment T\n(offer)"
    Y_LABEL = "Outcome Y\n(conversion)"

    G = nx.DiGraph()
    G.add_edges_from([(X_LABEL, T_LABEL), (X_LABEL, Y_LABEL), (T_LABEL, Y_LABEL)])
    pos = {X_LABEL: (0, 1), T_LABEL: (-1, 0), Y_LABEL: (1, 0)}

    fig, ax = plt.subplots(figsize=(7, 5))
    nx.draw_networkx_nodes(G, pos, node_color="#e9f5f3", edgecolors="#2a9d8f",
                           node_size=6500, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=9, ax=ax)
    nx.draw_networkx_edges(G, pos, arrowsize=22, edge_color="#264653",
                           node_size=6500, width=2, ax=ax)
    ax.text(0, 0.02, "backdoor path  T <- X -> Y  (block by adjusting for X)",
            ha="center", fontsize=8, color="#e76f51")
    ax.set_axis_off()
    ax.set_title("Causal DAG — the offer's effect on conversion")
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
    return fig
