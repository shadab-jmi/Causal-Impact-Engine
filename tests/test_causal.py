"""
Unit tests for Part B (src/causal.py): a naïve estimate on biased data is wrong,
and IPW/PSM recover the RCT truth
"""

import numpy as np
import pytest

from src import data_prep as dp
from src import causal

OUTCOME = "conversion"


# Load the RCT, its true effect, a biased subsample, and its propensity scores
@pytest.fixture(scope="module")
def biased_setup():
    df = dp.add_treatment_flag(dp.load_raw())
    truth = causal.naive_effect(df, OUTCOME)["estimate"]      # unbiased on randomized data
    biased = causal.inject_selection_bias(df, seed=42)
    ps, _ = causal.fit_propensity(biased)
    return {"df": df, "truth": truth, "biased": biased, "ps": ps}


def test_injecting_bias_shrinks_sample_and_breaks_balance(biased_setup):
    df, biased = biased_setup["df"], biased_setup["biased"]
    assert len(biased) < len(df)
    max_smd_rct = dp.balance_table(df)["abs_smd"].max()
    max_smd_biased = dp.balance_table(biased)["abs_smd"].max()
    assert max_smd_rct < 0.1
    assert max_smd_biased > 0.3
    assert max_smd_biased > max_smd_rct


def test_naive_estimate_overstates_the_truth(biased_setup):
    truth = biased_setup["truth"]
    naive = causal.naive_effect(biased_setup["biased"], OUTCOME)["estimate"]
    assert naive > truth
    assert (naive - truth) > 0.001


def test_ipw_recovers_rct_effect(biased_setup):
    truth, biased, ps = biased_setup["truth"], biased_setup["biased"], biased_setup["ps"]
    naive = causal.naive_effect(biased, OUTCOME)["estimate"]
    ipw = causal.ipw_ate(biased, OUTCOME, ps)
    assert abs(ipw - truth) < abs(naive - truth)
    assert abs(ipw - truth) < 0.0015


def test_psm_recovers_rct_effect_and_matches_almost_everyone(biased_setup):
    truth, biased, ps = biased_setup["truth"], biased_setup["biased"], biased_setup["ps"]
    naive = causal.naive_effect(biased, OUTCOME)["estimate"]
    res = causal.psm_att(biased, OUTCOME, ps)
    assert res["matched_fraction"] > 0.9
    assert abs(res["att"] - truth) < abs(naive - truth)
    assert abs(res["att"] - truth) < 0.0015


def test_matching_restores_covariate_balance(biased_setup):
    biased, ps = biased_setup["biased"], biased_setup["ps"]
    res = causal.psm_att(biased, OUTCOME, ps)
    matched = causal.matched_sample(biased, res)
    assert dp.balance_table(matched)["abs_smd"].max() < 0.1


def test_propensity_scores_are_clipped(biased_setup):
    ps = biased_setup["ps"]
    assert ps.min() >= 0.02 - 1e-9
    assert ps.max() <= 0.98 + 1e-9
