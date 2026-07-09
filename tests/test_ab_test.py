# Unit tests for Part A (src/ab_test.py

import numpy as np
import pytest

from src import ab_test as ab


def test_srm_not_detected_on_balanced_split():
    res = ab.srm_check(5000, 5000, expected_treat_share=0.5)
    assert not res["srm_detected"]
    assert res["p_value"] > 0.05


def test_srm_not_detected_on_matching_two_thirds():
    res = ab.srm_check(20000, 10000, expected_treat_share=2 / 3)
    assert not res["srm_detected"]


def test_srm_detected_on_real_mismatch():
    res = ab.srm_check(5000, 5000, expected_treat_share=2 / 3)
    assert res["srm_detected"]
    assert res["p_value"] < 0.001


def test_ztest_recovers_known_proportions():
    res = ab.two_proportion_ztest(count_treat=1200, n_treat=10000,
                                  count_control=1000, n_control=10000)
    assert res.p_control == pytest.approx(0.10, abs=1e-9)
    assert res.p_treat == pytest.approx(0.12, abs=1e-9)
    assert res.abs_lift == pytest.approx(0.02, abs=1e-9)
    assert res.rel_lift == pytest.approx(0.20, abs=1e-9)
    assert res.p_value < 0.001
    assert res.ci_low < 0.02 < res.ci_high


def test_ztest_no_effect_is_not_significant():
    res = ab.two_proportion_ztest(1000, 10000, 1000, 10000)
    assert res.abs_lift == pytest.approx(0.0, abs=1e-9)
    assert res.p_value > 0.05
    assert res.ci_low < 0 < res.ci_high


def test_smaller_effect_needs_larger_sample():
    n_small_effect = ab.required_sample_size(0.10, 0.11)
    n_big_effect = ab.required_sample_size(0.10, 0.15)
    assert n_small_effect > n_big_effect


def test_power_increases_with_sample_size():
    low = ab.power_at_sample_size(0.10, 0.12, n_per_group=500)
    high = ab.power_at_sample_size(0.10, 0.12, n_per_group=5000)
    assert 0 < low < high <= 1.0


def test_cohens_h_zero_when_equal():
    assert ab.cohens_h(0.2, 0.2) == pytest.approx(0.0, abs=1e-12)


def test_welch_recovers_mean_difference():
    rng = np.random.default_rng(0)
    treat = rng.normal(6.0, 2.0, size=4000)
    control = rng.normal(5.0, 2.0, size=4000)
    res = ab.welch_ttest(treat, control)
    assert res.diff == pytest.approx(1.0, abs=0.15)
    assert res.p_value < 0.001
    assert res.ci_low < res.diff < res.ci_high


def test_welch_no_difference_not_significant():
    rng = np.random.default_rng(1)
    treat = rng.normal(5.0, 2.0, size=4000)
    control = rng.normal(5.0, 2.0, size=4000)
    res = ab.welch_ttest(treat, control)
    assert res.p_value > 0.05


def test_bonferroni_scales_and_caps():
    out = ab.correct_pvalues([0.01, 0.04, 0.5], method="bonferroni")
    assert out["p_adjusted"][0] == pytest.approx(0.03, abs=1e-9)
    assert out["p_adjusted"][2] == pytest.approx(1.0)
    assert out["reject_h0"][0]


def test_business_impact_arithmetic():
    out = ab.business_impact(spend_ate=0.60, n_targeted=1000, margin=0.30, cost_per_send=0.10)
    assert out["incremental_margin_per_customer"] == pytest.approx(0.18)
    assert out["net_profit_per_customer"] == pytest.approx(0.08)
    assert out["roi"] == pytest.approx(0.80)
    assert out["total_net_profit"] == pytest.approx(80.0)