"""
Part A — A/B test statistics: SRM, power/sample size, z-test, Welch's t-test,
effect sizes, multiple-testing correction, and the dollar translation
"""

from dataclasses import dataclass, asdict

import numpy as np
from scipy import stats
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import (
    proportions_ztest,
    confint_proportions_2indep,
    proportion_effectsize,
)
from statsmodels.stats.multitest import multipletests


# Chi-square goodness-of-fit test for sample ratio mismatch (p < 0.001 = broken split
def srm_check(n_treat: int, n_control: int, expected_treat_share: float = 2 / 3) -> dict:
    total = n_treat + n_control
    observed = [n_treat, n_control]
    expected = [total * expected_treat_share, total * (1 - expected_treat_share)]
    chi2, p = stats.chisquare(f_obs=observed, f_exp=expected)
    return {
        "observed": observed,
        "expected": [round(e, 1) for e in expected],
        "chi2": chi2,
        "p_value": p,
        "srm_detected": p < 0.001,
    }


# Cohen's h effect size for a difference between two proportions
def cohens_h(p1: float, p2: float) -> float:
    return 2 * np.arcsin(np.sqrt(p1)) - 2 * np.arcsin(np.sqrt(p2))


# Sample size per group needed to detect a control->treat proportion change
def required_sample_size(p_control: float, p_treat: float,
                         power: float = 0.8, alpha: float = 0.05) -> float:
    h = proportion_effectsize(p_treat, p_control)
    analysis = NormalIndPower()
    return analysis.solve_power(effect_size=abs(h), alpha=alpha, power=power,
                                ratio=1.0, alternative="two-sided")


# Statistical power to detect a control->treat change given n per group
def power_at_sample_size(p_control: float, p_treat: float, n_per_group: float,
                         alpha: float = 0.05) -> float:
    h = proportion_effectsize(p_treat, p_control)
    analysis = NormalIndPower()
    return analysis.solve_power(effect_size=abs(h), nobs1=n_per_group, alpha=alpha,
                                power=None, ratio=1.0, alternative="two-sided")


@dataclass
class ProportionResult:
    p_control: float
    p_treat: float
    abs_lift: float      # p_treat - p_control
    rel_lift: float      # abs_lift / p_control
    z_stat: float
    p_value: float
    ci_low: float        # 95% CI for abs_lift
    ci_high: float
    cohens_h: float

    def as_dict(self):
        return asdict(self)


# Two-proportion z-test for a binary outcome, with the 95% CI of the lift
def two_proportion_ztest(count_treat: int, n_treat: int,
                         count_control: int, n_control: int,
                         alpha: float = 0.05) -> ProportionResult:
    p_treat = count_treat / n_treat
    p_control = count_control / n_control
    abs_lift = p_treat - p_control
    rel_lift = abs_lift / p_control if p_control > 0 else np.nan

    z_stat, p_value = proportions_ztest([count_treat, count_control], [n_treat, n_control])
    ci_low, ci_high = confint_proportions_2indep(
        count_treat, n_treat, count_control, n_control,
        compare="diff", method="wald", alpha=alpha,
    )
    return ProportionResult(
        p_control=p_control, p_treat=p_treat, abs_lift=abs_lift, rel_lift=rel_lift,
        z_stat=z_stat, p_value=p_value, ci_low=ci_low, ci_high=ci_high,
        cohens_h=cohens_h(p_treat, p_control),
    )


@dataclass
class MeanResult:
    mean_control: float
    mean_treat: float
    diff: float
    t_stat: float
    p_value: float
    ci_low: float
    ci_high: float
    cohens_d: float

    def as_dict(self):
        return asdict(self)


# Welch's t-test for a continuous outcome (unequal variances)
def welch_ttest(treat_values, control_values, alpha: float = 0.05) -> MeanResult:
    a = np.asarray(treat_values, dtype=float)
    b = np.asarray(control_values, dtype=float)
    na, nb = len(a), len(b)
    va, vb = a.var(ddof=1), b.var(ddof=1)

    diff = a.mean() - b.mean()
    t_stat, p_value = stats.ttest_ind(a, b, equal_var=False)

    se = np.sqrt(va / na + vb / nb)
    dof = (va / na + vb / nb) ** 2 / ((va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1))
    tcrit = stats.t.ppf(1 - alpha / 2, dof)

    pooled_sd = np.sqrt((va + vb) / 2)
    cohens_d = diff / pooled_sd if pooled_sd > 0 else np.nan

    return MeanResult(
        mean_control=b.mean(), mean_treat=a.mean(), diff=diff,
        t_stat=t_stat, p_value=p_value,
        ci_low=diff - tcrit * se, ci_high=diff + tcrit * se,
        cohens_d=cohens_d,
    )


# Adjust p-values for multiple testing ('bonferroni' or 'fdr_bh')
def correct_pvalues(pvalues, method: str = "bonferroni") -> dict:
    reject, p_adj, _, _ = multipletests(pvalues, alpha=0.05, method=method)
    return {"p_adjusted": list(p_adj), "reject_h0": list(reject)}


# Translate the causal spend lift into net profit and ROI
def business_impact(spend_ate: float, n_targeted: int,
                    margin: float = 0.30, cost_per_send: float = 0.10) -> dict:
    margin_per_customer = margin * spend_ate
    net_per_customer = margin_per_customer - cost_per_send
    return {
        "incremental_spend_per_customer": spend_ate,
        "incremental_margin_per_customer": margin_per_customer,
        "cost_per_customer": cost_per_send,
        "net_profit_per_customer": net_per_customer,
        "roi": net_per_customer / cost_per_send,
        "n_targeted": n_targeted,
        "total_net_profit": net_per_customer * n_targeted,
    }
