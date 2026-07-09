# Three tabs: A/B test, observational causal, uplift targeting

import sys
from pathlib import Path

# Making the project root importable regardless of where Streamlit is launched from
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

from src import data_prep as dp
from src import ab_test as ab
from src import causal
from src import uplift
from src import viz

viz.apply_theme()
TEAL, DARK, ORANGE, SAND, GREY = (
    viz.ACCENT, viz.REFERENCE, viz.SERIES_WARM, viz.SAND, viz.MUTED,
)

st.set_page_config(page_title="Causal Impact & Experimentation Engine",
                   page_icon="📈", layout="wide")

_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

:root {
  --accent: #5A8DEE;
  --elevated: #212839;
  --border: rgba(255,255,255,0.06);
  --heading: #EEF1F6;
  --muted: #8B94A7;
}

/* Typography */
html, body, [data-testid="stAppViewContainer"], [class*="css"] {
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
}
h1 {
  font-weight: 800;
  letter-spacing: -0.03em;
  color: var(--heading);
  text-shadow: none;
  filter: none;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  text-rendering: optimizeLegibility;
}
h2, h3 { font-weight: 650; letter-spacing: -0.02em; color: var(--heading); }
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p { color: var(--muted); }

/* Layout rhythm */
[data-testid="stMainBlockContainer"] {
  max-width: 1180px;
  padding-top: 3rem;
  padding-bottom: 4rem;
}
[data-testid="stVerticalBlock"] { gap: 1.05rem; }
hr { border-color: var(--border) !important; margin: 0.4rem 0 1.2rem 0; }

/* Metric cards */
[data-testid="stMetric"] {
  background: var(--elevated);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 15px 18px;
  box-shadow: 0 2px 8px rgba(0,0,0,0.35), 0 1px 2px rgba(0,0,0,0.25);
}
[data-testid="stMetricLabel"],
[data-testid="stMetricLabel"] > div,
[data-testid="stMetricLabel"] p {
  color: var(--muted);
  font-size: 0.76rem;
  font-weight: 600;
  letter-spacing: 0.03em;
  text-transform: uppercase;
  white-space: normal;
  overflow: visible;
  text-overflow: unset;
}
[data-testid="stMetricValue"] {
  font-weight: 700;
  font-size: 1.85rem;
  line-height: 1.1;
  letter-spacing: -0.02em;
  font-feature-settings: "tnum" 1;
}
[data-testid="stMetricDelta"] { font-weight: 600; }

/* Verdict KPI gets accent priority */
.st-key-verdict-kpi [data-testid="stMetric"] {
  border-color: rgba(90,141,238,0.50);
  background:
    linear-gradient(180deg, rgba(90,141,238,0.12), rgba(90,141,238,0.03)),
    var(--elevated);
  box-shadow: 0 4px 16px rgba(90,141,238,0.14), 0 2px 8px rgba(0,0,0,0.35);
}

/* Segment cards use neutral grey badges, not coloured deltas */
.st-key-segment-cards [data-testid="stMetricDelta"] {
  color: var(--muted) !important;
  background: rgba(255,255,255,0.05);
  border-radius: 6px;
  padding: 1px 8px;
  display: inline-block;
  font-size: 0.72rem;
}
.st-key-segment-cards [data-testid="stMetricDelta"] svg { display: none; }

/* Numbered tab nav (01 / 02 / 03) */
[data-testid="stTabs"] [data-baseweb="tab-list"] {
  gap: 4px;
  border-bottom: 1px solid var(--border);
  counter-reset: tab;
}
[data-testid="stTabs"] button[data-baseweb="tab"] {
  height: auto;
  padding: 11px 20px;
  background: transparent;
  border-radius: 8px 8px 0 0;
  color: var(--muted);
  font-weight: 600;
}
[data-testid="stTabs"] button[data-baseweb="tab"]::before {
  counter-increment: tab;
  content: counter(tab, decimal-leading-zero);
  margin-right: 9px;
  color: var(--accent);
  opacity: 0.5;
  font-weight: 700;
  font-variant-numeric: tabular-nums;
}
[data-testid="stTabs"] button[data-baseweb="tab"]:hover { color: #C3CAD6; }
[data-testid="stTabs"] button[data-baseweb="tab"][aria-selected="true"] { color: var(--heading); }
[data-testid="stTabs"] button[data-baseweb="tab"][aria-selected="true"]::before { opacity: 1; }
[data-testid="stTabs"] [data-baseweb="tab-highlight"] { background-color: var(--accent); height: 2.5px; }
[data-testid="stTabs"] [data-baseweb="tab-border"] { background-color: transparent; }

/* Sliders: two-tone track */
[data-testid="stSlider"] [data-baseweb="slider"] [role="slider"] {
  border: 2px solid var(--accent) !important;
  box-shadow: 0 0 0 4px rgba(90,141,238,0.18) !important;
}
[data-testid="stSlider"] [data-baseweb="slider"] div:has(> [role="slider"]) { height: 6px; }

/* Selectbox: pointer cursor on hover instead of text caret */
[data-testid="stSelectbox"] [data-baseweb="select"],
[data-testid="stSelectbox"] [data-baseweb="select"] * {
  cursor: pointer;
}

/* Alerts & expanders */
[data-testid="stAlert"], [data-testid="stExpander"] {
  border-radius: 12px;
  border: 1px solid var(--border);
}
[data-testid="stExpander"] summary { font-weight: 600; }
</style>
"""
st.markdown(_CSS, unsafe_allow_html=True)

MARGIN, COST_PER_SEND = 0.30, 0.10

# Cached data & model computations (run once, reused across widget interactions)
@st.cache_data(show_spinner=False)
def load_data():
    return dp.add_treatment_flag(dp.load_raw())


@st.cache_data(show_spinner=False)
def ab_result(outcome: str) -> dict:
    df = load_data()
    t = df[df.treatment == 1]
    c = df[df.treatment == 0]
    if outcome == "spend":
        r = ab.welch_ttest(t["spend"].values, c["spend"].values)
        return {"control": r.mean_control, "treat": r.mean_treat, "lift": r.diff,
                "ci": (r.ci_low, r.ci_high), "p": r.p_value, "rel": r.diff / r.mean_control}
    r = ab.two_proportion_ztest(int(t[outcome].sum()), len(t), int(c[outcome].sum()), len(c))
    return {"control": r.p_control, "treat": r.p_treat, "lift": r.abs_lift,
            "ci": (r.ci_low, r.ci_high), "p": r.p_value, "rel": r.rel_lift}


@st.cache_data(show_spinner=False)
def group_sizes() -> tuple:
    df = load_data()
    return int((df.treatment == 1).sum()), int((df.treatment == 0).sum())


@st.cache_data(show_spinner="Injecting bias and fitting propensity model…")
def partB_estimates(outcome: str, bias_strength: float) -> dict:
    df = load_data()
    truth = causal.naive_effect(df, outcome)["estimate"]        # unbiased on the real RCT
    lo, hi = 0.5 - bias_strength, 0.5 + bias_strength
    biased = causal.inject_selection_bias(df, seed=42, lo=lo, hi=hi)
    ps, _ = causal.fit_propensity(biased)
    return {
        "truth": truth,
        "naive": causal.naive_effect(biased, outcome)["estimate"],
        "psm": causal.psm_att(biased, outcome, ps)["att"],
        "ipw": causal.ipw_ate(biased, outcome, ps),
        "n_biased": len(biased),
    }


"""
Fit the T-learner ONCE and keep the fitted model, held-out test set, and the
feature-column template. Cached as a *resource* so both the aggregate charts and the
single-customer scorer score against the exact same fitted object (no retraining)
"""
@st.cache_resource(show_spinner="Training the uplift model…")
def partC_model(outcome: str = "conversion"):
    df = load_data()
    train, test = uplift.split_train_test(df)
    model = uplift.fit_t_learner(train, outcome)
    feature_columns = list(dp.encode_features(train).columns)
    return model, test, feature_columns


@st.cache_data(show_spinner="Training the uplift model…")
def partC_data(outcome: str = "conversion") -> dict:
    model, test, _ = partC_model(outcome)
    up = uplift.predict_uplift(model, test, outcome)
    p0, p1 = uplift.t_learner_p0_p1(model, test, outcome)
    tau, p0_median = uplift.segment_thresholds(p0, p1)

    grid = np.round(np.linspace(0.05, 0.95, 19), 4)
    cap_up = uplift.capture_fractions(test, up, outcome, fracs=tuple(grid))
    cap_resp = uplift.capture_fractions(test, p1, outcome, fracs=tuple(grid))
    # Bootstrap the profit curve (evaluation only, no refit) for a median line with a 5-95% band.
    fracs_profit, profit, profit_lo, profit_hi = uplift.targeting_profit_curve(
        test, up, margin=MARGIN, cost=COST_PER_SEND, n_boot=300)

    seg = uplift.segment_customers(p0, p1)
    seg_counts = {s: int((seg == s).sum()) for s in
                  ["persuadable", "sure_thing", "lost_cause", "sleeping_dog"]}
    dogs = uplift.sleeping_dog_holdout(test, up, outcome)

    return {
        "n_test": len(test),
        "qini": uplift.evaluate(test, up, outcome)["qini_auc"],
        "tau": tau,
        "p0_median": p0_median,
        "uplift_sorted": np.sort(up),         # for percentile-ranking one customer
        "p1_sorted": np.sort(p1),             # response-model score (P[convert | emailed]), for the same
        "grid": grid,
        "cap_up": np.array([cap_up[f] for f in grid]),
        "cap_resp": np.array([cap_resp[f] for f in grid]),
        "fracs_profit": fracs_profit,
        "profit": profit,
        "profit_lower": profit_lo,
        "profit_upper": profit_hi,
        "segments": seg_counts,
        "dogs": dogs,
    }


# Header
st.title("Causal Impact & Experimentation Engine")
st.caption("Measuring the **true causal effect** of a retention offer — three ways: a clean A/B "
           "test, messy observational data, and per-customer uplift targeting.")

conv = ab_result("conversion")
spend = ab_result("spend")
net = MARGIN * spend["lift"] - COST_PER_SEND
h1, h2, h3, h4 = st.columns(4)
h1.metric("Conversion lift (causal)", f"+{conv['lift'] * 100:.2f} pp", f"{conv['rel'] * 100:+.0f}% relative")
h2.metric("Spend lift (causal)", f"+${spend['lift']:.2f}", "per customer")
h3.metric("Net profit / customer", f"${net:.3f}", f"ROI {net / COST_PER_SEND * 100:+.0f}%")
with h4.container(key="verdict-kpi"):
    st.metric("Verdict", "Roll out", "profitable for ~all")

tab_a, tab_b, tab_c = st.tabs([
    "A/B Test", "Observational", "Uplift Targeting",
])


# TAB A — plan an experiment (power) + read this experiment's result
with tab_a:
    st.subheader("Plan an experiment: how many customers do you need?")
    st.write("Statistical power is the probability of detecting a real effect. Pick a baseline rate "
             "and the smallest lift you'd care about (the **MDE**) to see the sample size required.")

    c1, c2 = st.columns([1, 1.4])
    with c1:
        base = st.slider("Baseline conversion rate (%)", 0.2, 5.0, 0.57, 0.01) / 100
        mde_rel = st.slider("Minimum detectable effect (relative %)", 5, 100, 20, 5) / 100
        alpha = st.select_slider("Significance level α", [0.10, 0.05, 0.01], value=0.05)
        target_power = st.slider("Target power", 0.5, 0.95, 0.80, 0.05)

        p_treat = base * (1 + mde_rel)
        n_req = ab.required_sample_size(base, p_treat, power=target_power, alpha=alpha)
        st.metric("Required sample size **per group**", f"{n_req:,.0f}")
        n_t, n_c = group_sizes()
        st.caption(f"This experiment ran with **{n_t:,}** treated and **{n_c:,}** control. "
                   f"A {mde_rel * 100:.0f}% lift on a {base * 100:.2f}% base needs "
                   f"{'more than we have — underpowered.' if n_req > n_c else 'fewer than we have — well powered.'}")

    with c2:
        ns = np.linspace(500, max(n_req * 1.5, 90_000), 120)
        powers = [ab.power_at_sample_size(base, p_treat, n, alpha=alpha) for n in ns]
        fig, ax = plt.subplots(figsize=(6, 3.6))
        ax.plot(ns, powers, color=TEAL, lw=2.2)
        ax.axhline(target_power, color=GREY, ls="--", lw=1)
        ax.axvline(n_req, color=ORANGE, ls="--", lw=1)
        ax.axvline(n_c, color=DARK, ls=":", lw=1.4)
        ax.text(n_req, 0.05, f"  need {n_req:,.0f}/grp", color=ORANGE, fontsize=8)
        ax.text(n_c, 0.92, f" actual {n_c:,}", color=DARK, fontsize=8, ha="right")
        ax.set_xlabel("Sample size per group"); ax.set_ylabel("Power")
        ax.set_ylim(0, 1.02); ax.set_title("Power curve")
        st.pyplot(fig, width="stretch")

    st.divider()
    st.subheader("Read this experiment: what actually happened")
    st.write("Hillstrom was a **properly randomized** experiment (we verified balance and found no "
             "sample-ratio mismatch), so the treated−control gap *is* the causal effect.")

    labels = {"conversion": "Conversion", "visit": "Site visit", "spend": "Spend ($)"}
    cols = st.columns(3)
    fig2, axes = plt.subplots(1, 3, figsize=(9, 3.0))
    for ax, (key, name) in zip(axes, labels.items()):
        r = ab_result(key)
        unit = "$" if key == "spend" else "pp"
        scale = 1 if key == "spend" else 100
        lift, lo, hi = r["lift"] * scale, r["ci"][0] * scale, r["ci"][1] * scale
        cols[list(labels).index(key)].metric(
            name,
            (f"+${lift:.2f}" if key == "spend" else f"+{lift:.3f} pp"),
            f"{r['rel'] * 100:+.0f}% · p={r['p']:.1e}",
        )
        ax.axhline(0, color=GREY, lw=1)
        ax.errorbar([0], [lift], yerr=[[lift - lo], [hi - lift]], fmt="o",
                    color=TEAL, capsize=5, lw=2, ms=8)
        ax.set_xticks([]); ax.set_title(f"{name}\n95% CI ({unit})", fontsize=9)
    fig2.tight_layout()
    st.pyplot(fig2, width="stretch")


# TAB B — can we trust observational data?
with tab_b:
    st.subheader("When you can't randomize, does a naïve comparison lie?")
    st.write("We take the same data and **deliberately bias it** — sending the offer mostly to "
             "already-loyal (high-history) customers. Then we compare a naïve treated−control "
             "gap against propensity-score matching (PSM) and inverse-propensity weighting (IPW), "
             "with the true RCT effect as the yardstick.")

    c1, c2 = st.columns([1, 1.6])
    with c1:
        outcome = st.selectbox("Outcome", ["conversion", "visit", "spend"], index=0)
        strength = st.slider("Selection-bias strength", 0.0, 0.45, 0.35, 0.05,
                             help="0 = no bias (random). 0.35 ≈ the report's setup.")
        est = partB_estimates(outcome, strength)
        scale = 1 if outcome == "spend" else 100
        unit = "$" if outcome == "spend" else "pp"
        bias_pct = (est["naive"] - est["truth"]) / est["truth"] * 100 if est["truth"] else 0
        st.metric("Naïve bias vs truth", f"{bias_pct:+.0f}%",
                  "overstates" if bias_pct > 0 else "off")
        st.caption(f"Biased subsample: **{est['n_biased']:,}** customers. The naïve gap drifts as bias "
                   "rises; PSM and IPW stay on the true effect.")

    with c2:
        methods = ["Naïve", "PSM", "IPW"]
        vals = [est["naive"] * scale, est["psm"] * scale, est["ipw"] * scale]
        colours = [ORANGE, TEAL, TEAL]
        fig, ax = plt.subplots(figsize=(6.4, 3.8))
        bars = ax.bar(methods, vals, color=colours, width=0.6)
        ax.axhline(est["truth"] * scale, color=DARK, ls="--", lw=1.6,
                   label=f"True RCT effect ({est['truth'] * scale:+.3f} {unit})")
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:+.3f}",
                    ha="center", va="bottom", fontsize=9)
        ax.set_ylabel(f"Estimated lift ({unit})")
        ax.set_title(f"Naïve vs corrected — {outcome}")
        ax.legend(loc="upper right", fontsize=8)
        st.pyplot(fig, width="stretch")

    st.info("**Takeaway:** an unadjusted observational read-out credits the offer for loyalty it "
            "didn't create. PSM/IPW remove that confounding and land back on the truth — so "
            "observational data is usable *with* proper causal adjustment, not without.")


# TAB C — who should we target?
with tab_c:
    st.subheader("Under a budget, whom do we contact first?")
    st.write("An **uplift model** ranks customers by their individual predicted effect. Choose how "
             "deep into that ranking you contact, and compare the incremental conversions captured "
             "against a conventional 'most-likely-to-convert' response model.")

    d = partC_data("conversion")
    reach = st.slider("Targeting reach — contact the top X% by predicted uplift", 5, 95, 30, 5) / 100

    cap_up = float(np.interp(reach, d["grid"], d["cap_up"]))
    cap_resp = float(np.interp(reach, d["grid"], d["cap_resp"]))
    profit_at = float(np.interp(reach, d["fracs_profit"], d["profit"]))
    n_targeted = int(reach * d["n_test"])

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Customers contacted", f"{n_targeted:,}", f"top {reach * 100:.0f}%")
    m2.metric("Conversions captured", f"{cap_up * 100:.0f}%", "by uplift ranking")
    m3.metric("vs response model", f"{cap_resp * 100:.0f}%", f"{(cap_up - cap_resp) * 100:+.0f} pp")
    m4.metric("Model quality (Qini AUC)", f"{d['qini']:.3f}", "higher = better ranking")

    c1, c2 = st.columns(2)
    with c1:
        fig, ax = plt.subplots(figsize=(5.6, 3.8))
        g = d["grid"] * 100
        ax.plot(g, d["cap_up"] * 100, color=TEAL, lw=2.2, label="Uplift targeting")
        ax.plot(g, d["cap_resp"] * 100, color=ORANGE, lw=2, label="Response model")
        ax.plot([0, 100], [0, 100], color=GREY, ls="--", lw=1, label="Random")
        ax.axvline(reach * 100, color=DARK, ls=":", lw=1.4)
        ax.set_xlabel("% of customers contacted")
        ax.set_ylabel("% of incremental conversions captured")
        ax.set_title("Capture curve"); ax.legend(fontsize=8, loc="lower right")
        st.pyplot(fig, width="stretch")
    with c2:
        fig, ax = plt.subplots(figsize=(5.6, 3.8))
        ax.fill_between(d["fracs_profit"] * 100, d["profit_lower"], d["profit_upper"],
                        color=TEAL, alpha=0.15, label="5–95% bootstrap band")
        ax.plot(d["fracs_profit"] * 100, d["profit"], color=TEAL, lw=2.2, label="median profit")
        ax.axvline(reach * 100, color=DARK, ls=":", lw=1.4)
        ax.scatter([reach * 100], [profit_at], color=ORANGE, zorder=5, s=40)
        ax.axhline(0, color=GREY, lw=1)
        ax.set_xlabel("% of customers contacted")
        ax.set_ylabel("Incremental profit on test set ($)")
        ax.set_title("Profit vs reach — broad plateau")
        ax.legend(fontsize=8, loc="lower center")
        st.pyplot(fig, width="stretch")
        st.caption("The 5–95% bootstrap band is wide and flat across the top ~60–100%, so there's no "
                   "precise optimum — the honest read is a broad plateau, so target broadly.")

    seg = d["segments"]
    with st.expander("Customer segments & the honest 'sleeping dog' check"):
        with st.container(key="segment-cards"):
            s1, s2, s3, s4 = st.columns(4)
            s1.metric("Persuadables", f"{seg['persuadable']:,}", "target these", delta_color="off")
            s2.metric("Sure things", f"{seg['sure_thing']:,}", "buy anyway", delta_color="off")
            s3.metric("Lost causes", f"{seg['lost_cause']:,}", "won't buy", delta_color="off")
            s4.metric("Sleeping dogs?", f"{seg['sleeping_dog']:,}", "flagged", delta_color="off")
        dogs = d["dogs"]
        verdict = "**confirmed**" if dogs["confirmed"] else "**not confirmed** — noise"
        st.caption(
            f"The {seg['sleeping_dog']:,} flagged 'sleeping dogs' were validated on held-out data: "
            f"treated {dogs['treated_rate'] * 100:.2f}% vs control {dogs['control_rate'] * 100:.2f}% "
            f"(lift {dogs['lift'] * 100:+.2f} pp, p = {dogs['p_value']:.2f}) → {verdict}. "
            "We only claim a segment exists if it survives this test."
        )

    st.info(f"**Takeaway:** with today's economics the offer is profitable at nearly any reach, so "
            f"**target broadly**. Uplift ranking matters **under a budget** — at the top "
            f"{reach * 100:.0f}% it captures {cap_up * 100:.0f}% of incremental conversions vs "
            f"{cap_resp * 100:.0f}% for a response model, the same spend for more incremental sales.")

    # Single-customer scorer 
    # Self-contained: reuses the cached T-learner from partC_model() — no retraining.
    # Everything to the end of this tab is one contiguous block; delete it to remove
    # the feature without touching any chart, KPI, or the segments expander above.

    st.divider()
    st.subheader("Score an individual customer")
    st.write("Enter one hypothetical customer's profile to get their **personal predicted uplift** "
             "and a targeting call — scored by the *same* T-learner that powers the charts above")

    _ZIP_RAW = {"Urban": "Urban", "Suburban": "Surburban", "Rural": "Rural"}  # dataset spells it 'Surburban'
    _HISTORY_BINS = [(100, "1) $0 - $100"), (200, "2) $100 - $200"), (350, "3) $200 - $350"),
                     (500, "4) $350 - $500"), (750, "5) $500 - $750"),
                     (1000, "6) $750 - $1,000"), (float("inf"), "7) $1,000 +")]
    _SEG_LABEL = {"persuadable": "Persuadable", "sure_thing": "Sure thing",
                  "lost_cause": "Lost cause", "sleeping_dog": "Sleeping dog"}
    _SEG_REASON = {
        "persuadable": "predicted to convert <b>because of</b> the offer — the clearest win, so target.",
        "sure_thing": "predicted to buy <b>with or without</b> the email, so the send mostly wastes offer cost.",
        "lost_cause": "unlikely to convert <b>either way</b>, so there is little incremental gain.",
        "sleeping_dog": "predicted to respond <b>less</b> if emailed — contacting may backfire, so leave alone.",
    }

    def _history_segment(h):
        return next(label for hi, label in _HISTORY_BINS if h < hi)

    model_sc, _, feat_cols = partC_model("conversion")     # same fitted object as the charts

    sc_in, sc_out = st.columns([1, 1.15])
    with sc_in:
        st.markdown("**Customer profile**")
        a, b = st.columns(2)
        recency = a.number_input("Recency (months)", 1, 12, 6, 1,
                                 help="Months since last purchase")
        history = b.number_input("History ($ last year)", 0.0, 3500.0, 150.0, 10.0,
                                 help="Total spend in the prior year")
        mens = a.toggle("Bought men's merchandise", value=True)
        womens = b.toggle("Bought women's merchandise", value=False)
        newbie = a.toggle("New customer (newbie)", value=True)
        zip_disp = b.selectbox("Zip code type", ["Urban", "Suburban", "Rural"], index=0)
        channel = a.selectbox("Acquisition channel", ["Phone", "Web", "Multichannel"], index=1)

    customer = {
        "recency": recency, "history": history,
        "mens": int(mens), "womens": int(womens), "newbie": int(newbie),
        "history_segment": _history_segment(history),
        "zip_code": _ZIP_RAW[zip_disp], "channel": channel,
    }
    p0_c, p1_c, u_c = uplift.predict_customer(model_sc, customer, feat_cols)
    seg_c = uplift.segment_for(p0_c, p1_c, d["tau"], d["p0_median"])
    pct_c = float((d["uplift_sorted"] <= u_c).mean())          # rank vs ALL customers
    cut_pct = float((d["uplift_sorted"] < d["tau"]).mean())    # percentile you must clear to be persuadable
    top_pct = (1.0 - cut_pct) * 100                            # persuadables ≈ the top N% by uplift
    do_target = seg_c == "persuadable"

    with sc_in:
        st.markdown(
            f"""
            <div style="margin-top:16px; color:var(--muted); font-size:0.85rem; line-height:1.65;">
            <ul style="padding-left:18px; margin:0;">
                <li>Uses the <b>same segment cut-offs</b> as the cards on the right.</li>
                <li><b>Persuadables</b> — the strongest <i>positive</i>-uplift customers, above the
                    median of the <b>positive</b> uplifts — are roughly the
                    <b style="color:var(--heading);">top {top_pct:.0f}%</b>
                    (the ~{cut_pct * 100:.0f}% mark), <i>not</i> simply everyone above the
                    overall median.</li>
                <li><code>history_segment</code> is derived from the $ history.</li>
            </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Response-model verdict — the "most-likely-to-convert" baseline the charts compare against.
    # It ranks by P[convert | emailed] (p1), so for an equal-budget call it targets the top
    # `top_pct%` by that score. Same fitted T-learner (its treated arm), no retraining

    resp_pct_c = float((d["p1_sorted"] <= p1_c).mean())        # this customer's p1 rank vs all
    do_target_resp = resp_pct_c >= cut_pct                     # inside the same top-N% budget
    models_disagree = do_target != do_target_resp

    with sc_out:
        st.markdown("**Prediction**")
        o1, o2, o3 = st.columns(3)
        o1.metric("If not emailed (p₀)", f"{p0_c * 100:.2f}%")
        o2.metric("If emailed (p₁)", f"{p1_c * 100:.2f}%")
        o3.metric("Predicted uplift", f"{u_c * 100:+.2f} pp")

        col = viz.POSITIVE if do_target else viz.NEGATIVE
        label = "Target" if do_target else "Don't target"
        badge_bg = "rgba(63,185,80,0.15)" if do_target else "rgba(248,81,73,0.15)"
        flag_col = viz.SERIES_WARM
        border_col = flag_col if models_disagree else col
        st.markdown(
            f"""
            <div style="background:var(--elevated); border:1px solid {border_col}44;
                        border-radius:12px; padding:15px 18px;
                        box-shadow:0 2px 8px rgba(0,0,0,0.35);">
              <div style="display:flex; align-items:center; gap:11px; margin-bottom:9px;">
                <span style="color:var(--muted); font-size:0.72rem; text-transform:uppercase;
                             letter-spacing:0.06em;">Uplift model</span>
                <span style="background:{badge_bg}; color:{col}; border:1px solid {col}66;
                             font-weight:700; font-size:0.80rem; padding:3px 13px;
                             border-radius:999px; text-transform:uppercase;
                             letter-spacing:0.04em;">{label}</span>
                <span style="color:var(--muted); font-size:0.86rem;">resembles a
                  <b style="color:var(--heading);">{_SEG_LABEL[seg_c]}</b></span>
              </div>
              <div style="color:#C3CAD6; font-size:0.92rem; line-height:1.5;">
                This customer is {_SEG_REASON[seg_c]} Their uplift beats
                <b style="color:var(--heading);">{pct_c * 100:.0f}%</b> of customers, while the
                <b>persuadable</b> bar is higher — the top
                <b style="color:var(--heading);">{top_pct:.0f}%</b> (≈ the {cut_pct * 100:.0f}% mark).
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Response-model verdict (the "most-likely-to-convert" baseline)
        r_col = viz.POSITIVE if do_target_resp else viz.NEGATIVE
        r_label = "Target" if do_target_resp else "Don't target"
        r_badge_bg = "rgba(63,185,80,0.15)" if do_target_resp else "rgba(248,81,73,0.15)"
        r_border = flag_col if models_disagree else r_col
        st.markdown(
            f"""
            <div style="background:var(--elevated); border:1px solid {r_border}44;
                        border-radius:12px; padding:15px 18px; margin-top:11px;
                        box-shadow:0 2px 8px rgba(0,0,0,0.35);">
              <div style="display:flex; align-items:center; gap:11px; margin-bottom:9px;">
                <span style="color:var(--muted); font-size:0.72rem; text-transform:uppercase;
                             letter-spacing:0.06em;">Response model</span>
                <span style="background:{r_badge_bg}; color:{r_col}; border:1px solid {r_col}66;
                             font-weight:700; font-size:0.80rem; padding:3px 13px;
                             border-radius:999px; text-transform:uppercase;
                             letter-spacing:0.04em;">{r_label}</span>
                <span style="color:var(--muted); font-size:0.86rem;">P(convert&nbsp;|&nbsp;emailed)
                  <b style="color:var(--heading);">{p1_c * 100:.2f}%</b></span>
              </div>
              <div style="color:#C3CAD6; font-size:0.92rem; line-height:1.5;">
                Targets whoever is most likely to <b>convert</b> — this customer's conversion
                chance beats <b style="color:var(--heading);">{resp_pct_c * 100:.0f}%</b> of
                customers, {"inside" if do_target_resp else "below"} the same top-<b
                style="color:var(--heading);">{top_pct:.0f}%</b> budget.
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Disagreement flag — the single most useful moment on the page
        if models_disagree:
            st.markdown(
                f"""
                <div style="background:rgba(232,164,76,0.12); border:1px solid {flag_col}55;
                            border-radius:12px; padding:12px 16px; margin-top:11px;">
                  <span style="color:{flag_col}; font-weight:700; font-size:0.80rem;
                               text-transform:uppercase; letter-spacing:0.04em;">⚠ Models disagree</span>
                  <div style="color:#C3CAD6; font-size:0.90rem; line-height:1.5; margin-top:6px;">
                    The response model targets anyone likely to buy, even if they'd buy anyway — the
                    uplift model only targets people the offer actually persuades. For this customer
                    those calls part ways.
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

st.divider()
st.caption("Built from the `src/` modules that power notebooks 01–05. Assumptions: "
           f"{MARGIN:.0%} margin on spend, ${COST_PER_SEND:.2f} per send. Dataset: Hillstrom "
           "MineThatData (a real 64k-customer randomized experiment).")
