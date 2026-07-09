# Retention Offer: Did it work, can we trust it, and who should we target?

**Stakeholder report · Growth / Product** · Prepared by the Data Science team

---

## TL;DR — the decision

**Roll out the retention offer.** It causes a real, statistically significant, and profitable increase in
customer activity and spend. Each emailed customer converts about **0.5 percentage points** more often
(a **relative +87%** on a low base) and spends **≈ $0.60** more — worth **≈ $0.08 net profit per
customer (ROI +79%)** after the $0.10 send cost. With today's economics the offer is profitable for
**almost every customer**, so **target broadly**; when send budget is limited, **rank customers by
predicted uplift** to get the most incremental conversions per dollar.

We can stand behind this even though we won't always be able to run a clean experiment: on deliberately
biased data, a naïve analysis overstated the effect by **~40%**, but our correction methods recovered the
true number — so the measurement approach is trustworthy going forward.

---

## The three questions leadership asked

### 1. Did the offer work? *(Yes — and it pays.)*

The campaign was a genuine randomized experiment, so the difference between emailed and non-emailed
customers is the **true causal effect** — not a correlation.

| Metric | Control | Treated | Causal lift (95% CI) | Significant? |
|---|---|---|---|---|
| Site visit | 10.6% | 16.7% | **+6.09 pp** [+5.54, +6.63] | yes (p ≈ 10⁻⁹³) |
| Conversion *(primary)* | 0.57% | 1.07% | **+0.495 pp** [+0.36, +0.64] | yes (p ≈ 4×10⁻¹⁰) |
| Spend | $0.65 | $1.25 | **+$0.597** [+$0.38, +$0.82] | yes (p ≈ 10⁻⁷) |

**Economics** (assumptions: 30% margin on spend, $0.10 per send):

- Incremental margin per customer: 30% × $0.597 = **$0.179**
- Minus send cost $0.10 → **$0.079 net profit per customer**, an **ROI of +79%**
- Across the 42,694 treated customers: **≈ $3,374 incremental profit** — or about **$7,900 per 100,000
  customers** contacted.

*Caveat:* conversion is a rare event (~0.9% overall), so the absolute lift is small even though it is
highly significant and nearly doubles the rate. We are well-powered for the effect we saw, but detecting
a *smaller* future change would require a much larger sample.

See `figures/05_exec_partA.png`.

### 2. Can we trust the analysis if we can't randomize? *(Yes, with the right method.)*

Next time we may not be able to run a clean experiment — we may just observe who happened to receive the
offer. To test whether that's safe, we deliberately created a **biased** dataset (where the offer went
disproportionately to high-spending customers) and compared approaches against the known true effect of
**+0.495 pp**:

| Approach | Estimated conversion lift | Verdict |
|---|---|---|
| **Naïve** (just compare who got it) | **+0.704 pp** | ❌ overstates the truth by **+42%** |
| Propensity Score Matching (PSM) | +0.534 pp | ✅ recovers the truth |
| Inverse-Propensity Weighting (IPW) | +0.582 pp | ✅ recovers the truth |

A naïve comparison would have credited the offer for loyalty it didn't create. Our adjustment methods
removed that bias and landed back on the true effect — and the result held up under formal robustness
checks (placebo and stability tests via the DoWhy framework). **Takeaway for the business:** observational
read-outs are usable *if* we apply proper causal adjustment; a raw comparison is not.

See `figures/05_exec_partB.png`.

### 3. Who should we target next time? *(Persuadables — and here's how much it helps.)*

We built an **uplift model** that estimates the offer's effect *per customer*, and ranks customers from
most to least persuadable.

- **Under a fixed budget, uplift targeting beats a conventional "most-likely-to-convert" model.** At the
  top 30% of customers it captures **47%** of all incremental conversions, versus **32%** for a response
  model and 30% for random — the same spend, materially more incremental sales.
- **Segments:** the model separates *persuadables* (target these), *sure-things* (would buy anyway —
  wasted offer), *lost-causes* (won't buy either way), and potential *sleeping-dogs* (offer might annoy).
- **Sleeping dogs — checked, not assumed.** The model flagged ~934 candidates, but on held-out data their
  treated-vs-control gap was **+0.32 pp (p = 0.61)** — statistically indistinguishable from zero. **We do
  not claim a sleeping-dog segment exists**; the negative predictions are noise. Confirming any real harm
  would need a dedicated follow-up test.
- **From policy to a per-customer tool.** The dashboard exposes a **single-customer scorer**: a campaign
  manager can enter one customer's profile (recency, prior spend, product history, region, channel) and
  immediately see that customer's predicted uplift, a *target / don't-target* recommendation, and the
  segment they fall into — the same model used for the aggregate ranking, applied to one record at a time.

See `figures/05_exec_partC.png`.

---

## Recommendation

1. **Roll out the offer** to the broad eligible base — with current cost/margin it is profitable for
   almost everyone.
2. **When send budget is constrained**, prioritize by **predicted uplift** rather than by likelihood to
   convert, to maximize incremental conversions per dollar.
3. **Re-evaluate the targeting cutoff if economics change.** If the send cost rises or margin falls, the
   profit-maximizing reach shrinks and uplift-based *exclusion* begins to pay — the model already gives us
   that lever.
4. **For future non-randomized read-outs**, always apply causal adjustment (PSM/IPW); never trust a raw
   treated-vs-control comparison.

## Caveats & assumptions

- **Economics are assumptions**, not measured: 30% margin, $0.10/send. All dollar figures scale with these.
- **Rare outcome:** conversion ≈ 0.9%; absolute effects are small and future tests need large samples.
- **Measurement window** is 2 weeks — a short-run effect that may differ from steady state (novelty).
- **Observational validity** rests on having measured the relevant confounders (unconfoundedness),
  overlap between groups, and no interference between customers (SUTVA). These held in our test by
  construction; in the wild they must be argued, not assumed.
- **No unsubscribe/complaint guardrails** are present in this dataset — a real rollout should monitor them.

---

*Methods and reproducible analysis: notebooks `01`–`05` and the `src/` modules in this repository.*
