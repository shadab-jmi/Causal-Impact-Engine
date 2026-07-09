"""
Unit tests for the Part C single-customer scorer (src/uplift.py): a one-row prediction
must reproduce the batch prediction exactly (correct one-hot column alignment) and the
single-customer segment label must agree with the population segmentation
"""

import pytest

from src import data_prep as dp
from src import uplift

OUTCOME = "conversion"
RAW = ["recency", "history", "mens", "womens", "newbie", "history_segment", "zip_code", "channel"]


# Fit the T-learner once and expose the model, test set, feature template, and p0/p1
@pytest.fixture(scope="module")
def fitted():
    df = dp.add_treatment_flag(dp.load_raw())
    train, test = uplift.split_train_test(df, seed=42)
    model = uplift.fit_t_learner(train, OUTCOME)
    feat_cols = list(dp.encode_features(train).columns)
    p0, p1 = uplift.t_learner_p0_p1(model, test, OUTCOME)
    return {"test": test, "model": model, "feat_cols": feat_cols, "p0": p0, "p1": p1}


"""
Scoring one row via predict_customer == that same customer's prediction in the full
test matrix. This is the real risk: a single row's get_dummies only yields the categories
present (8 cols, not 12), so without column realignment the probabilities would silently be
wrong — realignment must reproduce the population prediction exactly.
"""
def test_predict_customer_matches_batch_prediction(fitted):
    test, model, fc = fitted["test"], fitted["model"], fitted["feat_cols"]
    p0_all, p1_all = fitted["p0"], fitted["p1"]
    for i in [0, 100, 5000, len(test) - 1]:
        row = test.iloc[i]
        customer = {c: row[c] for c in RAW}
        p0, p1, u = uplift.predict_customer(model, customer, fc)
        assert p0 == pytest.approx(float(p0_all[i]), abs=1e-9)
        assert p1 == pytest.approx(float(p1_all[i]), abs=1e-9)
        assert u == pytest.approx(p1 - p0, abs=1e-12)


def test_predict_customer_outputs_valid_probabilities(fitted):
    model, fc = fitted["model"], fitted["feat_cols"]
    customer = dict(recency=6, history=150.0, mens=1, womens=0, newbie=1,
                    history_segment="2) $100 - $200", zip_code="Urban", channel="Web")
    p0, p1, u = uplift.predict_customer(model, customer, fc)
    assert 0.0 <= p0 <= 1.0
    assert 0.0 <= p1 <= 1.0
    assert u == pytest.approx(p1 - p0)


# The refactor must not change behaviour: threshold-based labels == the original
def test_classify_segment_matches_segment_customers(fitted):
    p0, p1 = fitted["p0"], fitted["p1"]
    tau, p0_median = uplift.segment_thresholds(p0, p1)
    assert (uplift.classify_segment(p0, p1, tau, p0_median) == uplift.segment_customers(p0, p1)).all()


# A customer labelled one-at-a-time gets the same segment as in the batch call
def test_segment_for_single_matches_batch(fitted):
    p0, p1 = fitted["p0"], fitted["p1"]
    tau, p0_median = uplift.segment_thresholds(p0, p1)
    batch = uplift.segment_customers(p0, p1)
    for i in [0, 1, 50, 500, 9000]:
        assert uplift.segment_for(float(p0[i]), float(p1[i]), tau, p0_median) == batch[i]


def test_all_four_segments_are_reachable(fitted):
    p0, p1 = fitted["p0"], fitted["p1"]
    tau, _ = uplift.segment_thresholds(p0, p1)
    seg = uplift.segment_customers(p0, p1)
    assert set(seg) == {"persuadable", "sure_thing", "lost_cause", "sleeping_dog"}
    assert tau > 0            # persuadable cutoff is a genuine positive-uplift bar
