import pandas as pd
from src.model import load_data, evaluate_features


def test_data_loading():
    df = load_data("data/soil_measures.csv")
    assert not df.empty
    assert "crop" in df.columns


def test_feature_evaluation_returns_scores():
    df = load_data("data/soil_measures.csv")
    results = evaluate_features(df)

    assert isinstance(results, dict)
    assert len(results) == 4


def test_f1_scores_are_valid():
    df = load_data("data/soil_measures.csv")
    results = evaluate_features(df)

    for score in results.values():
        assert 0.0 <= score <= 1.0
