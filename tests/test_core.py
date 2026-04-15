"""
핵심 헬퍼 함수에 대한 단위 테스트.

실행:
    python -m pytest tests/
"""

import numpy as np
import pandas as pd
import pytest

from data_handler import prepare_train_test_split, inverse_transform
from utils import calculate_metrics, detect_seasonal_period
from evaluation import _walk_forward_splits


def _make_series(n=100):
    idx = pd.date_range('2020-01-01', periods=n, freq='W')
    return pd.Series(np.arange(n, dtype=float) + 10.0, index=idx)


def test_prepare_train_test_split_shapes():
    s = _make_series(100)
    train, test, tr_x, te_x = prepare_train_test_split(s, test_size=12)
    assert len(train) == 88
    assert len(test) == 12
    assert tr_x is None and te_x is None
    # 연속성
    assert train.index[-1] < test.index[0]


def test_prepare_train_test_split_with_exog():
    s = _make_series(50)
    exog = pd.DataFrame({'x': np.arange(50.0)}, index=s.index)
    train, test, tr_x, te_x = prepare_train_test_split(s, exog=exog, test_size=10)
    assert len(tr_x) == 40 and len(te_x) == 10
    assert np.all(tr_x['x'].values == np.arange(40.0))


def test_prepare_train_test_split_invalid():
    s = _make_series(20)
    with pytest.raises(ValueError):
        prepare_train_test_split(s, test_size=0)
    with pytest.raises(ValueError):
        prepare_train_test_split(s, test_size=20)


def test_inverse_transform_identity():
    s = _make_series(10)
    store = pd.DataFrame({'Weekly_Sales': np.arange(1.0, 11.0)})
    out = inverse_transform(s.values, store, transformed=False)
    np.testing.assert_array_equal(out, s.values)


def test_inverse_transform_log_roundtrip():
    raw = np.array([1.0, 2.0, 5.0, 10.0])
    store = pd.DataFrame({'Weekly_Sales': raw})
    logged = np.log(raw)
    back = inverse_transform(logged, store, transformed=True)
    np.testing.assert_allclose(back, raw, rtol=1e-10)


def test_inverse_transform_log1p_with_zero():
    raw = np.array([0.0, 1.0, 3.0])
    store = pd.DataFrame({'Weekly_Sales': raw})
    min_val = raw.min()  # 0
    logged = np.log1p(raw + abs(min_val) + 1)
    back = inverse_transform(logged, store, transformed=True)
    np.testing.assert_allclose(back, raw, rtol=1e-10, atol=1e-10)


def test_calculate_metrics_perfect():
    actual = np.array([10.0, 20.0, 30.0, 40.0])
    m = calculate_metrics(actual, actual.copy())
    assert m['mae'] == 0.0
    assert m['rmse'] == 0.0
    assert m['wmape'] == pytest.approx(0.0, abs=1e-6)
    assert m['smape'] == pytest.approx(0.0, abs=1e-6)


def test_calculate_metrics_known_values():
    actual = np.array([100.0, 200.0])
    pred = np.array([110.0, 190.0])
    m = calculate_metrics(actual, pred)
    # |10|+|10| = 20; MAE = 10
    assert m['mae'] == pytest.approx(10.0)
    # RMSE = sqrt((100+100)/2) = 10
    assert m['rmse'] == pytest.approx(10.0)
    # WMAPE = 20/300 * 100
    assert m['wmape'] == pytest.approx(20.0 / 300.0 * 100, rel=1e-4)


def test_calculate_metrics_handles_zero_actual():
    actual = np.array([0.0, 10.0])
    pred = np.array([1.0, 9.0])
    m = calculate_metrics(actual, pred)
    # mape는 0이 아닌 샘플만 사용 → |10-9|/10 = 10%
    assert m['mape'] == pytest.approx(10.0, rel=1e-6)
    assert m['n_mape_samples'] == 1


def test_walk_forward_splits_properties():
    splits = _walk_forward_splits(n=100, n_splits=5, test_size=10)
    assert len(splits) == 5
    for train_idx, test_idx in splits:
        assert len(test_idx) == 10
        # 훈련 인덱스가 테스트 이전 구간이어야 함 (누수 없음)
        assert train_idx[-1] < test_idx[0]
    # 마지막 fold의 test가 시리즈 끝까지
    _, last_test = splits[-1]
    assert last_test[-1] == 99


def test_detect_seasonal_period_sine():
    # 주기 12짜리 싸인 신호
    n = 200
    t = np.arange(n)
    s = pd.Series(np.sin(2 * np.pi * t / 12) + 0.05 * np.random.RandomState(0).randn(n))
    detected = detect_seasonal_period(s, candidates=(4, 7, 12, 24))
    assert detected == 12
