"""
모델 평가 및 교차 검증 기능
"""

import os
import json
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy import stats
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.stats.diagnostic import acorr_ljungbox

from config import CV_SPLITS, FIGURE_SIZE, DPI, OUTPUT_DIR, CV_MODE
from utils import calculate_metrics

logger = logging.getLogger(__name__)


def _walk_forward_splits(n, n_splits, test_size):
    """
    고정 테스트 크기의 walk-forward 분할 인덱스 생성.
    train 구간은 확장(expanding)되고 test는 연속된 `test_size` 구간.
    """
    splits = []
    for i in range(n_splits):
        test_end = n - (n_splits - 1 - i) * test_size
        test_start = test_end - test_size
        if test_start <= 0:
            continue
        train_idx = np.arange(0, test_start)
        test_idx = np.arange(test_start, test_end)
        splits.append((train_idx, test_idx))
    return splits


def time_series_cv(series, exog=None, model_type='arima', order=(1, 1, 1),
                   seasonal_order=(0, 0, 0, 12), n_splits=CV_SPLITS, test_size=None):
    """
    시계열 교차 검증 (walk-forward, 고정 테스트 창).
    """
    n = len(series)
    if test_size is None:
        test_size = max(1, n // (n_splits + 2))

    splits = _walk_forward_splits(n, n_splits, test_size)
    if not splits:
        raise ValueError(f"분할 실패: n={n}, n_splits={n_splits}, test_size={test_size}")

    fold_metrics = []
    logger.info(f"{len(splits)}겹 walk-forward CV ({model_type.upper()}), test_size={test_size}")

    for i, (train_idx, test_idx) in enumerate(splits, start=1):
        train = series.iloc[train_idx]
        test = series.iloc[test_idx]
        train_exog = exog.iloc[train_idx] if exog is not None else None
        test_exog = exog.iloc[test_idx] if exog is not None else None

        try:
            if model_type.lower() == 'arima':
                model = ARIMA(train, exog=train_exog, order=order).fit()
            else:
                model = SARIMAX(train, exog=train_exog, order=order,
                                seasonal_order=seasonal_order).fit(disp=False)
            forecast = model.forecast(steps=len(test), exog=test_exog)
        except Exception as e:
            logger.warning(f"Fold {i} 학습/예측 실패: {e}")
            continue

        m = calculate_metrics(test, forecast)
        m['fold'] = i
        fold_metrics.append(m)
        logger.info(f"Fold {i} - MAE={m['mae']:.2f} RMSE={m['rmse']:.2f} "
                    f"WMAPE={m['wmape']:.2f}% SMAPE={m['smape']:.2f}%")

    if not fold_metrics:
        raise RuntimeError("모든 CV fold가 실패했습니다.")

    avg = {k: float(np.mean([f[k] for f in fold_metrics if f.get(k) is not None]))
           for k in ('mae', 'rmse', 'smape', 'wmape')}
    logger.info(f"평균: MAE={avg['mae']:.2f} RMSE={avg['rmse']:.2f} "
                f"WMAPE={avg['wmape']:.2f}% SMAPE={avg['smape']:.2f}%")

    # 시각화
    plt.figure(figsize=FIGURE_SIZE)
    x = [f['fold'] for f in fold_metrics]
    for label in ('mae', 'rmse', 'smape', 'wmape'):
        plt.plot(x, [f[label] for f in fold_metrics], 'o-', label=label.upper())
    plt.xlabel('Fold')
    plt.ylabel('Error')
    plt.title(f'{model_type.upper()} walk-forward CV')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(OUTPUT_DIR, 'figures', f'{model_type}_cv_performance.png'), dpi=DPI)
    plt.close()

    results_summary = {
        "model_type": model_type,
        "cv_mode": CV_MODE,
        "order": order,
        "seasonal_order": seasonal_order if model_type.lower() == 'sarima' else None,
        "n_splits": len(fold_metrics),
        "test_size": test_size,
        "avg": avg,
        "folds": fold_metrics,
    }
    with open(os.path.join(OUTPUT_DIR, 'results', f'{model_type}_cv_results.json'), 'w') as f:
        json.dump(results_summary, f, indent=4)

    return avg


def residual_analysis(model, model_name='model'):
    """모델 잔차를 분석하여 모델의 적합성을 검증합니다."""
    residuals = model.resid

    plt.figure(figsize=FIGURE_SIZE)
    plt.subplot(2, 2, 1); plt.plot(residuals); plt.title('Residual timeseries'); plt.grid(True)
    plt.subplot(2, 2, 2); plt.hist(residuals, bins=30); plt.title('Residual histogram')
    plt.subplot(2, 2, 3)
    from statsmodels.graphics.tsaplots import plot_acf
    plot_acf(residuals, lags=40, ax=plt.gca())
    plt.subplot(2, 2, 4); stats.probplot(residuals, dist="norm", plot=plt); plt.title('Residual Q-Q plot')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'figures', f'{model_name}_residual_analysis.png'), dpi=DPI)
    plt.close()

    lb_test = acorr_ljungbox(residuals, lags=[10, 20, 30, 40])
    logger.info(f"{model_name} Ljung-Box 검정 (H0: 잔차 무자기상관)")

    lb_results = []
    if isinstance(lb_test, pd.DataFrame):
        iterator = [(lag, lb_test['lb_stat'].iloc[i], lb_test['lb_pvalue'].iloc[i])
                    for i, lag in enumerate([10, 20, 30, 40])]
    else:
        iterator = [(lag, lb_test[0][i], lb_test[1][i]) for i, lag in enumerate([10, 20, 30, 40])]

    for lag, q_stat, p_value in iterator:
        ok = p_value > 0.05
        logger.info(f"  Lag {lag}: Q={q_stat:.2f}, p={p_value:.4f} "
                    f"→ {'자기상관 없음' if ok else '자기상관 있음'}")
        lb_results.append({
            "lag": lag,
            "q_stat": float(q_stat),
            "p_value": float(p_value),
            "conclusion": "no autocorrelation" if ok else "autocorrelation present",
        })

    with open(os.path.join(OUTPUT_DIR, 'results', f'{model_name}_ljung_box_test.json'), 'w') as f:
        json.dump(lb_results, f, indent=4)


def evaluate_forecast(actual, predicted, model_name='model'):
    """예측 모델의 성능을 평가합니다."""
    metrics = calculate_metrics(actual, predicted)
    metrics['model_name'] = model_name

    logger.info(f"[{model_name}] MAE={metrics['mae']:.2f} RMSE={metrics['rmse']:.2f} "
                f"WMAPE={metrics['wmape']:.2f}% SMAPE={metrics['smape']:.2f}% "
                f"MAPE={metrics['mape'] if metrics['mape'] is not None else 'N/A'}")

    actual_values = np.asarray(actual.values if hasattr(actual, 'values') else actual, dtype=float)
    predicted_values = np.asarray(predicted.values if hasattr(predicted, 'values') else predicted, dtype=float)

    plt.figure(figsize=FIGURE_SIZE)

    plt.subplot(2, 2, 1)
    plt.scatter(actual_values, predicted_values, alpha=0.7)
    lo = min(np.min(actual_values), np.min(predicted_values))
    hi = max(np.max(actual_values), np.max(predicted_values))
    plt.plot([lo, hi], [lo, hi], 'r--')
    plt.xlabel('Real'); plt.ylabel('Predict'); plt.title('Real vs Predict'); plt.grid(True)

    plt.subplot(2, 2, 2)
    plt.hist(actual_values - predicted_values, bins=20)
    plt.xlabel('Error'); plt.ylabel('Frequency'); plt.title('Prediction error distribution')

    plt.subplot(2, 1, 2)
    x = range(len(actual_values))
    plt.plot(x, actual_values, label='Real', marker='o')
    plt.plot(x, predicted_values, label='Predict', marker='x')
    plt.xlabel('Time'); plt.ylabel('Value'); plt.title('Actual vs predicted time series')
    plt.legend(); plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'figures', f'{model_name}_forecast_evaluation.png'), dpi=DPI)
    plt.close()

    with open(os.path.join(OUTPUT_DIR, 'results', f'{model_name}_performance.json'), 'w') as f:
        json.dump(metrics, f, indent=4)

    return metrics


def compare_models(model_results):
    """여러 모델의 성능을 비교합니다."""
    models = list(model_results.keys())
    metrics = ['mae', 'rmse', 'smape', 'wmape']

    plt.figure(figsize=FIGURE_SIZE)
    x = np.arange(len(models))
    width = 0.2
    for i, metric in enumerate(metrics):
        values = [model_results[m][metric] for m in models]
        plt.bar(x + (i - 1.5) * width, values, width, label=metric.upper())

    plt.xlabel('Model'); plt.ylabel('Value'); plt.title('Model performance comparison')
    plt.xticks(x, models); plt.legend(); plt.grid(True, axis='y')
    plt.savefig(os.path.join(OUTPUT_DIR, 'figures', 'model_comparison.png'), dpi=DPI)
    plt.close()

    with open(os.path.join(OUTPUT_DIR, 'results', 'model_comparison.json'), 'w') as f:
        json.dump(model_results, f, indent=4)
