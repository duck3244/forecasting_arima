"""
예측 수행 기능
"""

import os
import json
import logging
import numpy as np
import pandas as pd
import joblib

from statsmodels.tsa.statespace.sarimax import SARIMAX

from config import FORECAST_STEPS, OUTPUT_DIR, PI_ALPHA
from data_handler import inverse_transform

logger = logging.getLogger(__name__)


def _forecast_with_ci(model, steps, exog=None, alpha=PI_ALPHA):
    """
    가능한 경우 get_forecast()로 점 예측과 신뢰구간을 함께 반환합니다.
    실패 시 ARIMA(statsmodels)의 get_forecast도 지원하며, 최종 fallback은 점 예측만.
    """
    try:
        gf = model.get_forecast(steps=steps, exog=exog)
        mean = gf.predicted_mean
        ci = gf.conf_int(alpha=alpha)
        if isinstance(ci, pd.DataFrame):
            lower = ci.iloc[:, 0]
            upper = ci.iloc[:, 1]
        else:
            lower = pd.Series(ci[:, 0])
            upper = pd.Series(ci[:, 1])
        return mean, lower, upper
    except Exception as e:
        logger.warning(f"신뢰구간 계산 실패, 점 예측으로 대체: {e}")
        mean = model.forecast(steps=steps, exog=exog)
        nan_series = pd.Series([np.nan] * steps, index=getattr(mean, 'index', None))
        return mean, nan_series, nan_series


def ensemble_forecast(models, steps=FORECAST_STEPS, exog=None):
    """여러 모델의 예측을 결합합니다."""
    if not models:
        return None
    aligned = []
    for model in models:
        f = model.forecast(steps=steps, exog=exog)
        aligned.append(f.values if isinstance(f, pd.Series) else np.asarray(f))
    return np.mean(aligned, axis=0)


def predict_future(model, steps=FORECAST_STEPS, exog=None, series=None,
                   store_data=None, transformed=False):
    """
    미래 값을 예측하고 (예측값, 하한, 상한, 날짜)를 반환합니다.
    """
    mean, lower, upper = _forecast_with_ci(model, steps, exog=exog)

    future_mean = inverse_transform(mean, store_data, transformed) if transformed else mean
    future_lower = inverse_transform(lower, store_data, transformed) if transformed else lower
    future_upper = inverse_transform(upper, store_data, transformed) if transformed else upper

    if series is not None and isinstance(series.index[-1], pd.Timestamp):
        last_date = series.index[-1]
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=7), periods=steps, freq='W')
    else:
        future_dates = range(steps)

    future_df = pd.DataFrame({
        'Date': future_dates,
        'Predicted_Sales': np.asarray(future_mean),
        'Lower_CI': np.asarray(future_lower),
        'Upper_CI': np.asarray(future_upper),
    })

    results_path = os.path.join(OUTPUT_DIR, 'results', 'future_predictions.csv')
    future_df.to_csv(results_path, index=False)

    json_data = []
    for _, row in future_df.iterrows():
        date_str = row['Date'].strftime('%Y-%m-%d') if isinstance(row['Date'], pd.Timestamp) else str(row['Date'])
        json_data.append({
            "date": date_str,
            "prediction": float(row['Predicted_Sales']),
            "lower_ci": float(row['Lower_CI']) if not np.isnan(row['Lower_CI']) else None,
            "upper_ci": float(row['Upper_CI']) if not np.isnan(row['Upper_CI']) else None,
        })
    with open(os.path.join(OUTPUT_DIR, 'results', 'future_predictions.json'), 'w') as f:
        json.dump(json_data, f, indent=4)

    return np.asarray(future_mean), future_dates, np.asarray(future_lower), np.asarray(future_upper)


def create_final_model(series, exog=None, order=(1, 1, 1), seasonal_order=(0, 0, 0, 12)):
    """최종 예측 모델을 생성합니다 (미래 예측 전용, 전체 데이터로 학습)."""
    logger.info(f"최종 예측 모델 학습: SARIMAX{order}x{seasonal_order}")
    final_model = SARIMAX(series, exog=exog, order=order, seasonal_order=seasonal_order).fit(disp=False)
    logger.debug(final_model.summary())

    model_path = os.path.join(OUTPUT_DIR, 'models', 'final_model.pkl')
    joblib.dump(final_model, model_path)
    logger.info(f"최종 모델 저장: {model_path}")
    return final_model


def prepare_future_exog(exog, steps=FORECAST_STEPS):
    """
    미래 예측을 위한 외생 변수를 준비합니다.

    주의: 실제 운영 환경에서는 외생 변수의 미래값을 별도 예측하거나
    기획된 값을 사용해야 합니다. 현재 구현은 마지막 N개 값을 그대로
    재사용하는 단순한 플레이스홀더로, 해석 시 유의해야 합니다.
    """
    if exog is None:
        return None
    future_exog = exog.iloc[-steps:].reset_index(drop=True)
    logger.warning("미래 외생변수는 마지막 구간을 복제 사용 중 (점진적 개선 필요)")
    return future_exog
