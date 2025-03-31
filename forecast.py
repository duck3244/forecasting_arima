"""
예측 수행 기능
"""

import os
import json
import numpy as np
import pandas as pd

from statsmodels.tsa.statespace.sarimax import SARIMAX

from config import FORECAST_STEPS, OUTPUT_DIR


def ensemble_forecast(models, steps=FORECAST_STEPS, exog=None):
    """
    여러 모델의 예측을 결합합니다.
    
    Args:
        models (list): 학습된 모델 리스트
        steps (int): 예측 기간
        exog (pandas.DataFrame, optional): 외생 변수 데이터
        
    Returns:
        numpy.ndarray: 앙상블 예측 결과
    """
    forecasts = []

    for model in models:
        # 모델 예측 수행
        forecast = model.forecast(steps=steps, exog=exog)
        forecasts.append(forecast)

    # 모든 예측값의 평균 계산
    if forecasts:
        # 모든 예측을 같은 형태로 변환
        aligned_forecasts = []
        for f in forecasts:
            if isinstance(f, pd.Series):
                aligned_forecasts.append(f.values)
            else:
                aligned_forecasts.append(f)

        # 평균 계산
        ensemble_result = np.mean(aligned_forecasts, axis=0)
        return ensemble_result
    else:
        return None


def predict_future(model, steps=FORECAST_STEPS, exog=None, series=None, store_data=None, transformed=False):
    """
    미래 값을 예측합니다.
    
    Args:
        model: 학습된 모델
        steps (int): 예측 기간
        exog (pandas.DataFrame, optional): 외생 변수 데이터
        series (pandas.Series): 원본 시계열 데이터 (인덱스 참조용)
        store_data (pandas.DataFrame): 변환된 경우 역변환에 사용
        transformed (bool): 변환 여부
        
    Returns:
        tuple: (예측 결과, 예측 날짜)
    """
    # 미래 예측
    future_forecast = model.forecast(steps=steps, exog=exog)
    
    # 변환된 경우 역변환
    if transformed and store_data is not None:
        if (store_data['Weekly_Sales'] <= 0).any():
            # log1p 변환을 사용한 경우
            min_val = store_data['Weekly_Sales'].min()
            future_forecast_orig = np.expm1(future_forecast) - abs(min_val) - 1
        else:
            # 일반 로그 변환을 사용한 경우
            future_forecast_orig = np.exp(future_forecast)
    else:
        future_forecast_orig = future_forecast
    
    # 마지막 날짜 이후로 날짜 생성
    if series is not None:
        last_date = series.index[-1]
        if isinstance(last_date, pd.Timestamp):
            # 주간 데이터로 가정
            future_dates = pd.date_range(start=last_date + pd.Timedelta(days=7), periods=steps, freq='W')
        else:
            future_dates = range(len(series), len(series) + steps)
    else:
        future_dates = range(steps)
    
    # 미래 예측 결과를 데이터프레임으로 변환
    future_df = pd.DataFrame({
        'Date': future_dates,
        'Predicted_Sales': future_forecast_orig
    })
    
    # 결과 저장
    results_path = os.path.join(OUTPUT_DIR, 'results', 'future_predictions.csv')
    future_df.to_csv(results_path, index=False)
    
    # JSON 형식으로도 저장 (날짜 처리)
    json_data = []
    for i, row in future_df.iterrows():
        date_str = row['Date'].strftime('%Y-%m-%d') if isinstance(row['Date'], pd.Timestamp) else str(row['Date'])
        json_data.append({
            "date": date_str,
            "prediction": float(row['Predicted_Sales'])
        })
    
    json_path = os.path.join(OUTPUT_DIR, 'results', 'future_predictions.json')
    with open(json_path, 'w') as f:
        json.dump(json_data, f, indent=4)
    
    return future_forecast_orig, future_dates


def create_final_model(series, exog=None, order=(1, 1, 1), seasonal_order=(0, 0, 0, 12)):
    """
    최종 예측 모델을 생성합니다.
    
    Args:
        series (pandas.Series): 시계열 데이터
        exog (pandas.DataFrame, optional): 외생 변수 데이터
        order (tuple): ARIMA 모델 차수 (p, d, q)
        seasonal_order (tuple): 계절성 차수 (P, D, Q, s)
        
    Returns:
        statsmodels 모델: 학습된 SARIMAX 모델
    """
    print("\n=== 최종 예측 모델 학습 ===")
    
    # SARIMAX 모델 학습
    final_model = SARIMAX(
        series,
        exog=exog,
        order=order,
        seasonal_order=seasonal_order
    ).fit(disp=False)
    
    print(f"최종 모델: SARIMAX{order}x{seasonal_order}")
    print(final_model.summary())
    
    # 모델 저장
    model_path = os.path.join(OUTPUT_DIR, 'models', 'final_model.pkl')
    import joblib
    joblib.dump(final_model, model_path)
    print(f"최종 모델 저장: {model_path}")
    
    return final_model


def prepare_future_exog(exog, steps=FORECAST_STEPS):
    """
    미래 예측을 위한 외생 변수를 준비합니다.
    
    Args:
        exog (pandas.DataFrame): 외생 변수 데이터
        steps (int): 예측 기간
        
    Returns:
        pandas.DataFrame: 미래 외생 변수
    """
    if exog is None:
        return None
    
    # 실제로는 미래 외생 변수 값을 적절히 예측해야 함
    # 이 함수에서는 간단히 마지막 값들을 반복 사용
    future_exog = exog.iloc[-steps:].reset_index(drop=True)
    
    # 다른 방법: 시계열 예측 또는 평균 사용
    # future_exog = pd.DataFrame({col: [exog[col].mean()] * steps for col in exog.columns})
    
    return future_exog

