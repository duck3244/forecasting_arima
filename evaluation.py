"""
모델 평가 및 교차 검증 기능
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy import stats
from statsmodels.tsa.arima.model import ARIMA
from sklearn.model_selection import TimeSeriesSplit
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.stats.diagnostic import acorr_ljungbox
from sklearn.metrics import mean_absolute_error, mean_squared_error

from config import CV_SPLITS, FIGURE_SIZE, DPI, OUTPUT_DIR


def time_series_cv(series, exog=None, model_type='arima', order=(1, 1, 1), seasonal_order=(0, 0, 0, 12),
                   n_splits=CV_SPLITS):
    """
    시계열 교차 검증을 통해 모델의 성능을 평가합니다.

    Args:
        series (pandas.Series): 시계열 데이터
        exog (pandas.DataFrame, optional): 외생 변수 데이터
        model_type (str): 모델 유형 ('arima' 또는 'sarima')
        order (tuple): ARIMA/SARIMA 모델 차수 (p, d, q)
        seasonal_order (tuple): SARIMA 계절성 차수 (P, D, Q, s)
        n_splits (int): 교차 검증 분할 수

    Returns:
        tuple: (평균 MAE, 평균 RMSE, 평균 MAPE)
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)

    mae_scores = []
    rmse_scores = []
    mape_scores = []
    smape_scores = []

    cv_results = []  # 각 분할별 결과 저장

    print(f"\n=== {n_splits}겹 시계열 교차 검증 ({model_type.upper()}) ===")

    for i, (train_idx, test_idx) in enumerate(tscv.split(series)):
        train = series.iloc[train_idx]
        test = series.iloc[test_idx]

        # 외생 변수가 있는 경우
        train_exog = None
        test_exog = None
        if exog is not None:
            train_exog = exog.iloc[train_idx]
            test_exog = exog.iloc[test_idx]

        # 모델 학습
        if model_type.lower() == 'arima':
            model = ARIMA(train, exog=train_exog, order=order).fit()
        else:  # 'sarima'
            model = SARIMAX(train, exog=train_exog, order=order, seasonal_order=seasonal_order).fit(disp=False)

        # 예측
        forecast = model.forecast(steps=len(test), exog=test_exog)

        # 성능 평가
        mae = mean_absolute_error(test, forecast)
        rmse = np.sqrt(mean_squared_error(test, forecast))

        # Series를 numpy 배열로 변환하여 인덱스 문제 해결
        test_values = test.values if hasattr(test, 'values') else np.array(test)
        forecast_values = forecast.values if hasattr(forecast, 'values') else np.array(forecast)

        # MAPE 계산 (0 값 처리)
        epsilon = 1e-10  # 매우 작은 값
        non_zero_mask = test_values > epsilon

        mape = np.nan
        if np.any(non_zero_mask):
            mape = np.mean(np.abs(
                (test_values[non_zero_mask] - forecast_values[non_zero_mask]) / test_values[non_zero_mask])) * 100
            mape_scores.append(mape)

        # SMAPE 계산
        smape = np.mean(2.0 * np.abs(test_values - forecast_values) / (
                np.abs(test_values) + np.abs(forecast_values) + epsilon)) * 100
        smape_scores.append(smape)

        mae_scores.append(mae)
        rmse_scores.append(rmse)

        # 분할별 결과 저장
        fold_result = {
            "fold": i + 1,
            "mae": float(mae),
            "rmse": float(rmse),
            "mape": float(mape) if not np.isnan(mape) else None,
            "smape": float(smape)
        }
        cv_results.append(fold_result)

        print(f"분할 {i + 1} - MAE: {mae:.2f}, RMSE: {rmse:.2f}, SMAPE: {smape:.2f}%",
              f", MAPE: {mape:.2f}%" if not np.isnan(mape) else "")

    # 평균 성능 계산
    avg_mae = np.mean(mae_scores)
    avg_rmse = np.mean(rmse_scores)
    avg_mape = np.mean(mape_scores) if mape_scores else np.nan
    avg_smape = np.mean(smape_scores)

    print(f"\n평균 MAE: {avg_mae:.2f}")
    print(f"평균 RMSE: {avg_rmse:.2f}")
    print(f"평균 SMAPE: {avg_smape:.2f}%")
    if not np.isnan(avg_mape):
        print(f"평균 MAPE: {avg_mape:.2f}%")
    else:
        print("평균 MAPE: 계산 불가 (0 값 존재)")

    # 교차 검증 결과 시각화
    plt.figure(figsize=FIGURE_SIZE)
    x = range(1, n_splits + 1)
    plt.plot(x, mae_scores, 'o-', label='MAE')
    plt.plot(x, rmse_scores, 's-', label='RMSE')
    plt.plot(x, smape_scores, '^-', label='SMAPE')

    if mape_scores:
        plt.plot(x[:len(mape_scores)], mape_scores, 'd-', label='MAPE')

    plt.xlabel('Split number')
    plt.ylabel('Error')
    plt.title(f'{model_type.upper()} Model cross validation performance')
    plt.legend()
    plt.grid(True)

    # 그림 저장
    save_path = os.path.join(OUTPUT_DIR, 'figures', f'{model_type}_cv_performance.png')
    plt.savefig(save_path, dpi=DPI)
    plt.close()

    # 결과 저장
    results_summary = {
        "model_type": model_type,
        "order": order,
        "seasonal_order": seasonal_order if model_type.lower() == 'sarima' else None,
        "cv_folds": n_splits,
        "avg_mae": float(avg_mae),
        "avg_rmse": float(avg_rmse),
        "avg_mape": float(avg_mape) if not np.isnan(avg_mape) else None,
        "avg_smape": float(avg_smape),
        "fold_results": cv_results
    }

    json_path = os.path.join(OUTPUT_DIR, 'results', f'{model_type}_cv_results.json')
    with open(json_path, 'w') as f:
        json.dump(results_summary, f, indent=4)

    return avg_mae, avg_rmse, avg_mape


def residual_analysis(model, model_name='model'):
    """
    모델 잔차를 분석하여 모델의 적합성을 검증합니다.
    
    Args:
        model (statsmodels 모델): 학습된 모델
        model_name (str): 모델 이름 (파일 저장용)
    """
    residuals = model.resid

    plt.figure(figsize=FIGURE_SIZE)
    plt.subplot(2, 2, 1)
    plt.plot(residuals)
    plt.title('Residual timeseries')
    plt.grid(True)

    plt.subplot(2, 2, 2)
    plt.hist(residuals, bins=30)
    plt.title('Residual histogram')

    plt.subplot(2, 2, 3)
    from statsmodels.graphics.tsaplots import plot_acf
    plot_acf(residuals, lags=40, ax=plt.gca())

    plt.subplot(2, 2, 4)
    stats.probplot(residuals, dist="norm", plot=plt)
    plt.title('Residual Q-Q plot')

    plt.tight_layout()
    
    # 그림 저장
    save_path = os.path.join(OUTPUT_DIR, 'figures', f'{model_name}_residual_analysis.png')
    plt.savefig(save_path, dpi=DPI)
    plt.close()

    # Ljung-Box 검정
    lb_test = acorr_ljungbox(residuals, lags=[10, 20, 30, 40])
    print(f"\n=== {model_name} Ljung-Box 검정 결과 ===")
    print("H0: 잔차에 자기상관이 없다 (모델이 적합함)")

    # acorr_ljungbox의 반환 값 형식 확인
    lb_results = []
    
    if isinstance(lb_test, pd.DataFrame):
        # 최신 버전의 statsmodels에서는 DataFrame 반환
        for i, lag in enumerate([10, 20, 30, 40]):
            p_value = lb_test['lb_pvalue'].iloc[i]
            q_stat = lb_test['lb_stat'].iloc[i]
            
            print(f"Lag {lag}: Q={q_stat:.2f}, p-value={p_value:.4f}")
            if p_value > 0.05:
                print(f"  → lag {lag}에서 자기상관 없음 (모델 적합)")
            else:
                print(f"  → lag {lag}에서 자기상관 있음 (모델 개선 필요)")
            
            lb_results.append({
                "lag": lag,
                "q_stat": float(q_stat),
                "p_value": float(p_value),
                "conclusion": "no autocorrelation" if p_value > 0.05 else "autocorrelation present"
            })
    else:
        # 이전 버전의 statsmodels에서는 딕셔너리 반환
        for i, lag in enumerate([10, 20, 30, 40]):
            p_value = lb_test[1][i]
            q_stat = lb_test[0][i]
            
            print(f"Lag {lag}: Q={q_stat:.2f}, p-value={p_value:.4f}")
            if p_value > 0.05:
                print(f"  → lag {lag}에서 자기상관 없음 (모델 적합)")
            else:
                print(f"  → lag {lag}에서 자기상관 있음 (모델 개선 필요)")
            
            lb_results.append({
                "lag": lag,
                "q_stat": float(q_stat),
                "p_value": float(p_value),
                "conclusion": "no autocorrelation" if p_value > 0.05 else "autocorrelation present"
            })
    
    # 검정 결과 저장
    results_path = os.path.join(OUTPUT_DIR, 'results', f'{model_name}_ljung_box_test.json')
    with open(results_path, 'w') as f:
        json.dump(lb_results, f, indent=4)


def evaluate_forecast(actual, predicted, model_name='model'):
    """
    예측 모델의 성능을 평가합니다.
    
    Args:
        actual: 실제 값
        predicted: 예측 값
        model_name (str): 모델 이름 (파일 저장용)
        
    Returns:
        dict: 평가 지표 결과
    """
    # MAE와 RMSE는 그대로 계산
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))

    # Series를 numpy 배열로 변환
    actual_values = actual.values if hasattr(actual, 'values') else np.array(actual)
    predicted_values = predicted.values if hasattr(predicted, 'values') else np.array(predicted)

    # MAPE 계산 (0 값 처리)
    epsilon = 1e-10  # 매우 작은 값
    # 양수 값만 사용
    non_zero_mask = actual_values > epsilon

    mape = np.nan
    if np.any(non_zero_mask):
        mape = np.mean(np.abs(
            (actual_values[non_zero_mask] - predicted_values[non_zero_mask]) / actual_values[non_zero_mask])) * 100

    # SMAPE 계산 (대칭적 MAPE - 0 값에 덜 민감)
    smape = np.mean(2.0 * np.abs(actual_values - predicted_values) / (
                np.abs(actual_values) + np.abs(predicted_values) + epsilon)) * 100

    # WMAPE (Weighted MAPE)
    wmape = np.sum(np.abs(actual_values - predicted_values)) / np.sum(np.abs(actual_values) + epsilon) * 100

    print(f"\n=== {model_name} 예측 성능 평가 ===")
    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"SMAPE: {smape:.2f}%")
    if not np.isnan(mape):
        print(f"MAPE: {mape:.2f}%")
    else:
        print("MAPE: 계산 불가 (0 값 존재)")
    print(f"WMAPE: {wmape:.2f}%")
    
    # 오차 시각화
    plt.figure(figsize=FIGURE_SIZE)
    
    # 실제 vs 예측 산점도
    plt.subplot(2, 2, 1)
    plt.scatter(actual_values, predicted_values, alpha=0.7)
    
    # 대각선 (완벽한 예측)
    min_val = min(np.min(actual_values), np.min(predicted_values))
    max_val = max(np.max(actual_values), np.max(predicted_values))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    
    plt.xlabel('Real')
    plt.ylabel('Predict')
    plt.title('Real vs Predict')
    plt.grid(True)
    
    # 오차 히스토그램
    plt.subplot(2, 2, 2)
    errors = actual_values - predicted_values
    plt.hist(errors, bins=20)
    plt.xlabel('Error')
    plt.ylabel('Frequency')
    plt.title('Prediction error distribution')
    
    # 실제 vs 예측 시계열
    plt.subplot(2, 1, 2)
    x = range(len(actual_values))
    plt.plot(x, actual_values, label='Real', marker='o')
    plt.plot(x, predicted_values, label='Predict', marker='x')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.title('Actual vs predicted time series')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    
    # 그림 저장
    save_path = os.path.join(OUTPUT_DIR, 'figures', f'{model_name}_forecast_evaluation.png')
    plt.savefig(save_path, dpi=DPI)
    plt.close()
    
    # 결과 저장
    results = {
        "model_name": model_name,
        "mae": float(mae),
        "rmse": float(rmse),
        "mape": float(mape) if not np.isnan(mape) else None,
        "smape": float(smape),
        "wmape": float(wmape)
    }
    
    json_path = os.path.join(OUTPUT_DIR, 'results', f'{model_name}_performance.json')
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=4)

    return results


def compare_models(model_results):
    """
    여러 모델의 성능을 비교합니다.
    
    Args:
        model_results (dict): 모델별 성능 결과
    """
    models = list(model_results.keys())
    metrics = ['mae', 'rmse', 'smape', 'wmape']
    
    # 그래프 생성
    plt.figure(figsize=FIGURE_SIZE)
    
    # 막대 그래프 위치 설정
    x = np.arange(len(models))
    width = 0.2
    
    # 각 지표별 막대 그래프
    for i, metric in enumerate(metrics):
        values = [model_results[model][metric] for model in models]
        plt.bar(x + (i - 1.5) * width, values, width, label=metric.upper())
    
    plt.xlabel('Model')
    plt.ylabel('Value')
    plt.title('Model performance comparison')
    plt.xticks(x, models)
    plt.legend()
    plt.grid(True, axis='y')
    
    # 그림 저장
    save_path = os.path.join(OUTPUT_DIR, 'figures', 'model_comparison.png')
    plt.savefig(save_path, dpi=DPI)
    plt.close()
    
    # 결과 저장
    comparison_results = {model: model_results[model] for model in models}
    
    json_path = os.path.join(OUTPUT_DIR, 'results', 'model_comparison.json')
    with open(json_path, 'w') as f:
        json.dump(comparison_results, f, indent=4)

