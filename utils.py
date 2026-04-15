"""
유틸리티 함수
"""

import os
import json
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from datetime import datetime
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import acf
from sklearn.metrics import mean_absolute_error, mean_squared_error

from config import OUTPUT_DIR, DPI, EPSILON

logger = logging.getLogger(__name__)


def calculate_metrics(actual, predicted):
    """
    공통 예측 성능 지표 계산 헬퍼.

    반환: dict(mae, rmse, mape, smape, wmape)
    - mape: 실제값이 epsilon보다 큰 표본만 사용 (표본 수도 반환)
    - wmape: 0값에 강건한 가중 절대 백분율 오차 (주 지표 권장)
    """
    actual = np.asarray(actual.values if hasattr(actual, 'values') else actual, dtype=float)
    predicted = np.asarray(predicted.values if hasattr(predicted, 'values') else predicted, dtype=float)

    mae = float(mean_absolute_error(actual, predicted))
    rmse = float(np.sqrt(mean_squared_error(actual, predicted)))

    non_zero = np.abs(actual) > EPSILON
    mape = float('nan')
    if np.any(non_zero):
        mape = float(np.mean(np.abs((actual[non_zero] - predicted[non_zero]) / actual[non_zero])) * 100)

    smape = float(np.mean(2.0 * np.abs(actual - predicted) /
                          (np.abs(actual) + np.abs(predicted) + EPSILON)) * 100)
    wmape = float(np.sum(np.abs(actual - predicted)) / (np.sum(np.abs(actual)) + EPSILON) * 100)

    return {
        'mae': mae,
        'rmse': rmse,
        'mape': mape if not np.isnan(mape) else None,
        'smape': smape,
        'wmape': wmape,
        'n_mape_samples': int(non_zero.sum()),
    }


def detect_seasonal_period(series, candidates=(4, 7, 12, 24, 26, 52), min_obs_multiplier=2):
    """
    ACF 피크를 기준으로 계절성 주기를 자동 감지합니다.
    - 후보 중 관측 수가 충분한(period * min_obs_multiplier 이상) 항목만 고려
    - 후보 lag에서의 ACF 절대값이 가장 큰 주기를 반환
    - 감지 실패 시 None
    """
    series = series.dropna()
    n = len(series)
    valid = [p for p in candidates if n >= p * min_obs_multiplier]
    if not valid:
        return None
    max_lag = min(n - 1, max(valid) + 1)
    try:
        acf_vals = acf(series, nlags=max_lag, fft=True)
    except Exception as e:
        logger.warning(f"ACF 계산 실패, 계절성 자동 감지를 건너뜁니다: {e}")
        return None
    best = max(valid, key=lambda p: abs(acf_vals[p]))
    if abs(acf_vals[best]) < 0.2:
        return None
    return best


def save_summary_report(models_info, performance_results, store_id):
    """
    분석 결과를 종합한 보고서를 생성합니다.
    
    Args:
        models_info (dict): 모델 정보
        performance_results (dict): 성능 평가 결과
        store_id (int): 매장 ID
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    report = {
        "timestamp": timestamp,
        "store_id": store_id,
        "models": models_info,
        "performance": performance_results,
        "best_model": get_best_model(performance_results)
    }
    
    # JSON 형식으로 저장
    report_path = os.path.join(OUTPUT_DIR, 'results', 'summary_report.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=4)
    
    # HTML 보고서 생성 (간단한 버전)
    html_report = create_html_report(report)
    
    html_path = os.path.join(OUTPUT_DIR, 'results', 'summary_report.html')
    with open(html_path, 'w') as f:
        f.write(html_report)
    
    logger.info(f"분석 보고서가 '{OUTPUT_DIR}/results/' 디렉토리에 저장되었습니다.")


def get_best_model(performance_results):
    """
    성능 지표를 기반으로 최적의 모델을 선택합니다.
    
    Args:
        performance_results (dict): 성능 평가 결과
        
    Returns:
        dict: 최적 모델 정보
    """
    models = list(performance_results.keys())

    # WMAPE 기준으로 최적 모델 선택 (0값에 강건)
    def _score(name):
        m = performance_results[name]
        return m.get('wmape') if m.get('wmape') is not None else m['rmse']
    best_model = min(models, key=_score)
    
    return {
        "name": best_model,
        "metrics": performance_results[best_model]
    }


def create_html_report(report):
    """
    HTML 형식의 보고서를 생성합니다.
    
    Args:
        report (dict): 보고서 데이터
        
    Returns:
        str: HTML 코드
    """
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>시계열 예측 분석 보고서</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1, h2 {{ color: #2c3e50; }}
            table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            tr:nth-child(even) {{ background-color: #f9f9f9; }}
            .best {{ background-color: #d4edda; }}
            img {{ max-width: 100%; height: auto; margin: 10px 0; }}
        </style>
    </head>
    <body>
        <h1>시계열 예측 분석 보고서</h1>
        <p><strong>생성 시간:</strong> {report['timestamp']}</p>
        <p><strong>대상 매장:</strong> Store {report['store_id']}</p>
        
        <h2>최적 모델 (WMAPE 기준)</h2>
        <table>
            <tr>
                <th>모델</th>
                <th>MAE</th>
                <th>RMSE</th>
                <th>SMAPE</th>
                <th>WMAPE</th>
            </tr>
            <tr class="best">
                <td>{report['best_model']['name']}</td>
                <td>{report['best_model']['metrics']['mae']:.2f}</td>
                <td>{report['best_model']['metrics']['rmse']:.2f}</td>
                <td>{report['best_model']['metrics']['smape']:.2f}%</td>
                <td>{report['best_model']['metrics'].get('wmape', float('nan')):.2f}%</td>
            </tr>
        </table>

        <h2>모델 성능 비교</h2>
        <table>
            <tr>
                <th>모델</th>
                <th>MAE</th>
                <th>RMSE</th>
                <th>SMAPE</th>
                <th>WMAPE</th>
            </tr>
    """
    
    # 모든 모델의 성능 테이블 생성
    for model in report['performance']:
        metrics = report['performance'][model]
        is_best = model == report['best_model']['name']
        row_class = 'class="best"' if is_best else ''
        
        html += f"""
            <tr {row_class}>
                <td>{model}</td>
                <td>{metrics['mae']:.2f}</td>
                <td>{metrics['rmse']:.2f}</td>
                <td>{metrics['smape']:.2f}%</td>
                <td>{metrics.get('wmape', float('nan')):.2f}%</td>
            </tr>
        """
    
    html += """
        </table>
        
        <h2>시각화 결과</h2>
        <div>
            <img src="../figures/model_comparison.png" alt="모델 성능 비교">
            <p>모델 성능 비교</p>
        </div>
        <div>
            <img src="../figures/final_model_future_forecast.png" alt="미래 예측 결과">
            <p>미래 예측 결과</p>
        </div>
    </body>
    </html>
    """
    
    return html


def perform_seasonal_decomposition(series, model_name='decomposition'):
    """
    시계열 데이터의 계절성 분해를 수행합니다.
    
    Args:
        series (pandas.Series): 시계열 데이터
        model_name (str): 모델 이름 (파일 저장용)
        
    Returns:
        statsmodels 결과 객체: 분해 결과
    """
    # 결측값이 있으면 제거 또는 보간
    series = series.dropna()
    
    # 계절성 분해
    if isinstance(series.index, pd.DatetimeIndex):
        # 날짜 인덱스가 있으면 해당 빈도 사용
        try:
            result = seasonal_decompose(series, model='additive')
        except:
            # 주기를 명시적으로 설정
            result = seasonal_decompose(series, model='additive', period=52)  # 주간 데이터 가정
    else:
        # 날짜 인덱스가 없으면 주기 설정 필요
        result = seasonal_decompose(series, model='additive', period=52)  # 주간 데이터 가정
    
    # 결과 시각화
    plt.figure(figsize=(15, 12))
    
    plt.subplot(411)
    plt.plot(result.observed)
    plt.title('Original time series')
    plt.grid(True)
    
    plt.subplot(412)
    plt.plot(result.trend)
    plt.title('Trend Component')
    plt.grid(True)
    
    plt.subplot(413)
    plt.plot(result.seasonal)
    plt.title('Seasonal components')
    plt.grid(True)
    
    plt.subplot(414)
    plt.plot(result.resid)
    plt.title('Residual component')
    plt.grid(True)
    
    plt.tight_layout()
    
    # 그림 저장
    save_path = os.path.join(OUTPUT_DIR, 'figures', f'{model_name}_seasonal_decomposition.png')
    plt.savefig(save_path, dpi=DPI)
    plt.close()
    
    return result

