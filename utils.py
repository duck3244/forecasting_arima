"""
유틸리티 함수
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from datetime import datetime
from statsmodels.tsa.seasonal import seasonal_decompose

from config import OUTPUT_DIR, DPI


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
    
    print(f"\n분석 보고서가 '{OUTPUT_DIR}/results/' 디렉토리에 저장되었습니다.")


def get_best_model(performance_results):
    """
    성능 지표를 기반으로 최적의 모델을 선택합니다.
    
    Args:
        performance_results (dict): 성능 평가 결과
        
    Returns:
        dict: 최적 모델 정보
    """
    models = list(performance_results.keys())
    
    # RMSE 기준으로 최적 모델 선택
    best_model = min(models, key=lambda x: performance_results[x]['rmse'])
    
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
        
        <h2>최적 모델</h2>
        <table>
            <tr>
                <th>모델</th>
                <th>MAE</th>
                <th>RMSE</th>
                <th>SMAPE</th>
            </tr>
            <tr class="best">
                <td>{report['best_model']['name']}</td>
                <td>{report['best_model']['metrics']['mae']:.2f}</td>
                <td>{report['best_model']['metrics']['rmse']:.2f}</td>
                <td>{report['best_model']['metrics']['smape']:.2f}%</td>
            </tr>
        </table>
        
        <h2>모델 성능 비교</h2>
        <table>
            <tr>
                <th>모델</th>
                <th>MAE</th>
                <th>RMSE</th>
                <th>SMAPE</th>
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

