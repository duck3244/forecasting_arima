"""
메인 실행 파일 - ARIMA 기반 판매 예측 분석
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.tsa.seasonal import seasonal_decompose

# 로컬 모듈 임포트
import config
from data_handler import (
    load_data, preprocess_data, select_exog_variables, 
    check_stationarity, prepare_train_test_split, 
    inverse_transform, get_future_dates
)
from model_selection import (
    grid_search_arima, grid_search_sarima, create_base_model
)
from evaluation import (
    time_series_cv, residual_analysis, evaluate_forecast, compare_models
)
from visualization import (
    plot_time_series, plot_acf_pacf, plot_forecast, 
    plot_future_forecast, plot_seasonal_decomposition,
    plot_feature_importance
)
from forecast import (
    ensemble_forecast, predict_future, create_final_model, 
    prepare_future_exog
)
from utils import (
    save_summary_report, get_best_model, perform_seasonal_decomposition
)


def main():
    """
    메인 실행 함수
    """
    # 결과 저장 디렉토리 생성
    output_dir = config.create_output_dirs()
    print(f"분석 결과는 '{output_dir}' 디렉토리에 저장됩니다.")
    
    # 1. 데이터 로드 및 전처리
    print("\n=== 1. 데이터 로드 및 전처리 ===")
    df = load_data(config.DATA_PATH)
    df = preprocess_data(df)
    
    # 특정 매장 데이터로 필터링
    store_id = config.STORE_ID
    store_data = df[df['Store'] == store_id].copy()
    
    # 날짜 인덱스 설정
    if 'Date' in store_data.columns:
        store_data.set_index('Date', inplace=True)
    
    # 2. 외생 변수 선택
    print("\n=== 2. 외생 변수 선택 ===")
    exog_vars = select_exog_variables(store_data)
    exog = store_data[exog_vars] if exog_vars else None
    
    # 3. 시계열 데이터 준비
    print("\n=== 3. 시계열 데이터 준비 ===")
    if 'Weekly_Sales_Transformed' in store_data.columns:
        sales_series_orig = store_data['Weekly_Sales']
        sales_series = store_data['Weekly_Sales_Transformed']
        transformed = True
        print("\n로그 변환된 Weekly_Sales를 사용합니다.")
    else:
        sales_series = store_data['Weekly_Sales']
        sales_series_orig = sales_series
        transformed = False
        print("\n원본 Weekly_Sales를 사용합니다.")
    
    # 4. 데이터 시각화
    print("\n=== 4. 데이터 시각화 ===")
    # 시계열 데이터 플롯
    plot_time_series(sales_series, "Weekly sales time series", "Date", "Sales", store_id)
    
    # 변수 중요도 시각화
    plot_feature_importance(df, target='Weekly_Sales', top_n=10)
    
    # ACF, PACF 플롯
    plot_acf_pacf(sales_series, lags=40, title="Weekly_Sales ACF & PACF")
    
    # 시계열 분해 수행 및 시각화
    try:
        decomposition = perform_seasonal_decomposition(sales_series)
    except Exception as e:
        print(f"시계열 분해 오류: {str(e)}")
    
    # 5. 정상성 검정
    print("\n=== 5. 정상성 검정 ===")
    is_stationary = check_stationarity(sales_series)
    
    # 적절한 차분 차수 결정
    d = 0 if is_stationary else 1
    
    # 6. 기본 ARIMA 모델 학습
    print("\n=== 6. 기본 ARIMA 모델 학습 ===")
    arima_model = create_base_model(sales_series, exog=exog, d=d, s=config.SEASONAL_PERIOD)
    
    # 7. 그리드 서치로 최적 ARIMA 모델 찾기
    print("\n=== 7. 최적 ARIMA 모델 탐색 ===")
    best_arima, best_order = grid_search_arima(
        sales_series,
        exog=exog,
        p_range=range(config.MAX_P + 1),
        d_range=[d],
        q_range=range(config.MAX_Q + 1)
    )
    
    # 8. 그리드 서치로 최적 SARIMA 모델 찾기
    print("\n=== 8. 최적 SARIMA 모델 탐색 ===")
    best_sarima, best_sarima_order, best_seasonal_order = grid_search_sarima(
        sales_series,
        exog=exog,
        p_range=range(config.MAX_P),
        d_range=[d],
        q_range=range(config.MAX_Q),
        P_range=range(config.MAX_P_SEASONAL + 1),
        D_range=range(config.MAX_D_SEASONAL + 1),
        Q_range=range(config.MAX_Q_SEASONAL + 1),
        s=config.SEASONAL_PERIOD
    )
    
    # 9. 시계열 교차 검증
    print("\n=== 9. 모델 교차 검증 ===")
    # ARIMA 모델 교차 검증
    arima_cv_results = time_series_cv(
        sales_series,
        exog=exog,
        model_type='arima',
        order=best_order
    )
    
    # SARIMA 모델 교차 검증
    sarima_cv_results = time_series_cv(
        sales_series,
        exog=exog,
        model_type='sarima',
        order=best_sarima_order,
        seasonal_order=best_seasonal_order
    )
    
    # 10. 잔차 분석
    print("\n=== 10. 모델 잔차 분석 ===")
    residual_analysis(best_arima, model_name='best_arima')
    residual_analysis(best_sarima, model_name='best_sarima')
    
    # 11. 테스트 데이터 분리 및 모델 평가
    print("\n=== 11. 테스트 데이터 분리 및 평가 ===")
    train, test, train_exog, test_exog = prepare_train_test_split(
        sales_series, 
        exog=exog, 
        test_size=config.FORECAST_STEPS
    )
    
    # 변환된 데이터의 경우, 원본 데이터도 분리
    if transformed:
        train_orig, test_orig, _, _ = prepare_train_test_split(
            sales_series_orig, 
            test_size=config.FORECAST_STEPS
        )
    else:
        train_orig, test_orig = train, test
    
    # 12. 앙상블 모델 준비
    print("\n=== 12. 앙상블 모델 준비 ===")
    models = [arima_model, best_arima, best_sarima]
    
    # 13. 개별 모델 성능 평가
    print("\n=== 13. 개별 모델 성능 평가 ===")
    model_performance = {}
    
    # ARIMA 모델 성능
    arima_forecast = arima_model.forecast(steps=len(test), exog=test_exog)
    arima_forecast_orig = inverse_transform(arima_forecast, store_data, transformed)
    
    print("\nARIMA 모델 성능:")
    arima_performance = evaluate_forecast(test_orig, arima_forecast_orig, model_name='arima')
    model_performance['ARIMA'] = arima_performance
    
    # 최적 ARIMA 모델 성능
    best_arima_forecast = best_arima.forecast(steps=len(test), exog=test_exog)
    best_arima_forecast_orig = inverse_transform(best_arima_forecast, store_data, transformed)
    
    print("\n최적 ARIMA 모델 성능:")
    best_arima_performance = evaluate_forecast(test_orig, best_arima_forecast_orig, model_name='best_arima')
    model_performance['Best_ARIMA'] = best_arima_performance
    
    # 최적 SARIMA 모델 성능
    best_sarima_forecast = best_sarima.forecast(steps=len(test), exog=test_exog)
    best_sarima_forecast_orig = inverse_transform(best_sarima_forecast, store_data, transformed)
    
    print("\n최적 SARIMA 모델 성능:")
    best_sarima_performance = evaluate_forecast(test_orig, best_sarima_forecast_orig, model_name='best_sarima')
    model_performance['Best_SARIMA'] = best_sarima_performance
    
    # 14. 앙상블 예측 및 평가
    print("\n=== 14. 앙상블 예측 및 평가 ===")
    ensemble_result = ensemble_forecast(models, steps=len(test), exog=test_exog)
    ensemble_orig = inverse_transform(ensemble_result, store_data, transformed)
    
    print("\n앙상블 모델 성능:")
    ensemble_performance = evaluate_forecast(test_orig, ensemble_orig, model_name='ensemble')
    model_performance['Ensemble'] = ensemble_performance
    
    # 모델 성능 비교
    compare_models(model_performance)
    
    # 15. 예측 시각화
    print("\n=== 15. 예측 시각화 ===")
    plot_forecast(
        series=sales_series_orig if transformed else sales_series,
        forecast=best_sarima_forecast_orig,
        title="SARIMA model prediction results",
        model_name='best_sarima'
    )
    
    plot_forecast(
        series=sales_series_orig if transformed else sales_series,
        forecast=ensemble_orig,
        title="Ensemble model prediction results",
        model_name='ensemble'
    )
    
    # 16. 미래 예측
    print("\n=== 16. 미래 예측 ===")
    # 전체 데이터로 최종 모델 학습
    final_model = create_final_model(
        sales_series,
        exog=exog,
        order=best_sarima_order,
        seasonal_order=best_seasonal_order
    )
    
    # 미래 외생 변수 준비
    future_exog = prepare_future_exog(exog, steps=config.FORECAST_STEPS)
    
    # 미래 예측 수행
    future_forecast_orig, future_dates = predict_future(
        final_model,
        steps=config.FORECAST_STEPS,
        exog=future_exog,
        series=sales_series,
        store_data=store_data,
        transformed=transformed
    )
    
    print("\n미래 예측 결과:")
    future_df = pd.DataFrame({
        'Date': future_dates,
        'Predicted_Sales': future_forecast_orig
    })
    print(future_df)
    
    # 미래 예측 시각화
    plot_future_forecast(
        series=sales_series_orig if transformed else sales_series,
        forecast=future_forecast_orig,
        future_dates=future_dates,
        title=f"Store {store_id} In the future {config.FORECAST_STEPS}weekly sales forecast",
        model_name='final_model'
    )
    
    # 17. 최종 보고서 생성
    print("\n=== 17. 최종 보고서 생성 ===")
    models_info = {
        "ARIMA": {"order": "(1,1,1)"},
        "Best_ARIMA": {"order": str(best_order)},
        "Best_SARIMA": {"order": str(best_sarima_order), "seasonal_order": str(best_seasonal_order)},
        "Ensemble": {"models": ["ARIMA", "Best_ARIMA", "Best_SARIMA"]}
    }
    
    save_summary_report(models_info, model_performance, store_id)
    
    print("\n=== 분석 완료 ===")
    print(f"모든 결과는 '{output_dir}' 디렉토리에 저장되었습니다.")


if __name__ == '__main__':
    main()

