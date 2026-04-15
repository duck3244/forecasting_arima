"""
메인 실행 파일 - ARIMA 기반 판매 예측 분석

주요 구조 (데이터 누수 방지):
  1. 데이터 로드/전처리/EDA → 전체 시리즈 사용 (비파라메트릭)
  2. 훈련/테스트 분할 (조기)
  3. 모든 모델 학습/그리드 서치/CV → 훈련 데이터로만 수행
  4. 테스트 데이터로 홀드아웃 평가
  5. 최종 미래 예측 모델만 전체 데이터로 refit 후 저장
"""

import argparse
import logging
import numpy as np
import pandas as pd

import config
from config import set_seeds, setup_logging
from data_handler import (
    load_data, preprocess_data, select_exog_variables,
    check_stationarity, prepare_train_test_split,
    inverse_transform,
)
from model_selection import (
    grid_search_arima, grid_search_sarima, create_base_model
)
from evaluation import (
    time_series_cv, residual_analysis, evaluate_forecast, compare_models
)
from visualization import (
    plot_time_series, plot_acf_pacf, plot_forecast,
    plot_future_forecast, plot_feature_importance
)
from forecast import (
    ensemble_forecast, predict_future, create_final_model,
    prepare_future_exog
)
from utils import (
    save_summary_report, perform_seasonal_decomposition, detect_seasonal_period
)


def parse_args():
    parser = argparse.ArgumentParser(description="ARIMA 기반 판매 예측 분석")
    parser.add_argument('--data', type=str, default=config.DATA_PATH, help="입력 CSV 경로")
    parser.add_argument('--store', type=int, default=config.STORE_ID, help="분석할 매장 ID")
    parser.add_argument('--steps', type=int, default=config.FORECAST_STEPS, help="예측 기간(주)")
    parser.add_argument('--seed', type=int, default=config.RANDOM_SEED, help="난수 시드")
    parser.add_argument('--auto-seasonal', action='store_true',
                        help="ACF 기반 계절성 주기 자동 감지")
    parser.add_argument('--hybrid', action='store_true',
                        help="GBM_Pred 잔차 하이브리드 모드 (target = Weekly_Sales - GBM_Pred)")
    parser.add_argument('--baseline-preproc', action='store_true',
                        help="옛 전처리 재현(IQR 캡핑 on, 휴일 더미 off)")
    parser.add_argument('--skip-future', action='store_true',
                        help="미래 예측 단계를 건너뛰고 테스트 평가까지만 수행")
    parser.add_argument('--log-level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])
    return parser.parse_args()


def main():
    args = parse_args()

    output_dir = config.create_output_dirs()
    setup_logging(output_dir=output_dir, level=getattr(logging, args.log_level))
    set_seeds(args.seed)
    logger = logging.getLogger(__name__)
    logger.info(f"분석 결과 저장 경로: {output_dir}")

    # 1. 데이터 로드 및 전처리
    logger.info("=== 1. 데이터 로드 및 전처리 ===")
    df = load_data(args.data)
    df = preprocess_data(
        df,
        apply_iqr_cap=args.baseline_preproc,
        add_holidays=not args.baseline_preproc,
    )

    store_data = df[df['Store'] == args.store].copy()
    if 'Date' in store_data.columns:
        store_data.set_index('Date', inplace=True)

    # 2. 외생 변수 선택
    logger.info("=== 2. 외생 변수 선택 ===")
    exog_vars = select_exog_variables(store_data)
    exog = store_data[exog_vars] if exog_vars else None

    # 3. 시계열 데이터 준비
    logger.info("=== 3. 시계열 데이터 준비 ===")
    gbm_pred_series = None
    if args.hybrid:
        if 'GBM_Pred' not in store_data.columns:
            raise ValueError("--hybrid 모드는 'GBM_Pred' 컬럼이 필요합니다.")
        gbm_pred_series = store_data['GBM_Pred']
        sales_series_orig = store_data['Weekly_Sales']
        sales_series = store_data['Weekly_Sales'] - gbm_pred_series
        sales_series.name = 'GBM_Residual'
        transformed = False
        logger.info(f"하이브리드 모드: 잔차 target (mean={sales_series.mean():.2f}, "
                    f"std={sales_series.std():.2f})")
    elif 'Weekly_Sales_Transformed' in store_data.columns:
        sales_series_orig = store_data['Weekly_Sales']
        sales_series = store_data['Weekly_Sales_Transformed']
        transformed = True
        logger.info("로그 변환된 Weekly_Sales를 사용합니다.")
    else:
        sales_series = store_data['Weekly_Sales']
        sales_series_orig = sales_series
        transformed = False

    # 4. EDA 시각화
    logger.info("=== 4. 데이터 시각화 ===")
    plot_time_series(sales_series, "Weekly sales time series", "Date", "Sales", args.store)
    plot_feature_importance(df, target='Weekly_Sales', top_n=10)
    plot_acf_pacf(sales_series, lags=40, title="Weekly_Sales ACF & PACF")
    try:
        perform_seasonal_decomposition(sales_series)
    except Exception as e:
        logger.warning(f"시계열 분해 오류: {e}")

    # 4.5. 계절성 주기 결정
    seasonal_period = config.SEASONAL_PERIOD
    if args.auto_seasonal:
        detected = detect_seasonal_period(sales_series)
        if detected:
            seasonal_period = detected
            logger.info(f"자동 감지된 계절성 주기: {seasonal_period}")
        else:
            logger.info(f"자동 감지 실패, config 기본값 사용: {seasonal_period}")

    # 5. 훈련/테스트 조기 분할 (누수 방지)
    logger.info("=== 5. 훈련/테스트 분할 ===")
    train, test, train_exog, test_exog = prepare_train_test_split(
        sales_series, exog=exog, test_size=args.steps
    )
    # sales_series가 변환/잔차 등으로 원본과 다를 경우 원본 스케일도 따로 분할
    if sales_series is not sales_series_orig:
        train_orig, test_orig, _, _ = prepare_train_test_split(
            sales_series_orig, test_size=args.steps
        )
    else:
        train_orig, test_orig = train, test
    logger.info(f"train={len(train)}, test={len(test)}")

    # 6. 정상성 검정 (훈련 데이터)
    logger.info("=== 6. 정상성 검정 ===")
    is_stationary = check_stationarity(train)
    d = 0 if is_stationary else 1

    # 7. 기본 ARIMA 모델 (훈련 데이터)
    logger.info("=== 7. 기본 ARIMA 모델 학습 ===")
    arima_model = create_base_model(train, exog=train_exog, d=d, s=seasonal_period)

    # 8. 최적 ARIMA 그리드 서치 (훈련 데이터)
    logger.info("=== 8. 최적 ARIMA 모델 탐색 ===")
    best_arima, best_order = grid_search_arima(
        train, exog=train_exog,
        p_range=range(config.MAX_P + 1),
        d_range=[d],
        q_range=range(config.MAX_Q + 1),
    )

    # 9. 최적 SARIMA 그리드 서치 (훈련 데이터)
    logger.info("=== 9. 최적 SARIMA 모델 탐색 ===")
    best_sarima, best_sarima_order, best_seasonal_order = grid_search_sarima(
        train, exog=train_exog,
        p_range=range(config.MAX_P),
        d_range=[d],
        q_range=range(config.MAX_Q),
        P_range=range(config.MAX_P_SEASONAL + 1),
        D_range=range(config.MAX_D_SEASONAL + 1),
        Q_range=range(config.MAX_Q_SEASONAL + 1),
        s=seasonal_period,
    )

    # 10. 교차 검증 (훈련 데이터 내부에서)
    logger.info("=== 10. 교차 검증 ===")
    time_series_cv(train, exog=train_exog, model_type='arima', order=best_order)
    time_series_cv(train, exog=train_exog, model_type='sarima',
                   order=best_sarima_order, seasonal_order=best_seasonal_order)

    # 11. 잔차 분석
    logger.info("=== 11. 잔차 분석 ===")
    residual_analysis(best_arima, model_name='best_arima')
    residual_analysis(best_sarima, model_name='best_sarima')

    # 12. 테스트 홀드아웃 평가
    logger.info("=== 12. 테스트 홀드아웃 평가 ===")
    model_performance = {}
    gbm_test = gbm_pred_series.iloc[-len(test):].values if args.hybrid else None

    def _to_sales_scale(forecast_vals):
        arr = inverse_transform(forecast_vals, store_data, transformed)
        arr = np.asarray(arr.values if hasattr(arr, 'values') else arr, dtype=float)
        if args.hybrid:
            arr = arr + gbm_test
        return arr

    def _eval(model, label, key):
        forecast = model.forecast(steps=len(test), exog=test_exog)
        forecast_orig = _to_sales_scale(forecast)
        metrics = evaluate_forecast(test_orig, forecast_orig, model_name=label)
        model_performance[key] = metrics
        return forecast_orig

    # 하이브리드 모드에서는 GBM 단독 성능도 동일 구간에서 측정하여 비교
    if args.hybrid:
        gbm_metrics = evaluate_forecast(test_orig, gbm_test, model_name='gbm_only')
        model_performance['GBM_Only'] = gbm_metrics

    _eval(arima_model, 'arima', 'ARIMA')
    _eval(best_arima, 'best_arima', 'Best_ARIMA')
    best_sarima_forecast_orig = _eval(best_sarima, 'best_sarima', 'Best_SARIMA')

    # 앙상블
    ensemble_result = ensemble_forecast(
        [arima_model, best_arima, best_sarima],
        steps=len(test), exog=test_exog,
    )
    ensemble_orig = _to_sales_scale(ensemble_result)
    model_performance['Ensemble'] = evaluate_forecast(test_orig, ensemble_orig, model_name='ensemble')

    compare_models(model_performance)

    # 13. 테스트 예측 시각화
    logger.info("=== 13. 예측 시각화 ===")
    history_for_plot = sales_series_orig if transformed else sales_series
    plot_forecast(history_for_plot, best_sarima_forecast_orig,
                  title="SARIMA model prediction results", model_name='best_sarima')
    plot_forecast(history_for_plot, ensemble_orig,
                  title="Ensemble model prediction results", model_name='ensemble')

    # 14. 최종 미래 예측용 모델 (전체 데이터로 refit)
    if args.hybrid or args.skip_future:
        reason = "하이브리드 모드는 GBM_Pred 미래값을 필요로 함" if args.hybrid else "옵션"
        logger.info(f"=== 14. 미래 예측 건너뜀 ({reason}) ===")
        save_summary_report(
            {"order": "n/a", "note": "future skipped"},
            model_performance, args.store,
        )
        logger.info(f"=== 분석 완료. 결과: {output_dir} ===")
        return

    logger.info("=== 14. 미래 예측 (최종 모델 refit) ===")
    final_model = create_final_model(
        sales_series, exog=exog,
        order=best_sarima_order,
        seasonal_order=best_seasonal_order,
    )
    future_exog = prepare_future_exog(exog, steps=args.steps)
    future_mean, future_dates, future_lower, future_upper = predict_future(
        final_model,
        steps=args.steps,
        exog=future_exog,
        series=sales_series,
        store_data=store_data,
        transformed=transformed,
    )

    future_df = pd.DataFrame({
        'Date': future_dates,
        'Predicted_Sales': future_mean,
        'Lower_CI': future_lower,
        'Upper_CI': future_upper,
    })
    logger.info(f"미래 예측 결과:\n{future_df}")

    plot_future_forecast(
        series=history_for_plot,
        forecast=future_mean,
        future_dates=future_dates,
        title=f"Store {args.store} next {args.steps}-weekly forecast",
        model_name='final_model',
        lower=future_lower,
        upper=future_upper,
    )

    # 15. 요약 보고서
    logger.info("=== 15. 최종 보고서 생성 ===")
    models_info = {
        "ARIMA": {"order": "(1,%d,1)" % d},
        "Best_ARIMA": {"order": str(best_order)},
        "Best_SARIMA": {"order": str(best_sarima_order),
                        "seasonal_order": str(best_seasonal_order)},
        "Ensemble": {"models": ["ARIMA", "Best_ARIMA", "Best_SARIMA"]},
        "seasonal_period": seasonal_period,
        "random_seed": args.seed,
    }
    save_summary_report(models_info, model_performance, args.store)

    logger.info(f"=== 분석 완료. 결과: {output_dir} ===")


if __name__ == '__main__':
    main()
