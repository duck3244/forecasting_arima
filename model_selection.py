"""
모델 선택 및 파라미터 그리드 서치 기능
"""

import os
import logging
import joblib
import numpy as np

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

from config import MAX_SARIMA_TRIALS, OUTPUT_DIR

logger = logging.getLogger(__name__)

_FIT_EXCEPTIONS = (ValueError, np.linalg.LinAlgError, RuntimeError, MemoryError)


def grid_search_arima(series, exog=None, p_range=range(3), d_range=range(2), q_range=range(3)):
    """
    그리드 서치를 통해 최적의 ARIMA 파라미터를 찾습니다.
    
    Args:
        series (pandas.Series): 시계열 데이터
        exog (pandas.DataFrame, optional): 외생 변수 데이터
        p_range (range): AR 차수 범위
        d_range (range): 차분 차수 범위
        q_range (range): MA 차수 범위
        
    Returns:
        tuple: (최적 모델, 최적 파라미터)
    """
    best_aic = float('inf')
    best_order = None
    best_model = None
    results_dict = {}  # 모든 모델 결과 저장

    logger.info("ARIMA 그리드 서치 시작")

    for p in p_range:
        for d in d_range:
            for q in q_range:
                try:
                    results = ARIMA(series, exog=exog, order=(p, d, q)).fit()
                    model_name = f"ARIMA({p},{d},{q})"
                    results_dict[model_name] = {
                        "order": (p, d, q),
                        "aic": results.aic,
                        "bic": results.bic,
                    }
                    logger.debug(f"{model_name} AIC={results.aic:.2f} BIC={results.bic:.2f}")
                    if results.aic < best_aic:
                        best_aic = results.aic
                        best_order = (p, d, q)
                        best_model = results
                except _FIT_EXCEPTIONS as e:
                    logger.warning(f"ARIMA({p},{d},{q}) 학습 실패: {e}")

    if best_model is not None:
        logger.info(f"최적 ARIMA 모델: {best_order} (AIC: {best_aic:.2f})")
        joblib.dump(best_model, os.path.join(OUTPUT_DIR, 'models', f'best_arima_{best_order}.pkl'))
        joblib.dump(results_dict, os.path.join(OUTPUT_DIR, 'results', 'arima_grid_search_results.pkl'))
    else:
        logger.warning("유효한 ARIMA 모델을 찾지 못했습니다. 기본 모델을 사용합니다.")
        best_model = ARIMA(series, exog=exog, order=(1, 1, 1)).fit()
        best_order = (1, 1, 1)
    
    return best_model, best_order


def grid_search_sarima(series, exog=None, p_range=range(3), d_range=range(2), q_range=range(3),
                        P_range=range(2), D_range=range(2), Q_range=range(2), s=12):
    """
    그리드 서치를 통해 최적의 SARIMA 파라미터를 찾습니다.
    
    Args:
        series (pandas.Series): 시계열 데이터
        exog (pandas.DataFrame, optional): 외생 변수 데이터
        p_range (range): AR 차수 범위
        d_range (range): 차분 차수 범위
        q_range (range): MA 차수 범위
        P_range (range): 계절성 AR 차수 범위
        D_range (range): 계절성 차분 차수 범위
        Q_range (range): 계절성 MA 차수 범위
        s (int): 계절성 주기
        
    Returns:
        tuple: (최적 모델, 최적 비계절성 파라미터, 최적 계절성 파라미터)
    """
    best_aic = float('inf')
    best_order = None
    best_seasonal_order = None
    best_model = None
    results_dict = {}  # 모든 모델 결과 저장

    logger.info("SARIMA 그리드 서치 시작 (모든 조합 탐색은 오래 걸릴 수 있음)")

    max_trials = MAX_SARIMA_TRIALS
    trial_count = 0

    for p in p_range:
        for d in d_range:
            for q in q_range:
                for P in P_range:
                    for D in D_range:
                        for Q in Q_range:
                            if trial_count >= max_trials:
                                break
                            try:
                                results = SARIMAX(
                                    series, exog=exog,
                                    order=(p, d, q),
                                    seasonal_order=(P, D, Q, s),
                                ).fit(disp=False)
                                model_name = f"SARIMA({p},{d},{q})({P},{D},{Q},{s})"
                                results_dict[model_name] = {
                                    "order": (p, d, q),
                                    "seasonal_order": (P, D, Q, s),
                                    "aic": results.aic,
                                    "bic": results.bic,
                                }
                                logger.debug(f"{model_name} AIC={results.aic:.2f} BIC={results.bic:.2f}")
                                if results.aic < best_aic:
                                    best_aic = results.aic
                                    best_order = (p, d, q)
                                    best_seasonal_order = (P, D, Q, s)
                                    best_model = results
                                trial_count += 1
                            except _FIT_EXCEPTIONS as e:
                                logger.warning(f"SARIMA({p},{d},{q})({P},{D},{Q},{s}) 학습 실패: {e}")

    if best_model is not None:
        logger.info(f"최적 SARIMA 모델: {best_order}x{best_seasonal_order} (AIC: {best_aic:.2f})")
        joblib.dump(best_model,
                    os.path.join(OUTPUT_DIR, 'models', f'best_sarima_{best_order}x{best_seasonal_order}.pkl'))
        joblib.dump(results_dict, os.path.join(OUTPUT_DIR, 'results', 'sarima_grid_search_results.pkl'))
    else:
        logger.warning("유효한 SARIMA 모델을 찾지 못했습니다. 기본 모델을 사용합니다.")
        best_model = SARIMAX(series, exog=exog, order=(1, 1, 1), seasonal_order=(1, 0, 1, s)).fit(disp=False)
        best_order = (1, 1, 1)
        best_seasonal_order = (1, 0, 1, s)
    
    return best_model, best_order, best_seasonal_order


def create_base_model(series, exog=None, d=1, s=12):
    """
    기본 ARIMA 모델을 생성합니다.
    
    Args:
        series (pandas.Series): 시계열 데이터
        exog (pandas.DataFrame, optional): 외생 변수 데이터
        d (int): 차분 차수
        s (int): 계절성 주기
        
    Returns:
        statsmodels 모델: 학습된 ARIMA 모델
    """
    logger.info("기본 ARIMA 모델 학습")
    arima_model = ARIMA(series, exog=exog, order=(1, d, 1)).fit()
    logger.debug(arima_model.summary())
    
    # 모델 저장
    model_path = os.path.join(OUTPUT_DIR, 'models', 'base_arima_model.pkl')
    joblib.dump(arima_model, model_path)
    
    return arima_model

