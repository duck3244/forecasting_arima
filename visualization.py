"""
데이터 및 결과 시각화 기능
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

from config import FIGURE_SIZE, DPI, OUTPUT_DIR


def plot_time_series(series, title="Timeseries data", xlabel="Time", ylabel="Value", store_id=None):
    """
    시계열 데이터를 시각화합니다.
    
    Args:
        series (pandas.Series): 시계열 데이터
        title (str): 그래프 제목
        xlabel (str): x축 레이블
        ylabel (str): y축 레이블
        store_id (int, optional): 매장 ID
    """
    plt.figure(figsize=FIGURE_SIZE)
    series.plot()
    
    if store_id:
        plt.title(f'{title} (Store {store_id})')
    else:
        plt.title(title)
    
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    
    # 그림 저장
    filename = title.lower().replace(' ', '_')
    if store_id:
        filename += f'_store_{store_id}'
    
    save_path = os.path.join(OUTPUT_DIR, 'figures', f'{filename}.png')
    plt.savefig(save_path, dpi=DPI)
    plt.close()


def plot_acf_pacf(series, lags=40, title="ACF & PACF"):
    """
    시계열 데이터의 ACF와 PACF를 시각화합니다.
    
    Args:
        series (pandas.Series): 시계열 데이터
        lags (int): 최대 지연(lag) 수
        title (str): 그래프 제목
    """
    plt.figure(figsize=FIGURE_SIZE)
    
    plt.subplot(211)
    plot_acf(series.dropna(), lags=lags, ax=plt.gca())
    plt.title('Auto correlation function (ACF)')
    
    plt.subplot(212)
    plot_pacf(series.dropna(), lags=lags, ax=plt.gca())
    plt.title('Partial auto correlation function (PACF)')
    
    plt.suptitle(title)
    plt.tight_layout()
    
    # 그림 저장
    filename = title.lower().replace(' ', '_')
    save_path = os.path.join(OUTPUT_DIR, 'figures', f'{filename}.png')
    plt.savefig(save_path, dpi=DPI)
    plt.close()


def plot_forecast(series, forecast, title="Forecast Timeseries", confidence_interval=None, model_name='model'):
    """
    예측 결과를 시각화합니다.
    
    Args:
        series (pandas.Series): 원본 시계열 데이터
        forecast: 예측 데이터
        title (str): 그래프 제목
        confidence_interval (pandas.DataFrame, optional): 신뢰 구간
        model_name (str): 모델 이름 (파일 저장용)
    """
    plt.figure(figsize=FIGURE_SIZE)

    # 원본 데이터
    plt.plot(series.index, series, label='Real data', color='blue')

    # 예측 데이터의 인덱스 생성
    if isinstance(forecast, pd.Series):
        forecast_idx = forecast.index
    else:
        # 예측 날짜 생성 (마지막 관측 날짜 이후)
        last_date = series.index[-1]
        if isinstance(last_date, pd.Timestamp):
            # 주간 데이터로 가정
            forecast_idx = pd.date_range(start=last_date + pd.Timedelta(days=7),
                                        periods=len(forecast), freq='W')
        else:
            forecast_idx = np.arange(len(series), len(series) + len(forecast))

    # 예측 데이터
    plt.plot(forecast_idx, forecast, label='Forecast', color='red', linestyle='--')

    # 신뢰 구간이 있는 경우
    if confidence_interval is not None:
        plt.fill_between(forecast_idx,
                        confidence_interval.iloc[:, 0],
                        confidence_interval.iloc[:, 1],
                        color='pink', alpha=0.3)

    plt.title(title, fontsize=15)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    # 그림 저장
    filename = f'{model_name}_forecast'
    save_path = os.path.join(OUTPUT_DIR, 'figures', f'{filename}.png')
    plt.savefig(save_path, dpi=DPI)
    plt.close()


def plot_future_forecast(series, forecast, future_dates, title="Forecast",
                         model_name='final_model', lower=None, upper=None):
    """
    미래 예측 결과를 시각화합니다.
    
    Args:
        series (pandas.Series): 원본 시계열 데이터
        forecast: 미래 예측 데이터
        future_dates: 미래 날짜
        title (str): 그래프 제목
        model_name (str): 모델 이름 (파일 저장용)
    """
    plt.figure(figsize=FIGURE_SIZE)

    # 원본 데이터
    plt.plot(series.index, series, label='Historical data', color='blue')

    # 미래 예측
    plt.plot(future_dates, forecast, label='Forecast', color='red', linestyle='--')

    # 신뢰 구간 음영
    if lower is not None and upper is not None:
        lower_arr = np.asarray(lower)
        upper_arr = np.asarray(upper)
        if not (np.all(np.isnan(lower_arr)) or np.all(np.isnan(upper_arr))):
            plt.fill_between(future_dates, lower_arr, upper_arr,
                             color='red', alpha=0.15, label='95% CI')

    # 그림자 강조 표시 (미래 영역)
    min_val = min(series.min(), forecast.min() if hasattr(forecast, 'min') else min(forecast))
    max_val = max(series.max(), forecast.max() if hasattr(forecast, 'max') else max(forecast))
    
    # 미래 영역에 음영 추가
    if isinstance(future_dates[0], pd.Timestamp):
        plt.axvspan(future_dates[0], future_dates[-1], color='gray', alpha=0.1)

    plt.title(title, fontsize=15)
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    # 그림 저장
    filename = f'{model_name}_future_forecast'
    save_path = os.path.join(OUTPUT_DIR, 'figures', f'{filename}.png')
    plt.savefig(save_path, dpi=DPI)
    plt.close()


def plot_seasonal_decomposition(result, title="Decomposition", model_name='decomposition'):
    """
    시계열 분해 결과를 시각화합니다.
    
    Args:
        result: 분해 결과 객체
        title (str): 그래프 제목
        model_name (str): 모델 이름 (파일 저장용)
    """
    plt.figure(figsize=(15, 12))
    
    # 원본 데이터
    plt.subplot(4, 1, 1)
    plt.plot(result.observed)
    plt.title('Original data')
    plt.grid(True)
    
    # 추세 성분
    plt.subplot(4, 1, 2)
    plt.plot(result.trend)
    plt.title('Trend component')
    plt.grid(True)
    
    # 계절성 성분
    plt.subplot(4, 1, 3)
    plt.plot(result.seasonal)
    plt.title('Seasonal components')
    plt.grid(True)
    
    # 잔차 성분
    plt.subplot(4, 1, 4)
    plt.plot(result.resid)
    plt.title('Residual component')
    plt.grid(True)
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    # 그림 저장
    filename = f'{model_name}_decomposition'
    save_path = os.path.join(OUTPUT_DIR, 'figures', f'{filename}.png')
    plt.savefig(save_path, dpi=DPI)
    plt.close()


def plot_feature_importance(df, target='Weekly_Sales', top_n=10, title="Variable importance"):
    """
    변수 중요도를 시각화합니다.
    
    Args:
        df (pandas.DataFrame): 데이터프레임
        target (str): 타겟 변수
        top_n (int): 표시할 상위 변수 개수
        title (str): 그래프 제목
    """
    import logging as _lg
    if target not in df.columns:
        _lg.getLogger(__name__).warning(f"{target} column not found in DataFrame")
        return
    
    # 모든 수치형 변수에 대한 상관관계 계산
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    correlations = df[numeric_cols].corr()[target].abs().sort_values(ascending=False)
    
    # 타겟 변수 자신은 제외
    correlations = correlations.drop(target, errors='ignore')
    
    # 상위 N개 변수 선택
    top_corr = correlations.head(top_n)
    
    plt.figure(figsize=FIGURE_SIZE)
    plt.barh(range(len(top_corr)), top_corr.values, align='center')
    plt.yticks(range(len(top_corr)), top_corr.index)
    plt.xlabel('Correlation coefficient (absolute)')
    plt.title(title)
    plt.grid(True, axis='x')
    plt.tight_layout()
    
    # 그림 저장
    filename = 'feature_importance'
    save_path = os.path.join(OUTPUT_DIR, 'figures', f'{filename}.png')
    plt.savefig(save_path, dpi=DPI)
    plt.close()
