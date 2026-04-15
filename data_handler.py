"""
데이터 로드 및 전처리 기능
"""

import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.tsa.stattools import adfuller, kpss

from config import (
    FIGURE_SIZE, DPI, OUTPUT_DIR,
    OUTLIER_IQR_MULTIPLIER, EXOG_CORR_THRESHOLD, EXOG_NAN_RATIO_MAX,
)

logger = logging.getLogger(__name__)


def load_data(file_path):
    """CSV 파일을 로드하고 기본 정보를 로그로 출력합니다."""
    df = pd.read_csv(file_path)
    logger.info(f"데이터 로드 완료: {df.shape[0]} 행, {df.shape[1]} 열")
    return df


HOLIDAY_WEEKS = {
    'Is_SuperBowl': {6},
    'Is_LaborDay': {36},
    'Is_Thanksgiving': {47},
    'Is_Christmas': {51, 52},
}


def _add_holiday_dummies(df):
    """
    WeekOfYear를 이용해 소매 매출에 강하게 영향을 주는 주요 휴일을
    0/1 더미로 생성합니다.
    """
    if 'WeekOfYear' not in df.columns:
        logger.warning("WeekOfYear 컬럼이 없어 휴일 더미를 추가하지 못했습니다.")
        return df
    woy = df['WeekOfYear']
    for name, weeks in HOLIDAY_WEEKS.items():
        df[name] = woy.isin(weeks).astype(int)
    logger.info(f"휴일 더미 추가: {list(HOLIDAY_WEEKS.keys())}")
    return df


def preprocess_data(df, apply_iqr_cap=False, add_holidays=True):
    """
    데이터 전처리를 수행합니다.

    Args:
        df: 원본 데이터프레임
        apply_iqr_cap: True이면 Weekly_Sales에 IQR 캡핑 적용 (블랙프라이데이 등 중요
            피크 신호를 지우므로 기본값은 False). 베이스라인 재현용 옵션.
        add_holidays: 휴일 더미 컬럼 추가 여부.
    """
    for col in df.select_dtypes(include=[np.number]).columns:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].interpolate(method='linear')

    if 'Date' not in df.columns and 'Month' in df.columns and 'DayOfYear' in df.columns:
        df['Date'] = pd.to_datetime('2022-01-01') + pd.to_timedelta(df['DayOfYear'] - 1, unit='d')

    if apply_iqr_cap and 'Weekly_Sales' in df.columns:
        q1 = df['Weekly_Sales'].quantile(0.25)
        q3 = df['Weekly_Sales'].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - OUTLIER_IQR_MULTIPLIER * iqr
        upper = q3 + OUTLIER_IQR_MULTIPLIER * iqr
        n_clip = ((df['Weekly_Sales'] < lower) | (df['Weekly_Sales'] > upper)).sum()
        df.loc[df['Weekly_Sales'] < lower, 'Weekly_Sales'] = lower
        df.loc[df['Weekly_Sales'] > upper, 'Weekly_Sales'] = upper
        logger.info(f"IQR 캡핑 적용: {n_clip}개 관측치를 경계값으로 치환")

    if add_holidays:
        df = _add_holiday_dummies(df)

    # 로그 변환
    if 'Weekly_Sales' in df.columns:
        min_val = df['Weekly_Sales'].min()
        if min_val <= 0:
            df['Weekly_Sales_Transformed'] = np.log1p(df['Weekly_Sales'] + abs(min_val) + 1)
            logger.info("음수/0 값이 포함되어 log1p 변환(오프셋)을 적용했습니다.")
        else:
            df['Weekly_Sales_Transformed'] = np.log(df['Weekly_Sales'])
            logger.info("로그 변환을 적용했습니다.")

    return df


def select_exog_variables(df):
    """ARIMAX 모델에 사용할 외생 변수를 선택합니다."""
    potential = ['Temperature', 'Fuel_Price', 'CPI', 'Unemployment',
                 'MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5',
                 'Is_SuperBowl', 'Is_LaborDay', 'Is_Thanksgiving', 'Is_Christmas']
    exog_vars = [v for v in potential if v in df.columns]
    forced_keep = {'Is_SuperBowl', 'Is_LaborDay', 'Is_Thanksgiving', 'Is_Christmas'}

    if exog_vars:
        exog_vars = [v for v in exog_vars if df[v].isnull().sum() / len(df) < EXOG_NAN_RATIO_MAX]

        if 'Weekly_Sales' in df.columns and exog_vars:
            correlations = df[exog_vars + ['Weekly_Sales']].corr()['Weekly_Sales'].abs().sort_values(ascending=False)
            logger.info(f"외생 변수와 Weekly_Sales의 상관관계:\n{correlations}")

            plt.figure(figsize=FIGURE_SIZE)
            corr_matrix = df[exog_vars + ['Weekly_Sales']].corr()
            plt.imshow(corr_matrix, cmap='coolwarm', interpolation='none', aspect='auto')
            plt.colorbar()
            plt.xticks(range(len(corr_matrix.columns)), corr_matrix.columns, rotation=45)
            plt.yticks(range(len(corr_matrix.columns)), corr_matrix.columns)
            for i in range(len(corr_matrix.columns)):
                for j in range(len(corr_matrix.columns)):
                    plt.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}', ha="center", va="center", color="black")
            plt.title('Heatmap of correlations between variables')
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_DIR, 'figures', 'correlation_heatmap.png'), dpi=DPI)
            plt.close()

            # 휴일 더미는 상관관계가 낮아도 도메인 지식상 유지 (희귀 이벤트)
            exog_vars = [v for v in exog_vars
                         if v in forced_keep or abs(correlations[v]) >= EXOG_CORR_THRESHOLD]

    logger.info(f"선택된 외생 변수: {exog_vars}")
    return exog_vars


def check_stationarity(series):
    """
    ADF + KPSS 합의 기반 정상성 판정.
    - ADF H0: 단위근 있음(비정상). p<=0.05 → 정상
    - KPSS H0: 정상. p>0.05 → 정상
    - 두 검정이 모두 정상일 때만 정상으로 판정
    """
    series = series.dropna()
    adf_stat, adf_p, *_ = adfuller(series)
    adf_stationary = adf_p <= 0.05
    logger.info(f"ADF stat={adf_stat:.4f}, p={adf_p:.4f} → {'정상' if adf_stationary else '비정상'}")

    kpss_stationary = True
    try:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            kpss_stat, kpss_p, *_ = kpss(series, regression='c', nlags='auto')
        kpss_stationary = kpss_p > 0.05
        logger.info(f"KPSS stat={kpss_stat:.4f}, p={kpss_p:.4f} → {'정상' if kpss_stationary else '비정상'}")
    except Exception as e:
        logger.warning(f"KPSS 검정 실패: {e}")

    is_stationary = adf_stationary and kpss_stationary
    logger.info(f"정상성 합의 판정: {is_stationary}")
    return is_stationary


def prepare_train_test_split(series, exog=None, test_size=12):
    """시계열 데이터를 훈련/테스트로 분할."""
    if test_size <= 0 or test_size >= len(series):
        raise ValueError(f"유효하지 않은 test_size={test_size} (series len={len(series)})")
    train_size = len(series) - test_size
    train = series.iloc[:train_size]
    test = series.iloc[train_size:]

    train_exog = test_exog = None
    if exog is not None:
        train_exog = exog.iloc[:train_size]
        test_exog = exog.iloc[train_size:]

    return train, test, train_exog, test_exog


def inverse_transform(values, store_data, transformed=True):
    """변환된 예측값을 원 스케일로 역변환."""
    if not transformed:
        return values

    if (store_data['Weekly_Sales'] <= 0).any():
        min_val = store_data['Weekly_Sales'].min()
        return np.expm1(values) - abs(min_val) - 1
    return np.exp(values)


def get_future_dates(last_date, steps=12):
    """마지막 날짜 이후 미래 날짜 생성."""
    if isinstance(last_date, pd.Timestamp):
        return pd.date_range(start=last_date + pd.Timedelta(days=7), periods=steps, freq='W')
    return range(steps)
