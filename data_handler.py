"""
데이터 로드 및 전처리 기능
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.tsa.stattools import adfuller

from config import FIGURE_SIZE, DPI, OUTPUT_DIR


def load_data(file_path):
    """
    CSV 파일을 로드하고 기본 정보를 출력합니다.
    
    Args:
        file_path (str): 데이터 파일 경로
        
    Returns:
        pandas.DataFrame: 로드된 데이터프레임
    """
    df = pd.read_csv(file_path)
    print(f"데이터 로드 완료: {df.shape[0]} 행, {df.shape[1]} 열")
    return df


def preprocess_data(df):
    """
    데이터 전처리를 수행합니다.
    
    Args:
        df (pandas.DataFrame): 원본 데이터프레임
        
    Returns:
        pandas.DataFrame: 전처리된 데이터프레임
    """
    # 결측치 처리
    for col in df.select_dtypes(include=[np.number]).columns:
        if df[col].isnull().sum() > 0:
            # 시계열 데이터는 보간법으로 채우는 것이 좋음
            df[col] = df[col].interpolate(method='time')
    
    # 날짜 처리
    if 'Date' not in df.columns and 'Month' in df.columns and 'DayOfYear' in df.columns:
        # 년도 정보가 없다면 2022년으로 가정
        df['Date'] = pd.to_datetime('2022-01-01') + pd.to_timedelta(df['DayOfYear'] - 1, unit='d')
    
    # 이상치 탐지 및 처리 (IQR 방법)
    for col in ['Weekly_Sales']:
        if col in df.columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # 이상치를 경계값으로 대체
            df.loc[df[col] < lower_bound, col] = lower_bound
            df.loc[df[col] > upper_bound, col] = upper_bound
    
    # 로그 변환 (양수 값에만 적용)
    if 'Weekly_Sales' in df.columns:
        if (df['Weekly_Sales'] <= 0).any():
            # 로그 변환 대신 박스-콕스 변환 또는 상수 추가 후 로그 변환
            min_val = df['Weekly_Sales'].min()
            if min_val <= 0:
                df['Weekly_Sales_Transformed'] = np.log1p(df['Weekly_Sales'] + abs(min_val) + 1)
                print("음수 또는 0 값이 포함되어 있어 log1p 변환을 적용했습니다.")
            else:
                df['Weekly_Sales_Transformed'] = np.log(df['Weekly_Sales'])
                print("로그 변환을 적용했습니다.")
        else:
            df['Weekly_Sales_Transformed'] = np.log(df['Weekly_Sales'])
            print("로그 변환을 적용했습니다.")
    
    return df


def select_exog_variables(df):
    """
    ARIMAX 모델에 사용할 외생 변수를 선택합니다.
    
    Args:
        df (pandas.DataFrame): 데이터프레임
        
    Returns:
        list: 선택된 외생 변수 리스트
    """
    potential_exog_vars = ['Temperature', 'Fuel_Price', 'CPI', 'Unemployment', 'MarkDown1', 'MarkDown2', 
                          'MarkDown3', 'MarkDown4', 'MarkDown5']
    
    # 사용 가능한 변수만 필터링
    exog_vars = [var for var in potential_exog_vars if var in df.columns]
    
    if exog_vars:
        # 결측치가 많은 변수 제외
        exog_vars = [var for var in exog_vars if df[var].isnull().sum() / len(df) < 0.1]
        
        # 상관관계 확인 (Weekly_Sales와 상관관계가 높은 변수만 선택)
        if 'Weekly_Sales' in df.columns and exog_vars:
            correlations = df[exog_vars + ['Weekly_Sales']].corr()['Weekly_Sales'].abs().sort_values(ascending=False)
            print("\n외생 변수와 Weekly_Sales의 상관관계:")
            print(correlations)
            
            # 시각화 - 상관관계 히트맵
            plt.figure(figsize=FIGURE_SIZE)
            corr_matrix = df[exog_vars + ['Weekly_Sales']].corr()
            plt.imshow(corr_matrix, cmap='coolwarm', interpolation='none', aspect='auto')
            plt.colorbar()
            plt.xticks(range(len(corr_matrix.columns)), corr_matrix.columns, rotation=45)
            plt.yticks(range(len(corr_matrix.columns)), corr_matrix.columns)
            
            # 히트맵에 상관계수 표시
            for i in range(len(corr_matrix.columns)):
                for j in range(len(corr_matrix.columns)):
                    text = plt.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                                   ha="center", va="center", color="black")
            
            plt.title('Heatmap of correlations between variables')
            plt.tight_layout()
            
            # 그림 저장
            save_path = os.path.join(OUTPUT_DIR, 'figures', 'correlation_heatmap.png')
            plt.savefig(save_path, dpi=DPI)
            plt.close()
            
            # 상관관계가 0.1 이상인 변수만 선택
            exog_vars = [var for var in exog_vars if abs(correlations[var]) >= 0.1]
    
    print(f"\n선택된 외생 변수: {exog_vars}")
    return exog_vars


def check_stationarity(series):
    """
    시계열 데이터의 정상성을 검사합니다.
    
    Args:
        series (pandas.Series): 시계열 데이터
        
    Returns:
        bool: 정상성 여부 (True/False)
    """
    result = adfuller(series.dropna())
    print("\n=== ADF 테스트 결과 ===")
    print(f'ADF 통계량: {result[0]}')
    print(f'p-value: {result[1]}')
    print('임계값:')
    for key, value in result[4].items():
        print(f'\t{key}: {value}')
    
    # p-value가 0.05보다 작으면 정상성을 가진다고 판단
    if result[1] <= 0.05:
        print("결론: 정상 시계열 (차분이 필요하지 않음)")
        return True
    else:
        print("결론: 비정상 시계열 (차분이 필요함)")
        return False


def prepare_train_test_split(series, exog=None, test_size=12):
    """
    시계열 데이터를 훈련 세트와 테스트 세트로 분할합니다.
    
    Args:
        series (pandas.Series): 시계열 데이터
        exog (pandas.DataFrame, optional): 외생 변수 데이터
        test_size (int, optional): 테스트 세트 크기
        
    Returns:
        tuple: (train, test, train_exog, test_exog)
    """
    train_size = len(series) - test_size
    train = series[:train_size]
    test = series[train_size:]
    
    train_exog = None
    test_exog = None
    
    if exog is not None:
        train_exog = exog[:train_size]
        test_exog = exog[train_size:]
    
    return train, test, train_exog, test_exog


def inverse_transform(values, store_data, transformed=True):
    """
    변환된 데이터를 원래 스케일로 되돌립니다.
    
    Args:
        values: 변환할 값(들)
        store_data (pandas.DataFrame): 원본 데이터
        transformed (bool): 변환 여부
        
    Returns:
        변환된 값(들)
    """
    if not transformed:
        return values
    
    if (store_data['Weekly_Sales'] <= 0).any():
        # log1p 변환을 사용한 경우
        min_val = store_data['Weekly_Sales'].min()
        return np.expm1(values) - abs(min_val) - 1
    else:
        # 일반 로그 변환을 사용한 경우
        return np.exp(values)


def get_future_dates(last_date, steps=12):
    """
    마지막 날짜 이후로 미래 날짜를 생성합니다.
    
    Args:
        last_date: 마지막 날짜
        steps (int): 생성할 날짜 수
        
    Returns:
        pandas.DatetimeIndex: 미래 날짜
    """
    if isinstance(last_date, pd.Timestamp):
        # 주간 데이터로 가정
        return pd.date_range(start=last_date + pd.Timedelta(days=7), periods=steps, freq='W')
    else:
        return range(len(last_date), len(last_date) + steps)

