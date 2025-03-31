"""
설정 파일 - ARIMA 모델 및 전반적인 파라미터 설정
"""

import os

from datetime import datetime


# 데이터 설정
DATA_PATH = 'sales.csv'
STORE_ID = 1  # 분석할 매장 ID

# 모델 파라미터 설정
FORECAST_STEPS = 12  # 예측 기간 (주)
SEASONAL_PERIOD = 52  # 계절성 주기 (주간 데이터: 52, 월간 데이터: 12)

# 그리드 서치 설정
MAX_P = 3
MAX_D = 2
MAX_Q = 3
MAX_P_SEASONAL = 1
MAX_D_SEASONAL = 1
MAX_Q_SEASONAL = 1
MAX_SARIMA_TRIALS = 20  # SARIMA 그리드 서치 최대 시도 횟수

# 교차 검증 설정
CV_SPLITS = 5

# 시각화 설정
FIGURE_SIZE = (15, 8)
DPI = 300
SAVE_FIGURES = True

# 저장 경로 설정
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = f'output_{TIMESTAMP}'


# 결과 저장 디렉토리 생성
def create_output_dirs():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        os.makedirs(os.path.join(OUTPUT_DIR, 'figures'))
        os.makedirs(os.path.join(OUTPUT_DIR, 'models'))
        os.makedirs(os.path.join(OUTPUT_DIR, 'results'))
    return OUTPUT_DIR

