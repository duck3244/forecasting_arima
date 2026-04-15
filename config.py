"""
설정 파일 - ARIMA 모델 및 전반적인 파라미터 설정
"""

import os
import logging
import random

from datetime import datetime

import numpy as np


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
CV_MODE = 'walk_forward'  # 'walk_forward' | 'expanding'

# 수치/전처리 공통 상수
EPSILON = 1e-10
OUTLIER_IQR_MULTIPLIER = 1.5
EXOG_CORR_THRESHOLD = 0.1
EXOG_NAN_RATIO_MAX = 0.1

# 재현성
RANDOM_SEED = 42

# 시각화 설정
FIGURE_SIZE = (15, 8)
DPI = 300
SAVE_FIGURES = True

# 예측 구간 신뢰 수준
PI_ALPHA = 0.05  # 95% 신뢰구간

# 저장 경로 설정
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = f'output_{TIMESTAMP}'


def create_output_dirs():
    """결과 저장 디렉토리 생성"""
    for sub in ('', 'figures', 'models', 'results', 'logs'):
        path = os.path.join(OUTPUT_DIR, sub) if sub else OUTPUT_DIR
        os.makedirs(path, exist_ok=True)
    return OUTPUT_DIR


def set_seeds(seed=RANDOM_SEED):
    """재현성을 위한 난수 시드 고정"""
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def setup_logging(output_dir=None, level=logging.INFO):
    """
    루트 로거를 설정합니다. 콘솔 + (선택) 파일 핸들러를 부착합니다.
    반복 호출 시 중복 핸들러가 붙지 않도록 방지합니다.
    """
    root = logging.getLogger()
    root.setLevel(level)

    # 기존 핸들러 제거 (반복 호출 대비)
    for h in list(root.handlers):
        root.removeHandler(h)

    fmt = logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s',
                            datefmt='%H:%M:%S')

    console = logging.StreamHandler()
    console.setFormatter(fmt)
    root.addHandler(console)

    if output_dir:
        log_path = os.path.join(output_dir, 'logs', 'run.log')
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        file_handler = logging.FileHandler(log_path, encoding='utf-8')
        file_handler.setFormatter(fmt)
        root.addHandler(file_handler)

    return root
