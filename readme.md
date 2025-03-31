# ARIMA 기반 판매 예측 시스템

이 프로젝트는 시계열 데이터를 분석하고 ARIMA/SARIMA 모델을 사용하여 미래 매출을 예측하는 분석 시스템입니다.

## 프로젝트 구조

```
forecasting_arima/
├── config.py               # 설정 파일
├── data_handler.py         # 데이터 로드 및 전처리 기능
├── model_selection.py      # 모델 선택 및 파라미터 그리드 서치 기능
├── evaluation.py           # 모델 평가 및 교차 검증 기능
├── visualization.py        # 데이터 및 결과 시각화 기능
├── forecast.py             # 예측 수행 기능
├── utils.py                # 유틸리티 함수
└── main.py                 # 메인 실행 파일
```

## 주요 기능

1. **데이터 전처리**
   - 결측치 처리
   - 이상치 탐지 및 처리
   - 로그 변환을 통한 정규화
   - 외생 변수 선택 및 상관관계 분석

2. **모델 선택 및 최적화**
   - 정상성 검정
   - ARIMA 모델 그리드 서치
   - SARIMA 모델 그리드 서치
   - 최적 모델 저장

3. **모델 평가**
   - 시계열 교차 검증
   - 잔차 분석
   - 다양한 성능 지표 (MAE, RMSE, MAPE, SMAPE, WMAPE)
   - 모델 비교

4. **결과 시각화**
   - 시계열 데이터 시각화
   - ACF/PACF 플롯
   - 예측 결과 시각화
   - 모델 성능 비교 시각화
   - 모든 그래프 PNG 파일로 저장

5. **미래 예측**
   - 앙상블 예측
   - 미래 매출 예측
   - 예측 결과 CSV 및 JSON 파일로 저장

6. **결과 리포트**
   - HTML 형식의 종합 보고서 생성
   - 최적 모델 요약
   - 모델 성능 비교 테이블

## 설치 방법

1. 필요한 패키지 설치:

```bash
pip install -r requirements.txt
```

2. 요구사항:
   - Python 3.7 이상
   - pandas, numpy, matplotlib, statsmodels, scikit-learn, scipy

## 사용 방법

1. 설정 파일(`config.py`)에서 분석할 매장 ID와 기타 매개변수 설정
2. 데이터 파일(`sales.csv`)을 프로젝트 디렉토리에 위치
3. 메인 스크립트 실행:

```bash
python main.py
```

4. 결과는 `output_{timestamp}` 디렉토리에 저장됨:
   - `figures/`: 모든 시각화 결과 (PNG 파일)
   - `models/`: 학습된 모델 파일
   - `results/`: 분석 결과 및 예측 데이터 (JSON, CSV)

## 주요 분석 과정

1. 데이터 로드 및 전처리
2. 외생 변수 선택
3. 데이터 시각화 및 정상성 검정
4. 최적 ARIMA/SARIMA 모델 탐색
5. 모델 교차 검증 및 잔차 분석
6. 테스트 데이터로 모델 평가
7. 앙상블 모델 생성 및 평가
8. 미래 예측 수행
9. 결과 리포트 생성

## 예시 출력

- 시계열 데이터 시각화
- ACF/PACF 플롯
- 모델 잔차 분석
- 예측 결과 시각화
- 미래 예측 시각화
- 모델 성능 비교 차트
