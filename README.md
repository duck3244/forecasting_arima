# ARIMA 기반 판매 예측 시스템

시계열 데이터를 분석해 ARIMA / SARIMA / 앙상블 / **GBM 잔차 하이브리드** 모델로 주간
매출을 예측하는 엔드투엔드 분석 파이프라인입니다. 재현성, 데이터 누수 방지,
walk-forward 교차검증, 예측 구간(신뢰 구간) 등 실무 운영에 필요한 요소를 갖추도록
설계되었습니다.

---

## 프로젝트 구조

```
forecasting_arima/
├── main.py                 # 엔트리포인트 (argparse CLI)
├── config.py               # 설정·상수·시드·로깅 셋업
├── data_handler.py         # 로드/전처리/분할/변환/정상성 검정(ADF+KPSS)
├── model_selection.py      # ARIMA·SARIMA 그리드 서치
├── evaluation.py           # walk-forward CV, 잔차 분석, 메트릭
├── forecast.py             # 최종 학습, 미래 예측, 예측 구간
├── visualization.py        # 모든 플롯 (headless 안전, CI 음영 지원)
├── utils.py                # 공통 메트릭, 계절성 자동 감지, HTML 리포트
│
├── docs/                   # 프로젝트 문서
│   ├── architecture.md     # 레이어 구조·모듈 책임·설계 결정
│   └── uml.md              # Mermaid UML 다이어그램 8종
│
├── tests/                  # pytest 단위 테스트
│   └── test_core.py        # 분할/역변환/메트릭/walk-forward/계절성 감지
│
├── output_{timestamp}/     # 실행별 산출물 (.gitignore에 의해 제외)
│   ├── figures/            # PNG 시각화 (EDA, 예측, 잔차, 모델 비교)
│   ├── models/             # joblib 직렬화된 ARIMA/SARIMA/final 모델
│   ├── results/            # JSON/CSV 지표, summary_report.html
│   │                       # future_predictions.csv (Lower/Upper CI 포함)
│   └── logs/run.log        # 해당 실행의 전체 로그
│
├── sales.csv               # 샘플 데이터 (Walmart 주간 매출)
├── requirements.txt        # 의존성
├── .gitignore              # output_*/, *.pkl, PNG 등 제외
├── LICENSE
└── README.md
```

---

## 주요 기능

### 1. 데이터 전처리
- 수치형 결측치 선형 보간
- **IQR 캡핑은 기본 OFF** (블랙프라이데이 등 예측해야 할 피크 신호 보존). 옛 동작은 `--baseline-preproc`로 재현.
- **휴일 더미** 자동 생성: 슈퍼볼(w6), 노동절(w36), 추수감사절(w47), 크리스마스(w51–52)
- `Weekly_Sales` 로그 변환 (0/음수 대응 `log1p`+오프셋)
- 상관관계 기반 외생 변수 선택 (휴일 더미는 도메인 지식으로 강제 유지)

### 2. 정상성 검정
- **ADF + KPSS 합의 기반** 판정 — 두 검정이 모두 정상일 때만 `d=0`

### 3. 모델 선택
- ARIMA 그리드 서치 (`MAX_P`, `MAX_Q` 구간)
- SARIMA 그리드 서치 (시도 수 제한)
- 학습 실패 시 `ValueError` / `LinAlgError` / `RuntimeError` 등 **세분화 예외 처리**
- 모든 best 모델은 `joblib`으로 `output_*/models/`에 저장

### 4. 모델 평가
- **Walk-forward CV** (고정 창, 누수 없음) — 기존 `TimeSeriesSplit` 대체
- 잔차 분석: 시계열/히스토그램/ACF/Q-Q + Ljung-Box (lag 10/20/30/40)
- 통합 메트릭 헬퍼 `utils.calculate_metrics`: MAE, RMSE, MAPE, SMAPE, **WMAPE**
- **WMAPE를 best 모델 선정 기준**으로 사용 (0값에 강건)

### 5. 예측
- 테스트 홀드아웃 평가 + **앙상블 예측**
- `get_forecast().conf_int()` 기반 **95% 예측 구간** 반환 및 CSV/JSON/플롯 저장
- 하이브리드 모드 (`--hybrid`): `target = Weekly_Sales − GBM_Pred`, 최종 예측은
  `GBM_Pred + ARIMA(residual)`

### 6. 재현성 & 운영 편의
- 전역 **난수 시드 고정** (`config.set_seeds`)
- 표준 `logging` 도입 (콘솔 + `output_*/logs/run.log`)
- `argparse` CLI (`--store`, `--steps`, `--seed`, `--hybrid`, …)
- **pytest 단위 테스트** (분할/역변환/메트릭/walk-forward/계절성 감지)

### 7. 계절성 자동 감지
- `--auto-seasonal`: ACF 피크 기반으로 `SEASONAL_PERIOD`를 자동 추정

### 8. 리포트
- JSON 요약 + HTML 리포트 (`output_*/results/summary_report.html`)
- 모델 비교 차트, 예측 구간 시각화

---

## 데이터 누수 방지 구조

기존 구현은 `sales_series` 전체로 모델을 학습한 뒤 그 안의 일부 구간을 테스트로
평가해 **정보가 새는 구조**였습니다. 새 파이프라인은 다음 순서로 동작합니다.

1. 데이터 로드/전처리/EDA (비파라메트릭)
2. **훈련/테스트 조기 분할**
3. 정상성 검정, 그리드 서치, 교차 검증, 잔차 분석 → **훈련 데이터만** 사용
4. 테스트 데이터로 홀드아웃 평가
5. 미래 예측용 최종 모델만 전체 데이터로 refit

---

## 설치

```bash
pip install -r requirements.txt
```

요구사항: Python 3.9+, pandas, numpy, matplotlib, statsmodels, scikit-learn,
scipy, joblib, pytest.

---

## 사용 방법

기본 실행:

```bash
python main.py
```

CLI 옵션:

```bash
python main.py \
    --data sales.csv \
    --store 1 \
    --steps 12 \
    --seed 42 \
    --auto-seasonal \
    --log-level INFO
```

| 플래그                | 설명                                               |
| -------------------- | ------------------------------------------------- |
| `--data PATH`        | 입력 CSV 경로 (기본 `sales.csv`)                    |
| `--store ID`         | 분석할 매장 ID                                     |
| `--steps N`          | 예측 기간(주)                                      |
| `--seed N`           | 난수 시드                                          |
| `--auto-seasonal`    | ACF 기반 계절성 주기 자동 감지                      |
| `--hybrid`           | GBM 잔차 하이브리드 모드 (`GBM_Pred` 컬럼 필요)     |
| `--baseline-preproc` | 옛 전처리(IQR 캡 ON, 휴일 더미 OFF) 재현            |
| `--skip-future`      | 미래 예측 단계를 건너뛰고 테스트 평가까지만 수행    |
| `--log-level`        | DEBUG / INFO / WARNING / ERROR                    |

실행 결과는 `output_{timestamp}/`에 저장됩니다.

```
output_YYYYMMDD_HHMMSS/
├── figures/    # 모든 PNG 시각화
├── models/     # 학습된 모델 (joblib)
├── results/    # JSON/CSV (summary_report.html, future_predictions.csv, …)
└── logs/       # run.log
```

---

## 하이브리드 모드 사용법

데이터에 `GBM_Pred` 컬럼이 있을 때 활용 가능합니다.

```bash
python main.py --hybrid
```

동작:
- target = `Weekly_Sales − GBM_Pred`
- ARIMA/SARIMA는 잔차의 시계열 구조만 학습
- 평가 시 `GBM_Pred`를 더해 원 스케일로 복원
- **Ljung-Box로 잔차 자기상관 여부를 먼저 확인할 것** — 잔차가 white noise면
  하이브리드는 오히려 성능을 해칩니다 (Store 1 실험 참고).

참고용 벤치마크 (Store 1, 12주 홀드아웃):

| 구성                              | Best_ARIMA WMAPE | Ensemble WMAPE |
| --------------------------------- | ---------------: | -------------: |
| A. 옛 전처리 (`--baseline-preproc`) |            4.96% |          9.66% |
| B. 기본 (IQR해제 + 휴일)            |            5.44% |          9.14% |
| C. `--hybrid`                     |            2.26% |          1.94% |
| **참고: GBM_Pred 단독**            |                — |      **0.73%** |

Store 1에서는 GBM 잔차가 사실상 white noise(Ljung-Box p ≥ 0.80)라 하이브리드가
`GBM_Pred` 단독을 이기지 못합니다. 다른 매장/데이터셋에서는 매장별로 확인이
필요합니다.

---

## 단위 테스트

```bash
python -m pytest tests/ -q
```

검증 대상:
- `prepare_train_test_split` (형상, exog 정렬, 예외)
- `inverse_transform` (항등, log/log1p 라운드트립)
- `calculate_metrics` (완전 예측, 알려진 값, 0 처리)
- `_walk_forward_splits` (누수 없음, 끝점 정렬)
- `detect_seasonal_period` (사인파 감지)

---

## 설정 (`config.py`)

주요 상수:

```python
FORECAST_STEPS = 12           # 예측 기간(주)
SEASONAL_PERIOD = 52          # 기본 계절성 주기
CV_SPLITS = 5                 # walk-forward fold 수
CV_MODE = 'walk_forward'

EPSILON = 1e-10               # 메트릭 분모 안정화
OUTLIER_IQR_MULTIPLIER = 1.5  # IQR 캡핑 배수
EXOG_CORR_THRESHOLD = 0.1     # 외생변수 상관 임계치
PI_ALPHA = 0.05               # 95% 신뢰 구간
RANDOM_SEED = 42
```

---

## 알려진 제약

- 주간 데이터에서 `SEASONAL_PERIOD=52`는 최소 104~156주 이상의 훈련 데이터를
  요구합니다. Store 1처럼 90주 남짓 매장은 SARIMA가 불안정할 수 있으며, Fourier
  terms 기반 외생 모델링이 더 유리할 수 있습니다.
- `prepare_future_exog`는 현재 마지막 N개 외생변수 값을 단순 복제합니다. 운영
  환경에서는 외생변수의 미래값을 별도로 예측/확보해야 합니다.
- 하이브리드 모드는 `GBM_Pred`의 미래값이 필요하므로 미래 예측 단계를
  자동으로 건너뜁니다.

---

## 향후 개선 아이디어

- 매장별 **조건부 하이브리드** (잔차 자기상관이 있는 매장에만 ARIMA 적용)
- **Fourier terms**로 다중 계절성 처리 (주간 단기 + 연간 장기)
- `pmdarima.auto_arima` 도입으로 탐색 공간 확대
- **검증 WMAPE 역수 가중 앙상블**
- Lag 피처(`Weekly_Sales_prev_yr`, `prev_week`)를 외생 변수로 투입
- 다중 매장 **hierarchical / global 모델**로 정보 공유
