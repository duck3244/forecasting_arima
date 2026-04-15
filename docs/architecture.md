# Architecture

본 문서는 `forecasting_arima` 파이프라인의 전체 구조, 모듈 책임, 데이터 흐름,
그리고 주요 설계 결정을 설명합니다.

---

## 1. 목적

주간 매출(`Weekly_Sales`) 시계열에 대해 ARIMA / SARIMA / 앙상블 / GBM 잔차
하이브리드 모델을 학습·평가·예측하는 **엔드투엔드 배치 분석 파이프라인**입니다.
프로덕션 서비스가 아닌, **재현 가능한 일회성 실험 실행기**로 설계되었습니다.

핵심 비기능 요구:

- **재현성**: 시드 고정, 결정론적 전처리, 타임스탬프 출력 격리
- **데이터 누수 방지**: 평가 파이프라인 전체가 train split 내부에서 동작
- **관측 가능성**: 표준 `logging`, 파일 로그, HTML/JSON 리포트
- **검증 가능성**: 단위 테스트(`tests/test_core.py`)로 핵심 헬퍼 보호

---

## 2. 레이어 구조

파이프라인은 **단방향 의존성을 가진 4개 레이어**로 구성됩니다.

```
┌─────────────────────────────────────────────────────────┐
│ L4  Orchestration      main.py                          │
│                        (argparse → 전 스텝 조합)         │
├─────────────────────────────────────────────────────────┤
│ L3  Modeling           model_selection.py  forecast.py  │
│                        evaluation.py                    │
├─────────────────────────────────────────────────────────┤
│ L2  Data & Metrics     data_handler.py     utils.py     │
├─────────────────────────────────────────────────────────┤
│ L1  Config / Infra     config.py  (로깅, 시드, 상수,     │
│                                    출력 경로)            │
└─────────────────────────────────────────────────────────┘
```

상위 레이어만 하위 레이어를 import 합니다. L1은 어떠한 도메인 로직도
포함하지 않습니다.

---

## 3. 모듈 책임

| 모듈 | 책임 | 주요 심볼 |
|---|---|---|
| `config.py` | 상수, 시드, 로거 설정, 출력 디렉토리 생성 | `create_output_dirs`, `set_seeds`, `setup_logging`, `EPSILON`, `PI_ALPHA` |
| `data_handler.py` | CSV 로드, 결측/이상치 처리, 휴일 더미, 로그 변환, 외생변수 선택, 정상성 검정(ADF+KPSS), 훈련/테스트 분할, 역변환 | `load_data`, `preprocess_data`, `select_exog_variables`, `check_stationarity`, `prepare_train_test_split`, `inverse_transform` |
| `model_selection.py` | 기본 ARIMA 모델, ARIMA/SARIMA 그리드 서치 (AIC 기준), 모델 직렬화 | `create_base_model`, `grid_search_arima`, `grid_search_sarima` |
| `evaluation.py` | Walk-forward CV, 잔차 분석(Ljung-Box), 홀드아웃 평가, 모델 비교 | `time_series_cv`, `residual_analysis`, `evaluate_forecast`, `compare_models`, `_walk_forward_splits` |
| `forecast.py` | 앙상블 점 예측, 신뢰 구간 포함 미래 예측, 최종 모델 refit | `ensemble_forecast`, `predict_future`, `create_final_model`, `prepare_future_exog` |
| `visualization.py` | 모든 PNG 산출물 (시계열, ACF/PACF, 예측, CI 음영, 잔차, 성능 비교) | `plot_time_series`, `plot_acf_pacf`, `plot_forecast`, `plot_future_forecast`, `plot_feature_importance` |
| `utils.py` | 공통 메트릭, ACF 기반 계절성 감지, HTML 리포트, 계절성 분해 | `calculate_metrics`, `detect_seasonal_period`, `save_summary_report`, `perform_seasonal_decomposition`, `get_best_model` |
| `main.py` | 전체 파이프라인 실행 흐름과 CLI 인자 해석 | `main`, `parse_args` |
| `tests/test_core.py` | 핵심 헬퍼 단위 테스트 | (pytest) |

---

## 4. 실행 파이프라인 (L4 관점)

`main.main()`이 실행하는 단계는 다음과 같습니다.

1. **초기화** — `create_output_dirs`, `setup_logging`, `set_seeds`
2. **데이터 로드/전처리** — `load_data` → `preprocess_data`
3. **외생변수 선택** — `select_exog_variables` (휴일 더미는 도메인 강제 유지)
4. **Target 구성** — 일반 모드: 로그 변환 시리즈 / 하이브리드 모드: `Weekly_Sales - GBM_Pred`
5. **EDA 시각화** — 시계열, 변수 중요도, ACF/PACF, 계절성 분해
6. **훈련/테스트 조기 분할** — 이후 모든 모델 학습은 train만 사용 → **누수 방지**
7. **정상성 검정** — train에 ADF+KPSS 합의 기반
8. **모델 학습/선택** — 기본 ARIMA → ARIMA 그리드 서치 → SARIMA 그리드 서치
9. **Walk-forward CV** — train 내부에서 수행, best order를 검증
10. **잔차 분석** — best_arima, best_sarima 각각 Ljung-Box
11. **테스트 홀드아웃 평가** — 여러 모델 + 앙상블, WMAPE 기준 비교
12. **예측 시각화** — 예측 구간 음영 포함
13. **미래 예측** (옵션) — 전체 데이터로 refit 후 `predict_future`
14. **리포트 생성** — JSON + HTML 요약, 모델 비교 차트

하이브리드 모드(`--hybrid`)와 `--skip-future`는 14단계를 건너뜁니다.

---

## 5. 데이터 흐름

```
sales.csv
   │
   ▼
load_data ──► preprocess_data ──► select_exog_variables
   │             │ (holiday dummies, log transform)        │
   │             │                                         │
   │             ▼                                         │
   │        store_data (DataFrame)                         │
   │             │                                         │
   │             ▼                                         │
   │       sales_series / sales_series_orig                │
   │             │                                         │
   │             ▼                                         │
   │       prepare_train_test_split ◄── exog ──────────────┘
   │             │
   │             ▼
   │        (train, test, train_exog, test_exog)
   │             │
   ▼             ▼
check_stationarity ──► d
                       │
                       ▼
         grid_search_arima / grid_search_sarima  (train only)
                       │
                       ▼
               best_model + best_order
                       │
          ┌────────────┼────────────┐
          ▼            ▼            ▼
    time_series_cv  residual_   evaluate_forecast
    (walk-forward)   analysis    (test holdout)
                       │            │
                       ▼            ▼
                  결과 저장     model_performance
                                    │
                                    ▼
                               compare_models
                                    │
                                    ▼
                          create_final_model (refit)
                                    │
                                    ▼
                           predict_future (mean, CI)
                                    │
                                    ▼
                           save_summary_report
                                    │
                                    ▼
                      output_*/results/summary_report.html
```

---

## 6. 출력 구조

실행 시 `output_{YYYYMMDD_HHMMSS}/`가 생성되며, 내부는 네 하위 디렉토리로
격리됩니다.

```
output_20260415_110451/
├── figures/         # 모든 PNG (EDA, 예측, 잔차, 비교)
├── models/          # joblib 직렬화된 ARIMA/SARIMA/final
├── results/         # JSON/CSV 지표, future_predictions, summary_report.html
└── logs/run.log     # 해당 실행의 전체 로그
```

타임스탬프 격리 덕분에 **여러 실험을 간섭 없이 병행** 실행할 수 있습니다.

---

## 7. 주요 설계 결정

### 7.1 데이터 누수 방지

기존 구현은 `sales_series` 전체에 fit한 뒤 구간 일부를 테스트로 평가했습니다.
현재는 EDA를 제외한 **모든 파라미터 추정 과정(정상성, 그리드 서치, CV,
잔차 분석)이 `train` 분할에만 의존**합니다. `create_final_model`만 예외적으로
전체 데이터로 refit되며, 이는 미래 예측 전용입니다.

### 7.2 Walk-forward CV

`sklearn.TimeSeriesSplit`(확장 창)은 fold별 test 길이가 균일하지 않아 시계열
평가에 불리합니다. `_walk_forward_splits`는 **고정 test window**를 유지하며,
train 끝 인덱스가 test 시작보다 항상 작음을 보장합니다.

### 7.3 WMAPE를 주 지표로

`MAPE`는 0이나 음수값에서 발산합니다. `WMAPE = Σ|actual−pred| / Σ|actual|`는
0값에 강건하고, 0-샘플 필터링으로 인한 표본 편향이 없습니다. 최적 모델 선정
기준이 WMAPE이며, 호환용으로 MAE/RMSE/MAPE/SMAPE도 함께 저장합니다.

### 7.4 정상성 검정 합의 로직

ADF(H0: 비정상)와 KPSS(H0: 정상)는 귀무가설 방향이 반대입니다. 두 검정이
**동시에 정상을 지시할 때만** `d=0`으로 결정해 차분을 생략합니다.

### 7.5 신뢰 구간

`forecast.forecast()`는 점 예측만 반환합니다. `_forecast_with_ci`는
`get_forecast().conf_int(alpha=PI_ALPHA)`를 호출해 95% 예측 구간을
생성하고, 실패 시 안전한 fallback으로 점 예측과 NaN CI를 반환합니다.

### 7.6 하이브리드 모드

잔차 하이브리드(`Weekly_Sales − GBM_Pred`)는 GBM이 놓친 시계열 구조만을
ARIMA가 포착하게 합니다. 미래 예측에는 `GBM_Pred`의 미래 값이 필요하므로,
하이브리드 모드에서는 **14단계 미래 예측을 자동 스킵**하고 테스트 평가까지만
수행합니다.

### 7.7 로깅 전략

표준 `logging` 모듈로 통일. 루트 로거에 콘솔 + 파일 핸들러를 부착하고,
`logger = logging.getLogger(__name__)` 패턴을 모든 모듈에서 사용합니다.
`--log-level`로 실행 시 출력 상세도를 제어합니다.

---

## 8. 확장 포인트

| 요구 | 확장 지점 |
|---|---|
| 새 모델 추가 | `model_selection.py`에 search 함수 + `main.py`에서 `_eval` 호출 |
| 새 평가 지표 | `utils.calculate_metrics` 단일 지점에 추가 (`compare_models` 자동 반영) |
| 새 시각화 | `visualization.py`에 함수 추가, `main.py`의 해당 스텝에서 호출 |
| 외생 변수 변경 | `data_handler.select_exog_variables`의 `potential` 리스트 |
| CV 전략 교체 | `evaluation._walk_forward_splits` 또는 `time_series_cv` 대체 |
| 새 CLI 옵션 | `main.parse_args` + 해당 스텝 분기 |

---

## 9. 의존성

- **numpy / pandas**: 자료 구조, 수치
- **statsmodels**: ARIMA, SARIMAX, ADF/KPSS, Ljung-Box, ACF/PACF, 계절성 분해
- **scikit-learn**: `mean_absolute_error`, `mean_squared_error`
- **scipy**: Q-Q plot (`stats.probplot`)
- **matplotlib**: 모든 시각화 (headless 안전 `plt.close()` 사용)
- **joblib**: 모델 직렬화
- **pytest**: 단위 테스트

`requirements.txt`에 고정되어 있습니다.

---

## 10. 알려진 제약

- `SEASONAL_PERIOD=52`는 최소 100주 이상의 훈련 데이터를 요구하며, 짧은
  시계열(Store 1: 90주)에서는 SARIMA가 불안정할 수 있습니다.
- `prepare_future_exog`는 미래 외생변수 값을 마지막 구간 복제로 대체합니다.
- 하이브리드 모드는 `GBM_Pred`의 미래값을 확보하는 로직이 없어 미래 예측을
  수행할 수 없습니다.
