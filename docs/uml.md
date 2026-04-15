# UML 다이어그램

본 문서는 `forecasting_arima` 파이프라인의 구조와 동작을 **Mermaid** 문법의
UML 다이어그램으로 정리합니다. GitHub, JetBrains IDE, VS Code의 Markdown
프리뷰에서 바로 렌더링됩니다.

포함된 다이어그램:

1. [모듈 의존성 그래프](#1-모듈-의존성-그래프)
2. [컴포넌트 다이어그램](#2-컴포넌트-다이어그램)
3. [주요 함수 클래스 뷰](#3-주요-함수-클래스-뷰)
4. [전체 실행 시퀀스](#4-전체-실행-시퀀스)
5. [하이브리드 모드 시퀀스](#5-하이브리드-모드-시퀀스)
6. [모델 학습·평가 상태 다이어그램](#6-모델-학습평가-상태-다이어그램)
7. [CLI 실행 활동 다이어그램](#7-cli-실행-활동-다이어그램)
8. [데이터 플로우](#8-데이터-플로우)

---

## 1. 모듈 의존성 그래프

모듈 간 import 관계 (↓ 하위 레이어로만 단방향).

```mermaid
%%{init: {'theme':'base','themeVariables':{'primaryColor':'#fcfcfc','primaryTextColor':'#111','primaryBorderColor':'#111','lineColor':'#111','secondaryColor':'#ffffff','tertiaryColor':'#ffffff','textColor':'#111','actorTextColor':'#111','actorBorder':'#111','actorBkg':'#fcfcfc','noteTextColor':'#111','noteBkgColor':'#ffffff','noteBorderColor':'#111','labelTextColor':'#111','classText':'#111','titleColor':'#111','edgeLabelBackground':'#ffffff','mainBkg':'#fcfcfc','nodeBorder':'#111','clusterBkg':'#ffffff','clusterBorder':'#111'}}}%%
graph TD
    main[main.py]
    config[config.py]
    data[data_handler.py]
    model[model_selection.py]
    eval[evaluation.py]
    forecast[forecast.py]
    viz[visualization.py]
    utils[utils.py]
    tests[tests/test_core.py]

    main --> config
    main --> data
    main --> model
    main --> eval
    main --> forecast
    main --> viz
    main --> utils

    data --> config
    model --> config
    eval --> config
    eval --> utils
    forecast --> config
    forecast --> data
    viz --> config
    utils --> config

    tests --> data
    tests --> utils
    tests --> eval

    classDef l1 fill:#fafcff,stroke:#0a2a5e,color:#111,stroke-width:2px
    classDef l2 fill:#fafefb,stroke:#0b4a1c,color:#111,stroke-width:2px
    classDef l3 fill:#fffcf5,stroke:#6b3d00,color:#111,stroke-width:2px
    classDef l4 fill:#fff7f8,stroke:#5e0f18,color:#111,stroke-width:2px
    class config l1
    class data,utils l2
    class model,eval,forecast,viz l3
    class main,tests l4
```

---

## 2. 컴포넌트 다이어그램

파이프라인의 실행 컴포넌트와 산출물.

```mermaid
%%{init: {'theme':'base','themeVariables':{'primaryColor':'#fcfcfc','primaryTextColor':'#111','primaryBorderColor':'#111','lineColor':'#111','secondaryColor':'#ffffff','tertiaryColor':'#ffffff','textColor':'#111','actorTextColor':'#111','actorBorder':'#111','actorBkg':'#fcfcfc','noteTextColor':'#111','noteBkgColor':'#ffffff','noteBorderColor':'#111','labelTextColor':'#111','classText':'#111','titleColor':'#111','edgeLabelBackground':'#ffffff','mainBkg':'#fcfcfc','nodeBorder':'#111','clusterBkg':'#ffffff','clusterBorder':'#111'}}}%%
graph LR
    subgraph Inputs
        CSV[(sales.csv)]
        CLI[CLI args]
    end

    subgraph Pipeline
        PRE[Preprocess<br/>data_handler]
        SPLIT[Train/Test Split]
        SEL[Model Search<br/>model_selection]
        CV[Walk-forward CV<br/>evaluation]
        HOLD[Holdout Eval<br/>evaluation]
        FINAL[Final Refit<br/>forecast]
        FUT[Future Predict<br/>forecast]
        REPORT[Report<br/>utils]
    end

    subgraph Outputs
        FIG[(figures/*.png)]
        MDL[(models/*.pkl)]
        RES[(results/*.json,csv)]
        LOG[(logs/run.log)]
        HTML[(summary_report.html)]
    end

    CSV --> PRE
    CLI --> PRE
    PRE --> SPLIT --> SEL --> CV --> HOLD --> FINAL --> FUT --> REPORT
    SEL --> MDL
    HOLD --> RES
    FUT --> RES
    FUT --> FIG
    CV --> FIG
    HOLD --> FIG
    REPORT --> HTML
    Pipeline --> LOG
```

---

## 3. 주요 함수 클래스 뷰

모듈을 "의사 클래스(pseudo class)"로 표현. 실제 Python 코드는 함수형이지만,
각 모듈이 하나의 네임스페이스로서 기능한다는 시각화입니다.

```mermaid
%%{init: {'theme':'base','themeVariables':{'primaryColor':'#fcfcfc','primaryTextColor':'#111','primaryBorderColor':'#111','lineColor':'#111','secondaryColor':'#ffffff','tertiaryColor':'#ffffff','textColor':'#111','actorTextColor':'#111','actorBorder':'#111','actorBkg':'#fcfcfc','noteTextColor':'#111','noteBkgColor':'#ffffff','noteBorderColor':'#111','labelTextColor':'#111','classText':'#111','titleColor':'#111','edgeLabelBackground':'#ffffff','mainBkg':'#fcfcfc','nodeBorder':'#111','clusterBkg':'#ffffff','clusterBorder':'#111'}}}%%
classDiagram
    class Config {
        +DATA_PATH: str
        +STORE_ID: int
        +FORECAST_STEPS: int
        +SEASONAL_PERIOD: int
        +CV_SPLITS: int
        +EPSILON: float
        +PI_ALPHA: float
        +RANDOM_SEED: int
        +OUTPUT_DIR: str
        +create_output_dirs()
        +set_seeds(seed)
        +setup_logging(output_dir, level)
    }

    class DataHandler {
        +load_data(path) DataFrame
        +preprocess_data(df, apply_iqr_cap, add_holidays) DataFrame
        +select_exog_variables(df) List~str~
        +check_stationarity(series) bool
        +prepare_train_test_split(series, exog, test_size) tuple
        +inverse_transform(values, store_data, transformed) ndarray
        +get_future_dates(last_date, steps)
        -_add_holiday_dummies(df)
    }

    class ModelSelection {
        +grid_search_arima(series, exog, p,d,q) tuple
        +grid_search_sarima(series, exog, ...) tuple
        +create_base_model(series, exog, d, s)
    }

    class Evaluation {
        +time_series_cv(series, exog, model_type, order, ...) dict
        +residual_analysis(model, model_name)
        +evaluate_forecast(actual, predicted, model_name) dict
        +compare_models(model_results)
        -_walk_forward_splits(n, n_splits, test_size) list
    }

    class Forecast {
        +ensemble_forecast(models, steps, exog) ndarray
        +predict_future(model, steps, exog, ...) tuple
        +create_final_model(series, exog, order, seasonal_order)
        +prepare_future_exog(exog, steps) DataFrame
        -_forecast_with_ci(model, steps, exog, alpha) tuple
    }

    class Visualization {
        +plot_time_series(series, title, ...)
        +plot_acf_pacf(series, lags, title)
        +plot_forecast(series, forecast, ...)
        +plot_future_forecast(series, forecast, ..., lower, upper)
        +plot_feature_importance(df, target, top_n, title)
    }

    class Utils {
        +calculate_metrics(actual, predicted) dict
        +detect_seasonal_period(series, candidates) int
        +save_summary_report(models_info, performance, store_id)
        +get_best_model(performance_results) dict
        +create_html_report(report) str
        +perform_seasonal_decomposition(series, model_name)
    }

    class Main {
        +parse_args() Namespace
        +main()
    }

    Main --> Config
    Main --> DataHandler
    Main --> ModelSelection
    Main --> Evaluation
    Main --> Forecast
    Main --> Visualization
    Main --> Utils
    Evaluation --> Utils
    Forecast --> DataHandler
    DataHandler --> Config
    ModelSelection --> Config
    Evaluation --> Config
    Forecast --> Config
    Visualization --> Config
    Utils --> Config
```

---

## 4. 전체 실행 시퀀스

표준 모드(`python main.py`)의 런타임 상호작용.

```mermaid
%%{init: {'theme':'base','themeVariables':{'primaryColor':'#fcfcfc','primaryTextColor':'#111','primaryBorderColor':'#111','lineColor':'#111','secondaryColor':'#ffffff','tertiaryColor':'#ffffff','textColor':'#111','actorTextColor':'#111','actorBorder':'#111','actorBkg':'#fcfcfc','noteTextColor':'#111','noteBkgColor':'#ffffff','noteBorderColor':'#111','labelTextColor':'#111','classText':'#111','titleColor':'#111','edgeLabelBackground':'#ffffff','mainBkg':'#fcfcfc','nodeBorder':'#111','clusterBkg':'#ffffff','clusterBorder':'#111'}}}%%
sequenceDiagram
    autonumber
    actor User
    participant Main as main.py
    participant Cfg as config
    participant DH as data_handler
    participant MS as model_selection
    participant EV as evaluation
    participant FC as forecast
    participant VZ as visualization
    participant UT as utils
    participant FS as Filesystem

    User->>Main: python main.py --store 1 --steps 12
    Main->>Cfg: create_output_dirs()
    Cfg->>FS: mkdir output_*/{figures,models,results,logs}
    Main->>Cfg: setup_logging() / set_seeds()

    Main->>DH: load_data(DATA_PATH)
    DH->>FS: read sales.csv
    Main->>DH: preprocess_data(df)
    Main->>DH: select_exog_variables(store_data)
    Main->>DH: prepare_train_test_split(series, exog)
    DH-->>Main: (train, test, train_exog, test_exog)

    Main->>DH: check_stationarity(train)
    DH-->>Main: d (0 또는 1)

    Main->>MS: create_base_model(train, ...)
    MS-->>Main: arima_model

    Main->>MS: grid_search_arima(train, ...)
    MS->>FS: joblib.dump(best_arima.pkl)
    MS-->>Main: (best_arima, best_order)

    Main->>MS: grid_search_sarima(train, ...)
    MS->>FS: joblib.dump(best_sarima.pkl)
    MS-->>Main: (best_sarima, order, seasonal)

    Main->>EV: time_series_cv(train, ..., 'arima')
    EV->>UT: calculate_metrics(test_fold, forecast)
    EV->>FS: save cv_results.json
    Main->>EV: time_series_cv(train, ..., 'sarima')

    Main->>EV: residual_analysis(best_arima)
    Main->>EV: residual_analysis(best_sarima)

    loop 모델별 테스트 홀드아웃 평가
        Main->>Main: model.forecast(test_exog)
        Main->>DH: inverse_transform(forecast)
        Main->>EV: evaluate_forecast(test_orig, forecast_orig)
        EV->>UT: calculate_metrics(...)
    end

    Main->>FC: ensemble_forecast([arima, best_arima, best_sarima])
    Main->>EV: compare_models(model_performance)

    Main->>VZ: plot_forecast(best_sarima)
    Main->>VZ: plot_forecast(ensemble)

    Main->>FC: create_final_model(full_series, ...)
    FC->>FS: joblib.dump(final_model.pkl)
    Main->>FC: predict_future(final_model, ...)
    FC->>FS: future_predictions.csv/.json
    FC-->>Main: (mean, dates, lower, upper)

    Main->>VZ: plot_future_forecast(... lower, upper)

    Main->>UT: save_summary_report(info, performance, store_id)
    UT->>FS: summary_report.json/.html

    Main-->>User: 로그 + output_*/ 산출물
```

---

## 5. 하이브리드 모드 시퀀스

`--hybrid` 플래그가 있을 때의 분기 흐름. target을 `Weekly_Sales − GBM_Pred`로
재정의하고, 평가 시 `GBM_Pred`를 다시 더한 뒤, 미래 예측 단계는 건너뜁니다.

```mermaid
%%{init: {'theme':'base','themeVariables':{'primaryColor':'#fcfcfc','primaryTextColor':'#111','primaryBorderColor':'#111','lineColor':'#111','secondaryColor':'#ffffff','tertiaryColor':'#ffffff','textColor':'#111','actorTextColor':'#111','actorBorder':'#111','actorBkg':'#fcfcfc','noteTextColor':'#111','noteBkgColor':'#ffffff','noteBorderColor':'#111','labelTextColor':'#111','classText':'#111','titleColor':'#111','edgeLabelBackground':'#ffffff','mainBkg':'#fcfcfc','nodeBorder':'#111','clusterBkg':'#ffffff','clusterBorder':'#111'}}}%%
sequenceDiagram
    autonumber
    actor User
    participant Main as main.py
    participant DH as data_handler
    participant MS as model_selection
    participant EV as evaluation
    participant UT as utils

    User->>Main: python main.py --hybrid
    Main->>DH: load + preprocess
    Main->>Main: gbm_pred_series = store_data['GBM_Pred']
    Main->>Main: sales_series = Weekly_Sales - GBM_Pred
    note over Main: transformed = False<br/>(잔차는 음수 가능)

    Main->>DH: prepare_train_test_split(sales_series, exog)
    DH-->>Main: (train_residual, test_residual, ...)
    Main->>DH: prepare_train_test_split(sales_series_orig)
    note right of Main: 원 스케일 test_orig도 별도 분할

    Main->>MS: grid_search_arima(train_residual)
    Main->>MS: grid_search_sarima(train_residual)

    Main->>EV: evaluate_forecast(test_orig, gbm_test)
    note right of EV: GBM_Only 기준선 평가

    loop 각 후보 모델
        Main->>Main: forecast_residual = model.forecast(test_exog)
        Main->>Main: forecast_sales = forecast_residual + gbm_test
        Main->>EV: evaluate_forecast(test_orig, forecast_sales)
    end

    Main->>EV: compare_models(performance)
    Main->>UT: save_summary_report(...)
    note over Main: ⚠ 14단계 미래 예측 스킵<br/>(GBM_Pred 미래값 부재)
    Main-->>User: 하이브리드 결과 + output_*/
```

---

## 6. 모델 학습·평가 상태 다이어그램

단일 모델이 파이프라인 내에서 거치는 상태 전이.

```mermaid
%%{init: {'theme':'base','themeVariables':{'primaryColor':'#fcfcfc','primaryTextColor':'#111','primaryBorderColor':'#111','lineColor':'#111','secondaryColor':'#ffffff','tertiaryColor':'#ffffff','textColor':'#111','actorTextColor':'#111','actorBorder':'#111','actorBkg':'#fcfcfc','noteTextColor':'#111','noteBkgColor':'#ffffff','noteBorderColor':'#111','labelTextColor':'#111','classText':'#111','titleColor':'#111','edgeLabelBackground':'#ffffff','mainBkg':'#fcfcfc','nodeBorder':'#111','clusterBkg':'#ffffff','clusterBorder':'#111'}}}%%
stateDiagram-v2
    [*] --> Created : create_base_model / grid_search

    Created --> Fitted : .fit() 성공
    Created --> FitFailed : ValueError / LinAlgError
    FitFailed --> [*] : logger.warning, skip

    Fitted --> CVEvaluated : time_series_cv(train)
    CVEvaluated --> ResidualChecked : residual_analysis
    ResidualChecked --> HoldoutEvaluated : evaluate_forecast(test)
    HoldoutEvaluated --> InEnsemble : ensemble_forecast
    InEnsemble --> Compared : compare_models

    Compared --> Refit : create_final_model(full_series)
    note right of Refit
        미래 예측 전용.
        하이브리드 모드에서는
        이 전이가 발생하지 않음.
    end note
    Refit --> FutureForecast : predict_future(steps, CI)
    FutureForecast --> Saved : joblib + csv + json
    Saved --> [*]
```

---

## 7. CLI 실행 활동 다이어그램

`main.main()`의 분기 포함 활동 흐름.

```mermaid
%%{init: {'theme':'base','themeVariables':{'primaryColor':'#fcfcfc','primaryTextColor':'#111','primaryBorderColor':'#111','lineColor':'#111','secondaryColor':'#ffffff','tertiaryColor':'#ffffff','textColor':'#111','actorTextColor':'#111','actorBorder':'#111','actorBkg':'#fcfcfc','noteTextColor':'#111','noteBkgColor':'#ffffff','noteBorderColor':'#111','labelTextColor':'#111','classText':'#111','titleColor':'#111','edgeLabelBackground':'#ffffff','mainBkg':'#fcfcfc','nodeBorder':'#111','clusterBkg':'#ffffff','clusterBorder':'#111'}}}%%
flowchart TD
    START([python main.py]) --> INIT[로깅·시드·출력 디렉토리 초기화]
    INIT --> LOAD[데이터 로드 + 전처리]
    LOAD --> EXOG[외생변수 선택]
    EXOG --> MODE{--hybrid?}
    MODE -- yes --> HYB["target = Weekly_Sales - GBM_Pred<br/>transformed = False"]
    MODE -- no --> STD["target = log(Weekly_Sales)"]
    HYB --> EDA[EDA 시각화]
    STD --> EDA
    EDA --> AUTO{--auto-seasonal?}
    AUTO -- yes --> DETECT[detect_seasonal_period]
    AUTO -- no --> SKIPDET[config 기본값 사용]
    DETECT --> SPLIT
    SKIPDET --> SPLIT[train/test 분할]
    SPLIT --> STAT[정상성 검정]
    STAT --> BASE[기본 ARIMA 학습]
    BASE --> GA[ARIMA 그리드 서치]
    GA --> GS[SARIMA 그리드 서치]
    GS --> CV[Walk-forward CV]
    CV --> RES[잔차 분석]
    RES --> EVAL[테스트 홀드아웃 평가]
    EVAL --> ENS[앙상블 평가]
    ENS --> CMP[compare_models]
    CMP --> PLOT[예측 시각화]
    PLOT --> FSKIP{hybrid or skip-future?}
    FSKIP -- yes --> SAVEH[요약 저장 - 미래 스킵]
    FSKIP -- no --> REFIT[최종 모델 refit]
    REFIT --> FUT[predict_future + CI]
    FUT --> FPLOT[plot_future_forecast]
    FPLOT --> SAVE[save_summary_report]
    SAVEH --> END([종료])
    SAVE --> END
```

---

## 8. 데이터 플로우

시리즈·모델·아티팩트의 흐름을 ER 스타일로.

```mermaid
%%{init: {'theme':'base','themeVariables':{'primaryColor':'#fcfcfc','primaryTextColor':'#111','primaryBorderColor':'#111','lineColor':'#111','secondaryColor':'#ffffff','tertiaryColor':'#ffffff','textColor':'#111','actorTextColor':'#111','actorBorder':'#111','actorBkg':'#fcfcfc','noteTextColor':'#111','noteBkgColor':'#ffffff','noteBorderColor':'#111','labelTextColor':'#111','classText':'#111','titleColor':'#111','edgeLabelBackground':'#ffffff','mainBkg':'#fcfcfc','nodeBorder':'#111','clusterBkg':'#ffffff','clusterBorder':'#111'}}}%%
erDiagram
    SALES_CSV ||--o{ RAW_DF : "read_csv"
    RAW_DF ||--|| STORE_DF : "filter Store=id"
    STORE_DF ||--|| SALES_SERIES : "Weekly_Sales(_Transformed)"
    STORE_DF ||--o| GBM_PRED : "hybrid only"
    SALES_SERIES ||--|| TRAIN_SERIES : "split(-test_size)"
    SALES_SERIES ||--|| TEST_SERIES : "split(last test_size)"
    STORE_DF ||--o{ EXOG_DF : "select_exog_variables"
    EXOG_DF ||--|| TRAIN_EXOG : "align"
    EXOG_DF ||--|| TEST_EXOG : "align"

    TRAIN_SERIES ||--o{ ARIMA_MODEL : "grid_search_arima"
    TRAIN_SERIES ||--o{ SARIMA_MODEL : "grid_search_sarima"
    ARIMA_MODEL ||--o{ ARIMA_FORECAST : "forecast(test)"
    SARIMA_MODEL ||--o{ SARIMA_FORECAST : "forecast(test)"
    ARIMA_FORECAST ||--o{ METRICS : "calculate_metrics"
    SARIMA_FORECAST ||--o{ METRICS : "calculate_metrics"
    METRICS ||--|| SUMMARY_REPORT : "save_summary_report"

    SALES_SERIES ||--o{ FINAL_MODEL : "create_final_model (refit)"
    FINAL_MODEL ||--o{ FUTURE_FORECAST : "predict_future"
    FUTURE_FORECAST ||--|| FUTURE_CSV : "to_csv"
    FUTURE_FORECAST ||--|| FUTURE_JSON : "json.dump"

    SALES_CSV {
        int Store
        float Weekly_Sales
        float GBM_Pred
        int WeekOfYear
    }
    METRICS {
        float mae
        float rmse
        float wmape
        float smape
        float mape
    }
    FINAL_MODEL {
        tuple order
        tuple seasonal_order
        string path
    }
```

---

## 렌더링 팁

- **GitHub**: `.md` 파일을 커밋하면 자동으로 Mermaid 블록이 렌더링됩니다.
- **PyCharm / IntelliJ**: Markdown 플러그인이 Mermaid를 기본 지원합니다. 프리뷰
  창(`Ctrl+Shift+A` → "Markdown Preview")에서 확인.
- **VS Code**: `Markdown Preview Mermaid Support` 확장 설치.
- **정적 HTML**: `mmdc -i docs/uml.md -o docs/uml.html` (mermaid-cli).
