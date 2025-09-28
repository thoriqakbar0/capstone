import marimo
from pathlib import Path

try:  # Optional fastapi integration
    from fastapi import FastAPI
except ImportError:  # pragma: no cover
    FastAPI = None

from marimo import create_asgi_app


def build_fastapi_app(
    base_path: str = "/",
    *,
    include_code: bool = True,
    quiet: bool = False,
):
    """Create a FastAPI server that mounts this marimo notebook."""

    if FastAPI is None:
        raise ImportError(
            "FastAPI is not installed. Install it with `uv add fastapi uvicorn`"
        )

    normalized = base_path if base_path.startswith("/") else f"/{base_path}"
    if normalized != "/":
        normalized = normalized.rstrip("/")

    server = create_asgi_app(include_code=include_code, quiet=quiet).with_app(
        path=normalized,
        root=Path(__file__).resolve(),
    )

    fastapi_app = FastAPI()

    @fastapi_app.get("/health", tags=["internal"])
    def health_check():  # pragma: no cover - trivial endpoint
        return {"status": "ok"}

    fastapi_app.mount(normalized, server.build())
    return fastapi_app


if FastAPI is not None:
    fastapi_app = build_fastapi_app()
else:  # pragma: no cover - FastAPI optional
    fastapi_app = None


__generated_with = "0.16.2"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _():
    return


@app.cell(hide_code=True)
def _():
    import numpy as np
    import pandas as pd
    import re
    from dataclasses import dataclass
    from joblib import dump, load
    import lightgbm as lgb
    from lightgbm import LGBMRegressor
    from pathlib import Path
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    ARTIFACT_DIR = Path("artifacts")
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    DATA_PATH = Path("property_dataset.csv")
    MODEL_PATH = ARTIFACT_DIR / "property_lightgbm.joblib"

    BASE_CONTRACT = [
        {"name": "STOREY/LEVEL", "kind": "numeric", "description": "Floor number of the unit."},
        {"name": "AREA", "kind": "numeric", "description": "Land area (square metres)."},
        {"name": "MFA", "kind": "numeric", "description": "Built-up / strata floor area."},
        {"name": "SALE_YEAR", "kind": "numeric", "description": "Sale calendar year."},
        {"name": "SALE_MONTH", "kind": "numeric", "description": "Sale month (1-12)."},
        {"name": "SALE_QUARTER", "kind": "numeric", "description": "Sale quarter (1-4)."},
        {"name": "LEASE_PERIOD_YEARS", "kind": "numeric", "description": "Lease tenure in years."},
        {"name": "TENURE", "kind": "categorical", "description": "Ownership tenure class."},
        {"name": "CATEGORY", "kind": "categorical", "description": "High-level land-use type."},
        {"name": "BUILDING TYPE", "kind": "categorical", "description": "Specific building type."},
        {"name": "LOCATION", "kind": "categorical", "description": "Development / locality label."},
        {"name": "SALE TYPE", "kind": "categorical", "description": "Primary or secondary sale."},
        {"name": "TITLE TYPE", "kind": "categorical", "description": "Title document classification."},
    ]

    def make_field_id(name: str) -> str:
        slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
        return slug

    FEATURE_CONTRACT = [
        {**feature, "field_id": make_field_id(feature["name"])} for feature in BASE_CONTRACT
    ]

    NUMERIC_FEATURE_NAMES = [
        feature["name"] for feature in FEATURE_CONTRACT if feature["kind"] == "numeric"
    ]
    CATEGORICAL_FEATURE_NAMES = [
        feature["name"] for feature in FEATURE_CONTRACT if feature["kind"] == "categorical"
    ]

    def normalize_category_text(value: object) -> object:
        if not isinstance(value, str):
            return value
        collapsed = re.sub(r"\s+", " ", value).strip()
        return collapsed if collapsed else np.nan

    def parse_price(value: str) -> float:
        if pd.isna(value) or not isinstance(value, str):
            return np.nan
        cleaned = re.sub(r"[^0-9.]", "", value)
        return float(cleaned) if cleaned else np.nan

    def parse_share(value: str) -> float:
        if not isinstance(value, str) or "/" not in value:
            return np.nan
        try:
            numerator, denominator = value.split("/", 1)
            numerator = float(numerator)
            denominator = float(denominator)
            return numerator / denominator if denominator else np.nan
        except ValueError:
            return np.nan

    def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
        frame = df.copy()
        frame["PRICE"] = frame["PRICE"].apply(parse_price)
        frame["STOREY/LEVEL"] = pd.to_numeric(frame["STOREY/LEVEL"], errors="coerce")
        frame["AREA"] = pd.to_numeric(frame["AREA"], errors="coerce")
        frame["MFA"] = pd.to_numeric(frame["MFA"], errors="coerce")

        sale_date = pd.to_datetime(frame["Date"], format="%d-%b-%y", errors="coerce")
        frame["SALE_YEAR"] = sale_date.dt.year
        frame["SALE_MONTH"] = sale_date.dt.month
        frame["SALE_QUARTER"] = sale_date.dt.quarter
        frame["SALE_DATE"] = sale_date

        lease_period = frame.get("LEASE PERIOD", pd.Series(dtype=str))
        frame["LEASE_PERIOD_YEARS"] = (
            lease_period.fillna("").str.extract(r"(\d+)").astype(float).squeeze()
        )

        frame["OWNERSHIP_SHARE"] = frame.get("SHARE", "").apply(parse_share)

        for column in CATEGORICAL_FEATURE_NAMES:
            if column in frame:
                frame[column] = frame[column].map(normalize_category_text)
        return frame

    raw_df = pd.read_csv(DATA_PATH)
    engineered_df = engineer_features(raw_df)

    def _feature_defaults(df: pd.DataFrame) -> dict:
        defaults: dict[str, float | str | None] = {}
        for feature in FEATURE_CONTRACT:
            name = feature["name"]
            if feature["kind"] == "numeric" and name in df:
                median_value = df[name].median()
                defaults[name] = float(median_value) if not pd.isna(median_value) else 0.0
            elif feature["kind"] == "categorical" and name in df:
                series = df[name].dropna()
                defaults[name] = series.mode(dropna=True).iloc[0] if not series.empty else ""
            else:
                defaults[name] = 0.0 if feature["kind"] == "numeric" else ""
        return defaults

    def _feature_options(df: pd.DataFrame) -> dict[str, list[str]]:
        options: dict[str, list[str]] = {}
        for feature in FEATURE_CONTRACT:
            name = feature["name"]
            if feature["kind"] == "categorical" and name in df:
                series = df[name].dropna()
                if series.empty:
                    options[name] = []
                    continue
                cleaned = (
                    series.astype(str)
                    .str.replace(r"\s+", " ", regex=True)
                    .str.strip()
                )
                cleaned = cleaned[cleaned != ""]
                options[name] = sorted(cleaned.drop_duplicates().tolist())
        return options

    feature_defaults_map = _feature_defaults(engineered_df)
    feature_options_map = _feature_options(engineered_df)

    feature_order = NUMERIC_FEATURE_NAMES + CATEGORICAL_FEATURE_NAMES

    location_activity = engineered_df.groupby("LOCATION").size().to_dict()
    storey_suggestions = (
        engineered_df.groupby("BUILDING TYPE")["STOREY/LEVEL"].median().dropna().to_dict()
    )

    with np.errstate(divide="ignore", invalid="ignore"):
        price_per_sqft_series = engineered_df["PRICE"].astype(float) / engineered_df[
            "AREA"
        ].astype(float)
        area_ratio_series = engineered_df["MFA"].astype(float) / engineered_df[
            "AREA"
        ].astype(float)

    valid_area_mask = (
        engineered_df["PRICE"].astype(float).notna()
        & engineered_df["AREA"].astype(float).notna()
        & (engineered_df["AREA"].astype(float) > 0)
    )

    area_series = engineered_df.loc[valid_area_mask, "AREA"].astype(float)

    area_quantiles = {
        "min": float(area_series.min()) if not area_series.empty else 0.0,
        "q10": float(area_series.quantile(0.10)) if not area_series.empty else 0.0,
        "median": float(area_series.quantile(0.50)) if not area_series.empty else 0.0,
        "q90": float(area_series.quantile(0.90)) if not area_series.empty else 0.0,
        "max": float(area_series.max()) if not area_series.empty else 0.0,
    }

    price_per_sqm_by_location: dict[str, float] = {}
    if valid_area_mask.any():
        per_location = engineered_df.loc[
            valid_area_mask & engineered_df["LOCATION"].notna(),
            ["LOCATION", "AREA", "PRICE"],
        ].copy()
        if not per_location.empty:
            per_location["AREA"] = per_location["AREA"].astype(float)
            per_location["PRICE"] = per_location["PRICE"].astype(float)
            per_location["PRICE_PER_SQM"] = per_location["PRICE"] / per_location["AREA"]
            price_per_sqm_by_location = (
                per_location.groupby("LOCATION")["PRICE_PER_SQM"].median().dropna().astype(float).to_dict()
            )

    price_per_sqft_median = (
        float(np.nanmedian(price_per_sqft_series))
        if price_per_sqft_series.notna().any()
        else 0.0
    )
    area_to_mfa_ratio = (
        float(np.nanmedian(area_ratio_series))
        if area_ratio_series.notna().any()
        else 1.0
    )

    parsed_dates = engineered_df["SALE_DATE"].dropna()
    date_window = {
        "min": str(parsed_dates.min().date()) if not parsed_dates.empty else "",
        "max": str(parsed_dates.max().date()) if not parsed_dates.empty else "",
    }

    price_history = engineered_df.dropna(subset=["PRICE"]).copy()
    price_history = price_history.dropna(subset=["SALE_DATE"])

    monthly_price_trend = (
        price_history.groupby(pd.Grouper(key="SALE_DATE", freq="M"))
        .agg(
            median_price=("PRICE", "median"),
            average_price=("PRICE", "mean"),
            transactions=("PRICE", "size"),
        )
        .reset_index()
        .sort_values("SALE_DATE")
    )
    if not monthly_price_trend.empty:
        monthly_price_trend["SALE_MONTH_LABEL"] = monthly_price_trend["SALE_DATE"].dt.strftime("%Y-%m")

    top_location_summary = (
        price_history[price_history["LOCATION"].notna()]
        .groupby("LOCATION")
        .agg(
            median_price=("PRICE", "median"),
            average_price=("PRICE", "mean"),
            transactions=("PRICE", "size"),
        )
        .reset_index()
        .sort_values("transactions", ascending=False)
        .head(12)
    )
    top_location_summary["LOCATION"] = top_location_summary["LOCATION"].astype(str)

    building_type_summary = (
        price_history[price_history["BUILDING TYPE"].notna()]
        .groupby("BUILDING TYPE")
        .agg(
            median_price=("PRICE", "median"),
            average_price=("PRICE", "mean"),
            transactions=("PRICE", "size"),
        )
        .reset_index()
        .sort_values("median_price", ascending=False)
        .head(12)
    )
    building_type_summary["BUILDING TYPE"] = building_type_summary["BUILDING TYPE"].astype(str)

    def _as_float(value: float | int | None) -> float | None:
        if pd.isna(value):
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _as_int(value: object) -> int:
        if pd.isna(value):
            return 0
        try:
            return int(value)
        except (TypeError, ValueError):
            return 0

    trend_summary: dict[str, object] = {}
    if not monthly_price_trend.empty:
        latest_row = monthly_price_trend.iloc[-1]
        latest_label = latest_row.get("SALE_MONTH_LABEL") or latest_row["SALE_DATE"].strftime("%Y-%m")
        latest_median = _as_float(latest_row.get("median_price"))
        latest_transactions = _as_int(latest_row.get("transactions", 0))

        trend_summary.update(
            {
                "latest_month_label": latest_label,
                "latest_median": latest_median,
                "latest_transactions": latest_transactions,
            }
        )

        if len(monthly_price_trend) >= 2:
            prev_row = monthly_price_trend.iloc[-2]
            prev_median = _as_float(prev_row.get("median_price"))
            if prev_median is not None and prev_median != 0 and latest_median is not None:
                mom_change = latest_median - prev_median
                trend_summary.update(
                    {
                        "mom_change": mom_change,
                        "mom_pct": mom_change / prev_median,
                    }
                )

        latest_date = latest_row["SALE_DATE"]
        try:
            yoy_target = (latest_date - pd.DateOffset(years=1)).to_period("M")
        except Exception:
            yoy_target = None
        if yoy_target is not None:
            monthly_periods = monthly_price_trend["SALE_DATE"].dt.to_period("M")
            yoy_matches = monthly_price_trend.loc[monthly_periods == yoy_target]
            if not yoy_matches.empty:
                yoy_row = yoy_matches.iloc[-1]
                yoy_median = _as_float(yoy_row.get("median_price"))
                if yoy_median is not None and yoy_median != 0 and latest_median is not None:
                    yoy_change = latest_median - yoy_median
                    trend_summary.update(
                        {
                            "yoy_change": yoy_change,
                            "yoy_pct": yoy_change / yoy_median,
                            "yoy_month_label": yoy_row.get("SALE_MONTH_LABEL") or yoy_row["SALE_DATE"].strftime("%Y-%m"),
                        }
                    )

    if not top_location_summary.empty:
        top_by_deals = top_location_summary.iloc[0]
        trend_summary["top_location_by_deals"] = {
            "name": str(top_by_deals.get("LOCATION", "")),
            "transactions": _as_int(top_by_deals.get("transactions", 0)),
            "median_price": _as_float(top_by_deals.get("median_price")),
        }
        top_by_price = top_location_summary.sort_values("median_price", ascending=False).iloc[0]
        trend_summary["top_location_by_price"] = {
            "name": str(top_by_price.get("LOCATION", "")),
            "median_price": _as_float(top_by_price.get("median_price")),
            "transactions": _as_int(top_by_price.get("transactions", 0)),
        }

    if not building_type_summary.empty:
        building_focus = building_type_summary.iloc[0]
        trend_summary["top_building_type"] = {
            "name": str(building_focus.get("BUILDING TYPE", "")),
            "median_price": _as_float(building_focus.get("median_price")),
            "transactions": _as_int(building_focus.get("transactions", 0)),
        }


    def build_preprocessor() -> ColumnTransformer:
        numeric_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )

        categorical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]
        )

        return ColumnTransformer(
            transformers=[
                ("numeric", numeric_pipeline, NUMERIC_FEATURE_NAMES),
                ("categorical", categorical_pipeline, CATEGORICAL_FEATURE_NAMES),
            ]
        )

    def build_model_pipeline() -> Pipeline:
        return Pipeline(
            [
                ("preprocess", build_preprocessor()),
                ("model", LGBMRegressor(
                    random_state=42,
                    n_estimators=500,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                )),
            ]
        )


    def _build_training_history_df(evals_result: dict[str, dict[str, list[float]]]) -> pd.DataFrame:
        if not evals_result:
            return pd.DataFrame(columns=["iteration", "dataset", "metric", "value"])
        records: list[dict[str, object]] = []
        for dataset_name, metrics_map in evals_result.items():
            friendly_dataset = dataset_name.replace('validation_0', 'Train').replace('validation_1', 'Validation')
            for metric_name, values in metrics_map.items():
                for idx, metric_value in enumerate(values, start=1):
                    records.append(
                        {
                            "iteration": idx,
                            "dataset": friendly_dataset,
                            "metric": metric_name.upper(),
                            "value": float(metric_value),
                        }
                    )
        return pd.DataFrame.from_records(records)

    def _compute_feature_importance(preprocessor: ColumnTransformer | None, model: LGBMRegressor) -> pd.DataFrame:
        importances = getattr(model, "feature_importances_", None)
        if importances is None:
            return pd.DataFrame(columns=["feature", "importance"])
        try:
            feature_names = preprocessor.get_feature_names_out() if preprocessor is not None else None
        except Exception:
            feature_names = None
        if feature_names is None:
            feature_names = [f"feature_{idx}" for idx in range(len(importances))]
        importance_df = pd.DataFrame(
            {
                "feature": feature_names,
                "importance": importances,
            }
        )
        importance_df = importance_df.sort_values("importance", ascending=False).reset_index(drop=True)
        return importance_df

    def train_model(clicks: int, reuse_existing: bool):
        should_load = reuse_existing and MODEL_PATH.exists() and clicks == 0
        if should_load:
            pipeline = load(MODEL_PATH)
            metrics = None
            training_history_df = pd.DataFrame()
            feature_importance_df = pd.DataFrame()
            if pipeline is not None:
                try:
                    model = pipeline.named_steps.get("model")
                    preprocessor = pipeline.named_steps.get("preprocess")
                    if model is not None and hasattr(model, "feature_importances_"):
                        feature_importance_df = _compute_feature_importance(preprocessor, model)
                except Exception:
                    feature_importance_df = pd.DataFrame()
            status = f"Loaded cached model from {MODEL_PATH}"
            return status, metrics, pipeline, training_history_df, feature_importance_df

        prepared = engineered_df.dropna(subset=["PRICE"])
        missing_columns = [
            name for name in feature_order if name not in prepared.columns
        ]
        if missing_columns:
            raise ValueError(f"Missing engineered features: {missing_columns}")

        X = prepared[feature_order]
        y = prepared["PRICE"].astype(float)

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
        )

        preprocessor = build_preprocessor()
        X_train_processed = preprocessor.fit_transform(X_train, y_train)
        X_test_processed = preprocessor.transform(X_test)

        model = LGBMRegressor(
            random_state=42,
            n_estimators=500,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
        )

        evals_result: dict[str, dict[str, list[float]]] = {}
        callbacks = [
            lgb.callback.record_evaluation(evals_result),
            lgb.callback.log_evaluation(period=0),
        ]

        model.fit(
            X_train_processed,
            y_train,
            eval_set=[(X_train_processed, y_train), (X_test_processed, y_test)],
            eval_metric=["rmse", "l1"],
            callbacks=callbacks,
        )

        y_pred = model.predict(X_test_processed)
        metrics = {
            "MAE": float(mean_absolute_error(y_test, y_pred)),
            "RMSE": float(np.sqrt(mean_squared_error(y_test, y_pred))),
            "R2": float(r2_score(y_test, y_pred)),
        }

        training_history_df = _build_training_history_df(evals_result)
        feature_importance_df = _compute_feature_importance(preprocessor, model)

        pipeline = Pipeline([
            ("preprocess", preprocessor),
            ("model", model),
        ])

        dump(pipeline, MODEL_PATH)
        status = (
            f"Trained LightGBM on {len(X_train)} rows and saved to {MODEL_PATH.name}."
        )
        return status, metrics, pipeline, training_history_df, feature_importance_df
    context = {
        "train_model": train_model,
        "feature_contract": FEATURE_CONTRACT,
        "feature_defaults": feature_defaults_map,
        "feature_options": feature_options_map,
        "feature_order": feature_order,
        "location_activity": location_activity,
        "storey_suggestions": storey_suggestions,
        "price_per_sqft_median": price_per_sqft_median,
        "area_to_mfa_ratio": area_to_mfa_ratio,
        "area_stats": {"quantiles": area_quantiles},
        "price_per_sqm_by_location": price_per_sqm_by_location,
        "date_window": date_window,
        "monthly_price_trend": monthly_price_trend,
        "location_summary": top_location_summary,
        "building_type_summary": building_type_summary,
        "trend_summary": trend_summary,
    }
    return context, pd


@app.cell
def _(context, mo):
    train_button = mo.ui.run_button(label="Train / Refresh LightGBM", full_width=True)
    reuse_toggle = mo.ui.switch(label="Reuse saved model if available", value=True)

    training_controls = mo.hstack(
        [
            train_button,
            reuse_toggle,
        ],
        align="center",
        gap=1.0,
        wrap=True,
    )

    @mo.cache
    def compute_training(clicks: int, reuse_existing: bool):
        status, metrics, pipeline, training_history_df, feature_importance_df = context["train_model"](clicks, reuse_existing)
        return status, metrics, pipeline, training_history_df, feature_importance_df

    return compute_training, reuse_toggle, train_button, training_controls


@app.cell
def _(compute_training, reuse_toggle, train_button):
    status_msg, metrics_dict, pipeline, training_history_df, feature_importance_df = compute_training(
        train_button.value, reuse_toggle.value
    )
    return metrics_dict, pipeline, status_msg, training_history_df, feature_importance_df


@app.cell
def _(context, mo):
    feature_contract = context["feature_contract"]
    defaults_map = context["feature_defaults"]
    options_map = context["feature_options"]
    feature_order_list = context["feature_order"]

    controls: dict[str, object] = {}
    name_to_field: dict[str, str] = {}

    for feature in feature_contract:
        name = feature["name"]
        field_id = feature["field_id"]
        if feature["kind"] == "numeric":
            initial_value = defaults_map.get(name, 0.0)
            element = mo.ui.number(
                label=name,
                value=float(initial_value) if initial_value is not None else 0.0,
                step=1,
                full_width=True,
            )
        else:
            options = options_map.get(name, [])
            initial_choice = defaults_map.get(name, options[0] if options else "")
            element = mo.ui.dropdown(
                options=options or [initial_choice],
                value=initial_choice,
                label=name,
                full_width=True,
                searchable=True,
            )

        controls[field_id] = element
        name_to_field[name] = field_id

    bundled_controls = mo.ui.dictionary(controls)
    prediction_form = mo.ui.form(
        bundled_controls,
        submit_button_label="Run Detailed Predict",
        bordered=False,
        show_clear_button=True,
        clear_button_label="Reset",
    )
    predict_button = mo.ui.run_button(label="Run Quick Predict")
    prediction_mode_switch = mo.ui.switch(
        label="Detailed Predict",
        value=False,
    )
    return (
        controls,
        feature_contract,
        feature_order_list,
        name_to_field,
        prediction_form,
        prediction_mode_switch,
        predict_button,
    )


@app.cell
def _(
    context,
    controls: dict[str, object],
    feature_contract,
    feature_order_list,
    metrics_dict,
    mo,
    name_to_field: dict[str, str],
    pd,
    pipeline,
    prediction_form,
    prediction_mode_switch,
    predict_button,
    status_msg,
    training_controls,
    training_history_df,
    feature_importance_df,
):
    ctx_defaults = context["feature_defaults"]
    ctx_location_activity = context["location_activity"]
    ctx_storey_suggestions = context["storey_suggestions"]
    ctx_price_per_sqft_median = context["price_per_sqft_median"]
    ctx_area_to_mfa_ratio = context["area_to_mfa_ratio"]
    ctx_area_stats = context.get("area_stats", {})
    ctx_price_per_sqm_by_location = context.get("price_per_sqm_by_location", {})
    ctx_date_window = context["date_window"]
    ctx_monthly_trend = context.get("monthly_price_trend", pd.DataFrame())
    ctx_location_summary = context.get("location_summary", pd.DataFrame())
    ctx_building_summary = context.get("building_type_summary", pd.DataFrame())
    ctx_trend_summary = context.get("trend_summary", {})

    ctx_area_quantiles = ctx_area_stats.get("quantiles", {})
    SQM_TO_SQFT = 10.7639

    primary_names = ["AREA", "BUILDING TYPE", "LOCATION"]
    advanced_names = [
        feature["name"]
        for feature in feature_contract
        if feature["name"] not in primary_names
    ]

    def control_value(name: str):
        field_id = name_to_field[name]
        element = controls[field_id]
        value = element.value
        return value if value not in (None, "") else ctx_defaults.get(name)

    live_area = float(control_value("AREA") or 0.0)
    live_location = control_value("LOCATION") or ""
    live_building = control_value("BUILDING TYPE") or ""
    button_triggered = predict_button.value > 0

    storey_value = control_value("STOREY/LEVEL")
    if storey_value in (None, ""):
        storey_value = ctx_storey_suggestions.get(
            live_building, ctx_defaults.get("STOREY/LEVEL", 1)
        )
    live_storey = float(storey_value or 0.0)

    floor_area_estimate = (
        live_area * ctx_area_to_mfa_ratio if live_area else ctx_defaults.get("MFA", 0.0)
    )
    location_activity_count = int(ctx_location_activity.get(str(live_location), 0))

    metrics_lines = (
        "\n".join(f"- **{name}**: {value:,.2f}" for name, value in metrics_dict.items())
        if metrics_dict
        else "- Metrics unavailable (loaded cached model)."
    )

    status_callout = mo.callout(
        mo.md(
            f"""**{status_msg}**\n\n{metrics_lines}\n\n_Data window: {ctx_date_window["min"]} → {ctx_date_window["max"]}_"""
        ),
        kind="success",
    )

    header = mo.vstack(
        [
            mo.hstack(
                [
                    mo.icon("lucide:home", size=26),
                    mo.md("## Malaysian Property Price Predictor"),
                ],
                gap=0.5,
                align="center",
                wrap=True,
            ),
            mo.md(
                "_Predict property price based on area, building type, and location_"
            ),
            status_callout,
            training_controls,
        ],
        align="stretch",
        gap=0.6,
    )

    def card(title: str, element, kind: str = "neutral"):
        return mo.callout(
            mo.vstack([mo.md(f"**{title}**"), element], align="stretch", gap=0.35),
            kind=kind,
        )

    property_cards = [
        card("Property Size (sq m)", controls[name_to_field["AREA"]]),
        card("Building Type", controls[name_to_field["BUILDING TYPE"]]),
        card("Location", controls[name_to_field["LOCATION"]]),
    ]

    q10_area = ctx_area_quantiles.get("q10") if isinstance(ctx_area_quantiles, dict) else None
    q90_area = ctx_area_quantiles.get("q90") if isinstance(ctx_area_quantiles, dict) else None
    min_area = ctx_area_quantiles.get("min") if isinstance(ctx_area_quantiles, dict) else None
    max_area = ctx_area_quantiles.get("max") if isinstance(ctx_area_quantiles, dict) else None

    area_range_notice = None
    if (
        isinstance(q10_area, (int, float))
        and isinstance(q90_area, (int, float))
        and q10_area > 0
        and q90_area > 0
    ):
        range_text = (
            f"Model training focused on areas between **{q10_area:,.0f}** and **{q90_area:,.0f}** sq m. "
            "Enter property size in square metres (1 sq m ≈ 10.764 sq ft)."
        )
        if isinstance(min_area, (int, float)) and isinstance(max_area, (int, float)) and max_area > 0:
            range_text += f" Observed dataset range: {min_area:,.0f}–{max_area:,.0f} sq m."
        area_range_notice = mo.callout(mo.md(range_text), kind="neutral")

    detailed_mode = bool(prediction_mode_switch.value)
    mode_label = "Detailed" if detailed_mode else "Quick"

    mode_toggle = mo.hstack(
        [mo.md("**Detailed Predict**"), prediction_mode_switch],
        gap=0.35,
        align="center",
        wrap=True,
    )

    mode_controls: list[object] = []
    if not detailed_mode:
        mode_controls.append(predict_button)
    mode_controls.append(mode_toggle)

    mode_controls_row = mo.hstack(mode_controls, gap=0.8, align="center", wrap=True)

    if detailed_mode:
        mode_help_text = (
            "**Detailed Predict** lets you override every feature using the full form. Submit the detailed form to refresh the estimate."
        )
    else:
        mode_help_text = (
            "**Quick Predict** uses the property cards above. Toggle Detailed Predict to switch into the comprehensive workflow."
        )

    prediction_mode_card = card(
        "Prediction Modes",
        mo.vstack([mo.md(mode_help_text)], align="stretch", gap=0.2),
        kind="neutral",
    )

    advanced_elements = [controls[name_to_field[name]] for name in advanced_names]
    advanced_section = (
        mo.accordion(
            {"Advanced Inputs": mo.vstack(advanced_elements, gap=0.45)}, multiple=False
        )
        if advanced_elements
        else None
    )

    auto_cards = mo.hstack(
        [
            card(
                "Floor Area (est.)",
                mo.md(f"{floor_area_estimate:,.0f} sq m"),
                kind="info",
            ),
            card("Storeys", mo.md(f"{live_storey:,.0f}"), kind="info"),
            card(
                "Location Activity",
                mo.md(f"{location_activity_count} transactions"),
                kind="info",
            ),
        ],
        gap=1,
        widths=[1, 1, 1],
        wrap=True,
    )
    auto_section = mo.vstack(
        [mo.md("### Auto-Calculated Values"), auto_cards], align="stretch", gap=0.6
    )


    altair_chart_fn = getattr(mo.ui, "altair_chart", None)
    alt_module = None
    if altair_chart_fn is not None:
        try:
            import altair as alt  # type: ignore

            alt.data_transformers.disable_max_rows()
            alt_module = alt
        except ImportError:
            alt_module = None

    def calibrate_prediction(
        area: float | None,
        raw_price: float | None,
        reference_rate: float | None,
        quantiles: dict[str, float],
    ) -> dict[str, float | None]:
        try:
            area_val = float(area) if area is not None else 0.0
        except (TypeError, ValueError):
            area_val = 0.0

        try:
            raw_price_val = float(raw_price) if raw_price is not None else None
        except (TypeError, ValueError):
            raw_price_val = None

        raw_rate = raw_price_val / area_val if raw_price_val is not None and area_val > 0 else None

        try:
            reference_rate_val = float(reference_rate) if reference_rate is not None else None
        except (TypeError, ValueError):
            reference_rate_val = None

        min_area = float(quantiles.get("min") or 0.0)
        q10 = float(quantiles.get("q10") or 0.0)
        q90 = float(quantiles.get("q90") or 0.0)
        max_area = float(quantiles.get("max") or 0.0)

        blend_weight = 0.0
        if area_val > 0 and q10 > 0 and q90 > 0 and max_area > 0:
            if area_val < q10:
                denominator = max(q10 - max(min_area, 0.0), 1e-9)
                blend_weight = min(1.0, (q10 - area_val) / denominator)
            elif area_val > q90:
                denominator = max(max_area - q90, 1e-9)
                blend_weight = min(1.0, (area_val - q90) / denominator)

        blend_weight = max(0.0, min(blend_weight, 1.0))

        if reference_rate_val is None:
            reference_rate_val = raw_rate if raw_rate is not None else 0.0

        if raw_rate is None:
            adjusted_rate = reference_rate_val
        else:
            adjusted_rate = (1.0 - blend_weight) * raw_rate + blend_weight * reference_rate_val

        adjusted_price = adjusted_rate * area_val if area_val > 0 else raw_price_val

        return {
            "raw_rate": raw_rate,
            "adjusted_rate": adjusted_rate,
            "adjusted_price": adjusted_price,
            "blend_weight": blend_weight,
            "reference_rate": reference_rate_val,
        }

    def format_rm(value: float | None) -> str:
        if value is None:
            return "RM —"
        try:
            if pd.isna(value):
                return "RM —"
        except TypeError:
            pass
        return f"RM {float(value):,.0f}"

    def format_rate(value: float | None) -> str:
        if value is None:
            return "—"
        try:
            if pd.isna(value):
                return "—"
        except TypeError:
            pass
        return f"{float(value):,.0f} RM/m²"

    def format_rate_pair(value: float | None) -> str:
        if value is None or (isinstance(value, (int, float)) and value <= 0):
            return "—"
        try:
            rate_val = float(value)
        except (TypeError, ValueError):
            return "—"
        per_sqft = rate_val / SQM_TO_SQFT if SQM_TO_SQFT else 0.0
        if per_sqft <= 0:
            return f"{rate_val:,.0f} RM/m²"
        return f"{rate_val:,.0f} RM/m²\n\n_{per_sqft:,.0f} RM/ft²_"

    def format_pct(value: float | None) -> str:
        if value is None:
            return "—"
        try:
            if pd.isna(value):
                return "—"
        except TypeError:
            pass
        return f"{float(value):+.1%}"

    def build_trend_highlights(summary: dict[str, object]) -> list[str]:
        lines: list[str] = []
        latest_label = summary.get("latest_month_label") if isinstance(summary, dict) else None
        latest_median = summary.get("latest_median") if isinstance(summary, dict) else None
        latest_transactions = summary.get("latest_transactions", 0) if isinstance(summary, dict) else 0
        if latest_label and isinstance(latest_median, (int, float)):
            base_line = f"- **{latest_label} median**: {format_rm(float(latest_median))}"
            if isinstance(latest_transactions, (int, float)) and latest_transactions > 0:
                base_line += f" across {int(latest_transactions)} recorded sales"
            lines.append(base_line)

        mom_pct_val = summary.get("mom_pct") if isinstance(summary, dict) else None
        if isinstance(mom_pct_val, (int, float)):
            mom_change_val = summary.get("mom_change") if isinstance(summary, dict) else None
            delta_fragment = ""
            if isinstance(mom_change_val, (int, float)):
                delta_fragment = f" ({format_rm(float(mom_change_val))} vs prior month)"
            lines.append(f"- Month-over-month: {format_pct(float(mom_pct_val))}{delta_fragment}")

        yoy_pct_val = summary.get("yoy_pct") if isinstance(summary, dict) else None
        if isinstance(yoy_pct_val, (int, float)):
            yoy_month_label = summary.get("yoy_month_label") if isinstance(summary, dict) else None
            yoy_change_val = summary.get("yoy_change") if isinstance(summary, dict) else None
            change_fragment = ""
            if isinstance(yoy_change_val, (int, float)):
                comparison_label = yoy_month_label or "last year"
                change_fragment = f" ({format_rm(float(yoy_change_val))} vs {comparison_label})"
            lines.append(f"- Year-over-year: {format_pct(float(yoy_pct_val))}{change_fragment}")

        top_location = summary.get("top_location_by_deals") if isinstance(summary, dict) else None
        if isinstance(top_location, dict) and top_location.get("name"):
            lines.append(
                f"- **Most active area**: {top_location['name']} ("
                f"{int(top_location.get('transactions', 0))} sales, median {format_rm(top_location.get('median_price'))})"
            )

        top_price_location = summary.get("top_location_by_price") if isinstance(summary, dict) else None
        if (
            isinstance(top_price_location, dict)
            and top_price_location.get("name")
            and (not isinstance(top_location, dict) or top_price_location.get("name") != top_location.get("name"))
        ):
            lines.append(
                f"- **Highest median price**: {top_price_location['name']} (median {format_rm(top_price_location.get('median_price'))})"
            )

        top_building = summary.get("top_building_type") if isinstance(summary, dict) else None
        if isinstance(top_building, dict) and top_building.get("name"):
            lines.append(
                f"- **Top building type**: {top_building['name']} (median {format_rm(top_building.get('median_price'))}, "
                f"{int(top_building.get('transactions', 0))} sales)"
            )

        return lines

    market_section = None
    has_price_history = not ctx_monthly_trend.empty

    highlight_lines = build_trend_highlights(ctx_trend_summary)
    highlight_block = (
        mo.callout(
            mo.md("**Market snapshot**\n\n" + "\n".join(highlight_lines)),
            kind="info",
        )
        if highlight_lines
        else None
    )

    if alt_module is not None and callable(altair_chart_fn):
        accordion_content: list[object] = []
        if highlight_block is not None:
            accordion_content.append(highlight_block)

        monthly_trend_df = ctx_monthly_trend.copy()
        price_trend_card = None
        if not monthly_trend_df.empty:
            price_trend_chart = (
                alt_module.Chart(monthly_trend_df)
                .mark_line(point=True, color="#2563eb")
                .encode(
                    x=alt_module.X("SALE_DATE:T", title="Sale month"),
                    y=alt_module.Y("median_price:Q", title="Median price (RM)", axis=alt_module.Axis(format="~s")),
                    tooltip=[
                        alt_module.Tooltip("SALE_MONTH_LABEL:N", title="Month"),
                        alt_module.Tooltip("median_price:Q", title="Median Price", format=",.0f"),
                        alt_module.Tooltip("transactions:Q", title="Transactions"),
                    ],
                )
                .properties(height=260)
                .interactive()
            )
            price_trend_card = card("Median Price Trend", altair_chart_fn(price_trend_chart), kind="info")

        bar_cards: list[object] = []
        location_df = ctx_location_summary.copy()
        if not location_df.empty:
            location_chart = (
                alt_module.Chart(location_df)
                .mark_bar()
                .encode(
                    x=alt_module.X("transactions:Q", title="Transactions"),
                    y=alt_module.Y("LOCATION:N", sort="-x", title="Location"),
                    color=alt_module.Color(
                        "median_price:Q",
                        title="Median price (RM)",
                        scale=alt_module.Scale(scheme="blues"),
                    ),
                    tooltip=[
                        alt_module.Tooltip("LOCATION:N", title="Location"),
                        alt_module.Tooltip("median_price:Q", title="Median Price", format=",.0f"),
                        alt_module.Tooltip("transactions:Q", title="Transactions"),
                    ],
                )
                .properties(height=300)
            )
            bar_cards.append(card("Top Locations", altair_chart_fn(location_chart), kind="neutral"))

        building_df = ctx_building_summary.copy()
        if not building_df.empty:
            building_chart = (
                alt_module.Chart(building_df)
                .mark_bar(color="#059669")
                .encode(
                    x=alt_module.X("median_price:Q", title="Median price (RM)", axis=alt_module.Axis(format="~s")),
                    y=alt_module.Y("BUILDING TYPE:N", sort="-x", title="Building type"),
                    tooltip=[
                        alt_module.Tooltip("BUILDING TYPE:N", title="Building type"),
                        alt_module.Tooltip("median_price:Q", title="Median Price", format=",.0f"),
                        alt_module.Tooltip("transactions:Q", title="Transactions"),
                    ],
                )
                .properties(height=300)
            )
            bar_cards.append(
                card("Median Price by Building Type", altair_chart_fn(building_chart), kind="neutral")
            )

        if price_trend_card is not None:
            accordion_content.append(price_trend_card)
        if bar_cards:
            accordion_content.append(
                mo.hstack(bar_cards, gap=1, wrap=True, widths=[1] * len(bar_cards))
            )

        if highlight_block is not None or price_trend_card is not None or bar_cards:
            market_section = mo.accordion(
                {"Market Trends": mo.vstack(accordion_content, align="stretch", gap=0.8)},
                multiple=False,
            )



    model_section = None
    diag_history_df = training_history_df if isinstance(training_history_df, pd.DataFrame) else pd.DataFrame()
    diag_importance_df = feature_importance_df if isinstance(feature_importance_df, pd.DataFrame) else pd.DataFrame()

    if alt_module is not None and callable(altair_chart_fn):
        model_cards: list[object] = []

        history_df = diag_history_df.copy()
        if not history_df.empty:
            if "metric" in history_df:
                metric_series = history_df["metric"].astype(str)
                history_rmse = history_df[metric_series.str.upper() == "RMSE"]
            else:
                history_rmse = history_df
            if not history_rmse.empty:
                training_chart = (
                    alt_module.Chart(history_rmse)
                    .mark_line(point=True)
                    .encode(
                        x=alt_module.X("iteration:Q", title="Iteration"),
                        y=alt_module.Y("value:Q", title="RMSE"),
                        color=alt_module.Color("dataset:N", title="Dataset"),
                        tooltip=[
                            alt_module.Tooltip("dataset:N", title="Dataset"),
                            alt_module.Tooltip("iteration:Q", title="Iteration"),
                            alt_module.Tooltip("value:Q", title="RMSE", format=",.3f"),
                        ],
                    )
                    .properties(height=260)
                    .interactive()
                )
                model_cards.append(card("LightGBM RMSE Curve", altair_chart_fn(training_chart), kind="info"))

        importance_df = diag_importance_df.copy()
        if not importance_df.empty:
            top_importance = importance_df.head(20)
            importance_chart = (
                alt_module.Chart(top_importance)
                .mark_bar(color="#7c3aed")
                .encode(
                    x=alt_module.X("importance:Q", title="Importance"),
                    y=alt_module.Y("feature:N", sort="-x", title="Feature"),
                    tooltip=[
                        alt_module.Tooltip("feature:N", title="Feature"),
                        alt_module.Tooltip("importance:Q", title="Importance", format=",.0f"),
                    ],
                )
                .properties(height=300)
            )
            model_cards.append(card("Feature Importance", altair_chart_fn(importance_chart), kind="neutral"))

        if model_cards:
            model_section = mo.accordion(
                {"Model Training": mo.vstack(model_cards, align="stretch", gap=0.8)},
                multiple=False,
            )

    if model_section is None:
        fallback = mo.callout(
            "Train or refresh the model to view LightGBM diagnostics.",
            kind="neutral",
        )
        model_section = mo.accordion({"Model Training": fallback}, multiple=False)

    if market_section is None:
        fallback_items: list[object] = []
        if highlight_block is not None:
            fallback_items.append(highlight_block)

        if alt_module is None or not callable(altair_chart_fn):
            fallback_items.append(
                mo.callout(
                    "Altair is unavailable; install it with `uv add altair` to unlock trend charts.",
                    kind="warn",
                )
            )
        elif not has_price_history:
            fallback_items.append(
                mo.callout(
                    "Trend charts become available after loading transactions with price data.",
                    kind="neutral",
                )
            )
        else:
            fallback_items.append(mo.callout("Trend charts are unavailable.", kind="neutral"))

        market_section = mo.accordion(
            {"Market Trends": mo.vstack(fallback_items, align="stretch", gap=0.6)},
            multiple=False,
        )

    def form_payload(form_value: dict | None) -> dict:
        payload: dict[str, float | str | None] = {}
        if not form_value:
            for feature in feature_contract:
                payload[feature["name"]] = ctx_defaults.get(feature["name"])
            return payload
        for feature in feature_contract:
            field_id = feature["field_id"]
            payload[feature["name"]] = form_value.get(
                field_id, ctx_defaults.get(feature["name"])
            )
        return payload

    def payload_from_controls() -> dict[str, float | str | None]:
        live_payload: dict[str, float | str | None] = {}
        for feature in feature_contract:
            name = feature["name"]
            live_payload[name] = control_value(name)
        return live_payload

    def predict_price(payload: dict):
        if pipeline is None or not payload:
            return None
        ordered = {
            name: payload.get(name, ctx_defaults.get(name))
            for name in feature_order_list
        }
        frame = pd.DataFrame([ordered], columns=feature_order_list)
        feature_names = getattr(pipeline, "feature_names_in_", None)
        if feature_names is not None:
            frame = frame.reindex(columns=feature_names)
        prediction = float(pipeline.predict(frame)[0])
        spread = 0.1 * prediction
        return {
            "prediction": prediction,
            "low": prediction - spread,
            "high": prediction + spread,
        }

    form_submission = prediction_form.value
    triggered_by_form = bool(form_submission)
    triggered_by_button = button_triggered and not detailed_mode

    quick_payload = payload_from_controls()
    detailed_payload = form_payload(form_submission) if triggered_by_form else {}

    if detailed_mode:
        submitted_payload = detailed_payload
        prediction_ready = triggered_by_form
    else:
        submitted_payload = quick_payload
        prediction_ready = triggered_by_button

    prediction = predict_price(submitted_payload) if prediction_ready else None

    def format_currency(value: float | None) -> str:
        if value is None:
            return "RM —"
        try:
            return f"RM {float(value):,.0f}"
        except (TypeError, ValueError):
            return "RM —"

    if pipeline is None:
        prediction_section = mo.callout(
            "Train or load the model to enable predictions.", kind="warn"
        )
    elif not prediction_ready or prediction is None:
        if detailed_mode:
            prompt = (
                "Complete the detailed form below and click **Run Detailed Predict** to generate an estimate."
            )
        else:
            prompt = (
                "Adjust the property details above and click **Run Quick Predict** to generate an estimate."
            )
        prediction_section = mo.callout(prompt, kind="neutral")
    else:
        estimate = prediction["prediction"]
        low = prediction["low"]
        high = prediction["high"]

        reference_rate = ctx_price_per_sqm_by_location.get(
            str(live_location), ctx_price_per_sqft_median
        )
        calibration = calibrate_prediction(
            live_area,
            estimate,
            reference_rate,
            ctx_area_quantiles if isinstance(ctx_area_quantiles, dict) else {},
        )

        adjusted_estimate = calibration.get("adjusted_price") or estimate
        blend_weight = float(calibration.get("blend_weight") or 0.0)
        raw_rate = calibration.get("raw_rate")
        calibrated_rate = calibration.get("adjusted_rate") or raw_rate
        comparison_rate = calibration.get("reference_rate") or ctx_price_per_sqft_median

        price_per_sq_m = (
            calibrated_rate
            if calibrated_rate is not None
            else (estimate / max(live_area, 1.0) if live_area else None)
        )
        raw_price_per_sq_m = (
            raw_rate
            if raw_rate is not None
            else (estimate / max(live_area, 1.0) if live_area else None)
        )

        adjusted_low = adjusted_estimate * 0.9 if adjusted_estimate is not None else low
        adjusted_high = adjusted_estimate * 1.1 if adjusted_estimate is not None else high

        price_callout = card(
            "Calibrated Price", mo.md(format_currency(adjusted_estimate)), kind="success"
        )
        raw_price_callout = card(
            "Raw Model Price", mo.md(format_currency(estimate)), kind="neutral"
        )
        pps_callout = card(
            "Calibrated Price per sq m",
            mo.md(format_rate_pair(price_per_sq_m)),
            kind="info",
        )
        raw_pps_callout = card(
            "Raw Price per sq m",
            mo.md(format_rate_pair(raw_price_per_sq_m)),
            kind="neutral",
        )
        range_callout = card(
            "Calibrated Range",
            mo.md(
                f"{format_currency(adjusted_low)} – {format_currency(adjusted_high)}"
            ),
            kind="success",
        )

        comparison_label = (
            f"Reference Median ({live_location})"
            if str(live_location) in ctx_price_per_sqm_by_location
            else "Dataset Median"
        )
        comparison_callout = card(
            comparison_label,
            mo.md(format_rate_pair(comparison_rate)),
            kind="info",
        )

        calibration_notice = None
        if blend_weight > 0 and isinstance(q10_area, (int, float)) and isinstance(q90_area, (int, float)):
            raw_rate_text = format_rate(raw_price_per_sq_m)
            calibrated_rate_text = format_rate(price_per_sq_m)
            reference_rate_text = format_rate(comparison_rate)
            calibration_notice = mo.callout(
                mo.md(
                    (
                        f"Input area **{live_area:,.0f} sq m** sits outside the training band "
                        f"{q10_area:,.0f}–{q90_area:,.0f} sq m. We blended {blend_weight*100:,.0f}% of the "
                        f"reference rate ({reference_rate_text}) with the raw model rate ({raw_rate_text}) to produce "
                        f"the calibrated rate ({calibrated_rate_text})."
                    )
                ),
                kind="warn",
            )

        prediction_blocks = [
            mo.md(f"### {mode_label} Prediction"),
            mo.hstack([price_callout, raw_price_callout], gap=1, widths=[1, 1], wrap=True),
            mo.hstack([pps_callout, raw_pps_callout], gap=1, widths=[1, 1], wrap=True),
            mo.hstack([range_callout, comparison_callout], gap=1, widths=[1, 1], wrap=True),
        ]
        if calibration_notice is not None:
            prediction_blocks.append(calibration_notice)

        prediction_section = mo.vstack(
            prediction_blocks,
            align="stretch",
            gap=0.6,
        )

    property_content = [
        mo.md("### Property Details"),
        mo.hstack(
            property_cards, gap=1, widths=[1] * len(property_cards), wrap=True
        ),
    ]
    if area_range_notice is not None:
        property_content.append(area_range_notice)
    property_content.append(mode_controls_row)
    if not detailed_mode:
        property_content.append(prediction_section)
    property_content.append(prediction_mode_card)
    property_section = mo.vstack(
        property_content,
        align="stretch",
        gap=0.6,
    )

    detailed_form_section = None
    if detailed_mode:
        detailed_form_section = card(
            "Detailed Predict",
            mo.vstack(
                [
                    mo.md(
                        "Use the full feature dictionary for nuanced scenarios, then run **Run Detailed Predict**."
                    ),
                    prediction_form,
                    prediction_section,
                ],
                align="stretch",
                gap=0.45,
            ),
            kind="danger",
        )

    summary_source = (
        submitted_payload
        if prediction_ready and submitted_payload
        else {name: control_value(name) for name in primary_names}
    )
    summary_lines = [
        f"- **Area**: {summary_source.get('AREA', 0):,} sq m",
        f"- **Building Type**: {summary_source.get('BUILDING TYPE', live_building)}",
        f"- **Location**: {summary_source.get('LOCATION', live_location)}",
        f"- **Storeys**: {live_storey:,.0f}",
    ]
    summary_section = mo.accordion(
        {"Summary": mo.md("\n".join(summary_lines))}, multiple=False
    )

    contract_table = "\n".join(
        f"{feature['name']} | {feature['kind']} | {feature['description']}"
        for feature in feature_contract
    )
    contract_section = mo.accordion(
        {
            "Feature Contract": mo.md(
                """Name | Kind | Description\n--- | --- | ---\n""" + contract_table
            )
        },
        multiple=False,
    )

    sections = [header, property_section]
    if detailed_form_section is not None:
        sections.append(detailed_form_section)
    if advanced_section is not None:
        sections.append(advanced_section)
    sections.append(auto_section)
    sections.append(market_section)
    sections.append(model_section)
    sections.extend([summary_section, contract_section])

    mo.vstack(sections, align="stretch", gap=1.2)
    return


if __name__ == "__main__":
    app.run()
