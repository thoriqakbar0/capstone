import marimo

__generated_with = "0.16.2"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    from marimo import create_asgi_app

    return (create_asgi_app,)


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

        lease_period = frame.get("LEASE PERIOD", pd.Series(dtype=str))
        frame["LEASE_PERIOD_YEARS"] = (
            lease_period.fillna("")
            .str.extract(r"(\d+)")
            .astype(float)
            .squeeze()
        )

        frame["OWNERSHIP_SHARE"] = frame.get("SHARE", "").apply(parse_share)
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
                values = df[name].dropna().unique().tolist()
                options[name] = sorted(map(str, values))
        return options

    feature_defaults_map = _feature_defaults(engineered_df)
    feature_options_map = _feature_options(engineered_df)

    numeric_features = [feature["name"] for feature in FEATURE_CONTRACT if feature["kind"] == "numeric"]
    categorical_features = [feature["name"] for feature in FEATURE_CONTRACT if feature["kind"] == "categorical"]
    feature_order = numeric_features + categorical_features

    location_activity = engineered_df.groupby("LOCATION").size().to_dict()
    storey_suggestions = (
        engineered_df.groupby("BUILDING TYPE")["STOREY/LEVEL"].median().dropna().to_dict()
    )

    with np.errstate(divide="ignore", invalid="ignore"):
        price_per_sqft_series = engineered_df["PRICE"].astype(float) / engineered_df["AREA"].astype(float)
        area_ratio_series = engineered_df["MFA"].astype(float) / engineered_df["AREA"].astype(float)

    price_per_sqft_median = float(np.nanmedian(price_per_sqft_series)) if price_per_sqft_series.notna().any() else 0.0
    area_to_mfa_ratio = float(np.nanmedian(area_ratio_series)) if area_ratio_series.notna().any() else 1.0

    parsed_dates = pd.to_datetime(engineered_df["Date"], format="%d-%b-%y", errors="coerce").dropna()
    date_window = {
        "min": str(parsed_dates.min().date()) if not parsed_dates.empty else "",
        "max": str(parsed_dates.max().date()) if not parsed_dates.empty else "",
    }

    def build_model_pipeline() -> Pipeline:
        numeric_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )

        categorical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                (
                    "encoder",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                ),
            ]
        )

        preprocessor = ColumnTransformer(
            transformers=[
                ("numeric", numeric_pipeline, numeric_features),
                ("categorical", categorical_pipeline, categorical_features),
            ]
        )

        model = LGBMRegressor(
            random_state=42,
            n_estimators=500,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
        )

        return Pipeline([
            ("preprocess", preprocessor),
            ("model", model),
        ])

    def train_model(clicks: int, reuse_existing: bool):
        should_load = reuse_existing and MODEL_PATH.exists() and clicks == 0
        if should_load:
            pipeline = load(MODEL_PATH)
            metrics = None
            status = f"Loaded cached model from {MODEL_PATH}"
            return status, metrics, pipeline

        prepared = engineered_df.dropna(subset=["PRICE"])
        missing_columns = [name for name in feature_order if name not in prepared.columns]
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

        pipeline = build_model_pipeline()
        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)
        metrics = {
            "MAE": float(mean_absolute_error(y_test, y_pred)),
            "RMSE": float(np.sqrt(mean_squared_error(y_test, y_pred))),
            "R2": float(r2_score(y_test, y_pred)),
        }

        dump(pipeline, MODEL_PATH)
        status = f"Trained LightGBM on {len(X_train)} rows and saved to {MODEL_PATH.name}."
        return status, metrics, pipeline

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
        "date_window": date_window,
    }
    return context, pd


@app.cell
def _(context, mo):
    train_button = mo.ui.run_button(label="Train / Refresh LightGBM")
    reuse_toggle = mo.ui.switch(label="Reuse saved model if available", value=True)

    training_controls = mo.hstack([train_button, reuse_toggle], align="center", gap=1.0)

    @mo.cache
    def compute_training(clicks: int, reuse_existing: bool):
        status, metrics, pipeline = context["train_model"](clicks, reuse_existing)
        return status, metrics, pipeline
    return compute_training, reuse_toggle, train_button, training_controls


@app.cell
def _(compute_training, reuse_toggle, train_button):
    status_msg, metrics_dict, pipeline = compute_training(train_button.value, reuse_toggle.value)
    return metrics_dict, pipeline, status_msg


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
            )
        else:
            options = options_map.get(name, [])
            initial_choice = defaults_map.get(name, options[0] if options else "")
            element = mo.ui.dropdown(
                options=options or [initial_choice],
                value=initial_choice,
                label=name,
            )

        controls[field_id] = element
        name_to_field[name] = field_id

    bundled_controls = mo.ui.dictionary(controls)
    prediction_form = mo.ui.form(
        bundled_controls,
        submit_button_label="Predict Property Price",
        bordered=False,
        show_clear_button=True,
        clear_button_label="Reset",
    )
    return (
        controls,
        feature_contract,
        feature_order_list,
        name_to_field,
        prediction_form,
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
    status_msg,
    training_controls,
):
    ctx_defaults = context["feature_defaults"]
    ctx_location_activity = context["location_activity"]
    ctx_storey_suggestions = context["storey_suggestions"]
    ctx_price_per_sqft_median = context["price_per_sqft_median"]
    ctx_area_to_mfa_ratio = context["area_to_mfa_ratio"]
    ctx_date_window = context["date_window"]

    primary_names = ["AREA", "BUILDING TYPE", "LOCATION"]
    advanced_names = [feature["name"] for feature in feature_contract if feature["name"] not in primary_names]

    def control_value(name: str):
        field_id = name_to_field[name]
        element = controls[field_id]
        value = element.value
        return value if value not in (None, "") else ctx_defaults.get(name)

    live_area = float(control_value("AREA") or 0.0)
    live_location = control_value("LOCATION") or ""
    live_building = control_value("BUILDING TYPE") or ""

    storey_value = control_value("STOREY/LEVEL")
    if storey_value in (None, ""):
        storey_value = ctx_storey_suggestions.get(live_building, ctx_defaults.get("STOREY/LEVEL", 1))
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
            f"""**{status_msg}**\n\n{metrics_lines}\n\n_Data window: {ctx_date_window['min']} → {ctx_date_window['max']}_"""
        ),
        kind="success",
    )
    
    header = mo.vstack(
        [
            mo.hstack([
                mo.icon("lucide:home", size=26),
                mo.md("## Malaysian Property Price Predictor"),
            ], gap=0.5, align="center"),
            mo.md("_Simplified version — just 3 inputs: Area + Building Type + Location_"),
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

    property_section = mo.vstack(
        [mo.md("### Property Details"), mo.hstack(property_cards, gap=1, widths="equal")],
        align="stretch",
        gap=0.6,
    )

    advanced_elements = [controls[name_to_field[name]] for name in advanced_names]
    advanced_section = (
        mo.accordion({"Advanced Inputs": mo.vstack(advanced_elements, gap=0.45)}, multiple=False)
        if advanced_elements
        else None
    )

    auto_cards = mo.hstack(
        [
            card("Floor Area (est.)", mo.md(f"{floor_area_estimate:,.0f} sq m"), kind="info"),
            card("Storeys", mo.md(f"{live_storey:,.0f}"), kind="info"),
            card("Location Activity", mo.md(f"{location_activity_count} transactions"), kind="info"),
        ],
        gap=1,
        widths="equal",
    )
    auto_section = mo.vstack(
        [mo.md("### Auto-Calculated Values"), auto_cards], align="stretch", gap=0.6
    )

    bundled_form_section = mo.callout(prediction_form, kind="danger")

    def form_payload(form_value: dict | None) -> dict:
        payload: dict[str, float | str | None] = {}
        if not form_value:
            for feature in feature_contract:
                payload[feature["name"]] = ctx_defaults.get(feature["name"])
            return payload
        for feature in feature_contract:
            field_id = feature["field_id"]
            payload[feature["name"]] = form_value.get(field_id, ctx_defaults.get(feature["name"]))
        return payload

    def predict_price(payload: dict):
        if pipeline is None or not payload:
            return None
        ordered = {name: payload.get(name, ctx_defaults.get(name)) for name in feature_order_list}
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

    submitted_payload = form_payload(prediction_form.value)
    prediction = predict_price(submitted_payload)

    def format_currency(value: float) -> str:
        return f"RM {value:,.0f}" if value is not None else "RM 0"

    if pipeline is None:
        prediction_section = mo.callout("Train or load the model to enable predictions.", kind="warn")
    elif prediction is None or not prediction_form.value:
        prediction_section = mo.callout(
            "Fill in the form above and click **Predict Property Price** to generate an estimate.",
            kind="neutral",
        )
    else:
        estimate = prediction["prediction"]
        low = prediction["low"]
        high = prediction["high"]
        price_per_sq_m = estimate / max(live_area, 1.0)

        price_callout = card("Estimated Price", mo.md(format_currency(estimate)), kind="success")
        pps_callout = card("Price per sq m", mo.md(f"{price_per_sq_m:,.0f}"), kind="neutral")
        range_callout = card(
            "Estimated Range",
            mo.md(f"{format_currency(low)} – {format_currency(high)}"),
            kind="success",
        )

        comparison_callout = card(
            "Dataset Median Price/m²",
            mo.md(f"{ctx_price_per_sqft_median:,.0f}"),
            kind="info",
        )

        prediction_section = mo.vstack(
            [
                mo.md("### Price Prediction"),
                mo.hstack([price_callout, pps_callout], gap=1, widths="equal"),
                mo.hstack([range_callout, comparison_callout], gap=1, widths="equal"),
            ],
            align="stretch",
            gap=0.6,
        )

    summary_source = submitted_payload if prediction_form.value else {
        name: control_value(name) for name in primary_names
    }
    summary_lines = [
        f"- **Area**: {summary_source.get('AREA', 0):,} sq m",
        f"- **Building Type**: {summary_source.get('BUILDING TYPE', live_building)}",
        f"- **Location**: {summary_source.get('LOCATION', live_location)}",
        f"- **Storeys**: {live_storey:,.0f}",
    ]
    summary_section = mo.accordion({"Summary": mo.md("\n".join(summary_lines))}, multiple=False)

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
    if advanced_section is not None:
        sections.append(advanced_section)
    sections.extend([auto_section, bundled_form_section, prediction_section, summary_section, contract_section])

    mo.vstack(sections, align="stretch", gap=1.2)
    return


@app.cell
def _(Path, create_asgi_app):
    try:
        from fastapi import FastAPI
    except ImportError:  # pragma: no cover - fastapi optional
        FastAPI = None

    def build_fastapi_app(
        base_path: str = "/",
        *,
        include_code: bool = True,
        quiet: bool = False,
    ):
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
        def health_check():  # pragma: no cover - simple endpoint
            return {"status": "ok"}

        fastapi_app.mount(normalized, server.build())
        return fastapi_app

    return (build_fastapi_app,)


@app.cell
def _(build_fastapi_app):
    try:
        fastapi_app = build_fastapi_app()
    except ImportError:  # FastAPI not available; expose None for clarity
        fastapi_app = None

    return (fastapi_app,)


if __name__ == "__main__":
    app.run()
