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
                ("numeric", numeric_pipeline, NUMERIC_FEATURE_NAMES),
                ("categorical", categorical_pipeline, CATEGORICAL_FEATURE_NAMES),
            ]
        )

        model = LGBMRegressor(
            random_state=42,
            n_estimators=500,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
        )

        return Pipeline(
            [
                ("preprocess", preprocessor),
                ("model", model),
            ]
        )

    def train_model(clicks: int, reuse_existing: bool):
        should_load = reuse_existing and MODEL_PATH.exists() and clicks == 0
        if should_load:
            pipeline = load(MODEL_PATH)
            metrics = None
            status = f"Loaded cached model from {MODEL_PATH}"
            return status, metrics, pipeline

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

        pipeline = build_model_pipeline()
        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)
        metrics = {
            "MAE": float(mean_absolute_error(y_test, y_pred)),
            "RMSE": float(np.sqrt(mean_squared_error(y_test, y_pred))),
            "R2": float(r2_score(y_test, y_pred)),
        }

        dump(pipeline, MODEL_PATH)
        status = (
            f"Trained LightGBM on {len(X_train)} rows and saved to {MODEL_PATH.name}."
        )
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
        status, metrics, pipeline = context["train_model"](clicks, reuse_existing)
        return status, metrics, pipeline

    return compute_training, reuse_toggle, train_button, training_controls


@app.cell
def _(compute_training, reuse_toggle, train_button):
    status_msg, metrics_dict, pipeline = compute_training(
        train_button.value, reuse_toggle.value
    )
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
    ctx_monthly_trend = context.get("monthly_price_trend", pd.DataFrame())
    ctx_location_summary = context.get("location_summary", pd.DataFrame())
    ctx_building_summary = context.get("building_type_summary", pd.DataFrame())
    ctx_trend_summary = context.get("trend_summary", {})

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

    property_section = mo.vstack(
        [
            mo.md("### Property Details"),
            mo.hstack(
                property_cards, gap=1, widths=[1] * len(property_cards), wrap=True
            ),
        ],
        align="stretch",
        gap=0.6,
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

    def format_rm(value: float | None) -> str:
        if value is None:
            return "RM —"
        try:
            if pd.isna(value):
                return "RM —"
        except TypeError:
            pass
        return f"RM {float(value):,.0f}"

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

    bundled_form_section = mo.callout(prediction_form, kind="danger")


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

    submitted_payload = form_payload(prediction_form.value)
    prediction = predict_price(submitted_payload)

    def format_currency(value: float) -> str:
        return f"RM {value:,.0f}" if value is not None else "RM 0"

    if pipeline is None:
        prediction_section = mo.callout(
            "Train or load the model to enable predictions.", kind="warn"
        )
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

        price_callout = card(
            "Estimated Price", mo.md(format_currency(estimate)), kind="success"
        )
        pps_callout = card(
            "Price per sq m", mo.md(f"{price_per_sq_m:,.0f}"), kind="neutral"
        )
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
                mo.hstack(
                    [price_callout, pps_callout],
                    gap=1,
                    widths=[1, 1],
                    wrap=True,
                ),
                mo.hstack(
                    [range_callout, comparison_callout],
                    gap=1,
                    widths=[1, 1],
                    wrap=True,
                ),
            ],
            align="stretch",
            gap=0.6,
        )

    summary_source = (
        submitted_payload
        if prediction_form.value
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
    if advanced_section is not None:
        sections.append(advanced_section)
    sections.append(auto_section)
    sections.append(market_section)
    sections.extend(
        [
            bundled_form_section,
            prediction_section,
            summary_section,
            contract_section,
        ]
    )

    mo.vstack(sections, align="stretch", gap=1.2)
    return


if __name__ == "__main__":
    app.run()
