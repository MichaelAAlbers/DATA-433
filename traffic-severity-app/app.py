import os
import tempfile
from pathlib import Path

import joblib
import pandas as pd
import requests
import streamlit as st

# -----------------------
# LOAD FILES
# -----------------------
APP_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = APP_DIR / "crash_model.pkl"
ENCODERS_PATH = APP_DIR / "encoders.pkl"
FEATURE_COLUMNS_PATH = APP_DIR / "feature_columns.pkl"


def get_setting(name: str):
    value = os.getenv(name)
    if value:
        return value

    try:
        return st.secrets.get(name)
    except Exception:
        return None


def download_model(url: str) -> Path:
    cache_dir = Path(tempfile.gettempdir()) / "traffic-severity-app"
    cache_dir.mkdir(parents=True, exist_ok=True)

    filename = Path(url.split("?")[0]).name or "crash_model.pkl"
    model_path = cache_dir / filename

    if model_path.exists():
        return model_path

    with st.spinner("Downloading model artifact..."):
        response = requests.get(url, stream=True, timeout=120)
        response.raise_for_status()

        with model_path.open("wb") as file_handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    file_handle.write(chunk)

    return model_path


def resolve_model_path() -> Path:
    configured_path = get_setting("CRASH_MODEL_PATH")
    configured_url = get_setting("CRASH_MODEL_URL")

    if configured_path:
        model_path = Path(configured_path).expanduser()
        if model_path.exists():
            return model_path
        raise FileNotFoundError(
            f"CRASH_MODEL_PATH points to a missing file: {model_path}"
        )

    if DEFAULT_MODEL_PATH.exists():
        return DEFAULT_MODEL_PATH

    if configured_url:
        return download_model(configured_url)

    raise FileNotFoundError(
        "Model file not found. Add crash_model.pkl next to app.py, "
        "or set CRASH_MODEL_PATH / CRASH_MODEL_URL before starting Streamlit."
    )


@st.cache_resource(show_spinner="Loading model artifacts...")
def load_artifacts():
    model_path = resolve_model_path()
    model = joblib.load(model_path)
    encoders = joblib.load(ENCODERS_PATH)
    feature_columns = joblib.load(FEATURE_COLUMNS_PATH)
    return model, encoders, feature_columns, model_path


try:
    model, encoders, feature_columns, model_path = load_artifacts()
except Exception as exc:
    st.error(f"Unable to load model artifacts: {exc}")
    st.info(
        "For local runs, set CRASH_MODEL_PATH to a model file on your machine. "
        "For hosted deployments, store the model outside GitHub and set "
        "CRASH_MODEL_URL to a direct download link."
    )
    st.stop()

st.title("Crash Severity Predictor")
st.caption(f"Model source: {model_path}")

# -----------------------
# HELPER FUNCTION
# -----------------------
def encode(col, val):
    if col in encoders:
        le = encoders[col]
        if val in le.classes_:
            return le.transform([val])[0]
        return -1
    return val


# -----------------------
# USER INPUTS (keep this small)
# -----------------------
weather = st.selectbox(
    "Weather",
    ["CLEAR", "RAIN", "SNOW", "UNKNOWN"],
)

lighting = st.selectbox(
    "Lighting",
    ["DAYLIGHT", "DARKNESS", "DUSK", "DAWN"],
)

crash_type = st.selectbox(
    "Crash Type",
    ["REAR END", "TURNING", "ANGLE", "SIDESWIPE SAME DIRECTION"],
)

speed = st.slider("Speed Limit", 0, 70, 30)
hour = st.slider("Crash Hour", 0, 23, 12)
month = st.slider("Month", 1, 12, 6)

# -----------------------
# BUILD INPUT ROW
# -----------------------

# Initialize everything as -1 so missing features still align.
data = {col: -1 for col in feature_columns}

# Fill numeric
data["POSTED_SPEED_LIMIT"] = speed
data["CRASH_HOUR"] = hour
data["CRASH_MONTH"] = month

# Fill categorical
data["WEATHER_CONDITION"] = encode("WEATHER_CONDITION", weather)
data["LIGHTING_CONDITION"] = encode("LIGHTING_CONDITION", lighting)
data["FIRST_CRASH_TYPE"] = encode("FIRST_CRASH_TYPE", crash_type)

# Convert to dataframe in correct order
input_df = pd.DataFrame([data])[feature_columns]

# -----------------------
# PREDICTION
# -----------------------
if st.button("Predict"):
    prediction = model.predict(input_df)[0]

    if prediction == 0:
        st.success("Prediction: No Injury")
    else:
        st.error("Prediction: Injury Likely")
