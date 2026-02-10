# app.py
# ============================================================
# UHPC GUI — predicts:
#   1) Mu_rc    = M_{u,rc}
#   2) Mu_comp  = M_{u,comp}
#   3) Failure mode (FULL name)
#
# Requires (same folder):
#   - UHPC_GUI_bundle.joblib
#
# Bundle keys (your exact):
#   FAILURE_MAP, IFACE_LEVELS, LAYOUT_LEVELS, input_features,
#   label_encoder_fail, model_fail, model_mu_comp, model_mu_rc,
#   preprocess_reg, target_fail, target_mu_comp, target_mu_rc
# ============================================================

import os
import numpy as np
import pandas as pd
import joblib
import streamlit as st

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="UHPC Beam Capacity & Failure Mode GUI",
    page_icon="🧱",
    layout="wide",
)

# -----------------------------
# CSS Theme (colorful dark)
# -----------------------------

st.markdown(
    """
<style>
div[data-testid="stWidgetLabel"] label,
div[data-testid="stWidgetLabel"] > label,
label,
.stNumberInput label,
.stSelectbox label,
.stTextInput label {
    font-size: 22px !important;
    font-weight: 900 !important;
    color: #FBBF24 !important;
    letter-spacing: 0.2px !important;
}

div[data-testid="stWidgetLabel"] { margin-bottom: 2px !important; }

div[data-baseweb="input"] input {
    color: #60A5FA !important;
    font-size: 20px !important;
    font-weight: 800 !important;
}

div[data-baseweb="select"] div[role="combobox"] {
    color: #60A5FA !important;
    font-size: 20px !important;
    font-weight: 800 !important;
}

div[data-baseweb="input"] input::placeholder {
    color: #94A3B8 !important;
    font-size: 18px !important;
}

div[role="listbox"] * { font-size: 18px !important; }

div[data-baseweb="input"] > div,
div[data-baseweb="select"] > div {
    min-height: 46px !important;
}

.stCaption {
    font-size: 14px !important;
    color: #A7F3D0 !important;
}
</style>
""",
    unsafe_allow_html=True,
)

# ============================================================
# Load bundle
# ============================================================

@st.cache_resource(show_spinner=True)
def load_bundle():
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, "UHPC_GUI_bundle.joblib")
    return joblib.load(path)

B = load_bundle()

st.write("✅ Bundle loaded")
st.write("Bundle keys:", sorted(B.keys()))
st.write("sklearn version:", __import__("sklearn").__version__)
st.write("xgboost version:", __import__("xgboost").__version__)
try:
    import lightgbm
    st.write("lightgbm version:", lightgbm.__version__)
except Exception as e:
    st.error("LightGBM import failed")
    st.exception(e)

# Required objects (your exact keys)
preprocess_reg = B["preprocess_reg"]           # fitted ColumnTransformer (regression)
model_mu_rc    = B["model_mu_rc"]              # fitted regressor (expects processed X)
model_mu_comp  = B["model_mu_comp"]            # fitted regressor (expects processed X)
model_fail     = B["model_fail"]               # fitted Pipeline (prep + clf)
le_fail        = B["label_encoder_fail"]       # fitted LabelEncoder
input_features = B["input_features"]           # EXACT feature names used in training

target_mu_rc   = B.get("target_mu_rc", r"M$_{u,rc}$")
target_mu_comp = B.get("target_mu_comp", r"M$_{u,comp}$")
target_fail    = B.get("target_fail", "Failure mode")

LAYOUT_LEVELS  = B.get("LAYOUT_LEVELS", ["T-sided","3-sided","2-sided","C-sided","1-sided"])
IFACE_LEVELS   = B.get("IFACE_LEVELS", [
    "Roughned and UHPC casting",
    "Roughned, Epoxy adhesive and UHPC casting",
    "Pre-cast UHPC and Anchorage",
])

FAILURE_MAP_LONG_TO_SHORT = B.get("FAILURE_MAP", {})  # long -> short
FAILURE_MAP_SHORT_TO_LONG = {v: k for k, v in FAILURE_MAP_LONG_TO_SHORT.items()}

# Fallback (your requested full names)
FAILURE_SHORT_TO_LONG_FALLBACK = {
    "FLX": "Flexure",
    "CCR": "Concrete crushing",
    "SHR": "Shear",
    "IDB": "Interface debonding",
    "CSF": "Combined shear and flexure",
}
for k, v in FAILURE_SHORT_TO_LONG_FALLBACK.items():
    FAILURE_MAP_SHORT_TO_LONG.setdefault(k, v)

# ============================================================
# Input display metadata (range + units)
# IMPORTANT: Your trained model uses ONLY the 14 features in input_features.
# You gave ranges for more variables (L, a, bw, d, etc.), but they are NOT in input_features.
# So we display only for variables that are actually part of input_features.
# ============================================================

RANGE_UNITS = {
    r"f$_{c,rc}$":   ("MPa", 20.1, 70.07),
    r"ρ$_{sl,rc}$":  ("%",   0.0,  8.31),
    r"f$_{y,rc}$":   ("MPa", 0.0,  600.0),
    r"f$_{yv,rc}$":  ("MPa", 0.0,  610.0),
    r"ρ$_{v,rc}$":   ("%",   0.0,  1.43),

    r"t$_{uhpc}$":   ("mm",  5.0,  70.0),
    r"E$_{uhpc}$":   ("GPa", 34.6, 145.0),
    r"f$_{t,uhpc}$": ("MPa", 5.0,  16.0),
    r"f$_{c,uhpc}$": ("MPa", 102.2, 204.0),
    r"ρ$_{uhpc}$":   ("%",   0.0,  3.35),
    r"v$_{f}$":      ("%",   0.5,  3.0),
    r"λ$_{s}$":      ("-",   26.79,125.0),
}

PRETTY_LABELS = {
    r"f$_{c,rc}$":   "Concrete compressive strength (RC)",
    r"ρ$_{sl,rc}$":  "Longitudinal reinforcement ratio (RC)",
    r"f$_{y,rc}$":   "Longitudinal steel yield strength (RC)",
    r"f$_{yv,rc}$":  "Shear reinforcement yield strength (RC)",
    r"ρ$_{v,rc}$":   "Shear reinforcement ratio (RC)",
    "layout":        "UHPC layout",
    r"t$_{uhpc}$":   "UHPC layer thickness",
    r"E$_{uhpc}$":   "UHPC elastic modulus",
    r"f$_{t,uhpc}$": "UHPC tensile strength",
    r"f$_{c,uhpc}$": "UHPC compressive strength",
    r"ρ$_{uhpc}$":   "UHPC reinforcement ratio",
    r"v$_{f}$":      "Fiber volume fraction",
    r"λ$_{s}$":      "Steel fiber aspect ratio",
    "iface":         "RC–UHPC interface preparation",
}

def mid(vmin, vmax):
    return float(vmin + 0.5*(vmax - vmin))

# ============================================================
# Title
# ============================================================
st.title("🧱 UHPC-Strengthened RC Beam: Capacity & Failure Mode Predictor")
st.caption("Predict $M_{u,rc}$, $M_{u,comp}$, and Failure Mode using your saved ML bundle.")

# ============================================================
# Inputs (3 columns)
# ============================================================
st.subheader("Inputs")

user = {}
cols = st.columns(3, gap="large")

for i, feat in enumerate(input_features):
    col = cols[i % 3]
    with col:
        label = PRETTY_LABELS.get(feat, feat)

        if feat == "layout":
            default_idx = 0 if "T-sided" not in LAYOUT_LEVELS else LAYOUT_LEVELS.index("T-sided")
            user[feat] = st.selectbox(label, options=LAYOUT_LEVELS, index=default_idx)
            st.caption("Options: " + ", ".join(LAYOUT_LEVELS))

        elif feat == "iface":
            user[feat] = st.selectbox(label, options=IFACE_LEVELS, index=0)
            st.caption("Options: " + "; ".join(IFACE_LEVELS))

        else:
            unit, vmin, vmax = RANGE_UNITS.get(feat, ("", 0.0, 1.0))
            v0 = mid(vmin, vmax)
            step = float((vmax - vmin) / 200.0) if vmax > vmin else 0.1

            user[feat] = st.number_input(
                f"{label} ({unit})" if unit else label,
                min_value=float(vmin),
                max_value=float(vmax),
                value=float(v0),
                step=step,
                format="%.4f" if (abs(vmax) < 100 and unit not in ["mm"]) else "%.2f"
            )
            if feat in RANGE_UNITS:
                st.caption(f"Range: {vmin} – {vmax} {unit}")

st.markdown("---")

with st.expander("Show feature ranges & units (for inputs used by the models)"):
    rows = []
    for feat in input_features:
        label = PRETTY_LABELS.get(feat, feat)
        if feat in RANGE_UNITS:
            unit, vmin, vmax = RANGE_UNITS[feat]
            dr = f"{vmin} – {vmax}"
        else:
            unit, dr = "-", "Categorical"
        rows.append([label, feat, unit, dr])

    st.dataframe(pd.DataFrame(rows, columns=["Feature", "Symbol", "Unit", "Data Range"]),
                 use_container_width=True)

# ============================================================
# Predict
# ============================================================
left, right = st.columns([1, 2], gap="large")
with left:
    run = st.button("🚀 Predict", use_container_width=True)

with right:
    st.info(
        "For **$M_{u,rc}$** and **$M_{u,comp}$**, the app applies your saved `preprocess_reg` and then predicts. "
        "For **Failure mode**, it uses your saved pipeline `model_fail` (which includes preprocessing)."
    )

def decode_failure(pred_numeric_or_array):
    # returns FULL name
    if isinstance(pred_numeric_or_array, (list, np.ndarray)):
        pred0 = pred_numeric_or_array[0]
    else:
        pred0 = pred_numeric_or_array

    # XGB classifier returns numeric class IDs here (because you label-encoded)
    try:
        short = le_fail.inverse_transform([int(pred0)])[0]
    except Exception:
        short = str(pred0)

    return FAILURE_MAP_SHORT_TO_LONG.get(short, short)

if run:
    # Build 1-row raw dataframe with EXACT column names
    X_in = pd.DataFrame([{k: user[k] for k in input_features}], columns=input_features)

    # --- Regression preprocessing + predictions ---
    try:
        X_reg = preprocess_reg.transform(X_in)
        mu_rc_pred   = float(model_mu_rc.predict(X_reg).ravel()[0])
        mu_comp_pred = float(model_mu_comp.predict(X_reg).ravel()[0])
    except Exception as e:
        st.error("Regression prediction failed (preprocess_reg or regressors mismatch).")
        st.exception(e)
        st.stop()

    # --- Classification prediction (pipeline expects RAW) ---
    try:
        y_fail_num = model_fail.predict(X_in)
        fail_long = decode_failure(y_fail_num)
    except Exception as e:
        st.error("Failure mode prediction failed (model_fail pipeline mismatch).")
        st.exception(e)
        st.stop()

    # ========================================================
    # Outputs
    # ========================================================
    st.subheader("Outputs")

    o1, o2, o3 = st.columns(3, gap="large")

    with o1:
        st.markdown(r"### $M_{u,rc}$")
        st.metric(label=" ", value=f"{mu_rc_pred:.2f}")

    with o2:
        st.markdown(r"### $M_{u,comp}$")
        st.metric(label=" ", value=f"{mu_comp_pred:.2f}")

    with o3:
        st.markdown("### Failure mode")
        st.metric(label=" ", value=fail_long)

    st.success("Done ✅ Change inputs and click **Predict** again.")

# ============================================================
# Debug
# ============================================================
with st.expander("Debug: bundle keys"):
    st.write(sorted(list(B.keys())))


