"""
Stellar Classifier — production inference console.

Loads models/production_bundle.pkl (built by Notebook 05) and serves
predictions through a Streamlit UI styled as a quiet instrument console:
one muted accent colour, no gradients, monospace for every number.

──────────────────────────────────────────────────────────────────────────
EXTENDING THIS APP
──────────────────────────────────────────────────────────────────────────
1. New engineered feature       -> add its name to the right list in
                                    FEATURE_GROUPS. Anything in the bundle's
                                    feature_names that isn't grouped yet
                                    automatically falls into "Other Features",
                                    so nothing silently disappears.
2. New model family (e.g. XGB)  -> add a loader branch in load_bundle(),
                                    a weight key in bundle['blend_weights'],
                                    and one block in run_inference() that
                                    mirrors the lgb/cat blocks already there.
3. New explainability panel     -> add a function near render_shap_panel()
                                    and call it from render_result() the
                                    same way the existing panels are called.
──────────────────────────────────────────────────────────────────────────
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from catboost import Pool

APP_DIR = Path(__file__).parent
BUNDLE_PATH = APP_DIR / "models" / "production_bundle.pkl"

# The bundle file is too large to keep in the app repo, so it lives on the
# Hugging Face Hub instead and gets downloaded into BUNDLE_PATH the first
# time the app starts (then reused from disk on every rerun after that).
HF_REPO_ID = "sathyanjali00/predicting-stellar-class-model"
HF_FILENAME = "production_bundle.pkl"

# ── Feature layout (purely for UI organisation; source of truth for WHICH
#    features exist is always bundle['feature_names']) ─────────────────────
FEATURE_GROUPS = {
    "Astrometry": ["alpha", "delta", "delta_abs"],
    "Photometry (ugriz)": ["u", "g", "r", "i", "z"],
    "Colour Indices": [
        "u_g", "g_r", "r_i", "i_z", "g_i", "r_z", "u_r", "g_z", "u_i", "u_z",
        "r_g", "g_div_z", "r_div_i", "u_div_r", "g_div_r", "gz_x_ri", "gr_x_iz",
    ],
    "Band Statistics": [
        "band_mean", "band_std", "band_min", "band_max", "band_range",
        "band_sum", "band_skew", "band_median", "band_iqr", "band_cv",
    ],
    "Redshift": [
        "redshift", "redshift_log1p", "redshift_sq", "redshift_sqrt",
        "redshift_abs", "is_high_z", "is_star_z", "is_redshift_zero",
        "is_very_high_z", "is_negative_z", "redshift_x_gz",
        "redshift_x_bandstd", "redshift_x_bandmean", "redshift_over_bandmean",
        "g_r_x_redshift", "u_g_x_redshift",
    ],
    "Encodings": ["spectral_type_enc", "galaxy_population_enc"],
}
BINARY_FEATURES = {
    "is_high_z", "is_star_z", "is_redshift_zero", "is_very_high_z", "is_negative_z",
}

ACCENT = "#4A93A0"
MUTED = "#5B6474"
SURFACE = "#131826"
BORDER = "#232837"
TEXT = "#E8E9ED"
TEXT_DIM = "#8B92A5"
NEGATIVE = "#8C6B5A"


# ── Data / model loading ────────────────────────────────────────────────
def download_bundle_if_missing():
    """Fetch the bundle from the Hugging Face Hub the first time it's needed.

    On later runs BUNDLE_PATH already exists, so this becomes a no-op and
    the app starts instantly.
    """
    if BUNDLE_PATH.exists():
        return

    from huggingface_hub import hf_hub_download

    BUNDLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    downloaded_path = hf_hub_download(repo_id=HF_REPO_ID, filename=HF_FILENAME)
    Path(downloaded_path).rename(BUNDLE_PATH)


@st.cache_resource(show_spinner="Loading production bundle…")
def load_bundle():
    import sys
    sys.path.insert(0, str(APP_DIR))
    import joblib
    from src.models.predict import wrap_lgb_model

    download_bundle_if_missing()
    if not BUNDLE_PATH.exists():
        return None
    bundle = joblib.load(BUNDLE_PATH)
    bundle["_wrap_lgb_model"] = wrap_lgb_model
    return bundle


def run_inference(bundle: dict, row: pd.DataFrame):
    """Blend LightGBM + CatBoost fold predictions. Returns (proba, per_model)."""
    wrap_lgb_model = bundle["_wrap_lgb_model"]
    w = bundle["blend_weights"]
    n_classes = len(bundle["class_map"])

    lgb_fold_preds = []
    for m in bundle["lightgbm_models"]:
        model = wrap_lgb_model(m) if hasattr(m, "predict") and not hasattr(m, "predict_proba") else m
        lgb_fold_preds.append(model.predict_proba(row)[0])
    lgb_avg = np.mean(lgb_fold_preds, axis=0) if lgb_fold_preds else np.zeros(n_classes)

    cat_fold_preds = [m.predict_proba(row)[0] for m in bundle["catboost_models"]]
    cat_avg = np.mean(cat_fold_preds, axis=0) if cat_fold_preds else np.zeros(n_classes)

    blended = lgb_avg * w.get("lgb", 0) + cat_avg * w.get("cat", 0)
    if blended.sum() > 0:
        blended = blended / blended.sum()

    per_model = {
        "lgb_folds": lgb_fold_preds,
        "cat_folds": cat_fold_preds,
        "lgb_avg": lgb_avg,
        "cat_avg": cat_avg,
    }
    return blended, per_model


@st.cache_data(show_spinner=False)
def global_shap_importance(_bundle_id: str, _bundle: dict, top_n: int = 12):
    """Global |SHAP| ranking from the stored background sample (cheap, cached).

    Both _bundle_id and _bundle start with an underscore on purpose: st.cache_data
    tries to hash every argument to know when to reuse a cached result, and a
    dict full of model objects can't be hashed. The underscore tells Streamlit
    "don't hash this one, trust me" -- _bundle_id (a plain string, the bundle's
    file path) is what actually keys the cache instead.
    """
    cat_model = _bundle["catboost_models"][0]
    bg = _bundle["shap_background_sample"]
    raw = np.asarray(cat_model.get_feature_importance(type="ShapValues", data=Pool(bg)))
    values = raw[:, :, :-1]  # drop CatBoost's trailing bias term
    mean_abs = np.abs(values).mean(axis=(0, 1))
    order = np.argsort(mean_abs)[::-1][:top_n]
    names = np.array(_bundle["feature_names"])[order]
    return list(zip(names, mean_abs[order]))


def local_shap_contribution(bundle: dict, row: pd.DataFrame, class_index: int, top_n: int = 8):
    cat_model = bundle["catboost_models"][0]
    raw = np.asarray(cat_model.get_feature_importance(type="ShapValues", data=Pool(row)))
    values = raw[0, class_index, :-1]
    order = np.argsort(np.abs(values))[::-1][:top_n]
    names = np.array(bundle["feature_names"])[order]
    return list(zip(names, values[order]))


# ── Styling ──────────────────────────────────────────────────────────────
def inject_css():
    st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600&family=Inter:wght@400;500&family=IBM+Plex+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] {{
        font-family: 'Inter', sans-serif;
    }}
    h1, h2, h3 {{
        font-family: 'Space Grotesk', sans-serif !important;
        letter-spacing: -0.01em;
    }}
    .mono {{
        font-family: 'IBM Plex Mono', monospace;
    }}
    .app-header {{
        border-bottom: 1px solid {BORDER};
        padding-bottom: 14px;
        margin-bottom: 6px;
    }}
    .eyebrow {{
        font-family: 'IBM Plex Mono', monospace;
        font-size: 12px;
        letter-spacing: 0.12em;
        color: {ACCENT};
        text-transform: uppercase;
    }}
    .panel {{
        background: {SURFACE};
        border: 1px solid {BORDER};
        border-radius: 4px;
        padding: 20px 22px;
        margin-bottom: 14px;
    }}
    .result-class {{
        font-family: 'Space Grotesk', sans-serif;
        font-size: 40px;
        font-weight: 600;
        color: {TEXT};
        margin: 0;
    }}
    .result-conf {{
        font-family: 'IBM Plex Mono', monospace;
        font-size: 14px;
        color: {ACCENT};
        margin-top: 2px;
    }}
    .meter-row {{
        display: flex;
        align-items: center;
        gap: 12px;
        margin: 10px 0;
    }}
    .meter-label {{
        font-family: 'IBM Plex Mono', monospace;
        font-size: 13px;
        color: {TEXT_DIM};
        width: 64px;
        flex-shrink: 0;
    }}
    .meter-track {{
        position: relative;
        flex-grow: 1;
        height: 10px;
        background: #1B2130;
        border: 1px solid {BORDER};
        border-radius: 2px;
        overflow: hidden;
    }}
    .meter-fill {{
        position: absolute;
        top: 0; left: 0; bottom: 0;
        background: {ACCENT};
    }}
    .meter-fill.dim {{ background: {MUTED}; }}
    .meter-tick {{
        position: absolute;
        top: 0; bottom: 0;
        width: 1px;
        background: rgba(0,0,0,0.35);
    }}
    .meter-value {{
        font-family: 'IBM Plex Mono', monospace;
        font-size: 13px;
        color: {TEXT};
        width: 52px;
        text-align: right;
        flex-shrink: 0;
    }}
    .shap-bar-row {{
        display: flex;
        align-items: center;
        gap: 10px;
        margin: 6px 0;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 12px;
    }}
    .shap-name {{
        width: 190px;
        color: {TEXT_DIM};
        flex-shrink: 0;
        text-align: right;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
    }}
    .shap-track {{
        flex-grow: 1;
        height: 8px;
        display: flex;
        align-items: center;
        position: relative;
    }}
    .shap-fill {{
        height: 8px;
    }}
    .footnote {{
        color: {TEXT_DIM};
        font-size: 12px;
        margin-top: 4px;
    }}
    </style>
    """, unsafe_allow_html=True)


def meter(label: str, value: float, is_top: bool):
    pct = max(0.0, min(1.0, value)) * 100
    fill_class = "" if is_top else "dim"
    ticks = "".join(f'<div class="meter-tick" style="left:{p}%"></div>' for p in (25, 50, 75))
    st.markdown(f"""
    <div class="meter-row">
        <div class="meter-label">{label}</div>
        <div class="meter-track">
            <div class="meter-fill {fill_class}" style="width:{pct:.1f}%"></div>
            {ticks}
        </div>
        <div class="meter-value">{value*100:5.1f}%</div>
    </div>
    """, unsafe_allow_html=True)


def shap_bar(name: str, value: float, max_abs: float):
    width_pct = (abs(value) / max_abs * 100) if max_abs > 0 else 0
    color = ACCENT if value >= 0 else NEGATIVE
    # bar grows from centre so sign is visually legible without extra colour noise
    st.markdown(f"""
    <div class="shap-bar-row">
        <div class="shap-name">{name}</div>
        <div class="shap-track">
            <div class="shap-fill" style="background:{color}; width:{width_pct:.1f}%;"></div>
        </div>
        <div class="mono" style="width:58px; color:{TEXT_DIM};">{value:+.3f}</div>
    </div>
    """, unsafe_allow_html=True)


# ── Sidebar: input console ──────────────────────────────────────────────
def render_sidebar(bundle: dict) -> pd.DataFrame:
    st.sidebar.markdown('<div class="eyebrow">Input Console</div>', unsafe_allow_html=True)
    st.sidebar.markdown("### Observation")

    feature_names = bundle["feature_names"]
    grouped = {name for names in FEATURE_GROUPS.values() for name in names}
    ungrouped = [f for f in feature_names if f not in grouped]
    groups = dict(FEATURE_GROUPS)
    if ungrouped:
        groups["Other Features"] = ungrouped

    def load_sample():
        sample = bundle["shap_background_sample"].sample(n=1).iloc[0]
        for f in feature_names:
            key = f"feat_{f}"
            if f in BINARY_FEATURES:
                st.session_state[key] = bool(round(float(sample[f])))
            else:
                st.session_state[key] = float(sample[f])

    st.sidebar.button("⟳ Load random sample", on_click=load_sample, width='stretch')
    st.sidebar.markdown("<div class='footnote'>Pulls one real row from the training background sample.</div>", unsafe_allow_html=True)
    st.sidebar.markdown("---")

    values = {}
    first = True
    for group_name, names in groups.items():
        present = [n for n in names if n in feature_names]
        if not present:
            continue
        with st.sidebar.expander(group_name, expanded=first):
            for f in present:
                key = f"feat_{f}"
                # Seed session_state once, then let the widget read purely from
                # its key -- passing `value=` on every rerun *and* pre-setting
                # session_state for the same key is what Streamlit warns about,
                # and for checkboxes it hard-fails on a float default.
                if f in BINARY_FEATURES:
                    st.session_state.setdefault(key, False)
                    values[f] = int(st.checkbox(f, key=key))
                else:
                    st.session_state.setdefault(key, 0.0)
                    values[f] = st.number_input(f, format="%.4f", key=key)
        first = False

    row = pd.DataFrame([[values[f] for f in feature_names]], columns=feature_names)
    return row


# ── Main panels ──────────────────────────────────────────────────────────
def render_header(bundle):
    st.markdown('<div class="app-header">', unsafe_allow_html=True)
    st.markdown('<div class="eyebrow">Production Inference</div>', unsafe_allow_html=True)
    st.markdown("## Stellar Object Classifier")
    n_lgb = len(bundle["lightgbm_models"])
    n_cat = len(bundle["catboost_models"])
    w = bundle["blend_weights"]
    st.markdown(
        f'<span class="mono footnote">{n_lgb}-fold LightGBM · {n_cat}-fold CatBoost · '
        f'blend {w.get("lgb",0):.2f} / {w.get("cat",0):.2f}</span>',
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)


def render_result(bundle, row, proba, per_model):
    class_map = bundle["class_map"]
    top_idx = int(np.argmax(proba))
    top_label = class_map[top_idx]

    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="eyebrow">Prediction</div>', unsafe_allow_html=True)
    st.markdown(f'<p class="result-class">{top_label}</p>', unsafe_allow_html=True)
    st.markdown(f'<p class="result-conf">confidence {proba[top_idx]*100:.1f}%</p>', unsafe_allow_html=True)
    st.markdown('<div style="height:14px;"></div>', unsafe_allow_html=True)
    for idx in sorted(range(len(proba)), key=lambda i: -proba[i]):
        meter(class_map[idx], proba[idx], idx == top_idx)
    st.markdown("</div>", unsafe_allow_html=True)

    with st.expander("Model Breakdown"):
        st.markdown('<div class="eyebrow">Fold Agreement</div>', unsafe_allow_html=True)
        lgb_votes = [int(np.argmax(p)) for p in per_model["lgb_folds"]]
        cat_votes = [int(np.argmax(p)) for p in per_model["cat_folds"]]
        lgb_agree = sum(v == top_idx for v in lgb_votes)
        cat_agree = sum(v == top_idx for v in cat_votes)
        col1, col2 = st.columns(2)
        col1.markdown(
            f'<span class="mono">LightGBM: {lgb_agree}/{len(lgb_votes)} folds agree</span>',
            unsafe_allow_html=True,
        )
        col2.markdown(
            f'<span class="mono">CatBoost: {cat_agree}/{len(cat_votes)} folds agree</span>',
            unsafe_allow_html=True,
        )
        st.markdown('<div style="height:10px;"></div>', unsafe_allow_html=True)
        st.markdown('<div class="eyebrow">Averaged Probabilities by Family</div>', unsafe_allow_html=True)
        for idx in range(len(proba)):
            st.markdown(f'<span class="mono footnote">{class_map[idx]}</span>', unsafe_allow_html=True)
            meter("LGB", per_model["lgb_avg"][idx], idx == top_idx)
            meter("Cat", per_model["cat_avg"][idx], idx == top_idx)

    with st.expander("Why this prediction — local feature contribution"):
        try:
            contributions = local_shap_contribution(bundle, row, top_idx)
            max_abs = max(abs(v) for _, v in contributions) or 1.0
            for name, val in contributions:
                shap_bar(name, val, max_abs)
            st.markdown(
                '<div class="footnote">CatBoost SHAP values for the predicted class. '
                'Teal pushes toward this class, rust pushes away.</div>',
                unsafe_allow_html=True,
            )
        except Exception as e:
            st.info(f"Local explanation unavailable: {e}")

    with st.expander("Global feature importance"):
        try:
            ranking = global_shap_importance(str(BUNDLE_PATH), bundle)
            max_val = max(v for _, v in ranking) or 1.0
            for name, val in ranking:
                shap_bar(name, val, max_val)
            st.markdown(
                '<div class="footnote">Mean |SHAP| across the stored background sample — '
                'a fixed reference, independent of the current input.</div>',
                unsafe_allow_html=True,
            )
        except Exception as e:
            st.info(f"Global importance unavailable: {e}")

    with st.expander("Raw feature vector"):
        st.dataframe(row.T.rename(columns={0: "value"}), width='stretch')

    with st.expander("About this model"):
        st.markdown(f"""
        <span class="footnote">
        Blend of independently trained LightGBM and CatBoost fold ensembles.
        XGBoost was trained during development but carries a 0.0 blend weight
        in this production configuration, so it is not loaded.<br><br>
        Classes: {", ".join(class_map.values())}<br>
        Feature count: {len(bundle["feature_names"])}
        </span>
        """, unsafe_allow_html=True)


# ── Entry point ────────────────────────────────────────────────────────
def main():
    st.set_page_config(page_title="Stellar Classifier", page_icon="◐", layout="wide")
    inject_css()

    bundle = load_bundle()
    if bundle is None:
        st.error(f"No bundle found at `{BUNDLE_PATH}`. Run Notebook 05 first to produce it.")
        st.stop()

    render_header(bundle)
    row = render_sidebar(bundle)

    left, right = st.columns([1, 1])
    with left:
        st.markdown('<div class="eyebrow">Console</div>', unsafe_allow_html=True)
        st.markdown("Adjust values in the sidebar, then classify.")
        classify = st.button("Classify Observation", type="primary", width='content')
    with right:
        st.empty()

    if classify or "last_proba" in st.session_state:
        if classify:
            proba, per_model = run_inference(bundle, row)
            st.session_state["last_proba"] = proba
            st.session_state["last_per_model"] = per_model
            st.session_state["last_row"] = row
        render_result(
            bundle,
            st.session_state["last_row"],
            st.session_state["last_proba"],
            st.session_state["last_per_model"],
        )


if __name__ == "__main__":
    main()