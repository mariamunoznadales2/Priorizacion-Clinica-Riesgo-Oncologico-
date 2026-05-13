
from __future__ import annotations

import base64
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.calibration import calibration_curve as sk_cal_curve
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler


APP_DIR = Path(__file__).resolve().parent
DATA_PATH = APP_DIR / "data" / "cancer_dataset_unido.csv"
ASSET_DIR = APP_DIR / "assets"
HERO_IMAGE = ASSET_DIR / "heart.png"

RANDOM_STATE = 42
TARGET = "cancer"
ID_COL = "paciente_id"
LEAKAGE = ["coste_total", "coste_farmaco", "num_ingresos", "dias_hospital", "vive", "alcohol"]

THRESHOLDS = {"alto": 0.7, "medio": 0.3}

MUTATION_COLS = ["mut_BRCA1", "mut_TP53", "mut_EGFR", "mut_KRAS", "mut_PIK3CA", "mut_ALK", "mut_BRAF"]
COMORBIDITY_COLS = {
    "diabetes": "Diabetes",
    "hipertension": "HTA",
    "obesidad": "Obesidad",
    "enfermedad_cardiaca": "Cardiopatía",
    "asma": "Asma",
    "epoc": "EPOC",
}
BIOMARKERS = ["glucosa", "colesterol", "trigliceridos", "hemoglobina", "leucocitos", "plaquetas", "creatinina"]


# ────────────────────────────────────────────────────────────────────────────
# Data + model
# ────────────────────────────────────────────────────────────────────────────

def categorize(p: float) -> str:
    if p > THRESHOLDS["alto"]:
        return "Alto"
    if p >= THRESHOLDS["medio"]:
        return "Medio"
    return "Bajo"


@st.cache_resource(show_spinner="Inicializando motor clínico…")
def load_and_train():
    df = pd.read_csv(DATA_PATH)

    excluded = [TARGET, ID_COL] + LEAKAGE
    X = df.drop(columns=excluded)
    y = df[TARGET].astype(int)
    ids = df[ID_COL]

    numeric = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical = X.select_dtypes(exclude=["number", "bool"]).columns.tolist()

    pre = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical),
        ],
        remainder="drop",
    )

    X_train, X_test, y_train, y_test, _ids_train, ids_test = train_test_split(
        X, y, ids, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )

    X_train_p = pre.fit_transform(X_train)
    X_test_p = pre.transform(X_test)

    model = HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.06, l2_regularization=0.1,
        class_weight="balanced", random_state=RANDOM_STATE,
    )
    model.fit(X_train_p, y_train)

    proba_test = model.predict_proba(X_test_p)[:, 1]
    pred_test = (proba_test >= 0.65).astype(int)

    cm = confusion_matrix(y_test, pred_test)
    cal_frac_pos, cal_mean_pred = sk_cal_curve(y_test, proba_test, n_bins=10, strategy="uniform")

    risk_cats = np.array([categorize(p) for p in proba_test])
    mut_by_risk: dict[str, dict[str, float]] = {}
    for cat in ["Alto", "Bajo"]:
        mask = risk_cats == cat
        if mask.sum() > 0:
            subset = X_test[mask]
            mut_by_risk[cat] = {m.replace("mut_", ""): float(subset[m].mean() * 100) for m in MUTATION_COLS}
        else:
            mut_by_risk[cat] = {m.replace("mut_", ""): 0.0 for m in MUTATION_COLS}

    fpr, tpr, _ = roc_curve(y_test, proba_test)
    prec, rec, _ = precision_recall_curve(y_test, proba_test)

    rf = RandomForestClassifier(
        n_estimators=50, min_samples_leaf=2,
        class_weight="balanced", n_jobs=-1, random_state=RANDOM_STATE,
    )
    rf.fit(X_train_p, y_train)
    feature_names = pre.get_feature_names_out()

    def to_original(name: str) -> str:
        if name.startswith("num__"):
            return name.replace("num__", "")
        if name.startswith("cat__"):
            stem = name.replace("cat__", "")
            for col in categorical:
                if stem == col or stem.startswith(f"{col}_"):
                    return col
            return stem
        return name

    imp_df = (
        pd.DataFrame({"f": feature_names, "imp": rf.feature_importances_})
        .assign(orig=lambda d: d["f"].map(to_original))
        .groupby("orig", as_index=False)["imp"].sum()
        .sort_values("imp", ascending=False)
        .head(10)
    )

    metrics = {
        "auc": roc_auc_score(y_test, proba_test),
        "recall": recall_score(y_test, pred_test, zero_division=0),
        "f1": f1_score(y_test, pred_test, zero_division=0),
        "prevalence": float(y.mean()),
        "n_test": int(len(y_test)),
        "n_total": int(len(y)),
    }

    test_df = X_test.copy()
    test_df[ID_COL] = ids_test.values
    test_df["proba"] = proba_test
    test_df["y_true"] = y_test.values

    # Heatmap zona x age bin (mean predicted risk on test)
    test_df["edad_bin"] = pd.cut(test_df["edad"], bins=[0, 35, 50, 60, 70, 120],
                                 labels=["<35", "35-49", "50-59", "60-69", "70+"])
    heatmap = (
        test_df.groupby(["zona", "edad_bin"], observed=True)["proba"].mean()
        .unstack("edad_bin").reindex(columns=["<35", "35-49", "50-59", "60-69", "70+"])
    )

    # Cumulative incidence by age (real data, full cohort)
    age_sorted = df.sort_values("edad")
    cum_incidence = (
        age_sorted.assign(cum=age_sorted["cancer"].cumsum() / np.arange(1, len(age_sorted) + 1))
        [["edad", "cum"]]
    )

    # Biomarker percentiles for top patient (vs full cohort)
    top_pid = test_df.sort_values("proba", ascending=False).iloc[0][ID_COL]
    top_row = df.loc[df[ID_COL] == top_pid].iloc[0]
    radar_values = []
    for b in BIOMARKERS:
        pct = (df[b] <= top_row[b]).mean() * 100
        radar_values.append(pct)

    curves = {
        "fpr": fpr.tolist(), "tpr": tpr.tolist(),
        "prec": prec.tolist(), "rec": rec.tolist(),
        "importance": imp_df.to_dict("records"),
        "heatmap": heatmap,
        "cum_incidence": cum_incidence,
        "radar_values": radar_values,
        "radar_labels": [b.capitalize() for b in BIOMARKERS],
        "top_pid": top_pid,
        "confusion_matrix": cm.tolist(),
        "cal_frac_pos": cal_frac_pos.tolist(),
        "cal_mean_pred": cal_mean_pred.tolist(),
        "mut_by_risk": mut_by_risk,
    }

    return df, test_df, metrics, curves


def image_to_base64(path: Path) -> str:
    if not path.exists():
        return ""
    return base64.b64encode(path.read_bytes()).decode("utf-8")


# ────────────────────────────────────────────────────────────────────────────
# Page setup + CSS
# ────────────────────────────────────────────────────────────────────────────

def configure_page() -> None:
    st.set_page_config(
        page_title="OncoPriority · Clinical AI",
        layout="wide",
        initial_sidebar_state="collapsed",
    )


def inject_css() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500;700&display=swap');

        :root {
            --bg-0: #050a17;
            --bg-1: #0a1428;
            --bg-2: #0e1a35;
            --surface: rgba(255,255,255,0.035);
            --surface-2: rgba(255,255,255,0.06);
            --ink: #e8eef9;
            --ink-2: #c1cce0;
            --muted: #7a8aa6;
            --line: rgba(255,255,255,0.08);
            --line-2: rgba(255,255,255,0.14);
            --blue: #4f8bff;
            --blue-2: #6ea4ff;
            --cyan: #22d3ee;
            --violet: #8b6dff;
            --red: #ef4f6c;
            --amber: #f5b942;
            --green: #2dd4a7;
        }

        html, body, [data-testid="stAppViewContainer"] {
            background:
                radial-gradient(1200px 700px at 15% -10%, rgba(79,139,255,0.18) 0%, transparent 50%),
                radial-gradient(900px 600px at 95% 5%, rgba(34,211,238,0.10) 0%, transparent 55%),
                radial-gradient(800px 500px at 50% 110%, rgba(239,79,108,0.08) 0%, transparent 60%),
                linear-gradient(180deg, #050a17 0%, #08111f 50%, #050a17 100%);
            color: var(--ink);
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        }
        * { font-family: 'Inter', -apple-system, sans-serif; }
        [data-testid="stHeader"] { background: transparent; }
        [data-testid="stSidebar"] { display: none; }
        #MainMenu, footer { visibility: hidden; }
        .mono { font-family: 'JetBrains Mono', monospace; }

        .block-container {
            max-width: 1520px;
            padding: 1.2rem 2rem 4rem;
        }

        /* ─────── Top brand bar ─────── */
        .topbar {
            display: flex; justify-content: space-between; align-items: center;
            margin-bottom: 0.8rem;
        }
        .brand {
            display: flex; align-items: center; gap: 0.7rem;
            font-weight: 700; color: var(--ink); font-size: 0.95rem;
        }
        .brand-mark {
            width: 30px; height: 30px; border-radius: 9px;
            background: linear-gradient(135deg, #4f8bff 0%, #22d3ee 100%);
            display: grid; place-items: center; color: #050a17;
            font-weight: 800; font-size: 0.85rem;
            box-shadow: 0 0 24px rgba(79,139,255,0.55);
        }
        .brand-sub { color: var(--muted); font-weight: 500; font-size: 0.78rem; margin-left: 0.4rem; }
        .nav { display: flex; gap: 0.4rem; align-items: center; }
        .nav-item {
            padding: 0.4rem 0.85rem; border-radius: 999px;
            font-size: 0.78rem; font-weight: 500; color: var(--muted);
            border: 1px solid transparent;
        }
        .nav-item.active {
            color: var(--ink); background: var(--surface);
            border-color: var(--line); box-shadow: 0 0 18px rgba(79,139,255,0.18);
        }
        .status-pill {
            display: inline-flex; align-items: center; gap: 0.5rem;
            padding: 0.4rem 0.85rem; background: var(--surface);
            border: 1px solid var(--line); border-radius: 999px;
            font-size: 0.76rem; color: var(--ink-2); font-weight: 500;
            font-family: 'JetBrains Mono', monospace;
        }
        .pulse-dot {
            width: 7px; height: 7px; border-radius: 50%;
            background: var(--green);
            box-shadow: 0 0 0 0 rgba(45,212,167,0.6);
            animation: pulse 2.4s infinite;
        }
        @keyframes pulse {
            0% { box-shadow: 0 0 0 0 rgba(45,212,167,0.5); }
            70% { box-shadow: 0 0 0 8px rgba(45,212,167,0); }
            100% { box-shadow: 0 0 0 0 rgba(45,212,167,0); }
        }

        /* ─────── Big centered title ─────── */
        .page-title-wrap {
            text-align: center !important;
            margin: 1.2rem auto 1.8rem;
            position: relative;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            width: 100%;
        }
        .page-eyebrow {
            display: inline-flex; align-items: center; gap: 0.5rem;
            padding: 0.35rem 0.85rem;
            background: linear-gradient(90deg, rgba(79,139,255,0.12), rgba(34,211,238,0.12));
            border: 1px solid rgba(79,139,255,0.22);
            border-radius: 999px;
            color: var(--cyan); font-weight: 600;
            font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.18em;
            margin-bottom: 1rem;
        }
        .page-eyebrow .dot { width: 6px; height: 6px; background: var(--cyan); border-radius: 50%; box-shadow: 0 0 10px var(--cyan); }
        .page-title {
            font-size: clamp(2.4rem, 4.6vw, 4rem);
            font-weight: 800; letter-spacing: -0.04em;
            line-height: 1.02;
            background: linear-gradient(180deg, #ffffff 0%, #b9c6e0 100%);
            -webkit-background-clip: text;
            background-clip: text;
            -webkit-text-fill-color: transparent;
            margin: 0;
        }
        .page-title .accent {
            color: #4f8bff;
            background: none;
            -webkit-text-fill-color: #4f8bff;
        }
        .page-title-wrap .page-subtitle,
        .page-subtitle {
            display: block !important;
            color: var(--muted) !important;
            font-size: 1.02rem !important;
            margin: 1rem auto 0 auto !important;
            max-width: 720px !important;
            line-height: 1.55 !important;
            text-align: center !important;
        }
        .page-title-wrap p { text-align: center !important; margin-left: auto !important; margin-right: auto !important; }

        /* ─────── Section headers ─────── */
        .section {
            margin-top: 2.4rem;
            margin-bottom: 1.1rem;
            display: flex; align-items: baseline; justify-content: space-between;
            gap: 1rem;
        }
        .section-head { display: flex; align-items: baseline; gap: 0.85rem; }
        .section-num {
            font-family: 'JetBrains Mono', monospace;
            color: var(--blue); font-weight: 700; font-size: 0.78rem;
            opacity: 0.85;
        }
        .section-title {
            font-size: 1.25rem; font-weight: 700;
            color: var(--ink); margin: 0; letter-spacing: -0.02em;
        }
        .section-subtitle {
            font-size: 0.85rem; color: var(--muted);
            font-weight: 500;
        }
        .section-divider {
            height: 1px; flex: 1;
            background: linear-gradient(90deg, var(--line) 0%, transparent 100%);
            margin: 0 0.5rem;
            align-self: center;
        }

        /* ─────── Glass card ─────── */
        .card {
            position: relative;
            background: linear-gradient(180deg, rgba(255,255,255,0.04) 0%, rgba(255,255,255,0.015) 100%);
            border: 1px solid var(--line);
            border-radius: 16px;
            padding: 1.1rem 1.2rem;
            backdrop-filter: blur(14px);
            -webkit-backdrop-filter: blur(14px);
            box-shadow: 0 1px 0 rgba(255,255,255,0.04) inset, 0 8px 30px rgba(0,0,0,0.25);
            transition: transform 0.25s ease, border-color 0.25s ease, box-shadow 0.25s ease;
        }
        .card::before {
            content: ""; position: absolute;
            top: 0; left: 1.2rem; right: 1.2rem; height: 1px;
            background: linear-gradient(90deg, transparent, rgba(79,139,255,0.3), transparent);
        }
        .card:hover {
            transform: translateY(-2px);
            border-color: var(--line-2);
            box-shadow: 0 1px 0 rgba(255,255,255,0.06) inset, 0 14px 40px rgba(0,0,0,0.35), 0 0 30px rgba(79,139,255,0.08);
        }

        /* ─────── KPI cards ─────── */
        .kpi {
            display: flex; flex-direction: column; gap: 0.4rem;
            padding: 1rem 1.15rem 0.95rem;
            min-height: 138px;
        }
        .kpi-head {
            display: flex; justify-content: space-between; align-items: center;
            color: var(--muted); font-size: 0.72rem; font-weight: 600;
            text-transform: uppercase; letter-spacing: 0.1em;
        }
        .kpi-icon {
            width: 24px; height: 24px; border-radius: 7px;
            background: linear-gradient(135deg, rgba(79,139,255,0.18), rgba(34,211,238,0.12));
            border: 1px solid rgba(79,139,255,0.25);
            display: grid; place-items: center;
            font-size: 0.78rem; color: var(--cyan);
        }
        .kpi-value {
            font-size: 2rem; font-weight: 700; color: var(--ink);
            letter-spacing: -0.04em; line-height: 1;
            font-family: 'JetBrains Mono', monospace;
        }
        .kpi-meta {
            display: flex; justify-content: space-between; align-items: center;
            font-size: 0.78rem; color: var(--muted); margin-top: auto;
        }
        .kpi-trend.up { color: var(--green); }
        .kpi-trend.down { color: var(--red); }
        .sparkline { width: 100%; height: 28px; }

        /* ─────── Hero ─────── */
        .hero {
            position: relative;
            border-radius: 22px;
            background: linear-gradient(180deg, rgba(15,28,55,0.72) 0%, rgba(8,17,33,0.85) 100%);
            border: 1px solid var(--line);
            padding: 1.4rem 1.6rem 1.6rem;
            overflow: hidden;
            box-shadow: 0 20px 60px rgba(0,0,0,0.4);
        }
        .hero::before {
            content: ""; position: absolute; inset: 0;
            background:
                radial-gradient(800px 500px at 50% 50%, rgba(239,79,108,0.16) 0%, transparent 55%),
                radial-gradient(500px 400px at 50% 30%, rgba(79,139,255,0.18) 0%, transparent 60%);
            pointer-events: none;
        }
        .hero::after {
            content: ""; position: absolute; inset: 0;
            background-image:
                linear-gradient(rgba(255,255,255,0.025) 1px, transparent 1px),
                linear-gradient(90deg, rgba(255,255,255,0.025) 1px, transparent 1px);
            background-size: 38px 38px;
            mask-image: radial-gradient(ellipse at center, black 30%, transparent 75%);
            -webkit-mask-image: radial-gradient(ellipse at center, black 30%, transparent 75%);
            pointer-events: none;
        }
        .hero-inner { position: relative; z-index: 1; }
        .hero-header {
            display: flex; justify-content: space-between; align-items: flex-start;
            margin-bottom: 0.8rem;
        }
        .hero-eyebrow {
            font-size: 0.7rem; font-weight: 700;
            text-transform: uppercase; letter-spacing: 0.18em;
            color: var(--cyan);
            font-family: 'JetBrains Mono', monospace;
        }
        .hero-title {
            font-size: 1.25rem; font-weight: 700;
            color: var(--ink); margin: 0.3rem 0 0;
            letter-spacing: -0.02em;
        }
        .hero-meta { font-size: 0.82rem; color: var(--muted); text-align: right; line-height: 1.6; }
        .hero-meta strong { color: var(--ink); font-family: 'JetBrains Mono', monospace; }

        .hero-stage {
            position: relative;
            width: 100%;
            height: 540px;
        }
        .hero-organ {
            position: absolute;
            top: 50%; left: 50%;
            transform: translate(-50%, -50%);
            width: auto;
            height: auto;
            display: grid; place-items: center;
            z-index: 1;
        }
        .hero-organ::before {
            content: "";
            position: absolute;
            top: 50%; left: 50%;
            transform: translate(-50%, -50%);
            width: 480px; height: 480px;
            border-radius: 50%;
            background:
                radial-gradient(circle, rgba(239,79,108,0.32) 0%, rgba(79,139,255,0.18) 35%, transparent 70%);
            filter: blur(40px);
            z-index: 0;
            animation: organ-pulse 4s ease-in-out infinite;
        }
        @keyframes organ-pulse {
            0%, 100% { transform: translate(-50%, -50%) scale(1); opacity: 0.85; }
            50% { transform: translate(-50%, -50%) scale(1.06); opacity: 1; }
        }
        .hero-organ::after {
            content: "";
            position: absolute;
            top: 50%; left: 50%;
            transform: translate(-50%, -50%);
            width: 380px; height: 380px;
            border-radius: 50%;
            border: 1px dashed rgba(79,139,255,0.25);
            animation: ring-spin 28s linear infinite;
            z-index: 0;
        }
        @keyframes ring-spin {
            from { transform: translate(-50%, -50%) rotate(0); }
            to { transform: translate(-50%, -50%) rotate(360deg); }
        }
        .hero-organ img {
            position: relative; z-index: 2;
            width: auto;
            height: auto;
            max-width: 100%;
            max-height: 500px;
            object-fit: contain;
            filter: drop-shadow(0 0 30px rgba(239,79,108,0.45)) drop-shadow(0 0 60px rgba(79,139,255,0.3));
        }

        .overlay {
            position: absolute;
            background: linear-gradient(180deg, rgba(20,32,60,0.9) 0%, rgba(12,22,42,0.92) 100%);
            border: 1px solid rgba(255,255,255,0.10);
            border-radius: 12px;
            padding: 0.7rem 0.85rem;
            box-shadow: 0 14px 36px rgba(0,0,0,0.45), 0 0 24px rgba(79,139,255,0.08);
            width: 200px;
            z-index: 3;
            backdrop-filter: blur(18px);
            -webkit-backdrop-filter: blur(18px);
        }
        .overlay::before {
            content: ""; position: absolute; top: 50%;
            width: 50px; height: 1px;
            transform: translateY(-50%);
        }
        .overlay::after {
            content: ""; position: absolute; top: 50%;
            width: 8px; height: 8px; border-radius: 50%;
            background: var(--cyan);
            transform: translateY(-50%);
            box-shadow: 0 0 10px var(--cyan), 0 0 0 4px rgba(34,211,238,0.18);
        }
        .overlay.left::before {
            right: -50px;
            background: linear-gradient(90deg, rgba(34,211,238,0.5), transparent);
        }
        .overlay.left::after { right: -54px; }
        .overlay.right::before {
            left: -50px;
            background: linear-gradient(270deg, rgba(34,211,238,0.5), transparent);
        }
        .overlay.right::after { left: -54px; }

        .ov-l1 { top: 6%;   left: 0; }
        .ov-l2 { top: 42%;  left: 0; }
        .ov-l3 { top: 78%;  left: 0; }
        .ov-r1 { top: 6%;   right: 0; }
        .ov-r2 { top: 42%;  right: 0; }
        .ov-r3 { top: 78%;  right: 0; }

        .overlay-head {
            display: flex; align-items: center; gap: 0.45rem;
            color: var(--muted); font-size: 0.7rem; font-weight: 600;
            text-transform: uppercase; letter-spacing: 0.06em;
        }
        .overlay-icon {
            width: 18px; height: 18px; border-radius: 5px;
            background: rgba(79,139,255,0.16);
            color: var(--blue-2);
            display: grid; place-items: center;
            font-size: 0.7rem;
        }
        .overlay-value {
            color: var(--ink); font-size: 1.2rem; font-weight: 700;
            margin-top: 0.3rem; letter-spacing: -0.02em; line-height: 1.05;
            font-family: 'JetBrains Mono', monospace;
        }
        .overlay-tag {
            color: var(--blue-2); font-size: 0.74rem; font-weight: 600;
            margin-left: 0.4rem; letter-spacing: 0.02em;
        }
        .overlay-tag.alto { color: var(--red); }
        .overlay-tag.medio { color: var(--amber); }
        .overlay-tag.bajo { color: var(--green); }
        .overlay-bar {
            height: 3px; border-radius: 2px;
            background: rgba(255,255,255,0.07); margin-top: 0.45rem; overflow: hidden;
        }
        .overlay-bar > span {
            display: block; height: 100%;
            background: linear-gradient(90deg, var(--blue) 0%, var(--cyan) 100%);
            border-radius: 2px;
            box-shadow: 0 0 8px rgba(79,139,255,0.5);
        }

        /* ─────── Priority panel ─────── */
        .panel-title {
            font-size: 1rem; font-weight: 700; color: var(--ink);
            margin: 0 0 0.2rem; letter-spacing: -0.015em;
        }
        .panel-subtitle { font-size: 0.78rem; color: var(--muted); margin: 0 0 1rem; }

        .patient-row {
            display: flex; align-items: center; justify-content: space-between;
            padding: 0.7rem 0;
            border-bottom: 1px solid var(--line);
        }
        .patient-row:last-child { border-bottom: 0; }
        .patient-left { display: flex; align-items: center; gap: 0.7rem; }
        .patient-rank {
            width: 24px; height: 24px; border-radius: 7px;
            background: rgba(79,139,255,0.10); color: var(--blue-2);
            display: grid; place-items: center;
            font-size: 0.74rem; font-weight: 700;
            font-family: 'JetBrains Mono', monospace;
            border: 1px solid rgba(79,139,255,0.18);
        }
        .patient-id {
            font-weight: 600; color: var(--ink); font-size: 0.85rem;
            font-family: 'JetBrains Mono', monospace;
        }
        .patient-cat { font-size: 0.7rem; color: var(--muted); margin-top: 0.1rem; }
        .patient-right { display: flex; align-items: center; gap: 0.6rem; }
        .risk-bar { width: 60px; height: 4px; border-radius: 3px; background: rgba(255,255,255,0.06); overflow: hidden; }
        .risk-bar > span { display: block; height: 100%; border-radius: 3px; }
        .risk-bar.alto > span { background: linear-gradient(90deg, var(--red), #ff8aa0); box-shadow: 0 0 8px rgba(239,79,108,0.5); }
        .risk-bar.medio > span { background: linear-gradient(90deg, var(--amber), #fcd76d); }
        .risk-bar.bajo > span { background: linear-gradient(90deg, var(--green), #6deeca); }
        .risk-value {
            font-weight: 700; font-size: 0.84rem; color: var(--ink);
            font-family: 'JetBrains Mono', monospace; min-width: 36px; text-align: right;
        }
        .badge {
            display: inline-block; padding: 0.14rem 0.5rem;
            border-radius: 999px; font-size: 0.66rem; font-weight: 700;
            text-transform: uppercase; letter-spacing: 0.06em;
        }
        .badge.alto { background: rgba(239,79,108,0.14); color: var(--red); border: 1px solid rgba(239,79,108,0.3); }
        .badge.medio { background: rgba(245,185,66,0.13); color: var(--amber); border: 1px solid rgba(245,185,66,0.3); }
        .badge.bajo { background: rgba(45,212,167,0.13); color: var(--green); border: 1px solid rgba(45,212,167,0.3); }

        /* ─────── Model card ─────── */
        .model-card {
            background:
                radial-gradient(400px 250px at 100% 0%, rgba(34,211,238,0.18), transparent 70%),
                linear-gradient(135deg, rgba(15,28,55,0.85) 0%, rgba(40,30,80,0.6) 100%);
            border: 1px solid rgba(79,139,255,0.22);
            border-radius: 16px;
            padding: 1.3rem 1.4rem;
            position: relative; overflow: hidden;
            box-shadow: 0 14px 40px rgba(0,0,0,0.4), 0 0 30px rgba(79,139,255,0.08);
        }
        .model-eyebrow {
            font-size: 0.7rem; font-weight: 700;
            text-transform: uppercase; letter-spacing: 0.12em;
            color: var(--cyan);
            font-family: 'JetBrains Mono', monospace;
        }
        .model-name {
            font-size: 1.5rem; font-weight: 700;
            margin: 0.45rem 0 0.5rem;
            letter-spacing: -0.025em; color: var(--ink);
        }
        .model-desc {
            font-size: 0.86rem; line-height: 1.55;
            color: var(--ink-2);
        }
        .model-tags {
            display: flex; flex-wrap: wrap; gap: 0.35rem;
            margin-top: 0.85rem;
        }
        .model-tag {
            font-size: 0.7rem; padding: 0.25rem 0.55rem;
            background: rgba(79,139,255,0.12);
            border: 1px solid rgba(79,139,255,0.2);
            border-radius: 6px; color: var(--blue-2);
            font-family: 'JetBrains Mono', monospace;
        }

        /* ─────── Clinical message ─────── */
        .clinical-msg {
            display: flex; align-items: flex-start; gap: 0.95rem;
            background: linear-gradient(90deg, rgba(245,185,66,0.06), rgba(245,185,66,0.02));
            border: 1px solid rgba(245,185,66,0.20);
            border-radius: 14px;
            padding: 1rem 1.2rem;
            margin-top: 1.4rem;
        }
        .clinical-icon {
            width: 28px; height: 28px; border-radius: 8px;
            background: rgba(245,185,66,0.14); color: var(--amber);
            display: grid; place-items: center;
            font-weight: 800; flex-shrink: 0;
        }
        .clinical-text { font-size: 0.88rem; line-height: 1.55; color: var(--ink-2); }
        .clinical-text strong { color: var(--ink); }

        /* ─────── Streamlit overrides ─────── */
        .stPlotlyChart > div { background: transparent !important; }
        div[data-testid="stPlotlyChart"] { background: transparent !important; }

        /* Kill any white default backgrounds Streamlit injects */
        div[data-testid="stVerticalBlockBorderWrapper"],
        div[data-testid="stVerticalBlock"],
        div[data-testid="stHorizontalBlock"],
        div[data-testid="column"],
        div[data-testid="stMarkdown"],
        div[data-testid="stMarkdownContainer"],
        div[data-testid="element-container"],
        div[data-testid="stElementContainer"],
        section[data-testid="stMain"],
        .main .block-container,
        .element-container,
        .stApp,
        .main {
            background: transparent !important;
            background-color: transparent !important;
        }
        /* Also nuke any white in hero descendants except declared cards */
        .hero [data-testid="stMarkdown"],
        .hero [data-testid="stMarkdownContainer"],
        .hero [data-testid="element-container"],
        .hero [data-testid="stVerticalBlock"] {
            background: transparent !important;
        }

        /* Chart cards (containers with key starting chartcard_) */
        [class*="st-key-chartcard_"] {
            background: linear-gradient(180deg, rgba(255,255,255,0.04) 0%, rgba(255,255,255,0.015) 100%) !important;
            border: 1px solid var(--line) !important;
            border-radius: 16px !important;
            backdrop-filter: blur(14px);
            -webkit-backdrop-filter: blur(14px);
            box-shadow: 0 1px 0 rgba(255,255,255,0.04) inset, 0 8px 30px rgba(0,0,0,0.25);
            padding: 1rem 1.1rem 0.6rem !important;
            transition: border-color 0.25s ease, box-shadow 0.25s ease;
        }
        [class*="st-key-chartcard_"]:hover {
            border-color: var(--line-2) !important;
            box-shadow: 0 1px 0 rgba(255,255,255,0.06) inset, 0 14px 40px rgba(0,0,0,0.35), 0 0 30px rgba(79,139,255,0.08);
        }

        /* Selectbox dark style */
        div[data-testid="stSelectbox"] label { color: var(--muted) !important; font-size: 0.74rem !important;
            font-weight: 600 !important; text-transform: uppercase !important; letter-spacing: 0.12em !important; }
        div[data-testid="stSelectbox"] div[data-baseweb="select"] > div {
            background: rgba(255,255,255,0.04) !important;
            border: 1px solid var(--line) !important;
            border-radius: 10px !important;
            color: var(--ink) !important;
            font-family: 'JetBrains Mono', monospace !important;
            min-height: 42px !important;
        }
        div[data-testid="stSelectbox"] div[data-baseweb="select"] > div:hover {
            border-color: rgba(79,139,255,0.4) !important;
            box-shadow: 0 0 18px rgba(79,139,255,0.10) !important;
        }
        div[data-baseweb="popover"] {
            background: rgba(10,20,40,0.98) !important;
            border: 1px solid var(--line) !important;
            border-radius: 10px !important;
        }
        div[data-baseweb="popover"] li { color: var(--ink-2) !important; font-family: 'JetBrains Mono', monospace !important; }
        div[data-baseweb="popover"] li:hover { background: rgba(79,139,255,0.12) !important; color: var(--ink) !important; }

        /* Patient picker container */
        .picker-wrap {
            display: flex; align-items: center; gap: 1rem;
            padding: 0.85rem 1.1rem;
            background: linear-gradient(180deg, rgba(255,255,255,0.04) 0%, rgba(255,255,255,0.015) 100%);
            border: 1px solid var(--line);
            border-radius: 14px;
            margin-bottom: 0.9rem;
            backdrop-filter: blur(12px);
        }
        .picker-label {
            color: var(--cyan); font-size: 0.7rem;
            font-weight: 700; text-transform: uppercase; letter-spacing: 0.18em;
            font-family: 'JetBrains Mono', monospace;
            white-space: nowrap;
        }
        .picker-stat {
            display: flex; gap: 1.4rem;
            margin-left: auto;
            font-family: 'JetBrains Mono', monospace;
        }
        .picker-stat-item { display: flex; flex-direction: column; align-items: flex-end; }
        .picker-stat-label { color: var(--muted); font-size: 0.66rem; text-transform: uppercase; letter-spacing: 0.12em; }
        .picker-stat-value { color: var(--ink); font-size: 1rem; font-weight: 700; }
        .picker-stat-value.alto { color: var(--red); }
        .picker-stat-value.medio { color: var(--amber); }
        .picker-stat-value.bajo { color: var(--green); }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ────────────────────────────────────────────────────────────────────────────
# UI components
# ────────────────────────────────────────────────────────────────────────────

def render_topbar(metrics: dict) -> None:
    st.markdown(
        f"""
        <div class="topbar">
            <div class="brand">
                <div class="brand-mark">O</div>
                <div>Onco<span style="color:var(--cyan);">Priority</span><span class="brand-sub">· Clinical Intelligence Platform</span></div>
            </div>
            <div class="status-pill">
                <span class="pulse-dot"></span>
                MODEL · LIVE · n={metrics['n_test']:,}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_hero_title() -> None:
    st.markdown(
        """
        <div class="page-title-wrap">
            <span class="page-eyebrow"><span class="dot"></span>Predictive Oncology · ML-assisted Triage</span>
            <h1 class="page-title">
                Sistema de priorización clínica<br>
                de <span class="accent">riesgo oncológico</span>
            </h1>
            <p class="page-subtitle">
                Plataforma de apoyo a la decisión clínica que ordena pacientes según probabilidad
                de cáncer, integra biomarcadores, mutaciones y comorbilidades, y opera bajo
                supervisión médica continua.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_section_header(num: str, title: str, subtitle: str) -> None:
    st.markdown(
        f"""
        <div class="section">
            <div class="section-head">
                <span class="section-num">{num}</span>
                <h2 class="section-title">{title}</h2>
            </div>
            <div class="section-divider"></div>
            <span class="section-subtitle">{subtitle}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def sparkline_svg(values: list[float], color: str = "#4f8bff", fill: str = "rgba(79,139,255,0.18)") -> str:
    if not values:
        return ""
    n = len(values)
    w, h = 200, 28
    vmin, vmax = min(values), max(values)
    rng = (vmax - vmin) or 1
    pts = [(i * w / (n - 1), h - ((v - vmin) / rng) * (h - 4) - 2) for i, v in enumerate(values)]
    path = "M " + " L ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
    fill_path = path + f" L {w},{h} L 0,{h} Z"
    return (
        f'<svg class="sparkline" viewBox="0 0 {w} {h}" preserveAspectRatio="none">'
        f'<path d="{fill_path}" fill="{fill}" stroke="none"/>'
        f'<path d="{path}" fill="none" stroke="{color}" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/>'
        f'</svg>'
    )


def render_kpis(metrics: dict, curves: dict, test_df: pd.DataFrame) -> None:
    # Real sparkline data
    tpr_spark = curves["tpr"][::max(1, len(curves["tpr"]) // 30)]
    prec_spark = curves["prec"][::max(1, len(curves["prec"]) // 30)]
    proba_sorted = sorted(test_df["proba"].tolist(), reverse=True)[:60]
    n_high = int((test_df["proba"] > THRESHOLDS["alto"]).sum())

    items = [
        {
            "icon": "◈", "label": "AUC",
            "value": f"{metrics['auc']:.3f}".replace(".", ","),
            "trend_class": "up", "trend": "↑ ordenación",
            "spark": tpr_spark, "color": "#4f8bff", "fill": "rgba(79,139,255,0.18)",
        },
        {
            "icon": "❤", "label": "Recall",
            "value": f"{metrics['recall']:.3f}".replace(".", ","),
            "trend_class": "up", "trend": "↑ sensibilidad",
            "spark": prec_spark, "color": "#22d3ee", "fill": "rgba(34,211,238,0.16)",
        },
        {
            "icon": "◐", "label": "F1",
            "value": f"{metrics['f1']:.3f}".replace(".", ","),
            "trend_class": "up", "trend": "→ equilibrio",
            "spark": tpr_spark[::-1], "color": "#8b6dff", "fill": "rgba(139,109,255,0.16)",
        },
        {
            "icon": "▲", "label": "Prevalencia",
            "value": f"{metrics['prevalence']*100:.1f}%",
            "trend_class": "down", "trend": f"{metrics['n_total']:,} pacientes",
            "spark": proba_sorted, "color": "#ef4f6c", "fill": "rgba(239,79,108,0.14)",
        },
        {
            "icon": "⬢", "label": "Alto riesgo",
            "value": f"{n_high:,}",
            "trend_class": "down", "trend": "p > 0.70",
            "spark": [v for v in proba_sorted if v > 0.5] or proba_sorted, "color": "#f5b942", "fill": "rgba(245,185,66,0.14)",
        },
    ]

    cols = st.columns(5, gap="small")
    for col, k in zip(cols, items):
        with col:
            spark = sparkline_svg(k["spark"], k["color"], k["fill"])
            st.markdown(
                f"""
                <div class="card kpi">
                    <div class="kpi-head">
                        <span>{k['label']}</span>
                        <span class="kpi-icon">{k['icon']}</span>
                    </div>
                    <div class="kpi-value">{k['value']}</div>
                    {spark}
                    <div class="kpi-meta">
                        <span class="kpi-trend {k['trend_class']}">{k['trend']}</span>
                        <span class="mono">test set</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def build_overlays(row: pd.Series, proba: float) -> tuple[list[dict], list[dict]]:
    cat = categorize(proba)
    cat_class = cat.lower()

    mutations = [c.replace("mut_", "") for c in MUTATION_COLS if int(row.get(c, 0)) == 1]
    mut_value = mutations[0] if mutations else "Ninguna"
    mut_tag = f"+{len(mutations)-1} más" if len(mutations) > 1 else ("Detectada" if mutations else "No detectada")

    comorbidities = [label for col, label in COMORBIDITY_COLS.items() if int(row.get(col, 0)) == 1]
    comorb_value = ", ".join(comorbidities[:2]) if comorbidities else "Ninguna"
    comorb_tag = f"+{len(comorbidities)-2}" if len(comorbidities) > 2 else ""

    cardio = "Alterada" if int(row.get("enfermedad_cardiaca", 0)) == 1 else "Normal"
    cardio_tag = "Cardiopatía" if cardio == "Alterada" else "Sin alt."

    left = [
        {
            "icon": "♥", "label": "Riesgo cáncer",
            "value": f"{proba:.2f}", "tag": cat, "tag_class": cat_class,
            "bar": int(proba * 100),
        },
        {
            "icon": "👤", "label": "Edad",
            "value": f"{int(row['edad'])}", "tag": "años", "tag_class": "",
        },
        {
            "icon": "🧬", "label": "Mutación",
            "value": mut_value, "tag": mut_tag, "tag_class": "alto" if mutations else "bajo",
        },
    ]
    right = [
        {
            "icon": "📋", "label": "Comorbilidades",
            "value": comorb_value, "tag": comorb_tag, "tag_class": "",
        },
        {
            "icon": "〰", "label": "F. cardíaca",
            "value": cardio, "tag": cardio_tag,
            "tag_class": "alto" if cardio == "Alterada" else "bajo",
        },
        {
            "icon": "◉", "label": "Prioridad",
            "value": cat,
            "tag": "Revisión 24h" if cat == "Alto" else "Seguimiento",
            "tag_class": cat_class,
        },
    ]
    return left, right


def overlay_html(o: dict, position_class: str) -> str:
    bar = ""
    if "bar" in o:
        bar = f'<div class="overlay-bar"><span style="width:{o["bar"]}%"></span></div>'
    tag_class = o.get("tag_class", "")
    return (
        f'<div class="overlay {position_class}">'
        f'<div class="overlay-head">'
        f'<span class="overlay-icon">{o["icon"]}</span>'
        f'<span>{o["label"]}</span>'
        f'</div>'
        f'<div>'
        f'<span class="overlay-value">{o["value"]}</span>'
        f'<span class="overlay-tag {tag_class}">{o["tag"]}</span>'
        f'</div>'
        f'{bar}'
        f'</div>'
    )


def render_hero(test_df: pd.DataFrame, full_df: pd.DataFrame, selected_pid: str) -> tuple[float, pd.Series]:
    image_b64 = image_to_base64(HERO_IMAGE)
    img_tag = (
        f'<img src="data:image/png;base64,{image_b64}" alt="heart" />'
        if image_b64
        else '<div style="width:380px;height:380px;border-radius:50%;background:rgba(79,139,255,0.15);"></div>'
    )

    proba = float(test_df.loc[test_df[ID_COL] == selected_pid, "proba"].iloc[0])
    row = full_df.loc[full_df[ID_COL] == selected_pid].iloc[0]
    cat = categorize(proba)

    left, right = build_overlays(row, proba)
    left_classes = ["left ov-l1", "left ov-l2", "left ov-l3"]
    right_classes = ["right ov-r1", "right ov-r2", "right ov-r3"]
    overlays_html = "".join(overlay_html(o, c) for o, c in zip(left, left_classes))
    overlays_html += "".join(overlay_html(o, c) for o, c in zip(right, right_classes))

    hero_html = (
        f'<div class="hero"><div class="hero-inner">'
        f'<div class="hero-header">'
        f'<div><div class="hero-eyebrow">// PATIENT · {selected_pid}</div>'
        f'<h2 class="hero-title">Perfil clínico individual</h2></div>'
        f'<div class="hero-meta">Categoría · <strong>{cat}</strong><br>'
        f'Probabilidad · <strong>{proba:.3f}</strong></div>'
        f'</div>'
        f'<div class="hero-stage">'
        f'<div class="hero-organ">{img_tag}</div>'
        f'{overlays_html}'
        f'</div>'
        f'</div></div>'
    )
    st.markdown(hero_html, unsafe_allow_html=True)
    return proba, row


def patient_radar_values(full_df: pd.DataFrame, pid: str) -> list[float]:
    row = full_df.loc[full_df[ID_COL] == pid].iloc[0]
    return [float((full_df[b] <= row[b]).mean() * 100) for b in BIOMARKERS]


def render_patient_picker(test_df: pd.DataFrame) -> str:
    sorted_df = test_df.sort_values("proba", ascending=False).reset_index(drop=True)
    options = sorted_df[ID_COL].tolist()

    def fmt(pid: str) -> str:
        r = sorted_df.loc[sorted_df[ID_COL] == pid].iloc[0]
        return f"{pid}  ·  riesgo {float(r['proba']):.3f}  ·  {int(r['edad'])} años"

    sel_col, info_col = st.columns([2, 1.1], gap="medium")
    with sel_col:
        selected = st.selectbox(
            "Seleccionar paciente",
            options=options,
            index=0,
            format_func=fmt,
            key="patient_picker",
        )
    with info_col:
        proba = float(sorted_df.loc[sorted_df[ID_COL] == selected, "proba"].iloc[0])
        rank = int(sorted_df.index[sorted_df[ID_COL] == selected][0]) + 1
        cat = categorize(proba)
        st.markdown(
            f"""
            <div class="picker-wrap" style="margin-top: 1.55rem;">
                <span class="picker-label">// SELECTED</span>
                <div class="picker-stat">
                    <div class="picker-stat-item">
                        <span class="picker-stat-label">Rank</span>
                        <span class="picker-stat-value">#{rank}</span>
                    </div>
                    <div class="picker-stat-item">
                        <span class="picker-stat-label">Riesgo</span>
                        <span class="picker-stat-value">{proba:.3f}</span>
                    </div>
                    <div class="picker-stat-item">
                        <span class="picker-stat-label">Categoría</span>
                        <span class="picker-stat-value {cat.lower()}">{cat}</span>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    return selected


def render_priority_panel(test_df: pd.DataFrame) -> None:
    top = test_df.sort_values("proba", ascending=False).head(7)
    rows_html = ""
    for i, (_, r) in enumerate(top.iterrows(), start=1):
        proba = float(r["proba"])
        cat = categorize(proba)
        cls = cat.lower()
        bar_w = int(proba * 100)
        rows_html += f"""
        <div class="patient-row">
            <div class="patient-left">
                <div class="patient-rank">{i:02d}</div>
                <div>
                    <div class="patient-id">{r[ID_COL]}</div>
                    <div class="patient-cat">{int(r['edad'])} años</div>
                </div>
            </div>
            <div class="patient-right">
                <div class="risk-bar {cls}"><span style="width:{bar_w}%"></span></div>
                <div class="risk-value">{proba:.2f}</div>
                <span class="badge {cls}">{cat}</span>
            </div>
        </div>
        """

    n_alto = int((test_df["proba"] > THRESHOLDS["alto"]).sum())
    n_medio = int(((test_df["proba"] >= THRESHOLDS["medio"]) & (test_df["proba"] <= THRESHOLDS["alto"])).sum())

    st.markdown(
        f"""
        <div class="card" style="height:100%;">
            <h3 class="panel-title">Priority Queue</h3>
            <p class="panel-subtitle">{n_alto} alto · {n_medio} medio · top 7 mostrados</p>
            {rows_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_model_card(metrics: dict) -> None:
    st.markdown(
        f"""
        <div class="model-card">
            <div class="model-eyebrow">// ACTIVE MODEL</div>
            <div class="model-name">HistGradientBoosting</div>
            <div class="model-desc">
                Calibrado con <strong style="color:var(--cyan);">class_weight balanced</strong> sobre {metrics['n_total']:,} pacientes.
                Optimizado para recall en cribado clínico cuando la capacidad asistencial es limitada.
            </div>
            <div class="model-tags">
                <span class="model-tag">max_iter=300</span>
                <span class="model-tag">lr=0.06</span>
                <span class="model-tag">l2=0.1</span>
                <span class="model-tag">stratify</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ────────────────────────────────────────────────────────────────────────────
# Plotly charts (dark biotech style)
# ────────────────────────────────────────────────────────────────────────────

PLOT_BASE = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, sans-serif", color="#c1cce0", size=12),
    margin=dict(l=20, r=20, t=14, b=36),
    height=300,
    hoverlabel=dict(bgcolor="rgba(8,17,33,0.95)", bordercolor="#4f8bff",
                    font=dict(color="#e8eef9", family="JetBrains Mono", size=12)),
)

GRID = "rgba(255,255,255,0.06)"


def fig_gauge(proba: float) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=proba * 100,
        number=dict(suffix="%", font=dict(size=42, color="#e8eef9", family="JetBrains Mono")),
        gauge=dict(
            axis=dict(range=[0, 100], tickwidth=1, tickcolor="rgba(255,255,255,0.2)",
                      tickfont=dict(color="#7a8aa6", size=10)),
            bar=dict(color="rgba(0,0,0,0)", thickness=0.001),
            bgcolor="rgba(255,255,255,0.04)",
            borderwidth=0,
            steps=[
                dict(range=[0, 30], color="rgba(45,212,167,0.55)"),
                dict(range=[30, 70], color="rgba(245,185,66,0.55)"),
                dict(range=[70, 100], color="rgba(239,79,108,0.7)"),
            ],
            threshold=dict(line=dict(color="#22d3ee", width=4), thickness=0.85, value=proba * 100),
        ),
    ))
    layout = {**PLOT_BASE, "height": 240, "margin": dict(l=20, r=20, t=20, b=20)}
    fig.update_layout(**layout)
    return fig


def fig_roc(curves: dict, auc: float) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines",
        line=dict(color="rgba(255,255,255,0.15)", width=1, dash="dash"),
        hoverinfo="skip", showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=curves["fpr"], y=curves["tpr"], mode="lines",
        line=dict(color="#4f8bff", width=3.5, shape="spline"),
        fill="tozeroy", fillcolor="rgba(79,139,255,0.18)",
        hovertemplate="FPR %{x:.2f} · TPR %{y:.2f}<extra></extra>",
        showlegend=False,
    ))
    fig.add_annotation(
        x=0.95, y=0.08, xref="paper", yref="paper",
        text=f"<b>AUC</b><br><span style='font-size:22px;color:#22d3ee;'>{auc:.3f}</span>",
        showarrow=False, align="right",
        font=dict(color="#c1cce0", family="JetBrains Mono", size=11),
    )
    fig.update_layout(**PLOT_BASE)
    fig.update_xaxes(title="False positive rate", gridcolor=GRID, zeroline=False, range=[0, 1])
    fig.update_yaxes(title="True positive rate", gridcolor=GRID, zeroline=False, range=[0, 1])
    return fig


def fig_pr(curves: dict, prevalence: float) -> go.Figure:
    fig = go.Figure()
    fig.add_hline(y=prevalence, line=dict(color="rgba(255,255,255,0.18)", width=1, dash="dash"),
                  annotation_text=f"baseline {prevalence*100:.1f}%",
                  annotation_position="bottom right",
                  annotation_font=dict(color="#7a8aa6", size=10))
    fig.add_trace(go.Scatter(
        x=curves["rec"], y=curves["prec"], mode="lines",
        line=dict(color="#22d3ee", width=3.5, shape="spline"),
        fill="tozeroy", fillcolor="rgba(34,211,238,0.14)",
        hovertemplate="Recall %{x:.2f} · Precision %{y:.2f}<extra></extra>",
        showlegend=False,
    ))
    fig.update_layout(**PLOT_BASE)
    fig.update_xaxes(title="Recall", gridcolor=GRID, zeroline=False, range=[0, 1])
    fig.update_yaxes(title="Precision", gridcolor=GRID, zeroline=False, range=[0, 1])
    return fig


def fig_distribution(test_df: pd.DataFrame) -> go.Figure:
    pos = test_df.loc[test_df["y_true"] == 1, "proba"]
    neg = test_df.loc[test_df["y_true"] == 0, "proba"]
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=neg, xbins=dict(start=0, end=1, size=1/40),
        marker=dict(color="rgba(122,138,166,0.55)", line=dict(width=0)),
        name="No cáncer", opacity=0.85,
    ))
    fig.add_trace(go.Histogram(
        x=pos, xbins=dict(start=0, end=1, size=1/40),
        marker=dict(color="rgba(239,79,108,0.85)", line=dict(width=0)),
        name="Cáncer", opacity=0.9,
    ))
    fig.add_vline(x=THRESHOLDS["medio"], line=dict(color="#f5b942", width=1.2, dash="dot"))
    fig.add_vline(x=THRESHOLDS["alto"], line=dict(color="#ef4f6c", width=1.2, dash="dot"))
    fig.update_layout(
        **PLOT_BASE, barmode="overlay",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0,
                    font=dict(color="#c1cce0", size=11), bgcolor="rgba(0,0,0,0)"),
    )
    fig.update_xaxes(title="Probabilidad predicha", gridcolor=GRID, zeroline=False, range=[0, 1])
    fig.update_yaxes(title="Pacientes", gridcolor=GRID, zeroline=False)
    return fig


def fig_importance(curves: dict) -> go.Figure:
    items = list(reversed(curves["importance"]))
    fig = go.Figure(go.Bar(
        x=[i["imp"] for i in items],
        y=[i["orig"] for i in items],
        orientation="h",
        marker=dict(
            color=[i["imp"] for i in items],
            colorscale=[[0, "#1e3a8a"], [0.5, "#4f8bff"], [1, "#22d3ee"]],
            line=dict(color="rgba(255,255,255,0.06)", width=0.5),
        ),
        hovertemplate="<b>%{y}</b><br>%{x:.3f}<extra></extra>",
    ))
    fig.update_layout(**PLOT_BASE, showlegend=False)
    fig.update_xaxes(title="Importancia agregada", gridcolor=GRID, zeroline=False)
    fig.update_yaxes(gridcolor=GRID, zeroline=False, automargin=True, tickfont=dict(size=11))
    return fig


def fig_heatmap(curves: dict) -> go.Figure:
    hm = curves["heatmap"]
    fig = go.Figure(data=go.Heatmap(
        z=hm.values * 100,
        x=list(hm.columns), y=list(hm.index),
        colorscale=[[0, "#0e1a35"], [0.4, "#4f8bff"], [0.7, "#ef4f6c"], [1, "#ffb4c1"]],
        colorbar=dict(title=dict(text="% riesgo", font=dict(color="#7a8aa6", size=10)),
                      tickfont=dict(color="#7a8aa6", size=10), thickness=10, len=0.8),
        hovertemplate="<b>%{y} · %{x}</b><br>Riesgo medio %{z:.1f}%<extra></extra>",
    ))
    fig.update_layout(**PLOT_BASE)
    fig.update_xaxes(title="Edad", gridcolor=GRID, zeroline=False)
    fig.update_yaxes(title="Zona", gridcolor=GRID, zeroline=False, automargin=True)
    return fig


def fig_radar(curves: dict, values_override: list[float] | None = None) -> go.Figure:
    labels = curves["radar_labels"]
    values = values_override if values_override is not None else curves["radar_values"]
    values_closed = values + [values[0]]
    labels_closed = labels + [labels[0]]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values_closed, theta=labels_closed,
        fill="toself",
        line=dict(color="#22d3ee", width=2.5),
        fillcolor="rgba(34,211,238,0.18)",
        hovertemplate="<b>%{theta}</b><br>P%{r:.0f}<extra></extra>",
        name="Paciente",
    ))
    fig.add_trace(go.Scatterpolar(
        r=[50] * len(labels_closed), theta=labels_closed,
        line=dict(color="rgba(255,255,255,0.18)", width=1, dash="dash"),
        hoverinfo="skip", showlegend=False,
    ))
    radar_layout = {**PLOT_BASE, "margin": dict(l=40, r=40, t=20, b=20)}
    fig.update_layout(
        **radar_layout,
        polar=dict(
            bgcolor="rgba(255,255,255,0.02)",
            radialaxis=dict(visible=True, range=[0, 100], gridcolor=GRID,
                            tickfont=dict(color="#7a8aa6", size=9), tickvals=[25, 50, 75]),
            angularaxis=dict(gridcolor=GRID, tickfont=dict(color="#c1cce0", size=11)),
        ),
        showlegend=False,
    )
    return fig


def fig_cum_incidence(curves: dict) -> go.Figure:
    df = curves["cum_incidence"]
    sample = df.iloc[::max(1, len(df) // 400)]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=sample["edad"], y=sample["cum"] * 100,
        mode="lines",
        line=dict(color="#8b6dff", width=3, shape="spline"),
        fill="tozeroy", fillcolor="rgba(139,109,255,0.14)",
        hovertemplate="Edad %{x} · %{y:.1f}% acumulado<extra></extra>",
        showlegend=False,
    ))
    fig.update_layout(**PLOT_BASE)
    fig.update_xaxes(title="Edad", gridcolor=GRID, zeroline=False)
    fig.update_yaxes(title="Incidencia acumulada (%)", gridcolor=GRID, zeroline=False)
    return fig


def fig_confusion_matrix(curves: dict) -> go.Figure:
    cm = curves["confusion_matrix"]
    tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
    total = tn + fp + fn + tp
    z = [[2, -2], [-1, 1]]  # TP=+2, FN=-2, FP=-1, TN=+1
    text = [
        [f"<b>TP</b><br>{tp:,}<br>{tp/total*100:.1f}%", f"<b>FN</b><br>{fn:,}<br>{fn/total*100:.1f}%"],
        [f"<b>FP</b><br>{fp:,}<br>{fp/total*100:.1f}%", f"<b>TN</b><br>{tn:,}<br>{tn/total*100:.1f}%"],
    ]
    fig = go.Figure(go.Heatmap(
        z=z,
        x=["Pred: Cáncer", "Pred: No cáncer"],
        y=["Real: Cáncer", "Real: No cáncer"],
        text=text,
        texttemplate="%{text}",
        textfont=dict(size=13, color="#e8eef9"),
        colorscale=[
            [0.0, "rgba(239,79,108,0.70)"],
            [0.375, "rgba(245,185,66,0.50)"],
            [0.75, "rgba(79,139,255,0.40)"],
            [1.0, "rgba(45,212,167,0.60)"],
        ],
        showscale=False,
        hovertemplate="%{text}<extra></extra>",
        zmin=-2, zmax=2,
    ))
    layout = {**PLOT_BASE, "height": 280, "margin": dict(l=20, r=20, t=44, b=20)}
    fig.update_layout(**layout)
    fig.update_xaxes(side="top", tickfont=dict(size=11, color="#c1cce0"), gridcolor="rgba(0,0,0,0)")
    fig.update_yaxes(tickfont=dict(size=11, color="#c1cce0"), gridcolor="rgba(0,0,0,0)")
    return fig


def fig_calibration(curves: dict) -> go.Figure:
    mean_pred = curves["cal_mean_pred"]
    frac_pos = curves["cal_frac_pos"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines",
        line=dict(color="rgba(255,255,255,0.18)", width=1.5, dash="dash"),
        hoverinfo="skip", showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=mean_pred, y=frac_pos, mode="lines+markers",
        line=dict(color="#8b6dff", width=3, shape="spline"),
        marker=dict(size=8, color="#8b6dff", line=dict(color="#e8eef9", width=1.5)),
        fill="tozeroy", fillcolor="rgba(139,109,255,0.10)",
        hovertemplate="Pred: %{x:.2f} · Real: %{y:.2f}<extra></extra>",
        showlegend=False,
    ))
    fig.update_layout(**PLOT_BASE)
    fig.update_xaxes(title="Probabilidad media predicha", gridcolor=GRID, zeroline=False, range=[0, 1])
    fig.update_yaxes(title="Fracción positivos reales", gridcolor=GRID, zeroline=False, range=[0, 1])
    return fig


def fig_mutations(curves: dict) -> go.Figure:
    mut_data = curves["mut_by_risk"]
    mutations = list(mut_data.get("Alto", {}).keys())
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=mutations,
        y=[mut_data.get("Alto", {}).get(m, 0) for m in mutations],
        name="Alto riesgo",
        marker=dict(color="rgba(239,79,108,0.80)", line=dict(width=0)),
        hovertemplate="<b>%{x}</b><br>%{y:.1f}% prevalencia<extra>Alto</extra>",
    ))
    fig.add_trace(go.Bar(
        x=mutations,
        y=[mut_data.get("Bajo", {}).get(m, 0) for m in mutations],
        name="Bajo riesgo",
        marker=dict(color="rgba(79,139,255,0.65)", line=dict(width=0)),
        hovertemplate="<b>%{x}</b><br>%{y:.1f}% prevalencia<extra>Bajo</extra>",
    ))
    fig.update_layout(
        **PLOT_BASE,
        barmode="group",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0,
                    font=dict(color="#c1cce0", size=11), bgcolor="rgba(0,0,0,0)"),
    )
    fig.update_xaxes(gridcolor=GRID, zeroline=False, tickfont=dict(size=11))
    fig.update_yaxes(title="Prevalencia (%)", gridcolor=GRID, zeroline=False)
    return fig


_CHART_COUNTER = {"n": 0}

def chart_card(title: str, subtitle: str, fig: go.Figure) -> None:
    _CHART_COUNTER["n"] += 1
    key = f"chartcard_{_CHART_COUNTER['n']}"
    with st.container(key=key):
        st.markdown(
            f'<h3 class="panel-title">{title}</h3>'
            f'<p class="panel-subtitle">{subtitle}</p>',
            unsafe_allow_html=True,
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def render_clinical_msg() -> None:
    st.markdown(
        """
        <div class="clinical-msg">
            <div class="clinical-icon">!</div>
            <div class="clinical-text">
                <strong>Este sistema no sustituye al médico.</strong>
                Prioriza pacientes cuando la capacidad asistencial es limitada.
                Toda decisión diagnóstica corresponde al equipo clínico responsable.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ────────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────────

def main() -> None:
    configure_page()
    inject_css()

    full_df, test_df, metrics, curves = load_and_train()

    render_topbar(metrics)
    render_hero_title()

    # § 01 — Clinical Overview (KPIs)
    render_section_header("§ 01", "Clinical Overview", "Métricas en tiempo real · validación")
    render_kpis(metrics, curves, test_df)

    # § 02 — Predictive Intelligence (picker + hero + priority + model)
    render_section_header("§ 02", "Predictive Intelligence", "Inferencia individual · cola de prioridad")
    selected_pid = render_patient_picker(test_df)
    selected_proba = float(test_df.loc[test_df[ID_COL] == selected_pid, "proba"].iloc[0])
    selected_radar = patient_radar_values(full_df, selected_pid)

    left, right = st.columns([1.6, 1], gap="large")
    with left:
        render_hero(test_df, full_df, selected_pid)
    with right:
        render_priority_panel(test_df)
        st.write("")
        render_model_card(metrics)

    # § 03 — Risk Stratification (gauge + heatmap + distribution)
    render_section_header("§ 03", "Risk Stratification", "Estratificación cohorte · subgrupos")
    cols = st.columns([1, 1.2, 1.4], gap="medium")
    with cols[0]:
        chart_card("Risk gauge",
                   f"Paciente {selected_pid} · pred {selected_proba:.3f}",
                   fig_gauge(selected_proba))
    with cols[1]:
        chart_card("Heatmap zona × edad",
                   "Riesgo medio predicho",
                   fig_heatmap(curves))
    with cols[2]:
        chart_card("Distribución de riesgo",
                   "Cáncer vs no cáncer · umbrales 0.30 / 0.70",
                   fig_distribution(test_df))

    # § 04 — Explainability Engine (importance + radar)
    render_section_header("§ 04", "Explainability Engine", "Variables clave · perfil biomédico")
    cols = st.columns([1.4, 1], gap="medium")
    with cols[0]:
        chart_card("Variables más informativas",
                   "Top 10 features por importancia agregada",
                   fig_importance(curves))
    with cols[1]:
        chart_card("Perfil biomarcadores",
                   f"Percentiles del paciente {selected_pid} vs cohorte",
                   fig_radar(curves, values_override=selected_radar))

    # § 05 — Cohort Analytics (ROC + PR + cumulative)
    render_section_header("§ 05", "Cohort Analytics", "Validación cuantitativa del modelo")
    cols = st.columns(3, gap="medium")
    with cols[0]:
        chart_card("Curva ROC",
                   f"AUC {metrics['auc']:.3f}",
                   fig_roc(curves, metrics["auc"]))
    with cols[1]:
        chart_card("Curva Precision–Recall",
                   f"baseline = {metrics['prevalence']*100:.1f}%",
                   fig_pr(curves, metrics["prevalence"]))
    with cols[2]:
        chart_card("Incidencia acumulada",
                   "Cohorte completa por edad",
                   fig_cum_incidence(curves))

    # § 06 — Clinical Validation (confusion matrix + calibration + mutations)
    render_section_header("§ 06", "Clinical Validation", "Validación clínica · distribución molecular")
    cols = st.columns(3, gap="medium")
    with cols[0]:
        chart_card("Matriz de confusión",
                   "Umbral 0.65 · TP / FP / FN / TN",
                   fig_confusion_matrix(curves))
    with cols[1]:
        chart_card("Curva de calibración",
                   "Probabilidad predicha vs real observada",
                   fig_calibration(curves))
    with cols[2]:
        chart_card("Mutaciones por grupo de riesgo",
                   "Prevalencia (%) en pacientes alto vs bajo",
                   fig_mutations(curves))

    render_clinical_msg()


if __name__ == "__main__":
    main()
