import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

# =========================================
# CONFIG
# =========================================
st.set_page_config(page_title="Solemne 2 | Bank Marketing - Clasificador", layout="wide")

DATA_PATH = "bank-additional-full.csv"
MODEL_PATH = "model.joblib"

POS_LABEL = "yes"   # clase positiva del negocio
NEG_LABEL = "no"

# Variables que deben ser enteras (conteos)
INT_FEATURES = {"age", "campaign", "pdays", "previous"}

# =========================================
# TRAIN / LOAD
# =========================================
@st.cache_resource
def train_and_save_model(data_path: str = DATA_PATH, model_path: str = MODEL_PATH):
    """
    Entrena desde cero un modelo de clasificación para predecir y (yes/no)
    y guarda un payload completo en model.joblib (pipeline + metadata + métricas).
    """
    df = pd.read_csv(data_path, sep=";")

    # Evitar data leakage: duration se conoce después de la llamada
    if "duration" in df.columns:
        df = df.drop(columns=["duration"])

    X = df.drop(columns=["y"])
    y = df["y"].map({NEG_LABEL: 0, POS_LABEL: 1}).astype(int)

    cat_cols = [c for c in X.columns if X[c].dtype == "object"]
    num_cols = [c for c in X.columns if c not in cat_cols]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    preprocess = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("scaler", StandardScaler())]), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ]
    )

    # Modelo simple + explicable (ideal para el curso) y robusto ante desbalance
    model = LogisticRegression(max_iter=2000, class_weight="balanced", solver="liblinear")
    pipeline = Pipeline([("preprocess", preprocess), ("model", model)])
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
    }
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

    payload = {
        "pipeline": pipeline,
        "feature_names": list(X.columns),
        "categorical_cols": cat_cols,
        "numeric_cols": num_cols,
        "label_map": {0: NEG_LABEL, 1: POS_LABEL},
        "test_metrics": metrics,
        "test_confusion_matrix": cm.tolist(),
        "test_report": report,
        "domain_options": {c: sorted(X[c].astype(str).unique().tolist()) for c in cat_cols},
        "numeric_ranges": {
            c: {
                "min": float(X[c].min()),
                "max": float(X[c].max()),
                "median": float(X[c].median()),
            }
            for c in num_cols
        },
        "notes": {
            "leakage_avoidance": "Se excluye duration para evitar data leakage.",
            "split": "train/test = 80/20 stratified (random_state=42).",
            "model": "LogisticRegression + OneHotEncoder + StandardScaler (class_weight='balanced').",
        },
    }

    joblib.dump(payload, model_path)
    return payload


@st.cache_resource
def load_or_train():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    return train_and_save_model(DATA_PATH, MODEL_PATH)


# =========================================
# SIDEBAR (profe-friendly)
# =========================================
st.sidebar.title("Solemne 2 (40%)")
st.sidebar.caption("Taller de Aplicaciones – Bank Marketing")

st.sidebar.markdown(
    """
**Objetivo:** Publicar los datos y resultados del clasificador desarrollado para predecir  
si un cliente se suscribirá a un depósito a plazo (**y: yes/no**).

**Requisitos (Streamlit):**
- Mostrar métricas: accuracy, precision, recall, F1-score
- Mostrar matriz de confusión
- Permitir ingresar variables y probar el modelo
- Mostrar predicción final (yes/no) + probabilidad
"""
)

# =========================================
# VALIDACIONES BÁSICAS
# =========================================
if not os.path.exists(DATA_PATH):
    st.error(
        f"No se encontró `{DATA_PATH}` en el proyecto.\n\n"
        "✅ Solución: sube `bank-additional-full.csv` al repo (misma carpeta que `app.py`)."
    )
    st.stop()

payload = load_or_train()
pipe = payload["pipeline"]

# =========================================
# MAIN
# =========================================
st.title("📞 Bank Marketing – Clasificador de Suscripción (y)")

with st.expander("ℹ️ Contexto (CRISP-DM)", expanded=False):
    st.markdown(
        f"""
**Business Understanding:** predecir suscripción a depósito a plazo para mejorar focalización de campañas.  
**Data Understanding:** dataset `bank-additional-full` (UCI).  
**Evaluation:** métricas sobre conjunto de prueba (*test 20%*): accuracy, precision, recall, F1 + matriz de confusión.

**Target:** `y` → {POS_LABEL} (se suscribe), {NEG_LABEL} (no se suscribe)

**Nota:** se excluye `duration` para evitar *data leakage* (la duración se conoce después de la llamada).
"""
    )

tab_eval, tab_pred = st.tabs(["📊 Resultados del clasificador", "🧪 Probar el modelo"])

# =========================================
# TAB 1: Evaluation
# =========================================
with tab_eval:
    st.subheader("Resultados en conjunto de prueba (test 20%)")

    m = payload.get("test_metrics", {})
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy", f"{m.get('accuracy', 0):.3f}")
    c2.metric("Precision (yes)", f"{m.get('precision', 0):.3f}")
    c3.metric("Recall (yes)", f"{m.get('recall', 0):.3f}")
    c4.metric("F1-score (yes)", f"{m.get('f1', 0):.3f}")

    left, right = st.columns([1.0, 1.0], gap="large")

    with left:
        st.markdown("### Matriz de confusión")
        st.caption("Filas = Real, Columnas = Predicción. Clase positiva = **yes**.")
        cm = np.array(payload.get("test_confusion_matrix", [[0, 0], [0, 0]]))

        fig, ax = plt.subplots()
        ax.imshow(cm)
        ax.set_xlabel("Predicción")
        ax.set_ylabel("Real")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels([f"{NEG_LABEL} (0)", f"{POS_LABEL} (1)"])
        ax.set_yticklabels([f"{NEG_LABEL} (0)", f"{POS_LABEL} (1)"])

        for i in range(2):
            for j in range(2):
                ax.text(j, i, int(cm[i, j]), ha="center", va="center")

        st.pyplot(fig, clear_figure=True)

    with right:
        st.markdown("### Classification report")
        report = payload.get("test_report")
        if report:
            rep_df = (
                pd.DataFrame(report).T
                .rename(columns={"f1-score": "f1_score"})
                .loc[["0", "1", "accuracy", "macro avg", "weighted avg"]]
            )
            st.dataframe(rep_df, use_container_width=True)
        else:
            st.info("No se encontró `test_report` en el modelo. Reentrena para generarlo.")

    with st.expander("🔁 Reentrenar modelo (opcional)", expanded=False):
        st.write(
            "Si actualizaste el dataset en el repo, puedes reentrenar y sobrescribir `model.joblib`."
        )
        if st.button("Reentrenar y guardar model.joblib"):
            payload = train_and_save_model(DATA_PATH, MODEL_PATH)
            st.success("Listo: modelo reentrenado y guardado. Recarga la página para ver cambios.")

# =========================================
# TAB 2: Prediction form
# =========================================
with tab_pred:
    st.subheader("Ingresar variables del cliente y probar el modelo")

    domain_opts = payload.get("domain_options", {})
    num_ranges = payload.get("numeric_ranges", {})
    feature_names = payload["feature_names"]

    with st.form("predict_form", clear_on_submit=False):
        st.markdown("#### Variables categóricas")
        user_input = {}

        for c in payload["categorical_cols"]:
            options = domain_opts.get(c, [])
            if len(options) == 0:
                options = ["unknown"]
            user_input[c] = st.selectbox(c, options=options, index=0)

        st.markdown("#### Variables numéricas")
        for c in payload["numeric_cols"]:
            r = num_ranges.get(c, {"min": 0.0, "max": 1.0, "median": 0.0})

            # Enteros donde corresponde
            if c in INT_FEATURES:
                min_v = int(np.floor(r["min"]))
                max_v = int(np.ceil(r["max"]))
                val = int(round(r["median"]))
                user_input[c] = st.number_input(
                    c,
                    min_value=min_v,
                    max_value=max_v,
                    value=val,
                    step=1,
                )
            else:
                # Decimales (macroeconómicas)
                min_v = float(r["min"])
                max_v = float(r["max"])
                val = float(r["median"])

                # step razonable para floats
                span = max_v - min_v
                step = float(span / 200.0) if span > 0 else 0.1
                step = max(step, 0.01)

                user_input[c] = st.number_input(
                    c,
                    min_value=min_v,
                    max_value=max_v,
                    value=val,
                    step=step,
                    format="%.4f",
                )

        st.divider()
        submitted = st.form_submit_button("✅ Predecir")

    if submitted:
        try:
            input_df = pd.DataFrame([user_input])[feature_names]
            proba_yes = float(pipe.predict_proba(input_df)[0, 1])
            pred = int(pipe.predict(input_df)[0])
            label = payload["label_map"][pred]
    
            st.markdown("### Resultado")
            st.write(f"**Predicción final:** `y = {label}`")
            st.write(f"**Probabilidad asociada (yes): {proba_yes:.3f}**")
    
            # Frase interpretativa
            if proba_yes >= 0.70:
                st.success("🟢 Alta probabilidad de suscripción al depósito a plazo.")
            elif proba_yes >= 0.40:
                st.warning("🟡 Probabilidad media de suscripción. Depende de condiciones adicionales.")
            else:
                st.error("🔴 Baja probabilidad de suscripción al depósito a plazo.")
    
            st.progress(min(max(proba_yes, 0.0), 1.0))
            st.caption("La barra representa la probabilidad estimada de suscripción (yes).")
    
        except Exception as e:
            st.error(
                "Ocurrió un error al predecir. Revisa que el dataset y el modelo correspondan a las mismas columnas."
            )
            st.exception(e)


st.caption(
    "Modelo cargado desde `model.joblib` y usado por la app. (Se excluye `duration` para evitar data leakage)."
)
