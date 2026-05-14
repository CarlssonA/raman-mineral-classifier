import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import joblib
import io

from src.preprocessing import full_pipeline
from src.features import extract_peak_features

# --- Seitenkonfiguration ---
st.set_page_config(
    page_title="Raman Mineral Classifier",
    page_icon="🔬",
    layout="wide"
)

# --- Modell laden (einmalig, gecacht) ---
@st.cache_resource
def load_model():
    """Lädt Modell und Encoder beim ersten App-Start (gecacht für Performance)."""
    models_dir = Path(__file__).parent.parent / "models"
    try:
        rf_model = joblib.load(models_dir / "rf_model.pkl")
        encoder = joblib.load(models_dir / "label_encoder.pkl")
        target_wn = np.load(models_dir / "target_wavenumbers.npy")
        return rf_model, encoder, target_wn
    except FileNotFoundError:
        return None, None, None


# --- Beispiel-Spektrum erzeugen (für Demo ohne eigene Datei) ---
def make_demo_spectrum() -> pd.DataFrame:
    """
    Erzeugt ein synthetisches Quartz-ähnliches Spektrum als Demo.
    In einer echten Messung würde diese CSV von einem Raman-Spektrometer kommen.
    """
    wn = np.linspace(200, 3500, 500)
    # Synthetische Peaks bei typischen Quartz-Positionen (~465, ~206, ~128 cm⁻¹)
    spectrum = np.random.normal(0.02, 0.01, len(wn))
    for pos, amp, width in [(465, 1.0, 8), (206, 0.3, 10), (128, 0.2, 12), (1085, 0.05, 15)]:
        spectrum += amp * np.exp(-0.5 * ((wn - pos) / width) ** 2)
    # Leichte Fluoreszenz-Baseline simulieren
    spectrum += 0.05 * np.exp(-(wn - 200) / 1000)
    spectrum = np.clip(spectrum, 0, None)
    return pd.DataFrame({"wavenumber": wn, "intensity": spectrum})


# =============================================================================
# App-Layout
# =============================================================================

st.title("🔬 Raman Mineral Classifier")
st.markdown(
    "Lade ein Raman-Spektrum als CSV hoch und erhalte eine automatische Mineral-Vorhersage "
    "— basierend auf dem [RRUFF-Datensatz](https://rruff.info) und einem Random Forest Klassifikator."
)

# Sidebar mit Infos
with st.sidebar:
    st.header("ℹ️ Über dieses Projekt")
    st.markdown("""
    **Raman Mineral Classifier**
    M.Sc. Nanoscience Portfolio-Projekt

    **Modell:** Random Forest (200 Bäume)
    **Training:** RRUFF `fair_oriented`
    **Preprocessing:** SNIP Baseline + Min-Max-Normierung

    **CSV-Format:**
    ```
    wavenumber,intensity
    200.0,0.012
    201.5,0.015
    ...
    ```

    ---
    **Tech Stack:**
    Python · scikit-learn · SciPy · Streamlit · Plotly
    """)

    st.markdown("---")
    demo_btn = st.button("📊 Demo-Spektrum laden", use_container_width=True)

# Modell laden
rf_model, encoder, target_wn = load_model()

if rf_model is None:
    st.warning(
        "⚠️ Kein trainiertes Modell gefunden (`models/rf_model.pkl`).  \n"
        "Bitte zuerst `notebooks/03_modeling.ipynb` ausführen."
    )
    st.stop()

# Datei-Upload
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📁 Spektrum hochladen")

    uploaded_file = st.file_uploader(
        "CSV-Datei mit Spalten: wavenumber, intensity",
        type=["csv"],
        help="Das Spektrum wird genau so preprocessed wie die Trainingsdaten."
    )

    # Demo-Spektrum oder Upload
    if demo_btn or uploaded_file is None:
        demo_df = make_demo_spectrum()
        if demo_btn:
            st.info("Demo-Spektrum geladen (synthetisches Quartz-ähnliches Spektrum).")
        input_df = demo_df
    else:
        try:
            input_df = pd.read_csv(uploaded_file)
            if "wavenumber" not in input_df.columns or "intensity" not in input_df.columns:
                st.error("❌ CSV muss Spalten 'wavenumber' und 'intensity' enthalten.")
                st.stop()
            st.success(f"✅ Datei geladen: {len(input_df)} Datenpunkte")
        except Exception as e:
            st.error(f"Fehler beim Lesen: {e}")
            st.stop()

    # Rohspektrum anzeigen
    fig_raw = go.Figure()
    fig_raw.add_trace(go.Scatter(
        x=input_df["wavenumber"],
        y=input_df["intensity"],
        mode="lines",
        line=dict(color="steelblue", width=1.5),
        name="Rohspektrum"
    ))
    fig_raw.update_layout(
        title="Rohspektrum (vor Preprocessing)",
        xaxis_title="Wellenzahl (cm⁻¹)",
        yaxis_title="Intensität (a.u.)",
        height=350,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    st.plotly_chart(fig_raw, use_container_width=True)

with col2:
    st.subheader("🔍 Vorhersage")

    # Preprocessing
    with st.spinner("Preprocessing läuft..."):
        wn_raw = input_df["wavenumber"].values
        intensity_raw = input_df["intensity"].values

        # Auf 2D bringen (Pipeline erwartet batch)
        wn_2d = np.tile(wn_raw, (1, 1))
        int_2d = intensity_raw.reshape(1, -1)

        try:
            proc_wn, X_proc = full_pipeline(wn_2d, int_2d, target_wn=target_wn)
            X_feat = X_proc
        except Exception as e:
            st.error(f"Preprocessing-Fehler: {e}")
            st.stop()

    # Vorhersage
    proba = rf_model.predict_proba(X_feat)[0]
    top5_idx = np.argsort(proba)[::-1][:5]
    top5_minerals = encoder.classes_[top5_idx]
    top5_proba = proba[top5_idx]

    best_mineral = top5_minerals[0]
    best_conf = top5_proba[0]

    # Ergebnis-Anzeige
    conf_color = "green" if best_conf > 0.7 else ("orange" if best_conf > 0.4 else "red")
    st.markdown(f"""
    <div style='background: #f0f2f6; padding: 20px; border-radius: 10px; text-align: center; margin-bottom: 20px;'>
        <h2 style='color: #1f77b4; margin: 0;'>{best_mineral}</h2>
        <p style='font-size: 18px; color: {conf_color}; margin: 5px 0;'>
            Konfidenz: <strong>{best_conf:.1%}</strong>
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Top-5-Balkendiagramm
    fig_pred = go.Figure(go.Bar(
        x=top5_proba * 100,
        y=top5_minerals,
        orientation="h",
        marker_color=["#1f77b4" if i == 0 else "#aec7e8" for i in range(5)],
        text=[f"{p:.1f}%" for p in top5_proba * 100],
        textposition="outside"
    ))
    fig_pred.update_layout(
        title="Top-5 Vorhersagen",
        xaxis_title="Konfidenz (%)",
        xaxis=dict(range=[0, 110]),
        yaxis=dict(autorange="reversed"),
        height=280,
        margin=dict(l=20, r=40, t=40, b=20)
    )
    st.plotly_chart(fig_pred, use_container_width=True)

# Preprocessed Spektrum anzeigen
st.markdown("---")
st.subheader("⚙️ Vorverarbeitetes Spektrum (Input ins Modell)")

fig_proc = go.Figure()
fig_proc.add_trace(go.Scatter(
    x=proc_wn,
    y=X_proc[0],
    mode="lines",
    line=dict(color="darkorange", width=1.5),
    name="Preprocessed"
))
fig_proc.update_layout(
    xaxis_title="Wellenzahl (cm⁻¹)",
    yaxis_title="Normierte Intensität [0, 1]",
    height=300,
    margin=dict(l=20, r=20, t=20, b=20),
    showlegend=False
)
st.plotly_chart(fig_proc, use_container_width=True)

st.caption(
    "Preprocessing-Pipeline: Resampling auf gemeinsame Wellenzahl-Achse → "
    "SNIP Baseline-Korrektur → Min-Max-Normierung auf [0,1]"
)

# CSV-Download (Demo)
st.markdown("---")
st.subheader("💾 Eigenes Spektrum als CSV speichern")
with st.expander("CSV-Vorlage herunterladen"):
    demo_csv = make_demo_spectrum().to_csv(index=False)
    st.download_button(
        label="📥 Demo-CSV herunterladen",
        data=demo_csv,
        file_name="example_spectrum.csv",
        mime="text/csv"
    )
    st.code("wavenumber,intensity\n200.0,0.012\n201.5,0.015\n...", language="csv")