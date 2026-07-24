"""
streamlit_app.py
------------------
Dashboard interaktif untuk memprediksi keterlambatan pengiriman.
Ditujukan untuk tim operasional — cukup isi form, tidak perlu tahu API/kode.

Jalankan dengan:
    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Agar bisa dijalankan langsung via `streamlit run app/streamlit_app.py`
sys.path.append(str(Path(__file__).resolve().parent.parent))

from app.model_utils import (  # noqa: E402
    DEFAULT_NUMERIC_RANGES,
    ModelNotFoundError,
    get_categorical_options,
    get_feature_importance,
    load_config,
    load_pipeline,
    predict_delay,
)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Prediksi Keterlambatan Pengiriman",
    page_icon="🚚",
    layout="wide",
)

LABELS = {
    "distance_km": "Jarak Pengiriman (km)",
    "package_weight_kg": "Berat Paket (kg)",
    "expected_time_hours": "Estimasi Waktu Pengiriman (jam)",
    "delivery_cost": "Biaya Pengiriman (Rp)",
    "delivery_partner": "Partner Logistik",
    "package_type": "Jenis Paket",
    "vehicle_type": "Jenis Kendaraan",
    "delivery_mode": "Moda Pengiriman",
    "region": "Wilayah",
    "weather_condition": "Kondisi Cuaca",
}


@st.cache_resource(show_spinner="Memuat model prediksi...")
def get_pipeline():
    return load_pipeline()


@st.cache_data(show_spinner=False)
def get_config():
    return load_config()


@st.cache_data(show_spinner=False)
def get_options(_pipeline) -> dict[str, list[str]]:
    return get_categorical_options(_pipeline)


def render_probability_gauge(probability: float) -> go.Figure:
    if probability >= 0.7:
        bar_color = "#C62828"
    elif probability >= 0.4:
        bar_color = "#F9A825"
    else:
        bar_color = "#2E7D32"

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=probability * 100,
            number={"suffix": "%", "font": {"size": 40}},
            title={"text": "Probabilitas Terlambat", "font": {"size": 16}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1},
                "bar": {"color": bar_color, "thickness": 0.3},
                "steps": [
                    {"range": [0, 40], "color": "#E8F5E9"},
                    {"range": [40, 70], "color": "#FFF8E1"},
                    {"range": [70, 100], "color": "#FFEBEE"},
                ],
                "threshold": {
                    "line": {"color": "black", "width": 3},
                    "thickness": 0.8,
                    "value": probability * 100,
                },
            },
        )
    )
    fig.update_layout(height=280, margin=dict(l=20, r=20, t=50, b=10))
    return fig


def render_sidebar(config: dict) -> None:
    st.sidebar.header("ℹ️ Tentang Model")
    if config:
        st.sidebar.markdown(f"**Tipe model:** {config.get('model_type', '-')}")
        metrics = config.get("test_metrics", {})
        if metrics:
            st.sidebar.markdown("**Performa pada data test:**")
            st.sidebar.metric("ROC-AUC", f"{metrics.get('roc_auc', 0):.3f}")
            c1, c2 = st.sidebar.columns(2)
            c1.metric("Precision", f"{metrics.get('precision', 0):.2%}")
            c2.metric("Recall", f"{metrics.get('recall', 0):.2%}")
        trained_at = config.get("trained_at")
        if trained_at:
            st.sidebar.caption(f"Terakhir dilatih: {trained_at}")
    else:
        st.sidebar.info("`config.json` tidak ditemukan — menampilkan info dasar saja.")

    st.sidebar.divider()
    st.sidebar.markdown(
        "**Catatan:** Model ini memprediksi risiko keterlambatan **sebelum** "
        "pengiriman berlangsung, menggunakan informasi yang tersedia saat "
        "pesanan dibuat (jarak, berat, moda, cuaca, dll.) — bukan data hasil "
        "akhir pengiriman."
    )


def render_form(pipeline) -> dict | None:
    options = get_options(pipeline)

    with st.form("prediction_form"):
        st.subheader("📦 Detail Pengiriman")

        col1, col2 = st.columns(2)

        with col1:
            distance_km = st.slider(
                LABELS["distance_km"],
                min_value=DEFAULT_NUMERIC_RANGES["distance_km"][0],
                max_value=DEFAULT_NUMERIC_RANGES["distance_km"][1],
                value=DEFAULT_NUMERIC_RANGES["distance_km"][2],
                step=1.0,
            )
            package_weight_kg = st.slider(
                LABELS["package_weight_kg"],
                min_value=DEFAULT_NUMERIC_RANGES["package_weight_kg"][0],
                max_value=DEFAULT_NUMERIC_RANGES["package_weight_kg"][1],
                value=DEFAULT_NUMERIC_RANGES["package_weight_kg"][2],
                step=0.5,
            )
            expected_time_hours = st.slider(
                LABELS["expected_time_hours"],
                min_value=DEFAULT_NUMERIC_RANGES["expected_time_hours"][0],
                max_value=DEFAULT_NUMERIC_RANGES["expected_time_hours"][1],
                value=DEFAULT_NUMERIC_RANGES["expected_time_hours"][2],
                step=1.0,
            )
            delivery_cost = st.number_input(
                LABELS["delivery_cost"],
                min_value=DEFAULT_NUMERIC_RANGES["delivery_cost"][0],
                value=DEFAULT_NUMERIC_RANGES["delivery_cost"][2],
                step=50.0,
            )

        with col2:
            delivery_partner = st.selectbox(
                LABELS["delivery_partner"], options["delivery_partner"]
            )
            package_type = st.selectbox(
                LABELS["package_type"], options["package_type"]
            )
            vehicle_type = st.selectbox(
                LABELS["vehicle_type"], options["vehicle_type"]
            )
            delivery_mode = st.selectbox(
                LABELS["delivery_mode"], options["delivery_mode"]
            )
            region = st.selectbox(LABELS["region"], options["region"])
            weather_condition = st.selectbox(
                LABELS["weather_condition"], options["weather_condition"]
            )

        submitted = st.form_submit_button(
            "🔮 Prediksi Keterlambatan", use_container_width=True, type="primary"
        )

    if not submitted:
        return None

    return {
        "distance_km": distance_km,
        "package_weight_kg": package_weight_kg,
        "expected_time_hours": expected_time_hours,
        "delivery_cost": delivery_cost,
        "delivery_partner": delivery_partner,
        "package_type": package_type,
        "vehicle_type": vehicle_type,
        "delivery_mode": delivery_mode,
        "region": region,
        "weather_condition": weather_condition,
    }


def render_result(pipeline, input_data: dict) -> None:
    try:
        result = predict_delay(pipeline, input_data)
    except ValueError as exc:
        st.error(f"Input tidak valid: {exc}")
        return

    st.divider()
    st.subheader("📊 Hasil Prediksi")

    res_col, gauge_col = st.columns([1, 1])

    with res_col:
        if result.is_delayed:
            st.error(f"### ⚠️ {result.label}")
        else:
            st.success(f"### ✅ {result.label}")

        risk_badge = {
            "High": ("🔴", "Risiko Tinggi"),
            "Medium": ("🟡", "Risiko Sedang"),
            "Low": ("🟢", "Risiko Rendah"),
        }[result.risk_level]
        st.metric("Tingkat Risiko", f"{risk_badge[0]} {risk_badge[1]}")
        st.metric("Probabilitas Terlambat", f"{result.probability:.1%}")

        st.markdown("**Rekomendasi tindakan operasional:**")
        if result.risk_level == "High":
            st.markdown(
                "- Pertimbangkan penambahan buffer waktu SLA\n"
                "- Kirim notifikasi proaktif ke pelanggan\n"
                "- Evaluasi rute/moda alternatif jika memungkinkan"
            )
        elif result.risk_level == "Medium":
            st.markdown(
                "- Pantau status pengiriman lebih ketat dari biasanya\n"
                "- Siapkan komunikasi cadangan ke pelanggan"
            )
        else:
            st.markdown("- Proses normal, tidak perlu mitigasi tambahan.")

    with gauge_col:
        st.plotly_chart(
            render_probability_gauge(result.probability),
            use_container_width=True,
        )

    with st.expander("Lihat data input yang dikirim ke model"):
        st.dataframe(pd.DataFrame([input_data]), use_container_width=True)


def render_feature_importance(pipeline) -> None:
    fi = get_feature_importance(pipeline, top_n=10)
    if fi is None or fi.empty:
        return
    with st.expander("📈 Faktor paling berpengaruh terhadap prediksi (model-level)"):
        st.bar_chart(fi.sort_values())
        st.caption(
            "Menunjukkan fitur mana yang paling memengaruhi prediksi model "
            "secara umum (bukan spesifik untuk input di atas)."
        )


def main() -> None:
    st.title("🚚 Prediksi Keterlambatan Pengiriman")
    st.caption(
        "Isi detail pengiriman di bawah untuk melihat prediksi dan probabilitas "
        "keterlambatan secara instan — tanpa perlu memanggil API."
    )

    try:
        pipeline = get_pipeline()
    except ModelNotFoundError as exc:
        st.warning(str(exc))
        st.info(
            "Jalankan `python scripts/train_model.py --data <path_csv> --out model/` "
            "atau salin `model.pkl` hasil training dari notebook ke folder `model/`."
        )
        st.stop()

    config = get_config()
    render_sidebar(config)

    input_data = render_form(pipeline)
    if input_data is not None:
        render_result(pipeline, input_data)

    render_feature_importance(pipeline)


if __name__ == "__main__":
    main()
