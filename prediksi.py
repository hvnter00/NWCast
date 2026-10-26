from __future__ import annotations
from datetime import datetime
from decimal import Decimal
from html import escape
import json
from pathlib import Path
from zoneinfo import ZoneInfo

from prediction_engine import load_bundle, predict_latest, resolve_model_path
from prediction_io import make_template, read_upload, validate_and_prepare
from prediction_report import DISCLAIMER_TEXT, build_prediction_pdf

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
GUIDE_PATH = BASE_DIR / "Assets" / "Manual_Guide.pdf"
UP = "#15803d"
DOWN = "#dc2626"
BLUE = "#2563eb"


@st.cache_resource(show_spinner=False)
def _cached_models(path: str, modified_ns: int, size: int) -> tuple[dict, dict]:
    return load_bundle(path)


@st.cache_data(show_spinner=False)
def _template_bytes(format: str, date_values: tuple) -> bytes:
    """Membuat template input prediksi."""
    return make_template(format=format, dates=date_values)


def _manual_frame(last_date) -> pd.DataFrame:
    dates = pd.bdate_range(end=pd.Timestamp(last_date), periods=31)[::-1]
    frame = pd.DataFrame({"Date": dates.date})
    for column in ("Coal", "Oil", "Gas", "DXY"):
        frame[column] = pd.Series("", index=frame.index, dtype="string")
    return frame


def _manual_comma_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Keep manual prices as comma-decimal text without rounding their values."""
    result = frame.copy()
    for column in ("Coal", "Oil", "Gas", "DXY"):
        def display(value):
            if pd.isna(value):
                return ""
            if isinstance(value, str):
                return value.replace(".", ",")
            return format(Decimal(str(value)), "f").replace(".", ",")

        result[column] = result[column].map(display).astype("string")
    return result


def _store_upload() -> None:
    uploaded = st.session_state.get("prediction_upload_widget")
    if uploaded is None:
        st.session_state.pop("prediction_upload_bytes", None)
        st.session_state.pop("prediction_upload_name", None)
        return
    st.session_state["prediction_upload_bytes"] = uploaded.getvalue()
    st.session_state["prediction_upload_name"] = uploaded.name
    st.session_state["prediction_input_mode"] = "File Excel"
    st.session_state["prediction_input_mode_widget"] = "File Excel"


def _store_input_mode() -> None:
    st.session_state["prediction_input_mode"] = st.session_state["prediction_input_mode_widget"]


def _date_label(value) -> str:
    return pd.Timestamp(value).strftime("%d %B %Y")


def _change_horizon(horizon: int) -> None:
    st.session_state["prediction_selected_horizon"] = horizon


def _return_to_input() -> None:
    st.session_state["prediction_view"] = "input"


def _metrics_for_snapshot(snapshot: dict) -> dict | None:
    identity = snapshot.get("model_identity", {})
    filename = identity.get("filename", "")
    if Path(filename).name != filename or not filename:
        return None
    metrics_path = BASE_DIR / "Assets" / f"{Path(filename).stem}_metrics.json"
    try:
        metrics = json.loads(metrics_path.read_text(encoding="utf-8-sig"))
        if metrics.get("model_sha256") != identity.get("sha256"):
            return None
        return metrics
    except (OSError, ValueError, TypeError):
        return None


def _render_input() -> None:
    for key in ("prediction_input_mode", "prediction_input_mode_widget"):
        if st.session_state.get(key) == "File Excel / CSV":
            st.session_state[key] = "File Excel"
    if Path(st.session_state.get("prediction_upload_name", "")).suffix.lower() == ".csv":
        for key in ("prediction_upload_bytes", "prediction_upload_name", "prediction_upload_widget"):
            st.session_state.pop(key, None)
    st.html('<h1 class="prediction-title">Newcastle Coal Price Change Prediction</h1>')
    _render_disclaimer()
    resources = None
    try:
        path = resolve_model_path(BASE_DIR / "Models")
        stamp = path.stat()
        resources = _cached_models(str(path), stamp.st_mtime_ns, stamp.st_size)
        bundle, _ = resources
        version = bundle["metadata"].get("version", path.stem)
    except (OSError, ValueError) as error:
        st.error(str(error))

    today = datetime.now(ZoneInfo("Asia/Jakarta")).date()
    default_end = pd.bdate_range(end=pd.Timestamp(today), periods=1)[0].date()
    if "prediction_manual_data" not in st.session_state:
        st.session_state["prediction_manual_data"] = _manual_frame(default_end)
        st.session_state["prediction_manual_last_date"] = default_end
    if st.session_state.get("prediction_manual_decimal_format") != "comma":
        st.session_state["prediction_manual_data"] = _manual_comma_frame(
            st.session_state["prediction_manual_data"]
        )
        st.session_state.pop("prediction_manual_editor", None)
        st.session_state["prediction_manual_decimal_format"] = "comma"

    left, right = st.columns([1.55, 1], gap="large")
    with left:
        st.html('<h2 class="prediction-section-title">Manual Input</h2>')
        last_date = st.date_input(
            "Tanggal pengamatan terakhir",
            value=st.session_state["prediction_manual_last_date"],
            key="prediction_manual_date_widget",
            format="DD/MM/YYYY",
        )
        if last_date is None:
            st.warning("Pilih tanggal terakhir.")
            st.stop()
        if last_date != st.session_state["prediction_manual_last_date"]:
            st.session_state["prediction_manual_last_date"] = last_date
            st.session_state["prediction_manual_data"] = _manual_frame(last_date)
            st.session_state.pop("prediction_manual_editor", None)

        edited = st.data_editor(
            st.session_state["prediction_manual_data"],
            key="prediction_manual_editor",
            hide_index=True,
            width="stretch",
            height=610,
            num_rows="fixed",
            row_height=32,
            column_config={
                "Date": st.column_config.DateColumn("DATE", format="DD/MM/YYYY", required=True),
                "Coal": st.column_config.TextColumn(
                    "COAL", alignment="right", required=False,
                    help="Gunakan koma untuk desimal, tanpa pemisah ribuan.",
                    validate=r"^(?:[0-9]+(?:,[0-9]+)?)?$",
                ),
                "Oil": st.column_config.TextColumn(
                    "OIL", alignment="right", required=False,
                    help="Gunakan koma untuk desimal, tanpa pemisah ribuan.",
                    validate=r"^(?:[0-9]+(?:,[0-9]+)?)?$",
                ),
                "Gas": st.column_config.TextColumn(
                    "GAS", alignment="right", required=False,
                    help="Gunakan koma untuk desimal, tanpa pemisah ribuan.",
                    validate=r"^(?:[0-9]+(?:,[0-9]+)?)?$",
                ),
                "DXY": st.column_config.TextColumn(
                    "DXY", alignment="right", required=False,
                    help="Gunakan koma untuk desimal, tanpa pemisah ribuan.",
                    validate=r"^(?:[0-9]+(?:,[0-9]+)?)?$",
                ),
            },
        )
        st.session_state["prediction_manual_data"] = edited.copy()
        st.caption("Gunakan koma untuk desimal, misalnya 146,48 atau 3,125. Tanpa pemisah ribuan.")

    with right:
        st.html('<h2 class="prediction-section-title">Upload Dataset</h2>')
        st.file_uploader(
            "File gabungan",
            type=["xlsx"],
            max_upload_size=15,
            accept_multiple_files=False,
            key="prediction_upload_widget",
            on_change=_store_upload,
            label_visibility="collapsed",
        )
        st.caption("Minimal 31 tanggal hari kerja. Harga kosong memakai harga sebelumnya. Lengkapi harga pada tanggal paling awal.")
        if st.session_state.get("prediction_upload_name"):
            st.caption(f"File dipilih: {st.session_state['prediction_upload_name']}")
        dates = tuple(pd.bdate_range(end=pd.Timestamp(last_date), periods=31)[::-1].date)
        excel_column, guide_column = st.columns(2)
        with excel_column:
            st.download_button(
                "Template Excel",
                data=_template_bytes("xlsx", dates),
                file_name="newcastle_input_template.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key="prediction_template_excel",
                width="stretch",
            )
        with guide_column:
            guide_available = GUIDE_PATH.is_file()
            st.download_button(
                "Manual Guide",
                data=GUIDE_PATH.read_bytes() if guide_available else b"",
                file_name="Manual_Guide.pdf",
                mime="application/pdf",
                disabled=not guide_available,
                key="prediction_manual_guide",
                width="stretch",
            )
            if not guide_available:
                st.caption("Panduan segera tersedia.")

        if "prediction_input_mode_widget" not in st.session_state:
            st.session_state["prediction_input_mode_widget"] = st.session_state.get(
                "prediction_input_mode", "Tabel manual"
            )
        st.radio(
            "Sumber data prediksi",
            ["Tabel manual", "File Excel"],
            key="prediction_input_mode_widget",
            on_change=_store_input_mode,
        )
        st.session_state["prediction_input_mode"] = st.session_state["prediction_input_mode_widget"]

        if st.button(
            "MAKE PREDICTIONS",
            key="prediction_make",
            type="primary",
            width="stretch",
            disabled=resources is None,
        ):
            try:
                if st.session_state["prediction_input_mode"] == "File Excel":
                    payload = st.session_state.get("prediction_upload_bytes")
                    filename = st.session_state.get("prediction_upload_name", "")
                    if not payload:
                        raise ValueError("Unggah file terlebih dahulu.")
                    if Path(filename).suffix.lower() != ".xlsx":
                        raise ValueError("Gunakan file Excel (.xlsx).")
                    frame = read_upload(payload, filename)
                    source_name = filename
                else:
                    if pd.Timestamp(last_date).dayofweek >= 5:
                        raise ValueError("Gunakan tanggal Senin–Jumat.")
                    frame = edited.copy()
                    frame["Date"] = pd.to_datetime(frame["Date"], errors="raise")
                    if frame["Date"].isna().any():
                        raise ValueError("Lengkapi tanggal pada tabel.")
                    if (frame["Date"].dt.dayofweek >= 5).any():
                        raise ValueError("Gunakan tanggal Senin–Jumat.")
                    source_name = "Tabel manual"
                with st.spinner("Mengolah data dan menghitung tiga horizon..."):
                    prices = validate_and_prepare(frame)
                    bundle, models = resources
                    snapshot = predict_latest(prices, bundle, models)
                    snapshot["filled_weekdays"] += prices.attrs.get("filled_price_rows", 0)
                    snapshot["history"] = prices.loc[:, ["coal"]].copy()
                    snapshot["input_source"] = source_name
                    st.session_state["prediction_result"] = snapshot
                    st.session_state["prediction_selected_horizon"] = 1
                    st.session_state["prediction_view"] = "results"
                st.rerun()
            except (OSError, ValueError, TypeError, OverflowError) as error:
                st.error(str(error))


def _prediction_chart(snapshot: dict, selected: dict) -> alt.LayerChart:
    history = snapshot["history"].tail(60).reset_index()
    history.columns = ["Date", "Price"]
    history["Series"] = "Harga historis"
    future = pd.DataFrame({
        "Date": pd.to_datetime([snapshot["origin_date"], selected["forecast_date"]]),
        "Price": [snapshot["current_price"], selected["predicted_price"]],
        "Series": [f"Prediksi H+{selected['h']}"] * 2,
    })
    trend_color = UP if selected["change_usd"] >= 0 else DOWN
    historical_line = alt.Chart(history).mark_line(color=BLUE, strokeWidth=2.5)
    predicted_line = alt.Chart(future).mark_line(
        color=trend_color,
        strokeWidth=3,
        point=alt.OverlayMarkDef(filled=True, size=65),
    )
    return (
        alt.layer(historical_line, predicted_line)
        .encode(
            x=alt.X("Date:T", title=None, axis=alt.Axis(format="%d %b", tickCount=6, labelAngle=0)),
            y=alt.Y("Price:Q", title="USD/tonne", scale=alt.Scale(zero=False)),
            tooltip=[
                alt.Tooltip("Date:T", title="Tanggal", format="%d %b %Y"),
                alt.Tooltip("Price:Q", title="USD/tonne", format=",.2f"),
                alt.Tooltip("Series:N", title="Seri"),
            ],
        )
        .properties(height=280, background="#ffffff")
        .configure_view(stroke=None)
        .configure_axis(
            labelColor="#475569", titleColor="#334155", labelFont="Arial", titleFont="Arial",
            labelFontSize=11, titleFontSize=11, gridColor="#e2e8f0", gridOpacity=0.8,
            domainColor="#cbd5e1", tickColor="#cbd5e1",
        )
        .interactive()
    )


def _metric_label(value, digits: int) -> str:
    return f"{value:.{digits}f}" if value is not None and np.isfinite(value) else "—"


def _render_metrics(snapshot: dict, selected: dict) -> None:
    metrics = _metrics_for_snapshot(snapshot)
    st.html('<h2 class="prediction-section-title">Evaluasi Historis</h2>')
    if metrics is None:
        st.info("Metrik pengujian historis untuk versi model ini belum tersedia.")
    else:
        evaluation = metrics["evaluation"]
        if evaluation.get("kind") == "historical_holdout_before_full_refit":
            st.caption("Hasil pengujian kandidat sebelum model dilatih final pada seluruh data.")
        st.caption(
            f"{evaluation['n']} pengamatan pengujian · "
            f"{_date_label(evaluation['origin_start'])} – {_date_label(evaluation['origin_end'])}"
        )
        rows = []
        for name, scores in metrics["models"].items():
            if int(scores["horizon"]) != selected["h"]:
                continue
            active = name == selected["model_name"]
            badge = ' <span class="prediction-model-badge">Terpilih</span>' if active else ""
            rows.append(
                f'<tr><td>{escape(scores["family"])}{badge}</td>'
                f'<td>{_metric_label(scores["rmse_usd"], 3)}</td>'
                f'<td>{_metric_label(scores["mae_usd"], 3)}</td>'
                f'<td>{_metric_label(scores["r2_change"], 4)}</td>'
                f'<td>{_metric_label(scores["da_nonzero_pct"], 1)}{"%" if scores["da_nonzero_pct"] is not None else ""}</td></tr>'
            )
        st.html(
            '<div class="prediction-evaluation"><table><thead><tr>'
            '<th>Model</th><th>RMSE</th><th>MAE</th><th>R²</th><th>DA</th>'
            '</tr></thead><tbody>' + "".join(rows) + '</tbody></table></div>'
        )
        st.caption(
            "RMSE dan MAE dalam USD/tonne perubahan harga. R² dihitung pada perubahan harga. "
            "DA mengevaluasi arah perubahan aktual yang tidak nol."
        )

    if selected.get("cv_rmse") is not None:
        st.html(
            '<div class="prediction-cv-score">'
            f'RMSE CV model terpilih: <strong>{selected["cv_rmse"]:.4f}</strong>'
        )


def _render_disclaimer() -> None:
    st.html(
        '<div class="prediction-disclaimer"><strong>Disclaimer - Not a Financial Advice</strong><br>'
        + escape(DISCLAIMER_TEXT) + '</div>'
    )


@st.cache_data(show_spinner=False)
def _result_pdf(snapshot: dict, report_stamp: tuple) -> bytes:
    return build_prediction_pdf(snapshot)


def _render_results() -> None:
    snapshot = st.session_state["prediction_result"]
    st.html('<h1 class="prediction-title">Price Change Prediction Results</h1>')
    _render_disclaimer()
    st.caption(
        f"Tanggal acuan: {_date_label(snapshot['origin_date'])} · "
        f"Harga Coal: ${snapshot['current_price']:,.2f}/tonne · "
        f"Versi model: {snapshot['version']}"
    )
    selected_horizon = st.session_state.get("prediction_selected_horizon", 1)
    colors = []
    for item in snapshot["results"]:
        color = UP if item["change_usd"] >= 0 else DOWN
        colors.append(
            f'.st-key-prediction_horizon_{item["h"]} button p {{color:{color} !important;}}'
        )
        if item["h"] == selected_horizon:
            colors.append(
                f'.st-key-prediction_horizon_{item["h"]} button '
                '{border-color:#b91c1c !important;box-shadow:0 0 0 3px rgba(185,28,28,0.12) !important;}'
            )
    st.html("<style>" + "".join(colors) + "</style>")
    for column, item in zip(st.columns(3, gap="large"), snapshot["results"]):
        with column:
            st.html(f'<h2 class="prediction-horizon-label">H+{item["h"]}</h2>')
            st.button(
                f'{item["change_usd"]:+.2f} USD',
                key=f'prediction_horizon_{item["h"]}',
                on_click=_change_horizon,
                args=(item["h"],),
                width="stretch",
            )
            st.html(
                '<p class="prediction-card-caption">'
                f'Harga ${item["predicted_price"]:,.2f} · {item["change_pct"]:+.2f}%<br>'
                f'{_date_label(item["forecast_date"])}</p>'
            )

    selected = next(item for item in snapshot["results"] if item["h"] == selected_horizon)
    left, right = st.columns([1.25, 1], gap="large")
    with left:
        st.html(
            '<div class="prediction-detail-heading">'
            f'<p><strong>Horizon: H+{selected_horizon}</strong> · '
            f'{_date_label(selected["forecast_date"])}</p>'
            f'<p>Model: <strong>{escape(selected["model_name"])}</strong></p></div>'
        )
        st.altair_chart(_prediction_chart(snapshot, selected), width="stretch", theme=None)
        trend_color = UP if selected["change_usd"] >= 0 else DOWN
        st.html(
            '<p class="prediction-chart-legend">'
            f'<span style="color:{BLUE}">● Harga historis</span> &nbsp; '
            f'<span style="color:{trend_color}">● Prediksi H+{selected_horizon}</span></p>'
        )
        st.caption("Segmen prediksi menghubungkan harga acuan dengan estimasi pada tanggal tujuan.")
    with right:
        _render_metrics(snapshot, selected)

    with st.expander("Detail hasil dan data input"):
        st.write(f"Sumber input: {snapshot['input_source']}")
        st.write(f"Baris tanggal input: {snapshot['actual_observations']}")
        st.write(f"Hari dengan harga yang diisi otomatis: {snapshot['filled_weekdays']}")

    download_column, _, return_column = st.columns([1.3, 1, 0.8])
    with download_column:
        try:
            report_files = (BASE_DIR / "prediction_report.py", BASE_DIR / "Assets" / "nwcast-logo.png")
            report_stamp = tuple((path.stat().st_mtime_ns, path.stat().st_size) for path in report_files)
            st.download_button(
                "Download Hasil PDF", data=_result_pdf(snapshot, report_stamp), mime="application/pdf",
                file_name=f'newcastle_prediction_{snapshot["version"]}_{snapshot["origin_date"]}.pdf',
                key="prediction_export", width="stretch",
            )
        except (OSError, ValueError, TypeError) as error:
            st.error(f"Laporan PDF belum dapat dibuat: {error}")
    with return_column:
        st.button("RETURN", key="prediction_return", on_click=_return_to_input, width="stretch")


def render() -> None:
    st.html(BASE_DIR / "Assets" / "prediction.css")
    if st.session_state.get("prediction_view") == "results" and "prediction_result" in st.session_state:
        _render_results()
    else:
        _render_input()
