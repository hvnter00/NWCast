from __future__ import annotations

import hmac
import hashlib
import json
from io import BytesIO
from pathlib import Path
import pandas as pd
import streamlit as st

from model_registry import activate_version, list_versions
from prediction_engine import load_bundle, resolve_model_path
from training import build_training_set
from training_data import load_historical, make_admin_template, merge_data, parse_new_data, prepare_training_prices
from training_jobs import cancel_retraining, get_job, is_running, start_retraining


BASE_DIR = Path(__file__).resolve().parent
GUIDE_PATH = BASE_DIR / "Assets" / "admin_manual.pdf"
DISPLAY_COLUMNS = {"coal": "Coal", "wti": "Oil", "gas": "Gas", "dxy": "DXY"}


@st.cache_resource(show_spinner=False)
def _cached_bundle(path: str, modified_ns: int, size: int):
    return load_bundle(path)


@st.cache_data(show_spinner=False)
def _template(format: str) -> bytes:
    """Membuat template input Admin."""
    return make_admin_template(format)


@st.cache_data(show_spinner=False)
def _dataset_excel(frame: pd.DataFrame) -> bytes:
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl", date_format="yyyy-mm-dd", datetime_format="yyyy-mm-dd") as writer:
        frame.to_excel(writer, sheet_name="Data", index=False)
        sheet = writer.sheets["Data"]
        sheet.freeze_panes = "A2"
        sheet.auto_filter.ref = sheet.dimensions
        for row in sheet.iter_rows(min_row=2, max_row=sheet.max_row, min_col=1, max_col=1):
            row[0].number_format = "yyyy-mm-dd"
        for column in ("A", "B", "C", "D", "E"):
            sheet.column_dimensions[column].width = 18
    return output.getvalue()


def _date(value) -> str:
    return pd.Timestamp(value).strftime("%d %b %Y") if value is not None else "—"


def _time(value) -> str:
    if not value:
        return "—"
    stamp = pd.Timestamp(value)
    stamp = stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp
    return stamp.tz_convert("Asia/Jakarta").strftime("%d %b %Y, %H:%M WIB")


def _dismiss_notice(state_key: str, token: str) -> None:
    st.session_state[state_key] = token


def _notice(level: str, message: str, *, slot: str, identity: str = "") -> bool:
    """Hide only the dismissed message; a new job/status/message is shown again."""
    token = hashlib.sha256(f"{identity}\n{level}\n{message}".encode("utf-8")).hexdigest()
    state_key = f"admin_dismissed_notice_{slot}"
    if st.session_state.get(state_key) == token:
        return False
    with st.container(key=f"admin_notice_{level}_{slot}", horizontal=True, wrap=False,
                      vertical_alignment="center", gap=None):
        getattr(st, level)(message)
        st.button("×", key=f"admin_notice_close_{slot}_{token[:12]}", width=40,
                  help="Tutup notifikasi", on_click=_dismiss_notice, args=(state_key, token))
    return True


def _admin_credentials() -> tuple[str, str] | None:
    try:
        settings = st.secrets["admin"]
        username, password = settings["username"], settings["password"]
    except (FileNotFoundError, KeyError, TypeError):
        return None
    if not isinstance(username, str) or not isinstance(password, str):
        return None
    if not username.strip() or not password.strip():
        return None
    return username, password


def _login() -> None:
    username = str(st.session_state.get("admin_login_username", ""))
    password = str(st.session_state.get("admin_login_password", ""))
    credentials = _admin_credentials()
    valid = credentials is not None and (
        hmac.compare_digest(username.encode(), credentials[0].encode())
        and hmac.compare_digest(password.encode(), credentials[1].encode())
    )
    st.session_state["admin_authenticated"] = valid
    st.session_state["admin_login_failed"] = not valid
    st.session_state.pop("admin_dismissed_notice_login", None)
    st.session_state.pop("admin_login_password", None)


def _logout() -> None:
    for key in tuple(st.session_state):
        if key.startswith("admin_"):
            st.session_state.pop(key, None)


def _remember_upload() -> None:
    uploaded = st.session_state.get("admin_upload_widget")
    if uploaded is None:
        st.session_state.pop("admin_upload_bytes", None)
        st.session_state.pop("admin_upload_name", None)
    else:
        st.session_state["admin_upload_bytes"] = uploaded.getvalue()
        st.session_state["admin_upload_name"] = uploaded.name


def _login_page() -> None:
    st.html('<h1 class="admin-title">Admin Login</h1>')
    _, middle, _ = st.columns([1, 1.5, 1])
    with middle:
        configured = _admin_credentials() is not None
        if not configured:
            st.info("Login Admin belum tersedia. Hubungi pengelola aplikasi.")
        with st.form("admin_login_form"):
            st.text_input("Username", key="admin_login_username")
            st.text_input("Password", type="password", key="admin_login_password")
            st.form_submit_button("LOGIN", type="primary", on_click=_login, disabled=not configured, width="stretch")
        if st.session_state.get("admin_login_failed"):
            _notice("error", "Username atau password salah.", slot="login")


def _read_metrics(bundle: dict) -> dict | None:
    identity = bundle.get("_model_identity", {})
    stem = Path(identity.get("filename", "")).stem
    try:
        metrics = json.loads((BASE_DIR / "Assets" / f"{stem}_metrics.json").read_text(encoding="utf-8-sig"))
        return metrics if metrics.get("model_sha256") == identity.get("sha256") else None
    except (OSError, ValueError):
        return None


def _active_model(bundle: dict, raw: pd.DataFrame) -> None:
    metadata = bundle["metadata"]
    metrics = _read_metrics(bundle)
    data_end = metadata.get("data_end") or (metrics or {}).get("provenance", {}).get("data_end")
    if not data_end:
        data_end = prepare_training_prices(raw).index.max().date().isoformat()
    version, latest, training_end = st.columns(3)
    version.metric("Model aktif", metadata.get("version", "—"))
    latest.metric("Data model hingga", _date(data_end))
    training_end.metric("Sampel training hingga", _date(metadata.get("train_end")))
    if metadata.get("trained_at"):
        st.caption(f"Terakhir dilatih: {_time(metadata['trained_at'])}")
    else:
        st.caption(f"Model disimpan: {_time(metadata.get('saved_at'))}")
    rows = [{"Horizon": f"H+{h}", "Model terpilih": name, "RMSE CV": bundle.get("cv_rmse", {}).get(name)}
            for h, name in bundle["selected_models"].items()]
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    with st.expander("Hyperparameter model aktif"):
        model_name = st.selectbox("Pilih model", sorted(bundle["models"]), key=f"admin_parameters_{metadata.get('version', 'model')}")
        st.json(bundle.get("hyperparameters", {}).get(model_name, {}))
        settings = metadata.get("training_configuration")
        if settings:
            st.caption(f"Tuning {settings['n_iter']} kombinasi · {settings['n_splits']} fold · jeda {settings['gap']} periode")
    with st.expander("Evaluasi historis model aktif"):
        if not metrics:
            _notice("info", "Metrik versi ini belum tersedia.", slot="metrics")
        else:
            evaluation = metrics["evaluation"]
            if evaluation.get("kind") == "historical_holdout_before_full_refit":
                st.caption("Evaluasi berikut dihitung sebelum pelatihan final pada seluruh data.")
            st.caption(f"{evaluation['n']} sampel · {_date(evaluation['origin_start'])} – {_date(evaluation['origin_end'])}")
            rows = [{"Model": name, "RMSE (USD/tonne)": scores["rmse_usd"], "MAE (USD/tonne)": scores["mae_usd"],
                     "R²": scores["r2_change"], "DA (%)": scores["da_nonzero_pct"]} for name, scores in metrics["models"].items()]
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    with st.expander("Dataset aktif"):
        st.caption(f"{len(raw):,} tanggal · {_date(raw.index.min())} – {_date(raw.index.max())}")
        displayed = raw.rename(columns=DISPLAY_COLUMNS).reset_index()
        st.dataframe(displayed.tail(20).iloc[::-1], hide_index=True, width="stretch")
        st.download_button("Download Dataset Aktif", data=_dataset_excel(displayed),
                           file_name=f"newcastle_dataset_{metadata.get('version', 'aktif')}.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", key="admin_dataset_download")


@st.fragment(run_every="2s")
def _job_panel() -> None:
    if not st.session_state.get("admin_authenticated"):
        return
    job = get_job(BASE_DIR)
    if not job:
        return
    status = job["status"]
    if status in {"queued", "running", "cancelling", "publishing"}:
        st.html('<h2 class="admin-section-title">Progres Retraining</h2>')
        progress_column, cancel_column = st.columns([5, 1], vertical_alignment="center")
        with progress_column:
            st.progress(float(job.get("progress", 0)), text=job.get("message", "Memproses..."))
        with cancel_column:
            if st.button("Cancel", key="admin_cancel", width="stretch", disabled=status in {"cancelling", "publishing"}):
                outcome = cancel_retraining(BASE_DIR, job["id"])
                if not outcome["accepted"]:
                    _notice("info", outcome["message"], slot="cancel", identity=job["id"])
                st.rerun()
        if status == "publishing":
            st.caption("Menyimpan dan mengaktifkan versi baru. Pembatalan dinonaktifkan pada tahap akhir ini.")
        elif status == "cancelling":
            st.caption("Pembatalan sedang diproses. Model aktif tetap tersedia.")
        else:
            st.caption("Anda boleh berpindah halaman. Gunakan Cancel untuk membatalkan retraining.")
    elif status == "cancelled":
        if st.session_state.get("admin_seen_cancelled_job") != job["id"]:
            st.session_state["admin_seen_cancelled_job"] = job["id"]
            st.rerun()
        _notice("info", job.get("message", "Retraining dibatalkan. Model aktif tetap tersedia."),
                slot="job", identity=f"{job['id']}:{status}")
    elif status == "completed":
        if st.session_state.get("admin_seen_completed_job") != job["id"]:
            st.session_state["admin_seen_completed_job"] = job["id"]
            st.rerun()
        _notice("success", job["message"], slot="job", identity=f"{job['id']}:{status}")
    elif status in {"failed", "interrupted"}:
        visible = _notice("error", job.get("error") or job.get("message", "Retraining terhenti."),
                          slot="job", identity=f"{job['id']}:{status}")
        if visible and job.get("error_detail"):
            with st.expander("Rincian error retraining"):
                st.code(job["error_detail"], language="text")


def _retraining_form(raw: pd.DataFrame) -> None:
    st.html('<h2 class="admin-section-title">Update Model</h2>')
    busy = is_running(BASE_DIR)
    mode = st.radio("Sumber data retraining", ["Data historis tersimpan", "Historis + data baru"], horizontal=True, key="admin_data_mode", disabled=busy)
    candidate = raw
    source_name = "Data historis tersimpan"
    summary = {"dates_added": 0, "dates_updated": 0, "values_added": 0, "values_changed": 0,
               "rows_before": len(raw), "rows_after": len(raw), "uploaded_rows": 0,
               "date_start": raw.index.min().date().isoformat(), "date_end": raw.index.max().date().isoformat()}
    can_start = True
    if mode == "Historis + data baru":
        st.file_uploader("Upload Excel gabungan", type=["xlsx"], max_upload_size=15, key="admin_upload_widget", on_change=_remember_upload, disabled=busy)
    excel, guide, _ = st.columns([1, 1, 2])
    with excel:
        st.download_button("Template Excel", data=_template("xlsx"), file_name="newcastle_admin_template.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", key="admin_template_excel", width="stretch")
    with guide:
        guide_available = GUIDE_PATH.is_file()
        st.download_button("Manual Guide", data=GUIDE_PATH.read_bytes() if guide_available else b"",
                           file_name=GUIDE_PATH.name, mime="application/pdf", disabled=not guide_available,
                           key="admin_manual_guide", width="stretch")
        if not guide_available:
            st.caption("Panduan segera tersedia.")
    if mode == "Historis + data baru":
        payload = st.session_state.get("admin_upload_bytes")
        if payload:
            try:
                source_name = st.session_state["admin_upload_name"]
                if Path(source_name).suffix.lower() != ".xlsx":
                    raise ValueError("Gunakan file Excel (.xlsx).")
                new = parse_new_data(payload, source_name)
                candidate, summary = merge_data(raw, new)
                st.caption(f"File dipilih: {source_name}")
            except (OSError, ValueError, TypeError) as error:
                _notice("error", str(error), slot="upload")
                can_start = False
        else:
            _notice("info", "Unggah data baru terlebih dahulu.", slot="upload")
            can_start = False
    if can_start:
        try:
            aligned = prepare_training_prices(candidate)
            supervised = build_training_set(aligned)
            first, second, third = st.columns(3)
            first.metric("Tanggal baru", summary["dates_added"])
            second.metric("Tanggal diperbarui", summary["dates_updated"])
            third.metric("Sampel bertarget", f"{len(supervised['X']):,}")
            st.caption(f"Data bersama sampai {_date(aligned.index.max())} · sampel training final sampai {_date(supervised['X'].index.max())}")
            if summary["dates_updated"]:
                _notice("info", "Harga terisi pada tanggal yang sama akan mengganti nilai lama setelah retraining berhasil.", slot="merge")
            with st.expander("Preview data gabungan"):
                st.dataframe(candidate.rename(columns=DISPLAY_COLUMNS).tail(20).iloc[::-1].reset_index(), hide_index=True, width="stretch")
        except (OSError, ValueError, TypeError) as error:
            _notice("error", str(error), slot="preview")
            can_start = False
    st.caption("Tuning 6 model · 30 kombinasi · 5 fold")
    if st.button("RETRAIN MODEL", type="primary", key="admin_retrain", disabled=busy or not can_start, width="stretch"):
        if not st.session_state.get("admin_authenticated"):
            _notice("error", "Silakan login kembali.", slot="auth")
            return
        try:
            st.session_state.pop("admin_dismissed_notice_retrain", None)
            job = start_retraining(BASE_DIR, candidate, summary, source_name)
            st.session_state["admin_current_job"] = job["id"]
            st.rerun()
        except (OSError, ValueError, TypeError) as error:
            _notice("error", str(error), slot="retrain")


def _version_history(records: list[dict]) -> None:
    st.html('<h2 class="admin-section-title">Riwayat Versi</h2>')
    rows = []
    for item in records:
        metadata = item.get("metadata", {})
        rows.append({"Versi": item["version"], "Status": "Aktif" if item.get("active") else "Tersimpan",
                     "Waktu simpan": _time(metadata.get("saved_at") or item.get("published_at")),
                     "Data hingga": _date(metadata.get("data_end") or item.get("data_end")),
                     "Training hingga": _date(metadata.get("train_end"))})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    alternatives = [item["version"] for item in records if not item.get("active")]
    if alternatives:
        with st.expander("Gunakan versi tersimpan"):
            st.caption("Model dan dataset akan mengikuti versi yang dipilih.")
            version = st.selectbox("Versi model", alternatives, key="admin_restore_version")
            if st.button("AKTIFKAN VERSI", key="admin_activate", disabled=is_running(BASE_DIR)):
                if not st.session_state.get("admin_authenticated"):
                    _notice("error", "Silakan login kembali.", slot="auth")
                    return
                try:
                    activate_version(BASE_DIR, version)
                    st.session_state.pop("admin_dismissed_notice_activation", None)
                    st.session_state["admin_activation_notice"] = f"Model {version} diaktifkan."
                    st.rerun()
                except (OSError, ValueError, TypeError) as error:
                    _notice("error", str(error), slot="activation_error")


def render() -> None:
    st.html(BASE_DIR / "Assets" / "admin.css")
    if not st.session_state.get("admin_authenticated"):
        _login_page()
        return
    title, logout = st.columns([5, 1])
    with title:
        st.html('<h1 class="admin-title admin-title-left">Panel Admin</h1>')
    with logout:
        st.button("LOGOUT", key="admin_logout", on_click=_logout, width="stretch")
    notice = st.session_state.get("admin_activation_notice")
    if notice:
        _notice("success", notice, slot="activation")
    _job_panel()
    try:
        path = resolve_model_path(BASE_DIR / "Models")
        stamp = path.stat()
        bundle, _ = _cached_bundle(str(path), stamp.st_mtime_ns, stamp.st_size)
        raw = load_historical(BASE_DIR)
        records = list_versions(BASE_DIR)
        _active_model(bundle, raw)
        st.divider()
        _retraining_form(raw)
        st.divider()
        _version_history(records)
    except (OSError, ValueError, TypeError) as error:
        _notice("error", str(error), slot="load")
