from __future__ import annotations

from io import BytesIO
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from xgboost import XGBRegressor


VARIABLES = ("coal", "wti", "gas", "dxy")
HORIZONS = (1, 7, 14)
MIN_OBSERVATIONS = 31


def resolve_model_path(model_dir: str | Path) -> Path:
    """Read the active filename; the initial model is used before activation exists."""
    directory = Path(model_dir).resolve()
    pointer = directory / "active_model.json"
    filename = "nwcpredmodelv1.joblib"

    if pointer.exists():
        try:
            settings = json.loads(pointer.read_text(encoding="utf-8-sig"))
            if not isinstance(settings, dict):
                raise ValueError("Format penanda model harus berupa objek JSON.")
            filename = settings.get("model_file", settings.get("filename"))
            if (
                not isinstance(filename, str)
                or not filename
                or "/" in filename
                or "\\" in filename
                or Path(filename).name != filename
                or Path(filename).suffix.lower() != ".joblib"
            ):
                raise ValueError("Penanda model harus berisi nama file .joblib saja.")
        except (OSError, ValueError, TypeError) as error:
            raise ValueError(f"Penanda model aktif tidak valid: {error}") from error

    path = (directory / filename).resolve()
    if path.parent != directory:
        raise ValueError("Lokasi model aktif harus berada di folder Models.")
    if not path.is_file():
        raise FileNotFoundError(f"File model tidak ditemukan: {path.name}")
    return path


def load_bundle(path: str | Path) -> tuple[dict, dict]:
    """Load a trusted local export and check its saved reference predictions."""
    model_path = Path(path)
    try:
        payload = model_path.read_bytes()
        bundle = joblib.load(BytesIO(payload))
    except Exception as error:
        raise ValueError(
            "Paket model gagal dimuat. Gunakan ekspor terbaru yang menyimpan "
            f"XGBoost sebagai JSON native. Detail: {error}"
        ) from error

    if not isinstance(bundle, dict):
        raise ValueError("Isi paket model harus berupa dictionary.")

    required = {"models", "selected_models", "feature_columns", "metadata"}
    if not required.issubset(bundle):
        missing = ", ".join(sorted(required - bundle.keys()))
        raise ValueError(f"Paket model belum lengkap: {missing}.")

    expected_columns = expected_feature_columns()
    columns = bundle["feature_columns"]
    if len(columns) != len(set(columns)) or set(columns) != set(expected_columns):
        raise ValueError("Paket ini tidak sesuai dengan pipeline 48 fitur aplikasi.")

    metadata = bundle["metadata"]
    if not isinstance(metadata, dict):
        raise ValueError("Metadata model tidak valid.")
    if metadata.get("target") != "volatility_normalized_log_return":
        raise ValueError("Normalisasi target model berbeda dengan pipeline aplikasi.")
    if int(metadata.get("volatility_window", 0)) != 30:
        raise ValueError("Model harus memakai volatilitas Coal 30 periode.")

    loaded_models = {}
    for name, entry in bundle["models"].items():
        if not isinstance(entry, dict):
            raise ValueError(f"Format ekspor model {name} belum sesuai.")
        model_format = entry.get("format")
        try:
            if model_format == "xgboost_json":
                params = dict(bundle.get("hyperparameters", {}).get(name, {}))
                params.update(device="cpu", n_jobs=1)
                model = XGBRegressor(**params)
                model.load_model(bytearray(entry["data"]))
                model.set_params(device="cpu", n_jobs=1)
            elif model_format == "sklearn_joblib":
                model = entry["data"]
            else:
                raise ValueError(f"Format model tidak didukung: {model_format}.")
            if not callable(getattr(model, "predict", None)):
                raise ValueError("Objek model tidak memiliki fungsi predict.")
            if int(getattr(model, "n_features_in_", -1)) != len(columns):
                raise ValueError("Jumlah fitur model berbeda dengan daftar fitur.")
            loaded_models[name] = model
        except Exception as error:
            raise ValueError(f"Model {name} gagal dimuat: {error}") from error

    selected = bundle["selected_models"]
    normalized_selection = {}
    for horizon in HORIZONS:
        name = selected.get(horizon, selected.get(str(horizon)))
        if name not in loaded_models:
            raise ValueError(f"Model terpilih H+{horizon} tidak tersedia.")
        if not name.endswith(f"_H{horizon}"):
            raise ValueError(f"Model terpilih tidak sesuai horizon H+{horizon}.")
        normalized_selection[horizon] = name
    bundle["selected_models"] = normalized_selection

    verification = bundle.get("verification")
    if not isinstance(verification, dict):
        raise ValueError("Paket belum menyimpan contoh verifikasi prediksi dari Colab.")
    try:
        check_x = pd.DataFrame(verification["X"], columns=columns)
        if check_x.empty or not np.isfinite(check_x.to_numpy(dtype=float)).all():
            raise ValueError("Input verifikasi kosong atau tidak valid.")
        for name, model in loaded_models.items():
            actual = np.asarray(model.predict(check_x), dtype=float)
            expected = np.asarray(verification["predictions"][name], dtype=float)
            if actual.shape != expected.shape:
                raise ValueError(f"Dimensi prediksi verifikasi {name} berbeda.")
            if not np.isfinite(actual).all() or not np.isfinite(expected).all():
                raise ValueError(f"Nilai prediksi verifikasi {name} tidak valid.")
            np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)
    except Exception as error:
        raise ValueError(f"Verifikasi prediksi model tidak sesuai dengan Colab: {error}") from error

    bundle["_model_identity"] = {
        "filename": model_path.name,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
    }
    return bundle, loaded_models


def expected_feature_columns() -> list[str]:
    columns = []
    for variable in VARIABLES:
        columns.extend(f"{variable}_r_l{lag}" for lag in (0, 1, 2, 3, 5, 10))
        columns.extend(f"{variable}_mom{window}" for window in (5, 10, 20))
        columns.extend(f"{variable}_vol{window}" for window in (10, 30))
        columns.append(f"{variable}_dev_ma20")
    return columns


def prepare_prices(prices: pd.DataFrame) -> pd.DataFrame:
    """Validate actual rows, then causally fill the notebook's Monday-Friday grid."""
    if not isinstance(prices, pd.DataFrame) or not isinstance(prices.index, pd.DatetimeIndex):
        raise ValueError("Data harga harus memakai indeks tanggal yang valid.")
    if not set(VARIABLES).issubset(prices.columns) or prices.columns.duplicated().any():
        raise ValueError("Data harus memiliki kolom coal, wti, gas, dan dxy yang unik.")
    if len(prices) < MIN_OBSERVATIONS:
        raise ValueError("Diperlukan minimal 31 pengamatan lengkap untuk membentuk 48 fitur.")
    if prices.index.hasnans or prices.index.has_duplicates:
        raise ValueError("Tanggal tidak boleh kosong atau duplikat.")
    if prices.index.tz is not None:
        raise ValueError("Gunakan tanggal harian tanpa zona waktu.")
    if (prices.index != prices.index.normalize()).any():
        raise ValueError("Gunakan tanggal harian tanpa komponen jam.")
    if (prices.index.dayofweek >= 5).any():
        raise ValueError("Tanggal pengamatan harus Senin sampai Jumat sesuai pipeline model.")

    try:
        actual = prices.loc[:, list(VARIABLES)].astype(float).sort_index()
    except (TypeError, ValueError) as error:
        raise ValueError("Semua nilai harga harus berupa angka.") from error
    if not np.isfinite(actual.to_numpy()).all():
        raise ValueError("Semua baris pengamatan harus lengkap dan berisi angka yang valid.")
    if (actual.tail(MIN_OBSERVATIONS) <= 0).any().any():
        raise ValueError(
            "Harga pada 31 pengamatan terakhir harus lebih besar dari nol karena "
            "model menggunakan log harga."
        )

    grid = pd.bdate_range(actual.index.min(), actual.index.max(), name="Date")
    return actual.reindex(grid).ffill()


def build_features(prices: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reproduce the notebook transformations without targets or future data."""
    prepared = prepare_prices(prices)
    log_prices = np.log(prepared.where(prepared > 0))
    returns = log_prices.diff()
    columns = {}
    for variable in VARIABLES:
        for lag in (0, 1, 2, 3, 5, 10):
            columns[f"{variable}_r_l{lag}"] = returns[variable].shift(lag)
        for window in (5, 10, 20):
            columns[f"{variable}_mom{window}"] = log_prices[variable].diff(window)
        for window in (10, 30):
            columns[f"{variable}_vol{window}"] = returns[variable].rolling(window).std()
        columns[f"{variable}_dev_ma20"] = (
            log_prices[variable] - log_prices[variable].rolling(20).mean()
        )
    return pd.DataFrame(columns, index=prepared.index), prepared


def predict_latest(prices: pd.DataFrame, bundle: dict, models: dict) -> dict:
    """Predict from the final supplied date; never merge the system dataset."""
    features, prepared = build_features(prices)
    origin = prepared.index[-1]
    feature_columns = bundle["feature_columns"]
    feature_row = features.loc[[origin], feature_columns]
    if not np.isfinite(feature_row.to_numpy(dtype=float)).all():
        raise ValueError(
            "Fitur pada tanggal terakhir belum lengkap. Periksa 31 pengamatan "
            "terakhir dan pastikan setiap variabel berisi harga positif."
        )

    current_price = float(prepared.loc[origin, "coal"])
    volatility = float(np.log(prepared["coal"]).diff().rolling(30).std().loc[origin])
    if not np.isfinite(volatility) or volatility <= 0:
        raise ValueError(
            "Volatilitas Coal 30 periode harus lebih besar dari nol. "
            "Data harga Coal yang konstan belum dapat digunakan oleh model ini."
        )

    results = []
    for horizon in HORIZONS:
        name = bundle["selected_models"][horizon]
        prediction = np.asarray(models[name].predict(feature_row), dtype=float).reshape(-1)
        if prediction.size != 1 or not np.isfinite(prediction[0]):
            raise ValueError(f"Model H+{horizon} menghasilkan nilai yang tidak valid.")
        predicted_z = float(prediction[0])
        predicted_log_return = predicted_z * volatility * np.sqrt(horizon)
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            predicted_price = float(current_price * np.exp(predicted_log_return))
        if not np.isfinite(predicted_price) or predicted_price <= 0:
            raise ValueError(f"Harga prediksi H+{horizon} berada di luar rentang numerik.")
        change_usd = predicted_price - current_price
        cv_rmse = bundle.get("cv_rmse", {}).get(name)
        results.append({
            "h": horizon,
            "model_name": name,
            "cv_rmse": float(cv_rmse) if cv_rmse is not None else None,
            "predicted_z": predicted_z,
            "current_price": current_price,
            "predicted_price": predicted_price,
            "change_usd": float(change_usd),
            "change_pct": float(change_usd / current_price * 100),
            "forecast_date": (origin + pd.offsets.BDay(horizon)).date().isoformat(),
            "origin_date": origin.date().isoformat(),
        })

    return {
        "version": bundle["metadata"].get("version", "Tidak tercatat"),
        "metadata": dict(bundle["metadata"]),
        "model_identity": dict(bundle.get("_model_identity", {})),
        "origin_date": origin.date().isoformat(),
        "current_price": current_price,
        "volatility30": volatility,
        "actual_observations": len(prices),
        "filled_weekdays": len(prepared) - len(prices),
        "results": results,
        "history": prepared.loc[:, ["coal"]].copy(),
    }
