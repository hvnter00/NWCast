from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import platform
import tempfile
from typing import Callable

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
import xgboost
from xgboost import XGBRegressor

from prediction_engine import HORIZONS, VARIABLES, build_features


SOURCE_NOTEBOOK_SHA256 = "6b50050ca10df4e64254f6cb61d52bed0db7a77daf761c16211e1492721c99d5"
PARAMETER_CHOICES = {
    "XGBoost": {
        "n_estimators": [200, 400, 600, 1000],
        "learning_rate": [0.01, 0.03],
        "max_depth": [5, 8, 10],
        "min_child_weight": [1, 5, 10],
        "subsample": [0.7],
        "colsample_bytree": [0.5, 0.8],
        "gamma": [0],
        "reg_alpha": [0, 0.1, 0.5, 1, 3, 5, 10],
        "reg_lambda": [1, 5, 10, 20, 50],
    },
    "GBM": {
        "n_estimators": [200, 400, 600, 1000],
        "learning_rate": [0.01, 0.03],
        "max_depth": [5, 8, 10],
        "min_samples_leaf": [20, 50],
        "subsample": [0.7],
    },
}


@dataclass(frozen=True)
class TrainingConfig:
    """Pengaturan pembagian data dan pelatihan model."""

    n_splits: int = 5
    n_iter: int = 30
    train_ratio: float = 0.80
    gap: int = 14
    seed: int = 42
    search_jobs: int = 1
    parameter_choices: dict | None = None


def _date(value) -> str:
    return pd.Timestamp(value).date().isoformat()


def _configuration(config: TrainingConfig | dict | None) -> TrainingConfig:
    settings = TrainingConfig(**config) if isinstance(config, dict) else config or TrainingConfig()
    if not isinstance(settings, TrainingConfig):
        raise TypeError("Konfigurasi pelatihan tidak valid.")
    if (
        not 0 < settings.train_ratio < 1
        or settings.gap < max(HORIZONS)
        or settings.n_splits < 2
        or settings.n_iter < 1
        or settings.search_jobs < 1
    ):
        raise ValueError("Konfigurasi split/CV tidak valid.")
    return settings


def build_training_set(prices: pd.DataFrame) -> dict:
    """Build the common complete feature/target rows used by all six estimators."""
    features, prepared = build_features(prices)
    log_coal = np.log(prepared["coal"].where(prepared["coal"] > 0))
    volatility = log_coal.diff().rolling(30).std()
    scale_all = pd.DataFrame({h: volatility * np.sqrt(h) for h in HORIZONS})
    with np.errstate(divide="ignore", invalid="ignore"):
        raw_returns = pd.DataFrame({
            h: np.log(prepared.coal.shift(-h) / prepared.coal) for h in HORIZONS
        })
        targets = raw_returns / scale_all
    usd_all = pd.DataFrame({
        h: prepared.coal.shift(-h) - prepared.coal for h in HORIZONS
    })
    dates = pd.Series(prepared.index, index=prepared.index)
    target_dates = pd.DataFrame({h: dates.shift(-h) for h in HORIZONS})
    valid = (
        features.notna().all(axis=1)
        & np.isfinite(features).all(axis=1)
        & targets.notna().all(axis=1)
        & np.isfinite(targets).all(axis=1)
    )
    x = features.loc[valid].copy()
    if x.empty:
        raise ValueError("Data belum cukup untuk membentuk fitur dan target.")
    if not target_dates.loc[valid].notna().all().all():
        raise ValueError("Tanggal target pelatihan tidak lengkap.")
    return {
        "X": x,
        "y": targets.loc[valid].copy(),
        "scale": scale_all.loc[valid].copy(),
        "usd": usd_all.loc[valid].copy(),
        "target_dates": target_dates.loc[valid].copy(),
        "current_coal": prepared.coal.loc[x.index].copy(),
        "prices": prepared,
    }


def _metrics(actual: np.ndarray, predicted: np.ndarray) -> dict:
    nonzero = actual != 0
    return {
        "rmse_usd": float(np.sqrt(mean_squared_error(actual, predicted))),
        "mae_usd": float(mean_absolute_error(actual, predicted)),
        "r2_change": (
            float(r2_score(actual, predicted))
            if len(actual) > 1 and np.ptp(actual) > 0 else None
        ),
        "da_nonzero_pct": (
            float(100 * np.mean(np.sign(actual[nonzero]) == np.sign(predicted[nonzero])))
            if nonzero.any() else None
        ),
        "da_all_pct": float(100 * np.mean(np.sign(actual) == np.sign(predicted))),
        "nonzero_actual_n": int(nonzero.sum()),
    }


def train_models(
    prices: pd.DataFrame,
    version: str,
    progress: Callable[[float, str], None] | None = None,
    *,
    config: TrainingConfig | dict | None = None,
) -> tuple[dict, dict]:
    """Tune/evaluate six models, refit all valid rows, and return export + metrics.

    ``prices`` must contain aligned weekday coal/wti/gas/dxy prices. The caller
    handles uploaded data, persistence, version reservation, and activation.
    ``progress`` is an optional plain callback; no Streamlit UI is called here.
    """
    settings = _configuration(config)
    if not isinstance(version, str) or not version.startswith("v") or not version[1:].isdigit():
        raise ValueError("Versi model harus berupa v diikuti angka.")
    emit = progress or (lambda fraction, message: None)
    emit(0.0, "Menyiapkan fitur dan target.")
    data = build_training_set(prices)
    x, y = data["X"], data["y"]
    cut = int(len(x) * settings.train_ratio)
    if (
        cut - settings.gap < (settings.n_splits + 1) * (settings.gap + 2)
        or len(x) - cut < 2
    ):
        raise ValueError("Data terlalu pendek untuk split dan CV.")
    x_train, x_test = x.iloc[:cut - settings.gap], x.iloc[cut:]
    y_train = y.loc[x_train.index]
    target_dates = data["target_dates"]
    for h in HORIZONS:
        if target_dates.loc[x_train.index, h].max() >= x_test.index[0]:
            raise ValueError("Target training melewati batas data test.")
    cv = list(TimeSeriesSplit(n_splits=settings.n_splits, gap=settings.gap).split(x_train))
    cv_preview = []
    for fold, (train_pos, validation_pos) in enumerate(cv, 1):
        train_index, validation_index = x_train.index[train_pos], x_train.index[validation_pos]
        for h in HORIZONS:
            if target_dates.loc[train_index, h].max() >= validation_index[0]:
                raise ValueError("Target training melewati batas validasi CV.")
        cv_preview.append({
            "fold": fold,
            "train_rows": len(train_index),
            "validation_rows": len(validation_index),
            "last_training_target_h14": _date(target_dates.loc[train_index, 14].max()),
            "validation_start": _date(validation_index[0]),
        })
    emit(0.04, "Pembagian training, CV, dan test selesai.")

    templates = {
        "XGBoost": XGBRegressor(
            objective="reg:pseudohubererror", tree_method="hist", device="cpu",
            n_jobs=1, random_state=settings.seed, verbosity=0,
        ),
        "GBM": GradientBoostingRegressor(loss="huber", random_state=settings.seed),
    }
    spaces = settings.parameter_choices if settings.parameter_choices is not None else PARAMETER_CHOICES
    if set(spaces) != set(templates):
        raise ValueError("Ruang parameter harus mencakup XGBoost dan GBM.")
    model_specs = [(family, h) for family in templates for h in HORIZONS]
    cv_rmse, chosen_params, evaluation_models, baselines = {}, {}, {}, {}
    base = data["current_coal"].loc[x_test.index].to_numpy()
    for index, (family, h) in enumerate(model_specs):
        name = f"{family}_H{h}"
        start_fraction = 0.05 + index * 0.10
        emit(start_fraction, f"Tuning {name} ({index + 1}/6).")
        search = RandomizedSearchCV(
            estimator=clone(templates[family]), param_distributions=spaces[family],
            n_iter=settings.n_iter, scoring="neg_root_mean_squared_error", cv=cv,
            n_jobs=settings.search_jobs, refit=False, random_state=settings.seed,
            error_score="raise",
        )
        search.fit(x_train, y_train[h])
        cv_rmse[name] = float(-search.best_score_)
        if not np.isfinite(cv_rmse[name]):
            raise ValueError(f"Skor CV {name} tidak valid.")
        chosen_params[name] = dict(search.best_params_)
        emit(start_fraction + 0.07, f"Mengevaluasi {name} pada test historis.")
        evaluation_model = clone(templates[family]).set_params(**chosen_params[name])
        evaluation_model.fit(x_train, y_train[h])
        predicted_z = np.asarray(evaluation_model.predict(x_test), dtype=float)
        predicted_lr = predicted_z * data["scale"].loc[x_test.index, h].to_numpy()
        with np.errstate(over="ignore", invalid="ignore"):
            predicted_usd = base * (np.exp(predicted_lr) - 1)
        if not np.isfinite(predicted_usd).all():
            raise ValueError(f"Prediksi evaluasi {name} tidak valid.")
        actual_usd = data["usd"].loc[x_test.index, h].to_numpy()
        evaluation_models[name] = {
            "horizon": h, "family": family,
            **_metrics(actual_usd, predicted_usd),
            "cv_rmse_normalized": cv_rmse[name],
        }
        baselines[str(h)] = _metrics(actual_usd, np.zeros_like(actual_usd))
        emit(start_fraction + 0.09, f"Tuning dan evaluasi {name} selesai.")

    selected_models = {
        h: min((f"{family}_H{h}" for family in templates), key=lambda name: cv_rmse[name])
        for h in HORIZONS
    }
    final_models = {}
    for index, (family, h) in enumerate(model_specs):
        name = f"{family}_H{h}"
        emit(0.66 + index * 0.045, f"Pelatihan final {name} pada seluruh data bertarget.")
        final_model = clone(templates[family]).set_params(**chosen_params[name])
        final_model.fit(x, y[h])
        final_models[name] = final_model
        emit(0.70 + index * 0.045, f"Pelatihan final {name} selesai.")

    emit(0.95, "Menyiapkan paket model dan evaluasi.")
    exported_models = {}
    with tempfile.TemporaryDirectory() as temporary_dir:
        for name, model in final_models.items():
            if isinstance(model, XGBRegressor):
                model_path = Path(temporary_dir) / f"{name}.json"
                model.save_model(model_path)
                exported_models[name] = {"format": "xgboost_json", "data": model_path.read_bytes()}
            else:
                exported_models[name] = {"format": "sklearn_joblib", "data": model}
    now = datetime.now(timezone.utc).isoformat()
    prepared = data["prices"]
    configuration = {
        "n_splits": settings.n_splits, "n_iter": settings.n_iter,
        "train_ratio": settings.train_ratio, "gap": settings.gap,
        "seed": settings.seed, "search_jobs": settings.search_jobs,
        "scoring": "neg_root_mean_squared_error", "selection": "minimum_cv_rmse_per_horizon",
        "xgboost_device": "cpu", "parameter_choices": spaces,
    }
    metadata = {
        "version": version, "saved_at": now, "trained_at": now,
        "train_start": _date(x.index.min()), "train_end": _date(x.index.max()),
        "train_rows": len(x), "fit_phase": "full_refit_after_historical_evaluation",
        "data_start": _date(prepared.index.min()), "data_end": _date(prepared.index.max()),
        "data_rows": len(prepared),
        "evaluation_train_start": _date(x_train.index.min()),
        "evaluation_train_end": _date(x_train.index.max()),
        "evaluation_train_rows": len(x_train),
        "target": "volatility_normalized_log_return", "volatility_window": 30,
        "training_configuration": configuration,
        "source_notebook_sha256": SOURCE_NOTEBOOK_SHA256,
        "training_module_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "versions": {
            "python": platform.python_version(), "scikit_learn": sklearn.__version__,
            "xgboost": xgboost.__version__, "numpy": np.__version__,
            "pandas": pd.__version__, "joblib": joblib.__version__,
        },
    }
    check_x = x_test.head(3).copy()
    bundle = {
        "models": exported_models, "selected_models": selected_models,
        "cv_rmse": cv_rmse, "feature_columns": list(x.columns),
        "horizons": list(HORIZONS),
        "hyperparameters": {name: model.get_params() for name, model in final_models.items()},
        "metadata": metadata,
        "verification": {
            "X": check_x.to_numpy().tolist(),
            "predictions": {name: model.predict(check_x).tolist() for name, model in final_models.items()},
        },
    }
    metrics = {
        "schema_version": 1, "model_file": f"nwcpredmodel{version}.joblib",
        "created_at_utc": now,
        "provenance": {
            "method": "Fixed chronological holdout evaluated before final full-data refit.",
            "preprocessing": "Joint weekday grid; causal forward fill; shared 48-feature recipe; common complete targets.",
            "data_start": metadata["data_start"], "data_end": metadata["data_end"],
            "data_grid_rows": len(prepared), "common_supervised_rows": len(x),
            "gap_observations": settings.gap, "model_refit_performed": True,
            "refit_scope": "All common complete feature-target rows; fixed tuned hyperparameters; no retuning.",
            "source_notebook_sha256": SOURCE_NOTEBOOK_SHA256,
            "training_configuration": configuration, "cv_folds": cv_preview,
            "checks": {"train_target_precedes_holdout": True, "cv_training_targets_precede_validation": True},
        },
        "evaluation": {
            "kind": "historical_holdout_before_full_refit",
            "origin_start": _date(x_test.index.min()), "origin_end": _date(x_test.index.max()),
            "n": len(x_test),
            "training_origin_start": _date(x_train.index.min()),
            "training_origin_end": _date(x_train.index.max()),
            "training_n": len(x_train),
            "target_date_end_by_horizon": {
                str(h): _date(target_dates.loc[x_test.index, h].max()) for h in HORIZONS
            },
            "target_units": "USD/tonne change",
            "r2_definition": "R2 on future Coal USD price changes, relative to holdout mean change.",
            "direction_accuracy": "Sign agreement; primary DA excludes zero actual USD changes.",
            "cv_units": "Volatility-normalized log return; different units from USD RMSE.",
            "baseline": "Zero future price change (persistence).",
            "fresh_prediction_has_actual": False,
            "describes_final_refitted_estimators": False,
        },
        "selected_models": {str(h): name for h, name in selected_models.items()},
        "models": evaluation_models, "baselines": baselines,
    }
    emit(1.0, "Pelatihan selesai; paket siap diverifikasi dan disimpan.")
    return bundle, metrics
