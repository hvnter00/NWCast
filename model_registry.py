from __future__ import annotations
from contextlib import contextmanager
from datetime import datetime, timezone

import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import uuid
import joblib
import pandas as pd


_VERSION = re.compile(r"v[1-9]\d*")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _version(value: str) -> str:
    if not isinstance(value, str) or not _VERSION.fullmatch(value):
        raise ValueError("Versi model tidak valid.")
    return value


def _read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError) as error:
        raise ValueError(f"Berkas tidak valid: {path.name}.") from error
    if not isinstance(value, dict):
        raise ValueError(f"Format berkas tidak valid: {path.name}.")
    return value


def _safe_relative(base: Path, value: str, parent: str, suffix: str) -> Path:
    if not isinstance(value, str) or not value or Path(value).is_absolute():
        raise ValueError("Lokasi berkas versi tidak valid.")
    path = (base / value).resolve()
    allowed = (base / parent).resolve()
    if not path.is_relative_to(allowed) or path.suffix.lower() != suffix:
        raise ValueError("Lokasi berkas versi di luar folder aplikasi.")
    return path


def _active_pointer(base: Path) -> dict:
    pointer = base / "Models" / "active_model.json"
    return _read_json(pointer) if pointer.is_file() else {}


def active_dataset_path(base_dir: str | Path) -> Path | None:
    base = Path(base_dir).resolve()
    pointer = _active_pointer(base)
    if "data_file" not in pointer:
        return None
    path = _safe_relative(base, pointer["data_file"], "Data/Versions", ".csv")
    if not path.is_file():
        raise ValueError("Dataset model aktif tidak ditemukan.")
    return path


@contextmanager
def training_lock(base_dir: str | Path):
    """An OS lock is released on process exit; its metadata file may persist."""
    directory = Path(base_dir).resolve() / "Models"
    directory.mkdir(parents=True, exist_ok=True)
    lock = directory / ".training.lock"
    payload = {"pid": os.getpid(), "started_at": _now(), "token": uuid.uuid4().hex}
    descriptor = os.open(lock, os.O_CREAT | os.O_RDWR, 0o600)
    handle = os.fdopen(descriptor, "r+b")
    try:
        handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            raise ValueError("Proses model lain masih berjalan.") from error
        try:
            handle.seek(0)
            handle.write(json.dumps(payload).encode("utf-8"))
            handle.truncate()
            handle.flush()
            yield payload
        finally:
            handle.seek(0)
            if os.name == "nt":
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    finally:
        handle.close()


def reserve_version(base_dir: str | Path) -> str:
    """Choose the next unused version while holding training_lock."""
    base = Path(base_dir)
    numbers = [1]
    for pattern, root in (("nwcpredmodelv*.joblib", base / "Models"),
                          ("v*.json", base / "Models" / "Versions"),
                          ("v*", base / "Data" / "Versions")):
        for path in root.glob(pattern):
            match = re.fullmatch(r"(?:nwcpredmodel)?v([1-9]\d*)(?:\.joblib|\.json)?", path.name)
            if match:
                numbers.append(int(match[1]))
    return f"v{max(numbers) + 1}"


def _atomic_pointer(base: Path, record: dict) -> None:
    directory = base / "Models"
    temporary = directory / f".active-{uuid.uuid4().hex}.json"
    try:
        temporary.write_text(json.dumps({
            "model_file": record["model_file"],
            "data_file": record.get("data_file"),
            "metrics_file": record.get("metrics_file"),
            "version": record["version"],
            "activated_at": _now(),
        }, indent=2, ensure_ascii=False), encoding="utf-8")
        # v1 may use the original four CSVs rather than a versioned combined file.
        pointer = _read_json(temporary)
        pointer = {key: value for key, value in pointer.items() if value is not None}
        temporary.write_text(json.dumps(pointer, indent=2, ensure_ascii=False), encoding="utf-8")
        os.replace(temporary, directory / "active_model.json")
    finally:
        temporary.unlink(missing_ok=True)


def publish_version(base_dir: str | Path, version: str, bundle: dict,
                    metrics: dict, raw_data: pd.DataFrame) -> dict:
    """Write a verified complete release, then replace the active pointer last."""
    from prediction_engine import load_bundle
    from training_data import _validate_raw

    base = Path(base_dir).resolve()
    version = _version(version)
    model_name = f"nwcpredmodel{version}.joblib"
    model_path = base / "Models" / model_name
    metrics_path = base / "Assets" / f"nwcpredmodel{version}_metrics.json"
    data_path = base / "Data" / "Versions" / version / "combined.csv"
    manifest_path = base / "Models" / "Versions" / f"{version}.json"
    outputs = (model_path, metrics_path, data_path, manifest_path)
    if any(path.exists() for path in outputs):
        raise ValueError("Versi model sudah tersedia.")
    raw = _validate_raw(raw_data)
    exported = dict(bundle)
    exported.pop("_model_identity", None)
    exported["metadata"] = dict(exported.get("metadata", {}), version=version)
    staging = base / "Models" / f".publish-{uuid.uuid4().hex}"
    staging.mkdir(parents=True)
    written = []
    activated = False
    try:
        staged_model = staging / model_name
        joblib.dump(exported, staged_model, compress=3)
        load_bundle(staged_model)
        model_sha = hashlib.sha256(staged_model.read_bytes()).hexdigest()
        published_metrics = dict(metrics)
        published_metrics.update(model_file=model_name, model_sha256=model_sha)
        staged_metrics = staging / metrics_path.name
        staged_metrics.write_text(json.dumps(published_metrics, indent=2, ensure_ascii=False,
                                            allow_nan=False), encoding="utf-8")
        staged_data = staging / "combined.csv"
        raw.to_csv(staged_data, index_label="Date", date_format="%Y-%m-%d")
        record = {
            "version": version,
            "model_file": model_name,
            "model_sha256": model_sha,
            "metrics_file": metrics_path.relative_to(base).as_posix(),
            "data_file": data_path.relative_to(base).as_posix(),
            "data_sha256": hashlib.sha256(staged_data.read_bytes()).hexdigest(),
            "metadata": exported["metadata"],
            "published_at": _now(),
            "data_rows": len(raw),
            "data_start": raw.index.min().date().isoformat(),
            "data_end": raw.index.max().date().isoformat(),
        }
        staged_manifest = staging / manifest_path.name
        staged_manifest.write_text(json.dumps(record, indent=2, ensure_ascii=False,
                                             allow_nan=False), encoding="utf-8")
        for source, target in zip((staged_model, staged_metrics, staged_data, staged_manifest), outputs):
            target.parent.mkdir(parents=True, exist_ok=True)
            # Exclusive lock guards version reservation; never replace an old release.
            if target.exists():
                raise ValueError("Versi model sudah tersedia.")
            os.replace(source, target)
            written.append(target)
        _atomic_pointer(base, record)
        activated = True
        return {**record, "active": True}
    except Exception:
        if not activated:
            for path in reversed(written):
                path.unlink(missing_ok=True)
        raise
    finally:
        shutil.rmtree(staging)


def _v1_record(base: Path) -> dict | None:
    path = base / "Models" / "nwcpredmodelv1.joblib"
    if not path.is_file():
        return None
    try:
        bundle = joblib.load(path)
        metadata = dict(bundle.get("metadata", {}))
    except Exception:
        metadata = {"version": "v1"}
    fingerprint = hashlib.sha256(path.read_bytes()).hexdigest()
    provenance = {}
    try:
        initial_metrics = _read_json(base / "Assets" / "nwcpredmodelv1_metrics.json")
        if initial_metrics.get("model_sha256") == fingerprint:
            provenance = initial_metrics.get("provenance", {})
    except ValueError:
        pass
    return {
        "version": "v1", "model_file": path.name, "metadata": metadata,
        "model_sha256": fingerprint,
        "metrics_file": "Assets/nwcpredmodelv1_metrics.json",
        "published_at": metadata.get("saved_at"),
        "data_start": provenance.get("data_start"), "data_end": provenance.get("data_end"),
    }


def list_versions(base_dir: str | Path) -> list[dict]:
    base = Path(base_dir).resolve()
    pointer = _active_pointer(base)
    active_name = pointer.get("model_file", "nwcpredmodelv1.joblib")
    records = []
    for manifest in (base / "Models" / "Versions").glob("v*.json"):
        if _VERSION.fullmatch(manifest.stem):
            record = _read_json(manifest)
            _version(record.get("version"))
            records.append(record)
    if not any(record["version"] == "v1" for record in records):
        initial = _v1_record(base)
        if initial is not None:
            records.append(initial)
    return [{**record, "active": record.get("model_file") == active_name}
            for record in sorted(records, key=lambda item: int(item["version"][1:]), reverse=True)]


def _activate_version(base_dir: str | Path, version: str) -> dict:
    """Restore a complete saved release while holding training_lock."""
    from prediction_engine import load_bundle

    base = Path(base_dir).resolve()
    version = _version(version)
    records = {record["version"]: record for record in list_versions(base)}
    if version not in records:
        raise ValueError("Versi model tidak ditemukan.")
    record = records[version]
    filename = record.get("model_file")
    if not isinstance(filename, str) or Path(filename).name != filename or filename != f"nwcpredmodel{version}.joblib":
        raise ValueError("Nama berkas model tidak valid.")
    model_path = _safe_relative(base, f"Models/{filename}", "Models", ".joblib")
    load_bundle(model_path)
    if hashlib.sha256(model_path.read_bytes()).hexdigest() != record.get("model_sha256"):
        raise ValueError("Berkas model berubah setelah disimpan.")
    if record.get("data_file"):
        data_path = _safe_relative(base, record["data_file"], "Data/Versions", ".csv")
        if not data_path.is_file() or hashlib.sha256(data_path.read_bytes()).hexdigest() != record.get("data_sha256"):
            raise ValueError("Dataset versi tidak lengkap atau berubah.")
    if record.get("metrics_file"):
        metrics_path = _safe_relative(base, record["metrics_file"], "Assets", ".json")
        metrics = _read_json(metrics_path)
        if metrics.get("model_sha256") != record.get("model_sha256"):
            raise ValueError("Metrik tidak cocok dengan versi model.")
    _atomic_pointer(base, record)
    return {**record, "active": True}


def activate_version(base_dir: str | Path, version: str) -> dict:
    with training_lock(base_dir):
        return _activate_version(base_dir, version)
