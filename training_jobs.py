from __future__ import annotations
from model_registry import publish_version, reserve_version, training_lock
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import threading
import time
import traceback
from uuid import uuid4
import joblib
import pandas as pd

_GUARD = threading.RLock()
_JOBS: dict[tuple[str, str], dict] = {}
_RUNNING = {"queued", "running", "cancelling", "publishing"}
_CANCELLABLE = {"queued", "running", "cancelling"}
_CANCEL_MESSAGE = "Menghentikan proses retraining..."
_LOG_LIMIT = 16_384

class _Cancelled(Exception):
    """Internal signal; a cancellation is a normal terminal job state."""

def _now() -> str:
    return datetime.now(timezone.utc).isoformat()

def _job_directory(base_dir: str | Path) -> Path:
    return Path(base_dir).resolve() / "Models" / "Jobs"

def _cancel_path(base: Path, job_id: str) -> Path:
    return _job_directory(base) / f".{job_id}.cancel"

@contextmanager
def _state_lock(base: Path):
    """Serialize cancellation versus publication across server processes."""
    with _GUARD:
        directory = _job_directory(base)
        directory.mkdir(parents=True, exist_ok=True)
        handle = (directory / ".state.lock").open("a+b")
        try:
            handle.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                handle.seek(0)
                if os.name == "nt":
                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()

def _write_job(base: Path, job: dict) -> None:
    directory = _job_directory(base)
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / f"{job['id']}.json"
    temporary = directory / f".{job['id']}.{uuid4().hex}.tmp"
    try:
        temporary.write_text(json.dumps(job, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        for attempt in range(20):
            try:
                os.replace(temporary, target)
                break
            except PermissionError:
                if attempt == 19:
                    raise
                time.sleep(0.01)
    finally:
        temporary.unlink(missing_ok=True)

def _process_alive(pid: int) -> bool:
    if pid == os.getpid():
        return True
    if pid <= 0:
        return False
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
        kernel.GetExitCodeProcess.restype = wintypes.BOOL
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        handle = kernel.OpenProcess(0x1000, False, pid)
        if not handle:
            return ctypes.get_last_error() == 5
        try:
            code = wintypes.DWORD()
            return bool(kernel.GetExitCodeProcess(handle, ctypes.byref(code))) and code.value == 259
        finally:
            kernel.CloseHandle(handle)
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def _load_record(path: Path) -> dict | None:
    try:
        job = json.loads(path.read_text(encoding="utf-8"))
        return job if isinstance(job, dict) and job.get("id") == path.stem else None
    except (OSError, ValueError, TypeError):
        return None


def _latest_record(base: Path, job_id: str) -> dict | None:
    live = _JOBS.get((str(base), job_id))
    disk = _load_record(_job_directory(base) / f"{job_id}.json")
    # Another process may have requested cancellation since the local update.
    if disk and (not live or disk.get("updated_at", "") > live.get("updated_at", "")):
        return deepcopy(disk)
    return deepcopy(live or disk)


def _read_job(base: Path, job_id: str) -> dict | None:
    job = _latest_record(base, job_id)
    if job and job.get("status") in _RUNNING:
        try:
            pid = int(job.get("owner_pid", 0))
        except (ValueError, TypeError):
            return None
        thread_alive = any(thread.name == f"newcastle-retrain-{job_id}" for thread in threading.enumerate())
        if not _process_alive(pid) or (pid == os.getpid() and not thread_alive):
            job.update(status="interrupted", message="Pelatihan terhenti. Jalankan ulang retraining.")
            _cleanup_interrupted(base, job_id, job)
        elif job["status"] in _CANCELLABLE and _cancel_path(base, job_id).is_file():
            job.update(status="cancelling", message=_CANCEL_MESSAGE)
    return job


def _cleanup_interrupted(base: Path, job_id: str, job: dict) -> None:
    """Recover scratch files after server death, only when its child is gone."""
    work = _job_directory(base) / f".work-{job_id}"
    try:
        worker_pid = _worker_pid(work) or job.get("worker_pid")
        if worker_pid is not None and not _process_alive(int(worker_pid)):
            if work.exists():
                shutil.rmtree(work)
            _cancel_path(base, job_id).unlink(missing_ok=True)
    except (OSError, ValueError, TypeError):
        # A later status read can retry; never remove another running job's data.
        pass


def _worker_pid(work: Path) -> int | None:
    try:
        value = int(json.loads((work / "worker.json").read_text(encoding="utf-8"))["pid"])
        return value if value > 0 else None
    except (OSError, ValueError, KeyError, TypeError):
        return None


def get_job(base_dir: str | Path, job_id: str | None = None) -> dict | None:
    base = Path(base_dir).resolve()
    with _GUARD:
        if job_id:
            if not re.fullmatch(r"[a-f0-9]{32}", job_id):
                return None
            return _read_job(base, job_id)
        directory = _job_directory(base)
        if not directory.is_dir():
            return None
        for path in sorted(directory.glob("*.json"), key=lambda item: item.stat().st_mtime_ns, reverse=True):
            job = _read_job(base, path.stem)
            if job:
                return job
    return None


def is_running(base_dir: str | Path) -> bool:
    base = Path(base_dir).resolve()
    with _GUARD:
        directory = _job_directory(base)
        ids = {key[1] for key in _JOBS if key[0] == str(base)}
        if directory.is_dir():
            ids.update(path.stem for path in directory.glob("*.json"))
        return any((job := _read_job(base, job_id)) and job.get("status") in _RUNNING for job_id in ids)


def cancel_retraining(base_dir: str | Path, job_id: str) -> dict:
    """Request cancellation; never kill a PID supplied by a record or caller.

    Returns {accepted, status, message}. Repeated requests while cancelling are
    accepted. Publication/terminal jobs cannot be cancelled. The supervisor stops
    its own Popen child before recording cancelled. A disk request and shared
    state lock also support requests from another server process.
    """
    base = Path(base_dir).resolve()
    if not isinstance(job_id, str) or not re.fullmatch(r"[a-f0-9]{32}", job_id):
        return {"accepted": False, "status": "missing", "message": "Job retraining tidak ditemukan."}
    with _state_lock(base):
        job = _read_job(base, job_id)
        if not job:
            return {"accepted": False, "status": "missing", "message": "Job retraining tidak ditemukan."}
        status = job.get("status")
        if status not in _CANCELLABLE:
            message = ("Model sedang disimpan dan diaktifkan. Tunggu sampai selesai."
                       if status == "publishing" else "Retraining sudah tidak berjalan.")
            return {"accepted": False, "status": status, "message": message}
        marker = _cancel_path(base, job_id)
        if not marker.exists():
            marker.write_text(job_id, encoding="ascii")
        job.update(status="cancelling", message=_CANCEL_MESSAGE, updated_at=_now())
        job.setdefault("cancel_requested_at", _now())
        _JOBS[(str(base), job_id)] = job
        _write_job(base, job)
        return {"accepted": True, "status": "cancelling", "message": _CANCEL_MESSAGE}


def start_retraining(
    base_dir: str | Path,
    raw_data: pd.DataFrame,
    summary: dict | None = None,
    source_name: str = "Data historis tersimpan",
    *,
    config=None,
) -> dict:
    """Freeze inputs and start a supervisor that survives page changes/logout."""
    base = Path(base_dir).resolve()
    if not isinstance(raw_data, pd.DataFrame) or raw_data.empty:
        raise ValueError("Dataset belum tersedia.")
    frozen_data = raw_data.copy(deep=True)
    frozen_config = deepcopy(config)
    with _state_lock(base):
        if is_running(base):
            raise ValueError("Retraining masih berjalan.")
        job_id = uuid4().hex
        job = {
            "id": job_id, "status": "queued", "progress": 0.0,
            "message": "Menyiapkan retraining...", "version": None,
            "created_at": _now(), "updated_at": _now(),
            "started_at": None, "finished_at": None, "owner_pid": os.getpid(),
            "source": source_name, "data_summary": deepcopy(summary or {}),
        }
        _JOBS[(str(base), job_id)] = job
        try:
            _write_job(base, job)
            thread = threading.Thread(
                target=_run, args=(base, job_id, frozen_data, frozen_config),
                name=f"newcastle-retrain-{job_id}", daemon=True,
            )
            thread.start()
        except Exception:
            _JOBS.pop((str(base), job_id), None)
            (_job_directory(base) / f"{job_id}.json").unlink(missing_ok=True)
            raise
        return deepcopy(job)


def _update(base: Path, job_id: str, **changes) -> None:
    with _state_lock(base):
        job = _latest_record(base, job_id)
        if job is None:
            raise ValueError("Job retraining tidak ditemukan.")
        if _cancel_path(base, job_id).exists() and changes.get("status", job["status"]) in _CANCELLABLE:
            changes.update(status="cancelling", message=_CANCEL_MESSAGE)
        job.update(changes, updated_at=_now())
        _JOBS[(str(base), job_id)] = job
        _write_job(base, job)


def _check_cancel(base: Path, job_id: str) -> None:
    if _cancel_path(base, job_id).is_file():
        raise _Cancelled()


def _begin_publishing(base: Path, job_id: str) -> None:
    # This is the cancellation cutoff. No child computation is running now.
    with _state_lock(base):
        _check_cancel(base, job_id)
        job = _latest_record(base, job_id)
        if not job or job["status"] != "running":
            raise ValueError("Status job tidak valid untuk menyimpan model.")
        job.update(status="publishing", progress=0.97,
                   message="Memeriksa dan mengaktifkan model baru...", updated_at=_now())
        _JOBS[(str(base), job_id)] = job
        _write_job(base, job)


def _drain_output(stream, tail: bytearray) -> None:
    try:
        while chunk := stream.read(4096):
            tail.extend(chunk)
            del tail[:-_LOG_LIMIT]
    except (OSError, ValueError):
        pass


def _stop_worker(process: subprocess.Popen, work: Path) -> None:
    if process.stdin:
        process.stdin.close()
    try:
        process.wait(timeout=1.0)
    except subprocess.TimeoutExpired:
        if _worker_pid(work) == process.pid:
            process.terminate()
            try:
                process.wait(timeout=3.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=3.0)
        else:
            process.wait()


def _compute(base: Path, job_id: str, version: str, raw_data: pd.DataFrame, config) -> tuple[dict, dict]:
    work = _job_directory(base) / f".work-{job_id}"
    work.mkdir(exist_ok=False)
    process = None
    reader = None
    tail = bytearray()
    try:
        _check_cancel(base, job_id)
        joblib.dump({"raw_data": raw_data, "version": version, "config": config}, work / "input.joblib")
        _check_cancel(base, job_id)
        worker = Path(__file__).resolve().with_name("training_worker.py")
        process = subprocess.Popen(
            [sys.executable, "-B", str(worker), "--work-dir", str(work)],
            cwd=str(worker.parent), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        reader = threading.Thread(target=_drain_output, args=(process.stdout, tail), daemon=True)
        reader.start()
        _update(base, job_id, launcher_pid=process.pid)
        last_sequence = -1
        reported_pid = None
        while process.poll() is None:
            _check_cancel(base, job_id)
            actual_pid = _worker_pid(work)
            if actual_pid and actual_pid != reported_pid:
                _update(base, job_id, worker_pid=actual_pid)
                reported_pid = actual_pid
            try:
                event = json.loads((work / "progress.json").read_text(encoding="utf-8"))
                if int(event["sequence"]) != last_sequence:
                    _update(base, job_id,
                            progress=min(0.96, max(0.01, float(event["fraction"])) * 0.96),
                            message=str(event["message"])[:500])
                    last_sequence = int(event["sequence"])
            except (OSError, ValueError, KeyError, TypeError):
                pass
            try:
                process.wait(timeout=0.15)
            except subprocess.TimeoutExpired:
                pass
        _check_cancel(base, job_id)
        if process.returncode != 0:
            reader.join(timeout=2)
            try:
                detail = json.loads((work / "error.json").read_text(encoding="utf-8"))
                message = str(detail.get("error", "Proses pelatihan gagal."))
                trace = str(detail.get("traceback", ""))
            except (OSError, ValueError, TypeError):
                message, trace = f"Proses pelatihan berhenti (kode {process.returncode}).", ""
            output = tail.decode("utf-8", errors="replace")
            raise RuntimeError(f"{message[:8_000]}\n{trace[-6_000:]}\n{output[-6_000:]}")
        bundle, metrics = joblib.load(work / "result.joblib")
        _check_cancel(base, job_id)
        return bundle, metrics
    finally:
        if process:
            try:
                _stop_worker(process, work)
            finally:
                worker_pid = _worker_pid(work)
                while process.poll() is None or (worker_pid and _process_alive(worker_pid)):
                    time.sleep(0.05)
                    worker_pid = _worker_pid(work) or worker_pid
            if reader:
                reader.join(timeout=2)
            if process.stdout:
                process.stdout.close()
        shutil.rmtree(work, ignore_errors=False)


def _finish(base: Path, job_id: str, **changes) -> None:
    try:
        _update(base, job_id, finished_at=_now(), **changes)
    except OSError as error:
        with _GUARD:
            job = _JOBS.get((str(base), job_id), {"id": job_id})
            job.update(changes, finished_at=_now(), updated_at=_now(), status_save_warning=str(error))
            _JOBS[(str(base), job_id)] = job
    finally:
        _cancel_path(base, job_id).unlink(missing_ok=True)


def _run(base: Path, job_id: str, raw_data: pd.DataFrame, config) -> None:
    published_record = None
    try:
        with training_lock(base):
            _check_cancel(base, job_id)
            version = reserve_version(base)
            _update(base, job_id, status="running", version=version, started_at=_now(), message="Menyiapkan data dan target...")
            bundle, metrics = _compute(base, job_id, version, raw_data, config)
            _begin_publishing(base, job_id)
            published_record = publish_version(base, version, bundle, metrics, raw_data)
        _finish(base, job_id, status="completed", progress=1.0,
                message=f"Model {version} berhasil diaktifkan.", record=published_record)
    except _Cancelled:
        _finish(base, job_id, status="cancelled",
                message="Retraining dibatalkan. Model aktif tetap menggunakan versi sebelumnya.")
    except Exception as error:
        if published_record is not None:
            _finish(base, job_id, status="completed", progress=1.0, record=published_record,
                    message=f"Model {published_record['version']} berhasil diaktifkan.", status_save_warning=str(error))
        elif _cancel_path(base, job_id).is_file():
            _finish(base, job_id, status="cancelled",
                    message="Retraining dibatalkan. Model aktif tetap menggunakan versi sebelumnya.")
        else:
            _finish(base, job_id, status="failed", message="Retraining gagal. Model aktif tetap tersedia.",
                    error=str(error)[-20_000:], error_detail=traceback.format_exc()[-24_000:])
