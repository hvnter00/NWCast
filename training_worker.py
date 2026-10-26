from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import threading
import time
import traceback
from uuid import uuid4


def _watch_supervisor() -> None:
    try:
        if os.name == "nt":
            import ctypes
            from ctypes import wintypes
            import msvcrt

            kernel = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel.ReadFile.argtypes = [wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD,
                                        ctypes.POINTER(wintypes.DWORD), ctypes.c_void_p]
            kernel.ReadFile.restype = wintypes.BOOL
            received = wintypes.DWORD()
            buffer = ctypes.create_string_buffer(1)
            kernel.ReadFile(msvcrt.get_osfhandle(sys.stdin.fileno()), buffer, 1,
                            ctypes.byref(received), None)
        else:
            os.read(sys.stdin.fileno(), 1)
    finally:
        os._exit(93)


def _write_json(path: Path, content: dict) -> None:
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(content, ensure_ascii=False, allow_nan=False), encoding="utf-8")
        for attempt in range(20):
            try:
                os.replace(temporary, path)
                break
            except PermissionError:
                if attempt == 19:
                    raise
                # A Windows status reader can briefly hold the target open.
                time.sleep(0.01)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-dir", type=Path, required=True)
    args = parser.parse_args()
    work = args.work_dir.resolve()
    try:
        _write_json(work / "worker.json", {"pid": os.getpid()})
        import joblib

        from training import train_models
        from training_data import prepare_training_prices

        threading.Thread(target=_watch_supervisor, daemon=True).start()
        payload = joblib.load(work / "input.joblib")
        sequence = 0

        def report(fraction: float, message: str) -> None:
            nonlocal sequence
            sequence += 1
            _write_json(work / "progress.json", {
                "sequence": sequence, "fraction": float(fraction), "message": str(message)[:500],
            })

        prices = prepare_training_prices(payload["raw_data"])
        result = train_models(prices, payload["version"], progress=report, config=payload["config"])
        temporary = work / ".result.joblib.tmp"
        joblib.dump(result, temporary, compress=3)
        os.replace(temporary, work / "result.joblib")
        return 0
    except Exception as error:
        _write_json(work / "error.json", {
            "error": str(error)[-8_000:], "traceback": traceback.format_exc()[-16_000:],
        })
        return 1


if __name__ == "__main__":
    raise SystemExit(main())