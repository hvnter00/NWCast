from __future__ import annotations
import csv
import math
import numbers
import re
import unicodedata
from datetime import date, datetime
from io import BytesIO, StringIO
from pathlib import Path
from typing import Iterable
import numpy as np
import pandas as pd


MIN_OBSERVATIONS = 31
INPUT_COLUMNS = ["Date", "Coal", "Oil", "Gas", "DXY"]
PRICE_COLUMNS = ["coal", "wti", "gas", "dxy"]
_COLUMN_NAMES = {"date": "Date", "coal": "Coal", "wti": "Oil", "gas": "Gas", "dxy": "DXY"}
_ALIASES = {
    "date": "date",
    "tanggal": "date",
    "coal": "coal",
    "oil": "wti",
    "wti": "wti",
    "gas": "gas",
    "henryhub": "gas",
    "dxy": "dxy",
}
_EXCEL_DEPENDENCY = (
    "Dukungan Excel belum tersedia. Install openpyxl>=3.1.5 "
    "di virtual environment aplikasi, lalu jalankan kembali."
)


def read_upload(payload: bytes, filename: str) -> pd.DataFrame:
    """Baca satu CSV atau XLSX tanpa mengubah nama kolom/nilai input."""
    if not payload:
        raise ValueError("File kosong. Unggah CSV atau Excel yang berisi data.")

    suffix = Path(filename).suffix.lower()
    if suffix == ".xlsx":
        try:
            with pd.ExcelFile(BytesIO(payload), engine="openpyxl") as workbook:
                sheet = "Input" if "Input" in workbook.sheet_names else workbook.sheet_names[0]
                return pd.read_excel(
                    workbook,
                    sheet_name=sheet,
                    dtype=object,
                    keep_default_na=False,
                )
        except ImportError as error:
            raise ValueError(_EXCEL_DEPENDENCY) from error
        except Exception as error:
            raise ValueError(
                "File Excel tidak dapat dibaca. Gunakan file .xlsx yang valid "
                "dengan kolom Date, Coal, Oil, Gas, dan DXY."
            ) from error

    if suffix != ".csv":
        raise ValueError("Format belum didukung. Unggah satu file .xlsx atau .csv.")

    try:
        if payload.startswith((b"\xff\xfe", b"\xfe\xff")):
            decoded = payload.decode("utf-16")
        else:
            try:
                decoded = payload.decode("utf-8-sig")
            except UnicodeDecodeError:
                decoded = payload.decode("cp1252")
        if not decoded.strip():
            raise ValueError("File CSV kosong.")
        try:
            delimiter = csv.Sniffer().sniff(decoded[:16384], delimiters=",;\t|").delimiter
        except csv.Error:
            delimiter = ";" if ";" in decoded.splitlines()[0] else ","
        return pd.read_csv(
            StringIO(decoded),
            sep=delimiter,
            dtype=object,
            keep_default_na=False,
            skip_blank_lines=True,
        )
    except (UnicodeError, pd.errors.ParserError, pd.errors.EmptyDataError, csv.Error) as error:
        raise ValueError(
            "File CSV tidak dapat dibaca. Gunakan kolom Date, Coal, Oil, Gas, dan DXY."
        ) from error


def _blank(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    missing = pd.isna(value)
    return bool(missing) if np.isscalar(missing) else False


def _heading(value: object) -> str:
    normalized = unicodedata.normalize("NFKC", str(value)).casefold()
    return re.sub(r"[^a-z0-9]", "", normalized)


def _date_value(value: object) -> pd.Timestamp:
    if isinstance(value, (pd.Timestamp, datetime, date, np.datetime64)):
        timestamp = pd.Timestamp(value)
        if timestamp.tzinfo is not None:
            raise ValueError("Tanggal tidak boleh memiliki zona waktu.")
        if timestamp != timestamp.normalize():
            raise ValueError("Gunakan tanggal tanpa jam.")
        return timestamp.normalize()
    if not isinstance(value, str):
        raise ValueError("Gunakan tanggal Excel, YYYY-MM-DD, atau DD/MM/YYYY; bukan nomor serial.")

    value = value.strip()
    formats = (
        (r"\d{4}-\d{2}-\d{2}", "%Y-%m-%d"),
        (r"\d{1,2}/\d{1,2}/\d{4}", "%d/%m/%Y"),
        (r"\d{1,2}\s+\d{1,2}\s+\d{4}", "%d %m %Y"),
    )
    for pattern, date_format in formats:
        if re.fullmatch(pattern, value):
            try:
                return pd.Timestamp(datetime.strptime(value, date_format))
            except ValueError as error:
                raise ValueError("Tanggal tidak valid.") from error
    raise ValueError("Gunakan tanggal Excel, YYYY-MM-DD, atau DD/MM/YYYY.")


def _number_value(value: object) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError("Gunakan angka harga.")
    if isinstance(value, numbers.Real):
        result = float(value)
    elif isinstance(value, str):
        number = value.strip().replace("\u00a0", " ").replace("\u202f", " ")
        if " " in number:
            if not re.fullmatch(r"[+-]?\d{1,3}(?: \d{3})+(?:[.,]\d+)?", number):
                raise ValueError("Pemisah angka tidak valid.")
            number = number.replace(" ", "")

        if "," in number and "." in number:
            decimal = "," if number.rfind(",") > number.rfind(".") else "."
            grouping = "." if decimal == "," else ","
            if number.count(decimal) != 1:
                raise ValueError("Pemisah angka tidak valid.")
            integer, fraction = number.rsplit(decimal, 1)
            grouped = rf"[+-]?\d{{1,3}}(?:{re.escape(grouping)}\d{{3}})+"
            if not re.fullmatch(grouped, integer) or not re.fullmatch(r"\d+", fraction):
                raise ValueError("Pemisah ribuan/desimal tidak valid.")
            number = integer.replace(grouping, "") + "." + fraction
        elif number.count(",") > 1 or number.count(".") > 1:
            grouping = "," if "," in number else "."
            if not re.fullmatch(rf"[+-]?\d{{1,3}}(?:{re.escape(grouping)}\d{{3}})+", number):
                raise ValueError("Pemisah ribuan tidak valid.")
            number = number.replace(grouping, "")
        else:
            number = number.replace(",", ".")
        if not re.fullmatch(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?", number):
            raise ValueError("Gunakan angka harga tanpa simbol mata uang.")
        result = float(number)
    else:
        raise ValueError("Gunakan angka harga.")

    if not math.isfinite(result):
        raise ValueError("Harga harus berupa angka finite; bukan NaN atau infinity.")
    return result


def validate_and_prepare(frame: pd.DataFrame) -> pd.DataFrame:
    """Validasi >=31 tanggal unik dan isi harga kosong dari tanggal sebelumnya."""
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ValueError("Isi minimal 31 tanggal pengamatan.")

    selected = {}
    for column in frame.columns:
        canonical = _ALIASES.get(_heading(column))
        if canonical is None:
            if frame[column].map(_blank).all():
                continue
            raise ValueError(f"Kolom tidak dikenal: {column}. Gunakan Date, Coal, Oil, Gas, dan DXY.")
        if canonical in selected:
            raise ValueError(f"Kolom {_COLUMN_NAMES[canonical]} muncul lebih dari sekali.")
        selected[canonical] = column

    missing = [key for key in ["date", *PRICE_COLUMNS] if key not in selected]
    if missing:
        names = ", ".join(_COLUMN_NAMES[key] for key in missing)
        raise ValueError(f"Kolom belum lengkap: {names}.")

    ordered_columns = ["date", *PRICE_COLUMNS]
    rows = frame.loc[:, [selected[key] for key in ordered_columns]].copy()
    rows.columns = ordered_columns
    parsed = []
    for row_number, (_, row) in enumerate(rows.iterrows(), start=1):
        blanks = [_blank(row[key]) for key in ordered_columns]
        if all(blanks):
            continue
        if blanks[0]:
            raise ValueError(f"Baris data {row_number}: lengkapi tanggal.")
        try:
            timestamp = _date_value(row["date"])
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError(f"Baris data {row_number}, Date: {error}") from error
        if timestamp.dayofweek >= 5:
            raise ValueError(f"Baris data {row_number}: {timestamp:%Y-%m-%d} jatuh pada akhir pekan. Gunakan Senin–Jumat.")
        values = {}
        for key in PRICE_COLUMNS:
            if _blank(row[key]):
                values[key] = np.nan
                continue
            try:
                values[key] = _number_value(row[key])
            except (TypeError, ValueError, OverflowError) as error:
                raise ValueError(f"Baris data {row_number}, {_COLUMN_NAMES[key]}: {error}") from error
        parsed.append({"Date": timestamp, **values})

    if not parsed:
        raise ValueError("Isi minimal 31 tanggal pengamatan.")
    result = pd.DataFrame(parsed).set_index("Date").sort_index()
    if result.index.has_duplicates:
        duplicated = result[result.index.duplicated(keep=False)]
        conflicts = duplicated.groupby(level=0).nunique().gt(1).any(axis=1)
        if conflicts.any():
            first = conflicts[conflicts].index[0]
            raise ValueError(f"Tanggal {first:%Y-%m-%d} memiliki harga yang berbeda pada baris duplikat.")
        result = result.groupby(level=0).first()
    if len(result) < MIN_OBSERVATIONS:
        raise ValueError(
            f"Tersedia {len(result)} tanggal unik. Diperlukan minimal 31."
        )
    filled_rows = int(result.isna().any(axis=1).sum())
    result = result.ffill()
    if result.isna().any().any():
        raise ValueError("Lengkapi harga pada tanggal paling awal.")
    recent = result.tail(MIN_OBSERVATIONS)
    nonpositive = recent.le(0)
    if nonpositive.any().any():
        bad_date, bad_column = next(
            (timestamp, column)
            for timestamp, row in nonpositive.iterrows()
            for column, invalid in row.items()
            if invalid
        )
        raise ValueError(
            f"Harga {_COLUMN_NAMES[bad_column]} pada {bad_date:%Y-%m-%d} harus lebih dari 0. "
            "Seluruh harga pada 31 pengamatan terbaru harus positif untuk fitur log."
        )
    prepared = result.loc[:, PRICE_COLUMNS].astype(float)
    prepared.attrs["filled_price_rows"] = filled_rows
    return prepared


def make_template(format: str = "xlsx", dates: Iterable[object] | None = None) -> bytes:
    """Template 31 tanggal; harga kosong memakai harga dari tanggal sebelumnya."""
    format = format.lower().lstrip(".")
    if dates is None:
        date_values = [None] * MIN_OBSERVATIONS
    else:
        date_values = list(dates)
        if len(date_values) != MIN_OBSERVATIONS:
            raise ValueError("Template harus berisi tepat 31 tanggal.")
        date_values = [None if _blank(value) else _date_value(value).to_pydatetime() for value in date_values]
    template = pd.DataFrame({"Date": date_values})
    for name in INPUT_COLUMNS[1:]:
        template[name] = None

    if format == "csv":
        # Serialize dates explicitly so object/datetime dtype changes cannot add a time.
        template["Date"] = ["" if value is None else value.strftime("%Y-%m-%d") for value in date_values]
        return template.to_csv(index=False).encode("utf-8-sig")
    if format != "xlsx":
        raise ValueError("Format template harus xlsx atau csv.")

    instructions = pd.DataFrame({
        "Panduan": [
            "Isi sheet Input; jangan mengubah lima judul kolom.",
            "Masukkan minimal 31 tanggal pengamatan unik, masing-masing satu tanggal Senin–Jumat.",
            "Tanggal boleh diurutkan naik atau turun. Gunakan tanggal Excel, YYYY-MM-DD, atau DD/MM/YYYY.",
            "Harga kosong memakai harga sebelumnya setelah tanggal diurutkan. Lengkapi Coal, Oil, Gas, dan DXY pada tanggal paling awal.",
            "Harga terisi harus berupa angka finite; harga pada 31 pengamatan terbaru harus lebih dari 0 untuk fitur log.",
            "Harga historis nol/negatif di luar 31 pengamatan terbaru dipertahankan; tidak diganti secara otomatis.",
            "Gunakan titik atau koma untuk satu tanda desimal. Contoh 3.125 dan 3,125 berarti 3.125.",
            "Pemisah ribuan boleh memakai format 1,234.56 / 1.234,56 atau spasi 1 234,56.",
            "Jangan masukkan simbol mata uang atau satuan pada sel harga.",
            "Baris yang seluruhnya kosong diabaikan; tanggal wajib diisi. Tanggal dengan seluruh harga kosong tetap diterima.",
            "Duplikat yang saling melengkapi digabung menjadi satu tanggal; harga berbeda pada tanggal yang sama ditolak.",
            "Tanggal bukan hari bursa resmi; aplikasi mengikuti kalender Senin–Jumat dari notebook.",
            "Data unggahan dipakai untuk prediksi ini dan tidak menambah dataset pelatihan sistem.",
        ]
    })
    output = BytesIO()
    try:
        with pd.ExcelWriter(output, engine="openpyxl", date_format="yyyy-mm-dd", datetime_format="yyyy-mm-dd") as writer:
            template.to_excel(writer, sheet_name="Input", index=False)
            instructions.to_excel(writer, sheet_name="Panduan", index=False)
            sheet = writer.sheets["Input"]
            sheet.freeze_panes = "A2"
            sheet.auto_filter.ref = "A1:E32"
            for row in sheet.iter_rows(min_row=2, max_row=32, min_col=1, max_col=1):
                row[0].number_format = "yyyy-mm-dd"
            for name in ("A", "B", "C", "D", "E"):
                sheet.column_dimensions[name].width = 18
            for row in sheet.iter_rows(min_row=2, max_row=32, min_col=2, max_col=5):
                for cell in row:
                    cell.number_format = "General"
            writer.sheets["Panduan"].column_dimensions["A"].width = 125
    except ImportError as error:
        raise ValueError(_EXCEL_DEPENDENCY) from error
    return output.getvalue()
