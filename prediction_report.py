from __future__ import annotations

from io import BytesIO
import math
from pathlib import Path
from threading import RLock

import matplotlib.dates as mdates
from matplotlib.figure import Figure
from matplotlib.image import imread
from matplotlib.font_manager import FontProperties
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch
from matplotlib.textpath import TextToPath
from matplotlib.ticker import FuncFormatter, MaxNLocator
import numpy as np
import pandas as pd


DISCLAIMER_TEXT = (
    "Aplikasi ini merupakan proof of concept untuk penelitian. Hasil prediksi "
    "dapat berbeda dari harga aktual dan bukan merupakan nasihat keuangan "
    "atau rekomendasi investasi."
)

_RED = "#b91c1c"
_INK = "#171717"
_MUTED = "#525252"
_BORDER = "#e5e7eb"
_BLUE = "#2563eb"
_UP = "#15803d"
_DOWN = "#dc2626"
_MONTHS = ("Jan", "Feb", "Mar", "Apr", "Mei", "Jun", "Jul", "Agu", "Sep", "Okt", "Nov", "Des")
_PDF_LOCK = RLock()


def _date_label(value: object) -> str:
    stamp = pd.Timestamp(value)
    if pd.isna(stamp):
        raise ValueError("Tanggal hasil prediksi tidak valid.")
    return f"{stamp.day:02d} {_MONTHS[stamp.month - 1]} {stamp.year}"


def _short_text(value: object, maximum: int) -> str:
    """Bound labels and strip control characters before drawing them."""
    text = " ".join(str(value).split())
    text = "".join(character for character in text if character.isprintable())
    return text if len(text) <= maximum else text[: maximum - 3].rstrip() + "..."


def _number(value: object, label: str, *, positive: bool = False) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{label} pada hasil prediksi tidak valid.") from error
    if not math.isfinite(number) or (positive and number <= 0):
        raise ValueError(f"{label} pada hasil prediksi tidak valid.")
    return number


def _fit_label(value: object, width_points: float, fontsize: float, weight: str = "normal") -> str:
    """Fit user/model labels to their allocated physical width, even wide glyphs."""
    text = _short_text(value, 120)
    properties = FontProperties(family="DejaVu Sans", size=fontsize, weight=weight)
    measuring = TextToPath()
    if measuring.get_text_width_height_descent(text, properties, ismath=False)[0] <= width_points:
        return text
    while text and measuring.get_text_width_height_descent(text + "...", properties, ismath=False)[0] > width_points:
        text = text[:-1]
    return text.rstrip() + "..."


def _validated_result(snapshot: dict) -> tuple[pd.Timestamp, float, pd.DataFrame, list[dict]]:
    try:
        origin = pd.Timestamp(snapshot["origin_date"])
        if pd.isna(origin) or origin.tzinfo is not None:
            raise ValueError("Tanggal acuan tidak valid.")
        current = _number(snapshot["current_price"], "Harga acuan", positive=True)
        history = snapshot["history"].loc[:, ["coal"]].copy().tail(60)
        if history.empty or not isinstance(history.index, pd.DatetimeIndex):
            raise ValueError("Riwayat harga untuk laporan belum tersedia.")
        if history.index.hasnans or history.index.has_duplicates or history.index.tz is not None:
            raise ValueError("Tanggal riwayat harga tidak valid.")
        history = history.sort_index()
        history["coal"] = pd.to_numeric(history["coal"], errors="raise")
        if not np.isfinite(history["coal"].to_numpy(dtype=float)).all():
            raise ValueError("Riwayat harga pada laporan tidak valid.")
        if history.index.max() != origin or not np.isclose(float(history.iloc[-1, 0]), current):
            raise ValueError("Riwayat harga tidak sesuai dengan tanggal dan harga acuan.")
        results_by_horizon = {}
        for raw in snapshot["results"]:
            result = dict(raw)
            horizon = int(result["h"])
            if horizon not in (1, 7, 14) or horizon in results_by_horizon:
                raise ValueError("Horizon hasil prediksi tidak lengkap atau duplikat.")
            result["h"] = horizon
            result["predicted_price"] = _number(result["predicted_price"], "Harga prediksi", positive=True)
            result["change_usd"] = _number(result["change_usd"], "Perubahan harga")
            result["change_pct"] = _number(result["change_pct"], "Persentase perubahan")
            result["forecast_date"] = pd.Timestamp(result["forecast_date"])
            if pd.isna(result["forecast_date"]) or result["forecast_date"].tzinfo is not None or result["forecast_date"] <= origin:
                raise ValueError("Tanggal tujuan hasil prediksi tidak valid.")
            results_by_horizon[horizon] = result
        if set(results_by_horizon) != {1, 7, 14}:
            raise ValueError("Laporan membutuhkan hasil H+1, H+7, dan H+14.")
        return origin, current, history, [results_by_horizon[h] for h in (1, 7, 14)]
    except (KeyError, AttributeError, TypeError, OverflowError) as error:
        raise ValueError("Hasil prediksi belum lengkap untuk dibuat menjadi PDF.") from error


def _text(figure: Figure, x: float, y: float, text: str, **options) -> None:
    options.setdefault("color", _INK)
    figure.text(
        x, y, text, fontfamily="DejaVu Sans",
        va="baseline", parse_math=False, **options,
    )


def _draw_logo(figure: Figure) -> None:
    width = 0.28
    height = width * figure.get_figwidth() / figure.get_figheight() / 4.5
    axis = figure.add_axes([0.062, 0.909, width, height], label="nwcast_logo")
    axis.imshow(imread(Path(__file__).resolve().parent / "Assets" / "nwcast-logo.png"),
                origin="upper", aspect="equal", interpolation="lanczos")
    axis.set_xlim(420, 1500)
    axis.set_ylim(630, 390)
    axis.set_axis_off()


def build_prediction_pdf(snapshot: dict) -> bytes:
    """Return one A4 PDF with all three forecasts and their actual web data.

    The history and forecast segments are the same as the web graph: up to 60
    supplied observations, followed by the straight origin-to-target segment.
    A segment is not a daily forecast between those two dates.
    """
    origin, current, history, results = _validated_result(snapshot)
    with _PDF_LOCK:
        figure = Figure(figsize=(210 / 25.4, 297 / 25.4), facecolor="white")
        try:
            _draw_logo(figure)
            _text(figure, 0.07, 0.899, "RINGKASAN HASIL PREDIKSI", fontsize=9, fontweight="bold", alpha=0.72)
            meta = (
                (0.07, "TANGGAL ACUAN", _date_label(origin)),
                (0.365, "HARGA ACUAN", f"${current:,.2f} / tonne"),
                (0.66, "VERSI MODEL", _fit_label(snapshot.get("version", "Tidak tercatat"), 154, 10.2, "bold")),
            )
            for x, label, value in meta:
                _text(figure, x, 0.867, label, fontsize=7.4, fontweight="bold", alpha=0.63)
                _text(figure, x, 0.847, value, fontsize=10.2, fontweight="bold")

            for x, result in zip((0.07, 0.365, 0.66), results):
                trend = _UP if result["change_usd"] >= 0 else _DOWN
                card = FancyBboxPatch(
                    (x, 0.735), 0.27, 0.092, boxstyle="round,pad=0.008,rounding_size=0.008",
                    transform=figure.transFigure, linewidth=0.8, edgecolor=_BORDER,
                    facecolor="#fafafa", zorder=-1,
                )
                figure.add_artist(card)
                _text(figure, x + 0.012, 0.805, f"H+{result['h']}", fontsize=9.4, fontweight="bold")
                _text(figure, x + 0.012, 0.782, f"${result['predicted_price']:,.2f}", fontsize=14, fontweight="bold")
                _text(figure, x + 0.012, 0.762,
                      f"{result['change_usd']:+,.2f} USD  |  {result['change_pct']:+.2f}%",
                      fontsize=8.2, fontweight="bold", color=trend)
                _text(figure, x + 0.012, 0.744, _date_label(result["forecast_date"]), fontsize=8, alpha=0.72)

            for top, result in zip((0.704, 0.492, 0.280), results):
                horizon = result["h"]
                trend = _UP if result["change_usd"] >= 0 else _DOWN
                _text(figure, 0.07, top, f"H+{horizon}  |  {_date_label(result['forecast_date'])}",
                      fontsize=9.3, fontweight="bold")
                _text(figure, 0.93, top, _fit_label(f"Model: {result.get('model_name', '-')}", 246, 8),
                      fontsize=8, ha="right", alpha=0.72)
                axis = figure.add_axes([0.105, top - 0.159, 0.815, 0.139], facecolor="white")
                axis.plot(history.index.to_pydatetime(), history["coal"].to_numpy(dtype=float),
                          color=_BLUE, linewidth=1.6, label="Harga historis", zorder=3)
                axis.plot([origin.to_pydatetime(), result["forecast_date"].to_pydatetime()],
                          [current, result["predicted_price"]], color=trend, linewidth=1.8,
                          marker="o", markersize=3.4, label=f"Prediksi H+{horizon}", zorder=4)
                axis.set_ylabel("USD/tonne", color=_MUTED, fontsize=7.4, labelpad=8)
                axis.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=4, maxticks=6))
                axis.xaxis.set_major_formatter(FuncFormatter(
                    lambda value, position: f"{mdates.num2date(value).day:02d} {_MONTHS[mdates.num2date(value).month - 1]}"
                ))
                axis.yaxis.set_major_locator(MaxNLocator(nbins=4, min_n_ticks=3))
                axis.tick_params(axis="both", colors=_MUTED, labelsize=7.4, length=0, pad=5)
                axis.grid(axis="y", color=_BORDER, linewidth=0.7, zorder=0)
                axis.spines[["top", "right", "left"]].set_visible(False)
                axis.spines["bottom"].set_color(_BORDER)
                axis.margins(x=0.025, y=0.18)
                axis.legend(loc="upper left", bbox_to_anchor=(0, 1.03), frameon=False,
                            ncols=2, fontsize=6.8, handlelength=2, borderaxespad=0)

            _text(figure, 0.07, 0.080,
                  "Horizon mengikuti kalender Senin-Jumat. Garis prediksi menghubungkan harga acuan dan estimasi tujuan.",
                  fontsize=6.9, alpha=0.78)
            _text(figure, 0.07, 0.063,
                  "Bukan prediksi harian di antara kedua tanggal tersebut.", fontsize=6.9, alpha=0.78)
            figure.add_artist(Line2D([0.07, 0.93], [0.053, 0.053], transform=figure.transFigure,
                                     color=_BORDER, linewidth=0.8))
            _text(figure, 0.07, 0.037,
                  "Aplikasi ini merupakan proof of concept untuk penelitian. Hasil prediksi dapat berbeda dari harga aktual",
                  fontsize=7.2, alpha=0.85)
            _text(figure, 0.07, 0.024,
                  "dan bukan merupakan nasihat keuangan atau rekomendasi investasi.", fontsize=7.2, alpha=0.85)
            output = BytesIO()
            figure.savefig(
                output, format="pdf", facecolor="white",
                metadata={
                    "Title": "Newcastle Coal Prediction - Ringkasan Hasil",
                    "Author": "Newcastle Coal Prediction App",
                    "Subject": "Ringkasan tiga horizon prediksi untuk penelitian",
                },
            )
            return output.getvalue()
        finally:
            figure.clear()
