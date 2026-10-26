from base64 import b64encode
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st
from model_registry import active_dataset_path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "Data" / "Historical"

DATASETS = (
    (
        "Coal.csv",
        "Newcastle Coal Futures Price",
        "USD/tonne",
        "https://id.investing.com/commodities/newcastle-coal-futures-streaming-chart",
    ),
    (
        "WTI OIL.csv",
        "WTI Crude Oil Spot Price",
        "USD/barrel",
        "https://fred.stlouisfed.org/series/DCOILWTICO",
    ),
    (
        "HenryHub.csv",
        "Henry Hub Natural Gas Spot Price",
        "USD/MMBtu",
        "https://www.eia.gov/dnav/ng/ng_pri_fut_s1_d.htm",
    ),
    (
        "DXY.csv",
        "US Dollar Index (DXY)",
        "Index",
        "https://id.investing.com/indices/usdollar?cid=1224074",
    ),
)

@st.cache_data(show_spinner=False)
def read_history(file_path: str, modified_ns: int) -> pd.DataFrame:
    raw = pd.read_csv(file_path, sep=None, engine="python", dtype=str)
    if len(raw.columns) != 2:
        raise ValueError("CSV harus memiliki dua kolom: tanggal dan harga.")

    date_text = raw.iloc[:, 0].str.strip()
    dates = pd.Series(pd.NaT, index=raw.index, dtype="datetime64[ns]")
    for date_format in ("%d %m %Y", "%b %d, %Y", "%Y-%m-%d", "%d/%m/%Y"):
        parsed = pd.to_datetime(date_text, format=date_format, errors="coerce")
        dates = dates.fillna(parsed)

    price_text = raw.iloc[:, 1].str.strip().str.replace(",", ".", regex=False)
    prices = pd.to_numeric(price_text, errors="coerce")
    history = pd.DataFrame({"Date": dates, "Price": prices})
    history = history.dropna(subset=["Date", "Price"])
    history = history.loc[history["Price"].abs() != float("inf")]
    history = history.drop_duplicates("Date", keep="last").sort_values("Date")
    if history.empty:
        raise ValueError("Tidak ada tanggal dan harga valid di dalam CSV.")
    return history.reset_index(drop=True)


@st.cache_data(show_spinner=False)
def read_version_history(file_path: str, modified_ns: int, variable: str) -> pd.DataFrame:
    raw = pd.read_csv(file_path)
    history = pd.DataFrame({
        "Date": pd.to_datetime(raw["Date"], format="%Y-%m-%d", errors="raise"),
        "Price": pd.to_numeric(raw[variable], errors="raise"),
    }).dropna()
    history = history.loc[history["Price"].abs() != float("inf")].sort_values("Date")
    if history.empty:
        raise ValueError("Dataset aktif belum memiliki harga untuk variabel ini.")
    return history.reset_index(drop=True)


def history_chart(history: pd.DataFrame, unit: str) -> alt.Chart:
    return (
        alt.Chart(history)
        .mark_line(color="#2563eb", strokeWidth=2)
        .encode(
            x=alt.X(
                "Date:T",
                title=None,
                axis=alt.Axis(format="%Y", tickCount=7, grid=False, labelAngle=0),
            ),
            y=alt.Y(
                "Price:Q",
                title=unit,
                scale=alt.Scale(zero=False),
                axis=alt.Axis(tickCount=5),
            ),
            tooltip=[
                alt.Tooltip("Date:T", title="Date", format="%d %b %Y"),
                alt.Tooltip("Price:Q", title=unit, format=",.2f"),
            ],
        )
        .properties(height=190, background="#ffffff")
        .configure_view(stroke=None)
        .configure_axis(
            labelColor="#657080",
            titleColor="#17202b",
            gridColor="#e9ecf0",
            gridOpacity=1,
            domainColor="#dce0e6",
            tickColor="#dce0e6",
            labelFontSize=11,
            titleFontSize=11,
            labelPadding=7,
        )
        .interactive()
    )


def show_dataset(filename: str, title: str, unit: str, source_url: str, published_path: Path | None = None) -> None:
    path = DATA_DIR / filename

    try:
        if published_path is None:
            history = read_history(str(path), path.stat().st_mtime_ns)
        else:
            variable = {"Coal.csv": "coal", "WTI OIL.csv": "wti", "HenryHub.csv": "gas", "DXY.csv": "dxy"}[filename]
            history = read_version_history(str(published_path), published_path.stat().st_mtime_ns, variable)
        latest = history["Date"].max().strftime("%d %B %Y")

        with st.container(gap=None, border=True):
            st.altair_chart(
                history_chart(history, unit),
                theme=None,
                width="stretch",
            )

            st.html(
                f"""
                <div class="chart-caption">
                    <p class="chart-label">
                        <a href="{source_url}"
                           target="_blank"
                           rel="noopener noreferrer">
                            {title}
                        </a>
                    </p>
                    <p class="chart-date">
                        System Latest Data: {latest}
                    </p>
                </div>
                """
            )

    except (OSError, ValueError, pd.errors.ParserError) as error:
        st.error(f"{title}: {error}")


def render() -> None:
    logo = b64encode((BASE_DIR / "Assets" / "nwcast-logo.png").read_bytes()).decode("ascii")
    st.html(
        '<h1 class="home-title"><span class="home-logo">'
        f'<img src="data:image/png;base64,{logo}" alt="NWCast" width="1920" height="1080">'
        '</span></h1>'
    )
    try:
        published_path = active_dataset_path(BASE_DIR)
    except (OSError, ValueError) as error:
        st.error(str(error))
        return
    for row_start in (0, 2):
        columns = st.columns(2, gap="large")
        for column, dataset in zip(columns, DATASETS[row_start : row_start + 2]):
            with column:
                show_dataset(*dataset, published_path=published_path)


