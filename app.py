"""
Advanced VaR & Expected Shortfall Engine

Computes market risk (and upside) for a single stock from historical data.

VaR / ES     downside: how much you can lose in a day, and how much on a bad day.
Upside / EU  upside: how much you can gain in a day, and how much on a great day.

All math runs on numpy, no loops. Data is fetched once and cached,
so the sliders and the language switch stay instant.

Run:
    pip install -r requirements.txt
    streamlit run app.py
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

# colors and constants
CORP_BLUE = "#1F6FEB"        # blue for the histogram
TAIL_RED = "#E5484D"         # red for the VaR line
TAIL_RED_DARK = "#8B0000"    # dark red for the ES line
TAIL_GREEN = "#2FB344"       # green for the Upside line
TAIL_GREEN_DARK = "#1A7431"  # dark green for the Expected Upside line
TRADING_DAYS = 252           # trading days per year, used for annual figures


# --------------------------------------------------------------------------- #
#  Localization. All user facing text lives here, keyed by language.          #
#  The app just picks TEXTS[lang] at render time, so switching language is     #
#  a plain rerun with no data reload.                                          #
# --------------------------------------------------------------------------- #
TEXTS = {
    "EN": {
        "lang_label": "🌐 Language / Язык",
        "title_caption": "Market risk & upside from historical data · numpy · transparent math",
        # sidebar
        "sidebar_header": "⚙️ Risk parameters",
        "ticker_label": "Asset ticker",
        "ticker_help": "Yahoo Finance symbol: NVDA, AAPL, MSFT, SPY, etc.",
        "investment_label": "Investment amount (USD)",
        "investment_help": "Position size we measure the risk on.",
        "years_label": "History window (years)",
        "years_help": "How many years of history we use for the estimate.",
        "confidence_label": "Confidence level",
        "confidence_help": "Higher confidence pushes the estimate further into the tail.",
        "sidebar_caption": "We compute a **1-day** historical VaR/ES and its upside twin. "
                           "Data is cached for an hour, so risk recomputes instantly.",
        # main flow
        "enter_ticker": "Enter a ticker in the sidebar to start.",
        "loading": "Loading {ticker} history…",
        "no_data": "Not enough data for **{ticker}**. Check the symbol or widen the window.",
        "metrics_header": "{ticker} · key metrics",
        # metric cards
        "m_price": "Current price",
        "m_return": "Expected annual return",
        "m_vol": "Annual volatility",
        "m_var": "1-Day VaR (Normal Risk)",
        "m_es": "Expected Shortfall (Crisis Risk)",
        "m_upside": "Upside Potential (Normal Gain)",
        "m_eu": "Expected Upside (Rally Gain)",
        # explanation panel
        "what_header": "📖 What it means",
        "what_body": (
            "If we invest **\\${inv}**, there is a **{conf}%** chance our daily loss "
            "won't exceed **\\${var_d}**. That's our 'normal' maximum risk.\n\n"
            "But if panic hits the market (the worst **{alpha}%** of days), we can "
            "expect to lose **\\${es_d}** on average. This is our crisis risk."
        ),
        "what_body_upside": (
            "**Upside scenario.** There is a **{alpha}%** chance your daily profit will "
            "exceed **\\${z}** (Upside Potential). On the absolute best **{alpha}%** of "
            "days, your average profit will be **\\${w}** (Expected Upside)."
        ),
        # stats table
        "tbl_col_metric": "Metric",
        "tbl_col_value": "Value",
        "tbl_days_sample": "Days in sample",
        "tbl_days_tail": "Days in tail",
        "tbl_mean_return": "Mean daily return",
        "tbl_skew": "Skew",
        "tbl_kurtosis": "Kurtosis",
        "kurtosis_caption": "High kurtosis means hard crashes happen more often than "
                            "perfect theory predicts. That's why we measure risk from "
                            "real history, not formulas.",
        # methodology
        "method_title": "🧠 How it's computed",
        "method_body": (
            "**1. Data.** Adj Close prices from yfinance, already adjusted for "
            "dividends and splits. Fetched once and kept in cache.\n\n"
            "**2. Returns.** Log returns: $r_t = \\ln(P_t / P_{t-1})$. They add up "
            "over time and compute in $O(n)$ without loops.\n\n"
            "**3. VaR & Upside.** The $\\alpha$ and $1-\\alpha$ quantiles of history. "
            "`np.percentile` sorts the array ($O(n \\log n)$) and takes the element "
            "at position $\\alpha \\cdot n$. No normality assumed, we take the real "
            "shape of both tails.\n\n"
            "**4. Expected Shortfall / Upside.** Cut each tail with a mask "
            "(`returns <= VaR`, `returns >= Upside`) in $O(n)$ and average it. That's "
            "the mean loss / gain given the tail is reached.\n\n"
            "**5. Into dollars.** Exact, no approximation: "
            "loss $= V \\cdot (1 - e^{\\,r})$, gain $= V \\cdot (e^{\\,r} - 1)$."
        ),
        "disclaimer": "⚠️ This is a demo tool, not investment advice. "
                      "Past performance doesn't guarantee future results.",
        # chart
        "chart_title": "Distribution of daily log returns",
        "chart_xaxis": "Daily log return",
        "chart_yaxis": "Frequency (number of days)",
        "chart_hover": "Return: %{x:.2%}<br>Days: %{y}<extra></extra>",
        "chart_tail_zone": "Risk tail (ES)",
        "chart_upside_zone": "Upside tail (EU)",
    },
    "RU": {
        "lang_label": "🌐 Language / Язык",
        "title_caption": "Рыночный риск и потенциал на исторических данных · numpy · прозрачная математика",
        # sidebar
        "sidebar_header": "⚙️ Параметры риска",
        "ticker_label": "Тикер актива",
        "ticker_help": "Символ с Yahoo Finance: NVDA, AAPL, MSFT, SPY и т.д.",
        "investment_label": "Сумма инвестиций (USD)",
        "investment_help": "Размер позиции, для которой считаем риск.",
        "years_label": "Период истории (лет)",
        "years_help": "Сколько лет истории берём для оценки.",
        "confidence_label": "Доверительный интервал",
        "confidence_help": "Чем выше доверие, тем дальше в хвост уходит оценка.",
        "sidebar_caption": "Считаем **дневной** VaR/ES и его зеркало для прибыли. "
                           "Данные кэшируются на час, пересчёт риска мгновенный.",
        # main flow
        "enter_ticker": "Введите тикер в сайдбаре, чтобы начать.",
        "loading": "Загружаю историю {ticker}…",
        "no_data": "Мало данных по тикеру **{ticker}**. Проверьте символ или увеличьте период.",
        "metrics_header": "{ticker} · ключевые метрики",
        # metric cards
        "m_price": "Текущая цена",
        "m_return": "Ожидаемая годовая доходность",
        "m_vol": "Годовая волатильность",
        "m_var": "VaR (Риск на 1 день)",
        "m_es": "Expected Shortfall (Риск в кризис)",
        "m_upside": "Upside Potential (Рост за 1 день)",
        "m_eu": "Expected Upside (Рост в ралли)",
        # explanation panel
        "what_header": "📖 Как это понимать",
        "what_body": (
            "Если мы инвестируем **\\${inv}**, то с вероятностью **{conf}%** наш убыток "
            "за день не превысит **\\${var_d}**. Это наш «обычный» максимальный риск.\n\n"
            "Но если на рынке начнётся паника (**{alpha}%** худших сценариев), то в "
            "среднем мы потеряем **\\${es_d}**. Это оценка потерь при форс-мажоре."
        ),
        "what_body_upside": (
            "**Позитивный сценарий.** С вероятностью **{alpha}%** ваш дневной доход "
            "превысит **\\${z}** (Upside Potential). А в дни аномального роста рынка "
            "(лучшие **{alpha}%** дней) средняя прибыль составит **\\${w}** (Expected Upside)."
        ),
        # stats table
        "tbl_col_metric": "Показатель",
        "tbl_col_value": "Значение",
        "tbl_days_sample": "Дней в выборке",
        "tbl_days_tail": "Дней в хвосте",
        "tbl_mean_return": "Средняя дневная доходность",
        "tbl_skew": "Асимметрия (skew)",
        "tbl_kurtosis": "Эксцесс (kurtosis)",
        "kurtosis_caption": "Высокий эксцесс значит, что жёсткие обвалы случаются чаще, "
                            "чем в идеальной теории. Поэтому мы считаем риски по реальной "
                            "истории, а не по формулам.",
        # methodology
        "method_title": "🧠 Как это считается",
        "method_body": (
            "**1. Данные.** Цены Adj Close из yfinance, уже с учётом дивидендов и "
            "сплитов. Качаем один раз и держим в кэше.\n\n"
            "**2. Доходности.** Логарифмические: $r_t = \\ln(P_t / P_{t-1})$. "
            "Складываются во времени, считаются за $O(n)$ без циклов.\n\n"
            "**3. VaR и Upside.** Это $\\alpha$- и $1-\\alpha$-квантили истории. "
            "`np.percentile` сортирует массив ($O(n \\log n)$) и берёт элемент на "
            "позиции $\\alpha \\cdot n$. Нормальность не предполагаем, берём реальную "
            "форму обоих хвостов.\n\n"
            "**4. Expected Shortfall / Upside.** Отрезаем каждый хвост маской "
            "(`returns <= VaR`, `returns >= Upside`) за $O(n)$ и усредняем. Это средний "
            "убыток / доход при условии, что хвост достигнут.\n\n"
            "**5. В доллары.** Точно, без приближений: "
            "убыток $= V \\cdot (1 - e^{\\,r})$, доход $= V \\cdot (e^{\\,r} - 1)$."
        ),
        "disclaimer": "⚠️ Это демо-инструмент, не инвестиционная рекомендация. "
                      "Прошлая доходность не гарантирует будущую.",
        # chart
        "chart_title": "Распределение дневных логарифмических доходностей",
        "chart_xaxis": "Дневная логарифмическая доходность",
        "chart_yaxis": "Частота (число дней)",
        "chart_hover": "Доходность: %{x:.2%}<br>Дней: %{y}<extra></extra>",
        "chart_tail_zone": "Хвост риска (ES)",
        "chart_upside_zone": "Хвост роста (EU)",
    },
}


# full width dashboard
st.set_page_config(
    page_title="VaR & Expected Shortfall Engine",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="expanded",
)

# a bit of styling so the metric cards look premium. works in both dark and light themes
st.markdown(
    """
    <style>
        [data-testid="stMetric"] {
            background: rgba(31, 111, 235, 0.06);
            border: 1px solid rgba(31, 111, 235, 0.20);
            border-radius: 14px;
            padding: 18px 22px;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.06);
        }
        [data-testid="stMetricValue"] {
            font-size: 1.9rem;
            font-weight: 700;
            letter-spacing: -0.02em;
        }
        [data-testid="stMetricLabel"] p {
            font-size: 0.82rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            opacity: 0.75;
        }
        div.block-container { padding-top: 2.2rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


# data loading
@st.cache_data(ttl=3600, show_spinner=False)
def load_price_data(ticker: str, years: int) -> pd.Series:
    """
    Pulls price history (Adj Close) from yfinance.

    Cache is keyed on (ticker, years) only, language is not an argument here,
    so flipping EN/RU never triggers a reload. While ticker and years don't
    change we never hit the network again.

    Returns a price series by date. An empty series means the ticker wasn't found.
    """
    end = datetime.today()
    start = end - timedelta(days=int(years * 365.25))

    # auto_adjust=False so we explicitly get the Adj Close column (dividends and splits included)
    raw = yf.download(
        ticker,
        start=start,
        end=end,
        auto_adjust=False,
        progress=False,
    )

    if raw is None or raw.empty:
        return pd.Series(dtype=float)

    # recent yfinance sometimes returns two level columns like ('Adj Close', 'NVDA').
    # handle both shapes so the code doesn't break
    if isinstance(raw.columns, pd.MultiIndex):
        field = "Adj Close" if "Adj Close" in raw.columns.get_level_values(0) else "Close"
        prices = raw[field]
        if isinstance(prices, pd.DataFrame):   # one column per ticker, take it
            prices = prices.iloc[:, 0]
    else:
        field = "Adj Close" if "Adj Close" in raw.columns else "Close"
        prices = raw[field]

    prices = prices.dropna()
    prices.name = ticker.upper()
    return prices


# math
def compute_log_returns(prices: pd.Series) -> np.ndarray:
    """
    Daily log returns: r = ln(price today / price yesterday).

    We use log returns because they add up over time and the distribution
    comes out closer to normal.

    Computed as one formula over the whole array, no loops.
    """
    log_returns = np.log(prices / prices.shift(1))
    # the first day has no yesterday price, that's a NaN, drop it
    return log_returns.dropna().to_numpy()


def compute_risk_metrics(log_returns: np.ndarray, confidence: float) -> dict:
    """
    Computes both tails from history: downside (VaR/ES) and upside (Upside/EU).

    alpha = 1 - confidence is the tail probability. For 95% confidence alpha = 0.05.

    Downside is the alpha quantile, upside is the mirror 1-alpha quantile.
    np.percentile sorts the array (O(n log n)) and picks the element at that
    position. ES / EU then average the days beyond each threshold (a mask, O(n)).
    ES sits below VaR and EU sits above Upside, because both are means of the
    extreme days, not single points on the edge.
    """
    alpha = 1.0 - confidence

    # --- downside (left tail) ---
    # VaR: the alpha quantile, a negative number
    var_return = float(np.percentile(log_returns, alpha * 100.0))
    # ES: every day worse than VaR, then their mean
    left_tail = log_returns[log_returns <= var_return]
    es_return = float(left_tail.mean()) if left_tail.size > 0 else var_return

    # --- upside (right tail), mirror image of the above ---
    # Upside Potential: the 1-alpha quantile, a positive number
    upside_return = float(np.percentile(log_returns, (1.0 - alpha) * 100.0))
    # Expected Upside: every day better than Upside, then their mean
    right_tail = log_returns[log_returns >= upside_return]
    exp_upside_return = float(right_tail.mean()) if right_tail.size > 0 else upside_return

    # turn returns into loss / gain fractions.
    # for a log return r the new value is V * exp(r).
    # loss = 1 - exp(r)  (r < 0, comes out positive)
    # gain = exp(r) - 1  (r > 0, comes out positive)
    var_loss_fraction = 1.0 - np.exp(var_return)
    es_loss_fraction = 1.0 - np.exp(es_return)
    upside_gain_fraction = np.exp(upside_return) - 1.0
    eu_gain_fraction = np.exp(exp_upside_return) - 1.0

    return {
        "alpha": alpha,
        # downside
        "var_return": var_return,
        "es_return": es_return,
        "var_loss_fraction": var_loss_fraction,
        "es_loss_fraction": es_loss_fraction,
        "tail_size": int(left_tail.size),
        # upside
        "upside_return": upside_return,
        "exp_upside_return": exp_upside_return,
        "upside_gain_fraction": upside_gain_fraction,
        "eu_gain_fraction": eu_gain_fraction,
        "upside_tail_size": int(right_tail.size),
    }


# chart
def build_distribution_chart(
    log_returns: np.ndarray,
    var_return: float,
    es_return: float,
    upside_return: float,
    exp_upside_return: float,
    confidence_pct: int,
    t: dict,
) -> go.Figure:
    """
    Histogram of daily returns. Labels come from the language dict `t`.

    Blue hill in the middle. On the left, the red risk tail: dashed VaR line,
    shaded zone, dotted ES line. On the right, the mirror in green: dashed
    Upside line, shaded zone, dotted Expected Upside line.
    """
    fig = go.Figure()

    # the return distribution itself
    fig.add_trace(
        go.Histogram(
            x=log_returns,
            nbinsx=120,
            marker_color=CORP_BLUE,
            marker_line_width=0,
            opacity=0.85,
            hovertemplate=t["chart_hover"],
        )
    )

    # --- left side: red risk tail (everything left of VaR) ---
    fig.add_vrect(
        x0=float(log_returns.min()),
        x1=var_return,
        fillcolor=TAIL_RED,
        opacity=0.16,
        line_width=0,
        layer="below",
        annotation_text=t["chart_tail_zone"],
        annotation_position="top left",
        annotation_font_size=12,
        annotation_font_color=TAIL_RED,
    )
    fig.add_vline(
        x=var_return,
        line_width=2.6,
        line_dash="dash",
        line_color=TAIL_RED,
        annotation_text=f"VaR {confidence_pct}%",
        annotation_position="top",
        annotation_font_size=13,
        annotation_font_color=TAIL_RED,
    )
    fig.add_vline(
        x=es_return,
        line_width=2.0,
        line_dash="dot",
        line_color=TAIL_RED_DARK,
        annotation_text="ES",
        annotation_position="bottom",
        annotation_font_size=13,
        annotation_font_color=TAIL_RED_DARK,
    )

    # --- right side: green upside tail (everything right of Upside) ---
    fig.add_vrect(
        x0=upside_return,
        x1=float(log_returns.max()),
        fillcolor=TAIL_GREEN,
        opacity=0.16,
        line_width=0,
        layer="below",
        annotation_text=t["chart_upside_zone"],
        annotation_position="top right",
        annotation_font_size=12,
        annotation_font_color=TAIL_GREEN_DARK,
    )
    fig.add_vline(
        x=upside_return,
        line_width=2.6,
        line_dash="dash",
        line_color=TAIL_GREEN,
        annotation_text=f"Upside {confidence_pct}%",
        annotation_position="top",
        annotation_font_size=13,
        annotation_font_color=TAIL_GREEN_DARK,
    )
    fig.add_vline(
        x=exp_upside_return,
        line_width=2.0,
        line_dash="dot",
        line_color=TAIL_GREEN_DARK,
        annotation_text="EU",
        annotation_position="bottom",
        annotation_font_size=13,
        annotation_font_color=TAIL_GREEN_DARK,
    )

    fig.update_layout(
        template="plotly_white",
        title=t["chart_title"],
        xaxis_title=t["chart_xaxis"],
        yaxis_title=t["chart_yaxis"],
        bargap=0.01,
        height=520,
        margin=dict(l=50, r=40, t=70, b=50),
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, Segoe UI, sans-serif", size=13),
        hoverlabel=dict(font_size=13),
    )
    fig.update_xaxes(tickformat=".1%", showgrid=True, gridcolor="rgba(128,128,128,0.15)")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(128,128,128,0.15)")
    return fig


# sidebar with settings
def render_sidebar() -> dict:
    """
    Language switch first, then all inputs. The language choice is read before
    the rest of the sidebar so every label below is already localized.
    """
    lang_display = st.sidebar.radio(
        TEXTS["EN"]["lang_label"],   # this one label is bilingual on purpose
        options=["🇬🇧 EN", "🇷🇺 RU"],
        horizontal=True,
    )
    lang = "RU" if "RU" in lang_display else "EN"
    t = TEXTS[lang]

    st.sidebar.header(t["sidebar_header"])

    ticker = st.sidebar.text_input(
        t["ticker_label"],
        value="NVDA",
        help=t["ticker_help"],
    ).strip().upper()

    investment = st.sidebar.number_input(
        t["investment_label"],
        min_value=1_000,
        max_value=1_000_000_000,
        value=100_000,
        step=1_000,
        help=t["investment_help"],
    )

    years = st.sidebar.slider(
        t["years_label"],
        min_value=1,
        max_value=15,
        value=5,
        help=t["years_help"],
    )

    # only let the user pick the three standard confidence levels
    confidence_pct = st.sidebar.select_slider(
        t["confidence_label"],
        options=[90, 95, 99],
        value=95,
        help=t["confidence_help"],
    )

    st.sidebar.caption(t["sidebar_caption"])

    return {
        "lang": lang,
        "ticker": ticker,
        "investment": float(investment),
        "years": int(years),
        "confidence_pct": int(confidence_pct),
        "confidence": confidence_pct / 100.0,
    }


# main flow
def main() -> None:
    st.title("📉 Advanced VaR & Expected Shortfall Engine")

    params = render_sidebar()
    t = TEXTS[params["lang"]]     # active language dict for this run

    st.caption(t["title_caption"])

    if not params["ticker"]:
        st.info(t["enter_ticker"])
        st.stop()

    # fetch data (from cache if already downloaded, unaffected by language)
    with st.spinner(t["loading"].format(ticker=params["ticker"])):
        prices = load_price_data(params["ticker"], params["years"])

    if prices.empty or len(prices) < 30:
        st.error(t["no_data"].format(ticker=params["ticker"]))
        st.stop()

    # compute returns and both tails
    log_returns = compute_log_returns(prices)
    metrics = compute_risk_metrics(log_returns, params["confidence"])

    # tails in dollars = position size * fraction
    var_dollar = params["investment"] * metrics["var_loss_fraction"]
    es_dollar = params["investment"] * metrics["es_loss_fraction"]
    upside_dollar = params["investment"] * metrics["upside_gain_fraction"]
    eu_dollar = params["investment"] * metrics["eu_gain_fraction"]

    # headline numbers
    current_price = float(prices.iloc[-1])
    day_change = current_price / float(prices.iloc[-2]) - 1.0
    ann_vol = float(np.std(log_returns, ddof=1) * np.sqrt(TRADING_DAYS))
    # annualized mean: geometric annualization of the average daily log return
    annual_return = float(np.exp(log_returns.mean() * TRADING_DAYS) - 1.0)

    st.subheader(t["metrics_header"].format(ticker=params["ticker"]))

    # top row: price, expected annual return, annual volatility
    r1c1, r1c2, r1c3 = st.columns(3)
    r1c1.metric(t["m_price"], f"${current_price:,.2f}", f"{day_change:+.2%}")
    r1c2.metric(t["m_return"], f"{annual_return:+.2%}")
    r1c3.metric(t["m_vol"], f"{ann_vol:.2%}")

    # second row: downside in red (minus delta), upside in green (plus delta)
    r2c1, r2c2, r2c3, r2c4 = st.columns(4)
    r2c1.metric(t["m_var"], f"${var_dollar:,.0f}", f"-{metrics['var_loss_fraction']:.2%}")
    r2c2.metric(t["m_es"], f"${es_dollar:,.0f}", f"-{metrics['es_loss_fraction']:.2%}")
    r2c3.metric(t["m_upside"], f"${upside_dollar:,.0f}", f"+{metrics['upside_gain_fraction']:.2%}")
    r2c4.metric(t["m_eu"], f"${eu_dollar:,.0f}", f"+{metrics['eu_gain_fraction']:.2%}")

    st.divider()

    # chart and explanation
    left, right = st.columns([2, 1], gap="large")

    with left:
        fig = build_distribution_chart(
            log_returns,
            metrics["var_return"],
            metrics["es_return"],
            metrics["upside_return"],
            metrics["exp_upside_return"],
            params["confidence_pct"],
            t,
        )
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.markdown(f"#### {t['what_header']}")
        st.markdown(
            t["what_body"].format(
                inv=f"{params['investment']:,.0f}",
                conf=params["confidence_pct"],
                var_d=f"{var_dollar:,.0f}",
                alpha=f"{metrics['alpha'] * 100:.0f}",
                es_d=f"{es_dollar:,.0f}",
            )
        )
        st.markdown(
            t["what_body_upside"].format(
                alpha=f"{metrics['alpha'] * 100:.0f}",
                z=f"{upside_dollar:,.0f}",
                w=f"{eu_dollar:,.0f}",
            )
        )
        st.markdown(
            f"""
            | {t['tbl_col_metric']} | {t['tbl_col_value']} |
            |---|---|
            | {t['tbl_days_sample']} | {len(log_returns):,} |
            | {t['tbl_days_tail']} | {metrics['tail_size']:,} |
            | {t['tbl_mean_return']} | {log_returns.mean():.3%} |
            | {t['tbl_skew']} | {pd.Series(log_returns).skew():.3f} |
            | {t['tbl_kurtosis']} | {pd.Series(log_returns).kurtosis():.3f} |
            """
        )
        st.caption(t["kurtosis_caption"])

    # methodology, handy to open in an interview
    with st.expander(t["method_title"]):
        st.markdown(t["method_body"])

    st.caption(t["disclaimer"])


if __name__ == "__main__":
    main()
