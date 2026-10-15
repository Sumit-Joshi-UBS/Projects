    h_sel, _ = load_history(ticker, "1y", "1d", False)
    std_sel = annualized_std(h_sel["Close"]) if not h_sel.empty else None
    if std_sel is not None:
        rc1, rc2 = st.columns(2)
        rc1.metric("Std Dev (Annualized)", f"{std_sel * 100:.1f}%")
        rc2.metric("Daily Std Dev", f"{(std_sel / np.sqrt(252)) * 100:.2f}%")
    else:
        st.info("Insufficient price history to compute standard deviation.")

    with st.expander("Raw Financial Data"):
        st.write("Income Statement", income)
        st.write("Balance Sheet", balance)
        st.write("Cash Flow", cashflow)
        st.write("Computed Cash Flow Metrics", cf_data)


# ====================================================================== #
# 9. HEADER HELPERS
# ====================================================================== #
def render_market_strip(fi, period, sector):
    price = _safe_float(fi.get("currentPrice"))
    pe = _safe_float(fi.get("trailingPE"))
    roe = _safe_float(fi.get("returnOnEquity"))
    roce = _safe_float(fi.get("returnOnCapitalEmployed"))
    cap = _safe_float(fi.get("marketCap"))
    cap_text = _fmt_num(cap, prefix="₹", decimals=1)
    cards = [
        ("Last price", _fmt_num(price, prefix="₹", decimals=2), "Live / cached equity price"),
        ("Market cap", cap_text, sector + " universe"),
        ("P / E", _fmt_ratio(pe), "Trailing valuation multiple"),
        ("ROE / ROCE", f"{_fmt_pct(roe)} / {_fmt_pct(roce)}", f"Selected timeframe · {period.upper()}"),
    ]
    html = ['<div class="ticker-strip">']
    for label, value, meta in cards:
        html.append(f'<div class="ticker-card"><div class="ticker-label">{label}</div>'
                    f'<div class="ticker-value">{value}</div><div class="ticker-meta">{meta}</div></div>')
    html.append('</div>')
    st.markdown("".join(html), unsafe_allow_html=True)


# ====================================================================== #
# 10. MAIN
# ====================================================================== #
def main():
    inject_css()

    with st.sidebar:
        render_brand()
        st.markdown('<div class="small-note">INSTITUTIONAL RESEARCH WORKSPACE</div>', unsafe_allow_html=True)
        st.markdown("<hr>", unsafe_allow_html=True)

        st.markdown("**Research Universe**")
        sector = st.selectbox("Sector", list(SECTORS.keys()), index=0, label_visibility="collapsed")
        sector_tickers = SECTORS[sector]
        ticker_names = [n for n, _ in sector_tickers]
        selected_name = st.selectbox("Stock", ticker_names, index=0, label_visibility="collapsed")
        selected_ticker = dict(sector_tickers)[selected_name]

        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        st.markdown("**Chart Controls**")
        interval = st.selectbox("Candle interval", ["1d", "1h", "15m"], index=0,
                                format_func=lambda x: {"1d": "Daily", "1h": "1 Hour", "15m": "15 Minutes"}[x])
        period_options = {
            "15m": ["1d", "5d", "1mo"],
            "1h": ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y"],
            "1d": ["1mo", "6mo", "1y", "5y", "max"],
        }
        period_labels = {
            "1d": "1 Day", "5d": "5 Days", "1mo": "1 Month", "3mo": "3 Months",
            "6mo": "6 Months", "1y": "1 Year", "2y": "2 Years", "5y": "5 Years", "max": "Max",
        }
        available_periods = period_options[interval]
        default_index = 0 if interval in {"15m", "1h"} else min(len(available_periods) - 1, 2)
        period = st.selectbox("Lookback", available_periods, index=default_index,
                              format_func=lambda x: period_labels[x])
        allow_demo = st.toggle("Allow synthetic fallback", value=True)

        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("**Coverage**")
        st.markdown('<div class="small-note">Indian equities across Pharmaceuticals, IT & Technology and Energy & Power. '
                    'Portfolio returns use trailing two-month adjusted-price performance; risk uses annualized standard deviation of daily returns. '
                    'DCF valuation remains separate and does not feed portfolio expected returns.</div>',
                    unsafe_allow_html=True)
        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
        st.markdown('<div class="small-note">Data: Yahoo Finance · Cached analytics · Fallback mode available</div>',
                    unsafe_allow_html=True)

    fi = load_fundamentals(selected_ticker)
    df_hist, is_demo = load_history(selected_ticker, period, interval, allow_demo)

    render_hero(selected_name, selected_ticker, sector, is_demo)
    render_market_strip(fi, period, sector)
    render_index_marquee()
    st.caption("Portfolio selection uses five stocks each from Pharmaceuticals, Energy & Power, and IT & Technology, ranked by trailing two-month adjusted-price return. No hedging instruments are used.")

    if is_demo:
        st.warning("Live data was unavailable — synthetic fallback is being displayed for the selected timeframe.")

    st.markdown(
        '<div class="section-head"><div><h3>Research modules</h3><p>Move from screening to price action, peers, factor risk, DCF valuation and portfolio allocation.</p></div></div>',
        unsafe_allow_html=True,
    )

    tab_screener, tab_charts, tab_compare, tab_factor, tab_val, tab_alloc = st.tabs(
        ["01 · Overview", "02 · Price Action", "03 · Peer Lab",
         "04 · Factor Models", "05 · Valuation", "06 · Portfolio Allocation"]
    )

    with tab_screener:
        render_screener(selected_ticker, fi)

    with tab_charts:
        st.markdown('<div class="section-head"><div><h3>Price action & technical structure</h3>'
                    '<p>Candlesticks · moving averages · VWAP · volume · RSI</p></div></div>',
                    unsafe_allow_html=True)
        st.markdown(f'<div class="chart-state">◉ {period_labels[period]} lookback · {interval.upper()} candles</div>',
                    unsafe_allow_html=True)
        render_charts(selected_ticker, df_hist, interval=interval, period=period)

    with tab_compare:
        render_compare(sector, sector_tickers)

    with tab_factor:
        render_factor_models(selected_ticker, sector, ALL_STOCKS)

    with tab_val:
        render_valuation(selected_ticker, fi, sector_tickers)

    with tab_alloc:
        render_universe_valuation()


if __name__ == "__main__":
    main()
