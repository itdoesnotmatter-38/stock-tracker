import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import os

# --- CONFIGURATION ---
st.set_page_config(layout="wide", page_title="Global Stock Tracker")
FILE_NAME = "portfolio.csv"
USD_TO_HKD = 7.78
HKD_TO_USD = 1 / USD_TO_HKD

# --- 🔐 PASSWORD PROTECTION ---
def check_password():
    def password_entered():
        if st.session_state["password"] == "admin123": 
            st.session_state["password_correct"] = True
            del st.session_state["password"]
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        st.text_input("Password", type="password", on_change=password_entered, key="password")
        return False
    elif not st.session_state["password_correct"]:
        st.text_input("Password", type="password", on_change=password_entered, key="password")
        st.error("😕 Password incorrect")
        return False
    return True

if not check_password():
    st.stop()

# --- STOCK NAMES DICTIONARY ---
STOCK_NAMES = {
    "TSLA": "Tesla Inc", "NVDA": "NVIDIA Corp", "META": "Meta Platforms",
    "ORCL": "Oracle Corp", "PLTR": "Palantir Tech", "SOFI": "SoFi Tech",
    "QUBT": "Quantum Comp", "FLNC": "Fluence Energy", "SNOW": "Snowflake",
    "ZM": "Zoom Video", "FIG": "Figma (Private?)", "AAPL": "Apple Inc", "MSFT": "Microsoft",
    "9698.HK": "GDS Holdings", "0388.HK": "HKEX", "9988.HK": "Alibaba",
    "0981.HK": "SMIC", "0700.HK": "Tencent", "1810.HK": "Xiaomi",
    "3993.HK": "CMOC", "1211.HK": "BYD Co", "0285.HK": "BYD Electronic",
    "0909.HK": "Ming Yuan Cloud", "9618.HK": "JD.com"
}

# --- SESSION STATE (For Click Selection) ---
if 'selected_ticker' not in st.session_state:
    st.session_state.selected_ticker = None

# --- DATA FUNCTIONS ---
def load_portfolio():
    if os.path.exists(FILE_NAME): return pd.read_csv(FILE_NAME)
    return pd.DataFrame(columns=["Date", "Ticker", "Action", "Quantity", "Price", "Total_Value"])

def save_transaction(date, ticker_string, action, quantity, price):
    df = load_portfolio()
    clean_ticker = ticker_string.split(" | ")[0].strip() if " | " in ticker_string else ticker_string.strip()
    total = quantity * price
    if action == "Sell": total = -total
    new_row = pd.DataFrame({"Date": [date], "Ticker": [clean_ticker], "Action": [action], "Quantity": [quantity], "Price": [price], "Total_Value": [total]})
    pd.concat([df, new_row], ignore_index=True).to_csv(FILE_NAME, index=False)

def calculate_holdings(df):
    if df.empty: return pd.DataFrame()
    portfolio = {}
    for _, row in df.iterrows():
        if row['Ticker'] == 'USD' or row['Action'] in ['Deposit Cash', 'Withdraw Cash']: continue
        t = row['Ticker']
        if t not in portfolio: portfolio[t] = {'shares': 0.0, 'total_cost': 0.0}
        if row['Action'] == 'Buy':
            portfolio[t]['shares'] += row['Quantity']
            portfolio[t]['total_cost'] += (row['Quantity'] * row['Price'])
        elif row['Action'] == 'Sell':
            portfolio[t]['shares'] -= row['Quantity']
            if portfolio[t]['shares'] > 0:
                avg = portfolio[t]['total_cost'] / (portfolio[t]['shares'] + row['Quantity'])
                portfolio[t]['total_cost'] -= (avg * row['Quantity'])
            else: portfolio[t]['shares'] = 0; portfolio[t]['total_cost'] = 0

    results = []
    for t, data in portfolio.items():
        if data['shares'] > 0:
diff --git a/app.py b/app.py
index 74c15172cded2d8b320d56fa70c5bb4f08724d80..c008dd8d1f4efbc6f385df006b3bcc1634bc3ce1 100644
--- a/app.py
+++ b/app.py
@@ -84,60 +84,113 @@ def calculate_holdings(df):
             results.append({
                 "Ticker": t, "Name": STOCK_NAMES.get(t, t),
                 "Shares": data['shares'], "Avg Cost": data['total_cost'] / data['shares'],
                 "Total Cost Basis": data['total_cost']
             })
     return pd.DataFrame(results)
 
 def get_market_data(ticker):
     try:
         df = yf.download(ticker, period="2y", progress=False)
         if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.droplevel(1)
         df.reset_index(inplace=True)
         # Indicators
         df['SMA_20'] = df['Close'].rolling(window=20).mean()
         df['SMA_50'] = df['Close'].rolling(window=50).mean()
         df['SMA_100'] = df['Close'].rolling(window=100).mean()
         df['SMA_250'] = df['Close'].rolling(window=250).mean()
         delta = df['Close'].diff()
         gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
         loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
         rs = gain / loss
         df['RSI'] = 100 - (100 / (1 + rs))
         return df
     except: return None
 
+
+@st.cache_data(ttl=900)
+def build_portfolio_timeseries(holdings_df, period="3mo"):
+    """
+    Build a HKD-denominated portfolio value time series using historical closes.
+    Cached to avoid repeated downloads when toggling between stocks.
+    """
+    tickers = holdings_df['Ticker'].tolist()
+    if not tickers:
+        return pd.DataFrame()
+
+    try:
+        hist = yf.download(" ".join(tickers), period=period, interval="1d", progress=False)
+    except Exception:
+        return pd.DataFrame()
+
+    if hist.empty:
+        return pd.DataFrame()
+
+    # Normalize close prices to a DataFrame keyed by ticker
+    if isinstance(hist.columns, pd.MultiIndex):
+        closes = hist['Close']
+    else:
+        closes = hist[['Close']]
+        closes.columns = [tickers[0]]
+
+    closes = closes.ffill().dropna(how="all")
+    if closes.empty:
+        return pd.DataFrame()
+
+    values = pd.DataFrame(index=closes.index)
+    for ticker in tickers:
+        if ticker not in closes.columns:
+            continue
+        shares = holdings_df.loc[holdings_df['Ticker'] == ticker, 'Shares'].sum()
+        conversion = 1 if ".HK" in ticker else USD_TO_HKD
+        values[ticker] = closes[ticker] * shares * conversion
+
+    values['Total Value (HKD)'] = values.sum(axis=1)
+    values.index = pd.to_datetime(values.index)
+    return values[['Total Value (HKD)']]
+
 # --- MAIN APP ---
 st.title("🌏 Global Stock Tracker")
 
 # SIDEBAR
 with st.sidebar:
     st.header("Control Panel")
     
     # 1. REFRESH BUTTON
     if st.button("🔄 Refresh Data", type="primary", use_container_width=True):
         st.rerun()
+
+    # Force clear cached data (e.g., after deploying a new version)
+    if st.button("♻️ Force Refresh (Clear Cache)", use_container_width=True):
+        build_portfolio_timeseries.clear()
+        st.rerun()
+
+    # Clear all Streamlit caches to pull the freshest code and data
+    if st.button("🧹 Full Cache Reset", use_container_width=True):
+        st.cache_data.clear()
+        st.cache_resource.clear()
+        st.rerun()
         
     st.divider()
     
     # 2. ADD TRADE FORM
     with st.form("entry"):
         st.subheader("Add Trade")
         d = st.date_input("Date")
         a = st.selectbox("Action", ["Buy", "Sell"])
         t_input = st.text_input("Ticker (e.g. 9988.HK)", "").upper()
         q = st.number_input("Quantity", 1.0)
         p = st.number_input("Price", 0.01)
         if st.form_submit_button("Save Transaction"):
             save_transaction(d, t_input, a, q, p)
             st.success("Saved")
             st.rerun()
 
 # --- PROCESSING ---
 df_raw = load_portfolio()
 holdings = calculate_holdings(df_raw)
 
 if not holdings.empty:
     # FETCH PRICES
     tickers = " ".join(holdings['Ticker'].tolist())
     try:
         hist_data = yf.download(tickers, period="5d", progress=False)
@@ -146,127 +199,218 @@ if not holdings.empty:
             try:
                 if len(holdings) == 1: closes = hist_data['Close']
                 else: closes = hist_data['Close'][ticker]
                 valid = closes.dropna()
                 if len(valid) >= 2: current_prices.append(valid.iloc[-1]); prev_closes.append(valid.iloc[-2])
                 else: current_prices.append(valid.iloc[-1]); prev_closes.append(valid.iloc[-1])
             except: current_prices.append(0); prev_closes.append(0)
         holdings['Current Price'] = current_prices
         holdings['Prev Close'] = prev_closes
     except:
         holdings['Current Price'] = holdings['Avg Cost']; holdings['Prev Close'] = holdings['Avg Cost']
 
     # --- CALCULATIONS ---
     holdings['Value (Native)'] = holdings['Current Price'] * holdings['Shares']
     holdings['Profit (Native)'] = holdings['Value (Native)'] - holdings['Total Cost Basis']
     holdings['Return %'] = (holdings['Profit (Native)'] / holdings['Total Cost Basis']) * 100
     holdings['Day Change $'] = (holdings['Current Price'] - holdings['Prev Close']) * holdings['Shares']
     holdings['Day Change %'] = ((holdings['Current Price'] - holdings['Prev Close']) / holdings['Prev Close']) * 100
 
     # Conversions
     def to_hkd(row, col): return row[col] if ".HK" in row['Ticker'] else row[col] * USD_TO_HKD
     holdings['Value (HKD)'] = holdings.apply(lambda x: to_hkd(x, 'Value (Native)'), axis=1)
     holdings['Profit (HKD)'] = holdings.apply(lambda x: to_hkd(x, 'Profit (Native)'), axis=1)
     holdings['Day Change (HKD)'] = holdings.apply(lambda x: to_hkd(x, 'Day Change $'), axis=1)
 
+    # --- PORTFOLIO SNAPSHOT (Mix, Trend) ---
+    us_value_hkd = holdings.loc[~holdings['Ticker'].str.contains(".HK"), 'Value (HKD)'].sum()
+    hk_value_hkd = holdings.loc[holdings['Ticker'].str.contains(".HK"), 'Value (HKD)'].sum()
+    total_value_hkd = holdings['Value (HKD)'].sum()
+
+    us_mix = (us_value_hkd / total_value_hkd * 100) if total_value_hkd > 0 else 0
+    hk_mix = (hk_value_hkd / total_value_hkd * 100) if total_value_hkd > 0 else 0
+
+    avg_return = holdings['Return %'].mean() if not holdings['Return %'].empty else 0
+    winners = (holdings['Return %'] > 0).sum()
+    losers = (holdings['Return %'] <= 0).sum()
+    day_trend = holdings['Day Change (HKD)'].sum()
+    positions = len(holdings)
+
+    st.subheader("📌 Portfolio Snapshot")
+    snap_c1, snap_c2, snap_c3 = st.columns(3)
+    snap_c1.metric("Mix", f"US {us_mix:.1f}% / HK {hk_mix:.1f}%", help="Share of portfolio value by market in HKD terms")
+    snap_c2.metric("Trend Today", f"HK$ {day_trend:,.0f}", delta=f"{day_trend / (total_value_hkd - day_trend) * 100:+.2f}%" if total_value_hkd - day_trend > 0 else None)
+    snap_c3.metric("Breadth", f"{winners} gainers / {losers} laggards", delta=f"Avg return {avg_return:+.1f}%")
+    st.caption(f"Currently tracking {positions} open positions across {holdings['Ticker'].nunique()} tickers.")
+
+    # --- PORTFOLIO VALUE TREND ---
+    trend_data = build_portfolio_timeseries(holdings)
+    if not trend_data.empty:
+        latest_val = trend_data['Total Value (HKD)'].iloc[-1]
+        prev_val = trend_data['Total Value (HKD)'].iloc[-2] if len(trend_data) > 1 else latest_val
+        delta_val = latest_val - prev_val
+        delta_pct = (delta_val / prev_val * 100) if prev_val else 0
+
+        t1, t2 = st.columns([1, 3])
+        t1.metric("Value Trend", f"HK$ {latest_val:,.0f}", delta=f"{delta_pct:+.2f}%", help="Based on last available close")
+
+        trend_fig = go.Figure(
+            go.Scatter(
+                x=trend_data.index,
+                y=trend_data['Total Value (HKD)'],
+                mode="lines",
+                line=dict(color="#3b7dd8", width=3),
+                fill="tozeroy",
+            )
+        )
+        trend_fig.update_layout(title="Portfolio Value (HKD)", yaxis_title="Value", xaxis_title="Date", margin=dict(t=60, b=40))
+        t2.plotly_chart(trend_fig, use_container_width=True)
+    else:
+        st.info("No trend yet — add positions to see your portfolio value over time.")
+    st.divider()
+
+    # --- PORTFOLIO SUMMARY CHARTS ---
+    st.subheader("📈 Portfolio Summary Charts")
+    chart_c1, chart_c2 = st.columns(2)
+
+    with chart_c1:
+        if total_value_hkd > 0:
+            mix_fig = go.Figure(go.Pie(
+                labels=["US Market", "HK Market"],
+                values=[us_value_hkd, hk_value_hkd],
+                hole=0.45,
+                marker=dict(colors=["#3498db", "#e67e22"]),
+                textinfo="label+percent",
+            ))
+            mix_fig.update_layout(title_text="Market Mix (by value)")
+            st.plotly_chart(mix_fig, use_container_width=True)
+        else:
+            st.info("Add positions to see market mix.")
+
+    with chart_c2:
+        perf_view = holdings.sort_values('Value (HKD)', ascending=False)
+        perf_fig = go.Figure()
+        perf_fig.add_bar(
+            x=perf_view['Ticker'],
+            y=perf_view['Profit (HKD)'],
+            marker_color=["#2ecc71" if v >= 0 else "#e74c3c" for v in perf_view['Profit (HKD)']],
+            text=[f"HK$ {v:,.0f}" for v in perf_view['Profit (HKD)']],
+            textposition="outside",
+        )
+        perf_fig.update_layout(
+            title_text="P/L by Ticker (HKD)",
+            yaxis_title="Profit / Loss",
+            xaxis_title="Ticker",
+            margin=dict(t=50, b=50),
+            uniformtext_minsize=8,
+        )
+        st.plotly_chart(perf_fig, use_container_width=True)
+
     # --- GLOBAL TOTALS ---
-    total_val_hkd = holdings['Value (HKD)'].sum()
+    total_val_hkd = total_value_hkd
     total_profit_hkd = holdings['Profit (HKD)'].sum()
     total_day_change_hkd = holdings['Day Change (HKD)'].sum()
     
     # Global Day % 
     prev_total_val_hkd = total_val_hkd - total_day_change_hkd
     global_day_pct = (total_day_change_hkd / prev_total_val_hkd * 100) if prev_total_val_hkd > 0 else 0.0
 
     # --- GLOBAL HEADLINE ---
     c1, c2, c3 = st.columns(3)
     c1.metric("💰 Portfolio Value", f"HK$ {total_val_hkd:,.0f}")
     c2.metric("📈 Total Profit", f"HK$ {total_profit_hkd:,.0f}", delta_color="normal" if total_profit_hkd >= 0 else "inverse")
     c3.metric("⚡ Day's P/L", f"HK$ {total_day_change_hkd:,.0f}", delta=f"{global_day_pct:+.2f}%")
     st.divider()
 
     # --- TABS ---
     tab_us, tab_hk, tab_hist = st.tabs(["🇺🇸 US Market", "🇭🇰 HK Market", "📋 History"])
 
-    def render_interactive_table(df, currency):
+    def render_interactive_table(df, currency, key_prefix):
         # Market Specific Totals
         mkt_val = df['Value (Native)'].sum()
         mkt_profit = df['Profit (Native)'].sum()
         mkt_day_pl = df['Day Change $'].sum()
         
         # Calculate Market Day %
         prev_mkt_val = mkt_val - mkt_day_pl
         mkt_pct = (mkt_day_pl / prev_mkt_val * 100) if prev_mkt_val > 0 else 0.0
 
         # 3 METRICS PER MARKET
         c1, c2, c3 = st.columns(3)
         c1.metric("Market Value", f"{currency} {mkt_val:,.0f}")
         c2.metric("Total Profit", f"{currency} {mkt_profit:,.0f}", delta_color="normal" if mkt_profit >= 0 else "inverse")
         c3.metric("Day's P/L", f"{currency} {mkt_day_pl:,.0f}", delta=f"{mkt_pct:+.2f}%")
         
         # Interactive Table
         cols = ['Ticker', 'Name', 'Shares', 'Avg Cost', 'Current Price', 'Value (Native)', 'Profit (Native)', 'Return %', 'Day Change %']
         
         event = st.dataframe(
             df[cols].style.format({
                 "Avg Cost": f"{currency} {{:.2f}}", "Current Price": f"{currency} {{:.2f}}",
                 "Value (Native)": f"{currency} {{:,.0f}}", "Profit (Native)": f"{currency} {{:+,.0f}}",
                 "Return %": "{:+.1f}%", "Day Change %": "{:+.2f}%"
             }).map(lambda x: f"color: {'green' if x > 0 else 'red'}", subset=['Profit (Native)', 'Return %', 'Day Change %']),
             column_config={
                 "Name": st.column_config.TextColumn("Stock", width="medium"),
                 "Return %": st.column_config.NumberColumn("Total Return %"),
                 "Day Change %": st.column_config.NumberColumn("Day %")
             },
-            use_container_width=True, hide_index=True, on_select="rerun", selection_mode="single-row"
+            use_container_width=True, hide_index=True, on_select="rerun", selection_mode="single-row",
+            key=f"{key_prefix}_table"
         )
         
         # CLICK LOGIC: Update Session State
         if len(event.selection['rows']) > 0:
             row_idx = event.selection['rows'][0]
             st.session_state.selected_ticker = df.iloc[row_idx]['Ticker']
+            st.session_state.scroll_to_analysis = True
+            st.rerun()
 
     with tab_us:
-        render_interactive_table(holdings[~holdings['Ticker'].str.contains(".HK")], "$")
+        render_interactive_table(holdings[~holdings['Ticker'].str.contains(".HK")], "$", "us")
 
     with tab_hk:
-        render_interactive_table(holdings[holdings['Ticker'].str.contains(".HK")], "HK$")
+        render_interactive_table(holdings[holdings['Ticker'].str.contains(".HK")], "HK$", "hk")
 
     with tab_hist:
         st.dataframe(df_raw.sort_index(ascending=False))
 
     # --- ANALYSIS SECTION ---
     st.divider()
-    
+    anchor_id = "analysis"
+    st.markdown(f"<div id='{anchor_id}'></div>", unsafe_allow_html=True)
+
     # Determine Target: 1. Clicked Row, 2. Dropdown Fallback
     current_selection = st.session_state.selected_ticker if st.session_state.selected_ticker else holdings.iloc[0]['Ticker']
     
     st.subheader(f"📊 Analysis: {STOCK_NAMES.get(current_selection, current_selection)}")
     
     col_a, col_b = st.columns([1,4])
     with col_a:
         # Dropdown syncs with click
-        selected_dd = st.selectbox("Select Stock", holdings['Ticker'].unique(), 
+        selected_dd = st.selectbox("Select Stock", holdings['Ticker'].unique(),
                                    index=list(holdings['Ticker']).index(current_selection) if current_selection in list(holdings['Ticker']) else 0)
         if selected_dd != current_selection:
             st.session_state.selected_ticker = selected_dd
             st.rerun()
+
+        if st.session_state.pop("scroll_to_analysis", False):
+            st.markdown(f"<script>window.location.hash = '{anchor_id}'</script>", unsafe_allow_html=True)
             
     with col_b:
         data = get_market_data(current_selection)
         if data is not None:
             fig = make_subplots(rows=2, cols=1, row_heights=[0.7, 0.3], shared_xaxes=True)
             fig.add_trace(go.Candlestick(x=data['Date'], open=data['Open'], close=data['Close'], high=data['High'], low=data['Low'], name="Price"), row=1, col=1)
             fig.add_trace(go.Scatter(x=data['Date'], y=data['SMA_20'], line=dict(color='#2ecc71', width=1), name="SMA 20"), row=1, col=1)
             fig.add_trace(go.Scatter(x=data['Date'], y=data['SMA_50'], line=dict(color='#3498db', width=1), name="SMA 50"), row=1, col=1)
             fig.add_trace(go.Scatter(x=data['Date'], y=data['SMA_100'], line=dict(color='#f39c12', width=1), name="SMA 100"), row=1, col=1)
             fig.add_trace(go.Scatter(x=data['Date'], y=data['SMA_250'], line=dict(color='#e74c3c', width=2), name="SMA 250"), row=1, col=1)
             fig.add_trace(go.Scatter(x=data['Date'], y=data['RSI'], line=dict(color='purple'), name="RSI"), row=2, col=1)
             fig.add_hline(y=30, row=2, col=1, line_dash="dot", line_color="green")
             fig.add_hline(y=70, row=2, col=1, line_dash="dot", line_color="red")
             fig.update_layout(height=700, xaxis_rangeslider_visible=False)
             st.plotly_chart(fig, use_container_width=True)
         else:
             st.error("Could not load data.")
-
 else:
-    st.info("Add a trade to see the dashboard.")
\ No newline at end of file
+    st.info("Add a trade to see the dashboard.")
