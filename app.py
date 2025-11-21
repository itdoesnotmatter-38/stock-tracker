import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import os
from datetime import datetime

# --- CONFIGURATION ---
st.set_page_config(layout="wide", page_title="Global Stock Tracker Pro")
FILE_NAME = "portfolio.csv"
USD_TO_HKD = 7.78
HKD_TO_USD = 1 / USD_TO_HKD

# --- 🔐 PASSWORD PROTECTION ---
def check_password():
    def password_entered():
        if st.session_state["password"] == "everyday38": 
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
    else:
        return True

if not check_password():
    st.stop()

# --- 1. STOCK NAMES (With Traditional Chinese) ---
STOCK_NAMES = {
    # US Stocks (English)
    "TSLA": "Tesla Inc", "NVDA": "NVIDIA Corp", "META": "Meta Platforms",
    "ORCL": "Oracle Corp", "PLTR": "Palantir Tech", "SOFI": "SoFi Tech",
    "QUBT": "Quantum Comp", "FLNC": "Fluence Energy", "SNOW": "Snowflake",
    "ZM": "Zoom Video", "FIG": "Figma (Private?)", "AAPL": "Apple Inc", 
    "MSFT": "Microsoft", "GOOGL": "Alphabet", "AMZN": "Amazon",
    
    # HK Stocks (Traditional Chinese)
    "9988.HK": "阿里巴巴", "0700.HK": "騰訊控股", "0388.HK": "香港交易所",
    "0981.HK": "中芯國際", "1810.HK": "小米集團", "1211.HK": "比亞迪股份",
    "0285.HK": "比亞迪電子", "9698.HK": "萬國數據", "3993.HK": "洛陽鉬業",
    "0909.HK": "明源雲", "9618.HK": "京東集團"
}

# --- SESSION STATE ---
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
            results.append({
                "Ticker": t, "Name": STOCK_NAMES.get(t, t),
                "Shares": data['shares'], "Avg Cost": data['total_cost'] / data['shares'],
                "Total Cost Basis": data['total_cost']
            })
    return pd.DataFrame(results)

# --- NEW: 10 YEAR DATA ---
def get_market_data(ticker):
    try:
        # Changed period to 10y
        df = yf.download(ticker, period="10y", progress=False)
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.droplevel(1)
        df.reset_index(inplace=True)
        
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

# --- NEW: NEWS FETCHER ---
def get_stock_news(ticker):
    try:
        stock = yf.Ticker(ticker)
        return stock.news[:5] # Return top 5 stories
    except:
        return []

# --- MAIN APP ---
st.title("🌏 Global Stock Tracker Pro")

# SIDEBAR
with st.sidebar.form("entry"):
    st.header("Add Trade")
    d = st.date_input("Date")
    a = st.selectbox("Action", ["Buy", "Sell"])
    t_input = st.text_input("Ticker (e.g. 9988.HK)", "").upper()
    q = st.number_input("Quantity", 1.0)
    p = st.number_input("Price", 0.01)
    if st.form_submit_button("Save"):
        save_transaction(d, t_input, a, q, p)
        st.success("Saved")
        st.rerun()

# --- PROCESSING ---
df_raw = load_portfolio()
holdings = calculate_holdings(df_raw)

if not holdings.empty:
    # FETCH 5 DAYS DATA FOR P/L
    tickers = " ".join(holdings['Ticker'].tolist())
    try:
        hist_data = yf.download(tickers, period="5d", progress=False)
        current_prices, prev_closes = [], []
        for ticker in holdings['Ticker']:
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

    # CALCULATIONS
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

    # GLOBAL TOTALS
    total_val_hkd = holdings['Value (HKD)'].sum()
    total_profit_hkd = holdings['Profit (HKD)'].sum()
    total_day_change_hkd = holdings['Day Change (HKD)'].sum()
    prev_total_val_hkd = total_val_hkd - total_day_change_hkd
    global_day_pct = (total_day_change_hkd / prev_total_val_hkd * 100) if prev_total_val_hkd > 0 else 0.0

    # HEADLINE
    c1, c2, c3 = st.columns(3)
    c1.metric("💰 Portfolio Value", f"HK$ {total_val_hkd:,.0f}")
    c2.metric("📈 Total Profit", f"HK$ {total_profit_hkd:,.0f}", delta_color="normal" if total_profit_hkd >= 0 else "inverse")
    c3.metric("⚡ Day's P/L", f"HK$ {total_day_change_hkd:,.0f}", delta=f"{global_day_pct:+.2f}%")
    st.divider()

    # TABS
    tab_us, tab_hk, tab_hist = st.tabs(["🇺🇸 US Market", "🇭🇰 HK Market", "📋 History"])

    def render_interactive_table(df, currency):
        mkt_val = df['Value (Native)'].sum()
        mkt_day_pl = df['Day Change $'].sum()
        prev_mkt_val = mkt_val - mkt_day_pl
        mkt_pct = (mkt_day_pl / prev_mkt_val * 100) if prev_mkt_val > 0 else 0.0

        c1, c2 = st.columns(2)
        c1.metric("Market Value", f"{currency} {mkt_val:,.0f}")
        c2.metric("Day's P/L", f"{currency} {mkt_day_pl:,.0f}", delta=f"{mkt_pct:+.2f}%")
        
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
            use_container_width=True, hide_index=True, on_select="rerun", selection_mode="single-row"
        )
        
        if len(event.selection['rows']) > 0:
            row_idx = event.selection['rows'][0]
            st.session_state.selected_ticker = df.iloc[row_idx]['Ticker']

    with tab_us: render_interactive_table(holdings[~holdings['Ticker'].str.contains(".HK")], "$")
    with tab_hk: render_interactive_table(holdings[holdings['Ticker'].str.contains(".HK")], "HK$")
    with tab_hist: st.dataframe(df_raw.sort_index(ascending=False))

    # --- ANALYSIS SECTION ---
    st.divider()
    st.subheader("📊 Professional Analysis & News")
    
    current_selection = st.session_state.selected_ticker if st.session_state.selected_ticker else holdings.iloc[0]['Ticker']
    
    col_chart, col_news = st.columns([3, 1]) # Chart is 3x wider than news
    
    # --- CHART COLUMN ---
    with col_chart:
        col_a, col_b = st.columns([1,4])
        with col_a:
            selected_dd = st.selectbox("Select Stock", holdings['Ticker'].unique(), 
                                    index=list(holdings['Ticker']).index(current_selection) if current_selection in list(holdings['Ticker']) else 0)
            if selected_dd != current_selection:
                st.session_state.selected_ticker = selected_dd
                st.rerun()
                
        target = st.session_state.selected_ticker if st.session_state.selected_ticker else selected_dd
        st.markdown(f"### {STOCK_NAMES.get(target, target)} ({target})")
        data = get_market_data(target)
        
        if data is not None:
            fig = make_subplots(rows=2, cols=1, row_heights=[0.7, 0.3], shared_xaxes=True)
            
            # Price & MA
            fig.add_trace(go.Candlestick(x=data['Date'], open=data['Open'], close=data['Close'], high=data['High'], low=data['Low'], name="Price"), row=1, col=1)
            fig.add_trace(go.Scatter(x=data['Date'], y=data['SMA_20'], line=dict(color='#2ecc71', width=1), name="SMA 20"), row=1, col=1)
            fig.add_trace(go.Scatter(x=data['Date'], y=data['SMA_50'], line=dict(color='#3498db', width=1), name="SMA 50"), row=1, col=1)
            fig.add_trace(go.Scatter(x=data['Date'], y=data['SMA_100'], line=dict(color='#f39c12', width=1), name="SMA 100"), row=1, col=1)
            fig.add_trace(go.Scatter(x=data['Date'], y=data['SMA_250'], line=dict(color='#e74c3c', width=2), name="SMA 250"), row=1, col=1)
            
            # RSI
            fig.add_trace(go.Scatter(x=data['Date'], y=data['RSI'], line=dict(color='purple'), name="RSI"), row=2, col=1)
            fig.add_hline(y=30, row=2, col=1, line_dash="dot", line_color="green")
            fig.add_hline(y=70, row=2, col=1, line_dash="dot", line_color="red")
            
            # NEW: INTERACTIVE SELECTORS
            fig.update_layout(
                height=700, 
                xaxis_rangeslider_visible=False,
                xaxis=dict(
                    rangeselector=dict(
                        buttons=list([
                            dict(count=1, label="1m", step="month", stepmode="backward"),
                            dict(count=6, label="6m", step="month", stepmode="backward"),
                            dict(count=1, label="YTD", step="year", stepmode="todate"),
                            dict(count=1, label="1y", step="year", stepmode="backward"),
                            dict(step="all")
                        ])
                    ),
                    type="date"
                )
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.error("Could not load data.")

    # --- NEWS COLUMN ---
    with col_news:
        st.write(f"**📰 Latest News**")
        news_items = get_stock_news(target)
        if news_items:
            for item in news_items:
                try:
                    title = item.get('title', 'No Title')
                    link = item.get('link', '#')
                    publisher = item.get('publisher', 'Unknown')
                    
                    # Simple card layout
                    st.markdown(f"""
                    <div style="padding: 10px; border-radius: 5px; margin-bottom: 10px; background-color: #f0f2f6;">
                        <a href="{link}" target="_blank" style="text-decoration: none; color: #333; font-weight: bold;">{title}</a>
                        <br>
                        <span style="font-size: 0.8em; color: #666;">{publisher}</span>
                    </div>
                    """, unsafe_allow_html=True)
                except:
                    pass
        else:
            st.info("No news found.")

else:
    st.info("Add a trade to see the dashboard.")