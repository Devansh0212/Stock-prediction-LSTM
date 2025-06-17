# Stock Analysis Pro 
import numpy as np
import pandas as pd
import yfinance as yf
from keras.models import load_model
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from newsapi import NewsApiClient
from textblob import TextBlob
import plotly.express as px
import time
from datetime import datetime, timedelta

# Set pandas options to handle future warnings
#pd.set_option('future.no_silent_downcasting', True)

# Apple-style Configuration
st.set_page_config(
    layout="wide", 
    page_title="Stock Analysis Pro", 
    page_icon="📈",
    initial_sidebar_state="expanded"
)

# Custom CSS for Apple-like styling

st.markdown("""
<style>
    /* Main styling for dark mode */
    .stApp {
        background-color: #18191A;
        color: #F3F3F3;
        font-family: -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #23272F !important;
        border-right: 1px solid #444950;
        color: #F3F3F3 !important;
    }

    /* Headers */
    h1, h2, h3, h4, h5, h6 {
        color: #F3F3F3 !important;
        font-weight: 600 !important;
    }

    /* Cards and containers */
    .stMetric, .apple-card {
        background-color: #23272F;
        border-radius: 12px;
        padding: 15px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.3);
        color: #F3F3F3;
    }

    /* Buttons */
    .stButton>button {
        background-color: #0071e3;
        color: #F3F3F3;
        border-radius: 980px;
        padding: 8px 16px;
        font-size: 14px;
        font-weight: 400;
        border: none;
    }

    .stButton>button:hover {
        background-color: #005bb5;
    }

    /* Tabs */
    [data-baseweb="tab-list"] {
        gap: 10px;
    }

    [data-baseweb="tab"] {
        background-color: #23272F;
        border-radius: 8px 8px 0 0 !important;
        padding: 8px 16px;
        margin: 0 !important;
        color: #F3F3F3 !important;
    }

    [aria-selected="true"] {
        background-color: #0071e3 !important;
        color: #F3F3F3 !important;
    }

    /* Inputs */
    .stTextInput>div>div>input {
        border-radius: 12px !important;
        padding: 10px 12px !important;
        background-color: #23272F !important;
        color: #F3F3F3 !important;
        border: 1px solid #444950 !important;
    }

    /* Dataframes */
    .stDataFrame {
        border-radius: 12px;
        overflow: hidden;
        color: #F3F3F3 !important;
        background-color: #23272F !important;
    }

    /* Custom classes */
    .apple-header {
        font-size: 32px !important;
        font-weight: 700 !important;
        margin-bottom: 8px !important;
        color: #F3F3F3 !important;
    }

    .apple-subheader {
        font-size: 24px !important;
        font-weight: 600 !important;
        margin-bottom: 16px !important;
        color: #B0B3B8 !important;
    }

    .apple-card {
        background-color: #23272F;
        border-radius: 18px;
        padding: 20px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        margin-bottom: 20px;
        color: #F3F3F3;
    }

    .apple-metric {
        font-size: 42px;
        font-weight: 600;
        color: #F3F3F3;
    }

    .apple-metric-label {
        font-size: 14px;
        color: #B0B3B8;
        margin-bottom: 4px;
    }

    .apple-divider {
        height: 1px;
        background-color: #444950;
        margin: 20px 0;
    }

    .positive {
        color: #34C759 !important;
    }

    .negative {
        color: #FF3B30 !important;
    }
</style>
""", unsafe_allow_html=True)


# Constants
MODEL_PATH = 'Stock Predictions Model.keras'
DEFAULT_STOCK = 'AAPL'
DEFAULT_START_DATE = '2012-01-01'
DEFAULT_END_DATE = '2022-12-31'

@st.cache_resource
def load_lstm_model(path):
    try:
        return load_model(path)
    except Exception as e:
        st.error(f"Model loading failed: {e}")
        return None

@st.cache_data(ttl=3600)
def fetch_stock_data(symbol, start_date, end_date, max_retries=3):
    try:
        if start_date > end_date:
            st.error("Error: Start date must be before end date")
            return pd.DataFrame()
        
        symbol_variations = [symbol, f"{symbol}.NS", f"{symbol}.BO", f"{symbol}.AX", f"{symbol}.TO"]
        
        for sym in symbol_variations:
            try:
                df = yf.download(
                    sym,
                    start=start_date,
                    end=end_date + timedelta(days=1),
                    progress=False,
                    auto_adjust=True,
                    threads=True,
                    timeout=10
                )
                if not df.empty:
                    return df
            except Exception:
                continue
        
        st.error(f"Could not fetch data for {symbol}")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Data download failed: {str(e)}")
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def get_fundamental_data(ticker):
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        if not info:
            stock = yf.Ticker(ticker + ".NS")
            info = stock.info
        
        fundamentals = {
            'valuation': {
                'market_cap': info.get('marketCap', info.get('totalAssets', 'N/A')),
                'pe_ratio': info.get('trailingPE', info.get('forwardPE', 'N/A')),
                'forward_pe': info.get('forwardPE', 'N/A'),
                'price_to_book': info.get('priceToBook', 'N/A'),
                'peg_ratio': info.get('pegRatio', 'N/A'),
                'enterprise_value': info.get('enterpriseValue', 'N/A')
            },
            'profitability': {
                'roa': info.get('returnOnAssets', info.get('returnOnEquity', 'N/A')),
                'roe': info.get('returnOnEquity', 'N/A'),
                'profit_margins': info.get('profitMargins', 'N/A'),
                'operating_margins': info.get('operatingMargins', 'N/A')
            },
            'financial_health': {
                'debt_to_equity': info.get('debtToEquity', 'N/A'),
                'current_ratio': info.get('currentRatio', 'N/A'),
                'quick_ratio': info.get('quickRatio', 'N/A')
            }
        }
        
        for category in fundamentals:
            for metric, value in fundamentals[category].items():
                if isinstance(value, str) and value.replace('.','',1).isdigit():
                    fundamentals[category][metric] = float(value)
                    
        return fundamentals
        
    except Exception as e:
        st.error(f"Fundamental data fetch error: {e}")
        return None

@st.cache_data(ttl=3600)
def get_enhanced_fundamentals(ticker):
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        if not info:
            stock = yf.Ticker(ticker + ".NS")
            info = stock.info
        
        financials = stock.financials
        balance_sheet = stock.balance_sheet
        cashflow = stock.cashflow
        
        enterprise_value = info.get('enterpriseValue', None)
        ebitda = info.get('ebitda', None)
        free_cash_flow = info.get('freeCashflow', None)
        shares_outstanding = info.get('sharesOutstanding', None)
        
        ev_to_ebitda = enterprise_value / ebitda if (enterprise_value and ebitda) else None
        price_to_sales = info.get('priceToSalesTrailing12Months', None)
        free_cash_flow_yield = (free_cash_flow / enterprise_value) if (free_cash_flow and enterprise_value) else None
        
        if financials is not None and not financials.empty:
            revenue_growth = (financials.loc['Total Revenue'].iloc[0] / financials.loc['Total Revenue'].iloc[1] - 1) * 100 \
                            if len(financials.loc['Total Revenue']) > 1 else None
            eps_growth = (info.get('earningsGrowth', None) * 100) if info.get('earningsGrowth', None) else None
        else:
            revenue_growth = eps_growth = None
        
        roic = info.get('returnOnInvestedCapital', None)
        dividend_yield = info.get('dividendYield', None) * 100 if info.get('dividendYield', None) else None
        payout_ratio = info.get('payoutRatio', None)
        
        return {
            'valuation': {
                'ev_ebitda': ev_to_ebitda,
                'price_to_sales': price_to_sales,
                'fcf_yield': free_cash_flow_yield,
                'dividend_yield': dividend_yield,
                'payout_ratio': payout_ratio
            },
            'growth': {
                'revenue_growth': revenue_growth,
                'eps_growth': eps_growth
            },
            'profitability': {
                'roic': roic
            }
        }
    except Exception as e:
        st.error(f"Enhanced fundamentals fetch error: {e}")
        return None

@st.cache_data(ttl=3600)
def get_financial_statements(ticker):
    try:
        stock = yf.Ticker(ticker)
        income = stock.financials
        balance = stock.balance_sheet
        cashflow = stock.cashflow
        
        if income is None or income.empty:
            st.warning(f"Income statement data not available for {ticker}")
            return None
            
        income_clean = income.replace(0, np.nan).infer_objects(copy=False)
        income_growth = None
        try:
            with np.errstate(divide='ignore', invalid='ignore'):
                income_growth = income_clean.pct_change(axis=1, fill_method=None) * 100
                income_growth = income_growth.replace([np.inf, -np.inf], np.nan)
        except Exception as e:
            st.warning(f"Could not calculate income growth: {str(e)}")
        
        balance_ratios = None
        if balance is not None and not balance.empty:
            try:
                balance = balance.copy()
                if 'Current Assets' in balance.index and 'Current Liabilities' in balance.index:
                    with np.errstate(divide='ignore', invalid='ignore'):
                        current_ratio = balance.loc['Current Assets'] / balance.loc['Current Liabilities']
                        current_ratio = current_ratio.replace([np.inf, -np.inf], np.nan).infer_objects(copy=False)
                        balance.loc['Current Ratio'] = current_ratio
                
                if 'Total Liab' in balance.index and 'Stockholders Equity' in balance.index:
                    with np.errstate(divide='ignore', invalid='ignore'):
                        debt_equity = balance.loc['Total Liab'] / balance.loc['Stockholders Equity']
                        debt_equity = debt_equity.replace([np.inf, -np.inf], np.nan)
                        balance.loc['Debt/Equity'] = debt_equity
            except Exception as e:
                st.warning(f"Could not calculate balance sheet ratios: {str(e)}")
        
        return {
            'income_statement': income,
            'income_growth': income_growth,
            'balance_sheet': balance,
            'cash_flow': cashflow
        }
    except Exception as e:
        st.error(f"Financial statements fetch error: {e}")
        return None  

def plot_financial_statement(df, title, height=400):
    if df is None or df.empty:
        st.warning(f"No {title.lower()} data available")
        return
    
    df = df.copy()
    df.columns = df.columns.strftime('%Y-%m-%d')
    df = df.sort_index(axis=1)
    
    def format_numbers(x):
        if abs(x) >= 1e9:
            return f"${x/1e9:.2f}B"
        elif abs(x) >= 1e6:
            return f"${x/1e6:.2f}M"
        elif abs(x) >= 1e3:
            return f"${x/1e3:.2f}K"
        return f"${x:.2f}"
    
    styled_df = df.style.format(format_numbers)
    st.markdown(f"#### {title}")
    st.dataframe(styled_df, height=height)

def plot_financial_growth(df, title):
    if df is None or df.empty:
        return
    
    df = df.copy()
    df.columns = df.columns.strftime('%Y')
    df = df.sort_index(axis=1)
    
    metrics = st.multiselect(
        f"Select metrics to plot ({title})",
        options=df.index.tolist(),
        default=df.index[[0, -1]].tolist()
    )
    
    if metrics:
        fig = px.line(
            df.loc[metrics].T,
            title=f"{title} Growth Trends",
            labels={'value': 'Growth %', 'variable': 'Metric'},
            height=400
        )
        fig.update_layout(
            hovermode='x unified',
            yaxis_tickformat='.2f%',
            legend_title_text='Metric'
        )
        st.plotly_chart(fig, use_container_width=True)

@st.cache_data(ttl=3600)
def get_peers(ticker):
    try:
        stock = yf.Ticker(ticker)
        peers = stock.info.get('peerSymbols', [])
        
        if not peers:
            industry = stock.info.get('industry', '')
            sector = stock.info.get('sector', '')
            
            sector_peers = {
                'Technology': ['AAPL', 'MSFT', 'GOOGL', 'META', 'NVDA'],
                'Financial Services': ['JPM', 'BAC', 'GS', 'MS', 'C'],
                'Healthcare': ['PFE', 'JNJ', 'MRK', 'ABT', 'GILD'],
                'Consumer Cyclical': ['AMZN', 'TSLA', 'HD', 'NKE', 'MCD']
            }
            
            if sector in sector_peers:
                peers = sector_peers[sector]
            else:
                peers = ['SPY', 'QQQ', 'DIA']
        
        return list(set(peers))[:5]
    except Exception as e:
        st.error(f"Peer identification error: {e}")
        return []

@st.cache_data(ttl=3600)
def get_comps_data(main_ticker, peer_tickers):
    comps_data = []
    
    main_data = get_enhanced_fundamentals(main_ticker)
    if main_data:
        comps_data.append({
            'ticker': main_ticker,
            'name': str(yf.Ticker(main_ticker).info.get('shortName', main_ticker)),
            'pe_ratio': safe_float(main_data['valuation'].get('pe_ratio')),
            'ev_ebitda': safe_float(main_data['valuation'].get('ev_ebitda')),
            'price_to_sales': safe_float(main_data['valuation'].get('price_to_sales')),
            'price_to_book': safe_float(main_data['valuation'].get('price_to_book'))
        })
    
    for ticker in peer_tickers:
        try:
            peer_data = get_enhanced_fundamentals(ticker)
            if peer_data:
                comps_data.append({
                    'ticker': ticker,
                    'name': str(yf.Ticker(ticker).info.get('shortName', ticker)),
                    'pe_ratio': safe_float(peer_data['valuation'].get('pe_ratio')),
                    'ev_ebitda': safe_float(peer_data['valuation'].get('ev_ebitda')),
                    'price_to_sales': safe_float(peer_data['valuation'].get('price_to_sales')),
                    'price_to_book': safe_float(peer_data['valuation'].get('price_to_book'))
                })
        except Exception as e:
            st.warning(f"Could not get data for {ticker}: {str(e)}")
            continue
    
    df = pd.DataFrame(comps_data).set_index('ticker')
    return df

def safe_float(value):
    try:
        return float(value) if value is not None else np.nan
    except (ValueError, TypeError):
        return np.nan

def plot_comps_chart(comps_df, metric):
    if comps_df.empty or metric not in comps_df.columns:
        return None
        
    fig = px.bar(
        comps_df.reset_index(),
        x='name',
        y=metric,
        title=f'{metric.upper()} Comparison',
        labels={'name': 'Company', metric: metric.upper()},
        color='name',
        height=400
    )
    fig.update_layout(showlegend=False)
    return fig

def generate_recommendation(fundamentals, technicals, sentiment):
    try:
        scores = {
            'valuation': 0,
            'profitability': 0,
            'growth': 0,
            'dividends': 0,
            'financial_health': 0,
            'technical': 0,
            'sentiment': 0
        }
        
        pe = fundamentals['valuation']['pe_ratio']
        if pe != 'N/A':
            if isinstance(pe, (int, float)):
                if pe < 15:
                    scores['valuation'] = 1
                elif pe < 25:
                    scores['valuation'] = 0.5
                else:
                    scores['valuation'] = -0.5
        
        roe = fundamentals['profitability']['roe']
        if roe != 'N/A':
            if isinstance(roe, (int, float)):
                if roe > 0.15:
                    scores['profitability'] = 1
                elif roe > 0:
                    scores['profitability'] = 0.5
                else:
                    scores['profitability'] = -1
        
        rsi = technicals.get('rsi', float('nan'))
        if not np.isnan(rsi):
            if rsi < 30:
                scores['technical'] = 1
            elif rsi > 70:
                scores['technical'] = -1
        
        if isinstance(sentiment, (int, float)):
            if sentiment > 0.2:
                scores['sentiment'] = 1
            elif sentiment < -0.2:
                scores['sentiment'] = -1
        
        total_score = sum(scores.values())
        
        if total_score >= 3:
            return "Strong Buy", total_score, "positive"
        elif total_score >= 1:
            return "Buy", total_score, "positive"
        elif total_score >= -1:
            return "Hold", total_score, ""
        elif total_score >= -3:
            return "Sell", total_score, "negative"
        else:
            return "Strong Sell", total_score, "negative"
    
    except Exception as e:
        st.error(f"Recommendation generation failed: {e}")
        return "Hold", 0.0, ""

@st.cache_data(ttl=3600)
def get_dcf_inputs(ticker):
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        cashflow = stock.cashflow
        if cashflow is not None and not cashflow.empty:
            fcf = cashflow.loc['Free Cash Flow', cashflow.columns[0]]
        else:
            fcf = info.get('freeCashflow', None)
        
        current_data = {
            'free_cash_flow': fcf,
            'cash': info.get('totalCash', None),
            'debt': info.get('totalDebt', None),
            'shares': info.get('sharesOutstanding', None),
            'beta': info.get('beta', None),
            'current_price': info.get('currentPrice', None)
        }
        
        return current_data
    except Exception as e:
        st.error(f"DCF data fetch error: {e}")
        return None

def calculate_dcf(fcf, growth_rate, discount_rate, terminal_growth, years=5):
    try:
        cash_flows = []
        for year in range(1, years + 1):
            cash_flows.append(fcf * (1 + growth_rate) ** year)
        
        terminal_value = (cash_flows[-1] * (1 + terminal_growth)) / (discount_rate - terminal_growth)
        
        discount_factors = [(1 + discount_rate) ** year for year in range(1, years + 1)]
        
        pv_cash_flows = [cf / df for cf, df in zip(cash_flows, discount_factors)]
        pv_terminal = terminal_value / ((1 + discount_rate) ** years)
        
        enterprise_value = sum(pv_cash_flows) + pv_terminal
        
        return {
            'enterprise_value': enterprise_value,
            'cash_flows': cash_flows,
            'terminal_value': terminal_value,
            'discount_factors': discount_factors,
            'years': years
        }
    except Exception as e:
        st.error(f"DCF calculation error: {e}")
        return None

def calculate_indicators(data):
    try:
        if data.empty or 'Close' not in data.columns:
            return pd.Series(), pd.Series(), pd.Series(), float('nan')
        
        delta = data['Close'].diff(1)
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        
        avg_gain = gain.rolling(window=14, min_periods=1).mean()
        avg_loss = loss.rolling(window=14, min_periods=1).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        ema_12 = data['Close'].ewm(span=12, adjust=False).mean()
        ema_26 = data['Close'].ewm(span=26, adjust=False).mean()
        macd = ema_12 - ema_26
        signal_line = macd.ewm(span=9, adjust=False).mean()
        
        last_rsi = float(rsi.iloc[-1].item()) if not rsi.empty else float('nan')
        
        return rsi, macd, signal_line, last_rsi
    except Exception as e:
        st.error(f"Technical indicator calculation failed: {e}")
        return pd.Series(), pd.Series(), pd.Series(), float('nan')

def calculate_risk_metrics(data):
    try:
        if data.empty or 'Close' not in data.columns:
            return {
                'annual_volatility': float('nan'),
                'max_drawdown': float('nan'),
                'sharpe_ratio': float('nan'),
                'sortino_ratio': float('nan')
            }
            
        returns = data['Close'].pct_change().dropna()
        if len(returns) < 2:
            return {
                'annual_volatility': float('nan'),
                'max_drawdown': float('nan'),
                'sharpe_ratio': float('nan'),
                'sortino_ratio': float('nan')
            }
            
        annual_vol = returns.std() * np.sqrt(252)
        max_dd = (data['Close'] / data['Close'].cummax() - 1).min()
        sharpe = (returns.mean() / returns.std()) * np.sqrt(252)
        sortino = (returns.mean() / returns[returns < 0].std()) * np.sqrt(252)
        
        return {
            'annual_volatility': float(annual_vol.iloc[0]) if isinstance(annual_vol, pd.Series) else float(annual_vol),
            'max_drawdown': float(max_dd.iloc[0]) if isinstance(max_dd, pd.Series) else float(max_dd),
            'sharpe_ratio': float(sharpe.iloc[0]) if isinstance(sharpe, pd.Series) else float(sharpe),
            'sortino_ratio': float(sortino.iloc[0]) if isinstance(sortino, pd.Series) else float(sortino)
        }
    except Exception as e:
        st.error(f"Risk calculation failed: {e}")
        return {
            'annual_volatility': float('nan'),
            'max_drawdown': float('nan'),
            'sharpe_ratio': float('nan'),
            'sortino_ratio': float('nan')
        }

def prepare_data(data, model):
    try:
        if len(data) < 120:
            st.error("Not enough data for prediction. Need at least 120 days of data.")
            return None, None

        scaler = MinMaxScaler(feature_range=(0, 1))
        data_scaled = scaler.fit_transform(data[['Close']])
        
        X, y = [], []
        for i in range(100, len(data_scaled)):
            X.append(data_scaled[i-100:i])
            y.append(data_scaled[i, 0])
            
        X, y = np.array(X), np.array(y)
        predictions = model.predict(X)
        
        scale_factor = 1 / scaler.scale_[0]
        predictions = predictions * scale_factor
        y = y * scale_factor
        
        return y, predictions
    except Exception as e:
        st.error(f"Data preparation failed: {e}")
        return None, None

def create_apple_metric(label, value, delta=None, delta_type=None):
    delta_class = ""
    if delta_type == "positive":
        delta_class = "positive"
    elif delta_type == "negative":
        delta_class = "negative"
    
    html = f"""
    <div class="apple-card" style="padding: 16px; text-align: center;">
        <div class="apple-metric-label">{label}</div>
        <div class="apple-metric">{value}</div>
        {f'<div class="{delta_class}" style="font-size: 14px; margin-top: 4px;">{delta}</div>' if delta is not None else ''}
    </div>
    """
    return html

@st.cache_data(ttl=3600)
def get_company_info(ticker):
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        if not info:
            stock = yf.Ticker(ticker + ".NS")
            info = stock.info
        
        return {
            'name': info.get('longName', info.get('shortName', ticker)),
            'sector': info.get('sector', 'N/A'),
            'industry': info.get('industry', 'N/A'),
            'country': info.get('country', 'N/A'),
            'website': info.get('website', 'N/A'),
            'summary': info.get('longBusinessSummary', 'No description available')
        }
    except Exception as e:
        st.error(f"Company info fetch error: {e}")
        return {
            'name': ticker,
            'sector': 'N/A',
            'industry': 'N/A',
            'country': 'N/A',
            'website': 'N/A',
            'summary': 'No description available'
        }

def main():
    model = load_lstm_model(MODEL_PATH)
    if model is None:
        st.stop()

    # Sidebar
    with st.sidebar:
        st.markdown("""
        <div style="padding: 16px 0 24px 0;">
            <h1 style="font-size: 24px; margin-bottom: 0;">Stock Analysis Pro</h1>
            <p style="color: #86868b; font-size: 14px; margin-top: 4px;">Premium financial insights</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("### Stock Selection")
        stock_symbol = st.text_input('Symbol', DEFAULT_STOCK).upper()
        
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input('Start Date', pd.to_datetime(DEFAULT_START_DATE))
        with col2:
            end_date = st.date_input('End Date', pd.to_datetime(DEFAULT_END_DATE))
        
        if st.button('Analyze', use_container_width=True):
            st.session_state.analyze_clicked = True
        else:
            st.session_state.analyze_clicked = False
            
        st.markdown("---")
        st.markdown("""
        <div style="color: #86868b; font-size: 12px; text-align: center;">
            <p>Stock Analysis Pro</p>
            <p>Version 2.0</p>
        </div>
        """, unsafe_allow_html=True)

    if not st.session_state.get('analyze_clicked', False):
        st.markdown("""
        <div style="text-align: center; padding: 80px 20px;">
            <h1 class="apple-header">Stock Analysis Pro</h1>
            <p style="font-size: 20px; color: #86868b; max-width: 600px; margin: 0 auto 40px auto;">
                Professional-grade stock analysis with institutional-quality insights
            </p>
            <div style="margin-top: 40px;">
                <p style="color: #86868b;">Enter a stock symbol and date range to begin</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.stop()

    # Fetch data
    with st.spinner('Loading data...'):
        st.session_state.data = pd.DataFrame()
        st.session_state.fundamentals = None
        st.session_state.sentiment = {
            'overall_sentiment': 0,
            'source_details': {
                'news': {'score': 0, 'top_news': []},
                'twitter': {'score': 0},
                'reddit': {'score': 0}
            }
        }
        st.session_state.technical = {
            'rsi': pd.Series(),
            'macd': pd.Series(),
            'signal_line': pd.Series(),
            'last_rsi': float('nan')
        }
        st.session_state.risk_metrics = {
            'annual_volatility': float('nan'),
            'max_drawdown': float('nan'),
            'sharpe_ratio': float('nan'),
            'sortino_ratio': float('nan')
        }
        
        data = fetch_stock_data(stock_symbol, start_date, end_date)
        if data.empty:
            st.error("No data available for the selected stock and date range")
            st.stop()
        
        st.session_state.data = data
        st.session_state.fundamentals = get_fundamental_data(stock_symbol)
        enhanced_fundamentals = get_enhanced_fundamentals(stock_symbol)
        financials = get_financial_statements(stock_symbol)
        rsi, macd, signal_line, last_rsi = calculate_indicators(data)
        st.session_state.technical = {
            'rsi': rsi,
            'macd': macd,
            'signal_line': signal_line,
            'last_rsi': last_rsi
        }
        st.session_state.risk_metrics = calculate_risk_metrics(data)

    if 'data' not in st.session_state or st.session_state.data.empty:
        st.info("Enter a stock symbol and click 'Analyze' to begin")
        st.stop()

    company_info = get_company_info(stock_symbol)
    
    st.markdown(f"""
    <div style="margin-bottom: 24px;">
        <div style="display: flex; align-items: center; gap: 16px; margin-bottom: 8px;">
            <h1 class="apple-header">{company_info['name']}</h1>
            <div style="background-color: #f5f5f7; padding: 4px 12px; border-radius: 20px; 
                        font-size: 16px; font-weight: 500; color: #86868b;">
                {stock_symbol}
            </div>
        </div>
        <p style="color: #86868b; font-size: 16px; margin-bottom: 0;">
            {company_info['sector']} • {company_info['industry']} • {company_info['country']}
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    with st.expander("Company Overview", expanded=False):
        st.markdown(f"""
        <div style="margin-bottom: 16px;">
            <p style="margin-bottom: 8px;"><strong>Website:</strong> <a href="{company_info['website']}" target="_blank">{company_info['website']}</a></p>
            <p>{company_info['summary']}</p>
        </div>
        """, unsafe_allow_html=True)

    if st.session_state.fundamentals:
        rec, score, rec_type = generate_recommendation(
            st.session_state.fundamentals,
            {'rsi': st.session_state.technical['last_rsi']},
            st.session_state.sentiment['overall_sentiment']
        )
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(create_apple_metric(
            "Recommendation", 
            rec, 
            f"Score: {score:.1f}/7.0",
            rec_type
        ), unsafe_allow_html=True)
    
    with col2:
        try:
            price_series = st.session_state.data['Close'].iloc[-1]
            current_price = float(price_series.item()) if isinstance(price_series, pd.Series) else float(price_series)
            st.markdown(create_apple_metric(
                "Current Price", 
                f"${current_price:.2f}"
            ), unsafe_allow_html=True)
        except Exception:
            st.markdown(create_apple_metric(
                "Current Price", 
                "N/A"
            ), unsafe_allow_html=True)
    
    with col3:
        sentiment_value = st.session_state.sentiment['overall_sentiment']
        sentiment_label = "Positive" if sentiment_value > 0.2 else "Negative" if sentiment_value < -0.2 else "Neutral"
        sentiment_type = "positive" if sentiment_value > 0.2 else "negative" if sentiment_value < -0.2 else ""
        st.markdown(create_apple_metric(
            "Market Sentiment", 
            f"{sentiment_value:.2f}",
            sentiment_label,
            sentiment_type
        ), unsafe_allow_html=True)
    
    with col4:
        rsi_value = st.session_state.technical.get('last_rsi', float('nan'))
        rsi_display = f"{rsi_value:.1f}" if not np.isnan(rsi_value) else "N/A"
        rsi_status = ""
        if not np.isnan(rsi_value):
            rsi_status = "Overbought (>70)" if rsi_value > 70 else "Oversold (<30)" if rsi_value < 30 else "Neutral"
            rsi_type = "negative" if rsi_value > 70 else "positive" if rsi_value < 30 else ""
        st.markdown(create_apple_metric(
            "RSI (14-day)", 
            rsi_display,
            rsi_status,
            rsi_type if not np.isnan(rsi_value) else ""
        ), unsafe_allow_html=True)

    st.markdown("### Price Performance")
    with st.container():
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(st.session_state.data['Close'], color='#0071e3', linewidth=2)
        ax.set_facecolor('#f5f5f7')
        ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_color('#d2d2d7')
        ax.spines['left'].set_color('#d2d2d7')
        st.pyplot(fig)

    tabs = st.tabs([
        "📊 Fundamentals", 
        "📉 Technical Analysis",
        "🔮 Predictions",
        "😀 Sentiment",
        "📝 Raw Data",
        "🏢 Comps Analysis",
        "💰 DCF Valuation"
    ])  

    # Fundamentals Tab
    with tabs[0]:
        st.markdown("### Fundamental Analysis")
        
        if not st.session_state.fundamentals:
            st.warning("Fundamental data not available for this stock")
            st.stop()
        
        st.markdown("#### Valuation")
        val_cols = st.columns(4)
        
        market_cap = st.session_state.fundamentals['valuation'].get('market_cap', 'N/A')
        if isinstance(market_cap, (int, float)):
            if market_cap >= 1e12:
                display = f"${market_cap/1e12:.2f}T"
            elif market_cap >= 1e9:
                display = f"${market_cap/1e9:.2f}B"
            else:
                display = f"${market_cap/1e6:.2f}M"
        else:
            display = "N/A"
        
        val_cols[0].markdown(create_apple_metric("Market Cap", display), unsafe_allow_html=True)
        
        pe = st.session_state.fundamentals['valuation'].get('pe_ratio', 'N/A')
        pe_display = str(pe) if pe != 'N/A' else "N/A"
        val_cols[1].markdown(create_apple_metric("P/E Ratio", pe_display), unsafe_allow_html=True)
        
        fpe = st.session_state.fundamentals['valuation'].get('forward_pe', 'N/A')
        fpe_display = str(fpe) if fpe != 'N/A' else "N/A"
        val_cols[2].markdown(create_apple_metric("Forward P/E", fpe_display), unsafe_allow_html=True)
        
        pb = st.session_state.fundamentals['valuation'].get('price_to_book', 'N/A')
        pb_display = str(pb) if pb != 'N/A' else "N/A"
        val_cols[3].markdown(create_apple_metric("P/B Ratio", pb_display), unsafe_allow_html=True)
        
        st.markdown("#### Profitability")
        prof_cols = st.columns(3)
        
        roe = st.session_state.fundamentals['profitability'].get('roe', 'N/A')
        roe_display = f"{roe*100:.2f}%" if isinstance(roe, (int, float)) else "N/A"
        prof_cols[0].markdown(create_apple_metric("Return on Equity", roe_display), unsafe_allow_html=True)
        
        margins = st.session_state.fundamentals['profitability'].get('profit_margins', 'N/A')
        margin_display = f"{margins*100:.2f}%" if isinstance(margins, (int, float)) else "N/A"
        prof_cols[1].markdown(create_apple_metric("Profit Margins", margin_display), unsafe_allow_html=True)
        
        op_margins = st.session_state.fundamentals['profitability'].get('operating_margins', 'N/A')
        op_margin_display = f"{op_margins*100:.2f}%" if isinstance(op_margins, (int, float)) else "N/A"
        prof_cols[2].markdown(create_apple_metric("Operating Margins", op_margin_display), unsafe_allow_html=True)
        
        st.markdown("#### Financial Health")
        health_cols = st.columns(3)
        
        de = st.session_state.fundamentals['financial_health'].get('debt_to_equity', 'N/A')
        de_display = str(de) if de != 'N/A' else "N/A"
        health_cols[0].markdown(create_apple_metric("Debt/Equity", de_display), unsafe_allow_html=True)
        
        cr = st.session_state.fundamentals['financial_health'].get('current_ratio', 'N/A')
        cr_display = str(cr) if cr != 'N/A' else "N/A"
        health_cols[1].markdown(create_apple_metric("Current Ratio", cr_display), unsafe_allow_html=True)
        
        qr = st.session_state.fundamentals['financial_health'].get('quick_ratio', 'N/A')
        qr_display = str(qr) if qr != 'N/A' else "N/A"
        health_cols[2].markdown(create_apple_metric("Quick Ratio", qr_display), unsafe_allow_html=True)

        if enhanced_fundamentals:
            st.markdown("#### Advanced Valuation")
            adv_val_cols = st.columns(4)
                
            ev_ebitda = enhanced_fundamentals['valuation']['ev_ebitda']
            adv_val_cols[0].markdown(create_apple_metric(
                "EV/EBITDA", 
                f"{ev_ebitda:.2f}" if ev_ebitda else "N/A"
            ), unsafe_allow_html=True)
            
            ps = enhanced_fundamentals['valuation']['price_to_sales']
            adv_val_cols[1].markdown(create_apple_metric(
                "Price/Sales", 
                f"{ps:.2f}" if ps else "N/A"
            ), unsafe_allow_html=True)
            
            fcf_yield = enhanced_fundamentals['valuation']['fcf_yield']
            adv_val_cols[2].markdown(create_apple_metric(
                "FCF Yield", 
                f"{fcf_yield:.2%}" if fcf_yield else "N/A"
            ), unsafe_allow_html=True)
            
            div_yield = enhanced_fundamentals['valuation']['dividend_yield']
            adv_val_cols[3].markdown(create_apple_metric(
                "Div Yield", 
                f"{div_yield:.2%}" if div_yield else "N/A"
            ), unsafe_allow_html=True)
            
            st.markdown("#### Growth Metrics")
            growth_cols = st.columns(2)
            
            rev_growth = enhanced_fundamentals['growth']['revenue_growth']
            growth_cols[0].markdown(create_apple_metric(
                "Revenue Growth (YoY)", 
                f"{rev_growth:.2f}%" if rev_growth else "N/A"
            ), unsafe_allow_html=True)
            
            eps_growth = enhanced_fundamentals['growth']['eps_growth']
            growth_cols[1].markdown(create_apple_metric(
                "EPS Growth", 
                f"{eps_growth:.2f}%" if eps_growth else "N/A"
            ), unsafe_allow_html=True)
            
            st.markdown("#### Advanced Profitability")
            adv_prof_cols = st.columns(2)
            
            roic = enhanced_fundamentals['profitability']['roic']
            adv_prof_cols[0].markdown(create_apple_metric(
                "ROIC", 
                f"{roic:.2%}" if roic else "N/A"
            ), unsafe_allow_html=True)
            
            payout = enhanced_fundamentals['valuation']['payout_ratio']
            adv_prof_cols[1].markdown(create_apple_metric(
                "Payout Ratio", 
                f"{payout:.2%}" if payout else "N/A"
            ), unsafe_allow_html=True)
                
        st.markdown("---")
        st.markdown("### Financial Statements Analysis")
        
        if financials is not None:
            if financials['income_statement'] is not None and not financials['income_statement'].empty:
                st.markdown("#### Income Statement Trends")
                plot_financial_statement(financials['income_statement'], "Income Statement")
                
                if financials['income_growth'] is not None and not financials['income_growth'].empty:
                    plot_financial_growth(financials['income_growth'], "Income Statement Growth")
                else:
                    st.info("Income statement growth calculation not available (possibly due to data limitations)")
            else:
                st.warning("Income statement data not available")
            
            if financials['balance_sheet'] is not None and not financials['balance_sheet'].empty:
                st.markdown("#### Balance Sheet Trends")
                plot_financial_statement(financials['balance_sheet'], "Balance Sheet")
                
                ratio_available = False
                if 'Current Ratio' in financials['balance_sheet'].index and 'Debt/Equity' in financials['balance_sheet'].index:
                    ratios = financials['balance_sheet'].loc[['Current Ratio', 'Debt/Equity']]
                    if not ratios.empty:
                        st.markdown("#### Key Ratio Trends")
                        fig = px.line(
                            ratios.T,
                            title="Key Financial Ratios",
                            labels={'value': 'Ratio', 'variable': 'Metric'},
                            height=400
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        ratio_available = True
                
                if not ratio_available:
                    st.info("Key ratio calculation not available (possibly missing required balance sheet items)")
            else:
                st.warning("Balance sheet data not available")
            
            if financials['cash_flow'] is not None and not financials['cash_flow'].empty:
                st.markdown("#### Cash Flow Trends")
                plot_financial_statement(financials['cash_flow'], "Cash Flow Statement")
            else:
                st.warning("Cash flow statement data not available")
        else:
            st.warning("Financial statements data not available for this company")

    # Technical Analysis Tab
    with tabs[1]:
        st.markdown("### Technical Analysis")
        
        st.markdown("#### Risk Metrics")
        risk_cols = st.columns(4)
        
        risk_cols[0].markdown(create_apple_metric(
            "Annual Volatility", 
            f"{st.session_state.risk_metrics['annual_volatility']*100:.2f}%" if not np.isnan(st.session_state.risk_metrics['annual_volatility']) else "N/A"
        ), unsafe_allow_html=True)
        
        risk_cols[1].markdown(create_apple_metric(
            "Max Drawdown", 
            f"{st.session_state.risk_metrics['max_drawdown']*100:.2f}%" if not np.isnan(st.session_state.risk_metrics['max_drawdown']) else "N/A"
        ), unsafe_allow_html=True)
        
        risk_cols[2].markdown(create_apple_metric(
            "Sharpe Ratio", 
            f"{st.session_state.risk_metrics['sharpe_ratio']:.2f}" if not np.isnan(st.session_state.risk_metrics['sharpe_ratio']) else "N/A"
        ), unsafe_allow_html=True)
        
        risk_cols[3].markdown(create_apple_metric(
            "Sortino Ratio", 
            f"{st.session_state.risk_metrics['sortino_ratio']:.2f}" if not np.isnan(st.session_state.risk_metrics['sortino_ratio']) else "N/A"
        ), unsafe_allow_html=True)
        
        st.markdown("#### Relative Strength Index (RSI)")
        if not st.session_state.technical['rsi'].empty:
            fig, ax = plt.subplots(figsize=(12, 4))
            ax.plot(st.session_state.technical['rsi'], label='RSI', color='#0071e3', linewidth=2)
            ax.axhline(70, color='#FF3B30', linestyle='--', linewidth=1, label='Overbought (70)')
            ax.axhline(30, color='#34C759', linestyle='--', linewidth=1, label='Oversold (30)')
            ax.set_facecolor('#f5f5f7')
            ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_color('#d2d2d7')
            ax.spines['left'].set_color('#d2d2d7')
            ax.legend()
            st.pyplot(fig)
        else:
            st.warning("No RSI data available")
        
        st.markdown("#### MACD")
        if not st.session_state.technical['macd'].empty:
            fig, ax = plt.subplots(figsize=(12, 4))
            ax.plot(st.session_state.technical['macd'], label='MACD', color='#0071e3', linewidth=2)
            ax.plot(st.session_state.technical['signal_line'], label='Signal Line', color='#FF9500', linewidth=2)
            ax.set_facecolor('#f5f5f7')
            ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_color('#d2d2d7')
            ax.spines['left'].set_color('#d2d2d7')
            ax.legend()
            st.pyplot(fig)
        else:
            st.warning("No MACD data available")

    # Predictions Tab
    with tabs[2]:
        st.markdown("### Price Predictions")
        y, predictions = prepare_data(st.session_state.data, model)
        if y is not None and predictions is not None:
            fig, ax = plt.subplots(figsize=(12, 5))
            ax.plot(y, label='Actual Prices', color='#0071e3', linewidth=2)
            ax.plot(predictions, label='Predicted Prices', color='#34C759', linewidth=2)
            ax.set_facecolor('#f5f5f7')
            ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_color('#d2d2d7')
            ax.spines['left'].set_color('#d2d2d7')
            ax.legend()
            st.pyplot(fig)
            
            pred_cols = st.columns(2)
            pred_cols[0].markdown(create_apple_metric(
                "Mean Absolute Error", 
                f"{mean_absolute_error(y, predictions):.2f}"
            ), unsafe_allow_html=True)
            
            pred_cols[1].markdown(create_apple_metric(
                "Root Mean Squared Error", 
                f"{np.sqrt(mean_squared_error(y, predictions)):.2f}"
            ), unsafe_allow_html=True)
        else:
            st.warning("Not enough data for predictions")

    # Sentiment Tab
    with tabs[3]:
        st.markdown("### Market Sentiment Analysis")
        
        if not st.secrets.get("NEWSAPI_KEY"):
            st.warning("NewsAPI key not configured")
            st.info("To enable sentiment analysis, add your NewsAPI key to secrets.toml")
            st.stop()
        
        try:
            sentiment = st.session_state.sentiment
            
            st.markdown(create_apple_metric(
                "Overall Sentiment Score", 
                f"{sentiment['overall_sentiment']:.2f}",
                "Positive" if sentiment['overall_sentiment'] > 0.2 else "Negative" if sentiment['overall_sentiment'] < -0.2 else "Neutral",
                "positive" if sentiment['overall_sentiment'] > 0.2 else "negative" if sentiment['overall_sentiment'] < -0.2 else ""
            ), unsafe_allow_html=True)
            
            if sentiment['source_details']['news']['top_news']:
                st.markdown("#### Top News Headlines")
                for i, headline in enumerate(sentiment['source_details']['news']['top_news'][:5], 1):
                    st.markdown(f"""
                    <div class="apple-card" style="padding: 16px; margin-bottom: 12px;">
                        <p style="margin: 0; font-weight: 500;">{headline}</p>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("No news headlines found for this stock")
                
        except Exception as e:
            st.error(f"Sentiment analysis failed: {e}")

    # Raw Data Tab
    with tabs[4]:
        st.markdown("### Raw Stock Data")
        
        min_date = st.session_state.data.index.min()
        max_date = st.session_state.data.index.max()
        
        col1, col2 = st.columns(2)
        with col1:
            raw_start = st.date_input("From", min_date)
        with col2:
            raw_end = st.date_input("To", max_date)
        
        filtered_data = st.session_state.data.loc[
            (st.session_state.data.index >= pd.to_datetime(raw_start)) & 
            (st.session_state.data.index <= pd.to_datetime(raw_end))
        ]
        
        st.dataframe(filtered_data.style.format("{:.2f}"), height=400)
        
        csv = filtered_data.to_csv().encode('utf-8')
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name=f'{stock_symbol}_stock_data.csv',
            mime='text/csv',
            use_container_width=True
        )
    
    # Comps Analysis Tab
    with tabs[5]:
        st.markdown("### Comparable Company Analysis")

        with st.spinner("Identifying comparable companies..."):
            peers = get_peers(stock_symbol)
            company_info = yf.Ticker(stock_symbol).info
            industry = company_info.get('industry', 'Unknown')
            sector = company_info.get('sector', 'Unknown')
        
        st.markdown(f"""
        **Sector:** {sector}  
        **Industry:** {industry}
        """)
        
        if not peers:
            st.warning("""
            Could not find direct comparable companies for this stock.  
            Possible reasons:
            - Stock may be too small or obscure
            - Company may be in a niche industry
            - Data limitations from Yahoo Finance
            """)
            
            st.info("""
            Try analyzing against:
            1. Major sector ETFs (e.g., XLK for tech, XLF for financials)
            2. Industry leaders
            3. Custom-selected peers
            """)
            st.stop()
        
        st.markdown(f"**Selected Peer Companies:** {', '.join(peers)}")
        
        with st.spinner("Gathering comparison data..."):
            comps_df = get_comps_data(stock_symbol, peers)
        
        if comps_df.empty:
            st.warning("No comparable data available")
            st.stop()
        
        st.markdown("#### Valuation Multiples Comparison")
        
        def safe_format(val):
            if pd.isna(val):
                return "N/A"
            try:
                if isinstance(val, (int, float)):
                    return f"{val:.2f}"
                return str(val)
            except:
                return str(val)
        
        format_cols = ['pe_ratio', 'ev_ebitda', 'price_to_sales', 'price_to_book']
        styled_df = comps_df.style.format({
            col: safe_format for col in format_cols
        })
        
        st.dataframe(
            styled_df,
            height=min(300, 35 * (len(comps_df) + 1)),
            use_container_width=True
        )
        
        metric = st.selectbox(
            "Select metric to visualize",
            ['pe_ratio', 'ev_ebitda', 'price_to_sales', 'price_to_book'],
            format_func=lambda x: x.replace('_', ' ').title()
        )
        
        fig = plot_comps_chart(comps_df, metric)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning(f"Could not generate chart for {metric}")
        
        st.markdown("#### Summary Statistics")
        summary_cols = st.columns(4)

        numeric_cols = comps_df.select_dtypes(include=[np.number]).columns
        for i, col in enumerate(numeric_cols):
            with summary_cols[i % 4]:
                try:
                    avg = comps_df[col].mean()
                    median = comps_df[col].median()
                    st.metric(
                        label=col.replace('_', ' ').title(),
                        value=f"{median:.2f}",
                        delta=f"Mean: {avg:.2f}",
                        delta_color="off"
                    )
                except Exception as e:
                    st.warning(f"Could not calculate stats for {col}: {str(e)}")

        st.markdown("#### Relative Valuation")
        selected_metric = st.selectbox(
            "Base valuation on",
            ['pe_ratio', 'ev_ebitda'],
            index=0,
            key='rel_val_metric'
        )
        
        if selected_metric in comps_df.columns:
            main_company = comps_df.loc[stock_symbol]
            peer_median = comps_df[selected_metric].median()
            
            # Safely get the main company's value
            main_value = main_company[selected_metric]
            if isinstance(main_value, (int, float)) and not pd.isna(main_value) and not pd.isna(peer_median):
                diff_pct = (main_value - peer_median) / peer_median * 100
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric(
                        label=f"Company {selected_metric.replace('_', ' ').title()}",
                        value=f"{main_value:.2f}"
                    )
                
                with col2:
                    st.metric(
                        label="vs Peer Median",
                        value=f"{peer_median:.2f}",
                        delta=f"{diff_pct:.1f}%",
                        delta_color="inverse"
                    )
                
                if diff_pct > 0:
                    st.info(f"{stock_symbol} is trading at a {abs(diff_pct):.1f}% premium to peers based on {selected_metric.replace('_', ' ')}")
                else:
                    st.info(f"{stock_symbol} is trading at a {abs(diff_pct):.1f}% discount to peers based on {selected_metric.replace('_', ' ')}")
            else:
                st.warning("Not enough data for relative valuation analysis")
        else:
            st.warning("Selected metric not available for comparison")
    # DCF Valuation Tab
    with tabs[6]:
        st.markdown("## Discounted Cash Flow Valuation")
        
        with st.spinner("Loading financial data..."):
            dcf_data = get_dcf_inputs(stock_symbol)
        
        if not dcf_data or dcf_data['free_cash_flow'] is None:
            st.warning("Free cash flow data not available for DCF analysis")
            st.stop()
        
        with st.expander("DCF Parameters", expanded=True):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Growth Rates**")
                initial_growth = st.slider(
                    "Initial Growth Rate (% next 5 years)",
                    min_value=0.0,
                    max_value=50.0,
                    value=10.0,
                    step=0.5
                ) / 100
                
                terminal_growth = st.slider(
                    "Terminal Growth Rate (%)",
                    min_value=0.0,
                    max_value=5.0,
                    value=2.5,
                    step=0.1
                ) / 100
            
            with col2:
                st.markdown("**Risk Parameters**")
                risk_free = st.slider(
                    "Risk-Free Rate (%)",
                    min_value=0.0,
                    max_value=10.0,
                    value=2.5,
                    step=0.1
                ) / 100
                
                market_risk_premium = st.slider(
                    "Market Risk Premium (%)",
                    min_value=0.0,
                    max_value=10.0,
                    value=5.5,
                    step=0.1
                ) / 100
                
                beta = st.number_input(
                    "Beta",
                    min_value=0.0,
                    max_value=3.0,
                    value=dcf_data['beta'] if dcf_data['beta'] else 1.0,
                    step=0.1
                )
                
                discount_rate = risk_free + beta * market_risk_premium
                st.metric("Discount Rate (WACC)", f"{discount_rate:.1%}")
        
        dcf_result = calculate_dcf(
            fcf=dcf_data['free_cash_flow'],
            growth_rate=initial_growth,
            discount_rate=discount_rate,
            terminal_growth=terminal_growth
        )
        
        if not dcf_result:
            st.error("DCF calculation failed")
            st.stop()
        
        st.markdown("### DCF Valuation Results")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(
                "Enterprise Value",
                f"${dcf_result['enterprise_value']/1e9:.2f}B"
            )
        
        with col2:
            st.metric(
                "Terminal Value",
                f"${dcf_result['terminal_value']/1e9:.2f}B"
            )
        
        with col3:
            st.metric(
                "Present Value of FCF",
                f"${sum(dcf_result['cash_flows'])/1e9:.2f}B"
            )
        
        if dcf_data['cash'] and dcf_data['debt'] and dcf_data['shares']:
            equity_value = dcf_result['enterprise_value'] + dcf_data['cash'] - dcf_data['debt']
            price_per_share = equity_value / dcf_data['shares']
            
            st.markdown("### Equity Valuation")
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric(
                    "Calculated Price per Share",
                    f"${price_per_share:.2f}"
                )
            
            with col2:
                if dcf_data['current_price']:
                    premium = (price_per_share - dcf_data['current_price']) / dcf_data['current_price'] * 100
                    st.metric(
                        "Current Price",
                        f"${dcf_data['current_price']:.2f}",
                        delta=f"{premium:.1f}% {'premium' if premium > 0 else 'discount'}",
                        delta_color="inverse"
                    )
        
        st.markdown("### Cash Flow Projections")
        years = list(range(1, dcf_result['years'] + 1))
        projected_cash_flows = dcf_result['cash_flows']
        
        fig = px.line(
            x=years,
            y=projected_cash_flows,
            labels={'x': 'Year', 'y': 'Free Cash Flow'},
            title='Projected Free Cash Flows'
        )
        fig.update_traces(mode='lines+markers')
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("### Sensitivity Analysis")
        st.info("Vary the inputs in the parameters section to see how the valuation changes")

if __name__ == "__main__":
    main()
