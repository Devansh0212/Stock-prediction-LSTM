# Stock Analysis Pro - Apple Style UI
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
    /* Main styling */
    .stApp {
        background-color: #f5f5f7;
        color: #1d1d1f;
        font-family: -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #f5f5f7 !important;
        border-right: 1px solid #d2d2d7;
    }
    
    /* Headers */
    h1, h2, h3, h4, h5, h6 {
        color: #1d1d1f !important;
        font-weight: 600 !important;
    }
    
    /* Cards and containers */
    .stMetric {
        background-color: white;
        border-radius: 12px;
        padding: 15px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    
    /* Buttons */
    .stButton>button {
        background-color: #0071e3;
        color: white;
        border-radius: 980px;
        padding: 8px 16px;
        font-size: 14px;
        font-weight: 400;
        border: none;
    }
    
    .stButton>button:hover {
        background-color: #0077ed;
    }
    
    /* Tabs */
    [data-baseweb="tab-list"] {
        gap: 10px;
    }
    
    [data-baseweb="tab"] {
        background-color: white;
        border-radius: 8px 8px 0 0 !important;
        padding: 8px 16px;
        margin: 0 !important;
    }
    
    [aria-selected="true"] {
        background-color: #0071e3 !important;
        color: white !important;
    }
    
    /* Inputs */
    .stTextInput>div>div>input {
        border-radius: 12px !important;
        padding: 10px 12px !important;
    }
    
    /* Dataframes */
    .stDataFrame {
        border-radius: 12px;
        overflow: hidden;
    }
    
    /* Custom classes */
    .apple-header {
        font-size: 32px !important;
        font-weight: 700 !important;
        margin-bottom: 8px !important;
    }
    
    .apple-subheader {
        font-size: 24px !important;
        font-weight: 600 !important;
        margin-bottom: 16px !important;
        color: #1d1d1f !important;
    }
    
    .apple-card {
        background-color: white;
        border-radius: 18px;
        padding: 20px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        margin-bottom: 20px;
    }
    
    .apple-metric {
        font-size: 42px;
        font-weight: 600;
        color: #1d1d1f;
    }
    
    .apple-metric-label {
        font-size: 14px;
        color: #86868b;
        margin-bottom: 4px;
    }
    
    .apple-divider {
        height: 1px;
        background-color: #d2d2d7;
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
DEFAULT_STOCK = 'AAPL'  # Changed to Apple for demo purposes
DEFAULT_START_DATE = '2012-01-01'
DEFAULT_END_DATE = '2022-12-31'

# Utility Functions (same as before, but with improved visual outputs)
@st.cache_resource
def load_lstm_model(path):
    try:
        return load_model(path)
    except Exception as e:
        st.error(f"Model loading failed: {e}")
        return None

@st.cache_data(ttl=3600)
def fetch_stock_data(symbol, start_date, end_date, max_retries=3):
    """Robust stock data fetcher with error handling"""
    try:
        # Validate dates
        if start_date > end_date:
            st.error("Error: Start date must be before end date")
            return pd.DataFrame()
        
        # Try different symbol variations
        symbol_variations = [symbol, f"{symbol}.NS", f"{symbol}.BO", f"{symbol}.AX", f"{symbol}.TO"]
        
        for sym in symbol_variations:
            try:
                df = yf.download(
                    sym,
                    start=start_date,
                    end=end_date + timedelta(days=1),  # Include end date
                    progress=False,
                    auto_adjust=True,
                    threads=True,
                    timeout=10
                )
                if not df.empty:
                    return df
            except Exception as e:
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
        
        # Alternative data sources if primary fails
        if not info:
            stock = yf.Ticker(ticker + ".NS")  # Try NSE
            info = stock.info
        
        # Default values dictionary
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
        
        # Convert numeric values from strings if needed
        for category in fundamentals:
            for metric, value in fundamentals[category].items():
                if isinstance(value, str) and value.replace('.','',1).isdigit():
                    fundamentals[category][metric] = float(value)
                    
        return fundamentals
        
    except Exception as e:
        st.error(f"Fundamental data fetch error: {e}")
        return None

def generate_recommendation(fundamentals, technicals, sentiment):
    """Generate investment recommendation based on multiple factors"""
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
        
        # Valuation scoring
        pe = fundamentals['valuation']['pe_ratio']
        if pe != 'N/A':
            if isinstance(pe, (int, float)):
                if pe < 15:
                    scores['valuation'] = 1
                elif pe < 25:
                    scores['valuation'] = 0.5
                else:
                    scores['valuation'] = -0.5
        
        # Profitability scoring
        roe = fundamentals['profitability']['roe']
        if roe != 'N/A':
            if isinstance(roe, (int, float)):
                if roe > 0.15:
                    scores['profitability'] = 1
                elif roe > 0:
                    scores['profitability'] = 0.5
                else:
                    scores['profitability'] = -1
        
        # Technical scoring
        rsi = technicals.get('rsi', float('nan'))
        if not np.isnan(rsi):
            if rsi < 30:
                scores['technical'] = 1
            elif rsi > 70:
                scores['technical'] = -1
        
        # Sentiment scoring
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

def calculate_indicators(data):
    """Calculate technical indicators with proper error handling"""
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
        
        # Proper float conversion
        last_rsi = float(rsi.iloc[-1].item()) if not rsi.empty else float('nan')
        
        return rsi, macd, signal_line, last_rsi
        
    except Exception as e:
        st.error(f"Technical indicator calculation failed: {e}")
        return pd.Series(), pd.Series(), pd.Series(), float('nan')

def calculate_risk_metrics(data):
    """Calculate risk metrics with proper error handling"""
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
            
        # Proper float conversion without warnings
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
    """Prepare data for LSTM prediction with error handling"""
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
        
        # Inverse transform the predictions
        scale_factor = 1 / scaler.scale_[0]
        predictions = predictions * scale_factor
        y = y * scale_factor
        
        return y, predictions
        
    except Exception as e:
        st.error(f"Data preparation failed: {e}")
        return None, None

def create_apple_metric(label, value, delta=None, delta_type=None):
    """Create an Apple-style metric component"""
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
    """Fetch company name and other basic info"""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        # Try alternative exchanges if primary fails
        if not info:
            stock = yf.Ticker(ticker + ".NS")  # Try NSE
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

    # Sidebar - Apple Style
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
        # Hero section when no analysis has been run yet
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

    # Fetch data when analyze is clicked
    with st.spinner('Loading data...'):
        # Initialize session state with default values
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
        
        # Fetch data
        data = fetch_stock_data(stock_symbol, start_date, end_date)
        if data.empty:
            st.error("No data available for the selected stock and date range")
            st.stop()
        
        st.session_state.data = data
        st.session_state.fundamentals = get_fundamental_data(stock_symbol)
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

    # Main Content
    def main():
    # [Previous code: model loading, data fetching, etc...]
    
        if 'data' not in st.session_state or st.session_state.data.empty:
            st.info("Enter a stock symbol and click 'Analyze' to begin")
            st.stop()

    # ====== REPLACE THIS SECTION ======
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
        <!-- Rest of your new header HTML -->
    </div>
    """, unsafe_allow_html=True)
    
    with st.expander("Company Overview", expanded=False):
        # [Company description content]
    # ====== END REPLACEMENT SECTION ======

   

    # Recommendation and Key Metrics
        if st.session_state.fundamentals:
            rec, score, rec_type = generate_recommendation(
                st.session_state.fundamentals,
                {'rsi': st.session_state.technical['last_rsi']},
            st.session_state.sentiment['overall_sentiment']
        )
    
    # Create a grid of metrics
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
        except Exception as e:
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

    # Price Chart - Apple Style
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

    # Tabs - Apple Style
    tabs = st.tabs([
        "📊 **Fundamentals**", 
        "📉 **Technical Analysis**",
        "🔮 **Predictions**",
        "😀 **Sentiment**",
        "📝 **Raw Data**"
    ])

    # Fundamentals Tab
    with tabs[0]:
        st.markdown("### Fundamental Analysis")
        
        if not st.session_state.fundamentals:
            st.warning("Fundamental data not available for this stock")
            st.stop()
        
        # Valuation Metrics
        st.markdown("#### Valuation")
        val_cols = st.columns(4)
        
        # Market Cap
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
        
        # P/E Ratio
        pe = st.session_state.fundamentals['valuation'].get('pe_ratio', 'N/A')
        pe_display = str(pe) if pe != 'N/A' else "N/A"
        val_cols[1].markdown(create_apple_metric("P/E Ratio", pe_display), unsafe_allow_html=True)
        
        # Forward P/E
        fpe = st.session_state.fundamentals['valuation'].get('forward_pe', 'N/A')
        fpe_display = str(fpe) if fpe != 'N/A' else "N/A"
        val_cols[2].markdown(create_apple_metric("Forward P/E", fpe_display), unsafe_allow_html=True)
        
        # P/B Ratio
        pb = st.session_state.fundamentals['valuation'].get('price_to_book', 'N/A')
        pb_display = str(pb) if pb != 'N/A' else "N/A"
        val_cols[3].markdown(create_apple_metric("P/B Ratio", pb_display), unsafe_allow_html=True)
        
        # Profitability Metrics
        st.markdown("#### Profitability")
        prof_cols = st.columns(3)
        
        # ROE
        roe = st.session_state.fundamentals['profitability'].get('roe', 'N/A')
        roe_display = f"{roe*100:.2f}%" if isinstance(roe, (int, float)) else "N/A"
        prof_cols[0].markdown(create_apple_metric("Return on Equity", roe_display), unsafe_allow_html=True)
        
        # Profit Margins
        margins = st.session_state.fundamentals['profitability'].get('profit_margins', 'N/A')
        margin_display = f"{margins*100:.2f}%" if isinstance(margins, (int, float)) else "N/A"
        prof_cols[1].markdown(create_apple_metric("Profit Margins", margin_display), unsafe_allow_html=True)
        
        # Operating Margins
        op_margins = st.session_state.fundamentals['profitability'].get('operating_margins', 'N/A')
        op_margin_display = f"{op_margins*100:.2f}%" if isinstance(op_margins, (int, float)) else "N/A"
        prof_cols[2].markdown(create_apple_metric("Operating Margins", op_margin_display), unsafe_allow_html=True)
        
        # Financial Health Metrics
        st.markdown("#### Financial Health")
        health_cols = st.columns(3)
        
        # Debt/Equity
        de = st.session_state.fundamentals['financial_health'].get('debt_to_equity', 'N/A')
        de_display = str(de) if de != 'N/A' else "N/A"
        health_cols[0].markdown(create_apple_metric("Debt/Equity", de_display), unsafe_allow_html=True)
        
        # Current Ratio
        cr = st.session_state.fundamentals['financial_health'].get('current_ratio', 'N/A')
        cr_display = str(cr) if cr != 'N/A' else "N/A"
        health_cols[1].markdown(create_apple_metric("Current Ratio", cr_display), unsafe_allow_html=True)
        
        # Quick Ratio
        qr = st.session_state.fundamentals['financial_health'].get('quick_ratio', 'N/A')
        qr_display = str(qr) if qr != 'N/A' else "N/A"
        health_cols[2].markdown(create_apple_metric("Quick Ratio", qr_display), unsafe_allow_html=True)

    # Technical Analysis Tab
    with tabs[1]:
        st.markdown("### Technical Analysis")
        
        # Risk metrics
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
        
        # RSI Chart
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
        
        # MACD Chart
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
        
        # Add date range filter for raw data
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

if __name__ == "__main__":
    main()