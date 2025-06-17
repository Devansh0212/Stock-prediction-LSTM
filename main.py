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

# Configuration
st.set_page_config(layout="wide", page_title="Stock Analysis Pro", page_icon="📈")

# Constants
MODEL_PATH = 'Stock Predictions Model.keras'
DEFAULT_STOCK = 'GOOGL'  # Changed from GOOG to GOOGL
DEFAULT_START_DATE = '2012-01-01'
DEFAULT_END_DATE = '2022-12-31'

# Utility Functions
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
            return "Strong Buy", total_score
        elif total_score >= 1:
            return "Buy", total_score
        elif total_score >= -1:
            return "Hold", total_score
        elif total_score >= -3:
            return "Sell", total_score
        else:
            return "Strong Sell", total_score
    
    except Exception as e:
        st.error(f"Recommendation generation failed: {e}")
        return "Hold", 0.0


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
def main():
    model = load_lstm_model(MODEL_PATH)
    if model is None:
        st.stop()

    # Sidebar
    st.sidebar.header('Stock Selection')
    stock_symbol = st.sidebar.text_input('Enter Stock Symbol', DEFAULT_STOCK).upper()
    start_date = st.sidebar.date_input('Start Date', pd.to_datetime(DEFAULT_START_DATE))
    end_date = st.sidebar.date_input('End Date', pd.to_datetime(DEFAULT_END_DATE))
    
    if st.sidebar.button('Analyze'):
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

    # Tabs
    tabs = st.tabs([
        "📈 Overview",
        "📊 Fundamentals", 
        "📉 Technical",
        "🧠 Predictions",
        "😀 Sentiment",
        "📝 Raw Data"
    ])

    # Overview Tab
    with tabs[0]:
        st.header(f"{stock_symbol} Comprehensive Analysis")
        
        # Recommendation
        rec, score = "Hold", 0.0
        if st.session_state.fundamentals:
            rec, score = generate_recommendation(
                st.session_state.fundamentals,
                {'rsi': st.session_state.technical['last_rsi']},
                st.session_state.sentiment['overall_sentiment']
            )
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Recommendation", rec, f"Score: {score:.1f}/7.0")        
        with col2:
            try:
                price_series = st.session_state.data['Close'].iloc[-1]
                current_price = float(price_series.item()) if isinstance(price_series, pd.Series) else float(price_series)
                st.metric("Current Price", f"${current_price:.2f}")
            except Exception as e:
                st.metric("Current Price", "N/A")
        with col3:
            st.metric("Sentiment", 
                     f"{st.session_state.sentiment['overall_sentiment']:.2f}",
                     "Positive" if st.session_state.sentiment['overall_sentiment'] > 0.2 
                     else "Negative" if st.session_state.sentiment['overall_sentiment'] < -0.2 
                     else "Neutral")
        
        # Mini charts
        fig1, ax1 = plt.subplots(figsize=(12, 4))
        ax1.plot(st.session_state.data['Close'], label='Price')
        ax1.set_title('Price Trend')
        st.pyplot(fig1)
        
        # Key metrics
        st.subheader("Key Metrics")
        cols = st.columns(7)
        # P/E Ratio with better handling
        pe_ratio = st.session_state.fundamentals['valuation'].get('pe_ratio', 'N/A')
        pe_display = str(pe_ratio) if pe_ratio != 'N/A' else "Data Unavailable"
        cols[0].metric("P/E Ratio", pe_display)


        # ROE with better handling
        roe = st.session_state.fundamentals['profitability'].get('roe', 'N/A')
        roe_display = f"{roe*100:.2f}%" if isinstance(roe, (int, float)) else "Data Unavailable"
        cols[1].metric("ROE", roe_display)

        # Volatility with proper checks
        volatility = st.session_state.risk_metrics.get('annual_volatility', float('nan'))
        vol_display = f"{volatility*100:.2f}%" if not np.isnan(volatility) else "Calculating..."
        cols[2].metric("Volatility", vol_display)

        # RSI with proper checks
        rsi_value = st.session_state.technical.get('last_rsi', float('nan'))
        rsi_display = f"{rsi_value:.2f}" if not np.isnan(rsi_value) else "Calculating..."
        cols[3].metric("RSI", rsi_display)
        cols[4].metric("P/E", str(st.session_state.fundamentals['valuation']['pe_ratio']) if st.session_state.fundamentals else "N/A")
        cols[5].metric("ROE", f"{float(st.session_state.fundamentals['profitability']['roe'])*100:.2f}%" if st.session_state.fundamentals and st.session_state.fundamentals['profitability']['roe'] != 'N/A' else "N/A")
        volatility = st.session_state.risk_metrics.get('annual_volatility', float('nan'))
        if isinstance(volatility, (pd.Series, pd.DataFrame)):
            volatility = float(volatility.iloc[0]) if not volatility.empty else float('nan')
        vol_display = f"{volatility*100:.2f}%" if not np.isnan(volatility) else "N/A"
        
        volatility = st.session_state.risk_metrics.get('annual_volatility', float('nan'))
        vol_display = "N/A"  # Initialize with default value

    try:
        if isinstance(volatility, (pd.Series, pd.DataFrame)):
            volatility = float(volatility.iloc[0]) if not volatility.empty else float('nan')
        vol_display = f"{volatility*100:.2f}%" if not np.isnan(volatility) else "N/A"
    except Exception as e:
        st.error(f"Volatility calculation error: {e}")
        vol_display = "N/A"
        volatility = st.session_state.risk_metrics.get('annual_volatility', float('nan'))
    try:
        if isinstance(volatility, (pd.Series, pd.DataFrame)):
            volatility = float(volatility.iloc[0].item())
        vol_display = f"{volatility*100:.2f}%" if not np.isnan(volatility) else "N/A"
    except:
        vol_display = "N/A"
        cols[6].metric("Volatility", vol_display)
        cols[7].metric("RSI", f"{st.session_state.technical['last_rsi']:.2f}" if not np.isnan(st.session_state.technical['last_rsi']) else "N/A")
    
    # Fundamentals Tab
    
   
    with tabs[1]:  # Fundamentals Tab
        st.header("Fundamental Analysis")
        
        if not st.session_state.fundamentals:
            st.warning("Fundamental data not available for this stock")
            st.stop()
        
        with st.expander("💰 Valuation Metrics"):
            cols = st.columns(3)
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
            cols[0].metric("Market Cap", display)
            
            # P/E Ratio
            pe = st.session_state.fundamentals['valuation'].get('pe_ratio', 'N/A')
            cols[1].metric("P/E Ratio", str(pe) if pe != 'N/A' else "N/A")
            
            # Price/Book
            pb = st.session_state.fundamentals['valuation'].get('price_to_book', 'N/A')
            cols[2].metric("P/B Ratio", str(pb) if pb != 'N/A' else "N/A")
        
        with st.expander("📈 Profitability"):
            cols = st.columns(2)
            # ROE
            roe = st.session_state.fundamentals['profitability'].get('roe', 'N/A')
            roe_display = f"{roe*100:.2f}%" if isinstance(roe, (int, float)) else "N/A"
            cols[0].metric("Return on Equity", roe_display)
            
            # Profit Margins
            margins = st.session_state.fundamentals['profitability'].get('profit_margins', 'N/A')
            margin_display = f"{margins*100:.2f}%" if isinstance(margins, (int, float)) else "N/A"
            cols[1].metric("Profit Margins", margin_display)
        
        with st.expander("🏦 Financial Health"):
            cols = st.columns(3)
            # Debt/Equity
            de = st.session_state.fundamentals['financial_health'].get('debt_to_equity', 'N/A')
            cols[0].metric("Debt/Equity", str(de) if de != 'N/A' else "N/A")
            
            # Current Ratio
            cr = st.session_state.fundamentals['financial_health'].get('current_ratio', 'N/A')
            cols[1].metric("Current Ratio", str(cr) if cr != 'N/A' else "N/A")
            
            # Quick Ratio
            qr = st.session_state.fundamentals['financial_health'].get('quick_ratio', 'N/A')
            cols[2].metric("Quick Ratio", str(qr) if qr != 'N/A' else "N/A")
    # Technical Tab
    with tabs[2]:
        st.header("Technical Analysis")
        
        # Risk metrics
        st.subheader("Risk Metrics")
        cols = st.columns(5)
        cols[0].metric("Annual Volatility", f"{st.session_state.risk_metrics['annual_volatility']*100:.2f}%" if not np.isnan(st.session_state.risk_metrics['annual_volatility']) else "N/A")
        cols[1].metric("Max Drawdown", f"{st.session_state.risk_metrics['max_drawdown']*100:.2f}%" if not np.isnan(st.session_state.risk_metrics['max_drawdown']) else "N/A")
        cols[2].metric("Sharpe Ratio", f"{st.session_state.risk_metrics['sharpe_ratio']:.2f}" if not np.isnan(st.session_state.risk_metrics['sharpe_ratio']) else "N/A")
        cols[3].metric("Sortino Ratio", f"{st.session_state.risk_metrics['sortino_ratio']:.2f}" if not np.isnan(st.session_state.risk_metrics['sortino_ratio']) else "N/A")
        
        # RSI
        st.subheader("Relative Strength Index (RSI)")
        if not st.session_state.technical['rsi'].empty:
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(st.session_state.technical['rsi'], label='RSI', color='purple')
            ax.axhline(70, color='red', linestyle='--', label='Overbought (70)')
            ax.axhline(30, color='green', linestyle='--', label='Oversold (30)')
            ax.legend()
            st.pyplot(fig)
        else:
            st.warning("No RSI data available")
        
        # MACD
        st.subheader("MACD")
        if not st.session_state.technical['macd'].empty:
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(st.session_state.technical['macd'], label='MACD', color='blue')
            ax.plot(st.session_state.technical['signal_line'], label='Signal Line', color='orange')
            ax.legend()
            st.pyplot(fig)
        else:
            st.warning("No MACD data available")

    # Predictions Tab
    with tabs[3]:
        st.header("Price Predictions")
        y, predictions = prepare_data(st.session_state.data, model)
        if y is not None and predictions is not None:
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(y, label='Actual Prices', color='green')
            ax.plot(predictions, label='Predicted Prices', color='red')
            ax.set_title(f'{stock_symbol} Predicted vs Actual Prices')
            ax.legend()
            st.pyplot(fig)
            
            st.metric("MAE", f"{mean_absolute_error(y, predictions):.2f}")
            st.metric("RMSE", f"{np.sqrt(mean_squared_error(y, predictions)):.2f}")
        else:
            st.warning("Not enough data for predictions")

    # Sentiment Tab
    with tabs[4]:  # Sentiment Tab
        st.header("Market Sentiment Analysis")
        
        if not st.secrets.get("NEWSAPI_KEY"):
            st.warning("NewsAPI key not configured")
            st.info("To enable sentiment analysis, add your NewsAPI key to secrets.toml")
            st.stop()
        
        try:
            sentiment = enhanced_sentiment_analysis(stock_symbol)
            st.session_state.sentiment = sentiment
            
            st.metric("Overall Sentiment Score", 
                    f"{sentiment['overall_sentiment']:.2f}",
                    "Positive" if sentiment['overall_sentiment'] > 0.2 
                    else "Negative" if sentiment['overall_sentiment'] < -0.2 
                    else "Neutral")
            
            if sentiment['source_details']['news']['top_news']:
                st.subheader("Top News Headlines")
                for i, headline in enumerate(sentiment['source_details']['news']['top_news'][:5], 1):
                    st.write(f"{i}. {headline}")
            else:
                st.info("No news headlines found for this stock")
                
        except Exception as e:
            st.error(f"Sentiment analysis failed: {e}")

    # Raw Data Tab
    with tabs[5]:
        st.header("Raw Stock Data")
        st.dataframe(st.session_state.data)
        
        csv = st.session_state.data.to_csv().encode('utf-8')
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name=f'{stock_symbol}_stock_data.csv',
            mime='text/csv'
        )

if __name__ == "__main__":
    main()