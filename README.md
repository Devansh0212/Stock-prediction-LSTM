# 📈 Stock Market Predictor

This project is a **Stock Market Prediction and Visualization Tool** built with Streamlit. It allows users to analyze stock data, visualize trends, and predict future stock prices using a pre-trained LSTM model.

---

## 🛠 Features

1. **Stock Data Fetching**:  
   Fetch historical stock market data using [Yahoo Finance](https://finance.yahoo.com/) by providing a stock symbol and date range.

2. **Visualization Tools**:  
   - Interactive plots for stock prices and moving averages (EMA50, EMA100, EMA200).  
   - Technical Indicators:  
     - **RSI** (Relative Strength Index): Highlights overbought and oversold conditions.  
     - **MACD** (Moving Average Convergence Divergence): Shows momentum trends.  

3. **Stock Price Prediction**:  
   - Predicts future stock prices based on past trends using an **LSTM (Long Short-Term Memory)** model.  
   - Displays actual vs. predicted prices and performance metrics like **Mean Absolute Error (MAE)** and **Root Mean Squared Error (RMSE)**.

4. **Export Data**:  
   Download raw stock data as a CSV file.

---

## 🚀 Technologies Used

- **Python**: Core programming language.
- **Streamlit**: Interactive web interface.
- **Keras**: Pre-trained LSTM model for predictions.
- **Yahoo Finance API**: Data source for stock prices.
- **Matplotlib**: For creating detailed visualizations.
- **Scikit-learn**: Data scaling and evaluation metrics.

---

## 📂 Project Structure
 ```bash
📁 Stock_Market_Predictor
├──📄 app.py                        # Main Streamlit application to run the web app
├──📄 requirements.txt              # Python dependencies for the project
├──📄 README.md                     # Project documentation (this file)
└──📄 Stock_Predictions_Model.keras  # Pre-trained LSTM model file for stock predictions
 ```
---

## 🔧 Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your_username/Stock_Market_Predictor.git
   cd Stock_Market_Predictor

2. **Install Dependencies:**

Ensure you have Python 3.9+ installed. You can verify your Python version by running:

```bash
python --version
```

Then, install the required dependencies for the project by running the following command in your terminal:
```bash
pip install -r requirements.txt
```
This will install all the necessary libraries, including Streamlit, TensorFlow, and other dependencies needed to run the application.


3. **Add the Pre-trained Model:**

To enable the stock price prediction feature, you need to add the pre-trained LSTM model.
	1.	Download or use your own pre-trained LSTM model file. The file should be in the .keras format, for example, Stock_Predictions_Model.keras.
	2.	Place the Stock_Predictions_Model.keras file in the root directory of the project (the same directory as app.py).

The directory structure should look like this:
 ```bash
📁 Stock_Market_Predictor
├── 📄 app.py
├── 📄 requirements.txt
├── 📄 README.md
└── 📄 Stock_Predictions_Model.keras   # Pre-trained LSTM model
 ```

## ⚙️ Usage

 1. **Run the App:**

Once you have installed all dependencies and added the pre-trained model, you can run the application using Streamlit.

In your terminal, navigate to the project directory and run the following command:

```bash
streamlit run app.py
```
This will start the Streamlit app and open it in your default web browser.

2. **Input Details**

On the app’s sidebar, you’ll see the following input fields:
	•	Enter Stock Symbol: Type the stock symbol (e.g., GOOG for Google, AAPL for Apple).
	•	Select Start Date: Pick the start date for the stock data (default is 2012-01-01).
	•	Select End Date: Choose the end date for the stock data (default is 2022-12-31).

Once you’ve entered the details, click the Fetch Stock Data button to retrieve the stock data from Yahoo Finance.

3. **Explore**

After fetching the stock data, you can explore the following features:
	•	Stock Prices and Moving Averages: View historical stock prices along with selected moving averages (EMA50, EMA100, EMA200).
	•	Predictions: Compare the actual stock prices with the predicted values using the pre-trained LSTM model. You’ll also see performance metrics like MAE (Mean Absolute Error) and RMSE (Root Mean Squared Error).
	•	Technical Indicators: Analyze the stock using technical indicators like the RSI (Relative Strength Index) and MACD (Moving Average Convergence Divergence).
	•	Raw Data: View and download the raw stock data in CSV format for further analysis.

### Example Workflow:

1. Enter a stock symbol, such as `GOOG` (Google).
2. Set the desired date range for fetching the stock data.
3. Click on the **Fetch Stock Data** button.
4. Explore the various tabs like **Stock Prices**, **Predictions**, **Technical Indicators**, and **Raw Data**.
5. Download the stock data as CSV if needed.

Enjoy analyzing and predicting stock data with ease!


## 📊 Output Screens

1. **Stock Prices and Moving Averages**:
   - Displays historical stock prices with EMA50, EMA100, and EMA200 overlays.
   
2. **Predictions**:
   - Compares actual prices with LSTM model predictions.
   
3. **Technical Indicators**:
   - RSI chart with overbought/oversold levels.
   - MACD with signal line.
   
4. **Raw Data**:
   - Tabular display of fetched stock data.

---

## 🛡 Limitations

- Predictions are based on historical data and may not reflect actual market movements.
- The LSTM model is pre-trained and might need fine-tuning for improved accuracy on new datasets.

---

## 📜 License

This project is licensed under the MIT License.

---

## 🌟 Acknowledgments

- Yahoo Finance API for data.
- Keras and TensorFlow for deep learning models.
- Streamlit for making interactive web apps easier.

---

## 🧑‍💻 Author

**Devansh Brahmbhatt**  
Feel free to reach out with suggestions or questions!  
[LinkedIn](www.linkedin.com/in/devansh-b-36251a25b)
