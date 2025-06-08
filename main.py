import os
import numpy as np
import yfinance as yf
import pandas as pd
from flask import Flask, request, abort
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import MessageEvent, TextMessage, TextSendMessage
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from scipy.optimize import minimize

app = Flask(__name__)

line_bot_api = LineBotApi(os.environ['LINE_CHANNEL_ACCESS_TOKEN'])
handler = WebhookHandler(os.environ['LINE_CHANNEL_SECRET'])

# === 載入模型 ===
return_model = load_model("return_model.h5")
features = ['Close', 'MA_5', 'MA_10', 'RSI', 'MACD_diff', 'Volatility']

thresholds = {
    'AAPL': 0.6,
    'GOOGL': 0.48,
    'META': 0.65,
    'NVDA': 0.63
}

def predict_tomorrow_direction(model, symbol, features):
    df = yf.download(symbol, period="6mo")[['Close']].dropna()
    df['MA_5'] = df['Close'].rolling(window=5).mean()
    df['MA_10'] = df['Close'].rolling(window=10).mean()
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))
    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26
    df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_diff'] = df['MACD'] - df['MACD_signal']
    df['Volatility'] = df['Close'].pct_change().rolling(window=10).std()
    df = df.dropna()

    if len(df) < 20:
        return "資料不足無法預測"

    scaler = MinMaxScaler()
    df_scaled = scaler.fit_transform(df[features])
    X_input = df_scaled[-10:].reshape(1, 10, len(features))

    pred = model.predict(X_input)[0][0]
    threshold = thresholds.get(symbol, 0.5)
    result = "漲📈" if pred > threshold else "跌📉"
    confidence = pred * 100 if pred > threshold else (1 - pred) * 100
    return f"預測 {symbol} 明天會 {result}\n"

def run_optimized_portfolio(user_input: str):
    parts = user_input.strip().upper().split()
    if not parts[-1].replace('.', '', 1).isdigit():
        return "請正確輸入格式，例如：最佳化 AAPL META 10000"

    tickers = parts[:-1]
    investment_amount = float(parts[-1])
    features = ['Close', 'MA_5', 'MA_10', 'RSI', 'MACD_diff', 'Volatility']
    window = 10

    latest_preds = {}
    for symbol in tickers:
        df = yf.download(symbol, period="6mo")[['Close']].dropna()
        df['MA_5'] = df['Close'].rolling(window=5).mean()
        df['MA_10'] = df['Close'].rolling(window=10).mean()
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss
        df['RSI'] = 100 - (100 / (1 + rs))
        ema12 = df['Close'].ewm(span=12, adjust=False).mean()
        ema26 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = ema12 - ema26
        df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_diff'] = df['MACD'] - df['MACD_signal']
        df['Volatility'] = df['Close'].pct_change().rolling(window=10).std()
        df = df.dropna()

        if len(df) < window:
            continue

        scaler = MinMaxScaler()
        df_scaled = scaler.fit_transform(df[features])
        X_input = df_scaled[-window:].reshape(1, window, len(features))
        pred = return_model.predict(X_input)[0][0]
        latest_preds[symbol] = pred

    if not latest_preds:
        return "❌ 找不到可分析的股票資料，請確認股票代碼是否正確或資料是否足夠"

    # 獲取歷史報酬與協方差
    hist_close = {}
    for sym in latest_preds:
        df_hist = yf.download(sym, period='6mo')[['Close']].dropna()
        hist_close[sym] = df_hist.squeeze()

    hist_df = pd.DataFrame(hist_close).dropna()
    log_returns = np.log(hist_df / hist_df.shift(1)).dropna()
    cov_matrix = log_returns.cov().values
    expected_returns = np.array([latest_preds[sym] for sym in hist_df.columns])

    def portfolio_variance(weights, cov_matrix):
        return np.dot(weights.T, np.dot(cov_matrix, weights))

    def neg_sharpe_ratio(weights, returns, cov_matrix):
        port_return = np.dot(weights, returns)
        port_std = np.sqrt(portfolio_variance(weights, cov_matrix))
        return -port_return / port_std

    num_assets = len(hist_df.columns)
    initial_weights = np.ones(num_assets) / num_assets
    bounds = tuple((0.05, 0.5) for _ in range(num_assets))
    constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]

    opt_result = minimize(neg_sharpe_ratio, initial_weights,
                          args=(expected_returns, cov_matrix),
                          method='SLSQP', bounds=bounds,
                          constraints=constraints)

    optimal_weights = opt_result.x
    allocation = (optimal_weights * investment_amount).round(2)
    daily_contribution = optimal_weights * expected_returns
    annualized_return = np.sum(daily_contribution) * 252

    lines = ["📊 最佳投資組合建議："]
    for i, sym in enumerate(hist_df.columns):
        lines.append(f"{sym}: 比重 {optimal_weights[i]*100:.1f}%、金額 ${allocation[i]:.0f}")
    return "\n".join(lines)

@app.route("/callback", methods=['POST'])
def callback():
    signature = request.headers['X-Line-Signature']
    body = request.get_data(as_text=True)
    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        abort(400)
    return 'OK'

@handler.add(MessageEvent, message=TextMessage)
def handle_message(event):
    text = event.message.text.strip()
    upper_text = text.upper()

    if upper_text.startswith("最佳化"):
        try:
            response = run_optimized_portfolio(text.replace("最佳化", "", 1).strip())
        except Exception as e:
            response = f"❌ 執行錯誤：{str(e)}"
    elif upper_text in thresholds:
        response = predict_tomorrow_direction(return_model, upper_text, features)
    else:
        response = "請輸入：\n- 股票代碼如 AAPL 查詢漲跌\n- 或輸入：最佳化 AAPL META 10000 進行資產配置建議"

    line_bot_api.reply_message(event.reply_token, TextSendMessage(text=response))

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
