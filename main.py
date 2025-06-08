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

app = Flask(__name__)

# === LINE API 金鑰設定 ===
line_bot_api = LineBotApi(os.environ['LINE_CHANNEL_ACCESS_TOKEN'])
handler = WebhookHandler(os.environ['LINE_CHANNEL_SECRET'])

# === 載入模型與特徵欄位 ===
model = load_model("dirction_model.h5")
features = ['Close', 'MA_5', 'MA_10', 'RSI', 'MACD_diff', 'Volatility']

# 股票門檻設定（根據你訓練的結果調整）
thresholds = {
    'AAPL': 0.6,
    'GOOGL': 0.48,
    'META': 0.65,
    'NVDA': 0.63
}

# === 漲跌預測函數 ===
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
    return f"預測 {symbol} 明天會 {result}\n信心度：約 {confidence:.2f}%"

# === Webhook ===
@app.route("/callback", methods=['POST'])
def callback():
    signature = request.headers['X-Line-Signature']
    body = request.get_data(as_text=True)
    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        abort(400)
    return 'OK'

# === 處理訊息 ===
@handler.add(MessageEvent, message=TextMessage)
def handle_message(event):
    text = event.message.text.strip().upper()
    if text in thresholds:
        reply = predict_tomorrow_direction(model, text, features)
    else:
        reply = "請輸入股票代碼，例如 AAPL 或 NVDA。"
    line_bot_api.reply_message(event.reply_token, TextSendMessage(text=reply))

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
