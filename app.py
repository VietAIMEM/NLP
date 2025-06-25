from flask import Flask, request, jsonify
from deep_translator import GoogleTranslator
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import re
import os
import gdown
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

MODEL_PATH = "news_classifier.keras"
TOKENIZER_PATH = "tokenizer.pkl"

# Google Drive file IDs
MODEL_FILE_ID = "1c6Ls2RVKCPMUJ3VtcaqUFjbDjSQbb_jE"       # <- Thay bằng ID thật
TOKENIZER_FILE_ID = "1eoOgifu2n1a3WNSHrBtMZrUgWsBNvJLE"   # <- Thay bằng ID thật

def download_model():
    os.makedirs("model", exist_ok=True)
    if not os.path.exists(MODEL_PATH):
        print("🔽 Downloading model from Google Drive...")
        gdown.download(f"https://drive.google.com/uc?id={MODEL_FILE_ID}", MODEL_PATH, quiet=False)
    if not os.path.exists(TOKENIZER_PATH):
        print("🔽 Downloading tokenizer from Google Drive...")
        gdown.download(f"https://drive.google.com/uc?id={TOKENIZER_FILE_ID}", TOKENIZER_PATH, quiet=False)

# Tải mô hình và tokenizer nếu cần
download_model()

# Load model và tokenizer
model = tf.keras.models.load_model(MODEL_PATH)
with open(TOKENIZER_PATH, "rb") as f:
    tokenizer = pickle.load(f)

MAXLEN = 177
LABELS = ["Class 1", "Class 2", "Class 3", "Class 4"]

def clean_text(text):
    text = re.sub(r'\\[a-z]', ' ', text)
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip().lower()

def translate_vi_to_en(text):
    try:
        return GoogleTranslator(source='vi', target='en').translate(text)
    except Exception as e:
        print("Translation failed:", e)
        return text  # fallback

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    title = data.get("title", "")
    desc = data.get("description", "")
    full_text = f"{title} {desc}"

    translated_text = translate_vi_to_en(full_text)
    clean = clean_text(translated_text)
    seq = tokenizer.texts_to_sequences([clean])
    padded = pad_sequences(seq, maxlen=MAXLEN, padding='post', truncating='post')
    pred = model.predict(padded)[0]
    predicted_label = LABELS[pred.argmax()]
    confidence = float(pred.max())

    return jsonify({
        "prediction": predicted_label,
        "confidence": round(confidence, 4),
        "translated_input": translated_text
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
