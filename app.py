from flask import Flask, request, jsonify
from deep_translator import GoogleTranslator
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import re
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load model và tokenizer
model = tf.keras.models.load_model("news_classifier.keras")
with open("tokenizer.pkl", "rb") as f:
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
        return text  # fallback: return original if translation fails

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    title = data.get("title", "")
    desc = data.get("description", "")
    full_text = f"{title} {desc}"

    # ✨ Dịch sang tiếng Anh
    translated_text = translate_vi_to_en(full_text)

    # Làm sạch và dự đoán
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
    app.run(debug=True)
