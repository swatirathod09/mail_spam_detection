import joblib
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)
bundle = joblib.load("model.pkl")
vec, model = bundle["vec"], bundle["model"]
inv_vocab = {i: w for w, i in vec.vocabulary_.items()}
coef = model.coef_[0]          # positive = ham, negative = spam

@app.get("/")
def index():
    return render_template("index.html")

@app.post("/predict")
def predict():
    text = (request.get_json(silent=True) or {}).get("text", "").strip()
    if not text:
        return jsonify(error="Paste an email first."), 400
    X = vec.transform([text])
    p_ham = float(model.predict_proba(X)[0][1])
    row = X.tocoo()
    # weight > 0 pushes toward spam, < 0 toward not spam
    words = [{"word": inv_vocab[j], "weight": float(-v * coef[j])} for j, v in zip(row.col, row.data)]
    words.sort(key=lambda w: -abs(w["weight"]))
    return jsonify(label="ham" if p_ham >= 0.5 else "spam", spam_prob=1 - p_ham, words=words[:12])

if __name__ == "__main__":
    app.run(debug=True)
