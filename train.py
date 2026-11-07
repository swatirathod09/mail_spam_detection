"""Train the spam detector (same steps as the notebook) and save model.pkl."""
import os, joblib, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

path = "data/mail_data.csv" if os.path.exists("data/mail_data.csv") else "data/sample_mail_data.csv"
print("Training on", path)
df = pd.read_csv(path)
df = df.where(pd.notnull(df), "")
df["Category"] = df["Category"].map({"spam": 0, "ham": 1})   # spam = 0, ham = 1
X, Y = df["Message"], df["Category"].astype("int")
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=3)

vec = TfidfVectorizer(min_df=1, stop_words="english", lowercase=True)
X_train_f, X_test_f = vec.fit_transform(X_train), vec.transform(X_test)
model = LogisticRegression().fit(X_train_f, Y_train)

print("Train accuracy:", accuracy_score(Y_train, model.predict(X_train_f)))
print("Test accuracy: ", accuracy_score(Y_test, model.predict(X_test_f)))
joblib.dump({"vec": vec, "model": model}, "model.pkl")
print("Saved model.pkl")
