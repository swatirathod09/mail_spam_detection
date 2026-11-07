# Mailcheck

Mailcheck is a lightweight web application that classifies email and message text as **spam** or **not spam**. It combines a scikit-learn text-classification model with a simple Flask web interface to show a spam probability and highlight words that influenced the prediction.

## Features

- Classifies pasted text as spam or not spam.
- Shows the estimated probability that a message is spam.
- Highlights message terms and lists influential words.
- Includes sample messages and a short recent-check history.
- Provides a responsive interface for desktop and mobile screens.

## How it works

The training script converts text to lowercase TF-IDF features and fits a logistic regression classifier. It evaluates the model on a held-out test split and saves the vectorizer and classifier together in `model.pkl`. The Flask application loads that model and serves the interface and prediction endpoint.

## Requirements

- Python 3.10 or later
- Packages listed in `requirements.txt`

## Run locally

From the project directory, install dependencies, train the model, and start the web server:

```bash
python -m pip install -r requirements.txt
python train.py
python app.py
```

Open [http://127.0.0.1:5000](http://127.0.0.1:5000) in your browser. Stop the development server with `Ctrl+C`.

On Windows, if `python` is not available as a command, use the Python launcher (`py`) or activate your virtual environment before running the commands.

## Training with your own data

Place a CSV file at `data/mail_data.csv` with these column names:

| Column | Description | Accepted values |
|---|---|---|
| `Category` | Message classification | `spam` or `ham` |
| `Message` | Message text | The email or message body |

Then run:

```bash
python train.py
```

When `data/mail_data.csv` is absent, the training script uses `data/sample_mail_data.csv`. The bundled sample is intentionally small and is for demonstration only; its reported accuracy should not be taken as evidence of real-world performance. For meaningful evaluation, use a sufficiently large, representative, and correctly labeled dataset.

## Project files

```text
app.py                       Flask application and prediction endpoint
train.py                     Model training and evaluation
model.pkl                    Saved vectorizer and classifier
data/sample_mail_data.csv    Demonstration training data
templates/index.html         Web interface
requirements.txt             Python dependencies
```

## Important note

Mailcheck is a demonstration project, not a production security or anti-phishing service. Predictions can be wrong. Do not rely on it as the sole basis for opening attachments, following links, or deciding whether a message is safe. The Flask development server is intended for local development and should not be exposed as a production service.
