import re
import warnings
import joblib
import numpy as np
import pandas as pd
import torch

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
from transformers import AutoTokenizer, AutoModelForSequenceClassification

warnings.filterwarnings("ignore")

MAX_LENGTH = 192
DISTILBERT_OUTPUT_DIR = "models/distilbert_fake_news_model"

LOGREG_MODEL_PATH = "models/tfidf_logreg_fake_news.pkl"
NB_MODEL_PATH = "models/tfidf_nb_fake_news.pkl"

LABEL_MAP = {
    0: "FAKE",
    1: "TRUE"
}

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def clean_text(text):
    if pd.isna(text):
        return ""
    text = str(text)
    text = text.replace("\n", " ").replace("\r", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def build_input_text(headline, contents):
    headline = clean_text(headline)
    contents = clean_text(contents)
    return f"Headline: {headline} [SEP] Content: {contents}"


def load_models_into_app(app: FastAPI):
    device = get_device()

    app.state.logreg_model = joblib.load(LOGREG_MODEL_PATH)
    app.state.nb_model = joblib.load(NB_MODEL_PATH)

    app.state.tokenizer = AutoTokenizer.from_pretrained(DISTILBERT_OUTPUT_DIR)
    app.state.bert_model = AutoModelForSequenceClassification.from_pretrained(
        DISTILBERT_OUTPUT_DIR
    )

    app.state.device = device
    app.state.bert_model.to(device)
    app.state.bert_model.eval()

    warmup_text = "Headline: warmup [SEP] Content: warmup"
    inputs = app.state.tokenizer(
        warmup_text,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LENGTH,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        _ = app.state.bert_model(**inputs)


def clear_models_from_app(app: FastAPI):
    if hasattr(app.state, "bert_model"):
        del app.state.bert_model
    if hasattr(app.state, "tokenizer"):
        del app.state.tokenizer
    if hasattr(app.state, "logreg_model"):
        del app.state.logreg_model
    if hasattr(app.state, "nb_model"):
        del app.state.nb_model
    if hasattr(app.state, "device"):
        del app.state.device

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def ensure_models_loaded(app: FastAPI):
    required_attrs = [
        "logreg_model",
        "nb_model",
        "tokenizer",
        "bert_model",
        "device",
    ]
    for attr in required_attrs:
        if not hasattr(app.state, attr):
            raise RuntimeError(f"Missing required app.state attribute: {attr}")


def predict_all_models(app: FastAPI, headline, content):
    ensure_models_loaded(app)

    logreg_model = app.state.logreg_model
    nb_model = app.state.nb_model
    tokenizer = app.state.tokenizer
    bert_model = app.state.bert_model
    device = app.state.device

    text = build_input_text(headline, content)

    #Logistic Regression prediction
    logreg_pred = int(logreg_model.predict([text])[0])
    logreg_probs = logreg_model.predict_proba([text])[0]
    logreg_result = {
        "label": LABEL_MAP[logreg_pred],
        "fake_prob": float(logreg_probs[0]),
        "true_prob": float(logreg_probs[1]),
    }

    #Naive Bayes prediction
    nb_pred = int(nb_model.predict([text])[0])
    nb_probs = nb_model.predict_proba([text])[0]
    nb_result = {
        "label": LABEL_MAP[nb_pred],
        "fake_prob": float(nb_probs[0]),
        "true_prob": float(nb_probs[1]),
    }

    #DistilBERT prediction
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LENGTH,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = bert_model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()[0]

    bert_pred = int(np.argmax(probs))
    bert_result = {
        "label": LABEL_MAP[bert_pred],
        "fake_prob": float(probs[0]),
        "true_prob": float(probs[1]),
    }

    return {
        "logistic_regression": logreg_result,
        "naive_bayes": nb_result,
        "distilbert": bert_result,
    }

@asynccontextmanager
async def lifespan(app: FastAPI):
    load_models_into_app(app)
    yield
    clear_models_from_app(app)

app = FastAPI(
    title="Fake News Detection API",
    version="1.0.0",
    description="Predict fake or true news using Logistic Regression, Naive Bayes, and DistilBERT.",
    lifespan=lifespan,
)

class PredictRequest(BaseModel):
    headline: str = Field(..., min_length=1, description="News headline")
    content: str = Field(..., min_length=1, description="News article content")


class ModelPrediction(BaseModel):
    label: str
    fake_prob: float
    true_prob: float


class PredictResponse(BaseModel):
    logistic_regression: ModelPrediction
    naive_bayes: ModelPrediction
    distilbert: ModelPrediction

@app.get("/health")
def health(request: Request):
    try:
        ensure_models_loaded(request.app)
        return {
            "status": "ok",
            "device": str(request.app.state.device),
            "models_loaded": True,
        }
    except Exception as e:
        return {
            "status": "error",
            "models_loaded": False,
            "detail": str(e),
        }


@app.post("/predict", response_model=PredictResponse)
def predict(request: Request, body: PredictRequest):
    try:
        result = predict_all_models(
            app=request.app,
            headline=body.headline,
            content=body.content,
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))