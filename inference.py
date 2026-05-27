import re
import json
import joblib
import pandas as pd
import numpy as np
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)

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

def build_input_text(headline, contents):
    headline = clean_text(headline)
    contents = clean_text(contents)
    return f"Headline: {headline} [SEP] Content: {contents}"

def clean_text(text):
    if pd.isna(text):
        return ""
    text = str(text)
    text = text.replace("\n", " ").replace("\r", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text

def predict_all_models(headline, content,
                       logreg_path=LOGREG_MODEL_PATH,
                       nb_path=NB_MODEL_PATH,
                       bert_dir=DISTILBERT_OUTPUT_DIR):
    """
    Predict fake/real using all trained models.
    """
    logreg_model = joblib.load(logreg_path)
    nb_model = joblib.load(nb_path)

    tokenizer = AutoTokenizer.from_pretrained(bert_dir)
    bert_model = AutoModelForSequenceClassification.from_pretrained(bert_dir)

    device = get_device()
    bert_model.to(device)
    bert_model.eval()

    text = build_input_text(headline, content)

    logreg_pred = int(logreg_model.predict([text])[0])
    logreg_probs = logreg_model.predict_proba([text])[0]

    logreg_result = {
        "label": LABEL_MAP[logreg_pred],
        "fake_prob": float(logreg_probs[0]),
        "true_prob": float(logreg_probs[1]),
    }

   
    nb_pred = int(nb_model.predict([text])[0])
    nb_probs = nb_model.predict_proba([text])[0]

    nb_result = {
        "label": LABEL_MAP[nb_pred],
        "fake_prob": float(nb_probs[0]),
        "true_prob": float(nb_probs[1]),
    }

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

result = predict_all_models(
    headline="Deux actes violents, des victimes sous le choc",
    content="""Deux incidents inquiétants se sont produits à Moka dans la matinée du vendredi 17 avril, semant la peur chez les victimes.

Vers 4 heures du matin, un agent de sécurité a vécu des moments de terreur alors qu’il était dans son bureau à Bagatelle. Deux individus armés d’un couteau et d’une matraque ont fait irruption en forçant une ouverture. Ils l’ont maîtrisé, lui ont attaché les mains avec des serre-câbles et l’ont menacé avant de le contraindre à les accompagner. Les malfaiteurs se sont emparés d’un coffre-fort avant d’enfermer la victime, pieds et mains liés, dans une pièce. Sous le choc, l’agent de sécurité n’a subi aucune blessure. Selon les premières informations, le coffre ne contenait rien de valeur.

Quelques heures plus tôt, un retraité de 76 ans a lui aussi été confronté à une situation terrifiante à son domicile, à Moka. Deux hommes, armés d’un couteau, l’ont menacé en lui réclamant ses objets de valeur. Pris de panique, il a crié à l’aide et a réussi à s’enfuir, mettant ainsi les agresseurs en fuite. Ces derniers ont quitté les lieux à bord d’une voiture grise. Là encore, aucune blessure n’a été rapportée et rien n’a été volé. La victime affirme reconnaître l’un des suspects, un homme habitant la région.

La police de Moka, appuyée par la Criminal Investigation Division et la Divisional Crime Intelligence Unit, a ouvert une enquête pour faire la lumière sur ces deux affaires."""
)

print(json.dumps(result, indent=4, ensure_ascii=False))