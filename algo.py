import os
import re
import random
import warnings
import joblib
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
    matthews_corrcoef,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    auc,
)

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)

warnings.filterwarnings("ignore")

REAL_CSV_PATH = "real_news_with_labels.csv"
FAKE_CSV_PATH = "fake_news.csv"

RANDOM_SEED = 42
TEST_SIZE = 0.15
VAL_SIZE = 0.15

DISTILBERT_MODEL_NAME = "distilbert-base-multilingual-cased"
MAX_LENGTH = 192
DISTILBERT_OUTPUT_DIR = "./distilbert_fake_news_model"

LOGREG_MODEL_PATH = "./tfidf_logreg_fake_news.pkl"
NB_MODEL_PATH = "./tfidf_nb_fake_news.pkl"

CLASSICAL_SAMPLE_SIZE = None
DISTILBERT_SAMPLE_SIZE = None

TRAIN_BATCH_SIZE = 4
EVAL_BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 2
LEARNING_RATE = 2e-5
NUM_EPOCHS = 2
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1

LABEL_MAP = {
    0: "FAKE",
    1: "TRUE"
}


def is_kaggle_environment():
    return os.path.exists("/kaggle/input") or "KAGGLE_URL_BASE" in os.environ


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def configure_torch_for_device(device):
    if device.type != "cuda":
        return

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")


DEVICE = get_device()
configure_torch_for_device(DEVICE)


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(RANDOM_SEED)


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


def normalize_label(value):
    if pd.isna(value):
        return np.nan

    value_str = str(value).strip().lower()

    label_mapping = {
        "fake": 0,
        "real": 1,
        "true": 1,
        "0": 0,
        "1": 1,
    }

    return label_mapping.get(value_str, np.nan)


def load_balanced_real_fake_dataset(real_csv_path, fake_csv_path, sample_size_per_class=6000):
    real_df = pd.read_csv(real_csv_path)
    fake_df = pd.read_csv(fake_csv_path)

    required_columns = {"headline", "content", "label"}

    if not required_columns.issubset(real_df.columns):
        raise ValueError(f"real_news_dataset.csv must contain columns: {required_columns}")
    if not required_columns.issubset(fake_df.columns):
        raise ValueError(f"fake_news_dataset.csv must contain columns: {required_columns}")

    real_df = real_df[["headline", "content", "label"]].copy()
    fake_df = fake_df[["headline", "content", "label"]].copy()

    print(len(real_df))
    print(len(fake_df))

    if len(real_df) < sample_size_per_class:
        raise ValueError(
            f"real_news_dataset.csv only has {len(real_df)} rows, "
            f"but {sample_size_per_class} rows are required."
        )

    if len(fake_df) < sample_size_per_class:
        raise ValueError(
            f"fake_news_dataset.csv only has {len(fake_df)} rows, "
            f"but {sample_size_per_class} rows are required."
        )

    real_df = real_df.sample(n=sample_size_per_class, random_state=RANDOM_SEED).reset_index(drop=True)
    fake_df = fake_df.sample(n=sample_size_per_class, random_state=RANDOM_SEED).reset_index(drop=True)

    for df in (real_df, fake_df):
        df["headline"] = df["headline"].apply(clean_text)
        df["content"] = df["content"].apply(clean_text)
        df["label"] = df["label"].apply(normalize_label)

    if real_df["label"].isna().any():
        raise ValueError("Invalid label values found in real_news_dataset.csv.")

    if fake_df["label"].isna().any():
        raise ValueError("Invalid label values found in fake_news_dataset.csv.")

    real_df["label"] = real_df["label"].astype(int)
    fake_df["label"] = fake_df["label"].astype(int)

    df = pd.concat([real_df, fake_df], ignore_index=True)
    df = df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

    df["combined_text"] = df.apply(
        lambda row: build_input_text(row["headline"], row["content"]),
        axis=1
    )

    df = df[df["combined_text"].str.strip().ne("")].reset_index(drop=True)
    return df


def maybe_sample(df, sample_size):
    if sample_size is None or sample_size >= len(df):
        return df.reset_index(drop=True)
    return df.sample(n=sample_size, random_state=RANDOM_SEED).reset_index(drop=True)


def split_dataset(df):
    train_val_df, test_df = train_test_split(
        df,
        test_size=TEST_SIZE,
        stratify=df["label"],
        random_state=RANDOM_SEED,
    )

    val_relative_size = VAL_SIZE / (1 - TEST_SIZE)

    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_relative_size,
        stratify=train_val_df["label"],
        random_state=RANDOM_SEED,
    )

    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def evaluate_model(y_true, y_pred, model_name, y_proba=None):
    acc = accuracy_score(y_true, y_pred)
    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )

    report = classification_report(
        y_true,
        y_pred,
        target_names=["Fake", "True"],
        zero_division=0,
        output_dict=True,
    )

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    metrics = {
        "model": model_name,
        "accuracy": acc,
        "balanced_accuracy": balanced_acc,
        "mcc": mcc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "weighted_precision": weighted_precision,
        "weighted_recall": weighted_recall,
        "weighted_f1": weighted_f1,
        "confusion_matrix": cm,
        "classification_report": report,
        "y_true": np.array(y_true),
        "y_pred": np.array(y_pred),
        "y_proba": np.array(y_proba) if y_proba is not None else None,
    }

    if y_proba is not None:
        metrics["roc_auc"] = roc_auc_score(y_true, y_proba)

    print("\n" + "=" * 70)
    print(model_name)
    print("=" * 70)
    print(f"Accuracy : {acc:.4f}")
    print(f"Balanced Accuracy : {balanced_acc:.4f}")
    print(f"MCC      : {mcc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-score : {f1:.4f}")
    print(f"Macro Precision/Recall/F1   : {macro_precision:.4f} / {macro_recall:.4f} / {macro_f1:.4f}")
    print(f"Weighted Precision/Recall/F1: {weighted_precision:.4f} / {weighted_recall:.4f} / {weighted_f1:.4f}")

    if y_proba is not None:
        print(f"ROC-AUC  : {metrics['roc_auc']:.4f}")

    print("\nClassification Report:")
    report_df = pd.DataFrame(report).transpose().round(4)
    print(report_df.to_string())

    print("Confusion Matrix:")
    cm_df = pd.DataFrame(
        cm,
        index=["actual_fake", "actual_true"],
        columns=["pred_fake", "pred_true"],
    )
    print(cm_df.to_string())

    return metrics


def build_tfidf_vectorizer():
    return TfidfVectorizer(
        stop_words="english",
        max_df=0.7,
        min_df=2,
        ngram_range=(1, 2),
        lowercase=True,
    )


def train_tfidf_logreg(train_df, test_df):
    model = Pipeline([
        ("tfidf", build_tfidf_vectorizer()),
        ("clf", LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=RANDOM_SEED,
        )),
    ])

    model.fit(train_df["combined_text"], train_df["label"])
    preds = model.predict(test_df["combined_text"])
    probs = model.predict_proba(test_df["combined_text"])[:, 1]

    metrics = evaluate_model(
        test_df["label"],
        preds,
        "TF-IDF + Logistic Regression",
        y_proba=probs
    )

    return model, metrics


def train_tfidf_naive_bayes(train_df, test_df):
    model = Pipeline([
        ("tfidf", build_tfidf_vectorizer()),
        ("clf", MultinomialNB(alpha=1.0)),
    ])

    model.fit(train_df["combined_text"], train_df["label"])
    preds = model.predict(test_df["combined_text"])
    probs = model.predict_proba(test_df["combined_text"])[:, 1]

    metrics = evaluate_model(
        test_df["label"],
        preds,
        "TF-IDF + Naive Bayes",
        y_proba=probs
    )

    return model, metrics


def predict_classical(model, headline, contents):
    text = build_input_text(headline, contents)
    pred = int(model.predict([text])[0])

    result = {
        "label_id": pred,
        "label": LABEL_MAP[pred]
    }

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba([text])[0]
        result["fake_probability"] = float(probs[0])
        result["true_probability"] = float(probs[1])

    return result


@dataclass
class BertArtifacts:
    tokenizer: AutoTokenizer
    model: AutoModelForSequenceClassification
    trainer: Trainer


def build_hf_dataset(df):
    return Dataset.from_pandas(df[["combined_text", "label"]], preserve_index=False)


def tokenize_function(examples, tokenizer):
    return tokenizer(
        examples["combined_text"],
        truncation=True,
        max_length=MAX_LENGTH,
    )


def compute_hf_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary", zero_division=0
    )

    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def train_distilbert(train_df, val_df, test_df):
    tokenizer = AutoTokenizer.from_pretrained(DISTILBERT_MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        DISTILBERT_MODEL_NAME,
        num_labels=2,
    )

    train_ds = build_hf_dataset(train_df)
    val_ds = build_hf_dataset(val_df)
    test_ds = build_hf_dataset(test_df)

    train_ds = train_ds.map(lambda x: tokenize_function(x, tokenizer), batched=True)
    val_ds = val_ds.map(lambda x: tokenize_function(x, tokenizer), batched=True)
    test_ds = test_ds.map(lambda x: tokenize_function(x, tokenizer), batched=True)

    columns_to_keep = ["input_ids", "attention_mask", "label"]
    train_ds.set_format(type="torch", columns=columns_to_keep)
    val_ds.set_format(type="torch", columns=columns_to_keep)
    test_ds.set_format(type="torch", columns=columns_to_keep)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    use_cuda = DEVICE.type == "cuda"
    use_fp16 = use_cuda
    dataloader_workers = 2 if is_kaggle_environment() else 0

    training_args = TrainingArguments(
        output_dir=DISTILBERT_OUTPUT_DIR,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=100,
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        num_train_epochs=NUM_EPOCHS,
        warmup_ratio=WARMUP_RATIO,
        weight_decay=WEIGHT_DECAY,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        save_total_limit=2,
        report_to="none",
        fp16=use_fp16,
        dataloader_num_workers=dataloader_workers,
        dataloader_pin_memory=use_cuda,
        gradient_checkpointing=use_cuda,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=data_collator,
        processing_class=tokenizer,
        compute_metrics=compute_hf_metrics,
    )

    trainer.train()

    preds_output = trainer.predict(test_ds)
    probs = torch.softmax(torch.tensor(preds_output.predictions), dim=-1).numpy()
    preds = np.argmax(probs, axis=-1)
    y_true = test_df["label"].to_numpy()

    metrics = evaluate_model(
        y_true,
        preds,
        "DistilBERT",
        y_proba=probs[:, 1]
    )

    trainer.save_model(DISTILBERT_OUTPUT_DIR)
    tokenizer.save_pretrained(DISTILBERT_OUTPUT_DIR)

    return BertArtifacts(
        tokenizer=tokenizer,
        model=trainer.model,
        trainer=trainer
    ), metrics


def predict_distilbert(artifacts, headline, contents):
    text = build_input_text(headline, contents)

    device = get_device()
    artifacts.model.to(device)
    artifacts.model.eval()

    inputs = artifacts.tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LENGTH,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = artifacts.model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()[0]

    pred = int(np.argmax(probs))

    return {
        "label_id": pred,
        "label": LABEL_MAP[pred],
        "fake_probability": float(probs[0]),
        "true_probability": float(probs[1]),
    }


def save_classical_models(logreg_model, nb_model):
    joblib.dump(logreg_model, LOGREG_MODEL_PATH)
    joblib.dump(nb_model, NB_MODEL_PATH)
    print(f"\nSaved Logistic Regression model to: {LOGREG_MODEL_PATH}")
    print(f"Saved Naive Bayes model to: {NB_MODEL_PATH}")


def load_classical_model(path):
    return joblib.load(path)


def load_saved_distilbert(model_dir=DISTILBERT_OUTPUT_DIR):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.to(get_device())
    model.eval()

    return BertArtifacts(
        tokenizer=tokenizer,
        model=model,
        trainer=None
    )


def plot_metric_comparison(results, save_path=None):
    rows = []
    for r in results:
        rows.append({
            "model": r["model"],
            "accuracy": r["accuracy"],
            "precision": r["precision"],
            "recall": r["recall"],
            "f1": r["f1"],
            "roc_auc": r.get("roc_auc", np.nan),
            "balanced_accuracy": r["balanced_accuracy"],
            "mcc": r["mcc"],
        })

    df = pd.DataFrame(rows)

    plot_df = df.set_index("model")[[
        "accuracy", "precision", "recall", "f1", "roc_auc"
    ]]

    plot_df.plot(kind="bar", figsize=(10, 6))
    plt.title("Model Evaluation Metrics Comparison")
    plt.ylabel("Score")
    plt.ylim(0, 1.05)
    plt.xticks(rotation=15)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()

    return df


def plot_confusion_matrix(cm, model_name, save_path=None):
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation="nearest")
    plt.title(f"Confusion Matrix - {model_name}")
    plt.colorbar()

    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ["Fake", "True"])
    plt.yticks(tick_marks, ["Fake", "True"])

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")

    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def plot_roc_curves(results, save_path=None):
    plt.figure(figsize=(8, 6))

    for r in results:
        if r["y_proba"] is None:
            continue

        fpr, tpr, _ = roc_curve(r["y_true"], r["y_proba"])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{r['model']} (AUC = {roc_auc:.3f})")

    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.title("ROC Curves")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def plot_precision_recall_curves(results, save_path=None):
    plt.figure(figsize=(8, 6))

    for r in results:
        if r["y_proba"] is None:
            continue

        precision_vals, recall_vals, _ = precision_recall_curve(r["y_true"], r["y_proba"])
        pr_auc = auc(recall_vals, precision_vals)
        plt.plot(recall_vals, precision_vals, label=f"{r['model']} (AUC = {pr_auc:.3f})")

    plt.title("Precision-Recall Curves")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def plot_distilbert_training_history(trainer, save_path=None):
    if trainer is None:
        print("No trainer available for DistilBERT history plotting.")
        return

    log_history = trainer.state.log_history
    if not log_history:
        print("No training history found.")
        return

    train_steps = []
    train_losses = []
    eval_steps = []
    eval_losses = []

    for entry in log_history:
        if "loss" in entry and "epoch" in entry:
            train_steps.append(entry.get("step", len(train_steps)))
            train_losses.append(entry["loss"])

        if "eval_loss" in entry and "epoch" in entry:
            eval_steps.append(entry.get("step", len(eval_steps)))
            eval_losses.append(entry["eval_loss"])

    plt.figure(figsize=(8, 6))

    if train_steps and train_losses:
        plt.plot(train_steps, train_losses, label="Training Loss")

    if eval_steps and eval_losses:
        plt.plot(eval_steps, eval_losses, label="Validation Loss")

    plt.title("DistilBERT Training and Validation Loss")
    plt.xlabel("Training Step")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def main():
    print("Device being used:", DEVICE)
    if DEVICE.type == "cuda":
        device_name = torch.cuda.get_device_name(0)
        print(f"CUDA device: {device_name}")
        if "P100" in device_name:
            print("Kaggle Tesla P100 detected. Training will use mixed precision (fp16).")

    print(f"Loading real dataset from: {REAL_CSV_PATH}")
    print(f"Loading fake dataset from: {FAKE_CSV_PATH}")

    df = load_balanced_real_fake_dataset(
        real_csv_path=REAL_CSV_PATH,
        fake_csv_path=FAKE_CSV_PATH,
        sample_size_per_class=6000
    )

    print(f"Total samples: {len(df)}")
    print("Class distribution:")
    print(df["label"].value_counts().sort_index())

    train_df, val_df, test_df = split_dataset(df)

    classical_train_df = maybe_sample(train_df, CLASSICAL_SAMPLE_SIZE)
    classical_test_df = test_df.copy()

    distilbert_train_df = maybe_sample(train_df, DISTILBERT_SAMPLE_SIZE)
    distilbert_val_df = val_df.copy()
    distilbert_test_df = test_df.copy()

    print("\nDataset split:")
    print(f"Train: {len(train_df)}")
    print(f"Val  : {len(val_df)}")
    print(f"Test : {len(test_df)}")

    if DISTILBERT_SAMPLE_SIZE is not None:
        print(f"\nDistilBERT training sample size: {len(distilbert_train_df)}")

    results = []

    print("\nTraining TF-IDF + Logistic Regression...")
    logreg_model, logreg_metrics = train_tfidf_logreg(classical_train_df, classical_test_df)
    results.append(logreg_metrics)

    print("\nTraining TF-IDF + Naive Bayes...")
    nb_model, nb_metrics = train_tfidf_naive_bayes(classical_train_df, classical_test_df)
    results.append(nb_metrics)

    print("\nTraining DistilBERT...")
    distilbert_artifacts, bert_metrics = train_distilbert(
        distilbert_train_df,
        distilbert_val_df,
        distilbert_test_df,
    )
    results.append(bert_metrics)

    save_classical_models(logreg_model, nb_model)

    if is_kaggle_environment():
        metrics_df = plot_metric_comparison(
            results,
            save_path="/kaggle/working/model_metrics_comparison.png"
        )
    else:
        metrics_df = plot_metric_comparison(
            results,
            save_path="/metrics/model_metrics_comparison.png"
        )

    print("\nMetrics summary:")
    print(metrics_df.round(4).to_string(index=False))

    for r in results:
        safe_name = (
            r["model"]
            .lower()
            .replace(" ", "_")
            .replace("+", "plus")
            .replace("-", "_")
        )

        if is_kaggle_environment():

            plot_confusion_matrix(
                r["confusion_matrix"],
                r["model"],
                save_path=f"/metrics/{safe_name}_confusion_matrix.png"
            )
        else:
            plot_confusion_matrix(
                r["confusion_matrix"],
                r["model"],
                save_path=f"/metrics/{safe_name}_confusion_matrix.png"
            )

    if is_kaggle_environment():
        plot_roc_curves(results, save_path="/kaggle/working/roc_curves.png")
        plot_precision_recall_curves(results, save_path="/kaggle/working/precision_recall_curves.png")
        plot_distilbert_training_history(
            distilbert_artifacts.trainer,
            save_path="/kaggle/working/distilbert_training_history.png"
        )
    else:
        plot_roc_curves(results, save_path="/metrics/roc_curves.png")
        plot_precision_recall_curves(results, save_path="/metrics/precision_recall_curves.png")
        plot_distilbert_training_history(
            distilbert_artifacts.trainer,
            save_path="/metrics/distilbert_training_history.png"
        )

    if is_kaggle_environment():
        metrics_df.to_csv("/kaggle/working/model_comparison.csv", index=False)
        print("\nSaved evaluation summary to /kaggle/working/model_comparison.csv")
    else:
        metrics_df.to_csv("/metrics/model_comparison.csv", index=False)
        print("\nSaved evaluation summary to /metrics/model_comparison.csv")

    sample_headline = "Government announces urgent reform after major backlash"
    sample_contents = """
    Officials said the policy would be revised after public criticism spread online.
    Analysts remain divided on whether the original reports overstated the situation.
    """

    print("\n" + "=" * 70)
    print("SAMPLE INFERENCE")
    print("=" * 70)

    print("\nLogistic Regression:")
    print(predict_classical(logreg_model, sample_headline, sample_contents))

    print("\nNaive Bayes:")
    print(predict_classical(nb_model, sample_headline, sample_contents))

    print("\nDistilBERT:")
    print(predict_distilbert(distilbert_artifacts, sample_headline, sample_contents))


if __name__ == "__main__":
    main()