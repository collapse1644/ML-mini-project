"""Beginner-friendly ML paradigm classifier.

This file trains a small text classifier that identifies whether a business
problem sounds like Supervised Learning, Unsupervised Learning, or
Reinforcement Learning.

The project intentionally keeps the steps simple:
1. Load a CSV dataset.
2. Clean the text.
3. Convert text into TF-IDF features.
4. Train Logistic Regression.
5. Combine a tiny keyword rule layer with model predictions.
"""

from __future__ import annotations

from io import StringIO
import string
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


DATA_PATH = Path(__file__).resolve().parent / "data" / "ml_paradigm_dataset.csv"
RANDOM_STATE = 42


FALLBACK_DATA_CSV = """text,label
"Predict customer churn using previous purchase and support history",supervised
"Classify emails as spam or not spam",supervised
"Forecast monthly sales from historical revenue data",supervised
"Estimate house prices from size location and number of rooms",supervised
"Predict whether a loan applicant will default",supervised
"Classify product reviews as positive neutral or negative",supervised
"Detect fraudulent credit card transactions using labeled examples",supervised
"Predict delivery time from distance traffic and weather data",supervised
"Classify medical images as healthy or diseased",supervised
"Estimate employee attrition from HR records",supervised
"Predict demand for a new product based on past orders",supervised
"Classify support tickets into billing technical or account categories",supervised
"Predict insurance claim amount from customer profile data",supervised
"Identify whether a customer will click an advertisement",supervised
"Forecast machine failure using sensor readings and past labels",supervised
"Cluster customers into segments based on buying behavior",unsupervised
"Group similar products using only product descriptions",unsupervised
"Find hidden patterns in website browsing behavior",unsupervised
"Discover groups of users with similar app usage habits",unsupervised
"Segment stores based on sales volume location and product mix",unsupervised
"Group news articles by topic without predefined categories",unsupervised
"Detect unusual network traffic without labeled attack examples",unsupervised
"Find natural clusters in survey responses",unsupervised
"Organize documents into themes automatically",unsupervised
"Reduce many customer features into a smaller set of useful dimensions",unsupervised
"Group suppliers based on delivery time quality and cost",unsupervised
"Find similar songs based on listening patterns",unsupervised
"Discover market basket patterns from transaction data",unsupervised
"Cluster patients by symptoms when diagnosis labels are unavailable",unsupervised
"Identify outlier transactions that look different from normal behavior",unsupervised
"Train an agent to choose prices that maximize long term profit",reinforcement
"Build a robot agent that learns to navigate a warehouse using rewards",reinforcement
"Teach a game AI to win by receiving rewards for good moves",reinforcement
"Optimize ad bidding where an agent learns from clicks and rewards",reinforcement
"Create a recommendation agent that learns from user feedback over time",reinforcement
"Control traffic lights using rewards for reducing waiting time",reinforcement
"Train a delivery drone to choose routes and avoid obstacles",reinforcement
"Develop an agent that learns inventory ordering decisions",reinforcement
"Let a chatbot learn conversation strategies from reward signals",reinforcement
"Optimize energy usage with an agent that receives rewards for savings",reinforcement
"Train a trading agent to buy sell or hold based on market rewards",reinforcement
"Teach an autonomous vehicle agent to stay safe and reach destinations",reinforcement
"Use rewards to help a system learn the best sequence of marketing actions",reinforcement
"Train a scheduling agent to assign jobs and reduce machine idle time",reinforcement
"Build a learning agent that adapts game difficulty based on player responses",reinforcement
"""


KEYWORD_RULES: Dict[str, List[str]] = {
    "supervised": ["predict", "classify"],
    "unsupervised": ["cluster", "group"],
    "reinforcement": ["reward", "agent"],
}


ALGORITHM_SUGGESTIONS: Dict[str, List[str]] = {
    "supervised": [
        "Logistic Regression",
        "Decision Tree",
        "Random Forest",
        "Support Vector Machine",
        "Gradient Boosting",
    ],
    "unsupervised": [
        "K-Means Clustering",
        "Hierarchical Clustering",
        "DBSCAN",
        "Principal Component Analysis",
        "Gaussian Mixture Models",
    ],
    "reinforcement": [
        "Q-Learning",
        "Deep Q-Networks",
        "Policy Gradient Methods",
        "Actor-Critic Methods",
        "Multi-Armed Bandits",
    ],
}


WORKFLOWS: Dict[str, List[str]] = {
    "supervised": [
        "Collect examples where the correct answer is already known.",
        "Clean the data and choose useful input features.",
        "Split the data into training and testing sets.",
        "Train a model to map inputs to the known labels or values.",
        "Evaluate the model and use it to predict new cases.",
    ],
    "unsupervised": [
        "Collect data without needing predefined target labels.",
        "Clean and scale the features so patterns are easier to compare.",
        "Choose a method for grouping, reducing, or detecting unusual data.",
        "Fit the model and inspect the discovered patterns.",
        "Name, validate, and use the groups or patterns in the business process.",
    ],
    "reinforcement": [
        "Define the environment where decisions are made.",
        "Define the agent, possible actions, and reward signal.",
        "Let the agent try actions and observe rewards or penalties.",
        "Update the strategy so better actions become more likely.",
        "Test the learned policy safely before using it in the real world.",
    ],
}


@dataclass
class PredictionResult:
    """A readable result object for the app and command line output."""

    ml_type: str
    confidence: float
    reason: str
    suggested_algorithms: List[str]
    workflow: List[str]
    model_prediction: str
    model_confidence: float
    matched_keyword: Optional[str]


def clean_text(text: str) -> str:
    """Lowercase text, remove punctuation, and normalize extra spaces."""

    text = str(text).lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = " ".join(text.split())
    return text


def load_dataset(csv_path: Path = DATA_PATH) -> pd.DataFrame:
    """Load the CSV dataset and check that the required columns exist.

    The CSV file is preferred because it is easy to expand. The fallback keeps
    the Streamlit Cloud demo working even if the data folder is not deployed.
    """

    csv_path = Path(csv_path)

    if csv_path.exists():
        data = pd.read_csv(csv_path)
    else:
        data = pd.read_csv(StringIO(FALLBACK_DATA_CSV))

    required_columns = {"text", "label"}

    if not required_columns.issubset(data.columns):
        raise ValueError('Dataset must contain "text" and "label" columns.')

    return data


def find_keyword_rule(text: str) -> Tuple[Optional[str], Optional[str]]:
    """Return the rule label and keyword if a simple keyword rule matches."""

    cleaned = clean_text(text)
    words = set(cleaned.split())

    for label, keywords in KEYWORD_RULES.items():
        for keyword in keywords:
            if keyword in words:
                return label, keyword

    return None, None


def build_pipeline() -> Pipeline:
    """Create the TF-IDF + Logistic Regression pipeline."""

    return Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(preprocessor=clean_text)),
            (
                "classifier",
                LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
            ),
        ]
    )


def train_model(
    csv_path: Path = DATA_PATH,
) -> Tuple[Pipeline, pd.Series, pd.Series, pd.Series, pd.Series]:
    """Load data, split it, train the model, and return useful test data."""

    data = load_dataset(csv_path)
    X = data["text"]
    y = data["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    model = build_pipeline()
    model.fit(X_train, y_train)

    return model, X_train, X_test, y_train, y_test


def evaluate_model(model: Pipeline, X_test: pd.Series, y_test: pd.Series) -> None:
    """Print accuracy and confusion matrix for a trained model."""

    predictions = model.predict(X_test)
    labels = list(model.classes_)

    print("Accuracy:", round(accuracy_score(y_test, predictions), 3))
    print("\nLabels:", labels)
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, predictions, labels=labels))


def predict_with_model(model: Pipeline, text: str) -> Tuple[str, float]:
    """Use the trained model and return the label with its probability."""

    predicted_label = model.predict([text])[0]
    probabilities = model.predict_proba([text])[0]
    class_index = list(model.classes_).index(predicted_label)
    confidence = float(probabilities[class_index])

    return predicted_label, confidence


def combine_rule_and_model(
    rule_label: Optional[str],
    matched_keyword: Optional[str],
    model_label: str,
    model_confidence: float,
) -> Tuple[str, float, str]:
    """Combine the keyword rule layer with the machine learning model.

    If a rule keyword appears, we use the rule result and boost confidence.
    If no keyword appears, we trust the model prediction.
    """

    if rule_label and matched_keyword:
        confidence = max(model_confidence, 0.85)

        if rule_label == model_label:
            reason = (
                f'Keyword "{matched_keyword}" detected, and the model agreed.'
            )
            confidence = max(confidence, 0.90)
        else:
            reason = (
                f'Keyword "{matched_keyword}" detected, so the rule layer chose '
                f"{rule_label} even though the model predicted {model_label}."
            )

        return rule_label, confidence, reason

    reason = "No rule keyword detected, so the model prediction was used."
    return model_label, model_confidence, reason


def predict_ml_type(model: Pipeline, text: str) -> PredictionResult:
    """Return an enhanced prediction result for a business problem sentence."""

    rule_label, matched_keyword = find_keyword_rule(text)
    model_label, model_confidence = predict_with_model(model, text)

    final_label, final_confidence, reason = combine_rule_and_model(
        rule_label=rule_label,
        matched_keyword=matched_keyword,
        model_label=model_label,
        model_confidence=model_confidence,
    )

    return PredictionResult(
        ml_type=final_label,
        confidence=final_confidence,
        reason=reason,
        suggested_algorithms=ALGORITHM_SUGGESTIONS[final_label],
        workflow=WORKFLOWS[final_label],
        model_prediction=model_label,
        model_confidence=model_confidence,
        matched_keyword=matched_keyword,
    )


def print_prediction(result: PredictionResult) -> None:
    """Print a prediction in a clean beginner-friendly format."""

    print("\nPrediction Result")
    print("-----------------")
    print("ML Type:", result.ml_type.title())
    print("Confidence:", f"{result.confidence:.2%}")
    print("Reason:", result.reason)
    print("Algorithms:", ", ".join(result.suggested_algorithms))
    print("5-Step Workflow:")

    for step_number, step in enumerate(result.workflow, start=1):
        print(f"{step_number}. {step}")


def main() -> None:
    """Train, evaluate, and show a sample prediction."""

    model, _X_train, X_test, _y_train, y_test = train_model()
    evaluate_model(model, X_test, y_test)

    sample_text = "Predict customer churn using past data"
    result = predict_ml_type(model, sample_text)
    print_prediction(result)


if __name__ == "__main__":
    main()
