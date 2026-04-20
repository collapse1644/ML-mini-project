# ML Paradigm Classifier

This is a beginner-friendly Python project that classifies business problem
descriptions into one of three machine learning paradigms:

- Supervised Learning
- Unsupervised Learning
- Reinforcement Learning

Example input:

```text
Predict customer churn using past data
```

Example output:

```text
ML Type: Supervised
Confidence: 90.00%
Reason: Keyword "predict" detected, and the model agreed.
Algorithms: Logistic Regression, Decision Tree, Random Forest, Support Vector Machine, Gradient Boosting
```

## Project Structure

```text
.
├── app.py
├── data/
│   └── ml_paradigm_dataset.csv
├── ml_paradigm_classifier.py
├── README.md
└── requirements.txt
```

## How The Code Works

1. **Dataset**
   The CSV file in `data/ml_paradigm_dataset.csv` contains two columns:
   `text` and `label`. You can add more rows later to improve the model.

2. **Preprocessing**
   The `clean_text()` function lowercases the text, removes punctuation, and
   removes extra spaces.

3. **Feature Extraction**
   `TfidfVectorizer` converts each sentence into numbers. TF-IDF gives higher
   importance to useful words and lower importance to very common words.

4. **Model**
   Logistic Regression learns patterns from the TF-IDF features and predicts
   one of three labels: `supervised`, `unsupervised`, or `reinforcement`.

5. **Rule-Based Layer**
   Simple keyword rules are checked before producing the final answer:

   - `predict`, `classify` -> supervised
   - `cluster`, `group` -> unsupervised
   - `reward`, `agent` -> reinforcement

   If a keyword is found, the rule layer helps choose the final result and
   explains why.

6. **Enhanced Output**
   The final response includes the ML type, confidence, reason, suggested
   algorithms, and a 5-step workflow.

## Install Dependencies

```bash
pip install -r requirements.txt
```

## Run In The Terminal

```bash
python ml_paradigm_classifier.py
```

This prints:

- accuracy
- confusion matrix
- sample prediction result

## Run The Streamlit App

```bash
streamlit run app.py
```

Then open the local URL shown in your terminal.

If you are using VS Code, do not click **Run Python File** for `app.py`.
That runs the file like a normal Python script and Streamlit will show
`missing ScriptRunContext` warnings. Instead, use one of these options:

- In the terminal, run `python -m streamlit run app.py`
- In VS Code, open **Run and Debug** and choose **Run Streamlit App**
- On Windows, double-click `run_app.bat`

## Deploy On Streamlit Cloud

When deploying, make sure these files are pushed to GitHub:

- `app.py`
- `ml_paradigm_classifier.py`
- `requirements.txt`
- `data/ml_paradigm_dataset.csv`

In Streamlit Cloud:

1. Choose your GitHub repository.
2. Set the main file path to `app.py`.
3. Deploy the app.

If `data/ml_paradigm_dataset.csv` is missing from GitHub, the app will still
run by using the built-in fallback dataset, but pushing the CSV is better
because it keeps the dataset easy to expand.

## Expanding The Dataset

To improve the project, add more examples to `data/ml_paradigm_dataset.csv`.
Keep the same two columns:

```csv
text,label
"Predict customer churn using past data",supervised
```

More examples usually make the model more reliable.
