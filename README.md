# Offensive Language Classification

## Project Overview
This project aims to build a machine learning model that detects various types of offensive content in online feedback. The goal is to predict which of the following labels apply to each comment: toxic, abusive, vulgar, menace, offense, and bigotry. The classification is binary, with 1 indicating the presence of the offensive content and 0 indicating its absence.

## Dataset Description
The dataset consists of two main files:

1. **train.csv** (Labeled Training Data):
    - `id`: Unique identifier for each comment
    - `feedback_text`: The feedback to be classified
    - `toxic`: 1 if the comment is toxic
    - `abusive`: 1 if the comment contains severe toxicity
    - `vulgar`: 1 if the comment contains obscene language
    - `menace`: 1 if the comment contains threats
    - `offense`: 1 if the comment contains insults
    - `bigotry`: 1 if the comment contains identity-based hate

2. **test.csv** (Unlabeled data for prediction):
    - The same structure as `train.csv` without labels for prediction.

The dataset allows multiple labels to be active for a single comment.

## Model Implementation Details

### Step 1: Exploratory Data Analysis (EDA)
- Visualize the label distribution across all toxicity types.
- Analyze sentence structure such as length, word distribution, and common words.
- Check for missing values or outliers.

### Step 2: Text Preprocessing
- Tokenization: Split sentences into words.
- Lowercasing: Convert text to lowercase.
- Remove stop words, special characters, and punctuation.
- Stemming/Lemmatization: Normalize words to their root form.
- Feature Extraction: Convert text into numeric representations using techniques like TF-IDF, Word2Vec, or Transformer embeddings.

### Step 3: Model Creation
1. **Baseline Model**: Logistic Regression or Random Forest.
2. **Advanced Models**: LSTM or GRU for sequential nature of text.
3. **Transformer-Based Models**: Fine-tune BERT or XLM for the task.

### Step 4: Model Evaluation
- Compute metrics such as accuracy, precision, recall, and F1-score.
- Plot a Confusion Matrix to analyze misclassifications.
- Generate an AUC-ROC curve for evaluation.

### Step 5: Model Tuning and Optimization
- Experiment with different optimizers (Adam, SGD, etc.) and activation functions.
- Adjust learning rate, batch size, and the number of epochs.
- Use Grid Search or Random Search for hyperparameter tuning.

## Steps to Run the Code

1. Clone the repository.
2. Install the required dependencies from the `requirements.txt` file.
   ```bash
   pip install -r requirements.txt
