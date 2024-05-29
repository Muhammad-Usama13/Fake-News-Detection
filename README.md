# Fake News Detection using Logistic Regression

## Introduction

This project aims to classify news articles as real or fake using natural language processing (NLP) techniques and a logistic regression model. The dataset contains news articles labeled as real (0) or fake (1).

## Dataset Description

The dataset consists of the following columns:
1. **id**: Unique identifier for a news article.
2. **title**: The title of the news article.
3. **author**: The author of the news article.
4. **text**: The main text of the news article (could be incomplete).
5. **label**: The label indicating whether the news is real (0) or fake (1).

## Workflow

### 1. Importing Dependencies

The project begins by importing necessary libraries for data processing, text preprocessing, model building, and evaluation:

- **numpy** and **pandas** for data manipulation.
- **re** for regular expression operations.
- **nltk** for natural language processing tasks, specifically stopwords.
- **sklearn** for machine learning models and evaluation metrics.

### 2. Data Preprocessing

**Loading the Dataset:**
- The dataset is loaded into a pandas DataFrame from a CSV file.

**Handling Missing Values:**
- Missing values in the dataset are replaced with empty strings to ensure consistency.

**Merging Columns:**
- The author name and the title of the article are combined into a single column called `content`. This provides a more substantial textual input for the model.

**Separating Features and Labels:**
- The dataset is split into features (`X`) and labels (`Y`). Features include all columns except the label, and labels include only the `label` column.

### 3. Stemming

**Stemming Process:**
- Stemming is the process of reducing words to their root form. For example, "actor", "actress", and "acting" are reduced to "act".
- The `PorterStemmer` from NLTK is used for stemming.
- The content of each news article is preprocessed by:
  - Removing non-alphabetical characters.
  - Converting text to lowercase.
  - Splitting the text into words.
  - Removing stopwords (common words that do not contribute much to the meaning, such as "is", "and", "the").
  - Applying stemming to each word.
  - Joining the stemmed words back into a single string.

**Applying Stemming:**
- The stemming function is applied to the `content` column, transforming the text data.

### 4. Converting Text to Numerical Data

**TF-IDF Vectorization:**
- TF-IDF (Term Frequency-Inverse Document Frequency) vectorization converts textual data into numerical data. It measures the importance of each word in the document relative to the entire dataset.
- The `TfidfVectorizer` from `sklearn` is used to fit and transform the text data.

### 5. Splitting the Dataset

**Train-Test Split:**
- The dataset is split into training and testing sets. 80% of the data is used for training, and 20% is used for testing.
- The split is stratified to ensure that the distribution of real and fake news is consistent across both sets.

### 6. Model Training

**Logistic Regression:**
- A logistic regression model is chosen for this classification task.
- The model is trained using the training data.

### 7. Model Evaluation

**Accuracy Score:**
- The accuracy of the model is evaluated on both the training and testing sets.
- Predictions are made on the training and testing data, and accuracy scores are calculated to assess the model's performance.

### 8. Making Predictions

**Predictive System:**
- The trained model is used to make predictions on new data.
- An example prediction is made using one of the test samples. The predicted label is compared to the actual label to verify the prediction.

## Conclusion

This project demonstrates the process of detecting fake news using NLP and machine learning techniques. Key steps include data preprocessing, text vectorization, model training, and evaluation. The logistic regression model provides a baseline approach to the classification task, with potential improvements achievable through more advanced models and feature engineering.
