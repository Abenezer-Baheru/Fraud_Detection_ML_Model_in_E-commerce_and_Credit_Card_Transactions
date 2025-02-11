# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE
import mlflow
import mlflow.sklearn
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Flatten, LSTM, SimpleRNN
from tensorflow.keras.utils import to_categorical

# Load the dataset
data = pd.read_csv('../../src/data/e-commerce_processed_data.csv')

# Feature and Target Separation
X = data.drop(columns=['class'])
y = data['class']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply SMOTE to the training data
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# Initialize and train traditional machine learning models
lr_model = LogisticRegression(max_iter=1000, random_state=42)
dt_model = DecisionTreeClassifier(random_state=42)
rf_model = RandomForestClassifier(random_state=42)
gb_model = GradientBoostingClassifier(random_state=42)
mlp_model = MLPClassifier(random_state=42)

# Train the models
lr_model.fit(X_train_smote, y_train_smote)
dt_model.fit(X_train_smote, y_train_smote)
rf_model.fit(X_train_smote, y_train_smote)
gb_model.fit(X_train_smote, y_train_smote)
mlp_model.fit(X_train_smote, y_train_smote)

# Build the CNN model
cnn_model = Sequential([
    Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(X_train_smote.shape[1], 1)),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(2, activation='softmax')
])

# Compile the CNN model
cnn_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the CNN model
cnn_model.fit(X_train_smote.values.reshape(-1, X_train_smote.shape[1], 1), to_categorical(y_train_smote), epochs=10, batch_size=32, validation_data=(X_test.values.reshape(-1, X_test.shape[1], 1), to_categorical(y_test)))

# Build the RNN model
rnn_model = Sequential([
    SimpleRNN(64, input_shape=(X_train_smote.shape[1], 1), activation='relu'),
    Dense(64, activation='relu'),
    Dense(2, activation='softmax')
])

# Compile the RNN model
rnn_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the RNN model
rnn_model.fit(X_train_smote.values.reshape(-1, X_train_smote.shape[1], 1), to_categorical(y_train_smote), epochs=10, batch_size=32, validation_data=(X_test.values.reshape(-1, X_test.shape[1], 1), to_categorical(y_test)))

# Build the LSTM model
lstm_model = Sequential([
    LSTM(64, input_shape=(X_train_smote.shape[1], 1), activation='relu'),
    Dense(64, activation='relu'),
    Dense(2, activation='softmax')
])

# Compile the LSTM model
lstm_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the LSTM model
lstm_model.fit(X_train_smote.values.reshape(-1, X_train_smote.shape[1], 1), to_categorical(y_train_smote), epochs=10, batch_size=32, validation_data=(X_test.values.reshape(-1, X_test.shape[1], 1), to_categorical(y_test)))

# Predict and evaluate the traditional machine learning models
models = {
    "Logistic Regression": lr_model,
    "Decision Tree": dt_model,
    "Random Forest": rf_model,
    "Gradient Boosting": gb_model,
    "Multi-Layer Perceptron": mlp_model
}

evaluation_results = []

for model_name, model in models.items():
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    evaluation_results.append({
        "Model": model_name,
        "Precision": report["weighted avg"]["precision"],
        "Recall": report["weighted avg"]["recall"],
        "F1-Score": report["weighted avg"]["f1-score"]
    })

# Evaluate the neural network models
cnn_loss, cnn_accuracy = cnn_model.evaluate(X_test.values.reshape(-1, X_test.shape[1], 1), to_categorical(y_test))
rnn_loss, rnn_accuracy = rnn_model.evaluate(X_test.values.reshape(-1, X_test.shape[1], 1), to_categorical(y_test))
lstm_loss, lstm_accuracy = lstm_model.evaluate(X_test.values.reshape(-1, X_test.shape[1], 1), to_categorical(y_test))

# Collect evaluation results for neural networks
nn_evaluation_results = [
    {"Model": "CNN", "Accuracy": cnn_accuracy},
    {"Model": "RNN", "Accuracy": rnn_accuracy},
    {"Model": "LSTM", "Accuracy": lstm_accuracy}
]

# Create DataFrames for evaluation results
evaluation_df = pd.DataFrame(evaluation_results)
nn_evaluation_df = pd.DataFrame(nn_evaluation_results)

# Combine evaluation results from all models
all_evaluation_df = pd.concat([evaluation_df, nn_evaluation_df], ignore_index=True)

# Display the combined evaluation results in tabular form
print("\nEvaluation Results for All Models:")
print(all_evaluation_df.to_string(index=False))

# Save the evaluation results to a CSV file
all_evaluation_df.to_csv('evaluation_results.csv', index=False)

# MLflow logging
for model_name, model in models.items():
    with mlflow.start_run(run_name=model_name):
        mlflow.sklearn.log_model(model, model_name)
        y_pred = model.predict(X_test)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)

# Log neural network models
for model_name, model, acc in zip(["CNN", "RNN", "LSTM"], [cnn_model, rnn_model, lstm_model], [cnn_accuracy, rnn_accuracy, lstm_accuracy]):
    with mlflow.start_run(run_name=model_name):
        mlflow.keras.log_model(model, model_name)
        mlflow.log_metric("accuracy", acc)