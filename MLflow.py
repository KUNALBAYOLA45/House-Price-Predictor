import mlflow
import mlflow.sklearn
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge

# Load dataset
data = pd.read_csv("Cleaned_data.csv")

# Preprocessing
X = data[['total_sqft', 'bath', 'bhk', 'location']]
y = data['price']

# One-Hot Encoding for 'location'
X = pd.get_dummies(X, drop_first=True)

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize MLflow
mlflow.set_experiment("House Price Prediction")

with mlflow.start_run():
    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train)

    # Log parameters
    mlflow.log_param("alpha", 1.0)

    # Log model performance
    score = model.score(X_test, y_test)
    mlflow.log_metric("r2_score", score)

    # Log and save model
    mlflow.sklearn.log_model(model, "house_price_model")

    # Save locally
    pickle.dump(model, open("RidgeModel.pkl", "wb"))

    print(f"Model logged with R² score: {score}")
