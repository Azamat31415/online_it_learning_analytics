import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

def train_predict_model(df):
    X = df.drop(columns=["user_id", "completion_status"])
    y = df["completion_status"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LogisticRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Model trained. Accuracy: {acc:.2f}")

    joblib.dump(model, "models/trained_model.pkl")
    return model
