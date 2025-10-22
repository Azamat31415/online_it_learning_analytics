import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import joblib

def train_predict_model(df):
    X = df.drop(columns=["user_id", "completion_status"])
    y = df["completion_status"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    models = {
        "Logistic Regression": LogisticRegression(max_iter=500),
        "Random Forest": RandomForestClassifier(n_estimators=120, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42)
    }

    results = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        results[name] = acc
        print(f"\n{name} Accuracy: {acc:.3f}")
        print(classification_report(y_test, y_pred))

    # Save best model
    best_model_name = max(results, key=results.get)
    best_model = models[best_model_name]
    print(f"\nBest model: {best_model_name} (Accuracy: {results[best_model_name]:.3f})")

    joblib.dump(best_model, "models/trained_model.pkl")

    # Plot model comparison
    plt.barh(list(results.keys()), list(results.values()), color='skyblue')
    plt.xlabel("Accuracy")
    plt.title("Model Comparison")
    plt.show()

    return best_model
