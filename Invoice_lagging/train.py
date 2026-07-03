from pathlib import Path
import joblib

from data_preprocessing import (
    load_invoice_data,
    split_data,
    scale_features,
    apply_labels
)

from modeling_evaluation import (
    train_random_forest,
    evaluate_classifier
)

FEATURES = [
    "invoice_quantity",
    "invoice_dollars",
    "Freight",
    "total_item_quantity",
    "total_item_dollars"
]

TARGET = "flag_invoice"


def main():

    BASE_DIR = Path(__file__).resolve().parent.parent

    DB_PATH = BASE_DIR / "data" / "inventory.db"

    MODEL_DIR = BASE_DIR / "models"
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    df = load_invoice_data(DB_PATH)
    df = apply_labels(df)

    # Split data
    X_train, X_test, y_train, y_test = split_data(
        df,
        FEATURES,
        TARGET
    )

    # Scale features
    X_train_scaled, X_test_scaled = scale_features(
        X_train,
        X_test,
        MODEL_DIR / "scaler.pkl"
    )

    # Train model
    grid_search = train_random_forest(
        X_train_scaled,
        y_train
    )

    # Evaluate
    evaluate_classifier(
        grid_search.best_estimator_,
        X_test_scaled,
        y_test,
        "Random Forest Classifier"
    )

    # Save model
    joblib.dump(
        grid_search.best_estimator_,
        MODEL_DIR / "predict_flag_invoice.pkl"
    )

    print("\nModel saved successfully!")
    print(MODEL_DIR / "predict_flag_invoice.pkl")


if __name__ == "__main__":
    main()