import joblib
import pandas as pd

MODEL_PATH = "models/predict_flag_invoice.pkl"
SCALER_PATH = "models/scaler.pkl"

# Must match the exact order used during training
# (see Invoice_lagging/train.py -> FEATURES)
FEATURES = [
    "invoice_quantity",
    "invoice_dollars",
    "Freight",
    "total_item_quantity",
    "total_item_dollars"
]


def load_model(model_path: str = MODEL_PATH):
    with open(model_path, "rb") as f:
        return joblib.load(f)


def load_scaler(path: str = SCALER_PATH):
    with open(path, "rb") as f:
        return joblib.load(f)


def predict_invoice_flag(input_data):
    """
    Predict whether a vendor invoice should be flagged for manual approval.

    Parameters
    ----------
    input_data : dict or pd.DataFrame
        Must contain: invoice_quantity, invoice_dollars, Freight,
        total_item_quantity, total_item_dollars.

    Returns
    -------
    pd.DataFrame
        Original input columns plus a new "Predicted_Flag" column
        (1 = manual review required, 0 = approved).
    """
    model = load_model()
    scaler = load_scaler()

    df = pd.DataFrame(input_data)

    missing = [f for f in FEATURES if f not in df.columns]
    if missing:
        raise ValueError(f"Missing required column(s) for invoice flag prediction: {missing}")

    # Reorder/select columns to exactly match training feature order
    X = scaler.transform(df[FEATURES])

    df["Predicted_Flag"] = model.predict(X)

    return df


if __name__ == "__main__":

    sample_data = {
        "invoice_quantity": [120],
        "invoice_dollars": [18500],
        "Freight": [450],
        "total_item_quantity": [120],
        "total_item_dollars": [18495]
    }

    prediction = predict_invoice_flag(sample_data)

    print(prediction)
