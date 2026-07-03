import joblib
import pandas as pd

MODEL_PATH = "models/predict_freight_model.pkl"

# The model was trained ONLY on "Dollars" (see Freight_Cost_Prediction/data_preprocessing.py
# -> prepare_features()). Any extra columns passed in (e.g. Quantity) must be dropped
# before calling model.predict(), otherwise sklearn raises a feature-mismatch error.
FEATURES = ["Dollars"]


def load_model(model_path: str = MODEL_PATH):
    """
    Load trained freight cost prediction model.
    """
    with open(model_path, "rb") as f:
        model = joblib.load(f)

    return model


def predict_freight_cost(input_data):
    """
    Predict freight cost for new vendor invoices.

    Parameters
    ----------
    input_data : dict or pd.DataFrame
        Must contain a "Dollars" column. Any other columns (e.g. "Quantity")
        are kept in the returned DataFrame for display but are NOT sent to
        the model, since it was only trained on "Dollars".

    Returns
    -------
    pd.DataFrame
        Original input columns plus a new "Predicted_Freight" column.
    """
    model = load_model()

    input_df = pd.DataFrame(input_data)

    missing = [f for f in FEATURES if f not in input_df.columns]
    if missing:
        raise ValueError(f"Missing required column(s) for freight prediction: {missing}")

    input_df["Predicted_Freight"] = model.predict(input_df[FEATURES]).round()

    return input_df


if __name__ == "__main__":

    # Example inference run (local testing)
    sample_data = {
        "Dollars": [18500, 9000]
    }

    prediction = predict_freight_cost(sample_data)

    print(prediction)
