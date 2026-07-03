import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from inference.predict_freight import predict_freight_cost
from inference.predict_invoice_flag import predict_invoice_flag


# --------------------------------------------------------
# Page Config
# --------------------------------------------------------

st.set_page_config(
    page_title="Vendor Invoice Intelligence Portal",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------------------------------------
# CSS
# --------------------------------------------------------

st.markdown("""
<style>

.main{
    background:#F8FAFC;
}

.big-title{
    font-size:42px;
    font-weight:700;
    color:#2563EB;
}

.sub-title{
    color:#475569;
    font-size:18px;
    margin-bottom: 10px;
}

.section-header{
    font-size:26px;
    font-weight:700;
    color:#1E293B;
}

.metric-card{
    background:white;
    padding:20px;
    border-radius:12px;
    box-shadow:0px 2px 8px rgba(0,0,0,.1);
}

.safe{
    color:green;
    font-size:28px;
    font-weight:bold;
}

.risk{
    color:red;
    font-size:28px;
    font-weight:bold;
}

.impact-item{
    font-size:15px;
    color:#334155;
    margin-bottom:6px;
}

</style>
""", unsafe_allow_html=True)

# --------------------------------------------------------
# Sidebar - Model Selection
# --------------------------------------------------------

st.sidebar.markdown("### 🔍 Model Selection")

module = st.sidebar.radio(
    "Choose Prediction Module",
    [
        "Freight Cost Prediction",
        "Invoice Manual Approval Flag"
    ]
)

st.sidebar.divider()

st.sidebar.markdown("### Business Impact")

st.sidebar.markdown(
    "<div class='impact-item'>📉 Improved cost forecasting</div>"
    "<div class='impact-item'>🚩 Reduced invoice fraud & anomalies</div>"
    "<div class='impact-item'>⚙️ Faster finance operations</div>",
    unsafe_allow_html=True
)

# --------------------------------------------------------
# Hero Section
# --------------------------------------------------------

st.markdown(
    "<div class='big-title'>📦 Vendor Invoice Intelligence Portal</div>",
    unsafe_allow_html=True
)

st.markdown(
    "<div class='sub-title'>AI-Driven Freight Cost Prediction & Invoice Risk Flagging</div>",
    unsafe_allow_html=True
)

st.write("This internal analytics portal leverages machine learning to")

st.markdown(
    """
- **Forecast freight costs accurately**
- **Detect risky or abnormal vendor invoices**
- **Reduce financial leakage and manual workload**
    """
)

st.divider()

# --------------------------------------------------------
# Freight Cost Prediction Module
# --------------------------------------------------------

if module == "Freight Cost Prediction":

    st.markdown(
        "<div class='section-header'>📦 Freight Cost Prediction</div>",
        unsafe_allow_html=True
    )

    st.write("**Objective:**")
    st.write(
        "Predict the expected freight cost for a vendor invoice based on "
        "its invoice dollar amount, using a trained regression model."
    )

    st.write("")

    with st.form("freight_form"):

        c1, c2 = st.columns(2)

        with c1:
            dollars = st.number_input(
                "Invoice Dollars",
                value=18500.0,
                min_value=0.0
            )

        with c2:
            quantity = st.number_input(
                "Quantity",
                value=120,
                min_value=1
            )

        predict = st.form_submit_button("Predict Freight")

    if predict:

        # Quantity is captured for context/history but the trained model
        # only uses "Dollars" as its feature (see inference/predict_freight.py)
        input_data = {
            "Dollars": [dollars],
            "Quantity": [quantity]
        }

        result = predict_freight_cost(input_data)

        freight = result["Predicted_Freight"].iloc[0]

        st.success("Prediction Completed")

        st.metric("Estimated Freight", f"${freight:,.2f}")

        history = pd.DataFrame({
            "Feature": ["Invoice Dollars", "Quantity"],
            "Value": [dollars, quantity]
        })

        fig = px.bar(
            history,
            x="Feature",
            y="Value",
            color="Feature",
            title="Input Overview"
        )

        st.plotly_chart(fig, use_container_width=True)

        if "history" not in st.session_state:
            st.session_state.history = []

        st.session_state.history.append({
            "Module": "Freight",
            "Prediction": float(freight)
        })

# --------------------------------------------------------
# Invoice Manual Approval Flag Module
# --------------------------------------------------------

elif module == "Invoice Manual Approval Flag":

    st.markdown(
        "<div class='section-header'>🚩 Invoice Manual Approval Prediction</div>",
        unsafe_allow_html=True
    )

    st.write("**Objective:**")
    st.write(
        "Predict whether a vendor invoice should be flagged for manual "
        "approval based on abnormal cost, freight, or delivery patterns."
    )

    st.write("")

    with st.form("invoice_form"):

        col1, col2 = st.columns(2)

        with col1:
            invoice_quantity = st.number_input(
                "Invoice Quantity",
                min_value=1,
                value=120
            )

            invoice_dollars = st.number_input(
                "Invoice Dollars",
                min_value=0.0,
                value=18500.0
            )

            freight = st.number_input(
                "Freight",
                min_value=0.0,
                value=450.0
            )

        with col2:
            total_item_quantity = st.number_input(
                "Total Item Quantity",
                min_value=1,
                value=120
            )

            total_item_dollars = st.number_input(
                "Total Item Dollars",
                min_value=0.0,
                value=18495.0
            )

        submit = st.form_submit_button("Evaluate Invoice")

    if submit:

        sample = {
            "invoice_quantity": [invoice_quantity],
            "invoice_dollars": [invoice_dollars],
            "Freight": [freight],
            "total_item_quantity": [total_item_quantity],
            "total_item_dollars": [total_item_dollars]
        }

        result = predict_invoice_flag(sample)

        prediction = int(result["Predicted_Flag"].iloc[0])

        st.divider()

        if prediction == 1:

            st.error("Manual Review Required")

            st.metric("Invoice Status", "HIGH RISK")

            reasons = pd.DataFrame({
                "Reason": [
                    "Invoice Mismatch",
                    "Large Freight",
                    "Receiving Delay"
                ],
                "Score": [85, 78, 91]
            })

            fig = px.bar(
                reasons,
                x="Reason",
                y="Score",
                color="Score",
                title="Risk Factors"
            )

            st.plotly_chart(fig, use_container_width=True)

        else:

            st.success("Invoice Approved")

            st.metric("Invoice Status", "SAFE")

            st.balloons()

        if "history" not in st.session_state:
            st.session_state.history = []

        st.session_state.history.append({
            "Module": "Invoice",
            "Prediction": "Manual Review" if prediction == 1 else "Approved"
        })

# --------------------------------------------------------
# Sidebar - Recent Predictions
# --------------------------------------------------------

if "history" not in st.session_state:
    st.session_state.history = []

st.sidebar.divider()
st.sidebar.subheader("Recent Predictions")

if len(st.session_state.history) == 0:
    st.sidebar.write("No predictions yet.")
else:
    history_df = pd.DataFrame(st.session_state.history)
    st.sidebar.dataframe(history_df.tail(5), use_container_width=True)

# --------------------------------------------------------
# Footer
# --------------------------------------------------------

st.divider()

st.markdown(
    """
<center>

### Vendor Invoice Intelligence Portal

Machine Learning Powered Procurement Analytics

Built using

Python • Streamlit • Scikit-Learn • SQLite • Plotly

</center>
""",
    unsafe_allow_html=True
)
