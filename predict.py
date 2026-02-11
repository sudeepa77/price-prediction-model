import pandas as pd
import joblib

# Load trained pipeline
model = joblib.load("model/price_model.pkl")

# ✅ SAMPLE INPUT — MATCHING CSV DATA TYPES
sample_input = {
    "Crop Type": ["Maize"],
    "Quality Grade": ["A"],
    "Farmer Region": ["Tamil Nadu"],
    "Production Cost (per unit)": [120.5],
    "Quantity Available (tons)": [25.0],
    "Previous Contract Price (per unit)": [180.0],
    "Market Price (per unit)": [195.0],
    "Demand Level": ["High"],
    "Supply Level": ["Medium"],
    "Buyer Type": ["Wholesaler"],
    "Buyer Region": ["Karnataka"],
    "Purchase History": [4],              # ✅ NUMERIC (NOT "Frequent")
    "Negotiation Rounds": [3],
    "Payment Terms": ["Immediate"],
    "Transport Cost (per unit)": [12.5],
    "Seasonality": ["Kharif"],
    "Weather Impact": ["Normal"],
    "Government Policies": ["Subsidy"],
    "Crop Variety": ["Flint Corn"]
}

input_df = pd.DataFrame(sample_input)

prediction = model.predict(input_df)

print("✅ Predicted Final Negotiated Price (per unit):", round(prediction[0], 2))
