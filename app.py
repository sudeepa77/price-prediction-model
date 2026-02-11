from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load trained model
model = joblib.load("model/price_model.pkl")

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None

    if request.method == "POST":
        input_data = {
            "Crop Type": [request.form["crop_type"]],
            "Quality Grade": [request.form["quality"]],
            "Farmer Region": [request.form["farmer_region"]],
            "Production Cost (per unit)": [float(request.form["production_cost"])],
            "Quantity Available (tons)": [float(request.form["quantity"])],
            "Previous Contract Price (per unit)": [float(request.form["previous_price"])],
            "Market Price (per unit)": [float(request.form["market_price"])],
            "Demand Level": [request.form["demand"]],
            "Supply Level": [request.form["supply"]],
            "Buyer Type": [request.form["buyer_type"]],
            "Buyer Region": [request.form["buyer_region"]],
            "Purchase History": [int(request.form["purchase_history"])],
            "Negotiation Rounds": [int(request.form["negotiation_rounds"])],
            "Payment Terms": [request.form["payment_terms"]],
            "Transport Cost (per unit)": [float(request.form["transport_cost"])],
            "Seasonality": [request.form["seasonality"]],
            "Weather Impact": [request.form["weather"]],
            "Government Policies": [request.form["policy"]],
            "Crop Variety": [request.form["crop_variety"]],
        }

        df = pd.DataFrame(input_data)
        prediction = round(model.predict(df)[0], 2)

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
