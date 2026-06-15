

from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib
import os
import threading
import webbrowser
 
app = Flask(__name__)

USD_TO_INR = 83.5
 
MODEL_PATH  = "house_price_model.pkl"
SCALER_PATH = "scaler.pkl"
 
if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
    raise FileNotFoundError(f"Model files not found!\n" 
        "Please run these first:\n"
         "  python processing.py\n"
    "  python data_model_training.py"
    )
 
House_model  = joblib.load(MODEL_PATH)
price_scale = joblib.load(SCALER_PATH)
 
House_features = ["GrLivArea", "BedroomAbvGr", "TotalBath"]

@app.route("/")
def home():
    return render_template("home_main.html")
 
@app.route("/predict", methods=["POST"])
def predict():
    try:
        sqft = float(request.form["sqft"])
        bedrooms  = int(request.form["bedrooms"])
        full_bath = int(request.form["full_bath"])
        half_bath = int(request.form.get("half_bath", 0))
 
        if sqft <= 0 or sqft > 15000:
            return jsonify({"error": "Square footage must be between 1 and 15,000"}), 400
        if not (0 <= bedrooms <= 10):
            return jsonify({"error": "Bedrooms must be between 0 and 10"}), 400
 
        total_bath = full_bath + (0.5 * half_bath)
 
        house_measure_input     = pd.DataFrame([[sqft, bedrooms, total_bath]], columns=House_features)
        scale_input = price_scale.transform(house_measure_input)
 
        price_usd = float(House_model.predict(scale_input)[0])
        price_usd = max(price_usd, 0)

        price_inr      = price_usd * USD_TO_INR
        range_low_inr  = price_inr * 0.90
        range_high_inr = price_inr * 1.10
 
        return jsonify({
            "price"      : round(price_inr),
            "range_low"  : round(range_low_inr),
            "range_high" : round(range_high_inr),
        })
 
    except KeyError as e:
        return jsonify({"error": f"Missing field: {e}"}), 400
    except ValueError as e:
        return jsonify({"error": f"Invalid value: {e}"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500
 
 
if __name__ == "__main__":
    print(f"=" * 50)
    print(f"House Price Predictor — Flask Server")
    print(f"Opening browser at: http://127.0.0.1:5000")
    print(f"=" * 50)

    print("House Price Predictor Server Started")
    threading.Timer(1.0, lambda: webbrowser.open("http://127.0.0.1:5000")).start()
    app.run(debug=False)
