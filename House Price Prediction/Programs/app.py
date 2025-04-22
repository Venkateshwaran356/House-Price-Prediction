from flask import Flask, request, jsonify, render_template_string
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# Load trained model and scaler
model = load_model("house_price_model.h5")
scaler = joblib.load("scaler.pkl")

# Flask app
app = Flask(__name__)

# HTML Template (UI)
html_template = '''
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>🏠 House Price Predictor</title>
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
  <style>
    body { background-color: #f8f9fa; }
    .container { max-width: 600px; margin-top: 50px; }
    h2 { margin-bottom: 30px; }
    .result { margin-top: 20px; font-weight: bold; font-size: 1.2rem; }
  </style>
</head>
<body>
  <div class="container">
    <h2 class="text-center">🏠 House Price Predictor</h2>

    <form id="predictionForm">
      <div id="inputs" class="row g-3">
        {% for label in labels %}
        <div class="col-md-6">
          <label class="form-label">{{ label }}</label>
          <input type="number" class="form-control" name="feature" required>
        </div>
        {% endfor %}
      </div>

      <button type="submit" class="btn btn-primary mt-4 w-100">Predict Price</button>
    </form>

    <div id="result" class="result text-center mt-4"></div>
  </div>

  <script>
    const form = document.getElementById('predictionForm');
    const resultDiv = document.getElementById('result');

    form.addEventListener('submit', function (e) {
      e.preventDefault();
      const inputs = document.getElementsByName('feature');
      const values = Array.from(inputs).map(input => Number(input.value));

      fetch('/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ data: values })
      })
      .then(response => response.json())
      .then(data => {
        if (data.predicted_price) {
          resultDiv.innerHTML = `💰 Estimated Price: ₹ ${data.predicted_price.toFixed(2)}`;
        } else {
          resultDiv.innerHTML = `❌ Error: ${data.error}`;
        }
      })
      .catch(() => {
        resultDiv.innerHTML = `⚠️ Server Error.`;
      });
    });
  </script>
</body>
</html>
'''

# Labels for the form fields
feature_labels = [
    'Area (sqft)', 'Bedrooms', 'Bathrooms', 'Stories',
    'Main Road (1=Yes, 0=No)', 'Guest Room (1/0)',
    'Basement (1/0)', 'Hot Water Heating (1/0)',
    'Air Conditioning (1/0)', 'Parking',
    'Preferred Area (1/0)', 'Furnishing Status (Encoded)'
]

@app.route('/')
def index():
    return render_template_string(html_template, labels=feature_labels)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json['data']
        input_array = np.array(data).reshape(1, -1)
        input_scaled = scaler.transform(input_array)
        prediction = model.predict(input_scaled)
        return jsonify({'predicted_price': float(prediction[0][0])})
    except Exception as e:
        return jsonify({'error': str(e)})

# Run the app
if __name__ == '__main__':
    app.run(debug=False)
