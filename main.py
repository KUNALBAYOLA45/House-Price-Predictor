import pandas as pd
import pickle
from flask import Flask, render_template, request, jsonify
import numpy as np

app = Flask(__name__)
data = pd.read_csv('Cleaned_data.csv')
pipe = pickle.load(open("RidgeModel.pkl", 'rb'))

@app.route('/')
def index():
    locations = sorted(data['location'].unique().tolist())
    return render_template('index.html', locations=locations)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        location = data.get('location')
        bhk = float(data.get('bhk'))
        bath = float(data.get('bath'))
        sqft = float(data.get('total_sqft'))

        # Creating DataFrame for prediction
        input_df = pd.DataFrame([[location, sqft, bath, bhk]],
                                columns=['location', 'total_sqft', 'bath', 'bhk'])
        prediction = pipe.predict(input_df)[0] * 100000

        return jsonify({'predicted_price': round(prediction, 2)})

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5001)
