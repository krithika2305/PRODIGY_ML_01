from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

app = Flask(__name__)  

# Step 1: Load trained model and test data using pickle
with open('model.pkl', 'rb') as f:
    model = joblib.load(f)

with open('test_data.pkl', 'rb') as f:
    X_test, y_test = joblib.load(f)  # Assumes test_data.pkl was saved as (X_test, y_test)

@app.route('/')
def home():
    return render_template('homepage.html')  # input form page

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Step 2: Collect inputs from the form
        gr_liv_area = float(request.form['GrLivArea'])
        bedroom_abv_gr = int(request.form['BedroomAbvGr'])
        full_bath = int(request.form['FullBath'])

        # Step 3: Convert to DataFrame
        input_df = pd.DataFrame([[gr_liv_area, bedroom_abv_gr, full_bath]],
                                columns=['GrLivArea', 'BedroomAbvGr', 'FullBath'])

        # Step 4: Predict
        predicted_price = model.predict(input_df)[0]

        # Step 5: Evaluate
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)

        # Step 6: Return result
        return render_template('result.html',
                               prediction=round(predicted_price, 2),
                               mae=round(mae, 2),
                               mse=round(mse, 2),
                               rmse=round(rmse, 2),
                               r2=round(r2, 4))

    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == '__main__': 
    app.run(debug=True)
