from flask import Flask, render_template, request, jsonify
import pandas as pd
import pickle

app = Flask(__name__)

# Load dataset
df = pd.read_csv("Cleaned_Car_data.csv")

# Load trained model
model = pickle.load(open("LinearRegressionModel.pkl", "rb"))

@app.route('/')
def home():
    return render_template(
        'index.html',
        companies=sorted(df['company'].unique()),
        years=sorted(df['year'].unique(), reverse=True),
        fuels=sorted(df['fuel_type'].unique()),
        prediction=None
    )

@app.route('/get_models/<company>')
def get_models(company):
    models = df[df['company'] == company]['name'].unique()
    return jsonify(sorted(models))

# @app.route('/predict', methods=['POST'])
# def predict():
#     company = request.form['company']
#     car_model = request.form['car_model']
#     year = int(request.form['year'])
#     fuel = request.form['fuel_type']
#     kms = int(request.form['kilo_driven'])

#     prediction = model.predict([[year, kms, fuel, company, car_model]])[0]

#     return render_template(
#         'index.html',
#         companies=sorted(df['company'].unique()),
#         years=sorted(df['year'].unique(), reverse=True),
#         fuels=sorted(df['fuel_type'].unique()),
#         prediction=round(prediction, 2)
#     )
@app.route('/predict', methods=['POST'])
def predict():
    company = request.form['company']
    car_model = request.form['car_model']
    year = int(request.form['year'])
    fuel = request.form['fuel_type']
    kms = int(request.form['kilo_driven'])

    # 🔑 CREATE DATAFRAME (THIS FIXES THE ERROR)
    input_df = pd.DataFrame(
        [[year, kms, fuel, company, car_model]],
        columns=['year', 'kms_driven', 'fuel_type', 'company', 'name']
    )

    prediction = model.predict(input_df)[0]

    return render_template(
        'index.html',
        companies=sorted(df['company'].unique()),
        years=sorted(df['year'].unique(), reverse=True),
        fuels=sorted(df['fuel_type'].unique()),
        prediction=round(prediction, 2)
    )

if __name__ == "__main__":
    app.run(debug=True)
