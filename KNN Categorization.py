import pandas as pd
from flask import Flask, render_template, request
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

app = Flask(__name__)

# Step 1: Load and preprocess the data
def load_and_preprocess_data_knn_regression():
    # Load the dataset
    data = pd.read_csv('Dataset.csv')

    # Define target variable and features
    y = data['PercentSalaryHike']
    columns_to_exclude = [
        'PercentSalaryHike', 'Attrition', 'Over18', 'StandardHours', 'EmployeeNumber',
        'DailyRate', 'HourlyRate', 'JobSatisfaction', 'EnvironmentSatisfaction', 'JobInvolvement',
        'RelationshipSatisfaction', 'StockOptionLevel', 'TrainingTimesLastYear', 'WorkLifeBalance','BusinessTravel','MonthlyRate'
    ]
    X = data.drop(columns=columns_to_exclude)
#  columns are missing: {'BusinessTravel', 'Department', 'EducationField', 'MonthlyRate'}
    # Identify categorical and numerical columns
    categorical_cols = X.select_dtypes(include=['object']).columns
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns

    # Preprocessing for numerical and categorical data
    numerical_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(drop='first', sparse_output=False)

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ]
    )

    # Create the pipeline
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', KNeighborsRegressor(10))
    ])

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the pipeline
    pipeline.fit(X_train, y_train)

    return pipeline, X_test, y_test

# Train the model and get test data
pipeline, X_test, y_test = load_and_preprocess_data_knn_regression()

# Step 2: Define the API routes
@app.route('/')
def home():
    return render_template('Index2.html')  # Input form for employee data

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            # Collect input data from the form
            form_data = {
                'Age': int(request.form.get('Age', 0)),
                'DistanceFromHome': int(request.form.get('DistanceFromHome', 0)),
                'Education': int(request.form.get('Education', 0)),
                'Gender': request.form.get('Gender', 'Unknown'),
                'JobLevel': int(request.form.get('JobLevel', 0)),
                'JobRole': request.form.get('JobRole', 'Unknown'),
                'MaritalStatus': request.form.get('MaritalStatus', 'Unknown'),
                'NumCompaniesWorked': int(request.form.get('NumCompaniesWorked', 0)),
                'OverTime': request.form.get('OverTime', 'No'),
                'TotalWorkingYears': int(request.form.get('TotalWorkingYears', 0)),
                'YearsAtCompany': int(request.form.get('YearsAtCompany', 0)),
                'YearsWithCurrManager': int(request.form.get('YearsWithCurrManager', 0)),
                'MonthlyIncome': float(request.form.get('MonthlyIncome', 0.0)),
                'YearsInCurrentRole': int(request.form.get('YearsInCurrentRole', 0)),
                'PerformanceRating': int(request.form.get('PerformanceRating', 0)),
                'YearsSinceLastPromotion': int(request.form.get('YearsSinceLastPromotion', 0)),
                'Department': request.form.get('Department',0),
                'EducationField': request.form.get('EducationField',0)            }

            # Convert input to DataFrame
            input_df = pd.DataFrame([form_data])

            # Predict using the trained pipeline
            prediction = pipeline.predict(input_df)

            # Calculate R² and RMSE for test data
            y_pred = pipeline.predict(X_test)
            r2 = r2_score(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)  # Calculate MSE
            rmse = np.sqrt(mse)  # Take the square root to get RMSE

            # Return the result to the template
            return render_template('Index2.html', prediction=round(prediction[0], 2), r2=round(r2, 4), rmse=round(rmse, 2))

        except Exception as e:
            # Handle exceptions and return error message
            return render_template('Index2.html', error=str(e))


if __name__ == '__main__':
    app.run(debug=True)
