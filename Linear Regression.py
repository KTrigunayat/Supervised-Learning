import pandas as pd
from flask import Flask, render_template, request
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np  # Required for square root calculation

app = Flask(__name__)

# Step 1: Load and preprocess the data
def load_and_preprocess_data():
    # Load the dataset
    data = pd.read_csv('Dataset.csv')

    # Drop unwanted columns
    columns_to_exclude = [
        'Attrition', 'Department', 'Over18', 'BusinessTravel', 'StandardHours',
        'StockOptionLevel', 'JobSatisfaction', 'WorkLifeBalance', 'PerformanceRating',
        'TrainingTimesLastYear', 'HourlyRate', 'JobInvolvement', 'EducationField',
        'YearsSinceLastPromotion', 'YearsInCurrentRole', 'EmployeeNumber', 'MonthlyRate',
        'RelationshipSatisfaction', 'DailyRate', 'EnvironmentSatisfaction',
        'PercentSalaryHike', 'YearsAtCompany', 'YearsWithCurrManager'
    ]
    X = data.drop(columns=['MonthlyIncome'] + columns_to_exclude)
    y = data['MonthlyIncome']

    # Identify categorical and numerical columns
    categorical_cols = X.select_dtypes(include=['object']).columns
    numerical_cols = X.select_dtypes(include=['int64']).columns

    # Define preprocessing for numerical and categorical features
    numerical_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(drop='first', sparse_output=False)

    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ]
    )

    # Create a pipeline with preprocessing and linear regression
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', LinearRegression())
    ])

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Fit the pipeline on the training data
    pipeline.fit(X_train, y_train)

    # Calculate R^2 and RMSE
    y_pred = pipeline.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"R^2 Score: {r2:.4f}")
    print(f"RMSE: {rmse:.4f}")

    return pipeline, X_test, r2, rmse

# Train the model and return the pipeline, test data, R^2 score, and RMSE
pipeline, X_test, r2, rmse = load_and_preprocess_data()

# Step 2: Define the API routes
@app.route('/')
def home():
    return render_template('index.html', r2=round(r2, 4), rmse=round(rmse, 4))  # Pass R^2 and RMSE to the template

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Collect input data from the form
        form_data = {
            'Age': int(request.form['Age']),
            'DistanceFromHome': int(request.form['DistanceFromHome']),
            'Education': int(request.form['Education']),
            'Gender': request.form['Gender'],
            'JobLevel': int(request.form['JobLevel']),
            'JobRole': request.form['JobRole'],
            'MaritalStatus': request.form['MaritalStatus'],
            'NumCompaniesWorked': int(request.form['NumCompaniesWorked']),
            'OverTime': request.form['OverTime'],
            'TotalWorkingYears': int(request.form['TotalWorkingYears'])
        }
        
        # Convert input to DataFrame
        input_df = pd.DataFrame([form_data])

        # Predict using the trained pipeline
        prediction = pipeline.predict(input_df)

        # Return the result
        return render_template(
            'index.html',
            prediction=round(prediction[0], 2),
            r2=round(r2, 4),
            rmse=round(rmse, 4)
        )

if __name__ == '__main__':
    app.run(debug=True)
