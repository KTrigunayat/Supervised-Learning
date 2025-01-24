import pandas as pd
from flask import Flask, render_template, request
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import numpy as np

app = Flask(__name__)

# Step 1: Load and preprocess the data
def load_and_preprocess_data_logistic_regression():
    # Load the dataset
    data = pd.read_csv('Dataset.csv')

    # Define target variable and features
    y = data['Attrition'].map({'Yes': 1, 'No': 0})  # Convert Attrition to binary
    columns_to_exclude = [
        'Attrition', 'Over18', 'StandardHours', 'EmployeeNumber', 'DailyRate', 'HourlyRate',
        'MonthlyRate', 'JobSatisfaction', 'EnvironmentSatisfaction', 'RelationshipSatisfaction',
        'EducationField', 'StockOptionLevel', 'WorkLifeBalance', 'Department',
        'PercentSalaryHike', 'TrainingTimesLastYear', 'BusinessTravel', 'JobInvolvement'
    ]
    X = data.drop(columns=columns_to_exclude)

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

    # Define the Logistic Regression model
    model = LogisticRegression(
        random_state=42,
        max_iter=1000,  # Ensure convergence
        solver='liblinear'  # Suitable for smaller datasets and binary classification
    )

    # Create the pipeline
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', model)
    ])

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the pipeline
    pipeline.fit(X_train, y_train)

    return pipeline, X_test, y_test, y_train

# Train the model and get test data
pipeline, X_test, y_test, y_train = load_and_preprocess_data_logistic_regression()

# Step 2: Define the API routes
@app.route('/')
def home():
    return render_template('Index3.html')  # Input form for employee data

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
                'YearsSinceLastPromotion': int(request.form.get('YearsSinceLastPromotion', 0))
            }

            # Convert input to DataFrame
            input_df = pd.DataFrame([form_data])

            # Predict using the trained pipeline
            prediction = pipeline.predict(input_df)
            prediction_text = 'Yes' if prediction[0] == 1 else 'No'

            # Calculate evaluation metrics for the model
            y_pred = pipeline.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)

            # Return the result to the template
            return render_template(
                'Index3.html',
                prediction=prediction_text,
                accuracy=round(accuracy, 4),
                precision=round(precision, 4),
                recall=round(recall, 4),
                f1=round(f1, 4)
            )

        except Exception as e:
            # Handle exceptions and return error message
            return render_template('Index3.html', error=str(e))

if __name__ == '__main__':
    app.run(debug=True)
