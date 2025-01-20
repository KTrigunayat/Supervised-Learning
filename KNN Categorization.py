import pandas as pd
from flask import Flask, render_template, request
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score

app = Flask(__name__)

# Step 1: Load and preprocess the data
def load_and_preprocess_data_knn_regression():
    # Load the dataset
    data = pd.read_csv('Dataset.csv')

    # Define target variable and features
    y = data['PercentSalaryHike']
    columns_to_exclude = [
        'PercentSalaryHike', 'Attrition', 'Over18', 'StandardHours', 'EmployeeNumber',
        'DailyRate', 'EnvironmentSatisfaction', 'RelationshipSatisfaction', 'StockOptionLevel',
        'MonthlyRate', 'BusinessTravel', 'TrainingTimesLastYear', 'Department', 'JobInvolvement',
        'HourlyRate', 'JobSatisfaction', 'YearsSinceLastPromotion', 'EducationField', 'WorkLifeBalance'
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

    # Create the pipeline
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', KNeighborsRegressor(n_neighbors=5))  # Default K=5
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
            'TotalWorkingYears': int(request.form['TotalWorkingYears']),
            'YearsAtCompany': int(request.form['YearsAtCompany']),
            'YearsWithCurrManager': int(request.form['YearsWithCurrManager']),
            'MonthlyIncome': float(request.form['MonthlyIncome']),
            'YearsInCurrentRole': int(request.form['YearsInCurrentRole']),
            'PerformanceRating': int(request.form['PerformanceRating']),
        }

        # Convert input to DataFrame
        input_df = pd.DataFrame([form_data])

        # Predict using the trained pipeline
        prediction = pipeline.predict(input_df)

        # Return the result
        return render_template('Index2.html', prediction=round(prediction[0], 2))

if __name__ == '__main__':
    app.run(debug=True)
