import pandas as pd
from flask import Flask, render_template, request
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report

app = Flask(__name__)

# Step 1: Load and preprocess the data
def load_and_preprocess_data():
    # Load the dataset
    data = pd.read_csv('Dataset.csv')

    # Drop unwanted columns
    columns_to_exclude = [
        'Over18', 'EmployeeNumber', 'StandardHours', 'DailyRate', 'MonthlyRate',
        'PercentSalaryHike', 'YearsAtCompany', 'YearsWithCurrManager',
        'EnvironmentSatisfaction', 'JobSatisfaction', 'RelationshipSatisfaction',
        'WorkLifeBalance', 'YearsSinceLastPromotion', 'PerformanceRating', 'StockOptionLevel',
        'HourlyRate', 'TrainingTimesLastYear', 'JobInvolvement', 'EducationField'
    ]
    X = data.drop(columns=['Attrition'] + columns_to_exclude)
    y = data['Attrition']

    # Convert the target variable to binary (Yes/No -> 1/0)
    y = y.map({'Yes': 1, 'No': 0})

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

    # Create a pipeline with preprocessing and K-NN classifier
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', KNeighborsClassifier(n_neighbors=5))
    ])

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Fit the pipeline on the training data
    pipeline.fit(X_train, y_train)

    # Evaluate the model
    y_pred = pipeline.predict(X_test)
    print(classification_report(y_test, y_pred))

    return pipeline, X_test

# Train the model and return the pipeline and test data
pipeline, X_test = load_and_preprocess_data()

# Step 2: Define the API routes
@app.route('/')
def home():
    return render_template('Index2.html')

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
        prediction_text = "Yes" if prediction[0] == 1 else "No"

        # Return the result
        return render_template('Index2.html', prediction=prediction_text)

if __name__ == '__main__':
    app.run(debug=True)
