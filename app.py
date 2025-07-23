import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib

# Load and preprocess the dataset
df = pd.read_csv(r"C:\Users\Yamini\Downloads\adult 3.csv")
df['workclass'] = df['workclass'].replace('?', 'Others')
df['occupation'] = df['occupation'].replace('?', 'Others')

# Prepare features and target
x = df[['age', 'educational-num', 'hours-per-week', 'occupation']]
y = df['income']
x = pd.get_dummies(x)

# Scale numerical features
numerical_cols = ['age', 'educational-num', 'hours-per-week']
scaler = StandardScaler()
x[numerical_cols] = scaler.fit_transform(x[numerical_cols])

# Drop NA
df_clean = pd.concat([x, y], axis=1).dropna()
x = df_clean.drop('income', axis=1)
y = df_clean['income']

# Train-test split and train Random Forest
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

# Save the trained model
joblib.dump(rf_model, "best_model_rf.pkl")

# Streamlit app
st.title("Employee Salary Classification App")
st.write("Predict whether an employee's income is <=50K or >50K based on their details.")

# Input fields
age = st.number_input("Age", min_value=0, max_value=100, value=30)
education = st.selectbox("Education", ['11th', 'HS-grad', 'Assoc-acdm', 'Some-college', '10th', 'Prof-school', '7th-8th', 'Masters', 'Bachelors', 'Doctorate', '5th-6th', '9th', '12th', '1st-4th', 'Preschool'])
occupation = st.selectbox("Occupation", ['Machine-op-inspct', 'Farming-fishing', 'Protective-serv', 'Other-service', 'Prof-specialty', 'Craft-repair', 'Adm-clerical', 'Exec-managerial', 'Tech-support', 'Sales', 'Transport-moving', 'Handlers-cleaners', 'Others', 'Priv-house-serv', 'Armed-Forces'])

# Education to educational-num mapping
education_map = {
    'Preschool': 1, '1st-4th': 2, '5th-6th': 3, '7th-8th': 4, '9th': 5,
    '10th': 6, '11th': 7, '12th': 8, 'HS-grad': 9, 'Some-college': 10,
    'Assoc-acdm': 12, 'Assoc-voc': 11, 'Bachelors': 13, 'Masters': 14,
    'Prof-school': 15, 'Doctorate': 16
}

# Prepare input data for prediction
educational_num = education_map[education]
input_data = pd.DataFrame({
    'age': [age],
    'educational-num': [educational_num],
    'hours-per-week': [40]  # Default to 40 hours as in most dataset entries
})
input_data = pd.get_dummies(input_data.assign(occupation=occupation))

# Align input data with training features
missing_cols = set(x.columns) - set(input_data.columns)
for col in missing_cols:
    input_data[col] = 0
input_data = input_data[x.columns]

# Scale input data
input_data[numerical_cols] = scaler.transform(input_data[numerical_cols])

# Prediction
if st.button("Predict Salary"):
    prediction = rf_model.predict(input_data)
    probability = rf_model.predict_proba(input_data)
    st.write(f"Prediction: Income is **{prediction[0]}**")
    st.write(f"Probability: <=50K: {probability[0][0]:.2f}, >50K: {probability[0][1]:.2f}")