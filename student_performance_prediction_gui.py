
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Load the dataset and train the model
@st.cache
def load_data():
    # Assuming the dataset is pre-loaded here (update with your dataset path)
    df = pd.read_csv('dataset.csv')
    return df

def train_model(df):
    # Fill missing values
    df.fillna(method='ffill', inplace=True)

    # Encode categorical features
    label_encoder = LabelEncoder()
    df['encoded_column'] = label_encoder.fit_transform(df['categorical_column'])

    # Split data into features and target variable
    X = df.drop('target_column', axis=1)
    y = df['target_column']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scaling features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Train Random Forest model
    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    return model, scaler, label_encoder

# Function to predict student performance
def predict_performance(model, scaler, label_encoder, user_input):
    user_input = np.array(user_input).reshape(1, -1)
    user_input_scaled = scaler.transform(user_input)
    prediction = model.predict(user_input_scaled)
    
    return label_encoder.inverse_transform(prediction)[0]

# Streamlit Interface
st.title('Student Performance Prediction')

# Input fields
study_time = st.slider('Study Time', min_value=1, max_value=10, value=5)
parental_education = st.selectbox('Parental Education Level', ['None', 'High School', 'Bachelors', 'Masters', 'PhD'])

# Assuming there are more features; for simplicity, just use study_time here
user_input = [study_time, parental_education]

# Load the dataset and model
df = load_data()
model, scaler, label_encoder = train_model(df)

# Predict and display result
if st.button('Predict Performance'):
    result = predict_performance(model, scaler, label_encoder, user_input)
    st.write(f'Predicted Performance: {result}')
