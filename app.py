import streamlit as st
import pickle
import numpy as np

# Title of the app
st.title("K-Nearest Neighbors (KNN)")

# Subtitle
st.subheader("Practice using KNN for regression with Salary Prediction and classification tasks with Diabetes Dataset.")

# Sidebar for app navigation
st.sidebar.title("Navigation")
app_mode = st.sidebar.selectbox("Choose the mode", ["Regression with Salary Prediction", "Classification with Diabetes Dataset"])

@st.cache_data
def load_reg_model(path):
    """Load a model and scaler from the given path."""
    with open(f'{path}/reg_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open(f'{path}/label_encoder.pkl', 'rb') as f:
        label_encoder = pickle.load(f)
    with open(f'{path}/onehot_encoder.pkl', 'rb') as f:
        onehot_encoder = pickle.load(f)
    return model, label_encoder, onehot_encoder

@st.cache_data
def load_clf_model(path):
    """Load a model and scaler from the given path."""
    with open(f'{path}/diabetes_model.pkl', 'rb') as f:
        model = pickle.load(f)
    return model

@st.dialog("Predict Salary")
def reg_predict(age, gender, education, experience):
    st.write("Input Parameters:")
    st.write(f"Age: {age}")
    st.write(f"Gender: {gender}")
    st.write(f"Education Level: {education}")
    st.write(f"Years of Experience: {experience}")

    # Load model and scaler
    model, label_encoder, onehot_encoder = load_reg_model('model')

    gender_encoded = label_encoder.transform([gender])[0]
    
    education_encoded = onehot_encoder.transform([[education]]).toarray().flatten()
    print("fsdfsdf", education_encoded)
    # Prepare the input data
    input_data = np.array([[*education_encoded, age, gender_encoded, experience]])
    print("hjkhlkjhgljkhglj", input_data)
    # Predict the salary
    prediction = model.predict(input_data)
    st.write(f"Predicted Salary: ${prediction[0]:,.2f}")

@st.dialog("Predict Diabetic or Non-Diabetic")
def clf_predict(pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree, age):
    st.write("Input Parameters:")
    st.write(f"Pregnancies: {pregnancies}")
    st.write(f"Glucose: {glucose}")
    st.write(f"Blood Pressure: {blood_pressure}")
    st.write(f"Skin Thickness: {skin_thickness}")
    st.write(f"Insulin: {insulin}")
    st.write(f"BMI: {bmi}")
    st.write(f"Diabetes Pedigree Function: {diabetes_pedigree}")
    st.write(f"Age: {age}")

    # Prepare the input data
    input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree, age]])

    # Load model and scaler
    model= load_clf_model('model')

    # Predict diabetes
    prediction = model.predict(input_data)
    result = "Diabetic" if prediction[0] == 1 else "Non-Diabetic"
    st.write(f"Prediction: **{result}**")

if app_mode == "Regression with Salary Prediction":
    st.header("Predict Salary")

    # Feature inputs for Salary Prediction
    age = st.number_input('Age', min_value=18, max_value=100, value=30, step=1)
    gender = st.radio('Gender', options=['Male', 'Female'])
    education_level = st.selectbox('Education Level', options=['Bachelor\'s', 'Master\'s', 'PhD'])
    experience = st.number_input('Years of Experience', min_value=0, max_value=50, value=5, step=1)

    inputs_filled = all([
        age is not None,
        gender is not None,
        education_level is not None,
        experience is not None
    ])

    if st.button('Predict Salary', disabled=not inputs_filled):
        reg_predict(age, gender, education_level, experience)

    if not inputs_filled:
        st.write("Please fill out all the inputs to enable the prediction.")

elif app_mode == "Classification with Diabetes Dataset":
    st.header("Predict Diabetes")

    # Feature inputs for Diabetes Dataset
    pregnancies = st.number_input('Pregnancies', min_value=0, value=1, step=1)
    glucose = st.number_input('Glucose Level', min_value=0, value=120, step=1)
    blood_pressure = st.number_input('Blood Pressure (mm Hg)', min_value=0, value=70, step=1)
    skin_thickness = st.number_input('Skin Thickness (mm)', min_value=0, value=20, step=1)
    insulin = st.number_input('Insulin Level (mu U/ml)', min_value=0.0, value=80.0, step=0.1)
    bmi = st.number_input('BMI', min_value=0.0, value=25.0, step=0.1)
    diabetes_pedigree = st.number_input('Diabetes Pedigree Function', min_value=0.0, value=0.5, step=0.1)
    age = st.number_input('Age', min_value=0, value=30, step=1)

    inputs_filled = all([
        pregnancies is not None,
        glucose is not None,
        blood_pressure is not None,
        skin_thickness is not None,
        insulin is not None,
        bmi is not None,
        diabetes_pedigree is not None,
        age is not None
    ])

    if st.button('Predict Diabetes', disabled=not inputs_filled):
        clf_predict(pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree, age)

    if not inputs_filled:
        st.write("Please fill out all the inputs to enable the prediction.")
