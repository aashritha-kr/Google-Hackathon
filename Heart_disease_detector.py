import streamlit as st
import pytesseract
from PIL import Image
from pdf2image import convert_from_path
import os
import spacy
import re
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier

def extract_text_from_image(image):
    return pytesseract.image_to_string(image)

def extract_text_from_pdf(pdf_path):
    pages = convert_from_path(pdf_path)
    text = ""
    for page in pages:
        text += pytesseract.image_to_string(page) + "\n"
    return text

def main():
    st.title("Heart Disease Prediction")
    st.write("Enter the patient's details to predict the risk of heart disease.")
    option = st.radio("Choose Input Method", ["Upload a File", "Fill the Form"])
    df = pd.read_csv('dataset.csv')
    df=df.dropna()
    df=df.drop_duplicates()
    X, y = df.drop(columns=['target']), df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123)
    scaler = StandardScaler()
    X_train[['age', 'trestbps', 'chol', 'thalach', 'oldpeak']] = scaler.fit_transform(X_train[['age', 'trestbps', 'chol', 'thalach', 'oldpeak']])
    gdb_model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=123).fit(X_train, y_train)
        
    # age = sex = cp = trestbps = chol = fbs = restecg = thalach = exang = oldpeak = slope = ca = thal = None
    
    if option == "Upload a File":
        uploaded_file = st.file_uploader("Upload an Image or PDF", type=["png", "jpg", "jpeg", "pdf"])
        if uploaded_file:
            file_extension = uploaded_file.name.split(".")[-1].lower()
            if file_extension in ["png", "jpg", "jpeg"]:
                image = Image.open(uploaded_file)
                extracted_text = extract_text_from_image(image)
                
            elif file_extension == "pdf":
                pdf_path = f"temp_{uploaded_file.name}"
                with open(pdf_path, "wb") as f:
                    f.write(uploaded_file.read())
                extracted_text = extract_text_from_pdf(pdf_path)
                os.remove(pdf_path)
            
            nlp = spacy.load("en_core_web_sm")
            doc = nlp(extracted_text)
            age_pattern = re.search(r"age\s*(?::|is|are|of)?\s*(\d{1,3})", extracted_text, re.IGNORECASE)
            sex_pattern = re.search(r"(?:sex|gender)\s*(?::|is|are)?\s*([MF])", extracted_text, re.IGNORECASE)
            cp_pattern = re.search(r"chest pain type\s*(?::|is|are)?\s*(\d+)", extracted_text, re.IGNORECASE)
            trestbps_pattern = re.search(r"resting blood pressure\s*(?::|is|are|of)?\s*(\d+/\d+|\d+)", extracted_text, re.IGNORECASE)
            chol_pattern = re.search(r"cholesterol\s*(?:level(?:s)?)?\s*(?::|is|are)?\s*(\d+)\s*(?:mg/dL)?", extracted_text, re.IGNORECASE)
            fbs_pattern = re.search(r"fasting blood sugar\s*(?::|is|are)?\s*(\d+)\s*(?:mg/dL)?", extracted_text, re.IGNORECASE)
            restecg_pattern = re.search(r"resting ecg\s*(?::|is|are)?\s*(\d+)", extracted_text, re.IGNORECASE)
            thalach_pattern = re.search(r"maximum heart rate\s*(?:achieved)?\s*(?::|is|are)?\s*(\d+)", extracted_text, re.IGNORECASE)
            exang_pattern = re.search(r"exercise-induced angina\s*(?::|is|are)?\s*(\d+)", extracted_text, re.IGNORECASE)
            oldpeak_pattern = re.search(r"st depression\s*(?::|is|are)?\s*(\d+(?:\.\d+)?)", extracted_text, re.IGNORECASE)
            slope_pattern = re.search(r"slope\s*(?:of)?\s*(?:the)?\s*(?:peak)?\s*(?:exercise)?\s*st segment\s*(?::|is|are)?\s*(\d+)", extracted_text, re.IGNORECASE)
            ca_pattern = re.search(r"(?:number of|major)?\s*coronary arteries\s*(?:colored)?\s*(?:are|is)?\s*(\d+)", extracted_text, re.IGNORECASE)
            thal_pattern = re.search(r"thal(?:essemia|assemia)?\s*result\s*(?::|is|are)?\s*(\d+)", extracted_text, re.IGNORECASE)

            if age_pattern:
                age=age_pattern.group(1)
            if sex_pattern:
                sex=sex_pattern.group(1)
                if(sex=='M'):
                    sex=1
                elif(sex=='F'):
                    sex=0
            if cp_pattern:
                cp= cp_pattern.group(1)
            if trestbps_pattern:
                trestbps=trestbps_pattern.group(1)
            if chol_pattern:
                chol=chol_pattern.group(1)
            if fbs_pattern:
                fbs = int(fbs_pattern.group(1))
                if fbs>120:
                    fbs=1
                else:
                    fbs=0
            if restecg_pattern:
                restecg=restecg_pattern.group(1)
            if thalach_pattern:
                thalach=thalach_pattern.group(1)
            if exang_pattern:
                exang=exang_pattern.group(1)
            if oldpeak_pattern:
                oldpeak=oldpeak_pattern.group(1) # Extract decimal correctly
            if slope_pattern:
                slope= slope_pattern.group(1)
            if ca_pattern:
                ca=ca_pattern.group(1)
            if thal_pattern:
                thal=thal_pattern.group(1)
    elif option == "Fill the Form":
       
        age = st.number_input("Age", min_value=1, max_value=120, value=50)
        sex = st.radio("Sex", options=[0, 1], format_func=lambda x: "Male" if x == 1 else "Female")
        cp = st.selectbox("Chest Pain Type", options=[0, 1, 2, 3])
        trestbps = st.number_input("Resting Blood Pressure", min_value=50, max_value=200, value=120)
        chol = st.number_input("Cholesterol", min_value=100, max_value=400, value=200)
        fbs = st.radio("Fasting Blood Sugar > 120 mg/dl", options=[0, 1])
        restecg = st.selectbox("Resting ECG", options=[0, 1, 2])
        thalach = st.number_input("Max Heart Rate", min_value=50, max_value=250, value=150)
        exang = st.radio("Exercise-Induced Angina", options=[0, 1])
        oldpeak = st.number_input("ST Depression", min_value=0.0, max_value=6.0, value=1.0)
        slope = st.selectbox("Slope", options=[0, 1, 2])
        ca = st.selectbox("Major Vessels Colored", options=[0, 1, 2, 3, 4])
        thal = st.selectbox("Thalassemia", options=[0, 1, 2, 3])
    
    if st.button("Predict"):
    # and None not in [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]:
        user_input = pd.DataFrame([{ 'age': age, 'sex': sex, 'cp': cp, 'trestbps': trestbps, 'chol': chol, 'fbs': fbs, 'restecg': restecg, 'thalach': thalach, 'exang': exang, 'oldpeak': oldpeak, 'slope': slope, 'ca': ca, 'thal': thal }])
        prediction = gdb_model.predict(user_input)
        st.write(f"Prediction: {'Heart Disease Detected' if prediction[0] == 1 else 'No Heart Disease Detected'}")

if __name__ == "__main__":
    main()
