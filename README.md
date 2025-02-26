# Google-Hackathon
# Heart Disease Prediction

This project is a **Heart Disease Prediction** application built using **Streamlit**.  
Users can either manually fill out a form with their test results  or upload an image/PDF of their medical report.  
Based on these values, the application predicts the possibility of the patient having heart disease using a trained ML model.  

## Features

- **Upload a File**:  
  - Accepts files with the extensions **.pdf, .png, .jpg, .jpeg**.  
  - Uses **Tesseract OCR** to extract the necessary fields from the file.  

- **Manual Entry**:  
  - Users can fill out a form with patient details.

- **ML Model**:  
  - Utilizes a **Gradient Boosting Classifier** trained on heart disease data.

## Data Fields Required for Prediction

- **Age**  
- **Sex** (Male/Female)  
- **Chest Pain Type**  
- **Resting Blood Pressure**  
- **Cholesterol**  
- **Fasting Blood Sugar**  
- **Resting ECG**  
- **Max Heart Rate Achieved**  
- **Exercise-Induced Angina**  
- **ST Depression**  
- **Slope of the Peak ST Segment**  
- **Number of Major Vessels Colored**  
- **Thalassemia**  

## Installation

1. **Clone the Repository**  
   ```bash
   git clone <Git link>
   cd heart-disease-prediction

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt

3. **Install Tesseract OCR**


