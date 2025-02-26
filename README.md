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
- **Chest Pain Type** (0-Typical Angina 1-Atypical Angina 2-Non-Anginal Pain 3-Asymptomatic)
- **Resting Blood Pressure**  
- **Cholesterol**  
- **Fasting Blood Sugar**  
- **Resting ECG** (0 = Normal, 1 = ST-T wave abnormality, 2 = Left ventricular hypertrophy)
- **Max Heart Rate Achieved**  
- **Exercise-Induced Angina**(0 = No, 1 = Yes)  
- **ST Depression**  
- **Slope of the Peak ST Segment** (0 = Upsloping, 1 = Flat, 2 = Downsloping)
- **Number of Major Vessels Colored**  
- **Thalassemia**  (0 = Normal, 1 = Fixed Defect, 2 = Reversible Defect)

## Installation

1. **Clone the Repository**  
   ```bash
   git clone https://github.com/aashritha-kr/Google-Hackathon.git

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt

3. **Install Tesseract OCR**


