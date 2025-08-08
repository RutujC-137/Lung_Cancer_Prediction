import streamlit as st
from datetime import datetime
import numpy as np
import joblib
import pandas as pd

# Set the page layout to wide
st.set_page_config(layout="wide")

min_date = datetime(1900, 1, 1)
max_date = datetime.now()

st.markdown(
    """
    <style>
    label {
        color: #c07a7a !important;
        font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.sidebar.title("Info")
st.sidebar.write("Group No. 5 PG-DBDA project")
st.sidebar.write("[Tableau Visualizations](https://public.tableau.com/app/profile/aditya.gawande5304/viz/LungCancerUsingPredictiveModelingandAnalysis/LungCancerUsingPredictiveModelingandAnalysis?publish=yes)")

st.markdown("<h1 style='color: #c07a7a;'>Lung Cancer Using Predictive Modeling and Analysis</h1>", unsafe_allow_html=True)

# --- 1. Load the ML Model ---
model_path = 'best_lung_cancer_model.pkl'
@st.cache_resource
def load_model(path):
    try:
        return joblib.load(path)
    except FileNotFoundError:
        st.error(f'Error: "{path}" file not found. Please make sure the model file is in the same directory as this script.')
        st.stop()
    except Exception as e:
        st.error(f'Error loading model: {e}')
        st.stop()

model = load_model(model_path)


# User Form
with st.form(key="user_info_form"):
    col1, col2, col3 = st.columns(3)
    with col1:
        name = st.text_input("Name : ", placeholder="Enter your name here")
        age = st.text_input("Age : ", placeholder="Enter your age here")
        bmi = st.text_input("BMI : ", placeholder="Enter your BMI here")
        cholesterol = st.text_input("Cholesterol level : ", placeholder="Enter cholestrol level")
        # Naya input add kiya hai country ke liye
        country = st.selectbox("Country :", ['Portugal','Croatia','Italy'])
        gender = st.selectbox("Gender", ["Male", "Female"])
        family_history = st.selectbox("Family History :", ["Yes", "No"])
    
    with col2:
        smoking_status = st.selectbox("Smoking Status :", ['Never Smoked', 'Former Smoker', 'Passive Smoker', 'Current Smoker'])
        treatment_type = st.selectbox("Treatment Type :", ['Surgery', 'Radiation', 'Chemotherapy', 'Combined'])
        diagnosis_date = st.date_input("Enter Date of Diagnosis", value=datetime.now(), min_value=min_date, max_value=max_date)
        beginning_of_treatment_date = st.date_input("Beginning of treatment date : ", value=datetime.now(), min_value=min_date, max_value=max_date)
        end_treatment_date = st.date_input("End of treatment date : ", value=datetime.now(), min_value=min_date, max_value=max_date)
        cancer_stage = st.selectbox("Cancer Stage", ["I", "II", "III", "IV"])

    with col3:
        hypertension = st.radio("Hypertension", ["No","Yes"])
        st.markdown("<br>", unsafe_allow_html=True)
        asthma = st.radio("Asthma", ["No","Yes"])
        st.markdown("<br>", unsafe_allow_html=True)
        cirrhosis = st.radio("Cirrhosis", ["No","Yes"])
        st.markdown("<br>", unsafe_allow_html=True)
        other_cancer = st.radio("Other Cancer", ["No","Yes"])
        
    submit_button = st.form_submit_button("Submit")

if submit_button:
    # --- Data Preprocessing (CRUCIAL STEP - FINAL VERSION) ---
    try:
        if not age or not bmi or not cholesterol:
            st.error("Error: Please fill in all the required fields (Age, BMI, and Cholesterol).")
            st.stop()
            
        age_val = float(age)
        bmi_val = float(bmi)
        cholesterol_val = float(cholesterol)
        
        treatment_delay_days = (beginning_of_treatment_date - diagnosis_date).days
        treatment_duration_days = (end_treatment_date - beginning_of_treatment_date).days

        # Sabse pehle ek empty DataFrame banate hain with 44 features
        final_features = [
            "age", "bmi", "cholesterol_level", "hypertension", "asthma", 
            "cirrhosis", "other_cancer", "gender_Male", "country_Belgium", 
            "country_Bulgaria", "country_Croatia", "country_Cyprus", 
            "country_Czech Republic", "country_Denmark", "country_Estonia", 
            "country_Finland", "country_France", "country_Germany", 
            "country_Greece", "country_Hungary", "country_Ireland", 
            "country_Italy", "country_Latvia", "country_Lithuania", 
            "country_Luxembourg", "country_Malta", "country_Netherlands", 
            "country_Poland", "country_Portugal", "country_Romania", 
            "country_Slovakia", "country_Slovenia", "country_Spain", 
            "country_Sweden", "cancer_stage_Stage II", "cancer_stage_Stage III", 
            "cancer_stage_Stage IV", "family_history_Yes", 
            "smoking_status_Former Smoker", "smoking_status_Never Smoked", 
            "smoking_status_Passive Smoker", "treatment_type_Combined", 
            "treatment_type_Radiation", "treatment_type_Surgery"
        ]
        
        input_df = pd.DataFrame(columns=final_features, index=[0])
        input_df.fillna(0, inplace=True)
        
        input_df['age'] = age_val
        input_df['bmi'] = bmi_val
        input_df['cholesterol_level'] = cholesterol_val
        
        if gender == 'Male':
            input_df['gender_Male'] = 1
        
        if family_history == 'Yes':
            input_df['family_history_Yes'] = 1

        if hypertension == 'Yes':
            input_df['hypertension'] = 1
        if asthma == 'Yes':
            input_df['asthma'] = 1
        if cirrhosis == 'Yes':
            input_df['cirrhosis'] = 1
        if other_cancer == 'Yes':
            input_df['other_cancer'] = 1

        if cancer_stage == 'II':
            input_df['cancer_stage_Stage II'] = 1
        elif cancer_stage == 'III':
            input_df['cancer_stage_Stage III'] = 1
        elif cancer_stage == 'IV':
            input_df['cancer_stage_Stage IV'] = 1
            
        if smoking_status == 'Former Smoker':
            input_df['smoking_status_Former Smoker'] = 1
        elif smoking_status == 'Never Smoked':
            input_df['smoking_status_Never Smoked'] = 1
        elif smoking_status == 'Passive Smoker':
            input_df['smoking_status_Passive Smoker'] = 1
        
        if treatment_type == 'Combined':
            input_df['treatment_type_Combined'] = 1
        elif treatment_type == 'Radiation':
            input_df['treatment_type_Radiation'] = 1
        elif treatment_type == 'Surgery':
            input_df['treatment_type_Surgery'] = 1
            
        country_col_name = f'country_{country}'
        if country_col_name in input_df.columns:
            input_df[country_col_name] = 1

        prediction = model.predict(input_df)
        
        st.toast("✅ Prediction successful!", icon="✅")
        
        result = prediction[0]
        if result == 1:
            st.success("✅ The model predicts a HIGH chance of survival!")
        else:
            st.error("⚠ The model predicts a LOW chance of survival.")
            
    except ValueError as e:
        st.error(f"Error: Invalid input. Please enter valid numbers for Age, BMI, and Cholesterol. Details: {e}")
    except Exception as e:
        st.error(f"An unexpected error occurred during prediction. Please check your inputs. Error: {e}")