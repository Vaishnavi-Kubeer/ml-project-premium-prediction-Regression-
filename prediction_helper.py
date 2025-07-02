import pandas as pd
#from PIL.ImageOps import scale
from joblib import load

#from project_1_build_an_app_using_streamlit_resources.app.main import prediction
#from project_1_build_an_app_using_streamlit_resources.app.prediction_helper import handle_scaling

model_rest=load('artifacts/model_rest.joblib')
model_young=load('artifacts/model_young.joblib')
scaler_rest=load('artifacts/scaler_rest.joblib')
scaler_young=load('artifacts/scaler_young.joblib')

def calculate_normalised_risk_score(medical_history):
    risk_factors = {
        'diabetes': 6,
        'heart disease': 8,
        'high blood pressure': 6,
        'thyroid': 5,
        'no disease': 0,
        'none': 0
    }
    diseases = medical_history.lower().strip().split(" & ")

    total_risk_score = sum(risk_factors.get(d, 0) for d in diseases[:2])

    MIN_SCORE = 0
    MAX_SCORE = 14

    normalized_score = (total_risk_score - MIN_SCORE) / (MAX_SCORE - MIN_SCORE)

    return normalized_score


# take input data from streamlit app which is stored in input_dict and form a dataframe out of it so that
# it can be given to model
def preprocess_input(input_dict):
    expected_columns=['age', 'number_of_dependants', 'income_lakhs', 'insurance_plan',
       'genetical_risk', 'normalized_risk_score', 'gender_Male',
       'region_Northwest', 'region_Southeast', 'region_Southwest',
       'marital_status_Unmarried', 'bmi_category_Obesity',
       'bmi_category_Overweight', 'bmi_category_Underweight',
       'smoking_status_Occasional', 'smoking_status_Regular',
       'employment_status_Salaried', 'employment_status_Self-Employed']

    insurance_plan_encoding = {'Bronze': 1, 'Silver': 2, 'Gold': 3}

    df=pd.DataFrame(0,columns=expected_columns,index=[0])

    for key,value in input_dict.items():
        if key=='Gender' and value=='Male':
            df['gender_Male']=1
        elif key=='Region':
            if value=='Northwest':
                df['region_Northwest']=1
            elif value=='Southeast':
                df['region_Southeast']=1
            elif value=='Southwest':
                df['region_Southwest']=1
        elif key=='Marital Status':
            if value=='Unmarried':
                df['marital_status_Unmarried']=1
        elif key=='BMI Category':
            if value=='Underweight':
                df['bmi_category_Underweight']=1
            elif value=='Obesity':
                df['bmi_category_Obesity']=1
            elif value=='Overweight':
                df['bmi_category_Overweight']=1
        elif key=='Smoking Status':
            if value=='Regular':
                df['smoking_status_Regular']=1
            elif value=='Occasional':
                df['smoking_status_Occasional']=1
        elif key=='Employment Status':
            if value=='Salaried':
                df['employment_status_Salaried']=1
            elif value == 'Self-Employed':
                df['employment_status_Self-Employed'] = 1
        elif key=='Insurance Plan':
            df['insurance_plan']=insurance_plan_encoding.get(value,1)
        elif key=='Age':
            df['age']=value
        elif key=='Number of Dependants':
            df['number_of_dependants']=value
        elif key=='Income in Lakhs':
            df['income_lakhs']=value
        elif key=='Genetical Risk':
            df['genetical_risk']=value

    df['normalized_risk_score']=calculate_normalised_risk_score(input_dict['Medical History'])
    df=handle_scaling(input_dict['Age'],df)

    return df

def handle_scaling(age,df):
    if age<=25:
        scaler_object=scaler_young
    else:
        scaler_object=scaler_rest

    cols_to_scale=scaler_object['cols_to_scale']
    scaler=scaler_object['scaler']

    # cols_to_scale has income_level column and whenwe run streamlit app it shows income_level not in index
    # we can add dummy column income_level and apply scaling and then drop it later
    df['income_level'] = None
    df[cols_to_scale]=scaler.transform(df[cols_to_scale])
    df.drop('income_level',axis='columns',inplace=True)

    return df

def predict(input_dict):
    input_df=preprocess_input(input_dict)
    #print(input_df)
    if input_dict['Age']<=25:
        prediction=model_young.predict(input_df)
    else:
        prediction=model_rest.predict(input_df)
    return int(prediction)