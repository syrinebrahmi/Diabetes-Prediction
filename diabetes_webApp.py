# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 22:49:33 2023

@author: brahm
"""
import numpy as np
import pickle
import streamlit as st
# creating a function for Prediction
loaded_model = pickle.load(open('C:/Users/brahm/Desktop/Stage-Diabetes/svm_model.sav', 'rb'))

def diabetes_prediction(input_data):
    
    # changing the input_data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)

    if (prediction[0] == 0):
        return 'Le patient est diabétique'
    else:
        return 'Le patient est non diabétique'
  
    
  
def main():
    
    
    # giving a title
    st.title('Diabetes Prediction')
    
    
    # getting the input data from the user
    
    
    Pregnancies = st.text_input('Number of Pregnancies')
    Glucose = st.text_input('Glucose Level')
    BloodPressure = st.text_input('Blood Pressure value')
    SkinThickness = st.text_input('Skin Thickness value')
    Insulin = st.text_input('Insulin Level')
    BMI = st.text_input('BMI value')
    DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function value')
    Age = st.text_input('Age')
    
    
    # code for Prediction
    diagnosis = ''
    
    # creating a button for Prediction
    
    if st.button('Test'):
        diagnosis = diabetes_prediction([Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age])
        
        
    st.success(diagnosis)
    
    
    
    
    
if __name__ == '__main__':
    main()