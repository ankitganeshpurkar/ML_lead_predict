# Import libraries
import numpy as np
import pandas as pd
import streamlit as st
import pickle
from PIL import Image
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors


st.write(""" # A Web Application for Prediction of Inhibitor for Alzheimer's Disease   """)
result = [2]

user_input = st.text_input("Enter the SMILES string", ' ')
try:
    mol = Chem.MolFromSmiles(user_input)
    df_des =  pd.DataFrame()
    calc = MoleculeDescriptors.MolecularDescriptorCalculator([x[0] for x in Descriptors._descList])
    header = list(calc.GetDescriptorNames())
    d2 = list(calc.CalcDescriptors(mol))
    d = {x:y for x,y in zip(header, d2)}
    df_des =  df_des.append(d, ignore_index = True)
    
except:
    result = [-1]
    


target_name = st.selectbox('Select target', ('BACE1', 'GSK3B'))

if user_input == " ":
    result[0] = -1
    
else:
    if target_name == "BACE1":
        feat = ['MinEStateIndex', 'MinAbsEStateIndex', 'qed', 'NumRadicalElectrons', 'BCUT2D_MWLOW', 'BCUT2D_CHGHI', 'BCUT2D_LOGPHI', 'BCUT2D_LOGPLOW', 'BCUT2D_MRLOW', 'HallKierAlpha', 'PEOE_VSA1', 'PEOE_VSA13', 'PEOE_VSA14', 'PEOE_VSA7', 'SMR_VSA5', 'SMR_VSA6', 'EState_VSA3', 'EState_VSA6', 'VSA_EState2', 'VSA_EState4', 'VSA_EState9', 'NumAliphaticHeterocycles', 'NumAliphaticRings', 'fr_alkyl_halide', 'fr_guanido']
        X = df_des[feat]
        option1 = st.selectbox('Select the model', ('Random forest', 'SVC', 'KNN'))
        if option1 == 'Random forest':
            loaded_model = pickle.load(open('./data/Final_model_BACE1_RF.pkl', 'rb'))
            result = loaded_model.predict(X)
                
        elif option1 == 'SVC':
            loaded_model = pickle.load(open('./data/Final_model_BACE1_SVC.pkl', 'rb'))
            scaler = pickle.load(open('./data/ss_BACE1.pkl', 'rb'))
            Xtest = scaler.transform(X)
            result = loaded_model.predict(Xtest)
            
        elif option1 == 'KNN':
            loaded_model = pickle.load(open('./data/Final_model_BACE1_KNN.pkl', 'rb'))
            scaler = pickle.load(open('./data/ss_BACE1.pkl', 'rb'))
            Xtest = scaler.transform(X)
            result = loaded_model.predict(Xtest)

    if target_name == "GSK3B":
        feat = ['MinEStateIndex', 'qed', 'MolWt', 'MaxAbsPartialCharge', 'BCUT2D_MWLOW', 'BCUT2D_CHGHI', 'BCUT2D_CHGLO', 'BCUT2D_LOGPHI', 'BCUT2D_LOGPLOW', 'BCUT2D_MRHI', 'BertzCT', 'HallKierAlpha', 'Ipc', 'PEOE_VSA11', 'PEOE_VSA13', 'PEOE_VSA2', 'PEOE_VSA6', 'PEOE_VSA7', 'PEOE_VSA9', 'SlogP_VSA11', 'SlogP_VSA2', 'VSA_EState2', 'VSA_EState3', 'NumSaturatedRings', 'fr_NH1']
        X = df_des[feat]
        option1 = st.selectbox('Select the model', ('Random forest', 'SVC', 'KNN'))
        if option1 == 'Random forest':
            loaded_model = pickle.load(open('./data/Final_model_GSK3B_RF.pkl', 'rb'))
            result = loaded_model.predict(X)
                
        elif option1 == 'SVC':
            loaded_model = pickle.load(open('./data/Final_model_GSK3B_SVC.pkl', 'rb'))
            scaler = pickle.load(open('./data/ss_GSK3B.pkl', 'rb'))
            Xtest = scaler.transform(X)
            result = loaded_model.predict(Xtest)
            
        elif option1 == 'KNN':
            loaded_model = pickle.load(open('./data/Final_model_GSK3B_KNN.pkl', 'rb'))
            scaler = pickle.load(open('./data/ss_GSK3B.pkl', 'rb'))
            Xtest = scaler.transform(X)
            result = loaded_model.predict(Xtest)

   
     
st.header('Prediction')
if result[0] == 1:
    st.write("Active")
elif result[0] == 0:
    st.write("Moderately Active")
elif result[0] == -1:
    st.write("Enter the correct smiles string.")
    
    
st.write("""

Key: IC50 <= 5000 nM : Active and IC50 > 5000 nM : Moderately Active  """)
