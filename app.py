import streamlit as st
import numpy as np
import pandas as pd
import pickle




#load the trained model
with open('model.pkl','rb') as file:
    model = pickle.load(file)

#load the encoder and scaler
with open('label_encoder_age.pkl','rb') as file:
    label_encoder_age = pickle.load(file)

with open('label_encoder_breast_pos.pkl','rb') as file:
    label_encoder_breast_pos = pickle.load(file)

with open('label_encoder_class.pkl','rb') as file:
    label_encoder_class = pickle.load(file)

with open('label_encoder_inv_nodes.pkl','rb') as file:
    label_encoder_inv_nodes = pickle.load(file)

with open('label_encoder_irradiat.pkl','rb') as file:
    label_encoder_irradiat = pickle.load(file)

with open('label_encoder_node_caps.pkl','rb') as file:
    label_encoder_node_caps = pickle.load(file)    

with open('label_encoder_tumorsize.pkl','rb') as file:
    label_encoder_tumorsize = pickle.load(file)

with open('onehot_encoder_breast_quad.pkl','rb') as file:
    onehot_encoder_breast_quad = pickle.load(file)

with open('onehot_encoder_menopause.pkl','rb') as file:
    onehot_encoder_menupause = pickle.load(file)

with open('scaler.pkl','rb') as file:
    scaler = pickle.load(file)


st.sidebar.title("Navigation Menu")
menu_option = st.sidebar.selectbox(
"Choose a page:",
["Home", "Prediction"]
)


if menu_option == 'Home':
    st.title('Breast Cancer Recurrance Event Prediction Using ML')
    st.write('This project harnesses the power of machine learning to predict the likelihood of an individual having  recurrence of Breast Cancer ' \
    'based on key health indicators. By analyzing a dataset of patient information including factors like age, tumor size, inv-nodes , irradiat , menopause, breast quad, node-caps and many others. Our Predictive model provides a data-driven risk assessment.')
    st.title('')
    st.write("Developed By : Fahad Farooq")
else:
    #streamlit app
    st.title("Breast Cancer Recurrance Prediction")


    #user input

    age = st.selectbox('Age',label_encoder_age.classes_)
    breast = st.selectbox('Breast',label_encoder_breast_pos.classes_)
    inv_nodes = st.selectbox("Inv Nodes", label_encoder_inv_nodes.classes_)
    irradiat = st.selectbox('Irradiat', label_encoder_irradiat.classes_)
    node_caps = st.selectbox('Node Caps', label_encoder_node_caps.classes_)
    tumor_size = st.selectbox('Tumor Size',label_encoder_tumorsize.classes_)
    diag_malig = st.selectbox('Diag Malig',[1,2,3])
    breast_quad = st.selectbox('Breast Quad',onehot_encoder_breast_quad.categories_[0])
    menopause = st.selectbox('Menupause',onehot_encoder_menupause.categories_[0])



    #preparing the data

    input_data = pd.DataFrame({
        'age':[label_encoder_age.transform([age])[0]],
        'tumor-size':[label_encoder_tumorsize.transform([tumor_size])[0]],
        'inv-nodes':[label_encoder_inv_nodes.transform([inv_nodes])[0]],
        'node-caps':[label_encoder_node_caps.transform([node_caps])[0]],
        'deg-malig':[diag_malig],
        'breast':[label_encoder_breast_pos.transform([breast])[0]],  
        'irradiat': [label_encoder_irradiat.transform([irradiat])[0]]

    })

    #onehot encoder
    breast_quad_encoded = onehot_encoder_breast_quad.transform([[breast_quad]]).toarray()
    menopause_encoded = onehot_encoder_menupause.transform([[menopause]]).toarray()
    menopause_encoded_df = pd.DataFrame(menopause_encoded,columns=onehot_encoder_menupause.get_feature_names_out(['menopause']))
    breast_quad_encoded_df = pd.DataFrame(breast_quad_encoded,columns=onehot_encoder_breast_quad.get_feature_names_out(['breast-quad']))

    #combining encoded column with input data
    data = pd.concat([input_data.reset_index(drop=True),menopause_encoded_df],axis=1)
    input_data = pd.concat([data.reset_index(drop=True),breast_quad_encoded_df],axis=1)
    print(input_data.head())

    #scaling the data
    input_data_scaled = scaler.transform(input_data)


    #prediction
    prediction = model.predict(input_data_scaled)

    st.write("Breast Cancer Recurrence Event: ",prediction[0])

    if prediction==1:
        st.write('Patient is likely to have Recurrence Cancer')
    else:

        st.write('Patient is likely not to have Recurrence Cancer')


