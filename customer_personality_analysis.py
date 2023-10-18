
import numpy as np
import pickle
import streamlit as st
import xgboost

# loading the saved model
loaded_model = pickle.load(open('trained_model.sav', 'rb'))


# creating a function for Prediction

def cluster_prediction(input_data):
    # Convert input_data to a list of float values, handling empty strings
    input_data = [float(value) if value.strip() else 0.0 for value in input_data]

    # Create a NumPy array from the input data
    input_data_as_numpy_array = np.array(input_data)

    # Reshape the array for prediction
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

    # Make the prediction
    prediction = loaded_model.predict(input_data_reshaped)

    # Return the cluster prediction
    if prediction[0] == 0:
        return 'Cluster 0'
    elif prediction[0] == 1:
        return 'Cluster 1'
    else:
        return 'Cluster 2'


def main():
    # giving a title
    st.title('Cluster Prediction Web App')

    # getting the input data from the user

    Marital_Status = st.text_input('Marital Status')
    Income = st.text_input('Income')
    Kidhome = st.text_input('Number of kids in home')
    Teenhome = st.text_input('Number of teenagers in home')
    Recency = st.text_input('Recency')
    MntWines = st.text_input('Amount of wines')
    MntFruits = st.text_input('Amount of fruits')
    MntMeatProducts = st.text_input('Amount of meat products')
    MntFishProducts = st.text_input('Amount of fish products')
    MntSweetProducts = st.text_input('Amount of sweet products')
    MntGoldProds = st.text_input('Amount of gold products')
    NumDealsPurchases = st.text_input('Number of deals purchases')
    NumWebPurchases = st.text_input('Number of web purchases')
    NumCatalogPurchases = st.text_input('Number of catalog purchases')
    NumStorePurchases = st.text_input('Number of store purchases')
    NumWebVisitsMonth = st.text_input('Number of Vists per month')
    AcceptedCmp5 = st.text_input('Accepted in camp 5')
    AcceptedCmp4 = st.text_input('Accepted in camp 4')
    AcceptedCmp3 = st.text_input('Accepted in camp 3')
    AcceptedCmp2 = st.text_input('Accepted in camp 2')
    AcceptedCmp1 = st.text_input('Accepted in camp 1')
    Complain = st.text_input('Complain')
    Response = st.text_input('Response')
    age = st.text_input('Age')
    years_of_enrollment = st.text_input('Years of enrollment')
    Education_Graduation = st.text_input('Graduate or not')
    Education_Master = st.text_input('Masters or not')
    Education_Others = st.text_input('Others')
    Education_PhD = st.text_input('Phd or not')



    # code for Prediction
    clusters = ''

    # creating a button for Prediction

    if st.button('Cluster Result'):
        clusters = cluster_prediction(
            [Marital_Status, Income, Kidhome, Teenhome, Recency,
             MntWines, MntFruits, MntMeatProducts, MntFishProducts,
             MntSweetProducts, MntGoldProds, NumDealsPurchases,
             NumWebPurchases, NumCatalogPurchases,
             NumStorePurchases,NumWebVisitsMonth, AcceptedCmp3, AcceptedCmp4, AcceptedCmp5,
             AcceptedCmp1, AcceptedCmp2, Complain, Response, age,
             years_of_enrollment, Education_Graduation, Education_Master,
             Education_Others, Education_PhD])

    st.success(clusters)


if __name__ == '__main__':
    main()
