# Importing streamlit
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split

# Heading
st.header("Customer Churn Prediction")
st.write("Welcome to the Customer Churn Prediction App.")

st.header("Our Dataset")
# Taking csv file as input
# uploaded_file = st.file_uploader("Choose a csv file",type="csv")
# if uploaded_file is not None:
#     df=pd.read_csv(uploaded_file)
#     df
df=pd.read_csv("dataset/customer.csv")
df

# How to comment multiple lines : CTRL+/

st.header("After Preprocessing..")
# Preprocess the data 
df ['Gender'] = df['Gender'].map({'Male':0,'Female':1})
df ['HasCrCard'] = df['HasCrCard'].map({'Yes':0,'No':1})
df ['IsActiveMember'] = df['IsActiveMember'].map({'Yes':0,'No':1})
df

st.write("Features and target variables")
X = df[['Gender','Age','Tenure','AccountBalance','ProductsNumber','IsActiveMember','HasCrCard','EstimatedSalary']]
y = df['Exited']

X
y

# Split the data into training and testing sets
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42)

st.write("X_train")
X_train

st.write("X_test")
X_test

st.write("y_test")
y_test

st.write("y_train")
y_train

model = LogisticRegression()
model.fit(X_train,y_train)

y_pred = model.predict(X_test)

# Display the accuracy
accuracy = accuracy_score(y_test,y_pred)
st.write(accuracy*100)

st.write("User Input:")
gender = st.selectbox("Gender",options=["Male","Female"])
age = st.number_input("Age",min_value=18,max_value=100)
tenure = st.number_input("Tenure (years)",min_value=0,max_value=100,value=5)
balance = st.number_input("Account Balance",min_value=0,max_value=100000,value=5000)
product_number = st.number_input("Number of Products",min_value=0,max_value=10,value=1)
has_crcard = st.selectbox("Has Credit Card)",options=["Yes",'No'])
estimated_salary = st.number_input("Estimated Salary",min_value=0,max_value=1000000,value=5000)
is_active_member = st.selectbox("is_active_member)",options=["Yes",'No'])

# Convert back
gender = 0 if gender == "Male" else 1
has_crcard = 1 if has_crcard == "Yes" else 0
is_active_member = 1 if is_active_member == "Yes" else 0

new_data = [[gender,age,tenure,balance,product_number,has_crcard,is_active_member,estimated_salary]]

# Make Prediction
prediction = model.predict(new_data)
prediction_text = "Customer will churn" if prediction[0] == 1 else "Customer will not churn"

st.write(prediction_text)