import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

st.title("Passenger Survival Prediction")

# Load dataset
data = pd.read_csv("titanic.csv")

# Data preprocessing
data['Age'].fillna(data['Age'].mean(), inplace=True)
data['Fare'].fillna(data['Fare'].mean(), inplace=True)
data['Embarked'].fillna(data['Embarked'].mode()[0], inplace=True)

data['Sex'] = data['Sex'].map({'male':0,'female':1})
data['Embarked'] = data['Embarked'].map({'S':0,'C':1,'Q':2})

# Features and target
X = data[['Pclass','Sex','Age','Fare','Embarked']]
y = data['Survived']

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=42)

# Model training
model = LogisticRegression(max_iter=1000)
model.fit(x_train,y_train)

# Model prediction for evaluation
y_pred = model.predict(x_test)

# Metrics calculation
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

st.header("Enter Passenger Details")

# 2 columns layout
col1, col2 = st.columns(2)

with col1:
    pclass = st.selectbox("Passenger Class",[1,2,3])
    age = st.slider("Age",0,80,25)
    embarked = st.selectbox("Embarkment Port",["Southampton","Cherbourg","Queenstown"])

with col2:
    sex = st.selectbox("Gender",["Male","Female"])
    fare = st.slider("Fare",0,500,50)

# Convert inputs
sex_value = 1 if sex=="Female" else 0

if embarked == "Southampton":
    embarked_value = 0
elif embarked == "Cherbourg":
    embarked_value = 1
else:
    embarked_value = 2

# Prediction button
# Prediction button
if st.button("Predict Survival"):

    # Make prediction
    prediction = model.predict([[pclass, sex_value, age, fare, embarked_value]])

    # Show result
    if prediction[0] == 1:
        st.success("🎉 Passenger Survived")
    else:
        st.error("❌ Passenger Did Not Survive")

    # Show metrics table after prediction
    st.subheader("Model Evaluation Metrics")

    metrics_table = pd.DataFrame({
        "Model Name": ["Logistic Regression"],
        "Type": ["Classification"],
        "Score Name": ["Accuracy"],
        "Score": [round(accuracy, 2)]
    })

    st.table(metrics_table)