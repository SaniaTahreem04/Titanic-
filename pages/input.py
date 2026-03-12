import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

st.title("Passenger Survival Prediction")

data = pd.read_csv("titanic.csv")

data['Age'].fillna(data['Age'].mean(), inplace=True)
data['Fare'].fillna(data['Fare'].mean(), inplace=True)

data['Sex'] = data['Sex'].map({'male':0,'female':1})

X = data[['Pclass','Sex','Age','Fare']]
y = data['Survived']

x_train, x_test, y_train, y_test = train_test_split(X,y,test_size=0.2)

model = LogisticRegression()
model.fit(x_train,y_train)

st.header("Enter Passenger Details")

pclass = st.selectbox("Passenger Class",[1,2,3])
sex = st.selectbox("Gender",["Male","Female"])
age = st.slider("Age",0,80,25)
fare = st.slider("Fare",0,500,50)

sex_value = 1 if sex=="Female" else 0

if st.button("Predict Survival"):

    prediction = model.predict([[pclass,sex_value,age,fare]])

    if prediction[0] == 1:
        st.success("🎉 Passenger Survived")
        st.image("survived.png", width=200)

    else:
        st.error("❌ Passenger Did Not Survive")
        st.image("not-survived.png", width=200)