import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# -----------------------
# PAGE CONFIG
# -----------------------
st.set_page_config(
    page_title="Titanic Survival Prediction",
    layout="wide"
)

# -----------------------
# TITLE SECTION
# -----------------------
st.title("🚢 Titanic Survival Prediction")

st.image(
    "https://upload.wikimedia.org/wikipedia/commons/f/fd/RMS_Titanic_3.jpg",
    use_container_width=True
)



# -----------------------
# LOAD DATA
# -----------------------
data = pd.read_csv("titanic.csv")

# Data Cleaning
data['Age'].fillna(data['Age'].mean(), inplace=True)
data['Fare'].fillna(data['Fare'].mean(), inplace=True)

data['Sex'] = data['Sex'].map({'male':0,'female':1})

X = data[['Pclass','Sex','Age','Fare']]
y = data['Survived']

x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

model = LogisticRegression()
model.fit(x_train,y_train)



# -----------------------
# DATA VISUALIZATION
# -----------------------
st.header("Data Analysis")

col1,col2 = st.columns(2)

with col1:

    st.subheader("Survival Count")

    fig1,ax1 = plt.subplots()
    sns.countplot(x='Survived',data=data,ax=ax1)

    st.pyplot(fig1)

with col2:

    st.subheader("Survival by Gender")

    fig2,ax2 = plt.subplots()
    sns.countplot(x='Sex',hue='Survived',data=data,ax=ax2)

    st.pyplot(fig2)

# Scatter Plot
st.subheader("Passenger Data Visualization")

col3, col4 = st.columns(2)

# Age Distribution
with col3:
    st.subheader("Age Distribution")

    fig1, ax1 = plt.subplots(figsize=(5,3))
    sns.histplot(data['Age'], bins=30, ax=ax1)

    st.pyplot(fig1)


# Age vs Fare
with col4:
    st.subheader("Age vs Fare")

    age_range = st.slider("Select Age Range", 0, 80, (10, 60))

    filtered_data = data[(data['Age'] >= age_range[0]) & (data['Age'] <= age_range[1])]

    fig = px.scatter(
        filtered_data,
        x="Age",
        y="Fare",
        color="Survived",
        hover_data=["Pclass","Sex"],
        title="Age vs Fare (Interactive)"
    )

    st.plotly_chart(fig, use_container_width=True)