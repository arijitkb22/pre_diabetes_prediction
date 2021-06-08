import pandas as pd
import numpy as np

import streamlit as st
from sklearn import tree
from sklearn.model_selection import train_test_split
from PIL import Image
import io

# Adding Title
st.title("Pre Diabetes prediction")

st.write(""" #Pre Diabetes Prediction using Decission Tree Algorithm """)

#image
image=Image.open(r'C:\Users\HP\Desktop\images.jpg')
st.image(image,caption="Diabetes Prediction" )


# Data
data = pd.read_csv(r'C:\Users\HP\Desktop\Data\diabetes_data_upload.csv')

for col in data.columns:
    data.loc[data["Obesity"] == "Yes", "Obesity"] = 1
    data.loc[data["Obesity"] == "No", "Obesity"] = 0
for col in data.columns:
    data.loc[data["Gender"] == "Male", "Gender"] = 1
    data.loc[data["Gender"] == "Female", "Gender"] = 0
for col in data.columns:
    data.loc[data["Polyuria"] == "Yes", "Polyuria"] = 1
    data.loc[data["Polyuria"] == "No", "Polyuria"] = 0
for col in data.columns:
    data.loc[data["Polydipsia"] == "Yes", "Polydipsia"] = 1
    data.loc[data["Polydipsia"] == "No", "Polydipsia"] = 0
for col in data.columns:
    data.loc[data["sudden weight loss"] == "Yes", "sudden weight loss"] = 1
    data.loc[data["sudden weight loss"] == "No", "sudden weight loss"] = 0
for col in data.columns:
    data.loc[data["weakness"] == "Yes", "weakness"] = 1
    data.loc[data["weakness"] == "No", "weakness"] = 0
for col in data.columns:
    data.loc[data["Polyphagia"] == "Yes", "Polyphagia"] = 1
    data.loc[data["Polyphagia"] == "No", "Polyphagia"] = 0
for col in data.columns:
    data.loc[data["Genital thrush"] == "Yes", "Genital thrush"] = 1
    data.loc[data["Genital thrush"] == "No", "Genital thrush"] = 0
for col in data.columns:
    data.loc[data["visual blurring"] == "Yes", "visual blurring"] = 1
    data.loc[data["visual blurring"] == "No", "visual blurring"] = 0
for col in data.columns:
    data.loc[data["Itching"] == "Yes", "Itching"] = 1
    data.loc[data["Itching"] == "No", "Itching"] = 0
for col in data.columns:
    data.loc[data["Irritability"] == "Yes", "Irritability"] = 1
    data.loc[data["Irritability"] == "No", "Irritability"] = 0
for col in data.columns:
    data.loc[data["delayed healing"] == "Yes", "delayed healing"] = 1
    data.loc[data["delayed healing"] == "No", "delayed healing"] = 0
for col in data.columns:
    data.loc[data["partial paresis"] == "Yes", "partial paresis"] = 1
    data.loc[data["partial paresis"] == "No", "partial paresis"] = 0
for col in data.columns:
    data.loc[data["muscle stiffness"] == "Yes", "muscle stiffness"] = 1
    data.loc[data["muscle stiffness"] == "No", "muscle stiffness"] = 0
for col in data.columns:
    data.loc[data["Alopecia"] == "Yes", "Alopecia"] = 1
    data.loc[data["Alopecia"] == "No", "Alopecia"] = 0
for col in data.columns:
    data.loc[data["class"] == "Positive", "class"] = 1
    data.loc[data["class"] == "Negative", "class"] = 0

x = pd.DataFrame(data.iloc[:, :-1])
y = data[["class"]]
y=y.astype('int')





x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 1, stratify = y)

# fitting the model
model = tree.DecisionTreeClassifier()
model.fit(x_train, y_train)

# making Prediction

y_pred = model.predict(x_test)
#y_pred = pd.DataFrame(y_pred, columns=["Predicted"])

gender_type = st.sidebar.selectbox("Select gender", ("male", "female"))
def get_gender(gender):
    if gender == "male":
        return 1
    else:
        return 0

Polyuria_type = st.sidebar.selectbox("Polyuria", ("yes", "no"))
def get_Polyuria(Polyuria):
    if Polyuria == "yes":
        return 1
    else:
        return 0

Polydipsia_type = st.sidebar.selectbox("Polydipsia", ("yes", "no"))
def get_Polydipsia(Polydipsia):
    if Polydipsia=="yes":
        return 1
    else:
        return 0


sudden_weight_loss_type = st.sidebar.selectbox("sudden weight loss", ("yes", "no"))
def get_SWL(sudden_weight_loss):
    if sudden_weight_loss == "yes":
        return 1
    else:
        return 0

weakness_type = st.sidebar.selectbox("weakness", ("yes", "no"))
def get_weakness(weakness):
    if weakness == "yes":
        return 1
    else:
        return 0

Polyphagia_type = st.sidebar.selectbox("Polyphagia", ("yes", "no"))
def get_Polyphagia(Polyphagia):
    if Polyphagia == "yes":
        return 1
    else:
        return 0

Genital_thrush_type = st.sidebar.selectbox("Genital thrush", ("yes", "no"))
def get_Genital_thrush(Genital_thrush):
    if Genital_thrush == "yes":
        return 1
    else:
        return 0

visual_blurring_type = st.sidebar.selectbox("visual blurring", ("yes", "no"))
def get_visual_blurring(visual_blurring):
    if visual_blurring == "yes":
        return 1
    else:
        return 0

Itching_type = st.sidebar.selectbox("Itching", ("yes", "no"))
def get_Itching(Itching):
    if Itching == "yes":
        return 1
    else:
        return 0

Irritability_type = st.sidebar.selectbox("Irritability", ("yes", "no"))
def get_Irritability(Irritability):
    if Irritability == "yes":
        return 1
    else:
        return 0

delayed_healing_type = st.sidebar.selectbox("delayed_healing", ("yes", "no"))
def get_delayed_healing(delayed_healing):
    if delayed_healing == "yes":
        return 1
    else:
        return 0

partial_paresis_type = st.sidebar.selectbox("partial_paresis", ("yes", "no"))
def get_partial_paresis(partial_paresis):
    if partial_paresis == "yes":
        return 1
    else:
        return 0

muscle_stiffness_type = st.sidebar.selectbox("muscle_stiffness", ("yes", "no"))
def get_muscle_stiffness(muscle_stiffness):
    if muscle_stiffness == "yes":
        return 1
    else:
        return 0

Alopecia_type = st.sidebar.selectbox("Alopecia", ("yes", "no"))
def get_Alopecia(Alopecia):
    if Alopecia == "yes":
        return 1
    else:
        return 0

Obesity_type = st.sidebar.selectbox("Obesity", ("yes", "no"))
def get_Obesity(Obesity):
    if Obesity == "yes":
        return 1
    else:
        return 0

age =st.sidebar.slider("age", 1, 110)

new_pr= np.array([[age, get_gender(gender_type), get_Polyuria(Polyuria_type), get_Polydipsia(Polydipsia_type), get_SWL(sudden_weight_loss_type), get_weakness(weakness_type), get_Polyphagia(Polyphagia_type), get_Genital_thrush(Genital_thrush_type), get_visual_blurring(visual_blurring_type), get_Itching(Itching_type), get_Irritability(Irritability_type), get_delayed_healing(delayed_healing_type), get_partial_paresis(partial_paresis_type), get_muscle_stiffness(muscle_stiffness_type), get_Alopecia(Alopecia_type), get_Obesity(Obesity_type)]])
new_pr = new_pr.reshape(1, -1)

if st.button("Predict"):
    price = model.predict(new_pr)
    #st.write("diabetes:", price)
    if price==1:
        st.write("""you have a tendency to became diabetes patient over 50%.
         the following suggestions are for you:""")
    else:
        st.write("""you does not have any tendency to be a diabetes patient.
        you are suggested the following tips to be safe and healthy""")
