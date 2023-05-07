from catboost import CatBoostRegressor, CatBoost
import streamlit as st
import pandas as pd

model = CatBoostRegressor()
model.load_model("model.cbm")

def predict_price(cbm,df):
    pred = model.predict(df)
    return round(pred[0],2)

st.title("Elmas fiyatı tahminleyicisi")
st.write("Seçmiş olduğunuz özellikler için elmas fiyatı 98% doğrulukla tahmin edilmektedir.")

carat = st.sidebar.slider(label="Carat",min_value=0.0,max_value=5.0,value=2.5,step=0.1)

cut_dict = {'Fair': 0, 'Good': 1, 'Ideal': 2, 'Premium': 3, 'Very Good': 4}
cut = cut_dict[st.sidebar.selectbox("Cut",cut_dict.keys())]

color_dict = {'D': 0, 'E': 1, 'F': 2, 'G': 3, 'H': 4, 'I': 5, 'J': 6}
color = color_dict[st.sidebar.selectbox("Color",color_dict.keys())]

clarity_dict = {'I1': 0,
  'IF': 1,
  'SI1': 2,
  'SI2': 3,
  'VS1': 4,
  'VS2': 5,
  'VVS1': 6,
  'VVS2': 7}

clarity = clarity_dict[st.sidebar.selectbox("Clarity",(clarity_dict.keys()))]

table = st.sidebar.slider(label="Table",min_value=40.0,max_value=100.0,value=70.0,step=0.1)
depth = st.sidebar.slider(label="Depth",min_value=30.0,max_value=80.0,value=40.0,step=0.1)

x = st.sidebar.slider(label="x",min_value=0.00,max_value=11.00, value=5.50,step=0.01)
y = st.sidebar.slider(label="y",min_value=0.00,max_value=60.00, value=30.00,step=0.01)
z = st.sidebar.slider(label="z",min_value=2.00,max_value=40.00, value=20.00,step=0.01)

features = {'carat':carat,'cut':cut,'color':color,'clarity':clarity,'table':table,'depth':depth,'x':x,'y':y,'z':z}
features_df = pd.DataFrame([features])

st.table(features_df)

if st.button("Tahmin Et"):
    pred = predict_price(model,features_df)
    st.write(f"Tahmin edilen değer: {pred}")




