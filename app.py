import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from utils import load_data
from model import train_model

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Ocean Waste Tracker", layout="wide")

st.title("🌊 AI Ocean Waste Tracking System")

# ---------------- LOAD DATA ----------------
df, le_waste, le_area, le_pollution = load_data()

# Train model
model = train_model(df)

# ---------------- SIDEBAR INPUT ----------------
st.sidebar.header("Enter Waste Details")

waste_type = st.sidebar.selectbox("Waste Type", le_waste.classes_)
quantity = st.sidebar.slider("Quantity", 0, 100)
area_type = st.sidebar.selectbox("Area Type", le_area.classes_)

# Encode input
waste_encoded = le_waste.transform([waste_type])[0]
area_encoded = le_area.transform([area_type])[0]

# ---------------- PREDICTION ----------------
if st.sidebar.button("Predict Pollution"):
    result = model.predict([[waste_encoded, quantity, area_encoded]])
    
    level = le_pollution.inverse_transform(result)[0]
    
    st.subheader("🌡️ Predicted Pollution Level")
    st.success(level)

# ---------------- PREPARE DISPLAY DATA ----------------
df_display = df.copy()
df_display["Waste_Type"] = le_waste.inverse_transform(df["Waste_Type"])
df_display["Area_Type"] = le_area.inverse_transform(df["Area_Type"])
df_display["Pollution_Level"] = le_pollution.inverse_transform(df["Pollution_Level"])

# ---------------- CHARTS ----------------
st.subheader("📊 Waste Quantity by Waste Type")

waste_group = df_display.groupby("Waste_Type")["Quantity"].sum()

fig1, ax1 = plt.subplots()
waste_group.plot(kind="bar", ax=ax1)

st.pyplot(fig1)

# ---------------- LOCATION CHART ----------------
st.subheader("📍 Waste Quantity by Location")

location_group = df_display.groupby("Location")["Quantity"].sum()

fig2, ax2 = plt.subplots()
location_group.plot(kind="bar", ax=ax2)

st.pyplot(fig2)

# ---------------- MAP ----------------
st.subheader("🗺️ Waste Location Map")

map_data = df_display.rename(columns={
    "Latitude": "lat",
    "Longitude": "lon"
})

st.map(map_data)

# ---------------- INSIGHTS ----------------
st.subheader("🚨 Insights")

st.write("• High quantity areas require immediate cleanup.")
st.write("• Plastic waste is the major contributor to pollution.")
st.write("• Coastal and industrial regions show higher pollution levels.")

# ---------------- DATA TABLE ----------------
st.subheader("📄 Dataset Preview")
st.dataframe(df_display)