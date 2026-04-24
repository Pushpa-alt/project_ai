import pandas as pd
from sklearn.preprocessing import LabelEncoder

def load_data():
    df = pd.read_csv("data/ocean_waste.csv")

    le_waste = LabelEncoder()
    le_area = LabelEncoder()
    le_pollution = LabelEncoder()

    df["Waste_Type"] = le_waste.fit_transform(df["Waste_Type"])
    df["Area_Type"] = le_area.fit_transform(df["Area_Type"])
    df["Pollution_Level"] = le_pollution.fit_transform(df["Pollution_Level"])

    return df, le_waste, le_area, le_pollution