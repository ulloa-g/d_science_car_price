import pandas as pd
import numpy as np

PATH = "/Users/ulloa/programing/projects/d_science_car_price/cars.csv"
DATA_SET_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data"

df = pd.read_csv(DATA_SET_URL, header=None)
headers = ["symboling", "normalized-losses", "make", "fuel-type", "aspiration", "num-of-doors", "body-style", "drive-wheels", "engine-location", "wheel-base", "length", "width", "height",
           "curb-weight", "engine-type", "num-of-cylinders", "engine-size", "fuel-system", "bore", "stroke", "compression-ratio", "horsepower", "peak-rpm", "city-mpg", "highway-mpg", "price"]
df.columns = headers
df1 = df.replace('?', np.NaN)
df = df1.dropna(subset=["price"], axis=0)
df.to_csv(PATH, index=False)  # index=False mean the row names will not be written
