import pandas as pd
import numpy as np

PATH = "/Users/ulloa/programing/projects/d_science_car_price/cars.csv"
DATA_SET_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data"


def show_missing_data(dataframe):
    missing_data = dataframe.isnull()
    for c in missing_data.columns.values.tolist():
        print(missing_data[c].value_counts())
    return


df = pd.read_csv(DATA_SET_URL, header=None)
headers = ["symboling", "normalized-losses", "make", "fuel-type", "aspiration", "num-of-doors", "body-style", "drive-wheels", "engine-location", "wheel-base", "length", "width", "height",
           "curb-weight", "engine-type", "num-of-cylinders", "engine-size", "fuel-system", "bore", "stroke", "compression-ratio", "horsepower", "peak-rpm", "city-mpg", "highway-mpg", "price"]
df.columns = headers
df = df.replace('?', np.NaN)

# replace by mean:
avg_norm_loss = df["normalized-losses"].astype("float").mean(axis=0)
avg_bore = df["bore"].astype("float").mean(axis=0)
avg_stroke = df["stroke"].astype("float").mean(axis=0)
avg_horsepower = df["horsepower"].astype("float").mean(axis=0)
avg_peakrpm=df['peak-rpm'].astype('float').mean(axis=0)
df["normalized-losses"].replace(np.nan, avg_norm_loss, inplace=True)
df["bore"].replace(np.nan, avg_bore, inplace=True)
df["stroke"].replace(np.nan, avg_stroke, inplace=True)
df["horsepower"].replace(np.nan, avg_horsepower, inplace=True)
df['peak-rpm'].replace(np.nan, avg_peakrpm, inplace=True)

# replace by frecuency:
#print(df["num-of-doors"].value_counts())  # --> show count on every element
#print(df["num-of-doors"].value_counts().idxmax())  # --> calculate the most frecuent one automatically
df["num-of-doors"].replace(np.nan, "four", inplace=True)

# drop the whole row:
df = df.dropna(subset=["price"], axis=0)

# print(df.dtypes)  # --> we need to change some datatypes here:
df[["bore", "stroke"]] = df[["bore", "stroke"]].astype("float")
df[["normalized-losses"]] = df[["normalized-losses"]].astype("int")
df[["price"]] = df[["price"]].astype("float")
df[["peak-rpm"]] = df[["peak-rpm"]].astype("float")

# standarization
df["city-L/100km"] = 235 / df["city-mpg"]  # --> we create a new column with the results
df["highway-mpg"] = 235 / df["highway-mpg"]  # --> we modified the existing column
df.rename(columns={"highway-mpg":"highway-L/100km"}, inplace=True)  # --> then we change the name

# normalization
df["length"] = df["length"] / df["length"].max()
df["width"] = df["width"] / df["width"].max()
df["height"] = df["height"] / df["height"].max()

# binning --> horsepower from 59 unique values to 3 groups
df["horsepower"] = df["horsepower"].astype(int)
bins = np.linspace(min(df["horsepower"]), max(df["horsepower"]), 4)
group_name = ["Low", "Medium", "High"]
df["horsepower-binned"] = pd.cut(df["horsepower"], bins, labels=group_name, include_lowest=True)

# index=False mean the row names will not be written
df.to_csv(PATH, index=False)
