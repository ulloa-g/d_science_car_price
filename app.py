import pandas as pd

data_set_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data"
df = pd.read_csv(data_set_url, header=None)
