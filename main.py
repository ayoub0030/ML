import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
import matplotlib.pyplot as plt
import seaborn as sns

data = fetch_california_housing(as_frame=True)
df = data.frame  # Convert to a pandas DataFrame

print("Shape of the dataset:", df.shape)
print("\nFirst 5 rows:")
print(df.head())