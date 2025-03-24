#load lhs samples and compute the correlation between them
import ast
from data_analysis import plot_correlation
import matplotlib.pyplot as plt
import numpy as np

with open("1000_lhs_samples.txt", "r") as f:
    combinations = [ast.literal_eval(line.strip()) for line in f.readlines()]

# np.random.seed(42)

# combinations = np.random.uniform(0, 1, (4000, 4))

#create pandas dataframe from the combinations
import pandas as pd

df = pd.DataFrame(combinations,
                  columns=["box_xsize", "box_ysize", "box_zsize", "box_mass"])

#compute the correlation matrix
correlation_matrix = df.corr()

#plot the correlation matrix
plot_correlation(correlation_matrix, "lhs_samples")

plt.show()
