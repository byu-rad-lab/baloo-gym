#load lhs samples and compute the correlation between them
import ast
from data_analysis import plot_correlation

with open("lhs_samples.txt", "r") as f:
    combinations = [
        ast.literal_eval(line.strip()) for line in f.readlines()
    ]


#create pandas dataframe from the combinations
import pandas as pd
df = pd.DataFrame(combinations, columns=["box_xsize", "box_ysize", "box_zsize", "box_position"])

#compute the correlation matrix
correlation_matrix = df.corr()

#plot the correlation matrix
plot_correlation(correlation_matrix, "lhs_samples")

