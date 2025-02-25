import json
import pandas as pd
import plotly.express as px
import numpy as np
import glob
import os


def plot_data(data):

    # Create a DataFrame from the data
    df = pd.DataFrame(data)

    # Add jitter to the data
    jitter_strength = 0.01  # Adjust the jitter strength as needed
    df["xsize_jitter"] = df["xsize"] + np.random.uniform(
        -jitter_strength, jitter_strength, df.shape[0])
    df["ysize_jitter"] = df["ysize"] + np.random.uniform(
        -jitter_strength, jitter_strength, df.shape[0])
    df["zsize_jitter"] = df["zsize"] + np.random.uniform(
        -jitter_strength, jitter_strength, df.shape[0])
    df["mass_jitter"] = df["mass"] + np.random.uniform(
        -jitter_strength, jitter_strength, df.shape[0])

    # Create a parallel coordinate plot
    fig_parallel = px.parallel_coordinates(
        df,
        dimensions=["xsize", "ysize", "zsize", "mass"],
        color="success_rate",
        color_continuous_scale=px.colors.sequential.Viridis)

    # Show the parallel coordinate plot
    fig_parallel.show()

    # Aggregate data to calculate the mean success rate for each combination of variables
    agg_df = df.groupby(["xsize", "ysize", "zsize",
                         "mass"]).agg(avg_success_rate=("success_rate",
                                                        "mean")).reset_index()

    # Add jitter to the aggregated data
    agg_df["xsize_jitter"] = agg_df["xsize"] + np.random.uniform(
        -jitter_strength, jitter_strength, agg_df.shape[0])
    agg_df["ysize_jitter"] = agg_df["ysize"] + np.random.uniform(
        -jitter_strength, jitter_strength, agg_df.shape[0])
    agg_df["zsize_jitter"] = agg_df["zsize"] + np.random.uniform(
        -jitter_strength, jitter_strength, agg_df.shape[0])
    agg_df["mass_jitter"] = agg_df["mass"] + np.random.uniform(
        -jitter_strength, jitter_strength, agg_df.shape[0])

    # Create a scatter plot matrix with the aggregated data and jitter
    fig_matrix = px.scatter_matrix(
        agg_df,
        dimensions=[
            "xsize_jitter", "ysize_jitter", "zsize_jitter", "mass_jitter"
        ],
        color="avg_success_rate",
        color_continuous_scale=px.colors.sequential.Viridis)

    # Show the scatter plot matrix
    fig_matrix.show()

    # Function to create and show plots for average success rate as a function of a variable
    def plot_avg_success_rate(df, variable):
        avg_success_rate_df = df.groupby(
            variable)["success_rate"].mean().reset_index()
        fig = px.line(avg_success_rate_df,
                      x=variable,
                      y="success_rate",
                      title=f"Average Success Rate vs {variable.capitalize()}")
        fig.update_layout(yaxis_range=[0, 1])
        fig.show()

    # Plot average success rate as a function of xsize, ysize, zsize, and mass
    plot_avg_success_rate(df, "xsize")
    plot_avg_success_rate(df, "ysize")
    plot_avg_success_rate(df, "zsize")
    plot_avg_success_rate(df, "mass")


def plot_correlation(correlation, runid):
    # Create a heatmap of the correlation matrix
    fig = px.imshow(correlation,
                    color_continuous_scale=px.colors.sequential.Bluered,
                    title=f"Correlation Matrix for {runid}")
    fig.show()


#get all the data files in this directory using glob
data_files = glob.glob("data/*.txt")

for file_path in data_files:
    # Initialize variables to store the total success rate and the number of trials
    total_success_rate = 0
    num_trials = 0
    data = []

    # Read the JSON data from the file
    with open(file_path, "r") as f:
        for line in f:
            trial_data = json.loads(line)
            total_success_rate += trial_data["success_rate"]
            num_trials += 1
            data.append(trial_data)

    # Calculate the average success rate
    average_success_rate = total_success_rate / num_trials if num_trials > 0 else 0

    # Print the average success rate
    print(
        f"{file_path}: Average success rate over {num_trials} trials: {average_success_rate}"
    )

    #perform correlation analysis on over all trials against sizes and mass
    data = pd.DataFrame(data)

    correlation_matrix = data.corr()

    plot_correlation(correlation_matrix, os.path.basename(file_path))
