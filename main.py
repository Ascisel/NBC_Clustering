from tools import load_dataframe, code_nominal, split_labels_from_features, remove_nans
import argparse
from nbc import NBC
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
import time
import os
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-k', '--k_value', type=int, required=True)
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--target_column', type=str, required=False, default=None)
    parser.add_argument('--plot', type=bool, default=False, required=False)
    parser.add_argument('--n_rows', type=int, required=False)
    parser.add_argument('--n_cols', type=int, required=False, default=None)

    return parser.parse_args()


def print_plot(numeric_features, clusters, data_path, k_value):
    directory, file_name = os.path.split(data_path)
    file_name_no_extension, file_extension = os.path.splitext(file_name)

    x = numeric_features[:, 0]
    y = numeric_features[:, 1]
    df = pd.DataFrame({'x': x, 'y': y, 'class': clusters})
    
    classes = df['class'].unique()
    class_colors = sns.color_palette("husl", n_colors=len(classes))

    # Create a dictionary to map classes to colors
    class_color_dict = dict(zip(classes, class_colors))
    
    colors = df['class'].map(class_color_dict)

    if numeric_features.shape[1]==3:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        df['z'] = numeric_features[:, 2]
        ax.scatter(df['x'], df['y'], df['z'], s=5, color=colors)
        # Set labels
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
        
    else:
        plt.scatter(df['x'], df['y'], s=5, color=colors)

        # Add labels and legend
        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')
        handlelist = [plt.plot([], marker="o", ls="", color=color)[0] for color in class_colors]
        plt.legend(handlelist, classes)

        # Show the plot
    plt.title(f'Plot for {file_name_no_extension} dataset, k={k_value}')
    plt.savefig(f'plot_{file_name_no_extension}_k{k_value}.png')
    plt.show()
    

def run():
    args = parse_args()

    df = load_dataframe(args.data_path, args.target_column)
    df = remove_nans(df)
    target_column = args.target_column

    if target_column:
        features, labels = split_labels_from_features(df, args.target_column)
    else:
        features = df
        labels = None

    numeric_features = code_nominal(features)

    if args.n_rows is not None and args.n_rows <= numeric_features.shape[0]:
        numeric_features = numeric_features.iloc[:args.n_rows, :]
    if args.n_cols is not None and args.n_cols <= numeric_features.shape[0]:
        numeric_features = numeric_features.iloc[:, :args.n_cols]

    numeric_features = numeric_features.values


    start_time = time.time()

    nbc = NBC(args.k_value, numeric_features)
    clusters = nbc.run()

    end_time = time.time()

    runtime = end_time-start_time
    print(f'Elapsed time for NBC algorithm: {runtime:.3f}s')

    print(f'Clusters get: {set(clusters)}')
    if labels is not None:
        print(f'Classes in original data: {sorted(labels.unique())}')

    if numeric_features.shape[1] in [2, 3] and args.plot==True:
        print_plot(numeric_features, clusters, args.data_path, args.k_value)


    return clusters, runtime


if __name__ == '__main__':
    run()

