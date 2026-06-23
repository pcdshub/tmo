#!/usr/bin/python

import numpy as np
import matplotlib.pyplot as plt
import argparse
import json

# Set up the argument parser
parser = argparse.ArgumentParser(description="Process a JSON file.")
parser.add_argument("json_file", type=str, help="Path to the JSON file")

# Parse arguments
args = parser.parse_args()

# Load and print the JSON content
with open(args.json_file, 'r') as f:
    data = json.load(f)
    x_points = [str(p) for p in np.linspace(10.635, 10.935, 2)]
    z_points = [str(p) for p in np.linspace(0.436, 0.736, 2)]
    pressure = []
    for dp in data["data_points"]:
        pressure.append(dp["avg_pressure"])
    pressure = np.reshape(pressure, (len(x_points), len(z_points)))
    fig, ax = plt.subplots()
    heatmap = ax.imshow(pressure, cmap='viridis', origin='upper')

    # Set ticks and labels
    ax.set_xticks(np.arange(len(x_points)))
    ax.set_yticks(np.arange(len(z_points)))
    ax.set_xticklabels(x_points)
    ax.set_yticklabels(z_points)

    # Rotate the tick labels and show values on cells
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    for i in range(len(z_points)):
        for j in range(len(x_points)):
            ax.text(j, i, "{:.3e}".format(pressure[i, j]), ha='center', va='center', color='w')

    plt.colorbar(heatmap)
    plt.xlabel("X-axis")
    plt.ylabel("Z-axis")
    plt.title("Nozzle jet scan via RGA")
    plt.tight_layout()
    plt.show()
