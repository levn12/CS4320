#!/usr/bin/env python3
"""
local_starter.py - Test script to verify ML development environment setup
This script demonstrates that pandas, matplotlib, and numpy are correctly installed.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def main():
    print("=" * 60)
    print("ML Development Environment Test Script")
    print("=" * 60)
    print()

    # Step 1: Load CSV with pandas
    csv_file = r"C:\Users\levid\School_Programming\CS4320\cs-4320-hw1\2026-20-aaa-220-Rice_Cammeo_Osmancik.csv"
    print("Loading CSV file with pandas...")
    df = pd.read_csv(csv_file)
    print(f"Successfully loaded {len(df)} rows and {len(df.columns)} columns")
    print()

    # Step 2: Show a few lines from the file
    print("First 5 rows of the dataset:")
    print(df.head())
    print()

    print("Dataset info:")
    print(df.info())
    print()

    # Step 3: Create scatter plot
    print("Creating scatter plot: Major_Axis_Length vs Minor_Axis_Length...")
    plt.figure(figsize=(10, 6))

    # Plot different classes with different colors
    for class_name in df["Class"].unique():
        class_data = df[df["Class"] == class_name]
        plt.scatter(
            class_data["Major_Axis_Length"],
            class_data["Minor_Axis_Length"],
            label=class_name,
            alpha=0.6,
        )

    plt.xlabel("Major Axis Length")
    plt.ylabel("Minor Axis Length")
    plt.title("Rice Grain Dimensions: Major vs Minor Axis Length")
    plt.legend()
    plt.grid(True, alpha=0.3)

    output_file = "rice_scatter_plot.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Scatter plot saved as {output_file}")
    print()

    # Step 4: Aggregate computation - mean and std dev of Perimeter for Class Cammeo
    print("Aggregate Statistics:")
    print("-" * 40)
    cammeo_data = df[df["Class"] == "Cammeo"]
    perimeter_mean = cammeo_data["Perimeter"].mean()
    perimeter_std = cammeo_data["Perimeter"].std()

    print(f"Class: Cammeo")
    print(f"  Perimeter Mean: {perimeter_mean:.2f}")
    print(f"  Perimeter Std Dev: {perimeter_std:.2f}")
    print(f"  Sample Count: {len(cammeo_data)}")
    print()

    print("=" * 60)
    print("Test completed successfully! ✓")
    print("Your ML development environment is properly configured.")
    print("=" * 60)


if __name__ == "__main__":
    main()
