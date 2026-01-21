import pandas as pd
import kagglehub
from pathlib import Path


def main():
    print("=" * 60)
    print("ML Dataset Loading Test Script")
    print("=" * 60)
    print()

    # Step 1: Load CSV with pandas
    # Download latest version
    path = kagglehub.dataset_download("esathyaprakash/electrical-fault-detection-and-classification")

    print("Path to dataset files:", path)

    # Using the path, navigate to where the csv is stored.
    dataset_dir = Path(path)

    # Find CSV file -- looks for the first csv file in given directory
    csv_file = next(dataset_dir.glob("*.csv"))

    print("Loading CSV:", csv_file)

    df = pd.read_csv(csv_file)
    print(f"Successfully loaded {len(df)} rows and {len(df.columns)} columns")

    # Step 2: Show a few lines from the file
    print("First 5 rows of the dataset:")
    print(df.head())
    
    print()

    print("Dataset info:")
    print(df.info())
    print()

    print("=" * 60)
    print("Test completed successfully! ✓")
    print("=" * 60)


if __name__ == "__main__":
    main()
