import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from datetime import datetime
import logging
import argparse
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DataVisualizer:
    def __init__(self, dataset_path):
        """Initialize the visualizer with a dataset path."""
        self.dataset_path = dataset_path
        self.data = None
        self.numeric_columns = []
        self.categorical_columns = []
        self.datetime_columns = []
        self.charts_dir = 'charts'
        os.makedirs(self.charts_dir, exist_ok=True)

    def load_data(self):
        """Load and analyze the dataset."""
        try:
            # Try to infer data types automatically
            self.data = pd.read_csv(self.dataset_path)
            
            # Clean column names
            self.data.columns = self.data.columns.str.strip().str.strip('"').str.strip()
            
            # Analyze column types
            for column in self.data.columns:
                if pd.api.types.is_numeric_dtype(self.data[column]):
                    self.numeric_columns.append(column)
                else:
                    # Try to convert to datetime
                    try:
                        pd.to_datetime(self.data[column])
                        self.datetime_columns.append(column)
                    except:
                        self.categorical_columns.append(column)
            
            # Print dataset description
            print("\nDataset Overview:")
            print(f"Dataset Name: {os.path.basename(self.dataset_path)}")
            print(f"Memory Usage: {self.data.memory_usage().sum() / 1024**2:.2f} MB")
            print("\nFirst few rows of the dataset:")
            print(self.data.head())
            
            print("\nGeneral Information:")
            print(f"Missing Values:")
            for column in self.data.columns:
                missing = self.data[column].isnull().sum()
                if missing > 0:
                    print(f"  - {column}: {missing} missing values")
            
            print("\nDataset Analysis:")
            print(f"Total Rows: {len(self.data)}")
            print(f"Total Columns: {len(self.data.columns)}")
            print(f"\nColumns by Type:")
            print(f"Numeric Columns: {', '.join(self.numeric_columns)}")
            print(f"Categorical Columns: {', '.join(self.categorical_columns)}")
            print(f"DateTime Columns: {', '.join(self.datetime_columns)}")
            
            return True
        except Exception as e:
            print(f"Error loading dataset: {str(e)}")
            return False

    def generate_visualizations(self):
        """Generate appropriate visualizations based on data types."""
        charts = []

        # 1. Distribution plots for numeric columns
        for col in self.numeric_columns:
            try:
                plt.figure(figsize=(10, 6))
                sns.histplot(data=self.data, x=col, kde=True)
                plt.title(f'Distribution of {col}')
                plt.xticks(rotation=45)
                plt.tight_layout()
                filename = f'distribution_{col}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
                plt.savefig(os.path.join(self.charts_dir, filename))
                plt.close()
                charts.append({
                    'type': 'distribution',
                    'filename': filename,
                    'columns': [col]
                })
            except Exception as e:
                print(f"Error generating distribution plot for {col}: {str(e)}")

        # 2. Bar plots for categorical columns
        for col in self.categorical_columns:
            try:
                if self.data[col].nunique() <= 30:  # Only if there aren't too many categories
                    plt.figure(figsize=(12, 6))
                    value_counts = self.data[col].value_counts()
                    plt.bar(value_counts.index[:15], value_counts.values[:15])
                    plt.title(f'Top 15 Categories in {col}')
                    plt.xticks(rotation=45, ha='right')
                    plt.tight_layout()
                    filename = f'categorical_{col}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
                    plt.savefig(os.path.join(self.charts_dir, filename))
                    plt.close()
                    charts.append({
                        'type': 'bar',
                        'filename': filename,
                        'columns': [col]
                    })
            except Exception as e:
                print(f"Error generating bar plot for {col}: {str(e)}")

        # 3. Correlation heatmap for numeric columns
        if len(self.numeric_columns) > 1:
            try:
                plt.figure(figsize=(10, 8))
                correlation = self.data[self.numeric_columns].corr()
                sns.heatmap(correlation, annot=True, cmap='coolwarm', center=0)
                plt.title('Correlation Heatmap')
                plt.tight_layout()
                filename = f'correlation_heatmap_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
                plt.savefig(os.path.join(self.charts_dir, filename))
                plt.close()
                charts.append({
                    'type': 'heatmap',
                    'filename': filename,
                    'columns': self.numeric_columns
                })
            except Exception as e:
                print(f"Error generating correlation heatmap: {str(e)}")

        # 4. Scatter plots for pairs of numeric columns (up to 3 pairs)
        if len(self.numeric_columns) >= 2:
            pairs = list(zip(self.numeric_columns[:-1], self.numeric_columns[1:]))[:3]
            for x_col, y_col in pairs:
                try:
                    plt.figure(figsize=(10, 6))
                    plt.scatter(self.data[x_col], self.data[y_col], alpha=0.5)
                    plt.xlabel(x_col)
                    plt.ylabel(y_col)
                    plt.title(f'{x_col} vs {y_col}')
                    plt.tight_layout()
                    filename = f'scatter_{x_col}_vs_{y_col}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
                    plt.savefig(os.path.join(self.charts_dir, filename))
                    plt.close()
                    charts.append({
                        'type': 'scatter',
                        'filename': filename,
                        'columns': [x_col, y_col]
                    })
                except Exception as e:
                    print(f"Error generating scatter plot for {x_col} vs {y_col}: {str(e)}")

        return charts

    def print_summary_statistics(self):
        """Print summary statistics for the dataset."""
        print("\nSummary Statistics:")
        
        # Numeric columns
        if self.numeric_columns:
            print("\nNumeric Columns:")
            print(self.data[self.numeric_columns].describe())
        
        # Categorical columns
        if self.categorical_columns:
            print("\nCategorical Columns:")
            for col in self.categorical_columns:
                unique_values = self.data[col].nunique()
                print(f"\n{col}:")
                print(f"Unique values: {unique_values}")
                if unique_values <= 10:
                    print("Value counts:")
                    print(self.data[col].value_counts().head())

def main():
    """Main function to run the visualization tool."""
    parser = argparse.ArgumentParser(description='Generate visualizations from a dataset.')
    parser.add_argument('dataset_path', type=str, help='Path to the dataset (CSV file)')
    args = parser.parse_args()

    # Validate file path
    if not os.path.exists(args.dataset_path):
        print(f"Error: File not found at {args.dataset_path}")
        return

    if not args.dataset_path.lower().endswith('.csv'):
        print("Error: Please provide a CSV file")
        return

    try:
        # Initialize visualizer
        visualizer = DataVisualizer(args.dataset_path)
        
        # Load and analyze data
        if not visualizer.load_data():
            return

        # Generate visualizations
        print("\nGenerating visualizations...")
        charts = visualizer.generate_visualizations()

        # Print results
        if charts:
            print("\nGenerated Visualizations:")
            for chart in charts:
                print(f"\nType: {chart['type']}")
                print(f"Filename: {chart['filename']}")
                print(f"Columns used: {', '.join(chart['columns'])}")
            
            print(f"\nAll visualizations have been saved to the '{visualizer.charts_dir}' directory")
        else:
            print("\nNo visualizations were generated")

        # Print summary statistics
        visualizer.print_summary_statistics()

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        logging.error(f"Error in main function: {str(e)}")

if __name__ == "__main__":
    main()
