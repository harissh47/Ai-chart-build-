from transformers import GPT2LMHeadModel, GPT2Tokenizer, TextDataset, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
import pandas as pd
import json
import os
import logging
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, List, Dict
from datetime import datetime
import argparse
import sys

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def prepare_training_data(csv_file: str):
    """Prepare training data from the dataset"""
    logging.info('Preparing training data from %s', csv_file)
    df = pd.read_csv(csv_file)
    
    # Analyze dataset structure
    data_info = {
        "columns": list(df.columns),
        "numeric_columns": df.select_dtypes(include=['int64', 'float64']).columns.tolist(),
        "categorical_columns": df.select_dtypes(include=['object']).columns.tolist(),
        "sample_data": df.head(10).to_dict('records')
    }
    
    # Create training examples for different chart types
    training_examples = []
    
    # Pie chart examples
    for col in data_info['categorical_columns']:
        example = {
            "input": f"Create a pie chart showing distribution of {col}",
            "chart_type": "pie",
            "sql_query": f"SELECT {col}, COUNT(*) as count FROM data_table GROUP BY {col} ORDER BY count DESC",
            "visualization": "pie_chart"
        }
        training_examples.append(example)
    
    # Bar chart examples
    for col in data_info['categorical_columns']:
        example = {
            "input": f"Show bar chart of {col} counts",
            "chart_type": "bar",
            "sql_query": f"SELECT {col}, COUNT(*) as count FROM data_table GROUP BY {col} ORDER BY count DESC",
            "visualization": "bar_chart"
        }
        training_examples.append(example)
    
    # Line chart examples
    for num_col in data_info['numeric_columns']:
        example = {
            "input": f"Display line chart of {num_col} trend",
            "chart_type": "line",
            "sql_query": f"SELECT {num_col}, COUNT(*) as count FROM data_table GROUP BY {num_col} ORDER BY {num_col}",
            "visualization": "line_chart"
        }
        training_examples.append(example)
    
    return training_examples

def format_training_text(example):
    """Format training example into text format"""
    return f"""
Input: {example['input']}
Chart Type: {example['chart_type']}
SQL Query: {example['sql_query']}
Visualization: {example['visualization']}
---
"""

def train_model(csv_file: str, output_dir: str):
    """Train the model on the dataset"""
    logging.info('Starting model training with data from %s', csv_file)
    # Load pre-trained model and tokenizer
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    
    # Prepare training data
    training_examples = prepare_training_data(csv_file)
    
    # Create training text file
    training_text = ""
    for example in training_examples:
        training_text += format_training_text(example)
    
    # Save training text
    with open('training_data.txt', 'w') as f:
        f.write(training_text)
    
    # Create dataset
    dataset = TextDataset(
        tokenizer=tokenizer,
        file_path='training_data.txt',
        block_size=128
    )
    
    # Create data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    # Set up training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        save_steps=500,
        save_total_limit=2,
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset,
    )
    
    # Train the model
    trainer.train()
    
    # Save the trained model
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    return model, tokenizer

class LocationAnalyzer:
    def __init__(self):
        self.scaler = StandardScaler()
        self.kmeans = None
        self.data = None
        self.lat_col = None
        self.lon_col = None
        self.category_col = None
        
    def detect_columns(self, df: pd.DataFrame) -> Tuple[str, str, str]:
        """Automatically detect latitude, longitude and category columns"""
        # Look for common latitude column names
        lat_columns = [col for col in df.columns if any(name in col.lower() for name in ['lat', 'latitude'])]
        # Look for common longitude column names
        lon_columns = [col for col in df.columns if any(name in col.lower() for name in ['lon', 'longitude'])]
        
        if not lat_columns or not lon_columns:
            # If no lat/lon columns found, look for numeric columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) >= 2:
                lat_columns = [numeric_cols[0]]
                lon_columns = [numeric_cols[1]]
            else:
                raise ValueError("Could not detect latitude and longitude columns")
        
        # Find a suitable category column (string type, not lat/lon)
        category_candidates = [col for col in df.columns 
                             if col not in lat_columns + lon_columns 
                             and df[col].dtype == 'object'
                             and df[col].nunique() < len(df) * 0.5]  # Less than 50% unique values
        
        category_col = category_candidates[0] if category_candidates else None
        
        return lat_columns[0], lon_columns[0], category_col
        
    def load_data(self, file_path: str) -> pd.DataFrame:
        """Load and prepare dataset"""
        try:
            # Try to determine file type from extension
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
            elif file_path.endswith('.xlsx') or file_path.endswith('.xls'):
                df = pd.read_excel(file_path)
            else:
                raise ValueError("Unsupported file format. Please use CSV or Excel files.")
            
            # Detect relevant columns
            self.lat_col, self.lon_col, self.category_col = self.detect_columns(df)
            logging.info(f"Detected columns - Latitude: {self.lat_col}, Longitude: {self.lon_col}, Category: {self.category_col}")
            
            # Clean the data
            df = df.dropna(subset=[self.lat_col, self.lon_col])
            self.data = df
            return df
            
        except Exception as e:
            logging.error(f"Error loading data: {str(e)}")
            sys.exit(1)
    
    def train_clusters(self, n_clusters: int = 5) -> np.ndarray:
        """Train KMeans clustering on location data"""
        locations = self.data[[self.lat_col, self.lon_col]].values
        scaled_locations = self.scaler.fit_transform(locations)
        
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = self.kmeans.fit_predict(scaled_locations)
        return clusters
    
    def create_density_heatmap(self, save_path: str = None) -> None:
        """Create a heatmap showing location density"""
        plt.figure(figsize=(15, 10))
        
        plt.hist2d(self.data[self.lon_col], self.data[self.lat_col], 
                  bins=50, cmap='YlOrRd')
        
        plt.colorbar(label='Number of Points')
        plt.title('Location Density Heatmap')
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()
    
    def create_cluster_map(self, save_path: str = None) -> None:
        """Create a scatter plot with cluster assignments"""
        if self.kmeans is None:
            self.train_clusters()
        
        plt.figure(figsize=(15, 10))
        scatter = plt.scatter(self.data[self.lon_col], self.data[self.lat_col], 
                            c=self.kmeans.labels_, cmap='viridis', alpha=0.6)
        plt.colorbar(scatter, label='Cluster')
        plt.title('Location Clusters')
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()
    
    def create_category_distribution(self, save_path: str = None) -> None:
        """Create a bar plot showing distribution across categories"""
        if not self.category_col:
            logging.warning("No suitable category column found. Skipping category distribution plot.")
            return
            
        plt.figure(figsize=(15, 8))
        category_counts = self.data[self.category_col].value_counts()
        category_counts[:15].plot(kind='bar')
        plt.title(f'Distribution by {self.category_col}')
        plt.xlabel(self.category_col)
        plt.ylabel('Count')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()
    
    def analyze_and_visualize(self) -> Dict[str, str]:
        """Run complete analysis and generate all visualizations"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs('charts', exist_ok=True)
        
        charts = {}
        
        # Generate heatmap
        heatmap_path = f'charts/density_heatmap_{timestamp}.png'
        self.create_density_heatmap(heatmap_path)
        charts['heatmap'] = heatmap_path
        
        # Generate cluster map
        cluster_path = f'charts/location_clusters_{timestamp}.png'
        self.create_cluster_map(cluster_path)
        charts['clusters'] = cluster_path
        
        # Generate category distribution if possible
        if self.category_col:
            dist_path = f'charts/category_distribution_{timestamp}.png'
            self.create_category_distribution(dist_path)
            charts['category_distribution'] = dist_path
        
        return charts

def main():
    """Main function to run the analysis"""
    parser = argparse.ArgumentParser(description='Analyze and visualize location data')
    parser.add_argument('file_path', help='Path to the data file (CSV or Excel)')
    parser.add_argument('--clusters', type=int, default=5, help='Number of clusters for K-means (default: 5)')
    
    args = parser.parse_args()
    
    logging.info(f"Loading data from: {args.file_path}")
    
    analyzer = LocationAnalyzer()
    analyzer.load_data(args.file_path)
    chart_paths = analyzer.analyze_and_visualize()
    
    print("\nAnalysis complete! Generated the following visualizations:")
    for chart_type, path in chart_paths.items():
        print(f"- {chart_type}: {path}")

if __name__ == "__main__":
    csv_file = "data/mlb_players.csv"
    output_dir = "trained_model"
    
    print("Starting model training...")
    model, tokenizer = train_model(csv_file, output_dir)
    print("Model training completed!") 

    main()