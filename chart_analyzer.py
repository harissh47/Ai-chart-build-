import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime
import logging
from typing import List, Dict
from enum import StrEnum

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ChartType(StrEnum):
    bar = "bar"
    line = "line"
    scatter = "scatter"
    pie = "pie"
    doughnut = "doughnut"
    bubble = "bubble"
    radar = "radar"
    polarArea = "polarArea"

class ChartGenerator:
    def __init__(self):
        """Initialize chart generator"""
        self.charts_dir = "charts"
        os.makedirs(self.charts_dir, exist_ok=True)
        self.default_colors = [
            "#FF6384", "#36A2EB", "#FFCE56", "#4BC0C0", "#9966FF",
            "#FF9F40", "#4BC0C0", "#9966FF", "#C9CBCF", "#36A2EB"
        ]

    def analyze_and_generate_charts(self, df: pd.DataFrame) -> List[Dict]:
        """Analyze data and generate appropriate charts"""
        charts = []
        
        # Generate different types of charts based on data types
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
        categorical_cols = df.select_dtypes(include=['object']).columns
        datetime_cols = df.select_dtypes(include=['datetime64']).columns
        
        # Time series charts
        if len(datetime_cols) > 0 and len(numeric_cols) > 0:
            for num_col in numeric_cols:
                plt.figure(figsize=(12, 7))
                plt.plot(df[datetime_cols[0]], df[num_col], marker='o')
                plt.title(f"{num_col} over Time")
                plt.xticks(rotation=45)
                
                # Save chart
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"line_chart_{num_col}_{timestamp}.png"
                plt.savefig(os.path.join(self.charts_dir, filename))
                plt.close()
                
                charts.append({
                    "filename": filename,
                    "type": "line",
                    "reason": "Time series analysis",
                    "columns": [datetime_cols[0], num_col]
                })
        
        # Bar charts for categorical data
        if len(categorical_cols) > 0 and len(numeric_cols) > 0:
            for cat_col in categorical_cols:
                for num_col in numeric_cols:
                    plt.figure(figsize=(12, 7))
                    sns.barplot(data=df, x=cat_col, y=num_col)
                    plt.title(f"{num_col} by {cat_col}")
                    plt.xticks(rotation=45)
                    
                    # Save chart
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"bar_chart_{cat_col}_{num_col}_{timestamp}.png"
                    plt.savefig(os.path.join(self.charts_dir, filename))
                    plt.close()
                    
                    charts.append({
                        "filename": filename,
                        "type": "bar",
                        "reason": "Category comparison",
                        "columns": [cat_col, num_col]
                    })
        
        # Scatter plots for numeric correlations
        if len(numeric_cols) >= 2:
            for i, col1 in enumerate(numeric_cols[:-1]):
                for col2 in numeric_cols[i+1:]:
                    plt.figure(figsize=(12, 7))
                    plt.scatter(df[col1], df[col2], alpha=0.5)
                    plt.xlabel(col1)
                    plt.ylabel(col2)
                    plt.title(f"Correlation between {col1} and {col2}")
                    
                    # Save chart
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"scatter_plot_{col1}_{col2}_{timestamp}.png"
                    plt.savefig(os.path.join(self.charts_dir, filename))
                    plt.close()
                    
                    charts.append({
                        "filename": filename,
                        "type": "scatter",
                        "reason": "Numeric correlation analysis",
                        "columns": [col1, col2]
                    })
        
        # Pie charts for categorical distributions
        if len(categorical_cols) > 0:
            for cat_col in categorical_cols:
                value_counts = df[cat_col].value_counts()
                if len(value_counts) <= 10:  # Only create pie charts for reasonable number of categories
                    plt.figure(figsize=(12, 7))
                    plt.pie(value_counts, labels=value_counts.index, autopct='%1.1f%%',
                           colors=self.default_colors[:len(value_counts)])
                    plt.title(f"Distribution of {cat_col}")
                    
                    # Save chart
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"pie_chart_{cat_col}_{timestamp}.png"
                    plt.savefig(os.path.join(self.charts_dir, filename))
                    plt.close()
                    
                    charts.append({
                        "filename": filename,
                        "type": "pie",
                        "reason": "Category distribution",
                        "columns": [cat_col]
                    })
        
        if not charts:
            self._generate_fallback_chart(df)
        
        return charts

    def _generate_fallback_chart(self, df: pd.DataFrame):
        """Generate a basic fallback chart when other attempts fail"""
        try:
            plt.figure(figsize=(12, 7))
            if len(df.columns) >= 2:
                sns.barplot(data=df, x=df.columns[0], y=df.columns[1])
                plt.xticks(rotation=45)
                plt.title(f"Fallback Chart: {df.columns[0]} vs {df.columns[1]}")
                
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"fallback_chart_{timestamp}.png"
                os.makedirs(self.charts_dir, exist_ok=True)
                plt.savefig(os.path.join(self.charts_dir, filename))
                plt.close()
                
                logging.info(f"Generated fallback chart: {filename}")
            else:
                logging.warning("Cannot generate fallback chart: insufficient columns")
        except Exception as e:
            logging.error(f"Error generating fallback chart: {str(e)}")
