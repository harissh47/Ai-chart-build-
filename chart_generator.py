import json
import logging
from typing import List, Dict, Optional
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
from data_analyzer import DataAnalyzer, ChartType

class ChartGenerator:
    def __init__(self):
        self.data_analyzer = DataAnalyzer()
        self.default_colors = [
            "#FF6384", "#36A2EB", "#FFCE56", "#4BC0C0", "#9966FF",
            "#FF9F40", "#4BC0C0", "#9966FF", "#C9CBCF", "#36A2EB"
        ]

    def analyze_and_generate_charts(self, df: pd.DataFrame) -> List[Dict]:
        """Analyze data and generate appropriate charts"""
        analysis = self.data_analyzer.analyze_data(df)
        charts = []
        
        for suggestion in analysis["suggested_charts"]:
            chart_type = suggestion["type"]
            columns = suggestion["columns"]
            reason = suggestion["reason"]
            
            try:
                # Create visualization
                plt.figure(figsize=(12, 7))
                self._create_chart(df[columns], chart_type, reason)
                
                # Save the chart
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"chart_{chart_type}_{timestamp}.png"
                os.makedirs("charts", exist_ok=True)
                plt.savefig(os.path.join("charts", filename))
                plt.close()
                
                charts.append({
                    "filename": filename,
                    "type": chart_type,
                    "reason": reason,
                    "columns": columns
                })
                
                logging.info(f"Generated {chart_type} chart: {filename}")
                logging.info(f"Reason: {reason}")
                logging.info(f"Columns used: {', '.join(columns)}")
                
            except Exception as e:
                logging.error(f"Error generating {chart_type} chart: {str(e)}")
                continue
        
        if not charts:
            self._generate_fallback_chart(df)
        
        return charts

    def _create_chart(self, df: pd.DataFrame, chart_type: ChartType, title: str):
        """Create specific type of chart"""
        if chart_type == ChartType.bar:
            sns.barplot(data=df, x=df.columns[0], y=df.columns[1])
            plt.xticks(rotation=45)
            
        elif chart_type == ChartType.line:
            plt.plot(df[df.columns[0]], df[df.columns[1]], marker='o')
            plt.xticks(rotation=45)
            
        elif chart_type == ChartType.scatter:
            sns.scatterplot(data=df, x=df.columns[0], y=df.columns[1])
            
        elif chart_type == ChartType.pie:
            plt.pie(df[df.columns[1]], labels=df[df.columns[0]], autopct='%1.1f%%',
                   colors=self.default_colors[:len(df)])
            
        elif chart_type == ChartType.bubble:
            plt.scatter(df[df.columns[0]], df[df.columns[1]], 
                       s=df[df.columns[2]]*100, alpha=0.5,
                       c=self.default_colors[0])
            
        elif chart_type == ChartType.radar:
            # Implement radar chart
            pass
            
        plt.title(title)
        plt.tight_layout()

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
                os.makedirs("charts", exist_ok=True)
                plt.savefig(os.path.join("charts", filename))
                plt.close()
                
                logging.info(f"Generated fallback chart: {filename}")
            else:
                logging.warning("Cannot generate fallback chart: insufficient columns")
        except Exception as e:
            logging.error(f"Error generating fallback chart: {str(e)}")
