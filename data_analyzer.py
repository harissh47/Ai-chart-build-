import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from enum import StrEnum

class ChartType(StrEnum):
    bar = "bar"
    line = "line"
    scatter = "scatter"
    pie = "pie"
    doughnut = "doughnut"
    bubble = "bubble"
    radar = "radar"
    polarArea = "polarArea"

class DataAnalyzer:
    def __init__(self):
        self.correlation_threshold = 0.7
        self.categorical_threshold = 10  # Max unique values to consider categorical
        
    def analyze_data(self, df: pd.DataFrame) -> Dict:
        """Analyze dataset and suggest appropriate visualizations"""
        analysis = {
            "row_count": len(df),
            "column_count": len(df.columns),
            "columns": {},
            "relationships": [],
            "suggested_charts": []
        }
        
        # Analyze each column
        for col in df.columns:
            col_analysis = self._analyze_column(df[col])
            analysis["columns"][col] = col_analysis
            
        # Analyze relationships between columns
        if len(df.columns) >= 2:
            analysis["relationships"] = self._analyze_relationships(df)
            
        # Suggest appropriate charts
        analysis["suggested_charts"] = self._suggest_charts(df, analysis)
        
        return analysis
    
    def _analyze_column(self, series: pd.Series) -> Dict:
        """Analyze a single column"""
        unique_count = series.nunique()
        is_numeric = pd.api.types.is_numeric_dtype(series)
        
        analysis = {
            "type": "numeric" if is_numeric else "categorical",
            "unique_count": unique_count,
            "null_count": series.isnull().sum(),
            "is_temporal": pd.api.types.is_datetime64_any_dtype(series)
        }
        
        if is_numeric:
            analysis.update({
                "min": float(series.min()) if not pd.isna(series.min()) else None,
                "max": float(series.max()) if not pd.isna(series.max()) else None,
                "mean": float(series.mean()) if not pd.isna(series.mean()) else None,
                "distribution": "discrete" if series.dtype in [np.int64, np.int32] else "continuous"
            })
        
        return analysis
    
    def _analyze_relationships(self, df: pd.DataFrame) -> List[Dict]:
        """Analyze relationships between columns"""
        relationships = []
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        # Analyze correlations between numeric columns
        if len(numeric_cols) >= 2:
            corr_matrix = df[numeric_cols].corr()
            for i in range(len(numeric_cols)):
                for j in range(i+1, len(numeric_cols)):
                    corr = corr_matrix.iloc[i, j]
                    if abs(corr) > self.correlation_threshold:
                        relationships.append({
                            "type": "correlation",
                            "columns": [numeric_cols[i], numeric_cols[j]],
                            "strength": float(corr)
                        })
        
        return relationships
    
    def _suggest_charts(self, df: pd.DataFrame, analysis: Dict) -> List[Dict]:
        """Suggest appropriate chart types based on data analysis"""
        suggestions = []
        col_count = len(df.columns)
        
        if col_count < 2:
            return suggestions
            
        # Get column types
        numeric_cols = [col for col, info in analysis["columns"].items() 
                       if info["type"] == "numeric"]
        categorical_cols = [col for col, info in analysis["columns"].items() 
                          if info["type"] == "categorical"]
        
        # Always suggest pie chart first if we have at least one numeric and one categorical column
        if numeric_cols and categorical_cols:
            suggestions.append({
                "type": ChartType.pie,
                "columns": [categorical_cols[0], numeric_cols[0]],
                "confidence": 1.0,
                "reason": "Default visualization type for categorical and numeric data"
            })
        
        # Add other chart suggestions as fallbacks
        if len(numeric_cols) >= 2:
            suggestions.append({
                "type": ChartType.scatter,
                "columns": numeric_cols[:2],
                "confidence": 0.8,
                "reason": "Relationship between numeric variables"
            })
        
        if categorical_cols and numeric_cols:
            suggestions.append({
                "type": ChartType.bar,
                "columns": [categorical_cols[0], numeric_cols[0]],
                "confidence": 0.7,
                "reason": "Compare values across categories"
            })
        
        return suggestions
