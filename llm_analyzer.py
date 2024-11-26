import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, List, Dict, Optional
import logging
import os
from datetime import datetime
import openai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class LLMDataAnalyzer:
    def __init__(self):
        self.data = None
        self.columns = None
        self.analysis = None
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key not found. Please set OPENAI_API_KEY in .env file")
        openai.api_key = self.api_key

    def load_data(self, file_path: str) -> pd.DataFrame:
        """Load data from file and analyze its structure using LLM"""
        try:
            # Load data based on file extension
            if file_path.endswith('.csv'):
                self.data = pd.read_csv(file_path)
            elif file_path.endswith(('.xlsx', '.xls')):
                self.data = pd.read_excel(file_path)
            else:
                raise ValueError("Unsupported file format. Please use CSV or Excel files.")

            # Get data info
            self.columns = list(self.data.columns)
            data_info = self._get_data_info()
            
            # Analyze data structure using LLM
            self.analysis = self._analyze_data_structure(data_info)
            
            return self.data

        except Exception as e:
            logging.error(f"Error loading data: {str(e)}")
            raise

    def _get_data_info(self) -> str:
        """Get information about the dataset structure"""
        info = []
        info.append(f"Number of rows: {len(self.data)}")
        info.append(f"Number of columns: {len(self.columns)}")
        info.append("\nColumn information:")
        
        for col in self.columns:
            dtype = str(self.data[col].dtype)
            unique_count = self.data[col].nunique()
            sample_values = self.data[col].dropna().head(3).tolist()
            info.append(f"- {col}: {dtype}, {unique_count} unique values, samples: {sample_values}")
        
        return "\n".join(info)

    def _analyze_data_structure(self, data_info: str) -> Dict:
        """Use LLM to analyze the data structure and suggest visualizations"""
        prompt = f"""
        Analyze this dataset structure and suggest appropriate visualizations:
        {data_info}
        
        Provide response in the following JSON format:
        {{
            "coordinate_columns": {{"latitude": "col_name", "longitude": "col_name"}},
            "categorical_columns": ["col1", "col2"],
            "numerical_columns": ["col1", "col2"],
            "temporal_columns": ["col1"],
            "suggested_visualizations": [
                {{
                    "type": "visualization_type",
                    "columns": ["col1", "col2"],
                    "title": "chart_title",
                    "description": "what_this_shows"
                }}
            ]
        }}
        
        Only include columns that exist in the dataset. If certain types don't exist, use empty lists or null values.
        """

        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a data analysis expert. Analyze the dataset and suggest appropriate visualizations."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.4
            )
            
            analysis = eval(response.choices[0].message.content)
            logging.info("LLM analysis completed successfully")
            return analysis
            
        except Exception as e:
            logging.error(f"Error in LLM analysis: {str(e)}")
            return self._fallback_analysis()

    def _fallback_analysis(self) -> Dict:
        """Fallback analysis when LLM fails"""
        # Try to identify coordinate columns
        lat_cols = [col for col in self.columns if 'lat' in col.lower()]
        lon_cols = [col for col in self.columns if 'lon' in col.lower()]
        
        # Identify numerical and categorical columns
        numerical_cols = self.data.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = self.data.select_dtypes(include=['object']).columns.tolist()
        
        return {
            "coordinate_columns": {
                "latitude": lat_cols[0] if lat_cols else None,
                "longitude": lon_cols[0] if lon_cols else None
            },
            "categorical_columns": categorical_cols,
            "numerical_columns": numerical_cols,
            "temporal_columns": [],
            "suggested_visualizations": [
                {
                    "type": "scatter",
                    "columns": numerical_cols[:2] if len(numerical_cols) >= 2 else [],
                    "title": "Scatter Plot",
                    "description": "Basic relationship visualization"
                }
            ]
        }

    def create_visualization(self, viz_config: Dict, save_path: Optional[str] = None) -> None:
        """Create a visualization based on the configuration"""
        plt.figure(figsize=(12, 8))
        
        try:
            if viz_config["type"] == "scatter":
                self._create_scatter(viz_config)
            elif viz_config["type"] == "bar":
                self._create_bar(viz_config)
            elif viz_config["type"] == "line":
                self._create_line(viz_config)
            elif viz_config["type"] == "heatmap":
                self._create_heatmap(viz_config)
            elif viz_config["type"] == "histogram":
                self._create_histogram(viz_config)
            
            plt.title(viz_config["title"])
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path)
                plt.close()
            else:
                plt.show()
                
        except Exception as e:
            logging.error(f"Error creating visualization: {str(e)}")
            plt.close()

    def _create_scatter(self, config: Dict) -> None:
        """Create scatter plot"""
        plt.scatter(self.data[config["columns"][0]], 
                   self.data[config["columns"][1]], 
                   alpha=0.5)
        plt.xlabel(config["columns"][0])
        plt.ylabel(config["columns"][1])

    def _create_bar(self, config: Dict) -> None:
        """Create bar plot"""
        data = self.data[config["columns"][0]].value_counts()[:15]
        data.plot(kind='bar')
        plt.xticks(rotation=45, ha='right')

    def _create_line(self, config: Dict) -> None:
        """Create line plot"""
        self.data.plot(x=config["columns"][0], 
                      y=config["columns"][1], 
                      kind='line')

    def _create_heatmap(self, config: Dict) -> None:
        """Create heatmap"""
        if len(config["columns"]) == 2:
            # For coordinate data
            plt.hist2d(self.data[config["columns"][0]], 
                      self.data[config["columns"][1]], 
                      bins=50, 
                      cmap='YlOrRd')
            plt.colorbar(label='Count')
        else:
            # For correlation heatmap
            correlation = self.data[config["columns"]].corr()
            sns.heatmap(correlation, annot=True, cmap='coolwarm')

    def _create_histogram(self, config: Dict) -> None:
        """Create histogram"""
        self.data[config["columns"][0]].hist(bins=30)
        plt.xlabel(config["columns"][0])
        plt.ylabel('Count')

    def analyze_and_visualize(self) -> Dict[str, str]:
        """Generate all suggested visualizations"""
        if not self.analysis:
            raise ValueError("Please load and analyze data first using load_data()")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs('charts', exist_ok=True)
        
        chart_paths = {}
        
        for i, viz in enumerate(self.analysis["suggested_visualizations"]):
            chart_name = f"{viz['type']}_{timestamp}_{i}.png"
            chart_path = os.path.join('charts', chart_name)
            
            try:
                self.create_visualization(viz, chart_path)
                chart_paths[viz['title']] = {
                    'path': chart_path,
                    'description': viz['description']
                }
            except Exception as e:
                logging.error(f"Error creating {viz['type']} visualization: {str(e)}")
        
        return chart_paths

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='LLM-based Data Analysis and Visualization')
    parser.add_argument('file_path', help='Path to the data file (CSV or Excel)')
    args = parser.parse_args()
    
    try:
        analyzer = LLMDataAnalyzer()
        analyzer.load_data(args.file_path)
        chart_paths = analyzer.analyze_and_visualize()
        
        print("\nAnalysis complete! Generated the following visualizations:")
        for title, info in chart_paths.items():
            print(f"\n- {title}")
            print(f"  Path: {info['path']}")
            print(f"  Description: {info['description']}")
            
    except Exception as e:
        logging.error(f"Analysis failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
