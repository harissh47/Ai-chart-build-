import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple
import logging
import os
from datetime import datetime
import json
import requests

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DynamicAnalyzer:
    def __init__(self):
        self.data = None
        self.columns = None
        self.analysis = None
        self.models = {}
        self.encoders = {}
        self.scaler = StandardScaler()
        
        # LLM Configuration
        self.base_url = "https://infinitllmservice.sifymdp.digital/v1"
        self.model_uid = "llama-3.1-instruct"

    def load_data(self, file_path: str) -> pd.DataFrame:
        """Load and analyze data using LLM"""
        try:
            # Load data based on file extension
            if file_path.endswith('.csv'):
                self.data = pd.read_csv(file_path)
            elif file_path.endswith(('.xlsx', '.xls')):
                self.data = pd.read_excel(file_path)
            else:
                raise ValueError("Unsupported file format. Please use CSV or Excel files.")

            self.columns = list(self.data.columns)
            data_info = self._get_data_info()
            self.analysis = self._analyze_with_llm(data_info)
            
            return self.data

        except Exception as e:
            logging.error(f"Error loading data: {str(e)}")
            raise

    def _get_data_info(self) -> str:
        """Get dataset information for LLM analysis"""
        info = []
        info.append(f"Number of rows: {len(self.data)}")
        info.append(f"Number of columns: {len(self.columns)}")
        info.append("\nColumn information:")
        
        for col in self.columns:
            dtype = str(self.data[col].dtype)
            unique_count = self.data[col].nunique()
            sample_values = self.data[col].dropna().head(3).tolist()
            null_count = self.data[col].isnull().sum()
            info.append(f"- {col}: {dtype}, {unique_count} unique values, {null_count} null values, samples: {sample_values}")
        
        return "\n".join(info)

    def _analyze_with_llm(self, data_info: str) -> Dict:
        """Use LLM to analyze data and suggest training/visualization approach"""
        prompt = f"""
        Analyze this dataset and suggest machine learning approach and visualizations:
        {data_info}
        
        Provide response in the following JSON format:
        {{
            "features": {{
                "numerical": ["col1", "col2"],
                "categorical": ["col1", "col2"],
                "target": "target_column",
                "coordinates": {{"latitude": "lat_col", "longitude": "lon_col"}}
            }},
            "ml_approach": {{
                "type": "classification|regression|clustering",
                "suggested_model": "model_name",
                "target_type": "numerical|categorical",
                "preprocessing": ["scale_numerical", "encode_categorical"]
            }},
            "Give me a default visualization": "pie chart",
            "visualizations": [
                {{
                    "type": "plot_type",
                    "columns": ["col1", "col2"],
                    "title": "chart_title",
                    "description": "what_this_shows"
                }}
            ]
        }}
        
        Only include columns that exist in the dataset. If certain types don't exist, use empty lists or null values.
        """

        try:
            response = self._call_llm(prompt)
            analysis = json.loads(response)
            logging.info("LLM analysis completed successfully")
            return analysis
            
        except Exception as e:
            logging.error(f"Error in LLM analysis: {str(e)}")
            return self._fallback_analysis()

    def _call_llm(self, prompt: str) -> str:
        """Call the LLM service with the given prompt"""
        try:
            url = f"{self.base_url}/chat/completions"
            headers = {
                "Content-Type": "application/json"
            }
            data = {
                "model": self.model_uid,
                "messages": [
                    {"role": "system", "content": "You are a data science expert. Analyze the dataset and suggest appropriate ML approach and visualizations."},
                    {"role": "user", "content": prompt}
                ]
            }
            
            response = requests.post(url, headers=headers, json=data)
            response.raise_for_status()
            
            result = response.json()
            return result['choices'][0]['message']['content']
            
        except Exception as e:
            logging.error(f"Error calling LLM service: {str(e)}")
            return None

    def _fallback_analysis(self) -> Dict:
        """Fallback analysis when LLM fails"""
        numerical_cols = self.data.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = self.data.select_dtypes(include=['object']).columns.tolist()
        
        # Try to identify target variable (last column if numerical)
        target = numerical_cols[-1] if numerical_cols else None
        features = numerical_cols[:-1] if target else numerical_cols
        
        return {
            "features": {
                "numerical": features,
                "categorical": categorical_cols,
                "target": target,
                "coordinates": {
                    "latitude": None,
                    "longitude": None
                }
            },
            "ml_approach": {
                "type": "regression" if target else "clustering",
                "suggested_model": "random_forest" if target else "kmeans",
                "target_type": "numerical" if target else None,
                "preprocessing": ["scale_numerical", "encode_categorical"]
            },
            "Give me a default visualization": "pie chart",
            "visualizations": [
                {
                    "type": "scatter",
                    "columns": features[:2] if len(features) >= 2 else features,
                    "title": "Feature Relationships",
                    "description": "Relationship between main numerical features"
                }
            ]
        }

    def preprocess_data(self) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Preprocess data based on LLM analysis"""
        if not self.analysis:
            raise ValueError("Run load_data first to analyze the dataset")

        features = []
        
        # Process numerical features
        numerical = self.analysis["features"]["numerical"]
        if numerical:
            num_data = self.data[numerical].copy()
            num_data = num_data.fillna(num_data.mean())
            num_scaled = self.scaler.fit_transform(num_data)
            features.append(num_scaled)

        # Process categorical features
        categorical = self.analysis["features"]["categorical"]
        if categorical:
            for col in categorical:
                encoder = LabelEncoder()
                cat_encoded = encoder.fit_transform(self.data[col].fillna('missing'))
                self.encoders[col] = encoder
                features.append(cat_encoded.reshape(-1, 1))

        # Combine features
        X = np.hstack(features) if features else None

        # Process target if exists
        target = self.analysis["features"]["target"]
        y = None
        if target:
            if self.analysis["ml_approach"]["target_type"] == "categorical":
                encoder = LabelEncoder()
                y = encoder.fit_transform(self.data[target].fillna('missing'))
                self.encoders['target'] = encoder
            else:
                y = self.data[target].fillna(self.data[target].mean()).values

        return X, y

    def train_model(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> None:
        """Train model based on LLM suggestions"""
        approach = self.analysis["ml_approach"]
        
        if approach["type"] == "clustering":
            model = KMeans(n_clusters=min(5, len(self.data)//2), random_state=42)
            self.models["clustering"] = model
            self.models["clustering"].fit(X)
            
        elif approach["type"] == "classification":
            model = RandomForestClassifier(random_state=42)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model.fit(X_train, y_train)
            score = model.score(X_test, y_test)
            logging.info(f"Classification accuracy: {score:.2f}")
            self.models["classification"] = model
            
        elif approach["type"] == "regression":
            model = RandomForestRegressor(random_state=42)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model.fit(X_train, y_train)
            score = model.score(X_test, y_test)
            logging.info(f"Regression R² score: {score:.2f}")
            self.models["regression"] = model

    def create_visualization(self, viz_config: Dict, save_path: Optional[str] = None) -> None:
        """Create visualization based on LLM suggestions"""
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
            elif viz_config["type"] == "cluster":
                self._create_cluster_plot(viz_config)
            
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
        plt.scatter(self.data[config["columns"][0]], 
                   self.data[config["columns"][1]], 
                   alpha=0.5)
        plt.xlabel(config["columns"][0])
        plt.ylabel(config["columns"][1])

    def _create_bar(self, config: Dict) -> None:
        data = self.data[config["columns"][0]].value_counts()[:15]
        data.plot(kind='bar')
        plt.xticks(rotation=45, ha='right')

    def _create_line(self, config: Dict) -> None:
        self.data.plot(x=config["columns"][0], 
                      y=config["columns"][1], 
                      kind='line')

    def _create_heatmap(self, config: Dict) -> None:
        if len(config["columns"]) == 2:
            plt.hist2d(self.data[config["columns"][0]], 
                      self.data[config["columns"][1]], 
                      bins=50, 
                      cmap='YlOrRd')
            plt.colorbar(label='Count')
        else:
            correlation = self.data[config["columns"]].corr()
            sns.heatmap(correlation, annot=True, cmap='coolwarm')

    def _create_cluster_plot(self, config: Dict) -> None:
        if "clustering" not in self.models:
            logging.warning("No clustering model found. Skipping cluster plot.")
            return
            
        plt.scatter(self.data[config["columns"][0]], 
                   self.data[config["columns"][1]], 
                   c=self.models["clustering"].labels_,
                   cmap='viridis')
        plt.colorbar(label='Cluster')

    def analyze_and_visualize(self) -> Dict[str, str]:
        """Run complete analysis, training, and visualization"""
        if not self.analysis:
            raise ValueError("Please load and analyze data first")

        # Preprocess and train
        X, y = self.preprocess_data()
        self.train_model(X, y)

        # Generate visualizations
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs('charts', exist_ok=True)
        
        chart_paths = {}
        
        for i, viz in enumerate(self.analysis["visualizations"]):
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
    
    parser = argparse.ArgumentParser(description='Dynamic Data Analysis and Visualization')
    parser.add_argument('file_path', help='Path to the data file (CSV or Excel)')
    args = parser.parse_args()
    
    try:
        analyzer = DynamicAnalyzer()
        analyzer.load_data(args.file_path)
        chart_paths = analyzer.analyze_and_visualize()
        
        print("\nAnalysis complete! Generated the following visualizations:")
        for title, info in chart_paths.items():
            print(f"\n- {title}")
            print(f"  Path: {info['path']}")
            print(f"  Description: {info['description']}")
            
    except Exception as e:
        logging.error(f"Analysis failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
