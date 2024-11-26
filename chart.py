from transformers import GPT2LMHeadModel, GPT2Tokenizer
import pandas as pd
import matplotlib.pyplot as plt
import sqlite3
import seaborn as sns
import os
from datetime import datetime
import logging
import plotly.express as px

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ChartGenerator:
    def __init__(self, model_path: str):
        """Initialize with trained model"""
        logging.info('Initializing ChartGenerator with model path: %s', model_path)
        self.model = GPT2LMHeadModel.from_pretrained(model_path)
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    
    def generate_chart_suggestion(self, question: str, data_info: dict):
        """Generate chart suggestion using trained model"""
        logging.info('Generating chart suggestion for question: %s', question)
        prompt = f"""
Input: {question}
Dataset Info:
- Columns: {data_info['columns']}
- Numeric columns: {data_info['numeric_columns']}
- Categorical columns: {data_info['categorical_columns']}
"""
        
        inputs = self.tokenizer.encode(prompt, return_tensors='pt')
        outputs = self.model.generate(
            inputs,
            max_length=200,
            num_return_sequences=1,
            temperature=0.7
        )
        
        response = self.tokenizer.decode(outputs[0])
        return self._parse_response(response)
    
    def _parse_response(self, response: str):
        """Parse model response into chart suggestion"""
        # Extract chart type and SQL query from response
        lines = response.split('\n')
        chart_type = None
        sql_query = None
        
        for line in lines:
            if 'Chart Type:' in line:
                chart_type = line.split(':')[1].strip().lower()
            elif 'SQL Query:' in line:
                sql_query = line.split(':')[1].strip()
        
        return {
            'chart_type': chart_type,
            'sql_query': sql_query
        }

def create_visualization(df: pd.DataFrame, chart_type: str, title: str):
    """Create interactive visualization with Plotly"""
    logging.info(f'Creating {chart_type} visualization...')
    charts_dir = os.path.join(os.path.dirname(__file__), "charts")
    os.makedirs(charts_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(charts_dir, f"{chart_type}_{timestamp}.html")

    try:
        if chart_type == 'pie':
            fig = px.pie(df, names=df.columns[0], values=df.columns[1], title=title)
        elif chart_type == 'bar':
            # Ensure data is aggregated and sorted
            df = df.sort_values(by=df.columns[1], ascending=False)
            fig = px.bar(df, x=df.columns[0], y=df.columns[1], title=title,
                         labels={df.columns[0]: 'Category', df.columns[1]: 'Value'},
                         text_auto=True)
            fig.update_traces(marker_color='indigo', marker_line_color='rgb(8,48,107)',
                              marker_line_width=1.5, opacity=0.6)
            fig.update_layout(xaxis_tickangle=-45)
        elif chart_type == 'line':
            fig = px.line(df, x=df.columns[0], y=df.columns[1], title=title)
        else:
            logging.warning(f'Unsupported chart type: {chart_type}')
            return

        fig.write_html(filename)
        logging.info(f"Chart saved as: {filename}")

    except Exception as e:
        logging.error(f"Error creating visualization: {str(e)}")

def main():
    # Initialize chart generator with trained model
    generator = ChartGenerator("trained_model")
    
    # Load dataset
    csv_file = "data/mlb_players.csv"
    df = pd.read_csv(csv_file)
    
    # Get dataset info
    data_info = {
        "columns": list(df.columns),
        "numeric_columns": df.select_dtypes(include=['int64', 'float64']).columns.tolist(),
        "categorical_columns": df.select_dtypes(include=['object']).columns.tolist(),
    }
    
    while True:
        question = input("\nWhat would you like to visualize? (or 'exit' to quit): ")
        if question.lower() == 'exit':
            break
        
        # Generate chart suggestion
        suggestion = generator.generate_chart_suggestion(question, data_info)
        print(f"\nSuggested Chart Type: {suggestion['chart_type']}")
        print(f"Generated SQL Query: {suggestion['sql_query']}")
        
        # Execute query and create visualization
        conn = sqlite3.connect(':memory:')
        df.to_sql('data_table', conn, index=False)
        
        try:
            result_df = pd.read_sql_query(suggestion['sql_query'], conn)
            create_visualization(
                result_df,
                suggestion['chart_type'],
                question
            )
        except Exception as e:
            print(f"Error: {str(e)}")
        finally:
            conn.close()

if __name__ == "__main__":
    main()
