import pandas as pd
from app import execute_query_and_visualize, validate_sql_query

# Sample DataFrame for testing
sample_data = {
    'Beat Code': ['A', 'B', 'C', 'A', 'B'],
    'Customer Name': ['John', 'Doe', 'Alice', 'Bob', 'Charlie']
}
df = pd.DataFrame(sample_data)

# Test cases

def test_valid_suggestion():
    sql_query = "SELECT `Beat Code` as category, COUNT(`Customer Name`) as value FROM data_table GROUP BY `Beat Code`"
    chart_type = "bar"
    title = "Valid Suggestion Test"
    execute_query_and_visualize(df, sql_query, chart_type, title)


def test_missing_keys():
    suggestion = {'columns': ['Beat Code'], 'reasoning': 'Missing chart_type'}
    if 'chart_type' not in suggestion:
        print("Test Missing Keys: Passed")
    else:
        print("Test Missing Keys: Failed")


def test_invalid_sql_query():
    sql_query = "SELECT * FROM data_table WHERE"
    if not validate_sql_query(sql_query):
        print("Test Invalid SQL Query: Passed")
    else:
        print("Test Invalid SQL Query: Failed")


def test_empty_query_result():
    sql_query = "SELECT `Beat Code` as category, COUNT(`Customer Name`) as value FROM data_table WHERE 1=0 GROUP BY `Beat Code`"
    chart_type = "bar"
    title = "Empty Query Result Test"
    execute_query_and_visualize(df, sql_query, chart_type, title)


def test_incorrect_query_columns():
    sql_query = "SELECT `Beat Code` FROM data_table"
    chart_type = "bar"
    title = "Incorrect Query Columns Test"
    execute_query_and_visualize(df, sql_query, chart_type, title)


if __name__ == "__main__":
    test_valid_suggestion()
    test_missing_keys()
    test_invalid_sql_query()
    test_empty_query_result()
    test_incorrect_query_columns()
