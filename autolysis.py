import os
import sys
import pandas as pd
import seaborn as sns
import numpy  as np
import matplotlib.pyplot as plt
import httpx
import chardet

# Constants
API_URL = "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions"
AIPROXY_TOKEN = "eyJhbGciOiJIUzI1NiJ9.eyJlbWFpbCI6IjIyZjEwMDAyMTNAZHMuc3R1ZHkuaWl0bS5hYy5pbiJ9.UrF4VGAcZE6JaDyAjlGqKg5VS1Hhl4rFlhr4uEkNn3M"

def load_data(file_path):
    """Load CSV data with encoding detection."""
    with open(file_path, 'rb') as f:
        result = chardet.detect(f.read())
    encoding = result['encoding']
    return pd.read_csv(file_path, encoding=encoding)



def visualize_data(df):
    """Generate and save visualizations."""
    sns.set(style="whitegrid")
    numeric_columns = df.select_dtypes(include=['number']).columns
    for column in numeric_columns:
        plt.figure()
        sns.histplot(df[column].dropna(), kde=True)
        plt.title(f'Distribution of {column}')
        plt.savefig(f'{column}_distribution.png')
        plt.close()


def identify_and_remove_identifiers(df):
    """
    Identifies potential identifier columns in a DataFrame and removes all but the one with the most unique values.

    Args:
        df: The input DataFrame.

    Returns:
        A new DataFrame with the identified columns removed, except for the identifier column with the most unique values.
    """

    id_cols = []
    for col in df.columns:
      # Check for columns that are likely identifiers
      conditions = [
          # Columns with 100% unique values

          # Columns with names suggesting identification
          col.lower() in ['id', 'identifier', 'key', 'uid', 'uuid',
                        'index', 'name', 'username', 'email',
                        'code', 'reference', 'guid'],
          # Columns with string type and high cardinality
          (df[col].dtype == 'object' and
           df[col].nunique() > len(df) * 0.9),

          # Columns that look like typical ID formats
          (df[col].dtype == 'object' and
           df[col].str.match(r'^[A-Za-z0-9\-_]+$').all())
      ]

      if any(conditions):
          id_cols.append(col)

    if not id_cols:
        print("No potential identifier columns found.")
        return df

    # Find the identifier column with the most unique values
    best_id_col = max(id_cols, key=lambda col: df[col].nunique())

    #Remove other identifier columns
    cols_to_remove = [col for col in id_cols if col != best_id_col]

    print(f"Removing identifier columns: {cols_to_remove}")
    df = df.drop(columns=cols_to_remove)

    return df

def Drop_URL_Columns(df):
    import re

    # Function to check if a string contains a URL
    def contains_url(text):
      url_pattern = re.compile(r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+')
      return bool(url_pattern.search(text))


    # Identify columns with URLs
    url_columns = []
    for col in df.columns:
      if df[col].apply(lambda x: isinstance(x, str) and contains_url(x)).any():
        url_columns.append(col)

    # Remove URL columns from the DataFrame
    df = df.drop(columns=url_columns)
    return df

def columns_dtype(df):
  # Get a list of numerical columns
  numerical_cols = df.select_dtypes(include=np.number).columns.tolist()
  # Get a list of object columns
  object_cols = df.select_dtypes(include='object').columns.tolist()
  return f"Numerical:{numerical_cols}, Objects: {object_cols}",object_cols,numerical_cols

def Value_composition(df,object_cols):
    tmp = ""
    for col in object_cols:
        value_counts = df[col].value_counts(normalize=True) * 100
        tmp = tmp + f"\nPercentage of unique values for column '{col}':{value_counts}"
    
    
    headers = {
        'Authorization': f'Bearer {AIPROXY_TOKEN}',
        'Content-Type': 'application/json'
    }
    prompt = f"Summarize the following data to be futher passed to an LLM for analysis make it as compact as possible: {tmp}"
    data = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": prompt}]
    }
    try:
        response = httpx.post(API_URL, headers=headers, json=data, timeout=30.0)
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content']
    except httpx.HTTPStatusError as e:
        print(f"HTTP error occurred: {e}")
    except httpx.RequestError as e:
        print(f"Request error occurred: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    return "Narrative generation failed due to an error."

def calculate_stats(df, numerical_cols):
    stats = {}
    for col in numerical_cols:
        # Remove rows with NaN or 'unknown' in the current column
        cleaned_data = df[df[col].notna() & ~df[col].astype(str).str.lower().str.contains('unknown')][col]

        if not cleaned_data.empty:
          stats[col] = {
                'mean': cleaned_data.mean(),
                'median': cleaned_data.median(),
                'mode': cleaned_data.mode().iloc[0] if not cleaned_data.mode().empty else None
          }
        else:
          stats[col] = {
              'mean': None,
              'median': None,
              'mode': None
            }
    headers = {
        'Authorization': f'Bearer {AIPROXY_TOKEN}',
        'Content-Type': 'application/json'
    }
    prompt = f"You being a Data Scientist to Your Understading pick imp part of this data and Summarize the following data to be futher passed to an LLM for analysis make it as compact as possible: {stats}"
    data = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": prompt}]
    }
   
    response = httpx.post(API_URL, headers=headers, json=data, timeout=30.0)
    response.raise_for_status()
    return response.json()['choices'][0]['message']['content']


def generate_narrative(analysis):
    """Generate narrative using LLM."""
    headers = {
        'Authorization': f'Bearer {AIPROXY_TOKEN}',
        'Content-Type': 'application/json'
    }
    prompt = f"""Provide a detailed analysis based on the following data summary 
    Have it describe:

    The data you received, briefly
    The analysis you carried out
    The insights you discovered
    The implications of your findings (i.e. what to do with the insights): {analysis}"""
    data = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": prompt}]
    }
    try:
        response = httpx.post(API_URL, headers=headers, json=data, timeout=30.0)
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content']
    except httpx.HTTPStatusError as e:
        print(f"HTTP error occurred: {e}")
    except httpx.RequestError as e:
        print(f"Request error occurred: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    return "Narrative generation failed due to an error."

def main(file_path):
    compact_summarization = ""
    df = load_data(file_path)
    df = identify_and_remove_identifiers(df)
    df = Drop_URL_Columns(df)
    _,object_cols,numerical_cols = columns_dtype(df)
    compact_summarization = compact_summarization + _
    compact_summarization = compact_summarization + " " + Value_composition(df,object_cols)
    compact_summarization = compact_summarization + " " + calculate_stats(df,numerical_cols)




    # visualize_data(df)
    narrative = generate_narrative(compact_summarization)
    with open('README.md', 'w') as f:
        f.write(narrative)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python autolysis.py <dataset.csv>")
        sys.exit(1)
    main(sys.argv[1])
