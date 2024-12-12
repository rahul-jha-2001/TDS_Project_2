# import os
# import sys
# import pandas as pd
# import seaborn as sns
# import numpy  as np
# import matplotlib.pyplot as plt
# import httpx
# import chardet
# import os
# import dotenv
# import matplotlib.pyplot as plt
# import seaborn as sns
# dotenv.load_dotenv()
# # Constants
# API_URL = "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions"
# AIPROXY_TOKEN = os.environ["AIPROXY_TOKEN"]

# def load_data(file_path):
#     """Load CSV data with encoding detection."""
#     with open(file_path, 'rb') as f:
#         result = chardet.detect(f.read())
#     encoding = result['encoding']
#     return pd.read_csv(file_path, encoding=encoding)


# def identify_and_remove_identifiers(df):
#     """
#     Identifies potential identifier columns in a DataFrame and removes all but the one with the most unique values.

#     Args:
#         df: The input DataFrame.

#     Returns:
#         A new DataFrame with the identified columns removed, except for the identifier column with the most unique values.
#     """

#     id_cols = []
#     for col in df.columns:
#       # Check for columns that are likely identifiers
#       conditions = [
#           # Columns with 100% unique values

#           # Columns with names suggesting identification
#           col.lower() in ['id', 'identifier', 'key', 'uid', 'uuid',
#                         'index', 'name', 'username', 'email',
#                         'code', 'reference', 'guid'],
#           # Columns with string type and high cardinality
#           (df[col].dtype == 'object' and
#            df[col].nunique() > len(df) * 0.5),

#           # Columns that look like typical ID formats
#           (df[col].dtype == 'object' and
#            df[col].str.match(r'^[A-Za-z0-9\-_]+$').all())
#       ]

#       if any(conditions):
#           id_cols.append(col)

#     if not id_cols:
#         print("No potential identifier columns found.")
#         return df

#     # Find the identifier column with the most unique values
#     best_id_col = max(id_cols, key=lambda col: df[col].nunique())

#     #Remove other identifier columns
#     cols_to_remove = [col for col in id_cols if col != best_id_col]

#     print(f"Removing identifier columns: {cols_to_remove}")
#     df = df.drop(columns=cols_to_remove)

#     return df

# def Drop_URL_Columns(df):
#     import re

#     # Function to check if a string contains a URL
#     def contains_url(text):
#       url_pattern = re.compile(r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+')
#       return bool(url_pattern.search(text))


#     # Identify columns with URLs
#     url_columns = []
#     for col in df.columns:
#       if df[col].apply(lambda x: isinstance(x, str) and contains_url(x)).any():
#         url_columns.append(col)

#     # Remove URL columns from the DataFrame
#     df = df.drop(columns=url_columns)
#     return df

# def columns_dtype(df):
#   # Get a list of numerical columns
#   numerical_cols = df.select_dtypes(include=np.number).columns.tolist()
#   # Get a list of object columns
#   object_cols = df.select_dtypes(include='object').columns.tolist()
#   return f"Numerical:{numerical_cols}, Objects: {object_cols}",object_cols,numerical_cols

# def Value_composition(df,object_cols):
#     tmp = ""
#     for col in object_cols:
#         value_counts = df[col].value_counts(normalize=True) * 100
#         tmp = tmp + f"\nPercentage of unique values for column '{col}':{value_counts}"
    
    
#     headers = {
#         'Authorization': f'Bearer {AIPROXY_TOKEN}',
#         'Content-Type': 'application/json'
#     }
#     prompt = f"Summarize the following data to be futher passed to an LLM for analysis: {tmp}"
#     data = {
#         "model": "gpt-4o-mini",
#         "messages": [{"role": "user", "content": prompt}]
#     }
#     try:
#         response = httpx.post(API_URL, headers=headers, json=data, timeout=30.0)
#         response.raise_for_status()
#         return response.json()['choices'][0]['message']['content']
#     except httpx.HTTPStatusError as e:
#         print(f"HTTP error occurred: {e}")
#     except httpx.RequestError as e:
#         print(f"Request error occurred: {e}")
#     except Exception as e:
#         print(f"An unexpected error occurred: {e}")
#     return "Narrative generation failed due to an error."

# def calculate_stats(df, numerical_cols):
#     stats = {}
#     for col in numerical_cols:
#         # Remove rows with NaN or 'unknown' in the current column
#         cleaned_data = df[df[col].notna() & ~df[col].astype(str).str.lower().str.contains('unknown')][col]

#         if not cleaned_data.empty:
#           stats[col] = {
#                 'mean': cleaned_data.mean(),
#                 'median': cleaned_data.median(),
#                 'mode': cleaned_data.mode().iloc[0] if not cleaned_data.mode().empty else None
#           }
#         else:
#           stats[col] = {
#               'mean': None,
#               'median': None,
#               'mode': None
#             }
#     headers = {
#         'Authorization': f'Bearer {AIPROXY_TOKEN}',
#         'Content-Type': 'application/json'
#     }
#     prompt = f"You being a Data Scientist to Your Understading pick imp part of this data and Summarize the following data to be futher passed to an LLM for analysis: {stats}"
#     data = {
#         "model": "gpt-4o-mini",
#         "messages": [{"role": "user", "content": prompt}]
#     }
   
#     response = httpx.post(API_URL, headers=headers, json=data, timeout=30.0)
#     response.raise_for_status()
#     return response.json()['choices'][0]['message']['content']


# def generate_narrative(analysis):
#     """Generate narrative using LLM."""
#     headers = {
#         'Authorization': f'Bearer {AIPROXY_TOKEN}',
#         'Content-Type': 'application/json'
#     }
#     prompt = f"""Provide a detailed analytic stroy based on the following data summary 
#     Have it describe:

#     The data you received, briefly
#     The analysis you carried out
#     The insights you discovered
#     The implications of your findings (i.e. what to do with the insights): {analysis}"""
#     data = {
#         "model": "gpt-4o-mini",
#         "messages": [{"role": "user", "content": prompt}]
#     }
#     try:
#         response = httpx.post(API_URL, headers=headers, json=data, timeout=30.0)
#         response.raise_for_status()
#         return response.json()['choices'][0]['message']['content']
#     except httpx.HTTPStatusError as e:
#         print(f"HTTP error occurred: {e}")
#     except httpx.RequestError as e:
#         print(f"Request error occurred: {e}")
#     except Exception as e:
#         print(f"An unexpected error occurred: {e}")
#     return "Narrative generation failed due to an error."


# def generate_dist_plots(df, numerical_cols):
#     num_plots = len(numerical_cols)
#     cols_per_row = 2  # Adjust as needed
#     rows = (num_plots + cols_per_row - 1) // cols_per_row
#     plt.figure(figsize=(15, 5 * rows))  # Adjust figure size dynamically

#     for i, col in enumerate(numerical_cols):
#         plt.subplot(rows, cols_per_row, i + 1)  # Create subplots dynamically
#         plt.hist(df[col], bins=100)  # Adjust the number of bins as needed
#         plt.title(f'Distribution of {col}')
#         plt.xlabel(col)
#         plt.ylabel('Frequency')

#     plt.tight_layout()  # Adjust layout to prevent overlapping
#     plt.savefig("Distribution")

# def plot_correlation_heatmap(df, numerical_cols):
#     numerical_df = df[numerical_cols]
#     correlation_matrix = numerical_df.corr()
#     plt.figure(figsize=(10, 8))
#     sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
#     plt.title('Correlation Matrix Heatmap')
#     plt.savefig(f'corr.png')
#     plt.close()

# def create_and_change_dir(filename):
#     import os
#     # Get the directory name from the filename
#     directory_name = os.path.splitext(filename)[0]
    
#     # Create the directory if it doesn't exist
#     if not os.path.exists(directory_name):
#         os.mkdir(directory_name)
#         print(f"Directory '{directory_name}' created.")
#     else:
#         print(f"Directory '{directory_name}' already exists.")
    
#     # Change to the new directory
#     os.chdir(directory_name)
#     print(f"Changed directory to '{os.getcwd()}'")

# def main(file_path):
#     df = load_data(file_path)
#     create_and_change_dir(file_path)
#     compact_summarization = ""
    
#     df = identify_and_remove_identifiers(df)
#     df = Drop_URL_Columns(df)
#     _,object_cols,numerical_cols = columns_dtype(df)
#     compact_summarization = compact_summarization + _
#     compact_summarization = compact_summarization + " " + Value_composition(df,object_cols)
#     compact_summarization = compact_summarization + " " + calculate_stats(df,numerical_cols)


#     generate_dist_plots(df,numerical_cols)
#     plot_correlation_heatmap(df,numerical_cols)
#     # visualize_data(df)
#     narrative = generate_narrative(compact_summarization)
#     with open('README.md', 'w') as f:
#         f.write(narrative)

# if __name__ == "__main__":
#     if len(sys.argv) != 2:
#         print("Usage: python autolysis.py <dataset.csv>")
#         sys.exit(1)
#     main(sys.argv[1])
import os
import sys
import re
import chardet
import httpx
import dotenv
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

class DataAnalyzer:
    """
    A comprehensive data analysis utility for processing and analyzing CSV datasets.
    
    Features:
    - Automatic encoding detection
    - Identifier column removal
    - URL column elimination
    - Statistical analysis
    - Visualization generation
    - Narrative generation using LLM
    """

    def __init__(self):
        """Initialize the DataAnalyzer with environment variables."""
        dotenv.load_dotenv()
        self.API_URL = "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions"
        self.AIPROXY_TOKEN = os.environ.get("AIPROXY_TOKEN")
        
        if not self.AIPROXY_TOKEN:
            raise ValueError("AIPROXY_TOKEN is not set in the environment variables.")

    def load_data(self, file_path):
        """
        Load CSV data with automatic encoding detection.
        
        Args:
            file_path (str): Path to the CSV file
        
        Returns:
            pd.DataFrame: Loaded DataFrame
        """
        with open(file_path, 'rb') as f:
            result = chardet.detect(f.read())
        
        return pd.read_csv(file_path, encoding=result['encoding'])

    def _is_potential_identifier(self, series):
        """
        Determine if a column is likely an identifier.
        
        Args:
            series (pd.Series): Column to analyze
        
        Returns:
            bool: Whether the column is likely an identifier
        """
        identifier_keywords = [
            'id', 'identifier', 'key', 'uid', 'uuid', 
            'index', 'name', 'username', 'email',
            'code', 'reference', 'guid'
        ]

        # Check if column name suggests identifier
        name_condition = any(keyword in series.name.lower() for keyword in identifier_keywords)

        # Check for string columns with high cardinality
        high_cardinality_condition = (
            series.dtype == 'object' and 
            series.nunique() > len(series) * 0.5
        )

        # Check for string columns matching typical ID formats
        id_format_condition = (
            series.dtype == 'object' and 
            series.str.match(r'^[A-Za-z0-9\-_]+$').all()
        )

        return name_condition or high_cardinality_condition or id_format_condition

    def remove_identifier_columns(self, df):
        """
        Remove redundant identifier columns, keeping the most informative one.
        
        Args:
            df (pd.DataFrame): Input DataFrame
        
        Returns:
            pd.DataFrame: DataFrame with minimal identifier columns
        """
        id_cols = [col for col in df.columns if self._is_potential_identifier(df[col])]

        if not id_cols:
            print("No potential identifier columns found.")
            return df

        # Keep the identifier column with most unique values
        best_id_col = max(id_cols, key=lambda col: df[col].nunique())
        cols_to_remove = [col for col in id_cols if col != best_id_col]

        print(f"Removing identifier columns: {cols_to_remove}")
        return df.drop(columns=cols_to_remove)

    def remove_url_columns(self, df):
        """
        Remove columns containing URLs.
        
        Args:
            df (pd.DataFrame): Input DataFrame
        
        Returns:
            pd.DataFrame: DataFrame without URL columns
        """
        def contains_url(text):
            url_pattern = re.compile(r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+')
            return isinstance(text, str) and bool(url_pattern.search(text))

        url_columns = [
            col for col in df.columns 
            if df[col].apply(contains_url).any()
        ]

        if url_columns:
            print(f"Removing URL columns: {url_columns}")
            return df.drop(columns=url_columns)
        
        return df

    def analyze_column_types(self, df):
        """
        Analyze and categorize DataFrame columns.
        
        Args:
            df (pd.DataFrame): Input DataFrame
        
        Returns:
            tuple: Narrative description, object columns, numerical columns
        """
        numerical_cols = df.select_dtypes(include=np.number).columns.tolist()
        object_cols = df.select_dtypes(include='object').columns.tolist()
        
        description = (
            f"Numerical Columns: {numerical_cols}\n"
            f"Object Columns: {object_cols}"
        )
        
        return description, object_cols, numerical_cols

    def analyze_object_columns(self, df, object_cols):
        """
        Generate summary of object column value compositions.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            object_cols (list): List of object column names
        
        Returns:
            str: Summarized value composition
        """
        value_compositions = []
        for col in object_cols:
            value_counts = df[col].value_counts(normalize=True) * 100
            value_compositions.append(f"Percentage of unique values for column '{col}':\n{value_counts}")
        
        return "\n".join(value_compositions)

    def _call_llm_api(self, prompt, model="gpt-4o-mini"):
        """
        Call the Language Model API for text generation.
        
        Args:
            prompt (str): Input prompt for the LLM
            model (str, optional): LLM model to use
        
        Returns:
            str: Generated text from LLM
        """
        headers = {
            'Authorization': f'Bearer {self.AIPROXY_TOKEN}',
            'Content-Type': 'application/json'
        }
        
        data = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}]
        }
        
        try:
            response = httpx.post(self.API_URL, headers=headers, json=data, timeout=30.0)
            response.raise_for_status()
            return response.json()['choices'][0]['message']['content']
        except Exception as e:
            print(f"LLM API call error: {e}")
            return "LLM generation failed."

    def calculate_column_stats(self, df, numerical_cols):
        """
        Calculate basic statistics for numerical columns.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            numerical_cols (list): List of numerical column names
        
        Returns:
            dict: Column statistics
        """
        stats = {}
        for col in numerical_cols:
            # Remove rows with NaN or 'unknown' values
            cleaned_data = df[
                df[col].notna() & 
                ~df[col].astype(str).str.lower().str.contains('unknown')
            ][col]

            if not cleaned_data.empty:
                stats[col] = {
                    'mean': cleaned_data.mean(),
                    'median': cleaned_data.median(),
                    'mode': cleaned_data.mode().iloc[0] if not cleaned_data.mode().empty else None
                }
            else:
                stats[col] = {'mean': None, 'median': None, 'mode': None}
        
        return stats

    def generate_visualizations(self, df, numerical_cols):
        """
        Generate distribution plots and correlation heatmap.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            numerical_cols (list): List of numerical column names
        """
        # Distribution Plots
        num_plots = len(numerical_cols)
        cols_per_row = 2
        rows = (num_plots + cols_per_row - 1) // cols_per_row
        
        plt.figure(figsize=(15, 5 * rows))
        for i, col in enumerate(numerical_cols):
            plt.subplot(rows, cols_per_row, i + 1)
            plt.hist(df[col], bins=100)
            plt.title(f'Distribution of {col}')
            plt.xlabel(col)
            plt.ylabel('Frequency')
        plt.tight_layout()
        plt.savefig("distribution_plots.png")
        plt.close()

        # Correlation Heatmap
        numerical_df = df[numerical_cols]
        correlation_matrix = numerical_df.corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
        plt.title('Correlation Matrix Heatmap')
        plt.savefig('correlation_heatmap.png')
        plt.close()

    def generate_data_narrative(self, compact_summary):
        """
        Generate a comprehensive narrative using LLM.
        
        Args:
            compact_summary (str): Compact data summary
        
        Returns:
            str: Generated narrative
        """
        prompt = f"""Provide a detailed analytical story based on the following data summary:

        Data Overview
        - Describe the data you received
        - Explain the analysis carried out
        - Highlight key insights discovered
        - Discuss implications and potential actions

        Data Summary:
        {compact_summary}"""

        return self._call_llm_api(prompt)

    def create_analysis_directory(self, filename):
        """
        Create a directory for the analysis and change to it.
        
        Args:
            filename (str): Original filename
        """
        directory_name = os.path.splitext(filename)[0]
        os.makedirs(directory_name, exist_ok=True)
        os.chdir(directory_name)
        print(f"Analysis directory: {os.getcwd()}")

    def analyze_dataset(self, file_path):
        """
        Comprehensive dataset analysis pipeline.
        
        Args:
            file_path (str): Path to the CSV file
        """
        # Setup analysis directory
        

        # Load and preprocess data
        df = self.load_data(file_path)
        df = self.remove_identifier_columns(df)
        df = self.remove_url_columns(df)
        self.create_analysis_directory(file_path)
        # Analyze column types
        column_type_desc, object_cols, numerical_cols = self.analyze_column_types(df)

        # Collect analysis components
        compact_summary = [column_type_desc]
        compact_summary.append(self.analyze_object_columns(df, object_cols))
        
        # Calculate statistics
        column_stats = self.calculate_column_stats(df, numerical_cols)
        column_stats_summary = self._call_llm_api(
            f"Analyze these column statistics from a Data Scientist perspective: {column_stats}"
        )
        compact_summary.append(column_stats_summary)

        # Generate visualizations
        self.generate_visualizations(df, numerical_cols)

        # Generate narrative
        narrative = self.generate_data_narrative(" ".join(compact_summary))
        
        # Write narrative to README
        with open('README.md', 'w') as f:
            f.write(narrative)

def main():
    """Main script entry point."""
    if len(sys.argv) != 2:
        print("Usage: python data_analyzer.py <dataset.csv>")
        sys.exit(1)

    try:
        analyzer = DataAnalyzer()
        analyzer.analyze_dataset(sys.argv[1])
    except Exception as e:
        print(f"Analysis failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()