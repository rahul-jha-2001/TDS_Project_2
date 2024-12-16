# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "seaborn",
#   "pandas",
#   "matplotlib",
#   "httpx",
#   "chardet",
#   "ipykernel",
#   "openai",
#   "numpy",
#   "scipy",
#   "python-dotenv"
# ]
# ///


import os
import sys
import re
import httpx
import dotenv
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
import chardet

import matplotlib.font_manager as fm

# Install or use a CJK-compatible font
plt.rcParams['font.sans-serif'] = ['SimHei']  # Chinese (Simplified) font
plt.rcParams['axes.unicode_minus'] = False

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
    
    def _analyze_column_patterns(self, series):
        """
        Analyze patterns in a categorical column.
        
        Args:
            series (pd.Series): Input series of categorical data
        
        Returns:
            dict: Pattern analysis results
        """
        pattern_analysis = {
            'starts_with': {},
            'ends_with': {},
            'contains': {}
        }
        
        # Most common starting characters/substrings
        try:
            starts_with = series.str[:3].value_counts().head(5)
            pattern_analysis['starts_with'] = starts_with.to_dict()
        except Exception:
            pass
        
        # Most common ending characters/substrings
        try:
            ends_with = series.str[-3:].value_counts().head(5)
            pattern_analysis['ends_with'] = ends_with.to_dict()
        except Exception:
            pass
        
        # Most common substrings
        try:
            # Extract substrings of length 3-4 and count
            contains_substrings = series.str.findall(r'\w{3,4}').explode().value_counts().head(5)
            pattern_analysis['contains'] = contains_substrings.to_dict()
        except Exception:
            pass
        
        return pattern_analysis

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
        analysis_results = {}
    
        for col in object_cols:
            # 1. Basic Value Counts and Percentages
            value_counts = df[col].value_counts()
            percentages = df[col].value_counts(normalize=True) * 100
            
            # 2. Unique Value Analysis
            unique_values = df[col].nunique()
            most_common = value_counts.head(3)
            least_common = value_counts.tail(3)
            
            # 3. Missing Value Analysis
            missing_values = df[col].isnull().sum()
            missing_percentage = (missing_values / len(df)) * 100
            
            # 4. Entropy Analysis (measure of diversity)
            try:
                value_probs = percentages / 100
                entropy = stats.entropy(value_probs)
            except Exception:
                entropy = None
            
            # 5. Top Value Details
            top_value_details = {
                'top_value': value_counts.index[0],
                'top_value_count': value_counts.iloc[0],
                'top_value_percentage': percentages.iloc[0]
            }
            
            # 6. Length Analysis (for string columns)
            try:
                length_stats = df[col].str.len().describe()
            except Exception:
                length_stats = None
            
            # 7. Categorical Encoding Potential
            encoding_potential = {
                'unique_values': unique_values,
                'is_binary': unique_values == 2,
                'low_cardinality': unique_values <= 10,
                'high_cardinality': unique_values > 50
            }
            
            # 8. Pattern Analysis (regex matching)
            pattern_analysis = self._analyze_column_patterns(df[col])
            
            # Compile results
            analysis_results[col] = {
                'value_counts': value_counts.to_dict(),
                'percentages': percentages.to_dict(),
                'unique_values_count': unique_values,
                'most_common_values': most_common.to_dict(),
                'least_common_values': least_common.to_dict(),
                'missing_values': {
                    'count': missing_values,
                    'percentage': missing_percentage
                },
                'entropy': entropy,
                'top_value_details': top_value_details,
                'length_stats': length_stats.to_dict() if length_stats is not None else None,
                'encoding_potential': encoding_potential,
                'pattern_analysis': pattern_analysis
            }
    
        return "\n".join(analysis_results)

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

    def analyze_numerical_columns(self, df, numerical_cols):
        """
        Perform comprehensive statistical analysis on numerical columns.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            numerical_cols (list): List of numerical column names
        
        Returns:
            dict: Comprehensive statistical analysis of numerical columns
        """
        comprehensive_stats = {}
        
        for col in numerical_cols:
            # Clean data (remove NaN, 'unknown', etc.)
            cleaned_data = df[
                df[col].notna() & 
                ~df[col].astype(str).str.lower().str.contains('unknown')
            ][col]
            
            # Skip if no valid data
            if cleaned_data.empty:
                comprehensive_stats[col] = {
                    'data_valid': False,
                    'error': 'No valid numerical data found'
                }
                continue
            
            # 1. Basic Descriptive Statistics
            desc_stats = cleaned_data.describe()
            
            # 2. Advanced Distributional Statistics
            try:
                # Skewness and Kurtosis
                skewness = cleaned_data.skew()
                kurtosis = cleaned_data.kurtosis()
                
                # Normality Tests
                _, shapiro_p_value = stats.shapiro(cleaned_data)
                
                # Detailed Percentiles
                percentiles = [1, 5, 25, 50, 75, 95, 99]
                detailed_percentiles = {f'{p}th': np.percentile(cleaned_data, p) for p in percentiles}
            except Exception as e:
                skewness = None
                kurtosis = None
                shapiro_p_value = None
                detailed_percentiles = None
            
            # 3. Outlier Analysis
            try:
                # Interquartile Range (IQR) method
                Q1 = cleaned_data.quantile(0.25)
                Q3 = cleaned_data.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - (1.5 * IQR)
                upper_bound = Q3 + (1.5 * IQR)
                
                outliers = cleaned_data[(cleaned_data < lower_bound) | (cleaned_data > upper_bound)]
                outlier_stats = {
                    'count': len(outliers),
                    'percentage': (len(outliers) / len(cleaned_data)) * 100,
                    'lower_bound': lower_bound,
                    'upper_bound': upper_bound
                }
            except Exception as e:
                outlier_stats = None
            
            # 4. Correlation Analysis
            try:
                # Compute correlations with other numerical columns
                correlations = df[numerical_cols].corr()[col].drop(col)
                top_correlations = correlations.abs().nlargest(3)
            except Exception as e:
                correlations = None
                top_correlations = None
            
            # 5. Frequency and Binning Analysis
            try:
                # Create histogram-like binning
                hist, bin_edges = np.histogram(cleaned_data, bins='auto')
                bin_analysis = {
                    'bin_edges': bin_edges.tolist(),
                    'frequencies': hist.tolist()
                }
            except Exception as e:
                bin_analysis = None
            
            # 6. Statistical Tests
            try:
                # One-sample t-test against 0
                t_statistic, t_p_value = stats.ttest_1samp(cleaned_data, 0)
            except Exception as e:
                t_statistic = None
                t_p_value = None
            
            # Compile comprehensive statistics
            comprehensive_stats[col] = {
                'data_valid': True,
                'basic_stats': {
                    'count': desc_stats['count'],
                    'mean': desc_stats['mean'],
                    'median': desc_stats['50%'],
                    'mode': cleaned_data.mode().iloc[0] if not cleaned_data.mode().empty else None,
                    'std_dev': desc_stats['std'],
                    'min': desc_stats['min'],
                    'max': desc_stats['max']
                },
                'distribution': {
                    'skewness': skewness,
                    'kurtosis': kurtosis,
                    'shapiro_test_p_value': shapiro_p_value,
                    'is_normal_distribution': shapiro_p_value > 0.05 if shapiro_p_value is not None else None
                },
                'percentiles': detailed_percentiles,
                'outliers': outlier_stats,
                'correlations': {
                    'values': correlations.to_dict() if correlations is not None else None,
                    'top_3_correlations': top_correlations.to_dict() if top_correlations is not None else None
                },
                'binning_analysis': bin_analysis,
                'statistical_tests': {
                    't_test_against_zero': {
                        't_statistic': t_statistic,
                        'p_value': t_p_value
                    }
                }
            }
        
        return "\n".join(comprehensive_stats)

    def additional_advanced_analysis(self, df, numerical_cols):
        """
        Perform additional advanced analyses on numerical columns.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            numerical_cols (list): List of numerical column names
        
        Returns:
            dict: Advanced statistical analyses
        """
        advanced_analysis = {}
        
        for col in numerical_cols:
            cleaned_data = df[
                df[col].notna() & 
                ~df[col].astype(str).str.lower().str.contains('unknown')
            ][col]
            
            if cleaned_data.empty:
                continue
            
            # 1. Time Series-like Analysis (if applicable)
            try:
                rolling_mean = cleaned_data.rolling(window=3).mean()
                rolling_std = cleaned_data.rolling(window=3).std()
            except Exception as e:
                rolling_mean = None
                rolling_std = None
            
            # 2. Extreme Value Analysis
            try:
                extreme_values = {
                    'min_3': cleaned_data.nsmallest(3).tolist(),
                    'max_3': cleaned_data.nlargest(3).tolist()
                }
            except Exception as e:
                extreme_values = None
            
            # 3. Advanced Statistical Distributions
            try:
                # Fit various distributions
                distributions = {
                    'normal': stats.norm.fit(cleaned_data),
                    'exponential': stats.expon.fit(cleaned_data),
                    'gamma': stats.gamma.fit(cleaned_data)
                }
            except Exception as e:
                distributions = None
            
            advanced_analysis[col] = {
                'rolling_analysis': {
                    'rolling_mean': rolling_mean.tolist() if rolling_mean is not None else None,
                    'rolling_std': rolling_std.tolist() if rolling_std is not None else None
                },
                'extreme_values': extreme_values,
                'distribution_fits': distributions
            }
        return "\n".join(advanced_analysis)

    def generate_visualizations(self, df, numerical_cols, obj_cols):
        """
        Generate comprehensive visualizations for numerical and categorical data.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            numerical_cols (list): List of numerical column names
            obj_cols (list): List of categorical column names
        """
        # Determine layout for plots
        num_plots = len(numerical_cols)
        cols_per_row = 2
        rows = (num_plots + cols_per_row - 1) // cols_per_row

        # 1. Histogram Distribution Plots
        plt.figure(figsize=(15, 5 * rows))
        for i, col in enumerate(numerical_cols):
            plt.subplot(rows, cols_per_row, i + 1)
            plt.hist(df[col], bins=100)
            plt.title(f'Distribution of {col}')
            plt.xlabel(col)
            plt.ylabel('Frequency')
        plt.tight_layout()
        plt.savefig("1_distribution_plots.png")
        plt.close()

        # 2. Box Plots
        plt.figure(figsize=(15, 5 * rows))
        for i, col in enumerate(numerical_cols):
            plt.subplot(rows, cols_per_row, i + 1)
            plt.boxplot(df[col])
            plt.title(f'Box Plot of {col}')
            plt.ylabel(col)
        plt.tight_layout()
        plt.savefig("2_boxplots.png")
        plt.close()

        # 3. Violin Plots
        plt.figure(figsize=(15, 5 * rows))
        for i, col in enumerate(numerical_cols):
            plt.subplot(rows, cols_per_row, i + 1)
            sns.violinplot(x=df[col])
            plt.title(f'Violin Plot of {col}')
            plt.xlabel(col)
        plt.tight_layout()
        plt.savefig("3_violin_plots.png")
        plt.close()

        # 4. Kernel Density Estimation (KDE) Plots
        plt.figure(figsize=(15, 5 * rows))
        for i, col in enumerate(numerical_cols):
            plt.subplot(rows, cols_per_row, i + 1)
            sns.kdeplot(df[col], fill=True)
            plt.title(f'Density Plot of {col}')
            plt.xlabel(col)
            plt.ylabel('Density')
        plt.tight_layout()
        plt.savefig("4_kde_plots.png")
        plt.close()

        # 5. Correlation Heatmap
        numerical_df = df[numerical_cols]
        correlation_matrix = numerical_df.corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
        plt.title('Correlation Matrix Heatmap')
        plt.savefig('5_correlation_heatmap.png')
        plt.close()

        # # 6. Pair Plot
        # try:
        #     plt.figure(figsize=(15, 15))
        #     sns.pairplot(df[numerical_cols])
        #     plt.suptitle('Pair Plot of Numerical Columns', y=1.02)
        #     plt.savefig('6_pairplot.png')
        #     plt.close()
        # except Exception as e:
        #     print(f"Could not generate pair plot: {e}")

        # 7. Categorical Distribution Plots
        if obj_cols and len(obj_cols) > 0:
            plt.figure(figsize=(15, 5 * len(obj_cols)))
            for i, col in enumerate(obj_cols):
                plt.subplot(len(obj_cols), 1, i + 1)
                df[col].value_counts().plot(kind='bar')
                plt.title(f'Distribution of {col}')
                plt.xlabel('Categories')
                plt.ylabel('Count')
                plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig("7_categorical_distributions.png")
            plt.close()

        # # 8. Scatter Plot Matrix
        # plt.figure(figsize=(15, 15))
        # pd.plotting.scatter_matrix(df[numerical_cols], figsize=(15, 15), 
        #                             diagonal='kde', alpha=0.7)
        # plt.suptitle('Scatter Plot Matrix', y=0.95)
        # plt.savefig('8_scatter_matrix.png')
        # plt.close()

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
       # self.create_analysis_directory(file_path)
        # Analyze column types
        column_type_desc, object_cols, numerical_cols = self.analyze_column_types(df)

        # Collect analysis components
        compact_summary = [column_type_desc]
         
       
        object_columns_summary = self._call_llm_api(
            f"Analyze these column statistics from a Data Scientist perspective: {self.analyze_object_columns(df, object_cols)}"
        )
        compact_summary.append(object_columns_summary)
        # Calculate statistics
        numerical_stats = self.analyze_numerical_columns(df, numerical_cols)
        column_stats_summary = self._call_llm_api(
            f"Analyze these column statistics from a Data Scientist perspective: {numerical_stats}"
        )
        compact_summary.append(column_stats_summary)

        advan_numerical_stats = self.additional_advanced_analysis(df, numerical_cols)
        column_stats_summary = self._call_llm_api(
            f"Analyze these column statistics from a Data Scientist perspective: {advan_numerical_stats}"
        )
        compact_summary.append(column_stats_summary)

        # Generate visualizations
        self.generate_visualizations(df, numerical_cols,object_cols)

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
