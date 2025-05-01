"""Pandas DataFrame analysis toolkit.

This module provides comprehensive analysis capabilities for pandas DataFrames,
including statistical summaries, visualizations, and HTML report generation.

Key Features:
- Automated analysis for numeric, categorical and datetime data
- Interactive visualizations using Plotly
- HTML report generation with styled tables and charts
- Built-in data quality checks and outlier detection

Example:
    >>> from pagri_data_tools import DataAnalyzer
    >>> analyzer = DataAnalyzer(df)
    >>> report = analyzer.analyze_column("age")
    >>> print(report)  # HTML formatted analysis
"""

from typing import Union, List, Dict, Tuple, Any
from enum import Enum, auto
import io
import base64
from pandas.io.formats.style import Styler
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from IPython.display import display
from scipy import stats
from sklearn.ensemble import IsolationForest
from statsmodels import robust
import numpy as np
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from nltk.sentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
import plotly.express as px
from typing import Optional, Dict, Union
from nltk import download as nltk_download
import pagri_data_tools as pgdt
from plotly.subplots import make_subplots
    

class ColumnType(Enum):
    """Supported column data types"""

    NUMERIC = auto()
    OBJECT = auto()
    DATE = auto()


class DataAnalyzer:
    """Comprehensive pandas DataFrame analysis tool.

    Provides automated statistical analysis, visualization and reporting
    capabilities for pandas DataFrames.

    Key Methods:
        - analyze_column(): Generate complete analysis report for a column
        - _generate_*_summary(): Type-specific statistical summaries
        - _generate_*_chart(): Interactive visualizations

    Example:
        >>> analyzer = DataAnalyzer(df)
        >>> # Analyze numeric column
        >>> report = analyzer.analyze_column("age")
        >>> # Analyze categorical column with dual histogram
        >>> report = analyzer.analyze_column("category", hist_mode="dual")

    Args:
        df: Input pandas DataFrame for analysis

    Attributes:
        df: The DataFrame being analyzed
        display_mode: Histogram display mode (SIMPLE or DUAL)
    """
    def __init__(self, df: pd.DataFrame, visualization_config: Optional[Dict] = None):
        """
        Initialize DataAnalyzer with optional visualization configuration.
        
        Args:
            df: Input pandas DataFrame for analysis
            visualization_config: Dictionary with visualization settings including:
                - colors: Color palette for charts
                - sizes: Default dimensions for visualizations
                - font: Font settings
                - template: Plotly template name
        """
        self._validate_input(df)
        self.df = df
        self._cache = {}
        self.visualization_config = visualization_config or {
            'colors': px.colors.qualitative.Plotly,
            'sizes': {'width': 800, 'height': 500},
            'font': {'family': "Arial", 'size': 12},
            'template': 'plotly_white'
        }

    def _validate_input(self, df: pd.DataFrame):
        """Validate input DataFrame"""
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Input must be pandas DataFrame")
        if len(df) == 0:
            raise ValueError("DataFrame is empty")

    def analyze_column(self, column: str) -> str:
        """
        Generate complete analysis report for specified column.

        Args:
            column: Column name to analyze

        Returns:
            HTML formatted analysis report

        Raises:
            ValueError: For invalid column or unsupported hist_mode
        """
        if column not in self.df.columns:
            raise ValueError(f"Column '{column}' not found in DataFrame")

        col_type = self._get_column_type(column)
        funcs = [self._generate_dataframe_overview]
        funcs += self._get_analysis_functions(col_type)

        return self._format_column_report(column, funcs)

    # ========
    # All dataframe
    # ========
    def generate_dataframe_overview(self) -> Styler:
        """
        Generates a styled overview table with key statistics about the DataFrame.

        Calculates:
        - Total rows and columns
        - Memory usage
        - Duplicate rows (both exact and fuzzy matches)

        Returns:
            A pandas Styler object with formatted overview table ready for HTML display
        """
        # Calculate basic statistics
        total_rows = self.df.shape[0]
        total_cols = self.df.shape[1]
        ram = round(self.df.__sizeof__() / 1_048_576)
        if ram == 0:
            ram = "<1 Mb"
        # Calculate missing values
        total_cells = total_rows * total_cols
        missing_cells = self.df.isna().sum().sum()
        missing_cells = self._format_count_with_percentage(missing_cells, total_cells)
        # Calculate column type counts
        def is_float_column(col):
            if not pd.api.types.is_numeric_dtype(self.df[col]):
                return False
            col_dropped = self.df[col].dropna()
            if col_dropped.empty:
                return False
            return sum(col_dropped % 1) != 0
        # for col in self.df.columns:
        #     print(is_int_column(col))
        col_types = {
            'Text': sum(self._is_text_column(col) for col in self.df.columns),
            'Categorical': sum(self._is_categorical_column(col) for col in self.df.columns),
            'Int': sum(self._is_int_column(col) for col in self.df.columns),
            'Float': sum(self._is_float_column(col) for col in self.df.columns),
            'Datetime': sum(self._is_datetime_column(col) for col in self.df.columns)
        }        
        # Calculate duplicate statistics
        exact_dups = self._calculate_duplicates_in_df(exact=True)
        fuzzy_dups = self._calculate_duplicates_in_df(exact=False)

        # Create summary table
        summary_data = {
            "Rows": self._format_number(total_rows),
            "Features": total_cols,
            "Missing cells": missing_cells,
            "Exact Duplicates": exact_dups,
            "Fuzzy Duplicates": fuzzy_dups,
            "Memory Usage": ram,
        }

        # Convert to DataFrame and style
        summary_df = pd.DataFrame.from_dict(
            summary_data, orient="index", columns=["Value"]
        ).reset_index(names='Metric')
        # Convert to DataFrame and style
        col_types_df = pd.DataFrame.from_dict(
            col_types, 
            orient="index", 
            columns=["Value"]
        ).reset_index(names='Metric')
        col_types_df['Value'] = col_types_df['Value'].astype(str)
        # Concatenate all DataFrames along the columns (axis=1)
        full_summary = pd.concat([summary_df, col_types_df], axis=1)
        full_summary.columns = pd.MultiIndex.from_tuples(
            [('Summary', 'Metric'), ('Summary', 'Value'),
            ('Column Types', 'Metric'), ('Column Types', 'Value')]

        )
        full_summary = self._add_empty_columns_for_df(full_summary, [2])
        caption='Dataframe Overview'
        styled_summary = self._style_dataframe(full_summary, level=1, caption=caption, header_alignment='center')
        return styled_summary  
    def generate_combined_numeric_report(self, column: str) -> Tuple[Styler, go.Figure]:
        """Generate combined numeric report with summary and histogram"""
        if column not in self.df.columns:
            raise ValueError(f"Column '{column}' not found")
        
        if not pd.api.types.is_numeric_dtype(self.df[column]):
            raise ValueError("Column must be numeric")
        
        summary = self._generate_full_numeric_summary(self.df[column])
        fig = self._generate_histogram(self.df[column])
        return summary, fig

    def generate_combined_categorical_report(self, column: str) -> Tuple[Styler, go.Figure]:
        """Generate combined categorical report with summary and visualizations"""
        if column not in self.df.columns:
            raise ValueError(f"Column '{column}' not found")
        
        if not (pd.api.types.is_string_dtype(self.df[column]) or 
                self._is_categorical_column(column)):
            raise ValueError("Column must be categorical/text")
        
        summary = self._generate_summary_for_categorical(self.df[column])
        fig = self._generate_combined_charts(self.df[column])
        return summary, fig

    def generate_combined_datetime_report(self, column: str) -> Tuple[Styler, go.Figure]:
        """Generate combined datetime report with summary and visualizations"""
        if column not in self.df.columns:
            raise ValueError(f"Column '{column}' not found")
        
        if not pd.api.types.is_datetime64_any_dtype(self.df[column]):
            raise ValueError("Column must be datetime")
        
        summary = self._generate_full_datetime_summary(self.df[column])
        fig = self._generate_datetime_visualizations(self.df[column])
        return summary, fig    
    
    # ========
    # Datetime
    # ========
    def _generate_basic_stats_datetime(self, column: pd.Series) -> pd.DataFrame:
        """
        Returns basic datetime statistics with exactly specified fields.
        """
        valid_dates = column.dropna()
        timedeltas = valid_dates.sort_values().diff().dropna() if len(valid_dates) > 1 else pd.Series(dtype='timedelta64[ns]')
        ram = round(column.memory_usage(deep=True) / 1048576)
        if ram == 0:
            ram = "<1 Mb"
        stats = {
            "First date": valid_dates.min().date() if not valid_dates.empty else "N/A",
            "Last date": valid_dates.max().date() if not valid_dates.empty else "N/A",
            "Avg Days Frequency": str(round(timedeltas.mean().total_seconds() / (24 * 3600), 2)) if not timedeltas.empty else "N/A",
            "Min Days Interval": timedeltas.min().days if not timedeltas.empty else "N/A",
            "Max Days Interval": timedeltas.max().days if not timedeltas.empty else "N/A",
            "Memory Usage": ram
        }
        return pd.DataFrame(stats.items(), columns=["Metric", "Value"])

    def _generate_data_quality_stats_datetime(self, column: pd.Series) -> pd.DataFrame:
        """
        Returns data quality stats for datetime column.
        """
        valid_dates = column.dropna()
        value_counts = column.value_counts()
        len_column = len(column)
        values = column.count()
        formatted_values = self._format_count_with_percentage(values, len_column)
        zeros = (valid_dates == 0).sum()
        formatted_zeros = self._format_count_with_percentage(zeros, len_column)
        missings = column.isna().sum()
        formatted_missings = self._format_count_with_percentage(missings, len_column)
        distinct = valid_dates.nunique()
        formatted_distinct = self._format_count_with_percentage(distinct, len_column)
        duplicates = column.duplicated().sum()
        formatted_duplicates = self._format_count_with_percentage(duplicates, len_column)
        dup_values = column.value_counts()[column.value_counts() > 1].count()
        formatted_dup_values = self._format_count_with_percentage(dup_values, len_column)
        stats = {
            "Values": formatted_values,
            "Zeros": formatted_zeros,
            "Missings": formatted_missings,
            "Distinct": formatted_distinct,
            "Duplicates": formatted_duplicates,
            "Dup. Values": formatted_dup_values
        }
        return pd.DataFrame(stats.items(), columns=["Metric", "Value"])

    def _generate_temporal_stats_datetime(self, column: pd.Series) -> pd.DataFrame:
        """
        Returns temporal patterns stats for datetime column.
        """
        valid_dates = column.dropna()
        stats = {}
        
        if not valid_dates.empty:
            date_range = pd.date_range(valid_dates.min(), valid_dates.max())
            missing_days = date_range.difference(valid_dates)
            
            stats = {
                "Missing Years": len(set(date_range.year) - set(valid_dates.dt.year)),
                "Missing Months": len(set(zip(date_range.year, date_range.month)) - 
                                    set(zip(valid_dates.dt.year, valid_dates.dt.month))),
                "Missing Weeks": len(set(zip(date_range.year, date_range.isocalendar().week)) - 
                                set(zip(valid_dates.dt.year, valid_dates.dt.isocalendar().week))),
                "Missing Days": len(missing_days),
                "Weekend Percentage": f"{valid_dates.dt.weekday.isin([5,6]).mean():.1%}",
                "Most Common Weekday": valid_dates.dt.day_name().mode()[0] if not valid_dates.empty else "N/A"
            }
        
        return pd.DataFrame(stats.items(), columns=["Metric", "Value"])
    
    def _generate_full_datetime_summary(self, column: pd.Series) -> pd.DataFrame:
        """
        Combines all datetime statistics into a single DataFrame.
        """
        column_name = column.name
        basic_stats = self._generate_basic_stats_datetime(column)
        data_quality_stats = self._generate_data_quality_stats_datetime(column)
        temporal_stats = self._generate_temporal_stats_datetime(column)
        # Concatenate all DataFrames along the columns (axis=1)
        full_summary = pd.concat([basic_stats, data_quality_stats, temporal_stats], axis=1)

        full_summary.columns = pd.MultiIndex.from_tuples(
            [('Summary', 'Metric'), ('Summary', 'Value'),
            ('Data Quality Stats', 'Metric'), ('Data Quality Stats', 'Value'),
            ('Temporal Stats', 'Metric'), ('Temporal Stats', 'Value')]

        )
        full_summary = self._add_empty_columns_for_df(full_summary, [2, 4])
        caption = f'Summary Statistics for "{column_name}" (Type: Datetime)'
        styled_summary = self._style_dataframe(full_summary, level=1, caption=caption, header_alignment='center')
        return styled_summary    

    def _generate_datetime_visualizations(self, column: pd.Series) -> go.Figure:
        """Generate interactive visualizations for datetime columns"""
        # Create subplots: 1 row, 2 columns
        fig = make_subplots(rows=1, cols=2, 
                        subplot_titles=("Timeline Distribution", "Temporal Patterns"))
        
        # Timeline histogram
        fig.add_trace(
            go.Histogram(x=column, name="Distribution"),
            row=1, col=1
        )
        
        # Temporal patterns - aggregate by different time units
        df_temp = pd.DataFrame({'date': column, 'count': 1})
        df_temp['year'] = df_temp['date'].dt.year
        df_temp['month'] = df_temp['date'].dt.month
        df_temp['weekday'] = df_temp['date'].dt.weekday
        
        # Year-Month heatmap
        year_month = df_temp.groupby(['year', 'month']).size().unstack()
        
        fig.add_trace(
            go.Heatmap(
                z=year_month.values,
                x=year_month.columns,
                y=year_month.index,
                colorscale='Blues',
                name="Year-Month"
            ),
            row=1, col=2
        )
        
        # Update layout
        fig.update_layout(
            title_text=f"Datetime Analysis: {column.name}",
            height=400,
            showlegend=False,
        )
        
        # Update axes
        fig.update_xaxes(title_text="Date", row=1, col=1)
        fig.update_yaxes(title_text="Count", row=1, col=1)
        fig.update_xaxes(title_text="Month", row=1, col=2)
        fig.update_yaxes(title_text="Year", row=1, col=2)
        
        return fig

    def _generate_datetime_distribution_plot(self, column: pd.Series) -> go.Figure:
        """Generate distribution plot for datetime column"""
        fig = px.histogram(
            x=column,
            title=f"Distribution of {column.name}",
            labels={'x': column.name},
            template='plotly_white'
        )
        
        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Count",
        )
        
        return fig

    def _generate_temporal_patterns_plot(self, column: pd.Series) -> go.Figure:
        """Generate temporal patterns visualization"""
        # Create dataframe with extracted temporal features
        df_temp = pd.DataFrame({
            'date': column,
            'year': column.dt.year,
            'month': column.dt.month,
            'weekday': column.dt.weekday,
            'hour': column.dt.hour
        })
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=("By Year", "By Month", "By Weekday", "By Hour")
        )
        
        # Year plot
        year_counts = df_temp['year'].value_counts().sort_index()
        fig.add_trace(
            go.Bar(x=year_counts.index, y=year_counts.values, name="Year"),
            row=1, col=1
        )
        
        # Month plot
        month_counts = df_temp['month'].value_counts().sort_index()
        fig.add_trace(
            go.Bar(x=month_counts.index, y=month_counts.values, name="Month"),
            row=1, col=2
        )
        
        # Weekday plot
        weekday_counts = df_temp['weekday'].value_counts().sort_index()
        fig.add_trace(
            go.Bar(x=weekday_counts.index, y=weekday_counts.values, name="Weekday"),
            row=2, col=1
        )
        
        # Hour plot (if available)
        if 'hour' in df_temp:
            hour_counts = df_temp['hour'].value_counts().sort_index()
            fig.add_trace(
                go.Bar(x=hour_counts.index, y=hour_counts.values, name="Hour"),
                row=2, col=2
            )
        
        # Update layout
        fig.update_layout(
            title_text=f"Temporal Patterns: {column.name}",
            height=600,
            showlegend=False,
            hovermode='x unified'
        )
        
        return fig
    # ========
    # Numeric
    # ========
    def _get_cached_result(self, column: pd.Series, func_name: str) -> Any:
        """
        Gets cached result or calculates and caches if not available.
        
        Args:
            column: Column to use as cache key
            func_name: Name of the generating function
            
        Returns:
            Cached or newly calculated result
        """
        cache_key = (column.name, func_name)
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        # Dynamically call the function based on name
        func = getattr(self, func_name)
        result = func(column)
        self._cache[cache_key] = result
        return result

    def _generate_summary_for_numeric(self, column: pd.Series) -> pd.DataFrame:
        """
        Generates basic summary statistics for numeric columns in HTML-friendly format.
        
        Note: Results are cached based on column name
        """
        len_column = len(column)
        # Calculate basic metrics
        values = column.count()
        values = self._format_count_with_percentage(values, len_column)
        
        # Missing values
        missing = column.isna().sum()
        missing = self._format_count_with_percentage(missing, len_column) if missing else '---'
        
        # Distinct values
        distinct = column.nunique()
        distinct = self._format_count_with_percentage(distinct, len_column)

        # Non-Duplicate 
        unique_once = column.value_counts()
        unique_once = (unique_once == 1).sum()
        unique_once = self._format_count_with_percentage(unique_once, len_column)

        # Duplicates
        duplicates = column.duplicated().sum()
        duplicates = self._format_count_with_percentage(duplicates, len_column) if duplicates > 0 else "---"

        # Count of values with duplicates
        values_with_duplicates = column.value_counts()[column.value_counts() > 1].count()        
        values_with_duplicates = self._format_count_with_percentage(values_with_duplicates, len_column) if duplicates != "---" else "---"

        # Zeros and negatives
        zeros = (column == 0).sum()
        zeros = self._format_count_with_percentage(zeros, len_column) if zeros > 0 else "---"
        
        negative = (column < 0).sum()
        negative = self._format_count_with_percentage(negative, len_column) if negative > 0 else "---"

        # Infinite
        infinite = np.isinf(column).sum()
        infinite = self._format_count_with_percentage(infinite, len_column) if infinite > 0 else "---"

        # Memory usage
        ram = round(column.memory_usage(deep=True) / 1048576)
        if ram == 0:
            ram = "<1 Mb"
        
        df_res = pd.DataFrame({
            "Total": [values],
            "Missing": [missing],
            "Distinct": [distinct],
            "Non-Duplicate": [unique_once],
            "Duplicates": [duplicates],
            "Dup. Values": [values_with_duplicates],
            "Zeros": [zeros],
            "Negative": [negative],
            "Memory Usage": [ram]
        }).T.reset_index()
        df_res.columns = ['Metric', 'Value']
        return df_res
    
    def _generate_percentiles_for_numeric(self, column: pd.Series) -> pd.DataFrame:
        """
        Generates percentile statistics for numeric columns.
        """
        percentiles = {
            "Max": column.max(),
            "99%": column.quantile(0.99),
            "95%": column.quantile(0.95),
            "75%": column.quantile(0.75),
            "50%": column.median(),
            "25%": column.quantile(0.25),
            "5%": column.quantile(0.05),
            "1%": column.quantile(0.01),
            "Min": column.min()
        }
        
        formatted = {k: self._format_number(v) for k, v in percentiles.items()}
        df_res = pd.DataFrame(formatted, index=[0]).T.reset_index()
        df_res.columns = ['Metric', 'Value']
        return df_res

    def _generate_stats_for_numeric(self, column: pd.Series) -> pd.DataFrame:
        """
        Generates statistical measures for numeric columns including trimmed mean.
        
        Args:
            column: Numeric pandas Series
            
        Returns:
            DataFrame with statistics in [Metric, Value] format
        """
     
        clean_col = column.dropna()
        
        stats_dict = {
            "Mean": clean_col.mean(),
            "Trimmed Mean (10%)": stats.trim_mean(clean_col, proportiontocut=0.1),
            "Mode": clean_col.mode()[0] if len(clean_col.mode()) == 1 else "Multiple",
            "Range": clean_col.max() - clean_col.min(),
            "IQR": clean_col.quantile(0.75) - clean_col.quantile(0.25),
            "Std": clean_col.std(),
            "MAD": robust.mad(clean_col),
            "Kurt": clean_col.kurtosis(),
            "Skew": clean_col.skew()
        }
        
        formatted = {
            k: self._format_number(v) if isinstance(v, (int, float)) else v 
            for k, v in stats_dict.items()
        }
        
        result = pd.DataFrame(formatted.items(), columns=['Metric', 'Value'])
        return result

    def _generate_value_counts_for_numeric(self, column: pd.Series, top_n: int = 9) -> pd.DataFrame:
        """
        Generates value counts for numeric columns with binning.
        """
        # Handle empty/missing data
        clean_col = column.dropna()
        len_column = len(column)
        if len(clean_col) == 0:
            return pd.DataFrame({"Message": ["No numeric data available"]})
        # Get top distinct values
        top_values = clean_col.value_counts().head(top_n)
        # Create a DataFrame for results
        df_res = pd.DataFrame({
            'Value': top_values.index,
            'Count': top_values.values,
        })
        # Format the Percent column to show "<1%" for values less than 1
        df_res['Count'] = df_res['Count'].apply(lambda x: self._format_count_with_percentage(x, len_column))
        return df_res.reset_index(drop=True)
    
    def _combine_numeric_stats(self, *stats_dfs: pd.DataFrame) -> pd.DataFrame:
        """
        Combines multiple statistics DataFrames into a single styled summary.
        
        Args:
            *stats_dfs: Variable number of DataFrames to combine
            
        Returns:
            Styled DataFrame with multi-level columns
        """
        # Concatenate all DataFrames along the columns (axis=1)
        combined_df = pd.concat(stats_dfs, axis=1)
        
        # Create multi-level column names
        column_tuples = []
        for i, df in enumerate(stats_dfs):
            section_name = [
                'Summary', 'Percentiles', 
                'Detailed Stats', 'Value Counts'
            ][i]
            for col in df.columns:
                column_tuples.append((section_name, col))
                
        combined_df.columns = pd.MultiIndex.from_tuples(column_tuples)
        return combined_df

    def _generate_full_numeric_summary(self, column: pd.Series) -> pd.DataFrame:
        """
        Generates complete styled summary for numeric columns.
        
        Args:
            column: Numeric pandas Series to analyze
            
        Returns:
            Styled DataFrame with all statistics
        """
        column_name = column.name
        column_type = self._get_column_type(column_name)
        
        # Generate all component statistics
        stats = [
            self._generate_summary_for_numeric(column),
            self._generate_percentiles_for_numeric(column),
            self._generate_stats_for_numeric(column),
            self._generate_value_counts_for_numeric(column)
        ]
        
        # Combine and style
        full_summary = self._combine_numeric_stats(*stats)
        full_summary = self._add_empty_columns_for_df(full_summary, [2, 4, 6])
        caption = f'Summary Statistics for "{column_name}" (Type: {column_type})'
        return self._style_dataframe(
            full_summary, 
            level=1, 
            caption=caption, 
            header_alignment='center'
        )

    def _generate_histogram(self, column: pd.Series, **kwargs) -> go.Figure:
        """
        Generates an interactive histogram with box plot for numeric data.

        Args:
            column: Numeric pandas Series to visualize
            **kwargs: Override default visualization settings:
                - color: Bar color
                - width: Figure width
                - height: Figure height
                - title: Custom title

        Returns:
            Plotly Figure object with histogram visualization
        """
        # Merge config with kwargs (kwargs take precedence)
        config = {**self.visualization_config, **kwargs}
        
        fig = pgdt.histogram(
            x=column,
            mode='dual_hist_qq',
            title=config.get('title', f'Histogram and qq-plot for "{column.name}"'),
            width=config['sizes']['width'],
            height=config['sizes']['height'],
            template=config['template']
        )
        
        # Update font if specified
        if 'font' in config:
            fig.update_layout(
                font_family=config['font']['family'],
                font_size=config['font']['size']
            )
            
        return fig

    # ========
    # Categorical
    # ========
    def _generate_summary_for_categorical(self, column: pd.Series) -> pd.DataFrame:
        """
        Generates comprehensive summary for categorical/text columns
        Returns DataFrame with metrics in consistent format
        """
        column_name = column.name
        # Basic counts
        total = len(column)
        non_null = column.count()
        missing = column.isna().sum()
        empty = (column.str.strip() == "").sum()
        
        # Distinct values analysis
        distinct = column.nunique()
        unique_once = column.value_counts().eq(1).sum()
        most_common = column.mode()
        most_common_count = (column == most_common[0]).sum() if len(most_common) > 0 else 0
        
        # Duplicates analysis
        exact_duplicates = column.duplicated().sum()
        
        # Fuzzy duplicates (case and whitespace insensitive)
        fuzzy_duplicates = (
            column.str.lower()
            .str.strip()
            .str.replace(r'\s+', ' ', regex=True)
            .duplicated()
            .sum()
        )

        # Count of values with duplicates
        values_with_duplicates = column.value_counts()[column.value_counts() > 1].count()
        
        # Text length analysis 
        lengths = column.str.len()
        avg_len = lengths.mean()
        median_len = lengths.median()
        min_len = lengths.min()
        max_len = lengths.max()
        word_counts = column.str.split().str.len()
        avg_words = word_counts.mean()
        digit_count = column.str.count(r'\d') 
        avg_digit_ratio = (digit_count / lengths.replace(0, np.nan)).mean()
        most_common_length = lengths.mode()

        if not most_common_length.empty:
            most_common_length_value = most_common_length[0]
            most_common_length_count = lengths.value_counts().iloc[0]
            most_common_length_str = f"{most_common_length_value} ({most_common_length_count / len(lengths):.1%})"
        else:
            most_common_length_str = "N/A"

        ram = round(column.memory_usage(deep=True) / 1048576)
        if ram == 0:
            ram = "<1 Mb"
        # Prepare metrics
        quality_metrics = {
            "Total Values": self._format_count_with_percentage(total, total),
            "Missing Values": self._format_count_with_percentage(missing, total),
            "Empty Strings": self._format_count_with_percentage(empty, total),
            "Distinct Values": self._format_count_with_percentage(distinct, total),
            "Non-Duplicates": self._format_count_with_percentage(unique_once, total),
            "Exact Duplicates": self._format_count_with_percentage(exact_duplicates, total),
            "Fuzzy Duplicates": self._format_count_with_percentage(fuzzy_duplicates, total),
            "Values with Duplicates": self._format_count_with_percentage(values_with_duplicates, total),
            "Memory Usage": ram,
        }
        text_metrics = {
            "Avg Word Count": f"{avg_words:.1f}" if not pd.isna(avg_words) else "N/A",
            "Max Length (chars)": f"{max_len:.1f}" if not pd.isna(max_len) else "N/A",
            "Avg Length (chars)": f"{avg_len:.1f}" if not pd.isna(avg_len) else "N/A",
            "Median Length (chars)": f"{median_len:.1f}" if not pd.isna(median_len) else "N/A",
            "Min Length (chars)": f"{min_len:.1f}" if not pd.isna(min_len) else "N/A",
            "Most Common Length": most_common_length_str,
            "Avg Digit Ratio": avg_digit_ratio
        }
        
        quality_metrics_df = pd.DataFrame(quality_metrics.items(), columns=["Metric", "Value"])
        text_metrics_df = pd.DataFrame(text_metrics.items(), columns=["Metric", "Value"])
        # Concatenate all DataFrames along the columns (axis=1)
        full_summary = pd.concat([quality_metrics_df, text_metrics_df], axis=1)
        full_summary.columns = pd.MultiIndex.from_tuples(
            [('Summary', 'Metric'), ('Summary', 'Value'),
            ('Text Metrics', 'Metric'), ('Text Metrics', 'Value')]
        )
        full_summary = self._add_empty_columns_for_df(full_summary, [2])
        column_type = self._get_column_type(column_name)
        caption = f'Summary Statistics for "{column_name}" (Type: {column_type})'
        styled_summary = self._style_dataframe(full_summary, level=1, caption=caption, header_alignment='center')
        return styled_summary
        
    def _generate_value_counts_for_categorical(self, column: pd.Series, top_n: int = 10) -> pd.DataFrame:
        """
        Analyzes categorical value distribution and returns formatted counts.
        
        Args:
            column: Categorical pandas Series to analyze
            top_n: Number of top values to show (default: 10)
            
        Returns:
            DataFrame with columns: ['Value', 'Count', 'Percent']
            where Count is formatted as "count (percentage%)"
        """
        # Calculate frequencies
        total_count = len(column)
        value_counts = column.value_counts()
        value_percents = column.value_counts(normalize=True) * 100
        
        # Handle empty data
        if total_count == 0:
            return pd.DataFrame({"Message": ["No data available"]})
        
        # Prepare top values
        results = []
        for val, count in value_counts.head(top_n).items():
            percent = value_percents[val]
            results.append({
                'Value': str(val)[:50] + '...' if len(str(val)) > 50 else str(val),
                'Count': count,
                'Percent': percent
            })
        
        # Create DataFrame
        df_result = pd.DataFrame(results)
        
        # Format counts with percentages
        df_result['Count'] = df_result.apply(
            lambda row: self._format_count_with_percentage(row['Count'], total_count),
            axis=1
        )
        
        # Add missing values row if present
        if column.isna().sum() > 0:
            missing_row = {
                'Value': 'Missing',
                'Count': column.isna().sum(),
                'Percent': (column.isna().sum() / total_count) * 100
            }
            df_result = pd.concat([
                df_result,
                pd.DataFrame([missing_row])
            ], ignore_index=True)
        
        # Add empty strings row if present (for string columns)
        if pd.api.types.is_string_dtype(column) and (column == "").sum() > 0:
            empty_count = (column == "").sum()
            empty_row = {
                'Value': 'Empty Strings',
                'Count': empty_count,
                'Percent': (empty_count / total_count) * 100
            }
            df_result = pd.concat([
                df_result,
                pd.DataFrame([empty_row])
            ], ignore_index=True)
        
        return df_result[['Value', 'Count']]  # Return same columns as numeric version
    
    def _generate_bar_chart(
        self, column: pd.Series, top_n: int = 10
    ) -> go.Figure:
        """
        Generates an interactive horizontal bar chart for categorical data.

        Args:
            column: Categorical pandas Series to visualize
            top_n: Number of top categories to display (default: 10)

        Returns:
            Plotly Figure object with these features:
            - Horizontal bars sorted by frequency
            - Value annotations
            - Adaptive text wrapping
            - Smart truncation for long labels
            - Responsive design settings

        Visual Features:
        - Top N categories by count
        - Percentage and absolute value labels
        - Color gradient by frequency
        - Dynamic label sizing
        - Mobile-optimized layout
        """
        # Prepare data - count values and calculate percentages
        value_counts = column.value_counts().nlargest(top_n)
        percentages = (value_counts / len(column)) * 100
        df_plot = pd.DataFrame(
            {
                "Category": value_counts.index,
                "Count": value_counts.values,
                "Percentage": percentages.round(1),
            }
        )

        # Truncate long category names
        max_label_length = 30
        df_plot["DisplayName"] = df_plot["Category"].apply(
            lambda x: (
                (x[:max_label_length] + "...")
                if len(x) > max_label_length
                else x
            )
        )

        # Create figure
        fig = px.bar(
            df_plot,
            x="Count",
            y="DisplayName",
            orientation="h",
            text="Percentage",
            template="plotly_white",
            height=400,
            width=700,
        )

        # Style configuration
        fig.update_traces(
            texttemplate="%{text}%",
            textfont_size=12,
            marker_line_width=0.5,
            hovertemplate=(
                "<b>%{y}</b><br>"
                + "Count: %{x:,}<br>"
                + "Percentage: %{text}%"
                + "<extra></extra>"
            ),
        )

        # Layout configuration
        fig.update_layout(
            title=f"Top {top_n} Categories in {column.name}",
            title_x=0.5,
            title_font_size=16,
            margin=dict(
                l=120, r=40, t=80, b=40
            ),  # Extra left margin for labels
            xaxis_title="Count",
            yaxis_title=None,
            yaxis={"categoryorder": "total ascending"},
            plot_bgcolor="white",
            paper_bgcolor="white",
            xaxis={"gridcolor": "#f0f0f0", "tickformat": ","},
            uniformtext_minsize=8,
            uniformtext_mode="hide",
        )

        # Add full category names as hover text
        fig.update_traces(
            customdata=df_plot["Category"],
            hovertemplate="<b>%{customdata}</b><br>Count: %{x:,}<br>Percentage: %{text}%<extra></extra>",
        )

        return fig

    def _generate_wordcloud(self, column: pd.Series, 
                       max_words: int = 100,
                       background_color: str = 'white',
                       colormap='viridis',
                       width: int = 800,
                       height: int = 400,
                       collocations: bool = True) -> plt.Figure:
        """
        Generates a word cloud visualization for categorical/text data
        
        Args:
            column: Categorical or text column to analyze
            max_words: Maximum number of words to display (default: 100)
            background_color: Background color (default: 'white')
            width: Image width in pixels (default: 800)
            height: Image height in pixels (default: 400)
            stopwords: Set of words to exclude (default: None)
            collocations: Whether to include bigrams (default: True)
            
        Returns:
            Matplotlib Figure object with the word cloud visualization
        """
        # Prepare text data
        text = ' '.join(column.dropna().astype(str))
        # Create the word cloud
        wordcloud = WordCloud(
            width=width,
            height=height,
            background_color=background_color,
            max_words=max_words,
            colormap=colormap,
            collocations=collocations,
        ).generate(text)

        # Create figure
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        fig.tight_layout()
        return fig
    
    def _generate_combined_charts(self, column: pd.Series, top_n: int = 10, max_words: int = 100) -> go.Figure:
        """
        Generates combined visualization with bar chart and properly scaled word cloud.
        Args:
            column: Categorical/text pandas Series to visualize
            top_n: Number of top categories for bar chart (default: 10)
            max_words: Maximum words for word cloud (default: 100)
        Returns:
            Plotly Figure with both visualizations (word cloud maintains aspect ratio)
        """
        # 1. Create word cloud
        text = ' '.join(column.dropna().astype(str))
        wordcloud = WordCloud(width=2000, height=1000, background_color='white', max_words=max_words).generate(text)
        # Convert the word cloud to an array
        wordcloud_array = np.array(wordcloud)
        # 2. Create Plotly figure with subplots
        fig = make_subplots(rows=1, cols=2,
                        column_widths=[0.3, 0.7],
                        specs=[[{"type": "bar"}, {"type": "scatter"}]],
                        horizontal_spacing=0.01)
        # 3. Add bar chart (first subplot)
        bar_fig = self._generate_bar_chart(column, top_n)
        for trace in bar_fig.data:
            fig.add_trace(trace, row=1, col=1)
        # 4. Add word cloud using Plotly Express
        fig.add_trace(
            go.Image(
                z=wordcloud_array,
                hoverinfo='skip' 
            ),
            row=1, col=2
        )
        fig.update_yaxes(categoryorder="total ascending", row=1, col=1)
        fig.update_yaxes(visible=False, row=1, col=2)
        fig.update_xaxes(visible=False, row=1, col=2)
        # 5. Update layout for proper scaling
        fig.update_layout(
            title_text=f'Analysis of "{column.name}"',
            showlegend=False,
            plot_bgcolor='white',
            paper_bgcolor='white',
            margin=dict(l=10, r=10, b=10, t=50),
            height=350,
            width=1000,
        )
        return fig
    
    # ========
    # Helper methods
    # ========

    def _analyze_sentiment(self, 
                        column: str, 
                        method: str = 'vader',
                        clean_text=True) -> Tuple[go.Figure, pd.DataFrame]:
        """
        Perform sentiment analysis on text data.
        
        Args:
            column: Name of the text column to analyze
            method: 'vader' or 'textblob' (default: 'vader')
            
        Returns:
            Tuple containing:
            - Plotly Figure with histogram
            - DataFrame with statistics
        """
        if column not in self.df.columns:
            raise ValueError(f"Column '{column}' not found in DataFrame")
            
        if not pd.api.types.is_string_dtype(self.df[column]):
            raise ValueError("Column must contain text data")
        if clean_text:
            text_series = self.df[column].str.replace(r'[^\w\s]', '', regex=True).str.lower()
        else:
            text_series = self.df[column]
        # Initialize analyzer and calculate scores
        if method == 'vader':
            nltk_download('vader_lexicon', quiet=True)
            analyzer = SentimentIntensityAnalyzer()
            scores = text_series.apply(
                lambda x: analyzer.polarity_scores(str(x))['compound']
            )
        elif method == 'textblob':
            scores = text_series.apply(
                lambda x: TextBlob(str(x)).sentiment.polarity
            )
        else:
            raise ValueError("Method must be either 'vader' or 'textblob'")
        # Categorize sentiment
        def categorize(score):
            if score > 0: return "Positive"
            elif score < 0: return "Negative"
            return "Neutral"
            
        # Calculate statistics
        avg_sentiment = round(scores.mean(), 2)
        sentiment_75_pct = round(scores.quantile(0.75), 2)
        median_sentiment = round(scores.median(), 2)
        sentiment_25_pct = round(scores.quantile(0.25), 2)

        # Categorize sentiment
        scores_category = scores.apply(categorize)

        # Count sentiment categories
        sentiment_counts = scores_category.value_counts().astype(str).reset_index()
        sentiment_counts.columns = ['col1', 'col2']

        # Create result DataFrame
        statistics = pd.DataFrame(
            {
                "Mean": [avg_sentiment],
                "75%": [sentiment_75_pct],
                "Median": [median_sentiment],
                "25%": [sentiment_25_pct],
            }
        )
        statistics = statistics.T.astype(str).reset_index()
        statistics.columns = ['col1', 'col2']
        res_df = pd.concat([statistics, sentiment_counts], axis=1).T.reset_index(drop=True).T
        res_df = self._add_empty_columns_for_df(res_df, [2])
        res_df = res_df.fillna('')
        # Create histogram
        res_df = self._style_dataframe(res_df, caption='Sentiment analysis')
        # Create histogram
        labels = dict(x='Sentiment Score')
        fig = pgdt.histogram(
            x=scores, 
            labels=labels, 
            title=f'Sentiment Distribution ({method})',
            color_discrete_sequence=['#4C78A8']
        )
        return fig, res_df    

    def _format_number(self, num: Union[int, float]) -> str:
        """
        Format numbers with appropriate scaling and decimal precision for display.

        Args:
            num: Number to format (int or float)

        Returns:
            Formatted string with:
            - Thousands separators for large numbers
            - Appropriate unit suffixes (k, m, b)
            - Smart decimal precision

        Examples:
            >>> _format_number(1234.5678)
            '1,234.57'

            >>> _format_number(1_500_000)
            '1.50m'

            >>> _format_number(3_000_000_000)
            '3.00b'
        """
        # Handle None/NaN cases
        if pd.isna(num):
            return "NA"

        # Handle zero case
        if num == 0:
            return "0"

        abs_num = abs(num)

        # Determine appropriate scaling
        if abs_num < 1_000:
            # Small numbers - format with 2 decimals if not whole
            if isinstance(num, int) or num.is_integer():
                return f"{int(num):,}"
            return f"{num:,.2f}"

        elif abs_num < 1_000_000:
            scaled = num / 1_000
            return f"{scaled:,.2f}k"

        elif abs_num < 1_000_000_000:
            scaled = num / 1_000_000
            return f"{scaled:,.2f}m"

        elif abs_num < 1_000_000_000_000:
            scaled = num / 1_000_000_000
            return f"{scaled:,.2f}b"

        else:
            scaled = num / 1_000_000_000_000
            return f"{scaled:,.2f}t"

    def _format_count_with_percentage(
        self, count: int, total: int, precision: int = 0
    ) -> str:
        """
        Formats a count with its percentage of total in a human-readable way.

        Args:
            count: The subset count to format
            total: The total count for percentage calculation
            precision: Number of decimal places for percentages (default: 1)

        Returns:
            Formatted string in "count (percentage%)" format.
            Special cases:
            - Returns "0" if total is 0
            - Returns "count (100%)" if count equals total
            - Handles edge percentages (<1%, >99%)

        Examples:
            >>> _format_count_with_percentage(5, 100)
            '5 (5.0%)'

            >>> _format_count_with_percentage(999, 1000)
            '999 (99.9%)'

            >>> _format_count_with_percentage(1, 1000)
            '1 (<1%)'
        """
        # Handle zero total case
        if total == 0:
            return "0"

        # Handle 100% case
        if count == total:
            return f"{self._format_number(count)} (100%)"

        percentage = (count / total) * 100

        # Format percentage based on magnitude
        if percentage < 1:
            percentage_str = f"<{10**-precision}"  # <1% or <0.1% etc
        elif percentage > 99 and percentage < 100:
            percentage_str = f"{100 - 10**-precision:.{precision}f}"  # 99.9%
        else:
            percentage_str = f"{percentage:.{precision}f}"

        # Remove trailing .0 for whole numbers when precision=1
        if precision == 1 and percentage_str.endswith(".0"):
            percentage_str = percentage_str[:-2]

        return f"{self._format_number(count)} ({percentage_str}%)"

    def _format_frequency(
        self, value: Union[str, int], frequency: float
    ) -> str:
        """Helper to format value with frequency percentage"""
        return f"{value} ({frequency:.1%})"

    def _get_table_styles(
        self, font_size: str = "14px"
    ) -> List[Dict[str, Union[str, List]]]:
        """
        Defines consistent styling rules for all analysis tables.

        Args:
            font_size: Base font size for table elements (default '14px')

        Returns:
            List of CSS style dictionaries compatible with pandas Styler

        Style Features:
        - Clean minimalist design
        - Responsive column widths
        - Header highlighting
        - Zebra striping for readability
        - Customizable font properties
        """
        return [
            {
                "selector": "caption",
                "props": [
                    ("font-size", "18px"),
                    ("text-align", "left"),
                    ("font-weight", "bold"),
                ],
            }
        ]

    def _calculate_duplicates_in_df(self, exact: bool = True) -> str:
        """
        Calculate duplicate rows statistics with configurable matching strictness.

        Args:
            exact: If True, finds exact duplicates. If False, finds fuzzy matches
                  (ignoring case and whitespace differences for string columns)

        Returns:
            Formatted string with count and percentage of duplicates.
            Returns "---" if no duplicates found.

        Examples:
            >>> _calculate_duplicates_in_df(exact=True)
            "10 (1.5%)"

            >>> _calculate_duplicates_in_df(exact=False)
            "15 (2.2%)"
        """
        try:
            if exact:
                # Count exact duplicates
                dup_count = self.df.duplicated().sum()
            else:
                # Check if there are any string columns
                if not any(self.df.dtypes == "object"):
                    return "---"  # No string columns to check for fuzzy duplicates

                # Count fuzzy duplicates (normalized strings)
                dup_count = (
                    self.df.apply(
                        lambda col: (
                            col.str.lower()
                            .str.strip()
                            .str.replace(r"\s+", " ", regex=True)
                            if col.dtype == "object"
                            else col
                        )
                    )
                    .duplicated(keep=False)
                    .sum()
                )

            if dup_count == 0:
                return "---"

            # Calculate percentage
            total_rows = len(self.df)
            percentage = (dup_count / total_rows) * 100

            # Format percentage based on magnitude
            if 0 < percentage < 1:
                percentage_str = "<1"
            elif 99 < percentage < 100:
                percentage_str = f"{percentage:.1f}".replace("100.0", "99.9")
            else:
                percentage_str = f"{round(percentage)}"

            # Format final output
            formatted_count = self._format_number(dup_count)
            return f"{formatted_count} ({percentage_str}%)"

        except Exception as e:
            # Graceful fallback for any calculation errors
            # print(f"Duplicate calculation error: {str(e)}")
            return "---"

    def _format_timedelta(self, td: pd.Timedelta) -> str:
        """Format timedelta for display"""
        if pd.isna(td):
            return "N/A"
        days = td.days
        if days >= 365:
            return f"{days//365} year{'s' if days//365>1 else ''}"
        if days >= 30:
            return f"{days//30} month{'s' if days//30>1 else ''}"
        if days >= 7:
            return f"{days//7} week{'s' if days//7>1 else ''}"
        return f"{days} day{'s' if days>1 else ''}"

    def _format_frequency(self, value: str, frequency: float) -> str:
        """Format value with frequency percentage"""
        return f"{value} ({frequency:.1%})"

    def _add_empty_columns_for_df(self, df: pd.DataFrame, positions: list) -> pd.DataFrame:
        """
        Adds empty columns to the DataFrame at specified positions.
        Args:
            df (pd.DataFrame): The input DataFrame to which empty columns will be added.
            positions (list): A list of indices after which to insert empty columns. Must verify 0 <= loc <= len(columns).
        Returns:
            pd.DataFrame: The modified DataFrame with empty columns added.
        Raises:
            Exception: If there is an error inserting the empty columns at the specified positions.
        """
        # Sort positions in descending order to avoid index shifting issues
        for i, pos in enumerate(positions):
            try:
                # Insert empty column with 10 spaces, each column must be unique
                df.insert(pos + i, ' ' * (i + 1), " " * 10)  
            except Exception as e:
                print(f"Error inserting at position {pos}: {e}")
        df = df.fillna('')  # Fill NaN values with empty strings
        return df
    
    def _style_dataframe(self, df: pd.DataFrame, hide_index: bool = True, 
                        hide_columns: bool = True, level: int = 0, 
                        caption: str = None, header_alignment='left') -> Styler:
        """
        Styles the given DataFrame with options to hide index and columns.
        Args:
            df (pd.DataFrame): The input DataFrame to be styled.
            hide_index (bool, optional): Whether to hide the index. Defaults to True.
            hide_columns (bool, optional): Whether to hide the specified level of columns. Defaults to True.
            level (int, optional): The level of the columns to hide if hide_columns is True. Defaults to 0.
            caption (str, optional): The caption for the styled DataFrame. Defaults to "Datetime Summary Statistics".
        Returns:
            pd.io.formats.style.Styler: The styled DataFrame.
        """
        styled_df = (df.style.set_caption(caption)
                    .set_table_styles(
                        [
                            {
                                "selector": "caption",
                                "props": [
                                    ("font-size", "18px"),
                                    ("text-align", "left"),
                                    ("font-weight", "bold"),
                                ],
                            },
                            {
                                "selector": "th",
                                "props": [("text-align", header_alignment)]  # Align header text to the left
                            }
                        ]
                    )
                    .set_properties(**{"text-align": "left"}))
        if hide_index:
            styled_df = styled_df.hide(axis="index")
        if hide_columns:
            styled_df = styled_df.hide(axis="columns", level=level)
        return styled_df    
        
    def _is_categorical_column(self, col: str) -> bool:
        """
        Determines if a column should be treated as categorical based on its properties.
        
        Args:
            col: Name of the column to check
            
        Returns:
            bool: True if column is considered categorical, False otherwise
            
        Logic:
            - Column is categorical if:
            1. Ratio of unique values to total values < 50% (low cardinality), OR
            2. Column has pandas Categorical dtype
        """
        if col not in self.df.columns:
            return False
            
        series = self.df[col]
        return (not pd.api.types.is_numeric_dtype(series) and 
                (series.nunique() / len(series) < 0.5) or 
                isinstance(series.dtype, pd.CategoricalDtype))

    def _is_text_column(self, col: str) -> bool:
        """
        Determines if a column contains text data (non-categorical strings).
        
        Args:
            col: Name of the column to check
            
        Returns:
            bool: True if column contains text data, False otherwise
            
        Logic:
            - Column is text if:
            1. Has dtype 'object', AND
            2. Does not meet categorical criteria
        """
        if col not in self.df.columns:
            return False
            
        return (pd.api.types.is_string_dtype(self.df[col]) and 
                not self._is_categorical_column(col))

    def _is_int_column(self, col: str) -> bool:
        """
        Determines if a column contains integer values (including nullable integers).
        
        Args:
            col: Name of the column to check
            
        Returns:
            bool: True if column contains only integers, False otherwise
            
        Logic:
            - Column is integer if:
            1. Is numeric dtype, AND
            2. All non-null values are whole numbers
        """
        if col not in self.df.columns:
            return False
        
        if not pd.api.types.is_numeric_dtype(self.df[col]):
            return False
            
        col_clean = self.df[col].dropna()
        if col_clean.empty:
            return False
            
        # Handle nullable integer types (Int8, Int16, etc.)
        if pd.api.types.is_integer_dtype(self.df[col]):
            return True
            
        # Check if all values are whole numbers
        return (col_clean % 1 == 0).all()

    def _is_datetime_column(self, col: str) -> bool:
        """
        Determines if a column contains datetime values.
        
        Args:
            col: Name of the column to check
            
        Returns:
            bool: True if column contains datetime values, False otherwise
            
        Note:
            - Recognizes both datetime64[ns] and timezone-aware datetimes
        """
        if col not in self.df.columns:
            return False
            
        return pd.api.types.is_datetime64_any_dtype(self.df[col])
    
    def _is_float_column(self, col: str) -> bool:
        """
        Determines if a column contains floating-point values (excluding integers).
        
        Args:
            col: Name of the column to check
            
        Returns:
            bool: True if column contains floating-point numbers, False otherwise
            
        Logic:
            - Column is float if:
            1. Is numeric dtype, AND
            2. Contains decimal numbers (not all whole numbers), AND
            3. Is not a datetime column
        """
        if col not in self.df.columns:
            return False
        
        # First check if it's numeric and not datetime
        if not pd.api.types.is_numeric_dtype(self.df[col]) or self._is_datetime_column(col):
            return False
            
        # Exclude integer columns
        if self._is_int_column(col):
            return False
            
        # Handle empty columns
        col_clean = self.df[col].dropna()
        if col_clean.empty:
            return False
            
        # Check if contains decimal numbers
        return not (col_clean % 1 == 0).all()    
    
    def _get_column_type(self, col: str) -> str:
        """
        Determines the data type of a specified column in the DataFrame.
        Parameters:
        - col (str): The name of the column for which to determine the data type.
        Returns:
        - str: A string representing the type of the column. Possible values are:
            - "Categorical" for categorical columns
            - "Text" for string (text) columns
            - "Integer" for integer columns
            - "Float" for float (decimal) columns
            - "Datetime" for datetime columns
            - "Unknown" if the column type is not recognized.
        """
        if self._is_categorical_column(col):
            return "Categorical"
        elif self._is_text_column(col):
            return "Text"
        elif self._is_int_column(col):
            return "Integer"
        elif self._is_float_column(col):
            return "Float"
        elif self._is_datetime_column(col):
            return "Datetime"
        else:
            return "Unknown"    
      