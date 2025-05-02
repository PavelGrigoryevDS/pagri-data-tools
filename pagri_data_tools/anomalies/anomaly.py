import pandas as pd
import numpy as np
import plotly.express as px
from ipywidgets import widgets, Layout
from IPython.display import display, HTML
from IPython.display import display_html
from tqdm.auto import tqdm
import itertools
from pymystem3 import Mystem
import io
import base64 
import re
from typing import Union, List, Dict, Tuple, Any, Optional
from enum import Enum, auto
import io
import base64
from pandas.io.formats.style import Styler
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from IPython.display import display
from scipy import stats

class Anomaly:
    def __init__(self, parent_df):
        self._parent = parent_df
    
    def check_duplicated_rows(self, return_df: bool = True) -> Union[str, pd.DataFrame]:
        """
        Check the DataFrame for completely duplicated rows.
        
        Args:
            return_df: If True, returns a DataFrame with duplicate rows when found.
                      If False, returns a summary string.
        
        Returns:
            Union[str, pd.DataFrame]: Either a message about duplicates or a DataFrame
                                     containing the duplicate rows with their counts.
        """
        duplicate_count = self._parent.duplicated().sum()
        total_rows = self._parent.shape[0]
        
        if duplicate_count == 0:
            return "No duplicate rows found in the DataFrame."
            
        summary = f"Found {duplicate_count} duplicated rows ({(duplicate_count / total_rows):.1%} of total)"
        
        if not return_df:
            return summary
            
        print(summary)
        
        # Normalize string data for better duplicate detection
        normalized_df = self._parent.apply(
            lambda x: (
                x.astype(str)
                 .str.lower()
                 .str.strip()
                 .str.replace(r"\s+", " ", regex=True)
                if x.dtype == "object"
                else x
            )
        )
        
        duplicates = (
            normalized_df.value_counts(dropna=False)
            .to_frame(name='count')
            .sort_values('count', ascending=False)
            .query("count > 1")
        )
        
        return duplicates
    
    def find_columns_with_duplicates(self, keep: str = 'first') -> Optional[pd.Series]:
        """
        Analyze each column for duplicate values and return results.
        
        Args:
            keep: Controls which duplicates to mark:
                  - 'first' : Mark duplicates as True except for the first occurrence.
                  - 'last' : Mark duplicates as True except for the last occurrence.
                  - False : Mark all duplicates as True.
        
        Returns:
            Optional[pd.Series]: A Series where each entry contains a DataFrame with 
                                duplicates for that column, or None if no duplicates found.
        """
        column_duplicates = pd.Series(dtype=object)
        duplicate_counts = pd.Series(dtype=int)
        total_rows = self._parent.shape[0]
        
        for col in self._parent.columns:
            is_duplicated = self._parent[col].duplicated(keep=keep)
            if is_duplicated.any():
                column_duplicates[col] = self._parent.loc[is_duplicated]
                duplicate_counts[col] = is_duplicated.sum()
        
        if duplicate_counts.empty:
            print('No duplicated values found in any columns.')
            return None
            
        # Display formatted summary
        display(
            duplicate_counts.apply(lambda x: f"{x} ({(x / total_rows):.2%})")
            .to_frame('Duplicate Count')
            .style.set_caption("Column Duplicates Summary")
            .set_table_styles([{
                "selector": "caption",
                "props": [
                    ("font-size", "18px"),
                    ("text-align", "left"),
                    ("font-weight", "bold"),
                ]
            }])
        )
        
        return column_duplicates