from anomalies.anomaly import Anomaly
import pandas as pd
from IPython.display import display
from typing import Union, Optional

class SmartDataFrame:
    def __init__(self, df: pd.DataFrame):
        self._df = df.copy()
        # self._info = InfoAnalyzer(self)
        self._anomalies = Anomaly(self)
        # self._preprocessing = PreprocessingEngine(self)
        # self._visualization = VisualizationEngine(self)

    # @property
    # def info(self):
    #     return self._info

    # @property
    # def preprocessing(self):
    #     return self._preprocessing

    # @property
    # def viz(self):
    #     return self._visualization
    def __getattr__(self, attr):
        """Delegate unknown attributes to the underlying DataFrame."""
        if hasattr(self._df, attr):
            return getattr(self._df, attr)
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{attr}'")

    @property
    def anomalies(self):
        return self._anomalies
