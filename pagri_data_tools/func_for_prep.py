    def make_distribution_for_numeric(self, column: pd.Series) -> pd.DataFrame:
        """
        Analyzes data distribution characteristics including:
        - Normality tests
        - Outlier detection
        - Clustering tendency
        """
        clean_col = column.dropna()
        
        # Normality tests
        shapiro_test = stats.shapiro(clean_col) if len(clean_col) < 5000 else (np.nan, np.nan)
        norm_stats = {
            "Is Normal (Shapiro)": "Yes" if shapiro_test[1] > 0.05 else "No",
            "Shapiro p-value": f"{shapiro_test[1]:.4f}",
            "Skewness Direction": "Right" if clean_col.skew() > 0 else "Left",
            "Kurtosis Type": "Leptokurtic" if clean_col.kurtosis() > 0 else "Platykurtic"
        }
        
        # Outlier detection (3 methods)
        q1, q3 = clean_col.quantile(0.25), clean_col.quantile(0.75)
        iqr = q3 - q1
        classic_outliers = ((clean_col < (q1 - 1.5*iqr)) | (clean_col > (q3 + 1.5*iqr))).sum()
        
        z_scores = np.abs(stats.zscore(clean_col))
        z_outliers = (z_scores > 3).sum()
        
        isolation_forest = IsolationForest(contamination='auto')
        iso_outliers = isolation_forest.fit_predict(clean_col.values.reshape(-1,1))
        iso_outliers = (iso_outliers == -1).sum()
        
        outlier_stats = {
            "IQR Outliers": self._format_count_with_percentage(classic_outliers, len(clean_col)),
            "Z-Score Outliers": self._format_count_with_percentage(z_outliers, len(clean_col)),
            "Isolation Forest Outliers": self._format_count_with_percentage(iso_outliers, len(clean_col))
        }
        
        # Combine all stats
        combined_stats = {**norm_stats, **outlier_stats}
        return pd.DataFrame(combined_stats.items(), columns=["Metric", "Value"])



def make_correlations_for_numeric(self, column: pd.Series, target_col: str = None) -> pd.DataFrame:
    """
    Analyzes correlations with other numeric columns
    """
    numeric_cols = self.df.select_dtypes(include=np.number).columns.tolist()
    
    if target_col:
        if target_col not in numeric_cols:
            return pd.DataFrame({"Error": ["Target column is not numeric"]})
        corr_value = self.df[column.name].corr(self.df[target_col])
        return pd.DataFrame({
            "Target Column": [target_col],
            "Pearson r": [f"{corr_value:.3f}"],
            "Relationship Strength": ["Strong" if abs(corr_value) > 0.7 else 
                                    "Moderate" if abs(corr_value) > 0.3 else 
                                    "Weak"]
        })
    
    corrs = []
    for col in numeric_cols:
        if col != column.name:
            corr = self.df[column.name].corr(self.df[col])
            if abs(corr) > 0.3:  # Only show significant correlations
                corrs.append({
                    "Column": col,
                    "Correlation": f"{corr:.3f}",
                    "Type": "Positive" if corr > 0 else "Negative"
                })
    
    if not corrs:
        return pd.DataFrame({"Message": ["No significant correlations (>0.3) found"]})
    
    return pd.DataFrame(corrs).sort_values("Correlation", key=abs, ascending=False)


