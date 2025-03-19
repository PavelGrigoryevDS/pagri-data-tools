# import importlib
# importlib.reload(pgdt)
import pandas as pd
import numpy as np
import plotly.express as px
import statsmodels.stats.api as stm
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from IPython.display import display
from termcolor import colored
import scipy.stats as stats
import pingouin as pg
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from tqdm.auto import tqdm
import scikit_posthocs

def plot_feature_importances_classifier(df: pd.DataFrame, target: str, titles_for_axis: dict = None, title=None):
    """
    Plot the feature importances of a random forest classifier using Plotly Express.

    Parameters:
    df (pandas.DataFrame): The input DataFrame containing the features and target variable.
    target (str): The name of the target variable column in the DataFrame.
    titles_for_axis (dict):  A dictionary containing titles for the axes.
    title (str): Title.

    Returns:
    fig (plotly.graph_objs.Figure): The feature importance plot.

    Notes:
    This function trains a random forest classifier on the input DataFrame, extracts the feature importances,
    and plots them using Plotly Express. 
    Examples:
        titles_for_axis = dict(
            debt = 'долга'
            , children = 'Кол-во детей'
            , age = 'Возраст'
            , total_income = 'Доход')
)
    """

    # Select numeric columns and the target variable
    num_columns = [
        col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
    df_tmp = df[num_columns + [target]].dropna()
    df_features = df_tmp[num_columns]
    target_str = target
    target = df_tmp[target]
    # Get the feature names
    features = df_features.columns
    if titles_for_axis:
        features = [titles_for_axis[feature] for feature in features]
        target_str = titles_for_axis[target_str]
    # Normalize the data using Standard Scaler
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df_features)

    # Train a random forest classifier
    clf = RandomForestClassifier(n_estimators=100, random_state=0)
    clf.fit(df_scaled, target)

    # Get the feature importances
    importances = clf.feature_importances_
    feature_importances = pd.DataFrame(
        {'Feature': features, 'Importance': importances})

    # Sort the feature importances in descending order
    feature_importances = feature_importances.sort_values(
        'Importance', ascending=False)

    # Create the bar chart
    if title:
        title=title # f'График важности признаков для предсказания {target}'
    else: 
        title='График важности признаков'        
    # Create the bar chart
    fig = px.bar(feature_importances, x='Importance', y='Feature', title=title)
    fig.update_layout(
        yaxis=dict(categoryorder='total ascending', title=dict(
            font=dict(size=18, color="rgba(0, 0, 0, 0.5)"), text='Название признака')),
        xaxis=dict(title=dict(font=dict(size=18, color="rgba(0, 0, 0, 0.5)"),
                   text='Оценка важности'), showgrid=True, gridwidth=1, gridcolor="rgba(0, 0, 0, 0.1)"),
        width=700,  # Set the width of the graph
        height=500,  # Set the height of the graph
        template='simple_white',  # Set the template to simple_white
        title_font=dict(size=18, color="rgba(0, 0, 0, 0.7)"),     
        font=dict(size=14, family="Segoe UI", color="rgba(0, 0, 0, 0.7)"),
        xaxis_title_font=dict(size=16, color="rgba(0, 0, 0, 0.7)"),
        yaxis_title_font=dict(size=16, color="rgba(0, 0, 0, 0.7)"),
        xaxis_tickfont=dict(size=14, color="rgba(0, 0, 0, 0.7)"),
        yaxis_tickfont=dict(size=14, color="rgba(0, 0, 0, 0.7)"),
        xaxis_linecolor="rgba(0, 0, 0, 0.4)",
        yaxis_linecolor="rgba(0, 0, 0, 0.4)", 
        xaxis_tickcolor="rgba(0, 0, 0, 0.4)",
        yaxis_tickcolor="rgba(0, 0, 0, 0.4)",  
        legend_title_font_color='rgba(0, 0, 0, 0.7)',
        legend_font_color='rgba(0, 0, 0, 0.7)',
        # xaxis_linewidth=2,
        # yaxis_linewidth=2
        margin=dict(l=50, r=50, b=50, t=70),
        hoverlabel=dict(bgcolor="white"),
        # xaxis=dict(
        #     showgrid=True
        #     , gridwidth=1
        #     , gridcolor="rgba(0, 0, 0, 0.1)"
        # ),
        # yaxis=dict(
        #     showgrid=True
        #     , gridwidth=1
        #     , gridcolor="rgba(0, 0, 0, 0.07)"
        # )        
    )
    # Set the bar color to mediumpurple
    fig.update_traces(marker_color='rgba(128, 60, 170, 0.9)')

    return fig


def plot_feature_importances_regression(df: pd.DataFrame, target: str, titles_for_axis: dict = None, title=None):
    """
    Plot the feature importances of a random forest regressor using Plotly Express.

    Parameters:
    df (pandas.DataFrame): The input DataFrame containing the features and target variable.
    target (str): The name of the target variable column in the DataFrame.
    titles_for_axis (dict):  A dictionary containing titles for the axes.
    title (str): Title.
    Returns:
    fig (plotly.graph_objs.Figure): The feature importance plot.

    Notes:
    This function trains a random forest classifier on the input DataFrame, extracts the feature importances,
    and plots them using Plotly Express. 
    Examples:
        titles_for_axis = dict(
            debt = 'долга'
            , children = 'Кол-во детей'
            , age = 'Возраст'
            , total_income = 'Доход')
        title = 'График важности признаков для предсказания цены'
    Notes:
    This function trains a random forest regressor on the input DataFrame, extracts the feature importances,
    and plots them using Plotly Express.
    """
    # Select numeric columns and the target variable
    num_columns = [
        col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
    df_tmp = df[num_columns].dropna()
    df_features = df_tmp[num_columns].drop(columns=target)
    target_series = df_tmp[target]
    # Get the feature names
    feature_names = df_features.columns
    if titles_for_axis:
        feature_names = [titles_for_axis[feature] for feature in feature_names]
        target = titles_for_axis[target]
    # Normalize the data using Standard Scaler
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df_features)

    # Train a random forest regressor
    clf = RandomForestRegressor(n_estimators=100, random_state=0)
    clf.fit(scaled_data, target_series)

    # Get the feature importances
    importances = clf.feature_importances_
    feature_importances = pd.DataFrame(
        {'Feature': feature_names, 'Importance': importances})

    # Sort the feature importances in descending order
    feature_importances = feature_importances.sort_values(
        'Importance', ascending=False)
    if title:
        title=title # f'График важности признаков для предсказания {target}'
    else: 
        title='График важности признаков'
    # Create the bar chart
    fig = px.bar(feature_importances, x='Importance', y='Feature', title=title)
    fig.update_layout(
        yaxis=dict(categoryorder='total ascending', title=dict(
            font=dict(size=18, color="rgba(0, 0, 0, 0.5)"), text='Название признака')),
        xaxis=dict(title=dict(font=dict(size=18, color="rgba(0, 0, 0, 0.5)"),
                   text='Оценка важности'), showgrid=True, gridwidth=1, gridcolor="rgba(0, 0, 0, 0.1)"),
        width=700,  # Set the width of the graph
        height=500,  # Set the height of the graph
        template='simple_white',  # Set the template to simple_white
        title_font=dict(size=18, color="rgba(0, 0, 0, 0.7)"),     
        font=dict(size=14, family="Segoe UI", color="rgba(0, 0, 0, 0.7)"),
        xaxis_title_font=dict(size=16, color="rgba(0, 0, 0, 0.7)"),
        yaxis_title_font=dict(size=16, color="rgba(0, 0, 0, 0.7)"),
        xaxis_tickfont=dict(size=14, color="rgba(0, 0, 0, 0.7)"),
        yaxis_tickfont=dict(size=14, color="rgba(0, 0, 0, 0.7)"),
        xaxis_linecolor="rgba(0, 0, 0, 0.4)",
        yaxis_linecolor="rgba(0, 0, 0, 0.4)", 
        xaxis_tickcolor="rgba(0, 0, 0, 0.4)",
        yaxis_tickcolor="rgba(0, 0, 0, 0.4)",  
        legend_title_font_color='rgba(0, 0, 0, 0.7)',
        legend_font_color='rgba(0, 0, 0, 0.7)',
        # xaxis_linewidth=2,
        # yaxis_linewidth=2
        margin=dict(l=50, r=50, b=50, t=70),
        hoverlabel=dict(bgcolor="white"),
        # xaxis=dict(
        #     showgrid=True
        #     , gridwidth=1
        #     , gridcolor="rgba(0, 0, 0, 0.1)"
        # ),
        # yaxis=dict(
        #     showgrid=True
        #     , gridwidth=1
        #     , gridcolor="rgba(0, 0, 0, 0.07)"
        # )        
    )
    # Set the bar color to mediumpurple
    hovertemplate = 'Оценка важности = %{x}<br>Название признака = %{y}<extra></extra>'
    fig.update_traces(marker_color='rgba(128, 60, 170, 0.9)', hovertemplate=hovertemplate)

    return fig


def linear_regression_with_vif(df: pd.DataFrame, target_column: str) -> None:
    """
    Perform linear regression with variance inflation factor (VIF) analysis.

    Parameters:
    df (pd.DataFrame): Input DataFrame containing the data.
    target_column (str): Name of the target column.

    Returns:
    None

    Description:
    This function performs linear regression on the input DataFrame with the specified target column.
    It first selects only the numeric columns, drops any rows with missing values, and then splits the data into features (X) and target (y).
    A constant term is added to the independent variables (X) using `sm.add_constant(X)`.
    The function then fits an ordinary least squares (OLS) model with heteroscedasticity-consistent standard errors (HC1).
    The variance inflation factor (VIF) is calculated for each feature, and the results are displayed along with the model summary.
    """
    # Select numeric columns and drop rows with missing values
    num_columns = [
        col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
    df_tmp = df[num_columns].dropna()
    if target_column not in df_tmp.columns:
        raise ValueError(
            f"Target column '{target_column}' not found in the DataFrame.")
    # Split data into features (X) and target (y)
    X = df_tmp.drop(columns=target_column)
    y = df_tmp[target_column]

    # Add a constant term to the independent variables
    # This operation is necessary to allow the OLS model to capture the effect of the constant term on the dependent variable, 
    # as assuming it is zero when all independent variables are zero may not always be accurate.
    X = sm.add_constant(X)

    # Fit OLS model with HC1 standard errors
    model = sm.OLS(y, X).fit(cov_type='HC1')

    # Calculate VIF for each feature
    vif = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

    # Create a DataFrame with coefficients and VIF
    res = pd.DataFrame({'Coef': model.params, 'VIF': vif})

    # Display results
    display(res.iloc[1:])  # exclude the constant term
    display(model.summary())


def calculate_cohens_d(sample1: pd.Series, sample2: pd.Series, equal_var=False) -> float:
    """
    Calculate Cohen's d from two independent samples.  
    Cohen's d is a measure of effect size used to quantify the standardized difference between the means of two groups.

    Parameters:
    sample1 (pd.Series): First sample
    sample2 (pd.Series): Second sample
    equal_var (bool): Whether to assume equal variances between the two samples. If `True`, the pooled standard deviation is used.   
    If `False`, the standard error is calculated using the separate variances of each sample. Defaults to `False`.

    Returns:
    float: Cohen's d
    """
    # Check if inputs are pd.Series
    if not isinstance(sample1, pd.Series) or not isinstance(sample2, pd.Series):
        raise ValueError("Both inputs must be pd.Series")

    # Check if samples are not empty
    if sample1.empty or sample2.empty:
        raise ValueError("Both samples must be non-empty")

    # Calculate means and variances
    mean1, var1 = sample1.mean(), sample1.var(ddof=1)
    mean2, var2 = sample2.mean(), sample2.var(ddof=1)

    # Calculate sample sizes
    n1 = len(sample1)
    n2 = len(sample2)

    if equal_var:
        # Calculate pooled standard deviation
        pooled_std = np.sqrt(
            ((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
        cohens_d = (mean1 - mean2) / pooled_std
    else:
        varn1 = var1 / n1
        varn2 = var2 / n2
        standard_error = np.sqrt(varn1 + varn2)
        cohens_d = (mean1 - mean2) / standard_error

    return cohens_d

# Хи-квадрат Пирсона
# Не чувствителен к гетероскедастичности (неравномерной дисперсии) данных.

def chisquare(sample: pd.Series, alpha: float = 0.05, return_results: bool = False) -> None:
    """
    Calculate a one-way chi-square test.
    Perform Pearson's chi-squared test for the null hypothesis that the categorical data has the given frequencies.

    Parameters:
    - sample(pd.Series): First categorical variable
    - alpha (float, optional): Significance level (default: 0.05)
    - return_results (bool, optional): Return (chi2, p_value, dof, expected) instead of printing (default=False).

    Returns:
    - If return_results is False: 
    None
    - If return_results is True
        - chi2 : (float) 
            The test statistic.
        - p : (float) 
            The p-value of the test
        - dof : (int)
            Degrees of freedom
        - expected : (ndarray, same shape as observed)
            The expected frequencies, based on the marginal sums of the table.
    """
    if alpha < 0 or alpha > 1:
        raise Exception(f"alpha must be between 0 and 1, but got {alpha}")
    if not isinstance(sample, pd.Series):
        raise ValueError("Input samples must be pd.Series")
    if not len(sample) > 0:
        raise ValueError("All samples must have at least one value")
    if alpha < 0 or alpha > 1:
        raise Exception(f"alpha must be between 0 and 1, but got {alpha}")
    if sample.isna().sum():
        raise ValueError(
            f'column1 and column2 must not have missing values.\ncolumn1 have {sample.isna().sum()} missing values')

    chi2, p_value = stats.chisquare(sample)

    if not return_results:
        print('Хи-квадрат Пирсона')
        print('alpha = ', alpha)
        print('p-value = ', round(p_value, 3))
        if p_value < alpha:
            print(colored(
                "Отклоняем нулевую гипотезу, поскольку p-value меньше уровня значимости", 'red'))
        else:
            print(colored(
                "Нет оснований отвергнуть нулевую гипотезу, поскольку p-value больше или равно уровню значимости", 'green'))
    else:
        return chi2, p_value

def chi2_pearson(sample1: pd.Series, sample2: pd.Series, alpha: float = 0.05, return_results: bool = False) -> None:
    """
    Perform Pearson's chi-squared test for independence between two categorical variables.

    Parameters:
    - column1 (pd.Series): First categorical variable
    - column2 (pd.Series): Second categorical variable
    - alpha (float, optional): Significance level (default: 0.05)
    - return_results (bool, optional): Return (chi2, p_value, dof, expected) instead of printing (default=False).

    Returns:
    - If return_results is False: 
    None
    - If return_results is True
        - chi2 : (float) 
            The test statistic.
        - p : (float) 
            The p-value of the test
        - dof : (int)
            Degrees of freedom
        - expected : (ndarray, same shape as observed)
            The expected frequencies, based on the marginal sums of the table.
    """
    if alpha < 0 or alpha > 1:
        raise Exception(f"alpha must be between 0 and 1, but got {alpha}")
    if not all(isinstance(sample, pd.Series) for sample in [sample1, sample2]):
        raise ValueError("Input samples must be pd.Series")
    if not all(len(sample) > 0 for sample in [sample1, sample2]):
        raise ValueError("All samples must have at least one value")
    if alpha < 0 or alpha > 1:
        raise Exception(f"alpha must be between 0 and 1, but got {alpha}")
    if sample1.isna().sum() or sample2.isna().sum():
        raise ValueError(
            f'column1 and column2 must not have missing values.\ncolumn1 have {sample1.isna().sum()} missing values\ncolumn2 have {sample2.isna().sum()} missing values')
    crosstab_for_chi2_pearson = pd.crosstab(sample1, sample2)
    chi2, p_value, dof, expected = stats.chi2_contingency(
        crosstab_for_chi2_pearson)

    if not return_results:
        print('Хи-квадрат Пирсона')
        print('alpha = ', alpha)
        print('p-value = ', round(p_value, 3))
        if p_value < alpha:
            print(colored(
                "Отклоняем нулевую гипотезу, поскольку p-value меньше уровня значимости", 'red'))
        else:
            print(colored(
                "Нет оснований отвергнуть нулевую гипотезу, поскольку p-value больше или равно уровню значимости", 'green'))
    else:
        return chi2, p_value, dof, expected


def ttest_ind_df(df: pd.DataFrame, alpha: float = 0.05, equal_var=False, alternative: str = 'two-sided', first_for_alternative: str = None, return_results: bool = False) -> None:
    """
    Perform t-test for independent samples.

    Parameters:
    - df (pd.DataFrame): DataFrame containing two columns,
        where the first column contains sample labels (e.g., "male" and "female") and
        the second column contains corresponding values
    - alpha (float, optional): Significance level (default: 0.05)
    - equal_var (bool, optional): If True (default), perform a standard independent 2 sample test that assumes equal population variances.  
        If False, perform Welch's t-test, which does not assume equal population variance.
    - alternative (str, optional): Alternative hypothesis ('two-sided', '2s', 'larger', 'l', 'smaller', 's') (default: 'two-sided')
    - first_for_alternative (str, optional): Lable name in first column of df for define first sample for scipy.stats.ttest_ind
    - return_results (bool, optional): Return (statistic, p_value, beta, cohens_d) instead of printing (default=False).

    Returns:
    - If return_results is False: None
    - If return_results is True
        - statistic : (float or array)
            The calculated t-statistic.
        - pvalue : (float or array)
            The two-tailed p-value.
        - beta : (float)
            The probability of Type II error (beta).
        - cohens_d : (float)
            The effect size (Cohen's d).
    """
    if alternative not in ["two-sided", "2s", "larger", "l", "smaller", "s"]:
        raise Exception(
            f"alternative must be 'two-sided', '2s', 'larger', 'l', 'smaller', 's', but got {alternative}")
    alternative_map = {
        "two-sided": "two-sided",
        "2s": "two-sided",
        "larger": "greater",
        "l": "greater",
        "smaller": "less",
        "s": "less"
    }
    alternative = alternative_map[alternative]
    if alpha < 0 or alpha > 1:
        raise Exception(f"alpha must be between 0 and 1, but got {alpha}")
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input df must be pd.DataFrame")
    if df.shape[1] != 2:
        raise ValueError("Input DataFrame must have exactly two columns")

    sample_column = df.iloc[:, 0]
    value_column = df.iloc[:, 1]
    if not pd.api.types.is_numeric_dtype(value_column):
        raise ValueError("Value column must contain numeric values")
    if sample_column.isna().sum() or value_column.isna().sum():
        raise ValueError(
            f'sample_column and value_column must not have missing values.\nsample_column have {sample_column.isna().sum()} missing values\nvalue_column have {value_column.isna().sum()} missing values')
    unique_samples = sample_column.unique()
    if len(unique_samples) != 2:
        raise ValueError(
            "Sample column must contain exactly two unique labels")
    if first_for_alternative:
        if first_for_alternative not in unique_samples:
            raise ValueError('first_for_alternative must be a lable in first column of df')
        if first_for_alternative != unique_samples[0]:
            unique_samples = unique_samples[::-1]
    sample1 = value_column[sample_column == unique_samples[0]]
    sample2 = value_column[sample_column == unique_samples[1]]
    warning_issued = False
    for sample in [sample1, sample2]:
        if len(sample) < 2:
            raise ValueError("Each sample must have at least two elements")
        elif len(sample) < 30:
            warning_issued = True
    if warning_issued:
        print(colored(
            "Warning: Sample size is less than 30 for one or more samples. Results may be unreliable.", 'red'))

    nobs1 = len(sample1)
    nobs2 = len(sample2)
    # Calculate Cohen's d
    cohens_d = calculate_cohens_d(sample1, sample2, equal_var=equal_var)
    # Calculate the power of the test
    power = sm.stats.TTestIndPower().solve_power(
        effect_size=cohens_d, nobs1=nobs1, ratio=nobs2/nobs1, alpha=alpha)
    # Calculate the type II error rate (β)
    beta = 1 - power

    res = stats.ttest_ind(
        sample1, sample2, equal_var=equal_var, alternative=alternative)
    statistic = res.statistic
    p_value = res.pvalue
    ci_low = float(res.confidence_interval().low.round(2))
    ci_high = float(res.confidence_interval().high.round(2))

    if not return_results:
        print('T-критерий')
        print('p-value = ', round(p_value, 3))
        print('alpha = ', alpha)
        print('beta = ', beta)
        print(f'ci = ({ci_low}, {ci_high})')

        if p_value < alpha:
            print(colored(
                "Отклоняем нулевую гипотезу, поскольку p-value меньше уровня значимости", 'red'))
        else:
            print(colored(
                "Нет оснований отвергнуть нулевую гипотезу, поскольку p-value больше или равно уровню значимости", 'green'))
    else:
        return statistic, p_value, beta, cohens_d


def ttest_ind(sample1: pd.Series, sample2: pd.Series, alpha: float = 0.05, equal_var=False, alternative: str = 'two-sided', return_results: bool = False) -> None:
    """
    Perform t-test for independent samples.

    Parameters:
    - sample1 (pd.Series): First sample values
    - sample2 (pd.Series): Second sample values
    - alpha (float, optional): Significance level (default: 0.05)
    - equal_var (bool, optional): If True (default), perform a standard independent 2 sample test that assumes equal population variances.  
        If False, perform Welch's t-test, which does not assume equal population variance.
    - alternative (str, optional): Alternative hypothesis ('two-sided', '2s', 'larger', 'l', 'smaller', 's') (default: 'two-sided')
    - return_results (bool, optional): Return (statistic, p_value, beta, cohens_d) instead of printing (default=False).

    Returns:
    - If return_results is False: None
    - If return_results is True
        - statistic : (float or array)
            The calculated t-statistic.
        - pvalue : (float or array)
            The two-tailed p-value.
        - beta : (float)
            The probability of Type II error (beta).
        - cohens_d : (float)
            The effect size (Cohen's d).
    """
    if alternative not in ["two-sided", "2s", "larger", "l", "smaller", "s"]:
        raise Exception(
            f"alternative must be 'two-sided', '2s', 'larger', 'l', 'smaller', 's', but got {alternative}")
    alternative_map = {
        "two-sided": "two-sided",
        "2s": "two-sided",
        "larger": "greater",
        "l": "greater",
        "smaller": "less",
        "s": "less"
    }
    alternative = alternative_map[alternative]
    if alpha < 0 or alpha > 1:
        raise Exception(f"alpha must be between 0 and 1, but got {alpha}")
    if not all(isinstance(sample, pd.Series) for sample in [sample1, sample2]):
        raise ValueError("Input samples must be pd.Series")
    if not all(len(sample) > 0 for sample in [sample1, sample2]):
        raise ValueError("All samples must have at least one value")
    if not pd.api.types.is_numeric_dtype(sample1) or not pd.api.types.is_numeric_dtype(sample2):
        raise ValueError("sample1 and sample2 must contain numeric values")
    if sample1.isna().sum() or sample2.isna().sum():
        raise ValueError(
            f'sample1 and sample2 must not have missing values.\nsample1 have {sample1.isna().sum()} missing values\nsample2 have {sample2.isna().sum()} missing values')
    warning_issued = False
    for sample in [sample1, sample2]:
        if len(sample) < 2:
            raise ValueError("Each sample must have at least two elements")
        elif len(sample) < 30:
            warning_issued = True
    if warning_issued:
        print(colored(
            "Warning: Sample size is less than 30 for one or more samples. Results may be unreliable.", 'red'))
    nobs1 = len(sample1)
    nobs2 = len(sample2)
    # Calculate Cohen's d
    cohens_d = calculate_cohens_d(sample1, sample2, equal_var=equal_var)
    # Calculate the power of the test
    power = sm.stats.TTestIndPower().solve_power(
        effect_size=cohens_d, nobs1=nobs1, ratio=nobs2/nobs1, alpha=alpha)
    # Calculate the type II error rate (β)
    beta = 1 - power
    res = stats.ttest_ind(
        sample1, sample2, equal_var=equal_var, alternative=alternative)
    statistic = res.statistic
    p_value = res.pvalue
    ci_low = float(res.confidence_interval().low.round(2))
    ci_high = float(res.confidence_interval().high.round(2))
    if not return_results:
        print('T-критерий Уэлча')
        print('p-value = ', round(p_value, 3))
        print('alpha = ', alpha)
        print('beta = ', beta)
        print(f'ci = ({ci_low}, {ci_high})')
        if p_value < alpha:
            print(colored(
                "Отклоняем нулевую гипотезу, поскольку p-value меньше уровня значимости", 'red'))
        else:
            print(colored(
                "Нет оснований отвергнуть нулевую гипотезу, поскольку p-value больше или равно уровню значимости", 'green'))
    else:
        return statistic, p_value, beta, cohens_d


def mannwhitneyu_df(df: pd.DataFrame, alpha: float = 0.05, alternative: str = 'two-sided', first_for_alternative: str = None, return_results: bool = False) -> None:
    """
    Perform the Mann-Whitney U rank test on two independent samples.

    Parameters:
    - df (pd.DataFrame): DataFrame containing two columns,
        where the first column contains sample labels (e.g., "male" and "female") and
        the second column contains corresponding values
    - alpha (float, optional): Significance level (default: 0.05)
    - alternative (str, optional): Alternative hypothesis ('two-sided', '2s', 'larger', 'l', 'smaller', 's') (default: 'two-sided')
    - first_for_alternative (str, optional): Lable name in first column of df for define first sample for scipy.stats.ttest_ind
    - return_results (bool, optional): Return (statistic, p_value) instead of printing (default=False).

    Returns:
    - If return_results is False: None
    - If return_results is True
        - statistic : (float)
            The Mann-Whitney U statistic corresponding with sample x. See Notes for the test statistic corresponding with sample y.
        - pvalue : (float)
            The associated *p*-value for the chosen alternative.
    """
    if alternative not in ["two-sided", "2s", "larger", "l", "smaller", "s"]:
        raise Exception(
            f"alternative must be 'two-sided', '2s', 'larger', 'l', 'smaller', 's', but got {alternative}")
    alternative_map = {
        "two-sided": "two-sided",
        "2s": "two-sided",
        "larger": "greater",
        "l": "greater",
        "smaller": "less",
        "s": "less"
    }
    alternative = alternative_map[alternative]
    if alpha < 0 or alpha > 1:
        raise Exception(f"alpha must be between 0 and 1, but got {alpha}")
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input df must be pd.DataFrame")
    if df.shape[1] != 2:
        raise ValueError("Input DataFrame must have exactly two columns")
    sample_column = df.iloc[:, 0]
    value_column = df.iloc[:, 1]
    if not pd.api.types.is_numeric_dtype(value_column):
        raise ValueError("Value column must contain numeric values")
    if sample_column.isna().sum() or value_column.isna().sum():
        raise ValueError(
            f'sample_column and value_column must not have missing values.\nsample_column have {sample_column.isna().sum()} missing values\nvalue_column have {value_column.isna().sum()} missing values')
    unique_samples = sample_column.unique()
    if len(unique_samples) != 2:
        raise ValueError(
            "Sample column must contain exactly two unique labels")
    if first_for_alternative:
        if first_for_alternative not in unique_samples:
            raise ValueError('first_for_alternative must be a lable in first column of df')
        if first_for_alternative != unique_samples[0]:
            unique_samples = unique_samples[::-1]
    sample1_values = value_column[sample_column == unique_samples[0]]
    sample2_values = value_column[sample_column == unique_samples[1]]
    warning_issued = False
    for sample in [sample1_values, sample2_values]:
        if len(sample) < 2:
            raise ValueError("Each sample must have at least two elements")
        elif len(sample) < 30:
            warning_issued = True
    if warning_issued:
        print(colored(
            "Warning: Sample size is less than 30 for one or more samples. Results may be unreliable.", 'red'))
    statistic, p_value = stats.mannwhitneyu(
        sample1_values, sample2_values, alternative=alternative)

    if not return_results:
        print('U-критерий Манна-Уитни')
        print('alpha = ', alpha)
        print('p-value = ', round(p_value, 3))
        if p_value < alpha:
            print(colored(
                "Отклоняем нулевую гипотезу, поскольку p-value меньше уровня значимости", 'red'))
        else:
            print(colored(
                "Нет оснований отвергнуть нулевую гипотезу, поскольку p-value больше или равно уровню значимости", 'green'))
    else:
        return statistic, p_value


def mannwhitneyu(sample1: pd.Series, sample2: pd.Series, alpha: float = 0.05, alternative: str = 'two-sided', return_results: bool = False) -> None:
    """
    Perform the Mann-Whitney U rank test on two independent samples.

    Parameters:
    - sample1 (pd.Series): First sample values
    - sample2 (pd.Series): Second sample values
    - alpha (float, optional): Significance level (default: 0.05)
    - alternative (str, optional): Alternative hypothesis ('two-sided', '2s', 'larger', 'l', 'smaller', 's') (default: 'two-sided')
    - return_results (bool, optional): Return (statistic, p_value) instead of printing (default=False).

    Returns:
    - If return_results is False: None
    - If return_results is True
        - statistic : (float)
            The Mann-Whitney U statistic corresponding with sample x. See Notes for the test statistic corresponding with sample y.
        - pvalue : (float)
            The associated *p*-value for the chosen alternative.
    """
    if alternative not in ["two-sided", "2s", "larger", "l", "smaller", "s"]:
        raise Exception(
            f"alternative must be 'two-sided', '2s', 'larger', 'l', 'smaller', 's', but got {alternative}")
    alternative_map = {
        "two-sided": "two-sided",
        "2s": "two-sided",
        "larger": "greater",
        "l": "greater",
        "smaller": "less",
        "s": "less"
    }
    alternative = alternative_map[alternative]
    if alpha < 0 or alpha > 1:
        raise Exception(f"alpha must be between 0 and 1, but got {alpha}")
    if not all(isinstance(sample, pd.Series) for sample in [sample1, sample2]):
        raise ValueError("Input samples must be pd.Series")
    if not all(len(sample) > 0 for sample in [sample1, sample2]):
        raise ValueError("All samples must have at least one value")
    if not pd.api.types.is_numeric_dtype(sample1) or not pd.api.types.is_numeric_dtype(sample2):
        raise ValueError("sample1 and sample2 must contain numeric values")
    if sample1.isna().sum() or sample2.isna().sum():
        raise ValueError(
            f'sample1 and sample2 must not have missing values.\nsample1 have {sample1.isna().sum()} missing values\nsample2 have {sample2.isna().sum()} missing values')
    warning_issued = False
    for sample in [sample1, sample2]:
        if len(sample) < 2:
            raise ValueError("Each sample must have at least two elements")
        elif len(sample) < 30:
            warning_issued = True
    if warning_issued:
        print(colored(
            "Warning: Sample size is less than 30 for one or more samples. Results may be unreliable.", 'red'))
    statistic, p_value = stats.mannwhitneyu(
        sample1, sample2, alternative=alternative)

    if not return_results:
        print('U-критерий Манна-Уитни')
        print('alpha = ', alpha)
        print('p-value = ', round(p_value, 3))
        if p_value < alpha:
            print(colored(
                "Отклоняем нулевую гипотезу, поскольку p-value меньше уровня значимости", 'red'))
        else:
            print(colored(
                "Нет оснований отвергнуть нулевую гипотезу, поскольку p-value больше или равно уровню значимости", 'green'))
    else:
        return statistic, p_value


def proportion_ztest_1sample(count: int, n: int, p0: float, alpha: float = 0.05, alternative: str = 'two-sided', return_results: bool = False) -> None:
    """
    Perform a one-sample z-test for a proportion.

    Parameters:
    - count (int): Number of successes in the sample
    - n (int): Total number of observations in the sample
    - p0 (float): Known population proportion under the null hypothesis
    - alpha (float, optional): Significance level (default: 0.05)
    - alternative (str, optional): Alternative hypothesis ('two-sided', '2s', 'larger', 'l', 'smaller', 's') (default: 'two-sided')
    - return_results (bool, optional): Return (z_stat, p_value) instead of printing (default=False)

    Returns:
    - If return_results is False: None
    - If return_results is True
        - zstat : (float)
            test statistic for the z-test
        - p-value : (float)
            p-value for the z-test
    """
    if alternative not in ["two-sided", "2s", "larger", "l", "smaller", "s"]:
        raise Exception(
            f"alternative must be 'two-sided', '2s', 'larger', 'l', 'smaller', 's', but got {alternative}")
    alternative_map = {
        "two-sided": "two-sided",
        "2s": "two-sided",
        "larger": "larger",
        "l": "larger",
        "smaller": "smaller",
        "s": "smaller"
    }
    alternative = alternative_map[alternative]
    if alpha < 0 or alpha > 1:
        raise Exception(f"alpha must be between 0 and 1, but got {alpha}")
    if not all(isinstance(x, int) for x in [count, n]):
        raise ValueError("count and n must be integers")
    if not isinstance(p0, float) or p0 < 0 or p0 > 1:
        raise ValueError("p0 must be a float between 0 and 1")

    z_stat, p_value = stm.proportions_ztest(
        count=count, nobs=n, value=p0, alternative=alternative)

    if not return_results:
        print('Один выборочный Z-тест для доли')
        print('alpha = ', alpha)
        print('p-value = ', round(p_value, 3))
        if p_value < alpha:
            print(colored(
                "Отклоняем нулевую гипотезу, поскольку p-value меньше уровня значимости", 'red'))
        else:
            print(colored(
                "Нет оснований отвергнуть нулевую гипотезу, поскольку p-value больше или равно уровню значимости", 'green'))
    else:
        return z_stat, p_value


def proportions_ztest_2sample(count1: int, count2: int, n1: int, n2: int, alpha: float = 0.05, alternative: str = 'two-sided', return_results: bool = False) -> None:
    """
    Perform a z-test for proportions.

    Parameters:
    - count1 (int): Number of successes in the first sample
    - count2 (int): Number of successes in the second sample
    - n1 (int): Total number of observations in the first sample
    - n2 (int): Total number of observations in the second sample
    - alpha (float, optional): Significance level (default: 0.05)
    - alternative (str, optional): Alternative hypothesis ('two-sided', '2s', 'larger', 'l', 'smaller', 's') (default: 'two-sided')
    - return_results (bool, optional): Return (z_stat, p_value) instead of printing (default=False).

    Returns:
    - If return_results is False: None
    - If return_results is True
        - zstat : (float)
            test statistic for the z-test
        - p-value : (float)
            p-value for the z-test
    """
    if alternative not in ["two-sided", "2s", "larger", "l", "smaller", "s"]:
        raise Exception(
            f"alternative must be 'two-sided', '2s', 'larger', 'l', 'smaller', 's', but got {alternative}")
    alternative_map = {
        "two-sided": "two-sided",
        "2s": "two-sided",
        "larger": "larger",
        "l": "larger",
        "smaller": "smaller",
        "s": "smaller"
    }
    alternative = alternative_map[alternative]
    if alpha < 0 or alpha > 1:
        raise Exception(f"alpha must be between 0 and 1, but got {alpha}")
    if not all(isinstance(x, int) for x in [count1, count2, n1, n2]):
        raise ValueError("All input parameters must be integers")
    count = [count1, count2]
    nobs = [n1, n2]
    z_stat, p_value = stm.proportions_ztest(
        count=count, nobs=nobs, alternative=alternative)

    if not return_results:
        print('Z тест для долей')
        print('alpha = ', alpha)
        print('p-value = ', round(p_value, 3))
        if p_value < alpha:
            print(colored(
                "Отклоняем нулевую гипотезу, поскольку p-value меньше уровня значимости", 'red'))
        else:
            print(colored(
                "Нет оснований отвергнуть нулевую гипотезу, поскольку p-value больше или равно уровню значимости", 'green'))
    else:
        return z_stat, p_value


def proportions_ztest_column_2sample(column: pd.Series, alpha: float = 0.05, alternative: str = 'two-sided', return_results: bool = False):
    """
    Perform a z-test for proportions on a single column.

    Parameters:
    - column (pandas Series): The input column with two unique values.
    - alpha (float, optional): The significance level (default=0.05).
    - alternative (str, optional): The alternative hypothesis ('two-sided', '2s', 'larger', 'l', 'smaller', 's') (default='two-sided').
    - return_results (bool, optional): Return (z_stat, p_value) instead of printing (default=False).

    Returns:
    - If return_results is False: None
    - If return_results is True
        - zstat : (float)
            test statistic for the z-test
        - p-value : (float)
            p-value for the z-test
    """
    if alternative not in ["two-sided", "2s", "larger", "l", "smaller", "s"]:
        raise Exception(
            f"alternative must be 'two-sided', '2s', 'larger', 'l', 'smaller', 's', but got {alternative}")
    alternative_map = {
        "two-sided": "two-sided",
        "2s": "two-sided",
        "larger": "larger",
        "l": "larger",
        "smaller": "smaller",
        "s": "smaller"
    }
    alternative = alternative_map[alternative]
    if alpha < 0 or alpha > 1:
        raise Exception(f"alpha must be between 0 and 1, but got {alpha}")
    if not isinstance(column, pd.Series):
        raise ValueError("Input column must be pd.Series")
    if len(column) < 1:
        raise ValueError("Input column must have at least one value")
    if column.isna().sum():
        raise Exception(
            f'column must not have missing values.\ncolumn have {column.isna().sum()} missing values')

    if column.unique().size != 2:
        raise Exception(
            f'column must have exactly two unique values.\ncolumn have {column.unique().size} unique values')

    value_counts = column.value_counts()
    count1 = value_counts.values[0]
    count2 = value_counts.values[1]

    if count1 < 2 or count2 < 2:
        raise ValueError("Each sample must have at least two elements")
    elif count1 < 30 or count2 < 30:
        print(colored(
            "Warning: Sample size is less than 30 for one or more samples. Results may be unreliable.", 'red'))

    n1 = n2 = column.size
    count = [count1, count2]
    nobs = [n1, n2]

    z_stat, p_value = stm.proportions_ztest(
        count=count, nobs=nobs, alternative=alternative)
    if not return_results:
        print('Z тест для долей')
        print('alpha = ', alpha)
        print('p-value = ', round(p_value, 3))
        if p_value < alpha:
            print(colored(
                "Отклоняем нулевую гипотезу, поскольку p-value меньше уровня значимости", 'red'))
        else:
            print(colored(
                "Нет оснований отвергнуть нулевую гипотезу, поскольку p-value больше или равно уровню значимости", 'green'))
    else:
        return z_stat, p_value


def proportions_chi2(count1: int, count2: int, n1: int, n2: int, alpha: float = 0.05, alternative: str = 'two-sided', return_results: bool = False) -> None:
    """
    Perform a chi-squared test for proportions.

    Parameters:
    - count1 (int): Number of successes in the first sample
    - count2 (int): Number of successes in the second sample
    - n1 (int): Total number of observations in the first sample
    - n2 (int): Total number of observations in the second sample
    - alpha (float, optional): Significance level (default: 0.05)
    - alternative (str, optional): Alternative hypothesis ('two-sided', '2s', 'larger', 'l', 'smaller', 's') (default: 'two-sided')
    - return_results (bool, optional): Return (chi2_stat, p_value) instead of printing (default=False)

    Returns:
    - If return_results is False: None
    - If return_results is True
        - chi2_stat : (float)
            test statistic for the chi-squared test, asymptotically chi-squared distributed
        - p-value : (float)
            p-value for the chi-squared test
    """
    if alternative not in ["two-sided", "2s", "larger", "l", "smaller", "s"]:
        raise Exception(
            f"alternative must be 'two-sided', '2s', 'larger', 'l', 'smaller', 's', but got {alternative}")
    alternative_map = {
        "two-sided": "two-sided",
        "2s": "two-sided",
        "larger": "larger",
        "l": "larger",
        "smaller": "smaller",
        "s": "smaller"
    }
    alternative = alternative_map[alternative]
    if alpha < 0 or alpha > 1:
        raise Exception(f"alpha must be between 0 and 1, but got {alpha}")
    if not all(isinstance(x, int) for x in [count1, count2, n1, n2]):
        raise ValueError("All input parameters must be integers")

    chi2_stat, p_value = stm.test_proportions_2indep(
        count1, n1, count2, n2, alternative=alternative).tuple

    if not return_results:
        print('Хи-квадрат тест для долей')
        print('alpha = ', alpha)
        print('p-value = ', round(p_value, 3))
        if p_value < alpha:
            print(colored(
                "Отклоняем нулевую гипотезу, поскольку p-value меньше уровня значимости", 'red'))
        else:
            print(colored(
                "Нет оснований отвергнуть нулевую гипотезу, поскольку p-value больше или равно уровню значимости", 'green'))
    else:
        return chi2_stat, p_value


def proportions_chi2_column(column: pd.Series, alpha: float = 0.05, alternative: str = 'two-sided', return_results: bool = False):
    """
    Perform a chi-squared test for proportions on a single column.

    Parameters:
    - column (pandas Series): The input column with two unique values.
    - alpha (float, optional): The significance level (default=0.05).
    - alternative (str, optional): The alternative hypothesis ('two-sided', '2s', 'larger', 'l', 'smaller', 's') (default='two-sided').
    - return_results (bool, optional): Return (chi2_stat, p_value) instead of printing (default=False).

    Returns:
    - If return_results is False: None
    - If return_results is True
        - chi2_stat : (float)
            test statistic for the chi-squared test, asymptotically chi-squared distributed
        - p-value : (float)
            p-value for the chi-squared test
    """
    if alternative not in ["two-sided", "2s", "larger", "l", "smaller", "s"]:
        raise Exception(
            f"alternative must be 'two-sided', '2s', 'larger', 'l', 'smaller', 's', but got {alternative}")
    alternative_map = {
        "two-sided": "two-sided",
        "2s": "two-sided",
        "larger": "larger",
        "l": "larger",
        "smaller": "smaller",
        "s": "smaller"
    }
    alternative = alternative_map[alternative]
    if alpha < 0 or alpha > 1:
        raise Exception(f"alpha must be between 0 and 1, but got {alpha}")
    if not isinstance(column, pd.Series):
        raise ValueError("Input column must be pd.Series")
    if len(column) < 1:
        raise ValueError("Input column must have at least one value")
    if column.isna().sum():
        raise Exception(
            f'column must not have missing values.\ncolumn have {column.isna().sum()} missing values')

    if column.unique().size != 2:
        raise Exception(
            f'column must have exactly two unique values.\ncolumn have {column.unique().size} unique values')

    value_counts = column.value_counts()
    count1 = value_counts.values[0]
    count2 = value_counts.values[1]
    n1 = n2 = column.size

    chi2_stat, p_value = stm.test_proportions_2indep(
        count1, n1, count2, n2, alternative=alternative).tuple

    if not return_results:
        print('Хи-квадрат тест для долей')
        print('alpha = ', alpha)
        print('p-value = ', round(p_value, 3))
        if p_value < alpha:
            print(colored(
                "Отклоняем нулевую гипотезу, поскольку p-value меньше уровня значимости", 'red'))
        else:
            print(colored(
                "Нет оснований отвергнуть нулевую гипотезу, поскольку p-value больше или равно уровню значимости", 'green'))
    else:
        return chi2_stat, p_value


def anova_oneway_df(df: pd.DataFrame, alpha: float = 0.05, return_results: bool = False) -> None:
    """
    Perform a one-way ANOVA test.

    Parameters:
    - df (pd.DataFrame): DataFrame containing two columns,
        where the first column contains labels and
        the second column contains corresponding values
    - alpha (float, optional): Significance level (default: 0.05)
    - return_results (bool, optional): Return (statistic, p_value) instead of printing (default=False).

    Returns:
    - If return_results is False: None
    - If return_results is True
        - statistic : (float)
            The calculated F-statistic.
        - pvalue : (float)
            The associated p-value from the F distribution
    """
    if alpha < 0 or alpha > 1:
        raise Exception(f"alpha must be between 0 and 1, but got {alpha}")
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input df must be pd.DataFrame")
    if df.shape[1] != 2:
        raise ValueError("Input DataFrame must have exactly two columns")
    labels = df.iloc[:, 0]
    values = df.iloc[:, 1]
    if not pd.api.types.is_numeric_dtype(values):
        raise ValueError("Value column must contain numeric values")
    if labels.isna().sum() or values.isna().sum():
        raise ValueError(
            f'labels and values must not have missing values.\nlabels have {labels.isna().sum()} missing values\nvalues have {values.isna().sum()} missing values')
    unique_labels = labels.unique()
    if len(unique_labels) < 2:
        raise ValueError("Labels must contain at least two unique values")

    samples = [values[labels == label] for label in unique_labels]
    warning_issued = False
    for sample in samples:
        if len(sample) < 2:
            raise ValueError("Each sample must have at least two elements")
        elif len(sample) < 30:
            warning_issued = True
    if warning_issued:
        print(colored(
            "Warning: Sample size is less than 30 for one or more samples. Results may be unreliable.", 'red'))
    statistic, p_value = stats.f_oneway(*samples)

    if not return_results:
        print('Однофакторный дисперсионный анализ')
        print('alpha = ', alpha)
        print('p-value = ', round(p_value, 3))
        if p_value < alpha:
            print(colored(
                "Отклоняем нулевую гипотезу, поскольку p-value меньше уровня значимости", 'red'))
        else:
            print(colored(
                "Нет оснований отвергнуть нулевую гипотезу, поскольку p-value больше или равно уровню значимости", 'green'))
    else:
        return statistic, p_value


def anova_oneway(samples: list, alpha: float = 0.05, return_results: bool = False) -> None:
    """
    Perform a one-way ANOVA test.

    Parameters:
    - samples (list): List of pd.Series, where each pd.Series contains values. There must be at least two samples.
    - alpha (float, optional): Significance level (default: 0.05)
    - return_results (bool, optional): Return (statistic, p_value) instead of printing (default=False)

    Returns:
    - If return_results is False: None
    - If return_results is True
        - statistic : (float)
            The calculated F-statistic.
        - pvalue : (float)
            The associated p-value from the F distribution
    """
    if alpha < 0 or alpha > 1:
        raise Exception(f"alpha must be between 0 and 1, but got {alpha}")
    if not all(isinstance(sample, pd.Series) for sample in samples):
        raise ValueError("Input samples must be a list of pd.Series")
    if not all(pd.api.types.is_numeric_dtype(sample) for sample in samples):
        raise ValueError("All values in samples must be numeric")
    if not all(len(sample) > 0 for sample in samples):
        raise ValueError("All samples must have at least one value")
    if len(samples) < 2:
        raise ValueError("Must have at least two samples")
    warning_issued = False
    for sample in samples:
        if len(sample) < 2:
            raise ValueError("Each sample must have at least two elements")
        elif len(sample) < 30:
            warning_issued = True
    if warning_issued:
        print(colored(
            "Warning: Sample size is less than 30 for one or more samples. Results may be unreliable.", 'red'))
    statistic, p_value = stats.f_oneway(*samples)
    if not return_results:
        print('Однофакторный дисперсионный анализ')
        print('alpha = ', alpha)
        print('p-value = ', round(p_value, 3))
        if p_value < alpha:
            print(colored(
                "Отклоняем нулевую гипотезу, поскольку p-value меньше уровня значимости", 'red'))
        else:
            print(colored(
                "Нет оснований отвергнуть нулевую гипотезу, поскольку p-value больше или равно уровню значимости", 'green'))
    else:
        return statistic, p_value


def tukey_hsd_df(df: pd.DataFrame, alpha: float = 0.05) -> None:
    """
    Perform a Tukey's HSD test for pairwise comparisons.   
    This test is commonly used to identify significant differences between groups in an ANOVA analysis,

    Parameters:
    - df (pd.DataFrame): DataFrame containing two columns,
        where the first column contains labels and
        the second column contains corresponding values
    - alpha (float, optional): Significance level (default: 0.05)

    Returns:
    - None
    """
    if alpha < 0 or alpha > 1:
        raise Exception(f"alpha must be between 0 and 1, but got {alpha}")
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input df must be pd.DataFrame")
    if df.shape[1] != 2:
        raise ValueError("Input DataFrame must have exactly two columns")
    labels = df.iloc[:, 0]
    values = df.iloc[:, 1]
    if not pd.api.types.is_numeric_dtype(values):
        raise ValueError("Value column must contain numeric values")
    if labels.isna().sum() or values.isna().sum():
        raise ValueError(
            f'labels and values must not have missing values.\nlabels have {labels.isna().sum()} missing values\nvalues have {values.isna().sum()} missing values')
    unique_labels = labels.unique()
    if len(unique_labels) < 2:
        raise ValueError("Labels must contain at least two unique values")

    tukey = pairwise_tukeyhsd(endog=values, groups=labels, alpha=alpha)
    print(tukey)

def games_howell_df(df: pd.DataFrame, alpha: float = 0.05) -> None:
    """
    Perform a Games-Howell test for pairwise comparisons.
    This test is used to identify significant differences between groups when variances are unequal.

    Parameters:
    - df (pd.DataFrame): DataFrame containing two columns,
        where the first column contains labels and
        the second column contains corresponding values
    - alpha (float, optional): Significance level (default: 0.05)

    Returns:
    - None
    """
    # Input validation
    if alpha < 0 or alpha > 1:
        raise Exception(f"alpha must be between 0 and 1, but got {alpha}")
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input df must be pd.DataFrame")
    if df.shape[1] != 2:
        raise ValueError("Input DataFrame must have exactly two columns")
    labels = df.iloc[:, 0]
    values = df.iloc[:, 1]
    if not pd.api.types.is_numeric_dtype(values):
        raise ValueError("Value column must contain numeric values")
    if labels.isna().sum() or values.isna().sum():
        raise ValueError(
            f'labels and values must not have missing values.\nlabels have {labels.isna().sum()} missing values\nvalues have {values.isna().sum()} missing values')
    unique_labels = labels.unique()
    if len(unique_labels) < 2:
        raise ValueError("Labels must contain at least two unique values")

    # Perform Games-Howell test
    gh_test = pg.pairwise_gameshowell(data=df, dv=df.columns[1], between=df.columns[0], alpha=alpha)
    print(gh_test)

def anova_oneway_welch_df(df: pd.DataFrame, alpha: float = 0.05, return_results: bool = False) -> None:
    """
    Perform a one-way ANOVA test using Welch's ANOVA. It is more reliable when the two samples   
    have unequal variances and/or unequal sample sizes.

    Parameters:
    - df (pd.DataFrame): DataFrame containing two columns,
        where the first column contains labels and
        the second column contains corresponding values
    - alpha (float, optional): Significance level (default: 0.05)
    - return_results (bool, optional): Return (ANOVA summary) instead of printing (default=False).

    Returns:
    - If return_results is False: None
    - If return_results is True: (pandas.DataFrame) aov
            ANOVA summary:
            - 'Source': Factor names
            - 'SS': Sums of squares
            - 'DF': Degrees of freedom
            - 'MS': Mean squares
            - 'F': F-values
            - 'p-unc': uncorrected p-values
            - 'np2': Partial eta-squared
    """
    if alpha < 0 or alpha > 1:
        raise Exception(f"alpha must be between 0 and 1, but got {alpha}")
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input df must be pd.DataFrame")
    if df.shape[1] != 2:
        raise ValueError("Input DataFrame must have exactly two columns")
    labels_column = df.columns[0]
    value_column = df.columns[1]
    labels = df[labels_column]
    values = df[value_column]

    if not pd.api.types.is_numeric_dtype(values):
        raise ValueError("Value column must contain numeric values")
    if labels.isna().sum() or values.isna().sum():
        raise ValueError(
            f'labels and values must not have missing values.\nlabels have {labels.isna().sum()} missing values\nvalues have {values.isna().sum()} missing values')
    unique_labels = labels.unique()
    if len(unique_labels) < 2:
        raise ValueError("Labels must contain at least two unique values")

    anova_results = pg.welch_anova(
        dv=value_column, between=labels_column, data=df)
    p_value = anova_results['p-unc'][0]
    if not return_results:
        print('Однофакторный дисперсионный анализ Welch')
        print('alpha = ', alpha)
        print('p-value = ', round(p_value, 3))
        if p_value < alpha:
            print(colored(
                "Отклоняем нулевую гипотезу, поскольку p-value меньше уровня значимости", 'red'))
        else:
            print(colored(
                "Нет оснований отвергнуть нулевую гипотезу, поскольку p-value больше или равно уровню значимости", 'green'))
    else:
        return anova_results


def kruskal_df(df: pd.DataFrame, alpha: float = 0.05, return_results: bool = False) -> None:
    """
    Perform a Kruskal-Wallis test. The Kruskal-Wallis H-test tests the null hypothesis  
    that the population median of all of the groups are equal. It is a non-parametric version of ANOVA. 

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing two columns
        - first column contains labels and
        - second column contains corresponding values
    alpha : float, optional
        Significance level (default: 0.05)
    return_results : bool, optional
        Whether to return results (statistic, p_value) instead of printing (default=False).

    Returns
    ----------
    - If return_results is False: None
    - If return_results is True
        - statistic : (float)
            The calculated H-statistic.
        - pvalue : (float)
            The associated p-value from the chi-squared distribution
    """
    if alpha < 0 or alpha > 1:
        raise Exception(f"alpha must be between 0 and 1, but got {alpha}")
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input df must be pd.DataFrame")
    if df.shape[1] != 2:
        raise ValueError("Input DataFrame must have exactly two columns")
    labels = df.iloc[:, 0]
    values = df.iloc[:, 1]
    if not pd.api.types.is_numeric_dtype(values):
        raise ValueError("Value column must contain numeric values")
    if labels.isna().sum() or values.isna().sum():
        raise ValueError(
            f'labels and values must not have missing values.\nlabels have {labels.isna().sum()} missing values\nvalues have {values.isna().sum()} missing values')
    unique_labels = labels.unique()
    if len(unique_labels) < 2:
        raise ValueError("Labels must contain at least two unique values")

    samples = [values[labels == label] for label in unique_labels]
    warning_issued = False
    for sample in samples:
        if len(sample) < 2:
            raise ValueError("Each sample must have at least two elements")
        elif len(sample) < 30:
            warning_issued = True
    if warning_issued:
        print(colored(
            "Warning: Sample size is less than 30 for one or more samples. Results may be unreliable.", 'red'))
    statistic, p_value = stats.kruskal(*samples)

    if not return_results:
        print('Тест Краскела-Уоллиса (H-критерий)')
        print('alpha = ', alpha)
        print('p-value = ', round(p_value, 3))
        if p_value < alpha:
            print(colored(
                "Отклоняем нулевую гипотезу, поскольку p-value меньше уровня значимости", 'red'))
        else:
            print(colored(
                "Нет оснований отвергнуть нулевую гипотезу, поскольку p-value больше или равно уровню значимости", 'green'))
    else:
        return statistic, p_value


def kruskal(samples: list, alpha: float = 0.05, return_results: bool = False) -> None:
    """
    Perform a Kruskal-Wallis test. The Kruskal-Wallis H-test tests the null hypothesis  
    that the population median of all of the groups are equal. It is a non-parametric version of ANOVA. 

    Parameters:
    - samples (list): List of pd.Series, where each pd.Series contains values. There must be at least two samples.
    - alpha (float, optional): Significance level (default: 0.05)
    - return_results (bool, optional): Return (statistic, p_value) instead of printing (default=False)

    Returns:
    - If return_results is False: None
    - If return_results is True
        - statistic : (float)
            The calculated H-statistic.
        - pvalue : (float)
            The associated p-value from the chi-squared distribution
    """
    if alpha < 0 or alpha > 1:
        raise Exception(f"alpha must be between 0 and 1, but got {alpha}")
    if not all(isinstance(sample, pd.Series) for sample in samples):
        raise ValueError("Input samples must be a list of pd.Series")
    if not all(pd.api.types.is_numeric_dtype(sample) for sample in samples):
        raise ValueError("All values in samples must be numeric")
    if not all(len(sample) > 0 for sample in samples):
        raise ValueError("All samples must have at least one value")
    if len(samples) < 2:
        raise ValueError("Must have at least two samples")
    warning_issued = False
    for sample in samples:
        if len(sample) < 2:
            raise ValueError("Each sample must have at least two elements")
        elif len(sample) < 30:
            warning_issued = True
    if warning_issued:
        print(colored(
            "Warning: Sample size is less than 30 for one or more samples. Results may be unreliable.", 'red'))
    statistic, p_value = stats.kruskal(*samples)
    if not return_results:
        print('Тест Краскела-Уоллиса (H-критерий)')
        print('alpha = ', alpha)
        print('p-value = ', round(p_value, 3))
        if p_value < alpha:
            print(colored(
                "Отклоняем нулевую гипотезу, поскольку p-value меньше уровня значимости", 'red'))
        else:
            print(colored(
                "Нет оснований отвергнуть нулевую гипотезу, поскольку p-value больше или равно уровню значимости", 'green'))
    else:
        return statistic, p_value


def levene_df(df: pd.DataFrame, alpha: float = 0.05, return_results: bool = False) -> None:
    """
    Perform a Levene's test. Levene's test is a statistical test used to check if the variances of multiple samples are equal.

    Parameters:
    - df (pd.DataFrame): DataFrame containing two columns,
        where the first column contains labels and
        the second column contains corresponding values
    - alpha (float, optional): Significance level (default: 0.05)
    - return_results (bool, optional): Return (statistic, p_value) instead of printing (default=False).

    Returns:
    - If return_results is False: None
    - If return_results is True
        - statistic : (float)
            The calculated W-statistic.
        - pvalue : (float)
            The associated p-value from the F distribution
    """
    if alpha < 0 or alpha > 1:
        raise Exception(f"alpha must be between 0 and 1, but got {alpha}")
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input df must be pd.DataFrame")
    if df.shape[1] != 2:
        raise ValueError("Input DataFrame must have exactly two columns")
    labels = df.iloc[:, 0]
    values = df.iloc[:, 1]
    if not pd.api.types.is_numeric_dtype(values):
        raise ValueError("Value column must contain numeric values")
    if labels.isna().sum() or values.isna().sum():
        raise ValueError(
            f'labels and values must not have missing values.\nlabels have {labels.isna().sum()} missing values\nvalues have {values.isna().sum()} missing values')
    unique_labels = labels.unique()
    if len(unique_labels) < 2:
        raise ValueError("Labels must contain at least two unique values")

    samples = [values[labels == label] for label in unique_labels]
    warning_issued = False
    for sample in samples:
        if len(sample) < 2:
            raise ValueError("Each sample must have at least two elements")
        elif len(sample) < 30:
            warning_issued = True
    if warning_issued:
        print(colored(
            "Warning: Sample size is less than 30 for one or more samples. Results may be unreliable.", 'red'))
    statistic, p_value = stats.levene(*samples)

    if not return_results:
        print('Тест Левена')
        print('alpha = ', alpha)
        print('p-value = ', round(p_value, 3))
        if p_value < alpha:
            print(colored(
                "Отклоняем нулевую гипотезу, поскольку p-value меньше уровня значимости", 'red'))
        else:
            print(colored(
                "Нет оснований отвергнуть нулевую гипотезу, поскольку p-value больше или равно уровню значимости", 'green'))
    else:
        return statistic, p_value


def levene(samples: list, alpha: float = 0.05, return_results: bool = False) -> None:
    """
    Perform a Levene's test. Levene's test is a statistical test used to check if the variances of multiple samples are equal.

    Parameters:
    - samples (list): List of pd.Series, where each pd.Series contains values. There must be at least two samples.
    - alpha (float, optional): Significance level (default: 0.05)
    - return_results (bool, optional): Return (statistic, p_value) instead of printing (default=False)

    Returns:
    - If return_results is False: None
    - If return_results is True
        - statistic : (float)
            The calculated W-statistic.
        - pvalue : (float)
            The associated p-value from the F distribution
    """
    if alpha < 0 or alpha > 1:
        raise Exception(f"alpha must be between 0 and 1, but got {alpha}")
    if not all(isinstance(sample, pd.Series) for sample in samples):
        raise ValueError("Input samples must be a list of pd.Series")
    if not all(pd.api.types.is_numeric_dtype(sample) for sample in samples):
        raise ValueError("All values in samples must be numeric")
    if not all(len(sample) > 0 for sample in samples):
        raise ValueError("All samples must have at least one value")
    if len(samples) < 2:
        raise ValueError("Must have at least two samples")
    warning_issued = False
    for sample in samples:
        if len(sample) < 2:
            raise ValueError("Each sample must have at least two elements")
        elif len(sample) < 30:
            warning_issued = True
    if warning_issued:
        print(colored(
            "Warning: Sample size is less than 30 for one or more samples. Results may be unreliable.", 'red'))

    statistic, p_value = stats.levene(*samples)
    if not return_results:
        print('Тест Левена на гомогенность дисперсии')
        print('alpha = ', alpha)
        print('p-value = ', round(p_value, 3))
        if p_value < alpha:
            print(colored(
                "Отклоняем нулевую гипотезу, поскольку p-value меньше уровня значимости", 'red'))
        else:
            print(colored(
                "Нет оснований отвергнуть нулевую гипотезу, поскольку p-value больше или равно уровню значимости", 'green'))
    else:
        return statistic, p_value


def bartlett_df(df: pd.DataFrame, alpha: float = 0.05, return_results: bool = False) -> None:
    """
    Perform a Bartlett's test. Bartlett's test is a statistical test used to check if the variances of multiple samples are equal.

    Parameters:
    - df (pd.DataFrame): DataFrame containing two columns,
        where the first column contains labels and
        the second column contains corresponding values
    - alpha (float, optional): Significance level (default: 0.05)
    - return_results (bool, optional): Return (statistic, p_value) instead of printing (default=False).

    Returns:
    - If return_results is False: None
    - If return_results is True
        - statistic : (float)
            The calculated chi-squared statistic.
        - pvalue : (float)
            The associated p-value from the chi-squared distribution
    """
    if alpha < 0 or alpha > 1:
        raise Exception(f"alpha must be between 0 and 1, but got {alpha}")
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input df must be pd.DataFrame")
    if df.shape[1] != 2:
        raise ValueError("Input DataFrame must have exactly two columns")
    labels = df.iloc[:, 0]
    values = df.iloc[:, 1]
    if not pd.api.types.is_numeric_dtype(values):
        raise ValueError("Value column must contain numeric values")
    if labels.isna().sum() or values.isna().sum():
        raise ValueError(
            f'labels and values must not have missing values.\nlabels have {labels.isna().sum()} missing values\nvalues have {values.isna().sum()} missing values')
    unique_labels = labels.unique()
    if len(unique_labels) < 2:
        raise ValueError("Labels must contain at least two unique values")

    samples = [values[labels == label] for label in unique_labels]
    warning_issued = False
    for sample in samples:
        if len(sample) < 2:
            raise ValueError("Each sample must have at least two elements")
        elif len(sample) < 30:
            warning_issued = True
    if warning_issued:
        print(colored(
            "Warning: Sample size is less than 30 for one or more samples. Results may be unreliable.", 'red'))
    statistic, p_value = stats.bartlett(*samples)

    if not return_results:
        print('Тест Бартлетта')
        print('alpha = ', alpha)
        print('p-value = ', round(p_value, 3))
        if p_value < alpha:
            print(colored(
                "Отклоняем нулевую гипотезу, поскольку p-value меньше уровня значимости", 'red'))
        else:
            print(colored(
                "Нет оснований отвергнуть нулевую гипотезу, поскольку p-value больше или равно уровню значимости", 'green'))
    else:
        return statistic, p_value


def bartlett(samples: list, alpha: float = 0.05, return_results: bool = False) -> None:
    """
    Perform a Bartlett's test. Bartlett's test is a statistical test used to check if the variances of multiple samples are equal.

    Parameters:
    - samples (list): List of pd.Series, where each pd.Series contains values. There must be at least two samples.
    - alpha (float, optional): Significance level (default: 0.05)
    - return_results (bool, optional): Return (statistic, p_value) instead of printing (default=False)

    Returns:
    - If return_results is False: None
    - If return_results is True
        - statistic : (float)
            The calculated chi-squared statistic.
        - pvalue : (float)
            The associated p-value from the chi-squared distribution
    """
    if alpha < 0 or alpha > 1:
        raise Exception(f"alpha must be between 0 and 1, but got {alpha}")
    if not all(isinstance(sample, pd.Series) for sample in samples):
        raise ValueError("Input samples must be a list of pd.Series")
    if not all(pd.api.types.is_numeric_dtype(sample) for sample in samples):
        raise ValueError("All values in samples must be numeric")
    if not all(len(sample) > 0 for sample in samples):
        raise ValueError("All samples must have at least one value")
    if len(samples) < 2:
        raise ValueError("Must have at least two samples")
    warning_issued = False
    for sample in samples:
        if len(sample) < 2:
            raise ValueError("Each sample must have at least two elements")
        elif len(sample) < 30:
            warning_issued = True
    if warning_issued:
        print(colored(
            "Warning: Sample size is less than 30 for one or more samples. Results may be unreliable.", 'red'))
    statistic, p_value = stats.bartlett(*samples)
    if not return_results:
        print('Тест Бартлетта')
        print('alpha = ', alpha)
        print('p-value = ', round(p_value, 3))
        if p_value < alpha:
            print(colored(
                "Отклоняем нулевую гипотезу, поскольку p-value меньше уровня значимости", 'red'))
        else:
            print(colored(
                "Нет оснований отвергнуть нулевую гипотезу, поскольку p-value больше или равно уровню значимости", 'green'))
    else:
        return statistic, p_value


def confint_t_2samples(sample1: pd.Series, sample2: pd.Series, alpha: float = 0.05, alternative: str = 'two-sided', equal_var=False) -> tuple:
    """
    Calculate the confidence interval using t-statistic for the difference in means between two samples.

    Parameters:
    - sample1, sample2: Pandas Series objects
    - alpha : float (default=0.05).
        Significance level for the confidence interval, coverage is
        ``1-alpha``.
    - alternative : (str, optional) (default='two-sided').
        The alternative hypothesis, H1, has to be one of the following

           * 'two-sided' : H1: ``value1 - value2 - diff`` not equal to 0.
           * 'larger' :   H1: ``value1 - value2 - diff > 0``
           * 'smaller' :  H1: ``value1 - value2 - diff < 0``
    - equal_var (bool): Whether to assume equal variances between the two samples. If `True`, the pooled standard deviation is used.   
    If `False`, the standard error is calculated using the separate variances of each sample. Defaults to `False`.

    Returns
    -------
    - ci: tuple, confidence interval (lower, upper)
        - lower : float
            Lower confidence limit. This is -inf for the one-sided alternative
            "smaller".
        - upper : float
            Upper confidence limit. This is inf for the one-sided alternative
            "larger".
    """
    if alternative not in ["two-sided", "2s", "larger", "l", "smaller", "s"]:
        raise Exception(
            f"alternative must be 'two-sided', '2s', 'larger', 'l', 'smaller', 's', but got {alternative}")
    alternative_map = {
        "two-sided": "two-sided",
        "2s": "two-sided",
        "larger": "larger",
        "l": "larger",
        "smaller": "smaller",
        "s": "smaller"
    }
    alternative = alternative_map[alternative]
    if alpha < 0 or alpha > 1:
        raise Exception(f"alpha must be between 0 and 1, but got {alpha}")
    # Check if samples are Pandas Series objects
    if not (isinstance(sample1, pd.Series) and isinstance(sample2, pd.Series)):
        raise ValueError("Samples must be Pandas Series objects")
    warning_issued = False
    for sample in [sample1, sample2]:
        if len(sample) < 2:
            raise ValueError("Each sample must have at least two elements")
        elif len(sample) < 30:
            warning_issued = True
    if warning_issued:
        print(colored(
            "Warning: Sample size is less than 30 for one or more samples. Results may be unreliable.", 'red'))
    # Calculate means and variances
    mean1, var1 = sample1.mean(), sample1.var(ddof=1)
    mean2, var2 = sample2.mean(), sample2.var(ddof=1)

    # Calculate sample sizes
    n1 = len(sample1)
    n2 = len(sample2)

    if equal_var:
        # Calculate pooled standard deviation
        pooled_std = np.sqrt(
            ((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
        standard_error = pooled_std * np.sqrt(1/n1 + 1/n2)

        dof = n1 + n2 - 2
    else:
        varn1 = var1 / n1
        varn2 = var2 / n2
        dof = (varn1 + varn2)**2 / (varn1**2 / (n1 - 1) + varn2**2 / (n2 - 1))
        standard_error = np.sqrt(varn1 + varn2)

    # Calculate critical value and confidence interval bounds
    if alternative in ["two-sided", "2s"]:
        tcrit = stats.t.ppf(1 - alpha / 2.0, dof)
        lower = mean1 - mean2 - tcrit * standard_error
        upper = mean1 - mean2 + tcrit * standard_error
    elif alternative in ["larger", "l"]:
        tcrit = stats.t.ppf(alpha, dof)
        lower = mean1 - mean2 + tcrit * standard_error
        upper = np.inf
    elif alternative in ["smaller", "s"]:
        tcrit = stats.t.ppf(1 - alpha, dof)
        lower = -np.inf
        upper = mean1 - mean2 + tcrit * standard_error

    return lower, upper


def confint_t_1sample(sample: pd.Series, alpha: float = 0.05, alternative: str = 'two-sided') -> tuple:
    """
    Calculate the confidence interval using t-statistic for the mean of one sample.

    Parameters:
    - sample: Pandas Series object
    - alpha : float (default=0.05).
        Significance level for the confidence interval, coverage is
        ``1-alpha``.
    - alternative : (str, optional) (default='two-sided').
        The alternative hypothesis, H1, has to be one of the following

           * 'two-sided' : H1: ``value - mu`` not equal to 0.
           * 'larger' :   H1: ``value - mu > 0``
           * 'smaller' :  H1: ``value - mu < 0``

    Returns
    -------
    - ci: tuple, confidence interval (lower, upper)
        - lower : float
            Lower confidence limit. This is -inf for the one-sided alternative
            "smaller".
        - upper : float
            Upper confidence limit. This is inf for the one-sided alternative
            "larger".
    """
    if alternative not in ["two-sided", "2s", "larger", "l", "smaller", "s"]:
        raise Exception(
            f"alternative must be 'two-sided', '2s', 'larger', 'l', 'smaller', 's', but got {alternative}")
    alternative_map = {
        "two-sided": "two-sided",
        "2s": "two-sided",
        "larger": "larger",
        "l": "larger",
        "smaller": "smaller",
        "s": "smaller"
    }
    alternative = alternative_map[alternative]
    if alpha < 0 or alpha > 1:
        raise Exception(f"alpha must be between 0 and 1, but got {alpha}")
    # Check if sample is Pandas Series object
    if not isinstance(sample, pd.Series):
        raise ValueError("Sample must be Pandas Series object")
    if len(sample) < 2:
        raise ValueError("Sample must have at least two elements")
    elif len(sample) < 30:
        print(colored(
            "Warning: Sample size is less than 30. Results may be unreliable.", 'red'))
    # Calculate mean and variance
    mean, var = sample.mean(), sample.var(ddof=1)

    # Calculate sample size
    n = len(sample)

    # Calculate standard error
    standard_error = np.sqrt(var / n)

    # Calculate degrees of freedom
    dof = n - 1

    # Calculate critical value and confidence interval bounds
    if alternative in ["two-sided", "2s"]:
        tcrit = stats.t.ppf(1 - alpha / 2.0, dof)
        lower = mean - tcrit * standard_error
        upper = mean + tcrit * standard_error
    elif alternative in ["larger", "l"]:
        tcrit = stats.t.ppf(alpha, dof)
        lower = mean + tcrit * standard_error
        upper = np.inf
    elif alternative in ["smaller", "s"]:
        tcrit = stats.t.ppf(1 - alpha, dof)
        lower = -np.inf
        upper = mean + tcrit * standard_error

    return lower, upper


def confint_t_2samples_df(df: pd.DataFrame, alpha: float = 0.05, alternative: str = 'two-sided', first_for_alternative: str = None, equal_var=False) -> tuple:
    """
    Calculate the confidence interval using t-statistic for the difference in means between two samples.

    Parameters:
    - df (pd.DataFrame): DataFrame containing two columns, where the first column contains labels and the second column contains values
    - alpha : float (default=0.05).
        Significance level for the confidence interval, coverage is
        ``1-alpha``.
    - alternative : (str, optional) (default='two-sided').
        The alternative hypothesis, H1, has to be one of the following

           * 'two-sided' : H1: ``value1 - value2 - diff`` not equal to 0.
           * 'larger' :   H1: ``value1 - value2 - diff > 0``
           * 'smaller' :  H1: ``value1 - value2 - diff < 0``
    - first_for_alternative (str, optional): Lable name in first column of df for define first sample for scipy.stats.ttest_ind
    - equal_var (bool): Whether to assume equal variances between the two samples. If `True`, the pooled standard deviation is used.   
    If `False`, the standard error is calculated using the separate variances of each sample. Defaults to `False`.

    Returns
    -------
    - ci: tuple, confidence interval (lower, upper)
        - lower : float
            Lower confidence limit. This is -inf for the one-sided alternative
            "smaller".
        - upper : float
            Upper confidence limit. This is inf for the one-sided alternative
            "larger".
    """
    if alternative not in ["two-sided", "2s", "larger", "l", "smaller", "s"]:
        raise Exception(
            f"alternative must be 'two-sided', '2s', 'larger', 'l', 'smaller', 's', but got {alternative}")
    alternative_map = {
        "two-sided": "two-sided",
        "2s": "two-sided",
        "larger": "larger",
        "l": "larger",
        "smaller": "smaller",
        "s": "smaller"
    }
    alternative = alternative_map[alternative]
    if alpha < 0 or alpha > 1:
        raise Exception(f"alpha must be between 0 and 1, but got {alpha}")
    # Check if input is a Pandas DataFrame
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input must be a Pandas DataFrame")
    if df.shape[1] != 2:
        raise ValueError("Input DataFrame must have exactly two columns")

    # Extract labels and values from the DataFrame
    labels = df.iloc[:, 0]
    values = df.iloc[:, 1]

    # Check if values are numeric
    if not pd.api.types.is_numeric_dtype(values):
        raise ValueError("Values must be numeric")

    # Check for missing values
    if labels.isna().sum() or values.isna().sum():
        raise ValueError("Labels and values must not have missing values")

    # Extract unique labels
    unique_labels = labels.unique()
    if len(unique_labels) != 2:
        raise ValueError("Labels must contain exactly two unique values")
    if first_for_alternative:
        if first_for_alternative not in unique_labels:
            raise ValueError('first_for_alternative must be a lable in first column of df')
        if first_for_alternative != unique_labels[0]:
            unique_labels = unique_labels[::-1]
    # Split values into two samples based on labels
    sample1 = values[labels == unique_labels[0]]
    sample2 = values[labels == unique_labels[1]]
    warning_issued = False
    for sample in [sample1, sample2]:
        if len(sample) < 2:
            raise ValueError("Each sample must have at least two elements")
        elif len(sample) < 30:
            warning_issued = True
    if warning_issued:
        print(colored(
            "Warning: Sample size is less than 30 for one or more samples. Results may be unreliable.", 'red'))
    # Calculate means and variances
    mean1, var1 = sample1.mean(), sample1.var(ddof=1)
    mean2, var2 = sample2.mean(), sample2.var(ddof=1)

    # Calculate sample sizes
    n1 = len(sample1)
    n2 = len(sample2)
    if equal_var:
        # Calculate pooled standard deviation
        pooled_std = np.sqrt(
            ((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
        standard_error = pooled_std * np.sqrt(1/n1 + 1/n2)

        dof = n1 + n2 - 2
    else:
        varn1 = var1 / n1
        varn2 = var2 / n2
        dof = (varn1 + varn2)**2 / (varn1**2 / (n1 - 1) + varn2**2 / (n2 - 1))
        standard_error = np.sqrt(varn1 + varn2)

    # Calculate critical value and confidence interval bounds
    if alternative in ["two-sided", "2s"]:
        tcrit = stats.t.ppf(1 - alpha / 2.0, dof)
        lower = mean1 - mean2 - tcrit * standard_error
        upper = mean1 - mean2 + tcrit * standard_error
    elif alternative in ["larger", "l"]:
        tcrit = stats.t.ppf(alpha, dof)
        lower = mean1 - mean2 + tcrit * standard_error
        upper = np.inf
    elif alternative in ["smaller", "s"]:
        tcrit = stats.t.ppf(1 - alpha, dof)
        lower = -np.inf
        upper = mean1 - mean2 + tcrit * standard_error

    return lower, upper


def confint_proportion_ztest_1sample_column(sample: pd.Series, alpha: float = 0.05, alternative: str = 'two-sided') -> tuple:
    """
    Calculate the confidence interval for a proportion in one sample containing 0 or 1.

    Parameters:
    - sample: Pandas Series object containing 0 or 1
    - alpha : float (default=0.05).
        Significance level for the confidence interval, coverage is
        ``1-alpha``.
    - alternative : (str, optional) (default='two-sided').
        The alternative hypothesis, H1, has to be one of the following

           * 'two-sided' : H1: ``p - p0`` not equal to 0.
           * 'larger' :   H1: ``p - p0 > 0``
           * 'smaller' :  H1: ``p - p0 < 0``

    Returns
    -------
    - ci: tuple, confidence interval (lower, upper)
        - lower : float
            Lower confidence limit. This is -inf for the one-sided alternative
            "smaller".
        - upper : float
            Upper confidence limit. This is inf for the one-sided alternative
            "larger".
    """
    if alternative not in ["two-sided", "2s", "larger", "l", "smaller", "s"]:
        raise Exception(
            f"alternative must be 'two-sided', '2s', 'larger', 'l', 'smaller', 's', but got {alternative}")
    alternative_map = {
        "two-sided": "two-sided",
        "2s": "two-sided",
        "larger": "larger",
        "l": "larger",
        "smaller": "smaller",
        "s": "smaller"
    }
    alternative = alternative_map[alternative]
    # Check if sample contains only 0 and 1
    if not ((sample == 0) | (sample == 1)).all():
        raise ValueError("Sample must contain only 0 and 1")
    if alpha < 0 or alpha > 1:
        raise Exception(f"alpha must be between 0 and 1, but got {alpha}")
    # Check if sample is Pandas Series object
    if not isinstance(sample, pd.Series):
        raise ValueError("Sample must be Pandas Series object")
    if len(sample) < 2:
        raise ValueError("Sample must have at least two elements")
    elif len(sample) < 30:
        print(colored(
            "Warning: Sample size is less than 30. Results may be unreliable.", 'red'))
    # Calculate proportion
    p = sample.mean()

    # Calculate standard error
    standard_error = np.sqrt(p * (1 - p) / len(sample))

    # Calculate critical value and confidence interval bounds
    if alternative in ["two-sided", "2s"]:
        zcrit = stats.norm.ppf(1 - alpha / 2.0)
        lower = p - zcrit * standard_error
        upper = p + zcrit * standard_error
    elif alternative in ["larger", "l"]:
        zcrit = stats.norm.ppf(alpha)
        lower = p + zcrit * standard_error
        upper = 1
    elif alternative in ["smaller", "s"]:
        zcrit = stats.norm.ppf(1 - alpha)
        lower = 0
        upper = p + zcrit * standard_error

    return lower, upper


def confint_proportion_ztest_1sample(count: int, nobs: int, alpha: float = 0.05, alternative: str = 'two-sided') -> tuple:
    """
    Calculate the confidence interval for a proportion in one sample.

    Parameters:
    - count: int, number of successes (1s) in the sample
    - nobs: int, total number of observations in the sample
    - alpha : float (default=0.05).
        Significance level for the confidence interval, coverage is
        ``1-alpha``.
    - alternative : (str, optional) (default='two-sided').
        The alternative hypothesis, H1, has to be one of the following

           * 'two-sided' : H1: ``p - p0`` not equal to 0.
           * 'larger' :   H1: ``p - p0 > 0``
           * 'smaller' :  H1: ``p - p0 < 0``

    Returns
    -------
    - ci: tuple, confidence interval (lower, upper)
        - lower : float
            Lower confidence limit. This is -inf for the one-sided alternative
            "smaller".
        - upper : float
            Upper confidence limit. This is inf for the one-sided alternative
            "larger".
    """
    if alternative not in ["two-sided", "2s", "larger", "l", "smaller", "s"]:
        raise Exception(
            f"alternative must be 'two-sided', '2s', 'larger', 'l', 'smaller', 's', but got {alternative}")
    alternative_map = {
        "two-sided": "two-sided",
        "2s": "two-sided",
        "larger": "larger",
        "l": "larger",
        "smaller": "smaller",
        "s": "smaller"
    }
    alternative = alternative_map[alternative]
    if alpha < 0 or alpha > 1:
        raise Exception(f"alpha must be between 0 and 1, but got {alpha}")
    # Additional checks
    if nobs < 2:
        raise ValueError("Sample (nobs) must have at least two observations")
    if count > nobs:
        raise ValueError("Count cannot be greater than sample size")
    elif nobs < 30:
        print(colored(
            "Warning: Sample size is less than 30. Results may be unreliable.", 'red'))
    # Calculate proportion
    p = count / nobs

    # Calculate standard error
    standard_error = np.sqrt(p * (1 - p) / nobs)

    # Calculate critical value and confidence interval bounds
    if alternative in ["two-sided", "2s"]:
        zcrit = stats.norm.ppf(1 - alpha / 2.0)
        lower = p - zcrit * standard_error
        upper = p + zcrit * standard_error
    elif alternative in ["larger", "l"]:
        zcrit = stats.norm.ppf(alpha)
        lower = p + zcrit * standard_error
        upper = 1
    elif alternative in ["smaller", "s"]:
        zcrit = stats.norm.ppf(1 - alpha)
        lower = 0
        upper = p + zcrit * standard_error

    return lower, upper


def confint_proportion_ztest_2sample(count1: int, nobs1: int, count2: int, nobs2: int, alpha: float = 0.05, alternative: str = 'two-sided') -> tuple:
    """
    Calculate the confidence interval using normal distribution for the difference of two proportions.

    Parameters:
    - count1: int, number of successes (1s) in the first sample
    - nobs1: int, total number of observations in the first sample
    - count2: int, number of successes (1s) in the second sample
    - nobs2: int, total number of observations in the second sample
    - alpha : float (default=0.05).
        Significance level for the confidence interval, coverage is
        ``1-alpha``.
    - alternative : (str, optional) (default='two-sided').
        The alternative hypothesis, H1, has to be one of the following

           * 'two-sided' : H1: ``p1 - p2`` not equal to 0.
           * 'larger' :   H1: ``p1 - p2 > 0``
           * 'smaller' :  H1: ``p1 - p2 < 0``

    Returns
    -------
    - ci: tuple, confidence interval (lower, upper)
        - lower : float
            Lower confidence limit. This is -inf for the one-sided alternative
            "smaller".
        - upper : float
            Upper confidence limit. This is inf for the one-sided alternative
            "larger".
    """
    if alternative not in ["two-sided", "2s", "larger", "l", "smaller", "s"]:
        raise Exception(
            f"alternative must be 'two-sided', '2s', 'larger', 'l', 'smaller', 's', but got {alternative}")
    alternative_map = {
        "two-sided": "two-sided",
        "2s": "two-sided",
        "larger": "larger",
        "l": "larger",
        "smaller": "smaller",
        "s": "smaller"
    }
    alternative = alternative_map[alternative]
    if alpha < 0 or alpha > 1:
        raise Exception(f"alpha must be between 0 and 1, but got {alpha}")
    # Additional checks
    if nobs1 < 2 or nobs2 < 2:
        raise ValueError(
            "Each sample (nobs1 and nobs2) must have at least two observations")
    if count1 > nobs1 or count2 > nobs2:
        raise ValueError("Count cannot be greater than sample size")
    elif nobs1 < 30 or nobs2 < 30:
        print(colored(
            "Warning: Sample size is less than 30. Results may be unreliable.", 'red'))
    # Calculate proportions
    p1 = count1 / nobs1
    p2 = count2 / nobs2

    # Calculate standard error
    standard_error = np.sqrt(p1 * (1 - p1) / nobs1 + p2 * (1 - p2) / nobs2)

    # Calculate critical value and confidence interval bounds
    if alternative in ["two-sided", "2s"]:
        zcrit = stats.norm.ppf(1 - alpha / 2.0)
        lower = (p1 - p2) - zcrit * standard_error
        upper = (p1 - p2) + zcrit * standard_error
    elif alternative in ["larger", "l"]:
        zcrit = stats.norm.ppf(alpha)
        lower = (p1 - p2) + zcrit * standard_error
        upper = 1
    elif alternative in ["smaller", "s"]:
        zcrit = stats.norm.ppf(1 - alpha)
        lower = -1
        upper = (p1 - p2) + zcrit * standard_error

    return lower, upper


def confint_proportion_ztest_column_2sample(column: pd.Series, alpha: float = 0.05, alternative: str = 'two-sided') -> tuple:
    """
    Calculate the confidence interval using normal distribution for the difference of two proportions.

    Parameters:
    - column: pd.Series, input column with two unique values
    - alpha : float (default=0.05).
        Significance level for the confidence interval, coverage is
        ``1-alpha``.
    - alternative : (str, optional) (default='two-sided').
        The alternative hypothesis, H1, has to be one of the following

           * 'two-sided' : H1: ``p1 - p2`` not equal to 0.
           * 'larger' :   H1: ``p1 - p2 > 0``
           * 'smaller' :  H1: ``p1 - p2 < 0``

    Returns
    -------
    - ci: tuple, confidence interval (lower, upper)
        - lower : float
            Lower confidence limit. This is -inf for the one-sided alternative
            "smaller".
        - upper : float
            Upper confidence limit. This is inf for the one-sided alternative
            "larger".
    """
    if alternative not in ["two-sided", "2s", "larger", "l", "smaller", "s"]:
        raise Exception(
            f"alternative must be 'two-sided', '2s', 'larger', 'l', 'smaller', 's', but got {alternative}")
    alternative_map = {
        "two-sided": "two-sided",
        "2s": "two-sided",
        "larger": "larger",
        "l": "larger",
        "smaller": "smaller",
        "s": "smaller"
    }
    alternative = alternative_map[alternative]
    if alpha < 0 or alpha > 1:
        raise Exception(f"alpha must be between 0 and 1, but got {alpha}")
    # Validate input column
    if not isinstance(column, pd.Series):
        raise ValueError("Input column must be pd.Series")
    if len(column) < 1:
        raise ValueError("Input column must have at least one value")
    if column.isna().sum():
        raise Exception(
            f'column must not have missing values.\ncolumn have {column.isna().sum()} missing values')
    if column.unique().size != 2:
        raise Exception(
            f'column must have exactly two unique values.\ncolumn have {column.unique().size} unique values')

    value_counts = column.value_counts()
    count1 = value_counts.values[0]
    count2 = value_counts.values[1]
    nobs1 = nobs2 = column.size

    # Additional checks
    if nobs1 < 2 or nobs2 < 2:
        raise ValueError(
            "Each sample (nobs1 and nobs2) must have at least two observations")
    if count1 > nobs1 or count2 > nobs2:
        raise ValueError("Count cannot be greater than sample size")
    elif nobs1 < 30 or nobs2 < 30:
        print(colored(
            "Warning: Sample size is less than 30. Results may be unreliable.", 'red'))

    # Calculate proportions
    p1 = count1 / nobs1
    p2 = count2 / nobs2

    # Calculate standard error
    standard_error = np.sqrt(p1 * (1 - p1) / nobs1 + p2 * (1 - p2) / nobs2)

    # Calculate critical value and confidence interval bounds
    if alternative in ["two-sided", "2s"]:
        zcrit = stats.norm.ppf(1 - alpha / 2.0)
        lower = (p1 - p2) - zcrit * standard_error
        upper = (p1 - p2) + zcrit * standard_error
    elif alternative in ["larger", "l"]:
        zcrit = stats.norm.ppf(alpha)
        lower = (p1 - p2) + zcrit * standard_error
        upper = 1
    elif alternative in ["smaller", "s"]:
        zcrit = stats.norm.ppf(1 - alpha)
        lower = -1
        upper = (p1 - p2) + zcrit * standard_error

    return lower, upper


def confint_proportion_2sample_statsmodels(count1: int, nobs1: int, count2: int, nobs2: int, alpha: float = 0.05) -> tuple:
    """
    Calculate the confidence interval for the difference of two proportions only 'two-sided' alternative.

    Parameters:
    - count1: int, number of successes (1s) in the first sample
    - nobs1: int, total number of observations in the first sample
    - count2: int, number of successes (1s) in the second sample
    - nobs2: int, total number of observations in the second sample
    - alpha : float (default=0.05).
        Significance level for the confidence interval, coverage is
        ``1-alpha``.

    Returns
    -------
    - ci: tuple, confidence interval (lower, upper)
        - lower : float
            Lower confidence limit. This is -inf for the one-sided alternative
            "smaller".
        - upper : float
            Upper confidence limit. This is inf for the one-sided alternative
            "larger".
    """
    if alpha < 0 or alpha > 1:
        raise Exception(f"alpha must be between 0 and 1, but got {alpha}")
    # Additional checks
    if nobs1 < 2 or nobs2 < 2:
        raise ValueError(
            "Each sample (nobs1 and nobs2) must have at least two observations")
    if count1 > nobs1 or count2 > nobs2:
        raise ValueError("Count cannot be greater than sample size")
    elif nobs1 < 30 or nobs2 < 30:
        print(colored(
            "Warning: Sample size is less than 30. Results may be unreliable.", 'red'))

    lower, upper = stm.confint_proportions_2indep(
        count1=count1, nobs1=nobs1, count2=count2, nobs2=nobs2, alpha=alpha)
    return lower, upper


def confint_proportion_coluns_2sample_statsmodels(column: pd.Series, alpha: float = 0.05) -> tuple:
    """
    Calculate the confidence interval for the difference of two proportions only 'two-sided' alternative.

    Parameters:
    - column: pd.Series, input column with two unique values
    - alpha : float (default=0.05).
        Significance level for the confidence interval, coverage is
        ``1-alpha``.

    Returns
    -------
    - ci: tuple, confidence interval (lower, upper)
        - lower : float
            Lower confidence limit. This is -inf for the one-sided alternative
            "smaller".
        - upper : float
            Upper confidence limit. This is inf for the one-sided alternative
            "larger".
    """
    if alpha < 0 or alpha > 1:
        raise Exception(f"alpha must be between 0 and 1, but got {alpha}")
    # Validate input column
    if not isinstance(column, pd.Series):
        raise ValueError("Input column must be pd.Series")
    if len(column) < 1:
        raise ValueError("Input column must have at least one value")
    if column.isna().sum():
        raise Exception(
            f'column must not have missing values.\ncolumn have {column.isna().sum()} missing values')
    if column.unique().size != 2:
        raise Exception(
            f'column must have exactly two unique values.\ncolumn have {column.unique().size} unique values')

    value_counts = column.value_counts()
    count1 = value_counts.values[0]
    count2 = value_counts.values[1]
    nobs1 = nobs2 = column.size
    # Additional checks
    if nobs1 < 2 or nobs2 < 2:
        raise ValueError(
            "Each sample (nobs1 and nobs2) must have at least two observations")
    if count1 > nobs1 or count2 > nobs2:
        raise ValueError("Count cannot be greater than sample size")
    elif nobs1 < 30 or nobs2 < 30:
        print(colored(
            "Warning: Sample size is less than 30. Results may be unreliable.", 'red'))

    lower, upper = stm.confint_proportions_2indep(
        count1=count1, nobs1=nobs1, count2=count2, nobs2=nobs2, alpha=alpha)
    return lower, upper


def bootstrap_diff_2sample(sample1: pd.Series, sample2: pd.Series,
                           stat_func: callable = np.mean,
                           bootstrap_conf_level: float = 0.95,
                           num_boot: int = 10000,
                           alpha: float = 0.05,
                           p_value_method: str = 'normal_approx',
                           plot: bool = True,
                           return_boot_data: bool = False,
                           return_results: bool = False,
                           with_tqdm: bool = False) -> tuple:
    """
    Perform bootstrap resampling to estimate the difference of a statistic between two samples.

    Parameters:
    - sample1, sample2: pd.Series, two samples to compare
    - stat_func: callable, statistical function to apply to each bootstrap sample (default: np.mean)
    - bootstrap_conf_level: float, significance level for confidence interval (must be between 0 and 1 inclusive) (default: 0.95)
    - num_boot: int, number of bootstrap iterations (default: 10000)
    - alpha (float, optional): Significance level (default: 0.05)
    - p_value_method: str, method for calculating the p-value (default: 'normal_approx', options: 'normal_approx', 'kde')
    - plot: bool, whether to show the plot of the bootstrap distribution (default: True)
    - return_boot_data: bool, whether to return the bootstrap data (default: False)
    - return_results (bool, optional): Return (chi2, p_value, dof, expected) instead of printing (default=False).

    Returns:
    - If return_results is True
        - If return_boot_data is False: ci: tuple, confidence interval for the difference in means, p_value: float, p-value for the null hypothesis that the means are equal
        - If return_boot_data is True: boot_data: list, bootstrap estimates of the difference in means, ci: tuple, confidence interval for the difference in means, p_value: float, p-value for the null hypothesis that the means are equal
        - If plot is True: additionaly return plotly fig object 
        Default (ci, p_value, fig)
    - Else None
    """
    def human_readable_number(x):
        if x >= 1e6 or x <= -1e6:
            return f"{x/1e6:.1f}M"
        elif x >= 1e3 or x <= -1e3:
            return f"{x/1e3:.1f}k"
        else:
            return f"{x:.1f}"

    def plot_data(boot_data):
        # Create bins and histogram values using NumPy

        bins = np.linspace(boot_data.min(), boot_data.max(), 30)
        hist, bin_edges = np.histogram(boot_data, bins=bins)
        text = [
            f'{human_readable_number(bin_edges[i])} - {human_readable_number(bin_edges[i+1])}' for i in range(len(bin_edges)-1)]
        bins = [0.5 * (bin_edges[i] + bin_edges[i+1])
                for i in range(len(bin_edges)-1)]

        # Create a Plotly figure with the histogram values and bin edges
        fig = px.bar(x=bins, y=hist,
                     title="Бутстреп-распределение разницы")

        # Color the bars outside the CI orange
        ci_lower, ci_upper = ci
        colors = ["#049CB3" if x < ci_lower or x >
                  ci_upper else 'rgba(128, 60, 170, 0.9)' for x in bins]
        fig.data[0].marker.color = colors
        fig.data[0].text = text
        fig.add_vline(x=ci_lower, line_width=2, line_color="#049CB3",
                      annotation_text=f"{ci_lower:.2f}", annotation_position='top', line_dash="dash", annotation_font_color='#049CB3')
        fig.add_vline(x=ci_upper, line_width=2, line_color="#049CB3",
                      annotation_text=f"{ci_upper:.2f}", annotation_position='top', line_dash="dash", annotation_font_color='#049CB3')
        fig.update_annotations(font_size=16)
        fig.update_traces(
            hovertemplate='Количество = %{y}<br>Разница = %{text}', textposition='none')
        # Remove gap between bars and show white line
        fig.update_layout(
            width=800,
            bargap=0,
            xaxis_title="Разница",
            yaxis_title="Количество",
            title_font=dict(size=24, color="rgba(0, 0, 0, 0.5)"),
            title={'text': f'{fig.layout.title.text}'},
            # Для подписей и меток
            font=dict(size=14, family="Open Sans", color="rgba(0, 0, 0, 1)"),
            xaxis_title_font=dict(size=18, color="rgba(0, 0, 0, 0.5)"),
            yaxis_title_font=dict(size=18, color="rgba(0, 0, 0, 0.5)"),
            xaxis_tickfont=dict(size=14, color="rgba(0, 0, 0, 0.5)"),
            yaxis_tickfont=dict(size=14, color="rgba(0, 0, 0, 0.5)"),
            xaxis_linecolor="rgba(0, 0, 0, 0.5)",
            # xaxis_linewidth=2,
            yaxis_linecolor="rgba(0, 0, 0, 0.5)",
            # yaxis_linewidth=2
            margin=dict(l=50, r=50, b=50, t=70),
            hoverlabel=dict(bgcolor="white")
        )
        # Show the plot
        return fig

    # Check input types and lengths
    if not isinstance(sample1, pd.Series) or not isinstance(sample2, pd.Series):
        raise ValueError("Input samples must be pd.Series")
    warning_issued = False
    for sample in [sample1, sample2]:
        if len(sample) < 2:
            raise ValueError("Each sample must have at least two elements")
        elif len(sample) < 30:
            warning_issued = True
    if warning_issued:
        print(colored(
            "Warning: Sample size is less than 30 for one or more samples. Results may be unreliable.", 'red'))
    # Bootstrap Sampling
    boot_data = np.empty(num_boot)
    max_len = max(len(sample1), len(sample2))
    if with_tqdm:
        for i in tqdm(range(num_boot), desc="Bootstrapping"):
            samples_1 = sample1.sample(max_len, replace=True).values
            samples_2 = sample2.sample(max_len, replace=True).values
            boot_data[i] = stat_func(samples_1) - stat_func(samples_2)
    else:
        for i in range(num_boot):
            samples_1 = sample1.sample(max_len, replace=True).values
            samples_2 = sample2.sample(max_len, replace=True).values
            boot_data[i] = stat_func(samples_1) - stat_func(samples_2)        

    # Confidence Interval Calculation
    lower_bound = (1 - bootstrap_conf_level) / 2
    upper_bound = 1 - lower_bound
    ci = tuple(np.percentile(
        boot_data, [100 * lower_bound, 100 * upper_bound]))

    # P-value Calculation
    if p_value_method == 'normal_approx':
        p_value = 2 * min(
            stats.norm.cdf(0, np.mean(boot_data), np.std(boot_data)),
            stats.norm.cdf(0, -np.mean(boot_data), np.std(boot_data))
        )
    elif p_value_method == 'kde':
        kde = stats.gaussian_kde(boot_data)
        p_value = 2 * min(kde.integrate_box_1d(-np.inf, 0),
                          kde.integrate_box_1d(0, np.inf))
    else:
        raise ValueError(
            "Invalid p_value_method. Must be 'normal_approx' or 'kde'")

    if not return_results:
        print('Bootstrap resampling to estimate the difference')
        print('alpha = ', alpha)
        print('p-value = ', round(p_value, 3))
        print('ci = ', ci)
        if p_value < alpha:
            print(colored(
                "Отклоняем нулевую гипотезу, поскольку p-value меньше уровня значимости", 'red'))
        else:
            print(colored(
                "Нет оснований отвергнуть нулевую гипотезу, поскольку p-value больше или равно уровню значимости", 'green'))
        return plot_data(boot_data)
    else:
        res = []
        if return_boot_data:
            res.extend([boot_data, ci, p_value])
        else:
            res.extend([ci, p_value])
        if plot:
            res.append(plot_data(boot_data))

        return tuple(res)


def bootstrap_single_sample(sample: pd.Series,
                            stat_func: callable = np.mean,
                            bootstrap_conf_level: float = 0.95,
                            num_boot: int = 1000,
                            plot: bool = True,
                            return_boot_data: bool = False,
                            return_results: bool = False,
                            with_tqdm: bool = False) -> tuple:
    """
    Perform bootstrap resampling to estimate the variability of a statistic for a single sample.

    Parameters:
    - sample1, sample2: pd.Series, two samples to compare
    - stat_func: callable, statistical function to apply to each bootstrap sample (default: np.mean)
    - bootstrap_conf_level: float, significance level for confidence interval (must be between 0 and 1 inclusive) (default: 0.95)
    - num_boot: int, number of bootstrap iterations (default: 1000)
    - plot: bool, whether to show the plot of the bootstrap distribution (default: True)
    - return_boot_data: bool, whether to return the bootstrap data (default: False)
    - return_results (bool, optional): Return (chi2, p_value, dof, expected) instead of printing (default=False).

    Returns:
    - If return_results is True
        - If return_boot_data is False: ci: tuple, confidence interval for the difference in means
        - If return_boot_data is True: boot_data: list, bootstrap estimates of the difference in means, ci: tuple, confidence interval for the difference in means
        - If plot is True: additionaly return plotly fig object 
        Default (ci, p_value, fig)
    - Else None
    """
    def human_readable_number(x):
        if x >= 1e6 or x <= -1e6:
            return f"{x/1e6:.1f}M"
        elif x >= 1e3 or x <= -1e3:
            return f"{x/1e3:.1f}k"
        else:
            return f"{x:.1f}"

    def plot_data(boot_data):
        # Create bins and histogram values using NumPy

        bins = np.linspace(boot_data.min(), boot_data.max(), 30)
        hist, bin_edges = np.histogram(boot_data, bins=bins)
        text = [
            f'{human_readable_number(bin_edges[i])} - {human_readable_number(bin_edges[i+1])}' for i in range(len(bin_edges)-1)]
        bins = [0.5 * (bin_edges[i] + bin_edges[i+1])
                for i in range(len(bin_edges)-1)]

        # Create a Plotly figure with the histogram values and bin edges
        fig = px.bar(x=bins, y=hist, title="Bootstrap Distribution")

        # Color the bars outside the CI orange
        ci_lower, ci_upper = ci
        colors = ["#049CB3" if x < ci_lower or x >
                  ci_upper else 'rgba(128, 60, 170, 0.9)' for x in bins]
        fig.data[0].marker.color = colors
        fig.data[0].text = text
        fig.add_vline(x=ci_lower, line_width=2, line_color="#049CB3",
                      annotation_text=f"CI Lower: {ci_lower:.2f}")
        fig.add_vline(x=ci_upper, line_width=2, line_color="#049CB3",
                      annotation_text=f"CI Upper: {ci_upper:.2f}")
        fig.update_annotations(font_size=16)
        fig.update_traces(
            hovertemplate='count=%{y}<br>x=%{text}', textposition='none')
        # Remove gap between bars and show white line
        fig.update_layout(
            width=800,
            bargap=0,
            xaxis_title="",
            yaxis_title="Count",
            title_font=dict(size=24, color="rgba(0, 0, 0, 0.6)"),
            title={'text': f'<b>{fig.layout.title.text}</b>'},
            # Для подписей и меток
            font=dict(size=14, family="Open Sans", color="rgba(0, 0, 0, 1)"),
            xaxis_title_font=dict(size=18, color="rgba(0, 0, 0, 0.5)"),
            yaxis_title_font=dict(size=18, color="rgba(0, 0, 0, 0.5)"),
            xaxis_tickfont=dict(size=14, color="rgba(0, 0, 0, 0.5)"),
            yaxis_tickfont=dict(size=14, color="rgba(0, 0, 0, 0.5)"),
            xaxis_linecolor="rgba(0, 0, 0, 0.5)",
            # xaxis_linewidth=2,
            yaxis_linecolor="rgba(0, 0, 0, 0.5)",
            # yaxis_linewidth=2
            margin=dict(l=50, r=50, b=50, t=70),
            hoverlabel=dict(bgcolor="white")
        )
        # Show the plot
        return fig

    # Check input types and lengths
    if not isinstance(sample, pd.Series):
        raise ValueError("Input sample must be pd.Series")
    warning_issued = False
    if len(sample) < 2:
        raise ValueError("Each sample must have at least two elements")
    elif len(sample) < 30:
        warning_issued = True
    if warning_issued:
        print(colored(
            "Warning: Sample size is less than 30 for one or more samples. Results may be unreliable.", 'red'))
    # Bootstrap Sampling
    boot_data = np.empty(num_boot)
    max_len = len(sample)
    if with_tqdm:
        for i in tqdm(range(num_boot), desc="Bootstrapping"):
            samples_1 = sample.sample(max_len, replace=True).values
            boot_data[i] = stat_func(samples_1)
    else:
        for i in range(num_boot):
            samples_1 = sample.sample(max_len, replace=True).values
            boot_data[i] = stat_func(samples_1)        
    # Confidence Interval Calculation
    lower_bound = (1 - bootstrap_conf_level) / 2
    upper_bound = 1 - lower_bound
    ci = tuple(np.percentile(
        boot_data, [100 * lower_bound, 100 * upper_bound]))

    if not return_results:
        print('Bootstrap resampling')
        print('ci = ', ci)
        plot_data(boot_data).show()
    else:
        res = []
        if return_boot_data:
            res.extend([boot_data, ci])
        else:
            res.extend([ci])
        if plot:
            res.append(plot_data(boot_data))

        return tuple(res)

def dunn_df(df: pd.DataFrame, p_adjust: str = 'holm') -> None:
    """
    Perform a Dunn's test. The Dunn's test is a non-parametric test used to compare the medians of multiple groups.
    It is a post-hoc test used to determine which pairs of groups are significantly different after a Kruskal-Wallis test.

    Parameters:
    ----------
    df : pd.DataFrame
        DataFrame containing two columns
        - first column contains labels and
        - second column contains corresponding values
    p_adjust : str, optional
        Method for adjusting p-values for multiple comparisons (default: None). Available methods are:
        - 'holm' (Holm-Bonferroni method)
        - 'bonferroni' (Bonferroni method)
        - 'fdr_bh' (Benjamini-Hochberg method)
        - 'fdr_by' (Benjamini-Yekutieli method)
        - 'none' (no adjustment)

    Returns:
    ----------
    None
    """

    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input df must be pd.DataFrame")
    if df.shape[1] != 2:
        raise ValueError("Input DataFrame must have exactly two columns")
    labels = df.iloc[:, 0]
    values = df.iloc[:, 1]
    if not pd.api.types.is_numeric_dtype(values):
        raise ValueError("Value column must contain numeric values")
    if labels.isna().sum() or values.isna().sum():
        raise ValueError(
            f'labels and values must not have missing values.\nlabels have {labels.isna().sum()} missing values\nvalues have {values.isna().sum()} missing values')
    unique_labels = labels.unique()
    if len(unique_labels) < 2:
        raise ValueError("Labels must contain at least two unique values")

    unique_labels = df.iloc[:, 0].unique()
    warning_issued = False
    for label in unique_labels:
        sample_size = (df.iloc[:, 0] == label).sum()
        if sample_size < 2:
            raise ValueError("Each sample must have at least two elements")
        elif sample_size < 30:
            warning_issued = True
    if warning_issued:
        print(colored(
            "Warning: Sample size is less than 30 for one or more samples. Results may be unreliable.", 'red'))

    group_col, val_col = df.columns
    p_values = scikit_posthocs.posthoc_dunn(a=df, val_col=val_col, group_col=group_col, p_adjust=p_adjust)
    display(
        p_values.style.set_caption("Тест Данна")
        .format("{:.3f}")
        .set_table_styles(
            [
                {
                    "selector": "caption",
                    "props": [
                        ("font-size", "14px"),
                        ("text-align", "left"),
                        ("font-weight", "bold"),
                    ],
                }
            ]
        )
    )

def ttest_ind_report(
    df: pd.DataFrame,
    category_column: str,
    numeric_column: str,
    alpha: float = 0.05,
    alternative: str = 'two-sided',
    reference_group: str = None
) -> dict:
    """
    Perform t-test for independent samples and generate a detailed report.

    Parameters:
    ----------
    df : pd.DataFrame
        The DataFrame containing the data for analysis.
    category_column : str
        The name of the column in `df` that contains the group labels (categories).
        This column should have exactly two unique values representing the two groups to compare.
    numeric_column : str
        The name of the column in `df` that contains the numeric values to compare between the two groups.
    alpha : float, optional
        The significance level for the hypothesis tests. Default is 0.05.
    alternative : str, optional
        The alternative hypothesis for the t-test. Possible values are:
        - 'two-sided' or '2s': The means of the two groups are not equal (default).
        - 'larger' or 'l': The mean of the first group is larger than the mean of the second group.
        - 'smaller' or 's': The mean of the first group is smaller than the mean of the second group.
    reference_group : str, optional
        The label in `category_column` that defines the reference group for the alternative hypothesis.
        If not provided, the first group is determined by the order of unique values in `category_column`.

    Returns:
    -------
    None
    """
    # Check if the category column exists in the DataFrame
    if category_column not in df.columns:
        raise ValueError(f"Column '{category_column}' not found in DataFrame.")
    # Check if the numeric column exists in the DataFrame
    if numeric_column not in df.columns:
        raise ValueError(f"Column '{numeric_column}' not found in DataFrame.")
    # Ensure the numeric column contains numeric data
    if not pd.api.types.is_numeric_dtype(df[numeric_column]):
        raise ValueError(f"Column '{numeric_column}' must contain numeric data.")

    # Verify that the category column contains exactly two unique groups
    groups = df[category_column].unique()
    if len(groups) != 2:
        raise ValueError(f"Column '{category_column}' must contain exactly two unique groups.")

    # Remove missing values from the data
    df_clean = df[[category_column, numeric_column]].dropna()

    # State the null and alternative hypotheses for the t-test
    print("Шаг 1: Формулировка гипотез.")
    print("H0: Средние значения в двух группах равны.")
    print("H1: Средние значения в двух группах не равны.")

    # State the null and alternative hypotheses for the Levene's test
    print("Шаг 2: Проверка равенства дисперсий.")
    print("H0: Дисперсии в двух группах равны.")
    print("H1: Дисперсии в двух группах не равны.")
    print("Используем тест Левена для проверки равенства дисперсий.")
    print(f"Уровень значимости (alpha) установлен на {alpha}.")

    # Perform Levene's test to check for equal variances
    _, p_value_levene = pgdt.levene_df(df_clean[[category_column, numeric_column]], alpha=alpha, return_results=True)
    equal_var = p_value_levene >= alpha
    print("Шаг 3: Выбор подходящего t-теста.")
    # Determine whether to use Welch's correction based on the Levene's test result
    if not equal_var:
        print("Так как дисперсии в группах разные, будем использовать тест Уэлча.")
    else:
        print("Так как нет оснований утверждать, что дисперсии в группах отличаются, используем стандартный t-тест.")
    print(f"Уровень значимости alpha выберем {alpha}.")
    print(f"Альтернативная гипотеза: {alternative}.")

    # Check the sample size in each group
    print("Шаг 4: Проверка размеров групп.")
    display(df_clean[category_column].value_counts())

    # Perform the independent t-test (with or without Welch's correction)
    print("Шаг 5: Проведение теста.")
    pgdt.ttest_ind_df(df_clean[[category_column, numeric_column]]
                      , alpha=alpha
                      , equal_var=equal_var
                      , alternative=alternative
                      , first_for_alternative=reference_group
    )

