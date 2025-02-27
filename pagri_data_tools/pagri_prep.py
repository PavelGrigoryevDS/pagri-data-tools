# import importlib
# importlib.reload(pgdt)
import pandas as pd
import numpy as np
import plotly.express as px

# import seaborn as sns
# import matplotlib.pyplot as plt
from ipywidgets import widgets, Layout
from IPython.display import display, HTML
from tqdm.auto import tqdm
import itertools
from pymystem3 import Mystem
import io
import base64 

def count_share(series: pd.Series) -> str:
    """
    Calculate count and percentage of True values in a boolean Series.

    Parameters
    ----------
    series : pd.Series
        Boolean Series containing True/False values from a condition

    Returns
    -------
    str
        Formatted string with count and percentage in format: "count (percentage%)"
        Example: "150 (23.5%)"

    Examples
    --------
    >>> df = pd.DataFrame({'price': [100, 200, 300, 400, 500]})
    >>> count_share(df['price'] > 300)
    '2 (40.0%)'
    """    
    total_count = len(series)
    matching_count = series.sum()
    percentage = (matching_count / total_count) * 100
    return f"{matching_count} ({percentage:.1f}%)"

def pretty_value(value):
    """
    Функция делает удобное представление числа с пробелами после разрядов.
    Работает как с целыми числами, так и с числами с плавающей точкой.
    """
    if value == 0:
        return "0"

    # Определяем, отрицательное ли число
    is_negative = value < 0
    value = abs(value)

    # Разделяем целую и дробную часть
    integer_part = int(value)
    fractional_part = value - integer_part

    # Форматируем целую часть
    parts = []
    while integer_part > 0:
        parts.append(f"{integer_part % 1000:03d}")  # Добавляем последние три цифры
        integer_part //= 1000  # Убираем последние три цифры

    # Объединяем части целой части в обратном порядке
    result = ' '.join(reversed(parts)).lstrip('0')
    if fractional_part and not result:
        result = '0'
    # Форматируем дробную часть, если она не нулевая
    if fractional_part > 0:
        # Убираем ведущие нули из дробной части
        fractional_str = str(round(fractional_part, 2))[2:]  # Получаем строку после "0."
        result += f".{fractional_str}"

    return f"-{result}" if is_negative else result

def format_number(num):
    if num < 1000:
        return f"{num:.2f}"
    elif num < 1_000_000:
        return f"{num / 1000:.2f}k"
    elif num < 1_000_000_000:
        return f"{num / 1_000_000:.2f}m"
    elif num < 1_000_000_000_000:
        return f"{num / 1_000_000_000:.2f}b"
    else:
        return f"{num / 1_000_000_000_000:.2f}t"

def make_widget_all_frame(df):
    dupl = df.duplicated().sum()
    duplicates = dupl
    if duplicates == 0:
        duplicates = "---"
    else:
        duplicates = pretty_value(duplicates)
        duplicates_pct = dupl * 100 / df.shape[0]
        if 0 < duplicates_pct < 1:
            duplicates_pct = "<1"
        elif duplicates_pct > 99 and duplicates_pct < 100:
            duplicates_pct = round(duplicates_pct, 1)
            if duplicates_pct == 100:
                duplicates_pct = 99.9
        else:
            duplicates_pct = round(duplicates_pct)
        duplicates = f"{duplicates} ({duplicates_pct}%)"
    dupl_keep_false = df.duplicated(keep=False).sum()
    dupl_sub = (
        df.apply(
            lambda x: (
                x.str.lower().str.strip().str.replace(r"\s+", " ", regex=True)
                if x.dtype == "object"
                else x
            )
        )
        .duplicated(keep=False)
        .sum()
    )
    duplicates_sub_minis_origin = pretty_value(dupl_sub - dupl_keep_false)
    duplicates_sub_minis_origin_pct = (dupl_sub - dupl_keep_false) * 100 / dupl
    if 0 < duplicates_sub_minis_origin_pct < 1:
        duplicates_sub_minis_origin_pct = "<1"
    elif duplicates_sub_minis_origin_pct > 99 and duplicates_sub_minis_origin_pct < 100:
        duplicates_sub_minis_origin_pct = round(duplicates_sub_minis_origin_pct, 1)
    else:
        duplicates_sub_minis_origin_pct = round(duplicates_sub_minis_origin_pct)
    duplicates_sub_minis_origin = (
        f"{duplicates_sub_minis_origin} ({duplicates_sub_minis_origin_pct}%)"
    )
    all_rows = pd.DataFrame(
        {
            "Rows": [pretty_value(df.shape[0])],
            "Features": [df.shape[1]],
            "RAM (Mb)": [round(df.__sizeof__() / 1_048_576)],
            "Duplicates": [duplicates],
            "Dupl (sub - origin)": [duplicates_sub_minis_origin],
        }
    )
    # widget_DataFrame = widgets.Output()
    # with widget_DataFrame:
    #      display_markdown('**DataFrame**', raw=True)
    widget_all_frame = widgets.Output()
    with widget_all_frame:
        # display_html('<h4>DataFrame</h4>', raw=True, height=3)
        if pd.__version__ == "1.3.5":
            display(
                all_rows.style.set_caption("DataFrame")
                .set_table_styles(
                    [
                        {
                            "selector": "caption",
                            "props": [
                                ("font-size", "18px"),
                                ("text-align", "left"),
                                ("font-weight", "bold"),
                            ],
                        }
                    ]
                )
                .set_properties(**{"text-align": "left"})
                .hide(axis="index")
            )
        else:
            display(
                all_rows.style.set_caption("DataFrame")
                .set_table_styles(
                    [
                        {
                            "selector": "caption",
                            "props": [
                                ("font-size", "18px"),
                                ("text-align", "left"),
                                ("font-weight", "bold"),
                            ],
                        }
                    ]
                )
                .set_properties(**{"text-align": "left"})
                # .hide(axis="columns")
                .hide(axis="index")
            )
    # widget_DataFrame.layout.margin = '0px 0px 0px 0px'
    return widget_all_frame




def make_widget_range_date(column):
    column_name = column.name
    fist_date = column.min()
    last_date = column.max()
    ram = round(column.__sizeof__() / 1_048_576)
    if ram == 0:
        ram = "<1 Mb"
    column_summary = pd.DataFrame(
        {"First date": [fist_date], "Last date": [last_date], "RAM (Mb)": [ram]}
    )
    widget_summary = widgets.Output()
    with widget_summary:
        # display_html(f'<h4>{column_name}</h4>', raw=True)
        display(
            column_summary.T.reset_index()
            .style.set_caption(f"{column_name}")
            .set_table_styles(
                [
                    {
                        "selector": "caption",
                        "props": [
                            ("font-size", "16px"),
                            ("text-align", "left"),
                            ("font-weight", "bold"),
                        ],
                    }
                ]
            )
            .set_properties(**{"text-align": "left"})
            .hide(axis="columns")
            .hide(axis="index")
        )
    return widget_summary

def make_widget_summary_date(column):
    column_name = column.name
    zeros = ((column == 0) | (column == "")).sum()
    if zeros == 0:
        zeros = "---"
    else:
        zeros = pretty_value(((column == 0) | (column == "")).sum())
        zeros_pct = round(((column == 0) | (column == "")).sum() * 100 / column.size)
        if zeros_pct == 0:
            zeros_pct = "<1"
        zeros = f"{zeros} ({zeros_pct}%)"
    missing = column.isna().sum()
    if missing == 0:
        missing = "---"
    else:
        missing = pretty_value(column.isna().sum())
        missing_pct = round(column.isna().sum() * 100 / column.size)
        if missing_pct == 0:
            missing_pct = "<1"
        missing = f"{missing} ({missing_pct}%)"
    distinct = pretty_value(column.nunique())
    distinct_pct = column.nunique() * 100 / column.size
    if distinct_pct > 99 and distinct_pct < 100:
        distinct_pct = round(distinct_pct, 1)
        if distinct_pct == 100:
            distinct_pct = 99.9
    else:
        distinct_pct = round(distinct_pct)
    if distinct_pct == 0:
        distinct_pct = "<1"
    distinct = f"{distinct} ({distinct_pct}%)"
    duplicates = column.duplicated().sum()
    if duplicates == 0:
        duplicates = "---"
    else:
        duplicates = pretty_value(duplicates)
        duplicates_pct = column.duplicated().sum() * 100 / column.size
        if 0 < duplicates_pct < 1:
            duplicates_pct = "<1"
        elif duplicates_pct > 99 and duplicates_pct < 100:
            duplicates_pct = round(duplicates_pct, 1)
            if duplicates_pct == 100:
                duplicates_pct = 99.9
        else:
            duplicates_pct = round(duplicates_pct)
        duplicates = f"{duplicates} ({duplicates_pct}%)"
    column_summary = pd.DataFrame(
        {
            "Zeros": [zeros],
            "Missing": [missing],
            "Distinct": [distinct],
            "Duplicates": [duplicates],
        }
    )
    widget_summary = widgets.Output()
    with widget_summary:
        # display_html(f'<h4>{column_name}</h4>', raw=True)
        display(
            column_summary.T.reset_index()
            .style
            # .set_caption(f'{column_name}')
            .set_table_styles(
                [
                    {
                        "selector": "caption",
                        "props": [
                            ("font-size", "16px"),
                            ("text-align", "left"),
                            ("font-weight", "bold"),
                        ],
                    }
                ]
            )
            .set_properties(**{"text-align": "left"})
            .hide(axis="columns")
            .hide(axis="index")
        )
    return widget_summary




def make_widget_check_missing_date(column):
    column_name = column.name
    fist_date = column.min()
    last_date = column.max()
    date_range = pd.date_range(start=fist_date, end=last_date, freq="D")
    years = date_range.year.unique()
    years_missed_pct = (~years.isin(column.dt.year.unique())).sum() * 100 / years.size
    if 0 < years_missed_pct < 1:
        years_missed_pct = "<1"
    elif years_missed_pct > 99:
        years_missed_pct = round(years_missed_pct, 1)
    else:
        years_missed_pct = round(years_missed_pct)
    months = date_range.to_period("M").unique()
    months_missed_pct = (
        (~months.isin(column.dt.to_period("M").unique())).sum() * 100 / months.size
    )
    if 0 < months_missed_pct < 1:
        months_missed_pct = "<1"
    elif months_missed_pct > 99:
        months_missed_pct = round(months_missed_pct, 1)
    else:
        months_missed_pct = round(months_missed_pct)
    weeks = date_range.to_period("W").unique()
    weeks_missed_pct = (
        (~weeks.isin(column.dt.to_period("W").unique())).sum() * 100 / weeks.size
    )
    if 0 < weeks_missed_pct < 1:
        weeks_missed_pct = "<1"
    elif weeks_missed_pct > 99:
        weeks_missed_pct = round(weeks_missed_pct, 1)
    else:
        weeks_missed_pct = round(weeks_missed_pct)
    days = date_range.unique().to_period("D")
    days_missed_pct = (
        (~days.isin(column.dt.to_period("D").unique())).sum() * 100 / days.size
    )
    if 0 < days_missed_pct < 1:
        days_missed_pct = "<1"
    elif days_missed_pct > 99:
        days_missed_pct = round(days_missed_pct, 1)
    else:
        days_missed_pct = round(days_missed_pct)

    column_summary = pd.DataFrame(
        {
            "Years missing": [f"{years_missed_pct}%"],
            "Months missing": [f"{months_missed_pct}%"],
            "Weeks missing": [f"{weeks_missed_pct}%"],
            "Days missing": [f"{days_missed_pct}%"],
        }
    )
    widget_summary = widgets.Output()
    with widget_summary:
        # display_html(f'<h4>{column_name}</h4>', raw=True)
        display(
            column_summary.T.reset_index()
            .style
            # .set_caption(f'{column_name}')
            .set_table_styles(
                [
                    {
                        "selector": "caption",
                        "props": [
                            ("font-size", "16px"),
                            ("text-align", "left"),
                            ("font-weight", "bold"),
                        ],
                    }
                ]
            )
            .set_properties(**{"text-align": "left"})
            .hide(axis="columns")
            .hide(axis="index")
            # .format('{:.0f}', subset=0)
        )
    return widget_summary


def make_widget_summary(column):
    column_name = column.name
    values = column.count()
    values = pretty_value(column.count())
    values_pct = column.count() * 100 / column.size
    if 0 < values_pct < 1:
        values_pct = "<1"
    elif values_pct > 99 and values_pct < 100:
        values_pct = round(values_pct, 1)
        if values_pct == 100:
            values_pct = 99.9
    else:
        values_pct = round(values_pct)
    values = f"{values} ({values_pct}%)"
    missing = column.isna().sum()
    if missing == 0:
        missing = "---"
    else:
        missing = pretty_value(column.isna().sum())
        missing_pct = round(column.isna().sum() * 100 / column.size)
        if missing_pct == 0:
            missing_pct = "<1"
        missing = f"{missing} ({missing_pct}%)"
    distinct = pretty_value(column.nunique())
    distinct_pct = column.nunique() * 100 / column.size
    if distinct_pct > 99 and distinct_pct < 100:
        distinct_pct = round(distinct_pct, 1)
        if distinct_pct == 100:
            distinct_pct = 99.9
    else:
        distinct_pct = round(distinct_pct)
    if distinct_pct == 0:
        distinct_pct = "<1"
    distinct = f"{distinct} ({distinct_pct}%)"
    zeros = ((column == 0) | (column == "")).sum()
    if zeros == 0:
        zeros = "---"
    else:
        zeros = pretty_value(((column == 0) | (column == "")).sum())
        zeros_pct = round(((column == 0) | (column == "")).sum() * 100 / column.size)
        if zeros_pct == 0:
            zeros_pct = "<1"
        zeros = f"{zeros} ({zeros_pct}%)"
    negative = (column < 0).sum()
    if negative == 0:
        negative = "---"
    else:
        negative = pretty_value(negative)
        negative_pct = round((column < 0).sum() * 100 / column.size)
        if negative_pct == 0:
            negative_pct = "<1"
        negative = f"{negative} ({negative_pct}%)"
    duplicates = column.duplicated().sum()
    if duplicates == 0:
        duplicates = "---"
    else:
        duplicates = pretty_value(duplicates)
        duplicates_pct = column.duplicated().sum() * 100 / column.size
        if 0 < duplicates_pct < 1:
            duplicates_pct = "<1"
        elif duplicates_pct > 99 and duplicates_pct < 100:
            duplicates_pct = round(duplicates_pct, 1)
            if duplicates_pct == 100:
                duplicates_pct = 99.9
        else:
            duplicates_pct = round(duplicates_pct)
        duplicates = f"{duplicates} ({duplicates_pct}%)"
    ram = round(column.__sizeof__() / 1_048_576)
    if ram == 0:
        ram = "<1 Mb"
    column_summary = pd.DataFrame(
        {
            "Values": [values],
            "Missing": [missing],
            "Distinct": [distinct],
            "Duplicates": [duplicates],
            "Zeros": [zeros],
            "Negative": [negative],
            "RAM (Mb)": [ram],
        }
    )
    widget_summary = widgets.Output()
    with widget_summary:
        # display_html(f'<h4>{column_name}</h4>', raw=True)
        display(
            column_summary.T.reset_index()
            .style.set_caption(f"{column_name}")
            .set_table_styles(
                [
                    {
                        "selector": "caption",
                        "props": [
                            ("font-size", "16px"),
                            ("text-align", "left"),
                            ("font-weight", "bold"),
                        ],
                    }
                ]
            )
            .set_properties(**{"text-align": "left"})
            .hide(axis="columns")
            .hide(axis="index")
        )
    return widget_summary





def make_widget_pct(column):
    max_ = pretty_value(column.max())
    q_95 = pretty_value(column.quantile(0.95))
    q_75 = pretty_value(column.quantile(0.75))
    median_ = pretty_value(column.median())
    q_25 = pretty_value(column.quantile(0.25))
    q_5 = pretty_value(column.quantile(0.05))
    min_ = pretty_value(column.min())
    column_summary = pd.DataFrame(
        {
            "Max": [max_],
            "95%": [q_95],
            "75%": [q_75],
            "Median": [median_],
            "25%": [q_25],
            "5%": [q_5],
            "Min": [min_],
        }
    )
    widget_pct = widgets.Output()
    with widget_pct:
        display(
            column_summary.T.reset_index()
            .style.set_caption(f"")
            .set_table_styles(
                [{"selector": "caption", "props": [("font-size", "15px")]}]
            )
            .set_properties(**{"text-align": "left"})
            .hide(axis="columns")
            .hide(axis="index")
        )
    return widget_pct




def make_widget_std(column):
    avg_ = pretty_value(column.mean())
    mode_ = column.mode()
    if mode_.size > 1:
        mode_ = "---"
    else:
        mode_ = pretty_value(mode_.iloc[0])
    range_ = pretty_value(column.max() - column.min())
    iQR = pretty_value(column.quantile(0.75) - column.quantile(0.25))
    std = pretty_value(column.std())
    kurt = f"{column.kurtosis():.2f}"
    skew = f"{column.skew():.2f}"
    column_summary = pd.DataFrame(
        {
            "Avg": [avg_],
            "Mode": [mode_],
            "Range": [range_],
            "iQR": [iQR],
            "std": [std],
            "kurt": [kurt],
            "skew": [skew],
        }
    )
    widget_std = widgets.Output()
    with widget_std:
        display(
            column_summary.T.reset_index()
            .style.set_caption(f"")
            .set_table_styles(
                [{"selector": "caption", "props": [("font-size", "15px")]}]
            )
            .set_properties(**{"text-align": "left"})
            .hide(axis="columns")
            .hide(axis="index")
        )
    return widget_std






def make_widget_value_counts(column):
    column_name = column.name
    val_cnt = column.value_counts().iloc[:7]
    val_cnt_norm = column.value_counts(normalize=True).iloc[:7]
    column_name_pct = column_name + "_pct"
    val_cnt_norm.name = column_name_pct

    def make_value_counts_row(x):
        if x[column_name_pct] < 0.01:
            pct_str = "<1%"
        else:
            pct_str = f"({x[column_name_pct]:.0%})"
        return f"{x[column_name]:.0f} {pct_str}"

    top_5 = (
        pd.concat([val_cnt, val_cnt_norm], axis=1)
        .reset_index()
        .apply(make_value_counts_row, axis=1)
        .to_frame()
    )
    widget_value_counts = widgets.Output()
    with widget_value_counts:
        display(
            top_5.style
            # .set_caption(f'Value counts top')
            .set_table_styles(
                [
                    {
                        "selector": "caption",
                        "props": [("font-size", "16px"), ("text-align", "left")],
                    }
                ]
            )
            .set_properties(**{"text-align": "left"})
            .hide(axis="columns")
            .hide(axis="index")
        )
    # widget_value_counts.
    return widget_value_counts


def num_trunc(x):
    """
    Функция обрезает порядок числа и доабвляет К или М
    """
    if abs(x) < 1e3:
        if round(abs(x), 2) > 0:
            return f"{x:.2f}"
        else:
            return "0"
    if abs(x) < 1e6:
        return f"{x / 1e3:.0f}K"
    if abs(x) < 1e9:
        return f"{x / 1e6:.0f}M"


# def make_widget_hist(column):
#     fig, ax = plt.subplots(figsize=(3.2, 2.2))
#     sns.histplot(column, bins=20,  stat='percent', ax=ax, color='#9370db')

#     ax.set(ylabel='', xlabel='')
#     ax.locator_params(axis='x', nbins=5)
#     bins = ax.get_xticks()
#     vect = np.vectorize(num_trunc)
#     bins = bins[(bins >= column.min()) & (bins <= column.max())]
#     ax.set_xticks(ticks=bins, labels=vect(bins))
#     plt.xticks(alpha=.9)
#     plt.yticks(alpha=.9)
#     plt.gca().spines['top'].set_alpha(0.3)
#     plt.gca().spines['left'].set_alpha(0.3)
#     plt.gca().spines['right'].set_alpha(0.3)
#     plt.gca().spines['bottom'].set_alpha(0.3)
#     plt.close()
#     widget_hist = widgets.Output()
#     # fig.tight_layout()
#     with widget_hist:
#         display(fig)
#     return widget_hist


def make_widget_hist_plotly(column):
    fig = px.histogram(
        column,
        nbins=20,
        histnorm="percent",
        template="simple_white",
        height=220,
        width=350,
    )
    fig.update_traces(
        marker_color="rgba(128, 60, 170, 0.9)",
        text=f"*",
        textfont=dict(color="rgba(128, 60, 170, 0.9)"),
    )
    fig.update_layout(
        margin=dict(l=0, r=10, b=0, t=10),
        showlegend=False,
        hoverlabel=dict(
            bgcolor="white",
        ),
        xaxis_title="",
        yaxis_title="",
    )
    # fig.layout.yaxis.visible = False
    widget_hist = widgets.Output()
    with widget_hist:
        fig.show(config=dict(displayModeBar=False), renderer="png")
    return widget_hist




def make_widget_violine_plotly(column):
    fig = px.violin(column, template="simple_white", height=220, width=300)
    fig.update_traces(marker_color="rgba(128, 60, 170, 0.9)")
    fig.update_layout(
        margin=dict(l=20, r=20, b=0, t=10),
        showlegend=False,
        hoverlabel=dict(
            bgcolor="white",
        ),
        xaxis_title="",
        yaxis_title="",
        xaxis=dict(ticks="", showticklabels=False),
    )
    # fig.layout.yaxis.visible = False
    widget_hist = widgets.Output()
    with widget_hist:
        fig.show(config=dict(displayModeBar=False), renderer="png")
    return widget_hist


def make_widget_summary_obj(column):
    column_name = column.name
    values = column.count()
    values = pretty_value(column.count())
    values_pct = column.count() * 100 / column.size
    if 0 < values_pct < 1:
        values_pct = "<1"
    elif values_pct > 99 and values_pct < 100:
        values_pct = round(values_pct, 1)
        if values_pct == 100:
            values_pct = 99.9
    else:
        values_pct = round(values_pct)
    values = f"{values} ({values_pct}%)"
    missing = column.isna().sum()
    if missing == 0:
        missing = "---"
    else:
        missing = pretty_value(column.isna().sum())
        missing_pct = round(column.isna().sum() * 100 / column.size)
        if missing_pct == 0:
            missing_pct = "<1"
        missing = f"{missing} ({missing_pct}%)"
    distinct = pretty_value(column.nunique())
    distinct_pct = column.nunique() * 100 / column.size
    if distinct_pct > 99 and distinct_pct < 100:
        distinct_pct = round(distinct_pct, 1)
        if distinct_pct == 100:
            distinct_pct = 99.9
    else:
        distinct_pct = round(distinct_pct)
    if distinct_pct == 0:
        distinct_pct = "<1"
    distinct = f"{distinct} ({distinct_pct}%)"
    zeros = ((column == 0) | (column == "")).sum()
    if zeros == 0:
        zeros = "---"
    else:
        zeros = pretty_value(((column == 0) | (column == "")).sum())
        zeros_pct = round(((column == 0) | (column == "")).sum() * 100 / column.size)
        if zeros_pct == 0:
            zeros_pct = "<1"
        zeros = f"{zeros} ({zeros_pct}%)"
    duplicates = column.duplicated().sum()
    if duplicates == 0:
        duplicates = "---"
        duplicates_sub_minis_origin = "---"
    else:
        duplicates = pretty_value(duplicates)
        duplicates_pct = column.duplicated().sum() * 100 / column.size
        if 0 < duplicates_pct < 1:
            duplicates_pct = "<1"
        elif duplicates_pct > 99 and duplicates_pct < 100:
            duplicates_pct = round(duplicates_pct, 1)
            if duplicates_pct == 100:
                duplicates_pct = 99.9
        else:
            duplicates_pct = round(duplicates_pct)
        duplicates = f"{duplicates} ({duplicates_pct}%)"
        duplicates_keep_false = column.duplicated(keep=False).sum()
        duplicates_sub = (
            column.str.lower()
            .str.strip()
            .str.replace(r"\s+", " ", regex=True)
            .duplicated(keep=False)
            .sum()
        )
        duplicates_sub_minis_origin = duplicates_sub - duplicates_keep_false
        if duplicates_sub_minis_origin == 0:
            duplicates_sub_minis_origin = "---"
        else:
            duplicates_sub_minis_origin = pretty_value(duplicates_sub_minis_origin)
            duplicates_sub_minis_origin_pct = (
                (duplicates_sub - duplicates_keep_false) * 100 / duplicates_sub
            )
            if 0 < duplicates_sub_minis_origin_pct < 1:
                duplicates_sub_minis_origin_pct = "<1"
            elif (
                duplicates_sub_minis_origin_pct > 99
                and duplicates_sub_minis_origin_pct < 100
            ) or duplicates_sub_minis_origin_pct < 1:
                duplicates_sub_minis_origin_pct = round(
                    duplicates_sub_minis_origin_pct, 1
                )
            else:
                duplicates_sub_minis_origin_pct = round(duplicates_sub_minis_origin_pct)
            duplicates_sub_minis_origin = (
                f"{duplicates_sub_minis_origin} ({duplicates_sub_minis_origin_pct}%)"
            )

    ram = round(column.__sizeof__() / 1_048_576)
    if ram == 0:
        ram = "<1 Mb"
    column_summary = pd.DataFrame(
        {
            "Values": [values],
            "Missing": [missing],
            "Distinct": [distinct],
            "Duplicated origin": [duplicates],
            "Dupl (modify - origin)": [duplicates_sub_minis_origin],
            "Empty": [zeros],
            "RAM (Mb)": [ram],
        }
    )
    widget_summary_obj = widgets.Output()
    with widget_summary_obj:
        # display_html(f'<h4>{column_name}</h4>', raw=True)
        display(
            column_summary.T.reset_index()
            .style.set_caption(f"{column_name}")
            .set_table_styles(
                [
                    {
                        "selector": "caption",
                        "props": [
                            ("font-size", "16px"),
                            ("text-align", "left"),
                            ("font-weight", "bold"),
                        ],
                    }
                ]
            )
            .set_properties(**{"text-align": "left"})
            .hide(axis="columns")
            .hide(axis="index")
        )
    return widget_summary_obj



def make_widget_value_counts_obj(column):
    column_name = column.name
    val_cnt = column.value_counts().iloc[:8]
    val_cnt_norm = column.value_counts(normalize=True).iloc[:8]
    column_name_pct = column_name + "_pct"
    val_cnt_norm.name = column_name_pct

    def make_value_counts_row(x):
        if x[column_name_pct] < 0.01:
            pct_str = "<1%"
        else:
            pct_str = f"({x[column_name_pct]:.0%})"
        return f"{x[column_name]} {pct_str}"

    top_5 = (
        pd.concat([val_cnt, val_cnt_norm], axis=1)
        .reset_index()
        .apply(make_value_counts_row, axis=1)
        .to_frame()
    )
    widget_value_counts_obj = widgets.Output()
    with widget_value_counts_obj:
        display(
            top_5.style
            # .set_caption(f'Value counts top')
            .set_table_styles(
                [
                    {
                        "selector": "caption",
                        "props": [("font-size", "16px"), ("text-align", "left")],
                    }
                ]
            )
            .set_properties(**{"text-align": "left"})
            .hide(axis="columns")
            .hide(axis="index")
        )
    # widget_value_counts.
    return widget_value_counts_obj





def make_widget_bar_obj(column):
    df_fig = column.value_counts(ascending=True).iloc[-10:]
    text_labels = [label[:30] for label in df_fig.index.to_list()]
    fig = px.bar(
        df_fig, orientation="h", template="simple_white", height=220, width=350
    )
    fig.update_traces(marker_color="rgba(128, 60, 170, 0.8)")
    fig.update_layout(
        margin=dict(l=0, r=0, b=0, t=5),
        showlegend=False,
        hoverlabel=dict(
            bgcolor="white",
        ),
        xaxis_title="",
        yaxis_title="",
        font=dict(size=13, family="Segoe UI", color="rgba(0, 0, 0, 0.8)"),
        xaxis_tickfont=dict(size=13, color="rgba(0, 0, 0, 0.8)"),
        yaxis_tickfont=dict(size=13, color="rgba(0, 0, 0, 0.8)"),
        xaxis_linecolor="rgba(0, 0, 0, 0.5)",
        yaxis_linecolor="rgba(0, 0, 0, 0.5)",
        xaxis_tickcolor="rgba(0, 0, 0, 0.5)",
        yaxis_tickcolor="rgba(0, 0, 0, 0.5)",
    )
    fig.update_traces(y=text_labels)
    widget_bar_obj = widgets.Output()
    # fig.tight_layout()
    with widget_bar_obj:
        fig.show(config=dict(displayModeBar=False), renderer="png")
    return widget_bar_obj




def make_hbox(widgets_: list):
    # add some CSS styles to distribute free space
    hbox_layout = Layout(
        display="flex",
        flex_flow="row",
        # justify_content='space-around',
        width="auto",
        grid_gap="20px",
        align_items="flex-end",
    )
    # create Horisontal Box container
    hbox = widgets.HBox(widgets_, layout=hbox_layout)
    return hbox


def my_info_old(df, graphs=True, num=True, obj=True, date=True):
    """
    Show information about a pandas DataFrame.

    This function provides a comprehensive overview of a DataFrame, including:
    - Value counts for the fourth column (before graphs)
    - Summary statistics and visualizations for numeric, object, and date columns

    Parameters:
    df: pandas.DataFrame
        The input DataFrame containing the data
    graphs: bool, default True
        If True, visualizations are displayed
    num: bool, default True
        If True, summary statistics and visualizations are generated for numeric columns
    obj: bool, default True
        If True, summary statistics and visualizations are generated for object columns
    date: bool, default True
        If True, summary statistics and visualizations are generated for date columns
    Return:
        None
    """
    if not num and not obj and not date:
        return
    vbox_layout = Layout(
        display="flex",
        # flex_flow='column',
        justify_content="space-around",
        # width='auto',
        # grid_gap = '20px',
        # align_items = 'flex-end'
    )
    funcs_num = [
        make_widget_summary,
        make_widget_pct,
        make_widget_std,
        make_widget_value_counts,
    ]
    func_obj = [make_widget_summary_obj, make_widget_value_counts_obj]
    func_date = [
        make_widget_range_date,
        make_widget_summary_date,
        make_widget_check_missing_date,
    ]
    if graphs:
        funcs_num += [make_widget_hist_plotly, make_widget_violine_plotly]
        func_obj += [make_widget_bar_obj]
    boxes = []
    if date:
        date_columns = filter(
            lambda x: pd.api.types.is_datetime64_any_dtype(df[x]), df.columns
        )
        for column in tqdm(date_columns):
            widgets_ = [func(df[column]) for func in func_date]
            boxes.extend((widgets_))
        layout = widgets.Layout(grid_template_columns="1fr 1fr 1fr 1fr 1fr")
        date_grid = widgets.GridBox(boxes, layout=layout)
    boxes = []
    if num:
        num_columns = filter(lambda x: pd.api.types.is_numeric_dtype(df[x]), df.columns)
        for column in tqdm(num_columns):
            widgets_ = [func(df[column]) for func in funcs_num]
            boxes.extend((widgets_))
        if graphs:
            layout = widgets.Layout(
                grid_template_columns="auto auto auto auto auto auto"
            )
        else:
            layout = widgets.Layout(grid_template_columns="repeat(4, 0.2fr)")
        num_grid = widgets.GridBox(boxes, layout=layout)
    boxes = []
    if obj:
        obj_columns = filter(
            lambda x: not pd.api.types.is_numeric_dtype(df[x])
            and not pd.api.types.is_datetime64_any_dtype(df[x]),
            df.columns,
        )
        for column in tqdm(obj_columns):
            widgets_ = [func(df[column]) for func in func_obj]
            boxes.extend((widgets_))
        if graphs:
            layout = widgets.Layout(grid_template_columns="auto auto auto")
        else:
            layout = widgets.Layout(grid_template_columns="repeat(2, 0.3fr)")
        obj_grid = widgets.GridBox(boxes, layout=layout)

    # widgets.Layout(grid_template_columns="200px 200px 200px 200px 200px 200px")))
    display(make_widget_all_frame(df))
    if date:
        display(date_grid)
    if num:
        display(num_grid)
    if obj:
        display(obj_grid)


def my_info_widget_gen(df, graphs=True, num=True, obj=True, date=True):
    """
    Generates information about a pandas DataFrame.

    This function provides a comprehensive overview of a DataFrame, including:
    - Value counts for the fourth column (before graphs)
    - Summary statistics and visualizations for numeric, object, and date columns

    Parameters:
    df: pandas.DataFrame
        The input DataFrame containing the data
    graphs: bool, default True
        If True, visualizations are displayed
    num: bool, default True
        If True, summary statistics and visualizations are generated for numeric columns
    obj: bool, default True
        If True, summary statistics and visualizations are generated for object columns
    date: bool, default True
        If True, summary statistics and visualizations are generated for date columns
    Yields:
    A generator of widgets and visualizations for the input DataFrame
    """
    if not num and not obj and not date:
        return
    vbox_layout = Layout(
        display="flex",
        # flex_flow='column',
        justify_content="space-around",
        # width='auto',
        # grid_gap = '20px',
        # align_items = 'flex-end'
    )
    yield make_widget_all_frame(df)

    funcs_num = [
        make_widget_summary,
        make_widget_pct,
        make_widget_std,
        make_widget_value_counts,
    ]
    func_obj = [make_widget_summary_obj, make_widget_value_counts_obj]
    func_date = [
        make_widget_range_date,
        make_widget_summary_date,
        make_widget_check_missing_date,
    ]
    if graphs:
        funcs_num += [make_widget_hist_plotly, make_widget_violine_plotly]
        func_obj += [make_widget_bar_obj]
    if date:
        date_columns = filter(
            lambda x: pd.api.types.is_datetime64_any_dtype(df[x]), df.columns
        )
        layout = widgets.Layout(grid_template_columns="auto auto")
        for column in date_columns:
            widgets_ = [func(df[column]) for func in func_date]
            yield widgets.GridBox(widgets_, layout=layout)

    if num:
        num_columns = filter(lambda x: pd.api.types.is_numeric_dtype(df[x]), df.columns)
        if graphs:
            layout = widgets.Layout(
                grid_template_columns="auto auto auto auto auto auto"
            )
        else:
            layout = widgets.Layout(grid_template_columns="repeat(4, 0.2fr)")
        for column in num_columns:
            widgets_ = [func(df[column]) for func in funcs_num]
            yield widgets.GridBox(widgets_, layout=layout)

    if obj:
        obj_columns = filter(
            lambda x: not pd.api.types.is_numeric_dtype(df[x])
            and not pd.api.types.is_datetime64_any_dtype(df[x]),
            df.columns,
        )
        if graphs:
            layout = widgets.Layout(grid_template_columns="auto auto auto")
        else:
            layout = widgets.Layout(grid_template_columns="repeat(2, 0.3fr)")
        for column in obj_columns:
            widgets_ = [func(df[column]) for func in func_obj]
            yield widgets.GridBox(widgets_, layout=layout)

class info_gen:
    def __init__(self, df, column = None, graphs=True, num=True, obj=True, date=True,  mode='gen'):
        '''
        mode: 'gen' - генератор, 'column' - столбец
        '''
        if mode == 'column' and not column:
            raise ValueError("Для режима 'column' необходимо указать столбец")
        self.hist_mode = 'simple'  # Значение по умолчанию она гистограама без обрезания по квантилям
        self.only_summary = False
        self.graphs=graphs
        if mode == 'gen':
            self.gen = self.info_gen(df, graphs, num, obj, date)  # Создаем генератор
        else:
            self.gen = self.info_column(df, column, graphs)  # Создаем генератор
    def next(self, hist_mode=None, only_summary=None):
        """Возвращает следующее значение из генератора.
        
        Если new_mode равно 'dual', меняет текущий режим на 'dual' (2 гистограмы, вторая обрезанная по квантилям)
        Если new_mode равен None, возвращает к значению по умолчанию 'simple'.
        """
        if hist_mode:
            if hist_mode not in ['simple', 'dual']:
                return "Invalid hist_mode. hist_mode must be 'simple' or 'dual'"  
            self.hist_mode = hist_mode  # Меняем режим
        else:
            self.hist_mode = 'simple'  # Возвращаем к значению по умолчанию
        if only_summary==True:
            self.only_summary = True
        else:
            self.only_summary = False
        try:
            return next(self.gen)  # Получаем следующее значение из генератора
        except StopIteration:
            return "Generator has finished"  # Генератор закончился, возвращаем None
        

    def info_gen(self, df, graphs=True, num=True, obj=True, date=True):
        if not num and not obj and not date:
            return
        yield self.make_all_frame_for_html(df)
        funcs_num = [
            self.make_summary_for_html,
            self.make_pct_for_html,
            self.make_std_for_html,
            self.make_value_counts_for_html,
        ]
        funcs_obj = [self.make_summary_obj_for_html, self.make_value_counts_obj_for_html]
        funcs_date = [
            self.make_range_date_for_html,
            self.make_summary_date_for_html,
            self.make_check_missing_date_for_html,
        ]
        if graphs:
            funcs_num += [self.make_hist_plotly_for_html]
            funcs_obj += [self.make_bar_obj_for_html]
            funcs_date += [None]  
        else:
            funcs_num += [None]
            funcs_obj += [None]
            funcs_date += [None]              
        if date:
            date_columns = filter(
                lambda x: pd.api.types.is_datetime64_any_dtype(df[x]), df.columns
            )
            for column in date_columns:
                row_for_html = self.make_row_for_html(df, column, funcs_date)
                # Отображение HTML-кода
                display(HTML(row_for_html))
                yield
        if num:
            num_columns = filter(lambda x: pd.api.types.is_numeric_dtype(df[x]), df.columns)
            for column in num_columns:
                row_for_html = self.make_row_for_html(df, column, funcs_num)
                # Отображение HTML-кода
                display(HTML(row_for_html))
                yield

        if obj:
            obj_columns = filter(
                lambda x: not pd.api.types.is_numeric_dtype(df[x])
                and not pd.api.types.is_datetime64_any_dtype(df[x]),
                df.columns,
            )
            for column in obj_columns:
                row_for_html = self.make_row_for_html(df, column, funcs_obj)
                # Отображение HTML-кода
                display(HTML(row_for_html))
                yield        
        
    def info_column(self, df, column, graphs=True):

        funcs_num = [
            self.make_summary_for_html,
            self.make_pct_for_html,
            self.make_std_for_html,
            self.make_value_counts_for_html,
        ]
        funcs_obj = [self.make_summary_obj_for_html, self.make_value_counts_obj_for_html]
        funcs_date = [
            self.make_range_date_for_html,
            self.make_summary_date_for_html,
            self.make_check_missing_date_for_html,
        ]
        if graphs:
            funcs_num += [self.make_hist_plotly_for_html]
            funcs_obj += [self.make_bar_obj_for_html]
            funcs_date += [None]
        if pd.api.types.is_datetime64_any_dtype(df[column]):
            row_for_html = self.make_row_for_html(df, column, funcs_date)
            # Отображение HTML-кода
            display(HTML(row_for_html))
            yield
        if pd.api.types.is_numeric_dtype(df[column]):
            row_for_html = self.make_row_for_html(df, column, funcs_num)
            # Отображение HTML-кода
            display(HTML(row_for_html))
            yield

        if not pd.api.types.is_numeric_dtype(df[column]) and not pd.api.types.is_datetime64_any_dtype(df[column]):
            row_for_html = self.make_row_for_html(df, column, funcs_obj)
            # Отображение HTML-кода
            display(HTML(row_for_html))
            yield                    
        
    def pretty_value(value):
        """
        Функция делает удобное представление числа с пробелами после разрядов.
        Работает как с целыми числами, так и с числами с плавающей точкой.
        """
        if value == 0:
            return "0"

        # Определяем, отрицательное ли число
        is_negative = value < 0
        value = abs(value)

        # Разделяем целую и дробную часть
        integer_part = int(value)
        fractional_part = value - integer_part

        # Форматируем целую часть
        parts = []
        while integer_part > 0:
            parts.append(f"{integer_part % 1000:03d}")  # Добавляем последние три цифры
            integer_part //= 1000  # Убираем последние три цифры

        # Объединяем части целой части в обратном порядке
        result = ' '.join(reversed(parts)).lstrip('0')
        if fractional_part and not result:
            result = '0'
        # Форматируем дробную часть, если она не нулевая
        if fractional_part > 0:
            # Убираем ведущие нули из дробной части
            fractional_str = str(round(fractional_part, 2))[2:]  # Получаем строку после "0."
            result += f".{fractional_str}"

        return f"-{result}" if is_negative else result        

    def format_number(num):
        if num < 1000:
            return f"{num:.2f}"
        elif num < 1_000_000:
            return f"{num / 1000:.2f}k"
        elif num < 1_000_000_000:
            return f"{num / 1_000_000:.2f}m"
        elif num < 1_000_000_000_000:
            return f"{num / 1_000_000_000:.2f}b"
        else:
            return f"{num / 1_000_000_000_000:.2f}t"    
        
    def make_all_frame_for_html(self, df):
        dupl = df.duplicated().sum()
        duplicates = dupl
        if duplicates == 0:
            duplicates = "---"
            duplicates_sub_minis_origin = "---"
        else:
            duplicates = format_number(duplicates)
            duplicates_pct = dupl * 100 / df.shape[0]
            if 0 < duplicates_pct < 1:
                duplicates_pct = "<1"
            elif duplicates_pct > 99 and duplicates_pct < 100:
                duplicates_pct = round(duplicates_pct, 1)
                if duplicates_pct == 100:
                    duplicates_pct = 99.9
            else:
                duplicates_pct = round(duplicates_pct)
            duplicates = f"{duplicates} ({duplicates_pct}%)"
            dupl_keep_false = df.duplicated(keep=False).sum()
            dupl_sub = (
                df.apply(
                    lambda x: (
                        x.str.lower().str.strip().str.replace(r"\s+", " ", regex=True)
                        if x.dtype == "object"
                        else x
                    )
                )
                .duplicated(keep=False)
                .sum()
            )
            duplicates_sub_minis_origin = format_number(dupl_sub - dupl_keep_false)
            duplicates_sub_minis_origin_pct = (dupl_sub - dupl_keep_false) * 100 / dupl
            if 0 < duplicates_sub_minis_origin_pct < 1:
                duplicates_sub_minis_origin_pct = "<1"
            elif duplicates_sub_minis_origin_pct > 99 and duplicates_sub_minis_origin_pct < 100:
                duplicates_sub_minis_origin_pct = round(duplicates_sub_minis_origin_pct, 1)
            else:
                duplicates_sub_minis_origin_pct = round(duplicates_sub_minis_origin_pct)
            duplicates_sub_minis_origin = (
                f"{duplicates_sub_minis_origin} ({duplicates_sub_minis_origin_pct}%)"
            )
        all_rows = pd.DataFrame(
            {
                "Rows": [format_number(df.shape[0])],
                "Features": [df.shape[1]],
                "RAM (Mb)": [round(df.__sizeof__() / 1_048_576)],
                "Duplicates": [duplicates],
                "Dupl (sub - origin)": [duplicates_sub_minis_origin],
            }
        )
        # widget_DataFrame = widgets.Output()
        # with widget_DataFrame:
        #      display_markdown('**DataFrame**', raw=True)
        return (
            all_rows.style.set_caption("DataFrame")
            .set_table_styles(
                [
                    {
                        "selector": "caption",
                        "props": [
                            ("font-size", "18px"),
                            ("text-align", "left"),
                            ("font-weight", "bold"),
                        ],
                    }
                ]
            )
            .set_properties(**{"text-align": "left"})
            # .hide(axis="columns")
            .hide(axis="index")
        )

    def make_range_date_for_html(self, column):
        column_name = column.name
        fist_date = column.min()
        last_date = column.max()
        ram = round(column.__sizeof__() / 1_048_576)
        if ram == 0:
            ram = "<1 Mb"
        values = column.count()
        values = format_number(column.count())
        values_pct = column.count() * 100 / column.size
        if 0 < values_pct < 1:
            values_pct = "<1"
        elif values_pct > 99 and values_pct < 100:
            values_pct = round(values_pct, 1)
            if values_pct == 100:
                values_pct = 99.9
        else:
            values_pct = round(values_pct)
        values = f"{values} ({values_pct}%)"
        column_summary = pd.DataFrame(
            {"First date": [fist_date]
            , "Last date": [last_date]
            , "Values": [values]
            , "RAM (Mb)": [ram]}
        )
        return column_summary.T.reset_index()

    def make_summary_date_for_html(self, column):
        column_name = column.name
        zeros = ((column == 0) | (column == "")).sum()
        if zeros == 0:
            zeros = "---"
        else:
            zeros = format_number(((column == 0) | (column == "")).sum())
            zeros_pct = round(((column == 0) | (column == "")).sum() * 100 / column.size)
            if zeros_pct == 0:
                zeros_pct = "<1"
            zeros = f"{zeros} ({zeros_pct}%)"
        missing = column.isna().sum()
        if missing == 0:
            missing = "---"
        else:
            missing = format_number(column.isna().sum())
            missing_pct = round(column.isna().sum() * 100 / column.size)
            if missing_pct == 0:
                missing_pct = "<1"
            missing = f"{missing} ({missing_pct}%)"
        distinct = format_number(column.nunique())
        distinct_pct = column.nunique() * 100 / column.size
        if distinct_pct > 99 and distinct_pct < 100:
            distinct_pct = round(distinct_pct, 1)
            if distinct_pct == 100:
                distinct_pct = 99.9
        else:
            distinct_pct = round(distinct_pct)
        if distinct_pct == 0:
            distinct_pct = "<1"
        distinct = f"{distinct} ({distinct_pct}%)"
        duplicates = column.duplicated().sum()
        if duplicates == 0:
            duplicates = "---"
        else:
            duplicates = format_number(duplicates)
            duplicates_pct = column.duplicated().sum() * 100 / column.size
            if 0 < duplicates_pct < 1:
                duplicates_pct = "<1"
            elif duplicates_pct > 99 and duplicates_pct < 100:
                duplicates_pct = round(duplicates_pct, 1)
                if duplicates_pct == 100:
                    duplicates_pct = 99.9
            else:
                duplicates_pct = round(duplicates_pct)
            duplicates = f"{duplicates} ({duplicates_pct}%)"
        column_summary = pd.DataFrame(
            {
                "Zeros": [zeros],
                "Missing": [missing],
                "Distinct": [distinct],
                "Duplicates": [duplicates],
            }
        )
        return column_summary.T.reset_index()

    def make_check_missing_date_for_html(self, column):
        column_name = column.name
        fist_date = column.min()
        last_date = column.max()
        date_range = pd.date_range(start=fist_date, end=last_date, freq="D")
        years = date_range.year.unique()
        years_missed_pct = (~years.isin(column.dt.year.unique())).sum() * 100 / years.size
        if 0 < years_missed_pct < 1:
            years_missed_pct = "<1"
        elif years_missed_pct > 99:
            years_missed_pct = round(years_missed_pct, 1)
        else:
            years_missed_pct = round(years_missed_pct)
        months = date_range.to_period("M").unique()
        months_missed_pct = (
            (~months.isin(column.dt.to_period("M").unique())).sum() * 100 / months.size
        )
        if 0 < months_missed_pct < 1:
            months_missed_pct = "<1"
        elif months_missed_pct > 99:
            months_missed_pct = round(months_missed_pct, 1)
        else:
            months_missed_pct = round(months_missed_pct)
        weeks = date_range.to_period("W").unique()
        weeks_missed_pct = (
            (~weeks.isin(column.dt.to_period("W").unique())).sum() * 100 / weeks.size
        )
        if 0 < weeks_missed_pct < 1:
            weeks_missed_pct = "<1"
        elif weeks_missed_pct > 99:
            weeks_missed_pct = round(weeks_missed_pct, 1)
        else:
            weeks_missed_pct = round(weeks_missed_pct)
        days = date_range.unique().to_period("D")
        days_missed_pct = (
            (~days.isin(column.dt.to_period("D").unique())).sum() * 100 / days.size
        )
        if 0 < days_missed_pct < 1:
            days_missed_pct = "<1"
        elif days_missed_pct > 99:
            days_missed_pct = round(days_missed_pct, 1)
        else:
            days_missed_pct = round(days_missed_pct)

        column_summary = pd.DataFrame(
            {
                "Years missing": [f"{years_missed_pct}%"],
                "Months missing": [f"{months_missed_pct}%"],
                "Weeks missing": [f"{weeks_missed_pct}%"],
                "Days missing": [f"{days_missed_pct}%"],
            }
        )
        # display_html(f'<h4>{column_name}</h4>', raw=True)
        return column_summary.T.reset_index()

    def make_summary_for_html(self, column):
        column_name = column.name
        values = column.count()
        values = format_number(column.count())
        values_pct = column.count() * 100 / column.size
        if 0 < values_pct < 1:
            values_pct = "<1"
        elif values_pct > 99 and values_pct < 100:
            values_pct = round(values_pct, 1)
            if values_pct == 100:
                values_pct = 99.9
        else:
            values_pct = round(values_pct)
        values = f"{values} ({values_pct}%)"
        missing = column.isna().sum()
        if missing == 0:
            missing = "---"
        else:
            missing = format_number(column.isna().sum())
            missing_pct = round(column.isna().sum() * 100 / column.size)
            if missing_pct == 0:
                missing_pct = "<1"
            missing = f"{missing} ({missing_pct}%)"
        distinct = format_number(column.nunique())
        distinct_pct = column.nunique() * 100 / column.size
        if distinct_pct > 99 and distinct_pct < 100:
            distinct_pct = round(distinct_pct, 1)
            if distinct_pct == 100:
                distinct_pct = 99.9
        else:
            distinct_pct = round(distinct_pct)
        if distinct_pct == 0:
            distinct_pct = "<1"
        distinct = f"{distinct} ({distinct_pct}%)"
        zeros = ((column == 0) | (column == "")).sum()
        if zeros == 0:
            zeros = "---"
        else:
            zeros = format_number(((column == 0) | (column == "")).sum())
            zeros_pct = round(((column == 0) | (column == "")).sum() * 100 / column.size)
            if zeros_pct == 0:
                zeros_pct = "<1"
            zeros = f"{zeros} ({zeros_pct}%)"
        negative = (column < 0).sum()
        if negative == 0:
            negative = "---"
        else:
            negative = format_number(negative)
            negative_pct = round((column < 0).sum() * 100 / column.size)
            if negative_pct == 0:
                negative_pct = "<1"
            negative = f"{negative} ({negative_pct}%)"
        duplicates = column.duplicated().sum()
        if duplicates == 0:
            duplicates = "---"
        else:
            duplicates = format_number(duplicates)
            duplicates_pct = column.duplicated().sum() * 100 / column.size
            if 0 < duplicates_pct < 1:
                duplicates_pct = "<1"
            elif duplicates_pct > 99 and duplicates_pct < 100:
                duplicates_pct = round(duplicates_pct, 1)
                if duplicates_pct == 100:
                    duplicates_pct = 99.9
            else:
                duplicates_pct = round(duplicates_pct)
            duplicates = f"{duplicates} ({duplicates_pct}%)"
        ram = round(column.__sizeof__() / 1_048_576)
        if ram == 0:
            ram = "<1 Mb"
        column_summary = pd.DataFrame(
            {
                "Values": [values],
                "Missing": [missing],
                "Distinct": [distinct],
                "Duplicates": [duplicates],
                "Zeros": [zeros],
                "Negative": [negative],
                "RAM (Mb)": [ram],
            }
        )
        # display_html(f'<h4>{column_name}</h4>', raw=True)
        return column_summary.T.reset_index()

    def make_pct_for_html(self, column):
        max_ = format_number(column.max())
        q_95 = format_number(column.quantile(0.95))
        q_75 = format_number(column.quantile(0.75))
        median_ = format_number(column.median())
        q_25 = format_number(column.quantile(0.25))
        q_5 = format_number(column.quantile(0.05))
        min_ = format_number(column.min())
        column_summary = pd.DataFrame(
            {
                "Max": [max_],
                "95%": [q_95],
                "75%": [q_75],
                "50%": [median_],
                "25%": [q_25],
                "5%": [q_5],
                "Min": [min_],
            }
        )
        return column_summary.T.reset_index()

    def make_std_for_html(self, column):
        avg_ = format_number(column.mean())
        mode_ = column.mode()
        if mode_.size > 1:
            mode_ = "---"
        else:
            mode_ = format_number(mode_.iloc[0])
        range_ = format_number(column.max() - column.min())
        iQR = format_number(column.quantile(0.75) - column.quantile(0.25))
        std = format_number(column.std())
        kurt = format_number(column.kurtosis())
        skew = format_number(column.skew())
        column_summary = pd.DataFrame(
            {
                "Avg": [avg_],
                "Mode": [mode_],
                "Range": [range_],
                "iQR": [iQR],
                "std": [std],
                "kurt": [kurt],
                "skew": [skew],
            }
        )
        return column_summary.T.reset_index()

    def make_value_counts_for_html(self, column):
        column_name = column.name
        val_cnt = column.value_counts().iloc[:7]
        val_cnt_norm = column.value_counts(normalize=True).iloc[:7]
        column_name_pct = column_name + "_pct"
        val_cnt_norm.name = column_name_pct

        def make_value_counts_row(x):
            if x[column_name_pct] < 0.01:
                pct_str = "<1%"
            else:
                pct_str = f"({x[column_name_pct]:.0%})"
            # return f"{x[column_name]:.0f} {pct_str}"
            return f"{format_number(x[column_name])} {pct_str}"

        top_5 = (
            pd.concat([val_cnt, val_cnt_norm], axis=1)
            .reset_index()
            .apply(make_value_counts_row, axis=1)
            .to_frame()
        )

        return top_5.reset_index(drop=True)

    def make_hist_plotly_for_html(self, column):
        fig = px.histogram(
            column,
            nbins=20,
            histnorm="percent",
            template="simple_white",
            marginal='box',
            height=220,
            width=300,
        )
        fig.update_traces(
            marker_color="rgba(128, 60, 170, 0.9)",
            # text="*",
            # textfont=dict(color="rgba(128, 60, 170, 0.9)"),
        )
        fig.update_layout(
            xaxis2_visible=False,
            # xaxis=dict(
            #     anchor='y',  # Привязываем ось X к оси Y
            #     position=0,  # Устанавливаем позицию оси X
            # ),
            yaxis_domain=[0, 0.9],
            yaxis2_domain=[0.9, 1],
            margin=dict(l=30, r=0, b=0, t=0),
            showlegend=False,
            hoverlabel=dict(
                bgcolor="white",
            ),
            xaxis_title=None,
            yaxis_title=None,
            font=dict(size=13, family="Segoe UI", color="rgba(0, 0, 0, 0.8)"),
            xaxis_tickfont=dict(size=13, color="rgba(0, 0, 0, 0.8)"),
            yaxis_tickfont=dict(size=13, color="rgba(0, 0, 0, 0.8)"),
            xaxis_linecolor="rgba(0, 0, 0, 0.5)",
            yaxis_linecolor="rgba(0, 0, 0, 0.5)",
            xaxis_tickcolor="rgba(0, 0, 0, 0.5)",
            yaxis_tickcolor="rgba(0, 0, 0, 0.5)",
        )
        # fig.layout.yaxis.visible = False
        return fig

    def make_summary_obj_for_html(self, column):
        column_name = column.name
        values = column.count()
        values = format_number(column.count())
        values_pct = column.count() * 100 / column.size
        if 0 < values_pct < 1:
            values_pct = "<1"
        elif values_pct > 99 and values_pct < 100:
            values_pct = round(values_pct, 1)
            if values_pct == 100:
                values_pct = 99.9
        else:
            values_pct = round(values_pct)
        values = f"{values} ({values_pct}%)"
        missing = column.isna().sum()
        if missing == 0:
            missing = "---"
        else:
            missing = format_number(column.isna().sum())
            missing_pct = round(column.isna().sum() * 100 / column.size)
            if missing_pct == 0:
                missing_pct = "<1"
            missing = f"{missing} ({missing_pct}%)"
        distinct = format_number(column.nunique())
        distinct_pct = column.nunique() * 100 / column.size
        if distinct_pct > 99 and distinct_pct < 100:
            distinct_pct = round(distinct_pct, 1)
            if distinct_pct == 100:
                distinct_pct = 99.9
        else:
            distinct_pct = round(distinct_pct)
        if distinct_pct == 0:
            distinct_pct = "<1"
        distinct = f"{distinct} ({distinct_pct}%)"
        zeros = ((column == 0) | (column == "")).sum()
        if zeros == 0:
            zeros = "---"
        else:
            zeros = format_number(((column == 0) | (column == "")).sum())
            zeros_pct = round(((column == 0) | (column == "")).sum() * 100 / column.size)
            if zeros_pct == 0:
                zeros_pct = "<1"
            zeros = f"{zeros} ({zeros_pct}%)"
        duplicates = column.duplicated().sum()
        if duplicates == 0:
            duplicates = "---"
            duplicates_sub_minis_origin = "---"
        else:
            duplicates = format_number(duplicates)
            duplicates_pct = column.duplicated().sum() * 100 / column.size
            if 0 < duplicates_pct < 1:
                duplicates_pct = "<1"
            elif duplicates_pct > 99 and duplicates_pct < 100:
                duplicates_pct = round(duplicates_pct, 1)
                if duplicates_pct == 100:
                    duplicates_pct = 99.9
            else:
                duplicates_pct = round(duplicates_pct)
            duplicates = f"{duplicates} ({duplicates_pct}%)"
            duplicates_keep_false = column.duplicated(keep=False).sum()
            duplicates_sub = (
                column.str.lower()
                .str.strip()
                .str.replace(r"\s+", " ", regex=True)
                .duplicated(keep=False)
                .sum()
            )
            duplicates_sub_minis_origin = duplicates_sub - duplicates_keep_false
            if duplicates_sub_minis_origin == 0:
                duplicates_sub_minis_origin = "---"
            else:
                duplicates_sub_minis_origin = format_number(duplicates_sub_minis_origin)
                duplicates_sub_minis_origin_pct = (
                    (duplicates_sub - duplicates_keep_false) * 100 / duplicates_sub
                )
                if 0 < duplicates_sub_minis_origin_pct < 1:
                    duplicates_sub_minis_origin_pct = "<1"
                elif (
                    duplicates_sub_minis_origin_pct > 99
                    and duplicates_sub_minis_origin_pct < 100
                ) or duplicates_sub_minis_origin_pct < 1:
                    duplicates_sub_minis_origin_pct = round(
                        duplicates_sub_minis_origin_pct, 1
                    )
                else:
                    duplicates_sub_minis_origin_pct = round(duplicates_sub_minis_origin_pct)
                duplicates_sub_minis_origin = (
                    f"{duplicates_sub_minis_origin} ({duplicates_sub_minis_origin_pct}%)"
                )

        ram = round(column.__sizeof__() / 1_048_576)
        if ram == 0:
            ram = "<1 Mb"
        column_summary = pd.DataFrame(
            {
                "Values": [values],
                "Missing": [missing],
                "Distinct": [distinct],
                "Duplicated origin": [duplicates],
                "Dupl (modify - origin)": [duplicates_sub_minis_origin],
                "Empty": [zeros],
                "RAM (Mb)": [ram],
            }
        )
        return column_summary.T.reset_index()

    def make_value_counts_obj_for_html(self, column):
        column_name = column.name
        val_cnt = column.value_counts().iloc[:7]
        val_cnt_norm = column.value_counts(normalize=True).iloc[:7]
        column_name_pct = column_name + "_pct"
        val_cnt_norm.name = column_name_pct

        def make_value_counts_row(x):
            if x[column_name_pct] < 0.01:
                pct_str = "(<1%)"
            else:
                pct_str = f"({x[column_name_pct]:.0%})"
            return f"{x[column_name][:20]} {pct_str}"
        top_5 = (
            pd.concat([val_cnt, val_cnt_norm], axis=1)
            .reset_index()
            .apply(make_value_counts_row, axis=1)
            .to_frame()
        )
        return top_5.reset_index(drop=True)

    def make_bar_obj_for_html(self, column):
        df_fig = column.value_counts(ascending=True).iloc[-10:]
        text_labels = [label[:30] for label in df_fig.index.to_list()]
        fig = px.bar(
            df_fig, orientation="h", template="simple_white", height=220, width=500
        )
        fig.update_traces(marker_color="rgba(128, 60, 170, 0.8)")
        fig.update_layout(
            margin=dict(l=30, r=0, b=0, t=0),
            showlegend=False,
            hoverlabel=dict(
                bgcolor="white",
            ),
            xaxis_title=None,
            yaxis_title=None,
            font=dict(size=13, family="Segoe UI", color="rgba(0, 0, 0, 0.8)"),
            xaxis_tickfont=dict(size=13, color="rgba(0, 0, 0, 0.8)"),
            yaxis_tickfont=dict(size=13, color="rgba(0, 0, 0, 0.8)"),
            xaxis_linecolor="rgba(0, 0, 0, 0.5)",
            yaxis_linecolor="rgba(0, 0, 0, 0.5)",
            xaxis_tickcolor="rgba(0, 0, 0, 0.5)",
            yaxis_tickcolor="rgba(0, 0, 0, 0.5)",  
        )
        fig.update_traces(y=text_labels)
        # fig.tight_layout()
        return fig

    def make_row_for_html(self, df, column, funcs):
        if self.only_summary:
            funcs = [funcs[0]] + [None]
        fig_func = funcs[-1]
        if fig_func:
            fig = fig_func(df[column])
            if pd.api.types.is_numeric_dtype(df[column]):
                lower_bound = df[column].quantile(0.05)
                upper_bound = df[column].quantile(0.95)
                fig_without_outliers = fig_func(df[(df[column] >= lower_bound) & (df[column] <= upper_bound)][column])
        row_for_html = []
        for func in funcs[:-1]:
            row_for_html.append(func(df[column]))
            # display(func(df[column]))
        res_df = pd.concat(row_for_html, axis=1).T.reset_index(drop=True).T
        res_df.insert(2, '', " "*10)
        try:
            res_df.insert(5, ' ', " "*10)
            res_df.insert(8, '  ', " "*10)
        except:
            pass
        # display(res_df)
        res_df = res_df.fillna('')
        if self.only_summary:
            res_df_caption = f'Summary for "{column}"'
        else:
            if pd.api.types.is_numeric_dtype(df[column]):
                res_df_caption = f'Статистика и гистограмма столбца "{column}"'
            elif pd.api.types.is_datetime64_any_dtype(df[column]):
                res_df_caption = f'Статистика столбца "{column}"'
            else:
                res_df_caption = f'Статистика и топ-10 значений столбца "{column}"'
        if not pd.api.types.is_numeric_dtype(df[column]) and not pd.api.types.is_datetime64_any_dtype(df[column]):
            table_style = [
                    {
                        "selector": "caption",
                        "props": [
                            ("font-size", "16px"),
                            ("text-align", "left"),
                            ("font-weight", "bold"),
                        ],
                    },
                    {
                        "selector": "th, td",  # Применяем стиль к заголовкам и ячейкам
                        "props": [("min-width", "100px")]  # Устанавливаем минимальную ширину столбца
                    }                    
                ]
        else:
            table_style = [
                    {
                        "selector": "caption",
                        "props": [
                            ("font-size", "16px"),
                            ("text-align", "left"),
                            ("font-weight", "bold"),
                        ],
                    }                 
                ]            
        res_df = (res_df
            .style.set_caption(res_df_caption)
            .set_table_styles(table_style
            )
            .set_properties(**{"text-align": "left"})
            .hide(axis="columns")
            .hide(axis="index"))
        if fig_func:
            buf = io.BytesIO()
            fig.write_image(buf, format='png')
            buf.seek(0)
            img_str = base64.b64encode(buf.read()).decode('utf-8')
            if pd.api.types.is_numeric_dtype(df[column]) and self.hist_mode == 'dual':
                buf = io.BytesIO()
                fig_without_outliers.write_image(buf, format='png')
                buf.seek(0)
                img_without_outliers_str = base64.b64encode(buf.read()).decode('utf-8')        
            final_html = f"""
            <div style="display: flex; justify-content: flex-start; align-items: flex-end;">
                {res_df.to_html()}
                <div>
                    <img src="data:image/png;base64,{img_str}" alt="График"/>
                </div>
            """
            if pd.api.types.is_numeric_dtype(df[column]) and self.hist_mode == 'dual':
                final_html += f"""
                <div>
                    <img src="data:image/png;base64,{img_without_outliers_str}" alt="График"/>
                </div>            
            </div>
            """
            else:
                final_html += f"""          
            </div>
            """
        else:
            final_html = f"""
            <div style="display: flex; justify-content: flex-start; align-items: flex-end;">
                {res_df.to_html()}
            </div>
            """
            #  font-size: 10px;
        # {res_df.to_html(index=False)}
        return final_html


def check_duplicated(df, is_return_df=True):
    """
    Функция проверяет датафрейм на дубли.
    Если дубли есть, то возвращает датафрейм с дублями.
    """
    dupl = df.duplicated().sum()
    size = df.shape[0]
    if dupl == 0:
        return "no duplicates"
    if not is_return_df:
        return f"Duplicated is {dupl} ({(dupl / size):.1%}) rows"

    print(f"Duplicated is {dupl} ({(dupl / size):.1%}) rows")
    # приводим строки к нижнему регистру, удаляем пробелы
    return (
        df.apply(
            lambda x: (
                x.str.lower().str.strip().str.replace(r"\s+", " ", regex=True)
                if x.dtype == "object"
                else x
            )
        )
        .value_counts(dropna=False)
        .to_frame()
        .sort_values("count", ascending=False)
        .query("count > 1")
        # .rename(columns={0: 'Count'})
    )


def find_columns_with_duplicates(df, keep: str='first') -> pd.Series:
    """
    Фукнция проверяет каждый столбец в таблице,
    если есть дубликаты, то помещает строки исходного
    дата фрейма с этими дубликатами в Series.
    Индекс - название колонки.
    Если нужно соеденить фреймы в один, то используем
    keep : {'first', 'last', False}, default 'first'
    Determines which duplicates (if any) to mark.

    - ``first`` : Mark duplicates as ``True`` except for the first occurrence.
    - ``last`` : Mark duplicates as ``True`` except for the last occurrence.
    - False : Mark all duplicates as ``True``.
    pd.concat(res.to_list())
    """
    dfs_duplicated = pd.Series(dtype=int)
    cnt_duplicated = pd.Series(dtype=int)
    size = df.shape[0]
    for col in df.columns:
        is_duplicated = df[col].duplicated(keep=keep)
        if is_duplicated.any():
            dfs_duplicated[col] = df[is_duplicated]
            cnt_duplicated[col] = dfs_duplicated[col].shape[0]
    if cnt_duplicated.empty:
        print('There are no duplicated values')
    else:               
        display(
            cnt_duplicated.apply(lambda x: f"{x} ({(x / size):.2%})")
            .to_frame()
            .style.set_caption("Duplicates")
            .set_table_styles(
                [
                    {
                        "selector": "caption",
                        "props": [
                            ("font-size", "18px"),
                            ("text-align", "left"),
                            ("font-weight", "bold"),
                        ],
                    }
                ]
            )
            .hide(axis="columns")
        )
    return dfs_duplicated


def check_duplicated_combinations_gen(df, n=2):
    """
    Функция считает дубликаты между всеми возможными комбинациями между столбцами.
    Сначала для проверки на дубли берутся пары столбцов.
    Затем по 3 столбца. И так все возможные комибнации.
    Можно выбрать до какого количества комбинаций двигаться.
    n - максимальное возможное количество столбцов в комбинациях. По умолчанию беруться все столбцы
    """
    if n < 2:
        return
    df_copy = df.apply(
        lambda x: (
            x.str.lower().str.strip().str.replace(r"\s+", " ", regex=True)
            if x.dtype == "object"
            else x
        )
    )
    c2 = itertools.combinations(df.columns, 2)
    dupl_df_c2 = pd.DataFrame([], index=df.columns, columns=df.columns)
    df_size = df.shape[0]
    print(f"Group by 2 columns")
    for c in c2:
        duplicates = df_copy[list(c)].duplicated().sum()
        dupl_df_c2.loc[c[1], c[0]] = f'{pretty_value(duplicates)} ({(duplicates / df_size):.1%})' if duplicates / df_size >= 0.01 else f'{pretty_value(duplicates)} < 1%'
    display(
        dupl_df_c2.fillna("")
        .style.set_caption("Duplicates in both columns")
        .set_table_styles(
            [
                {
                    "selector": "caption",
                    "props": [
                        ("font-size", "18px"),
                        ("text-align", "left"),
                        ("font-weight", "bold"),
                    ],
                }
            ]
        )
    )
    yield
    if n < 3:
        return
    c3 = itertools.combinations(df.columns, 3)
    dupl_c3_list = []
    print(f"Group by 3 columns")
    for c in c3:
        duplicates = df_copy[list(c)].duplicated().sum()
        if duplicates:
            dupl_c3_list.append([" | ".join(c), duplicates])
    dupl_df_c3 = pd.DataFrame(dupl_c3_list)
    # разобьем таблицу на 3 части, чтобы удобнее читать
    yield (
        pd.concat(
            [
                # part_df.reset_index(drop=True)
                # for part_df in np.array_split(dupl_df_c3, 3)
                dupl_df_c3[i:i + 10].reset_index(drop=True)
                    for i in range(0, dupl_df_c3.shape[0], 10)
            ],
            axis=1,
        )
        .style.format({1: "{:.0f}"}, na_rep="")
        .set_caption("Duplicates in 3 columns")
        .hide(axis="index")
        .hide(axis="columns")
    )
    if n < 4:
        return
    for col_n in range(4, df.columns.size + 1):
        print(f"Group by {col_n} columns")
        cn = itertools.combinations(df.columns, col_n)
        dupl_cn_list = []
        for c in cn:
            duplicates = df_copy[list(c)].duplicated().sum()
            if duplicates:
                dupl_cn_list.append([" | ".join(c), duplicates])
        dupl_df_cn = pd.DataFrame(dupl_cn_list)
        # разобьем таблицу на 3 части, чтобы удобнее читать
        yield (
            pd.concat(
                [
                    dupl_df_c3[i:i + 10].reset_index(drop=True)
                        for i in range(0, dupl_df_c3.shape[0], 10)
                ],
                axis=1,
            )
            .style.format({1: "{:.0f}"}, na_rep="")
            .set_caption(f"Duplicates in {col_n} columns")
            .hide(axis="index")
            .hide(axis="columns")
        )
        if n < col_n + 1:
            return

def show_missings(df) -> None:
    """
    Функция выводит количество пропусков в датафрейме
    """
    cnt_missing = pd.Series(dtype=int)
    size = df.shape[0]
    for col in df.columns:
        na_cnt = df[col].isna().sum()
        if na_cnt:
            cnt_missing[col] = na_cnt
    if cnt_missing.empty:
        print('There are no missing values')
    else:
        display(
            cnt_missing.apply(lambda x: f"{x} ({(x / size):.2%})")
            .to_frame()
            .style.set_caption("Missings")
            .set_table_styles(
                [
                    {
                        "selector": "caption",
                        "props": [
                            ("font-size", "18px"),
                            ("text-align", "left"),
                            ("font-weight", "bold"),
                        ],
                    }
                ]
            )
            .hide(axis="columns")
        )

def find_columns_with_missing_values(df, is_display=True) -> pd.Series:
    """
    Фукнция проверяет каждый столбец в таблице,
    если есть пропуски, то помещает строки исходного
    дата фрейма с этими пропусками в Series.
    Индекс - название колонки.
    Если нужно соеденить фреймы в один, то используем
    pd.concat(res.to_list())
    """
    dfs_na = pd.Series(dtype=int)
    cnt_missing = pd.Series(dtype=int)
    size = df.shape[0]
    for col in df.columns:
        is_na = df[col].isna()
        if is_na.any():
            dfs_na[col] = df[is_na]
            cnt_missing[col] = dfs_na[col].shape[0]
    if not is_display:
        if cnt_missing.empty:
            return 'no missings'
        else:
            return 'there are missings'
    if cnt_missing.empty:
        print('There are no missing values')
    else:            
        if pd.__version__ == "1.3.5":
            display(
                cnt_missing.apply(lambda x: f"{x} ({(x / size):.2%})")
                .to_frame()
                .style.set_caption("Missings")
                .set_table_styles(
                    [
                        {
                            "selector": "caption",
                            "props": [
                                ("font-size", "18px"),
                                ("text-align", "left"),
                                ("font-weight", "bold"),
                            ],
                        }
                    ]
                )
                .hide_columns()
            )
        else:
            display(
                cnt_missing.apply(lambda x: f"{x} ({(x / size):.2%})")
                .to_frame()
                .style.set_caption("Missings")
                .set_table_styles(
                    [
                        {
                            "selector": "caption",
                            "props": [
                                ("font-size", "18px"),
                                ("text-align", "left"),
                                ("font-weight", "bold"),
                            ],
                        }
                    ]
                )
                .hide(axis="columns")
            )
    return dfs_na


def check_na_in_both_columns(df, cols: list) -> pd.DataFrame:
    """
    Фукнция проверяет есть ли пропуски одновременно во всех указанных столбцах
    и возвращает датафрейм только со строками, в которых пропуски одновременно во всех столбцах
    """
    size = df.shape[0]
    mask = df[cols].isna().all(axis=1)
    na_df = df[mask]
    cols_missings = [df[col].isna().sum() for col in cols]
    print(
        f"{na_df.shape[0]} ({(na_df.shape[0] / size):.2%} of all)"
        + ' '.join([f'({(na_df.shape[0] / cols_missings[i]):.2%} of {cols[i]})' for i in range(len(cols_missings))])
        + f' rows with missings simultaneously in {cols}'
    )
        
    return na_df

def check_zeros_in_both_columns(df, cols: list) -> pd.DataFrame:
    """
    Функция проверяет есть ли нулевые значения одновременно во всех указанных столбцах
    и возвращает датафрейм только со строками, в которых нулевые значения одновременно во всех столбцах.
    """
    size = df.shape[0]
    mask = (df[cols] == 0).all(axis=1)
    zero_df = df[mask]
    print(
        f"{zero_df.shape[0]} ({(zero_df.shape[0] / size):.2%}) rows with zeros simultaneously in {cols}"
    )
    return zero_df


def get_missing_value_proportion_by_category(
    df: pd.DataFrame, column_with_missing_values: str, category_column: str = None
):
    """
    Return a DataFrame with the proportion of missing values for each category.

    Parameters:
    df (pd.DataFrame): Input DataFrame
    column_with_missing_values (str): Column with missing values
    category_column (str): Category column

    Returns:
    if category_column not None: retrun result for category_column
        - pd.DataFrame: DataFrame with the proportion of missing values for each category
    else: generator for all catogorical column in df
        - pd.DataFrame: DataFrame with the proportion of missing values for each category
    """
    if category_column:
        # Create a mask to select rows with missing values in the specified column
        mask = df[column_with_missing_values].isna()
        size = df[column_with_missing_values].size
        # Group by category and count the number of rows with missing values
        missing_value_counts = (
            df[mask].groupby(category_column).size().reset_index(name="missing_count")
        )
        summ_missing_counts = missing_value_counts["missing_count"].sum()
        # Get the total count for each category
        total_counts = (
            df.groupby(category_column).size().reset_index(name="total_count")
        )

        # Merge the two DataFrames to calculate the proportion of missing values
        result_df = pd.merge(missing_value_counts, total_counts, on=category_column)
        result_df["missing_value_in_category_pct"] = (
            result_df["missing_count"] / result_df["total_count"]
        ).apply(lambda x: f"{x:.1%}")
        result_df["missing_value_in_column_pct"] = (
            result_df["missing_count"] / summ_missing_counts
        ).apply(lambda x: f"{x:.1%}")
        result_df["total_count_pct"] = (result_df["total_count"] / size).apply(
            lambda x: f"{x:.1%}"
        )
        # Return the result DataFrame
        display(
            result_df[
                [
                    category_column,
                    "total_count",
                    "missing_count",
                    "missing_value_in_category_pct",
                    "missing_value_in_column_pct",
                    "total_count_pct",
                ]
            ]
            .style.set_caption(
                f'Missing values in "{column_with_missing_values}" by categroy "{category_column}"'
            )
            .set_table_styles(
                [
                    {
                        "selector": "caption",
                        "props": [
                            ("font-size", "18px"),
                            ("text-align", "left"),
                            ("font-weight", "bold"),
                        ],
                    }
                ]
            )
        )
        yield
    else:
        categroy_columns = [
            col for col in df.columns if pd.api.types.is_categorical_dtype(df[col])
        ]
        for category_column in categroy_columns:
            # Create a mask to select rows with missing values in the specified column
            mask = df[column_with_missing_values].isna()
            size = df[column_with_missing_values].size
            # Group by category and count the number of rows with missing values
            missing_value_counts = (
                df[mask]
                .groupby(category_column)
                .size()
                .reset_index(name="missing_count")
            )
            summ_missing_counts = missing_value_counts["missing_count"].sum()
            # Get the total count for each category
            total_counts = (
                df.groupby(category_column).size().reset_index(name="total_count")
            )

            # Merge the two DataFrames to calculate the proportion of missing values
            result_df = pd.merge(missing_value_counts, total_counts, on=category_column)
            result_df["missing_value_in_category_pct"] = (
                result_df["missing_count"] / result_df["total_count"]
            ).apply(lambda x: f"{x:.1%}")
            result_df["missing_value_in_column_pct"] = (
                result_df["missing_count"] / summ_missing_counts
            ).apply(lambda x: f"{x:.1%}")
            result_df["total_count_pct"] = (result_df["total_count"] / size).apply(
                lambda x: f"{x:.1%}"
            )
            # Return the result DataFrame
            display(
                result_df[
                    [
                        category_column,
                        "total_count",
                        "missing_count",
                        "missing_value_in_category_pct",
                        "missing_value_in_column_pct",
                        "total_count_pct",
                    ]
                ]
                .style.set_caption(
                    f'Missing values in "{column_with_missing_values}" by categroy "{category_column}"'
                )
                .set_table_styles(
                    [
                        {
                            "selector": "caption",
                            "props": [
                                ("font-size", "18px"),
                                ("text-align", "left"),
                                ("font-weight", "bold"),
                            ],
                        }
                    ]
                )
            )
            yield


def missings_by_category_gen(df, series_missed):
    """
    Генератор.
    Для каждой колонки в series_missed функция выводит выборку датафрейма с пропусками в этой колонке.
    И затем выводит информацию о пропусках по каждой категории в таблице.
    """
    for col in series_missed.index:
        display(
            series_missed[col]
            .sample(10)
            .style.set_caption(f"Sample missings in {col}")
            .set_table_styles(
                [
                    {
                        "selector": "caption",
                        "props": [
                            ("font-size", "18px"),
                            ("text-align", "left"),
                            ("font-weight", "bold"),
                        ],
                    }
                ]
            )
        )
        yield
        gen = get_missing_value_proportion_by_category(df, col)
        for _ in gen:
            yield


def get_duplicates_value_proportion_by_category(
    df: pd.DataFrame, column_with_dublicated_values: str, category_column: str
) -> pd.DataFrame:
    """
    Return a DataFrame with the proportion of duplicated values for each category.

    Parameters:
    df (pd.DataFrame): Input DataFrame
    column_with_dublicated_values (str): Column with dublicated values
    category_column (str): Category column

    Returns:
    pd.DataFrame: DataFrame with the proportion of missing values for each category
    """
    # Create a mask to select rows with dublicated values in the specified column
    mask = df[column_with_dublicated_values].duplicated()

    # Group by category and count the number of rows with dublicated values
    dublicated_value_counts = (
        df[mask].groupby(category_column).size().reset_index(name="dublicated_count")
    )
    summ_dublicated_value_counts = dublicated_value_counts["dublicated_count"].sum()
    # Get the total count for each category
    total_counts = df.groupby(category_column).size().reset_index(name="total_count")

    # Merge the two DataFrames to calculate the proportion of dublicated values
    result_df = pd.merge(dublicated_value_counts, total_counts, on=category_column)
    result_df["dublicated_value_in_category_pct"] = (
        result_df["dublicated_count"] / result_df["total_count"]
    ).apply(lambda x: f"{x:.1%}")
    result_df["dublicated_value_in_column_pct"] = (
        result_df["dublicated_count"] / summ_dublicated_value_counts
    ).apply(lambda x: f"{x:.1%}")
    # Return the result DataFrame
    return result_df[
        [
            category_column,
            "total_count",
            "dublicated_count",
            "dublicated_value_in_category_pct",
            "dublicated_value_in_column_pct",
        ]
    ]


def check_or_fill_missing_values(df, target_column, identifier_columns, check=True):
    """
    Fill missing values in the target column by finding matching rows without missing values
    in the identifier columns.

    Parameters:
    df (pd.DataFrame): The input DataFrame.
    target_column (str): The column with missing values to be filled.
    identifier_columns (list of str): The columns that uniquely identify the rows.
    check: Is check or fill, default True

    Returns:
    pd.DataFrame: The input DataFrame with missing values filled in the target column.
    """
    # Identify rows with missing values in the target column
    missing_rows = df[df[target_column].isna()]

    # Extract unique combinations of identifying columns from the rows with missing values
    unique_identifiers = missing_rows[identifier_columns].drop_duplicates()

    # Find matching rows without missing values in the target column
    df_unique_identifiers_for_compare = (
        df[identifier_columns].set_index(identifier_columns).index
    )
    unique_identifiers_for_comapre = unique_identifiers.set_index(
        identifier_columns
    ).index
    matching_rows = df[
        df_unique_identifiers_for_compare.isin(unique_identifiers_for_comapre)
        & (~df["total_income"].isna())
    ]
    # Check if there are matching rows without missing values
    if not matching_rows.empty:
        if check:
            print(
                f"Found {matching_rows.shape[0]} matching rows without missing values"
            )
            return
        # Replace missing values with values from matching rows
        df.loc[missing_rows.index, target_column] = matching_rows[target_column].values
        print(f"Fiiled {matching_rows.shape[0]} matching rows without missing values")
    else:
        print("No matching rows without missing values found.")


def get_non_matching_rows(df, col1, col2):
    """
    Возвращает строки DataFrame, для которых значения в col1 имеют разные значения в col2.

    Parameters:
    df (pd.DataFrame): DataFrame с данными
    col1 (str): Название колонки с значениями, для которых нужно проверить уникальность
    col2 (str): Название колонки с значениями, которые нужно проверить на совпадение

    Returns:
    pd.DataFrame: Строки DataFrame, для которых значения в col1 имеют разные значения в col2
    """
    non_unique_values = (
        df.groupby(col1, observed=False)[col2].nunique()[lambda x: x > 1].index
    )
    non_matching_rows = df[df[col1].isin(non_unique_values)]
    if non_matching_rows.empty:
        print("Нет строк для которых значения в col1 имеют разные значения в col2")
    else:
        return non_matching_rows


def detect_outliers_Zscore(df: pd.DataFrame, z_level: float = 3.5) -> pd.Series:
    """
    Detect outliers in a DataFrame using the Modified Z-score method.

    Parameters:
    df (pd.DataFrame): DataFrame to detect outliers in.
    z_level (float, optional): Modified Z-score threshold for outlier detection. Defaults to 3.5.

    Returns:
    pd.Series: Series with column names as indices and outlier DataFrames as values.
    """
    outliers = pd.Series(dtype=object)
    cnt_outliers = pd.Series(dtype=int)
    for col in filter(lambda x: pd.api.types.is_numeric_dtype(df[x]), df.columns):
        median = df[col].median()
        mad = np.median(np.abs(df[col] - median))
        modified_z_scores = 0.6745 * (df[col] - median) / mad
        outliers[col] = df[np.abs(modified_z_scores) > z_level]
        cnt_outliers[col] = outliers[col].shape[0]
    display(
        cnt_outliers.to_frame()
        .T.style.set_caption("Outliers")
        .set_table_styles(
            [
                {
                    "selector": "caption",
                    "props": [
                        ("font-size", "18px"),
                        ("text-align", "left"),
                        ("font-weight", "bold"),
                    ],
                }
            ]
        )
        .hide(axis="index")
    )
    return outliers


def detect_outliers_quantile(
    df: pd.DataFrame, lower_quantile: float = 0.05, upper_quantile: float = 0.95
) -> pd.Series:
    """
    Detect outliers in a DataFrame using quantile-based method.

    Parameters:
    df (pd.DataFrame): DataFrame to detect outliers in.
    lower_quantile (float, optional): Lower quantile threshold for outlier detection. Defaults to 0.25.
    upper_quantile (float, optional): Upper quantile threshold for outlier detection. Defaults to 0.75.

    Returns:
    pd.Series: Series with column names as indices and outlier DataFrames as values.
    """
    outliers = pd.Series(dtype=object)
    cnt_outliers = pd.Series(dtype=int)
    size = df.shape[0]
    for col in filter(lambda x: pd.api.types.is_numeric_dtype(df[x]), df.columns):
        lower_bound = df[col].quantile(lower_quantile)
        upper_bound = df[col].quantile(upper_quantile)
        outliers[col] = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        cnt_outliers[col] = outliers[col].shape[0]
    display(
        cnt_outliers.apply(lambda x: f"{x} ({(x / size):.2%})")
        .to_frame()
        .style.set_caption("Outliers")
        .set_table_styles(
            [
                {
                    "selector": "caption",
                    "props": [
                        ("font-size", "18px"),
                        ("text-align", "left"),
                        ("font-weight", "bold"),
                    ],
                }
            ]
        )
        .hide(axis="columns")
    )
    return outliers


def fill_missing_values_using_helper_column(df, categorical_column, helper_column):
    """
    Заполнить пропуски в категориальной переменной на основе значений другой переменной.

    Parameters:
    df (pd.DataFrame): Исходная таблица.
    categorical_column (str): Имя категориальной переменной с пропусками.
    helper_column (str): Имя переменной без пропусков, используемой для заполнения пропусков.

    Returns:
    pd.DataFrame: Таблица с заполненными пропусками.
    """
    # Создать таблицу справочника с уникальными значениями helper_column
    helper_df = df[[helper_column, categorical_column]].drop_duplicates(helper_column)

    # Удалить строки с пропусками в categorical_column
    helper_df = helper_df.dropna(subset=[categorical_column])

    # Создать новую таблицу с заполненными пропусками
    filled_df = df.drop(categorical_column, axis=1)
    filled_df = filled_df.merge(helper_df, on=helper_column, how="left")

    return filled_df


def get_outlier_quantile_proportion_by_category(
    df: pd.DataFrame,
    column_with_outliers: str,
    category_column: str = None,
    lower_quantile: float = 0.05,
    upper_quantile: float = 0.95,
):
    """
    Return a DataFrame with the proportion of outliers for each category.

    Parameters:
    df (pd.DataFrame): Input DataFrame
    column_with_outliers (str): Column with outliers
    category_column (str): Category column
    lower_quantile (float): Lower quantile (e.g., 0.25 for 25th percentile)
    upper_quantile (float): Upper quantile (e.g., 0.75 for 75th percentile)

    Returns:
    None
    """
    # Calculate the lower and upper bounds for outliers
    lower_bound = df[column_with_outliers].quantile(lower_quantile)
    upper_bound = df[column_with_outliers].quantile(upper_quantile)

    # Create a mask to select rows with outliers in the specified column
    mask = (df[column_with_outliers] < lower_bound) | (
        df[column_with_outliers] > upper_bound
    )
    size = df[column_with_outliers].size
    if category_column:
        # Group by category and count the number of rows with outliers
        outlier_counts = (
            df[mask].groupby(category_column).size().reset_index(name="outlier_count")
        )
        summ_outlier_counts = outlier_counts["outlier_count"].sum()
        # Get the total count for each category
        total_counts = (
            df.groupby(category_column).size().reset_index(name="total_count")
        )

        # Merge the two DataFrames to calculate the proportion of outliers
        result_df = pd.merge(outlier_counts, total_counts, on=category_column)
        result_df["outlier_in_category_pct"] = (
            result_df["outlier_count"] / result_df["total_count"]
        ).apply(lambda x: f"{x:.1%}")
        result_df["outlier_in_column_pct"] = (
            result_df["outlier_count"] / summ_outlier_counts
        ).apply(lambda x: f"{x:.1%}")
        result_df["total_count_pct"] = (result_df["total_count"] / size).apply(
            lambda x: f"{x:.1%}"
        )
        display(
            result_df[
                [
                    category_column,
                    "total_count",
                    "outlier_count",
                    "outlier_in_category_pct",
                    "outlier_in_column_pct",
                    "total_count_pct",
                ]
            ]
            .style.set_caption(
                f'Outliers in "{column_with_outliers}" by category "{category_column}"'
            )
            .set_table_styles(
                [
                    {
                        "selector": "caption",
                        "props": [
                            ("font-size", "18px"),
                            ("text-align", "left"),
                            ("font-weight", "bold"),
                        ],
                    }
                ]
            )
            .hide(axis="index")
        )
        yield
    else:
        categroy_columns = [
            col for col in df.columns if pd.api.types.is_categorical_dtype(df[col])
        ]
        for category_column in categroy_columns:
            # Group by category and count the number of rows with outliers
            outlier_counts = (
                df[mask]
                .groupby(category_column)
                .size()
                .reset_index(name="outlier_count")
            )
            summ_outlier_counts = outlier_counts["outlier_count"].sum()
            # Get the total count for each category
            total_counts = (
                df.groupby(category_column).size().reset_index(name="total_count")
            )

            # Merge the two DataFrames to calculate the proportion of outliers
            result_df = pd.merge(outlier_counts, total_counts, on=category_column)
            result_df["outlier_in_category_pct"] = (
                result_df["outlier_count"] / result_df["total_count"]
            ).apply(lambda x: f"{x:.1%}")
            result_df["outlier_in_column_pct"] = (
                result_df["outlier_count"] / summ_outlier_counts
            ).apply(lambda x: f"{x:.1%}")
            result_df["total_count_pct"] = (result_df["total_count"] / size).apply(
                lambda x: f"{x:.1%}"
            )
            display(
                result_df[
                    [
                        category_column,
                        "total_count",
                        "outlier_count",
                        "outlier_in_category_pct",
                        "outlier_in_column_pct",
                        "total_count_pct",
                    ]
                ]
                .style.set_caption(
                    f'Outliers in "{column_with_outliers}" by category "{category_column}"'
                )
                .set_table_styles(
                    [
                        {
                            "selector": "caption",
                            "props": [
                                ("font-size", "18px"),
                                ("text-align", "left"),
                                ("font-weight", "bold"),
                            ],
                        }
                    ]
                )
                .hide(axis="index")
            )
            yield


def outliers_by_category_gen(
    df, series_outliers, lower_quantile: float = 0.05, upper_quantile: float = 0.95
):
    """
    Генератор.
    Для каждой колонки в series_outliers функция выводит выборку датафрейма с выбросами (определяется по квантилям) в этой колонке.
    И затем выводит информацию о выбросах по каждой категории в таблице.
    """
    for col in series_outliers.index:
        print(f"Value counts outliers")
        display(
            series_outliers[col][col]
            .value_counts()
            .to_frame("outliers")
            .head(10)
            .style.set_caption(f"{col}")
            .set_table_styles(
                [
                    {
                        "selector": "caption",
                        "props": [
                            ("font-size", "18px"),
                            ("text-align", "left"),
                            ("font-weight", "bold"),
                        ],
                    }
                ]
            )
        )
        yield
        display(
            series_outliers[col]
            .sample(10)
            .style.set_caption(f"Sample outliers in {col}")
            .set_table_styles(
                [
                    {
                        "selector": "caption",
                        "props": [
                            ("font-size", "18px"),
                            ("text-align", "left"),
                            ("font-weight", "bold"),
                        ],
                    }
                ]
            )
        )
        yield
        gen = get_outlier_quantile_proportion_by_category(
            df, col, lower_quantile=lower_quantile, upper_quantile=upper_quantile
        )
        for _ in gen:
            yield


def get_outlier_proportion_by_category_modified_z_score(
    df: pd.DataFrame,
    column_with_outliers: str,
    category_column: str,
    threshold: float = 3.5,
) -> None:
    """
    Return a DataFrame with the proportion of outliers for each category.

    Parameters:
    df (pd.DataFrame): Input DataFrame
    column_with_outliers (str): Column with outliers
    category_column (str): Category column
    threshold (float): Threshold for modified z-score

    Returns:
    None
    """
    # Calculate the median and median absolute deviation (MAD) for the specified column
    median = df[column_with_outliers].median()
    mad = np.median(np.abs(df[column_with_outliers] - median))

    # Create a mask to select rows with outliers in the specified column
    mask = np.abs(0.6745 * (df[column_with_outliers] - median) / mad) > threshold

    # Group by category and count the number of rows with outliers
    outlier_counts = (
        df[mask].groupby(category_column).size().reset_index(name="outlier_count")
    )
    summ_outlier_counts = outlier_counts["outlier_count"].sum()

    # Get the total count for each category
    total_counts = df.groupby(category_column).size().reset_index(name="total_count")

    # Merge the two DataFrames to calculate the proportion of outliers
    result_df = pd.merge(outlier_counts, total_counts, on=category_column)
    result_df["outlier_in_category_pct"] = (
        result_df["outlier_count"] / result_df["total_count"]
    ).apply(lambda x: f"{x:.1%}")
    result_df["outlier_in_column_pct"] = (
        result_df["outlier_count"] / summ_outlier_counts
    ).apply(lambda x: f"{x:.1%}")

    display(
        result_df[
            [
                category_column,
                "total_count",
                "outlier_count",
                "outlier_in_category_pct",
                "outlier_in_column_pct",
            ]
        ]
        .style.set_caption(
            f'Outliers in "{column_with_outliers}" by category "{category_column}"'
        )
        .set_table_styles(
            [
                {
                    "selector": "caption",
                    "props": [
                        ("font-size", "18px"),
                        ("text-align", "left"),
                        ("font-weight", "bold"),
                    ],
                }
            ]
        )
        .hide(axis="index")
    )


def find_columns_with_negative_values(df, df_name=None) -> pd.Series:
    """
    Фукнция проверяет каждый столбец в таблице,
    если есть отрицательные значения, то помещает строки исходного
    дата фрейма с этими значениями в Series.
    Индекс - название колонки.
    Если нужно соеденить фреймы в один, то используем
    pd.concat(res.to_list())
    """
    dfs_na = pd.Series(dtype=int)
    cnt_negative = pd.Series(dtype=int)
    size = df.shape[0]
    num_columns = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
    for col in num_columns:
        is_negative = df[col] < 0
        if is_negative.any():
            dfs_na[col] = df[is_negative]
            cnt_negative[col] = dfs_na[col].shape[0]
    if cnt_negative.empty:
        if df_name:
            return
        print('There are no negative values')
    else:
        if df_name:
            caption = f'Negative in "{df_name}"'
        else:
            caption = 'Negative'
        display(cnt_negative.apply(lambda x: f"{x} ({(x / size):.2%})")
            .to_frame()
            .style.set_caption(caption)
            .set_table_styles(
                [
                    {
                        "selector": "caption",
                        "props": [
                            ("font-size", "18px"),
                            ("text-align", "left"),
                            ("font-weight", "bold"),
                        ],
                    }
                ]
            )
            .hide(axis="columns"))
    if not df_name:
        return dfs_na



def get_negative_proportion_by_category(
    df: pd.DataFrame, column_with_negative: str, category_column: str = None
):
    """
    Return a DataFrame with the proportion of negative value for each category.

    Parameters:
    df (pd.DataFrame): Input DataFrame
    column_with_negative (str): Column with negative value
    category_column (str): Category column

    Returns:
    None
    """

    # Create a mask to select rows with outliers in the specified column
    mask = df[column_with_negative] < 0
    size = df[column_with_negative].size
    if category_column:
        # Group by category and count the number of rows with outliers
        negative_counts = (
            df[mask].groupby(category_column).size().reset_index(name="negative_count")
        )
        summ_negative_counts = negative_counts["negative_count"].sum()
        # Get the total count for each category
        total_counts = (
            df.groupby(category_column).size().reset_index(name="total_count")
        )

        # Merge the two DataFrames to calculate the proportion of negatives
        result_df = pd.merge(negative_counts, total_counts, on=category_column)
        result_df["negative_in_category_pct"] = (
            result_df["negative_count"] / result_df["total_count"]
        ).apply(lambda x: f"{x:.1%}")
        result_df["negative_in_column_pct"] = (
            result_df["negative_count"] / summ_negative_counts
        ).apply(lambda x: f"{x:.1%}")
        result_df["total_count_pct"] = (result_df["total_count"] / size).apply(
            lambda x: f"{x:.1%}"
        )
        display(
            result_df[
                [
                    category_column,
                    "total_count",
                    "negative_count",
                    "negative_in_category_pct",
                    "negative_in_column_pct",
                    "total_count_pct",
                ]
            ]
            .style.set_caption(
                f'negatives in "{column_with_negative}" by category "{category_column}"'
            )
            .set_table_styles(
                [
                    {
                        "selector": "caption",
                        "props": [
                            ("font-size", "18px"),
                            ("text-align", "left"),
                            ("font-weight", "bold"),
                        ],
                    }
                ]
            )
            .hide(axis="index")
        )
        yield
    else:
        categroy_columns = [
            col for col in df.columns if pd.api.types.is_categorical_dtype(df[col])
        ]
        for category_column in categroy_columns:
            # Group by category and count the number of rows with negatives
            negative_counts = (
                df[mask]
                .groupby(category_column)
                .size()
                .reset_index(name="negative_count")
            )
            summ_negative_counts = negative_counts["negative_count"].sum()
            # Get the total count for each category
            total_counts = (
                df.groupby(category_column).size().reset_index(name="total_count")
            )

            # Merge the two DataFrames to calculate the proportion of negatives
            result_df = pd.merge(negative_counts, total_counts, on=category_column)
            result_df["negative_in_category_pct"] = (
                result_df["negative_count"] / result_df["total_count"]
            ).apply(lambda x: f"{x:.1%}")
            result_df["negative_in_column_pct"] = (
                result_df["negative_count"] / summ_negative_counts
            ).apply(lambda x: f"{x:.1%}")
            result_df["total_count_pct"] = (result_df["total_count"] / size).apply(
                lambda x: f"{x:.1%}"
            )
            display(
                result_df[
                    [
                        category_column,
                        "total_count",
                        "negative_count",
                        "negative_in_category_pct",
                        "negative_in_column_pct",
                        "total_count_pct",
                    ]
                ]
                .style.set_caption(
                    f'negatives in "{column_with_negative}" by category "{category_column}"'
                )
                .set_table_styles(
                    [
                        {
                            "selector": "caption",
                            "props": [
                                ("font-size", "18px"),
                                ("text-align", "left"),
                                ("font-weight", "bold"),
                            ],
                        }
                    ]
                )
                .hide(axis="index")
            )
            yield


def negative_by_category_gen(df, series_negative):
    """
    Генератор.
    Для каждой колонки в series_negative функция выводит выборку датафрейма с отрицательными значениями.
    И затем выводит информацию об отрицательных значениях по каждой категории в таблице.
    """
    for col in series_negative.index:
        print(f"Value counts negative")
        display(
            series_negative[col][col]
            .value_counts()
            .to_frame("negative")
            .head(10)
            .style.set_caption(f"{col}")
            .set_table_styles(
                [
                    {
                        "selector": "caption",
                        "props": [
                            ("font-size", "18px"),
                            ("text-align", "left"),
                            ("font-weight", "bold"),
                        ],
                    }
                ]
            )
        )
        yield
        display(
            series_negative[col]
            .sample(10)
            .style.set_caption(f"Sample negative in {col}")
            .set_table_styles(
                [
                    {
                        "selector": "caption",
                        "props": [
                            ("font-size", "18px"),
                            ("text-align", "left"),
                            ("font-weight", "bold"),
                        ],
                    }
                ]
            )
        )
        yield
        gen = get_negative_proportion_by_category(df, col)
        for _ in gen:
            yield


def find_columns_with_zeros_values(df, df_name=None) -> pd.Series:
    """
    Фукнция проверяет каждый столбец в таблице,
    если есть нулевые значения, то помещает строки исходного
    дата фрейма с этими значениями в Series.
    Индекс - название колонки.
    Если нужно соеденить фреймы в один, то используем
    pd.concat(res.to_list())
    """
    dfs_na = pd.Series(dtype=int)
    cnt_zeros = pd.Series(dtype=int)
    size = df.shape[0]
    num_columns = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
    for col in num_columns:
        is_zeros = df[col] == 0
        if is_zeros.any():
            dfs_na[col] = df[is_zeros]
            cnt_zeros[col] = dfs_na[col].shape[0]
    if cnt_zeros.empty:
        if df_name:
            return
        print('There are no zeros values')
    else:
        if df_name:
            caption = f'Zeros in "{df_name}"'
        else:
            caption = 'Zeros'
        display(
            cnt_zeros.apply(lambda x: f"{x} ({(x / size):.2%})")
            .to_frame()
            .style.set_caption(caption)
            .set_table_styles(
                [
                    {
                        "selector": "caption",
                        "props": [
                            ("font-size", "18px"),
                            ("text-align", "left"),
                            ("font-weight", "bold"),
                        ],
                    }
                ]
            )
            .hide(axis="columns")
        )
    if not df_name:
        return dfs_na


def get_zeros_proportion_by_category(
    df: pd.DataFrame, column_with_zeros: str, category_column: str = None
):
    """
    Return a DataFrame with the proportion of zeros value for each category.

    Parameters:
    df (pd.DataFrame): Input DataFrame
    column_with_zeros (str): Column with zeros value
    category_column (str): Category column

    Returns:
    None
    """

    # Create a mask to select rows with outliers in the specified column
    mask = df[column_with_zeros] == 0
    size = df[column_with_zeros].size
    if category_column:
        # Group by category and count the number of rows with outliers
        zeros_counts = (
            df[mask].groupby(category_column).size().reset_index(name="zeros_count")
        )
        summ_zeros_counts = zeros_counts["zeros_count"].sum()
        # Get the total count for each category
        total_counts = (
            df.groupby(category_column).size().reset_index(name="total_count")
        )

        # Merge the two DataFrames to calculate the proportion of zeross
        result_df = pd.merge(zeros_counts, total_counts, on=category_column)
        result_df["zeros_in_category_pct"] = (
            result_df["zeros_count"] / result_df["total_count"]
        ).apply(lambda x: f"{x:.1%}")
        result_df["zeros_in_column_pct"] = (
            result_df["zeros_count"] / summ_zeros_counts
        ).apply(lambda x: f"{x:.1%}")
        result_df["total_count_pct"] = (result_df["total_count"] / size).apply(
            lambda x: f"{x:.1%}"
        )
        display(
            result_df[
                [
                    category_column,
                    "total_count",
                    "zeros_count",
                    "zeros_in_category_pct",
                    "zeros_in_column_pct",
                    "total_count_pct",
                ]
            ]
            .style.set_caption(
                f'zeros in "{column_with_zeros}" by category "{category_column}"'
            )
            .set_table_styles(
                [
                    {
                        "selector": "caption",
                        "props": [
                            ("font-size", "18px"),
                            ("text-align", "left"),
                            ("font-weight", "bold"),
                        ],
                    }
                ]
            )
            .hide(axis="index")
        )
        yield
    else:
        categroy_columns = [
            col for col in df.columns if pd.api.types.is_categorical_dtype(df[col])
        ]
        for category_column in categroy_columns:
            # Group by category and count the number of rows with zeross
            zeros_counts = (
                df[mask].groupby(category_column).size().reset_index(name="zeros_count")
            )
            summ_zeros_counts = zeros_counts["zeros_count"].sum()
            # Get the total count for each category
            total_counts = (
                df.groupby(category_column).size().reset_index(name="total_count")
            )

            # Merge the two DataFrames to calculate the proportion of zeross
            result_df = pd.merge(zeros_counts, total_counts, on=category_column)
            result_df["zeros_in_category_pct"] = (
                result_df["zeros_count"] / result_df["total_count"]
            ).apply(lambda x: f"{x:.1%}")
            result_df["zeros_in_column_pct"] = (
                result_df["zeros_count"] / summ_zeros_counts
            ).apply(lambda x: f"{x:.1%}")
            result_df["total_count_pct"] = (result_df["total_count"] / size).apply(
                lambda x: f"{x:.1%}"
            )
            display(
                result_df[
                    [
                        category_column,
                        "total_count",
                        "zeros_count",
                        "zeros_in_category_pct",
                        "zeros_in_column_pct",
                        "total_count_pct",
                    ]
                ]
                .style.set_caption(
                    f'zeros in "{column_with_zeros}" by category "{category_column}"'
                )
                .set_table_styles(
                    [
                        {
                            "selector": "caption",
                            "props": [
                                ("font-size", "18px"),
                                ("text-align", "left"),
                                ("font-weight", "bold"),
                            ],
                        }
                    ]
                )
                .hide(axis="index")
            )
            yield


def zeros_by_category_gen(df, series_zeros):
    """
    Генератор.
    Для каждой колонки в series_zeros функция выводит выборку датафрейма с нулевыми значениями.
    И затем выводит информацию об нулевых значениях по каждой категории в таблице.
    """
    for col in series_zeros.index:
        print(f"Value counts zeros")
        display(
            series_zeros[col][col]
            .value_counts()
            .to_frame("zeros")
            .head(10)
            .style.set_caption(f"{col}")
            .set_table_styles(
                [
                    {
                        "selector": "caption",
                        "props": [
                            ("font-size", "18px"),
                            ("text-align", "left"),
                            ("font-weight", "bold"),
                        ],
                    }
                ]
            )
        )
        yield
        display(
            series_zeros[col]
            .sample(10)
            .style.set_caption(f"Sample zeros in {col}")
            .set_table_styles(
                [
                    {
                        "selector": "caption",
                        "props": [
                            ("font-size", "18px"),
                            ("text-align", "left"),
                            ("font-weight", "bold"),
                        ],
                    }
                ]
            )
        )
        yield
        gen = get_zeros_proportion_by_category(df, col)
        for _ in gen:
            yield


def merge_duplicates(df, duplicate_column, merge_functions):
    """
    Объединяет дубли в датафрейме по указанной колонке с помощью функций из словаря.

    Parameters:
    df (pd.DataFrame): датафрейм для объединения дублей
    duplicate_column (str): название колонки с дублями
    merge_functions (dict): словарь с функциями для объединения, где ключ - название колонки, а значение - функция для объединения

    Returns:
    pd.DataFrame: датафрейм с объединенными дублями
    """
    return df.groupby(duplicate_column, as_index=False).agg(merge_functions)


def create_category_column(
    column,
    method="custom_intervals",
    labels=None,
    n_intervals=None,
    bins=None,
    right=True,
    fillnavalue=None
):
    """
    Create a new category column based on the chosen method.

    Parameters:
    - column (pandas Series): input column
    - method (str, optional): either 'custom_intervals' or 'quantiles' (default is 'custom_intervals')
    - labels (list, optional): list of labels for future categories (default is None)
    - n_intervals (int, optional): number of intervals for 'custom_intervals' or 'quantiles' method (default is len(labels) + 1)
    - bins (list, optional): list of bins for pd.cut function (default is None). The length of `bins` should be `len(labels) + 1`.
    - right (bool, optional): Whether to include the rightmost edge or not. Default is True.
    - fillnavalue (str): Value for fill na. If None , then not filling na.

    Returns:
    - pandas Series: new category column (categorical type pandas)

    Example:
    ```
    # Create a sample dataframe
    df = pd.DataFrame({'values': np.random.rand(100)})

    # Create a category column using custom intervals
    category_column = create_category_column(df['values'], method='custom_intervals', labels=['low', 'medium', 'high'], n_intervals=3)
    df['category'] = category_column

    # Create a category column using quantiles
    category_column = create_category_column(df['values'], method='quantiles', labels=['Q1', 'Q2', 'Q3', 'Q4'], n_intervals=4)
    df['category_quantile'] = category_column
    ```
    """
    if method == "custom_intervals":
        if bins is None:
            if n_intervals is None:
                # default number of intervals
                n_intervals = len(labels) + 1 if labels is not None else 10
            # Calculate равные интервалы
            intervals = np.linspace(column.min(), column.max(), n_intervals)
            if labels is None:
                category_column = pd.cut(column, bins=intervals, right=right)
            else:
                category_column = pd.cut(
                    column, bins=intervals, labels=labels, right=right
                )
        else:
            if labels is None:
                category_column = pd.cut(column, bins=bins, right=right)
            else:
                category_column = pd.cut(column, bins=bins, labels=labels, right=right)
    elif method == "quantiles":
        if n_intervals is None:
            # default number of intervals
            n_intervals = len(labels) if labels is not None else 10
        if labels is None:
            category_column = pd.qcut(column.rank(method="first"), q=n_intervals)
        else:
            category_column = pd.qcut(
                column.rank(method="first"), q=n_intervals, labels=labels
            )
    else:
        raise ValueError(
            "Invalid method. Choose either 'custom_intervals' or 'quantiles'."
        )
    if fillnavalue:
        category_column = category_column.cat.add_categories([fillnavalue])
        return category_column.fillna(fillnavalue).astype("category")
    else:
        return category_column.astype("category")


def lemmatize_column(column):
    """
    Лемматизация столбца с текстовыми сообщениями.

    Parameters:
    column (pd.Series): Колонка для лемматизации.

    Returns:
    pd.Series: Лемматизированная колонка в виде строки.
    """
    m = Mystem()  # Создаем экземпляр Mystem внутри функции

    def lemmatize_text(text):
        """Приведение текста к леммам с помощью библиотеки Mystem."""
        if not text:
            return ""

        try:
            lemmas = m.lemmatize(text)
            return " ".join(lemmas)
        except Exception as e:
            print(f"Ошибка при лемматизации текста: {e}")
            return ""

    return column.map(lemmatize_text)


def categorize_column_by_lemmatize(
    column: pd.Series, categorization_dict: dict, use_cache: bool = False
):
    """
    Категоризация столбца с помощью лемматизации.

    Parameters:
    column (pd.Series): Столбец для категоризации.
    categorization_dict (dict): Словарь для категоризации, где ключи - категории, а значения - списки лемм.
    use_cache (bool): Если истина, то  результат будет сохранен в кэше. Нужно учитывать, что если данных будет много,
    то память может переполниться. default (False)

    Returns:
    pd.Series: Категоризированный столбец. (categorical type pandas)

    Пример использования:
    ```
    # Создайте образец dataframe
    data = {'text': ['This is a sample text', 'Another example text', 'This is a test']}
    df = pd.DataFrame(data)

    # Определите словарь категоризации
    categorization_dict = {
        'Sample': ['sample', 'example'],
        'Test': ['test']
    }

    # Вызовите функцию
    categorized_column = categorize_column_by_lemmatize(df['text'], categorization_dict)

    print(categorized_column)
    ```
    """
    if column.empty:
        return pd.Series([])

    m = Mystem()
    buffer = dict()

    def lemmatize_text(text):
        try:
            if use_cache:
                if text in buffer:
                    return buffer[text]
                else:
                    lemas = m.lemmatize(text)
                    buffer[text] = lemas
                    return lemas
            else:
                return m.lemmatize(text)
        except Exception as e:
            print(f"Ошибка при лемматизации текста: {e}")
            return []

    def categorize_text(lemmas):
        for category, category_lemmas in categorization_dict.items():
            if set(lemmas) & set(category_lemmas):
                return category
        return "Unknown"

    lemmatized_column = column.map(lemmatize_text)
    return lemmatized_column.map(categorize_text).astype("category")


def target_encoding_linear(df, category_col, value_col, func="mean", alpha=0.1):
    """
    Функция для target encoding.

    Parameters:
    df (pd.DataFrame): Датафрейм с данными.
    category_col (str): Название колонки с категориями.
    value_col (str): Название колонки со значениями.
    func (callable or str): Функция для target encoding (может быть строкой, например "mean", или вызываемой функцией, которая возвращает одно число).
    alpha (float, optional): Параметр регуляризации. Defaults to 0.1.

    Returns:
    pd.Series: Колонка с target encoding.

    Используется линейная регуляризация, x * (1 - alpha) + alpha * np.mean(x)
    Она основана на идее о том, что среднее значение по группе нужно сгладить, добавляя к нему часть среднего значения по всей таблице.
    """
    available_funcs = {"median", "mean", "max", "min", "std", "count"}

    if isinstance(func, str):
        if func not in available_funcs:
            raise ValueError(f"Unknown function: {func}")
        # Если func является строкой, используйте соответствующий метод pandas
        encoding = df.groupby(category_col)[value_col].agg(func)
    else:
        # Если func является вызываемым, примените его к каждой группе значений
        encoding = df.groupby(category_col)[value_col].apply(func)

    # Добавляем линейную регуляризацию
    def regularize(x, alpha=alpha):
        return x * (1 - alpha) + alpha * np.mean(x)

    encoding_reg = encoding.apply(regularize)

    # Заменяем категории на средние значения
    encoded_col = df[category_col].map(encoding_reg.to_dict())

    return encoded_col


def target_encoding_bayes(df, category_col, value_col, func="mean", reg_group_size=100):
    """
    Функция для target encoding с использованием байесовского метода регуляризации.

    Parameters:
    df (pd.DataFrame): Датафрейм с данными.
    category_col (str): Название колонки с категориями.
    value_col (str): Название колонки со значениями.
    func (callable or str): Функция для target encoding (может быть строкой, например "mean", или вызываемой функцией, которая возвращает одно число).
    reg_group_size (int, optional): Размер группы регуляризации. Defaults to 10.

    Returns:
    pd.Series: Колонка с target encoding.

    Эта функция использует байесовский метод регуляризации, который основан на идее о том,
    что среднее значение по группе нужно сгладить, добавляя к нему часть среднего значения по всей таблице,
    а также учитывая дисперсию значений в группе.
    """
    if reg_group_size <= 0:
        raise ValueError("reg_group_size must be a positive integer")

    available_funcs = {"median", "mean", "max", "min", "std", "count"}

    if isinstance(func, str):
        if func not in available_funcs:
            raise ValueError(f"Unknown function: {func}")
        # Если func является строкой, используйте соответствующий метод pandas
        encoding = df.groupby(category_col)[value_col].agg(
            func_val=(func), count=("count")
        )
    else:
        # Если func является вызываемым, примените его к каждой группе значений
        encoding = df.groupby(category_col)[value_col].agg(
            func_val=(func), count=("count")
        )

    global_mean = df[value_col].mean()
    # Добавляем байесовскую регуляризацию
    encoding_reg = (
        encoding["func_val"] * encoding["count"] + global_mean * reg_group_size
    ) / (encoding["count"] + reg_group_size)

    # Заменяем категории на средние значения
    encoded_col = df[category_col].map(encoding_reg.to_dict())

    return encoded_col


def check_duplicated_value_in_df(df):
    """
    Функция проверяет на дубли столбцы датафрейма и выводит количество дублей в каждом столбце
    """
    cnt_duplicated = pd.Series(dtype=int)
    size = df.shape[0]
    for col in df.columns:
        is_duplicated = df[col].duplicated()
        if is_duplicated.any():
            cnt_duplicated[col] = df[is_duplicated].shape[0]
    display(
        cnt_duplicated.apply(lambda x: f"{x} ({(x / size):.2%})")
        .to_frame()
        .style.set_caption("Duplicates")
        .set_table_styles(
            [
                {
                    "selector": "caption",
                    "props": [
                        ("font-size", "18px"),
                        ("text-align", "left"),
                        ("font-weight", "bold"),
                    ],
                }
            ]
        )
        .hide(axis="columns")
    )


def check_negative_value_in_df(df):
    """
    Функция проверяет на негативные значения числовые столбцы датафрейма и выводит количество отрицательных значений
    """
    size = df.shape[0]
    num_columns = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
    df_num_columns = df[num_columns]
    negative = (df_num_columns < 0).sum()
    display(
        negative[negative != 0]
        .apply(lambda x: f"{x} ({(x / size):.1%})")
        .to_frame(name="negative")
    )


def check_zeros_value_in_df(df):
    """
    Функция проверяет на нулевые значения числовые столбцы датафрейма и выводит количество нулевых значений
    """
    size = df.shape[0]
    num_columns = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
    df_num_columns = df[num_columns]
    zeros = (df_num_columns == 0).sum()
    display(
        zeros[zeros != 0]
        .apply(lambda x: f"{x} ({(x / size):.1%})")
        .to_frame(name="zeros")
    )


def check_missed_value_in_df(df):
    """
    Функция проверяет на пропуски датафрейме и выводит количество пропущенных значений
    """
    size = df.shape[0]
    missed = df.isna().sum()
    display(
        missed[missed != 0]
        .apply(lambda x: f"{x} ({(x / size):.1%})")
        .to_frame(name="missed")
    )


def normalize_string_series(column: pd.Series, symbols: list=None) -> pd.Series:
    """
    Normalize a pandas Series of strings by removing excess whitespace, trimming leading and trailing whitespace,
    and converting all words to lowercase.

    Args:
        column (pd.Series): The input Series of strings to normalize
        symbols (list): List of symbols to remove from the input strings
            Example: symbols = ['.', ',', '«', '»', '(', ')', '"', "'", "`"]
    Returns:
        pd.Series: The normalized Series of strings
    """
    if not isinstance(column, pd.Series):
        raise ValueError("Input must be a pandas Series")
    if not isinstance(column.dropna().iloc[0], str):
        raise ValueError("Series must contain strings")
    res = column.copy()
    if symbols:
        for symbol in symbols:
            res = res.str.replace(symbol, " ")
    res = res.str.lower().str.strip().str.replace(r"\s+", " ", regex=True)        
    if isinstance(column.dtype, pd.CategoricalDtype):
        res = res.astype('category')
    return res


def analyze_filtered_df_by_category(
    df: pd.DataFrame,
    df_for_analys: pd.DataFrame,
    column_for_analys: str,
    is_dash: bool = False,
):
    """
    Show statisctic column by categories in DataFrame

    Parameters:
    df (pd.DataFrame): origin DataFrame
    df_for_analys (pd.DataFrame): DataFrame for analysis

    Returns:
    None
    """
    size_all = df.shape[0]
    category_columns = [
        col for col in df.columns if pd.api.types.is_categorical_dtype(df[col])
    ]
    for category_column in category_columns:
        analys_df = (
            df_for_analys.groupby(category_column, observed=False, dropna=False)
            .size()
            .reset_index(name="count")
        )
        summ_counts = analys_df["count"].sum()
        all_df = (
            df.groupby(category_column, observed=False, dropna=False).size().reset_index(name="total")
        )
        result_df = pd.merge(analys_df, all_df, on=category_column)
        result_df["count_in_total_pct"] = result_df["count"] / result_df["total"]
        result_df["count_in_sum_count_pct"] = result_df["count"] / summ_counts
        result_df["total_in_sum_total_pct"] = result_df["total"] / size_all
        result_df["diff_sum_pct"] = (
            result_df["count_in_sum_count_pct"] - result_df["total_in_sum_total_pct"]
        )
        if is_dash:
            result_df = result_df[
                [
                    category_column,
                    "total",
                    "count",
                    "count_in_total_pct",
                    "count_in_sum_count_pct",
                    "total_in_sum_total_pct",
                    "diff_sum_pct",
                ]
            ]
            for col in result_df.columns:
                if pd.api.types.is_float_dtype(result_df[col]):
                    result_df[col] = result_df[col].apply(lambda x: f"{x:.1%}")
            caption = f'Value in "{column_for_analys}" by category "{category_column}"'
            yield caption, 'by_category', column_for_analys, category_column, result_df

        else:
            display(
                result_df[
                    [
                        category_column,
                        "total",
                        "count",
                        "count_in_total_pct",
                        "count_in_sum_count_pct",
                        "total_in_sum_total_pct",
                        "diff_sum_pct",
                    ]
                ]
                .style.set_caption(
                    f'Value in "{column_for_analys}" by category "{category_column}"'
                )
                .set_table_styles(
                    [
                        {
                            "selector": "caption",
                            "props": [
                                ("font-size", "18px"),
                                ("text-align", "left"),
                                ("font-weight", "bold"),
                            ],
                        }
                    ]
                )
                .format(
                    "{:.1%}",
                    subset=[
                        "count_in_total_pct",
                        "count_in_sum_count_pct",
                        "total_in_sum_total_pct",
                        "diff_sum_pct",
                    ],
                )
                .hide(axis="index")
            )
            yield


def analyze_by_category_gen(df, series_for_analys, is_dash=False):
    """
    Генератор.
    Для каждой колонки в series_for_analys функция выводит выборку датафрейма.
    И затем выводит информацию по каждой категории в таблице.

    is_dash (bool):  режим работы в Dash или нет

    """
    df_size = df.shape[0]
    for col in series_for_analys.index:
        cnt_for_display_in_sample = series_for_analys[col].shape[0] / df_size
        if not series_for_analys[col][col].value_counts().empty:
            if is_dash:
                caption = f"Value counts ({cnt_for_display_in_sample:.2%})"
                yield caption, 'value_counts', col, None, series_for_analys[col][col].value_counts().reset_index().head(10)
            else:
                print(f"Value counts ({cnt_for_display_in_sample:.2%})")
                display(
                    series_for_analys[col][col]
                    .value_counts()
                    .to_frame("count")
                    .head(10)
                    .style.set_caption(f"{col}")
                    .set_table_styles(
                        [
                            {
                                "selector": "caption",
                                "props": [
                                    ("font-size", "18px"),
                                    ("text-align", "left"),
                                    ("font-weight", "bold"),
                                ],
                            }
                        ]
                    )
                )
                yield
                yield series_for_analys[col].sort_values(col, ascending=False).head(10)
        if is_dash:
            caption  = f"Sample in {col} ({cnt_for_display_in_sample:.2%})"
            yield caption, 'sample', col, None, series_for_analys[col].sort_values(col, ascending=False).head(10)
        else: 
            display(
                series_for_analys[col]
                .sort_values(col, ascending=False)
                .head(10)
                .style.set_caption(f"Sample in {col} ({cnt_for_display_in_sample:.2%})")
                .set_table_styles(
                    [
                        {
                            "selector": "caption",
                            "props": [
                                ("font-size", "18px"),
                                ("text-align", "left"),
                                ("font-weight", "bold"),
                            ],
                        }
                    ]
                )
            )
            yield
        if is_dash:
            gen = analyze_filtered_df_by_category(
                df, series_for_analys[col], col, is_dash=True
            )
        else:
            gen = analyze_filtered_df_by_category(df, series_for_analys[col], col)
        for _ in gen:
            if is_dash:
                yield _
            else:
                yield


def check_group_count(df, category_columns, value_column):
    """
    Функция выводит информацию о количестве элементов в группах.
    Это функция нужна для  проверки того, что количество элементов в группах соответствует ожидаемому
    для заполнения пропусков через группы.
    """
    temp = (
        df.groupby(category_columns, observed=False)[value_column]
        .agg(lambda x: 1 if x.isna().sum() else -1)
        .dropna()
    )
    # -1 это группы без пропусков
    group_with_miss = (temp != -1).sum() / temp.size
    print(f"{group_with_miss:.2%} groups have missing values")
    # Посмотрим какой процент групп с пропусками имеют больше 30 элементов
    temp = (
        df.groupby(category_columns, observed=False)[value_column]
        .agg(lambda x: x.count() > 30 if x.isna().sum() else -1)
        .dropna()
    )
    temp = temp[temp != -1]
    group_with_more_30_elements = (temp == True).sum() / temp.size
    print(
        f"{group_with_more_30_elements:.2%}  groups with missings have more than 30 elements"
    )
    # Посмотрим какой процент групп с пропусками имеют больше 10 элементов
    temp = (
        df.groupby(category_columns, observed=False)[value_column]
        .agg(lambda x: x.count() > 10 if x.isna().sum() else -1)
        .dropna()
    )
    temp = temp[temp != -1]
    group_with_more_10_elements = (temp == True).sum() / temp.size
    print(
        f"{group_with_more_10_elements:.2%}  groups with missings have more than 10 elements"
    )
    # Посмотрим какой процент групп с пропусками имеют больше 5 элементов
    temp = (
        df.groupby(category_columns, observed=False)[value_column]
        .agg(lambda x: x.count() > 5 if x.isna().sum() else -1)
        .dropna()
    )
    temp = temp[temp != -1]
    group_with_more_5_elements = (temp == True).sum() / temp.size
    print(
        f"{group_with_more_5_elements:.2%}  groups with missings have more than 5 elements"
    )
    # Посмотрим какой процент групп содержат только NA
    temp = (
        df.groupby(category_columns, observed=False)[value_column]
        .agg(lambda x: x.count() if x.isna().sum() else -1)
        .dropna()
    )
    temp = temp[temp != -1]
    group_with_ontly_missings = (temp == 0).sum() / temp.size
    print(f"{group_with_ontly_missings:.2%}  groups have only missings")
    # Посмотрим сколько всего значений в группах, где только прпоуски
    temp = (
        df.groupby(category_columns, observed=False)[value_column]
        .agg(lambda x: -1 if x.count() else x.isna().sum())
        .dropna()
    )
    temp = temp[temp != -1]
    missing_cnt = temp.sum()
    print(f"{missing_cnt:.0f} missings in groups with only missings")


def fill_na_with_function_by_categories(
    df, category_columns, value_column, func="median", minimal_group_size=10
):
    """
    Fills missing values in the value_column with the result of the func function,
    grouping by the category_columns.

    Parameters:
    - df (pandas.DataFrame): DataFrame to fill missing values
    - category_columns (list): list of column names to group by
    - value_column (str): name of the column to fill missing values
    - func (callable or str): function to use for filling missing values
    (can be a string, e.g. "mean", or a callable function that returns a single number)
    - minimal_group_size (int): Minimal group size for fills missings.
    Returns:
    - pd.Series: Modified column with filled missing values
    """
    if not all(col in df.columns for col in category_columns):
        raise ValueError("Invalid category column(s). Column must be in df")
    if value_column not in df.columns:
        raise ValueError("Invalid value column. Column must be in df")

    available_funcs = {"median", "mean", "max", "min"}

    if isinstance(func, str):
        if func not in available_funcs:
            raise ValueError(f"Unknown function: {func}")
        # If func is a string, use the corresponding pandas method
        return df.groupby(category_columns, observed=False)[value_column].transform(
            lambda x: x.fillna(x.apply(func)) if x.count() >= minimal_group_size else x
        )
    else:
        # If func is a callable, apply it to each group of values
        return df.groupby(category_columns, observed=False)[value_column].transform(
            lambda x: x.fillna(func(x)) if x.count() >= minimal_group_size else x
        )


def quantiles_columns(column: pd.Series, quantiles=[0.05, 0.25, 0.5, 0.75, 0.95]):
    max_ = pretty_value(column.max())
    column_summary = pd.DataFrame({"Max": [max_]})
    for quantile in quantiles[::-1]:
        column_summary[f"{quantile * 100:.0f}"] = pretty_value(
            column.quantile(quantile)
        )
    min_ = pretty_value(column.min())
    column_summary["Min"] = min_
    display(
        column_summary.T.reset_index()
        .style.set_caption(f"Quantiles")
        .set_table_styles([{"selector": "caption", "props": [("font-size", "15px")]}])
        .set_properties(**{"text-align": "left"})
        .hide(axis="columns")
        .hide(axis="index")
    )


def top_n_values_gen(
    df: pd.DataFrame, value_column: str, n: int = 10, threshold: int = 20, func="sum"
):
    """
    Возвращает топ n значений в категориальных столбцах df, где значений больше 20, по значению в столбце value_column.

    Parameters:
    df (pd.DataFrame): Датасет.
    column (str): Название столбца, который нужно проанализировать.
    n (int): Количество топ значений, которые нужно вернуть.
    value_column (str): Название столбца, по которому нужно рассчитать топ значения.
    threshold (int, optional): Количество уникальных значений, при котором нужно рассчитать топ значения. Defaults to 20.
    func (calable): Функция для аггрегации в столбце value_column

    Returns:
    pd.DataFrame: Топ n значений в столбце column по значению в столбце value_column.
    """
    # Проверяем, есть ли в столбце больше 20 уникальных значений
    categroy_columns = [
        col for col in df.columns if pd.api.types.is_categorical_dtype(df[col])
    ]
    for column in categroy_columns:
        if df[column].nunique() > threshold:
            # Группируем данные по столбцу column и рассчитываем сумму по столбцу value_column
            display(
                df.groupby(column)[value_column]
                .agg(func)
                .sort_values(ascending=False)
                .head(n)
                .to_frame()
                .reset_index()
                .style.set_caption(f'Top in "{column}"')
                .set_table_styles(
                    [
                        {
                            "selector": "caption",
                            "props": [
                                ("font-size", "18px"),
                                ("text-align", "left"),
                                ("font-weight", "bold"),
                            ],
                        }
                    ]
                )
                .format("{:.2f}", subset=value_column)
                .hide(axis="index")
            )
            yield

def analyze_share_by_category(
    df: pd.DataFrame,
    df_for_analys: pd.DataFrame,
    column_for_analys: str,
    category_column: str
):
    """
    Show statisctic column by categories in DataFrame

    Parameters:
    df (pd.DataFrame): origin DataFrame
    df_for_analys (pd.DataFrame): DataFrame for analysis

    Returns:
    None
    """
    size_all = df.shape[0]
    analys_df = (
        df_for_analys.groupby(category_column, observed=False, dropna=False)
        .size()
        .reset_index(name="count")
    )
    summ_counts = analys_df["count"].sum()
    all_df = (
        df.groupby(category_column, observed=False, dropna=False).size().reset_index(name="total")
    )
    result_df = pd.merge(analys_df, all_df, on=category_column)
    result_df["count_in_total_pct"] = result_df["count"] / result_df["total"]
    result_df["count_in_sum_count_pct"] = result_df["count"] / summ_counts
    result_df["total_in_sum_total_pct"] = result_df["total"] / size_all
    result_df["diff_sum_pct"] = (
        result_df["count_in_sum_count_pct"] - result_df["total_in_sum_total_pct"]
    )
    display(
        result_df[
            [
                category_column,
                "total",
                "count",
                "count_in_total_pct",
                "count_in_sum_count_pct",
                "total_in_sum_total_pct",
                "diff_sum_pct",
            ]
        ]
        .style.set_caption(
            f'Share in "{column_for_analys}" by category "{category_column}"'
        )
        .set_table_styles(
            [
                {
                    "selector": "caption",
                    "props": [
                        ("font-size", "18px"),
                        ("text-align", "left"),
                        ("font-weight", "bold"),
                    ],
                }
            ]
        )
        .format(
            "{:.1%}",
            subset=[
                "count_in_total_pct",
                "count_in_sum_count_pct",
                "total_in_sum_total_pct",
                "diff_sum_pct",
            ],
        )
        .hide(axis="index")
    )


def analyze_anomaly_by_category(df, series_for_analys, mode, col=None, category=None):
    """
    Для каждой колонки в series_for_analys функция выводит выборку датафрейма.
    И затем выводит информацию по каждой категории в таблице.

    df - исходный датафрейм
    series_for_analys - series c датафреймами, которые нужно проанализировать по категориям
    col - колонка, по которой будет проводиться анализ
    category - категория, по которой будет проводиться анализ
    mode - режим, в котором будет проводиться анализ (value_counts, sample, by_category)
    """
    df_size = df.shape[0]
    cnt_for_display_in_sample = series_for_analys[col].shape[0] / df_size
    if mode == 'value_counts':
        display(f"Value counts in {col} ({cnt_for_display_in_sample:.2%})")
        display(
            series_for_analys[col][col]
            .value_counts()
            .to_frame("count")
            .head(10)
            .style.set_caption(f"{col}")
            .set_table_styles(
                [
                    {
                        "selector": "caption",
                        "props": [
                            ("font-size", "18px"),
                            ("text-align", "left"),
                            ("font-weight", "bold"),
                        ],
                    }
                ]
            )
        )
        return
    if mode == 'sample':
        display(
            series_for_analys[col]
            .sort_values(col, ascending=False)            
            .head(10)
            .style.set_caption(f"Sample in {col} ({cnt_for_display_in_sample:.2%})")
            .set_table_styles(
                [
                    {
                        "selector": "caption",
                        "props": [
                            ("font-size", "18px"),
                            ("text-align", "left"),
                            ("font-weight", "bold"),
                        ],
                    }
                ]
            )
        )
        return
    if mode == 'by_category':
        analyze_share_by_category(df, series_for_analys[col], col, category)
        
def value_counts_table(df, column, chunk_size=10, tables_in_row=5):
    """
    Генерирует таблицу с подсчетом уникальных значений и их пропорций для указанного столбца DataFrame.
 
    Параметры:
    df : pandas.DataFrame
        Входной DataFrame, содержащий данные для анализа.
   
    column : str
        Имя столбца, для которого необходимо подсчитать уникальные значения и их пропорции.
    
    chunk_size : int, по умолчанию 10
        Количество строк, отображаемых в каждой отдельной таблице (чанке). Если общее количество уникальных значений 
        превышает произведение chunk_size и tables_in_row, выводится предупреждение.

    tables_in_row : int, по умолчанию 5
        Максимальное количество таблиц, отображаемых в одной строке.

    Возвращает:
    pandas.io.formats.style.Styler
    """
    # Подсчет значений и пропорций
    if chunk_size * tables_in_row < df[column].nunique():
        print("Недостаточны размер chunk_size и tables_in_row")
        print("Всего значений в столбце:", df[column].nunique())
        print('Текущее значение chunk_size:', chunk_size)
        print('Текущее значение tables_in_row:', tables_in_row)
    result = pd.concat([df[column].value_counts(), df[column].value_counts(normalize=True)], axis=1).reset_index()
    result.columns = ['name', 'count', 'proportion']
    
    # Формирование колонки 'share'
    result['share'] = result['count'].apply(lambda x: pretty_value(x)) + ' (' + (result['proportion'] * 100).round(2).astype(str) + '%)'
    result.drop(['count', 'proportion'], axis=1, inplace=True)
    
    # Разделение результата на чанки
    row_for_html = [result[i:i + chunk_size].reset_index(drop=True).rename(columns={'name': f'name_{i}', 'share': f'share_{i}'})
                    for i in range(0, result.shape[0], chunk_size)]
    row_for_html = row_for_html[:tables_in_row]
    # Конкатенация чанков в одну таблицу
    res_df = pd.concat(row_for_html, axis=1)
    
    # Добавление пустых колонок для форматирования
    # for j in range(2, 2*tables_in_row, 3):
    #     res_df.insert(j, f'{j}', "|")
    # Заполнение NaN значений пустыми строками
    
        # Добавление разделителей
    for j in range(1, tables_in_row):
        # Индекс для вставки разделителя
        insert_index = j * 3 - 1
        if insert_index < 2 + tables_in_row * 3 and insert_index < 2 + (len(row_for_html) - 1)* 3:  # Проверка, чтобы не выйти за пределы
            res_df.insert(insert_index, f'{j}', "|")
            
    # res_df = res_df.fillna('')
    
    # Настройка стиля таблицы
    table_style = [
        {
            "selector": "caption",
            "props": [
                ("font-size", "16px"),
                ("text-align", "left"),
                ("font-weight", "bold"),
            ],
        }                 
    ]
    
    # Применение стилей к таблице
    styled_res_df = (res_df
        .style.set_caption(f'value counts for "{column}"')
        .set_table_styles(table_style)
        .set_properties(**{"text-align": "left"})
        .hide(axis="columns")
        .hide(axis="index")
        .format(na_rep='')
    )
    
    return styled_res_df

def check_na_combinations_gen(df, n=2):
    """
    Функция считает пропуски между всеми возможными комбинациями между столбцами.
    Сначала для проверки на дубли берутся пары столбцов.
    Затем по 3 столбца. И так все возможные комибнации.
    Можно выбрать до какого количества комбинаций двигаться.
    n - максимальное возможное количество столбцов в комбинациях. По умолчанию беруться все столбцы
    """
    if n < 2:
        return
    c2 = itertools.combinations(df.columns, 2)
    dupl_df_c2 = pd.DataFrame([], index=df.columns, columns=df.columns)
    df_size = df.shape[0]
    # print(f"Group by 2 columns")
    for c in c2:
        missings = df[list(c)].isna().all(axis=1).sum()
        col1_missings = df[c[0]].isna().sum()
        col2_missings = df[c[1]].isna().sum()
        if missings == 0:
            dupl_df_c2.loc[c[0], c[1]] = f''
        else:
            dupl_df_c2.loc[c[1], c[0]] = f'< {(missings / col2_missings):.1%} / ^ {(missings / col1_missings):.1%}'
    display(
        dupl_df_c2.fillna("")
        .style.set_caption("Missings in both columns")
        .set_table_styles(
            [
                {
                    "selector": "caption",
                    "props": [
                        ("font-size", "18px"),
                        ("text-align", "left"),
                        ("font-weight", "bold"),
                    ],
                }
            ]
        )
    )
    yield
    if n < 3:
        return
    c3 = itertools.combinations(df.columns, 3)
    dupl_c3_list = []
    # print(f"Group by 3 columns")
    for c in c3:
        missings = df[list(c)].isna().all(axis=1).sum()
        if missings:
            missings = f'{pretty_value(missings)} ({(missings / df_size):.1%} of all)' if missings / df_size >= 0.01 else f'{pretty_value(missings)} < 1% of all'
            dupl_c3_list.append([" | ".join(c), missings])
    dupl_df_c3 = pd.DataFrame(dupl_c3_list)
    # display(dupl_df_c3)
    # разобьем таблицу на 3 части, чтобы удобнее читать
    yield (
        pd.concat(
            [
                # part_df.reset_index(drop=True)
                # for part_df in np.array_split(dupl_df_c3, 3)
                dupl_df_c3[i:i + 10].reset_index(drop=True)
                    for i in range(0, dupl_df_c3.shape[0], 10)
            ],
            axis=1,
        )
        # .style.format({1: "{:.0f}"}, na_rep="")
        .style.format(na_rep="")
        .set_caption("Missings simultaneously in 3 columns")
        .set_table_styles(
            [
                {
                    "selector": "caption",
                    "props": [
                        ("font-size", "18px"),
                        ("text-align", "left"),
                        ("font-weight", "bold"),
                    ],
                }
            ]  
        )      
        .hide(axis="index")
        .hide(axis="columns")
    )
    if n < 4:
        return
    for col_n in range(4, df.columns.size + 1):
        # print(f"Group by {col_n} columns")
        cn = itertools.combinations(df.columns, col_n)
        dupl_cn_list = []
        for c in cn:
            missings = df[list(c)].isna().all(axis=1).sum()
            if missings:
                missings = f'{pretty_value(missings)} ({(missings / df_size):.1%})' if missings / df_size >= 0.01 else f'{pretty_value(missings)} < 1%'
                missings = f'{pretty_value(missings)} ({(missings / df_size):.1%})' if missings / df_size >= 0.01 else f'{pretty_value(missings)} < 1%'
                dupl_cn_list.append([" | ".join(c), missings])
        dupl_df_cn = pd.DataFrame(dupl_cn_list)
        # разобьем таблицу на 3 части, чтобы удобнее читать
        yield (
            pd.concat(
                [
                    part_df.reset_index(drop=True)
                    for part_df in np.array_split(dupl_df_cn, 2)
                ],
                axis=1,
            )
            # .style.format({1: "{:.0f}"}, na_rep="")
            .style.format(na_rep="")
            .set_caption(f"Missings simultaneously in {col_n} columns")
            .set_table_styles(
                [
                    {
                        "selector": "caption",
                        "props": [
                            ("font-size", "18px"),
                            ("text-align", "left"),
                            ("font-weight", "bold"),
                        ],
                    }
                ]      
            )      
            .hide(axis="index")
            .hide(axis="columns")
        )
        if n < col_n + 1:
            return

def detect_df_relationship(left_df, right_df, on=None, left_on=None, right_on=None):
    """
    Detects the relationship between two DataFrames based on specified columns.

    Parameters:
    left_df (pd.DataFrame): The left DataFrame.
    right_df (pd.DataFrame): The right DataFrame.
    on (str): The column name to check for relationships in both DataFrames.
    left_on (str): The column name in left_df to check for relationships.
    right_on (str): The column name in right_df to check for relationships.

    Returns:
    str: A description of the relationship type.
    """

    # Check if DataFrames are empty
    if left_df.empty or right_df.empty:
        return "One or both DataFrames are empty."

    # Determine the columns to use for counting
    if on is not None:
        left_on = on
        right_on = on

    # Check if left_on and right_on are provided
    if left_on is None or right_on is None:
        return "Please provide either 'on' or both 'left_on' and 'right_on'."

    # Count occurrences of each value in the specified columns
    left_counts = left_df[left_on].value_counts()
    right_counts = right_df[right_on].value_counts()

    # Determine the maximum counts
    max_left = left_counts.max()
    max_right = right_counts.max()

    # Determine the relationship type
    if max_left == 1 and max_right == 1:
        return "one-to-one"
    elif max_left == 1:
        return "one-to-many"
    elif max_right == 1:
        return "many-to-one"
    else:
        return "many-to-many"

def count_df_mismatches_by_key(left_df, right_df, on=None, left_on=None, right_on=None, return_diffs=False):
    """
    Compares values in the specified field of two DataFrames and returns the count
    of mismatched values.

    Parameters:
    left_df (pd.DataFrame): The left DataFrame for comparison.
    right_df (pd.DataFrame): The right DataFrame for comparison.
    on (str): The column name to check for relationships in both DataFrames.
    left_on (str): The column name in left_df to check for relationships.
    right_on (str): The column name in right_df to check for relationships.

    Returns:
    if return_diffs is True
    tuple (mismatches_left_dif_right, count_right_dif_left_mismatches)
    """
    # Check if DataFrames are empty
    if left_df.empty or right_df.empty:
        return "One or both DataFrames are empty."

    # Determine the columns to use for counting
    if on is not None:
        left_on = on
        right_on = on

    # Check if left_on and right_on are provided
    if left_on is None or right_on is None:
        return "Please provide either 'on' or both 'left_on' and 'right_on'."
    left_key_missting = left_df[left_on].isna().sum()
    right_key_missting = right_df[right_on].isna().sum()
    if left_key_missting:
        print(f'В левом датафрейме в ключе есть пропуски: {left_key_missting}')
    if right_key_missting:
        print(f'В правом датафрейме в ключе есть пропуски: {right_key_missting}')
    left_values = set(left_df[left_on].dropna())
    right_values = set(right_df[right_on].dropna())

    mismatches_left_dif_right = left_values - right_values
    mismatches_right_dif_left = right_values - left_values
    count_left_dif_right_mismatches = len(mismatches_left_dif_right)
    count_right_dif_left_mismatches = len(mismatches_right_dif_left)
    print('Строки в левой таблице, отсутствующие в правой: ', count_left_dif_right_mismatches)
    print('Строки в правой таблице, отсутствующие в левой: ', count_right_dif_left_mismatches)
    if return_diffs:
        return mismatches_left_dif_right, count_right_dif_left_mismatches

def haversine(lat1, lon1, lat2, lon2):
    """
    Calculate the great-circle distance between two points
    on the Earth's surface given their latitude and longitude in decimal degrees.

    Parameters:
    lat1 : float
        Latitude of the first point in decimal degrees.
    lon1 : float
        Longitude of the first point in decimal degrees.
    lat2 : float
        Latitude of the second point in decimal degrees.
    lon2 : float
        Longitude of the second point in decimal degrees.

    Returns:
    distance : float
        The great-circle distance between the two points in kilometers.
    """

    # Convert decimal degrees to radians
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])

    # Haversine formula components
    dlat = lat2 - lat1  # Difference in latitude
    dlon = lon2 - lon1  # Difference in longitude

    # Haversine formula calculation
    a = math.sin(dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2
    c = 2 * math.asin(math.sqrt(a))  # Central angle in radians

    r = 6371  # Radius of Earth in kilometers
    return c * r  # Distance in kilometers

def haversine_vectorized(lat1, lon1, lat2, lon2):
    """
    Calculate the great-circle distance between two points
    on the Earth's surface given their latitude and longitude in decimal degrees.

    Parameters:
    lat1 : array-like
        Latitude of the first point in decimal degrees.
    lon1 : array-like
        Longitude of the first point in decimal degrees.
    lat2 : array-like
        Latitude of the second point in decimal degrees.
    lon2 : array-like
        Longitude of the second point in decimal degrees.

    Returns:
    distance : array-like
        The great-circle distance between the two points in kilometers.
    """

    # Convert decimal degrees to radians
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

    # Haversine formula components
    dlat = lat2 - lat1  # Difference in latitude
    dlon = lon2 - lon1  # Difference in longitude

    # Haversine formula calculation
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    c = 2 * np.arcsin(np.sqrt(a))  # Central angle in radians

    r = 6371  # Radius of Earth in kilometers
    return c * r  # Distance in kilometers

def df_summary(dataframes: list | pd.DataFrame):
    # Create a list to hold the information
    info = []
    if isinstance(dataframes, pd.DataFrame):
        dataframes = [dataframes]
    # List of DataFrames and their names
    names = [f'DataFrame {i+1}' for i in range(len(dataframes))]

    for df, name in zip(dataframes, names):
        rows = df.shape[0]
        rows = f"{rows:,}".replace(',', ' ')
        cols = df.shape[1]
        ram = df.memory_usage(deep=True).sum() / (1024 ** 2)  # Convert bytes to MB and round
        duplicates = df.duplicated().sum()

        if duplicates == 0:
            duplicates = "---"
        else:
            dupl = duplicates
            duplicates = format_number(duplicates)
            duplicates_pct = dupl * 100 / df.shape[0]
            if 0 < duplicates_pct < 1:
                duplicates_pct = "<1"
            elif duplicates_pct > 99 and duplicates_pct < 100:
                duplicates_pct = round(duplicates_pct, 1)
                if duplicates_pct == 100:
                    duplicates_pct = 99.9
            else:
                duplicates_pct = round(duplicates_pct)
            duplicates = f"{duplicates} ({duplicates_pct}%)"
        info.append({
            'DataFrame': name,
            'Rows': rows,
            'Cols': cols,
            'RAM (Mb)': ram,
            'Duplicates': duplicates
        })

    # Create a DataFrame from the info list
    result_df = pd.DataFrame(info)
    return (
        result_df.style
        .set_caption("DataFrame info")
        .set_table_styles(
            [
                {
                    "selector": "caption",
                    "props": [
                        ("font-size", "16px"),
                        ("text-align", "left"),
                        ("font-weight", "bold"),
                    ],
                }
            ]
        )
        .format('{:.2f}', subset='RAM (Mb)')
        # .set_properties(**{"text-align": "left"})
        # .hide(axis="columns")
        .hide(axis="index")
    )

def column_info_short(column):
    mean_ = format_number(column.mean())
    mode_ = column.mode()
    if mode_.size == 1:
        mode_ = format_number(mode_[0])
    else:
        mode_ = [format_number(mode_el) for mode_el in mode_]
    q_75 = format_number(column.quantile(0.75))
    median_ = format_number(column.median())
    q_25 = format_number(column.quantile(0.25))
    column_summary = pd.DataFrame(
        {
            "Mean": [mean_],
            "Mode": [mode_],
            "75%": [q_75],
            "Median": [median_],
            "25%": [q_25],
        }
    )
    result_df =  column_summary.T.reset_index()
    return (
        result_df.style
        .set_caption("Column info")
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
        # .format('{:.2f}', subset='RAM (Mb)')
        # .set_properties(**{"text-align": "left"})
        # .hide(axis="columns")
        .hide(axis="index")
        .hide(axis="columns")
    )
