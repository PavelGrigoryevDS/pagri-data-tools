import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from ipywidgets import widgets, Layout
from IPython.display import display, display_html, display_markdown
from tqdm.auto import tqdm
import re
import itertools
from pymystem3 import Mystem
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import plotly.io as pio
import plotly.graph_objects as go
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import statsmodels.stats.api as stm
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from termcolor import colored
import scipy.stats as stats
import pingouin as pg
import warnings
import nbformat
from nbformat import v4
import json
from pyaspeller import YandexSpeller
# from nltk.tokenize import word_tokenize

colorway_for_line = ['rgb(127, 60, 141)', 'rgb(17, 165, 121)', 'rgb(231, 63, 116)',
                     '#03A9F4', 'rgb(242, 183, 1)', '#8B9467', '#FFA07A', '#005A5B', '#66CCCC', '#B690C4', 'rgb(127, 60, 141)', 'rgb(17, 165, 121)', 'rgb(231, 63, 116)',
                     '#03A9F4', 'rgb(242, 183, 1)', '#8B9467', '#FFA07A', '#005A5B', '#66CCCC', '#B690C4']
colorway_for_bar = ['rgba(128, 60, 170, 0.9)', '#049CB3', "rgba(112, 155, 219, 0.9)", "rgba(99, 113, 156, 0.9)", '#5c6bc0', '#B690C4', 'rgba(17, 100, 120, 0.9)', 'rgba(194, 143, 113, 0.8)', '#B690C4', '#03A9F4', '#8B9467', '#a771f2', 'rgba(102, 204, 204, 0.9)', 'rgba(168, 70, 90, 0.9)', 'rgba(50, 152, 103, 0.8)', '#8F7A7A', 'rgba(156, 130, 217, 0.9)'
                    ]
# colorway_for_bar = ['rgba(128, 60, 170, 0.9)', '#049CB3', '#84a9e9', '#B690C4',
#                     '#5c6bc0', '#005A5B', '#63719C', '#03A9F4', '#66CCCC', '#a771f2', 'rgba(128, 60, 170, 0.9)', '#049CB3', '#84a9e9', '#B690C4',
#                     '#5c6bc0', '#005A5B', '#63719C', '#03A9F4', '#66CCCC', '#a771f2', 'rgba(128, 60, 170, 0.9)', '#049CB3', '#84a9e9', '#B690C4',
#                     '#5c6bc0', '#005A5B', '#63719C', '#03A9F4', '#66CCCC', '#a771f2', 'rgba(128, 60, 170, 0.9)', '#049CB3', '#84a9e9', '#B690C4',
#                     '#5c6bc0', '#005A5B', '#63719C', '#03A9F4', '#66CCCC', '#a771f2']
colorway_for_treemap = [
    'rgba(148, 100, 170, 1)',
    'rgba(50, 156, 179, 1)',
    'rgba(99, 113, 156, 1)',
    'rgba(92, 107, 192, 1)',
    'rgba(0, 90, 91, 1)',
    'rgba(3, 169, 244, 1)',
    'rgba(217, 119, 136, 1)',
    'rgba(64, 134, 87, 1)',
    'rgba(134, 96, 147, 1)',
    'rgba(132, 169, 233, 1)']
# default setting for Plotly
# for line plot
pio.templates["custom_theme_for_line"] = go.layout.Template(
    layout=go.Layout(
        colorway=colorway_for_line
    )
)
# pio.templates.default = 'simple_white+custom_theme_for_line'
# for bar plot
pio.templates["custom_theme_for_bar"] = go.layout.Template(
    layout=go.Layout(
        colorway=colorway_for_bar
    )
)
pio.templates.default = 'simple_white+custom_theme_for_bar'

# default setting for Plotly express
px.defaults.template = "simple_white"
px.defaults.color_continuous_scale = color_continuous_scale = [
    [0, 'rgba(0.018, 0.79, 0.703, 1.0)'],
    [0.5, 'rgba(64, 120, 200, 0.9)'],
    [1, 'rgba(128, 60, 170, 0.9)']
]
# px.defaults.color_discrete_sequence = colorway_for_line
px.defaults.color_discrete_sequence = colorway_for_bar
# px.defaults.color_discrete_sequence =  px.colors.qualitative.Bold
# px.defaults.width = 500
# px.defaults.height = 300


def plotly_default_settings(fig):
    # Segoe UI Light
    fig.update_layout(
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


def pretty_value(value):
    '''
    Функция делает удобное представление числа с пробелами после разрядов
    '''
    if value == 0:
        return 0
    if value > 0:
        part1 = int(value % 1e3) if value % 1e3 != 0 else ''
        part2 = f'{(value // 1e3) % 1e3:.0f} ' if value // 1e3 != 0 else ''
        part3 = f'{(value // 1e6) % 1e3:.0f} ' if int((value // 1e6) %
                                                      1e3) != 0 else ''
        part4 = f'{(value // 1e9) % 1e3:.0f} ' if int((value // 1e9) %
                                                      1e3) != 0 else ''
        part5 = f'{(value // 1e12) % 1e3:.0f} ' if int((value // 1e12) %
                                                       1e3) != 0 else ''
        return f'{part5}{part4}{part3}{part2}{part1}'
    else:
        value = abs(value)
        part1 = int(value % 1e3) if value % 1e3 != 0 else ''
        part2 = f'{(value // 1e3) % 1e3:.0f} ' if value // 1e3 != 0 else ''
        part3 = f'{(value // 1e6) % 1e3:.0f} ' if int((value // 1e6) %
                                                      1e3) != 0 else ''
        part4 = f'{(value // 1e9) % 1e3:.0f} ' if int((value // 1e9) %
                                                      1e3) != 0 else ''
        part5 = f'{(value // 1e12) % 1e3:.0f} ' if int((value // 1e12) %
                                                       1e3) != 0 else ''
        return f'-{part5}{part4}{part3}{part2}{part1}'


def make_widget_all_frame(df):
    dupl = df.duplicated().sum()
    duplicates = dupl
    if duplicates == 0:
        duplicates = '---'
    else:
        duplicates = pretty_value(duplicates)
        duplicates_pct = dupl * 100 / df.shape[0]
        if 0 < duplicates_pct < 1:
            duplicates_pct = '<1'
        elif duplicates_pct > 99 and duplicates_pct < 100:
            duplicates_pct = round(duplicates_pct, 1)
            if duplicates_pct == 100:
                duplicates_pct = 99.9
        else:
            duplicates_pct = round(duplicates_pct)
        duplicates = f'{duplicates} ({duplicates_pct}%)'
    dupl_keep_false = df.duplicated(keep=False).sum()
    dupl_sub = df.apply(lambda x: x.str.lower().str.strip().str.replace(
        r'\s+', ' ', regex=True) if x.dtype == 'object' else x).duplicated(keep=False).sum()
    duplicates_sub_minis_origin = pretty_value(dupl_sub - dupl_keep_false)
    duplicates_sub_minis_origin_pct = (
        dupl_sub - dupl_keep_false) * 100 / dupl
    if 0 < duplicates_sub_minis_origin_pct < 1:
        duplicates_sub_minis_origin_pct = '<1'
    elif (duplicates_sub_minis_origin_pct > 99 and duplicates_sub_minis_origin_pct < 100):
        duplicates_sub_minis_origin_pct = round(
            duplicates_sub_minis_origin_pct, 1)
    else:
        duplicates_sub_minis_origin_pct = round(
            duplicates_sub_minis_origin_pct)
    duplicates_sub_minis_origin = f'{duplicates_sub_minis_origin} ({duplicates_sub_minis_origin_pct}%)'
    all_rows = pd.DataFrame({
        'Rows': [pretty_value(df.shape[0])], 'Features': [df.shape[1]], 'RAM (Mb)': [round(df.__sizeof__() / 1_048_576)], 'Duplicates': [duplicates], 'Dupl (sub - origin)': [duplicates_sub_minis_origin]
    })
    # widget_DataFrame = widgets.Output()
    # with widget_DataFrame:
    #      display_markdown('**DataFrame**', raw=True)
    widget_all_frame = widgets.Output()
    with widget_all_frame:
        # display_html('<h4>DataFrame</h4>', raw=True, height=3)
        display(all_rows.style
                .set_caption('DataFrame')
                .set_table_styles([{'selector': 'caption',
                                    'props': [('font-size', '18px'), ("text-align", "left"), ("font-weight", "bold")]
                                    }])
                .set_properties(**{'text-align': 'left'})
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
        ram = '<1 Mb'
    column_summary = pd.DataFrame({
        'First date': [fist_date], 'Last date': [last_date], 'RAM (Mb)': [ram]
    })
    widget_summary = widgets.Output()
    with widget_summary:
        # display_html(f'<h4>{column_name}</h4>', raw=True)
        display(column_summary.T.reset_index().style
                .set_caption(f'{column_name}')
                .set_table_styles([{'selector': 'caption',
                                    'props': [('font-size', '16px'), ("text-align", "left"), ("font-weight", "bold")]
                                    }])
                .set_properties(**{'text-align': 'left'})
                .hide(axis="columns")
                .hide(axis="index")
                )
    return widget_summary


def make_widget_summary_date(column):
    column_name = column.name
    values = column.count()
    values = pretty_value(column.count())
    values_pct = column.count() * 100 / column.size
    if 0 < values_pct < 1:
        values_pct = '<1'
    elif values_pct > 99 and values_pct < 100:
        values_pct = round(values_pct, 1)
        if values_pct == 100:
            values_pct = 99.9
    else:
        values_pct = round(values_pct)
    values = f'{values} ({values_pct}%)'
    missing = column.isna().sum()
    if missing == 0:
        missing = '---'
    else:
        missing = pretty_value(column.isna().sum())
        missing_pct = round(column.isna().sum() * 100 / column.size)
        if missing_pct == 0:
            missing_pct = '<1'
        missing = f'{missing} ({missing_pct}%)'
    distinct = pretty_value(column.nunique())
    distinct_pct = column.nunique() * 100 / column.size
    if distinct_pct > 99 and distinct_pct < 100:
        distinct_pct = round(distinct_pct, 1)
        if distinct_pct == 100:
            distinct_pct = 99.9
    else:
        distinct_pct = round(distinct_pct)
    if distinct_pct == 0:
        distinct_pct = '<1'
    distinct = f'{distinct} ({distinct_pct}%)'
    duplicates = column.duplicated().sum()
    if duplicates == 0:
        duplicates = '---'
    else:
        duplicates = pretty_value(duplicates)
        duplicates_pct = column.duplicated().sum() * 100 / column.size
        if 0 < duplicates_pct < 1:
            duplicates_pct = '<1'
        elif duplicates_pct > 99 and duplicates_pct < 100:
            duplicates_pct = round(duplicates_pct, 1)
            if duplicates_pct == 100:
                duplicates_pct = 99.9
        else:
            duplicates_pct = round(duplicates_pct)
        duplicates = f'{duplicates} ({duplicates_pct}%)'
    column_summary = pd.DataFrame({
        'Values': [values], 'Missing': [missing], 'Distinct': [distinct], 'Duplicates': [duplicates]
    })
    widget_summary = widgets.Output()
    with widget_summary:
        # display_html(f'<h4>{column_name}</h4>', raw=True)
        display(column_summary.T.reset_index().style
                # .set_caption(f'{column_name}')
                .set_table_styles([{'selector': 'caption',
                                    'props': [('font-size', '16px'), ("text-align", "left"), ("font-weight", "bold")]
                                    }])
                .set_properties(**{'text-align': 'left'})
                .hide(axis="columns")
                .hide(axis="index")
                )
    return widget_summary


def make_widget_check_missing_date(column):
    column_name = column.name
    fist_date = column.min()
    last_date = column.max()
    date_range = pd.date_range(start=fist_date, end=last_date, freq='D')
    years = date_range.year.unique()
    years_missed_pct = (~years.isin(column.dt.year.unique())
                        ).sum() * 100 / years.size
    if 0 < years_missed_pct < 1:
        years_missed_pct = '<1'
    elif years_missed_pct > 99:
        years_missed_pct = round(years_missed_pct, 1)
    else:
        years_missed_pct = round(years_missed_pct)
    months = date_range.to_period("M").unique()
    months_missed_pct = (~months.isin(column.dt.to_period(
        "M").unique())).sum() * 100 / months.size
    if 0 < months_missed_pct < 1:
        months_missed_pct = '<1'
    elif months_missed_pct > 99:
        months_missed_pct = round(months_missed_pct, 1)
    else:
        months_missed_pct = round(months_missed_pct)
    weeks = date_range.to_period("W").unique()
    weeks_missed_pct = (~weeks.isin(column.dt.to_period(
        "W").unique())).sum() * 100 / weeks.size
    if 0 < weeks_missed_pct < 1:
        weeks_missed_pct = '<1'
    elif weeks_missed_pct > 99:
        weeks_missed_pct = round(weeks_missed_pct, 1)
    else:
        weeks_missed_pct = round(weeks_missed_pct)
    days = date_range.unique().to_period("D")
    days_missed_pct = (~days.isin(column.dt.to_period(
        "D").unique())).sum() * 100 / days.size
    if 0 < days_missed_pct < 1:
        days_missed_pct = '<1'
    elif days_missed_pct > 99:
        days_missed_pct = round(days_missed_pct, 1)
    else:
        days_missed_pct = round(days_missed_pct)

    column_summary = pd.DataFrame({
        'Years missing': [f'{years_missed_pct}%'], 'Months missing': [f'{months_missed_pct}%'], 'Weeks missing': [f'{weeks_missed_pct}%'], 'Days missing': [f'{days_missed_pct}%']
    })
    widget_summary = widgets.Output()
    with widget_summary:
        # display_html(f'<h4>{column_name}</h4>', raw=True)
        display(column_summary.T.reset_index().style
                # .set_caption(f'{column_name}')
                .set_table_styles([{'selector': 'caption',
                                    'props': [('font-size', '16px'), ("text-align", "left"), ("font-weight", "bold")]
                                    }])
                .set_properties(**{'text-align': 'left'})
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
        values_pct = '<1'
    elif values_pct > 99 and values_pct < 100:
        values_pct = round(values_pct, 1)
        if values_pct == 100:
            values_pct = 99.9
    else:
        values_pct = round(values_pct)
    values = f'{values} ({values_pct}%)'
    missing = column.isna().sum()
    if missing == 0:
        missing = '---'
    else:
        missing = pretty_value(column.isna().sum())
        missing_pct = round(column.isna().sum() * 100 / column.size)
        if missing_pct == 0:
            missing_pct = '<1'
        missing = f'{missing} ({missing_pct}%)'
    distinct = pretty_value(column.nunique())
    distinct_pct = column.nunique() * 100 / column.size
    if distinct_pct > 99 and distinct_pct < 100:
        distinct_pct = round(distinct_pct, 1)
        if distinct_pct == 100:
            distinct_pct = 99.9
    else:
        distinct_pct = round(distinct_pct)
    if distinct_pct == 0:
        distinct_pct = '<1'
    distinct = f'{distinct} ({distinct_pct}%)'
    zeros = ((column == 0) | (column == '')).sum()
    if zeros == 0:
        zeros = '---'
    else:
        zeros = pretty_value(((column == 0) | (column == '')).sum())
        zeros_pct = round(((column == 0) | (column == '')
                           ).sum() * 100 / column.size)
        if zeros_pct == 0:
            zeros_pct = '<1'
        zeros = f'{zeros} ({zeros_pct}%)'
    negative = (column < 0).sum()
    if negative == 0:
        negative = '---'
    else:
        negative = pretty_value(negative)
        negative_pct = round((column < 0).sum() * 100 / column.size)
        if negative_pct == 0:
            negative_pct = '<1'
        negative = f'{negative} ({negative_pct}%)'
    duplicates = column.duplicated().sum()
    if duplicates == 0:
        duplicates = '---'
    else:
        duplicates = pretty_value(duplicates)
        duplicates_pct = column.duplicated().sum() * 100 / column.size
        if 0 < duplicates_pct < 1:
            duplicates_pct = '<1'
        elif duplicates_pct > 99 and duplicates_pct < 100:
            duplicates_pct = round(duplicates_pct, 1)
            if duplicates_pct == 100:
                duplicates_pct = 99.9
        else:
            duplicates_pct = round(duplicates_pct)
        duplicates = f'{duplicates} ({duplicates_pct}%)'
    ram = round(column.__sizeof__() / 1_048_576)
    if ram == 0:
        ram = '<1 Mb'
    column_summary = pd.DataFrame({
        'Values': [values], 'Missing': [missing], 'Distinct': [distinct], 'Duplicates': [duplicates], 'Zeros': [zeros], 'Negative': [negative], 'RAM (Mb)': [ram]
    })
    widget_summary = widgets.Output()
    with widget_summary:
        # display_html(f'<h4>{column_name}</h4>', raw=True)
        display(column_summary.T.reset_index().style
                .set_caption(f'{column_name}')
                .set_table_styles([{'selector': 'caption',
                                    'props': [('font-size', '16px'), ("text-align", "left"), ("font-weight", "bold")]
                                    }])
                .set_properties(**{'text-align': 'left'})
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
    column_summary = pd.DataFrame({
        'Max': [max_], '95%': [q_95], '75%': [q_75], 'Median': [median_], '25%': [q_25], '5%': [q_5], 'Min': [min_]
    })
    widget_pct = widgets.Output()
    with widget_pct:
        display(column_summary.T.reset_index().style
                .set_caption(f'')
                .set_table_styles([{'selector': 'caption',
                                    'props': [('font-size', '15px')]
                                    }])
                .set_properties(**{'text-align': 'left'})
                .hide(axis="columns")
                .hide(axis="index")
                )
    return widget_pct


def make_widget_std(column):
    avg_ = pretty_value(column.mean())
    mode_ = column.mode()
    if mode_.size > 1:
        mode_ = '---'
    else:
        mode_ = pretty_value(mode_.iloc[0])
    range_ = pretty_value(column.max() - column.min())
    iQR = pretty_value(column.quantile(0.75) - column.quantile(0.25))
    std = pretty_value(column.std())
    kurt = f'{column.kurtosis():.2f}'
    skew = f'{column.skew():.2f}'
    column_summary = pd.DataFrame({
        'Avg': [avg_], 'Mode': [mode_], 'Range': [range_], 'iQR': [iQR], 'std': [std], 'kurt': [kurt], 'skew': [skew]
    })
    widget_std = widgets.Output()
    with widget_std:
        display(column_summary.T.reset_index().style
                .set_caption(f'')
                .set_table_styles([{'selector': 'caption',
                                    'props': [('font-size', '15px')]
                                    }])
                .set_properties(**{'text-align': 'left'})
                .hide(axis="columns")
                .hide(axis="index")
                )
    return widget_std


def make_widget_value_counts(column):
    column_name = column.name
    val_cnt = column.value_counts().iloc[:7]
    val_cnt_norm = column.value_counts(normalize=True).iloc[:7]
    column_name_pct = column_name + '_pct'
    val_cnt_norm.name = column_name_pct

    def make_value_counts_row(x):
        if x[column_name_pct] < 0.01:
            pct_str = '<1%'
        else:
            pct_str = f'({x[column_name_pct]:.0%})'
        return f'{x[column_name]:.0f} {pct_str}'
    top_5 = pd.concat([val_cnt, val_cnt_norm], axis=1).reset_index().apply(
        make_value_counts_row, axis=1).to_frame()
    widget_value_counts = widgets.Output()
    with widget_value_counts:
        display(top_5.style
                # .set_caption(f'Value counts top')
                .set_table_styles([{'selector': 'caption',
                                    'props': [('font-size', '16px'), ("text-align", "left")]
                                    }])
                .set_properties(**{'text-align': 'left'})
                .hide(axis="columns")
                .hide(axis="index")
                )
    # widget_value_counts.
    return widget_value_counts


def num_trunc(x):
    '''
    Функция обрезает порядок числа и доабвляет К или М
    '''
    if abs(x) < 1e3:
        if round(abs(x), 2) > 0:
            return f'{x:.2f}'
        else:
            return '0'
    if abs(x) < 1e6:
        return f'{x / 1e3:.0f}K'
    if abs(x) < 1e9:
        return f'{x / 1e6:.0f}M'


def make_widget_hist(column):
    fig, ax = plt.subplots(figsize=(3.2, 2.2))
    sns.histplot(column, bins=20,  stat='percent', ax=ax, color='#9370db')

    ax.set(ylabel='', xlabel='')
    ax.locator_params(axis='x', nbins=5)
    bins = ax.get_xticks()
    vect = np.vectorize(num_trunc)
    bins = bins[(bins >= column.min()) & (bins <= column.max())]
    ax.set_xticks(ticks=bins, labels=vect(bins))
    plt.xticks(alpha=.9)
    plt.yticks(alpha=.9)
    plt.gca().spines['top'].set_alpha(0.3)
    plt.gca().spines['left'].set_alpha(0.3)
    plt.gca().spines['right'].set_alpha(0.3)
    plt.gca().spines['bottom'].set_alpha(0.3)
    plt.close()
    widget_hist = widgets.Output()
    # fig.tight_layout()
    with widget_hist:
        display(fig)
    return widget_hist


def make_widget_hist_plotly(column):
    fig = px.histogram(column, nbins=20, histnorm='percent',
                       template="simple_white", height=250, width=370)
    fig.update_traces(marker_color='rgba(128, 60, 170, 0.9)', text=f'*',
                      textfont=dict(color='rgba(128, 60, 170, 0.9)'))
    fig.update_layout(
        margin=dict(l=0, r=10, b=0, t=10), showlegend=False, hoverlabel=dict(
            bgcolor="white",
        ), xaxis_title="", yaxis_title=""
    )
    # fig.layout.yaxis.visible = False
    widget_hist = widgets.Output()
    with widget_hist:
        fig.show(config=dict(displayModeBar=False), renderer="png")
    return widget_hist


def make_widget_violin(column):
    fig, ax = plt.subplots(figsize=(2, 2.44))
    sns.violinplot(column, ax=ax, color='#9370db')
    ax.set(ylabel='', xlabel='')
    # ax.tick_params(right= False,top= False,left= False, bottom= False)
    # plt.axis('off')
    ax.set_xticks([])
    ax.set_yticks([])
    plt.gca().spines['top'].set_alpha(0.3)
    plt.gca().spines['left'].set_alpha(0.3)
    plt.gca().spines['right'].set_alpha(0.3)
    plt.gca().spines['bottom'].set_alpha(0.3)
    plt.close()
    widget_violin = widgets.Output()
    with widget_violin:
        display(fig)
    return widget_violin


def make_widget_violine_plotly(column):
    fig = px.violin(column, template="simple_white", height=250, width=300)
    fig.update_traces(marker_color='rgba(128, 60, 170, 0.9)')
    fig.update_layout(
        margin=dict(l=20, r=20, b=0, t=10), showlegend=False, hoverlabel=dict(
            bgcolor="white",
        ), xaxis_title="", yaxis_title="", xaxis=dict(ticks='', showticklabels=False)
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
        values_pct = '<1'
    elif values_pct > 99 and values_pct < 100:
        values_pct = round(values_pct, 1)
        if values_pct == 100:
            values_pct = 99.9
    else:
        values_pct = round(values_pct)
    values = f'{values} ({values_pct}%)'
    missing = column.isna().sum()
    if missing == 0:
        missing = '---'
    else:
        missing = pretty_value(column.isna().sum())
        missing_pct = round(column.isna().sum() * 100 / column.size)
        if missing_pct == 0:
            missing_pct = '<1'
        missing = f'{missing} ({missing_pct}%)'
    distinct = pretty_value(column.nunique())
    distinct_pct = column.nunique() * 100 / column.size
    if distinct_pct > 99 and distinct_pct < 100:
        distinct_pct = round(distinct_pct, 1)
        if distinct_pct == 100:
            distinct_pct = 99.9
    else:
        distinct_pct = round(distinct_pct)
    if distinct_pct == 0:
        distinct_pct = '<1'
    distinct = f'{distinct} ({distinct_pct}%)'
    zeros = ((column == 0) | (column == '')).sum()
    if zeros == 0:
        zeros = '---'
    else:
        zeros = pretty_value(((column == 0) | (column == '')).sum())
        zeros_pct = round(((column == 0) | (column == '')
                           ).sum() * 100 / column.size)
        if zeros_pct == 0:
            zeros_pct = '<1'
        zeros = f'{zeros} ({zeros_pct}%)'
    duplicates = column.duplicated().sum()
    if duplicates == 0:
        duplicates = '---'
        duplicates_sub_minis_origin = '---'
    else:
        duplicates = pretty_value(duplicates)
        duplicates_pct = column.duplicated().sum() * 100 / column.size
        if 0 < duplicates_pct < 1:
            duplicates_pct = '<1'
        elif duplicates_pct > 99 and duplicates_pct < 100:
            duplicates_pct = round(duplicates_pct, 1)
            if duplicates_pct == 100:
                duplicates_pct = 99.9
        else:
            duplicates_pct = round(duplicates_pct)
        duplicates = f'{duplicates} ({duplicates_pct}%)'
        duplicates_keep_false = column.duplicated(keep=False).sum()
        duplicates_sub = column.str.lower().str.strip().str.replace(
            r'\s+', ' ', regex=True).duplicated(keep=False).sum()
        duplicates_sub_minis_origin = duplicates_sub - duplicates_keep_false
        if duplicates_sub_minis_origin == 0:
            duplicates_sub_minis_origin = '---'
        else:
            duplicates_sub_minis_origin = pretty_value(
                duplicates_sub_minis_origin)
            duplicates_sub_minis_origin_pct = (
                duplicates_sub - duplicates_keep_false) * 100 / duplicates_sub
            if 0 < duplicates_sub_minis_origin_pct < 1:
                duplicates_sub_minis_origin_pct = '<1'
            elif (duplicates_sub_minis_origin_pct > 99 and duplicates_sub_minis_origin_pct < 100) \
                    or duplicates_sub_minis_origin_pct < 1:
                duplicates_sub_minis_origin_pct = round(
                    duplicates_sub_minis_origin_pct, 1)
            else:
                duplicates_sub_minis_origin_pct = round(
                    duplicates_sub_minis_origin_pct)
            duplicates_sub_minis_origin = f'{duplicates_sub_minis_origin} ({duplicates_sub_minis_origin_pct}%)'

    ram = round(column.__sizeof__() / 1_048_576)
    if ram == 0:
        ram = '<1 Mb'
    column_summary = pd.DataFrame({
        'Values': [values], 'Missing': [missing], 'Distinct': [distinct], 'Duplicated origin': [duplicates], 'Dupl (modify - origin)': [duplicates_sub_minis_origin], 'Empty': [zeros], 'RAM (Mb)': [ram]
    })
    widget_summary_obj = widgets.Output()
    with widget_summary_obj:
        # display_html(f'<h4>{column_name}</h4>', raw=True)
        display(column_summary.T.reset_index().style
                .set_caption(f'{column_name}')
                .set_table_styles([{'selector': 'caption',
                                    'props': [('font-size', '16px'), ("text-align", "left"), ("font-weight", "bold")]
                                    }])
                .set_properties(**{'text-align': 'left'})
                .hide(axis="columns")
                .hide(axis="index")
                )
    return widget_summary_obj


def make_widget_value_counts_obj(column):
    column_name = column.name
    val_cnt = column.value_counts().iloc[:8]
    val_cnt_norm = column.value_counts(normalize=True).iloc[:8]
    column_name_pct = column_name + '_pct'
    val_cnt_norm.name = column_name_pct

    def make_value_counts_row(x):
        if x[column_name_pct] < 0.01:
            pct_str = '<1%'
        else:
            pct_str = f'({x[column_name_pct]:.0%})'
        return f'{x[column_name]:.0f} {pct_str}'
    top_5 = pd.concat([val_cnt, val_cnt_norm], axis=1).apply(
        make_value_counts_row, axis=1).reset_index()
    widget_value_counts_obj = widgets.Output()
    with widget_value_counts_obj:
        display(top_5.style
                # .set_caption(f'Value counts top')
                .set_table_styles([{'selector': 'caption',
                                    'props': [('font-size', '16px'), ("text-align", "left")]
                                    }])
                .set_properties(**{'text-align': 'left'})
                .hide(axis="columns")
                .hide(axis="index")
                )
    # widget_value_counts.
    return widget_value_counts_obj


def make_widget_bar_obj(column):
    df_fig = column.value_counts(ascending=True).iloc[-10:]
    text_labels = [label[:30] for label in df_fig.index.to_list()]
    fig = px.bar(df_fig, orientation='h',
                 template="simple_white", height=220, width=500)
    fig.update_traces(marker_color='rgba(128, 60, 170, 0.9)')
    fig.update_layout(
        margin=dict(l=0, r=0, b=0, t=5), showlegend=False, hoverlabel=dict(
            bgcolor="white",
        ), xaxis_title="", yaxis_title="")
    fig.update_traces(y=text_labels)
    widget_bar_obj = widgets.Output()
    # fig.tight_layout()
    with widget_bar_obj:
        fig.show(config=dict(displayModeBar=False), renderer="png")
    return widget_bar_obj


def make_hbox(widgets_: list):
    # add some CSS styles to distribute free space
    hbox_layout = Layout(display='flex',
                         flex_flow='row',
                         # justify_content='space-around',
                         width='auto',
                         grid_gap='20px',
                         align_items='flex-end'
                         )
    # create Horisontal Box container
    hbox = widgets.HBox(widgets_, layout=hbox_layout)
    return hbox


def my_info(df, graphs=True, num=True, obj=True, date=True):
    '''
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
    '''
    if not num and not obj and not date:
        return
    vbox_layout = Layout(display='flex',
                         # flex_flow='column',
                         justify_content='space-around',
                         # width='auto',
                         # grid_gap = '20px',
                         # align_items = 'flex-end'
                         )
    funcs_num = [make_widget_summary, make_widget_pct,
                 make_widget_std, make_widget_value_counts]
    func_obj = [make_widget_summary_obj, make_widget_value_counts_obj]
    func_date = [make_widget_range_date,
                 make_widget_summary_date, make_widget_check_missing_date]
    if graphs:
        funcs_num += [make_widget_hist_plotly, make_widget_violine_plotly]
        func_obj += [make_widget_bar_obj]
    boxes = []
    if date:
        date_columns = filter(
            lambda x: pd.api.types.is_datetime64_any_dtype(df[x]), df.columns)
        for column in tqdm(date_columns):
            widgets_ = [func(df[column]) for func in func_date]
            boxes.extend((widgets_))
        layout = widgets.Layout(
            grid_template_columns='1fr 1fr 1fr 1fr 1fr')
        date_grid = widgets.GridBox(boxes, layout=layout)
    boxes = []
    if num:
        num_columns = filter(
            lambda x: pd.api.types.is_numeric_dtype(df[x]), df.columns)
        for column in tqdm(num_columns):
            widgets_ = [func(df[column]) for func in funcs_num]
            boxes.extend((widgets_))
        if graphs:
            layout = widgets.Layout(
                grid_template_columns='auto auto auto auto auto auto')
        else:
            layout = widgets.Layout(
                grid_template_columns='repeat(4, 0.2fr)')
        num_grid = widgets.GridBox(boxes, layout=layout)
    boxes = []
    if obj:
        obj_columns = filter(
            lambda x: not pd.api.types.is_numeric_dtype(df[x]) and not pd.api.types.is_datetime64_any_dtype(df[x]), df.columns)
        for column in tqdm(obj_columns):
            widgets_ = [func(df[column]) for func in func_obj]
            boxes.extend((widgets_))
        if graphs:
            layout = widgets.Layout(
                grid_template_columns='auto auto auto')
        else:
            layout = widgets.Layout(
                grid_template_columns='repeat(2, 0.3fr)')
        obj_grid = widgets.GridBox(boxes, layout=layout)

    # widgets.Layout(grid_template_columns="200px 200px 200px 200px 200px 200px")))
    display(make_widget_all_frame(df))
    if date:
        display(date_grid)
    if num:
        display(num_grid)
    if obj:
        display(obj_grid)


def my_info_gen(df, graphs=True, num=True, obj=True, date=True):
    '''
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
    '''
    if not num and not obj and not date:
        return
    vbox_layout = Layout(display='flex',
                         # flex_flow='column',
                         justify_content='space-around',
                         # width='auto',
                         # grid_gap = '20px',
                         # align_items = 'flex-end'
                         )
    yield make_widget_all_frame(df)

    funcs_num = [make_widget_summary, make_widget_pct,
                 make_widget_std, make_widget_value_counts]
    func_obj = [make_widget_summary_obj, make_widget_value_counts_obj]
    func_date = [make_widget_range_date,
                 make_widget_summary_date, make_widget_check_missing_date]
    if graphs:
        funcs_num += [make_widget_hist_plotly, make_widget_violine_plotly]
        func_obj += [make_widget_bar_obj]
    if date:
        date_columns = filter(
            lambda x: pd.api.types.is_datetime64_any_dtype(df[x]), df.columns)
        layout = widgets.Layout(
            grid_template_columns='auto auto')
        for column in tqdm(date_columns):
            widgets_ = [func(df[column]) for func in func_date]
            yield widgets.GridBox(widgets_, layout=layout)

    if num:
        num_columns = filter(
            lambda x: pd.api.types.is_numeric_dtype(df[x]), df.columns)
        if graphs:
            layout = widgets.Layout(
                grid_template_columns='auto auto auto auto auto auto')
        else:
            layout = widgets.Layout(
                grid_template_columns='repeat(4, 0.2fr)')
        for column in tqdm(num_columns):
            widgets_ = [func(df[column]) for func in funcs_num]
            yield widgets.GridBox(widgets_, layout=layout)

    if obj:
        obj_columns = filter(
            lambda x: not pd.api.types.is_numeric_dtype(df[x]) and not pd.api.types.is_datetime64_any_dtype(df[x]), df.columns)
        if graphs:
            layout = widgets.Layout(
                grid_template_columns='auto auto auto')
        else:
            layout = widgets.Layout(
                grid_template_columns='repeat(2, 0.3fr)')
        for column in tqdm(obj_columns):
            widgets_ = [func(df[column]) for func in func_obj]
            yield widgets.GridBox(widgets_, layout=layout)


def check_duplicated(df):
    '''
    Функция проверяет датафрейм на дубли.  
    Если дубли есть, то возвращает датафрейм с дублями.
    '''
    dupl = df.duplicated().sum()
    size = df.shape[0]
    if dupl == 0:
        return 'no duplicates'
    print(f'Duplicated is {dupl} ({(dupl / size):.1%}) rows')
    # приводим строки к нижнему регистру, удаляем пробелы
    return (df.apply(lambda x: x.str.lower().str.strip().str.replace(r'\s+', ' ', regex=True) if x.dtype == 'object' else x)
            .value_counts(dropna=False)
            .to_frame()
            .sort_values(0, ascending=False)
            .rename(columns={0: 'Count'}))


def find_columns_with_duplicates(df) -> pd.Series:
    '''
    Фукнция проверяет каждый столбец в таблице,  
    если есть дубликаты, то помещает строки исходного 
    дата фрейма с этими дубликатами в Series. 
    Индекс - название колонки. 
    Если нужно соеденить фреймы в один, то используем 
    pd.concat(res.to_list())
    '''
    dfs_duplicated = pd.Series(dtype=int)
    cnt_duplicated = pd.Series(dtype=int)
    size = df.shape[0]
    for col in df.columns:
        is_duplicated = df[col].duplicated()
        if is_duplicated.any():
            dfs_duplicated[col] = df[is_duplicated]
            cnt_duplicated[col] = dfs_duplicated[col].shape[0]
    display(cnt_duplicated.apply(lambda x: f'{x} ({(x / size):.2%})').to_frame().style
            .set_caption('Duplicates')
            .set_table_styles([{'selector': 'caption',
                                'props': [('font-size', '18px'), ("text-align", "left"), ("font-weight", "bold")]}])
            .hide(axis="columns"))
    return dfs_duplicated


def check_duplicated_combinations_gen(df, n=2):
    '''
    Функция считает дубликаты между всеми возможными комбинациями между столбцами.
    Сначала для проверки на дубли берутся пары столбцов.  
    Затем по 3 столбца. И так все возможные комибнации.  
    Можно выбрать до какого количества комбинаций двигаться.
    n - максимальное возможное количество столбцов в комбинациях. По умолчанию беруться все столбцы
    '''
    if n < 2:
        return
    df_copy = df.apply(lambda x: x.str.lower().str.strip().str.replace(
        r'\s+', ' ', regex=True) if x.dtype == 'object' else x)
    c2 = itertools.combinations(df.columns, 2)
    dupl_df_c2 = pd.DataFrame([], index=df.columns, columns=df.columns)
    print(f'Group by 2 columns')
    for c in c2:
        duplicates = df_copy[list(c)].duplicated().sum()
        dupl_df_c2.loc[c[1], c[0]] = duplicates
    display(dupl_df_c2.fillna('').style.set_caption('Duplicates').set_table_styles([{'selector': 'caption',
                                                                                     'props': [('font-size', '18px'), ("text-align", "left"), ("font-weight", "bold")]
                                                                                     }]))
    yield
    if n < 3:
        return
    c3 = itertools.combinations(df.columns, 3)
    dupl_c3_list = []
    print(f'Group by 3 columns')
    for c in c3:
        duplicates = df_copy[list(c)].duplicated().sum()
        if duplicates:
            dupl_c3_list.append([' | '.join(c), duplicates])
    dupl_df_c3 = pd.DataFrame(dupl_c3_list)
    # разобьем таблицу на 3 части, чтобы удобнее читать
    yield (pd.concat([part_df.reset_index(drop=True) for part_df in np.array_split(dupl_df_c3, 3)], axis=1)
           .style.format({1: '{:.0f}'}, na_rep='').hide(axis="index").hide(axis="columns"))
    if n < 4:
        return
    for col_n in range(4, df.columns.size + 1):
        print(f'Group by {col_n} columns')
        cn = itertools.combinations(df.columns, col_n)
        dupl_cn_list = []
        for c in cn:
            duplicates = df_copy[list(c)].duplicated().sum()
            if duplicates:
                dupl_cn_list.append([' | '.join(c), duplicates])
        dupl_df_cn = pd.DataFrame(dupl_cn_list)
        # разобьем таблицу на 3 части, чтобы удобнее читать
        yield (pd.concat([part_df.reset_index(drop=True) for part_df in np.array_split(dupl_df_cn, 2)], axis=1)
               .style.format({1: '{:.0f}'}, na_rep='').hide(axis="index").hide(axis="columns"))
        if n < col_n+1:
            return


def find_columns_with_missing_values(df) -> pd.Series:
    '''
    Фукнция проверяет каждый столбец в таблице,  
    если есть пропуски, то помещает строки исходного 
    дата фрейма с этими пропусками в Series. 
    Индекс - название колонки. 
    Если нужно соеденить фреймы в один, то используем 
    pd.concat(res.to_list())
    '''
    dfs_na = pd.Series(dtype=int)
    cnt_missing = pd.Series(dtype=int)
    size = df.shape[0]
    for col in df.columns:
        is_na = df[col].isna()
        if is_na.any():
            dfs_na[col] = df[is_na]
            cnt_missing[col] = dfs_na[col].shape[0]
    display(cnt_missing.apply(lambda x: f'{x} ({(x / size):.2%})').to_frame().style
            .set_caption('Missings')
            .set_table_styles([{'selector': 'caption',
                                'props': [('font-size', '18px'), ("text-align", "left"), ("font-weight", "bold")]}])
            .hide(axis="columns"))
    return dfs_na


def check_na_in_both_columns(df, cols: list) -> pd.DataFrame:
    '''
    Фукнция проверяет есть ли пропуски одновременно во всех указанных столбцах
    и возвращает датафрейм только со строками, в которых пропуски одновременно во всех столбцах
    '''
    size = df.shape[0]
    mask = df[cols].isna().all(axis=1)
    na_df = df[mask]
    print(
        f'{na_df.shape[0]} ({(na_df.shape[0] / size):.2%}) rows with missings simultaneously in {cols}')
    return na_df


def get_missing_value_proportion_by_category(df: pd.DataFrame, column_with_missing_values: str, category_column: str = None) -> pd.DataFrame:
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
        missing_value_counts = df[mask].groupby(
            category_column).size().reset_index(name='missing_count')
        summ_missing_counts = missing_value_counts['missing_count'].sum()
        # Get the total count for each category
        total_counts = df.groupby(
            category_column).size().reset_index(name='total_count')

        # Merge the two DataFrames to calculate the proportion of missing values
        result_df = pd.merge(missing_value_counts,
                             total_counts, on=category_column)
        result_df['missing_value_in_category_pct'] = (
            result_df['missing_count'] / result_df['total_count']).apply(lambda x: f'{x:.1%}')
        result_df['missing_value_in_column_pct'] = (
            result_df['missing_count'] / summ_missing_counts).apply(lambda x: f'{x:.1%}')
        result_df['total_count_pct'] = (
            result_df['total_count'] / size).apply(lambda x: f'{x:.1%}')
        # Return the result DataFrame
        display(result_df[[category_column, 'total_count', 'missing_count', 'missing_value_in_category_pct', 'missing_value_in_column_pct', 'total_count_pct']]
                .style.set_caption(f'Missing values in "{column_with_missing_values}" by categroy "{category_column}"').set_table_styles([{'selector': 'caption',
                                                                                                                                           'props': [('font-size', '18px'), ("text-align", "left"), ("font-weight", "bold")]}])
                )
        yield
    else:
        categroy_columns = [
            col for col in df.columns if pd.api.types.is_categorical_dtype(df[col])]
        for category_column in categroy_columns:
            # Create a mask to select rows with missing values in the specified column
            mask = df[column_with_missing_values].isna()
            size = df[column_with_missing_values].size
            # Group by category and count the number of rows with missing values
            missing_value_counts = df[mask].groupby(
                category_column).size().reset_index(name='missing_count')
            summ_missing_counts = missing_value_counts['missing_count'].sum()
            # Get the total count for each category
            total_counts = df.groupby(
                category_column).size().reset_index(name='total_count')

            # Merge the two DataFrames to calculate the proportion of missing values
            result_df = pd.merge(missing_value_counts,
                                 total_counts, on=category_column)
            result_df['missing_value_in_category_pct'] = (
                result_df['missing_count'] / result_df['total_count']).apply(lambda x: f'{x:.1%}')
            result_df['missing_value_in_column_pct'] = (
                result_df['missing_count'] / summ_missing_counts).apply(lambda x: f'{x:.1%}')
            result_df['total_count_pct'] = (
                result_df['total_count'] / size).apply(lambda x: f'{x:.1%}')
            # Return the result DataFrame
            display(result_df[[category_column, 'total_count', 'missing_count', 'missing_value_in_category_pct', 'missing_value_in_column_pct', 'total_count_pct']]
                    .style.set_caption(f'Missing values in "{column_with_missing_values}" by categroy "{category_column}"').set_table_styles([{'selector': 'caption',
                                                                                                                                               'props': [('font-size', '18px'), ("text-align", "left"), ("font-weight", "bold")]}]))
            yield


def missings_by_category_gen(df, series_missed):
    '''
    Генератор.
    Для каждой колонки в series_missed функция выводит выборку датафрейма с пропусками в этой колонке.  
    И затем выводит информацию о пропусках по каждой категории в таблице.
    '''
    for col in series_missed.index:
        display(series_missed[col].sample(10).style.set_caption(f'Sample missings in {col}').set_table_styles([{'selector': 'caption',
                                                                                                                'props': [('font-size', '18px'), ("text-align", "left"), ("font-weight", "bold")]}]))
        yield
        gen = get_missing_value_proportion_by_category(df, col)
        for _ in gen:
            yield


def get_duplicates_value_proportion_by_category(df: pd.DataFrame, column_with_dublicated_values: str, category_column: str) -> pd.DataFrame:
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
    dublicated_value_counts = df[mask].groupby(
        category_column).size().reset_index(name='dublicated_count')
    summ_dublicated_value_counts = dublicated_value_counts['dublicated_count'].sum(
    )
    # Get the total count for each category
    total_counts = df.groupby(
        category_column).size().reset_index(name='total_count')

    # Merge the two DataFrames to calculate the proportion of dublicated values
    result_df = pd.merge(dublicated_value_counts,
                         total_counts, on=category_column)
    result_df['dublicated_value_in_category_pct'] = (
        result_df['dublicated_count'] / result_df['total_count']).apply(lambda x: f'{x:.1%}')
    result_df['dublicated_value_in_column_pct'] = (
        result_df['dublicated_count'] / summ_dublicated_value_counts).apply(lambda x: f'{x:.1%}')
    # Return the result DataFrame
    return result_df[[category_column, 'total_count', 'dublicated_count', 'dublicated_value_in_category_pct', 'dublicated_value_in_column_pct']]


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
    df_unique_identifiers_for_compare = df[identifier_columns].set_index(
        identifier_columns).index
    unique_identifiers_for_comapre = unique_identifiers.set_index(
        identifier_columns).index
    matching_rows = df[df_unique_identifiers_for_compare.isin(unique_identifiers_for_comapre) &
                       (~df['total_income'].isna())]
    # Check if there are matching rows without missing values
    if not matching_rows.empty:
        if check:
            print(
                f'Found {matching_rows.shape[0]} matching rows without missing values')
            return
        # Replace missing values with values from matching rows
        df.loc[missing_rows.index,
               target_column] = matching_rows[target_column].values
        print(
            f'Fiiled {matching_rows.shape[0]} matching rows without missing values')
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
    non_unique_values = df.groupby(col1)[col2].nunique()[lambda x: x > 1].index
    non_matching_rows = df[df[col1].isin(non_unique_values)]
    if non_matching_rows.empty:
        print('Нет строк для которых значения в col1 имеют разные значения в col2')
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
    display(cnt_outliers.to_frame().T.style
            .set_caption('Outliers')
            .set_table_styles([{'selector': 'caption',
                                'props': [('font-size', '18px'), ("text-align", "left"), ("font-weight", "bold")]}])
            .hide(axis="index"))
    return outliers


def detect_outliers_quantile(df: pd.DataFrame, lower_quantile: float = 0.05, upper_quantile: float = 0.95) -> pd.Series:
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
    display(cnt_outliers.apply(lambda x: f'{x} ({(x / size):.2%})').to_frame().style
            .set_caption('Outliers')
            .set_table_styles([{'selector': 'caption',
                                'props': [('font-size', '18px'), ("text-align", "left"), ("font-weight", "bold")]}])
            .hide(axis="columns"))
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
    helper_df = df[[helper_column, categorical_column]
                   ].drop_duplicates(helper_column)

    # Удалить строки с пропусками в categorical_column
    helper_df = helper_df.dropna(subset=[categorical_column])

    # Создать новую таблицу с заполненными пропусками
    filled_df = df.drop(categorical_column, axis=1)
    filled_df = filled_df.merge(helper_df, on=helper_column, how='left')

    return filled_df


def get_outlier_quantile_proportion_by_category(df: pd.DataFrame, column_with_outliers: str, category_column: str = None, lower_quantile: float = 0.05, upper_quantile: float = 0.95) -> None:
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
        df[column_with_outliers] > upper_bound)
    size = df[column_with_outliers].size
    if category_column:
        # Group by category and count the number of rows with outliers
        outlier_counts = df[mask].groupby(
            category_column).size().reset_index(name='outlier_count')
        summ_outlier_counts = outlier_counts['outlier_count'].sum()
        # Get the total count for each category
        total_counts = df.groupby(
            category_column).size().reset_index(name='total_count')

        # Merge the two DataFrames to calculate the proportion of outliers
        result_df = pd.merge(outlier_counts,
                             total_counts, on=category_column)
        result_df['outlier_in_category_pct'] = (
            result_df['outlier_count'] / result_df['total_count']).apply(lambda x: f'{x:.1%}')
        result_df['outlier_in_column_pct'] = (
            result_df['outlier_count'] / summ_outlier_counts).apply(lambda x: f'{x:.1%}')
        result_df['total_count_pct'] = (
            result_df['total_count'] / size).apply(lambda x: f'{x:.1%}')
        display(result_df[[category_column, 'total_count', 'outlier_count', 'outlier_in_category_pct', 'outlier_in_column_pct', 'total_count_pct']].style
                .set_caption(f'Outliers in "{column_with_outliers}" by category "{category_column}"')
                .set_table_styles([{'selector': 'caption',
                                    'props': [('font-size', '18px'), ("text-align", "left"), ("font-weight", "bold")]}])
                .hide(axis="index"))
        yield
    else:
        categroy_columns = [
            col for col in df.columns if pd.api.types.is_categorical_dtype(df[col])]
        for category_column in categroy_columns:
            # Group by category and count the number of rows with outliers
            outlier_counts = df[mask].groupby(
                category_column).size().reset_index(name='outlier_count')
            summ_outlier_counts = outlier_counts['outlier_count'].sum()
            # Get the total count for each category
            total_counts = df.groupby(
                category_column).size().reset_index(name='total_count')

            # Merge the two DataFrames to calculate the proportion of outliers
            result_df = pd.merge(outlier_counts,
                                 total_counts, on=category_column)
            result_df['outlier_in_category_pct'] = (
                result_df['outlier_count'] / result_df['total_count']).apply(lambda x: f'{x:.1%}')
            result_df['outlier_in_column_pct'] = (
                result_df['outlier_count'] / summ_outlier_counts).apply(lambda x: f'{x:.1%}')
            result_df['total_count_pct'] = (
                result_df['total_count'] / size).apply(lambda x: f'{x:.1%}')
            display(result_df[[category_column, 'total_count', 'outlier_count', 'outlier_in_category_pct', 'outlier_in_column_pct', 'total_count_pct']].style
                    .set_caption(f'Outliers in "{column_with_outliers}" by category "{category_column}"')
                    .set_table_styles([{'selector': 'caption',
                                        'props': [('font-size', '18px'), ("text-align", "left"), ("font-weight", "bold")]}])
                    .hide(axis="index"))
            yield


def outliers_by_category_gen(df, series_outliers, lower_quantile: float = 0.05, upper_quantile: float = 0.95):
    '''
    Генератор.
    Для каждой колонки в series_outliers функция выводит выборку датафрейма с выбросами (определяется по квантилям) в этой колонке.  
    И затем выводит информацию о выбросах по каждой категории в таблице.
    '''
    for col in series_outliers.index:
        print(f'Value counts outliers')
        display(series_outliers[col][col].value_counts().to_frame('outliers').head(10).style.set_caption(f'{col}')
                .set_table_styles([{'selector': 'caption',
                                    'props': [('font-size', '18px'), ("text-align", "left"), ("font-weight", "bold")]
                                    }]))
        yield
        display(series_outliers[col].sample(10).style.set_caption(f'Sample outliers in {col}').set_table_styles([{'selector': 'caption',
                                                                                                                  'props': [('font-size', '18px'), ("text-align", "left"), ("font-weight", "bold")]}]))
        yield
        gen = get_outlier_quantile_proportion_by_category(
            df, col, lower_quantile=lower_quantile, upper_quantile=upper_quantile)
        for _ in gen:
            yield


def get_outlier_proportion_by_category_modified_z_score(df: pd.DataFrame, column_with_outliers: str, category_column: str, threshold: float = 3.5) -> None:
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
    mask = np.abs(
        0.6745 * (df[column_with_outliers] - median) / mad) > threshold

    # Group by category and count the number of rows with outliers
    outlier_counts = df[mask].groupby(
        category_column).size().reset_index(name='outlier_count')
    summ_outlier_counts = outlier_counts['outlier_count'].sum()

    # Get the total count for each category
    total_counts = df.groupby(
        category_column).size().reset_index(name='total_count')

    # Merge the two DataFrames to calculate the proportion of outliers
    result_df = pd.merge(outlier_counts,
                         total_counts, on=category_column)
    result_df['outlier_in_category_pct'] = (
        result_df['outlier_count'] / result_df['total_count']).apply(lambda x: f'{x:.1%}')
    result_df['outlier_in_column_pct'] = (
        result_df['outlier_count'] / summ_outlier_counts).apply(lambda x: f'{x:.1%}')

    display(result_df[[category_column, 'total_count', 'outlier_count', 'outlier_in_category_pct', 'outlier_in_column_pct']].style
            .set_caption(f'Outliers in "{column_with_outliers}" by category "{category_column}"')
            .set_table_styles([{'selector': 'caption',
                                'props': [('font-size', '18px'), ("text-align", "left"), ("font-weight", "bold")]}])
            .hide(axis="index"))


def find_columns_with_negative_values(df) -> pd.Series:
    '''
    Фукнция проверяет каждый столбец в таблице,  
    если есть отрицательные значения, то помещает строки исходного 
    дата фрейма с этими значениями в Series. 
    Индекс - название колонки. 
    Если нужно соеденить фреймы в один, то используем 
    pd.concat(res.to_list())
    '''
    dfs_na = pd.Series(dtype=int)
    cnt_negative = pd.Series(dtype=int)
    size = df.shape[0]
    num_columns = [
        col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
    for col in num_columns:
        is_negative = df[col] < 0
        if is_negative.any():
            dfs_na[col] = df[is_negative]
            cnt_negative[col] = dfs_na[col].shape[0]
    display(cnt_negative.apply(lambda x: f'{x} ({(x / size):.2%})').to_frame().style
            .set_caption('Negative')
            .set_table_styles([{'selector': 'caption',
                                'props': [('font-size', '18px'), ("text-align", "left"), ("font-weight", "bold")]}])
            .hide(axis="columns"))
    return dfs_na


def get_negative_proportion_by_category(df: pd.DataFrame, column_with_negative: str, category_column: str = None) -> None:
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
        negative_counts = df[mask].groupby(
            category_column).size().reset_index(name='negative_count')
        summ_negative_counts = negative_counts['negative_count'].sum()
        # Get the total count for each category
        total_counts = df.groupby(
            category_column).size().reset_index(name='total_count')

        # Merge the two DataFrames to calculate the proportion of negatives
        result_df = pd.merge(negative_counts,
                             total_counts, on=category_column)
        result_df['negative_in_category_pct'] = (
            result_df['negative_count'] / result_df['total_count']).apply(lambda x: f'{x:.1%}')
        result_df['negative_in_column_pct'] = (
            result_df['negative_count'] / summ_negative_counts).apply(lambda x: f'{x:.1%}')
        result_df['total_count_pct'] = (
            result_df['total_count'] / size).apply(lambda x: f'{x:.1%}')
        display(result_df[[category_column, 'total_count', 'negative_count', 'negative_in_category_pct', 'negative_in_column_pct', 'total_count_pct']].style
                .set_caption(f'negatives in "{column_with_negative}" by category "{category_column}"')
                .set_table_styles([{'selector': 'caption',
                                    'props': [('font-size', '18px'), ("text-align", "left"), ("font-weight", "bold")]}])
                .hide(axis="index"))
        yield
    else:
        categroy_columns = [
            col for col in df.columns if pd.api.types.is_categorical_dtype(df[col])]
        for category_column in categroy_columns:
            # Group by category and count the number of rows with negatives
            negative_counts = df[mask].groupby(
                category_column).size().reset_index(name='negative_count')
            summ_negative_counts = negative_counts['negative_count'].sum()
            # Get the total count for each category
            total_counts = df.groupby(
                category_column).size().reset_index(name='total_count')

            # Merge the two DataFrames to calculate the proportion of negatives
            result_df = pd.merge(negative_counts,
                                 total_counts, on=category_column)
            result_df['negative_in_category_pct'] = (
                result_df['negative_count'] / result_df['total_count']).apply(lambda x: f'{x:.1%}')
            result_df['negative_in_column_pct'] = (
                result_df['negative_count'] / summ_negative_counts).apply(lambda x: f'{x:.1%}')
            result_df['total_count_pct'] = (
                result_df['total_count'] / size).apply(lambda x: f'{x:.1%}')
            display(result_df[[category_column, 'total_count', 'negative_count', 'negative_in_category_pct', 'negative_in_column_pct', 'total_count_pct']].style
                    .set_caption(f'negatives in "{column_with_negative}" by category "{category_column}"')
                    .set_table_styles([{'selector': 'caption',
                                        'props': [('font-size', '18px'), ("text-align", "left"), ("font-weight", "bold")]}])
                    .hide(axis="index"))
            yield


def negative_by_category_gen(df, series_negative):
    '''
    Генератор.
    Для каждой колонки в series_negative функция выводит выборку датафрейма с отрицательными значениями.  
    И затем выводит информацию об отрицательных значениях по каждой категории в таблице.
    '''
    for col in series_negative.index:
        print(f'Value counts negative')
        display(series_negative[col][col].value_counts().to_frame('negative').head(10).style.set_caption(f'{col}')
                .set_table_styles([{'selector': 'caption',
                                    'props': [('font-size', '18px'), ("text-align", "left"), ("font-weight", "bold")]
                                    }]))
        yield
        display(series_negative[col].sample(10).style.set_caption(f'Sample negative in {col}').set_table_styles([{'selector': 'caption',
                                                                                                                  'props': [('font-size', '18px'), ("text-align", "left"), ("font-weight", "bold")]}]))
        yield
        gen = get_negative_proportion_by_category(df, col)
        for _ in gen:
            yield


def find_columns_with_zeros_values(df) -> pd.Series:
    '''
    Фукнция проверяет каждый столбец в таблице,  
    если есть нулевые значения, то помещает строки исходного 
    дата фрейма с этими значениями в Series. 
    Индекс - название колонки. 
    Если нужно соеденить фреймы в один, то используем 
    pd.concat(res.to_list())
    '''
    dfs_na = pd.Series(dtype=int)
    cnt_zeros = pd.Series(dtype=int)
    size = df.shape[0]
    num_columns = [
        col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
    for col in num_columns:
        is_zeros = df[col] == 0
        if is_zeros.any():
            dfs_na[col] = df[is_zeros]
            cnt_zeros[col] = dfs_na[col].shape[0]
    display(cnt_zeros.apply(lambda x: f'{x} ({(x / size):.2%})').to_frame().style
            .set_caption('Zeros')
            .set_table_styles([{'selector': 'caption',
                                'props': [('font-size', '18px'), ("text-align", "left"), ("font-weight", "bold")]}])
            .hide(axis="columns"))
    return dfs_na


def get_zeros_proportion_by_category(df: pd.DataFrame, column_with_zeros: str, category_column: str = None) -> None:
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
        zeros_counts = df[mask].groupby(
            category_column).size().reset_index(name='zeros_count')
        summ_zeros_counts = zeros_counts['zeros_count'].sum()
        # Get the total count for each category
        total_counts = df.groupby(
            category_column).size().reset_index(name='total_count')

        # Merge the two DataFrames to calculate the proportion of zeross
        result_df = pd.merge(zeros_counts,
                             total_counts, on=category_column)
        result_df['zeros_in_category_pct'] = (
            result_df['zeros_count'] / result_df['total_count']).apply(lambda x: f'{x:.1%}')
        result_df['zeros_in_column_pct'] = (
            result_df['zeros_count'] / summ_zeros_counts).apply(lambda x: f'{x:.1%}')
        result_df['total_count_pct'] = (
            result_df['total_count'] / size).apply(lambda x: f'{x:.1%}')
        display(result_df[[category_column, 'total_count', 'zeros_count', 'zeros_in_category_pct', 'zeros_in_column_pct', 'total_count_pct']].style
                .set_caption(f'zeros in "{column_with_zeros}" by category "{category_column}"')
                .set_table_styles([{'selector': 'caption',
                                    'props': [('font-size', '18px'), ("text-align", "left"), ("font-weight", "bold")]}])
                .hide(axis="index"))
        yield
    else:
        categroy_columns = [
            col for col in df.columns if pd.api.types.is_categorical_dtype(df[col])]
        for category_column in categroy_columns:
            # Group by category and count the number of rows with zeross
            zeros_counts = df[mask].groupby(
                category_column).size().reset_index(name='zeros_count')
            summ_zeros_counts = zeros_counts['zeros_count'].sum()
            # Get the total count for each category
            total_counts = df.groupby(
                category_column).size().reset_index(name='total_count')

            # Merge the two DataFrames to calculate the proportion of zeross
            result_df = pd.merge(zeros_counts,
                                 total_counts, on=category_column)
            result_df['zeros_in_category_pct'] = (
                result_df['zeros_count'] / result_df['total_count']).apply(lambda x: f'{x:.1%}')
            result_df['zeros_in_column_pct'] = (
                result_df['zeros_count'] / summ_zeros_counts).apply(lambda x: f'{x:.1%}')
            result_df['total_count_pct'] = (
                result_df['total_count'] / size).apply(lambda x: f'{x:.1%}')
            display(result_df[[category_column, 'total_count', 'zeros_count', 'zeros_in_category_pct', 'zeros_in_column_pct', 'total_count_pct']].style
                    .set_caption(f'zeros in "{column_with_zeros}" by category "{category_column}"')
                    .set_table_styles([{'selector': 'caption',
                                        'props': [('font-size', '18px'), ("text-align", "left"), ("font-weight", "bold")]}])
                    .hide(axis="index"))
            yield


def zeros_by_category_gen(df, series_zeros):
    '''
    Генератор.
    Для каждой колонки в series_zeros функция выводит выборку датафрейма с нулевыми значениями.  
    И затем выводит информацию об нулевых значениях по каждой категории в таблице.
    '''
    for col in series_zeros.index:
        print(f'Value counts zeros')
        display(series_zeros[col][col].value_counts().to_frame('zeros').head(10).style.set_caption(f'{col}')
                .set_table_styles([{'selector': 'caption',
                                    'props': [('font-size', '18px'), ("text-align", "left"), ("font-weight", "bold")]
                                    }]))
        yield
        display(series_zeros[col].sample(10).style.set_caption(f'Sample zeros in {col}').set_table_styles([{'selector': 'caption',
                                                                                                            'props': [('font-size', '18px'), ("text-align", "left"), ("font-weight", "bold")]}]))
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


def create_category_column(column, method='custom_intervals', labels=None, n_intervals=None, bins=None, right=True):
    """
    Create a new category column based on the chosen method.

    Parameters:
    - column (pandas Series): input column
    - method (str, optional): either 'custom_intervals' or 'quantiles' (default is 'custom_intervals')
    - labels (list, optional): list of labels for future categories (default is None)
    - n_intervals (int, optional): number of intervals for 'custom_intervals' or 'quantiles' method (default is len(labels) + 1)
    - bins (list, optional): list of bins for pd.cut function (default is None). The length of `bins` should be `len(labels) + 1`.
    - right (bool, optional): Whether to include the rightmost edge or not. Default is True.

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
    if method == 'custom_intervals':
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
                    column, bins=intervals, labels=labels, right=right)
        else:
            if labels is None:
                category_column = pd.cut(column, bins=bins, right=right)
            else:
                category_column = pd.cut(
                    column, bins=bins, labels=labels, right=right)
    elif method == 'quantiles':
        if n_intervals is None:
            # default number of intervals
            n_intervals = len(labels) if labels is not None else 10
        if labels is None:
            category_column = pd.qcut(
                column.rank(method='first'), q=n_intervals)
        else:
            category_column = pd.qcut(column.rank(
                method='first'), q=n_intervals, labels=labels)
    else:
        raise ValueError(
            "Invalid method. Choose either 'custom_intervals' or 'quantiles'.")

    return category_column.astype('category')


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
            return ''

        try:
            lemmas = m.lemmatize(text)
            return ' '.join(lemmas)
        except Exception as e:
            print(f"Ошибка при лемматизации текста: {e}")
            return ''

    return column.map(lemmatize_text)


def categorize_column_by_lemmatize(column: pd.Series, categorization_dict: dict, use_cache: bool = False):
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
        return 'Unknown'

    lemmatized_column = column.map(lemmatize_text)
    return lemmatized_column.map(categorize_text).astype('category')


def target_encoding_linear(df, category_col, value_col, func='mean', alpha=0.1):
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
    available_funcs = {'median', 'mean', 'max', 'min', 'std', 'count'}

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


def target_encoding_bayes(df, category_col, value_col, func='mean', reg_group_size=100):
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

    available_funcs = {'median', 'mean', 'max', 'min', 'std', 'count'}

    if isinstance(func, str):
        if func not in available_funcs:
            raise ValueError(f"Unknown function: {func}")
        # Если func является строкой, используйте соответствующий метод pandas
        encoding = df.groupby(category_col)[value_col].agg(
            func_val=(func), count=('count'))
    else:
        # Если func является вызываемым, примените его к каждой группе значений
        encoding = df.groupby(category_col)[value_col].agg(
            func_val=(func), count=('count'))

    global_mean = df[value_col].mean()
    # Добавляем байесовскую регуляризацию
    encoding_reg = (encoding['func_val'] * encoding['count'] +
                    global_mean * reg_group_size) / (encoding['count'] + reg_group_size)

    # Заменяем категории на средние значения
    encoded_col = df[category_col].map(encoding_reg.to_dict())

    return encoded_col


def heatmap(df, title='', xtick_text=None, ytick_text=None, xaxis_label=None, yaxis_label=None, width=None, height=None, decimal_places=2, font_size=14):
    """
    Creates a heatmap from a Pandas DataFrame using Plotly.

    Parameters:
    - `df`: The Pandas DataFrame to create the heatmap from.
    - `title`: The title of the heatmap (default is an empty string).
    - `xtick_text`: The custom tick labels for the x-axis (default is None).
    - `ytick_text`: The custom tick labels for the y-axis (default is None).
    - `xaxis_label`: The label for the x-axis (default is None).
    - `yaxis_label`: The label for the y-axis (default is None).
    - `width`: The width of the heatmap (default is None).
    - `height`: The height of the heatmap (default is None).
    - `decimal_places`: The number of decimal places to display in the annotations (default is 2).
    - `font_size`: The font size for the text in the annotations (default is 14).

    Returns:
    - A Plotly figure object representing the heatmap.

    Notes:
    - If `xtick_text` or `ytick_text` is provided, it must have the same length as the number of columns or rows in the DataFrame, respectively.
    - The heatmap is created with a custom colorscale and hover labels.
    - The function returns a Plotly figure object, which can be displayed using `fig.show()`.
    """
    # Create figure
    fig = go.Figure(data=go.Heatmap(
        z=df.values,
        x=df.columns,
        y=df.index,
        xgap=3,
        ygap=3,
        colorscale=[[0, 'rgba(204, 153, 255, 0.1)'], [1, 'rgb(127, 60, 141)']],
        hoverongaps=False,
        hoverinfo="x+y+z",
        hoverlabel=dict(
            bgcolor="white",
            # Increase font size to font_size
            font=dict(color="black", size=font_size)
        )
    ))

    # Create annotations
    center_color_bar = (df.max().max() + df.min().min()) * 0.7
    annotations = [
        dict(
            text=f"{df.values[row, col]:.{decimal_places}f}",
            x=col,
            y=row,
            showarrow=False,
            font=dict(
                color="black" if df.values[row, col] <
                center_color_bar else "white",
                size=font_size
            )
        )
        for row, col in np.ndindex(df.values.shape)
    ]

    # Update layout
    fig.update_layout(
        title=title,
        annotations=annotations,
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=False)
    )

    # Update axis labels if custom labels are provided
    if xtick_text is not None:
        if len(xtick_text) != len(df.columns):
            raise ValueError(
                "xtick_text must have the same length as the number of columns in the DataFrame")
        fig.update_layout(xaxis=dict(tickvals=range(
            len(xtick_text)), ticktext=xtick_text))

    if ytick_text is not None:
        if len(ytick_text) != len(df.index):
            raise ValueError(
                "ytick_text must have the same length as the number of rows in the DataFrame")
        fig.update_layout(yaxis=dict(tickvals=range(
            len(ytick_text)), ticktext=ytick_text))

    # Update axis labels if custom labels are provided
    if xaxis_label is not None:
        fig.update_layout(xaxis=dict(title=xaxis_label))

    if yaxis_label is not None:
        fig.update_layout(yaxis=dict(title=yaxis_label))

    # Update figure size if custom size is provided
    if width is not None:
        fig.update_layout(width=width)
    if height is not None:
        fig.update_layout(height=height)

    return fig


def plot_feature_importances_classifier(df: pd.DataFrame, target: str):
    """
    Plot the feature importances of a random forest classifier using Plotly Express.

    Parameters:
    df (pandas.DataFrame): The input DataFrame containing the features and target variable.
    target (str): The name of the target variable column in the DataFrame.

    Returns:
    fig (plotly.graph_objs.Figure): The feature importance plot.

    Notes:
    This function trains a random forest classifier on the input DataFrame, extracts the feature importances,
    and plots them using Plotly Express. 
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
    fig = px.bar(feature_importances, x='Importance', y='Feature',
                 title=f'Feature Importances for {target_str}')
    fig.update_layout(
        yaxis=dict(categoryorder='total ascending'),
        width=700,  # Set the width of the graph
        height=500,  # Set the height of the graph
        hoverlabel=dict(bgcolor='white'),
        template='simple_white'  # Set the template to simple_white
    )
    # Set the bar color to mediumpurple
    fig.update_traces(marker_color='rgba(128, 60, 170, 0.9)')

    return fig


def plot_feature_importances_regression(df: pd.DataFrame, target: str):
    """
    Plot the feature importances of a random forest regressor using Plotly Express.

    Parameters:
    df (pandas.DataFrame): The input DataFrame containing the features and target variable.
    target (str): The name of the target variable column in the DataFrame.

    Returns:
    fig (plotly.graph_objs.Figure): The feature importance plot.

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

    # Create the bar chart
    fig = px.bar(feature_importances, x='Importance', y='Feature',
                 title=f'Feature Importances for {target}')
    fig.update_layout(
        yaxis=dict(categoryorder='total ascending'),
        width=700,  # Set the width of the graph
        height=500,  # Set the height of the graph
        hoverlabel=dict(bgcolor='white'),
        template='simple_white'  # Set the template to simple_white
    )
    # Set the bar color to mediumpurple
    fig.update_traces(marker_color='rgba(128, 60, 170, 0.9)')

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


def categorical_heatmap_matrix_gen(df, titles_for_axis: dict = None, width=None, height=None):
    """
    Generate a heatmap matrix for all possible combinations of categorical variables in a dataframe.

    This function takes a pandas DataFrame as input and generates a heatmap matrix for each pair of categorical variables.
    The heatmap matrix is a visual representation of the cross-tabulation of two categorical variables, which can help identify patterns and relationships between them.

    Parameters:
    df (pandas DataFrame): Input DataFrame containing categorical variables.
    titles_for_axis (dict):  A dictionary containing titles for the axes.

    Returns:
    None
    """
    def human_readable_number(x):
        if x >= 1e6 or x <= -1e6:
            return f"{x/1e6:.1f} M"
        elif x >= 1e3 or x <= -1e3:
            return f"{x/1e3:.1f} k"
        else:
            return f"{x:.0f}"
    # Получаем список категориальных переменных
    categorical_cols = df.select_dtypes(include=['category']).columns
    size = df.shape[0]
    # Перебираем все возможные комбинации категориальных переменных
    for col1, col2 in itertools.combinations(categorical_cols, 2):
        # Создаем матрицу тепловой карты
        heatmap_matrix = pd.crosstab(df[col1], df[col2])

        # Визуализируем матрицу тепловой карты

        if not titles_for_axis:
            title = f'Тепловая карта количества для {col1} и {col2}'
            xaxis_title = f'{col2}'
            yaxis_title = f'{col1}'
        else:
            title = f'Тепловая карта количества для {titles_for_axis[col1][1]} и {titles_for_axis[col2][1]}'
            xaxis_title = f'{titles_for_axis[col2][0]}'
            yaxis_title = f'{titles_for_axis[col1][0]}'
        hovertemplate = xaxis_title + \
            ' = %{x}<br>' + yaxis_title + \
            ' = %{y}<br>Количество = %{z}<extra></extra>'
        fig = heatmap(heatmap_matrix, title=title)
        fig.update_traces(hovertemplate=hovertemplate, showlegend=False)
        center_color_bar = (heatmap_matrix.max().max() +
                            heatmap_matrix.min().min()) * 0.7
        annotations = [
            dict(
                text=f"{human_readable_number(heatmap_matrix.values[row, col])} ({(heatmap_matrix.values[row, col] * 100 / size):.0f} %)" if heatmap_matrix.values[row, col] * 100 / size >= 1
                else f"{human_readable_number(heatmap_matrix.values[row, col])} (<1 %)" if heatmap_matrix.values[row, col] * 100 / size > 0
                else '-',
                x=col,
                y=row,
                showarrow=False,
                font=dict(
                    color="black" if heatmap_matrix.values[row, col] <
                    center_color_bar else "white",
                    size=16
                )
            )
            for row, col in np.ndindex(heatmap_matrix.values.shape)
        ]
        fig.update_layout(
            # , title={'text': f'<b>{title}</b>'}
            width=width, height=height, xaxis_title=xaxis_title, yaxis_title=yaxis_title, annotations=annotations
        )
        plotly_default_settings(fig)
        yield fig


def treemap_dash(df, columns):
    """
    Создает интерактивный treemap с помощью Dash и Plotly.

    Параметры:
    df (pandas.DataFrame): датафрейм с данными для treemap.
    columns (list): список столбцов, которые будут использоваться для создания treemap.

    Возвращает:
    app (dash.Dash): прилоожение Dash с интерактивным treemap.
    """
    date_columns = filter(
        lambda x: pd.api.types.is_datetime64_any_dtype(df[x]), df.columns)
    app = dash.Dash(__name__)

    app.layout = html.Div([
        dcc.Dropdown(
            id='reorder-dropdown',
            options=[{'label': col, 'value': col} for col in columns],
            value=columns,
            multi=True
        ),
        dcc.Graph(id='treemap-graph')
    ])

    @app.callback(
        Output('treemap-graph', 'figure'),
        [Input('reorder-dropdown', 'value')]
    )
    def update_treemap(value):
        fig = px.treemap(df, path=[px.Constant('All')] + value,
                         color_discrete_sequence=[
                             'rgba(148, 100, 170, 1)',
                             'rgba(50, 156, 179, 1)',
                             'rgba(99, 113, 156, 1)',
                             'rgba(92, 107, 192, 1)',
                             'rgba(0, 90, 91, 1)',
                             'rgba(3, 169, 244, 1)',
                             'rgba(217, 119, 136, 1)',
                             'rgba(64, 134, 87, 1)',
                             'rgba(134, 96, 147, 1)',
                             'rgba(132, 169, 233, 1)'
        ])
        fig.update_traces(root_color="lightgrey",
                          hovertemplate="<b>%{label}<br>%{value}</b>")
        fig.update_layout(margin=dict(t=50, l=25, r=25, b=25))
        fig.update_traces(hoverlabel=dict(bgcolor="white"))
        return fig

    return app


def treemap(df, columns, values=None):
    """
    Creates an interactive treemap using Plotly.

    Parameters:
    df (pandas.DataFrame): dataframe with data for the treemap.
    columns (list): list of columns to use for the treemap.
    values (str): column for values, if None - values  will be calculated as count.
    Returns:
    fig (plotly.graph_objs.Figure): interactive treemap figure.
    """
    fig = px.treemap(df, path=[px.Constant('All')] + columns, values=values, color_discrete_sequence=[
        'rgba(148, 100, 170, 1)',
        'rgba(50, 156, 179, 1)',
        'rgba(99, 113, 156, 1)',
        'rgba(92, 107, 192, 1)',
        'rgba(0, 90, 91, 1)',
        'rgba(3, 169, 244, 1)',
        'rgba(217, 119, 136, 1)',
        'rgba(64, 134, 87, 1)',
        'rgba(134, 96, 147, 1)',
        'rgba(132, 169, 233, 1)'
    ])
    fig.update_traces(root_color="silver",
                      hovertemplate="<b>%{label}<br>%{value:.2f}</b>")
    fig.update_layout(margin=dict(t=50, l=25, r=25, b=25))
    fig.update_traces(hoverlabel=dict(bgcolor="white"))
    return fig


def treemap_dash(df):
    """
    Создает интерактивный treemap с помощью Dash и Plotly.

    Параметры:
    df (pandas.DataFrame): датафрейм с данными для treemap.
    columns (list): список столбцов, которые будут использоваться для создания treemap.

    Возвращает:
    app (dash.Dash): прилоожение Dash с интерактивным treemap.

    ```
    app = treemap_dash(df)
    if __name__ == '__main__':
        app.run_server(debug=True)
    ```
    """
    categroy_columns = [
        col for col in df.columns if pd.api.types.is_categorical_dtype(df[col])]
    app = dash.Dash(__name__)

    app.layout = html.Div([
        dcc.Dropdown(
            id='reorder-dropdown',
            options=[{'label': col, 'value': col} for col in categroy_columns],
            value=categroy_columns[:2],
            multi=True
        ),
        dcc.Graph(id='treemap-graph')
    ])

    @app.callback(
        Output('treemap-graph', 'figure'),
        [Input('reorder-dropdown', 'value')]
    )
    def update_treemap(value):
        fig = px.treemap(df, path=[px.Constant('All')] + value,
                         color_discrete_sequence=[
                             'rgba(148, 100, 170, 1)',
                             'rgba(50, 156, 179, 1)',
                             'rgba(99, 113, 156, 1)',
                             'rgba(92, 107, 192, 1)',
                             'rgba(0, 90, 91, 1)',
                             'rgba(3, 169, 244, 1)',
                             'rgba(217, 119, 136, 1)',
                             'rgba(64, 134, 87, 1)',
                             'rgba(134, 96, 147, 1)',
                             'rgba(132, 169, 233, 1)'
        ])
        fig.update_traces(root_color="lightgrey",
                          hovertemplate="<b>%{label}<br>%{value}</b>")
        fig.update_layout(margin=dict(t=50, l=25, r=25, b=25))
        fig.update_traces(hoverlabel=dict(bgcolor="white"))
        return fig

    return app


def parallel_categories(df, columns):
    """
    Creates an interactive parallel_categories using Plotly.

    Parameters:
    df (pandas.DataFrame): dataframe with data for the parallel_categories.
    columns (list): list of columns to use for the parallel_categories.

    Returns:
    fig (plotly.graph_objs.Figure): interactive parallel_categories figure.
    """
    # Создание значений цвета
    color_values = [1 for _ in range(df.shape[0])]

    # Создание параллельных категорий
    fig = px.parallel_categories(df, dimensions=columns, color=color_values,
                                 color_continuous_scale=[
                                     [0, 'rgba(128, 60, 170, 0.9)'],
                                     [1, 'rgba(128, 60, 170, 0.9)']]
                                 )

    # Скрытие цветовой шкалы
    if fig.layout.coloraxis:
        fig.update_layout(coloraxis_showscale=False)
    else:
        print("Цветовая шкала не существует")

    # Обновление макета
    fig.update_layout(margin=dict(t=50, l=150, r=150, b=25))

    return fig


def parallel_categories_dash(df):
    """
    Creates a Dash application with an interactive parallel_categories using Plotly.

    Parameters:
    df (pandas.DataFrame): dataframe with data for the parallel_categories.

    Returns:
    app (dash.Dash): Dash application with interactive parallel_categories figure.

    ```
    app = treemap_dash(df)
    if __name__ == '__main__':
        app.run_server(debug=True)
    ```
    """
    categroy_columns = [
        col for col in df.columns if pd.api.types.is_categorical_dtype(df[col])]
    # Создание Dash-приложения
    app = dash.Dash(__name__)

    app.layout = html.Div([
        # html.H1('Parallel Categories'),
        dcc.Dropdown(
            id='columns-dropdown',
            options=[{'label': col, 'value': col} for col in categroy_columns],
            value=categroy_columns[:2],  # Значение по умолчанию
            multi=True
        ),
        dcc.Graph(id='parallel-categories-graph')
    ])

    # Обновление графика при изменении выбора столбцов
    @app.callback(
        Output('parallel-categories-graph', 'figure'),
        [Input('columns-dropdown', 'value')]
    )
    def update_graph(selected_columns):
        # Создание параллельных категорий
        color_values = [1 for _ in range(df.shape[0])]
        fig = px.parallel_categories(df, dimensions=selected_columns, color=color_values,
                                     color_continuous_scale=[
                                         [0, 'rgba(128, 60, 170, 0.7)'],
                                         [1, 'rgba(128, 60, 170, 0.7)']]
                                     )

        # Скрытие цветовой шкалы
        if fig.layout.coloraxis:
            fig.update_layout(coloraxis_showscale=False)
        else:
            print("Цветовая шкала не существует")

        # Обновление макета
        fig.update_layout(margin=dict(t=50, l=150, r=150, b=25))

        return fig

    return app


def sankey(df, columns, values_column=None, func='sum', mode='fig'):
    """
    Создает Sankey-диаграмму

    Parameters:
    df (pandas.DataFrame): входной DataFrame
    columns (list): список столбцов для Sankey-диаграммы

    Returns:
    fig (plotly.graph_objects.Figure): Sankey-диаграмма
    """
    def prepare_data(df, columns, values_column, func):
        """
        Подготавливает данные для Sankey-диаграммы.

        Parameters:
        df (pandas.DataFrame): входной DataFrame
        columns (list): список столбцов для Sankey-диаграммы

        Returns:
        sankey_df (pandas.DataFrame): подготовленный DataFrame для Sankey-диаграммы
        """
        df_in = df.fillna(value={values_column: 0}).copy()
        columns_len = len(columns)
        temp_df = pd.DataFrame()
        if func == 'mode':
            def func(x): return x.mode().iloc[0]
        if func == 'range':
            def func(x): return x.max() - x.min()
        for i in range(columns_len - 1):
            current_columns = columns[i:i+2]
            if values_column:
                df_grouped = df_in[current_columns+[values_column]].groupby(
                    current_columns)[[values_column]].agg(value=(values_column, func)).reset_index()
            else:
                df_grouped = df_in[current_columns].groupby(
                    current_columns).size().reset_index().rename(columns={0: 'value'})
            temp_df = pd.concat([temp_df, df_grouped
                                 .rename(columns={columns[i]: 'source_name', columns[i+1]: 'target_name'})], axis=0)
        sankey_df = temp_df.reset_index(drop=True)
        return sankey_df

    def create_sankey_nodes(sankey_df):
        """
        Создает узлы для Sankey-диаграммы.

        Parameters:
        sankey_df (pandas.DataFrame): подготовленный DataFrame для Sankey-диаграммы
        colors (list): список цветов для узлов

        Returns:
        nodes_with_indexes (dict): словарь узлов с индексами
        node_colors (list): список цветов узлов
        """
        nodes = pd.concat(
            [sankey_df['source_name'], sankey_df['target_name']], axis=0).unique().tolist()
        nodes_with_indexes = {key: [val] for val, key in enumerate(nodes)}
        colors = [
            'rgba(148, 100, 170, 1)',
            'rgba(50, 156, 179, 1)',
            'rgba(99, 113, 156, 1)',
            'rgba(92, 107, 192, 1)',
            'rgba(0, 90, 91, 1)',
            'rgba(3, 169, 244, 1)',
            'rgba(217, 119, 136, 1)',
            'rgba(64, 134, 87, 1)',
            'rgba(134, 96, 147, 1)',
            'rgba(132, 169, 233, 1)']
        node_colors = []
        colors = itertools.cycle(colors)
        for node in nodes_with_indexes.keys():
            color = next(colors)
            nodes_with_indexes[node].append(color)
            node_colors.append(color)
        return nodes_with_indexes, node_colors

    def create_sankey_links(sankey_df, nodes_with_indexes):
        """
        Создает связи для Sankey-диаграммы.

        Parameters:
        sankey_df (pandas.DataFrame): подготовленный DataFrame для Sankey-диаграммы
        nodes_with_indexes (dict): словарь узлов с индексами

        Returns:
        link_color (list): список цветов связей
        """
        link_color = [nodes_with_indexes[source][1].replace(
            ', 1)', ', 0.2)') for source in sankey_df['source_name']]
        return link_color
    sankey_df = prepare_data(df, columns, values_column, func)
    nodes_with_indexes, node_colors = create_sankey_nodes(sankey_df)
    link_color = create_sankey_links(sankey_df, nodes_with_indexes)
    sankey_df['source'] = sankey_df['source_name'].apply(
        lambda x: nodes_with_indexes[x][0])
    sankey_df['target'] = sankey_df['target_name'].apply(
        lambda x: nodes_with_indexes[x][0])
    sankey_df['sum_value'] = sankey_df.groupby(
        'source_name')['value'].transform('sum')
    sankey_df['value_percent'] = round(
        sankey_df['value'] * 100 / sankey_df['sum_value'], 2)
    sankey_df['value_percent'] = sankey_df['value_percent'].apply(
        lambda x: f"{x}%")
    if mode == 'fig':
        fig = go.Figure(data=[go.Sankey(
            domain=dict(
                x=[0, 1],
                y=[0, 1]
            ),
            orientation="h",
            valueformat=".0f",
            node=dict(
                pad=10,
                thickness=15,
                line=dict(color="black", width=0.1),
                label=list(nodes_with_indexes.keys()),
                color=node_colors
            ),
            link=dict(
                source=sankey_df['source'],
                target=sankey_df['target'],
                value=sankey_df['value'],
                label=sankey_df['value_percent'],
                color=link_color
            )
        )])

        layout = dict(
            title=f"Sankey Diagram for {', '.join(columns+[values_column])}" if values_column else
            f"Sankey Diagram for {', '.join(columns)}",
            height=772,
            font=dict(
                size=10),)

        fig.update_layout(layout)
        return fig
    if mode == 'data':
        sankey_dict = {}
        sankey_dict['sankey_df'] = sankey_df
        sankey_dict['nodes_with_indexes'] = nodes_with_indexes
        sankey_dict['node_colors'] = node_colors
        sankey_dict['link_color'] = link_color
        return sankey_dict


def sankey_dash(df):
    """
    Создает Sankey-диаграмму

    Parameters:
    df (pandas.DataFrame): входной DataFrame
    columns (list): список столбцов для Sankey-диаграммы

    Returns:
    app (dash.Dash): Dash application with interactive parallel_categories figure.

    ```
    app = sankey_dash(df)
    if __name__ == '__main__':
        app.run_server(debug=True)
    ```
    """
    def prepare_data(df, columns):
        """
        Подготавливает данные для Sankey-диаграммы.

        Parameters:
        df (pandas.DataFrame): входной DataFrame
        columns (list): список столбцов для Sankey-диаграммы

        Returns:
        sankey_df (pandas.DataFrame): подготовленный DataFrame для Sankey-диаграммы
        """
        df_in = df.dropna().copy()
        columns_len = len(columns)
        temp_df = pd.DataFrame()
        for i in range(columns_len - 1):
            current_columns = columns[i:i+2]
            df_grouped = df_in[current_columns].groupby(
                current_columns).size().reset_index()
            temp_df = pd.concat([temp_df, df_grouped
                                 .rename(columns={columns[i]: 'source_name', columns[i+1]: 'target_name'})], axis=0)
        sankey_df = temp_df.reset_index(drop=True).rename(columns={0: 'value'})
        return sankey_df

    def create_sankey_nodes(sankey_df):
        """
        Создает узлы для Sankey-диаграммы.

        Parameters:
        sankey_df (pandas.DataFrame): подготовленный DataFrame для Sankey-диаграммы
        colors (list): список цветов для узлов

        Returns:
        nodes_with_indexes (dict): словарь узлов с индексами
        node_colors (list): список цветов узлов
        """
        nodes = pd.concat(
            [sankey_df['source_name'], sankey_df['target_name']], axis=0).unique().tolist()
        nodes_with_indexes = {key: [val] for val, key in enumerate(nodes)}
        colors = [
            'rgba(148, 100, 170, 1)',
            'rgba(50, 156, 179, 1)',
            'rgba(99, 113, 156, 1)',
            'rgba(92, 107, 192, 1)',
            'rgba(0, 90, 91, 1)',
            'rgba(3, 169, 244, 1)',
            'rgba(217, 119, 136, 1)',
            'rgba(64, 134, 87, 1)',
            'rgba(134, 96, 147, 1)',
            'rgba(132, 169, 233, 1)']
        node_colors = []
        colors = itertools.cycle(colors)
        for node in nodes_with_indexes.keys():
            color = next(colors)
            nodes_with_indexes[node].append(color)
            node_colors.append(color)
        return nodes_with_indexes, node_colors

    def create_sankey_links(sankey_df, nodes_with_indexes):
        """
        Создает связи для Sankey-диаграммы.

        Parameters:
        sankey_df (pandas.DataFrame): подготовленный DataFrame для Sankey-диаграммы
        nodes_with_indexes (dict): словарь узлов с индексами

        Returns:
        link_color (list): список цветов связей
        """
        link_color = [nodes_with_indexes[source][1].replace(
            ', 1)', ', 0.2)') for source in sankey_df['source_name']]
        return link_color

    categroy_columns = [
        col for col in df.columns if pd.api.types.is_categorical_dtype(df[col])]
    # Создание Dash-приложения
    app = dash.Dash(__name__)
    app.layout = html.Div([
        # html.H1('Parallel Categories'),
        dcc.Dropdown(
            id='columns-dropdown',
            options=[{'label': col, 'value': col} for col in categroy_columns],
            value=categroy_columns[:2],  # Значение по умолчанию
            multi=True
        ),
        dcc.Graph(id='sankey-graph')
    ])

    # Обновление графика при изменении выбора столбцов
    @app.callback(
        Output('sankey-graph', 'figure'),
        [Input('columns-dropdown', 'value')]
    )
    def update_graph(selected_columns):
        # Создание sankey
        if len(selected_columns) < 2:
            selected_columns = categroy_columns[:2]
        sankey_df = prepare_data(df, selected_columns)
        nodes_with_indexes, node_colors = create_sankey_nodes(sankey_df)
        link_color = create_sankey_links(sankey_df, nodes_with_indexes)
        sankey_df['source'] = sankey_df['source_name'].apply(
            lambda x: nodes_with_indexes[x][0])
        sankey_df['target'] = sankey_df['target_name'].apply(
            lambda x: nodes_with_indexes[x][0])
        sankey_df['sum_value'] = sankey_df.groupby(
            'source_name')['value'].transform('sum')
        sankey_df['value_percent'] = round(
            sankey_df['value'] * 100 / sankey_df['sum_value'], 2)
        sankey_df['value_percent'] = sankey_df['value_percent'].apply(
            lambda x: f"{x}%")
        fig = go.Figure(data=[go.Sankey(
            domain=dict(
                x=[0, 1],
                y=[0, 1]
            ),
            orientation="h",
            valueformat=".0f",
            node=dict(
                pad=10,
                thickness=15,
                line=dict(color="black", width=0.1),
                label=list(nodes_with_indexes.keys()),
                color=node_colors
            ),
            link=dict(
                source=sankey_df['source'],
                target=sankey_df['target'],
                value=sankey_df['value'],
                label=sankey_df['value_percent'],
                color=link_color
            )
        )])

        layout = dict(
            title=f"Sankey Diagram for {', '.join(selected_columns)}",
            height=772,
            font=dict(
                size=10),)

        fig.update_layout(layout)

        return fig

    return app


def graph_analysis(df, cat_coluns, num_column):
    """
    Perform graph analysis and create visualizations based on the provided dataframe and configuration.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe.
    cat_columns : list
        List of categorical columns.
    num_column : str
        Name of the numeric column.

    Returns
    -------
    None

    Notes
    -----
    This function prepares the dataframe for plotting, creates visualizations (such as bar, line, and area plots, heatmaps, treemaps, and sankey diagrams),
    and updates the layout of the plots based on the provided configuration.
    It uses the `prepare_df` and `prepare_data_treemap` functions to prepare the data for plotting.
    """
    if len(cat_coluns) != 2:
        raise Exception('cat_coluns must be  a list of two columns')
    if not isinstance(num_column, str):
        raise Exception('num_column must be  str')
    df_coluns = df.columns
    if cat_coluns[0] not in df_coluns or cat_coluns[1] not in df_coluns or num_column not in df_coluns:
        raise Exception('cat_coluns and num_column must be  in df.columns')
    if not pd.api.types.is_categorical_dtype(df[cat_coluns[0]]) or not pd.api.types.is_categorical_dtype(df[cat_coluns[1]]):
        raise Exception('cat_coluns must be categorical')
    if not pd.api.types.is_numeric_dtype(df[num_column]):
        raise Exception('num_column must be numeric')

    config = {
        'df': df,
        'num_column_y': num_column,
        'cat_columns': cat_coluns,
        'cat_column_x': cat_coluns[0],
        'cat_column_color': cat_coluns[1],
        'func': 'mean'
    }
    colorway_for_bar = ['rgba(128, 60, 170, 0.9)', '#049CB3', '#84a9e9', '#B690C4',
                        '#5c6bc0', '#005A5B', '#63719C', '#03A9F4', '#66CCCC', '#a771f2', 'rgba(128, 60, 170, 0.9)', '#049CB3', '#84a9e9', '#B690C4',
                        '#5c6bc0', '#005A5B', '#63719C', '#03A9F4', '#66CCCC', '#a771f2', 'rgba(128, 60, 170, 0.9)', '#049CB3', '#84a9e9', '#B690C4',
                        '#5c6bc0', '#005A5B', '#63719C', '#03A9F4', '#66CCCC', '#a771f2', 'rgba(128, 60, 170, 0.9)', '#049CB3', '#84a9e9', '#B690C4',
                        '#5c6bc0', '#005A5B', '#63719C', '#03A9F4', '#66CCCC', '#a771f2']

    def prepare_df(config):
        """
        Prepare a dataframe for plotting by grouping and aggregating data.

        Parameters
        ----------
        config : dict
            Configuration dictionary containing dataframe, numeric column, categorical columns, and aggregation function.

        Returns
        -------
        func_df : pandas.DataFrame
            A dataframe with aggregated data, sorted by the numeric column.

        Notes
        -----
        This function groups the dataframe by the categorical columns, applies the aggregation function to the numeric column,
        and sorts the resulting dataframe by the numeric column in descending order.
        If a color column is specified, the function unstacks the dataframe and sorts it by the sum of the numeric column.
        """
        df = config['df']
        cat_column_color = [config['cat_column_color']
                            ] if config['cat_column_color'] else []
        cat_columns = [config['cat_column_x']] + cat_column_color
        num_column = config['num_column_y']
        # print(config)
        # print(cat_columns)
        # print(num_column)
        func = config.get('func', 'mean')  # default to 'sum' if not provided
        if func == 'mode':
            def func(x): return x.mode().iloc[0]
            def func_for_modes(x): return tuple(x.mode().to_list())
        else:
            def func_for_modes(x): return ''
        if func == 'range':
            def func(x): return x.max() - x.min()
        func_df = (df[[*cat_columns, num_column]]
                   .groupby(cat_columns)
                   .agg(num=(num_column, func), modes=(num_column, func_for_modes))
                   .sort_values('num', ascending=False)
                   .rename(columns={'num': num_column})
                   )
        if config['cat_column_color']:
            func_df = func_df.unstack(level=1)
            func_df['sum'] = func_df.sum(axis=1, numeric_only=True)
            func_df = func_df.sort_values(
                'sum', ascending=False).drop('sum', axis=1)
            func_df = pd.concat(
                [func_df[num_column], func_df['modes']], keys=['num', 'modes'])
            func_df = func_df.sort_values(
                func_df.index[0], axis=1, ascending=False)
            return func_df
        else:
            return func_df

    def prepare_data_treemap(df, cat_columns, value_column, func='sum'):
        """
        Prepare data for a treemap plot by grouping and aggregating data.

        Parameters
        ----------
        df : pandas.DataFrame
            Input dataframe.
        cat_columns : list
            List of categorical columns.
        value_column : str
            Name of the numeric column.
        func : str, optional
            Aggregation function (default is 'sum').

        Returns
        -------
        res_df : pandas.DataFrame
            A dataframe with aggregated data, ready for treemap plotting.

        Notes
        -----
        This function groups the dataframe by the categorical columns, applies the aggregation function to the numeric column,
        and creates a hierarchical structure for the treemap plot.
        """
        df_in = df[cat_columns + [value_column]].copy()
        prefix = 'All/'
        if func == 'mode':
            def func(x): return x.mode().iloc[0]
        if func == 'range':
            def func(x): return x.max() - x.min()
        df_grouped_second_level = df_in[[*cat_columns, value_column]].groupby(
            cat_columns).agg({value_column: func}).reset_index()
        df_grouped_second_level['ids'] = df_grouped_second_level[cat_columns].apply(
            lambda x: f'{prefix}{x[cat_columns[0]]}/{x[cat_columns[1]]}', axis=1)
        df_grouped_second_level['parents'] = df_grouped_second_level[cat_columns].apply(
            lambda x: f'{prefix}{x[cat_columns[0]]}', axis=1)
        df_grouped_second_level = df_grouped_second_level.sort_values(
            cat_columns[::-1], ascending=False)
        # df_grouped = df_grouped.drop(cat_columns[0], axis=1)
        df_grouped_first_level = df_grouped_second_level.groupby(
            cat_columns[0]).sum().reset_index()
        df_grouped_first_level['ids'] = df_grouped_first_level[cat_columns[0]].apply(
            lambda x: f'{prefix}{x}')
        df_grouped_first_level['parents'] = 'All'
        df_grouped_first_level = df_grouped_first_level.sort_values(
            cat_columns[0], ascending=False)
        all_value = df_grouped_first_level[value_column].sum()
        res_df = pd.concat([df_grouped_second_level.rename(columns={cat_columns[1]: 'labels', value_column: 'values'}).drop(cat_columns[0], axis=1), df_grouped_first_level.rename(
            columns={cat_columns[0]: 'labels', value_column: 'values'}), pd.DataFrame({'parents': '', 'labels': 'All',  'values': all_value, 'ids': 'All'}, index=[0])], axis=0)
        return res_df

    def create_bars_lines_area_figure(config):
        """
        Create a figure with bar, line, and area traces based on the provided configuration.

        Parameters
        ----------
        config : dict
            Configuration dictionary containing dataframe, numeric column, and categorical columns.

        Returns
        -------
        fig : plotly.graph_objects.Figure
            A figure object containing bar, line, and area traces.
        """
        fig = go.Figure()
        # 1
        config['cat_column_x'] = config['cat_columns'][0]
        config['cat_column_color'] = ''
        df_for_fig = prepare_df(config)
        x = df_for_fig.index.tolist()
        y = df_for_fig[config['num_column_y']].values.tolist()
        bar_traces = px.bar(x=x, y=y
                            ).data
        line_traces = px.line(x=x, y=y, markers=True
                              ).data
        area_traces = px.area(x=x, y=y, markers=True
                              ).data
        fig.add_traces(bar_traces + line_traces + area_traces)
        # 2
        config['cat_column_x'] = config['cat_columns'][1]
        config['cat_column_color'] = ''
        df_for_fig = prepare_df(config)
        x = df_for_fig.index.tolist()
        y = df_for_fig[config['num_column_y']].values.tolist()
        bar_traces = px.bar(x=x, y=y
                            ).data
        line_traces = px.line(x=x, y=y, markers=True
                              ).data
        area_traces = px.area(x=x, y=y, markers=True
                              ).data
        fig.add_traces(bar_traces + line_traces + area_traces)
        # 12
        config['cat_column_x'] = config['cat_columns'][0]
        config['cat_column_color'] = config['cat_columns'][1]
        df_for_fig = prepare_df(config).loc['num', :].stack(
        ).reset_index(name=config['num_column_y'])
        x = df_for_fig[config['cat_column_x']].values.tolist()
        y = df_for_fig[config['num_column_y']].values.tolist()
        color = df_for_fig[config['cat_column_color']
                           ].values if config['cat_column_color'] else None
        bar_traces = px.bar(x=x, y=y, color=color, barmode='group'
                            ).data
        config['traces_cnt12'] = len(bar_traces)
        line_traces = px.line(x=x, y=y, color=color, markers=True
                              ).data
        area_traces = px.area(x=x, y=y, color=color, markers=True
                              ).data
        fig.add_traces(bar_traces + line_traces + area_traces)

        # 21
        config['cat_column_x'] = config['cat_columns'][1]
        config['cat_column_color'] = config['cat_columns'][0]
        df_for_fig = prepare_df(config).loc['num', :].stack(
        ).reset_index(name=config['num_column_y'])
        x = df_for_fig[config['cat_column_x']].values.tolist()
        y = df_for_fig[config['num_column_y']].values.tolist()
        color = df_for_fig[config['cat_column_color']
                           ].values if config['cat_column_color'] else None
        bar_traces = px.bar(x=x, y=y, color=color, barmode='group'
                            ).data
        config['traces_cnt21'] = len(bar_traces)
        line_traces = px.line(x=x, y=y, color=color, markers=True
                              ).data
        area_traces = px.area(x=x, y=y, color=color, markers=True
                              ).data
        fig.add_traces(bar_traces + line_traces + area_traces)

        for i, trace in enumerate(fig.data):
            # при старте показываем только первый trace
            if i:
                trace.visible = False
            if trace.type == 'scatter':
                trace.line.width = 2
                # trace.marker.size = 7
        return fig

    def create_heatmap_treemap_sankey_figure(config):
        """
        Create a figure with heatmap, treemap, and sankey diagrams based on the provided configuration.

        Parameters
        ----------
        config : dict
            Configuration dictionary containing dataframe, numeric column, and categorical columns.

        Returns
        -------
        fig : plotly.graph_objects.Figure
            A figure object containing heatmap, treemap, and sankey diagrams.
        """
        fig = go.Figure()

        # # heatmap
        pivot_for_heatmap = config['df'].pivot_table(
            index=config['cat_columns'][0], columns=config['cat_columns'][1], values=config['num_column_y'])
        heatmap_trace = px.imshow(pivot_for_heatmap, text_auto=".0f").data[0]
        heatmap_trace.xgap = 3
        heatmap_trace.ygap = 3
        fig.add_trace(heatmap_trace)
        fig.update_layout(coloraxis=dict(colorscale=[
                          (0, 'rgba(204, 153, 255, 0.1)'), (1, 'rgb(127, 60, 141)')]), hoverlabel=dict(bgcolor='white'))
        # treemap
        treemap_trace = columns = treemap(
            config['df'], config['cat_columns'], config['num_column_y']).data[0]
        fig.add_trace(treemap_trace)

        # sankey
        sankey_trace = sankey(
            config['df'], config['cat_columns'], config['num_column_y'], func='sum').data[0]
        fig.add_trace(sankey_trace)
        for i, trace in enumerate(fig.data):
            # при старте показываем только первый trace
            if i:
                trace.visible = False
        return fig

    def create_buttons_bars_lines_ares(config):
        """
        Create buttons for updating the layout of a figure with bar, line, and area traces.

        Parameters
        ----------
        config : dict
            Configuration dictionary containing dataframe, numeric column, and categorical columns.

        Returns
        -------
        buttons : list
            A list of button objects for updating the layout of a figure.
        """
        buttons = []
        buttons.append(dict(label='Ver', method='restyle', args=[{'orientation': ['v'] * 3 + ['v'] * 3
                                                                  + ['v'] * config['traces_cnt12'] * 3
                                                                  + ['v'] * config['traces_cnt21'] * 3}
                                                                 ]))
        buttons.append(dict(label='Hor', method='restyle', args=[{'orientation': ['h'] * 3 + ['h'] * 3
                                                                  + ['h'] * config['traces_cnt12'] * 3
                                                                  + ['h'] * config['traces_cnt21'] * 3}]))
        buttons.append(dict(label='stack', method='relayout',
                       args=[{'barmode': 'stack'}]))
        buttons.append(dict(label='group', method='relayout',
                       args=[{'barmode': 'group'}]))
        # buttons.append(dict(label='overlay', method='relayout', args=[{'barmode': 'overlay'}]))

    #    add range, distinct count
        for i, func in enumerate(['sum', 'mean', 'median', 'count', 'nunique', 'mode', 'std', 'min', 'max', 'range']):
            config['func'] = func
            # 12
            config['cat_column_x'] = config['cat_columns'][0]
            config['cat_column_color'] = config['cat_columns'][1]
            df_for_update = prepare_df(config)
            df_num12 = df_for_update.loc['num', :]
            x_12 = df_num12.index.tolist()
            y_12 = df_num12.values.T.tolist()
            name_12 = df_num12.columns.tolist()
            modes_array12 = df_for_update.loc['modes', :].fillna('').values.T
            if func == 'mode':
                text_12 = [
                    [', '.join(map(str, col)) if col else '' for col in row] for row in modes_array12]
                colors = [['orange' if len(col) > 1 else colorway_for_bar[i]
                           for col in row] for i, row in enumerate(modes_array12)]
                colors12 = [{'color': col_list} for col_list in colors]
            else:
                text_12 = [['' for col in row] for row in modes_array12]
                colors = [[colorway_for_bar[i] for col in row]
                          for i, row in enumerate(modes_array12)]
                colors12 = [{'color': col_list} for col_list in colors]

            # 21
            config['cat_column_x'] = config['cat_columns'][1]
            config['cat_column_color'] = config['cat_columns'][0]
            df_for_update = prepare_df(config)
            df_num21 = df_for_update.loc['num', :]
            x_21 = df_num21.index.tolist()
            y_21 = df_num21.values.T.tolist()
            name_21 = df_num21.columns.tolist()
            modes_array21 = df_for_update.loc['modes', :].fillna('').values.T
            if func == 'mode':
                text_21 = [
                    [', '.join(map(str, col)) if col else '' for col in row] for row in modes_array21]
                colors = [['orange' if len(col) > 1 else colorway_for_bar[i]
                           for col in row] for i, row in enumerate(modes_array21)]
                colors21 = [{'color': col_list} for col_list in colors]
            else:
                text_21 = [[''for col in row] for row in modes_array21]
                colors = [[colorway_for_bar[i] for col in row]
                          for i, row in enumerate(modes_array21)]
                colors21 = [{'color': col_list} for col_list in colors]
            # 1
            config['cat_column_x'] = config['cat_columns'][0]
            config['cat_column_color'] = ''
            df_for_update = prepare_df(config)
            x_1 = df_for_update.index.tolist()
            y_1 = df_for_update[config['num_column_y']].values.tolist()
            modes_array1 = df_for_update['modes'].to_list()
            if func == 'mode':
                text_1 = [[', '.join(map(str, x)) for x in modes_array1]]
                colors_1 = [{'color': ['orange' if len(
                    x) > 1 else colorway_for_bar[0] for x in modes_array1]}]
            else:
                text_1 = ['']
                colors_1 = [{'color': [colorway_for_bar[0]
                                       for x in modes_array1]}]
            # 2
            config['cat_column_x'] = config['cat_columns'][1]
            config['cat_column_color'] = ''
            df_for_update = prepare_df(config)
            x_2 = df_for_update.index.tolist()
            y_2 = df_for_update[config['num_column_y']].values.tolist()
            modes_array2 = df_for_update['modes'].to_list()
            if func == 'mode':
                text_2 = [[', '.join(map(str, x)) for x in modes_array2]]
                colors_2 = [{'color': ['orange' if len(
                    x) > 1 else colorway_for_bar[0] for x in modes_array2]}]
            else:
                text_2 = ['']
                colors_2 = [{'color': [colorway_for_bar[0]
                                       for x in modes_array2]}]

            args = [{
                'orientation': ['v'] * 3 + ['v'] * 3
                + ['v'] * config['traces_cnt12'] * 3
                # для каждго trace должент быть свой x, поэтому x умножаем на количество trace
                + ['v'] * config['traces_cnt21'] * 3, 'x': [x_1] * 3 + [x_2] * 3
                + [x_12] * config['traces_cnt12'] * 3
                # для y1 и y2 нужно обренуть в список
                # для 1 и 2 нет цветов, поэтому названия делаем пустыми
                + [x_21] * config['traces_cnt21'] * 3, 'y': [y_1] * 3 + [y_2] * 3 + y_12 * 3 + y_21 * 3, 'name': [''] * 3 + [''] * 3 + name_12 * 3 + name_21 * 3 + [''] + [''] + [''], 'text': text_1 * 3 + text_2 * 3 + text_12 * 3 + text_21 * 3, 'marker': colors_1 * 3 + colors_2 * 3 + colors12 * 3 + colors21 * 3, 'textposition': 'none', 'textfont': {'color': 'black'}
            }, {'title': f"num = {config['num_column_y']}&nbsp;&nbsp;&nbsp;&nbsp; cat1 = {config['cat_columns'][0]}&nbsp;&nbsp;&nbsp;&nbsp; cat2 = {config['cat_columns'][1]}", 'updatemenus[0].active': 0}
            ]
            if func == 'mode':
                args[0]['hovertemplate'] = ['x=%{x}<br>y=%{y}<br>modes=%{text}'] * 6 \
                    + ['x=%{x}<br>y=%{y}<br>color=%{data.name}<br>modes=%{text}'] * \
                    (config['traces_cnt12'] + config['traces_cnt21']) * 3
            else:
                args[0]['hovertemplate'] = ['x=%{x}<br>y=%{y}'] * 6 \
                    + ['x=%{x}<br>y=%{y}<br>color=%{data.name}'] * \
                    (config['traces_cnt12'] + config['traces_cnt21']) * 3

            buttons.append(dict(label=f'{func.capitalize()}', method='update'                                # , args2=[{'orientation': 'h'}, {'title': f"{func} &nbsp;&nbsp;&nbsp;&nbsp;num = {config['num_column_y']}&nbsp;&nbsp;&nbsp;&nbsp; cat1 = {config['cat_columns'][0]}&nbsp;&nbsp;&nbsp;&nbsp; cat2 = {config['cat_columns'][1]}"}]
                                , args=args))

        traces_visible = {'1b': [[False]], '1l': [[False]], '1a': [[False]], '2b': [[False]], '2l': [[False]], '2a': [[False]], '12b': [[False] * config['traces_cnt12']], '12l': [[False] * config['traces_cnt12']], '12a': [[False] * config['traces_cnt12']], '21b': [[False] * config['traces_cnt21']], '21l': [[False] * config['traces_cnt21']], '21a': [[False] * config['traces_cnt21']]
                          }
        traces_visible_df = pd.DataFrame(traces_visible)
        traces_lables_bar = {'1b': '1', '2b': '2', '12b': '12', '21b': '21'}
        traces_lables_line = {'1l': '1', '2l': '2', '12l': '12', '21l': '21'}
        traces_lables_area = {'1a': '1', '2a': '2', '12a': '12', '21a': '21'}
        traces_lables = {**traces_lables_bar, **
                         traces_lables_line, **traces_lables_area}
        for button_label in traces_lables:
            traces_visible_df_copy = traces_visible_df.copy()
            traces_visible_df_copy[button_label] = traces_visible_df_copy[button_label].apply(
                lambda x: [True for _ in x])
            visible_mask = [
                val for l in traces_visible_df_copy.loc[0].values for val in l]
            data = {'visible': visible_mask, 'xaxis': {
                'visible': False}, 'yaxis': {'visible': False}}
            visible = True if button_label in list(
                traces_lables_bar.keys()) else False
            buttons.append(dict(label=traces_lables[button_label], method='restyle', args=[
                           data], visible=visible))

        buttons.append(dict(
            label='Bar',
            method='relayout',
            args=[{**{f'updatemenus[2].buttons[{i}].visible': True for i in range(
                4)}, **{f'updatemenus[2].buttons[{i}].visible': False for i in range(4, 12)}}]
        ))
        buttons.append(dict(
            label='Line',
            method='relayout',
            args=[{**{f'updatemenus[2].buttons[{i}].visible': False for i in list(range(4)) + list(
                range(8, 12))}, **{f'updatemenus[2].buttons[{i}].visible': True for i in range(4, 8)}}]
        ))
        buttons.append(dict(
            label='Area',
            method='relayout',
            args=[{**{f'updatemenus[2].buttons[{i}].visible': False for i in range(
                8)}, **{f'updatemenus[2].buttons[{i}].visible': True for i in range(8, 12)}}]
        ))

        return buttons

    def update_layout_bars_lines_ares(fig, buttons):
        """
        Update the layout of a figure with bar, line, and area traces based on the provided buttons.

        Parameters
        ----------
        fig : plotly.graph_objects.Figure
            A figure object containing bar, line, and area traces.
        buttons : list
            A list of button objects for updating the layout of a figure.
        """
        fig.update_layout(
            updatemenus=[
                dict(
                    type="buttons",
                    direction="left",
                    buttons=buttons[:4],  # first 3 buttons (Bar, Line, Area)
                    pad={"r": 10, "t": 70},
                    showactive=True,
                    x=0,
                    xanchor="left",
                    y=1.05,
                    yanchor="bottom"
                ),
                dict(
                    type="buttons",
                    direction="left",
                    buttons=buttons[4:14],  # first 3 buttons (Bar, Line, Area)
                    pad={"l": 240, "r": 10, "t": 70},
                    showactive=True,
                    x=0,
                    xanchor="left",
                    y=1.05,
                    yanchor="bottom"
                ),
                dict(
                    type="buttons",
                    direction="left",
                    # first 3 buttons (Bar, Line, Area)
                    buttons=buttons[14:26],
                    pad={"r": 10, "t": 70},
                    showactive=True,
                    x=0,
                    xanchor="left",
                    y=1.2,
                    yanchor="bottom"
                ),
                dict(
                    type="buttons",
                    direction="left",
                    buttons=buttons[26:],  # first 3 buttons (Bar, Line, Area)
                    pad={"l": 180, "r": 10, "t": 70},
                    showactive=True,
                    x=0,
                    xanchor="left",
                    y=1.2,
                    yanchor="bottom"
                ),
            ]
        )

    def create_buttons_heatmap_treemap_sankey(config):
        """
        Create buttons for updating the layout of a figure with heatmap, treemap, and sankey diagrams.

        Parameters
        ----------
        config : dict
            Configuration dictionary containing dataframe, numeric column, and categorical columns.

        Returns
        -------
        buttons : list
            A list of button objects for updating the layout of a figure.
        """
        buttons = []
        buttons.append(dict(label='Ver', method='restyle', args=[
                       {'orientation': ['v'] + ['v'] + ['h']}]))
        buttons.append(dict(label='Hor', method='restyle', args=[
                       {'orientation': ['h'] + ['h'] + ['h']}]))
        # buttons.append(dict(label='overlay', method='relayout', args=[{'barmode': 'overlay'}]))

        buttons.append(dict(label='heatmap', method='update', args=[{'visible': [
                       True, False, False]}, {'xaxis': {'visible': True}, 'yaxis': {'visible': True}}]))
        buttons.append(dict(label='treemap', method='update', args=[{'visible': [
                       False, True, False]}, {'xaxis': {'visible': False}, 'yaxis': {'visible': False}}]))
        buttons.append(dict(label='sankey', method='update', args=[{'visible': [
                       False, False, True]}, {'xaxis': {'visible': False}, 'yaxis': {'visible': False}}]))
    #    add range, distinct count
        for i, func in enumerate(['sum', 'mean', 'count', 'nunique']):
            config['func'] = func

            # heatmap
            pivot_for_heatmap = config['df'].pivot_table(
                index=config['cat_columns'][0], columns=config['cat_columns'][1], values=config['num_column_y'], aggfunc=func)
            x_heatmap = pivot_for_heatmap.index.tolist()
            y_heatmap = pivot_for_heatmap.columns.tolist()
            z_heatmap = pivot_for_heatmap.values
            if i % 2 == 0:
                z_heatmap = z_heatmap.T
            # treemap
            df_treemap = prepare_data_treemap(
                config['df'], config['cat_columns'], config['num_column_y'], func)
            treemap_ids = df_treemap['ids'].to_numpy()
            treemap_parents = df_treemap['parents'].to_numpy()
            treemap_labels = df_treemap['labels'].to_numpy()
            treemap_values = df_treemap['values'].to_numpy()

            # sankey
            sankey_dict = sankey(
                config['df'], config['cat_columns'], config['num_column_y'], func, mode='data')
            sankey_df = sankey_dict['sankey_df']
            nodes_with_indexes = sankey_dict['nodes_with_indexes']
            node_colors = sankey_dict['node_colors']
            link_color = sankey_dict['link_color']
            link = dict(
                source=sankey_df['source'],
                target=sankey_df['target'],
                value=sankey_df['value'],
                label=sankey_df['value_percent'],
                color=link_color
            )
            sankey_labels = list(nodes_with_indexes.keys())

            buttons.append(dict(label=f'{func.capitalize()}', method='update', args2=[{'orientation': 'h'}]  # , {'title': f"{func} &nbsp;&nbsp;&nbsp;&nbsp;num = {config['num_column_y']}&nbsp;&nbsp;&nbsp;&nbsp; cat1 = {config['cat_columns'][0]}&nbsp;&nbsp;&nbsp;&nbsp; cat2 = {config['cat_columns'][1]}"}]

                                , args=[{
                                    # для y1 и y2 нужно обренуть в список
                                    # , 'z': [z_heatmap]
                                    # , 'orientation': 'v'
                                    # treemap
                                    # sankey
                                    # , 'layout.annotations': annotations
                                    # для 1 и 2 нет цветов, поэтому названия делаем пустыми
                                    'orientation': 'h', 'x': [x_heatmap] + [None] + [None], 'y': [y_heatmap] + [None] + [None], 'z': [z_heatmap] + [None] + [None], 'ids': [None] + [treemap_ids] + [None], 'labels': [None] + [treemap_labels] + [None], 'parents': [None] + [treemap_parents] + [None], 'values': [None] + [treemap_values] + [None], 'label': [None] + [None] + [sankey_labels], 'color':[None] + [None] + [node_colors], 'link': [None] + [None] + [link], 'hovertemplate':  ['x=%{x}<br>y=%{y}<br>z=%{z:.0f}']
                                    + ['%{label}<br>%{value}'] + [None]}, {'title': f"num = {config['num_column_y']}&nbsp;&nbsp;&nbsp;&nbsp; cat1 = {config['cat_columns'][0]}&nbsp;&nbsp;&nbsp;&nbsp; cat2 = {config['cat_columns'][1]}", 'updatemenus[0].active': 0}
            ]))

        buttons.append(dict(
            label='Small',
            method='relayout',
            args=[{'height': 600}],
            args2=[{'height': 800}]
        ))

        return buttons

    def update_layout_heatmap_treemap_sankey(fig, buttons):
        """
        Update the layout of a figure with heatmap, treemap, and sankey diagrams based on the provided buttons.

        Parameters
        ----------
        fig : plotly.graph_objects.Figure
            A figure object containing heatmap, treemap, and sankey diagrams.
        buttons : list
            A list of button objects for updating the layout of a figure.
        """
        fig.update_layout(
            updatemenus=[
                dict(
                    type="buttons",
                    direction="left",
                    buttons=buttons[:2],  # first 3 buttons (Bar, Line, Area)
                    pad={"r": 10, "t": 70},
                    showactive=True,
                    x=0,
                    xanchor="left",
                    y=1.05,
                    yanchor="bottom"
                ),
                dict(
                    type="buttons",
                    direction="left",
                    buttons=buttons[2:5],  # first 3 buttons (Bar, Line, Area)
                    pad={"l": 120, "r": 10, "t": 70},
                    showactive=True,
                    x=0,
                    xanchor="left",
                    y=1.05,
                    yanchor="bottom"
                ),
                dict(
                    type="buttons",
                    direction="left",
                    buttons=buttons[5:8],  # first 3 buttons (Bar, Line, Area)
                    pad={"l": 380, "r": 10, "t": 70},
                    showactive=True,
                    x=0,
                    xanchor="left",
                    y=1.05,
                    yanchor="bottom"
                ),
                dict(
                    type="buttons",
                    direction="left",
                    buttons=[buttons[8]],  # first 3 buttons (Bar, Line, Area)
                    pad={"l": 600, "r": 10, "t": 70},
                    showactive=True,
                    x=0,
                    xanchor="left",
                    y=1.05,
                    yanchor="bottom"
                ),
            ]
        )

    fig = create_bars_lines_area_figure(config)
    buttons = create_buttons_bars_lines_ares(config)
    update_layout_bars_lines_ares(fig, buttons)
    fig.update_layout(height=600, title={'text': f"num = {config['num_column_y']}&nbsp;&nbsp;&nbsp;&nbsp; cat1 = {config['cat_columns'][0]}&nbsp;&nbsp;&nbsp;&nbsp; cat2 = {config['cat_columns'][1]}", 'y': 0.92}, xaxis={'title': None}, yaxis={'title': None}
                      #   , margin=dict(l=50, r=50, b=50, t=70),
                      )
    fig.show()

    fig = create_heatmap_treemap_sankey_figure(config)
    buttons = create_buttons_heatmap_treemap_sankey(config)
    update_layout_heatmap_treemap_sankey(fig, buttons)

    fig.update_layout(height=600, title={'text': f"num = {config['num_column_y']}&nbsp;&nbsp;&nbsp;&nbsp; cat1 = {config['cat_columns'][0]}&nbsp;&nbsp;&nbsp;&nbsp; cat2 = {config['cat_columns'][1]}", 'y': 0.92}, xaxis={'title': None}, yaxis={'title': None}
                      #   , margin=dict(l=50, r=50, b=50, t=70),
                      )
    fig.show()


def graph_analysis_gen(df):
    category_columns = [
        col for col in df.columns if pd.api.types.is_categorical_dtype(df[col])]
    num_columns = [
        col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
    c2 = itertools.combinations(category_columns, 2)
    for cat_pair in c2:
        for num_column in num_columns:
            # print(list(cat_pair) + num_column)
            graph_analysis(df, list(cat_pair), num_column)
            yield [num_column] + list(cat_pair)


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
        print('p-value = ', p_value)
        if p_value < alpha:
            print(colored(
                "Отклоняем нулевую гипотезу, поскольку p-value меньше уровня значимости", 'red'))
        else:
            print(colored(
                "Нет оснований отвергнуть нулевую гипотезу, поскольку p-value больше или равно уровню значимости", 'green'))
    else:
        return chi2, p_value, dof, expected


def ttest_ind_df(df: pd.DataFrame, alpha: float = 0.05, equal_var=False, alternative: str = 'two-sided', return_results: bool = False) -> None:
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

    statistic, p_value = stats.ttest_ind(
        sample1, sample2, equal_var=equal_var, alternative=alternative)

    if not return_results:
        print('T-критерий')
        print('p-value = ', p_value)
        print('alpha = ', alpha)
        print('beta = ', beta)

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
    statistic, p_value = stats.ttest_ind(
        sample1, sample2, equal_var=equal_var, alternative=alternative)

    if not return_results:
        print('T-критерий Уэлча')
        print('p-value = ', p_value)
        print('alpha = ', alpha)
        print('beta = ', beta)

        if p_value < alpha:
            print(colored(
                "Отклоняем нулевую гипотезу, поскольку p-value меньше уровня значимости", 'red'))
        else:
            print(colored(
                "Нет оснований отвергнуть нулевую гипотезу, поскольку p-value больше или равно уровню значимости", 'green'))
    else:
        return statistic, p_value, beta, cohens_d


def mannwhitneyu_df(df: pd.DataFrame, alpha: float = 0.05, alternative: str = 'two-sided', return_results: bool = False) -> None:
    """
    Perform the Mann-Whitney U rank test on two independent samples.

    Parameters:
    - df (pd.DataFrame): DataFrame containing two columns,
        where the first column contains sample labels (e.g., "male" and "female") and
        the second column contains corresponding values
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
        print('p-value = ', p_value)
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
        print('p-value = ', p_value)
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
        print('p-value = ', p_value)
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
        print('p-value = ', p_value)
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
        print('p-value = ', p_value)
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
        print('p-value = ', p_value)
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
        print('p-value = ', p_value)
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
        print('p-value = ', p_value)
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
        print('p-value = ', p_value)
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
        print('p-value = ', p_value)
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
        print('p-value = ', p_value)
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
        print('p-value = ', p_value)
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
        print('p-value = ', p_value)
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
        print('p-value = ', p_value)
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
        print('p-value = ', p_value)
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
        print('p-value = ', p_value)
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


def confint_t_2samples_df(df: pd.DataFrame, alpha: float = 0.05, alternative: str = 'two-sided', equal_var=False) -> tuple:
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
                           num_boot: int = 1000,
                           alpha: float = 0.05,
                           p_value_method: str = 'normal_approx',
                           plot: bool = True,
                           return_boot_data: bool = False,
                           return_results: bool = False) -> tuple:
    """
    Perform bootstrap resampling to estimate the difference of a statistic between two samples.

    Parameters:
    - sample1, sample2: pd.Series, two samples to compare
    - stat_func: callable, statistical function to apply to each bootstrap sample (default: np.mean)
    - bootstrap_conf_level: float, significance level for confidence interval (must be between 0 and 1 inclusive) (default: 0.95)
    - num_boot: int, number of bootstrap iterations (default: 1000)
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
                      annotation_text=f"{ci_lower:.2f}", annotation_position='top', line_dash="dash")
        fig.add_vline(x=ci_upper, line_width=2, line_color="#049CB3",
                      annotation_text=f"{ci_upper:.2f}", annotation_position='top', line_dash="dash")
        fig.update_annotations(font_size=16)
        fig.update_traces(
            hovertemplate='Количество = %{y}<br>Разница = %{text}', textposition='none')
        # Remove gap between bars and show white line
        fig.update_layout(
            width=800,
            bargap=0,
            xaxis_title="Разница",
            yaxis_title="Количество",
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
    for i in tqdm(range(num_boot), desc="Bootstrapping"):
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
        print('p-value = ', p_value)
        print('ci = ', ci)
        if p_value < alpha:
            print(colored(
                "Отклоняем нулевую гипотезу, поскольку p-value меньше уровня значимости", 'red'))
        else:
            print(colored(
                "Нет оснований отвергнуть нулевую гипотезу, поскольку p-value больше или равно уровню значимости", 'green'))
        plot_data(boot_data).show()
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
                            return_results: bool = False) -> tuple:
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
    for i in tqdm(range(num_boot), desc="Bootstrapping"):
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


def check_duplicated_value_in_df(df):
    '''
    Функция проверяет на дубли столбцы датафрейма и выводит количество дублей в каждом столбце
    '''
    cnt_duplicated = pd.Series(dtype=int)
    size = df.shape[0]
    for col in df.columns:
        is_duplicated = df[col].duplicated()
        if is_duplicated.any():
            cnt_duplicated[col] = df[is_duplicated].shape[0]
    display(cnt_duplicated.apply(lambda x: f'{x} ({(x / size):.2%})').to_frame().style
            .set_caption('Duplicates')
            .set_table_styles([{'selector': 'caption',
                                'props': [('font-size', '18px'), ("text-align", "left"), ("font-weight", "bold")]}])
            .hide(axis="columns"))


def check_negative_value_in_df(df):
    '''
    Функция проверяет на негативные значения числовые столбцы датафрейма и выводит количество отрицательных значений
    '''
    size = df.shape[0]
    num_columns = [
        col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
    df_num_columns = df[num_columns]
    negative = (df_num_columns < 0).sum()
    display(negative[negative != 0].apply(
        lambda x: f'{x} ({(x / size):.1%})').to_frame(name='negative'))


def check_zeros_value_in_df(df):
    '''
    Функция проверяет на нулевые значения числовые столбцы датафрейма и выводит количество нулевых значений
    '''
    size = df.shape[0]
    num_columns = [
        col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
    df_num_columns = df[num_columns]
    zeros = (df_num_columns < 0).sum()
    display(zeros[zeros != 0].apply(
        lambda x: f'{x} ({(x / size):.1%})').to_frame(name='zeros'))


def check_missed_value_in_df(df):
    '''
    Функция проверяет на пропуски датафрейме и выводит количество пропущенных значений
    '''
    size = df.shape[0]
    missed = df.isna().sum()
    display(missed[missed != 0].apply(
        lambda x: f'{x} ({(x / size):.1%})').to_frame(name='missed'))


def normalize_string_series(column: pd.Series) -> pd.Series:
    """
    Normalize a pandas Series of strings by removing excess whitespace, trimming leading and trailing whitespace,
    and converting all words to lowercase.

    Args:
        column (pd.Series): The input Series of strings to normalize

    Returns:
        pd.Series: The normalized Series of strings
    """
    if not isinstance(column, pd.Series):
        raise ValueError("Input must be a pandas Series")
    if not isinstance(column.dropna().iloc[0], str):
        raise ValueError("Series must contain strings")
    return column.str.lower().str.strip().str.replace(r'\s+', ' ', regex=True)


def analys_column_by_category(df: pd.DataFrame, df_for_analys: pd.DataFrame, column_for_analys: str) -> None:
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
        col for col in df.columns if pd.api.types.is_categorical_dtype(df[col])]
    for category_column in category_columns:
        analys_df = df_for_analys.groupby(
            category_column).size().reset_index(name='count')
        summ_counts = analys_df['count'].sum()
        all_df = df.groupby(
            category_column).size().reset_index(name='total')
        result_df = pd.merge(analys_df, all_df, on=category_column)
        result_df['count_in_total_pct'] = (
            result_df['count'] / result_df['total'])
        result_df['count_in_sum_count_pct'] = (
            result_df['count'] / summ_counts)
        result_df['total_in_sum_total_pct'] = (
            result_df['total'] / size_all)
        result_df['diff_sum_pct'] = result_df['count_in_sum_count_pct'] - \
            result_df['total_in_sum_total_pct']
        display(result_df[[category_column, 'total', 'count', 'count_in_total_pct', 'count_in_sum_count_pct', 'total_in_sum_total_pct', 'diff_sum_pct']].style
                .set_caption(f'Value in "{column_for_analys}" by category "{category_column}"')
                .set_table_styles([{'selector': 'caption',
                                    'props': [('font-size', '18px'), ("text-align", "left"), ("font-weight", "bold")]}])
                .format('{:.1%}', subset=['count_in_total_pct', 'count_in_sum_count_pct', 'total_in_sum_total_pct', 'diff_sum_pct'])
                .hide(axis="index"))
        yield


def analys_by_category_gen(df, series_for_analys):
    '''
    Генератор.
    Для каждой колонки в series_for_analys функция выводит выборку датафрейма.  
    И затем выводит информацию по каждой категории в таблице.
    '''
    for col in series_for_analys.index:
        if not series_for_analys[col][col].value_counts().empty:
            print(f'Value counts outliers')
            display(series_for_analys[col][col].value_counts().to_frame('outliers').head(10).style.set_caption(f'{col}')
                    .set_table_styles([{'selector': 'caption',
                                        'props': [('font-size', '18px'), ("text-align", "left"), ("font-weight", "bold")]
                                        }]))
            yield
        display(series_for_analys[col].sample(10).style.set_caption(f'Sample outliers in {col}').set_table_styles([{'selector': 'caption',
                                                                                                                  'props': [('font-size', '18px'), ("text-align", "left"), ("font-weight", "bold")]}]))
        yield
        gen = analys_column_by_category(
            df, series_for_analys[col], col)
        for _ in gen:
            yield


def check_group_count(df, category_columns, value_column):
    '''
    Функция выводит информацию о количестве элементов в группах.  
    Это функция нужна для  проверки того, что количество элементов в группах соответствует ожидаемому  
    для заполнения пропусков через группы.
    '''
    temp = df.groupby(category_columns)[value_column].agg(
        lambda x: 1 if x.isna().sum() else -1).dropna()
    # -1 это группы без пропусков
    group_with_miss = (temp != -1).sum() / temp.size
    print(f'{group_with_miss:.2%} groups have missing values')
    # Посмотрим какой процент групп с пропусками имеют больше 30 элементов
    temp = df.groupby(category_columns)[value_column].agg(
        lambda x: x.count() > 30 if x.isna().sum() else -1).dropna()
    temp = temp[temp != -1]
    group_with_more_30_elements = (temp == True).sum() / temp.size
    print(f'{group_with_more_30_elements:.2%}  groups with missings have more than 30 elements')
    # Посмотрим какой процент групп с пропусками имеют больше 10 элементов
    temp = df.groupby(category_columns)[value_column].agg(
        lambda x: x.count() > 10 if x.isna().sum() else -1).dropna()
    temp = temp[temp != -1]
    group_with_more_10_elements = (temp == True).sum() / temp.size
    print(f'{group_with_more_10_elements:.2%}  groups with missings have more than 10 elements')
    # Посмотрим какой процент групп с пропусками имеют больше 5 элементов
    temp = df.groupby(category_columns)[value_column].agg(
        lambda x: x.count() > 5 if x.isna().sum() else -1).dropna()
    temp = temp[temp != -1]
    group_with_more_5_elements = (temp == True).sum() / temp.size
    print(f'{group_with_more_5_elements:.2%}  groups with missings have more than 5 elements')
    # Посмотрим какой процент групп содержат только NA
    temp = df.groupby(category_columns)[value_column].agg(
        lambda x: x.count() if x.isna().sum() else -1).dropna()
    temp = temp[temp != -1]
    group_with_ontly_missings = (temp == 0).sum() / temp.size
    print(f'{group_with_ontly_missings:.2%}  groups have only missings')
    # Посмотрим сколько всего значений в группах, где только прпоуски
    temp = df.groupby(category_columns)[value_column].agg(
        lambda x: -1 if x.count() else x.isna().sum()).dropna()
    temp = temp[temp != -1]
    missing_cnt = temp.sum()
    print(f'{missing_cnt:.0f} missings in groups with only missings')


def fill_na_with_function_by_categories(df, category_columns, value_column, func='median', minimal_group_size=10):
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

    available_funcs = {'median', 'mean', 'max', 'min'}

    if isinstance(func, str):
        if func not in available_funcs:
            raise ValueError(f"Unknown function: {func}")
        # If func is a string, use the corresponding pandas method
        return df.groupby(category_columns)[value_column].transform(
            lambda x: x.fillna(x.apply(func)) if x.count() >= minimal_group_size else x)
    else:
        # If func is a callable, apply it to each group of values
        return df.groupby(category_columns)[value_column].transform(
            lambda x: x.fillna(func(x)) if x.count() >= minimal_group_size else x)


def quantiles_columns(column, quantiles=[0.05, 0.25, 0.5, 0.75, 0.95]):
    max_ = pretty_value(column.max())
    column_summary = pd.DataFrame({'Max': [max_]})
    for quantile in quantiles:
        column_summary[f'{quantile * 100:.0f}'] = pretty_value(
            column.quantile(quantile))
    min_ = pretty_value(column.min())
    column_summary['Min'] = min_
    display(column_summary.T.reset_index().style
            .set_caption(f'Quantiles')
            .set_table_styles([{'selector': 'caption',
                                'props': [('font-size', '15px')]
                                }])
            .set_properties(**{'text-align': 'left'})
            .hide(axis="columns")
            .hide(axis="index")
            )


def top_n_values_gen(df: pd.DataFrame, value_column: str, n: int = 10, threshold: int = 20, func='sum'):
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
        col for col in df.columns if pd.api.types.is_categorical_dtype(df[col])]
    for column in categroy_columns:
        if df[column].nunique() > threshold:
            # Группируем данные по столбцу column и рассчитываем сумму по столбцу value_column
            display(df.groupby(column)[value_column].agg(func).sort_values(ascending=False).head(n).to_frame().reset_index().style
                    .set_caption(f'Top in "{column}"')
                    .set_table_styles([{'selector': 'caption',
                                        'props': [('font-size', '18px'), ("text-align", "left"), ("font-weight", "bold")]}])
                    .format('{:.2f}', subset=value_column)
                    .hide(axis="index"))
            yield


def bar(config: dict, titles_for_axis: dict = None):
    """
    Creates a bar chart using the Plotly Express library.

    Parameters:
    config (dict): A dictionary containing parameters for creating the chart.
        - df (DataFrame): A DataFrame containing data for creating the chart.
        - x (str): The name of the column in the DataFrame to be used for creating the X-axis.
        - x_axis_label (str): The label for the X-axis.
        - y (str): The name of the column in the DataFrame to be used for creating the Y-axis.
        - y_axis_label (str): The label for the Y-axis.
        - category (str): The name of the column in the DataFrame to be used for creating categories.  
        If None or an empty string, the chart will be created without category.
        - category_axis_label (str): The label for the categories.
        - title (str): The title of the chart.
        - func (str): The function to be used for aggregating data (default is 'mean').
        - barmode (str): The mode for displaying bars (default is 'group').
        - width (int): The width of the chart (default is None).
        - height (int): The height of the chart (default is None).
        - text (bool):  Whether to display text on the chart (default is False).
        - textsize (int): Text size (default 14)
        - xaxis_show (bool):  Whether to show the X-axis (default is True).
        - yaxis_show (bool):  Whether to show the Y-axis (default is True).
        - showgrid_x (bool):   Whether to show grid on X-axis (default is True).
        - showgrid_y (bool):   Whether to show grid on Y-axis (default is True).

    titles_for_axis (dict):  A dictionary containing titles for the axes.

    Returns:
    fig (plotly.graph_objs.Figure): The created chart.

    Example:
    titles_for_axis = dict(
        # numeric column (0 - средний род, 1 - мужской род, 2 - женский род) (Середнее образовние, средний доход, средняя температура) )
        children = ['Количество детей', 'количество детей', 0]
        , age = ['Возраст, лет', 'возраст', 1]
        , total_income = ['Ежемесячный доход', 'ежемесячный доход', 1]    
        # category column
        , education = ['Уровень образования', 'уровня образования']
        , family_status = ['Семейное положение', 'семейного положения']
        , gender = ['Пол', 'пола']
        , income_type = ['Тип занятости', 'типа занятости']
        , debt = ['Задолженность (1 - имеется, 0 - нет)', 'задолженности']
        , purpose = ['Цель получения кредита', 'цели получения кредита']
        , dob_cat = ['Возрастная категория, лет', 'возрастной категории']
        , total_income_cat = ['Категория дохода', 'категории дохода']
    )
    config = dict(
        df = df
        , x = 'education'  
        , x_axis_label = 'Образование'
        , y = 'total_income'
        , y_axis_label = 'Доход'
        , category = 'gender'
        , category_axis_label = 'Пол'
        , title = 'Доход в зависимости от пола и уровня образования'
        , func = 'mean'
        , barmode = 'group'
        , width = None
        , height = None
        , orientation = 'v'
        , text = False
        , textsize = 14
    )
    bar(config)
    """
    # Проверка входных данных
    if not isinstance(config, dict):
        raise TypeError("config must be a dictionary")
    if 'df' not in config or not isinstance(config['df'], pd.DataFrame):
        raise ValueError("df must be a pandas DataFrame")
    if 'x' not in config or not isinstance(config['x'], str):
        raise ValueError("x must be a string")
    if 'y' not in config or not isinstance(config['y'], str):
        raise ValueError("y must be a string")
    if 'func' in config and not isinstance(config['func'], str):
        raise ValueError("func must be a string")
    if 'barmode' in config and not isinstance(config['barmode'], str):
        raise ValueError("barmode must be a string")
    if 'func' not in config:
        config['func'] = 'mean'
    if 'barmode' not in config:
        config['barmode'] = 'group'
    if 'width' not in config:
        config['width'] = None
    if 'height' not in config:
        config['height'] = None
    if 'textsize' not in config:
        config['textsize'] = 14
    if 'xaxis_show' not in config:
        config['xaxis_show'] = True
    if 'yaxis_show' not in config:
        config['yaxis_show'] = True
    if 'showgrid_x' not in config:
        config['showgrid_x'] = True
    if 'showgrid_y' not in config:
        config['showgrid_y'] = True
    if pd.api.types.is_numeric_dtype(config['df'][config['y']]) and 'orientation' in config and config['orientation'] == 'h':
        config['x'], config['y'] = config['y'], config['x']

    if titles_for_axis:
        if config['func'] not in ['mean', 'median', 'sum']:
            raise ValueError("func must be in ['mean', 'median', 'sum']")
        func_for_title = {'mean': ['Среднее', 'Средний', 'Средняя'], 'median': [
            'Медианное', 'Медианный', 'Медианная'], 'sum': ['Суммарное', 'Суммарный', 'Суммарная']}
        config['x_axis_label'] = titles_for_axis[config['x']][0]
        config['y_axis_label'] = titles_for_axis[config['y']][0]
        config['category_axis_label'] = titles_for_axis[config['category']
                                                        ][0] if 'category' in config else None
        func = config['func']
        if pd.api.types.is_numeric_dtype(config['df'][config['y']]):
            numeric = titles_for_axis[config["y"]][1]
            cat = titles_for_axis[config["x"]][1]
            suffix_type = titles_for_axis[config["y"]][2]
        else:
            numeric = titles_for_axis[config["x"]][1]
            cat = titles_for_axis[config["y"]][1]
            suffix_type = titles_for_axis[config["x"]][2]
        title = f'{func_for_title[func][suffix_type]}'
        title += f' {numeric} в зависимости от {cat}'
        if 'category' in config and config['category']:
            title += f' и {titles_for_axis[config["category"]][1]}'
        config['title'] = title
    else:
        if 'x_axis_label' not in config:
            config['x_axis_label'] = None
        if 'y_axis_label' not in config:
            config['y_axis_label'] = None
        if 'category_axis_label' not in config:
            config['category_axis_label'] = None
        if 'title' not in config:
            config['title'] = None
    if 'category' not in config:
        config['category'] = None
        config['category_axis_label'] = None
    if not isinstance(config['category'], str) and config['category'] is not None:
        raise ValueError("category must be a string")

    def human_readable_number(x):
        if x >= 1e6 or x <= -1e6:
            return f"{x/1e6:.1f} M"
        elif x >= 1e3 or x <= -1e3:
            return f"{x/1e3:.1f} k"
        else:
            return f"{x:.1f}"

    def prepare_df(config: dict):
        df = config['df']
        color = [config['category']] if config['category'] else []
        if not (pd.api.types.is_numeric_dtype(df[config['x']]) or pd.api.types.is_numeric_dtype(df[config['y']])):
            raise ValueError("At least one of x or y must be numeric.")
        elif pd.api.types.is_numeric_dtype(df[config['y']]):
            cat_columns = [config['x']] + color
            num_column = config['y']
        else:
            cat_columns = [config['y']] + color
            num_column = config['x']
        func = config.get('func', 'mean')  # default to 'mean' if not provided
        if pd.api.types.is_numeric_dtype(config['df'][config['y']]):
            ascending = False
        else:
            ascending = True
        func_df = (df[[*cat_columns, num_column]]
                   .groupby(cat_columns)
                   .agg(num=(num_column, func), count=(num_column, 'count'))
                   .reset_index())
        func_df['temp'] = func_df.groupby(cat_columns[0])[
            'num'].transform('sum')
        func_df['count'] = func_df['count'].apply(
            lambda x: f'= {x}' if x <= 1e3 else 'больше 1000')
        # func_df['sum_cnt'] = func_df.groupby(cat_columns[0])['cnt'].transform('sum')
        # size = df.shape[0]
        # func_df['sum_cnt_pct'] = func_df['sum_cnt'].apply(lambda x: f'{(x / size):.1%}')
        # func_df['cnt_in_sum_pct'] = (func_df['cnt'] / func_df['sum_cnt']).apply(lambda x: f'{x:.1%}')
        func_df = (func_df.sort_values(['temp', 'num'], ascending=ascending)
                   .drop('temp', axis=1)
                   .rename(columns={'num': num_column})
                   # .sort_values(columns[0], ascending=ascending)
                   )

        return func_df
    df_for_fig = prepare_df(config)
    x = df_for_fig[config['x']].values
    y = df_for_fig[config['y']].values
    x_axis_label = config['x_axis_label']
    y_axis_label = config['y_axis_label']
    color_axis_label = config['category_axis_label']
    color = df_for_fig[config['category']
                       ].values if config['category'] else None
    custom_data = [df_for_fig['count']]
    if 'text' in config and config['text']:
        if pd.api.types.is_numeric_dtype(config['df'][config['y']]):
            text = [human_readable_number(el) for el in y]
        else:
            text = [human_readable_number(el) for el in x]
    else:
        text = None
    fig = px.bar(x=x, y=y, color=color,
                 barmode=config['barmode'], text=text, custom_data=custom_data)
    color = []
    for trace in fig.data:
        color.append(trace.marker.color)
    if x_axis_label:
        hovertemplate_x = f'{x_axis_label} = '
    else:
        hovertemplate_x = f'x = '
    if x_axis_label:
        hovertemplate_y = f'{y_axis_label} = '
    else:
        hovertemplate_y = f'y = '
    if x_axis_label:
        hovertemplate_color = f'<br>{color_axis_label} = '
    else:
        hovertemplate_color = f'color = '
    if pd.api.types.is_numeric_dtype(config['df'][config['y']]):
        hovertemplate = hovertemplate_x + \
            '%{x}<br>' + hovertemplate_y + '%{y:.4s}'
    else:
        hovertemplate = hovertemplate_x + \
            '%{x:.4s}<br>' + hovertemplate_y + '%{y}'
    if config['category']:
        hovertemplate += hovertemplate_color + '%{data.name}'
    hovertemplate += f'<br>Размер группы '
    hovertemplate += '%{customdata[0]}'
    # hovertemplate += f'<br>cnt_in_sum_pct = '
    # hovertemplate += '%{customdata[1]}'
    hovertemplate += '<extra></extra>'
    fig.update_traces(hovertemplate=hovertemplate, textfont=dict(
        family='Open Sans', size=config['textsize']  # Размер шрифта
        # color='black'  # Цвет текста
    ), textposition='auto'  # Положение текстовых меток (outside или inside))
    )
    fig.update_layout(
        # , title={'text': f'<b>{title}</b>'}
        # , margin=dict(l=50, r=50, b=50, t=70)
        width=config['width'], height=config['height'], title_font=dict(size=24, color="rgba(0, 0, 0, 0.6)"), title={'text': config["title"]}, xaxis_title=x_axis_label, yaxis_title=y_axis_label, legend_title_text=color_axis_label, font=dict(size=14, family="Open Sans", color="rgba(0, 0, 0, 1)"), xaxis_title_font=dict(size=18, color="rgba(0, 0, 0, 0.5)"), yaxis_title_font=dict(size=18, color="rgba(0, 0, 0, 0.5)"), xaxis_tickfont=dict(size=14, color="rgba(0, 0, 0, 0.5)"), yaxis_tickfont=dict(size=14, color="rgba(0, 0, 0, 0.5)"), legend_title_font_color='rgba(0, 0, 0, 0.5)', legend_font_color='rgba(0, 0, 0, 0.5)', xaxis_linecolor="rgba(0, 0, 0, 0.5)", yaxis_linecolor="rgba(0, 0, 0, 0.5)", hoverlabel=dict(bgcolor="white"), xaxis=dict(
            visible=config['xaxis_show'], showgrid=config['showgrid_x'], gridwidth=1, gridcolor="rgba(0, 0, 0, 0.1)"
        ), yaxis=dict(
            visible=config['yaxis_show'], showgrid=config['showgrid_y'], gridwidth=1, gridcolor="rgba(0, 0, 0, 0.07)"
        )
    )
    if pd.api.types.is_numeric_dtype(config['df'][config['x']]):
        # Чтобы сортировка была по убыванию вернего значения, нужно отсортировать по последнего значению в x
        traces = list(fig.data)
        traces.sort(key=lambda x: x.x[-1])
        fig.data = traces
        color = color[::-1]
        for i, trace in enumerate(fig.data):
            trace.marker.color = color[i]
        fig.update_layout(legend={'traceorder': 'reversed'})
    return fig


def pairplot(df: pd.DataFrame, titles_for_axis: dict = None):
    """
    Create a pairplot of a given DataFrame with customized appearance.

    Parameters:
    df (pd.DataFrame): The input DataFrame containing numerical variables.
    titles_for_axis (dict):  A dictionary containing titles for the axes.

    Returns:
    None

    Example:
    titles_for_axis = dict(
        # numeric column
        children = 'Кол-во детей'
        , age = 'Возраст'
        , total_income = 'Доход'    
    )
    """
    def human_readable_number(x):
        if x >= 1e6 or x <= -1e6:
            return f"{x/1e6:.1f}M"
        elif x >= 1e3 or x <= -1e3:
            return f"{x/1e3:.1f}k"
        else:
            return f"{x:.1f}"
    g = sns.pairplot(df, markers=["o"],
                     plot_kws={'color': (128/255, 60/255, 170/255, 0.9)},
                     diag_kws={'color': (128/255, 60/255, 170/255, 0.9)})
    for ax in g.axes.flatten():
        xlabel = ax.get_xlabel()
        ylabel = ax.get_ylabel()
        if titles_for_axis:
            if xlabel:
                xlabel = titles_for_axis[xlabel]
            if ylabel:
                ylabel = titles_for_axis[ylabel]
        ax.set_xlabel(xlabel, alpha=0.6)
        ax.set_ylabel(ylabel, alpha=0.6)
        xticklabels = ax.get_xticklabels()
        for label in xticklabels:
            # if label.get_text():
            #     label.set_text(human_readable_number(int(label.get_text().replace('−', '-'))))  # modify the label text
            label.set_alpha(0.6)
        yticklabels = ax.get_yticklabels()
        for label in yticklabels:
            # if label.get_text():
            #     label.set_text(human_readable_number(int(label.get_text().replace('−', '-'))))  # modify the label text
            label.set_alpha(0.6)
        ax.spines['top'].set_alpha(0.3)
        ax.spines['left'].set_alpha(0.3)
        ax.spines['right'].set_alpha(0.3)
        ax.spines['bottom'].set_alpha(0.3)
    g.fig.suptitle('Зависимости между числовыми переменными', fontsize=15,
                   x=0.07, y=1.05, fontfamily='open-sans', alpha=0.7, ha='left')


def histogram(column: pd.Series, titles_for_axis: dict = None, nbins: int = 30, width: int = 800, height: int = None, left_quantile: float = 0, right_quantile: float = 1):
    """
    Plot a histogram of a Pandas Series using Plotly Express.

    Args:
    column (pd.Series): The input Pandas Series.
    titles_for_axis (dict, optional): A dictionary containing the titles for the x-axis and y-axis. Defaults to None.
    nbins (int, optional): The number of bins in the histogram. Defaults to 30.
    width (int, optional): The width of the plot. Defaults to 800.
    height (int, optional): The height of the plot. Defaults to None.
    left_quantile (float, optional): The left quantile for trimming the data. Defaults to 0.
    right_quantile (float, optional): The right quantile for trimming the data. Defaults to 1.

    Returns:
        fig: The Plotly Express figure.
    """
    # Обрезаем данные между квантилями
    trimmed_column = column.between(column.quantile(
        left_quantile), column.quantile(right_quantile))
    column = column[trimmed_column]
    if not titles_for_axis:
        title = f'Гистограмма для {column.name}'
        xaxis_title = 'Значение'
        yaxis_title = 'Частота'
    else:
        title = f'Гистограмма {titles_for_axis[column.name][1]}'
        xaxis_title = f'{titles_for_axis[column.name][0]}'
        yaxis_title = 'Частота'
    fig = px.histogram(column, title=title, histnorm='percent', nbins=nbins)
    fig.update_layout(
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title
    )
    fig.update_traces(
        hovertemplate='Значение = %{x}<br>Частота = %{y:.2f}', showlegend=False)
    fig.update_layout(
        # , title={'text': f'<b>{title}</b>'}
        width=width, height=height, title_font=dict(size=24, color="rgba(0, 0, 0, 0.6)"), xaxis_title=xaxis_title, yaxis_title=yaxis_title, font=dict(size=14, family="Open Sans", color="rgba(0, 0, 0, 1)"), xaxis_title_font=dict(size=18, color="rgba(0, 0, 0, 0.5)"), yaxis_title_font=dict(size=18, color="rgba(0, 0, 0, 0.5)"), xaxis_tickfont=dict(size=14, color="rgba(0, 0, 0, 0.5)"), yaxis_tickfont=dict(size=14, color="rgba(0, 0, 0, 0.5)"), legend_title_font_color='rgba(0, 0, 0, 0.5)', legend_font_color='rgba(0, 0, 0, 0.5)', xaxis_linecolor="rgba(0, 0, 0, 0.5)", yaxis_linecolor="rgba(0, 0, 0, 0.5)"        # , margin=dict(l=50, r=50, b=50, t=70)
        , hoverlabel=dict(bgcolor="white"), xaxis=dict(
            showgrid=True, gridwidth=1, gridcolor="rgba(0, 0, 0, 0.1)"
        ), yaxis=dict(
            showgrid=True, gridwidth=1, gridcolor="rgba(0, 0, 0, 0.07)"
        )
    )
    return fig


def categorical_graph_analys_gen(df, titles_for_axis: dict = None, width=None, height=None):
    """
    Generate graphics for all possible combinations of categorical variables in a dataframe.

    This function takes a pandas DataFrame as input and generates graphics for each pair of categorical variables.
    The heatmap matrix is a visual representation of the cross-tabulation of two categorical variables, which can help identify patterns and relationships between them.

    Parameters:
    df (pandas DataFrame): Input DataFrame containing categorical variables.
    titles_for_axis (dict):  A dictionary containing titles for the axes.

    Returns:
    None

    Example:
    titles_for_axis = dict(
        # numeric column (0 - средний род, 1 - мужской род, 2 - женский род) (Середнее образовние, средний доход, средняя температура) )
        children = ['Количество детей', 'количество детей', 0]
        , age = ['Возраст, лет', 'возраст', 1]
        , total_income = ['Ежемесячный доход', 'ежемесячный доход', 1]    
        # category column
        , education = ['Уровень образования', 'уровня образования']
        , family_status = ['Семейное положение', 'семейного положения']
        , gender = ['Пол', 'пола']
        , income_type = ['Тип занятости', 'типа занятости']
        , debt = ['Задолженность (1 - имеется, 0 - нет)', 'задолженности']
        , purpose = ['Цель получения кредита', 'цели получения кредита']
        , dob_cat = ['Возрастная категория, лет', 'возрастной категории']
        , total_income_cat = ['Категория дохода', 'категории дохода']
    )    
    """
    def human_readable_number(x):
        if x >= 1e6 or x <= -1e6:
            return f"{x/1e6:.1f} M"
        elif x >= 1e3 or x <= -1e3:
            return f"{x/1e3:.1f} k"
        else:
            return f"{x:.0f}"
    # Получаем список категориальных переменных
    categorical_cols = df.select_dtypes(include=['category']).columns
    size = df.shape[0]
    # Перебираем все возможные комбинации категориальных переменных
    for col1, col2 in itertools.combinations(categorical_cols, 2):
        # Создаем матрицу тепловой карты
        crosstab_for_figs = pd.crosstab(df[col1], df[col2])

        fig = go.Figure()

        if not titles_for_axis:
            title_heatmap = f'Тепловая карта долей для {col1} и {col2}'
            title_bar = f'Распределение долей для {col1} и {col2}'
            xaxis_title_heatmap = f'{col2}'
            yaxis_title_heatmap = f'{col1}'
            xaxis_title_for_figs_all = f'{col1}'
            yaxis_title_for_figs_all = 'Доля'
            legend_title_all = f'{col2}'
            xaxis_title_for_figs_normolized_by_col = f'{col1}'
            yaxis_title_for_figs_normolized_by_col = 'Доля'
            legend_title_normolized_by_col = f'{col2}'
            xaxis_title_for_figs_normolized_by_row = f'{col2}'
            yaxis_title_for_figs_normolized_by_row = 'Доля'
            legend_title_normolized_by_row = f'{col1}'
        else:
            title_heatmap = f'Тепловая карта долей для {titles_for_axis[col1][1]} и {titles_for_axis[col2][1]}'
            title_bar = f'Распределение долей для {titles_for_axis[col1][1]} и {titles_for_axis[col2][1]}'
            xaxis_title_heatmap = f'{titles_for_axis[col2][0]}'
            yaxis_title_heatmap = f'{titles_for_axis[col1][0]}'
            xaxis_title_for_figs_all = f'{titles_for_axis[col1][0]}'
            yaxis_title_for_figs_all = 'Доля'
            legend_title_all = f'{titles_for_axis[col2][0]}'
            xaxis_title_for_figs_normolized_by_col = f'{titles_for_axis[col1][0]}'
            yaxis_title_for_figs_normolized_by_col = 'Доля'
            legend_title_normolized_by_col = f'{titles_for_axis[col2][0]}'
            xaxis_title_for_figs_normolized_by_row = f'{titles_for_axis[col2][0]}'
            yaxis_title_for_figs_normolized_by_row = 'Доля'
            legend_title_normolized_by_row = f'{titles_for_axis[col1][0]}'

            # title = f'Тепловая карта количества для {titles_for_axis[col1][1]} и {titles_for_axis[col2][1]}'
            # xaxis_title = f'{titles_for_axis[col1][0]}'
            # yaxis_title = f'{titles_for_axis[col2][0]}'
        # hovertemplate = xaxis_title + ' = %{x}<br>' + yaxis_title + ' = %{y}<br>Количество = %{z}<extra></extra>'

        # all
        size_all = crosstab_for_figs.sum().sum()
        crosstab_for_figs_all = crosstab_for_figs * 100 / size_all
        crosstab_for_figs_all = pd.concat(
            [crosstab_for_figs_all, crosstab_for_figs], axis=1, keys=['data', 'customdata'])
        crosstab_for_figs_all['sum_row'] = crosstab_for_figs_all.sum(axis=1)
        crosstab_for_figs_all = crosstab_for_figs_all.sort_values(
            'sum_row', ascending=False).drop('sum_row', axis=1, level=0)
        crosstab_for_figs_all = pd.concat(
            [crosstab_for_figs_all['data'], crosstab_for_figs_all['customdata']], axis=0, keys=['data', 'customdata'])
        crosstab_for_figs_all = crosstab_for_figs_all.sort_values(
            crosstab_for_figs_all.index[0], axis=1, ascending=False)
        crosstab_for_figs_all = pd.concat(
            [crosstab_for_figs_all.loc['data'], crosstab_for_figs_all.loc['customdata']], axis=1, keys=['data', 'customdata'])
        customdata_all = crosstab_for_figs_all['customdata'].values.T.tolist()
        # col
        col_sum_count = crosstab_for_figs.sum()
        crosstab_for_figs_normolized_by_col = crosstab_for_figs * 100 / col_sum_count
        crosstab_for_figs_normolized_by_col = pd.concat(
            [crosstab_for_figs_normolized_by_col, crosstab_for_figs], axis=1, keys=['data', 'customdata'])
        crosstab_for_figs_normolized_by_col['sum_row'] = crosstab_for_figs_normolized_by_col['data'].sum(
            axis=1)
        crosstab_for_figs_normolized_by_col = crosstab_for_figs_normolized_by_col.sort_values(
            'sum_row', ascending=False).drop('sum_row', axis=1, level=0)
        crosstab_for_figs_normolized_by_col = pd.concat(
            [crosstab_for_figs_normolized_by_col['data'], crosstab_for_figs_normolized_by_col['customdata']], axis=0, keys=['data', 'customdata'])
        crosstab_for_figs_normolized_by_col = crosstab_for_figs_normolized_by_col.sort_values(
            crosstab_for_figs_normolized_by_col.index[0], axis=1, ascending=False)
        crosstab_for_figs_normolized_by_col = pd.concat(
            [crosstab_for_figs_normolized_by_col.loc['data'], crosstab_for_figs_normolized_by_col.loc['customdata']], axis=1, keys=['data', 'customdata'])
        customdata_normolized_by_col = crosstab_for_figs_normolized_by_col['customdata'].values.T.tolist(
        )
        # row
        row_sum_count = crosstab_for_figs.T.sum()
        crosstab_for_figs_normolized_by_row = crosstab_for_figs.T * 100 / row_sum_count
        crosstab_for_figs_normolized_by_row = pd.concat(
            [crosstab_for_figs_normolized_by_row, crosstab_for_figs.T], axis=1, keys=['data', 'customdata'])
        crosstab_for_figs_normolized_by_row['sum_row'] = crosstab_for_figs_normolized_by_row.sum(
            axis=1)
        crosstab_for_figs_normolized_by_row = crosstab_for_figs_normolized_by_row.sort_values(
            'sum_row', ascending=False).drop('sum_row', axis=1, level=0)
        crosstab_for_figs_normolized_by_row = pd.concat(
            [crosstab_for_figs_normolized_by_row['data'], crosstab_for_figs_normolized_by_row['customdata']], axis=0, keys=['data', 'customdata'])
        crosstab_for_figs_normolized_by_row = crosstab_for_figs_normolized_by_row.sort_values(
            crosstab_for_figs_normolized_by_row.index[0], axis=1, ascending=False)
        crosstab_for_figs_normolized_by_row = pd.concat(
            [crosstab_for_figs_normolized_by_row.loc['data'], crosstab_for_figs_normolized_by_row.loc['customdata']], axis=1, keys=['data', 'customdata'])
        customdata_normolized_by_row = crosstab_for_figs_normolized_by_row['customdata'].values.T.tolist(
        )
        # bar
        bar_fig_all = px.bar(
            crosstab_for_figs_all['data'], barmode='group', text_auto=".0f")
        bar_traces_len_all = len(bar_fig_all.data)
        bar_fig_normolized_by_col = px.bar(
            crosstab_for_figs_normolized_by_col['data'], barmode='group', text_auto=".0f")
        bar_traces_len_normolized_by_col = len(bar_fig_normolized_by_col.data)
        bar_fig_normolized_by_row = px.bar(
            crosstab_for_figs_normolized_by_row['data'], barmode='group', text_auto=".0f")
        bar_traces_len_normolized_by_row = len(bar_fig_normolized_by_row.data)
        # heatmap
        heatmap_fig_all = px.imshow(
            crosstab_for_figs_all['data'], text_auto=".0f")
        heatmap_fig_normolized_by_col = px.imshow(
            crosstab_for_figs_normolized_by_col['data'], text_auto=".0f")
        heatmap_fig_normolized_by_row = px.imshow(
            crosstab_for_figs_normolized_by_row['data'], text_auto=".0f")
        # add traces
        heatmap_figs = [
            heatmap_fig_all, heatmap_fig_normolized_by_col, heatmap_fig_normolized_by_row]
        for fig_heatmap, customdata in zip(heatmap_figs, [crosstab_for_figs_all['customdata'].values, crosstab_for_figs_normolized_by_col['customdata'].values, crosstab_for_figs_normolized_by_row['customdata'].values]):
            fig_heatmap.update_traces(hovertemplate=f'{xaxis_title_heatmap}'+' = %{x}<br>'+f'{yaxis_title_heatmap}'+'= %{y}<br>Доля = %{z:.1f} %<br>Количество = %{customdata}<extra></extra>', textfont=dict(
                family='Open Sans', size=14))
            heatmap_traces = fig_heatmap.data
            for trace in heatmap_traces:
                trace.xgap = 3
                trace.ygap = 3
                trace.visible = False
                trace.customdata = customdata
            fig.add_traces(heatmap_traces)

        fig.update_layout(coloraxis=dict(colorscale=[
                          (0, 'rgba(204, 153, 255, 0.1)'), (1, 'rgb(127, 60, 141)')]), hoverlabel=dict(bgcolor='white'))
        for fig_i, (fig_bar, customdata) in enumerate(zip([bar_fig_all, bar_fig_normolized_by_col, bar_fig_normolized_by_row], [customdata_all, customdata_normolized_by_col, customdata_normolized_by_row])):
            fig_bar.update_traces(hovertemplate=f'{xaxis_title_for_figs_all}'+' = %{x}<br>'+f'{legend_title_all}' +
                                  '= %{data.name}<br>Доля = %{y:.1f} %<br>Количество = %{customdata}<extra></extra>')
            bar_traces = fig_bar.data
            for i, trace in enumerate(bar_traces):
                if fig_i > 0:
                    trace.visible = False
                trace.customdata = customdata[i]
            fig.add_traces(bar_traces)
        bar_traces_len = len(bar_traces)

        buttons = [
            dict(label="Общее сравнение", method="update", args=[{
                "visible": [False, False, False]
                + [True] * bar_traces_len_all
                + [False] * bar_traces_len_normolized_by_col
                + [False] * bar_traces_len_normolized_by_row, 'xaxis': {'visible': False}, 'yaxis': {'visible': False}, 'hovertemplate': f'{xaxis_title_for_figs_all}'+' = %{x}<br>'+f'{legend_title_all}'+'= %{data.name}<br>Доля = %{y:.1f} %<br>Количество = %{customdata}<extra></extra>'
            }, {'title.text': title_bar, 'xaxis.title': xaxis_title_for_figs_all, 'yaxis.title': yaxis_title_for_figs_all, 'legend.title.text': legend_title_all
                }]), dict(label="Heatmap", method="update", args=[{
                    "visible": [True, False, False]
                    + [False] * bar_traces_len_all
                    + [False] * bar_traces_len_normolized_by_col
                    + [False] * bar_traces_len_normolized_by_row, 'xaxis': {'visible': False}, 'yaxis': {'visible': False}, 'hovertemplate': f'{xaxis_title_heatmap}'+' = %{x}<br>'+f'{yaxis_title_heatmap}'+'= %{y}<br>Доля = %{z:.1f} %<br>Количество = %{customdata}<extra></extra>'
                }, {'title.text': title_heatmap, 'xaxis.title': xaxis_title_heatmap, 'yaxis.title': yaxis_title_heatmap
                    }]), dict(label=f"Сравнение ({xaxis_title_for_figs_normolized_by_col.lower()})", method="update", args=[{
                        "visible": [False, False, False]
                        + [False] *
                        bar_traces_len_all
                        + [True] * bar_traces_len_normolized_by_col
                        + [False] * bar_traces_len_normolized_by_row, 'xaxis': {'visible': False}, 'yaxis': {'visible': False}, 'hovertemplate': f'{xaxis_title_for_figs_normolized_by_col}'+' = %{x}<br>'+f'{legend_title_normolized_by_col}'+'= %{data.name}<br>Доля = %{y:.1f} %<br>Количество = %{customdata}<extra></extra>'
                    }, {'title.text': title_bar, 'xaxis.title': xaxis_title_for_figs_normolized_by_col, 'yaxis.title': yaxis_title_for_figs_normolized_by_col, 'legend.title.text': legend_title_normolized_by_col
                        }]), dict(label="Heatmap", method="update", args=[{
                            "visible": [False, True, False]
                            + [False] * bar_traces_len_all
                            + [False] *
                            bar_traces_len_normolized_by_col
                            + [False] * bar_traces_len_normolized_by_row, 'xaxis': {'visible': False}, 'yaxis': {'visible': False}, 'hovertemplate': f'{xaxis_title_heatmap}'+' = %{x}<br>'+f'{yaxis_title_heatmap}'+'= %{y}<br>Доля = %{z:.1f} %<br>Количество = %{customdata}<extra></extra>'
                        }, {'title.text': title_heatmap, 'xaxis.title': xaxis_title_heatmap, 'yaxis.title': yaxis_title_heatmap
                            }]), dict(label=f"Сравнение ({xaxis_title_for_figs_normolized_by_row.lower()})", method="update", args=[{
                                "visible": [False, False, False]
                                + [False] * bar_traces_len_all
                                + [False] * bar_traces_len_normolized_by_col
                                + [True] * bar_traces_len_normolized_by_row, 'xaxis': {'visible': False}, 'yaxis': {'visible': False}, 'hovertemplate': f'{xaxis_title_for_figs_normolized_by_row}'+' = %{x}<br>'+f'{legend_title_normolized_by_row}'+'= %{data.name}<br>Доля = %{y:.1f} %<br>Количество = %{customdata}<extra></extra>'
                            }, {'title.text': title_bar, 'xaxis.title': xaxis_title_for_figs_normolized_by_row, 'yaxis.title': yaxis_title_for_figs_normolized_by_row, 'legend.title.text': legend_title_normolized_by_row
                                }]), dict(label="Heatmap", method="update", args=[{
                                    "visible": [False, False, True]
                                    + [False] *
                                    bar_traces_len_all
                                    + [False] * bar_traces_len_normolized_by_col
                                    + [False] * bar_traces_len_normolized_by_row, 'xaxis': {'visible': False}, 'yaxis': {'visible': False}, 'hovertemplate': f'{yaxis_title_heatmap}'+' = %{x}<br>'+f'{xaxis_title_heatmap}'+'= %{y}<br>Доля = %{z:.1f} %<br>Количество = %{customdata}<extra></extra>'
                                }, {'title.text': title_heatmap, 'xaxis.title': yaxis_title_heatmap, 'yaxis.title': xaxis_title_heatmap
                                    }])
        ]
        # for button in buttons:
        #     button['font'] = dict(color = "rgba(0, 0, 0, 0.6)")
        # Add the buttons to the figure
        fig.update_layout(
            height=500, updatemenus=[
                dict(
                    type="buttons", font=dict(color="rgba(0, 0, 0, 0.6)"), buttons=buttons, direction="left", pad={"r": 10, "t": 70}, showactive=True, x=0, xanchor="left", y=1.15, yanchor="bottom")], title_font=dict(size=20, color="rgba(0, 0, 0, 0.6)"), font=dict(size=14, family="Open Sans", color="rgba(0, 0, 0, 1)"), title=dict(text=title_bar, y=0.9), xaxis_title=xaxis_title_for_figs_all, yaxis_title=yaxis_title_for_figs_all, legend_title_text=legend_title_all, xaxis_title_font=dict(size=18, color="rgba(0, 0, 0, 0.5)"), yaxis_title_font=dict(size=18, color="rgba(0, 0, 0, 0.5)"), xaxis_tickfont=dict(size=14, color="rgba(0, 0, 0, 0.5)"), yaxis_tickfont=dict(size=14, color="rgba(0, 0, 0, 0.5)"), legend_title_font_color='rgba(0, 0, 0, 0.5)', legend_font_color='rgba(0, 0, 0, 0.5)', xaxis_linecolor="rgba(0, 0, 0, 0.5)", yaxis_linecolor="rgba(0, 0, 0, 0.5)"            # , xaxis=dict(
            #     showgrid=True
            #     , gridwidth=1
            #     , gridcolor="rgba(0, 0, 0, 0.1)"
            # )
            , yaxis=dict(
                showgrid=True, gridwidth=1, gridcolor="rgba(0, 0, 0, 0.07)"
            )            # , margin=dict(l=50, r=50, b=50, t=70)
            , hoverlabel=dict(bgcolor="white"))
        # print(col1, col2)
        yield fig


def add_links_and_numbers_to_headings(notebook_path: str, mode: str = 'draft', link_type: str = "name", start_level: int = 2):
    """
    Добавляет ссылки к заголовкам в ноутбуке.

    Args:
        notebook_path (str): Путь к ноутбуку.
        link_type (str, optional): Тип ссылки. Может быть "name" или "id". Defaults to "name".
        start_level (int): Уровень глав (количество #), с которого начинать нумерацию
        mode (str): Режим работы функции. Либо 'draft', в этом случае создаться копия файла, либо 'final' в этом случае изменения будут сделаны в исходном файле. Default to 'draft'
    Returns:
        None

    Example:
        - Вариант, который работает почти везде
            в оглавлении пишем
            <a href="#Глава-1">Ссылка на главу 1</a>
            в названии главы пишем
            # 1 Введение <a name="Глава-1"></a>
        - Вариант, который не везде работает
            в оглавлении пишем
            [Ссылка на главу 1](#name-id)
            в названии главы пишем
            <a class="anchor" id="Название-главы"></a>
            ### Название главы
        - Упрощенный вариант, работает только в jupyter notebook
            в оглавлении пишем
            [Ссылка на главу 1](#Глава-1)
            в названии главы пишем
            # 1 Введение
    """
    if mode not in ['draft', 'final']:
        raise ValueError(
            "Invalid mode. Mode must be either 'draft' or 'final'.")
    if link_type not in ['name', 'id']:
        raise ValueError(
            "Invalid link_type. link_type must be either 'name' or 'id'.")
    try:
        with open(notebook_path, 'r', encoding='utf-8') as in_f:
            nb_json = json.load(in_f)
    except FileNotFoundError:
        print(f"Error: File not found - {notebook_path}")
        return
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format - {notebook_path}")
        return
    regex_for_sub = re.compile(r'[^0-9a-zA-Zа-яА-Я]')
    # headers = []
    # Создаем список счетчиков для каждого уровня заголовка
    counters = [0] * 100
    is_previous_cell_header = False
    for cell in nb_json["cells"]:
        if cell["cell_type"] == "markdown":
            source = cell["source"]
            new_source = []
            for line in source:
                level = line.count("#")
                if level >= start_level and not line.strip().endswith("</a>") and not line.strip().endswith("<skip>"):
                    # Уменьшаем на 1, чтобы название отчета не нумеровалось
                    level = level - 1
                    title = line.strip().lstrip("#").lstrip()
                    # Обновляем счетчики
                    counters[level - 1] += 1
                    for i in range(level, len(counters)):
                        counters[i] = 0
                    number_parts = [str(counter)
                                    for counter in counters[:level]]
                    chapter_number = ".".join(number_parts)
                    # для глав с одной цифрой добавляем точку, чтобы было 1. Название главы, для остальных уровней в конкце не будет точки
                    if level + 1 == start_level:
                        chapter_number += '.'
                    # Формируем пронумерованный заголовок
                    title_text = f"{(level + 1) * '#'} {chapter_number} {title.strip()}"
                    text_for_ref = f'{chapter_number}-{title.strip()}'
                    text_for_ref = regex_for_sub.sub('-', text_for_ref)
                    if link_type == "name":
                        header = f"{title_text}<a name='{text_for_ref}'></a>"
                    elif link_type == "id":
                        header = f"{title_text}<a class='anchor' id='{text_for_ref}'></a>"
                    else:
                        raise ValueError(
                            "Неправильный тип ссылки. Должно быть 'name' или 'id'.")
                    if not is_previous_cell_header:
                        header += '\n<a href="#ref-to-toc">вернуться к оглавлению</a>'
                    new_source.append(header)
                    is_previous_cell_header = True
                else:
                    is_previous_cell_header = False
                    new_source.append(line)
            cell["source"] = new_source
        else:
            is_previous_cell_header = False

    # Convert nb_json to nbformat object
    nb = nbformat.reads(json.dumps(nb_json), as_version=4)

    # Save the nbformat object to the file
    if mode == 'draft':
        output_filename_splited = notebook_path.split('.')
        output_filename_splited[-2] += '_temp'
        output_filename = '.'.join(output_filename_splited)
    else:
        output_filename = notebook_path
    with open(output_filename, 'w', encoding='utf-8') as out_f:
        nbformat.write(nb, out_f, version=4)
    print(f"Numbers of headers and links added to {output_filename}")


def generate_toc(notebook_path: str, mode: str = 'draft', indent_char: str = "&emsp;", link_type: str = "html", start_level: int = 2):
    """
    Генерирует оглавление для ноутбука.

    Args:
        notebook_path (str): Путь к ноутбуку.
        indent_char (str, optional): Символ для отступа. Defaults to "&emsp;".
        link_type (str, optional): Тип ссылки. Может быть "markdown" или "html". Defaults to "html".
        start_level (int): Уровень глав (количество #), с которого начинать нумерацию
        mode (str): Режим работы функции. Либо 'draft', в этом случае создаться копия файла, либо 'final' в этом случае изменения будут сделаны в исходном файле. Default to 'draft'

    Returns:
        None

    Example:
        Для link_type="markdown":
            &emsp;[Глава-1](#Глава-1)<br>
            &emsp;&emsp;[Подглава-1.1](#Подглава-1.1)<br>
        Для link_type="html":
            &emsp;<a href="Глава-1">Глава 1</a><br>
            &emsp;&emsp;<a href="#Подглава-1.1">Подглава 1.1</a><br>

        - Вариант, который работает почти везде
            в оглавлении пишем
            <a href="#лава-1">Ссылка на главу 1</a>
            в названии главы пишем
            # Глава 1 <a name="Глава-1"></a>
        - Вариант, который не везде работает
            в оглавлении пишем
            [Ссылка на главу 1](#name-id)
            в названии главы пишем
            <a class="anchor" id="Название-главы"></a>
            ### Название главы
        - Упрощенный вариант, работает только в jupyter notebook
            в оглавлении пишем
            [Ссылка на главу 1](#Глава-1)
            в названии главы пишем
            # Глава 1
    """
    if mode not in ['draft', 'final']:
        raise ValueError(
            "Invalid mode. Mode must be either 'draft' or 'final'.")
    if link_type not in ['markdown', 'html']:
        raise ValueError(
            "Invalid link_type. link_type must be either 'markdown' or 'html'.")
    if mode == 'draft':
        notebook_path_splited = notebook_path.split('.')
        notebook_path_splited[-2] += '_temp'
        notebook_path = '.'.join(notebook_path_splited)

    def is_markdown(it): return "markdown" == it["cell_type"]
    def is_title(it): return it.strip().startswith(
        "#") and it.strip().lstrip("#").lstrip()
    toc = ['**Оглавление**<a name="ref-to-toc"></a>\n\n',]
    try:
        with open(notebook_path, 'r', encoding='utf-8') as in_f:
            nb_json = json.load(in_f)
    except FileNotFoundError:
        print(f"Error: File not found - {notebook_path}")
        return
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format - {notebook_path}")
        return
    for cell in filter(is_markdown, nb_json["cells"]):
        for line in filter(is_title, cell["source"]):
            level = line.count("#")
            if level < start_level or line.endswith('<skip>'):
                continue
            line = line.strip()
            indent = indent_char * (level * 2 - 3)
            title_for_ref = re.findall(
                r'<a name=[\"\']+(.*)[\"\']+></a>', line)
            if not title_for_ref:
                raise ValueError(
                    f'В строке "{line}" нет ссылки для создания оглавления')
            title_for_ref = title_for_ref[0]
            title = re.sub(r'<a.*</a>', '', line).lstrip("#").lstrip()
            if link_type == "markdown":
                toc_line = f"{indent}[{title}](#{title_for_ref})  \n"
            elif link_type == "html":
                toc_line = f"{indent}<a href='#{title_for_ref}'>{title}</a>  \n"
            else:
                raise ValueError(
                    "Неправильный тип ссылки. Должно быть 'markdown' или 'html'.")
            toc.append(toc_line)
    toc_cell = v4.new_markdown_cell([''.join(toc)])
    nb_json['cells'].insert(0, toc_cell)
    # display(nb_json)
    # Convert nb_json to nbformat object
    nb = nbformat.reads(json.dumps(nb_json), as_version=4)

    # Save the nbformat object to the file
    output_filename = notebook_path
    with open(output_filename, 'w', encoding='utf-8') as out_f:
        nbformat.write(nb, out_f, version=4)
    print(f"Table of content added to {output_filename}")


def make_headers_link_and_toc(notebook_path: str, mode: str = 'draft', start_level: int = 2, link_type_header: str = "name", indent_char: str = "&emsp;", link_type_toc: str = "html", is_make_headers_link: bool = True, is_make_toc: bool = True):
    ''' 
    Функция добавляет ссылки в название headers и создает содеражние

    Args:
        - notebook_path (str): Путь к ноутбуку.
        - mode (str): Режим работы функции. Либо 'draft', в этом случае создаться копия файла, либо 'final' в этом случае изменения будут сделаны в исходном файле. Default to 'draft'
        - link_type_header (str, optional): Тип ссылки в заголовке. Может быть "name" или "id". Defaults to "name".
        - start_level (int): Уровень глав (количество #), с которого начинать нумерацию
        - indent_char (str, optional): Символ для отступа в оглавлении. Defaults to "&emsp;".
        - link_type_toc (str, optional): Тип ссылки в оглавлении на заголовок. Может быть "markdown" или "html". Defaults to "html".
        - start_level (int): Уровень глав (количество #), с которого начинать нумерацию
        - is_make_headers_link (book): Делать ссылки в заголовках. Defaults to 'True'
        - is_make_toc (book): Делать оглавление. Defaults to 'True'
    Returns:
        None    
    Example:
        Для link_type="markdown":
            &emsp;[Глава-1](#Глава-1)<br>
            &emsp;&emsp;[Подглава-1.1](#Подглава-1.1)<br>
        Для link_type="html":
            &emsp;<a href="Глава-1">Глава 1</a><br>
            &emsp;&emsp;<a href="#Подглава-1.1">Подглава 1.1</a><br>

        - Вариант, который работает почти везде
            в оглавлении пишем
            <a href="#лава-1">Ссылка на главу 1</a>
            в названии главы пишем
            # Глава 1 <a name="Глава-1"></a>
        - Вариант, который не везде работает
            в оглавлении пишем
            [Ссылка на главу 1](#name-id)
            в названии главы пишем
            <a class="anchor" id="Название-главы"></a>
            ### Название главы
        - Упрощенный вариант, работает только в jupyter notebook
            в оглавлении пишем
            [Ссылка на главу 1](#Глава-1)
            в названии главы пишем
            # Глава 1
    '''
    if mode not in ['draft', 'final']:
        raise ValueError(
            "Invalid mode. Mode must be either 'draft' or 'final'.")
    if link_type_header not in ['name', 'id']:
        raise ValueError(
            "Invalid link_type. link_type must be either 'name' or 'id'.")
    if link_type_toc not in ['html', 'markdown']:
        raise ValueError(
            "Invalid link_type. link_type must be either 'html' or 'markdown'.")
    if is_make_headers_link:
        add_links_and_numbers_to_headings(
            notebook_path, mode=mode, link_type=link_type_header, start_level=start_level)
    if is_make_toc:
        generate_toc(notebook_path, mode=mode, link_type=link_type_toc,
                     start_level=start_level, indent_char=indent_char)


def add_conclusions_and_anomalies(notebook_path: str, mode: str = 'draft', link_type: str = 'html', order: dict = None):
    """
    This function adds conclusions and anomalies sections to a Jupyter notebook.

    Args:
        notebook_path (str): The path to the Jupyter notebook file.
        mode (str): The mode of the output file, either 'draft' or 'final'. Defaults to 'draft'.
        link_type (str): The type of link to use, either 'html' or 'markdown'. Defaults to 'html'.
        order (dict): dict of lists with ordered  conclusions and anomalie (key: conclusions and anomalies)

    Examples:
        order = dict(
                    conclusions =[ 'Женщины чаще возвращают кредит, чем мужчины.']
                    , anomalies = ['В датафрейме есть строки дубликаты. 54 строки. Меньше 1 % от всего датафрейма.  ']
                )
    """
    if mode not in ['draft', 'final']:
        raise ValueError(
            "Invalid mode. Mode must be either 'draft' or 'final'.")
    if link_type not in ['html', 'markdown']:
        raise ValueError(
            "Invalid link_type. link_type must be either 'html' or 'markdown'.")

    try:
        with open(notebook_path, 'r', encoding='utf-8') as in_f:
            nb_json = json.load(in_f)
    except FileNotFoundError:
        print(f"Error: File not found - {notebook_path}")
        return
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format - {notebook_path}")
        return
    regex_for_sub = re.compile(r'[^0-9a-zA-Zа-яА-Я]')
    conclusions = []
    anomalies = []

    for cell in nb_json["cells"]:
        cell_has_ref_to_toc = False
        source = cell["source"]
        new_source = []
        for line in source:
            if line.strip().startswith("_conclusion_") or line.strip().startswith("_anomalies_"):
                cell["cell_type"] = "markdown"
                if line.strip().startswith("_conclusion_"):
                    conclusion_or_anomaly = line.strip().replace("_conclusion_ ", '')
                if line.strip().startswith("_anomalies_"):
                    conclusion_or_anomaly = line.strip().replace("_anomalies_ ", '')
                conclusion_or_anomaly_for_ref = regex_for_sub.sub(
                    '-', conclusion_or_anomaly)
                if link_type == "html":
                    toc_conclusion_or_anomaly = f"- <a href='#{conclusion_or_anomaly_for_ref}'>{conclusion_or_anomaly}</a>  \n"
                elif link_type == "markdown":
                    toc_conclusion_or_anomaly = f"[- {conclusion_or_anomaly}](#{conclusion_or_anomaly_for_ref})  \n"
                else:
                    raise ValueError(
                        "Неправильный тип ссылки. Должно быть 'markdown' или 'html'.")
                if link_type == "html":
                    conclusion_or_anomaly_for_ref = f"<a name='{conclusion_or_anomaly_for_ref}'></a>"
                elif link_type == "markdown":
                    conclusion_or_anomaly_for_ref = f"<a class='anchor' id='{conclusion_or_anomaly_for_ref}'></a>"
                else:
                    raise ValueError(
                        "Неправильный тип ссылки. Должно быть 'name' или 'id'.")

                if line.strip().startswith("_conclusion_"):
                    conclusions.append(
                        (toc_conclusion_or_anomaly, conclusion_or_anomaly))
                    if not cell_has_ref_to_toc:
                        conclusion_or_anomaly_for_ref += '\n<a href="#ref-to-conclusions">вернуться к оглавлению</a>'
                if line.strip().startswith("_anomalies_"):
                    anomalies.append(
                        (toc_conclusion_or_anomaly, conclusion_or_anomaly))
                    if not cell_has_ref_to_toc:
                        conclusion_or_anomaly_for_ref += '\n<a href="#ref-to-anomalies">вернуться к оглавлению</a>'
                new_source.append(conclusion_or_anomaly_for_ref)
                cell_has_ref_to_toc = True
            else:
                new_source.append(line)
        cell["source"] = new_source
    # Отсортируем  выводы и аномалии как в переданном словаре
    if order:
        ordered_conclusions = order['conclusions']
        ordered_anomalies = order['anomalies']
        index_map = {x.strip(): i for i, x in enumerate(ordered_conclusions)}
        conclusions = sorted(
            conclusions, key=lambda x: index_map[x[1].strip()])
        index_map = {x.strip(): i for i, x in enumerate(ordered_anomalies)}
        anomalies = sorted(anomalies, key=lambda x: index_map[x[1].strip()])
    conclusions = [conclusion[0] for conclusion in conclusions]
    anomalies = [anomalie[0] for anomalie in anomalies]
    conclusions = [
        '**Главные выводы:**<a name="ref-to-conclusions"></a>\n'] + conclusions
    anomalies = [
        '**Аномалии и особенности в данных:**<a name="ref-to-anomalies"></a>\n'] + anomalies
    conclusions_cell = v4.new_markdown_cell([''.join(conclusions)])
    anomalies_cell = v4.new_markdown_cell([''.join(anomalies)])
    nb_json['cells'].insert(0, conclusions_cell)
    nb_json['cells'].insert(0, anomalies_cell)

    # Convert nb_json to nbformat object
    nb = nbformat.reads(json.dumps(nb_json), as_version=4)

    # Save the nbformat object to the file
    if mode == 'draft':
        output_filename_splited = notebook_path.split('.')
        output_filename_splited[-2] += '_temp'
        output_filename = '.'.join(output_filename_splited)
    else:
        output_filename = notebook_path
    with open(output_filename, 'w', encoding='utf-8') as out_f:
        nbformat.write(nb, out_f, version=4)
    print(f"Corrected notebook saved to {output_filename}")


def correct_text(text: str, speller=None) -> str:
    """
    Исправляет орфографические ошибки в тексте.

    Args:
        text (str): Текст, который нужно проверить.
        speller:  Объект, который будет использоваться для проверки орфографии.
            если None, то используется YandexSpeller()
    Returns:
        str: Исправленный текст.
    """
    if not speller:
        speller = YandexSpeller()
    # Используем регулярное выражение, чтобы найти все слова в тексте
    words = re.findall(r'\b\w+\b', text)

    # Создаем словарь, где ключами являются слова, а значениями - списки их позиций в тексте
    word_positions = {}
    for word in words:
        if word in word_positions:
            if word_positions[word][-1] + 1 < len(text):
                word_positions[word].append(
                    text.find(word, word_positions[word][-1] + 1))
            else:
                word_positions[word].append(text.find(word))
        else:
            word_positions[word] = [text.find(word)]
    # Удаляем все символы, кроме слов
    cleaned_text = re.sub(r'\W', '', text)
    # Исправляем ошибки в словах
    corrected_words = []
    max_attempts = 5  # Максимальное количество попыток
    attempts = 0
    text_has_errors = False
    while attempts < max_attempts:
        for word in words:
            # Исправляем ошибки в слове
            errors = speller.spell_text(word)
            if errors:
                text_has_errors = True
                text_for_highlight = list(text)
                highlight_word = f"<span style='color:yellow'>{word}</span>"
                # Если есть ошибки, берем первое предложенное исправление
                text_for_highlight = ''
                last_position = 0
                for start in word_positions[word]:
                    text_for_highlight += text[last_position:start] + \
                        highlight_word
                    last_position = start + len(word)
                text_for_highlight += text[last_position:]
                text_for_highlight = ''.join(text_for_highlight)
                display_html(text_for_highlight, raw=True)
                display_html(
                    f'Возможные верные варианты для слова {highlight_word}:', raw=True)
                print(errors[0]['s'])
                answer = input(
                    'Выберите индекс верного варианта (пустая строка - это индекс 0) или предложите свой вариант:\n')
                if answer.isdigit():
                    corrected_word = errors[0]['s'][int(answer)]
                elif answer == '':
                    corrected_word = errors[0]['s'][0]
                else:
                    corrected_word = answer

            else:
                corrected_word = word
            corrected_words.append(corrected_word)
        # print(words)
        # print(corrected_words)
        # Восстанавливаем слова на свои места
        if not text_has_errors:
            return text
        last_position = 0
        corrected_text = ''
        for i, word in enumerate(words):
            corrected_word = corrected_words[i]
            position = word_positions[word][0]
            word_positions[word].pop(0)
            corrected_text += text[last_position:position] + corrected_word
            last_position = position + len(word)
        corrected_text += text[last_position:]
        print('Исправленный вариант:')
        print(corrected_text)
        answer = input('Если верно, введите любой символ, -1 для повтора\n')
        if answer != '-1':
            break
        attempts += 1
    else:
        print("Максимальное количество попыток превышено. Исправленный текст не сохранен.")
        return text
    return corrected_text  # Объединяем символы обратно в строку


def correct_notebook_text(notebook_path: str, save_mode: str = 'final', work_mode: str = 'logging') -> None:
    """
    Corrects orthographic errors in a Jupyter Notebook.

    Args:
        notebook_path (str): Path to the Jupyter Notebook file.
        save_mode (str, optional): Mode of saving. Can be 'draft' or 'final'. Defaults to 'final'.
        work_mode (str, optional): Mode of working. Can be 'interactive', 'logging' or 'auto'. Defaults to 'logging'
    Returns:
        None
    """
    if save_mode not in ['draft', 'final']:
        raise ValueError(
            "Invalid save_mode. save_mode must be either 'draft' or 'final'.")
    if work_mode not in ['interactive', 'logging', 'auto']:
        raise ValueError(
            "Invalid work_mode. work_mode must be one of 'interactive', 'logging', 'auto'")
    speller = YandexSpeller()
    # Используем регулярное выражение, чтобы найти все слова в тексте
    regex_for_find = re.compile(r'\b\w+\b')

    def correct_text(text: str) -> str:
        """
        Исправляет орфографические ошибки в тексте.

        Args:
            text (str): Текст, который нужно проверить.
        Returns:
            str: Исправленный текст.
        """
        # Проверим есть ли ошибки, если нет, то  вернем исходный текст
        # print('in correct_text')
        errors = speller.spell_text(text)
        if not errors:
            return text
        # Используем регулярное выражение, чтобы найти все слова в тексте
        words = regex_for_find.findall(text)

        # Создаем словарь, где ключами являются слова, а значениями - списки их позиций в тексте
        word_positions = {}
        for word in words:
            pattern = r'\b' + word + r'\b'
            indices = [m.start() for m in re.finditer(pattern, text)]
            word_positions[word] = indices
        # Исправляем ошибки в словах
        corrected_words = []
        max_attempts = 5  # Максимальное количество попыток
        attempts = 0
        while attempts < max_attempts:
            for word in words:
                # Исправляем ошибки в слове
                errors = speller.spell_text(word)
                if errors:
                    if work_mode in {'interactive', 'logging'}:
                        text_for_highlight = list(text)
                        highlight_word = f"<span style='color:yellow'>{word}</span>"
                        # Если есть ошибки, берем первое предложенное исправление
                        text_for_highlight = ''
                        last_position = 0
                        for start in word_positions[word]:
                            text_for_highlight += text[last_position:start] + \
                                highlight_word
                            last_position = start + len(word)
                        text_for_highlight += text[last_position:]
                        text_for_highlight = ''.join(text_for_highlight)
                        display_html(text_for_highlight, raw=True)
                        display_html(
                            f'Возможные верные варианты для слова {highlight_word}:', raw=True)
                        print(errors[0]['s'])
                        if work_mode == 'interactive':
                            answer = input(
                                'Выберите индекс верного варианта (пустая строка - это индекс 0) или предложите свой вариант:\n')
                            if answer.isdigit():
                                corrected_word = errors[0]['s'][int(answer)]
                            elif answer == '':
                                corrected_word = errors[0]['s'][0]
                            else:
                                corrected_word = answer
                        else:
                            corrected_word = errors[0]['s'][0]
                    else:
                        corrected_word = errors[0]['s'][0]
                else:
                    corrected_word = word
                corrected_words.append(corrected_word)
            # print(words)
            # print(corrected_words)
            # Восстанавливаем слова на свои места
            last_position = 0
            corrected_text = ''
            for i, word in enumerate(words):
                corrected_word = corrected_words[i]
                position = word_positions[word][0]
                word_positions[word].pop(0)
                corrected_text += text[last_position:position] + corrected_word
                last_position = position + len(word)
            corrected_text += text[last_position:]
            # Если нужно делать с подтверждением
            # print('Исправленный вариант:')
            # print(corrected_text)
            # answer = input('Если верно, введите любой символ, -1 для повтора\n')
            # if answer != '-1':
            #     break
            # attempts += 1
            break
        else:
            print(
                "Максимальное количество попыток превышено. Исправленный текст не сохранен.")
            return text
        return corrected_text  # Объединяем символы обратно в строку

    def read_notebook(notebook_path: str) -> dict:
        """
        Reads a Jupyter Notebook file and returns its contents as a JSON object.

        Args:
            notebook_path (str): Path to the Jupyter Notebook file.

        Returns:
            dict: Notebook contents as a JSON object.
        """
        try:
            with open(notebook_path, 'r', encoding='utf-8') as in_f:
                return json.load(in_f)
        except FileNotFoundError:
            print(f"Error: File not found - {notebook_path}")
            return None
        except json.JSONDecodeError:
            print(f"Error: Invalid JSON format - {notebook_path}")
            return None

    def correct_source(source: list) -> list:
        """
        Corrects orthographic errors in a markdown cell.

        Args:
            source (list): Markdown cell contents.

        Returns:
            source: Corrected markdown cell contents.
        """
        corrected_lines = []
        for line in source:
            corrected_line = correct_text(line)
            corrected_lines.append(corrected_line)
        return corrected_lines

    def save_notebook(nb: dict, output_filename: str) -> None:
        """
        Saves a Jupyter Notebook to a file.

        Args:
            nb (dict): Notebook contents as a JSON object.
            output_filename (str): Output file name.

        Returns:
            None
        """
        with open(output_filename, 'w', encoding='utf-8') as out_f:
            nbformat.write(nb, out_f, version=4)

    def is_markdown(it):
        return it["cell_type"] == "markdown"

    nb_json = read_notebook(notebook_path)
    if nb_json is None:
        return

    for cell in tqdm(filter(is_markdown, nb_json["cells"]), desc="Correcting cells", total=len(nb_json["cells"])):
        text = ' '.join(cell["source"])
        # print(text)
        if speller.spell_text(text):
            # print('before correct_source')
            cell["source"] = correct_source(cell["source"])

    if save_mode == 'draft':
        output_filename_splited = notebook_path.split('.')
        output_filename_splited[-2] += '_temp'
        output_filename = '.'.join(output_filename_splited)
    else:
        output_filename = notebook_path
    print('End')
    answer = input(
        'Был выбран режим работы "draft", результат сохраниться в файл "{output_filename}"\nЕсли хотите сохранить в исходный файл, то введите "final":\n')
    if answer == 'final':
        output_filename = notebook_path
    nb = nbformat.reads(json.dumps(nb_json), as_version=4)
    save_notebook(nb, output_filename)

    print(f"Corrected notebook saved to {output_filename}")


def add_hypotheses_links_and_toc(notebook_path: str, mode: str = 'draft', link_type: str = 'html'):
    """
    Добавляет список гипотез в начало ноутбука Jupyter и ссылки на гипотезы. 

    Args:
        notebook_path (str): путь к ноутбуку Jupyter в формате JSON.
        mode (str, optional): режим работы функции. Defaults to 'draft'.
        link_type (str, optional): тип ссылок, которые будут добавлены в ноутбук. Defaults to 'html'.

    Raises:
        ValueError: если mode или link_type имеют недопустимые значения.
    """
    def is_markdown(it):
        return "markdown" == it["cell_type"]

    if mode not in ['draft', 'final']:
        raise ValueError(
            "Invalid mode. Mode must be either 'draft' or 'final'.")
    if link_type not in ['html', 'markdown']:
        raise ValueError(
            "Invalid link_type. link_type must be either 'html' or 'markdown'.")
    try:
        with open(notebook_path, 'r', encoding='utf-8') as in_f:
            nb_json = json.load(in_f)
    except FileNotFoundError:
        print(f"Error: File not found - {notebook_path}")
        return
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format - {notebook_path}")
        return
    regex_for_sub = re.compile(r'[^0-9a-zA-Zа-яА-Я]')
    toc_hypotheses = []
    for cell in filter(is_markdown, nb_json["cells"]):
        source = cell["source"]
        new_source = []
        for line in source:
            if line.strip().startswith("_hypothesis_"):
                hypothesis = line.strip().replace("_hypothesis_ ", '')
                hypothesis_title = hypothesis.strip().strip('**')
                hypothesis_for_ref = regex_for_sub.sub(
                    '-', hypothesis_title)
                if link_type == "html":
                    toc_hypotheses.append(
                        f"- <a href='#{hypothesis_for_ref}'>{hypothesis_title}</a>  \n**Результат:**  \n")
                    hypothesis_for_ref = f"{hypothesis}<a name='{hypothesis_for_ref}'></a>"
                elif link_type == "markdown":
                    toc_hypotheses.append(
                        f"[- {hypothesis_title}](#{hypothesis_for_ref})  \n**Результат:**  \n")
                    hypothesis_for_ref = f"{hypothesis}<a class='anchor' id='{hypothesis_for_ref}'></a>"
                else:
                    raise ValueError(
                        "Неправильный тип ссылки. Должно быть 'markdown' или 'html'.")
                hypothesis_for_ref += '  \n<a href="#ref-to-toc-hypotheses">вернуться к оглавлению</a>'
                new_source.append(hypothesis_for_ref)
            else:
                new_source.append(line)
        cell["source"] = new_source
    toc_hypotheses = [
        '**Результаты проверки гипотез:**<a name="ref-to-toc-hypotheses"></a>\n\n',] + toc_hypotheses
    hypotheses_cell = v4.new_markdown_cell([''.join(toc_hypotheses)])
    nb_json['cells'].insert(0, hypotheses_cell)
    # Convert nb_json to nbformat object
    nb = nbformat.reads(json.dumps(nb_json), as_version=4)
    # Save the nbformat object to the file
    if mode == 'draft':
        output_filename_splited = notebook_path.split('.')
        output_filename_splited[-2] += '_temp'
        output_filename = '.'.join(output_filename_splited)
    else:
        output_filename = notebook_path
    with open(output_filename, 'w', encoding='utf-8') as out_f:
        nbformat.write(nb, out_f, version=4)
    print(f"Corrected notebook saved to {output_filename}")
