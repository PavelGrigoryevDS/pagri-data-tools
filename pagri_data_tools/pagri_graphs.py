# import importlib
# importlib.reload(pagri_data_tools)
import pandas as pd
import numpy as np
import plotly.express as px
from plotly.subplots import make_subplots
import itertools
import plotly.graph_objects as go
import plotly.io as pio
from dash import Dash, dcc, html
from dash.dependencies import Input, Output
from scipy.stats import gaussian_kde
import plotly.figure_factory as ff
from scipy.stats import t
 
pio.renderers.default = "notebook"
colorway_for_line = ['rgb(127, 60, 141)', 'rgb(17, 165, 121)', 'rgb(231, 63, 116)',
                     '#03A9F4', 'rgb(242, 183, 1)', '#8B9467', '#FFA07A', '#005A5B', '#66CCCC', '#B690C4', 'rgb(127, 60, 141)', 'rgb(17, 165, 121)', 'rgb(231, 63, 116)',
                     '#03A9F4', 'rgb(242, 183, 1)', '#8B9467', '#FFA07A', '#005A5B', '#66CCCC', '#B690C4']
colorway_for_bar = ['rgba(128, 60, 170, 0.9)', '#049CB3', "rgba(112, 155, 219, 0.9)", "rgba(99, 113, 156, 0.9)", '#5c6bc0', '#B690C4', 'rgba(17, 100, 120, 0.9)', 'rgba(194, 143, 113, 0.8)', '#B690C4', '#03A9F4', '#8B9467', '#a771f2', 'rgba(102, 204, 204, 0.9)', 'rgba(168, 70, 90, 0.9)', 'rgba(50, 152, 103, 0.8)', '#8F7A7A', 'rgba(156, 130, 217, 0.9)'
                    ]
colorway_for_stacked_histogram =['#2ecc71', '#e74c3c', '#3498db', '#f1c40f', '#9b59b6']
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
        # Для подписей и меток
        title_font=dict(size=16, color="rgba(0, 0, 0, 0.7)"),     
        font=dict(size=14, family="Segoe UI", color="rgba(0, 0, 0, 0.7)"),
        xaxis_title_font=dict(size=14, color="rgba(0, 0, 0, 0.7)"),
        yaxis_title_font=dict(size=14, color="rgba(0, 0, 0, 0.7)"),
        xaxis_tickfont=dict(size=14, color="rgba(0, 0, 0, 0.7)"),
        yaxis_tickfont=dict(size=14, color="rgba(0, 0, 0, 0.7)"),
        xaxis_linecolor="rgba(0, 0, 0, 0.4)",
        yaxis_linecolor="rgba(0, 0, 0, 0.4)", 
        xaxis_tickcolor="rgba(0, 0, 0, 0.4)",
        yaxis_tickcolor="rgba(0, 0, 0, 0.4)",  
        legend_title_font_color='rgba(0, 0, 0, 0.7)',
        legend_title_font_size = 14,
        legend_font_color='rgba(0, 0, 0, 0.7)',
        # xaxis_linewidth=2,
        # yaxis_linewidth=2
        margin=dict(l=50, r=50, b=50, t=70),
        hoverlabel=dict(bgcolor="white"),
        xaxis=dict(
            showgrid=True
            , gridwidth=1
            , gridcolor="rgba(0, 0, 0, 0.1)"
        ),
        yaxis=dict(
            showgrid=True
            , gridwidth=1
            , gridcolor="rgba(0, 0, 0, 0.07)"
        )
    )
    return fig


def heatmap_simple(
    df: pd.DataFrame,
    title: str = '',
    xtick_text: list = None,
    ytick_text: list = None,
    xaxis_title: str = None,
    yaxis_title: str = None,
    width: int = None,
    height: int = None,
    decimal_places: int = 1,
    font_size: int = 14,
    show_text: bool = True,
    is_show_in_pct: bool = False,
    do_pretty_value: bool = False
) -> go.Figure:
    """
    Creates a heatmap from a Pandas DataFrame using Plotly.

    Parameters
    ----------
    df : pandas.DataFrame
        The Pandas DataFrame to create the heatmap from
    title : str, optional
        The title of the heatmap. Default is ''
    xtick_text : array-like, optional
        The custom tick labels for the x-axis. Default is None
    ytick_text : array-like, optional
        The custom tick labels for the y-axis. Default is None
    xaxis_title : str, optional
        The title for the x-axis. Default is None
    yaxis_title : str, optional
        The title for the y-axis. Default is None
    width : int, optional
        The width of the heatmap. Default is None
    height : int, optional
        The height of the heatmap. Default is None
    decimal_places : int, optional
        The number of decimal places to display in the annotations. Default is 2
    show_text : bool, optional
        Whether to show text in the annotations. Default is True
    font_size : int, optional
        The font size for the text in the annotations. Default is 14
    is_show_in_pct : bool, optional
        Whether to show the values in percentage. Default is True

    Returns
    -------
    plotly.graph_objects.Figure
        A Plotly figure object representing the heatmap

    Notes
    -----
    - If `xtick_text` or `ytick_text` is provided, it must have the same length as
      the number of columns or rows in the DataFrame, respectively
    - The heatmap is created with a custom colorscale and hover labels
    - The function returns a Plotly figure object, which can be displayed using `fig.show()`
    """
    def format_number(num):
        magnitude = 0
        while abs(num) >= 1000:
            magnitude += 1
            num /= 1000
        return f'{num:.1f}{["", "k", "M", "B"][magnitude]}'
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
    vmax = df.max().max()
    vmin = df.min().min()
    val = vmax - vmin
    if val > 0:
        center_color_bar =  vmin + (vmax - vmin) * 0.7
    else:
        center_color_bar = vmin + (vmax - vmin) * 0.3
    if show_text:
        if not is_show_in_pct:
            annotations = [
                dict(
                    text= '' if np.isnan(df.values[row, col]) else format_number(df.values[row, col]) if do_pretty_value else f"{df.values[row, col]:.{decimal_places}f}" ,
                    x=col,
                    y=row,
                    showarrow=False,
                    font=dict(
                        family='Segoe UI',
                        color="rgba(0, 0, 0, 0.7)" if df.values[row, col] <
                        center_color_bar else "white",
                        size=font_size
                    )
                )
                for row, col in np.ndindex(df.values.shape)
            ]
        else:
            annotations = [
                dict(
                    text= '' if np.isnan(df.values[row, col]) else f"{df.values[row, col]:.{decimal_places}%}",
                    x=col,
                    y=row,
                    showarrow=False,
                    font=dict(
                        family='Segoe UI',
                        color="rgba(0, 0, 0, 0.7)" if df.values[row, col] <
                        center_color_bar else "white",
                        size=font_size
                    )
                )
                for row, col in np.ndindex(df.values.shape)
            ]            
        fig.update_layout(
            annotations=annotations
        )
    # Update layout
    fig.update_layout(
        title=title,
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
    if xaxis_title is not None:
        fig.update_layout(xaxis=dict(title=xaxis_title))

    if yaxis_title is not None:
        fig.update_layout(yaxis=dict(title=yaxis_title))

    # Update figure size if custom size is provided
    if width is not None:
        fig.update_layout(width=width)
    if height is not None:
        fig.update_layout(height=height)
    fig.update_layout(
        title_font=dict(size=16, color="rgba(0, 0, 0, 0.7)"),     
        font=dict(size=14, family="Segoe UI", color="rgba(0, 0, 0, 0.7)"),
        xaxis_title_font=dict(size=14, color="rgba(0, 0, 0, 0.7)"),
        yaxis_title_font=dict(size=14, color="rgba(0, 0, 0, 0.7)"),
        xaxis_tickfont=dict(size=14, color="rgba(0, 0, 0, 0.7)"),
        yaxis_tickfont=dict(size=14, color="rgba(0, 0, 0, 0.7)"),
        xaxis_linecolor="rgba(0, 0, 0, 0.4)",
        yaxis_linecolor="rgba(0, 0, 0, 0.4)", 
        xaxis_tickcolor="rgba(0, 0, 0, 0.4)",
        yaxis_tickcolor="rgba(0, 0, 0, 0.4)",  
        legend_title_font_color='rgba(0, 0, 0, 0.7)',
        legend_title_font_size = 14,
        legend_font_color='rgba(0, 0, 0, 0.7)',
        # yaxis_linewidth=2
        margin=dict(l=50, r=50, b=50, t=70),
        hoverlabel=dict(bgcolor="white")
    )
    return fig


def heatmap_corr(
    df: pd.DataFrame,
    title: str = None,
    titles_for_axis: dict = None,
    xtick_text: list = None, 
    ytick_text: list = None,
    xaxis_title: str = None,
    yaxis_title: str = None,
    width: int = None,
    height: int = None,
    decimal_places: int = 2,
    font_size: int = 14
) -> go.Figure:
    """
    Creates a correlation heatmap from a Pandas DataFrame using Plotly.

    Parameters
    ----------
    df : pd.DataFrame
        The Pandas DataFrame to create the heatmap from
    title : str, optional
        The title of the heatmap. Default is None
    titles_for_axis : dict, optional
        A dictionary containing titles for the axes. Default is None
    xtick_text : list, optional
        The custom tick labels for the x-axis. Default is None
    ytick_text : list, optional
        The custom tick labels for the y-axis. Default is None
    xaxis_label : str, optional
        The label for the x-axis. Default is None
    yaxis_label : str, optional
        The label for the y-axis. Default is None
    width : int, optional
        The width of the heatmap. Default is None
    height : int, optional
        The height of the heatmap. Default is None
    decimal_places : int, optional
        The number of decimal places to display in the annotations. Default is 2
    font_size : int, optional
        The font size for the text in the annotations. Default is 14

    Returns
    -------
    go.Figure
        A Plotly figure object representing the heatmap

    Notes
    -----
    - If `xtick_text` or `ytick_text` is provided, it must have the same length as 
      the number of columns or rows in the DataFrame, respectively
    - The heatmap is created with a custom colorscale and hover labels
    - The function returns a Plotly figure object, which can be displayed using `fig.show()`
    """
    if not title:
        title = 'Тепловая карта корреляционных связей между числовыми столбцами'
    num_columns = filter(
        lambda x: pd.api.types.is_numeric_dtype(df[x]), df.columns)
    df_corr = df[num_columns].corr()
    if titles_for_axis:
        df_corr.columns = [titles_for_axis[column]
                           for column in df_corr.columns]
        df_corr.index = [titles_for_axis[index] for index in df_corr.index]
    # Create figure
    fig = go.Figure(data=go.Heatmap(
        z=df_corr.values,
        x=df_corr.columns,
        y=df_corr.index,
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
    center_color_bar = (df_corr.max().max() + df_corr.min().min()) * 0.7
    annotations = [
        dict(
            text=f"{df_corr.values[row, col]:.{decimal_places}f}",
            x=col,
            y=row,
            showarrow=False,
            font=dict(
                color="black" if df_corr.values[row, col] <
                center_color_bar else "white",
                size=font_size
            )
        )
        for row, col in np.ndindex(df_corr.values.shape)
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
        if len(xtick_text) != len(df_corr.columns):
            raise ValueError(
                "xtick_text must have the same length as the number of columns in the DataFrame")
        fig.update_layout(xaxis=dict(tickvals=range(
            len(xtick_text)), ticktext=xtick_text))

    if ytick_text is not None:
        if len(ytick_text) != len(df_corr.index):
            raise ValueError(
                "ytick_text must have the same length as the number of rows in the DataFrame")
        fig.update_layout(yaxis=dict(tickvals=range(
            len(ytick_text)), ticktext=ytick_text))

    # Update axis labels if custom labels are provided
    if xaxis_title is not None:
        fig.update_layout(xaxis=dict(title=xaxis_title))

    if yaxis_title is not None:
        fig.update_layout(yaxis=dict(title=yaxis_title))

    # Update figure size if custom size is provided
    if width is not None:
        fig.update_layout(width=width)
    if height is not None:
        fig.update_layout(height=height)
    hovertemplate = 'ось X = %{x}<br>ось Y = %{y}<br>Коэффициент корреляции = %{z:.2f}<extra></extra>'
    fig.update_traces(hovertemplate=hovertemplate)
    fig.update_layout(
        title_font=dict(size=16, color="rgba(0, 0, 0, 0.7)"),     
        font=dict(size=14, family="Segoe UI", color="rgba(0, 0, 0, 0.7)"),
        xaxis_title_font=dict(size=14, color="rgba(0, 0, 0, 0.7)"),
        yaxis_title_font=dict(size=14, color="rgba(0, 0, 0, 0.7)"),
        xaxis_tickfont=dict(size=14, color="rgba(0, 0, 0, 0.7)"),
        yaxis_tickfont=dict(size=14, color="rgba(0, 0, 0, 0.7)"),
        xaxis_linecolor="rgba(0, 0, 0, 0.4)",
        yaxis_linecolor="rgba(0, 0, 0, 0.4)", 
        xaxis_tickcolor="rgba(0, 0, 0, 0.4)",
        yaxis_tickcolor="rgba(0, 0, 0, 0.4)",  
        legend_title_font_color='rgba(0, 0, 0, 0.7)',
        legend_title_font_size = 14,
        legend_font_color='rgba(0, 0, 0, 0.7)',
        # yaxis_linewidth=2
        margin=dict(l=50, r=50, b=50, t=70),
        hoverlabel=dict(bgcolor="white")
    )
    return fig

def heatmap_corr_gen(
    df: pd.DataFrame,
    part_size: int = 10,
    title: str = None,
    titles_for_axis: dict = None,
    xtick_text: list = None,
    ytick_text: list = None,
    xaxis_label: str = None,
    yaxis_label: str = None,
    width: int = None,
    height: int = None,
    decimal_places: int = 2,
    font_size: int = 14
) -> go.Figure:
    """
    Creates a heatmap from a Pandas DataFrame using Plotly.

    Parameters
    ----------
    df : pd.DataFrame
        The Pandas DataFrame to create the heatmap from
    part_size : int, optional
        Max rows in corr matrix. Default is 10
    title : str, optional
        The title of the heatmap. Default is None
    titles_for_axis : dict, optional
        A dictionary containing titles for the axes. Default is None
    xtick_text : list, optional
        The custom tick labels for the x-axis. Default is None
    ytick_text : list, optional
        The custom tick labels for the y-axis. Default is None
    xaxis_label : str, optional
        The label for the x-axis. Default is None
    yaxis_label : str, optional
        The label for the y-axis. Default is None
    width : int, optional
        The width of the heatmap. Default is None
    height : int, optional
        The height of the heatmap. Default is None
    decimal_places : int, optional
        The number of decimal places to display in the annotations. Default is 2
    font_size : int, optional
        The font size for the text in the annotations. Default is 14

    Returns
    -------
    go.Figure
        A Plotly figure object representing the heatmap

    Notes
    -----
    - If `xtick_text` or `ytick_text` is provided, it must have the same length as 
      the number of columns or rows in the DataFrame, respectively
    - The heatmap is created with a custom colorscale and hover labels
    - The function returns a Plotly figure object, which can be displayed using `fig.show()`
    """

    if not title:
        title = 'Тепловая карта корреляционных связей между числовыми столбцами'
    num_columns = filter(
        lambda x: pd.api.types.is_numeric_dtype(df[x]), df.columns)
    df_corr = df[num_columns].corr()
    
    # Разделение на группы по 10 на 10
    part_size = 10
    num_rows = df_corr.shape[0]
    num_groups = (num_rows + part_size - 1) // part_size  # Количество групп
    # print(num_groups)
    # # Создание списка для хранения матриц
    correlation_matrices = []

    for i in range(num_groups):
        for j in range(num_groups):
            start_index_row = i * part_size
            start_index_col = j * part_size
            end_index_row = min(start_index_row + part_size, num_rows)
            end_index_col = min(start_index_col + part_size, num_rows)
            group_matrix = df_corr.iloc[start_index_row:end_index_row, start_index_col:end_index_col]
            correlation_matrices.append(group_matrix)
    for correlation_matrice in correlation_matrices:
        if titles_for_axis:
            correlation_matrice.columns = [titles_for_axis[column]
                            for column in correlation_matrice.columns]
            correlation_matrice.index = [titles_for_axis[index] for index in correlation_matrice.index]
        # Create figure
        fig = go.Figure(data=go.Heatmap(
            z=correlation_matrice.values,
            x=correlation_matrice.columns,
            y=correlation_matrice.index,
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
        center_color_bar = (correlation_matrice.max().max() + correlation_matrice.min().min()) * 0.7
        annotations = [
            dict(
                text=f"{correlation_matrice.values[row, col]:.{decimal_places}f}",
                x=col,
                y=row,
                showarrow=False,
                font=dict(
                    color="black" if correlation_matrice.values[row, col] <
                    center_color_bar else "white",
                    size=font_size
                )
            )
            for row, col in np.ndindex(correlation_matrice.values.shape)
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
            if len(xtick_text) != len(correlation_matrice.columns):
                raise ValueError(
                    "xtick_text must have the same length as the number of columns in the DataFrame")
            fig.update_layout(xaxis=dict(tickvals=range(
                len(xtick_text)), ticktext=xtick_text))

        if ytick_text is not None:
            if len(ytick_text) != len(correlation_matrice.index):
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
        hovertemplate = 'ось X = %{x}<br>ось Y = %{y}<br>Коэффициент корреляции = %{z:.2f}<extra></extra>'
        fig.update_traces(hovertemplate=hovertemplate)
        fig.update_layout(
            title_font=dict(size=16, color="rgba(0, 0, 0, 0.7)"),     
            font=dict(size=14, family="Segoe UI", color="rgba(0, 0, 0, 0.7)"),
            xaxis_title_font=dict(size=14, color="rgba(0, 0, 0, 0.7)"),
            yaxis_title_font=dict(size=14, color="rgba(0, 0, 0, 0.7)"),
            xaxis_tickfont=dict(size=14, color="rgba(0, 0, 0, 0.7)"),
            yaxis_tickfont=dict(size=14, color="rgba(0, 0, 0, 0.7)"),
            xaxis_linecolor="rgba(0, 0, 0, 0.4)",
            yaxis_linecolor="rgba(0, 0, 0, 0.4)", 
            xaxis_tickcolor="rgba(0, 0, 0, 0.4)",
            yaxis_tickcolor="rgba(0, 0, 0, 0.4)",  
            legend_title_font_color='rgba(0, 0, 0, 0.7)',
            legend_title_font_size = 14,
            legend_font_color='rgba(0, 0, 0, 0.7)',
            # yaxis_linewidth=2
            margin=dict(l=50, r=50, b=50, t=70),
            hoverlabel=dict(bgcolor="white")
        )
        yield fig

def categorical_heatmap_matrix(
    df: pd.DataFrame,
    col1: str,
    col2: str,
    titles_for_axis: dict = None,
    width: int = None,
    height: int = None
) -> go.Figure:
    """
    Generate a heatmap matrix for all possible combinations of categorical variables in a dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing categorical variables
    col1 : str
        Name of the first categorical column
    col2 : str
        Name of the second categorical column
    titles_for_axis : dict, optional
        A dictionary containing titles for the axes. Default is None
    width : int, optional
        Width of the heatmap. Default is None
    height : int, optional
        Height of the heatmap. Default is None

    Returns
    -------
    go.Figure
        A Plotly figure object representing the heatmap matrix
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
    fig = heatmap_simple(heatmap_matrix, title=title)
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
    return fig

def categorical_heatmap_matrix_gen(
    df: pd.DataFrame,
    titles_for_axis: dict = None,
    width: int = None,
    height: int = None
) -> go.Figure:
    """
    Generate a heatmap matrix for all possible combinations of categorical variables in a dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing categorical variables
    titles_for_axis : dict, optional
        A dictionary containing titles for the axes. Default is None
    width : int, optional
        Width of the heatmap. Default is None
    height : int, optional
        Height of the heatmap. Default is None

    Returns
    -------
    None
        This function displays the heatmap matrix but does not return any value
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
        fig = heatmap_simple(heatmap_matrix, title=title)
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
    app = Dash(__name__)

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


def treemap(
    df: pd.DataFrame,
    columns: list,
    values: str = None
) -> go.Figure:
    """
    Creates an interactive treemap using Plotly.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with data for the treemap
    columns : list
        List of columns to use for the treemap
    values : str, optional
        Column for values, if None - values will be calculated as count. Default is None

    Returns
    -------
    go.Figure
        Interactive treemap figure
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
    app = Dash(__name__)

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
    app = Dash(__name__)

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


def sankey(df, columns, values_column=None, func='sum', mode='fig', titles_for_axis: dict = None):
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
                    current_columns, observed=True)[[values_column]].agg(value=(values_column, func)).reset_index()
            else:
                df_grouped = df_in[current_columns].groupby(
                    current_columns, observed=True).size().reset_index().rename(columns={0: 'value'})
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
        colors = ['rgba(128, 60, 170, 1)', 'rgba(4, 156, 179, 1)', 'rgba(112, 155, 221, 1)', 'rgba(99, 113, 156, 1)', 'rgba(92, 107, 192, 1)', 'rgba(182, 144, 196, 1)', 'rgba(17, 100, 120, 1)', 'rgba(194, 143, 113, 1)',
                  'rgba(182, 144, 196, 1)', 'rgba(3, 169, 244, 1)', 'rgba(139, 148, 103, 1)', 'rgba(167, 113, 242, 1)', 'rgba(102, 204, 204, 1)', 'rgba(168, 70, 90, 1)', 'rgba(50, 152, 103, 1)', 'rgba(143, 122, 122, 1)', 'rgba(156, 130, 217, 1)']
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
        # Количество уровней
        num_levels = len(columns)
        step = 1 / (num_levels-1)
        # Аннотации для названий уровней
        level_x = 0
        for column in columns:
            if titles_for_axis:
                column = titles_for_axis[column][0]
            fig.add_annotation(x=level_x, y=1.05, xref="paper", yref="paper",
                               text=column, showarrow=False, font=dict(size=16, family="Open Sans", color="rgba(0, 0, 0, 0.6)"), xanchor='center')
            level_x += step

        layout = dict(
            # title=f"Sankey Diagram for {', '.join(columns+[values_column])}" if values_column else
            # f"Sankey Diagram for {', '.join(columns)}",
            title='Санки диаграмма категорий',
            height=772,
            title_font_size=16,
            # margin=dict(l=50, r=50, t=90, b=50),
            title_font=dict(size=24, color="rgba(0, 0, 0, 0.5)",
                            family="Open Sans"),
            # Для подписей и меток
            # font=dict(size=14, family="Open Sans", color="rgba(0, 0, 0, 1)"),
        )

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
    app = Dash(__name__)
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
            
def _create_base_fig_for_bar_line_area(config: dict, graph_type: str = 'bar'):
    """
    Creates a figure for bar, line or area function using the Plotly Express library.
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
    if not config['agg_func'] or not isinstance(config['agg_func'], str):
        raise ValueError("agg_func must be a string")
    if 'barmode' in config and not isinstance(config['barmode'], str):
        raise ValueError("barmode must be a string")
    if 'agg_mode' in config and config['agg_mode'] != 'groupby' and 'resample_freq' not in config:
        raise ValueError("resample_freq must be define")
    # if 'agg_func' not in config:
    #     config['agg_func'] = None
    # if 'barmode' not in config:
    #     config['barmode'] = 'group'
    # if 'width' not in config:
    #     config['width'] = None
    # if 'height' not in config:
    #     config['height'] = None
    # if 'show_text' not in config:
    #     config['show_text'] = False            
    # if 'textsize' not in config:
    #     config['textsize'] = 14
    # if 'xaxis_show' not in config:
    #     config['xaxis_show'] = True
    # if 'yaxis_show' not in config:
    #     config['yaxis_show'] = True
    # if 'showgrid_x' not in config:
    #     config['showgrid_x'] = True
    # if 'showgrid_y' not in config:
    #     config['showgrid_y'] = True
    # if 'sort' not in config:
    #     config['sort'] = True        
    # if 'top_n_trim_axis' not in config:
    #     config['top_n_trim_axis'] = None
    # if 'top_n_trim_legend' not in config:
    #     config['top_n_trim_legend'] = None    
    # if 'sort_axis' not in config:
    #     config['sort_axis'] = True
    # if 'sort_legend' not in config:
    #     config['sort_legend'] = True   
    # if 'textposition' not in config:
    #     config['textposition'] = None   
    # if 'legend_position' not in config:
    #     config['legend_position'] = 'right'          
    # if 'decimal_places' not in config:                        
    #     config['decimal_places'] = 1
    # if 'show_group_size' not in config:
    #     config['show_group_size'] = False
    # if 'agg_mode' not in config:
    #     config['agg_mode'] = None
    # if 'title' not in config:
    #     config['title'] = None
    # if 'groupby_col' not in config:
    #     config['groupby_col'] = None

    if pd.api.types.is_numeric_dtype(config['df'][config['y']]) and 'orientation' in config and config['orientation'] == 'h':
        config['x'], config['y'] = config['y'], config['x']

    # if titles_for_axis:
    #     # if not (config['func'] is None) and config['func'] not in ['mean', 'median', 'sum', 'count', 'nunique']:
    #     #     raise ValueError("func must be in ['mean', 'median', 'sum', 'count', 'nunique']")
    #     # func_for_title = {'mean': ['Среднее', 'Средний', 'Средняя', 'Средние'], 'median': [
    #     #     'Медианное', 'Медианный', 'Медианная', 'Медианные'], 'sum': ['Суммарное', 'Суммарный', 'Суммарная', 'Суммарное']
    #     #     , 'count': ['Общее', 'Общее', 'Общее', 'Общие']}
    #     config['x_axis_label'] = titles_for_axis[config['x']]
    #     config['y_axis_label'] = titles_for_axis[config['y']]
    #     config['category_axis_label'] = titles_for_axis[config['category']
    #                                             ] if 'category' in config else None
    #     # func = config['func']
    #     # if pd.api.types.is_numeric_dtype(config['df'][config['y']]):
    #     #     numeric = titles_for_axis[config["y"]][1]
    #     #     cat = titles_for_axis[config["x"]][1]
    #     #     suffix_type = titles_for_axis[config["y"]][2]
    #     # else:
    #     #     numeric = titles_for_axis[config["x"]][1]
    #     #     cat = titles_for_axis[config["y"]][1]
    #     #     suffix_type = titles_for_axis[config["x"]][2]
    #     # if func == 'nunique':
    #     #     numeric_list = numeric.split()[1:]
    #     #     title = f'Количество уникальных {' '.join(numeric_list)}'
    #     #     title += f' в зависимости от {cat}'
    #     # elif func is None:
    #     #     title = f' {numeric.capitalize()} в зависимости от {cat}'
    #     # else:
    #     #     title = f'{func_for_title[func][suffix_type]}'
    #     #     title += f' {numeric} в зависимости от {cat}'
    #     # if 'category' in config and config['category']:
    #     #     title += f' и {titles_for_axis[config["category"]][1]}'
    # else:
    #     if 'x_axis_label' not in config:
    #         config['x_axis_label'] = None
    #     if 'y_axis_label' not in config:
    #         config['y_axis_label'] = None
    #     if 'category_axis_label' not in config:
    #         config['category_axis_label'] = None
    #     if 'title' not in config:
    #         config['title'] = None
    # if 'category' not in config:
    #     config['category'] = None
    #     config['category_axis_label'] = None
    if not isinstance(config['category'], str) and config['category'] is not None:
        raise ValueError("category must be a string")

    def human_readable_number(x, decimal_places):
        format_string = f"{{:.{decimal_places}f}}"
        
        if x >= 1e6 or x <= -1e6:
            return f"{format_string.format(x / 1e6)}M"
        elif x >= 1e3 or x <= -1e3:
            return f"{format_string.format(x / 1e3)}k"
        else:
            return format_string.format(x)

    def prepare_df(config: dict):
        df = config['df']
        color = [config['category']] if config['category'] else []
        if config['groupby_cols']:
            num_column =  set([config['x'], config['y']]) - set(config['groupby_cols'])
            if len(num_column) != 1:
                raise ValueError("Set([x,y]) - Set(groupby_cols) must have result with one element")
            num_column = num_column.pop()
            cat_columns = config['groupby_cols']
        else:
            if not (pd.api.types.is_numeric_dtype(df[config['x']]) or pd.api.types.is_numeric_dtype(df[config['y']])):
                raise ValueError("At least one of x or y must be numeric.")
            elif pd.api.types.is_numeric_dtype(df[config['y']]):
                cat_columns = [config['x']] + color
                num_column = config['y']
            else:
                cat_columns = [config['y']] + color
                num_column = config['x']
        if config['agg_func'] is None:
            func = 'first'
        else:
            func = config.get('agg_func', 'mean')  # default to 'mean' if not provided
        if config['y'] == num_column:
            ascending = False
        else:
            ascending = True
        func_df = (df[[*cat_columns, num_column]]
                   .groupby(cat_columns, observed=True)
                   .agg(num=(num_column, func), count=(num_column, 'count'))
                   .reset_index())
        if config['sort_axis']:
            func_df['temp'] = func_df.groupby(cat_columns[0], observed=True)[
                'num'].transform('sum')
            func_df = (func_df.sort_values(['temp', 'num'], ascending=ascending)
                    .drop('temp', axis=1)
                    )
        if not config['sort_legend']:
            if config['sort_axis']:
                func_df = (func_df.sort_values([cat_columns[0], cat_columns[1]], ascending=[False, True])
                        )            
        func_df['count'] = func_df['count'].apply(
            lambda x: f'= {x}' if x <= 1e3 else 'больше 1000')
        func_df['pretty_value'] = func_df['num'].apply(human_readable_number, args = [config['decimal_places']])
        func_df[cat_columns] = func_df[cat_columns].astype('str')
        return func_df.rename(columns={'num': num_column})
    xaxis_title = config['xaxis_title']
    yaxis_title = config['yaxis_title']
    category_axis_title = config['category_axis_title']
    if config['agg_mode'] == 'resample':
        if config['agg_func'] is None:
            func = 'first'
        else:
            func = config['agg_func']
        columns = [config['x'], config['y']]
        if config['category']:
            columns.append(config['category'])
            df_for_fig = config['df'][columns].set_index(config['x']).groupby(config['category'], observed=True ).resample(config['resample_freq'])[config['y']].agg(func).reset_index()
        else:
            df_for_fig = config['df'][columns].set_index(config['x']).resample(config['resample_freq']).agg(func).reset_index()
        # x = config['df'][config['x']].values
        # y = config['df'][config['y']].values
        custom_data = [df_for_fig[config['y']].apply(human_readable_number, args = [config['decimal_places']])]
        # if pd.api.types.is_numeric_dtype(config['df'][config['y']]):
        #     custom_data = [df_for_fig[config['y']].apply(human_readable_number, args = [config['decimal_places']])]
        # else:
        #     custom_data = [df_for_fig[config['x']].apply(human_readable_number, args = [config['decimal_places']])]
        if graph_type == 'bar':
            fig = px.bar(df_for_fig, x=config['x'], y=config['y'], color=config['category'],
                        barmode=config['barmode'], custom_data=custom_data)
        elif graph_type == 'line':
            fig = px.line(df_for_fig, x=config['x'], y=config['y'], color=config['category'], custom_data=custom_data)
        elif graph_type == 'area':
            fig = px.area(df_for_fig, x=config['x'], y=config['y'], color=config['category'], custom_data=custom_data)
    elif config['agg_mode'] == 'groupby':
        df_for_fig = prepare_df(config)
        if config['top_n_trim_axis']:
            df_for_fig = df_for_fig.iloc[:config['top_n_trim_axis']]
        # if config['top_n_trim_legend']:
        #     df_for_fig = pd.concat([df_for_fig['data'].iloc[:, :config['top_n_trim_legend']], df_for_fig['data'].iloc[:, :config['top_n_trim_legend']]], axis=1, keys=['data', 'customdata'])        
        # display(df_for_fig)
        x = df_for_fig[config['x']].values
        y = df_for_fig[config['y']].values
        color = df_for_fig[config['category']
                        ].values if config['category'] else None
        custom_data = [df_for_fig['count'], df_for_fig['pretty_value']]
        # display(df_for_fig)
        if 'show_text' in config and config['show_text']:
            if pd.api.types.is_numeric_dtype(df_for_fig[config['y']]):
                text = [human_readable_number(el, config['decimal_places']) for el in y]
            else:
                text = [human_readable_number(el, config['decimal_places']) for el in x]
        else:
            text = None
        # display(df_for_fig)
        # display(custom_data)
        if graph_type == 'bar':
            fig = px.bar(x=x, y=y, color=color,
                        barmode=config['barmode'], text=text, custom_data=custom_data)
        elif graph_type == 'line':
            fig = px.line(x=x, y=y, color=color,
                        text=text, custom_data=custom_data)   
        elif graph_type == 'area':
            fig = px.area(x=x, y=y, color=color,
                        text=text, custom_data=custom_data)               
        color = []
        for trace in fig.data:
            color.append(trace.marker.color)
        if graph_type == 'bar':
            fig.update_traces(textposition='auto')
        elif graph_type == 'line':
            fig.update_traces(textposition='top center')
        elif graph_type == 'area':
            fig.update_traces(textposition='top center')   
        if pd.api.types.is_numeric_dtype(df_for_fig[config['x']]):
            # Чтобы сортировка была по убыванию вернего значения, нужно отсортировать по последнего значению в x
            traces = list(fig.data)
            traces.sort(key=lambda x: x.x[-1])
            fig.data = traces
            color = color[::-1]
            for i, trace in enumerate(fig.data):
                trace.marker.color = color[i]
            fig.update_layout(legend={'traceorder': 'reversed'})
        if config['textposition']:
            fig.update_traces(textposition=config['textposition'])
    else:
        if pd.api.types.is_numeric_dtype(config['df'][config['y']]):
            if not config['sort_axis'] or pd.api.types.is_datetime64_any_dtype(config['df'][config['x']]):
                df = config['df']
            else:
                num_for_sort = config['y']
                ascending_for_sort = False
                df = config['df'].sort_values(num_for_sort, ascending=ascending_for_sort)
            custom_data = [df[config['y']].apply(human_readable_number, args = [config['decimal_places']])]
        else:
            if config['sort_axis']:
                num_for_sort = config['x']
                ascending_for_sort = True
                df = config['df'].sort_values(num_for_sort, ascending=ascending_for_sort)
            else:
                df = config['df']
            custom_data = [df[config['x']].apply(human_readable_number, args = [config['decimal_places']])]
        if graph_type == 'bar':
            fig = px.bar(df, x=config['x'], y=config['y'], color=config['category'],
                        barmode=config['barmode'], custom_data=custom_data)
        elif graph_type == 'line':
            fig = px.line(df, x=config['x'], y=config['y'], color=config['category'], custom_data=custom_data)
        elif graph_type == 'area':
            fig = px.area(df, x=config['x'], y=config['y'], color=config['category'], custom_data=custom_data)
    if config['legend_position'] == 'top':
        fig.update_layout(
            yaxis = dict(
                domain=[0, 0.9]
            )
            , legend = dict(
                title_text=category_axis_title
                , title_font_color='rgba(0, 0, 0, 0.7)'
                , font_color='rgba(0, 0, 0, 0.7)'
                , orientation="h"  # Горизонтальное расположение
                , yanchor="top"    # Привязка к верхней части
                , y=1.09         # Положение по вертикали (отрицательное значение переместит вниз)
                , xanchor="center" # Привязка к центру
                , x=0.5              # Центрирование по горизонтали
            )
        )
    elif config['legend_position'] == 'right':
        fig.update_layout(
                legend = dict(
                title_text=category_axis_title
                , title_font_color='rgba(0, 0, 0, 0.7)'
                , font_color='rgba(0, 0, 0, 0.7)'
                , orientation="v"  # Горизонтальное расположение
                # , yanchor="bottom"    # Привязка к верхней части
                , y=1         # Положение по вертикали (отрицательное значение переместит вниз)
                # , xanchor="center" # Привязка к центру
                # , x=0.5              # Центрирование по горизонтали
            )
        )
    else:
        raise ValueError("Invalid legend_position. Please choose 'top' or 'right'.")
    if xaxis_title:
        hovertemplate_x = f'{xaxis_title} = '
    else:
        hovertemplate_x = f'x = '
    if yaxis_title:
        hovertemplate_y = f'{yaxis_title} = '
    else:
        hovertemplate_y = f'y = '
    if category_axis_title:
        hovertemplate_color = f'<br>{category_axis_title} = '
    else:
        hovertemplate_color = f'color = '
    if config['agg_mode'] == 'groupby':
        if pd.api.types.is_numeric_dtype(df_for_fig[config['y']]):
            hovertemplate = hovertemplate_x + \
                '%{x}<br>' + hovertemplate_y + '%{customdata[1]}'
        else:
            hovertemplate = hovertemplate_x + \
                '%{customdata[1]}<br>' + hovertemplate_y + '%{y}'
    elif config['agg_mode'] == 'resample':
        hovertemplate = hovertemplate_x + \
                '%{x}<br>' + hovertemplate_y + '%{customdata[0]}'
    else:
        if pd.api.types.is_numeric_dtype(config['df'][config['y']]):
            hovertemplate = hovertemplate_x + \
                '%{x}<br>' + hovertemplate_y + '%{customdata[0]}'
        else:
            hovertemplate = hovertemplate_x + \
                '%{customdata[0]}<br>' + hovertemplate_y + '%{y}'
    if config['category']:
        hovertemplate += hovertemplate_color + '%{data.name}'
    if config['show_group_size']:
        hovertemplate += f'<br>Размер группы '
        hovertemplate += '%{customdata[0]}'
    # hovertemplate += f'<br>cnt_in_sum_pct = '
    # hovertemplate += '%{customdata[1]}'
    hovertemplate += '<extra></extra>'
    fig.update_traces(hovertemplate=hovertemplate, hoverlabel=dict(bgcolor="white"), textfont=dict(
        family='Segoe UI', size=config['textsize']  # Размер шрифта
        # color='black'  # Цвет текста
    ) # Положение текстовых меток (outside или inside))
    )        
    fig.update_layout(
        # , title={'text': f'<b>{title}</b>'}
        # , margin=dict(l=50, r=50, b=50, t=70)
        margin=dict(t=80),
        width=config['width'], height=config['height'],
        title={'text': config["title"]}, xaxis_title=xaxis_title, yaxis_title=yaxis_title, 
        title_font=dict(size=16, color="rgba(0, 0, 0, 0.7)"),     
        font=dict(size=14, family="Segoe UI", color="rgba(0, 0, 0, 0.7)"),
        xaxis_title_font=dict(size=14, color="rgba(0, 0, 0, 0.7)"),
        yaxis_title_font=dict(size=14, color="rgba(0, 0, 0, 0.7)"),
        xaxis_tickfont=dict(size=14, color="rgba(0, 0, 0, 0.7)"),
        yaxis_tickfont=dict(size=14, color="rgba(0, 0, 0, 0.7)"),
        xaxis_linecolor="rgba(0, 0, 0, 0.4)",
        yaxis_linecolor="rgba(0, 0, 0, 0.4)", 
        xaxis_tickcolor="rgba(0, 0, 0, 0.4)",
        yaxis_tickcolor="rgba(0, 0, 0, 0.4)",  
        legend_title_font_color='rgba(0, 0, 0, 0.7)',
        legend_title_font_size = 14,
        legend_font_color='rgba(0, 0, 0, 0.7)',
        hoverlabel=dict(bgcolor="white"), xaxis=dict(
            visible=config['xaxis_show'], showgrid=config['showgrid_x'], gridwidth=1, gridcolor="rgba(0, 0, 0, 0.1)"
        ), yaxis=dict(
            visible=config['yaxis_show'], showgrid=config['showgrid_y'], gridwidth=1, gridcolor="rgba(0, 0, 0, 0.07)"
        ),
        legend=dict(
            title_font_color="rgba(0, 0, 0, 0.5)", font_color="rgba(0, 0, 0, 0.5)"
        )
    )      
    return fig    
            
def bar(
    df: pd.DataFrame,
    x: str,
    y: str,
    category: str = None,
    agg_mode: str = None,
    agg_func: str = None,    
    groupby_cols: list = None,
    resample_freq: str = None,     
    barmode: str = 'group',
    width: int = None,
    height: int = None,    
    title: str = None,    
    xaxis_title: str = None,
    yaxis_title: str = None,        
    category_axis_title: str = None,    
    showgrid_x: bool = True,
    showgrid_y: bool = True,
    legend_position: str = 'top',    
    top_n_trim_axis: int = None,
    top_n_trim_legend: int = None,
    sort_axis: bool = True,
    sort_legend: bool = True,
    show_text: bool = False,
    textsize: int = 14,
    textposition: str = 'auto',
    xaxis_show: bool = True,
    yaxis_show: bool = True,
    decimal_places: int = 0,
    show_group_size: bool = False,
) -> go.Figure:
    """
    Creates a bar chart using the Plotly Express library.

    Parameters
    ----------
    df : pd.DataFrame
        A DataFrame containing data for creating the chart
    x : str
        The name of the column in the DataFrame to be used for creating the X-axis
    y : str
        The name of the column in the DataFrame to be used for creating the Y-axis
    xaxis_title : str, optional
        The title for the X-axis
    yaxis_title : str, optional
        The title for the Y-axis
    category : str, optional
        The name of the column in the DataFrame to be used for creating categories
    top_n_trim_axis : int, optional
        The number of top categories axis to include in the chart
    top_n_trim_legend : int, optional
        The number of top categories legend to include in the chart
    sort_axis : bool, optional
        Whether to sort the categories on the axis. Default is True
    sort_legend : bool, optional
        Whether to sort the categories in the legend. Default is True
    category_axis_title : str, optional
        The title for the categories
    title : str, optional
        The title of the chart
    agg_func : str, optional
        The function to be used for aggregating data. May be mean, median, sum, count, nunique. Default is 'mean'
    barmode : str, optional
        The mode for displaying bars. Default is 'group'
    width : int, optional
        The width of the chart
    height : int, optional
        The height of the chart
    show_text : bool, optional
        Whether to display text on the chart. Default is False
    textsize : int, optional
        Text size. Default is 14
    textposition : str, optional
        Text position. May be 'auto', 'inside', 'outside', 'none'. Default is 'auto'
    xaxis_show : bool, optional
        Whether to show the X-axis. Default is True
    yaxis_show : bool, optional
        Whether to show the Y-axis. Default is True
    showgrid_x : bool, optional
        Whether to show grid on X-axis. Default is True
    showgrid_y : bool, optional
        Whether to show grid on Y-axis. Default is True
    legend_position : str, optional
        Position of the legend ('top', 'right'). Default is 'top'
    decimal_places : int, optional
        The number of decimal places to display. Default is 2
    show_group_size : bool, optional
        Whether to show the group size. Default is False
    agg_mode : str, optional
        Aggregation mode. May be 'groupby', 'resample', None. Default is None
    groupby_cols : list, optional
        Columns for groupby
    resample_freq : str, optional
        Resample frequency for resample

    Returns
    -------
    go.Figure
        The created chart
    """
    config = {
        'df': df,
        'x': x,
        'y': y,
        'xaxis_title': xaxis_title,
        'yaxis_title': yaxis_title,
        'category': category,
        'top_n_trim_axis': top_n_trim_axis,
        'top_n_trim_legend': top_n_trim_legend,
        'sort_axis': sort_axis,
        'sort_legend': sort_legend,
        'category_axis_title': category_axis_title,
        'title': title,
        'agg_func': agg_func,
        'barmode': barmode,
        'width': width,
        'height': height,
        'show_text': show_text,
        'textsize': textsize,
        'textposition': textposition,
        'xaxis_show': xaxis_show,
        'yaxis_show': yaxis_show,
        'showgrid_x': showgrid_x,
        'showgrid_y': showgrid_y,
        'legend_position': legend_position,
        'decimal_places': decimal_places,
        'show_group_size': show_group_size,
        'agg_mode': agg_mode,
        'groupby_cols': groupby_cols,
        'resample_freq': resample_freq
    }
    return _create_base_fig_for_bar_line_area(config, 'bar')

def line(
    df: pd.DataFrame,
    x: str,
    y: str,
    category: str = None,
    agg_mode: str = None,
    agg_func: str = None,    
    groupby_cols: list = None,
    resample_freq: str = None,     
    barmode: str = 'group',
    width: int = None,
    height: int = None,    
    title: str = None,    
    xaxis_title: str = None,
    yaxis_title: str = None,        
    category_axis_title: str = None,    
    showgrid_x: bool = True,
    showgrid_y: bool = True,
    legend_position: str = 'top',    
    top_n_trim_axis: int = None,
    top_n_trim_legend: int = None,
    sort_axis: bool = True,
    sort_legend: bool = True,
    show_text: bool = False,
    textsize: int = 14,
    textposition: str = 'auto',
    xaxis_show: bool = True,
    yaxis_show: bool = True,
    decimal_places: int = 0,
    show_group_size: bool = False,
) -> go.Figure:
    """
    Creates a bar chart using the Plotly Express library.

    Parameters
    ----------
    df : pd.DataFrame
        A DataFrame containing data for creating the chart
    x : str
        The name of the column in the DataFrame to be used for creating the X-axis
    y : str
        The name of the column in the DataFrame to be used for creating the Y-axis
    xaxis_title : str, optional
        The title for the X-axis
    yaxis_title : str, optional
        The title for the Y-axis
    category : str, optional
        The name of the column in the DataFrame to be used for creating categories
    top_n_trim_axis : int, optional
        The number of top categories axis to include in the chart
    top_n_trim_legend : int, optional
        The number of top categories legend to include in the chart
    sort_axis : bool, optional
        Whether to sort the categories on the axis. Default is True
    sort_legend : bool, optional
        Whether to sort the categories in the legend. Default is True
    category_axis_title : str, optional
        The title for the categories
    title : str, optional
        The title of the chart
    agg_func : str, optional
        The function to be used for aggregating data. May be mean, median, sum, count, nunique. Default is 'mean'
    barmode : str, optional
        The mode for displaying bars. Default is 'group'
    width : int, optional
        The width of the chart
    height : int, optional
        The height of the chart
    show_text : bool, optional
        Whether to display text on the chart. Default is False
    textsize : int, optional
        Text size. Default is 14
    textposition : str, optional
        Text position. May be 'auto', 'inside', 'outside', 'none'. Default is 'auto'
    xaxis_show : bool, optional
        Whether to show the X-axis. Default is True
    yaxis_show : bool, optional
        Whether to show the Y-axis. Default is True
    showgrid_x : bool, optional
        Whether to show grid on X-axis. Default is True
    showgrid_y : bool, optional
        Whether to show grid on Y-axis. Default is True
    legend_position : str, optional
        Position of the legend ('top', 'right'). Default is 'top'
    decimal_places : int, optional
        The number of decimal places to display. Default is 2
    show_group_size : bool, optional
        Whether to show the group size. Default is False
    agg_mode : str, optional
        Aggregation mode. May be 'groupby', 'resample', None. Default is None
    groupby_cols : list, optional
        Columns for groupby
    resample_freq : str, optional
        Resample frequency for resample

    Returns
    -------
    go.Figure
        The created chart
    """
    config = {
        'df': df,
        'x': x,
        'y': y,
        'xaxis_title': xaxis_title,
        'yaxis_title': yaxis_title,
        'category': category,
        'top_n_trim_axis': top_n_trim_axis,
        'top_n_trim_legend': top_n_trim_legend,
        'sort_axis': sort_axis,
        'sort_legend': sort_legend,
        'category_axis_title': category_axis_title,
        'title': title,
        'agg_func': agg_func,
        'barmode': barmode,
        'width': width,
        'height': height,
        'show_text': show_text,
        'textsize': textsize,
        'textposition': textposition,
        'xaxis_show': xaxis_show,
        'yaxis_show': yaxis_show,
        'showgrid_x': showgrid_x,
        'showgrid_y': showgrid_y,
        'legend_position': legend_position,
        'decimal_places': decimal_places,
        'show_group_size': show_group_size,
        'agg_mode': agg_mode,
        'groupby_cols': groupby_cols,
        'resample_freq': resample_freq
    }
    return _create_base_fig_for_bar_line_area(config, 'line')

def area(
    df: pd.DataFrame,
    x: str,
    y: str,
    category: str = None,
    agg_mode: str = None,
    agg_func: str = None,    
    groupby_cols: list = None,
    resample_freq: str = None,     
    barmode: str = 'group',
    width: int = None,
    height: int = None,    
    title: str = None,    
    xaxis_title: str = None,
    yaxis_title: str = None,        
    category_axis_title: str = None,    
    showgrid_x: bool = True,
    showgrid_y: bool = True,
    legend_position: str = 'top',    
    top_n_trim_axis: int = None,
    top_n_trim_legend: int = None,
    sort_axis: bool = True,
    sort_legend: bool = True,
    show_text: bool = False,
    textsize: int = 14,
    textposition: str = 'auto',
    xaxis_show: bool = True,
    yaxis_show: bool = True,
    decimal_places: int = 0,
    show_group_size: bool = False,
) -> go.Figure:
    """
    Creates a bar chart using the Plotly Express library.

    Parameters
    ----------
    df : pd.DataFrame
        A DataFrame containing data for creating the chart
    x : str
        The name of the column in the DataFrame to be used for creating the X-axis
    y : str
        The name of the column in the DataFrame to be used for creating the Y-axis
    xaxis_title : str, optional
        The title for the X-axis
    yaxis_title : str, optional
        The title for the Y-axis
    category : str, optional
        The name of the column in the DataFrame to be used for creating categories
    top_n_trim_axis : int, optional
        The number of top categories axis to include in the chart
    top_n_trim_legend : int, optional
        The number of top categories legend to include in the chart
    sort_axis : bool, optional
        Whether to sort the categories on the axis. Default is True
    sort_legend : bool, optional
        Whether to sort the categories in the legend. Default is True
    category_axis_title : str, optional
        The title for the categories
    title : str, optional
        The title of the chart
    agg_func : str, optional
        The function to be used for aggregating data. May be mean, median, sum, count, nunique. Default is 'mean'
    barmode : str, optional
        The mode for displaying bars. Default is 'group'
    width : int, optional
        The width of the chart
    height : int, optional
        The height of the chart
    show_text : bool, optional
        Whether to display text on the chart. Default is False
    textsize : int, optional
        Text size. Default is 14
    textposition : str, optional
        Text position. May be 'auto', 'inside', 'outside', 'none'. Default is 'auto'
    xaxis_show : bool, optional
        Whether to show the X-axis. Default is True
    yaxis_show : bool, optional
        Whether to show the Y-axis. Default is True
    showgrid_x : bool, optional
        Whether to show grid on X-axis. Default is True
    showgrid_y : bool, optional
        Whether to show grid on Y-axis. Default is True
    legend_position : str, optional
        Position of the legend ('top', 'right'). Default is 'top'
    decimal_places : int, optional
        The number of decimal places to display. Default is 2
    show_group_size : bool, optional
        Whether to show the group size. Default is False
    agg_mode : str, optional
        Aggregation mode. May be 'groupby', 'resample', None. Default is None
    groupby_cols : list, optional
        Columns for groupby
    resample_freq : str, optional
        Resample frequency for resample

    Returns
    -------
    go.Figure
        The created chart
    """
    config = {
        'df': df,
        'x': x,
        'y': y,
        'xaxis_title': xaxis_title,
        'yaxis_title': yaxis_title,
        'category': category,
        'top_n_trim_axis': top_n_trim_axis,
        'top_n_trim_legend': top_n_trim_legend,
        'sort_axis': sort_axis,
        'sort_legend': sort_legend,
        'category_axis_title': category_axis_title,
        'title': title,
        'agg_func': agg_func,
        'barmode': barmode,
        'width': width,
        'height': height,
        'show_text': show_text,
        'textsize': textsize,
        'textposition': textposition,
        'xaxis_show': xaxis_show,
        'yaxis_show': yaxis_show,
        'showgrid_x': showgrid_x,
        'showgrid_y': showgrid_y,
        'legend_position': legend_position,
        'decimal_places': decimal_places,
        'show_group_size': show_group_size,
        'agg_mode': agg_mode,
        'groupby_cols': groupby_cols,
        'resample_freq': resample_freq
    }
    return _create_base_fig_for_bar_line_area(config, 'area')

# def pairplot_seaborn(df: pd.DataFrame, titles_for_axis: dict = None):
#     """
#     Create a pairplot of a given DataFrame with customized appearance.

#     Parameters:
#     df (pd.DataFrame): The input DataFrame containing numerical variables.
#     titles_for_axis (dict):  A dictionary containing titles for the axes.

#     Returns:
#     None

#     Example:
#     titles_for_axis = dict(
#         # numeric column
#         children = 'Кол-во детей'
#         , age = 'Возраст'
#         , total_income = 'Доход'    
#     )
#     """
#     def human_readable_number(x):
#         if x >= 1e6 or x <= -1e6:
#             return f"{x/1e6:.1f}M"
#         elif x >= 1e3 or x <= -1e3:
#             return f"{x/1e3:.1f}k"
#         else:
#             return f"{x:.1f}"
#     g = sns.pairplot(df, markers=["o"],
#                      plot_kws={'color': (128/255, 60/255, 170/255, 0.9)},
#                      diag_kws={'color': (128/255, 60/255, 170/255, 0.9)})
#     for ax in g.axes.flatten():
#         xlabel = ax.get_xlabel()
#         ylabel = ax.get_ylabel()
#         if titles_for_axis:
#             if xlabel:
#                 xlabel = titles_for_axis[xlabel]
#             if ylabel:
#                 ylabel = titles_for_axis[ylabel]
#         ax.set_xlabel(xlabel, alpha=0.6, fontfamily='serif')
#         ax.set_ylabel(ylabel, alpha=0.6, fontfamily='serif')
#         xticklabels = ax.get_xticklabels()
#         for label in xticklabels:
#             # if label.get_text():
#             #     label.set_text(human_readable_number(int(label.get_text().replace('−', '-'))))  # modify the label text
#             label.set_alpha(0.6)
#         yticklabels = ax.get_yticklabels()
#         for label in yticklabels:
#             # if label.get_text():
#             #     label.set_text(human_readable_number(int(label.get_text().replace('−', '-'))))  # modify the label text
#             label.set_alpha(0.6)
#         ax.spines['top'].set_alpha(0.3)
#         ax.spines['left'].set_alpha(0.3)
#         ax.spines['right'].set_alpha(0.3)
#         ax.spines['bottom'].set_alpha(0.3)
#     g.fig.suptitle('Зависимости между числовыми переменными', fontsize=15,
#                    x=0.07, y=1.05, fontfamily='serif', alpha=0.7, ha='left')

def histogram(
    column: pd.Series,
    title: str = None,
    xaxis_title: str = None,
    yaxis_title: str = None,
    nbins: int = 30,
    width: int = 600,
    height: int = 400,
    left_quantile: float = 0,
    right_quantile: float = 1,
    show_marginal_box: bool = True,
) -> go.Figure:
    """
    Creates an interactive histogram using Plotly.

    Parameters
    ----------
    column : pd.Series
        The data series to plot
    title : str, optional
        Title of the histogram
    xaxis_title : str, optional
        Label for x-axis
    yaxis_title : str, optional
        Label for y-axis
    nbins : int, optional
        Number of bins in histogram. Default is 30
    width : int, optional
        Width of the plot in pixels. Default is 600
    height : int, optional
        Height of the plot in pixels. Default is 400
    left_quantile : float, optional
        Left boundary for data trimming (0-1). Default is 0
    right_quantile : float, optional
        Right boundary for data trimming (0-1). Default is 1
    show_marginal_box : bool, optional
        Whether to include a box plot. Defaults to True.

    Returns
    -------
    go.Figure
        Interactive Plotly histogram figure
    """
    trimmed_column = column.between(column.quantile(
        left_quantile), column.quantile(right_quantile))
    column = column[trimmed_column]
    if not title:
        title = f'Распределенеие {column.name}'
    if not xaxis_title:
        xaxis_title = 'Значение'
    if not yaxis_title:
        yaxis_title = 'Доля от общего'
    
    if show_marginal_box:
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True)
    else:
        fig = go.Figure()

    # Add histogram to bottom subplot
    if show_marginal_box:
        fig.add_trace(
            go.Histogram(
                x=column,
                nbinsx=nbins,
                histnorm='percent',
                marker_color='rgba(128, 60, 170, 0.9)'
            ),
            row=1, col=1
        )

        # Add box plot to top subplot
        fig.add_trace(
            go.Box(
                x=column,
                marker_color='rgba(128, 60, 170, 0.9)',
            ),
            row=2, col=1
        )
    else:
        fig.add_trace(
            go.Histogram(
                x=column,
                nbinsx=nbins,
                histnorm='percent',
                marker_color='rgba(128, 60, 170, 0.9)'
            ),
        )
    fig.update_layout(
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title
    )
    fig.update_traces(showlegend=False)
    fig.update_traces(
        hovertemplate=f'{xaxis_title} = ' + '%{x}<br>' + f'{yaxis_title}' + ' = %{y:.2f}<extra></extra>'
        , selector=dict(type='histogram')
    )
    fig.update_traces(
        hovertemplate= f'{xaxis_title} = ' + '%{x:.2f}<br><extra></extra>'
        , selector=dict(type='box')
    )
    if show_marginal_box:
        fig.update_layout(
            yaxis2 = dict(
                domain=[0.95, 1]
                , visible = False
            )
            , xaxis2 = dict(
                visible=False
            )
            , yaxis = dict(
                domain=[0, 0.9]
            )
        )
    fig.update_layout(
        title=title,
        width=width, height=height,
        title_font=dict(size=16, color="rgba(0, 0, 0, 0.7)"),
        title_y=0.95,
        font=dict(size=14, family="Segoe UI", color="rgba(0, 0, 0, 0.7)"),
        xaxis_showticklabels=True,
        xaxis_title_font=dict(size=14, color="rgba(0, 0, 0, 0.7)"),
        yaxis_title_font=dict(size=14, color="rgba(0, 0, 0, 0.7)"),
        xaxis_tickfont=dict(size=14, color="rgba(0, 0, 0, 0.7)"),
        yaxis_tickfont=dict(size=14, color="rgba(0, 0, 0, 0.7)"),
        xaxis_linecolor="rgba(0, 0, 0, 0.4)",
        yaxis_linecolor="rgba(0, 0, 0, 0.4)",
        xaxis_tickcolor="rgba(0, 0, 0, 0.4)",
        yaxis_tickcolor="rgba(0, 0, 0, 0.4)",
        legend_title_font_color='rgba(0, 0, 0, 0.7)',
        legend_title_font_size = 14,
        legend_font_color='rgba(0, 0, 0, 0.7)',
        margin=dict(l=50, r=50, b=10, t=50),
        hoverlabel=dict(bgcolor="white")
        , xaxis=dict(
            showgrid=True, gridwidth=1, gridcolor="rgba(0, 0, 0, 0.1)"
        )
        , yaxis=dict(
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
            crosstab_for_figs_all['data'], barmode='group')
        for trace in bar_fig_all.data:
            trace.text = [f'{y:.0f}' if y > 0.5 else '' for y in trace.y]
        bar_traces_len_all = len(bar_fig_all.data)
        bar_fig_normolized_by_col = px.bar(
            crosstab_for_figs_normolized_by_col['data'], barmode='group')
        for trace in bar_fig_normolized_by_col.data:
            trace.text = [f'{y:.0f}' if y > 0.5 else '' for y in trace.y]
        bar_traces_len_normolized_by_col = len(bar_fig_normolized_by_col.data)
        bar_fig_normolized_by_row = px.bar(
            crosstab_for_figs_normolized_by_row['data'], barmode='group')
        for trace in bar_fig_normolized_by_row.data:
            trace.text = [f'{y:.0f}' if y > 0.5 else '' for y in trace.y]
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
            }, {'title.text': title_bar, 'xaxis.title': xaxis_title_for_figs_all, 'yaxis.showgrid': True, 'yaxis.title': yaxis_title_for_figs_all, 'legend.title.text': legend_title_all
                }]), dict(label="Heatmap", method="update", args=[{
                    "visible": [True, False, False]
                    + [False] * bar_traces_len_all
                    + [False] * bar_traces_len_normolized_by_col
                    + [False] * bar_traces_len_normolized_by_row, 'xaxis': {'visible': False}, 'yaxis': {'visible': False}, 'hovertemplate': f'{xaxis_title_heatmap}'+' = %{x}<br>'+f'{yaxis_title_heatmap}'+'= %{y}<br>Доля = %{z:.1f} %<br>Количество = %{customdata}<extra></extra>'
                }, {'title.text': title_heatmap, 'xaxis.title': xaxis_title_heatmap, 'yaxis.title': yaxis_title_heatmap, 'yaxis.showgrid': False
                    }]), dict(label=f"Сравнение ({xaxis_title_for_figs_normolized_by_row.lower()})", method="update", args=[{
                        "visible": [False, False, False]
                        + [False] *
                        bar_traces_len_all
                        + [True] * bar_traces_len_normolized_by_col
                        + [False] * bar_traces_len_normolized_by_row, 'xaxis': {'visible': False}, 'yaxis': {'visible': False}, 'hovertemplate': f'{xaxis_title_for_figs_normolized_by_col}'+' = %{x}<br>'+f'{legend_title_normolized_by_col}'+'= %{data.name}<br>Доля = %{y:.1f} %<br>Количество = %{customdata}<extra></extra>'
                    }, {'title.text': title_bar, 'xaxis.title': xaxis_title_for_figs_normolized_by_col, 'yaxis.showgrid': True, 'yaxis.title': yaxis_title_for_figs_normolized_by_col, 'legend.title.text': legend_title_normolized_by_col
                        }]), dict(label="Heatmap", method="update", args=[{
                            "visible": [False, True, False]
                            + [False] * bar_traces_len_all
                            + [False] *
                            bar_traces_len_normolized_by_col
                            + [False] * bar_traces_len_normolized_by_row, 'xaxis': {'visible': False}, 'yaxis': {'visible': False}, 'hovertemplate': f'{xaxis_title_heatmap}'+' = %{x}<br>'+f'{yaxis_title_heatmap}'+'= %{y}<br>Доля = %{z:.1f} %<br>Количество = %{customdata}<extra></extra>'
                        }, {'title.text': title_heatmap, 'xaxis.title': xaxis_title_heatmap, 'yaxis.title': yaxis_title_heatmap, 'yaxis.showgrid': False
                            }]), dict(label=f"Сравнение ({xaxis_title_for_figs_normolized_by_col.lower()})", method="update", args=[{
                                "visible": [False, False, False]
                                + [False] * bar_traces_len_all
                                + [False] * bar_traces_len_normolized_by_col
                                + [True] * bar_traces_len_normolized_by_row, 'xaxis': {'visible': False}, 'yaxis': {'visible': False}, 'hovertemplate': f'{xaxis_title_for_figs_normolized_by_row}'+' = %{x}<br>'+f'{legend_title_normolized_by_row}'+'= %{data.name}<br>Доля = %{y:.1f} %<br>Количество = %{customdata}<extra></extra>'
                            }, {'title.text': title_bar, 'xaxis.title': xaxis_title_for_figs_normolized_by_row, 'yaxis.showgrid': True, 'yaxis.title': yaxis_title_for_figs_normolized_by_row, 'legend.title.text': legend_title_normolized_by_row
                                }]), dict(label="Heatmap", method="update", args=[{
                                    "visible": [False, False, True]
                                    + [False] *
                                    bar_traces_len_all
                                    + [False] * bar_traces_len_normolized_by_col
                                    + [False] * bar_traces_len_normolized_by_row, 'xaxis': {'visible': False}, 'yaxis': {'visible': False}, 'hovertemplate': f'{yaxis_title_heatmap}'+' = %{x}<br>'+f'{xaxis_title_heatmap}'+'= %{y}<br>Доля = %{z:.1f} %<br>Количество = %{customdata}<extra></extra>'
                                }, {'title.text': title_heatmap, 'xaxis.title': yaxis_title_heatmap, 'yaxis.title': xaxis_title_heatmap, 'yaxis.showgrid': False
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
            
def pairplot(
    df: pd.DataFrame,
    title: str = None,
    width: int = 800,
    height: int = 800,
    horizontal_spacing: float = None,
    vertical_spacing: float = None,
    rows: int = None,
    cols: int = None,
    category: str = None,
    legend_position: str = 'top',
    titles_for_axis: dict = None
) -> go.Figure:
    """
    Create a pairplot of numerical variables in a dataframe using Plotly.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    title : str, optional
        Title for figure
    width : int, optional
        Width of the plot. Default is 800
    height : int, optional
        Height of the plot. Default is 800
    horizontal_spacing : float, optional
        Horizontal spacing between subplots
    vertical_spacing : float, optional
        Vertical spacing between subplots
    rows : int, optional
        Number of rows in the subplot grid
    cols : int, optional
        Number of columns in the subplot grid
    category : str, optional
        Category column for coloring the scatter plots
    legend_position : str, optional
        Position of the legend ('top' or 'right'). Default is 'top'
    titles_for_axis : dict, optional
        Dictionary of custom axis titles. For example dict(price='Price', quantity='Quantity'), where price and quantity are column names in the dataframe

    Returns
    -------
    go.Figure
        The resulting pairplot figure
    """

    if df.empty:
        raise ValueError("Input dataframe is empty")
    num_columns = list(
        filter(lambda x: pd.api.types.is_numeric_dtype(df[x]), df.columns))
    if len(num_columns) < 2:
        raise ValueError(
            "Input dataframe must contain at least 2 numerical columns")
    combinations = list(itertools.combinations(df[num_columns].columns, 2))
    len_comb = len(combinations)
    # Определить размер сетки
    if rows is None or cols is None:
        size = int(np.ceil(np.sqrt(len_comb)))
        rows = size
        cols = size
    else:
        if rows * cols < len_comb:
            raise ValueError(
                "The number of rows and columns is not enough to accommodate all combinations")
        rows = rows
        cols = cols
    fig = make_subplots(
        rows=rows, cols=cols, horizontal_spacing=horizontal_spacing, vertical_spacing=vertical_spacing)
    colorway_for_line = ['rgb(127, 60, 141)', 'rgb(17, 165, 121)', 'rgb(231, 63, 116)',
                        '#03A9F4', 'rgb(242, 183, 1)', '#8B9467', '#FFA07A', '#005A5B', '#66CCCC', '#B690C4', 'rgb(127, 60, 141)', 'rgb(17, 165, 121)', 'rgb(231, 63, 116)',
                        '#03A9F4', 'rgb(242, 183, 1)', '#8B9467', '#FFA07A', '#005A5B', '#66CCCC', '#B690C4']
    for i, (col1, col2) in enumerate(combinations):
        row, col = divmod(i, cols)
        if titles_for_axis:
            xaxes_title = titles_for_axis[col1]
            yaxes_title = titles_for_axis[col2]
        else:
            xaxes_title = col1
            yaxes_title = col2
        fig_scatter = px.scatter(df, x=col1, y=col2, color=category)
        fig_scatter.update_traces(marker=dict(
            line=dict(color='white', width=0.5)))
        fig_scatter.update_traces(
            hovertemplate=xaxes_title + ' = %{x}<br>' + yaxes_title + ' = %{y}')
        for trace, color in  zip(fig_scatter.data, colorway_for_line):    
            trace.marker.color = color
            fig.add_trace(trace, row=row+1, col=col+1)
        fig.update_xaxes(
            title_text=xaxes_title,
            title_font=dict(size=16, color="rgba(0, 0, 0, 0.5)"),
            tickfont=dict(size=14, color="rgba(0, 0, 0, 0.5)"),
            linecolor="rgba(0, 0, 0, 0.5)",
            row=row+1, col=col+1,
            showgrid=True,
            gridwidth=1,
            gridcolor="rgba(0, 0, 0, 0.1)"
        )
        fig.update_yaxes(
            title_text=yaxes_title,
            title_font=dict(size=16, color="rgba(0, 0, 0, 0.5)"),
            tickfont=dict(size=14, color="rgba(0, 0, 0, 0.5)"),
            linecolor="rgba(0, 0, 0, 0.5)",
            row=row+1, col=col+1,
            showgrid=True,
            gridwidth=1,
            gridcolor="rgba(0, 0, 0, 0.07)"
        )

    # # Update the layout
    fig.update_layout(
        height=height,
        width=width,
        margin=dict(l=50, r=50, t=90, b=50),
        title={'text': title if title else 'Зависимости между числовыми переменными'},
        # Для подписей и меток
        title_font=dict(size=16, color="rgba(0, 0, 0, 0.7)"),     
        font=dict(size=14, family="Segoe UI", color="rgba(0, 0, 0, 0.7)"),
        xaxis_title_font=dict(size=14, color="rgba(0, 0, 0, 0.7)"),
        yaxis_title_font=dict(size=14, color="rgba(0, 0, 0, 0.7)"),
        xaxis_tickfont=dict(size=14, color="rgba(0, 0, 0, 0.7)"),
        yaxis_tickfont=dict(size=14, color="rgba(0, 0, 0, 0.7)"),
        xaxis_linecolor="rgba(0, 0, 0, 0.4)",
        yaxis_linecolor="rgba(0, 0, 0, 0.4)", 
        xaxis_tickcolor="rgba(0, 0, 0, 0.4)",
        yaxis_tickcolor="rgba(0, 0, 0, 0.4)",  
        legend_title_font_color='rgba(0, 0, 0, 0.7)',
        legend_title_font_size = 14,
        legend_font_color='rgba(0, 0, 0, 0.7)',
        hoverlabel=dict(bgcolor="white"),
    )
    if category:
        if legend_position == 'top':
            fig.update_layout(
                legend = dict(
                    title_text=titles_for_axis[category]
                    , title_font_color='rgba(0, 0, 0, 0.7)'
                    , font_color='rgba(0, 0, 0, 0.7)'
                    , orientation="h"  # Горизонтальное расположение
                    , yanchor="top"    # Привязка к верхней части
                    , y=1.05         # Положение по вертикали (отрицательное значение переместит вниз)
                    , xanchor="center" # Привязка к центру
                    , x=0.1              # Центрирование по горизонтали
                )
            )
        elif legend_position == 'right':
            fig.update_layout(
                    legend = dict(
                    title_text=titles_for_axis[category]
                    , title_font_color='rgba(0, 0, 0, 0.7)'
                    , font_color='rgba(0, 0, 0, 0.7)'
                    , orientation="v"  # Горизонтальное расположение
                    # , yanchor="bottom"    # Привязка к верхней части
                    , y=0.8         # Положение по вертикали (отрицательное значение переместит вниз)
                    # , xanchor="center" # Привязка к центру
                    # , x=0.5              # Центрирование по горизонтали
                )
            )
        else:
            raise ValueError("Invalid legend_position. Please choose 'top' or 'right'.")
    return fig
            
def heatmap_categories(
    df: pd.DataFrame,
    x: str,
    y: str,
    xaxis_title: str = None,
    yaxis_title: str = None,
    top_n_trim_axis: int = None,
    top_n_trim_legend: int = None,
    title: str = None,
    barmode: str = 'group',
    normalized_mode: str = 'all',
    orientation: str = 'v',
    width: int = None,
    height: int = None,
    show_text: bool = False,
    textsize: int = 14,
    xaxis_show: bool = True,
    yaxis_show: bool = True,
    showgrid_x: bool = True,
    showgrid_y: bool = True,
) -> go.Figure:
    """
    Creates a heatmap chart for categorical columns using Plotly Express.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing data for creating the chart
    x : str
        Column name for x-axis
    y : str
        Column name for y-axis (categories/legend)
    xaxis_title : str, optional
        Label for x-axis
    yaxis_title : str, optional
        Label for y-axis
    top_n_trim_axis : int, optional
        Number of top categories to show on axis
    top_n_trim_legend : int, optional
        Number of top categories to show in legend
    title : str, optional
        Title of the chart
    barmode : str, optional
        Mode for displaying bars. Default is 'group'
    normalized_mode : str, optional
        Mode for normalizing bars. Default is 'all'
    orientation : str, optional
        Chart orientation ('v' or 'h'). Default is 'v'
    width : int, optional
        Width of the chart in pixels
    height : int, optional
        Height of the chart in pixels
    show_text : bool, optional
        Whether to display text on chart. Default is False
    textsize : int, optional
        Size of text. Default is 14
    xaxis_show : bool, optional
        Show x-axis. Default is True
    yaxis_show : bool, optional
        Show y-axis. Default is True
    showgrid_x : bool, optional
        Show grid on x-axis. Default is True
    showgrid_y : bool, optional
        Show grid on y-axis. Default is True

    Returns
    -------
    go.Figure
        The created heatmap chart
    """
 
    # if titles_for_axis:
    #     config['column_for_x_label'] = titles_for_axis[config['column_for_x']][0]
    #     config['column_for_y_label'] = titles_for_axis[config['column_for_y']][0]
    #     column_for_x_label_for_title = titles_for_axis[config['column_for_x']][1]
    #     column_for_y_label_for_title= titles_for_axis[config['column_for_y']][1]
    #     temp_title = f'Распределение долей для {column_for_x_label_for_title} и {column_for_y_label_for_title}'
    #     if config['normalized_mode'] == 'col':
    #         config['title'] = temp_title + f"<br>c нормализацией по {titles_for_axis[config['column_for_y']][2]}"
    #     elif config['normalized_mode'] == 'row':
    #         config['title'] = temp_title + f"<br>c нормализацией по {titles_for_axis[config['column_for_x']][2]}"
    #     else:
    #         config['title'] = temp_title
    # else:
    #     if 'column_for_x_label' not in config:
    #         config['column_for_x_label'] = None
    #     if 'column_for_y_label' not in config:
    #         config['column_for_y_label'] = None
    #     if 'title' not in config:
    #         config['title'] = None 
    def make_df_for_fig(crosstab_for_figs, mode):
        if mode == 'all':
            # normolized by all size df
            sum_for_normolized = crosstab_for_figs.sum().sum()
        if mode == 'col':
            # normolized by sum of coll
            sum_for_normolized = crosstab_for_figs.sum()
        if mode == 'row':
            # normolized by sum of row
            crosstab_for_figs = crosstab_for_figs.T
            sum_for_normolized = crosstab_for_figs.sum()           
        crosstab_for_figs_all = crosstab_for_figs * 100 / sum_for_normolized
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
        return crosstab_for_figs_all
    def fig_update_layout(fig):
        fig.update_layout(          
            title_font=dict(size=16, color="rgba(0, 0, 0, 0.7)"),     
            font=dict(size=14, family="Segoe UI", color="rgba(0, 0, 0, 0.7)"),
            xaxis_title_font=dict(size=14, color="rgba(0, 0, 0, 0.7)"),
            yaxis_title_font=dict(size=14, color="rgba(0, 0, 0, 0.7)"),
            xaxis_tickfont=dict(size=14, color="rgba(0, 0, 0, 0.7)"),
            yaxis_tickfont=dict(size=14, color="rgba(0, 0, 0, 0.7)"),
            xaxis_linecolor="rgba(0, 0, 0, 0.4)",
            yaxis_linecolor="rgba(0, 0, 0, 0.4)", 
            xaxis_tickcolor="rgba(0, 0, 0, 0.4)",
            yaxis_tickcolor="rgba(0, 0, 0, 0.4)",  
            legend_title_font_color='rgba(0, 0, 0, 0.7)',
            legend_title_font_size = 14,
            legend_font_color='rgba(0, 0, 0, 0.7)',
            margin=dict(l=50, r=50, b=50, t=70),       
        ) 
        return fig     
    column_for_x = x
    column_for_y = y
    column_for_x_label = xaxis_title
    column_for_y_label = yaxis_title
    crosstab = pd.crosstab(df[column_for_x], df[column_for_y])
    crosstab_for_figs= make_df_for_fig(crosstab, normalized_mode)
    if top_n_trim_axis:
        crosstab_for_figs = crosstab_for_figs.iloc[:top_n_trim_axis]
    if top_n_trim_legend:
        crosstab_for_figs = pd.concat([crosstab_for_figs['data'].iloc[:, :top_n_trim_legend], crosstab_for_figs['data'].iloc[:, :top_n_trim_legend]], axis=1, keys=['data', 'customdata'])        
    # data_for_fig = crosstab_for_figs['data']    
    # customdata = crosstab_for_figs['customdata'].values.T
    if orientation == 'v':
        data_for_fig = crosstab_for_figs['data'][::-1]
        customdata = crosstab_for_figs['customdata'][::-1].values
        xaxis_title = column_for_y_label
        yaxis_title = column_for_x_label  
        if normalized_mode == 'row':
            xaxis_title, yaxis_title = yaxis_title, xaxis_title
        hovertemplate=f'{column_for_y_label}'+' = %{x}<br>'+f'{column_for_x_label}'+' = %{y}<br>Доля = %{z:.1f} %<br>Количество = %{customdata}<extra></extra>'    
    else:
        data_for_fig = crosstab_for_figs['data'].T.iloc[::-1]
        customdata = crosstab_for_figs['customdata'].values.T[::-1]            
        xaxis_title = column_for_x_label
        yaxis_title = column_for_y_label
        if normalized_mode == 'row':
            xaxis_title, yaxis_title = yaxis_title, xaxis_title
        hovertemplate=f'{column_for_x_label}'+' = %{x}<br>'+f'{column_for_y_label}'+' = %{y}<br>Доля = %{z:.1f} %<br>Количество = %{customdata}<extra></extra>'

    fig = heatmap_simple(data_for_fig, font_size=14, title=title)
    fig.update_traces(hovertemplate=hovertemplate, textfont=dict(
        family='Segoe UI', size=14, color="rgba(0, 0, 0, 0.7)"), hoverlabel=dict(bgcolor="white", font=dict(color='rgba(0, 0, 0, 0.7)', size=14)))
    for trace in fig.data:
        trace.xgap = 3
        trace.ygap = 3
        trace.customdata = customdata
    # center_color_bar = (crosstab_for_figs.max().max() + crosstab_for_figs.min().min()) * 0.7
    # for annot in fig.layout.annotations:
    #     annot.font.color = "#d4d4d4"
    fig =  fig_update_layout(fig)
    fig.update_layout(coloraxis=dict(colorscale=[
                            (0, 'rgba(204, 153, 255, 0.1)'), (1, 'rgb(127, 60, 141)')]))        
    fig.update_layout(xaxis_title=xaxis_title, yaxis_title=yaxis_title)      
    return fig

def bar_categories(
    df: pd.DataFrame,
    column_for_axis: str,
    column_for_legend: str = None,
    axis_title: str = None,
    legend_title: str = None,
    top_n_trim_axis: int = None,
    top_n_trim_legend: int = None,
    title: str = None,
    barmode: str = 'group',
    normalized_mode: str = 'all',
    orientation: str = 'v',
    width: int = None,
    height: int = None,
    show_text: bool = False,
    textsize: int = 14,
    textposition: str = 'auto',
    text_decimal_places: int = 0,
    xaxis_show: bool = True,
    yaxis_show: bool = True,
    showgrid_x: bool = True,
    showgrid_y: bool = True,
    legend_position: str = 'top',
    sort_axis: bool = True,
    sort_legend: bool = True,
    swap_axis_legend: bool = True,
) -> go.Figure:
    """
    Creates a bar chart for categorical columns using Plotly Express.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the data
    column_for_axis : str
        Column name for creating the axis
    column_for_legend : str, optional
        Column name for creating categories/legend
    axis_title : str, optional
        Title for the axis
    legend_title : str, optional
        Title for the categories
    top_n_trim_axis : int, optional
        Number of top categories to show on axis
    top_n_trim_legend : int, optional
        Number of top categories to show in legend
    title : str, optional
        Title of the chart
    barmode : str, optional
        Mode for displaying bars ('group', 'stack'). Default is 'group'
    normalized_mode : str, optional
        Mode for normalizing bars ('all', 'col', 'row'). Default is 'all'
    orientation : str, optional
        Chart orientation ('v' or 'h'). Default is 'v'
    width : int, optional
        Width of the chart in pixels
    height : int, optional
        Height of the chart in pixels
    show_text : bool, optional
        Whether to display text on bars. Default is False
    textsize : int, optional
        Size of text. Default is 14
    textposition : str, optional
        Position of text ('auto', 'inside', 'outside', 'none'). Default is 'auto'
    text_decimal_places : int, optional
        Number of decimal places in text. Default is 0
    xaxis_show : bool, optional
        Show x-axis. Default is True
    yaxis_show : bool, optional
        Show y-axis. Default is True
    showgrid_x : bool, optional
        Show grid on x-axis. Default is True
    showgrid_y : bool, optional
        Show grid on y-axis. Default is True
    legend_position : str, optional
        Position of legend ('top' or 'right'). Default is 'top'
    sort_axis : bool, optional
        Sort categories on axis. Default is True
    sort_legend : bool, optional
        Sort categories in legend. Default is True
    swap_axis_legend : bool, optional
        Swap axis and legend. Default is True

    Returns
    -------
    go.Figure
        The created bar chart

    """
                                           
    # if 'orientation' in config and config['orientation'] == 'h':
    #     config['x'], config['y'] = config['y'], config['x']

    # if titles_for_axis:
    #     if not config['title']:
    #         config['column_for_axis_label'] = titles_for_axis[config['column_for_axis']][0]
    #         column_for_axis_label_for_title = titles_for_axis[config['column_for_axis']][1]
    #         if config['column_for_legend']:
    #             config['column_for_legend_label'] = titles_for_axis[config['column_for_legend']][0]
    #             column_for_legend_label_for_title = titles_for_axis[config['column_for_legend']][1]
    #             temp_title = f'Распределение долей для {column_for_axis_label_for_title} и {column_for_legend_label_for_title}'
    #         else:
    #             config['column_for_legend_label'] = None
    #             temp_title = f'Распределение долей для {column_for_axis_label_for_title}'
    #         if config['normalized_mode'] == 'col':
    #             config['title'] = temp_title + f" c нормализацией"
    #         elif config['normalized_mode'] == 'row':
    #             config['title'] = temp_title + f" c нормализацией"
    #         else:
    #             config['title'] = temp_title
    #     else:
    #         config['column_for_axis_label'] = titles_for_axis[config['column_for_axis']]
    #         if config['column_for_legend']:
    #             config['column_for_legend_label'] = titles_for_axis[config['column_for_legend']]
    #         else:
    #             config['column_for_legend_label'] = None
    # else:
    #     if 'column_for_axis_label' not in config:
    #         config['column_for_axis_label'] = None
    #     if 'column_for_legend_label' not in config:
    #         config['column_for_legend_label'] = None
    #     if 'title' not in config:
    #         config['title'] = None 
            
    def make_df_for_fig(crosstab_for_figs, mode):
        if mode == 'all':
            # normolized by all size df
            sum_for_normolized = crosstab_for_figs.sum().sum()
        if mode == 'col':
            # normolized by sum of coll
            sum_for_normolized = crosstab_for_figs.sum()
        if mode == 'row':
            # normolized by sum of row
            crosstab_for_figs = crosstab_for_figs.T
            sum_for_normolized = crosstab_for_figs.sum()           
        crosstab_for_figs_all = crosstab_for_figs * 100 / sum_for_normolized
        if normalized_mode in ['row', 'col']:
            crosstab_for_figs_all = crosstab_for_figs_all.T
            crosstab_for_figs = crosstab_for_figs.T
        crosstab_for_figs_all = pd.concat(
            [crosstab_for_figs_all, crosstab_for_figs], axis=1, keys=['data', 'customdata'])      
        crosstab_for_figs_all['sum_row'] = crosstab_for_figs_all['data'].max(axis=1)
        # display(crosstab_for_figs_all)        
        # display(crosstab_for_figs_all)  
        if sort_axis :
            max_sum_index = 0        
            crosstab_for_figs_all = crosstab_for_figs_all.sort_values(
                'sum_row', ascending=False).drop('sum_row', axis=1, level=0)
        else:
                max_sum_index = crosstab_for_figs_all['sum_row'].idxmax()
                max_sum_index = crosstab_for_figs_all.index.get_loc(max_sum_index)
                crosstab_for_figs_all = crosstab_for_figs_all.drop('sum_row', axis=1, level=0)        
        if sort_legend :
            crosstab_for_figs_all = pd.concat(
                [crosstab_for_figs_all['data'], crosstab_for_figs_all['customdata']], axis=0, keys=['data', 'customdata'])
            crosstab_for_figs_all = crosstab_for_figs_all.sort_values(
                crosstab_for_figs_all.index[max_sum_index], axis=1, ascending=False)
            crosstab_for_figs_all = pd.concat(
                [crosstab_for_figs_all.loc['data'], crosstab_for_figs_all.loc['customdata']], axis=1, keys=['data', 'customdata'])
        return crosstab_for_figs_all
    
    def fig_update_layout(fig):
        fig.update_layout(          
            title_font=dict(size=16, color="rgba(0, 0, 0, 0.7)"),     
            font=dict(size=14, family="Segoe UI", color="rgba(0, 0, 0, 0.7)"),
            xaxis_title_font=dict(size=14, color="rgba(0, 0, 0, 0.7)"),
            yaxis_title_font=dict(size=14, color="rgba(0, 0, 0, 0.7)"),
            xaxis_tickfont=dict(size=14, color="rgba(0, 0, 0, 0.7)"),
            yaxis_tickfont=dict(size=14, color="rgba(0, 0, 0, 0.7)"),
            xaxis_linecolor="rgba(0, 0, 0, 0.4)",
            yaxis_linecolor="rgba(0, 0, 0, 0.4)", 
            xaxis_tickcolor="rgba(0, 0, 0, 0.4)",
            yaxis_tickcolor="rgba(0, 0, 0, 0.4)",  
            legend_title_font_color='rgba(0, 0, 0, 0.7)',
            legend_title_font_size = 14,
            legend_font_color='rgba(0, 0, 0, 0.7)',
            xaxis=dict(
                visible=xaxis_show, showgrid=showgrid_x, gridwidth=1, gridcolor="rgba(0, 0, 0, 0.1)"
            ), yaxis=dict(
                visible=yaxis_show, showgrid=showgrid_y, gridwidth=1, gridcolor="rgba(0, 0, 0, 0.07)"
            ),            
            margin=dict(l=50, r=50, b=50, t=70),     
        ) 
        return fig     
  
    column_for_axis_label = axis_title
    column_for_legend_label = legend_title
    if column_for_legend:
        crosstab = pd.crosstab(df[column_for_axis], df[column_for_legend])
    else:
        crosstab = df.groupby(column_for_axis, observed=True).size().to_frame('share') 
    crosstab_for_figs= make_df_for_fig(crosstab, normalized_mode)
    if top_n_trim_axis:
        crosstab_for_figs = crosstab_for_figs.iloc[:top_n_trim_axis]
    if top_n_trim_legend:
        crosstab_for_figs = pd.concat([crosstab_for_figs['data'].iloc[:, :top_n_trim_legend], crosstab_for_figs['data'].iloc[:, :top_n_trim_legend]], axis=1, keys=['data', 'customdata'])    
    data_for_fig = crosstab_for_figs['data']    
    customdata = crosstab_for_figs['customdata'].values.T    
    if orientation == 'h':
        data_for_fig = data_for_fig.iloc[::-1,::-1]
        customdata = customdata[::-1,::-1]
        xaxis_title = 'Доля'
        yaxis_title = column_for_axis_label
        legend_title_text = column_for_legend_label    
        if column_for_legend and normalized_mode in ['col']:
            legend_title_text, yaxis_title = yaxis_title, legend_title_text    
    else:
        xaxis_title = column_for_axis_label
        yaxis_title = 'Доля'
        legend_title_text = column_for_legend_label
        if column_for_legend and normalized_mode in ['col']:
            xaxis_title, legend_title_text = legend_title_text, xaxis_title
    if column_for_legend and normalized_mode in ['col']:
        column_for_axis_label, column_for_legend_label = column_for_legend_label, column_for_axis_label
        # if orientation == 'h':
            
    fig = px.bar(
        data_for_fig, barmode=barmode, orientation=orientation, title=title)
    fig.update_layout(xaxis_title=xaxis_title, yaxis_title=yaxis_title)
    trace_colors = [trace.marker.color for trace in reversed(fig.data)]            
    for i, trace in enumerate(fig.data):
        if orientation == 'h':
            # trace.x, trace.y = trace.y, trace.x
            trace.marker.color = trace_colors[i]
            trace.textangle=0
            if show_text:
                trace.text = [f'{x:.{text_decimal_places}f}' if x >= 0.1 else '<0.1' for x in trace.x]
            if column_for_legend:
                hovertemplate=f'{column_for_axis_label}'+' = %{y}<br>'+f'{column_for_legend_label}' +\
                                    ' = %{data.name}<br>Доля = %{x:.1f} %<br>Количество = %{customdata}<extra></extra>'                    
            else:
                hovertemplate=f'{column_for_axis_label}'+' = %{y}<br>Доля = %{x:.1f} %<br>Количество = %{customdata}<extra></extra>'                    
        else:
            if show_text:
                trace.text = [f'{y:.{text_decimal_places}f}' if y >= 0.1 else '<0.1' for y in trace.y]

            if column_for_legend:
                hovertemplate=f'{column_for_axis_label}'+' = %{x}<br>'+f'{column_for_legend_label}' +\
                                    ' = %{data.name}<br>Доля = %{y:.1f} %<br>Количество = %{customdata}<extra></extra>'       
            else:
                hovertemplate=f'{column_for_axis_label}'+' = %{x}<br>Доля = %{y:.1f} %<br>Количество = %{customdata}<extra></extra>'   
        trace.customdata = customdata[i]
    fig.update_traces(hovertemplate=hovertemplate, hoverlabel=dict(bgcolor="white", font=dict(color='rgba(0, 0, 0, 0.7)', size=14))
                      , textfont=dict(family='Segoe UI', size=textsize))   
    if textposition:
        fig.update_traces(textposition=textposition)
    if not column_for_legend:
        fig.update_layout(showlegend=False)
    if orientation == 'h':
        fig.update_layout(legend_traceorder='reversed')        
    fig.update_layout(
        height=height,
        width=width,
    )        
    if legend_position == 'top':
        fig.update_layout(
            legend = dict(
                title_text=legend_title_text
                , title_font_color='rgba(0, 0, 0, 0.7)'
                , font_color='rgba(0, 0, 0, 0.7)'
                , orientation="h"  # Горизонтальное расположение
                , yanchor="top"    # Привязка к верхней части
                , y=1.09         # Положение по вертикали (отрицательное значение переместит вниз)
                , xanchor="center" # Привязка к центру
                , x=0.5              # Центрирование по горизонтали
            )     
        )    
    elif legend_position == 'right':
        fig.update_layout(
                legend = dict(
                title_text=legend_title_text
                , title_font_color='rgba(0, 0, 0, 0.7)'
                , font_color='rgba(0, 0, 0, 0.7)'
                , orientation="v"  # Горизонтальное расположение
                # , yanchor="bottom"    # Привязка к верхней части
                , y=0.8         # Положение по вертикали (отрицательное значение переместит вниз)
                # , xanchor="center" # Привязка к центру
                # , x=0.5              # Центрирование по горизонтали
            )
        )
    else:
        raise ValueError("Invalid legend_position. Please choose 'top' or 'right'.")        
            
    return fig_update_layout(fig)

def heatmap(
    df: pd.DataFrame,
    x: str,
    y: str,
    z: str,
    xaxis_title: str = None,
    yaxis_title: str = None,
    zaxis_title: str = None,
    title: str = None,
    width: int = None,
    height: int = None,
    agg_func: str = None,
    show_text: bool = True,
    textsize: int = 14,
    xaxis_show: bool = True,
    yaxis_show: bool = True,
    showgrid_x: bool = False,
    showgrid_y: bool = False,
    top_n_trim_axis: int = None,
    top_n_trim_legend: int = None,
    top_n_trim_from_axis: str = 'end',
    top_n_trim_from_legend: str = 'end',
    sort_x: bool = True,
    sort_y: bool = True,
    decimal_places: int = 2,
    is_reversed_y: bool = False,
    x_axis_position: str = 'bottom',
    skip_first_col_for_cohort: bool = False,
    is_show_in_pct: bool = False,
    do_pretty_value: bool = False,
    orientation: str = 'v',
    titles_for_axis: dict = None
) -> go.Figure:
    """
    Creates a heatmap chart using Plotly Express.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the data
    x : str
        Column name for x-axis
    y : str
        Column name for y-axis
    z : str
        Column name for heatmap values
    xaxis_title : str, optional
        Title for x-axis
    yaxis_title : str, optional
        Title for y-axis
    title : str, optional
        Title of the chart
    width : int, optional
        Width of the chart in pixels
    height : int, optional
        Height of the chart in pixels
    agg_func : str, optional
        Aggregation function to apply to z-values.
    show_text : bool, optional
        Show text on heatmap. Default is True
    textsize : int, optional
        Size of text. Default is 14
    xaxis_show : bool, optional
        Show x-axis. Default is True
    yaxis_show : bool, optional
        Show y-axis. Default is True
    showgrid_x : bool, optional
        Show grid on x-axis. Default is False
    showgrid_y : bool, optional
        Show grid on y-axis. Default is False
    top_n_trim_axis : int, optional
        Number of top categories to show on axis
    top_n_trim_legend : int, optional
        Number of top categories to show in legend
    top_n_trim_from_axis : str, optional
        Direction to trim categories in axis ('start' or 'end'). Default is 'end'
    top_n_trim_from_legend : str, optional
        Direction to trim categories in legend ('start' or 'end'). Default is 'end'        
    sort_x : bool, optional
        Sort x-axis categories. Default is True
    sort_y : bool, optional
        Sort y-axis categories. Default is True
    decimal_places : int, optional
        Number of decimal places. Default is 2
    is_reversed_y : bool, optional
        Reverse y-axis order. Default is False
    x_axis_position : str, optional
        Position of x-axis ('top' or 'bottom'). Default is 'bottom'
    skip_first_col_for_cohort : bool, optional
        Skip first column for cohort analysis. Default is False
    is_show_in_pct : bool, optional
        Show values as percentages. Default is False
    do_pretty_value : bool, optional
        Format large numbers (e.g., 1000 as 1k). Default is False
    orientation : str, optional
        Chart orientation ('v' or 'h'). Default is 'v'

    Returns
    -------
    go.Figure
        The created heatmap chart
    """

    if not agg_func:
        raise ValueError("func must be a string")
    if not zaxis_title:
        raise ValueError('zaxis_title must for hover')
    # if titles_for_axis:
    #     # func_for_title = {'mean': ['Среднее', 'Средний', 'Средняя', 'Средние'], 'median': [
    #     #     'Медианное', 'Медианный', 'Медианная', 'Медианные'], 'sum': ['Суммарное', 'Суммарный', 'Суммарная', 'Суммарные']}
    #     # config['column_for_x_label'] = titles_for_axis[config['column_for_x']][0]
    #     # config['column_for_y_label'] = titles_for_axis[config['column_for_y']][0]
    #     # config['column_for_value_label'] = titles_for_axis[config['column_for_value']][0]
    #     # func = config['func']
    #     # numeric = titles_for_axis[config["column_for_value"]][1]
    #     # cat = titles_for_axis[config["column_for_x"]][1]
    #     # suffix_type = titles_for_axis[config["column_for_value"]][2]
    #     # title = f'{func_for_title[func][suffix_type]}'
    #     # title += f' {numeric} в зависимости от {cat}'
    #     # title += f' и {titles_for_axis[config["column_for_y"]][1]}'
    #     # config['title'] = title
    #     config['column_for_x_label'] = titles_for_axis[config['column_for_x']]
    #     config['column_for_y_label'] = titles_for_axis[config['column_for_y']]
    #     config['column_for_value_label'] = titles_for_axis[config['column_for_value']]
    # else:
    #     if 'column_for_x_label' not in config:
    #         config['column_for_x_label'] = None
    #     if 'column_for_y_label' not in config:
    #         config['column_for_y_label'] = None
    #     if 'column_for_value_label' not in config:
    #         config['column_for_value_label'] = None            
    #     if 'title' not in config:
    #         config['title'] = None 
            
    def make_df_for_fig(df, column_for_x, column_for_y, column_for_value, finc, orientation):
        func_df =  pd.pivot_table(df, index=column_for_x, columns=column_for_y, values=column_for_value, aggfunc=func, observed=True)
        if orientation == 'h':
            func_df = func_df.T
        ascending_sum = True
        ascending_index = False
        na_position = 'last'
        sort_position = -1
        if sort_x:
            func_df['sum'] = func_df.sum(axis=1, numeric_only=True)
            func_df = func_df.sort_values(
                'sum', ascending=ascending_sum).drop('sum', axis=1)
        if sort_y:
            func_df = func_df.sort_values(func_df.index[sort_position], axis=1, ascending=ascending_index, na_position=na_position)
        if is_reversed_y:
            func_df = func_df.iloc[::-1]
        if skip_first_col_for_cohort:
            func_df = func_df.iloc[:, 1:]
            if is_reversed_y:
                func_df = func_df.iloc[1:]
            else:
                func_df = func_df.iloc[:-1]
        return func_df
    
    def fig_update_layout(fig):
        fig.update_layout(          
            title_font=dict(size=16, color="rgba(0, 0, 0, 0.7)"),     
            font=dict(size=14, family="Segoe UI", color="rgba(0, 0, 0, 0.7)"),
            
            xaxis_title_font=dict(size=14, color="rgba(0, 0, 0, 0.7)"),
            yaxis_title_font=dict(size=14, color="rgba(0, 0, 0, 0.7)"),
            xaxis_tickfont=dict(size=14, color="rgba(0, 0, 0, 0.7)"),
            yaxis_tickfont=dict(size=14, color="rgba(0, 0, 0, 0.7)"),
            xaxis_linecolor="rgba(0, 0, 0, 0.4)",
            yaxis_linecolor="rgba(0, 0, 0, 0.4)", 
            xaxis_tickcolor="rgba(0, 0, 0, 0.4)",
            yaxis_tickcolor="rgba(0, 0, 0, 0.4)",  
            legend_title_font_color='rgba(0, 0, 0, 0.7)',
            legend_title_font_size = 14,
            legend_font_color='rgba(0, 0, 0, 0.7)',
            margin=dict(l=50, r=50, b=50, t=70),       
        ) 
        return fig     
    column_for_x = x
    column_for_y = y
    column_for_x_label = xaxis_title
    column_for_y_label = yaxis_title
    column_for_value_label = zaxis_title
    column_for_value = z
    orientation = orientation
    func = agg_func
    title = title
    df_for_fig= make_df_for_fig(df, column_for_x, column_for_y, column_for_value, func, orientation)
    if top_n_trim_axis:
        if top_n_trim_from_axis == 'start':
            x_nunique = df[column_for_x].nunique()
            df_for_fig = df_for_fig.iloc[x_nunique-top_n_trim_axis:]
        else:
            df_for_fig = df_for_fig.iloc[:top_n_trim_axis]
    if top_n_trim_legend:
        if top_n_trim_from_legend == 'start':
            y_nunique = df[column_for_y].nunique()
            df_for_fig = df_for_fig.iloc[:, y_nunique-top_n_trim_legend:]        
        else:
            df_for_fig = df_for_fig.iloc[:, :top_n_trim_legend]        
    fig = heatmap_simple(df_for_fig, font_size=textsize, decimal_places=decimal_places, show_text=show_text, is_show_in_pct=is_show_in_pct, do_pretty_value=do_pretty_value)
    if titles_for_axis:
        if orientation == 'h':
            hovertemplate = f'{column_for_x_label}'+' = %{x}<br>'+f'{column_for_y_label}'+' = %{y}<br>' +f'{column_for_value_label}' +' = %{z:.1f}<extra></extra>'
        else:
            hovertemplate = f'{column_for_y_label}'+' = %{x}<br>'+f'{column_for_x_label}'+' = %{y}<br>' +f'{column_for_value_label}' +' = %{z:.1f}<extra></extra>'
    else:
        hovertemplate = None
    fig.update_traces(hovertemplate=hovertemplate, textfont=dict(
        family='Segoe UI', size=14), hoverlabel=dict(bgcolor="white", font=dict(family='Segoe UI', color='rgba(0, 0, 0, 0.7)', size=14)))
    for trace in fig.data:
        trace.xgap = 3
        trace.ygap = 3
    # for annot in fig.layout.annotations:
    #     annot.font.color = "rgba(0, 0, 0, 0.7)"
    # fig.update_layout(coloraxis=dict(colorscale=[
    #                         (0, 'rgba(204, 153, 255, 0.1)'), (1, 'rgb(127, 60, 141)')]))     
    fig.update_layout(
        height=height
        , width=width
    )        
    if x_axis_position == 'top':
        fig.update_layout(
            xaxis=dict(
                side='top'  # Устанавливаем ось X сверху
            )
            # , yaxis_domain = [0, 0.9]
        )        
    if orientation == 'h':
        fig.update_layout(title=title, xaxis_title=column_for_x_label, yaxis_title=column_for_y_label)
    else:
        fig.update_layout(title=title, xaxis_title=column_for_y_label, yaxis_title=column_for_x_label)
    fig =  fig_update_layout(fig)        
    return fig

def pairplot_pairs(
    df: pd.DataFrame,
    pairs: dict[tuple, dict],
    coloring: bool = True,
    width: int = 850,
    height: int = 800,
    density_mode: str = 'count',
    bins: int = 20,
    rows: int = 3,
    cols: int = 3,
    horizontal_spacing: float = None,
    vertical_spacing: float = None,
    titles_for_axis: dict = None
) -> go.Figure:
    """
    Create a pairplot of numerical variables using Plotly with specified column pairs.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    pairs : dict[tuple, dict]
        Dictionary of column pairs to plot
        Key: tuple of column names
        Value: dictionary with column ranges for truncation
    coloring : bool, optional
        Color points based on density. Default is True
    width : int, optional
        Width of plot in pixels. Default is 850
    height : int, optional
        Height of plot in pixels. Default is 800
    density_mode : str, optional
        Mode for density calculation ('kde' or 'count'). Default is 'count'
    bins : int, optional
        Number of bins for density calculation. Default is 20
    rows : int, optional
        Number of subplot rows. Default is 3
    cols : int, optional
        Number of subplot columns. Default is 3
    horizontal_spacing : float, optional
        Horizontal spacing between subplots
    vertical_spacing : float, optional
        Vertical spacing between subplots
    titles_for_axis : dict, optional
        Custom axis titles

    Returns
    -------
    go.Figure
        Pairplot figure

    Examples
    --------
    >>> pairs = {
    ...     ('total_images', 'last_price'): None,
    ...     ('total_images', 'floors_total'): {
    ...         'total_images': [-2.15, 22.45],
    ...         'floors_total': [0.51, 28.54]
    ...     }
    ... }
    >>> pairplot_pairs(df, pairs, density_mode='kde', bins=30)
    """

    if df.empty:
        raise ValueError("Input dataframe is empty")
    num_columns = list(
        filter(lambda x: pd.api.types.is_numeric_dtype(df[x]), df.columns))
    if not pairs:
        raise ValueError(
            "pairs must contains at least one pair of column names")
    combinations = list(pairs.keys())
    len_comb = len(combinations)
    df_size = df.shape[0]
    # Определить размер сетки
    if rows is None or cols is None:
        size = int(np.ceil(np.sqrt(len_comb)))
        rows = size
        cols = size
    else:
        if rows * cols < len_comb:
            raise ValueError(
                "The number of rows and columns is not enough to accommodate all combinations")
        rows = rows
        cols = cols

    fig = make_subplots(
        rows=rows, cols=cols, horizontal_spacing=horizontal_spacing, vertical_spacing=vertical_spacing)

    for i, (col1, col2) in enumerate(combinations):
        row, col = divmod(i, cols)
        if titles_for_axis:
            xaxes_title = titles_for_axis[col1]
            yaxes_title = titles_for_axis[col2]
        else:
            xaxes_title = col1
            yaxes_title = col2
            
        if coloring:
            title = 'Зависимости числовых переменных с учетом плотности точек'
            if pairs[(col1, col2)]:
                if pairs[(col1, col2)][col1]:
                    df_trim = df[df[col1].between(*pairs[(col1, col2)][col1])]
                if pairs[(col1, col2)][col2]:
                    df_trim = df_trim[df_trim[col2].between(*pairs[(col1, col2)][col2])]
            else:
                df_trim = df.copy()        
            if density_mode == 'count':
                df_trim['x_sector'] = pd.cut(df_trim[col1], bins=bins)
                df_trim['y_sector'] = pd.cut(df_trim[col2], bins=bins)
                # df_trim.head()
                df_trim['density'] = df_trim.groupby(['x_sector', 'y_sector'], observed=True)[col1].transform('size')
            elif density_mode == 'kde':
                xy = np.vstack([df_trim[col1], df_trim[col2]])
                df_trim['density']  = gaussian_kde(xy)(xy)
            if i == 0:
                max_density = df_trim['density'].max()
            fig.add_trace(go.Scattergl(
                x=df_trim[col1], 
                y=df_trim[col2], 
                hovertemplate=xaxes_title + ' = %{x}<br>' + yaxes_title + ' = %{y}<extra></extra>', 
                mode='markers', 
                marker=dict(
                    color=df_trim['density'],  # Используем значения для цветовой шкалы
                    coloraxis=f"coloraxis{i+1}",
                    # showscale=False,
                    line=dict(color='white', width=0.5)
                    # Убираем coloraxis, чтобы не показывать colorbar
                ),
            ), row=row+1, col=col+1)            
            # fig_scatter = px.scatter(
            #     df, x=col1, y=col2,
            #     color='density',
            #     color_continuous_scale = 'Viridis',
            #     # color_continuous_scale=[(0, 'rgba(204, 153, 255, 0.1)'), (1, 'rgb(127, 60, 141)')],
            #     render_mode='webgl',
            # )
            # fig_scatter.update_traces(marker=dict(line=dict(color='white', width=0.5)))
            # fig_scatter.update_layout(coloraxis=dict(colorbar=dict(title='Density', titlefont=dict(size=12))), coloraxis_colorbar=coloraxis_name)
            # fig_scatter.update_layout(coloraxis_colorbar=dict(title='Density'))
        else:
            title = 'Зависимости между числовыми переменными'
            if pairs[(col1, col2)]:
                if pairs[(col1, col2)][col1]:
                    df_trim = df[df[col1].between(*pairs[(col1, col2)][col1])]
                if pairs[(col1, col2)][col2]:
                    df_trim = df_trim[df_trim[col2].between(*pairs[(col1, col2)][col2])]      
            else:
                df_trim = df.copy()
            fig_scatter = px.scatter(df_trim, x=col1, y=col2, render_mode='webgl')
            fig_scatter.update_traces(marker=dict(line=dict(color='white', width=0.5)))
            fig.add_trace(fig_scatter.data[0], row=row+1, col=col+1)
        # fig_scatter.update_traces(marker=dict(
        #     line=dict(color='white', width=0.5))) #, coloraxis=f"coloraxis{i+1}", showscale=False))
        # fig.add_trace(fig_scatter.data[0], row=row+1, col=col+1)
        # fig.update_coloraxes
        fig.update_xaxes(
            title_text=xaxes_title,
            title_font=dict(size=16, color="rgba(0, 0, 0, 0.5)"),
            tickfont=dict(size=14, color="rgba(0, 0, 0, 0.5)"),
            linecolor="rgba(0, 0, 0, 0.5)",
            row=row+1, col=col+1,
            showgrid=True,
            gridwidth=1,
            gridcolor="rgba(0, 0, 0, 0.1)"
        )
        fig.update_yaxes(
            title_text=yaxes_title,
            title_font=dict(size=16, color="rgba(0, 0, 0, 0.5)"),
            tickfont=dict(size=14, color="rgba(0, 0, 0, 0.5)"),
            linecolor="rgba(0, 0, 0, 0.5)",
            row=row+1, col=col+1,
            showgrid=True,
            gridwidth=1,
            gridcolor="rgba(0, 0, 0, 0.07)"
        )
    # Добавляем цветовую шкалу

    # fig.update_layout(coloraxis=dict(colorbar=dict(title='Density', showticks=False)))
    # fig.update_layout(coloraxis=dict(colorbar=dict(title='Density', titlefont=dict(size=12))), coloraxis_colorbar=coloraxis_name)
    # fig.update_layout(coloraxis1=dict(colorbar=dict(title=f'Colorbar {i+1}', x=1.05, y=0.5, thickness=20)))
        # coloraxis2=dict(colorbar=dict(title='Colorbar 2', x=1.05, y=0.5, thickness=20)))
    # # Update the layout
    fig.update_layout(coloraxis1=dict(colorbar=dict(title='Плотность', tickvals=[max_density * i / 10 for i in range(10)], ticktext=[f'{i * 10}%' for i in range(10)])))
    for i in range(2, len(fig.data)+1):
        fig.update_layout(**{f'coloraxis{i}': dict(showscale=False)})
        
    fig.update_layout(
        height=height,
        width=width,
        margin=dict(l=50, r=50, t=90, b=50),
        title=title,
        # Для подписей и меток
        title_font=dict(size=16, color="rgba(0, 0, 0, 0.7)"),     
        font=dict(size=14, family="Segoe UI", color="rgba(0, 0, 0, 0.7)"),
        xaxis_title_font=dict(size=14, color="rgba(0, 0, 0, 0.7)"),
        yaxis_title_font=dict(size=14, color="rgba(0, 0, 0, 0.7)"),
        xaxis_tickfont=dict(size=14, color="rgba(0, 0, 0, 0.7)"),
        yaxis_tickfont=dict(size=14, color="rgba(0, 0, 0, 0.7)"),
        xaxis_linecolor="rgba(0, 0, 0, 0.4)",
        yaxis_linecolor="rgba(0, 0, 0, 0.4)", 
        xaxis_tickcolor="rgba(0, 0, 0, 0.4)",
        yaxis_tickcolor="rgba(0, 0, 0, 0.4)",  
        legend_title_font_color='rgba(0, 0, 0, 0.7)',
        legend_title_font_size = 14,
        legend_font_color='rgba(0, 0, 0, 0.7)',
        hoverlabel=dict(bgcolor="white"),
    )
    fig.update_layout(showlegend=False)
    return fig


def histograms_stacked(
    df: pd.DataFrame,
    x: str,
    category: str,
    xaxis_title: str = None,
    yaxis_title: str = None,
    title: str = None,
    top_n: int = 20,
    lower_quantile: float = 0,
    upper_quantile: float = 1,
    nbins: int = 20,
    line_width: int = 5,
    opacity: float = 0.7,
    height: int = None,
    width: int = None,
    mode: str = 'normal',
    barmode: str = 'group',
    legend_position: str = 'top',
    show_box: bool = True,
    use_contrast_colors: bool = False,
) -> go.Figure:
    """
    Creates stacked histograms for categorical variables.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    x : str
        Name of numerical variable
    category : str
        Name of categorical variable        
    xaxis_title : str, optional
        Title for x-axis
    yaxis_title : str, optional
        Title for y-axis        
    title : str, optional
        Title for plot
    top_n : int, optional
        Number of top categories to display. Default is 20
    lower_quantile : float, optional
        Lower quantile for data trimming (0-1). Default is 0
    upper_quantile : float, optional
        Upper quantile for data trimming (0-1). Default is 1
    nbins : int, optional
        Number of histogram bins. Default is 20
    line_width : int, optional
        Width of plot lines. Default is 5
    opacity : float, optional
        Transparency of lines (0-1). Default is 0.7
    height : int, optional
        Plot height in pixels
    width : int, optional
        Plot width in pixels
    mode : str, optional
        Display mode ('step' or 'normal'). Default is 'normal'
    barmode : str, optional
        Multi-plot display mode ('group', 'stack', 'overlay', 'relative'). Default is 'group'
    legend_position : str, optional
        Legend position ('top' or 'right'). Default is 'top'
    show_box : bool, optional
        Show box plot. Default is True
    use_contrast_colors : bool, optional
        Use special contrasting colors for histograms. Default is False

    Returns
    -------
    go.Figure
        Plotly figure with stacked histograms

    """

    cat_var = category
    num_var = x
    bins = nbins
    if not xaxis_title:
        xaxis_title = num_var
    if not yaxis_title:
        yaxis_title = 'Доля от общего'
    if not legend_title:
        legend_title = cat_var

    
        
    # if not titles_for_axis:
    #     title = f'Гистограмма для {num_var} в зависимости от {cat_var}'
    #     xaxis_title = num_var
    #     yaxis_title = 'Частота'
    #     legend_title = cat_var
    # else:
    #     title = config['title']
    #     xaxis_title = f'{titles_for_axis[num_var]}'
    #     yaxis_title = 'Частота'    
    #     legend_title = f'{titles_for_axis[cat_var]}'
    # if legend_position == 'top':
    #     legend_title = None
    # Получение топ N категорий
    categories = df[cat_var].value_counts().nlargest(top_n).index.tolist()

    if mode == 'step':
        # Создание графика
        dist_plot = go.Figure()
        box_plot = go.Figure()
        # colors = ['rgb(127, 60, 141)', 'rgb(17, 165, 121)',
        #         '#03A9F4', 'rgb(242, 183, 1)', 'rgb(231, 63, 116)', '#8B9467', '#FFA07A', '#005A5B', 
        #         '#66CCCC', '#B690C4']
        # colors_box = ['rgba(128, 60, 170, 0.9)', "rgba(112, 155, 219, 0.9)", '#049CB3', "rgba(99, 113, 156, 0.9)", '#5c6bc0', '#B690C4', 'rgba(17, 100, 120, 0.9)', 'rgba(194, 143, 113, 0.8)', '#B690C4', '#03A9F4', '#8B9467', '#a771f2', 'rgba(102, 204, 204, 0.9)', 'rgba(168, 70, 90, 0.9)', 'rgba(50, 152, 103, 0.8)', '#8F7A7A', 'rgba(156, 130, 217, 0.9)'
        #                 ]    
        # colors = ['rgba(128, 60, 170, 0.9)', '#049CB3', "rgba(112, 155, 219, 0.9)", "rgba(99, 113, 156, 0.9)", '#5c6bc0', '#B690C4', 'rgba(17, 100, 120, 0.9)', 'rgba(194, 143, 113, 0.8)', '#B690C4', '#03A9F4', '#8B9467', '#a771f2', 'rgba(102, 204, 204, 0.9)', 'rgba(168, 70, 90, 0.9)', 'rgba(50, 152, 103, 0.8)', '#8F7A7A', 'rgba(156, 130, 217, 0.9)']
        if use_contrast_colors:
            colors = colorway_for_stacked_histogram
            colors_box = colorway_for_stacked_histogram
        else:
            colors = ['rgba(128, 60, 170, 0.9)', '#049CB3', "rgba(112, 155, 219, 0.9)", "rgba(99, 113, 156, 0.9)", '#5c6bc0', '#B690C4', 'rgba(17, 100, 120, 0.9)', 'rgba(194, 143, 113, 0.8)', '#B690C4', '#03A9F4', '#8B9467', '#a771f2', 'rgba(102, 204, 204, 0.9)', 'rgba(168, 70, 90, 0.9)', 'rgba(50, 152, 103, 0.8)', '#8F7A7A', 'rgba(156, 130, 217, 0.9)']
            colors_box = colors            
        hist_data = []
        max_y = -np.inf
        # Проход по каждой категории и построение гистограммы
        for indx, category in enumerate(categories):
            data = df[df[cat_var] == category][num_var]
            
            # Обрезка данных по квантилям
            lower_bound = data.quantile(lower_quantile)
            upper_bound = data.quantile(upper_quantile)
            trimmed_data = data[(data >= lower_bound) & (data <= upper_bound)]
            hist_data.append(trimmed_data)
            if show_box:
                box_plot.add_trace(go.Box(
                    x=data,
                    name=str(category),
                    # boxmean='sd',  # Отображение среднего и стандартного отклонения
                    orientation='h',
                    notched = True,
                    showlegend = False,
                    marker_color = colors_box[indx]
                ))
            # Вычисление гистограммы
            hist_values, bin_edges = np.histogram(trimmed_data, bins=bins)
            hist_values = np.append(hist_values, 0)
            # Нормирование значений гистограммы в процентах
            hist_values_percent = hist_values / hist_values.sum() * 100
            if hist_values_percent.max() > max_y:
                max_y = hist_values_percent.max() + 0.01 * hist_values_percent.max()
            # Подготовка данных для ступенчатого графика
            x_step = []
            y_step = []

            for i in range(len(hist_values_percent)):
                x_step.append(bin_edges[i])  # Точка на оси X
                y_step.append(0 if i == 0 else hist_values_percent[i-1])  # Если первая точка, то 0, иначе - предыдущее значение
                x_step.append(bin_edges[i])  # Точка на оси X для вертикального подъема
                y_step.append(hist_values_percent[i])  # Значение гистограммы
            # Добавление линии ступеней на график
            dist_plot.add_trace(go.Scatter(
                x=x_step,
                y=y_step,
                mode='lines',
                name=str(category),
                line=dict(width=line_width, color=colors[indx % len(colors)]),
                opacity=opacity,  # Установка прозрачности
            ))
        # Настройка графика
        dist_plot.update_traces(
            hovertemplate= xaxis_title + ' = %{x}<br>Частота = %{y:.2f}<extra></extra>')
        box_plot.update_traces(
            hovertemplate= xaxis_title + ' = %{x}<extra></extra>')
        # Объединяем графики
        if show_box:
            fig = make_subplots(rows=2, cols=1, row_heights=[0.1, 0.9], shared_xaxes=True)        
            # Добавляем распределительный график
            if show_box:
                for trace in box_plot.data:
                    fig.add_trace(trace, row=1, col=1)
            for trace in dist_plot.data:    
            # Добавляем boxplot
                fig.add_trace(trace, row=2, col=1)
        else:
            fig = dist_plot
        # Отображение графика
        # fig.update_xaxes(
        #     title=xaxis_title,
        #     title_font=dict(size=14, color="rgba(0, 0, 0, 0.7)"),
        #     tickfont=dict(size=14, color="rgba(0, 0, 0, 0.7)"),
        #     linecolor="rgba(0, 0, 0, 0.4)",
        #     tickcolor="rgba(0, 0, 0, 0.4)",
        #     showgrid=True, gridwidth=1, gridcolor="rgba(0, 0, 0, 0.1)"
        #     , row=2, col=1
        # )
        # fig.update_yaxes(
        #     # domain=[0.8, 0.9],
        #     title=yaxis_title,
        #     title_font=dict(size=14, color="rgba(0, 0, 0, 0.7)"),
        #     tickfont=dict(size=14, color="rgba(0, 0, 0, 0.7)"),
        #     linecolor="rgba(0, 0, 0, 0.4)",
        #     tickcolor="rgba(0, 0, 0, 0.4)",
        #     showgrid=True, gridwidth=1, gridcolor="rgba(0, 0, 0, 0.1)"
        #     , row=2, col=1
        # )    
        # fig.update_legends(
        #     title_text=legend_title,
        #     title_font_color='rgba(0, 0, 0, 0.7)',
        #     font_color='rgba(0, 0, 0, 0.7)',
        #     orientation="h",  # Горизонтальное расположение
        #     yanchor="top",    # Привязка к верхней части
        #     y=1.1,           # Положение по вертикали (отрицательное значение переместит вниз)
        #     xanchor="center",  # Привязка к центру
        #     x=0.5              # Центрирование по горизонтали                       
        # )
    elif mode == 'normal':
        # group_labels = categories
        # colors = ['#A56CC1', '#A6ACEC'] #, '#63F5EF']
        # colors = ['rgba(128, 60, 170, 0.9)', "rgba(112, 155, 219, 0.9)", '#049CB3']
        # Create distplot with curve_type set to 'normal'
        if show_box:
            marginal = 'box'
        else:
            marginal = None
        lower_quantile = df[num_var].quantile(lower_quantile)  # 25-й процентиль
        upper_quantile = df[num_var].quantile(upper_quantile)  # 75-й процентиль
        # Фильтруем DataFrame по квантилям
        filtered_df = df[(df[num_var] >= lower_quantile) & (df[num_var] <= upper_quantile)]
        if use_contrast_colors:    
            fig = px.histogram(filtered_df, x=num_var, color=cat_var, marginal=marginal, barmode=barmode, nbins=bins, histnorm='percent')
        # fig.update_traces(hovertemplate = xaxis_title + ' = %{x}<br>Частота = %{y:.2f}<extra></extra>')     
        # Обновление hovertemplate для гистограммы
        fig.update_traces(hovertemplate=xaxis_title + ' = %{x}<br>Частота = %{y:.2f}<extra></extra>', 
                          selector=dict(type='histogram'))        
        # Обновление hovertemplate для боксплота
        fig.update_traces(hovertemplate=xaxis_title + ' = %{x}<br><extra></extra>', 
                          selector=dict(type='box'))             
    else:
        raise ValueError("Invalid mode. Please choose 'box' or 'normal'.")
    if show_box:
        xaxis = dict(
                visible=False
            )  
        yaxis = dict(
                domain=[0.85, 0.95]
                , visible=False
            )   
        xaxis2 = dict(
            title=xaxis_title
            , title_font=dict(size=14, color="rgba(0, 0, 0, 0.7)")
            , tickfont=dict(size=14, color="rgba(0, 0, 0, 0.7)")
            , linecolor="rgba(0, 0, 0, 0.4)"
            , tickcolor="rgba(0, 0, 0, 0.4)"
            , showgrid=True, gridwidth=1, gridcolor="rgba(0, 0, 0, 0.1)"
        )
        yaxis2 = dict(
            domain=[0, 0.8]
            # , range=[0, max_y]
            , title=yaxis_title
            , title_font=dict(size=14, color="rgba(0, 0, 0, 0.7)")
            , tickfont=dict(size=14, color="rgba(0, 0, 0, 0.7)")
            , linecolor="rgba(0, 0, 0, 0.4)"
            , tickcolor="rgba(0, 0, 0, 0.4)"
            , showgrid=True, gridwidth=1, gridcolor="rgba(0, 0, 0, 0.1)"
        )        
        if mode == 'normal':        
            xaxis, xaxis2 = xaxis2, xaxis
            yaxis, yaxis2 = yaxis2, yaxis
        fig.update_layout(  
            xaxis = xaxis                  
            , yaxis = yaxis                  
            , xaxis2 = xaxis2
            , yaxis2 = yaxis2 
            , barmode=barmode
            , height=height
            , width=width
            , title=title
            , margin=dict(l=50, r=50, b=10, t=50)
            , title_font=dict(size=16, color="rgba(0, 0, 0, 0.7)")   
            , font=dict(size=14, family="Segoe UI", color="rgba(0, 0, 0, 0.7)")
            , hoverlabel=dict(bgcolor="white")
        )
        if mode == 'step':
            fig.update_layout(
                yaxis_range = [0, max_y]
            )
    else:
        fig.update_layout(
            xaxis = dict(
                title=xaxis_title
                , title_font=dict(size=14, color="rgba(0, 0, 0, 0.7)")
                , tickfont=dict(size=14, color="rgba(0, 0, 0, 0.7)")
                , linecolor="rgba(0, 0, 0, 0.4)"
                , tickcolor="rgba(0, 0, 0, 0.4)"
                , showgrid=True, gridwidth=1, gridcolor="rgba(0, 0, 0, 0.1)"
            )
            , yaxis = dict(
                domain=[0, 0.95]
                # , range=[0, max_y]
                , title=yaxis_title
                , title_font=dict(size=14, color="rgba(0, 0, 0, 0.7)")
                , tickfont=dict(size=14, color="rgba(0, 0, 0, 0.7)")
                , linecolor="rgba(0, 0, 0, 0.4)"
                , tickcolor="rgba(0, 0, 0, 0.4)"
                , showgrid=True, gridwidth=1, gridcolor="rgba(0, 0, 0, 0.1)"
            )     
            , barmode=barmode
            , height=height
            , width=width
            , title=title
            , margin=dict(l=50, r=50, b=10, t=50)
            , title_font=dict(size=16, color="rgba(0, 0, 0, 0.7)")   
            , font=dict(size=14, family="Segoe UI", color="rgba(0, 0, 0, 0.7)")
            , hoverlabel=dict(bgcolor="white")
        )
    if legend_position == 'top':
        fig.update_layout(
            legend = dict(
                title_text=legend_title
                , title_font=dict(size=14, color="rgba(0, 0, 0, 0.7)")
                , font_color='rgba(0, 0, 0, 0.7)'
                , orientation="h"  # Горизонтальное расположение
                , yanchor="top"    # Привязка к верхней части
                , y=1.05         # Положение по вертикали (отрицательное значение переместит вниз)
                , xanchor="center" # Привязка к центру
                , x=0.1              # Центрирование по горизонтали
            )     
        )    
    elif legend_position == 'right':
        fig.update_layout(
                legend = dict(
                title_text=legend_title
                , title_font=dict(size=14, color="rgba(0, 0, 0, 0.7)")
                , font_color='rgba(0, 0, 0, 0.7)'
                , orientation="v"  # Горизонтальное расположение
                # , yanchor="bottom"    # Привязка к верхней части
                , y=0.8         # Положение по вертикали (отрицательное значение переместит вниз)
                # , xanchor="center" # Привязка к центру
                # , x=0.5              # Центрирование по горизонтали
            )
        )
    else:
        raise ValueError("Invalid legend_position. Please choose 'top' or 'right'.")
    # fi, g.update_xaxes(visible=False, row=1, col=1)  # Убираем ось X для верхнего графика
    # fig.update_yaxes(visible=False, row=1, col=1)  # Убираем ось Y для верхнего графика       
    return fig

def boxplots_stacked(
    df: pd.DataFrame,
    x: str,
    y: str,
    category: str,
    xaxis_title: str = None,
    yaxis_title: str = None,
    legend_title: str = None,
    top_n: int = 20,
    height: int = None,
    width: int = None,
    lower_quantile: float = 0,
    upper_quantile: float = 1,
    sort: bool = False,
    legend_position: str = 'top',
    orientation: str = None,
    title: str = None,
) -> go.Figure:
    """
    Creates stacked box plots for categorical variables.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    x : str
        Name of categorical variable
    y : str
        Name of numerical variable
    top_n : int, optional
        Number of top categories to display. Default is 20
    height : int, optional
        Plot height in pixels
    width : int, optional
        Plot width in pixels
    lower_quantile : float, optional
        Lower quantile for data trimming (0-1). Default is 0
    upper_quantile : float, optional
        Upper quantile for data trimming (0-1). Default is 1
    sort : bool, optional
        Sort categories by values. Default is False
    category : str, optional
        Variable to use for legend grouping
    legend_position : str, optional
        Legend position ('top' or 'right'). Default is 'top'
    orientation : str, optional
        Orientation of the plot ('h' for horizontal, 'v' for vertical). Default is None
    title : str, optional
        Plot title
    titles_for_axis : dict, optional
        Dictionary of custom axis titles

    Returns
    -------
    go.Figure
        Plotly figure with stacked box plots
    """
    cat_var = x
    num_var = y
    legend_var = category

    def trim_by_quantiles(group):
        lower_bound = group.quantile(lower_quantile)
        upper_bound = group.quantile(upper_quantile)
        return group[(group >= lower_bound) & (group <= upper_bound)]        
    # if not titles_for_axis:
    #     title = f'Распределение {num_var} в зависимости от {cat_var}'
    #     xaxis_title = cat_var
    #     yaxis_title = num_var
    #     if legend_var:
    #         legend_title_text = legend_var
    # else:
    #     title = config['title']
    #     xaxis_title = f'{titles_for_axis[cat_var]}'
    #     yaxis_title = f'{titles_for_axis[num_var]}'
    #     if legend_var:
    #         legend_title_text = f'{titles_for_axis[legend_var]}'
    if not xaxis_title:
        xaxis_title = cat_var
    if not yaxis_title:
        yaxis_title = num_var
    if legend_title:
        legend_title = legend_var
    # Получение топ N категорий
    if not sort:
        categories = df[cat_var].value_counts().nlargest(top_n).index.tolist()
    else:
        if pd.api.types.is_categorical_dtype(df[cat_var]):
            categories = df[cat_var].cat.categories.tolist()[:top_n]
        elif pd.api.types.is_numeric_dtype(df[cat_var]):
            categories = sorted(df[cat_var].unique().tolist())[:top_n]
        else:
            categories = df[cat_var].unique().tolist()[:top_n]
    df = df[df[cat_var].isin(categories)]
    # Создание графика
    fig = go.Figure()

    # Применение функции к каждой категории
    columns_for_groupby = [cat_var]
    if legend_var:
        columns_for_groupby.append(legend_var)
    trimmed_df = df.groupby(columns_for_groupby, observed=True)[num_var].apply(trim_by_quantiles).reset_index()
    fig = px.box(trimmed_df, x=cat_var, y=num_var, color=legend_var, orientation=orientation)
    # # Проход по каждой категории и построение боксплота
    # for category in categories:
    #     data = df[df[cat_var] == category][num_var]
    #     if data.count() == 0:
    #         continue
    #     lower_bound = data.quantile(lower_quantile)
    #     upper_bound = data.quantile(upper_quantile)
    #     data = data[(data >= lower_bound) & (data <= upper_bound)]        
    #     fig.add_trace(go.Box(
    #         y=data,
    #         name=str(category),
    #         # boxmean='sd',  # Отображение среднего и стандартного отклонения
    #         orientation='v',
    #         notched = True,
    #     ))
    # Настройка графика
    if legend_var:
        showlegend = True
        if legend_position == 'top':
            fig.update_layout(
                legend = dict(
                    title_text=legend_title
                    , title_font_color='rgba(0, 0, 0, 0.7)'
                    , font_color='rgba(0, 0, 0, 0.7)'
                    , orientation="h"  # Горизонтальное расположение
                    , yanchor="top"    # Привязка к верхней части
                    , y=1.09         # Положение по вертикали (отрицательное значение переместит вниз)
                    , xanchor="center" # Привязка к центру
                    , x=0.5              # Центрирование по горизонтали
                )     
            )    
        elif legend_position == 'right':
            fig.update_layout(
                    legend = dict(
                    title_text=legend_title
                    , title_font_color='rgba(0, 0, 0, 0.7)'
                    , font_color='rgba(0, 0, 0, 0.7)'
                    , orientation="v"  # Горизонтальное расположение
                    # , yanchor="bottom"    # Привязка к верхней части
                    , y=0.8         # Положение по вертикали (отрицательное значение переместит вниз)
                    # , xanchor="center" # Привязка к центру
                    # , x=0.5              # Центрирование по горизонтали
                )
            )
        else:
            raise ValueError("Invalid legend_position. Please choose 'top' or 'right'.")                   
    else:
        showlegend = False
    fig.update_traces(showlegend=showlegend) #, marker_color='rgba(128, 60, 170, 0.9)')
    #     hovertemplate='Значение = %{x}<br>Частота = %{y:.2f}<extra></extra>')
    # Настройка графика
    fig.update_layout(
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        # legend_title_text=legend_title,
        barmode='overlay',
        height=height,
        width=width,
        title=title,
        # Для подписей и меток
        title_font=dict(size=16, color="rgba(0, 0, 0, 0.7)"),     
        font=dict(size=14, family="Segoe UI", color="rgba(0, 0, 0, 0.7)"),
        xaxis_title_font=dict(size=14, color="rgba(0, 0, 0, 0.7)"),
        yaxis_title_font=dict(size=14, color="rgba(0, 0, 0, 0.7)"),
        xaxis_tickfont=dict(size=14, color="rgba(0, 0, 0, 0.7)"),
        yaxis_tickfont=dict(size=14, color="rgba(0, 0, 0, 0.7)"),
        xaxis_linecolor="rgba(0, 0, 0, 0.4)",
        yaxis_linecolor="rgba(0, 0, 0, 0.4)", 
        xaxis_tickcolor="rgba(0, 0, 0, 0.4)",
        yaxis_tickcolor="rgba(0, 0, 0, 0.4)",  
        legend_title_font_color='rgba(0, 0, 0, 0.7)',
        legend_title_font_size = 14,
        legend_font_color='rgba(0, 0, 0, 0.7)',
        hoverlabel=dict(bgcolor="white"),
        xaxis=dict(
            visible=True, showgrid=True, gridwidth=1, gridcolor="rgba(0, 0, 0, 0.1)"
        ), yaxis=dict(
            range=[0, None], visible=True, showgrid=True, gridwidth=1, gridcolor="rgba(0, 0, 0, 0.07)"
        ),          
    )
    return fig

def violins_stacked(
    df: pd.DataFrame,
    x: str,
    y: str,
    category: str,
    xaxis_title: str = None,
    yaxis_title: str = None,
    legend_title: str = None,
    top_n: int = 20,
    height: int = None,
    width: int = None,
    lower_quantile: float = 0,
    upper_quantile: float = 1,
    sort: bool = False,
    legend_position: str = 'top',
    orientation: str = None,
    title: str = None,
) -> go.Figure:
    """
    Creates stacked box plots for categorical variables.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    x : str
        Name of categorical variable
    y : str
        Name of numerical variable
    top_n : int, optional
        Number of top categories to display. Default is 20
    height : int, optional
        Plot height in pixels
    width : int, optional
        Plot width in pixels
    lower_quantile : float, optional
        Lower quantile for data trimming (0-1). Default is 0
    upper_quantile : float, optional
        Upper quantile for data trimming (0-1). Default is 1
    sort : bool, optional
        Sort categories by values. Default is False
    category : str, optional
        Variable to use for legend grouping
    legend_position : str, optional
        Legend position ('top' or 'right'). Default is 'top'
    orientation : str, optional
        Orientation of the plot ('h' for horizontal, 'v' for vertical). Default is None
    title : str, optional
        Plot title
    xaxis_title : str, optional
        Title for the x-axis
    yaxis_title : str, optional
        Title for the y-axis
    legend_title : str, optional
        Title for the legend

    Returns
    -------
    go.Figure
        Plotly figure with stacked box plots
    """
        

    cat_var = category
    num_var = x
    
    # if not titles_for_axis:
    #     title = f'Распределение {num_var} в зависимости от {cat_var}'
    #     xaxis_title = cat_var
    #     yaxis_title = num_var
    # else:
    #     title = config['title']
    #     xaxis_title = f'{titles_for_axis[cat_var]}'
    #     yaxis_title = f'{titles_for_axis[num_var]}'
    xaxis_title = cat_var
    yaxis_title = num_var
    # Получение топ N категорий
    if not sort:
        categories = df[cat_var].value_counts().nlargest(top_n).index.tolist()
    else:
        if pd.api.types.is_categorical_dtype(df[cat_var]):
            categories = df[cat_var].cat.categories.tolist()[:top_n]
        elif pd.api.types.is_numeric_dtype(df[cat_var]):
            categories = sorted(df[cat_var].unique().tolist())[:top_n]
        else:
            categories = df[cat_var].unique().tolist()[:top_n]
    # Создание графика
    fig = go.Figure()

    # Проход по каждой категории и построение боксплота
    for category in categories:
        data = df[df[cat_var] == category][num_var]
        if data.count() == 0:
            continue
        lower_bound = data.quantile(lower_quantile)
        upper_bound = data.quantile(upper_quantile)
        data = data[(data >= lower_bound) & (data <= upper_bound)]        
        fig.add_trace(go.Violin(
            y=data,
            name=str(category),
            # boxmean='sd',  # Отображение среднего и стандартного отклонения
            orientation='v',
            box=dict(visible=True), 
        ))
    # Настройка графика

    fig.update_traces(showlegend=False, marker_color='rgba(128, 60, 170, 0.9)')
    #     hovertemplate='Значение = %{x}<br>Частота = %{y:.2f}<extra></extra>')
    # Настройка графика
    fig.update_layout(
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        # legend_title_text=legend_title,
        barmode='overlay',
        height=height,
        width=width,
        title=title,
        # Для подписей и меток
        title_font=dict(size=16, color="rgba(0, 0, 0, 0.7)"),     
        font=dict(size=14, family="Segoe UI", color="rgba(0, 0, 0, 0.7)"),
        xaxis_title_font=dict(size=14, color="rgba(0, 0, 0, 0.7)"),
        yaxis_title_font=dict(size=14, color="rgba(0, 0, 0, 0.7)"),
        xaxis_tickfont=dict(size=14, color="rgba(0, 0, 0, 0.7)"),
        yaxis_tickfont=dict(size=14, color="rgba(0, 0, 0, 0.7)"),
        xaxis_linecolor="rgba(0, 0, 0, 0.4)",
        yaxis_linecolor="rgba(0, 0, 0, 0.4)", 
        xaxis_tickcolor="rgba(0, 0, 0, 0.4)",
        yaxis_tickcolor="rgba(0, 0, 0, 0.4)",  
        legend_title_font_color='rgba(0, 0, 0, 0.7)',
        legend_title_font_size = 14,
        legend_font_color='rgba(0, 0, 0, 0.7)',
        hoverlabel=dict(bgcolor="white"),
        xaxis=dict(
            visible=True, showgrid=True, gridwidth=1, gridcolor="rgba(0, 0, 0, 0.1)"
        ), yaxis=dict(
            visible=True, showgrid=True, gridwidth=1, gridcolor="rgba(0, 0, 0, 0.07)"
        ),          
    )
    return fig

def plot_confidence_intervals_old(df, categorical_col, numerical_col, confidence_level=0.95, orientation='vertical', height=600, width=800, titles_for_axis=None):
    """
    Функция для построения графика с средними значениями и доверительными интервалами с использованием t-статистики.

    Параметры:
    df (pd.DataFrame): Входной DataFrame.
    categorical_col (str): Название категориальной переменной.
    numerical_col (str): Название числовой переменной.
    confidence_level (float): Уровень доверия для доверительного интервала (по умолчанию 0.95).
    orientation (str): Ориентация графика ('vertical' или 'horizontal').
    
    Пример словаря для подписей осей и заголовка:
    titles_for_axis = dict(
        # numeric column ['Именительный падеж', 'мменительный падеж с маленькой буквы', 'род цифорой']
        # (0 - средний род, 1 - мужской род, 2 - женский род[) (Середнее образовние, средний доход, средняя температура) )
        # для функций count и nunique пишем - Количество <чего / кого количество> - и также с маленькой буквы, цифра 0 в качестве рода
        body_mass_g = ['Вес', 'вес', 1]
        # categorical column ['Именительный падеж', 'для кого / чего', 'по кому чему']
        # Распределение долей по городу и тарифу с нормализацией по городу
        , island = ['Остров', 'острова', 'острову']
    )
    """
    func_for_title = ['Среднее', 'Средний', 'Средняя', 'Средние']
    suffix_type = titles_for_axis[numerical_col][2]
    if not titles_for_axis:
        title = f'Среднее {numerical_col} в зависимости от {categorical_col} с {int(confidence_level*100)}% доверительными интервалами'
        xaxis_title = categorical_col
        yaxis_title = numerical_col
    else:
        title = f'{func_for_title[suffix_type]} {titles_for_axis[numerical_col][1]} в зависимости от {titles_for_axis[categorical_col][1]} с {int(confidence_level*100)}% доверительными интервалами'
        xaxis_title = f'{titles_for_axis[categorical_col][0]}'
        yaxis_title = f'{titles_for_axis[numerical_col][0]}'
    # Группируем данные и вычисляем среднее, стандартное отклонение и количество наблюдений
    summary_df = df.groupby(categorical_col)[numerical_col].agg(["mean", "std", "count"]).reset_index()
    # Вычисляем t-статистику для заданного уровня доверия
    degrees_of_freedom = summary_df["count"] - 1  # Степени свободы
    alpha = 1 - confidence_level  # Уровень значимости
    t_score = t.ppf(1 - alpha / 2, degrees_of_freedom)  # t-статистика
    
    # Вычисляем доверительный интервал
    summary_df["ci"] = t_score * summary_df["std"] / (summary_df["count"] ** 0.5)
    
    # Определяем ориентацию графика
    if orientation == 'v':
        x_col = categorical_col
        y_col = "mean"
        if titles_for_axis:
            hovertemplate = 'Среднее = %{y}<br>' + f'{titles_for_axis[categorical_col][0]} = ' + '%{x}'
    elif orientation == 'h':
        x_col = "mean"
        y_col = categorical_col
        xaxis_title, yaxis_title = yaxis_title, xaxis_title
        if titles_for_axis:
            hovertemplate = 'Среднее = %{x}<br>' + f'{titles_for_axis[categorical_col][0]} = ' + '%{y}'
    else:
        raise ValueError("Ориентация должна быть 'vertical' или 'horizontal'.")
    # error_y в scatter-графике (точечном графике) вы указываете значение, которое определяет длину отрезка (ошибки) вокруг каждой точки по оси Y.
    # Создаем график
    fig = px.scatter(summary_df, x=x_col, y=y_col, 
                     error_y="ci" if orientation == 'v' else None,
                     error_x="ci" if orientation == 'h' else None
    )
    fig.update_traces(hovertemplate=hovertemplate)
    fig.update_layout(height=height, width=width, title_text=title, xaxis_title=xaxis_title, yaxis_title=yaxis_title)
    return plotly_default_settings(fig)

def plot_confidence_intervals(
    df: pd.DataFrame,
    categorical_col: str,
    numerical_col: str,
    second_categorical_col: str = None,
    confidence_level: float = 0.95,
    orientation: str = 'vertical',
    height: int = 600,
    width: int = 800,
    legend_position: str = 'top',
    title: str = None,
    xaxis_title: str = None, 
    yaxis_title: str = None,
    legend_title: str = None
) -> go.Figure:
    """
    Creates a plot with mean values and confidence intervals using t-statistics.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    categorical_col : str
        Name of primary categorical variable
    numerical_col : str
        Name of numerical variable
    second_categorical_col : str, optional
        Name of secondary categorical variable for data grouping
    confidence_level : float, optional
        Confidence level for interval calculation (0-1). Default is 0.95
    orientation : str, optional
        Plot orientation ('vertical' or 'horizontal'). Default is 'vertical'
    height : int, optional
        Plot height in pixels. Default is 600
    width : int, optional
        Plot width in pixels. Default is 800
    legend_position : str, optional
        Legend position ('top' or 'right'). Default is 'top'
    title : str, optional
        Plot title. Default is None
    xaxis_title : str, optional
        X-axis title. Default is None
    yaxis_title : str, optional
        Y-axis title. Default is None
    legend_title : str, optional
        Legend title. Default is None

    Returns
    -------
    go.Figure
        Plotly figure with confidence intervals

    Examples
    --------
    >>> plot_confidence_intervals(df, 'category', 'values')
    
    >>> plot_confidence_intervals(
    ...     df, 
    ...     'island', 
    ...     'body_mass_g', 
    ...     second_categorical_col='species',
    ... )
    """

    # func_for_title = ['Среднее', 'Средний', 'Средняя', 'Средние']
    # suffix_type = titles_for_axis[numerical_col][2] if titles_for_axis else 0

    # if not titles_for_axis:
    #     # title = f'Среднее {numerical_col} в зависимости от {categorical_col} с {int(confidence_level*100)}% доверительными интервалами'
    #     xaxis_title = categorical_col
    #     yaxis_title = numerical_col
    # else:
    #     if second_categorical_col:
    #         title = f'{func_for_title[suffix_type]} {titles_for_axis[numerical_col][1]} в зависимости от {titles_for_axis[categorical_col][1]} и {titles_for_axis[second_categorical_col][1]} с {int(confidence_level*100)}% доверительными интервалами'
    #     else:
    #         title = f'{func_for_title[suffix_type]} {titles_for_axis[numerical_col][1]} в зависимости от {titles_for_axis[categorical_col][1]} с {int(confidence_level*100)}% доверительными интервалами'
    #     xaxis_title = f'{titles_for_axis[categorical_col][0]}'
    #     yaxis_title = f'{titles_for_axis[numerical_col][0]}'

    # Группируем данные и вычисляем среднее, стандартное отклонение и количество наблюдений
    if second_categorical_col:
        summary_df = df.groupby([categorical_col, second_categorical_col])[numerical_col].agg(["mean", "std", "count"]).reset_index()
    else:
        summary_df = df.groupby(categorical_col)[numerical_col].agg(["mean", "std", "count"]).reset_index()

    # Вычисляем t-статистику для заданного уровня доверия
    degrees_of_freedom = summary_df["count"] - 1  # Степени свободы
    alpha = 1 - confidence_level  # Уровень значимости
    t_score = t.ppf(1 - alpha / 2, degrees_of_freedom)  # t-статистика

    # Вычисляем доверительный интервал
    summary_df["ci"] = t_score * summary_df["std"] / (summary_df["count"] ** 0.5)

    # Определяем ориентацию графика
    if orientation == 'v':
        x_col = categorical_col
        y_col = "mean"
        if xaxis_title:
            hovertemplate = 'Среднее = %{y}<br>' + f'{xaxis_title} = ' + '%{x}'
    elif orientation == 'h':
        x_col = "mean"
        y_col = categorical_col
        xaxis_title, yaxis_title = yaxis_title, xaxis_title
        if xaxis_title:
            hovertemplate = 'Среднее = %{x}<br>' + f'{xaxis_title} = ' + '%{y}'
    else:
        raise ValueError("Ориентация должна быть 'vertical' или 'horizontal'.")

    if second_categorical_col:
        # Преобразуем категориальные значения в числовые для расчета смещения
        unique_categories = summary_df[x_col].unique()
        category_to_num = {category: i for i, category in enumerate(unique_categories)}
        
        # Создаем фигуру
        fig = go.Figure()

        # Для каждой категории в second_categorical_col добавляем отдельный trace
        for i, category in enumerate(summary_df[second_categorical_col].unique()):
            df_subset = summary_df[summary_df[second_categorical_col] == category]
            
            # Преобразуем категории в числовые значения и добавляем смещение
            x_values = df_subset[x_col].map(category_to_num)
            x_values = x_values + 0.1 * i
            
            fig.add_trace(go.Scatter(
                x=x_values,
                y=df_subset[y_col],
                error_y=dict(type='data', array=df_subset["ci"], visible=True) if orientation == 'v' else None,
                error_x=dict(type='data', array=df_subset["ci"], visible=True) if orientation == 'h' else None,
                mode='markers',
                name=category,
                hovertemplate=hovertemplate
            ))
        
        # Настраиваем ось X, чтобы отображать категории
        fig.update_xaxes(
            tickvals=list(category_to_num.values()),
            ticktext=list(category_to_num.keys())
        )
    else:
        # Если вторая категориальная переменная не указана, строим обычный график
        fig = px.scatter(summary_df, x=x_col, y=y_col, 
                        error_y="ci" if orientation == 'v' else None,
                        error_x="ci" if orientation == 'h' else None,
                        title=title)

    # Настраиваем подсказки (hovertemplate)
    fig.update_traces(hovertemplate=hovertemplate)
    if second_categorical_col:
        if legend_position == 'top':
            fig.update_layout(
                yaxis = dict(
                    domain=[0, 0.95]
                )              
                , legend = dict(
                    title_text=legend_title if legend_title else second_categorical_col
                    , title_font_color='rgba(0, 0, 0, 0.7)'
                    , font_color='rgba(0, 0, 0, 0.7)'
                    , orientation="h"  # Горизонтальное расположение
                    , yanchor="top"    # Привязка к верхней части
                    , y=1.05         # Положение по вертикали (отрицательное значение переместит вниз)
                    , xanchor="center" # Привязка к центру
                    , x=0.1              # Центрирование по горизонтали
                )     
            )    
        elif legend_position == 'right':
            fig.update_layout(
                    legend = dict(
                    title_text=legend_title if legend_title else second_categorical_col
                    , title_font_color='rgba(0, 0, 0, 0.7)'
                    , font_color='rgba(0, 0, 0, 0.7)'
                    , orientation="v"  # Горизонтальное расположение
                    # , yanchor="bottom"    # Привязка к верхней части
                    , y=1         # Положение по вертикали (отрицательное значение переместит вниз)
                    # , xanchor="center" # Привязка к центру
                    # , x=0.5              # Центрирование по горизонтали
                )
            )
        else:
            raise ValueError("Invalid legend_position. Please choose 'top' or 'right'.")         
    # Настраиваем макет графика
    fig.update_layout(
        height=height,
        width=width,
        title_text=title,
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title
    )
    return plotly_default_settings(fig)

def subplots(
    configs,
    title=None,
    width=None,
    height=None,
    rows=1,
    cols=2,
    shared_xaxes=False,
    shared_yaxes=False,
    horizontal_spacing=None,
    specs=None,
    column_widths=None,
    row_heights=None,
    subplot_titles=None
):
    """
    Creates a figure with multiple subplots using Plotly.

    Parameters:
    configs (list): List of dictionaries containing configuration for each subplot
        Required keys:
        - fig: The plotly figure object
        - layout: The plotly layout object
        - row (int): Row position of the subplot
        - col (int): Column position of the subplot
        Optional keys:
        - is_margin (bool): Boolean to indicate if the plot is a margin plot
        - domain_x (list): X-axis domain range
        - domain_y (list): Y-axis domain range
        - showgrid_x (bool): Show X-axis grid
        - showgrid_y (bool): Show Y-axis grid
        - showticklabels_x (bool): Show X-axis tick labels
        - xaxis_visible (bool): X-axis visibility
        - yaxis_visible (bool): Y-axis visibility
        - show_yaxis_title (bool): Whether show yaxis title
    title (str): Main figure title
    width (int): Figure width in pixels
    height (int): Figure height in pixels
    rows (int): Number of rows in subplot grid
    cols (int): Number of columns in subplot grid
    shared_xaxes (bool): Share X axes between subplots
    shared_yaxes (bool): Share Y axes between subplots
    horizontal_spacing (float): Spacing between subplots horizontally
    specs (list): Subplot specifications
    column_widths (list): List of relative column widths
    row_heights (list): List of relative row heights
    subplot_titles (list): List of subplot titles

    Returns:
    plotly.graph_objects.Figure: The created figure with subplots
    """

    # Create subplot layout
    fig = make_subplots(
        rows=rows,
        cols=cols,
        column_widths=column_widths,
        row_heights=row_heights,
        specs=specs,
        subplot_titles=subplot_titles,
        shared_xaxes=shared_xaxes,
        shared_yaxes=shared_yaxes,
        horizontal_spacing=horizontal_spacing
    )

    # Process each subplot configuration
    if configs:
        for config in configs:
            if not config['fig']:
                continue

            # Set default values for configuration
            config['is_margin'] = config.get('is_margin', False)
            config['with_margin'] = config.get('with_margin', False)
            config['show_yaxis_title'] = config.get('show_yaxis_title', True)
            config['show_xaxis_title'] = config.get('show_xaxis_title', True)

            # Handle margin plot settings
            if config['is_margin']:
                config['domain_y'] = config.get('domain_y', [0.95, 1])
                config['showgrid_x'] = config.get('showgrid_x', False)
                config['showgrid_y'] = config.get('showgrid_y', False)
                config['showticklabels_x'] = config.get('showticklabels_x', False)
                config['xaxis_visible'] = config.get('xaxis_visible', False)
                config['yaxis_visible'] = config.get('yaxis_visible', False)
            else:
                # Handle regular plot settings
                if 'with_margin' in config:
                    config['domain_y'] = config.get('domain_y', [0, 0.9])
                config['showgrid_x'] = config.get('showgrid_x', True)
                config['showgrid_y'] = config.get('showgrid_y', True)
                config['showticklabels_x'] = config.get('showticklabels_x', True)
                config['xaxis_visible'] = config.get('xaxis_visible', True)
                config['yaxis_visible'] = config.get('yaxis_visible', True)

            # Set axis titles
            if config['show_xaxis_title']:
                config['xaxis_title_text'] = config['layout'].xaxis.title.text if 'layout' in config else None
            else:
                config['xaxis_title_text'] = None
            if config['show_yaxis_title']:
                config['yaxis_title_text'] = config['layout'].yaxis.title.text if 'layout' in config else None
            else:
                config['yaxis_title_text'] = None

            # Add trace and update axes
            fig.add_trace(config['fig'], row=config['row'], col=config['col'])

            # Update X axes
            fig.update_xaxes(
                row=config['row'],
                col=config['col'],
                showgrid=config['showgrid_x'],
                showticklabels=config['showticklabels_x'],
                visible=config['xaxis_visible'],
                title_text=config['xaxis_title_text'],
                gridwidth=1,
                gridcolor="rgba(0, 0, 0, 0.1)"
            )

            # Update domains if specified
            if 'domain_x' in config:
                fig.update_xaxes(row=config['row'], col=config['col'], domain=config['domain_x'])
            if 'domain_y' in config:
                fig.update_yaxes(row=config['row'], col=config['col'], domain=config['domain_y'])

            # Update Y axes
            fig.update_yaxes(
                row=config['row'],
                col=config['col'],
                showgrid=config['showgrid_y'],
                gridwidth=1,
                gridcolor="rgba(0, 0, 0, 0.1)",
                visible=config['yaxis_visible'],
                title_text=config['yaxis_title_text']
            )

    # Adjust subplot titles position
    if subplot_titles:
        for i, _ in enumerate(subplot_titles):
            fig['layout']['annotations'][i-1]['y'] = 1.04

    # Update figure layout
    fig.update_layout(
        title=title,
        width=width,
        height=height,
        title_font=dict(size=16, color="rgba(0, 0, 0, 0.7)"),
        font=dict(size=14, family="Segoe UI", color="rgba(0, 0, 0, 0.7)"),
        xaxis_showticklabels=True,
        xaxis_title_font=dict(size=14, color="rgba(0, 0, 0, 0.7)"),
        yaxis_title_font=dict(size=14, color="rgba(0, 0, 0, 0.7)"),
        xaxis_tickfont=dict(size=14, color="rgba(0, 0, 0, 0.7)"),
        yaxis_tickfont=dict(size=14, color="rgba(0, 0, 0, 0.7)"),
        xaxis_linecolor="rgba(0, 0, 0, 0.4)",
        yaxis_linecolor="rgba(0, 0, 0, 0.4)",
        xaxis_tickcolor="rgba(0, 0, 0, 0.4)",
        yaxis_tickcolor="rgba(0, 0, 0, 0.4)",
        legend_title_font_color='rgba(0, 0, 0, 0.7)',
        legend_title_font_size=14,
        legend_font_color='rgba(0, 0, 0, 0.7)',
        hoverlabel=dict(bgcolor="white")
    )

    return fig
