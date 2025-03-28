# import importlib
# importlib.reload(pgdt)
# make k format plotly - texttemplate='%{z:.2s}'
# png render fig.show(config=dict(displayModeBar=False), renderer="png")
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
import re
from pingouin import qqplot as pg_qqplot
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
from statsmodels.api import qqplot as sm_qqplot

pd_style_cmap = LinearSegmentedColormap.from_list("custom_white_purple", ['#f1edf5', '#7f3c8d'])
pio.renderers.default = "notebook"
# colorway_for_line = ['rgb(127, 60, 141)', 'rgb(17, 165, 121)', 'rgb(231, 63, 116)',
#                      '#03A9F4', 'rgb(242, 183, 1)', '#8B9467', '#FFA07A', '#005A5B', '#66CCCC', '#B690C4', 'rgb(127, 60, 141)', 'rgb(17, 165, 121)', 'rgb(231, 63, 116)',
#                      '#03A9F4', 'rgb(242, 183, 1)', '#8B9467', '#FFA07A', '#005A5B', '#66CCCC', '#B690C4']
colorway_for_line = [
    'rgb(127, 60, 141)',  # Цвет 1
    'rgb(17, 165, 121)',   # Цвет 2
    'rgb(231, 63, 116)',   # Цвет 3
    'rgb(3, 169, 244)',    # Цвет 4
    'rgb(242, 183, 1)',     # Цвет 5
    'rgb(139, 148, 103)',   # Цвет 6
    'rgb(255, 160, 122)',   # Цвет 7
    'rgb(0, 90, 91)',       # Цвет 8
    'rgb(102, 204, 204)',    # Цвет 9
    'rgb(182, 144, 196)'     # Цвет 10
]
colorway_tableau = ['#1f77b4',  # muted blue
 '#ff7f0e',  # safety orange
 '#2ca02c',  # cooked asparagus green
 '#d62728',  # brick red
 '#9467bd',  # muted purple
 '#8c564b',  # chestnut brown
 '#e377c2',  # raspberry yogurt pink
 '#7f7f7f',  # middle gray
 '#bcbd22',  # curry yellow-green
 '#17becf']  # blue-teal

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
colorway_for_heatmap = [[0, 'rgba(204, 153, 255, 0.1)'], [1, 'rgb(127, 60, 141)']]
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

def fig_update(
    fig: go.Figure,
    title: str = None,
    xaxis_title: str = None,
    yaxis_title: str = None,
    legend_title: str = None,
    height: int = None,
    width: int = None,
    showlegend: bool = True,
    xaxis_showgrid: bool = True,
    yaxis_showgrid: bool = True,
    xaxis_tickformat: str = None,
    yaxis_tickformat: str = None,
    xaxis_tickprefix: str = None,
    xaxis_ticksuffix: str = None,
    yaxis_tickprefix: str = None,
    yaxis_ticksuffix: str = None,
    hovertemplate: str = None,
    xaxis_dtick: float = None,
    yaxis_dtick: float = None,
    xaxis_ticktext: list = None,
    xaxis_tickvals: list = None,
    yaxis_ticktext: list = None,
    yaxis_tickvals: list = None,
    texttemplate: str = None,
    textfont: dict = None,
    legend_position: str = None,
    opacity: float = None,
    textposition: str = None,
    template: str = "simple_white",
    hovermode: str = None,
    bargap: float = None,
    bargroupgap: float = None,
    legend_x: float = None,
    legend_y: float = None,
    legend_orientation: str = None,
    hoverlabel_align: str = None,
    xaxis_range: list = None,
    yaxis_range: list = None,
    margin: dict = dict(l=50, r=50, b=50, t=50),
    xgap: int = None,
    ygap: int = None,
    yaxis_domain: list = None
) -> go.Figure:
    """
    Apply consistent styling settings to a Plotly figure.

    Parameters
    ----------
    fig : go.Figure
        The Plotly figure object to style. Must be created before passing to this function.
        Example: fig = go.Figure() or fig = px.line()

    title : str, optional
        Main chart title displayed at the top of the figure.
        Example: title="Sales Performance 2023"

    xaxis_title, yaxis_title : str, optional
        Axis labels for x and y axes.
        Example: xaxis_title="Date", yaxis_title="Revenue ($)"

    legend_title : str, optional
        Title displayed above the legend.
        Example: legend_title="Product Categories"

    xaxis_showgrid, yaxis_showgrid : bool, default=True
        Whether to show grid lines on each axis.
        Example: xaxis_showgrid=False to hide x-axis grid

    xaxis_tickformat, yaxis_tickformat : str, optional
        Format string for axis ticks.
        Examples:
        - '%Y-%m-%d' for dates
        - '.0f' for integers
        - '.2f' for 2 decimal places
        - '$,.0f' for currency without decimals
        - '.1%' for percentages

    template : str, default="plotly_white"
        Plotly template name for consistent styling.
        Options: "plotly", "plotly_white", "plotly_dark", "ggplot2", "seaborn"

    height, width : int, optional
        Figure dimensions in pixels.
        Example: height=600, width=800

    showlegend : bool, default=True
        Whether to display the legend.
        Example: showlegend=False to hide legend

    hovermode : str, default="x unified"
        Hover interaction mode.
        Options:
        - 'x unified': single tooltip for all traces at x position
        - 'x': separate tooltip for each trace at x position
        - 'y': separate tooltip for each trace at y position
        - 'closest': tooltip for closest point
        - False: disable hover tooltips

    bargap : float, default=0.2
        Gap between bars in bar charts (0 to 1).
        Example: bargap=0.3 for 30% gap

    bargroupgap : float, default=0.1
        Gap between bar groups (0 to 1).
        Example: bargroupgap=0.2 for 20% gap

    legend_x, legend_y : float
        Legend position coordinates (0 to 1).
        Example: legend_x=1.02, legend_y=1.0 for outside right

    legend_orientation : str, default='v'
        Legend orientation.
        Options: 'v' (vertical), 'h' (horizontal)

    yaxis_tickprefix, yaxis_ticksuffix : str, default=''
        Prefix/suffix for axis tick labels.
        Examples:
        - yaxis_tickprefix='$'
        - yaxis_ticksuffix='%'

    hovertemplate : str, optional
        Custom hover tooltip template.
        Example: hovertemplate='Date: %{x}<br>Value: %{y:.2f}$<extra></extra>'

    hoverlabel_align : str, default='auto'
        Hover label text alignment.
        Options: 'left', 'right', 'auto'

    xaxis_range, yaxis_range : List, optional
        Custom axis ranges [min, max].
        Example: yaxis_range=[0, 100]

    yaxis_ticktext, xaxis_ticktext : list, optional
        Custom tick labels for axes.
        Example: yaxis_ticktext=['Low', 'Medium', 'High']

    yaxis_tickvals, xaxis_tickvals : list, optional
        Values where custom tick labels should be placed.
        Example: yaxis_tickvals=[0, 50, 100]

    texttemplate : str, optional
        Template for text displayed on the plot.
        Example: texttemplate='%{y:.1f}%'

    textfont : dict, optional
        Font settings for displayed text.
        Example: textfont=dict(size=12, color='red', family='Arial')

    xaxis_tickprefix, xaxis_ticksuffix : str, default=''
        Prefix/suffix for x-axis tick labels.
        Example: xaxis_ticksuffix=' km'

    showmodebar : bool, default=True
        Whether to show the mode bar (navbar).
        Example: showmodebar=False to hide toolbar

    legend_position : str, optional
        Predefined legend position.
        Options: 'top', 'right', 'bottom'
        Example: legend_position='top'

    xaxis_dtick, yaxis_dtick : float, optional
        Step size between axis ticks.
        Examples:
        - xaxis_dtick=1 for integer steps
        - xaxis_dtick=0.5 for half steps
        - xaxis_dtick='M1' for monthly steps in time series

    textposition : str, optional
        Position of text labels relative to data points.
        Options:
        - 'top'
        - 'bottom'
        - 'middle'
        - 'auto'
        - 'top center'
        - 'bottom center'
        - 'middle center'
        Example: textposition='top center'
    margin : dict, optional
        Plot margins in pixels.
        Example: margin=dict(l=50, r=50, t=50, b=50)
        where l=left, r=right, t=top, b=bottom
    opacity : float, optional
        Opacity for figure
    xgap, ygap : int, optional
        xgap and ygap for cells in heatmap

    Returns
    -------
    go.Figure
        The styled Plotly figure
    """
    # Fonts and Colors
    TITLE_FONT_SIZE = 15
    FONT_SIZE = 13
    AXIS_TITLE_FONT_SIZE = 13
    TICK_FONT_SIZE = 13
    LEGEND_TITLE_FONT_SIZE = 13
    FONT_FAMILY = "Noto Sans"
    FONT_COLOR = "rgba(0, 0, 0, 0.7)"
    LINE_COLOR = "rgba(0, 0, 0, 0.4)"
    GRID_COLOR = "rgba(0, 0, 0, 0.1)"
    HOVER_BGCOLOR = "white"
    GRID_WIDTH = 1
    # Layout updates
    layout_updates = {
        'title_text': title,
        'width': width,
        'height': height,
        'legend_title_text': legend_title,
        'showlegend': showlegend,
        'template': template,
        # 'hoverlabel': {'bgcolor': HOVER_BGCOLOR, 'align': hoverlabel_align},
        'hovermode': hovermode,
        'bargap': bargap,
        'bargroupgap': bargroupgap,
        'margin': margin,
        'font': {'size': FONT_SIZE, 'family': FONT_FAMILY, 'color': FONT_COLOR},
        'title_font': {'size': TITLE_FONT_SIZE, 'family': FONT_FAMILY, 'color': FONT_COLOR}
    }
    # Update layout only if there are updates
    layout_updates = {k: v for k, v in layout_updates.items() if v is not None}
    if layout_updates:
        fig.update_layout(**layout_updates)

    # X-axis settings
    xaxis_updates = {
        'linecolor': LINE_COLOR,
        'tickcolor': LINE_COLOR,
        'showgrid': xaxis_showgrid,
        'gridwidth': GRID_WIDTH,
        'gridcolor': GRID_COLOR,
        'dtick': xaxis_dtick,
        'title_text': xaxis_title,
        'tickformat': xaxis_tickformat,
        'range': xaxis_range,
        'tickprefix': xaxis_tickprefix,
        'ticksuffix': xaxis_ticksuffix,
        'ticktext': xaxis_ticktext,
        'tickvals': xaxis_tickvals,
        'title_font': {'size': FONT_SIZE, 'family': FONT_FAMILY, 'color': FONT_COLOR},
        'tickfont': {'size': FONT_SIZE, 'family': FONT_FAMILY, 'color': FONT_COLOR},
    }
    fig.update_xaxes(**{k: v for k, v in xaxis_updates.items() if v is not None})

    # Y-axis settings
    yaxis_updates = {
        'linecolor': LINE_COLOR,
        'tickcolor': LINE_COLOR,
        'showgrid': yaxis_showgrid,
        'gridwidth': GRID_WIDTH,
        'gridcolor': GRID_COLOR,
        'dtick': yaxis_dtick,
        'title_text': yaxis_title,
        'tickformat': yaxis_tickformat,
        'range': yaxis_range,
        'tickprefix': yaxis_tickprefix,
        'ticksuffix': yaxis_ticksuffix,
        'ticktext': yaxis_ticktext,
        'tickvals': yaxis_tickvals,
        'title_font': {'size': FONT_SIZE, 'family': FONT_FAMILY, 'color': FONT_COLOR},
        'tickfont': {'size': FONT_SIZE, 'family': FONT_FAMILY, 'color': FONT_COLOR},
    }
    fig.update_yaxes(**{k: v for k, v in yaxis_updates.items() if v is not None})

    # Legend settings
    legend_updates = {
        'legend_title_font': {'size': FONT_SIZE, 'family': FONT_FAMILY, 'color': FONT_COLOR},
        'legend_font_color': FONT_COLOR,
        'legend_orientation': legend_orientation,
        'legend_x': legend_x,
        'legend_y': legend_y,
    }
    fig.update_layout(**{k: v for k, v in legend_updates.items() if v is not None})

    # Update traces if necessary
    trace_updates = {}
    if hovertemplate is not None:
        trace_updates['hovertemplate'] = hovertemplate
    if textposition is not None:
        trace_updates['textposition'] = textposition
    if texttemplate is not None:
        fig.update_traces(texttemplate=texttemplate)
    if textfont is not None:
        fig.update_layout(textfont=textfont)
    if opacity is not None:
        trace_updates['opacity'] = opacity

    if trace_updates:
        fig.update_traces(**trace_updates)

    if fig.data and fig.data[0].type == 'heatmap':
        fig.update_traces(xgap=xgap, ygap=ygap)
        fig.update_layout(xaxis_showgrid=False, yaxis_showgrid=False, coloraxis_colorbar_title_text=None)
    if fig.data and fig.data[0].type == 'scattergl':
        fig.update_traces(marker=dict(
                line=dict(color='white', width=0.5)))
    for trace in fig.data:
        if trace.hovertemplate:
            trace.hovertemplate = re.sub(r'\s*=\s*', ' = ', trace.hovertemplate)
        # trace.hovertemplate = trace.hovertemplate.replace('{x}', '{x:.2f}')
        # trace.hovertemplate = trace.hovertemplate.replace('{y}', '{y:.2f}')
    if legend_position == 'top':
        fig.update_layout(
            yaxis = dict(
                domain=yaxis_domain if yaxis_domain else [0, 0.92]
            )
            , legend = dict(
                orientation="h"  # Горизонтальное расположение
                , yanchor="top"    # Привязка к верхней части
                , y=1.05         # Положение по вертикали (отрицательное значение переместит вниз)
                , xanchor="center" # Привязка к центру
                , x=0.5              # Центрирование по горизонтали
            )
        )
    elif legend_position == 'right':
        fig.update_layout(
                yaxis = dict(
                    domain=[0, 1]
                )
                , legend = dict(
                    xanchor=None
                    , yanchor=None
                    , orientation="v"  # Горизонтальное расположение
                    , y=1         # Положение по вертикали (отрицательное значение переместит вниз)
                    , x=None              # Центрирование по горизонтали
                )
        )
    elif legend_position == 'bottom':
        fig.update_layout(
            legend = dict(
                orientation="h"  # Горизонтальное расположение
                , yanchor="bottom"    # Привязка к верхней части
                , y=-0.15         # Положение по вертикали (отрицательное значение переместит вниз)
                , xanchor="center" # Привязка к центру
                , x=0.5              # Центрирование по горизонтали
            )
        )
    return fig

def _create_base_fig_for_bar_line_area(
    df: pd.DataFrame
    , config: dict
    , kwargs: dict
    , graph_type: str = 'bar'
    ) -> go.Figure:
    """
    Creates a figure for bar, line or area function using the Plotly Express library.

    Parameters
    ----------
    df : pd.DataFrame, optional
        DataFrame containing the data to be plotted
    config: dict
        Settings that are not passed to functions for creating Plotly graphs.
    kwargs: dict
        Settings that are passed to functions for creating Plotly graphs.
    graph_type: str
        Type of graphics

    Returns
    -------
    go.Figure
        The created chart
    """
    # Check for 'resample_freq' in config if aggregation mode is 'resample'
    if config.get('agg_mode') == 'resample' and 'resample_freq' not in config:
        raise ValueError("For resample mode resample_freq must be define")

    # Check for aggregation function if aggregation mode is set
    if config.get('agg_mode') and 'agg_func' not in config:
        raise ValueError('resample or groupby mode agg_func must be defined')

    # Ensure the first argument or 'data_frame' in kwargs is a DataFrame
    if not isinstance(df, pd.DataFrame):
        raise ValueError('data_frame must be pandas DataFrame')

    # Assign the DataFrame to variable df
    if df is None:
        raise ValueError('data_frame must be pandas DataFrame and defined')

    # Check data types of x and y in 'groupby' mode
    if config.get('agg_mode') == 'groupby':
        if pd.api.types.is_datetime64_any_dtype(df[kwargs['x']]) or pd.api.types.is_datetime64_any_dtype(df[kwargs['y']]):
            raise ValueError("For resample mode groupby x and y must not be datetime")

    # Ensure 'x' and 'y' keys are present in kwargs
    if 'x' not in kwargs or 'y' not in kwargs:
        raise ValueError('x and y must be defined')
    # Retrieve mode from config
    df = df.copy()
    agg_mode = config.get('agg_mode')
    norm_by = config.get('norm_by')
    if graph_type in ['line', 'area']:
        if kwargs.get('color'):
            kwargs.setdefault('color_discrete_sequence', colorway_for_line)
        kwargs.setdefault('line_shape', 'spline')
    if graph_type == 'bar':
        kwargs.setdefault('barmode', 'group')
    if 'color' in kwargs and kwargs['color'] is None:
        kwargs.pop('color')
    # Determine numeric and categorical columns
    def _determine_numeric_and_categorical_columns(config: dict, kwargs: dict):
        color = [kwargs['color']] if kwargs.get('color') else []
        facet_col = [kwargs['facet_col']] if kwargs.get('facet_col') else []
        facet_row = [kwargs['facet_row']] if kwargs.get('facet_row') else []
        animation_frame = [kwargs['animation_frame']] if kwargs.get('animation_frame') else []
        if 'agg_column' in config:
            num_column = config['agg_column']
            if kwargs['x'] == num_column:
                cat_column_axis = kwargs['y']
            else:
                cat_column_axis = kwargs['x']
        else:
            if pd.api.types.is_numeric_dtype(df[kwargs['x']]):
                num_column = kwargs['x']
                cat_column_axis = kwargs['y']
            else:
                num_column = kwargs['y']
                cat_column_axis = kwargs['x']
        cat_columns = facet_col + facet_row + [cat_column_axis] + color + animation_frame
        config['cat_column_axis'] = cat_column_axis
        config['num_column'] = num_column
        config['cat_columns'] = cat_columns
        return num_column, cat_columns

    # Function to create a combined filter mask
    def _create_filter_mask(df: pd.DataFrame, config: dict, kwargs: dict):
        """Create a combined filter mask based on top_n_trim_color and top_n_trim_axis"""
        mask = None
        num_column  = config.get('num_column')
        cat_columns = config.get('cat_columns')
        mode_top_or_bottom = config['trim_top_or_bottom']
        agg_func_for_top_n = config.get('agg_func_for_top_n')
        top_n_trim_x = config.get('top_n_trim_x')
        top_n_trim_color = config.get('top_n_trim_color')
        top_n_trim_y = config.get('top_n_trim_y')
        top_n_trim_facet_col = config.get('top_n_trim_facet_col')
        top_n_trim_facet_row = config.get('top_n_trim_facet_row')
        top_n_trim_facet_animation_frame = config.get('top_n_trim_facet_animation_frame')
        if mode_top_or_bottom == 'top':
            ascending = False
        elif mode_top_or_bottom == 'bottom':
            ascending = True
        else:
            raise ValueError('unknown mask_func')

        # Filter by color
        if top_n_trim_color:
            if kwargs.get('color') is None:
                raise ValueError('For top_n_trim_color color must be defined')
            top_color = (
                df.groupby(kwargs['color'], observed=False)[num_column]
                .agg(agg_func_for_top_n)
                .sort_values(ascending=ascending)[:top_n_trim_color]
                .index
            )
            color_mask = df[kwargs['color']].isin(top_color)
            mask = mask & color_mask if mask is not None else color_mask

        # Filter by x axis
        if top_n_trim_x:
            if kwargs.get('x') is None:
                raise ValueError('For top_n_trim_x x must be defined')
            top_x = (
                df.groupby(kwargs['x'], observed=False)[num_column]
                .agg(agg_func_for_top_n)
                .sort_values(ascending=ascending)[:top_n_trim_x]
                .index
            )
            x_mask = df[kwargs['x']].isin(top_x)
            mask = mask & x_mask if mask is not None else x_mask

        # Filter by y axis
        if top_n_trim_y:
            if kwargs.get('y') is None:
                raise ValueError('For top_n_trim_y y must be defined')
            top_y = (
                df.groupby(kwargs['y'], observed=False)[num_column]
                .agg(agg_func_for_top_n)
                .sort_values(ascending=ascending)[:top_n_trim_y]
                .index
            )
            y_mask = df[kwargs['y']].isin(top_y)
            mask = mask & y_mask if mask is not None else y_mask

        # Filter by facet_col
        if top_n_trim_facet_col:
            if kwargs.get('facet_col') is None:
                raise ValueError('For top_n_trim_facet_col facet_col must be defined')
            top_facet_col = (
                df.groupby(kwargs['facet_col'], observed=False)[num_column]
                .agg(agg_func_for_top_n)
                .sort_values(ascending=ascending)[:top_n_trim_facet_col]
                .index
            )
            facet_col_mask = df[kwargs['facet_col']].isin(top_facet_col)
            mask = mask & facet_col_mask if mask is not None else facet_col_mask

        # Filter by facet_row
        if top_n_trim_facet_row:
            if kwargs.get('facet_row') is None:
                raise ValueError('For top_n_trim_facet_row facet_row must be defined')
            top_facet_row = (
                df.groupby(kwargs['facet_row'], observed=False)[num_column]
                .agg(agg_func_for_top_n)
                .sort_values(ascending=ascending)[:top_n_trim_facet_row]
                .index
            )
            facet_row_mask = df[kwargs['facet_row']].isin(top_facet_row)
            mask = mask & facet_row_mask if mask is not None else facet_row_mask

        # Filter by facet_col
        if top_n_trim_facet_animation_frame:
            if kwargs.get('animation_frame') is None:
                raise ValueError('For top_n_trim_facet_animation_frame animation_frame must be defined')
            top_animation_frame = (
                df.groupby(kwargs['animation_frame'], observed=False)[num_column]
                .agg(agg_func_for_top_n)
                .sort_values(ascending=ascending)[:top_n_trim_facet_animation_frame]
                .index
            )
            animation_frame_mask = df[kwargs['animation_frame']].isin(top_animation_frame)
            mask = mask & animation_frame_mask if mask is not None else animation_frame_mask

        return mask

    def _normalize_data(func_df, config: dict, kwargs: dict):
        norm_by = config.get('norm_by')
        color = [kwargs['color']] if kwargs.get('color') else []
        cat_column_axis = config['cat_column_axis']
        facet_col = [kwargs['facet_col']] if kwargs.get('facet_col') else []
        facet_row = [kwargs['facet_row']] if kwargs.get('facet_row') else []
        animation_frame = [kwargs['animation_frame']] if kwargs.get('animation_frame') else []
        if norm_by == 'all':
            columns_for_groupby_share = facet_col + facet_row + animation_frame
            if columns_for_groupby_share:
                func_df['all_sum'] = func_df.groupby(columns_for_groupby_share, observed=False)['num_in_prepare_df'].transform('sum')
            else:
                func_df['all_sum'] = func_df['num_in_prepare_df'].sum()
            func_df['origin_num'] = func_df['num_in_prepare_df']
            func_df['num_in_prepare_df'] = func_df['num_in_prepare_df'] / func_df['all_sum']
            func_df = func_df.drop('all_sum', axis=1)
        if norm_by == cat_column_axis:
            columns_for_groupby_share = [cat_column_axis] + facet_col + facet_row + animation_frame
            func_df['category_sum'] = func_df.groupby(columns_for_groupby_share, observed=False)['num_in_prepare_df'].transform('sum')
            func_df['origin_num'] = func_df['num_in_prepare_df']
            func_df['num_in_prepare_df'] = func_df['num_in_prepare_df'] / func_df['category_sum']
            func_df = func_df.drop('category_sum', axis=1)
        if color and norm_by == color[0]:
            columns_for_groupby_share = [color[0]] + facet_col + facet_row + animation_frame
            func_df['color_sum'] = func_df.groupby(columns_for_groupby_share, observed=False)['num_in_prepare_df'].transform('sum')
            func_df['origin_num'] = func_df['num_in_prepare_df']
            func_df['num_in_prepare_df'] = func_df['num_in_prepare_df'] / func_df['color_sum']
            func_df = func_df.drop('color_sum', axis=1)
        return func_df

    def _sort_data(func_df, config: dict, kwargs: dict):
        num_column = config.get('num_column')
        sort_axis = config.get('sort_axis')
        sort_color = config.get('sort_color')
        sort_facet_row = config.get('sort_facet_row')
        sort_facet_col = config.get('sort_facet_col')
        sort_animation_frame = config.get('sort_animation_frame')
        observed_for_groupby = config.get('observed_for_groupby')
        cat_column_axis = config['cat_column_axis']
        color = kwargs['color'] if kwargs.get('color') else None
        facet_col = kwargs['facet_col'] if kwargs.get('facet_col') else None
        facet_row = kwargs['facet_row'] if kwargs.get('facet_row') else None
        animation_frame = kwargs['animation_frame'] if kwargs.get('animation_frame') else None
        norm_by = config.get('norm_by')
        agg_func = config.get('agg_func')
        if sort_axis or sort_color or sort_facet_row or sort_facet_col + sort_animation_frame:
            num_column_for_sort = 'origin_num' if norm_by else 'num_in_prepare_df'
            # Determine sorting order
            if config['trim_top_or_bottom'] == 'top':
                ascending_for_axis = False if kwargs['y'] == num_column else True
            elif config['trim_top_or_bottom'] == 'bottom':
                ascending_for_axis = True if kwargs['y'] == num_column else False
            else:
                raise ValueError('Unknown trim_top_or_bottom. Must be "top" or "bottom"')
            ascending_for_color = False
            ascending_for_facet_col = False
            ascending_for_facet_row = False
            ascending_for_animation_frame = False
            columns_for_sort = []
            ascending = []
            func_for_sort = 'sum' if agg_func in ['count', 'nunique'] else agg_func
            if sort_animation_frame == True and animation_frame:
                ascending.append(ascending_for_animation_frame)
                func_df['sum_for_sort_animation_frame'] = func_df.groupby(animation_frame, observed=observed_for_groupby)[num_column_for_sort].transform(func_for_sort)
                columns_for_sort.append('sum_for_sort_animation_frame')
                if kwargs.get('category_orders') and animation_frame in kwargs['category_orders']:
                    kwargs['category_orders'].pop(animation_frame)
            if sort_facet_col == True and facet_col:
                ascending.append(ascending_for_facet_col)
                func_df['sum_for_sort_facet_col'] = func_df.groupby(facet_col, observed=observed_for_groupby)[num_column_for_sort].transform(func_for_sort)
                columns_for_sort.append('sum_for_sort_facet_col')
                if kwargs.get('category_orders') and facet_col in kwargs['category_orders']:
                    kwargs['category_orders'].pop(facet_col)
            if sort_facet_row == True and facet_row:
                ascending.append(ascending_for_facet_row)
                func_df['sum_for_sort_facet_row'] = func_df.groupby(facet_row, observed=observed_for_groupby)[num_column_for_sort].transform(func_for_sort)
                columns_for_sort.append('sum_for_sort_facet_row')
                if kwargs.get('category_orders') and facet_row in kwargs['category_orders']:
                    kwargs['category_orders'].pop(facet_row)
            if sort_axis == True and cat_column_axis:
                ascending.append(ascending_for_axis)
                func_df['sum_for_sort_axis'] = func_df.groupby(cat_column_axis, observed=observed_for_groupby)[num_column_for_sort].transform(func_for_sort)
                columns_for_sort.append('sum_for_sort_axis')
                if kwargs.get('category_orders') and cat_column_axis in kwargs['category_orders']:
                    kwargs['category_orders'].pop(cat_column_axis)
            if sort_color == True and color:
                ascending.append(ascending_for_color)
                func_df['sum_for_sort_color'] = func_df.groupby(color, observed=observed_for_groupby)[num_column_for_sort].transform(func_for_sort)
                columns_for_sort.append('sum_for_sort_color')
                if kwargs.get('category_orders') and color in kwargs['category_orders']:
                    kwargs['category_orders'].pop(color)
            func_df = func_df.sort_values(columns_for_sort, ascending=ascending)
            func_df = func_df.drop(columns_for_sort, axis=1)
        return func_df

    # Function to prepare the DataFrame for plotting
    def _prepare_df(df: pd.DataFrame, config: dict, kwargs: dict):
        # Check for 'agg_column' in config
        if 'agg_column' not in config and pd.api.types.is_numeric_dtype(df[kwargs['x']]) and pd.api.types.is_numeric_dtype(df[kwargs['y']]):
            raise ValueError('If x and y are numeric, agg_column must be defined')
        norm_by = config.get('norm_by')
        agg_func = config.get('agg_func')
        observed_for_groupby = config.get('observed_for_groupby')
        num_column, cat_columns = _determine_numeric_and_categorical_columns(config, kwargs)
        min_group_size = config.get('min_group_size')
        # Fiter by min group size
        if min_group_size:
            if cat_columns is None:
                raise ValueError('Error fiter min_group_size, cat_columns is None')
            df = df.groupby(cat_columns, observed=observed_for_groupby).filter(lambda x: len(x) >= min_group_size)
        # Create filter mask
        mask = _create_filter_mask(df, config, kwargs)

        # Apply mask to DataFrame
        func_df = df[mask] if mask is not None else df
        # display(func_df[func_df.order_status == 'unavailable'].head())
        # Aggregate data
        # print(cat_columns)
        if pd.api.types.is_numeric_dtype(df[num_column]):
            func_df = (func_df[[*cat_columns, num_column]]
                    .groupby(cat_columns, observed=observed_for_groupby)
                    .agg(num_in_prepare_df=(num_column, agg_func)
                            , count_for_subplots=(num_column, 'count')
                            , margin_of_error = (num_column, 'sem'))
                    .reset_index())
            func_df['margin_of_error'] = 1.96 * func_df['margin_of_error']
        else:
            func_df = (func_df[[*cat_columns, num_column]]
                    .groupby(cat_columns, observed=observed_for_groupby)
                    .agg(num_in_prepare_df=(num_column, agg_func)
                            , count_for_subplots=(num_column, 'count'))
                    .reset_index())
            func_df['margin_of_error'] = None
        # display(func_df[kwargs['animation_frame']].unique())
        if norm_by:
            func_df = _normalize_data(func_df, config, kwargs)
        # Sort data by axis if specified
        # display(func_df.head())
        # чтобы использовать одновременно и facet и animation_frame нам нужно чтобы в датафрейме были все комбинации занчений срезов
        # Иначе будут баги, нужно чтобы в plotly передались все возможные комбинации, чтоыб он их поставил на свои места, пусть там и будет None
        if (kwargs.get('facet_col') or kwargs.get('facet_row')) and kwargs.get('animation_frame'):
            all_combinations = pd.MultiIndex.from_product([func_df[col].unique().tolist() for col in cat_columns], names=cat_columns)
            all_combinations = all_combinations.to_list()
            temp_df = pd.DataFrame(all_combinations, columns=cat_columns)
            func_df = temp_df.merge(func_df, on=cat_columns, how='left')
        # print()
        func_df = _sort_data(func_df, config, kwargs)
        # Format the 'count' column
        func_df['count_for_show_group_size'] = func_df['count_for_subplots'].apply(lambda x: f'= {x}' if x <= 1e3 else 'больше 1000' if x > 1e3 else 0)
        func_df[cat_columns] = func_df[cat_columns].astype('str')
        return func_df.rename(columns={'num_in_prepare_df': num_column})

    def _create_top_and_bottom_fig(fig, df, config: dict, kwargs: dict):
        fig_subplots = make_subplots(rows=1, cols=2, horizontal_spacing=0.15)
        fig_subplots.add_trace(fig.data[0], row=1, col=1)
        config['trim_top_or_bottom'] = 'bottom'
        df_for_fig_bottom = _prepare_df(df, config, kwargs)
        custom_data = [df_for_fig_bottom['count_for_show_group_size']]
        # display(df_for_fig.head())
        if norm_by:
            custom_data += [df_for_fig_bottom['origin_num']]
            # if kwargs['labels'] is not None:
            #     num_column_for_hover_label = kwargs['labels'][num_column_for_hover[0]]
            #     kwargs['labels'][num_column_for_hover] = 'Доля'
            is_num_integer = False
        else:
            is_num_integer = pd.api.types.is_integer_dtype(df_for_fig_bottom[config['num_column']])
        kwargs['custom_data'] = custom_data
        fig_bottom = px.bar(df_for_fig_bottom, **kwargs)
        fig.update_traces(error_x_color='rgba(50, 50, 50, 0.7)')
        fig_subplots.add_trace(fig_bottom.data[0], row=1, col=2)
        fig_subplots.update_layout(title_text=kwargs.get('title'))
        fig_subplots.update_xaxes(title_text=kwargs.get('labels')[kwargs['x']], row=1, col=1)
        fig_subplots.update_xaxes(title_text=kwargs.get('labels')[kwargs['x']], row=1, col=2)
        fig_subplots.update_yaxes(title_text=kwargs.get('labels')[kwargs['y']], row=1, col=1)
        fig = fig_subplots
        return fig

    def _create_base_fig_and_countplot(fig, df_for_fig, config: dict, kwargs: dict):
        kwargs_for_count = kwargs.copy()
        num_column = config.get('num_column')
        kwargs_for_count['error_x'] = None
        kwargs_for_count['error_y'] = None
        kwargs_for_count['hover_data'].update({'margin_of_error': False, num_column: False})
        if 'count_for_subplots' not in kwargs_for_count['labels']:
            kwargs_for_count['labels']['count_for_subplots'] = 'Количество'
        if config['num_column'] == kwargs['x']:
            kwargs_for_count['x'] = 'count_for_subplots'
            count_xaxis_title = kwargs_for_count['labels']['count_for_subplots']
            count_yaxis_title = None
            shared_yaxes = True
            horizontal_spacing = 0.05
        else:
            kwargs_for_count['y'] = 'count_for_subplots'
            count_yaxis_title = kwargs_for_count['labels']['count_for_subplots']
            count_xaxis_title = kwargs_for_count.get('labels')[kwargs_for_count['x']]
            shared_yaxes = False
            horizontal_spacing = None
        fig_subplots = make_subplots(rows=1, cols=2, shared_yaxes=shared_yaxes, horizontal_spacing=horizontal_spacing)
        fig_subplots.add_trace(fig.data[0], row=1, col=1)
        fig_subplots.update_layout(title_text=kwargs.get('title'))
        fig_subplots.update_xaxes(title_text=kwargs.get('labels').get(kwargs['x']), row=1, col=1)
        fig_subplots.update_yaxes(title_text=kwargs.get('labels').get(kwargs['y']), row=1, col=1)
        fig_count = px.bar(df_for_fig, **kwargs_for_count)
        fig_subplots.add_trace(fig_count.data[0], row=1, col=2)
        fig_subplots.update_xaxes(title_text=count_xaxis_title, row=1, col=2)
        fig_subplots.update_yaxes(title_text=count_yaxis_title, row=1, col=2)
        fig = fig_subplots
        return fig

    def _add_boxplot(fig, df, config: dict, kwargs: dict):
        upper_quantile = config.get('upper_quantile_for_box')
        lower_quantile = config.get('lower_quantile_for_box')
        fig_subplots = make_subplots(rows=1, cols=2, shared_yaxes=True)
        if config.get('agg_column') == kwargs['x'] or pd.api.types.is_numeric_dtype(df[kwargs['x']]):
            orientation = 'h'
        else:
            orientation = 'v'
        if upper_quantile or lower_quantile:
            # Set default upper quantile if not provided
            if upper_quantile is None:
                upper_quantile= 1
            # Set default lower quantile if not provided
            if lower_quantile is None:
                lower_quantile = 0

            # Prepare columns for grouping
            columns_for_groupby_for_range = config['cat_column_axis']
            # If color is specified, include it in the grouping
            if kwargs.get('color'):
                columns_for_groupby_for_range.append(kwargs['color'])

            # Apply the trim_by_quantiles function to the grouped data
            temp_for_range = df.groupby(columns_for_groupby_for_range, observed=False)[config['num_column']].quantile([lower_quantile, upper_quantile]).unstack()
            lower_range = temp_for_range.iloc[:, 0].min()
            upper_range = temp_for_range.iloc[:, 1].max()
            lower_range -= (upper_range - lower_range) * 0.05
        if orientation == 'h':
            categories = fig.data[0].y
            hover_data[kwargs['y']] = None
        else:
            categories = fig.data[0].x
            hover_data[kwargs['x']] = None
        # print(categories)
        # category_orders_for_box = {config['cat_column_axis']: categories}
        df[config['cat_column_axis']] = df[config['cat_column_axis']].astype(str).astype('category')
        fig_box = box(df
                        , x=kwargs['x']
                        , y=kwargs['y']
                        , labels=kwargs.get('labels')
                        , orientation=orientation
                        , hover_data=hover_data
                        # , category_orders = category_orders_for_box
                    )
        fig_subplots.add_trace(fig.data[0], row=1, col=1)
        fig_subplots.add_trace(fig_box.data[0], row=1, col=2)
        fig_subplots.update_layout(title_text=kwargs.get('title'))
        fig_subplots.update_xaxes(title_text=kwargs.get('labels')[kwargs['x']], row=1, col=1)
        fig_subplots.update_xaxes(title_text=kwargs.get('labels')[kwargs['x']], row=1, col=2)
        fig_subplots.update_yaxes(title_text=kwargs.get('labels')[kwargs['y']], row=1, col=1)
        fig = fig_subplots
        # print(fig)
        if upper_quantile or lower_quantile:
            if orientation == 'h':
                fig.update_layout(xaxis2_range=[lower_range, upper_range])
            else:
                fig.update_layout(yaxis2_range=[lower_range, upper_range])
        return fig

    def _update_fig(fig, config: dict, kwargs: dict):
        # Set x-axis format and figure dimensions
        fig_update_config = dict()
        is_x_datetime = pd.api.types.is_datetime64_any_dtype(df[kwargs['x']])
        is_x_numeric = pd.api.types.is_numeric_dtype(df[kwargs['x']])
        is_x_integer = pd.api.types.is_integer_dtype(df[kwargs['x']])
        is_y_numeric = pd.api.types.is_numeric_dtype(df[kwargs['y']])
        is_y_integer = pd.api.types.is_integer_dtype(df[kwargs['y']])
        if graph_type == 'bar':
            if not is_x_numeric:
                fig_update_config['xaxis_showgrid'] = False
            if not is_y_numeric:
                fig_update_config['yaxis_showgrid'] = False
            if config.get('agg_column') == kwargs['x']:
                fig_update_config['xaxis_showgrid'] = True
                fig_update_config['yaxis_showgrid'] = False
            if config.get('agg_column') == kwargs['y']:
                fig_update_config['xaxis_showgrid'] = False
                fig_update_config['yaxis_showgrid'] = True
        if pd.api.types.is_datetime64_any_dtype(df[kwargs['x']]):
            # fig_update_config['xaxis_tickformat'] = "%b'%y"
            if kwargs.get('width') is None:
                fig_update_config['width'] = 1000
            if kwargs.get('height') is None:
                fig_update_config['height'] = 450
        else:
            if kwargs.get('width') is None:
                fig_update_config['width'] = 600
                if kwargs.get('color'):
                    fig_update_config['width'] = 800
                if config.get('show_box'):
                    fig_update_config['width'] = 1000
                if config.get('top_and_bottom'):
                    fig_update_config['width'] = 900
                if config.get('show_count'):
                    fig_update_config['width'] = 900
            else:
                fig_update_config['width'] = kwargs.get('width')
            if not kwargs.get('height'):
                fig_update_config['height'] = 400
        if kwargs.get('color'):
            fig_update_config['legend_position'] = 'top'
            if graph_type in ['line', 'area']:
                fig_update_config['opacity'] = 0.7
            if config.get('show_legend_title') == False:
                fig_update_config['legend_title'] = ''
        if config['update_layout'] == True:
            fig = fig_update(fig, **fig_update_config)
        return fig

    # Handle data in 'resample' mode
    if agg_mode == 'resample':
        if 'agg_func' not in config:
            raise ValueError('agg_func must be defined')
        if 'resample_freq' not in config:
            raise ValueError('For resample agg_mode resample_freq must be defined')
        if not pd.api.types.is_datetime64_any_dtype(df[kwargs['x']]):
            raise ValueError('x must be datetime type')
        agg_func = config.get('agg_func')
        agg_func_for_top_n = config.get('agg_func_for_top_n')
        # Define columns for aggregation
        columns = [kwargs['x'], kwargs['y']]

        # Filter by color if specified
        if config.get('top_n_trim_color'):
            if 'color' not in kwargs:
                raise ValueError('For top_n_trim_color color must be defined')
            if config.get('trim_top_or_bottom') == 'bottom':
                top_color = df.groupby(kwargs['color'], observed=False)[kwargs['y']].agg(agg_func_for_top_n).nsmallest(config.get('top_n_trim_color')).index.to_list()
            else:
                top_color = df.groupby(kwargs['color'], observed=False)[kwargs['y']].agg(agg_func_for_top_n).nlargest(config.get('top_n_trim_color')).index.to_list()
            if graph_type in ['line', 'area']:
                kwargs.setdefault('category_orders', {kwargs.get('color'): top_color})
        else:
            if kwargs.get('color'):
                top_color = df.groupby(kwargs['color'], observed=False)[kwargs['y']].agg(agg_func_for_top_n).nlargest(10).index.to_list()
                if graph_type in ['line', 'area']:
                    kwargs.setdefault('category_orders', {kwargs.get('color'): top_color})
        # Create DataFrame for the figure
        if 'color' in kwargs:
            columns.append(kwargs['color'])
            if config.get('top_n_trim_color'):
                if pd.api.types.is_categorical_dtype(df[kwargs['color']]):
                    df[kwargs['color']] = df[kwargs['color']].cat.set_categories(top_color)
                df = df[columns][df[kwargs['color']].isin(top_color)]
            else:
                df = df[columns]
            df_for_fig = (df.groupby([pd.Grouper(key=kwargs['x'], freq=config['resample_freq']), kwargs['color']], observed=config['observed_for_groupby'])
                            .agg(num_in_resample_mode = (kwargs['y'], agg_func)
                                , count_for_show_group_size = (kwargs['y'], 'count')
                                # , margin_of_error = (kwargs['y'], 'sem')
                                )
                            .reset_index()
                            .rename(columns={'num_in_resample_mode': kwargs['y']})
            )
            # df_for_fig['margin_of_error'] = 1.96 * df_for_fig['margin_of_error']
            # Since when grouping by two or more fields, missing dates in a variable of type datetime are not preserved, it is necessary to restore all missing dates and fill them with zeros.
            full_index = pd.MultiIndex.from_product([pd.date_range(df_for_fig[kwargs['x']].min(), df_for_fig[kwargs['x']].max(), freq=config['resample_freq']), df_for_fig[kwargs['color']].unique()], names=[kwargs['x'], kwargs['color']])
            df_for_fig = df_for_fig.set_index([kwargs['x'], kwargs['color']]).reindex(full_index, fill_value=0).reset_index()
        else:
            df_for_fig = (df[columns].set_index(kwargs['x']).resample(config['resample_freq'])
                            .agg(num_in_resample_mode = (kwargs['y'], agg_func)
                                , count_for_show_group_size = (kwargs['y'], 'count')
                                # , margin_of_error = (kwargs['y'], 'sem')
                                )
                            .reset_index()
                            .rename(columns={'num_in_resample_mode': kwargs['y']})
            )
            # df_for_fig['margin_of_error'] = 1.96 * df_for_fig['margin_of_error']
        custom_data = [df_for_fig['count_for_show_group_size']]
        kwargs['custom_data'] = custom_data
        # Create the figure using Plotly Express
        figure_creators = {
            'bar': px.bar,
            'line': px.line,
            'area': px.area
        }
        if kwargs.get('hover_data') is None and not pd.api.types.is_integer_dtype(df_for_fig[kwargs['y']]):
            kwargs['hover_data'] = {kwargs['y']: ':.2f'}
        fig = figure_creators[graph_type](df_for_fig, **kwargs)
        if config.get('show_group_size') == True:
            for trace in fig.data:
                trace.hovertemplate += '<br>Размер группы = %{customdata[0]}'
    # Handle data in 'groupby' mode
    elif agg_mode == 'groupby':
        df_for_fig = _prepare_df(df, config, kwargs)
        custom_data = [df_for_fig['count_for_show_group_size']]
        # display(df_for_fig.head())
        if norm_by:
            custom_data += [df_for_fig['origin_num']]
            # if kwargs['labels'] is not None:
            #     num_column_for_hover_label = kwargs['labels'][config['num_column']]
            #     kwargs['labels'][num_column_for_hover] = 'Доля'
            is_num_integer = False
        else:
            is_num_integer = pd.api.types.is_integer_dtype(df_for_fig[config['num_column']])
        figure_creators = {
            'bar': px.bar,
            'line': px.line,
            'area': px.area
        }
        hover_data = dict()
        if kwargs.get('hover_data') is None and not is_num_integer:
            hover_data[config['num_column']] =  ':.2f'
        kwargs['custom_data'] = custom_data
        if graph_type == 'bar' and config.get('show_ci') == True:
            if config['num_column'] == kwargs['x']:
                kwargs['error_x'] = 'margin_of_error'
            else:
                kwargs['error_y'] = 'margin_of_error'
            hover_data['margin_of_error'] =  ':.2f'
            if kwargs.get('labels') is not None:
                kwargs['labels']['margin_of_error'] = 'Предельная ошибка'
        kwargs['hover_data'] = hover_data
        fig = figure_creators[graph_type](df_for_fig, **kwargs)
        fig.update_traces(error_x_color='rgba(50, 50, 50, 0.7)')
        if graph_type == 'bar' and config['top_and_bottom']:
            fig = _create_top_and_bottom_fig(fig, df, config, kwargs)

        if graph_type == 'bar' and config['show_count']:
            fig  = _create_base_fig_and_countplot(fig, df_for_fig, config, kwargs)

        if graph_type == 'bar' and config['show_box']:
            fig = _add_boxplot(fig, df, config, kwargs)
        if config.get('show_group_size') == True:
            for trace in fig.data:
                trace.hovertemplate += '<br>Размер группы = %{customdata[0]}'
    # Handle data in normal mode
    else:
        num_column_for_subplots = None
        if not pd.api.types.is_datetime64_any_dtype(df[kwargs['x']]):
            orientation = kwargs.get('orientation')
            if kwargs.get('orientation') is None:
                orientation='v'
            if pd.api.types.is_numeric_dtype(df[kwargs['y']]) and orientation == 'v':
                if config.get('sort_axis'):
                    df[kwargs['x']] = df[kwargs['x']].astype(str)
                    num_for_sort = kwargs['y']
                    # num_column_for_subplots = kwargs['y']
                    ascending_for_sort = False
                    df = df.sort_values(num_for_sort, ascending=ascending_for_sort)
            else:
                if config.get('sort_axis'):
                    df[kwargs['y']] = df[kwargs['y']].astype(str)
                    num_for_sort = kwargs['x']
                    # num_column_for_subplots = kwargs['x']
                    ascending_for_sort = True
                    df = df.sort_values(num_for_sort, ascending=ascending_for_sort)
        if kwargs.get('color') is not None:
            df[kwargs['color']] = df[kwargs['color']].astype('str')
            if kwargs.get('category_orders') is not None and kwargs.get('category_orders').get(kwargs['color']) is not None:
                kwargs['category_orders'][kwargs['color']] = map(str, kwargs['category_orders'][kwargs['color']])
        # Create the figure using Plotly Express
        figure_creators = {
            'bar': px.bar,
            'line': px.line,
            'area': px.area
        }
        if pd.api.types.is_datetime64_any_dtype(df[kwargs['x']]):
            kwargs['hover_data'] = {kwargs['y']: ':.2f'}
        elif kwargs.get('hover_data') is not None and not pd.api.types.is_integer_dtype(df_for_fig[num_for_sort]):
            kwargs['hover_data'] = {num_for_sort: ':.2f'}
        fig = figure_creators[graph_type](df, **kwargs)
        if graph_type == 'bar' and config['show_count']:
            fig  = _create_base_fig_and_countplot(fig, df)

    fig = _update_fig(fig, config, kwargs)

    return fig


def bar(
    data_frame: pd.DataFrame,
    agg_mode: str = None,
    agg_func: str = None,
    agg_column: str = None,
    resample_freq: str = None,
    norm_by: str = None,
    top_n_trim_x: int = None,
    top_n_trim_y: int = None,
    top_n_trim_color: int = None,
    top_n_trim_facet_col: int = None,
    top_n_trim_facet_row: int = None,
    top_n_trim_facet_animation_frame: int = None,
    sort_axis: bool = True,
    sort_color: bool = True,
    sort_facet_col: bool = True,
    sort_facet_row: bool = True,
    sort_animation_frame: bool = True,
    show_group_size: bool = False,
    min_group_size: int = None,
    decimal_places: int = 2,
    update_layout: bool = True,
    show_box: bool = False,
    show_count: bool = False,
    lower_quantile_for_box: float = None,
    upper_quantile_for_box: float = None,
    top_and_bottom: bool = False,
    show_ci: bool = False,
    trim_top_or_bottom: str = 'top',
    show_legend_title: bool = False,
    observed_for_groupby: bool = False,
    agg_func_for_top_n: bool = 'count',
    **kwargs
) -> go.Figure:
    """
    Creates a bar chart using the Plotly Express library. This function is a wrapper around Plotly Express bar and accepts all the same parameters, allowing for additional customization and functionality.

    Parameters
    ----------
    data_frame : pd.DataFrame, optional
        DataFrame containing the data to be plotted
    agg_mode : str, optional
        Aggregation mode. Options:
        - 'groupby': Group by categorical columns
        - 'resample': Resample time series data
        - None: No aggregation (default)
    agg_func : str, optional
        Aggregation function. Options: 'mean', 'median', 'sum', 'count', 'nunique'
    agg_column : str, optional
         Column to aggregate
    resample_freq : str, optional
        Resample frequency for resample. Options: 'ME', 'W', D' and others
    norm_by: str, optional
        The name of the column to normalize by. If set to 'all', normalization will be performed based on the sum of all values in the dataset.
    top_n_trim_x : int, optional
        Ontly for aggregation mode. The number of top categories x axis to include in the chart. For top using num column and agg_func.
    top_n_trim_y : int, optional
        Ontly for aggregation mode. The number of top categories y axis to include in the chart. For top using num column and agg_func
    top_n_trim_color : int, optional
        Ontly for aggregation mode. The number of top categories legend to include in the chart. For top using num column and agg_func
    top_n_trim_facet_col : int, optional
        Ontly for aggregation mode. The number of top categories in facet_col to include in the chart. For top using num column and agg_func
    top_n_trim_facet_row : int, optional
        Ontly for aggregation mode. The number of top categories in facet_row to include in the chart. For top using num column and agg_func
    top_n_trim_facet_animation_frame : int, optional
        Ontly for aggregation mode. The number of top categories in animation_frame to include in the chart. For top using num column and agg_func
    trim_top_or_bottom : str, optional
        Trim from bottom or from top. Default is 'top'
    sort_axis, sort_color, sort_facet_col, sort_facet_row, sort_animation_frame : bool, optional
        Controls whether to sort the corresponding dimension (axis, color, facet columns, facet rows, or animation frames) based on the sum of numeric values across each slice. When True (default), the dimension will be sorted in descending order by the sum of numeric values. When False, no sorting will be applied and the original order will be preserved.
    show_group_size : bool, optional
        Whether to show the group size (only for groupby mode). Default is False
    min_group_size : int, optional
        The minimum number of observations required in a category to include it in the calculation.
        Categories with fewer observations than this threshold will be excluded from the analysis.
        This ensures that the computed mean is based on a sufficiently large sample size,
        improving the reliability of the results. Default is None (no minimum size restriction).
    decimal_places : int, optional
        The number of decimal places to display in hover. Default is 2
    x : str, optional
        Column to use for the x-axis
    y : str, optional
        Column to use for the y-axis
    color : str, optional
        Column to use for color encoding
    pattern : str, optional
        Column to use for pattern encoding
    hover_name : str, optional
        Column to use for hover text
    hover_data : list, optional
        List of columns to use for hover data
    custom_data : list, optional
        List of columns to use for custom data
    text : str, optional
        Column to use for text
    facet_row : str, optional
        Column to use for facet row
    facet_col : str, optional
        Column to use for facet column
    facet_col_wrap : int, optional
        Number of columns to use for facet column wrap
    error_x : str, optional
        Column to use for x-axis error bars
    error_x_minus : str, optional
        Column to use for x-axis error bars (minus)
    error_y : str, optional
        Column to use for y-axis error bars
    error_y_minus : str, optional
        Column to use for y-axis error bars (minus)
    animation_frame : str, optional
        Column to use for animation frame
    animation_group : str, optional
        Column to use for animation group
    range_x : list, optional
        List of values to use for x-axis range
    range_y : list, optional
        List of values to use for y-axis range
    log_x : bool, optional
        Whether to use a logarithmic scale for the x-axis. Default is False
    log_y : bool, optional
        Whether to use a logarithmic scale for the y-axis. Default is False
    range_x_equals : list, optional
        List of values to use for x-axis range, with equal scaling. Default is None
    range_y_equals : list, optional
        List of values to use for y-axis range, with equal scaling. Default is None
    title : str, optional
        Title of the chart. Default is None
    template : str, optional
        Template to use for the chart. Default is None
    width : int, optional
        Width of the chart in pixels. Default is None
    height : int, optional
        Height of the chart in pixels. Default is None
    show_box : bool, optional
        Whether to show boxplot in subplots
    show_count : bool, optional
            Whether to show countplot in subplots
    lower_quantile : float, optional
        The lower quantile for filtering the data. Value should be in the range [0, 1].
    upper_quantile : float, optional
        The upper quantile for filtering the data. Value should be in the range [0, 1].
    show_ci : bool, optional
        Whether to show confidence intervals. Default is False
    top_and_bottom : bool, optional
        Whether to show only top or both top and bottom
    show_legend_title : bool, optional
        Whether to show legend title. Default is False
    observed_for_groupby : bool, optional
        This only applies if any of the groupers are Categoricals.
        If True: only show observed values for categorical groupers.
        If False: show all values for categorical groupers.
        default False
    agg_func_for_top_n : str, optional
        Aggregation function for top_n_trim. Options: 'mean', 'median', 'sum', 'count', 'nunique'.
        By default agg_func_for_top_n = 'count'
    **kwargs
        Additional keyword arguments to pass to the Plotly Express function. Default is None

    Returns
    -------
    go.Figure
        The created chart
    """
    config = {
        'top_n_trim_x': top_n_trim_x,
        'top_n_trim_y': top_n_trim_y,
        'top_n_trim_color': top_n_trim_color,
        'top_n_trim_facet_col': top_n_trim_facet_col,
        'top_n_trim_facet_row': top_n_trim_facet_row,
        'top_n_trim_facet_animation_frame': top_n_trim_facet_animation_frame,
        'agg_column': agg_column,
        'sort_axis': sort_axis,
        'sort_color': sort_color,
        'sort_facet_col': sort_facet_col,
        'sort_facet_row': sort_facet_row,
        'sort_animation_frame': sort_animation_frame,
        'agg_func': agg_func,
        'show_group_size': show_group_size,
        'agg_mode': agg_mode,
        'resample_freq': resample_freq,
        'decimal_places': decimal_places,
        'norm_by': norm_by,
        'update_layout': update_layout,
        'show_box': show_box,
        'show_count': show_count,
        'lower_quantile_for_box': lower_quantile_for_box,
        'upper_quantile_for_box': upper_quantile_for_box,
        'show_ci': show_ci,
        'top_and_bottom': top_and_bottom,
        'trim_top_or_bottom': trim_top_or_bottom,
        'min_group_size': min_group_size,
        'show_legend_title': show_legend_title,
        'observed_for_groupby': observed_for_groupby,
        'agg_func_for_top_n': agg_func_for_top_n,
    }
    config = {k: v for k,v in config.items() if v is not None}
    return _create_base_fig_for_bar_line_area(df=data_frame, config=config, kwargs=kwargs, graph_type='bar')

def line(
    data_frame: pd.DataFrame,
    agg_mode: str = None,
    agg_func: str = None,
    agg_column: str = None,
    resample_freq: str = None,
    norm_by: str = None,
    top_n_trim_x: int = None,
    top_n_trim_y: int = None,
    top_n_trim_color: int = None,
    top_n_trim_facet_col: int = None,
    top_n_trim_facet_row: int = None,
    top_n_trim_facet_animation_frame: int = None,
    sort_axis: bool = True,
    sort_color: bool = True,
    sort_facet_col: bool = True,
    sort_facet_row: bool = True,
    sort_animation_frame: bool = True,
    show_group_size: bool = False,
    min_group_size: int = None,
    decimal_places: int = 2,
    update_layout: bool = True,
    trim_top_or_bottom: str = 'top',
    show_legend_title: bool = False,
    observed_for_groupby: bool = False,
    agg_func_for_top_n: bool = 'count',
    **kwargs
) -> go.Figure:
    """
    Creates a line chart using the Plotly Express library. This function is a wrapper around Plotly Express bar and accepts all the same parameters, allowing for additional customization and functionality.

    Parameters
    ----------
    data_frame : pd.DataFrame, optional
        DataFrame containing the data to be plotted
    agg_mode : str, optional
        Aggregation mode. Options:
        - 'groupby': Group by categorical columns
        - 'resample': Resample time series data
        - None: No aggregation (default)
    agg_func : str, optional
        Aggregation function. Options: 'mean', 'median', 'sum', 'count', 'nunique'
    agg_column : str, optional
         Column to aggregate
    resample_freq : str, optional
        Resample frequency for resample. Options: 'ME', 'W', D' and others
    norm_by: str, optional
        The name of the column to normalize by. If set to 'all', normalization will be performed based on the sum of all values in the dataset.
    top_n_trim_x : int, optional
        Ontly for aggregation mode. The number of top categories x axis to include in the chart. For top using num column and agg_func.
    top_n_trim_y : int, optional
        Ontly for aggregation mode. The number of top categories y axis to include in the chart. For top using num column and agg_func
    top_n_trim_color : int, optional
        Ontly for aggregation mode. The number of top categories legend to include in the chart. For top using num column and agg_func
    top_n_trim_facet_col : int, optional
        Ontly for aggregation mode. The number of top categories in facet_col to include in the chart. For top using num column and agg_func
    top_n_trim_facet_row : int, optional
        Ontly for aggregation mode. The number of top categories in facet_row to include in the chart. For top using num column and agg_func
    top_n_trim_facet_animation_frame : int, optional
        Ontly for aggregation mode. The number of top categories in animation_frame to include in the chart. For top using num column and agg_func
    trim_top_or_bottom : str, optional
        Trim from bottom or from top. Default is 'top'
    sort_axis, sort_color, sort_facet_col, sort_facet_row, sort_animation_frame : bool, optional
        Controls whether to sort the corresponding dimension (axis, color, facet columns, facet rows, or animation frames) based on the sum of numeric values across each slice. When True (default), the dimension will be sorted in descending order by the sum of numeric values. When False, no sorting will be applied and the original order will be preserved.
    show_group_size : bool, optional
        Whether to show the group size (only for groupby mode). Default is False
    min_group_size : int, optional
        The minimum number of observations required in a category to include it in the calculation.
        Categories with fewer observations than this threshold will be excluded from the analysis.
        This ensures that the computed mean is based on a sufficiently large sample size,
        improving the reliability of the results. Default is None (no minimum size restriction).
    decimal_places : int, optional
        The number of decimal places to display in hover. Default is 2
    x : str, optional
        Column to use for the x-axis
    y : str, optional
        Column to use for the y-axis
    color : str, optional
        Column to use for color encoding
    line_dash : str, optional
        Column to use for line dash encoding
    line_shape : str, optional
        Column to use for line shape encoding
    hover_name : str, optional
        Column to use for hover text
    hover_data : list, optional
        List of columns to use for hover data
    custom_data : list, optional
        List of columns to use for custom data
    text : str, optional
        Column to use for text
    facet_row : str, optional
        Column to use for facet row
    facet_col : str, optional
        Column to use for facet column
    facet_col_wrap : int, optional
        Number of columns to use for facet column wrap
    error_x : str, optional
        Column to use for x-axis error bars
    error_x_minus : str, optional
        Column to use for x-axis error bars (minus)
    error_y : str, optional
        Column to use for y-axis error bars
    error_y_minus : str, optional
        Column to use for y-axis error bars (minus)
    animation_frame : str, optional
        Column to use for animation frame
    animation_group : str, optional
        Column to use for animation group
    range_x : list, optional
        List of values to use for x-axis range
    range_y : list, optional
        List of values to use for y-axis range
    log_x : bool, optional
        Whether to use a logarithmic scale for the x-axis. Default is False
    log_y : bool, optional
        Whether to use a logarithmic scale for the y-axis. Default is False
    range_x_equals : list, optional
        List of values to use for x-axis range, with equal scaling. Default is None
    range_y_equals : list, optional
        List of values to use for y-axis range, with equal scaling. Default is None
    title : str, optional
        Title of the chart. Default is None
    template : str, optional
        Template to use for the chart. Default is None
    width : int, optional
        Width of the chart in pixels. Default is None
    height : int, optional
        Height of the chart in pixels. Default is None
    show_legend_title : bool, optional
        Whether to show legend title. Default is False
    observed_for_groupby : bool, optional
        This only applies if any of the groupers are Categoricals.
        If True: only show observed values for categorical groupers.
        If False: show all values for categorical groupers.
        default False
    agg_func_for_top_n : str, optional
        Aggregation function for top_n_trim. Options: 'mean', 'median', 'sum', 'count', 'nunique'
        By default agg_func_for_top_n = 'count'
    **kwargs
        Additional keyword arguments to pass to the Plotly Express function. Default is None
    Returns
    -------
    go.Figure
        The created chart
    """
    config = {
        'top_n_trim_x': top_n_trim_x,
        'top_n_trim_y': top_n_trim_y,
        'top_n_trim_color': top_n_trim_color,
        'top_n_trim_facet_col': top_n_trim_facet_col,
        'top_n_trim_facet_row': top_n_trim_facet_row,
        'top_n_trim_facet_animation_frame': top_n_trim_facet_animation_frame,
        'agg_column': agg_column,
        'sort_axis': sort_axis,
        'sort_color': sort_color,
        'sort_facet_col': sort_facet_col,
        'sort_facet_row': sort_facet_row,
        'sort_animation_frame': sort_animation_frame,
        'agg_func': agg_func,
        'show_group_size': show_group_size,
        'agg_mode': agg_mode,
        'resample_freq': resample_freq,
        'decimal_places': decimal_places,
        'norm_by': norm_by,
        'update_layout': update_layout,
        'trim_top_or_bottom': trim_top_or_bottom,
        'min_group_size': min_group_size,
        'show_legend_title': show_legend_title,
        'observed_for_groupby': observed_for_groupby,
        'agg_func_for_top_n': agg_func_for_top_n,
    }
    config = {k: v for k,v in config.items() if v is not None}
    return _create_base_fig_for_bar_line_area(df=data_frame, config=config, kwargs=kwargs, graph_type='line')

def area(
    data_frame: pd.DataFrame,
    agg_mode: str = None,
    agg_func: str = None,
    agg_column: str = None,
    resample_freq: str = None,
    norm_by: str = None,
    top_n_trim_x: int = None,
    top_n_trim_y: int = None,
    top_n_trim_color: int = None,
    top_n_trim_facet_col: int = None,
    top_n_trim_facet_row: int = None,
    top_n_trim_facet_animation_frame: int = None,
    sort_axis: bool = True,
    sort_color: bool = True,
    sort_facet_col: bool = True,
    sort_facet_row: bool = True,
    sort_animation_frame: bool = True,
    show_group_size: bool = False,
    min_group_size: int = None,
    decimal_places: int = 2,
    update_layout: bool = True,
    trim_top_or_bottom: str = 'top',
    show_legend_title: bool = False,
    observed_for_groupby: bool = False,
    agg_func_for_top_n: bool = 'count',
    **kwargs
) -> go.Figure:
    """
    Creates an area chart using the Plotly Express library. This function is a wrapper around Plotly Express bar and accepts all the same parameters, allowing for additional customization and functionality.

    Parameters
    ----------
    data_frame : pd.DataFrame, optional
        DataFrame containing the data to be plotted
    agg_mode : str, optional
        Aggregation mode. Options:
        - 'groupby': Group by categorical columns
        - 'resample': Resample time series data
        - None: No aggregation (default)
    agg_func : str, optional
        Aggregation function. Options: 'mean', 'median', 'sum', 'count', 'nunique'
    agg_column : str, optional
         Column to aggregate
    resample_freq : str, optional
        Resample frequency for resample. Options: 'ME', 'W', D' and others
    norm_by: str, optional
        The name of the column to normalize by. If set to 'all', normalization will be performed based on the sum of all values in the dataset.
    top_n_trim_x : int, optional
        Ontly for aggregation mode. The number of top categories x axis to include in the chart. For top using num column and agg_func.
    top_n_trim_y : int, optional
        Ontly for aggregation mode. The number of top categories y axis to include in the chart. For top using num column and agg_func
    top_n_trim_color : int, optional
        Ontly for aggregation mode. The number of top categories legend to include in the chart. For top using num column and agg_func
    top_n_trim_facet_col : int, optional
        Ontly for aggregation mode. The number of top categories in facet_col to include in the chart. For top using num column and agg_func
    top_n_trim_facet_row : int, optional
        Ontly for aggregation mode. The number of top categories in facet_row to include in the chart. For top using num column and agg_func
    top_n_trim_facet_animation_frame : int, optional
        Ontly for aggregation mode. The number of top categories in animation_frame to include in the chart. For top using num column and agg_func
    trim_top_or_bottom : str, optional
        Trim from bottom or from top. Default is 'top'
    sort_axis, sort_color, sort_facet_col, sort_facet_row, sort_animation_frame : bool, optional
        Controls whether to sort the corresponding dimension (axis, color, facet columns, facet rows, or animation frames) based on the sum of numeric values across each slice. When True (default), the dimension will be sorted in descending order by the sum of numeric values. When False, no sorting will be applied and the original order will be preserved.
    show_group_size : bool, optional
        Whether to show the group size (only for groupby mode). Default is False
    min_group_size : int, optional
        The minimum number of observations required in a category to include it in the calculation.
        Categories with fewer observations than this threshold will be excluded from the analysis.
        This ensures that the computed mean is based on a sufficiently large sample size,
        improving the reliability of the results. Default is None (no minimum size restriction).
    decimal_places : int, optional
        The number of decimal places to display in hover. Default is 2
    x : str, optional
        Column to use for the x-axis
    y : str, optional
        Column to use for the y-axis
    color : str, optional
        Column to use for color encoding
    line_dash : str, optional
        Column to use for line dash encoding
    line_shape : str, optional
        Column to use for line shape encoding
    hover_name : str, optional
        Column to use for hover text
    hover_data : list, optional
        List of columns to use for hover data
    custom_data : list, optional
        List of columns to use for custom data
    text : str, optional
        Column to use for text
    facet_row : str, optional
        Column to use for facet row
    facet_col : str, optional
        Column to use for facet column
    facet_col_wrap : int, optional
        Number of columns to use for facet column wrap
    error_x : str, optional
        Column to use for x-axis error bars
    error_x_minus : str, optional
        Column to use for x-axis error bars (minus)
    error_y : str, optional
        Column to use for y-axis error bars
    error_y_minus : str, optional
        Column to use for y-axis error bars (minus)
    animation_frame : str, optional
        Column to use for animation frame
    animation_group : str, optional
        Column to use for animation group
    range_x : list, optional
        List of values to use for x-axis range
    range_y : list, optional
        List of values to use for y-axis range
    log_x : bool, optional
        Whether to use a logarithmic scale for the x-axis. Default is False
    log_y : bool, optional
        Whether to use a logarithmic scale for the y-axis. Default is False
    range_x_equals : list, optional
        List of values to use for x-axis range, with equal scaling. Default is None
    range_y_equals : list, optional
        List of values to use for y-axis range, with equal scaling. Default is None
    title : str, optional
        Title of the chart. Default is None
    template : str, optional
        Template to use for the chart. Default is None
    width : int, optional
        Width of the chart in pixels. Default is None
    height : int, optional
        Height of the chart in pixels. Default is None
    groupnorm : str, optional
        Normalization method for groups. Options: 'fraction', 'percent'. Default is None
    stackgroup : str, optional
        Method for stacking groups. Options: 'absolute', 'relative', 'percent'. Default is None
    show_legend_title : bool, optional
        Whether to show legend title. Default is False
    observed_for_groupby : bool, optional
        This only applies if any of the groupers are Categoricals.
        If True: only show observed values for categorical groupers.
        If False: show all values for categorical groupers.
        default False
    agg_func_for_top_n : str, optional
        Aggregation function for top_n_trim. Options: 'mean', 'median', 'sum', 'count', 'nunique'
        By default agg_func_for_top_n = 'count'
    **kwargs
        Additional keyword arguments to pass to the Plotly Express function. Default is None
    Returns
    -------
    go.Figure
        The created chart
    """
    config = {
        'top_n_trim_x': top_n_trim_x,
        'top_n_trim_y': top_n_trim_y,
        'top_n_trim_color': top_n_trim_color,
        'top_n_trim_facet_col': top_n_trim_facet_col,
        'top_n_trim_facet_row': top_n_trim_facet_row,
        'top_n_trim_facet_animation_frame': top_n_trim_facet_animation_frame,
        'agg_column': agg_column,
        'sort_axis': sort_axis,
        'sort_color': sort_color,
        'sort_facet_col': sort_facet_col,
        'sort_facet_row': sort_facet_row,
        'sort_animation_frame': sort_animation_frame,
        'agg_func': agg_func,
        'show_group_size': show_group_size,
        'agg_mode': agg_mode,
        'resample_freq': resample_freq,
        'decimal_places': decimal_places,
        'norm_by': norm_by,
        'update_layout': update_layout,
        'trim_top_or_bottom': trim_top_or_bottom,
        'min_group_size': min_group_size,
        'show_legend_title': show_legend_title,
        'observed_for_groupby': observed_for_groupby,
        'agg_func_for_top_n': agg_func_for_top_n,
    }
    config = {k: v for k,v in config.items() if v is not None}
    return _create_base_fig_for_bar_line_area(df=data_frame, config=config, kwargs=kwargs, graph_type='area')



def box(
    data_frame: pd.DataFrame = None,
    top_n: int = None,
    lower_quantile: float = None,
    upper_quantile: float = None,
    sort: bool = False,
    **kwargs
) -> go.Figure:
    """
    Creates a box plot using Plotly Express.  This function is a wrapper around Plotly Express bar and accepts all the same parameters, allowing for additional customization and

    Parameters:
    ----------
    data_frame : pd.DataFrame, optional
        The DataFrame containing the data to be plotted.

    top_n : int, optional
        The number of top categories to display. If None, all categories are shown.

    lower_quantile : float, optional
        The lower quantile for filtering the data. Value should be in the range [0, 1].

    upper_quantile : float, optional
        The upper quantile for filtering the data. Value should be in the range [0, 1].

    sort : bool, optional
        If True, sorts categories by median. Default is False.

    x : str, optional
        Column name to be used for the x-axis.

    y : str, optional
        Column name to be used for the y-axis.

    color : str, optional
        Column name to be used for color encoding.

    category_orders : dict, optional
        A dictionary specifying the order of categories for the x-axis.

    labels : dict, optional
        A dictionary mapping column names to their display names.

    title : str, optional
        The title of the plot.

    template : str, optional
        The template to use for the plot.

    **kwargs :
        Additional parameters that can be passed to the Plotly Express box function.

    Returns:
    -------
    go.Figure
        A Figure object containing the box plot.
    """

    # Function to trim the data based on specified quantiles
    # def trim_by_quantiles(group):
    #     # Calculate the lower bound using the lower quantile
    #     lower_bound = group.quantile(lower_quantile)
    #     # Calculate the upper bound using the upper quantile
    #     upper_bound = group.quantile(upper_quantile)
    #     # Return the group filtered between the lower and upper bounds
    #     return group[(group >= lower_bound) & (group <= upper_bound)]

    # Assign the input data frame to a variable
    df = data_frame

    # Retrieve the orientation parameter from kwargs
    orientation = kwargs.get('orientation')
    # Retrieve the x-axis variable from kwargs
    x = kwargs.get('x')
    # Retrieve the y-axis variable from kwargs
    y = kwargs.get('y')
    # Retrieve the color variable from kwargs
    color = kwargs.get('color')

    # Determine the categorical and numerical variables based on orientation
    if orientation is None or orientation == 'v':
        cat_var = x
        num_var = y
    else:
        cat_var = y
        num_var = x

    # Initialize categories variable
    categories = None
    # Check if sorting is not required
    if not sort:
        # If top_n is specified, get the top n categories based on value counts
        if top_n:
            # categories = df[cat_var].value_counts().nlargest(top_n).index.tolist()
            categories = df.groupby(cat_var, observed=False)[num_var].median().nlargest(top_n).index.tolist()
    else:
        # If the categorical variable is of type Categorical, get its categories
        # if isinstance(df[cat_var], pd.CategoricalDtype):
        #     categories = df[cat_var].cat.categories.tolist()
        # # If the categorical variable is numeric, sort and get unique values
        # elif pd.api.types.is_numeric_dtype(df[cat_var]):
        #     categories = sorted(df[cat_var].unique().tolist())
        # # For other types, get unique values directly
        # else:
        #     categories = df[cat_var].unique().tolist()
        categories = df.groupby(cat_var, observed=False)[num_var].median().sort_values(ascending=False).index.tolist()
        kwargs['category_orders'] = {cat_var: categories}

        # If top_n is specified, limit the categories to top_n
        if top_n:
            categories = categories[:top_n]

    # Filter the dataframe to include only the selected categories
    if categories:
        df = df[df[cat_var].isin(categories)]

    # Initialize trimmed_df with the original dataframe
    trimmed_df = df

    # Check if quantile trimming is required
    if upper_quantile or lower_quantile:
        # Set default upper quantile if not provided
        if not upper_quantile:
            upper_quantile = 1
        # Set default lower quantile if not provided
        if not lower_quantile:
            lower_quantile = 0

        # Prepare columns for grouping
        columns_for_groupby = [cat_var]
        # If color is specified, include it in the grouping
        if color:
            columns_for_groupby.append(color)

        # Apply the trim_by_quantiles function to the grouped data
        temp_for_range = df.groupby(columns_for_groupby, observed=False)[num_var].quantile([lower_quantile, upper_quantile]).unstack()
        lower_range = temp_for_range.iloc[:, 0].min()
        upper_range = temp_for_range.iloc[:, 1].max()
        lower_range -= (upper_range - lower_range) * 0.05

    # Create a box plot using the trimmed dataframe
    fig = px.box(trimmed_df, **kwargs)

    # Initialize a dictionary to update figure configuration
    fig_update_config = dict()

    # Configure the figure based on orientation
    if orientation is None or orientation == 'v':
        # Disable grid lines for the y-axis
        fig_update_config['yaxis_showgrid'] = False
        # Update hover template for each trace
        for trace in fig.data:
            trace.hovertemplate = trace.hovertemplate.replace('{x}', '{x:.2f}')
    else:
        # Disable grid lines for the x-axis
        fig_update_config['xaxis_showgrid'] = False
        # Update hover template for each trace
        for trace in fig.data:
            trace.hovertemplate = trace.hovertemplate.replace('{y}', '{y:.2f}')

    # Set default width if not specified
    if not kwargs.get('width'):
        fig_update_config['width'] = 800
    # Set default height if not specified
    if not kwargs.get('height'):
        fig_update_config['height'] = 400

    # Configure legend settings if color is specified
    if kwargs.get('color'):
        fig_update_config['legend_position'] = 'top'
        fig_update_config['legend_title'] = ''
    if upper_quantile or lower_quantile:
        if kwargs.get('orientation') == 'h':
            fig.update_xaxes(range=[lower_range, upper_range])
        else:
            fig.update_yaxes(range=[lower_range, upper_range])
    # Update the figure with the specified configurations
    fig = fig_update(fig, **fig_update_config)

    # Return the final figure
    return fig

def violin(
    data_frame: pd.DataFrame = None,
    top_n: int = None,
    lower_quantile: float = None,
    upper_quantile: float = None,
    sort: bool = False,
    **kwargs
) -> go.Figure:
    """
    Creates a violin plot using Plotly Express.  This function is a wrapper around Plotly Express bar and accepts all the same parameters, allowing for additional customization and

    Parameters:
    ----------
    data_frame : pd.DataFrame, optional
        The DataFrame containing the data to be plotted.

    top_n : int, optional
        The number of top categories to display. If None, all categories are shown.

    lower_quantile : float, optional
        The lower quantile for filtering the data. Value should be in the range [0, 1].

    upper_quantile : float, optional
        The upper quantile for filtering the data. Value should be in the range [0, 1].

    sort : bool, optional
        If True, sorts categories by median. Default is False.

    x : str, optional
        Column name to be used for the x-axis.

    y : str, optional
        Column name to be used for the y-axis.

    color : str, optional
        Column name to be used for color encoding.

    category_orders : dict, optional
        A dictionary specifying the order of categories for the x-axis.

    labels : dict, optional
        A dictionary mapping column names to their display names.

    title : str, optional
        The title of the plot.

    template : str, optional
        The template to use for the plot.

    **kwargs :
        Additional parameters that can be passed to the Plotly Express box function.

    Returns:
    -------
    go.Figure
        A Figure object containing the box plot.
    """

    # Function to trim the data based on specified quantiles
    # def trim_by_quantiles(group):
    #     # Calculate the lower bound using the lower quantile
    #     lower_bound = group.quantile(lower_quantile)
    #     # Calculate the upper bound using the upper quantile
    #     upper_bound = group.quantile(upper_quantile)
    #     # Return the group filtered between the lower and upper bounds
    #     return group[(group >= lower_bound) & (group <= upper_bound)]

    # Assign the input data frame to a variable
    df = data_frame

    # Retrieve the orientation parameter from kwargs
    orientation = kwargs.get('orientation')
    kwargs.setdefault('box', True)
    # Retrieve the x-axis variable from kwargs
    x = kwargs.get('x')
    # Retrieve the y-axis variable from kwargs
    y = kwargs.get('y')
    # Retrieve the color variable from kwargs
    color = kwargs.get('color')

    # Determine the categorical and numerical variables based on orientation
    if orientation is None or orientation == 'v':
        cat_var = x
        num_var = y
    else:
        cat_var = y
        num_var = x

    # Initialize categories variable
    categories = None
    # Check if sorting is not required
    if not sort:
        # If top_n is specified, get the top n categories based on value counts
        if top_n:
            # categories = df[cat_var].value_counts().nlargest(top_n).index.tolist()
            categories = df.groupby(cat_var, observed=False)[num_var].median().nlargest(top_n).index.tolist()
    else:
        # If the categorical variable is of type Categorical, get its categories
        # if isinstance(df[cat_var], pd.CategoricalDtype):
        #     categories = df[cat_var].cat.categories.tolist()
        # # If the categorical variable is numeric, sort and get unique values
        # elif pd.api.types.is_numeric_dtype(df[cat_var]):
        #     categories = sorted(df[cat_var].unique().tolist())
        # # For other types, get unique values directly
        # else:
        #     categories = df[cat_var].unique().tolist()
        categories = df.groupby(cat_var, observed=False)[num_var].median().sort_values(ascending=False).index.tolist()
        kwargs['category_orders'] = {cat_var: categories}

        # If top_n is specified, limit the categories to top_n
        if top_n:
            categories = categories[:top_n]

    # Filter the dataframe to include only the selected categories
    if categories:
        df = df[df[cat_var].isin(categories)]

    # Initialize trimmed_df with the original dataframe
    trimmed_df = df

    # Check if quantile trimming is required
    if upper_quantile or lower_quantile:
        # Set default upper quantile if not provided
        if not upper_quantile:
            upper_quantile = 1
        # Set default lower quantile if not provided
        if not lower_quantile:
            lower_quantile = 0

        # Prepare columns for grouping
        columns_for_groupby = [cat_var]
        # If color is specified, include it in the grouping
        if color:
            columns_for_groupby.append(color)

        # Apply the trim_by_quantiles function to the grouped data
        temp_for_range = df.groupby(columns_for_groupby, observed=False)[num_var].quantile([lower_quantile, upper_quantile]).unstack()
        lower_range = temp_for_range.iloc[:, 0].min()
        upper_range = temp_for_range.iloc[:, 1].max()
        lower_range -= (upper_range - lower_range) * 0.05

    # Create a box plot using the trimmed dataframe
    fig = px.box(trimmed_df, **kwargs)

    # Initialize a dictionary to update figure configuration
    fig_update_config = dict()

    # Configure the figure based on orientation
    if orientation is None or orientation == 'v':
        # Disable grid lines for the x-axis
        fig_update_config['xaxis_showgrid'] = False
        # Update hover template for each trace
        for trace in fig.data:
            trace.hovertemplate = trace.hovertemplate.replace('{y}', '{y:.2f}')
    else:
        # Disable grid lines for the y-axis
        fig_update_config['yaxis_showgrid'] = False
        # Update hover template for each trace
        for trace in fig.data:
            trace.hovertemplate = trace.hovertemplate.replace('{x}', '{x:.2f}')

    # Set default width if not specified
    if not kwargs.get('width'):
        fig_update_config['width'] = 800
    # Set default height if not specified
    if not kwargs.get('height'):
        fig_update_config['height'] = 400

    # Configure legend settings if color is specified
    if kwargs.get('color'):
        fig_update_config['legend_position'] = 'top'
        fig_update_config['legend_title'] = ''
    if upper_quantile or lower_quantile:
        if kwargs.get('orientation') == 'h':
            fig.update_xaxes(range=[lower_range, upper_range])
        else:
            fig.update_yaxes(range=[lower_range, upper_range])
    # Update the figure with the specified configurations
    fig = fig_update(fig, **fig_update_config)

    # Return the final figure
    return fig

def heatmap(
    data_frame: pd.DataFrame,
    x: str = None,
    y: str = None,
    z: str = None,
    do_pivot: bool = False,
    agg_func: str = None,
    top_n_trim_x: int = None,
    top_n_trim_y: int = None,
    top_n_trim_from_x: str = 'start',
    top_n_trim_from_y: str = 'start',
    sort_x: bool = False,
    sort_y: bool = False,
    reverse_x: bool = False,
    reverse_y: bool = False,
    skip_first_col_for_cohort: bool = False,
    texttemplate: str = None,
    **kwargs
) -> go.Figure:
    """
    Creates a heatmap chart using the Plotly Express library. It can accept either a raw DataFrame, in which case aggregation and pivot table creation will be performed internally, or a pre-aggregated pivot table, in which case the data will be plotted directly (x, y, and z parameters are not required). This function is a wrapper around Plotly Express's heatmap functionality and accepts all the same parameters, allowing for additional customization and functionality.

    Parameters
    ----------
    data_frame : pd.DataFrame, optional
        DataFrame containing the data to be plotted
    x : str, optional
        Column name in the DataFrame to use for the x-axis
    y : str, optional
        Column name in the DataFrame to use for the y-axis
    z : str, optional
        Column name in the DataFrame to use for the z-values
    do_pivot : bool, optional
        Whether to do pivot table before creating the heatmap
    agg_func : str, optional
        Aggregation function to use if is_agg is True. Options:
        - 'mean': Calculate the mean of the values
        - 'sum': Calculate the sum of the values
        - 'count': Count the number of values
        - 'max': Calculate the maximum of the values
        - 'min': Calculate the minimum of the values
    top_n_trim_x : int, optional
        Number of top columns to display on the x-axis
    top_n_trim_y : int, optional
        Number of top rows to display on the y-axis
    top_n_trim_from_x : str, optional
        Whether to start counting from the beginning or end of the x-axis. Options:
        - 'start': Start counting from the beginning
        - 'end': Start counting from the end
    top_n_trim_from_y : str, optional
        Whether to start counting from the beginning or end of the y-axis. Options:
        - 'start': Start counting from the beginning
        - 'end': Start counting from the end
    sort_x : bool, optional
        Whether to sort the x-axis
    sort_y : bool, optional
        Whether to sort the y-axis
    reverse_x : bool, optional
        Whether to reverse the x-axis
    reverse_y : bool, optional
        Whether to reverse the y-axis
    skip_first_col_for_cohort : bool, optional
        Whether to skip the first column for cohort analysis
    texttemplate : str, optional
        Template for text displayed on the plot.
        Example: texttemplate='%{z:.1f}%'
    **kwargs : optional
        Additional keyword arguments to pass to Plotly Express

    Returns
    -------
    go.Figure
        The created heatmap figure
    """
    # Check if data_frame is a pandas DataFrame
    if not isinstance(data_frame, pd.DataFrame):
        raise ValueError('data_frame must be pandas DataFrame')

    # Copy the DataFrame for further processing
    df = data_frame

    # If aggregation is enabled, check that x, y, and z are defined
    if do_pivot:
        if x is None:
            raise ValueError('x must be defined')
        if y is None:
            raise ValueError('y must be defined')
        if z is None:
            raise ValueError('z must be defined')
        # If aggregation is enabled, check that agg_func is defined
        if not agg_func:
            raise ValueError("For agg mode agg_func must be defined")

    # Set default parameters for Plotly Express
    kwargs.setdefault('aspect', True)      # Set aspect ratio to True
    kwargs.setdefault('text_auto', True)    # Set text auto to True
    kwargs.setdefault('color_continuous_scale', colorway_for_heatmap)  # Set color continuous scale

    # Function to create a DataFrame for the figure
    def make_df_for_fig(df, x, y, z, agg_func):
        # Create a pivoted DataFrame with aggregated data
        df_pivoted =  pd.pivot_table(df, index=y, columns=x, values=z, aggfunc=agg_func, observed=False)

        # Sort columns and rows
        ascending_x = False
        ascending_y = False
        if sort_x:
            df_pivoted.loc['sum'] = df_pivoted.sum(numeric_only=True)
            df_pivoted = df_pivoted.sort_values(
                'sum', ascending=ascending_x, axis=1) #.drop('sum')
            display(df_pivoted)
        if sort_y:
            df_pivoted['sum'] = df_pivoted.sum(axis=1, numeric_only=True)
            df_pivoted = df_pivoted.sort_values(
                'sum', ascending=ascending_y).drop('sum', axis=1)

        # Reverse columns and rows if necessary
        if reverse_x:
            df_pivoted = df_pivoted.iloc[:, ::-1]
        if reverse_y:
            df_pivoted = df_pivoted.iloc[::-1]

        # Skip the first column if necessary
        if skip_first_col_for_cohort:
            df_pivoted = df_pivoted.iloc[:, 1:]

        return df_pivoted

    # If aggregation is enabled, create a DataFrame for the figure
    if do_pivot:
        df_for_fig = make_df_for_fig(df, x, y, z, agg_func)

        # Select top N columns and rows if necessary
        if top_n_trim_x:
            if top_n_trim_from_x == 'end':
                x_nunique = df[x].nunique()
                df_for_fig = df_for_fig.iloc[:, x_nunique-top_n_trim_x:]
            else:
                df_for_fig = df_for_fig.iloc[:, :top_n_trim_x]
        if top_n_trim_y:
            if top_n_trim_from_y == 'end':
                y_nunique = df[y].nunique()
                df_for_fig = df_for_fig.iloc[y_nunique-top_n_trim_y:]
            else:
                df_for_fig = df_for_fig.iloc[:top_n_trim_y]

        # Create the figure using Plotly Express
        fig = px.imshow(df_for_fig, **kwargs)
    else:
        # Create the figure using Plotly Express
        fig = px.imshow(df, **kwargs)

    # Check if the z column is numeric and not integer
    if z:
        if isinstance(z, str):
            z_column = df[z]
        else:
            z_column = z
        is_z_numeric = pd.api.types.is_numeric_dtype(z_column)
        is_z_integer = pd.api.types.is_integer_dtype(z_column)
        if is_z_numeric and not is_z_integer:
            # Format the hover text to display two decimal places
            for trace in fig.data:
                trace.hovertemplate = trace.hovertemplate.replace('{z}', '{z:.2f}')

    # Update the text template if necessary
    if texttemplate:
        fig.update_traces(texttemplate=texttemplate)

    fig = fig_update(fig)
    # Return the figure
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
                        family='Noto Sans',
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
                        family='Noto Sans',
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
        font=dict(size=14, family="Noto Sans", color="rgba(0, 0, 0, 0.7)"),
        title_font=dict(size=16, color="rgba(0, 0, 0, 0.7)"),     
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
        font=dict(size=14, family="Noto Sans", color="rgba(0, 0, 0, 0.7)"),
        title_font=dict(size=16, color="rgba(0, 0, 0, 0.7)"),
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
            font=dict(size=14, family="Noto Sans", color="rgba(0, 0, 0, 0.7)"),
            title_font=dict(size=16, color="rgba(0, 0, 0, 0.7)"),
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
                    current_columns, observed=False)[[values_column]].agg(value=(values_column, func)).reset_index()
            else:
                df_grouped = df_in[current_columns].groupby(
                    current_columns, observed=False).size().reset_index().rename(columns={0: 'value'})
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
            
# def _create_base_fig_for_bar_line_area(config: dict, args: list, kwargs: dict, graph_type: str = 'bar'):
#     """
#     Creates a figure for bar, line or area function using the Plotly Express library.
#     """
#
#     if 'agg_mode' in config and config['agg_mode'] == 'resample' and 'resample_freq' not in config:
#         raise ValueError("For resample mode resample_freq must be define")
#     if 'agg_mode' in config and config['agg_mode'] == 'groupby' and 'groupby_by' not in config:
#         raise ValueError("For groupby mode groupby_by must be define")
#
#     def human_readable_number(x, decimal_places):
#         format_string = f"{{:.{decimal_places}f}}"
#
#         if x >= 1e6 or x <= -1e6:
#             return f"{format_string.format(x / 1e6)}M"
#         elif x >= 1e3 or x <= -1e3:
#             return f"{format_string.format(x / 1e3)}k"
#         else:
#             return format_string.format(x)
#
#     def prepare_df(config: dict):
#         df = args[0]
#         color = [kwargs['color']] if 'color' in kwargs else []
#         if config['groupby_by']:
#             if 'agg_func' not in config:
#                 raise ValueError(
#             num_column = set([kwargs['x'], kwargs['y']]) - set(config['groupby_by'])
#             if len(num_column) != 1:
#                 raise ValueError(
#                     "Error: There must be exactly one numeric value among the parameters 'x' and 'y' "
#                     "that is not included in 'groupby_by'. Ensure that only one of 'x' or 'y' "
#                     "is not present in the 'groupby_by' list."
#                 )
#             num_column = num_column.pop()
#             cat_columns = config['groupby_by']
#         else:
#             if not (pd.api.types.is_numeric_dtype(df[config['x']]) or pd.api.types.is_numeric_dtype(df[config['y']])):
#                 raise ValueError("At least one of x or y must be numeric.")
#             elif pd.api.types.is_numeric_dtype(df[config['y']]):
#                 cat_columns = [config['x']] + color
#                 num_column = config['y']
#             else:
#                 cat_columns = [config['y']] + color
#                 num_column = config['x']
#         if config['agg_func'] is None:
#             func = 'first'
#         else:
#             func = config.get('agg_func', 'mean')  # default to 'mean' if not provided
#         if config['y'] == num_column:
#             ascending = False
#         else:
#             ascending = True
#         func_df = (df[[*cat_columns, num_column]]
#                    .groupby(cat_columns, observed=False)
#                    .agg(num=(num_column, func), count=(num_column, 'count'))
#                    .reset_index())
#         if config['sort_axis']:
#             func_df['temp'] = func_df.groupby(cat_columns[0], observed=False)[
#                 'num'].transform('sum')
#             func_df = (func_df.sort_values(['temp', 'num'], ascending=ascending)
#                     .drop('temp', axis=1)
#                     )
#         if not config['sort_legend']:
#             if config['sort_axis']:
#                 func_df = (func_df.sort_values([cat_columns[0], cat_columns[1]], ascending=[False, True])
#                         )
#         func_df['count'] = func_df['count'].apply(
#             lambda x: f'= {x}' if x <= 1e3 else 'больше 1000')
#         func_df['pretty_value'] = func_df['num'].apply(human_readable_number, args = [config['decimal_places']])
#         func_df[cat_columns] = func_df[cat_columns].astype('str')
#         return func_df.rename(columns={'num': num_column})
#     xaxis_title = config['xaxis_title']
#     yaxis_title = config['yaxis_title']
#     category_axis_title = config['category_axis_title']
#     if config['agg_mode'] == 'resample':
#         if config['agg_func'] is None:
#             func = 'first'
#         else:
#             func = config['agg_func']
#         columns = [config['x'], config['y']]
#         if config['category']:
#             columns.append(config['category'])
#             df_for_fig = config['df'][columns].set_index(config['x']).groupby(config['category'], observed=False ).resample(config['resample_freq'])[config['y']].agg(func).reset_index()
#         else:
#             df_for_fig = config['df'][columns].set_index(config['x']).resample(config['resample_freq']).agg(func).reset_index()
#         # x = config['df'][config['x']].values
#         # y = config['df'][config['y']].values
#         custom_data = [df_for_fig[config['y']].apply(human_readable_number, args = [config['decimal_places']])]
#         # if pd.api.types.is_numeric_dtype(config['df'][config['y']]):
#         #     custom_data = [df_for_fig[config['y']].apply(human_readable_number, args = [config['decimal_places']])]
#         # else:
#         #     custom_data = [df_for_fig[config['x']].apply(human_readable_number, args = [config['decimal_places']])]
#         if graph_type == 'bar':
#             fig = px.bar(df_for_fig, x=config['x'], y=config['y'], color=config['category'],
#                         barmode=config['barmode'], custom_data=custom_data)
#         elif graph_type == 'line':
#             fig = px.line(df_for_fig, x=config['x'], y=config['y'], color=config['category'], custom_data=custom_data)
#         elif graph_type == 'area':
#             fig = px.area(df_for_fig, x=config['x'], y=config['y'], color=config['category'], custom_data=custom_data)
#     elif config['agg_mode'] == 'groupby':
#         df_for_fig = prepare_df(config)
#         if config['top_n_trim_axis']:
#             df_for_fig = df_for_fig.iloc[:config['top_n_trim_axis']]
#         # if config['top_n_trim_legend']:
#         #     df_for_fig = pd.concat([df_for_fig['data'].iloc[:, :config['top_n_trim_legend']], df_for_fig['data'].iloc[:, :config['top_n_trim_legend']]], axis=1, keys=['data', 'customdata'])
#         # display(df_for_fig)
#         x = df_for_fig[config['x']].values
#         y = df_for_fig[config['y']].values
#         color = df_for_fig[config['category']
#                         ].values if config['category'] else None
#         custom_data = [df_for_fig['count'], df_for_fig['pretty_value']]
#         # display(df_for_fig)
#         if 'show_text' in config and config['show_text']:
#             if pd.api.types.is_numeric_dtype(df_for_fig[config['y']]):
#                 text = [human_readable_number(el, config['decimal_places']) for el in y]
#             else:
#                 text = [human_readable_number(el, config['decimal_places']) for el in x]
#         else:
#             text = None
#         # display(df_for_fig)
#         # display(custom_data)
#         if graph_type == 'bar':
#             fig = px.bar(x=x, y=y, color=color,
#                         barmode=config['barmode'], text=text, custom_data=custom_data)
#         elif graph_type == 'line':
#             fig = px.line(x=x, y=y, color=color,
#                         text=text, custom_data=custom_data)
#         elif graph_type == 'area':
#             fig = px.area(x=x, y=y, color=color,
#                         text=text, custom_data=custom_data)
#         color = []
#         for trace in fig.data:
#             color.append(trace.marker.color)
#         if graph_type == 'bar':
#             fig.update_traces(textposition='auto')
#         elif graph_type == 'line':
#             fig.update_traces(textposition='top center')
#         elif graph_type == 'area':
#             fig.update_traces(textposition='top center')
#         if pd.api.types.is_numeric_dtype(df_for_fig[config['x']]):
#             # Чтобы сортировка была по убыванию вернего значения, нужно отсортировать по последнего значению в x
#             traces = list(fig.data)
#             traces.sort(key=lambda x: x.x[-1])
#             fig.data = traces
#             color = color[::-1]
#             for i, trace in enumerate(fig.data):
#                 trace.marker.color = color[i]
#             fig.update_layout(legend={'traceorder': 'reversed'})
#         if config['textposition']:
#             fig.update_traces(textposition=config['textposition'])
#     else:
#         if pd.api.types.is_numeric_dtype(config['df'][config['y']]):
#             if not config['sort_axis'] or pd.api.types.is_datetime64_any_dtype(config['df'][config['x']]):
#                 df = config['df']
#             else:
#                 num_for_sort = config['y']
#                 ascending_for_sort = False
#                 df = config['df'].sort_values(num_for_sort, ascending=ascending_for_sort)
#             custom_data = [df[config['y']].apply(human_readable_number, args = [config['decimal_places']])]
#         else:
#             if config['sort_axis']:
#                 num_for_sort = config['x']
#                 ascending_for_sort = True
#                 df = config['df'].sort_values(num_for_sort, ascending=ascending_for_sort)
#             else:
#                 df = config['df']
#             custom_data = [df[config['x']].apply(human_readable_number, args = [config['decimal_places']])]
#         if graph_type == 'bar':
#             fig = px.bar(df, x=config['x'], y=config['y'], color=config['category'],
#                         barmode=config['barmode'], custom_data=custom_data)
#         elif graph_type == 'line':
#             fig = px.line(df, x=config['x'], y=config['y'], color=config['category'], custom_data=custom_data)
#         elif graph_type == 'area':
#             fig = px.area(df, x=config['x'], y=config['y'], color=config['category'], custom_data=custom_data)
#     if config['legend_position'] == 'top':
#         fig.update_layout(
#             yaxis = dict(
#                 domain=[0, 0.9]
#             )
#             , legend = dict(
#                 title_text=category_axis_title
#                 , title_font_color='rgba(0, 0, 0, 0.7)'
#                 , font_color='rgba(0, 0, 0, 0.7)'
#                 , orientation="h"  # Горизонтальное расположение
#                 , yanchor="top"    # Привязка к верхней части
#                 , y=1.09         # Положение по вертикали (отрицательное значение переместит вниз)
#                 , xanchor="center" # Привязка к центру
#                 , x=0.5              # Центрирование по горизонтали
#             )
#         )
#     elif config['legend_position'] == 'right':
#         fig.update_layout(
#                 legend = dict(
#                 title_text=category_axis_title
#                 , title_font_color='rgba(0, 0, 0, 0.7)'
#                 , font_color='rgba(0, 0, 0, 0.7)'
#                 , orientation="v"  # Горизонтальное расположение
#                 # , yanchor="bottom"    # Привязка к верхней части
#                 , y=1         # Положение по вертикали (отрицательное значение переместит вниз)
#                 # , xanchor="center" # Привязка к центру
#                 # , x=0.5              # Центрирование по горизонтали
#             )
#         )
#     else:
#         raise ValueError("Invalid legend_position. Please choose 'top' or 'right'.")
#     if xaxis_title:
#         hovertemplate_x = f'{xaxis_title} = '
#     else:
#         hovertemplate_x = f'x = '
#     if yaxis_title:
#         hovertemplate_y = f'{yaxis_title} = '
#     else:
#         hovertemplate_y = f'y = '
#     if category_axis_title:
#         hovertemplate_color = f'<br>{category_axis_title} = '
#     else:
#         hovertemplate_color = f'color = '
#     if config['agg_mode'] == 'groupby':
#         if pd.api.types.is_numeric_dtype(df_for_fig[config['y']]):
#             hovertemplate = hovertemplate_x + \
#                 '%{x}<br>' + hovertemplate_y + '%{customdata[1]}'
#         else:
#             hovertemplate = hovertemplate_x + \
#                 '%{customdata[1]}<br>' + hovertemplate_y + '%{y}'
#     elif config['agg_mode'] == 'resample':
#         hovertemplate = hovertemplate_x + \
#                 '%{x}<br>' + hovertemplate_y + '%{customdata[0]}'
#     else:
#         if pd.api.types.is_numeric_dtype(config['df'][config['y']]):
#             hovertemplate = hovertemplate_x + \
#                 '%{x}<br>' + hovertemplate_y + '%{customdata[0]}'
#         else:
#             hovertemplate = hovertemplate_x + \
#                 '%{customdata[0]}<br>' + hovertemplate_y + '%{y}'
#     if config['category']:
#         hovertemplate += hovertemplate_color + '%{data.name}'
#     if config['show_group_size']:
#         hovertemplate += f'<br>Размер группы '
#         hovertemplate += '%{customdata[0]}'
#     # hovertemplate += f'<br>cnt_in_sum_pct = '
#     # hovertemplate += '%{customdata[1]}'
#     hovertemplate += '<extra></extra>'
#     fig.update_traces(hovertemplate=hovertemplate, hoverlabel=dict(bgcolor="white"), textfont=dict(
#         family='Noto Sans', size=config['textsize']  # Размер шрифта
#         # color='black'  # Цвет текста
#     ) # Положение текстовых меток (outside или inside))
#     )
#     fig.update_layout(
#         # , title={'text': f'<b>{title}</b>'}
#         # , margin=dict(l=50, r=50, b=50, t=70)
#         margin=dict(t=80),
#         width=config['width'], height=config['height'],
#         title={'text': config["title"]}, xaxis_title=xaxis_title, yaxis_title=yaxis_title,
#         title_font=dict(size=16, color="rgba(0, 0, 0, 0.7)"),
#         font=dict(size=14, family="Noto Sans", color="rgba(0, 0, 0, 0.7)"),
#         xaxis_title_font=dict(size=14, color="rgba(0, 0, 0, 0.7)"),
#         yaxis_title_font=dict(size=14, color="rgba(0, 0, 0, 0.7)"),
#         xaxis_tickfont=dict(size=14, color="rgba(0, 0, 0, 0.7)"),
#         yaxis_tickfont=dict(size=14, color="rgba(0, 0, 0, 0.7)"),
#         xaxis_linecolor="rgba(0, 0, 0, 0.4)",
#         yaxis_linecolor="rgba(0, 0, 0, 0.4)",
#         xaxis_tickcolor="rgba(0, 0, 0, 0.4)",
#         yaxis_tickcolor="rgba(0, 0, 0, 0.4)",
#         legend_title_font_color='rgba(0, 0, 0, 0.7)',
#         legend_title_font_size = 14,
#         legend_font_color='rgba(0, 0, 0, 0.7)',
#         hoverlabel=dict(bgcolor="white"), xaxis=dict(
#             visible=config['xaxis_show'], showgrid=config['showgrid_x'], gridwidth=1, gridcolor="rgba(0, 0, 0, 0.1)"
#         ), yaxis=dict(
#             visible=config['yaxis_show'], showgrid=config['showgrid_y'], gridwidth=1, gridcolor="rgba(0, 0, 0, 0.07)"
#         ),
#         legend=dict(
#             title_font_color="rgba(0, 0, 0, 0.5)", font_color="rgba(0, 0, 0, 0.5)"
#         )
#     )
#     return fig
            

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


def histogram_go(
    column: pd.Series,
    title: str = None,
    xaxis_title: str = None,
    yaxis_title: str = None,
    nbins: int = 30,
    width: int = 600,
    height: int = 400,
    left_quantile: float = 0,
    right_quantile: float = 1,
    show_marginal_box: bool = True
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
    if pd.api.types.is_numeric_dtype(column):
        trimmed_column = column.between(column.quantile(
            left_quantile), column.quantile(right_quantile))
        column = column[trimmed_column]
    if not title:
        title = f'Распределенеие {column.name}'
    if not xaxis_title:
        xaxis_title = 'Значение'
    if not yaxis_title:
        yaxis_title = 'Доля'
    
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
                histnorm='probability',
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
                histnorm='probability',
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
        font=dict(size=14, family="Noto Sans", color="rgba(0, 0, 0, 0.7)"),
        title_font=dict(size=16, color="rgba(0, 0, 0, 0.7)"),
        title_y=0.95,
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
    labels: str = None,
    width: int = None,
    height: int = None,
    horizontal_spacing: float = None,
    vertical_spacing: float = None,
    rows: int = None,
    cols: int = None,
    color: str = None,
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
    color : str, optional
        Category column for coloring the scatter plots
    legend_position : str, optional
        Position of the legend ('top' or 'right'). Default is 'top'

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
        fig_scatter = px.scatter(df, x=col1, y=col2, color=color, labels=labels)
        fig_scatter.update_traces(marker=dict(
            line=dict(color='white', width=0.5)))

        for trace, color_for_line in  zip(fig_scatter.data, colorway_for_line):
            trace.marker.color = color_for_line
            fig.add_trace(trace, row=row+1, col=col+1)
        fig.update_xaxes(
            title_text=labels[col1],
        )
        fig.update_yaxes(
            title_text=labels[col2],
        )
    # if color:
    #     if legend_position == 'top':
    #         fig.update_layout(
    #             legend = dict(
    #                 title_text=labels[color]
    #                 , title_font_color='rgba(0, 0, 0, 0.7)'
    #                 , font_color='rgba(0, 0, 0, 0.7)'
    #                 , orientation="h"  # Горизонтальное расположение
    #                 , yanchor="top"    # Привязка к верхней части
    #                 , y=1.05         # Положение по вертикали (отрицательное значение переместит вниз)
    #                 , xanchor="center" # Привязка к центру
    #                 , x=0.1              # Центрирование по горизонтали
    #             )
    #         )
    #     elif legend_position == 'right':
    #         fig.update_layout(
    #                 legend = dict(
    #                 title_text=labels[color]
    #                 , title_font_color='rgba(0, 0, 0, 0.7)'
    #                 , font_color='rgba(0, 0, 0, 0.7)'
    #                 , orientation="v"  # Горизонтальное расположение
    #                 # , yanchor="bottom"    # Привязка к верхней части
    #                 , y=0.8         # Положение по вертикали (отрицательное значение переместит вниз)
    #                 # , xanchor="center" # Привязка к центру
    #                 # , x=0.5              # Центрирование по горизонтали
    #             )
    #         )
    #     else:
    #         raise ValueError("Invalid legend_position. Please choose 'top' or 'right'.")
    fig_update(fig, height=height, width=width)
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
            font=dict(size=14, family="Noto Sans", color="rgba(0, 0, 0, 0.7)"),
            title_font=dict(size=16, color="rgba(0, 0, 0, 0.7)"),     
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
        family='Noto Sans', size=14, color="rgba(0, 0, 0, 0.7)"), hoverlabel=dict(bgcolor="white", font=dict(color='rgba(0, 0, 0, 0.7)', size=14)))
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
            font=dict(size=14, family="Noto Sans", color="rgba(0, 0, 0, 0.7)"),
            title_font=dict(size=16, color="rgba(0, 0, 0, 0.7)"),     
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
        crosstab = df.groupby(column_for_axis, observed=False).size().to_frame('share') 
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
                      , textfont=dict(family='Noto Sans', size=textsize))
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

def heatmap_no_wrapper(
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
        func_df =  pd.pivot_table(df, index=column_for_x, columns=column_for_y, values=column_for_value, aggfunc=func, observed=False)
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
            font=dict(size=14, family="Noto Sans", color="rgba(0, 0, 0, 0.7)"),
            title_font=dict(size=16, color="rgba(0, 0, 0, 0.7)"),     
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
        family='Noto Sans', size=14), hoverlabel=dict(bgcolor="white", font=dict(family='Noto Sans', color='rgba(0, 0, 0, 0.7)', size=14)))
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
                df_trim['density'] = df_trim.groupby(['x_sector', 'y_sector'], observed=False)[col1].transform('size')
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
        font=dict(size=14, family="Noto Sans", color="rgba(0, 0, 0, 0.7)"),
        title_font=dict(size=16, color="rgba(0, 0, 0, 0.7)"),     
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
    legend_title: str = None,
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
            , font=dict(size=14, family="Noto Sans", color="rgba(0, 0, 0, 0.7)")
            , title_font=dict(size=16, color="rgba(0, 0, 0, 0.7)")
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
            , font=dict(size=14, family="Noto Sans", color="rgba(0, 0, 0, 0.7)")
            , title_font=dict(size=16, color="rgba(0, 0, 0, 0.7)")   
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
    trimmed_df = df.groupby(columns_for_groupby, observed=False)[num_var].apply(trim_by_quantiles).reset_index()
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
        font=dict(size=14, family="Noto Sans", color="rgba(0, 0, 0, 0.7)"),
        title_font=dict(size=16, color="rgba(0, 0, 0, 0.7)"),     
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
        font=dict(size=14, family="Noto Sans", color="rgba(0, 0, 0, 0.7)"),
        title_font=dict(size=16, color="rgba(0, 0, 0, 0.7)"),     
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

def plot_ci(
    df: pd.DataFrame,
    categorical_col: str,
    numerical_col: str,
    second_categorical_col: str = None,
    confidence_level: float = 0.95,
    orientation: str = 'v',
    height: int = 300,
    width: int = 400,
    legend_position: str = 'top',
    title: str = None,
    xaxis_title: str = None, 
    yaxis_title: str = None,
    legend_title: str = None,
    labels: dict = None,
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

    if labels is None:
        # title = f'Среднее {numerical_col} в зависимости от {categorical_col} с {int(confidence_level*100)}% доверительными интервалами'
        xaxis_title = categorical_col
        yaxis_title = numerical_col
        legend_title = second_categorical_col if second_categorical_col else None
    else:
        xaxis_title = f'{labels[categorical_col]}'
        yaxis_title = f'{labels[numerical_col]}'
        legend_title = f'{labels[second_categorical_col]}' if second_categorical_col else None

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
    summary_df["ci"] = (t_score * summary_df["std"] / (summary_df["count"] ** 0.5)).round(2)
    summary_df["mean"] = summary_df["mean"].round(2)
    if df[categorical_col].nunique() == 2 and second_categorical_col is None:
        ticktext = summary_df[categorical_col].unique()
        tickvals = [0.25, 0.75]
        summary_df[categorical_col] = [0.25, 0.75]
    else:
        ticktext = None
        tickvals = None
    # print(tickvals, ticktext)
    # display(summary_df)
    # Определяем ориентацию графика
    if orientation == 'v':
        x_col = categorical_col
        y_col = "mean"
        hovertemplate = f'{yaxis_title}' + ' = %{y}<br>'
        if xaxis_title:
            hovertemplate += f'{xaxis_title} = ' + '%{x}<extra></extra>'
        else:
            hovertemplate += '%{x}<extra></extra>'
    elif orientation == 'h':
        x_col = "mean"
        y_col = categorical_col
        xaxis_title, yaxis_title = yaxis_title, xaxis_title
        hovertemplate = f'{yaxis_title}' + ' = %{x}<br>'
        if xaxis_title:
            hovertemplate ++ f'{xaxis_title} = ' + '%{y}<extra></extra>'
        else:
            hovertemplate += '%{x}<extra></extra>'
    else:
        raise ValueError("Ориентация должна быть 'v' или 'h'.")

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
                hovertemplate=hovertemplate,
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

        # Обновляем оси X: задаем метки и их позиции
        fig.update_xaxes(
            tickvals=tickvals
            , ticktext=ticktext
            , range=[0,1]
        )
    fig.update_traces(
        error_x=dict(width=10),  # Ширина черточек для ошибок по оси X
        error_y=dict(width=10)  # Ширина черточек для ошибок по оси Y
    )
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
    update_layout_config = dict(
        height=height,
        width=width,
        title=title,
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title
    )
    return fig_update(fig, **update_layout_config)

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
    ----------
    configs : list
        List of dictionaries containing configuration for each subplot.
        Each dictionary must contain the following keys:
        - fig : plotly.graph_objects.Figure
            The plotly figure object for the subplot.
        - layout : plotly.graph_objects.Layout
            The plotly layout object for the subplot.
        - row : int
            Row position of the subplot (1-indexed).
        - col : int
            Column position of the subplot (1-indexed).

        Optional keys:
        - is_margin : bool, default=False
            Boolean to indicate if the plot is a margin plot.
        - domain_x : list, default=None
            X-axis domain range as [min, max].
        - domain_y : list, default=None
            Y-axis domain range as [min, max].
        - showgrid_x : bool, default=True
            Show X-axis grid lines.
        - showgrid_y : bool, default=True
            Show Y-axis grid lines.
        - showticklabels_x : bool, default=True
            Show X-axis tick labels.
        - xaxis_visible : bool, default=True
            X-axis visibility.
        - yaxis_visible : bool, default=True
            Y-axis visibility.
        - show_yaxis_title : bool, default=True
            Whether to show the Y-axis title.

    title : str, default=''
        Main figure title.

    width : int, default=800
        Figure width in pixels.

    height : int, default=600
        Figure height in pixels.

    rows : int, default=1
        Number of rows in subplot grid.

    cols : int, default=1
        Number of columns in subplot grid.

    shared_xaxes : bool, default=False
        Share X axes between subplots.

    shared_yaxes : bool, default=False
        Share Y axes between subplots.

    horizontal_spacing : float, default=0.1
        Spacing between subplots horizontally (0 to 1).

    specs : list, default=None
        Subplot specifications, where each element defines the type of plot in that subplot.

    column_widths : list, default=None
        List of relative column widths.

    row_heights : list, default=None
        List of relative row heights.

    subplot_titles : list, default=None
        List of subplot titles.

    Returns:
    -------
    plotly.graph_objects.Figure
        The created figure with subplots.

    Example:
    -------
    configs = [
        dict(
            fig = fig_cnt.data[0]
            , layout = fig_cnt.layout
            , showgrid_y = False
            , row = 1
            , col = 1
        )
        , dict(
            fig = fig_sum.data[0]
            , layout = fig_sum.layout
            , showgrid_y = False
            , row = 1
            , col = 2
            , show_yaxis_title = False
        )
    ]
    """
    # Implementation of the function goes here

    # Create subplot layout
    # if subplot_titles is None:
    #     subplot_titles = []
    #     for fig in configs:
    #         subplot_titles.append(fig['layout'].title.text)
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
                if config['with_margin']:
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
                # visible=config['xaxis_visible'],
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
                # showgrid=config['showgrid_y'],
                gridwidth=1,
                gridcolor="rgba(0, 0, 0, 0.1)",
                visible=config['yaxis_visible'],
                title_text=config['yaxis_title_text']
            )
            fig.update_traces(selector=dict(type='pie'),
                  marker=dict(colors=colorway_for_bar))
            fig_update(fig, xaxis_showgrid = config['showgrid_x'], yaxis_showgrid = config['showgrid_y'])

    # Adjust subplot titles position
    if subplot_titles:
        for i, _ in enumerate(subplot_titles):
            fig['layout']['annotations'][i-1]['y'] = 1.04


    # Update figure layout
    fig.update_layout(
        title_text=title,
        width=width,
        height=height,
        margin =  dict(l=50, r=50, b=50, t=50),
        # font=dict(size=14, family="Noto Sans", color="rgba(0, 0, 0, 0.7)"),
        # title_font_size= 16,
        # xaxis_showticklabels=True,
        # xaxis_title_font=dict(size=14, color="rgba(0, 0, 0, 0.7)"),
        # yaxis_title_font=dict(size=14, color="rgba(0, 0, 0, 0.7)"),
        # xaxis_tickfont=dict(size=14, color="rgba(0, 0, 0, 0.7)"),
        # yaxis_tickfont=dict(size=14, color="rgba(0, 0, 0, 0.7)"),
        # xaxis_linecolor="rgba(0, 0, 0, 0.4)",
        # yaxis_linecolor="rgba(0, 0, 0, 0.4)",
        # xaxis_tickcolor="rgba(0, 0, 0, 0.4)",
        # yaxis_tickcolor="rgba(0, 0, 0, 0.4)",
        # legend_title_font_color='rgba(0, 0, 0, 0.7)',
        # legend_title_font_size=14,
        # legend_font_color='rgba(0, 0, 0, 0.7)',
        # hoverlabel=dict(bgcolor="white")
    )

    return fig

def pie_bar(
    data_frame: pd.DataFrame,
    agg_mode: str = None,
    agg_func: str = None,
    agg_column: str = None,
    resample_freq: str = None,
    norm_by: str = None,
    top_n_trim_x: int = None,
    top_n_trim_y: int = None,
    top_n_trim_color: int = None,
    top_n_trim_facet_col: int = None,
    top_n_trim_facet_row: int = None,
    top_n_trim_facet_animation_frame: int = None,
    sort_axis: bool = True,
    sort_color: bool = True,
    sort_facet_col: bool = True,
    sort_facet_row: bool = True,
    sort_animation_frame: bool = True,
    show_group_size: bool = False,
    min_group_size: int = None,
    decimal_places: int = 2,
    update_layout: bool = True,
    show_box: bool = False,
    show_count: bool = False,
    lower_quantile_for_box: float = None,
    upper_quantile_for_box: float = None,
    top_and_bottom: bool = False,
    show_ci: bool = False,
    trim_top_or_bottom: str = 'top',
    show_legend_title: bool = False,
    observed_for_groupby: bool = False,
    agg_func_for_top_n: bool = None,
    hole: float = None,
    label_for_others_in_pie: str = 'others',
    **kwargs
) -> go.Figure:
    """
    Creates a bar chart using the Plotly Express library. This function is a wrapper around Plotly Express bar and accepts all the same parameters, allowing for additional customization and functionality.

    Parameters
    ----------
    data_frame : pd.DataFrame, optional
        DataFrame containing the data to be plotted
    agg_mode : str, optional
        Aggregation mode. Options:
        - 'groupby': Group by categorical columns
        - 'resample': Resample time series data
        - None: No aggregation (default)
    agg_func : str, optional
        Aggregation function. Options: 'mean', 'median', 'sum', 'count', 'nunique'
    agg_column : str, optional
         Column to aggregate
    resample_freq : str, optional
        Resample frequency for resample. Options: 'ME', 'W', D' and others
    norm_by: str, optional
        The name of the column to normalize by. If set to 'all', normalization will be performed based on the sum of all values in the dataset.
    top_n_trim_x : int, optional
        Ontly for aggregation mode. The number of top categories x axis to include in the chart. For top using num column and agg_func.
    top_n_trim_y : int, optional
        Ontly for aggregation mode. The number of top categories y axis to include in the chart. For top using num column and agg_func
    top_n_trim_color : int, optional
        Ontly for aggregation mode. The number of top categories legend to include in the chart. For top using num column and agg_func
    top_n_trim_facet_col : int, optional
        Ontly for aggregation mode. The number of top categories in facet_col to include in the chart. For top using num column and agg_func
    top_n_trim_facet_row : int, optional
        Ontly for aggregation mode. The number of top categories in facet_row to include in the chart. For top using num column and agg_func
    top_n_trim_facet_animation_frame : int, optional
        Ontly for aggregation mode. The number of top categories in animation_frame to include in the chart. For top using num column and agg_func
    trim_top_or_bottom : str, optional
        Trim from bottom or from top. Default is 'top'
    sort_axis, sort_color, sort_facet_col, sort_facet_row, sort_animation_frame : bool, optional
        Controls whether to sort the corresponding dimension (axis, color, facet columns, facet rows, or animation frames) based on the sum of numeric values across each slice. When True (default), the dimension will be sorted in descending order by the sum of numeric values. When False, no sorting will be applied and the original order will be preserved.
    show_group_size : bool, optional
        Whether to show the group size (only for groupby mode). Default is False
    min_group_size : int, optional
        The minimum number of observations required in a category to include it in the calculation.
        Categories with fewer observations than this threshold will be excluded from the analysis.
        This ensures that the computed mean is based on a sufficiently large sample size,
        improving the reliability of the results. Default is None (no minimum size restriction).
    decimal_places : int, optional
        The number of decimal places to display in hover. Default is 2
    x : str, optional
        Column to use for the x-axis
    y : str, optional
        Column to use for the y-axis
    color : str, optional
        Column to use for color encoding
    pattern : str, optional
        Column to use for pattern encoding
    hover_name : str, optional
        Column to use for hover text
    hover_data : list, optional
        List of columns to use for hover data
    custom_data : list, optional
        List of columns to use for custom data
    text : str, optional
        Column to use for text
    facet_row : str, optional
        Column to use for facet row
    facet_col : str, optional
        Column to use for facet column
    facet_col_wrap : int, optional
        Number of columns to use for facet column wrap
    error_x : str, optional
        Column to use for x-axis error bars
    error_x_minus : str, optional
        Column to use for x-axis error bars (minus)
    error_y : str, optional
        Column to use for y-axis error bars
    error_y_minus : str, optional
        Column to use for y-axis error bars (minus)
    animation_frame : str, optional
        Column to use for animation frame
    animation_group : str, optional
        Column to use for animation group
    range_x : list, optional
        List of values to use for x-axis range
    range_y : list, optional
        List of values to use for y-axis range
    log_x : bool, optional
        Whether to use a logarithmic scale for the x-axis. Default is False
    log_y : bool, optional
        Whether to use a logarithmic scale for the y-axis. Default is False
    range_x_equals : list, optional
        List of values to use for x-axis range, with equal scaling. Default is None
    range_y_equals : list, optional
        List of values to use for y-axis range, with equal scaling. Default is None
    title : str, optional
        Title of the chart. Default is None
    template : str, optional
        Template to use for the chart. Default is None
    width : int, optional
        Width of the chart in pixels. Default is None
    height : int, optional
        Height of the chart in pixels. Default is None
    show_box : bool, optional
        Whether to show boxplot in subplots
    show_count : bool, optional
            Whether to show countplot in subplots
    lower_quantile : float, optional
        The lower quantile for filtering the data. Value should be in the range [0, 1].
    upper_quantile : float, optional
        The upper quantile for filtering the data. Value should be in the range [0, 1].
    show_ci : bool, optional
        Whether to show confidence intervals. Default is False
    top_and_bottom : bool, optional
        Whether to show only top or both top and bottom
    show_legend_title : bool, optional
        Whether to show legend title. Default is False
    observed_for_groupby : bool, optional
        This only applies if any of the groupers are Categoricals.
        If True: only show observed values for categorical groupers.
        If False: show all values for categorical groupers.
        default False
    agg_func_for_top_n : str, optional
        Aggregation function for top_n_trim. Options: 'mean', 'median', 'sum', 'count', 'nunique'.
        By default agg_func_for_top_n = agg_func
    hole : float, optional
        Size of pie hole. May be from 0 to 1.
    label_for_others_in_pie : str, optional
        Label for others part in pie
    **kwargs
        Additional keyword arguments to pass to the Plotly Express function. Default is None

    Returns
    -------
    go.Figure
        The created chart
    """
    config = {
        'top_n_trim_x': top_n_trim_x,
        'top_n_trim_y': top_n_trim_y,
        'top_n_trim_color': top_n_trim_color,
        'top_n_trim_facet_col': top_n_trim_facet_col,
        'top_n_trim_facet_row': top_n_trim_facet_row,
        'top_n_trim_facet_animation_frame': top_n_trim_facet_animation_frame,
        'agg_column': agg_column,
        'sort_axis': sort_axis,
        'sort_color': sort_color,
        'sort_facet_col': sort_facet_col,
        'sort_facet_row': sort_facet_row,
        'sort_animation_frame': sort_animation_frame,
        'agg_func': agg_func,
        'show_group_size': show_group_size,
        'agg_mode': agg_mode,
        'resample_freq': resample_freq,
        'decimal_places': decimal_places,
        'norm_by': norm_by,
        'update_layout': update_layout,
        'show_box': show_box,
        'show_count': show_count,
        'lower_quantile_for_box': lower_quantile_for_box,
        'upper_quantile_for_box': upper_quantile_for_box,
        'show_ci': show_ci,
        'top_and_bottom': top_and_bottom,
        'trim_top_or_bottom': trim_top_or_bottom,
        'min_group_size': min_group_size,
        'show_legend_title': show_legend_title,
        'observed_for_groupby': observed_for_groupby,
        'agg_func_for_top_n': agg_func_for_top_n,
        'hole': hole,
    }
    config = {k: v for k,v in config.items() if v is not None}
    if kwargs.get('color_discrete_sequence') is None:
        kwargs['color_discrete_sequence'] = colorway_for_bar[1:]
    labels = kwargs.get('labels', None)
    if 'agg_column' in config:
        num_column = config['agg_column']
        if kwargs['x'] == num_column:
            cat_column_axis = kwargs['y']
        else:
            cat_column_axis = kwargs['x']
    else:
        if pd.api.types.is_numeric_dtype(data_frame[kwargs['x']]):
            num_column = kwargs['x']
            cat_column_axis = kwargs['y']
        else:
            num_column = kwargs['y']
            cat_column_axis = kwargs['x']
    if config.get('agg_func') is None:
        raise ValueError('agg_func must be defined')
    df_for_pie = data_frame.groupby(cat_column_axis, observed=config.get('observed_for_groupby'))[num_column].agg(config['agg_func'])
    max_cat_for_exclude = df_for_pie.idxmax()
    df_for_pie = df_for_pie.reset_index()
    df_for_pie['new_cat_column'] = df_for_pie[cat_column_axis].apply(lambda x: max_cat_for_exclude if x == max_cat_for_exclude else 'others')
    df_for_pie = df_for_pie.sort_values(num_column, ascending=False)
    labels['new_cat_column'] = kwargs.get('labels').get(cat_column_axis)
    pie_fig = px.pie(df_for_pie
                     , values=num_column
                     , names='new_cat_column'
                     , labels=labels
                     , hover_data = {num_column: False} if norm_by == 'all' else None
                     , hole=hole
    )
    # pie_fig.update_traces(textinfo='value')
    df_for_bar = data_frame[data_frame[cat_column_axis] != max_cat_for_exclude]
    bar_fig = _create_base_fig_for_bar_line_area(df=df_for_bar, config=config, kwargs=kwargs, graph_type='bar')
    configs = [
        dict(
            fig = pie_fig.data[0]
            , layout = pie_fig.layout
            , row = 1 , col = 1
            )
        , dict(
            fig = bar_fig.data[0]
            , layout = bar_fig.layout
            , showgrid_y = False
            , show_yaxis_title = False
            , row = 1, col = 2
        )
    ]
    fig = subplots(configs
              , specs=[[{'type': 'pie'}, {'type': 'bar'}]]
              , title = kwargs.get('title', None)
    )
    width =  kwargs.get('width')
    if width is None:
        width = 900
    height =  kwargs.get('height')
    if height is None:
        height = 400
    fig.update_layout(legend_y=-0.1
                      , legend_x=0.11
                      , title_y=0.96
                      , margin = dict(l=50, r=50, b=50, t=70)
                      , legend_orientation='h'
                      , width=width
                      , height=height
    )
    return fig

def qqplot_plotly(x, show_skew_curt=True, **kwargs):
    """
    Create an interactive Q-Q plot using Plotly

    Parameters:
    -----------
    x : array_like
        Sample data.

    Returns:
    --------
    fig : plotly.graph_objects.Figure
        An interactive Plotly figure displaying the Q-Q plot.

    """
    # Create a Q-Q plot using pingouin
    if not isinstance(x, pd.Series):
        x = pd.Series(x)
    skew = x.skew()
    kurt = x.kurtosis()
    fig = sm_qqplot(
        x
        , line='s'
        , marker='o'
        , markersize=5
        # , markeredgecolor='white'
        # , markeredgewidth=0.5
    )
    # Close the matplotlib plot to prevent it from displaying
    plt.close()

    qqplot_data = fig.gca().lines

    # Извлечение точек и линии
    points_x = qqplot_data[0].get_xdata()  # Теоретические квантили
    points_y = qqplot_data[0].get_ydata()  # Эмпирические квантили
    red_line_x = qqplot_data[1].get_xdata()  # Линия теоретических квантилей
    red_line_y = qqplot_data[1].get_ydata()  # Линия эмпирических квантилей
    point_color ='rgba(25, 108, 181, 0.9)'
    line_color ='rgba(200, 0, 0, 0.9)'
    # Create a Plotly figure
    fig = go.Figure()

    # Add points (Ordered quantiles)
    fig.add_trace(go.Scatter(
        x=points_x, y=points_y, mode='markers',
        marker=dict(color=point_color, size=8),
        hovertemplate='%{x:.2f}, %{y:.2f}<extra></extra>'
    ))

    # Add reference line
    fig.add_trace(go.Scatter(
        x=red_line_x, y=red_line_y, mode='lines',
        line=dict(color=line_color, width=2),
        hoverinfo='none'
    ))

    # Set axis ranges
    x_min, x_max = min(points_x), max(points_x)
    y_min, y_max = min(points_y), max(points_y)
    x_padding = (x_max - x_min) * 0.1  # 10% of the range
    y_padding = (y_max - y_min) * 0.1  # 10% of the range

    # Относительные координаты (например, 95% по X, 5% по Y)
    x_rel = 1.02
    y_rel = -0.01

    # Преобразуем относительные координаты в данные
    x_annotation = x_min + (x_max - x_min) * x_rel
    y_annotation = y_min + (y_max - y_min) * y_rel

    fig.add_annotation(
        x=x_annotation,  # Координата X (0.95 = 95% от ширины графика)
        y=y_annotation,  # Координата Y (0.05 = 5% от высоты графика)
        xref='x',  # Используем относительные координаты по оси X (0-1)
        yref='y',  # Используем относительные координаты по оси Y (0-1)
        text=f"Skew: {skew:.2f}<br>Kurt: {kurt:.2f}",  # Текст аннотации
        showarrow=False,  # Не показывать стрелку
        xanchor='right',  # Выравнивание текста по правому краю
        yanchor='bottom',  # Выравнивание текста по нижнему краю
        align='right',
        # font=dict(size=12, color="black"),  # Настройки шрифта
        # bgcolor="white",  # Фон текста
        # bordercolor="black",  # Цвет границы
        # borderwidth=1  # Толщина границы
    )
    # Update figure layout
    fig_update(fig
                    # , xaxis_showgrid=False
                    # , yaxis_showgrid=False
                    , showlegend=False
                    , title='Q-Q Plot'
                    , xaxis_title='Theoretical Quantiles'
                    , yaxis_title='Ordered Quantiles'
                    , width=500
                    , height=400
                    , xaxis_range=[x_min - x_padding, x_max + x_padding]
                    , yaxis_range=[y_min - y_padding, y_max + y_padding]
                    , margin = dict(l=20, r=20, b=20, t=50)
                    # , xaxis_dtick=0.5
                    # , yaxis_dtick=0.5
    )

    return fig

def qqplot(data, numeric_col=None, facet_col=None, facet_col_wrap=None, plot_width=4, plot_height=3, show_skew_curt=True):
    """
    Plots Q-Q plots with custom styling, optionally faceted by a categorical column.

    Parameters:
        data (pd.DataFrame or pd.Series): The input DataFrame.
        numeric_col (str): The column name for the numeric data to plot.
        facet_col (str, optional): The column name for faceting the plots by a categorical variable.
        facet_col_wrap (int, optional): The number of subplots per row for faceting.
        plot_width (float): The width of each individual plot.
        plot_height (float): The height of each individual plot.
    """
    def _style_qqplot(ax, font_color, line_color, grid_color, font_family, grid_width, title_font_size, font_size, tick_font_size):
        """
        Applies custom styling to a Q-Q plot.

        Parameters:
            ax (matplotlib.axes.Axes): The axes object to style.
            font_color (str): Color for text.
            line_color (str): Color for lines.
            grid_color (str): Color for the grid.
            font_family (str): Font family for text.
            grid_width (float): Width of the grid lines.
            title_font_size (int): Font size for the title.
            font_size (int): Font size for axis labels.
            tick_font_size (int): Font size for tick labels.
        """
        # Remove top and right spines (borders)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(True)
        ax.spines['bottom'].set_visible(True)
        ax.spines['left'].set_color(line_color)
        # ax.spines['bottom'].set_color(line_color)
        # Add a light gray grid
        ax.grid(True, color=grid_color, linestyle='-', linewidth=grid_width)

        # Set the axis labels with custom styling
        ax.set_xlabel('Theoretical Quantiles', fontsize=font_size, fontfamily=font_family, color=font_color)
        ax.set_ylabel('Ordered Quantiles', fontsize=font_size, fontfamily=font_family, color=font_color)

        # Customize tick parameters (color and font size)
        ax.tick_params(axis='both', which='major', colors=line_color, labelsize=tick_font_size)  # Major tick
        ax.tick_params(axis='both', which='minor', colors=line_color, labelsize=tick_font_size)  # Minor ticks

    # Constants for styling
    FONT_COLOR = "#4d4d4d"  # rgba(0, 0, 0, 0.7)
    LINE_COLOR = "#666666"  # rgba(0, 0, 0, 0.4)
    GRID_COLOR = "#e6e6e6"  # rgba(0, 0, 0, 0.1)
    FONT_FAMILY = 'Noto Sans'
    GRID_WIDTH = 0.7
    TITLE_FONT_SIZE = 12
    FONT_SIZE = 10
    TICK_FONT_SIZE = 8

    # If no faceting, create a single Q-Q plot
    if facet_col is None:
        if isinstance(data, pd.DataFrame):
            if not numeric_col or numeric_col not in data.columns:
                raise ValueError('If data is pandas DataFrame then numeric_col must be in data.columns')
            data = data[numeric_col]
        fig = sm_qqplot(
            data
            , line='s'
            , marker='o'
            , markersize=5
            # , markeredgecolor='white'
            # , markeredgewidth=0.5
        )
        ax = fig.gca()
        _style_qqplot(ax, FONT_COLOR, LINE_COLOR, GRID_COLOR, FONT_FAMILY, GRID_WIDTH, TITLE_FONT_SIZE, FONT_SIZE, TICK_FONT_SIZE)
        ax.set_title('Q-Q Plot', fontsize=TITLE_FONT_SIZE, fontfamily=FONT_FAMILY, color=FONT_COLOR, loc='left', pad=20)
        if show_skew_curt:
            skewness = data.skew().round(2)
            kurt = data.kurtosis().round(2)

            # Добавляем текст с skewness и kurtosis справа внизу
            ax.text(
                0.9, 0.1,  # Координаты (x, y) в относительных единицах (0-1)
                f"Skew: {skewness:.2f}\nKurt: {kurt:.2f}",  # Текст
                transform=ax.transAxes,  # Используем относительные координаты
                fontsize=9,  # Размер шрифта
                color=FONT_COLOR,  # Цвет текста
                ha='right',  # Выравнивание по правому краю
                va='bottom',  # Выравнивание по нижнему краю
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='none')  # Фон текста
            )
        plt.show()
    else:
        if not isinstance(data, pd.DataFrame) or not numeric_col:
            raise ValueError('For facet_col data must be pandas DataFrame and numeric_col must be defined')
        # Get unique categories for faceting
        categories = data[facet_col].unique()
        n_categories = len(categories)

        # Determine the layout of subplots
        if facet_col_wrap is None:
            n_rows = int(np.ceil(n_categories / 3))  # Default to 3 columns
            n_cols = 3
        else:
            n_rows = int(np.ceil(n_categories / facet_col_wrap))
            n_cols = facet_col_wrap

        # Calculate the total figure size
        n_rows = int(np.ceil(len(categories) / n_cols))  # Number of rows
        figsize = (n_cols * plot_width, n_rows * plot_height)

        # Create the figure and subplots
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        axes = axes.flatten()  # Flatten the axes array for easy iteration
        # fig.suptitle('Q-Q Plot', fontsize=TITLE_FONT_SIZE, fontfamily=FONT_FAMILY, color=FONT_COLOR, x=0.01, y=1.01)
        fig.text(0.05, 1, 'Q-Q Plot', fontsize=TITLE_FONT_SIZE, fontfamily=FONT_FAMILY, color=FONT_COLOR, ha='left')
        # Plot Q-Q plots for each category
        for i, category in enumerate(categories):
            ax = axes[i]
            subset = data[data[facet_col] == category]
            sm_qqplot(
                subset[numeric_col],
                line='s',
                ax=ax,
                marker='o',
                markersize=5,  # Размер точек
                # markeredgecolor='white',  # Цвет границы точек
                # markeredgewidth=0.5  # Толщина границы точек
            )
            _style_qqplot(ax, FONT_COLOR, LINE_COLOR, GRID_COLOR, FONT_FAMILY, GRID_WIDTH, TITLE_FONT_SIZE, FONT_SIZE, TICK_FONT_SIZE)
            ax.set_title(category, fontsize=FONT_SIZE, fontfamily=FONT_FAMILY, color=FONT_COLOR, pad=0)
            if show_skew_curt:
                skewness = subset[numeric_col].skew().round(2)
                kurt = subset[numeric_col].kurtosis().round(2)
                # Добавляем текст с skewness и kurtosis справа внизу
                ax.text(
                    0.9, 0.1,  # Координаты (x, y) в относительных единицах (0-1)
                    f"Skew: {skewness:.2f}\nKurt: {kurt:.2f}",  # Текст
                    transform=ax.transAxes,  # Используем относительные координаты
                    fontsize=9,  # Размер шрифта
                    color=FONT_COLOR,  # Цвет текста
                    ha='right',  # Выравнивание по правому краю
                    va='bottom',  # Выравнивание по нижнему краю
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor='none')  # Фон текста
                )
            if i % n_cols != 0:
                ax.set_ylabel('')
            if i < len(categories) - n_cols:
                ax.set_xlabel('')
        # Hide unused subplots
        for j in range(i + 1, len(axes)):
            axes[j].axis('off')

        plt.tight_layout(pad=2)
        plt.show()

def histogram_old(
    data_frame: pd.DataFrame = None,
    lower_quantile: float = 0,
    upper_quantile: float = 1,
    norm_by: str = None,
    sort: bool = None,
    dual: bool = False,
    show_qqplot: bool = False,
    render_png: bool = False,
    **kwargs
) -> go.Figure:
    """
    Creates a histogram chart using the Plotly Express library. This function is a wrapper around Plotly Express bar and accepts all the same parameters, allowing for additional customization and functionality.

    Parameters
    ----------
    data_frame : pd.DataFrame, optional
        DataFrame containing the data to be plotted
    lower_quantile : float, optional
        The lower quantile for data filtering (default is 0).
    upper_quantile : float, optional
        The upper quantile for data filtering (default is 1).
    norm_by : str, optional
        Column name to normalize the histogram by.
        If specified, the histogram will be normalized based on this column.
    sort: bool, optional
        Whether to sort the categories in the histogram.
        If True, categories will be ordered based on their frequency.
    x : str, optional
        The name of the column in `data_frame` to be used for the x-axis. If not provided, the function will attempt to use the first column.
    y : str, optional
        The name of the column in `data_frame` to be used for the y-axis. If not provided, the function will count occurrences.
    color : str, optional
        The name of the column in `data_frame` to be used for color encoding. This will create a separate histogram for each unique value in this column.
    barmode : str, optional
        The mode for the bars in the histogram. Options include 'group', 'overlay', and 'relative'. Default is 'overlay'.
    nbins : int, optional
        The number of bins to use for the histogram. If not specified, the function will automatically determine the number of bins.
    histnorm : str, optional
        Normalization method for the histogram. Options include 'percent', 'probability', 'density', and 'probability density'. Default is None (no normalization).
    barnorm : str, optional
        Specifies how to normalize the heights of the bars in the histogram. Possible values include:
        - 'fraction': normalizes the heights of the bars so that the sum of all heights equals 1 (fraction of the total count).
        - 'percent': normalizes the heights of the bars so that the sum of all heights equals 100 (percentage of the total count).
        - 'density': normalizes the heights of the bars so that the area under the histogram equals 1 (probability density).
        - None: by default, the heights of the bars are not normalized.
    marginal : str, optional
        If set, adds a marginal histogram or box plot to the figure. Options include 'rug', 'box', and 'violin'.
    template : str, optional
        The name of the template to use for the figure. Default is None, which uses the default Plotly template.
    title : str, optional
        The title of the histogram. Default is None.
    labels : dict, optional
        A dictionary mapping column names to labels for the axes and legend.
    dual: bool, optional
        Whether to show 2 graphs, left origin, right trimmed by quantile
    **kwargs : dict
        Any additional keyword arguments accepted by `px.histogram`. This includes parameters like `opacity`, `hover_data`, `text`, `category_orders`, and more.

    Returns
    -------
    go.Figure
        Interactive Plotly histogram figure with custom hover labels and layout adjustments.
    """

    # Set default values for the figure dimensions and bar mode
    if dual or show_qqplot:
        kwargs.setdefault('width', 900)
    else:
        kwargs.setdefault('width', 600)
    kwargs.setdefault('height', 400)
    kwargs.setdefault('barmode', 'overlay')
    kwargs.setdefault('nbins', 30)
    kwargs.setdefault('marginal', 'box')
    yaxis_domain = None
    # Extract x, y, and color parameters from kwargs
    x = kwargs.get('x')
    y = kwargs.get('y')
    color = kwargs.get('color')
    # Set histogram normalization to 'probability' if no color is specified
    if not color:
        kwargs.setdefault('histnorm', 'probability')

    # Adjust normalization based on the norm_by parameter and color
    if norm_by and color is not None:
        if norm_by in [x, y]:
            kwargs['barnorm'] = 'fraction'
        if norm_by == color:
            kwargs['histnorm'] = 'probability'
    elif color is not None:
        kwargs['histnorm'] = 'probability'
    if dual:
        kwargs_dual = kwargs.copy()
    # If quantiles are provided, trim the data based on these quantiles
    if lower_quantile or upper_quantile:
        if isinstance(x, str):
            if pd.api.types.is_numeric_dtype(data_frame[x]):
                # Trim x based on the specified quantiles
                lower_quantile_x = data_frame[x].quantile(lower_quantile)
                upper_quantile_x = data_frame[x].quantile(upper_quantile)
                data_frame = data_frame[data_frame[x].between(lower_quantile_x, upper_quantile_x)]
        else:
            if pd.api.types.is_numeric_dtype(x):
                # Trim x based on the specified quantiles
                trimmed_column = x.between(x.quantile(lower_quantile), x.quantile(upper_quantile))
                x = x[trimmed_column]
        if isinstance(y, str):
            if pd.api.types.is_numeric_dtype(data_frame[y]):
                # Trim x based on the specified quantiles
                lower_quantile_y = data_frame[y].quantile(lower_quantile)
                upper_quantile_y = data_frame[y].quantile(upper_quantile)
                data_frame = data_frame[data_frame[y].between(lower_quantile_y, upper_quantile_y)]
        else:
            if pd.api.types.is_numeric_dtype(y):
                # Trim x based on the specified quantiles
                trimmed_column = y.between(y.quantile(lower_quantile), y.quantile(upper_quantile))
                y = y[trimmed_column]
        if dual:
            kwargs_dual['x'] = x
            kwargs_dual['y'] = y
        else:
            kwargs['x'] = x
            kwargs['y'] = y
    # If sorting is requested, prepare category orders for x and y
    if sort:
        category_orders = dict()
        if x is not None:
            if isinstance(x, str):
                x = data_frame[x]
            x_name = x.name
            # Get the order of categories based on value counts
            category_orders_x = x.value_counts().index.tolist()
            if isinstance(x, str):
                category_orders[x_name] = category_orders_x
            else:
                category_orders['x'] = category_orders_x
            # If color is specified, get the order of categories for color based on the top x category
            if color:
                top_x = category_orders_x[0]
                category_orders_color = data_frame[data_frame[x_name] == top_x][color].value_counts().index.tolist()
                category_orders[color] = category_orders_color

        if y is not None:
            if isinstance(y, str):
                y = data_frame[y]
            y_name = y.name
            # Get the order of categories for y based on value counts
            category_orders_y = y.value_counts().index.tolist()
            if isinstance(y, str):
                category_orders[y_name] = category_orders_y
            else:
                category_orders['y'] = category_orders_y
            # If color is specified, get the order of categories for color based on the top y category
            if color:
                top_y = category_orders_y[0]
                category_orders_color = data_frame[data_frame[y_name] == top_y][color].value_counts().index.tolist()
                category_orders[color] = category_orders_color

        # Set the category orders in kwargs
        kwargs['category_orders'] = category_orders

    # Create the histogram figure using Plotly Express
    fig = px.histogram(data_frame, **kwargs)
    # Update hover templates for better readability
    for trace in fig.data:
        trace.hovertemplate = trace.hovertemplate.replace('probability', 'Доля')  # Replace 'probability' with 'Доля'
        trace.hovertemplate = trace.hovertemplate.replace('count', 'Количество')  # Replace 'count' with 'Количество'
        if x is not None and kwargs.get('histnorm') is not None:
            trace.hovertemplate = trace.hovertemplate.replace('{y}', '{y:.2f}')  # Format y values
        if y is not None and kwargs.get('histnorm') is not None:
            trace.hovertemplate = trace.hovertemplate.replace('{x}', '{x:.2f}')  # Format x values
    if dual == True:
        if kwargs.get('marginal') is None:
            fig_subplots = make_subplots(rows=1, cols=2, horizontal_spacing=0.07)
            fig_subplots.add_trace(fig.data[0], row=1, col=2)
            fig_subplots.add_trace(
                go.Histogram(
                    x=kwargs_dual['x'],
                    nbinsx=kwargs['nbins'],
                    histnorm=kwargs['histnorm'],
                    marker_color='rgba(128, 60, 170, 0.9)',
                    hovertemplate=fig.data[0].hovertemplate
                ),
                row=1, col=1
            )
            fig_subplots.update_xaxes(title_text=x_for_box_hovertemplate, row=1, col=1)
            fig_subplots.update_xaxes(title_text=x_for_box_hovertemplate, row=1, col=2)
            if kwargs.get('histnorm') == 'probability':
                fig_subplots.update_yaxes(title_text='Доля', row=1, col=1)  # Set x-axis title to 'Доля' for probability
            if kwargs.get('histnorm') is None:
                fig_subplots.update_yaxes(title_text='Количество', row=1, col=1)  # Set x-axis title to 'Количество' for count
        else:
            fig_subplots = make_subplots(rows=2, cols=2, horizontal_spacing=0.07)
            fig_subplots.add_trace(fig.data[0], row=2, col=2)
            x_for_box_hovertemplate = kwargs['labels']['x'] if kwargs.get('labels') is not None else 'x'
            fig_subplots.add_trace(
                go.Box(
                    x=kwargs['x'],
                    marker_color='rgba(128, 60, 170, 0.9)',
                    hovertemplate=f'{x_for_box_hovertemplate} = ' + '%{x}<extra></extra>'
                ),
                row=1, col=2
            )
            fig_subplots.add_trace(
                go.Histogram(
                    x=kwargs_dual['x'],
                    nbinsx=kwargs['nbins'],
                    histnorm=kwargs['histnorm'],
                    marker_color='rgba(128, 60, 170, 0.9)',
                    hovertemplate=fig.data[0].hovertemplate
                ),
                row=2, col=1
            )
            fig_subplots.add_trace(
                go.Box(
                    x=kwargs_dual['x'],
                    marker_color='rgba(128, 60, 170, 0.9)',
                    hovertemplate=f'{x_for_box_hovertemplate} = ' + '%{x}<extra></extra>'
                ),
                row=1, col=1
            )
            fig_subplots.update_layout(
                yaxis1 = dict(
                    domain=[0.93, 1]
                    , visible = False
                )
                , yaxis2 = dict(
                    domain=[0.93, 1]
                    , visible = False
                )
                , yaxis3 = dict(
                    domain=[0, 0.9]
                )
                , yaxis4 = dict(
                    domain=[0, 0.9]
                )
                , xaxis2 = dict(
                    visible=False
                )
                , xaxis1 = dict(
                    visible=False
                )
            )
            fig_subplots.update_xaxes(title_text=x_for_box_hovertemplate, row=2, col=1)
            fig_subplots.update_xaxes(title_text=x_for_box_hovertemplate, row=2, col=2)
            if kwargs.get('histnorm') == 'probability':
                fig_subplots.update_yaxes(title_text='Доля', row=2, col=1)  # Set x-axis title to 'Доля' for probability
            if kwargs.get('histnorm') is None:
                fig_subplots.update_yaxes(title_text='Количество', row=2, col=1)  # Set x-axis title to 'Количество' for count
        fig_subplots.update_layout(title_text=kwargs.get('title'))
        fig = fig_subplots
        fig.update_traces(showlegend=False)
        fig.update_layout(height=kwargs['height'], width=kwargs['width'])

    else:
        if kwargs.get('marginal') is not None:
            if kwargs.get('color'):
                yaxis_domain = [0, 0.8]
                fig.update_layout(
                    yaxis2 = dict(
                        domain=[0.85, 0.93]
                        , visible = False
                    )
                    , xaxis2 = dict(
                        visible=False
                    )
                )
            else:
                yaxis_domain = [0, 0.9]
                fig.update_layout(
                    yaxis2 = dict(
                        domain=[0.93, 1]
                        , visible = False
                    )
                    , xaxis2 = dict(
                        visible=False
                    )
                )
        if show_qqplot:
            if x is not None:
                if isinstance(x, str):
                    data_for_qqplot = data_frame[x]
                else:
                    data_for_qqplot = x
            elif y is not None:
                if isinstance(y, str):
                    data_for_qqplot = data_frame[x]
                else:
                    data_for_qqplot = y
            else:
                raise ValueError('For qqplot must be define x or y, not both')
            fig_subplots = make_subplots(rows=1, cols=2, horizontal_spacing=0.1)
            fig_subplots.add_trace(fig.data[0], row=1, col=1)
            qqplot = qqplot_plotly(data_for_qqplot)
            for trace in qqplot.data:
                fig_subplots.add_trace(
                    trace
                    , row=1, col=2
                )
            for annotation in qqplot.layout.annotations:
                # Обновляем ссылки на оси (xref и yref) для подграфика
                annotation.update(xref='x2', yref='y2')
                fig_subplots.add_annotation(annotation)
            fig_subplots.update_xaxes(title_text='Theoretical Quantiles', row=1, col=2)
            fig_subplots.update_yaxes(title_text='Ordered Quantiles', row=1, col=2)
            if kwargs.get('histnorm') == 'probability':
                fig_subplots.update_yaxes(title_text='Доля', row=1, col=1)  # Set x-axis title to 'Доля' for probability
            if kwargs.get('histnorm') is None:
                fig_subplots.update_yaxes(title_text='Количество', row=1, col=1)  # Set x-axis title to 'Количество' for count
            fig_subplots.update_layout(title_text=kwargs.get('title'))
            fig = fig_subplots
            fig.update_traces(showlegend=False)
            fig.update_layout(height=kwargs['height'], width=kwargs['width'])
    # Update axis titles based on normalization and sorting
    num_rows = len(fig._grid_ref)
    if y is not None:
        if kwargs.get('histnorm') == 'probability':
            for row in range(1, num_rows + 1):
                fig.update_xaxes(title_text='Доля', row=row, col=1)
        if kwargs.get('histnorm') is None:
            for row in range(1, num_rows + 1):
                fig.update_xaxes(title_text='Количество', row=row, col=1)
        if sort:
            fig.data = fig.data[::-1]  # Reverse the order of traces if sorting
            fig.update_layout(legend={'traceorder': 'reversed'})  # Reverse legend order

    if x is not None:
        if kwargs.get('histnorm') == 'probability':
            for row in range(1, num_rows + 1):
                fig.update_yaxes(title_text='Доля', row=row, col=1)
        if kwargs.get('histnorm') is None:
            for row in range(1, num_rows + 1):
                fig.update_yaxes(title_text='Количество', row=row, col=1)
    # Update the figure with any additional modifications
    fig_update_config = dict()
    if  x is not None:
        if isinstance(x, str):
            is_x_numeric = pd.api.types.is_numeric_dtype(data_frame[x])
        else:
            is_x_numeric = pd.api.types.is_numeric_dtype(x)
        if not is_x_numeric:
            fig_update_config['xaxis_showgrid'] = False
    if y is not None:
        if isinstance(y, str):
            is_x_numeric = pd.api.types.is_numeric_dtype(data_frame[y])
        else:
            is_x_numeric = pd.api.types.is_numeric_dtype(y)
        if not is_x_numeric:
            fig_update_config['yaxis_showgrid'] = False
    if kwargs.get('color'):
        fig_update_config['legend_position'] = 'top'
        fig_update_config['legend_title'] = ''
    if yaxis_domain:
        fig_update_config['yaxis_domain'] = yaxis_domain
    fig = fig_update(fig, **fig_update_config)
    if render_png:
        fig.show(config=dict(displayModeBar=False), renderer="png")
    else:
        return fig  # Return the final figure

def distplot(
    data_frame: pd.DataFrame,
    num_col: str,
    cat_col: str,
    labels: dict = None,
    lower_quantile: float = 0,
    upper_quantile: float = 1,
    sort: bool = None,
    show_box: bool = True,
    legend_position: str = 'top',
    height: int = None,
    width: int = 800,
    colorway: list = None,
    title: str = None,
    category_orders: dict = None,
) -> go.Figure:
    """
    Creates a distribution plot with optional box plots using Plotly.

    This function generates a distribution plot (KDE) for numerical data grouped by a categorical column.
    It also supports adding box plots for each category and customizing the legend position, colors, and layout.

    Parameters
    ----------
    data_frame : pd.DataFrame
        DataFrame containing the data to be plotted.
    num_col : str
        Name of the numerical variable to plot.
    cat_col : str
        Name of the categorical variable for grouping.
    labels : dict, optional
        A dictionary mapping column names to labels for the axes and legend.
    lower_quantile : float, optional
        The lower quantile for data filtering (default is 0).
    upper_quantile : float, optional
        The upper quantile for data filtering (default is 1).
    sort : bool, optional
        Whether to sort the categories (default is None).
    show_box : bool, optional
        Whether to show box plots for each category (default is True).
    legend_position : str, optional
        Position of the legend ('top' or 'right', default is 'top').
    height : int, optional
        Height of the plot in pixels (default is None).
    width : int, optional
        Width of the plot in pixels (default is None).
    colorway : list, optional
        List of colors for lines and boxes (default is None).
    title : str, optional
        Title of the plot (default is None).
    category_orders : dict, optional
        Dictionary specifying the order of categories (default is None).

    Returns
    -------
    go.Figure
        Interactive Plotly figure with the distribution plot and optional box plots.
    """


    # Validate input data
    if not isinstance(data_frame, pd.DataFrame):
        raise TypeError("data_frame must be a pandas DataFrame.")

    if num_col not in data_frame.columns:
        raise ValueError(f"Column '{num_col}' not found in the DataFrame.")

    if cat_col not in data_frame.columns:
        raise ValueError(f"Column '{cat_col}' not found in the DataFrame.")

    if category_orders is not None and not isinstance(category_orders, dict):
        raise TypeError("category_orders must be a dictionary.")

    if colorway is not None and not isinstance(colorway, list):
        raise TypeError("colorway must be a list of colors.")

    # Group data by categories
    distplot_data = [data_frame[data_frame[cat_col] == cat][num_col] for cat in data_frame[cat_col].unique()]
    # Trim data based on quantiles
    if upper_quantile != 1 or lower_quantile != 0:
        if not (0 <= lower_quantile <= 1 and 0 <= upper_quantile <= 1):
            raise ValueError("lower_quantile and upper_quantile must be between 0 and 1.")
        if lower_quantile > upper_quantile:
            raise ValueError("lower_quantile must be less than or equal to upper_quantile.")

        distplot_data_trimmed = []
        for data in distplot_data:
            # for kde function need at leest 2 values
            if len(data) <= 2:
                distplot_data_trimmed.append(data)
                continue
            lower_bound = data.quantile(lower_quantile)
            upper_bound = data.quantile(upper_quantile)
            trimmed_data = data[(data >= lower_bound) & (data <= upper_bound)]
            distplot_data_trimmed.append(trimmed_data)
        distplot_data = distplot_data_trimmed
    group_labels = data_frame[cat_col].unique()
    # Validate and apply category order if provided
    if category_orders is not None and cat_col in category_orders:
        ordered_categories = category_orders[cat_col]

        # Check if all categories in ordered_categories are present in the data
        missing_in_data = set(ordered_categories) - set(group_labels)
        if missing_in_data:
            raise ValueError(f"Categories {missing_in_data} from category_orders are missing in the data.")

        # Check if all unique categories in the data are present in ordered_categories
        missing_in_order = set(group_labels) - set(ordered_categories)
        if missing_in_order:
            raise ValueError(f"Categories {missing_in_order} from the data are missing in category_orders.")

        # Create a mapping from group_labels to their indices
        labels_map = {label: index for index, label in enumerate(group_labels)}

        # Reorder distplot_data based on ordered_categories
        distplot_data = [distplot_data[labels_map[cat]] for cat in ordered_categories]
        group_labels = ordered_categories

    # Set default colorway if not provided
    if colorway is None:
        colorway = colorway_for_line
    # Create the distribution plot
    fig = ff.create_distplot(
        distplot_data,
        group_labels,
        show_hist=False,  # Show histogram
        show_curve=True,  # Show density curve
        show_rug=False,   # Hide rug plot
        colors=colorway
    )
    # Update hover template with axis labels
    if labels:
        xaxis_title = labels[num_col]
    else:
        xaxis_title = None
    fig.update_traces(
        hovertemplate=f"{xaxis_title} = %{{x}}<br>Density = %{{y:.2f}}<extra></extra>"
    )

    # Adjust legend position
    y_for_legend = 1.07

    # Add box plots if enabled
    if show_box:
        categories_cnt = len(group_labels)
        box_height = 0.05 - 0.003 * (categories_cnt - 1)
        y_for_legend = y_for_legend - 0.005 * (categories_cnt - 1)

        # Calculate height increase based on the number of categories
        height_increase = categories_cnt * 30  # Increase height by 30 pixels for each box plot
        base_height = 400  # Base height of the plot
        total_height = base_height + height_increase

        # Adjust y-axis domains for subplots
        if legend_position == 'top':
            y_legend_for_domain = 0.98 - 0.003 * (categories_cnt - 1)
            if categories_cnt <= 3:
                y_legend_for_domain -= 0.003
        else:
            y_legend_for_domain = 1.05

        total_box_height = categories_cnt * box_height  # Total height for box plots
        yaxis_domain = [y_legend_for_domain + 0.01 - total_box_height, y_legend_for_domain]
        yaxis2_domain = [0, y_legend_for_domain - 0.01 - total_box_height]

        # Create subplots for box plots and distribution plot
        fig_subplots = make_subplots(rows=2, cols=1, shared_xaxes=True)
        for trace in fig.data:
            fig_subplots.add_trace(trace, row=2, col=1)

        # Create box plots
        box_plots = go.Figure()
        for indx, category in enumerate(group_labels):
            data = distplot_data[indx]

            # Add box plot trace
            box_plots.add_trace(go.Box(
                x=data,
                name=str(category),
                orientation='h',
                notched=True,
                showlegend=False,
                marker_color=colorway[indx]
            ))

        # Update hover template for box plots
        box_plots.update_traces(
            hovertemplate=f"{xaxis_title} = %{{x}}<extra></extra>"
        )

        # Add box plot traces to subplots
        for trace in box_plots.data:
            fig_subplots.add_trace(trace, row=1, col=1)

        # Update subplot layout
        fig_subplots.update_layout(
            height=total_height if height is None else height,
            xaxis=dict(showticklabels=False, showline=False, ticks='', showgrid=True),
            yaxis=dict(domain=yaxis_domain, visible=False),
            yaxis2=dict(domain=yaxis2_domain)
        )

        fig = fig_subplots

    # Set legend position
    if legend_position == 'top':
        fig.update_layout(
            legend=dict(
                orientation="h",  # Horizontal orientation
                yanchor="top",    # Anchor to the top
                y=y_for_legend,   # Vertical position
                xanchor="center", # Anchor to the center
                x=0.5             # Horizontal position
            )
        )
    elif legend_position == 'right':
        fig.update_layout(
            legend=dict(
                orientation="v",  # Vertical orientation
                yanchor="top",    # Anchor to the top
                y=1,              # Vertical position
                xanchor="left",   # Anchor to the left
                x=1.05            # Horizontal position
            )
        )

    # Update axis titles
    if show_box:
        fig.update_layout(yaxis2_title='Density', xaxis2_title=xaxis_title)
    else:
        fig.update_layout(yaxis_title='Density', xaxis_title=xaxis_title)

    # Finalize figure layout
    fig_update(fig, height=height, width=width, title=title, margin=dict(l=50, r=50, b=50, t=60))

    return fig  # Return the final figure

def histogram(
    data_frame: pd.DataFrame = None,
    lower_quantile: float = 0,
    upper_quantile: float = 1,
    dual: bool = False,
    show_qqplot: bool = False,
    show_kde: bool = False,
    show_hist: bool = True,
    show_box: bool = True,
    render_png: bool = False,
    legend_position: str = 'top',
    **kwargs
) -> go.Figure:
    """
    Creates a histogram chart using the Plotly Express library. This function is a wrapper around Plotly Express bar and accepts all the same parameters, allowing for additional customization and functionality.

    Parameters
    ----------
    data_frame : pd.DataFrame, optional
        DataFrame containing the data to be plotted
    lower_quantile : float, optional
        The lower quantile for data filtering (default is 0).
    upper_quantile : float, optional
        The upper quantile for data filtering (default is 1).
    x : str, optional
        The name of the column in `data_frame` to be used for the x-axis. If not provided, the function will attempt to use the first column.
    y : str, optional
        The name of the column in `data_frame` to be used for the y-axis. If not provided, the function will count occurrences.
    color : str, optional
        The name of the column in `data_frame` to be used for color encoding. This will create a separate histogram for each unique value in this column.
    barmode : str, optional
        The mode for the bars in the histogram. Options include 'group', 'overlay', and 'relative'. Default is 'overlay'.
    nbins : int, optional
        The number of bins to use for the histogram. If not specified, the function will automatically determine the number of bins.
    histnorm : str, optional
        Normalization method for the histogram. Options include 'percent', 'probability', 'density', and 'probability density'. Default is None (no normalization).
    barnorm : str, optional
        Specifies how to normalize the heights of the bars in the histogram. Possible values include:
        - 'fraction': normalizes the heights of the bars so that the sum of all heights equals 1 (fraction of the total count).
        - 'percent': normalizes the heights of the bars so that the sum of all heights equals 100 (percentage of the total count).
        - 'density': normalizes the heights of the bars so that the area under the histogram equals 1 (probability density).
        - None: by default, the heights of the bars are not normalized.
    template : str, optional
        The name of the template to use for the figure. Default is None, which uses the default Plotly template.
    title : str, optional
        The title of the histogram. Default is None.
    labels : dict, optional
        A dictionary mapping column names to labels for the axes and legend.
    dual: bool, optional
        Whether to show 2 graphs, left origin, right trimmed by quantile
    **kwargs : dict
        Any additional keyword arguments accepted by `px.histogram`. This includes parameters like `opacity`, `hover_data`, `text`, `category_orders`, and more.

    Returns
    -------
    go.Figure
        Interactive Plotly histogram figure with custom hover labels and layout adjustments.
    """
    def trim_by_quantiles(config, kwargs):
        x = kwargs.get('x')
        y = kwargs.get('y')
        lower_quantile = config['lower_quantile']
        upper_quantile = config['upper_quantile']
        color = [kwargs['color']] if kwargs.get('color') else []
        facet_col = [kwargs['facet_col']] if kwargs.get('facet_col') else []
        facet_row = [kwargs['facet_row']] if kwargs.get('facet_row') else []
        animation_frame = [kwargs['animation_frame']] if kwargs.get('animation_frame') else []
        columns_for_groupby = color + facet_col + facet_row + animation_frame
        data_frame = config['data_frame']
        if columns_for_groupby:
            # Функция для обрезки значений по квантилям
            def trim_by_quantiles_in(group):
                lower_bound = group.quantile(lower_quantile)  # 0-й квантиль
                upper_bound = group.quantile(upper_quantile)  # 95-й квантиль
                return group[(group >= lower_bound) & (group <= upper_bound)]

            # Применение функции к каждой категории
            trimmed_data_frame = data_frame.groupby(columns_for_groupby, observed=False)[config['num_col']].apply(trim_by_quantiles_in).reset_index()
        else:
            if isinstance(x, str):
                if pd.api.types.is_numeric_dtype(data_frame[x]):
                    # Trim x based on the specified quantiles
                    lower_quantile_x = data_frame[x].quantile(lower_quantile)
                    upper_quantile_x = data_frame[x].quantile(upper_quantile)
                    trimmed_data_frame = data_frame[data_frame[x].between(lower_quantile_x, upper_quantile_x)]
            else:
                if pd.api.types.is_numeric_dtype(x):
                    # Trim x based on the specified quantiles
                    trimmed_column = x.between(x.quantile(lower_quantile), x.quantile(upper_quantile))
                    x = x[trimmed_column]
            if isinstance(y, str):
                if pd.api.types.is_numeric_dtype(data_frame[y]):
                    # Trim x based on the specified quantiles
                    lower_quantile_y = data_frame[y].quantile(lower_quantile)
                    upper_quantile_y = data_frame[y].quantile(upper_quantile)
                    trimmed_data_frame = data_frame[data_frame[y].between(lower_quantile_y, upper_quantile_y)]
            else:
                if pd.api.types.is_numeric_dtype(y):
                    # Trim x based on the specified quantiles
                    trimmed_column = y.between(y.quantile(lower_quantile), y.quantile(upper_quantile))
                    y = y[trimmed_column]
            if dual:
                kwargs_dual['x'] = x
                kwargs_dual['y'] = y
            else:
                kwargs['x'] = x
                kwargs['y'] = y
        if dual:
            config['trimmed_data_frame'] = trimmed_data_frame
        else:
            config['data_frame'] = trimmed_data_frame

    def get_row_col_from_axis(axis_name, config):
        cols = config['cols']
        if not axis_name:
            return 1, 1
        # Имя оси имеет формат "xaxis", "xaxis2", "xaxis3" и т.д.
        if axis_name == 'x':
            axis_number = 1
        else:
            axis_number = int(axis_name.replace("x", "")) if axis_name.startswith("x") else 1
        row = (axis_number - 1) // cols + 1
        col = (axis_number - 1) % cols + 1
        return row, col
    def make_kde_trace(trace, config):
        label_for_kde = trace.name
        kde_data = trace.x
        color_for_kde = trace.marker.color
        if len(kde_data) < 2:
            print(f'In group "{label_for_kde}" number of elements less then 2. KDE line cannot be constructed.')
            return
        kde_obj = gaussian_kde(kde_data)
        x_values_kde = np.linspace(min(kde_data), max(kde_data), 1000)
        y_values_kde = kde_obj(x_values_kde)
        # Добавление линии KDE на график
        if kwargs.get('labels') is not None and config['num_col'] in kwargs['labels']:
            label_for_kde_hovertemplate = kwargs['labels'][config['num_col']]
        else:
            label_for_kde_hovertemplate = 'Значение'
        hovertemplate_kde = ''
        if kwargs.get('color') is not None:
            if kwargs['color'] in kwargs['labels']:
                hovertemplate_kde += f'{kwargs['labels'][kwargs['color']]} = {trace.name}<br>'
        if 'labels' in kwargs and kwargs['x'] in kwargs['labels']:
            hovertemplate_kde += f'{kwargs['labels'][kwargs['x']]} = ' + '%{x:.2f}<br>'
        hovertemplate_kde += f'Плотность = ' + '%{y:.2f}'
        hovertemplate_kde += '<extra></extra>'
        kde_trace = go.Scatter(
            x=x_values_kde
            , y=y_values_kde.round(7)
            , mode='lines'
            , name=label_for_kde
            , legendgroup=label_for_kde
            , line=dict(color=color_for_kde)
            , hovertemplate=hovertemplate_kde
            , xaxis=trace.xaxis
            , yaxis=trace.yaxis
        )
        if config['show_hist']:
            kde_trace.showlegend = False
        elif config['dual']:
            kde_trace.showlegend = trace.showlegend
        else:
            True
        return kde_trace
    def make_subplots_fig(data, config, kwargs):
        rows = config['rows']
        cols = config['cols']
        show_kde = config['show_kde']
        show_box = config['show_box']
        if show_box:
            new_rows = rows * 2
        else:
            new_rows = rows

        # Add box plots if enabled
        if show_box:
            if 'color' in kwargs:
                group_labels = config['data_frame'][kwargs['color']].unique()
                categories_cnt = len(group_labels)
                # # Calculate height increase based on the number of categories
                # if categories_cnt <= 5:
                #     coef_for_increase = 20
                # else:
                #     coef_for_increase = 22
                height_increase = categories_cnt * 20  # Increase height by 30 pixels for each box plot
                base_height = 400
                total_height = base_height + height_increase
                kwargs.setdefault('height', total_height)
                if categories_cnt == 1:
                    row_heights_box = 0.07
                    row_heights_hist = 0.93
                if categories_cnt <= 2:
                    row_heights_box = 0.13
                    row_heights_hist = 0.87
                elif categories_cnt <= 5:
                    row_heights_box = 0.2
                    row_heights_hist = 0.8
                elif categories_cnt <= 8:
                    row_heights_box = 0.25
                    row_heights_hist = 0.75
                else:
                    row_heights_box = 0.3
                    row_heights_hist = 0.7
            else:
                row_heights_box = 0.07
                row_heights_hist = 0.93
        subplots_fig = make_subplots(
            rows=new_rows, cols=cols,  # Удваиваем строки
            shared_xaxes=True,  # Синхронизируем оси X
            shared_yaxes=True, #False if config['dual'] else True,  # Синхронизируем оси Y
            vertical_spacing=0.05, horizontal_spacing=0.05,
            row_heights=[row_heights_box, row_heights_hist] * rows if show_box else None,  # Боксплоты занимают 20% высоты, гистограммы — 80%
            start_cell='top-left',
        )
        for trace in data:
            # Определяем строку и столбец по оси
            # print(trace.xaxis)
            row, col = get_row_col_from_axis(trace.xaxis, config)
            # print(row, col)
            # Гистограммы размещаем в нижних строк
            if show_box:
                last_row = rows * 2
                if rows ==  1:
                    row_hist = row + 1
                else:
                    row_hist = row * 2
                    # нужно перевернуть порядок
                    row_hist = rows * 2 - row_hist + 2
            else:
                last_row = rows
                row_hist = rows - row + 1
            # print(rows, cols)
            # print(row, col)
            # print(row_hist)
            # print()
            subplots_fig.add_trace(trace, row=row_hist, col=col)
            if show_kde:
                kde_trace = make_kde_trace(trace, config)
                subplots_fig.add_trace(kde_trace, row=row_hist, col=col)
            if col == 1:
                subplots_fig.update_yaxes(
                    title=config.get('yaxis_title')
                    , row=row_hist, col=col
                )
            else:
                subplots_fig.update_yaxes(showticklabels=False, row=row_hist, col=col)
            if row_hist == last_row:
                subplots_fig.update_xaxes(
                    title=config.get('xaxis_title')
                    , row=row_hist, col=col
                )
            subplots_fig.update_xaxes(showgrid=True, row=row_hist, col=col)
            subplots_fig.update_yaxes(showgrid=True, row=row_hist, col=col)
            if show_box:
                # Создаем боксплот для данных гистограммы
                if kwargs.get('labels') is not None and config['num_col'] in kwargs['labels']:
                    label_for_box_hovertemplate = kwargs['labels'][config['num_col']]
                else:
                    label_for_box_hovertemplate = 'Значение'
                if kwargs.get('color') is not None:
                    if kwargs['color'] in kwargs['labels']:
                        hovertemplate_box = f'{kwargs['labels'][kwargs['color']]} = {trace.name}<br>' + f'{label_for_box_hovertemplate} = ' + '%{x:.2f}<extra></extra>'
                else:
                    hovertemplate_box = f'{label_for_box_hovertemplate} = ' + '%{x:.2f}<extra></extra>'
                box = go.Box(
                    x=trace.x,
                    showlegend= False if config['show_kde'] or config['show_hist'] else True,
                    hovertemplate=hovertemplate_box,
                    marker_color=trace.marker.color,
                    legendgroup=trace.name
                )
                # Боксплоты размещаем в верхних строках
                row_box = row_hist - 1
                subplots_fig.add_trace(box, row=row_box, col=col)
                subplots_fig.update_xaxes(
                    showticklabels=False, showline=False, ticks='', showgrid=False, row=row_box, col=col
                )
                subplots_fig.update_yaxes(
                    visible=False, matches=None, row=row_box, col=col
                )

        return subplots_fig

    def update_fig(fig, config, kwargs):
        TITLE_FONT_SIZE = 15
        FONT_SIZE = 13
        AXIS_TITLE_FONT_SIZE = 13
        TICK_FONT_SIZE = 13
        LEGEND_TITLE_FONT_SIZE = 13
        FONT_FAMILY = "Noto Sans"
        FONT_COLOR = "rgba(0, 0, 0, 0.7)"
        LINE_COLOR = "rgba(0, 0, 0, 0.4)"
        GRID_COLOR = "rgba(0, 0, 0, 0.1)"
        HOVER_BGCOLOR = "white"
        GRID_WIDTH = 1
        # Layout updates
        layout_updates = {
            'font': {'size': FONT_SIZE, 'family': FONT_FAMILY, 'color': FONT_COLOR},
            'title_font': {'size': TITLE_FONT_SIZE, 'family': FONT_FAMILY, 'color': FONT_COLOR},
            'title_text': kwargs.get('title'),
            'barmode': kwargs.get('barmode'),
            'height': kwargs.get('height'),
            'width': kwargs.get('width'),
            'margin': None if 'color' in kwargs else dict(l=50, r=50, b=50, t=50),
        }
        # Update layout only if there are updates
        layout_updates = {k: v for k, v in layout_updates.items() if v is not None}
        if layout_updates:
            fig.update_layout(**layout_updates)
        # X-axis settings
        xaxis_updates = {
            'linecolor': LINE_COLOR,
            'tickcolor': LINE_COLOR,
            'gridwidth': GRID_WIDTH,
            'gridcolor': GRID_COLOR,
            'title_font': {'size': FONT_SIZE, 'family': FONT_FAMILY, 'color': FONT_COLOR},
            'tickfont': {'size': FONT_SIZE, 'family': FONT_FAMILY, 'color': FONT_COLOR},
        }
        fig.update_xaxes(**{k: v for k, v in xaxis_updates.items() if v is not None})

        # Y-axis settings
        yaxis_updates = {
            'linecolor': LINE_COLOR,
            'tickcolor': LINE_COLOR,
            'gridwidth': GRID_WIDTH,
            'gridcolor': GRID_COLOR,
            'title_font': {'size': FONT_SIZE, 'family': FONT_FAMILY, 'color': FONT_COLOR},
            'tickfont': {'size': FONT_SIZE, 'family': FONT_FAMILY, 'color': FONT_COLOR},
        }
        fig.update_yaxes(**{k: v for k, v in yaxis_updates.items() if v is not None})

        # Legend settings
        legend_updates = {
            'legend_title_font': {'size': FONT_SIZE, 'family': FONT_FAMILY, 'color': FONT_COLOR},
            'legend_font_color': FONT_COLOR,
            'legend_itemwidth': 30 ,
            'legend_orientation': 'h',
            'legend_xanchor': "center",
            'legend_x': 0.5,
            'legend_y': 1.15 ,
            'legend_itemsizing': "constant",
        }
        fig.update_layout(**{k: v for k, v in legend_updates.items() if v is not None})
        if not config['show_hist']:
            fig.update_traces(selector={'type': 'histogram'}, visible=False)
        if 'color' not in kwargs:
            fig.update_traces(selector={'type': 'histogram'}, opacity=1)
        for trace in fig.data:
            if trace.hovertemplate is not None:
                # print(trace.hovertemplate)
                trace.hovertemplate = trace.hovertemplate.replace('probability density', 'Плотность')
                trace.hovertemplate = trace.hovertemplate.replace('count', 'Количество')
                trace.hovertemplate = trace.hovertemplate.replace('y', 'y:.2f')
                trace.hovertemplate = re.sub(r'\s*=\s*', ' = ', trace.hovertemplate)
        return fig

    if data_frame is not None and not isinstance(data_frame, pd.DataFrame):
        raise TypeError("data_frame must be a pandas DataFrame.")
    if kwargs.get('marginal') is not None:
        raise ValueError("marginal can not be used, use show_box instead")
    if dual and show_qqplot:
        raise ValueError('dual mode can not be use together with show_qqplot')
    if dual and (kwargs.get('facet_col') is not None or kwargs.get('facet_row') is not None or kwargs.get('facet_col_wrap') is not None
                 or kwargs.get('animation_frame') is not None):
        raise ValueError('dual mode can not be use together with any facet or animation_frame')
    if show_qqplot and (kwargs.get('facet_col') is not None or kwargs.get('facet_row') is not None or kwargs.get('facet_col_wrap') is not None
                 or kwargs.get('animation_frame') is not None):
        raise ValueError('show_qqplot can not be use together with any facet or animation_frame')
    if kwargs.get('color') is not None and data_frame is None:
        raise ValueError('For use color data_frame must be difine')
    if kwargs.get('facet_col') is not None and data_frame is None:
        raise ValueError('For use facet_col data_frame must be difine')
    if kwargs.get('facet_row') is not None and data_frame is None:
        raise ValueError('For use facet_row data_frame must be difine')
    if kwargs.get('animation_frame') is not None and data_frame is None:
        raise ValueError('For use animation_frame data_frame must be difine')
    # Set default values for the figure dimensions and bar mode
    config = dict(
        data_frame = data_frame,
        lower_quantile = lower_quantile,
        upper_quantile = upper_quantile,
        dual = dual,
        show_qqplot = show_qqplot,
        show_kde = show_kde,
        show_hist = show_hist,
        show_box = show_box,
        render_png = render_png,
        colorway = colorway_for_stacked_histogram if kwargs.get('color_discrete_sequence') is None else kwargs['color_discrete_sequence'],
        legend_position = legend_position,
    )
    if 'color' in kwargs:
        kwargs.setdefault('barmode', 'overlay')
    kwargs.setdefault('nbins', 30)
    kwargs.setdefault('histnorm', 'probability density')
    kwargs.setdefault('color_discrete_sequence', config['colorway'])
    if dual or show_qqplot or kwargs.get('facet_col'):
        kwargs.setdefault('width', 900)
    else:
        kwargs.setdefault('width', 600)
    kwargs.setdefault('height', 400)
    if kwargs.get('x') and kwargs.get('y'):
        raise ValueError('Must be define x or y not both')
    if kwargs.get('x'):
        config['num_col'] = kwargs.get('x')
    else:
        config['num_col'] = kwargs.get('y')
    kwargs['hover_data'] = {config['num_col']: ':.2f'}
    if kwargs.get('labels') is not None and config['num_col'] in kwargs['labels']:
        config['xaxis_title'] = kwargs['labels'][config['num_col']]
    if kwargs.get('histnorm') == 'probability density':
        config['yaxis_title'] = 'Плотность'
    elif kwargs.get('histnorm') is None:
        config['yaxis_title'] = 'Количество'
    if dual:
        kwargs_dual = kwargs.copy()

    if upper_quantile != 1 or lower_quantile != 0:
        trim_by_quantiles(config, kwargs)

    # чтобы синхронизировать порядок trace в режиме dual, зададим порядок, если он не задан
    if kwargs.get('color') is not None:
        sorted_color_labels = sorted(data_frame[kwargs['color']].unique())
        if kwargs.get('category_orders') is None:
            kwargs['category_orders'] = dict()
        if dual and kwargs_dual.get('category_orders') is None:
            kwargs_dual['category_orders'] = dict()
        kwargs['category_orders'][kwargs['color']] = sorted_color_labels
        if dual:
            kwargs_dual['category_orders'][kwargs_dual['color']] = sorted_color_labels

    # Create the histogram figure using Plotly Express
    fig = px.histogram(config['data_frame'], **kwargs)
    # print(fig._grid_str)
    if not dual and not show_qqplot:
        if kwargs.get('marginal') is None:
            config['rows'] = len(fig._grid_ref)
            config['cols'] = len(fig._grid_ref[0]) if config['rows'] > 0 else 0
            fig_new = make_subplots_fig(fig.data, config, kwargs)
            fig_frames = fig.frames
            if fig_frames:
                fig_new.frames = fig_frames
                for i, frame in enumerate(fig_frames):
                    fig_for_frame = make_subplots_fig(frame.data, config, kwargs)
                    fig_new.frames[i].data = fig_for_frame.data

            fig_new.update_layout(
                annotations = fig.layout.annotations
            )
            fig_new.layout.updatemenus = fig.layout.updatemenus
            fig_new.layout.sliders = fig.layout.sliders
            for layout_key in fig.layout:
                if layout_key.startswith('xaxis'):
                    fig_new.layout[layout_key].domain = fig.layout[layout_key].domain
    elif dual == True:
        if 'trimmed_data_frame' not in config:
            raise ValueError('For dual mode must be define lower or upper quantile')
        fig_trimmed = px.histogram(config['trimmed_data_frame'], **kwargs_dual)
        # labels_fig = {label: index for index, label in enumerate(group_labels)}
        # distplot_data = [distplot_data[labels_map[cat]] for cat in ordered_categories]
        fig_dual = make_subplots(rows=1, cols=2, horizontal_spacing=0.07)
        for trace in fig.data:
            trace.bingroup = None
            fig_dual.add_trace(trace, row=1, col=1)
        for trace in fig_trimmed.data:
            trace.bingroup = None
            trace.showlegend = False
            fig_dual.add_trace(trace, row=1, col=2)
        config['rows'] = len(fig_dual._grid_ref)
        config['cols'] = len(fig_dual._grid_ref[0]) if config['rows'] > 0 else 0
        fig_new = make_subplots_fig(fig_dual.data, config, kwargs)
        # fig_subplots.update_xaxes(title_text=x_for_box_hovertemplate, row=1, col=1)
        # fig_subplots.update_xaxes(title_text=x_for_box_hovertemplate, row=1, col=2)
        # if kwargs.get('histnorm') == 'probability':
        #     fig_subplots.update_yaxes(title_text='Доля', row=1, col=1)  # Set x-axis title to 'Доля' for probability
        # if kwargs.get('histnorm') is None:
        #     fig_subplots.update_yaxes(title_text='Количество', row=1, col=1)  # Set x-axis title to 'Количество' for count

    elif show_qqplot == True:
        if kwargs['x'] is not None:
            if isinstance(kwargs['x'], str):
                data_for_qqplot = data_frame[kwargs['x']]
            else:
                data_for_qqplot = kwargs['x']
        elif kwargs['y'] is not None:
            if isinstance(kwargs['y'], str):
                data_for_qqplot = data_frame[kwargs['y']]
            else:
                data_for_qqplot = kwargs['y']
        else:
            raise ValueError('For qqplot must be define x or y, not both')
        config['rows'] = len(fig._grid_ref)
        config['cols'] = len(fig._grid_ref[0]) if config['rows'] > 0 else 0
        fig_hist = make_subplots_fig(fig.data, config, kwargs)
        if show_box:
            fig_new = make_subplots(rows=2, cols=2, horizontal_spacing=0.1
                                    , row_heights=[0.07, 0.93]
                                    , specs=[
                                        [{'colspan': 1}, {'rowspan': 2}],
                                        [{'colspan': 1}, None]
                                    ])
            for trace in fig_hist.data:
                if trace.type == 'histogram':
                    fig_new.add_trace(trace, row=2, col=1)
                else:
                    fig_new.add_trace(trace, row=1, col=1)
                    fig_new.update_xaxes(
                        showticklabels=False, showline=False, ticks='', showgrid=True, row=1, col=1
                    )
                    fig_new.update_yaxes(
                        visible=False, row=1, col=1
                    )
        else:
            fig_new = make_subplots(rows=1, cols=2, horizontal_spacing=0.1)
            for trace in fig_hist.data:
                fig_new.add_trace(trace, row=1, col=1)
        qqplot = qqplot_plotly(data_for_qqplot)
        for trace in qqplot.data:
            fig_new.add_trace(
                trace
                , row=1, col=2
            )
        for annotation in qqplot.layout.annotations:
            # Обновляем ссылки на оси (xref и yref) для подграфика
            annotation.update(xref='x2', yref='y2')
            fig_new.add_annotation(annotation)
        fig_new.update_xaxes(title_text='Теоретические квантили', row=1, col=2)
        fig_new.update_yaxes(title_text='Упорядоченные квантили', row=1, col=2)
        if kwargs.get('histnorm') == 'probability density':
            fig_new.update_yaxes(title_text='Плотность', row=1, col=1)  # Set x-axis title to 'Доля' for probability
        if kwargs.get('histnorm') is None:
            fig_new.update_yaxes(title_text='Количество', row=1, col=1)  # Set x-axis title to 'Количество' for count
        fig_new.update_traces(showlegend=False)
    fig_new.update_layout(annotations=fig.layout.annotations)
    fig_new = update_fig(fig_new, config, kwargs)
    return fig_new

