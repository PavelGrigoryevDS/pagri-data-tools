import pandas as pd
import numpy as np
import plotly.express as px
from plotly.subplots import make_subplots
import itertools
import plotly.graph_objects as go
import plotly.io as pio
from dash import Dash, dcc, html
from dash.dependencies import Input, Output


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
    fig.update_layout(
        title_font=dict(size=24, color="rgba(0, 0, 0, 0.6)"),
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
    return fig


def heatmap_corr(df, title='Тепловая карта корреляционных связей между числовыми столбцами', titles_for_axis: dict = None, xtick_text=None, ytick_text=None, xaxis_label=None, yaxis_label=None, width=None, height=None, decimal_places=2, font_size=14):
    """
    Creates a heatmap from a Pandas DataFrame using Plotly.

    Parameters:
    - `df`: The Pandas DataFrame to create the heatmap from.
    - `title`: The title of the heatmap (default is an empty string).
    - `titles_for_axis` (dict):  A dictionary containing titles for the axes.
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
    if xaxis_label is not None:
        fig.update_layout(xaxis=dict(title=xaxis_label))

    if yaxis_label is not None:
        fig.update_layout(yaxis=dict(title=yaxis_label))

    # Update figure size if custom size is provided
    if width is not None:
        fig.update_layout(width=width)
    if height is not None:
        fig.update_layout(height=height)
    hovertemplate = 'Название столбца (ось X) = %{x}<br>Название столбца (ось Y) = %{y}<br>Коэффициент корреляции = %{z:.2f}<extra></extra>'
    fig.update_traces(hovertemplate=hovertemplate)
    fig.update_layout(
        title_font=dict(size=24, color="rgba(0, 0, 0, 0.6)"),
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
    return fig




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
                   .groupby(cat_columns, observed=True)
                   .agg(num=(num_column, func), count=(num_column, 'count'))
                   .reset_index())
        func_df['temp'] = func_df.groupby(cat_columns[0], observed=True)[
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
        margin=dict(t=80),
        width=config['width'], height=config['height'], title_font=dict(size=24, color="rgba(0, 0, 0, 0.5)"), title={'text': config["title"]}, xaxis_title=x_axis_label, yaxis_title=y_axis_label, legend_title_text=color_axis_label, font=dict(size=16, family="Open Sans", color="rgba(0, 0, 0, 1)"), xaxis_title_font=dict(size=18, color="rgba(0, 0, 0, 0.5)"), yaxis_title_font=dict(size=18, color="rgba(0, 0, 0, 0.5)"), xaxis_tickfont=dict(size=16, color="rgba(0, 0, 0, 0.5)"), yaxis_tickfont=dict(size=16, color="rgba(0, 0, 0, 0.5)"), legend_title_font_color='rgba(0, 0, 0, 0.5)', legend_font_color='rgba(0, 0, 0, 0.5)', xaxis_linecolor="rgba(0, 0, 0, 0.5)", yaxis_linecolor="rgba(0, 0, 0, 0.5)", hoverlabel=dict(bgcolor="white"), xaxis=dict(
            visible=config['xaxis_show'], showgrid=config['showgrid_x'], gridwidth=1, gridcolor="rgba(0, 0, 0, 0.1)"
        ), yaxis=dict(
            visible=config['yaxis_show'], showgrid=config['showgrid_y'], gridwidth=1, gridcolor="rgba(0, 0, 0, 0.07)"
        ),
        legend=dict(
            title_font_color="rgba(0, 0, 0, 0.5)", font_color="rgba(0, 0, 0, 0.5)"
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
        hovertemplate='Значение = %{x}<br>Частота = %{y:.2f}<extra></extra>', showlegend=False)
    fig.update_layout(
        # , title={'text': f'<b>{title}</b>'}
        width=width, height=height, title_font=dict(size=24, color="rgba(0, 0, 0, 0.5)"), xaxis_title=xaxis_title, yaxis_title=yaxis_title, font=dict(size=14, family="Open Sans", color="rgba(0, 0, 0, 1)"), xaxis_title_font=dict(size=18, color="rgba(0, 0, 0, 0.5)"), yaxis_title_font=dict(size=18, color="rgba(0, 0, 0, 0.5)"), xaxis_tickfont=dict(size=14, color="rgba(0, 0, 0, 0.5)"), yaxis_tickfont=dict(size=14, color="rgba(0, 0, 0, 0.5)"), legend_title_font_color='rgba(0, 0, 0, 0.5)', legend_font_color='rgba(0, 0, 0, 0.5)', xaxis_linecolor="rgba(0, 0, 0, 0.5)", yaxis_linecolor="rgba(0, 0, 0, 0.5)"        # , margin=dict(l=50, r=50, b=50, t=70)
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
            
            
def pairplot(df, width=800, height=800, titles_for_axis: dict = None, horizontal_spacing=None, vertical_spacing=None, rows=None, cols=None):
    """
    Create a pairplot of numerical variables in a dataframe using Plotly.

    Parameters:
    df (pandas.DataFrame): Input dataframe
    width (int, optional): Width of the plot. Defaults to 800.
    height (int, optional): Height of the plot. Defaults to 800.
    titles_for_axis (dict, optional): Dictionary of custom axis titles. Defaults to None.
    horizontal_spacing (float, optional): Horizontal spacing between subplots. Defaults to None.
    vertical_spacing (float, optional): Vertical spacing between subplots. Defaults to None.
    rows (int, optional): Number of rows in the subplot grid. Defaults to None.
    cols (int, optional): Number of columns in the subplot grid. Defaults to None.

    Returns:
    fig (plotly.graph_objs.Figure): The resulting pairplot figure
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

    for i, (col1, col2) in enumerate(combinations):
        row, col = divmod(i, cols)
        if titles_for_axis:
            xaxes_title = titles_for_axis[col1]
            yaxes_title = titles_for_axis[col2]
        else:
            xaxes_title = col1
            yaxes_title = col2
        fig_scatter = px.scatter(df, x=col1, y=col2)
        fig_scatter.update_traces(marker=dict(
            line=dict(color='white', width=0.5)))
        fig_scatter.update_traces(
            hovertemplate=xaxes_title + ' = %{x}<br>' + yaxes_title + ' = %{y}')
        fig.add_trace(fig_scatter.data[0], row=row+1, col=col+1)
        fig.update_xaxes(
            title_text=xaxes_title,
            title_font=dict(size=18, color="rgba(0, 0, 0, 0.5)"),
            tickfont=dict(size=14, color="rgba(0, 0, 0, 0.5)"),
            linecolor="rgba(0, 0, 0, 0.5)",
            row=row+1, col=col+1,
            showgrid=True,
            gridwidth=1,
            gridcolor="rgba(0, 0, 0, 0.1)"
        )
        fig.update_yaxes(
            title_text=yaxes_title,
            title_font=dict(size=18, color="rgba(0, 0, 0, 0.5)"),
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
        title_font_size=16,
        margin=dict(l=50, r=50, t=90, b=50),
        title_font=dict(size=24, color="rgba(0, 0, 0, 0.6)"),
        title={'text': f'Зависимости между числовыми переменными'},
        # Для подписей и меток
        font=dict(size=14, family="Open Sans", color="rgba(0, 0, 0, 1)"),
        xaxis_title_font=dict(size=18, color="rgba(0, 0, 0, 0.5)"),
        yaxis_title_font=dict(size=18, color="rgba(0, 0, 0, 0.5)"),
        xaxis_tickfont=dict(size=14, color="rgba(0, 0, 0, 0.5)"),
        yaxis_tickfont=dict(size=14, color="rgba(0, 0, 0, 0.5)"),
        xaxis_linecolor="rgba(0, 0, 0, 0.5)",
        yaxis_linecolor="rgba(0, 0, 0, 0.5)",
        hoverlabel=dict(bgcolor="white"),
    )

    return fig
            