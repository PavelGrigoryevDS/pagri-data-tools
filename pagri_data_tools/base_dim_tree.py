import pandas as pd
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from igraph import Graph
from frameon.utils.miscellaneous import format_number

def dim_tree(
    data_frame,
    dimensions,
    format_count=True,
    title=None,
    node_text_size=12,
    info_text_size=11,
    node_size=60,
    width=1100,
    height=500,
    line_width=1,
    margin_l=0,
    margin_r=10,
    margin_b=20,
    margin_t=20
):
    """
    Creates an interactive hierarchical tree visualization from a pandas DataFrame.
    """
    df = data_frame
    # 1. Data preparation
    total_users = len(df)  # We just count the number of lines
    agg_df = df.groupby(dimensions, observed=False).size().reset_index(name='count') 
    
    # 2. Graph construction
    G = Graph(directed=True)
    node_names = []      # Node labels for circle display
    hover_texts = []     # Full hover text with statistics
    info_texts = []      # Info text below nodes (without node names)
    node_dict = {}       # Mapping between paths and node IDs
    
    # Add root node
    root_id = 0
    G.add_vertex(name="0")
    node_names.append("All")
    formatted_total_users = format_number(total_users) if format_count else total_users
    hover_texts.append(f"<b>All</b><br>Count: {formatted_total_users}<br>100% of All")
    info_texts.append(f"{formatted_total_users}")
    node_dict[tuple()] = root_id
    
    # 3. Recursive node addition
    def add_nodes(parent_path, current_level):
        if current_level >= len(dimensions):
            return
            
        # Get unique values for current level
        if parent_path:
            mask = True
            for i, dim in enumerate(dimensions[:current_level]):
                mask &= (agg_df[dim] == parent_path[i])
            current_values = agg_df[mask][dimensions[current_level]].unique()
        else:
            current_values = agg_df[dimensions[current_level]].unique()
        
        for value in current_values:
            current_path = parent_path + (value,)
            
            # Calculate counts and percentages
            grouped = agg_df.copy()
            for i, dim in enumerate(dimensions[:current_level+1]):
                grouped = grouped[grouped[dim] == current_path[i]]
            count = grouped['count'].sum() if not grouped.empty else 0
            pct_total = round(count / total_users * 100, 1)
            
            formatted_count = format_number(count) if format_count else count
            
            # Build hover text (with node name)
            hover_parts = [f"<b>{value}</b>", f"Count: {formatted_count} ({pct_total}% of All)"]
            
            # Build info text below node (without node name)
            info_parts = [f"{formatted_count} ({pct_total}% of All)"]
            
            # Add parent level percentages
            if current_level >= 1:
                parent_mask_l1 = True
                for i, dim in enumerate(dimensions[:1]):
                    parent_mask_l1 &= (agg_df[dim] == current_path[i])
                parent_count_l1 = agg_df[parent_mask_l1]['count'].sum()
                pct_parent_l1 = round(count / parent_count_l1 * 100, 1) if parent_count_l1 > 0 else 0
                hover_parts.append(f"{pct_parent_l1}% of {current_path[0]}")
                info_parts.append(f"{pct_parent_l1}% of {current_path[0]}")

            if current_level >= 2:
                parent_mask_l2 = True
                for i, dim in enumerate(dimensions[:2]):
                    parent_mask_l2 &= (agg_df[dim] == current_path[i])
                parent_count_l2 = agg_df[parent_mask_l2]['count'].sum()
                pct_parent_l2 = round(count / parent_count_l2 * 100, 1) if parent_count_l2 > 0 else 0
                hover_parts.append(f"{pct_parent_l2}% of {current_path[0]}>{current_path[1]}")
                info_parts.append(f"{pct_parent_l2}% of {current_path[0]}>{current_path[1]}")
            
            hover_text = "<br>".join(hover_parts)
            info_text = "<br>".join(info_parts)
            
            # Add node to graph
            node_id = len(node_names)
            G.add_vertex(name=str(node_id))
            node_names.append(value)
            hover_texts.append(hover_text)
            info_texts.append(info_text)
            node_dict[current_path] = node_id
            
            # Add connection to parent
            if parent_path in node_dict:
                parent_id = node_dict[parent_path]
                G.add_edge(parent_id, node_id)
            
            add_nodes(current_path, current_level + 1)
    
    # Start building the tree
    add_nodes(tuple(), 0)
    
    # 4. Visualization
    lay = G.layout('rt')
    position = {k: lay[k] for k in range(len(node_names))}
    Xn = [position[k][0] for k in range(len(node_names))]
    Yn = [2*max(lay[k][1] for k in range(len(node_names))) - position[k][1] for k in range(len(node_names))]
    
    fig = go.Figure()
    
    # Add connecting lines
    for edge in G.es:
        x0, y0 = position[edge.source]
        x1, y1 = position[edge.target]
        fig.add_trace(go.Scatter(
            x=[x0, x1, None], 
            y=[2*max(lay[k][1] for k in range(len(node_names))) - y0, 
              2*max(lay[k][1] for k in range(len(node_names))) - y1, None],
            mode='lines',
            line=dict(color='rgb(210,210,210)', width=line_width),
            hoverinfo='none'
        ))
    
    # Add nodes with labels
    fig.add_trace(go.Scatter(
        x=Xn, 
        y=Yn,
        mode='markers+text',
        text=node_names,
        textposition="middle center",
        textfont=dict(color='white', size=node_text_size),
        marker=dict(
            size=node_size,
            color='#6175c1',
            line=dict(width=1, color='DarkSlateGrey')
        ),
        hovertext=hover_texts,
        hoverinfo="text",
        name=""
    ))
    
    # Add info text below nodes with dynamic offsets
    y_offsets = []
    for text in info_texts:
        line_count = len(text.split('<br>'))
        base_offset = 0.42
        line_spacing = 0.09
        y_offsets.append(base_offset + (line_count - 1) * line_spacing)
    
    fig.add_trace(go.Scatter(
        x=[x - 0.25 for x in Xn],
        y=[y - offset for y, offset in zip(Yn, y_offsets)],
        mode='text',
        text=info_texts,
        textposition="middle right",
        hoverinfo='none',
        textfont=dict(size=info_text_size, color='black')
    ))
    
    # Final layout adjustments
    fig.update_layout(
        title=title if title else 'Hierarchical Breakdown',
        title_y=0.97,
        showlegend=False,
        xaxis_visible=False,
        yaxis_visible=False,
        margin=dict(l=margin_l, r=margin_r, b=margin_b, t=margin_t),
        height=height,
        width=width,
    )
    
    return fig