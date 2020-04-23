'''Contains the functions required to either generate more complex html structures, or
process/visualize data. These feed in to the page content as defined in Index.py'''

import base64
import pickle
from io import BytesIO

import dash_html_components as html
import networkx as nx
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from numpy import log
from numpy.random import random
from wordcloud import WordCloud, STOPWORDS

from models.train_classifier import load_data



####################################################################################################
# Initialization                                                                                   #
####################################################################################################

# Load External Dependencies #######################################################################

with open('../models/model.pkl', 'rb') as f:
    classifier = pickle.load(f)

X, y, cat_names = load_data('../data/udacity.db')

cat_counts = {
    cat: y[:, inx].sum()
    for inx, cat in enumerate(cat_names)
}



####################################################################################################
# Category Display                                                                                 #
####################################################################################################

def gen_cat_card(cat):
    '''Generates an html element to display one of the categories which a message has
    been tagged with'''

    # Format for display
    cat = cat.replace('_', ' ')
    cat = cat.title()

    # Set final layout
    card = html.Div(
        html.P(
            cat,
            className='text-center m-0'
        ),
        className='alert alert-primary align-middle py-1 px-2 mr-2',
        role='alert'
    )

    return card


def gen_jumbotron(usr_input):
    '''Generates a bootstrap 'jumbotron' element to display the provided message, and
    all of the categories that it's been tagged with'''

    # Use the pretrained model to make a prediction
    X_usr = [usr_input]
    y_usr = classifier.predict(X_usr)
    y_usr = y_usr[0]

    # Turn binary array into a list of category names
    cats = [
        name
        for cat, name in zip(y_usr, cat_names)
        if cat
    ]

    # Create a card for each category
    cats = [gen_cat_card(cat) for cat in cats]

    # Set final layout
    jumbotron = html.Div(
        [
            html.H1("Message Categories"),
            html.P(usr_input, id='msg-display', className='lead'),
            html.Hr(className='my-4'),
            html.Div(
                cats,
                id='cat-display',
                className='d-flex flex-row flex-wrap'),
        ],
        className='jumbotron'
    )

    return jumbotron



####################################################################################################
# Word Cloud                                                                                       #
####################################################################################################

def fig_to_uri(in_fig, close_all=True, **save_args):
    """Save a matplotlib figure as a base-64 encoded string. This is based on advice from
    the plotly community forums, as saving the plot to the assets directory wouldn't
    work for a multi-user dashboard."""

    # In-memory binary file writer
    out_img = BytesIO()

    # Save figure to writer, clear matplotlib canvas
    in_fig.savefig(out_img, format='png', **save_args)
    if close_all:
        in_fig.clf()
        plt.close('all')

    # Go to start of file, encode it as a base64 string
    out_img.seek(0)
    encoded = base64.b64encode(out_img.read()).decode("ascii").replace("\n", "")

    # Format into string which can be used as 'src' in an image element
    imstr = f"data:image/png;base64,{encoded}"
    return imstr


def generate_word_cloud(category):
    '''For a given category, find all messages from the training dataset which were
    associated with it. Generate a wordcloud using these messages and place it into an
    html element for display'''

    # Get column number for the given column
    cat_selector = cat_names.index(category)

    # Select messages where column value == 1
    cat_mask = pd.Series(y[:, cat_selector]).astype(bool)
    cat_text = pd.Series(X).loc[cat_mask]

    # Concatenate into a single string
    cat_text = " ".join(cat_text.dropna()).lower()

    # Generate the wordcloud
    cloud = WordCloud(
        background_color='#f7f7f7',
        mode='RGBA',
        stopwords=STOPWORDS,
        normalize_plurals=True,
        collocations=True,
        width=1240,
        height=720
    ).generate(cat_text)

    # Set figure dimensions, tidy up plot area
    fig = plt.figure(1, figsize=(12.4, 7.2), dpi=100)
    plt.axes([0, 0, 1, 1,], label=str(random()))
    plt.imshow(cloud, interpolation='bilinear', aspect='equal')
    plt.axis('off')

    # Convert figure to base-64 encoded string
    imstr = fig_to_uri(fig, bbox_inches=0, dpi=100, figsize=(12.4, 7.2), transparent=True)

    return imstr



####################################################################################################
# Network Graph                                                                                    #
####################################################################################################

def create_network():
    '''Use the training dataset to create a NetworkX graph which represents the
    strength of the connections between categories. The more messages there are which
    are tagged with the same two topics (e.g. food & water), the stronger the
    connection between them will be'''

    # Convert category tags to dataframe
    cat_df = pd.DataFrame(y, columns=cat_names)

    # Remove child_alone category as it doesn't map to any other categories
    # and distorts the visualization
    cat_df = cat_df.drop(columns=['child_alone'])
    cat_list = cat_names[:]
    cat_list.remove('child_alone')

    # For each category, assign a node number. Create dictionaries
    # allowing mapping from node to category & vice-versa
    nodes_to_cats = {}
    cats_to_nodes = {}
    for inx, cat in enumerate(cat_list):
        nodes_to_cats[inx] = cat
        cats_to_nodes[cat] = inx

    # Create a list of node indices
    node_list = nodes_to_cats.keys()

    # Calculate the connection weight from each node to every other node
    edges = {}
    for inx, row in cat_df.iterrows():
        # For each message, get a list of associated categories
        cats = row.loc[row == 1]
        cats = list(cats.keys())
        if len(cats) >= 2:
            # At least 2 categories needed to create a connection
            for cat_from in cats:
                # For each start category, get a list all end (connected) categories
                cats_copy = cats[:]
                cats_copy.remove(cat_from)

                # Get the node number which matches the start category
                node_from = cats_to_nodes[cat_from]

                for cat_to in cats_copy:
                    # For each end category, get the corresponding node number
                    node_to = cats_to_nodes[cat_to]

                    # Create a connection between nodes
                    connection = tuple(sorted([node_from, node_to]))

                    # Increment the weight of the connection by 0.5
                    # - Each connection is bidirectional, increment weight by
                    # - 0.5 to account for this (otherwise one connection adds a weight
                    # - of two)
                    if connection in edges:
                        edges[connection]["Weight"] += 0.5
                    else:
                        edges[connection] = {"Weight": 0.5}

    # Parse connections & weights into a suitable format for NetworkX to interpret
    edges_to_add = [tuple(list(key) + [value['Weight']]) for key, value in edges.items()]

    # Initialize graph with one node for each category
    graph = nx.Graph()
    for node, cat in zip(node_list, cat_list):
        graph.add_node(node, code=cat, desc=cat)

    # Add edges to the graph
    graph.add_weighted_edges_from(edges_to_add, weight='weight')

    # Package everything up for output
    output = {
        'graph': graph,
        'nodes': node_list,
        'cats_to_nodes': cats_to_nodes,
        'nodes_to_cats': nodes_to_cats
    }

    return output


def plot_network_2d(X_nodes, Y_nodes,
                    X_edges, Y_edges,
                    L_nodes, C_nodes):
    '''Create a two-dimensional plot of the network graph.
    Adapted from: https://plotly.com/python/network-graphs/'''

    # Draw the lines between nodes
    edge_trace = go.Scatter(
        x=X_edges,
        y=Y_edges,
        line=dict(
            width=1/log(len(X_nodes)),
            color='rgb(150,150,150)'
        )
    )

    # Set marker properties
    marker = dict(
        showscale=False,
        colorscale='Viridis',
        reversescale=False,
        color=C_nodes,
        size=10,
    )

    # Draw a point for each node
    node_trace = go.Scatter(
        x=X_nodes,
        y=Y_nodes,
        text=L_nodes,
        mode='markers',
        hoverinfo='text',
        hoverlabel=dict(
            font=dict(
                size=10
            )
        ),
        marker=marker
    )

    data = [edge_trace, node_trace]

    # Set layout options
    layout = go.Layout(
        showlegend=False,
        hovermode='closest',
        margin=dict(b=0, l=0, r=0, t=0),
        xaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=False
        ),
        yaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            scaleanchor="x"
        ),
        autosize=True,
        height=None,
        width=None,
        plot_bgcolor='#f7f7f7'
    )

    figure = go.Figure(data, layout)

    return figure

def plot_network_3d(X_nodes, Y_nodes, Z_nodes,
                    X_edges, Y_edges, Z_edges,
                    L_nodes, C_nodes):
    '''Create a three-dimensional plot of the network graph.
    Adapted from: https://plotly.com/python/network-graphs/'''

    # Draw the lines between nodes
    edge_trace = go.Scatter3d(
        x=X_edges,
        y=Y_edges,
        z=Z_edges,
        mode='lines',
        line_width=1/log(len(X_nodes)),
        line_color='rgb(150,150,150)',
        hoverinfo='none'
    )

    # Set marker properties
    marker = dict(
        showscale=False,
        symbol='circle',
        colorscale='Viridis',
        reversescale=False,
        color=C_nodes,
        size=6
    )

    # Draw a point for each node
    node_trace = go.Scatter3d(
        x=X_nodes,
        y=Y_nodes,
        z=Z_nodes,
        text=L_nodes,
        mode='markers',
        hoverinfo='text',
        hoverlabel=dict(
            font=dict(
                size=10
            )
        ),
        marker=marker,
    )

    data = [edge_trace, node_trace]

    # Define properties common to each axis
    axis = dict(
        showbackground=False,
        showline=False,
        zeroline=False,
        showgrid=False,
        showticklabels=False,
        showspikes=False,
        title=dict(text='')
    )

    # Set layout options
    layout = go.Layout(
        showlegend=False,
        hovermode='closest',
        margin=dict(b=0, l=0, r=0, t=0),
        scene=dict(
            xaxis=axis,
            yaxis=axis,
            zaxis=axis,
        ),
        autosize=True,
        height=None,
        width=None,
        plot_bgcolor='#f7f7f7'
    )

    figure = go.Figure(data, layout)

    return figure


def create_plot_network(n_dims):
    '''Create a n-dimensional plot which visualizes the connections between
    categories in the training dataset'''

    # Generate the network graph, unpack output variables
    g_out = create_network()
    graph = g_out['graph']
    node_list = g_out['nodes']
    nodes_to_cats = g_out['nodes_to_cats']

    # Use the graph to calculate node positions in the n-dimensional space,
    # based on the relative connection weights
    layout = nx.drawing.layout.spring_layout(graph, weight='weight', dim=n_dims)

    # Unpack node positions from generated layout
    X_nodes = [layout[k][0] for k in node_list]
    Y_nodes = [layout[k][1] for k in node_list]
    if n_dims == 3:
        Z_nodes = [layout[k][2] for k in node_list]

    # Create corresponding array of node labels
    L_nodes = [nodes_to_cats[k] for k in node_list]

    # Set node colour according to the number of times the
    # category appeared in the training dataset
    C_nodes = [log(cat_counts[l]) for l in L_nodes]

    # Unpack start/end points for each edge in the graph
    X_edges, Y_edges, Z_edges, W_edges = [], [], [], []
    for edge in graph.edges(data=True):
        X_edges += [layout[edge[0]][0], layout[edge[1]][0], None]
        Y_edges += [layout[edge[0]][1], layout[edge[1]][1], None]
        if n_dims == 3:
            Z_edges += [layout[edge[0]][2], layout[edge[1]][2], None]

        W_edges += [edge[2]['weight'], edge[2]['weight'], None] # weight of connection

    # Create the plot
    if n_dims == 2:
        figure = plot_network_2d(X_nodes, Y_nodes,
                                 X_edges, Y_edges,
                                 L_nodes, C_nodes)
    elif n_dims == 3:
        figure = plot_network_3d(X_nodes, Y_nodes, Z_nodes,
                                 X_edges, Y_edges, Z_edges,
                                 L_nodes, C_nodes)

    return figure
