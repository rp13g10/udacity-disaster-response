'''This module defines the overall page layout, and the content of static page elements
such as the nav bar.'''

import base64
import pickle
from io import BytesIO

import dash_core_components as dcc
import dash_html_components as html
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import plotly.graph_objects as go
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
from numpy import log
from numpy.random import random
from wordcloud import WordCloud, STOPWORDS

from App import app
from models.train_classifier import load_data, Tokenizer

# Load external dependencies #######################################################################

with open('../models/model.pkl', 'rb') as f:
    classifier = pickle.load(f)

X, y, cat_names = load_data('../data/udacity.db')

cat_counts = {
    cat: y[:, inx].sum()
    for inx, cat in enumerate(cat_names)
}

# URL Bar ##########################################################################################

# Set up the element which allows the report to determine
# which page the user is attempting to access.
url_bar_content = html.Div(
    [
        # Represents the URL bar, doesn't render anything
        dcc.Location(id='url', refresh=False)
    ]
)


# Page Header ######################################################################################

# Udacity logo
# "Disaster Response Project"
# GitHub repo link

page_header = html.Nav(
    [
        # Left hand side - Logo & page title
        html.A(
            [
                html.Img(
                    src='./assets/udacity_logo.png',
                    style={'height': '48px', 'width': '48px'},
                    className="d-inline-block mr-2 rounded align-top"),
                html.P(
                    'Disaster Response Project',
                    id='page-title',
                    className='m-0 p-0 d-inline-block align-top')
            ],
            className='navbar-brand align-middle',
            style={'font-size': '32px'},
            href='/'
        ),
        html.A(
            html.Img(
                src='./assets/github_logo.png',
                style={'height': '48px', 'width': '48px'},
                className="d-inline-block mr-2"
            ),
            className='align-middle my-0',
            href="https://github.com/rp13g10/"
        )
    ],
    className='navbar navbar-dark bg-dark py-1 my-0'
)


# Page Content #####################################################################################

# Headline text ------------------------------------------------------------------------------------
page_title = html.Div(
    [
        html.H1(
            "Disaster Response Project",
            className="display-3 text-center w-100"
        ),
        html.H2(
            "Analyzing message data for disaster response",
            className="text-muted text-center w-100"
        )
    ],
    className='row pb-4'
)


# User input ---------------------------------------------------------------------------------------
msg_control = html.Div(
    [
        html.Div(
            dcc.Input(
                id='msg-input',
                type='text',
                className='w-100 h-100 m-0 p-0'
            ),
            className='col-11 m-0 p-0'
        ),
        html.Div(
            dcc.Link(
                html.Button(
                    'Go',
                    id='go-button',
                    className='btn btn-default w-100 h-100',
                    n_clicks=0
                ),
                id='go-button-link',
                className='w-100 h-100 m-0 p-0'
            ),
            className='col-1 m-0 p-0'
        )
    ],
    className='row p-0 m-0 h-100'
)

msg_input = html.Div(
    html.Div(
        [
            html.Div(
                html.Div(
                    'Classify Message',
                    className='input-group-text'
                ),
                className='input-group-prepend'
            ),
            html.Div(
                msg_control,
                className='form-control h-100 p-0'
            )
        ],
        className='input-group col-10 offset-1'
    ),
    className='row mb-3'
)


# Category display ---------------------------------------------------------------------------------

def gen_cat_card(cat):
    cat = cat.replace('_', ' ')
    cat = cat.title()
    card = html.Div(
        html.P(
            cat,
            className='text-center m-0'
        ),
        className='alert alert-primary align-middle py-1 px-2 mr-2',
        role='alert'
    )
    return card


def gen_jumbotron(msg_input):
    X = [msg_input]
    y = classifier.predict(X)
    y = y[0]

    cats = [
        name
        for cat, name in zip(y, cat_names)
        if cat
    ]

    cats = [gen_cat_card(cat) for cat in cats]

    jumbotron = html.Div(
        [
            html.H1("Message Categories"),
            html.P(msg_input, id='msg-display', className='lead'),
            html.Hr(className='my-4'),
            html.Div(
                cats,
                id='cat-display',
                className='d-flex flex-row flex-wrap'),
        ],
        className='jumbotron'
    )

    return jumbotron


cat_display = html.Div(
    html.Div(
        html.Div(),
        id='jumbotron',
        className='col-10 offset-1'
    ),
    className='row mb-3'
)


# Charts -------------------------------------------------------------------------------------------

chart_header = html.Div(
    [
        html.H1(
            "Training Data Summary",
            className="display-4 text-center w-100"
        )
    ],
    className='row pb-4'
)

blank_figure = go.Figure(
    layout=go.Layout(
        showlegend=False,
        margin=dict(b=0,l=0,r=0,t=0),
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
)

# Chart 1 - Wordcloud per topic

cloud_controls = html.Div(
    [
        html.Div(
            "Word Cloud",
            className='card-header'
        ),
        html.Div(
            [
                html.P(
                    "This is a word cloud! Description coming soon!",
                    className='card-text'),
            ],
            className='card-body'
        ),
        html.Div(
            dcc.Dropdown(
                id='cloud-category',
                options=[
                    {'label': x, 'value': x}
                    for x in cat_names
                    if cat_counts[x] > 1
                ],
                value=cat_names[0]
            ),
            className='card-footer text-center'
        )
    ],
    className='card'
)

def fig_to_uri(in_fig, close_all=True, **save_args):
    # type: (plt.Figure) -> str
    """
    Save a figure as a URI
    :param in_fig:
    :return:
    """
    out_img = BytesIO()
    in_fig.savefig(out_img, format='png', **save_args)
    if close_all:
        in_fig.clf()
        plt.close('all')
    out_img.seek(0)  # rewind file
    encoded = base64.b64encode(out_img.read()).decode("ascii").replace("\n", "")
    return "data:image/png;base64,{}".format(encoded)


def generate_word_cloud(category):
    cat_selector = cat_names.index(category)
    cat_mask = pd.Series(y[:,cat_selector]).astype(bool)
    cat_text = pd.Series(X).loc[cat_mask]

    cat_text = " ".join(cat_text.dropna()).lower()

    cloud = WordCloud(
        background_color='#f7f7f7',
        mode='RGBA',
        stopwords=STOPWORDS,
        normalize_plurals=True,
        collocations=True,
        width=1240,
        height=720
    ).generate(cat_text)

    fig = plt.figure(1, figsize=(12.4, 7.2), dpi=100)

    plt.axes([0, 0, 1, 1,], label=str(random()))

    plt.imshow(cloud, interpolation='bilinear', aspect='equal')
    plt.axis('off')
    imstr = fig_to_uri(fig, bbox_inches=0,  dpi=100, figsize=(12.4, 7.2), transparent=True)

    return imstr

word_cloud = html.Div(
    [
        html.Div(
            html.Img(
                style={'align-self': 'center'},
                className='img-fluid mx-auto rounded border border-light',
                id='word-cloud'
            ),
            className='col-7 offset-1'
        ),
        html.Div(
            cloud_controls,
            className='col-3'
        )
    ],
    className='row mb-3'
)

# Chart 2 - Network graph showing co-occurrences
def create_network():
    cat_df = pd.DataFrame(y, columns=cat_names)

    cat_df = cat_df.drop(columns=['child_alone'])
    cat_names_copy = cat_names[:]
    cat_names_copy.remove('child_alone')

    nodes_to_cats = {}
    cats_to_nodes = {}
    for inx, cat in enumerate(cat_names_copy):
        nodes_to_cats[inx] = cat
        cats_to_nodes[cat] = inx

    node_list = nodes_to_cats.keys()
    cat_list = [nodes_to_cats[x] for x in node_list]

    edges = {}
    for inx, row in cat_df.iterrows():
        cats = row.loc[row==1]
        cats = list(cats.keys())
        if len(cats) >= 2:
            for cat_from in cats:
                cats_copy = cats[:]
                cats_copy.remove(cat_from)

                node_from = cats_to_nodes[cat_from]

                for cat_to in cats_copy:
                    node_to = cats_to_nodes[cat_to]
                    connection = tuple(sorted([node_from, node_to]))

                    if connection in edges:
                        edges[connection]["Weight"] += 0.5
                    else:
                        edges[connection] = {"Weight": 0.5}

    edges_to_add = [tuple(list(key) + [value['Weight']]) for key, value in edges.items()]

    G = nx.Graph()

    for node, cat in zip(node_list, cat_list):
        G.add_node(node, code=cat, desc=cat)

    G.add_weighted_edges_from(edges_to_add, weight='weight')

    output = {
        'graph': G,
        'nodes': node_list,
        'cats_to_nodes': cats_to_nodes,
        'nodes_to_cats': nodes_to_cats
    }

    return output

def plot_network_2d(X_nodes, Y_nodes,
                    X_edges, Y_edges,
                    L_nodes, C_nodes):

    edge_trace = go.Scatter(
        x=X_edges,
        y=Y_edges,
        line=dict(
            width=1/log(len(X_nodes)),
            color='rgb(150,150,150)'
        )
    )

    marker = dict(
        showscale=False,
        colorscale='Viridis',
        reversescale=False,
        color=C_nodes,
        size=10,
    )

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

    layout = go.Layout(
        showlegend=False,
        hovermode='closest',
        margin=dict(b=0,l=0,r=0,t=0),
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

    edge_trace = go.Scatter3d(
        x=X_edges,
        y=Y_edges,
        z=Z_edges,
        mode='lines',
        line_width=1/log(len(X_nodes)),
        line_color='rgb(150,150,150)',
        hoverinfo='none'
    )

    marker = dict(
        showscale=False,
        symbol='circle',
        colorscale='Viridis',
        reversescale=False,
        color=C_nodes,
        size=6
    )

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

    axis = dict(
        showbackground=False,
        showline=False,
        zeroline=False,
        showgrid=False,
        showticklabels=False,
        showspikes=False,
        title=dict(text='')
    )

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

    n_dims = int(n_dims)

    g_out = create_network()
    G = g_out['graph']
    node_list = g_out['nodes']
    cats_to_nodes = g_out['cats_to_nodes']
    nodes_to_cats = g_out['nodes_to_cats']

    layout = nx.drawing.layout.spring_layout(G, weight='weight', dim=n_dims)

    X_nodes = [layout[k][0] for k in node_list]
    Y_nodes = [layout[k][1] for k in node_list]
    if n_dims == 3:
        Z_nodes = [layout[k][2] for k in node_list]

    L_nodes = [nodes_to_cats[k] for k in node_list]
    C_nodes = [log(cat_counts[l]) for l in L_nodes]

    X_edges, Y_edges, Z_edges, W_edges = [], [], [], []

    for e in G.edges(data=True):
        X_edges += [layout[e[0]][0], layout[e[1]][0], None]
        Y_edges += [layout[e[0]][1], layout[e[1]][1], None]
        if n_dims == 3:
            Z_edges += [layout[e[0]][2], layout[e[1]][2], None]

        W_edges += [e[2]['weight'], e[2]['weight'], None] # weight of connection

    if n_dims == 2:
        figure = plot_network_2d(X_nodes, Y_nodes,
                                 X_edges, Y_edges,
                                 L_nodes, C_nodes)
    elif n_dims == 3:
        figure = plot_network_3d(X_nodes, Y_nodes, Z_nodes,
                                 X_edges, Y_edges, Z_edges,
                                 L_nodes, C_nodes)

    return figure


network_controls = html.Div(
    [
        html.Div(
            "Network Graph",
            className='card-header'
        ),
        html.Div(
            [
                html.P(
                    "This is a network graph! Description coming soon!",
                    className='card-text')
            ],
            className='card-body'
        ),
        html.Div(
            [
                dcc.RadioItems(
                    id='network-dims',
                    options=[
                        {'label': '2D ', 'value': '2'},
                        {'label': '3D ', 'value': '3'}
                    ],
                    value='2',
                    className='d-inline'
                ),
                html.Button(
                    'Draw',
                    id='network-update',
                    className='btn btn-primary h-100 px-3 mx-1 d-inline',
                    n_clicks=0
                ),
            ],
            className='card-footer text-center'
        )
    ],
    className='card'
)

network_graph = html.Div(
    [
        html.Div(
            dcc.Graph(figure=blank_figure, id='network-graph'),
            className='col-7 offset-1'
        ),
        html.Div(
            network_controls,
            className='col-3'
        )
    ],
    className='row mb-3'
)

# Content Layout -----------------------------------------------------------------------------------
page_content = html.Div(
    [
        page_title,
        msg_input,
        cat_display,
        chart_header,
        network_graph,
        word_cloud
    ],
    id='page-content'
)


# Page Initialization ##############################################################################

def serve_layout():
    '''Defines the macro-level page layout (nav bar, page content, etc)'''
    layout = html.Div(
        [
            url_bar_content,
            page_header,
            html.Div(
                page_content,
                className='container-fluid pb-5 pt-3'
            ),
            html.Div(className='row m-0 p-0 w-100', style={'height': '100px'})
        ],
        className='container-fluid p-0 m-0'
    )

    return layout

app.layout = serve_layout


# Callbacks ########################################################################################

# pylint: disable=no-member
@app.callback(
    [Output('msg-input', 'value'), Output('jumbotron', 'children')],
    [Input('go-button', 'n_clicks')],
    [State('msg-input', 'value')])
def display_input(btn_clicks, msg_input):
    '''Displays the input text, tagged with named entities and sentiment'''
    if not msg_input:
        return "", html.Div()
    else:
        jumbotron = gen_jumbotron(msg_input)
        return "", jumbotron

@app.callback(
    Output('network-graph', 'figure'),
    [Input('network-update', 'n_clicks')],
    [State('network-dims', 'value')])
def update_network_graph(btn_clicks, n_dims):
    if not btn_clicks:
        raise PreventUpdate
    figure = create_plot_network(n_dims)
    return figure

@app.callback(
    Output('word-cloud', 'src'),
    [Input('cloud-category', 'value')])
def update_word_cloud(category):
    return generate_word_cloud(category)


# Run Server #######################################################################################
if __name__ == '__main__':
    app.run_server(
        debug=True,
        threaded=True
    )
