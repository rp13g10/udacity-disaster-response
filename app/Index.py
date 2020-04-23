'''This module defines the overall page layout, and the content of static page elements
such as the nav bar.'''

import dash_core_components as dcc
import dash_daq as daq
import dash_html_components as html
import plotly.graph_objects as go
import waitress
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

# pylint: disable=unused-import
# Tokenizer import needed to unpickle the classifier (needs to be in __main__)
from App import app
from models.train_classifier import Tokenizer
from Utilities import cat_names, cat_counts
from Utilities import gen_jumbotron, create_plot_network, generate_word_cloud


####################################################################################################
# Page Header                                                                                      #
####################################################################################################

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

page_header = html.Nav(
    [
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



####################################################################################################
# Page Content                                                                                     #
####################################################################################################

# Headline text ####################################################################################

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


# User input #######################################################################################

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


# Category display #################################################################################

cat_display = html.Div(
    html.Div(
        html.Div(),
        id='jumbotron',
        className='col-10 offset-1'
    ),
    className='row mb-3'
)


# Charts ###########################################################################################

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
)


# Word Cloud ---------------------------------------------------------------------------------------

cloud_desc = "Select a category to view the words which most commonly appeared \
within it. The larger the word, the more common it was in the training dataset."

cloud_controls = html.Div(
    [
        html.Div(
            "Word Cloud",
            className='card-header'
        ),
        html.Div(
            [
                html.P(
                    cloud_desc,
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


word_cloud = html.Div(
    [
        html.Div(
            dcc.Loading(
                html.Img(
                    style={'align-self': 'center'},
                    className='img-fluid mx-auto rounded border border-light',
                    id='word-cloud'
                )
            ),
            className='col-7 offset-1 text-center'
        ),
        html.Div(
            cloud_controls,
            className='col-3'
        )
    ],
    className='row mb-3'
)


# Network Graph ------------------------------------------------------------------------------------

network_desc = "This network graph shows how frequently different categories appeared \
together in the same message. The closer together two categories are, the more frequently \
they appeared together. For example, 'buildings' and 'earthquake' are commonly reported \
together."


network_controls = html.Div(
    [
        html.Div(
            "Network Graph",
            className='card-header'
        ),
        html.Div(
            [
                html.P(
                    network_desc,
                    className='card-text'),
                html.P(
                    "Activate the toggle below to enable an exciting 3D plot!",
                    className='card-text'
                )
            ],
            className='card-body'
        ),
        html.Div(
            [
                html.Div(
                    [
                        html.Div(
                            # pylint: disable=not-callable
                            # False positive error raised on BooleanSwitch
                            daq.BooleanSwitch(
                                id='network-dims',
                                on=False,
                                className='d-inline'
                            ),
                            className='col-6'
                        ),
                        html.Div(
                            html.Button(
                                'Draw',
                                id='network-update',
                                className='btn btn-primary h-100 px-3 mx-1 d-inline',
                                n_clicks=0
                            ),
                            className='col-6'
                        )
                    ],
                    className='row'
                )
            ],
            className='card-footer text-center'
        )
    ],
    className='card'
)

network_graph = html.Div(
    [
        html.Div(
            dcc.Loading(
                dcc.Graph(figure=blank_figure, id='network-graph'),
            ),
            className='col-7 offset-1'
        ),
        html.Div(
            network_controls,
            className='col-3'
        )
    ],
    className='row mb-3'
)


# Content Layout ###################################################################################

# Set the order in which each element is displayed
page_content = html.Div(
    [
        page_title,
        msg_input,
        cat_display,
        chart_header,
        word_cloud,
        network_graph
    ],
    id='page-content'
)



####################################################################################################
# App Configuration                                                                                #
####################################################################################################

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

# pylint: disable=no-member, unused-argument
@app.callback(
    [Output('msg-input', 'value'), Output('jumbotron', 'children')],
    [Input('go-button', 'n_clicks')],
    [State('msg-input', 'value')])
def display_input(btn_clicks, usr_input):
    '''Displays the input text, tagged with named entities and sentiment'''
    if not usr_input:
        return "", html.Div()
    else:
        jumbotron = gen_jumbotron(usr_input)
        return "", jumbotron

@app.callback(
    Output('network-graph', 'figure'),
    [Input('network-update', 'n_clicks')],
    [State('network-dims', 'on')])
def update_network_graph(btn_clicks, excitement_flag):
    '''Generates a network graph which helps to visualize the categories which
    most frequently appear together in the training dataset'''

    if not btn_clicks:
        raise PreventUpdate
    n_dims = 3 if excitement_flag else 2
    figure = create_plot_network(n_dims)
    return figure

@app.callback(
    Output('word-cloud', 'src'),
    [Input('cloud-category', 'value')])
def update_word_cloud(category):
    '''Generate a word cloud showing the most common words for a
    given category in the training dataset'''
    return generate_word_cloud(category)


# Run Server #######################################################################################

if __name__ == '__main__':
    server = app.server
    waitress.serve(server)
