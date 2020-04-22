'''This module defines the overall page layout, and the content of static page elements
such as the nav bar.'''

import pickle

import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

from App import app
from models.train_classifier import load_data, Tokenizer

# Load external dependencies #######################################################################

with open('../models/model.pkl', 'rb') as f:
    classifier = pickle.load(f)

X, y, cat_names = load_data('../data/udacity.db')

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

header_text = html.Div(
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

cat_display = html.Div(
    html.Div(
        html.Div(),
        id='jumbotron',
        className='col-10 offset-1'
    ),
    className='row mb-3'
)


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



# Temporary placeholder
page_content = html.Div(
    [
        header_text,
        msg_input,
        cat_display
    ],
    id='page-content'
)

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


if __name__ == '__main__':
    app.run_server(
        debug=True,
        threaded=True
    )
