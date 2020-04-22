'''This module defines the underlying flask app which will run the
dashboard. Keeping it separate is helpful for larger apps which implement logins
and other more complex functionality.'''

import dash

app = dash.Dash(
    __name__)

server = app.server
