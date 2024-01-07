# visit http://127.0.0.1:8050/ in your web browser.


from dash import Dash, html, dcc, Input, Output, State, callback
import plotly.express as px
import pandas as pd
import random

app = Dash(__name__)

policy_csv = pd.read_csv("policy.csv")
FMDP_csv = pd.read_csv("FDMP.csv")

states = list(policy_csv["State"].unique())
best_actions = list(policy_csv["Best Actions"])
actions = list(FMDP_csv["Action"].unique())

app.layout = html.Div(children=[
    html.H1(children='Game test 📱'),

    html.H3(children='''
        Choose the best action so that users can send packages
    '''),
    dcc.Store(id='random', data=0),
    
    html.Div(children='Current state:',id="State_random"),
    
    html.Button('Random State', id='button-1'),
    
    html.Div(children='Choose actions:',id="action_list"),
    
    dcc.Dropdown(actions,id='actions_list'),
    
    html.Button('Response', id='button-2'),
    
    html.Div(id='response')
    
])

@app.callback(
    Output('random', 'data'),
    Input('button-1', 'n_clicks')
)
def update_store(n_clicks):
    random_state = random.randint(0, len(states)-1)
    return random_state

@app.callback(
    Output('State_random', 'children'),
    Input('random', 'data'),
)
def display_store_info(data):
    return f'Current state: {states[data]}'

@app.callback(
    Output('response', 'children'),
    Input('button-2', 'n_clicks'),
    State('random', 'data')
)
def update_output(n_clicks,data):
    if n_clicks is None:
        return ''
    return f'Best Actions: {best_actions[data]}'

if __name__ == '__main__':
    app.run(debug=True)
    