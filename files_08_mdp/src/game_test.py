# visit http://127.0.0.1:8050/ in your web browser.


from dash import Dash, html, dcc, Input, Output, State, callback
import plotly.express as px
import pandas as pd
import random

app = Dash(__name__)

policy_csv = pd.read_csv("policy.csv")
#FMDP_csv = pd.read_csv("FDMP.csv")

states = list(policy_csv["State"].unique())
best_actions = list(policy_csv["Best Actions"])
#actions = list(FMDP_csv["Action"].unique())

app.layout = html.Div(children=[
    html.H1(children='Game test 📱'),
    
    html.H2("Simulation"),

    html.Div(children='''
        In this scenario, you will find yourself in a communication cell with three users: 
        one at the edge of the cell and two close to the base station (BS), with these two
        users also being close to each other.
    '''),
    
    html.Div(children='''
        At each Transmission Time Interval (TTI), the base station can serve two users 
        simultaneously, using two frequency bands: low (L) and high (H).
    '''),
    
    html.Div(children='''
        Our resource allocation is designed to protect the peripheral user u2. Buffers are 
        represented by tuples (b0, b1, b2), where b0 is the buffer of u0, b1 is the buffer 
        of u1 and b2 is the buffer of u2, and the occurrence of interference is represented 
        by 'I' or 'N '. Transmission actions are represented by tuples (u0, u1, H, L),where 
        u0 and u1 are the selected users, H and L are the selected frequencies.
    '''),
    
    html.Div(children='''
        Packets transmitted per user are represented by tuples (d0, d1) or alternatively 
        (1, 1, 0). For example, if the buffers are initially empty (0, 0, 0) and there is 
        no interference ('N'), and we choose to serve u0 and u1 with high and low frequencies 
        respectively, we will have two packets transmitted per user,and the buffers in the next
        instant will become (0, 0, 1).
    '''),

    dcc.Store(id='random', data=0),
    
    html.H4("Given the following state, define the best action:"),
    
    html.Div(children='Current state:',id="State_random"),
    
    html.Button('Random State', id='button-1'),
    
    html.Div(children='Choose actions:',id="action_list"),
    
    #dcc.Dropdown(actions,id='actions_list'),
    
    html.Label('User (0/1/2):'),
    dcc.Checklist(id='checklist-u', options=[
        {'label': '0', 'value': 0},
        {'label': '1', 'value': 1},
        {'label': '2', 'value': 2},
    ], value=[]),
    
    html.Label('Carry 1 (Low/High):'),
    dcc.RadioItems(id='radio-1', options=[
        {'label': 'Low', 'value': 'L'},
        {'label': 'High', 'value': 'H'},
    ], value=''),
    
    html.Label('Carry 2 (Low/High):'),
    dcc.RadioItems(id='radio-2', options=[
        {'label': 'Low', 'value': "L"},
        {'label': 'High', 'value': "H"},
    ], value=""),
    
    html.Button('Submit', id='button-compare'),
    
    html.Div(id="result-text"),
    
    html.Button('Response', id='button-3'),
    
    html.Div(id='response'),
    
    dcc.Store(id="store_respose", data="")
    
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
    Input('button-3', 'n_clicks'),
    State('random', 'data'),
    State('store_respose', 'data')
)
def update_output(n_clicks,data,resp):
    if n_clicks is None:
        return ''
    if resp in best_actions[data]:
        resposta = "Correct action"
        #resposta = resp
    else:
        resposta = "Wrong action"
        #resposta = resp
    return f'Best Actions: {str(best_actions[data])}\n your action is {resposta}'
    
# Callback para atualizar o resultado quando o botão for pressionado
@app.callback(
    Output('store_respose', 'data'),
    [Input('button-compare', 'n_clicks')],
    [Input('radio-1', 'value'),
     Input('radio-2', 'value'),
     Input('checklist-u', 'value')
    ]
)
def comparar_solucao(n_clicks, c1, c2, u):
    if n_clicks is None:
        return ""
    else:
        #pontuacao_jogador = calcular_pontuacao(c, u)
        # Aqui você pode comparar pontuacao_jogador com a solução ótima e gerar uma mensagem de feedback.
        entrada = []
        u.sort()
        entrada.append(u[0])
        entrada.append(u[1])
        entrada.append(c1)
        entrada.append(c2)
        #mensagem = f"{entrada} Pontuação = {pontuacao_jogador}"
        #mensagem = f"{entrada}"
        
        mensagem = f"{tuple(entrada)}"
        return mensagem

        #return mensagem
@app.callback(
    Output('result-text', 'children'),
    Input('store_respose', 'data'),
)
def display_store_response(data):
    store = data
    return f'Action: {store}'
        

if __name__ == '__main__':
    app.run(debug=True)
    