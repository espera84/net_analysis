import dash
from dash import dcc, html
import plotly.graph_objs as go
from dash.dependencies import Input, Output
import random

# Crea un'app Dash
app = dash.Dash(__name__)

# Dati iniziali per un grafico 3D di esempio
x_data = [1, 2, 3, 4, 5]
y_data = [5, 4, 3, 2, 1]
z_data = [1, 2, 3, 4, 5]

# Traccia iniziale
trace = go.Scatter3d(
    x=x_data,
    y=y_data,
    z=z_data,
    mode='markers'
)

layout = go.Layout(
    scene=dict(
        xaxis=dict(range=[0, 6]),
        yaxis=dict(range=[0, 6]),
        zaxis=dict(range=[0, 6]),
    )
)

# Layout dell'app Dash
app.layout = html.Div([
    dcc.Graph(id='3d-plot', figure={'data': [trace], 'layout': layout}),
    html.Div(id='camera-info', style={'padding': '20px', 'fontSize': '18px'})
])


# Funzione di callback per aggiornare la posizione della camera e aggiungere punti
@app.callback(
    Output('3d-plot', 'figure'),
    Output('camera-info', 'children'),
    Input('3d-plot', 'relayoutData')
)
def update_camera_info(relayoutData):
    # Aggiorna la posizione della camera
    camera_info = "Muovi la vista per aggiornare la posizione della camera..."
    if 'scene.camera' in relayoutData:
        camera = relayoutData['scene.camera']
        camera_info = f"Posizione Camera: x={camera['eye']['x']}, y={camera['eye']['y']}, z={camera['eye']['z']}"

    # Aggiungi nuovi punti casuali ai dati
    new_x = random.randint(0, 6)
    new_y = random.randint(0, 6)
    new_z = random.randint(0, 6)

    # Aggiungi i nuovi punti ai dati esistenti
    x_data.append(new_x)
    y_data.append(new_y)
    z_data.append(new_z)

    # Crea una nuova traccia con i punti aggiunti
    new_trace = go.Scatter3d(
        x=x_data,
        y=y_data,
        z=z_data,
        mode='markers'
    )

    # Crea una nuova figura con la traccia aggiornata
    new_figure = {
        'data': [new_trace],
        'layout': layout
    }

    return new_figure, camera_info


if __name__ == '__main__':
    app.run_server(debug=True)
