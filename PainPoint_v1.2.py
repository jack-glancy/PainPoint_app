import dash
from dash import html, dcc
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
import numpy as np
import stl
from stl.mesh import Mesh

# Initialize the Dash app
app = dash.Dash(__name__)

# Dummy path to STL file - replace with your actual path
STL_PATH = "C:/Users/jackg/Desktop/Useful/human.stl"


def load_stl_file(file_path):
    """Load STL file and extract vertices and faces"""
    try:
        mesh_data = Mesh.from_file(file_path)
        vertices = mesh_data.vectors.reshape(-1, 3)
        unique_vertices, indices = np.unique(vertices, axis=0, return_inverse=True)
        faces = indices.reshape(-1, 3)
        return unique_vertices, faces
    except Exception as e:
        print(f"Error loading STL file: {str(e)}")
        return None, None


def find_nearest_vertex(vertices, click_point):
    """Find the index of the nearest vertex to the clicked point"""
    click_coords = np.array([click_point['x'], click_point['y'], click_point['z']])
    distances = np.linalg.norm(vertices - click_coords, axis=1)
    return np.argmin(distances)


def calculate_model_size(vertices):
    """Calculate the diagonal size of the model's bounding box"""
    if len(vertices) == 0:
        return 1.0

    min_coords = np.min(vertices, axis=0)
    max_coords = np.max(vertices, axis=0)
    diagonal = np.linalg.norm(max_coords - min_coords)
    return diagonal


def create_3d_visualization(vertices, faces, clicked_point=None, point_size=5):
    """Create a 3D visualization of the STL file"""
    if vertices is None or faces is None:
        return go.Figure().add_annotation(
            text="Error loading STL file",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )

    # Calculate model size for scaling
    model_size = calculate_model_size(vertices)
    size_reference = model_size / 100  # Adjust this factor to change the relative size

    # Create the mesh
    fig = go.Figure(data=[
        go.Mesh3d(
            x=vertices[:, 0],
            y=vertices[:, 1],
            z=vertices[:, 2],
            i=faces[:, 0],
            j=faces[:, 1],
            k=faces[:, 2],
            flatshading=True,
            color='lightblue',
            opacity=0.8,
            hoverinfo='none'
        )
    ])

    # Add highlighted point if one is clicked
    if clicked_point is not None:
        fig.add_trace(
            go.Scatter3d(
                x=[vertices[clicked_point][0]],
                y=[vertices[clicked_point][1]],
                z=[vertices[clicked_point][2]],
                mode='markers',
                marker=dict(
                    size=point_size,
                    sizemode='diameter',
                    sizeref=size_reference,
                    color='red',
                    symbol='circle'
                ),
                name='Selected Point',
                hoverinfo='none'
            )
        )

    # Update the layout
    fig.update_layout(
        scene=dict(
            aspectmode='data',
            camera=dict(
                up=dict(x=0, y=0, z=1),
                center=dict(x=0, y=0, z=0),
                eye=dict(x=1.5, y=1.5, z=1.5)
            ),
            xaxis=dict(visible=False, showgrid=False, zeroline=False, showticklabels=False, showbackground=False),
            yaxis=dict(visible=False, showgrid=False, zeroline=False, showticklabels=False, showbackground=False),
            zaxis=dict(visible=False, showgrid=False, zeroline=False, showticklabels=False, showbackground=False),
            bgcolor='rgba(0,0,0,0)'
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        uirevision='constant',
        showlegend=False
    )

    return fig

# Load the STL file once at startup
vertices, faces = load_stl_file(STL_PATH)
initial_figure = create_3d_visualization(vertices, faces)

# App layout
app.layout = html.Div([
    html.H1("PainPoint",
            style={'textAlign': 'center', 'margin': '20px'}),
    
    html.Div([
        # 3D visualization
        html.Div([
            dcc.Graph(
                id='3d-stl-viewer',
                figure=initial_figure,
                style={'height': '80vh'},
                config={
                    'displayModeBar': True,
                    'modeBarButtonsToRemove': ['toImage'],
                    'displaylogo': False
                }
            ),
        ], style={'width': '70%'}),
        
        # Controls and info panel
        html.Div([
            html.H3("Visualization Controls"),
            html.P("Use mouse to rotate and zoom:"),
            html.Ul([
                html.Li("Left click + drag: Rotate"),
                html.Li("Right click + drag: Pan"),
                html.Li("Scroll: Zoom"),
                html.Li("Double click: Reset view"),
                html.Li("Click mesh: Select point")
            ]),
            html.Div(id='click-data', style={'marginTop': '20px'}),
            
            # Point size slider (hidden by default)
            html.Div(
                id='slider-container',
                style={'display': 'none', 'marginTop': '20px'},
                children=[
                    html.P("Pain Score:"),
                    dcc.Slider(
                        id='point-size-slider',
                        min=1,
                        max=10,
                        step=0.5,
                        value=5,
                        marks={i: str(i) for i in range(1, 11)},
                    )
                ]
            )
        ], style={
            'width': '30%',
            'padding': '20px',
            'backgroundColor': '#f8f9fa',
            'borderRadius': '5px'
        })
    ], style={
        'display': 'flex',
        'flexDirection': 'row',
        'margin': '20px'
    }),
    
    # Store the currently selected point
    dcc.Store(id='selected-point', data=None)
])

@app.callback(
    Output('slider-container', 'style'),
    [Input('selected-point', 'data')]
)
def toggle_slider(selected_point):
    if selected_point is None:
        return {'display': 'none', 'marginTop': '20px'}
    return {'display': 'block', 'marginTop': '20px'}

@app.callback(
    [Output('3d-stl-viewer', 'figure'),
     Output('selected-point', 'data'),
     Output('click-data', 'children')],
    [Input('3d-stl-viewer', 'clickData'),
     Input('point-size-slider', 'value')],
    [State('selected-point', 'data')]
)
def update_visualization(clickData, point_size, current_point):
    ctx = dash.callback_context
    if not ctx.triggered:
        return create_3d_visualization(vertices, faces), None, "No point selected"
    
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    try:
        if trigger_id == '3d-stl-viewer':
            if clickData is None:
                return create_3d_visualization(vertices, faces), None, "No point selected"
            
            click_point = clickData['points'][0]
            point_index = find_nearest_vertex(vertices, click_point)
        else:  # slider value changed
            point_index = current_point
            if point_index is None:
                return create_3d_visualization(vertices, faces), None, "No point selected"
        
        updated_fig = create_3d_visualization(vertices, faces, point_index, point_size)
        
        click_info = html.Div([
            html.P(f"Selected Point Index: {point_index}"),
            html.P(f"Coordinates: ({vertices[point_index][0]:.2f}, "
                   f"{vertices[point_index][1]:.2f}, "
                   f"{vertices[point_index][2]:.2f})")
        ])
        
        return updated_fig, point_index, click_info
        
    except Exception as e:
        print(f"Error processing interaction: {str(e)}")
        return create_3d_visualization(vertices, faces), None, "Error processing interaction"

if __name__ == '__main__':
    app.run_server(debug=True)
