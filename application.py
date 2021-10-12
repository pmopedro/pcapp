import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.express as px

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.datasets import load_boston
from sklearn.preprocessing import StandardScaler
import umap


df = pd.read_csv('data.csv')

app = dash.Dash(__name__)
application = app.server
app.title='PCA analysis on an AWS EB'

app.layout = html.Div([
    html.H1('PCA: Youtube Dataset'),
    dcc.Graph(id="graph"),
    html.P("Number of components:"),
    dcc.Slider(
        id='slider',
        min=2, max=10, value=4,
        marks={i: str(i) for i in range(2,10)}),
    html.Div(style={'padding':'1.5em'}),
    html.Hr(),
    dcc.Graph(id="bidim-pca"),
    html.Div(style={'padding':'1.5em'}),
    html.Hr(),
    dcc.Graph(id="means"),
    html.Div(style={'padding':'1.5em'}),
    # html.Hr(),
    # dcc.Graph(id="u-map"),
    # html.Div(style={'padding':'1.5em'}),
    # html.Hr(),
    # dcc.Graph(id="mds")
])
from sklearn.metrics import silhouette_samples, silhouette_score

@app.callback(
    Output("graph", "figure"), 
    [Input("slider", "value")])
def run_and_plot(n_components):
    le = StandardScaler()
    pca_data = pd.DataFrame(le.fit_transform(df.iloc[:,:119]))
    pca = PCA(n_components=n_components)
    components = pca.fit_transform(pca_data)

    

    var = pca.explained_variance_ratio_.sum() * 100

    labels = {str(i): f"PC {i+1}" 
              for i in range(n_components)}
    labels['color'] = 'Median Price'
    
    silhouette_avg = silhouette_score(pca_data, df.category_name)
        
    fig = px.scatter_matrix(
        components,
        color=df.category_name,
        dimensions=range(n_components),
        labels=labels,
        title=f'PCA- Total Explained Variance: {var:.2f}%, silhouette: {silhouette_avg: .2f}')
    # fig.update_traces(diagonal_visible=False)
    fig.update_layout(
    dragmode='select',
    height=600,
    hovermode='closest',
    )

    return fig





@app.callback(
    Output("bidim-pca", "figure"), 
    [Input("slider", "value")])
def run_and_plot(n_components):
    le = StandardScaler()
    pca_data = pd.DataFrame(le.fit_transform(df.iloc[:,:119]))
    pca = PCA(n_components=n_components)
    components = pca.fit_transform(pca_data)

    fig =px.scatter(
        components,
        x=0,
        y=1,
        color=df.category_name,
        title='Bidimensional pca')
    
    return fig



@app.callback(
    Output("means", "figure"), 
    [Input("slider", "value")])
def run_and_plot(n_components):
    le = StandardScaler()
    means = pd.DataFrame(le.fit_transform(df.iloc[:,:119]))
    means['category_name']= df.category_name
    
    category_means= means.groupby('category_name').std().T.iloc[1:-1]
    fig =px.line(
        category_means,
        labels={'index':'feature'},
        title='Median per category'
        )

    return fig

# @app.callback(
#     Output("u-map", "figure"), 
#     [Input("slider", "value")])
# def run_and_plot(n_components):
#     le = StandardScaler()
#     data = pd.DataFrame(le.fit_transform(df.iloc[:,:119]))

#     umap_2d = umap.UMAP()

#     proj_2d = umap_2d.fit_transform(data)
    
#     fig = px.scatter(
#         proj_2d, x=0, y=1,
#         color=df.category_name, 
#         labels={'color': 'category'},
#         title='umap_2d'
#     )

#     return fig

# from sklearn.manifold import MDS

# @app.callback(
#     Output("mds", "figure"), 
#     [Input("slider", "value")])
# def run_and_plot(n_components):
#     le = StandardScaler()
#     data = pd.DataFrame(le.fit_transform(df.iloc[:,:119]))

#     mds = embedding = MDS(n_components=2)

#     proj_2d = mds.fit_transform(data)
    
#     fig = px.scatter(
#         proj_2d, x=0, y=1,
#         color=df.category_name, 
#         labels={'color': 'category'},
#         title='mds'
#     )

#     return fig

if __name__ == '__main__':
    application.run(debug=False,port=8000)