import plotly.express as px


def hello():
    return "Hello"
def plot_climate_strips(df, indicateur, periode_start, periode_end,  dict_indicateurs):
    fig = px.line(df, x='ANNEE', y='rolling_avg', labels={'rolling_avg': 'Écart à la moyenne (°C)', "ANNEE" : "Année"}, color_discrete_sequence=['green'], 
              title=f'{dict_indicateurs[indicateur]} entre le {periode_start} et le {periode_end}.<br>Écart à la moyenne de référence 1951 à 1980')

    fig.add_bar(x=df['ANNEE'], y=df['ANOM_'+indicateur], name=f'{indicateur} (°C)', marker=dict(color=df[indicateur], coloraxis="coloraxis"))
    fig.update_layout(coloraxis=dict(colorscale='RdYlBu_r'), width=1000)
    fig.update_traces(hovertemplate='Année: %{x}<br>Anomalie: %{y:.1f}°')

    return fig


def map_commune(commune, latitude, longitude):

    data = {'Latitude': latitude, 'Longitude': longitude}

    # Plot the point on the map
    fig = px.scatter_mapbox(data, lat='Latitude', lon='Longitude', zoom=10)

    # Customize map layout
    fig.update_layout(mapbox_style='open-street-map', mapbox_zoom=10, 
                      mapbox_center={'lat': latitude[0], 'lon': longitude[0]}, title = commune)
  

    # Show the plot
    return fig