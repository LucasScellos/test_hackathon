import plotly.express as px
import pandas as pd
from sklearn.linear_model import LinearRegression
import plotly.graph_objects as go

def hello():
    return "Hello"

#Luigia
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
                      mapbox_center={'lat': latitude[0], 'lon': longitude[0]}, title = commune, width=300)
  

    # Show the plot
    return fig

#tania
def filtre_temporel(df, date_debut, date_fin):

    date_debut = pd.to_datetime(date_debut).strftime('%Y-%m-%d')
    date_fin = pd.to_datetime(date_fin).strftime('%Y-%m-%d')

    df_filtre = df[(df['DATE'] >= date_debut) & (df['DATE'] <= date_fin)]

    return df_filtre

def apply_fct (df, func):
    return df.apply(func)


def temp_max(df):
    return df['T_Q'].max()


def temp_min(df):
    return df['T_Q'].min()


def temp_moy(df):
    return df['T_Q'].mean()


def calc_nb_j(df, seuil, signe):
    cpt = 0
    df['DATE'] = pd.to_datetime(df['DATE'])
    resultats = []

    for annee in df['DATE'].dt.year.unique():
        donnees_annee = df[df['DATE'].dt.year == annee]
        if signe == "+": 
            cpt = donnees_annee[donnees_annee['T_Q'] >=  seuil].shape[0]
            
        elif signe == "-":
            cpt = donnees_annee[donnees_annee['T_Q'] <=  seuil].shape[0]

        else: 
            print("Signe non reconnu, veuillez choisir entre '+' et '-' ")
    
        resultats.append({'Année': annee, 'Nb_jours_max': cpt})

    df_final = pd.DataFrame(resultats)

    return df_final



def calc_nb_episode(df, seuil, signe, nb_j_min):
    df['DATE'] = pd.to_datetime(df['DATE'])
    resultats = []

    for annee in df['DATE'].dt.year.unique():
        nb_fois_condition_true = 0
        donnees_annee = df[df['DATE'].dt.year == annee]
        if signe == "+": 
            donnees_annee['Condition_Respectee'] = donnees_annee.apply(lambda row: row['T_Q'] >= seuil, axis=1)
            
        elif signe == "-":
            donnees_annee['Condition_Respectee'] = donnees_annee.apply(lambda row: row['T_Q'] <= seuil, axis=1)

        else: 
            print("Signe non reconnu, veuillez choisir entre '+' et '-' ")
        
        donnees_annee.reset_index(drop=True, inplace=True)
        condition_true_consecutifs = donnees_annee['Condition_Respectee'].rolling(window=nb_j_min).apply(lambda x: x.all()).fillna(False)
        nb_fois_condition_true = (condition_true_consecutifs.diff() == True).sum()
        
        resultats.append({'Année': annee, 'nb_episodes': nb_fois_condition_true})

    df_final = pd.DataFrame(resultats)

    return df_final



#paul-etienne
def create_df_index_var_metier (df_index, df_var_metier):

    df_var_metier.rename(columns = {list(df_var_metier)[0]: 0}, inplace = True)
    df_var_metier.rename(columns = {list(df_var_metier)[1]: 1}, inplace = True)
    df_index.rename(columns = {list(df_index)[0]: "DATE"}, inplace = True)
    df_index.rename(columns = {list(df_index)[1]: "index"}, inplace = True)

    df = df_index.merge(df_var_metier, left_on="DATE", right_on=1)
    
    return df

def load_csv (df_index) :
    uploaded_file = st.file_uploader("Charger un fichier csv (col1 : variable, col2 : année)")
    if uploaded_file is not None:
        df_var_metier = pd.read_csv(uploaded_file)
        print("Chargé")
        #st.write(df_var_metier)

    return create_df_index_var_metier(df_index, df_var_metier)  


def corr_df (df) :
    return df[["index",0]].corr()#[0]["index"]



def modele_baseline_train (df):
    X, Y, annee = df[["index"]], df[0], df["DATE"]
    
    regr = LinearRegression()
    regr.fit(X,Y)
    
    return regr


def plot_reg (df, regr) :
    
    y_pred = regr.predict(df[["index"]])
    fig = go.Figure()
    # Add traces
    X, _,_ = df[["index"]], df[0], df["DATE"]

    fig.add_trace(go.Scatter(x=df["index"], y=df[0],
                        mode='markers',
                        name='observations'))
    fig.add_trace(go.Scatter(x=df["index"], y=y_pred,
                        mode='lines+markers',
                        name='predictions'))
    
    fig.update_layout(
                xaxis_title="index climatique", yaxis_title="variable"
            )
    return fig

def plot_reg_temporel (df, regr, df_drias) :
    
    annee = df["DATE"]
    Y = df[0]
    y_pred = regr.predict(df[["index"]])
    
    df_drias = df_drias[df_drias["Année"]>2024]
    annee_drias = df_drias["Année"]

    df_drias.rename(columns = {list(df_drias)[1]: "index"}, inplace = True)
    y_pred_drias = regr.predict(df_drias[["index"]])

    
    fig = go.Figure()
    # Add traces
    fig.add_trace(go.Scatter(x=annee, y=Y,
                        mode='markers',
                        name='observations'))
    fig.add_trace(go.Scatter(x=annee_drias, y=y_pred_drias,
                        mode='lines+markers',
                        name='predictions DRIAS'))
    fig.add_trace(go.Scatter(x=annee, y=y_pred,
                        mode='lines+markers',
                        name='predictions'))
    fig.update_layout(
                xaxis_title="Annee", yaxis_title="variable"
            )
    return fig


def main_inspect_csv(df_ind, df_mf, df_drias):
    #df = load_csv (df_index)
    df = create_df_index_var_metier(df_mf, df_ind)
    corr = corr_df(df)
    #st.write(f"Correlation entre la variable et l'indicateur climatique : \n{}")

    regr = modele_baseline_train (df)
    fig1 = plot_reg (df, regr)
    fig2 = plot_reg_temporel (df, regr, df_drias)
    return corr, fig1, fig2 