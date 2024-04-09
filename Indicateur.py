import streamlit as st
from utils import utils_hackathon as uh
import pandas as pd
st.title("Hello Météo-France , bienvenue sur Climate Viz")

dict_indicateurs = {"T_MAX" : "Temperature maximale"}
c1, c2 = st.columns(2)
ctn = c1.expander("Paramètre")
col11, col12 = ctn.columns(2)


fig2 = uh.map_commune("Montpellier", [43.61361315241169], [3.875541887925083])


end_day=False

commune = col11.selectbox("Choississez votre commune", ["Commune", "Marseille", "Montpellier", "Niort"])
if commune !="Commune":
    c2.plotly_chart(fig2)
scenario = col12.selectbox("Scénario Climatique", ["Scénario", "RCP2.6", "RCP4.5" , "RCP8.5"])
ind = col11.selectbox("Choississez un indicateur", ["Température Max","Température Moyenne",  "Température Min", "Température Seuil"])
date_perso = col11.checkbox("Date Personnalisée")

#selection date
if date_perso:
    exc1 = c1.expander("Sélection Date Personnalisée")
    exc11, exc12 = exc1.columns(2)
    start_day  = exc11.text_input("Date de Départ", "01/01")
    end_day = exc12.text_input("Date de Fin", "31/12")
    # start_day = exc11.number_input('Start Day', min_value=1, max_value=31, value=1)
    # start_month = exc12.number_input('Start Month', min_value=1, max_value=12, value=1)
    # end_day = exc11.number_input('End Day', min_value=1, max_value=31, value=1)
    # end_month = exc12.number_input('End Month', min_value=1, max_value=12, value=1)

if ind == "Température Seuil":
    nb_jour_cons = col12.number_input("Séléctionner un nombre de jour consécutif",1,365)
    seuil = col12.number_input("Séléctionner une température seuil (°C)", -10, 45)
    choix_seuil = col12.radio("Choix seuil", ["Température Min", "Température Supérieur"])
    if choix_seuil == "Température Min":
        signe = "-"
    else:
        signe = "+"
    #drias
    df = pd.read_csv("data/drias_montpellier_df.csv")
    df["T_Q"] = df["T_Q"]-273.15
    
    df_drias = uh.calc_nb_episode(df, seuil, signe, nb_jour_cons)
    #mf
    df_2 = pd.read_csv("data/Serie_tempo_T_montpellier_daily_1959_2024.csv")
    df_mf = uh.calc_nb_episode(df_2, seuil, signe, nb_jour_cons)
    st.dataframe(df_mf)
    st.dataframe(df_drias)



if (ind=="Température Max" and commune=="Montpellier" and end_day=="30/09"):
    df_drias_ind = pd.read_csv("data/test_plot.csv")
    #df = uh.filtre_temporel_periode(df, "01-07", "30-09")
    fig = uh.plot_climate_strips(df_drias_ind, "T_MAX", "01/07", "30/09",dict_indicateurs)

    st.plotly_chart(fig, width=2000)

#metrique
nb_1 = 5
nb_2 = 8
nb_3 = 11

if ((ind == "Température Seuil") or (ind=="Température Max")):
    container = st.expander("Nb jour supérieur à 25°C", expanded=True)
    col1, col2, col3 = container.columns(3)
    col1.metric("Horizon 1995", nb_1)
    col2.metric("Horizon 2020", nb_2, str(nb_2-nb_1)+" jours par rapport à la période de référence")
    col3.metric("Horizon 2050", nb_3, str(nb_3-nb_1)+" jours par rapport à la période de référence")








