import streamlit as st
from utils import utils_hackathon as uh
import pandas as pd
st.title("Hello Météo-France , bienvenue sur Climate Viz")

dict_indicateurs = {"T_MAX" : "Temperature maximale"}
c1, c2 = st.columns(2)
ctn = c1.expander("Paramètre")
col11, col12 = ctn.columns(2)


fig2 = uh.map_commune("Montpellier", [43.61361315241169], [3.875541887925083])




commune = col11.selectbox("Choississez votre commune", ["Commune", "Marseille", "Montpellier", "Niort"])
if commune !="Commune":
    c2.plotly_chart(fig2)
scenario = col12.selectbox("Scénario Climatique", ["Scénario", "RCP2.6", "RCP4.5" , "RCP8.5"])
ind = col11.selectbox("Choississez un indicateur", ["Température Max","Température Moyenne",  "Température Min", "Température Seuil"])
if ind == "Température Seuil":
    #col3, col4 = col12.columns(2)
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
    test = uh.calc_nb_episode(df, seuil, signe, nb_jour_cons)
    #mf
    df_2 = pd.read_csv("data/Serie_tempo_T_montpellier_daily_1959_2024.csv")
    test_2 = uh.calc_nb_episode(df_2, seuil, signe, nb_jour_cons)

if (ind=="Température Max"):
    df_drias_ind = pd.read_csv("data/test_plot.csv")
    fig = uh.plot_climate_strips(df_drias_ind, "T_MAX", "01/08", "31/08",dict_indicateurs)
    st.plotly_chart(fig, width=2000)










