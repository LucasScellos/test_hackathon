import streamlit as st
from utils import utils_hackathon as uh
import pandas as pd
st.title("Hello Météo-France , bienvenue sur Climate Viz")

dict_indicateurs = {"T_MAX" : "Temperature maximale"}

ctn = st.expander("Paramètre")
col1, col2 = ctn.columns(2)



commune = col1.selectbox("Choississez votre commune", ["Commune", "Marseille", "Montpellier", "Niort"])
scenario = col2.selectbox("Scénario Climatique", ["Scénario", "RCP2.6", "RCP4.5" , "RCP8.5"])
ind = col1.selectbox("Choississez un indicateur", ["Indicateur","Température Moyenne", "Température Max", "Température Min", "Température Seuil"])
if ind == "Température Seuil":
    col2.slider("Séléctionner un nombre de jour consécutif",1,10)
if (scenario!="Scénario" and commune!="Commune"):
    st.write("Nice jobs")
    df_drias_ind = pd.read_csv("test_plot.csv")
    fig = uh.plot_climate_strips(df_drias_ind, "T_MAX", "01/08", "31/08",dict_indicateurs)
    st.plotly_chart(fig, width=2000)


    fig2 = uh.map_commune("Montpellier", [43.61361315241169], [3.875541887925083])
    st.plotly_chart(fig2)







