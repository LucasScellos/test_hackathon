import pandas as pd
import xarray as xr 



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
    cpt = 0
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