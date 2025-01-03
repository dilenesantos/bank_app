import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

import scipy.stats as stats

import statsmodels.api

 
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

from streamlit_option_menu import option_menu
from streamlit_extras.no_default_selectbox import selectbox

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb
from xgboost import XGBClassifier

from sklearn import neighbors
from sklearn import svm
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.metrics import classification_report

import joblib
import shap


df=pd.read_csv('dilenesantos/bank.csv')

dff = df.copy()
dff = dff[dff['age'] < 75]
dff = dff.loc[dff["balance"] > -2257]
dff = dff.loc[dff["balance"] < 4087]
dff = dff.loc[dff["campaign"] < 6]
dff = dff.loc[dff["previous"] < 2.5]
bins = [-2, -1, 180, 855]
labels = ['Prospect', 'Reached-6M', 'Reached+6M']
dff['Client_Category_M'] = pd.cut(dff['pdays'], bins=bins, labels=labels)
dff['Client_Category_M'] = dff['Client_Category_M'].astype('object')
liste_annee =[]
for i in dff["month"] :
    if i == "jun" or i == "jul" or i == "aug" or i == "sep" or i == "oct" or i == "nov" or i == "dec" :
        liste_annee.append("2013")
    elif i == "jan" or i == "feb" or i == "mar" or i =="apr" or i =="may" :
        liste_annee.append("2014")
dff["year"] = liste_annee
dff['date'] = dff['day'].astype(str)+ '-'+ dff['month'].astype(str)+ '-'+ dff['year'].astype(str)
dff['date']= pd.to_datetime(dff['date'])
dff["weekday"] = dff["date"].dt.weekday
dic = {0 : "Lundi", 1 : "Mardi", 2 : "Mercredi", 3 : "Jeudi", 4 : "Vendredi", 5 : "Samedi", 6 : "Dimanche"}
dff["weekday"] = dff["weekday"].replace(dic)

dff = dff.drop(['contact'], axis=1)
dff = dff.drop(['pdays'], axis=1)
dff = dff.drop(['day'], axis=1)
dff = dff.drop(['date'], axis=1)
dff = dff.drop(['year'], axis=1)
dff['job'] = dff['job'].replace('unknown', np.nan)
dff['education'] = dff['education'].replace('unknown', np.nan)
dff['poutcome'] = dff['poutcome'].replace('unknown', np.nan)

X = dff.drop('deposit', axis = 1)
y = dff['deposit']

# Séparation des données en un jeu d'entrainement et jeu de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 48)
                        
# Remplacement des NaNs par le mode:
imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
X_train.loc[:,['job']] = imputer.fit_transform(X_train[['job']])
X_test.loc[:,['job']] = imputer.transform(X_test[['job']])

# On remplace les NaaN de 'poutcome' avec la méthode de remplissage par propagation où chaque valeur unknown est remplacée par la valeur de la ligne suivante (puis la dernière ligne par le Mode de cette variable).
# On l'applique au X_train et X_test :
X_train['poutcome'] = X_train['poutcome'].fillna(method ='bfill')
X_train['poutcome'] = X_train['poutcome'].fillna(X_train['poutcome'].mode()[0])

X_test['poutcome'] = X_test['poutcome'].fillna(method ='bfill')
X_test['poutcome'] = X_test['poutcome'].fillna(X_test['poutcome'].mode()[0])

# On fait de même pour les NaaN de 'education'
X_train['education'] = X_train['education'].fillna(method ='bfill')
X_train['education'] = X_train['education'].fillna(X_train['education'].mode()[0])

X_test['education'] = X_test['education'].fillna(method ='bfill')
X_test['education'] = X_test['education'].fillna(X_test['education'].mode()[0])
                        
# Standardisation des variables quantitatives:
scaler = StandardScaler()
cols_num = ['age', 'balance', 'duration', 'campaign', 'previous']
X_train [cols_num] = scaler.fit_transform(X_train [cols_num])
X_test [cols_num] = scaler.transform (X_test [cols_num])

# Encodage de la variable Cible 'deposit':
le = LabelEncoder()
y_train = le.fit_transform(y_train)
y_test = le.transform(y_test)

# Encodage des variables explicatives de type 'objet'
oneh = OneHotEncoder(drop = 'first', sparse_output = False)
cat1 = ['default', 'housing','loan']
X_train.loc[:, cat1] = oneh.fit_transform(X_train[cat1])
X_test.loc[:, cat1] = oneh.transform(X_test[cat1])

X_train[cat1] = X_train[cat1].astype('int64')
X_test[cat1] = X_test[cat1].astype('int64')

# 'education' est une variable catégorielle ordinale, remplacer les modalités de la variable par des nombres, en gardant l'ordre initial
X_train['education'] = X_train['education'].replace(['primary', 'secondary', 'tertiary'], [0, 1, 2])
X_test['education'] = X_test['education'].replace(['primary', 'secondary', 'tertiary'], [0, 1, 2])

# 'Client_Category_M' est une variable catégorielle ordinale, remplacer les modalités de la variable par des nombres, en gardant l'ordre initial
X_train['Client_Category_M'] = X_train['Client_Category_M'].replace(['Prospect', 'Reached-6M', 'Reached+6M'], [0, 1, 2])
X_test['Client_Category_M'] = X_test['Client_Category_M'].replace(['Prospect', 'Reached-6M', 'Reached+6M'], [0, 1, 2])


# Encoder les variables à plus de 2 modalités 'job', 'marital', 'poutome', 'month', 'weekday' pour X_train
dummies = pd.get_dummies(X_train['job'], prefix='job').astype(int)
X_train = pd.concat([X_train.drop('job', axis=1), dummies], axis=1)
dummies = pd.get_dummies(X_test['job'], prefix='job').astype(int)
X_test = pd.concat([X_test.drop('job', axis=1), dummies], axis=1)

dummies = pd.get_dummies(X_train['marital'], prefix='marital').astype(int)
X_train = pd.concat([X_train.drop('marital', axis=1), dummies], axis=1)
dummies = pd.get_dummies(X_test['marital'], prefix='marital').astype(int)
X_test = pd.concat([X_test.drop('marital', axis=1), dummies], axis=1)

dummies = pd.get_dummies(X_train['poutcome'], prefix='poutcome').astype(int)
X_train = pd.concat([X_train.drop('poutcome', axis=1), dummies], axis=1)
dummies = pd.get_dummies(X_test['poutcome'], prefix='poutcome').astype(int)
X_test = pd.concat([X_test.drop('poutcome', axis=1), dummies], axis=1)

dummies = pd.get_dummies(X_train['month'], prefix='month').astype(int)
X_train = pd.concat([X_train.drop('month', axis=1), dummies], axis=1)
dummies = pd.get_dummies(X_test['month'], prefix='month').astype(int)
X_test = pd.concat([X_test.drop('month', axis=1), dummies], axis=1)

dummies = pd.get_dummies(X_train['weekday'], prefix='weekday').astype(int)
X_train = pd.concat([X_train.drop('weekday', axis=1), dummies], axis=1)
dummies = pd.get_dummies(X_test['weekday'], prefix='weekday').astype(int)
X_test = pd.concat([X_test.drop('weekday', axis=1), dummies], axis=1)

#Récupération des valeurs originales à partir des données standardisées
X_train_original = X_train.copy()
X_test_original = X_test.copy()

#Inversion de la standardisation
X_train_original[cols_num] = scaler.inverse_transform(X_train[cols_num])
X_test_original[cols_num] = scaler.inverse_transform(X_test[cols_num])

#code python SANS DURATION
dff_sans_duration = df.copy()
dff_sans_duration = dff_sans_duration[dff_sans_duration['age'] < 75]
dff_sans_duration = dff_sans_duration.loc[dff_sans_duration["balance"] > -2257]
dff_sans_duration = dff_sans_duration.loc[dff_sans_duration["balance"] < 4087]
dff_sans_duration = dff_sans_duration.loc[dff_sans_duration["campaign"] < 6]
dff_sans_duration = dff_sans_duration.loc[dff_sans_duration["previous"] < 2.5]
dff_sans_duration = dff_sans_duration.drop('contact', axis = 1)

bins = [-2, -1, 180, 855]
labels = ['Prospect', 'Reached-6M', 'Reached+6M']
dff_sans_duration['Client_Category_M'] = pd.cut(dff_sans_duration['pdays'], bins=bins, labels=labels)
dff_sans_duration['Client_Category_M'] = dff_sans_duration['Client_Category_M'].astype('object')
dff_sans_duration = dff_sans_duration.drop('pdays', axis = 1)

liste_annee =[]
for i in dff_sans_duration["month"] :
    if i == "jun" or i == "jul" or i == "aug" or i == "sep" or i == "oct" or i == "nov" or i == "dec" :
        liste_annee.append("2013")
    elif i == "jan" or i == "feb" or i == "mar" or i =="apr" or i =="may" :
        liste_annee.append("2014")
dff_sans_duration["year"] = liste_annee
dff_sans_duration['date'] = dff_sans_duration['day'].astype(str)+ '-'+ dff_sans_duration['month'].astype(str)+ '-'+ dff_sans_duration['year'].astype(str)
dff_sans_duration['date']= pd.to_datetime(dff_sans_duration['date'])
dff_sans_duration["weekday"] = dff_sans_duration["date"].dt.weekday
dic = {0 : "Lundi", 1 : "Mardi", 2 : "Mercredi", 3 : "Jeudi", 4 : "Vendredi", 5 : "Samedi", 6 : "Dimanche"}
dff_sans_duration["weekday"] = dff_sans_duration["weekday"].replace(dic)

dff_sans_duration = dff_sans_duration.drop(['day'], axis=1)
dff_sans_duration = dff_sans_duration.drop(['date'], axis=1)
dff_sans_duration = dff_sans_duration.drop(['year'], axis=1)
dff_sans_duration = dff_sans_duration.drop(['duration'], axis=1)

dff_sans_duration['job'] = dff_sans_duration['job'].replace('unknown', np.nan)
dff_sans_duration['education'] = dff_sans_duration['education'].replace('unknown', np.nan)
dff_sans_duration['poutcome'] = dff_sans_duration['poutcome'].replace('unknown', np.nan)

X_sans_duration = dff_sans_duration.drop('deposit', axis = 1)
y_sans_duration = dff_sans_duration['deposit']

# Séparation des données en un jeu d'entrainement et jeu de test
X_train_sd, X_test_sd, y_train_sd, y_test_sd = train_test_split(X_sans_duration, y_sans_duration, test_size = 0.20, random_state = 48)
                
# Remplacement des NaNs par le mode:
imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
X_train_sd.loc[:,['job']] = imputer.fit_transform(X_train_sd[['job']])
X_test_sd.loc[:,['job']] = imputer.transform(X_test_sd[['job']])
 
# On remplace les NaaN de 'poutcome' avec la méthode de remplissage par propagation où chaque valeur unknown est remplacée par la valeur de la ligne suivante (puis la dernière ligne par le Mode de cette variable).
# On l'applique au X_train et X_test :
X_train_sd['poutcome'] = X_train_sd['poutcome'].fillna(method ='bfill')
X_train_sd['poutcome'] = X_train_sd['poutcome'].fillna(X_train_sd['poutcome'].mode()[0])

X_test_sd['poutcome'] = X_test_sd['poutcome'].fillna(method ='bfill')
X_test_sd['poutcome'] = X_test_sd['poutcome'].fillna(X_test_sd['poutcome'].mode()[0])

# On fait de même pour les NaaN de 'education'
X_train_sd['education'] = X_train_sd['education'].fillna(method ='bfill')
X_train_sd['education'] = X_train_sd['education'].fillna(X_train_sd['education'].mode()[0])

X_test_sd['education'] = X_test_sd['education'].fillna(method ='bfill')
X_test_sd['education'] = X_test_sd['education'].fillna(X_test_sd['education'].mode()[0])
            
# Standardisation des variables quantitatives:
scaler_sd = StandardScaler()
cols_num_sd = ['age', 'balance', 'campaign', 'previous']
X_train_sd[cols_num_sd] = scaler_sd.fit_transform(X_train_sd[cols_num_sd])
X_test_sd[cols_num_sd] = scaler_sd.transform (X_test_sd[cols_num_sd])

# Encodage de la variable Cible 'deposit':
le_sd = LabelEncoder()
y_train_sd = le_sd.fit_transform(y_train_sd)
y_test_sd = le_sd.transform(y_test_sd)

# Encodage des variables explicatives de type 'objet'
oneh_sd = OneHotEncoder(drop = 'first', sparse_output = False)
cat1_sd = ['default', 'housing','loan']
X_train_sd.loc[:, cat1_sd] = oneh_sd.fit_transform(X_train_sd[cat1_sd])
X_test_sd.loc[:, cat1_sd] = oneh_sd.transform(X_test_sd[cat1_sd])

X_train_sd[cat1_sd] = X_train_sd[cat1_sd].astype('int64')
X_test_sd[cat1_sd] = X_test_sd[cat1_sd].astype('int64')

# 'education' est une variable catégorielle ordinale, remplacer les modalités de la variable par des nombres, en gardant l'ordre initial
X_train_sd['education'] = X_train_sd['education'].replace(['primary', 'secondary', 'tertiary'], [0, 1, 2])
X_test_sd['education'] = X_test_sd['education'].replace(['primary', 'secondary', 'tertiary'], [0, 1, 2])

# 'Client_Category_M' est une variable catégorielle ordinale, remplacer les modalités de la variable par des nombres, en gardant l'ordre initial
X_train_sd['Client_Category_M'] = X_train_sd['Client_Category_M'].replace(['Prospect', 'Reached-6M', 'Reached+6M'], [0, 1, 2])
X_test_sd['Client_Category_M'] = X_test_sd['Client_Category_M'].replace(['Prospect', 'Reached-6M', 'Reached+6M'], [0, 1, 2])


# Encoder les variables à plus de 2 modalités 'job', 'marital', 'poutome', 'month', 'weekday' pour X_train
dummies_sd = pd.get_dummies(X_train_sd['job'], prefix='job').astype(int)
X_train_sd = pd.concat([X_train_sd.drop('job', axis=1), dummies_sd], axis=1)
dummies_sd = pd.get_dummies(X_test_sd['job'], prefix='job').astype(int)
X_test_sd = pd.concat([X_test_sd.drop('job', axis=1), dummies_sd], axis=1)

dummies_sd = pd.get_dummies(X_train_sd['marital'], prefix='marital').astype(int)
X_train_sd = pd.concat([X_train_sd.drop('marital', axis=1), dummies_sd], axis=1)
dummies_sd = pd.get_dummies(X_test_sd['marital'], prefix='marital').astype(int)
X_test_sd = pd.concat([X_test_sd.drop('marital', axis=1), dummies_sd], axis=1)

dummies_sd = pd.get_dummies(X_train_sd['poutcome'], prefix='poutcome').astype(int)
X_train_sd = pd.concat([X_train_sd.drop('poutcome', axis=1), dummies_sd], axis=1)
dummies_sd = pd.get_dummies(X_test_sd['poutcome'], prefix='poutcome').astype(int)
X_test_sd = pd.concat([X_test_sd.drop('poutcome', axis=1), dummies_sd], axis=1)

dummies_sd = pd.get_dummies(X_train_sd['month'], prefix='month').astype(int)
X_train_sd = pd.concat([X_train_sd.drop('month', axis=1), dummies_sd], axis=1)
dummies_sd = pd.get_dummies(X_test_sd['month'], prefix='month').astype(int)
X_test_sd = pd.concat([X_test_sd.drop('month', axis=1), dummies_sd], axis=1)

dummies_sd = pd.get_dummies(X_train_sd['weekday'], prefix='weekday').astype(int)
X_train_sd = pd.concat([X_train_sd.drop('weekday', axis=1), dummies_sd], axis=1)
dummies_sd = pd.get_dummies(X_test_sd['weekday'], prefix='weekday').astype(int)
X_test_sd = pd.concat([X_test_sd.drop('weekday', axis=1), dummies_sd], axis=1)

#Récupération des valeurs originales à partir des données standardisées
X_train_sd_original = X_train_sd.copy()
X_test_sd_original = X_test_sd.copy()

#Inversion de la standardisation
X_train_sd_original[cols_num_sd] = scaler_sd.inverse_transform(X_train_sd[cols_num_sd])
X_test_sd_original[cols_num_sd] = scaler_sd.inverse_transform(X_test_sd[cols_num_sd])

with st.sidebar:
    selected = option_menu(
        menu_title='Sections',
        options=['Introduction','DataVisualisation', "Pre-processing", "Modélisation", "Interprétation", "Recommandations & Perspectives", "Outil  Prédictif"]) 

if selected == 'Introduction':  
    st.title("Prédiction du succès d’une campagne Marketing pour une banque")
    st.subheader("Contexte du projet")
    st.write("Le projet vise à analyser des données marketing issues d'une banque qui a utilisé le télémarketing pour **promouvoir un produit financier appelé 'dépôt à terme'**. Ce produit nécessite que le client dépose une somme d'argent dans un compte dédié, sans possibilité de retrait avant une date déterminée. En retour, le client reçoit des intérêts à la fin de cette période. **L'objectif de cette analyse est d'examiner les informations personnelles des clients, comme l'âge, le statut matrimonial, le montant d'argent déposé, le nombre de contacts réalisés, etc., afin de comprendre les facteurs qui influencent la décision des clients de souscrire ou non à ce produit financier.**")
    

    st.write("#### Problématique : ")
    st.write("La principale problématique de ce projet est de **déterminer** les **facteurs qui influencent la probabilité qu'un client souscrive à un dépôt à terme à la suite d'une campagne de télémarketing.**")
    st.write("L'objectif est double :")
    st.write("- Identifier et analyser visuellement et statistiquement **les caractéristiques des clients** qui sont corrélées avec la souscription au 'dépôt à terme'.")
    st.write("- Utiliser des techniques de Machine Learning pour **prédire si un client va souscrire au 'dépôt à terme'.**")

    st.write("#### Les données : ")
    st.markdown("Le jeu de données comprend un total de **11 162 lignes** et **17 colonnes**.  \n\
    Ces colonnes fournissent 3 types d'informations :")
    st.write("") 
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**I. Infos socio-démo:**  \n\
        1. age  \n\
        2. job  \n\
        3. marital  \n\
        4. education")
    with col2:
        st.markdown("**II. Infos situation bancaire:**  \n\
        5. default  \n\
        6. balance  \n\
        7. housing  \n\
        8. loan")
        st.write("") 

    with col3:
        st.markdown("**III. Infos campagnes marketing:** \n\
        9. contact  \n\
        10. day  \n\
        11. month  \n\
        12. duration  \n\
        13. campaign  \n\
        14. pdays  \n\
        15. previous  \n\
        16. poutcome")
        st.write("") 
        
    st.write("**Notre variable cible:**  \n\
    17. deposit")
    

    
    

    
    
   
if selected == 'DataVisualisation':      
    pages = st.sidebar.radio("", ["Analyse Univariée", "Analyse Multivariée", "Profiling"])

    if pages == "Analyse Univariée" :  # Analyse Univariée
        st.title("Analyse Univariée")

        # Liste des variables qualitatives et quantitatives
        quantitative_vars = ["age", "duration", "campaign", "balance", "pdays", "previous"]
        qualitative_vars = ["job", "marital", "education", "default", "housing", "loan", 
                            "contact", "poutcome", "deposit", "weekday", "month"]

        # Sélection du type de variable        
        analysis_type = st.radio(
            " ",
            ["**VARIABLES QUALITATIVES**", "**VARIABLES QUANTITATIVES**"],
            key="type_variable_selectbox", horizontal=True
        )

        # Affichage des variables en fonction du type choisi
        if analysis_type == "**VARIABLES QUALITATIVES**":
            selected_variable = st.radio(
                " ",
                qualitative_vars,
                key="qualitative_var_selectbox", horizontal=True
            )
            st.write("____________________________________")

            st.write(f"Analyse de la variable qualitative : **{selected_variable}**")

            #creation des colonnes year, month_year, date, weekday
            liste_annee =[]
            for i in df["month"] :
                if i == "jun" or i == "jul" or i == "aug" or i == "sep" or i == "oct" or i == "nov" or i == "dec" :
                    liste_annee.append("2013")
                elif i == "jan" or i == "feb" or i == "mar" or i =="apr" or i =="may" :
                    liste_annee.append("2014")
            df["year"] = liste_annee

            df['date'] = df['day'].astype(str)+ '-'+ df['month'].astype(str)+ '-'+ df['year'].astype(str)
            df['date']= pd.to_datetime(df['date'])

            df["weekday"] = df["date"].dt.weekday
            dic = {0 : "Lundi",
            1 : "Mardi",
            2 : "Mercredi",
            3 : "Jeudi",
            4 : "Vendredi",
            5 : "Samedi",
            6 : "Dimanche"}
            df["weekday"] = df["weekday"].replace(dic)

            # Analyse spécifique pour les variables qualitatives
            st.write("### Distribution des catégories")

            # Calcul des pourcentages
            category_counts = df[selected_variable].value_counts()
            category_percentages = category_counts / category_counts.sum() * 100
            
            # Création du graphique avec barres horizontales
            fig, ax = plt.subplots(figsize=(7, 4))
            sns.countplot(
                y=selected_variable,  # Passer `y` pour un graphique horizontal
                data=df,
                color='c',
                order=category_counts.index,
                ax=ax
            )
            ax.set_ylabel("")
            
            # Ajouter les annotations pourcentages sur les barres
            for i, count in enumerate(category_counts):
                percentage = category_percentages.iloc[i]
                ax.text(count + 0.5, i, f"{percentage: .0f}%", va="center", fontsize=7)  # `va="center"` pour centrer verticalement
            
            # Afficher le graphique dans Streamlit
            st.pyplot(fig)
            st.write("Le graphique ci-dessus montre la proportion de chaque catégorie dans la variable.")


        elif analysis_type == "**VARIABLES QUANTITATIVES**":
            selected_variable = st.radio(
                " ",
                quantitative_vars,
                key="quantitative_var_selectbox", horizontal=True
            )
            st.write("____________________________________")

            st.write(f"Analyse de la variable quantitative : **{selected_variable}**")

            # 1. Histogramme avec KDE
            # Créer un bouton
            st.write("### Distribution (Histogramme et KDE)") 
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.histplot(df[selected_variable], bins=20, kde=True, color='b', ax=ax)
            ax.set_title(f'Distribution de {selected_variable}', fontsize=14)
            ax.set_xlabel(selected_variable, fontsize=12)
            ax.set_ylabel('Fréquence', fontsize=12)
            st.pyplot(fig)

            # 2. Boxplot
            st.write("### Box Plot") 
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.boxplot(df[selected_variable], vert=False, patch_artist=True, 
                    boxprops=dict(facecolor='lightblue'))
            ax.set_title(f'Box Plot de {selected_variable}', fontsize=14)
            ax.set_xlabel(selected_variable, fontsize=12)
            st.pyplot(fig)

            # 3. QQ Plot
            st.write("### QQ Plot") 
            fig = plt.figure(figsize=(10, 6))
            stats.probplot(df[selected_variable], dist="norm", plot=plt)
            plt.title(f"QQ Plot de {selected_variable}", fontsize=14)
            st.pyplot(fig)

            # Ajouter les commentaires spécifiques pour chaque variable
            if selected_variable == "age":
                st.write("""
                **Commentaires pour 'Age':**
                - La distribution de la variable 'age' s'approche d'une distribution normale malgré des distorsions aux extrémités.
                - Le jeu de données affiche une concentration des tranches d'âge 25-40 ans suivi de la tranche 40-65 ans.
                - 50% des clients ont entre 32 et 49 ans.
                - Le boxplot montre quelques valeurs extrêmes supérieures à 74 ans.
                """)
            elif selected_variable == "duration":
                st.write("""
                **Commentaires:**
                - On remarque que duration ne suit pas une distribution normale
                - 50% des appels ont une durée entre 138 et 496s (soit entre 2.3 et 8.26 min).
                - La variable présente de nombreuses valeurs extrêmes entre 1033 et 3000s.
                - Quelques valeurs très extrêmes dépassent 3000s.
                """)
            elif selected_variable == "campaign":
                st.write("""
                **Commentaires:**
                - On remarque que campaign ne suit pas une distribution normale
                - 50% du volume de contacts se situe entre 1 et 3 contacts.
                - Le boxplot montre de nombreuses valeurs extrêmes supérieures au seuil max de 6 contacts.
                - On note 3 valeurs très extrêmes supérieures à 40.
                """)
            elif selected_variable == "balance":
                st.write("""
                **Commentaires:**
                - On remarque que balance ne suit pas une distribution normale
                - 50% des clients ont une balance entre 122 et 1708€.
                - Le boxplot montre de nombreuses valeurs extrêmes concentrées entre 4087€ et 40 000€.
                - Quelques valeurs très extrêmes atteignent 81 204€.
                """)
            elif selected_variable == "pdays":
                st.write("""
                **Commentaires:**
                - On remarque que pdays ne suit pas une distribution normale
                - La valeur -1 revient constamment, signifiant que la personne n'a jamais été contactée auparavant.
                - Cette valeur a donc une signification qualitative.
                """)
            elif selected_variable == "previous":
                st.write("""
                **Commentaires:**
                - On remarque que previous ne suit pas une distribution normale
                - La valeur 0 correspond aux clients pour lesquels 'Pdays' est égal à -1.
                - Parmi les clients contactés auparavant, 50% l'ont été entre 1 et 4 fois.
                - Le boxplot montre quelques valeurs extrêmes supérieures à 8.5 contacts.
            """)

            
    


    if pages == "Analyse Multivariée" : 
        # Title and Introduction 
        st.title("Analyse Multivariée")
    
    # Define sub-pages
        sub_pages = [
            "Matrice de corrélation",
            "Tests statistiques" 
            "Évolution dans le temps"
        ]

        # Sidebar for sub-page selection
        
        if st.checkbox('**Matrice de corrélation**') :
            cor = df[['age', 'balance', 'duration', 'campaign', 'pdays', 'previous']].corr()
            fig, ax = plt.subplots()
            sns.heatmap(cor, annot=True, ax=ax, cmap='rainbow')
            st.write(fig)
            st.write("""Le tableau de corrélation entre toutes les variables quantitatives de notre base de donnée révèle des coefficients 
            de corrélation très proche de 0. Cela signifie que nos variables quantitatives ne sont pas corrélées entre elles.""")

        if st.checkbox("**Tests statistiques**") :
            submenu_tests = st.radio(" ", ["Tests d'ANOVA", "Tests de Chi-deux"], horizontal = True)
            
            if submenu_tests == "Tests d'ANOVA" : 
                st.header("Les variables quantitatives sont-elles liées à notre variable cible ?")
                sub_pages1 = st.radio(" ", ["Lien âge x deposit", "Lien balance x deposit", "Lien duration x deposit", "Lien campaign x deposit", "Lien previous x deposit", "Conclusion"]
                                      , key = "variable_selectbox",  horizontal=True)
                
                st.write("____________________________________")
    
                st.subheader(f"Analyse du {sub_pages1}")
    
                if sub_pages1 == "Lien âge x deposit" :
                    fig = plt.figure(figsize=(4,2))
                    sns.kdeplot(df[df['deposit'] == 'yes']['age'], label='Yes', color='blue')
                    sns.kdeplot(df[df['deposit'] == 'no']['age'], label='No', color='red')
                    
                    # Spécifier la taille de la police
                    plt.title('Distribution des âges selon la variable deposit', fontsize=5)  # Modifiez 10 par la taille souhaitée
                    plt.xlabel('Âge', fontsize=4)  
                    plt.ylabel('Densité', fontsize=4)  
                    plt.legend(fontsize=5) 
                    plt.yticks(fontsize=5)
                    plt.xticks(fontsize=5)
                    st.pyplot(fig)
                    st.write("Test Statistique d'ANOVA :")
                    
                    import statsmodels.api
                    result = statsmodels.formula.api.ols('age ~ deposit', data = df).fit()
                    table = statsmodels.api.stats.anova_lm(result)
                    st.write(table)

                    st.markdown("P_value = 0.0002  ➡️  **Il y a un lien significatif entre Age et Deposit**") 
                    st.write("____________________________________")
    
    
                if sub_pages1 == "Lien balance x deposit" :
                    fig = plt.figure(figsize=(4,2))
                    sns.kdeplot(df[df['deposit'] == 'yes']['balance'], label='Yes', color='blue')
                    sns.kdeplot(df[df['deposit'] == 'no']['balance'], label='No', color='red')
                    plt.title('Distribution de Balance selon la variable deposit', fontsize=5)
                    plt.xlabel('Balance', fontsize=4)  
                    plt.ylabel('Densité', fontsize=4)  
                    plt.legend(fontsize=5) 
                    plt.yticks(fontsize=5)
                    plt.xticks(fontsize=5)
                    st.write(fig)       
    
    
                    st.write("Test d'ANOVA :")
                    st.markdown("P_value = 9.126568e-18  ➡️  **Il y a un lien significatif entre Balance et Deposit**") 
                    st.write("____________________________________")
    
    
                if sub_pages1 == "Lien duration x deposit" :
                    fig = plt.figure(figsize=(4,2))
                    sns.kdeplot(df[df['deposit'] == 'yes']['duration'], label='Yes', color='blue')
                    sns.kdeplot(df[df['deposit'] == 'no']['duration'], label='No', color='red')
                    plt.title('Distribution de Duration selon la variable Deposit', fontsize=5)
                    plt.legend(fontsize=5)  
                    plt.xlabel('Duration', fontsize=4) 
                    plt.ylabel('Densité', fontsize=4)
                    plt.yticks(fontsize=5)
                    plt.xticks(fontsize=5)
                    st.write(fig)
    
                    st.write("Test d'ANOVA :")
    
                    result3 = statsmodels.formula.api.ols('duration ~ deposit', data = df).fit()
                    table3 = statsmodels.api.stats.anova_lm(result3)
                    st.write (table3)
    
                    st.markdown("P_value = 0  ➡️  **Il y a un lien significatif entre Duration et Deposit**") 
                    st.write("____________________________________")
    
    
                if sub_pages1 == "Lien campaign x deposit" :
                    fig = plt.figure(figsize=(4,2))
                    sns.kdeplot(df[df['deposit'] == 'yes']['campaign'], label='Yes', color='blue')
                    sns.kdeplot(df[df['deposit'] == 'no']['campaign'], label='No', color='red')
                    plt.title('Distribution de Campaign selon la variable Deposit', fontsize=5)
                    plt.legend(fontsize=5)  
                    plt.xlabel('Campaign', fontsize=4) 
                    plt.ylabel('Densité', fontsize=4) 
                    plt.yticks(fontsize=5)
                    plt.xticks(fontsize=5)
                    st.write(fig)
    
                    st.write("Test d'ANOVA :")
                    st.markdown("P_value = 4.831324e-42  ➡️  **Il y a un lien significatif entre Campaign et Deposit**") 
                    st.write("____________________________________")
    
    
                if sub_pages1 == "Lien previous x deposit" :
                    fig = plt.figure(figsize=(4,2))
                    sns.kdeplot(df[df['deposit'] == 'yes']['previous'], label='Yes', color='blue')
                    sns.kdeplot(df[df['deposit'] == 'no']['previous'], label='No', color='red')
                    plt.title('Distribution de Previous selon la variable Deposit', fontsize=5)
                    plt.legend(fontsize=5)  # Taille de police pour la légende
                    plt.xlabel('Previous', fontsize=4) 
                    plt.ylabel('Densité', fontsize=4)                     
                    plt.yticks(fontsize=5)
                    plt.xticks(fontsize=5)
                    plt.legend()
                    st.write(fig)
    
                    st.write("Test d'ANOVA :")
                    st.markdown("P_value = 7.125338e-50  ➡️  **Il y a un lien significatif entre Previous et Deposit**") 
                    
                if sub_pages1 == "Conclusion" :
                    st.subheader("Conclusion")
                    st.image("dilenesantos/recap_test_anova.png")
                    st.write("Au regard des p-values (qui sont toutes inférieures à 0.05), on peut conclure que **toutes les variables quantitatives ont un lien significatif avec notre variable cible.**")
                    st.write("____________________________________")


            if submenu_tests == "Tests de Chi-deux" :     
                st.header("Les variables qualitatives sont-elles liées à notre variable cible ?")
                sub_pages2= st.radio(" ", ["Lien job x deposit", "Lien marital x deposit", "Lien education x deposit", "Lien housing x deposit", "Lien poutcome x deposit", "Conclusion"], horizontal = True)
    
                st.write("____________________________________")
    
                st.subheader(f"Analyse du {sub_pages2}")
    
                if sub_pages2 == "Lien job x deposit" :
                    fig = plt.figure(figsize=(20,10))
                    sns.countplot(x="job", hue = 'deposit', data = df, palette =("g", "r"))
                    plt.legend()
                    st.pyplot(fig)
                
    
                    st.write("Test de Chi-deux :")
    
                    from scipy.stats import chi2_contingency
                    ct = pd.crosstab(df['job'], df['deposit'])
                    result = chi2_contingency(ct)
                    stat = result[0]
                    p_value = result[1]
                    st.write('Statistique: ', stat)
                    st.write('P_value: ', p_value)
    
                    st.write("**Il y a une dépendance entre Job et Deposit**") 
                    st.write("____________________________________")
    
    
                if sub_pages2 == "Lien marital x deposit" :
                    fig = plt.figure()
                    sns.countplot(x="marital", hue = 'deposit', data = df, palette =("g", "r"))
                    plt.legend()
                    st.pyplot(fig)
    
    
                    st.write("Test de Chi-deux :")
    
                    from scipy.stats import chi2_contingency
                    ct = pd.crosstab(df['marital'], df['deposit'])
                    result = chi2_contingency(ct)
                    stat = result[0]
                    p_value = result[1]
                    st.write('Statistique: ', stat)
                    st.write('P_value: ', p_value)
    
                    st.write("**Il y a une dépendance entre Marital et Deposit**")  
                    st.write("____________________________________")
                
                
                if sub_pages2 == "Lien education x deposit" :
                    fig = plt.figure()
                    sns.countplot(x="education", hue = 'deposit', data = df, palette =("g", "r"))
                    plt.legend()
                    st.pyplot(fig)
    
    
                    st.write("Test de Chi-deux :")
    
                    from scipy.stats import chi2_contingency
                    ct = pd.crosstab(df['education'], df['deposit'])
                    result = chi2_contingency(ct)
                    stat = result[0]
                    p_value = result[1]
                    st.write('Statistique: ', stat)
                    st.write('P_value: ', p_value)
    
                    st.write("**Il y a une dépendance entre Education et Deposit**")
                    st.write("____________________________________")
    
                
                if sub_pages2 == "Lien housing x deposit" :
                    fig = plt.figure()
                    sns.countplot(x="housing", hue = 'deposit', data = df, palette =("g", "r"))
                    plt.legend()
                    st.pyplot(fig)
    
    
                    st.write("Test de Chi-deux :")
    
                    from scipy.stats import chi2_contingency
                    ct = pd.crosstab(df['housing'], df['deposit'])
                    result = chi2_contingency(ct)
                    stat = result[0]
                    p_value = result[1]
                    st.write('Statistique: ', stat)
                    st.write('P_value: ', p_value)
    
                    st.write("**Il y a une dépendance entre Housing et Deposit**")
                    st.write("____________________________________")
    
                if sub_pages2 == "Lien poutcome x deposit" :
                    fig = plt.figure()
                    sns.countplot(x="poutcome", hue = 'deposit', data = df, palette =("g", "r"))
                    plt.legend()
                    st.pyplot(fig)
    
    
                    st.write("Test Statistique:")
    
                    from scipy.stats import chi2_contingency
                    ct = pd.crosstab(df['poutcome'], df['deposit'])
                    result = chi2_contingency(ct)
                    stat = result[0]
                    p_value = result[1]
                    st.write('Statistique: ', stat)
                    st.write('P_value: ', p_value)
    
                    st.write("**Il y a une dépendance entre Poutcome et Deposit**")  
                    st.write("____________________________________")
    
                if sub_pages2 == "Conclusion" :
    
                    st.subheader("Conclusion")
                    st.image("dilenesantos/recap_Chi-deux.png")
                    st.write("Au regard des p-values (qui sont toutes inférieures à 0.05), on peut conclure que **toutes les variables qualitatives ont un lien significatif avec notre variable cible.**")
                    st.write("____________________________________")


        if st.checkbox("**Évolution dans le temps**"):  
            st.header("Analyse de l'évolution de la variable deposit dans le temps")
            sub_pages3= st.radio(" ", ["Deposit x month", "Deposit x year", "Deposit x weekday"], horizontal = True)

            st.write("____________________________________")

            st.subheader(f"Analyse du {sub_pages3}")
            
            #creation des colonnes year, month_year, date, weekday
            liste_annee =[]
            for i in df["month"] :
                if i == "jun" or i == "jul" or i == "aug" or i == "sep" or i == "oct" or i == "nov" or i == "dec" :
                    liste_annee.append("2013")
                elif i == "jan" or i == "feb" or i == "mar" or i =="apr" or i =="may" :
                    liste_annee.append("2014")
            df["year"] = liste_annee

            df['date'] = df['day'].astype(str)+ '-'+ df['month'].astype(str)+ '-'+ df['year'].astype(str)
            df['date']= pd.to_datetime(df['date'])

            df["weekday"] = df["date"].dt.weekday
            dic = {0 : "Lundi",
            1 : "Mardi",
            2 : "Mercredi",
            3 : "Jeudi",
            4 : "Vendredi",
            5 : "Samedi",
            6 : "Dimanche"}
            df["weekday"] = df["weekday"].replace(dic)


            df['month_year'] = df['month'].astype(str)+ '-'+ df['year'].astype(str)
            df_order_month = df.copy()
            df_order_month = df_order_month.sort_values(by='date')
            df_order_month["month_year"] = df_order_month["month"].astype(str)+ '-'+ df_order_month["year"].astype(str)

            #creation de la colonne Client_Category_M selon pdays
            bins = [-2, -1, 180, 855]
            labels = ['Prospect', 'Reached-6M', 'Reached+6M']
            df['Client_Category_M'] = pd.cut(df['pdays'], bins=bins, labels=labels)
            # Transformation de 'Client_Category' en type 'objet'
            df['Client_Category_M'] = df['Client_Category_M'].astype('object')

            if sub_pages3 == "Deposit x month":
                fig = plt.figure(figsize=(20,10))
                sns.countplot(x='month_year', hue='deposit', data=df_order_month, palette =("g", "r"))
                plt.title("Évolution de notre variable cible selon les mois")
                plt.legend()
                st.pyplot(fig)
                st.write("""Nous pouvons remarquer qu'au début de notre période d'étude la proportion des clients qui
                ont souscrit à un dépôt à terme est inférieur à celle qui n'y ont pas souscrit.""")

            if sub_pages3 == "Deposit x year": 
                fig = plt.figure()
                sns.countplot(x='year', hue='deposit', data=df, palette =("g", "r"))
                plt.title("Évolution de notre variable cible selon l'année")
                plt.legend()
                st.pyplot(fig)
                st.write("""Nous pouvons remarquer ici que la proportion des clients (ayant souscrit ou non à un dépôt à terme)
                est supérieur durant l'année 2013 que 2014. Ceci serait surement dù à la période de l'étude (7 mois en 2013 et 5 mois en 2014) """)


            if sub_pages3 == "Deposit x weekday":
                fig = plt.figure()
                sns.countplot(x="weekday", hue = 'deposit', data = df, palette =("g", "r"))
                plt.title("Évolution de notre variable cible selon les jours de la semaine")
                plt.legend()
                st.pyplot(fig)
                st.write("""Nous remarquons qu'en général les clients souscrivent au dépôt à terme le week-end .""")
        

    if pages == "Profiling" :  
        if st.checkbox("Analyses"):
    
            # Title and Introduction
            st.title("Profil des clients 'Deposit YES'")
            
            # Filter the dataset
            dff = df[df['job'] != "unknown"]  # Remove rows with unknown job
            dff = dff[dff['education'] != "unknown"]  # Remove rows with unknown education
    
            # Replace 'unknown' in poutcome with NaN, then fill with the mode
            dff['poutcome2'] = dff['poutcome'].replace('unknown', np.nan)
            dff['poutcome2'] = dff['poutcome2'].fillna(dff['poutcome2'].mode()[0])
    
            # Drop the 'contact' column as it's not needed
            dff = dff.drop(['contact'], axis=1)
    
            #  Creation de categorie de client
    
            liste =[]
    
            for i in dff["pdays"] :
                if i == -1 :
                    liste.append("new_prospect")
                elif i != -1 :
                    liste.append("old_prospect")
    
            dff["type_prospect"] = liste
    
    
            # Filter clients who have subscribed
            clients_yes = dff[dff["deposit"] == "yes"]
            
    
            # Display the number of subscribed clients
            st.text(f"Nombre de clients ayant souscrit à un compte de dépôt à terme : {clients_yes.shape[0]}")
    
            # Define sub-pages
            sub_pages = st.sidebar.radio(" ", [
                "Age et Job",
                "Statut Matrimonial et Education",
                "Bancaire",
                "Campagnes Marketing",
                "Temporel",
                "Duration"
            ], horizontal = True)
            
            # Sidebar for sub-page selection
    
            # Logic for each sub-page
            if sub_pages == "Age et Job":
                st.write("### Analyse: Age et Job")
                plt.figure(figsize=(10, 6), dpi=120)
                sns.histplot(clients_yes['age'], kde=False, bins=30)
                plt.title("Distribution de l'âge des clients")
                plt.xlabel("Âge des clients")
                plt.ylabel("Nombre de clients")
            
            # Display the plot in Streamlit
                st.pyplot(plt)
    
       # Calcul du nombre de clients par job
                total_client_job = clients_yes.groupby('job').size().reset_index(name='Total Clients')
    
                # Calcul de la moyenne, du minimum, du maximum de la variable 'age' par job
                group_age_job = clients_yes.groupby('job')['age'].agg(['mean', 'min', 'max']).reset_index()
    
                # Renommage des colonnes
                group_age_job.columns = ['job', 'Age Moyen', 'Age Minimum', 'Age Maximum']
    
                # Fusion des deux DataFrames sur la colonne 'job'
                summary = pd.merge(total_client_job, group_age_job, on='job')
    
                # Triage par ordre décroissant du nombre de clients
                summary = summary.sort_values(by='Total Clients', ascending=False)
    
                # Réinitialiser l'index et supprimer la colonne d'index
                summary = summary.reset_index(drop=True)
    
                 # Affichage du DataFrame final dans Streamlit sans la colonne d'index
                st.write("### Résumé des clients par job avec les statistiques d'âge:")
                st.dataframe(summary)
                st.text("Nous remarquons sur ce tableau qu’il y a une grande diversification des âges pour tous les groupes.")
                st.write("____________________________________")
                
            elif sub_pages == "Statut Matrimonial et Education":
                st.write("### Analyse: Statut Matrimonial et Education")
        # --- Statut matrimonial ---
                marital_counts = clients_yes['marital'].value_counts()
                marital_percentage = marital_counts / marital_counts.sum() * 100
                plt.figure(figsize=(10, 8))
                sns.barplot(x=marital_percentage.index, y=marital_percentage.values, color='skyblue')
    
                # Ajouter les pourcentages sur les barres
                for i, v in enumerate(marital_percentage.values):
                    plt.text(i, v + 1, f"{v:.2f}%", ha='center', va='bottom', fontsize=10, color='black')
    
                # Titre et étiquettes des axes
                plt.title("Distribution du statut matrimonial des clients qui ont souscrit à un dépôt à terme")
                plt.xlabel("Statut matrimonial")
                plt.ylabel("Pourcentage de clients (%)")
    
                # Affichage du graphique avec Streamlit
                st.pyplot(plt)
    
                # --- Niveau d'éducation ---
                education_counts = clients_yes['education'].value_counts()
                education_percentage = education_counts / education_counts.sum() * 100
                plt.figure(figsize=(10, 8))
                sns.barplot(x=education_percentage.index, y=education_percentage.values, color='skyblue')
    
                # Ajouter les pourcentages sur les barres
                for i, v in enumerate(education_percentage.values):
                    plt.text(i, v + 1, f"{v:.2f}%", ha='center', va='bottom', fontsize=10, color='black')
    
                # Titre et étiquettes des axes
                plt.title("Distribution du niveau académique des clients qui ont souscrit à un dépôt à terme")
                plt.xlabel("Education")
                plt.ylabel("Pourcentage de clients (%)")
    
                # Affichage du graphique avec Streamlit
                st.pyplot(plt)
    
                # Texte explicatif
                st.text("Nous observons que la majorité des clients sont mariés, suivis par un groupe de clients célibataires. Les niveaux d’éducation des clients sont le secondaire et le tertiaire. Ceci montre que les clients détenant le DAT (dépôt à terme) ont un certain niveau académique.")
                st.write("____________________________________")

            elif sub_pages == "Bancaire":
                st.header("Analyse: Bancaire")
                st.subheader("Balance du compte")
            
                # Séparation des clients en fonction du solde
                clients_positif = clients_yes[clients_yes['balance'] > 0]
                clients_negatif = clients_yes[clients_yes['balance'] <= 0]
                
                nb_clients_positif = len(clients_positif)
                nb_clients_negatif = len(clients_negatif)
    
                pourcentage_positif = (nb_clients_positif / len(clients_yes)) * 100
                pourcentage_negatif = (nb_clients_negatif / len(clients_yes)) * 100
    
                
    
                # Labels pour les groupes
                labels = ['Solde positif', 'Solde négatif ou nul']
                counts = [nb_clients_positif, nb_clients_negatif]
    
                # Créer un DataFrame temporaire pour le plot
                data = pd.DataFrame({'Type de solde': labels, 'Nombre de clients': counts})
                
    
                # Créer un bar plot pour comparer les deux groupes
                plt.figure(figsize=(9, 6), dpi=100)
                sns.barplot(x='Type de solde', y='Nombre de clients', data=data, palette="pastel")
                plt.title("Comparaison des clients avec un solde positif et un solde négatif ou nul")
                plt.xlabel("Type de solde")
                plt.ylabel("Nombre de clients")
    
                # Ajouter les pourcentages sur les barres
                for i, v in enumerate(counts):
                    plt.text(i, v + 5, f"{(v / len(clients_yes)) * 100:.2f}%", ha='center', va='bottom', fontsize=10, color='black')
    
                st.pyplot(plt)
                st.write(f"Pourcentage de clients avec un solde positif : {pourcentage_positif:.2f}%")
                st.write(f"Pourcentage de clients avec un solde négatif ou nul : {pourcentage_negatif:.2f}%")
    
                st.subheader("Loan/Housing/default")
    
                # Statistiques pour 'housing'
                housing_counts = clients_yes['housing'].value_counts()
                housing_percentage = housing_counts / housing_counts.sum() * 100
                housing_stats = pd.DataFrame({
                    'Housing Status': housing_counts.index,
                    'Count': housing_counts.values,
                    'Percentage': housing_percentage.values
                })
    
                # Statistiques pour 'loan'
                loan_counts = clients_yes['loan'].value_counts()
                loan_percentage = loan_counts / loan_counts.sum() * 100
                loan_stats = pd.DataFrame({
                    'Loan Status': loan_counts.index,
                    'Count': loan_counts.values,
                    'Percentage': loan_percentage.values
                })
    
                # Statistiques pour 'default'
                default_counts = clients_yes['default'].value_counts()
                default_percentage = default_counts / default_counts.sum() * 100
                default_stats = pd.DataFrame({
                    'Default Status': default_counts.index,
                    'Count': default_counts.values,
                    'Percentage': default_percentage.values
                })
    
                
    
                # --- Bar plot pour housing ---
                plt.figure(figsize=(9, 6))
                sns.barplot(x=housing_percentage.index, y=housing_percentage.values, palette="pastel")
                plt.title("Distribution des prêts immobiliers parmi les clients ayant un dépôt à terme")
                plt.xlabel("Housing")
                plt.ylabel("Pourcentage de clients (%)")
                
                # Ajouter les pourcentages sur les barres
                for i, v in enumerate(housing_percentage.values):
                    plt.text(i, v + 1, f"{v:.2f}%", ha='center', va='bottom', fontsize=10, color='black')
                
                st.pyplot(plt)
    
                # --- Bar plot pour loan ---
                plt.figure(figsize=(9, 6))
                sns.barplot(x=loan_percentage.index, y=loan_percentage.values, palette="pastel")
                plt.title("Distribution des prêts personnels parmi les clients ayant un dépôt à terme")
                plt.xlabel("Loan")
                plt.ylabel("Pourcentage de clients (%)")
                
                # Ajouter les pourcentages sur les barres
                for i, v in enumerate(loan_percentage.values):
                    plt.text(i, v + 1, f"{v:.2f}%", ha='center', va='bottom', fontsize=10, color='black')
                
                st.pyplot(plt)
    
                # --- Bar plot pour default ---
                plt.figure(figsize=(9, 6))
                sns.barplot(x=default_percentage.index, y=default_percentage.values, palette="pastel")
                plt.title("Distribution de défaut de paiement parmi les clients ayant un dépôt à terme")
                plt.xlabel("Default")
                plt.ylabel("Pourcentage de clients (%)")
                
                # Ajouter les pourcentages sur les barres
                for i, v in enumerate(default_percentage.values):
                    plt.text(i, v + 1, f"{v:.2f}%", ha='center', va='bottom', fontsize=10, color='black')
                
                st.pyplot(plt)
    
                st.text("Parmi les clients qui ont un DAT :")
                st.text("Plus de 60% des clients n’ont pas de prêt immobilier.")
                st.text("90% des clients n’ont pas de prêt personnel.")
                st.text("99% des clients ayant des engagements bancaires ne sont pas en défaut de paiement.")
                st.write("____________________________________")

    
        
            elif sub_pages == "Campagnes Marketing":
                st.write("### Analyse: Caractéristiques des Campagnes marketing")
    
                if st.checkbox("Type de clients"):
                    st.write("Clients Jamais contactés ou déjà contactés lors de la précédente campagne marketing") 
                    # Nombre de clients par type de prospect
                    prospect_counts = clients_yes["type_prospect"].value_counts()
    
                    # Affichage des résultats
                    st.dataframe(prospect_counts)
    
                    # Fonction pour tracer les barres
            
            
            
                    #fonction
                    def plot_percentage(data, column, xlabel):
                        if column not in data.columns:
                            st.error(f"The column '{column}' does not exist in the dataset.")
                            return
            
                    # Calculate percentages
                        counts = data[column].value_counts(normalize=True) * 100
            
                        # Barplot de distribution
                        fig = plt.figure(figsize=(9,6))
                        sns.barplot(x=counts.index, y=counts.values, color='skyblue')
                        plt.title(f"Distribution de {column} (%)", fontsize=5)
                        plt.xlabel(xlabel, fontsize=4)
                        plt.ylabel("Percentage (%)", fontsize=3)
                        plt.xticks(rotation=45, fontsize=4)  
                        st.pyplot(plt)
                        plt.clf()  

                    
                    plot_percentage(clients_yes, "type_prospect", "Type de prospect")
                    st.write("On voit ici que plus de 60% des clients qui ont souscrit au DAT sont de nouveaux prospects.")
                    st.write("____________________________________")

                
                if st.checkbox("Poutcome"):
                    st.write("Résultat de la précédente campagne marketing")  
                    #fonction
                    def plot_percentage(data, column, xlabel):
                        if column not in data.columns:
                            st.error(f"The column '{column}' does not exist in the dataset.")
                            return
            
                    # Calculate percentages
                        counts = data[column].value_counts(normalize=True) * 100
            
                        # Barplot de distribution
                        plt.figure(figsize=(9, 6))
                        sns.barplot(x=counts.index, y=counts.values, color='skyblue')
                        plt.title(f"Distribution de {column} (%)")
                        plt.xlabel(xlabel)
                        plt.ylabel("Percentage (%)")
                        plt.xticks(rotation=45)  
                        st.pyplot(plt)
                        plt.clf()  
                    plot_percentage(clients_yes, "poutcome2", "Poutcome: Résultat de la précédente campagne")
                    st.write("Plus de 70 % des clients précédemment contactés, qui avaient refusé l'offre lors de la campagne précédente, ont accepté de souscrire à cette nouvelle campagne de dépôt à terme.")
                    st.write("____________________________________")

                if st.checkbox("Previous"):
                    #fonction
                    def plot_percentage(data, column, xlabel):
                        if column not in data.columns:
                            st.error(f"The column '{column}' does not exist in the dataset.")
                            return
            
                    # Calculate percentages
                        counts = data[column].value_counts(normalize=True) * 100
            
                        # Barplot de distribution
                        plt.figure(figsize=(9, 6))
                        sns.barplot(x=counts.index, y=counts.values, color='skyblue')
                        plt.title(f"Distribution de {column} (%)")
                        plt.xlabel(xlabel)
                        plt.ylabel("Percentage (%)")
                        plt.xticks(rotation=45)  
                        st.pyplot(plt)
                        plt.clf()  
                    st.write("Nombre de contacts réalisés avec le client avant la campagne")   
                    plot_percentage(clients_yes, "previous", "Nombre de contact réalisé avant la campagne")
                    st.write("Plus de 60% des clients qui ont souscrit au DAT n’avaient jamais été contacté par la banque avant cette campagne.")
    
                if st.checkbox("Campaign"):
                    #fonction
                    def plot_percentage(data, column, xlabel):
                        if column not in data.columns:
                            st.error(f"The column '{column}' does not exist in the dataset.")
                            return
            
                    # Calculate percentages
                        counts = data[column].value_counts(normalize=True) * 100
            
                        # Barplot de distribution
                        plt.figure(figsize=(9, 6))
                        sns.barplot(x=counts.index, y=counts.values, color='skyblue')
                        plt.title(f"Distribution de {column} (%)")
                        plt.xlabel(xlabel)
                        plt.ylabel("Percentage (%)")
                        plt.xticks(rotation=45)  
                        st.pyplot(plt)
                        plt.clf()  
                    st.write("Nombre de contacts réalisés avec le client pendant la campagne") 
                    plot_percentage(clients_yes, "campaign", "Nombre de contact réalisé pendant la campagne")
                    st.write("La plus grande proportion des clients qui ont souscrit au DAT a été contactée une fois pendant cette campagne. Donc en un appel le client a accepté l’offre.")
                    st.write("____________________________________")

            elif sub_pages == "Temporel":
                st.write("### Analyse: Temporel")
    
                liste_annee =[]
                for i in clients_yes["month"] :
                    if i == "jun" or i == "jul" or i == "aug" or i == "sep" or i == "oct" or i == "nov" or i == "dec" :
                        liste_annee.append("2013")
                    elif i == "jan" or i == "feb" or i == "mar" or i =="apr" or i =="may" :
                        liste_annee.append("2014")
                clients_yes["year"] = liste_annee
                clients_yes['date'] = clients_yes['day'].astype(str)+ '-'+ clients_yes['month'].astype(str)+ '-'+ clients_yes['year'].astype(str)
                clients_yes['date']= pd.to_datetime(clients_yes['date'])
                clients_yes["weekday"] = clients_yes["date"].dt.weekday
                dic = {0 : "Lundi", 1 : "Mardi", 2 : "Mercredi", 3 : "Jeudi", 4 : "Vendredi", 5 : "Samedi", 6 : "Dimanche"}
                clients_yes["weekday"] = clients_yes["weekday"].replace(dic)
               
    
                # Mois
                month_year_counts = clients_yes['month'].value_counts()
                month_year_percentage = month_year_counts / month_year_counts.sum() * 100
                plt.figure(figsize=(8, 5))
                sns.barplot(x=month_year_percentage.index, y=month_year_percentage.values, color='skyblue')
                plt.title("Distribution des mois où les clients  ont souscrit à un dépôt à terme")
                plt.xlabel("Mois")
                plt.ylabel("Pourcentage de clients (%)")
                plt.xticks(rotation=90)
                st.pyplot(plt)
    
                #  Jour de la semaine
                weekday_counts = clients_yes['weekday'].value_counts()
                weekday_percentage = weekday_counts / weekday_counts.sum() * 100
                plt.figure(figsize=(8, 5))
                sns.barplot(x=weekday_percentage.index, y=weekday_percentage.values, color='skyblue')
                plt.title("Distribution des jours de la semaine où les clients ont souscrit à un dépôt à terme")
                plt.xlabel("weekday")
                plt.ylabel("Pourcentage de clients (%)")
                st.pyplot(plt)
    
                st.text("Les périodes où les clients sont susceptibles de souscrire sont le printemps et l’été. Et les jours sont par ordre de souscription : dimanche, mardi, mercredi, lundi, jeudi, vendredi et samedi.")
                st.write("____________________________________")

    
            elif sub_pages == "Duration":
                st.write("### Analyse: Duration")
            
                # Conversion de la durée en minutes
                clients_yes = clients_yes.copy()
                clients_yes['duration_minutes'] = clients_yes['duration'] / 60
    
                # Calcul des valeurs de référence
                mean_duration = clients_yes['duration_minutes'].mean()
                min_duration = clients_yes['duration_minutes'].min()
                max_duration = clients_yes['duration_minutes'].max()
    
                # Calcul du pourcentage de clients avec une durée égale ou supérieure au minimum
                nb_clients_min_or_more = len(clients_yes[clients_yes['duration_minutes'] >= min_duration])
                pourcentage_min_or_more = (nb_clients_min_or_more / len(clients_yes)) * 100
    
                # Calcul du pourcentage de clients avec une durée égale ou supérieure au maximum
                nb_clients_max_or_more = len(clients_yes[clients_yes['duration_minutes'] >= max_duration])
                pourcentage_max_or_more = (nb_clients_max_or_more / len(clients_yes)) * 100
    
                # Calcul du pourcentage de clients avec une durée égale ou supérieure à la moyenne
                nb_clients_mean_or_more = len(clients_yes[clients_yes['duration_minutes'] >= mean_duration])
                pourcentage_mean_or_more = (nb_clients_mean_or_more / len(clients_yes)) * 100
    
                # Affichage des résultats sous forme textuelle
                st.write(f"Durée moyenne (minutes) : {mean_duration:.2f}")
                st.write(f"Durée minimum (minutes) : {min_duration:.2f}")
                st.write(f"Durée maximum (minutes) : {max_duration:.2f}")
    
    
    
                # Création d'un DataFrame pour les pourcentages à afficher dans le graphique
                duration_stats = {
                    'Durée': ['Moyenne', 'Minimum', 'Maximum'],
                    'Pourcentage': [pourcentage_mean_or_more, pourcentage_min_or_more, pourcentage_max_or_more]
                }
                duration_df = pd.DataFrame(duration_stats)
    
                # Plot des pourcentages
                plt.figure(figsize=(10, 6))
                sns.barplot(x='Durée', y='Pourcentage', data=duration_df, palette="pastel")
    
                # Ajouter les pourcentages sur les barres
                for i, v in enumerate(duration_df['Pourcentage']):
                    plt.text(i, v + 1, f"{v:.2f}%", ha='center', va='bottom', fontsize=10, color='black')
    
                # Titre et labels
                plt.title("Pourcentage de clients en fonction de la durée d'appel")
                plt.xlabel("Critère de durée")
                plt.ylabel("Pourcentage de clients (%)")
    
                # Affichage du graphique
                st.pyplot(plt)
    
                st.write(f"Pourcentage de clients avec une durée supérieure ou égale à la moyenne : {pourcentage_mean_or_more:.2f}%")
                st.write(f"Pourcentage de clients avec une durée supérieure ou égale au minimum : {pourcentage_min_or_more:.2f}%")
                st.write(f"Pourcentage de clients avec une durée supérieure ou égale au maximum : {pourcentage_max_or_more:.2f}%")
                st.write("____________________________________")



        if st.checkbox("Récapitulatif"):
            st.write("#### Le profil des clients ayant souscrit au produit DAT de la banque est le suivant :")
            st.write("* Clients **âgés entre 25 et 60 ans** avec des métiers de **manager, technicien, ouvrier, ou travaillant dans l’administration.**")
            st.write("* Ils sont **mariés** pour la plupart et ont un niveau **académique secondaire ou tertiaire.**")
            st.write("* La majorité des clients n’ont **pas d’engagement bancaire** (prêt personnel, prêt immobilier) et ne sont **pas en défaut de paiement.**")
            st.write("* Ils n’ont, pour la plupart, **jamais été contacté par la banque.**")
            st.write("* Ils souscrivent au DAT dans les périodes **fin printemps / l’été**, principalement, dans l’ordre, **le dimanche, mardi, mercredi, lundi.**")
            st.write("* Et la durée moyenne des appels pour convaincre un client de souscrire à un DAT est de **9 minutes.**")
        
if selected == "Pre-processing":  
    st.title("PRÉ-PROCESSING")
    option_submenu3 = st.radio(" ", ["**AVANT SÉPARATION DES DONNÉES**", "**APRÈS SÉPARATION DES DONNÉES**"], horizontal = True)
        
        
    if option_submenu3 == '**AVANT SÉPARATION DES DONNÉES**':
        submenupages=st.radio(" ", ["Suppression de lignes", "Création de colonnes", "Suppression de colonnes", "Gestion des Unknowns"], horizontal = True)

        dffpre_pros = df.copy()
        dffpre_pros2 = df.copy()
   
        if submenupages == "Suppression de lignes" :            
            st.subheader("Filtre sur la colonne 'age'")
            st.markdown("Notre analyse univariée a montré des **valeurs extrêmes au dessus de 74 ans.** \n\
            Nous avons décidé de retirer ces lignes de notre dataset.")
            
            dffpre_pros = dffpre_pros[dffpre_pros['age'] < 75]
            count_age_sup = df[df['age'] > 74.5].shape[0]
            st.write("Résultat =", count_age_sup,"**lignes supprimées**")
            
            st.subheader("Filtre sur la colonne 'balance'")
            st.markdown("Pour balance, nous avons également constaté des **valeurs extrêmes** pour **les valeurs inférieures à -2257** et les **valeurs supérieures à 4087**. \n\
            Nous retirons ces lignes.")
            dffpre_pros = dffpre_pros.loc[dffpre_pros["balance"] > -2257]
            dffpre_pros = dffpre_pros.loc[dffpre_pros["balance"] < 4087]
            count_balance_sup = df[df['balance'] < -2257].shape[0]
            count_balance_inf = df[df['balance'] > 4087].shape[0]
            total_balance_count = count_balance_sup + count_balance_inf
            st.write("Résultat =", total_balance_count, "**lignes supprimées**")
            
            st.subheader("Filtre sur la colonne 'campaign'")
            st.markdown("La variable campaign a également montré des **valeurs extrêmes pour les valeurs supérieures à 6**.  \n\
            Nous retirons également ces lignes.")
            dffpre_pros = dffpre_pros.loc[dffpre_pros["campaign"] < 6]
            count_campaign_sup = df[df['campaign'] > 6].shape[0]
            st.write("Résultat", count_campaign_sup,"**lignes supprimées**")
            
            st.subheader("Filtre sur la colonne 'previous'")
            st.markdown("Nous avons également constaté des **valeurs extrêmes pour les valeurs supérieures à 2**. \n\
            Nous retirons également ces lignes de notre dataset.")
            dffpre_pros = dffpre_pros.loc[dffpre_pros["previous"] < 2.5]
            count_previous_sup = df[df['previous'] > 2.5].shape[0]
            st.write("Résultat", count_previous_sup,"**lignes supprimées**")
            
            st.write("____________________________________")

            st.subheader("Résultat:")
            count_sup_lignes = df.shape[0] - dffpre_pros.shape[0]
            st.write("Nombre total de lignes supprimées = ", count_sup_lignes)
            nb_lignes = dffpre_pros.shape[0]
            st.write("**Notre jeu de données filtré compte désormais ", nb_lignes, "lignes.**")

        if submenupages == "Création de colonnes" :   
            st.subheader("Création de la colonne 'Client_Category'")
            st.write("La colonne **'pdays'** indique le nombre de jours depuis le dernier contact avec un client lors de la campagne précédente, mais contient souvent **la valeur -1, signalant des clients jamais contactés**.")
            st.write("Pour distinguer les clients ayant été contactés de ceux qui ne l'ont pas été, une nouvelle colonne **'Client_Category_M'** est créée à partir de 'pdays'.")
            st.markdown("Cette nouvelle colonne nouvellement créée comprend 3 valeurs :  \n\
            1. **Prospect** = clients qui n'ont jamais été contacté lors de la précédente campagne  \n\
            2. **Reached-6M** = clients contactés il y a moins de 6 mois lors de la précédente campagne  \n\
            3. **Reached+6M** = clients contactés il y a plus de 6 mois lors de la précédente campagne")

            
            bins = [-2, -1, 180, 855]
            labels = ['Prospect', 'Reached-6M', 'Reached+6M']
            dffpre_pros['Client_Category_M'] = pd.cut(dffpre_pros['pdays'], bins=bins, labels=labels)
            # Transformation de 'Client_Category' en type 'objet'
            dffpre_pros['Client_Category_M'] = dffpre_pros['Client_Category_M'].astype('object')
                        
            # Affichage du nouveau dataset
            st.dataframe(dffpre_pros.head(10))
            
            st.subheader("Création de la colonne 'weekday'")
            st.markdown("Avant de pouvoir créer la colonne weekday, nous devons passer par deux étapes :  \n\
            1. **ajouter une colonne year** : les données fournies par la banque portugaises sont datées de juin 2014. Nous en déduisons que les mois allant de juin à décembre correspondent à l'année 2023 et que les mois allant de janvier à mai correspondent à l'année 2014  \n\
            2. **ajouter une colonne date au format datetime** : cela est désormais possibles grâce aux colonnes mois, day et year")
            
            st.markdown("**Nous pouvons alors créer la colonne weekday grâce à la fonction 'dt.weekday'**")
            
            #creation des colonnes year, month_year, date, weekday
            liste_annee =[]
            for i in dffpre_pros["month"] :
                if i == "jun" or i == "jul" or i == "aug" or i == "sep" or i == "oct" or i == "nov" or i == "dec" :
                    liste_annee.append("2013")
                elif i == "jan" or i == "feb" or i == "mar" or i =="apr" or i =="may" :
                    liste_annee.append("2014")
            dffpre_pros["year"] = liste_annee
    
            dffpre_pros['date'] = dffpre_pros['day'].astype(str)+ '-'+ dffpre_pros['month'].astype(str)+ '-'+ dffpre_pros['year'].astype(str)
            dffpre_pros['date']= pd.to_datetime(dffpre_pros['date'])
    
            dffpre_pros["weekday"] = dffpre_pros["date"].dt.weekday
            dic = {0 : "Lundi", 1 : "Mardi", 2 : "Mercredi", 3 : "Jeudi", 4 : "Vendredi", 5 : "Samedi", 6 : "Dimanche"}
            dffpre_pros["weekday"] = dffpre_pros["weekday"].replace(dic)
            
            # Affichage du nouveau dataset
            st.dataframe(dffpre_pros.head(10))
            
        
        if submenupages == "Suppression de colonnes" :
            st.subheader("Suppressions de colonnes")
        
            st.write("- La colonne contact ne contribuerait pas de manière significative à la compréhension des données, nous décidons donc de la supprimer.")             
            st.write("- Puisque nous avons créé la colonne Client_Category à partir de la colonne 'pdays', nous supprimons la colonne 'pdays'") 
            st.write("- Puisque nous avons créé la colonne weeday à partir de la colonne 'date', nous supprimons la colonne 'day' ainsi que la colonne date qui nous a uniquement servi à crééer notre colonne weekday.")     
            st.write("- Enfin, nous nous pouvons supprimer la colonne 'year' car les années 2013 et 2014 ne sont pas complètes, nous ne pouvons donc pas les comparer.")

                        
            dffpre_pros2 = dffpre_pros2[dffpre_pros2['age'] < 75]
            dffpre_pros2 = dffpre_pros2.loc[dffpre_pros2["balance"] > -2257]
            dffpre_pros2 = dffpre_pros2.loc[dffpre_pros2["balance"] < 4087]
            dffpre_pros2 = dffpre_pros2.loc[dffpre_pros2["campaign"] < 6]
            dffpre_pros2 = dffpre_pros2.loc[dffpre_pros2["previous"] < 2.5]
            
            bins = [-2, -1, 180, 855]
            labels = ['Prospect', 'Reached-6M', 'Reached+6M']
            dffpre_pros2['Client_Category_M'] = pd.cut(dffpre_pros2['pdays'], bins=bins, labels=labels)
            # Transformation de 'Client_Category' en type 'objet'
            dffpre_pros2['Client_Category_M'] = dffpre_pros2['Client_Category_M'].astype('object')
            
            liste_annee =[]
            for i in dffpre_pros2["month"] :
                if i == "jun" or i == "jul" or i == "aug" or i == "sep" or i == "oct" or i == "nov" or i == "dec" :
                    liste_annee.append("2013")
                elif i == "jan" or i == "feb" or i == "mar" or i =="apr" or i =="may" :
                    liste_annee.append("2014")
            dffpre_pros2["year"] = liste_annee
    
            dffpre_pros2['date'] = dffpre_pros2['day'].astype(str)+ '-'+ dffpre_pros2['month'].astype(str)+ '-'+ dffpre_pros2['year'].astype(str)
            dffpre_pros2['date']= pd.to_datetime(dffpre_pros2['date'])
    
            dffpre_pros2["weekday"] = dffpre_pros2["date"].dt.weekday
            dic = {0 : "Lundi", 1 : "Mardi", 2 : "Mercredi", 3 : "Jeudi", 4 : "Vendredi", 5 : "Samedi", 6 : "Dimanche"}
            dffpre_pros2["weekday"] = dffpre_pros2["weekday"].replace(dic)
            dffpre_pros2 = dffpre_pros2.drop(['contact'], axis=1)
            dffpre_pros2 = dffpre_pros2.drop(['pdays'], axis=1)
            dffpre_pros2 = dffpre_pros2.drop(['day'], axis=1)
            dffpre_pros2 = dffpre_pros2.drop(['date'], axis=1)
            dffpre_pros2 = dffpre_pros2.drop(['year'], axis=1)
            # Transformation des 'unknown' en NaN
            dffpre_pros2['job'] = dffpre_pros2['job'].replace('unknown', np.nan)
            dffpre_pros2['education'] = dffpre_pros2['education'].replace('unknown', np.nan)
            dffpre_pros2['poutcome'] = dffpre_pros2['poutcome'].replace('unknown', np.nan)
            

            st.write("____________________________________")

            st.subheader("Résultat:")
            colonnes_count = dffpre_pros2.shape[1]
            nb_lignes = dffpre_pros2.shape[0]
            st.write("Notre dataset compte désormais :", colonnes_count, "colonnes et", nb_lignes, "lignes.")
            
            # Affichage du nouveau dataset
            st.dataframe(dffpre_pros2.head(5))


        if submenupages == "Gestion des Unknowns" : 
            st.subheader("Les colonnes 'job', 'education' et 'poutcome' contiennent des valeurs 'unknown', il nous faut donc les remplacer.")
            st.write("Pour cela nous allons tout d'abord transformer les valeurs 'unknown' en 'nan'.")
            
            # Transformation des 'unknown' en NaN déjà fait plus haut
                                    
            dffpre_pros2 = dffpre_pros2[dffpre_pros2['age'] < 75]
            dffpre_pros2 = dffpre_pros2.loc[dffpre_pros2["balance"] > -2257]
            dffpre_pros2 = dffpre_pros2.loc[dffpre_pros2["balance"] < 4087]
            dffpre_pros2 = dffpre_pros2.loc[dffpre_pros2["campaign"] < 6]
            dffpre_pros2 = dffpre_pros2.loc[dffpre_pros2["previous"] < 2.5]
            
            bins = [-2, -1, 180, 855]
            labels = ['Prospect', 'Reached-6M', 'Reached+6M']
            dffpre_pros2['Client_Category_M'] = pd.cut(dffpre_pros2['pdays'], bins=bins, labels=labels)
            # Transformation de 'Client_Category' en type 'objet'
            dffpre_pros2['Client_Category_M'] = dffpre_pros2['Client_Category_M'].astype('object')
            
            liste_annee =[]
            for i in dffpre_pros2["month"] :
                if i == "jun" or i == "jul" or i == "aug" or i == "sep" or i == "oct" or i == "nov" or i == "dec" :
                    liste_annee.append("2013")
                elif i == "jan" or i == "feb" or i == "mar" or i =="apr" or i =="may" :
                    liste_annee.append("2014")
            dffpre_pros2["year"] = liste_annee
    
            dffpre_pros2['date'] = dffpre_pros2['day'].astype(str)+ '-'+ dffpre_pros2['month'].astype(str)+ '-'+ dffpre_pros2['year'].astype(str)
            dffpre_pros2['date']= pd.to_datetime(dffpre_pros2['date'])
    
            dffpre_pros2["weekday"] = dffpre_pros2["date"].dt.weekday
            dic = {0 : "Lundi", 1 : "Mardi", 2 : "Mercredi", 3 : "Jeudi", 4 : "Vendredi", 5 : "Samedi", 6 : "Dimanche"}
            dffpre_pros2["weekday"] = dffpre_pros2["weekday"].replace(dic)
            dffpre_pros2 = dffpre_pros2.drop(['contact'], axis=1)
            dffpre_pros2 = dffpre_pros2.drop(['pdays'], axis=1)
            dffpre_pros2 = dffpre_pros2.drop(['day'], axis=1)
            dffpre_pros2 = dffpre_pros2.drop(['date'], axis=1)
            dffpre_pros2 = dffpre_pros2.drop(['year'], axis=1)
            # Transformation des 'unknown' en NaN
            dffpre_pros2['job'] = dffpre_pros2['job'].replace('unknown', np.nan)
            dffpre_pros2['education'] = dffpre_pros2['education'].replace('unknown', np.nan)
            dffpre_pros2['poutcome'] = dffpre_pros2['poutcome'].replace('unknown', np.nan)
            
            st.dataframe(dffpre_pros2.isna().sum())
            
            st.markdown("Nous nous occuperons du remplacement de ces NAns par la suite, une fois le jeu de donnée séparé en jeu d'entraînement et de test.  \n\
            **Cela dans le but de s'assurer que la même transformation des Nans est appliquée au jeu de données Train et Test.**")
            

    if option_submenu3 == '**APRÈS SÉPARATION DES DONNÉES**':
        submenupages2 = st.radio(" ", ["Séparation train test", "Traitement des valeurs manquantes", "Standardisation des variables", "Encodage"], horizontal = True)
         
        if submenupages2 == "Séparation train test" :
            st.subheader("Séparation train test")
            st.write("Nous appliquons un **ratio de 80/20 pour notre train test split** : 80% des données en Train et 20% en Test.")
            dffpre_pros2 = df.copy()                        
            dffpre_pros2 = dffpre_pros2[dffpre_pros2['age'] < 75]
            dffpre_pros2 = dffpre_pros2.loc[dffpre_pros2["balance"] > -2257]
            dffpre_pros2 = dffpre_pros2.loc[dffpre_pros2["balance"] < 4087]
            dffpre_pros2 = dffpre_pros2.loc[dffpre_pros2["campaign"] < 6]
            dffpre_pros2 = dffpre_pros2.loc[dffpre_pros2["previous"] < 2.5]
            
            bins = [-2, -1, 180, 855]
            labels = ['Prospect', 'Reached-6M', 'Reached+6M']
            dffpre_pros2['Client_Category_M'] = pd.cut(dffpre_pros2['pdays'], bins=bins, labels=labels)
            # Transformation de 'Client_Category' en type 'objet'
            dffpre_pros2['Client_Category_M'] = dffpre_pros2['Client_Category_M'].astype('object')
            
            liste_annee =[]
            for i in dffpre_pros2["month"] :
                if i == "jun" or i == "jul" or i == "aug" or i == "sep" or i == "oct" or i == "nov" or i == "dec" :
                    liste_annee.append("2013")
                elif i == "jan" or i == "feb" or i == "mar" or i =="apr" or i =="may" :
                    liste_annee.append("2014")
            dffpre_pros2["year"] = liste_annee
    
            dffpre_pros2['date'] = dffpre_pros2['day'].astype(str)+ '-'+ dffpre_pros2['month'].astype(str)+ '-'+ dffpre_pros2['year'].astype(str)
            dffpre_pros2['date']= pd.to_datetime(dffpre_pros2['date'])
    
            dffpre_pros2["weekday"] = dffpre_pros2["date"].dt.weekday
            dic = {0 : "Lundi", 1 : "Mardi", 2 : "Mercredi", 3 : "Jeudi", 4 : "Vendredi", 5 : "Samedi", 6 : "Dimanche"}
            dffpre_pros2["weekday"] = dffpre_pros2["weekday"].replace(dic)
            dffpre_pros2 = dffpre_pros2.drop(['contact'], axis=1)
            dffpre_pros2 = dffpre_pros2.drop(['pdays'], axis=1)
            dffpre_pros2 = dffpre_pros2.drop(['day'], axis=1)
            dffpre_pros2 = dffpre_pros2.drop(['date'], axis=1)
            dffpre_pros2 = dffpre_pros2.drop(['year'], axis=1)
            # Transformation des 'unknown' en NaN
            dffpre_pros2['job'] = dffpre_pros2['job'].replace('unknown', np.nan)
            dffpre_pros2['education'] = dffpre_pros2['education'].replace('unknown', np.nan)
            dffpre_pros2['poutcome'] = dffpre_pros2['poutcome'].replace('unknown', np.nan)
            
                
            # Séparation des données en un jeu de variables explicatives X et variable cible y
            X_pre_pros2 = dffpre_pros2.drop('deposit', axis = 1)
            y_pre_pros2 = dffpre_pros2['deposit']

            # Séparation des données en un jeu d'entrainement et jeu de test
            X_train_pre_pros2, X_test_pre_pros2, y_train_pre_pros2, y_test_pre_pros2 = train_test_split(X_pre_pros2, y_pre_pros2, test_size = 0.20, random_state = 48)


            colonnes_count = X_train_pre_pros2.shape[1]
            nb_lignes = X_train_pre_pros2.shape[0]
            st.write("Le dataframe X_train compte :", colonnes_count, "colonnes et", nb_lignes, "lignes :")
            st.dataframe(X_train_pre_pros2.head())
                
            colonnes_count = X_test_pre_pros2.shape[1]
            nb_lignes = X_test_pre_pros2.shape[0]
            st.write("Le dataframe X_test compte :", colonnes_count, "colonnes et", nb_lignes, "lignes :")
            st.dataframe(X_test_pre_pros2.head())
                
        if submenupages2 == "Traitement des valeurs manquantes" :    
            st.subheader("Traitement des valeurs manquantes")
            st.write("Pour la **colonne job**, on remplace les Nans par le **mode** de la variable.")
            st.write("S'agissant des **colonnes 'education' et 'poutcome'**, puisque le nombre de Nans est plus élevé, nous avons décidé de les remplacer en utilisant la **méthode de remplissage par propagation** : chaque Nan est remplacé par la valeur de la ligne suivante (pour la dernière ligne on utilise le mode de la variable).") 
            st.write("Ce processus est appliqué à X_train et X_test.")

            dffpre_pros2 = df.copy()                        
            dffpre_pros2 = dffpre_pros2[dffpre_pros2['age'] < 75]
            dffpre_pros2 = dffpre_pros2.loc[dffpre_pros2["balance"] > -2257]
            dffpre_pros2 = dffpre_pros2.loc[dffpre_pros2["balance"] < 4087]
            dffpre_pros2 = dffpre_pros2.loc[dffpre_pros2["campaign"] < 6]
            dffpre_pros2 = dffpre_pros2.loc[dffpre_pros2["previous"] < 2.5]
            
            bins = [-2, -1, 180, 855]
            labels = ['Prospect', 'Reached-6M', 'Reached+6M']
            dffpre_pros2['Client_Category_M'] = pd.cut(dffpre_pros2['pdays'], bins=bins, labels=labels)
            # Transformation de 'Client_Category' en type 'objet'
            dffpre_pros2['Client_Category_M'] = dffpre_pros2['Client_Category_M'].astype('object')
            
            liste_annee =[]
            for i in dffpre_pros2["month"] :
                if i == "jun" or i == "jul" or i == "aug" or i == "sep" or i == "oct" or i == "nov" or i == "dec" :
                    liste_annee.append("2013")
                elif i == "jan" or i == "feb" or i == "mar" or i =="apr" or i =="may" :
                    liste_annee.append("2014")
            dffpre_pros2["year"] = liste_annee
    
            dffpre_pros2['date'] = dffpre_pros2['day'].astype(str)+ '-'+ dffpre_pros2['month'].astype(str)+ '-'+ dffpre_pros2['year'].astype(str)
            dffpre_pros2['date']= pd.to_datetime(dffpre_pros2['date'])
    
            dffpre_pros2["weekday"] = dffpre_pros2["date"].dt.weekday
            dic = {0 : "Lundi", 1 : "Mardi", 2 : "Mercredi", 3 : "Jeudi", 4 : "Vendredi", 5 : "Samedi", 6 : "Dimanche"}
            dffpre_pros2["weekday"] = dffpre_pros2["weekday"].replace(dic)
            dffpre_pros2 = dffpre_pros2.drop(['contact'], axis=1)
            dffpre_pros2 = dffpre_pros2.drop(['pdays'], axis=1)
            dffpre_pros2 = dffpre_pros2.drop(['day'], axis=1)
            dffpre_pros2 = dffpre_pros2.drop(['date'], axis=1)
            dffpre_pros2 = dffpre_pros2.drop(['year'], axis=1)
            # Transformation des 'unknown' en NaN
            dffpre_pros2['job'] = dffpre_pros2['job'].replace('unknown', np.nan)
            dffpre_pros2['education'] = dffpre_pros2['education'].replace('unknown', np.nan)
            dffpre_pros2['poutcome'] = dffpre_pros2['poutcome'].replace('unknown', np.nan)
            
                
            # Séparation des données en un jeu de variables explicatives X et variable cible y
            X_pre_pros2 = dffpre_pros2.drop('deposit', axis = 1)
            y_pre_pros2 = dffpre_pros2['deposit']

            # Séparation des données en un jeu d'entrainement et jeu de test
            X_train_pre_pros2, X_test_pre_pros2, y_train_pre_pros2, y_test_pre_pros2 = train_test_split(X_pre_pros2, y_pre_pros2, test_size = 0.20, random_state = 48)

            # Remplacement des NaNs par le mode:
            imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
            X_train_pre_pros2.loc[:,['job']] = imputer.fit_transform(X_train_pre_pros2[['job']])
            X_test_pre_pros2.loc[:,['job']] = imputer.transform(X_test_pre_pros2[['job']])

            # On remplace les NaaN de 'poutcome' & 'education' avec la méthode de remplissage par propagation et on l'applique au X_train et X_test :
            X_train_pre_pros2['poutcome'] = X_train_pre_pros2['poutcome'].fillna(method ='bfill')
            X_train_pre_pros2['poutcome'] = X_train_pre_pros2['poutcome'].fillna(X_train_pre_pros2['poutcome'].mode()[0])

            X_test_pre_pros2['poutcome'] = X_test_pre_pros2['poutcome'].fillna(method ='bfill')
            X_test_pre_pros2['poutcome'] = X_test_pre_pros2['poutcome'].fillna(X_test_pre_pros2['poutcome'].mode()[0])

            # On fait de même pour les NaaN de 'education'
            X_train_pre_pros2['education'] = X_train_pre_pros2['education'].fillna(method ='bfill')
            X_train_pre_pros2['education'] = X_train_pre_pros2['education'].fillna(X_train_pre_pros2['education'].mode()[0])

            X_test_pre_pros2['education'] = X_test_pre_pros2['education'].fillna(method ='bfill')
            X_test_pre_pros2['education'] = X_test_pre_pros2['education'].fillna(X_test_pre_pros2['education'].mode()[0])

            col1, col2 = st.columns(2)
            with col1 :
             st.write("Vérification sur X_train, reste-t-il des Nans ?")
             st.dataframe(X_train_pre_pros2.isna().sum())
            with col2 :   
             st.write("Vérification sur X_test, reste-t-il des Nans ?")
             st.dataframe(X_test_pre_pros2.isna().sum())

                
        if submenupages2 == "Standardisation des variables" :    
            st.subheader("Standardisation des variables")
            st.write("On **standardise les variables quantitatives** à l'aide de la **fonction StandardScaler**.")
            dffpre_pros2 = df.copy()                        
            dffpre_pros2 = dffpre_pros2[dffpre_pros2['age'] < 75]
            dffpre_pros2 = dffpre_pros2.loc[dffpre_pros2["balance"] > -2257]
            dffpre_pros2 = dffpre_pros2.loc[dffpre_pros2["balance"] < 4087]
            dffpre_pros2 = dffpre_pros2.loc[dffpre_pros2["campaign"] < 6]
            dffpre_pros2 = dffpre_pros2.loc[dffpre_pros2["previous"] < 2.5]
            
            bins = [-2, -1, 180, 855]
            labels = ['Prospect', 'Reached-6M', 'Reached+6M']
            dffpre_pros2['Client_Category_M'] = pd.cut(dffpre_pros2['pdays'], bins=bins, labels=labels)
            # Transformation de 'Client_Category' en type 'objet'
            dffpre_pros2['Client_Category_M'] = dffpre_pros2['Client_Category_M'].astype('object')
            
            liste_annee =[]
            for i in dffpre_pros2["month"] :
                if i == "jun" or i == "jul" or i == "aug" or i == "sep" or i == "oct" or i == "nov" or i == "dec" :
                    liste_annee.append("2013")
                elif i == "jan" or i == "feb" or i == "mar" or i =="apr" or i =="may" :
                    liste_annee.append("2014")
            dffpre_pros2["year"] = liste_annee
    
            dffpre_pros2['date'] = dffpre_pros2['day'].astype(str)+ '-'+ dffpre_pros2['month'].astype(str)+ '-'+ dffpre_pros2['year'].astype(str)
            dffpre_pros2['date']= pd.to_datetime(dffpre_pros2['date'])
    
            dffpre_pros2["weekday"] = dffpre_pros2["date"].dt.weekday
            dic = {0 : "Lundi", 1 : "Mardi", 2 : "Mercredi", 3 : "Jeudi", 4 : "Vendredi", 5 : "Samedi", 6 : "Dimanche"}
            dffpre_pros2["weekday"] = dffpre_pros2["weekday"].replace(dic)
            dffpre_pros2 = dffpre_pros2.drop(['contact'], axis=1)
            dffpre_pros2 = dffpre_pros2.drop(['pdays'], axis=1)
            dffpre_pros2 = dffpre_pros2.drop(['day'], axis=1)
            dffpre_pros2 = dffpre_pros2.drop(['date'], axis=1)
            dffpre_pros2 = dffpre_pros2.drop(['year'], axis=1)
            # Transformation des 'unknown' en NaN
            dffpre_pros2['job'] = dffpre_pros2['job'].replace('unknown', np.nan)
            dffpre_pros2['education'] = dffpre_pros2['education'].replace('unknown', np.nan)
            dffpre_pros2['poutcome'] = dffpre_pros2['poutcome'].replace('unknown', np.nan)
            
                
            # Séparation des données en un jeu de variables explicatives X et variable cible y
            X_pre_pros2 = dffpre_pros2.drop('deposit', axis = 1)
            y_pre_pros2 = dffpre_pros2['deposit']

            # Séparation des données en un jeu d'entrainement et jeu de test
            X_train_pre_pros2, X_test_pre_pros2, y_train_pre_pros2, y_test_pre_pros2 = train_test_split(X_pre_pros2, y_pre_pros2, test_size = 0.20, random_state = 48)

            # Remplacement des NaNs par le mode:
            imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
            X_train_pre_pros2.loc[:,['job']] = imputer.fit_transform(X_train_pre_pros2[['job']])
            X_test_pre_pros2.loc[:,['job']] = imputer.transform(X_test_pre_pros2[['job']])
                
            # On remplace les NaaN de 'poutcome' & 'education' avec la méthode de remplissage par propagation et on l'applique au X_train et X_test :
            X_train_pre_pros2['poutcome'] = X_train_pre_pros2['poutcome'].fillna(method ='bfill')
            X_train_pre_pros2['poutcome'] = X_train_pre_pros2['poutcome'].fillna(X_train_pre_pros2['poutcome'].mode()[0])

            X_test_pre_pros2['poutcome'] = X_test_pre_pros2['poutcome'].fillna(method ='bfill')
            X_test_pre_pros2['poutcome'] = X_test_pre_pros2['poutcome'].fillna(X_test_pre_pros2['poutcome'].mode()[0])

            # On fait de même pour les NaaN de 'education'
            X_train_pre_pros2['education'] = X_train_pre_pros2['education'].fillna(method ='bfill')
            X_train_pre_pros2['education'] = X_train_pre_pros2['education'].fillna(X_train_pre_pros2['education'].mode()[0])

            X_test_pre_pros2['education'] = X_test_pre_pros2['education'].fillna(method ='bfill')
            X_test_pre_pros2['education'] = X_test_pre_pros2['education'].fillna(X_test_pre_pros2['education'].mode()[0])
                
            # Standardisation des variables quantitatives:
            scaler = StandardScaler()
            cols_num = ['age', 'balance', 'duration', 'campaign', 'previous']
            X_train_pre_pros2 [cols_num] = scaler.fit_transform(X_train_pre_pros2 [cols_num])
            X_test_pre_pros2 [cols_num] = scaler.transform (X_test_pre_pros2 [cols_num])
                
            st.write("Vérification de la standardisation des variables quantitatives sur X_train :")
            st.dataframe(X_train_pre_pros2.head())
                
            st.write("Sur X_test :")
            st.dataframe(X_test_pre_pros2.head())

                
        if submenupages2 == "Encodage" :    
            st.subheader("Encodage")
            st.write("On encode la **variable cible** avec le **Label Encoder**.")
            dffpre_pros2 = df.copy()                        
            dffpre_pros2 = dffpre_pros2[dffpre_pros2['age'] < 75]
            dffpre_pros2 = dffpre_pros2.loc[dffpre_pros2["balance"] > -2257]
            dffpre_pros2 = dffpre_pros2.loc[dffpre_pros2["balance"] < 4087]
            dffpre_pros2 = dffpre_pros2.loc[dffpre_pros2["campaign"] < 6]
            dffpre_pros2 = dffpre_pros2.loc[dffpre_pros2["previous"] < 2.5]
            
            bins = [-2, -1, 180, 855]
            labels = ['Prospect', 'Reached-6M', 'Reached+6M']
            dffpre_pros2['Client_Category_M'] = pd.cut(dffpre_pros2['pdays'], bins=bins, labels=labels)
            # Transformation de 'Client_Category' en type 'objet'
            dffpre_pros2['Client_Category_M'] = dffpre_pros2['Client_Category_M'].astype('object')
            
            liste_annee =[]
            for i in dffpre_pros2["month"] :
                if i == "jun" or i == "jul" or i == "aug" or i == "sep" or i == "oct" or i == "nov" or i == "dec" :
                    liste_annee.append("2013")
                elif i == "jan" or i == "feb" or i == "mar" or i =="apr" or i =="may" :
                    liste_annee.append("2014")
            dffpre_pros2["year"] = liste_annee
    
            dffpre_pros2['date'] = dffpre_pros2['day'].astype(str)+ '-'+ dffpre_pros2['month'].astype(str)+ '-'+ dffpre_pros2['year'].astype(str)
            dffpre_pros2['date']= pd.to_datetime(dffpre_pros2['date'])
    
            dffpre_pros2["weekday"] = dffpre_pros2["date"].dt.weekday
            dic = {0 : "Lundi", 1 : "Mardi", 2 : "Mercredi", 3 : "Jeudi", 4 : "Vendredi", 5 : "Samedi", 6 : "Dimanche"}
            dffpre_pros2["weekday"] = dffpre_pros2["weekday"].replace(dic)
            dffpre_pros2 = dffpre_pros2.drop(['contact'], axis=1)
            dffpre_pros2 = dffpre_pros2.drop(['pdays'], axis=1)
            dffpre_pros2 = dffpre_pros2.drop(['day'], axis=1)
            dffpre_pros2 = dffpre_pros2.drop(['date'], axis=1)
            dffpre_pros2 = dffpre_pros2.drop(['year'], axis=1)
            # Transformation des 'unknown' en NaN
            dffpre_pros2['job'] = dffpre_pros2['job'].replace('unknown', np.nan)
            dffpre_pros2['education'] = dffpre_pros2['education'].replace('unknown', np.nan)
            dffpre_pros2['poutcome'] = dffpre_pros2['poutcome'].replace('unknown', np.nan)
            
                
            # Séparation des données en un jeu de variables explicatives X et variable cible y
            X_pre_pros2 = dffpre_pros2.drop('deposit', axis = 1)
            y_pre_pros2 = dffpre_pros2['deposit']

            # Séparation des données en un jeu d'entrainement et jeu de test
            X_train_pre_pros2, X_test_pre_pros2, y_train_pre_pros2, y_test_pre_pros2 = train_test_split(X_pre_pros2, y_pre_pros2, test_size = 0.20, random_state = 48)

            # Remplacement des NaNs par le mode:
            imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
            X_train_pre_pros2.loc[:,['job']] = imputer.fit_transform(X_train_pre_pros2[['job']])
            X_test_pre_pros2.loc[:,['job']] = imputer.transform(X_test_pre_pros2[['job']])
                
            # On remplace les NaaN de 'poutcome' & 'education' avec la méthode de remplissage par propagation et on l'applique au X_train et X_test :
            X_train_pre_pros2['poutcome'] = X_train_pre_pros2['poutcome'].fillna(method ='bfill')
            X_train_pre_pros2['poutcome'] = X_train_pre_pros2['poutcome'].fillna(X_train_pre_pros2['poutcome'].mode()[0])

            X_test_pre_pros2['poutcome'] = X_test_pre_pros2['poutcome'].fillna(method ='bfill')
            X_test_pre_pros2['poutcome'] = X_test_pre_pros2['poutcome'].fillna(X_test_pre_pros2['poutcome'].mode()[0])

            # On fait de même pour les NaaN de 'education'
            X_train_pre_pros2['education'] = X_train_pre_pros2['education'].fillna(method ='bfill')
            X_train_pre_pros2['education'] = X_train_pre_pros2['education'].fillna(X_train_pre_pros2['education'].mode()[0])

            X_test_pre_pros2['education'] = X_test_pre_pros2['education'].fillna(method ='bfill')
            X_test_pre_pros2['education'] = X_test_pre_pros2['education'].fillna(X_test_pre_pros2['education'].mode()[0])

            # Standardisation des variables quantitatives:
            scaler = StandardScaler()
            cols_num = ['age', 'balance', 'duration', 'campaign', 'previous']
            X_train_pre_pros2 [cols_num] = scaler.fit_transform(X_train_pre_pros2 [cols_num])
            X_test_pre_pros2 [cols_num] = scaler.transform (X_test_pre_pros2 [cols_num])

            # Encodage de la variable Cible 'deposit':
            le = LabelEncoder()
            y_train_pre_pros2 = le.fit_transform(y_train_pre_pros2)
            y_test_pre_pros2 = le.transform(y_test_pre_pros2)
                
            st.write("S'agissant des variables qualitatives à 2 modalités **'default'**, **'housing'** et **'loan'**, on encode avec le **One Hot Encoder**.")
            # Encodage des variables explicatives de type 'objet'
            oneh = OneHotEncoder(drop = 'first', sparse_output = False)
            cat1 = ['default', 'housing','loan']
            X_train_pre_pros2.loc[:, cat1] = oneh.fit_transform(X_train_pre_pros2[cat1])
            X_test_pre_pros2.loc[:, cat1] = oneh.transform(X_test_pre_pros2[cat1])

            X_train_pre_pros2[cat1] = X_train_pre_pros2[cat1].astype('int64')
            X_test_pre_pros2[cat1] = X_test_pre_pros2[cat1].astype('int64')
                
            st.write("Pour les variables ordinales **'education'** et **'Client_Category'**, on **remplace les modalités par des nombres** en tenant compte de l'ordre.")
                
            # 'education' est une variable catégorielle ordinale, remplacer les modalités de la variable par des nombres, en gardant l'ordre initial
            X_train_pre_pros2['education'] = X_train_pre_pros2['education'].replace(['primary', 'secondary', 'tertiary'], [0, 1, 2])
            X_test_pre_pros2['education'] = X_test_pre_pros2['education'].replace(['primary', 'secondary', 'tertiary'], [0, 1, 2])

            # 'Client_Category_M' est une variable catégorielle ordinale, remplacer les modalités de la variable par des nombres, en gardant l'ordre initial
            X_train_pre_pros2['Client_Category_M'] = X_train_pre_pros2['Client_Category_M'].replace(['Prospect', 'Reached-6M', 'Reached+6M'], [0, 1, 2])
            X_test_pre_pros2['Client_Category_M'] = X_test_pre_pros2['Client_Category_M'].replace(['Prospect', 'Reached-6M', 'Reached+6M'], [0, 1, 2])


            st.write("Pour les **variables catégorielles à plus de 2 modalités** on applique le **get dummies** sur X_train et X_test.")
                
            # Encoder les variables à plus de 2 modalités 'job', 'marital', 'poutome', 'month', 'weekday' pour X_train
            dummies = pd.get_dummies(X_train_pre_pros2['job'], prefix='job').astype(int)
            X_train_pre_pros2 = pd.concat([X_train_pre_pros2.drop('job', axis=1), dummies], axis=1)
            dummies = pd.get_dummies(X_test_pre_pros2['job'], prefix='job').astype(int)
            X_test_pre_pros2 = pd.concat([X_test_pre_pros2.drop('job', axis=1), dummies], axis=1)

            dummies = pd.get_dummies(X_train_pre_pros2['marital'], prefix='marital').astype(int)
            X_train_pre_pros2 = pd.concat([X_train_pre_pros2.drop('marital', axis=1), dummies], axis=1)
            dummies = pd.get_dummies(X_test_pre_pros2['marital'], prefix='marital').astype(int)
            X_test_pre_pros2 = pd.concat([X_test_pre_pros2.drop('marital', axis=1), dummies], axis=1)

            dummies = pd.get_dummies(X_train_pre_pros2['poutcome'], prefix='poutcome').astype(int)
            X_train_pre_pros2 = pd.concat([X_train_pre_pros2.drop('poutcome', axis=1), dummies], axis=1)
            dummies = pd.get_dummies(X_test_pre_pros2['poutcome'], prefix='poutcome').astype(int)
            X_test_pre_pros2 = pd.concat([X_test_pre_pros2.drop('poutcome', axis=1), dummies], axis=1)

            dummies = pd.get_dummies(X_train_pre_pros2['month'], prefix='month').astype(int)
            X_train_pre_pros2 = pd.concat([X_train_pre_pros2.drop('month', axis=1), dummies], axis=1)
            dummies = pd.get_dummies(X_test_pre_pros2['month'], prefix='month').astype(int)
            X_test_pre_pros2 = pd.concat([X_test_pre_pros2.drop('month', axis=1), dummies], axis=1)

            dummies = pd.get_dummies(X_train_pre_pros2['weekday'], prefix='weekday').astype(int)
            X_train_pre_pros2 = pd.concat([X_train_pre_pros2.drop('weekday', axis=1), dummies], axis=1)
            dummies = pd.get_dummies(X_test_pre_pros2['weekday'], prefix='weekday').astype(int)
            X_test_pre_pros2 = pd.concat([X_test_pre_pros2.drop('weekday', axis=1), dummies], axis=1)


            st.write("Dataframe final X_train : ")
            st.dataframe(X_train_pre_pros2.head())
                
            #Afficher les dimensions des jeux reconstitués.
            st.write("**Dimensions du jeu d'entraînement :**",X_train_pre_pros2.shape)
            st.write("")
                
            st.write("Dataframe final X_test : ")
            st.dataframe(X_test_pre_pros2.head())
            st.write("**Dimensions du jeu de test :**",X_test_pre_pros2.shape)
                

if selected == "Modélisation":
    st.title("MODÉLISATION")
    st.sidebar.title("SOUS MENU MODÉLISATION")  
    pages=["Introduction", "Modélisation avec Duration", "Modélisation sans Duration"]
    page=st.sidebar.radio('Afficher', pages)
 
    
    #RÉSULTAT DES MODÈLES SANS PARAMÈTRES
    # ON A PRÉCÉDEMMENT FAIT TOURNER UN CODE POUR ENREGISTRER LES MODÈLES SANS PARAMÈTRES DANS JOBLIB
    

    #Liste des modèles enregistrés et leurs noms
    model_files = {
        "Random Forest": "dilenesantos/Random_Forest_model_avec_duration_sans_parametres.pkl",
        "Logistic Regression": "dilenesantos/Logistic_Regression_model_avec_duration_sans_parametres.pkl",
        "Decision Tree": "dilenesantos/Decision_Tree_model_avec_duration_sans_parametres.pkl",
        "KNN": "dilenesantos/KNN_model_avec_duration_sans_parametres.pkl",
        "AdaBoost": "dilenesantos/AdaBoost_model_avec_duration_sans_parametres.pkl",
        "Bagging": "dilenesantos/Bagging_model_avec_duration_sans_parametres.pkl",
        "SVM": "dilenesantos/SVM_model_avec_duration_sans_parametres.pkl",
        "XGBOOST": "dilenesantos/XGBOOST_model_avec_duration_sans_parametres.pkl",
    }

        
    # Résultats des modèles
    results_sans_param = {}

    # Boucle pour charger les modèles et calculer les métriques
    for name, file_path in model_files.items():
        # Charger le modèle sauvegardé
        trained_clf = joblib.load(file_path)
        # Faire des prédictions
        y_pred = trained_clf.predict(X_test)

        # Calculer les métriques
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)

        # Stocker les résultats
        results_sans_param[name] = {
            "Accuracy": accuracy,
            "F1 Score": f1,
            "Precision": precision,
            "Recall": recall,
        }

    # Conversion des résultats en DataFrame
    results_sans_param = pd.DataFrame(results_sans_param).T
    results_sans_param.columns = ['Accuracy', 'F1 Score', 'Precision', 'Recall']
    results_sans_param = results_sans_param.sort_values(by="Recall", ascending=False)

    # Graphiques
    results_melted = results_sans_param.reset_index().melt(id_vars="index", var_name="Metric", value_name="Score")
    results_melted.rename(columns={"index": "Classifier"}, inplace=True)

    #HYPERPARAMÈTRES : ÉTAPE 1 - RECHERCHES TEAM POUR LES 3 TOPS MODÈLES (RANDOM FOREST / SVM / XGBOOST)
    # Initialisation des classifiers
    #classifiers_AD_TEAM = {
    #"RF_dounia": RandomForestClassifier(max_depth=None, max_features='log2',min_samples_leaf=2, min_samples_split=2, n_estimators=200, random_state=42),
    #"RF_dilene": RandomForestClassifier(class_weight='balanced', max_depth=25, max_features='sqrt',min_samples_leaf=1, min_samples_split=15, n_estimators=1500, random_state=42),
    #"RF_fatou": RandomForestClassifier(max_depth= None,max_features='log2',min_samples_leaf=2,min_samples_split=2,n_estimators=200,random_state=42),
    #"RF_carolle": RandomForestClassifier(class_weight= 'balanced', max_depth=20, max_features='sqrt',min_samples_leaf=2, min_samples_split=10, n_estimators= 200, random_state=42),
    #"SVM_dounia": svm.SVC(C = 1, class_weight = "balanced", gamma = 'scale', kernel = 'rbf', random_state=42),
    #"SVM_dilene": svm.SVC(C=0.1, class_weight='balanced', gamma=0.1, kernel='rbf', random_state=42),
    #"SVM_fatou": svm.SVC(kernel='rbf',gamma='scale', C=1, random_state=42),
    #"SVM_carolle": svm.SVC(C=0.1, class_weight='balanced', gamma='scale', kernel='rbf', random_state=42),
    #"XGBOOST_dounia": XGBClassifier(colsample_bytree=1.0, learning_rate=0.05,max_depth=7,min_child_weight=1,n_estimators=200,subsample=0.8,random_state=42),
    #"XGBOOST_dilene": XGBClassifier(base_score=0.3, gamma=14, learning_rate=0.6, max_delta_step=1, max_depth=27,min_child_weight=2, n_estimators=900,random_state=42),
    #"XGBOOST_carolle": XGBClassifier(colsample_bytree=0.8, gamma=10, max_depth=17,min_child_weight=1,n_estimators=1000, reg_lambda=0.89, random_state=42),
    #"XGBOOST_fatou": XGBClassifier(colsample_bytree=0.8, gamma= 5, learning_rate= 0.1, max_depth= 5, n_estimators= 100, subsample= 0.8, random_state=42)
    #}

    # Résultats des modèles
    #results_AD_top_3_hyperparam_TEAM = {}

    #Fonction pour entraîner et sauvegarder un modèle
    #def train_and_save_model_team(model_name, clf, X_train, y_train):
        #filename = f"{model_name.replace(' ', '_')}_model_AD_TOP_3_hyperparam_TEAM.pkl"  # Nom du fichier
        #try:
            # Charger le modèle si le fichier existe déjà
            #trained_clf = joblib.load(filename)
        #except FileNotFoundError:
            # Entraîner et sauvegarder le modèle
            #clf.fit(X_train, y_train)
            #joblib.dump(clf, filename)
            #trained_clf = clf
        #return trained_clf

    # Boucle pour entraîner ou charger les modèles
    #for name, clf in classifiers_AD_TEAM.items():
        # Entraîner ou charger le modèle
        #trained_clf = train_and_save_model_team(name, clf, X_train, y_train)
        #y_pred = trained_clf.predict(X_test)
            
        # Calculer les métriques
        #accuracy = accuracy_score(y_test, y_pred)
        #f1 = f1_score(y_test, y_pred)
        #precision = precision_score(y_test, y_pred)
        #recall = recall_score(y_test, y_pred)
            
        # Stocker les résultats
        #results_AD_top_3_hyperparam_TEAM[name] = {
            #"Accuracy": accuracy,
            #"F1 Score": f1,
            #"Precision": precision,
            #"Recall": recall,
        #}
    #COMME ON A ENREGISTRÉ LES MODÈLES, VOICI LE NOUVEAU CODE À UTILISER : 
    # Liste des modèles enregistrés et leurs fichiers correspondants
    model_files_team = {
        "RF_dounia": "dilenesantos/RF_dounia_model_AD_TOP_3_hyperparam_TEAM.pkl",
        "RF_fatou": "dilenesantos/RF_fatou_model_AD_TOP_3_hyperparam_TEAM.pkl",
        "RF_carolle": "dilenesantos/RF_carolle_model_AD_TOP_3_hyperparam_TEAM.pkl",
        "SVM_dounia": "dilenesantos/SVM_dounia_model_AD_TOP_3_hyperparam_TEAM.pkl",
        "SVM_dilene": "dilenesantos/SVM_dilene_model_AD_TOP_3_hyperparam_TEAM.pkl",
        "SVM_fatou": "dilenesantos/SVM_fatou_model_AD_TOP_3_hyperparam_TEAM.pkl",
        "SVM_carolle": "dilenesantos/SVM_carolle_model_AD_TOP_3_hyperparam_TEAM.pkl",
        "XGBOOST_dounia": "dilenesantos/XGBOOST_dounia_model_AD_TOP_3_hyperparam_TEAM.pkl",
        "XGBOOST_dilene": "dilenesantos/XGBOOST_dilene_model_AD_TOP_3_hyperparam_TEAM.pkl",
        "XGBOOST_carolle": "dilenesantos/XGBOOST_carolle_model_AD_TOP_3_hyperparam_TEAM.pkl",
        "XGBOOST_fatou": "dilenesantos/XGBOOST_fatou_model_AD_TOP_3_hyperparam_TEAM.pkl",
    }


    # Résultats des modèles
    results_AD_top_3_hyperparam_TEAM = {}

    # Boucle pour charger les modèles et calculer les métriques
    for name, file_path in model_files_team.items():
        # Charger le modèle sauvegardé
        trained_clf = joblib.load(file_path)
        
        # Faire des prédictions
        y_pred = trained_clf.predict(X_test)

        # Calculer les métriques
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)

        # Stocker les résultats
        results_AD_top_3_hyperparam_TEAM[name] = {
            "Accuracy": accuracy,
            "F1 Score": f1,
            "Precision": precision,
            "Recall": recall,
        }

    # Conversion des résultats en DataFrame
    df_results_AD_top_3_hyperparam_TEAM = pd.DataFrame(results_AD_top_3_hyperparam_TEAM).T
    df_results_AD_top_3_hyperparam_TEAM.columns = ['Accuracy', 'F1 Score', 'Precision', 'Recall']
    df_results_AD_top_3_hyperparam_TEAM = df_results_AD_top_3_hyperparam_TEAM.sort_values(by="Recall", ascending=False)
    
    melted_df_results_AD_top_3_hyperparam_TEAM = df_results_AD_top_3_hyperparam_TEAM.reset_index().melt(id_vars="index", var_name="Metric", value_name="Score")
    melted_df_results_AD_top_3_hyperparam_TEAM.rename(columns={"index": "Classifier"}, inplace=True)    


    #HYPERPARAMÈTRES : ÉTAPE 2 - 
    # Initialisation des classifiers
    #classifiers_grid_2 = {
    #"Random Forest GridSearch2": RandomForestClassifier(class_weight= 'balanced', max_depth = None, max_features = 'sqrt', min_samples_leaf= 2, min_samples_split= 15, n_estimators = 200, random_state=42),
    #"SVM GridSearch2": svm.SVC (C = 1, class_weight = 'balanced', gamma = 'scale', kernel ='rbf', random_state=42),
    #"XGBOOST GridSearch2": XGBClassifier (colsample_bytree = 0.8, gamma = 5, learning_rate = 0.05, max_depth = 17, min_child_weight = 1, n_estimators = 200, subsample = 0.8, random_state=42)
    #}

    # Résultats des modèles
    #results_hyperparam_gridsearch2 = {}

    #Fonction pour entraîner et sauvegarder un modèle
    #def train_and_save_model_team(model_name, clf, X_train, y_train):
        #filename = f"{model_name.replace(' ', '_')}_model_AD_TOP_3_hyperparam_TEAM.pkl"  # Nom du fichier
        #try:
            # Charger le modèle si le fichier existe déjà
            #trained_clf = joblib.load(filename)
        #except FileNotFoundError:
            # Entraîner et sauvegarder le modèle
            #clf.fit(X_train, y_train)
            #joblib.dump(clf, filename)
            #trained_clf = clf
        #return trained_clf

    # Boucle pour entraîner ou charger les modèles
    #for name, clf in classifiers_grid_2.items():
        # Entraîner ou charger le modèle
        #trained_clf = train_and_save_model_team(name, clf, X_train, y_train)
        #y_pred = trained_clf.predict(X_test)
            
        # Calculer les métriques
        #accuracy = accuracy_score(y_test, y_pred)
        #f1 = f1_score(y_test, y_pred)
        #precision = precision_score(y_test, y_pred)
        #recall = recall_score(y_test, y_pred)
            
        # Stocker les résultats
        #results_hyperparam_gridsearch2[name] = {
            #"Accuracy": accuracy,
            #"F1 Score": f1,
            #"Precision": precision,
            #"Recall": recall
        #}
    
    #LES MODÈLES PRÉCÉDENTS ONT ÉTÉ ENREGISTRÉS VIA JOBLIB donc nouveau code pour appeler ces modèles enregistrés
    # Liste des modèles enregistrés et leurs fichiers correspondants
    model_files_grid_2 = {
        "Random Forest GridSearch2": "dilenesantos/Random_Forest_GridSearch2_model_AD_TOP_3_hyperparam_TEAM.pkl",
        "SVM GridSearch2": "dilenesantos/SVM_GridSearch2_model_AD_TOP_3_hyperparam_TEAM.pkl",
        "XGBOOST GridSearch2": "dilenesantos/XGBOOST_GridSearch2_model_AD_TOP_3_hyperparam_TEAM.pkl",
    }

    # Résultats des modèles
    results_hyperparam_gridsearch2 = {}

    # Boucle pour charger les modèles et calculer les métriques
    for name, file_path in model_files_grid_2.items():
        # Charger le modèle sauvegardé
        trained_clf = joblib.load(file_path)
        
        # Faire des prédictions
        y_pred = trained_clf.predict(X_test)

        # Calculer les métriques
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)

        # Stocker les résultats
        results_hyperparam_gridsearch2[name] = {
            "Accuracy": accuracy,
            "F1 Score": f1,
            "Precision": precision,
            "Recall": recall,
        }

    # Conversion des résultats en DataFrame
    df_results_hyperparam_gridsearch2 = pd.DataFrame(results_hyperparam_gridsearch2).T
    df_results_hyperparam_gridsearch2.columns = ['Accuracy', 'F1 Score', 'Precision', 'Recall']
    df_results_hyperparam_gridsearch2 = df_results_hyperparam_gridsearch2.sort_values(by="Recall", ascending=False)
    
    melted_df_results_hyperparam_gridsearch2 = df_results_hyperparam_gridsearch2.reset_index().melt(id_vars="index", var_name="Metric", value_name="Score")
    melted_df_results_hyperparam_gridsearch2.rename(columns={"index": "Classifier"}, inplace=True)    



    if page == pages[0] : 
        st.subheader("Méthodologie")
        st.write("Nous avons effectué **deux modélisations**, l'une en **conservant la variable Duration** et **l'autre sans la variable Duration**: étant donné que cette variable **ne peut être connue qu'après le contact avec le client**.")
        st.write("Pour chaque modélisation, avec ou sans Duration, nous avons analysé les scores des principaux modèles de classification d'abord **sans paramètres** afin de sélectionner les 3 meilleurs modèles, **puis sur ces 3 modèles nous avons effectué des recherches d'hyperparamètres** à l'aide de la **fonction GridSearchCV** afin de sélectionner le modèle **le plus performant possible.**")
        st.write("Enfin sur le meilleur modèle trouvé, nous avons effectué une **analyse SHAP afin d'interpréter les décisions prises par le modèle** dans la détection des clients susceptibles de Deposit YES.")
        st.write("La **métrique principale** choisie est le **Recall pour la classe 1 (deposit = 1)**, afin d'**optimiser la détection des clients intéressés par le DAT en réduisant les faux négatifs**. ")
        st.write("L'objectif de notre modélisation est de **maximiser la performance selon cette métrique.** ")
                 
    if page == pages[1] : 
        #AVEC DURATION
        submenu_modelisation = st.radio("", ("Scores modèles sans paramètres", "Hyperparamètres et choix du modèle"), horizontal=True)
        if submenu_modelisation == "Scores modèles sans paramètres" :
            st.subheader("Scores modèles sans paramètres")
            st.write("Tableau avec les résultats des modèles :")
            st.dataframe(results_sans_param)
                
            st.write("Visualisation des différents scores :")
            # Visualisation des résultats des différents modèles :
            fig = plt.figure(figsize=(12, 6))
            sns.barplot(data=results_melted,x="Classifier",y="Score",hue="Metric",palette="rainbow")
            # Ajouter des titres et légendes
            plt.title("Performance des modèles par métrique", fontsize=16)
            plt.xlabel("Modèles", fontsize=14)
            plt.ylabel("Scores", fontsize=14)
            plt.xticks(rotation=45)
            plt.legend(title="Métrique", fontsize=12)
            plt.legend(loc='lower right')
            plt.tight_layout()
            st.pyplot(fig)
                
    
        if submenu_modelisation == "Hyperparamètres et choix du modèle" :
            st.subheader("Hyperparamètres et choix du modèle")
            st.write("")
            
            st.subheader("Étape 1 : Team GridSearch top 3 modèles")
            st.write("Recherches Gridsearch des 4 membres de la Team sur les top 3 modèles ressortis sans paramètres")
            st.write("Tableau des résultats des modèles hyperparamétrés")
            st.dataframe(df_results_AD_top_3_hyperparam_TEAM)
            
            
            st.subheader("Étape 2 : Nouvelles recherches GridSeach ")
            st.write("Melting de nos différents hyperparamètres trouvés pour chaque modèle pour une nouvelle recherche GridSearch")
            st.write("Tableau des résultats des modèles :")
            st.dataframe(df_results_hyperparam_gridsearch2)
            
            
            st.subheader("Étape 3 : Modèle sélectionné")
            st.write("Le modèle Random Forest 'RF_carolle' avec les hyperparamètres ci-dessous affiche la meilleure performance en termes de Recall, aussi nous choisisons de poursuivre notre modélisation avec ce modèle")
            st.write("RandomForestClassifier(**class_weight= 'balanced', max_depth=20, max_features='sqrt',min_samples_leaf=2, min_samples_split=10, n_estimators= 200, random_state=42**)")
                
            # Chargement du modèle enregistré
            filename = "dilenesantos/RF_carolle_model_AD_TOP_3_hyperparam_TEAM.pkl"
            rf_carolle_model = joblib.load(filename)

            # Prédictions sur les données test
            y_pred = rf_carolle_model.predict(X_test)

            # Calcul des métriques pour chaque classe
            report = classification_report(y_test, y_pred, target_names=["Classe 0", "Classe 1"], output_dict=True)

            # Conversion du rapport en DataFrame pour affichage en tableau
            report_df = pd.DataFrame(report).T

            # Arrondi des valeurs à 4 décimales pour un affichage propre
            report_df = report_df.round(4)

            # Suppression des colonnes inutiles si besoin
            report_df = report_df.drop(columns=["support"])

            # Affichage global du rapport sous forme de tableau
            st.write("**Rapport de classification du modèle:**")
            st.table(report_df)

            # Création de la matrice de confusion sous forme de DataFrame
            st.write("**Matrice de confusion du modèle:**")
            table_rf = pd.crosstab(y_test, y_pred, rownames=["Réalité"], colnames=["Prédiction"])
            st.dataframe(table_rf)

    if page == pages[2] :
        #SANS DURATION
        submenu_modelisation2 = st.radio(" ", ("Scores sans paramètres", "Hyperparamètres et choix du modèle"), horizontal = True)
    
        if submenu_modelisation2 == "Scores sans paramètres" :
            st.subheader("Scores des modèles sans paramètres")
            
            #RÉSULTAT DES MODÈLES SANS PARAMETRES (CODE UTILISÉ UNE FOIS POUR CHARGER LES MODÈLES)
            # Initialisation des classifiers
            #classifiers_SD= {
                #"Random Forest": RandomForestClassifier(random_state=42),
                #"Logistic Regression": LogisticRegression(random_state=42),
                #"Decision Tree": DecisionTreeClassifier(random_state=42),
                #"KNN": KNeighborsClassifier(),
                #"AdaBoost": AdaBoostClassifier(random_state=42),
                #"Bagging": BaggingClassifier(random_state=42),
                #"SVM": svm.SVC(random_state=42),
                #"XGBOOST": XGBClassifier(random_state=42),
            #}

            # Résultats des modèles
            #results_SD_sans_param = {}

            #Fonction pour entraîner et sauvegarder un modèle
            #def train_and_save_model_SD_sans_param(model_name, clf, X_train_sd, y_train_sd):
                #filename = f"{model_name.replace(' ', '_')}_model_sans_duration_sans_parametres.pkl"  # Nom du fichier
                #try:
                    # Charger le modèle si le fichier existe déjà
                    #trained_clf = joblib.load(filename)
                #except FileNotFoundError:
                    # Entraîner et sauvegarder le modèle
                    #clf.fit(X_train_sd, y_train_sd)
                    #joblib.dump(clf, filename)
                    #trained_clf = clf
                #return trained_clf

            # Boucle pour entraîner ou charger les modèles
            #for name, clf in classifiers_SD.items():
                # Entraîner ou charger le modèle
                #trained_clf = train_and_save_model_SD_sans_param(name, clf, X_train_sd, y_train_sd)
                #y_pred = trained_clf.predict(X_test_sd)
                    
                # Calculer les métriques
                #accuracy = accuracy_score(y_test_sd, y_pred)
                #f1 = f1_score(y_test_sd, y_pred)
                #precision = precision_score(y_test_sd, y_pred)
                #recall = recall_score(y_test_sd, y_pred)
                    
                # Stocker les résultats
                #results_SD_sans_param[name] = {
                    #"Accuracy": accuracy,
                    #"F1 Score": f1,
                    #"Precision": precision,
                    #"Recall": recall
                #}

            #CODE À UTILISER PUISQUE MODÈLES SAUVEGARDÉS
            #Chargement des modèles préalablement enregistrés
            models_SD = {
                "Random Forest": joblib.load("dilenesantos/Random_Forest_model_sans_duration_sans_parametres.pkl"),
                "Logistic Regression": joblib.load("dilenesantos/Logistic_Regression_model_sans_duration_sans_parametres.pkl"),
                "Decision Tree": joblib.load("dilenesantos/Decision_Tree_model_sans_duration_sans_parametres.pkl"),
                "KNN": joblib.load("dilenesantos/KNN_model_sans_duration_sans_parametres.pkl"),
                "AdaBoost": joblib.load("dilenesantos/AdaBoost_model_sans_duration_sans_parametres.pkl"),
                "Bagging": joblib.load("dilenesantos/Bagging_model_sans_duration_sans_parametres.pkl"),
                "SVM": joblib.load("dilenesantos/SVM_model_sans_duration_sans_parametres.pkl"),
                "XGBOOST": joblib.load("dilenesantos/XGBOOST_model_sans_duration_sans_parametres.pkl")
            }
            # Charger votre modèle
            filename = "dilenesantos/Random_Forest_model_sans_duration_sans_parametres.pkl"
            model = joblib.load(filename)

            # Sauvegarder le modèle avec compression de niveau 9
            joblib.dump(model, "dilenesantos/Random_Forest_model_sans_duration_sans_parametres.pkl", compress=5)
    
            # Résultats des modèles
            results_SD_sans_param = {}

            # Boucle pour charger les modèles et calculer les résultats
            for name, trained_clf in models_SD.items():
                # Prédictions sur les données test
                y_pred = trained_clf.predict(X_test_sd)

                # Calculer les métriques
                accuracy = accuracy_score(y_test_sd, y_pred)
                f1 = f1_score(y_test_sd, y_pred)
                precision = precision_score(y_test_sd, y_pred)
                recall = recall_score(y_test_sd, y_pred)

                # Stocker les résultats
                results_SD_sans_param[name] = {
                    "Accuracy": accuracy,
                    "F1 Score": f1,
                    "Precision": precision,
                    "Recall": recall
                }

            # Conversion des résultats en DataFrame
            df_results_SD_sans_param = pd.DataFrame(results_SD_sans_param).T
            df_results_SD_sans_param.columns = ['Accuracy', 'F1 Score', 'Precision', 'Recall']
            df_results_SD_sans_param = df_results_SD_sans_param.sort_values(by="Recall", ascending=False)
            
            melted_df_results_SD_sans_param = df_results_SD_sans_param.reset_index().melt(id_vars="index", var_name="Metric", value_name="Score")
            melted_df_results_SD_sans_param.rename(columns={"index": "Classifier"}, inplace=True)

            st.write("La variable **'duration'** a été **retirée** du dataset, les modèles ont été testés sans paramètres et classés selon le score 'Recall' afin de sélectionner les tops modèles pour optimisation ultérieure.")
            st.dataframe(df_results_SD_sans_param)
            
            st.write("Visualiation des résultats :")
            # Visualisation des résultats des différents modèles :
            fig = plt.figure(figsize=(12, 6))
            sns.barplot(data=melted_df_results_SD_sans_param,x="Classifier",y="Score",hue="Metric",palette="rainbow")
            # Ajouter des titres et légendes
            plt.title("Performance des modèles par métrique", fontsize=16)
            plt.xlabel("Modèles", fontsize=14)
            plt.ylabel("Scores", fontsize=14)
            plt.xticks(rotation=45)
            plt.legend(title="Métrique", fontsize=12)
            plt.legend(loc='lower right')
            plt.tight_layout()
            st.pyplot(fig)

            st.markdown("Ces scores nous permettent de sélectionner notre **top 3 des modèles** à tester avec le GridSearchCV :  \n\
            1. Le modèle **Decision Tree**  \n\
            2. Le modèle **XGBOOST**  \n\
            3. Le modèle **Random Forest**")

            st.markdown("Puisque le modèle **SVM** affiche un **meilleur résultat sur le score de Précision**, nous allons également effectuer des tests avec ce modèle, en plus des 3 modèles listés ci-dessus.")

        if submenu_modelisation2 == "Hyperparamètres et choix du modèle" :
            st.write("Scores des modèles hyperparamétrés :")            
            #CODE CHARGÉ UNE FOIS POUR LOAD PUIS RETIRÉ
            # Initialisation des classifiers
            #classifiers_SD_hyperparam= {
                #"Random Forest": RandomForestClassifier(class_weight='balanced', max_depth=8,  max_features='log2', min_samples_leaf=250, min_samples_split=300, n_estimators=400, random_state=42),
                #"Decision Tree": DecisionTreeClassifier(class_weight='balanced', criterion='entropy', max_depth=5,  max_features=None, min_samples_leaf=100, min_samples_split=2, random_state=42),
                #"SVM" : svm.SVC(C=0.01, class_weight='balanced', gamma='scale', kernel='linear',random_state=42),
                #"XGBOOST_1" : XGBClassifier(gamma=0.05,colsample_bytree=0.9, learning_rate=0.39, max_depth=6, min_child_weight=1.29, n_estimators=34, reg_alpha=1.29, reg_lambda=1.9, scale_pos_weight=2.6, subsample=0.99, random_state=42),
                #"XGBOOST_2" : XGBClassifier(gamma=0.05,colsample_bytree=0.88, learning_rate=0.39, max_depth=6, min_child_weight=1.2, n_estimators=30, reg_alpha=1.2, reg_lambda=1.8, scale_pos_weight=2.56, subsample=0.99, random_state=42),
                #"XGBOOST_3" : XGBClassifier(gamma=0.05,colsample_bytree=0.83, learning_rate=0.37, max_depth=6,  min_child_weight=1.2, n_estimators=30, reg_alpha=1.2, reg_lambda=1.7, scale_pos_weight=2.46, subsample=0.99, random_state=42),
                #"XGBOOST_TESTDIL" : XGBClassifier(gamma=0.05,colsample_bytree=0.83, learning_rate=0.37, max_depth=6,  min_child_weight=1.2, n_estimators=30, reg_alpha=1.2, reg_lambda=1.7, scale_pos_weight=2.46, subsample=0.99, random_state=42),
            #}

            # Résultats des modèles
            #results_SD_TOP_4_hyperparam = {}

            #Fonction pour entraîner et sauvegarder un modèle
            #def train_and_save_model_SD_hyperparam(model_name, clf, X_train_sd, y_train_sd):
                #filename = f"{model_name.replace(' ', '_')}_model_SD_TOP_4_hyperparam.pkl"  # Nom du fichier
                #try:
                    # Charger le modèle si le fichier existe déjà
                    #trained_clf = joblib.load(filename)
                #except FileNotFoundError:
                    # Entraîner et sauvegarder le modèle
                    #clf.fit(X_train_sd, y_train_sd)
                    #joblib.dump(clf, filename)
                    #trained_clf = clf
                #return trained_clf

            # Boucle pour entraîner ou charger les modèles
            #for name, clf in classifiers_SD_hyperparam.items():
                # Entraîner ou charger le modèle
                #trained_clf = train_and_save_model_SD_hyperparam(name, clf, X_train_sd, y_train_sd)
                #y_pred = trained_clf.predict(X_test_sd)
                    
                # Calculer les métriques
                #accuracy = accuracy_score(y_test_sd, y_pred)
                #f1 = f1_score(y_test_sd, y_pred)
                #precision = precision_score(y_test_sd, y_pred)
                #recall = recall_score(y_test_sd, y_pred)
                    
                # Stocker les résultats
                #results_SD_TOP_4_hyperparam[name] = {
                    #"Accuracy": accuracy,
                    #"F1 Score": f1,
                    #"Precision": precision,
                    #"Recall": recall
                #}
            
            #Chargement des modèles préalablement enregistrés
            models_SD_hyperparam = {
                "Random Forest": joblib.load("dilenesantos/Random_Forest_model_SD_TOP_4_hyperparam.pkl"),
                "Decision Tree": joblib.load("dilenesantos/Decision_Tree_model_SD_TOP_4_hyperparam.pkl"),
                "SVM": joblib.load("dilenesantos/SVM_model_SD_TOP_4_hyperparam.pkl"),
                "XGBOOST": joblib.load("dilenesantos/XGBOOST_1_model_SD_TOP_4_hyperparam.pkl"),
            }

            # Résultats des modèles
            results_SD_TOP_4_hyperparam = {}

            # Boucle pour charger les modèles et calculer les résultats
            for name, trained_clf in models_SD_hyperparam.items():
                # Prédictions sur les données test
                y_pred = trained_clf.predict(X_test_sd)

                # Calculer les métriques
                accuracy = accuracy_score(y_test_sd, y_pred)
                f1 = f1_score(y_test_sd, y_pred)
                precision = precision_score(y_test_sd, y_pred)
                recall = recall_score(y_test_sd, y_pred)

                # Stocker les résultats
                results_SD_TOP_4_hyperparam[name] = {
                    "Accuracy": accuracy,
                    "F1 Score": f1,
                    "Precision": precision,
                    "Recall": recall
                }
            
            # Conversion des résultats en DataFrame
            df_results_SD_TOP_4_hyperparam = pd.DataFrame(results_SD_TOP_4_hyperparam).T
            df_results_SD_TOP_4_hyperparam.columns = ['Accuracy', 'F1 Score', 'Precision', 'Recall']
            df_results_SD_TOP_4_hyperparam = df_results_SD_TOP_4_hyperparam.sort_values(by="Recall", ascending=False)
            
            melted_df_results_SD_TOP_4_hyperparam = df_results_SD_TOP_4_hyperparam.reset_index().melt(id_vars="index", var_name="Metric", value_name="Score")
            melted_df_results_SD_TOP_4_hyperparam.rename(columns={"index": "Classifier"}, inplace=True)
            
            st.dataframe(df_results_SD_TOP_4_hyperparam)
            
            st.write("Visualisation des résultats:")
            # Visualisation des résultats des différents modèles :
            fig = plt.figure(figsize=(12, 6))
            sns.barplot(data=melted_df_results_SD_TOP_4_hyperparam,x="Classifier",y="Score",hue="Metric",palette="rainbow")
            # Ajouter des titres et légendes
            plt.title("Performance des modèles par métrique", fontsize=16)
            plt.xlabel("Modèles", fontsize=14)
            plt.ylabel("Scores", fontsize=14)
            plt.xticks(rotation=45)
            plt.legend(title="Métrique", fontsize=12)
            plt.legend(loc='lower right')
            plt.tight_layout()
            st.pyplot(fig) 

            st.markdown("Étant donné les scores obtenus sur ces modèles avec hyperparamètres, **nous retenons le modèle XGBOOST** qui affiche un bien meilleur Recall Score sur la classe 1.")
     
                    
            st.subheader("Modèle sélectionné")
            st.write("Voici les hyperparamètres du modèle XGBOOST retenu :")
            st.write("XGBClassifier(**gamma=0.05,colsample_bytree=0.9, learning_rate=0.39, max_depth=6, min_child_weight=1.29, n_estimators=34, reg_alpha=1.29, reg_lambda=1.9, scale_pos_weight=2.6, subsample=0.99, random_state=42**)")
                
            # Chargement du modèle enregistré
            filename = "dilenesantos/XGBOOST_1_model_SD_TOP_4_hyperparam.pkl"
            model_XGBOOST_1_model_SD_TOP_4_hyperparam = joblib.load(filename)

            # Prédictions sur les données test
            y_pred_1 = model_XGBOOST_1_model_SD_TOP_4_hyperparam.predict(X_test_sd)

            # Calcul des métriques pour chaque classe
            report_1 = classification_report(y_test_sd, y_pred_1, target_names=["Classe 0", "Classe 1"], output_dict=True)

            # Conversion du rapport en DataFrame pour affichage en tableau
            report_df_1 = pd.DataFrame(report_1).T

            # Arrondi des valeurs à 4 décimales pour un affichage propre
            report_df_1 = report_df_1.round(4)

            # Suppression des colonnes inutiles si besoin
            report_df_1 = report_df_1.drop(columns=["support"])


            # Création de la matrice de confusion sous forme de DataFrame
            st.write("**Matrice de confusion du modèle :**")
            table_xgboost_1 = pd.crosstab(y_test_sd, y_pred_1, rownames=["Réalité"], colnames=["Prédiction"])
            st.dataframe(table_xgboost_1)

            # Affichage global du rapport sous forme de tableau
            st.write("**Rapport de classification du modèle :**")
            st.table(report_df_1)
            


        
if selected == 'Interprétation':      
    st.sidebar.title("SOUS MENU INTERPRÉTATION")
    pages=["INTERPRÉTATION AVEC DURATION", "INTERPRÉTATION SANS DURATION"]
    page=st.sidebar.radio('AVEC ou SANS Duration', pages)

    if page == pages[0] : 
        st.subheader("Interpréation SHAP avec la colonne Duration")
        #submenu_interpretation = st.selectbox("Menu", ("Summary plot", "Bar plot poids des variables", "Analyses des variables catégorielles", "Dependence plots"))
        submenu_interpretation_Duration = st.radio("", ("ANALYSE GLOBALE", "ANALYSE DES VARIABLES LES PLUS INFLUENTES"), horizontal=True)


        if submenu_interpretation_Duration == "ANALYSE GLOBALE" :
            submenu_globale = st.radio("", ("Summary plot", "Bar plot"), horizontal=True) 

            if submenu_globale == "Summary plot" :
                st.subheader("Summary plot")
                # Affichage des visualisations SHAP
                #SHAP
                #PARTIE DU CODE À VIRER UNE FOIS LES SHAP VALUES CHARGÉES
                #Chargement du modèle XGBOOST_1 déjà enregistré
                #filename_RF_carolle = "RF_carolle_model_AD_TOP_3_hyperparam_TEAM.pkl"
                #model_RF_carolle_model_AD_TOP_3_hyperparam_TEAM = joblib.load(filename_RF_carolle)

                #Chargement des données pour shap 
                #data_to_explain_RF_carolle = X_test  

                #Création de l'explainer SHAP pour XGBOOST_1
                #explainer_RF_carolle = shap.TreeExplainer(model_RF_carolle_model_AD_TOP_3_hyperparam_TEAM)

                #Calcul des shap values
                #shap_values_RF_carolle = explainer_RF_carolle(data_to_explain_RF_carolle)

                #Sauvegarder des shap values avec joblib
                #joblib.dump(shap_values_RF_carolle, "shap_values_RF_carolle_model_AD_TOP_3_hyperparam_TEAM.pkl")

                #CODE À UTILISER UNE FOIS LES SHAP VALUES CHARGÉES
                shap_values_RF_carolle = joblib.load("dilenesantos/shap_values_RF_carolle_model_AD_TOP_3_hyperparam_TEAM.pkl")

                fig = plt.figure()
                shap.summary_plot(shap_values_RF_carolle[:,:,1], X_test)  
                st.pyplot(fig)

            
            if submenu_globale == "Bar plot" :
                st.subheader("Bar plot")  

                #Affichage des barplot sans les moyennes des vaiables à plusieurs items
                #fig = plt.figure()
                #explanation_RF_carolle = shap.Explanation(values=shap_values_RF_carolle,
                                 #data=X_test.values, # Assumant que  X_test est un DataFrame
                                 #feature_names=X_test.columns)
                #shap.plots.bar(explanation_RF_carolle[:,:,1])
                #st.pyplot(fig)
                    

                st.write("Visualisons les Poids des Variables dans le modèle")

                ### 1 CREATION D'UN EXPLANATION FILTRER SANS LES COLONNES POUR LESQUELLES NOUS ALLONS CALCULER LES MOYENNES

                shap_values_rf_carolle_transformed = joblib.load("dilenesantos/shap_values_rf_carolle_transformed.pkl")

                #Étape 1 : Créer une liste des termes à exclure
                terms_to_exclude = ['month', 'weekday', 'job', 'poutcome', 'marital']

                #Étape 2 : Filtrer les colonnes qui ne contiennent pas les termes à exclure
                filtered_columns = [col for col in X_test.columns if not any(term in col for term in terms_to_exclude)]

                #Étape 3 : Identifier les indices correspondants dans X_test_sd
                filtered_indices = [X_test.columns.get_loc(col) for col in filtered_columns]
                shap_values_filtered = shap_values_rf_carolle_transformed[:, filtered_indices]

                # Étape 4 : On créé un nouvel Explanation avec les colonnes filtrées
                explanation_filtered = shap.Explanation(values=shap_values_filtered,
                                            data=X_test.values[:, filtered_indices],  # Garder uniquement les colonnes correspondantes
                                            feature_names=filtered_columns)  # Les noms des features

                ###2 CRÉATION D'UN NOUVEAU EXPLANATION AVEC LES MOYENNES SHAP POUR LES COLONNES MONTH / WEEKDAY / POUTCOME / JOB / MARITAL
                #Fonction pour récupérer les moyennes SHAP en valeur absolue pour les colonnes qui nous intéressent
                def get_mean_shap_values(column_names, shap_values):
                    # Assurez-vous d'accéder aux valeurs à l'intérieur de shap_values
                    indices = [X_test.columns.get_loc(col) for col in column_names]
                    values = shap_values.values[:, indices]  # Utilisez .values pour accéder aux valeurs brutes
                    return np.mean(np.abs(values), axis=0)

                # Étape 1 : On identifie les colonnes que l'on recherche
                month_columns = [col for col in X_test.columns if 'month' in col]
                weekday_columns = [col for col in X_test.columns if 'weekday' in col]
                poutcome_columns = [col for col in X_test.columns if 'poutcome' in col]
                job_columns = [col for col in X_test.columns if 'job' in col]
                marital_columns = [col for col in X_test.columns if 'marital' in col]

                # Étape 2 : On utilise notre fonction pour calculer les moyennes des valeurs SHAP absolues
                mean_shap_month = get_mean_shap_values(month_columns, shap_values_rf_carolle_transformed)
                mean_shap_weekday = get_mean_shap_values(weekday_columns, shap_values_rf_carolle_transformed)
                mean_shap_poutcome = get_mean_shap_values(poutcome_columns, shap_values_rf_carolle_transformed)
                mean_shap_job = get_mean_shap_values(job_columns, shap_values_rf_carolle_transformed)
                mean_shap_marital = get_mean_shap_values(marital_columns, shap_values_rf_carolle_transformed)

                # Étape 3 : On combine les différentes moyennes et on les nomme
                combined_values = [np.mean(mean_shap_month),
                                        np.mean(mean_shap_weekday),
                                        np.mean(mean_shap_poutcome),
                                        np.mean(mean_shap_job),
                                        np.mean(mean_shap_marital)]

                combined_feature_names = ['Mean SHAP Value for Month Features',
                                            'Mean SHAP Value for Weekday Features',
                                            'Mean SHAP Value for Poutcome Features',
                                            'Mean SHAP Value for Job Features',
                                            'Mean SHAP Value for Marital Features']

                # Étape 4 : On crée un nouvel Explanation avec les valeurs combinées
                explanation_combined = shap.Explanation(values=combined_values,
                                                            data=np.array([[np.nan]] * len(combined_values)),
                                                            feature_names=combined_feature_names)

                ###3 ON COMBINE LES 2 EXPLANTATION PRÉCÉDEMMENT CRÉÉS

                #Étape 1 : On récupére les nombre de lignes de explanation_filtered et on reshape explanation_combined pour avoir le même nombre de lignes
                num_samples = explanation_filtered.values.shape[0]
                combined_values_reshaped = np.repeat(np.array(explanation_combined.values)[:, np.newaxis], num_samples, axis=1).T

                #Étape 2: On concatenate les 2 explanations
                combined_values = np.concatenate([explanation_filtered.values, combined_values_reshaped], axis=1)

                #Étape 3: On combine le nom des colonnes provenant des 2 explanations
                combined_feature_names = (explanation_filtered.feature_names + explanation_combined.feature_names)

                #Étape 4: On créé un nouveau explanation avec les valeurs concatnées dans combined_values
                explanation_combined_new = shap.Explanation(values=combined_values,data=np.array([[np.nan]] * combined_values.shape[0]),feature_names=combined_feature_names)

                fig = plt.figure(figsize=(10, 6))
                shap.plots.bar(explanation_combined_new, max_display=len(explanation_combined_new.feature_names))
                st.pyplot(fig)

                st.write("")

        if submenu_interpretation_Duration == "ANALYSE DES VARIABLES LES PLUS INFLUENTES" :
            submenu_var_inf = st.radio("", ("DURATION", "HOUSING", "PREVIOUS"), horizontal=True) 

            if submenu_var_inf == "DURATION" :
                st.write("#### DURATION : Poids de +0.19 dans les prédictions de notre modèle")  
                st.subheader("Impact POSITIF de DURATION sur la classe 1")
                st.write("Summary plot :")

                shap_values_RF_carolle = joblib.load("dilenesantos/shap_values_RF_carolle_model_AD_TOP_3_hyperparam_TEAM.pkl")

                shap_values_RF_CAROLLE_1 = shap_values_RF_carolle[:,:,1]
                fig = plt.figure()
                shap.summary_plot(shap_values_RF_CAROLLE_1[:, [X_test.columns.get_loc("duration")]], 
                                  X_test[["duration"]], 
                                  feature_names=["duration"], 
                                  show=True)
                st.pyplot(fig)
                


                # Dependence plot de DURATION
                st.write("##### Dependence plot")
                shap_CAROLLE_VALUES = shap_values_RF_CAROLLE_1.values
                X_test_original_data = X_test_original
            
                feature_name = "duration"
                #st.write("blabla")
                fig = plt.figure(figsize=(20, 8))
                shap.dependence_plot(feature_name, shap_values=shap_CAROLLE_VALUES, features=X_test_original_data, interaction_index=feature_name, show=False)
                plt.axhline(0, color='red', linestyle='--', linewidth=1) 
                x_ticks = np.arange(0, X_test_original_data[feature_name].max() + 1,360)
                plt.xticks(x_ticks)
                fig = plt.gcf()          
                st.pyplot(fig)       
                plt.close() 
                
            
            if submenu_var_inf == "HOUSING" :
                st.write("#### HOUSING : poids de +0.05 dans les prédictions de notre modèle") 
                st.subheader("Impact NEGATIF de HOUSING sur la classe 1")
                st.write("Summary plot :")

                shap_values_RF_carolle = joblib.load("dilenesantos/shap_values_RF_carolle_model_AD_TOP_3_hyperparam_TEAM.pkl")
                
                shap_values_RF_CAROLLE_1 = shap_values_RF_carolle[:,:,1]
                fig = plt.figure()
                shap.summary_plot(shap_values_RF_CAROLLE_1[:, [X_test.columns.get_loc("housing")]], 
                                  X_test[["housing"]], 
                                  feature_names=["housing"], 
                                  show=True)
                st.pyplot(fig)
                
                
            if submenu_var_inf == "PREVIOUS" :
                st.write("#### PREVIOUS : poids de +0.03 dans les prédictions de notre modèle") 
                st.subheader("Impact POSITIF de PREVIOUS sur la classe 1")
                st.write("Summary plot :")

                shap_values_RF_carolle = joblib.load("dilenesantos/shap_values_RF_carolle_model_AD_TOP_3_hyperparam_TEAM.pkl")

                shap_values_RF_CAROLLE_1 = shap_values_RF_carolle[:,:,1]
                fig = plt.figure()
                shap.summary_plot(shap_values_RF_CAROLLE_1[:, [X_test.columns.get_loc("previous")]], 
                                  X_test[["previous"]], 
                                  feature_names=["preious"], 
                                  show=True)
                st.pyplot(fig)
                

    
            #st.subheader("Dépendences plots & Analyses")
            #st.write("blablabla")



    
    if page == pages[1] : 
        #SHAP
        #PARTIE DU CODE À VIRER UNE FOIS LES SHAP VALUES CHARGÉES
        #Chargement du modèle XGBOOST_1 déjà enregistré
        #filename_XGBOOST_1 = "XGBOOST_1_model_SD_TOP_4_hyperparam.pkl"
        #model_XGBOOST_1_model_SD_TOP_4_hyperparam = joblib.load(filename_XGBOOST_1)
    
        #Chargement des données pour shap 
        #data_to_explain_XGBOOST_1 = X_test_sd  # Remplacez par vos données
    
        #Création de l'explainer SHAP pour XGBOOST_1
        #explainer_XGBOOST_1 = shap.TreeExplainer(model_XGBOOST_1_model_SD_TOP_4_hyperparam)
    
        #Calcul des shap values
        #shap_values_XGBOOST_1 = explainer_XGBOOST_1(data_to_explain_XGBOOST_1)
    
        #Sauvegarder des shap values avec joblib
        #joblib.dump(shap_values_XGBOOST_1, "shap_values_XGBOOST_1_SD_TOP_4_hyperparam.pkl")
    
        #CODE À UTILISER UNE FOIS LES SHAP VALUES CHARGÉES
        shap_values_XGBOOST_1 = joblib.load("dilenesantos/shap_values_XGBOOST_1_SD_TOP_4_hyperparam.pkl")

        st.subheader("Interprétation du modèle XGBOOST")
        #MODÈLE UTILISÉ : XGBOOST_1_model_SD_TOP_4_hyperparam.pkl         

        submenu_interpretation = st.radio("", ("ANALYSE GLOBALE", "ANALYSE DES VARIABLES LES PLUS IMPORTANTES"), horizontal = True)

        if submenu_interpretation == "ANALYSE GLOBALE" :
            submenu_global = st.radio("", ("Summary plot", "Bar plot"), horizontal=True)
            
            if submenu_global == "Summary plot" :
                st.subheader("Summary plot")
                st.write("Le summary plot de SHAP permet d'**évaluer l'impact positif ou négatif de chaque variable sur les prédictions** du modèle.")

                fig = plt.figure()
                shap.summary_plot(shap_values_XGBOOST_1, X_test_sd)  
                st.pyplot(fig)
                
            if submenu_global == "Bar plot" :
                st.subheader("Bar plot")
                st.write("Pour évaluer l'**impact des variables sur les prédictions du modèle**, le bar plot de la librairie SHAP permet d'afficher les moyennes absolues des valeurs SHAP.")
                st.write("Nous avons par ailleurs regroupé certaines variables catégorielles dispatchées en plusieurs colonnes après encodage afin d'avoir une vue d'ensemble de leur effet positif ou négatif sur les prédictions.")

                explanation_XGBOOST_1 = shap.Explanation(values=shap_values_XGBOOST_1,
                                     data=X_test_sd.values, # Assumant que  X_test est un DataFrame
                                     feature_names=X_test_sd.columns)
                shap.plots.bar(explanation_XGBOOST_1)
            
                ### 1 CREATION D'UN EXPLANATION FILTRER SANS LES COLONNES POUR LESQUELLES NOUS ALLONS CALCULER LES MOYENNES
    
                #Étape 1 : Créer une liste des termes à exclure
                terms_to_exclude = ['month', 'weekday', 'job', 'poutcome', 'marital']
    
                #Étape 2 : Filtrer les colonnes qui ne contiennent pas les termes à exclure
                filtered_columns = [col for col in X_test_sd.columns if not any(term in col for term in terms_to_exclude)]
    
                #Étape 3 : Identifier les indices correspondants dans X_test_sd
                filtered_indices = [X_test_sd.columns.get_loc(col) for col in filtered_columns]
                shap_values_filtered_XGBOOST_1 = shap_values_XGBOOST_1[:, filtered_indices]
    
                # Étape 4 : On créé un nouvel Explanation avec les colonnes filtrées
                explanation_filtered_XGBOOST_1 = shap.Explanation(values=shap_values_filtered_XGBOOST_1,
                                                data=X_test_sd.values[:, filtered_indices],  # Garder uniquement les colonnes correspondantes
                                                feature_names=filtered_columns)  # Les noms des features
    
                ###2 CRÉATION D'UN NOUVEAU EXPLANATION AVEC LES MOYENNES SHAP POUR LES COLONNES MONTH / WEEKDAY / POUTCOME / JOB / MARITAL
                #Fonction pour récupérer les moyennes SHAP en valeur absolue pour les colonnes qui nous intéressent
                def get_mean_shap_values(column_names, shap_values):
                    # Assurez-vous d'accéder aux valeurs à l'intérieur de shap_values
                    indices = [X_test_sd.columns.get_loc(col) for col in column_names]
                    values = shap_values.values[:, indices]  # Utilisez .values pour accéder aux valeurs brutes
                    return np.mean(np.abs(values), axis=0)
    
                # Étape 1 : On identifie les colonnes que l'on recherche
                month_columns = [col for col in X_test_sd.columns if 'month' in col]
                weekday_columns = [col for col in X_test_sd.columns if 'weekday' in col]
                poutcome_columns = [col for col in X_test_sd.columns if 'poutcome' in col]
                job_columns = [col for col in X_test_sd.columns if 'job' in col]
                marital_columns = [col for col in X_test_sd.columns if 'marital' in col]
    
                # Étape 2 : On utilise notre fonction pour calculer les moyennes des valeurs SHAP absolues
                mean_shap_month = get_mean_shap_values(month_columns, shap_values_XGBOOST_1)
                mean_shap_weekday = get_mean_shap_values(weekday_columns, shap_values_XGBOOST_1)
                mean_shap_poutcome = get_mean_shap_values(poutcome_columns, shap_values_XGBOOST_1)
                mean_shap_job = get_mean_shap_values(job_columns, shap_values_XGBOOST_1)
                mean_shap_marital = get_mean_shap_values(marital_columns, shap_values_XGBOOST_1)
    
                # Étape 3 : On combine les différentes moyennes et on les nomme
                combined_values_XGBOOST_1 = [np.mean(mean_shap_month),
                                            np.mean(mean_shap_weekday),
                                            np.mean(mean_shap_poutcome),
                                            np.mean(mean_shap_job),
                                            np.mean(mean_shap_marital)]
    
                combined_feature_names_XGBOOST1 = ['Mean SHAP Value for Month Features',
                                                'Mean SHAP Value for Weekday Features',
                                                'Mean SHAP Value for Poutcome Features',
                                                'Mean SHAP Value for Job Features',
                                                'Mean SHAP Value for Marital Features']
    
                # Étape 4 : On crée un nouvel Explanation avec les valeurs combinées
                explanation_combined_XGBOOST_1 = shap.Explanation(values=combined_values_XGBOOST_1,
                                                                data=np.array([[np.nan]] * len(combined_values_XGBOOST_1)),
                                                                feature_names=combined_feature_names_XGBOOST1)
    
                ###3 ON COMBINE LES 2 EXPLANTATION PRÉCÉDEMMENT CRÉÉS
    
                #Étape 1 : On récupére les nombre de lignes de explanation_filtered et on reshape explanation_combined pour avoir le même nombre de lignes
                num_samples = explanation_filtered_XGBOOST_1.values.shape[0]
                combined_values_reshaped__XGBOOST_1 = np.repeat(np.array(explanation_combined_XGBOOST_1.values)[:, np.newaxis], num_samples, axis=1).T
    
                #Étape 2: On concatenate les 2 explanations
                combined_values_XGBOOST_1 = np.concatenate([explanation_filtered_XGBOOST_1.values, combined_values_reshaped__XGBOOST_1], axis=1)
    
                #Étape 3: On combine le nom des colonnes provenant des 2 explanations
                combined_feature_names_XGBOOST_1 = (explanation_filtered_XGBOOST_1.feature_names + explanation_combined_XGBOOST_1.feature_names)
    
                #Étape 4: On créé un nouveau explanation avec les valeurs concatnées dans combined_values
                explanation_combined_new_XGBOOST_1 = shap.Explanation(values=combined_values_XGBOOST_1,data=np.array([[np.nan]] * combined_values_XGBOOST_1.shape[0]),feature_names=combined_feature_names_XGBOOST_1)
    
                fig = plt.figure(figsize=(10, 6))
                shap.plots.bar(explanation_combined_new_XGBOOST_1, max_display=len(explanation_combined_new_XGBOOST_1.feature_names))
                st.pyplot(fig)
                
                st.subheader("Choix des variables les plus importantes")
                st.write("1. **HOUSING** : détention ou non d’un prêt immobilier")
                st.write("2. **BALANCE** : solde bancaire du client")
                st.write("3. **ÂGE**")
                st.write("4. **PREVIOUS** : nombre de contacts effectués avant cette campagne avec le client")
                st.write("5. **CAMPAIGN** : nombre de contacts effectués avec le client pendant la campagne (dernier contact inclus)")
                st.write("6. **EDUCATION** : niveau scolaire du client")                

        if submenu_interpretation == "ANALYSE DES VARIABLES LES PLUS IMPORTANTES" :
            submenu_local = st.radio("", ("HOUSING", "BALANCE", "AGE", "PREVIOUS", "CAMPAIGN", "EDUCATION"), horizontal=True)
            shap_XGBOOST_1_VALUES = shap_values_XGBOOST_1.values
            X_test_original_figures = X_test_sd_original 
            
            if submenu_local == "HOUSING" :
                st.title("HOUSING : POIDS +0.26")
                st.subheader("IMPACT NÉGATIF DE HOUSING SUR LA CLASSE 1")
                st.write("Détenir ou non un prêt immobilier joue un rôle déterminant dans les prédictions de notre modèle.")
                
                fig = plt.figure()
                shap.summary_plot(shap_values_XGBOOST_1[:, [X_test_sd.columns.get_loc("housing")]], 
                                  X_test_sd[["housing"]], 
                                  feature_names=["housing"], 
                                  show=True)
                st.pyplot(fig)
                

                st.write("Les clients avec un prêt immobilier (Housing = 1) ont une probabilité plus faible de souscrire, tandis que **les clients sans prêt (Housing = 0) ont une probabilité plus élevée de souscrire à un dépôt à terme**.")

         
            if submenu_local == "BALANCE" :
                st.title("BALANCE : POIDS +0.24")
                st.subheader("IMPACT POSITIF DE BALANCE SUR LA CLASSE 1")
                st.write("Le solde du client semble être déterminant pour la prédiction. Valeurs comprises entre -1451€ et 4048€")
                fig = plt.figure()
                shap.summary_plot(shap_values_XGBOOST_1[:, [X_test_sd.columns.get_loc("balance")]], 
                                  X_test_sd[["balance"]], 
                                  feature_names=["balance"], 
                                  show=True)
                st.pyplot(fig)

                st.write("On constate ici qu'**un solde bancaire moyen (violet) ou élevé (rouge) augmente la probabilité d'appartenir à la classe 'YES'.**")         

                #GRAPHIQUE DEPENDENCE PLOT
                feature_name = "balance"
                st.write("Le dependence plot ci-dessous présente une distribution en courbe confirmant notre précédent constat : **plus la balance est élevée, plus les valeurs SHAP sont positives.**")
                st.write("On constate cependant qu'au centre de cette courbe, les valeurs de shap sont à la fois positives et négatives.")
                shap.dependence_plot(feature_name, shap_values=shap_XGBOOST_1_VALUES, features=X_test_original_figures, interaction_index=feature_name, show=False)
                plt.axhline(0, color='red', linestyle='--', linewidth=1) 
                xticks = range(-1500, 4300, 300)
                plt.grid(True, which='both', linestyle='--', linewidth=0.5) 
                plt.xticks(xticks, fontsize=5)
                plt.yticks(fontsize=7) 
                plt.xlabel('balance',fontsize=7)  
                plt.ylabel('shap values', fontsize=7)
                fig = plt.gcf()          
                st.pyplot(fig)       
                plt.close() 
                st.write("")
                st.write("Effectuons un zoom pour les balances entre 0 et 1800€ pour une meilleure visibilité")
                shap.dependence_plot(feature_name, shap_values=shap_XGBOOST_1_VALUES, features=X_test_original_figures, interaction_index=feature_name, show=False)
                plt.axhline(0, color='red', linestyle='--', linewidth=1) 
                xticks = range(0, 1800, 100)
                plt.grid(True, which='both', linestyle='--', linewidth=0.5) 
                plt.xticks(xticks, fontsize=5)
                plt.yticks(fontsize=7) 
                plt.xlabel('balance',fontsize=7)  
                plt.ylabel('shap values', fontsize=7)
                plt.xlim(0, 1800)  # Limites de l'axe x
                fig = plt.gcf()          
                st.pyplot(fig)       
                plt.close() 
                

                st.markdown("Ce zoom nous permet de mieux visualiser les choses : \n\
                Les clients dont la balance est comprise entre 0€ et 200 affichent majoritairement des valeurs shap négatives. \n\
                Les clients dont la balance est supérieure à 800€ sont affichent majoritairement des valeurs shap positives. \n\
                Les clients affichant un solde entre 200 et 800€ sont globalement scindés en 2 groupes : une moitié de ces clients ne souscrit pas au produit, mais l’autre moitié oui.")
                
                st.subheader("Recherche d'autres dépendances")
                st.write("Pour tenter de départager ces clients dont la balance est comprise entre 200 et 800€, examinons les relations avec d'autres variables pour voir si nous pouvons identifier des tendances.")
                # Extraction des valeurs SHAP
                shap_values = shap_XGBOOST_1_VALUES
                X_data = X_test_original_figures  # Remplacez-le par vos données d'entrée réelle
                
                # Liste des variables pour interaction_index
                interaction_variables = ["housing", "age", "education", "marital status", "job"]
                
                # radio
                selected_variable = st.radio("Choix de la variable", interaction_variables, horizontal=True)
                
                # Vérification si la variable sélectionnée est "housing", "age" ou "education"
                if selected_variable in ["housing", "age", "education"]:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    shap.dependence_plot("balance", shap_XGBOOST_1_VALUES, X_test_original_figures, 
                                         interaction_index=selected_variable, show=False, ax=ax)
                  
                    # Titre et axe horizontal rouge
                    ax.axhline(0, color='red', linewidth=1, linestyle='--')
                    xticks = range(200, 800, 100)
                    plt.xlim(200, 800) 

                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()
                    st.write("Il n'est pas possible d'établir un lien clair entre la balance des clients et ces autres variables numériques.")
                
                elif selected_variable == "marital status":
                    # Variables associées à marital status
                    marital_variables = ["marital_married", "marital_single", "marital_divorced"]
                    
                    # Créer un graphique pour chaque variable associée à marital status
                    fig, axes = plt.subplots(3, 1, figsize=(10, 18))
                
                    for i, variable in enumerate(marital_variables):
                        shap.dependence_plot(
                            "balance", shap_XGBOOST_1_VALUES, X_test_original_figures, 
                            interaction_index=variable, show=False, ax=axes[i]
                        )
                        axes[i].set_title(f'Balance x {variable}', fontsize=14)
                        axes[i].axhline(0, color='red', linewidth=1, linestyle='--')
                        #focus valeurs entre 200 et 800 
                        axes[i].set_xlim(200, 800)  # Définir les limites de l'axe x
                        xticks = range(200, 801, 100)  # Créer une gamme de ticks
                        axes[i].set_xticks(xticks)
       
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()
                    st.write("Il n'est pas non plus possible d'établir un lien clair entre la balance des clients et leur statut marital.")

                
                elif selected_variable == "job":
                    # Variables associées à job
                    job_variables = ['job_admin.', 'job_blue-collar', 'job_entrepreneur', 'job_housemaid', 'job_management', 
                                     'job_retired', 'job_self-employed', 'job_services', 'job_student', 'job_technician', 'job_unemployed']
                
                    # Créer un graphique pour chaque variable associée à job
                    fig, axes = plt.subplots(len(job_variables), 1, figsize=(10, len(job_variables) * 6))
                
                    for i, variable in enumerate(job_variables):
                        shap.dependence_plot(
                            "balance", shap_XGBOOST_1_VALUES, X_test_original_figures, 
                            interaction_index=variable, show=False, ax=axes[i]
                        )
                        axes[i].set_title(f'Balance x {variable}', fontsize=14)
                        axes[i].axhline(0, color='red', linewidth=1, linestyle='--')
                        #focus valeurs entre 200 et 800 
                        axes[i].set_xlim(200, 800)  # Définir les limites de l'axe x
                        xticks = range(200, 801, 100)  # Créer une gamme de ticks
                        axes[i].set_xticks(xticks) 
                     
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()
                    st.write("Il en est de même pour la variable job.")


            if submenu_local == "AGE" :
                st.title("ÂGE : POIDS +0.23")
                st.subheader("IMPACT POSITIF CHEZ LES JEUNES ET LES PLUS ÂGÉS")
                st.subheader("IMPACT NÉGATIF DES TRANCHES D’ÂGES MOYENNES")
                st.write("L’âge joue un rôle significatif dans l’orientation des prédictions. Valeurs comprises entre 18 et 74 ans.")
                fig = plt.figure()
                shap.summary_plot(shap_values_XGBOOST_1[:, [X_test_sd.columns.get_loc("age")]], 
                                  X_test_sd[["age"]], 
                                  feature_names=["age"], 
                                  show=True)
                st.pyplot(fig)
                st.write("Ce summary plot montre assez clairement qu'une **majorité de 'violet' soit des âges intermédiaires présentent des shap values négatives, ils ont donc tendance à ne pas souscrire au dépôt à terme.**")         

                st.write("Pour une meilleure représentation de la distribution de la variable âge, affichons son dépendance plot.")
                feature_name = "age"
                fig, ax = plt.subplots(figsize=(20, 7))
                shap.dependence_plot(feature_name, shap_values=shap_XGBOOST_1_VALUES, features=X_test_original_figures, interaction_index=feature_name, show=False)
                plt.axhline(0, color='red', linestyle='--', linewidth=1) 
                fig = plt.gcf()  
                ax.set_xlim(17, 76)
                ax.set_xticks(np.arange(17, 77, 1))
                ax.axhline(0, color='red', linewidth=1.5, linestyle='--')
                st.pyplot(fig)       
                plt.close()
     
                st.write("Le graphique en U confirme que **les souscriptions au dépôt à terme concernent principalement des clients jeunes (18-28 ans) ou les clients plus âgés (59 ans et plus)**, tandis que les clients d'âge intermédiaire (29-59 ans) sont majoritairement associés à des valeurs SHAP négatives, indiquant une tendance à ne pas souscrire.") 


            if submenu_local == "PREVIOUS" :
                st.title("PREVIOUS : POIDS +0.22")
                st.subheader("IMPACT POSITIF DE PREVIOUS SUR LA CLASSE 1")
                st.write("Le nombre de contacts effectués avant la campagne avec le client semble également jouer un rôle important dans la prédiction. Valeurs comprises entre 0 et 2 fois.")
                fig = plt.figure()
                shap.summary_plot(shap_values_XGBOOST_1[:, [X_test_sd.columns.get_loc("previous")]], 
                                  X_test_sd[["previous"]], 
                                  feature_names=["previous"], 
                                  show=True)
                st.pyplot(fig)
                
                st.markdown("**Les clients ayant eu des interactions avec la banque par le passé** ont une **probabilité plus élevée d'appartenir à la classe 'YES'.**")
                
                feature_name = "previous"
                
                shap.dependence_plot(feature_name, shap_values=shap_XGBOOST_1_VALUES, features=X_test_original_figures, interaction_index=feature_name, show=False)
                plt.axhline(0, color='red', linestyle='--', linewidth=1) 
                fig = plt.gcf()          
                st.pyplot(fig)       
                plt.close() 
                st.write("La distribution des valeurs de previous montre très clairement que lorsque les clients n’ont jamais été contactés (previous = 0) alors la shap value est négative, tandis que **les clients qui ont été contactés par le passé affichent des valeurs shap très nettement positives, ils sont donc plus susceptibles de souscrire au produit.**")
                        
            if submenu_local == "CAMPAIGN" :
                st.title("PREVIOUS : POIDS +0.10")
                st.subheader("IMPACT POSITIF DE PREVIOUS SUR LA CLASSE 1")
                st.write("Le nombre de contacts effectués avec le client pendant la campagne (dernier contact inclus) est également un paramètre relativement important dans la prédiction de notre modèle. Valeurs comprises entre 1 et 5.")
                fig = plt.figure()
                shap.summary_plot(shap_values_XGBOOST_1[:, [X_test_sd.columns.get_loc("campaign")]], 
                                  X_test_sd[["campaign"]], 
                                  feature_names=["campaign"], 
                                  show=True)
                st.pyplot(fig)

                st.write("Bien que cette variable ait un impact relativement faible, elle reste positive dans notre modèle. Il semble que plus le nombre de contacts avec le client pendant la campagne est élevé (points violets et rouges), plus cela a un effet négatif sur la prédiction : un nombre élevé d’appels semble entraîner un échec à convaincre le client à souscrire au produit.")
                
                feature_name = "campaign"
                
                shap.dependence_plot(feature_name, shap_values=shap_XGBOOST_1_VALUES, features=X_test_original_figures, interaction_index=feature_name, show=False)
                plt.axhline(0, color='red', linestyle='--', linewidth=1) 
                fig = plt.gcf()          
                st.pyplot(fig)       
                plt.close() 
                st.write("Ce graphique montre clairement que **les clients qui n’ont été contactés qu’une seule fois affichent très majoritairement des SHAP values positives.**")

            if submenu_local == "EDUCATION" :
                st.title("EDUCATION : POIDS +0.09")
                st.subheader("IMPACT POSITIF DE ÉDUCATION SUR LA CLASSE 1")
                st.write("Il est clair que les clients ayant un niveau d'éducation élévé ont davantage tendance à souscrire au dépôt à terme.")
                fig = plt.figure()
                shap.summary_plot(shap_values_XGBOOST_1[:, [X_test_sd.columns.get_loc("education")]], 
                                  X_test_sd[["education"]], 
                                  feature_names=["education"], 
                                  show=True)
                st.pyplot(fig)
                st.write("Cela est confirmé par le dependence plot.")         

                feature_name = "education"
                
                shap.dependence_plot(feature_name, shap_values=shap_XGBOOST_1_VALUES, features=X_test_original_figures, interaction_index=feature_name, show=False)
                plt.axhline(0, color='red', linestyle='--', linewidth=1) 
                fig = plt.gcf()          
                st.pyplot(fig)       
                plt.close() 

                st.subheader("Recherche d'autres dépendances")
                # Extraction des valeurs SHAP
                shap_values = shap_XGBOOST_1_VALUES
                X_data = X_test_original_figures  # Remplacez-le par vos données d'entrée réelle
                
                # Liste des variables pour interaction_index
                interaction_variables = ["age", "job"]
                
                # radio
                selected_variable = st.radio("Choix de la variable", interaction_variables, horizontal=True)
                
                # Vérification si la variable sélectionnée est "housing", "age" ou "education"
                if selected_variable in ["age"]:
                    st.write("Côté âges, pas de dépendance claire avec le niveau d'éducation.")
                    fig, ax = plt.subplots(figsize=(10, 6))
                    shap.dependence_plot("education", shap_XGBOOST_1_VALUES, X_test_original_figures, 
                                         interaction_index=selected_variable, show=False, ax=ax)
                
                    # Titre et axe horizontal rouge
                    ax.axhline(0, color='red', linewidth=1, linestyle='--')
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()
                
                elif selected_variable == "job":
                    # Variables associées à job
                    job_variables = ['job_admin.', 'job_blue-collar', 'job_entrepreneur', 'job_housemaid', 'job_management', 
                                     'job_retired', 'job_self-employed', 'job_services', 'job_student', 'job_technician', 'job_unemployed']
                
                    # Créer un graphique pour chaque variable associée à job
                    st.write("Les jobs ont l'on peut observer une vraie tendance ne concernent que les **Blue-collar** (plutôt d'un niveau primaire d'éducation) et **emplois dans le Management** (niveau d'éducation plutôt tertiaire)")

                    fig, axes = plt.subplots(len(job_variables), 1, figsize=(10, len(job_variables) * 6))
                
                    for i, variable in enumerate(job_variables):
                        shap.dependence_plot(
                            "education", shap_XGBOOST_1_VALUES, X_test_original_figures, 
                            interaction_index=variable, show=False, ax=axes[i]
                        )
                        axes[i].set_title(f'Balance x {variable}', fontsize=14)
                        axes[i].axhline(0, color='red', linewidth=1, linestyle='--')
                
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()


if selected == "Recommandations & Perspectives":
      st.subheader("Recommandations & Perspectives")
      submenu_reco = st.radio("", ("PROFIL DES CLIENTS A CONTACTER", "NOMBRE ET DUREE D’APPEL", "RECAP"), horizontal=True)

      if submenu_reco == "PROFIL DES CLIENTS A CONTACTER" :
            submenu_profil = st.radio("", ("HOUSING", "ÂGE", "BALANCE", "PREVIOUS", "EDUCATION"), horizontal=True) 

            if submenu_profil == "HOUSING" :
                st.write("#### HOUSING:","Détention ou non d’un prêt immobilier")
                st.write("##### Prioriser les clients qui n’ont pas de prêt immobilier")
                st.write("Le modèle montre que les **clients ayant un prêt immobilier ont une probabilité plus faible de souscrire au DAT**.")
                st.write("Il faudrait analyser plus en profondeur le contexte métier pour comprendre pourquoi la possession d'un prêt immobilier est négativement corrélée à la classe 1. Par exemple :")
                st.write(" - La classe 1 de ‘housing’ pourrait représenter des clients à risque de défaut de paiement. Dans ce cas, les clients ayant déjà un prêt pourraient être considérés comme plus à risque, car ils ont déjà une dette.")
                st.write(" - La classe 1 de ‘housing’ pourrait représenter des clients ciblés pour une nouvelle offre de prêt. Dans ce cas, les clients ayant déjà un prêt seraient moins susceptibles d'être sollicités pour un nouveau crédit.")
                

            if submenu_profil == "ÂGE" :
                st.write("#### ÂGE:")
                st.write("##### Prioriser les clients âgés de 18 à 28 ans et de 59 ans ou plus.")
                st.write("S’agissant de la tranche d’âge intermédiaire, **entre 29 et 58 ans**, la tendance est inversée, notre modèle montre **une prédiction majoritairement négative**. Étant donné qu’une grande majorité des clients de la base de donnée concerne justement cette tranche d’âge intermédiaire, il faudra veiller à bien suivre nos recommandations concernant les autres variables pour sélectionner, parmi ces clients âgés de 29 à 58 ans, ceux à contacter en priorité : ")
                st.write(" -  Ils n’ont pas de prêt immobilier")
                st.write(" -  Ils ont une balance supérieure à 200€ et idéalement supérieure à 800€")
            
            if submenu_profil == "BALANCE" :
                st.write("#### BALANCE:","Solde bancaire du client")
                st.write("##### Contacter en priorité les clients dont la balance est supérieure à 800€.")
                st.write("Pour les clients dont la balance se situe **entre 0 et 800€ le modèle prédit 2 groupes de taille quasi identiques**, une partie souscrit au produit et l’autre non. Ainsi pour ces clients en particulier, il faudra là encore veiller à suivre nos recommandations sur les autres variables pour choisir ceux à contacter en priorité : ")
                st.write(" -  Ils n’ont pas de prêt immobilier")
                st.write(" -  Ils sont âgés de moins de 29 ans ou de plus de 58 ans")
          
            if submenu_profil == "PREVIOUS" :
                st.write("#### PREVIOUS:","Nombre de contacts effectués avant la campagne avec le client")
                st.write("##### Prioriser les clients déjà contactés")
                st.write("L'objectif managérial serait de **renforcer les actions auprès des clients déjà engagés** tout en explorant des méthodes pour **réactiver ceux qui n'ont jamais été contactés.** ")
                st.write(" -  **Prioriser les clients ayant déjà été contactés :** Étant donné que les clients ayant été contactés précédemment présentent des SHAP values positives, il est recommandé de concentrer les efforts de marketing et de vente sur ces clients. Ils semblent plus susceptibles de souscrire à un produit comme un dépôt à terme. ")
                st.write(" -  **Cibler les clients sans contact préalable :** Pour les clients n'ayant jamais été contactés (avec une SHAP value négative), il pourrait être utile **d'envisager des actions spécifiques pour les réengager**, comme des campagnes de sensibilisation ou des offres adaptées, afin de **stimuler leur intérêt pour l'achat d'un dépôt à terme. Le manque de contact préalable ne signifie pas nécessairement un manque d'intérêt**, mais cela peut refléter un manque de communication qui pourrait être corrigé.")
                st.write(" -  **Optimiser les campagnes de relance :** Pour les clients ayant été contactés une ou plusieurs fois, il est important d'adapter les messages de communication en fonction de leur historique. **Les relances personnalisées** et ciblées peuvent être plus efficaces pour ces clients, en leur offrant des informations supplémentaires sur les avantages des dépôts à terme.")
                st.write(" -  **Utiliser des stratégies de segmentation avancées :** Les clients ayant été contactés plusieurs fois semblent réceptifs aux produits proposés. **Il pourrait être utile de segmenter davantage cette population en fonction de l’intensité et du type de contact préalable** (par exemple, par téléphone, e-mail, ou rencontre en agence) pour **personnaliser encore plus les offres et maximiser les chances de conversion.**")
          
            if submenu_profil == "EDUCATION" :
                st.write("#### EDUCATION:","Niveau d'étude du client")
                

      if submenu_reco == "NOMBRE ET DUREE D’APPEL" :
            submenu_appel = st.radio("", ("DURATION", "CAMPAIGN"), horizontal=True) 

            if submenu_appel == "DURATION" :
                st.write("#### DURATION:","Durée du dernier contact en secondes")
                st.write("Le temps consacré à chaque client s'avère être un facteur clé de succès dans le processus de conversion. **Les commerciaux devraient privilégier des interactions plus longues**, en particulier lors des premières prises de contact. ")
                st.write("**La qualité des échanges est également déterminante.** Des discussions approfondies permettent de mieux cerner les besoins et attentes des clients, ce qui favorise la proposition d’offres personnalisées et augmente les chances de souscription. ")
                st.write("**Les résultats de nos analyses soulignent l’importance stratégique de la durée des interactions commerciales.** En optimisant le temps consacré à chaque client, la banque peut **significativement améliorer** ses taux de conversion et renforcer son efficacité commerciale.")
                st.write("Pour **maximiser les chances de souscription** au produit bancaire DAT, voici les recommandations clés : ")
                st.write(" -  **Encourager des échanges plus longs :** Former les équipes commerciales à adopter une approche engageante dès le premier contact, afin de mieux comprendre les attentes des clients et de leur proposer des solutions sur mesure. ")
                st.write(" -  **Fixer un objectif minimal de 6 minutes par appel** pour les prospects à fort potentiel. Les analyses montrent en effet que **les interactions dépassant 360 secondes (6 minutes) ont un impact significatif**, les SHAP values traduisant une corrélation positive à partir de cette durée. ")
          
            if submenu_appel == "CAMPAIGN" :
                st.write("#### CAMPAIGN:","nombre de contacts effectués avec le client pendant la campagne (dernier contact inclus)")
                st.write("##### Il ne semble pas pertinent de contacter les clients plus d’une fois pendant la campagne. ")
                st.write("Notre modèle montre qu’il ne semble pas payant de contacter les clients plus d’une fois pour leur proposer le produit. Il vaut mieux capitaliser sur des clients pas encore contactés pour leur proposer le produit plutôt que de contacter plusieurs fois le même client. ")

    
      if submenu_reco == "RECAP" :
          st.write("Nous pouvons résumer les résultats de notre **Modèle pour prédire le succès d’une campagne Marketing pour une banque** dans les points suivants: ")
          st.write("##### -  Prioriser les clients qui n’ont pas de prêt immobilier")
          st.write("##### -  Prioriser les clients âgés de 18 à 28 ans et de 59 ans ou plus.")
          st.write("##### -  Contacter en priorité les clients dont la balance est supérieure à 800€.")
          st.write("##### -  Prioriser les clients déjà contactés")
          st.write("##### -  Prioriser xxxxxxxx pr EDUCATION")
          st.write("##### -  Maintenir autant que possible une durée d’appel de minimum 6 minutes ")
          st.write("##### -  Il ne semble pas pertinent de contacter les clients plus d’une fois pendant la campagne. ")
          






if selected == 'Outil  Prédictif':  

    dff_TEST = df.copy()
    dff_TEST = dff_TEST[dff_TEST['age'] < 75]
    dff_TEST = dff_TEST.loc[dff_TEST["balance"] > -2257]
    dff_TEST = dff_TEST.loc[dff_TEST["balance"] < 4087]
    dff_TEST = dff_TEST.loc[dff_TEST["campaign"] < 6]
    dff_TEST = dff_TEST.loc[dff_TEST["previous"] < 2.5]
    dff_TEST = dff_TEST.drop('contact', axis = 1)

    dff_TEST = dff_TEST.drop('pdays', axis = 1)

    dff_TEST = dff_TEST.drop(['day'], axis=1)
    dff_TEST = dff_TEST.drop(['duration'], axis=1)
    dff_TEST = dff_TEST.drop(['job'], axis=1)
    dff_TEST = dff_TEST.drop(['default'], axis=1)
    dff_TEST = dff_TEST.drop(['month'], axis=1)
    dff_TEST = dff_TEST.drop(['poutcome'], axis=1)
    dff_TEST = dff_TEST.drop(['marital'], axis=1)
    dff_TEST = dff_TEST.drop(['loan'], axis=1)
    dff_TEST = dff_TEST.drop(['campaign'], axis=1)   
     
    dff_TEST['education'] = dff_TEST['education'].replace('unknown', np.nan)

    X_dff_TEST = dff_TEST.drop('deposit', axis = 1)
    y_dff_TEST = dff_TEST['deposit']
    
    dff_TEST = dff_TEST.drop(['deposit'], axis=1)   

    # Séparation des données en un jeu d'entrainement et jeu de test
    X_train_o, X_test_o, y_train_o, y_test_o = train_test_split(X_dff_TEST, y_dff_TEST, test_size = 0.20, random_state = 48)
                    
    # On fait de même pour les NaaN de 'education'
    X_train_o['education'] = X_train_o['education'].fillna(method ='bfill')
    X_train_o['education'] = X_train_o['education'].fillna(X_train_o['education'].mode()[0])

    X_test_o['education'] = X_test_o['education'].fillna(method ='bfill')
    X_test_o['education'] = X_test_o['education'].fillna(X_test_o['education'].mode()[0])
                
    # Standardisation des variables quantitatives:
    scaler_o = StandardScaler()
    cols_num_sd = ['age', 'balance', 'previous']
    X_train_o[cols_num_sd] = scaler_o.fit_transform(X_train_o[cols_num_sd])
    X_test_o[cols_num_sd] = scaler_o.transform (X_test_o[cols_num_sd])

    # Encodage de la variable Cible 'deposit':
    le_o = LabelEncoder()
    y_train_o = le_o.fit_transform(y_train_o)
    y_test_o = le_o.transform(y_test_o)

    # Encodage des variables explicatives de type 'objet'
    oneh_o = OneHotEncoder(drop = 'first', sparse_output = False)
    cat1_o = ['housing']
    X_train_o.loc[:, cat1_o] = oneh_o.fit_transform(X_train_o[cat1_o])
    X_test_o.loc[:, cat1_o] = oneh_o.transform(X_test_o[cat1_o])

    X_train_o[cat1_o] = X_train_o[cat1_o].astype('int64')
    X_test_o[cat1_o] = X_test_o[cat1_o].astype('int64')

    # 'education' est une variable catégorielle ordinale, remplacer les modalités de la variable par des nombres, en gardant l'ordre initial
    X_train_o['education'] = X_train_o['education'].replace(['primary', 'secondary', 'tertiary'], [0, 1, 2])
    X_test_o['education'] = X_test_o['education'].replace(['primary', 'secondary', 'tertiary'], [0, 1, 2])

    dff_TEST_loan = df.copy()
    dff_TEST_loan = dff_TEST_loan[dff_TEST_loan['age'] < 75]
    dff_TEST_loan = dff_TEST_loan.loc[dff_TEST_loan["balance"] > -2257]
    dff_TEST_loan = dff_TEST_loan.loc[dff_TEST_loan["balance"] < 4087]
    dff_TEST_loan = dff_TEST_loan.loc[dff_TEST_loan["campaign"] < 6]
    dff_TEST_loan = dff_TEST_loan.loc[dff_TEST_loan["previous"] < 2.5]
    dff_TEST_loan = dff_TEST_loan.drop('contact', axis = 1)
    
    dff_TEST_loan = dff_TEST_loan.drop('pdays', axis = 1)
    
    dff_TEST_loan = dff_TEST_loan.drop(['day'], axis=1)
    dff_TEST_loan = dff_TEST_loan.drop(['duration'], axis=1)
    dff_TEST_loan = dff_TEST_loan.drop(['job'], axis=1)
    dff_TEST_loan = dff_TEST_loan.drop(['default'], axis=1)
    dff_TEST_loan = dff_TEST_loan.drop(['month'], axis=1)
    dff_TEST_loan = dff_TEST_loan.drop(['poutcome'], axis=1)
    dff_TEST_loan = dff_TEST_loan.drop(['marital'], axis=1)
    dff_TEST_loan = dff_TEST_loan.drop(['campaign'], axis=1)   
     
    dff_TEST_loan['education'] = dff_TEST_loan['education'].replace('unknown', np.nan)
    X_dff_TEST_loan = dff_TEST_loan.drop('deposit', axis = 1)
    y_dff_TEST_loan = dff_TEST_loan['deposit']
        
    dff_TEST_loan = dff_TEST_loan.drop(['deposit'], axis=1)   
    
    # Séparation des données en un jeu d'entrainement et jeu de test
    X_train_o_loan, X_test_o_loan, y_train_o_loan, y_test_o_loan = train_test_split(X_dff_TEST_loan, y_dff_TEST_loan, test_size = 0.20, random_state = 48)
                        
    # On fait de même pour les NaaN de 'education'
    X_train_o_loan['education'] = X_train_o_loan['education'].fillna(method ='bfill')
    X_train_o_loan['education'] = X_train_o_loan['education'].fillna(X_train_o_loan['education'].mode()[0])
    
    X_test_o_loan['education'] = X_test_o_loan['education'].fillna(method ='bfill')
    X_test_o_loan['education'] = X_test_o_loan['education'].fillna(X_test_o_loan['education'].mode()[0])
                    
    # Standardisation des variables quantitatives:
    scaler_o = StandardScaler()
    cols_num_sd = ['age', 'balance', 'previous']
    X_train_o_loan[cols_num_sd] = scaler_o.fit_transform(X_train_o_loan[cols_num_sd])
    X_test_o_loan[cols_num_sd] = scaler_o.transform (X_test_o_loan[cols_num_sd])
    
    # Encodage de la variable Cible 'deposit':
    le_o = LabelEncoder()
    y_train_o_loan = le_o.fit_transform(y_train_o_loan)
    y_test_o_loan = le_o.transform(y_test_o_loan)
    
    # Encodage des variables explicatives de type 'objet'
    oneh_o = OneHotEncoder(drop = 'first', sparse_output = False)
    cat1_o = ['housing', 'loan']
    X_train_o_loan.loc[:, cat1_o] = oneh_o.fit_transform(X_train_o_loan[cat1_o])
    X_test_o_loan.loc[:, cat1_o] = oneh_o.transform(X_test_o_loan[cat1_o])
    
    X_train_o_loan[cat1_o] = X_train_o_loan[cat1_o].astype('int64')
    X_test_o_loan[cat1_o] = X_test_o_loan[cat1_o].astype('int64')
        
    # 'education' est une variable catégorielle ordinale, remplacer les modalités de la variable par des nombres, en gardant l'ordre initial
    X_train_o_loan['education'] = X_train_o_loan['education'].replace(['primary', 'secondary', 'tertiary'], [0, 1, 2])
    X_test_o_loan['education'] = X_test_o_loan['education'].replace(['primary', 'secondary', 'tertiary'], [0, 1, 2])

    #DATAFRAME POUR PRED AVEC MARITAL
    dff_TEST_marital = df.copy()
    dff_TEST_marital = dff_TEST_marital[dff_TEST_marital['age'] < 75]
    dff_TEST_marital = dff_TEST_marital.loc[dff_TEST_marital["balance"] > -2257]
    dff_TEST_marital = dff_TEST_marital.loc[dff_TEST_marital["balance"] < 4087]
    dff_TEST_marital = dff_TEST_marital.loc[dff_TEST_marital["campaign"] < 6]
    dff_TEST_marital = dff_TEST_marital.loc[dff_TEST_marital["previous"] < 2.5]
    dff_TEST_marital = dff_TEST_marital.drop('contact', axis = 1)
    
    dff_TEST_marital = dff_TEST_marital.drop('pdays', axis = 1)
    
    dff_TEST_marital = dff_TEST_marital.drop(['day'], axis=1)
    dff_TEST_marital = dff_TEST_marital.drop(['duration'], axis=1)
    dff_TEST_marital = dff_TEST_marital.drop(['job'], axis=1)
    dff_TEST_marital = dff_TEST_marital.drop(['default'], axis=1)
    dff_TEST_marital = dff_TEST_marital.drop(['month'], axis=1)
    dff_TEST_marital = dff_TEST_marital.drop(['loan'], axis=1)
    dff_TEST_marital = dff_TEST_marital.drop(['poutcome'], axis=1)
    dff_TEST_marital = dff_TEST_marital.drop(['campaign'], axis=1)   
     
    dff_TEST_marital['education'] = dff_TEST_marital['education'].replace('unknown', np.nan)
    X_dff_TEST_marital = dff_TEST_marital.drop('deposit', axis = 1)
    y_dff_TEST_marital = dff_TEST_marital['deposit']
        
    dff_TEST_marital = dff_TEST_marital.drop(['deposit'], axis=1) 

    dummies = pd.get_dummies(dff_TEST_marital['marital'], prefix='marital').astype(int)
    dff_TEST_marital = pd.concat([dff_TEST_marital.drop('marital', axis=1), dummies], axis=1)
    


    #DATAFRAME POUR PRED AVEC POUTCOME
    dff_TEST_poutcome = df.copy()
    dff_TEST_poutcome = dff_TEST_poutcome[dff_TEST_poutcome['age'] < 75]
    dff_TEST_poutcome = dff_TEST_poutcome.loc[dff_TEST_poutcome["balance"] > -2257]
    dff_TEST_poutcome = dff_TEST_poutcome.loc[dff_TEST_poutcome["balance"] < 4087]
    dff_TEST_poutcome = dff_TEST_poutcome.loc[dff_TEST_poutcome["campaign"] < 6]
    dff_TEST_poutcome = dff_TEST_poutcome.loc[dff_TEST_poutcome["previous"] < 2.5]
    dff_TEST_poutcome = dff_TEST_poutcome.drop('contact', axis = 1)
    
    dff_TEST_poutcome = dff_TEST_poutcome.drop('pdays', axis = 1)
    
    dff_TEST_poutcome = dff_TEST_poutcome.drop(['day'], axis=1)
    dff_TEST_poutcome = dff_TEST_poutcome.drop(['duration'], axis=1)
    dff_TEST_poutcome = dff_TEST_poutcome.drop(['job'], axis=1)
    dff_TEST_poutcome = dff_TEST_poutcome.drop(['default'], axis=1)
    dff_TEST_poutcome = dff_TEST_poutcome.drop(['month'], axis=1)
    dff_TEST_poutcome = dff_TEST_poutcome.drop(['loan'], axis=1)
    dff_TEST_poutcome = dff_TEST_poutcome.drop(['marital'], axis=1)
    dff_TEST_poutcome = dff_TEST_poutcome.drop(['campaign'], axis=1)   
     
    dff_TEST_poutcome['education'] = dff_TEST_poutcome['education'].replace('unknown', np.nan)
    dff_TEST_poutcome['poutcome'] = dff_TEST_poutcome['poutcome'].replace('unknown', np.nan)
    
    X_dff_TEST_poutcome = dff_TEST_poutcome.drop('deposit', axis = 1)
    y_dff_TEST_poutcome = dff_TEST_poutcome['deposit']
        
    dff_TEST_poutcome = dff_TEST_poutcome.drop(['deposit'], axis=1)

    dff_TEST_poutcome['poutcome'] = dff_TEST_poutcome['poutcome'].fillna(method ='bfill')
    dff_TEST_poutcome['poutcome'] = dff_TEST_poutcome['poutcome'].fillna(dff_TEST_poutcome['poutcome'].mode()[0])    

    dummies = pd.get_dummies(dff_TEST_poutcome['poutcome'], prefix='poutcome').astype(int)
    dff_TEST_poutcome = pd.concat([dff_TEST_poutcome.drop('poutcome', axis=1), dummies], axis=1)

    #DATAFRAME POUR PRED JOB
    dff_TEST_job = df.copy()
    dff_TEST_job = dff_TEST_job[dff_TEST_job['age'] < 75]
    dff_TEST_job = dff_TEST_job.loc[dff_TEST_job["balance"] > -2257]
    dff_TEST_job = dff_TEST_job.loc[dff_TEST_job["balance"] < 4087]
    dff_TEST_job = dff_TEST_job.loc[dff_TEST_job["campaign"] < 6]
    dff_TEST_job = dff_TEST_job.loc[dff_TEST_job["previous"] < 2.5]
    dff_TEST_job = dff_TEST_job.drop('contact', axis = 1)
    
    dff_TEST_job = dff_TEST_job.drop('pdays', axis = 1)
    
    dff_TEST_job = dff_TEST_job.drop(['day'], axis=1)
    dff_TEST_job = dff_TEST_job.drop(['duration'], axis=1)
    dff_TEST_job = dff_TEST_job.drop(['poutcome'], axis=1)
    dff_TEST_job = dff_TEST_job.drop(['default'], axis=1)
    dff_TEST_job = dff_TEST_job.drop(['month'], axis=1)
    dff_TEST_job = dff_TEST_job.drop(['loan'], axis=1)
    dff_TEST_job = dff_TEST_job.drop(['marital'], axis=1)
    dff_TEST_job = dff_TEST_job.drop(['campaign'], axis=1)   
     
    dff_TEST_job['education'] = dff_TEST_job['education'].replace('unknown', np.nan)
    dff_TEST_job['job'] = dff_TEST_job['job'].replace('unknown', np.nan)


    imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
    dff_TEST_job.loc[:,['job']] = imputer.fit_transform(dff_TEST_job[['job']])
    dff_TEST_job.loc[:,['job']] = imputer.transform(dff_TEST_job[['job']])

    dummies = pd.get_dummies(dff_TEST_job['job'], prefix='job').astype(int)
    dff_TEST_job = pd.concat([dff_TEST_job.drop('job', axis=1), dummies], axis=1)
    
    dff_TEST_job = dff_TEST_job.drop(['deposit'], axis=1)  

    #DATAFRAME POUR PRED CLIENT CATEGORY
    dff_TEST_client_category = df.copy()
    dff_TEST_client_category = dff_TEST_client_category[dff_TEST_client_category['age'] < 75]
    dff_TEST_client_category = dff_TEST_client_category.loc[dff_TEST_client_category["balance"] > -2257]
    dff_TEST_client_category = dff_TEST_client_category.loc[dff_TEST_client_category["balance"] < 4087]
    dff_TEST_client_category = dff_TEST_client_category.loc[dff_TEST_client_category["campaign"] < 6]
    dff_TEST_client_category = dff_TEST_client_category.loc[dff_TEST_client_category["previous"] < 2.5]
    dff_TEST_client_category = dff_TEST_client_category.drop('contact', axis = 1)
    
    
    dff_TEST_client_category = dff_TEST_client_category.drop(['day'], axis=1)
    dff_TEST_client_category = dff_TEST_client_category.drop(['duration'], axis=1)
    dff_TEST_client_category = dff_TEST_client_category.drop(['poutcome'], axis=1)
    dff_TEST_client_category = dff_TEST_client_category.drop(['default'], axis=1)
    dff_TEST_client_category = dff_TEST_client_category.drop(['month'], axis=1)
    dff_TEST_client_category = dff_TEST_client_category.drop(['loan'], axis=1)
    dff_TEST_client_category = dff_TEST_client_category.drop(['marital'], axis=1)
    dff_TEST_client_category = dff_TEST_client_category.drop(['campaign'], axis=1)   
    dff_TEST_client_category = dff_TEST_client_category.drop(['job'], axis=1)   
    
    # Les catégories de clients selon qu'ils soient contactés pour la première fois, il y a moins de 6 mois, ou plus de 6 mois:
    bins = [-2, -1, 180, 855]
    labels = ['Prospect', 'Reached-6M', 'Reached+6M']
    dff_TEST_client_category['Client_Category_M'] = pd.cut(dff_TEST_client_category['pdays'], bins=bins, labels=labels)
    
    # Transformation de 'Client_Category' en type 'objet'
    dff_TEST_client_category['Client_Category_M'] = dff_TEST_client_category['Client_Category_M'].astype('object')
    
    # Suppression de pdays
    dff_TEST_client_category = dff_TEST_client_category.drop('pdays', axis =1)
    
    dff_TEST_client_category['education'] = dff_TEST_client_category['education'].replace('unknown', np.nan)
    
    # 'Client_Category_M' est une variable catégorielle ordinale, remplacer les modalités de la variable par des nombres, en gardant l'ordre initial
    dff_TEST_client_category['Client_Category_M'] = dff_TEST_client_category['Client_Category_M'].replace(['Prospect', 'Reached-6M', 'Reached+6M'], [0, 1, 2])


    st.title("Démonstration et application de notre modèle à votre cas")               

    st.subheader('Informations sur le client')
    # Collecte de l'âge sans valeur par défaut
    age_input = st.text_input("Quel est l'âge du client ?")  
    age = None
    if age_input:  # Vérifie si age_input n'est pas vide
        try:
            # Convertir l'entrée en entier
            age = int(age_input)
        
        # Vérifier si l'âge est dans la plage valide
            if age < 18 or age > 95:
                st.error("L'âge doit être compris entre 18 et 95 ans.")

        except ValueError:
            st.error("Veuillez entrer un nombre valide pour l'âge.")
    else:
    # Quand le champ est vide, on ne fait rien
        pass  # Vous pouvez aussi utiliser `st.write("")` pour rien afficher.
    
    education = st.selectbox("Quel est son niveau d'étude ?", ("tertiary", "secondary", "unknown", "primary"))
    #conditions d'affichage pour education : 
    if education == "tertiary":
        niveau_etude = "Tertiaire"
    elif education == "secondary":
        niveau_etude = "Secondaire"
    elif education == "primary":
        niveau_etude = "Primaire"
    elif education == "unknown":
        niveau_etude = "Inconnu"
    else:
        niveau_etude = "Inconnu"  # Par défaut si `education` a une valeur inattendue
         
    # Collecte du solde bancaire avec vérification
    balance_input = st.text_input("Quel est le solde de son compte en banque ?")  # Pas de valeur par défaut
    
    # Validation de l'entrée pour le solde
    balance = None
    if balance_input:  # Vérifie si balance_input n'est pas vide
        try:
            # Convertir l'entrée en int pour gérer le solde comme un entier
            balance = int(balance_input)
        except ValueError:
            st.error("Veuillez entrer un nombre entier valide pour le solde.")
    else : 
        pass
    
    housing = st.selectbox("As-t-il un crédit immobilier ?", ('yes', 'no'))

    previous = st.slider("Lors de la précédente campagne marketing, combien de fois le client a-t-il été contacté ?", 0,6,1)
    

    # Vérifiez si age et balance sont correctement remplis
    if age is not None and balance is not None:
        # Affichage du récapitulatif
        st.write(f'### Récapitulatif')
        st.write("Le client a :  ", age, "ans")   
        st.write("Le client a un niveau d'étude :  ", niveau_etude)
        st.write("Le solde de son compte en banque est de :  ", balance, "euros")
        st.write("Le client est-il propriétaire :  ", "Oui" if housing == 1 else "Non")
        st.write("Le clients a été contacté  ", previous, " fois lors de la dernière campagne marketing")
  
        
        # Créer un dataframe récapitulatif des données du prospect
        infos_prospect = pd.DataFrame({
            'age': [age], 
            'education': [education], 
            'balance': [balance], 
            'housing': [housing], 
            'previous': [previous],
        }, index=[0]) 
    
        # Affichage pour vérifier le nouvel index
        #st.subheader("Voici le tableau avec vos informations")
        #st.dataframe(infos_prospect)
    
        # Construction du DataFrame pour le prospect à partir de infos_prospect
        pred_df = infos_prospect.copy()
    
        # Remplacer 'unknown' par NaN uniquement pour les colonnes spécifiques
        cols_to_check = ['education']  # Colonnes à vérifier
        for col in cols_to_check:
            if (pred_df[col] == 'unknown').any():  # Vérifie si la valeur est "unknown"
                pred_df[col] = np.nan  # Remplace "unknown" par NaN
    
        # Remplissage par le mode pour 'education' et 'poutcome' dans le cas où il y a des NaN
        if pred_df['education'].isna().any():
            # Utiliser le mode de 'education' dans dff
            pred_df['education'] = dff_TEST['education'].mode()[0]
    
        # Transformation de 'education' et 'Client_Category_M' pour respecter l'ordre ordinal
        pred_df['education'] = pred_df['education'].replace(['primary', 'secondary', 'tertiary'], [0, 1, 2])
        
    
        # Remplacer 'yes' par 1 et 'no' par 0 pour chaque colonne
        cols_to_replace = ['housing']
        for col in cols_to_replace:
            pred_df[col] = pred_df[col].replace({'yes': 1, 'no': 0})
    
    
        # Réorganiser les colonnes pour correspondre exactement à celles de dff
        pred_df = pred_df.reindex(columns=dff_TEST.columns, fill_value=0)
        
        # Affichage du DataFrame transformé avant la standardisation
        #st.write("Affichage du dataframe transformé (avant standardisation):")
        #st.dataframe(pred_df)
    
        # Liste des colonnes numériques à standardiser
        num_cols = ['age', 'balance','previous']
    
        # Étape 1 : Créer un index spécifique pour pred_df
        # Utiliser un index unique pour pred_df, en le commençant après la dernière ligne de dff
        pred_df.index = range(dff_TEST.shape[0], dff_TEST.shape[0] + len(pred_df))
    
        # Étape 2 : Concaténer dff et pred_df
        # Concaténer les deux DataFrames dff et pred_df sur les colonnes numériques
        combined_df = pd.concat([dff_TEST[num_cols], pred_df[num_cols]], axis=0)
    
        # Étape 3 : Standardisation des données numériques
        scaler = StandardScaler()
        combined_df[num_cols] = scaler.fit_transform(combined_df[num_cols])
    
        # Étape 4 : Séparer à nouveau pred_df des autres données
        # On récupère uniquement les lignes correspondant à pred_df en utilisant l'index spécifique
        pred_df[num_cols] = combined_df.loc[pred_df.index, num_cols]
    
        # Réinitialiser l'index de pred_df après la manipulation (facultatif)
        pred_df = pred_df.reset_index(drop=True)
      
        
        # Interface utilisateur
        filename = "dilenesantos/XGBOOST_1_SD_model_PRED_AVEC_parametres.pkl"
        model_XGBOOST_1_SD_model_PRED_AVEC_parametres = joblib.load(filename)

        # Prédiction
        prediction = model_XGBOOST_1_SD_model_PRED_AVEC_parametres.predict(pred_df)
        prediction_proba = model_XGBOOST_1_SD_model_PRED_AVEC_parametres.predict_proba(pred_df)
        max_proba = np.max(prediction_proba[0]) * 100
    
        st.write("__________________________________________________")
        # Affichage des résultats
        st.subheader(f"Prediction : {prediction[0]}")
        st.markdown(f"**Niveau de confiance: {max_proba:.2f}%**")
    
        if prediction[0] == 0:
            st.write("Conclusion: Ce client n'est pas susceptible de souscrire à un dépôt à terme.")
        else:
            st.write("Conclusion: Ce client est susceptible de souscrire à un dépôt à terme.")
            st.write("\nRecommandations : ")
            st.write("- Durée d'appel : pour maximiser les chances de souscription au dépôt, veiller à rester le plus longtemps possible au téléphone avec ce client.")
            st.write("- Nombre de contacts : il serait contre-productif de le contacter plus d'une fois.")
    
            st.write("__________________________________________________")
    
            st.markdown("**Si vous le souhaitez, vous pouvez affiner la prédiction en ajoutant une autre information concernant votre client.**")
            
            # Afficher le sélecteur d'option pour le raffinement, incluant l'option pour ne rien ajouter
            option_to_add = st.radio("Choisir une information à ajouter :", 
                                           ["None", "loan", "marital", "poutcome", "job", "Dernier_contact"], horizontal=True)
            
            if option_to_add != "None":
                # Ajout de la logique pour chaque option sélectionnée
                if option_to_add == "loan":
                    loan = st.selectbox("A-t-il un crédit personnel ?", ('yes', 'no'))
                    pred_df['loan'] = loan
                    # Remplacer 'yes' par 1 et 'no' par 0 pour chaque colonne
                    cols_to_replace = ['loan']
                    for col in cols_to_replace:
                        pred_df[col] = pred_df[col].replace({'yes': 1, 'no': 0})
                    # Réorganiser les colonnes pour correspondre exactement à celles de dff
                    pred_df = pred_df.reindex(columns=dff_TEST_loan.columns, fill_value=0)
                     # Utiliser un index unique pour pred_df, en le commençant après la dernière ligne de dff
                    pred_df.index = range(dff_TEST_loan.shape[0], dff_TEST_loan.shape[0] + len(pred_df))
                
                    # Étape 2 : Concaténer dff et pred_df
                    # Concaténer les deux DataFrames dff et pred_df sur les colonnes numériques
                    num_cols = ['age', 'balance','previous']
                    combined_df_loan = pd.concat([dff_TEST_loan[num_cols], pred_df[num_cols]], axis=0)
    
                    # Étape 4 : Séparer à nouveau pred_df des autres données
                    # On récupère uniquement les lignes correspondant à pred_df en utilisant l'index spécifique
                    pred_df[num_cols] = combined_df_loan.loc[pred_df.index, num_cols]
                
                    # Réinitialiser l'index de pred_df après la manipulation (facultatif)
                    pred_df = pred_df.reset_index(drop=True)
    
                    # Conditions pour charger le modèle approprié
                    filename_LOAN = "dilenesantos/XGBOOST_1_SD_model_PRED_loan_XGBOOST_1.pkl"
                    additional_model = joblib.load(filename_LOAN)
                
                    # Prédiction avec le DataFrame optimisé
                    prediction_opt_loan = additional_model.predict(pred_df)
                    prediction_proba_opt_loan = additional_model.predict_proba(pred_df)
                    max_proba_opt_loan = np.max(prediction_proba_opt_loan[0]) * 100
                
                    # Affichage des résultats
                    st.markdown(f"Prediction après affinage : **{prediction_opt_loan[0]}**")
                    st.markdown(f"Niveau de confiance après affinage : **{max_proba_opt_loan:.2f}%**")
                    if prediction_opt_loan[0] == 0:
                        st.write("Conclusion : Ce client n'est pas susceptible de souscrire à un dépôt à terme.")
                    else:
                        st.write("Conclusion : Ce client est susceptible de souscrire à un dépôt à terme.")
                
        
                elif option_to_add == "marital":
                    marital = st.selectbox("Quelle est la situation maritale du client ?", ("married", "single", "divorced"))
                    pred_df['marital'] = marital
                    st.write("Situation maritale : ", marital)
                    
                    # Liste des variables catégorielles multi-modales à traiter
                    cat_cols_multi_modal = ['marital']
                    # Parcourir chaque variable catégorielle multi-modale pour gérer les colonnes manquantes
                    for col in cat_cols_multi_modal:
                        # Effectuer un encodage des variables catégorielles multi-modales
                        dummies = pd.get_dummies(pred_df[col], prefix=col).astype(int)
                        pred_df = pd.concat([pred_df.drop(col, axis=1), dummies], axis=1)
                                
                    # Réorganiser les colonnes pour correspondre exactement à celles de dff
                    pred_df = pred_df.reindex(columns=dff_TEST_marital.columns, fill_value=0)
                    
                    # Étape 2 : Concaténer dff et pred_df
                    # Concaténer les deux DataFrames dff et pred_df sur les colonnes numériques
                    num_cols = ['age', 'balance','previous']
                    
                    # Utiliser un index unique pour pred_df, en le commençant après la dernière ligne de dff
                    pred_df.index = range(dff_TEST_marital.shape[0], dff_TEST_marital.shape[0] + len(pred_df))
                
                    combined_df_marital = pd.concat([dff_TEST_marital[num_cols], pred_df[num_cols]], axis=0)
        
                    # Étape 3 : Standardisation des données numériques
                    scaler = StandardScaler()
                    combined_df_marital[num_cols] = scaler.fit_transform(combined_df_marital[num_cols])
        
                    # Étape 4 : Séparer à nouveau pred_df des autres données
                    # On récupère uniquement les lignes correspondant à pred_df en utilisant l'index spécifique
                    pred_df[num_cols] = combined_df_marital.loc[pred_df.index, num_cols]
                
                    # Réinitialiser l'index de pred_df après la manipulation (facultatif)
                    pred_df = pred_df.reset_index(drop=True)
              
                     # Conditions pour charger le modèle approprié
                    filename_marital = "dilenesantos/XGBOOST_1_SD_model_PRED_marital_XGBOOST_1.pkl"
                    additional_model = joblib.load(filename_marital)
                
                    # Prédiction avec le DataFrame optimisé
                    prediction_opt_marital = additional_model.predict(pred_df)
                    prediction_proba_opt_marital = additional_model.predict_proba(pred_df)
                    max_proba_opt_marital = np.max(prediction_proba_opt_marital[0]) * 100
                
                    # Affichage des résultats
                    st.markdown(f"Prediction après affinage : **{prediction_opt_marital[0]}**")
                    st.markdown(f"Niveau de confiance après affinage : **{max_proba_opt_marital:.2f}%**")
                    if prediction_opt_marital[0] == 0:
                        st.write("Conclusion : Ce client n'est pas susceptible de souscrire à un dépôt à terme.")
                    else:
                        st.write("Conclusion : Ce client est susceptible de souscrire à un dépôt à terme.")
                
        
                elif option_to_add == "poutcome":
                    poutcome = st.selectbox("Quel a été le résultat de la précédente campagne avec le client ?", ('success', 'failure', 'other'))
                    pred_df['poutcome'] = poutcome
                    st.write("Résultat de la campagne : ", poutcome)
                    
                   
                    # Liste des variables catégorielles multi-modales à traiter
                    cat_cols_multi_modal_poutcome = ['poutcome']
                    # Parcourir chaque variable catégorielle multi-modale pour gérer les colonnes manquantes
                    for col in cat_cols_multi_modal_poutcome:
                        # Effectuer un encodage des variables catégorielles multi-modales
                        dummies = pd.get_dummies(pred_df[col], prefix=col).astype(int)
                        pred_df = pd.concat([pred_df.drop(col, axis=1), dummies], axis=1)
                                
                    # Réorganiser les colonnes pour correspondre exactement à celles de dff
                    pred_df = pred_df.reindex(columns=dff_TEST_poutcome.columns, fill_value=0)
                    
                    # Étape 2 : Concaténer dff et pred_df
                    # Concaténer les deux DataFrames dff et pred_df sur les colonnes numériques
                    num_cols = ['age', 'balance','previous']
                    
                    # Utiliser un index unique pour pred_df, en le commençant après la dernière ligne de dff
                    pred_df.index = range(dff_TEST_poutcome.shape[0], dff_TEST_poutcome.shape[0] + len(pred_df))
                
                    combined_df_poutcome = pd.concat([dff_TEST_poutcome[num_cols], pred_df[num_cols]], axis=0)
        
                    # Étape 3 : Standardisation des données numériques
                    scaler = StandardScaler()
                    combined_df_poutcome[num_cols] = scaler.fit_transform(combined_df_poutcome[num_cols])
        
                    # Étape 4 : Séparer à nouveau pred_df des autres données
                    # On récupère uniquement les lignes correspondant à pred_df en utilisant l'index spécifique
                    pred_df[num_cols] = combined_df_poutcome.loc[pred_df.index, num_cols]
                
                    # Réinitialiser l'index de pred_df après la manipulation (facultatif)
                    pred_df = pred_df.reset_index(drop=True)
              
                     # Conditions pour charger le modèle approprié
                    filename_poutcome = "dilenesantos/XGBOOST_1_SD_model_PRED_poutcome_XGBOOST_quater.pkl"
                    additional_model = joblib.load(filename_poutcome)
                
                    # Prédiction avec le DataFrame optimisé
                    prediction_opt_poutcome = additional_model.predict(pred_df)
                    prediction_proba_opt_poutcome = additional_model.predict_proba(pred_df)
                    max_proba_opt_poutcome = np.max(prediction_proba_opt_poutcome[0]) * 100
                
                    # Affichage des résultats
                    st.markdown(f"Prediction après affinage : **{prediction_opt_poutcome[0]}**")
                    st.markdown(f"Niveau de confiance après affinage : **{max_proba_opt_poutcome:.2f}%**")
                    if prediction_opt_poutcome[0] == 0:
                        st.write("Conclusion : Ce client n'est pas susceptible de souscrire à un dépôt à terme.")
                    else:
                        st.write("Conclusion : Ce client est susceptible de souscrire à un dépôt à terme.")
                
        
        
                elif option_to_add == "job":
                    job = st.selectbox("Quel est l'emploi du client ?", ('admin.', 'blue-collar', 'entrepreneur',
                                                                         'housemaid', 'management', 'retired', 
                                                                         'self-employed', 'services', 'student', 
                                                                         'technician', 'unemployed'))
                    pred_df['job'] = job
                    st.write("Emploi : ", job)
                 
                    # Liste des variables catégorielles multi-modales à traiter
                    cat_cols_multi_modal_job = ['job']
                    # Parcourir chaque variable catégorielle multi-modale pour gérer les colonnes manquantes
                    for col in cat_cols_multi_modal_job:
                        # Effectuer un encodage des variables catégorielles multi-modales
                        dummies = pd.get_dummies(pred_df[col], prefix=col).astype(int)
                        pred_df = pd.concat([pred_df.drop(col, axis=1), dummies], axis=1)
                                
                    # Réorganiser les colonnes pour correspondre exactement à celles de dff
                    pred_df = pred_df.reindex(columns=dff_TEST_job.columns, fill_value=0)
                    
                    # Étape 2 : Concaténer dff et pred_df
                    # Concaténer les deux DataFrames dff et pred_df sur les colonnes numériques
                    num_cols = ['age', 'balance','previous']
                    
                    # Utiliser un index unique pour pred_df, en le commençant après la dernière ligne de dff
                    pred_df.index = range(dff_TEST_job.shape[0], dff_TEST_job.shape[0] + len(pred_df))
                
                    combined_df_job = pd.concat([dff_TEST_job[num_cols], pred_df[num_cols]], axis=0)
        
                    # Étape 3 : Standardisation des données numériques
                    scaler = StandardScaler()
                    combined_df_job[num_cols] = scaler.fit_transform(combined_df_job[num_cols])
        
                    # Étape 4 : Séparer à nouveau pred_df des autres données
                    # On récupère uniquement les lignes correspondant à pred_df en utilisant l'index spécifique
                    pred_df[num_cols] = combined_df_job.loc[pred_df.index, num_cols]
                
                    # Réinitialiser l'index de pred_df après la manipulation (facultatif)
                    pred_df = pred_df.reset_index(drop=True)
              
                     # Conditions pour charger le modèle approprié
                    filename_job = "dilenesantos/XGBOOST_1_SD_model_PRED_job_XGBOOST_1.pkl"
                    additional_model = joblib.load(filename_job)
                
                    # Prédiction avec le DataFrame optimisé
                    prediction_opt_job = additional_model.predict(pred_df)
                    prediction_proba_opt_job = additional_model.predict_proba(pred_df)
                    max_proba_opt_job = np.max(prediction_proba_opt_job[0]) * 100
                
                    # Affichage des résultats
                    st.markdown(f"Prediction après affinage : **{prediction_opt_job[0]}**")
                    st.markdown(f"Niveau de confiance après affinage : **{max_proba_opt_job:.2f}%**")
                    if prediction_opt_job[0] == 0:
                        st.write("Conclusion : Ce client n'est pas susceptible de souscrire à un dépôt à terme.")
                    else:
                        st.write("Conclusion : Ce client est susceptible de souscrire à un dépôt à terme.")
                
        
                
                elif option_to_add == "Dernier_contact":
                    Dernier_contact = st.selectbox("À quand remonte le dernier contact avec le client lors de la précédente campagne?", ('Client jamais contacté', 'Client contacté il y a moins de 6 mois', 'Client contacté il y a plus de 6  mois'))
                        
                    pred_df['Client_Category_M'] = Dernier_contact
                    pred_df['Client_Category_M'] = pred_df['Client_Category_M'].replace(['Client jamais contacté', 'Client contacté il y a moins de 6 mois', 'Client contacté il y a plus de 6  mois'], [0, 1, 2])
                    st.write("Dernier contact de la banque avec le client lors de la précédente campagne : ", Dernier_contact)
                    
                    # Étape 2 : Concaténer dff et pred_df
                    # Concaténer les deux DataFrames dff et pred_df sur les colonnes numériques
                    num_cols = ['age', 'balance','previous']
                    
                    # Utiliser un index unique pour pred_df, en le commençant après la dernière ligne de dff
                    pred_df.index = range(dff_TEST_client_category.shape[0], dff_TEST_client_category.shape[0] + len(pred_df))
                
                    combined_df_client_category = pd.concat([dff_TEST_client_category[num_cols], pred_df[num_cols]], axis=0)
        
                    # Étape 3 : Standardisation des données numériques
                    scaler = StandardScaler()
                    combined_df_client_category[num_cols] = scaler.fit_transform(combined_df_client_category[num_cols])
        
                    # Étape 4 : Séparer à nouveau pred_df des autres données
                    # On récupère uniquement les lignes correspondant à pred_df en utilisant l'index spécifique
                    pred_df[num_cols] = combined_df_client_category.loc[pred_df.index, num_cols]
                
                    # Réinitialiser l'index de pred_df après la manipulation (facultatif)
                    pred_df = pred_df.reset_index(drop=True)
                    
                     # Conditions pour charger le modèle approprié
                    filename_client_category = "dilenesantos/XGBOOST_1_SD_model_PRED_client_category_XGBOOST_1.pkl"
                    additional_model = joblib.load(filename_client_category)
                
                    # Prédiction avec le DataFrame optimisé
                    prediction_opt_client_category = additional_model.predict(pred_df)
                    prediction_proba_opt_client_category = additional_model.predict_proba(pred_df)
                    max_proba_opt_client_category = np.max(prediction_proba_opt_client_category[0]) * 100
                
                    # Affichage des résultats
                    st.markdown(f"Prediction après affinage : **{prediction_opt_client_category[0]}**")
                    st.markdown(f"Niveau de confiance après affinage : **{max_proba_opt_client_category:.2f}%**")
                    if prediction_opt_client_category[0] == 0:
                        st.write("Conclusion : Ce client n'est pas susceptible de souscrire à un dépôt à terme.")
                    else:
                        st.write("Conclusion : Ce client est susceptible de souscrire à un dépôt à terme.")
                
    
    
                # Afficher le récapitulatif
                st.write(f'### Récapitulatif')
                st.write("Le client a : ", age, "ans")
                st.write("Le client a un niveau d'étude : ", niveau_etude)
                st.write("Le solde de son compte en banque est de : ", balance, "euros")
                st.write("Le client est-il propriétaire : ", "Oui" if housing == 1 else "Non")
                st.write("Le client a été contacté ", previous, " fois lors de la précédente campagne marketing")
                
                # Afficher les informations supplémentaires définies
                if option_to_add == "loan":
                    st.write(f"A un crédit personnel : {loan}")
                elif option_to_add == "marital":
                    st.write(f"Situation maritale : {marital}")
                elif option_to_add == "poutcome":
                    st.write(f"Résultat de la campagne : {poutcome}")
                elif option_to_add == "job":
                    st.write(f"Emploi : {job}")
                elif option_to_add == "Dernier_contact":
                    st.write("Dernier contact avec le client lors de la précédente campagne : ", Dernier_contact)
         
    
    
