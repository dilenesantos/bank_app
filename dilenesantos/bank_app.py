import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

import scipy.stats as stats
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
        options=['Introduction','DataVisualisation', "Pre-processing", "Modélisation", "Interprétation", "Recommandations & Perspectives", "TEST PRED SCORES", "Outil Prédictif", "TEST PRÉDICTIF", "PRED POUSSÉ"]) 

if selected == 'Introduction':  
    st.subheader("Contexte du projet")
    st.write("Le projet vise à analyser des données marketing issues d'une banque qui a utilisé le télémarketing pour promouvoir un produit financier appelé 'dépôt à terme'. Ce produit nécessite que le client dépose une somme d'argent dans un compte dédié, sans possibilité de retrait avant une date déterminée. En retour, le client reçoit des intérêts à la fin de cette période. L'objectif de cette analyse est d'examiner les informations personnelles des clients, comme l'âge, le statut matrimonial, le montant d'argent déposé, le nombre de contacts réalisés, etc., afin de comprendre les facteurs qui influencent la décision des clients de souscrire ou non à ce produit financier.")
    

    st.write("Problématique : ")
    st.write("La principale problématique de ce projet est de déterminer les facteurs qui influencent la probabilité qu'un client souscrive à un dépôt à terme à la suite d'une campagne de télémarketing.")
    st.write("L'objectif est double :")
    st.write("- Identifier et analyser visuellement et statistiquement les caractéristiques des clients qui sont corrélées avec la souscription au 'dépôt à terme'.")
    st.write("- Utiliser des techniques de Machine Learning pour prédire si un client va souscrire au 'dépôt à terme'.")
    
    st.write("BLABLABLA")
    st.write("FATOU")
    st.write("carolle")

if selected == 'DataVisualisation':      
    st.title("DATAVISUALISATION")
    st.sidebar.title("SOUS MENU DATAVISUALISATION")
    option_submenu = st.sidebar.selectbox('Sélection', ("Description des données", "Analyse des variables", "Analyse des variables qualitatives", "Corrélations entre les variables", "Évolution de la variable deposit dans le temps"))
    if option_submenu == 'Description des données':
        st.subheader("Description des données")
        pages=["Describe", "Valeurs uniques des variables catégorielles", "Afficher les NAns et les Unknowns", "Répartition Deposit"]
        page=st.sidebar.radio('Afficher', pages)
    
    
        if page == pages[0] :
            st.dataframe(df.describe())
        
        if page == pages[1] : 
            var_quali = df.select_dtypes(include='object')
            for col in var_quali :
                st.write(col)
                st.dataframe(df[col].unique())
    
        if page == pages[2] :
            st.write('Volume de NAns du dataframe :')
            st.dataframe(df.isna().sum())
            st.write("----------------")
            #affichage du % des valeurs affichant 'unknown' pour les colonnes concernées Job, Education, Contact et poutcome
            col_unknown = ['job', 'education', 'contact', 'poutcome']
        
            st.write("____________________________________")
        
            st.write("Volume de Unknows : ")
            for col in col_unknown:
                st.write(col)
                result = round((df[col].value_counts(normalize=True)['unknown']*100),2)
                st.write(result,"%")
 
        if page == pages[3] :
            fig = plt.figure()
            sns.countplot(x = 'deposit', hue = 'deposit', data=df, palette =("g", "r"), legend=False)
            plt.title("Répartition de notre variable cible")
            st.pyplot(fig)
            st.write("Commentaires : blabla")
    
    
        st.write("____________________________________")

    var_quali = df.select_dtypes(include='object')
    var_quanti = df.select_dtypes(exclude='object')

    if option_submenu == 'Analyse des variables':
        st.subheader("Analyse des variables")
        pages=["Distribution des variables quantitatives", "Boxplot des variables quantitatives", "Boxplot des variables quantitatives selon Deposit"]
        page=st.sidebar.radio('Afficher', pages)
    
        if page == pages[0] :
            st.write("Distribution des variables quantitatives")
            fig = plt.figure(figsize=(20,60))
            plotnumber =1
            for column in var_quanti :
                ax = plt.subplot(12,3,plotnumber)
                sns.kdeplot(df[column], fill=True)
                plotnumber+=1
            st.pyplot(fig)
            st.write("Commentaires : blabla")
    
        if page == pages[1] :
            st.write("Boxplot des variables quantitatives")
            fig = plt.figure(figsize=(20,60), facecolor='white')
            plotnumber =1
            for column in var_quanti :
                ax = plt.subplot(12,3,plotnumber)
                sns.boxplot(df[column])
                plotnumber+=1
            st.pyplot(fig)
            st.write("Commentaires : blabla")

        if page == pages[2] :
            st.write("Boxplot des variables quantitatives selon Deposit")
            fig = plt.figure(figsize=(20,60), facecolor='white')
            plotnumber =1
            for column in var_quanti :
                ax = plt.subplot(12,3,plotnumber)
                sns.boxplot(y= df[column], hue = "deposit", data=df, palette =("g", "r"), legend=False)
                plt.xlabel('deposit')
                plotnumber+=1
            st.pyplot(fig)
            st.write("Commentaires : blabla")
    
        st.write("____________________________________")
    
    if option_submenu == 'Analyse des variables qualitatives':
    
        st.write("Distribution des variables qualitatives")
        fig = plt.figure(figsize=(25,70), facecolor='white')
        plotnumber =1
        for column in var_quali:
            ax = plt.subplot(12,3,plotnumber)
            sns.countplot(y=column, data=df, order = df[column].value_counts().index, color = "c")
            plt.xlabel(column)
            plotnumber+=1
        st.pyplot(fig) 

        st.write("____________________________________")


    
        st.write("Deposit selon les caractéristiques socio-démo des clients :")
    
        # Store the initial value of widgets in session state
        col1, col2 = st.columns(2)

        with col1:
            boxchoices21 = selectbox("Sélectionner", ["Deposit selon leur âge", "Deposit selon leur statut marital", "Deposit selon leur job"])

        with col2:
            st.write("Sélection : ",boxchoices21)
    
            if boxchoices21 == "Deposit selon leur âge" :
                fig = sns.displot(x = 'age', hue = 'deposit', data = df, palette =("g", "r"), legend=False)
                st.pyplot(fig)
    
            if boxchoices21 == "Deposit selon leur statut marital" :
                fig = plt.figure()
                sns.countplot(x="marital", hue = 'deposit', data = df, palette =("g", "r"), legend=False)
                st.pyplot(fig)
    
            if boxchoices21 == "Deposit selon leur job" :
                fig = plt.figure(figsize=(20,10))
                sns.countplot(x="job", hue = 'deposit', data = df, palette =("g", "r"), legend=False)
                st.pyplot(fig)
        
        st.write("____________________________________")
    
    
    if option_submenu == 'Corrélations entre les variables':
        option_submenu4 = st.sidebar.selectbox('Sous-Menu', ("Matrice de corrélation", "Analyses et Tests statistiques des variables numeriques", "Analyses et Tests statistiques des variables quantitatives"))
        if option_submenu4 == 'Matrice de corrélation':
            st.subheader("Matrice de corrélation")
            cor = df[['age', 'balance', 'duration', 'campaign', 'pdays', 'previous']].corr()
            fig, ax = plt.subplots()
            sns.heatmap(cor, annot=True, ax=ax, cmap='rainbow')
            st.write(fig)
            st.write("Commentaires = blabla")
    
        if option_submenu4 == 'Analyses et Tests statistiques des variables numeriques':   
            pages=["Lien âge x deposit", "Lien balance x deposit", "Lien duration x deposit", "Lien campaign x deposit", "Lien previous x deposit"]
            page=st.sidebar.radio('Afficher', pages)
    
    
            if page == pages[0] :
                fig = plt.figure()
                sns.kdeplot(df[df['deposit'] == 'yes']['age'], label='Yes', color='blue');
                sns.kdeplot(df[df['deposit'] == 'no']['age'], label='No', color='red');
                plt.title('Distribution des âges par groupe yes/no de la variable deposit')
                plt.xlabel('Âge')
                plt.ylabel('Densité')
                st.write(fig)
        
                st.write("Test Statistique:")
                st.write("H0 : Il n'y a pas d'effet significatif de l'age sur la souscrition au Deposit")
                st.write("H1 : Il y a un effet significatif de l'age sur la souscrition au Deposit")
        
                import statsmodels.api
                result = statsmodels.formula.api.ols('age ~ deposit', data = df).fit()
                table = statsmodels.api.stats.anova_lm(result)
                st.write(table)
        
                st.write("P_value = 0.0002")
                st.write("On rejette H1 : PAS DE LIEN SIGNIFICATIF entre Age et Deposit")
        
        
        
    
            if page == pages[1] :
                fig = plt.figure()
                sns.kdeplot(df[df['deposit'] == 'yes']['balance'], label='Yes', color='blue');
                sns.kdeplot(df[df['deposit'] == 'no']['balance'], label='No', color='red');
                plt.title('Distribution de Balance par groupe yes/no de la variable deposit')
                plt.xlabel('Balance')
                plt.ylabel('Densité')
                st.write(fig)       
        
        
                st.write("Test Statistique:")
                st.write("H0 : Il n'y a pas d'effet significatif de balance sur la souscrition au Deposit")
                st.write("H1 : Il y a un effet significatif de balance sur la souscrition au Deposit")
        
                st.image("dilenesantos/stats_balance_deposit.png")
        
                st.write("P_value = 9.126568e-18")
                st.write("On rejette H0 : IL Y A UN LIEN SIGNIFICATIF entre Balance et Deposit")
        
    

            if page == pages[2] :
                fig = plt.figure()
                sns.kdeplot(df[df['deposit'] == 'yes']['duration'], label='Yes', color='blue');
                sns.kdeplot(df[df['deposit'] == 'no']['duration'], label='No', color='red');
                plt.title('Distribution de Duration par groupe yes/no de la variable Deposit')
                plt.xlabel('Duration')
                plt.ylabel('Densité')
                st.write(fig)
        
                st.write("Test Statistique:")
                st.write("H0 : Il n'y a pas d'effet significatif de duration sur la souscrition au Deposit")
                st.write("H1 : Il y a un effet significatif de duration sur la souscrition au Deposit")
        
                st.image("dilenesantos/stats_duration_deposit.png")

        
                st.write("P_value = 0")
                st.write("On rejette H0 : IL Y A UN LIEN SIGNIFICATIF entre Duration et Deposit")
        
    
            if page == pages[3] :
                fig = plt.figure()
                sns.kdeplot(df[df['deposit'] == 'yes']['campaign'], label='Yes', color='blue');
                sns.kdeplot(df[df['deposit'] == 'no']['campaign'], label='No', color='red');
                plt.title('Distribution de Campaign par groupe yes/no de la variable Deposit')
                plt.xlabel('Campaign')
                plt.ylabel('Densité')
                st.write(fig)
        
                st.write("Test Statistique:")
                st.write("H0 : Il n'y a pas d'effet significatif de campaign sur la souscrition au Deposit")
                st.write("H1 : Il y a un effet significatif de campaign la souscrition au Deposit")
        
                st.image("dilenesantos/stats_campaign_deposit.png")

        
                st.write("P_value = 4.831324e-42")
                st.write("On rejette H0 : IL Y A UN LIEN SIGNIFICATIF entre Campaign et Deposit")
    
            if page == pages[4] :
                fig = plt.figure()
                sns.kdeplot(df[df['deposit'] == 'yes']['previous'], label='Yes', color='blue');
                sns.kdeplot(df[df['deposit'] == 'no']['previous'], label='No', color='red');
                plt.title('Distribution de Previous par groupe yes/no de la variable Deposit')
                plt.xlabel('Previous')
                plt.ylabel('Densité')
                st.write(fig)
        
                st.write("Test Statistique:")
                st.write("H0 : Il n'y a pas d'effet significatif de previous sur la souscrition au Deposit")
                st.write("H1 : Il y a un effet significatif de previous sur la souscrition au Deposit")
        
                st.image("dilenesantos/stats_previous_deposit.png")

        
                st.write("P_value = 7.125338e-50")
                st.write("On rejette H0 : IL Y A UN LIEN SIGNIFICATIF entre Previous et Deposit")
    
            st.write("____________________________________")
    
        if option_submenu4 == 'Analyses et Tests statistiques des variables quantitatives': 
            st.subheader("Analyses et Tests statistiques des variables quantitatives")
            pages=["Lien job x deposit", "Lien marital x deposit", "Lien education x deposit", "Lien housing x deposit", "Lien poutcome x deposit"]
            page=st.sidebar.radio('Afficher', pages)
            
            if page == pages[0] :
                fig = plt.figure(figsize=(20,10))
                sns.countplot(x="job", hue = 'deposit', data = df, palette =("g", "r"), legend=False)
                st.pyplot(fig)
        
                st.write("Test Statistique:")
                st.write("H0 : Les variables Job et Deposit sont indépendantes")
                st.write("H1 : La variable Job n'est pas indépendante de la variable Deposit")
        
                from scipy.stats import chi2_contingency
                ct = pd.crosstab(df['job'], df['deposit'])
                result = chi2_contingency(ct)
                stat = result[0]
                p_value = result[1]
                st.write('Statistique: ', stat)
                st.write('P_value: ', p_value)
        
                st.write("On rejette H0 : Il y a une dépendance entre Job et Deposit")
        
    
            if page == pages[1] :
                fig = plt.figure()
                sns.countplot(x="marital", hue = 'deposit', data = df, palette =("g", "r"), legend=False)
                st.pyplot(fig)
        
        
                st.write("Test Statistique:")
                st.write("H0 : Les variables Marital et Deposit sont indépendantes")
                st.write("H1 : La variable Marital n'est pas indépendante de la variable Deposit")

                from scipy.stats import chi2_contingency
                ct = pd.crosstab(df['marital'], df['deposit'])
                result = chi2_contingency(ct)
                stat = result[0]
                p_value = result[1]
                st.write('Statistique: ', stat)
                st.write('P_value: ', p_value)
        
                st.write("On rejette H0 : Il y a une dépendance entre Marital et Deposit")

            if page == pages[2] :
                fig = plt.figure()
                sns.countplot(x="education", hue = 'deposit', data = df, palette =("g", "r"), legend=False)
                st.pyplot(fig)
        
        
                st.write("Test Statistique:")
                st.write("H0 : Les variables Education et Deposit sont indépendantes")
                st.write("H1 : La variable Education n'est pas indépendante de la variable Deposit")
        
                from scipy.stats import chi2_contingency
                ct = pd.crosstab(df['education'], df['deposit'])
                result = chi2_contingency(ct)
                stat = result[0]
                p_value = result[1]
                st.write('Statistique: ', stat)
                st.write('P_value: ', p_value)
        
                st.write("On rejette H0 : Il y a une dépendance entre Education et Deposit")
    
            if page == pages[3] :
                fig = plt.figure()
                sns.countplot(x="housing", hue = 'deposit', data = df, palette =("g", "r"), legend=False)
                st.pyplot(fig)
        
        
                st.write("Test Statistique:")
                st.write("H0 : Les variables Housing et Deposit sont indépendantes")
                st.write("H1 : La variable Housing n'est pas indépendante de la variable Deposit")
        
                from scipy.stats import chi2_contingency
                ct = pd.crosstab(df['housing'], df['deposit'])
                result = chi2_contingency(ct)
                stat = result[0]
                p_value = result[1]
                st.write('Statistique: ', stat)
                st.write('P_value: ', p_value)
        
                st.write("On rejette H0 : Il y a une dépendance entre Housing et Deposit")
    
            if page == pages[4] :
                fig = plt.figure()
                sns.countplot(x="poutcome", hue = 'deposit', data = df, palette =("g", "r"), legend=False)
                st.pyplot(fig)
        
        
                st.write("Test Statistique:")
                st.write("H0 : Les variables Poutcome et Deposit sont indépendantes")
                st.write("H1 : La variable Poutcome n'est pas indépendante de la variable Deposit")
        
                from scipy.stats import chi2_contingency
                ct = pd.crosstab(df['poutcome'], df['deposit'])
                result = chi2_contingency(ct)
                stat = result[0]
                p_value = result[1]
                st.write('Statistique: ', stat)
                st.write('P_value: ', p_value)
        
                st.write("On rejette H0 : Il y a une dépendance entre Poutcome et Deposit")

            st.write("____________________________________")
    
    if option_submenu == "Évolution de la variable deposit dans le temps":
        
        option_submenu2 = st.sidebar.selectbox('SOUS-MENU', ("Deposit x month", "Deposit x year", "Deposit x weekday", "Deposit x Month x Âge", "Deposit x Month x Balance", "Deposit x Month x Campaign", "Deposit x Month x Previous", "Deposit x Month x Pdays"))
                
        
        st.subheader("Analyse de l'évolution de la variable deposit dans le temps")
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

            
        if option_submenu2 == 'Deposit x month':
            fig = plt.figure(figsize=(30,15))
            sns.countplot(x='month_year', hue='deposit', data=df_order_month, palette =("g", "r"), legend=False)
            plt.title("Évolution de notre variable cible selon les mois")
            plt.legend()
            st.pyplot(fig)
        
    
        if option_submenu2 == 'Deposit x year' :
            fig = plt.figure(figsize=(30,10))
            sns.countplot(x='year', hue='deposit', data=df, palette =("g", "r"), legend=False)
            plt.title("Évolution de notre variable cible selon l'année")
            plt.legend()
            st.pyplot(fig)

        if option_submenu2 == 'Deposit x weekday':
            fig = plt.figure()
            sns.countplot(x="weekday", hue = 'deposit', data = df, palette =("g", "r"), legend=False)
            st.pyplot(fig)
        
    
        if option_submenu2 == 'Deposit x Month x Âge' :
            fig, ax = plt.subplots(1, 1, figsize=(30, 10))
            sns.lineplot(x="month_year", y="age", hue= "deposit", data= df_order_month, palette =("g", "r"), ax=ax, errorbar=None)
            plt.grid(True)
            st.pyplot(fig)
    
        if option_submenu2 == 'Deposit x Month x Balance' :
            fig, ax = plt.subplots(1, 1, figsize=(30, 10))
            sns.lineplot(x="month_year", y="balance", hue= "deposit", data= df_order_month, palette =("g", "r"), ax=ax, errorbar=None)
            plt.grid(True)
            st.pyplot(fig)
    
        if option_submenu2 == 'Deposit x Month x Campaign' :
            fig, ax = plt.subplots(1, 1, figsize=(30, 10))
            sns.lineplot(x="month_year", y="campaign", hue= "deposit", data= df_order_month, palette =("g", "r"), ax=ax, errorbar=None)
            plt.grid(True)
            st.pyplot(fig)
    
        if option_submenu2 == 'Deposit x Month x Previous' :
            fig, ax = plt.subplots(1, 1, figsize=(30, 10))
            sns.lineplot(x="month_year", y="previous", hue= "deposit", data= df_order_month, palette =("g", "r"), ax=ax, errorbar=None)
            plt.grid(True)
            st.pyplot(fig)
    
        if option_submenu2 == 'Deposit x Month x Pdays' :
            fig, ax = plt.subplots(1, 1, figsize=(30, 10))
            sns.lineplot(x="month_year", y="pdays", hue= "deposit", data= df_order_month, palette =("g", "r"), ax=ax, errorbar=None)
            plt.grid(True)
            st.pyplot(fig)
    


if selected == "Pre-processing":  
    st.title("PRÉ-PROCESSING")
    st.sidebar.title("MENU PRÉ-PROCESSING")  
    option_submenu3 = st.sidebar.selectbox('Sélection', ("TRAITEMENT AVANT TRAIN-TEST-SPLIT", "TRAITEMENT APRÈS TRAIN-TEST-SPLIT"))
        
        
    if option_submenu3 == 'TRAITEMENT AVANT TRAIN-TEST-SPLIT':
        pages=["Suppression de lignes", "Création de colonnes", "Suppression de colonnes", "Gestion des Unknowns"]
        page=st.sidebar.radio('Afficher', pages)        

        dffpre_pros = df.copy()
        dffpre_pros2 = df.copy()
   
        if page == pages[0] :            
            st.subheader("Filtre sur la colonne 'age'")
            st.write("Notre analyse univariée a montré des valeurs extrêmes au dessus de 75 ans, aussi nous retirons ces lignes de notre dataset")
            dffpre_pros = dffpre_pros[dffpre_pros['age'] < 75]
            count_age_sup = df[df['age'] > 74.5].shape[0]
            st.write("Résultat = nombre de lignes concernées:", count_age_sup)
            
            st.subheader("Filtre sur la colonne 'balance'")
            st.write("Notre analyse univariée a montré des valeurs extrêmes de la variable balance pour les valeurs inférieures à -2257 et les valeurs supérieures à 4087, aussi nous décidons de retirer ces lignes de notre dataset")
            dffpre_pros = dffpre_pros.loc[dffpre_pros["balance"] > -2257]
            dffpre_pros = dffpre_pros.loc[dffpre_pros["balance"] < 4087]
            count_balance_sup = df[df['balance'] < -2257].shape[0]
            count_balance_inf = df[df['balance'] > 4087].shape[0]
            total_balance_count = count_balance_sup + count_balance_inf
            st.write("Résultat = nombre de lignes concernées:", total_balance_count)
            
            st.subheader("Filtre sur la colonne 'campaign'")
            st.write("Notre analyse univariée a montré des valeurs extrêmes de la variable campaign pour les valeurs supérieures à 6,  nous décidons de retirer ces lignes de notre dataset")
            dffpre_pros = dffpre_pros.loc[dffpre_pros["campaign"] < 6]
            count_campaign_sup = df[df['campaign'] > 6].shape[0]
            st.write("Résultat = nombre de lignes concernées:", count_campaign_sup)
            
            st.subheader("Filtre sur la colonne 'previous'")
            st.write("Notre analyse univariée a montré des valeurs extrêmes de la variable previous pour les valeurs supérieures à 2.5 : nous décidons de retirer ces lignes de notre dataset")
            dffpre_pros = dffpre_pros.loc[dffpre_pros["previous"] < 2.5]
            count_previous_sup = df[df['previous'] > 2.5].shape[0]
            st.write("Résultat = nombre de lignes concernées:", count_previous_sup)
            
            st.write("____________________________________")

            st.subheader("Résultat:")
            count_sup_lignes = df.shape[0] - dffpre_pros.shape[0]
            st.write("Nombre total de lignes supprimées de notre dataset = ", count_sup_lignes)
            nb_lignes = dffpre_pros.shape[0]
            st.write("Notre dataset filtré compte désormais ", nb_lignes, "lignes.")

        if page == pages[1] :   
            st.subheader("Création de la colonne 'Client_Category'")
            st.write("Afin de pouvoir classifier les clients selon la colonne pdays, nous décidons de créer à partir de 'pdays' une nouvelle colonne 'Client_Category' qui ")

            
            bins = [-2, -1, 180, 855]
            labels = ['Prospect', 'Reached-6M', 'Reached+6M']
            dffpre_pros['Client_Category_M'] = pd.cut(dffpre_pros['pdays'], bins=bins, labels=labels)
            # Transformation de 'Client_Category' en type 'objet'
            dffpre_pros['Client_Category_M'] = dffpre_pros['Client_Category_M'].astype('object')
                        
            # Affichage du nouveau dataset
            st.dataframe(dffpre_pros.head(10))
            
            st.subheader("Création de la colonne 'weekday'")
            st.write("Pour créer la colonne weekday, nous devons passer par plusieurs étapes : ")
            st.write("- ajouter une colonne 'year' : les données du dataset sont datées du juin 2014 ainsi nous pouvons déduire que les mois allant de juin à décembre correspondent à l'année 2023 et que les mois allant de janvier à mai correspondent à l'année 2014")
            st.write("- ajouter une colonne date : grâce à la colonne mois, day et year")
            st.write("- nous pouvons alors créer la colonne weekday grâce à la fonction 'dt.weekday'")
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
            
        
        if page == pages[2] :
            st.subheader("Suppressions de colonnes")
        
            st.write("- La colonne contact comprend bla blabla , nous décidons donc de la supprimer")             
            st.write("- Puisque nous avons créé la colonne Client_Category à partir de la colonne 'pdays', nous pouvons supprimer la colonne 'pdays'") 
            st.write("- Puisque nous avons créé la colonne weeday à partir de la colonne 'date', nous pouvons supprimer la colonne 'day' ainsi que la colonne date")     
            st.write("- Enfin, nous nous pouvons supprimer la colonne 'year' puisqu'elle n'apporte aucune valeur - en effet ...blablabla")

                        
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


        if page == pages[3] : 
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
            
            st.write("Nous nous occuperons du remplacement de ces NAns par la suite, une fois le jeu de donnée séparé en jeu d'entraînement et de test. En effet...blabla bla expliquer pourquoi on le fait après le train test split")
            

    if option_submenu3 == 'TRAITEMENT APRÈS TRAIN-TEST-SPLIT':
        pages=["Séparation train test", "Traitement des valeurs manquantes", "Standardisation des variables", "Encodage"]
        page=st.sidebar.radio('Afficher', pages)
         
        if page == pages[0] :
            st.subheader("Séparation train test")
            st.write("Nous appliquons un ratio de 80/20 pour notre train test split, soit 80% des données en Train et 20% en Test.")
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

            st.write("Affichage de X_train :")
            colonnes_count = X_train_pre_pros2.shape[1]
            nb_lignes = X_train_pre_pros2.shape[0]
            st.write("Le dataframe X_train compte :", colonnes_count, "colonnes et", nb_lignes, "lignes.")
            st.dataframe(X_train_pre_pros2.head())
                
            st.write("Affichage de X_test :")
            colonnes_count = X_test_pre_pros2.shape[1]
            nb_lignes = X_test_pre_pros2.shape[0]
            st.write("Le dataframe X_test compte :", colonnes_count, "colonnes et", nb_lignes, "lignes.")
            st.dataframe(X_test_pre_pros2.head())
                
        if page == pages[1] :    
            st.subheader("Traitement des valeurs manquantes")
            st.write("Pour la colonne job, on remplace les Nans par le mode de la variable.")
            st.write("S'agissant des colonnes 'education' et 'poutcome', puisque le nombre de Nans est plus élevé, nous avons décidé de les remplacer en utilisant la méthode de remplissage par propagation : chaque Nan est remplacé par la valeur de la ligne suivante (pour la dernière ligne on utilise le Mode de la variable).") 
            st.write("On applique ce process à X_train et X_test.")

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


            st.write("Vérification sur X_train, reste-t-il des Nans ?")
            st.dataframe(X_train_pre_pros2.isna().sum())
                
            st.write("Vérification sur X_test, reste-t-il des Nans ?")
            st.dataframe(X_test_pre_pros2.isna().sum())

                
        if page == pages[2] :    
            st.write("Standardisation des variables")
            st.write("On standardise les variables quantitatives à l'aide de la fonction StandardScaler.")
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
                
            st.write("Vérification sur X_train, les données quantitatives sont-elles bien standardisées ?")
            st.dataframe(X_train_pre_pros2.head())
                
            st.write("Vérification sur X_test, les données quantitatives sont-elles bien standardisées ?")
            st.dataframe(X_test_pre_pros2.head())

                
        if page == pages[3] :    
            st.subheader("Encodage")
            st.write("On encode la variable cible avec le Label Encoder.")
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
                
            st.write("Pour les variables qualitatives 'default', 'housing' et 'loan', on encode avec le One Hot Encoder")
            # Encodage des variables explicatives de type 'objet'
            oneh = OneHotEncoder(drop = 'first', sparse_output = False)
            cat1 = ['default', 'housing','loan']
            X_train_pre_pros2.loc[:, cat1] = oneh.fit_transform(X_train_pre_pros2[cat1])
            X_test_pre_pros2.loc[:, cat1] = oneh.transform(X_test_pre_pros2[cat1])

            X_train_pre_pros2[cat1] = X_train_pre_pros2[cat1].astype('int64')
            X_test_pre_pros2[cat1] = X_test_pre_pros2[cat1].astype('int64')
                
            st.write("Pour les variables ordinales 'education' et et 'Client_Category', on remplace les modalités par des nombres en gardant l'ordre initial.")
                
            # 'education' est une variable catégorielle ordinale, remplacer les modalités de la variable par des nombres, en gardant l'ordre initial
            X_train_pre_pros2['education'] = X_train_pre_pros2['education'].replace(['primary', 'secondary', 'tertiary'], [0, 1, 2])
            X_test_pre_pros2['education'] = X_test_pre_pros2['education'].replace(['primary', 'secondary', 'tertiary'], [0, 1, 2])

            # 'Client_Category_M' est une variable catégorielle ordinale, remplacer les modalités de la variable par des nombres, en gardant l'ordre initial
            X_train_pre_pros2['Client_Category_M'] = X_train_pre_pros2['Client_Category_M'].replace(['Prospect', 'Reached-6M', 'Reached+6M'], [0, 1, 2])
            X_test_pre_pros2['Client_Category_M'] = X_test_pre_pros2['Client_Category_M'].replace(['Prospect', 'Reached-6M', 'Reached+6M'], [0, 1, 2])


            st.write("Pour les autres variables catégorielles à plus de 2 modalités on applique le get dummies à la fois à X_train et X_test.")
                
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
            st.write("Dimensions du jeu d'entraînement:",X_train_pre_pros2.shape)
                
            st.write("Dataframe final X_test : ")
            st.dataframe(X_test_pre_pros2.head())
            st.write("Dimensions du jeu de test:",X_test_pre_pros2.shape)
                
            st.write("Vérification sur X_train, reste-t-il des Nans ?")
            st.dataframe(X_train_pre_pros2.isna().sum())
                
            st.write("Vérification sur X_test, reste-t-il des Nans ?")
            st.dataframe(X_test_pre_pros2.isna().sum())
                

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
        st.write("On va effectuer deux modélisations, l'une en conservant la variable Duration et l'autre sans la variable Duration : on explique pourquoi blablabla.")
        st.write("Pour chaque modélisation, avec ou sans Duration, nous analysons les scores des principaux modèles de classification d'abord dans paramètres afin de sélectionner les 3 meilleurs modèles, puis sur ces 3 modèles nous effectuons des recherches d'hyperparamètres à l'aide de la fonction GridSearchCV afin de sélectionner le modèle le plus performant possible.")
        st.write("Enfin sur le meilleur modèle trouvé, nous effectuons une analyse SHAP afin d'interpréter les décisions prises par le modèle dans la détection des clients susceptibles de Deposit YES")
        st.write("Lister les modèles de Classification testés...")
                 
    if page == pages[1] : 
        #AVEC DURATION
        submenu_modelisation = st.radio("", ("Scores modèles sans paramètres", "Hyperparamètres et choix du modèle"), horizontal=True)
        if submenu_modelisation == "Scores modèles sans paramètres" :
            st.subheader("Scores modèles sans paramètres")
            st.write("On affiche le tableau des résultats des modèles :")
            st.dataframe(results_sans_param)
                
            st.write("Graphique :")
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
            st.write("blabla GridSearchCV ....")
            
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
            st.write("RandomForestClassifier(class_weight= 'balanced', max_depth=20, max_features='sqrt',min_samples_leaf=2, min_samples_split=10, n_estimators= 200, random_state=42)")
                
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
            st.write("Rapport de classification du modèle")
            st.table(report_df)

            # Création de la matrice de confusion sous forme de DataFrame
            st.write("Matrice de confusion du modèle")
            table_rf = pd.crosstab(y_test, y_pred, rownames=["Réalité"], colnames=["Prédiction"])
            st.dataframe(table_rf)

            st.write("PARTIE SHAP SUR MODÈLE SÉLECTIONNÉ")
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
            st.write(shap_values_RF_carolle.shape)
            
            fig = plt.figure()
            shap.summary_plot(shap_values_RF_carolle[:,:,1], X_test)  
            st.pyplot(fig)
            
            fig = plt.figure()
            explanation_RF_carolle = shap.Explanation(values=shap_values_RF_carolle,
                                 data=X_test.values, # Assumant que  X_test est un DataFrame
                                 feature_names=X_test.columns)
            shap.plots.bar(explanation_RF_carolle[:,:,1])
            st.pyplot(fig)


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


    if page == pages[2] :
        #SANS DURATION
        submenu_modelisation2 = st.selectbox("Menu", ("Scores modèles sans paramètres", "Hyperparamètres et choix du modèle"))
    
        if submenu_modelisation2 == "Scores modèles sans paramètres" :
            st.subheader("Scores modèles sans paramètres")
            
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

            st.write("On affiche le tableau des résultats des modèles sans paramètres avec load modèles :")
            st.dataframe(df_results_SD_sans_param)
            
            st.write("Graphique :")
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
    

        if submenu_modelisation2 == "Hyperparamètres et choix du modèle" :
            st.write("Recherche d'hyperparamètres et choix du modèle")
            st.write("blabla GridSearchCV ....")
            
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
                "XGBOOST_1": joblib.load("dilenesantos/XGBOOST_1_model_SD_TOP_4_hyperparam.pkl"),
                "XGBOOST_2": joblib.load("dilenesantos/XGBOOST_2_model_SD_TOP_4_hyperparam.pkl"),
                "XGBOOST_3": joblib.load("dilenesantos/XGBOOST_3_model_SD_TOP_4_hyperparam.pkl"),
                "XGBOOST_TESTDIL": joblib.load("dilenesantos/XGBOOST_TESTDIL_model_SD_TOP_4_hyperparam.pkl")
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
            
            st.write("On affiche le tableau des résultats des modèles hyper paramétrés loadés Joblib :")
            st.dataframe(df_results_SD_TOP_4_hyperparam)
            
            st.write("Graphique avec load:")
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
     
                    
            st.subheader("Modèle sélectionné 1")
            st.write("XGBOOST_1_model_SD_TOP_4_hyperparam.pkl avec les hyperparamètres ci-dessous affiche la meilleure performance en termes de Recall, aussi nous choisisons de poursuivre notre modélisation avec ce modèle")
            st.write("XGBClassifier(gamma=0.05,colsample_bytree=0.9, learning_rate=0.39, max_depth=6, min_child_weight=1.29, n_estimators=34, reg_alpha=1.29, reg_lambda=1.9, scale_pos_weight=2.6, subsample=0.99, random_state=42)")
                
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

            # Affichage global du rapport sous forme de tableau
            st.write("Rapport de classification du modèle")
            st.table(report_df_1)

            # Création de la matrice de confusion sous forme de DataFrame
            st.write("Matrice de confusion du modèle")
            table_xgboost_1 = pd.crosstab(y_test_sd, y_pred_1, rownames=["Réalité"], colnames=["Prédiction"])
            st.dataframe(table_xgboost_1)
            

        

        
if selected == 'Interprétation':      
    st.sidebar.title("SOUS MENU INTERPRÉTATION")
    pages=["INTERPRÉTATION AVEC DURATION", "INTERPRÉTATION SANS DURATION"]
    page=st.sidebar.radio('AVEC ou SANS Duration', pages)

    if page == pages[0] : 
        st.subheader("Interpréation SHAP avec la colonne Duration")
        submenu_interpretation = st.selectbox("Menu", ("Summary plot", "Bar plot poids des variables", "Analyses des variables catégorielles", "Dependence plots"))
        
        if submenu_interpretation == "Summary plot" : 
            # Affichage des visualisations SHAP
            st.subheader("Summary plot")
            
        if submenu_interpretation == "Bar plot poids des variables" :
            st.subheader("Poids des variables dans le modèle")
            st.write("blablabla")

        if submenu_interpretation == "Analyses des variables catégorielles" :
            st.subheader("Zoom sur les variables catégorielles")
            st.write("blablabla")

        if submenu_interpretation == "Dependence plots" :
            st.subheader("Dépendences plots & Analyses")
            st.write("blablabla")
            
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

        submenu_interpretation = st.selectbox("", ("ANALYSE GLOBALE", "ANALYSE DES VARIABLES LES PLUS IMPORTANTES", "TESTS"))

        if submenu_interpretation == "ANALYSE GLOBALE" :
            submenu_global = st.radio("", ("Summary plot", "Bar plot"), horizontal=True)
            
            if submenu_global == "Summary plot" :
                st.subheader("Summary plot")
                fig = plt.figure()
                shap.summary_plot(shap_values_XGBOOST_1, X_test_sd)  
                st.pyplot(fig)
                
            if submenu_global == "Bar plot" :
                st.subheader("Bar plot")
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
                
                st.subheader("Choix des 5 variables les plus importantes")
                st.write("1. HOUSING : détention ou non d’un prêt immobilier")
                st.write("2. ÂGE")
                st.write("3. BALANCE : solde bancaire du client")
                st.write("4. PREVIOUS : nombre de contacts effectués avant la campagne avec le client")
                st.write("5. CAMPAIGN : nombre de contacts effectués avec le client pendant la campagne (dernier contact inclus)")
                

        if submenu_interpretation == "ANALYSE DES VARIABLES LES PLUS IMPORTANTES" :
            submenu_local = st.radio("", ("HOUSING", "ÂGE", "BALANCE", "PREVIOUS", "CAMPAIGN", "EDUCATION"), horizontal=True)
            shap_XGBOOST_1_VALUES = shap_values_XGBOOST_1.values
            X_test_original_figures = X_test_sd_original 
            
            if submenu_local == "HOUSING" :
                st.title("HOUSING : POIDS +0.27")
                st.subheader("IMPACT NÉGATIF DE HOUSING SUR LA CLASSE 1")
                st.write("Summary plot :")
                st.write("Shape of shap_values_XGBOOST_1:", shap_values_XGBOOST_1.shape)
                
                fig = plt.figure()
                shap.summary_plot(shap_values_XGBOOST_1[:, [X_test_sd.columns.get_loc("housing")]], 
                                  X_test_sd[["housing"]], 
                                  feature_names=["housing"], 
                                  show=True)
                st.pyplot(fig)
                

                st.write("blabla")

            if submenu_local == "ÂGE" :
                st.title("HOUSING : POIDS +0.25")
                st.subheader("IMPACT POSITIF DES TRANCHES D’ÂGES BASSES OU ÉLEVÉES")
                st.subheader("IMPACT NÉGATIF DES TRANCHES D’ÂGES MOYENNES")
                st.write("Summary plot :")
                fig = plt.figure()
                shap.summary_plot(shap_values_XGBOOST_1[:, [X_test_sd.columns.get_loc("age")]], 
                                  X_test_sd[["age"]], 
                                  feature_names=["age"], 
                                  show=True)
                st.pyplot(fig)
                st.write("blabla")         

                st.subheader("Dependence plot") 
                feature_name = "age"
                
                shap.dependence_plot(feature_name, shap_values=shap_XGBOOST_1_VALUES, features=X_test_original_figures, interaction_index=feature_name, show=False)
                plt.axhline(0, color='red', linestyle='--', linewidth=1) 
                fig = plt.gcf()          
                st.pyplot(fig)       
                plt.close() 
                # Exécuter le graphique de dépendance

                
                st.write("blabla") 


            if submenu_local == "BALANCE" :
                st.title("BALANCE : POIDS +0.20")
                st.subheader("IMPACT POSITIF DE BALANCE SUR LA CLASSE 1")
                st.write("Summary plot :")
                fig = plt.figure()
                shap.summary_plot(shap_values_XGBOOST_1[:, [X_test_sd.columns.get_loc("balance")]], 
                                  X_test_sd[["balance"]], 
                                  feature_names=["balance"], 
                                  show=True)
                st.pyplot(fig)

                st.write("blabla")         

                #GRAPHIQUE DEPENDENCE PLOT
                st.subheader("Dependence plot") 
                feature_name = "balance"
                
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
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()
                
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
                
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()
                
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
                
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()


            if submenu_local == "PREVIOUS" :
                st.title("PREVIOUS : POIDS +0.14")
                st.subheader("IMPACT POSITIF DE PREVIOUS SUR LA CLASSE 1")
                st.write("Summary plot :")
                fig = plt.figure()
                shap.summary_plot(shap_values_XGBOOST_1[:, [X_test_sd.columns.get_loc("previous")]], 
                                  X_test_sd[["previous"]], 
                                  feature_names=["previous"], 
                                  show=True)
                st.pyplot(fig)
                
                st.title("SHAP Dependence Plot")       
                feature_name = "previous"
                
                shap.dependence_plot(feature_name, shap_values=shap_XGBOOST_1_VALUES, features=X_test_original_figures, interaction_index=feature_name, show=False)
                plt.axhline(0, color='red', linestyle='--', linewidth=1) 
                fig = plt.gcf()          
                st.pyplot(fig)       
                plt.close() 
                        
            if submenu_local == "CAMPAIGN" :
                st.title("PREVIOUS : POIDS +0.14")
                st.subheader("IMPACT POSITIF DE PREVIOUS SUR LA CLASSE 1")
                st.write("Summary plot :")
                
                fig = plt.figure()
                shap.summary_plot(shap_values_XGBOOST_1[:, [X_test_sd.columns.get_loc("campaign")]], 
                                  X_test_sd[["campaign"]], 
                                  feature_names=["campaign"], 
                                  show=True)
                st.pyplot(fig)

                
                st.subheader("Dependence plot") 
                feature_name = "campaign"
                
                shap.dependence_plot(feature_name, shap_values=shap_XGBOOST_1_VALUES, features=X_test_original_figures, interaction_index=feature_name, show=False)
                plt.axhline(0, color='red', linestyle='--', linewidth=1) 
                fig = plt.gcf()          
                st.pyplot(fig)       
                plt.close() 


            if submenu_local == "EDUCATION" :
                st.title("EDUCATION : POIDS +0.09")
                st.subheader("IMPACT POSITIF OU NÉGATIF DE ÉDUCATION SUR LA CLASSE 1 ????")
                st.write("Summary plot :")
                fig = plt.figure()
                shap.summary_plot(shap_values_XGBOOST_1[:, [X_test_sd.columns.get_loc("education")]], 
                                  X_test_sd[["education"]], 
                                  feature_names=["education"], 
                                  show=True)
                st.pyplot(fig)
                st.write("blabla")         

                st.subheader("Dependence plot") 
                feature_name = "education"
                
                shap.dependence_plot(feature_name, shap_values=shap_XGBOOST_1_VALUES, features=X_test_original_figures, interaction_index=feature_name, show=False)
                plt.axhline(0, color='red', linestyle='--', linewidth=1) 
                fig = plt.gcf()          
                st.pyplot(fig)       
                plt.close() 
            
        
        if submenu_interpretation == "TESTS" :

            st.subheader("XGBOOST_2_model_SD_TOP_4_hyperparam.pkl")           
            #CODE À UTILISER UNE FOIS LES SHAP VALUES CHARGÉES
            shap_values_XGBOOST_2 = joblib.load("dilenesantos/shap_values_XGBOOST_2_SD_TOP_4_hyperparam.pkl")

            #CODE À UTILISER UNE FOIS LES SHAP VALUES CHARGÉES
            shap_values_XGBOOST_2 = joblib.load("dilenesantos/shap_values_XGBOOST_2_SD_TOP_4_hyperparam.pkl")

            fig = plt.figure()
            shap.summary_plot(shap_values_XGBOOST_2, X_test_sd)  
            st.pyplot(fig)
        
            explanation_XGBOOST_2 = shap.Explanation(values=shap_values_XGBOOST_2,
                                 data=X_test_sd.values, # Assumant que  X_test est un DataFrame
                                 feature_names=X_test_sd.columns)
            shap.plots.bar(explanation_XGBOOST_2)
            
            ### 1 CREATION D'UN EXPLANATION FILTRER SANS LES COLONNES POUR LESQUELLES NOUS ALLONS CALCULER LES MOYENNES

            #Étape 1 : Créer une liste des termes à exclure
            terms_to_exclude = ['month', 'weekday', 'job', 'poutcome', 'marital']

            #Étape 2 : Filtrer les colonnes qui ne contiennent pas les termes à exclure
            filtered_columns = [col for col in X_test_sd.columns if not any(term in col for term in terms_to_exclude)]

            #Étape 3 : Identifier les indices correspondants dans X_test_sd
            filtered_indices = [X_test_sd.columns.get_loc(col) for col in filtered_columns]
            shap_values_filtered_XGBOOST_2 = shap_values_XGBOOST_2[:, filtered_indices]

            # Étape 4 : On créé un nouvel Explanation avec les colonnes filtrées
            explanation_filtered_XGBOOST_2 = shap.Explanation(values=shap_values_filtered_XGBOOST_2,
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
            mean_shap_month = get_mean_shap_values(month_columns, shap_values_XGBOOST_2)
            mean_shap_weekday = get_mean_shap_values(weekday_columns, shap_values_XGBOOST_2)
            mean_shap_poutcome = get_mean_shap_values(poutcome_columns, shap_values_XGBOOST_2)
            mean_shap_job = get_mean_shap_values(job_columns, shap_values_XGBOOST_2)
            mean_shap_marital = get_mean_shap_values(marital_columns, shap_values_XGBOOST_2)

            # Étape 3 : On combine les différentes moyennes et on les nomme
            combined_values_XGBOOST_2 = [np.mean(mean_shap_month),
                                        np.mean(mean_shap_weekday),
                                        np.mean(mean_shap_poutcome),
                                        np.mean(mean_shap_job),
                                        np.mean(mean_shap_marital)]

            combined_feature_names_XGBOOST2 = ['Mean SHAP Value for Month Features',
                                            'Mean SHAP Value for Weekday Features',
                                            'Mean SHAP Value for Poutcome Features',
                                            'Mean SHAP Value for Job Features',
                                            'Mean SHAP Value for Marital Features']

            # Étape 4 : On crée un nouvel Explanation avec les valeurs combinées
            explanation_combined_XGBOOST_2 = shap.Explanation(values=combined_values_XGBOOST_2,
                                                            data=np.array([[np.nan]] * len(combined_values_XGBOOST_2)),
                                                            feature_names=combined_feature_names_XGBOOST2)

            ###3 ON COMBINE LES 2 EXPLANTATION PRÉCÉDEMMENT CRÉÉS

            #Étape 1 : On récupére les nombre de lignes de explanation_filtered et on reshape explanation_combined pour avoir le même nombre de lignes
            num_samples = explanation_filtered_XGBOOST_2.values.shape[0]
            combined_values_reshaped__XGBOOST_2 = np.repeat(np.array(explanation_combined_XGBOOST_2.values)[:, np.newaxis], num_samples, axis=1).T

            #Étape 2: On concatenate les 2 explanations
            combined_values_XGBOOST_2 = np.concatenate([explanation_filtered_XGBOOST_2.values, combined_values_reshaped__XGBOOST_2], axis=1)

            #Étape 3: On combine le nom des colonnes provenant des 2 explanations
            combined_feature_names_XGBOOST_2 = (explanation_filtered_XGBOOST_2.feature_names + explanation_combined_XGBOOST_2.feature_names)

            #Étape 4: On créé un nouveau explanation avec les valeurs concatnées dans combined_values
            explanation_combined_new_XGBOOST_2 = shap.Explanation(values=combined_values_XGBOOST_2,data=np.array([[np.nan]] * combined_values_XGBOOST_2.shape[0]),feature_names=combined_feature_names_XGBOOST_2)

            fig = plt.figure(figsize=(10, 6))
            shap.plots.bar(explanation_combined_new_XGBOOST_2, max_display=len(explanation_combined_new_XGBOOST_2.feature_names))
            st.pyplot(fig)
        
            
            st.subheader("XGBOOST_3_model_SD_TOP_4_hyperparam.pkl")           
            #CODE À UTILISER UNE FOIS LES SHAP VALUES CHARGÉES
            shap_values_XGBOOST_3 = joblib.load("dilenesantos/shap_values_XGBOOST_3_SD_TOP_4_hyperparam.pkl")

            fig = plt.figure()
            shap.summary_plot(shap_values_XGBOOST_3, X_test_sd)  
            st.pyplot(fig)    
        
            explanation_XGBOOST_3 = shap.Explanation(values=shap_values_XGBOOST_3,
                                 data=X_test_sd.values, # Assumant que  X_test est un DataFrame
                                 feature_names=X_test_sd.columns)
            shap.plots.bar(explanation_XGBOOST_3)
            
            ### 1 CREATION D'UN EXPLANATION FILTRER SANS LES COLONNES POUR LESQUELLES NOUS ALLONS CALCULER LES MOYENNES

            #Étape 1 : Créer une liste des termes à exclure
            terms_to_exclude = ['month', 'weekday', 'job', 'poutcome', 'marital']

            #Étape 2 : Filtrer les colonnes qui ne contiennent pas les termes à exclure
            filtered_columns = [col for col in X_test_sd.columns if not any(term in col for term in terms_to_exclude)]

            #Étape 3 : Identifier les indices correspondants dans X_test_sd
            filtered_indices = [X_test_sd.columns.get_loc(col) for col in filtered_columns]
            shap_values_filtered_XGBOOST_3 = shap_values_XGBOOST_3[:, filtered_indices]

            # Étape 4 : On créé un nouvel Explanation avec les colonnes filtrées
            explanation_filtered_XGBOOST_3 = shap.Explanation(values=shap_values_filtered_XGBOOST_3,
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
            mean_shap_month = get_mean_shap_values(month_columns, shap_values_XGBOOST_3)
            mean_shap_weekday = get_mean_shap_values(weekday_columns, shap_values_XGBOOST_3)
            mean_shap_poutcome = get_mean_shap_values(poutcome_columns, shap_values_XGBOOST_3)
            mean_shap_job = get_mean_shap_values(job_columns, shap_values_XGBOOST_3)
            mean_shap_marital = get_mean_shap_values(marital_columns, shap_values_XGBOOST_3)

            # Étape 3 : On combine les différentes moyennes et on les nomme
            combined_values_XGBOOST_3 = [np.mean(mean_shap_month),
                                        np.mean(mean_shap_weekday),
                                        np.mean(mean_shap_poutcome),
                                        np.mean(mean_shap_job),
                                        np.mean(mean_shap_marital)]

            combined_feature_names_XGBOOST3 = ['Mean SHAP Value for Month Features',
                                            'Mean SHAP Value for Weekday Features',
                                            'Mean SHAP Value for Poutcome Features',
                                            'Mean SHAP Value for Job Features',
                                            'Mean SHAP Value for Marital Features']

            # Étape 4 : On crée un nouvel Explanation avec les valeurs combinées
            explanation_combined_XGBOOST_3 = shap.Explanation(values=combined_values_XGBOOST_3,
                                                            data=np.array([[np.nan]] * len(combined_values_XGBOOST_3)),
                                                            feature_names=combined_feature_names_XGBOOST3)

            ###3 ON COMBINE LES 2 EXPLANTATION PRÉCÉDEMMENT CRÉÉS

            #Étape 1 : On récupére les nombre de lignes de explanation_filtered et on reshape explanation_combined pour avoir le même nombre de lignes
            num_samples = explanation_filtered_XGBOOST_3.values.shape[0]
            combined_values_reshaped__XGBOOST_3 = np.repeat(np.array(explanation_combined_XGBOOST_3.values)[:, np.newaxis], num_samples, axis=1).T

            #Étape 2: On concatenate les 2 explanations
            combined_values_XGBOOST_3 = np.concatenate([explanation_filtered_XGBOOST_3.values, combined_values_reshaped__XGBOOST_3], axis=1)

            #Étape 3: On combine le nom des colonnes provenant des 2 explanations
            combined_feature_names_XGBOOST_3 = (explanation_filtered_XGBOOST_3.feature_names + explanation_combined_XGBOOST_3.feature_names)

            #Étape 4: On créé un nouveau explanation avec les valeurs concatnées dans combined_values
            explanation_combined_new_XGBOOST_3 = shap.Explanation(values=combined_values_XGBOOST_3,data=np.array([[np.nan]] * combined_values_XGBOOST_3.shape[0]),feature_names=combined_feature_names_XGBOOST_3)

            fig = plt.figure(figsize=(10, 6))
            shap.plots.bar(explanation_combined_new_XGBOOST_3, max_display=len(explanation_combined_new_XGBOOST_3.feature_names))
            st.pyplot(fig)

           

if selected == "TEST PRED SCORES": 
        
    #code python SANS DURATION
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
    
    st.title("NOUVEAU DATAFRAME DÉDIÉ À L'OUTIL DE PRÉDICTIONS ")
    st.header("Structure du nouveau dataframe") 
    st.write("Explications pourquoi utiliser ce dataframe avec uniquement les colonnes AGE / BALANCE / HOUSING / EDUCATION / PREVIOUS")
    st.write("BALBLABLA")
    st.dataframe(dff_TEST)


    st.subheader("Scores sans paramètres") 
    
    #RÉSULTAT DES MODÈLES SANS PARAMÈTRES
 
    #CODE CHARGÉ UNE FOIS POUR SAUVEGARDE DES MODÈLES SUR JOBLIB
    # Initialisation des classifiers
    #classifiers_pred_df =  classifiers = {
        #"Random Forest": RandomForestClassifier(random_state=42),
        #"Logistic Regression": LogisticRegression(random_state=42),
        #"Decision Tree": DecisionTreeClassifier(random_state=42),
        #"KNN": KNeighborsClassifier(),
        #"AdaBoost": AdaBoostClassifier(random_state=42),
        #"Bagging": BaggingClassifier(random_state=42),
        #"SVM": svm.SVC(random_state=42),
        #"XGBOOST": XGBClassifier(random_state=42),
    #}
    
    #FICHIER RANDOM FOREST DE +25MO, on doit le compresser pour streamlit
    #Charger votre modèle
    #filename = "Random_Forest_model_PRED_sans_parametres.pkl"
    #model = joblib.load(filename)

    # Sauvegarder le modèle avec compression de niveau 9
    #joblib.dump(model, "Random_Forest_model_PRED_sans_parametres_compressed.pkl", compress=5)
  

    #Résultats des modèles
    #results_DF_PRED_sans_parametres = {}

    #Fonction pour entraîner et sauvegarder un modèle
    #def train_and_save_model_pred(model_name, clf, X_train_o, y_train_o):
        #filename = f"{model_name.replace(' ', '_')}_model_PRED_sans_parametres.pkl"  # Nom du fichier
        #try:
            #Charger le modèle si le fichier existe déjà
            #trained_clf = joblib.load(filename)
        #except FileNotFoundError:
            #Entraîner et sauvegarder le modèle
            #clf.fit(X_train_o, y_train_o)
            #joblib.dump(clf, filename)
            #trained_clf = clf
        #return trained_clf

    #Boucle pour entraîner ou charger les modèles
    #for name, clf in classifiers_pred_df.items():
        #Entraîner ou charger le modèle
        #trained_clf = train_and_save_model_pred(name, clf, X_train_o, y_train_o)
        #y_pred = trained_clf.predict(X_test_o)
            
        #Calculer les métriques
        #accuracy = accuracy_score(y_test_o, y_pred)
        #f1 = f1_score(y_test_o, y_pred)
        #precision = precision_score(y_test_o, y_pred)
        #recall = recall_score(y_test_o, y_pred)
            
        #Stocker les résultats
        #results_DF_PRED_sans_parametres[name] = {
            #"Accuracy": accuracy,
            #"F1 Score": f1,
            #"Precision": precision,
            #"Recall": recall,
        #}
        
  
    #COMME ON A ENREGISTRÉ LES MODÈLES, VOICI LE NOUVEAU CODE À UTILISER : 
    # Liste des modèles enregistrés et leurs fichiers correspondants
    model_files_pred = {
        "Random Forest": "dilenesantos/Random_Forest_model_PRED_sans_parametres_compressed.pkl",
        "Logistic Regression": "dilenesantos/Logistic_Regression_model_PRED_sans_parametres.pkl",
        "Decision Tree": "dilenesantos/Decision_Tree_model_PRED_sans_parametres.pkl",
        "KNN": "dilenesantos/KNN_model_PRED_sans_parametres.pkl",
        "AdaBoost": "dilenesantos/AdaBoost_model_PRED_sans_parametres.pkl",
        "Bagging": "dilenesantos/Bagging_model_PRED_sans_parametres.pkl",
        "SVM": "dilenesantos/SVM_model_PRED_sans_parametres.pkl",
        "XGBOOST": "dilenesantos/XGBOOST_model_PRED_sans_parametres.pkl",
    }


    # Résultats des modèles
    results_DF_PRED_sans_parametres = {}

    # Boucle pour charger les modèles et calculer les métriques
    for name, file_path in model_files_pred.items():
        # Charger le modèle sauvegardé
        trained_clf = joblib.load(file_path)
        
        # Faire des prédictions
        y_pred = trained_clf.predict(X_test_o)

        # Calculer les métriques
        accuracy = accuracy_score(y_test_o, y_pred)
        f1 = f1_score(y_test_o, y_pred)
        precision = precision_score(y_test_o, y_pred)
        recall = recall_score(y_test_o, y_pred)

        # Stocker les résultats
        results_DF_PRED_sans_parametres[name] = {
            "Accuracy": accuracy,
            "F1 Score": f1,
            "Precision": precision,
            "Recall": recall,
        }

    #Conversion des résultats en DataFrame
    df_results_DF_PRED_sans_parametres = pd.DataFrame(results_DF_PRED_sans_parametres).T
    df_results_DF_PRED_sans_parametres.columns = ['Accuracy', 'F1 Score', 'Precision', 'Recall']
    df_results_DF_PRED_sans_parametres = df_results_DF_PRED_sans_parametres.sort_values(by="Recall", ascending=False)
    
    melted_df_results_DF_PRED_sans_parametres = df_results_DF_PRED_sans_parametres.reset_index().melt(id_vars="index", var_name="Metric", value_name="Score")
    melted_df_results_DF_PRED_sans_parametres.rename(columns={"index": "Classifier"}, inplace=True)    

    st.dataframe(df_results_DF_PRED_sans_parametres)

    st.subheader("Scores avec paramètres") 
    st.write("À faire via VS code pour sauvegarde - fichier bank_copie")
    #RÉSULTATS DES MODÈLES AVEC PARAMÈTRES

    #CODE CHARGÉ UNE FOIS POUR SAUVEGARDE DES MODÈLES SUR JOBLIB
    # Initialisation des classifiers
    #classifiers_pred_df =  classifiers = {
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
        #"XGBOOST_fatou": XGBClassifier(colsample_bytree=0.8, gamma= 5, learning_rate= 0.1, max_depth= 5, n_estimators= 100, subsample= 0.8, random_state=42),
        
        #"Random Forest GridSearch2": RandomForestClassifier(class_weight= 'balanced', max_depth = None, max_features = 'sqrt', min_samples_leaf= 2, min_samples_split= 15, n_estimators = 200, random_state=42),
        #"SVM GridSearch2": svm.SVC (C = 1, class_weight = 'balanced', gamma = 'scale', kernel ='rbf', random_state=42),
        #"XGBOOST GridSearch2": XGBClassifier (colsample_bytree = 0.8, gamma = 5, learning_rate = 0.05, max_depth = 17, min_child_weight = 1, n_estimators = 200, subsample = 0.8, random_state=42),

        #"Random Forest SD": RandomForestClassifier(class_weight='balanced', max_depth=8,  max_features='log2', min_samples_leaf=250, min_samples_split=300, n_estimators=400, random_state=42),
        #"Decision Tree SD": DecisionTreeClassifier(class_weight='balanced', criterion='entropy', max_depth=5,  max_features=None, min_samples_leaf=100, min_samples_split=2, random_state=42),
        #"SVM SD" : svm.SVC(C=0.01, class_weight='balanced', gamma='scale', kernel='linear',random_state=42),
        #"XGBOOST_1 SD" : XGBClassifier(gamma=0.05,colsample_bytree=0.9, learning_rate=0.39, max_depth=6, min_child_weight=1.29, n_estimators=34, reg_alpha=1.29, reg_lambda=1.9, scale_pos_weight=2.6, subsample=0.99, random_state=42),
        #"XGBOOST_2 SD" : XGBClassifier(gamma=0.05,colsample_bytree=0.88, learning_rate=0.39, max_depth=6, min_child_weight=1.2, n_estimators=30, reg_alpha=1.2, reg_lambda=1.8, scale_pos_weight=2.56, subsample=0.99, random_state=42),
        #"XGBOOST_3 SD" : XGBClassifier(gamma=0.05,colsample_bytree=0.83, learning_rate=0.37, max_depth=6,  min_child_weight=1.2, n_estimators=30, reg_alpha=1.2, reg_lambda=1.7, scale_pos_weight=2.46, subsample=0.99, random_state=42),
        #"XGBOOST_TESTDIL SD" : XGBClassifier(gamma=0.05,colsample_bytree=0.83, learning_rate=0.37, max_depth=6,  min_child_weight=1.2, n_estimators=30, reg_alpha=1.2, reg_lambda=1.7, scale_pos_weight=2.46, subsample=0.99, random_state=42)

    #}


    #Résultats des modèles
    #results_DF_PRED_avec_parametres = {}

    #Fonction pour entraîner et sauvegarder un modèle
    #def train_and_save_model_pred(model_name, clf, X_train_o, y_train_o):
        #filename = f"{model_name.replace(' ', '_')}_model_PRED_AVEC_parametres.pkl"  # Nom du fichier
        #try:
            #Charger le modèle si le fichier existe déjà
            #trained_clf = joblib.load(filename)
        #except FileNotFoundError:
            #Entraîner et sauvegarder le modèle
            #clf.fit(X_train_o, y_train_o)
            #joblib.dump(clf, filename)
            #trained_clf = clf
        #return trained_clf

    #Boucle pour entraîner et charger les modèles
    #for name, clf in classifiers_pred_df.items():
        #Entraîner ou charger le modèle
        #trained_clf = train_and_save_model_pred(name, clf, X_train_o, y_train_o)
        #y_pred = trained_clf.predict(X_test_o)
            
        #Calculer les métriques
        #accuracy = accuracy_score(y_test_o, y_pred)
        #f1 = f1_score(y_test_o, y_pred)
        #precision = precision_score(y_test_o, y_pred)
        #recall = recall_score(y_test_o, y_pred)
            
        #Stocker les résultats
        #results_DF_PRED_avec_parametres[name] = {
            #"Accuracy": accuracy,
            #"F1 Score": f1,
            #"Precision": precision,
            #"Recall": recall,
        #}
        
    #COMME ON A ENREGISTRÉ LES MODÈLES, VOICI LE NOUVEAU CODE À UTILISER : 
    # Liste des modèles enregistrés et leurs fichiers correspondants
    model_files_pred_param = {
        "RF_dounia": "dilenesantos/RF_dounia_model_PRED_AVEC_parametres.pkl",
        "RF_fatou": "dilenesantos/RF_fatou_model_PRED_AVEC_parametres.pkl",
        "RF_carolle": "dilenesantos/RF_carolle_model_PRED_AVEC_parametres.pkl",
        "SVM_dounia": "dilenesantos/SVM_dounia_model_PRED_AVEC_parametres.pkl",
        "SVM_dilene": "dilenesantos/SVM_dilene_model_PRED_AVEC_parametres.pkl",
        "SVM_fatou": "dilenesantos/SVM_fatou_model_PRED_AVEC_parametres.pkl",
        "SVM_carolle": "dilenesantos/SVM_carolle_model_PRED_AVEC_parametres.pkl",
        "XGBOOST_dounia": "dilenesantos/XGBOOST_dounia_model_PRED_AVEC_parametres.pkl",
        "XGBOOST_dilene": "dilenesantos/XGBOOST_dilene_model_PRED_AVEC_parametres.pkl",
        "XGBOOST_carolle": "dilenesantos/XGBOOST_carolle_model_PRED_AVEC_parametres.pkl",
        "XGBOOST_fatou": "dilenesantos/XGBOOST_fatou_model_PRED_AVEC_parametres.pkl",
        
        "Random Forest GridSearch2": "dilenesantos/Random_Forest_GridSearch2_model_PRED_AVEC_parametres.pkl",
        "SVM GridSearch2": "dilenesantos/SVM_GridSearch2_model_PRED_AVEC_parametres.pkl",
        "XGBOOST GridSearch2": "dilenesantos/XGBOOST_GridSearch2_model_PRED_AVEC_parametres.pkl",

        "Random Forest SD": "dilenesantos/Random_Forest_SD_model_PRED_AVEC_parametres.pkl",
        "Decision Tree SD": "dilenesantos/Decision_Tree_SD_model_PRED_AVEC_parametres.pkl",
        "SVM SD" : "dilenesantos/SVM_SD_model_PRED_AVEC_parametres.pkl",
        "XGBOOST_1 SD" : "dilenesantos/XGBOOST_1_SD_model_PRED_AVEC_parametres.pkl",
        "XGBOOST_2 SD" : "dilenesantos/XGBOOST_2_SD_model_PRED_AVEC_parametres.pkl",
        "XGBOOST_3 SD" : "dilenesantos/XGBOOST_3_SD_model_PRED_AVEC_parametres.pkl",
        "XGBOOST_TESTDIL SD" : "dilenesantos/XGBOOST_TESTDIL_SD_model_PRED_AVEC_parametres.pkl"

    }

    # Résultats des modèles
    results_DF_PRED_avec_parametres = {}

    # Boucle pour charger les modèles et calculer les métriques
    for name, file_path in model_files_pred_param.items():
        # Charger le modèle sauvegardé
        trained_clf = joblib.load(file_path)
        
        # Faire des prédictions
        y_pred = trained_clf.predict(X_test_o)

        # Calculer les métriques
        accuracy = accuracy_score(y_test_o, y_pred)
        f1 = f1_score(y_test_o, y_pred)
        precision = precision_score(y_test_o, y_pred)
        recall = recall_score(y_test_o, y_pred)

        # Stocker les résultats
        results_DF_PRED_avec_parametres[name] = {
            "Accuracy": accuracy,
            "F1 Score": f1,
            "Precision": precision,
            "Recall": recall,
        }

    #Conversion des résultats en DataFrame
    df_results_DF_PRED_avec_parametres = pd.DataFrame(results_DF_PRED_avec_parametres).T
    df_results_DF_PRED_avec_parametres.columns = ['Accuracy', 'F1 Score', 'Precision', 'Recall']
    df_results_DF_PRED_avec_parametres = df_results_DF_PRED_avec_parametres.sort_values(by="Recall", ascending=False)
    
    melted_df_results_DF_PRED_avec_parametres = df_results_DF_PRED_avec_parametres.reset_index().melt(id_vars="index", var_name="Metric", value_name="Score")
    melted_df_results_DF_PRED_avec_parametres.rename(columns={"index": "Classifier"}, inplace=True)    

    st.dataframe(df_results_DF_PRED_avec_parametres)
    
    st.title("Scores des top 3 modèles") 
    st.subheader("Modèle top score 1 = XGBOOST_2 SD")
    st.write("dilenesantos/XGBOOST_2_SD_model_PRED_AVEC_parametres.pkl")
    st.write("XGBOOST_2 SD : XGBClassifier(gamma=0.05,colsample_bytree=0.88, learning_rate=0.39, max_depth=6, min_child_weight=1.2, n_estimators=30, reg_alpha=1.2, reg_lambda=1.8, scale_pos_weight=2.56, subsample=0.99, random_state=42)")
    # Chargement du modèle enregistré
    filename_1 = "dilenesantos/XGBOOST_2_SD_model_PRED_AVEC_parametres.pkl"
    model_XGBOOST_2_SD = joblib.load(filename_1)

    # Prédictions sur les données test
    y_pred_1 = model_XGBOOST_2_SD.predict(X_test_o)

    # Calcul des métriques pour chaque classe
    report_1 = classification_report(y_test_o, y_pred_1, target_names=["Classe 0", "Classe 1"], output_dict=True)

    # Conversion du rapport en DataFrame pour affichage en tableau
    report_df_1 = pd.DataFrame(report_1).T

    # Arrondi des valeurs à 4 décimales pour un affichage propre
    report_df_1 = report_df_1.round(4)

    # Suppression des colonnes inutiles si besoin
    report_df_1 = report_df_1.drop(columns=["support"])

    # Affichage global du rapport sous forme de tableau
    st.write("Rapport de classification du modèle")
    st.table(report_df_1)

    # Création de la matrice de confusion sous forme de DataFrame
    st.write("Matrice de confusion du modèle")
    table_xgboost_1 = pd.crosstab(y_test_o, y_pred_1, rownames=["Réalité"], colnames=["Prédiction"])
    st.dataframe(table_xgboost_1)

    #SHAP
    #PARTIE DU CODE À VIRER UNE FOIS LES SHAP VALUES CHARGÉES
    st.write("Shap XGBOOST_2")

    #Chargement du modèle XGBOOST_2 déjà enregistré
    #filename_XGBOOST_2 = "XGBOOST_2_SD_model_PRED_AVEC_parametres.pkl"
    #model_XGBOOST_2_SD_model_PRED_AVEC_parametres = joblib.load(filename_XGBOOST_2)

    #Chargement des données pour shap 
    #data_to_explain_XGBOOST_2 = X_test_o  # Remplacez par vos données

    #Création de l'explainer SHAP pour XGBOOST_2
    #explainer_XGBOOST_2 = shap.TreeExplainer(model_XGBOOST_2_SD_model_PRED_AVEC_parametres)

    #Calcul des shap values
    #shap_values_XGBOOST_2_SD_model_PRED_AVEC_parametres = explainer_XGBOOST_2(data_to_explain_XGBOOST_2)

    #Sauvegarder des shap values avec joblib
    #joblib.dump(shap_values_XGBOOST_2_SD_model_PRED_AVEC_parametres, "shap_values_XGBOOST_2_SD_model_PRED_AVEC_parametres")

    #CODE À UTILISER UNE FOIS LES SHAP VALUES CHARGÉES
    shap_values_XGBOOST_2_PRED = joblib.load("dilenesantos/shap_values_XGBOOST_2_SD_model_PRED_AVEC_parametres")

    fig = plt.figure()
    shap.summary_plot(shap_values_XGBOOST_2_PRED, X_test_o)  
    st.pyplot(fig)
            
    fig = plt.figure()
    explanation_XGBOOST_2_PRED = shap.Explanation(values=shap_values_XGBOOST_2_PRED,
                            data=X_test_o.values, # Assumant que  X_test est un DataFrame
                            feature_names=X_test_o.columns)
    shap.plots.bar(explanation_XGBOOST_2_PRED)
    st.pyplot(fig)    
    
    st.subheader("Modèle top score 2 = XGBOOST_1 SD")
    st.write("dilenesantos/XGBOOST_1_SD_model_PRED_AVEC_parametres.pkl")
    st.write("XGBOOST_1 SD : XGBClassifier(gamma=0.05,colsample_bytree=0.9, learning_rate=0.39, max_depth=6, min_child_weight=1.29, n_estimators=34, reg_alpha=1.29, reg_lambda=1.9, scale_pos_weight=2.6, subsample=0.99, random_state=42)")
    # Chargement du modèle enregistré
    filename_2 = "dilenesantos/XGBOOST_1_SD_model_PRED_AVEC_parametres.pkl"
    model_XGBOOST_1_SD = joblib.load(filename_2)

    # Prédictions sur les données test
    y_pred_2 = model_XGBOOST_1_SD.predict(X_test_o)

    # Calcul des métriques pour chaque classe
    report_2 = classification_report(y_test_o, y_pred_2, target_names=["Classe 0", "Classe 1"], output_dict=True)

    # Conversion du rapport en DataFrame pour affichage en tableau
    report_df_2 = pd.DataFrame(report_2).T

    # Arrondi des valeurs à 4 décimales pour un affichage propre
    report_df_2 = report_df_2.round(4)

    # Suppression des colonnes inutiles si besoin
    report_df_2 = report_df_2.drop(columns=["support"])

    # Affichage global du rapport sous forme de tableau
    st.write("Rapport de classification du modèle")
    st.table(report_df_2)

    # Création de la matrice de confusion sous forme de DataFrame
    st.write("Matrice de confusion du modèle")
    table_xgboost_2 = pd.crosstab(y_test_o, y_pred_2, rownames=["Réalité"], colnames=["Prédiction"])
    st.dataframe(table_xgboost_2)
    
    st.write("Shap XGBOOST_1")
    #filename_XGBOOST_1 = "XGBOOST_1_SD_model_PRED_AVEC_parametres.pkl"
    #model_XGBOOST_1_SD_model_PRED_AVEC_parametres = joblib.load(filename_XGBOOST_1)

    #Chargement des données pour shap 
    #data_to_explain_XGBOOST_1 = X_test_o  # Remplacez par vos données

    #Création de l'explainer SHAP pour XGBOOST_1
    #explainer_XGBOOST_1 = shap.TreeExplainer(model_XGBOOST_1_SD_model_PRED_AVEC_parametres)

    #Calcul des shap values
    #shap_values_XGBOOST_1_SD_model_PRED_AVEC_parametres = explainer_XGBOOST_1(data_to_explain_XGBOOST_1)

    #Sauvegarder des shap values avec joblib
    #joblib.dump(shap_values_XGBOOST_1_SD_model_PRED_AVEC_parametres, "shap_values_XGBOOST_1_SD_model_PRED_AVEC_parametres")

    #CODE À UTILISER UNE FOIS LES SHAP VALUES CHARGÉES
    shap_values_XGBOOST_1_PRED = joblib.load("dilenesantos/shap_values_XGBOOST_1_SD_model_PRED_AVEC_parametres")

    fig = plt.figure()
    shap.summary_plot(shap_values_XGBOOST_1_PRED, X_test_o)  
    st.pyplot(fig)
            
    fig = plt.figure()
    explanation_XGBOOST_1_PRED = shap.Explanation(values=shap_values_XGBOOST_1_PRED,
                            data=X_test_o.values, # Assumant que  X_test est un DataFrame
                            feature_names=X_test_o.columns)
    shap.plots.bar(explanation_XGBOOST_1_PRED)
    st.pyplot(fig) 
    
    st.subheader("Modèle top score 3 = XGBOOST_TESTDIL SD")
    st.write("dilenesantos/XGBOOST_TESTDIL_SD_model_PRED_AVEC_parametres.pkl")
    st.write("XGBOOST_TESTDIL SD : XGBClassifier(gamma=0.05,colsample_bytree=0.83, learning_rate=0.37, max_depth=6,  min_child_weight=1.2, n_estimators=30, reg_alpha=1.2, reg_lambda=1.7, scale_pos_weight=2.46, subsample=0.99, random_state=42)")
    # Chargement du modèle enregistré
    filename_3 = "dilenesantos/XGBOOST_TESTDIL_SD_model_PRED_AVEC_parametres.pkl"
    model_XGBOOST_TESTDIL_SD = joblib.load(filename_3)

    # Prédictions sur les données test
    y_pred_3 = model_XGBOOST_TESTDIL_SD.predict(X_test_o)

    # Calcul des métriques pour chaque classe
    report_3 = classification_report(y_test_o, y_pred_3, target_names=["Classe 0", "Classe 1"], output_dict=True)

    # Conversion du rapport en DataFrame pour affichage en tableau
    report_df_3 = pd.DataFrame(report_3).T

    # Arrondi des valeurs à 4 décimales pour un affichage propre
    report_df_3 = report_df_3.round(4)

    # Suppression des colonnes inutiles si besoin
    report_df_3 = report_df_3.drop(columns=["support"])

    # Affichage global du rapport sous forme de tableau
    st.write("Rapport de classification du modèle")
    st.table(report_df_3)

    # Création de la matrice de confusion sous forme de DataFrame
    st.write("Matrice de confusion du modèle")
    table_xgboost_3 = pd.crosstab(y_test_o, y_pred_3, rownames=["Réalité"], colnames=["Prédiction"])
    st.dataframe(table_xgboost_3)
            
    #SHAP
    #PARTIE DU CODE À VIRER UNE FOIS LES SHAP VALUES CHARGÉES
    st.write("Shap XGBOOST_3")

    #Chargement du modèle XGBOOST_3 déjà enregistré
    #filename_XGBOOST_3 = "XGBOOST_3_SD_model_PRED_AVEC_parametres.pkl"
    #model_XGBOOST_3_SD_model_PRED_AVEC_parametres = joblib.load(filename_XGBOOST_3)

    #Chargement des données pour shap 
    #data_to_explain_XGBOOST_3 = X_test_o  # Remplacez par vos données

    #Création de l'explainer SHAP pour XGBOOST_3
    #explainer_XGBOOST_3 = shap.TreeExplainer(model_XGBOOST_3_SD_model_PRED_AVEC_parametres)

    #Calcul des shap values
    #shap_values_XGBOOST_3_SD_model_PRED_AVEC_parametres = explainer_XGBOOST_3(data_to_explain_XGBOOST_3)

    #Sauvegarder des shap values avec joblib
    #joblib.dump(shap_values_XGBOOST_3_SD_model_PRED_AVEC_parametres, "shap_values_XGBOOST_3_SD_model_PRED_AVEC_parametres")

    #CODE À UTILISER UNE FOIS LES SHAP VALUES CHARGÉES
    shap_values_XGBOOST_3_PRED = joblib.load("dilenesantos/shap_values_XGBOOST_3_SD_model_PRED_AVEC_parametres")

    fig = plt.figure()
    shap.summary_plot(shap_values_XGBOOST_3_PRED, X_test_o)  
    st.pyplot(fig)
            
    fig = plt.figure()
    explanation_XGBOOST_3_PRED = shap.Explanation(values=shap_values_XGBOOST_3_PRED,
                            data=X_test_o.values, # Assumant que  X_test est un DataFrame
                            feature_names=X_test_o.columns)
    shap.plots.bar(explanation_XGBOOST_3_PRED)
    st.pyplot(fig)   

if selected == 'Outil Prédictif':    
    #code python SANS DURATION
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

    st.title("Outils de prédiction")
    submenu_predictions = st.radio("", ("Scores modèles & Hyperparamètres", "Prédictions"), horizontal=True)
    
    if submenu_predictions == "Scores modèles & Hyperparamètres" :
        st.write("Afficher df X_test")
        st.dataframe(X_test)
    
        st.write("Afficher df X_test_sd")
        st.dataframe(X_test_sd)
    
        st.write("Afficher df X_test_o")
        st.dataframe(X_test_o)    
    
        st.write("Afficher df X_train")
        st.dataframe(X_train)
    
        st.write("Afficher df X_train_sd")
        st.dataframe(X_train_sd)
    
        st.write("Afficher df X_train_o")
        st.dataframe(X_train_o)             
        
        st.title("SCORES DE CERTAINS MODÈLES SUR LE DATAFRAME FILTRE 4 COLONNES AGE EDUCATION BALANCE HOUSING PREVIOUS")
                        
        st.subheader("XGBOOST_1")
        st.write("Le modèle XGBOOST_1 avec les hyperparamètres ci-dessous affiche la meilleure performance en termes de Recall, aussi nous choisisons de poursuivre notre modélisation avec ce modèle")
        st.write("autre test= XGBClassifier(gamma=0.05,colsample_bytree=0.9, learning_rate=0.39, max_depth=6, min_child_weight=1.29, n_estimators=34, reg_alpha=1.29, reg_lambda=1.9, scale_pos_weight=2.6, subsample=0.99, random_state=42)")
        st.write("Affichons le rapport de classification de ce modèle")
        xgboost_best = XGBClassifier(gamma=0.05,colsample_bytree=0.9, learning_rate=0.39, max_depth=6, min_child_weight=1.29, n_estimators=34, reg_alpha=1.29, reg_lambda=1.9, scale_pos_weight=2.6, subsample=0.99, random_state=42)            
        xgboost_best.fit(X_train_o, y_train_o)
        score_train = xgboost_best.score(X_train_o, y_train_o)
        score_test = xgboost_best.score(X_test_o, y_test_o)
        y_pred = xgboost_best.predict(X_test_o)
        table_xgboost = pd.crosstab(y_test_o,y_pred, rownames=['Realité'], colnames=['Prédiction'])
        st.dataframe(table_xgboost)
        st.write("Classification report :")
        report_dict_xgboost = classification_report(y_test_o, y_pred, output_dict=True)
        # Convertir le dictionnaire en DataFrame
        report_df_xgboost = pd.DataFrame(report_dict_xgboost).T
        st.dataframe(report_df_xgboost)
                

        st.subheader("Modèle XGBOOST_2")
        st.write("Le modèle XGBOOST_2 avec les hyperparamètres ci-dessous affiche la meilleure performance en termes de Recall, aussi nous choisisons de poursuivre notre modélisation avec ce modèle")
        st.write("autre test= XGBClassifier(gamma=0.05,colsample_bytree=0.88, learning_rate=0.39, max_depth=6, min_child_weight=1.2, n_estimators=30, reg_alpha=1.2, reg_lambda=1.8, scale_pos_weight=2.56, subsample=0.99, random_state=42)")
        st.write("Affichons le rapport de classification de ce modèle")
        xgboost_best = XGBClassifier(gamma=0.05,colsample_bytree=0.88, learning_rate=0.39, max_depth=6, min_child_weight=1.2, n_estimators=30, reg_alpha=1.2, reg_lambda=1.8, scale_pos_weight=2.56, subsample=0.99, random_state=42)           
        xgboost_best.fit(X_train_o, y_train_o)
        score_train = xgboost_best.score(X_train_o, y_train_o)
        score_test = xgboost_best.score(X_test_o, y_test_o)
        y_pred = xgboost_best.predict(X_test_o)
        table_xgboost = pd.crosstab(y_test_o,y_pred, rownames=['Realité'], colnames=['Prédiction'])
        st.dataframe(table_xgboost)
        st.write("Classification report :")
        report_dict_xgboost = classification_report(y_test_o, y_pred, output_dict=True)
        # Convertir le dictionnaire en DataFrame
        report_df_xgboost = pd.DataFrame(report_dict_xgboost).T
        st.dataframe(report_df_xgboost)                 

        st.subheader("Modèle XGBOOST_3")
        st.write("Le modèle XGBOOST_3 avec les hyperparamètres ci-dessous affiche la meilleure performance en termes de Recall, aussi nous choisisons de poursuivre notre modélisation avec ce modèle")
        st.write("autre test= XGBClassifier(gamma=0.05,colsample_bytree=0.83, learning_rate=0.37, max_depth=6,  min_child_weight=1.2, n_estimators=30, reg_alpha=1.2, reg_lambda=1.7, scale_pos_weight=2.46, subsample=0.99, random_state=42)")
        st.write("Affichons le rapport de classification de ce modèle")
        xgboost_best = XGBClassifier(gamma=0.05,colsample_bytree=0.83, learning_rate=0.37, max_depth=6,  min_child_weight=1.2, n_estimators=30, reg_alpha=1.2, reg_lambda=1.7, scale_pos_weight=2.46, subsample=0.99, random_state=42)            
        xgboost_best.fit(X_train_o, y_train_o)
        score_train = xgboost_best.score(X_train_o, y_train_o)
        score_test = xgboost_best.score(X_test_o, y_test_o)
        y_pred = xgboost_best.predict(X_test_o)
        table_xgboost = pd.crosstab(y_test_o,y_pred, rownames=['Realité'], colnames=['Prédiction'])
        st.dataframe(table_xgboost)
        st.write("Classification report :")
        report_dict_xgboost = classification_report(y_test_o, y_pred, output_dict=True)
        # Convertir le dictionnaire en DataFrame
        report_df_xgboost = pd.DataFrame(report_dict_xgboost).T
        st.dataframe(report_df_xgboost)   


        st.title("Scores modèles sans param sur PRED DF avec joblib test")

       #RÉSULTAT DES MODÈLES SANS PARAMETRES
        models_pred_df = {
                "Random Forest": ("dilenesantos/Random_Forest_model_PRED_DF_sans_param.pkl"),
                "Logistic Regression": ("dilenesantos/Logistic_Regression_model_PRED_DF_sans_param.pkl"),
                "Decision Tree": ("dilenesantos/Decision_Tree_model_PRED_DF_sans_param.pkl"),
                "KNN": ("dilenesantos/KNN_model_PRED_DF_sans_param.pkl"),
                "AdaBoost": ("dilenesantos/AdaBoost_model_PRED_DF_sans_param.pkl"),
                "Bagging": ("dilenesantos/Bagging_model_PRED_DF_sans_param.pkl"),
                "SVM": ("dilenesantos/SVM_model_PRED_DF_sans_param.pkl"),
                "XGBOOST": ("dilenesantos/XGBOOST_model_PRED_DF_sans_param.pkl")
        }

        #Résultats des modèles
        results_sans_param_df_pred = {}

        #Boucle pour charger les modèles et calculer les métriques
        for name, file_path in models_pred_df.items():
            #Charger le modèle sauvegardé
            trained_clf = joblib.load(file_path)
            # Faire des prédictions
            y_pred = trained_clf.predict(X_test_o)
    
            # Calculer les métriques
            accuracy = accuracy_score(y_test_o, y_pred)
            f1 = f1_score(y_test_o, y_pred)
            precision = precision_score(y_test_o, y_pred)
            recall = recall_score(y_test_o, y_pred)
    
            # Stocker les résultats
            results_sans_param_df_pred[name] = {
                "Accuracy": accuracy,
                "F1 Score": f1,
                "Precision": precision,
                "Recall": recall,
            }

        # Conversion des résultats en DataFrame
        df_results_sans_param_df_pred = pd.DataFrame(results_sans_param_df_pred).T
        df_results_sans_param_df_pred.columns = ['Accuracy', 'F1 Score', 'Precision', 'Recall']
        df_results_sans_param_df_pred = df_results_sans_param_df_pred.sort_values(by="Recall", ascending=False)
            
        melted_df_results_sans_param_df_pred = df_results_sans_param_df_pred.reset_index().melt(id_vars="index", var_name="Metric", value_name="Score")
        melted_df_results_sans_param_df_pred.rename(columns={"index": "Classifier"}, inplace=True)

        st.dataframe(df_results_sans_param_df_pred)
        

    if submenu_predictions == "Prédictions" :
        
        st.title("Démonstration et application de notre modèle à votre cas")               

        st.subheader('Vos Informations sur le client')
        age = st.slider("Quel est l'âge du client ?", 17, 90, 1)
        education = st.selectbox("Quel est son niveau d'étude ?", ("tertiary", "secondary", "unknown", "primary"))
        balance = st.slider('Quel est le solde de son compte en banque ?', -3000, 10000, 1)
        housing = st.selectbox("As-t-il un crédit immobilier ?", ('yes', 'no'))
        previous = st.slider("Lors de la précédente campagne marketing, combien de fois avez-vous été appélé par votre banque", 0,6,1)
        
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
            pred_df['education'] = dff['education'].mode()[0]

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

        # Affichage du DataFrame après la standardisation
        st.write("Affichage de pred_df prêt pour la prédiction :")
        st.dataframe(pred_df)
        st.dataframe(dff_TEST)


        # Bouton pour lancer la prédiction
        prediction_button = st.button(label="Predict")
        
        xgboost_best_predict = XGBClassifier(gamma=0.05,colsample_bytree=0.83, learning_rate=0.37, max_depth=6,  min_child_weight=1.2, n_estimators=30, reg_alpha=1.2, reg_lambda=1.7, scale_pos_weight=1.46, subsample=0.99, random_state=42)            
        xgboost_best_predict.fit(X_train_o, y_train_o)
            
        # Prédiction
        if prediction_button:
            prediction = xgboost_best_predict.predict(pred_df)
            prediction_proba = xgboost_best_predict.predict_proba(pred_df)
            max_proba = np.max(prediction_proba[0]) * 100
            
            # Résultats
            if prediction[0] == 0:
                st.write(f"Prediction : {prediction[0]}")
                st.write(f"Niveau de confiance: {max_proba:.2f}%")
                st.write("Conclusion:", "\nCe client n'est pas susceptible de souscrire à un dépôt à terme.")
            else:
                st.write(f"Prediction : {prediction[0]}")
                st.write(f"Niveau de confiance: {max_proba:.2f}%")
                st.write("Conclusion:", "\nCe client est susceptible de souscrire à un dépôt à terme.")
                st.write("\n")
                st.write("Recommandations : ")
                st.write("- Durée d'appel : pour maximiser les chances de souscription au dépôt, il faudra veiller à rester le plus longtemps possible au téléphone avec ce client (idéalement au moins 6 minutes).")
                st.write("- Nombre de contacts pendant la campagne : il serait contre productif de le contacter plus d'une fois.")


       
if selected == 'TEST PRÉDICTIF':   

    #code python SANS DURATION
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

    st.title("Outils de prédiction")
    submenu_predictions = st.radio("", ("Scores modèles & Hyperparamètres", "Prédictions"), horizontal=True)
    
    if submenu_predictions == "Scores modèles & Hyperparamètres" :
 
        st.dataframe(X_dff_TEST)
        #RÉSULTAT DES MODÈLES SANS PARAMETRES
        # Initialisation des classifiers
        classifiers = {
            "Random Forest": RandomForestClassifier(random_state=42),
            "Logistic Regression": LogisticRegression(random_state=42),
            "Decision Tree": DecisionTreeClassifier(random_state=42),
            "KNN": KNeighborsClassifier(),
            "AdaBoost": AdaBoostClassifier(random_state=42),
            "Bagging": BaggingClassifier(random_state=42),
            "SVM": svm.SVC(random_state=42),
            "XGBOOST": XGBClassifier(random_state=42),
        }

        # Résultats des modèles
        results_sans_parametres = {}  # Affichage des résultats dans results

        for name, clf in classifiers.items():
            clf.fit(X_train_o, y_train_o)
            y_pred = clf.predict(X_test_o)

            accuracy = accuracy_score(y_test_o, y_pred)
            f1 = f1_score(y_test_o, y_pred)
            precision = precision_score(y_test_o, y_pred)
            recall = recall_score(y_test_o, y_pred)

            results_sans_parametres[name] = {
                "Accuracy": accuracy,
                "F1 Score": f1,
                "Precision": precision,
                "Recall": recall
            }

        # Conversion des résultats en DataFrame
        results_sans_param = pd.DataFrame(results_sans_parametres).T
        results_sans_param.columns = ['Accuracy', 'F1 Score', 'Precision', 'Recall']
        results_sans_param = results_sans_param.sort_values(by="Recall", ascending=False)

       #CLASSER LES RESULTATS DANS L'ORDRE DÉCROISSANT SELON LA COLONNE "Recall"
        results_sans_param = results_sans_param.sort_values(by='Recall', ascending=False)

        
        st.subheader("Scores modèles sans paramètres")
        st.dataframe(results_sans_param)
                    
        
        # dictionnaire avec les best modèles avec hyper paramètres trouvés AVEC DURATION !!!!
        classifiers_param_DURATION = {
            "Random Forest best": RandomForestClassifier(class_weight= 'balanced', max_depth=20, max_features='sqrt',min_samples_leaf=2, min_samples_split=10, n_estimators= 200, random_state=42),
            "Bagging": BaggingClassifier(random_state=42),
            "SVM best" : svm.SVC(C = 1, class_weight = 'balanced', gamma = 'scale', kernel ='rbf', random_state=42),
            "XGBOOST best" : XGBClassifier (colsample_bytree = 0.8, gamma = 5, learning_rate = 0.05, max_depth = 17, min_child_weight = 1, n_estimators = 200, subsample = 0.8, random_state=42)}

        results_avec_parametres_avDuration = {}  # Affichage des résultats dans results

        for name, clf in classifiers_param_DURATION.items():
            clf.fit(X_train_o, y_train_o)
            y_pred = clf.predict(X_test_o)

            accuracy = accuracy_score(y_test_o, y_pred)
            f1 = f1_score(y_test_o, y_pred)
            precision = precision_score(y_test_o, y_pred)
            recall = recall_score(y_test_o, y_pred)

            results_avec_parametres_avDuration[name] = {
                "Accuracy": accuracy,
                "F1 Score": f1,
                "Precision": precision,
                "Recall": recall
            }

        #créer un dataframe avec tous les résultats obtenus précédemment et pour tous les classifier
        results_best_param_DURATION = pd.DataFrame(results_avec_parametres_avDuration)
        results_best_param_DURATION = results_best_param_DURATION.T
        results_best_param_DURATION.columns = ['Accuracy', 'F1 Score', 'Precision', 'Recall']
                            
        #CLASSER LES RESULTATS DANS L'ORDRE DÉCROISSANT SELON LA COLONNE "Recall"
        results_best_param_DURATION = results_best_param_DURATION.sort_values(by='Recall', ascending=False)

        
        st.subheader("Scores modèles hyperamétrés avD :")
        st.dataframe(results_best_param_DURATION)
                    
            
        # dictionnaire avec les best modèles avec hyper paramètres trouvés SANS DURATION !!!!
        classifiers_param_sans_DURATION = {
            "Random Forest best param": RandomForestClassifier(class_weight='balanced', max_depth=8,  max_features='log2', min_samples_leaf=250, min_samples_split=300, n_estimators=400, random_state=42),
            "Decision Tree best param": DecisionTreeClassifier(class_weight='balanced', criterion='entropy', max_depth=5,  max_features=None, min_samples_leaf=100, min_samples_split=2, random_state=42),
            "Bagging": BaggingClassifier(random_state=42),
            "SVM best param" : svm.SVC(C=0.01, class_weight='balanced', gamma='scale', kernel='linear',random_state=42),
            "XGBOOST best param" : XGBClassifier(gamma=0.05,colsample_bytree=0.83, learning_rate=0.37, max_depth=6,  min_child_weight=1.2, n_estimators=30, reg_alpha=1.2, reg_lambda=1.7, scale_pos_weight=2.46, subsample=0.99, random_state=42)}
        results_avec_parametres_sansDuration = {}  # Affichage des résultats dans results

        for name, clf in classifiers_param_sans_DURATION.items():
            clf.fit(X_train_o, y_train_o)
            y_pred = clf.predict(X_test_o)

            accuracy = accuracy_score(y_test_o, y_pred)
            f1 = f1_score(y_test_o, y_pred)
            precision = precision_score(y_test_o, y_pred)
            recall = recall_score(y_test_o, y_pred)

            results_avec_parametres_sansDuration[name] = {
                "Accuracy": accuracy,
                "F1 Score": f1,
                "Precision": precision,
                "Recall": recall
            }
            
        #créer un dataframe avec tous les résultats obtenus précédemment et pour tous les classifier
        results_param_sans_duration = pd.DataFrame(results_avec_parametres_sansDuration)
        results_param_sans_duration = results_param_sans_duration.T
        results_param_sans_duration.columns = ['Accuracy', 'F1 Score', 'Precision', 'Recall']
                            
        #CLASSER LES RESULTATS DANS L'ORDRE DÉCROISSANT SELON LA COLONNE "Recall"
        results_param_sans_duration = results_param_sans_duration.sort_values(by='Recall', ascending=False)

        
        st.subheader("Scores modèles hyperparamétrés sans duration:")
        st.dataframe(results_param_sans_duration)
                    

        st.subheader("Modèle sélectionné")
        st.write("Le modèle Random Forest avec les hyperparamètres ci-dessous affiche la meilleure performance en termes de Recall, aussi nous choisisons de poursuivre notre modélisation avec ce modèle")
        st.write("RandomForestClassifier(class_weight= 'balanced', max_depth=20, max_features='sqrt',min_samples_leaf=2, min_samples_split=10, n_estimators= 200, random_state=42)")
                    
        st.write("Affichons le rapport de classification de ce modèle")
        rf_best = RandomForestClassifier(class_weight= 'balanced', max_depth=20, max_features='sqrt',min_samples_leaf=2, min_samples_split=10, n_estimators= 200, random_state=42)
        rf_best.fit(X_train_o, y_train_o)
        score_train = rf_best.score(X_train_o, y_train_o)
        score_test = rf_best.score(X_test_o, y_test_o)
        y_pred = rf_best.predict(X_test_o)
        table_rf = pd.crosstab(y_test_o,y_pred, rownames=['Realité'], colnames=['Prédiction'])
        st.dataframe(table_rf)
        st.write("Classification report :")
        report_dict = classification_report(y_test_o, y_pred, output_dict=True)
        # Convertir le dictionnaire en DataFrame
        report_df = pd.DataFrame(report_dict).T
        st.dataframe(report_df)

                        
        st.subheader("Modèle sélectionné")
        st.write("Le modèle XGBOOST avec les hyperparamètres ci-dessous affiche la meilleure performance en termes de Recall, aussi nous choisisons de poursuivre notre modélisation avec ce modèle")
        st.write("autre test= XGBClassifier(gamma=0.05,colsample_bytree=0.9, learning_rate=0.39, max_depth=6, min_child_weight=1.29, n_estimators=34, reg_alpha=1.29, reg_lambda=1.9, scale_pos_weight=2.6, subsample=0.99, random_state=42)")
        st.write("Affichons le rapport de classification de ce modèle")
        xgboost_best = XGBClassifier(gamma=0.05,colsample_bytree=0.9, learning_rate=0.39, max_depth=6, min_child_weight=1.29, n_estimators=34, reg_alpha=1.29, reg_lambda=1.9, scale_pos_weight=2.6, subsample=0.99, random_state=42)            
        xgboost_best.fit(X_train_o, y_train_o)
        score_train = xgboost_best.score(X_train_o, y_train_o)
        score_test = xgboost_best.score(X_test_o, y_test_o)
        y_pred = xgboost_best.predict(X_test_o)
        table_xgboost = pd.crosstab(y_test_o,y_pred, rownames=['Realité'], colnames=['Prédiction'])
        st.dataframe(table_xgboost)
        st.write("Classification report :")
        report_dict_xgboost = classification_report(y_test_o, y_pred, output_dict=True)
        # Convertir le dictionnaire en DataFrame
        report_df_xgboost = pd.DataFrame(report_dict_xgboost).T
        st.dataframe(report_df_xgboost)
                
        explainer = shap.TreeExplainer(xgboost_best)
        shap_values_xgboost_best = explainer.shap_values(X_test_o)
                
        fig = plt.figure()
        shap.summary_plot(shap_values_xgboost_best, X_test_o)  
        st.pyplot(fig)
                
        fig = plt.figure()
        explanation = shap.Explanation(values=shap_values_xgboost_best,
                                    data=X_test_o.values, # Assumant que  X_test est un DataFrame
                                    feature_names=X_test_o.columns)
        shap.plots.bar(explanation)
        st.pyplot(fig)                   


        st.subheader("Modèle XGBOOST 2")
        st.write("Le modèle XGBOOST avec les hyperparamètres ci-dessous affiche la meilleure performance en termes de Recall, aussi nous choisisons de poursuivre notre modélisation avec ce modèle")
        st.write("autre test= XGBClassifier(gamma=0.05,colsample_bytree=0.83, learning_rate=0.37, max_depth=6,  min_child_weight=1.2, n_estimators=30, reg_alpha=1.2, reg_lambda=1.7, scale_pos_weight=2.46, subsample=0.99, random_state=42)")
        st.write("Affichons le rapport de classification de ce modèle")
        xgboost_best = XGBClassifier(gamma=0.05,colsample_bytree=0.83, learning_rate=0.37, max_depth=6,  min_child_weight=1.2, n_estimators=30, reg_alpha=1.2, reg_lambda=1.7, scale_pos_weight=2.46, subsample=0.99, random_state=42)            
        xgboost_best.fit(X_train_o, y_train_o)
        score_train = xgboost_best.score(X_train_o, y_train_o)
        score_test = xgboost_best.score(X_test_o, y_test_o)
        y_pred = xgboost_best.predict(X_test_o)
        table_xgboost = pd.crosstab(y_test_o,y_pred, rownames=['Realité'], colnames=['Prédiction'])
        st.dataframe(table_xgboost)
        st.write("Classification report :")
        report_dict_xgboost = classification_report(y_test_o, y_pred, output_dict=True)
        # Convertir le dictionnaire en DataFrame
        report_df_xgboost = pd.DataFrame(report_dict_xgboost).T
        st.dataframe(report_df_xgboost)
                
        explainer = shap.TreeExplainer(xgboost_best)
        shap_values_xgboost_best = explainer.shap_values(X_test_o)
                
        fig = plt.figure()
        shap.summary_plot(shap_values_xgboost_best, X_test_o)  
        st.pyplot(fig)
                
        fig = plt.figure()
        explanation = shap.Explanation(values=shap_values_xgboost_best,
                                    data=X_test_o.values, # Assumant que  X_test est un DataFrame
                                    feature_names=X_test_o.columns)
        shap.plots.bar(explanation)
        st.pyplot(fig)                   


        st.subheader("RECHERCHE PARAMÈTRES XGBOOST 1")
        st.write("Le modèle XGBOOST avec les hyperparamètres ci-dessous affiche la meilleure performance en termes de Recall, aussi nous choisisons de poursuivre notre modélisation avec ce modèle")
        st.write("autre test= XGBClassifier(gamma=0.05,colsample_bytree=0.83, learning_rate=0.37, max_depth=6,  min_child_weight=1.2, n_estimators=30, reg_alpha=1.2, reg_lambda=1.7, scale_pos_weight=2.46, subsample=0.99, random_state=42)")
        st.write("Affichons le rapport de classification de ce modèle")
        xgboost_best = XGBClassifier(gamma=0.05,colsample_bytree=0.83, learning_rate=0.37, max_depth=6,  min_child_weight=1.2, n_estimators=30, reg_alpha=1.2, reg_lambda=1.7, scale_pos_weight=1.46, subsample=0.99, random_state=42)            
        xgboost_best.fit(X_train_o, y_train_o)
        score_train = xgboost_best.score(X_train_o, y_train_o)
        score_test = xgboost_best.score(X_test_o, y_test_o)
        y_pred = xgboost_best.predict(X_test_o)
        table_xgboost = pd.crosstab(y_test_o,y_pred, rownames=['Realité'], colnames=['Prédiction'])
        st.dataframe(table_xgboost)
        st.write("Classification report :")
        report_dict_xgboost = classification_report(y_test_o, y_pred, output_dict=True)
        # Convertir le dictionnaire en DataFrame
        report_df_xgboost = pd.DataFrame(report_dict_xgboost).T
        st.dataframe(report_df_xgboost)
                
        explainer = shap.TreeExplainer(xgboost_best)
        shap_values_xgboost_best = explainer.shap_values(X_test_o)
                
        fig = plt.figure()
        shap.summary_plot(shap_values_xgboost_best, X_test_o)  
        st.pyplot(fig)
                
        fig = plt.figure()
        explanation = shap.Explanation(values=shap_values_xgboost_best,
                                    data=X_test_o.values, # Assumant que  X_test est un DataFrame
                                    feature_names=X_test_o.columns)
        shap.plots.bar(explanation)
        st.pyplot(fig)                   


        st.subheader("RECHERCHE PARAMÈTRES XGBOOST 2")
        st.write("Le modèle XGBOOST avec les hyperparamètres ci-dessous affiche la meilleure performance en termes de Recall, aussi nous choisisons de poursuivre notre modélisation avec ce modèle")
        st.write("autre test= XGBClassifier(gamma=0.05,colsample_bytree=0.83, learning_rate=0.37, max_depth=6,  min_child_weight=1.2, n_estimators=30, reg_alpha=1.2, reg_lambda=1.7, scale_pos_weight=2.46, subsample=0.99, random_state=42)")
        st.write("Affichons le rapport de classification de ce modèle")
        xgboost_best_def = XGBClassifier(gamma=0.2,colsample_bytree=0.43, learning_rate=0.27, max_depth=6,  n_estimators=20, reg_alpha=3.2, reg_lambda=3.7, subsample=0.8, random_state=42)            
        xgboost_best_def.fit(X_train_o, y_train_o)
        score_train = xgboost_best_def.score(X_train_o, y_train_o)
        score_test = xgboost_best_def.score(X_test_o, y_test_o)
        y_pred = xgboost_best_def.predict(X_test_o)
        table_xgboost_best_def = pd.crosstab(y_test_o,y_pred, rownames=['Realité'], colnames=['Prédiction'])
        st.dataframe(table_xgboost_best_def)
        st.write("Classification report :")
        report_dict_xgboost = classification_report(y_test_o, y_pred, output_dict=True)
        # Convertir le dictionnaire en DataFrame
        report_df_xgboost = pd.DataFrame(report_dict_xgboost).T
        st.dataframe(report_df_xgboost)
                
        explainer = shap.TreeExplainer(xgboost_best)
        shap_values_xgboost_best = explainer.shap_values(X_test_o)
                
        fig = plt.figure()
        shap.summary_plot(shap_values_xgboost_best, X_test_o)  
        st.pyplot(fig)
                
        fig = plt.figure()
        explanation = shap.Explanation(values=shap_values_xgboost_best,
                                    data=X_test_o.values, # Assumant que  X_test est un DataFrame
                                    feature_names=X_test_o.columns)
        shap.plots.bar(explanation)
        st.pyplot(fig)   


    if submenu_predictions == "Prédictions" :
        
        st.title("Démonstration et application de notre modèle à votre cas")               

        st.subheader('Vos Informations sur le client')
        age = st.slider("Quel est l'âge du client ?", 17, 90, 1)
        education = st.selectbox("Quel est son niveau d'étude ?", ("tertiary", "secondary", "unknown", "primary"))
        balance = st.slider('Quel est le solde de son compte en banque ?', -3000, 10000, 1)
        housing = st.selectbox("As-t-il un crédit immobilier ?", ('yes', 'no'))
        previous = st.slider("Lors de la précédente campagne marketing, combien de fois avez-vous été appélé par votre banque", 0,6,1)
        
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
            pred_df['education'] = dff['education'].mode()[0]

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

        # Affichage du DataFrame après la standardisation
        st.write("Affichage de pred_df prêt pour la prédiction :")
        st.dataframe(pred_df)
        st.dataframe(dff_TEST)


        # Bouton pour lancer la prédiction
        prediction_button = st.button(label="Predict")
        
        filename = "dilenesantos/XGBOOST_2_SD_model_PRED_AVEC_parametres.pkl"
        model_XGBOOST_2 = joblib.load(filename)         
                
            
        # Prédiction
        if prediction_button:
            prediction = model_XGBOOST_2.predict(pred_df)
            prediction_proba = model_XGBOOST_2.predict_proba(pred_df)
            max_proba = np.max(prediction_proba[0]) * 100
            
            # Résultats
            if prediction[0] == 0:
                st.write(f"Prediction : {prediction[0]}")
                st.write(f"Niveau de confiance: {max_proba:.2f}%")
                st.write("Conclusion:", "\nCe client n'est pas susceptible de souscrire à un dépôt à terme.")
            else:
                st.write(f"Prediction : {prediction[0]}")
                st.write(f"Niveau de confiance: {max_proba:.2f}%")
                st.write("Conclusion:", "\nCe client est susceptible de souscrire à un dépôt à terme.")
                st.write("\n")
                st.write("Recommandations : ")
                st.write("- Durée d'appel : pour maximiser les chances de souscription au dépôt, il faudra veiller à rester le plus longtemps possible au téléphone avec ce client (idéalement au moins 6 minutes).")
                st.write("- Nombre de contacts pendant la campagne : il serait contre productif de le contacter plus d'une fois.")

if selected == 'PRED POUSSÉ':  

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

    #DATAFRAME PRED AVEC CAMPAIGN
    dff_TEST_campaign = df.copy()
    dff_TEST_campaign = dff_TEST_campaign[dff_TEST_campaign['age'] < 75]
    dff_TEST_campaign = dff_TEST_campaign.loc[dff_TEST_campaign["balance"] > -2257]
    dff_TEST_campaign = dff_TEST_campaign.loc[dff_TEST_campaign["balance"] < 4087]
    dff_TEST_campaign = dff_TEST_campaign.loc[dff_TEST_campaign["campaign"] < 6]
    dff_TEST_campaign = dff_TEST_campaign.loc[dff_TEST_campaign["previous"] < 2.5]
    dff_TEST_campaign = dff_TEST_campaign.drop('contact', axis = 1)
    
    dff_TEST_campaign = dff_TEST_campaign.drop('pdays', axis = 1)
    
    dff_TEST_campaign = dff_TEST_campaign.drop(['day'], axis=1)
    dff_TEST_campaign = dff_TEST_campaign.drop(['duration'], axis=1)
    dff_TEST_campaign = dff_TEST_campaign.drop(['job'], axis=1)
    dff_TEST_campaign = dff_TEST_campaign.drop(['default'], axis=1)
    dff_TEST_campaign = dff_TEST_campaign.drop(['month'], axis=1)
    dff_TEST_campaign = dff_TEST_campaign.drop(['poutcome'], axis=1)
    dff_TEST_campaign = dff_TEST_campaign.drop(['marital'], axis=1)
    dff_TEST_campaign = dff_TEST_campaign.drop(['loan'], axis=1)
    dff_TEST_campaign = dff_TEST_campaign.drop(['deposit'], axis=1)

    dff_TEST_campaign['education'] = dff_TEST_campaign['education'].replace('unknown', np.nan)
    

    st.title("Démonstration et application de notre modèle à votre cas")               

    st.subheader('Vos Informations sur le client')
    age = st.slider("Quel est l'âge du client ?", 17, 90, 1)
    education = st.selectbox("Quel est son niveau d'étude ?", ("tertiary", "secondary", "unknown", "primary"))
    balance = st.slider('Quel est le solde de son compte en banque ?', -3000, 10000, 1)
    housing = st.selectbox("As-t-il un crédit immobilier ?", ('yes', 'no'))
    previous = st.slider("Lors de la précédente campagne marketing, combien de fois avez-vous été appélé par votre banque", 0,6,1)
    
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

    
    # Bouton pour lancer la prédiction
    
    filename = "dilenesantos/XGBOOST_1_SD_model_PRED_AVEC_parametres.pkl"
    model_XGBOOST_1_SD_model_PRED_AVEC_parametres = joblib.load(filename)

    
    # Prédiction
    st.title("Prédiction")
    st.dataframe(pred_df)

    prediction = model_XGBOOST_1_SD_model_PRED_AVEC_parametres.predict(pred_df)
    prediction_proba = model_XGBOOST_1_SD_model_PRED_AVEC_parametres.predict_proba(pred_df)
    max_proba = np.max(prediction_proba[0]) * 100


    if prediction[0] == 0:
        st.write(f"Prediction : {prediction[0]}")
        st.write(f"Niveau de confiance: {max_proba:.2f}%")
        st.write("Conclusion:", "\nCe client n'est pas susceptible de souscrire à un dépôt à terme.")
    else:
        st.write(f"Prediction : {prediction[0]}")
        st.write(f"Niveau de confiance: {max_proba:.2f}%")
        st.write("Conclusion:", "\nCe client est susceptible de souscrire à un dépôt à terme.")
        st.write("\n")
        st.write("Recommandations : ")
        st.write("- Durée d'appel : pour maximiser les chances de souscription au dépôt, il faudra veiller à rester le plus longtemps possible au téléphone avec ce client (idéalement au moins 6 minutes).")
        st.write("- Nombre de contacts pendant la campagne : il serait contre productif de le contacter plus d'une fois.")
        

    if max_proba < 80:
        st.write("Le niveau de confiance est inférieur à 80%. 
        st.write("Vous pouvez affiner la prédiction en ajoutant une autre donnée à votre disposition concernant votre client.")
        
    
        # Afficher le sélecteur d'option pour le raffinement, incluant l'option pour ne rien ajouter
        option_to_add = st.radio("Choisir une variable à ajouter :", 
                                       ["None", "campaign", "loan", "marital", "poutcome", "job", "Client_Category_M"], horizontal=True)
        
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
                
                filename_LOAN = "dilenesantos/XGBOOST_1_SD_model_PRED_loan_XGBOOST_1.pkl"
                model_XGBOOST_1_SD_model_PRED_loan_XGBOOST_1 = joblib.load(filename_LOAN)
            
                
                # Prédiction avec le DataFrame optimisé
                prediction_opt_loan = model_XGBOOST_1_SD_model_PRED_loan_XGBOOST_1.predict(pred_df)
                prediction_proba_opt_loan = model_XGBOOST_1_SD_model_PRED_loan_XGBOOST_1.predict_proba(pred_df)
                max_proba_opt_loan = np.max(prediction_proba_opt_loan[0]) * 100
        
                # Affichage des résultats de l'affinage
                st.write(f"Prediction après affinage : {prediction_opt_loan[0]}")
                st.write(f"Niveau de confiance après affinage : {max_proba_opt_loan:.2f}%")
                if prediction_opt_loan[0] == 0:
                    st.write("Conclusion: Ce client n'est pas susceptible de souscrire à un dépôt à terme.")
                else:
                    st.write("Conclusion: Ce client est susceptible de souscrire à un dépôt à terme.")
                    st.write("\nRecommandations : ")
                    st.write("- Durée d'appel : Pour maximiser les chances de souscription au dépôt, il faudra veiller à rester le plus longtemps possible au téléphone avec ce client (idéalement au moins 6 minutes).")
                    st.write("- Nombre de contacts pendant la campagne : il serait contre-productif de le contacter plus d'une fois.")
        
        
            elif option_to_add == "campaign":
                campaign = st.slider("Combien de fois le client a-t-il été contacté durant la campagne ?", 0, 6, 1)
                pred_df['campaign'] = campaign
                st.write("Le client a été contacté ", campaign," au cours de la campagne")
    
                pred_df = pred_df.reindex(columns=dff_TEST_campaign.columns)
                
                # Étape 2 : Concaténer dff et pred_df
                # Concaténer les deux DataFrames dff et pred_df sur les colonnes numériques
                num_cols = ['age', 'balance','previous', 'campaign']
                
                # Utiliser un index unique pour pred_df, en le commençant après la dernière ligne de dff
                pred_df.index = range(dff_TEST_campaign.shape[0], dff_TEST_campaign.shape[0] + len(pred_df))
            
                combined_df_campaign = pd.concat([dff_TEST_campaign[num_cols], pred_df[num_cols]], axis=0)

                # Étape 3 : Standardisation des données numériques
                scaler = StandardScaler()
                combined_df_campaign[num_cols] = scaler.fit_transform(combined_df_campaign[num_cols])
    
                # Étape 4 : Séparer à nouveau pred_df des autres données
                # On récupère uniquement les lignes correspondant à pred_df en utilisant l'index spécifique
                pred_df[num_cols] = combined_df_campaign.loc[pred_df.index, num_cols]
            
                # Réinitialiser l'index de pred_df après la manipulation (facultatif)
                pred_df = pred_df.reset_index(drop=True)
   
                filename_CAMPAIGN = "dilenesantos/XGBOOST_1_SD_model_PRED_campaign_XGBOOST_1.pkl"
                model_XGBOOST_1_SD_model_PRED_campaign = joblib.load(filename_CAMPAIGN)         
                
                # Prédiction avec le DataFrame optimisé
                prediction_opt_campaign = model_XGBOOST_1_SD_model_PRED_campaign.predict(pred_df)
                prediction_proba_opt_campaign = model_XGBOOST_1_SD_model_PRED_campaign.predict_proba(pred_df)
                max_proba_opt_campaign = np.max(prediction_proba_opt_campaign[0]) * 100
        
                # Affichage des résultats de l'affinage
                st.write(f"Prediction après affinage : {prediction_opt_campaign[0]}")
                st.write(f"Niveau de confiance après affinage : {max_proba_opt_campaign:.2f}%")
                if prediction_opt_campaign[0] == 0:
                    st.write("Conclusion: Ce client n'est pas susceptible de souscrire à un dépôt à terme.")
                else:
                    st.write("Conclusion: Ce client est susceptible de souscrire à un dépôt à terme.")
                    st.write("\nRecommandations : ")
                    st.write("- Durée d'appel : Pour maximiser les chances de souscription au dépôt, il faudra veiller à rester le plus longtemps possible au téléphone avec ce client (idéalement au moins 6 minutes).")
                    st.write("- Nombre de contacts pendant la campagne : il serait contre-productif de le contacter plus d'une fois.")


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
          
                filename_MARITAL = "dilenesantos/XGBOOST_1_SD_model_PRED_marital_XGBOOST_1.pkl"
                model_XGBOOST_1_SD_model_PRED_marital = joblib.load(filename_MARITAL)         
                
                # Prédiction avec le DataFrame optimisé
                prediction_opt_marital = model_XGBOOST_1_SD_model_PRED_marital.predict(pred_df)
                prediction_proba_opt_marital = model_XGBOOST_1_SD_model_PRED_marital.predict_proba(pred_df)
                max_proba_opt_marital = np.max(prediction_proba_opt_marital[0]) * 100
        
                # Affichage des résultats de l'affinage
                st.write(f"Prediction après affinage : {prediction_opt_marital[0]}")
                st.write(f"Niveau de confiance après affinage : {max_proba_opt_marital:.2f}%")
                if prediction_opt_marital[0] == 0:
                    st.write("Conclusion: Ce client n'est pas susceptible de souscrire à un dépôt à terme.")
                else:
                    st.write("Conclusion: Ce client est susceptible de souscrire à un dépôt à terme.")
                    st.write("\nRecommandations : ")
                    st.write("- Durée d'appel : Pour maximiser les chances de souscription au dépôt, il faudra veiller à rester le plus longtemps possible au téléphone avec ce client (idéalement au moins 6 minutes).")
                    st.write("- Nombre de contacts pendant la campagne : il serait contre-productif de le contacter plus d'une fois.")


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
          
                filename_POUTCOME = "dilenesantos/XGBOOST_1_SD_model_PRED_poutcome_XGBOOST_quater.pkl"
                model_XGBOOST_1_SD_model_PRED_poutcome = joblib.load(filename_POUTCOME)         
                
                # Prédiction avec le DataFrame optimisé
                prediction_opt_poutcome = model_XGBOOST_1_SD_model_PRED_poutcome.predict(pred_df)
                prediction_proba_opt_poutcome = model_XGBOOST_1_SD_model_PRED_poutcome.predict_proba(pred_df)
                max_proba_opt_poutcome = np.max(prediction_proba_opt_poutcome[0]) * 100
        
                # Affichage des résultats de l'affinage
                st.write(f"Prediction après affinage : {prediction_opt_poutcome[0]}")
                st.write(f"Niveau de confiance après affinage : {max_proba_opt_poutcome:.2f}%")
                if prediction_opt_poutcome[0] == 0:
                    st.write("Conclusion: Ce client n'est pas susceptible de souscrire à un dépôt à terme.")
                else:
                    st.write("Conclusion: Ce client est susceptible de souscrire à un dépôt à terme.")
                    st.write("\nRecommandations : ")
                    st.write("- Durée d'appel : Pour maximiser les chances de souscription au dépôt, il faudra veiller à rester le plus longtemps possible au téléphone avec ce client (idéalement au moins 6 minutes).")
                    st.write("- Nombre de contacts pendant la campagne : il serait contre-productif de le contacter plus d'une fois.")


    
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
          
                filename_JOB = "dilenesantos/XGBOOST_1_SD_model_PRED_job_XGBOOST_1.pkl"
                model_XGBOOST_1_SD_model_PRED_job = joblib.load(filename_JOB)         
                
                # Prédiction avec le DataFrame optimisé
                prediction_opt_job = model_XGBOOST_1_SD_model_PRED_job.predict(pred_df)
                prediction_proba_opt_job = model_XGBOOST_1_SD_model_PRED_job.predict_proba(pred_df)
                max_proba_opt_job = np.max(prediction_proba_opt_job[0]) * 100
        
                # Affichage des résultats de l'affinage
                st.write(f"Prediction après affinage : {prediction_opt_job[0]}")
                st.write(f"Niveau de confiance après affinage : {max_proba_opt_job:.2f}%")
                if prediction_opt_job[0] == 0:
                    st.write("Conclusion: Ce client n'est pas susceptible de souscrire à un dépôt à terme.")
                else:
                    st.write("Conclusion: Ce client est susceptible de souscrire à un dépôt à terme.")
                    st.write("\nRecommandations : ")
                    st.write("- Durée d'appel : Pour maximiser les chances de souscription au dépôt, il faudra veiller à rester le plus longtemps possible au téléphone avec ce client (idéalement au moins 6 minutes).")
                    st.write("- Nombre de contacts pendant la campagne : il serait contre-productif de le contacter plus d'une fois.")


            
            elif option_to_add == "Client_Category_M":
                Client_Category_M = st.selectbox("Dernier appel de votre banque?", ('Prospect', 'Reached-6M', 'Reached+6M'))
                    #conditions d'affichage pour education : 
                if Client_Category_M == "Prospect":
                    Client_Category = "Jamais"
                elif Client_Category_M == "Reached-6M":
                    Client_Category = "Il y a moins de 6 mois"
                elif Client_Category_M == "Reached+6M":
                    Client_Category = "Il y a plus de 6  mois"
                else:
                    Client_Category = "Inconnu"  # Par défaut si `education` a une valeur inattendue
                    
                pred_df['Client_Category_M'] = Client_Category_M
                pred_df['Client_Category_M'] = pred_df['Client_Category_M'].replace(['Prospect', 'Reached-6M', 'Reached+6M'], [0, 1, 2])
                st.write("Dernier appel : ", Client_Category_M)
                
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
                st.dataframe(pred_df)
                
                filename_client_category = "dilenesantos/XGBOOST_1_SD_model_PRED_client_category_XGBOOST_1.pkl"
                model_XGBOOST_1_SD_model_PRED_client_category = joblib.load(filename_client_category)         
                
                # Prédiction avec le DataFrame optimisé
                prediction_opt_client_category = model_XGBOOST_1_SD_model_PRED_client_category.predict(pred_df)
                prediction_proba_opt_client_category = model_XGBOOST_1_SD_model_PRED_client_category.predict_proba(pred_df)
                max_proba_opt_client_category = np.max(prediction_proba_opt_client_category[0]) * 100
        
                # Affichage des résultats de l'affinage
                st.write(f"Prediction après affinage : {prediction_opt_client_category[0]}")
                st.write(f"Niveau de confiance après affinage : {max_proba_opt_client_category:.2f}%")
                if prediction_opt_client_category[0] == 0:
                    st.write("Conclusion: Ce client n'est pas susceptible de souscrire à un dépôt à terme.")
                else:
                    st.write("Conclusion: Ce client est susceptible de souscrire à un dépôt à terme.")
                    st.write("\nRecommandations : ")
                    st.write("- Durée d'appel : Pour maximiser les chances de souscription au dépôt, il faudra veiller à rester le plus longtemps possible au téléphone avec ce client (idéalement au moins 6 minutes).")
                    st.write("- Nombre de contacts pendant la campagne : il serait contre-productif de le contacter plus d'une fois.")


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
            elif option_to_add == "Client_Category_M":
                st.write("Dernier contact avec le client : ", Client_Category)
            elif option_to_add == "campaign":
                st.write(f"Nombre de contacts avec le client au cours de la campagne : {campaign}")
     
 
