import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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
shap.initjs()

df=pd.read_csv('bank.csv')

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

#code python SANS DURATION
dff_sans_duration = df.copy()
dff_sans_duration = dff_sans_duration[dff_sans_duration['age'] < 75]
dff_sans_duration = dff_sans_duration.loc[dff_sans_duration["balance"] > -2257]
dff_sans_duration = dff_sans_duration.loc[dff_sans_duration["balance"] < 4087]
dff_sans_duration = dff_sans_duration.loc[dff_sans_duration["campaign"] < 6]
dff_sans_duration = dff_sans_duration.loc[dff_sans_duration["previous"] < 2.5]
bins = [-2, -1, 180, 855]
labels = ['Prospect', 'Reached-6M', 'Reached+6M']
dff_sans_duration['Client_Category_M'] = pd.cut(dff_sans_duration['pdays'], bins=bins, labels=labels)
dff_sans_duration['Client_Category_M'] = dff_sans_duration['Client_Category_M'].astype('object')
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
dff_sans_duration = dff_sans_duration.drop(['contact'], axis=1)
dff_sans_duration = dff_sans_duration.drop(['pdays'], axis=1)
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

#APPEL du modèle rf_avec_duration _carolle_sauvegardé
@st.cache_resource
def load_model():
    return joblib.load("random_forest_model_carolle.pkl")
            
loaded_model_carolle = load_model()

#appel des shap values du modèle sauvegardé rf_carolle
shap_values_carolle = joblib.load("shap_values_carolle.pkl")

#APPEL du modèle xgboost_sans_duration sauvegardé
@st.cache_resource
def load_model():
    return joblib.load("xgboost_sd_opti.pkl")

# Charger le modèle
loaded_model_xgboost_sd = load_model()

#charger les shap values du modèle xgboost sauvegardé
shap_values_xgboost_sd = joblib.load("shap_values_xgboost_sd.pkl")          

with st.sidebar:
    selected = option_menu(
        menu_title='Sections',
        options=['Mes tests', 'Introduction','DataVisualisation', "Pre-processing", "Choix des métriques", "Modélisation", "Interprétation", "Conclusion", "Démo", "DémoSD"])



# Navigation dans les options
if selected == 'Mes tests':
    st.title("Projet de prédictions de souscription au dépôt à terme proposé par une banque")

    st.title("TESTS DIVERS POUR APPRENDRE")
    st.title("Titre : Test projet TEAM")
    st.header("Header : Introduction")
    st.subheader("Subheader : Projet Datascientest")
    if st.checkbox("Afficher") : 
        st.write("Suite du Streamlit BASE")

    st.write("Test affichage du dataframe")


    # Affichage
    st.dataframe(df.head(10))
    st.write(df.shape)
    st.dataframe(df.describe())
    
    #checkbox pour afficher ou non le nombre de Nans
    if st.checkbox("Afficher les NAns") :
        st.dataframe(df.isna().sum())


    st.write("test affichage d'un bouton")
    if st.button("Bouton : touche ici") :
        st.write("J'ai appuyé sur le bouton")

    st.write("test affichage d'une barre déroulante de sélection")
    st.selectbox("Quel est votre choix ?", ('Premier choix', "Second choix", 'Troisième choix'))



if selected == 'Introduction':  
    st.subheader("Contexte du projet")
    st.write("Le projet vise à analyser des données marketing issues d'une banque qui a utilisé le télémarketing pour promouvoir un produit financier appelé 'dépôt à terme'. Ce produit nécessite que le client dépose une somme d'argent dans un compte dédié, sans possibilité de retrait avant une date déterminée. En retour, le client reçoit des intérêts à la fin de cette période. L'objectif de cette analyse est d'examiner les informations personnelles des clients, comme l'âge, le statut matrimonial, le montant d'argent déposé, le nombre de contacts réalisés, etc., afin de comprendre les facteurs qui influencent la décision des clients de souscrire ou non à ce produit financier.")
    
    st.write("Problématique : ")
    st.write("La principale problématique de ce projet est de déterminer les facteurs qui influencent la probabilité qu'un client souscrive à un dépôt à terme à la suite d'une campagne de télémarketing.")
    st.write("L'objectif est double :")
    st.write("- Identifier et analyser visuellement et statistiquement les caractéristiques des clients qui sont corrélées avec la souscription au 'dépôt à terme'.")
    st.write("- Utiliser des techniques de Machine Learning pour prédire si un client va souscrire au 'dépôt à terme'.")


if selected == 'DataVisualisation':      
    st.title("DATAVISUALISATION")
    st.sidebar.title("MENU DATAVISUALISATION")
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
        
                st.image("stats_balance_deposit.png")
        
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
        
                st.image("stats_duration_deposit.png")

        
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
        
                st.image("stats_campaign_deposit.png")

        
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
        
                st.image("stats_previous_deposit.png")

        
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
            fig = plt.figure(figsize=(20,5))
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
                
        
if selected == 'Choix des métriques':
    st.title("Réflexion sur les métriques")



if selected == "Modélisation":
    
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
    results_sans_param = {}

    # Fonction pour entraîner et sauvegarder un modèle
    def train_and_save_model(model_name, clf, X_train, y_train):
        filename = f"{model_name.replace(' ', '_')}_model_avec_duration_sans_parametres.pkl"  # Nom du fichier
        try:
            # Charger le modèle si le fichier existe déjà
            trained_clf = joblib.load(filename)
        except FileNotFoundError:
            # Entraîner et sauvegarder le modèle
            clf.fit(X_train, y_train)
            joblib.dump(clf, filename)
            trained_clf = clf
        return trained_clf

    # Boucle pour entraîner ou charger les modèles
    for name, clf in classifiers.items():
        # Entraîner ou charger le modèle
        trained_clf = train_and_save_model(name, clf, X_train, y_train)
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
    results_melted = results_sans_param.reset_index().melt(
        id_vars="index", var_name="Metric", value_name="Score"
    )
    results_melted.rename(columns={"index": "Classifier"}, inplace=True)


    # dictionnaire avec les best modèles avec hyper paramètres trouvés AVEC DURATION !!!!
    classifiers_param_DURATION = {
        "Random Forest best": RandomForestClassifier(class_weight= 'balanced', max_depth=20, max_features='sqrt',min_samples_leaf=2, min_samples_split=10, n_estimators= 200, random_state=42),
        "Bagging": BaggingClassifier(random_state=42),
        "SVM best" : svm.SVC(C = 1, class_weight = 'balanced', gamma = 'scale', kernel ='rbf', random_state=42),
        "XGBOOST best" : XGBClassifier (colsample_bytree = 0.8, gamma = 5, learning_rate = 0.05, max_depth = 17, min_child_weight = 1, n_estimators = 200, subsample = 0.8, random_state=42)}
    results_best_param_DURATION = {}  # Affichage des résultats dans results


    # Fonction pour entraîner et sauvegarder un modèle
    def train_and_save_model_avec_param(model_name, clf, X_train, y_train):
        filename = f"{model_name.replace(' ', '_')}_model_avec_duration_hyperparam.pkl"  # Nom du fichier
        try:
            # Charger le modèle si le fichier existe déjà
            trained_clf = joblib.load(filename)
        except FileNotFoundError:
            # Entraîner et sauvegarder le modèle
            clf.fit(X_train, y_train)
            joblib.dump(clf, filename)
            trained_clf = clf
        return trained_clf

    # Boucle pour entraîner ou charger les modèles
    for name, clf in classifiers_param_DURATION.items():
        # Entraîner ou charger le modèle
        trained_clf = train_and_save_model_avec_param(name, clf, X_train, y_train)
        y_pred = trained_clf.predict(X_test)       

        # Calculer les métriques
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
            
        # Stocker les résultats
        results_best_param_DURATION[name] = {
             "Accuracy": accuracy,
             "F1 Score": f1,
             "Precision": precision,
             "Recall": recall,
        }

    #créer un dataframe avec tous les résultats obtenus précédemment et pour tous les classifier
    results_best_param_DURATION = pd.DataFrame(results_best_param_DURATION)
    results_best_param_DURATION = results_best_param_DURATION.T
    results_best_param_DURATION.columns = ['Accuracy', 'F1 Score', 'Precision', 'Recall']
                        
    #CLASSER LES RESULTATS DANS L'ORDRE DÉCROISSANT SELON LA COLONNE "Recall"
    results_best_param_DURATION = results_best_param_DURATION.sort_values(by='Recall', ascending=False)
    
    results_param_df_melted_DURATION = results_best_param_DURATION.reset_index().melt(id_vars="index", var_name="Metric", value_name="Score")
    results_param_df_melted_DURATION.rename(columns={"index": "Classifier"}, inplace=True)


    # dictionnaire avec les best modèles avec hyper paramètres trouvés SANS DURATION !!!!
    classifiers_param_sans_DURATION = {
        "Random Forest best param": RandomForestClassifier(class_weight='balanced', max_depth=8,  max_features='log2', min_samples_leaf=250, min_samples_split=300, n_estimators=400, random_state=42),
        "Decision Tree best param": DecisionTreeClassifier(class_weight='balanced', criterion='entropy', max_depth=5,  max_features=None, min_samples_leaf=100, min_samples_split=2, random_state=42),
        "Bagging": BaggingClassifier(random_state=42),
        "SVM best param" : svm.SVC(C=0.01, class_weight='balanced', gamma='scale', kernel='linear',random_state=42),
        "XGBOOST best param" : XGBClassifier(gamma=0.05,colsample_bytree=0.83, learning_rate=0.37, max_depth=6,  min_child_weight=1.2, n_estimators=30, reg_alpha=1.2, reg_lambda=1.7, scale_pos_weight=2.46, subsample=0.99, random_state=42)}
    results_param_sans_duration = {}  # Affichage des résultats dans results


    # Fonction pour entraîner et sauvegarder un modèle
    def train_and_save_model_avec_param_sans_duration(model_name, clf, X_train, y_train):
        filename = f"{model_name.replace(' ', '_')}_model_sans_duration_hyperparam_test_ad.pkl"  # Nom du fichier
        try:
            # Charger le modèle si le fichier existe déjà
            trained_clf = joblib.load(filename)
        except FileNotFoundError:
            # Entraîner et sauvegarder le modèle
            clf.fit(X_train, y_train)
            joblib.dump(clf, filename)
            trained_clf = clf
        return trained_clf

    # Boucle pour entraîner ou charger les modèles
    for name, clf in classifiers_param_sans_DURATION.items():
        # Entraîner ou charger le modèle
        trained_clf = train_and_save_model_avec_param_sans_duration(name, clf, X_train, y_train)
        y_pred = trained_clf.predict(X_test)
            
        # Calcul des métriques
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
            
        # Stocker les résultats
        results_param_sans_duration[name] = {
            "Accuracy": accuracy,
            "F1 Score": f1,
            "Precision": precision,
            "Recall": recall,
        }
         
    #créer un dataframe avec tous les résultats obtenus précédemment et pour tous les classifier
    results_param_sans_duration = pd.DataFrame(results_param_sans_duration)
    results_param_sans_duration = results_param_sans_duration.T
    results_param_sans_duration.columns = ['Accuracy', 'F1 Score', 'Precision', 'Recall']
                        
    #CLASSER LES RESULTATS DANS L'ORDRE DÉCROISSANT SELON LA COLONNE "Recall"
    results_param_sans_duration = results_param_sans_duration.sort_values(by='Recall', ascending=False)

    results_param_sans_duration_melted = results_param_sans_duration.reset_index().melt(id_vars="index", var_name="Metric", value_name="Score")
    results_param_sans_duration_melted.rename(columns={"index": "Classifier"}, inplace=True)


    
    st.title("MODÉLISATION")
    st.sidebar.title("MENU MODÉLISATION")   
    option_submenu = st.sidebar.selectbox('Sélection', ("Introduction", "Modélisation avec Duration", "Modélisation sans Duration", "Conclusion"))
        
    if option_submenu == 'Introduction':
        st.subheader("Méthodologie")
        st.write("On va effectuer deux modélisations, l'une en conservant la variable Duration et l'autre sans la variable Duration : on explique pourquoi blablabla.")
        st.write("Pour chaque modélisation, avec ou sans Duration, nous analysons les scores des principaux modèles de classification d'abord dans paramètres afin de sélectionner les 3 meilleurs modèles, puis sur ces 3 modèles nous effectuons des recherches d'hyperparamètres à l'aide de la fonction GridSearchCV afin de sélectionner le modèle le plus performant possible.")
        st.write("Enfin sur le meilleur modèle trouvé, nous effectuons une analyse SHAP afin d'interpréter les décisions prises par le modèle dans la détection des clients susceptibles de Deposit YES")
                
    if option_submenu == 'Modélisation avec Duration':
        pages=["Scores modèles sans paramètres", "Hyperparamètres et choix du modèle"]
        page=st.sidebar.radio('Afficher', pages)
               
        if page == pages[0] :
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
                
    
        if page == pages[1] :
            st.subheader("Hyperparamètres et choix du modèle")
            st.write("blabla GridSearchCV ....")
                
            st.write("On affiche le tableau des résultats des modèles :")
            st.dataframe(results_best_param_DURATION)
                
            st.write("Graphique :")
            # Visualisation des résultats des différents modèles :
            fig = plt.figure(figsize=(12, 6))
            sns.barplot(data=results_param_df_melted_DURATION,x="Classifier",y="Score",hue="Metric",palette="rainbow")
            # Ajouter des titres et légendes
            plt.title("Performance des modèles par métrique", fontsize=16)
            plt.xlabel("Modèles", fontsize=14)
            plt.ylabel("Scores", fontsize=14)
            plt.xticks(rotation=45)
            plt.legend(title="Métrique", fontsize=12)
            plt.legend(loc='lower right')
            plt.tight_layout()
            st.pyplot(fig)
                
                
            st.write("NB : ci-dessous résultats des hyper paramètres trouvés pour les 3 best modèles trouvés SANS DURATION - test sur le dataset avec duration*****")
            st.write("On affiche le tableau des résultats des modèles :")
            st.dataframe(results_param_sans_duration)
                
            st.write("Graphique :")
            # Visualisation des résultats des différents modèles :
            fig = plt.figure(figsize=(12, 6))
            sns.barplot(data=results_param_sans_duration_melted,x="Classifier",y="Score",hue="Metric",palette="rainbow")
            # Ajouter des titres et légendes
            plt.title("Performance des modèles par métrique", fontsize=16)
            plt.xlabel("Modèles", fontsize=14)
            plt.ylabel("Scores", fontsize=14)
            plt.xticks(rotation=45)
            plt.legend(title="Métrique", fontsize=12)
            plt.legend(loc='lower right')
            plt.tight_layout()
            st.pyplot(fig)
                
            st.subheader("Modèle sélectionné")
            st.write("Le modèle Random Forest avec les hyperparamètres ci-dessous affiche la meilleure performance en termes de Recall, aussi nous choisisons de poursuivre notre modélisation avec ce modèle")
            st.write("RandomForestClassifier(class_weight= 'balanced', max_depth=20, max_features='sqrt',min_samples_leaf=2, min_samples_split=10, n_estimators= 200, random_state=42)")
                
            st.write("Affichons le rapport de classification de ce modèle")
            rf_best = RandomForestClassifier(class_weight= 'balanced', max_depth=20, max_features='sqrt',min_samples_leaf=2, min_samples_split=10, n_estimators= 200, random_state=42)
            rf_best.fit(X_train, y_train)
            score_train = rf_best.score(X_train, y_train)
            score_test = rf_best.score(X_test, y_test)
            y_pred = rf_best.predict(X_test)
            table_rf = pd.crosstab(y_test,y_pred, rownames=['Realité'], colnames=['Prédiction'])
            st.dataframe(table_rf)
            st.write("Classification report :")
            report_dict = classification_report(y_test, y_pred, output_dict=True)
            # Convertir le dictionnaire en DataFrame
            report_df = pd.DataFrame(report_dict).T
            st.dataframe(report_df)


    
    if option_submenu == 'Modélisation sans Duration':
        pages=["Scores modèles sans paramètres", "Hyperparamètres et choix du modèle", "Hyperparamètres et choix du modèle TESTS"]
        page=st.sidebar.radio('Afficher', pages)
        classifiers_sd = {"Random Forest": RandomForestClassifier(random_state=42),"Logistic Regression": LogisticRegression(random_state=42),"Decision Tree": DecisionTreeClassifier(random_state=42),"KNN" : neighbors.KNeighborsClassifier(),"AdaBoost": AdaBoostClassifier(random_state=42),"Bagging": BaggingClassifier(random_state=42),"SVM" : svm.SVC(random_state=42),"XGBOOST" : XGBClassifier(random_state=42)}
        results_sd = {}  # Affichage des résultats dans results

        for name, clf in classifiers_sd.items():
            clf.fit(X_train_sd, y_train_sd)
            y_pred_sd = clf.predict(X_test_sd)
            accuracy_sd = accuracy_score(y_test_sd, y_pred_sd)
            f1_sd = f1_score(y_test_sd, y_pred_sd)
            precision_sd = precision_score(y_test_sd, y_pred_sd)
            recall_sd = recall_score(y_test_sd, y_pred_sd)
            results_sd[name] = {"Accuracy": accuracy_sd,"F1 Score": f1_sd,"Precision": precision_sd,"Recall": recall_sd}
            
        #créer un dataframe avec tous les résultats obtenus précédemment et pour tous les classifier
        results_df_sd = pd.DataFrame(results_sd)
        results_df_sd = results_df_sd.T
        results_df_sd.columns = ['Accuracy', 'F1 Score', 'Precision', 'Recall']
                
        #CLASSER LES RESULTATS DANS L'ORDRE DÉCROISSANT SELON LA COLONNE "Recall"
        results_df_sd = results_df_sd.sort_values(by='Recall', ascending=False)

        results_melted_sd = results_df_sd.reset_index().melt(id_vars="index", var_name="Metric", value_name="Score")
        results_melted_sd.rename(columns={"index": "Classifier"}, inplace=True)

        # dictionnaire avec les best modèles avec hyper paramètres trouvés AVEC DURATION !!!!
        classifiers_param_sd = {
            "Random Forest best": RandomForestClassifier(class_weight= 'balanced', max_depth = None, max_features = 'sqrt', min_samples_leaf= 2, min_samples_split= 15, n_estimators = 200, random_state=42),
            "Bagging": BaggingClassifier(random_state=42),
            "SVM best" : svm.SVC(C = 1, class_weight = 'balanced', gamma = 'scale', kernel ='rbf', random_state=42),
            "XGBOOST best" : XGBClassifier (colsample_bytree = 0.8, gamma = 5, learning_rate = 0.05, max_depth = 17, min_child_weight = 1, n_estimators = 200, subsample = 0.8, random_state=42)}
        results_param_sd = {}  # Affichage des résultats dans results

        for name, clf in classifiers_param_sd.items():
            clf.fit(X_train_sd, y_train_sd)
            y_pred_param_sd = clf.predict(X_test_sd)
            accuracy_param_sd = accuracy_score(y_test_sd, y_pred_param_sd)
            f1_param_sd = f1_score(y_test_sd, y_pred_param_sd)
            precision_param_sd = precision_score(y_test_sd, y_pred_param_sd)
            recall_param_sd = recall_score(y_test_sd, y_pred_param_sd)
            results_param_sd[name] = {"Accuracy": accuracy_param_sd,"F1 Score": f1_param_sd,"Precision": precision_param_sd,"Recall": recall_param_sd}
            
        #créer un dataframe avec tous les résultats obtenus précédemment et pour tous les classifier
        results_param_df_sd = pd.DataFrame(results_param_sd)
        results_param_df_sd = results_param_df_sd.T
        results_param_df_sd.columns = ['Accuracy', 'F1 Score', 'Precision', 'Recall']
                
        #CLASSER LES RESULTATS DANS L'ORDRE DÉCROISSANT SELON LA COLONNE "Recall"
        results_param_df_sd = results_param_df_sd.sort_values(by='Recall', ascending=False)

        results_param_df_melted_sd = results_param_df_sd.reset_index().melt(id_vars="index", var_name="Metric", value_name="Score")
        results_param_df_melted_sd.rename(columns={"index": "Classifier"}, inplace=True)



        # dictionnaire avec les best modèles avec hyper paramètres trouvés SANS DURATION !!!!
        classifiers_param_sd2 = {
            "Random Forest best param": RandomForestClassifier(class_weight='balanced', max_depth=8,  max_features='log2', min_samples_leaf=250, min_samples_split=300, n_estimators=400, random_state=42),
            "Decision Tree best param": DecisionTreeClassifier(class_weight='balanced', criterion='entropy', max_depth=5,  max_features=None, min_samples_leaf=100, min_samples_split=2, random_state=42),
            "SVM best param" : svm.SVC(C=0.01, class_weight='balanced', gamma='scale', kernel='linear',random_state=42),
            "XGBOOST best param" : XGBClassifier(gamma=0.05,colsample_bytree=0.83, learning_rate=0.37, max_depth=6,  min_child_weight=1.2, n_estimators=30, reg_alpha=1.2, reg_lambda=1.7, scale_pos_weight=2.46, subsample=0.99, random_state=42),
            "XGBOOST test_init" :  XGBClassifier(gamma=1.1,colsample_bytree=0.83, learning_rate=0.18, max_depth=5,  min_child_weight=1.2, n_estimators=33, reg_alpha=1.35, reg_lambda=1.8, scale_pos_weight=2.11, subsample=0.98, random_state=42),
            "XGBOOST test" : XGBClassifier(colsample_bytree=0.84, learning_rate=0.185, max_depth=5,  min_child_weight=1.55, n_estimators=31, reg_alpha=1.347, reg_lambda=1.75, scale_pos_weight=2.09, subsample=0.877, random_state=42),
            "XGBOOST test2" : XGBClassifier(colsample_bytree=0.85, learning_rate=0.187, max_depth=5,  min_child_weight=1.6, n_estimators=30, reg_alpha=1.343, reg_lambda=1.73, scale_pos_weight=2.09, subsample=0.87, random_state=42),
            "XGBOOST test3" : XGBClassifier(colsample_bytree=0.85, learning_rate=0.187, max_depth=5,  min_child_weight=1.6, n_estimators=30, reg_alpha=1.343, reg_lambda=1.73, scale_pos_weight=2.21, subsample=0.87, random_state=42),
            "XGBOOST test4" : XGBClassifier(colsample_bytree=0.85, learning_rate=0.187, max_depth=5,  min_child_weight=1.6, n_estimators=30, reg_alpha=1.343, reg_lambda=1.73, scale_pos_weight=2.18, subsample=0.87, random_state=42),
            "XGBOOST test3bis" : XGBClassifier(colsample_bytree=0.82, learning_rate=0.18, max_depth=5,  min_child_weight=1.4, n_estimators=32, reg_alpha=1.343, reg_lambda=1.73, scale_pos_weight=2.21, subsample=0.87, random_state=42),
            "XGBOOST test3ter" : XGBClassifier(gamma=0.05,colsample_bytree=0.82, learning_rate=0.18, max_depth=5,  min_child_weight=1.4, n_estimators=32, reg_alpha=1.343, reg_lambda=1.73, scale_pos_weight=2.21, subsample=0.87, random_state=42),
            "XGBOOST test3q" : XGBClassifier(gamma=0.05,colsample_bytree=0.82, learning_rate=0.18, max_depth=6,  min_child_weight=1.4, n_estimators=32, reg_alpha=1.343, reg_lambda=1.73, scale_pos_weight=2.20, subsample=0.87, random_state=42)}




        results_param_sd_2 = {}  # Affichage des résultats dans results

        for name, clf in classifiers_param_sd2.items():
            clf.fit(X_train_sd, y_train_sd)
            y_pred_param_sd2 = clf.predict(X_test_sd)
            accuracy_param_sd2 = accuracy_score(y_test_sd, y_pred_param_sd2)
            f1_param_sd2 = f1_score(y_test_sd, y_pred_param_sd2)
            precision_param_sd2 = precision_score(y_test_sd, y_pred_param_sd2)
            recall_param_sd2 = recall_score(y_test_sd, y_pred_param_sd2)
            results_param_sd_2[name] = {"Accuracy": accuracy_param_sd2,"F1 Score": f1_param_sd2,"Precision": precision_param_sd2,"Recall": recall_param_sd2}
            
        #créer un dataframe avec tous les résultats obtenus précédemment et pour tous les classifier
        results_param_df_sd2 = pd.DataFrame(results_param_sd_2)
        results_param_df_sd2 = results_param_df_sd2.T
        results_param_df_sd2.columns = ['Accuracy', 'F1 Score', 'Precision', 'Recall']
                
        #CLASSER LES RESULTATS DANS L'ORDRE DÉCROISSANT SELON LA COLONNE "Recall"
        results_param_df_sd2 = results_param_df_sd2.sort_values(by='Recall', ascending=False)

        results_param_df_melted_sd2 = results_param_df_sd2.reset_index().melt(id_vars="index", var_name="Metric", value_name="Score")
        results_param_df_melted_sd2.rename(columns={"index": "Classifier"}, inplace=True)



        if page == pages[0] :
            st.subheader("Scores modèles sans paramètres")
            st.write("On affiche le tableau des résultats des modèles :")
            st.dataframe(results_df_sd)
                    
            st.write("Graphique :")
            # Visualisation des résultats des différents modèles :
            fig = plt.figure(figsize=(12, 6))
            sns.barplot(data=results_melted_sd,x="Classifier",y="Score",hue="Metric",palette="rainbow")
            # Ajouter des titres et légendes
            plt.title("Performance des modèles par métrique", fontsize=16)
            plt.xlabel("Modèles", fontsize=14)
            plt.ylabel("Scores", fontsize=14)
            plt.xticks(rotation=45)
            plt.legend(title="Métrique", fontsize=12)
            plt.legend(loc='lower right')
            plt.tight_layout()
            st.pyplot(fig)

        if page == pages[1] :
            st.write("Recherche d'hyperparamètres et choix du modèle")
            st.write("blabla GridSearchCV ....")
                    
            st.write("On affiche le tableau des résultats des modèles :")
            st.dataframe(results_param_df_sd2)
                    
            st.write("Graphique :")
            # Visualisation des résultats des différents modèles :
            fig = plt.figure(figsize=(12, 6))
            sns.barplot(data=results_param_df_melted_sd2,x="Classifier",y="Score",hue="Metric",palette="rainbow")
            # Ajouter des titres et légendes
            plt.title("Performance des modèles par métrique", fontsize=16)
            plt.xlabel("Modèles", fontsize=14)
            plt.ylabel("Scores", fontsize=14)
            plt.xticks(rotation=45)
            plt.legend(title="Métrique", fontsize=12)
            plt.legend(loc='lower right')
            plt.tight_layout()
            st.pyplot(fig)
                    
            st.write("NB : ci-dessous résultats des hyper paramètres trouvés pour les 3 best modèles trouvés AVEC DURATION - test sur le dataset sans duration*****")
            st.write("On affiche le tableau des résultats des modèles :")
            st.dataframe(results_param_df_sd)
                    
            st.write("Graphique :")
            # Visualisation des résultats des différents modèles :
            fig = plt.figure(figsize=(12, 6))
            sns.barplot(data=results_param_df_melted_sd,x="Classifier",y="Score",hue="Metric",palette="rainbow")
            # Ajouter des titres et légendes
            plt.title("Performance des modèles par métrique", fontsize=16)
            plt.xlabel("Modèles", fontsize=14)
            plt.ylabel("Scores", fontsize=14)
            plt.xticks(rotation=45)
            plt.legend(title="Métrique", fontsize=12)
            plt.legend(loc='lower right')
            plt.tight_layout()
            st.pyplot(fig)
                    
            st.subheader("Modèle sélectionné")
            st.write("Le modèle XGBOOST avec les hyperparamètres ci-dessous affiche la meilleure performance en termes de Recall, aussi nous choisisons de poursuivre notre modélisation avec ce modèle")
            st.write("XGBClassifier 3ter(gamma=0.05,colsample_bytree=0.82, learning_rate=0.18, max_depth=5,  min_child_weight=1.4, n_estimators=32, reg_alpha=1.343, reg_lambda=1.73, scale_pos_weight=2.21, subsample=0.87, random_state=42)")
            st.write("Affichons le rapport de classification de ce modèle")
            xgboost_best = XGBClassifier(gamma=0.05,colsample_bytree=0.82, learning_rate=0.18, max_depth=5,  min_child_weight=1.4, n_estimators=32, reg_alpha=1.343, reg_lambda=1.73, scale_pos_weight=2.21, subsample=0.87, random_state=42)
            xgboost_best.fit(X_train_sd, y_train_sd)
            score_train = xgboost_best.score(X_train_sd, y_train_sd)
            score_test = xgboost_best.score(X_test_sd, y_test_sd)
            y_pred = xgboost_best.predict(X_test_sd)
            table_xgboost = pd.crosstab(y_test_sd,y_pred, rownames=['Realité'], colnames=['Prédiction'])
            st.dataframe(table_xgboost)
            st.write("Classification report :")
            report_dict_xgboost = classification_report(y_test_sd, y_pred, output_dict=True)
            # Convertir le dictionnaire en DataFrame
            report_df_xgboost = pd.DataFrame(report_dict_xgboost).T
            st.dataframe(report_df_xgboost)
                    
                    
            st.subheader("Modèle sélectionné")
            st.write("Le modèle XGBOOST avec les hyperparamètres ci-dessous affiche la meilleure performance en termes de Recall, aussi nous choisisons de poursuivre notre modélisation avec ce modèle")
            st.write("XGBClassifier INIT(seed=42, gamma=0.05,colsample_bytree=0.83, learning_rate=0.37, max_depth=6,  min_child_weight=1.2, n_estimators=30, reg_alpha=1.2, reg_lambda=1.7, scale_pos_weight=2.46, subsample=0.99, random_state=42)")
            st.write("Affichons le rapport de classification de ce modèle")
            xgboost_best = XGBClassifier(seed=42, gamma=0.05,colsample_bytree=0.83, learning_rate=0.37, max_depth=6,  min_child_weight=1.2, n_estimators=30, reg_alpha=1.2, reg_lambda=1.7, scale_pos_weight=2.46, subsample=0.99, random_state=42)
            xgboost_best.fit(X_train_sd, y_train_sd)
            score_train = xgboost_best.score(X_train_sd, y_train_sd)
            score_test = xgboost_best.score(X_test_sd, y_test_sd)
            y_pred = xgboost_best.predict(X_test_sd)
            table_xgboost = pd.crosstab(y_test_sd,y_pred, rownames=['Realité'], colnames=['Prédiction'])
            st.dataframe(table_xgboost)
            st.write("Classification report :")
            report_dict_xgboost = classification_report(y_test_sd, y_pred, output_dict=True)
            # Convertir le dictionnaire en DataFrame
            report_df_xgboost = pd.DataFrame(report_dict_xgboost).T
            st.dataframe(report_df_xgboost)
                    
            st.subheader("Modèle sélectionné")
            st.write("Le modèle XGBOOST avec les hyperparamètres ci-dessous affiche la meilleure performance en termes de Recall, aussi nous choisisons de poursuivre notre modélisation avec ce modèle")
            st.write("XGBClassifier test3ter (gamma=0.05,colsample_bytree=0.82, learning_rate=0.18, max_depth=5,  min_child_weight=1.4, n_estimators=32, reg_alpha=1.343, reg_lambda=1.73, scale_pos_weight=2.21, subsample=0.87, random_state=42)")
            st.write("Affichons le rapport de classification de ce modèle")
            xgboost_best = XGBClassifier(gamma=0.05,colsample_bytree=0.82, learning_rate=0.18, max_depth=5,  min_child_weight=1.4, n_estimators=32, reg_alpha=1.343, reg_lambda=1.73, scale_pos_weight=2.21, subsample=0.87, random_state=42)
            xgboost_best.fit(X_train_sd, y_train_sd)
            score_train = xgboost_best.score(X_train_sd, y_train_sd)
            score_test = xgboost_best.score(X_test_sd, y_test_sd)
            y_pred = xgboost_best.predict(X_test_sd)
            table_xgboost = pd.crosstab(y_test_sd,y_pred, rownames=['Realité'], colnames=['Prédiction'])
            st.dataframe(table_xgboost)
            st.write("Classification report :")
            report_dict_xgboost = classification_report(y_test_sd, y_pred, output_dict=True)
            # Convertir le dictionnaire en DataFrame
            report_df_xgboost = pd.DataFrame(report_dict_xgboost).T
            st.dataframe(report_df_xgboost)
                    
            ######test sur le shap pour vérifier classement OK 
            st.write("test sur le XGBOOST INIT pour vérifier classement OK ")

            XGBOOST_test_shap = XGBClassifier(seed=42, gamma=0.05,colsample_bytree=0.83, learning_rate=0.37, max_depth=6,  min_child_weight=1.2, n_estimators=30, reg_alpha=1.2, reg_lambda=1.7, scale_pos_weight=2.46, subsample=0.99, random_state=42)
            st.write("Modele testé ci-dessous = XGBClassifier(seed=42, gamma=0.05,colsample_bytree=0.83, learning_rate=0.37, max_depth=6,  min_child_weight=1.2, n_estimators=30, reg_alpha=1.2, reg_lambda=1.7, scale_pos_weight=2.46, subsample=0.99, random_state=42)")
            XGBOOST_test_shap.fit(X_train_sd, y_train_sd)
            y_pred = XGBOOST_test_shap.predict(X_test_sd)
            table_xgboost_test_shap = pd.crosstab(y_test_sd,y_pred, rownames=['Realité'], colnames=['Prédiction'])
            st.dataframe(table_xgboost_test_shap)
            st.write("Classification report :")
            report_dict_xgboost_test_shap = classification_report(y_test_sd, y_pred, output_dict=True)
            # Convertir le dictionnaire en DataFrame
            report_df_xgboost_test_shap = pd.DataFrame(report_dict_xgboost_test_shap).T
            st.dataframe(report_df_xgboost_test_shap)
            
            explainer = shap.TreeExplainer(XGBOOST_test_shap)
            shap_values_test_shap = explainer.shap_values(X_test_sd)
            
            fig = plt.figure()
            shap.summary_plot(shap_values_test_shap, X_test_sd)  
            st.pyplot(fig)
            
            fig = plt.figure()
            explanation = shap.Explanation(values=shap_values_test_shap,
                                 data=X_test_sd.values, # Assumant que  X_test est un DataFrame
                                 feature_names=X_test_sd.columns)
            shap.plots.bar(explanation)
            st.pyplot(fig)
            
            ### 1 CREATION D'UN EXPLANATION FILTRER SANS LES COLONNES POUR LESQUELLES NOUS ALLONS CALCULER LES MOYENNES

            #Étape 1 : Créer une liste des termes à exclure
            terms_to_exclude = ['month', 'weekday', 'job', 'poutcome', 'marital']

            #Étape 2 : Filtrer les colonnes qui ne contiennent pas les termes à exclure
            filtered_columns = [col for col in X_test_sd.columns if not any(term in col for term in terms_to_exclude)]

            #Étape 3 : Identifier les indices correspondants dans X_test_sd
            filtered_indices = [X_test_sd.columns.get_loc(col) for col in filtered_columns]
            shap_values_filtered = shap_values_test_shap[:, filtered_indices]

            # Étape 4 : On créé un nouvel Explanation avec les colonnes filtrées
            explanation_filtered = shap.Explanation(values=shap_values_filtered,
                                            data=X_test_sd.values[:, filtered_indices],  # Garder uniquement les colonnes correspondantes
                                            feature_names=filtered_columns)  # Les noms des features


            ###2 CRÉATION D'UN NOUVEAU EXPLANATION AVEC LES MOYENNES SHAP POUR LES COLONNES MONTH / WEEKDAY / POUTCOME / JOB / MARITAL

            #Fonction pour récupérer les moyennes SHAP en valeur absolue pour les colonnes qui nous intéressent
            def get_mean_shap_values(column_names, shap_values_test_shap):
                indices = [X_test_sd.columns.get_loc(col) for col in column_names]
                values = shap_values_test_shap[:, indices]
                return np.mean(np.abs(values), axis=0)

            #Étape 1 : On idenfie les colonnes que l'on recherche
            month_columns = [col for col in X_test_sd.columns if 'month' in col]
            weekday_columns = [col for col in X_test_sd.columns if 'weekday' in col]
            poutcome_columns = [col for col in X_test_sd.columns if 'poutcome' in col]
            job_columns = [col for col in X_test_sd.columns if 'job' in col]
            marital_columns = [col for col in X_test_sd.columns if 'marital' in col]

            #Étape 2 : On utiliser notre fonction pour calculer les moyennes des valeurs SHAP absolues
            mean_shap_month = get_mean_shap_values(month_columns, shap_values_test_shap)
            mean_shap_weekday = get_mean_shap_values(weekday_columns, shap_values_test_shap)
            mean_shap_poutcome = get_mean_shap_values(poutcome_columns, shap_values_test_shap)
            mean_shap_job = get_mean_shap_values(job_columns, shap_values_test_shap)
            mean_shap_marital = get_mean_shap_values(marital_columns, shap_values_test_shap)

            #Étape 3 : On combine les différentes moyennes et on les nomme
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

            #Étape 4 : On créé un nouvel Explanation avec les valeurs combinées
            explanation_combined = shap.Explanation(values=combined_values, data=np.array([[np.nan]] * len(combined_values)), feature_names=combined_feature_names)



            ###3 ON COMBINE LES 2 EXPLANTATION PRÉCÉDEMMENT CRÉÉS

            #Étape 1 : On récupére les nombre de lignes de explanation_filtered et on reshape explanation_combined pour avoir le même nombre de lignes
            num_samples = explanation_filtered.values.shape[0]
            combined_values_reshaped = np.repeat(np.array(explanation_combined.values)[:, np.newaxis], num_samples, axis=1).T

            #Étape 2: On concatenate les 2 explanations
            combined_values = np.concatenate([explanation_filtered.values, combined_values_reshaped], axis=1)

            #Étape 3: On combine le nom des colonnes provenant des 2 explanations
            combined_feature_names = (explanation_filtered.feature_names + explanation_combined.feature_names)

            #Étape 4: On créé un nouveau explanation avec les valeurs concatnées dans combined_values
            explanation_combined_new = shap.Explanation(values=combined_values,data=np.array([[np.nan]] * combined_values.shape[0]),feature_names=combined_feature_names,)

            fig = plt.figure(figsize=(10, 6))
            shap.plots.bar(explanation_combined_new, max_display=len(explanation_combined_new.feature_names))
            st.pyplot(fig)
            
            st.write("interpretation shap du modèle xgboost opti ici (XGBClassifier_test3ter)")
            #APPEL du modèle xgboost_sans_duration sauvegardé
            @st.cache_resource
            def load_model():
                return joblib.load("xgboost_sd_opti.pkl")

            # Charger le modèle
            loaded_model_xgboost_sd = load_model()

            #charger les shap values du modèle xgboost sauvegardé
            shap_values_xgboost_sd = joblib.load("shap_values_xgboost_sd.pkl") 
            
            fig = plt.figure()
            shap.summary_plot(shap_values_xgboost_sd, X_test_sd)  
            st.pyplot(fig)
            
            fig = plt.figure()
            explanation_2 = shap.Explanation(values=shap_values_xgboost_sd,
                                 data=X_test_sd.values, # Assumant que  X_test est un DataFrame
                                 feature_names=X_test_sd.columns)
            shap.plots.bar(explanation_2)
            st.pyplot(fig)
            
            y_pred = loaded_model_xgboost_sd.predict(X_test_sd)
            table_xgboost_ = pd.crosstab(y_test_sd,y_pred, rownames=['Realité'], colnames=['Prédiction'])
            st.dataframe(table_xgboost_)
            st.write("Classification report :")
            report_dict_xgboost_ = classification_report(y_test_sd, y_pred, output_dict=True)
            # Convertir le dictionnaire en DataFrame
            report_df_xgboost_ = pd.DataFrame(report_dict_xgboost_).T
            st.dataframe(report_df_xgboost_)
            
            ### 1 CREATION D'UN EXPLANATION FILTRER SANS LES COLONNES POUR LESQUELLES NOUS ALLONS CALCULER LES MOYENNES

            #Étape 1 : Créer une liste des termes à exclure
            terms_to_exclude = ['month', 'weekday', 'job', 'poutcome', 'marital']

            #Étape 2 : Filtrer les colonnes qui ne contiennent pas les termes à exclure
            filtered_columns = [col for col in X_test_sd.columns if not any(term in col for term in terms_to_exclude)]

            #Étape 3 : Identifier les indices correspondants dans X_test_sd
            filtered_indices = [X_test_sd.columns.get_loc(col) for col in filtered_columns]
            shap_values_filtered = shap_values_xgboost_sd[:, filtered_indices]

            # Étape 4 : On créé un nouvel Explanation avec les colonnes filtrées
            explanation_filtered = shap.Explanation(values=shap_values_filtered,
                                            data=X_test_sd.values[:, filtered_indices],  # Garder uniquement les colonnes correspondantes
                                            feature_names=filtered_columns)  # Les noms des features


            ###2 CRÉATION D'UN NOUVEAU EXPLANATION AVEC LES MOYENNES SHAP POUR LES COLONNES MONTH / WEEKDAY / POUTCOME / JOB / MARITAL

            #Fonction pour récupérer les moyennes SHAP en valeur absolue pour les colonnes qui nous intéressent
            def get_mean_shap_values(column_names, shap_values_xgboost_sd):
                indices = [X_test_sd.columns.get_loc(col) for col in column_names]
                values = shap_values_xgboost_sd[:, indices]
                return np.mean(np.abs(values), axis=0)

            #Étape 1 : On idenfie les colonnes que l'on recherche
            month_columns = [col for col in X_test_sd.columns if 'month' in col]
            weekday_columns = [col for col in X_test_sd.columns if 'weekday' in col]
            poutcome_columns = [col for col in X_test_sd.columns if 'poutcome' in col]
            job_columns = [col for col in X_test_sd.columns if 'job' in col]
            marital_columns = [col for col in X_test_sd.columns if 'marital' in col]

            #Étape 2 : On utiliser notre fonction pour calculer les moyennes des valeurs SHAP absolues
            mean_shap_month = get_mean_shap_values(month_columns, shap_values_xgboost_sd)
            mean_shap_weekday = get_mean_shap_values(weekday_columns, shap_values_xgboost_sd)
            mean_shap_poutcome = get_mean_shap_values(poutcome_columns, shap_values_xgboost_sd)
            mean_shap_job = get_mean_shap_values(job_columns, shap_values_xgboost_sd)
            mean_shap_marital = get_mean_shap_values(marital_columns, shap_values_xgboost_sd)

            #Étape 3 : On combine les différentes moyennes et on les nomme
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

            #Étape 4 : On créé un nouvel Explanation avec les valeurs combinées
            explanation_combined = shap.Explanation(values=combined_values, data=np.array([[np.nan]] * len(combined_values)), feature_names=combined_feature_names)



            ###3 ON COMBINE LES 2 EXPLANTATION PRÉCÉDEMMENT CRÉÉS

            #Étape 1 : On récupére les nombre de lignes de explanation_filtered et on reshape explanation_combined pour avoir le même nombre de lignes
            num_samples = explanation_filtered.values.shape[0]
            combined_values_reshaped = np.repeat(np.array(explanation_combined.values)[:, np.newaxis], num_samples, axis=1).T

            #Étape 2: On concatenate les 2 explanations
            combined_values = np.concatenate([explanation_filtered.values, combined_values_reshaped], axis=1)

            #Étape 3: On combine le nom des colonnes provenant des 2 explanations
            combined_feature_names = (explanation_filtered.feature_names + explanation_combined.feature_names)

            #Étape 4: On créé un nouveau explanation avec les valeurs concatnées dans combined_values
            explanation_combined_new = shap.Explanation(values=combined_values,data=np.array([[np.nan]] * combined_values.shape[0]),feature_names=combined_feature_names,)

            fig = plt.figure(figsize=(10, 6))
            shap.plots.bar(explanation_combined_new, max_display=len(explanation_combined_new.feature_names))
            st.pyplot(fig)
            
            
            st.write("____________________________")
            
            st.write("test sur le  XGBOOST test3q pour vérifier classement OK ")

            XGBOOST_test_shap = XGBClassifier(gamma=0.05,colsample_bytree=0.82, learning_rate=0.18, max_depth=6,  min_child_weight=1.4, n_estimators=32, reg_alpha=1.343, reg_lambda=1.73, scale_pos_weight=2.20, subsample=0.87, random_state=42)
            st.write("Modele testé ci-dessous = XGBClassifier(gamma=0.05,colsample_bytree=0.82, learning_rate=0.18, max_depth=6,  min_child_weight=1.4, n_estimators=32, reg_alpha=1.343, reg_lambda=1.73, scale_pos_weight=2.20, subsample=0.87, random_state=42)")
            XGBOOST_test_shap.fit(X_train_sd, y_train_sd)
            y_pred = XGBOOST_test_shap.predict(X_test_sd)
            table_xgboost_test_shap = pd.crosstab(y_test_sd,y_pred, rownames=['Realité'], colnames=['Prédiction'])
            st.dataframe(table_xgboost_test_shap)
            st.write("Classification report :")
            report_dict_xgboost_test_shap = classification_report(y_test_sd, y_pred, output_dict=True)
            # Convertir le dictionnaire en DataFrame
            report_df_xgboost_test_shap = pd.DataFrame(report_dict_xgboost_test_shap).T
            st.dataframe(report_df_xgboost_test_shap)
            
            explainer = shap.TreeExplainer(XGBOOST_test_shap)
            shap_values_test_shap = explainer.shap_values(X_test_sd)
            
            fig = plt.figure()
            shap.summary_plot(shap_values_test_shap, X_test_sd)  
            st.pyplot(fig)
            
            fig = plt.figure()
            explanation = shap.Explanation(values=shap_values_test_shap,
                                 data=X_test_sd.values, # Assumant que  X_test est un DataFrame
                                 feature_names=X_test_sd.columns)
            shap.plots.bar(explanation)
            st.pyplot(fig)
            
            ### 1 CREATION D'UN EXPLANATION FILTRER SANS LES COLONNES POUR LESQUELLES NOUS ALLONS CALCULER LES MOYENNES

            #Étape 1 : Créer une liste des termes à exclure
            terms_to_exclude = ['month', 'weekday', 'job', 'poutcome', 'marital']

            #Étape 2 : Filtrer les colonnes qui ne contiennent pas les termes à exclure
            filtered_columns = [col for col in X_test_sd.columns if not any(term in col for term in terms_to_exclude)]

            #Étape 3 : Identifier les indices correspondants dans X_test_sd
            filtered_indices = [X_test_sd.columns.get_loc(col) for col in filtered_columns]
            shap_values_filtered = shap_values_test_shap[:, filtered_indices]

            # Étape 4 : On créé un nouvel Explanation avec les colonnes filtrées
            explanation_filtered = shap.Explanation(values=shap_values_filtered,
                                            data=X_test_sd.values[:, filtered_indices],  # Garder uniquement les colonnes correspondantes
                                            feature_names=filtered_columns)  # Les noms des features


            ###2 CRÉATION D'UN NOUVEAU EXPLANATION AVEC LES MOYENNES SHAP POUR LES COLONNES MONTH / WEEKDAY / POUTCOME / JOB / MARITAL

            #Fonction pour récupérer les moyennes SHAP en valeur absolue pour les colonnes qui nous intéressent
            def get_mean_shap_values(column_names, shap_values_test_shap):
                indices = [X_test_sd.columns.get_loc(col) for col in column_names]
                values = shap_values_test_shap[:, indices]
                return np.mean(np.abs(values), axis=0)

            #Étape 1 : On idenfie les colonnes que l'on recherche
            month_columns = [col for col in X_test_sd.columns if 'month' in col]
            weekday_columns = [col for col in X_test_sd.columns if 'weekday' in col]
            poutcome_columns = [col for col in X_test_sd.columns if 'poutcome' in col]
            job_columns = [col for col in X_test_sd.columns if 'job' in col]
            marital_columns = [col for col in X_test_sd.columns if 'marital' in col]

            #Étape 2 : On utiliser notre fonction pour calculer les moyennes des valeurs SHAP absolues
            mean_shap_month = get_mean_shap_values(month_columns, shap_values_test_shap)
            mean_shap_weekday = get_mean_shap_values(weekday_columns, shap_values_test_shap)
            mean_shap_poutcome = get_mean_shap_values(poutcome_columns, shap_values_test_shap)
            mean_shap_job = get_mean_shap_values(job_columns, shap_values_test_shap)
            mean_shap_marital = get_mean_shap_values(marital_columns, shap_values_test_shap)

            #Étape 3 : On combine les différentes moyennes et on les nomme
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

            #Étape 4 : On créé un nouvel Explanation avec les valeurs combinées
            explanation_combined = shap.Explanation(values=combined_values, data=np.array([[np.nan]] * len(combined_values)), feature_names=combined_feature_names)



            ###3 ON COMBINE LES 2 EXPLANTATION PRÉCÉDEMMENT CRÉÉS

            #Étape 1 : On récupére les nombre de lignes de explanation_filtered et on reshape explanation_combined pour avoir le même nombre de lignes
            num_samples = explanation_filtered.values.shape[0]
            combined_values_reshaped = np.repeat(np.array(explanation_combined.values)[:, np.newaxis], num_samples, axis=1).T

            #Étape 2: On concatenate les 2 explanations
            combined_values = np.concatenate([explanation_filtered.values, combined_values_reshaped], axis=1)

            #Étape 3: On combine le nom des colonnes provenant des 2 explanations
            combined_feature_names = (explanation_filtered.feature_names + explanation_combined.feature_names)

            #Étape 4: On créé un nouveau explanation avec les valeurs concatnées dans combined_values
            explanation_combined_new = shap.Explanation(values=combined_values,data=np.array([[np.nan]] * combined_values.shape[0]),feature_names=combined_feature_names,)

            fig = plt.figure(figsize=(10, 6))
            shap.plots.bar(explanation_combined_new, max_display=len(explanation_combined_new.feature_names))
            st.pyplot(fig)
            
            st.write("____________________________")
            
            st.write("test sur le  XGBOOST test2 pour vérifier classement OK ")

            XGBOOST_test_shap = XGBClassifier(colsample_bytree=0.85, learning_rate=0.187, max_depth=5,  min_child_weight=1.6, n_estimators=30, reg_alpha=1.343, reg_lambda=1.73, scale_pos_weight=2.09, subsample=0.87, random_state=42)
            st.write("Modele testé ci-dessous = XGBClassifier(colsample_bytree=0.85, learning_rate=0.187, max_depth=5,  min_child_weight=1.6, n_estimators=30, reg_alpha=1.343, reg_lambda=1.73, scale_pos_weight=2.09, subsample=0.87, random_state=42)")
            XGBOOST_test_shap.fit(X_train_sd, y_train_sd)
            y_pred = XGBOOST_test_shap.predict(X_test_sd)
            table_xgboost_test_shap = pd.crosstab(y_test_sd,y_pred, rownames=['Realité'], colnames=['Prédiction'])
            st.dataframe(table_xgboost_test_shap)
            st.write("Classification report :")
            report_dict_xgboost_test_shap = classification_report(y_test_sd, y_pred, output_dict=True)
            # Convertir le dictionnaire en DataFrame
            report_df_xgboost_test_shap = pd.DataFrame(report_dict_xgboost_test_shap).T
            st.dataframe(report_df_xgboost_test_shap)
            
            explainer = shap.TreeExplainer(XGBOOST_test_shap)
            shap_values_test_shap = explainer.shap_values(X_test_sd)
            
            fig = plt.figure()
            shap.summary_plot(shap_values_test_shap, X_test_sd)  
            st.pyplot(fig)
            
            fig = plt.figure()
            explanation = shap.Explanation(values=shap_values_test_shap,
                                 data=X_test_sd.values, # Assumant que  X_test est un DataFrame
                                 feature_names=X_test_sd.columns)
            shap.plots.bar(explanation)
            st.pyplot(fig)
            
            ### 1 CREATION D'UN EXPLANATION FILTRER SANS LES COLONNES POUR LESQUELLES NOUS ALLONS CALCULER LES MOYENNES

            #Étape 1 : Créer une liste des termes à exclure
            terms_to_exclude = ['month', 'weekday', 'job', 'poutcome', 'marital']

            #Étape 2 : Filtrer les colonnes qui ne contiennent pas les termes à exclure
            filtered_columns = [col for col in X_test_sd.columns if not any(term in col for term in terms_to_exclude)]

            #Étape 3 : Identifier les indices correspondants dans X_test_sd
            filtered_indices = [X_test_sd.columns.get_loc(col) for col in filtered_columns]
            shap_values_filtered = shap_values_test_shap[:, filtered_indices]

            # Étape 4 : On créé un nouvel Explanation avec les colonnes filtrées
            explanation_filtered = shap.Explanation(values=shap_values_filtered,
                                            data=X_test_sd.values[:, filtered_indices],  # Garder uniquement les colonnes correspondantes
                                            feature_names=filtered_columns)  # Les noms des features


            ###2 CRÉATION D'UN NOUVEAU EXPLANATION AVEC LES MOYENNES SHAP POUR LES COLONNES MONTH / WEEKDAY / POUTCOME / JOB / MARITAL

            #Fonction pour récupérer les moyennes SHAP en valeur absolue pour les colonnes qui nous intéressent
            def get_mean_shap_values(column_names, shap_values_test_shap):
                indices = [X_test_sd.columns.get_loc(col) for col in column_names]
                values = shap_values_test_shap[:, indices]
                return np.mean(np.abs(values), axis=0)

            #Étape 1 : On idenfie les colonnes que l'on recherche
            month_columns = [col for col in X_test_sd.columns if 'month' in col]
            weekday_columns = [col for col in X_test_sd.columns if 'weekday' in col]
            poutcome_columns = [col for col in X_test_sd.columns if 'poutcome' in col]
            job_columns = [col for col in X_test_sd.columns if 'job' in col]
            marital_columns = [col for col in X_test_sd.columns if 'marital' in col]

            #Étape 2 : On utiliser notre fonction pour calculer les moyennes des valeurs SHAP absolues
            mean_shap_month = get_mean_shap_values(month_columns, shap_values_test_shap)
            mean_shap_weekday = get_mean_shap_values(weekday_columns, shap_values_test_shap)
            mean_shap_poutcome = get_mean_shap_values(poutcome_columns, shap_values_test_shap)
            mean_shap_job = get_mean_shap_values(job_columns, shap_values_test_shap)
            mean_shap_marital = get_mean_shap_values(marital_columns, shap_values_test_shap)

            #Étape 3 : On combine les différentes moyennes et on les nomme
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

            #Étape 4 : On créé un nouvel Explanation avec les valeurs combinées
            explanation_combined = shap.Explanation(values=combined_values, data=np.array([[np.nan]] * len(combined_values)), feature_names=combined_feature_names)



            ###3 ON COMBINE LES 2 EXPLANTATION PRÉCÉDEMMENT CRÉÉS

            #Étape 1 : On récupére les nombre de lignes de explanation_filtered et on reshape explanation_combined pour avoir le même nombre de lignes
            num_samples = explanation_filtered.values.shape[0]
            combined_values_reshaped = np.repeat(np.array(explanation_combined.values)[:, np.newaxis], num_samples, axis=1).T

            #Étape 2: On concatenate les 2 explanations
            combined_values = np.concatenate([explanation_filtered.values, combined_values_reshaped], axis=1)

            #Étape 3: On combine le nom des colonnes provenant des 2 explanations
            combined_feature_names = (explanation_filtered.feature_names + explanation_combined.feature_names)

            #Étape 4: On créé un nouveau explanation avec les valeurs concatnées dans combined_values
            explanation_combined_new = shap.Explanation(values=combined_values,data=np.array([[np.nan]] * combined_values.shape[0]),feature_names=combined_feature_names,)

            fig = plt.figure(figsize=(10, 6))
            shap.plots.bar(explanation_combined_new, max_display=len(explanation_combined_new.feature_names))
            st.pyplot(fig)
            
            
            st.write("____________________________")
            
            st.write("test sur le  XGBOOST test encore pour vérifier classement OK ")

            XGBOOST_test_shap = XGBClassifier(gamma=0.05,colsample_bytree=0.87, learning_rate=0.37, max_depth=6, min_child_weight=1.2, n_estimators=30, reg_alpha=1.2, reg_lambda=1.7, scale_pos_weight=2.46, subsample=0.99, random_state=42)
            st.write("Modele testé ci-dessous = XGBClassifier(gamma=0.05,colsample_bytree=0.87, learning_rate=0.37, max_depth=6, min_child_weight=1.2, n_estimators=30, reg_alpha=1.2, reg_lambda=1.7, scale_pos_weight=2.46, subsample=0.99, random_state=42)")
            XGBOOST_test_shap.fit(X_train_sd, y_train_sd)
            y_pred = XGBOOST_test_shap.predict(X_test_sd)
            table_xgboost_test_shap = pd.crosstab(y_test_sd,y_pred, rownames=['Realité'], colnames=['Prédiction'])
            st.dataframe(table_xgboost_test_shap)
            st.write("Classification report :")
            report_dict_xgboost_test_shap = classification_report(y_test_sd, y_pred, output_dict=True)
            # Convertir le dictionnaire en DataFrame
            report_df_xgboost_test_shap = pd.DataFrame(report_dict_xgboost_test_shap).T
            st.dataframe(report_df_xgboost_test_shap)
            
            explainer = shap.TreeExplainer(XGBOOST_test_shap)
            shap_values_test_shap = explainer.shap_values(X_test_sd)
            
            fig = plt.figure()
            shap.summary_plot(shap_values_test_shap, X_test_sd)  
            st.pyplot(fig)
            
            fig = plt.figure()
            explanation = shap.Explanation(values=shap_values_test_shap,
                                 data=X_test_sd.values, # Assumant que  X_test est un DataFrame
                                 feature_names=X_test_sd.columns)
            shap.plots.bar(explanation)
            st.pyplot(fig)
            
            ### 1 CREATION D'UN EXPLANATION FILTRER SANS LES COLONNES POUR LESQUELLES NOUS ALLONS CALCULER LES MOYENNES

            #Étape 1 : Créer une liste des termes à exclure
            terms_to_exclude = ['month', 'weekday', 'job', 'poutcome', 'marital']

            #Étape 2 : Filtrer les colonnes qui ne contiennent pas les termes à exclure
            filtered_columns = [col for col in X_test_sd.columns if not any(term in col for term in terms_to_exclude)]

            #Étape 3 : Identifier les indices correspondants dans X_test_sd
            filtered_indices = [X_test_sd.columns.get_loc(col) for col in filtered_columns]
            shap_values_filtered = shap_values_test_shap[:, filtered_indices]

            # Étape 4 : On créé un nouvel Explanation avec les colonnes filtrées
            explanation_filtered = shap.Explanation(values=shap_values_filtered,
                                            data=X_test_sd.values[:, filtered_indices],  # Garder uniquement les colonnes correspondantes
                                            feature_names=filtered_columns)  # Les noms des features


            ###2 CRÉATION D'UN NOUVEAU EXPLANATION AVEC LES MOYENNES SHAP POUR LES COLONNES MONTH / WEEKDAY / POUTCOME / JOB / MARITAL

            #Fonction pour récupérer les moyennes SHAP en valeur absolue pour les colonnes qui nous intéressent
            def get_mean_shap_values(column_names, shap_values_test_shap):
                indices = [X_test_sd.columns.get_loc(col) for col in column_names]
                values = shap_values_test_shap[:, indices]
                return np.mean(np.abs(values), axis=0)

            #Étape 1 : On idenfie les colonnes que l'on recherche
            month_columns = [col for col in X_test_sd.columns if 'month' in col]
            weekday_columns = [col for col in X_test_sd.columns if 'weekday' in col]
            poutcome_columns = [col for col in X_test_sd.columns if 'poutcome' in col]
            job_columns = [col for col in X_test_sd.columns if 'job' in col]
            marital_columns = [col for col in X_test_sd.columns if 'marital' in col]

            #Étape 2 : On utiliser notre fonction pour calculer les moyennes des valeurs SHAP absolues
            mean_shap_month = get_mean_shap_values(month_columns, shap_values_test_shap)
            mean_shap_weekday = get_mean_shap_values(weekday_columns, shap_values_test_shap)
            mean_shap_poutcome = get_mean_shap_values(poutcome_columns, shap_values_test_shap)
            mean_shap_job = get_mean_shap_values(job_columns, shap_values_test_shap)
            mean_shap_marital = get_mean_shap_values(marital_columns, shap_values_test_shap)

            #Étape 3 : On combine les différentes moyennes et on les nomme
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

            #Étape 4 : On créé un nouvel Explanation avec les valeurs combinées
            explanation_combined = shap.Explanation(values=combined_values, data=np.array([[np.nan]] * len(combined_values)), feature_names=combined_feature_names)



            ###3 ON COMBINE LES 2 EXPLANTATION PRÉCÉDEMMENT CRÉÉS

            #Étape 1 : On récupére les nombre de lignes de explanation_filtered et on reshape explanation_combined pour avoir le même nombre de lignes
            num_samples = explanation_filtered.values.shape[0]
            combined_values_reshaped = np.repeat(np.array(explanation_combined.values)[:, np.newaxis], num_samples, axis=1).T

            #Étape 2: On concatenate les 2 explanations
            combined_values = np.concatenate([explanation_filtered.values, combined_values_reshaped], axis=1)

            #Étape 3: On combine le nom des colonnes provenant des 2 explanations
            combined_feature_names = (explanation_filtered.feature_names + explanation_combined.feature_names)

            #Étape 4: On créé un nouveau explanation avec les valeurs concatnées dans combined_values
            explanation_combined_new = shap.Explanation(values=combined_values,data=np.array([[np.nan]] * combined_values.shape[0]),feature_names=combined_feature_names,)

            fig = plt.figure(figsize=(10, 6))
            shap.plots.bar(explanation_combined_new, max_display=len(explanation_combined_new.feature_names))
            st.pyplot(fig)
            
            
            st.write("____________________________")
            
            st.write("test sur le  XGBOOST test encore2 pour vérifier classement OK ")

            XGBOOST_test_shap = XGBClassifier(gamma=0.05,colsample_bytree=0.87, learning_rate=0.39, max_depth=6, min_child_weight=1.2, n_estimators=33, reg_alpha=1.2, reg_lambda=1.9, scale_pos_weight=2.56, subsample=0.99, random_state=42)
            st.write("modif = learning_rate=0.39 vs 0.37 + reg_lambda=1.9 vs 1.7 + n_estimators=33 vs 30 + scale_pos_weight=2.56 vs 2.46")
            XGBOOST_test_shap.fit(X_train_sd, y_train_sd)
            y_pred = XGBOOST_test_shap.predict(X_test_sd)
            table_xgboost_test_shap = pd.crosstab(y_test_sd,y_pred, rownames=['Realité'], colnames=['Prédiction'])
            st.dataframe(table_xgboost_test_shap)
            st.write("Classification report :")
            report_dict_xgboost_test_shap = classification_report(y_test_sd, y_pred, output_dict=True)
            # Convertir le dictionnaire en DataFrame
            report_df_xgboost_test_shap = pd.DataFrame(report_dict_xgboost_test_shap).T
            st.dataframe(report_df_xgboost_test_shap)
            
            explainer = shap.TreeExplainer(XGBOOST_test_shap)
            shap_values_test_shap = explainer.shap_values(X_test_sd)
            
            fig = plt.figure()
            shap.summary_plot(shap_values_test_shap, X_test_sd)  
            st.pyplot(fig)
            
            fig = plt.figure()
            explanation = shap.Explanation(values=shap_values_test_shap,
                                 data=X_test_sd.values, # Assumant que  X_test est un DataFrame
                                 feature_names=X_test_sd.columns)
            shap.plots.bar(explanation)
            st.pyplot(fig)
            
            ### 1 CREATION D'UN EXPLANATION FILTRER SANS LES COLONNES POUR LESQUELLES NOUS ALLONS CALCULER LES MOYENNES

            #Étape 1 : Créer une liste des termes à exclure
            terms_to_exclude = ['month', 'weekday', 'job', 'poutcome', 'marital']

            #Étape 2 : Filtrer les colonnes qui ne contiennent pas les termes à exclure
            filtered_columns = [col for col in X_test_sd.columns if not any(term in col for term in terms_to_exclude)]

            #Étape 3 : Identifier les indices correspondants dans X_test_sd
            filtered_indices = [X_test_sd.columns.get_loc(col) for col in filtered_columns]
            shap_values_filtered = shap_values_test_shap[:, filtered_indices]

            # Étape 4 : On créé un nouvel Explanation avec les colonnes filtrées
            explanation_filtered = shap.Explanation(values=shap_values_filtered,
                                            data=X_test_sd.values[:, filtered_indices],  # Garder uniquement les colonnes correspondantes
                                            feature_names=filtered_columns)  # Les noms des features


            ###2 CRÉATION D'UN NOUVEAU EXPLANATION AVEC LES MOYENNES SHAP POUR LES COLONNES MONTH / WEEKDAY / POUTCOME / JOB / MARITAL

            #Fonction pour récupérer les moyennes SHAP en valeur absolue pour les colonnes qui nous intéressent
            def get_mean_shap_values(column_names, shap_values_test_shap):
                indices = [X_test_sd.columns.get_loc(col) for col in column_names]
                values = shap_values_test_shap[:, indices]
                return np.mean(np.abs(values), axis=0)

            #Étape 1 : On idenfie les colonnes que l'on recherche
            month_columns = [col for col in X_test_sd.columns if 'month' in col]
            weekday_columns = [col for col in X_test_sd.columns if 'weekday' in col]
            poutcome_columns = [col for col in X_test_sd.columns if 'poutcome' in col]
            job_columns = [col for col in X_test_sd.columns if 'job' in col]
            marital_columns = [col for col in X_test_sd.columns if 'marital' in col]

            #Étape 2 : On utiliser notre fonction pour calculer les moyennes des valeurs SHAP absolues
            mean_shap_month = get_mean_shap_values(month_columns, shap_values_test_shap)
            mean_shap_weekday = get_mean_shap_values(weekday_columns, shap_values_test_shap)
            mean_shap_poutcome = get_mean_shap_values(poutcome_columns, shap_values_test_shap)
            mean_shap_job = get_mean_shap_values(job_columns, shap_values_test_shap)
            mean_shap_marital = get_mean_shap_values(marital_columns, shap_values_test_shap)

            #Étape 3 : On combine les différentes moyennes et on les nomme
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

            #Étape 4 : On créé un nouvel Explanation avec les valeurs combinées
            explanation_combined = shap.Explanation(values=combined_values, data=np.array([[np.nan]] * len(combined_values)), feature_names=combined_feature_names)



            ###3 ON COMBINE LES 2 EXPLANTATION PRÉCÉDEMMENT CRÉÉS

            #Étape 1 : On récupére les nombre de lignes de explanation_filtered et on reshape explanation_combined pour avoir le même nombre de lignes
            num_samples = explanation_filtered.values.shape[0]
            combined_values_reshaped = np.repeat(np.array(explanation_combined.values)[:, np.newaxis], num_samples, axis=1).T

            #Étape 2: On concatenate les 2 explanations
            combined_values = np.concatenate([explanation_filtered.values, combined_values_reshaped], axis=1)

            #Étape 3: On combine le nom des colonnes provenant des 2 explanations
            combined_feature_names = (explanation_filtered.feature_names + explanation_combined.feature_names)

            #Étape 4: On créé un nouveau explanation avec les valeurs concatnées dans combined_values
            explanation_combined_new = shap.Explanation(values=combined_values,data=np.array([[np.nan]] * combined_values.shape[0]),feature_names=combined_feature_names,)

            fig = plt.figure(figsize=(10, 6))
            shap.plots.bar(explanation_combined_new, max_display=len(explanation_combined_new.feature_names))
            st.pyplot(fig)
            
            
            st.write("____________________________")
            
            st.write("test sur le  XGBOOST test encore2 pour vérifier classement OK ")

            XGBOOST_test_shap = XGBClassifier(gamma=0.05,colsample_bytree=0.88, learning_rate=0.39, max_depth=6, min_child_weight=1.2, n_estimators=32, reg_alpha=1.2, reg_lambda=1.9, scale_pos_weight=2.56, subsample=0.99, random_state=42)
            st.write("modif = learning_rate=0.39 vs 0.37 + reg_lambda=1.9 vs 1.7 + n_estimators=33 vs 30 + scale_pos_weight=2.59 vs 2.46")
            XGBOOST_test_shap.fit(X_train_sd, y_train_sd)
            y_pred = XGBOOST_test_shap.predict(X_test_sd)
            table_xgboost_test_shap = pd.crosstab(y_test_sd,y_pred, rownames=['Realité'], colnames=['Prédiction'])
            st.dataframe(table_xgboost_test_shap)
            st.write("Classification report :")
            report_dict_xgboost_test_shap = classification_report(y_test_sd, y_pred, output_dict=True)
            # Convertir le dictionnaire en DataFrame
            report_df_xgboost_test_shap = pd.DataFrame(report_dict_xgboost_test_shap).T
            st.dataframe(report_df_xgboost_test_shap)
            
            explainer = shap.TreeExplainer(XGBOOST_test_shap)
            shap_values_test_shap = explainer.shap_values(X_test_sd)
            
            fig = plt.figure()
            shap.summary_plot(shap_values_test_shap, X_test_sd)  
            st.pyplot(fig)
            
            fig = plt.figure()
            explanation = shap.Explanation(values=shap_values_test_shap,
                                 data=X_test_sd.values, # Assumant que  X_test est un DataFrame
                                 feature_names=X_test_sd.columns)
            shap.plots.bar(explanation)
            st.pyplot(fig)
            
            ### 1 CREATION D'UN EXPLANATION FILTRER SANS LES COLONNES POUR LESQUELLES NOUS ALLONS CALCULER LES MOYENNES

            #Étape 1 : Créer une liste des termes à exclure
            terms_to_exclude = ['month', 'weekday', 'job', 'poutcome', 'marital']

            #Étape 2 : Filtrer les colonnes qui ne contiennent pas les termes à exclure
            filtered_columns = [col for col in X_test_sd.columns if not any(term in col for term in terms_to_exclude)]

            #Étape 3 : Identifier les indices correspondants dans X_test_sd
            filtered_indices = [X_test_sd.columns.get_loc(col) for col in filtered_columns]
            shap_values_filtered = shap_values_test_shap[:, filtered_indices]

            # Étape 4 : On créé un nouvel Explanation avec les colonnes filtrées
            explanation_filtered = shap.Explanation(values=shap_values_filtered,
                                            data=X_test_sd.values[:, filtered_indices],  # Garder uniquement les colonnes correspondantes
                                            feature_names=filtered_columns)  # Les noms des features


            ###2 CRÉATION D'UN NOUVEAU EXPLANATION AVEC LES MOYENNES SHAP POUR LES COLONNES MONTH / WEEKDAY / POUTCOME / JOB / MARITAL

            #Fonction pour récupérer les moyennes SHAP en valeur absolue pour les colonnes qui nous intéressent
            def get_mean_shap_values(column_names, shap_values_test_shap):
                indices = [X_test_sd.columns.get_loc(col) for col in column_names]
                values = shap_values_test_shap[:, indices]
                return np.mean(np.abs(values), axis=0)

            #Étape 1 : On idenfie les colonnes que l'on recherche
            month_columns = [col for col in X_test_sd.columns if 'month' in col]
            weekday_columns = [col for col in X_test_sd.columns if 'weekday' in col]
            poutcome_columns = [col for col in X_test_sd.columns if 'poutcome' in col]
            job_columns = [col for col in X_test_sd.columns if 'job' in col]
            marital_columns = [col for col in X_test_sd.columns if 'marital' in col]

            #Étape 2 : On utiliser notre fonction pour calculer les moyennes des valeurs SHAP absolues
            mean_shap_month = get_mean_shap_values(month_columns, shap_values_test_shap)
            mean_shap_weekday = get_mean_shap_values(weekday_columns, shap_values_test_shap)
            mean_shap_poutcome = get_mean_shap_values(poutcome_columns, shap_values_test_shap)
            mean_shap_job = get_mean_shap_values(job_columns, shap_values_test_shap)
            mean_shap_marital = get_mean_shap_values(marital_columns, shap_values_test_shap)

            #Étape 3 : On combine les différentes moyennes et on les nomme
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

            #Étape 4 : On créé un nouvel Explanation avec les valeurs combinées
            explanation_combined = shap.Explanation(values=combined_values, data=np.array([[np.nan]] * len(combined_values)), feature_names=combined_feature_names)



            ###3 ON COMBINE LES 2 EXPLANTATION PRÉCÉDEMMENT CRÉÉS

            #Étape 1 : On récupére les nombre de lignes de explanation_filtered et on reshape explanation_combined pour avoir le même nombre de lignes
            num_samples = explanation_filtered.values.shape[0]
            combined_values_reshaped = np.repeat(np.array(explanation_combined.values)[:, np.newaxis], num_samples, axis=1).T

            #Étape 2: On concatenate les 2 explanations
            combined_values = np.concatenate([explanation_filtered.values, combined_values_reshaped], axis=1)

            #Étape 3: On combine le nom des colonnes provenant des 2 explanations
            combined_feature_names = (explanation_filtered.feature_names + explanation_combined.feature_names)

            #Étape 4: On créé un nouveau explanation avec les valeurs concatnées dans combined_values
            explanation_combined_new = shap.Explanation(values=combined_values,data=np.array([[np.nan]] * combined_values.shape[0]),feature_names=combined_feature_names,)

            fig = plt.figure(figsize=(10, 6))
            shap.plots.bar(explanation_combined_new, max_display=len(explanation_combined_new.feature_names))
            st.pyplot(fig)
            
            st.write("____________________________")
            
            st.write("test sur le  XGBOOST test encore pour vérifier classement OK ")

            XGBOOST_test_shap = XGBClassifier(gamma=0.05,colsample_bytree=0.88, learning_rate=0.39, max_depth=6, min_child_weight=1.2, n_estimators=30, reg_alpha=1.2, reg_lambda=1.8, scale_pos_weight=2.56, subsample=0.99, random_state=42)
            st.write("31 puis 30 et 1.8")
            XGBOOST_test_shap.fit(X_train_sd, y_train_sd)
            y_pred = XGBOOST_test_shap.predict(X_test_sd)
            table_xgboost_test_shap = pd.crosstab(y_test_sd,y_pred, rownames=['Realité'], colnames=['Prédiction'])
            st.dataframe(table_xgboost_test_shap)
            st.write("Classification report :")
            report_dict_xgboost_test_shap = classification_report(y_test_sd, y_pred, output_dict=True)
            # Convertir le dictionnaire en DataFrame
            report_df_xgboost_test_shap = pd.DataFrame(report_dict_xgboost_test_shap).T
            st.dataframe(report_df_xgboost_test_shap)
            
            explainer = shap.TreeExplainer(XGBOOST_test_shap)
            shap_values_test_shap = explainer.shap_values(X_test_sd)
            
            fig = plt.figure()
            shap.summary_plot(shap_values_test_shap, X_test_sd)  
            st.pyplot(fig)
            
            fig = plt.figure()
            explanation = shap.Explanation(values=shap_values_test_shap,
                                 data=X_test_sd.values, # Assumant que  X_test est un DataFrame
                                 feature_names=X_test_sd.columns)
            shap.plots.bar(explanation)
            st.pyplot(fig)
            
            ### 1 CREATION D'UN EXPLANATION FILTRER SANS LES COLONNES POUR LESQUELLES NOUS ALLONS CALCULER LES MOYENNES

            #Étape 1 : Créer une liste des termes à exclure
            terms_to_exclude = ['month', 'weekday', 'job', 'poutcome', 'marital']

            #Étape 2 : Filtrer les colonnes qui ne contiennent pas les termes à exclure
            filtered_columns = [col for col in X_test_sd.columns if not any(term in col for term in terms_to_exclude)]

            #Étape 3 : Identifier les indices correspondants dans X_test_sd
            filtered_indices = [X_test_sd.columns.get_loc(col) for col in filtered_columns]
            shap_values_filtered = shap_values_test_shap[:, filtered_indices]

            # Étape 4 : On créé un nouvel Explanation avec les colonnes filtrées
            explanation_filtered = shap.Explanation(values=shap_values_filtered,
                                            data=X_test_sd.values[:, filtered_indices],  # Garder uniquement les colonnes correspondantes
                                            feature_names=filtered_columns)  # Les noms des features


            ###2 CRÉATION D'UN NOUVEAU EXPLANATION AVEC LES MOYENNES SHAP POUR LES COLONNES MONTH / WEEKDAY / POUTCOME / JOB / MARITAL

            #Fonction pour récupérer les moyennes SHAP en valeur absolue pour les colonnes qui nous intéressent
            def get_mean_shap_values(column_names, shap_values_test_shap):
                indices = [X_test_sd.columns.get_loc(col) for col in column_names]
                values = shap_values_test_shap[:, indices]
                return np.mean(np.abs(values), axis=0)

            #Étape 1 : On idenfie les colonnes que l'on recherche
            month_columns = [col for col in X_test_sd.columns if 'month' in col]
            weekday_columns = [col for col in X_test_sd.columns if 'weekday' in col]
            poutcome_columns = [col for col in X_test_sd.columns if 'poutcome' in col]
            job_columns = [col for col in X_test_sd.columns if 'job' in col]
            marital_columns = [col for col in X_test_sd.columns if 'marital' in col]

            #Étape 2 : On utiliser notre fonction pour calculer les moyennes des valeurs SHAP absolues
            mean_shap_month = get_mean_shap_values(month_columns, shap_values_test_shap)
            mean_shap_weekday = get_mean_shap_values(weekday_columns, shap_values_test_shap)
            mean_shap_poutcome = get_mean_shap_values(poutcome_columns, shap_values_test_shap)
            mean_shap_job = get_mean_shap_values(job_columns, shap_values_test_shap)
            mean_shap_marital = get_mean_shap_values(marital_columns, shap_values_test_shap)

            #Étape 3 : On combine les différentes moyennes et on les nomme
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

            #Étape 4 : On créé un nouvel Explanation avec les valeurs combinées
            explanation_combined = shap.Explanation(values=combined_values, data=np.array([[np.nan]] * len(combined_values)), feature_names=combined_feature_names)



            ###3 ON COMBINE LES 2 EXPLANTATION PRÉCÉDEMMENT CRÉÉS

            #Étape 1 : On récupére les nombre de lignes de explanation_filtered et on reshape explanation_combined pour avoir le même nombre de lignes
            num_samples = explanation_filtered.values.shape[0]
            combined_values_reshaped = np.repeat(np.array(explanation_combined.values)[:, np.newaxis], num_samples, axis=1).T

            #Étape 2: On concatenate les 2 explanations
            combined_values = np.concatenate([explanation_filtered.values, combined_values_reshaped], axis=1)

            #Étape 3: On combine le nom des colonnes provenant des 2 explanations
            combined_feature_names = (explanation_filtered.feature_names + explanation_combined.feature_names)

            #Étape 4: On créé un nouveau explanation avec les valeurs concatnées dans combined_values
            explanation_combined_new = shap.Explanation(values=combined_values,data=np.array([[np.nan]] * combined_values.shape[0]),feature_names=combined_feature_names,)

            fig = plt.figure(figsize=(10, 6))
            shap.plots.bar(explanation_combined_new, max_display=len(explanation_combined_new.feature_names))
            st.pyplot(fig)
            
            
            
            
            st.write("____________________________")
            

            # Charger le modèle sauvegardé
            @st.cache_resource
            def load_model():
                return joblib.load("xgboost_model_sd_OK.pkl")

            # Charger les SHAP values sauvegardées
            @st.cache_data
            def load_shap_values():
                return joblib.load("shap_values_xgboost_model_sd_OK.pkl")

            # Charger le modèle et les SHAP values
            model_xgboost_sd_OK = load_model()
            shap_values_xgboost_sd_ok = load_shap_values()

            st.write("test sur le  XGBOOST OK !!!!!!!! ")



            # Afficher un graphique SHAP sur Streamlit
            st.subheader("SHAP Summary Plot")
            fig = plt.figure()
            shap.summary_plot(shap_values_xgboost_sd_ok, X_test_sd)  # X_test_sd est ton jeu de test standardisé
            st.pyplot(fig)
            
            
            fig = plt.figure()
            shap.summary_plot(shap_values_xgboost_sd_ok, X_test_sd)  
            st.pyplot(fig)
            
            fig = plt.figure()
            explanation_ok = shap.Explanation(values=shap_values_xgboost_sd_ok,
                                 data=X_test_sd.values, # Assumant que  X_test est un DataFrame
                                 feature_names=X_test_sd.columns)
            shap.plots.bar(explanation_ok)
            st.pyplot(fig)
            

            ### 1 CREATION D'UN EXPLANATION FILTRER SANS LES COLONNES POUR LESQUELLES NOUS ALLONS CALCULER LES MOYENNES

            #Étape 1 : Créer une liste des termes à exclure
            terms_to_exclude = ['month', 'weekday', 'job', 'poutcome', 'marital']

            #Étape 2 : Filtrer les colonnes qui ne contiennent pas les termes à exclure
            filtered_columns = [col for col in X_test_sd.columns if not any(term in col for term in terms_to_exclude)]

            #Étape 3 : Identifier les indices correspondants dans X_test_sd
            filtered_indices = [X_test_sd.columns.get_loc(col) for col in filtered_columns]
            shap_values_filtered = shap_values_xgboost_sd_ok[:, filtered_indices]

            # Étape 4 : On créé un nouvel Explanation avec les colonnes filtrées
            explanation_filtered = shap.Explanation(values=shap_values_filtered,
                                            data=X_test_sd.values[:, filtered_indices],  # Garder uniquement les colonnes correspondantes
                                            feature_names=filtered_columns)  # Les noms des features


            ###2 CRÉATION D'UN NOUVEAU EXPLANATION AVEC LES MOYENNES SHAP POUR LES COLONNES MONTH / WEEKDAY / POUTCOME / JOB / MARITAL

            #Fonction pour récupérer les moyennes SHAP en valeur absolue pour les colonnes qui nous intéressent
            def get_mean_shap_values(column_names, shap_values_xgboost_sd_ok):
                indices = [X_test_sd.columns.get_loc(col) for col in column_names]
                values = shap_values_xgboost_sd_ok[:, indices]
                return np.mean(np.abs(values), axis=0)

            #Étape 1 : On idenfie les colonnes que l'on recherche
            month_columns = [col for col in X_test_sd.columns if 'month' in col]
            weekday_columns = [col for col in X_test_sd.columns if 'weekday' in col]
            poutcome_columns = [col for col in X_test_sd.columns if 'poutcome' in col]
            job_columns = [col for col in X_test_sd.columns if 'job' in col]
            marital_columns = [col for col in X_test_sd.columns if 'marital' in col]

            #Étape 2 : On utiliser notre fonction pour calculer les moyennes des valeurs SHAP absolues
            mean_shap_month = get_mean_shap_values(month_columns, shap_values_xgboost_sd_ok)
            mean_shap_weekday = get_mean_shap_values(weekday_columns, shap_values_xgboost_sd_ok)
            mean_shap_poutcome = get_mean_shap_values(poutcome_columns, shap_values_xgboost_sd_ok)
            mean_shap_job = get_mean_shap_values(job_columns, shap_values_xgboost_sd_ok)
            mean_shap_marital = get_mean_shap_values(marital_columns, shap_values_xgboost_sd_ok)

            #Étape 3 : On combine les différentes moyennes et on les nomme
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

            #Étape 4 : On créé un nouvel Explanation avec les valeurs combinées
            explanation_combined = shap.Explanation(values=combined_values, data=np.array([[np.nan]] * len(combined_values)), feature_names=combined_feature_names)



            ###3 ON COMBINE LES 2 EXPLANTATION PRÉCÉDEMMENT CRÉÉS

            #Étape 1 : On récupére les nombre de lignes de explanation_filtered et on reshape explanation_combined pour avoir le même nombre de lignes
            num_samples = explanation_filtered.values.shape[0]
            combined_values_reshaped = np.repeat(np.array(explanation_combined.values)[:, np.newaxis], num_samples, axis=1).T

            #Étape 2: On concatenate les 2 explanations
            combined_values = np.concatenate([explanation_filtered.values, combined_values_reshaped], axis=1)

            #Étape 3: On combine le nom des colonnes provenant des 2 explanations
            combined_feature_names = (explanation_filtered.feature_names + explanation_combined.feature_names)

            #Étape 4: On créé un nouveau explanation avec les valeurs concatnées dans combined_values
            explanation_combined_new = shap.Explanation(values=combined_values,data=np.array([[np.nan]] * combined_values.shape[0]),feature_names=combined_feature_names,)

            fig = plt.figure(figsize=(10, 6))
            shap.plots.bar(explanation_combined_new, max_display=len(explanation_combined_new.feature_names))
            st.pyplot(fig)


            st.write("____________________________")
            

            # Charger le modèle sauvegardé
            @st.cache_resource
            def load_model():
                return joblib.load("xgboost_model_sd_OK2_poub.pkl")

            # Charger les SHAP values sauvegardées
            @st.cache_data
            def load_shap_values():
                return joblib.load("shap_values_xgboost_model_sd_OK2_poub.pkl")

            # Charger le modèle et les SHAP values
            model_xgboost_sd_OK2_poub = load_model()
            shap_values_xgboost_sd_ok2_poub = load_shap_values()

            st.write("test sur le  XGBOOST OK !!!!!!!! ")



            # Afficher un graphique SHAP sur Streamlit
            st.subheader("SHAP Summary Plot")
            fig = plt.figure()
            shap.summary_plot(shap_values_xgboost_sd_ok2_poub, X_test_sd)  # X_test_sd est ton jeu de test standardisé
            st.pyplot(fig)
            
            
            fig = plt.figure()
            shap.summary_plot(shap_values_xgboost_sd_ok2_poub, X_test_sd)  
            st.pyplot(fig)
            
            fig = plt.figure()
            explanation_ok3 = shap.Explanation(values=shap_values_xgboost_sd_ok2_poub,
                                 data=X_test_sd.values, # Assumant que  X_test est un DataFrame
                                 feature_names=X_test_sd.columns)
            shap.plots.bar(explanation_ok3)
            st.pyplot(fig)
            

            ### 1 CREATION D'UN EXPLANATION FILTRER SANS LES COLONNES POUR LESQUELLES NOUS ALLONS CALCULER LES MOYENNES

            #Étape 1 : Créer une liste des termes à exclure
            terms_to_exclude = ['month', 'weekday', 'job', 'poutcome', 'marital']

            #Étape 2 : Filtrer les colonnes qui ne contiennent pas les termes à exclure
            filtered_columns = [col for col in X_test_sd.columns if not any(term in col for term in terms_to_exclude)]

            #Étape 3 : Identifier les indices correspondants dans X_test_sd
            filtered_indices = [X_test_sd.columns.get_loc(col) for col in filtered_columns]
            shap_values_filtered = shap_values_xgboost_sd_ok2_poub[:, filtered_indices]

            # Étape 4 : On créé un nouvel Explanation avec les colonnes filtrées
            explanation_filtered = shap.Explanation(values=shap_values_filtered,
                                            data=X_test_sd.values[:, filtered_indices],  # Garder uniquement les colonnes correspondantes
                                            feature_names=filtered_columns)  # Les noms des features


            ###2 CRÉATION D'UN NOUVEAU EXPLANATION AVEC LES MOYENNES SHAP POUR LES COLONNES MONTH / WEEKDAY / POUTCOME / JOB / MARITAL

            #Fonction pour récupérer les moyennes SHAP en valeur absolue pour les colonnes qui nous intéressent
            def get_mean_shap_values(column_names, shap_values_xgboost_sd_ok2_poub):
                indices = [X_test_sd.columns.get_loc(col) for col in column_names]
                values = shap_values_xgboost_sd_ok2_poub[:, indices]
                return np.mean(np.abs(values), axis=0)

            #Étape 1 : On idenfie les colonnes que l'on recherche
            month_columns = [col for col in X_test_sd.columns if 'month' in col]
            weekday_columns = [col for col in X_test_sd.columns if 'weekday' in col]
            poutcome_columns = [col for col in X_test_sd.columns if 'poutcome' in col]
            job_columns = [col for col in X_test_sd.columns if 'job' in col]
            marital_columns = [col for col in X_test_sd.columns if 'marital' in col]

            #Étape 2 : On utiliser notre fonction pour calculer les moyennes des valeurs SHAP absolues
            mean_shap_month = get_mean_shap_values(month_columns, shap_values_xgboost_sd_ok2_poub)
            mean_shap_weekday = get_mean_shap_values(weekday_columns, shap_values_xgboost_sd_ok2_poub)
            mean_shap_poutcome = get_mean_shap_values(poutcome_columns, shap_values_xgboost_sd_ok2_poub)
            mean_shap_job = get_mean_shap_values(job_columns, shap_values_xgboost_sd_ok2_poub)
            mean_shap_marital = get_mean_shap_values(marital_columns, shap_values_xgboost_sd_ok2_poub)

            #Étape 3 : On combine les différentes moyennes et on les nomme
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

            #Étape 4 : On créé un nouvel Explanation avec les valeurs combinées
            explanation_combined = shap.Explanation(values=combined_values, data=np.array([[np.nan]] * len(combined_values)), feature_names=combined_feature_names)



            ###3 ON COMBINE LES 2 EXPLANTATION PRÉCÉDEMMENT CRÉÉS

            #Étape 1 : On récupére les nombre de lignes de explanation_filtered et on reshape explanation_combined pour avoir le même nombre de lignes
            num_samples = explanation_filtered.values.shape[0]
            combined_values_reshaped = np.repeat(np.array(explanation_combined.values)[:, np.newaxis], num_samples, axis=1).T

            #Étape 2: On concatenate les 2 explanations
            combined_values = np.concatenate([explanation_filtered.values, combined_values_reshaped], axis=1)

            #Étape 3: On combine le nom des colonnes provenant des 2 explanations
            combined_feature_names = (explanation_filtered.feature_names + explanation_combined.feature_names)

            #Étape 4: On créé un nouveau explanation avec les valeurs concatnées dans combined_values
            explanation_combined_new = shap.Explanation(values=combined_values,data=np.array([[np.nan]] * combined_values.shape[0]),feature_names=combined_feature_names,)

            fig = plt.figure(figsize=(10, 6))
            shap.plots.bar(explanation_combined_new, max_display=len(explanation_combined_new.feature_names))
            st.pyplot(fig)
            
            
            st.write("____________________________")
            
            st.write("test ENCORE !!!!! ")

            XGBOOST_test_shap = XGBClassifier(gamma=0.08,colsample_bytree=0.87, learning_rate=0.39, max_depth=6, min_child_weight=1.2, n_estimators=32, reg_alpha=1.2, reg_lambda=1.9, scale_pos_weight=2.56, subsample=0.99, random_state=42)
            st.write("modif = 0.08 VS 0.05 + 0.87 VS 0.88")
            XGBOOST_test_shap.fit(X_train_sd, y_train_sd)
            y_pred = XGBOOST_test_shap.predict(X_test_sd)
            table_xgboost_test_shap = pd.crosstab(y_test_sd,y_pred, rownames=['Realité'], colnames=['Prédiction'])
            st.dataframe(table_xgboost_test_shap)
            st.write("Classification report :")
            report_dict_xgboost_test_shap = classification_report(y_test_sd, y_pred, output_dict=True)
            # Convertir le dictionnaire en DataFrame
            report_df_xgboost_test_shap = pd.DataFrame(report_dict_xgboost_test_shap).T
            st.dataframe(report_df_xgboost_test_shap)
            
            explainer = shap.TreeExplainer(XGBOOST_test_shap)
            shap_values_test_shap = explainer.shap_values(X_test_sd)
            
            fig = plt.figure()
            shap.summary_plot(shap_values_test_shap, X_test_sd)  
            st.pyplot(fig)
            
            fig = plt.figure()
            explanation = shap.Explanation(values=shap_values_test_shap,
                                 data=X_test_sd.values, # Assumant que  X_test est un DataFrame
                                 feature_names=X_test_sd.columns)
            shap.plots.bar(explanation)
            st.pyplot(fig)
            
            ### 1 CREATION D'UN EXPLANATION FILTRER SANS LES COLONNES POUR LESQUELLES NOUS ALLONS CALCULER LES MOYENNES

            #Étape 1 : Créer une liste des termes à exclure
            terms_to_exclude = ['month', 'weekday', 'job', 'poutcome', 'marital']

            #Étape 2 : Filtrer les colonnes qui ne contiennent pas les termes à exclure
            filtered_columns = [col for col in X_test_sd.columns if not any(term in col for term in terms_to_exclude)]

            #Étape 3 : Identifier les indices correspondants dans X_test_sd
            filtered_indices = [X_test_sd.columns.get_loc(col) for col in filtered_columns]
            shap_values_filtered = shap_values_test_shap[:, filtered_indices]

            # Étape 4 : On créé un nouvel Explanation avec les colonnes filtrées
            explanation_filtered = shap.Explanation(values=shap_values_filtered,
                                            data=X_test_sd.values[:, filtered_indices],  # Garder uniquement les colonnes correspondantes
                                            feature_names=filtered_columns)  # Les noms des features


            ###2 CRÉATION D'UN NOUVEAU EXPLANATION AVEC LES MOYENNES SHAP POUR LES COLONNES MONTH / WEEKDAY / POUTCOME / JOB / MARITAL

            #Fonction pour récupérer les moyennes SHAP en valeur absolue pour les colonnes qui nous intéressent
            def get_mean_shap_values(column_names, shap_values_test_shap):
                indices = [X_test_sd.columns.get_loc(col) for col in column_names]
                values = shap_values_test_shap[:, indices]
                return np.mean(np.abs(values), axis=0)

            #Étape 1 : On idenfie les colonnes que l'on recherche
            month_columns = [col for col in X_test_sd.columns if 'month' in col]
            weekday_columns = [col for col in X_test_sd.columns if 'weekday' in col]
            poutcome_columns = [col for col in X_test_sd.columns if 'poutcome' in col]
            job_columns = [col for col in X_test_sd.columns if 'job' in col]
            marital_columns = [col for col in X_test_sd.columns if 'marital' in col]

            #Étape 2 : On utiliser notre fonction pour calculer les moyennes des valeurs SHAP absolues
            mean_shap_month = get_mean_shap_values(month_columns, shap_values_test_shap)
            mean_shap_weekday = get_mean_shap_values(weekday_columns, shap_values_test_shap)
            mean_shap_poutcome = get_mean_shap_values(poutcome_columns, shap_values_test_shap)
            mean_shap_job = get_mean_shap_values(job_columns, shap_values_test_shap)
            mean_shap_marital = get_mean_shap_values(marital_columns, shap_values_test_shap)

            #Étape 3 : On combine les différentes moyennes et on les nomme
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

            #Étape 4 : On créé un nouvel Explanation avec les valeurs combinées
            explanation_combined = shap.Explanation(values=combined_values, data=np.array([[np.nan]] * len(combined_values)), feature_names=combined_feature_names)



            ###3 ON COMBINE LES 2 EXPLANTATION PRÉCÉDEMMENT CRÉÉS

            #Étape 1 : On récupére les nombre de lignes de explanation_filtered et on reshape explanation_combined pour avoir le même nombre de lignes
            num_samples = explanation_filtered.values.shape[0]
            combined_values_reshaped = np.repeat(np.array(explanation_combined.values)[:, np.newaxis], num_samples, axis=1).T

            #Étape 2: On concatenate les 2 explanations
            combined_values = np.concatenate([explanation_filtered.values, combined_values_reshaped], axis=1)

            #Étape 3: On combine le nom des colonnes provenant des 2 explanations
            combined_feature_names = (explanation_filtered.feature_names + explanation_combined.feature_names)

            #Étape 4: On créé un nouveau explanation avec les valeurs concatnées dans combined_values
            explanation_combined_new = shap.Explanation(values=combined_values,data=np.array([[np.nan]] * combined_values.shape[0]),feature_names=combined_feature_names,)

            fig = plt.figure(figsize=(10, 6))
            shap.plots.bar(explanation_combined_new, max_display=len(explanation_combined_new.feature_names))
            st.pyplot(fig)
            
            st.write("____________________________")

            
            st.write("test ENCORE !!!!! ")

            XGBOOST_test_shap = XGBClassifier(gamma=0.08,colsample_bytree=0.90, learning_rate=0.39, max_depth=6, min_child_weight=1.2, n_estimators=32, reg_alpha=1.2, reg_lambda=1.9, scale_pos_weight=2.57, subsample=0.99, random_state=42)
            st.write("modif = 0.09 VS 0.08 + 2.57 VS 2.56")
            XGBOOST_test_shap.fit(X_train_sd, y_train_sd)
            y_pred = XGBOOST_test_shap.predict(X_test_sd)
            table_xgboost_test_shap = pd.crosstab(y_test_sd,y_pred, rownames=['Realité'], colnames=['Prédiction'])
            st.dataframe(table_xgboost_test_shap)
            st.write("Classification report :")
            report_dict_xgboost_test_shap = classification_report(y_test_sd, y_pred, output_dict=True)
            # Convertir le dictionnaire en DataFrame
            report_df_xgboost_test_shap = pd.DataFrame(report_dict_xgboost_test_shap).T
            st.dataframe(report_df_xgboost_test_shap)
            
            explainer = shap.TreeExplainer(XGBOOST_test_shap)
            shap_values_test_shap = explainer.shap_values(X_test_sd)
            
            fig = plt.figure()
            shap.summary_plot(shap_values_test_shap, X_test_sd)  
            st.pyplot(fig)
            
            fig = plt.figure()
            explanation = shap.Explanation(values=shap_values_test_shap,
                                 data=X_test_sd.values, # Assumant que  X_test est un DataFrame
                                 feature_names=X_test_sd.columns)
            shap.plots.bar(explanation)
            st.pyplot(fig)
            
            ### 1 CREATION D'UN EXPLANATION FILTRER SANS LES COLONNES POUR LESQUELLES NOUS ALLONS CALCULER LES MOYENNES

            #Étape 1 : Créer une liste des termes à exclure
            terms_to_exclude = ['month', 'weekday', 'job', 'poutcome', 'marital']

            #Étape 2 : Filtrer les colonnes qui ne contiennent pas les termes à exclure
            filtered_columns = [col for col in X_test_sd.columns if not any(term in col for term in terms_to_exclude)]

            #Étape 3 : Identifier les indices correspondants dans X_test_sd
            filtered_indices = [X_test_sd.columns.get_loc(col) for col in filtered_columns]
            shap_values_filtered = shap_values_test_shap[:, filtered_indices]

            # Étape 4 : On créé un nouvel Explanation avec les colonnes filtrées
            explanation_filtered = shap.Explanation(values=shap_values_filtered,
                                            data=X_test_sd.values[:, filtered_indices],  # Garder uniquement les colonnes correspondantes
                                            feature_names=filtered_columns)  # Les noms des features


            ###2 CRÉATION D'UN NOUVEAU EXPLANATION AVEC LES MOYENNES SHAP POUR LES COLONNES MONTH / WEEKDAY / POUTCOME / JOB / MARITAL

            #Fonction pour récupérer les moyennes SHAP en valeur absolue pour les colonnes qui nous intéressent
            def get_mean_shap_values(column_names, shap_values_test_shap):
                indices = [X_test_sd.columns.get_loc(col) for col in column_names]
                values = shap_values_test_shap[:, indices]
                return np.mean(np.abs(values), axis=0)

            #Étape 1 : On idenfie les colonnes que l'on recherche
            month_columns = [col for col in X_test_sd.columns if 'month' in col]
            weekday_columns = [col for col in X_test_sd.columns if 'weekday' in col]
            poutcome_columns = [col for col in X_test_sd.columns if 'poutcome' in col]
            job_columns = [col for col in X_test_sd.columns if 'job' in col]
            marital_columns = [col for col in X_test_sd.columns if 'marital' in col]

            #Étape 2 : On utiliser notre fonction pour calculer les moyennes des valeurs SHAP absolues
            mean_shap_month = get_mean_shap_values(month_columns, shap_values_test_shap)
            mean_shap_weekday = get_mean_shap_values(weekday_columns, shap_values_test_shap)
            mean_shap_poutcome = get_mean_shap_values(poutcome_columns, shap_values_test_shap)
            mean_shap_job = get_mean_shap_values(job_columns, shap_values_test_shap)
            mean_shap_marital = get_mean_shap_values(marital_columns, shap_values_test_shap)

            #Étape 3 : On combine les différentes moyennes et on les nomme
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

            #Étape 4 : On créé un nouvel Explanation avec les valeurs combinées
            explanation_combined = shap.Explanation(values=combined_values, data=np.array([[np.nan]] * len(combined_values)), feature_names=combined_feature_names)



            ###3 ON COMBINE LES 2 EXPLANTATION PRÉCÉDEMMENT CRÉÉS

            #Étape 1 : On récupére les nombre de lignes de explanation_filtered et on reshape explanation_combined pour avoir le même nombre de lignes
            num_samples = explanation_filtered.values.shape[0]
            combined_values_reshaped = np.repeat(np.array(explanation_combined.values)[:, np.newaxis], num_samples, axis=1).T

            #Étape 2: On concatenate les 2 explanations
            combined_values = np.concatenate([explanation_filtered.values, combined_values_reshaped], axis=1)

            #Étape 3: On combine le nom des colonnes provenant des 2 explanations
            combined_feature_names = (explanation_filtered.feature_names + explanation_combined.feature_names)

            #Étape 4: On créé un nouveau explanation avec les valeurs concatnées dans combined_values
            explanation_combined_new = shap.Explanation(values=combined_values,data=np.array([[np.nan]] * combined_values.shape[0]),feature_names=combined_feature_names,)

            fig = plt.figure(figsize=(10, 6))
            shap.plots.bar(explanation_combined_new, max_display=len(explanation_combined_new.feature_names))
            st.pyplot(fig)
            
            st.write("____________________________")

        if page == pages[2] :
                        
            st.write("test ENCORE #1!!!!! ")

            XGBOOST_test_shap = XGBClassifier(gamma=0.08,colsample_bytree=0.87, learning_rate=0.39, max_depth=6, min_child_weight=1.2, n_estimators=32, reg_alpha=1.2, reg_lambda=1.9, scale_pos_weight=2.56, subsample=0.99, random_state=42)
            st.write("modif = 0.08 VS 0.05 + 0.87 VS 0.88")
            XGBOOST_test_shap.fit(X_train_sd, y_train_sd)
            y_pred = XGBOOST_test_shap.predict(X_test_sd)
            table_xgboost_test_shap = pd.crosstab(y_test_sd,y_pred, rownames=['Realité'], colnames=['Prédiction'])
            st.dataframe(table_xgboost_test_shap)
            st.write("Classification report :")
            report_dict_xgboost_test_shap = classification_report(y_test_sd, y_pred, output_dict=True)
            # Convertir le dictionnaire en DataFrame
            report_df_xgboost_test_shap = pd.DataFrame(report_dict_xgboost_test_shap).T
            st.dataframe(report_df_xgboost_test_shap)
            
            explainer = shap.TreeExplainer(XGBOOST_test_shap)
            shap_values_test_shap = explainer.shap_values(X_test_sd)
            
            fig = plt.figure()
            shap.summary_plot(shap_values_test_shap, X_test_sd)  
            st.pyplot(fig)
            
            fig = plt.figure()
            explanation = shap.Explanation(values=shap_values_test_shap,
                                 data=X_test_sd.values, # Assumant que  X_test est un DataFrame
                                 feature_names=X_test_sd.columns)
            shap.plots.bar(explanation)
            st.pyplot(fig)
            
            ### 1 CREATION D'UN EXPLANATION FILTRER SANS LES COLONNES POUR LESQUELLES NOUS ALLONS CALCULER LES MOYENNES

            #Étape 1 : Créer une liste des termes à exclure
            terms_to_exclude = ['month', 'weekday', 'job', 'poutcome', 'marital']

            #Étape 2 : Filtrer les colonnes qui ne contiennent pas les termes à exclure
            filtered_columns = [col for col in X_test_sd.columns if not any(term in col for term in terms_to_exclude)]

            #Étape 3 : Identifier les indices correspondants dans X_test_sd
            filtered_indices = [X_test_sd.columns.get_loc(col) for col in filtered_columns]
            shap_values_filtered = shap_values_test_shap[:, filtered_indices]

            # Étape 4 : On créé un nouvel Explanation avec les colonnes filtrées
            explanation_filtered = shap.Explanation(values=shap_values_filtered,
                                            data=X_test_sd.values[:, filtered_indices],  # Garder uniquement les colonnes correspondantes
                                            feature_names=filtered_columns)  # Les noms des features


            ###2 CRÉATION D'UN NOUVEAU EXPLANATION AVEC LES MOYENNES SHAP POUR LES COLONNES MONTH / WEEKDAY / POUTCOME / JOB / MARITAL

            #Fonction pour récupérer les moyennes SHAP en valeur absolue pour les colonnes qui nous intéressent
            def get_mean_shap_values(column_names, shap_values_test_shap):
                indices = [X_test_sd.columns.get_loc(col) for col in column_names]
                values = shap_values_test_shap[:, indices]
                return np.mean(np.abs(values), axis=0)

            #Étape 1 : On idenfie les colonnes que l'on recherche
            month_columns = [col for col in X_test_sd.columns if 'month' in col]
            weekday_columns = [col for col in X_test_sd.columns if 'weekday' in col]
            poutcome_columns = [col for col in X_test_sd.columns if 'poutcome' in col]
            job_columns = [col for col in X_test_sd.columns if 'job' in col]
            marital_columns = [col for col in X_test_sd.columns if 'marital' in col]

            #Étape 2 : On utiliser notre fonction pour calculer les moyennes des valeurs SHAP absolues
            mean_shap_month = get_mean_shap_values(month_columns, shap_values_test_shap)
            mean_shap_weekday = get_mean_shap_values(weekday_columns, shap_values_test_shap)
            mean_shap_poutcome = get_mean_shap_values(poutcome_columns, shap_values_test_shap)
            mean_shap_job = get_mean_shap_values(job_columns, shap_values_test_shap)
            mean_shap_marital = get_mean_shap_values(marital_columns, shap_values_test_shap)

            #Étape 3 : On combine les différentes moyennes et on les nomme
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

            #Étape 4 : On créé un nouvel Explanation avec les valeurs combinées
            explanation_combined = shap.Explanation(values=combined_values, data=np.array([[np.nan]] * len(combined_values)), feature_names=combined_feature_names)



            ###3 ON COMBINE LES 2 EXPLANTATION PRÉCÉDEMMENT CRÉÉS

            #Étape 1 : On récupére les nombre de lignes de explanation_filtered et on reshape explanation_combined pour avoir le même nombre de lignes
            num_samples = explanation_filtered.values.shape[0]
            combined_values_reshaped = np.repeat(np.array(explanation_combined.values)[:, np.newaxis], num_samples, axis=1).T

            #Étape 2: On concatenate les 2 explanations
            combined_values = np.concatenate([explanation_filtered.values, combined_values_reshaped], axis=1)

            #Étape 3: On combine le nom des colonnes provenant des 2 explanations
            combined_feature_names = (explanation_filtered.feature_names + explanation_combined.feature_names)

            #Étape 4: On créé un nouveau explanation avec les valeurs concatnées dans combined_values
            explanation_combined_new = shap.Explanation(values=combined_values,data=np.array([[np.nan]] * combined_values.shape[0]),feature_names=combined_feature_names,)

            fig = plt.figure(figsize=(10, 6))
            shap.plots.bar(explanation_combined_new, max_display=len(explanation_combined_new.feature_names))
            st.pyplot(fig)
            
            st.write("____________________________")

            
            st.write("test ENCORE #2!!!!! ")

            XGBOOST_test_shap = XGBClassifier(gamma=0.09,colsample_bytree=0.90, learning_rate=0.39, max_depth=6, min_child_weight=1.2, n_estimators=32, reg_alpha=1.2, reg_lambda=1.9, scale_pos_weight=2.57, subsample=0.99, random_state=42)
            st.write("modif = 0.09 VS 0.08 + 2.57 VS 2.56, 0.90 vs 0.87")
            XGBOOST_test_shap.fit(X_train_sd, y_train_sd)
            y_pred = XGBOOST_test_shap.predict(X_test_sd)
            table_xgboost_test_shap = pd.crosstab(y_test_sd,y_pred, rownames=['Realité'], colnames=['Prédiction'])
            st.dataframe(table_xgboost_test_shap)
            st.write("Classification report :")
            report_dict_xgboost_test_shap = classification_report(y_test_sd, y_pred, output_dict=True)
            # Convertir le dictionnaire en DataFrame
            report_df_xgboost_test_shap = pd.DataFrame(report_dict_xgboost_test_shap).T
            st.dataframe(report_df_xgboost_test_shap)
            
            explainer = shap.TreeExplainer(XGBOOST_test_shap)
            shap_values_test_shap = explainer.shap_values(X_test_sd)
            
            fig = plt.figure()
            shap.summary_plot(shap_values_test_shap, X_test_sd)  
            st.pyplot(fig)
            
            fig = plt.figure()
            explanation = shap.Explanation(values=shap_values_test_shap,
                                 data=X_test_sd.values, # Assumant que  X_test est un DataFrame
                                 feature_names=X_test_sd.columns)
            shap.plots.bar(explanation)
            st.pyplot(fig)
            
            ### 1 CREATION D'UN EXPLANATION FILTRER SANS LES COLONNES POUR LESQUELLES NOUS ALLONS CALCULER LES MOYENNES

            #Étape 1 : Créer une liste des termes à exclure
            terms_to_exclude = ['month', 'weekday', 'job', 'poutcome', 'marital']

            #Étape 2 : Filtrer les colonnes qui ne contiennent pas les termes à exclure
            filtered_columns = [col for col in X_test_sd.columns if not any(term in col for term in terms_to_exclude)]

            #Étape 3 : Identifier les indices correspondants dans X_test_sd
            filtered_indices = [X_test_sd.columns.get_loc(col) for col in filtered_columns]
            shap_values_filtered = shap_values_test_shap[:, filtered_indices]

            # Étape 4 : On créé un nouvel Explanation avec les colonnes filtrées
            explanation_filtered = shap.Explanation(values=shap_values_filtered,
                                            data=X_test_sd.values[:, filtered_indices],  # Garder uniquement les colonnes correspondantes
                                            feature_names=filtered_columns)  # Les noms des features


            ###2 CRÉATION D'UN NOUVEAU EXPLANATION AVEC LES MOYENNES SHAP POUR LES COLONNES MONTH / WEEKDAY / POUTCOME / JOB / MARITAL

            #Fonction pour récupérer les moyennes SHAP en valeur absolue pour les colonnes qui nous intéressent
            def get_mean_shap_values(column_names, shap_values_test_shap):
                indices = [X_test_sd.columns.get_loc(col) for col in column_names]
                values = shap_values_test_shap[:, indices]
                return np.mean(np.abs(values), axis=0)

            #Étape 1 : On idenfie les colonnes que l'on recherche
            month_columns = [col for col in X_test_sd.columns if 'month' in col]
            weekday_columns = [col for col in X_test_sd.columns if 'weekday' in col]
            poutcome_columns = [col for col in X_test_sd.columns if 'poutcome' in col]
            job_columns = [col for col in X_test_sd.columns if 'job' in col]
            marital_columns = [col for col in X_test_sd.columns if 'marital' in col]

            #Étape 2 : On utiliser notre fonction pour calculer les moyennes des valeurs SHAP absolues
            mean_shap_month = get_mean_shap_values(month_columns, shap_values_test_shap)
            mean_shap_weekday = get_mean_shap_values(weekday_columns, shap_values_test_shap)
            mean_shap_poutcome = get_mean_shap_values(poutcome_columns, shap_values_test_shap)
            mean_shap_job = get_mean_shap_values(job_columns, shap_values_test_shap)
            mean_shap_marital = get_mean_shap_values(marital_columns, shap_values_test_shap)

            #Étape 3 : On combine les différentes moyennes et on les nomme
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

            #Étape 4 : On créé un nouvel Explanation avec les valeurs combinées
            explanation_combined = shap.Explanation(values=combined_values, data=np.array([[np.nan]] * len(combined_values)), feature_names=combined_feature_names)



            ###3 ON COMBINE LES 2 EXPLANTATION PRÉCÉDEMMENT CRÉÉS

            #Étape 1 : On récupére les nombre de lignes de explanation_filtered et on reshape explanation_combined pour avoir le même nombre de lignes
            num_samples = explanation_filtered.values.shape[0]
            combined_values_reshaped = np.repeat(np.array(explanation_combined.values)[:, np.newaxis], num_samples, axis=1).T

            #Étape 2: On concatenate les 2 explanations
            combined_values = np.concatenate([explanation_filtered.values, combined_values_reshaped], axis=1)

            #Étape 3: On combine le nom des colonnes provenant des 2 explanations
            combined_feature_names = (explanation_filtered.feature_names + explanation_combined.feature_names)

            #Étape 4: On créé un nouveau explanation avec les valeurs concatnées dans combined_values
            explanation_combined_new = shap.Explanation(values=combined_values,data=np.array([[np.nan]] * combined_values.shape[0]),feature_names=combined_feature_names,)

            fig = plt.figure(figsize=(10, 6))
            shap.plots.bar(explanation_combined_new, max_display=len(explanation_combined_new.feature_names))
            st.pyplot(fig)
            
            st.write("____________________________")
            
            st.write("test ENCORE #3!!!!! ")

            XGBOOST_test_shap = XGBClassifier(gamma=0.09,colsample_bytree=0.90, learning_rate=0.40, max_depth=6, min_child_weight=1.2, n_estimators=35, reg_alpha=1.2, reg_lambda=1.9, scale_pos_weight=2.71, subsample=0.87, random_state=42)
            st.write("modif = 0.09 VS 0.05 + 0.87 VS 0.88 + 2.6 vs 2.56 + 0.90 vs 0.87 + 0.40 vs 0.39 + 34 vs 32 ")
            XGBOOST_test_shap.fit(X_train_sd, y_train_sd)
            y_pred = XGBOOST_test_shap.predict(X_test_sd)
            table_xgboost_test_shap = pd.crosstab(y_test_sd,y_pred, rownames=['Realité'], colnames=['Prédiction'])
            st.dataframe(table_xgboost_test_shap)
            st.write("Classification report :")
            report_dict_xgboost_test_shap = classification_report(y_test_sd, y_pred, output_dict=True)
            # Convertir le dictionnaire en DataFrame
            report_df_xgboost_test_shap = pd.DataFrame(report_dict_xgboost_test_shap).T
            st.dataframe(report_df_xgboost_test_shap)
            
            explainer = shap.TreeExplainer(XGBOOST_test_shap)
            shap_values_test_shap = explainer.shap_values(X_test_sd)
            
            fig = plt.figure()
            shap.summary_plot(shap_values_test_shap, X_test_sd)  
            st.pyplot(fig)
            
            fig = plt.figure()
            explanation = shap.Explanation(values=shap_values_test_shap,
                                 data=X_test_sd.values, # Assumant que  X_test est un DataFrame
                                 feature_names=X_test_sd.columns)
            shap.plots.bar(explanation)
            st.pyplot(fig)
            
            ### 1 CREATION D'UN EXPLANATION FILTRER SANS LES COLONNES POUR LESQUELLES NOUS ALLONS CALCULER LES MOYENNES

            #Étape 1 : Créer une liste des termes à exclure
            terms_to_exclude = ['month', 'weekday', 'job', 'poutcome', 'marital']

            #Étape 2 : Filtrer les colonnes qui ne contiennent pas les termes à exclure
            filtered_columns = [col for col in X_test_sd.columns if not any(term in col for term in terms_to_exclude)]

            #Étape 3 : Identifier les indices correspondants dans X_test_sd
            filtered_indices = [X_test_sd.columns.get_loc(col) for col in filtered_columns]
            shap_values_filtered = shap_values_test_shap[:, filtered_indices]

            # Étape 4 : On créé un nouvel Explanation avec les colonnes filtrées
            explanation_filtered = shap.Explanation(values=shap_values_filtered,
                                            data=X_test_sd.values[:, filtered_indices],  # Garder uniquement les colonnes correspondantes
                                            feature_names=filtered_columns)  # Les noms des features


            ###2 CRÉATION D'UN NOUVEAU EXPLANATION AVEC LES MOYENNES SHAP POUR LES COLONNES MONTH / WEEKDAY / POUTCOME / JOB / MARITAL

            #Fonction pour récupérer les moyennes SHAP en valeur absolue pour les colonnes qui nous intéressent
            def get_mean_shap_values(column_names, shap_values_test_shap):
                indices = [X_test_sd.columns.get_loc(col) for col in column_names]
                values = shap_values_test_shap[:, indices]
                return np.mean(np.abs(values), axis=0)

            #Étape 1 : On idenfie les colonnes que l'on recherche
            month_columns = [col for col in X_test_sd.columns if 'month' in col]
            weekday_columns = [col for col in X_test_sd.columns if 'weekday' in col]
            poutcome_columns = [col for col in X_test_sd.columns if 'poutcome' in col]
            job_columns = [col for col in X_test_sd.columns if 'job' in col]
            marital_columns = [col for col in X_test_sd.columns if 'marital' in col]

            #Étape 2 : On utiliser notre fonction pour calculer les moyennes des valeurs SHAP absolues
            mean_shap_month = get_mean_shap_values(month_columns, shap_values_test_shap)
            mean_shap_weekday = get_mean_shap_values(weekday_columns, shap_values_test_shap)
            mean_shap_poutcome = get_mean_shap_values(poutcome_columns, shap_values_test_shap)
            mean_shap_job = get_mean_shap_values(job_columns, shap_values_test_shap)
            mean_shap_marital = get_mean_shap_values(marital_columns, shap_values_test_shap)

            #Étape 3 : On combine les différentes moyennes et on les nomme
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

            #Étape 4 : On créé un nouvel Explanation avec les valeurs combinées
            explanation_combined = shap.Explanation(values=combined_values, data=np.array([[np.nan]] * len(combined_values)), feature_names=combined_feature_names)



            ###3 ON COMBINE LES 2 EXPLANTATION PRÉCÉDEMMENT CRÉÉS

            #Étape 1 : On récupére les nombre de lignes de explanation_filtered et on reshape explanation_combined pour avoir le même nombre de lignes
            num_samples = explanation_filtered.values.shape[0]
            combined_values_reshaped = np.repeat(np.array(explanation_combined.values)[:, np.newaxis], num_samples, axis=1).T

            #Étape 2: On concatenate les 2 explanations
            combined_values = np.concatenate([explanation_filtered.values, combined_values_reshaped], axis=1)

            #Étape 3: On combine le nom des colonnes provenant des 2 explanations
            combined_feature_names = (explanation_filtered.feature_names + explanation_combined.feature_names)

            #Étape 4: On créé un nouveau explanation avec les valeurs concatnées dans combined_values
            explanation_combined_new = shap.Explanation(values=combined_values,data=np.array([[np.nan]] * combined_values.shape[0]),feature_names=combined_feature_names,)

            fig = plt.figure(figsize=(10, 6))
            shap.plots.bar(explanation_combined_new, max_display=len(explanation_combined_new.feature_names))
            st.pyplot(fig)
            
            st.write("____________________________")

            st.write("test ENCORE #4!!!!! ")

            XGBOOST_test_shap = XGBClassifier(gamma=1,colsample_bytree=0.89, learning_rate=0.40, max_depth=6, min_child_weight=1.2, n_estimators=35, reg_alpha=1.2, reg_lambda=1.9, scale_pos_weight=2.75, subsample=0.87, random_state=42)
            st.write("modif = 0.08 vs 0.09 + 0.89 vs 0.90 + 2.75 vs 2.71")
            XGBOOST_test_shap.fit(X_train_sd, y_train_sd)
            y_pred = XGBOOST_test_shap.predict(X_test_sd)
            table_xgboost_test_shap = pd.crosstab(y_test_sd,y_pred, rownames=['Realité'], colnames=['Prédiction'])
            st.dataframe(table_xgboost_test_shap)
            st.write("Classification report :")
            report_dict_xgboost_test_shap = classification_report(y_test_sd, y_pred, output_dict=True)
            # Convertir le dictionnaire en DataFrame
            report_df_xgboost_test_shap = pd.DataFrame(report_dict_xgboost_test_shap).T
            st.dataframe(report_df_xgboost_test_shap)
            
            explainer = shap.TreeExplainer(XGBOOST_test_shap)
            shap_values_test_shap = explainer.shap_values(X_test_sd)
            
            fig = plt.figure()
            shap.summary_plot(shap_values_test_shap, X_test_sd)  
            st.pyplot(fig)
            
            fig = plt.figure()
            explanation = shap.Explanation(values=shap_values_test_shap,
                                 data=X_test_sd.values, # Assumant que  X_test est un DataFrame
                                 feature_names=X_test_sd.columns)
            shap.plots.bar(explanation)
            st.pyplot(fig)
            
            ### 1 CREATION D'UN EXPLANATION FILTRER SANS LES COLONNES POUR LESQUELLES NOUS ALLONS CALCULER LES MOYENNES

            #Étape 1 : Créer une liste des termes à exclure
            terms_to_exclude = ['month', 'weekday', 'job', 'poutcome', 'marital']

            #Étape 2 : Filtrer les colonnes qui ne contiennent pas les termes à exclure
            filtered_columns = [col for col in X_test_sd.columns if not any(term in col for term in terms_to_exclude)]

            #Étape 3 : Identifier les indices correspondants dans X_test_sd
            filtered_indices = [X_test_sd.columns.get_loc(col) for col in filtered_columns]
            shap_values_filtered = shap_values_test_shap[:, filtered_indices]

            # Étape 4 : On créé un nouvel Explanation avec les colonnes filtrées
            explanation_filtered = shap.Explanation(values=shap_values_filtered,
                                            data=X_test_sd.values[:, filtered_indices],  # Garder uniquement les colonnes correspondantes
                                            feature_names=filtered_columns)  # Les noms des features


            ###2 CRÉATION D'UN NOUVEAU EXPLANATION AVEC LES MOYENNES SHAP POUR LES COLONNES MONTH / WEEKDAY / POUTCOME / JOB / MARITAL

            #Fonction pour récupérer les moyennes SHAP en valeur absolue pour les colonnes qui nous intéressent
            def get_mean_shap_values(column_names, shap_values_test_shap):
                indices = [X_test_sd.columns.get_loc(col) for col in column_names]
                values = shap_values_test_shap[:, indices]
                return np.mean(np.abs(values), axis=0)

            #Étape 1 : On idenfie les colonnes que l'on recherche
            month_columns = [col for col in X_test_sd.columns if 'month' in col]
            weekday_columns = [col for col in X_test_sd.columns if 'weekday' in col]
            poutcome_columns = [col for col in X_test_sd.columns if 'poutcome' in col]
            job_columns = [col for col in X_test_sd.columns if 'job' in col]
            marital_columns = [col for col in X_test_sd.columns if 'marital' in col]

            #Étape 2 : On utiliser notre fonction pour calculer les moyennes des valeurs SHAP absolues
            mean_shap_month = get_mean_shap_values(month_columns, shap_values_test_shap)
            mean_shap_weekday = get_mean_shap_values(weekday_columns, shap_values_test_shap)
            mean_shap_poutcome = get_mean_shap_values(poutcome_columns, shap_values_test_shap)
            mean_shap_job = get_mean_shap_values(job_columns, shap_values_test_shap)
            mean_shap_marital = get_mean_shap_values(marital_columns, shap_values_test_shap)

            #Étape 3 : On combine les différentes moyennes et on les nomme
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

            #Étape 4 : On créé un nouvel Explanation avec les valeurs combinées
            explanation_combined = shap.Explanation(values=combined_values, data=np.array([[np.nan]] * len(combined_values)), feature_names=combined_feature_names)



            ###3 ON COMBINE LES 2 EXPLANTATION PRÉCÉDEMMENT CRÉÉS

            #Étape 1 : On récupére les nombre de lignes de explanation_filtered et on reshape explanation_combined pour avoir le même nombre de lignes
            num_samples = explanation_filtered.values.shape[0]
            combined_values_reshaped = np.repeat(np.array(explanation_combined.values)[:, np.newaxis], num_samples, axis=1).T

            #Étape 2: On concatenate les 2 explanations
            combined_values = np.concatenate([explanation_filtered.values, combined_values_reshaped], axis=1)

            #Étape 3: On combine le nom des colonnes provenant des 2 explanations
            combined_feature_names = (explanation_filtered.feature_names + explanation_combined.feature_names)

            #Étape 4: On créé un nouveau explanation avec les valeurs concatnées dans combined_values
            explanation_combined_new = shap.Explanation(values=combined_values,data=np.array([[np.nan]] * combined_values.shape[0]),feature_names=combined_feature_names,)

            fig = plt.figure(figsize=(10, 6))
            shap.plots.bar(explanation_combined_new, max_display=len(explanation_combined_new.feature_names))
            st.pyplot(fig)
            
            st.write("____________________________")
            

            st.write("test ENCORE #5!!!!! ")

            XGBOOST_test_shap = XGBClassifier(gamma=0.1,colsample_bytree=0.7, learning_rate=0.45, max_depth=6,  min_child_weight=1.7, n_estimators=36, reg_alpha=1, reg_lambda=1.9, scale_pos_weight=2.6, subsample=0.99, random_state=42)
            st.write("modif = 0.08 vs 0.09 + 0.89 vs 0.90 + 2.75 vs 2.71")
            XGBOOST_test_shap.fit(X_train_sd, y_train_sd)
            y_pred = XGBOOST_test_shap.predict(X_test_sd)
            table_xgboost_test_shap = pd.crosstab(y_test_sd,y_pred, rownames=['Realité'], colnames=['Prédiction'])
            st.dataframe(table_xgboost_test_shap)
            st.write("Classification report :")
            report_dict_xgboost_test_shap = classification_report(y_test_sd, y_pred, output_dict=True)
            # Convertir le dictionnaire en DataFrame
            report_df_xgboost_test_shap = pd.DataFrame(report_dict_xgboost_test_shap).T
            st.dataframe(report_df_xgboost_test_shap)
            
            explainer = shap.TreeExplainer(XGBOOST_test_shap)
            shap_values_test_shap = explainer.shap_values(X_test_sd)
            
            fig = plt.figure()
            shap.summary_plot(shap_values_test_shap, X_test_sd)  
            st.pyplot(fig)
            
            fig = plt.figure()
            explanation = shap.Explanation(values=shap_values_test_shap,
                                 data=X_test_sd.values, # Assumant que  X_test est un DataFrame
                                 feature_names=X_test_sd.columns)
            shap.plots.bar(explanation)
            st.pyplot(fig)
            
            ### 1 CREATION D'UN EXPLANATION FILTRER SANS LES COLONNES POUR LESQUELLES NOUS ALLONS CALCULER LES MOYENNES

            #Étape 1 : Créer une liste des termes à exclure
            terms_to_exclude = ['month', 'weekday', 'job', 'poutcome', 'marital']

            #Étape 2 : Filtrer les colonnes qui ne contiennent pas les termes à exclure
            filtered_columns = [col for col in X_test_sd.columns if not any(term in col for term in terms_to_exclude)]

            #Étape 3 : Identifier les indices correspondants dans X_test_sd
            filtered_indices = [X_test_sd.columns.get_loc(col) for col in filtered_columns]
            shap_values_filtered = shap_values_test_shap[:, filtered_indices]

            # Étape 4 : On créé un nouvel Explanation avec les colonnes filtrées
            explanation_filtered = shap.Explanation(values=shap_values_filtered,
                                            data=X_test_sd.values[:, filtered_indices],  # Garder uniquement les colonnes correspondantes
                                            feature_names=filtered_columns)  # Les noms des features


            ###2 CRÉATION D'UN NOUVEAU EXPLANATION AVEC LES MOYENNES SHAP POUR LES COLONNES MONTH / WEEKDAY / POUTCOME / JOB / MARITAL

            #Fonction pour récupérer les moyennes SHAP en valeur absolue pour les colonnes qui nous intéressent
            def get_mean_shap_values(column_names, shap_values_test_shap):
                indices = [X_test_sd.columns.get_loc(col) for col in column_names]
                values = shap_values_test_shap[:, indices]
                return np.mean(np.abs(values), axis=0)

            #Étape 1 : On idenfie les colonnes que l'on recherche
            month_columns = [col for col in X_test_sd.columns if 'month' in col]
            weekday_columns = [col for col in X_test_sd.columns if 'weekday' in col]
            poutcome_columns = [col for col in X_test_sd.columns if 'poutcome' in col]
            job_columns = [col for col in X_test_sd.columns if 'job' in col]
            marital_columns = [col for col in X_test_sd.columns if 'marital' in col]

            #Étape 2 : On utiliser notre fonction pour calculer les moyennes des valeurs SHAP absolues
            mean_shap_month = get_mean_shap_values(month_columns, shap_values_test_shap)
            mean_shap_weekday = get_mean_shap_values(weekday_columns, shap_values_test_shap)
            mean_shap_poutcome = get_mean_shap_values(poutcome_columns, shap_values_test_shap)
            mean_shap_job = get_mean_shap_values(job_columns, shap_values_test_shap)
            mean_shap_marital = get_mean_shap_values(marital_columns, shap_values_test_shap)

            #Étape 3 : On combine les différentes moyennes et on les nomme
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

            #Étape 4 : On créé un nouvel Explanation avec les valeurs combinées
            explanation_combined = shap.Explanation(values=combined_values, data=np.array([[np.nan]] * len(combined_values)), feature_names=combined_feature_names)



            ###3 ON COMBINE LES 2 EXPLANTATION PRÉCÉDEMMENT CRÉÉS

            #Étape 1 : On récupére les nombre de lignes de explanation_filtered et on reshape explanation_combined pour avoir le même nombre de lignes
            num_samples = explanation_filtered.values.shape[0]
            combined_values_reshaped = np.repeat(np.array(explanation_combined.values)[:, np.newaxis], num_samples, axis=1).T

            #Étape 2: On concatenate les 2 explanations
            combined_values = np.concatenate([explanation_filtered.values, combined_values_reshaped], axis=1)

            #Étape 3: On combine le nom des colonnes provenant des 2 explanations
            combined_feature_names = (explanation_filtered.feature_names + explanation_combined.feature_names)

            #Étape 4: On créé un nouveau explanation avec les valeurs concatnées dans combined_values
            explanation_combined_new = shap.Explanation(values=combined_values,data=np.array([[np.nan]] * combined_values.shape[0]),feature_names=combined_feature_names,)

            fig = plt.figure(figsize=(10, 6))
            shap.plots.bar(explanation_combined_new, max_display=len(explanation_combined_new.feature_names))
            st.pyplot(fig)
            
            st.write("____________________________")
            st.write("test ENCORE #6!!!!! ")

            XGBOOST_test_shap = XGBClassifier(gamma=0.1,colsample_bytree=0.90, learning_rate=0.3, max_delta_step =1, max_depth=6,  min_child_weight=1.7, n_estimators=50, scale_pos_weight=2.7, subsample=0.9, random_state=42)
            st.write("modif = 0.08 vs 0.09 + 0.89 vs 0.90 + 2.75 vs 2.71")
            XGBOOST_test_shap.fit(X_train_sd, y_train_sd)
            y_pred = XGBOOST_test_shap.predict(X_test_sd)
            table_xgboost_test_shap = pd.crosstab(y_test_sd,y_pred, rownames=['Realité'], colnames=['Prédiction'])
            st.dataframe(table_xgboost_test_shap)
            st.write("Classification report :")
            report_dict_xgboost_test_shap = classification_report(y_test_sd, y_pred, output_dict=True)
            # Convertir le dictionnaire en DataFrame
            report_df_xgboost_test_shap = pd.DataFrame(report_dict_xgboost_test_shap).T
            st.dataframe(report_df_xgboost_test_shap)
            
            explainer = shap.TreeExplainer(XGBOOST_test_shap)
            shap_values_test_shap = explainer.shap_values(X_test_sd)
            
            fig = plt.figure()
            shap.summary_plot(shap_values_test_shap, X_test_sd)  
            st.pyplot(fig)
            
            fig = plt.figure()
            explanation = shap.Explanation(values=shap_values_test_shap,
                                 data=X_test_sd.values, # Assumant que  X_test est un DataFrame
                                 feature_names=X_test_sd.columns)
            shap.plots.bar(explanation)
            st.pyplot(fig)
            
            ### 1 CREATION D'UN EXPLANATION FILTRER SANS LES COLONNES POUR LESQUELLES NOUS ALLONS CALCULER LES MOYENNES

            #Étape 1 : Créer une liste des termes à exclure
            terms_to_exclude = ['month', 'weekday', 'job', 'poutcome', 'marital']

            #Étape 2 : Filtrer les colonnes qui ne contiennent pas les termes à exclure
            filtered_columns = [col for col in X_test_sd.columns if not any(term in col for term in terms_to_exclude)]

            #Étape 3 : Identifier les indices correspondants dans X_test_sd
            filtered_indices = [X_test_sd.columns.get_loc(col) for col in filtered_columns]
            shap_values_filtered = shap_values_test_shap[:, filtered_indices]

            # Étape 4 : On créé un nouvel Explanation avec les colonnes filtrées
            explanation_filtered = shap.Explanation(values=shap_values_filtered,
                                            data=X_test_sd.values[:, filtered_indices],  # Garder uniquement les colonnes correspondantes
                                            feature_names=filtered_columns)  # Les noms des features


            ###2 CRÉATION D'UN NOUVEAU EXPLANATION AVEC LES MOYENNES SHAP POUR LES COLONNES MONTH / WEEKDAY / POUTCOME / JOB / MARITAL

            #Fonction pour récupérer les moyennes SHAP en valeur absolue pour les colonnes qui nous intéressent
            def get_mean_shap_values(column_names, shap_values_test_shap):
                indices = [X_test_sd.columns.get_loc(col) for col in column_names]
                values = shap_values_test_shap[:, indices]
                return np.mean(np.abs(values), axis=0)

            #Étape 1 : On idenfie les colonnes que l'on recherche
            month_columns = [col for col in X_test_sd.columns if 'month' in col]
            weekday_columns = [col for col in X_test_sd.columns if 'weekday' in col]
            poutcome_columns = [col for col in X_test_sd.columns if 'poutcome' in col]
            job_columns = [col for col in X_test_sd.columns if 'job' in col]
            marital_columns = [col for col in X_test_sd.columns if 'marital' in col]

            #Étape 2 : On utiliser notre fonction pour calculer les moyennes des valeurs SHAP absolues
            mean_shap_month = get_mean_shap_values(month_columns, shap_values_test_shap)
            mean_shap_weekday = get_mean_shap_values(weekday_columns, shap_values_test_shap)
            mean_shap_poutcome = get_mean_shap_values(poutcome_columns, shap_values_test_shap)
            mean_shap_job = get_mean_shap_values(job_columns, shap_values_test_shap)
            mean_shap_marital = get_mean_shap_values(marital_columns, shap_values_test_shap)

            #Étape 3 : On combine les différentes moyennes et on les nomme
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

            #Étape 4 : On créé un nouvel Explanation avec les valeurs combinées
            explanation_combined = shap.Explanation(values=combined_values, data=np.array([[np.nan]] * len(combined_values)), feature_names=combined_feature_names)



            ###3 ON COMBINE LES 2 EXPLANTATION PRÉCÉDEMMENT CRÉÉS

            #Étape 1 : On récupére les nombre de lignes de explanation_filtered et on reshape explanation_combined pour avoir le même nombre de lignes
            num_samples = explanation_filtered.values.shape[0]
            combined_values_reshaped = np.repeat(np.array(explanation_combined.values)[:, np.newaxis], num_samples, axis=1).T

            #Étape 2: On concatenate les 2 explanations
            combined_values = np.concatenate([explanation_filtered.values, combined_values_reshaped], axis=1)

            #Étape 3: On combine le nom des colonnes provenant des 2 explanations
            combined_feature_names = (explanation_filtered.feature_names + explanation_combined.feature_names)

            #Étape 4: On créé un nouveau explanation avec les valeurs concatnées dans combined_values
            explanation_combined_new = shap.Explanation(values=combined_values,data=np.array([[np.nan]] * combined_values.shape[0]),feature_names=combined_feature_names,)

            fig = plt.figure(figsize=(10, 6))
            shap.plots.bar(explanation_combined_new, max_display=len(explanation_combined_new.feature_names))
            st.pyplot(fig)
            

if selected == 'Interprétation':      
    st.title("INTERPRÉTATION")
    st.sidebar.title("MENU INTERPRÉTATION")
    option_submenu4 = st.sidebar.selectbox('Sélection', ("INTERPRÉTATION AVEC DURATION", "INTERPRÉTATION SANS DURATION"))
    if option_submenu4 == 'INTERPRÉTATION AVEC DURATION':
        st.subheader("Interpréation SHAP avec la colonne Duration")
        pages=["Summary plot", "Bar plot poids des variables", "Analyses des variables catégorielles", "Dependence plots"]
        page=st.sidebar.radio('Afficher', pages)
    
        if page == pages[0] :

            # Modèle avec les hyperparamètres spécifiés
            #rf_model_carolle = RandomForestClassifier(
                #class_weight='balanced',
                #max_depth=20,
                #max_features='sqrt',
                #min_samples_leaf=2,
                #min_samples_split=10,
                #n_estimators=200,
                #random_state=42,
            #)

            # Entraîner et sauvegarder le modèle
            #rf_model_carolle.fit(X_train, y_train)
            #joblib.dump(rf_model_carolle, "random_forest_model_carolle.pkl")

            # Cacher le modèle et les valeurs SHAP avec Streamlit
            @st.cache_resource
            def load_model():
                return joblib.load("random_forest_model_carolle.pkl")

            #@st.cache_data
            #def compute_shap_values(_model, data):
                #explainer = shap.TreeExplainer(_model)
                #shap_values_carolle = explainer.shap_values(data)
                #joblib.dump(shap_values_carolle, "shap_values_carolle.pkl")
                #return shap_values_carolle

            # Charger le modèle
            loaded_model_carolle = load_model()

            # Calculer ou charger les SHAP values
            #try:
                #shap_values_carolle = joblib.load("shap_values_carolle.pkl")
            #except FileNotFoundError:
                #shap_values_carolle = compute_shap_values(loaded_model_carolle, X_test)

            #modèle déjà chargé  auparavant,code a conserver
            shap_values_carolle = joblib.load("shap_values_carolle.pkl")

            
            # Affichage des visualisations SHAP
            st.subheader("Summary plot")

            fig = plt.figure()
            shap.summary_plot(shap_values_carolle[:,:,1], X_test)  # Supposant un problème de classification binaire
            st.pyplot(fig)

            # Prédiction et matrice de confusion
            y_pred = loaded_model_carolle.predict(X_test)
            table_rf = pd.crosstab(y_test, y_pred, rownames=["Réalité"], colnames=["Prédiction"])
            st.dataframe(table_rf)

            # Rapport de classification
            st.write("Classification report :")
            report_dict = classification_report(y_test, y_pred, output_dict=True)
            report_df = pd.DataFrame(report_dict).T
            st.dataframe(report_df)

        if page == pages[1] :
            st.subheader("Poids des variables dans le modèle")
            st.write("blablabla")

        if page == pages[2] :
            st.subheader("Zoom sur les variables catégorielles")
            st.write("blablabla")

        if page == pages[3] :
            st.subheader("Dépendences plots & Analyses")
            st.write("blablabla")
            
    if option_submenu4 == 'INTERPRÉTATION SANS DURATION':
        st.subheader("Interpréation SHAP sans la colonne Duration")
        pages=["Summary plot", "Bar plot poids des variables", "Analyses des variables catégorielles", "Dependence plots"]
        page=st.sidebar.radio('Afficher', pages)
    
        if page == pages[0] :
            st.subheader("Summary plot")
            st.write("blablabla")
 
            fig = plt.figure()
            shap.summary_plot(shap_values_xgboost_sd, X_test_sd)  
            st.pyplot(fig)

            # Prédiction et matrice de confusion
            y_pred = loaded_model_xgboost_sd.predict(X_test_sd)
            table_xgboost = pd.crosstab(y_test_sd, y_pred, rownames=["Réalité"], colnames=["Prédiction"])
            st.dataframe(table_xgboost)

            # Rapport de classification
            st.write("Classification report :")
            report_dict_xgboost = classification_report(y_test_sd, y_pred, output_dict=True)
            report_df_xgboost = pd.DataFrame(report_dict_xgboost).T
            st.dataframe(report_df_xgboost)
           
         

        if page == pages[1] :
            st.subheader("Poids des variables dans le modèle")
            st.write("blablabla")

        if page == pages[2] :
            st.subheader("Zoom sur les variables catégorielles")
            st.write("blablabla")

        if page == pages[3] :
            st.subheader("Dépendences plots & Analyses")
            st.write("blablabla")
           
if selected == 'Conclusion':    
    st.subheader("Conclusion & Recommandations") 
    st.write("balblabla")
    
if selected == 'Démo':    
    st.subheader("Démonstration et application de notre modèle à votre cas") 

    st.write(f'### Vos Informations')
    age = st.slider("Quel est votre âge ?", 18, 85, 1)
    education = st.selectbox("Quel est votre niveau d'étude ?", ("tertiary", "secondary", "unknown", "primary"))
    job = st.selectbox("Quel est votre profession ?", ('management', 'technician', 'entrepreneur', 'blue-collar', 'unknown', 'retired', 'admin.', 'services', 'self-employed', 'unemployed', 'housemaid', 'student'))
    marital = st.selectbox("Quel est votre statut marital ?", ('married', 'single', 'divorced'))
    balance = st.slider('Quel est le solde de votre compte en banque ?', -3000, 80000, 1)
    housing = st.selectbox("Etes-vous propriétaire ?", ('yes', 'no'))
    loan = st.selectbox("Avez-vous un crédit en cours ?", ('yes', 'no'))
    default = st.selectbox("Avez-vous déjà eu un défaut de paiement ?", ('yes', 'no'))
    month = st.selectbox("Quel mois avez-vous été contacté par votre banque, pour la dernière fois ?", ('jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'))
    weekday = st.selectbox("Quel jour avez-vous été contacté par votre banque, pour la dernière fois  ?",('Lundi', 'Mardi', 'Mercredi', 'Jeudi', 'Vendredi', 'Samedi', 'Dimanche'))
    duration = st.slider("Quelle est la durée, en seconde, de votre dernier contact avec votre banque ?", 0, 600, step=1)
    campaign = st.slider("Combien de fois avez-vous été contacté par votre banque lors de la campagne?", 0,20,1)
    Client_Category_M = st.selectbox("Dernier appel de votre banque?", ('Prospect', 'Reached-6M', 'Reached+6M'))
    previous = st.slider("Lors de la précédente campagne marketing, combien de fois avez-vous été appélé par votre banque", 0,10,1)
    poutcome = st.selectbox("Avez-vous souscris à l'offre lors de la dernière campagne marketing de votre banque ?", ('other', 'success', 'unknown', 'failure'))
    
    st.write(f'### Récapitulatif')
    st.write("Votre âge est :", age)
    st.write("Votre profession est :", job)
    st.write("Votre niveau d'étude est:", education)
    st.write("Votre statut marital est :", marital)
    st.write("Le solde de votre compte en banque est :", balance)
    st.write("Vous êtes propriétaire :", housing)
    st.write("Vous avez un crédit en cours :", loan)
    st.write("Avez-vous déjà eu un défaut de paiement :", default)
    st.write("Vous avez été contacté par votre banque en :", month)
    st.write("Vous avez été contacté par votre banque le :", weekday)
    st.write('La durée de votre derner contact avec votre banque est de:', duration)
    st.write("Le nombre d'appels que vous avez eu lors de la campagne est de :", campaign)
    st.write("Le nombre de jour entre les deux derniers contacts avec votre banque est de :", Client_Category_M)
    st.write("Le nombre de contact que vous avez eu lors de la dernière campagne est de :", previous)
    st.write("Avez-vous souscris lors de la dernière campagne :", poutcome)
    
    # Créer un dataframe récapitulatif des données du prospect
    infos_prospect = pd.DataFrame({
        'age': [age], 
        'job': [job], 
        'marital': [marital], 
        'education': [education], 
        'default': [default],
        'balance': [balance], 
        'housing': [housing], 
        'loan': [loan], 
        'month': [month], 
        'duration': [duration],
        'campaign': [campaign],
        'previous': [previous],
        'poutcome': [poutcome],
        'Client_Category_M': [Client_Category_M],
        'weekday': [weekday]
    }, index=[0]) 

    # Affichage pour vérifier le nouvel index
    st.write("Voici le tableau avec vos informations")
    st.dataframe(infos_prospect)

    # Préparation des données d'entraînement (hypothèse : dff est préalablement défini)
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
    dff = dff.drop(['deposit'], axis=1)
    dff['job'] = dff['job'].replace('unknown', np.nan)
    dff['education'] = dff['education'].replace('unknown', np.nan)
    dff['poutcome'] = dff['poutcome'].replace('unknown', np.nan)

    # Remplacement des NaNs par le mode:
    imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
    dff.loc[:,['job']] = imputer.fit_transform(dff[['job']])

    # On remplace les NaaN de 'poutcome' avec la méthode de remplissage par propagation où chaque valeur unknown est remplacée par la valeur de la ligne suivante (puis la dernière ligne par le Mode de cette variable).
    # On l'applique au X_train et X_test :
    dff['poutcome'] = dff['poutcome'].fillna(method ='bfill')
    dff['poutcome'] = dff['poutcome'].fillna(dff['poutcome'].mode()[0])

    # On fait de même pour les NaaN de 'education'
    dff['education'] = dff['education'].fillna(method ='bfill')
    dff['education'] = dff['education'].fillna(dff['education'].mode()[0])


    # Encodage des variables explicatives de type 'objet'
    oneh = OneHotEncoder(drop = 'first', sparse_output = False)
    cat1 = ['default', 'housing','loan']
    dff.loc[:, cat1] = oneh.fit_transform(dff[cat1])
    dff[cat1] = dff[cat1].astype('int64')

    # 'education' est une variable catégorielle ordinale, remplacer les modalités de la variable par des nombres, en gardant l'ordre initial
    dff['education'] = dff['education'].replace(['primary', 'secondary', 'tertiary'], [0, 1, 2])

    # 'Client_Category_M' est une variable catégorielle ordinale, remplacer les modalités de la variable par des nombres, en gardant l'ordre initial
    dff['Client_Category_M'] = dff['Client_Category_M'].replace(['Prospect', 'Reached-6M', 'Reached+6M'], [0, 1, 2])

    # Encoder les variables à plus de 2 modalités 'job', 'marital', 'poutome', 'month', 'weekday' 
    dummies = pd.get_dummies(dff['job'], prefix='job').astype(int)
    dff = pd.concat([dff.drop('job', axis=1), dummies], axis=1)

    dummies = pd.get_dummies(dff['marital'], prefix='marital').astype(int)
    dff = pd.concat([dff.drop('marital', axis=1), dummies], axis=1)

    dummies = pd.get_dummies(dff['poutcome'], prefix='poutcome').astype(int)
    dff = pd.concat([dff.drop('poutcome', axis=1), dummies], axis=1)

    dummies = pd.get_dummies(dff['month'], prefix='month').astype(int)
    dff = pd.concat([dff.drop('month', axis=1), dummies], axis=1)

    dummies = pd.get_dummies(dff['weekday'], prefix='weekday').astype(int)
    dff = pd.concat([dff.drop('weekday', axis=1), dummies], axis=1)

    # Construction du DataFrame pour le prospect à partir de infos_prospect
    pred_df = infos_prospect.copy()

    # Remplacer 'unknown' par NaN uniquement pour les colonnes spécifiques
    cols_to_check = ['job', 'education', 'poutcome']  # Colonnes à vérifier
    for col in cols_to_check:
        if (pred_df[col] == 'unknown').any():  # Vérifie si la valeur est "unknown"
            pred_df[col] = np.nan  # Remplace "unknown" par NaN

    # Imputation des valeurs manquantes dans 'job' avec SimpleImputer
    if pred_df['job'].isna().any():
        # Créer un imputer avec la stratégie 'most_frequent' (remplir par la valeur la plus fréquente)
        imputer = SimpleImputer(strategy='most_frequent')
        pred_df['job'] = imputer.fit_transform(pred_df[['job']]).flatten()

    # Remplissage par le mode pour 'education' et 'poutcome' dans le cas où il y a des NaN
    if pred_df['education'].isna().any():
        # Utiliser le mode de 'education' dans dff
        pred_df['education'] = dff['education'].mode()[0]

    if pred_df['poutcome'].isna().any():
        # Utiliser le mode de 'poutcome' dans dff
        pred_df['poutcome'] = dff['poutcome'].mode()[0]

    # Transformation de 'education' et 'Client_Category_M' pour respecter l'ordre ordinal
    pred_df['education'] = pred_df['education'].replace(['primary', 'secondary', 'tertiary'], [0, 1, 2])
    pred_df['Client_Category_M'] = pred_df['Client_Category_M'].replace(['Prospect', 'Reached-6M', 'Reached+6M'], [0, 1, 2])
    

    # Remplacer 'yes' par 1 et 'no' par 0 pour chaque colonne
    cols_to_replace = ['housing', 'loan', 'default']
    for col in cols_to_replace:
        pred_df[col] = pred_df[col].replace({'yes': 1, 'no': 0})

    # Liste des variables catégorielles multi-modales à traiter
    cat_cols_multi_modal = ['job', 'marital', 'poutcome', 'month', 'weekday']


    # Parcourir chaque variable catégorielle multi-modale pour gérer les colonnes manquantes
    for col in cat_cols_multi_modal:
        # Vérifier que la colonne existe dans pred_df
        if col in pred_df.columns:
            # Effectuer un encodage des variables catégorielles multi-modales
            dummies = pd.get_dummies(pred_df[col], prefix=col).astype(int)
            pred_df = pd.concat([pred_df.drop(col, axis=1), dummies], axis=1)

    # Réorganiser les colonnes pour correspondre exactement à celles de dff
    pred_df = pred_df.reindex(columns=dff.columns, fill_value=0)
    
    # Affichage du DataFrame transformé avant la standardisation
    st.write("Affichage du dataframe transformé (avant standardisation):")
    st.dataframe(pred_df)

    # Liste des colonnes numériques à standardiser
    num_cols = ['age', 'balance', 'duration', 'campaign', 'previous']

    # Étape 1 : Créer un index spécifique pour pred_df
    # Utiliser un index unique pour pred_df, en le commençant après la dernière ligne de dff
    pred_df.index = range(dff.shape[0], dff.shape[0] + len(pred_df))

    # Étape 2 : Concaténer dff et pred_df
    # Concaténer les deux DataFrames dff et pred_df sur les colonnes numériques
    combined_df = pd.concat([dff[num_cols], pred_df[num_cols]], axis=0)

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


    # Bouton pour lancer la prédiction
    prediction_button = st.button(label="Predict")


    # Prédiction
    if prediction_button:
        prediction = loaded_model_carolle.predict(pred_df)
        prediction_proba = loaded_model_carolle.predict_proba(pred_df)
        max_proba = np.max(prediction_proba[0]) * 100
        
        # Résultats
        if prediction[0] == 0:
            st.write(f"Prediction Outcome: {prediction[0]}")
            st.write(f"Confidence: {max_proba:.2f}%")
            st.write("Summary:", "\nThe customer is less likely to subscribe to a term deposit")
        else:
            st.write(f"Prediction Outcome: {prediction[0]}")
            st.write(f"Confidence: {max_proba:.2f}%")
            st.write("Summary:", "\nThe customer is more likely to subscribe to a term deposit")

if selected == 'DémoSD':    
    st.subheader("Démonstration et application de notre modèle à votre cas") 

    # Modèle avec les hyperparamètres spécifiés
    #xgboost_sd_opti = XGBClassifier(gamma=0.05,
                                     #colsample_bytree=0.82, 
                                     #learning_rate=0.18, 
                                     #max_depth=5,  
                                     #min_child_weight=1.4, 
                                     #n_estimators=32, reg_alpha=1.343, 
                                     #reg_lambda=1.73, scale_pos_weight=2.21, subsample=0.87, random_state=42)

    # Entraîner et sauvegarder le modèle
    #xgboost_sd_opti.fit(X_train_sd, y_train_sd)
    #joblib.dump(xgboost_sd_opti, "xgboost_sd_opti.pkl")

    # Cacher le modèle et les valeurs SHAP avec Streamlit
    @st.cache_resource
    def load_model():
        return joblib.load("xgboost_sd_opti.pkl")

    #@st.cache_data
    #def compute_shap_values(_model, data):
        #explainer = shap.TreeExplainer(_model)
        #shap_values_xgboost_sd = explainer.shap_values(data)
        #joblib.dump(shap_values_xgboost_sd, "shap_values_xgboost_sd.pkl")
        #return shap_values_xgboost_sd

    # Charger le modèle
    loaded_model_xgboost_sd = load_model()

    ## Calculer ou charger les SHAP values
    #try:
        #shap_values_xgboost_sd = joblib.load("shap_values_xgboost_sd.pkl")
    #except FileNotFoundError:
        #shap_values_xgboost_sd = compute_shap_values(loaded_model_xgboost_sd, X_test_sd)

                 

    st.write(f'### Vos Informations')
    age = st.slider("Quel est votre âge ?", 18, 85, 1)
    education = st.selectbox("Quel est votre niveau d'étude ?", ("tertiary", "secondary", "unknown", "primary"))
    job = st.selectbox("Quel est votre profession ?", ('management', 'technician', 'entrepreneur', 'blue-collar', 'unknown', 'retired', 'admin.', 'services', 'self-employed', 'unemployed', 'housemaid', 'student'))
    marital = st.selectbox("Quel est votre statut marital ?", ('married', 'single', 'divorced'))
    balance = st.slider('Quel est le solde de votre compte en banque ?', -3000, 80000, 1)
    housing = st.selectbox("Etes-vous propriétaire ?", ('yes', 'no'))
    loan = st.selectbox("Avez-vous un crédit en cours ?", ('yes', 'no'))
    default = st.selectbox("Avez-vous déjà eu un défaut de paiement ?", ('yes', 'no'))
    month = st.selectbox("Quel mois avez-vous été contacté par votre banque, pour la dernière fois ?", ('jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'))
    weekday = st.selectbox("Quel jour avez-vous été contacté par votre banque, pour la dernière fois  ?",('Lundi', 'Mardi', 'Mercredi', 'Jeudi', 'Vendredi', 'Samedi', 'Dimanche'))
    duration = st.slider("Quelle est la durée, en seconde, de votre dernier contact avec votre banque ?", 0, 600, step=1)
    campaign = st.slider("Combien de fois avez-vous été contacté par votre banque lors de la campagne?", 0,20,1)
    Client_Category_M = st.selectbox("Dernier appel de votre banque?", ('Prospect', 'Reached-6M', 'Reached+6M'))
    previous = st.slider("Lors de la précédente campagne marketing, combien de fois avez-vous été appélé par votre banque", 0,10,1)
    poutcome = st.selectbox("Avez-vous souscris à l'offre lors de la dernière campagne marketing de votre banque ?", ('other', 'success', 'unknown', 'failure'))
    
    st.write(f'### Récapitulatif')
    st.write("Votre âge est :", age)
    st.write("Votre profession est :", job)
    st.write("Votre niveau d'étude est:", education)
    st.write("Votre statut marital est :", marital)
    st.write("Le solde de votre compte en banque est :", balance)
    st.write("Vous êtes propriétaire :", housing)
    st.write("Vous avez un crédit en cours :", loan)
    st.write("Avez-vous déjà eu un défaut de paiement :", default)
    st.write("Vous avez été contacté par votre banque en :", month)
    st.write("Vous avez été contacté par votre banque le :", weekday)
    st.write('La durée de votre derner contact avec votre banque est de:', duration)
    st.write("Le nombre d'appels que vous avez eu lors de la campagne est de :", campaign)
    st.write("Le nombre de jour entre les deux derniers contacts avec votre banque est de :", Client_Category_M)
    st.write("Le nombre de contact que vous avez eu lors de la dernière campagne est de :", previous)
    st.write("Avez-vous souscris lors de la dernière campagne :", poutcome)
    
    # Créer un dataframe récapitulatif des données du prospect
    infos_prospect = pd.DataFrame({
        'age': [age], 
        'job': [job], 
        'marital': [marital], 
        'education': [education], 
        'default': [default],
        'balance': [balance], 
        'housing': [housing], 
        'loan': [loan], 
        'month': [month], 
        'duration': [duration],
        'campaign': [campaign],
        'previous': [previous],
        'poutcome': [poutcome],
        'Client_Category_M': [Client_Category_M],
        'weekday': [weekday]
    }, index=[0]) 

    # Affichage pour vérifier le nouvel index
    st.write("Voici le tableau avec vos informations")
    st.dataframe(infos_prospect)

    # Fonction pour charger le modèle
    def load_model():
        return joblib.load("random_forest_best_model.pkl")

    # Charger le modèle et les SHAP values
    joblib_random_forest_best_model = load_model()

    # Préparation des données d'entraînement (hypothèse : dff est préalablement défini)
    dff_sans_duration = df.copy()
    dff_sans_duration = dff_sans_duration[dff_sans_duration['age'] < 75]
    dff_sans_duration = dff_sans_duration.loc[dff_sans_duration["balance"] > -2257]
    dff_sans_duration = dff_sans_duration.loc[dff_sans_duration["balance"] < 4087]
    dff_sans_duration = dff_sans_duration.loc[dff_sans_duration["campaign"] < 6]
    dff_sans_duration = dff_sans_duration.loc[dff_sans_duration["previous"] < 2.5]
    bins = [-2, -1, 180, 855]
    labels = ['Prospect', 'Reached-6M', 'Reached+6M']
    dff_sans_duration['Client_Category_M'] = pd.cut(dff_sans_duration['pdays'], bins=bins, labels=labels)
    dff_sans_duration['Client_Category_M'] = dff_sans_duration['Client_Category_M'].astype('object')
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
    dff_sans_duration = dff_sans_duration.drop(['contact'], axis=1)
    dff_sans_duration = dff_sans_duration.drop(['pdays'], axis=1)
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

    st.dataframe(dff_sans_duration)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
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
    dff = dff.drop(['deposit'], axis=1)
    dff['job'] = dff['job'].replace('unknown', np.nan)
    dff['education'] = dff['education'].replace('unknown', np.nan)
    dff['poutcome'] = dff['poutcome'].replace('unknown', np.nan)

    # Remplacement des NaNs par le mode:
    imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
    dff.loc[:,['job']] = imputer.fit_transform(dff[['job']])

    # On remplace les NaaN de 'poutcome' avec la méthode de remplissage par propagation où chaque valeur unknown est remplacée par la valeur de la ligne suivante (puis la dernière ligne par le Mode de cette variable).
    # On l'applique au X_train et X_test :
    dff['poutcome'] = dff['poutcome'].fillna(method ='bfill')
    dff['poutcome'] = dff['poutcome'].fillna(dff['poutcome'].mode()[0])

    # On fait de même pour les NaaN de 'education'
    dff['education'] = dff['education'].fillna(method ='bfill')
    dff['education'] = dff['education'].fillna(dff['education'].mode()[0])


    # Encodage des variables explicatives de type 'objet'
    oneh = OneHotEncoder(drop = 'first', sparse_output = False)
    cat1 = ['default', 'housing','loan']
    dff.loc[:, cat1] = oneh.fit_transform(dff[cat1])
    dff[cat1] = dff[cat1].astype('int64')

    # 'education' est une variable catégorielle ordinale, remplacer les modalités de la variable par des nombres, en gardant l'ordre initial
    dff['education'] = dff['education'].replace(['primary', 'secondary', 'tertiary'], [0, 1, 2])

    # 'Client_Category_M' est une variable catégorielle ordinale, remplacer les modalités de la variable par des nombres, en gardant l'ordre initial
    dff['Client_Category_M'] = dff['Client_Category_M'].replace(['Prospect', 'Reached-6M', 'Reached+6M'], [0, 1, 2])

    # Encoder les variables à plus de 2 modalités 'job', 'marital', 'poutome', 'month', 'weekday' 
    dummies = pd.get_dummies(dff['job'], prefix='job').astype(int)
    dff = pd.concat([dff.drop('job', axis=1), dummies], axis=1)

    dummies = pd.get_dummies(dff['marital'], prefix='marital').astype(int)
    dff = pd.concat([dff.drop('marital', axis=1), dummies], axis=1)

    dummies = pd.get_dummies(dff['poutcome'], prefix='poutcome').astype(int)
    dff = pd.concat([dff.drop('poutcome', axis=1), dummies], axis=1)

    dummies = pd.get_dummies(dff['month'], prefix='month').astype(int)
    dff = pd.concat([dff.drop('month', axis=1), dummies], axis=1)

    dummies = pd.get_dummies(dff['weekday'], prefix='weekday').astype(int)
    dff = pd.concat([dff.drop('weekday', axis=1), dummies], axis=1)

    # Construction du DataFrame pour le prospect à partir de infos_prospect
    pred_df = infos_prospect.copy()

    # Remplacer 'unknown' par NaN uniquement pour les colonnes spécifiques
    cols_to_check = ['job', 'education', 'poutcome']  # Colonnes à vérifier
    for col in cols_to_check:
        if (pred_df[col] == 'unknown').any():  # Vérifie si la valeur est "unknown"
            pred_df[col] = np.nan  # Remplace "unknown" par NaN

    # Imputation des valeurs manquantes dans 'job' avec SimpleImputer
    if pred_df['job'].isna().any():
        # Créer un imputer avec la stratégie 'most_frequent' (remplir par la valeur la plus fréquente)
        imputer = SimpleImputer(strategy='most_frequent')
        pred_df['job'] = imputer.fit_transform(pred_df[['job']]).flatten()

    # Remplissage par le mode pour 'education' et 'poutcome' dans le cas où il y a des NaN
    if pred_df['education'].isna().any():
        # Utiliser le mode de 'education' dans dff
        pred_df['education'] = dff['education'].mode()[0]

    if pred_df['poutcome'].isna().any():
        # Utiliser le mode de 'poutcome' dans dff
        pred_df['poutcome'] = dff['poutcome'].mode()[0]

    # Transformation de 'education' et 'Client_Category_M' pour respecter l'ordre ordinal
    pred_df['education'] = pred_df['education'].replace(['primary', 'secondary', 'tertiary'], [0, 1, 2])
    pred_df['Client_Category_M'] = pred_df['Client_Category_M'].replace(['Prospect', 'Reached-6M', 'Reached+6M'], [0, 1, 2])
    

    # Remplacer 'yes' par 1 et 'no' par 0 pour chaque colonne
    cols_to_replace = ['housing', 'loan', 'default']
    for col in cols_to_replace:
        pred_df[col] = pred_df[col].replace({'yes': 1, 'no': 0})

    # Liste des variables catégorielles multi-modales à traiter
    cat_cols_multi_modal = ['job', 'marital', 'poutcome', 'month', 'weekday']


    # Parcourir chaque variable catégorielle multi-modale pour gérer les colonnes manquantes
    for col in cat_cols_multi_modal:
        # Vérifier que la colonne existe dans pred_df
        if col in pred_df.columns:
            # Effectuer un encodage des variables catégorielles multi-modales
            dummies = pd.get_dummies(pred_df[col], prefix=col).astype(int)
            pred_df = pd.concat([pred_df.drop(col, axis=1), dummies], axis=1)

    # Réorganiser les colonnes pour correspondre exactement à celles de dff
    pred_df = pred_df.reindex(columns=dff.columns, fill_value=0)
    
    # Affichage du DataFrame transformé avant la standardisation
    st.write("Affichage du dataframe transformé (avant standardisation):")
    st.dataframe(pred_df)

    # Liste des colonnes numériques à standardiser
    num_cols = ['age', 'balance', 'duration', 'campaign', 'previous']

    # Étape 1 : Créer un index spécifique pour pred_df
    # Utiliser un index unique pour pred_df, en le commençant après la dernière ligne de dff
    pred_df.index = range(dff.shape[0], dff.shape[0] + len(pred_df))

    # Étape 2 : Concaténer dff et pred_df
    # Concaténer les deux DataFrames dff et pred_df sur les colonnes numériques
    combined_df = pd.concat([dff[num_cols], pred_df[num_cols]], axis=0)

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


    # Bouton pour lancer la prédiction
    prediction_button = st.button(label="Predict")


    # Prédiction
    if prediction_button:
        prediction = joblib_random_forest_best_model.predict(pred_df)
        prediction_proba = joblib_random_forest_best_model.predict_proba(pred_df)
        max_proba = np.max(prediction_proba[0]) * 100
        
        # Résultats
        if prediction[0] == 0:
            st.write(f"Prediction Outcome: {prediction[0]}")
            st.write(f"Confidence: {max_proba:.2f}%")
            st.write("Summary:", "\nThe customer is less likely to subscribe to a term deposit")
        else:
            st.write(f"Prediction Outcome: {prediction[0]}")
            st.write(f"Confidence: {max_proba:.2f}%")
            st.write("Summary:", "\nThe customer is more likely to subscribe to a term deposit")
