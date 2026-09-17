# Bank App

Application Streamlit d'analyse et de prediction pour le projet bancaire.
URL = https://bankapp-meycfbeyrtgusvaljnvh4r.streamlit.app/

## Lancer en local

Depuis la racine du projet :

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

L'application sera disponible a l'adresse `http://localhost:8501`.

## Structure du projet

```text
.
├── app.py
├── requirements.txt
└── dilenesantos/
	├── bank_app.py
	├── bank.csv
	├── *.pkl
	└── *.png
```

Les fichiers de donnees, modeles et images utilises par l'application doivent
rester dans le dossier `dilenesantos/`.

## Deployer sur Streamlit Community Cloud

1. Placer ce projet dans un depot GitHub.
2. Ouvrir [Streamlit Community Cloud](https://share.streamlit.io/).
3. Cliquer sur **Deploy an app**.
4. Selectionner le depot et la branche `main`.
5. Choisir `app.py` comme fichier principal.
6. Cliquer sur **Deploy**.

Streamlit Cloud installera automatiquement les dependances de
`requirements.txt`.

## Fichiers a ne pas publier

Ne pas ajouter au depot :

- `.venv/`
- `__pycache__/`
- fichiers temporaires
- secrets ou cles privees
