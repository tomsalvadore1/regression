import pandas as pd
from sklearn.linear_model import LinearRegression 
from sklearn.compose import ColumnTransformer 
from sklearn.preprocessing import OneHotEncoder 
from sklearn.model_selection import train_test_split



data = pd.read_csv("ds_salaries.csv")




conditions_de_salaire = ["job_title", "company_location", "company_size","experience_level"]  #les colonnes que l'on juge utiles 
salaire = "salary_in_usd"  # ce que l'on veut prevoir



les_conditions = data[conditions_de_salaire]  #les colonnes nommées sont cherchées dans le csv 
le_salaire = data[salaire]  #pareil


ccomment_transformer = ColumnTransformer(transformers=[("", OneHotEncoder(), conditions_de_salaire)])  #on se sert de onehot pour la transformation a venir


conditions_transformes = ccomment_transformer.fit_transform(les_conditions) #pour faire une regression linéaire il faut transformer les donnéees en chiffres, et en une ligne c'est illisible



entrainement_conditions, test_conditions, entrainement_salaire, test_salaire = train_test_split(conditions_transformes, le_salaire, test_size=0.1) #on separe les données pour les test et les données pour le modèle, 10% sont utilisées pour les tests

modele = LinearRegression() 

modele.fit(entrainement_conditions, entrainement_salaire) #on entraine le modele

r = modele.score(test_conditions, test_salaire) # on calcule R²
print("R²:", r)

input = pd.DataFrame({
    "job_title": ["Data Analyst"], #ATTENTION AUX MAJUSCULES 
    "company_location": ["US"],
    "company_size": ["L"],
    "experience_level" : ["SE"]
})

input_transformé =ccomment_transformer.transform(input) #on applique la meme transformation a input
prédictions = modele.predict(input_transformé) #c'est la que se fait la magie 
print(prédictions)