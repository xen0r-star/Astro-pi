# Projet de Calcul de la Vitesse de l'ISS
- Nom de l'équipe : FloThiRaf
- Professeur : Laila Bouteglifine
- Étudiants : [Florian Berte](https://github.com/xen0r-star), [Thibaut Dudart](https://github.com/thibautddrt), [Rafaël Ravry](https://github.com/xansterrr)
- École : [Institut Saint-François de Sales](https://maps.app.goo.gl/fj6R5pSYGHteDu2t7)
- Ville : [Ath](https://maps.app.goo.gl/BtFSd77azyfDAs5f6)
- Pays : Belgique

## Introduction
Le projet vise à calculer la vitesse de la Station Spatiale Internationale (ISS) en utilisant une approche basée sur la capture d'images de la Terre à l'aide d'une caméra Raspberry Pi. L'objectif est d'obtenir des images simultanées de la Terre, de les analyser à l'aide de Google Coral pour distinguer les zones avec nuages de celles sans nuages, puis de calculer la vitesse de l'ISS en mesurant la distance entre des éléments repérés sur ces images.

## Méthodologie
### 1. Acquisition des Images
Utilisation d'une caméra Raspberry Pi pour capturer des images de la Terre depuis la Station Spatiale Internationale.
### 2. Analyse d'Images
Traitement des images à l'aide de Google Coral pour différencier les zones avec nuages de celles sans nuages.
### 3. Sélection des Paires d'Images Simultanées
Répétition du processus jusqu'à l'obtention de deux images ayant les mêmes caractéristiques (nuageuses ou sans nuages) et prises simultanément.
### 4. Calcul de la Distance
Mesure de la distance entre deux éléments identifiables sur les images en utilisant des coordonnées x et y.
### 5. Calcul de la Vitesse
Utilisation des coordonnées récupérées pour calculer la vitesse de l'ISS en analysant le déplacement relatif entre les deux images.
### 6. Répétition du Processus
Répétition du processus plusieurs fois pour obtenir une vitesse moyenne de l'ISS.


## Résultats Attendus
Le projet devrait fournir une méthode permettant de calculer la vitesse de l'ISS en utilisant des images de la Terre capturées à partir de la station spatiale. Les résultats seront basés sur une analyse précise des images et une mesure correcte de la distance entre des éléments repérés.

## Conclusion
En conclusion, cette approche offre une manière novatrice de calculer la vitesse de l'ISS en utilisant des technologies abordables telles que la caméra Raspberry Pi et le processeur Google Coral. Les résultats obtenus pourraient avoir des applications significatives dans la surveillance et l'étude du mouvement de la Station Spatiale Internationale.




## Probléme
- probléme taille image
- resulta 
- google coral
- probléme environement
- gestion d'erreur