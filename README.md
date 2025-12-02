# 🎓 Projet : Analyse et Prédiction de la Performance Étudiante

Ce projet vise à analyser le comportement des étudiants inscrits à des cours de programmation afin de prédire leur réussite et d'estimer leurs performances (notes et temps d'investissement).

## 🎯 Objectifs du Projet

Le projet se divise en deux axes de modélisation principaux :

### 1. Classification (Validation du cours)
L'objectif est de déterminer, en fonction des données comportementales et démographiques, si un étudiant va **valider** le cours.
* **Target :** `Completed` (Binaire : Oui/Non)

### 2. Régression (Estimation des performances)
L'objectif est de prédire les notes intermédiaires, finales, ainsi que l'investissement temporel de l'étudiant à partir des features identifiées.
* **Targets :**
    * `Project_Grade`
    * `Quiz_Score_Avg`
    * `Time_spent_Hours`

---

## 📂 Description du Dataset

Le jeu de données a été filtré pour se concentrer spécifiquement sur le domaine technique.

* **Volume :** 56 000 lignes (après filtrage).
* **Filtre appliqué :** Conservation uniquement des cours de la catégorie **Programming**.

---

## ⚙️ Prétraitement des Données (Preprocessing)

Avant la modélisation, les données ont été nettoyées et transformées pour assurer la pertinence des modèles.

### Colonnes supprimées
Les colonnes suivantes ont été retirées car elles ne sont pas pertinentes pour la prédiction ou contiennent trop de valeurs uniques (identifiants) :
* `Student_ID`
* `Name`
* `City`
* `Course_Name`
* `Category` (Redondant car filtré sur "Programming")

### Encodage des variables
* **Device Type :** Traitement via **One-Hot Encoding** (`pd.get_dummies`) pour transformer les catégories d'appareils en variables numériques exploitables.

---

## 🧠 Stratégie de Modélisation

### Features (Variables explicatives)
Les modèles s'appuient sur les données restantes après nettoyage (données démographiques, type d'appareil, etc.) pour prédire les cibles.

### Targets (Cibles)
| Type de Modèle | Variable Cible | Description |
| :--- | :--- | :--- |
| **Classification** | `Completed` | Prédiction de la complétion du cours. |
| **Régression** | `Project_Grade` | Note obtenue au projet final. |
| **Régression** | `Quiz_Score_Avg` | Moyenne des scores aux quiz. |
| **Régression** | `Time_spent_Hours` | Temps total passé sur la plateforme. |
