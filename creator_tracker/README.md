# Creator Tracker

Ce bot analyse les créateurs de tokens sur pump.fun pour identifier les plus performants.

## Installation

```bash
pip install -r requirements.txt
```

## Utilisation

1. Lancer le tracker (analyse en continu) :
```bash
python tracker.py
```

2. Lancer l'interface web (optionnel) :
```bash
python web_app.py
```
Puis ouvrir http://localhost:8080 dans votre navigateur.

## Fonctionnalités

- Surveille tous les tokens créés sur pump.fun
- Analyse les performances de chaque créateur :
  - Nombre de tokens créés
  - Taux de succès (x2 ou Raydium)
  - Temps moyen pour atteindre le succès
  - Montant SOL moyen investi
- Interface web pour visualiser les statistiques
- API pour récupérer les données

## Base de données

Les données sont stockées dans une base SQLite (`creators.db`) avec deux tables :

- `creators` : Statistiques des créateurs
- `tokens` : Détails de chaque token

## Configuration

Vous pouvez ajuster les paramètres dans `tracker.py` :
- Seuil minimum de tokens pour considérer un créateur
- Taux de succès minimum
- Intervalle de mise à jour des données
