# Pump.fun Trading Bot

Un bot de trading automatique pour les tokens sur pump.fun et Raydium.

## Stratégie

1. Détection des nouveaux tokens sur pump.fun via l'API GraphQL de BitQuery
2. Achat initial de 0.01 SOL sur pump.fun
3. Vente avec 30% de profit
4. Surveillance de la migration vers Raydium
5. Rachat à 100k de market cap sur Raydium
6. Vente finale avec x2 de profit

## Installation

1. Installer Python 3.12 ou supérieur
2. Installer les dépendances :
```bash
pip install -r requirements.txt
```

## Configuration

Créer un fichier `.env` avec les variables suivantes :
```
WALLET_PRIVATE_KEY=votre_clé_privée_solana
PUMP_FUN_API_KEY=votre_clé_api_bitquery
RPC_URL=https://api.mainnet-beta.solana.com
BITQUERY_API_URL=https://streaming.bitquery.io/graphql
```

## Utilisation

```bash
python bot.py
```

## Structure des fichiers

- `bot.py` : Point d'entrée principal et logique du bot
- `config.py` : Configuration des paramètres de trading
- `graphql_client.py` : Client GraphQL pour l'API BitQuery
- `trading_token.py` : Gestion des tokens et des prix
- `wallet.py` : Gestion du portefeuille Solana
- `requirements.txt` : Liste des dépendances Python
- `.env` : Variables d'environnement (à créer)
