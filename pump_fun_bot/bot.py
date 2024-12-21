import asyncio
import logging
from trading_token import TokenTrader

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

async def main():
    # Initialise le trader
    trader = TokenTrader()
    
    try:
        # Démarre le trader
        await trader.start()
        
        # Garde le bot en vie
        while True:
            await asyncio.sleep(1)
            
    except KeyboardInterrupt:
        logging.info("Arrêt du bot...")
        await trader.stop()
    except Exception as e:
        logging.error(f"Erreur: {str(e)}")
        await trader.stop()

if __name__ == "__main__":
    asyncio.run(main())
