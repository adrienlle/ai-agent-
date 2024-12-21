from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import asyncio
import json
from tracker import TokenTracker
from datetime import datetime

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

# Instance globale du tracker
tracker = TokenTracker(db_url="sqlite:///tokens.db")
tokens_cache = []

async def background_task():
    """Tâche de fond qui exécute le tracker"""
    global tokens_cache
    while True:
        try:
            new_tokens = await tracker.get_new_tokens()
            if new_tokens:
                # Ajoute timestamp
                for token in new_tokens:
                    token["found_at"] = datetime.now().isoformat()
                # Met à jour le cache
                tokens_cache = (new_tokens + tokens_cache)[:100]  # Garde les 100 derniers
        except Exception as e:
            print(f"Erreur dans la tâche de fond: {str(e)}")
        await asyncio.sleep(10)  # Attend 10s entre chaque analyse

@app.on_event("startup")
async def startup_event():
    """Démarre la tâche de fond au démarrage"""
    asyncio.create_task(background_task())

@app.get("/")
async def root():
    """Page d'accueil"""
    with open("static/index.html") as f:
        return HTMLResponse(f.read())

@app.get("/api/tokens")
async def get_tokens():
    """Renvoie les derniers tokens"""
    return tokens_cache

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """Endpoint WebSocket pour les mises à jour en temps réel"""
    await websocket.accept()
    try:
        last_update = ""
        while True:
            # Vérifie s'il y a des nouveaux tokens
            if tokens_cache and tokens_cache[0].get("found_at", "") != last_update:
                last_update = tokens_cache[0].get("found_at", "")
                await websocket.send_json(tokens_cache)
            await asyncio.sleep(1)
    except Exception as e:
        print(f"Erreur WebSocket: {str(e)}")
    finally:
        await websocket.close()
