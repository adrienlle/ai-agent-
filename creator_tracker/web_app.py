from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from tracker import CreatorTracker
import uvicorn
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime, timedelta

app = FastAPI()
tracker = CreatorTracker()

@app.get("/", response_class=HTMLResponse)
async def home():
    """Page d'accueil avec les créateurs actifs"""
    creators = tracker.db.get_active_creators()
    
    if not creators:
        return "Aucun créateur actif trouvé avec des patterns similaires"
    
    # Crée un DataFrame
    df = pd.DataFrame(creators)
    
    # Graphique des patterns
    fig_patterns = go.Figure()
    for _, creator in df.iterrows():
        fig_patterns.add_trace(go.Scatter(
            x=[creator["avg_time_to_success"], creator["avg_sol_invested"]],
            y=[creator["success_rate"], creator["pattern_score"]],
            mode="markers+text",
            name=creator["address"],
            text=[f"{creator['tokens_24h']} tokens/24h"],
            textposition="top center"
        ))
    fig_patterns.update_layout(
        title="Patterns des Créateurs Actifs",
        xaxis_title="Temps moyen pour x2 (minutes)",
        yaxis_title="Taux de succès (%)"
    )
    
    # Crée la page HTML
    html_content = f"""
    <html>
        <head>
            <title>Créateurs Actifs - Pump.fun</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
                .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
                table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
                th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
                th {{ background-color: #f8f9fa; }}
                .highlight {{ background-color: #e8f5e9; }}
                .graph {{ margin: 20px 0; background: white; padding: 15px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Créateurs Actifs sur Pump.fun</h1>
                <p>Créateurs avec au moins 2 tokens/jour, 50% de succès et des patterns similaires</p>
                
                <div class="graph">
                    {fig_patterns.to_html(full_html=False)}
                </div>
                
                <h2>Détails des Créateurs</h2>
                <table>
                    <tr>
                        <th>Adresse</th>
                        <th>Tokens 24h</th>
                        <th>Succès</th>
                        <th>Pattern</th>
                        <th>Temps x2</th>
                        <th>SOL Investi</th>
                        <th>Dernier Token</th>
                    </tr>
                    {"".join(f'''
                    <tr class="{'highlight' if c['pattern_score'] >= 80 else ''}">
                        <td>{c['address']}</td>
                        <td>{c['tokens_24h']}</td>
                        <td>{c['success_rate']:.1f}%</td>
                        <td>{c['pattern_score']:.1f}/100</td>
                        <td>{c['avg_time_to_success']:.1f}min</td>
                        <td>{c['avg_sol_invested']:.2f}</td>
                        <td>{datetime.fromisoformat(c['last_token']).strftime('%H:%M:%S')}</td>
                    </tr>
                    ''' for c in creators)}
                </table>
            </div>
        </body>
    </html>
    """
    
    return HTMLResponse(content=html_content)

@app.get("/api/creators")
async def get_creators(min_tokens: int = 5, min_success_rate: float = 50):
    """API endpoint pour récupérer les créateurs"""
    return tracker.get_top_creators(min_tokens, min_success_rate)

if __name__ == "__main__":
    tracker.db.init_db()  # Initialise la base de données au démarrage
    uvicorn.run(app, host="127.0.0.1", port=8081)
