from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from tracker import CreatorTracker
import uvicorn
import plotly.express as px
import pandas as pd

app = FastAPI()
tracker = CreatorTracker()

@app.get("/", response_class=HTMLResponse)
async def home():
    """Page d'accueil avec les statistiques globales"""
    creators = tracker.get_top_creators()
    
    # Crée un DataFrame pour les graphiques
    df = pd.DataFrame(creators)
    
    # Graphique des taux de succès
    fig_success = px.bar(
        df,
        x="address",
        y="success_rate",
        title="Taux de succès par créateur"
    )
    
    # Graphique du nombre de tokens
    fig_tokens = px.bar(
        df,
        x="address",
        y="total_tokens",
        title="Nombre total de tokens par créateur"
    )
    
    # Crée la page HTML
    html_content = f"""
    <html>
        <head>
            <title>Analyse des Créateurs de Tokens</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .container {{ max-width: 1200px; margin: 0 auto; }}
                table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
                th, td {{ padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }}
                th {{ background-color: #f2f2f2; }}
                .graph {{ margin: 20px 0; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Analyse des Créateurs de Tokens</h1>
                
                <h2>Meilleurs Créateurs</h2>
                <table>
                    <tr>
                        <th>Adresse</th>
                        <th>Tokens Total</th>
                        <th>Taux de Succès</th>
                        <th>Temps Moyen (min)</th>
                        <th>SOL Moyen</th>
                    </tr>
                    {"".join(f'''
                    <tr>
                        <td>{c["address"]}</td>
                        <td>{c["total_tokens"]}</td>
                        <td>{c["success_rate"]:.1f}%</td>
                        <td>{c["avg_time_to_success"]:.1f}</td>
                        <td>{c["avg_sol_invested"]:.1f}</td>
                    </tr>
                    ''' for c in creators)}
                </table>
                
                <div class="graph">
                    {fig_success.to_html(full_html=False)}
                </div>
                
                <div class="graph">
                    {fig_tokens.to_html(full_html=False)}
                </div>
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
    uvicorn.run(app, host="0.0.0.0", port=8080)
