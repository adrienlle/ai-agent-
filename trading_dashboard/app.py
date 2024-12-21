from flask import Flask, render_template
from flask_socketio import SocketIO
import threading
import time
import numpy as np
from dataclasses import dataclass
import logging
from datetime import datetime, timedelta
import pytz

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

@dataclass
class TrainingMetrics:
    loss: float
    accuracy: float
    status: str

class AITrainingSimulator:
    def __init__(self):
        self.us_tz = pytz.timezone('US/Eastern')
        current_time = datetime.now(self.us_tz)
        
        # Training start: Monday December 18, 2023 8:00 AM New York
        self.start_time = datetime(2023, 12, 18, 8, 0, 0, tzinfo=self.us_tz)
        
        # Performance data (3 points par jour)
        # Format: (performance, status, date)
        self.performance_data = [
            # Semaine 1 (Actuelle)
            (0.0, "Starting", "Mon 18/12"),    # Lundi 8h
            (-3.2, "Error", "Mon 18/12"),      # Lundi 12h
            (-8.5, "Running", "Mon 18/12"),    # Lundi 16h
            
            (-12.4, "Running", "Tue 19/12"),   # Mardi 8h
            (-14.8, "Running", "Tue 19/12"),   # Mardi 12h
            (-15.2, "Analysis", "Tue 19/12"),  # Mardi 16h
            
            (-15.0, "Running", "Wed 20/12"),   # Mercredi 8h
            (-14.7, "Running", "Wed 20/12"),   # Mercredi 12h
            (-14.9, "Analysis", "Wed 20/12"),  # Mercredi 16h
            
            (-14.6, "Running", "Thu 21/12"),   # Jeudi 8h
            (-14.3, "Running", "Thu 21/12"),   # Jeudi 12h
            (-14.5, "Analysis", "Thu 21/12"),  # Jeudi 16h
            
            (-14.2, "Running", "Fri 22/12"),   # Vendredi 8h
            (-14.0, "Running", "Fri 22/12"),   # Vendredi 12h
            (-14.1, "Analysis", "Fri 22/12")   # Vendredi 16h
        ]
        
        # Calculer le nombre d'epochs basé sur le temps écoulé depuis lundi
        self.total_epochs = len(self.performance_data)
        self.current_epoch = 0
        self.is_training = False
        
        # Période de maintenance: 16:00-20:00 New York time
        self.maintenance_start = 16
        self.maintenance_duration = 4
    
    def _get_current_us_time(self) -> datetime:
        return datetime.now(self.us_tz)
    
    def _is_maintenance_time(self, current_time: datetime) -> bool:
        current_hour = current_time.hour
        return self.maintenance_start <= current_hour < (self.maintenance_start + self.maintenance_duration)
    
    def _get_current_status(self) -> str:
        current_time = self._get_current_us_time()
        current_hour = current_time.hour
        
        if self._is_maintenance_time(current_time):
            return f" System Analysis Period ({self.maintenance_start}:00-{self.maintenance_start + self.maintenance_duration}:00 NY):\n" + \
                   "• Analyzing trade patterns\n" + \
                   "• Updating market models\n" + \
                   "• Processing daily results\n" + \
                   "• Optimizing strategies"
        else:
            next_analysis = f"{self.maintenance_start}:00"
            return f" Active Trading & Learning\nNext analysis period at {next_analysis} NY time"
    
    def start_training(self):
        if not self.is_training:
            self.is_training = True
            threading.Thread(target=self._training_loop, daemon=True).start()
    
    def _training_loop(self):
        while self.is_training and self.current_epoch < self.total_epochs:
            metrics = self._simulate_training_step()
            self._emit_training_updates(metrics)
            time.sleep(2)  # Update every 2 seconds
            self.current_epoch += 1
    
    def _simulate_training_step(self) -> TrainingMetrics:
        current_time = self._get_current_us_time()
        status = "Analysis" if self._is_maintenance_time(current_time) else "Running"
        
        # L'accuracy augmente lentement avec le temps
        base_accuracy = 0.23  # Accuracy initiale
        accuracy_improvement = min(0.05, (current_time - self.start_time).total_seconds() / (24 * 3600) * 0.008)
        current_accuracy = min(0.28, base_accuracy + accuracy_improvement)
        
        return TrainingMetrics(
            loss=0.0,
            accuracy=current_accuracy,
            status=status
        )
    
    def _emit_training_updates(self, metrics: TrainingMetrics):
        current_time = self._get_current_us_time()
        
        # Limiter l'historique au nombre actuel de points
        visible_history = [p[0] for p in self.performance_data[:self.current_epoch]]
        visible_status = [p[1] for p in self.performance_data[:self.current_epoch]]
        visible_dates = [p[2] for p in self.performance_data[:self.current_epoch]]
        
        if not visible_history:
            visible_history = [self.performance_data[0][0]]
            visible_status = [self.performance_data[0][1]]
            visible_dates = [self.performance_data[0][2]]
        
        current_status = self._get_current_status()
        
        # Calculer le temps restant jusqu'à la prochaine période d'analyse
        next_analysis = current_time.replace(hour=self.maintenance_start, minute=0, second=0)
        if current_time.hour >= self.maintenance_start:
            next_analysis = next_analysis + timedelta(days=1)
        
        time_to_analysis = next_analysis - current_time
        hours_to_analysis = time_to_analysis.total_seconds() / 3600
        
        logger.info(f"Updating training metrics: epoch={self.current_epoch}, accuracy={metrics.accuracy}, performance={visible_history[-1] if visible_history else 0.0}")
        
        update_data = {
            'epoch': self.current_epoch,
            'total_epochs': self.total_epochs,
            'loss': 0.0,
            'accuracy': metrics.accuracy,
            'performance': visible_history[-1] if visible_history else 0.0,
            'performance_history': visible_history,
            'dates': visible_dates,
            'status_history': visible_status,
            'current_status': current_status,
            'current_time': current_time.strftime("%m/%d/%y %I:%M %p") + " NY",
            'start_time': self.start_time.strftime("%m/%d/%y %I:%M %p") + " NY",
            'hours_to_analysis': round(hours_to_analysis, 1)
        }
        
        socketio.emit('training_update', update_data)

simulator = AITrainingSimulator()

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('connect')
def handle_connect():
    simulator.start_training()

@socketio.on('disconnect')
def handle_disconnect():
    simulator.is_training = False

if __name__ == '__main__':
    socketio.run(app, debug=True, host='127.0.0.1', port=5000)
