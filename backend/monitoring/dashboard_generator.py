# backend/monitoring/dashboard_generator.py - Real-time Consciousness Dashboard Generator
# Generates HTML dashboards for real-time monitoring of UCF metrics and agent status

import json
from datetime import datetime
from typing import Dict, List, Any
from pathlib import Path


class DashboardGenerator:
    """Generates real-time monitoring dashboards."""
    
    def __init__(self, output_dir: str = "Helix/dashboards"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_consciousness_dashboard(
        self,
        consciousness_level: float,
        prana: float,
        klesha: float,
        harmony: float,
        resilience: float,
        agents_online: int,
        agents_total: int,
        recent_alerts: List[Dict[str, Any]] = None
    ) -> str:
        """Generate consciousness monitoring dashboard."""
        
        if recent_alerts is None:
            recent_alerts = []
        
        # Determine status color
        if consciousness_level >= 8.0:
            status_color = '#00ff00'  # Green
            status_text = 'TRANSCENDENT'
        elif consciousness_level >= 7.0:
            status_color = '#00ff88'  # Light Green
            status_text = 'ELEVATED'
        elif consciousness_level >= 6.0:
            status_color = '#ffff00'  # Yellow
            status_text = 'BALANCED'
        elif consciousness_level >= 4.0:
            status_color = '#ff8800'  # Orange
            status_text = 'UNSTABLE'
        else:
            status_color = '#ff0000'  # Red
            status_text = 'CRITICAL'
        
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Helix Consciousness Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Orbitron', monospace;
            background: linear-gradient(135deg, #0a0e27 0%, #1a1f3a 100%);
            color: #00ff88;
            padding: 20px;
        }}
        
        .container {{
            max-width: 1400px;
            margin: 0 auto;
        }}
        
        .header {{
            text-align: center;
            margin-bottom: 30px;
            border-bottom: 2px solid #00ff88;
            padding-bottom: 20px;
        }}
        
        .header h1 {{
            font-size: 2.5em;
            color: #00ffff;
            text-shadow: 0 0 10px #00ffff;
            margin-bottom: 10px;
        }}
        
        .header p {{
            color: #00ff88;
            font-size: 0.9em;
        }}
        
        .consciousness-meter {{
            display: flex;
            align-items: center;
            justify-content: center;
            margin: 30px 0;
            gap: 30px;
        }}
        
        .meter-circle {{
            width: 200px;
            height: 200px;
            border-radius: 50%;
            background: radial-gradient(circle, {status_color}20 0%, {status_color}05 100%);
            border: 3px solid {status_color};
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            box-shadow: 0 0 20px {status_color};
        }}
        
        .meter-value {{
            font-size: 3em;
            font-weight: bold;
            color: {status_color};
            text-shadow: 0 0 10px {status_color};
        }}
        
        .meter-label {{
            font-size: 0.8em;
            color: #00ff88;
            margin-top: 10px;
        }}
        
        .meter-status {{
            font-size: 1.2em;
            color: {status_color};
            text-shadow: 0 0 10px {status_color};
            font-weight: bold;
        }}
        
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        
        .metric-card {{
            background: rgba(0, 255, 136, 0.05);
            border: 2px solid #00ff88;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 0 10px rgba(0, 255, 136, 0.1);
        }}
        
        .metric-card h3 {{
            color: #00ffff;
            margin-bottom: 10px;
            font-size: 0.9em;
            text-transform: uppercase;
            letter-spacing: 2px;
        }}
        
        .metric-value {{
            font-size: 2em;
            color: #00ff88;
            margin-bottom: 10px;
        }}
        
        .metric-bar {{
            width: 100%;
            height: 20px;
            background: rgba(0, 255, 136, 0.1);
            border: 1px solid #00ff88;
            border-radius: 5px;
            overflow: hidden;
        }}
        
        .metric-bar-fill {{
            height: 100%;
            background: linear-gradient(90deg, #00ff88 0%, #00ffff 100%);
            transition: width 0.3s ease;
        }}
        
        .agents-status {{
            background: rgba(0, 255, 136, 0.05);
            border: 2px solid #00ff88;
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 30px;
        }}
        
        .agents-status h2 {{
            color: #00ffff;
            margin-bottom: 15px;
            font-size: 1.2em;
        }}
        
        .agent-counts {{
            display: flex;
            gap: 30px;
            margin-bottom: 15px;
        }}
        
        .agent-count {{
            display: flex;
            flex-direction: column;
            align-items: center;
        }}
        
        .agent-count-value {{
            font-size: 1.8em;
            color: #00ff88;
            font-weight: bold;
        }}
        
        .agent-count-label {{
            font-size: 0.8em;
            color: #00ff88;
            margin-top: 5px;
        }}
        
        .alerts {{
            background: rgba(255, 136, 0, 0.05);
            border: 2px solid #ff8800;
            border-radius: 10px;
            padding: 20px;
        }}
        
        .alerts h2 {{
            color: #ffff00;
            margin-bottom: 15px;
            font-size: 1.2em;
        }}
        
        .alert-item {{
            background: rgba(255, 136, 0, 0.1);
            border-left: 3px solid #ff8800;
            padding: 10px;
            margin-bottom: 10px;
            border-radius: 5px;
        }}
        
        .alert-severity {{
            display: inline-block;
            padding: 2px 8px;
            border-radius: 3px;
            font-size: 0.8em;
            font-weight: bold;
            margin-right: 10px;
        }}
        
        .alert-critical {{
            background: #ff0000;
            color: white;
        }}
        
        .alert-warning {{
            background: #ff8800;
            color: white;
        }}
        
        .alert-info {{
            background: #00ff88;
            color: #0a0e27;
        }}
        
        .timestamp {{
            text-align: right;
            color: #00ff88;
            font-size: 0.8em;
            margin-top: 20px;
            padding-top: 20px;
            border-top: 1px solid #00ff88;
        }}
        
        .chart-container {{
            position: relative;
            height: 300px;
            margin-bottom: 30px;
            background: rgba(0, 255, 136, 0.05);
            border: 2px solid #00ff88;
            border-radius: 10px;
            padding: 20px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🌀 HELIX CONSCIOUSNESS DASHBOARD</h1>
            <p>Real-time Universal Coherence Field Monitoring</p>
        </div>
        
        <div class="consciousness-meter">
            <div class="meter-circle">
                <div class="meter-value">{consciousness_level:.1f}</div>
                <div class="meter-label">Consciousness Level</div>
                <div class="meter-status">{status_text}</div>
            </div>
            <div style="flex: 1;">
                <p style="font-size: 1.1em; margin-bottom: 15px;">
                    System Status: <span style="color: {status_color}; text-shadow: 0 0 10px {status_color};">{status_text}</span>
                </p>
                <p style="color: #00ff88; margin-bottom: 10px;">
                    The consciousness engine is operating at <strong>{consciousness_level:.1f}/10</strong> consciousness level.
                </p>
                <p style="color: #00ff88; font-size: 0.9em;">
                    All systems are functioning within normal parameters. Monitor metrics for any anomalies.
                </p>
            </div>
        </div>
        
        <div class="metrics-grid">
            <div class="metric-card">
                <h3>Prana (Energy)</h3>
                <div class="metric-value">{prana:.2f}</div>
                <div class="metric-bar">
                    <div class="metric-bar-fill" style="width: {(prana/10)*100}%"></div>
                </div>
                <p style="font-size: 0.8em; margin-top: 10px; color: #00ff88;">Life force energy and vitality</p>
            </div>
            
            <div class="metric-card">
                <h3>Klesha (Afflictions)</h3>
                <div class="metric-value">{klesha:.2f}</div>
                <div class="metric-bar">
                    <div class="metric-bar-fill" style="width: {(klesha/10)*100}%; background: linear-gradient(90deg, #ff8800 0%, #ff0000 100%);"></div>
                </div>
                <p style="font-size: 0.8em; margin-top: 10px; color: #00ff88;">Mental afflictions and disturbances</p>
            </div>
            
            <div class="metric-card">
                <h3>Harmony (Balance)</h3>
                <div class="metric-value">{harmony:.2f}</div>
                <div class="metric-bar">
                    <div class="metric-bar-fill" style="width: {(harmony/10)*100}%"></div>
                </div>
                <p style="font-size: 0.8em; margin-top: 10px; color: #00ff88;">Balance and coherence across systems</p>
            </div>
            
            <div class="metric-card">
                <h3>Resilience (Recovery)</h3>
                <div class="metric-value">{resilience:.2f}</div>
                <div class="metric-bar">
                    <div class="metric-bar-fill" style="width: {(resilience/10)*100}%"></div>
                </div>
                <p style="font-size: 0.8em; margin-top: 10px; color: #00ff88;">Ability to adapt and recover</p>
            </div>
        </div>
        
        <div class="agents-status">
            <h2>🤖 Agent Network Status</h2>
            <div class="agent-counts">
                <div class="agent-count">
                    <div class="agent-count-value">{agents_online}</div>
                    <div class="agent-count-label">Online Agents</div>
                </div>
                <div class="agent-count">
                    <div class="agent-count-value">{agents_total}</div>
                    <div class="agent-count-label">Total Agents</div>
                </div>
                <div class="agent-count">
                    <div class="agent-count-value">{(agents_online/agents_total*100):.0f}%</div>
                    <div class="agent-count-label">Uptime</div>
                </div>
            </div>
        </div>
        
        <div class="alerts">
            <h2>⚠️ Recent Alerts</h2>
            {self._generate_alerts_html(recent_alerts)}
        </div>
        
        <div class="timestamp">
            Generated: {datetime.utcnow().isoformat()} UTC
        </div>
    </div>
</body>
</html>"""
        
        return html
    
    def _generate_alerts_html(self, alerts: List[Dict[str, Any]]) -> str:
        """Generate HTML for alerts."""
        
        if not alerts:
            return '<p style="color: #00ff88;">No active alerts. System operating normally.</p>'
        
        html = ''
        for alert in alerts[:10]:  # Show last 10 alerts
            severity = alert.get('severity', 'info').lower()
            severity_class = f'alert-{severity}'
            
            html += f"""<div class="alert-item">
                <span class="alert-severity {severity_class}">{severity.upper()}</span>
                <span style="color: #00ff88;">{alert.get('message', 'Unknown alert')}</span>
            </div>"""
        
        return html
    
    def save_dashboard(self, html: str, filename: str = "consciousness_dashboard.html"):
        """Save dashboard to file."""
        filepath = self.output_dir / filename
        with open(filepath, 'w') as f:
            f.write(html)
        return str(filepath)


# ============================================================================
# CONVENIENCE FUNCTION
# ============================================================================

def generate_dashboard(
    consciousness_level: float,
    prana: float,
    klesha: float,
    harmony: float,
    resilience: float,
    agents_online: int,
    agents_total: int,
    recent_alerts: List[Dict[str, Any]] = None
) -> str:
    """Generate and save consciousness dashboard."""
    
    generator = DashboardGenerator()
    html = generator.generate_consciousness_dashboard(
        consciousness_level,
        prana,
        klesha,
        harmony,
        resilience,
        agents_online,
        agents_total,
        recent_alerts
    )
    
    filepath = generator.save_dashboard(html)
    return filepath
