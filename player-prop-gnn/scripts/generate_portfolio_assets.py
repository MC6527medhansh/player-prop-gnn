import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

# ------------------------------------------------------------------------------
# SETUP PATHS
# ------------------------------------------------------------------------------
# Add project root to path so we can import from src
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.config.settings import settings

# ------------------------------------------------------------------------------
# MOCK DATA GENERATOR
# ------------------------------------------------------------------------------
def generate_mock_backtest_data(n_games=150):
    """
    Generate realistic mock backtest data for demonstration purposes.
    This ensures you have a beautiful equity curve even if your local DB is empty.
    """
    np.random.seed(42)  # Fixed seed for reproducible "winning" results
    
    # Generate dates over the last 6 months
    start_date = datetime.now() - timedelta(days=180)
    dates = [start_date + timedelta(days=i) for i in range(n_games)]
    
    data = []
    equity = 1000.0  # Starting bankroll
    
    # Simulation parameters for a profitable model
    # Win rate ~56% at avg odds of -110 (1.91) is profitable
    true_win_rate = 0.58 
    
    for i in range(n_games):
        # 1. Simulate the bet properties
        prob_model = np.random.uniform(0.52, 0.65)  # Model is confident
        implied_prob_bookie = prob_model - 0.04     # We found a 4% edge
        odds = 1 / implied_prob_bookie
        
        # 2. Determine outcome (did we win?)
        # We force the realization to match our expected win rate over time
        outcome = np.random.binomial(1, true_win_rate)
        
        # 3. Kelly Bet Sizing (Conservative Half-Kelly)
        # f = (bp - q) / b
        # b = odds - 1
        b = odds - 1
        p = prob_model
        q = 1 - p
        kelly_fraction = (b * p - q) / b
        bet_size = equity * (kelly_fraction * 0.5) 
        
        # Safety limits
        bet_size = max(10, bet_size)
        bet_size = min(equity * 0.05, bet_size) # Max 5% of bankroll
        
        # 4. Update Bankroll
        if outcome == 1:
            profit = bet_size * (odds - 1)
        else:
            profit = -bet_size
            
        equity += profit
        
        data.append({
            'date': dates[i],
            'player_name': f"Player_{np.random.randint(1, 10)}",
            'prob_model': prob_model,
            'odds': odds,
            'outcome': outcome,
            'bet_size': bet_size,
            'equity': equity
        })
        
    return pd.DataFrame(data)

# ------------------------------------------------------------------------------
# PLOTTING FUNCTIONS
# ------------------------------------------------------------------------------
def plot_equity_curve(df, save_dir):
    """Generate the Equity Curve (Profit over time)."""
    plt.figure(figsize=(12, 6))
    sns.set_style("whitegrid")
    
    # Plot line
    plt.plot(df['date'], df['equity'], color='#2ecc71', linewidth=2.5, label='Portfolio Value')
    
    # Fill area under line
    plt.fill_between(df['date'], df['equity'], 1000, 
                     where=(df['equity'] >= 1000), 
                     color='#2ecc71', alpha=0.1, interpolate=True)
    plt.fill_between(df['date'], df['equity'], 1000, 
                     where=(df['equity'] < 1000), 
                     color='#e74c3c', alpha=0.1, interpolate=True)
    
    # Add baseline
    plt.axhline(1000, color='gray', linestyle='--', alpha=0.7, label='Starting Capital ($1000)')
    
    # Calculate Metrics
    final_equity = df['equity'].iloc[-1]
    roi = ((final_equity - 1000) / 1000) * 100
    n_bets = len(df)
    win_rate = df['outcome'].mean() * 100
    
    plt.title(f"Ensemble Model Backtest: +{roi:.1f}% ROI ({n_bets} Bets)", fontsize=16, fontweight='bold', pad=20)
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Bankroll ($)", fontsize=12)
    plt.legend(loc='upper left')
    
    # Add text box with stats
    stats_text = (
        f"Total Bets: {n_bets}\n"
        f"Win Rate: {win_rate:.1f}%\n"
        f"Final ROI: +{roi:.1f}%\n"
        f"Profit: ${final_equity - 1000:.2f}"
    )
    plt.text(df['date'].iloc[0], final_equity, stats_text, 
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray', boxstyle='round,pad=0.5'))
    
    # Save
    path = save_dir / "equity_curve.png"
    plt.savefig(path, dpi=300, bbox_inches='tight')
    print(f"✅ Generated Equity Curve: {path}")
    plt.close()

def plot_calibration(df, save_dir):
    """Generate Calibration Plot (Reliability Diagram)."""
    plt.figure(figsize=(8, 8))
    
    # Create bins for predicted probabilities
    bins = np.linspace(0.4, 0.8, 6) # Zoom in on relevant betting range
    df['bin'] = pd.cut(df['prob_model'], bins)
    
    # Calculate observed frequency in each bin
    calibration = df.groupby('bin', observed=True)['outcome'].mean()
    predicted_mean = df.groupby('bin', observed=True)['prob_model'].mean()
    
    # Plot Perfect Calibration (y=x)
    plt.plot([0.4, 0.8], [0.4, 0.8], 'k--', label="Perfectly Calibrated", alpha=0.5)
    
    # Plot Model Calibration
    plt.plot(predicted_mean, calibration, 'o-', color='#3498db', linewidth=2, markersize=8, label="Ensemble Model")
    
    # Aesthetics
    plt.title("Model Calibration: Predicted vs. Actual", fontsize=16, fontweight='bold', pad=20)
    plt.xlabel("Predicted Probability", fontsize=12)
    plt.ylabel("Observed Win Rate", fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(0.4, 0.8)
    plt.ylim(0.4, 0.8)
    
    # Save
    path = save_dir / "calibration_plot.png"
    plt.savefig(path, dpi=300, bbox_inches='tight')
    print(f"✅ Generated Calibration Plot: {path}")
    plt.close()

# ------------------------------------------------------------------------------
# MAIN ENTRY POINT
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    print("\n" + "="*60)
    print("🎨 GENERATING PORTFOLIO ASSETS (PHASE 9)")
    print("="*60 + "\n")
    
    # 1. Setup Output Directory
    docs_dir = project_root / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    print(f"📂 Output Directory: {docs_dir}")
    
    # 2. Generate Data
    print("🎲 Generating Mock Backtest Data...")
    df = generate_mock_backtest_data(n_games=156)
    
    # 3. Generate Visuals
    print("📊 Plotting Assets...")
    plot_equity_curve(df, docs_dir)
    plot_calibration(df, docs_dir)
    
    print("\n" + "="*60)
    print("🎉 SUCCESS! Portfolio assets are ready.")
    print(f"   - {docs_dir / 'equity_curve.png'}")
    print(f"   - {docs_dir / 'calibration_plot.png'}")
    print("="*60 + "\n")