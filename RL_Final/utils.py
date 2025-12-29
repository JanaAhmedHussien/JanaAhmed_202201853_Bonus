import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import List, Dict, Any
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

def plot_training_progress(rewards_history: List[float], title: str = "Training Progress") -> go.Figure:
    """Create interactive plot of training progress"""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Rewards per Episode', 'Moving Average (100 episodes)',
                       'Cumulative Rewards', 'Reward Distribution'),
        vertical_spacing=0.15,
        horizontal_spacing=0.1
    )
    
    episodes = list(range(1, len(rewards_history) + 1))
    
    # 1. Raw rewards
    fig.add_trace(
        go.Scatter(x=episodes, y=rewards_history, mode='lines',
                  name='Reward', line=dict(color='blue', width=1)),
        row=1, col=1
    )
    
    # 2. Moving average
    if len(rewards_history) > 100:
        window = 100
        moving_avg = np.convolve(rewards_history, np.ones(window)/window, mode='valid')
        fig.add_trace(
            go.Scatter(x=list(range(window, len(rewards_history) + 1)), 
                      y=moving_avg, mode='lines',
                      name=f'{window}-ep MA', line=dict(color='red', width=2)),
            row=1, col=2
        )
    
    # 3. Cumulative rewards
    cumulative_rewards = np.cumsum(rewards_history)
    fig.add_trace(
        go.Scatter(x=episodes, y=cumulative_rewards, mode='lines',
                  name='Cumulative', line=dict(color='green', width=2)),
        row=2, col=1
    )
    
    # 4. Histogram
    fig.add_trace(
        go.Histogram(x=rewards_history, nbinsx=20, name='Distribution',
                    marker_color='orange'),
        row=2, col=2
    )
    
    # Update layout
    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor='center'),
        showlegend=True,
        height=600,
        template='plotly_white'
    )
    
    # Update axes labels
    fig.update_xaxes(title_text="Episode", row=1, col=1)
    fig.update_yaxes(title_text="Reward", row=1, col=1)
    
    fig.update_xaxes(title_text="Episode", row=1, col=2)
    fig.update_yaxes(title_text="Reward", row=1, col=2)
    
    fig.update_xaxes(title_text="Episode", row=2, col=1)
    fig.update_yaxes(title_text="Cumulative Reward", row=2, col=1)
    
    fig.update_xaxes(title_text="Reward", row=2, col=2)
    fig.update_yaxes(title_text="Frequency", row=2, col=2)
    
    return fig

def display_policy(policy, env_name: str) -> pd.DataFrame:
    """Display policy in a readable format"""
    if callable(policy):
        # For function policies (continuous states)
        policy_df = pd.DataFrame({
            'Policy Type': ['Function'],
            'Description': [f'Policy is a function for {env_name}']
        })
    elif isinstance(policy, np.ndarray):
        if policy.ndim == 1:
            # Deterministic policy
            policy_df = pd.DataFrame({
                'State': range(len(policy)),
                'Action': policy
            })
        elif policy.ndim == 2:
            # Stochastic policy
            policy_df = pd.DataFrame(policy)
            policy_df.columns = [f'Action {i}' for i in range(policy.shape[1])]
            policy_df.insert(0, 'State', range(len(policy)))
    else:
        policy_df = pd.DataFrame({
            'Policy Type': [str(type(policy))],
            'Description': ['Unknown policy format']
        })
    
    return policy_df

def display_value_function(V, env_name: str) -> pd.DataFrame:
    """Display value function"""
    if isinstance(V, np.ndarray):
        if V.ndim == 1:
            value_df = pd.DataFrame({
                'State': range(len(V)),
                'Value': V
            })
        else:
            value_df = pd.DataFrame({
                'State': [f'State {i}' for i in range(V.shape[0])],
                'Mean Value': V.mean(axis=1),
                'Std Value': V.std(axis=1)
            })
    else:
        value_df = pd.DataFrame({
            'Value Function Type': [str(type(V))],
            'Description': [f'Value function for {env_name}']
        })
    
    return value_df

def create_stats_table(algorithm_stats: Dict[str, Any]) -> pd.DataFrame:
    """Create statistics table"""
    stats_data = []
    
    for key, value in algorithm_stats.items():
        if isinstance(value, dict):
            for subkey, subvalue in value.items():
                stats_data.append({
                    'Metric': f'{key}.{subkey}',
                    'Value': f'{subvalue:.4f}' if isinstance(subvalue, (int, float)) else str(subvalue)
                })
        else:
            stats_data.append({
                'Metric': key,
                'Value': f'{value:.4f}' if isinstance(value, (int, float)) else str(value)
            })
    
    return pd.DataFrame(stats_data)

def plot_value_function_heatmap(V, env_name: str) -> go.Figure:
    """Create heatmap of value function (for grid worlds)"""
    if not isinstance(V, np.ndarray) or V.ndim != 1:
        # Return empty figure if not 1D array
        fig = go.Figure()
        fig.add_annotation(text="Heatmap not available for this value function",
                          xref="paper", yref="paper",
                          x=0.5, y=0.5, showarrow=False)
        return fig
    
    # Try to reshape for grid worlds (assuming square grid)
    n = int(np.sqrt(len(V)))
    if n * n == len(V):
        V_grid = V.reshape((n, n))
        
        fig = go.Figure(data=go.Heatmap(
            z=V_grid,
            colorscale='Viridis',
            colorbar=dict(title="Value")
        ))
        
        fig.update_layout(
            title=f"Value Function Heatmap for {env_name}",
            xaxis_title="X",
            yaxis_title="Y",
            height=500
        )
    else:
        # Line plot for non-grid values
        fig = go.Figure(data=go.Scatter(
            y=V,
            mode='lines+markers',
            name='Value'
        ))
        
        fig.update_layout(
            title=f"Value Function for {env_name}",
            xaxis_title="State",
            yaxis_title="Value",
            height=400
        )
    
    return fig