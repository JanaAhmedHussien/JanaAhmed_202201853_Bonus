import streamlit as st
import gymnasium as gym
import numpy as np
import pandas as pd
import time
import sys
import os
from typing import Dict, Any, List, Tuple, Optional
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from PIL import Image
import io
import base64
import psutil
import gc

# Add current directory to path
sys.path.append(os.path.dirname(__file__))

# Import algorithms
from algorithms.qlearning import QLearning
from algorithms.sarsa import SARSA
from algorithms.monte_carlo import MonteCarlo
from algorithms.policy_iteration import PolicyIteration
from algorithms.value_iteration import ValueIteration

# Page configuration
st.set_page_config(
    page_title="RL Algorithm Suite - Enhanced",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS with better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
        padding: 1rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: bold;
    }
    .algorithm-step {
        background-color: #f0f7ff;
        border-left: 4px solid #1E88E5;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .value-display {
        background-color: #e3f2fd;
        padding: 0.75rem;
        border-radius: 0.5rem;
        margin: 0.25rem 0;
        font-family: 'Courier New', monospace;
        font-size: 0.9rem;
        border: 1px solid #bbdefb;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 0.5rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
    }
    .step-info {
        background-color: #e8f5e8;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        border-left: 4px solid #4CAF50;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .warning-box {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .info-box {
        background-color: #d1ecf1;
        border-left: 4px solid #17a2b8;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .memory-meter {
        width: 100%;
        height: 20px;
        background-color: #e0e0e0;
        border-radius: 10px;
        overflow: hidden;
        margin: 0.5rem 0;
    }
    .memory-fill {
        height: 100%;
        background: linear-gradient(90deg, #4CAF50, #8BC34A);
        transition: width 0.3s ease;
    }
    .tab-content {
        padding: 1rem;
        background-color: white;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">🧠 RL Algorithm Suite - Enhanced Visualization</h1>', unsafe_allow_html=True)
st.markdown("---")

# Initialize session state with memory optimization
if 'training_state' not in st.session_state:
    st.session_state.training_state = {
        'is_training': False,
        'current_iteration': 0,
        'algorithm': None,
        'iteration_details': [],
        'env': None,
        'optimal_policy': None,
        'value_function': None,
        'q_function': None,
        'policy_animation_frames': [],
        'policy_execution_details': [],
        'animation_speed': 0.5,
        'last_memory_check': time.time(),
        'memory_warning': False
    }

# Memory management functions
def check_memory_usage():
    """Check current memory usage and warn if high"""
    process = psutil.Process()
    memory_info = process.memory_info()
    memory_percent = memory_info.rss / psutil.virtual_memory().total * 100
    
    current_time = time.time()
    if current_time - st.session_state.training_state['last_memory_check'] > 5:  # Check every 5 seconds
        st.session_state.training_state['last_memory_check'] = current_time
        
        if memory_percent > 80:
            st.session_state.training_state['memory_warning'] = True
            return False, memory_percent
        else:
            st.session_state.training_state['memory_warning'] = False
            return True, memory_percent
    
    return not st.session_state.training_state['memory_warning'], memory_percent

def cleanup_memory():
    """Clean up memory-intensive objects"""
    gc.collect()
    
    # Clear large arrays from algorithm if exists
    if st.session_state.training_state['algorithm'] is not None:
        try:
            st.session_state.training_state['algorithm'].cleanup_memory()
        except:
            pass
    
    # Limit stored frames
    if len(st.session_state.training_state['policy_animation_frames']) > 50:
        st.session_state.training_state['policy_animation_frames'] = \
            st.session_state.training_state['policy_animation_frames'][-50:]
    
    # Limit iteration details
    if len(st.session_state.training_state['iteration_details']) > 20:
        st.session_state.training_state['iteration_details'] = \
            st.session_state.training_state['iteration_details'][-20:]
    
    gc.collect()

# Helper functions
def create_algorithm(env, algorithm_key: str, params: Dict[str, Any]):
    """Create algorithm instance"""
    params['name'] = algorithm_key.upper()
    
    if algorithm_key == "qlearning":
        return QLearning(env, params)
    elif algorithm_key == "sarsa":
        return SARSA(env, params)
    elif algorithm_key == "monte_carlo":
        return MonteCarlo(env, params)
    elif algorithm_key == "policy_iteration":
        return PolicyIteration(env, params)
    elif algorithm_key == "value_iteration":
        return ValueIteration(env, params)


    else:
        raise ValueError(f"Unknown algorithm: {algorithm_key}")

def display_algorithm_step(step_detail: Dict[str, Any], algorithm_name: str):
    """Display algorithm-specific step details with complete information"""
    with st.container():
        st.markdown(f"<div class='algorithm-step'>", unsafe_allow_html=True)
        
        # Common information
        cols = st.columns(4)
        with cols[0]:
            if 'state_idx' in step_detail:
                st.markdown(f"**State Index:** `{step_detail['state_idx']}`")
            if 'state' in step_detail:
                st.markdown(f"**State:** `{str(step_detail['state'])[:50]}`")
        
        with cols[1]:
            if 'action' in step_detail:
                st.markdown(f"**Action:** `{step_detail['action']}`")
            if 'next_action' in step_detail and step_detail['next_action'] is not None:
                st.markdown(f"**Next Action:** `{step_detail['next_action']}`")
        
        with cols[2]:
            if 'reward' in step_detail:
                st.markdown(f"**Reward:** `{step_detail['reward']:.4f}`")
        
        with cols[3]:
            if 'epsilon' in step_detail:
                st.markdown(f"**ε:** `{step_detail['epsilon']:.4f}`")
            if 'done' in step_detail:
                st.markdown(f"**Done:** `{step_detail['done']}`")
        
        st.markdown("---")
        
        # Q-Learning/SARSA specific details
        if 'old_Q(s,a)' in step_detail:
            st.markdown("#### Q-Value Details")
            
            cols = st.columns(4)
            with cols[0]:
                st.markdown(f"**Old Q(s,a):** `{step_detail.get('old_Q(s,a)', 0):.6f}`")
                if 'selected_Q' in step_detail:
                    st.markdown(f"**Selected Q:** `{step_detail['selected_Q']:.6f}`")
            
            with cols[1]:
                if 'max_Q(s\',a\')' in step_detail:
                    st.markdown(f"**max Q(s',a'):** `{step_detail['max_Q(s\',a\')']:.6f}`")
                elif 'Q(s\',a\')' in step_detail:
                    st.markdown(f"**Q(s',a'):** `{step_detail['Q(s\',a\')']:.6f}`")
            
            with cols[2]:
                st.markdown(f"**TD Target:** `{step_detail.get('TD_target', 0):.6f}`")
                st.markdown(f"**TD Error:** `{step_detail.get('TD_error', 0):.6f}`")
            
            with cols[3]:
                if 'new_Q(s,a)' in step_detail:
                    st.markdown(f"**New Q(s,a):** `{step_detail['new_Q(s,a)']:.6f}`")
                if 'learning_rate' in step_detail:
                    st.markdown(f"**α:** `{step_detail['learning_rate']:.4f}`")
        
        # Value function details
        if 'old_V(s)' in step_detail:
            st.markdown("#### Value Function Details")
            
            cols = st.columns(3)
            with cols[0]:
                st.markdown(f"**Old V(s):** `{step_detail.get('old_V(s)', 0):.6f}`")
            with cols[1]:
                st.markdown(f"**V(s'):** `{step_detail.get('V(s\')', 0):.6f}`")
            with cols[2]:
                st.markdown(f"**New V(s):** `{step_detail.get('new_V(s)', step_detail.get('old_V(s)', 0)):.6f}`")
            
            if 'TD_target' in step_detail:
                cols = st.columns(2)
                with cols[0]:
                    st.markdown(f"**TD Target:** `{step_detail['TD_target']:.6f}`")
                with cols[1]:
                    st.markdown(f"**TD Error:** `{step_detail.get('TD_error', 0):.6f}`")
        
        # Monte Carlo specific
        if 'return_G' in step_detail:
            st.markdown("#### Monte Carlo Details")
            
            cols = st.columns(3)
            with cols[0]:
                st.markdown(f"**Return G:** `{step_detail['return_G']:.6f}`")
            with cols[1]:
                st.markdown(f"**Samples:** `{step_detail.get('samples_count', 0)}`")
            with cols[2]:
                st.markdown(f"**Mean Return:** `{step_detail.get('mean_return', 0):.6f}`")
        
        # Policy details
        if 'policy_probs' in step_detail:
            st.markdown("#### Policy Details")
            
            probs = step_detail['policy_probs']
            cols = st.columns(len(probs))
            for i, prob in enumerate(probs):
                with cols[i]:
                    st.markdown(f"**π(a{i}):** `{prob:.4f}`")
            
            if 'optimal_action' in step_detail:
                st.markdown(f"**Optimal Action:** `{step_detail['optimal_action']}`")
        
        # Show all Q-values if available
        if 'all_Q_values' in step_detail:
            st.markdown("#### All Q-values for State")
            
            q_values = step_detail['all_Q_values']
            cols = st.columns(len(q_values))
            for i, q_val in enumerate(q_values):
                with cols[i]:
                    # Color code based on value
                    color = "green" if q_val == max(q_values) else "black"
                    st.markdown(f"<span style='color:{color};'>**Q(a{i}):** `{q_val:.6f}`</span>", 
                               unsafe_allow_html=True)
        
        # Action probabilities for exploration
        if 'action_probabilities' in step_detail:
            st.markdown("#### Action Probabilities (ε-greedy)")
            
            probs = step_detail['action_probabilities']
            cols = st.columns(len(probs))
            for i, prob in enumerate(probs):
                with cols[i]:
                    st.markdown(f"**P(a{i}):** `{prob:.4f}`")
        
        # Visit counts
        if 'visit_count' in step_detail:
            st.markdown(f"**Visit Count:** `{step_detail['visit_count']}`")
        
        # Additional metadata
        if 'step_type' in step_detail:
            st.markdown(f"**Step Type:** `{step_detail['step_type']}`")
        
        st.markdown("</div>", unsafe_allow_html=True)

def plot_training_progress(iteration_details: List[Dict[str, Any]]):
    """Plot training progress with enhanced information"""
    if not iteration_details:
        return None
    
    iterations = [d['iteration'] for d in iteration_details]
    mean_rewards = [d['mean_reward'] for d in iteration_details]
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Mean Reward', 'Value Function Stats', 
                       'Q-Table Stats', 'Exploration Rate'),
        vertical_spacing=0.15,
        horizontal_spacing=0.15
    )
    
    # 1. Mean reward
    fig.add_trace(
        go.Scatter(
            x=iterations,
            y=mean_rewards,
            mode='lines+markers',
            name='Mean Reward',
            line=dict(color='blue', width=2),
            marker=dict(size=6)
        ),
        row=1, col=1
    )
    
    # 2. Value function statistics
    if 'V_mean' in iteration_details[0]:
        v_means = [d.get('V_mean', 0) for d in iteration_details]
        v_maxs = [d.get('V_max', 0) for d in iteration_details]
        
        fig.add_trace(
            go.Scatter(x=iterations, y=v_means, mode='lines', 
                      name='Mean V', line=dict(color='green')),
            row=1, col=2
        )
        fig.add_trace(
            go.Scatter(x=iterations, y=v_maxs, mode='lines', 
                      name='Max V', line=dict(color='red')),
            row=1, col=2
        )
    
    # 3. Q-table statistics
    if 'Q_table_mean' in iteration_details[0]:
        q_means = [d.get('Q_table_mean', 0) for d in iteration_details]
        q_maxs = [d.get('Q_table_max', 0) for d in iteration_details]
        
        fig.add_trace(
            go.Scatter(x=iterations, y=q_means, mode='lines',
                      name='Mean Q', line=dict(color='purple')),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(x=iterations, y=q_maxs, mode='lines',
                      name='Max Q', line=dict(color='orange')),
            row=2, col=1
        )
    
    # 4. Exploration rate
    if 'epsilon' in iteration_details[0]:
        epsilons = [d.get('epsilon', 0) for d in iteration_details]
        fig.add_trace(
            go.Scatter(x=iterations, y=epsilons, mode='lines',
                      name='ε', line=dict(color='brown', width=2)),
            row=2, col=2
        )
    
    fig.update_layout(
        title="Training Progress Dashboard",
        height=700,
        template='plotly_white',
        showlegend=True
    )
    
    return fig

def plot_value_function_improved(V, algorithm_name: str, env_name: str):
    """Create improved value function visualization"""
    if V is None or len(V) == 0:
        return None
    
    V_clean = np.array(V).flatten()
    
    # Create histogram and line plot
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Value Function Distribution', 'Sorted Value Function'),
        column_widths=[0.4, 0.6]
    )
    
    # Histogram
    fig.add_trace(
        go.Histogram(
            x=V_clean,
            nbinsx=30,
            name='Distribution',
            marker_color='skyblue',
            opacity=0.7
        ),
        row=1, col=1
    )
    
    # Sorted values
    sorted_V = np.sort(V_clean)
    fig.add_trace(
        go.Scatter(
            y=sorted_V,
            mode='lines+markers',
            name='Sorted Values',
            line=dict(color='green', width=2),
            marker=dict(size=4)
        ),
        row=1, col=2
    )
    
    fig.update_layout(
        title=f"Value Function Analysis - {algorithm_name} on {env_name}",
        height=400,
        template='plotly_white',
        showlegend=False
    )
    
    fig.update_xaxes(title_text="Value", row=1, col=1)
    fig.update_yaxes(title_text="Frequency", row=1, col=1)
    fig.update_xaxes(title_text="State Rank", row=1, col=2)
    fig.update_yaxes(title_text="Value", row=1, col=2)
    
    return fig

def plot_q_function_improved(Q, algorithm_name: str):
    """Create improved Q-function visualization"""
    if Q is None:
        return None
    
    if Q.ndim != 2:
        return None
    
    # Calculate statistics for annotation
    mean_q = np.mean(Q)
    max_q = np.max(Q)
    min_q = np.min(Q)
    non_zero = np.count_nonzero(np.abs(Q) > 1e-10)
    total_cells = Q.size
    
    # Create heatmap with better colorscale
    fig = go.Figure(data=go.Heatmap(
        z=Q,
        colorscale='Viridis',
        colorbar=dict(
            title="Q-value",
            title_font=dict(size=12),
            tickfont=dict(size=10),
            thickness=20,
            len=0.8
        ),
        hovertemplate='State: %{y}<br>Action: %{x}<br>Q-value: %{z:.4f}<extra></extra>'
    ))
    
    fig.update_layout(
        title=dict(
            text=f"Q-Function Heatmap - {algorithm_name}",
            font=dict(size=16)
        ),
        xaxis_title="Action",
        yaxis_title="State",
        height=500,
        width=800,
        template='plotly_white',
        margin=dict(l=50, r=150, t=50, b=50)  # Extra space on right for annotation
    )
    
    # Add statistics as annotation on the side
    stats_text = (
        f"<b>Q-Function Statistics</b><br>"
        f"Mean Q: {mean_q:.4f}<br>"
        f"Max Q: {max_q:.4f}<br>"
        f"Min Q: {min_q:.4f}<br>"
        f"Std Q: {np.std(Q):.4f}<br>"
        f"Non-zero: {non_zero}/{total_cells}<br>"
        f"Positive: {np.sum(Q > 0)}<br>"
        f"Negative: {np.sum(Q < 0)}"
    )
    
    fig.add_annotation(
        text=stats_text,
        xref="paper", yref="paper",
        x=1.05,  # Position to the right of the plot
        y=0.5,
        showarrow=False,
        align="left",
        bgcolor="rgba(255, 255, 255, 0.8)",
        bordercolor="black",
        borderwidth=1,
        borderpad=10,
        font=dict(size=11)
    )
    
    # Add grid lines for better readability
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(0,0,0,0.1)')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(0,0,0,0.1)')
    
    return fig

def execute_policy_with_animation(algorithm, env_name: str, env_kwargs: Dict[str, Any], 
                                 max_steps: int = 50, capture_frames: bool = True):
    """Execute optimal policy with memory-efficient animation capture"""
    try:
        # Create environment for visualization
        viz_env = gym.make(env_name, render_mode="rgb_array", **env_kwargs)
        
        # Reset environment
        state, _ = viz_env.reset()
        done = False
        total_reward = 0
        frames = []
        execution_details = []
        
        step_count = 0
        
        while not done and step_count < max_steps:
            # Capture frame if requested and memory allows
            if capture_frames and len(frames) < 100:  # Limit frames to prevent memory explosion
                try:
                    frame = viz_env.render()
                    if frame is not None:
                        # Compress image to reduce memory
                        img = Image.fromarray(frame)
                        # Resize if large
                        if img.size[0] > 400:
                            img = img.resize((400, int(400 * img.size[1] / img.size[0])))
                        frames.append(img)
                except Exception as e:
                    st.warning(f"Frame capture error: {e}")
            
            # Get action from optimal policy
            try:
                action = algorithm.get_action(state, epsilon_greedy=False)
            except:
                action = viz_env.action_space.sample()
            
            # Take action
            next_state, reward, terminated, truncated, _ = viz_env.step(action)
            done = terminated or truncated
            
            # Get algorithm step details
            try:
                step_detail = algorithm.get_step_details(state, action, reward, next_state, done)
            except:
                step_detail = {
                    'step': step_count + 1,
                    'state': state,
                    'action': action,
                    'reward': reward,
                    'next_state': next_state,
                    'done': done
                }
            
            step_detail['total_reward_so_far'] = total_reward + reward
            step_detail['step_number'] = step_count + 1
            
            execution_details.append(step_detail)
            
            # Update state and reward
            state = next_state
            total_reward += reward
            step_count += 1
        
        # Capture final frame
        if capture_frames and len(frames) < 100:
            try:
                final_frame = viz_env.render()
                if final_frame is not None:
                    img = Image.fromarray(final_frame)
                    if img.size[0] > 400:
                        img = img.resize((400, int(400 * img.size[1] / img.size[0])))
                    frames.append(img)
            except:
                pass
        
        # Cleanup
        viz_env.close()
        
        # Add final summary
        execution_details.append({
            'step': 'Final',
            'total_reward': total_reward,
            'total_steps': step_count,
            'success': done and total_reward > 0,
            'episode_completed': done
        })
        
        return frames, execution_details, total_reward
        
    except Exception as e:
        st.error(f"Error during policy execution: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
        return [], [], 0

# Sidebar Configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Environment selection
    env_options = {
        "FrozenLake-v1": {
            "name": "FrozenLake (4x4)",
            "description": "Navigate from S to G without falling in holes",
            "type": "discrete",
            "render_mode": "rgb_array",
            "kwargs": {"is_slippery": False, "map_name": "4x4"}
        },
        "Taxi-v3": {
            "name": "Taxi",
            "description": "Pick up and drop off passenger",
            "type": "discrete",
            "render_mode": "rgb_array",
            "kwargs": {}
        },
        "CartPole-v1": {
            "name": "CartPole",
            "description": "Balance pole on cart",
            "type": "continuous",
            "render_mode": "rgb_array",
            "kwargs": {}
        },
        "MountainCar-v0": {
            "name": "MountainCar",
            "description": "Drive car to hilltop",
            "type": "continuous",
            "render_mode": "rgb_array",
            "kwargs": {}
        },
        "CliffWalking-v1": {
            "name": "CliffWalking",
            "description": "Walk along cliff without falling",
            "type": "discrete",
            "render_mode": "rgb_array",
            "kwargs": {}
        }
    }
    
    selected_env_key = st.selectbox(
        "Select Environment",
        list(env_options.keys()),
        format_func=lambda x: env_options[x]["name"]
    )
    
    env_info = env_options[selected_env_key]
    
    # Display environment info
    with st.expander("Environment Info"):
        st.write(f"**Type:** {env_info['type']}")
        st.write(f"**Description:** {env_info['description']}")
    
    # Algorithm selection
    algorithm_options = {
        "qlearning": "Q-Learning",
        "sarsa": "SARSA", 
        "monte_carlo": "Monte Carlo",
        "policy_iteration": "Policy Iteration",
        "value_iteration": "Value Iteration",
    }
    
    selected_algorithm_key = st.selectbox(
        "Select Algorithm",
        list(algorithm_options.keys()),
        format_func=lambda x: algorithm_options[x]
    )
    
    # Display algorithm info
    with st.expander("Algorithm Info"):
        alg_name = algorithm_options[selected_algorithm_key]
        st.write(f"**Selected:** {alg_name}")
        
        if selected_algorithm_key in ["qlearning", "sarsa"]:
            st.write("**Type:** Temporal Difference (Model-free)")
            st.write("**Policy:** ε-greedy")
        elif selected_algorithm_key == "monte_carlo":
            st.write("**Type:** Monte Carlo (Model-free)")
            st.write("**Policy:** Exploring starts")
        elif selected_algorithm_key in ["policy_iteration", "value_iteration"]:
            st.write("**Type:** Dynamic Programming (Model-based)")
            st.write("**Requires:** Transition model")

    
    st.markdown("---")
    st.header("📊 Hyperparameters")
    
    # Hyperparameter configuration
    col1, col2 = st.columns(2)
    with col1:
        learning_rate = st.slider("α (Learning Rate)", 0.001, 1.0, 0.1, 0.001, 
                                 help="Step size for updates")
        discount_factor = st.slider("γ (Discount)", 0.1, 0.99, 0.9, 0.01,
                                   help="Future reward discount factor")
    with col2:
        epsilon = st.slider("ε (Exploration)", 0.0, 1.0, 1.0, 0.05,
                           help="Exploration rate for ε-greedy")
        epsilon_decay = st.slider("ε Decay", 0.9, 0.999, 0.995, 0.001,
                                 help="Decay rate for exploration")
    
    col1, col2 = st.columns(2)
    with col1:
        iterations = st.number_input("Iterations", 1, 100, 10,
                                    help="Number of training iterations")
    with col2:
        episodes_per_iteration = st.number_input("Episodes/Iteration", 1, 1000, 100,
                                                help="Episodes per training iteration")
    
    if env_info['type'] == 'continuous':
        divisions = st.slider("State Discretization", 5, 50, 15,
                             help="Number of bins for continuous state discretization")
    else:
        divisions = 10
    

    
    # Memory management settings
    st.markdown("---")
    st.header("🧠 Memory Management")
    
    capture_frames = st.checkbox("Capture Animation Frames", value=True,
                                help="Disable to save memory during policy execution")
    max_frames = st.slider("Max Frames to Store", 10, 200, 50,
                          help="Limit frames to prevent memory overflow")
    
    # Check current memory usage
    memory_ok, memory_percent = check_memory_usage()
    
    if not memory_ok:
        st.error(f"⚠️ High memory usage: {memory_percent:.1f}%")
        st.button("🔄 Force Garbage Collection", on_click=cleanup_memory)
    
    st.markdown(f"<div class='memory-meter'><div class='memory-fill' style='width:{memory_percent}%'></div></div>", 
                unsafe_allow_html=True)
    st.caption(f"Memory Usage: {memory_percent:.1f}%")
    
    st.markdown("---")
    st.header("🎮 Control")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("▶️ Start Training", type="primary", width='stretch'):
            # Initialize training state
            st.session_state.training_state = {
                'is_training': True,
                'current_iteration': 0,
                'algorithm': None,
                'iteration_details': [],
                'env': None,
                'optimal_policy': None,
                'value_function': None,
                'q_function': None,
                'policy_animation_frames': [],
                'policy_execution_details': [],
                'animation_speed': 0.5,
                'last_memory_check': time.time(),
                'memory_warning': False
            }
            st.rerun()
    
    with col2:
        if st.button("⏹️ Stop Training", width='stretch'):
            st.session_state.training_state['is_training'] = False
            st.rerun()
    
    if st.button("🔄 Reset All", width='stretch'):
        st.session_state.training_state = {
            'is_training': False,
            'current_iteration': 0,
            'algorithm': None,
            'iteration_details': [],
            'env': None,
            'optimal_policy': None,
            'value_function': None,
            'q_function': None,
            'policy_animation_frames': [],
            'policy_execution_details': [],
            'animation_speed': 0.5,
            'last_memory_check': time.time(),
            'memory_warning': False
        }
        cleanup_memory()
        st.rerun()

# Main Content
# Create tabs
tab1, tab2, tab3, tab4 = st.tabs(["📈 Training", "🔍 Algorithm Details", "📋 Policy Table", "🎬 Animation"])

with tab1:
    # Training Dashboard
    st.subheader(f"{algorithm_options[selected_algorithm_key]} on {env_info['name']}")
    
    # Progress display
    if st.session_state.training_state['is_training']:
        progress = min(1.0, st.session_state.training_state['current_iteration'] / max(1, iterations))
        
        progress_bar = st.progress(progress)
        status_text = st.empty()
        
        current_iter = st.session_state.training_state['current_iteration']
        status_text.text(f"Training in progress... Iteration {current_iter}/{iterations}")
    
    # Display metrics if available
    if st.session_state.training_state['iteration_details']:
        last_iteration = st.session_state.training_state['iteration_details'][-1]
        
        # Metrics row
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Iteration", last_iteration['iteration'])
        with col2:
            st.metric("Mean Reward", f"{last_iteration['mean_reward']:.2f}")
        with col3:
            st.metric("Mean Steps", f"{last_iteration.get('mean_steps', 0):.0f}")
        with col4:
            if 'epsilon' in last_iteration:
                st.metric("Exploration ε", f"{last_iteration['epsilon']:.3f}")
            elif 'policy_stable' in last_iteration:
                st.metric("Policy Stable", "✓" if last_iteration['policy_stable'] else "✗")
        
        # Training progress plot
        fig = plot_training_progress(st.session_state.training_state['iteration_details'])
        if fig:
            st.plotly_chart(fig, width='stretch')
        
        # Value function visualization
        if st.session_state.training_state['value_function'] is not None:
            V = st.session_state.training_state['value_function']
            fig_v = plot_value_function_improved(V, algorithm_options[selected_algorithm_key], env_info['name'])
            if fig_v:
                st.plotly_chart(fig_v, width='stretch')

with tab2:
    # Algorithm Steps Dashboard
    st.subheader("🔍 Algorithm Step Details - Complete Information")
    
    if st.session_state.training_state['iteration_details']:
        # Select iteration to view
        iterations_list = [f"Iteration {d['iteration']} (Reward: {d['mean_reward']:.2f})" 
                          for d in st.session_state.training_state['iteration_details']]
        
        selected_iter_idx = st.selectbox(
            "Select Iteration", 
            range(len(iterations_list)), 
            format_func=lambda x: iterations_list[x], 
            key='iter_select_tab2'
        )
        
        selected_iteration = st.session_state.training_state['iteration_details'][selected_iter_idx]
        
        # Select episode to view
        if 'step_details' in selected_iteration and selected_iteration['step_details']:
            episodes_list = [f"Episode {i+1}" for i in range(len(selected_iteration['step_details']))]
            selected_ep_idx = st.selectbox(
                "Select Episode", 
                range(len(episodes_list)),
                format_func=lambda x: episodes_list[x], 
                key='ep_select_tab2'
            )
            
            step_details = selected_iteration['step_details'][selected_ep_idx]
            
            # Display episode summary
            if step_details:
                last_step = step_details[-1]
                total_reward = sum(step.get('reward', 0) for step in step_details if 'reward' in step)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Steps", len(step_details))
                with col2:
                    st.metric("Total Reward", f"{total_reward:.2f}")
                with col3:
                    if 'epsilon' in last_step:
                        st.metric("ε", f"{last_step['epsilon']:.3f}")
            
            # Display steps with expanders
            st.markdown("### 📝 Step-by-Step Details")
            
            # Limit display for performance
            max_steps_to_show = min(20, len(step_details))
            step_indices = list(range(max_steps_to_show))
            
            if len(step_details) > max_steps_to_show:
                st.info(f"Showing first {max_steps_to_show} of {len(step_details)} steps")
            
            for i in step_indices:
                step = step_details[i]
                with st.expander(f"Step {i+1}: State → Action → Reward", expanded=(i < 3)):
                    display_algorithm_step(step, algorithm_options[selected_algorithm_key])
        else:
            # For model-based algorithms, show different details
            if 'evaluation_details' in selected_iteration:
                st.markdown("### Policy Evaluation Details")
                for eval_detail in selected_iteration['evaluation_details'][:3]:
                    with st.expander(f"Evaluation Iteration {eval_detail['evaluation_iteration']}"):
                        st.write(f"**Max Delta:** {eval_detail['max_delta']:.6f}")
                        if 'state_details' in eval_detail:
                            for state_detail in eval_detail['state_details'][:3]:
                                st.write(f"State {state_detail['state']}: "
                                        f"V(s) = {state_detail['new_V(s)']:.4f} "
                                        f"(Δ = {state_detail['delta']:.6f})")
            
            if 'improvement_details' in selected_iteration:
                st.markdown("### Policy Improvement Details")
                for imp_detail in selected_iteration['improvement_details'][:5]:
                    with st.expander(f"State {imp_detail['state']}"):
                        cols = st.columns(4)
                        with cols[0]:
                            st.write(f"**Old Action:** {imp_detail['old_action']}")
                        with cols[1]:
                            st.write(f"**New Action:** {imp_detail['new_action']}")
                        with cols[2]:
                            st.write(f"**V(s):** {imp_detail['V(s)']:.4f}")
                        with cols[3]:
                            changed = "✓ Changed" if imp_detail['changed'] else "✗ Unchanged"
                            st.write(f"**Status:** {changed}")
                        
                        # Show Q-values
                        if 'q_values' in imp_detail:
                            st.write("**Q-values:**")
                            for a, q_val in enumerate(imp_detail['q_values']):
                                st.write(f"  Action {a}: {q_val:.4f}")
    else:
        st.info("ℹ️ No training data available. Start training to see algorithm steps.")

with tab3:
    # Policy Table Dashboard
    st.subheader("📋 Optimal Policy Table with Complete Information")
    
    if st.session_state.training_state['algorithm'] is not None:
        algorithm = st.session_state.training_state['algorithm']
        
        # Try to get policy table
        try:
            policy_table = algorithm.get_optimal_policy_table()
            
            if policy_table and len(policy_table) > 0:
                # Display as dataframe
                df = pd.DataFrame(policy_table)
                
                # Format columns for better display
                for col in df.columns:
                    if df[col].dtype == 'object':
                        # Truncate long strings
                        df[col] = df[col].apply(lambda x: str(x)[:50] + '...' if len(str(x)) > 50 else str(x))
                
                st.dataframe(df, width='stretch', height=400)
                
                # Download button
                csv = df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Policy Table",
                    data=csv,
                    file_name=f"policy_table_{algorithm_options[selected_algorithm_key]}_{env_info['name']}.csv",
                    mime="text/csv"
                )
                
                # Display statistics
                st.markdown("### 📊 Policy Statistics")
                
                if 'Optimal Action' in df.columns and df['Optimal Action'].dtype in [np.int64, np.float64]:
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        unique_actions = df['Optimal Action'].nunique()
                        st.metric("Unique Actions", unique_actions)
                    with col2:
                        most_common = df['Optimal Action'].mode().iloc[0] if not df.empty else 'N/A'
                        st.metric("Most Common Action", most_common)
                
                if 'V*(s)' in df.columns:
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        mean_v = df['V*(s)'].mean()
                        st.metric("Mean V*(s)", f"{mean_v:.4f}")
                    with col2:
                        max_v = df['V*(s)'].max()
                        st.metric("Max V*(s)", f"{max_v:.4f}")
                    with col3:
                        min_v = df['V*(s)'].min()
                        st.metric("Min V*(s)", f"{min_v:.4f}")
                
                # Visualize policy if discrete
                if 'Optimal Action' in df.columns and df['Optimal Action'].dtype in [np.int64, np.float64]:
                    st.markdown("### 📈 Policy Visualization")
                    
                    # Action distribution
                    action_counts = df['Optimal Action'].value_counts().sort_index()
                    
                    fig = go.Figure(data=[
                        go.Bar(
                            x=action_counts.index,
                            y=action_counts.values,
                            text=action_counts.values,
                            textposition='auto',
                            marker_color='lightblue'
                        )
                    ])
                    
                    fig.update_layout(
                        title="Action Distribution in Optimal Policy",
                        xaxis_title="Action",
                        yaxis_title="Count",
                        height=300
                    )
                    
                    st.plotly_chart(fig, width='stretch')
            else:
                st.warning("⚠️ Policy table is empty or could not be generated.")
                
                # Try alternative display
                try:
                    policy = algorithm.get_policy()
                    if policy is not None:
                        st.write("**Policy Type:**", type(policy))
                        
                        if isinstance(policy, np.ndarray):
                            st.write("**Policy Shape:**", policy.shape)
                            st.write("**Sample values (first 20):**")
                            st.write(policy[:20])
                        elif callable(policy):
                            st.write("**Policy is a function**")
                            # Test the function
                            test_state = np.zeros(env_info.get('state_dim', 1))
                            try:
                                test_action = policy(test_state)
                                st.write(f"**Test output for zero state:** {test_action}")
                            except:
                                st.write("Could not test policy function")
                except Exception as e:
                    st.error(f"Error displaying policy: {e}")
        
        except Exception as e:
            st.error(f"Error generating policy table: {str(e)}")
            import traceback
            with st.expander("Error Details"):
                st.code(traceback.format_exc())
    
    else:
        st.info("ℹ️ No algorithm trained yet. Please train an algorithm first!")

with tab4:
    # Animation Dashboard
    st.subheader("🎬 Policy Execution Animation")
    
    if st.session_state.training_state['algorithm'] is not None:
        algorithm = st.session_state.training_state['algorithm']
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("### 🎯 Quick Statistics")
            
            # Display current algorithm statistics
            try:
                if hasattr(algorithm, 'Q'):
                    q_stats = {
                        'Mean Q': np.mean(algorithm.Q),
                        'Max Q': np.max(algorithm.Q),
                        'Min Q': np.min(algorithm.Q),
                        'Non-zero Q': np.count_nonzero(np.abs(algorithm.Q) > 1e-10)
                    }
                    
                    for stat_name, stat_value in q_stats.items():
                        st.metric(stat_name, f"{stat_value:.4f}" if isinstance(stat_value, float) else stat_value)
                
                if hasattr(algorithm, 'V'):
                    v_stats = {
                        'Mean V': np.mean(algorithm.V),
                        'Max V': np.max(algorithm.V),
                        'Min V': np.min(algorithm.V)
                    }
                    
                    for stat_name, stat_value in v_stats.items():
                        st.metric(stat_name, f"{stat_value:.4f}")
                
                if hasattr(algorithm, 'epsilon'):
                    st.metric("Current ε", f"{algorithm.epsilon:.4f}")
            
            except Exception as e:
                st.warning(f"Could not load all statistics: {e}")
            
            # Q-function visualization
            st.markdown("### 🔢 Q-Function")
            try:
                Q = algorithm.get_q_function()
                if Q is not None and Q.ndim == 2:
                    fig_q = plot_q_function_improved(Q, algorithm_options[selected_algorithm_key])
                    if fig_q:
                        st.plotly_chart(fig_q, width='stretch')
                else:
                    st.info("Q-function not available or not 2D")
            except Exception as e:
                st.info(f"Q-function display not available: {e}")
        
        with col2:
            st.markdown("### 🎥 Policy Execution")
            
            # Check if we have existing animation
            if (st.session_state.training_state['policy_animation_frames'] and 
                st.session_state.training_state['policy_execution_details']):
                
                frames = st.session_state.training_state['policy_animation_frames']
                exec_details = st.session_state.training_state['policy_execution_details']
                
                if frames and exec_details:
                    # Display animation controls
                    col_anim1, col_anim2, col_anim3 = st.columns([2, 1, 1])
                    
                    with col_anim1:
                        play_speed = st.slider(
                            "Animation Speed", 
                            0.1, 2.0, 
                            st.session_state.training_state.get('animation_speed', 0.5), 
                            0.1,
                            key='anim_speed_tab4'
                        )
                        st.session_state.training_state['animation_speed'] = play_speed
                    
                    with col_anim2:
                        start_frame = st.number_input("Start Frame", 0, len(frames)-1, 0)
                    
                    with col_anim3:
                        frames_to_show = st.number_input("Frames to Show", 1, len(frames), min(10, len(frames)))
                    
                    # Play animation
                    if st.button("▶️ Play Animation", type="primary"):
                        animation_placeholder = st.empty()
                        progress_bar = st.progress(0)
                        
                        for i in range(start_frame, min(start_frame + frames_to_show, len(frames))):
                            # Update progress
                            progress = (i - start_frame + 1) / frames_to_show
                            progress_bar.progress(progress)
                            
                            # Display frame
                            frame = frames[i]
                            caption = f"Frame {i+1}/{len(frames)}"
                            
                            if i < len(exec_details) and 'step_number' in exec_details[i]:
                                step_info = exec_details[i]
                                caption += f" | Step {step_info['step_number']}"
                                if 'reward' in step_info:
                                    caption += f" | Reward: {step_info['reward']:.2f}"
                            
                            animation_placeholder.image(frame, caption=caption, width='stretch')
                            
                            # Delay for animation
                            time.sleep(play_speed)
                        
                        progress_bar.empty()
                        
                        # Show final result
                        if exec_details and len(exec_details) > 0:
                            final_detail = exec_details[-1]
                            if isinstance(final_detail, dict) and 'total_reward' in final_detail:
                                st.success(f"✅ Execution complete! Total reward: {final_detail['total_reward']:.2f}")
                    
                    # Show step-by-step details
                    st.markdown("### 📋 Execution Details")
                    
                    if exec_details and len(exec_details) > 1:
                        # Limit display for performance
                        max_details = min(10, len(exec_details) - 1)
                        
                        for i in range(max_details):
                            detail = exec_details[i]
                            if isinstance(detail, dict):
                                with st.expander(f"Step {detail.get('step_number', i+1)}", expanded=(i < 2)):
                                    display_algorithm_step(detail, algorithm_options[selected_algorithm_key])
                        
                        if len(exec_details) - 1 > max_details:
                            st.info(f"Showing first {max_details} of {len(exec_details)-1} steps")
                
                else:
                    st.warning("No animation data available")
            
            else:
                st.info("No animation captured yet.")
            
            # Button to capture new animation
            st.markdown("---")
            st.markdown("### 🆕 Capture New Animation")
            
            col_cap1, col_cap2 = st.columns(2)
            with col_cap1:
                max_steps = st.number_input("Max Steps", 10, 200, 50)
            
            with col_cap2:
                capture_new = st.checkbox("Capture Frames", value=True)
            
            if st.button("🎬 Execute Policy & Capture Animation", type="secondary"):
                with st.spinner("Executing policy and capturing animation..."):
                    # Clear previous animation to save memory
                    st.session_state.training_state['policy_animation_frames'] = []
                    st.session_state.training_state['policy_execution_details'] = []
                    
                    # Execute policy
                    frames, exec_details, total_reward = execute_policy_with_animation(
                        algorithm,
                        selected_env_key,
                        env_info.get('kwargs', {}),
                        max_steps,
                        capture_new
                    )
                    
                    # Store results
                    st.session_state.training_state['policy_animation_frames'] = frames
                    st.session_state.training_state['policy_execution_details'] = exec_details
                    
                    st.success(f"✅ Animation captured! Total reward: {total_reward:.2f}")
                    st.rerun()
    
    else:
        st.info("ℹ️ No algorithm trained yet. Please train an algorithm first!")

# Training logic
if st.session_state.training_state['is_training']:
    try:
        # Check memory before starting
        memory_ok, memory_percent = check_memory_usage()
        if not memory_ok:
            st.error(f"⚠️ Cannot start training: Memory usage too high ({memory_percent:.1f}%)")
            st.session_state.training_state['is_training'] = False
            st.button("🔄 Clean Memory and Retry", on_click=cleanup_memory)
            st.stop()
        
        # Create environment
        env = gym.make(selected_env_key, render_mode=None, **env_info.get('kwargs', {}))
        
        # Prepare parameters
        params = {
            'learning_rate': learning_rate,
            'discount_factor': discount_factor,
            'epsilon': epsilon,
            'epsilon_decay': epsilon_decay,
            'min_epsilon': 0.01,
            'divisions': divisions,
            'iterations': iterations,
            'episodes_per_iteration': episodes_per_iteration,
            'max_steps_per_episode': 500,
            'theta': 1e-6,
        }
        
        # Create algorithm
        if st.session_state.training_state['algorithm'] is None:
            algorithm = create_algorithm(env, selected_algorithm_key, params)
            st.session_state.training_state['algorithm'] = algorithm
            st.session_state.training_state['env'] = env
        else:
            algorithm = st.session_state.training_state['algorithm']
        
        # Training loop
        current_iter = st.session_state.training_state['current_iteration']
        
        for iteration in range(current_iter, iterations):
            if not st.session_state.training_state['is_training']:
                break
            
            # Check memory during training
            if iteration % 2 == 0:  # Check every 2 iterations
                memory_ok, memory_percent = check_memory_usage()
                if not memory_ok:
                    st.warning(f"High memory usage ({memory_percent:.1f}%), cleaning up...")
                    cleanup_memory()
            
            # Train one iteration
            iteration_detail = algorithm.train_iteration(iteration)
            
            # Update session state
            st.session_state.training_state['iteration_details'].append(iteration_detail)
            st.session_state.training_state['current_iteration'] = iteration + 1
            
            # Update UI after each iteration
            st.rerun()
        
        # Training completed
        st.session_state.training_state['is_training'] = False
        
        # Store final results
        if hasattr(algorithm, 'get_policy'):
            st.session_state.training_state['optimal_policy'] = algorithm.get_policy()
        if hasattr(algorithm, 'get_value_function'):
            st.session_state.training_state['value_function'] = algorithm.get_value_function()
        if hasattr(algorithm, 'get_q_function'):
            st.session_state.training_state['q_function'] = algorithm.get_q_function()
        
        # Auto-capture animation after training
        with st.spinner("Training complete! Capturing policy animation..."):
            frames, exec_details, total_reward = execute_policy_with_animation(
                algorithm,
                selected_env_key,
                env_info.get('kwargs', {}),
                max_frames,
                capture_frames
            )
            
            st.session_state.training_state['policy_animation_frames'] = frames
            st.session_state.training_state['policy_execution_details'] = exec_details
        
        st.success(f"✅ Training completed successfully! Final reward: {total_reward:.2f}")
        
    except Exception as e:
        st.error(f"Error during training: {str(e)}")
        import traceback
        with st.expander("Error Details"):
            st.code(traceback.format_exc())
        st.session_state.training_state['is_training'] = False

# Cleanup
if 'env' in locals() and env is not None:
    try:
        env.close()
    except:
        pass

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.9em; padding: 1rem;">
    <p>🧠 Enhanced RL Algorithm Suite | Complete algorithm details with memory optimization</p>
    <p>Shows probabilities, Q-values, V-values, and step-by-step computations</p>
    <p>✅ Fixed policy tables for all algorithms | ✅ Memory explosion prevention</p>
</div>
""", unsafe_allow_html=True)