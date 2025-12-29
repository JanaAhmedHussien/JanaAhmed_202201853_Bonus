import numpy as np
from typing import Dict, Any, List, Tuple
from collections import defaultdict
import sys
from .base_algorithm import BaseAlgorithm

class MonteCarlo(BaseAlgorithm):
    """Monte Carlo Control with Exploring Starts"""
    
    def __init__(self, env, params: Dict[str, Any]):
        super().__init__(env, params)
        
        # Initialize
        if hasattr(env.observation_space, 'n'):
            self.n_states = env.observation_space.n
            self.state_type = 'discrete'
        else:
            self.state_type = 'continuous'
            divisions = params.get('divisions', 10)
            self.bins = []
            for low, high in zip(env.observation_space.low, env.observation_space.high):
                self.bins.append(np.linspace(low, high, divisions))
            
            total_states = 1
            for b in self.bins:
                total_states *= (len(b) + 1)
            self.n_states = int(total_states)
        
        self.n_actions = env.action_space.n
        
        # Q-values and returns
        self.Q = np.zeros((self.n_states, self.n_actions))
        self.returns = defaultdict(list)
        self.policy = np.ones((self.n_states, self.n_actions)) / self.n_actions
        
        # Hyperparameters
        self.discount_factor = params.get('discount_factor', 0.9)
        
        # Tracking
        self.iteration_details = []
        
    def _state_to_index(self, state):
        """Convert state to integer index"""
        if self.state_type == 'discrete':
            return int(state)
        
        indices = []
        state = np.array(state).flatten()
        for i, (s, bin_edges) in enumerate(zip(state, self.bins)):
            idx = np.digitize(float(s), bin_edges)
            indices.append(min(idx, len(bin_edges)))
        
        state_idx = 0
        multiplier = 1
        for idx in reversed(indices):
            state_idx += int(idx) * multiplier
            multiplier *= (len(self.bins[0]) + 1)
        
        return max(0, min(state_idx, self.n_states - 1))
    
    def get_action(self, state, epsilon_greedy=True):
        """Get action from current policy"""
        state_idx = self._state_to_index(state)
        return np.random.choice(self.n_actions, p=self.policy[state_idx])
    
    def train_iteration(self, iteration: int) -> Dict[str, Any]:
        """Train for one iteration using Monte Carlo"""
        iteration_rewards = []
        iteration_steps = []
        iteration_step_details = []
        
        for episode in range(self.episodes_per_iteration):
            # Generate episode
            episode_trajectory = []
            state, _ = self.env.reset()
            done = False
            episode_reward = 0
            episode_steps = 0
            
            # Exploring start: random first action
            action = self.env.action_space.sample()
            
            while not done and episode_steps < self.max_steps_per_episode:
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                
                state_idx = self._state_to_index(state)
                episode_trajectory.append((state_idx, action, reward))
                
                episode_reward += reward
                episode_steps += 1
                
                if not done:
                    # Follow policy for subsequent actions
                    action = self.get_action(next_state)
                    state = next_state
                else:
                    state = next_state
            
            # Monte Carlo update after episode
            G = 0
            first_visit_details = []
            
            for t in reversed(range(len(episode_trajectory))):
                state_idx, action, reward = episode_trajectory[t]
                G = self.discount_factor * G + reward
                
                # First-visit Monte Carlo
                if (state_idx, action) not in [(s, a) for s, a, _ in episode_trajectory[:t]]:
                    self.returns[(state_idx, action)].append(G)
                    old_q = self.Q[state_idx, action]
                    self.Q[state_idx, action] = np.mean(self.returns[(state_idx, action)])
                    
                    # Update policy to be greedy w.r.t Q
                    best_action = np.argmax(self.Q[state_idx])
                    old_policy = self.policy[state_idx].copy()
                    self.policy[state_idx] = np.eye(self.n_actions)[best_action]
                    
                    first_visit_details.append({
                        'step_in_episode': t,
                        'state_idx': state_idx,
                        'state': self._get_state_representation(state_idx),
                        'action': action,
                        'reward': float(reward),
                        'return_G': float(G),
                        'old_Q(s,a)': float(old_q),
                        'new_Q(s,a)': float(self.Q[state_idx, action]),
                        'all_Q_values': [float(q) for q in self.Q[state_idx]],
                        'V(s)': float(np.max(self.Q[state_idx])),
                        'old_policy': [float(p) for p in old_policy],
                        'new_policy': [float(p) for p in self.policy[state_idx]],
                        'policy_change': not np.array_equal(old_policy, self.policy[state_idx]),
                        'samples_count': len(self.returns[(state_idx, action)]),
                        'mean_return': float(np.mean(self.returns[(state_idx, action)]))
                    })
            
            iteration_rewards.append(episode_reward)
            iteration_steps.append(episode_steps)
            iteration_step_details.append(first_visit_details)
        
        # Store iteration details
        iteration_detail = {
            'iteration': iteration,
            'mean_reward': float(np.mean(iteration_rewards)),
            'std_reward': float(np.std(iteration_rewards)),
            'mean_steps': float(np.mean(iteration_steps)),
            'step_details': iteration_step_details,
            'Q_table_mean': float(np.mean(self.Q)),
            'Q_table_max': float(np.max(self.Q)),
            'returns_count': sum(len(v) for v in self.returns.values()),
            'unique_state_action_pairs': len(self.returns)
        }
        
        self.iteration_details.append(iteration_detail)
        
        # Cleanup to prevent memory explosion
        if iteration % 5 == 0:
            self.cleanup_memory()
        
        return iteration_detail
    
    def _get_state_representation(self, state_idx):
        """Get human-readable state representation"""
        if self.state_type == 'discrete':
            return f"State {state_idx}"
        else:
            return f"Continuous State {state_idx}"
    
    def get_policy(self):
        """Get deterministic policy from Q-values"""
        if self.state_type == 'discrete':
            policy = np.zeros(self.n_states, dtype=int)
            for s in range(self.n_states):
                policy[s] = np.argmax(self.Q[s])
            return policy
        else:
            def policy(state):
                state_idx = self._state_to_index(state)
                return np.argmax(self.Q[state_idx])
            return policy
    
    def get_value_function(self):
        """Get V(s) = max_a Q(s,a)"""
        return np.max(self.Q, axis=1)
    
    def get_q_function(self):
        """Get the Q-function"""
        return self.Q
    
    def get_optimal_policy_table(self):
        """Get optimal policy in table format with complete details"""
        try:
            policy = self.get_policy()
            V = self.get_value_function()
            
            table = []
            for state in range(min(20, self.n_states)):
                row = {
                    'State': state,
                    'State_Rep': self._get_state_representation(state),
                    'Optimal Action': int(policy[state]) if isinstance(policy, np.ndarray) else 'Function',
                    'V*(s)': float(V[state]) if V is not None and len(V) > state else 0.0,
                    'All Q-values': [float(q) for q in self.Q[state]],
                    'Policy Probabilities': [float(p) for p in self.policy[state]]
                }
                
                # Add return statistics
                for a in range(self.n_actions):
                    returns = self.returns.get((state, a), [])
                    if returns:
                        row[f'Returns_Action{a}_Count'] = len(returns)
                        row[f'Returns_Action{a}_Mean'] = float(np.mean(returns))
                
                table.append(row)
            
            return table
            
        except Exception as e:
            return [{
                'State': 'Error',
                'Message': f'Failed to generate policy table: {str(e)}'
            }]