import numpy as np
from typing import Dict, Any, List, Tuple
from collections import defaultdict
from .base_algorithm import BaseAlgorithm

class QLearning(BaseAlgorithm):
    """Q-Learning algorithm"""
    
    def __init__(self, env, params: Dict[str, Any]):
        super().__init__(env, params)
        
        # Initialize Q-table
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
        
        # Initialize Q-table with small random values to break symmetry
        self.Q = np.random.uniform(-0.01, 0.01, (self.n_states, self.n_actions))
        
        # Hyperparameters
        self.learning_rate = params.get('learning_rate', 0.1)
        self.discount_factor = params.get('discount_factor', 0.9)
        self.epsilon = params.get('epsilon', 1.0)
        self.min_epsilon = params.get('min_epsilon', 0.01)
        self.epsilon_decay = params.get('epsilon_decay', 0.995)
        
        # Tracking
        self.iteration_details = []
        self.step_details_history = []
        self.visit_counts = np.zeros((self.n_states, self.n_actions))
        
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
        """Epsilon-greedy action selection"""
        if epsilon_greedy and np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        
        state_idx = self._state_to_index(state)
        state_idx = max(0, min(state_idx, self.n_states - 1))
        
        # Greedy action
        return np.argmax(self.Q[state_idx])
    
    def get_step_details(self, state, action, reward, next_state, done):
        """Get Q-learning specific step details with complete information"""
        state_idx = self._state_to_index(state)
        next_state_idx = self._state_to_index(next_state) if not done else 0
        
        old_q = self.Q[state_idx, action]
        max_next_q = np.max(self.Q[next_state_idx]) if not done else 0
        td_target = reward + self.discount_factor * max_next_q
        td_error = td_target - old_q
        
        # Get all Q-values for current state
        all_q_values = [float(q) for q in self.Q[state_idx]]
        
        # Get visit counts
        visit_count = self.visit_counts[state_idx, action]
        
        # Get probability of each action (for display)
        action_probs = self._get_action_probabilities(state_idx)
        
        return {
            'state': state,
            'action': action,
            'reward': float(reward),
            'next_state': next_state,
            'done': done,
            'state_idx': state_idx,
            'next_state_idx': next_state_idx,
            'old_Q(s,a)': float(old_q),
            'max_Q(s\',a\')': float(max_next_q),
            'all_Q_values': all_q_values,
            'TD_target': float(td_target),
            'TD_error': float(td_error),
            'learning_rate': float(self.learning_rate),
            'discount_factor': float(self.discount_factor),
            'epsilon': float(self.epsilon),
            'action_probabilities': action_probs,
            'visit_count': int(visit_count),
            'V(s)': float(np.max(self.Q[state_idx])),
            'V(s\')': float(np.max(self.Q[next_state_idx])) if not done else 0.0,
            'step_type': 'Q-Learning'
        }
    
    def _get_action_probabilities(self, state_idx):
        """Get epsilon-greedy action probabilities"""
        probs = np.ones(self.n_actions) * self.epsilon / self.n_actions
        best_action = np.argmax(self.Q[state_idx])
        probs[best_action] += (1.0 - self.epsilon)
        return [float(p) for p in probs]
    
    def train_iteration(self, iteration: int) -> Dict[str, Any]:
        """Train for one iteration"""
        iteration_rewards = []
        iteration_steps = []
        iteration_step_details = []
        
        for episode in range(self.episodes_per_iteration):
            state, _ = self.env.reset()
            done = False
            episode_reward = 0
            episode_steps = 0
            episode_step_details = []
            
            while not done and episode_steps < self.max_steps_per_episode:
                action = self.get_action(state)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                
                # Update visit count
                state_idx = self._state_to_index(state)
                self.visit_counts[state_idx, action] += 1
                
                # Get step details BEFORE update
                step_detail = self.get_step_details(state, action, reward, next_state, done)
                
                # Perform Q-learning update
                next_state_idx = self._state_to_index(next_state) if not done else 0
                
                old_q = self.Q[state_idx, action]
                max_next_q = np.max(self.Q[next_state_idx]) if not done else 0
                td_target = reward + self.discount_factor * max_next_q
                
                self.Q[state_idx, action] = old_q + self.learning_rate * (td_target - old_q)
                
                # Add post-update information
                step_detail['new_Q(s,a)'] = float(self.Q[state_idx, action])
                step_detail['update_applied'] = True
                step_detail['learning_rate_applied'] = float(self.learning_rate)
                
                # Store step details
                episode_step_details.append(step_detail)
                episode_reward += reward
                episode_steps += 1
                state = next_state
            
            iteration_rewards.append(episode_reward)
            iteration_steps.append(episode_steps)
            iteration_step_details.append(episode_step_details)
        
        # Decay epsilon
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
        
        # Calculate statistics
        q_non_zero = np.count_nonzero(np.abs(self.Q) > 1e-10)
        q_positive = np.sum(self.Q > 0)
        q_negative = np.sum(self.Q < 0)
        
        # Store iteration details
        iteration_detail = {
            'iteration': iteration,
            'mean_reward': float(np.mean(iteration_rewards)),
            'std_reward': float(np.std(iteration_rewards)),
            'mean_steps': float(np.mean(iteration_steps)),
            'epsilon': float(self.epsilon),
            'step_details': iteration_step_details,
            'Q_table_mean': float(np.mean(self.Q)),
            'Q_table_max': float(np.max(self.Q)),
            'Q_table_min': float(np.min(self.Q)),
            'Q_table_std': float(np.std(self.Q)),
            'V_mean': float(np.mean(np.max(self.Q, axis=1))),
            'q_non_zero_count': int(q_non_zero),
            'q_positive_count': int(q_positive),
            'q_negative_count': int(q_negative),
            'total_visits': int(np.sum(self.visit_counts)),
            'learning_rate': float(self.learning_rate),
            'discount_factor': float(self.discount_factor)
        }
        
        self.iteration_details.append(iteration_detail)
        
        # Memory cleanup
        if iteration % 5 == 0 and len(self.step_details_history) > 1000:
            self.step_details_history = self.step_details_history[-500:]
        
        return iteration_detail
    
    def get_policy(self):
        """Extract policy from Q-table"""
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
                    'State_Type': self.state_type,
                    'Optimal Action': int(policy[state]) if isinstance(policy, np.ndarray) else 'Function',
                    'V*(s)': float(V[state]) if V is not None and len(V) > state else 0.0,
                    'All Q-values': [float(q) for q in self.Q[state]],
                    'Visit Counts': [int(count) for count in self.visit_counts[state]],
                    'Epsilon': float(self.epsilon)
                }
                
                # Add statistics
                row['max_Q'] = float(np.max(self.Q[state]))
                row['min_Q'] = float(np.min(self.Q[state]))
                row['mean_Q'] = float(np.mean(self.Q[state]))
                
                # Add action probabilities
                row['Action_Probabilities'] = self._get_action_probabilities(state)
                
                table.append(row)
            
            return table
            
        except Exception as e:
            return [{
                'State': 'Error',
                'Message': f'Failed to generate policy table: {str(e)}'
            }]