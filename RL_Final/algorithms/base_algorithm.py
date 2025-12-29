from abc import ABC, abstractmethod
from typing import Dict, Any, List, Tuple, Optional
import numpy as np

class BaseAlgorithm(ABC):
    """Base class for all RL algorithms"""
    
    def __init__(self, env, params: Dict[str, Any]):
        self.env = env
        self.params = params
        self.name = params.get('name', 'Unknown Algorithm')
        self.iterations = params.get('iterations', 10)
        self.episodes_per_iteration = params.get('episodes_per_iteration', 100)
        self.max_steps_per_episode = params.get('max_steps_per_episode', 500)
        
        # Performance tracking
        self.step_details_history = []
        self.memory_usage = []
        
    @abstractmethod
    def train_iteration(self, iteration: int) -> Dict[str, Any]:
        """Train for one iteration, returns iteration details"""
        pass
    
    @abstractmethod
    def get_policy(self):
        """Get the current policy"""
        pass
    
    @abstractmethod
    def get_value_function(self):
        """Get the value function"""
        pass
    
    @abstractmethod
    def get_q_function(self):
        """Get the Q-function (for algorithms that have it)"""
        pass
    
    def evaluate_policy(self, num_episodes: int = 10) -> Dict[str, Any]:
        """Evaluate the current policy"""
        rewards = []
        steps = []
        
        for _ in range(num_episodes):
            state, _ = self.env.reset()
            episode_reward = 0
            episode_steps = 0
            done = False
            
            while not done and episode_steps < self.max_steps_per_episode:
                action = self.get_action(state, epsilon_greedy=False)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                
                state = next_state
                episode_reward += reward
                episode_steps += 1
            
            rewards.append(episode_reward)
            steps.append(episode_steps)
        
        return {
            'mean_reward': float(np.mean(rewards)),
            'mean_steps': float(np.mean(steps)),
            'all_rewards': rewards
        }
    
    def get_action(self, state, epsilon_greedy=True):
        """Get action from current policy - must be implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement get_action method")
    
    def get_step_details(self, state, action, reward, next_state, done) -> Dict[str, Any]:
        """Get algorithm-specific step details for display"""
        base_details = {
            'state': state,
            'action': action,
            'reward': reward,
            'next_state': next_state,
            'done': done,
            'step_type': 'base'
        }
        
        # Add algorithm-specific details
        try:
            if hasattr(self, 'V'):
                state_idx = self._state_to_index(state) if hasattr(self, '_state_to_index') else None
                if state_idx is not None:
                    base_details['V(s)'] = float(self.V[state_idx])
            
            if hasattr(self, 'Q'):
                state_idx = self._state_to_index(state) if hasattr(self, '_state_to_index') else None
                if state_idx is not None:
                    base_details['Q_values'] = [float(q) for q in self.Q[state_idx]]
                    base_details['max_Q'] = float(np.max(self.Q[state_idx]))
                    base_details['selected_Q'] = float(self.Q[state_idx, action])
        except:
            pass
        
        return base_details
    
    def get_optimal_policy_table(self) -> List[Dict[str, Any]]:
        """Get optimal policy in table format - default implementation"""
        try:
            policy = self.get_policy()
            V = self.get_value_function()
            Q = self.get_q_function()
            
            table = []
            if isinstance(policy, np.ndarray) and policy.ndim == 1:
                # Discrete deterministic policy
                for state in range(min(20, len(policy))):
                    row = {
                        'State': state,
                        'Optimal Action': int(policy[state])
                    }
                    
                    if V is not None and len(V) > state:
                        row['V*(s)'] = float(V[state])
                    
                    if Q is not None and Q.shape[0] > state:
                        for a in range(min(4, Q.shape[1])):
                            row[f'Q*(s,{a})'] = float(Q[state, a])
                    
                    table.append(row)
            
            elif hasattr(self, 'policy') and isinstance(self.policy, np.ndarray):
                # Stochastic policy
                for state in range(min(20, self.policy.shape[0])):
                    row = {
                        'State': state,
                        'Policy Probabilities': [float(p) for p in self.policy[state]]
                    }
                    
                    if V is not None and len(V) > state:
                        row['V(s)'] = float(V[state])
                    
                    table.append(row)
            
            return table
            
        except Exception as e:
            # Return empty table with error info
            return [{
                'State': 'N/A',
                'Message': f'Policy table generation failed: {str(e)}',
                'Algorithm': self.name
            }]
    
    def cleanup_memory(self):
        """Clean up memory-intensive attributes"""
        self.step_details_history = []
        if hasattr(self, 'returns'):
            if isinstance(self.returns, dict):
                self.returns.clear()
    
    def _state_to_index(self, state):
        """Default state discretization for continuous spaces"""
        raise NotImplementedError("Subclasses should implement state discretization")