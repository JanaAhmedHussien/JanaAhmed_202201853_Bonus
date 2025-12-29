import numpy as np
from typing import Dict, Any
from .policy_iteration import PolicyIteration

class ValueIteration(PolicyIteration):
    """Value Iteration algorithm"""
    
    def __init__(self, env, params: Dict[str, Any]):
        super().__init__(env, params)
        self.name = "Value Iteration"
    
    def train_iteration(self, iteration: int) -> Dict[str, Any]:
        """One iteration of value iteration"""
        delta = 0
        iteration_details = []
        
        for s in range(self.n_states):
            v = self.V[s]
            
            # Calculate Q-values for all actions
            q_values = np.zeros(self.n_actions)
            for a in range(self.n_actions):
                for next_s in range(self.n_states):
                    q_values[a] += self.P[s, a, next_s] * (
                        self.R[s, a, next_s] + self.discount_factor * self.V[next_s]
                    )
            
            # Bellman optimality update
            new_v = np.max(q_values)
            self.V[s] = new_v
            delta = max(delta, abs(v - new_v))
            
            iteration_details.append({
                'state': s,
                'old_V(s)': float(v),
                'new_V(s)': float(new_v),
                'delta': float(abs(v - new_v)),
                'max_Q': float(np.max(q_values)),
                'argmax_Q': int(np.argmax(q_values)),
                'all_Q_values': [float(q) for q in q_values],
                'V(s)': float(new_v)
            })
        
        # Extract policy from value function
        self._extract_policy()
        
        # Evaluate current policy
        eval_results = self.evaluate_policy(num_episodes=5)
        
        # Calculate Q-function
        Q = self.get_q_function()
        
        # Store iteration details
        iteration_detail = {
            'iteration': iteration,
            'mean_reward': eval_results['mean_reward'],
            'mean_steps': eval_results['mean_steps'],
            'max_delta': float(delta),
            'state_details': iteration_details[:10],  # Limit display
            'V_mean': float(np.mean(self.V)),
            'V_max': float(np.max(self.V)),
            'V_std': float(np.std(self.V)),
            'Q_mean': float(np.mean(Q)) if Q is not None else 0.0,
            'discount_factor': float(self.discount_factor),
            'theta': float(self.theta)
        }
        
        self.iteration_details.append(iteration_detail)
        
        return iteration_detail
    
    def _extract_policy(self):
        """Extract policy from value function"""
        for s in range(self.n_states):
            q_values = np.zeros(self.n_actions)
            for a in range(self.n_actions):
                for next_s in range(self.n_states):
                    q_values[a] += self.P[s, a, next_s] * (
                        self.R[s, a, next_s] + self.discount_factor * self.V[next_s]
                    )
            
            best_action = np.argmax(q_values)
            self.policy[s] = np.eye(self.n_actions)[best_action]
    
    def get_step_details(self, state, action, reward, next_state, done):
        """Get value iteration step details"""
        detail = super().get_step_details(state, action, reward, next_state, done)
        detail['step_type'] = 'Value Iteration'
        detail['algorithm'] = 'Value Iteration'
        return detail
    
    def get_optimal_policy_table(self):
        """Get optimal policy table for value iteration"""
        table = super().get_optimal_policy_table()
        for row in table:
            row['Algorithm'] = 'Value Iteration'
        return table