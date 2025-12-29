import numpy as np
from typing import Dict, Any
from .base_algorithm import BaseAlgorithm

class PolicyIteration(BaseAlgorithm):
    """Policy Iteration algorithm (requires model of environment)"""
    
    def __init__(self, env, params: Dict[str, Any]):
        super().__init__(env, params)
        
        # Only works for discrete environments
        self.n_states = env.observation_space.n
        self.n_actions = env.action_space.n
        
        # Initialize policy and value function
        self.policy = np.ones((self.n_states, self.n_actions)) / self.n_actions
        self.V = np.zeros(self.n_states)
        
        # Get transition model
        self.P, self.R = self._get_model()
        
        # Hyperparameters
        self.discount_factor = params.get('discount_factor', 0.9)
        self.theta = params.get('theta', 1e-6)
        
        # Tracking
        self.iteration_details = []
        self.policy_stable = False
        
    def _get_model(self):
        """Extract transition and reward model from environment"""
        P = np.zeros((self.n_states, self.n_actions, self.n_states))
        R = np.zeros((self.n_states, self.n_actions, self.n_states))
        
        try:
            for s in range(self.n_states):
                for a in range(self.n_actions):
                    transitions = self.env.unwrapped.P.get(s, {}).get(a, [])
                    for prob, next_s, reward, _ in transitions:
                        P[s, a, next_s] += prob
                        R[s, a, next_s] = reward
        except:
            # Create random model if not available
            P = np.random.dirichlet(np.ones(self.n_states), size=(self.n_states, self.n_actions))
            R = np.random.randn(self.n_states, self.n_actions, self.n_states)
        
        return P, R
    
    def get_action(self, state, epsilon_greedy=True):
        """Get action from current policy"""
        state_idx = int(state) if isinstance(state, (int, np.integer)) else 0
        return np.random.choice(self.n_actions, p=self.policy[state_idx])
    
    def policy_evaluation(self):
        """Evaluate current policy until convergence"""
        iteration = 0
        evaluation_details = []
        
        while True:
            delta = 0
            iteration_details = []
            
            for s in range(self.n_states):
                v = self.V[s]
                
                new_v = 0
                q_values = np.zeros(self.n_actions)
                
                for a in range(self.n_actions):
                    action_prob = self.policy[s, a]
                    q_sa = 0
                    
                    for next_s in range(self.n_states):
                        q_sa += self.P[s, a, next_s] * (
                            self.R[s, a, next_s] + self.discount_factor * self.V[next_s]
                        )
                    
                    q_values[a] = q_sa
                    new_v += action_prob * q_sa
                
                self.V[s] = new_v
                delta = max(delta, abs(v - new_v))
                
                iteration_details.append({
                    'state': s,
                    'old_V(s)': float(v),
                    'new_V(s)': float(new_v),
                    'delta': float(abs(v - new_v)),
                    'Q_values': [float(q) for q in q_values],
                    'policy_probs': [float(p) for p in self.policy[s]],
                    'state_type': 'discrete'
                })
            
            evaluation_details.append({
                'evaluation_iteration': iteration,
                'max_delta': float(delta),
                'state_details': iteration_details[:5]  # Limit display
            })
            
            iteration += 1
            if delta < self.theta:
                break
        
        return evaluation_details
    
    def policy_improvement(self):
        """Improve policy based on current value function"""
        policy_stable = True
        improvement_details = []
        
        for s in range(self.n_states):
            old_action = np.argmax(self.policy[s])
            
            # Calculate Q-values for all actions
            q_values = np.zeros(self.n_actions)
            for a in range(self.n_actions):
                for next_s in range(self.n_states):
                    q_values[a] += self.P[s, a, next_s] * (
                        self.R[s, a, next_s] + self.discount_factor * self.V[next_s]
                    )
            
            # Greedy improvement
            best_action = np.argmax(q_values)
            
            # Update policy
            old_policy = self.policy[s].copy()
            new_policy = np.eye(self.n_actions)[best_action]
            
            improvement_details.append({
                'state': s,
                'old_action': int(old_action),
                'new_action': int(best_action),
                'q_values': [float(q) for q in q_values],
                'V(s)': float(self.V[s]),
                'old_policy': [float(p) for p in old_policy],
                'new_policy': [float(p) for p in new_policy],
                'changed': old_action != best_action,
                'policy_change_magnitude': float(np.sum(np.abs(old_policy - new_policy)))
            })
            
            if old_action != best_action:
                policy_stable = False
            
            self.policy[s] = new_policy
        
        self.policy_stable = policy_stable
        return improvement_details, policy_stable
    
    def train_iteration(self, iteration: int) -> Dict[str, Any]:
        """One iteration of policy iteration = evaluation + improvement"""
        # Policy evaluation
        eval_details = self.policy_evaluation()
        
        # Policy improvement
        improvement_details, policy_stable = self.policy_improvement()
        
        # Evaluate current policy
        eval_results = self.evaluate_policy(num_episodes=5)
        
        # Calculate Q-function
        Q = self.get_q_function()
        
        # Store iteration details
        iteration_detail = {
            'iteration': iteration,
            'mean_reward': eval_results['mean_reward'],
            'mean_steps': eval_results['mean_steps'],
            'policy_stable': policy_stable,
            'evaluation_details': eval_details,
            'improvement_details': improvement_details[:10],  # Limit display
            'V_mean': float(np.mean(self.V)),
            'V_max': float(np.max(self.V)),
            'V_std': float(np.std(self.V)),
            'Q_mean': float(np.mean(Q)) if Q is not None else 0.0,
            'discount_factor': float(self.discount_factor),
            'theta': float(self.theta)
        }
        
        self.iteration_details.append(iteration_detail)
        
        return iteration_detail
    
    def get_policy(self):
        """Get deterministic policy"""
        return np.argmax(self.policy, axis=1)
    
    def get_value_function(self):
        """Get value function"""
        return self.V
    
    def get_q_function(self):
        """Get Q-function from policy and value function"""
        Q = np.zeros((self.n_states, self.n_actions))
        for s in range(self.n_states):
            for a in range(self.n_actions):
                for next_s in range(self.n_states):
                    Q[s, a] += self.P[s, a, next_s] * (
                        self.R[s, a, next_s] + self.discount_factor * self.V[next_s]
                    )
        return Q
    
    def get_step_details(self, state, action, reward, next_state, done):
        """Get policy iteration step details"""
        if isinstance(state, (int, np.integer)):
            state_idx = int(state)
            next_state_idx = int(next_state) if not done else 0
            
            detail = {
                'state': state,
                'state_idx': state_idx,
                'action': action,
                'reward': float(reward),
                'next_state': next_state,
                'next_state_idx': next_state_idx,
                'done': done,
                'V(s)': float(self.V[state_idx]),
                'V(s\')': float(self.V[next_state_idx]) if not done else 0.0,
                'policy_probs': [float(p) for p in self.policy[state_idx]],
                'optimal_action': int(np.argmax(self.policy[state_idx])),
                'step_type': 'Policy Iteration'
            }
            
            # Add Q-values if available
            Q = self.get_q_function()
            if Q is not None:
                detail['Q_values'] = [float(q) for q in Q[state_idx]]
                detail['selected_Q'] = float(Q[state_idx, action])
            
            return detail
        else:
            return super().get_step_details(state, action, reward, next_state, done)
    
    def get_optimal_policy_table(self):
        """Get optimal policy in table format with complete details"""
        policy = self.get_policy()
        V = self.get_value_function()
        Q = self.get_q_function()
        
        table = []
        for state in range(min(20, self.n_states)):
            row = {
                'State': state,
                'Optimal Action': int(policy[state]),
                'V*(s)': float(V[state]),
                'π(s)': [float(p) for p in self.policy[state]],
                'State_Type': 'discrete',
                'Policy_Stable': self.policy_stable
            }
            
            if Q is not None:
                for a in range(min(4, self.n_actions)):
                    row[f'Q*(s,{a})'] = float(Q[state, a])
                
                # Add Q-value statistics
                row['max_Q'] = float(np.max(Q[state]))
                row['min_Q'] = float(np.min(Q[state]))
            
            # Add transition model info
            if hasattr(self, 'P'):
                for a in range(min(2, self.n_actions)):
                    next_states = np.where(self.P[state, a] > 0)[0]
                    if len(next_states) > 0:
                        row[f'Transitions_Action{a}'] = f"{len(next_states)} states"
            
            table.append(row)
        
        return table