import numpy as np
from typing import Dict, Any
from .qlearning import QLearning

class SARSA(QLearning):
    """SARSA algorithm - inherits from QLearning but uses SARSA update"""
    
    def __init__(self, env, params: Dict[str, Any]):
        super().__init__(env, params)
        self.name = "SARSA"
    
    def get_step_details(self, state, action, reward, next_state, done, next_action=None):
        """Get SARSA specific step details with complete information"""
        state_idx = self._state_to_index(state)
        next_state_idx = self._state_to_index(next_state) if not done else 0
        
        old_q = self.Q[state_idx, action]
        
        if next_action is None and not done:
            next_action = self.get_action(next_state)
        
        next_q = self.Q[next_state_idx, next_action] if not done else 0
        td_target = reward + self.discount_factor * next_q
        td_error = td_target - old_q
        
        # Get all Q-values for current state
        all_q_values = [float(q) for q in self.Q[state_idx]]
        
        # Get visit counts
        visit_count = self.visit_counts[state_idx, action]
        
        # Get probability of each action
        action_probs = self._get_action_probabilities(state_idx)
        
        detail = {
            'state': state,
            'action': action,
            'reward': float(reward),
            'next_state': next_state,
            'done': done,
            'state_idx': state_idx,
            'next_state_idx': next_state_idx,
            'old_Q(s,a)': float(old_q),
            'Q(s\',a\')': float(next_q),
            'all_Q_values': all_q_values,
            'TD_target': float(td_target),
            'TD_error': float(td_error),
            'learning_rate': float(self.learning_rate),
            'discount_factor': float(self.discount_factor),
            'epsilon': float(self.epsilon),
            'action_probabilities': action_probs,
            'visit_count': int(visit_count),
            'next_action': int(next_action) if next_action is not None else None,
            'V(s)': float(np.max(self.Q[state_idx])),
            'V(s\')': float(np.max(self.Q[next_state_idx])) if not done else 0.0,
            'step_type': 'SARSA'
        }
        
        return detail
    
    def train_iteration(self, iteration: int) -> Dict[str, Any]:
        """Train for one iteration using SARSA update"""
        iteration_rewards = []
        iteration_steps = []
        iteration_step_details = []
        
        for episode in range(self.episodes_per_iteration):
            state, _ = self.env.reset()
            action = self.get_action(state)
            done = False
            episode_reward = 0
            episode_steps = 0
            episode_step_details = []
            
            while not done and episode_steps < self.max_steps_per_episode:
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                
                # Choose next action
                next_action = self.get_action(next_state) if not done else 0
                
                # Update visit count
                state_idx = self._state_to_index(state)
                self.visit_counts[state_idx, action] += 1
                
                # Get step details BEFORE update
                step_detail = self.get_step_details(state, action, reward, next_state, done, next_action)
                
                # Perform SARSA update
                next_state_idx = self._state_to_index(next_state) if not done else 0
                
                old_q = self.Q[state_idx, action]
                next_q = self.Q[next_state_idx, next_action] if not done else 0
                td_target = reward + self.discount_factor * next_q
                
                self.Q[state_idx, action] = old_q + self.learning_rate * (td_target - old_q)
                
                # Add post-update information
                step_detail['new_Q(s,a)'] = float(self.Q[state_idx, action])
                step_detail['update_applied'] = True
                step_detail['learning_rate_applied'] = float(self.learning_rate)
                
                # Store step details
                episode_step_details.append(step_detail)
                episode_reward += reward
                episode_steps += 1
                
                # Update for next step
                state = next_state
                action = next_action
            
            iteration_rewards.append(episode_reward)
            iteration_steps.append(episode_steps)
            iteration_step_details.append(episode_step_details)
        
        # Decay epsilon
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
        
        # Calculate statistics
        q_non_zero = np.count_nonzero(np.abs(self.Q) > 1e-10)
        
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
            'V_mean': float(np.mean(np.max(self.Q, axis=1))),
            'q_non_zero_count': int(q_non_zero),
            'learning_rate': float(self.learning_rate),
            'discount_factor': float(self.discount_factor)
        }
        
        self.iteration_details.append(iteration_detail)
        
        return iteration_detail