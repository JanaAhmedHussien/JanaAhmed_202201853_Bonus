# 🎮 RL Algorithms Suite - Comprehensive Reinforcement Learning Platform

## 📖 Overview

The **RL Algorithms Suite** is an interactive web application built with Streamlit that provides a comprehensive platform for experimenting with and visualizing various reinforcement learning algorithms. The application allows users to train RL agents on different environments, adjust hyperparameters in real-time, and visualize the learning process through detailed dashboards and animations.

## ✨ Key Features

### 🎯 **Core Capabilities**
- **Interactive Training**: Train RL algorithms in real-time with progress visualization
- **Multiple Environments**: Support for 5 distinct RL environments
- **Diverse Algorithms**: Implementation of 5 fundamental RL algorithms
- **Real-time Visualization**: Dynamic plots and dashboards showing training progress
- **Memory Optimization**: Intelligent memory management for extended training sessions
- **Step-by-Step Details**: Complete transparency into algorithm computations

### 🏗️ **Architecture**
- **Frontend**: Streamlit with Plotly for interactive visualizations
- **Backend**: Custom algorithm implementations with NumPy
- **State Management**: Session-based persistence for uninterrupted training
- **Memory Management**: Garbage collection and data limitation strategies

## 🎮 **Supported Environments**

### 1. **FrozenLake-v1** (4x4)
- **Description**: Navigate from starting position (S) to goal (G) without falling into holes
- **Type**: Discrete state space
- **Action Space**: 4 directions (Up, Down, Left, Right)
- **Reward**: +1 for reaching goal, 0 otherwise

### 2. **Taxi-v3**
- **Description**: Pick up and drop off passengers at specified locations
- **Type**: Discrete state space
- **Action Space**: 6 actions (Move, Pickup, Dropoff)
- **Reward**: +20 for successful dropoff, -1 per step, -10 for illegal actions

### 3. **CartPole-v1**
- **Description**: Balance a pole on a moving cart
- **Type**: Continuous state space
- **Action Space**: 2 actions (Left, Right)
- **Reward**: +1 for each timestep the pole remains balanced

### 4. **MountainCar-v0**
- **Description**: Drive a car to the top of a hill
- **Type**: Continuous state space
- **Action Space**: 3 actions (Accelerate left, Coast, Accelerate right)
- **Reward**: -1 per timestep until reaching goal

### 5. **CliffWalking-v1**
- **Description**: Navigate along a cliff without falling
- **Type**: Discrete state space
- **Action Space**: 4 directions
- **Reward**: -1 per step, -100 for falling off cliff

## 🧠 **Reinforcement Learning Algorithms**

### 1. **Q-Learning**
- **Type**: Model-free, off-policy TD control
- **Update Rule**: Q(s,a) ← Q(s,a) + α[r + γ·maxₐ'Q(s',a') - Q(s,a)]
- **Exploration**: ε-greedy strategy
- **Features**: 
  - Full Q-value visualization
  - Visit counts tracking
  - Action probability displays
  - Epsilon decay visualization

### 2. **SARSA** (State-Action-Reward-State-Action)
- **Type**: Model-free, on-policy TD control
- **Update Rule**: Q(s,a) ← Q(s,a) + α[r + γ·Q(s',a') - Q(s,a)]
- **Exploration**: ε-greedy strategy
- **Features**:
  - Next-action tracking
  - On-policy learning visualization
  - Comparison with Q-Learning

### 3. **Monte Carlo** with Exploring Starts
- **Type**: Model-free, policy evaluation and improvement
- **Approach**: Learns from complete episodes
- **Update Rule**: Average returns for state-action pairs
- **Features**:
  - First-visit implementation
  - Return statistics
  - Exploring starts strategy
  - Episode-based learning visualization

### 4. **Policy Iteration**
- **Type**: Model-based, dynamic programming
- **Components**: 
  - **Policy Evaluation**: Iterative computation of state values
  - **Policy Improvement**: Greedy policy updates
- **Requirements**: Transition model of environment
- **Features**:
  - Value iteration convergence visualization
  - Policy stability tracking
  - Complete Q-function derivation

### 5. **Value Iteration**
- **Type**: Model-based, dynamic programming
- **Approach**: Directly computes optimal value function using Bellman optimality equation
- **Update Rule**: V(s) ← maxₐ∑ₛ'P(s'|s,a)[R(s,a,s') + γ·V(s')]
- **Features**:
  - Bellman optimality updates
  - Policy extraction from value function
  - Convergence monitoring

## ⚙️ **Hyperparameter Configuration**

### **Learning Parameters**
- **Learning Rate (α)**: Controls update step size (0.001 - 1.0)
- **Discount Factor (γ)**: Future reward discount (0.1 - 0.99)
- **Exploration Rate (ε)**: Probability of random action (0.0 - 1.0)
- **Epsilon Decay**: Reduction rate for exploration over time (0.9 - 0.999)
- **Min Epsilon**: Minimum exploration probability (0.01)

### **Training Parameters**
- **Iterations**: Number of training iterations (1 - 100)
- **Episodes per Iteration**: Episodes per training iteration (1 - 1000)
- **State Discretization**: Number of bins for continuous state spaces (5 - 50)

## 📊 **Visualization Features**

### **1. Training Dashboard** (Tab 1)
- **Progress Bar**: Real-time training progress
- **Metrics Display**: Iteration, mean reward, mean steps, exploration rate
- **Training Progress Plot**: Multi-panel visualization including:
  - Mean reward over iterations
  - Value function statistics
  - Q-table statistics
  - Exploration rate decay
- **Value Function Analysis**: Distribution and sorted value plots

### **2. Algorithm Details** (Tab 2)
- **Iteration Selection**: Choose specific iteration to examine
- **Episode Selection**: Select episodes within iterations
- **Step-by-Step Details**: Complete algorithm computations including:
  - State representations
  - Action selections with probabilities
  - Q-value updates
  - TD errors and targets
  - Policy updates
  - Visit counts

### **3. Policy Table** (Tab 3)
- **Optimal Policy Display**: Tabular format showing optimal actions per state
- **Statistics**: Action distributions, value function statistics
- **Download Capability**: Export policy tables as CSV
- **Policy Visualization**: Action distribution bar charts

### **4. Animation** (Tab 4)
- **Policy Execution**: Animated visualization of learned policy
- **Controls**: Speed adjustment, frame selection
- **Statistics**: Real-time algorithm statistics
- **Q-Function Heatmap**: Visual representation of Q-values
- **Execution Details**: Step-by-step action and reward tracking

### **5. Summary** (Tab 5)
- **Overall Statistics**: Best reward, average reward, training time
- **Reward Progression**: Moving average and cumulative reward plots
- **Algorithm Performance**: Q-table and value function statistics
- **Export Functionality**: Download training summary as JSON

## 💾 **Memory Management**

### **Automatic Memory Optimization**
- **Usage Monitoring**: Real-time memory usage tracking
- **Automatic Cleanup**: Regular garbage collection during training
- **Data Limitation**: Automatic truncation of stored data
- **Warning System**: Alerts for high memory usage

### **Manual Controls**
- **Memory Check**: Current usage display
- **Manual Cleanup**: Aggressive memory cleanup
- **Session Reset**: Complete memory clearance

## 🛠️ **Technical Implementation**

### **Algorithm Base Class**
- **Abstract Methods**: train_iteration, get_policy, get_value_function
- **Common Functionality**: Policy evaluation, memory cleanup, statistics tracking
- **State Management**: Session state persistence

### **State Discretization**
- **Continuous Spaces**: Adaptive binning based on environment bounds
- **Discrete Conversion**: Consistent state indexing across algorithms
- **Flexible Handling**: Support for both discrete and continuous environments

### **Error Handling**
- **Graceful Degradation**: Dummy classes for missing imports
- **Detailed Error Reporting**: Full traceback display in expandable sections
- **Memory Safety**: Automatic cleanup on errors

## 🚀 **Getting Started**

### **Prerequisites**
```bash
pip install streamlit gymnasium numpy pandas plotly pillow psutil
```

### **Running the Application**
```bash
streamlit run app.py
```

### **Basic Workflow**
1. **Select Environment**: Choose from available environments
2. **Choose Algorithm**: Select RL algorithm to train
3. **Configure Parameters**: Adjust hyperparameters using sliders
4. **Start Training**: Click "Start Training" button
5. **Monitor Progress**: Watch training metrics in real-time
6. **Analyze Results**: Explore different tabs for detailed analysis

## 📈 **Performance Features**

### **Training Efficiency**
- **Iterative Processing**: One iteration at a time for responsive UI
- **Progress Saving**: Results persist across UI interactions
- **Batch Processing**: Multiple episodes per iteration for faster learning

### **Visualization Performance**
- **Plotly Integration**: Interactive, responsive plots
- **Data Limiting**: Automatic data truncation for large datasets
- **Lazy Loading**: On-demand visualization generation

## 🔧 **Advanced Features**

### **Policy Execution**
- **Real-time Animation**: Policy visualization with frame capture
- **Step Details**: Complete algorithm information per step
- **Reward Tracking**: Cumulative reward display during execution


## 📋 **Export Capabilities**

### **Data Export**
- **Policy Tables**: CSV export of optimal policies
- **Training Summaries**: JSON export of training results
- **Visualizations**: Interactive plots with export options

### **Reproducibility**
- **Parameter Persistence**: Hyperparameter settings saved in session
- **Deterministic Results**: Seeded random number generation
- **Complete State**: All training details preserved

## 🎨 **UI/UX Design**

### **Modern Interface**
- **Responsive Layout**: Adapts to different screen sizes
- **Intuitive Navigation**: Tab-based organization
- **Visual Hierarchy**: Clear information organization

### **Interactive Elements**
- **Real-time Controls**: Sliders, buttons, selectors
- **Expandable Sections**: Detailed information on demand
- **Progress Indicators**: Visual feedback for long operations

### **Visual Design**
- **Color Coding**: Consistent color scheme for different algorithms
- **Informative Cards**: Key metrics in visually distinct containers
- **Animations**: Smooth transitions and progress indicators

## 🔍 **Educational Value**

### **Learning Features**
- **Transparent Algorithms**: Complete step-by-step computations
- **Interactive Exploration**: Real-time parameter adjustment
- **Visual Learning**: Graphical representation of abstract concepts

### **Algorithm Insights**
- **Policy Evolution**: Watch policies improve over time
- **Value Function Development**: See value functions converge
- **Exploration-Exploitation**: Visualize the trade-off

## 🚨 **Limitations and Considerations**

### **Current Limitations**
- **Memory Intensive**: Large Q-tables for continuous spaces
- **Training Time**: Complex environments may require longer training
- **Model Requirements**: Some algorithms need environment models



---
