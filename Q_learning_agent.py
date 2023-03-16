import numpy as np

class QLearningAgent:
    def __init__(self, nr_states, alpha, gamma, exploration_rate, decay):
        self.possible_actions= [0, 1]

        self.nr_states = nr_states
        self.nr_actions = len(self.possible_actions)

        self.alpha = alpha
        self.gamma = gamma
        self.exploration_rate = exploration_rate
        self.decay = decay


        self.q_table = np.zeros((10, 10, 10, 10, self.nr_actions))
    

    def compute_action(self, state):
        self.state = state
        cart_position, cart_velocity, pole_angle, pole_velocity_at_tip = self.compute_discrete_state(state)
        self.exploration_rate = self.exploration_rate * self.decay
        if np.random.uniform() <= self.exploration_rate:
            # Exploration
            self.action = np.random.choice(self.possible_actions, 1)[0]
        else:
            # Exploitation
            self.action = np.argmax(self.q_table[cart_position, cart_velocity, pole_angle, pole_velocity_at_tip, :])
        return self.action
    
    def train_agent(self, state, new_state, reward):
        cart_position, cart_velocity, pole_angle, pole_velocity_at_tip = self.compute_discrete_state(state)
        cart_position_n, cart_velocity_n, pole_angle_n, pole_velocity_at_tip_n = self.compute_discrete_state(new_state)
        self.q_table[cart_position, cart_velocity, pole_angle, pole_velocity_at_tip, self.action] = (1-self.alpha) * self.q_table[cart_position, cart_velocity, pole_angle, pole_velocity_at_tip, self.action] + self.alpha * (reward + self.gamma * self.q_table[cart_position_n, cart_velocity_n, pole_angle_n, pole_velocity_at_tip_n, np.argmax(self.q_table[cart_position_n, cart_velocity_n, pole_angle_n, pole_velocity_at_tip_n, :])])
    

    def change_exploration(self):
        self.exploration_rate = self.exploration_rate * self.decay
    
    def create_bins(self, possible_states, nr_bins):
        nr_bins -= 1
        self.bins = np.zeros((len(possible_states), nr_bins))
        for i, possible_state in enumerate(possible_states):
            self.bins[i] = np.linspace(possible_state[0], possible_state[1], nr_bins)
        
    
    def compute_discrete_state(self, states):
        discrete_states: list = []
        for i, state in enumerate(states):
            discrete_states.append(np.digitize(state, self.bins[i]))
        return discrete_states
        