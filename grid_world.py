from typing import Union, List, Tuple
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from visualization import plot_heatmap_with_arrows as heat_map  

class GridWorld:
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3

    def __init__(
        self,
        height: int,
        width: Union[int, None] = None,
        actions: Union[List[int], None] = None,
        rewards: Union[List[float], None] = None,
        proba: float = 0.8,
        terminal_states: Union[List[tuple], None] = None,
        barrier_states: Union[List[tuple], None] = None,
        initial_state: Union[tuple, None] = None
    ):
        
        """
        Initialize a GridWorld object.

        Parameters:
        - height (int): The height of the grid.
        - width (Optional[int]): The width of the grid. If not provided, defaults to height.
        - actions (Optional[List[int]]): List of possible actions (default: [0, 1, 2, 3]).
        - rewards (Optional[List[float]]): The rewards grid. If not provided, defaults to zeros.
        - proba (float): Probability of actions succeeding (default: 0.8).
        - terminal_states (Optional[List[tuple]]): List of terminal states (default: None).
        - barrier_states (Optional[List[tuple]]): List of barrier states (default: None).
        - initial_state (Optional[tuple]): Initial state (default: None).
        """
            
        # Assert height and width
        assert isinstance(height, int) and height > 0, "Height should be a positive integer"
        self.height = height
        if width is not None:
            assert isinstance(width, int) and width > 0, "Width should be a positive integer"
            self.width = width
        else:
            self.width = self.height  # Set width as height if not specified

        # Actions 
        self.actions = actions if actions is not None else [self.UP, self.DOWN, self.LEFT, self.RIGHT]

        # Number of states and actions
        self.n_states = self.height * self.width
        self.n_actions = len(self.actions)

        # Transition probabilities
        assert 0 <= proba <= 1, "Probability should be between 0 and 1"
        self.transition_probabilities = [proba] + [(1 - proba) / (len(self.actions) - 1)] * (len(self.actions) - 1)

        # Terminal and barrier states
        if terminal_states is not None:
            assert all(isinstance(state, tuple) and len(state) == 2 and 0 <= state[0] < self.width and 0 <= state[1] < self.height for state in terminal_states), "Terminal states should be tuples of length 2 within grid bounds"
        else:
            terminal_states = []
        self.terminal_states = terminal_states

        if barrier_states is not None:
            assert all(isinstance(state, tuple) and len(state) == 2 and 0 <= state[0] < self.width and 0 <= state[1] < self.height for state in barrier_states), "Barrier states should be tuples of length 2 within grid bounds"
        else:
            barrier_states = []
        self.barrier_states = barrier_states

        self.initial_state = initial_state

        # Initialize rewards grid
        if rewards is None:
            self.rewards = np.full((self.height, self.width), -1)  # Fill with -1 everywhere

            # Update terminal states to have 0 reward
            if self.terminal_states != []:
                for terminal_state in self.terminal_states:
                    x, y = terminal_state
                    self.rewards[y, x] = 0
            else:
                # If no terminal states, set the reward of the bottom-right cell to 0
                self.rewards[self.height - 1, self.width - 1] = 0
                self.terminal_states = [(self.height - 1, self.width - 1)]
        else:
            self.rewards = np.array(rewards)
            assert self.rewards.shape == (self.height, self.width), "Rewards array size should match grid dimensions"

        # State initialization (ensure self.reset() exists)
        self.encoded_state = self.reset()
        self.state = self.decode_state(self.encoded_state)

        # Transition sequence
        self.transitions = []
        

    # --- State Handling ---
    def reset(self) -> int:
        self.transitions = []  # Reset transition sequence
        if self.initial_state is not None:
            # Reset to the initial state
            self.state = self.initial_state
        else:
            # Reset to a random starting position not in terminal states
            while True:
                x = np.random.randint(self.width)
                y = np.random.randint(self.height)
                if (x, y) not in self.terminal_states:
                    break
            self.state = (x,y)
        self.encoded_state = self.encode_state(self.state)
        return self.encoded_state

    def is_terminal_state(self, state: Tuple[int, int]) -> bool:
        # Check if given state is terminal based on self.rewards
        return state in self.terminal_states

    def get_reward(self, x: int, y: int) -> float:
        # Get reward for specific position in the grid
        return self.rewards[y][x]
    
    def encode_state(self, state: Tuple[int, int]) -> int:
        # Encode the state as a single integer
        x, y = state
        return x * self.width + y
    
    def decode_state(self, encoded_state: int) -> Tuple[int, int]:
        # Decode the integer state representation
        x = encoded_state // self.width
        y = encoded_state % self.width
        return x, y

    # --- Agent Movement ---
    def move_agent(self, action: int) -> Tuple[int, int]:
        # Move agent within the grid based on the action
        x, y = self.state

        if action == self.UP and y < self.height - 1 and (x, y + 1) not in self.barrier_states:
            return x, y + 1  # Up
        elif action == self.DOWN and y > 0 and (x, y - 1) not in self.barrier_states:
            return x, y - 1  # Down
        elif action == self.LEFT and x > 0 and (x - 1, y) not in self.barrier_states:
            return x - 1, y  # Left
        elif action == self.RIGHT and x < self.width - 1 and (x + 1, y) not in self.barrier_states:
            return x + 1, y  # Right
        else:
            return x, y  # No valid action or blocked, state remains the same
        
        
    # --- Step Functionality ---
    def step(self, input_action: int) -> Tuple[int, float, bool, dict]:
        # Perform a step in the GridWorld based on the input action

        current_state = self.state

        # Define the numbers and their probabilities
        actions = list(range(self.n_actions))
        other_actions = [action for action in actions if action != input_action]
        numbers = [input_action] + other_actions

        # Sample an action with the specified probabilities
        sampled_action = np.random.choice(numbers, p=self.transition_probabilities)

        # Get the new state after the agent's movement based on the sampled action
        new_state = self.move_agent(sampled_action)

        # Calculate the reward of the new state
        reward = self.get_reward(new_state[0], new_state[1])

        # Check if the episode is done (new state is a terminal state)
        done = self.is_terminal_state(new_state)

        # Update the state
        self.state = new_state
        self.encoded_state = self.encode_state(self.state)

        # Record the transition
        self.transitions.append((current_state, sampled_action, reward, new_state, done))

        return self.encoded_state, reward, done, {}
    
    
    # --- Visualization ---
    def arrow_map(self, action):
        """
        Converts action integers into arrow symbols.

        Parameters:
        - action (int): Integer representing the action.

        Returns:
        - str: Corresponding arrow symbol.
        """
        if action == self.UP:
            return '↓' # since the y-axis is inverted
        elif action == self.DOWN:
            return '↑' # since the y-axis is inverted
        elif action == self.LEFT:
            return '←'
        elif action == self.RIGHT:
            return '→'
        else:
            return '⊗'  # Symbol for an unknown or invalid action (optional)
    
    def render(self):
        """
        Render a heatmap depicting arrow transitions based on the agent's simulation.
        """
        # Generate arrow transitions based on the agent's actions during simulation
        arrow_transitions = [self.arrow_map(action) for (_, action, _, _, _) in self.transitions]
        
        # Extract locations from the simulation states
        locations = [(x, y) for ((x, y), _, _, _, _) in self.transitions]

        # Plot heatmap using rewards and arrow transitions
        heat_map(self.rewards, arrow_transitions, locations)

    def plot_policy(self, pi, values=None, title=None):
        """
        Plot the policy values for each state.

        Parameters:
        pi (np.ndarray): Policy values for each state.
        values (2D array): Values to be plotted as a heatmap (default is self.rewards).
        title (str): Title of the plot (default is 'Heat Map of Policy').
        """
        if values is None:
            values = self.rewards

        if title is None:
            title = "Heat Map of Policy"

        # Create arrows depicting policy values for non-terminal states
        arrow_policy = [self.arrow_map(pi[i]) for i in range(len(pi)) if self.decode_state(i) not in self.terminal_states]
        
        # Extract locations of non-terminal states for plotting policy values
        locations = [self.decode_state(i) for i in range(len(pi)) if self.decode_state(i) not in self.terminal_states]

        # Plot heatmap using policy values and arrow representations
        heat_map(values, arrow_policy, locations, title=title)

    def plot_trajectory(self, pi):
        assert self.initial_state is not None, "Initial state not specified"
        self.reset()
        self.state = self.initial_state
        locations = [self.state]
        actions = []
        while self.state not in self.terminal_states:
            action = pi[self.encode_state(self.state)]
            self.state = self.move_agent(action)
            actions.append(action)
            locations.append(self.state)
        arrows = [self.arrow_map(action) for action in actions]
        heat_map(self.rewards, arrows, locations, title="Trajectory")

    def __str__(self):
        # Print the GridWorld object
        return f"GridWorld(height={self.height}, width={self.width}, rewards={self.rewards}, terminal_states={self.terminal_states}, barrier_states={self.barrier_states})"

    # --- Close --- 
    def close(self):
        pass

        


