import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns


# Connect Four game logic
class ConnectFour:
    def __init__(self):
        self.board = np.zeros((6, 7))  # 6 rows, 7 columns
        self.current_player = 1  # Player 1 starts

    def drop_piece(self, column):
        for row1 in range(5, -1, -1):
            if self.board[row1][column] == 0:
                self.board[row1][column] = self.current_player
                return True
        return False

    def get_next_open_row(self, col):
        for r in range(6):
            if self.board[r][col] == 0:
                return r
        return None

    def is_winning_move(self):
        # Check horizontal, vertical, and diagonal for a win
        for c in range(7):
            for r in range(6):
                if self.board[r][c] == self.current_player:
                    if (self.check_direction(r, c, 1, 0) or  # Horizontal
                            self.check_direction(r, c, 0, 1) or  # Vertical
                            self.check_direction(r, c, 1, 1) or  # Diagonal /
                            self.check_direction(r, c, 1, -1)):  # Diagonal \
                        return True
        return False

    def check_direction(self, row, col, delta_row, delta_col):
        count = 0
        for i in range(4):
            r = row + i * delta_row
            c = col + i * delta_col
            if 0 <= r < 6 and 0 <= c < 7 and self.board[r][c] == self.current_player:
                count += 1
            else:
                break
        return count == 4

    def switch_player(self):
        self.current_player = 2 if self.current_player == 1 else 1

    def count_consecutive(self, row, col, delta_row, delta_col):
        """Helper function to count consecutive pieces in a given direction."""
        count = 0
        for i in range(4):
            r = row + i * delta_row
            c = col + i * delta_col
            if 0 <= r < 6 and 0 <= c < 7 and self.board[r][c] == self.current_player:
                count += 1
            else:
                break
        return count

    def get_favorable_reward(self):
        """Reward the agent based on the current board configuration."""
        reward = 0
        for r in range(6):
            for c in range(7):
                if self.board[r][c] == self.current_player:
                    # Check for sequences of connected pieces
                    if self.count_consecutive(r, c, 1, 0) == 2:  # Vertical
                        reward += 0.5
                    if self.count_consecutive(r, c, 0, 1) == 2:  # Horizontal
                        reward += 0.5
                    if self.count_consecutive(r, c, 1, 1) == 2:  # Diagonal /
                        reward += 0.5
                    if self.count_consecutive(r, c, 1, -1) == 2:  # Diagonal \
                        reward += 0.5
                    if self.count_consecutive(r, c, 1, 0) == 3:  # Vertical
                        reward += 1
                    if self.count_consecutive(r, c, 0, 1) == 3:  # Horizontal
                        reward += 1
                    if self.count_consecutive(r, c, 1, 1) == 3:  # Diagonal /
                        reward += 1
                    if self.count_consecutive(r, c, 1, -1) == 3:  # Diagonal \
                        reward += 1
        return reward

    def is_losing_move(self):
        """Check if the opponent can win on their next move."""
        opponent = 2 if self.current_player == 1 else 1
        for col in range(7):
            temp_board = np.copy(self.board)
            for row in range(5, -1, -1):
                if temp_board[row][col] == 0:
                    temp_board[row][col] = opponent
                    if self.is_winning_move_for_opponent(temp_board, opponent):
                        return True
                    break
        return False

    def is_winning_move_for_opponent(self, board, player):
        """Check if the opponent has a winning move on the given board."""
        for c in range(7):
            for r in range(6):
                if board[r][c] == player:
                    if (self.check_direction(r, c, 1, 0) or  # Horizontal
                            self.check_direction(r, c, 0, 1) or  # Vertical
                            self.check_direction(r, c, 1, 1) or  # Diagonal /
                            self.check_direction(r, c, 1, -1)):  # Diagonal \
                        return True
        return False


class QLearningAgent:
    def __init__(self):
        self.q_table = {}
        self.epsilon = 0.1
        self.alpha = 0.001  # Very low learning rate for fine-tuning
        self.gamma = 0.99
        self.epsilon_decay = 0.999


    def get_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, 6)  # Explore: choose a random column
        else:
            return np.argmax(self.q_table.get(state, np.zeros(7)))  # Exploit: choose best action

    def get_valid_action(self, env):
        """Ensure the selected action is valid (column is not full)."""
        valid_actions = [col for col in range(7) if env.get_next_open_row(col) is not None]
        if valid_actions:
            if random.random() < self.epsilon:
                return random.choice(valid_actions)  # Randomly explore among valid actions
            else:
                best_action = np.argmax(self.q_table.get(str(env.board), np.zeros(7)))
                if best_action in valid_actions:
                    return best_action
                else:
                    return random.choice(valid_actions)
        else:
            return random.randint(0, 6)  # Fallback, though this should never happen

    def update_q_value(self, state, action, reward, next_state):
        current_q = self.q_table.get(state, np.zeros(7))[action]
        max_future_q = np.max(self.q_table.get(next_state, np.zeros(7)))

        new_q = current_q + self.alpha * (reward + self.gamma * max_future_q - current_q)

        if state not in self.q_table:
            self.q_table[state] = np.zeros(7)

        self.q_table[state][action] = new_q

    def decay_epsilon(self):
        self.epsilon = max(0.01, self.epsilon * 0.999)  # Faster decay


def train_agent(episodes=10000):
    env = ConnectFour()
    agent = QLearningAgent()

    for episode in range(episodes):
        env.__init__()
        state = str(env.board)

        while True:
            action = agent.get_valid_action(env)
            valid_move = env.drop_piece(action)

            if env.is_winning_move():
                reward = 200  # High reward for winning
                agent.update_q_value(state, action, reward, None)
                break

            if env.is_losing_move():
                reward = -200  # High penalty for losing
                agent.update_q_value(state, action, reward, None)
                break

            if 0 not in env.board[0]:  # Full board, game ends in a draw
                reward = -20  # Penalty for drawing
                agent.update_q_value(state, action, reward, None)
                break

            reward = 0.001  # Very minimal intermediate reward to avoid distractions
            total_reward = -0.1  # Small penalty for taking too many moves
            total_reward += reward

            next_state = str(env.board)
            agent.update_q_value(state, action, total_reward, next_state)

            state = next_state
            env.switch_player()

        agent.decay_epsilon()
    return agent



# Track and visualize training performance
def track_training_performance(agent, episodes=10000, print_interval=100):
    win_rates = []  # Track win rate over time
    cumulative_rewards = []  # Track cumulative rewards per episode
    moves_per_game = []  # Track the number of moves in each episode
    exploration_counts = []  # Track how many times the agent explored

    for episode in range(1, episodes + 1):
        env = ConnectFour()
        state = str(env.board)  # Initial state
        total_reward = 0
        moves = 0
        exploration_count = 0

        while True:
            if random.random() < agent.epsilon:
                exploration_count += 1
            action = agent.get_valid_action(env)

            valid_move = env.drop_piece(action)
            if not valid_move:
                continue

            moves += 1

            if env.is_winning_move():
                reward = 10
                total_reward += reward
                break

            if env.is_losing_move():
                reward = -10
                total_reward += reward
                break

            if 0 not in env.board[0]:
                reward = -1
                total_reward += reward
                break

            reward = env.get_favorable_reward()
            total_reward += reward

            next_state = str(env.board)
            agent.update_q_value(state, action, reward, next_state)
            state = next_state
            env.switch_player()

            # Track statistics
        cumulative_rewards.append(total_reward)
        moves_per_game.append(moves)
        exploration_counts.append(exploration_count)

        # Check win rate (assuming player 1 is the trained agent)
        if env.current_player == 1:  # Agent won
            win_rates.append(1)
        else:  # Agent lost or drew
            win_rates.append(0)

        # Decay epsilon after each episode
        agent.decay_epsilon()

        # Print progress every print_interval episodes
        if episode % print_interval == 0:
            avg_win_rate = np.mean(win_rates[-print_interval:])
            avg_reward = np.mean(cumulative_rewards[-print_interval:])
            avg_moves = np.mean(moves_per_game[-print_interval:])
            print(
                f"Episode {episode}: Win Rate = {avg_win_rate:.2f}, Avg Reward = {avg_reward:.2f}, Avg Moves = {avg_moves:.2f}")

    return win_rates, cumulative_rewards, moves_per_game, exploration_counts

    # Visualization function
def plot_training_performance(win_rates, cumulative_rewards, moves_per_game, exploration_counts,
                                  print_interval=100):
    # Plot Win Rate
    plt.figure(figsize=(10, 6))
    plt.plot(np.convolve(win_rates, np.ones(print_interval) / print_interval, mode='valid'))
    plt.title("Win Rate Over Time")
    plt.xlabel("Episodes")
    plt.ylabel("Win Rate")
    plt.show()
    plt.close()

    # Plot Cumulative Rewards
    plt.figure(figsize=(10, 6))
    plt.plot(cumulative_rewards)
    plt.title("Cumulative Rewards Over Time")
    plt.xlabel("Episodes")
    plt.ylabel("Cumulative Rewards")
    plt.show()
    plt.close()

    # Plot Number of Moves per Game
    plt.figure(figsize=(10, 6))
    plt.plot(moves_per_game)
    plt.title("Number of Moves per Game")
    plt.xlabel("Episodes")
    plt.ylabel("Moves per Game")
    plt.show()
    plt.close()

    # Plot Exploration vs Exploitation
    plt.figure(figsize=(10, 6))
    plt.plot(exploration_counts)
    plt.title("Exploration Count Over Time")
    plt.xlabel("Episodes")
    plt.ylabel("Number of Explorations (Random Actions)")
    plt.show()
    plt.close()

    # Example usage of training and tracking performance
if __name__ == "__main__":
    # Train the agent
    agent = train_agent(episodes=15000)

    # Track and plot training performance
    win_rates, cumulative_rewards, moves_per_game, exploration_counts = track_training_performance(agent,
                                                                                                       episodes=15000)

    # Visualize the training performance
    plot_training_performance(win_rates, cumulative_rewards, moves_per_game, exploration_counts)
