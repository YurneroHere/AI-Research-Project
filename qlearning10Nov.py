import numpy as np
import random
import matplotlib.pyplot as plt
from collections import defaultdict


class ConnectFour:
    def __init__(self):
        self.board = np.zeros((6, 7))
        self.current_player = 1

    def reset(self):
        self.board = np.zeros((6, 7))
        self.current_player = 1

    def drop_piece(self, column):
        for row in range(5, -1, -1):
            if self.board[row][column] == 0:
                self.board[row][column] = self.current_player
                return True
        return False

    def get_next_open_row(self, col):
        for r in range(6):
            if self.board[r][col] == 0:
                return r
        return None

    def is_winning_move(self):
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


class QLearningAgent:
    def __init__(self):
        self.q_table = defaultdict(lambda: np.zeros(7))
        self.epsilon = 0.3
        self.alpha = 0.1
        self.gamma = 0.95
        self.epsilon_decay = 0.9995  # Slower decay

    def get_action(self, state, valid_actions):
        if random.random() < self.epsilon:
            return random.choice(valid_actions)
        else:
            action_values = self.q_table[state]
            best_action = np.argmax(action_values)
            return best_action if best_action in valid_actions else random.choice(valid_actions)

    def update_q_value(self, state, action, reward, next_state):
        current_q = self.q_table[state][action]
        max_future_q = np.max(self.q_table[next_state]) if next_state else 0
        self.q_table[state][action] = current_q + self.alpha * (reward + self.gamma * max_future_q - current_q)

    def decay_epsilon(self):
        self.epsilon = max(0.01, self.epsilon * self.epsilon_decay)


def train_agent(episodes=30000):
    env = ConnectFour()
    agent = QLearningAgent()

    # Tracking metrics for visualization
    win_rates, cumulative_rewards, moves_per_game, exploration_counts, accuracies = [], [], [], [], []
    total_wins, total_actions, optimal_actions = 0, 0, 0

    for episode in range(episodes):
        env.reset()
        state = tuple(env.board.flatten())
        total_reward, moves, exploration_count = 0, 0, 0
        win = False

        while True:
            valid_actions = [c for c in range(7) if env.get_next_open_row(c) is not None]
            if random.random() < agent.epsilon:
                exploration_count += 1

            action = agent.get_action(state, valid_actions)

            # Calculate if the action is optimal
            best_action = np.argmax(agent.q_table[state])
            if action == best_action:
                optimal_actions += 1
            total_actions += 1

            env.drop_piece(action)
            moves += 1

            if env.is_winning_move():
                reward = 10  # Higher reward for winning
                agent.update_q_value(state, action, reward, None)
                total_reward += reward
                win = True
                break

            if 0 not in env.board[0]:  # Board is full
                reward = -1  # Draw penalty
                agent.update_q_value(state, action, reward, None)
                total_reward += reward
                break

            reward = 0.1  # Minor reward for non-terminal moves
            next_state = tuple(env.board.flatten())
            agent.update_q_value(state, action, reward, next_state)
            total_reward += reward
            state = next_state
            env.switch_player()

        agent.decay_epsilon()

        # Track metrics for visualization
        cumulative_rewards.append(total_reward)
        moves_per_game.append(moves)
        exploration_counts.append(exploration_count)
        win_rates.append(1 if win else 0)
        accuracy = optimal_actions / total_actions if total_actions > 0 else 0
        accuracies.append(accuracy * 100)

    # Calculating win rates over episodes
    win_rates = np.convolve(win_rates, np.ones(100) / 100, mode="valid")  # Rolling average over 100 episodes
    accuracies = np.convolve(accuracies, np.ones(100) / 100, mode="valid")  # Rolling average over 100 episodes

    # Plot metrics
    plot_training_performance(win_rates, cumulative_rewards, moves_per_game, exploration_counts, accuracies)
    return agent


def plot_training_performance(win_rates, cumulative_rewards, moves_per_game, exploration_counts, accuracies):
    # Plot Win Rate
    plt.figure(figsize=(12, 6))
    plt.plot(win_rates)
    plt.title("Win Rate Over Time")
    plt.xlabel("Episodes")
    plt.ylabel("Win Rate")
    plt.show()

    # Plot Cumulative Rewards
    plt.figure(figsize=(12, 6))
    plt.plot(cumulative_rewards)
    plt.title("Cumulative Rewards Over Time")
    plt.xlabel("Episodes")
    plt.ylabel("Cumulative Rewards")
    plt.show()

    # Plot Number of Moves per Game
    plt.figure(figsize=(12, 6))
    plt.plot(moves_per_game)
    plt.title("Number of Moves per Game")
    plt.xlabel("Episodes")
    plt.ylabel("Moves per Game")
    plt.show()

    # Plot Exploration Counts
    plt.figure(figsize=(12, 6))
    plt.plot(exploration_counts)
    plt.title("Exploration Count Over Time")
    plt.xlabel("Episodes")
    plt.ylabel("Number of Explorations (Random Actions)")
    plt.show()

    # Plot Accuracy Percentage Over Time
    plt.figure(figsize=(12, 6))
    plt.plot(accuracies)
    plt.title("Accuracy Percentage Over Time")
    plt.xlabel("Episodes")
    plt.ylabel("Accuracy (%)")
    plt.show()


if __name__ == "__main__":
    agent = train_agent(episodes=15000)
