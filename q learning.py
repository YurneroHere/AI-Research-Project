import numpy as np
import random


class ConnectFour:
    def __init__(self):
        self.board = np.zeros((6, 7))  # 6 rows, 7 columns
        self.current_player = 1  # Player 1 starts

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
            if r >= 0 and r < 6 and c >= 0 and c < 7 and self.board[r][c] == self.current_player:
                count += 1
            else:
                break
        return count == 4

    def switch_player(self):
        self.current_player = 2 if self.current_player == 1 else 1


class QLearningAgent:
    def __init__(self):
        self.q_table = {}  # State-action values
        self.epsilon = 0.1  # Exploration rate
        self.alpha = 0.5  # Learning rate
        self.gamma = 0.9  # Discount factor

    def get_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, 6)  # Explore: choose a random column
        else:
            return np.argmax(self.q_table.get(state, np.zeros(7)))  # Exploit: choose best action

    def update_q_value(self, state, action, reward, next_state):
        current_q = self.q_table.get(state, np.zeros(7))[action]
        max_future_q = np.max(self.q_table.get(next_state, np.zeros(7)))

        new_q = current_q + self.alpha * (reward + self.gamma * max_future_q - current_q)

        if state not in self.q_table:
            self.q_table[state] = np.zeros(7)

        self.q_table[state][action] = new_q


def train_agent(episodes=10000):
    env = ConnectFour()
    agent = QLearningAgent()

    for episode in range(episodes):
        env.__init__()  # Reset environment for new episode
        state = str(env.board)  # Convert board to a string to use as a state

        while True:
            action = agent.get_action(state)
            valid_move = env.drop_piece(action)

            if not valid_move:
                continue

            if env.is_winning_move():
                reward = 1
                next_state = None
            else:
                reward = 0
                next_state = str(env.board)

            agent.update_q_value(state, action, reward, next_state)

            if next_state is None:
                break

            state = next_state

        env.switch_player()
    return agent

def generate_random_move():
    return random.randint(0, 6)

if __name__ == "__main__":
    agent = train_agent()
    cf = ConnectFour()

    game_over = False
    turn = 1
    draw = False

    while not game_over:
        row = None
        col = None
        if cf.current_player == 1:
            while row is None:  # Ensure the move is valid
                col = generate_random_move()
                row = cf.get_next_open_row(col)

        else:
            while row is None:
                col = agent.get_action(cf.board)
                row = cf.get_next_open_row(col)

        cf.drop_piece(col)

        if cf.is_winning_move():
            game_over = True

        if 0 not in cf.board[0]:
            game_over = True
            draw = True

    if draw:
        print("nobody won")
    else:
        print(cf.current_player, " won")