import numpy as np

LENGTH = 3

def play_game(p1, p2, env, draw=False):
	# print("Enter function play_game")
	# loop until the game is over
	current_player = None
	while not env.game_over():
		# alternate between players
		# p1 always starts first
		if current_player == p1:
			current_player = p2
		else:
			current_player = p1

		# draw the board for the user who wants to make a move
		if draw:
			if draw == 1 and current_player == p1:
				env.draw_board()
			if draw == 2 and current_player == p2:
				env.draw_board()

		# current player makes a move
		current_player.take_action(env)

		# update state histories
		state = env.get_state()
		p1.update_state_history(state)
		p2.update_state_history(state)

	if draw:
		env.draw_board()

	# do the value function update
	p1.update(env)
	p2.update(env)

def get_state_hash_and_winner(env, i=0, j=0):
	results = []

	for v in [0, env.x, env.o]:
		env.board[i,j] = v # if empty it should already be zero
		if j==2:
			# j goes back to 0, increase i (unless i=2), then we are done
			if i==2:
				# the board is full, collect results and return
				state = env.get_state()
				ended = env.game_over(force_recalculate=True)
				winner = env.winner
				results.append((state, winner, ended))
			else:
				results += get_state_hash_and_winner(env, i + 1, 0)
		else:
			# increment j, i stays the same
			results += get_state_hash_and_winner(env, i, j + 1)

	return results

def initialV_x(env, state_winner_triples):
	# initialize the value function as follows
	# if x wins, V(x) = 1,
	# if x loses or draws, V(x) = 0,
	# otherwise, V(x) = 0.5
	V = np.zeros(env.num_states)
	for state, winner, ended in state_winner_triples:
		if ended:
			if winner == env.x:
				v = 1
			else:
				v = 0
		else:
			v = 0.5
		V[state] = v
	return V

def initialV_o(env, state_winner_triples):
	# this is almost opposite of initial V of x
	# whenever x wins, o loses
	# but a draw is still 0 for o
	V = np.zeros(env.num_states)
	for state, winner, ended in state_winner_triples:
		if ended:
			if winner == env.o:
				v = 1
			else:
				v = 0
		else:
			v = 0.5
		V[state] = v
	return V

class Environment:
	def __init__(self):
		self.board = np.zeros((LENGTH, LENGTH))
		self.x = -1 # represents a x on the board, player 1
		self.o = 1 # represents a o on the board, player 2
		self.winner = None
		self.ended = False
		self.num_states = 3**(LENGTH*LENGTH)

	def is_empty(self, i, j):
		return self.board[i,j] == 0

	def reward(self, sym):
		# no reward until the game is finished
		if not self.game_over():
			return 0

		# if we get here that means the game is over
		# sym will be either self.x or self.o
		return 1 if self.winner == sym else 0

	def get_state(self):
		# print("Enter function get_state")
		# this will return the current state as an interer value
		# from 0...[S]-1, where [S] is set of all possible states
		# [S] = 3^BOARD_SIZE, since each cell can have 3 possible values, empty, x or o
		# some states are not possible but we ignore that detail here.
		k = 0
		h = 0
		for i in range(LENGTH):
			for j in range(LENGTH):
				if self.board[i,j] == 0:
					v = 0
				elif self.board[i,j] == self.x:
					v = 1
				elif self.board[i,j] == self.o:
					v = 2
				h += (3**k) * v
				k += 1
		return h

	def draw_board(self):
		for i in range(LENGTH):
			print("-------------")
			for j in range(LENGTH):
				print("  ", end="")
				if self.board[i,j] == self.x:
					print("x ", end="")
				elif self.board[i,j] == self.o:
					print("o ", end="")
				else:
					print("  ", end="")
			print("")
		print("-------------")

	def game_over(self, force_recalculate=False):
		# print("Enter function game_over")
		# returns true if game over (a player has won or it's a draw)
		# otherwise returns false
		# also sets 'winner' instance variable and 'ended' instance variable
		if not force_recalculate and self.ended:
			return self.ended

		# check rows
		for i in range(LENGTH):
			for player in (self.x, self.o):
				if self.board[i].sum() == player*LENGTH:
					self.winner = player
					self.ended = True
					return True

		# check columns
		for j in range(LENGTH):
			for player in (self.x, self.o):
				if self.board[:,j].sum() == player*LENGTH:
					self.winner = player
					self.ended = True
					return True

		# check diagonals
		for player in (self.x, self.o):
			# top-left -> bottom-right diagonal
			if self.board.trace() == player*LENGTH:
				self.winner = player
				self.ended = True
				return True
			# top-right -> bottom-left diagonal
			if np.fliplr(self.board).trace() == player*LENGTH:
				self.winner = player
				self.ended = True
				return True

		# check if draw
		if np.all((self.board == 0) == False):
			# winner stays None
			self.winner = None
			self.ended = True
			return True

		# game is not over
		self.winner = None
		return False

	def is_draw(self):
		return self.ended and self.winner is None

class Agent:
	def __init__(self, eps=0.1, alpha=0.5):
		self.eps = eps
		self.alpha = alpha
		self.verbose = False
		self.state_history = []

	def setV(self, V):
		self.V = V

	def set_symbol(self, sym):
		self.sym = sym

	def set_verbose(self, v):
		self.verbose = v

	def reset_history(self):
		self.state_history = []

	def take_action(self, env):
		# print("Enter function take_action")
		# choose an action based on epsilin-greedy strategy
		r = np.random.rand()
		best_state = None
		if r < self.eps:
			# take a random action
			if self.verbose:
				print("Taking a random action")
			possible_moves = []
			for i in range(LENGTH):
				for j in range(LENGTH):
					if env.is_empty(i,j):
						possible_moves.append((i, j))
			idx = np.random.choice(len(possible_moves))
			next_move = possible_moves[idx]
		else:
			pos2val = {} # for debugging
			next_move = None
			best_value = -1
			for i in range(LENGTH):
				for j in range(LENGTH):
					if env.is_empty(i, j):
						env.board[i, j] = self.sym
						state = env.get_state()
						env.board[i, j] = 0
						pos2val[(i, j)] = self.V[state]
						if self.V[state] > best_value:
							best_value = self.V[state]
							best_state = state
							next_move = (i, j)

			if self.verbose:
				print("Taking a greedy action")
				for i in range(LENGTH):
					print("------------------")
					for j in range(LENGTH):
						if env.is_empty(i, j):
							# print the value
							print(" %.2f|" % pos2val[(i, j)], end="")
						else:
							print("  ", end="")
							if env.board[i, j] == env.x:
								print("x  |", end="")
							elif env.board[i, j] == env.o:
								print("o  |", end="")
							else:
								print("   |", end="")
					print("")
				print("------------------")

		# make the move
		env.board[next_move[0], next_move[1]] = self.sym

	def update_state_history(self, s):
		self.state_history.append(s)

	def update(self, env):
		reward = env.reward(self.sym)
		target = reward
		for prev in reversed(self.state_history):
			value = self.V[prev] + self.alpha*(target - self.V[prev])
			self.V[prev] = value
			target = value
		self.reset_history()

class Human:
	def __inti__(self):
		pass

	def set_symbol(self, sym):
		self.sym = sym

	def take_action(self, env):
		while True:
			# break if we make a legal move
			move = input("Enter co-ordinates of i and j: ")
			i, j = move.split(",")
			i = int(i)
			j = int(j)
			if env.is_empty(i, j):
				env.board[i, j] = self.sym
				break

	def update(self, env):
		pass

	def update_state_history(self, s):
		pass

if __name__ == '__main__':
	# train the agent
	p1 = Agent()
	p2 = Agent()

	# set initial V for p1 and p2
	env = Environment()
	state_winner_triples = get_state_hash_and_winner(env)

	V_x = initialV_x(env, state_winner_triples)
	p1.setV(V_x)
	V_o = initialV_o(env, state_winner_triples)
	p2.setV(V_o)

	# give each player their symbol
	p1.set_symbol(env.x)
	p2.set_symbol(env.o)

	# play_game(p1, p2, Environment())

	T = 10000
	for t in range(T):
		if t % 200 == 0:
			print(t)
		play_game(p1, p2, Environment())

	# Human Verification
	# human_1 = Human()
	# human_1.set_symbol(env.x)
	human_2 = Human()
	human_2.set_symbol(env.o)
	while True:
		p1.set_verbose(True)
		play_game(p1, human_2, Environment(), draw=2)

		answer = input("Play again? [Y/n]: ")
		if answer and answer.lower()[0] == 'n':
			break