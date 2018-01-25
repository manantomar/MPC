import numpy as np
from cost_functions import trajectory_cost_fn
import time

class Controller():
	def __init__(self):
		pass

	# Get the appropriate action(s) for this state(s)
	def get_action(self, state):
		pass


class RandomController(Controller):
	def __init__(self, env):
		""" YOUR CODE HERE """
		self.env = env

	def get_action(self, state):
		""" YOUR CODE HERE """
		""" Your code should randomly sample an action uniformly from the action space """
		action = self.env.action_space.sample()
		return action


class MPCcontroller(Controller):
	""" Controller built using the MPC method outlined in https://arxiv.org/abs/1708.02596 """
	def __init__(self,
				 env,
				 dyn_model,
				 horizon=5,
				 cost_fn=None,
				 num_simulated_paths=10,
				 ):
		self.env = env
		self.dyn_model = dyn_model
		self.horizon = horizon
		self.cost_fn = cost_fn
		self.num_simulated_paths = num_simulated_paths

	def get_action(self, state):
		""" YOUR CODE HERE """
		""" Note: be careful to batch your simulations through the model for speed """

		# sample sequeces of action_space
		action_set = np.reshape([self.env.action_space.sample() for i in range(self.num_simulated_paths * self.horizon)], (self.num_simulated_paths, self.horizon, self.env.action_space.shape[0]))

		state = np.repeat(state.reshape([1,-1]), self.num_simulated_paths, axis=0)
		cost = np.zeros([self.num_simulated_paths], dtype = np.float32)

		# predict next next_states
		for i in range(self.horizon):
			next_state = self.dyn_model.predict(state, action_set[:,i,:])

			cost += self.cost_fn(state[:,:], action_set[:,i,:], next_state[:,:])

			state = next_state

		# calculate cost and choose optimal path
		act = np.argmin(cost)
		return action_set[act, 0]
