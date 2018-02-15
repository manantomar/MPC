import tensorflow as tf
import numpy as np

# Predefined function to build a feedforward neural network
def build_mlp(input_placeholder,
              output_size,
              scope,
              n_layers=2,
              size=500,
              activation=tf.nn.relu,
              output_activation=None
              ):
    out = input_placeholder
    with tf.variable_scope(scope):
        for _ in range(n_layers):
            out = tf.layers.dense(out, size, activation=activation)
        out = tf.layers.dense(out, output_size, activation=output_activation)
    return out

class NNDynamicsModel():
    def __init__(self,
                 env,
                 n_layers,
                 size,
                 activation,
                 output_activation,
                 normalization,
                 batch_size,
                 iterations,
                 learning_rate,
                 sess
                 ):
        """ YOUR CODE HERE """
        """ Note: Be careful about normalization """

        self.normalization = normalization
        self.batch_size = batch_size
        self.iterations = iterations

        self.states = tf.placeholder(shape = [None, env.observation_space.shape[0]], dtype = tf.float32)
        self.actions = tf.placeholder(shape = [None, env.action_space.shape[0]], dtype = tf.float32)
        self.deltas = tf.placeholder(shape = [None, env.observation_space.shape[0]], dtype = tf.float32)
        state_action_pair = tf.concat([self.states, self.actions], 1)

        self.model = build_mlp(state_action_pair, env.observation_space.shape[0], "model", n_layers, size, activation, output_activation)

        self.loss = tf.reduce_mean(tf.square((self.deltas) - self.model))
        self.update_op = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)

    def fit(self, data):
        """
        Write a function to take in a dataset of (unnormalized)states, (unnormalized)actions, (unnormalized)next_states and fit the dynamics model going from normalized states, normalized actions to normalized state differences (s_t+1 - s_t)
        """

        """YOUR CODE HERE """
        observations = np.concatenate([path['observations'] for path in data])
        actions = np.concatenate([path['actions'] for path in data])
        next_observations = np.concatenate([path['next_observations'] for path in data])
        deltas = next_observations - observations

        "Normalize the data"
        observations = (observations - self.normalization[0]) / (self.normalization[1] + 1e-10)
        actions = (actions - self.normalization[4]) / (self.normalization[5] + 1e-10)
        deltas = (deltas - self.normalization[2]) / (self.normalization[3] + 1e-10)

        for i in range(self.iterations):

            batch_id = np.random.choice(observations.shape[0], self.batch_size)#, replace = False)
            _ = tf.get_default_session().run(self.update_op, feed_dict = {self.states : observations[batch_id], self.actions : actions[batch_id], self.deltas : deltas[batch_id]})


    def predict(self, states, actions):
        """ Write a function to take in a batch of (unnormalized) states and (unnormalized) actions and return the (unnormalized) next states as predicted by using the model """
        """ YOUR CODE HERE """

        if len(states.shape) == 1:
            states = states.reshape((1, states.shape[0]))
        if len(actions.shape) == 1:
            actions = actions.reshape((1, actions.shape[0]))
            
        next_observations = states + tf.get_default_session().run(self.model, feed_dict = {self.states : states, self.actions : actions})
        return next_observations
