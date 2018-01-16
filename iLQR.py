import numpy as np


Class iLQR:

    def __init__(self, ):



    def forward(self, state, K, k):

        u = np.multiply(K, state) + k

    return u[0]

    def backward(self, x_seq, u_seq):

        "initialize F_t, C_t, f_t, c_t, V_t, v_t"

        F_t, f_t, C_t, c_t = finite_difference(x_seq, u_seq)

        "loop till horizon"
        for t in range(T):

            "update Q"

            Q_t = C_t + F_t.T * V_t1 * F_t
            q_t = c_t + F_t.T * V_t1 * f_t + F_t.T * v_t1

            "differentiate Q to get Q_uu, Q_xx, Q_ux, Q_u, Q_x"

            q_x = c_x + f_x * v_x
            q_u = c_u + f_u * v_x

            Q_xx = C_xx + F_x.T * V_xx * f_x + v_x * F_xx

            "update K, k, V, v"

            K = -inverse(Q_uu) * Q_ux
            k = -inverse(Q_uu) * q_u

            V = Q_xx + Q_xu * K + K.T * Q_ux + K.T * Q_uu * K
            v = q_x + Q_xu * k + K.T * q_u + K.T * Q_uu * k

    def differentiate(self,):

        "get gradient values using finite difference"


    def finite_difference(self, x, u, delta):

        "calling finite difference for delta perturbation"

        x_pred = x.copy + delta
        grad_x = (self.simulate_step(x_pred, u) - self.simulate_step(x_pred, u)) / 2 * delta

        u_pred = u.copy + delta
        grad_u = (self.simulate_step(x, u_pred) - self.simulate_step(x, u_pred)) / 2 * delta

        return [grad_x, grad_u]

    def simulate_step(self, env, state, action):

        next_state, _, _, _ = env.step(action)
        delta_state = next_state - state

        "get cost"
        C = None

        return next_state, C
