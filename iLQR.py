import numpy as np
from scipy.linalg import inv
from scipy.optimize import approx_fprime

class iLQR:

    def __init__(self, env, delta, T, dyn_model, cost_fn):

        self.env = env
        self.delta = delta
        self.T = T
        self.dyn_model = dyn_model
        self.cost_fn = cost_fn

    def simulate_step(self, x):

        xu = [x[:self.env.observation_space.shape[0]], x[self.env.observation_space.shape[0]:]]

        next_x = self.dyn_model.predict(xu[0], xu[1])
        delta_x = next_x - xu[0]

        "get cost"
        cost = self.cost_fn(xu[0], xu[1], next_x[0])

        return next_x[0], cost

    def simulate_next_state(self, x, i):
        return self.simulate_step(x)[0][i]

    def simulate_cost(self, x):
        return self.simulate_step(x)[1]

    def approx_fdoubleprime(self, x, i):
        return approx_fprime(x, self.simulate_cost, self.delta)[i]

    def finite_difference(self, x, u):

        "calling finite difference for delta perturbation"
        xu = np.concatenate((x, u))

        F = np.zeros((x.shape[0], xu.shape[0]))

        for i in range(x.shape[0]):
            F[i,:] = approx_fprime(xu, self.simulate_next_state, self.delta, i)

        c = approx_fprime(xu, self.simulate_cost, self.delta)

        C = np.zeros((len(xu), len(xu)))

        for i in range(xu.shape[0]):
            C[i,:] = approx_fprime(xu, self.approx_fdoubleprime, self.delta, i)

        f = np.zeros((len(x)))

        return C, F, c, f

    def differentiate(self, x_seq, u_seq):

        "get gradient values using finite difference"

        C, F, c, f = [], [], [], []

        for t in range(self.T - 1):

            Ct, Ft, ct, ft = self.finite_difference(x_seq[t], u_seq[t])

            C.append(Ct)
            F.append(Ft)
            c.append(ct)
            f.append(ft)

        "TODO : C, F, c, f for time step T are different. Why ?"

        u = np.zeros((u_seq[0].shape))

        Ct, Ft, ct, ft = self.finite_difference(x_seq[-1], u)

        C.append(Ct)
        F.append(Ft)
        c.append(ct)
        f.append(ft)

        return C, F, c, f

    def backward(self, x_seq, u_seq):

        "initialize F_t, C_t, f_t, c_t, V_t, v_t"

        C, F, c, f = self.differentiate(x_seq, u_seq)

        n = x_seq[0].shape[0]

        "initialize V_t1 and v_t1"

        c_x = c[-1][:n]
        c_u = c[-1][n:]

        C_xx = C[-1][:n,:n]
        C_xu = C[-1][:n,n:]
        C_ux = C[-1][n:,:n]
        C_uu = C[-1][n:,n:]
        print(C_uu)
        K = np.zeros((self.T+1, u_seq[0].shape[0], x_seq[0].shape[0]))
        k = np.zeros((self.T+1, u_seq[0].shape[0]))

        V = np.zeros((self.T+1, x_seq[0].shape[0], x_seq[0].shape[0]))
        v = np.zeros((self.T+1, x_seq[0].shape[0]))

        K[-1] = -np.dot(inv(C_uu), C_ux)
        k[-1] = -np.dot(inv(C_uu), c_u)

        V[-1] = C_xx + np.dot(C_xu, K[-1]) + np.dot(K[-1].T, C_ux) + np.dot(np.dot(K[-1].T, C_uu), K[-1])
        v[-1] = c_x + np.dot(C_xu, k[-1]) + np.dot(K[-1].T, c_u) + np.dot(np.dot(K[-1].T, C_uu), k[-1])

        "initialize Q_t1 and q_t1"

        Q = list(np.zeros((self.T)))
        q = list(np.zeros((self.T)))

        "loop till horizon"
        for t in range(self.T-1, -1, -1):

            "update Q"

            Q[t] = C[t] + np.dot(np.dot(F[t].T, V[t+1]), F[t])
            q[t] = c[t] + np.dot(np.dot(F[t].T, V[t+1]), f[t]) + np.dot(F[t].T, v[t+1])

            "differentiate Q to get Q_uu, Q_xx, Q_ux, Q_u, Q_x"

            q_x = q[t][:n]
            q_u = q[t][n:]

            Q_xx = Q[t][:n,:n]
            Q_xu = Q[t][:n,n:]
            Q_ux = Q[t][n:,:n]
            Q_uu = Q[t][n:,n:]

            "update K, k, V, v"

            K[t] = -np.dot(inv(Q_uu), Q_ux)
            k[t] = -np.dot(inv(Q_uu), q_u)

            V[t] = Q_xx + np.dot(Q_xu, K[t]) + np.dot(K[t].T, Q_ux) + np.dot(np.dot(K[t].T, Q_uu), K[t])
            v[t] = q_x + np.dot(Q_xu, k[t]) + np.dot(K[t].T, q_u) + np.dot(np.dot(K[t].T, Q_uu), k[t])

        self.K = K
        self.k = k
        self.std = inv(Q_uu)

    def get_action_one_step(self, state, t, x, u):

        "TODO : Add delta U's to given action array"

        mean = np.dot(self.K[t], (state - x)) + self.k[t] + u
        return np.random.normal(mean, 1)
