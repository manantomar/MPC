import numpy as np
from numpy.linalg import inv
from scipy.optimize import approx_fprime

def approx_fdoubleprime(x, i):
    return approx_fprime(x, simulate_step[1], self.delta)[i]

class iLQR():

    def __init__(self, ):

        self.delta = delta
        self.T = T

    def simulate_step(self, x):

        next_x, _, _, _ = self.env.step(x[1])
        delta_x = next_x - x[0]

        "get cost"
        cost = get_cost(x[0], x[1])

        return next_x, cost

    def finite_difference(self, x, u):

        "calling finite difference for delta perturbation"

        F = approx_fprime([x,u], simulate_step[0], self.delta)

        c = approx_fprime([x,u], simulate_step[1], self.delta)

        C = np.zeros((len([x,u]), len([x,u])))

        C[:self.n,:] = approx_fprime([x,u], approx_fdoubleprime, self.delta, 0)
        C[self.n:,:] = approx_fprime([x,u], approx_fdoubleprime, self.delta, 1)

        f = np.zeros((len([x,u])))

        return C, F, c, f

    def differentiate(self, x_seq, u_seq):

        "get gradient values using finite difference"

        C, F, c, f = [], [], [], []

        for t in range(self.T):

            Ct, Ft, ct, ft = finite_difference(x_seq[t], u_seq[t], delta)

            C.append(Ct)
            F.append(Ft)
            c.append(ct)
            f.append(ft)

        "TODO : C, F, c, f for time step T are different. Why ?"
        
        return C, F, c, f

    def backward(self, x_seq, u_seq):

        "initialize F_t, C_t, f_t, c_t, V_t, v_t"

        F, f, C, c = finite_difference(x_seq, u_seq)

        "initialize V_t1 and v_t1"

        c_x = c[-1][:n]
        c_u = c[-1][n:]

        C_xx = C[-1][:n,0]
        C_xu = C[-1][:n,1]
        C_ux = C[-1][n:,0]
        C_uu = C[-1][n:,1]

        K[-1] = -np.dot(inv(C_uu), C_ux)
        k[-1] = -np.dot(inv(C_uu), C_u)

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

            Q_xx = Q[t][:n,0]
            Q_xu = Q[t][:n,1]
            Q_ux = Q[t][n:,0]
            Q_uu = Q[t][n:,1]

            "update K, k, V, v"

            K[t] = -np.dot(inv(Q_uu), Q_ux)
            k[t] = -np.dot(inv(Q_uu), q_u)

            V[t] = Q_xx + np.dot(Q_xu, K[t]) + np.dot(K[t].T, Q_ux) + np.dot(np.dot(K[t].T, Q_uu), K[t])
            v[t] = q_x + np.dot(Q_xu, k[t]) + np.dot(K[t].T, q_u) + np.dot(np.dot(K[t].T, Q_uu), k[t])

        self.K = K
        self.k = k

    def forward(self, state, K, k):

        u = np.dot(K, state) + k

        return u

    def get_action(self, state):

        "TODO : Add delta U's to given action array"

        return  self.forward(state, self.K[0], self.k[0])
