import matplotlib.pyplot as plt
import numpy as np


class ExtendedKalmanFilter():
    """
    Implementation of an Extended Kalman Filter.
    """
    def __init__(self, mu, sigma, g, g_jac, h, h_jac, R=0., Q=0.):
        """
        :param mu: prior mean
        :param sigma: prior covariance
        :param g: process function
        :param g_jac: process function's jacobian
        :param h: measurement function
        :param h_jac: measurement function's jacobian
        :param R: process noise
        :param Q: measurement noise
        """
        # prior
        self.mu = mu
        self.sigma = sigma
        self.mu_init = mu
        self.sigma_init = sigma
        # process model
        self.g = g
        self.g_jac = g_jac
        self.R = R
        # measurement model
        self.h = h
        self.h_jac = h_jac
        self.Q = Q

    def reset(self):
        """
        Reset belief state to initial value.
        """
        self.mu = self.mu_init
        self.sigma = self.sigma_init

    def run(self, sensor_data):
        """
        Run the Kalman Filter using the given sensor updates.

        :param sensor_data: array of T sensor updates as a TxS array.

        :returns: A tuple of predicted means (as a TxD array) and predicted
                  covariances (as a TxDxD array) representing the KF's belief
                  state AFTER each update/predict cycle, over T timesteps.
        """
        T, D = sensor_data.shape[0], 2
        mus, sigmas = np.zeros((T, D)), np.zeros((T, D, D))
        for t in range(T):
            z = sensor_data[t:t+1]
            self._update(z)
            mus[t] = self.mu.reshape((-1))
            sigmas[t] = self.sigma
        return (mus, sigmas)

    def _predict(self):
        est_mu = self.g(self.mu)
        est_sigma = self.g_jac(self.mu) @ self.sigma @ self.g_jac(self.mu).T + self.R
        return est_mu, est_sigma

    def _update(self, z):
        est_mu, est_sigma = self._predict()
        H = self.h_jac(self.mu)
        K = est_sigma @ H.T @ np.linalg.inv(H @ est_sigma @ H.T + self.Q)
        self.mu = est_mu + K @ (z - self.h(self.mu))
        self.sigma = (np.identity(2) - K @ H) @ est_sigma


def plot_prediction(t, ground_truth, predict_mean, predict_cov, img_name):
    """
    Plot ground truth vs. predicted value.

    :param t: 1-dimensional array representing timesteps, in seconds.
    :param ground_truth: Tx1 array of ground truth values
    :param predict_mean: TxD array of mean vectors
    :param predict_cov: TxDxD array of covariance matrices
    """
    gt_x, gt_a = ground_truth[:, 0], ground_truth[:, 1]
    pred_x, pred_a = predict_mean[:, 0], predict_mean[:, 1]
    pred_x_std = np.sqrt(predict_cov[:, 0, 0])
    pred_a_std = np.sqrt(predict_cov[:, 1, 1])

    plt.figure(figsize=(7, 10))
    plt.subplot(211)
    plt.plot(t, gt_x, color='k')
    plt.plot(t, pred_x, color='g')
    plt.fill_between(
        t,
        pred_x-pred_x_std,
        pred_x+pred_x_std,
        color='g',
        alpha=0.5)
    plt.legend(("ground_truth", "prediction"))
    plt.xlabel("time (s)")
    plt.ylabel(r"$x$")
    plt.title(r"EKF estimation: $x$")

    plt.subplot(212)
    plt.plot(t, gt_a, color='k')
    plt.plot(t, pred_a, color='g')
    plt.fill_between(
        t,
        pred_a-pred_a_std,
        pred_a+pred_a_std,
        color='g',
        alpha=0.5)
    plt.legend(("ground_truth", "prediction"))
    plt.xlabel("time (s)")
    plt.ylabel(r"$\alpha$")
    plt.title(r"EKF estimation: $\alpha$")
    plt.savefig(img_name)
    plt.show()


def problem3():
    Q = 1
    R = np.array([0.5, 0, 0, 0]).reshape((2, 2))
    true_x0 = 2
    true_alpha = 0.1
    T = 20
    mu0 = np.array([1, 2]).reshape((2, 1))
    sigma0 = np.array([2, 0, 0, 2]).reshape((2, 2))

    def g(x_t):
        ret = np.array([x_t[0, 0] * x_t[1, 0], x_t[1, 0]]).reshape((2, 1))
        return ret

    def g_jac(x_t):
        ret = np.array([x_t[1, 0], x_t[0, 0], 0, 1]).reshape((2, 2))
        return ret

    def h(x_t):
        return pow(pow(x_t[0, 0], 2) + 1, 0.5)

    def h_jac(x_t):
        ret = np.array([x_t[0, 0] * pow(pow(x_t[0, 0], 2) + 1, -0.5), 0]).reshape((1, 2))
        return ret

    t = np.arange(0, T).reshape((-1, 1))
    ground_truth = true_x0 * np.power(true_alpha, t)

    EKF = ExtendedKalmanFilter(mu0, sigma0, g, g_jac, h, h_jac, R, Q)
    sensor_data = np.power(np.power(ground_truth, 2) + 1, 0.5) + np.random.normal(0, Q, ground_truth.shape)
    predict_mean, predict_cov = EKF.run(sensor_data[1:])

    predict_mean = np.concatenate((mu0.reshape((1, 2)), predict_mean), axis=0)
    predict_cov = np.concatenate((sigma0.reshape((1, 2, 2)), predict_cov), axis=0)

    new_ground_truth = np.zeros((T, 2))
    new_ground_truth[:, 0:1] = ground_truth
    new_ground_truth[:, 1:2] = true_alpha * np.ones((T, 1))
    print(predict_mean[:, 1])
    plot_prediction(t.reshape((-1)), new_ground_truth, predict_mean, predict_cov, "problem3_ekf_estimation.png")

if __name__ == '__main__':
    problem3()
