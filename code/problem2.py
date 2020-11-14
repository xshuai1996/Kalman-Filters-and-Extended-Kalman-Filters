import matplotlib.pyplot as plt
import numpy as np


class KalmanFilter():
    """
    Implementation of a Kalman Filter.
    """
    def __init__(self, mu, sigma, A, C, R=0., Q=0.):
        """
        :param mu: prior mean
        :param sigma: prior covariance
        :param A: process model
        :param C: measurement model
        :param R: process noise
        :param Q: measurement noise
        """
        # prior
        self.mu = mu
        self.sigma = sigma
        self.mu_init = mu
        self.sigma_init = sigma
        # process model
        self.A = A
        self.R = R
        # measurement model
        self.C = C
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
        T, D = sensor_data.shape[0], self.sigma.shape[0]
        mus, sigmas = np.zeros((T, D)), np.zeros((T, D, D))
        for t in range(T):
            z = sensor_data[t:t+1]
            self._update(z)
            mus[t] = self.mu.reshape(-1)
            sigmas[t] = self.sigma
        return (mus, sigmas)

    def _predict(self):
        # FILL in your code here
        est_mu = self.A @ self.mu
        est_sigma = self.A @ self.sigma @ self.A.T + self.R
        return est_mu, est_sigma

    def _update(self, z):
        est_mu, est_sigma = self._predict()
        K = est_sigma @ self.C.T @ np.linalg.inv(self.C @ est_sigma @ self.C.T + self.Q)
        self.mu = est_mu + K @ (z - self.C @ est_mu)
        self.sigma = (np.identity(self.A.shape[0]) - K @ self.C) @ est_sigma


def plot_prediction(t, ground_truth, measurement, predict_mean, predict_cov, img_name):
    """
    Plot ground truth vs. predicted value.

    :param t: 1-dimensional array representing timesteps, in seconds.
    :param ground_truth: Tx1 array of ground truth values
    :param measurement: Tx1 array of sensor values
    :param predict_mean: TxD array of mean vectors
    :param predict_cov: TxDxD array of covariance matrices
    """
    predict_pos_mean = predict_mean[:, 0]
    predict_pos_std = predict_cov[:, 0, 0]

    plt.figure()
    plt.plot(t, ground_truth, color='k')
    plt.plot(t, measurement, color='r')
    plt.plot(t, predict_pos_mean, color='g')
    plt.fill_between(
        t,
        predict_pos_mean-predict_pos_std,
        predict_pos_mean+predict_pos_std,
        color='g',
        alpha=0.5)
    plt.legend(("ground truth", "measurements", "predictions"))
    plt.xlabel("time (s)")
    plt.ylabel("position (m)")
    plt.title("Predicted Values")
    plt.show()


def plot_mse(t, ground_truth, predict_means, img_name):
    """
    Plot MSE of your KF over many trials.

    :param t: 1-dimensional array representing timesteps, in seconds.
    :param ground_truth: Tx1 array of ground truth values
    :param predict_means: NxTxD array of T mean vectors over N trials
    """
    predict_pos_means = predict_means[:, :, 0]
    errors = ground_truth.squeeze() - predict_pos_means
    mse = np.mean(errors, axis=0) ** 2

    plt.figure()
    plt.plot(t, mse)
    plt.xlabel("time (s)")
    plt.ylabel("position MSE (m^2)")
    plt.title("Prediction Mean-Squared Error")
    plt.show()


def problem2a():
    # FILL in your code here
    delta_t = 0.1
    A = np.array([[1, 0.1,  0,  0],
                  [0,  1,  0.1, 0],
                  [0,  0,   1, 0.1],
                  [0,  0,   0,  1]])
    C = np.array([[1, 0, 0, 0]])
    mu = np.array([[5, 1, 0, 0]]).T
    sigma = np.array([[10, 0, 0, 0],
                    [0, 10, 0, 0],
                    [0, 0, 10, 0],
                    [0, 0, 0, 10]])
    Q = 1.0
    T = 100
    num_trails = 10000
    t = delta_t * np.arange(1, T + 1).reshape((-1, 1))

    ground_truth = np.sin(0.1 * t)
    KF = KalmanFilter(mu, sigma, A, C, 0, Q)
    sensor_data = ground_truth + np.random.normal(0, Q, ground_truth.shape)
    predict_mean, predict_cov = KF.run(sensor_data)
    plot_prediction(t.reshape(-1), ground_truth, sensor_data, predict_mean,
                    predict_cov, "problem2a_kf_estimation.png")

    predict_means = np.zeros((num_trails, T, A.shape[0]))
    for trail in range(num_trails):
        KF.reset()
        sensor_data = ground_truth + np.random.normal(0, Q, ground_truth.shape)
        predict_mean, _ = KF.run(sensor_data)
        predict_means[trail] = predict_mean
    plot_mse(t.reshape(-1), ground_truth, predict_means, "problem2a_kf_mse.png")


def problem2b():
    # FILL in your code here
    delta_t = 0.1
    A = np.array([[1, 0.1,  0,  0],
                  [0,  1,  0.1, 0],
                  [0,  0,   1, 0.1],
                  [0,  0,   0,  1]])
    C = np.array([[1, 0, 0, 0]])
    mu = np.array([[5, 1, 0, 0]]).T
    sigma = np.array([[10, 0, 0, 0],
                    [0, 10, 0, 0],
                    [0, 0, 10, 0],
                    [0, 0, 0, 10]])
    R = np.array([[0.1, 0, 0, 0],
                  [0, 0.1, 0, 0],
                  [0, 0, 0.1, 0],
                  [0, 0, 0, 0.1]])
    Q = 1.0
    T = 100
    num_trails = 10000
    t = delta_t * np.arange(1, T + 1).reshape((-1, 1))

    ground_truth = np.sin(0.1 * t)
    KF = KalmanFilter(mu, sigma, A, C, R, Q)

    predict_means = np.zeros((num_trails, T, A.shape[0]))
    for trail in range(num_trails):
        KF.reset()
        sensor_data = ground_truth + np.random.normal(0, Q, ground_truth.shape)
        predict_mean, _ = KF.run(sensor_data)
        predict_means[trail] = predict_mean
    plot_mse(t.reshape(-1), ground_truth, predict_means, "problem2b_kf_mse.png")

if __name__ == '__main__':
    problem2a()
    problem2b()
