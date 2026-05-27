import numpy as np
from scipy.stats import norm, gaussian_kde
from scipy.special import gamma
from scipy.stats import gaussian_kde


class SIR_PF:
    def __init__(
        self,
        obs_fn: callable,
        state_fn: callable,
        param_fn: callable,
        backprop_eq: callable,
        N_particles,
        sigma_measurements,
        resampling_threshold,
        smoothing_order,
    ):

        self.state_fn = state_fn
        self.obs_fn = obs_fn
        self.param_fn = param_fn
        self.backprop_eq = backprop_eq

        self.N_particles = N_particles
        self.sigma_measurements = sigma_measurements

        self.smoothing_order = smoothing_order
        self.resampling_threshold = resampling_threshold

    def init_state(self, x_init, param_init):

        self.x = x_init
        self.param = param_init
        self.w = np.ones_like(x_init) / self.N_particles
        self.y_buffer = []

    def _gaussian_likelihood(self, y, y_pred):
        return norm.pdf(y - y_pred, self.sigma_measurements)

    def _gaussian_log_likelihood(self, y, x_next):
        return norm.logpdf(y - x_next, self.sigma_measurements)

    def mean(self):
        y = self.obs_fn(self.x)
        return (self.x * self.w).sum(), (self.param * self.w).sum(), (y * self.w).sum()

    # def quantile(self, q):
    #         y = self.obs_fn(self.x)
    #         try:
    #             return np.quantile(self.x, q, weights=self.w, method='inverted_cdf'), np.quantile(self.param, q, weights=self.w, method='inverted_cdf'), np.quantile(y, q, weights=self.w, method='inverted_cdf')
    #         except:
    #             print(self.w)
    #             assert False

    def update(self, y):
        # 1. Propagate particles with state equation
        x_next = self.state_fn(self.x, self.param)
        param_next = self.param_fn(self.param)
        y_pred = self.obs_fn(x_next)

        # 2. Compute the new wheights (assuming p(x_t|x_1:t-1) = q(x_t|x_1:t-1))
        # log_likelihood = self._gaussian_likelihood(y,x_next)
        log_w_next = np.log(self.w) + self._gaussian_log_likelihood(y, y_pred)
        log_w_next_max = log_w_next.max()
        w_next = np.exp(log_w_next - log_w_next_max)
        w_next = w_next / w_next.sum()

        # 3. Resampling to avoid degenerancy
        N_eff = 1 / ((w_next**2).sum())
        N_eff = N_eff / self.N_particles
        # print(N_eff )
        if N_eff < self.resampling_threshold:
            # print('Resampling')
            # _ = input()
            x_next = np.random.choice(x_next, self.N_particles, replace=True, p=w_next)

            param_next = np.random.choice(
                param_next, self.N_particles, replace=True, p=w_next
            )
            w_next = np.ones_like(w_next) / self.N_particles

        # 4. Asign the new state
        self.x = x_next
        self.w = w_next
        self.param = param_next

    def chi2_smoothing(self):

        # Step 1: compute chi2
        chi2 = 0
        l = len(self.y_buffer)
        for i in range(1, self.smoothing_order + 1):

            chi2 += (
                self.y_buffer[l - i]
                - self.obs_fn(self.backprop_eq(self.x, self.param, i))
            ) ** 2 / (self.sigma_measurements**2)

        # Step 2: compute the new wheights
        log_w_new = (
            np.log(chi2) * (self.smoothing_order / 2 - 1)
            - chi2 / 2
            - np.log(
                (2 ** (self.smoothing_order / 2) * gamma(self.smoothing_order / 2))
            )
        )
        w_next = np.exp(log_w_new - log_w_new.max())
        w_next = w_next / w_next.sum()
        self.w = w_next

        self.y_buffer = []

    def update_y_buffer(self, y):
        if len(self.y_buffer) < self.smoothing_order:
            self.y_buffer.append(y)
        else:
            self.y_buffer = self.y_buffer[1:] + [y]

    def update_smooting(self, y):

        # 0. update the y_buffer
        self.update_y_buffer(y)

        # 1. Propagate particles with state equation
        x_next = self.state_fn(self.x, self.param)
        param_next = self.param_fn(self.param)
        # y_pred = self.obs_fn(x_next)

        # 2. Compute the new wheights (assuming p(x_t|x_1:t-1) = q(x_t|x_1:t-1)) and chi2 smooting
        if self.smoothing_order // 2 < len(self.y_buffer):
            # Step 2.1: compute chi2
            chi2 = 0
            l = len(self.y_buffer)
            for i, y_ in enumerate(self.y_buffer):

                chi2 += (
                    y_ - self.obs_fn(self.backprop_eq(x_next, param_next, l - i))
                ) ** 2 / (self.sigma_measurements**2)
                # print(y_, self.obs_fn(self.backprop_eq(x_next, param_next,l-i)), l-i, len(self.y_buffer))

            # Step 2.2: compute the new wheights
            # w_new = (chi2**(M/2-1)*np.exp(-chi2/2))/(2**(M/2)*gamma(M/2))
            log_w_new = (
                np.log(chi2) * (self.smoothing_order / 2 - 1)
                - chi2 / 2
                - np.log(
                    (2 ** (self.smoothing_order / 2) * gamma(self.smoothing_order / 2))
                )
            )
            w_next2 = np.exp(log_w_new - log_w_new.max())
            w_next = w_next2 / w_next2.sum()

            # 3. Resampling to avoid degenerancy
            N_eff = 1 / ((w_next**2).sum())
            N_eff = N_eff / self.N_particles

            if np.isnan(N_eff):

                #     print(chi2)
                #     print(np.log(chi2))
                #     print(log_w_new)
                #     print(w_next2)
                #     print(w_next)

                assert False, "N eff is nan"

            # print(N_eff,self.resampling_threshold )

            if N_eff < self.resampling_threshold:
                # print('Resampling',N_eff,self.resampling_threshold, '\n')

                # _ = input()
                ind_choice = np.random.choice(
                    np.arange(self.N_particles),
                    self.N_particles,
                    replace=True,
                    p=w_next,
                )

                x_next = x_next[ind_choice]
                param_next = param_next[ind_choice]

                # if np.isnan(w_next.ravel()).sum():
                #     assert False, 'W eff is nan'

                # # print(np.isinf(x_next.ravel()).sum(), np.isnan(x_next.ravel()).sum())
                # # print(np.isinf(param_next.ravel()).sum(), np.isnan(param_next.ravel()).sum())
                # # print(np.isinf(w_next.ravel()).sum(), np.isnan(w_next.ravel()).sum(), '\n')

                # x_next = np.clip(gaussian_kde(x_next, weights=w_next.ravel()).resample(self.N_particles).ravel(), 0,1)
                # param_next = np.clip(gaussian_kde(param_next, weights=w_next.ravel()).resample(self.N_particles).ravel(),1e-4, np.inf)

                w_next = np.ones_like(w_next) / self.N_particles

            # assert False
        else:
            w_next = self.w

        # 4. Asign the new state
        self.x = x_next
        self.w = w_next
        self.param = param_next

    def prognostic_MC(self):

        alive_flag = np.ones_like(self.x)
        ttf = np.zeros_like(self.x)

        x = self.x
        p = self.param

        while alive_flag.sum() != 0:

            # Project one step
            x = self.state_fn(x, p)
            # p = self.param_fn(p)

            # Check failure
            fail_cond = 1 <= x

            # Update alive flag
            alive_flag = np.logical_and(
                np.logical_not(fail_cond), alive_flag
            )  # We are alive if we dont fail and we were alive previously

            # Add TTF
            ttf = ttf + alive_flag

        # Get RUL distribution
        # dist = gaussian_kde(ttf, weights=self.w)

        return ttf

    def prognostic_constant_param(self):
        ttf = (np.ones_like(self.x) - self.x) / self.param
        # print(ttf)
        # assert False
        # return ttf
        return np.clip(ttf, 0, 5e3)
