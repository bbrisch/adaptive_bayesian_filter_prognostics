import os
import shutil
import optuna
import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import gaussian_kde

from .utils import Dataloader, SIR_PF, ResultsBuffer, mse_loss_full, crps_loss_full


# Obs eq.
def poli_fn(hi, a, b, c, d, e):
    return e * hi**5 + d * hi**4 + a * hi**3 + b * hi**2 + c * hi


def initialize_data(dataloader_conf):

    # Instance the dataloader
    dataloader = Dataloader(**dataloader_conf)

    # Adjust the values of the measurement eq.
    data_train = []
    hi_train = []
    lengths = []
    damage_rate = []
    for f in dataloader.ids:
        data = dataloader.load_id(f)

        lengths.append(len(data))
        damage_rate.append(1 / len(data))

        data_train.extend(data)
        hi_train.extend(np.linspace(0, 1, len(data)))

    msg = curve_fit(
        poli_fn, hi_train, data_train, p0=(1, 1, 1, 1, 1), full_output=False
    )
    popt = msg[0]

    # get noise level and damage accumulation rates
    errors = []
    for y, h in zip(data_train, hi_train):
        y_ = poli_fn(h, *popt)
        errors.append(y - y_)
    obs_std = np.std(errors)

    return dataloader, obs_std, popt, damage_rate


def sample_parameters(config, trial: optuna.trial.Trial):
    param_dict = dict()
    if config["particle_filter"]["sample"]:
        for name, values in config["particle_filter"]["search_space"].items():
            if values[-1] == "float":
                p = trial.suggest_float(name, values[0], values[1], log=values[2])
                param_dict.update({name: p})

            elif values[-1] == "int":
                p = trial.suggest_int(name, values[0], values[1], log=values[2])
                param_dict.update({name: p})

            else:
                assert (
                    False
                ), f"Param type not recognized\nname: {name}\nparams: {values}"
    else:
        for name, values in config["particle_filter"]["manual_params"].items():
            param_dict.update({name: values})

    return param_dict


def initialize_filter(config, param_dict):

    # 1. get the data and observation parameters with damage rates
    dataloader, obs_std, popt, damage_rate = initialize_data(
        config["dataloader_config"]
    )

    # 2. Initialize the PF
    obs_eq = lambda x: poli_fn(x, *popt)
    state_eq = lambda x, p: np.clip(
        x
        + p
        + np.random.normal(
            0, param_dict["w_state"], config["particle_filter"]["n_particles"]
        ),
        0,
        2,
    )
    param_eq = lambda p: np.clip(
        p
        + np.random.normal(
            0, param_dict["w_param"], config["particle_filter"]["n_particles"]
        ),
        1e-12,
        np.inf,
    )
    backprop_eq = lambda x, p, order: x - p * order

    x_0 = np.zeros(config["particle_filter"]["n_particles"])
    p_0 = np.random.uniform(
        min(damage_rate) * 0.1,
        max(damage_rate) * 1.5,
        config["particle_filter"]["n_particles"],
    )
    # p_0 = np.random.choice(damage_rate, config['particle_filter']['n_particles'])
    # p_0 = np.clip(gaussian_kde(damage_rate ).resample(config['particle_filter']['n_particles']),min(damage_rate)*0.1, max(damage_rate)*1.5).ravel()

    particle_filter = SIR_PF(
        obs_eq,
        state_eq,
        param_eq,
        backprop_eq,
        config["particle_filter"]["n_particles"],
        param_dict["obs_std"],
        param_dict["resampling_thresh"],
        param_dict["smoothing_order"],
    )
    return particle_filter, dataloader, (x_0, p_0)


def compute_predictions(
    particle_filter: SIR_PF, dataloader: Dataloader, initial_state, n_samples_loss=1
):

    gt_x_list = []
    gt_param_list = []
    rul_gt_list = []
    results_list = []
    # print(min(initial_state[1]), max(initial_state[1]))
    # print(dataloader.ids)

    for id in dataloader.ids:
        for _ in range(n_samples_loss):
            # print(f'Processing id {id}')
            particle_filter.init_state(*initial_state)
            data_test = dataloader.load_id(id)

            gt_x = np.linspace(0, 1, len(data_test))
            rul_gt = len(data_test) - np.arange(len(data_test))
            gt_param = 1 / len(data_test)

            results = ResultsBuffer()

            for i, y in enumerate(data_test):

                # particle_filter.update(y)

                particle_filter.update_smooting(y)

                # particle_filter.y_buffer.append(y)
                results.append_x(particle_filter.x)
                results.append_p(particle_filter.param)
                results.append_w(particle_filter.w)
                results.append_prognostic(particle_filter.prognostic_constant_param())

            gt_x_list.append(gt_x)
            gt_param_list.append(gt_param)
            rul_gt_list.append(rul_gt)
            results_list.append(results)

        # break

    return gt_x_list, gt_param_list, rul_gt_list, results_list


def compute_loss(gt_x, gt_param, rul_gt, results: ResultsBuffer):

    # loss = mse_loss_full(gt_x,gt_param,rul_gt,results )
    loss = crps_loss_full(gt_x, gt_param, rul_gt, results)

    return loss


class Objective:
    def __init__(self, config):
        self.config = config
        self.config["dataloader_config"]["path_data"] = os.path.join(
            config["dataloader_config"]["path_data"], "train"
        )
        pass

    def __call__(self, trial):

        #  get the hyper parameters from the trial object
        param_dict = sample_parameters(self.config, trial)

        # Initialization
        particle_filter, dataloader, initial_state = initialize_filter(
            self.config, param_dict
        )

        # Compute predictions
        gt_x, gt_param, rul_gt, results = compute_predictions(
            particle_filter,
            dataloader,
            initial_state,
            n_samples_loss=self.config["particle_filter"]["n_samples_loss"],
        )

        # Compute loss function
        loss_diag, loss_param, loss_prog = compute_loss(gt_x, gt_param, rul_gt, results)
        # total_loss = loss_diag+loss_prog+loss_param*1e4
        total_loss = loss_diag
        print("\n", loss_diag, loss_param, loss_prog)
        # total_loss = loss_diag+loss_param+loss_prog

        # if 1e2 <total_loss:
        #     # print(loss_diag,loss_prog,loss_param*1e4)
        #     assert False, f'\nLoss exploded vith values {loss_diag}, {loss_prog}, and {loss_param*1e4}.\nTotal loss: {total_loss}'

        # return np.clip(total_loss, 0,5)
        return total_loss


class ValidationObj:
    def __init__(self, config, fold="train"):
        self.config = config

        self.config["dataloader_config"]["path_data"] = os.path.join(
            config["dataloader_config"]["path_data"], fold
        )

        pass

    def __call__(self, opt_params):
        # Initialization
        particle_filter, dataloader, initial_state = initialize_filter(
            self.config, opt_params
        )

        # Compute predictions
        gt_x, gt_param, rul_gt, results = compute_predictions(
            particle_filter, dataloader, initial_state
        )

        return gt_x, gt_param, rul_gt, results


def optimize_params(config):
    pruner = optuna.pruners.MedianPruner()
    study = optuna.create_study(
        pruner=pruner, sampler=optuna.samplers.TPESampler(n_startup_trials=30)
    )
    objective = Objective(config)

    study.optimize(
        objective,
        n_trials=100,
    )
    return study
