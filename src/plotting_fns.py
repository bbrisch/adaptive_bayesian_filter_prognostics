import matplotlib.pyplot as plt
import numpy as np

def quantile(x_inp,w_inp ,q):
    return [np.quantile(x, q, weights=w, method='inverted_cdf') for x, w in zip(x_inp,w_inp)]


def get_bounds(x, w, alpha):
    expected = (x*w).sum(1)
    lb = quantile(x,w, alpha/2   )
    ub = quantile(x,w, 1-alpha/2 )
    return expected, lb, ub

def correct_xgb_preds(preds, target_len):
    
    d_len = target_len - len(preds)
    
    preds_ = np.ones(target_len)*preds[0]
    preds_[d_len:] = preds
    return preds_




def plot_rul(gt_rul_list, results_list, alpha = 0.05):
    # plt.title(mse)
    
    fig, ax = plt.subplots(len(gt_rul_list))

    for i, (results, gt_rul) in enumerate(zip(results_list, gt_rul_list)):
    
        expected = (results.get_prognostic()*results.get_w()).sum(1)
        lb = quantile(results.get_prognostic(), results.get_w(), alpha/2   )
        ub = quantile(results.get_prognostic(), results.get_w(), 1-alpha/2 )
        
        x = np.arange(len(gt_rul))
        
        if len(gt_rul_list) == 1:
            ax.plot(gt_rul, color = 'r')
            ax.plot(expected, color = 'b')
            ax.fill_between(x, lb, ub, color = 'b', alpha = 0.2)
            ax.set_ylim([0,max(gt_rul)*2])
        else:
            ax[i].plot(gt_rul, color = 'r')
            ax[i].plot(expected, color = 'b')
            ax[i].fill_between(x, lb, ub, color = 'b', alpha = 0.2)
            ax[i].set_ylim([0,max(gt_rul)*2])
    plt.tight_layout()
    
def plot_hi(gt_x_list, results_list, alpha = 0.05):
    # plt.title(mse)
    
    fig, ax = plt.subplots(len(gt_x_list))
    
    for i, (results, gt_rul) in enumerate(zip(results_list, gt_x_list)):
         
        
        expected = (results.get_x()*results.get_w()).sum(1)
        lb = quantile(results.get_x(), results.get_w(), alpha/2  )
        ub = quantile(results.get_x(), results.get_w(), 1-alpha/2)
        
        x = np.arange(len(gt_rul))
        
        if len(gt_x_list) == 1:
            ax.plot(gt_rul, color = 'r')
            ax.plot(expected, color = 'b')
            ax.fill_between(x, lb, ub, color = 'b', alpha = 0.2)
        
        else:
            ax[i].plot(gt_rul, color = 'r')
            ax[i].plot(expected, color = 'b')
            ax[i].fill_between(x, lb, ub, color = 'b', alpha = 0.2)
    plt.tight_layout()
        
def plot_param(gt_param_list, results_list, alpha = 0.05):
    # plt.title(mse)
    
    fig, ax = plt.subplots(len(gt_param_list))
    
    for i, (results, gt_param) in enumerate(zip(results_list, gt_param_list)):
    
        expected = (results.get_p()*results.get_w()).sum(1)
        lb = quantile(results.get_p(), results.get_w(), alpha/2  )
        ub = quantile(results.get_p(), results.get_w(), 1-alpha/2)
        
        x = np.arange(len(expected))
        
        if len(gt_param_list) == 1:
            ax.axhline(gt_param, color = 'r')
            ax.plot(expected, color = 'b')
            ax.fill_between(x, lb, ub, color = 'b', alpha = 0.2)
            ax.set_ylim([gt_param*0.5, gt_param*1.5])
        
        else:
            ax[i].axhline(gt_param, color = 'r')
            ax[i].plot(expected, color = 'b')
            ax[i].fill_between(x, lb, ub, color = 'b', alpha = 0.2)
            # ax[i].set_ylim([1e-4,1e-2])
            # ax.set_ylim([gt_param*0.5, gt_param*1.5])
    plt.tight_layout()