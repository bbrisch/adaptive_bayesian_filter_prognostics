import numpy as np

def mse_loss_single(y_series:np.array, x_series:np.array, w_series:np.array):
    expected = (x_series*w_series).sum(1)
    bias_term = ((y_series-expected)**2).mean()
    variance = (((x_series-expected.reshape(-1,1))**2*w_series).sum(1)).mean()
    # print(bias_term, ' \t', variance)
    return bias_term+variance


def crps_loss_single_energy_score(y_series:np.array, x_series:np.array, w_series:np.array):
    term_1 = (abs(y_series-x_series)*w_series).sum(1)
    
    def sample_subset(x_series, w_series):
        return np.array([np.random.choice(x, 1, p=w) for x,w in zip(x_series, w_series)])
    
    subset_1 = np.stack([sample_subset(x_series, w_series) for _ in range(x_series.shape[1])])
    subset_2 = np.stack([sample_subset(x_series, w_series) for _ in range(x_series.shape[1])])
    term_2 = (abs(subset_1-subset_2)).mean()
    
    return (term_1 - term_2/2).mean()


def crps_loss_single(y_series:np.array, x_series:np.array, w_series:np.array):
    

    l,s = x_series.shape
    
    # Step 1, add the GT and 0 to the gt series and weights
    x_series = np.concatenate([x_series, np.zeros((l,1)), y_series.reshape(-1,1)], axis = 1)
    w_series = np.concatenate([w_series, np.zeros((l,2))], axis = 1)
    
    # Step 2, sort the arrays and generate the CDF
    inds = np.argsort(x_series, axis=1)
    x_series = np.take_along_axis(x_series, inds, axis=1)
    w_series = np.take_along_axis(w_series, inds, axis=1).cumsum(axis = 1)
    
    # Step 3, generate the heaviside function and the dx
    dx = x_series[:, 1:]-x_series[:, :-1]
    h = np.heaviside(x_series-y_series.reshape(-1,1), 1)
    
    # Step 4, integrate over the squared difference
    diff = (h-w_series)**2
    
    return (diff[:,1:]*dx).sum(1).mean()





def mse_loss_full(x_gt_series:list,param_gt_series:list, rul_gt_series: list, results_list:list, scale):
    
    # compute MSE for the diognostic
    mse_diagnostic_list = []
    mse_prognostic_list = []
    mse_param_list = []
    
    for x, param, rul, results in zip(x_gt_series,param_gt_series,rul_gt_series, results_list):
        mse_diagnostic = mse_loss_single(
                                        (x-scale['hi_bias'])/scale['hi_factor'],
                                        (results.get_x()-scale['hi_bias'])/scale['hi_factor'],
                                        results.get_w()                             
                                        )
        mse_param = mse_loss_single(
                                        (np.ones_like(x)*param-scale['p_bias'])/scale['p_factor'],
                                        (results.get_p()-scale['p_bias'])/scale['p_factor'],
                                        results.get_w()                             
                                        )
        
        mse_prognostic = mse_loss_single(
                                        (rul-scale['rul_bias'])/scale['rul_factor'],
                                        (results.get_prognostic()-scale['rul_bias'])/scale['rul_factor'],
                                        results.get_w()                             
                                        )
        
        
        mse_diagnostic_list.append(mse_diagnostic)
        mse_param_list.append(mse_param)
        mse_prognostic_list.append(mse_prognostic)
        # print( rul/max(rul))
        # print(results.get_prognostic()/max(rul))
        # print(results.get_w())
        # print((results.get_prognostic()/max(rul)*results.get_w()).sum(1))
        # print('\n')
        
    return np.mean(mse_diagnostic_list), np.mean(mse_param_list),np.mean(mse_prognostic_list)

def crps_loss_full(x_gt_series:list,param_gt_series:list, rul_gt_series: list, results_list:list, scale):
    
    # compute MSE for the diognostic
    crps_diagnostic_list = []
    crps_prognostic_list = []
    crps_param_list = []
    
    for x, param, rul, results in zip(x_gt_series,param_gt_series,rul_gt_series, results_list):
        crps_diagnostic= crps_loss_single(
                                        (x-scale['hi_bias'])/scale['hi_factor'],
                                        (results.get_x()-scale['hi_bias'])/scale['hi_factor'],
                                        results.get_w()                             
                                        )
        crps_param = crps_loss_single(
                                        (np.ones_like(x)*param-scale['p_bias'])/scale['p_factor'],
                                        (results.get_p()-scale['p_bias'])/scale['p_factor'],
                                        results.get_w()                             
                                        )
        crps_prognostic = crps_loss_single(
                                        (rul-scale['rul_bias'])/scale['rul_factor'],
                                        (results.get_prognostic()-scale['rul_bias'])/scale['rul_factor'],
                                        results.get_w()                             
                                        )
        
        
        crps_diagnostic_list.append(crps_diagnostic)
        crps_prognostic_list.append(crps_prognostic)
        crps_param_list.append(crps_param)
        
    return np.mean(crps_diagnostic_list), np.mean(crps_param_list),np.mean(crps_prognostic_list)
    
