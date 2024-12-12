from tqdm import tqdm

import numpy as np
import pandas as pd


from synthetic_data import st_random_center, st_random_sample_from_center, st_to_gr
from projection_retraction import gr_exp, gr_log

from barycenter import gr_projected_arithmetic_mean_evd, R_barycenter

from error_measures import dist2_eucl, dist2_gr


if __name__ == '__main__':
    # parameters
    p = 10
    k = 5

    nMC = 100

    n_all = [20,50,70,100,200,500]

    scale = 0.5

    # random Grassmann mean
    G_st, _ = st_random_center(p,k)
    G = st_to_gr(G_st)

    error_gr = np.zeros((nMC, len(n_all), 2))

    for it in range(nMC):
        print(f"it={it+1}/{nMC}")
        # generate all data for this Monte Carlo iteration
        P_all = np.zeros((np.max(n_all),p,p))
        for i in range(np.max(n_all)):
            P_all[i] = st_to_gr(st_random_sample_from_center(G_st, scale))
        
        for n_counter, n in enumerate(tqdm(n_all)):
            # select right amount of data
            P = P_all[:n]

            # compute means and errors
            R_proj_evd = gr_projected_arithmetic_mean_evd(P,k)
            error_gr[it,n_counter,0] = dist2_gr(G,R_proj_evd)

            init = P[0]
            R_riem_mean, _ = R_barycenter(P, gr_exp, gr_log, init, verbosity=False)
            error_gr[it,n_counter,1] = dist2_gr(G,R_riem_mean)

    # compute medians, 10% and 90% quantiles
    error_gr_median = np.median(error_gr, axis=0)
    error_gr_q10 = np.quantile(error_gr,q=0.1,axis=0)
    error_gr_q90 = np.quantile(error_gr,q=0.9,axis=0)

    # write file
    filename = f"./error_gr_p{p}_k{k}_scale{int(scale*10):02d}_nMC{nMC}.csv"
    columns = ['proj_evd','Riem_mean']
    content_median = pd.DataFrame(data=error_gr_median, index=n_all, columns=columns)
    content_q10 = pd.DataFrame(data=error_gr_q10, index=n_all, columns=columns)
    content_q90 = pd.DataFrame(data=error_gr_q90, index=n_all, columns=columns)
    content_full = pd.concat([content_median.add_suffix('_median'),content_q10.add_suffix('_q10'),content_q90.add_suffix('_q90')],axis=1)
    content_full.index.name = 'n'
    content_full.to_csv(filename,index=True)