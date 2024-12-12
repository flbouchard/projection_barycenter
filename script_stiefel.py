from tqdm import tqdm

import numpy as np
import pandas as pd


from synthetic_data import st_random_center, st_random_sample_from_center
from projection_retraction import st_retr_polar, st_inv_retr_polar, st_retr_qr, st_inv_retr_qr, st_retr_orthographic, st_inv_retr_orthographic

from barycenter import st_projected_arithmetic_mean_polar, st_projected_arithmetic_mean_qr, R_barycenter

from error_measures import dist2_eucl, err_st


if __name__ == '__main__':
    # parameters
    p = 10
    k = 5

    nMC = 100

    n_all = [20,50,70,100,200,500]

    scale_all = [0.3, 0.4, 0.5]

    # random Stiefel mean
    G, _ = st_random_center(p,k)

    for scale in scale_all:
        # error_eucl = np.zeros((nMC, len(n_all), 5))
        error_st = np.zeros((nMC, len(n_all), 5))
        similarity_polar_ortho_st = np.zeros((nMC,len(n_all)))
        for it in range(nMC):
            print(f"scale={scale} in {scale_all} \t it={it+1}/{nMC}")
            # generate all data for this Monte Carlo iteration
            U_all = np.zeros((np.max(n_all),p,k))
            for i in range(np.max(n_all)):
                U_all[i] = st_random_sample_from_center(G, scale)
            
            for n_counter, n in enumerate(tqdm(n_all)):
                # select right amount of data
                U = U_all[:n]

                # compute projected means and errors
                R_proj_polar = st_projected_arithmetic_mean_polar(U)
                # error_eucl[it,n_counter,0] = dist2_eucl(G,R_proj_polar)
                error_st[it,n_counter,0] = err_st(G,R_proj_polar)
               
                R_proj_qr = st_projected_arithmetic_mean_qr(U)
                # error_eucl[it,n_counter,1] = dist2_eucl(G,R_proj_qr)
                error_st[it,n_counter,1] = err_st(G,R_proj_qr)
                
                # compute R barycenters
                init = U[0]

                R_r_ortho, _ = R_barycenter(U, st_retr_orthographic, st_inv_retr_orthographic,init,verbosity=False)
                # error_eucl[it,n_counter,2] = dist2_eucl(G,R_r_ortho)
                error_st[it,n_counter,2] = err_st(G,R_r_ortho)

                R_r_polar, _ = R_barycenter(U, st_retr_polar, st_inv_retr_polar,init,verbosity=False)
                # error_eucl[it,n_counter,3] = dist2_eucl(G,R_r_polar)
                error_st[it,n_counter,3] = err_st(G,R_r_polar)

                R_r_qr, _ = R_barycenter(U, st_retr_qr, st_inv_retr_qr,init,verbosity=False)
                # error_eucl[it,n_counter,4] = dist2_eucl(G,R_r_qr)
                error_st[it,n_counter,4] = err_st(G,R_r_qr)

                # compute similarity between R_proj_polar and R_r_ortho
                similarity_polar_ortho_st[it,n_counter]=err_st(R_proj_polar,R_r_ortho)

        # compute medians, 10% and 90% quantiles
        error_st_median = np.median(error_st, axis=0)
        error_st_q10 = np.quantile(error_st,q=0.1,axis=0)
        error_st_q90 = np.quantile(error_st,q=0.9,axis=0)

        similarity_polar_ortho_st_median = np.median(similarity_polar_ortho_st, axis=0)
        similarity_polar_ortho_st_q10 = np.quantile(similarity_polar_ortho_st,q=0.1,axis=0)
        similarity_polar_ortho_st_q90 = np.quantile(similarity_polar_ortho_st,q=0.9,axis=0)

        # write files
        filename = f"./error_st_p{p}_k{k}_scale{int(scale*10):02d}_nMC{nMC}.csv"
        columns = ['proj_polar','proj_qr','R_ortho','R_polar','R_qr']
        content_median = pd.DataFrame(data=error_st_median, index=n_all, columns=columns)
        content_q10 = pd.DataFrame(data=error_st_q10, index=n_all, columns=columns)
        content_q90 = pd.DataFrame(data=error_st_q90, index=n_all, columns=columns)
        content_full = pd.concat([content_median.add_suffix('_median'),content_q10.add_suffix('_q10'),content_q90.add_suffix('_q90')],axis=1)
        content_full.index.name = 'n'
        content_full.to_csv(filename,index=True)

        filename = f"./similarity_polar_ortho_st_p{p}_k{k}_scale{int(scale*10):02d}_nMC{nMC}.csv"
        column = ['similarity_polar_ortho']
        content_median = pd.DataFrame(data=similarity_polar_ortho_st_median, index=n_all, columns=column)
        content_q10 = pd.DataFrame(data=similarity_polar_ortho_st_q10, index=n_all, columns=column)
        content_q90 = pd.DataFrame(data=similarity_polar_ortho_st_q90, index=n_all, columns=column)
        content_full = pd.concat([content_median.add_suffix('_median'),content_q10.add_suffix('_q10'),content_q90.add_suffix('_q90')],axis=1)
        content_full.index.name = 'n'
        content_full.to_csv(filename,index=True)