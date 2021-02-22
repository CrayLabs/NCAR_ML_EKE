import time
import numpy as np

def retrieve_all_ranks(client, key, nranks):
    ''' Retrieve a double-precision array from the database by its key
    '''
    retrieve_d = {}
    # Loop over all processors to retrieve each subdomain
    for rank in range(nranks):
        # MOM6 was setup so that keys sent from each processor would suffix its zero-padded MPI rank
        rank_id= f'{rank:06d}'        
        retrieve_d[rank_id] = client.get_tensor(f'{key}_{rank_id}')
    return retrieve_d

def reconstruct_domain(client, key, nranks, timestamp = None, halo_size=3, print_time_elapsed = False):
    """Reconstruct the domain of MOM6 by rank at the specified timestep of the model
    """
    start_iter = time.time()
    # Retrieve the array metadata needed to reconstruct a global array from each subdomain
    meta_rank = retrieve_all_ranks(client, "meta" ,nranks)
    t1 = time.time()
    # Retrieve layer thicknesses from each subdomain
    if timestamp:
        h_rank = retrieve_all_ranks(client, f'{key}_{timestamp}',nranks)
    else:
        h_rank = retrieve_all_ranks(client, f'{key}',nranks)
    # Initialize the global array
    h_glob = np.zeros([1440,1080])*np.nan
    # Loop over all the rtrieved subdomains and reconstruct the global array
    for rank in range(nranks):
        rank_id = f'{rank:06d}'
        ni, nj = h_rank[rank_id].shape
        si = int(meta_rank[rank_id][0]+halo_size)
        ei = si + ni - 2*halo_size + 1
        sj = int(meta_rank[rank_id][2]+halo_size)+halo_size
        ej = sj + nj - 2*halo_size + 1
        try:      
            h_glob[si:ei,sj:ej] = h_rank[f'{rank:06d}'][3:-2,3:-2] 
        except:
            pass
    iter_time = time.time() - start_iter
    if print_time_elapsed:
        print(f"Time elapsed in iteration: {iter_time}")
    return h_glob
