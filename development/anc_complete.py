import numpy as np
import time

def anc_complete(model, G, F, order, forget, delta=1e-7, weight_history=False, force_hermitian=False, p_normalization=True):
    """ Active Noise Cancelling
        Apply the active noise cancelling  RLS-based algorithm to compensate the
        noise by modeling the primary acoustic path and compensating the 
        secondary and feedback acoustic paths.
        
        @param model Instance of an acoustic model for simulation
        @param G Secondary acoustic path
        @param F Feedback acoustic path 
        @param order Order of the filter
        @param forget Forgetting factor used in RLS
        @param delta  P initialization parameter
        @param weight_history Enable recording the weight evolution throughout the simulation
        @param force_hermitian Forces P(n) to be an hermitian matrix
        @return Tuple containing error and coefficients of each iteration of the LMS algorithm
                (error_signal, coefficients_evolution)
    """    
    # Validate metaparameters
    if type(order) is int:
        if order < 1:
            raise ValueError("The minimum order is 1")
    else:
        raise ValueError("Order argument must be integer")
    if forget <= 0 or forget > 1:
        raise ValueError("The forgetting factor must be a positive value less or equal than 1")
        
    # Parameters
    N = len(model)
    
    # Variables for analysis
    propagation_duration_estimated = 0
    algorithm_duration_estimated = 0
    update_duration_estimated = 0
    
    # Variable initialization for the algorithm
    rg = np.zeros((len(G), 1))      # Buffer for the input of G(z)
    rw = np.zeros((order, 1))       # Buffer for the input of W(z)
    r_rls = np.zeros((order, 1))    # Buffer for the input of the RLS update equation
    y = np.zeros(len(F))            # Buffer for the input of F(z)
    w = np.zeros((order, 1))
    e_n = np.zeros(N)
    P = np.eye(order) / delta
    P_norm = 1

    if weight_history:
        w_n = np.zeros((order, N))
    i = 0
    
    # Sample processing loop
    while (model.step()):
        propagation_start = time.time()
        # Propagating new samples of input, output and error signals
        rg = np.roll(rg, 1)
        rw = np.roll(rw, 1)
        r_rls = np.roll(r_rls, 1)
        rg[0] = rw[0] = model.reference_microphone()
        feedback = np.dot(F, y)
        rg[0] -= feedback
        rw[0] -= feedback
        y = np.roll(y, 1)
        y[0] = -np.dot(w.T, rw)
        model.speaker(y[0])
        e = model.error_microphone()        
        r_rls[0] = np.dot(G, rg)
        propagation_end = time.time()
        propagation_duration = propagation_end - propagation_start

        algorithm_start = time.time()
        # Algorithm core computation        
        if e > 1:                                      # Re-initialise the algorithm if the conditions 
            P = np.eye(order) / delta                  # have changed so much that the error explodes
        lambda_eff = forget * P_norm                   # Compute the effective forget factor used in the algorithm
        if lambda_eff > 1.0:                           # and restrict its value to be always less than 1.0 (to avoid instability)
            lambda_eff = 1.0
        g_bar = (1 / lambda_eff) * np.dot(P, r_rls)     
        g = g_bar / (1 + np.dot(g_bar.T, r_rls))
        P = (1 / lambda_eff) * P - np.dot(g, g_bar.T)
        if p_normalization:
            P_norm = np.linalg.norm(P)                 # Tracking P matrix's norm
        if force_hermitian:
            P = (P + P.T) / 2                          # Force P matrix to be hermitian (to avoid instability)
        algorithm_end = time.time()
        algorithm_duration = algorithm_end - algorithm_start
        
        update_start = time.time()
        # Coefficient updating
        w += g * e        
        e_n[i] = e
        if weight_history:
            w_n[:,i] = w
        i += 1
        update_end = time.time()
        update_duration = update_end - update_start
        
        # Estimating duration of stages
        propagation_duration_estimated = propagation_duration_estimated * (i - 1) / i + propagation_duration / i
        algorithm_duration_estimated = algorithm_duration_estimated * (i - 1) / i + algorithm_duration / i
        update_duration_estimated = update_duration_estimated * (i - 1) / i + update_duration / i
    
    # Print timing analysis
    total_duration = propagation_duration_estimated + algorithm_duration_estimated + update_duration_estimated
    print(f'Took {total_duration} seconds')
    print(f'Propagation: {round((propagation_duration_estimated / total_duration) * 100, 3)} %')
    print(f'Algorithm: {round((algorithm_duration_estimated / total_duration) * 100, 3)} %')
    print(f'Update: {round((update_duration_estimated / total_duration) * 100, 3)} %')
    
    if weight_history:
        return e_n, w, w_n
    else:
        return e_n, w