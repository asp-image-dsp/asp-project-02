import numpy as np

def anc_primary_rls(model, order, forget, delta, weight_history=False, force_hermitian=False):
    """ Active Noise Cancelling
        Apply the active noise cancelling RLS-based algorithm to compensate the
        noise by modeling the primary acoustic path.
        
        @param model Instance of an acoustic model for simulation
        @param order Order of the filter
        @param forget RLS forgetting factor
        @param delta  P initialization parameter
        @param weight_history Enable recording the weight evolution throughout the simulation
        @param force_hermitian Forces P(n) to be an hermitian matrix
        @return Tuple containing the error signal, the last coefficients and the coefficient evolution (if enabled)
                (error_signal, ast_coefficients, coefficients_evolution)    
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
    lambda_inv = 1 / forget
    
    # Initialize arrays
    r = np.zeros((order, 1))
    w = np.zeros((order, 1))
    g = np.zeros((order, 1))
    g_bar = np.zeros((order, 1))
    e_n = np.zeros(N)
    P = np.eye(order) / delta
    if weight_history:
        w_n = np.zeros((order, N))
    i = 0

    # Sample processing loop
    while (model.step()):
        r = np.roll(r, 1)
        r[0] = model.reference_microphone()
        
        # Adaptation gain computation
        g_bar[:,:] = lambda_inv * np.dot(P, r)
        g[:,:] = g_bar / (1 + np.dot(g_bar.T, r))
        P[:,:] = lambda_inv * P - np.dot(g, g_bar.T)
        if force_hermitian:
            P = (P + P.T) / 2

        # Filtering
        y = -np.dot(w.T, r)
        model.speaker(y)
        e = model.error_microphone()
        e_n[i] = e

        # Coefficient updating
        w += g * e
        
        if weight_history:
            w_n[:,i] = w
        i += 1

    if weight_history:
        return e_n, w, w_n
    else:
        return e_n, w