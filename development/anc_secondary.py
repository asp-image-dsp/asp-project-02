import numpy as np

def anc_primary_secondary_rls(model, G, order, forget, delta, weight_history=False, force_hermitian=False):
    """ Active Noise Cancelling
        Apply the active noise cancelling algorithm to compensate the
        noise by modeling the primary acoustic path and compensating the 
        secondary acoustic path.
        
        @param model Instance of an acoustic model for simulation
        @param G Secondary acoustic path
        @param order Order of the filter
        @param forget RLS forgetting factor
        @param delta  P initialization parameter
        @param weight_history Enable recording the weight evolution throughout the simulation
        @param force_hermitian Forces P(n) to be an hermitian matrix
        @return Tuple containing the error signal, the last coefficients and the coefficient evolution (if enabled)
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
    lambda_inv = 1 / forget
    
    # Initialize arrays
    rg = np.zeros((len(G), 1))      # Buffer for the input of G(z)
    rw = np.zeros((order, 1))       # Buffer for the input of W(z)
    rrls = np.zeros((order, 1))     # Buffer for the input of the LMS update equation
    w = np.zeros((order, 1))
    e_n = np.zeros(N)
    P = np.eye(order) / delta
    if weight_history:
        w_n = np.zeros((order, N))
    i = 0

    # Sample processing loop
    while (model.step()):
        rg = np.roll(rg, 1)
        rw = np.roll(rw, 1)
        rrls = np.roll(rrls, 1)
        rg[0] = rw[0] = model.reference_microphone()
        rrls[0] = np.dot(G, rg)
        
        # Adaptation gain computation
        # g_bar = lambda_inv * np.dot(P, rrls)
        # g = g_bar / (1 + np.dot(g_bar.T, rrls))
        # P = lambda_inv * P - np.dot(g, g_bar.T)
        
        g_bar = lambda_inv * np.dot(P, rrls)
        g = g_bar / (1 + np.dot(g_bar.T, rrls))
        P = lambda_inv * P - np.dot(g, g_bar.T)
        P /= np.linalg.norm(P)
        
        if force_hermitian:
            P = (P + P.T) / 2

        if np.linalg.norm(P) > 10 :
            print(f"HELP, |P| = {np.linalg.norm(P)}")
        
        y = -np.dot(w.T, rw)
        model.speaker(y)
        e = model.error_microphone()
        w += g * e
        e_n[i] = e

        if weight_history:
            w_n[:,i] = w
        i += 1

    if weight_history:
        return e_n, w, w_n
    else:
        return e_n, w
