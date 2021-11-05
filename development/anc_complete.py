import numpy as np

def anc_complete(model, G, F, order, forget, delta=1e-7, weight_history=False, force_hermitian=False):
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
    lambda_inv = 1 / forget
    
    # Initialize arrays
    rg = np.zeros((len(G), 1))      # Buffer for the input of G(z)
    rw = np.zeros((order, 1))       # Buffer for the input of W(z)
    r_rls = np.zeros((order, 1))    # Buffer for the input of the RLS update equation
    y = np.zeros(len(F))       # Buffer for the input of F(z)
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
        
        # Adaptation gain computation
        g_bar = lambda_inv * np.dot(P, r_rls)
        g = g_bar / (1 + np.dot(g_bar.T, r_rls))
        P = lambda_inv * P - np.dot(g, g_bar.T)
        if force_hermitian:
            P = (P + P.T) / 2
        
        # Coefficient updating
        w += g * e        
        e_n[i] = e
        if weight_history:
            w_n[:,i] = w
        i += 1
    
    if weight_history:
        return e_n, w, w_n
    else:
        return e_n, w