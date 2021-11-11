import numpy as np

def anc_primary_lms(model, order, step, weight_history=False):
    """ Active Noise Cancelling
        Apply the active noise cancelling algorithm to compensate the
        noise by modeling the primary acoustic path.
        
        @param model Instance of an acoustic model for simulation
        @param order Order of the filter
        @param step Step size used for LMS
        @param weight_history Enable recording the weight evolution throughout the simulation
        @return Tuple containing the error signal, the last coefficients and the coefficient evolution (if enabled)
                (error_signal, ast_coefficients, coefficients_evolution)
    """
    # Validate metaparameters
    if type(order) is int:
        if order < 1:
            raise ValueError("The minimum order is 1")
    else:
        raise ValueError("Order argument must be integer")
    if step <= 0:
        raise ValueError("The step size must be a positive value")
        
    # Parameters
    N = len(model)
    
    # Initialize arrays
    r = np.zeros(order)
    w = np.zeros(order)
    e_n = np.zeros(N)
    if weight_history:
        w_n = np.zeros((order, N))
    i = 0
    
    # Sample processing loop
    while (model.step()):
        r = np.roll(r, 1)
        r[0] = model.reference_microphone()
        y = -np.dot(w, r)
        model.speaker(y)
        e = model.error_microphone()
        w += step * r * e
        e_n[i] = e
        if weight_history:
            w_n[:,i] = w
        i += 1
    
    if weight_history:
        return e_n, w, w_n
    else:
        return e_n, w