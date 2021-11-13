import matplotlib.pyplot as plt
from scipy import signal
import numpy as np
from pydsm.audio_weightings import a_weighting

def plot_anc_results(e, w=None, weight_plot='response', labels=None, alpha=None, log_scale=False):
    """ Plot the results of the ANC algorithm.
        @param error_history
        @param weight_history
    """
    
    if w is not None:
        fig, ax = plt.subplots(2, 1, figsize=(8, 16))
        error_ax = ax[0]
    else:
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        error_ax = ax
        
    if alpha is None:
        alpha = 1
    
    if labels is None:
        labels = ['Ruido con filtrado pasivo', 'Ruido con ANC']
    for k, error in enumerate(e):
        if not log_scale:
            error_ax.plot(error, label=labels[k], alpha=alpha if k > 0 else 1)
        else:
            error_ax.semilogy(error, label=labels[k], alpha=alpha if k > 0 else 1)
    error_ax.set_ylabel('$e(n)$', fontsize=16)
    error_ax.set_xlabel('$n$', fontsize=16)
    error_ax.grid()
    error_ax.legend(loc='upper right', fontsize=14)
    
    if w is not None:
        if weight_plot == 'response':
            ax[1].stem(w, use_line_collection=True)
            ax[1].set_ylabel('$w_n$', fontsize=16)
            ax[1].set_xlabel('$n$', fontsize=16)
            ax[1].grid()
        elif weight_plot == 'history':
            for i in range(w.shape[0]):
                ax[1].plot(w[i,:])
            ax[1].set_ylabel('$w_i(n)$', fontsize=16)
            ax[1].set_xlabel('$n$', fontsize=16)
            ax[1].grid()
    
    plt.show()
    
def plot_frequency_analysis(p, g, w, fs):
    """ Plot the frequency responses related to the ANC system.
        P/G is the target frequency response and W is the one reached by
        the FX-LMS algorithm. G's spectrum is also shown
        @param p  P(z) impulse response
        @param g  G(z) impulse response
        @param w  W(z) impulse response
        @param fs Sampling frequency
    """
    # Calculate P/G, W and G frequency responses
    w_pg, h_pg = signal.freqz(p, g, fs=fs)
    w_w, h_w = signal.freqz(w, [1.0], fs=fs)
    w_g, h_g = signal.freqz(g, [1.0], fs=fs)

    fig, ax = plt.subplots(2, 1, figsize=(8, 8))

    ax[0].set_ylabel('Amplitud [dB]', fontsize=16)
    ax[0].set_xlabel('Frecuencia [kHz]', fontsize=16)
    ax[0].grid()

    ax[0].plot(w_pg / 1e3, 20*np.log10(np.abs(h_pg)), label='$P/G$')
    ax[0].plot(w_w / 1e3,20*np.log10(np.abs(h_w)), label='$W$')
    ax[0].legend(fontsize=13)

    ax[1].set_ylabel('Amplitud [dB]', fontsize=16)
    ax[1].set_xlabel('Frecuencia [kHz]', fontsize=16)
    ax[1].grid()

    ax[1].plot(w_g / 1e3,20*np.log10(np.abs(h_g)), label='$G$')
    ax[1].legend(fontsize=13)

    plt.show()
    
def apply_a_weighting(f, spectrum):
    """ Applies A-Weighting to the given power spectrum
        @param f Frequency range
        @param spectrum 
        @returns weighted_spectrum
    """
    weights = a_weighting(f)
    return weights * spectrum

def total_attenuation(x, y, fs, nperseg):
    """ Calculate attenuation in dB
        @param x input signal
        @param y output signal
        @param fs  sampling frequency
        @param nperseg
        @returns fnn, Rnn, fee, Ree, A Attenuation in dB
    """
    # Estimate signals' power spectral density
    fnn, Rnn = signal.welch(x, fs=fs, window='hamming', nperseg=nperseg)
    fee, Ree = signal.welch(y, fs=fs, window='hamming', nperseg=nperseg)

    # Calculate energies on the time domain
    Ein = np.sum(x ** 2)
    Eout = np.sum(y ** 2)
    
    # Calculate Noise Reduction    
    A = -10*np.log10(Eout / Ein)
    
    return A, (fnn, Rnn, fee, Ree)

def plot_error_analysis(input_noise, output_error, fs, title, a_weighting=False):
    """ Plot power spectrum density of output signal and frequency response with 
        optional application of A-weighting
        @param input_noise System's input
        @param output_error System's output error
        @param fs Sampling frequency
        @param title Title
        @param a_weighting whether to apply A-weighting or not
    """
    # Calculate total attenuation
    A, (fnn, Rnn, fee, Ree) = total_attenuation(input_noise, output_error, fs, 512)
    
    # Calculate frequency response
    H = Ree / Rnn
    
    # Apply A-weighting if requested
    if a_weighting:
        Ree = apply_a_weighting(fee, Ree)
        H = apply_a_weighting(fee, H)
    
    # Plot results
    fig, ax = plt.subplots(1, 2, figsize=(16, 8))
    fig.suptitle(title + f' - Atenuación total = {A:.2f}dB', fontsize=25)

    ax[0].semilogy(fee / 1e3, Ree, label='Espectro de potencia del error de salida')
    ax[0].tick_params(labelsize=16)
    ax[0].legend(fontsize=16)
    ax[0].set_ylabel('$R_{ee}(f)$', fontsize=16)
    ax[0].set_xlabel('$f$ [kHz]', fontsize=16)
    ax[0].grid()
        
    ax[1].semilogy(fnn / 1e3, H, label=f'$E/X$')
    ax[1].tick_params(labelsize=16)
    ax[1].legend(fontsize=16)
    ax[1].set_ylabel('$|A| [dB]$', fontsize=16)
    ax[1].set_xlabel('$f$ [kHz]', fontsize=16)
    ax[1].grid()

def plot_frequency_responses(primaries, secondaries, weights, fs, labels=None):
    """ Plot the frequency responses related to the ANC system.
        P/G is the target frequency response and W is the one reached by
        the FX-LMS algorithm.
        @param primaries   list of P(z) impulse responses
        @param secondaries list of G(z) impulse responses
        @param weights     list of W(z) impulse responses
        @param fs Sampling frequency
    """
    N = len(primaries)
    
    fig, ax = plt.subplots(N, 1, figsize=(15, N*5))
            
    if N == 1:
        ax = [ax]
        
    plt.subplots_adjust(hspace=0.3)
    
    for k in range(N):
         
        # Configure plot
        ax[k].set_ylabel('Amplitud [dB]', fontsize=16)
        ax[k].set_xlabel('Frecuencia [kHz]', fontsize=16)
        ax[k].grid()
        ax[k].set_title(f'Segmento {k+1}', fontsize=18)
        
        # Calculate P/G frequency response
        f_pg, h_pg = signal.freqz(primaries[k], secondaries[k], fs=fs)
   
        # Plot P/G 
        ax[k].plot(f_pg / 1e3, 20*np.log10(np.abs(h_pg)), label=f'$P/G$')
        
        if labels is None:
            if len(weights) > 1:
                labels = ['W c/G idéntica', 'W c/G distinta']
            else:
                labels = ['W']
        # Calculate all W frequency responses
        for j in range(len(weights)):
            f_w, h_w = signal.freqz(weights[j][k], [1.0], fs=fs)
            ax[k].plot(f_w / 1e3, 20*np.log10(np.abs(h_w)), label=labels[j])
        
        ax[k].legend(fontsize=13)
      
    plt.show()

def plot_error_evolution(passive_noise, anc_errors, nperseg, overlap, fs, labels=['G idéntica', 'G distinta'], title = None):
    """ Plots the attenuation as a function of time, showing the system's adaptation
        @param passive_noise Noise that would be obtained with a passive system
        @param anc_errors    list of ANC output error for different conditions
        @param nperseg       Amount of samples used for each segment
        @param overlap       Segment overlap [0, 1.0] 
        @param fs            Sampling frequency
    """
    # Number of iterations and time axis
    N = min(len(passive_noise), len(anc_errors[0]))
    step = int(nperseg*(1 - overlap))
    it_count = int(N // step)
    t = np.linspace(0, N / fs, it_count)
    
    plt.figure(figsize=(9, 3))
    
    # Calculate energies on the time domain
    Ein = np.array([ np.sum(passive_noise[k*step:(k+1)*step] ** 2) for k in range(it_count)])
    for i in range(len(anc_errors)):
        Eout = np.array([ np.sum(anc_errors[i][k*step:(k+1)*step] ** 2) for k in range(it_count)])
        
        # Calculate Noise Reductions for each segment
        NRs = -10*np.log10(Eout / Ein)

        plt.plot(t, NRs, marker='o', label=labels[i])

    plt.ylabel("NR [dB]", fontsize=13)
    plt.xlabel("t [s]", fontsize=13)
    if title is not None:
        plt.title(title, fontsize=15)
    else:
        plt.title('Atenuación en función del tiempo', fontsize=15)
    plt.grid()
    if (len(anc_errors) > 1):
        plt.legend(fontsize=13)
    plt.show()

def plot_execution_times(times=[[]], lengths=[], labels=[]):
    """ Plots the time it took for a given list of algorithms to run with the lengths given.
        @param times         matrix containing in every row, the list of times it took for a given algorithm to run with the lengths given
        @param lengths       list of lengths used to run the algorithms
        @param labels        algorithms' names
    """

    plt.figure(figsize=(9, 3))

    for i, algorithm_times in enumerate(times):
        plt.plot(lengths, algorithm_times, marker='o', label=labels[i])

    plt.ylabel("Tiempo de ejecución [s]", fontsize=13)
    plt.xlabel("Largo N de las muestras de ruido", fontsize=13)
    plt.title('Tiempo de ejecución en función del largo de la muestra', fontsize=15)
    plt.legend()
    plt.grid()
    
    plt.show()
    

def plot_spectrogram(x, fs, title, N=256):
    """ Plot the spectrogram of the given samples
        @param x Samples
        @param fs Sampling frequency
        @param title Plot's title
        @param N Windows size
    """
    # Plot the spectrogram
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    fig.suptitle(title, fontsize=18)

    ax = [ax]
    
    ax[0].specgram(x, Fs=fs, NFFT=N, window=np.hamming(N), noverlap=0)
    ax[0].set_ylabel('$f$ [Hz]', fontsize=16)
    ax[0].set_xlabel('$t$ [s]', fontsize=16)
    ax[0].grid()

    plt.show()