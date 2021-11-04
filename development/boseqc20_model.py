from acoustic_model import AcousticModel
from scipy import signal
import numpy as np
import json

class BoseQC20(AcousticModel):
    """ Acoustic simulation of the BoseQC20 headphones based on the measurements
        performed by 
            'Acoust path database for ANC in-ear headphone development'
            Stefan LIEABICH, Johannes FABRY, Peter JAX, Peter VARY

        NOTE: It only supports simulating the primary acoustic path
    """
    
    def __init__(self, filepath, secondary_enable=False, feedback_enable=False, room='anechoic_chamber', index=0, order=2000):
        """ Initializes the BoseQC20 instance
            @param filepath Filepath to the JSON file containing the dataset
        """
        
        # Open the JSON file containing measurements of the BoseQC20 headphones
        self.acoustic_paths_filepath = filepath
        self.acoustic_paths_file = open(self.acoustic_paths_filepath, 'r')
        self.acoustic_paths_data = json.load(self.acoustic_paths_file)
        self.room = room
        self.index = index
        self.order = order
        
        self._load_measurements()
        
        # Initialize internal signals
        self.x = None                                                              # Input signal of the acoustic model for simulation
        self.xp = None                                                             # Filtered input signal (by the secondary acoustic path)
        self.y = np.zeros(len(self.g))                                             # Saves the speaker sample
        self.n = -1                                                                # Current sampling time or instant
        self.secondary_enable = secondary_enable                                   # Whether the secondary acoustic path is simulated
        self.feedback_enable = feedback_enable
    
    def __len__(self):
        """ Returns the length of the acoustic model, understood as the amount
            of samples for simulation that it contains.
        """
        return len(self.x)
    
    def _load_measurements(self):

        # Choose the physical or acoustic paths from the dataset
        if self.room == 'anechoic_chamber':
            self.p = self.acoustic_paths_data['anechoic_chamber']['primary'][self.index][0] # Primary acoustic path impulse response estimation
            self.g = self.acoustic_paths_data['anechoic_chamber']['secondary'][0]           # Secondary acoustic path impulse response estimation
            self.f = self.acoustic_paths_data['anechoic_chamber']['feedback'][0]            # Feedback acoustic path impulse response estimation
        elif self.room == 'acoustic_booth':       
            self.p = self.acoustic_paths_data['acoustic_booth'][self.index]['P'][0]
            self.g = self.acoustic_paths_data['acoustic_booth'][self.index]['G'][0]
            self.f = self.acoustic_paths_data['acoustic_booth'][self.index]['F'][0]
            
        # Cut the responses
        self.p = self.p[:self.order]
        self.g = self.g[:150]
        self.f = self.f[:150]
    
    def set_noise(self, x):
        """ Set the noise samples
            @param x Array containing noise samples for the acoustic simulation
        """
        # Set the noise samples
        self.x = x
        self.xp = signal.lfilter(self.g, [1.0], self.x)
        self.d = signal.lfilter(self.p, [1.0], self.x)

    def set_measurement_index(self, idx, order=None):
        """ Change the measurement index used. 
            Optionally also change the primary path's length
            Only valid for "acoustic_booth" measurements
        """
        if self.index != idx:
            self.index = idx
            if order is not None:
                self.order = order
            # Load new measurements
            self._load_measurements()
            # Re-filter the noise
            self.set_noise(self.x)
            
    def reference_microphone(self):
        """ Takes a sample from the reference microphone.
            @return Reference signal sample
        """
        # Calculate the feedback's path influence in the input measurement
        feedback_sample = np.dot(self.f, self.y) if self.feedback_enable else 0

        return self.x[self.n] + feedback_sample
    
    def error_microphone(self):
        """ Takes a sample from the error microphone.
            @return Error signal sample
        """
        if self.secondary_enable:
            return self.d[self.n] + np.dot(self.g, self.y)
        else:
            return self.d[self.n] + self.y[0]
    
    def speaker(self, y):
        """ Sets the output sample for the speaker.
            @param y Output signal sample
        """
        self.y = np.roll(self.y, 1)
        self.y[0] = y
    
    def step(self):
        """ Moves forward to the next sampling time or instant.
            @return Boolean indicating it has not reached the end of the samples
        """
        if self.n < len(self.x) - 1:
            self.n += 1
            return True
        else:
            return False
    
    def reset(self):
        """ Resets the sampling time to zero.
        """
        self.n = -1
        