from acoustic_model import AcousticModel
from scipy import signal
import numpy as np

class TestHeadphone(AcousticModel):
    
    def __init__(self, secondary_enable=False, feedback_enable=False):
        self.x = None
        self.p = np.concatenate((np.zeros(5), np.ones(10) / 10)) if secondary_enable else np.ones(10) / 10
        self.g = np.array([0, 0, 0, 0, 1], dtype=np.float32)
        self.f = np.array([0, 0, 0.5], dtype=np.float32)
        self.y = np.zeros(len(self.g))                         
        self.n = -1                      
        self.secondary_enable = secondary_enable
        self.feedback_enable = feedback_enable
    
    def __len__(self):
        """ Returns the length of the acoustic model, understood as the amount
            of samples for simulation that it contains.
        """
        return len(self.x)
    
    def set_noise(self, x):
        """ Set the noise samples
            @param x Array containing noise samples for the acoustic simulation
        """
        # Set the noise samples
        self.x = x
        
        # Calculate the response of the primary acoustic path
        self.d = signal.lfilter(self.p, [1.0], x) # Response of the primary acoustic path

    def reference_microphone(self):
        """ Takes a sample from the reference microphone.
            @return Reference signal sample
        """
        # Calculate the feedback's path influence in the input measurement
        feedback_sample = np.dot(self.f, self.y[:len(self.f)]) if self.feedback_enable else 0

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
