import numpy as np


class ResultsBuffer:
    def __init__(self):
        self.x_buffer = []
        self.w_buffer = []
        self.p_buffer = []
        self.prognostic_buffer = []

    def append_x(self, x):
        self.x_buffer.append(x)

    def append_w(self, x):
        self.w_buffer.append(x)

    def append_p(self, x):
        self.p_buffer.append(x)

    def append_prognostic(self, x):
        self.prognostic_buffer.append(x)

    def get_x(self):
        return np.array(self.x_buffer)

    def get_p(self):
        return np.array(self.p_buffer)

    def get_w(self):
        return np.array(self.w_buffer)

    def get_prognostic(self):
        return np.array(self.prognostic_buffer)
    
    

    def __getitem__(self, key):
        return (
            self.x_buffer[key],
            self.w_buffer[key],
            self.p_buffer[key],
            self.prognostic_buffer[key],
        )
