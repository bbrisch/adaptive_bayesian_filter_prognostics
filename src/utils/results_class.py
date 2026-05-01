class ResultsBuffer():
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
        
    def __getitem__(self, key):
        return self.x_buffer[key], self.w_buffer[key], self.p_buffer[key], self.prognostic_buffer[key]