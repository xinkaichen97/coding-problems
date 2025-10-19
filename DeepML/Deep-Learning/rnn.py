"""
Problems for RNNs
"""


def rnn_forward(input_sequence: list[list[float]], initial_hidden_state: list[float], Wx: list[list[float]], Wh: list[list[float]], b: list[float]) -> list[float]:
    """
    https://www.deep-ml.com/problems/54
    """
    # convert inputs to arrays
    X  = np.array(input_sequence, dtype=float) 
    Wx = np.array(Wx, dtype=float) 
    Wh = np.array(Wh, dtype=float)  
    b  = np.asarray(b, dtype=float)
    h  = np.asarray(initial_hidden_state, dtype=float)
  
    # process each time step
    for x_t in X:
        h = np.tanh(Wx @ x_t + Wh @ h + b)
    # get final hidden state and round
    final_hidden_state = np.round(h, 4)
    return final_hidden_state


class LSTM:
    """
    https://www.deep-ml.com/problems/59
    """
    def __init__(self, input_size, hidden_size):
      self.input_size = input_size
      self.hidden_size = hidden_size
    
      # Initialize weights and biases
      self.Wf = np.random.randn(hidden_size, input_size + hidden_size)
      self.Wi = np.random.randn(hidden_size, input_size + hidden_size)
      self.Wc = np.random.randn(hidden_size, input_size + hidden_size)
      self.Wo = np.random.randn(hidden_size, input_size + hidden_size)
    
      self.bf = np.zeros((hidden_size, 1))
      self.bi = np.zeros((hidden_size, 1))
      self.bc = np.zeros((hidden_size, 1))
      self.bo = np.zeros((hidden_size, 1))
      
    def forward(self, x, initial_hidden_state, initial_cell_state):
        # initialize states
        seq_len = x.shape[0]
        h_t = initial_hidden_state.reshape(self.hidden_size, 1)
        c_t = initial_cell_state.reshape(self.hidden_size, 1)
        # store states
        hidden_states = []
        cell_states = []

        # process each time step
        for t in range(seq_len):
            # get current input and reshape to column vector
            x_t = x[t].reshape(self.input_size, 1)
            # concatenate hidden state and input (h_t first, then x_t)
            combined = np.vstack((h_t, x_t))
          
            # forget gate
            f_t = self.sigmoid(self.Wf @ combined + self.bf)
            # input gate
            i_t = self.sigmoid(self.Wi @ combined + self.bi)
            # candidate cell state
            c_tilde_t = np.tanh(self.Wc @ combined + self.bc)
            # update cell gate
            c_t = f_t * c_t + i_t * c_tilde_t
            # output gate
            o_t = self.sigmoid(self.Wo @ combined + self.bo)
            # update hidden state
            h_t = o_t * np.tanh(c_t)
          
            # store states
            hidden_states.append(h_t.copy())
            cell_states.append(c_t.copy())
        
      return hidden_states, hidden_states[-1], cell_states[-1]
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
