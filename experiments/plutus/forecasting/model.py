from torch import nn

class Model(nn.Module):
    def __init__(self, input_size, hidden_size, prediction_horizon, output_size, num_layers=2, dropout=0.2):
        super().__init__()
        self.prediction_horizon = prediction_horizon
        self.output_size = output_size
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )
        
        self.linear = nn.Linear(hidden_size, prediction_horizon * output_size)
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]
        predictions = self.linear(last_hidden)
        return predictions.reshape(x.size(0), self.prediction_horizon, self.output_size)