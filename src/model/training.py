import os
import numpy as np
import torch
import torch.nn as nn
from src.logs.loggingProject import logger
from src.data.dataPrep import create_sequences

class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(RNNModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)

        out, _ = self.rnn(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out


def train_rnn_model(data,
                    input_size,
                    hidden_layer_size,
                    output_size,
                    num_layers,
                    lr,
                    num_epochs,
                    seq_len,
                    model_name):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
 
    model = RNNModel(input_size=input_size,
                     hidden_size=hidden_layer_size,
                     output_size=output_size,
                     num_layers=num_layers).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    X, y, min_val, max_val = create_sequences(data, seq_len)
    X = torch.tensor(X, dtype=torch.float32).unsqueeze(-1).to(device)
    y = torch.tensor(y, dtype=torch.float32).unsqueeze(-1).to(device) 

    model_dir = os.path.join('src', 'model')
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        logger.info(f'Каталог для модели создан: {model_dir}')

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()

        outputs = model(X)
        loss = criterion(outputs, y)

        loss.backward()
        optimizer.step()

        if (epoch + 1) % 100 == 0:
            logger.info(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

    model_path = os.path.join(model_dir, f'{model_name}.pth')
    torch.save(model.state_dict(), model_path)
    logger.info(f'Model saved to {model_path}')

    outputs = outputs.cpu().detach().numpy()
    outputs = outputs * (max_val - min_val) + min_val
    logger.info(f'Предсказание модели: {outputs[:5].flatten()}')
