import torch
import torch.nn as nn
import torchvision.models as models

class DeepfakeVideoModel(nn.Module):

    def __init__(self):

        super().__init__()

        resnet = models.resnet18(weights="DEFAULT")

        self.feature_extractor = nn.Sequential(
            *list(resnet.children())[:-1]
        )

        self.lstm = nn.LSTM(
            input_size=512,
            hidden_size=256,
            num_layers=2,
            batch_first=True
        )

        self.fc = nn.Linear(256,1)

    def forward(self,x):

        batch, seq, c, h, w = x.size()

        x = x.view(batch * seq, c, h, w)

        features = self.feature_extractor(x)

        features = features.view(batch, seq, 512)

        lstm_out, _ = self.lstm(features)

        final_output = lstm_out[:,-1,:]

        output = self.fc(final_output)

        return output