import torch.nn as nn


class AudioFeatureNN1(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(AudioFeatureNN1, self).__init__()
        self.layer1 = nn.Linear(input_dim, input_dim)
        self.layer2 = nn.Linear(input_dim, input_dim)
        self.layer3 = nn.Linear(input_dim, input_dim)
        self.layer4 = nn.Linear(input_dim, input_dim)
        self.layer5 = nn.Linear(input_dim, input_dim)
        self.output_layer = nn.Linear(input_dim, output_dim)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.relu(self.layer3(x))
        x = self.relu(self.layer4(x))
        x = self.relu(self.layer5(x))
        x = self.sigmoid(self.output_layer(x))
        return x


class AudioFeatureNN2(nn.Module):
    def __init__(self, input_dim, output_dim, dropout_prob=0.25):
        super(AudioFeatureNN2, self).__init__()
        self.layer1 = nn.Linear(input_dim, 2 * input_dim)
        self.layer2 = nn.Linear(2 * input_dim, 4 * input_dim)
        self.layer3 = nn.Linear(4 * input_dim, 2 * input_dim)
        self.layer4 = nn.Linear(2 * input_dim, input_dim)
        # self.layer5 = nn.Linear(2*input_dim, input_dim)
        self.output_layer = nn.Linear(input_dim, output_dim)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(
            dropout_prob
        )  # Dropout layer with the specified probability

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.dropout(x)  # Apply dropout after activation
        x = self.relu(self.layer2(x))
        x = self.dropout(x)
        x = self.relu(self.layer3(x))
        x = self.dropout(x)
        x = self.relu(self.layer4(x))
        x = self.dropout(x)
        # x = self.relu(self.layer5(x))
        # x = self.dropout(x)
        x = self.sigmoid(self.output_layer(x))
        return x
