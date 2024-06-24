import torch


class AudioFeatureCNN1(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.main = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, padding=1),
            # nn.BatchNorm1d(16),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(2),
            torch.nn.Dropout(0.25),
            torch.nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            # nn.BatchNorm1d(32),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(2),
            torch.nn.Dropout(0.25),
            torch.nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            # nn.BatchNorm1d(64),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(2),
            torch.nn.Dropout(0.25),
            torch.nn.Flatten(),
            torch.nn.Linear(192, 64),
            torch.nn.ReLU(),  # Added ReLU activation after Linear layer
            torch.nn.Dropout(0.5),
            torch.nn.Linear(64, 1),
            # torch.nn.Linear(1024, 517),
            # torch.nn.Linear(517, 1)
        )

    def forward(self, x):
        out = self.main(x)
        return out


class AudioFeatureCNN2(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.main = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            # torch.nn.BatchNorm1d(32),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(2),
            # torch.nn.Dropout(0.2),
            torch.nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            # torch.nn.BatchNorm1d(64),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(2),
            # torch.nn.Dropout(0.2),
            torch.nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            # torch.nn.BatchNorm1d(128),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(2),
            # torch.nn.Dropout(0.2),
            torch.nn.Flatten(),
            torch.nn.Linear(384, 192),
            torch.nn.ReLU(),  # Added ReLU activation after Linear layer
            torch.nn.Dropout(0.5),
            torch.nn.Linear(192, 1),
        )

    def forward(self, x):
        out = self.main(x)
        return out


class AudioFeatureCNN3(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.main = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            # nn.BatchNorm1d(32),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(2),
            # nn.Dropout(0.2),
            torch.nn.Conv1d(in_channels=32, out_channels=16, kernel_size=3, padding=1),
            # nn.BatchNorm1d(64),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(2),
            # nn.Dropout(0.2),
            torch.nn.Conv1d(in_channels=16, out_channels=8, kernel_size=3, padding=1),
            # nn.BatchNorm1d(128),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(2),
            # nn.Dropout(0.2),
            torch.nn.Flatten(),
            torch.nn.Linear(24, 12),
            torch.nn.ReLU(),  # Added ReLU activation after Linear layer
            torch.nn.Linear(12, 1),
        )

    def forward(self, x):
        out = self.main(x)
        return out
