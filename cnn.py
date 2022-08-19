import torch

class mnist_model(torch.nn.Module):
    def __init__(self):
        super(mnist_model, self).__init__()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.sequential = torch.nn.Sequential(
            torch.nn.Conv2d(1, 10, kernel_size=3, padding=1, device=device),
            torch.nn.ReLU(),
            torch.nn.Conv2d(10, 20, kernel_size=3, padding=1, device=device),
            torch.nn.ReLU(),
            torch.nn.Flatten(1),
            torch.nn.Linear(20 * 28 * 28, 10, device=device)
        )

    def forward(self, x):
        return self.sequential(x)