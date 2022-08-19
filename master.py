import wandb
from torchvision import datasets, transforms
import os

def main():
    # Log in to wandb
    os.system("wandb login")
    # Use mnist_pytorch project
    run = wandb.init(project="mnist_pytorch")
    run.name = "master"
    # Prepare data
    # MNIST
    train_dataset = datasets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
    test_dataset = datasets.MNIST(root='./data', train=False, transform=transforms.ToTensor())
    #  Upload data to wandb
    wandb_data = wandb.Artifact("minst_dataset", type="data")
    wandb_data.add_dir("./data")
    run.log_artifact(wandb_data, "minst_dataset")

    # Create a config for sweep
    config = {
        "name": "mnist-sweep",
        "method": "random",
        "parameters": {
            "batch_size": {
                "values" : [64, 128, 256]
                },
            "num_epochs": {
                "values" : [1, 2, 3]
                },
            "lr": {
                "values" : [0.001, 0.01, 0.1]
                }
            }
        }
    
    # Create a sweep
    sweep_id = wandb.sweep(config)    

if __name__ == "__main__":
    main()