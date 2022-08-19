import wandb
from torchvision import datasets, transforms
import torch
import os
from cnn import mnist_model

def train():
    # Prepare data    
    train_dataset = datasets.MNIST(root='./data', train=True, transform=transforms.ToTensor())

    # Get sweep parameters from wandb
    wandb.init()
    config = wandb.config
    model = mnist_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    criterion = torch.nn.CrossEntropyLoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=config.batch_size, shuffle=True)
    n_total_steps = len(train_loader)

    for epoch in range(config.num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            accuracy = (outputs.argmax(dim=1) == labels).float().mean()
            # Backward and optimize
            loss.backward()
            optimizer.step()
            if (i+1) % 100 == 0:
                print(f'Epoch [{epoch+1}/{config.num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}')
                wandb.log({"Loss": loss.item()}, step=i+1)
                wandb.log({"Accuracy": accuracy.item()}, step=i+1)
    
    save(model)

def save(model):    
    model.eval()
    torch.save(model.state_dict(), os.path.join(wandb.run.dir, "model.pt"))            
    # Save artifacts to wandb
    art = wandb.Artifact(f'mnist-nn-{wandb.run.id}', type="model")
    art.add_file(os.path.join(wandb.run.dir, "model.pt"))
    wandb.log_artifact(art)
    wandb.run.link_artifact(art, "shohamweiss/mnist_pytorch/mnist-cnn", aliases=['mnist-cnn'])  

def main():
    # Log in to wandb
    os.system("wandb login")    

    # Get dataset from wandb
    dataset_artifact = wandb.Api().artifact("shohamweiss/mnist_pytorch/minst_dataset:latest")
    dataset_artifact.download()
     
    # Run agent
    sweep_id = "tdq648pm"
    wandb.agent(sweep_id, function=train)

if __name__ == "__main__":
    main()