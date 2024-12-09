import torch
import torch.nn as nn
import torch.optim as optim
import os

from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms
from torchsummary import summary


from lib.defaults import Args
from lib.dataset.dataset import CatDogDataset
from lib.models.cnn import SimpleCNN
from evaluate import evaluate


transform = transforms.Compose([
    transforms.Resize((224, 224)),   # Resize images to 224x224
    transforms.ToTensor(),           # Convert to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

args = Args()

device = torch.device(args.cuda) if torch.cuda.is_available() else torch.device("cpu")

torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)


def train(model, train_loader, test_loader, criterion, optimizer, num_epochs=5, save_path=None, max_num_trials=3):
    model.train()

    # Initialize lists to store the loss and accuracy values
    results = {"train_loss": [],
            "train_acc": [],
            "test_loss": [],
            "test_acc": []
        }

    # Initialize variables to keep track of best test loss
    best_test_loss = float('inf')
    last_model_path = os.path.join(save_path, "last.pt")
    best_model_path = os.path.join(save_path, "best.pt")

    epochs_without_improvement = 0

    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0

        # Training loop with tqdm progress bar
        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training", unit="batch") as train_bar:
            for images, labels in train_bar:
                images = images.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                # Forward pass
                outputs = model(images)
                loss = criterion(outputs, labels)

                # Backward pass and optimize
                loss.backward()
                optimizer.step()

                # Track the loss and accuracy
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                # Update the progress bar description
                train_bar.set_postfix(loss=running_loss / (train_bar.n + 1), accuracy=(correct / total) * 100)

        avg_loss = running_loss / len(train_loader)

        # Evaluate on the test set
        test_loss, test_accuracy = evaluate(model, test_loader, criterion)

        print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}, Accuracy: {correct / total * 100:.2f}%, Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%\n")

        results["train_loss"].append(avg_loss)
        results["train_acc"].append(correct / total * 100)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_accuracy)

        # Save the model if it is the best so far
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            torch.save(model.state_dict(), best_model_path)
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        # Save the last model
        torch.save(model.state_dict(), last_model_path)

        if epochs_without_improvement >= max_num_trials:
            print(f"Early stopping triggered after {epoch+1} epochs without improvement.")
            break

    return results



def main():
    train_dataset = CatDogDataset(data_dir=args.train_src, transforms=transform)
    test_dataset = CatDogDataset(data_dir=args.test_src, transforms=transform)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    model = SimpleCNN(num_classes=2, dropout=args.dropout).to(device)
    summary(model, input_size=(3, 224, 224))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    results = train(model, train_loader, test_loader, criterion, optimizer, num_epochs=args.num_epochs, save_path=args.save_path, max_num_trials=args.max_num_trials)


if __name__ == "__main__":
    main()