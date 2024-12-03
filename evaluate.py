import torch
import torch.nn as nn

from lib.defaults import Args
from lib.models.cnn import SimpleCNN
from lib.dataset.dataset import CatDogDataset

from torchvision import transforms
from torch.utils.data import DataLoader

args = Args()

device = torch.device(args.cuda) if torch.cuda.is_available() else torch.device("cpu")

torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

transform = transforms.Compose([
    transforms.Resize((224, 224)),   # Resize images to 224x224
    transforms.ToTensor(),           # Convert to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def evaluate(model, test_loader, criterion):
    model.eval()  # Set the model to evaluation mode
    correct = 0
    total = 0
    running_test_loss = 0.0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            running_test_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_loss = running_test_loss / len(test_loader)
    test_accuracy = correct / total * 100
    return test_loss, test_accuracy

if __name__ == "__main__":
    test_dataset = CatDogDataset(data_dir=args.test_src, transforms=transform)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    model = SimpleCNN(num_classes=2, dropout=args.dropout).to(device)
    model.load_state_dict(torch.load("/weights/best.pt", weights_only=True))
    
    criterion = nn.CrossEntropyLoss()

    test_loss, test_accuracy = evaluate(model, test_loader, criterion)

    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")