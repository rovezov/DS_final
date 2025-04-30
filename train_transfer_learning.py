import os
import torch
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.models import vgg16, VGG16_Weights
from PIL import Image
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split
from torchvision import transforms
from torchvision.models import vgg16, VGG16_Weights
from glob import glob


def get_dataloaders(data_dir: str, batch_size: int = 16):
    # match the ImageClassification preset from the pretrained weights
    normalize = transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                     std=(0.229, 0.224, 0.225))
    common = [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.Grayscale(num_output_channels=3),  # if your mammos are single‐channel
        transforms.ToTensor(),
        normalize,
    ]
    train_ds = datasets.ImageFolder(os.path.join(data_dir, "train"),
                                    transform=transforms.Compose(common + [transforms.RandomHorizontalFlip()]))
    val_ds   = datasets.ImageFolder(os.path.join(data_dir, "val"),
                                    transform=transforms.Compose(common))
    return (
        torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=4),
        torch.utils.data.DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=4),
    )

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for batch_idx, (imgs, labels) in enumerate(tqdm(loader, desc="Training", unit="batch")):
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        out = model(imgs)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * imgs.size(0)
        if batch_idx % 10 == 0:
            print(f"[Train] Batch {batch_idx+1}/{len(loader)} — loss: {loss.item():.4f}")
    epoch_loss = running_loss / len(loader.dataset)
    print(f"[Train] Epoch complete — avg loss: {epoch_loss:.4f}")
    return epoch_loss

@torch.inference_mode()
def evaluate(model, loader, criterion, device):
    model.eval()
    val_loss, correct = 0.0, 0
    for batch_idx, (imgs, labels) in enumerate(tqdm(loader, desc="Validating", unit="batch")):
        imgs, labels = imgs.to(device), labels.to(device)
        out = model(imgs)
        val_loss += criterion(out, labels).item() * imgs.size(0)
        preds = out.argmax(dim=1)
        correct += (preds == labels).sum().item()
        if batch_idx % 10 == 0:
            batch_l = criterion(out, labels).item()
            print(f"[Val]   Batch {batch_idx+1}/{len(loader)} — loss: {batch_l:.4f}")
    n = len(loader.dataset)
    epoch_loss, epoch_acc = val_loss / n, correct / n
    print(f"[Val] Epoch complete — avg loss: {epoch_loss:.4f}, accuracy: {epoch_acc:.3f}")
    return epoch_loss, epoch_acc

def train_model():
    data_dir   = "./Dataset_BUSI_with_GT"
    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader = get_dataloaders(data_dir, batch_size=16)

    # load pretrained VGG16
    model = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
    # freeze all convolutional layers
    for p in model.features.parameters():
        p.requires_grad = False
    # replace the last classifier layer for 2‑way output
    model.classifier[6] = nn.Linear(4096, 2)
    model.classifier[6] = nn.Linear(4096, 3)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=1e-4)

    best_acc = 0.0
    for epoch in range(1, 21):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        print(f"Epoch {epoch:2d}: train_loss={train_loss:.4f}  "
              f"val_loss={val_loss:.4f}  val_acc={val_acc:.3f}")
        # save best
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "vgg16_mammo_best.pth")

import os
import glob
from PIL import Image
import torch
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split
from torchvision import transforms
from torchvision.models import vgg16, VGG16_Weights
class BUSIDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        classes = {"benign": 0, "malignant": 1}
    def __init__(self, root_dir, transform=None):
         # now handle three classes
        classes = {"benign": 0, "malignant": 1, "normal": 2}
        self.samples = []
        for cls, label in classes.items():
             class_path = os.path.join(root_dir, cls)
             if not os.path.isdir(class_path):
                 raise RuntimeError(f"Directory not found: {class_path}")
             # match png, jpg, jpeg
             for ext in ("*.png", "*.jpg", "*.jpeg"):
                 for fn in glob.glob(os.path.join(class_path, ext)):
                     self.samples.append((fn, label))
        if not self.samples:
             raise RuntimeError(f"No images found in {root_dir}. Check your path/subfolders!")
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label
    
def get_dataloaders(data_dir: str, batch_size: int = 16, val_split: float = 0.2):
    normalize = transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                     std=(0.229, 0.224, 0.225))
    common = [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ]
    # full BUSI dataset
    full_ds = BUSIDataset(data_dir, transform=transforms.Compose(common + [transforms.RandomHorizontalFlip()]))
    # split indices stratified by label
    labels = [lbl for _, lbl in full_ds.samples]
    train_idx, val_idx = train_test_split(
        list(range(len(full_ds))),
        test_size=val_split,
        stratify=labels,
        random_state=42,
    )
    train_ds = Subset(full_ds, train_idx)
    val_ds   = Subset(BUSIDataset(data_dir, transform=transforms.Compose(common)), val_idx)

    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=4),
        DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=4),
    )


def main():
    # Load the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)

    # Modify the classifier to match your trained model
    model.classifier[6] = torch.nn.Linear(4096, 3)  # 3 classes: benign, malignant, normal
    model.load_state_dict(torch.load("vgg16_mammo_best.pth", map_location=device))
    model.to(device)
    model.eval()
    preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),])
    def classify_image(image_path):
        # Load and preprocess the image
        img = Image.open(image_path).convert("RGB")
        img = preprocess(img).unsqueeze(0).to(device)  # Add batch dimension

        # Perform inference
        with torch.no_grad():
            output = model(img)
            probabilities = torch.nn.functional.softmax(output[0], dim=0)

        # Map class indices to labels
        classes = ["benign", "malignant", "normal"]
        predicted_class = classes[probabilities.argmax().item()]
        confidence = probabilities.max().item()

        return predicted_class, confidence
    def classify_images(folder_path):
        # Get all image files in the folder (supports .jpg, .jpeg, .png)
        image_paths = glob.glob(os.path.join(folder_path, "*.[jp][pn]g"))
        results = []
        for image_path in image_paths:
            predicted_class, confidence = classify_image(image_path)
            results.append((image_path, predicted_class, confidence))
            print(f"Image: {image_path}, Predicted class: {predicted_class}, Confidence: {confidence:.2f}")
        return results

    # Example usage
    folder_path = "./Dataset_BUSI_with_GT/malignant"  # Change this to your folder path
    print(f"Classifying images in folder: {folder_path}")
    classify_images(folder_path)
    # for path, cls, conf in results:
    #     print(f"Image: {path}, Predicted class: {cls}, Confidence: {conf:.2f}")
    




if __name__ == "__main__":
    main()