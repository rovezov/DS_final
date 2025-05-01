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
from datasets import load_dataset
from sklearn.metrics import roc_auc_score, classification_report
import torch.optim.lr_scheduler as lr_scheduler
import time
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
import random
import shutil
from transformers import ViTForImageClassification, ViTImageProcessor, TrainingArguments, Trainer
from datasets import load_dataset, DatasetDict
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, Grayscale
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torchvision.transforms import Grayscale
from transformers import ViTForImageClassification
from torchvision.datasets import ImageFolder
from torch.utils.data import random_split, DataLoader
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from transformers import TrainingArguments
import os
import torch
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from PIL import Image
from transformers import ViTForImageClassification
from sklearn.model_selection import KFold



def get_dataloaders(data_dir: str, batch_size: int = 16):
    # match the ImageClassification preset from the pretrained weights
    train_transforms = transforms.Compose([
    transforms.Resize(128),
    transforms.CenterCrop(128),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    train_ds = datasets.ImageFolder(os.path.join(data_dir, "train"),
                                    transform=transforms.Compose(train_transforms + [transforms.RandomHorizontalFlip()]))
    val_ds   = datasets.ImageFolder(os.path.join(data_dir, "val"),
                                    transform=transforms.Compose(train_transforms))
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
    data_dir   = "./balanced_mammogram_dataset"
    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader = get_dataloaders(data_dir, batch_size=64)

    # load pretrained VGG16
    model = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
    # freeze all convolutional layers
    for p in model.features.parameters():
        p.requires_grad = False
    # replace the last classifier layer for 2‑way output
    model.classifier[6] = nn.Linear(4096, 2)
    model.to(device)

    cancer_samples = len([s for s in train_loader.dataset if s[1] == 1])
    no_cancer_samples = len([s for s in train_loader.dataset if s[1] == 0])
    class_weights = torch.tensor([1.0 / no_cancer_samples, 1.0 / cancer_samples], dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.classifier.parameters(), lr=1e-5)

    best_acc = 0.0
    for epoch in range(1, 21):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        model.eval()
        val_loss, y_true, y_pred = 0.0, [], []
        with torch.inference_mode():
            for imgs, labels in tqdm(val_loader, desc=f"Epoch {epoch} - Validating", unit="batch"):
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                val_loss += criterion(outputs, labels).item() * imgs.size(0)
                probs = torch.softmax(outputs, dim=1)[:, 1]  # Get probabilities for the positive class
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(probs.cpu().numpy())

        val_loss /= len(val_loader.dataset)

        # Print classification report for additional metrics
        y_pred_classes = [1 if p > 0.5 else 0 for p in y_pred]
        print(classification_report(y_true, y_pred_classes, target_names=["no_cancer", "cancer"]))
        print(f"Epoch {epoch:2d}: train_loss={train_loss:.4f}  "
            f"val_loss={val_loss:.4f}  val_acc={val_acc:.3f}")
        # save best
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "vgg16_mammo_best.pth")


class BUSIDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        classes = {"no_cancer": 0, "cancer": 1}
        self.samples = []
        for cls, label in classes.items():
            class_path = os.path.join(root_dir, cls)
            if not os.path.isdir(class_path):
                raise RuntimeError(f"Directory not found: {class_path}")
            for ext in ("*.png", "*.jpg", "*.jpeg"):
                for fn in glob.glob(os.path.join(class_path, ext)):
                    self.samples.append((fn, label))

        if not self.samples:
            raise RuntimeError(f"No images found in {root_dir}. Check your path/subfolders!")

        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label

    
def get_dataloaders(data_dir: str, batch_size: int = 16, val_split: float = 0.2):
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
    # Load the full dataset with balanced classes
    full_ds = BUSIDataset(data_dir, transform=transforms.Compose(common + [transforms.RandomHorizontalFlip()]))

    # Ensure stratified split includes both classes
    labels = [lbl for _, lbl in full_ds.samples]
    train_idx, val_idx = train_test_split(
        list(range(len(full_ds))),
        test_size=val_split,
        stratify=labels,  # Ensure stratified split
        random_state=42,
    )
    train_ds = Subset(full_ds, train_idx)
    val_ds = Subset(full_ds, val_idx)  # Use the same dataset for validation

    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=8, prefetch_factor=2),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=8),
    )


def evaluateModel():
    # Load the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available():
        print("CUDA is not available. Using CPU.")
    else:
        print("Using GPU for inference.")
    
    # Load the trained VGG16 model
    model = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
    model.classifier[6] = torch.nn.Linear(4096, 2)  # Binary classification: no_cancer, cancer
    model.load_state_dict(torch.load("vgg16_binary_best.pth", map_location=device))  # Load the trained weights
    model.to(device)
    model.eval()
    normalize = transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                    std=(0.229, 0.224, 0.225))
    preprocess = transforms.Compose([
        transforms.Resize(128),
        transforms.CenterCrop(128),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

    # Function to classify a single image
    def classify_image(image_path):
        # Load and preprocess the image
        img = Image.open(image_path).convert("RGB")
        img = preprocess(img).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(img)
            probabilities = torch.nn.functional.softmax(output[0], dim=0)

        classes = ["no_cancer", "cancer"]
        predicted_class = classes[probabilities.argmax().item()]
        confidence = probabilities.max().item()

        return predicted_class, confidence

    # Function to classify all images in a folder
    def classify_images(folder_path):
        # Get all image files in the folder (supports .jpg, .jpeg, .png)
        image_paths = glob.glob(os.path.join(folder_path, "*.[jp][pn]g"))
        results = []
        correct = 0
        for image_path in image_paths:
            predicted_class, confidence = classify_image(image_path)
            if predicted_class == "no_cancer":
                correct += 1
            results.append((image_path, predicted_class, confidence))
            print(f"Image: {image_path}, Predicted class: {predicted_class}, Confidence: {confidence:.2f}")
        print(f"Accuracy: {correct / len(image_paths) * 100:.2f}%")
        return results

    # Example usage
    folder_path = "./test_mammogram_dataset/no_cancer"  # Change this to your folder path
    print(f"Classifying images in folder: {folder_path}")
    classify_images(folder_path)

    
def download_and_save_images(dataset_name, split, output_dir):
    # Load the dataset
    dataset = load_dataset(dataset_name, split=split)

    # Check if the dataset has a 'cancer' column
    if 'cancer' in dataset.column_names:
        # Map the 'cancer' column values to class names
        classes = {0: "no_cancer", 1: "cancer"}  # Assuming 0 = no cancer, 1 = cancer
    else:
        raise ValueError("The dataset does not have a 'cancer' column.")

    # Create output directories for each class
    for label, class_name in classes.items():
        os.makedirs(os.path.join(output_dir, class_name), exist_ok=True)

    # Iterate through the dataset and save images
    for idx, sample in enumerate(dataset):
        img = sample['image']  # Get the image field
        if isinstance(img, Image.Image):  # Check if it's already a PIL Image
            img = img.convert("RGB")
        else:
            raise ValueError(f"Unexpected image format at index {idx}: {type(img)}")

        label = sample['cancer']  # Get label from the 'cancer' column
        class_name = classes[label]  # Convert label to class name

        # Save the image in the corresponding class folder
        img.save(os.path.join(output_dir, class_name, f"{idx}.jpg"))

    print(f"Images successfully downloaded and saved to {output_dir}.")



def preprocess_and_save_balanced_dataset(root_dir, train_val_dir, test_split=0.2):
    """
    Preprocess and save a balanced dataset by undersampling the majority class.
    All data (train/validate and test) will be placed in the train_val_dir folder.

    Args:
        root_dir (str): Path to the root directory containing the raw dataset.
        train_val_dir (str): Path to the output directory for the balanced dataset.
        test_split (float): Proportion of the dataset to use for testing.
    """
    classes = {"no_cancer": 0, "cancer": 1}
    samples = []

    # Collect all samples
    for cls, label in classes.items():
        class_path = os.path.join(root_dir, cls)
        if not os.path.isdir(class_path):
            raise RuntimeError(f"Directory not found: {class_path}")
        for ext in ("*.png", "*.jpg", "*.jpeg"):
            for fn in glob.glob(os.path.join(class_path, ext)):
                samples.append((fn, label))

    # Split into train/validate and test sets
    train_val_samples, test_samples = train_test_split(
        samples, test_size=test_split, stratify=[s[1] for s in samples], random_state=42
    )

    # Combine train/validate and test samples into one balanced dataset
    balanced_samples = train_val_samples + test_samples

    # Balance the dataset by undersampling the majority class
    cancer_samples = [s for s in balanced_samples if s[1] == 1]
    no_cancer_samples = [s for s in balanced_samples if s[1] == 0]
    if len(cancer_samples) < len(no_cancer_samples):
        no_cancer_samples = random.sample(no_cancer_samples, len(cancer_samples))
    balanced_samples = cancer_samples + no_cancer_samples
    random.shuffle(balanced_samples)

    # Save the balanced dataset
    for cls, label in classes.items():
        class_output_dir = os.path.join(train_val_dir, cls)
        os.makedirs(class_output_dir, exist_ok=True)

    for idx, (img_path, label) in enumerate(balanced_samples):
        class_name = "cancer" if label == 1 else "no_cancer"
        output_path = os.path.join(train_val_dir, class_name, f"{idx}.jpg")
        shutil.copy(img_path, output_path)

    print(f"Balanced dataset saved to {train_val_dir}.")


def preprocess_and_save_balanced_dataset_oversample(root_dir, train_val_dir, test_dir, test_split=0.2):
    classes = {"no_cancer": 0, "cancer": 1}
    samples = []

    # Collect all samples
    for cls, label in classes.items():
        class_path = os.path.join(root_dir, cls)
        if not os.path.isdir(class_path):
            raise RuntimeError(f"Directory not found: {class_path}")
        for ext in ("*.png", "*.jpg", "*.jpeg"):
            for fn in glob.glob(os.path.join(class_path, ext)):
                samples.append((fn, label))

    # Split into train/validate and test sets
    train_val_samples, test_samples = train_test_split(
        samples, test_size=test_split, stratify=[s[1] for s in samples], random_state=42
    )

    # Balance train/validate set by oversampling the minority class
    cancer_samples = [s for s in train_val_samples if s[1] == 1]
    no_cancer_samples = [s for s in train_val_samples if s[1] == 0]
    if len(cancer_samples) < len(no_cancer_samples):
        cancer_samples = random.choices(cancer_samples, k=len(no_cancer_samples))
    balanced_samples = cancer_samples + no_cancer_samples
    random.shuffle(balanced_samples)

    # Save the train/validate set
    for cls, label in classes.items():
        class_output_dir = os.path.join(train_val_dir, cls)
        os.makedirs(class_output_dir, exist_ok=True)

    for idx, (img_path, label) in enumerate(balanced_samples):
        class_name = "cancer" if label == 1 else "no_cancer"
        output_path = os.path.join(train_val_dir, class_name, f"{idx}.jpg")
        shutil.copy(img_path, output_path)

    # Save the test set
    for cls, label in classes.items():
        class_output_dir = os.path.join(test_dir, cls)
        os.makedirs(class_output_dir, exist_ok=True)

    for idx, (img_path, label) in enumerate(test_samples):
        class_name = "cancer" if label == 1 else "no_cancer"
        output_path = os.path.join(test_dir, class_name, f"{idx}.jpg")
        shutil.copy(img_path, output_path)

    print(f"Balanced train/validate dataset saved to {train_val_dir}.")
    print(f"Test dataset saved to {test_dir}.")


def downloadAndPrep_mammogram_data():
    # Example usage
    dataset_name = "hongrui/mammogram_v_1"
    split = "train"
    output_dir = "./mammogram_dataset"
    train_val_dir = "./balanced_mammogram_dataset"
    test_dir = "./test_mammogram_dataset"

    # Download and save the raw dataset
    download_and_save_images(dataset_name, split, output_dir)

    # Preprocess and save the balanced dataset (choose undersampling or oversampling)
    preprocess_and_save_balanced_dataset_oversample(output_dir, train_val_dir, test_dir)
    # Or use undersampling:
    # preprocess_and_save_balanced_dataset(output_dir, train_val_dir, test_dir)



def preprocess_dataset(dataset_name, image_size=224, val_split=0.2, test_split=0.1):
    # Load the entire dataset (train split)
    dataset = load_dataset(dataset_name, split="train")
    processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")

    # Define preprocessing transforms
    transforms = Compose([
        Resize(image_size),
        CenterCrop(image_size),
        ToTensor(),
        Normalize(mean=processor.image_mean, std=processor.image_std),
        Grayscale(num_output_channels=3),  # Convert grayscale to RGB
    ])

    def transform_examples(example):
        img = example["image"]
        if isinstance(img, Image.Image):  # Ensure it's a PIL Image
            img = img.convert("RGB")  # Convert to RGB if needed
        example["pixel_values"] = transforms(img)
        return example

    # Apply preprocessing
    dataset = dataset.map(transform_examples, batched=False)
    dataset = dataset.remove_columns(["image"])

    # Split the dataset into train, validation, and test sets
    train_val_indices, test_indices = train_test_split(
        list(range(len(dataset))),
        test_size=test_split,
        stratify=dataset["label"],  # Ensure stratified split
        random_state=42,
    )
    train_indices, val_indices = train_test_split(
        train_val_indices,
        test_size=val_split / (1 - test_split),  # Adjust validation split relative to remaining data
        stratify=[dataset[i]["label"] for i in train_val_indices],
        random_state=42,
    )

    train_dataset = dataset.select(train_indices)
    val_dataset = dataset.select(val_indices)
    test_dataset = dataset.select(test_indices)

    return train_dataset, val_dataset, test_dataset


def preprocess_local_dataset(data_dir, image_size=224, val_split=0.2, test_split=0.1):
    # Define preprocessing transforms
    transforms = Compose([
        Resize(image_size),
        CenterCrop(image_size),
        ToTensor(),
        Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),  # Standard ImageNet normalization
    ])

    # Load the dataset using BUSIDataset
    dataset = BUSIDataset(root_dir=data_dir, transform=transforms)

    # Calculate split sizes
    test_size = int(len(dataset) * test_split)
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - test_size - val_size

    # Split the dataset
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    return train_dataset, val_dataset, test_dataset


class HuggingFaceDatasetWrapper(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        return {"pixel_values": image, "labels": label}


def train_model_google():
    # Path to the local dataset
    output_dir = "./mammogram_dataset"
    train_val_dir = "./balanced_mammogram_dataset"
    preprocess_and_save_balanced_dataset(output_dir, train_val_dir, test_split=0.2)


    # Preprocess and save the balanced dataset (choose undersampling or oversampling)
    # preprocess_and_save_balanced_dataset(output_dir, train_val_dir, test_dir)
    data_dir = train_val_dir

    # Preprocess the dataset
    train_dataset, val_dataset, test_dataset = preprocess_local_dataset(data_dir)

    # Wrap datasets for Hugging Face Trainer compatibility
    train_dataset = HuggingFaceDatasetWrapper(train_dataset)
    val_dataset = HuggingFaceDatasetWrapper(val_dataset)
    test_dataset = HuggingFaceDatasetWrapper(test_dataset)

    # Combine datasets into a DatasetDict-like structure
    dataset = {
        "train": train_dataset,
        "validation": val_dataset,
        "test": test_dataset,
    }

    # Load the model
    model = ViTForImageClassification.from_pretrained(
        "google/vit-base-patch16-224-in21k",
        num_labels=2,  # Binary classification: no_cancer, cancer
        id2label={0: "no_cancer", 1: "cancer"},
        label2id={"no_cancer": 0, "cancer": 1},
    )

    # Define metrics
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = torch.argmax(torch.tensor(logits), dim=-1)
        acc = accuracy_score(labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average="binary")
        return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}

    # Define training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        eval_strategy="epoch",  # Perform evaluation at the end of every epoch
        save_strategy="epoch",  # Save the model at the end of every epoch
        learning_rate=5e-5,
        per_device_train_batch_size=64,
        per_device_eval_batch_size=64,
        num_train_epochs=20,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=20,
        save_total_limit=2,
        load_best_model_at_end=True,  # Load the best model at the end of training
        metric_for_best_model="precision",  # Use F1-score to determine the best model
        gradient_accumulation_steps=4,
        fp16=True,  # Enable mixed precision training
        seed=42,
    )

    # Initialize the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        tokenizer=None,  # ViT does not require a tokenizer
        compute_metrics=compute_metrics,
    )

    # Train the model
    trainer.train()

    # Save the trained model
    trainer.save_model("./saved_model")

    # Evaluate the model on the test set
    metrics = trainer.evaluate(dataset["test"])
    print(metrics)


def evaluate_model_google():
    # Define preprocessing transforms (same as during training)
    preprocess = Compose([
        Resize(224),
        CenterCrop(224),
        ToTensor(),
        Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),  # Standard ImageNet normalization
    ])

    # Load the saved model
    model_path = "./saved_model"  # Path to the saved model directory
    model = ViTForImageClassification.from_pretrained(model_path)
    model.eval()  # Set the model to evaluation mode

    # Set the device (GPU if available, otherwise CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Define the classes
    classes = ["no_cancer", "cancer"]

    # Function to classify a single image
    def classify_image(image_path):
        # Load and preprocess the image
        img = Image.open(image_path).convert("RGB")
        img = preprocess(img).unsqueeze(0).to(device)  # Add batch dimension and move to device

        # Run inference
        with torch.no_grad():
            outputs = model(img)
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=1)

        # Get the predicted class and confidence
        predicted_class = classes[probabilities.argmax().item()]
        confidence = probabilities.max().item()

        return predicted_class, confidence

    def classify_images(folder_path):
        # Get all image files in the folder (supports .jpg, .jpeg, .png)
        image_paths = glob.glob(os.path.join(folder_path, "*.[jp][pn]g"))
        results = []
        correct = 0
        for image_path in image_paths:
            predicted_class, confidence = classify_image(image_path)
            if predicted_class == "no_cancer":
                correct += 1
            results.append((image_path, predicted_class, confidence))
            print(f"Image: {image_path}, Predicted class: {predicted_class}, Confidence: {confidence:.2f}")
        print(f"Accuracy: {correct / len(image_paths) * 100:.2f}%")
        return results

    # Path to the test dataset folder
    test_folder_path = "./mammogram_dataset/no_cancer"  # Update this path if needed

    # Run the model on the test dataset
    print(f"Classifying images in folder: {test_folder_path}")
    results = classify_images(test_folder_path)

def train_model_google_with_cv(k_folds=5):
    # Path to the local dataset
    data_dir = "./balanced_mammogram_dataset"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Preprocess the dataset
    full_dataset = BUSIDataset(root_dir=data_dir, transform=Compose([
        Resize(224),
        CenterCrop(224),
        ToTensor(),
        Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ]))

    # Initialize K-Fold Cross-Validation
    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)

    # Metrics to track performance across folds
    fold_metrics = []

    for fold, (train_idx, val_idx) in enumerate(kfold.split(full_dataset)):
        print(f"Training fold {fold + 1}/{k_folds}...")

        # Split the dataset into training and validation sets
        train_subset = Subset(full_dataset, train_idx)
        val_subset = Subset(full_dataset, val_idx)

        train_loader = DataLoader(train_subset, batch_size=64, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_subset, batch_size=64, shuffle=False, num_workers=4)

        # Load the model
        model = ViTForImageClassification.from_pretrained(
            "google/vit-base-patch16-224-in21k",
            num_labels=2,  # Binary classification: no_cancer, cancer
            id2label={0: "no_cancer", 1: "cancer"},
            label2id={"no_cancer": 0, "cancer": 1},
        )
        model.to(device)

        # Define optimizer and loss function
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
        criterion = torch.nn.CrossEntropyLoss()

        # Train the model for a few epochs
        for epoch in range(1, 6):  # Adjust the number of epochs as needed
            print(f"Fold {fold + 1}, Epoch {epoch}")
            train_one_epoch(model, train_loader, criterion, optimizer, device)

        # Evaluate the model on the validation set
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        print(f"Fold {fold + 1} — Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}")

        # Save metrics for this fold
        fold_metrics.append({"fold": fold + 1, "val_loss": val_loss, "val_acc": val_acc})

    # Print overall metrics
    avg_val_loss = sum(f["val_loss"] for f in fold_metrics) / k_folds
    avg_val_acc = sum(f["val_acc"] for f in fold_metrics) / k_folds
    print(f"Cross-Validation Complete — Avg Validation Loss: {avg_val_loss:.4f}, Avg Validation Accuracy: {avg_val_acc:.4f}")

if __name__ == "__main__":
    # train_model_google()
    # train_model_google_with_cv()
    evaluate_model_google()