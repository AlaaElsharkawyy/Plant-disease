import sys



import torch.nn as nn
from torchvision import models
print(sys.executable)
import torch

print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU Name:", torch.cuda.get_device_name(0))
    x = torch.rand(3,3).cuda()
    print("Tensor on GPU:", x)
import os
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader
#from torchvision import transforms, models Preprocessing + Transformations
from torchvision import transforms, models  # Preprocessing + Transformations



from torchvision import transforms

transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

transform_val = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
class PlantDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        self.image_paths = []
        self.plant_labels = []
        self.disease_labels = []

        plants = set()
        diseases = set()

        for class_name in os.listdir(root_dir):
            class_path = os.path.join(root_dir, class_name)
            if not os.path.isdir(class_path):
                continue

            plant, disease = class_name.split("___")

            plants.add(plant)
            diseases.add(disease)

            for img in os.listdir(class_path):
                self.image_paths.append(os.path.join(class_path, img))
                self.plant_labels.append(plant)
                self.disease_labels.append(disease)

        self.plant_to_idx = {p:i for i,p in enumerate(sorted(plants))}
        self.disease_to_idx = {d:i for i,d in enumerate(sorted(diseases))}

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")

        plant = self.plant_to_idx[self.plant_labels[idx]]
        disease = self.disease_to_idx[self.disease_labels[idx]]

        if self.transform:
            img = self.transform(img)

        return img, torch.tensor(plant), torch.tensor(disease)
base_path = r"D:\deep_projects\PLANET_DISESES\dataset_split"

train_dataset = PlantDataset(
    root_dir=os.path.join(base_path, "train"),
    transform=transform_train
)

val_dataset = PlantDataset(
    root_dir=os.path.join(base_path, "val"),
    transform=transform_val
)

test_dataset = PlantDataset(
    root_dir=os.path.join(base_path, "test"),
    transform=transform_val
)
from torch.utils.data import DataLoader

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
print(len(train_dataset))
print(len(val_dataset))
print(len(test_dataset))

print(train_dataset.plant_to_idx)
print(train_dataset.disease_to_idx)
class PlantModel(nn.Module):
    def __init__(self, num_plants, num_diseases):
        super().__init__()

        base = models.resnet18(pretrained=True)
        self.backbone = nn.Sequential(*list(base.children())[:-1])

        self.fc = nn.Linear(512, 256)

        self.plant_head = nn.Linear(256, num_plants)
        self.disease_head = nn.Linear(256, num_diseases)

    def forward(self, x):
        x = self.backbone(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        plant_out = self.plant_head(x)
        disease_out = self.disease_head(x)

        return plant_out, disease_out
import torch

print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
model = PlantModel(
    num_plants=len(train_dataset.plant_to_idx),
    num_diseases=len(train_dataset.disease_to_idx)
).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
print(next(model.parameters()).device)
for epoch in range(10):
    model.train()
    total_loss = 0

    for imgs, plant_labels, disease_labels in train_loader:
        imgs = imgs.to(device)
        plant_labels = plant_labels.to(device)
        disease_labels = disease_labels.to(device)

        plant_out, disease_out = model(imgs)

        loss1 = criterion(plant_out, plant_labels)
        loss2 = criterion(disease_out, disease_labels)

        loss = loss1 + loss2

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch [{epoch+1}/10] Loss: {total_loss:.4f}")
def accuracy(preds, labels):
    _, predicted = torch.max(preds, 1)
    return (predicted == labels).sum().item() / labels.size(0)


model.eval()
with torch.no_grad():
    plant_acc = 0
    disease_acc = 0
    n = 0

    for imgs, plant_labels, disease_labels in val_loader:
        imgs = imgs.to(device)
        plant_labels = plant_labels.to(device)
        disease_labels = disease_labels.to(device)

        plant_out, disease_out = model(imgs)

        plant_acc += accuracy(plant_out, plant_labels)
        disease_acc += accuracy(disease_out, disease_labels)
        n += 1

    print("Plant Accuracy:", plant_acc / n)
    print("Disease Accuracy:", disease_acc / n)
def predict(image_path, model):
    model.eval()

    img = Image.open(image_path).convert("RGB")
    img = transform_val(img).unsqueeze(0).to(device)

    with torch.no_grad():
        plant_out, disease_out = model(img)

        plant_pred = torch.argmax(plant_out, 1).item()
        disease_pred = torch.argmax(disease_out, 1).item()

    return plant_pred, disease_pred
torch.save(model.state_dict(), "model.pth")