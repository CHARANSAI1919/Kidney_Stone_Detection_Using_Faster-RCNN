from kidney_dataset import YoloToFasterRCNNDataset
from torch.utils.data import DataLoader
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torch

# Load dataset (make sure the paths point to your train/images and train/labels folders)
train_dataset = YoloToFasterRCNNDataset(
    "C:/Users/Charan Sai/AI_MODELS/FasterRcnn/KidneyStone/train/images",
    "C:/Users/Charan Sai/AI_MODELS/FasterRcnn/KidneyStone/train/labels"
)

# Debug: print number of samples
print(f"Found {len(train_dataset)} training samples.")

# DataLoader
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))

# Load pretrained Faster R-CNN model
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

# Modify the predictor head for 3 classes (background + 2 classes)
num_classes = 3  # 0 = background, 1 = kidney-stone, 2 = normal-kidney
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9)

# Training loop
for epoch in range(10):
    model.train()
    running_loss = 0.0
    for images, targets in train_loader:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        running_loss += losses.item()

    print(f"Epoch {epoch+1}, Loss: {running_loss:.4f}")

# Save the model
torch.save(model.state_dict(), "fasterrcnn_kidney.pth")
print("Training complete. Model saved as fasterrcnn_kidney.pth")
