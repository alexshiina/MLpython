import torch
import torchvision
from torch import nn
from torchvision import transforms

def Vit_b16_model(num_classes: int =43):
  """Creates an vit_b16 model for my gradio demo"""
  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  weights = torchvision.models.ViT_B_16_Weights.DEFAULT
  transform = torchvision.transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

  model = torchvision.models.vit_b_16(weights = weights).to(device)
  for param in model.parameters():
    param.requires_grad = False

  model.heads = torch.nn.Sequential(
    torch.nn.Linear(768, 512),
    torch.nn.ReLU(),
    torch.nn.Dropout(0.5),
    torch.nn.Linear(512, 43)
)
  return model,transform