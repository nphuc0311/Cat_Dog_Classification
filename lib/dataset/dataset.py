import os

from PIL import Image
from torch.utils.data import Dataset

class CatDogDataset(Dataset):
  def __init__(self, data_dir, transforms=None):
    self.data_dir = data_dir
    self.transforms = transforms
    self.image_paths = []
    self.labels = []

    self.class_names = os.listdir(data_dir)

    self.label_map = {class_name: i for i, class_name in enumerate(self.class_names)}

    for class_name in self.class_names:
      class_dir = os.path.join(data_dir, class_name)
      for image_name in os.listdir(class_dir):
        if image_name.endswith(('jpg', 'jpeg', 'png')):
          image_path = os.path.join(class_dir, image_name)
          self.image_paths.append(image_path)
          self.labels.append(self.label_map[class_name])

  def __len__(self):
    return len(self.image_paths)


  def __getitem__(self, idx):
    image_path = self.image_paths[idx]
    label = self.labels[idx]

    image = Image.open(image_path).convert("RGB")

    if self.transforms:
      image = self.transforms(image)

    return image, label