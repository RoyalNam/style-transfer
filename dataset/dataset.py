import os
from PIL import Image
from torch.utils.data import Dataset


class ImageDataset(Dataset):
    def __init__(self, dirA, dirB, transform=None):
        self.dirA = dirA
        self.dirB = dirB
        self.transform = transform

        self.pathsA = os.listdir(dirA)
        self.pathsB = os.listdir(dirB)

    def __len__(self):
        return max(len(self.pathsA), len(self.pathsB))

    def __getitem__(self, index):
        pathA = os.path.join(self.dirA, self.pathsA[index % len(self.pathsA)])
        pathB = os.path.join(self.dirB, self.pathsB[index % len(self.pathsB)])
        imgA = Image.open(pathA).convert('RGB')
        imgB = Image.open(pathB).convert('RGB')
        if self.transform:
            imgA = self.transform(imgA)
            imgB = self.transform(imgB)

        return imgA, imgB
