import os
import pickle
import torch
from torch.utils.data import Dataset


class CustomPickleDataset(Dataset):

    def __init__(self, annotations_file, img_dir, target_transform=None, device="cpu"):
        self.annotations = annotations_file
        self.img_dir = img_dir
        self.target_transform = target_transform
        self.device = device

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        row = self.annotations.iloc[idx]
        # join as directory path
        img_path = os.path.join(self.img_dir, row["image_name"] + ".pkl")
        # read pickle file
        with open(img_path, "rb") as f:
            image = pickle.load(f)

        # convert to tensor from 128x128 to 1x128x128
        image = torch.tensor(image).unsqueeze(0).to(self.device)
        # print(image.shape)
        label = row["LABEL"]
        original_sample = row["original_sample"]
        if self.target_transform:
            label = self.target_transform(label)
        return image, label, original_sample


class CustomClassicDataset(Dataset):

    def __init__(
        self, annotations_file, tag, transform=None, target_transform=None, device="cpu"
    ):
        self.annotations = annotations_file
        # self.transform = transform
        self.target_transform = target_transform
        self.device = device
        self.tag = tag

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        row = self.annotations.iloc[idx]
        features = row.drop(["original_sample", "LABEL"]).values
        label = row["LABEL"]
        original_sample = row["original_sample"]
        if self.target_transform:
            label = self.target_transform(label)

        features = torch.tensor([features], dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.long)

        if self.tag == "cnn":
            return features, label, original_sample
        elif self.tag == "nn":
            return features[0], label, original_sample
