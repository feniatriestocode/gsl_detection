import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from gsl_detect.augment import augment

class GSLDataset(Dataset):
    def __init__(self, csv_path, data_dir, label_encoder=None, training=False):
        self.data_dir = Path(data_dir)
        self.df = pd.read_csv(csv_path)
        self.training = training
        all_files_on_disk = {
            str(f.relative_to(self.data_dir).with_suffix(""))
            .replace('\\', '/')
            .lower()
            .strip()
            for f in self.data_dir.rglob("*.npz")
        }

        self.df["match_key"] = (
            self.df["Video"]
            .str.replace('\\', '/', regex=False)
            .str.lower()
            .str.strip()
        )

        self.df = self.df[
            self.df["match_key"].apply(lambda x: f"{x}_norm" in all_files_on_disk)
        ].reset_index(drop=True)

        self.df = self.df.drop(columns=["match_key"])

        print(f"--- MATCHING REPORT ---")
        print(f"Files found on disk: {len(all_files_on_disk)}")
        print(f"Samples in CSV: {len(self.df)}")

        if len(self.df) == 0:
            example_disk = list(all_files_on_disk)[0] if all_files_on_disk else "NONE"
            example_csv = "NONE" if self.df.empty else f"{self.df['Video'].iloc[0].lower().strip()}_norm"
            print(f"DISK EXAMPLE: '{example_disk}'")
            print(f"CSV  EXAMPLE: '{example_csv}'")

        # Label encoding
        if label_encoder is None:
            self.label_encoder = LabelEncoder()
            self.label_encoder.fit(self.df["Annotation_Greeklish"])  # adjust column name if needed
        else:
            self.label_encoder = label_encoder

        self.labels = self.label_encoder.transform(self.df["Annotation_Greeklish"])


    def __getitem__(self, idx):
        video_rel = self.df.iloc[idx]["Video"]
        npz_path = self.data_dir / f"{video_rel}_norm.npz"
        data = np.load(npz_path)
        # sequence = torch.tensor(data["sequence"], dtype=torch.float32)
        sequence = data["sequence"]
        mask = torch.tensor(data["mask"], dtype=torch.bool)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        if self.training:
            sequence = augment(sequence, p=0.5)
        sequence = torch.tensor(sequence, dtype=torch.float32)
        return sequence, mask, label
    
    def __len__(self):
        return len(self.df)

    @property
    def num_classes(self):
        return len(self.label_encoder.classes_)
    
    @property
    def label_to_gloss(self):
        return self.label_encoder.classes_
  

def get_dataloaders(csv_path, processed_dir, batch_size=32):
    train_dir = processed_dir / "train"
    val_dir   = processed_dir / "val"
    test_dir  = processed_dir / "test"

    # Fit label encoder on training set only
    train_dataset = GSLDataset(csv_path, train_dir, training=True)
    val_dataset   = GSLDataset(csv_path, val_dir,   label_encoder=train_dataset.label_encoder, training=False)
    test_dataset  = GSLDataset(csv_path, test_dir,  label_encoder=train_dataset.label_encoder, training=False)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  collate_fn=collate_fn)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    return train_loader, val_loader, test_loader, train_dataset.num_classes


def collate_fn(batch):
    sequences, masks, labels = zip(*batch)

    max_len = max(s.shape[0] for s in sequences)
    feat_dim = sequences[0].shape[1]

    padded_sequences = torch.zeros(len(sequences), max_len, feat_dim)
    padded_masks     = torch.zeros(len(sequences), max_len, dtype=torch.bool)

    for i, (seq, mask) in enumerate(zip(sequences, masks)):
        seq_len = seq.shape[0]
        padded_sequences[i, :seq_len] = seq
        padded_masks[i, :seq_len]     = mask

    labels = torch.stack(labels)

    return padded_sequences, padded_masks, labels