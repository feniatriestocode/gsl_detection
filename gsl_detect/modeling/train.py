from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from loguru import logger
from tqdm import tqdm
import typer

from gsl_detect.config import MODELS_DIR, PROCESSED_DATA_DIR, RAW_DATA_DIR
from gsl_detect.modeling.dataset import GSLDataset, collate_fn
from gsl_detect.modeling.model import GSLTransformer
from torch.utils.data import DataLoader

app = typer.Typer()

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0

    for sequences, masks, labels in tqdm(loader, desc="Training", leave=False)
    sequences = sequences.to(device)
    masks     = masks.to(device)
    labels    = labels.to(device)

    optimizer.zero_grad()
    logits = model(sequences, masks)
    loss   = criterion(logits, labels)
    loss.backward()
    optimizer.step()

    total_loss += loss.item()
    return total_loss / len(loader)

def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct    = 0
    total      = 0

    with torch.no_grad():
        for sequences, masks, labels in tqdm(loader, desc="Validating", leave=False):
            sequences = sequences.to(device)
            masks     = masks.to(device)
            labels    = labels.to(device)

            logits = model(sequences, masks)
            loss   = criterion(logits, labels)

            total_loss += loss.item()
            preds       = logits.argmax(dim=1)
            correct    += (preds == labels).sum().item()
            total      += labels.size(0)

    avg_loss = total_loss / len(loader)
    accuracy = correct / total 
    return avg_loss, accuracy

@app.command()
def main(
    data_dir:   Path = PROCESSED_DATA_DIR,
    csv_path:   Path = PROCESSED_DATA_DIR / "isolated_GSL_corpus.csv",
    model_path: Path = MODELS_DIR / "best_model.pt",
    d_model:    int  = 256,
    nhead:      int  = 8,
    num_layers: int  = 4,
    batch_size: int  = 32,
    max_epochs:  int  = 200,
    patience:    int  = 10,   # early stopping patience
    lr:          float = 1e-4,
):

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    logger.info("Loading Datasets...")
    train_dataset = GSLDataset(csv_path, data_dir / "train")
    val_dataset   = GSLDataset(csv_path, data_dir / "val", label_encoder=train_dataset.label_encoder)
    train_loader  = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collatr_fn=collate_fn)
    val_loader    = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    num_classes = train_dataset.num_classes
    logger.info(f"Classes: {num_classes} | "
                f"Train: {len(train_dataset) | Val: {len(val_dataset)}}")
    
    model = GSLTransformer(
        input_dim=390,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=d_model * 4,
        num_classes=num_classes,        
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    writer = SummaryWriter(f"runs/layers{num_layers}_dmodel{d_model}_lr{lr}")
    best_val_loss = float("inf")
    epochs_no_improve = 0

    for epoch in range(max_epochs):
        logger.info(f"Epoch {epoch+1}/{max_epochs}")

        train_loss            = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc     = validate(model, val_loader, criterion, device)

        logger.info(f"Train loss: {train_loss:.4f} | "
                    f"Val loss: {val_loss:.4f} | Val acc: {val_acc:.2%}")

        writer.add_scalar("Loss/train",    train_loss, epoch)
        writer.add_scalar("Loss/val",      val_loss,   epoch)
        writer.add_scalar("Accuracy/val",  val_acc,    epoch)

        # ── Save best model & early stopping ─────────────────────────────
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), model_path)
            logger.success(f"New best model saved (val loss: {val_loss:.4f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                logger.warning(f"Early stopping at epoch {epoch+1}")
                break

    writer.close()
    logger.success("Training complete.")    

if __name__ == "__main__":
    app()
