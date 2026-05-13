from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score
from loguru import logger
from tqdm import tqdm
import typer

from gsl_detect.config import MODELS_DIR, PROCESSED_DATA_DIR, RAW_DATA_DIR
from gsl_detect.modeling.dataset import GSLDataset, collate_fn
from gsl_detect.modeling.model import GSLTransformer
from torch.utils.data import DataLoader

app = typer.Typer()

def train_epoch(model, loader, optimizer, criterion, device, acc_metric):
    model.train()
    acc_metric.reset()
    total_loss = 0

    for sequences, masks, labels in tqdm(loader, desc="Training", leave=False):
        sequences = sequences.to(device)
        masks     = masks.to(device)
        labels    = labels.to(device)

        optimizer.zero_grad()
        logits = model(sequences, masks)
        loss   = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        acc_metric.update(logits, labels)

        # avg_loss = total_loss / len(loader)
    return total_loss / len(loader), acc_metric.compute()

def validate(model, loader, criterion, device, acc_metric, f1_metric):
    model.eval()
    total_loss = 0
    acc_metric.reset()
    f1_metric.reset()

    with torch.no_grad():
        for sequences, masks, labels in tqdm(loader, desc="Validating", leave=False):
            sequences = sequences.to(device)
            masks     = masks.to(device)
            labels    = labels.to(device)

            logits = model(sequences, masks)
            loss   = criterion(logits, labels)

            total_loss += loss.item()
            acc_metric.update(logits, labels)
            f1_metric.update(logits, labels)

    return total_loss / len(loader), acc_metric.compute(), f1_metric.compute()

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
    train_loader  = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader    = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    num_classes = train_dataset.num_classes
    logger.info(f"Classes: {num_classes} | "
                f"Train: {len(train_dataset)} | Val: {len(val_dataset)}")
    
    model = GSLTransformer(
        input_dim=318,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=d_model * 4,
        num_classes=num_classes,        
    ).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    train_acc_metric = MulticlassAccuracy(num_classes=num_classes).to(device)
    val_acc_metric   = MulticlassAccuracy(num_classes=num_classes).to(device)
    val_f1_metric    = MulticlassF1Score(num_classes=num_classes, average='weighted').to(device)


    writer = SummaryWriter(f"runs/layers{num_layers}_dmodel{d_model}_lr{lr}")
    best_val_loss = float("inf")
    epochs_no_improve = 0

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.info("="*30)
    logger.info("MODEL CHARACTERISTICS")
    logger.info(f"• Input Dimension:    {318}")
    logger.info(f"• d_model:            {d_model}")
    logger.info(f"• nhead:              {nhead}")
    logger.info(f"• num_layers:         {num_layers}")
    logger.info(f"• dim_feedforward:    {d_model * 4}")
    logger.info(f"• num_classes:        {num_classes}")
    logger.info(f"• Trainable Params:   {total_params:,}")
    logger.info(f"• Batch Size:         {batch_size}")
    logger.info(f"• Learning Rate:      {lr}")
    logger.info("="*30)

    for epoch in range(max_epochs):
        logger.info(f"Epoch {epoch+1}/{max_epochs}")

        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device, train_acc_metric)
        val_loss, val_acc, val_f1 = validate(model, val_loader, criterion, device, val_acc_metric, val_f1_metric)
        # scheduler.step(val_loss)

        logger.info(f"Train loss: {train_loss:.4f} | "
                    f"Val loss: {val_loss:.4f} | Val acc: {val_acc:.2%}")

        writer.add_scalar("Loss/train",     train_loss, epoch)
        writer.add_scalar("Loss/val",       val_loss,   epoch)
        writer.add_scalar("Accuracy/train", train_acc,  epoch)
        writer.add_scalar("Accuracy/val",   val_acc,    epoch)
        writer.add_scalar("F1/val",         val_f1,     epoch)
        logger.info(f"Epoch {epoch+1} | Train Acc: {train_acc:.2%} | Val Acc: {val_acc:.2%}")
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
