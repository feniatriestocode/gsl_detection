from pathlib import Path
from gsl_detect.config import FIGURES_DIR
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import matplotlib.pyplot as plt

# ea = EventAccumulator("home/fenia/Desktop/Engineering/10/Personal project/gsl_detection/gsl_detect/modeling/runs/layers3_dmodel256_lr0.0001_b64_ls01")
ea = EventAccumulator(str(Path(__file__).parent / "modeling" / "runs" / "layers3_dmodel256_lr0.0001_b64_ls01"))
ea.Reload()

# Extract scalars
val_acc  = ea.Scalars("Accuracy/val")
val_loss  = ea.Scalars("Loss/val")
# val_top5 = ea.Scalars("Accuracy/val_top5")
train_acc = ea.Scalars("Accuracy/train")
train_loss = ea.Scalars("Loss/train")

steps     = [x.step for x in val_acc]
val_vals  = [x.value for x in val_acc]
steps_loss     = [x.step for x in val_loss]
val_vals_loss  = [x.value for x in val_loss]
# top5_vals = [x.value for x in val_top5]
train_vals = [x.value for x in train_acc]
train_vals_loss = [x.value for x in train_loss]

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
})

plt.figure(figsize=(8, 4))
plt.plot(steps, train_vals, label="Train Acc")
plt.plot(steps, val_vals,   label="Val Acc")
# plt.plot(steps, top5_vals,  label="Val Top-5 Acc")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title(r"GSL Transformer — Accuracy — $d_{\mathrm{model}} = 256$ — $batch = 256$ - $ls = 0.1$")
plt.legend()
plt.tight_layout()
plt.savefig("accuracy_curve_b32_ls00_256_aug_b64_ls01.pdf", dpi=300)  # PDF for LaTeX, or PNG for Word


plt.figure(figsize=(8, 4))
plt.plot(steps_loss, train_vals_loss, label="Train Loss")
plt.plot(steps_loss, val_vals_loss,   label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("GSL Transformer — Loss - r$d_{\rm{model}} = 256$ - $batch = 256$ - $ls = 0.1$")

plt.legend()
plt.tight_layout()
plt.savefig("loss_curve_b32_ls00_256_aug_b64_ls01.pdf", dpi=300)