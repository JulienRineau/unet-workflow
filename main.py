import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn

from unet import UNet


class UNETModel(pl.LightningModule):
    """
    Attributes:
        model (UNet): An instance of the U-Net model for segmentation.
        loss_fn: Dice loss function for binary segmentation.
    """

    def __init__(self):
        super().__init__()
        self.model = UNet()
        self.loss_fn = smp.losses.DiceLoss(
            mode=smp.losses.BINARY_MODE, from_logits=True
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Passes the input image through the model to predict the segmentation mask."""
        return self.model(image)

    def shared_step(self, batch: tuple[torch.Tensor, torch.Tensor], stage: str) -> dict:
        """A shared step for training, validation, and testing to handle the input batch and compute metrics."""
        images, masks = batch
        assert (
            images.ndim == 4 and images.shape[2] % 32 == 0 and images.shape[3] % 32 == 0
        )
        assert masks.ndim == 4 and masks.max() <= 1.0 and masks.min() >= 0

        logits_masks = self(images)
        loss = self.loss_fn(logits_masks, masks)
        probabilities_masks = logits_masks.sigmoid()
        predicted_masks = (probabilities_masks > 0.5).float()

        (
            true_positives,
            false_positives,
            false_negatives,
            true_negatives,
        ) = smp.metrics.get_stats(predicted_masks.long(), masks.long(), mode="binary")

        return {
            "loss": loss,
            "tp": true_positives,
            "fp": false_positives,
            "fn": false_negatives,
            "tn": true_negatives,
        }

    def shared_epoch_end(self, outputs: list[dict], stage: str):
        """Aggregates metrics and logs them at the end of each epoch for the given stage (train/valid/test)."""
        true_positives = torch.cat([x["tp"] for x in outputs])
        false_positives = torch.cat([x["fp"] for x in outputs])
        false_negatives = torch.cat([x["fn"] for x in outputs])
        true_negatives = torch.cat([x["tn"] for x in outputs])
        total_loss = sum(x["loss"].item() for x in outputs) / len(outputs)

        metrics = {
            f"{stage}_loss": total_loss,
            f"{stage}_precision": smp.metrics.precision(
                true_positives,
                false_positives,
                false_negatives,
                true_negatives,
                reduction="micro",
            ),
            f"{stage}_recall": smp.metrics.recall(
                true_positives,
                false_positives,
                false_negatives,
                true_negatives,
                reduction="micro",
            ),
            f"{stage}_accuracy": smp.metrics.accuracy(
                true_positives,
                false_positives,
                false_negatives,
                true_negatives,
                reduction="macro",
            ),
            f"{stage}_f1_score": smp.metrics.f1_score(
                true_positives,
                false_positives,
                false_negatives,
                true_negatives,
                reduction="micro",
            ),
            f"{stage}_per_image_iou": smp.metrics.iou_score(
                true_positives,
                false_positives,
                false_negatives,
                true_negatives,
                reduction="micro-imagewise",
            ),
            f"{stage}_dataset_iou": smp.metrics.iou_score(
                true_positives,
                false_positives,
                false_negatives,
                true_negatives,
                reduction="micro",
            ),
        }

        self.log_dict(metrics, prog_bar=True)

    def training_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> dict:
        return self.shared_step(batch, "train")

    def training_epoch_end(self, outputs: list[dict]):
        self.shared_epoch_end(outputs, "train")

    def validation_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> dict:
        return self.shared_step(batch, "valid")

    def validation_epoch_end(self, outputs: list[dict]):
        self.shared_epoch_end(outputs, "valid")

    def test_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> dict:
        return self.shared_step(batch, "test")

    def test_epoch_end(self, outputs: list[dict]):
        self.shared_epoch_end(outputs, "test")

    def configure_optimizers(self):
        """Defines the optimizer and learning rate for training."""
        return torch.optim.Adam(self.parameters(), lr=0.0001)
