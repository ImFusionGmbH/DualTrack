import abc
from dataclasses import dataclass, field
import os
from typing import Callable, Literal
import h5py
from matplotlib import pyplot as plt
from tqdm import tqdm
from src.engine.tracking_estimator import BaseTrackingEstimator
from src.logger import Logger
import torch
from src.evaluator import TrackingEstimationEvaluator
import logging
from torch import clip_, nn
import pandas as pd
import time
from pathlib import Path
import logging
from src.utils.pose import get_global_and_relative_pred_trackings_from_vectors


class RelativeTrackingPredictionLogic(abc.ABC):
    @abc.abstractmethod
    def __call__(self, batch, model, device) -> torch.Tensor:
        """Takes a batch of data and returns predicted relative tracking.

        In other words, the output of the function should be 1 x (N - 1) x 6 dimensional
        relative tracking vectors (where N is the length of the true tracking sequence)
        """


class LossCalculationLogic(abc.ABC):
    """Defines the interface of a loss function suitable for the run_training_loop function"""

    @abc.abstractmethod
    def __call__(self, batch, model, device) -> torch.Tensor: ...


def default_pred_fn(batch, model, device):
    return model(batch["images"].to(device))


def default_loss_fn(batch, model, device, pred=None):

    if hasattr(model, "get_loss"):
        return model.get_loss(batch, device, pred)

    pred = pred if pred is not None else model(batch["images"].to(device))
    target = batch["targets"].to(device)
    return torch.nn.functional.mse_loss(pred, target)


@torch.no_grad()
def run_full_evaluation_loop(
    model: nn.Module,
    loader,
    pred_logic: (
        RelativeTrackingPredictionLogic | Callable[[dict, nn.Module, str], torch.Tensor]
    ) = None,
    loss_fn=None,
    device="cuda",
    suffix="/val",
    use_bfloat=False,
    use_amp=False,
    logger: Logger | None = None,
    epoch=None,
    log_metrics=False,
    log_image_indices=[],
    **evaluator_kw,
):
    evaluator = TrackingEstimationEvaluator(
        image_shape_hw=loader.dataset[0]["original_image_shape"],
        **evaluator_kw,
    )

    model.eval()
    total_items = 0
    for i, data in enumerate(tqdm(loader)):
        with torch.autocast(
            torch.device(device).type,
            torch.bfloat16 if use_bfloat else torch.float16,
            enabled=use_amp,
        ):
            if pred_logic:
                pred = pred_logic(data, model, device)
            else:
                assert isinstance(model, BaseTrackingEstimator)
                pred = model.predict(data)

            if loss_fn is not None:
                loss = loss_fn(data, model, device, pred=pred)
                evaluator.add_metric("loss", loss.item())
            elif isinstance(model, BaseTrackingEstimator):
                loss = model.get_loss(data, pred=pred)
                evaluator.add_metric("loss", loss.item())

        for item_idx in range(len(data["tracking"])):
            if isinstance(pred, torch.Tensor):
                pred_i = pred[item_idx].float().cpu().numpy()
                if "padding_size" in data:
                    pred_i = pred_i[: len(pred_i) - data["padding_size"][item_idx]]
                evaluator.set_current_pred_tracking_from_relative_pose_vector(
                    pred_i
                ).set_current_gt_tracking_from_world(
                    data["tracking"][item_idx], data["calibration"][item_idx]
                )
            elif isinstance(pred, dict):
                pred_loc = pred["local"]
                pred_loc_i = pred_loc[item_idx].float().cpu().numpy()
                pred_glob = pred["global"]
                pred_glob_i = pred_glob[item_idx].float().cpu().numpy()
                if "padding_size" in data:
                    pred_glob_i = pred_glob_i[
                        : len(pred_glob_i) - data["padding_size"][item_idx]
                    ]
                    pred_loc_i = pred_loc_i[
                        : len(pred_loc_i) - data["padding_size"][item_idx]
                    ]
                evaluator.set_current_pred_tracking_from_relative_pose_vector(
                    pred_loc_i, False
                )
                evaluator.set_current_pred_tracking_from_global_pose_vectors(
                    pred_glob_i
                )
                evaluator.set_current_gt_tracking_from_world(
                    data["tracking"][item_idx], data["calibration"][item_idx]
                )
            else:
                raise ValueError(f"Unexpected prediction output {pred}")
            metrics, figures = evaluator.complete_update(
                include_images=(logger is not None and total_items in log_image_indices)
            )
            if figures and (logger is not None):
                logger.log(
                    logger.add_suffix(figures, f"{suffix}/figure_{total_items}"),
                    epoch,
                )
                plt.close("all")
            total_items += 1

    metrics = evaluator.aggregate()
    if log_metrics and logger is not None:
        logger.log(logger.add_suffix(metrics, suffix), epoch)

    return metrics


def run_training_one_epoch(
    model: nn.Module,
    loader,
    optimizer,
    scheduler,
    scaler,
    loss_fn: (
        LossCalculationLogic | Callable[[dict, nn.Module, str], torch.Tensor] | None
    ) = None,
    logger=None,
    device="cuda",
    epoch=None,
    use_bfloat=False,
    use_amp=False,
    scheduler_mode: Literal["epoch", "step"] = "epoch",
    max_iter_per_epoch: int | None = None,
    clip_grad_norm=None,
    clip_grad_value=None,
):
    assert scheduler_mode in ["epoch", "step"]

    model.train()
    total_loss = 0

    with tqdm(loader, desc=f"Training epoch {epoch}") as pbar:
        for i, data in enumerate(pbar):
            if max_iter_per_epoch is not None and i >= max_iter_per_epoch:
                break

            with torch.autocast(
                torch.device(device).type,
                torch.bfloat16 if use_bfloat else torch.float16,
                enabled=use_amp,
            ):
                if loss_fn:
                    loss = loss_fn(data, model, device)
                else:
                    assert isinstance(model, BaseTrackingEstimator)
                    loss = model.get_loss(data)

            optimizer.zero_grad()

            scaler.scale(loss).backward()
            if clip_grad_value or clip_grad_norm:
                scaler.unscale_(optimizer)
                if clip_grad_value:
                    torch.nn.utils.clip_grad_value_(model.parameters(), clip_grad_value)
                elif clip_grad_norm:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)

            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()

            if scheduler_mode == "step":
                scheduler.step()
                if logger is not None:
                    logger.log(
                        {
                            f"lr_{i}": scheduler.get_last_lr()[i]
                            for i in range(len(scheduler.get_last_lr()))
                        },
                        epoch,
                    )

            if logger is not None:
                logger.log({"loss-step/train": loss.item()}, epoch)

            if i % 10 == 0:
                pbar.set_postfix(
                    {
                        "loss": total_loss / (i + 1),
                        "mem": f"{torch.cuda.max_memory_reserved() / 1e9:.2f}GB",
                    }
                )

    if logger is not None:
        logger.log(
            {"loss/train": total_loss / len(loader)},
            epoch,
        )

    if scheduler_mode == "epoch":
        scheduler.step()
        if logger is not None:
            logger.log(
                {
                    f"lr_{i}": scheduler.get_last_lr()[i]
                    for i in range(len(scheduler.get_last_lr()))
                },
                epoch,
            )


def run_validation_loss_loop(
    model: nn.Module,
    loader,
    loss_fn: (
        LossCalculationLogic | Callable[[dict, nn.Module, str], torch.Tensor] | None
    ) = None,
    logger=None,
    device="cuda",
    epoch=None,
    use_bfloat=False,
    use_amp=False,
):
    model.eval()
    total_loss = 0
    for i, batch in enumerate(tqdm(loader, desc=f"Validation epoch {epoch}")):

        with torch.autocast(
            torch.device(device).type,
            torch.bfloat16 if use_bfloat else torch.float16,
            enabled=use_amp,
        ):
            if loss_fn: 
                loss = loss_fn(batch, model, device)
            else: 
                assert isinstance(model, BaseTrackingEstimator)
                loss = model.get_loss(batch)

        total_loss += loss.item()
        if logger is not None:
            logger.log({"loss-step/val": loss.item()}, epoch)

    if logger is not None:
        logger.log({"loss/val": total_loss / len(loader)}, epoch)

    return total_loss / len(loader)


@torch.no_grad()
def run_full_test_loop(
    model: nn.Module,
    loader,
    pred_logic: (
        RelativeTrackingPredictionLogic
        | Callable[[dict, nn.Module, str], torch.Tensor]
        | None
    ) = None,
    output_dir: Path = Path("test_output"),
    device="cuda",
    use_bfloat=False,
    use_amp=False,
    sweep_ids: list[str] | None = None,
    save_predictions: bool = False,
    save_images_with_predictions: bool = False,
    images_key_for_save="images",
    **evaluator_kw,
) -> pd.DataFrame:
    """Run a full test loop saving predictions, metrics and visualizations.

    Args:
        model: The model to evaluate
        loader: DataLoader containing the test dataset
        pred_logic: Logic for getting predictions from the model
        output_dir: Directory to save results
        device: Device to run evaluation on
        use_bfloat: Whether to use bfloat16 precision
        use_amp: Whether to use automatic mixed precision
        sweep_ids: Optional list of sweep IDs to filter evaluation on
        **evaluator_kw: Additional kwargs passed to TrackingEstimationEvaluator

    Returns:
        DataFrame containing evaluation metrics for each sweep
    """
    # Setup output directories
    output_dir.mkdir(exist_ok=True, parents=True)
    (output_dir / "scans").mkdir(exist_ok=True, parents=True)

    # Initialize evaluator
    evaluator = TrackingEstimationEvaluator(
        image_shape_hw=loader.dataset[0]["original_image_shape"],
        include_images=True,
        **evaluator_kw,
    )

    # Set up output file for predictions
    predictions_output_file = h5py.File(output_dir / "predictions.h5", "w")

    model.eval()
    predictions_table = []
    max_mem = 0

    for batch in tqdm(loader, desc="Evaluating"):
        # Filter by sweep IDs if specified
        if sweep_ids is not None:
            if not list(
                filter(lambda id: id.lower() in batch["sweep_id"][0].lower(), sweep_ids)
            ):
                continue

        sweep_dir = output_dir / "scans" / batch["sweep_id"][0]
        sweep_dir.mkdir(exist_ok=True, parents=True)

        # Get predictions
        t0 = time.time()
        with torch.autocast(
            torch.device(device).type,
            torch.bfloat16 if use_bfloat else torch.float16,
            enabled=use_amp,
        ):
            if pred_logic:
                pred = pred_logic(batch, model, device)
            else:
                assert isinstance(model, BaseTrackingEstimator)
                pred = model.predict(batch)

        # Update evaluator with predictions
        torch.cuda.synchronize()
        inference_time = time.time() - t0
        max_mem = max(max_mem, torch.cuda.max_memory_reserved())

        evaluator.set_current_pred_tracking_from_relative_pose_vector(
            pred[0].float().cpu().numpy()
        ).set_current_gt_tracking_from_world(
            batch["tracking"][0], batch["calibration"][0]
        )

        # Update predictions file & save ground truth
        if save_predictions:
            images = batch[images_key_for_save][0]
            spacing = batch["spacing"][0]
            dimensions = batch["dimensions"][0]
            targets = batch["targets"][0]

            pred_tracking_sequence = (
                get_global_and_relative_pred_trackings_from_vectors(
                    pred[0].cpu().numpy()
                )[0]
            )
            gt_tracking_sequence = get_global_and_relative_pred_trackings_from_vectors(
                targets.cpu().numpy()
            )[0]

            with h5py.File(str(sweep_dir / "export.h5"), "a") as f:
                if save_images_with_predictions:
                    f["images"] = images
                f["spacing"] = spacing
                f["dimensions"] = dimensions
                f["gt_tracking"] = gt_tracking_sequence
                f["pred_tracking"] = pred_tracking_sequence

        # Get metrics and figures
        metrics, figures = evaluator.complete_update(include_images=True)
        metrics["inference_time"] = inference_time

        # Save figures
        for name, figure in figures.items():
            figure.savefig(str(sweep_dir / f"{name}.png"))
            plt.close(figure)

        # Record metrics
        predictions_table_row = {
            "sweep_id": batch["sweep_id"][0],
            "raw_sweep_path": batch["raw_sweep_path"][0],
            **metrics,
        }
        predictions_table.append(predictions_table_row)

        # Save metrics at each sweep so it updates as we evaluate
        results_df = pd.DataFrame(predictions_table)
        results_df.to_csv(output_dir / "metrics.csv", index=False)

    metrics = {}
    metrics["max_mem"] = max_mem

    # Save average metrics
    avg_metrics = results_df.drop(["sweep_id", "raw_sweep_path"], axis="columns").mean()
    avg_metrics.to_string(open(output_dir / "avg_metrics.txt", "w"))
    print("Average metrics:", avg_metrics)
    metrics.update(avg_metrics.to_dict())
    print(
        f"mem: {torch.cuda.max_memory_reserved() / 1e9:.2f}GB",
        open(output_dir / "stats.txt", "a"),
    )

    if predictions_output_file is not None:
        predictions_output_file.close()

    return metrics


class Callback:
    def on_best_score(self, score): ...

    def on_epoch_start(self, epoch): ...


def run_training(
    model,
    train_loader,
    val_loader,
    optimizer,
    scheduler,
    logger,
    epochs,
    scaler=None,
    pred_fn=None,
    loss_fn=None,
    best_score=float("inf"),
    start_epoch=0,
    use_amp=False,
    use_bfloat=False,
    device="cuda",
    tracked_metric="ddf/5pt-avg_global_displacement_error",
    validation_mode: Literal["loss", "full"] = "full",
    validate_every_n_epochs=10,
    callbacks: list[Callback] = [],
    clip_grad_norm=None,
    clip_grad_value=None,
    evaluator_kw={},
    log_image_indices=[],
):

    if scaler is None:
        scaler = torch.GradScaler(enabled=use_amp)

    def get_state():
        return {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "best_score": best_score,
            "epoch": epoch,
        }

    for epoch in range(start_epoch, epochs):
        logging.info(f"Epoch {epoch}")

        for callback in callbacks:
            callback.on_epoch_start(epoch)

        # save checkpoint
        logger.save_checkpoint(get_state())

        run_training_one_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            scaler,
            loss_fn,
            logger,
            device,
            epoch,
            scheduler_mode="step",
            use_amp=use_amp,
            use_bfloat=use_bfloat,
            clip_grad_norm=clip_grad_norm,
            clip_grad_value=clip_grad_value,
        )

        if (epoch + 1) % validate_every_n_epochs == 0:
            if validation_mode == "full":
                metrics = run_full_evaluation_loop(
                    model,
                    val_loader,
                    pred_fn,
                    loss_fn,
                    device,
                    logger=logger,
                    epoch=epoch,
                    log_metrics=True,
                    use_amp=use_amp,
                    use_bfloat=use_bfloat,
                    **evaluator_kw,
                    log_image_indices=log_image_indices,
                )
                if (m := metrics[tracked_metric]) < best_score:
                    logging.info(
                        f"Best metric ({tracked_metric}:{m:.2f}) observed - saving"
                    )
                    best_score = metrics[tracked_metric]
                    logger.save_checkpoint(get_state(), "best.pt")
                    for callback in callbacks:
                        callback.on_best_score(best_score)
            else:
                loss = run_validation_loss_loop(
                    model,
                    val_loader,
                    loss_fn,
                    logger,
                    device,
                    epoch,
                    use_bfloat=use_bfloat,
                    use_amp=use_amp,
                )
                if loss < best_score:
                    logging.info(f"Best metric (val_loss:{loss:.6f}) observed - saving")

                    best_score = loss
                    logger.save_checkpoint(get_state(), "best.pt")
                    for callback in callbacks:
                        callback.on_best_score(best_score)


@torch.no_grad()
def export_features(
    model,
    loader,
    output_filename,
    device="cuda",
    compute_features_fn=lambda batch, model, device: model(batch["images"].to(device)),
    dry_run=False,
):
    model.eval().to(device)

    os.makedirs(os.path.dirname(os.path.abspath(output_filename)), exist_ok=True)

    with h5py.File(output_filename, "a") as f:
        for batch in tqdm(loader):
            sweep_id = batch["sweep_id"][0]
            feats = compute_features_fn(batch, model, device)
            feats = feats.cpu().numpy()[0]
            if dry_run:
                print(output_filename, sweep_id, feats.shape)
            else:
                f[sweep_id] = feats


@dataclass
class TrainLogic:
    pred_fn: Callable
    std_loss: bool = False
    std_loss_weight: float = 0.1

    # def pred_fn(self, batch, model, device):
    #     inputs = dict(
    #         context_image_features=batch["context_image_features"].to(device),
    #         features=batch["local_features"].to(device),
    #     )
    #     return model(**inputs)

    def loss_fn(self, batch, model, device, pred=None):
        pred = pred if pred is not None else self.pred_fn(batch, model, device)

        targets = batch["targets"].to(device)
        padding_lengths = batch["padding_size"]

        mse_loss = nn.MSELoss(reduction="none")

        def _get_loss(pred, targets):
            B, N, D = (
                pred.shape
                if isinstance(pred, torch.Tensor)
                else list(pred.values())[0].shape
            )

            mask = torch.ones(B, N, D, dtype=torch.bool, device=device)
            for i, padding_length in enumerate(padding_lengths):
                if padding_length > 0:
                    mask[i, -padding_length:, :] = 0

            loss = mse_loss(pred, targets)
            masked_loss = torch.where(mask, loss, torch.nan)
            mse_loss_val = masked_loss.nanmean()

            masked_pred = torch.where(mask, pred, torch.nan)
            if self.std_loss:
                std_loss = torch.tensor(0.0, device=device)
                for pred_i, padding_length_i in zip(pred, padding_lengths):
                    if padding_length_i > 1:
                        pred_i = pred_i[:-padding_length_i]
                    std_loss += torch.std(pred_i, dim=0).mean()
                return mse_loss_val + std_loss * self.std_loss_weight

            return mse_loss_val

        if isinstance(pred, dict):
            loss = torch.tensor(0.0, device=device)
            if "local" in pred:
                loss += _get_loss(pred["local"], targets)
            if "global" in pred:
                loss += _get_loss(pred["global"], batch["targets_global"].to(device))
            if "absolute" in pred:
                loss += _get_loss(
                    pred["absolute"], batch["targets_absolute"].to(device)
                )
            return loss
        else:
            return _get_loss(pred, targets)


def nanstd(o, dim, keepdim=False):

    result = torch.sqrt(
        torch.nanmean(
            torch.pow(torch.abs(o - torch.nanmean(o, dim=dim).unsqueeze(dim)), 2),
            dim=dim,
        )
    )

    if keepdim:
        result = result.unsqueeze(dim)

    return result
