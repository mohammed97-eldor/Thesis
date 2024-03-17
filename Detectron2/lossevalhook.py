from detectron2.engine.hooks import HookBase
from detectron2.evaluation import inference_context
from detectron2.utils.logger import log_every_n_seconds
from detectron2.data import DatasetMapper, build_detection_test_loader
import detectron2.utils.comm as comm
import torch
import time
import datetime
import logging
import numpy as np 


class LossEvalHook(HookBase):
    """
    A custom hook for periodically evaluating the loss on a validation dataset during training.

    This hook leverages Detectron2's HookBase to integrate loss evaluation directly into the training loop.
    It computes the loss for each batch in the provided data loader and calculates the mean loss over the entire dataset.
    This mean loss is then logged and stored for monitoring the model's performance on unseen data during training.

    Attributes:
        _model (torch.nn.Module): The model being trained and evaluated.
        _period (int): The evaluation period, i.e., how often (in terms of training iterations) to evaluate.
        _data_loader (iterable): The data loader for the validation dataset.
    """

    def __init__(self, eval_period, model, data_loader):
        """
        Initializes the LossEvalHook.

        Args:
            eval_period (int): The number of training iterations between each evaluation.
            model (torch.nn.Module): The model that will be evaluated.
            data_loader (iterable): The DataLoader providing the validation dataset.
        """
        self._model = model
        self._period = eval_period
        self._data_loader = data_loader

    def _do_loss_eval(self):
        """
        Perform the loss evaluation on the validation dataset.

        Iterates over the validation dataset, computes the loss for each batch, and calculates the mean loss.
        This function also handles logging progress and synchronization in distributed training setups.

        Returns:
            List of loss values for each batch in the validation dataset.
        """
        total = len(self._data_loader)  # Total number of batches
        num_warmup = min(5, total - 1)  # Number of batches to skip for warm-up

        start_time = time.perf_counter()
        total_compute_time = 0
        losses = []  # Store losses for each batch
        for idx, inputs in enumerate(self._data_loader):
            # Reset timing and loss calculation after warm-up period
            if idx == num_warmup:
                start_time = time.perf_counter()
                total_compute_time = 0
            start_compute_time = time.perf_counter()
            if torch.cuda.is_available():
                torch.cuda.synchronize()  # Ensure synchronization in CUDA operations
            total_compute_time += time.perf_counter() - start_compute_time
            iters_after_start = idx + 1 - num_warmup * int(idx >= num_warmup)
            seconds_per_img = total_compute_time / iters_after_start
            # Log progress and ETA after warm-up period
            if idx >= num_warmup * 2 or seconds_per_img > 5:
                total_seconds_per_img = (time.perf_counter() - start_time) / iters_after_start
                eta = datetime.timedelta(seconds=int(total_seconds_per_img * (total - idx - 1)))
                log_every_n_seconds(
                    logging.INFO,
                    f"Loss on Validation  done {idx + 1}/{total}. {seconds_per_img:.4f} s / img. ETA={eta}",
                    n=5,
                )
            loss_batch = self._get_loss(inputs)  # Compute loss for the current batch
            losses.append(loss_batch)
        mean_loss = np.mean(losses)  # Calculate mean loss
        self.trainer.storage.put_scalar('validation_loss', mean_loss)
        comm.synchronize()  # Synchronize across all processes

        return losses

    def _get_loss(self, data):
        """
        Calculate and return the loss for a batch of data.

        This method forwards the data through the model and aggregates the loss values.

        Args:
            data (dict): A batch of data to be processed by the model.

        Returns:
            float: The total loss for the batch.
        """
        metrics_dict = self._model(data)
        # Ensure all metrics are scalars and detach any tensors from the graph
        metrics_dict = {
            k: v.detach().cpu().item() if isinstance(v, torch.Tensor) else float(v)
            for k, v in metrics_dict.items()
        }
        total_losses_reduced = sum(loss for loss in metrics_dict.values())  # Sum up the losses
        return total_losses_reduced

    def after_step(self):
        next_iter = self.trainer.iter + 1
        is_final = next_iter == self.trainer.max_iter
        if is_final or (self._period > 0 and next_iter % self._period == 0):
            self._do_loss_eval()
        self.trainer.storage.put_scalars(timetest=12)

if __name__ == "__main__":
    print("LoassEvalHook class")
