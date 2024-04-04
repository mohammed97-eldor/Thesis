import time
import datetime
import logging
import torch
import numpy as np
from detectron_conf import *
from detectron2.engine.hooks import HookBase
from detectron2.utils.logger import log_every_n_seconds
from detectron2.data import DatasetMapper, build_detection_test_loader, build_detection_train_loader
from detectron2.engine import DefaultTrainer
# from detectron2.data.transforms import RandomApply, RandomBrightness, RandomRotation, RandomFlip, RandomCrop, RandomContrast
# from detectron2.config import CfgNode
# from detectron2.solver.build import get_default_optimizer_params, maybe_add_gradient_clipping
import detectron2.utils.comm as comm


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
        

class CustomTrainer(DefaultTrainer):
    """
    This class extends Detectron2's DefaultTrainer to include custom behavior for the training process.
    It allows for the addition of a loss evaluation hook to periodically assess the model's performance
    on a validation or test set during training. Moreover, it customizes the data loading with specific
    data augmentations and utilizes a custom optimizer configuration.
    """

    def build_hooks(self):
        """
        Overrides the DefaultTrainer's build_hooks method to insert a custom hook for evaluating
        the loss on a validation or test set during the training process. This enables monitoring
        the model's performance beyond the training set, providing insights into its generalization capabilities.

        Returns:
            List[HookBase]: A list of hooks including the custom LossEvalHook for periodic loss evaluation.
        """
        # First, call the parent class's build_hooks method to get the default set of hooks.
        hooks = super().build_hooks()

        # Insert the custom LossEvalHook before the last hook.
        # This ensures that the loss evaluation is performed at the specified intervals.
        hooks.insert(-1, LossEvalHook(
            eval_period=20,  # Specifies the interval (in terms of training iterations) for performing loss evaluation.
            model=self.model,  # Passes the current model for evaluation.
            data_loader=build_detection_test_loader(
                self.cfg,
                self.cfg.DATASETS.TEST[0],  # Specifies the dataset used for evaluation.
                DatasetMapper(self.cfg, is_train=True)  # Uses the DatasetMapper with the current configuration.
            )
        ))

        return hooks
    # COMMENT THE WHOLE FUNCTION IF YOU DO NOT WANT TO APPLY AUGMENTATIONS
    # "@classmethod"
    # def build_train_loader(cls, cfg):
    #     """
    #     Customizes the training DataLoader by specifying data augmentations that are applied
    #     to the training dataset. This method enhances the model's ability to generalize by introducing
    #     variability into the training data.

    #     Args:
    #         cfg (CfgNode): Configuration node containing settings for data loading and augmentations.
 
    #     Returns:
    #         DataLoader: A DataLoader for training, configured with custom data augmentations.
    #     """
    #     # Defines a mapper that applies a series of data augmentations to each training example.
    #     mapper = DatasetMapper(cfg, is_train=True, augmentations=[
    #         # Randomly applies brightness adjustment with the specified probability.
    #         RandomApply(RandomBrightness(*Augmentation_cfg["RandomBrightness"][:-1]), Augmentation_cfg["RandomBrightness"][-1]),
    #         # Randomly applies rotation with the specified probability.
    #         RandomApply(RandomRotation(angle=Augmentation_cfg["RandomRotation"][:-1]), Augmentation_cfg["RandomRotation"][-1]),
    #         # Randomly applies horizontal flip with the specified probability.
    #         RandomApply(RandomFlip(), Augmentation_cfg["RandomFlip"][0]),
    #         # Randomly applies cropping with the specified probability.
    #         RandomApply(RandomCrop("relative", Augmentation_cfg["RandomCrop"][:-1]), Augmentation_cfg["RandomCrop"][-1]),
    #         # Randomly applies contrast adjustment with the specified probability.
    #         RandomApply(RandomContrast(*Augmentation_cfg["RandomContrast"][:-1]), Augmentation_cfg["RandomContrast"][-1])
    #     ])

    #     # Builds and returns a DataLoader using the defined mapper for data augmentation.
    #     return build_detection_train_loader(cfg, mapper=mapper)

    # COMMENT THE WHOLE FUNCTION IF YOU DO NOT WANT TO CHANGE THE OPTIMIZER
    # "@classmethod"
    # def build_optimizer(cls, cfg: CfgNode, model: torch.nn.Module) -> torch.optim.Optimizer:
    #     """
    #     Configures and returns a custom optimizer for the model, using the AdamW algorithm.
    #     This method allows for custom settings for the optimizer, including learning rates, weight decay, and possibly
    #     gradient clipping, based on the provided configuration.

    #     Args:
    #         cfg (CfgNode): Configuration node containing optimizer settings.
    #         model (torch.nn.Module): The model for which the optimizer is being configured.

    #     Returns:
    #         torch.optim.Optimizer: An instance of the AdamW optimizer with configured parameters.
    #     """
    #     # Retrieves default parameters for optimizer setup, including learning rate and weight decay adjustments.
    #     params = get_default_optimizer_params(
    #         model,
    #         base_lr=cfg.SOLVER.BASE_LR,
    #         weight_decay_norm=cfg.SOLVER.WEIGHT_DECAY_NORM,
    #         bias_lr_factor=cfg.SOLVER.BIAS_LR_FACTOR,
    #         weight_decay=cfg.SOLVER.WEIGHT_DECAY,
    #     )
    #     # change the optimizer here
    #     return maybe_add_gradient_clipping(cfg, torch.optim.AdamW)(
    #         params,
    #         lr=cfg.SOLVER.BASE_LR,
    #         weight_decay=cfg.SOLVER.WEIGHT_DECAY,
    #     )
