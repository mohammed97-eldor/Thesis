from detectron2.data import DatasetMapper, build_detection_test_loader, build_detection_train_loader
from detectron2.engine import DefaultTrainer
from detectron2.data.transforms import RandomApply, RandomBrightness, RandomRotation, RandomFlip, RandomCrop, RandomContrast
from detectron2.config import CfgNode
from detectron2.solver.build import get_default_optimizer_params, maybe_add_gradient_clipping
import torch
from lossevalhook import LossEvalHook


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

    @classmethod
    def build_train_loader(cls, cfg):
        """
        Customizes the training DataLoader by specifying data augmentations that are applied
        to the training dataset. This method enhances the model's ability to generalize by introducing
        variability into the training data.

        Args:
            cfg (CfgNode): Configuration node containing settings for data loading and augmentations.

        Returns:
            DataLoader: A DataLoader for training, configured with custom data augmentations.
        """
        # Defines a mapper that applies a series of data augmentations to each training example.
        mapper = DatasetMapper(cfg, is_train=True, augmentations=[
            # Randomly applies brightness adjustment with the specified probability.
            RandomApply(RandomBrightness(*Augmentation_cfg["RandomBrightness"][:-1]), Augmentation_cfg["RandomBrightness"][-1]),
            # Randomly applies rotation with the specified probability.
            RandomApply(RandomRotation(angle=Augmentation_cfg["RandomRotation"][:-1]), Augmentation_cfg["RandomRotation"][-1]),
            # Randomly applies horizontal flip with the specified probability.
            RandomApply(RandomFlip(), Augmentation_cfg["RandomFlip"][0]),
            # Randomly applies cropping with the specified probability.
            RandomApply(RandomCrop("relative", Augmentation_cfg["RandomCrop"][:-1]), Augmentation_cfg["RandomCrop"][-1]),
            # Randomly applies contrast adjustment with the specified probability.
            RandomApply(RandomContrast(*Augmentation_cfg["RandomContrast"][:-1]), Augmentation_cfg["RandomContrast"][-1])
        ])

        # Builds and returns a DataLoader using the defined mapper for data augmentation.
        return build_detection_train_loader(cfg, mapper=mapper)

    @classmethod
    def build_optimizer(cls, cfg: CfgNode, model: torch.nn.Module) -> torch.optim.Optimizer:
        """
        Configures and returns a custom optimizer for the model, using the AdamW algorithm.
        This method allows for custom settings for the optimizer, including learning rates, weight decay, and possibly
        gradient clipping, based on the provided configuration.

        Args:
            cfg (CfgNode): Configuration node containing optimizer settings.
            model (torch.nn.Module): The model for which the optimizer is being configured.

        Returns:
            torch.optim.Optimizer: An instance of the AdamW optimizer with configured parameters.
        """
        # Retrieves default parameters for optimizer setup, including learning rate and weight decay adjustments.
        params = get_default_optimizer_params(
            model,
            base_lr=cfg.SOLVER.BASE_LR,
            weight_decay_norm=cfg.SOLVER.WEIGHT_DECAY_NORM,
            bias_lr_factor=cfg.SOLVER.BIAS_LR_FACTOR,
            weight_decay=cfg.SOLVER.WEIGHT_DECAY,
        )
        return maybe_add_gradient_clipping(cfg, torch.optim.AdamW)(
            params,
            lr=cfg.SOLVER.BASE_LR,
            weight_decay=cfg.SOLVER.WEIGHT_DECAY,
        )
        