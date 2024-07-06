"""
Train and eval functions used in main.py
"""
import math
import sys
from typing import Iterable, Optional

import torch

from timm.data import Mixup
from timm.utils import ModelEma, accuracy

from .losses import DistillationLoss
from util import utils as utils


def set_bn_state(model):
    for m in model.modules():
        if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
            m.eval()

def train_one_epoch(model: torch.nn.Module, criterion: DistillationLoss,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    clip_grad: float = 0,
                    clip_mode: str = 'norm',
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None,
                    set_training_mode=True,
                    set_bn_eval=False,
                    writer=None,
                    args=None):
    """
        Train the model for one epoch.

        Args:
            model (torch.nn.Module): The model to be trained.
            criterion (DistillationLoss): The loss function used for training.
            data_loader (Iterable): The data loader for the training data.
            optimizer (torch.optim.Optimizer): The optimizer used for training.
            device (torch.device): The device used for training (CPU or GPU).
            epoch (int): The current training epoch.
            loss_scaler: The object used for gradient scaling.
            clip_grad (float, optional): The maximum value for gradient clipping. Default is 0, which means no gradient clipping.
            clip_mode (str, optional): The mode for gradient clipping, can be 'norm' or 'value'. Default is 'norm'.
            model_ema (Optional[ModelEma], optional): The EMA (Exponential Moving Average) model for saving model weights.
            mixup_fn (Optional[Mixup], optional): The function used for Mixup data augmentation.
            set_training_mode (bool, optional): Whether to set the model to training mode. Default is True.
            set_bn_eval (bool, optional): Whether to set the batch normalization layers to evaluation mode. Default is False.
            writer (Optional[Any], optional): The object used for writing TensorBoard logs.
            args (Optional[Any], optional): Additional arguments.

        Returns:
            Dict[str, float]: A dictionary containing the average values of the training metrics.
    """


    model.train(set_training_mode)
    num_steps = len(data_loader)

    if set_bn_eval:
        set_bn_state(model)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 50

    for idx, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        with torch.cuda.amp.autocast():
            outputs = model(samples)
            loss = criterion(samples, outputs, targets)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()

        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        with torch.cuda.amp.autocast():
            loss_scaler(loss, optimizer, clip_grad=clip_grad, clip_mode=clip_mode,
                        parameters=model.parameters(), create_graph=is_second_order)

        torch.cuda.synchronize()
        if model_ema is not None:
            model_ema.update(model)

        learning_rate = optimizer.param_groups[0]["lr"]
        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=learning_rate)


        if idx % print_freq == 0:
            if args.local_rank == 0:
                iter_all_count = epoch * num_steps + idx
                writer.add_scalar('loss', loss, iter_all_count)
                # writer.add_scalar('grad_norm', grad_norm, iter_all_count)
                writer.add_scalar('lr', learning_rate, iter_all_count)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.inference_mode()
def evaluate(data_loader: Iterable, model: torch.nn.Module,
             device: torch.device, epoch: int,
             writer, args,
             visualization=True):
    """
        Evaluate the model for one epoch.

        Args:
            data_loader (Iterable): The data loader for the valid data.
            model (torch.nn.Module): The model to be evaluated.
            device (torch.device): The device used for training (CPU or GPU).
            epoch (int): The current training epoch.
            writer (Optional[Any], optional): The object used for writing TensorBoard logs.
            args (Optional[Any], optional): Additional arguments.
            visualization (bool, optional): Whether to use TensorBoard visualization. Default is True.

        Returns:
            Dict[str, float]: A dictionary containing the average values of the training metrics.
    """

    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    # switch to evaluation mode
    model.eval()

    print_freq = 20
    for images, target in metric_logger.log_every(data_loader, print_freq, header):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            output = model(images)
            loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)


    if visualization and args.local_rank == 0:
        writer.add_scalar('Acc@1', acc1.item(), epoch)
        writer.add_scalar('Acc@5', acc5.item(), epoch)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
