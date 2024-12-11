import time
import torch
import torch.nn as nn
import argparse
import models
import numpy as np
from timm.models import create_model


parser = argparse.ArgumentParser(description='PyTorch MobileNetV4 Inference Speed Test')
# Model params
parser.add_argument('--model', type=str, default='mobilenetv4_small',
                    choices=['mobilenetv4_small', 'mobilenetv4_medium', 'mobilenetv4_large',
                             'mobilenetv4_hybrid_medium', 'mobilenetv4_hybrid_large'],
                    help='model architecture (default: mobilenetv4_small)')
parser.add_argument('--device', default='cuda', type=str)
parser.add_argument('--batch-size', default=32, type=int, help='batch size (default: 32)')
parser.add_argument('--img-size', default=224, type=int,
                    metavar='N', help='Input image dimension, uses model default if empty')
parser.add_argument('--nb-classes', type=int, default=5,
                    help='Number classes in datasets')


def do_pure_cpu_task():
    x = np.random.randn(1, 3, 512, 512).astype(np.float32)
    x = x * 1024 ** 0.5


@torch.inference_mode()
def cal_time3(model, x, args):
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    time_list = []
    for _ in range(50):
        # do_pure_cpu_task() ## cpu warm up, not necessary
        start_event.record()
        ret = model(x)
        end_event.record()
        end_event.synchronize()
        time_list.append(start_event.elapsed_time(end_event) / 1000)

    print(f"{args.model} inference avg time: {sum(time_list[5:]) / len(time_list[5:]):.5f}") ## warm up, remove start 5 times


def main(args):

    device = args.device
    model = create_model(
        args.model,
        num_classes=args.nb_classes
    )
    model.eval().to(device)

    x = torch.randn(size=(args.batch_size, 3, args.img_size, args.img_size), device=device)
    cal_time3(model, x, args)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)