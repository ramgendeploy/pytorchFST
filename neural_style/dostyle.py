import argparse
import os
import sys
import time
import re

import numpy as np
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
import torch.onnx

import utils
from transformer_net import TransformerNet
from vgg import Vgg16

def stylize(args):
    device = torch.device("cuda" if args.cuda else "cpu")

    content_image = utils.load_image(args.content_image, scale=args.content_scale)
    content_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])
    content_image = content_transform(content_image)
    content_image = content_image.unsqueeze(0).to(device)

    if args.model.endswith(".onnx"):
        output = stylize_onnx_caffe2(content_image, args)
    else:
        with torch.no_grad():
            style_model = TransformerNet()
            state_dict = torch.load(args.model)
            # remove saved deprecated running_* keys in InstanceNorm from the checkpoint
            for k in list(state_dict.keys()):
                if re.search(r'in\d+\.running_(mean|var)$', k):
                    del state_dict[k]
            style_model.load_state_dict(state_dict)
            style_model.to(device)
            if args.export_onnx:
                assert args.export_onnx.endswith(".onnx"), "Export model file should end with .onnx"
                output = torch.onnx._export(style_model, content_image, args.export_onnx).cpu()
            else:
                output = style_model(content_image).cpu()
    utils.save_image(args.output_image, output[0])
def main():
  parser = argparse.ArgumentParser(description="parser for fast-neural-style")
  parser.add_argument("--content-image", type=str, required=True,
                                 help="path to content image you want to stylize")
  parser.add_argument("--content-scale", type=float, default=None,
                                help="factor for scaling down the content image")
  parser.add_argument("--output-image", type=str, required=True,
                                help="path for saving the output image")
  parser.add_argument("--model", type=str, required=True,
                                help="saved model to be used for stylizing the image. If file ends in .pth - PyTorch path is used, if in .onnx - Caffe2 path")
  parser.add_argument("--cuda", type=int, required=True,
                                help="set it to 1 for running on GPU, 0 for CPU")
  parser.add_argument("--export_onnx", type=str,
                                 help="export ONNX model to a given file")