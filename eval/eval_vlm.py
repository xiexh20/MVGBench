"""
evaluate generated images with VLM
"""
import sys, os
import cv2
import torch
sys.path.append(os.getcwd())
import os.path as osp
from copy import deepcopy
import internvl_utils, json
import numpy as np
import imageio
import torch
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
import matplotlib.pyplot as plt
import glob
from transformers import AutoModel, AutoTokenizer
import textwrap
from tqdm import tqdm

class VLMEvaluator:
    def __init__(self, args):
        "initialize the model"


