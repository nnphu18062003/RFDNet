import os
import cv2
import sys
import glob
import time
import math
import argparse
import numpy as np
import tensorflow as tf 

from model import RFDNNet
from utils import *
from tensorflow.keras import Model, Input


def run(config, model):
    if not os.path.exists(config.test_path):
        print(f"Test path not found: {config.test_path}")
        return

    for name in os.listdir(config.test_path):
        fullname = os.path.join(config.test_path, name)
        if not os.path.isfile(fullname):
            continue
            
        # OpenCV reads in BGR, convert to RGB for model/utils compatibility
        lr = cv2.imread(fullname)
        if lr is None:
            continue
        lr = cv2.cvtColor(lr, cv2.COLOR_BGR2RGB)
        
        # upscale_image returns PIL images
        out, out_bilinear = upscale_image(model, lr)
        
        # Generate output filenames using splitext to handle any extension
        base, ext = os.path.splitext(fullname)
        out_sr_path = f"{base}_sr{ext}"
        out_bilinear_path = f"{base}_bilinear{ext}"
        
        # Save directly using PIL to avoid BGR<->RGB confusion
        out.save(out_sr_path)
        out_bilinear.save(out_bilinear_path)
        print(f"Saved: {out_sr_path}")

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()

	# Input Parameters
    parser.add_argument('--test_path', type=str, default="test/")
    parser.add_argument('--gpu', type=str, default='1')
    parser.add_argument('--weight_test_path', type=str, default= "weights/best.h5")
    parser.add_argument('--RSAfilter', type=int, default=64)
    parser.add_argument('--filter', type=int, default=64)
    parser.add_argument('--feat', type=int, default=64)
    parser.add_argument('--scale', type=int, default=3)

    config = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = config.gpu

    # Pass configuration to model
    rfanet_x = RFDNNet(feat=config.feat, filter_num=config.filter)
    x = Input(shape=(None, None, 3))
    out = rfanet_x.main_model(x, config.scale)
    rfa = Model(inputs=x, outputs=out)
    rfa.summary()
    rfa.load_weights(config.weight_test_path)

    run(config, rfa)
