import onnx
import os
import argparse
from onnx import numpy_helper
import torch
import numpy as np
import fastChat
parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model', help='model name', required=True)
parser.add_argument('-s', '--storage', help='storage path', default="Storage", required=False)



args = parser.parse_args()
model = args.model
storage = args.storage
# Preprocessing: load the ONNX model
modelPath=os.path.abspath(os.path.join(storage,model.replace("/","_")))

def check_torch_model(path):
    print(f"Checking model: {path}")

    # Load the model
    model = torch.load(path)

    # Print the layer's name, data type and size
    for layer in model:
        print('Layer:', layer)
        print('Data type:', model[layer].dtype)
        print('Size:', np.array(model[layer].size()).prod())

def scan_dir_for_torch(directory):
    # go through each file in the directory
    for root, _, files in os.walk(directory):
        for file in files:
            # check if the extension is on of the following
            if "pytorch" in file and( file.endswith('.bin') or file.endswith('.pt') or file.endswith('.safetensors')):
                # get the full path of the file
                full_path = os.path.join(root, file)
                # Check ONNX model
                check_torch_model(full_path)

scan_dir_for_torch(modelPath)