import onnx
import os
import argparse
from onnx import numpy_helper

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model', help='model name', required=True)
parser.add_argument('-s', '--storage', help='storage path', default="Storage", required=False)



args = parser.parse_args()
model = args.model
storage = args.storage
# Preprocessing: load the ONNX model
modelPath=os.path.abspath(os.path.join(storage,model.replace("/","_")+"_ONNX"))
if not os.path.exists(modelPath):
    modelPath=os.path.abspath(os.path.join(storage,model.replace("/","_")+"_ONNX_QUANT"))

def check_onnx_model(path):
    print(f"Checking model: {path}")

    # Load the ONNX model
    model = onnx.load(path,load_external_data=True)

    # Now model's initializers are loaded with tensor data
    for w in model.graph.initializer:
        print('Layer:', w.name)
        if w.raw_data:
            print('Data type:', w.data_type)
            tensor_weights = numpy_helper.to_array(w)
            print('Size:', tensor_weights.size)
        else:
            print('Not found raw_data in', w.name)


def scan_dir_for_onnx(directory):
    # go through each file in the directory
    for root, _, files in os.walk(directory):
        for file in files:
            # check if the extension is .onnx
            if file.endswith('.onnx'):
                # get the full path of the file
                full_path = os.path.join(root, file)
                # Check ONNX model
                check_onnx_model(full_path)

scan_dir_for_onnx(modelPath)