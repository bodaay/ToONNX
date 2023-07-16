import onnx
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model', help='model name', required=True)
parser.add_argument('-s', '--storage', help='storage path', default="Storage", required=False)
parser.add_argument('-d', '--destination', help='destination path', required=False)
parser.add_argument('-f', '--forcedownload', help='force downloading even if model path exists',default=False,action='store_true', required=False)
parser.add_argument('-q', '--quantize', help='Quantize The Model', required=False)


args = parser.parse_args()
model = args.model
storage = args.storage
# Preprocessing: load the ONNX model
modelPath=os.path.abspath(os.path.join(storage,model.replace("/","_")+"_ONNX"))
if not os.path.exists(modelPath):
    modelPath=os.path.abspath(os.path.join(storage,model.replace("/","_")+"_ONNX_QUANT"))

def check_onnx_model(path):
    print(f"Checking model: {path}")
    # load the model
    # onnx_model = onnx.load(path)
    #print("The model is:\n{}".format(onnx_model))

    # Check the model
    try:
        onnx.checker.check_model(path,full_check=True)
    except onnx.checker.ValidationError as e:
        print(f"The model at path {path} is invalid: {str(e)}")
    else:
        print("The model is valid!")


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