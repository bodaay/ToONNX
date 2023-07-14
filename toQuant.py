import argparse
import os

from optimum.onnxruntime import ORTQuantizer, ORTModelForSeq2SeqLM,ORTModelForCausalLM
from optimum.onnxruntime.configuration import AutoQuantizationConfig


parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model', help='model name', required=True)
parser.add_argument('-s', '--storage', help='storage path', default="Storage", required=False)
parser.add_argument('-d', '--destination', help='destination path', required=False)
args = parser.parse_args()

# Use the input arguments
model = args.model
storage = args.storage
destination = args.destination if args.destination else args.storage

def prepare_onnx_files(folder_path,destinationFolder):
    found_files = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.onnx'):
                full_path = os.path.join(root, file)
                destination = os.path.join(destinationFolder, file)
                found_files.append({
                    "filename": file,
                    "fullPath": full_path,
                    "destination": destination
                })
    return found_files

modelPath=os.path.abspath(os.path.join(storage,model.replace("/","_")+"_ONNX"))
modelDestination=os.path.join(destination,model.replace("/","_") + "_ONNX_QUANT")
# print(modelPath,modelDestination)
filesList=prepare_onnx_files(modelPath,modelDestination)

quantizer=[]
print(filesList)
for file_info in filesList:
    qm=ORTQuantizer.from_pretrained(modelPath,file_name=file_info["filename"])
    quantizer.append(qm)
dqconfig = AutoQuantizationConfig.avx2(is_static=False, per_channel=False)


for q in quantizer:
    print ("Qunatizing: %s"%q)
    q.quantize(save_dir="./ONNX_QUNAT/",quantization_config=dqconfig)  # doctest: +IGNORE_RESULT