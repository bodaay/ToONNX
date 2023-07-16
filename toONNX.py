import argparse
import os
import subprocess
import sys
from transformers import AutoTokenizer,AutoConfig
from optimum.exporters.tasks import TasksManager
from optimum.exporters.onnx.model_configs import LlamaOnnxConfig
from optimum.utils import NormalizedTextConfig
from optimum.onnxruntime import ORTQuantizer, ORTModelForSeq2SeqLM,ORTModelForCausalLM
from optimum.onnxruntime.configuration import AutoQuantizationConfig

def check_binary():
    # Check if binary "hdfdownloader" is in the current directory or system's PATH
    binary_name = "hfdownloader"
    if binary_name in os.listdir():
        print(f"'{binary_name}' found in the current directory.")
        return True
    elif subprocess.run(f"type {binary_name}", shell=True).returncode == 0:
        print(f"'{binary_name}' found in system's PATH.")
        return True
    else:
        print(f"'{binary_name}' not found.")
        
        # Prompt user for installation
        install = input("Would you like to download it? (yes/no): ")
        if install.lower() == "yes":
            # Execute the bash script to download "hdfdownloader"
            subprocess.run(["bash", "-c", "bash <(curl -sSL https://g.bodaay.io/hfd) -h"])
            return True
        else:
            print("Installation cancelled.")
            return False

# create the top-level parser
parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model', help='model name', required=True)
parser.add_argument('-s', '--storage', help='storage path', default="Storage", required=False)
parser.add_argument('-d', '--destination', help='destination path', required=False)
parser.add_argument('-f', '--forcedownload', help='force downloading even if model path exists',default=False,action='store_true', required=False)
parser.add_argument('-q', '--quantize', help='Quantize The Model', required=False)


args = parser.parse_args()

# Use the input arguments
model = args.model
storage = args.storage
destination = args.destination if args.destination else args.storage
force_redownload = args.forcedownload
quantize = args.quantize
# Check for binary
if check_binary():
    # Print the values
    print(f"Model: {model}")
    print(f"Storage Path: {storage}")
    print(f"Destination Path: {destination}")
    print(f"Qunatize Model: {quantize}")

else:
    print("Operation cancelled due to missing 'hdfdownloader'.")

modelPath=os.path.abspath(os.path.join(storage,model.replace("/","_")))

modelDestination=os.path.join(destination,model.replace("/","_") + "_ONNX")
if not os.path.exists(modelDestination):
    try:
        # Creates the directory
        os.makedirs(modelDestination, exist_ok=True)
        print(f"Directory '{modelDestination}' created successfully")
    except Exception as e:
        # If any error occurred while creating directory, the script will exit.
        print(f"Error occurred while creating directory '{modelDestination}'. Error message: {str(e)}")
        sys.exit(1)
modelDestination=os.path.abspath(modelDestination)
try:
    if not os.path.exists(modelPath) or force_redownload:
        process = subprocess.Popen(['hfdownloader', '-k', '-m', model, '-s', storage], stdout=subprocess.PIPE, universal_newlines=True)
        while True:
            output = process.stdout.readline()
            print(output.strip())
            # Return code is None while subprocess is running
            return_code = process.poll()
            if return_code is not None:
                print('RETURN CODE', return_code)
                # Process has finished, read rest of the output
                for output in process.stdout.readlines():
                    print(output.strip())
                break
    
    # print(modelPath)
    
    if not os.path.exists(modelPath):
        sys.exit("Could not find model")
    
except Exception as e:
    # got below from: https://stackoverflow.com/a/1278740/22133891
    exc_type, exc_obj, exc_tb = sys.exc_info()
    fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
    print(exc_type, fname, exc_tb.tb_lineno)
    sys.exit(1)



# transfomer_config = AutoConfig.from_pretrained("ehartford/WizardLM-7B-V1.0-Uncensored")

# print(modelPath)
# # shit=OnnxConfig(config=transfomer_config)
# onnx_config = LlamaOnnxConfig(config=transfomer_config,
#                                     use_past=False,
#                                     use_past_in_inputs=False,
#                                     use_present_in_outputs=False
#                                     ,task="text-generation") # this will connect to hugging face and automatically infer type of task
# onnx_config = TextDecoderOnnxConfig(config=config
#                                     ,task=TasksManager.infer_task_from_model(model)) # this will connect to hugging face and automatically infer type of task
# use_cache, is same as runnig task -with-past, which will store the precomputed values, you can read about
# https://discuss.huggingface.co/t/what-is-the-purpose-of-use-cache-in-decoder/958

mergeData=True
cache_folder=os.path.abspath(os.path.join(storage,"cache"))
os.makedirs(cache_folder,exist_ok=True)
os.environ["XDG_CACHE_HOME"] = cache_folder
os.environ["TRANSFORMERS_CACHE"] = cache_folder  # This whole cache_dir shit is not working, only way to play with this using XDG_CACHE_HOME env variable
config = AutoConfig.from_pretrained(modelPath, cache_dir=cache_folder) # This whole cache_dir shit is not working, only way to play with this using XDG_CACHE_HOME env variable
ort_model = ORTModelForCausalLM.from_pretrained(modelPath,config=config,cache_dir=cache_folder,local_files_only=True,export=True,**{"use_merged":mergeData}) 
tokenizer = AutoTokenizer.from_pretrained(modelPath,cache_folder=cache_folder)
ort_model.save_pretrained(modelDestination)
tokenizer.save_pretrained(modelDestination)
# below cli command will do all of the above and even better
exit (0)
# command as a list of arguments

# no optimization
command = ["optimum-cli", "export", "onnx", "--model", modelPath, "--task", "text-generation-with-past", modelDestination]

# with optimaization, I need to make the code smarted, of the original model dtype parameter in config.json already 16, no need to waste and generate fp32
# O1: basic general optimizations.
# O2: basic and extended general optimizations, transformers-specific fusions.
# O3: same as O2 with GELU approximation.
# O4: same as O3 with mixed precision (fp16, GPU-only, requires --device cuda).

# command = ["optimum-cli", "export", "onnx", "--model", modelPath, "--task", "text-generation","--optimize","O3", modelDestination] # does not require GPU

# command = ["optimum-cli", "export", "onnx", "--model", modelPath, "--task", "text-generation","--device","cuda","--optimize","O4", modelDestination] # require gpu
process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

while process.poll() is None:
    # Read stdout until it's exhausted
    while True:
        output = process.stdout.readline().strip()
        if output == b'':
            break
        print(output)

    # Read stderr until it's exhausted
    while True:
        err = process.stderr.readline().strip()
        if err == b'':
            break
        print(err)

# command = ["optimum-cli", "export", "onnx", "--model", modelPath, "--task", "text-generation","--optimize","O4", modelDestination]
# try:
#     # running the command and capturing the output
#     process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
#     while True:
#         output = process.stdout.readline()
#         if output == b'' and process.poll() is not None:
#             break
#         if output:
#             print(output.strip())
# except subprocess.CalledProcessError as e:
#     print(f"Error occurred: {str(e)}")
#     print(f"Error output:\n{e.output}")
# except Exception as e: # this will catch any exception
#     print(f"An error occurred: {str(e)}")