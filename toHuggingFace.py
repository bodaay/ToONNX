from transformers import AutoTokenizer, pipeline,TextStreamer,AutoConfig
from optimum.onnxruntime import ORTModelForCausalLM
import argparse
import os
from dotenv import load_dotenv
load_dotenv()
if not 'HF_API_KEY' in os.environ:
    print ('HF_API_KEY Env Variable is not set')
    exit (1)
HUGGINGFACE_API_KEY=os.environ['HF_API_KEY']
# create the top-level parser
parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model', help='model name', required=True)
parser.add_argument('-s', '--storage', help='storage path', default="Storage", required=False)
parser.add_argument('-u', '--username', help='HuggingFace User/Org Name', default="", required=True)
parser.add_argument('-r', '--reponame', help='Repo Name', default="", required=False)
parser.add_argument('-p', '--private', help='make it private',default=False,action='store_true', required=False)

args = parser.parse_args()

# Use the input arguments
model = args.model
storage = args.storage
username = args.username
reponame = args.reponame
private = args.private
modelPath=os.path.abspath(os.path.join(storage,model.replace("/","_")+"_ONNX"))
if reponame == "":
    reponame= model.split("/")[1] + "-ONNX"
if not os.path.exists(modelPath):
    modelPath=os.path.abspath(os.path.join(storage,model.replace("/","_")+"_ONNX_QUANT"))
    reponame= model.split("/")[1] + "-ONNX-QUANT"

useGPU=True
pastKey=True

provider="CPUExecutionProvider"
if useGPU:
    provider="CUDAExecutionProvider"

config=AutoConfig.from_pretrained(modelPath)
tokenizer = AutoTokenizer.from_pretrained(modelPath,return_token_type_ids=False)

if pastKey:
    model = ORTModelForCausalLM.from_pretrained(modelPath,provider=provider,config=config,local_files_only=True,use_io_binding=True,**{"use_cache":True}) # use this if you generated without past, or merged
else:
    model = ORTModelForCausalLM.from_pretrained(modelPath,provider=provider,config=config,local_files_only=True,use_io_binding=False,**{"use_cache":False}) # use this if you generated without past, or merged


model.push_to_hub(modelPath,repository_id="%s/%s"%(username,reponame),use_auth_token=HUGGINGFACE_API_KEY,private=private)