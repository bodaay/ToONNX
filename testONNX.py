from transformers import AutoTokenizer, pipeline,TextStreamer,AutoConfig
from optimum.onnxruntime import ORTModelForCausalLM
import argparse
import os
# create the top-level parser
parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model', help='model name', required=True)
parser.add_argument('-s', '--storage', help='storage path', default="Storage", required=False)


args = parser.parse_args()

# Use the input arguments
model = args.model
storage = args.storage

modelPath=os.path.abspath(os.path.join(storage,model.replace("/","_")+"_ONNX"))
if not os.path.exists(modelPath):
    modelPath=os.path.abspath(os.path.join(storage,model.replace("/","_")+"_ONNX_QUANT"))
print(modelPath)        
useGPU=True
pastKey=True

provider="CPUExecutionProvider"
if useGPU:
    provider="CUDAExecutionProvider"

config=AutoConfig.from_pretrained(modelPath)
tokenizer = AutoTokenizer.from_pretrained(modelPath,return_token_type_ids=False)

if pastKey:
    model = ORTModelForCausalLM.from_pretrained(modelPath,provider=provider,config=config,local_files_only=True,use_io_binding=True,**{"use_cache":True}) 
else:
    model = ORTModelForCausalLM.from_pretrained(modelPath,provider=provider,config=config,local_files_only=True,use_io_binding=False,**{"use_cache":False})
# model = ORTModelForCausalLM.from_pretrained(modePath,provider=provider,config=config,local_files_only=True) # ONNX checkpoint
inputs = tokenizer("write me a python code that will download a website by providing the link as argument",return_token_type_ids=False, return_tensors="pt")
if useGPU:
    inputs = inputs.to('cuda')
gen_tokens = model.generate(**inputs,top_k=50,top_p=0.95,temperature=0.9,repetition_penalty=1.2,use_cache=False, min_length=20,max_new_tokens=1024)
output=tokenizer.batch_decode(gen_tokens)



print(output)