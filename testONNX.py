from transformers import AutoTokenizer, pipeline,TextStreamer,AutoConfig
from optimum.onnxruntime import ORTModelForCausalLM

useGPU=True
pastKey=True

provider="CPUExecutionProvider"
if useGPU:
    provider="CUDAExecutionProvider"
modePath="/home/ubuntu/toONNX/Storage/ehartford_Wizard-Vicuna-7B-Uncensored_ONNX"
config=AutoConfig.from_pretrained(modePath)
tokenizer = AutoTokenizer.from_pretrained(modePath,return_token_type_ids=False)

if pastKey:
    model = ORTModelForCausalLM.from_pretrained(modePath,provider=provider,config=config,local_files_only=True,use_io_binding=True,**{"use_cache":True}) # use this if you generated without past, or merged
else:
    model = ORTModelForCausalLM.from_pretrained(modePath,provider=provider,config=config,local_files_only=True,use_io_binding=False,**{"use_cache":False}) # use this if you generated without past, or merged
# model = ORTModelForCausalLM.from_pretrained(modePath,provider=provider,config=config,local_files_only=True) # ONNX checkpoint
inputs = tokenizer("write me a python code that will download a website by providing the link as argument",return_token_type_ids=False, return_tensors="pt")
if useGPU:
    inputs = inputs.to('cuda')
gen_tokens = model.generate(**inputs,do_sample=True,temperature=0.9, min_length=20,max_length=200)
output=tokenizer.batch_decode(gen_tokens)



print(output)