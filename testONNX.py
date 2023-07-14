from transformers import AutoTokenizer, pipeline,TextStreamer,AutoConfig
from optimum.onnxruntime import ORTModelForCausalLM

modePath="/home/khalefa/projects/toONNX/Storage/ONNX_QUNAT"
config=AutoConfig.from_pretrained(modePath)
tokenizer = AutoTokenizer.from_pretrained(modePath,return_token_type_ids=False)
model = ORTModelForCausalLM.from_pretrained(modePath,config=config,local_files_only=True,use_io_binding=False,**{"use_cache":False}) # ONNX checkpoint
inputs = tokenizer("What is the biggest country in the world",return_token_type_ids=False, return_tensors="pt")

gen_tokens = model.generate(**inputs,do_sample=True,temperature=0.9, min_length=20,max_length=20)
output=tokenizer.batch_decode(gen_tokens)
print(output)



