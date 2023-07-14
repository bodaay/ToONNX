import os
from transformers import AutoModelForCausalLM

modelPath="/home/khalefa/projects/ToONNX/Storage/ehartford_WizardLM-7B-V1.0-Uncensored"
model=AutoModelForCausalLM.from_pretrained(modelPath)

# Get the last part of the path
folder_name = os.path.basename(modelPath) 

# Append the text to form the file name
file_name_model = folder_name + ".model.txt" 
file_name_state_dict = folder_name + ".state_dict.txt" 



with open(file_name_model, "w") as f: 
    f.write(str(model))

with open(file_name_state_dict, "w") as f: 
    f.write(str(model.state_dict()))