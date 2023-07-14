from optimum.onnxruntime import ORTQuantizer, ORTModelForSeq2SeqLM
from optimum.onnxruntime.configuration import AutoQuantizationConfig

model_id = "./ONNX"
onnx_model = ORTModelForSeq2SeqLM.from_pretrained(model_id)
model_dir = onnx_model.model_save_dir

encoder_quantizer = ORTQuantizer.from_pretrained(model_dir, file_name="encoder_model.onnx")

decoder_quantizer = ORTQuantizer.from_pretrained(model_dir, file_name="decoder_model.onnx")

decoder_wp_quantizer = ORTQuantizer.from_pretrained(model_dir, file_name="decoder_with_past_model.onnx")

quantizer = [encoder_quantizer, decoder_quantizer, decoder_wp_quantizer]

dqconfig = AutoQuantizationConfig.avx512_vnni(is_static=False, per_channel=False)

for q in quantizer:
    q.quantize(save_dir="./ONNX_QUNAT/",quantization_config=dqconfig)  # doctest: +IGNORE_RESULT