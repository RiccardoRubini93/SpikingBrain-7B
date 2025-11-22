# SpikingBrain-7B: Quick Start Guide

Get started with SpikingBrain-7B in 5 minutes! This guide provides the fastest path to running inference.

## Prerequisites

- NVIDIA GPU with CUDA support (>=24GB VRAM recommended)
- Python 3.10+
- CUDA 11.8+

## Quick Installation

```bash
# Clone repository
git clone https://github.com/BICLab/SpikingBrain-7B.git
cd SpikingBrain-7B

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# .\venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Install vllm_hymeta plugin
pip install -e . --no-build-isolation
```

## Download Model

Choose one method:

### Method 1: Using ModelScope (Recommended)

```bash
pip install modelscope
```

```python
from modelscope import snapshot_download

# Download chat model (recommended for beginners)
model_path = snapshot_download('Panyuqi/V1-7B-sft-s3-reasoning')
print(f"Model downloaded to: {model_path}")
```

### Method 2: Using Git

```bash
git clone https://www.modelscope.cn/Panyuqi/V1-7B-sft-s3-reasoning.git
```

## Run Your First Inference

### Option A: HuggingFace (Simple & Flexible)

Create `test_inference.py`:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load model
model_path = "V1-7B-sft-s3-reasoning"  # or your downloaded path
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16
).cuda()
model.eval()

# Run inference
prompt = "What is machine learning?"
inputs = tokenizer(prompt, return_tensors='pt').to('cuda')

with torch.no_grad():
    outputs = model.generate(
        inputs.input_ids,
        max_new_tokens=256,
        temperature=0.7,
        top_p=0.9
    )

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

Run it:
```bash
python test_inference.py
```

### Option B: vLLM (Fast & Production-Ready)

**Important**: First, remove `auto_map` from the model's `config.json`:

```python
import json

# Load config
with open('V1-7B-sft-s3-reasoning/config.json', 'r') as f:
    config = json.load(f)

# Remove auto_map
config.pop('auto_map', None)

# Save config
with open('V1-7B-sft-s3-reasoning/config.json', 'w') as f:
    json.dump(config, f, indent=2)
```

Create `test_vllm.py`:

```python
from vllm import LLM, SamplingParams
from vllm_hymeta.model_for_7B import register_7B_model

# Register model
register_7B_model()

# Load model
llm = LLM(
    model="V1-7B-sft-s3-reasoning",
    trust_remote_code=True,
    dtype='bfloat16',
    gpu_memory_utilization=0.8
)

# Run inference
sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.9,
    max_tokens=256
)

prompt = "What is machine learning?"
outputs = llm.generate([prompt], sampling_params)

print(outputs[0].outputs[0].text)
```

Run it:
```bash
python test_vllm.py
```

## Test Chat Format

For conversational AI:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "V1-7B-sft-s3-reasoning"
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16
).cuda()

# Format as chat
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Explain quantum computing simply."}
]

text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

inputs = tokenizer([text], return_tensors="pt").to("cuda")

with torch.no_grad():
    generated_ids = model.generate(
        inputs.input_ids,
        max_new_tokens=256,
        pad_token_id=tokenizer.pad_token_id
    )

generated_ids = [
    output_ids[len(input_ids):]
    for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(response)
```

## Run vLLM as API Server

For production deployment:

```bash
# Start server
vllm serve V1-7B-sft-s3-reasoning \
  --served-model-name spikingbrain-7b \
  --gpu-memory-utilization 0.8 \
  --dtype bfloat16 \
  --port 8000 \
  --trust-remote-code

# Test with curl (in new terminal)
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "spikingbrain-7b",
    "prompt": "What is AI?",
    "max_tokens": 100
  }'
```

## Test Quantized Model (W8ASpike)

For memory-efficient inference:

```bash
# Download quantized model
pip install modelscope
```

```python
from modelscope import snapshot_download

model_path = snapshot_download('Abel2076/SpikingBrain-7B-W8ASpike')
```

Test spike encoding:

```bash
cd W8ASpike/Int2Spike
python demo.py
```

Expected output: Generated spike visualization images and firing rate statistics.

## Test Vision-Language Model

For multimodal tasks:

```bash
# Install additional dependencies
pip install qwen-vl-utils

# Download VLM
```

```python
from modelscope import snapshot_download
vlm_path = snapshot_download('sherry12334/SpikingBrain-7B-VL')
```

```bash
# Run VLM test
cd hf_7B_VLM
python run_VLM_hf.py
```

## Docker Quick Start

```bash
# Build image
docker build -t spiking-brain:7b-v1.0 .

# Run container
docker run -itd \
    --entrypoint /bin/bash \
    --network host \
    --name spikingbrain \
    --shm-size 160g \
    --gpus all \
    -v $(pwd)/models:/workspace/models \
    spiking-brain:7b-v1.0

# Access container
docker exec -it spikingbrain bash
```

## Troubleshooting

### Out of Memory
- Reduce `gpu_memory_utilization` to 0.5
- Use smaller `max_new_tokens`
- Try quantized model (W8ASpike)

### Import Errors
```bash
# Make sure you're in the SpikingBrain-7B root directory
cd /path/to/SpikingBrain-7B

# Reinstall plugin
pip uninstall vllm_hymeta
pip install -e . --no-build-isolation --verbose
```

### vLLM Model Loading Error
Remove `auto_map` from `config.json` (see Option B above)

### Slow Downloads
```bash
# Use git clone with LFS
git clone https://www.modelscope.cn/Panyuqi/V1-7B-sft-s3-reasoning.git
cd V1-7B-sft-s3-reasoning
git lfs pull
```

## Next Steps

- 📖 Read the [Full Tutorial](LOCAL_TESTING_TUTORIAL.md) for detailed testing
- 📄 Review [Technical Reports](SpikingBrain_Report_Eng.pdf) for architecture details
- 🔬 Experiment with [Example Scripts](run_model/) 
- 💬 Check [GitHub Issues](https://github.com/BICLab/SpikingBrain-7B/issues) for help

## Performance Tips

1. **Use vLLM** for best inference speed (5-10x faster than HuggingFace)
2. **Use W8ASpike** for lowest memory usage (2-3x reduction)
3. **Tune `gpu_memory_utilization`** based on your GPU (0.8-0.9 for dedicated inference)
4. **Use `dtype=bfloat16`** for balance of speed and accuracy
5. **Enable multi-GPU** with `--tensor-parallel-size N` for large batches

## Model Variants

| Model | Size | Use Case | Download Link |
|-------|------|----------|---------------|
| Base | 7B | Pre-training, Fine-tuning | [V1-7B-base](https://www.modelscope.cn/models/Panyuqi/V1-7B-base) |
| Chat (SFT) | 7B | Conversational AI | [V1-7B-sft-s3-reasoning](https://www.modelscope.cn/models/Panyuqi/V1-7B-sft-s3-reasoning) |
| VLM | 7B | Vision-Language Tasks | [SpikingBrain-7B-VL](https://www.modelscope.cn/models/sherry12334/SpikingBrain-7B-VL) |
| Quantized | 7B | Low-Memory Inference | [SpikingBrain-7B-W8ASpike](https://www.modelscope.cn/models/Abel2076/SpikingBrain-7B-W8ASpike) |

## Success Checklist

- [ ] Environment activated and dependencies installed
- [ ] Model downloaded successfully
- [ ] HuggingFace inference works
- [ ] vLLM inference works (optional but recommended)
- [ ] Generated text is coherent
- [ ] Memory usage is acceptable

**Congratulations!** 🎉 You're now ready to use SpikingBrain-7B!

For comprehensive testing and advanced features, see [LOCAL_TESTING_TUTORIAL.md](LOCAL_TESTING_TUTORIAL.md).
