# SpikingBrain-7B: Local Testing Tutorial

This comprehensive tutorial guides you through testing all features of SpikingBrain-7B locally. The repository includes multiple components: HuggingFace models, vLLM inference, vision-language models, and quantized versions with spike encoding.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Environment Setup](#environment-setup)
3. [Testing HuggingFace Models](#testing-huggingface-models)
4. [Testing vLLM Inference](#testing-vllm-inference)
5. [Testing Vision-Language Model](#testing-vision-language-model)
6. [Testing W8ASpike Quantization](#testing-w8aspike-quantization)
7. [Testing Int2Spike Encoding](#testing-int2spike-encoding)
8. [Docker Deployment Testing](#docker-deployment-testing)
9. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### Hardware Requirements
- **GPU**: NVIDIA GPU with CUDA support (recommended: >= 24GB VRAM for 7B model)
- **RAM**: At least 32GB system RAM
- **Storage**: At least 50GB free space for models and dependencies

### Software Requirements
- **OS**: Linux (Ubuntu 20.04+ recommended)
- **Python**: 3.10 or higher
- **CUDA**: 11.8 or higher
- **Docker**: (Optional) For containerized deployment

---

## Environment Setup

### Step 1: Clone the Repository

```bash
git clone https://github.com/BICLab/SpikingBrain-7B.git
cd SpikingBrain-7B
```

### Step 2: Create Python Virtual Environment

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate  # On Linux/Mac
# or
.\venv\Scripts\activate  # On Windows
```

### Step 3: Install Dependencies

```bash
# Install base requirements
pip install -r requirements.txt

# Install the vllm_hymeta plugin
pip install -e . --no-build-isolation --verbose
```

### Step 4: Download Model Weights

Download the model weights from ModelScope. Choose based on your use case:

```python
# Install ModelScope SDK
pip install modelscope

# Download specific models (choose one or more)
from modelscope import snapshot_download

# Pre-trained base model
base_model_path = snapshot_download('Panyuqi/V1-7B-base')

# Chat model (SFT)
chat_model_path = snapshot_download('Panyuqi/V1-7B-sft-s3-reasoning')

# Vision-Language model
vlm_model_path = snapshot_download('sherry12334/SpikingBrain-7B-VL')

# Quantized W8ASpike model
quant_model_path = snapshot_download('Abel2076/SpikingBrain-7B-W8ASpike')
```

Alternatively, use Git:
```bash
# Download from ModelScope using Git
git clone https://www.modelscope.cn/Panyuqi/V1-7B-sft-s3-reasoning.git
```

---

## Testing HuggingFace Models

The HuggingFace version provides the most flexible way to test the model with full control over training and inference.

### Test 1: Basic Model Loading and Inference

Create a test script or use the provided example:

```bash
cd run_model
```

Edit `run_model_hf.py` to set your model path:

```python
# In run_model_hf.py, modify line 6:
PATH = '/path/to/your/model'  # e.g., 'V1-7B-sft-s3-reasoning'
```

Run the test:

```bash
python run_model_hf.py
```

**Expected Output:**
- ✅ Tokenizer loaded successfully
- ✅ Model loaded successfully
- Model configuration details
- ✅ Forward step successful (training test)
- Generated text outputs for multiple prompts
- ✅ Generation successful

### Test 2: Chat Template Inference

The chat model uses a conversation format with system and user roles.

Edit `run_model_hf_chat_template.py`:

```python
# Modify PATH variable
PATH = '/path/to/your/chat/model'
```

Run the test:

```bash
python run_model_hf_chat_template.py
```

**What this tests:**
- Chat template formatting
- Conversation-style input handling
- System and user role processing
- Conditional generation with chat context

### Test 3: Custom Inference Script

Create your own test script to validate specific features:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load model and tokenizer
model_path = "V1-7B-sft-s3-reasoning"
tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    padding_side='left',
    trust_remote_code=True
)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16
).cuda()
model.eval()

# Test generation
prompt = "Explain quantum computing in simple terms:"
inputs = tokenizer(prompt, return_tensors='pt').to('cuda')

with torch.no_grad():
    outputs = model.generate(
        inputs.input_ids,
        max_new_tokens=256,
        temperature=0.7,
        top_p=0.9,
        do_sample=True
    )

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"Response: {response}")
```

**Validation Checklist:**
- [ ] Model loads without errors
- [ ] Forward pass works with training mode
- [ ] Generation produces coherent text
- [ ] Multiple prompts work correctly
- [ ] GPU memory usage is reasonable

---

## Testing vLLM Inference

vLLM provides optimized inference with better throughput and lower latency, especially for long sequences.

### Step 1: Verify vLLM Installation

```bash
# Check vLLM version
python -c "import vllm; print(vllm.__version__)"

# Expected: 0.10.0 or compatible version
```

### Step 2: Prepare Model for vLLM

**Important:** Before using vLLM, you must remove the `auto_map` field from `config.json`:

```bash
cd /path/to/your/model

# Backup original config
cp config.json config.json.backup

# Remove auto_map field (manual edit or use sed)
# Delete the following block from config.json:
# "auto_map": {
#   "AutoConfig": "configuration_gla_swa.GLAswaConfig",
#   "AutoModelForCausalLM": "modeling_gla_swa.GLAswaForCausalLM"
# }
```

### Step 3: Run vLLM Inference

Edit `run_model/run_model_vllm.py` to set your model path:

```python
# In run_model_vllm.py, find the LLM initialization (around line 8-10)
llm = LLM(
    model="/path/to/your/model",  # Update this path
    trust_remote_code=True,
    block_size=64,
    dtype='bfloat16',
    max_model_len=32768,
    max_num_seqs=3,
    gpu_memory_utilization=0.35,
)
```

Run the test:

```bash
cd run_model
python run_model_vllm.py
```

**Expected Output:**
- Model initialization logs
- Successful generation output
- ✅ Generation successful

### Step 4: Test vLLM Server Mode

For production-like testing, run vLLM as a server:

```bash
# Start vLLM server
vllm serve /path/to/your/model \
  --served-model-name spikingbrain-7b \
  --gpu-memory-utilization 0.8 \
  --block-size 64 \
  --dtype bfloat16 \
  --port 8000 \
  --trust-remote-code
```

Test the server with curl:

```bash
# In a new terminal
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "spikingbrain-7b",
    "prompt": "Explain artificial intelligence:",
    "max_tokens": 256,
    "temperature": 0.7
  }'
```

### Step 5: Multi-GPU Testing (Optional)

For multi-GPU setups:

```bash
vllm serve /path/to/your/model \
  --tensor-parallel-size 2 \
  --pipeline-parallel-size 1 \
  --served-model-name spikingbrain-7b \
  --gpu-memory-utilization 0.8 \
  --dtype bfloat16 \
  --port 8000 \
  --trust-remote-code
```

**Validation Checklist:**
- [ ] vLLM plugin loads successfully
- [ ] Model registration works
- [ ] Generation is faster than HuggingFace
- [ ] Server mode responds to API calls
- [ ] Multi-GPU mode distributes load correctly

---

## Testing Vision-Language Model

The VLM version extends SpikingBrain to multimodal understanding.

### Step 1: Install Additional Dependencies

```bash
pip install qwen-vl-utils Pillow
```

### Step 2: Prepare Test Image

Ensure you have a test image. The repository includes `equation.png` for LaTeX extraction:

```bash
cd hf_7B_VLM
ls equation.png  # Should exist
```

### Step 3: Run VLM Test

Edit `hf_7B_VLM/run_VLM_hf.py` if needed:

```python
# Verify the model path (line 9)
from modelscope import snapshot_download
path = snapshot_download('sherry12334/SpikingBrain-7B-VL')
```

Run the test:

```bash
cd hf_7B_VLM
python run_VLM_hf.py
```

**Expected Output:**
- Model and processor loading logs
- LaTeX code extracted from the equation image
- Properly formatted LaTeX string

### Step 4: Test Custom Image

Create a custom test with your own image:

```python
import torch
from SpikingBrain_VL import SpikingBrain_VLForConditionalGeneration
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info
from modelscope import snapshot_download

# Load model
path = snapshot_download('sherry12334/SpikingBrain-7B-VL')
model = SpikingBrain_VLForConditionalGeneration.from_pretrained(
    path, torch_dtype="auto", device_map="auto"
)
processor = AutoProcessor.from_pretrained(path)

# Test with your image
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": "your_image.jpg"},
            {"type": "text", "text": "Describe this image in detail."},
        ],
    }
]

# Process and generate
text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
).to("cuda")

generated_ids = model.generate(**inputs, max_new_tokens=256, use_cache=True)
generated_ids_trimmed = [
    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_text[0])
```

**Validation Checklist:**
- [ ] VLM model loads successfully
- [ ] Image processing works correctly
- [ ] LaTeX extraction is accurate
- [ ] Custom images are processed
- [ ] Multimodal context is understood

---

## Testing W8ASpike Quantization

W8ASpike implements weight-8bit activation-spike quantization for efficient inference.

### Step 1: Navigate to W8ASpike Directory

```bash
cd W8ASpike
```

### Step 2: Understand the Architecture

Review the quantization components:

```bash
# Key files
ls -l *.py
# - activations.py: Spike activation functions
# - quant_linear.py: Quantized linear layers
# - modeling_gla_swa.py: Main model with quantization
```

### Step 3: Test Int2Spike Encoding

The Int2Spike module provides spike encoding methods:

```bash
cd Int2Spike
python demo.py
```

**Expected Output:**
```
int4 to 0/1: Time steps = X, firing rate = Y.YY
int4 to -1/0/1: Time steps = X, firing rate = Y.YY
int4 to Bitwise: Time steps = X, firing rate = Y.YY
```

Generated visualization images (filenames will include the actual time step count):
- `Binary_Lif_T{X}_N20.png` (e.g., `Binary_Lif_T16_N20.png`)
- `Ternary_Lif_T{X}_N20.png`
- `Bitwise_Lif_T{X}_N20.png`
- Bidirectional and complement variants

### Step 4: Run Comprehensive Tests

```bash
python test.py
```

**What this tests:**
- Binary spike encoding (0/1) for various bit widths
- Ternary spike encoding (-1/0/1)
- Bitwise spike encoding
- Spike matmul operations
- Weight bitwise conversion
- Numerical accuracy validation

**Expected Output Pattern:**
```
✔ Numerical match: SpikeCountBinaryLIFNode spike sum matches input spike counts.
✔ Numerical match: Binary spike sequence (0/1) matmul with SpikeCountBinaryLIFNode matches spike count matmul.
✔ Numerical match: SpikeCountTernaryLIFNode spike sum matches input spike counts.
...
✔ Passed: Recovered weights match original weights.
```

All tests should show ✔ (checkmark) for passing.

### Step 5: Load Quantized Model

To load and use the quantized model:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from modelscope import snapshot_download

# Download quantized model
model_path = snapshot_download('Abel2076/SpikingBrain-7B-W8ASpike')

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    trust_remote_code=True
)

# Load quantized model
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    trust_remote_code=True,
    torch_dtype=torch.float16
).cuda()
model.eval()

# Test inference
prompt = "What is machine learning?"
inputs = tokenizer(prompt, return_tensors='pt').to('cuda')

with torch.no_grad():
    outputs = model.generate(
        inputs.input_ids,
        max_new_tokens=128,
        temperature=0.7
    )

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

**Validation Checklist:**
- [ ] Int2Spike demo runs successfully
- [ ] All spike encoding tests pass
- [ ] Visualization images are generated
- [ ] Quantized model loads correctly
- [ ] Inference produces reasonable outputs
- [ ] Memory usage is lower than full precision

---

## Testing Int2Spike Encoding

Deep dive into the spike encoding mechanisms.

### Binary Spike Encoding (0/1)

Test unsigned and signed integer encoding:

```python
import torch
from W8ASpike.Int2Spike.neuron import SpikeCountBinaryLIFNode

# Create sample data
x = torch.randint(0, 16, size=(10, 512, 512))  # 4-bit unsigned

# Initialize neuron
lif = SpikeCountBinaryLIFNode()

# Generate spikes
spikes = lif(x.float())

# Check properties
print(f"Input shape: {x.shape}")
print(f"Spike shape: {spikes.shape}")  # (T, B, ...)
print(f"Time steps: {spikes.shape[0]}")
print(f"Firing rate: {lif.firing_rate():.2f}")

# Verify reconstruction
reconstructed = spikes.sum(dim=0)
assert torch.allclose(x.float(), reconstructed, rtol=1e-3)
print("✓ Binary encoding reconstruction passed")
```

### Ternary Spike Encoding (-1/0/1)

Test with signed values:

```python
from W8ASpike.Int2Spike.neuron import SpikeCountTernaryLIFNode

# Signed 4-bit values
x = torch.randint(-8, 8, size=(10, 512, 512))

lif = SpikeCountTernaryLIFNode()
spikes = lif(x.float())

print(f"Ternary spike range: [{spikes.min()}, {spikes.max()}]")
print(f"Expected range: [-1, 0, 1]")
print(f"Firing rate: {lif.firing_rate():.2f}")

# Verify
reconstructed = spikes.sum(dim=0)
assert torch.allclose(x.float(), reconstructed, rtol=1e-3)
print("✓ Ternary encoding reconstruction passed")
```

### Bitwise Spike Encoding

Test parallel bit-level encoding:

```python
from W8ASpike.Int2Spike.neuron import SpikeCountBitwiseNode

# 4-bit unsigned
x = torch.randint(0, 16, size=(10, 512, 512))

# Bitwise encoding
lif = SpikeCountBitwiseNode()
spikes = lif(x.float())

print(f"Bitwise time steps: {spikes.shape[0]}")  # Should be 4 for 4-bit
print(f"Each timestep is a bit position")

# Two's complement mode for signed values
x_signed = torch.randint(-8, 8, size=(10, 512, 512))
lif_signed = SpikeCountBitwiseNode(is_bidirectional=True, is_two_complement=True)
spikes_signed = lif_signed(x_signed.float())

print("✓ Bitwise encoding works for both unsigned and signed")
```

### Spike MatMul Operation

Test the core spike-based matrix multiplication:

```python
from W8ASpike.Int2Spike.neuron import spike_matmul, SpikeCountBinaryLIFNode

# Create input and weight matrices
x = torch.randint(0, 16, size=(8, 2048, 2048)).float()
w = torch.randint(-8, 8, size=(2048, 2048)).float()
x_zero = torch.zeros_like(x)
w_zero = torch.zeros_like(w)

# Spike-based matmul
lif = SpikeCountBinaryLIFNode()
y_spike = spike_matmul(x, w, x_zero=x_zero, lif_quantizer=lif, w_zero=w_zero)

# Reference dense matmul
y_dense = (x + x_zero) @ (w + w_zero)

# Verify equivalence
assert torch.allclose(y_spike, y_dense, rtol=1e-3, atol=1e-3)
print("✓ Spike matmul matches dense matmul")
print(f"Spike matmul allows {lif.firing_rate()*100:.1f}% sparsity")
```

**Validation Checklist:**
- [ ] Binary encoding preserves values
- [ ] Ternary encoding handles signed values
- [ ] Bitwise encoding reduces time steps
- [ ] Spike matmul is numerically correct
- [ ] Firing rates show expected sparsity
- [ ] Visualizations show spike patterns

---

## Docker Deployment Testing

Test containerized deployment for production environments.

### Step 1: Build Docker Image

```bash
cd /path/to/SpikingBrain-7B

# Build the image
docker build -t spiking-brain:7b-v1.0 .

# Monitor build progress (takes 10-20 minutes)
```

**What the Dockerfile does:**
- Uses vLLM v0.10.0 base image
- Installs build dependencies
- Installs vllm_hymeta plugin
- Patches FLA bitnet compatibility

### Step 2: Run Docker Container

```bash
# Run container with GPU access
docker run -itd \
    --entrypoint /bin/bash \
    --network host \
    --name spikingbrain-test \
    --shm-size 160g \
    --gpus all \
    --privileged \
    -v /path/to/models:/workspace/models \
    spiking-brain:7b-v1.0

# Access the container
docker exec -it spikingbrain-test bash
```

### Step 3: Test Inside Container

Inside the container:

```bash
# Verify installation
python -c "import vllm_hymeta; print('Plugin loaded')"

# Test vLLM inference
cd /workspace/run_model

# Edit run_model_vllm.py to use mounted model path
# model="/workspace/models/your-model-name"

python run_model_vllm.py
```

### Step 4: Run vLLM Server in Container

```bash
# Inside container, start server
vllm serve /workspace/models/V1-7B-sft-s3-reasoning \
  --served-model-name spikingbrain-7b \
  --gpu-memory-utilization 0.8 \
  --block-size 64 \
  --dtype bfloat16 \
  --port 8000 \
  --trust-remote-code
```

Test from host machine:

```bash
# From host (container uses --network host)
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "spikingbrain-7b",
    "prompt": "Test prompt",
    "max_tokens": 100
  }'
```

### Step 5: Clean Up

```bash
# Stop container
docker stop spikingbrain-test

# Remove container
docker rm spikingbrain-test

# (Optional) Remove image
docker rmi spiking-brain:7b-v1.0
```

**Validation Checklist:**
- [ ] Docker image builds successfully
- [ ] Container starts with GPU access
- [ ] vllm_hymeta plugin is available
- [ ] Models can be mounted and accessed
- [ ] vLLM server runs inside container
- [ ] API is accessible from host

---

## Troubleshooting

### Common Issues and Solutions

#### 1. CUDA Out of Memory

**Symptoms:**
```
torch.cuda.OutOfMemoryError: CUDA out of memory
```

**Solutions:**
- Reduce `gpu_memory_utilization` (try 0.5 or lower)
- Use smaller `max_model_len`
- Enable CPU offloading: `device_map="auto"`
- Use quantized model (W8ASpike)
- Reduce batch size or `max_num_seqs`

#### 2. vLLM Import Error

**Symptoms:**
```
ImportError: cannot import name 'register_7B_model'
```

**Solutions:**
```bash
# Reinstall plugin
pip uninstall vllm_hymeta
pip install -e . --no-build-isolation --verbose

# Verify installation
python -c "from vllm_hymeta.model_for_7B import register_7B_model; print('OK')"
```

#### 3. Model Loading Fails with auto_map

**Symptoms:**
```
ValueError: auto_map configuration not supported in vLLM
```

**Solution:**
Edit `config.json` and remove the `auto_map` section:
```bash
# Backup first
cp config.json config.json.bak

# Remove auto_map manually or use:
python -c "
import json
with open('config.json', 'r') as f:
    config = json.load(f)
config.pop('auto_map', None)
with open('config.json', 'w') as f:
    json.dump(config, f, indent=2)
"
```

#### 4. Flash Attention Compatibility

**Symptoms:**
```
RuntimeError: flash_attn is not installed
```

**Solutions:**
```bash
# Install flash-attn (may take time to compile)
pip install flash-attn==2.7.3 --no-build-isolation

# If you encounter dependency conflicts, try:
pip install flash-attn==2.7.3 --no-build-isolation --no-deps

# Or use SDPA backend as alternative
export ATTENTION_BACKEND=sdpa
```

#### 5. Triton Compilation Errors

**Symptoms:**
```
TritonCompilationError: ...
```

**Solutions:**
```bash
# Ensure correct version
pip install triton==3.3.1

# Clear triton cache
rm -rf ~/.triton/cache/

# Set environment variable
export TRITON_CACHE_DIR=/tmp/triton_cache
```

#### 6. ModelScope Download Slow/Failed

**Symptoms:**
- Slow download speeds
- Connection timeouts

**Solutions:**
```bash
# Use mirror (China users)
export MODELSCOPE_CACHE=/path/to/cache
pip install modelscope -U

# Or use git clone with resume capability
git clone https://www.modelscope.cn/Panyuqi/V1-7B-sft-s3-reasoning.git
cd V1-7B-sft-s3-reasoning
git lfs pull  # Resume if interrupted
```

#### 7. Int2Spike Test Failures

**Symptoms:**
```
✘ Mismatch: spike sum does not match input spike counts
```

**Solutions:**
- Check PyTorch version: `pip install torch==2.7.1`
- Verify input is integer-valued
- Check tolerance parameters in assertions
- Ensure no NaN/Inf in input tensors

#### 8. VLM Image Loading Issues

**Symptoms:**
```
PIL.UnidentifiedImageError: cannot identify image file
```

**Solutions:**
```bash
# Install/update PIL
pip install Pillow -U

# Check image file
python -c "from PIL import Image; img=Image.open('your_image.jpg'); print(img.size)"

# Convert image format if needed
convert your_image.webp your_image.jpg
```

#### 9. Docker Build Fails

**Symptoms:**
- Build hangs at package installation
- Network timeouts

**Solutions:**
```bash
# Use different mirror in Dockerfile
# Add to Dockerfile before pip install:
RUN pip config set global.index-url https://pypi.org/simple

# Build with more memory
docker build --memory=16g --shm-size=8g -t spiking-brain:7b-v1.0 .

# Use build cache
docker build --build-arg BUILDKIT_INLINE_CACHE=1 -t spiking-brain:7b-v1.0 .
```

#### 10. Permission Errors with Docker Volumes

**Symptoms:**
```
PermissionError: [Errno 13] Permission denied
```

**Solutions:**
```bash
# Run container with user ID
docker run --user $(id -u):$(id -g) ...

# Or change ownership
sudo chown -R $USER:$USER /path/to/models

# Or use SELinux context (if applicable)
docker run -v /path/to/models:/workspace/models:Z ...
```

### Getting Help

If you encounter issues not covered here:

1. **Check GitHub Issues**: https://github.com/BICLab/SpikingBrain-7B/issues
2. **Review Documentation**: README.md and technical reports
3. **Check vLLM Docs**: https://docs.vllm.ai/
4. **Contact**: Open an issue with:
   - Your environment details (OS, Python, CUDA, GPU)
   - Full error message and traceback
   - Steps to reproduce
   - What you've already tried

---

## Performance Benchmarking

### Measure Inference Speed

```python
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "V1-7B-sft-s3-reasoning"
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path, trust_remote_code=True, torch_dtype=torch.bfloat16
).cuda()
model.eval()

# Warmup
prompt = "Hello, world!"
inputs = tokenizer(prompt, return_tensors='pt').to('cuda')
_ = model.generate(inputs.input_ids, max_new_tokens=10)

# Benchmark
prompts = ["Test prompt number " + str(i) for i in range(10)]
inputs = tokenizer(prompts, padding=True, return_tensors='pt').to('cuda')

torch.cuda.synchronize()
start_time = time.time()

with torch.no_grad():
    outputs = model.generate(
        inputs.input_ids,
        max_new_tokens=128,
        temperature=0.7
    )

torch.cuda.synchronize()
elapsed_time = time.time() - start_time

total_tokens = outputs.numel()
tokens_per_second = total_tokens / elapsed_time

print(f"Total time: {elapsed_time:.2f}s")
print(f"Throughput: {tokens_per_second:.2f} tokens/s")
print(f"Average latency: {elapsed_time/len(prompts):.2f}s per request")
```

### Compare HuggingFace vs vLLM

```bash
# Create benchmark script
cat > benchmark.py << 'EOF'
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams
from vllm_hymeta.model_for_7B import register_7B_model

register_7B_model()

model_path = "V1-7B-sft-s3-reasoning"
prompts = ["Test prompt " + str(i) for i in range(20)]

# Benchmark HuggingFace
print("Benchmarking HuggingFace...")
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path, trust_remote_code=True, torch_dtype=torch.bfloat16
).cuda()
model.eval()

start = time.time()
for prompt in prompts:
    inputs = tokenizer(prompt, return_tensors='pt').to('cuda')
    with torch.no_grad():
        _ = model.generate(inputs.input_ids, max_new_tokens=128)
hf_time = time.time() - start
print(f"HuggingFace: {hf_time:.2f}s")

# Benchmark vLLM
print("Benchmarking vLLM...")
llm = LLM(model=model_path, trust_remote_code=True, gpu_memory_utilization=0.5)
sampling_params = SamplingParams(temperature=0.7, max_tokens=128)

start = time.time()
_ = llm.generate(prompts, sampling_params)
vllm_time = time.time() - start
print(f"vLLM: {vllm_time:.2f}s")

print(f"\nSpeedup: {hf_time/vllm_time:.2f}x")
EOF

python benchmark.py
```

---

## Summary

This tutorial covered comprehensive local testing of SpikingBrain-7B features:

✅ **Environment Setup**: Virtual environment, dependencies, model downloads
✅ **HuggingFace Models**: Base model, chat template, custom inference
✅ **vLLM Inference**: Plugin installation, server mode, multi-GPU
✅ **Vision-Language Model**: Multimodal processing, LaTeX extraction
✅ **W8ASpike Quantization**: Spike encoding, quantized inference
✅ **Int2Spike Encoding**: Binary, ternary, bitwise encoding methods
✅ **Docker Deployment**: Containerized deployment and testing
✅ **Troubleshooting**: Common issues and solutions

### Next Steps

1. **Experiment with different models**: Try base, SFT, and VLM versions
2. **Optimize for your use case**: Tune parameters for speed or quality
3. **Deploy to production**: Use Docker + vLLM for serving
4. **Explore spike encoding**: Understand sparsity and efficiency gains
5. **Contribute**: Report issues or improvements on GitHub

### Key Takeaways

- **HuggingFace** is best for flexibility and debugging
- **vLLM** is best for production inference and throughput
- **W8ASpike** reduces memory and compute requirements
- **Int2Spike** enables ultra-low-power neuromorphic inference
- **VLM** extends capabilities to multimodal understanding

Happy testing! 🚀
