# Local NIM Setup on the RTX 5060 Ti 16GB

Self-hosted NVIDIA Inference Microservice (NIM) containers, running on
the RTX 5060 Ti 16GB (Blackwell) via WSL2 + Docker. Optional path. Tier 2
(NVIDIA cloud) and Tier 4 (Ollama local) already cover the failure modes;
NIM is for when he wants OpenAI-compatible self-hosted inference with
TensorRT-LLM acceleration. Blackwell adds native INT4/FP4 kernels, so
quantized NIM images run faster on this card than they would on a 4080.

> Don't run any of this casually. Steps 2-5 take real time and disk.
> Read the whole doc first, then decide.

## What he gets at the end

- A local OpenAI-compatible endpoint at `http://0.0.0.0:8000/v1`
- Llama-3.1-8B-Instruct served from the RTX 5060 Ti 16GB with TensorRT-LLM kernels
- Drop-in tier candidate for ApplyPilot's chain
  (`nvidia-local/...` if a future resolver prefix is added; for now,
  configure as a generic OpenAI base URL override)

## Pre-flight

| Check | How | Expected |
| --- | --- | --- |
| GPU | `nvidia-smi` from PowerShell | RTX 5060 Ti, 16 GB (Blackwell) |
| WSL2 | `wsl --list -v` from PowerShell | Ubuntu (or similar), Version 2 |
| Docker Desktop | open from Start menu | Engine running, WSL2 backend on |
| NGC API key | go to https://org.ngc.nvidia.com/setup/api-key | a fresh `nvapi-...` key copied |

If WSL2 is missing: `wsl --install` from an elevated PowerShell. **UAC
elevation required.** Reboot after.

If Docker Desktop is missing: install from https://docs.docker.com/desktop/install/windows-install/ .
**UAC elevation required.** Pick the WSL2 backend during install. ~10-15 min.

## Step 1. NVIDIA Container Toolkit (inside WSL Ubuntu)

Open a WSL Ubuntu shell. Then:

```bash
distribution=$(. /etc/os-release; echo $ID$VERSION_ID)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
```

**sudo required** for all four commands.

Smoke test the GPU is visible to Docker:

```bash
docker run --rm --gpus all nvcr.io/nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

Expected: an `nvidia-smi` table inside the container showing the RTX 5060 Ti.

## Step 2. NGC login

Still in WSL:

```bash
echo "$NGC_API_KEY" | docker login nvcr.io -u '$oauthtoken' --password-stdin
```

Set `NGC_API_KEY` in his shell first (do not bake into a script):

```bash
export NGC_API_KEY="nvapi-..."
```

## Step 3. Pull and run Llama-3.1-8B-Instruct NIM

The image is multi-GB. **First pull on a typical broadband line: 15-40 min.**

```bash
docker run -it --rm \
  --gpus all \
  --shm-size=16GB \
  -e NGC_API_KEY \
  -v ~/.cache/nim:/opt/nim/.cache \
  -u $(id -u) \
  -p 8000:8000 \
  nvcr.io/nim/meta/llama-3.1-8b-instruct:latest
```

First run also downloads the model weights into `~/.cache/nim`. **Plan
on another 5-15 min.** Subsequent runs reuse the cache.

When ready, the container logs `Application startup complete.` and
`Uvicorn running on http://0.0.0.0:8000`.

## Step 4. Verify the endpoint

From a second shell (WSL or Windows curl):

```bash
curl -s http://0.0.0.0:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"meta/llama-3.1-8b-instruct","messages":[{"role":"user","content":"say hi"}],"max_tokens":20}'
```

Expected: a JSON response with `choices[0].message.content`.

## Step 5. (Optional) Wire into ApplyPilot

The ApplyPilot resolver already understands `nvidia/<model>` against
`integrate.api.nvidia.com`. To point at the local NIM instead, the
cleanest path is one of:

- **Run the NIM container, then stop using the cloud entry for that
  model.** Local NIM and cloud NIM use the same OpenAI shape; only the
  base URL differs. Adding a `nvidia_local/` prefix is a minor llm.py
  change for a future session.
- **Override at the LLMClient level via env** if a one-off test is
  wanted: temporarily point `LLM_URL=http://localhost:8000/v1` and
  `LLM_MODEL=meta/llama-3.1-8b-instruct` and run the smoke script.

Do not rewire the resolver in this session.

## Cost / time summary

| Step | Wall time | Disk | Network | Elevation |
| --- | --- | --- | --- | --- |
| WSL2 install | ~10 min + reboot | ~2 GB | low | yes |
| Docker Desktop install | ~10-15 min | ~3 GB | medium | yes |
| Container Toolkit | ~3 min | <500 MB | low | sudo |
| NGC login | <1 min | 0 | low | no |
| First container pull | 15-40 min | 5-12 GB | high | no |
| First model load | 5-15 min | 5-10 GB | high | no |
| Steady-state warm run | seconds | (cached) | none | no |

Total cold-start: **45-90 min** of mostly-unattended waiting, with two
elevation prompts up front.

## Notes

- 16 GB VRAM fits Llama-3.1-8B at the default INT4/AWQ quant comfortably.
  Blackwell exposes native INT4/FP4 tensor cores, so this card runs the
  quantized variant faster than an Ada-class 4080 would. Larger NIM
  images (70B, 405B) **will not fit** the 5060 Ti's 16 GB. Cloud only.
- FLUX NIM containers exist; check
  https://build.nvidia.com for the current image tag before pulling.
  ComfyUI on Windows native is the lower-friction path for image gen
  on this box.
- Don't bake `NGC_API_KEY` into a script that ends up in git. Source it
  from env every shell.
