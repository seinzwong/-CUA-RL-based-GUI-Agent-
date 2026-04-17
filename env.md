# 环境搭建说明（Feishu CUA 版本）

## 1. 你的机器配置与适配结论

根据你提供的硬件信息：
- CPU: AMD Ryzen 9 9900X3D
- 内存: 48GB DDR5
- GPU: NVIDIA GeForce RTX 5070 Ti 16GB
- 系统: Windows

结论：
1. 这套配置可以运行本项目的 Feishu CUA 全链路。
2. 由于项目包含 `vllm`、`flash_attn`、多处 `bash` 脚本，推荐在 Windows 上使用 WSL2（Ubuntu 22.04）作为主运行环境。
3. 16GB 显存可跑 Qwen3-8B 的 LoRA/推理流程，但训练时建议小 batch（脚本里已是保守配置）。

## 2. 迁移后影响说明（重要）

本仓库已做“纯飞书化”清理，影响如下：
1. 旧场景目录与数据已删除（如 webshop、trajectories 等）。
2. 任务配置仅保留 `eval_agent/configs/task/feishu_travel.json`。
3. 默认评测入口已统一为飞书：`bash eval/run_eval.sh`。
4. 旧 WebShop/ALFWorld/VirtualHome 脚本已不再是可用主路径。

如果你使用旧命令，会报找不到文件或配置，这是预期行为。

## 3. 推荐基础环境（WSL2）

### 3.1 安装 WSL2 + Ubuntu（Windows PowerShell 管理员）

```powershell
wsl --install -d Ubuntu-22.04
```

重启后进入 Ubuntu，安装基础工具：

```bash
sudo apt update
sudo apt install -y git wget curl build-essential cmake pkg-config
```

### 3.2 安装 Miniconda（WSL 内）

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
source ~/.bashrc
```

### 3.3 拉取代码并进入项目

```bash
cd ~/workspace
# git clone <your_repo_url>
cd SPA
```

## 4. Python 环境搭建

项目使用两个环境：
- `SPA`: 采样、评测、PRM 推理与数据处理
- `RL_train`: PPO 训练

### 4.1 SPA 环境

```bash
conda create -n SPA python=3.10 -y
conda activate SPA
```

先安装 PyTorch（建议 CUDA 12.4 轮子）：

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

再安装项目依赖：

```bash
pip install -r requirements.txt
```

说明：
1. `requirements.txt` 中有重型依赖（如 `vllm`、`flash_attn`）。在 WSL2 下通常可用。
2. 若遇到编译失败，先跳过失败包，完成主流程后再按需补装。

### 4.2 RL_train 环境

```bash
conda create -n RL_train python=3.10 -y
conda activate RL_train
```

先装 PyTorch：

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

再装 PPO 依赖（避免把 torch 降级到旧版本）：

```bash
pip install accelerate==0.33.0 datasets==2.21.0 numpy==1.24.3 peft==0.5.0 tqdm==4.67.1 transformers==4.43.1 trl==0.10.1 wandb==0.19.11
```

## 5. 运行前检查

在 `SPA` 环境检查 CUDA：

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

在系统层检查 GPU 驱动：

```bash
nvidia-smi
```

## 6. Feishu CUA 标准执行流程

### 6.1 生成 SFT 种子数据

```bash
conda activate SPA
cd sft
python build_feishu_sft.py --input_file ../eval_agent/data/feishu_travel/train.json --output_file data/feishu_travel_sft.json --replicas 8
```

### 6.2 SFT 训练

```bash
bash feishu_travel_qwen3_8b.sh
```

### 6.3 一键跑数据链路（探索 -> PRM -> RL 数据）

```bash
cd ..
bash eval/run_feishu_cua_pipeline.sh
```

### 6.4 PPO 训练

```bash
conda activate RL_train
bash ppo/train_ppo.sh
```

### 6.5 评测

```bash
conda activate SPA
bash eval/run_eval.sh
```

## 7. 常见问题

1. 问题：`vllm` 安装失败
- 处理：确保在 WSL2 Linux 环境安装；Windows 原生下不建议直接跑完整链路。

2. 问题：`flash_attn` 编译失败
- 处理：先完成不依赖它的流程；需要时再补装对应 CUDA/编译工具链。

3. 问题：脚本报“找不到 webshop/alfworld/virtualhome 文件”
- 原因：仓库已纯飞书化，旧场景已删除。
- 处理：只使用文档中的 Feishu 命令。

## 8. 推荐资源占用策略（针对 16GB 显存）

1. 保持 batch size = 1。
2. 打开梯度累积（脚本已默认保守）。
3. 优先 LoRA，不做全参数训练。
4. 多进程推理时减少 worker 数量，避免显存挤占。

---
如需“最稳妥一键版”，优先执行：
1. `bash eval/run_feishu_cua_pipeline.sh`
2. `bash eval/feishu_cua_eval.sh`

## 9. Windows 原生最小可跑版（不进 WSL）

如果你暂时不想进入 WSL，可以先在 Windows 原生做“最小验证”：
目标是跑通 `eval_agent.main` 的飞书 Mock 评测，不追求完整训练性能。

### 9.1 建议用途

1. 快速验证代码与配置是否可用。
2. 路演前快速演示流程。
3. 不建议在 Windows 原生直接跑完整 vLLM + flash_attn + PPO 训练链路。

### 9.2 环境创建（PowerShell）

```powershell
conda create -n SPA_win python=3.10 -y
conda activate SPA_win
```

安装最小依赖（避免重型包）：

```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install openai==0.28.1 backoff colorama tqdm
pip install transformers==4.50.1 datasets==2.19.1 peft==0.13.2 trl==0.7.7
```

### 9.3 配置 API（OpenAI 兼容）

编辑 `eval_agent/configs/model/openai.json`：
1. `api_key` 改为你自己的 Key。
2. `api_base` / `model_name` 改为你可用的服务与模型。

### 9.4 直接执行飞书评测（PowerShell）

```powershell
python -m eval_agent.main --exp_config feishu_travel --agent_config openai --split test --override --output_path eval/feishu_cua_eval_win
```

### 9.5 说明

1. 该流程不依赖 bash，不依赖 vLLM worker。
2. 如果成功，会在 `eval/feishu_cua_eval_win` 看到任务输出 JSON 与 log。
3. 完整训练与高性能推理仍建议使用 WSL2 流程。

## 10. 安全建议

1. 不要在仓库中提交真实 API Key。
2. 推荐使用私有配置文件或环境变量注入。
3. 对外分享仓库前，先检查 `eval_agent/configs/model/*.json` 是否包含明文凭据。
