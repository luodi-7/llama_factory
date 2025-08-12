# 多模态模型推理脚本

这个脚本用于在Eimage数据集和其他sharegpt格式的数据集上运行多模态模型的推理。

## 文件说明

- `multimodal_inference.py`: 主要的推理脚本
- `run_inference.sh`: 快速运行脚本
- `config_template.yaml`: 配置文件模板
- `README.md`: 使用说明

## 使用方法

### 方法1: 使用快速运行脚本

```bash
# 给脚本执行权限
chmod +x run_inference.sh

# 运行推理（脚本会自动切换到正确的目录）
./run_inference.sh
```

### 方法2: 直接使用Python脚本

```bash
# 首先切换到LLaMA-Factory根目录
cd /fs-computility/niuyazhe/shared/xueyingyi/xueyingyi/llama/LLaMA-Factory

# 使用配置文件模板
python eval_text/multimodal_inference.py \
    --config eval_text/config_template.yaml \
    --dataset_name Eimage \
    --output_path eval_text/inference_results.json \
    --batch_size 4 \
    --max_new_tokens 512
```

### 方法3: 自定义配置

1. 复制配置文件模板：
```bash
cp config_template.yaml my_config.yaml
```

2. 编辑配置文件，修改模型路径等参数

3. 运行推理：
```bash
# 切换到LLaMA-Factory根目录
cd /fs-computility/niuyazhe/shared/xueyingyi/xueyingyi/llama/LLaMA-Factory

python eval_text/multimodal_inference.py \
    --config eval_text/my_config.yaml \
    --dataset_name Eimage \
    --output_path eval_text/my_results.json
```

## 参数说明

- `--config`: YAML配置文件路径
- `--dataset_name`: 数据集名称（在dataset_info.json中注册的名称）
- `--output_path`: 输出文件路径
- `--batch_size`: 批处理大小（默认4）
- `--max_new_tokens`: 最大生成token数（默认512）
- `--max_samples`: 最大样本数（用于测试，可选）

## 支持的数据集

脚本支持所有在`dataset_info.json`中注册的sharegpt格式数据集，包括：

- Eimage
- mllm_demo
- llava_1k_en
- llava_1k_zh
- 等等...

## 输出格式

输出文件保持与原始数据集相同的格式，只是将最后一个assistant的回复替换为模型的推理结果：

```json
[
  {
    "id": "4422",
    "image": "/path/to/image.jpg",
    "conversations": [
      {
        "from": "human",
        "value": "用户输入..."
      },
      {
        "from": "assistant", 
        "value": "模型推理结果..."
      }
    ]
  }
]
```

## 注意事项

1. 确保模型路径正确
2. 选择合适的模板（如qwen2_vl, llava等）
3. 根据GPU内存调整batch_size
4. 对于大型数据集，建议先用`--max_samples`参数测试少量样本

## 故障排除

1. **内存不足**: 减小batch_size
2. **模型加载失败**: 检查模型路径和模板设置
3. **数据集加载失败**: 检查数据集名称和文件路径
4. **推理速度慢**: 考虑使用更小的模型或减少max_new_tokens
