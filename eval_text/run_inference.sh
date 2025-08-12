#!/bin/bash

# 多模态模型推理运行脚本

# 模型路径
MODEL_PATH="/fs-computility/niuyazhe/shared/xueyingyi/xueyingyi/llama/LLaMA-Factory/saves/keyevl-7b_Eimage_all"

# 数据集名称（在dataset_info.json中注册的名称）
DATASET_NAME="Eimage_eval"

# 输出路径
OUTPUT_PATH="inference_results_${DATASET_NAME}.json"

# 创建临时配置文件
cat > temp_config.yaml << EOL
model_name_or_path: ${MODEL_PATH}
template: keye_vl
trust_remote_code: true
image_max_pixels: 262144
video_max_pixels: 16384
infer_backend: huggingface
EOL

# 切换到LLaMA-Factory根目录
cd /fs-computility/niuyazhe/shared/xueyingyi/xueyingyi/llama/LLaMA-Factory

# 运行推理
echo "开始推理..."
echo "模型路径: ${MODEL_PATH}"
echo "数据集: ${DATASET_NAME}"
echo "输出文件: ${OUTPUT_PATH}"

python eval_text/multimodal_inference.py \
    --config eval_text/temp_config.yaml \
    --dataset_name ${DATASET_NAME} \
    --output_path eval_text/${OUTPUT_PATH} \
    --batch_size 4 \
    --max_new_tokens 9216

# 清理临时文件
rm temp_config.yaml

echo "推理完成！结果保存在: ${OUTPUT_PATH}"
