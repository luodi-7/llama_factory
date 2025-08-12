#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多模态模型推理脚本
支持Eimage数据集和其他sharegpt格式的数据集
"""

import os
import json
import yaml
import torch
import argparse
from tqdm import tqdm
from typing import Optional, Any, Dict, List
from pathlib import Path

# LLaMA-Factory imports
from llamafactory.hparams import get_infer_args, ModelArguments, DataArguments, FinetuningArguments
from llamafactory.model import load_tokenizer, load_model
from llamafactory.data import get_template_and_fix_tokenizer, MultiModalDataCollatorForSeq2Seq
from llamafactory.extras.misc import get_logits_processor


def load_dataset_info(dataset_name: str) -> Dict[str, Any]:
    """从dataset_info.json加载数据集信息"""
    # 确保在正确的路径下运行
    current_dir = Path(__file__).parent
    dataset_info_path = current_dir.parent / "data" / "dataset_info.json"
    
    if not dataset_info_path.exists():
        # 尝试从当前工作目录查找
        dataset_info_path = Path.cwd() / "llama" / "LLaMA-Factory" / "data" / "dataset_info.json"
    
    if not dataset_info_path.exists():
        raise FileNotFoundError(f"找不到dataset_info.json文件，尝试的路径: {dataset_info_path}")
    
    with open(dataset_info_path, 'r', encoding='utf-8') as f:
        dataset_info = json.load(f)
    
    if dataset_name not in dataset_info:
        raise ValueError(f"数据集 '{dataset_name}' 在dataset_info.json中未找到")
    
    return dataset_info[dataset_name]


def load_dataset_data(dataset_info: Dict[str, Any]) -> List[Dict[str, Any]]:
    """加载数据集数据"""
    file_path = dataset_info["file_name"]
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"数据集文件不存在: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    return data


def create_inference_examples(data: List[Dict[str, Any]], dataset_info: Dict[str, Any]) -> List[Dict[str, Any]]:
    """创建推理用的样本"""
    examples = []
    
    # 获取列名映射
    columns = dataset_info.get("columns", {})
    messages_col = columns.get("messages", "conversations")
    images_col = columns.get("images", "image")
    
    # 获取标签映射
    tags = dataset_info.get("tags", {})
    role_tag = tags.get("role_tag", "from")
    content_tag = tags.get("content_tag", "value")
    user_tag = tags.get("user_tag", "human")
    assistant_tag = tags.get("assistant_tag", "assistant")
    
    for item in data:
        # 提取对话和图片
        conversations = item.get(messages_col, [])
        image_path = item.get(images_col, "")
        
        if not conversations or not image_path:
            continue
        
        # 找到最后一个用户消息
        user_messages = []
        for conv in conversations:
            if conv.get(role_tag) == user_tag:
                user_messages.append(conv.get(content_tag, ""))
        
        if not user_messages:
            continue
        
        # 使用最后一个用户消息作为推理输入
        last_user_message = user_messages[-1]
        
        example = {
            "id": item.get("id", ""),
            "image": image_path,
            "messages": [{"role": "user", "content": last_user_message}],
            "original_conversations": conversations
        }
        examples.append(example)
    
    return examples


def process_single_example(example: Dict[str, Any], template, tokenizer, processor) -> Dict[str, Any]:
    """处理单个样本，使用LLaMA-Factory的标准方式"""
    messages = example["messages"]
    images = [example["image"]] if example["image"] else []
    
    # 确保消息格式正确
    if not messages:
        raise ValueError("消息列表为空")
    
    # 使用template.mm_plugin.process_messages来处理消息
    processed_messages = template.mm_plugin.process_messages(messages, images, [], [], processor)
    
    # 为推理添加空的assistant消息
    messages_for_encoding = processed_messages + [{"role": "assistant", "content": ""}]
    
    # 使用template的encode_oneturn方法
    prompt_ids, _ = template.encode_oneturn(tokenizer, messages_for_encoding, None, None)
    
    # 检查prompt_ids是否为空
    if not prompt_ids:
        raise ValueError("编码后的prompt_ids为空")
    
    # 获取多模态输入
    mm_inputs = template.mm_plugin.get_mm_inputs(
        images,
        [],  # videos
        [],  # audios
        [len(images)],  # imglens
        [0],  # vidlens
        [0],  # audlens
        [prompt_ids],  # batch_ids
        processor
    )
    
    return {
        "input_ids": prompt_ids,
        "attention_mask": [1] * len(prompt_ids),
        "images": images,
        "videos": [],
        "audios": [],
        **mm_inputs
    }


def batch_inference(model, dataloader, tokenizer, max_new_tokens: int = 9216, temperature: float = 0.7, top_p: float = 0.9, repetition_penalty: float = 1.1):
    """批量推理"""
    all_outputs = []
    device = next(model.parameters()).device
    print(f"使用设备进行推理: {device}")
    
    # 获取生成参数
    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": True,  # 启用采样
        "pad_token_id": tokenizer.eos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "temperature": temperature,  # 控制生成的随机性
        "top_p": top_p,        # 核采样参数
        "repetition_penalty": repetition_penalty,  # 重复惩罚
    }
    
    for i, batch in enumerate(tqdm(dataloader, desc="运行推理")):
        # 移动batch到设备
        batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
        
        # 处理image_grid_thw如果存在
        if 'image_grid_thw' in batch:
            if isinstance(batch['image_grid_thw'], torch.Tensor):
                batch['image_grid_thw'] = batch['image_grid_thw'].to(device)
            elif isinstance(batch['image_grid_thw'], (list, tuple)):
                batch['image_grid_thw'] = [item.to(device) if isinstance(item, torch.Tensor) else item 
                                          for item in batch['image_grid_thw']]
        
        with torch.no_grad():
            # 生成回复
            outputs = model.generate(**batch, **gen_kwargs)
            
            # 解码输出
            for j, output_ids in enumerate(outputs):
                # 找到输入长度
                input_length = batch["input_ids"][j].shape[0]
                # 提取生成的部分
                generated_ids = output_ids[input_length:]
                # 解码
                generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
                all_outputs.append(generated_text.strip())
    
    return all_outputs


def save_results(examples: List[Dict[str, Any]], outputs: List[str], output_path: str):
    """保存推理结果"""
    results = []
    
    for example, output in zip(examples, outputs):
        # 创建新的对话格式
        new_conversations = []
        
        # 复制原始对话，但替换最后一个assistant回复
        for conv in example["original_conversations"]:
            if conv.get("from") == "assistant" and conv == example["original_conversations"][-1]:
                # 替换最后一个assistant回复为模型输出
                new_conversations.append({
                    "from": "assistant",
                    "value": output
                })
            else:
                new_conversations.append(conv)
        
        # 创建结果项
        result_item = {
            "id": example["id"],
            "image": example["image"],
            "conversations": new_conversations
        }
        results.append(result_item)
    
    # 保存到文件
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"结果已保存到: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="多模态模型推理")
    parser.add_argument('--config', type=str, required=True, help='YAML配置文件路径')
    parser.add_argument('--dataset_name', type=str, required=True, help='数据集名称（在dataset_info.json中注册的名称）')
    parser.add_argument('--output_path', type=str, required=True, help='输出文件路径')
    parser.add_argument('--batch_size', type=int, default=4, help='批处理大小')
    parser.add_argument('--max_new_tokens', type=int, default=9216, help='最大生成token数')
    parser.add_argument('--max_samples', type=int, default=None, help='最大样本数（用于测试）')
    parser.add_argument('--temperature', type=float, default=0.7, help='生成温度')
    parser.add_argument('--top_p', type=float, default=0.9, help='核采样参数')
    parser.add_argument('--repetition_penalty', type=float, default=1.1, help='重复惩罚')
    
    args = parser.parse_args()
    
    # 加载配置
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 设置推理参数
    config['stage'] = 'sft'
    config['finetuning_type'] = 'lora' if 'adapter_name_or_path' in config else 'full'
    
    # 获取参数
    model_args, data_args, finetuning_args, generating_args = get_infer_args(config)
    
    # 加载tokenizer和template
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    processor = tokenizer_module.get("processor")
    tokenizer.padding_side = "right"
    template = get_template_and_fix_tokenizer(tokenizer, data_args)
    
    # 加载模型
    print(f"加载模型: {model_args.model_name_or_path}")
    model = load_model(tokenizer, model_args, finetuning_args)
    
    # 确定目标设备
    if torch.cuda.is_available():
        target_device = "cuda:0"
    else:
        target_device = "cpu"
    print(f"目标设备: {target_device}")
    
    # 移动模型到设备
    model = model.to(target_device)
    model.eval()
    
    # 加载数据集信息
    print(f"加载数据集: {args.dataset_name}")
    dataset_info = load_dataset_info(args.dataset_name)
    
    # 加载数据集数据
    data = load_dataset_data(dataset_info)
    print(f"加载了 {len(data)} 个样本")
    
    # 创建推理样本
    examples = create_inference_examples(data, dataset_info)
    print(f"创建了 {len(examples)} 个推理样本")
    
    # 限制样本数量（用于测试）
    if args.max_samples:
        examples = examples[:args.max_samples]
        print(f"限制为 {len(examples)} 个样本进行测试")
    
    # 创建自定义数据集
    class InferenceDataset(torch.utils.data.Dataset):
        def __init__(self, examples, template, tokenizer, processor):
            self.examples = examples
            self.template = template
            self.tokenizer = tokenizer
            self.processor = processor
        
        def __len__(self):
            return len(self.examples)
        
        def __getitem__(self, idx):
            example = self.examples[idx]
            return process_single_example(example, self.template, self.tokenizer, self.processor)
    
    # 创建数据集和数据加载器
    dataset = InferenceDataset(examples, template, tokenizer, processor)
    
    # 创建简单的数据收集器
    def simple_collate_fn(batch):
        # 简单的批处理函数
        input_ids = [item["input_ids"] for item in batch]
        attention_mask = [item["attention_mask"] for item in batch]
        
        # 填充到最大长度
        max_len = max(len(ids) for ids in input_ids)
        padded_input_ids = []
        padded_attention_mask = []
        
        for ids, mask in zip(input_ids, attention_mask):
            pad_len = max_len - len(ids)
            padded_input_ids.append(ids + [tokenizer.pad_token_id] * pad_len)
            padded_attention_mask.append(mask + [0] * pad_len)
        
        # 收集所有多模态输入
        batch_inputs = {
            "input_ids": torch.tensor(padded_input_ids),
            "attention_mask": torch.tensor(padded_attention_mask),
        }
        
        # 合并多模态输入
        for key in ["pixel_values", "image_grid_thw", "video_grid_thw"]:
            if any(key in item for item in batch):
                values = []
                for item in batch:
                    if key in item and item[key].numel() > 0:  # 只添加非空张量
                        values.append(item[key])
                
                if values:  # 只有当有非空值时才合并
                    try:
                        batch_inputs[key] = torch.cat(values, dim=0)
                    except Exception as e:
                        print(f"合并{key}时出错: {e}")
                        # 如果合并失败，跳过这个键
                        continue
        
        return batch_inputs
    
    # 创建数据加载器
    from torch.utils.data import DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        collate_fn=simple_collate_fn,
        shuffle=False
    )
    
    try:
        # 运行推理
        outputs = batch_inference(model, dataloader, tokenizer, args.max_new_tokens, args.temperature, args.top_p, args.repetition_penalty)
        
        # 保存结果
        save_results(examples, outputs, args.output_path)
        
        print(f"推理完成！处理了 {len(examples)} 个样本")
        
    except Exception as e:
        print(f"推理过程中出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
