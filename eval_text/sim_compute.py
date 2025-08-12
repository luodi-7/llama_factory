import json
import re
import numpy as np
import torch
from FlagEmbedding import FlagModel
import os

# ================= 配置参数 =================
INFERENCE_JSON_PATH = '/fs-computility/niuyazhe/shared/xueyingyi/xueyingyi/llama/LLaMA-Factory/eval_text/inference_results_Eimage_eval.json'  # 推理结果文件
REFERENCE_JSON_PATH = '/fs-computility/niuyazhe/shared/xueyingyi/xueyingyi/cot_picture/Eimages/annotations/all/llama/eval_data.json'  # 标准文件路径，请替换为实际路径
OUTPUT_JSONL_PATH = '/fs-computility/niuyazhe/shared/xueyingyi/xueyingyi/metric/other/keyevl_results.jsonl'  # 输出结果路径
SIMILARITY_MODEL_PATH = '/fs-computility/niuyazhe/shared/dilab/model/bge-base-zh-v1.5'
AVERAGE_OUTPUT_PATH = '/fs-computility/niuyazhe/shared/xueyingyi/xueyingyi/metric/other/keyevl_averages.txt'  # 平均值输出路径


# ================= 辅助函数 =================
def extract_meme_text(text):
    """从文本中提取meme文本（与原代码逻辑一致）"""
    meme_section = re.search(r'Text on the Meme:\s*\n(.*)', text, re.DOTALL)
    if not meme_section:
        return None
    
    cleaned_text = re.sub(
        r'^box\d+:\s*',
        '',
        meme_section.group(1).strip(),
        flags=re.MULTILINE
    )
    return cleaned_text.strip()


# ================= 相似度计算模型 =================
class SimilarityCalculator:
    def __init__(self):
        self.model = FlagModel(
            SIMILARITY_MODEL_PATH,
            query_instruction_for_retrieval="为这个句子生成表示以用于检索相关文章：",
            use_fp16=True
        )
    
    def calculate(self, text1, text2):
        if not text1 or not text2:  # 处理None或空字符串
            return 0.0
        emb1 = self.model.encode(text1)
        emb2 = self.model.encode(text2)
        return torch.nn.functional.cosine_similarity(
            torch.tensor(emb1).unsqueeze(0),
            torch.tensor(emb2).unsqueeze(0)
        ).item()

# ================= 流畅性评估模型 =================
class FluencyModel:
    def __init__(self):
        from transformers import GPT2LMHeadModel, GPT2Tokenizer
        model_path = '/fs-computility/niuyazhe/shared/dilab/model/gpt2'
        self.model = GPT2LMHeadModel.from_pretrained(model_path)
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_path)
        self.model.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)

    def evaluate_fluency(self, text):
        if not text.strip():
            return 0.0
        
        inputs = self.tokenizer(
            text, 
            return_tensors="pt",
            max_length=512,
            truncation=True
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(** inputs, labels=inputs["input_ids"])
            loss = outputs.loss
            perplexity = torch.exp(loss).item()
        
        return self.normalize_fluency_score(perplexity)
    
    def normalize_fluency_score(self, perplexity):
        min_perplexity = 10
        max_perplexity = 1000
        normalized = 1 - (perplexity - min_perplexity) / (max_perplexity - min_perplexity)
        return max(0.0, min(1.0, normalized))


# ================= 主处理流程 =================
def main():  
    # 初始化组件  
    similarity_model = SimilarityCalculator()  
    fluency_model = FluencyModel()  
    output_dir = os.path.dirname(OUTPUT_JSONL_PATH)
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载推理结果和标准文件
    print(f"加载推理结果文件: {INFERENCE_JSON_PATH}")
    with open(INFERENCE_JSON_PATH, 'r', encoding='utf-8') as f:
        inference_data = json.load(f)
    
    print(f"加载标准文件: {REFERENCE_JSON_PATH}")
    with open(REFERENCE_JSON_PATH, 'r', encoding='utf-8') as f:
        reference_data = json.load(f)
    
    # 将标准数据转为id映射，方便查找
    reference_map = {item['id']: item for item in reference_data}
    
    # 结果存储  
    all_scores = []  
    all_fluency_scores = [] 
    results = []  

    # 处理每个推理结果
    print("开始计算指标...")
    with open(OUTPUT_JSONL_PATH, 'w', encoding='utf-8') as output_file:  
        for inf_item in inference_data:
            inf_id = inf_item['id']
            # 查找对应的标准数据
            if inf_id not in reference_map:
                print(f"警告: 推理结果id {inf_id} 在标准文件中未找到匹配，跳过")
                continue
            ref_item = reference_map[inf_id]
            
            try:
                # 提取推理结果中的生成文本
                inf_assistant = next(c['value'] for c in inf_item['conversations'] if c['from'] == 'assistant')
                # 提取标准文件中的标签文本
                ref_assistant = next(c['value'] for c in ref_item['conversations'] if c['from'] == 'assistant')
                
                # 提取meme文本
                generated_meme = extract_meme_text(inf_assistant)
                label_meme = extract_meme_text(ref_assistant)
                
                print(f"处理id {inf_id}:")
                print(f"生成的meme文本: {generated_meme}")
                print(f"标准的meme文本: {label_meme}")
                
                # 计算指标
                similarity = similarity_model.calculate(generated_meme, label_meme)
                fluency = fluency_model.evaluate_fluency(generated_meme or "")  # 处理None
                
                all_scores.append(similarity)
                all_fluency_scores.append(fluency)
                
                # 保存结果
                result = {
                    'id': inf_id,
                    'image_path': inf_item['image'],
                    'generated_response': inf_assistant,
                    'reference_response': ref_assistant,
                    'generated_meme': generated_meme,
                    'label_meme': label_meme,
                    'similarity': similarity,
                    'fluency': fluency
                }
                results.append(result)
                output_file.write(json.dumps(result, ensure_ascii=False) + '\n')
                
            except Exception as e:
                print(f"处理id {inf_id} 时出错: {str(e)}")
                continue
    
    # 计算并输出平均指标
    print("\n计算完成，结果汇总:")
    if all_scores:
        average_similarity = np.mean(all_scores)
        print(f"所有样本的平均相似度: {average_similarity:.4f}")
    else:
        average_similarity = 0.0
        print("没有有效的相似度数据")
    
    if all_fluency_scores:
        avg_fluency = np.mean(all_fluency_scores)
        print(f"所有样本的平均流畅度: {avg_fluency:.4f}")
    else:
        avg_fluency = 0.0
        print("没有有效的流畅度数据")
    
    # 保存平均指标
    with open(AVERAGE_OUTPUT_PATH, 'w', encoding='utf-8') as f:
        f.write(f"keyevl Average similarity: {average_similarity:.4f}\n")
        f.write(f"keyevl Average fluency: {avg_fluency:.4f}\n")


if __name__ == "__main__":  
    main()