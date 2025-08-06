import json
import re
import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from FlagEmbedding import FlagModel
import os

# ================= 配置参数 =================
TEST_JSONL_PATH = '/fs-computility/niuyazhe/shared/xueyingyi/xueyingyi/cot_picture/Eimages/annotations/all/eval_data.jsonl'
MODEL_PATH = '/fs-computility/niuyazhe/shared/xueyingyi/xueyingyi/llama/LLaMA-Factory/output/qwen2_5vl_lora_sft_Eimage'
SIMILARITY_MODEL_PATH = '/fs-computility/niuyazhe/shared/dilab/model/bge-base-zh-v1.5'
OUTPUT_JSONL_PATH = '/fs-computility/niuyazhe/shared/xueyingyi/xueyingyi/metric/other/try.jsonl'


# ================= 图像加载 =================
def load_image(image_file):
    try:
        image = Image.open(image_file).convert('RGB')
        print(f"Loaded image: {image_file} (size: {image.size})")
        return image
    except Exception as e:
        print(f"Error loading image {image_file}: {e}")
        return None


# ================= 辅助函数 =================
def extract_meme_text(text):
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
        if not text1.strip() or not text2.strip():
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  
    similarity_model = SimilarityCalculator()  
    fluency_model = FluencyModel()  
    output_dir = os.path.dirname(OUTPUT_JSONL_PATH)
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载Qwen2.5-VL模型和处理器
    print(f"Loading Qwen2.5-VL model from {MODEL_PATH}...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    ).eval()
    processor = AutoProcessor.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True
    )
    processor.image_processor.size["shortest_edge"] = 448

    # 结果存储  
    all_scores = []  
    all_fluency_scores = [] 
    results = []  

    # 处理测试数据  
    with open(TEST_JSONL_PATH, 'r') as test_file, \
         open(OUTPUT_JSONL_PATH, 'w') as output_file:  

        for line in test_file:  
            try:
                data = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"解析错误: {e}")
                continue
            
            try:  
                # 解析输入数据  
                image_path = data['image']  
                human_input = next(c['value'] for c in data['conversations'] if c['from'] == 'human')  
                label_text = next(c['value'] for c in data['conversations'] if c['from'] == 'gpt')  
                
                # 加载图像
                image = load_image(image_path)
                if image is None:
                    continue
                
                # 构造对话格式
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": human_input},
                            {"type": "image"}
                        ]
                    }
                ]

                # 预处理输入
                text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs = processor(
                    text=text,
                    images=[image],
                    return_tensors="pt",
                    padding=True
                ).to(device, dtype=torch.bfloat16)

                # 记录输入序列长度（关键修改）
                input_length = inputs.input_ids.shape[1]

                # 生成文本
                generation_config = {
                    "max_new_tokens": 1024,
                    "do_sample": False,
                    "num_beams": 1,
                    "pad_token_id": processor.tokenizer.pad_token_id,
                    "eos_token_id": processor.tokenizer.eos_token_id
                }

                with torch.no_grad():
                    generate_ids = model.generate(
                        **inputs,** generation_config
                    )
                
                # 仅解码模型生成的部分（排除输入部分）
                # 从input_length开始截取，只保留新生成的token
                generated_tokens = generate_ids[:, input_length:]
                response = processor.batch_decode(
                    generated_tokens,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False
                )[0]
                print(f"生成结果: {response[:100]}...")
                
                # 提取关键文本并评估
                generated_meme = extract_meme_text(response)  
                label_meme = extract_meme_text(label_text)  
                print(f"提取的meme文本: {generated_meme}")  
                print(f"标签meme文本: {label_meme}")  
                
                # 计算指标
                similarity = similarity_model.calculate(generated_meme, label_meme)  
                fluency = fluency_model.evaluate_fluency(generated_meme or "")  
                all_scores.append(similarity)  
                all_fluency_scores.append(fluency)  

                # 保存结果  
                result = {  
                    'image_path': image_path,  
                    'human_input': human_input,  
                    'generated_response': response,  
                    'generated_meme': generated_meme,  
                    'label_meme': label_meme,  
                    'similarity': similarity,
                    'fluency': fluency
                }  
                results.append(result)  
                output_file.write(json.dumps(result) + '\n')  

            except Exception as e:  
                print(f"处理样本出错 {data.get('id', 'unknown')}: {str(e)}")
                continue  

        # 计算并输出平均指标
        if all_scores:  
            average_similarity = np.mean(all_scores)  
            print(f"所有样本的平均相似度为：{average_similarity:.4f}")  
        if all_fluency_scores:
            avg_fluency = np.mean(all_fluency_scores)  
            print(f"所有样本的平均流畅性为：{avg_fluency:.4f}")  

        # 保存平均指标
        with open("/fs-computility/niuyazhe/shared/xueyingyi/xueyingyi/metric/other.txt", 'a') as f:  
            f.write(f"qwen Average similarity: {average_similarity:.4f}\n")
            f.write(f"qwen Average fluency: {avg_fluency:.4f}\n")  


if __name__ == "__main__":  
    main()
