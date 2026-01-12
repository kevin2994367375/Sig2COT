# ==============================================================================
#                 model_final_manual_loop_ultimate_fix.py (最终手动循环完整版)
#
#   - [根本性修正] 彻底放弃所有与官方 Trainer 相关的代码。
#   - 完全回归到已被证明可在用户环境中启动的、完整的手动PyTorch训练循环。
#   - 整合了所有已确认的必要修正：
#     1. 正确的、手动屏蔽标签的 Dataset 实现。
#     2. 正确的、进行“闭卷考试”的 evaluate 函数。
#     3. 正确的、处理显存和模型加载的 from_adapter 函数。
#     4. 正确的、进行多条件保存的 train 函数逻辑。
#
# ==============================================================================
import json
import os
import gc  # <--- 加上这一行
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader as TorchDataLoader
from sklearn.metrics import accuracy_score
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import get_peft_model, LoraConfig, PeftModel, prepare_model_for_kbit_training
import logging
from tqdm import tqdm
from torch.optim import AdamW
import bitsandbytes as bnb
import re
import random
import math
import sys

# --- 辅助函数 ---
def parse_llm_output_to_hard_label(llm_output_text: str, class_names: list) -> str:
    """
    鲁棒的标签解析函数。
    不管模型输出什么妖魔鬼怪（比如重复、大小写、带标点），
    都强制映射回 class_names 里的标准名称。
    """
    if not isinstance(llm_output_text, str) or not class_names:
        return "Parse_Failed"
    
    # 1. 预处理：转小写，去标点
    text = llm_output_text.lower().strip()
    
    # 2. 定义标准类别的关键词映射 (根据你的数据集调整)
    # 这里的 Key 是 class_names 里的标准名，Value 是可能的变体
    # 注意：PU/CWRU/XJTU 的 class_names 可能不同，这里做通用处理
    
    # 动态构建映射逻辑
    matched_label = None
    
    # 策略：在文本中搜索标准类别名
    # 按照长度倒序搜索，防止 "Inner Race Fault" 被 "Inner" 截胡
    sorted_classes = sorted(class_names, key=len, reverse=True)
    
    # 优先看 "Final Confirmed Diagnosis:" 后面的内容
    if "final confirmed diagnosis:" in text:
        target_area = text.split("final confirmed diagnosis:")[-1]
    else:
        target_area = text # 没找到标签头，就搜全文
        
    for cls_name in sorted_classes:
        # 将标准名也转小写进行匹配
        cls_lower = cls_name.lower()
        
        # 检查是否包含
        if cls_lower in target_area:
            # 找到了！直接返回标准名 (cls_name)
            # 这样即使 target_area 是 "normalnormal"，只要它包含 "normal"，我们返回的就是 "Normal"
            matched_label = cls_name
            break
            
    if matched_label:
        return matched_label
        
    # 如果没找到，返回失败
    return "Parse_Failed"

def find_all_linear_names(model):
    """Dynamically finds all linear layers for LoRA injection."""
    cls = (bnb.nn.Linear4bit, bnb.nn.Linear8bitLt, torch.nn.Linear)
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[-1])
    if 'lm_head' in lora_module_names:
        lora_module_names.remove('lm_head')
    return sorted(list(lora_module_names))
class JudgeDataset(Dataset):
    """
    专门用于Judge模型训练的Dataset类。
    处理包含两个模型输出和裁决结果的训练数据。
    """
    def __init__(self, df, text_col, target_col, tokenizer, max_length, label_col_numeric):
        self.df = df
        self.texts = df[text_col].tolist()
        self.targets = df[target_col].tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # ✅ 修复：确保所有属性都存在
        # 标签列
        if label_col_numeric in df.columns:
            self.true_labels = df[label_col_numeric].tolist()
        else:
            self.true_labels = [0] * len(df)
            logging.warning(f"Label column '{label_col_numeric}' not found, using default labels")
        
        self.label_col = label_col_numeric
        
        # ✅ 修复：确保true_winners属性存在
        if 'true_winner_str' in df.columns:
            self.true_winners = df['true_winner_str'].tolist()
        elif 'true_winner' in df.columns:
            self.true_winners = df['true_winner'].tolist()
        else:
            self.true_winners = [''] * len(df)
            logging.warning("'true_winner_str' and 'true_winner' columns not found, using empty strings")
        
        # ✅ 修复：确保true_fault_types属性存在
        if 'true_fault_type' in df.columns:
            self.true_fault_types = df['true_fault_type'].tolist()
        else:
            self.true_fault_types = [''] * len(df)
            logging.warning("'true_fault_type' column not found, using empty strings")
        
        logging.info(f"JudgeDataset initialized with {len(self.df)} samples")
        logging.info(f"  - true_winners: {len(self.true_winners)} entries")
        logging.info(f"  - true_fault_types: {len(self.true_fault_types)} entries")
        logging.info(f"  - true_labels: {len(self.true_labels)} entries")

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        # 获取instruction和input
        instruction = str(self.texts[idx])
        try:
            input_json = json.loads(self.df.iloc[idx]['input'])
        except:
            input_json = {} # 容错
        
        # 从input中提取各个部分
        signal_data = input_json.get('signal_data', '')
        model_a_response = input_json.get('model_a_response', '')
        model_b_response = input_json.get('model_b_response', '')
        ground_truth = input_json.get('ground_truth', '')  # 用于训练时的监督
        
        raw_output_text = str(self.targets[idx])
        
        # [优化] 清洗 Output：如果开头有 "json"，把它去掉，只保留 { ... }
        # 这样模型在推理时会直接输出 JSON，而不会输出 "json" 这个词
        actual_judge_json = self._clean_output_text(raw_output_text)
        
        # 构建 Prompt
        context = self._build_fault_diagnosis_prompt(instruction, signal_data, model_a_response, model_b_response)
        response = actual_judge_json + self.tokenizer.eos_token
        
        # Tokenization过程保持不变
        context_encoding = self.tokenizer(context, add_special_tokens=True)
        response_encoding = self.tokenizer(response, add_special_tokens=False)

        context_ids = context_encoding['input_ids']
        response_ids = response_encoding['input_ids']

        # 合并输入和响应
        input_ids = context_ids + response_ids
        
        # 创建标签：上下文部分为-100（不计算损失），响应部分为实际token
        labels = [-100] * len(context_ids) + response_ids

        # 截断处理
        if len(input_ids) > self.max_length:
            input_ids = input_ids[-self.max_length:]
            labels = labels[-self.max_length:]
        
        # 填充处理
        padding_length = self.max_length - len(input_ids)
        if padding_length > 0:
            # 左侧填充
            input_ids = [self.tokenizer.pad_token_id] * padding_length + input_ids
            labels = [-100] * padding_length + labels
            attention_mask = [0] * padding_length + [1] * (self.max_length - padding_length)
        else:
            attention_mask = [1] * self.max_length

        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.long),
            'true_label': torch.tensor(self.true_labels[idx], dtype=torch.long),
            'context_len': torch.tensor(len(context_ids), dtype=torch.long),
            'original_indices': idx,
            'true_winner': self.true_winners[idx],
            'true_fault_type': self.true_fault_types[idx]  # 修改：使用故障类型
        }
    def _clean_output_text(self, text):
        """
        清洗 Teacher 模型生成的 Output。
        输入可能是: "json\n{\n...}"
        我们希望训练目标是: "{\n...}" (纯 JSON)
        """
        text = text.strip()
        # 去掉 markdown 代码块标记
        if text.startswith("```json"):
            text = text[7:]
        if text.endswith("```"):
            text = text[:-3]
        
        # 去掉开头的 "json" 单词 (你的数据里有这个)
        if text.lower().startswith("json"):
            text = text[4:].strip()
            
        return text.strip()
    def _extract_json_from_response(self, judge_response):
        """从judge_response中提取JSON内容（去掉```json和```标记）"""
        try:
            if "```json" in judge_response:
                json_match = re.search(r'```json\s*(.*?)\s*```', judge_response, re.DOTALL)
                if json_match:
                    return json_match.group(1).strip()
            return judge_response.strip()
        except Exception:
            return judge_response
    def _build_fault_diagnosis_prompt(self, instruction, signal_data, model_a_response, model_b_response):
        prompt = f"""As a final arbiter and expert diagnostician, your task is to evaluate the reasoning processes of two different AI models (Model A and Model B) that have analyzed the same bearing signal data.

**Your Goal:** Determine which model provides a better, more logical, and more accurate diagnosis, and explain your reasoning in a detailed Chain-of-Thought.

**Evaluation Criteria:**
1.  **Logical Soundness:** Does the model's Chain-of-Thought logically follow from the provided data?
2.  **Accuracy:** Does the model's final conclusion match the Ground Truth?
3.  **Insightfulness:** Does the model correctly identify the key features that lead to the diagnosis?

**Important Constraints:**
- Keep your response concise and focused (100-200 words maximum).
- Provide your judgment in the specified JSON format only.
- Do not include any additional explanations outside the JSON structure.

---
**Original Signal Data (The "Exam Question"):**
{signal_data}

---
**Model A's Answer (Based on Time-Domain Features):**
{model_a_response}

---
**Model B's Answer (Based on Other Features):**
{model_b_response}

---
**Your Task:**
Provide your judgment in the following JSON format. Do not add any text before or after the JSON block.
json
{{
"thought": "First, I will analyze Model A's reasoning. Then I will analyze Model B's reasoning. I will compare both against the ground truth and the original signal data. Finally, I will decide which model performed better and state my final conclusion.",
"analysis_of_model_a": "...",
"analysis_of_model_b": "...",
"comparison_and_reasoning": "...",
"winner": "Model A"
}}
Note: The "winner" can be "Model A", "Model B", "Both are equally good", "Both are equally bad", or "Neither is correct"."""
        return prompt
    
class FlexibleLabelDataset(Dataset):
    """
    Dataset with the final, corrected variable definitions for SFT.
    """
    def __init__(self, df, text_col, target_col, tokenizer, max_length, label_col_numeric):
        self.df = df
        self.texts = df[text_col].tolist()
        self.targets = df[target_col].tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.true_labels = df[label_col_numeric].tolist()
        self.label_col = label_col_numeric

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        source_text = str(self.texts[idx])
        target_text = str(self.targets[idx])

        context = f"""As a rotating machinery diagnostics expert, your task is to write a detailed Chain-of-Thought (CoT) process.
This CoT must logically explain how the data in the provided "Signal Analysis Summary" leads to its "Final Confirmed Diagnosis".

**Key requirements for your CoT:**
1.  Your reasoning must be clear, step-by-step, and based *only* on the provided data.
2.  Use professional and accurate terminology.
3.  **Keep your explanation concise, around 100 words.** This is a brief summary of your thoughts.
4.  **Give your reply in English.**
---
**Signal Analysis Summary:**
{source_text}
---

**Your Chain-of-Thought:**
"""
        response = target_text + self.tokenizer.eos_token
        
        # 1. 分别Tokenize
        context_encoding = self.tokenizer(context, add_special_tokens=True)
        response_encoding = self.tokenizer(response, add_special_tokens=False)

        # 2. 获取Token ID列表
        context_ids = context_encoding['input_ids']
        response_ids = response_encoding['input_ids']

        # 3. 合并
        input_ids = context_ids + response_ids
        
        # 4. 创建标签
        labels = [-100] * len(context_ids) + response_ids
        
        # 5. 截断和填充 - 修正为左填充
        if len(input_ids) > self.max_length:
            # 如果超长，从左侧截断（保留右侧重要内容）
            input_ids = input_ids[-self.max_length:]
            labels = labels[-self.max_length:]
        
        padding_length = self.max_length - len(input_ids)
        if padding_length > 0:
            # 左侧填充
            input_ids = [self.tokenizer.pad_token_id] * padding_length + input_ids
            labels = [-100] * padding_length + labels
            attention_mask = [0] * padding_length + [1] * (self.max_length - padding_length)
        else:
            attention_mask = [1] * self.max_length

        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.long),
            'true_label': torch.tensor(self.true_labels[idx], dtype=torch.long),
            'context_len': torch.tensor(len(context_ids), dtype=torch.long),
            'original_indices': idx
        }


class BaseModel:
    def __init__(self, model_name, num_labels, lora_config_dict=None, tuning_method='qlora', load_adapter_from=None):
        self.model_name = model_name
        self.num_labels = num_labels
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logging.info(f"Initializing BaseModel with base model: {model_name}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=False)
        self.tokenizer.padding_side = 'left'
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        if tuning_method == 'qlora':
            quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
        elif tuning_method == 'lora':
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        else:
            quantization_config = None

        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_name, quantization_config=quantization_config, torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2", device_map="auto", trust_remote_code=True)
        except Exception:
            model = AutoModelForCausalLM.from_pretrained(
                model_name, quantization_config=quantization_config, device_map="auto", trust_remote_code=True)
        
        model.resize_token_embeddings(len(self.tokenizer))
        model.config.pad_token_id = self.tokenizer.pad_token_id
        
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
        
        if lora_config_dict:
            target_modules = find_all_linear_names(model)
            peft_config = LoraConfig(
                task_type="CAUSAL_LM", inference_mode=False, r=lora_config_dict.get('r', 16),
                lora_alpha=lora_config_dict.get('lora_alpha', 32), lora_dropout=lora_config_dict.get('lora_dropout', 0.1),
                target_modules=target_modules)
            if load_adapter_from and os.path.exists(load_adapter_from):
                self.model = PeftModel.from_pretrained(model, load_adapter_from)
            else:
                self.model = get_peft_model(model, peft_config)
            self.model.print_trainable_parameters()
        else:
            self.model = model
    
    
    def prepare_data_loader(self, df, text_col, target_col, batch_size, max_length, label_col_numeric, dataset_type='diagnosis', is_train=True):
        """
        准备数据加载器，支持不同类型的Dataset
        
        Args:
            dataset_type: 'diagnosis' 用于诊断模型, 'judge' 用于Judge模型
        """
        if dataset_type == 'judge':
            dataset = JudgeDataset(df, text_col, target_col, self.tokenizer, max_length, label_col_numeric)
        else:
            dataset = FlexibleLabelDataset(df, text_col, target_col, self.tokenizer, max_length, label_col_numeric)
        
        if is_train:
            # 训练集：保持传入的 batch_size (通常是 1，为了省显存存梯度)
            final_batch_size = batch_size
        else:
            if dataset_type == 'judge':
                final_batch_size = 1  # <--- 保险起见，设为 1
                logging.info(f"Judge模型评估：强制 Batch Size = {final_batch_size} 以防 OOM。")
            else:
                final_batch_size = 4  # 普通模型可以大一点  
            
            # 如果遇到 OOM (爆显存)，请把这里改成 8 或 4
            logging.info(f"检测到评估模式，自动将 Batch Size 从 {batch_size} 提升至 {final_batch_size} 以加速推理。")
        return TorchDataLoader(dataset, batch_size=final_batch_size, shuffle=is_train, pin_memory=True, num_workers=4)

    def calculate_optimal_max_tokens(self, data_loader, sample_batches=20, buffer_ratio=1.2, max_cap=256):
        logging.info(f"Calculating optimal max_new_tokens from {sample_batches} batches...")
        target_lengths = []
        if not data_loader: return max_cap
        for i, batch in enumerate(data_loader):
            if i >= sample_batches: break
            labels = batch['labels']
            for sample_labels in labels:
                valid_length = (sample_labels != -100).sum().item()
                if valid_length > 0: target_lengths.append(valid_length)
        if not target_lengths: return max_cap
        p99_length = np.percentile(target_lengths, 99)
        optimal_length = math.ceil((p99_length * buffer_ratio) / 8) * 8
        final_length = min(optimal_length, max_cap)
        logging.info(f"Optimal max_new_tokens calculated: {final_length}")
        return final_length

    def train(self, train_loader, val_loader, epochs, learning_rate, output_dir, class_names=None, optimal_max_new_tokens=256, acc_freq=1):
        
        optimizer = AdamW(self.model.parameters(), lr=learning_rate, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
        
        history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
        best_val_acc = 0.0
        best_val_loss = float('inf')

        for epoch in range(epochs):
            self.model.train()
            self.model.config.use_cache = False # 训练时关闭 Cache 节省显存
            
            total_train_loss = 0
            
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} Training", leave=False, file=sys.stdout)
            
            for batch in progress_bar:
                # 数据搬运 (移除不需要梯度的非Tensor项)
                batch = {k: v.to(self.device) for k, v in batch.items() 
                         if k not in ['true_label', 'context_len']}
                
                optimizer.zero_grad()
                outputs = self.model(**batch)
                loss = outputs.loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                
                total_train_loss += loss.item()
                
                progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
            
            # 计算平均训练损失
            avg_train_loss = total_train_loss / len(train_loader)
            history['train_loss'].append(avg_train_loss)
            
            # --- 以下是验证逻辑 (保持之前的优化) ---
            if val_loader is not None:
                # 判断是否计算准确率
                should_calc_acc = ((epoch + 1) % acc_freq == 0) or ((epoch + 1) == epochs)
                
                desc = f"Epoch {epoch+1} Val (Loss Only)"
                if should_calc_acc:
                    desc = f"Epoch {epoch+1} Val (Loss + Acc)"

                val_loss, val_acc = self.evaluate(
                    val_loader, class_names, 
                    description=desc,
                    max_new_tokens=optimal_max_new_tokens,
                    calc_acc=should_calc_acc 
                )
                
                history['val_loss'].append(val_loss)
                
                if should_calc_acc:
                    history['val_acc'].append(val_acc)
                    log_msg = f"Epoch {epoch+1}/{epochs} -> Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
                else:
                    history['val_acc'].append(None)
                    log_msg = f"Epoch {epoch+1}/{epochs} -> Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: (Skipped)"
                
                logging.info(log_msg)

                save_triggered = False
                reason = ""
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    reason += " New best loss!"
                    save_triggered = True
                
                if should_calc_acc and val_acc > best_val_acc:
                    best_val_acc = val_acc
                    reason += " New best accuracy!"
                    save_triggered = True
                
                if save_triggered:
                    best_model_path = os.path.join(output_dir, "best_model_lora")
                    self.model.save_pretrained(best_model_path)
                    self.tokenizer.save_pretrained(best_model_path)
                    logging.info(f"  ---> Model saved! Reason: {reason.strip()}.")
                
                scheduler.step(val_loss)
        
        return history
    
    # [修改] 增加 calc_acc 参数，默认 True
    def evaluate(self, data_loader, class_names, description="Evaluating", max_new_tokens=256, calc_acc=True):
        self.model.eval()
        self.model.config.use_cache = True
        total_loss = 0
        all_preds = []
        all_ground_truth = []
        
        # 只有在需要计算准确率时，才检查 pad_token
        if calc_acc:
            self.tokenizer.padding_side = 'left'
            if self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        with torch.no_grad():
            for batch in tqdm(data_loader, desc=description, leave=False, file=sys.stdout):
                # 1. 准备数据 (计算 Loss 总是需要的)
                tensor_batch = {k: v.to(self.device) for k, v in batch.items() 
                                if isinstance(v, torch.Tensor) and k not in ['context_len', 'true_label']}
                
                # 2. 计算 Validation Loss (每一轮都做，速度快)
                outputs = self.model(**tensor_batch)
                loss = outputs.loss
                total_loss += loss.item()

                # =================================================
                # [核心控制] 只有在 calc_acc=True 时才进行生成
                # =================================================
                if calc_acc:
                    # 收集真值
                    if 'true_label' in batch:
                        all_ground_truth.extend(batch['true_label'].numpy())

                    # --- 以下是耗时的生成逻辑 ---
                    input_ids_cpu = tensor_batch['input_ids'].cpu()
                    labels_cpu = tensor_batch['labels'].cpu()
                    masks_cpu = tensor_batch['attention_mask'].cpu()
                    
                    batch_prompt_input_ids = []
                    batch_prompt_attention_mask = []
                    
                    for i in range(input_ids_cpu.shape[0]):
                        answer_start_indices = (labels_cpu[i] != -100).nonzero()
                        if len(answer_start_indices) > 0:
                            cut_idx = answer_start_indices[0].item()
                            prompt_ids = input_ids_cpu[i, :cut_idx]
                            prompt_mask = masks_cpu[i, :cut_idx]
                        else:
                            prompt_ids = input_ids_cpu[i]
                            prompt_mask = masks_cpu[i]
                        
                        valid_start = (prompt_mask == 1).nonzero()
                        if len(valid_start) > 0:
                            real_start = valid_start[0].item()
                            prompt_ids = prompt_ids[real_start:]
                            prompt_mask = prompt_mask[real_start:]

                        batch_prompt_input_ids.append(prompt_ids)
                        batch_prompt_attention_mask.append(prompt_mask)

                    inputs_for_gen = self.tokenizer.pad(
                        {'input_ids': batch_prompt_input_ids, 'attention_mask': batch_prompt_attention_mask},
                        padding=True, return_tensors='pt'
                    ).to(self.device)

                    generated_ids = self.model.generate(
                        **inputs_for_gen,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                        repetition_penalty=1.1 
                    )

                    input_len = inputs_for_gen['input_ids'].shape[1]
                    new_tokens = generated_ids[:, input_len:]
                    decoded_outputs = self.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)

                    for text in decoded_outputs:
                        pred = parse_llm_output_to_hard_label(text, class_names)
                        all_preds.append(pred)

        # 循环结束，计算指标
        avg_loss = total_loss / len(data_loader)
        accuracy = 0.0
        
        if calc_acc:
            valid_indices = [i for i, p in enumerate(all_preds) if p != -1]
            min_len = min(len(all_ground_truth), len(all_preds))
            valid_gt = [all_ground_truth[i] for i in valid_indices if i < min_len]
            valid_pr = [all_preds[i] for i in valid_indices if i < min_len]
            if valid_gt:
                accuracy = accuracy_score(valid_gt, valid_pr)
                logging.info(f"评估详细: 有效样本 {len(valid_gt)}/{min_len}, Acc: {accuracy:.4f}")
        
        self.model.config.use_cache = False
        return avg_loss, accuracy
    def predict(self, prompts: list, batch_size=16, max_new_tokens=256):
        """
        对列表中的 Prompts 进行高效的批量预测。
        """
        self.model.eval()
        self.model.config.use_cache = True
        
        # [关键] 确保 Tokenizer 是左填充 (生成任务必备)
        self.tokenizer.padding_side = 'left'
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        all_outputs = []
        total = len(prompts)
        
        logging.info(f"开始批量推理，总样本数: {total}, Batch Size: {batch_size}")
        
        # 批量处理循环
        for i in tqdm(range(0, total, batch_size), desc="Predicting", leave=False, file=sys.stdout):
            batch_prompts = prompts[i : i + batch_size]
            
            # Tokenize: 自动处理左填充和 Attention Mask
            # max_length 设置为 2048 防止极个别超长 Prompt 导致 OOM
            inputs = self.tokenizer(
                batch_prompts, 
                return_tensors="pt", 
                padding=True, 
                truncation=True, 
                max_length=2048
            ).to(self.device)
            
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,        # 贪婪解码，最快且确定性高
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            # [关键] 只解码新生成的部分 (跳过 Input Prompt)
            input_len = inputs.input_ids.shape[1]
            new_tokens = generated_ids[:, input_len:]
            
            decoded_batch = self.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
            all_outputs.extend(decoded_batch)
            
        return all_outputs
    def _generate_sample_by_sample(self, batch, context_lengths, max_new_tokens, class_names, batch_size):
        """逐样本生成的备用方法（处理左填充）"""
        batch_preds = []
        
        for i in range(batch_size):
            try:
                if context_lengths is not None and i < len(context_lengths):
                    context_len = min(context_lengths[i], batch['input_ids'].shape[1])
                    
                    # 处理左填充：跳过左侧填充，取右侧有效内容
                    valid_length = batch['attention_mask'][i].sum().item()
                    start_pos = batch['input_ids'].shape[1] - valid_length
                    end_pos = start_pos + min(context_len, valid_length)
                    
                    inputs_for_gen = {
                        "input_ids": batch['input_ids'][i:i+1, start_pos:end_pos],
                        "attention_mask": batch['attention_mask'][i:i+1, start_pos:end_pos]
                    }
                else:
                    # 回退方法：使用标签估算上下文长度
                    labels = batch['labels'][i]
                    # 找到第一个非-100的位置（上下文结束位置）
                    non_pad_mask = (labels != -100)
                    if non_pad_mask.any():
                        context_end = non_pad_mask.nonzero(as_tuple=True)[0][0].item()
                        inputs_for_gen = {
                            "input_ids": batch['input_ids'][i:i+1, :context_end],
                            "attention_mask": batch['attention_mask'][i:i+1, :context_end]
                        }
                    else:
                        # 如果没有有效标签，使用整个序列
                        inputs_for_gen = {
                            "input_ids": batch['input_ids'][i:i+1],
                            "attention_mask": batch['attention_mask'][i:i+1]
                        }
                
                generated_ids = self.model.generate(
                    **inputs_for_gen,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
                
                # 计算输入长度
                input_len = inputs_for_gen['input_ids'].shape[1]
                
                decoded_output = self.tokenizer.decode(
                    generated_ids[0, input_len:], 
                    skip_special_tokens=True
                )
                pred_label_index = parse_llm_output_to_hard_label(decoded_output, class_names)
                batch_preds.append(pred_label_index)
                
            except Exception as e:
                logging.error(f"样本 {i} 生成失败: {e}")
                # 记录详细错误信息用于调试
                logging.debug(f"输入形状: {batch['input_ids'][i].shape if 'input_ids' in batch else 'N/A'}")
                logging.debug(f"上下文长度: {context_lengths[i] if context_lengths is not None and i < len(context_lengths) else 'N/A'}")
                batch_preds.append(-1)  # 标记为无效预测
        
        return batch_preds
    def train_judge(self, train_loader, val_loader, val_df, epochs, learning_rate, output_dir, optimal_max_new_tokens=1024):
        """
        修正版：
        1. 验证前强制清理显存。
        2. 每个Epoch结束都保存 last_model，防止白跑。
        3. 增强OOM捕获机制，验证失败不中断训练。
        """
        optimizer = AdamW(self.model.parameters(), lr=learning_rate, weight_decay=0.01)
        # 移除 verbose=True 以兼容新版 PyTorch
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
        
        history = {'train_loss': [], 'val_loss': [], 'val_acc': [], 'val_diagnosis_acc': []}
        best_val_acc = 0.0
        best_val_diagnosis_acc = 0.0
        best_val_loss = float('inf')

        for epoch in range(epochs):
            self.model.train()
            self.model.config.use_cache = False
            total_train_loss = 0
            
            # --- 训练循环 ---
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} Training", leave=False, file=sys.stdout)
            for batch in progress_bar:
                tensor_batch = {}
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor) and k not in ['context_len', 'true_label', 'true_winner', 'true_fault_type']:
                        tensor_batch[k] = v.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(**tensor_batch)
                loss = outputs.loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                total_train_loss += loss.item()
                progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
            
            avg_train_loss = total_train_loss / len(train_loader)
            history['train_loss'].append(avg_train_loss)
            
            # ============================================================
            # [关键修正 1] 强制保存 Checkpoint (保底策略)
            # 无论验证是否成功，先保存当前模型，防止OOM导致前功尽弃
            # ============================================================
            last_model_path = os.path.join(output_dir, "last_model_lora")
            self.model.save_pretrained(last_model_path)
            self.tokenizer.save_pretrained(last_model_path)
            logging.info(f"  --> [Checkpoint] Epoch {epoch+1} completed. Model saved to {last_model_path}")

            # ============================================================
            # [关键修正 2] 验证前清理显存
            # ============================================================
            torch.cuda.empty_cache()
            gc.collect()

            if val_loader is not None and val_df is not None:
                try:
                    # 执行验证
                    val_loss, val_acc, val_diagnosis_acc, *rest= self.evaluate_judge(
                        val_loader, 
                        val_df, 
                        description=f"Epoch {epoch+1} Validation"
                    )
                    history['val_loss'].append(val_loss)
                    history['val_acc'].append(val_acc)
                    history['val_diagnosis_acc'].append(val_diagnosis_acc)
                    
                    logging.info(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, "
                            f"Val Acc: {val_acc:.4f}, Val Diagnosis Acc: {val_diagnosis_acc:.4f}")

                    # 保存最佳模型逻辑 (Best Model)
                    save_model = False
                    reason = ""
                    
                    if val_acc > best_val_acc:
                        best_val_acc = val_acc
                        save_model = True
                        reason += "New best accuracy!"
                    
                    if val_diagnosis_acc > best_val_diagnosis_acc:
                        best_val_diagnosis_acc = val_diagnosis_acc
                        save_model = True
                        reason += " New best diagnosis accuracy!"
                    
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        save_model = True
                        reason += " New best loss!"
                    
                    if save_model:
                        best_model_path = os.path.join(output_dir, "best_model_lora")
                        self.model.save_pretrained(best_model_path)
                        self.tokenizer.save_pretrained(best_model_path)
                        logging.info(f"  --> [Best Model] Saved! Reason: {reason.strip()}.")
                        
                    scheduler.step(val_loss)

                except RuntimeError as e:
                    # [关键修正 3] 捕获 OOM 错误，不让程序崩溃
                    if "out of memory" in str(e).lower():
                        logging.error(f"⚠️ 警告: 验证阶段发生显存溢出 (OOM)！跳过本轮验证，继续下一轮训练。")
                        logging.error("建议: 请进一步调小 `prepare_data_loader` 中的验证集 Batch Size。")
                        torch.cuda.empty_cache() # 再次清理
                        history['val_loss'].append(float('inf'))
                    else:
                        logging.error(f"验证评估发生未知错误: {e}")
                except Exception as e:
                    logging.error(f"验证评估失败: {e}")
                    
        return history
    def evaluate_judge(self, data_loader, test_df, description="Evaluating Judge"):
        self.model.eval()
        self.model.config.use_cache = True
        total_loss = 0
        
        # 记录列表
        all_pred_winners = []
        all_true_winners = []
        all_diagnosis_predictions = []
        all_ground_truths = []
        
        from collections import Counter
        bypass_count = 0
        # [新增] 详细记录列表
        detailed_records = []
        # [修改 1] 定义更精准的关键词映射
        # 只要命中同一个 Key 下的任意 Value，就视为该类故障
        FAULT_MAP = {
            'Normal': ['normal', 'healthy'],
            'Inner': ['inner', 'irf'],
            'Outer': ['outer', 'orf'],
            'Ball': ['ball', 'rolling', 'element']
        }

        def get_fault_type(text):
            """提取故障类型的核心类别"""
            if not isinstance(text, str): return "Unknown"
            t = text.lower()
            for key, keywords in FAULT_MAP.items():
                for kw in keywords:
                    if kw in t:
                        return key
            return "Unknown"

        def is_same_fault(pred1, pred2):
            """[增强版] 判断两个诊断是否一致"""
            type1 = get_fault_type(pred1)
            type2 = get_fault_type(pred2)
            
            # 如果都解析出了有效类型，且类型相同 -> 一致
            if type1 != "Unknown" and type1 == type2:
                return True
            
            # 兜底：简单的字符串比对
            p1 = str(pred1).lower().strip()
            p2 = str(pred2).lower().strip()
            return p1 == p2

        def parse_judge_output(json_string):
            try:
                # [核心修复 1] 物理切割：只保留最后一个 '}' 之前的内容
                # 这能干掉末尾的 "11111..." 或重复的内容
                if "}" in json_string:
                    # 找到最后一个关闭的大括号 (针对单个JSON的情况)
                    # 或者，如果存在 ```json 包裹，优先提取包裹内容
                    if "```json" in json_string:
                        matches = re.findall(r'```json\s*(.*?)\s*```', json_string, re.DOTALL)
                        if matches: 
                            json_string = matches[0]
                    else:
                        # 没有 markdown，尝试找最外层的括号对
                        # 简单策略：找到第一个 '{' 和它对应的闭合 '}' 比较难
                        # 粗暴策略：找到第一个 "winner" 及其后的 "}"
                        pass 
                
                # [核心修复 2] 尝试解析
                # 使用 strict=False 允许控制字符
                data = json.loads(json_string, strict=False)
                return data.get("winner", "Parse_Failed").strip()
                
            except json.JSONDecodeError:
                # [核心修复 3] 如果标准解析失败，使用正则“微创手术”提取 Winner
                # 不管 JSON 结构烂成什么样，只要有 "winner": "Model A" 就能提出来
                match = re.search(r'"winner"\s*:\s*"(Model [AB]|Both.*?)"', json_string, re.IGNORECASE)
                if match:
                    return match.group(1).strip()
                    
                # 关键词兜底 (最后一道防线)
                if "Model A" in json_string: return "Model A"
                if "Model B" in json_string: return "Model B"
                if "Both" in json_string: return "Both are equally good"
                
                return "Parse_Failed"
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(data_loader, desc=description, leave=False, file=sys.stdout)):
                # --- 1. 获取真实标签 ---
                try:
                    batch_true_winners = batch['true_winner']
                except KeyError:
                    if 'true_label' in batch:
                        label_to_winner = {0: 'Model A', 1: 'Model B', 2: 'Both are equally good'}
                        batch_true_winners = [label_to_winner.get(l.item(), 'Model A') for l in batch['true_label']]
                    else:
                        batch_true_winners = ['Unknown'] * len(batch['input_ids'])
                
                all_true_winners.extend(batch_true_winners)
                
                # --- 2. 准备数据 ---
                tensor_batch = {k: v.to(self.device) for k, v in batch.items() 
                                if isinstance(v, torch.Tensor) and k not in ['context_len', 'true_label', 'true_winner', 'true_fault_type']}
                
                outputs = self.model(**tensor_batch)
                total_loss += outputs.loss.item()
                
                # --- 3. 推理逻辑 ---
                context_lengths = (tensor_batch['labels'] == -100).sum(dim=1)
                
                for i in range(tensor_batch['input_ids'].shape[0]):
                    current_global_idx = batch_idx * data_loader.batch_size + i
                    
                    if current_global_idx < len(test_df):
                        row = test_df.iloc[current_global_idx]
                        sample_id = row.get('id', current_global_idx)
                        ground_truth = row.get('ground_truth', 'Unknown')
                        diag_a = row.get('model_a_diagnosis', 'Unknown')
                        diag_b = row.get('model_b_diagnosis', 'Unknown')
                    else:
                        sample_id = -1
                        ground_truth = "Unknown"
                        diag_a, diag_b = "Unknown", "Unknown"

                    all_ground_truths.append(ground_truth)
                    judge_raw_output = "SKIPPED (Smart Bypass)"
                    is_conflict = not is_same_fault(diag_a, diag_b)
                    if not is_conflict:
                        predicted_winner = "Both are equally good"
                        bypass_count += 1
                        final_diag = diag_a
                    else:
                        # [Call LLM]
                        context_len = context_lengths[i].item()
                        inputs_for_gen = {
                            "input_ids": tensor_batch['input_ids'][i:i+1, :context_len],
                            "attention_mask": tensor_batch['attention_mask'][i:i+1, :context_len]
                        }
                        
                        generated_ids = self.model.generate(
                            **inputs_for_gen,
                            max_new_tokens=1024,
                            do_sample=False,
                            pad_token_id=self.tokenizer.pad_token_id,
                            eos_token_id=self.tokenizer.eos_token_id,
                            repetition_penalty=1.1
                        )
                        
                        generated_text = self.tokenizer.decode(generated_ids[0, context_len:], skip_special_tokens=True)
                        judge_raw_output = generated_text # 保存原始输出用于分析 thought
                        
                        predicted_winner = parse_judge_output(generated_text)
                        
                        if predicted_winner == "Model B":
                            final_diag = diag_b
                        else:
                            final_diag = diag_a

                    # 记录统计信息
                    all_pred_winners.append(predicted_winner)
                    all_diagnosis_predictions.append(final_diag)
                    
                    # 尝试获取 True Winner 用于计算 Adjudication Acc
                    try:
                        t_winner_idx = batch['true_label'][i].item() if 'true_label' in batch else -1
                        # 简单的 index 转 string，仅用于记录
                        t_winner = "Unknown"
                        if t_winner_idx == 0: t_winner = "Model A"
                        elif t_winner_idx == 1: t_winner = "Model B"
                        elif t_winner_idx == 2: t_winner = "Both"
                        all_true_winners.append(t_winner)
                    except:
                        all_true_winners.append("Unknown")

                    # [核心新增] 构建详细记录
                    # 只记录分歧样本，或者全部记录（建议全部记录，后面分析时再过滤）
                    if current_global_idx < len(test_df):
                        
                        # [修复] 安全地转换 ID
                        try:
                            # 尝试转为 int
                            safe_id = int(sample_id)
                        except (ValueError, TypeError):
                            # 如果是 NaN 或无法转换，使用全局索引代替，或者设为 -1
                            safe_id = int(current_global_idx)

                        record = {
                            "id": safe_id,  # <--- 使用修复后的 safe_id
                            "is_conflict": bool(is_conflict),
                            "ground_truth": ground_truth,
                            "model_a_pred": diag_a,
                            "model_b_pred": diag_b,
                            "judge_winner": predicted_winner,
                            "final_diagnosis": final_diag,
                            "is_correct": bool(final_diag == ground_truth),
                            "judge_raw_output": judge_raw_output
                        }
                        detailed_records.append(record)

        # --- 计算指标 ---
        avg_loss = total_loss / len(data_loader)
        winner_distribution = dict(Counter(all_pred_winners))
        logging.info(f"\n📊 Judge 选择分布: {winner_distribution}")
        logging.info(f"⚡ Smart Bypass (Both) 触发次数: {bypass_count}/{len(all_ground_truths)}")
        
        # 1. 严格裁决准确率 (Strict Accuracy)
        valid_pairs = [(t, p) for t, p in zip(all_true_winners, all_pred_winners) if t and p != "Parse_Failed"]
        if valid_pairs:
            vt, vp = zip(*valid_pairs)
            strict_acc = accuracy_score(vt, vp)
            logging.info(f"⚖️ 严格裁决准确率 (Strict Acc): {strict_acc:.4f}")
            
            # [修改 2] 松弛裁决准确率 (Relaxed Accuracy)
            # 逻辑：如果 Truth 是 Both，那么选 A 或 B 或 Both 都算对
            relaxed_correct = 0
            for t, p in valid_pairs:
                if t == p:
                    relaxed_correct += 1
                elif t == "Both are equally good" and p in ["Model A", "Model B"]:
                    relaxed_correct += 1
            
            relaxed_acc = relaxed_correct / len(valid_pairs)
            logging.info(f"🤝 松弛裁决准确率 (Relaxed Acc): {relaxed_acc:.4f} (含 Both 兼容)")
            
        else:
            strict_acc = 0.0
            logging.warning("无有效裁决样本")

        # 2. 诊断准确率
        valid_diag = [(t, p) for t, p in zip(all_ground_truths, all_diagnosis_predictions) if t!="Unknown" and p!="Unknown"]
        valid_ground_truths = []
        valid_diagnosis_preds = []
        diag_acc = 0.0
        
        if valid_diag:
            valid_ground_truths, valid_diagnosis_preds = zip(*valid_diag)
            diag_acc = accuracy_score(valid_ground_truths, valid_diagnosis_preds)
            logging.info(f"🏆 诊断准确率 (Diagnosis Acc): {diag_acc:.4f}")
        
        self.model.config.use_cache = False
        
        # 返回值保持不变，用 strict_acc 兼容旧接口
        return avg_loss, strict_acc, diag_acc, valid_ground_truths, valid_diagnosis_preds, winner_distribution, detailed_records
    def predict_judge(self, prompt: str, max_new_tokens: int = 1024) -> str:
        """
        修改：返回裁决结果和对应的诊断预测
        """
        self.model.eval()
        self.model.config.use_cache = True
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.2,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # 解码生成部分
        decoded_output = self.tokenizer.decode(generated_ids[0, inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        
        # 解析winner
        def parse_winner_from_output(json_string):
            try:
                if "```json" in json_string:
                    json_match = re.search(r'```json\s*(.*?)\s*```', json_string, re.DOTALL)
                    if json_match:
                        json_string = json_match.group(1)
                
                data = json.loads(json_string)
                winner = data.get("winner", "").strip()
                return winner
            except:
                return "Parse_Failed"
        
        winner = parse_winner_from_output(decoded_output)
        
        return {
            'raw_output': decoded_output,
            'winner': winner,
            'diagnosis_insight': f"Judge选择了{winner}，对应的诊断结果将基于该模型的输出"
        }
    @classmethod
    def from_adapter(cls, base_model_name, adapter_dir, tuning_method='qlora'):
        logging.info(f"Loading model from adapter: {adapter_dir}")
        if tuning_method == 'qlora':
            quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
        elif tuning_method == 'lora':
            quant_config = BitsAndBytesConfig(load_in_8bit=True)
        else:
            quant_config = None
        tokenizer = AutoTokenizer.from_pretrained(adapter_dir)
        tokenizer.padding_side = 'left'
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name, quantization_config=quant_config, torch_dtype=torch.bfloat16,
            device_map="auto", trust_remote_code=True, attn_implementation="eager")
        base_model.resize_token_embeddings(len(tokenizer))
        
        # Load adapter AFTER resizing
        model = PeftModel.from_pretrained(base_model, adapter_dir)
        
        logging.info("Merging LoRA adapter for inference...")
        model = model.merge_and_unload()
        logging.info("Adapter merged.")
        
        instance = object.__new__(cls)
        instance.model = model
        instance.tokenizer = tokenizer
        instance.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        return instance