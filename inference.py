import os
import torch
import json
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import TypedDict, Literal
from sklearn.metrics import classification_report, accuracy_score

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

from langchain_core.language_models import LLM
from langchain_core.prompts import PromptTemplate
from langgraph.graph import StateGraph, END

# ==============================================================================
# 1. 核心类与函数 (LLM Wrapper & Utils)
# ==============================================================================
class LocalLoRALLM(LLM):
    """LangChain Wrapper for Local LoRA Model (4-bit)"""
    model: object = None
    tokenizer: object = None
    
    def __init__(self, base_model_path, adapter_path):
        super().__init__()
        print(f"🚀 [Loading] Base: {base_model_path} | Adapter: {adapter_path}")
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
        self.tokenizer.padding_side = 'left'
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True
        )
        
        # 关键：调整 Embedding 大小
        print(f"🔄 Resizing embeddings: {base_model.get_input_embeddings().weight.shape[0]} -> {len(self.tokenizer)}")
        base_model.resize_token_embeddings(len(self.tokenizer))
        
        self.model = PeftModel.from_pretrained(base_model, adapter_path)
        self.model.eval()

    @property
    def _llm_type(self) -> str:
        return "custom_lora_llm"

    def _call(self, prompt: str, stop=None, run_manager=None, **kwargs) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True).to(self.model.device)
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=1024,
                do_sample=False,
                repetition_penalty=1.1,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        input_len = inputs.input_ids.shape[1]
        return self.tokenizer.decode(generated_ids[0][input_len:], skip_special_tokens=True)

# 故障关键词 (用于提取和比对)
FAULT_KEYWORDS = ['normal', 'inner', 'outer', 'ball']

def extract_diagnosis(text):
    if not isinstance(text, str): return "Unknown"
    text_lower = text.lower()
    
    # 1. 优先匹配标准结尾
    if "final confirmed diagnosis:" in text_lower:
        tail = text_lower.split("final confirmed diagnosis:")[-1]
        for kw in FAULT_KEYWORDS:
            if kw in tail: return kw # 返回关键词本身，后续做标准化映射
            
    # 2. 兜底匹配
    for kw in FAULT_KEYWORDS:
        if kw in text_lower: return kw
    return "Unknown"

def is_diagnosis_same(diag_a, diag_b):
    a = diag_a.lower().strip()
    b = diag_b.lower().strip()
    if a == "unknown" or b == "unknown": return False
    return (a in b) or (b in a)

def parse_judge_json(text):
    try:
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0]
        data = json.loads(text.strip())
        return data.get("winner", "Unknown")
    except:
        if "Model A" in text: return "Model A"
        if "Model B" in text: return "Model B"
        if "Both" in text: return "Both are equally good"
        return "Unknown"

def merge_feature_texts(text_time, text_other):
    """合并时频特征用于 Judge"""
    # 简单合并逻辑，实际可复用之前 smart_judge 的逻辑
    return str(text_time) + "\n\n[Frequency Domain Features]\n" + str(text_other)

# ==============================================================================
# 2. Graph 构建 (LangGraph)
# ==============================================================================
class AgentState(TypedDict):
    input_time_features: str
    input_freq_features: str
    full_input_merged: str
    time_analysis: str
    freq_analysis: str
    final_diagnosis: str
    source: str

EXPERT_PROMPT = PromptTemplate.from_template(
    """As a rotating machinery diagnostics expert, your task is to write a detailed Chain-of-Thought (CoT) process.
This CoT must logically explain how the data in the provided "Signal Analysis Summary" leads to its "Final Confirmed Diagnosis".

**Key requirements for your CoT:**
1. Your reasoning must be clear, step-by-step, and based *only* on the provided data.
2. Use professional and accurate terminology.
3. **Keep your explanation concise, around 100 words.** This is a brief summary of your thoughts.
4. **Give your reply in English.**
---
**Signal Analysis Summary:**
{features}
---

**Your Chain-of-Thought:**
"""
)

JUDGE_PROMPT = PromptTemplate.from_template(
    """As a final arbiter and expert diagnostician, your task is to evaluate the reasoning processes of two different AI models (Model A and Model B) that have analyzed the same bearing signal data.

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
"winner": "..."
}}
Note: The "winner" can be "Model A", "Model B", "Both are equally good"."""
)

def build_workflow(llm_time, llm_freq, llm_judge):
    def time_node(state):
        return {"time_analysis": llm_time.invoke(EXPERT_PROMPT.format(features=state["input_time_features"]))}
    
    def freq_node(state):
        return {"freq_analysis": llm_freq.invoke(EXPERT_PROMPT.format(features=state["input_freq_features"]))}
    
    def router(state):
        da = extract_diagnosis(state["time_analysis"])
        db = extract_diagnosis(state["freq_analysis"])
        return "bypass" if is_diagnosis_same(da, db) else "judge"
    
    def bypass_node(state):
        return {"final_diagnosis": extract_diagnosis(state["time_analysis"]), "source": "Smart Bypass"}
    
    def judge_node(state):
        resp = llm_judge.invoke(JUDGE_PROMPT.format(
            signal_data=state["full_input_merged"],
            model_a_response=state["time_analysis"],
            model_b_response=state["freq_analysis"]
        ))
        winner = parse_judge_json(resp)
        final = extract_diagnosis(state["freq_analysis"]) if winner == "Model B" else extract_diagnosis(state["time_analysis"])
        return {"final_diagnosis": final, "source": f"Judge ({winner})"}

    workflow = StateGraph(AgentState)
    workflow.add_node("time", time_node)
    workflow.add_node("freq", freq_node)
    workflow.add_node("bypass", bypass_node)
    workflow.add_node("judge", judge_node)
    
    workflow.set_entry_point("time")
    workflow.add_edge("time", "freq")
    workflow.add_conditional_edges("freq", router, {"bypass": "bypass", "judge": "judge"})
    workflow.add_edge("bypass", END)
    workflow.add_edge("judge", END)
    
    return workflow.compile()

# ==============================================================================
# 3. 主程序 (Data Loading & Evaluation)
# ==============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="pu")
    
    # 路径配置 (请修改为你实际的路径)
    base_out = Path("outputs")
    parser.add_argument("--time_adapter", default=str(base_out / "Qwen2.5-3B-Instruct_pu_lora_SFT_time_domain/best_model_lora"))
    parser.add_argument("--freq_adapter", default=str(base_out / "Qwen2.5-3B-Instruct_pu_lora_SFT_other_features/best_model_lora"))
    parser.add_argument("--judge_adapter", default=str(base_out / "judge_model/Qwen2.5-7B-Instruct_pu_qlora_Judge/best_model_lora"))
    parser.add_argument("--base_model", default="models/Qwen2.5-3B-Instruct") 
    parser.add_argument("--judge_base", default="models/Qwen2.5-7B-Instruct")
    
    args = parser.parse_args()

    # 1. 加载数据 (Final Test Set)
    print("📂 Loading Final Test Data...")
    processed_dir = Path("data/processed")
    
    # 读取原始 CSV
    df_time = pd.read_csv(processed_dir / f"processed_dataset_{args.dataset}_time_domain.csv")
    df_freq = pd.read_csv(processed_dir / f"processed_dataset_{args.dataset}_other_features.csv")
    
    # 读取索引
    index_file = processed_dir / f"{args.dataset}_final_test_indices.json"
    if not index_file.exists():
        print(f"⚠️ Final Test Indices not found. Using standard Test Indices.")
        index_file = processed_dir / f"{args.dataset}_test_indices.json"
        
    with open(index_file, 'r') as f:
        test_indices = json.load(f)
        
    # 切分出测试集
    test_df_time = df_time.iloc[test_indices].reset_index(drop=True)
    test_df_freq = df_freq.iloc[test_indices].reset_index(drop=True)
    
    print(f"✅ Loaded {len(test_df_time)} samples for Final Testing.")

    # 2. 初始化模型与图
    print("🤖 Initializing Agents...")
    llm_time = LocalLoRALLM(args.base_model, args.time_adapter)
    llm_freq = LocalLoRALLM(args.base_model, args.freq_adapter)
    llm_judge = LocalLoRALLM(args.judge_base, args.judge_adapter)
    
    app = build_workflow(llm_time, llm_freq, llm_judge)

    # 3. 批量推理
    results = []
    y_true = []
    y_pred = []
    
    print("🚀 Starting Inference Pipeline...")
    
    # 定义标准标签映射 (用于计算指标)
    # 必须与训练时的 label_map 一致
    if args.dataset == 'pu':
        label_map = {0: 'normal', 1: 'inner', 2: 'outer'} 
    elif args.dataset == 'cwru':
        label_map = {0: 'normal', 1: 'ball', 2: 'inner', 3: 'outer'}
    elif args.dataset == 'xjtu':
        label_map = {0: 'normal', 1: 'inner', 2: 'outer'}
    
    for idx, row_t in tqdm(test_df_time.iterrows(), total=len(test_df_time)):
        row_f = test_df_freq.iloc[idx]
        
        # 准备 Input
        input_data = {
            "input_time_features": row_t['llm_input_text'],
            "input_freq_features": row_f['llm_input_text'],
            "full_input_merged": merge_feature_texts(row_t['llm_input_text'], row_f['llm_input_text'])
        }
        
        # 运行图
        try:
            output = app.invoke(input_data)
            final_pred_raw = output['final_diagnosis']
            source = output['source']
        except Exception as e:
            print(f"❌ Error on sample {idx}: {e}")
            final_pred_raw = "unknown"
            source = "Error"
            
        # 记录
        gt_numeric = row_t['label_numeric']
        gt_str = label_map.get(gt_numeric, "unknown")
        
        # 简单的标准化 (把 "Inner Race Fault" 转为 "inner")
        pred_std = "unknown"
        for kw in FAULT_KEYWORDS:
            if kw in final_pred_raw.lower():
                pred_std = kw
                break
                
        y_true.append(gt_str)
        y_pred.append(pred_std)
        
        results.append({
            "id": int(test_indices[idx]),
            "ground_truth": gt_str,
            "prediction": pred_std,
            "raw_prediction": final_pred_raw,
            "source": source,
            "is_correct": gt_str == pred_std
        })

    # 4. 计算指标与保存
    print("\n" + "="*50)
    print("🏆 Final Test Results")
    print("="*50)
    
    acc = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, digits=4)
    
    print(f"Accuracy: {acc:.4f}")
    print("\nClassification Report:\n")
    print(report)
    
    # 保存
    save_dir = Path("outputs/final_results")
    save_dir.mkdir(parents=True, exist_ok=True)
    
    with open(save_dir / f"{args.dataset}_final_report.txt", "w") as f:
        f.write(f"Accuracy: {acc:.4f}\n\n{report}")
        
    pd.DataFrame(results).to_json(save_dir / f"{args.dataset}_detailed_predictions.jsonl", orient='records', lines=True)
    print(f"✅ Results saved to {save_dir}")

if __name__ == "__main__":
    main()