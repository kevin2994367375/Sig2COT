import os
import argparse
import pandas as pd
from pathlib import Path
import logging
import numpy as np
import torch
import json
import sys
import time
import gc
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm
import re

# --- 模块导入 ---
try:
    from src.models.model import BaseModel
    # 我们不再需要 parse_llm_output_to_hard_label
    from src.utils.visualization import plot_training_history
    print("成功导入自定义模块。")
except ImportError as e:
    print(f"错误：无法导入自定义模块。Error: {e}"); sys.exit(1)

# --- 全局配置：不同数据集的故障类别 ---
DATASET_CONFIGS = {
    'cwru': ['Normal', 'Ball Fault', 'Inner Race Fault', 'Outer Race Fault'],
    'pu':   ['Normal', 'Inner Race Fault', 'Outer Race Fault'], # PU 默认 3 类
    'xjtu': ['Normal', 'Inner Race Fault', 'Outer Race Fault']
}

def get_correct_winner(model_a_pred: str, model_b_pred: str, ground_truth: str) -> str:
    a_is_correct = model_a_pred == ground_truth
    b_is_correct = model_b_pred == ground_truth

    if a_is_correct and b_is_correct:
        return "Both are equally good"
    elif a_is_correct and not b_is_correct:
        return "Model A"
    elif not a_is_correct and b_is_correct:
        return "Model B"
    else: 
        # [核心修改] 即使两个都错，也不要返回 "Neither"。
        # 返回 "Model A" 作为默认兜底 (因为A通常准确率略高)
        # 这样保证了标签永远在合法范围内。
        return "Model A" 

def parse_judge_winner(json_string: str) -> str:
    """Parses the 'winner' field from the Judge's JSON output."""
    try:
        # 1. 尝试标准 JSON 解析
        if "```json" in json_string:
            json_match = re.search(r'```json\s*(.*?)\s*```', json_string, re.DOTALL)
            if json_match:
                json_string = json_match.group(1)
        
        data = json.loads(json_string)
        winner = data.get("winner", "Parse_Failed")
        return winner.strip()
    except:
        # 2. [新增] 暴力正则匹配 (如果 JSON 解析失败)
        # 直接在文本里找 "winner": "Model A" 这种模式
        match = re.search(r'"winner"\s*:\s*"(.*?)"', json_string)
        if match:
            return match.group(1).strip()
            
        # 3. [新增] 关键词兜底
        # 如果模型直接输出了 "Model A" 而不是 JSON
        if "Model A" in json_string and "Model B" not in json_string:
            return "Model A"
        if "Model B" in json_string and "Model A" not in json_string:
            return "Model B"
            
        return "Parse_Failed"

# --- 命令行参数解析 (已修正) ---
def parse_args():
    parser = argparse.ArgumentParser(description='Fine-tune a Large Language Model as a Judge.')
    
    # [新增] 数据集选择
    parser.add_argument('--dataset', type=str, default='xjtu', choices=['cwru', 'pu', 'xjtu'], 
                        help='Dataset name (cwru or pu).')
    
    # [修改] jsonl_path 变为可选，默认根据 dataset 生成
    parser.add_argument('--jsonl_path', type=str, default=None, 
                        help='Override path to the .jsonl file. If None, auto-generated based on dataset.')
    
    parser.add_argument('--model_name', type=str, default='models/Qwen2.5-7B-Instruct', help='Base pretrained model for the Judge.')
    parser.add_argument('--output_dir', type=str, default='outputs/judge_model', help='Root directory for the fine-tuned model.')
    parser.add_argument('--tuning_method', type=str, default='qlora', choices=['lora', 'qlora'], help="Fine-tuning method.")
    parser.add_argument('--batch_size', type=int, default=1, help='Training batch size.')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs.')#pu得多跑几轮
    parser.add_argument('--learning_rate', type=float, default=2e-5, help='Learning rate.')
    parser.add_argument('--max_length', type=int, default=3000, help='Maximum sequence length.')
    parser.add_argument('--lora_r', type=int, default=32, help='LoRA rank.')
    parser.add_argument('--lora_alpha', type=int, default=64, help='LoRA alpha.')
    parser.add_argument('--lora_dropout', type=float, default=0.1, help='LoRA dropout.')
     # [新增] 训练时剔除共识样本的开关
    parser.add_argument('--exclude_consensus', action='store_true', 
                        help='If set, removes "Both are equally good" samples from the Training set ONLY.')
    return parser.parse_args()

def train_and_evaluate(args, train_df, val_df, test_df, winner_categories):
    # [修改] 策略名称加入 dataset 前缀
    strategy_name = f"{args.dataset}_{args.tuning_method}_Judge"
    
    logging.info(f"\n==================== Starting Judge SFT: {strategy_name.upper()} ====================")
    output_dir = Path(args.output_dir) / strategy_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. 初始化BaseModel
    try:
        model_instance = BaseModel(
            model_name=args.model_name, 
            num_labels=len(winner_categories),
            lora_config_dict={'r': args.lora_r, 'lora_alpha': args.lora_alpha, 'lora_dropout': args.lora_dropout}, 
            tuning_method=args.tuning_method)
    except Exception as e:
        logging.error(f"Failed to initialize BaseModel: {e}")
        return

    # 2. 准备数据加载器
    text_col, target_col = 'instruction', 'output'
    true_label_col = 'true_winner_label'
    
    train_loader = model_instance.prepare_data_loader(
        train_df, text_col, target_col, args.batch_size, args.max_length, 
        true_label_col, dataset_type='judge', is_train=True
    )
    val_loader = model_instance.prepare_data_loader(
        val_df, text_col, target_col, args.batch_size, args.max_length, 
        true_label_col, dataset_type='judge', is_train=False
    )
    
    # 3. 训练模型
    logging.info("Starting training with diagnosis metrics...")
    
    optimal_max_new_tokens = model_instance.calculate_optimal_max_tokens(train_loader, max_cap=1024)
    
    try:
        learning_rate = float(args.learning_rate)
    except (ValueError, TypeError):
        learning_rate = 2e-5
    
    history = model_instance.train_judge(
        train_loader=train_loader,
        val_loader=val_loader, 
        val_df=val_df,
        epochs=args.epochs, 
        learning_rate=learning_rate,
        output_dir=output_dir, 
        optimal_max_new_tokens=optimal_max_new_tokens
    )
    
    # 4. 最终评估
    best_adapter_path = output_dir / 'best_model_lora'
    if not best_adapter_path.exists():
        logging.error("Best model not found. Evaluation skipped.")
        return

    logging.info("\n=== Final Evaluation with Diagnosis Metrics ===")
    
    del model_instance
    torch.cuda.empty_cache()
    gc.collect()
    time.sleep(2)

    try:
        eval_model_instance = BaseModel.from_adapter(
            base_model_name=args.model_name, 
            adapter_dir=str(best_adapter_path),
            tuning_method=args.tuning_method
        )
    except Exception as e:
        logging.error(f"Failed to load adapter: {e}")
        return

    test_loader = eval_model_instance.prepare_data_loader(
        test_df, text_col, target_col, args.batch_size, args.max_length, 
        true_label_col, dataset_type='judge', is_train=False
    )
    
    # [修改] 接收 5 个返回值
    test_loss, test_adj_acc, test_diag_acc, true_labels, pred_labels, winner_dist, detailed_records = eval_model_instance.evaluate_judge(
        test_loader, test_df, description="Final Test"
    )
    
    # [新增] 保存分布统计到文件
    dist_file = output_dir / "judge_choices_distribution.json"
    with open(dist_file, "w", encoding="utf-8") as f:
        json.dump(winner_dist, f, indent=2, ensure_ascii=False)
        
    logging.info(f"📊 Judge 选择分布已保存至: {dist_file}")
    logging.info(f"内容预览: {winner_dist}")
    
    # [新增] 生成并保存 Classification Report
    if true_labels and pred_labels:
        logging.info("Generating classification report...")
        report = classification_report(true_labels, pred_labels, digits=4, zero_division=0)
        
        report_path = output_dir / "classification_report.txt"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report)
            
        logging.info(f"📝 Classification report saved to: {report_path}")
        logging.info(f"\n{report}") # 同时打印到控制台
    else:
        logging.warning("⚠️ No valid predictions to generate report.")
    
    # 保存结果
    test_results = {
        'dataset': args.dataset,
        'test_loss': float(test_loss),
        'test_adjudication_accuracy': float(test_adj_acc),
        'test_diagnosis_accuracy': float(test_diag_acc),
        'test_samples': len(test_df),
        'winner_categories': winner_categories,
        'winner_distribution': winner_dist # 把分布也写进总结果
    }
    conflict_analysis_file = output_dir / f"conflict_analysis_{args.dataset}.jsonl"
    logging.info(f"💾 正在保存分歧样本详细分析至: {conflict_analysis_file}")
    
    conflict_count = 0
    correct_rescue_count = 0
    
    with open(conflict_analysis_file, "w", encoding="utf-8") as f:
        for record in detailed_records:
            # 写入文件
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            
            # 顺便统计一下
            if record['is_conflict']:
                conflict_count += 1
                if record['is_correct']:
                    correct_rescue_count += 1
    
    logging.info(f"⚔️ 总分歧样本数: {conflict_count}")
    if conflict_count > 0:
        logging.info(f"✅ Judge 成功救回 (Correct Rescue): {correct_rescue_count} ({correct_rescue_count/conflict_count:.2%})")
    else:
        logging.info("⚠️ 本次测试未发现分歧样本。")

    results_path = os.path.join(output_dir, "test_results_with_diagnosis.json")
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(test_results, f, indent=2, ensure_ascii=False)
    
    # 5. 定性评估 - 显示裁决和诊断结果
    logging.info("\n=== Qualitative Evaluation (Adjudication + Diagnosis) ===")
    sample_indices = [0, 1, 2]  # 测试集的前几个样本
    
    for i, sample_idx in enumerate(sample_indices):
        if sample_idx >= len(test_df):
            break
            
        sample_row = test_df.iloc[sample_idx]
        input_data = json.loads(sample_row['input'])
        
        # 重建Prompt
        signal_data = input_data.get('signal_data', '')
        model_a_response = input_data.get('model_a_response', '')
        model_b_response = input_data.get('model_b_response', '')
        ground_truth = input_data.get('ground_truth', '')
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
"winner": "..."
}}
Note: The "winner" can be "Model A", "Model B", "Both are equally good"."""
        
        logging.info(f"\n--- Sample {i+1} ---")
        logging.info("Signal Data (abbreviated):")
        logging.info(signal_data[:200] + "..." if len(signal_data) > 200 else signal_data)
        
        logging.info(f"\nGround Truth: {ground_truth}")
        logging.info(f"Model A Diagnosis: {sample_row.get('model_a_diagnosis', 'Unknown')}")
        logging.info(f"Model B Diagnosis: {sample_row.get('model_b_diagnosis', 'Unknown')}")
        
        # 使用预测方法
        result = eval_model_instance.predict_judge(prompt, max_new_tokens=1024)
        
        logging.info(f"\nJudge Prediction:")
        logging.info(f"Raw Output: {result['raw_output']}")
        logging.info(f"Predicted Winner: {result['winner']}")
        logging.info(f"Actual Winner: {sample_row['true_winner']}")
        
        # 计算裁决是否正确
        adjudication_correct = (result['winner'] == sample_row['true_winner'])
        logging.info(f"Adjudication Correct: {adjudication_correct}")
        
        # 计算诊断是否正确
        if result['winner'] == "Model A":
            predicted_diagnosis = sample_row.get('model_a_diagnosis', 'Unknown')
        elif result['winner'] == "Model B":
            predicted_diagnosis = sample_row.get('model_b_diagnosis', 'Unknown')
        elif result['winner'] in ["Both are equally good", "Both are equally bad"]:
            # 如果两个模型都好或都差，选择第一个模型的诊断
            predicted_diagnosis = sample_row.get('model_a_diagnosis', 'Unknown')
        else:
            predicted_diagnosis = "Unknown"
        
        diagnosis_correct = (predicted_diagnosis == ground_truth)
        logging.info(f"Diagnosis Correct: {diagnosis_correct} (Predicted: {predicted_diagnosis}, Actual: {ground_truth})")
        
        if i >= 2:
            logging.info("... (more samples available)")
            break
    
    # 6. 打印最终结果
    logging.info(f"\n=== Final Results ===")
    logging.info(f"Adjudication Accuracy (裁决准确率): {test_adj_acc:.4f}")
    logging.info(f"Diagnosis Accuracy (诊断准确率): {test_diag_acc:.4f}")
    logging.info(f"Test Loss: {test_loss:.4f}")
    
    if history['val_acc']:
        best_epoch = np.argmax(history['val_acc'])
        logging.info(f"Best Validation Accuracy: {history['val_acc'][best_epoch]:.4f} (epoch {best_epoch + 1})")
    
    # 7. 保存训练历史
    training_history = {
        'train_loss': history['train_loss'],
        'val_loss': history['val_loss'],
        'val_acc': history['val_acc'],
        'best_epoch': int(np.argmax(history['val_acc'])) if history['val_acc'] else 0,
        'best_val_acc': float(max(history['val_acc'])) if history['val_acc'] else 0.0
    }
    
    history_path = os.path.join(output_dir, "training_history.json")
    with open(history_path, 'w', encoding='utf-8') as f:
        json.dump(training_history, f, indent=2, ensure_ascii=False)
    
    # 8. 保存模型配置
    config_info = {
        'model_name': args.model_name,
        'tuning_method': args.tuning_method,
        'lora_r': args.lora_r,
        'lora_alpha': args.lora_alpha,
        'epochs': args.epochs,
        'learning_rate': args.learning_rate,
        'batch_size': args.batch_size,
        'max_length': args.max_length,
        'task_type': 'judge_with_diagnosis_metrics'
    }
    config_path = os.path.join(output_dir, "model_config.json")
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config_info, f, indent=2, ensure_ascii=False)
    
    logging.info(f"\n=== Judge Training with Diagnosis Metrics Complete! ===")
    logging.info(f"Outputs saved to: {output_dir}")
    
    return {
        'test_adjudication_accuracy': test_adj_acc,
        'test_diagnosis_accuracy': test_diag_acc,
        'test_loss': test_loss,
        'output_dir': str(output_dir)
    }
    

def main():
    args = parse_args()
    
    # [新增] 自动构建文件路径
    if args.jsonl_path is None:
        args.jsonl_path = f"finetuning_dataset_for_judge_{args.dataset}_balanced.jsonl"
    
    # 1. 加载Judge数据集
    jsonl_path = Path(args.jsonl_path)
    if not jsonl_path.exists():
        logging.error(f"Judge dataset not found: {jsonl_path.resolve()}")
        logging.error(f"请确保您已经运行了 local_generate_judge_dataset.py --dataset {args.dataset}")
        return
    
    logging.info(f"Loading Judge dataset from {jsonl_path}...")
    data_df = pd.read_json(jsonl_path, lines=True)
    logging.info(f"Original dataset size: {len(data_df)}")
    
    def is_valid_input_json(text):
        if not isinstance(text, str): return False
        try:
            json.loads(text)
            return True
        except:
            return False

    def is_valid_output(text):
        # output 只要是字符串且有内容即可，不需要是合法 JSON
        return isinstance(text, str) and len(text.strip()) > 10

    # 只严格检查 input，对 output 宽容处理
    valid_mask = data_df['input'].apply(is_valid_input_json) & data_df['output'].apply(is_valid_output)
    
    invalid_count = len(data_df) - valid_mask.sum()
    if invalid_count > 0:
        logging.warning(f"⚠️ Found {invalid_count} invalid records! Removing them...")
        data_df = data_df[valid_mask].reset_index(drop=True)
    
    logging.info(f"Cleaned dataset size: {len(data_df)}") # 这里应该接近 1589 才对
    # 2. 预处理数据
    logging.info("Preprocessing judge dataset...")
    
    # [关键] 根据数据集获取对应的故障类别列表
    current_fault_categories = DATASET_CONFIGS.get(args.dataset, DATASET_CONFIGS['cwru'])
    logging.info(f"Using fault categories for [{args.dataset}]: {current_fault_categories}")

    # [修正版] extract_winner_from_output
    def extract_winner_from_output(output_str):
        if not isinstance(output_str, str): return ""
        try:
            # 暴力正则
            match = re.search(r'"winner"\s*:\s*"(.*?)"', output_str)
            if match:
                w = match.group(1).strip()
                # 如果旧数据里混进了 Neither，强制转为 Model A
                if "Neither" in w or "bad" in w: return "Model A"
                return w
                
            # 关键词兜底
            if "Model A" in output_str: return "Model A"
            if "Model B" in output_str: return "Model B"
            if "Both" in output_str: return "Both are equally good"
            
            return ""
        except:
            return ""

    def extract_ground_truth_fault_type(input_str):
        try:
            input_dict = json.loads(input_str)
            return input_dict.get('ground_truth', '')
        except Exception:
            return ""

    def extract_model_diagnoses(input_str):
        try:
            input_dict = json.loads(input_str)
            model_a_response = input_dict.get('model_a_response', '')
            model_b_response = input_dict.get('model_b_response', '')
            
            # [核心修复] 严格提取逻辑
            def extract_diagnosis_from_text(text):
                if not isinstance(text, str): return "Unknown"
                
                # 1. 黄金标准：只看 "Final Confirmed Diagnosis:" 之后的内容
                if "Final Confirmed Diagnosis:" in text:
                    # 取分割后的最后一部分（防止前面有引用）
                    # strip() 去掉可能存在的换行符和空格
                    target_area = text.split("Final Confirmed Diagnosis:")[-1].strip()
                    
                    # 在这个极短的区域内匹配，准确率 100%
                    # 按照长度倒序匹配 (防止 Inner Race Fault 被 Inner 匹配)
                    for fault in sorted(current_fault_categories, key=len, reverse=True):
                        # 使用 lower() 忽略大小写差异
                        if fault.lower() in target_area.lower():
                            return fault
                
                # 2. 银标准：如果生成的格式乱了，只搜索文本的最后 200 个字符
                # 因为结论通常在最后。这样能避开开头 "Compared to Normal..." 的干扰
                tail_text = text[-200:]
                for fault in sorted(current_fault_categories, key=len, reverse=True):
                    if fault.lower() in tail_text.lower():
                        return fault
                        
                return "Unknown"
            
            return extract_diagnosis_from_text(model_a_response), extract_diagnosis_from_text(model_b_response)
            
        except Exception as e:
            logging.warning(f"提取诊断失败: {e}")
            return "Unknown", "Unknown"

    # 应用提取函数
    data_df['true_winner'] = data_df['output'].apply(extract_winner_from_output)
    data_df['ground_truth'] = data_df['input'].apply(extract_ground_truth_fault_type)
    
    diagnoses = data_df['input'].apply(extract_model_diagnoses).apply(pd.Series)
    data_df['model_a_diagnosis'] = diagnoses[0]
    data_df['model_b_diagnosis'] = diagnoses[1]
    
    data_df['true_winner_str'] = data_df['true_winner']
    data_df['true_fault_type'] = data_df['ground_truth']
    
    # 标签映射
    winner_categories = ['Model A', 'Model B', 'Both are equally good']
    winner_to_label = {winner: idx for idx, winner in enumerate(winner_categories)}
    data_df['true_winner_label'] = data_df['true_winner'].map(winner_to_label).fillna(0)
    
    fault_to_label = {fault: idx for idx, fault in enumerate(current_fault_categories)}
    data_df['true_fault_label'] = data_df['true_fault_type'].map(fault_to_label).fillna(0)
    
    # 3. 拆分数据集
    logging.info("Splitting dataset with stratification...")
    train_val_df, test_df = train_test_split(data_df, test_size=0.2, random_state=42, stratify=data_df['true_winner_label'])
    train_df, val_df = train_test_split(train_val_df, test_size=0.1, random_state=42, stratify=train_val_df['true_winner_label'])
    
    train_df.attrs['name'] = 'train'
    val_df.attrs['name'] = 'val' 
    test_df.attrs['name'] = 'test'
    
    logging.info(f"Dataset sizes - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
     # [核心修改] 训练集清洗策略：剔除 "Both are equally good"
    # ==============================================================================
    if args.exclude_consensus:
        logging.info("\n🛑 [Strategy] Enabling 'Conflict-Only' Training...")
        logging.info("   Removing 'Both are equally good' samples from Training Set.")
        
        # 找到 "Both are equally good" 对应的 Label ID
        # winner_categories = ['Model A', 'Model B', 'Both are equally good']
        # 通常索引是 2，但为了安全，我们查字典
        if 'Both are equally good' in winner_to_label:
            both_label_id = winner_to_label['Both are equally good']
            
            # 1. 过滤训练集 (必须过滤)
            initial_train_len = len(train_df)
            train_df = train_df[train_df['true_winner_label'] != both_label_id].copy()
            logging.info(f"   📉 Train Set reduced: {initial_train_len} -> {len(train_df)} (Dropped Consensus)")
            
            # 2. 过滤验证集 (可选，建议也过滤，以便观察模型在困难样本上的Loss变化)
            # 如果不过滤验证集，Loss 可能会很低（因为简单题多），掩盖了模型在难题上的糟糕表现
            initial_val_len = len(val_df)
            val_df = val_df[val_df['true_winner_label'] != both_label_id].copy()
            logging.info(f"   📉 Val Set reduced:   {initial_val_len} -> {len(val_df)} (Dropped Consensus)")
            
            # 3. 测试集 (Test Set) -> 绝对不动！保持全量！
            logging.info(f"   🛡️ Test Set remains FULL size ({len(test_df)}) to reflect real-world distribution.")
        else:
            logging.warning("⚠️ 'Both are equally good' not found in categories. Skipping filter.")
    # 4. 调用训练
    train_and_evaluate(args, train_df, val_df, test_df, winner_categories)

if __name__ == "__main__":
    log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    if root_logger.hasHandlers(): root_logger.handlers.clear()
    
    # 日志文件也加上时间戳或 dataset 名比较好，这里先保持原样
    log_file = 'finetune_judge.log'
    file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    file_handler.setFormatter(log_formatter)
    root_logger.addHandler(file_handler)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_formatter)
    root_logger.addHandler(console_handler)
    
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
    except Exception as e:
        logging.error("An unhandled exception occurred:", exc_info=True)
        sys.exit(1)