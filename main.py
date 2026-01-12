# ==============================================================================
#                 main_final_manual_loop_complete.py (最终手动循环完整版)
# ==============================================================================
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
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm

# --- 模块导入 ---
try:
    from src.models.model import BaseModel, parse_llm_output_to_hard_label
    from src.utils.visualization import plot_training_history, plot_confusion_matrix, plot_classification_report
    from src.data.data_loader import split_data
    print("成功导入自定义模块。")
except ImportError as e:
    print(f"错误：无法导入自定义模块。Error: {e}"); sys.exit(1)


# --- 命令行参数解析 ---
def parse_args():
    parser = argparse.ArgumentParser(description='Fine-tune a Large Language Model from a preprocessed .csv file')
    parser.add_argument('--task', type=str, default='time_domain', choices=['time_domain', 'other_features'], help="Specify which preprocessed dataset to use.")
    parser.add_argument('--model_name', type=str, default='models/Qwen2.5-3B-Instruct', help='Path to the base pretrained model folder.')
    parser.add_argument('--output_dir', type=str, default='outputs', help='Root directory for all outputs.')
    parser.add_argument('--dataset', type=str, default='pu', choices=['cwru', 'pu', 'xjtu'], help="Dataset name")
    parser.add_argument('--tuning_method', type=str, default='lora', choices=['lora', 'qlora'], help="Fine-tuning method.")
    parser.add_argument('--batch_size', type=int, default=1, help='Training batch size.')
    parser.add_argument('--epochs', type=int, default=6, help='Number of training epochs.')
    parser.add_argument('--acc_freq', type=int, default=2, 
                    help='How many epochs to wait before calculating generative accuracy (default: 1).')
    parser.add_argument('--learning_rate', type=float, default=2e-4, help='Learning rate.')
    parser.add_argument('--max_length', type=int, default=1024, help='Maximum sequence length.')
    parser.add_argument('--lora_r', type=int, default=32, help='LoRA rank.')
    parser.add_argument('--lora_alpha', type=int, default=32, help='LoRA alpha.')
    parser.add_argument('--lora_dropout', type=float, default=0.1, help='LoRA dropout.')
    # [新增] 仅推理模式开关
    parser.add_argument('--inference_only', action='store_true', help='Skip training and only run inference on test set.')
    return parser.parse_args()

def calculate_safe_batch_size(prompts, tokenizer, max_length, base_batch_size=8):
    if not prompts: return 1
    try:
        sample_prompts = prompts[:20]
        avg_length = np.mean([len(tokenizer.encode(p, add_special_tokens=False)) for p in sample_prompts])
        if avg_length > max_length * 0.75: safe_batch_size = max(1, int(base_batch_size / 4))
        elif avg_length > max_length * 0.5: safe_batch_size = max(1, int(base_batch_size / 2))
        else: safe_batch_size = base_batch_size
        logging.info(f"Avg prompt token length: {avg_length:.1f}. Using safe inference batch size: {safe_batch_size}")
        return safe_batch_size
    except Exception: return 1

def clear_memory():
    logging.info("Clearing CUDA memory...")
    if torch.cuda.is_available():
        torch.cuda.synchronize(); torch.cuda.empty_cache()
    gc.collect(); time.sleep(1)
    if torch.cuda.is_available():
        logging.info(f"GPU Memory Allocated: {torch.cuda.memory_allocated()/1024**2:.2f} MB")

def train_and_evaluate(args, train_df, val_df, test_df, class_names, label_col, task_name, dataset_name):
    strategy_name = f"{dataset_name}_{args.tuning_method}_SFT_{task_name}"
    output_dir = Path(args.output_dir) / strategy_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logging.info(f"\n==================== Starting SFT Workflow: {strategy_name.upper()} ====================")
    logging.info(f"Output Directory: {output_dir}") # 打印一下确认路径
    if not args.inference_only:
        logging.info("--> Mode: Training + Inference")
        try:
            model_instance = BaseModel(
                model_name=args.model_name, num_labels=len(class_names),
                lora_config_dict={'r': args.lora_r, 'lora_alpha': args.lora_alpha, 'lora_dropout': args.lora_dropout}, 
                tuning_method=args.tuning_method)
        except Exception as e:
            logging.error(f"Failed to initialize BaseModel: {e}"); return
        text_col, target_col = 'llm_input_text', 'llm_target_output'
        train_loader = model_instance.prepare_data_loader(
            train_df, text_col, target_col, args.batch_size, args.max_length, label_col,
            is_train=True  # <--- 显式指定
        )
        
        # [修改 2] 验证集：传入 is_train=False (触发加速)
        val_loader = model_instance.prepare_data_loader(
            val_df, text_col, target_col, args.batch_size, args.max_length, label_col,
            is_train=False # <--- 显式指定，这会触发 model.py 里的加速逻辑
        )
        optimal_max_new_tokens = model_instance.calculate_optimal_max_tokens(val_loader)
        
        history = model_instance.train(
            train_loader, val_loader, args.epochs, args.learning_rate, 
            str(output_dir), class_names,
            optimal_max_new_tokens=optimal_max_new_tokens,
            acc_freq=args.acc_freq)
        # 清理显存，准备推理
        del model_instance
        clear_memory()
    else:
        logging.info("--> Mode: Inference Only (Skipping Training)")
        # 推理模式下，默认生成长度设为 1024 或其他合适的值
        optimal_max_new_tokens = 1024 
    best_adapter_path = output_dir / 'best_model_lora'
    if not best_adapter_path.exists():
        logging.error(f"Model adapter not found at: {best_adapter_path}"); return

    logging.info(f"\n=== Running Inference on Test Set ({len(test_df)} samples) ===")
    try:
        # 加载最佳模型
        eval_model_instance = BaseModel.from_adapter(
            base_model_name=args.model_name,
            adapter_dir=str(best_adapter_path),
            tuning_method=args.tuning_method)
    except Exception as e:
        logging.error(f"Failed to load adapter for evaluation: {e}"); return

    # 1. 构建所有 Prompts 列表
    logging.info("构建测试集 Prompts...")
    test_prompts = []
    for _, row in test_df.iterrows():
        full_summary_text = str(row['llm_input_text'])
        # 保持与 Dataset 一致的 Prompt
        prompt = f"""As a rotating machinery diagnostics expert, your task is to write a detailed Chain-of-Thought (CoT) process.
This CoT must logically explain how the data in the provided "Signal Analysis Summary" leads to its "Final Confirmed Diagnosis".

**Key requirements for your CoT:**
1.  Your reasoning must be clear, step-by-step, and based *only* on the provided data.
2.  Use professional and accurate terminology.
3.  **Keep your explanation concise, around 100 words.** This is a brief summary of your thoughts.
4.  **Give your reply in English.**
---
**Signal Analysis Summary:**
{full_summary_text}
---

**Your Chain-of-Thought:**
"""
        test_prompts.append(prompt)   
    
    # 2. [核心修改] 调用高效批量预测 (Batch Size = 16)
    # 这会比之前的循环快 10 倍以上
    logging.info(f"开始高效批量推理 (Batch Size=16, Samples={len(test_prompts)})...")
    
    generated_outputs = eval_model_instance.predict(
        prompts=test_prompts,
        batch_size=16,  # 显存允许的话，甚至可以开 32
        max_new_tokens=optimal_max_new_tokens
    )

    # 3. 解析结果 (后续代码保持不变)
    test_labels = test_df[label_col].to_numpy()
    test_preds = [parse_llm_output_to_hard_label(output, class_names) for output in generated_outputs]
    
    # ... (计算指标、保存结果的代码保持不变) ...
    parse_failures = sum(1 for p in test_preds if p == -1)
    if len(test_preds) > 0:
        parse_failure_rate = parse_failures / len(test_preds)
        logging.info(f"Label Parse Failure Rate: {parse_failure_rate:.2%} ({parse_failures}/{len(test_preds)})")

    valid_indices = [i for i, pred in enumerate(test_preds) if pred != -1]
    if not valid_indices:
        logging.error("All test predictions failed to parse! Cannot generate report.")
    else:
        valid_test_labels = test_labels[valid_indices]
        valid_test_preds = [test_preds[i] for i in valid_indices]
        
        report_target_names = [class_names[i] for i in np.unique(np.concatenate((valid_test_labels, valid_test_preds)))]
        
        final_accuracy = accuracy_score(valid_test_labels, valid_test_preds)
        logging.info(f"Final Test Accuracy (on successfully parsed predictions): {final_accuracy:.4f}")
        report_string = classification_report(valid_test_labels, valid_test_preds, target_names=report_target_names, digits=4, zero_division=0)
        logging.info("--- Classification Report ---\n" + report_string)
        with open(output_dir / "classification_report.txt", 'w', encoding='utf-8') as f: f.write(report_string)
        plot_confusion_matrix(valid_test_labels, valid_test_preds, report_target_names, f"CM - {strategy_name}", str(output_dir))

    logging.info("Saving prediction samples...")
    sample_count_to_save = 20
    samples_for_review = []
    for i in range(min(sample_count_to_save, len(test_df))):
        row_data = test_df.iloc[i]
        true_label_name = class_names[row_data[label_col]]
        pred_index = test_preds[i]
        pred_label_name = class_names[pred_index] if pred_index != -1 else "Parse_Failed"
        samples_for_review.append({
            "index": int(row_data.name), "full_prompt": test_prompts[i],
            "model_output": generated_outputs[i], "parsed_prediction": pred_label_name,
            "ground_truth": true_label_name, "is_correct": pred_label_name == true_label_name
        })
        
    full_output_path = output_dir / f"test_predictions_{dataset_name}_{task_name}.jsonl"
    logging.info(f"Saving ALL predictions to: {full_output_path}")
    
    with open(full_output_path, "w", encoding="utf-8") as f:
        for i in range(len(test_df)):
            row_data = test_df.iloc[i]
            true_label_name = class_names[row_data[label_col]]
            
            pred_idx = test_preds[i]
            pred_label_name = class_names[pred_idx] if pred_idx != -1 else "Parse_Failed"
            
            # 这是一个标准的结构，方便后续 Judge 数据构建脚本读取
            record = {
                "id": int(row_data.name) if 'name' in dir(row_data) else i, # 原始索引
                "original_input": row_data['llm_input_text'],
                "full_prompt": test_prompts[i],
                "model_output": generated_outputs[i], # 完整的 CoT
                "parsed_prediction": pred_label_name,
                "ground_truth": true_label_name,
                "is_correct": pred_label_name == true_label_name
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            
    logging.info("Done.")

def main():
    args = parse_args()
    if args.dataset == 'cwru':
        # CWRU 数据集通常包含 4 类
        class_names = ['Normal', 'Ball Fault', 'Inner Race Fault', 'Outer Race Fault']
        
    elif args.dataset == 'pu':
        # PU 数据集通常只关注内圈、外圈和正常（除非你包含混合故障）
        # 注意：这里的顺序必须和 preprocess_split_save.py 生成 CSV 时的顺序完全一致！
        class_names = ['Normal', 'Inner Race Fault', 'Outer Race Fault']
    elif args.dataset == 'xjtu':
        class_names = ['Normal', 'Inner Race Fault', 'Outer Race Fault']
    else:
        raise ValueError(f"未定义数据集 '{args.dataset}' 的类别列表！请在 main.py 中添加。")
    task_name = args.task
    dataset_name = args.dataset # 获取数据集名称
    
    logging.info(f"=== Starting Workflow: Dataset='{dataset_name}', Task='{task_name}' ===")
    
    processed_dir = Path('data/processed')
    
    # [关键] 读取带有数据集前缀的文件
    csv_path = processed_dir / f"processed_dataset_{dataset_name}_{task_name}.csv"
    
    # [关键] 读取带有数据集前缀的索引文件
    # 注意：你需要先运行预处理脚本生成这些文件
    train_indices_path = processed_dir / f"{dataset_name}_train_indices.json"
    val_indices_path = processed_dir / f"{dataset_name}_val_indices.json"
    test_indices_path = processed_dir / f"{dataset_name}_test_indices.json"

    if not all([csv_path.exists(), train_indices_path.exists(), val_indices_path.exists(), test_indices_path.exists()]):
        logging.error("必需的数据或索引文件不存在！")
        logging.error("请先运行 'preprocess_jsonl.py'，然后运行 'split_dataset_indices.py'。")
        return

    # 2. 加载数据和索引
    logging.info(f"Loading data from {csv_path}...")
    data_df = pd.read_csv(csv_path)
    
    logging.info("Loading pre-split indices...")
    with open(train_indices_path, "r") as f: train_indices = json.load(f)
    with open(val_indices_path, "r") as f: val_indices = json.load(f)
    with open(test_indices_path, "r") as f: test_indices = json.load(f)
        
    # 3. [关键] 使用预先拆分好的索引，来精确地划分DataFrame
    train_df = data_df.iloc[train_indices].copy()
    val_df = data_df.iloc[val_indices].copy()
    test_df = data_df.iloc[test_indices].copy()

    logging.info(f"Data successfully split using pre-defined indices. Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    label_col_numeric = "label_numeric"
        
    # 4. 调用训练和评估函数 (这部分逻辑完全不变)
    train_and_evaluate(args, train_df, val_df, test_df, class_names=class_names, label_col=label_col_numeric, task_name=task_name, dataset_name=args.dataset)
        
    logging.info("\n=== Workflow Complete! ===")

if __name__ == "__main__":
    log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    root_logger = logging.getLogger(); root_logger.setLevel(logging.INFO)
    if root_logger.hasHandlers(): root_logger.handlers.clear()
    log_file = 'finetune_from_csv.log'; file_handler = logging.FileHandler(log_file, mode='w'); file_handler.setFormatter(log_formatter); 
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_formatter)
    root_logger.addHandler(console_handler)
    
    main()