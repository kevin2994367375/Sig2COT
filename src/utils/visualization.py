import os
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report

# 【【【 核心修改：在顶部切换matplotlib后端 】】】
# 这一步也可以放在main.py的最顶端，放在这里确保此模块被导入时就切换
import matplotlib
matplotlib.use('Agg')

# 设置中文字体，确保图表能正确显示中文
# 注意: 'WenQuanYi Micro Hei' 需要你的系统已安装
plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'SimHei'] 
# 解决负号显示问题
plt.rcParams['axes.unicode_minus'] = False 

# 【【【 核心修改：新增辅助函数 】】】
def save_and_log_plot(output_dir, filename, title):
    """
    Helper function to save the current plot to a file, log the path, and close the figure.
    """
    if not output_dir:
        logging.warning(f"Output directory not provided for plot '{title}'. Skipping save.")
        plt.close() # 依然关闭以防万一
        return
        
    os.makedirs(output_dir, exist_ok=True)
    
    # 清理文件名，使其适合用作文件路径
    safe_filename = "".join([c for c in filename if c.isalpha() or c.isdigit() or c in (' ', '.', '_')]).rstrip()
    safe_filename = safe_filename.replace(' ', '_').lower() + ".png"
    
    full_path = os.path.join(output_dir, safe_filename)
    
    try:
        plt.savefig(full_path, bbox_inches='tight')
        logging.info(f"Plot '{title}' saved to: {full_path}")
    except Exception as e:
        logging.error(f"Failed to save plot '{title}' to {full_path}. Error: {e}")
    finally:
        # 无论成功与否，都关闭当前的绘图窗口以释放内存
        plt.close()


def plot_training_history(history, title_prefix="", output_dir=None):
    """
    Plots training and validation loss/accuracy curves and saves the plot.
    """
    if not history or not history.get('train_loss'):
        logging.warning("History object is empty or invalid. Skipping training history plot.")
        return
        
    plt.figure(figsize=(12, 5))
    full_title_prefix = f"{title_prefix} " if title_prefix else ""

    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Training Loss')
    if 'val_loss' in history and history['val_loss']:
        plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend(); plt.title(f'{full_title_prefix}Loss Curve')
    
    if 'val_acc' in history and history['val_acc']:
        plt.subplot(1, 2, 2)
        plt.plot(history['val_acc'], label='Validation Accuracy')
        plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.legend(); plt.title(f'{full_title_prefix}Accuracy Curve')
    
    plt.tight_layout()
    # 【【【 核心修改：替换 plt.show() 】】】
    save_and_log_plot(output_dir, f"{title_prefix}_training_history", f"{title_prefix} Training History")


def plot_confusion_matrix(y_true, y_pred, class_names=None, title='Confusion Matrix', output_dir=None):
    """
    Plots and saves the confusion matrix.
    """
    cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names))) # Ensure order
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names if class_names else 'auto',
                yticklabels=class_names if class_names else 'auto')
    plt.xlabel('Predicted'); plt.ylabel('True'); plt.title(title)
    plt.tight_layout()
    # 【【【 核心修改：替换 plt.show() 】】】
    save_and_log_plot(output_dir, title, title)


def plot_classification_report(y_true, y_pred, class_names=None, title='Classification Report', output_dir=None):
    """
    Logs the classification report and saves its visualization.
    """
    # labels参数确保了报告的顺序与class_names一致
    labels_for_report = list(range(len(class_names)))
    
    report_text = classification_report(y_true, y_pred, target_names=class_names, zero_division=0, labels=labels_for_report)
    logging.info(f"\n--- {title} ---\n{report_text}")
    
    report_dict = classification_report(y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0, labels=labels_for_report)
    df_report = pd.DataFrame(report_dict).transpose().round(3)
    
    # 【【【 核心修正：使用更健壮的方式删除行 】】】
    # 1. 定义要删除的总结性指标
    summary_metrics = ['accuracy', 'macro avg', 'weighted avg']
    # 2. 找到这些指标中，实际存在于df_report索引里的那些
    metrics_to_drop = [metric for metric in summary_metrics if metric in df_report.index]
    
    # 3. 只删除那些实际存在的行
    metrics_df = df_report.drop(index=metrics_to_drop)
    
    if not metrics_df.empty:
        plt.figure(figsize=(12, 6))
        # Plot only the precision, recall, f1-score columns for individual classes
        metrics_df[['precision', 'recall', 'f1-score']].plot(kind='bar', figsize=(12, 6))
        plt.title(title)
        plt.xlabel('Class')
        plt.ylabel('Score')
        plt.ylim(0, 1.05)
        plt.legend(loc='lower right')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        save_and_log_plot(output_dir, title, title)

def plot_label_distribution(labels, title='Label Distribution', class_names=None, output_dir=None):
    """
    Plots and saves the distribution of labels.
    """
    if len(labels) == 0:
        logging.warning("Cannot plot label distribution: labels are empty.")
        return
        
    label_counts = pd.Series(labels).value_counts().sort_index()
    if class_names:
        label_counts.index = label_counts.index.map(lambda i: class_names[i])

    plt.figure(figsize=(10, 6))
    sns.barplot(x=label_counts.index, y=label_counts.values)
    plt.title(title); plt.xlabel('Label'); plt.ylabel('Count'); plt.xticks(rotation=45)
    plt.tight_layout()
    # 【【【 核心修改：替换 plt.show() 】】】
    save_and_log_plot(output_dir, title, title)


def plot_text_length_distribution(texts, title='Text Length Distribution', bins=50, output_dir=None):
    """
    Plots and saves the distribution of text lengths.
    """
    try:
        text_lengths = [len(str(text).split()) for text in texts]
    except Exception:
        logging.warning("Could not parse text lengths. Skipping plot.")
        return

    if not text_lengths:
        logging.warning("No text lengths to plot."); return

    plt.figure(figsize=(12, 6))
    plt.hist(text_lengths, bins=bins, alpha=0.7)
    mean_len, median_len = np.mean(text_lengths), np.median(text_lengths)
    plt.axvline(x=mean_len, color='r', linestyle='--', label=f'Mean: {mean_len:.1f}')
    plt.axvline(x=median_len, color='g', linestyle='--', label=f'Median: {median_len:.1f}')
    plt.title(title); plt.xlabel('Number of Words'); plt.ylabel('Frequency')
    plt.legend(); plt.tight_layout()
    # 【【【 核心修改：替换 plt.show() 】】】
    save_and_log_plot(output_dir, title, title)
    
    logging.info(f"{title} -> Min: {min(text_lengths)}, Max: {max(text_lengths)}, Mean: {mean_len:.2f}, Median: {median_len:.2f}")