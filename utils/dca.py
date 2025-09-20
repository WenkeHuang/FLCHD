from matplotlib import pyplot as plt, ticker
import numpy as np
from sklearn.metrics import confusion_matrix
import torch


def calculate_net_benefit(y_true, y_pred_proba, thresholds):
    """
    计算模型、Treat All 和 Treat None 策略的净收益
    :param y_true: 真实标签 (0 或 1)
    :param y_pred_proba: 模型预测概率 (0 ~ 1)
    :param thresholds: 阈值列表 (0 ~ 1)
    :return: 各策略的净收益
    """
    n_samples = len(y_true)
    prevalence = float(sum(y_true)) / n_samples  # 患病率
    
    # 初始化净收益
    net_benefit_model = []
    net_benefit_all = []
    net_benefit_none = []
    
    # 转换为PyTorch tensor
    y_true_tensor = torch.tensor(y_true, dtype=torch.int)
    y_pred_proba_tensor = torch.tensor(y_pred_proba, dtype=torch.float32)
 
    for threshold in thresholds:
        # 根据阈值计算二分类预测结果
        y_pred = (y_pred_proba_tensor >= threshold).int()
        
        # 计算混淆矩阵
        cm = confusion_matrix(y_true, (y_pred_proba >= threshold).astype(int))
        
        # 提取TP, FP, TN, FN
        tn, fp, fn, tp = cm.ravel()
        
        # 计算模型的净收益
        net_benefit = (tp / n_samples) - (fp / n_samples) * (threshold / (1 - threshold))
        net_benefit_model.append(net_benefit)
        
        # Treat All 策略的净收益
        treat_all_benefit = prevalence - (1 - prevalence) * (threshold / (1 - threshold))
        net_benefit_all.append(treat_all_benefit)
        
        # Treat None 策略的净收益始终为 0
        net_benefit_none.append(0)
    
    return net_benefit_model, net_benefit_all, net_benefit_none
 
 
def plot_decision_curve(y_true, y_pred_proba, thresholds=None, save_path="DCA_single.png"):
    """
    Plot decision curve
    :param y_true: True labels (0 or 1)
    :param y_pred_proba: Model predicted probabilities (0 ~ 1)
    :param thresholds: List of thresholds (0 ~ 1)
    :param save_path: Path to save the image
    """
    if thresholds is None:
        thresholds = np.linspace(0.01, 0.99, 100)
    
    # 计算净收益
    net_benefit_model, net_benefit_all, net_benefit_none = calculate_net_benefit(
        y_true, y_pred_proba, thresholds
    )
    # 计算合适的y轴上限
    max_benefit = max(
        max(net_benefit_model),
        max(net_benefit_all),
        max(net_benefit_none)
    )
    # 增加20%的空间使图形更美观
    y_upper_limit = max_benefit * 1.2
    
    # 绘图
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, net_benefit_model, label="Model", lw=2)
    plt.plot(thresholds, net_benefit_all, label="Treat All", linestyle="--", lw=2)
    plt.plot(thresholds, net_benefit_none, label="Treat None", linestyle=":", lw=2)
    plt.xlabel("Threshold Probability")
    plt.ylabel("Net Benefit")
    plt.title("Decision Curve Analysis")
    plt.legend(loc="best")
    plt.grid(alpha=0.3)
    plt.ylim([-0.02, y_upper_limit])
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def custom_formatter(x, pos):
    """用于格式化图表中的数值"""
    return f"{x:.2f}"

def plot_multiple_decision_curves(y_true_dict, y_pred_proba_dict, thresholds=None, subplot_rows=1, save_path="DCA.png"):
    """
    Plot decision curves for multiple cohorts
    :param y_true_dict: Dictionary with cohort names as keys and true label arrays as values
    :param y_pred_proba_dict: Dictionary with cohort names as keys and prediction probability arrays as values
    :param thresholds: List of thresholds
    :param subplot_rows: Number of subplot rows
    :param save_path: Path to save the image
    """
    if thresholds is None:
        thresholds = np.linspace(0.01, 0.99, 100)
    
    n_datasets = len(y_true_dict.keys())  # 数据集数量
    subplot_cols = int(np.ceil(n_datasets / subplot_rows))  # 计算列数
    
    fig, axes = plt.subplots(subplot_rows, subplot_cols, figsize=(5 * subplot_cols, 4 * subplot_rows))
    if n_datasets == 1:
        axes = np.array([axes])
    axes = axes.flatten()  # 展平数组，便于索引
 
    for i, (cohort_name, y_true) in enumerate(y_true_dict.items()):
        y_pred_proba = y_pred_proba_dict[cohort_name]
 
        # 计算净收益
        net_benefit_model, net_benefit_all, net_benefit_none = calculate_net_benefit(
            y_true, y_pred_proba, thresholds
        )
        
        # 计算自适应 Y 轴上限
        max_benefit = max(
            max(net_benefit_model),
            max(net_benefit_all),
            max(net_benefit_none)
        )
        y_upper_limit = max_benefit * 1.2  # 增加20%的空间
        y_lower_limit = min(-0.02, -0.15 * max_benefit)
        
        # 绘制子图
        ax = axes[i]  # 获取当前子图的轴
        ax.plot(thresholds, net_benefit_model, label="Model", lw=2)
        ax.plot(thresholds, net_benefit_all, label="Treat All", linestyle="--", lw=2)
        ax.plot(thresholds, net_benefit_none, label="Treat None", linestyle=":", lw=2)
        
        ax.set_xlabel("Threshold Probability")
        if i % subplot_cols == 0:
            ax.set_ylabel("Net Benefit")
        ax.set_title(cohort_name)
        ax.legend(loc="best")
        ax.grid(alpha=0.3)
        ax.set_ylim([y_lower_limit, y_upper_limit])  # 自适应 Y 轴
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(custom_formatter))
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(custom_formatter))
        
    # 如果有空余子图，隐藏它们
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


# 单数据集示例
y_true = [0, 1, 1, 0, 1, 0, 1, 1, 0, 0]  # 真实标签
y_pred_proba = [0.1, 0.9, 0.8, 0.3, 0.7, 0.2, 0.6, 0.9, 0.4, 0.3]  # 预测概率

plot_decision_curve(y_true, y_pred_proba)

y_true_dict = {
    "Validation Set": [0, 1, 1, 0, 1, 0, 1, 1, 0, 0],
    "External Validation Set 1": [1, 0, 1, 1, 0, 0, 1, 0, 1, 0]
}
y_pred_proba_dict = {
    "Validation Set": [0.1, 0.9, 0.8, 0.3, 0.7, 0.2, 0.6, 0.9, 0.4, 0.3],
    "External Validation Set 1": [0.8, 0.2, 0.7, 0.9, 0.3, 0.1, 0.8, 0.4, 0.7, 0.2]
}

plot_multiple_decision_curves(y_true_dict, y_pred_proba_dict)