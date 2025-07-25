import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report, accuracy_score,
                             precision_score, recall_score, f1_score,
                             roc_curve, auc, precision_recall_curve,
                             average_precision_score)

# ================================
# 1. 自定义线性SVM模型（改进 + 调参）
# ================================
class LinearSVM:
    def __init__(self, learning_rate=0.0001, lambda_param=0.01, n_iters=1000, batch_size=64, patience=30, momentum=0.9):
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.batch_size = batch_size
        self.patience = patience
        self.momentum = momentum

        self.weights = None
        self.bias = None
        self.loss_history = []
        self.metrics_history = []
        self.best_weights = None
        self.best_bias = None
        self.best_val_acc = 0
        self.wait = 0

        # 动量项
        self.velocity_w = None
        self.velocity_b = 0

    def fit(self, X_train, y_train, X_val, y_val):
        n_samples, n_features = X_train.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        self.velocity_w = np.zeros_like(self.weights)
        self.velocity_b = 0

        y_train_ = np.where(y_train <= 0, -1, 1)
        y_val_ = np.where(y_val <= 0, -1, 1)

        for epoch in range(self.n_iters):
            indices = np.random.permutation(n_samples)

            total_loss = 0
            for i in range(0, n_samples, self.batch_size):
                batch_indices = indices[i:i + self.batch_size]
                X_batch = X_train[batch_indices]
                y_batch = y_train_[batch_indices]

                margins = y_batch * (X_batch @ self.weights - self.bias)
                hinge_mask = margins < 1

                # 梯度计算
                grad_w = 2 * self.lambda_param * self.weights
                if np.any(hinge_mask):
                    grad_w -= (X_batch[hinge_mask].T @ y_batch[hinge_mask]).sum()
                    grad_b = -y_batch[hinge_mask].sum()
                else:
                    grad_b = 0

                # 梯度归一化（防爆炸）
                grad_w /= len(batch_indices)
                grad_b /= len(batch_indices)

                # 动量更新
                self.velocity_w = self.momentum * self.velocity_w - self.lr * grad_w
                self.velocity_b = self.momentum * self.velocity_b - self.lr * grad_b

                self.weights += self.velocity_w
                self.bias += self.velocity_b

                loss = np.maximum(0, 1 - margins).mean() + self.lambda_param * np.dot(self.weights, self.weights)
                total_loss += loss * len(batch_indices)

            avg_loss = total_loss / n_samples
            self.loss_history.append(avg_loss)

            # 验证集预测
            y_val_pred = self.predict(X_val)
            y_val_pred_binary = np.where(y_val_pred == -1, 0, 1)

            val_acc = accuracy_score(y_val, y_val_pred_binary)
            val_precision = precision_score(y_val, y_val_pred_binary, zero_division=0)
            val_recall = recall_score(y_val, y_val_pred_binary, zero_division=0)
            val_f1 = f1_score(y_val, y_val_pred_binary, zero_division=0)

            self.metrics_history.append({
                'epoch': epoch + 1,
                'loss': avg_loss,
                'val_accuracy': val_acc,
                'val_precision': val_precision,
                'val_recall': val_recall,
                'val_f1': val_f1
            })

            # 更新最佳模型
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_weights = self.weights.copy()
                self.best_bias = self.bias
                self.wait = 0
            else:
                self.wait += 1

            # 早停
            if self.wait >= self.patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

            # 打印部分 epoch 的详细信息
            if (epoch + 1) % 20 == 0 or epoch == 0 or epoch == self.n_iters - 1 or self.wait >= self.patience:
                print(f"Epoch [{epoch+1}/{self.n_iters}], Loss: {avg_loss:.6f}, "
                      f"Val Acc: {val_acc:.4f}, Val P: {val_precision:.4f}, "
                      f"Val R: {val_recall:.4f}, Val F1: {val_f1:.4f}")

    def predict(self, X):
        approx = np.dot(X, self.weights) - self.bias
        return np.sign(approx)

    def use_best_model(self):
        """使用最佳模型参数进行预测"""
        self.weights = self.best_weights
        self.bias = self.best_bias


# ================================
# 2. 加载数据
# ================================
file_paths = ['SST_2dev.tsv', 'SST_2train.tsv']
dfs = []

for file in file_paths:
    if os.path.exists(file):
        df = pd.read_csv(file, sep='\t', header=0)
        dfs.append(df)
    else:
        print(f"警告：文件 {file} 不存在！")

df = pd.concat(dfs, ignore_index=True)
X = df['sentence'].values
y = df['label'].values.astype(int)

# TF-IDF 特征提取
vectorizer = TfidfVectorizer(max_features=10000)
X_vec = vectorizer.fit_transform(X).toarray()

# 划分训练集、验证集、测试集
X_train, X_temp, y_train, y_temp = train_test_split(
    X_vec, y, test_size=0.3, random_state=42, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

# ================================
# 3. 训练模型
# ================================
print("开始训练改进版SVM模型...\n")
svm = LinearSVM(learning_rate=0.0001, lambda_param=0.01, n_iters=1000, batch_size=64, patience=30, momentum=0.9)
svm.fit(X_train, y_train, X_val, y_val)

# 使用最佳模型参数
svm.use_best_model()

# ================================
# 4. 模型评估
# ================================
y_pred = svm.predict(X_test)
y_pred = np.where(y_pred == -1, 0, 1)

acc = accuracy_score(y_test, y_pred)
print(f"\n测试集准确率 Accuracy: {acc:.4f}")
print("\n分类报告：")
print(classification_report(y_test, y_pred))

# ================================
# 5. 损失曲线可视化 + 保存loss历史
# ================================
plt.figure(figsize=(10, 6))
plt.plot(svm.loss_history, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('SVM Training Loss Curve')
plt.grid(True)
plt.legend()
plt.savefig("loss_curve.png", dpi=300, bbox_inches='tight')
plt.show()

# 保存损失值到CSV
loss_df = pd.DataFrame({"Epoch": range(1, len(svm.loss_history)+1), "Loss": svm.loss_history})
loss_df.to_csv("loss.csv", index=False)
print("✅ 损失函数已保存至 loss.csv")

# ================================
# 6. 保存最佳模型参数
# ================================
model_save_path = "../1data/best_svm_model.pkl"
with open(model_save_path, "wb") as f:
    pickle.dump({
        'weights': svm.best_weights,
        'bias': svm.best_bias,
        'vectorizer': vectorizer
    }, f)
print(f"✅ 最佳模型已保存至 {model_save_path}")

# ================================
# 7. 生成详细评估报告并保存为 CSV 文件
# ================================
report_dict = classification_report(y_test, y_pred, output_dict=True)
report_df = pd.DataFrame(report_dict).transpose()
report_df.to_csv("评估.csv", index=True)
print(f"✅ 分类评估报告已保存至 '评估.csv'")

# ================================
# 8. ROC 曲线与 AUC 值（适用于二分类）
# ================================
scores = np.dot(X_test, svm.best_weights) - svm.best_bias
fpr, tpr, _ = roc_curve(y_test, scores)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")
plt.savefig("roc_curve.png", dpi=300, bbox_inches='tight')
plt.show()

# ================================
# 9. PR 曲线（Precision-Recall Curve）
# ================================
precision, recall, _ = precision_recall_curve(y_test, scores)
average_precision = average_precision_score(y_test, scores)

plt.figure()
plt.step(recall, precision, color='b', alpha=0.2, where='post')
plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title(f'Precision-Recall curve: AP={average_precision:.2f}')
plt.savefig("pr_curve.png", dpi=300, bbox_inches='tight')
plt.show()

# ================================
# 10. 将 AUC 和 AP 写入评估报告
# ================================
extra_metrics = {
    "AUC": roc_auc,
    "Average Precision": average_precision
}

report_df = pd.read_csv("../1data/评估.csv", index_col=0)
report_df.loc["extra"] = pd.Series(extra_metrics)
report_df.to_csv("评估.csv")
print(f"✅ 已将 AUC 和 Average Precision 添加到 '评估.csv'")

# ================================
# 11. 保存每个 epoch 的训练和验证指标
# ================================
metrics_df = pd.DataFrame(svm.metrics_history)
metrics_df.to_csv("training_metrics.csv", index=False)
print("✅ 每个 Epoch 的评估指标已保存至 training_metrics.csv")