
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
# 1. 自定义线性SVM模型（带损失记录 & 最佳模型保存）
# ================================
class LinearSVM:
    def __init__(self, learning_rate=0.01, lambda_param=0.01, n_iters=1000):
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.weights = None
        self.bias = None
        self.loss_history = []
        self.metrics_history = []  # 存储每个 epoch 的评估指标
        self.best_weights = None
        self.best_bias = None
        self.best_val_acc = 0

    def fit(self, X_train, y_train, X_val, y_val):
        n_samples, n_features = X_train.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        # 将标签转换为 -1 和 1
        y_train_ = np.where(y_train <= 0, -1, 1)
        y_val_ = np.where(y_val <= 0, -1, 1)

        for epoch in range(self.n_iters):
            total_loss = 0
            for idx, x_i in enumerate(X_train):
                margin = y_train_[idx] * (np.dot(x_i, self.weights) - self.bias)
                if margin >= 1:
                    self.weights -= self.lr * (2 * self.lambda_param * self.weights)
                else:
                    self.weights -= self.lr * (2 * self.lambda_param * self.weights - np.dot(x_i, y_train_[idx]))
                    self.bias -= self.lr * y_train_[idx]
                loss = max(0, 1 - margin) + self.lambda_param * np.dot(self.weights, self.weights)
                total_loss += loss

            avg_loss = total_loss / n_samples
            self.loss_history.append(avg_loss)

            # 验证集预测
            y_val_pred = self.predict(X_val)
            y_val_pred_binary = np.where(y_val_pred == -1, 0, 1)

            val_acc = accuracy_score(y_val, y_val_pred_binary)
            val_precision = precision_score(y_val, y_val_pred_binary, zero_division=0)
            val_recall = recall_score(y_val, y_val_pred_binary, zero_division=0)
            val_f1 = f1_score(y_val, y_val_pred_binary, zero_division=0)

            # 记录当前 epoch 的所有指标
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

            # 打印部分 epoch 的详细信息
            if (epoch + 1) % 50 == 0 or epoch == 0 or epoch == self.n_iters - 1:
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
# 2. 加载多个TSV文件的数据
# ================================
# file_paths = ['eval00.tsv', 'eval02.tsv', 'train02.tsv', 'train00.tsv']
file_paths = ['SST_2dev.tsv', 'SST_2train.tsv']
dfs = []

for file in file_paths:
    if os.path.exists(file):
        df = pd.read_csv(file, sep='\t', header=0)
        dfs.append(df)
    else:
        print(f"警告：文件 {file} 不存在！")

df = pd.concat(dfs, ignore_index=True)

# 合并两个句子作为输入
# df['combined'] = df['sentence1'] + " " + df['sentence2']
df['combined'] = df['sentence']
# 标签和文本
X = df['combined'].values
y = df['label'].values.astype(int)

# 转换为 TF-IDF 特征
vectorizer = TfidfVectorizer(max_features=5000)
X_vec = vectorizer.fit_transform(X).toarray()

# 划分训练集、验证集、测试集
X_train, X_temp, y_train, y_temp = train_test_split(
    X_vec, y, test_size=0.3, random_state=42, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

# ================================
# 3. 训练自定义SVM模型
# ================================
print("开始训练自定义SVM模型...\n")
svm = LinearSVM(learning_rate=0.001, lambda_param=0.01, n_iters=1000)
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
model_save_path = "best_svm_model.pkl"
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

report_df = pd.read_csv("评估.csv", index_col=0)
report_df.loc["extra"] = pd.Series(extra_metrics)
report_df.to_csv("评估.csv")
print(f"✅ 已将 AUC 和 Average Precision 添加到 '评估.csv'")

# ================================
# 11. 保存每个 epoch 的训练和验证指标
# ================================
metrics_df = pd.DataFrame(svm.metrics_history)
metrics_df.to_csv("training_metrics.csv", index=False)
print("✅ 每个 Epoch 的评估指标已保存至 training_metrics.csv")


