import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, accuracy_score

# 文件路径
train_path = r"E:\作业报告ppt\大三下\大数据分析和内存计算\实验\bert-pretraining-main1\1data\SST_2train.tsv"
dev_path = r"E:\作业报告ppt\大三下\大数据分析和内存计算\实验\bert-pretraining-main1\1data\SST_2dev.tsv"

# 加载训练集和验证集
print("Loading training and validation data...")
train_data = pd.read_csv(train_path, sep='\t', header=0)
dev_data = pd.read_csv(dev_path, sep='\t', header=0)

# 检查标签是否为整数
train_data['label'] = train_data['label'].astype(int)
dev_data['label'] = dev_data['label'].astype(int)

# 提取特征和标签
X_train = train_data['sentence'].values
y_train = train_data['label'].values

X_val = dev_data['sentence'].values
y_val = dev_data['label'].values

# 特征提取：TF-IDF
print("Extracting TF-IDF features...")
vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))

# 在训练集上 fit_transform，在验证集上 transform
X_train_vec = vectorizer.fit_transform(X_train)
X_val_vec = vectorizer.transform(X_val)

# 构建随机森林模型
print("Training Random Forest model...")
rf = RandomForestClassifier(n_estimators=100, random_state=42, verbose=1)
rf.fit(X_train_vec, y_train)

import joblib

# 模型评估
print("Evaluating model on validation set...")
preds = rf.predict(X_val_vec)

# 打印评估结果
print("Accuracy:", accuracy_score(y_val, preds))
print(classification_report(y_val, preds, digits=4))

# # 保存模型和向量化器
# model_save_path = r"E:\作业报告ppt\大三下\大数据分析和内存计算\实验\bert-pretraining-main1\1data\sst2_model.pkl"
# vectorizer_save_path = r"E:\作业报告ppt\大三下\大数据分析和内存计算\实验\bert-pretraining-main1\1data\tfidf_vectorizer.pkl"
#
# print("Saving model and vectorizer...")
# joblib.dump(rf, model_save_path)
# joblib.dump(vectorizer, vectorizer_save_path)
#
# print("Model saved to:", model_save_path)
# print("Vectorizer saved to:", vectorizer_save_path)