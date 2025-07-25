import matplotlib.pyplot as plt
import numpy as np

# 数据
models = ['RoBERTa+FC', 'RoBERTa+BiLSTM+FC', 'RoBERTa+BiLSTM+Attention +FFN+FC']
f1_scores = [81.8, 85.3, 86.1]
accuracies = [82.1, 84.7, 87.2]

# 更细的柱子宽度和更小的图尺寸
bar_width = 0.2
fig, ax = plt.subplots(figsize=(8, 8)) # 尺寸更小

index = np.arange(len(models))

bars1 = ax.bar(index, f1_scores, bar_width, label='F1 Score')
bars2 = ax.bar(index + bar_width, accuracies, bar_width, label='Accuracy')

ax.set_xlabel('Model Configurations')
ax.set_ylabel('Scores (%)')
ax.set_title('Comparison of F1 Scores and Accuracy by Model Configuration')
ax.set_xticks(index + bar_width / 2)
ax.set_xticklabels(models, rotation=5, ha='right', fontsize=12)
ax.legend()
# 调整图例位置到左上角外侧
ax.legend(bbox_to_anchor=(0.01, 0.94, 0.3, 0.1), loc="lower left", mode="expand", borderaxespad=0, ncol=2)

# 添加数值标签
def add_labels(bars):
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=12) # 字体大小调整

# ... [后面的代码保持不变]
add_labels(bars1)
add_labels(bars2)

plt.tight_layout()
# 调整字体大小和图例样式等可以进一步缩小图片的整体感觉
# 如果要保存为文件，取消下一行注释
plt.savefig("small_chart.png", dpi=150)
# 显示图像
plt.show()

