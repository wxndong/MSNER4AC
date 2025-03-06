import numpy as np 
import matplotlib.pyplot as plt
import matplotlib

# 设置字体（ACL推荐无衬线字体，支持中文）
matplotlib.rcParams['font.family'] = 'Helvetica'  # 替换为Helvetica，更学术化
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 数据
categories = ['Precision', 'Recall']
y_label = "Test Total F1 / Percentile (%)"
datasets = [
    ("Train Data", [654.81, 14.28], "#4C6AFD", "//"),
    ("Test Data(Bind Test)", [38.24, 710.10], "#CACACA", None),
]
    # ("PLM + CRF", [82.6, 83.4, 82.1], "#A9C0FF", None),
    # ("PLM + Softmax", [83.6, 80.6, 80.0], "#E6E6E6", None),

# 画布设置（高清图）
fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)  # 调整为更紧凑的尺寸
x = np.arange(len(categories))  # x 轴位置
width = 0.18  # 略宽的柱状图，提升视觉效果

# 绘制柱状图
for i, (label, values, color, hatch) in enumerate(datasets):
    if label == "Train Data":
        bars = ax.bar(
            x + (i - 1.5) * width, values, width, label=label, 
            color=color, edgecolor="white", hatch=hatch, linewidth=1.2, zorder=3
        )
    else:
        bars = ax.bar(
            x + (i - 1.5) * width, values, width, label=label, 
            color=color, hatch=hatch, linewidth=0, zorder=3
        )
    # 添加数值标签，调整位置
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8, f'{value:.1f}', 
            ha='center', va='bottom', fontsize=9, fontweight='bold', color='#333333'
        )

# 轴设置
ax.set_ylabel(y_label, fontsize=11, color='#333333', labelpad=2)  # 更紧凑的labelpad
ax.set_xticks(x)
ax.set_xticklabels(categories, fontsize=10, color='#333333')
ax.set_ylim(0, 100)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.12), ncol=4, fontsize=9, frameon=False)
ax.grid(axis='y', linestyle='--', alpha=0.6, zorder=0)  # 网格线稍明显
ax.set_axisbelow(True)

# 设置灰色边框
for spine in ax.spines.values():
    spine.set_color('#B0B0B0')  # 柔和灰色边框
    spine.set_linewidth(0.8)    # 细边框

# 调整标签与轴的距离
ax.tick_params(axis='x', pad=2, colors='#333333')
ax.tick_params(axis='y', pad=2, colors='#333333')

# Tight layout
plt.tight_layout(pad=0.3)  # 更紧凑的布局

# 保存高清图片
plt.savefig("../results/benchmark_acl.png", dpi=600, bbox_inches='tight')

# 显示图表
plt.show()