# 本地复现与验证记录

## 目的

确认交付的模型、依赖和示例能在本机工作；在验证之前，不重新训练，也不把已有报告当成新的独立结果。

## 项目交付内容

- `data/reddb_curated.parquet`：清洗后的 RedDB 标签表。
- `data/features_rdkit.parquet`：RDKit 特征。
- `data/features_xtb.parquet`：可成功得到 xTB 特征的子集。
- `models/model_rdkit_baseline.pkl`：快速 RDKit + Morgan 指纹模型。
- `models/model_hybrid.pkl`：RDKit + xTB 混合模型。
- `reports/`：随压缩包提供的评估表、预测与图片。

## 环境

推荐 Conda：

```bash
conda env create -f environment.yml
conda activate redox
pip install -e .
```

最小运行依赖为 RDKit、scikit-learn、joblib、NumPy、Pandas 和 PyArrow。混合模型还需要把 `xtb` 可执行程序放在 `PATH` 中。

## 验证顺序

1. 运行快速模型：

   ```bash
   python examples/quickstart.py
   ```

   即使没有安装 xTB，快速 RDKit 模型也应该返回对苯醌和批量候选分子的结果；混合模型部分会被跳过。

2. 确认命令行入口：

   ```bash
   predict-redox "O=C1C=CC(=O)C=C1"
   ```

3. 若已安装 xTB，再测试：

   ```bash
   predict-redox --model hybrid "O=C1C=CC(=O)C=C1"
   ```

4. 再决定是否复跑完整流程。完整流程会重新下载 RedDB、重新特征化并训练；xTB 特征化是计算密集步骤，项目说明估计 8 核上约 1 小时处理 350 个分子。

## 当前验证状态

- [x] 压缩包结构已检查。
- [x] 说明、训练模型、数据、报告文件均已恢复到本地项目。
- [x] 已创建隔离的 Python 3.11 Conda-forge 环境（`redox-rfb`），并安装 RDKit、scikit-learn、xTB 与项目本身。
- [x] RDKit 快速模型已本机执行：对苯醌为 `0.1853 V vs SHE`。
- [x] xTB 与混合模型已本机执行：示例对苯醌为 `0.233 V vs SHE`。
- [ ] 完整重训尚未进行；只有在数据审计与评价方案确认后才应执行。

运行项目时可使用：

```bash
MAMBA_ROOT_PREFIX=$HOME/.local/share/micromamba \
  $HOME/.local/bin/micromamba run \
  -p $HOME/.local/share/micromamba/envs/redox-rfb \
  predict-redox "O=C1C=CC(=O)C=C1"
```

## 评价注意事项

1. RedDB 标签是 DFT 派生标签，不是实验真值。
2. 15,673 分子的高分数可能受同骨架近邻分子影响；应补充 scaffold split 才能说明新骨架泛化能力。
3. 129 分子 xTB 子集很小，报告中的交叉验证误差应带不确定性解释。
4. 若下一阶段加入 MIST 等预训练模型，所有模型必须使用相同的标签、相同数据划分和相同最终测试集。
