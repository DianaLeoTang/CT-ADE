# CT-ADE
CT-ADE：基于临床试验结果的不良药物事件预测评估基准

## 引用

```
@article{yazdani2025evaluation,
  title={An Evaluation Benchmark for Adverse Drug Event Prediction from Clinical Trial Results},
  author={Yazdani, Anthony and Bornet, Alban and Khlebnikov, Philipp and Zhang, Boya and Rouhizadeh, Hossein and Amini, Poorya and Teodoro, Douglas},
  journal={Scientific Data},
  volume={12},
  number={1},
  pages={1--12},
  year={2025},
  publisher={Nature Publishing Group}
}
```

## 开发环境

- 操作系统：Ubuntu 22.04.3 LTS
    - 内核：Linux 4.18.0-513.18.1.el8_9.x86_64
    - 架构：x86_64
- Python：
    - 3.10.12

## 环境准备

1. 配置环境并安装 `requirements.txt` 中指定的 Python 库。注意：部分库需要从各自的 Git 仓库安装开发版。
2. 将解压后的 MedDRA 文件放入目录 `./data/MedDRA_25_0_English`，将 DrugBank XML 数据库放入目录 `./data/drugbank`。

请从以下 Git 仓库克隆并安装开发版库：

- [`transformers`](https://github.com/huggingface/transformers)
- [`trl`](https://github.com/huggingface/trl)

## 仓库结构

```plaintext
.
├── a0_download_clinical_trials.py
├── a1_extract_completed_or_terminated_interventional_results_clinical_trials.py
├── a2_extract_and_preprocess_monopharmacy_clinical_trials.py
├── b0_download_pubchem_cids.py
├── b1_download_pubchem_cid_details.py
├── c0_extract_drugbank_dbid_details.py
├── d0_extract_chembl_approved_CHEMBL_details.py
├── data
│   ├── MedDRA_25_0_English
│   │   └── empty.null
│   ├── chembl_approved
│   │   └── empty.null
│   ├── chembl_usan
│   │   └── empty.null
│   ├── clinicaltrials_gov
│   │   └── empty.null
│   ├── drugbank
│   │   └── empty.null
│   └── pubchem
│       └── empty.null
├── e0_extract_chembl_usan_CHEMBL_details.py
├── f0_create_unified_chemical_database.py
├── g0_create_ct_ade_raw.py
├── g1_create_ct_ade_meddra.py
├── g2_create_ct_ade_classification_datasets.py
├── g3_create_ct_ade_friendly_labels.py
├── modeling
│   ├── DLLMs
│   │   ├── config.py
│   │   ├── custom_metrics.py
│   │   ├── model.py
│   │   ├── train.py
│   │   └── utils.py
│   └── GLLMs
│       ├── config-llama3.py
│       ├── config-meditron.py
│       ├── config-openbiollm.py
│       ├── config.py
│       ├── train_S.py
│       ├── train_SG.py
│       └── train_SGE.py
├── requirements.txt
└── src
    └── meddra_graph.py
```

## 下载公开的 CT-ADE-SOC 与 CT-ADE-PT

可从 HuggingFace 下载公开的 CT-ADE-SOC 和 CT-ADE-PT 版本。这些数据集包含来自 ClinicalTrials.gov 的标准化标注：

- [`CT-ADE-SOC`](https://huggingface.co/datasets/anthonyyazdaniml/CT-ADE-SOC)
- [`CT-ADE-PT`](https://huggingface.co/datasets/anthonyyazdaniml/CT-ADE-PT)

也可从 Figshare 获取：

- [`CT-ADE-SOC & CT-ADE-PT`](https://figshare.com/articles/dataset/28142453)

上述数据集与「从检查点复现的典型流程」一节中生成的 SOC、PT 版本一致。

## 从检查点复现的典型流程

若希望复现论文中的数据集（CT-ADE-SOC、CT-ADE-PT），请按以下步骤操作。

### 1. 放置数据
将解压后的 MedDRA 文件放入目录 `./data/MedDRA_25_0_English`，将 DrugBank XML 数据库放入目录 `./data/drugbank`。

### 2. 从 HuggingFace 下载检查点
下载 [`chembl_approved、chembl_usan、clinicaltrials_gov、pubchem`](https://huggingface.co/datasets/anthonyyazdaniml/CTADE_v1_initial_release_checkpoint) 并放入对应目录。

### 3. 提取 DrugBank DBID 详情

从 DrugBank 数据库中提取药物详情。

```bash
python c0_extract_drugbank_dbid_details.py
```

### 4. 创建统一化学数据库

合并 PubChem、DrugBank 和 ChEMBL 信息，生成统一数据库。

```bash
python f0_create_unified_chemical_database.py
```

### 5. 生成原始 CT-ADE 数据集

基于已处理的临床试验数据生成原始 CT-ADE 数据集。

```bash
python g0_create_ct_ade_raw.py
```

### 6. 生成 MedDRA 标注

为 CT-ADE 数据集添加 MedDRA 术语标注。

```bash
python g1_create_ct_ade_meddra.py
```

### 7. 生成分类数据集

生成用于建模的最终分类数据集。

```bash
python g2_create_ct_ade_classification_datasets.py
```

### 8. （可选）生成用户友好标签

可选步骤：生成将 MedDRA 代码替换为可读文本标签的数据集版本。运行：

```bash
python g3_create_ct_ade_friendly_labels.py
```

## 模型训练

### 判别式模型（DLLMs）

进入 `modeling/DLLMs` 目录，按需配置后运行训练脚本。

```bash
cd modeling/DLLMs
```

单 GPU 训练：

```bash
export CUDA_VISIBLE_DEVICES="0"; \
export MIXED_PRECISION="bf16"; \
FIRST_GPU=$(echo $CUDA_VISIBLE_DEVICES | cut -d ',' -f 1); \
BASE_PORT=29500; \
PORT=$(( $BASE_PORT + $FIRST_GPU )); \
accelerate launch \
--mixed_precision=$MIXED_PRECISION \
--num_processes=$(( $(echo $CUDA_VISIBLE_DEVICES | grep -o "," | wc -l) + 1 )) \
--num_machines=1 \
--dynamo_backend=no \
--main_process_port=$PORT \
train.py
```

多 GPU 训练：

```bash
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"; \
export MIXED_PRECISION="bf16"; \
FIRST_GPU=$(echo $CUDA_VISIBLE_DEVICES | cut -d ',' -f 1); \
BASE_PORT=29500; \
PORT=$(( $BASE_PORT + $FIRST_GPU )); \
accelerate launch \
--mixed_precision=$MIXED_PRECISION \
--num_processes=$(( $(echo $CUDA_VISIBLE_DEVICES | grep -o "," | wc -l) + 1 )) \
--num_machines=1 \
--dynamo_backend=no \
--main_process_port=$PORT \
train.py
```

### 生成式模型（GLLMs）

进入 `modeling/GLLMs` 目录，按不同配置运行训练脚本。

```bash
cd modeling/GLLMs
```

目录中提供了 LLama3、OpenBioLLM、Meditron 的示例配置。可将所需配置复制到 `config.py` 并自行修改，然后执行以下命令运行 SGE 配置：

```bash
python train_SGE.py
```
