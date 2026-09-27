# Causal LLM

**Large Language Models for Causal Adjustment Set Selection**

基于大语言模型（Large Language Models, LLMs）的因果调整集（Adjustment Set）选择研究项目。

本项目旨在研究大语言模型在因果推断场景中的应用，重点关注因果效应估计中的**调整集选择问题**。与传统以预测性能为主要目标的变量选择方法不同，本项目关注变量之间的因果结构，并尝试利用后门准则（Backdoor Criterion）等因果推断原则指导调整集的生成、验证与修正。

---

## 1. 项目背景

在因果推断中，一个核心问题是估计某个处理变量（Treatment）对结果变量（Outcome）的因果效应。

为了获得可靠的因果效应估计，通常需要选择合适的变量构成调整集（Adjustment Set），从而阻断由混杂因素导致的非因果路径。

传统机器学习中的特征选择通常以预测任务为目标，例如：

* 提高分类准确率；
* 降低预测误差；
* 提高模型拟合能力；
* 选择与目标变量具有较强预测关系的变量。

然而，在因果推断中，**“对预测有用”并不等价于“应该进行控制”**。

一个变量是否应该进入调整集，需要结合变量之间的因果关系进行判断。例如：

* 混杂变量通常需要考虑调整；
* 中介变量通常不应简单地作为混杂因素进行调整；
* 碰撞变量（Collider）通常不应被纳入调整集；
* 调整集需要满足相应的因果识别条件。

因此，本项目将调整集选择问题从传统的预测型变量选择问题中区分出来，并探索：

> **能否利用大语言模型理解因果结构，并根据因果推断原则自动选择合理的调整集？**

---

# 2. 研究问题

本项目当前关注的核心问题是：

> **给定一个因果图，在估计 Treatment 对 Outcome 的因果效应时，应该控制哪些变量？**

可以将问题抽象为：

```text
输入：
    因果图 / 因果结构

        ↓

LLM：
    理解变量之间的因果关系

        ↓

输出：
    Treatment
    Outcome
    Adjustment Set
```

与普通的特征选择问题相比，本项目的选择标准不是简单追求预测准确率，而是关注：

```text
因果结构
    +
因果推断准则
    ↓
Adjustment Set
```

---

# 3. 项目目标

本项目当前阶段主要目标包括：

1. 探索 LLM 对因果图结构的理解能力；
2. 让 LLM 识别 Treatment 和 Outcome；
3. 识别因果图中的不同变量角色；
4. 让 LLM 生成候选 Adjustment Set；
5. 使用结构化输出保证模型输出格式稳定；
6. 建立可重复运行的实验流程；
7. 比较不同 LLM 推理框架的实验表现；
8. 为后续 Adjustment Set Selection Agent 的构建提供基础。

长期目标是进一步构建一个能够：

```text
提出调整集
    ↓
检查调整集
    ↓
发现错误
    ↓
修正调整集
    ↓
再次检查
    ↓
输出最终调整集
```

的因果调整集选择 Agent。

---

# 4. 核心研究思想

传统变量选择通常可以抽象为：

```text
数据
 ↓
预测模型
 ↓
预测性能
 ↓
选择变量
```

而本项目关注的是：

```text
因果图
 ↓
理解因果结构
 ↓
识别变量角色
 ↓
应用因果准则
 ↓
选择 Adjustment Set
```

因此，本项目希望探索一种不同于传统预测型特征选择的变量选择思想：

> **调整集选择的标准从“预测得准”转向“满足因果识别条件”。**

---

# 5. 因果图示例

当前实验使用结构化的因果图来测试 LLM 的因果推理能力。

例如：

```text
X → T
X → Y
T → M
M → Y
T → C
Y → C
```

其中：

* `X`：Treatment 和 Outcome 的共同原因；
* `T`：Treatment；
* `Y`：Outcome；
* `M`：Mediator；
* `C`：Collider。

可以进一步表示为：

```text
        X
       / \
      ↓   ↓
      T   Y
      │   ↑
      ↓   │
      M ──┘

      T → C ← Y
```

该结构包含不同类型的因果关系：

```text
X → T
X → Y
```

表示 `X` 同时影响 Treatment 和 Outcome。

因此存在：

```text
T ← X → Y
```

这样的非因果路径。

同时：

```text
T → M → Y
```

表示 `M` 位于 Treatment 到 Outcome 的因果路径上。

而：

```text
T → C ← Y
```

表示 `C` 是碰撞变量。

这些不同类型的变量需要在调整集选择过程中进行区分。

---

# 6. Adjustment Set

Adjustment Set 是本项目的核心概念之一。

在因果效应估计中，我们希望选择一个变量集合，使得在控制这些变量之后，可以更合理地识别 Treatment 对 Outcome 的因果效应。

例如：

```text
X → T
X → Y
```

存在：

```text
T ← X → Y
```

这一后门路径。

在满足相应因果条件的情况下，可以考虑通过调整 `X` 来阻断该非因果路径。

另一方面，对于：

```text
T → M → Y
```

`M` 是 Treatment 到 Outcome 的中介变量。

对于：

```text
T → C ← Y
```

`C` 是碰撞变量。

因此，调整集选择不能简单地根据：

* 相关性；
* 特征重要性；
* 预测准确率；
* 与 Outcome 的关联程度；

直接决定。

而需要结合完整的因果结构进行判断。

---

# 7. LLM 在项目中的作用

本项目尝试让 LLM 承担因果结构理解与调整集生成任务。

整体流程可以表示为：

```text
Causal Graph
     │
     ↓
Prompt
     │
     ↓
LLM
     │
     ├── Identify Treatment
     │
     ├── Identify Outcome
     │
     ├── Identify Variable Roles
     │
     └── Generate Adjustment Set
     │
     ↓
Structured Output
     │
     ↓
JSON Validation
     │
     ↓
Result Extraction
     │
     ↓
Experiment Results
```

当前阶段主要关注 LLM 是否能够正确理解给定因果结构，并产生符合预期格式的结构化结果。

---

# 8. Structured Output

为了避免 LLM 输出无法稳定解析的问题，本项目采用 JSON Schema 对模型输出进行结构化约束。

Schema 文件位于：

```text
src/schemas/adjustment_set_schema.json
```

模型需要输出预定义的数据结构，例如：

```json
{
    "treatment": "T",
    "outcome": "Y",
    "adjustment_set": ["X"]
}
```

实际字段和约束以：

```text
src/schemas/adjustment_set_schema.json
```

中的定义为准。

整体流程：

```text
LLM
 ↓
Structured Output
 ↓
JSON Schema
 ↓
Validation
 ↓
Result Extraction
```

Structured Output 的主要作用是保证：

* 输出格式统一；
* 字段名称固定；
* 输出结果能够被程序解析；
* 减少自由文本带来的解析问题；
* 方便后续实验结果统计。

---

# 9. Transformers 与 vLLM

项目目前支持两种主要的 LLM 推理方式：

```text
Transformers
vLLM
```

## 9.1 Transformers

Transformers 用于直接加载模型并完成推理。

运行：

```bash
python src/run_transformers.py
```

主要用于模型实验、功能验证以及基准实验。

---

## 9.2 vLLM

vLLM 用于高效的 LLM 推理。

项目中的 vLLM 推理程序：

```text
src/run_vllm.py
```

运行：

```bash
python src/run_vllm.py
```

在 Kaggle 环境中，项目结合 Kaggle GPU 环境运行 vLLM 实验。

同时使用 Structured Output 对模型结果进行结构化约束。

---

# 10. 项目结构

当前项目整体结构如下：

```text
causal_llm/
│
├── config/
│   └── experiment.yaml
│
├── src/
│   │
│   ├── config.py
│   ├── logger.py
│   ├── result_writer.py
│   ├── run_transformers.py
│   ├── run_vllm.py
│   ├── seed.py
│   │
│   └── schemas/
│       └── adjustment_set_schema.json
│
├── results/
│   ├── results.csv
│   └── results.json
│
├── logs/
│   └── causal_llm_baseline.log
│
├── requirements.txt
├── set_up_kaggle.sh
└── README.md
```

---

# 11. 核心文件说明

| 文件                                       | 作用                           |
| ---------------------------------------- | ---------------------------- |
| `config/experiment.yaml`                 | 实验配置                         |
| `src/config.py`                          | 配置文件读取                       |
| `src/seed.py`                            | 随机种子设置                       |
| `src/logger.py`                          | 实验日志管理                       |
| `src/result_writer.py`                   | 实验结果保存                       |
| `src/run_transformers.py`                | Transformers 推理程序            |
| `src/run_vllm.py`                        | vLLM 推理程序                    |
| `src/schemas/adjustment_set_schema.json` | LLM Structured Output Schema |
| `results/results.csv`                    | CSV 格式实验结果                   |
| `results/results.json`                   | JSON 格式实验结果                  |
| `logs/`                                  | 实验日志                         |
| `requirements.txt`                       | Python 环境依赖                  |
| `set_up_kaggle.sh`                       | Kaggle 实验环境配置                |
| `README.md`                              | 项目说明文档                       |

---

# 12. 实验配置

项目使用统一的配置文件管理实验参数：

```text
config/experiment.yaml
```

将实验配置与代码分离，可以减少实验过程中直接修改源代码的情况，同时提高实验的可复现性。

配置文件主要用于管理：

* 实验名称；
* 模型相关参数；
* 推理参数；
* 日志路径；
* 结果路径；
* 其他实验设置。

后续实验可以通过修改配置文件进行管理，而不需要频繁修改核心代码。

---

# 13. 随机种子与可复现性

为了提高实验的可重复性，项目统一设置随机种子。

相关代码：

```text
src/seed.py
```

实验运行过程中固定随机种子，从而尽可能减少由于随机初始化、采样等因素产生的实验差异。

项目后续实验将继续遵循：

```text
固定随机种子
+
固定配置
+
保存日志
+
保存结果
+
Git 版本管理
```

的实验规范。

---

# 14. 日志系统

实验运行过程中产生的日志保存于：

```text
logs/
```

例如：

```text
logs/causal_llm_baseline.log
```

日志用于记录：

* 实验启动信息；
* 模型加载信息；
* 推理过程；
* 错误信息；
* 实验结果；
* 其他运行状态。

日志系统可以帮助定位实验运行过程中出现的问题，并提高实验过程的可追踪性。

---

# 15. 实验结果

实验结果主要保存为两种格式：

```text
results/
├── results.csv
└── results.json
```

## 15.1 CSV

CSV 文件：

```text
results/results.csv
```

主要用于：

* Pandas 分析；
* Excel 查看；
* 实验结果统计；
* 表格生成。

---

## 15.2 JSON

JSON 文件：

```text
results/results.json
```

主要用于：

* 程序读取；
* 结构化结果保存；
* 后续 Agent 实验；
* 进一步的数据处理。

---

# 16. 环境安装

首先克隆项目：

```bash
git clone https://github.com/Xmyxpc/causal_llm.git
cd causal_llm
```

安装 Python 依赖：

```bash
pip install -r requirements.txt
```

具体 Python、CUDA、PyTorch、Transformers 和 vLLM 版本以项目实际配置为准。

---

# 17. 运行 Transformers 实验

进入项目根目录：

```bash
cd causal_llm
```

运行：

```bash
python src/run_transformers.py
```

实验完成后查看：

```text
results/
logs/
```

---

# 18. 运行 vLLM 实验

运行：

```bash
python src/run_vllm.py
```

在 Kaggle GPU 环境中运行时，程序会根据当前实验环境进行相应的 vLLM 初始化。

实验完成后：

```text
results/
├── results.csv
└── results.json
```

会保存对应实验结果。

---

# 19. Kaggle 实验环境

本项目使用 Kaggle 作为远程 GPU 实验环境之一。

Kaggle 的主要作用是：

* 提供 GPU 计算资源；
* 运行 LLM 实验；
* 执行 vLLM 推理；
* 生成实验结果；
* 保存实验日志。

项目代码则通过 GitHub 进行统一管理。

推荐工作流程：

```text
                GitHub
                   │
                   │ git clone / git pull
                   ↓
               Kaggle
                   │
                   │ Run Experiment
                   ↓
             Experiment
                   │
          ┌────────┴────────┐
          ↓                 ↓
       results/            logs/
          │
          ↓
     results.csv
     results.json
          │
          ↓
     Kaggle Dataset
```

---

# 20. GitHub 代码管理

GitHub 用于保存项目源代码、配置文件和实验脚本。

项目地址：

https://github.com/Xmyxpc/causal_llm

基本 Git 工作流程：

```bash
git status
```

查看当前修改。

```bash
git add .
```

添加修改。

```bash
git commit -m "update experiment"
```

创建版本提交。

```bash
git push origin main
```

推送到 GitHub。

---

# 21. Kaggle 获取最新代码

第一次使用：

```bash
git clone https://github.com/Xmyxpc/causal_llm.git
```

以后如果项目已经存在：

```bash
cd /kaggle/working/causal_llm
git pull origin main
```

因此推荐的日常工作流程是：

```text
本地开发
    ↓
Git commit
    ↓
Git push
    ↓
GitHub
    ↓
Kaggle git pull
    ↓
运行实验
```

这样可以保证 Kaggle 中使用的代码与 GitHub 中的代码保持同步。

---

# 22. GitHub 与 Kaggle Dataset 的职责划分

项目将**代码**与**实验结果**分开管理。

## GitHub

主要保存：

```text
源代码
配置文件
实验脚本
Schema
依赖文件
README
```

即：

```text
GitHub = Code + Version Control
```

---

## Kaggle Dataset

主要保存：

```text
results.csv
results.json
```

即：

```text
Kaggle Dataset = Experiment Outputs
```

当前结果数据集：

https://www.kaggle.com/datasets/erbaisuidelongnv/causal-llm-outputs

这种方式可以避免大量实验结果直接污染代码仓库，同时也方便对实验结果进行版本管理。

---

# 23. 实验版本管理

一个完整实验版本可以对应：

```text
Git Commit
    +
Experiment Configuration
    +
Random Seed
    +
Experiment Result
    +
Experiment Log
```

例如：

```text
GitHub
└── commit: xxx
        │
        ├── source code
        ├── experiment.yaml
        └── schema
                │
                ↓
             Kaggle
                │
                ├── log
                └── results
                       │
                       ├── results.csv
                       └── results.json
```

通过这种方式，可以在后续研究中追踪：

> 某一组实验结果究竟对应哪一个版本的代码和配置。

---

# 24. 当前实验流程

当前实验的基本流程为：

```text
1. 加载实验配置
        ↓
2. 设置随机种子
        ↓
3. 初始化日志系统
        ↓
4. 加载 LLM
        ↓
5. 加载 JSON Schema
        ↓
6. 构造因果推理 Prompt
        ↓
7. LLM 生成结构化结果
        ↓
8. JSON Schema 验证
        ↓
9. 提取 Treatment / Outcome / Adjustment Set
        ↓
10. 保存实验结果
        ↓
11. 保存实验日志
```

---

# 25. 当前实验任务

当前实验主要围绕一个结构化因果图开展基础验证。

模型需要根据给定的因果结构完成：

```text
Treatment Identification
        +
Outcome Identification
        +
Variable Role Identification
        +
Adjustment Set Selection
```

该实验作为后续更加复杂的因果调整集选择实验的基础。

---

# 26. 后续研究方向

项目后续计划进一步扩展以下方向。

## 26.1 更复杂的因果图

从简单 DAG 扩展到：

* 多变量因果图；
* 多条后门路径；
* 多个混杂变量；
* 多个中介变量；
* 多个碰撞变量；
* 更复杂的 DAG 结构。

---

## 26.2 Adjustment Set 自动验证

未来将增加独立的因果规则检查模块。

整体流程：

```text
LLM 提出 Adjustment Set
          ↓
      Validator
          ↓
     检查因果条件
          ↓
   ┌──────┴──────┐
   ↓             ↓
 Valid         Invalid
   ↓             ↓
 输出结果       返回 LLM
                 ↓
              修正集合
```

---

## 26.3 Adjustment Set Selection Agent

项目最终希望从单次 LLM 推理进一步发展为 Agent。

目标结构：

```text
              ┌──────────────┐
              │ Causal Graph │
              └──────┬───────┘
                     ↓
              ┌──────────────┐
              │     LLM      │
              │ Candidate AS │
              └──────┬───────┘
                     ↓
              ┌──────────────┐
              │  Validator   │
              └──────┬───────┘
                     ↓
                是否满足？
                /       \
              Yes        No
               ↓          ↓
          Final AS     Feedback
                           │
                           ↓
                          LLM
```

即：

> Agent 不只是“一次性回答调整集”，而是能够提出、检验、修正调整集。

---

# 27. 研究目标

最终希望构建一个面向因果调整集选择的智能 Agent，使其能够完成：

```text
Causal Graph
      ↓
Causal Structure Understanding
      ↓
Candidate Adjustment Set Generation
      ↓
Causal Criterion Checking
      ↓
Error Detection
      ↓
Adjustment Set Revision
      ↓
Final Adjustment Set
```

从而探索大语言模型与因果推断相结合的研究方法。

---

# 28. 实验规范

为了保证研究结果的可靠性和可复现性，项目遵循以下实验原则：

### 固定随机种子

保证实验具有较好的重复性。

### 配置文件管理

将实验参数统一放入配置文件。

### 日志落盘

保存实验运行过程。

### 结果落盘

统一保存 CSV 和 JSON 结果。

### Git 版本管理

所有核心代码通过 Git 进行版本控制。

### 代码与结果分离

GitHub 管理代码，Kaggle Dataset 管理实验结果。

---

# 29. 项目当前状态

当前项目处于研究开发阶段。

已经建立：

* LLM 因果推理实验框架；
* Adjustment Set Selection 基础任务；
* Transformers 推理流程；
* vLLM 推理流程；
* Structured Output；
* JSON Schema；
* 实验配置管理；
* 随机种子管理；
* 日志管理；
* 结果保存；
* GitHub 版本管理；
* Kaggle GPU 实验环境；
* Kaggle Dataset 实验结果管理。

后续将重点推进：

* 因果规则验证；
* Adjustment Set 自动检查；
* Candidate Adjustment Set 生成；
* Adjustment Set 自动修正；
* Agent 化流程；
* 多轮因果推理；
* 更复杂的因果图；
* 多模型实验；
* 系统化实验评价。

---

# 30. Project Links

### GitHub

https://github.com/Xmyxpc/causal_llm

### Kaggle Dataset

https://www.kaggle.com/datasets/erbaisuidelongnv/causal-llm-outputs

---

# 31. Citation

如果本项目后续形成论文或正式研究成果，将在此处补充对应的论文引用信息。

---

# 32. Acknowledgement

本项目主要用于因果学习、大语言模型以及智能 Agent 相关研究与实验。

项目仍处于持续开发阶段，代码结构、实验设置和研究方法可能会随着研究进展进行调整。
