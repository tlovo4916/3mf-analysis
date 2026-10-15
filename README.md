# 通用 3MF 几何分析器

**v0.5.0 · 面向机械件的网格证据分析与辅助逆向原型**

读取 3MF 中的网格、组件和变换，提取平面、圆柱、圆锥、球面候选，生成可追溯的质量检查、截面和用途证据。适用的单壳薄板可以提出待人工审核的重建计划。

> **不是“一键 3MF 转参数化 CAD”。** 几何候选不等于原设计特征，分析流程跑完不等于找全，更不等于 CAD 重建完成。当前未实际执行 SolidWorks 2024，未生成或验收 SLDPRT。

[快速开始](#快速开始) · [本版改进](#v050-改进) · [验证结果](#验证结果) · [详细使用说明](src/README.md) · [验证与限制](docs/VALIDATION.md) · [版本记录](docs/CHANGELOG.md)

## 能做什么

- **保留来源的 3MF 读取**：build 实例、嵌套组件、跨模型引用、单位及变换链；保留源文件快照、哈希和面/顶点索引。
- **解析曲面候选**：连通平面、任意方向圆柱、圆锥和球面，记录边界、拟合残差及整片三角面误差或保守上界。
- **多方向截面**：保留嵌套轮廓、原失败、来源映射和坐标系；完整证据以无损压缩 JSON 保存。
- **网格与关系检查**：退化面、开放/非流形边、有向体积及支持曲面关系；不把包围盒关系当成真实包含证明。
- **切片器角色证据**：保守解释 Bambu 配置中的普通材料、负工具、modifier 和 support 声明。
- **受限重建原型**：仅对符合前置条件的单壳薄板生成审核计划；来源或几何阻塞不能由批准标志覆盖。

**默认不补洞、不翻转源法向、不删除源面、不跨组件焊接、不静默做布尔或套用私有装配变换。** 仅在同一叶组件 occurrence 内精确同坐标合并顶点；拟合视图跳过的退化面仍保留在原始证据中。

## 快速开始

需要 **Python 3.11+**。分析不需要安装 SolidWorks。

获取当前版本后，以下命令均从项目根目录执行：

```sh
git clone --branch feature/3mf-v0.5.0 --single-branch https://github.com/tlovo4916/3mf-analysis.git
cd 3mf-analysis
```

### Windows：拖拽分析

1. 进入 `src`，双击 `setup.cmd` 安装依赖。
2. 将一个或多个 `.3mf` 文件拖到 `analyze.cmd`。
3. 打开 `src/results/000-文件名/index.html`。

重复分析会新建目录，不覆盖已有结果。也可从本项目根目录执行：

```powershell
py -m pip install -r src/requirements.txt
py src/assistant.py "C:\Models\part.3mf" --out results
```

### Linux / macOS

从本项目根目录执行：

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r src/requirements.txt
.venv/bin/python src/assistant.py /path/to/part.3mf --out results
```

Windows 命令中的模型路径、Linux/macOS 的 `/path/to/part.3mf` 均需替换为实际文件。依赖范围见 [requirements.txt](src/requirements.txt)，虚拟环境不随项目交付。

### 常用选项

```sh
python src/assistant.py part-a.3mf part-b.3mf --out results
python src/assistant.py part.3mf --out strict-results --strict
python src/assistant.py part.3mf --out results --tolerance 0.05
python src/assistant.py part.3mf --out quick-results --section-directions 1
python src/assistant.py part.3mf --out large-results --max-faces 500000 --max-bodies 256
```

- 默认兼容模式仅读取受支持的核心几何；未知必需几何扩展仍明确拒绝。
- `--strict` 恢复严格读取，不放行非 model 对象或必需外观扩展，不隔离 build 项错误。
- `--tolerance` 单位为毫米，范围为 `(0, 0.5]`；顶点达标不代表整片三角面达标。
- 默认候选分析预算为 **250,000 面、128 连通壳**；最多 3 个截面方向，主方向最多 40 层，其余最多 7 层。
- 超预算壳仍列出并保留源数组。读取、分壳和证据写出不受候选预算截断；这些预算不是严格时间或内存上限。

## 怎样阅读结果

```text
results/000-part/
├── index.html                         汇总报告
├── analysis.json                      参数、用途、来源、质量和状态
├── source.3mf                         原始快照
├── instance-000-source-mesh.npz        毫米顶点与完整面数组
└── instance-000-body-000/
    ├── sections.json                  截面摘要与完整证据指针
    ├── sections.full.json.gz          未舍入的完整几何和来源
    ├── reference-section.svg          仅参考截面成功时生成
    ├── section_direction_01/...
    └── plan.json / report.html        仅适用且通过闸门时生成
```

**先看状态和阻塞，再看候选及误差，最后核查原始来源；不要只看候选数量或程序退出码。**

| 字段 | 应如何理解 |
| --- | --- |
| `status` | `analysis_complete_not_cad_complete` 仅表示已配置分析尝试成功；`analysis_partial_not_cad_complete` 表示存在遗漏或失败。两者都不是 CAD 完成 |
| `candidate_analysis_coverage_complete` | 已配置面/壳的候选计算尝试无跳过或错误，不是几何召回完整 |
| `candidate_search_exhaustive` / `candidate_recall_certified` | 均为 `false`，不认证已找到所有特征 |
| `has_candidate_search_limits` | 本轮局部搜索是否触发预算截断；`false` 也不等于穷尽 |
| `has_reconstruction_blockers` | 来源、用途、覆盖、网格、壳关系、截面或计划阻塞汇总；没有阻塞也不代表已有适用重建器 |
| `material_semantics_complete` | 仅指纳入的叶组件角色均唯一映射为 `normal_part`，不是最终打印材料或制造语义认证 |

CLI 退出码 **0 只表示报告生成成功**，报告仍可能是部分分析。`sampling_complete` 始终为 `false`。

## v0.5.0 改进

### 1. 从混合曲率区域恢复局部圆柱

相邻原始面法向叉积形成任意方向轴提案，提取连通子区域，再调用**未放宽的原严格圆柱拟合器**。不依赖固定 XYZ、文件名、样本面号或源组件轴。

增加轴向可观测性门槛，避免将球面或环面上的偶然等半径窄带升级为圆柱；邻域步长按完整严格共面面片计算，避免不改变几何的重剖分虚增证据。每个候选保留父区域、提案依据和源面索引。

搜索按区域轮询并受预算约束：每区域最多 4096 轴代表、64 提案簇、12 轴；每壳最多 96 轴、192 次区域拟合，每轴最多 8 区域。它仍是有界启发式，短、噪声大或预算未覆盖的支持可能遗漏。

### 2. 区分材料与负工具声明

沿用全包安全 XML 解析与累计预算，读取 Bambu `Metadata/model_settings.config`。只接受可唯一关联的直接组件角色；未知、缺失、重复、嵌套、跨模型 ID 冲突或证据截断均明确未解析并阻止重建。

plate/assembly 仅作上下文，不应用私有矩阵、不跟随配置 URI、不计算布尔最终材料。builder 从原声明及叶组件证据重新计算否决，不能仅通过删除 blocker 列表解禁。

### 3. 事件两侧补充，不抹去失败

仅对 `ambiguous_section_event` 追加有限偏移截面；每侧最多 4 次，每方向额外预算为 `2 × max_samples`，不跨原始相邻顶点事件。间隙或数值清晰度不足时明确标为不可探测。

补充证据不进入原样本、不替代失败、不更改原 reference/profile 判断、不解除 CAD 否决，也不证明精确单侧极限。

## 验证结果

以下为已保存结果，不是总体准确率承诺。详见 [完整验证说明](docs/VALIDATION.md) 和 [核验摘要](validation/verification.json)。

- **400 项本地自动测试通过**，包括模拟 COM；模拟测试不等于 SolidWorks 实测。
- 新参数合成扫掠：**24/24** 预期圆柱支持匹配，**12** 个旋转环面未输出圆柱候选。
- **10 个官方例子**：9 个核心几何可读，其中 4 个仍部分分析；1 个必需梁晶格扩展明确拒绝。
- **3 份真实作者公开 3MF**：均生成报告，均仍为部分分析，均未生成板件计划或可编辑 CAD。
- **原用户样本**：退化面和侧向截面失败仍保留，没有为了生成计划而修复或隐藏失败。

### 同一批真实文件对照

| 样本 | 源三角面 / 连通壳 | 圆柱候选 v0.4.1 → v0.5.0 | 原截面成功 / 失败 | CAD 完成 |
| --- | --- | --- | --- | --- |
| 空气质量监测器 | 7,250 / 2 | 9 → 9 | 76 / 3 | 否 |
| 小 OLED | 42,512 / 43 | 32 → 32 | 1,210 / 180 | 否 |
| 复古 OLED | 219,816 / 120 | 0 → 24 | 2,884 / 4 | 否 |

复古 OLED 中，原诊断已确认遗漏的 **4 片支持面集合精确恢复**，对应 2 个位置 × 2 个实例。其余 20 片只确认通过几何拟合，未获得独立设计意图真值；不能称为“恢复 24 个孔”，也不能计算全模型召回率。

真实文件中的 7 个事件失败仍保留；14 侧仅 1 侧获得有效补充，13 侧不可安全探测。原主壳的部分截面方向受新候选引导发生变化，已在验证记录中披露，不能据失败计数变化推断准确率。

**来源边界：** 三份文件来自两个作者公开 GitHub，README 关联相应 MakerWorld 作品；不是 MakerWorld 站点直接下载，也未证明与站点 profile 逐字节相同。它们均为 Bambu 工程，不能代表多数出口软件。

## 开发与复现

从项目根目录运行自动测试：

```sh
cd src
python -m unittest discover -v
```

合成参数扫掠从项目根目录运行：

```sh
python validation/parameter_sweep.py
```

**源码与完整证据包分开理解：** 本仓库只包含源码、自动测试、说明、验证摘要和公开来源指针；不包含原始 3MF、用户模型、旧版基线、完整 runs、HTML/ZIP 证据包或凭据。它们不是运行分析器的依赖。全量核验脚本保留用于配套完整证据包，不能在缺少数据的源码克隆中直接运行。

[仓库范围与发布自检](docs/REPOSITORY.md)列出已纳入文件、证据包校验值及本次发布副本的检查结果。

在完整证据包根目录可运行：

```sh
python validation/verify_generalization.py
python validation/run_corpus_case.py 03-oled-retro inputs/real/03-oled-retro.3mf
```

新主流程默认写 `validation/reruns`，已存在的目标目录拒绝覆盖；核验脚本检查原 `validation/runs`，不核验新 reruns。核验和参数扫掠会更新自身派生 JSON，需要保留历史时先复制。

### 代码入口

| 文件 | 职责 |
| --- | --- |
| [assistant.py](src/assistant.py) | 批量分析入口、状态汇总与 HTML 报告 |
| [threemf_io.py](src/threemf_io.py) | 3MF/OPC 安全读取、来源及变换 |
| [geometry_analysis.py](src/geometry_analysis.py) | 定向、平面和圆柱支持拟合 |
| [cylinder_segmentation.py](src/cylinder_segmentation.py) | 有界局部圆柱区域恢复 |
| [analytic_surfaces.py](src/analytic_surfaces.py) | 圆锥、球面候选 |
| [slicer_metadata.py](src/slicer_metadata.py) | 切片器声明、关联证据与来源否决 |
| [section_analysis.py](src/section_analysis.py) | 实际截面、拓扑检查与事件补充 |
| [reverse_assistant.py](src/reverse_assistant.py) | 完整证据导出与薄板审核计划 |
| [build_solidworks.py](src/build_solidworks.py) | 审核校验与可选 SolidWorks 构建入口 |

## 可选 SolidWorks 2024 构建

仅适用于确已生成且经人工审核的薄板 `plan.json`。需要 Windows、已安装并许可的 SolidWorks 2024、默认零件模板及 `pywin32`。命令与审批字段见 [详细使用说明](src/README.md#solidworks-2024-可选薄板构建)。

离线校验不创建零件、不证明 COM 可用。体积检查也不能单独证明孔位、形状或源几何等价。**当前没有实际构建或参数修改验收回执。**

## 已知限制与项目边界

- 尚无通用裁剪 BRep、自由曲面/NURBS、完整设计历史恢复，以及全局自交或壳包含认证。
- 未执行最终材料布尔，未恢复切片器私有装配姿态或完整制造配置。
- 候选误差是声明范围内的几何证据，不是双向三维 CAD 等价证书；粗网格即使顶点拟合通过，整片弦高仍可能超差。
- 当前为独立 v0.5.0 版本，未合并覆盖原工程；请整套使用，勿逐文件混用版本。
- 参考 [USD-BRep](https://github.com/NVIDIA-Omniverse/usd-brep) 的支持曲面、边界拓扑、实例与计划分层思路；未集成其运行内核，不输出已认证 USD BRep。
- 第三方样本保留原作者权属与许可；公开访问不等于获得任意再分发授权，Git 发布时应单独核对许可，不默认纳入样本与大体积证据。

后续重点是跨作者、跨出口软件的真实标注与误报/遗漏评估，以及真实 SolidWorks 2024 的逐特征和参数修改验收，而不是仅增加候选数量。

## 许可与样本

当前仓库未声明开源许可证；代码公开可见不等于授予任意使用、修改或再分发许可，相关授权请向仓库所有者确认。第三方样本不随本仓库分发；`provenance/` 仅保存公开来源和校验记录，不替代原作者许可。
