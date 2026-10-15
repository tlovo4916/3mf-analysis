# 通用 3MF 几何分析器 v0.5.0

**用途：通用机械网格证据分析 + 需人工审核的单壳薄板重建原型。不是通用参数化 CAD 转换器。**

本次改进面向局部几何机制，不含文件名、固定面号、组件轴或样本专用分支。400 项本地自动测试通过；3 份真实作者公开文件、10 个官方例子及原用户样本完成复测。没有实际执行 SolidWorks 2024，没有生成或验收 SLDPRT。

这是 `src` 目录内的详细操作说明。项目概览、从项目根目录运行的命令和发布边界见 [仓库首页 README](../README.md)；验证细节见 [VALIDATION.md](../docs/VALIDATION.md)。本页未特别说明的分析与构建命令均在 `src` 内执行。

## 立即使用

Python 3.11+。Windows 解压后进入 `src`：

1. 双击 `setup.cmd` 安装依赖。
2. 把一个或多个 `.3mf` 拖到 `analyze.cmd`。
3. 打开 `results/000-文件名/index.html`。重复分析使用新目录，不覆盖旧结果。

或在命令行运行：

```sh
cd src
python -m pip install -r requirements.txt
python -m unittest discover -v
python assistant.py part.3mf --out results
```

Linux/macOS 建议先创建虚拟环境。源码仓库不含虚拟环境或输入样本，须在目标系统安装依赖，并将下面路径替换为你自己的 3MF 文件。本段仍在 `src` 内执行。

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python assistant.py /path/to/part.3mf --out results
```

`assistant.py` 默认输出完整参数、来源、质量与失败证据。退出码 0 **只表示报告已生成**；报告仍可能为部分分析、候选搜索截断或禁止重建。

## v0.5.0 改进

### 局部圆柱恢复，而不是放松拟合

相邻原始面法向叉积提出局部轴，再寻找与该轴近垂直的连通子区域，调用原版严格圆柱拟合器。不依赖世界 XYZ 或源组件轴，不改变网格、绕向、误差容差。

每片保留来源面、父法向 band、轴提案和轴向可观测性。新恢复还要求法向垂直轴误差 ≤0.05°、轴长/半径 ≥0.25；两侧平滑连接时，轴长相对完整严格共面邻片的轴向步长至少为 2。使用完整邻片而非单三角形，避免邻面不改变几何的细分虚增可观测性。

搜索是**有界启发式**：每 band 最多 4096 轴代表、64 提案簇、12 个轴；每壳最多 96 轴、192 区域拟合，每轴最多 8 区域。band 间轮询。短、噪声大、浅弧、未获预算的支持仍可能遗漏，不能据执行成功宣称找全。

真实复古 OLED：圆柱候选 0→24；原独立诊断的 4 片（2 个位置 × 2 实例）精确恢复。另 20 片只确认满足原版几何拟合及误差规则，未获得独立语义真值；不是新增 24 个已确认孔。官方齿轮例也出现恢复，但没有完整真值，不能计算总体召回率。

### 切片器用途证据

安全、有预算地读取 Bambu `Metadata/model_settings.config`，保守关联唯一、直接的对象/part，保留 `negative_part`、modifier、support 等声明。plate/assembly 只作上下文，不应用私有 transform、不跟随 URI、不执行布尔。

未知、缺失、重复、嵌套、重复 occurrence、跨模型 ID 冲突或证据截断均明确 unresolved 并否决重建。`normal_part` 不等于允许建模。builder 从原始声明与叶组件证据复算否决，不能仅清空 blocker 列表越过。

`material_semantics_complete` **仅指已纳入叶组件角色都唯一映射为 normal_part**，不代表制造配置或打印实体恢复。没有切片器证据时是 `not_applicable`、false，保留原 core-only 行为。

### 截面事件两侧证据

只对 `ambiguous_section_event` 补充有限偏移截面，保留原失败、reference/profile 状态和来源映射。每侧最多 4 次指数探测，每方向额外预算 `2 × max_samples`；不跨原始相邻顶点事件。间隙不足、数值分辨率不足、端点外侧明确 unavailable。

补充样本不进入原 `samples`、不替代失败、不解除 CAD 否决，也不证明精确单侧极限。真实 3 文件的 7 个事件失败仅获得 1 个有效侧面证据，其余 13 侧不可安全探测；180 个开放/分支截面不触发“偏层修复”。

## 输入和工作量选项

```sh
python assistant.py part.3mf --out results --tolerance 0.05
python assistant.py part.3mf --out strict-results --strict
python assistant.py part.3mf --out all-items --include-nonprintable
python assistant.py part.3mf --out quick-results --section-directions 1
python assistant.py part.3mf --out large-results --max-faces 500000 --max-bodies 256
```

- 默认兼容模式允许已知 Materials 外观扩展仅读取核心网格，并隔离可安全隔离的 build 项错误。必需未知几何扩展仍拒绝。
- `--strict` 不放行外观扩展或非 model 对象，不隔离 build 错误；新角色解析不会放宽 strict。
- `--tolerance` 毫米，范围 `(0,0.5]`。
- 默认 250,000 面、128 连通壳；最多 3 个独立截面方向，主方向最多 40 层，其余最多 7 层。局部圆柱另有上述预算。
- 超候选预算的壳仍列出，源数组保留。读取、分壳、源证据写出仍可能耗费时间/内存；这些不是严格运行时/内存上限。

## 输出

```text
results/000-part/
  source.3mf                         原始快照，保留哈希
  analysis.json                      schema 5；包/实例/壳/候选/用途/失败
  index.html                         可打开的汇总报告
  instance-000-source-mesh.npz        原毫米顶点和完整面数组
  instance-000-body-000/
    sections.json                    小体积摘要，含事件探测状态与 JSON 指针
    sections.full.json.gz            未舍入完整截面、探测、来源
    reference-section.svg            仅有成功参考截面时生成
    section_direction_01/...
    section_direction_02/...
    plan.json / report.html          仅适用且通过闸门的薄板才有
```

| 主要字段 | 边界 |
| --- | --- |
| `candidate_analysis_coverage_complete` | 已配置面/壳的候选计算尝试无跳过/错误，不是候选召回率 |
| `candidate_search_exhaustive` / `candidate_recall_certified` | 均为 false |
| `has_candidate_search_limits` | 本轮局部搜索是否发生预算截断，false 也不等于穷尽 |
| `section_analysis_attempts_complete` | 已配置原截面尝试全部成功，不是连续高度认证 |
| `event_probe_summary` | 补充探测统计，不与原样本成功数合并 |
| `has_reconstruction_blockers` | 来源/角色/覆盖/网格/壳关系/截面/计划阻塞汇总；false 不代表有适用重建器 |
| `material_semantics_complete` | 窄义声明角色映射，绝非最终打印材料认证 |

全部候选仅为支持几何。平三角片误差不等于顶点误差；粗圆柱即使顶点拟合成功，整片弦高超差仍标 false。`sampling_complete` 始终为 false。坐标是 `3MF_core_build_world_mm_not_slicer_assembly_pose`。

## 复测证据与复现

源码运行与完整证据核验分开：分析器和自动测试只需 `src` 及其依赖；完整核验还需要原始输入、旧版基线和运行结果。源码仓库不默认包含这些大体积或第三方数据，缺失时不能直接执行全量核验。

以下命令在**完整证据包根目录**执行；仅参数扫掠不依赖真实输入和旧版基线：

```sh
python validation/run_corpus_case.py 03-oled-retro inputs/real/03-oled-retro.3mf
python validation/parameter_sweep.py
python validation/verify_generalization.py
```

- 正式保存结果在 `validation/runs`；新主流程默认写 `validation/reruns`，已有目标拒绝覆盖。可用 `--runs-root` 指定另一个新目录。
- `verify_generalization.py` 核对**原 runs**，不比较新 reruns。输出 `verification.json` 和候选清单；参数扫掠覆盖自身 JSON，需保留历史时先复制交付包。
- `baseline/v041` 是原交付源码、3 份原实测报告和原诊断的只读基线；本版没有覆盖 v0.4.1 或原工程。
- 400 项本地自动测试；24 个新参数组合的合成圆柱全部找到预期支持，12 个合成环面未新增圆柱；这不是实际出口软件总体准确率。
- 3 份真实文件来自作者公开 GitHub，README 链接对应 MakerWorld 作品；**不是站点直接下载，未证明与站点配置文件逐字节相同**。
- 10 个官方例：9 个可读，其中 4 个保留部分状态；1 个必需梁晶格扩展按预期拒绝。原用户样本仍保留退化面和侧向截面失败。

详见 [完整验证说明](../docs/VALIDATION.md)、[版本记录](../docs/CHANGELOG.md)、[核验摘要](../validation/verification.json) 和 [仓库范围说明](../docs/REPOSITORY.md)。本仓库的 `provenance/` 只含公开来源指针与校验值；全量输入、原版基线和运行证据仅在配套完整证据包中。

## SolidWorks 2024 可选薄板构建

需 Windows、已安装并许可的 SolidWorks 2024、默认零件模板及 `pywin32`。只适用于确已生成并审核的 `plan.json`；先检查近似差异，再明确 `accepted_approximation`、`approved_for_build`。

```powershell
py -m pip install pywin32
py build_solidworks.py results\000-part\instance-000-body-000\plan.json --validate-only
py build_solidworks.py results\000-part\instance-000-body-000\plan.json --output C:\CAD\rebuilt_v1.sldprt
```

离线校验不会创建零件或证明 COM 可用。构建保留版本/单实体/重建/各步体积检查，不覆盖已有文件或关闭用户文档。体积一致也不是孔位/曲面等价认证；草图未完全约束，源姿态不应用到新零件。模拟 COM 测试不是实际 SolidWorks 验收。

## 未完成事项

仍未实现通用裁剪 BRep、布尔最终材料、自由曲面/NURBS、全局自交/包含认证或通用设计历史。3 份真实样本全部仍为部分分析，未生成计划或可编辑 CAD。应继续扩大跨作者/出口软件标注，评估遗漏与误报，并在真实 SolidWorks 2024 验证特征和参数修改。

本版本是独立的 v0.5.0，未合并覆盖并发修改的原工程；不要逐个混用两个版本的源码。USD-BRep 仅借鉴支持曲面/拓扑/实例/计划分层，未集成其运行内核、未输出已认证 USD BRep。

第三方样本保留原作者权属与许可；公开可下载不等于任意再分发授权。Git 发布时应单独核对许可，不默认将样本、完整运行结果或证据 ZIP 纳入仓库。
