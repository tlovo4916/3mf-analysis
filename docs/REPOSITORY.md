# 仓库范围与发布核验

本仓库发布 v0.5.0 的独立源码版本。生产 Python 文件与已经完成复测的交付版逐文件一致，未修改拟合阈值、默认预算或重建否决。为源码发布补充了仓库首页、克隆说明、样本缺失说明及忽略规则。

## 纳入与排除

已纳入：

- `src/`：分析器、SolidWorks 可选构建器、400 项自动测试、依赖和 Windows 启动脚本。
- `README.md` 与 `docs/`：功能、使用、已知问题和验证边界。
- `validation/`：原交付 JSON 验证摘要、源码基线哈希、合成扫掠，以及发布副本的自检摘要。
- `provenance/`：三份作者公开文件及官方示例的公开地址、固定版本和校验值；没有模型内容。

不纳入：

- 用户模型、第三方 3MF/STL/STEP、源网格 NPZ、CAD 零件。
- 大体积 `runs`、旧版完整基线、截图、HTML/ZIP 证据交付包。
- 虚拟环境、缓存、调试日志、SSH 私钥/公钥及任何访问凭据。
- 其他工作区项目或原工程的并发改动。

完整证据包此前已单独交付，本次不将它作为 Git 对象上传，也没有凭空创建外部下载地址。

## 可在源码克隆中运行

```sh
cd src
python -m pip install -r requirements.txt
python -m unittest discover -v
python assistant.py /path/to/part.3mf --out results
```

从仓库根目录执行合成参数扫掠：

```sh
python validation/parameter_sweep.py
```

真实输入自行提供后，也可以运行：

```sh
python validation/run_corpus_case.py my-part /path/to/part.3mf
```

新结果写入 `validation/reruns/my-part`；已有目录拒绝覆盖。全量 `verify_generalization.py` 则还需要配套证据包中的旧版源码、原始运行和输入，不能只凭仓库内摘要完成。

## 已单独交付的证据包

- 文件名：`3mf-analyzer-v0.5.0-source-evidence.zip`
- 大小：173,293,659 字节。
- SHA-256：`64257dfd40529ef9f1c99711df029308f1fc70f626bd7a644e9b9cc74316b2e1`

该证据包是此前的不可变交付，包含当时版本的说明；本仓库后续更新的 README 不会追溯改写其字节或校验值。第三方作品仍受原作者许可约束。

## 验证结果应如何理解

- 原交付：400 项本地自动测试通过，24 个合成正例匹配，12 个合成环面未输出圆柱；完整检查 738 份截面 gzip 和 52 对源数组。
- 新发布副本的实际重跑结果、生产源码一致性、语法/文档检查记录在 `validation/publication-check.json`。
- 三个真实样本仍全部部分分析；几何候选不是设计意图，搜索和召回未获穷尽认证。
- 没有实际 SolidWorks 2024 执行、原生特征树、参数修改或 SLDPRT 验收。

本仓库未声明开源许可证；公开可见并不替代代码使用授权或第三方样本再分发许可。没有因此修改、补授或推定原作品许可。
