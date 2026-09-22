# v1.1.0 — Scope & platform boundaries

**Learned-controller envelope characterised; Phase C closed.**

This release adds no trained models. It publishes the results of a scenario-fidelity
audit that maps where the learned following controller is trustworthy, where it is
not, and why — plus the closure of the self-play line of work and four errata.

---

## Highlights

1. **A measured trust envelope for the learned controller.** An 11-cell × 3-controller
   × 100-seed sweep (`results/20260922_D0_scenario_audit/`, 3300 episodes) pins the
   following-policy accuracy advantage to the trained speed band **v ≤ 0.50 m/s**.
   At 1.0 m/s the advantage *reverses* — the rule baseline with feed-forward wins
   every cell tested. The boundary and the reasoning are written up in
   [`docs/scenario_scope_statement.md`](docs/scenario_scope_statement.md): what this
   system can and cannot claim about itself.

2. **A reproducible failure regularity, not just a failure.** Extending the training
   distribution into the higher-speed band does not merely under-perform — it
   **destabilises training outright**, with policy action-noise (std) exploding
   rather than performance degrading gradually. Six runs at an identical update
   budget isolate the trigger: the *presence of the high-speed band*, independent of
   band width, curriculum shape, or action re-parameterisation.
   See [`docs/scenario_scope_statement.md`](docs/scenario_scope_statement.md) §2.3.

3. **Phase C closed honestly.** The self-play / adversarial-criterion line of work is
   stopped and RQ3 is recorded as an **open problem** rather than a result. Four
   criterion designs were each defeated by degenerate policies, and two successive
   "corrections" were themselves falsified. What survives is documented and reusable
   (a scripted-blocker test bed and a timing counter-script).

---

## What's new

### D0 scenario-fidelity audit

- **Envelope sweep** — `results/20260922_D0_scenario_audit/` : raw per-seed values,
  summary table, run config with the frozen-file hash, and the envelope figure
  (`fig_D0_envelope_v1.png`).

  | Speed band | Result (settled window, obs-cm) |
  |---|---|
  | v ≤ 0.50 (trained) | learned controller wins every cell, e.g. 0.53 vs 3.52; 0.62 vs 2.09 |
  | v = 0.55 (boundary) | margin collapses; flips to the rule baseline at the tightest gap |
  | v = 1.0 | reverses in all five cells (7.42/7.97/7.65 vs 3.30/5.73/6.21) |

- **Full write-up** — [`docs/scenario_audit.md`](docs/scenario_audit.md), including the
  reporting convention (settled window, observation units) and the in/out-of-distribution
  boundary, both registered *before* the run.

- **Platform speed bound** — six stage-2 runs at 610 updates each. The narrow trained
  band is stable (std 1.40); every run whose band includes the high-speed regime
  diverges (std 3.98, 4.40, 4.45, 5.17, 6.13). The bound and its open causal question
  are stated in the scope statement.

- **Scope statement** — [`docs/scenario_scope_statement.md`](docs/scenario_scope_statement.md):
  the authoritative statement of the trustworthy envelope and the platform speed bound.
  It is the document to cite when describing what this project claims.

### Phase C closure

- `research/roadmap.md` Phase C now carries a closure note: RQ3 is an open problem;
  the four defeated criterion versions and the two falsified corrections are recorded.
- **Retained assets** (reusable, if the line is ever reopened): a scripted-blocker
  test bed under which geometric degenerate policies score 0.00, and a timing
  counter-script whose completion rate is 0.77 (reproducible across three seed sets).
- The mechanism claim that was retracted twice is withdrawn from the record; do not
  cite its numbers.

---

## Corrections & errata

Four errata are published alongside this release. Each is a self-initiated correction
of previously committed content, following independent review and reproduction.

| File | What it corrects |
|---|---|
| [`ERRATUM_phaseC_findings.md`](ERRATUM_phaseC_findings.md) | A probe conclusion invalidated by an adapter bug (independently reproduced before retracting) |
| [`ERRATUM2_criterion.md`](ERRATUM2_criterion.md) | Withdrawal of a criterion design after review rejected it with three blocking issues, all reproduced |
| [`ERRATUM3_criterion_v234.md`](ERRATUM3_criterion_v234.md) | Withdrawal of criterion versions 2–4; records that on this track the dominant strategy is geometric, not adversarial |
| [`ERRATUM4_f23_correction_wrong.md`](ERRATUM4_f23_correction_wrong.md) | The *correction* of an earlier finding was itself wrong — two successive corrections, both recorded |

These errata extend the record; they do not revise any previously published number.
The v1.0.0 results stand as published.

---

## Assets

**No new model assets.** The checkpoints produced during the audit do not converge
and are deliberately not released. The v1.0.0 assets remain current:

| Asset | Where |
|---|---|
| `follow_stage2_final_v1.zip` — following policy | v1.0.0 |
| `overtake_final_v1.zip` — overtaking policy | v1.0.0 |
| `demos_v1.npz` — 300 episodes / 41,936 transitions | v1.0.0 |

No new model assets; see v1.0.0.

---

## Notes for readers

- The scope statement contains one externally-sourced reference that is **flagged
  as pending verification** within the document itself; it is not used as a load-bearing
  claim here.
- The audit's conventions (settled-window aggregation, observation units) are declared
  in `docs/scenario_audit.md` §1 so the new numbers are directly comparable with the
  v1.0.0 table.
- Everything new here is documentation, analysis scripts, and raw output. The
  frozen environment files are unchanged.

**Full changelog**: `git log --oneline v1.0.0..v1.1.0`


---

# v1.1.0 — 适用范围与平台边界

**学习控制器的包络已测绘；Phase C 已关闭。**

本版本不新增任何训练模型。它发布的是一项场景保真度审计的结果：测绘学习型跟随控制器
在何处可信、何处不可信、以及为什么——外加自博弈这条工作线的关闭与四份勘误。

---

## 要点

1. **为学习控制器测绘出一条可测量的可信包络。** 一项 11 格 × 3 控制器 × 100 种子
   的扫描（`results/20260922_D0_scenario_audit/`，共 3300 回合）把跟随策略的
   精度优势钉在训练速度带 **v ≤ 0.50 m/s** 之内。在 1.0 m/s 处该优势*反转*——
   带前馈的规则基线在所有测试格子获胜。该边界及其推理写在
   [`docs/scenario_scope_statement.md`](docs/scenario_scope_statement.md) 中：
   即本系统关于自身能声称什么、不能声称什么。

2. **一条可复现的失效规律，而不只是一次失效。** 把训练分布拓展进更高速度带，
   并非只是表现变差——而是**直接让训练失稳**：策略动作噪声（std）爆炸，
   而非性能渐变。六次相同更新预算的运行把触发器隔离了出来：**高速带的存在本身**，
   与频带宽度、课程形态、动作参数化方向均无关。
   见 [`docs/scenario_scope_statement.md`](docs/scenario_scope_statement.md) §2.3。

3. **诚实地关闭 Phase C。** 自博弈 / 对抗判据这条工作线已停止，RQ3 被记录为
   **未决问题**而非一项成果。四个判据设计各自被退化策略击穿，两次"更正"本身
   也被证伪。仍然成立的部分已被文档化且可复用（一个脚本化封堵者测试床，
   以及一个计时反制脚本）。

---

## 新增内容

### D0 场景保真度审计

- **包络扫描**——`results/20260922_D0_scenario_audit/`：逐种子的原始值、
  汇总表、含冻结文件哈希的运行配置，以及包络图（`fig_D0_envelope_v1.png`）。

  | 速度带 | 结果（稳定窗口，obs-cm） |
  |---|---|
  | v ≤ 0.50（训练带内） | 学习控制器在每个格子获胜，例如 0.53 对 3.52；0.62 对 2.09 |
  | v = 0.55（边界带） | 优势收窄；在最紧间距处翻转为规则基线领先 |
  | v = 1.0 | 五格全部反转（7.42/7.97/7.65 对 3.30/5.73/6.21） |

- **完整撰写**——[`docs/scenario_audit.md`](docs/scenario_audit.md)，含统计口径
  （稳定窗口、观测单位）与分布内/外划界，二者均在运行**之前**登记。

- **平台速度上界**——六次 stage-2 运行，各 610 次更新。窄训练带稳定（std 1.40）；
  每一个训练带包含高速区间的运行都发散了（std 3.98、4.40、4.45、5.17、6.13）。
  该上界及其未决的因果问题在范围声明中说明。

- **范围声明**——[`docs/scenario_scope_statement.md`](docs/scenario_scope_statement.md)：
  关于可信包络与平台速度上界的权威说明。描述本项目声称什么时，应引用该文件。

### Phase C 关闭

- `research/roadmap.md` 的 Phase C 现带有关闭注记：RQ3 是未决问题；
  四个被击穿的判据版本与两次被证伪的更正均已记录。
- **保留资产**（可复用，若该工作线日后重启）：一个脚本化封堵者测试床，
  几何退化策略在其下得分 0.00；以及一个计时反制脚本，其完成率为 0.77
  （在三个种子集上可复现）。
- 被撤回两次的机制主张已从记录中撤下；请勿引用其数字。

---

## 更正与勘误

本版本同时发布四份勘误。每一份都是对已提交内容的主动更正，均经过独立评审与复现。

| 文件 | 更正了什么 |
|---|---|
| [`ERRATUM_phaseC_findings.md`](ERRATUM_phaseC_findings.md) | 一个因适配器 bug 而失效的探针结论（撤回前已独立复现） |
| [`ERRATUM2_criterion.md`](ERRATUM2_criterion.md) | 撤回一个判据设计：评审以三条拦截级问题否决，且逐条已复现 |
| [`ERRATUM3_criterion_v234.md`](ERRATUM3_criterion_v234.md) | 撤回判据版本 2–4；记录本赛道上占优策略是几何的，而非博弈的 |
| [`ERRATUM4_f23_correction_wrong.md`](ERRATUM4_f23_correction_wrong.md) | 对某一发现的*更正*本身也是错的——两次相继的更正，均予记录 |

这些勘误是对记录的**扩展**；它们不修改任何此前已发布的数字。
v1.0.0 的结果按已发布原样成立。

---

## 资产

**无新模型资产。** 审计期间产出的 checkpoint 未收敛，故刻意不发布。
v1.0.0 的资产仍然有效：

| 资产 | 位置 |
|---|---|
| `follow_stage2_final_v1.zip` —— 跟随策略 | v1.0.0 |
| `overtake_final_v1.zip` —— 超车策略 | v1.0.0 |
| `demos_v1.npz` —— 300 集 / 41,936 条转移 | v1.0.0 |

无新模型资产；见 v1.0.0。

---

## 读者须知

- 范围声明中含一条外部来源引用，该引用已在文档内部**标注为待核实**；
  本说明中未将其用作承重论据。
- 审计的统计口径（稳定窗口聚合、观测单位）在 `docs/scenario_audit.md` §1 中声明，
  故新数字与 v1.0.0 的表格可直接比较。
- 本版本新增的内容全部是文档、分析脚本与原始输出。冻结的环境文件未改动。

**完整变更日志**：`git log --oneline v1.0.0..v1.1.0`
