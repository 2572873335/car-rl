# archive/d0_owner_ruling_text — D0 owner 裁决的原始文本（从 /tmp 抢救）

**为何在此**：这些文件原只存在于 `/tmp`（随重启蒸发），违反 §5b / F19 落点通则。
其中 `_d0_wrapper_mandate.py` 是**owner 裁决的唯一原始文本**——它本该把
「wrapper 硬约束 + §8 regime-extension」写进 `plan_D0_scenario_audit_v2.md`，
但**从未成功落地**，导致提交 `c0b7b7d` 的提交信息描述了一处不存在的 plan 修改。

**抢救日期**：2026-09-22
**抢救依据**：`AGENT_HANDOFF.md` §5b（凡被 commit / 文档 / 勘误引用的文件，必须在仓库内有落点）

---

## 清单

| 文件 | 内容 | 与当前文档的关系 |
|---|---|---|
| `_d0_wrapper_mandate.py` | **owner 裁决原始文本**：wrapper 硬约束（运行时覆写 `follow_env.D_DES`，冻结文件逐字节不动）+ Step 0 实测证据 + §8 regime-extension（5M 步、U(0.25,1.50)、D0.5–D0.7 真阈值） | 已被 `plan_D0_scenario_audit_v3.md` §1.1/§8 **正式吸收**；此文件为原始出处 |
| `_probe_d0_stageA.py` | 阶段 A 探针（runtime D_DES override，测 terminal gap 是否跟随 d_des） | 其问题已被 `results/20260922_D0_scenario_audit_scripts/_d0_step0_wrapper_probe.py` 取代（更完整） |
| `_commit_d0v2.txt` | `7a30995`（plan v2）的提交信息草稿 | 历史留档 |
| `_d0_wrapper_probe_tmpcopy.py` | `_d0_step0_wrapper_probe.py` 的 /tmp 副本 | **注意**：仓库内的 `results/20260922_D0_scenario_audit_scripts/_d0_step0_wrapper_probe.py` 为权威版本，此副本仅证明 /tmp 双副本现象存在 |

---

## 关键教训（对应已登记的两个坑）

1. **「写完的文档提交后被旧副本覆盖」**：编辑脚本在 `/tmp`，目标文件在仓库，
   两边副本未同步 ⇒ 编辑静默丢失，而提交信息仍按**意图**书写。
2. **「改一处代码，另一份没改」**：`_d0_step0_wrapper_probe.py` 存在仓库 + `/tmp` 双副本。

**通则**：核对提交是否真做了它声称的事，用 `git show --stat` + 对目标文件 grep
关键词，**不要信提交信息**。

---

## 未找到项

`_d0_review3`（`reviews/20260922_D0_review1.md` 的 R3/R4 段引用，给出 3.99/7.98 等数字）
在仓库与 `/tmp` 中**均无落点**，疑随重启蒸发。其作用已由
`results/20260922_D0_scenario_audit_scripts/_d0_reconcile_conventions.py` **替代**
（后者口径声明更明确：同时输出 whole 与 settled）。
