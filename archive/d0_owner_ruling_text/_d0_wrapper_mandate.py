"""Add the owner's wrapper hard-constraint to plan_D0_scenario_audit_v2.md,
plus the verified Step-0 evidence and the regime-extension fix experiment
(the closing act of D0 per the owner's ruling).
"""
import io

P = "/home/zy/car_rl/code0919/plan_D0_scenario_audit_v2.md"
with io.open(P, encoding="utf-8") as f:
    t = f.read()

# --- replace the section 1.1 conclusion with the wrapper mandate ---
OLD = """**⇒ 铁律 2 的正确用法不是"重训"而是"改常量 + 只重评"**。
改 `D_DES` 后须更新 sha256 表（`DATA_MANAGEMENT.md` §8），
但因 **gap 轴零样本成立**，**不触发重训**——评审已实测证明。"""

NEW = """**⇒ 铁律 2 的正确用法：走 WRAPPER 路径，`follow_env.py` 一个字节都不许动。**

> **裁决（项目负责人，2026-09-22）**：评审证明的是"**checkpoint 零样本跨间距**"，
> 而**不是**"冻结文件可改"。故实现走 **wrapper**（`RobustnessWrapper` 先例），
> 在 reset 时于运行期覆盖模块常量 `follow_env.D_DES`，**不改文件、changing sha256**。

**Step 0 已验证（本 plan 执行的第 0 步，实测）**：

```
  follow_env.py sha256[:16] before: 941c447b969c2fa1
  ... 跑完 d_des ∈ {0.20, 0.50, 1.00} 评估 ...
  follow_env.py sha256[:16] after:  941c447b969c2fa1   ← 未变
  frozen file untouched: True

  实测（wrapper 路径，20 ep）：
    d=0.20 v=0.30   mean|e| 2.73 true-cm   零碰撞
    d=0.50 v=0.55   mean|e| 2.69 true-cm   零碰撞
    d=0.50 v=1.00   mean|e| 5.91 true-cm   零碰撞   ← 反转格，复现
    d=1.00 v=1.00   mean|e| 4.59 true-cm   零碰撞
```

**⇒ wrapper 路径可行且不动冻结文件**；**跨间距零样本成立**。

> **★ 该"跨间距零样本成立"本身是一项 D0 正面发现**（负责人指定记录）：
> 同一控制器**适配不同车队间距规范（0.2/0.5/1.0 m）无需逐个重训**，
> 这是 **AGV 部署灵活性的直接论据**（现场通常按安全规范配置跟随间距）。"""

if OLD not in t:
    print("WARN: section 1.1 block not found")
else:
    t = t.replace(OLD, NEW, 1)
    print("1. OK: wrapper mandate + Step-0 evidence added")

# --- append the regime-extension fix experiment as the closing act ---
t += """

---

## 8. D0 收官：regime 扩展修复实验（**新增，负责人裁决**）

### 8.1 为什么需要它

D0 的审计发现（§1.2）只到"**反转存在、机理清楚**"为止。若停在这里，
D0 的故事是"我们发现自己的头条不成立"——诚实，但**不完整**。
真正的解药是**把训练分布扩到目标 regime**，并**验证优势在 AGV 区恢复**。

**完整故事**（负责人指定）：
> 审计发现问题 → 定位机理（残差 regime-local）→ **扩展训练分布修复** →
> **工作包络图闭合**。

情景适配性由此从**一份声明**变成**一条完整的科学论证**——
这也是 Phase D 采购决策真正该依据的证据。

### 8.2 实验设计

```
变量：把 v_set 训练分布从 U(0.25, 0.50) 扩到 U(0.25, 1.50)（覆盖 AGV 典型区）
      （**wrapper 路径**：训练时 monkeypatch v_set 上界，不改 follow_env.py）
训练：复用 train_ppo.py 的配方（stage1 easy → stage2 full），**5M 步**
      —— 注意这**是**重训（改了训练分布），走铁律 4（更新次数 ≥300）
评估：**同一张 9 格工作包络图**，与旧 checkpoint 并列
验收：v=1.0 格子上 RL 重新优于 P+FF（或至少不劣），且低速档不退化
```

### 8.3 预登记的验收阈值（评测前定死）

| # | 判据 | 阈值 | 依据 |
|---|---|---|---|
| **D0.5** | AGV 区优势恢复 | v=1.0 格 RL `mean\\|e\\|` **≤ P+FF**（obs-cm） | 反转的直接修复 |
| **D0.6** | 低速档不退化 | v≤0.55 各格 RL `mean\\|e\\|` 不劣于**原 checkpoint**（允许 ≤10% 放宽） | 防"修好高速、弄坏低速" |
| **D0.7** | 包络图闭合 | 9 格中 **≥7 格** RL 不劣于 P+FF 且零碰撞 | 工作包络 |

### 8.4 诚实条款（防"不能输判据"，承接 §2.4）

- **D0.5 若失败**（扩展分布仍不能恢复优势）：**如实报告**，
  并给出"RL 适用速度上界"的**量化结论**——这本身是有价值的负结果，
  **不判"修复失败"而是判"上界已刻画"**；
- 全部阈值**进 `research/ASSUMPTIONS.md` 预登记**，跑完不改。

### 8.5 与论文 v3 的关系

> 摘要与所有 headline 数字**同步加速度/区间限定**（W1 鲁棒性教训的同构应用）。
> "先验结构负责外推、RL 残差是 regime-local"这句话值一个**讨论段**。

"""

with io.open(P, "w", encoding="utf-8") as f:
    f.write(t)
print("2. OK: regime-extension fix experiment (section 8) appended")
