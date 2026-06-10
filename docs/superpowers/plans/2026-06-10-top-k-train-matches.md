# Top-K Train Matches Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 为每个 eval episode 记录与之相似度最高的 5 个 train episode,并在 `test_openpi_jax` 流程中输出为 JSON 文件。

**Architecture:** 在已有的 `[N_eval, N_train]` 相似度矩阵上,对每行做 `torch.topk`,产出 JSON 可序列化的纯数据结构。计算逻辑放在 `_similarity.py` 的纯函数 `top_k_train_matches`,文件 I/O 由 `main.py` 的 `test_openpi_jax` 负责接线。

**Tech Stack:** Python 3.11, PyTorch, pytest, json (标准库)

**Spec:** `docs/superpowers/specs/2026-06-10-top-k-train-matches-design.md`

---

### Task 1: 纯函数 `top_k_train_matches`

**Files:**
- Modify: `src/scaling_curve/_similarity.py` (在文件末尾追加函数)
- Test: `src/scaling_curve/tests/test_scaling_curve.py` (在文件末尾追加测试)

- [ ] **Step 1: Write the failing tests**

在 `src/scaling_curve/tests/test_scaling_curve.py` 末尾追加:

```python
from src.scaling_curve._similarity import top_k_train_matches


def test_top_k_matches_orders_by_score_descending():
    # 2 eval episodes × 4 train episodes
    sim = torch.tensor([
        [0.1, 0.9, 0.3, 0.5],   # eval 0: best train = 1, then 3, then 2
        [0.8, 0.2, 0.7, 0.6],   # eval 1: best train = 0, then 2, then 3
    ])
    c_min, c_max = 0.0, 1.0  # 归一化为恒等变换,便于手算
    result = top_k_train_matches(sim, c_min, c_max, k=3)

    assert isinstance(result, list)
    assert [r["eval_id"] for r in result] == [0, 1]

    ev0 = result[0]["top_k"]
    assert [m["train_id"] for m in ev0] == [1, 3, 2]
    assert ev0[0]["score"] == pytest.approx(0.9)
    assert ev0[1]["score"] == pytest.approx(0.5)
    assert ev0[2]["score"] == pytest.approx(0.3)

    ev1 = result[1]["top_k"]
    assert [m["train_id"] for m in ev1] == [0, 2, 3]


def test_top_k_matches_clamps_k_to_n_train():
    sim = torch.tensor([[0.2, 0.5]])  # 1 eval × 2 train
    result = top_k_train_matches(sim, 0.0, 1.0, k=5)
    assert len(result[0]["top_k"]) == 2  # clamp 到 N_train=2
    assert [m["train_id"] for m in result[0]["top_k"]] == [1, 0]


def test_top_k_matches_normalizes_scores():
    sim = torch.tensor([[0.4, 0.6]])  # 1 eval × 2 train
    # 归一化: (raw - 0.4) / (0.6 - 0.4)  → 0.6→1.0, 0.4→0.0
    result = top_k_train_matches(sim, 0.4, 0.6, k=2)
    scores = [m["score"] for m in result[0]["top_k"]]
    assert scores[0] == pytest.approx(1.0)
    assert scores[1] == pytest.approx(0.0)


def test_top_k_matches_degenerate_range_returns_one():
    sim = torch.tensor([[0.5, 0.5, 0.5]])  # denom < 1e-8
    result = top_k_train_matches(sim, 0.5, 0.5, k=2)
    for m in result[0]["top_k"]:
        assert m["score"] == 1.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest src/scaling_curve/tests/test_scaling_curve.py -k top_k -v`
Expected: FAIL with `ImportError` / `cannot import name 'top_k_train_matches'`

- [ ] **Step 3: Write minimal implementation**

在 `src/scaling_curve/_similarity.py` 末尾追加:

```python
def top_k_train_matches(
    sim_matrix: torch.Tensor,
    c_min: float,
    c_max: float,
    k: int = 5,
) -> list[dict]:
    """对每个 eval episode 返回归一化分数最高的前 k 个 train episode。

    Args:
        sim_matrix: [N_eval, N_train] 来自 compute_sim_matrix()。
        c_min, c_max: 来自 sim_norm_range(),用于把分数归一化到 [0, 1]。
        k: 每个 eval episode 保留的 train 匹配数,自动 clamp 到 N_train。

    Returns:
        list[dict],每项形如
        {"eval_id": int, "top_k": [{"train_id": int, "score": float}, ...]}。
        外层按 eval_id 升序,内层 top_k 按 score 降序,长度为 min(k, N_train)。
    """
    n_train = sim_matrix.shape[1]
    k = min(k, n_train)
    denom = c_max - c_min

    top_vals, top_idx = torch.topk(sim_matrix, k, dim=1)  # 各 [N_eval, k]

    result: list[dict] = []
    for eval_id in range(sim_matrix.shape[0]):
        matches = []
        for rank in range(k):
            raw = top_vals[eval_id, rank].item()
            score = 1.0 if denom < 1e-8 else (raw - c_min) / denom
            matches.append(
                {"train_id": int(top_idx[eval_id, rank].item()), "score": score}
            )
        result.append({"eval_id": eval_id, "top_k": matches})
    return result
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest src/scaling_curve/tests/test_scaling_curve.py -k top_k -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Run full test suite (no regression)**

Run: `uv run pytest src/scaling_curve/tests/test_scaling_curve.py -v`
Expected: PASS (all existing + 4 new)

- [ ] **Step 6: Commit**

```bash
git add src/scaling_curve/_similarity.py src/scaling_curve/tests/test_scaling_curve.py
git commit -m "[feat] add top_k_train_matches pure function"
```

---

### Task 2: 接线到 `test_openpi_jax` 输出 JSON

**Files:**
- Modify: `main.py` (顶部 import + `test_openpi_jax` 内 Step 4 区域)

- [ ] **Step 1: Add imports**

在 `main.py` 顶部,把现有这行:

```python
from src.scaling_curve._similarity import compute_sim_matrix, policy_embedding_similarity, sim_norm_range
```

改为(追加 `top_k_train_matches`):

```python
from src.scaling_curve._similarity import compute_sim_matrix, policy_embedding_similarity, sim_norm_range, top_k_train_matches
```

并确认文件顶部已 `import json`;若没有,在 `import sys` 下一行添加:

```python
import json
```

- [ ] **Step 2: Wire the JSON output**

在 `test_openpi_jax` 中,定位到 per-eval 图保存之后这两行:

```python
    print(f"      Per-eval plot saved to {per_eval_path}")
    plt.close(fig0)
```

在 `plt.close(fig0)` 之后插入:

```python
    # --- Top-5 train matches per eval episode ---
    matches = top_k_train_matches(sim_matrix, c_min, c_max, k=5)
    matches_path = "openpi_jax_top5_matches.json"
    with open(matches_path, "w") as f:
        json.dump(matches, f, indent=2)
    print(f"      Top-5 train matches saved to {matches_path}")
```

- [ ] **Step 3: Smoke-test the import and function wiring (no model needed)**

Run:
```bash
uv run python -c "
import json, torch
from src.scaling_curve._similarity import top_k_train_matches
sim = torch.tensor([[0.1, 0.9, 0.3], [0.8, 0.2, 0.7]])
m = top_k_train_matches(sim, 0.0, 1.0, k=5)
print(json.dumps(m, indent=2))
"
```
Expected: 打印合法 JSON,eval 0 的 top_k 第一项 `train_id` 为 1,且每个 `top_k` 长度为 3(clamp 到 N_train=3)。

- [ ] **Step 4: Verify main.py imports cleanly**

Run: `uv run python -c "import main"`
Expected: 无报错(导入成功,不执行任何流程)。

- [ ] **Step 5: Commit**

```bash
git add main.py
git commit -m "[feat] wire top-5 train matches JSON into test_openpi_jax"
```

---

## Self-Review

**Spec coverage:**
- 纯函数 `top_k_train_matches` → Task 1 ✓
- 归一化到 [0,1] + denom 退化分支 → Task 1 Step 3 + `test_top_k_matches_normalizes_scores` / `test_top_k_matches_degenerate_range_returns_one` ✓
- k clamp 到 N_train → Task 1 Step 3 + `test_top_k_matches_clamps_k_to_n_train` ✓
- 按 score 降序、eval_id 升序 → `test_top_k_matches_orders_by_score_descending` ✓
- JSON 可序列化结构 → 纯 dict/list/int/float,Task 2 Step 3 smoke test 验证 `json.dumps` ✓
- 接线到 test_openpi_jax 输出 `openpi_jax_top5_matches.json` → Task 2 ✓
- YAGNI(不画图/CSV/不改 OOP 类) → 计划中未涉及 ✓

**Placeholder scan:** 无 TBD/TODO,所有代码步骤含完整代码。

**Type consistency:** `top_k_train_matches(sim_matrix, c_min, c_max, k)` 签名在 Task 1 定义、Task 2 调用一致;返回结构字段 `eval_id` / `top_k` / `train_id` / `score` 在测试、实现、smoke test 中命名一致。
