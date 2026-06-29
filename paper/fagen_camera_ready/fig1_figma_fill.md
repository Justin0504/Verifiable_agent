# Figure 1 Figma 填空表

每个 box 的精确文字。复制 → 粘贴 → 调字号即可。

---

## 顶部 Band: 1. Structured Verification

### ① Claim box (蓝紫底)
- **Title (Bold 14pt)**: `Claim`
- **Body (Regular 11pt, italic, quoted)**:
  ```
  "60% of participants
   significantly improved"
  ```

### ② Source box (绿底)
- **Title**: `Source`
- **Body**:
  ```
  "60% of subjects
   showed improvement"
  ```
- **Micro-caption (9pt italic gray, 可选)**: `ANLI-derived · paraphrased`

### ③ SEVA Verifier (中间节点)
- **Title (Bold 18pt)**: `SEVA Verifier`
- **Subtitle (Regular 11pt)**: `Qwen2.5-3B · Process Reward GRPO`
- **Footer micro-text (9pt italic gray)**: `350 training steps · 5K structured samples`
- ⚠️ **删除当前的卡通人 + 放大镜 icon**,只留文字

### ④ Evidence Alignment sub-card (mint 绿底)
- **Title (Bold 11pt)**: `Evidence Alignment`
- **Body (Mono 9pt)**:
  ```
  ✓  "60% of participants"  →  "60% of subjects"   [match]
  ✗  "significantly improved"  →  NOT_FOUND        [not_found]
  ```
- **Micro-caption (8pt italic gray)**: `per-span grounding · 3 status values`

### ⑤ Reasoning Chain sub-card (ice 蓝底)
- **Title**: `Reasoning Chain`
- **Body (Mono 9pt)**:
  ```
  Step 1:  "60% matches"  →  supported   ✓
  Step 2:  "significantly" has no source  →  not_supported  ✗
  ```
- **Micro-caption**: `step-by-step · evidence cited at each step`

### ⑥ Label + Confidence sub-card (cream 黄底)
- **Title**: `Label + Confidence`
- **Body**:
  ```
  Not Attributable        γ = 0.85
  ```
- **进度条**:0–1 区间,填 85%
- **Micro-caption**: `binary label · calibrated confidence`

### ⑦ Error Diagnosis sub-card (peach 粉底)
- **Title**: `Error Diagnosis`
- **Body (Mono 9pt)**:
  ```
  type:  scope_inflation
  fix:   remove "significantly"
  ```
- **Micro-caption**: `6-category taxonomy · actionable fix`

---

## 跨 band 箭头标签

### ⑧ 左侧虚线箭头 (上 → 下,从 Error Diagnosis 出)
- **Label (10pt italic)**:
  ```
  structured errors reveal WHY,
  not just THAT, the model fails
  ```

### ⑨ 右侧虚线箭头 (下 → 上,从 Refine 回到 SEVA Verifier)
- **Label (10pt italic)**:
  ```
  refined verifier
  +15pp HaluEval / −12pp TruthfulQA
  ```

---

## 底部 Band: 2. Self-Evolution Loop

每个节点 = icon (保留几何 icon,不换) + Title + Subtitle

### ⑩ Verify (lavender 紫底)
- **Title (Bold 13pt)**: `Verify`
- **Subtitle (Regular 9.5pt)**:
  ```
  run on held-out claims;
  collect structured predictions
  ```

### ⑪ Reflect (cream 黄底)
- **Title**: `Reflect`
- **Subtitle**:
  ```
  build 6-bin
  weakness profile
  ```

### ⑫ Weakness profile mini-table (Reflect 右侧,小框)
- **Body (Mono 9pt,等宽对齐)**:
  ```
  entity_sub:        42% acc     ← weakest (probe target)
  tokens_brev:       60% acc
  scope_inflation:   61% acc
  fabrication:       78% acc     ← strongest
  ```
- ⚠️ **修正**: 当前 `accfabrication` 是 typo,改成 `fabrication`
- ⚠️ **保留 entity_sub 那行的红色高亮**,其余灰青色
- ⚠️ **`← weakest (probe target)` 和 `← strongest` 用 9pt italic,deep teal**

### ⑬ Probe (peach 粉底)
- **Title**: `Probe`
- **Subtitle**:
  ```
  generate adversarial samples
  ∝ per-category weakness
  ```

### ⑭ Refine (mint 绿底)
- **Title**: `Refine`
- **Subtitle**:
  ```
  fine-tune verifier with
  GRPO + process reward
  ```

### ⑮ Refine → Verify loopback 箭头
- **Mid-arrow label (10pt italic)**: `iterate (× 4 rounds)`
- **Sub-caption (loop 下方,9pt italic gray)**:
  ```
  R2: 1.1K probes  ·  R3: 2.0K  ·  R4: 7.8K (mega-FT)
  ```

---

## 4 条整体 polish 建议

1. **删除 SEVA Verifier 那个卡通人 + 放大镜 icon** —— 这是图里唯一剩下的"Canva 味",其余都很干净
2. **底部 loop 节点的 icon (checkmark / triangle / screwdriver / refresh) 保留** —— 几何 monoline icon 符合 best paper 风格(类似 DeepSeek/OpenAI 系)
3. **Reasoning Chain 的 ✓ / ✗ 颜色** —— 如果当前是红/绿,改成统一 teal(`#0F4C4C`)+ 单一 accent green(`#22C55E`),避免多色冲突
4. **Weakness profile mini-table 字体** —— 用 monospace(SF Mono / JetBrains Mono / Menlo),否则 % 数字对不齐

---

## 颜色 / 字号速查

| 元素 | 字号 | 字重 | 颜色 |
|---|---|---|---|
| Main title (SEVA Verifier) | 18pt | Bold | `#0F4C4C` |
| Box title | 14pt | Bold | `#0F4C4C` |
| Sub-card title | 11pt | Bold | `#0F4C4C` |
| Body text | 10–11pt | Regular | `#0F4C4C` |
| Mono body (Evidence/Reasoning/Diagnosis) | 9pt | Regular | `#4B7878` |
| Micro-caption | 9pt | Italic | `#4B7878` |
| Arrow label | 10pt | Italic | `#0F4C4C` (绿色 accent 用 `#22C55E`) |
| Subtitle 下方括号说明 | 9pt | Italic | `#4B7878` |

| 用途 | Hex |
|---|---|
| Primary teal (border + text) | `#0F4C4C` |
| Muted teal (sub-text) | `#4B7878` |
| Green accent (feedback arrow) | `#22C55E` |
| Mint fill (Evidence / Refine) | `#ECFDF5` |
| Ice fill (Reasoning) | `#EFF6FF` |
| Cream fill (Label+Conf / Reflect) | `#FFFBEB` |
| Peach fill (Error / Probe) | `#FEF2F2` |
| Lavender fill (Verify) | `#F5F3FF` |
| White | `#FFFFFF` |

---

## 填完之后

截图整个 Figma frame 发我,我会做最终 audit:每个 box 的位置、字号、颜色、对齐,告诉你哪里再挪几 px。
