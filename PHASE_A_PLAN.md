# Locus Phase A Plan — Typst Formula Rendering + Teaching Timeline

> 目的：这是一份“可以直接接着开发”的 Phase A 技术总结与计划文档，覆盖项目目标、技术规划、已实现能力、已知问题与风险、以及后续里程碑拆解（含验收标准与验证方式）。

---

## 0. 背景与范围

### 0.1 Phase A 的核心目标
在 Phase A 阶段构建一个**自包含、可控、可观测**的数学排版与渲染工作流，用于教学演示（例如勾股定理证明的逐步展示）。重点强调：

- **Typst 公式渲染链路可控**：尤其是字体 provisioning，避免系统字体扫描导致的性能和不可预测性。
- **渲染输出可用于实时演示**：将 Typst 布局产物（glyph outlines / rules / shapes）转为 Scene2D draw items，并由 GPU 实时绘制。
- **动画 / Timeline 最小可用**：可驱动对象透明度与变换（位移/缩放/旋转）以实现 step-by-step 教学演示。
- **demo 与库分层**：示例快速迭代，稳定能力逐步回迁到 `src/` 形成可复用 API。

### 0.2 Phase A 的非目标（明确不做）
Phase A 不追求“完全通用排版系统”，以下暂不实现或只做 best-effort：

- 全量系统字体扫描与自动 fallback（只做显式白名单或 typst-assets 兜底）
- Typst 复杂 paint（渐变、图片、滤镜等）
- 复杂描边（stroke）对任意 path 的完整支持（当前主要是 `Geometry::Line` quad 近似）
- per-glyph 动画（只做 group/object 层级动画）
- GPU batching / atlas 等高级性能优化（Phase A 仅需可用与可观测）

---

## 1. 架构总览（分层）

整体数据流：

1. **Font provisioning**
2. **Typst 编译（math-only / demo snippets）**
3. **Frame/Item 提取（递归 group transform）**
4. **轮廓 / Path tessellation（lyon）**
5. **Scene2D 组装与 flatten（含 alpha 级联）**
6. **GPU 渲染（wgpu）**
7. **Timeline 驱动（每帧 apply）**

对应模块概览：

- `locus/src/font/*`：字体与 tessellation 工具（Phase A：主要用于 outline tessellate）
- `locus/src/typst/engine/*`：Typst 编译与世界构建（math-only）
- `locus/src/typst/render/*`：Typst → meshes / draw items 的提取与构建
- `locus/src/typst/demo/*`：demo glue（snippet → baseline-anchored group mobject）
- `locus/src/scene/*`：Scene2D / Camera2D / Mobject2D / flatten / bounds
- `locus/src/render/*`：wgpu app runner + mesh renderer（solid color pipeline）
- `locus/src/anim/*`：Timeline（alpha/translate/scale/rotate keyframes）

---

## 2. 字体加载策略（Phase A）

### 2.1 目标
- 默认使用 `typst-assets` 字体集，保证：
  - 渲染稳定（同一环境/机器一致）
  - 启动快（避免扫描系统字体）
  - 依赖可控（对齐 Typst 0.14 生态）

### 2.2 策略
- **默认**：仅加载 typst-assets
- **可选扩展（未来）**：
  - 显式白名单加载系统字体（例如 STIX/Libertine/Latin Modern）
  - 提供“系统字体禁用开关”（Phase A 默认禁用）

### 2.3 风险
- 数学符号覆盖可能不足（需收集缺字问题）
- Typst 版本升级可能引入依赖变动（ttf-parser / fontdb 等）

---

## 3. Typst 编译与引擎层（`typst::engine`）

### 3.1 目标
- 提供真实 `typst::layout::PagedDocument`（用于 frame traversal）
- 提供最小可用的编译入口，便于 demo 快速渲染一段公式/标签

### 3.2 当前能力
- `compile_math_only(snippet)`：编译一个 math-only snippet 到 `PagedDocument`
- 结构树日志工具（用于调试 frame/group/item 的嵌套与 transform）

### 3.3 后续规划
- 编译入口扩展：
  - 多页 / 文档级资源
  - 更丰富的诊断输出（source diagnostics 结构化化）
  - 可选加载自定义字体/资源

---

## 4. Typst → Scene 提取与 tessellation（`typst::render`）

### 4.1 目标
将 Typst 的 layout 输出转换为 renderer-friendly 的 mesh（三角形），并可保持基本颜色信息（Phase A：solid color）。

### 4.2 支持范围（Phase A）
- `FrameItem::Group`：递归 traversal，累积 transform
- `FrameItem::Shape`：
  - `Geometry::Line`：thin quad 近似 stroke
  - `Geometry::Rect`：lyon path tessellate
  - `Geometry::Curve`：Move/Line/Cubic/Close → lyon path tessellate
- `FrameItem::Text(TextItem)`：
  - 遍历 `TextItem.glyphs`
  - 通过 Typst Font 的 ttf face + `ttf-parser` 提取 glyph outline
  - outline → lyon::Path → tessellate → Mesh2D

### 4.3 缓存策略（性能关键点）
- `GlyphMeshCache`：缓存“按缩放生成、无平移”的 glyph mesh
  - key：`(glyph_id, sx_bits, sy_bits)`
  - 每次绘制只做平移 + 追加 indices（避免重复 tessellate）
- 未来可能扩展：
  - 更健壮的 cache key（hinting、变体、字体 face id）
  - shared tessellator / path reuse

### 4.4 输出形态
- `build_meshes_from_paged_document(...) -> ExtractedMeshes`（legacy：glyphs/shapes/lines 合并）
- `build_draw_items_from_paged_document(...) -> ExtractedDrawItems`（推荐：保留 per-item color）

### 4.5 限制与 TODO
- paint：目前只映射 `Paint::Solid`
- stroke：只对 `Geometry::Line` best-effort
- text color：目前 demo glue 多数采用统一 fill（后续可引入 per-item colors）

---

## 5. Scene Graph（`scene`）与渲染（`render`）

### 5.1 Scene2D 设计目标
- 坐标系：pt-space（与 Typst 输出一致）
- 明确 transform 组合顺序，便于排版与动画一致
- 可 flatten 到 draw items（renderer-agnostic）

### 5.2 关键结构
- `Affine2`：2D affine（3x3，column-major）
- `Mesh2D`：CPU mesh（positions + u16 indices）
- `Mobject2D`：node（name, transforms, fill, mesh, children）
- `Scene2D`：roots + name index + camera
- `Camera2D`：world(pt) → clip space

### 5.3 Alpha 传播（已实现）
flatten 时父节点 alpha 会乘到子节点 mesh 的 draw item 上（group fade 可用）。

### 5.4 Transform 分层（重要）
为支持“静态摆放 + 动画驱动”并避免 Timeline 覆盖初始位置，引入：

- `base_from_parent`：静态 transform（baseline anchoring、layout positioning）
- `anim_from_parent`：动画 transform（Timeline 写入）
- 渲染组合：`local = base * anim`
- 迁移期保留 compat 字段 `local_from_parent`（需要逐步清理）

> 经验教训：如果存在“多处字段代表 transform”的分叉，很容易导致对象跑飞或不可见。Phase A 中必须确保渲染与 bounds 使用同一权威组合逻辑。

### 5.5 GPU MeshRenderer（wgpu）
- solid-color pipeline（每个 item 一次 draw）
- uniform：`mvp + color`
- shader：`solid_mesh.wgsl`（mvp 变换 + 输出 color）

---

## 6. Timeline 动画系统（`anim`）

### 6.1 目标
最小可用、确定性强、便于编排教学 step：

- Alpha（fade in/out）
- TranslateX/Y
- Rotate
- Scale（uniform）

### 6.2 Track 语义（重要约定）
- 当前 demo 中 TranslateX/Y 的 keyframe 值通常按“**绝对位置**”写（例如 -40 → -20）
- Timeline 生成 `anim_from_parent = T * R * S`（绝对动画层语义）

### 6.3 当前限制
- 没有 per-glyph 动画
- 没有组合轨道的复杂约束（但 Phase A 教学演示足够）

---

## 7. Demo glue：baseline-anchored Typst group

### 7.1 目标
让 Typst 文本/公式易于“像图形对象一样”摆放与动画：

- group/root：baseline anchor（(0,0) 即 Typst baseline 原点）
- child mesh：glyph triangles

### 7.2 当前能力
- `compile_snippet_to_group_mobject_baseline(snippet, opts)`：
  - 编译 snippet
  - 提取 mesh（glyphs + optional shapes）
  - 作为 child mesh 挂到 group root

### 7.3 实战经验（重要）
- group alpha 动画必须能影响 child mesh（已通过 flatten alpha 传播解决）
- 静态 baseline 摆放必须不被 Timeline 覆盖（通过 base/anim split 解决；需完成迁移收敛）

---

## 8. 当前已实现的用例与效果

### 8.1 `examples/pythagoras_step1.rs`
目标：教学演示的最小闭环（几何 + 公式 + 动画节奏）。

现状能力：
- 右三角形（填充 mesh）
- labels：`label_a/b/c`（Typst glyph outline 渲染）
- 公式：`label_eq = a^2 + b^2 = c^2`（Typst glyph outline 渲染 + scale-in）
- 注意力引导：
  - 公式出现时 triangle 变暗（alpha 降到 0.30）
  - 斜边高亮：`edge_c` 淡入（stroke quad 近似）

---

## 9. 已知问题（必须优先解决）

### 9.1 渲染 debug geometry / blank screen 反复出现（P0）
现象总结：
- clear color 可改变（render pass + present OK）
- 但“本应可见的几何”偶发不可见（全灰或只有背景色）

可能根因分类：
1) **renderer draw path**（顶点/索引/clip-space/状态）导致 draw call 无片元
2) **transform/compat 分叉**导致物体画在镜头外
3) **camera/framing** 与 bounds 不一致
4) backend 差异（Wayland/EGL 重建日志出现过）

这类问题必须在 Phase A 尽快收敛，否则后续每个教学 step 都会被不确定性拖慢。

---

## 10. 后续里程碑计划（可直接照此开发）

> 约定：每个 milestone 尽量 1–3 个 commit，且有清晰验收标准。

### Milestone 0（P0）：渲染管线“必然可见”的 Debug Quad
目标：
- 启用 debug mode 时，必然可见“覆盖层”几何（区别于 clear color）。
- 从根源区分“GPU pipeline 不出片元” vs “scene/camera 数据问题”。

建议实现与验收：
- clear color 固定深灰
- debug quad 输出亮色（洋红/黄）
- 只要进入 draw call，就必须看到覆盖层
- 若仍不可见：
  - 在 draw_items 中打印一次“debug 分支是否触发”
  - shader 临时改成直接输出 `a_pos`（绕过 mvp/uniform）验证顶点输入

验收：
- debug 覆盖层稳定可见（100%）

### Milestone 1（P0）：完成 transform split 迁移收敛（消灭 compat 分叉）
目标：
- 所有渲染、bounds、camera framing 只使用 `base * anim`
- Timeline 只写 `anim_from_parent`
- 删除或封存 `local_from_parent`（不再被任何核心路径读取）

步骤：
1) grep 全仓库 `local_from_parent`：
   - layout/static placement：迁移到 `base_from_parent`
   - Timeline：迁移到 `anim_from_parent`
   - flatten/bounds：使用 `composed_local_from_parent()`
2) 删除为了迁移而加的同步 hack（如 add_root 同步 compat）
3) 逐 example 逐模块跑通

验收：
- `pythagoras_step1` 不再出现“偶发全灰/跑飞”
- camera framing 与对象位置一致

### Milestone 2（P1）：Step2 — 构造 a²/b² 两个正方形
目标：
- 在两条直角边外侧构造正方形（`square_a2`, `square_b2`）
- Timeline 分步淡入（或描边出现）

实现建议：
- 先用纯 mesh（rect mesh）+ transform 放置（最快验证几何正确性）
- 之后再决定是否用 typst shapes/rules 统一生成

验收：
- 正方形对齐边长、位置正确
- 动画节奏清晰

### Milestone 3（P1）：Step3 — 最小面积拼合演示（教学观感）
目标：
- 不是完整严谨证明，但能演示“面积块移动/旋转拼合 → 强调等式”。

实现建议：
- 将正方形划分为若干 tile（rect mesh）
- tile 作为 group，Timeline 驱动移动/旋转
- z-order 管理（避免遮挡）

验收：
- 观感清晰：出现块 → 移动拼合 → 强调 `a^2 + b^2 = c^2`

### Milestone 4（P2）：API 收敛与库化
目标：
- 将 demo 中稳定能力迁回 `src/` 并提供更稳定入口：
  - snippet → group mobject API
  - renderer debug utilities
  - 统计/观测接口

验收：
- example 代码更薄，库接口更清晰

---

## 11. 开发与验证方式（建议）

### 11.1 常用运行命令（避免阻塞）
- `timeout 10s env RUST_LOG=info cargo run --example pythagoras_step1`

### 11.2 观测点建议（长期保留）
- glyph calls / tess calls / triangles（Typst 提取统计）
- flatten item count
- cache hit/miss（后续可加）
- Debug overlay（baseline box）仅在开发模式启用

### 11.3 Debug 策略（强烈建议标准化）
1) 先验证 renderer debug quad 可见（Milestone 0）
2) 再验证 scene/camera（固定 camera sanity-check）
3) 再验证 bounds/framing（frame_bounds 只作为最后一步）

---

## 12. 当前状态快照（截至本文件生成）
- 已实现 Typst 编译、glyph outline 提取、tessellate、Scene2D 渲染、Timeline 动画、pythagoras_step1 教学 demo。
- 已实现 alpha 级联与 baseline-anchored Typst group。
- 正在进行 transform split 的迁移收敛，但仍存在“全灰/调试几何不可见”类 P0 问题需要优先解决（Milestone 0/1）。

---

## 13. 建议的下一步行动（最短路径）
按优先级推荐：

1) **Milestone 0：让 debug quad 100%可见**（确保 GPU draw path 没问题）
2) **Milestone 1：清理 compat transform 分叉**（确保 scene/camera 一致）
3) 回到教学内容：正方形与面积拼合步骤

---