# Autoclip

Autoclip 逆向分析一个已经剪好的 MV：判断每帧来自哪个源视频、哪个时刻，生成支持多素材、变速、定格和帧混合的 FCPXML 时间线，并可渲染重建视频进行检查。

程序保持五步、函数式结构。每一步写出中间文件，可以单独运行、检查或人工修订；不调用大模型。新代码不要求与旧脚本输出一致，旧 Golden、PNG 兼容流水线和固定偏移算法已经删除。

## 安装与配置

需要 Python 3.12、FFmpeg 和 ffprobe。环境文件只声明直接依赖，不再绑定某台 Mac 的二进制构建或本地路径。

```bash
conda env create -f environment.yml
conda activate autoclip
cp config.example.toml config.toml
# 大素材库可选 Faiss；安装后设置 matcher = "faiss"
python -m pip install -r requirements-optimized.txt
```

编辑 `config.toml` 中的 `paths.edited_video` 和 `paths.source_dir`。source 可包含多个视频和子目录，支持 mp4、mov、mkv、avi、webm、flv。文件 stem 必须唯一，且不能与目标 MV 重名。目标 MV 本身会从 source 搜索中排除。

所有配置见 [config.example.toml](config.example.toml)。默认使用紧凑缓存、NumPy 精确检索、像素复核、逐帧细化和短跳转复核。参考图、遮罩、LUT 都可省略；`frame_rate=0` 从目标视频读取帧率。

路径规则：`working_dir` 相对于 TOML 所在目录，其余路径相对于 working_dir。未知配置项明确报错。本地 config.toml 不提交 Git。

从旧配置迁移：删除 Step 3 的 `reconstructed_video`、`stitched_images`、`visuals_dir`；不再支持 `frame_storage="png"`、`matcher="legacy"`、`alignment="legacy"`。只保留一个通用配置示例，不再区分普通版和优化版。旧 CSV 应重新运行 Step 4，不会自动套用旧版等速解释。

## 运行

```bash
python main.py all --config config.toml
# 调参后只重跑某一步
python main.py step4 --config config.toml
python main.py step5 --config config.toml
# 依据最终片段表生成视频，而不是使用粗匹配的第一候选
python render_reconstruction.py --config config.toml \
  --output output/reconstructed.mp4 --comparison output/comparison.mp4 \
  --preset veryfast --threads 4
```

对照视频上方是目标 MV，下方是重建结果。渲染参数只改变编码速度，不改变选帧。运行不同方案时使用不同的输出目录或片段 CSV 文件名，避免覆盖需要比较的结果。

## Step 1：预处理与紧凑提帧

1. 可选参考图通过 SIFT（不可用时 ORB）、特征匹配和 RANSAC 单应性估计源画面的裁剪/缩放范围。
2. 源视频依次裁剪、缩放到目标分辨率、应用可选 LUT；所有视频再应用目标画面坐标系中的遮罩。没有参考图时，不同尺寸 source 仍先统一尺寸再遮罩。
3. FFmpeg 读取全部视频帧并缩为 64×64 灰度数组，不做稀疏抽帧。
4. ffprobe 提供真实逐帧时间戳，保存为每个视频目录下的 `compact_frames.npy` 和 `compact_frames.json`。

缓存核对素材路径、大小、修改时间、预处理参数、LUT 内容及 FFmpeg 路径/版本/构建标识。未完成的写入不可复用。升级工具链后应从 Step 1 继续运行至 Step 4，不要混用旧像素与新解码结果作验收。

## Step 2：256 位 pHash

对灰度数组批量做二维 DCT，取左上 16×16 系数，用中位数阈值生成 256 位哈希。按视频输出 `<stem>_phash.csv`，只包含视频名、帧号、毫秒时间戳和哈希；不再生成虚构的 PNG 文件名或空图片路径。

只处理当前输入素材的缓存，不扫描输出目录里的旧素材或备份。哈希缓存绑定紧凑缓存清单的摘要，变化时重新计算。

## Step 3：批量候选检索

在所有当前 source 的哈希中按 Hamming 距离检索每个目标帧的 top-k。默认 k=20；可选 numpy、balltree_batch、faiss 三种精确后端，以及实验性的 faiss_hnsw 近似检索。不同后端的等距候选顺序可能不同，不要求复现旧 BallTree 的顺序。

输出 `coarse_match_results2.csv`，保留每帧候选的素材名、帧号、真实时间和距离。候选用于下一步，并不是最终选帧；旧版“取第一候选拼视频”的入口已删除。

## Step 4：时间对齐、逐帧细化与短跳转复核

在多个时间窗口内寻找候选对应关系的共识，提出分段模型：

```text
source_time = speed × edited_time + offset
```

带切换代价的路径选择决定连续片段；`affine_pixels` 再用灰度画面验证，像素候选补救纯色、定格等 pHash 不可靠情形，并拒绝证据不足的对应关系。没有可靠匹配的范围保留为 gap，不强行补齐。

默认接着做逐帧细化和短跳转复核。可关闭这些步骤进行消融实验，或使用 affine / affine_ransac 比较初始对齐方法；这些是现有算法的实验开关，不是旧兼容实现。

最终表包含素材名、目标帧区间、speed、source_start_time_ms、排他终点 source_end_time_ms、edited_total_frames，以及可选 time_map 和 frame_sampling。time_map 有值时优先于 speed；以目标片段内帧偏移为横坐标、源时间毫秒为纵坐标，最后一点必须位于片段排他终点。

### 逐帧校正如何工作

1. 先保留已经识别的 source 和剪辑边界，在每帧的初始源位置前后各搜索 `refine_radius` 帧。
2. 用 64×64 灰度像素的平均绝对误差评价候选。动态规划要求源时间不倒退，并以小的偏移/步长惩罚维持连续性，减少重复画面中的跳动。
3. 自动混合检测对相邻源帧拟合 `目标 ≈ (1−α)×前帧 + α×后帧`，限制 `0≤α≤1`。混合必须比最好的单帧改善超过 1 个灰度级、20%，且 `α` 在 0.05～0.95 内；自动模式还要求一个最多 60 帧的局部窗口中至少有 3 帧、5% 的帧支持这一解释，避免长静止段稀释短暂的混合证据。单帧候选始终保留。
4. 将选帧路径压缩为分段线性时间点。单帧采样必须逐帧保持原选帧身份；混合采样的曲线误差不超过源帧间隔的 0.2%，纯帧保持在源时间戳上。
5. FCPXML 分别使用 `floor` 或 `frame-blending`；音乐挂在不变速的容器上。源素材采用自己的帧率、尺寸和可读取的真实时长。帧率归一化区分近似 60 fps 与 59.94 fps。

这里的相邻帧混合是帧率转换的一种采样方式，不是跨镜头叠化，也不是光流合成。格式依据是 Apple 的 [timeMap](https://developer.apple.com/documentation/professional-video-applications/timemap) 和 [Frame Sampling](https://developer.apple.com/documentation/professional-video-applications/frame-sampling) 文档；编辑器实际导入效果仍需在目标软件中验证。

### 中间结果与人工调试

#### 短暂跳转复核

`short_jump_review.py` 是逐帧细化之后的独立函数步骤，不调用大模型。它审查“正常片段 → 短暂远跳 → 回到原 source”这种结构，不使用素材文件名、已知切点或案例帧号。只有两侧目标帧相邻、同源、状态为 `verified`，而且源时间能够正向接上时，才提出连续替代方案；不会填补未知 gap，也不处理缺少双侧锚点的首尾片段。

连续方案由两个锚点插值给出先验，在两侧各最多 0.5 秒的上下文中复用已有局部动态规划和帧混合检测。远跳距离必须超过 `max(0.25 秒, (refine_radius+1)×源帧间隔)`，或中间片段来自另一 source。仅短片段内的帧可以改变，两侧原有选帧及混合比例保持不变。

接受修复需同时满足：

- 连续方案仍在双侧源时间锚点之间，每帧误差不超过 `review_mae`。
- 相比原方案，平均 MAE 最多增加 1 灰度级，任何单帧最多增加 2 灰度级；不能用长片段平均值掩盖一个坏帧。
- 至少一半帧属于低信息画面，或者连续方案平均误差改善超过 1 灰度级且超过 20%。有明确画面证据的单帧/三帧闪切不会因时长短而删除。

修复后合并这三个片段，但保留逐帧时间曲线而非强行拟合一个等速片段。修复区间仍标为 `ambiguous`：这是有证据约束的连续性优先选择，不是原工程出处的证明。若锚点不可靠，或细节画面没有足够的更改依据，则保留原映射并标记待检查。所有审查、保留和修复理由均输出，不会悄悄抹平。

假设最终文件名为 `final_video_segments_refined.csv`，启用逐帧校正后会额外输出：

- `final_video_segments_refined.affine.csv`：局部校正前的片段，便于比较倍率和边界。
- `final_video_segments_refined.frames.csv`：每帧原始位置、校正后位置、混合比例、像素误差和状态。
- `final_video_segments_refined.audit.json`：覆盖数、误差分位数、混合帧数、逐片段统计。
- `final_video_segments_refined.review.jpg`：残余误差最大的帧；依次为目标、初始映射、最终重建小图。

启用短跳转复核后，还会输出 `.pre_jump.csv`、`.pre_jump.frames.csv`、`.pre_jump.audit.json` 保存该步骤之前的完整结果；`.jump_review.json` 记录每次审查的区间、原映射、连续候选、逐帧误差、采用/保留理由和使用的阈值。最终 `.frames.csv` 中的 `jump_review_id` 对应这份日志，`jump_review_action` 为 `bridged`、`retained`、`needs_review` 或空。`affine_original_video_name` 保留初始来源；跨 source 修复时 `frame_adjustment` 留空，因为两个视频的帧号差没有意义。原版本产物不会因为启用功能而自动备份，比较不同运行请使用不同的 `final_segments_csv` 和视频输出路径。

状态为 `verified`（像素阈值内）、`review`（建议检查）、`ambiguous`（来源存在歧义，包括低信息帧及短跳转复核结果）或 `unmatched`（没有已接纳的映射）。小图指标用于定位问题，最终仍应看重建视频。不要把“每帧都有来源”当成“每帧都正确”。

调试时先检查遮罩、参考图、LUT 和 source 范围；不同分辨率 source 的遮罩现在会统一到 MV 尺寸后应用。检查 `.affine.csv` 判断粗定位是否正确，再检查 `.frames.csv` 的误差是否集中在切点、混合帧或持续漂移。必要时调整上述少量配置或手工修订最终 CSV，然后只重跑 Step 5 和渲染器。有 `time_map` 时只修改 `speed` 不会改变结果，应直接修改时间点；修改片段长度时也要同步调整其终点。不要把某个 MV 的帧号、时间点或片段倍率写进算法。


## Step 5：导出可编辑时间线

每个源视频使用实际路径、自己的帧率和尺寸；资产时长优先读取媒体元数据，缺失时使用已引用源时间范围的上界，不再写固定两小时占位值。

每个片段的 offset / duration 使用目标帧率计算；timeMap 使用源视频时间。没有逐帧曲线的 affine 实验结果使用两个时间端点。未匹配范围输出 gap。目标 MV 音频单独引用，挂在不变速的容器上，覆盖完整时间线。要求保留音频却没有音轨时明确报错。

渲染器、逐帧检查和 FCPXML 共用时间映射与 floor / frame-blending 采样规则。不能表示任意倒放、跨镜头转场或光流补帧；纯色、静态、多集重复画面仍可能无法唯一定位。confidence 不是校准概率，verified 只表示通过像素阈值，不是出处真值证明。

尚未在剪辑器中完成所有变速情形的实际导入验证。FCPXML 可用于 DaVinci Resolve / Final Cut Pro 工作流，不承诺 Premiere 可直接无损导入。

## 测试与实验

单元测试无需用户视频（部分测试使用 FFmpeg 生成极短素材）：

```bash
python -m unittest discover -s tests -p 'test_*.py' -v
```

### 变速实验

明确提供一个有音轨的视频，建议长度至少 30 秒。生成器按固定种子在实际源帧范围内采样剪辑位置，不再绑定最初那部 MV 的哈希或 300 秒后的固定帧号。

```bash
python tests/build_retime_fixture.py --input source/sora_2nd_op.webm \
  --destination tests/.work/retime_current --seed 91847
python tests/benchmark_retime.py --work-dir tests/.work/retime_current \
  --case development --matcher numpy --top-k 20
python tests/benchmark_retime.py --work-dir tests/.work/retime_current \
  --case holdout --matcher numpy --variants refined reviewed
```

基准默认比较 affine、affine_ransac、affine_pixels、refined、reviewed，分别记录匹配、细化和导出耗时及真值误差。可以设置多个 top-k；大库可选择 faiss。每个方案同时导出 FCPXML。真值只交给评价器，不交给匹配算法。

每次实验都会调用 Step 1 / 2 的当前缓存校验，不再依赖旧 preparation.json 跳过校验；耗时是本次实际耗时，可能包含缓存命中，不能冒充冷启动性能。检索微基准默认重复三次，可用 --skip-retrieval-benchmark 跳过。已删除旧备份加载、--baseline-dir 和 --include-png。

### 多素材泛化实验

输入 first 至少 125 秒，second 至少 60 秒。包含不同尺寸与帧率、水印、曝光变化、变速、重复帧、定格、独立短镜头和 24 帧无关画面。

```bash
python tests/build_general_fixture.py \
  --first tests/.work/retime_current/sources/reference.mp4 \
  --second source/sora_2nd_op.webm --destination tests/.work/general_current --seed 81591
python main.py all --config tests/.work/general_current/config.toml
python tests/build_general_fixture.py --destination tests/.work/general_current --evaluate
```

默认开启短跳转复核；加 --no-jump-review 并使用另一个 destination 可生成关闭复核的对照。评价同时核对源素材身份、帧号、意外帧混合以及未知画面的误接纳；--segments 和 --output-dir 可指定其他方案的结果及缓存目录。

两种生成器均记录输入 SHA-256、生成参数及工具链；只有身份一致且产物完整时才复用。修改输入、参数、工具链或缺失产物时要求新的 destination 或显式 --force。强制重建会覆盖该实验目录的生成文件，不影响原始素材。development / holdout / validation 是可重复的剪辑划分，不等同于从未参与调试的独立数据集。

### 实际视频验收

```bash
python tests/audit_render.py --config config.toml \
  --video output/reconstructed.mp4 --comparison output/comparison.mp4 \
  --output output/render-audit.json
```

直接重新解码目标与输出，核对帧数、音频包哈希，报告灰度 MAE 分位数。像素指标用于诊断，不能单独证明源帧身份准确。

素材、缓存、视频及实验结果保存在被 Git 忽略的 source/、output/、tests/.work/。旧测试流程已删除，不再迁移或执行旧版兼容检查。仓库保留可重复运行的实验工具和使用说明，不将一次性的清理验证报告作为项目文档提交。
