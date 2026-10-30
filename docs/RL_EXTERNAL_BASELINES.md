# 外部爬行 baseline 调研（2026-10-15）

本轮仅搜索官方项目页、公开仓库和配置；未安装依赖、运行外部权重或启动训练。
将“有源码”“有可取得的推理权重”“已本机复现”分开；下面没有任何项目被标记为本机复现成功。

**后续用户决定：以改装 g1_crawl 为主要参考，而非 APEX。** 下文保留搜索时的候选排序，
不代表当前执行优先级。最新只读兼容性核对见 `RL_G1_CRAWL_REFERENCE.md`。

## 首选核查：APEX-humanoid

- [官方仓库](https://github.com/apex-humanoid/APEX-humanoid)
- [论文项目页](https://apex-humanoid.github.io/)
- 核查 tree：`e606ff7186099e824a62d906688e48173a5ca66e`。
- README 明确列出 `Isaac-G1-Velocity-Crawl` 与 `checkpoints/crawl.pt`，另有 walk、
  climb-up/down、stand-up、lie-down。官方文档要求 Linux、NVIDIA GPU、Isaac Lab 1.3.0、
  Isaac Sim 4.2.0，不能把原生训练安装直接搬到当前 M4 Mac。
- GitHub tree 显示 crawl.pt 为 1,890,034 bytes，读取开头为 ZIP 格式 `PK`，不是 LFS 文本指针。
  未下载完整模型、反序列化或推理；这只确认二进制文件可访问，不证明权重匹配或有效。
- README 说明发布的是推理策略，不是可直接续训的完整 checkpoint。
- [爬行环境配置](https://github.com/apex-humanoid/APEX-humanoid/blob/e606ff7186099e824a62d906688e48173a5ca66e/source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/manager_based/locomotion/velocity/config/g1/g1_crawl_cfg.py)
- [PPO 配置](https://github.com/apex-humanoid/APEX-humanoid/blob/e606ff7186099e824a62d906688e48173a5ca66e/source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/manager_based/locomotion/velocity/config/g1/agents/crawl_ppo_cfg.py)
- [分阶段配置](https://github.com/apex-humanoid/APEX-humanoid/blob/e606ff7186099e824a62d906688e48173a5ca66e/source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/manager_based/locomotion/velocity/config/g1/g1_crawl_stages_cfg.py)
- 当前源码包含分阶段 reset/reward、经验观测归一化、对称数据增强，以及 512/256/128 网络。
  这些与我们的小网络/小预算方案不同；不能只拷贝一个奖励权重来称为复现。
- 默认 4096 环境 ×24 steps ×8000 iterations =786432000 transitions。
  我们每轮 32×32×300=307200，仅为该默认预算的 1/2560。
  这是配置预算比较，不是作者实际 checkpoint 训练日志或相同样本效率比较。
- 论文使用 29-DoF G1；当前项目为 23 执行器，必须核对实际动作选择、关节映射、
  观测历史/归一化、PD 增益、碰撞模型等，不能直接加载进现有控制接口。

## 最贴近现有模型的参照：jloganolson/g1_crawl

- [原始仓库](https://github.com/jloganolson/g1_crawl)
- [README](https://github.com/jloganolson/g1_crawl/blob/main/README.md)
- 与本地 `/Users/loopy/Desktop/g1_ghost/g1_crawl` remote 相同；远程 tree 仍为
  `935e8b9c857bd494b952603c0453800a0cf5c9d2`，不是发现了另一个独立 baseline。
- 作者明确说明最终策略对应的奖励项记忆不完整，尤其最终 crawl；因此适合做行为/部署参照，
  不应声称按当前训练配置即可严格重训出相同结果。
- `sim2sim_mj/policies/policy_crawl.pt` 等 Git tree 条目为 131 bytes，与本地已知 LFS 指针一致。
  本轮未取 LFS 权重；“用户本地没有完整策略”不等于“上游从未发布策略”。
- [lez666-g1-sim2crawl](https://github.com/lez666/lez666-g1-sim2crawl) 是其派生项目，
  README 声称增强键盘/MuJoCo 回放；不是独立训练算法的交叉证据，权重与模型匹配尚未验证。

## 后续复杂地形方法参考：ToddlerBot / Locomotion Beyond Feet

- [官方代码](https://github.com/hshi74/toddlerbot)
- [项目页](https://locomotion-beyond-feet.github.io/)
- 覆盖爬行、障碍穿越和上下楼梯，以物理约束关键帧参考、RL 运动跟踪及技能选择组织任务。
  README 提供训练入口及 checkpoint 下载链接，也给出 macOS MuJoCo 动作回放命令。
- 机器人不是 G1，权重不能迁移到当前 23 执行器模型；适合作为方法参照，不是即插即用 G1 策略。
  本轮未验证下载链接中的各技能文件或 Mac 上完整训练支持。

## 对当前计划的修正

1. 暂缓继续盲调本地奖励。先以 APEX crawl 为外部候选，核对模型/观测/动作/归一化/控制频率契约。
2. 优先在作者原始环境复现，或明确标记为需要验证的 MuJoCo sim2sim 适配；先不改成 23 关节。
3. 权重回放成功后才建立性能基线，再决定移植训练任务到 UniLab，或在 Linux/NVIDIA 上保留原训练环境。
4. 外部基线的任务定义、允许接触部位可能与我们的手脚专用爬行不同，必须实测核对，不能只看视频。
5. 现有 300 轮实验继续保留为小预算诊断结果，不再用它们否定算法可行性。
   预算差距并不消除已确认的数据链路错误，也不保证单纯加预算就能解决问题。
