# Transition v1 — 实验入口与前倾阶段验收

日期：2026-10-28。版本标识：`0.1.0-experimental`。

**已交付可运行的实验入口和通过验收的前倾准备段，但完整“站立→爬行”仍未完成。**
默认运行前倾段；完整模式需要显式 `--experimental`，不能作为实机策略或已完成的转换版本发布。
没有改冻结爬行策略、质量/惯量、碰撞体、摩擦、执行器实际力矩上限或原有逐2ms验收门槛。
没有commit、push、创建分支或发布tag。没有后台训练。

## 怎么运行

在 `g1-poser` 仓库根目录执行。本轮实际使用 `/private/tmp/g1-transition-v1/bin/python`。
此前临时环境仅剩部分目录，缺少有效venv配置和依赖，未覆盖它或修改UniLab环境。
重新安装固定版本后，冻结爬行轨迹哈希回归完全一致。

建议使用可重建的独立环境，不依赖临时目录永久存在：

```bash
python3.13 -m venv .venv-transition
.venv-transition/bin/python -m pip install -r rl/transition_v1_requirements.txt

# 默认：经过验证的前倾准备段，不会切换爬行
.venv-transition/bin/python rl/run_transition_v1.py

# 名义初态 + 8个固定速度扰动初态
.venv-transition/bin/python rl/run_transition_v1.py --validate

# 单元/集成检查及已有报告核对
.venv-transition/bin/python rl/check_transition_v1.py

# 未通过的完整转换，仅用于诊断；预期报告失败
.venv-transition/bin/python rl/run_transition_v1.py --mode full --experimental
```

入口生成 `output/rl/transition_v1/<时间>/report.json` 及NPZ轨迹。
依赖当前工作区已有冻结模型、策略、bundle和历史报告；**尚不是独立可分发的软件包**。
NPZ重放是验证执行轨迹，不意味着把状态回放当作物理控制。未修改主程序启动行为或万圣节场景。

## 已通过的部分

`rl/run_transition_v1.py` 的 forward 模式：
1. 真实站立、无外力，先保持0.5s。
2. 双侧髋俯仰目标在0.6s内减少0.15rad，前倾保持1.5s。
3. 在0.6s内返回站立目标，稳定到总时长5s。
4. 增加世界系俯仰角速度反馈，髋偏置为 `0.03 * pitch_rate`，限幅±0.03rad。
5. native PD及80%瞬时目标力矩保护；手臂仍处于原站姿，没有与张臂动作合并验证。

最终报告：`output/rl/transition_v1/2026-10-28_15-32-07-891249/report.json`。
名义零速度 + 种子28000–28007的初始xy速度±0.02m/s，**9/9通过**：
- 持续5s及全部原物理门槛；
- 1.6–2.5s俯仰角在0.1–0.4rad；全过程骨盆高于0.65m；
- 最后0.5s俯仰绝对值<0.08rad、线速度范数<0.1m/s、角速度范数<0.15rad/s；
- 全部控制信号逐2ms精确重放qpos/qvel。

这些样本参与了控制器迭代，**不是最终独立盲测，更不是完整站转爬成功率**。
报告分别使用 `forward_stage_passed` 和 `full_transition_passed`，前倾通过不抬高完整任务成功率。

## 基础问题查到了什么

`rl/check_standing_pitch_sign.py`：7个保持/成对关节偏置试验，0.5s保持、0.4s变化、总1.5s。
报告：`output/rl/standing_pitch_sign/2026-10-28_11-53-33-494683/report.json`。

| 目标偏置 | 末态身体俯仰 |
| --- | ---: |
| 保持站立 | −0.026rad |
| 踝俯仰−0.15rad | +0.526rad，已伴随明显向前速度 |
| 踝俯仰+0.15rad | −0.928rad |
| 髋俯仰−0.15rad | +0.236rad |
| 髋俯仰+0.15rad | −0.315rad |
| 膝+0.15rad | −0.605rad |

负向髋/踝偏置确实可以前倾；不能把“屈膝增大”当作自然向前下蹲。
这些是指定初态与时间尺度下的响应，不是独立于姿态、频率和接触的通用控制增益。

返回站立的第一版有剩余摆动；0.15角速度反馈还引发较快振荡，未通过速度验收。
独立20ms脉冲检查发现：成对髋偏置从0到+0.01rad，末端俯仰速度由约−0.0668变为−0.2195rad/s。
这说明不能把慢速姿态方向检查直接当成离散反馈增益设计。降低反馈到0.03后，同一验收时间和速度阈值下9例通过。
旧入口源代码保存在 `output/rl/transition_v1/runner_pre_damping.py` 与 `runner_high_damping.py`，旧失败报告保留。

## 完整转换试了什么，为什么未完成

### 前倾事件后折腿

`rl/transition_v1.py`、`rl/search_transition_v1.py`：以测得的正向俯仰事件触发折腿；17个参数，三个独立CEM重启。
预算固定576次搜索，不靠单条最佳轨迹不断追加迭代。
报告：`output/rl/transition_v1_search/2026-10-28_11-56-32-238032/report.json`。

候选可以到达接近爬姿的位置/关节角，但典型下落速度仍约1.5–3m/s。
760次实际冻结策略接管探针全部失败；穿深、力矩和非支撑部位触地是主要原因。
**姿态接近不等于动量可接管。** 未保存全部搜索失败轨迹；默认及各重启选中项保存并精确重放，其他候选参数/统计留档。

### 保留脚部支撑的参考、反馈与提前落手

- `stance_lowering_v1.py`：9个保持脚部世界位置/朝向的IK参考，两个执行时长，共18例，均未完整通过。
- `check_stance_feedback_v1.py`：两个参考、两个时长，原始对照与躯干反馈/偏置力补偿，共28例，均未完整通过。
- `check_forward_reach_v1.py`：三个前倾候选加世界坐标提前落手，共18例；能够触地，但部分单手峰值超过3kN，仍未通过。

报告分别为：
- `output/rl/stance_lowering_v1/2026-10-28_14-59-00-617321/report.json`
- `output/rl/stance_feedback_v1/2026-10-28_15-01-43-934037/report.json`
- `output/rl/forward_reach_v1/2026-10-28_15-04-36-582031/report.json`

因此没有继续加密这类插值或提前落手参数。

### 全身逆动力学QP

`whole_body_transition_v1.py`：显式浮动基动力学、足部6D加速度约束、8个单边点支撑力、摩擦棱锥、80%力矩上限、关节加速度/预测位置边界；同时跟踪质心、躯干和手部任务。
预测支撑力仅用于求解，**没有作为外力施加到机器人**。COM雅可比导数为20ms差分近似。
这是局部接触模型：足部4角支撑假设未被实际接触状态保证，没有碰撞避免约束，也没有完成接触切换。
QP可行不代表仿真接触状态、跟踪或转换成功。

第一轮每20ms将力矩转换成一次位置目标，运动中不等价于恒定力矩。
`whole_body_torque_bridge_v1.py` 修正为每2ms按当前执行器位置/速度重新转换，仍用原position actuator及真实限幅。
两轮站立保持对照均通过4s物理执行，但下蹲例仍出现失衡、关节越界、自碰撞或QP不可行。

- `output/rl/whole_body_lowering_v1/2026-10-28_15-20-06-786890/report.json`
- `output/rl/whole_body_torque_bridge_v1/2026-10-28_15-24-26-299011/report.json`

没有把QP方案替换进主运行程序；它目前只能作为接触模型一致性与全身控制研究实现。

## 当前完整模式验收

入口 full 模式不使用离线oracle；首次满足俯仰>1.1rad、骨盆<0.4m就实际接管一次。
这是明确标记的未验证切换规则，不是从局部接管测试推导出的安全区域。
最终报告：`output/rl/transition_v1/2026-10-28_15-36-55-527957/report.json`，**落地穿深门槛失败**。
`full_transition_validated` 保持false，未发布成功checkpoint，不能把本实验版称为完成的站转爬控制器。

## 检查与下一步边界

`rl/check_transition_v1.py` 已通过：目标限位、状态不被规划器改写、非法参数拒绝、QP约束残差、力矩位置桥重放、9条前倾报告/轨迹哈希及状态初值检查。
冻结爬行回归通过：`output/rl/transition_execution/2026-10-28_15-36-56-716577/report.json`。
名义爬行轨迹SHA仍为 `686214684b7045787b53d8c7456deea3ba8c5503e28594d001b501c0b3274121`。

下一步应先验证**真实足部接触状态与全身控制预测支撑力的一致性**，并解决低位承重/接触切换，再考虑训练或扩展动作。
不继续增加同一类CEM参数扫描，不把冲击触地当作成功，不通过降低碰撞或力矩门槛制造版本完成。
当前完成的是实验入口和前倾阶段，而不是用户最终要求的完整转换。
