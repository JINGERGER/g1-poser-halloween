# UniLab G1 crawl integration

## Active IK Soft-Reference RL

The new independent `G1IKTracking23` task uses loaded support resets, a checked
periodic IK reference, full-body residual actions and nonnegative physical-task
rewards. IK is a soft guide, not assumed executable expert data. Actor/critic
share a new 178d observation; old checkpoints are intentionally incompatible.
Plan, budgets, exact commands and measured results: `docs/RL_IK_TRACKING_PLAN.md`.

Use `ik_assets.py` to prepare the immutable snapshot, `check_ik.py` to verify
interfaces, `train_ik.py ASSETS --seed 1` for the bounded default smoke test, and
`run_ik_screen.py ASSETS --dry-run` to inspect the two-seed 614,400-transition
screen. The screen also evaluates held-support and zero-residual controls on the
same 100 starts. It does not auto-extend its budget or pick best-reward weights.
`evaluate_ik.py CHECKPOINT --controller learned|zero|hold` uses physical gait
criteria rather than pose imitation alone. The completed two-seed screen also
failed crawling: seed1 remains supported without gait, seed2 terminates early.
All optimizer updates were finite; this is a negative experiment, not a usable
crawling controller. No automatic training extension is running.

## Short Physical Motion Prototype

`support_handoff.py` now records paired 10s loaded-state anchors from the two
existing support policies and compares three-second continued feedback, frozen
targets and a bounded right-leg IK residual. Both anchors sustain support, but
the residual does not unload the right foot. No new training or crawling success
is claimed. See `docs/RL_SUPPORT_HANDOFF.md`; CPU checks are in
`check_support_handoff.py --report REPORT`.

`short_crawl.py` constructs a smooth periodic joint reference and executes it
through the existing physical position servos, without base pinning, external
forces or state writes after initialization. Its bounded 12-candidate run has
**no passing crawling example**: some forward motion occurs, but contact,
slip and complete limb-cycle requirements fail. Do not use it as expert data.
See `docs/RL_SHORT_CRAWL.md` for results, recorded-control GIF replay and commands.
`check_short_crawl.py --report REPORT` checks interpolation and reproduces all
saved position/velocity traces by integrating the recorded controls.

## Active reference-inspired experiment

The four-run screen completed on 2026-10-16: 12,002,304 training transitions,
zero nonfinite update skips, and **0/100 crawling passes for every run**.
Current-PD policies fail quickly; reference-PD policies survive but do not crawl.
Do not automatically extend these runs. Full results and the next bounded
motion-reference/tracking plan are in `docs/RL_EXPERIMENT_MATRIX.md`.
Recorded training timers were about nine minutes each, but calendar timestamps
span about 8h14m for this matrix; the timer is not end-to-end delivery time.
The runner now checks both monotonic and calendar deadlines.

The no-training counterfactual audit continues frozen policies after physical
failure to measure omitted rewards, without changing task defaults or weights:

```bash
/Users/loopy/UniLab/.venv/bin/python -B rl/check_termination_returns.py
/Users/loopy/UniLab/.venv/bin/python -B rl/audit_reference_termination.py output/rl/reference_matrix/2026-10-15_19-34-38-730759/matrix.json
```

The second command needs Metal access. Its `termination_counterfactual.json`
shows negative omitted tails for all 200 current-PD episodes. This establishes
a local stopping incentive under this continuation, not causal proof of the
training outcome or equivalence to PPO's bootstrapped objective. Suppressing
safety termination is a diagnostic intervention, not the proposed training fix.

The whole experiment matrix and stopping rules are in `docs/RL_EXPERIMENT_MATRIX.md`.
`G1CrawlReference23` is isolated from V1–V5: a genuinely separate 75d actor / 81d
critic, name-mapped 23d actions, no prescribed limb phase, random animation reset
with explicit floor projection, and current/reference-PD immutable snapshots.
This is a **simplified reference-inspired task**, not exact Isaac Lab parity.
Symmetry, physical DR and force-history contacts are not implemented; see the matrix
for all differences. Existing V2 evaluation/policy loaders do not support this task.

From `g1-poser`, using the existing UniLab environment:

```bash
/Users/loopy/UniLab/.venv/bin/python -B rl/audit_reference.py
/Users/loopy/UniLab/.venv/bin/python -B rl/check_reference.py
/Users/loopy/UniLab/.venv/bin/python -B rl/check_reference_model.py
/Users/loopy/UniLab/.venv/bin/python -B rl/train_reference.py --pd current --iterations 10 --seed 1
/Users/loopy/UniLab/.venv/bin/python -B rl/train_reference.py --pd reference --iterations 10 --seed 1
```

The last three commands need Metal access outside the CLI sandbox. The training
entry point defaults to stage-0 (maximum 100 iterations), starts fresh, records model /
source hashes and learner metrics, and fails on nonfinite/skipped updates. A local
equivalent finite-ELU implementation avoids the upstream activation's reproduced
large-positive-input gradient overflow; UniLab files are unchanged. KL statistics
and learning-rate adaptation are explicitly updated each minibatch.
Both final 10-iteration smoke runs applied 200 optimizer updates with zero nonfinite
skips; trained weights reload strictly. This is not a validated crawling controller.
The phase-free evaluator and bounded four-run screen are now available:

```bash
/Users/loopy/UniLab/.venv/bin/python -B rl/check_reference_gait.py
/Users/loopy/UniLab/.venv/bin/python -B rl/check_reference_matrix.py
/Users/loopy/UniLab/.venv/bin/python -B rl/evaluate_reference.py CHECKPOINT --record-first
/Users/loopy/UniLab/.venv/bin/python -B rl/run_reference_matrix.py --dry-run
/Users/loopy/UniLab/.venv/bin/python -B rl/run_reference_matrix.py
```

Evaluation uses the frozen `rl/reference_eval.json`: 100 first episodes, seed 1001,
20 seconds, 0.1 m/s straight command, clean actor observations and the training
reset distribution. All transitions, including the pre-auto-reset terminal state,
are replayed in native MuJoCo with the applied control. Each limb must first bear
load, then unload with actual geom clearance and no floor contact, and finally
bear load again. Initial drops, unfinished flights and one-frame flicker do not
count. No prescribed phase is used. Survival alone is not success.
GIF recording is from the exact first episode in the same batch; changing batch
size changes the sampled initial states. No best-episode selection is performed.
Metrics are 20ms endpoint samples, not substep maxima. Thresholds are engineering
screening criteria, not hardware safety specifications.

The matrix runs current/reference PD × seeds 1/2 sequentially, each from scratch
for 3907 × 32 × 24 = 3,000,576 transitions. Only final checkpoints are evaluated.
Limits: 30 minutes per training process, 5 minutes per evaluation, 90 minutes total.
Nonfinite learner updates, source/config changes, process failures or timeouts
stop the matrix; there are no automatic retries or budget extensions.
`output/rl/reference_matrix/<timestamp>/matrix.json` records status and results;
each run has `progress.json`, `training.log`, `training.json`, and `evaluation.log`.
The explicit `train_reference.py --stage screen` mode permits only the frozen
32-env/3907-iteration/seed-1-or-2 configuration; use the matrix wrapper for wall-clock
limits. `--dry-run` prints the complete plan without starting training.

## Historical baseline work

Current reference: the modified `jloganolson/g1_crawl`, not a standard G1 model.
See `docs/RL_G1_CRAWL_REFERENCE.md` for the pinned source/interface comparison.
`rl/check_reference_pd.py` compares current and reference-training PD numbers
on identical in-memory models and paired initial states, using fixed home
targets. It requires `rl/audit_reference.py` first and rejects a model hash
mismatch. No XML or training defaults are changed. Reference gains alone caused
earlier torso contact (1.28–1.46 s versus 6.02–6.22 s), not better support.
The isolated reference-interface implementation is now available above; no gain sweep is planned.

**Baseline audit warning:** V2–V4 force rewards used control-dependent sensors
refreshed without the applied command. Those stages are now blocked by default;
`--allow-legacy-force` is only for reproducing the known flawed experiments in
the historical commands below. Do not use them to choose a new controller.
See `docs/RL_BASELINE_AUDIT.md` for reproductions and limits of the findings.
V5 is an untrained 93-dimensional diagnostic candidate with validated fixed-PD
force reconstruction and anchor observations, not a validated policy. Current
work uses a bounded force-corrected V2 support baseline, not V5 training.

The force fix is now available independently of V5: V2 accepts
`force_source: analytic_pd`, selected by `train.py --stage direct-support
--correct-force` (also available for `direct-crawl`). It changes only the force
source, not rewards, observations or the physical model. Old configurations
default to `legacy_sensor` for reproducibility; the known flawed training path
remains blocked without `--allow-legacy-force`. Corrected configs have a
`_correct_force` suffix; saved run configs retain the source choice.

Run `rl/check_servo_force.py`, `rl/check_integrity.py` and
`rl/check_support_baseline.py` with the UniLab interpreter before training.
The last check holds home targets for five independent ten-second rollouts,
without auto-reset or a pinned base, and measures actual normal loads and
contact-point slip. All five current runs settle with torso support after about
six seconds: this is not successful four-limb support. Reports go to
`output/rl/{servo_force_check,integrity_check,force_only_comparison,support_baseline_check}.json`.
One corrected support run (300 iterations, 32 environments, seed 1) is complete:
`output/rl/mlx_ppo/G1CrawlDirect23V2/2026-10-15_11-42-03_mujoco/model_299.safetensors`.
All 100 held-out episodes survived 20 seconds. Ten additional native replays had
no non-support floor contacts at sampled endpoints, but mean loaded contact slip
remained 0.0172–0.0184 m/s. This is preliminary support, not validated crawling.
The same-budget seed-2 run is at
`output/rl/mlx_ppo/G1CrawlDirect23V2/2026-10-15_11-48-28_mujoco/model_299.safetensors`.
It also survived 100/100 held-out episodes and had no sampled non-support floor
contacts in ten native audits. The last ten seconds still averaged 0.0178/0.0143
m/s loaded contact slip for seed 1/2 respectively: not merely startup settling.
Do not compare a zero-speed support
run with the historical moving V2 run as if only the force source changed.

`rl/evaluate_support.py CHECKPOINT --episodes 10 --seed 2001` audits zero-command
DirectEnv policies using native replay with the applied control, including the
terminal transition before batch auto-reset. It writes per-episode support loads,
contact-point slip, non-support contact and torque summaries next to the weights.
The report includes the first full trace and model hash; metrics are 20 ms
endpoint samples, not substep maxima. There is no success label based only on
survival. Use `rl/evaluate.py` separately for the broader 100-episode assessment.
Each episode also reports 0–1, 1–5, 5–10 and 10–20 second slip/load windows,
including per-limb slip conditioned on >1 N load. Window time comes from the
control-step counter. `rl/check_support_windows.py` tests boundaries and unloaded
samples. Empty windows are not treated as zero slip or successful support.

For positive-command V2 policies, use `rl/evaluate_support.py CHECKPOINT
--moving --episodes 10 --seed 2001`. This writes a separate `_crawl_audit_`
report with native signed geom gaps, floor contacts, loads and phase-window
statistics, including near-limit joint counts after startup. Peak clearance
requires all of >=15 mm gap, no floor contact and <=1 N load. Independent landing
window statistics are not proof of paired lift/landing cycles. Run
`rl/check_gait_audit.py` for geometry/contact/load and missing-window checks.

The first corrected low-speed run (300 iterations, seed 1) is at
`output/rl/mlx_ppo/G1CrawlDirect23V2/2026-10-15_11-56-20_mujoco/model_299.safetensors`.
It survives 100/100 episodes but advances only 6.37 cm on average in 20 s. Both
feet have zero effective 15 mm peak clearance in ten native audits; loaded
contact slip is about 3 cm/s. This is a failed crawling screen, not a successful
controller. No additional training budget was used; see the training plan for
the bounded reward-module comparison proposed next.

Read-only geometry/static-support audit (no policy required):

```bash
uv run --project /Users/loopy/UniLab --no-sync --offline python -B rl/audit_geometry.py
```

This compares source/training collision and inertial parameters, probes floor
contacts, then searches 60 local pose seeds per rear foot for nonnegative
vertical-force equilibrium within torque/control limits. Up to two distinct
screened candidates per side receive a three-second fixed-PD physical hold.
Results go to `output/rl/geometry_audit.json`; source assets remain unchanged.
Passing the static screen is not a physically stable hold, and finding no
successful hold is not proof of impossibility. `--self-check-only` runs the
solver's feasible/infeasible small-model controls without the robot search.

Follow-up native contact-equilibrium audit:

```bash
uv run --project /Users/loopy/UniLab --no-sync --offline python -B rl/audit_equilibrium.py
```

Requires the geometry report and an unchanged model hash. Optimizes root height,
joint positions and fixed controls from four previous seeds using zero-velocity
native `mj_forward` acceleration residuals, then performs three-second physical
holds. Writes `output/rl/equilibrium_audit.json`. The bounded local search does
not establish reachability or impossibility; a successful process exit only
means the diagnostic completed, not that equilibrium or stability passed.

This directory registers `G1CrawlFlat23` without editing the UniLab checkout.
Use the existing UniLab interpreter, not the poser's Python environment.
Run commands from the `g1-poser` repository root.

## Prepare and diagnose

```bash
uv run --project /Users/loopy/UniLab --no-sync --offline python -B rl/prepare.py
uv run --project /Users/loopy/UniLab --no-sync --offline python -B rl/check.py
```

Preparation generates a training-only XML, a model manifest, and offline IK
joint-reference tables in `output/rl/assets/`. It changes coarse arm collision
capsules, uses 3.5 cm hand contact spheres, and excludes only the overlapping
upper-arm/forearm pair at each elbow. Other self collisions and ground contacts
remain enabled. The original poser model is untouched. This is a simulation
prototype, not a verified hardware collision model.

Physics uses 2 ms steps, implicitfast integration, and 50 solver iterations;
policy control is 50 Hz. There are 23 normalized residual actions and 85 scalar
observations. The actor currently receives world-frame base velocity and torso
up direction; this is a fixed-heading simulation task, not a deployable
proprioception-only controller or a heading-invariant locomotion policy.

`check.py` checks finite states, observation shapes, selective reset isolation,
initial contacts, and a 20-second zero-action PD baseline. A successful script
exit means the checks ran, not that the untrained baseline is stable: inspect
`failures_per_env` in `output/rl/zero_action_check.json`.

## Stage 1: support

### Single-limb diagnostic before further training

```bash
uv run --project /Users/loopy/UniLab --no-sync --offline python -B rl/diagnose_lift.py --height 0.02
```

This runs nine deterministic, independent physics trials: nominal all-support
PD, then each limb with/without a reference root shift toward the remaining
three supports (capped at 8 cm). Each trial lasts at most five seconds:
settle, shift, lift, hold, lower, recover. Only the separate IK reference data
is repositioned; the simulated floating base is never pinned or teleported.
No learned policy, training, or changes to the demonstration scene are involved.

`output/rl/lift_diagnostic.json` contains model hash, summaries and 50 Hz traces:
site lift relative to the initial pose, actual collision-geometry floor gap,
contacts, actual stance-site triangle COM margin, joint tracking error,
orientation deviation, stance drift, reference IK error/penetration and forces.
Reference poses are diagnostic candidates, not collision-certified trajectories.
The COM margin assumes all three remaining supports bear weight; inspect their
contact flags too. It is not a dynamic stability certificate. Actuator-level
force limits are absent, but joint-level `actuatorfrcrange` limits are active.
Actuator saturation ratios are `null`; joint saturation ratios use the actual
generalized actuator force after joint-level clamping. These XML limits are
not a hardware calibration certificate.

The default report is overwritten on rerun; use `--output PATH` to retain
experiments and `--model PATH` to select a training model snapshot. A completed
command is not a successful lift. A failed all-support control also confounds
attribution to the lift itself. The five-second test cannot establish long-term
support stability or validate the separately trained support policy.

Compare a bounded load-compensated prototype without changing gains or limits:

```bash
uv run --project /Users/loopy/UniLab --no-sync --offline python -B rl/diagnose_lift.py \
  --controller load-compensated --output output/rl/lift_compensated.json
uv run --project /Users/loopy/UniLab --no-sync --offline python -B rl/diagnose_lift.py \
  --output output/rl/lift_nominal_v2.json
```

The prototype allocates nonnegative vertical support forces by enumerating
active support subsets and solving the floating-base static wrench equations.
It blends four-support and three-support solutions during unloading, maps the
remaining joint torque to position-servo offsets using existing gains, and adds
bounded joint-error integral compensation. Integral bounds are 0.15 rad and
total compensation bounds 0.5 rad; final commands respect actuator control limits.
No external base wrench, state reset, gain increase or torque-limit increase is
used. Wrench residuals are diagnostic (mixed force/moment units), not a feasibility
guarantee. The force allocation is quasi-static, vertical-only, and scheduled;
it is **not yet a contact-feedback whole-body controller**. Traces include measured
normal contact forces and names of joints at >=99% of their force limits.
Joint tracking error compares the physical joint to the IK target, not the
load-offset servo command. The current compensated experiments still fail to
lift: do not promote this controller to training reference or hardware deployment.

```bash
uv run --project /Users/loopy/UniLab --no-sync --offline python -B rl/train.py \
  --stage support algo.num_envs=64 algo.max_iterations=500
```

MLX PPO is the default. `--algo ppo` selects the existing PyTorch/MPS runner;
its 3-iteration smoke check passed, but the policy evaluation tools below
currently support MLX `.safetensors` checkpoints only.

## Evaluate before advancing

```bash
uv run --project /Users/loopy/UniLab --no-sync --offline python -B rl/evaluate.py \
  /absolute/path/to/model_499.safetensors --episodes 100 --seed 1001
uv run --project /Users/loopy/UniLab --no-sync --offline python -B rl/record.py \
  /absolute/path/to/model_499.safetensors --seconds 20
```

Evaluation uses a held-out reset seed and larger joint reset noise, averages
the first episode of every environment, and distinguishes failure from timeout.
GIF recording uses the first rollout, stops on failure, and requires no FFmpeg.
Checkpoints, configuration, model snapshots, evaluations and GIFs stay under
`output/rl/`. Checkpoint and optimizer files must come from a trusted source.

## Stage 2: flat crawling

Only advance after evaluating support. Replace the example path with the
validated support checkpoint:

```bash
uv run --project /Users/loopy/UniLab --no-sync --offline python -B rl/train.py \
  --stage crawl algo.num_envs=64 algo.max_iterations=1000 \
  algo.load_run=/absolute/path/to/model_499.safetensors
```

The crawling stage uses phase/speed-interpolated offline IK references plus
learned residual joint targets. These references are not an RL policy and are
not guaranteed dynamically feasible. Physics evolves solely through position
actuators and the UniLab backend; reference playback does not teleport the base
or joints during a rollout. Initial state can be set only at reset.

Commands are 0.05–0.15 m/s, cycle length is 2 s, duty factor is 0.75, and target
swing clearance is 8 cm. The reference tables were generated for those cycle
and clearance settings: changing them requires regenerating corresponding
references, not just editing a reward target.

Do not accept forward sliding as crawling. The numerical flat-terrain gate
requires 95% 20-second survival, speed MAE < 0.05 m/s, peak-swing contact
fraction < 30%, and peak-swing mean clearance > 4 cm. Video review and multiple
training seeds remain required even when the numerical gate passes.

## Independent direct-action V2 experiment

The reference repository is design input only; no external policy is required.
`G1CrawlDirect23V2` preserves the existing 23-joint model, collision geometry,
mass, gains and joint torque limits. It replaces observation entries 6:9 with
pelvis-frame projected gravity and uses that vector for orientation reward and
termination. New model sensors measure joint actuator forces and every enabled
non-support collision geom against the floor. Non-support floor contact ends
the episode; rewards penalize normalized torque and proximity to torque limits.

V2 does not use an IK joint reference. Its actions are default joint angles plus
0.5 times the policy output. It still uses phase-conditioned support/clearance
rewards and phase observations (4-second cycle), so this is not a fully
phase-free gait learner. The initial target is 2 cm lift at 0.02–0.05 m/s.
Actor observations still include simulation-only world velocity/contact data;
actor/critic separation and deployable observations are future work.

```bash
uv run --project /Users/loopy/UniLab --no-sync --offline python -B rl/check.py --direct
uv run --project /Users/loopy/UniLab --no-sync --offline python -B rl/train.py \
  --stage direct-crawl algo.num_envs=32 algo.max_iterations=300
```

`--stage direct-support` trains the new protocol with zero velocity commands.
Both V2 entry points currently start fresh and reject resume overrides to avoid
silently loading old 85-dimensional weights with incompatible semantics. Old
`support`/`crawl` stages and their weights retain their original behavior.
Evaluate/record select the task from the checkpoint's saved run config.
V2 generated assets live in `output/rl/assets/direct_v2`; immutable model snapshots
and task-specific run directories separate the experiments.

The evaluator retains the original >4 cm flat-gait gate; a 2 cm experiment does
not waive that later gate. `low_lift_screen_passed` is only an initial screening
metric (95% survival, <30% peak swing contact, >=1.5 cm mean site rise). It is not
proof of 2 cm collision-geometry clearance, per-limb success, forward tracking,
or dynamic stability; video, per-limb measurements and multiple seeds remain
necessary. No terrain training is enabled by passing this screen.

## Balanced swing/landing V3 comparison

`G1CrawlBalanced23V3` keeps V2's model physics, observation protocol, action
scale and phase schedule. It adds four signed collision-geometry distance
sensors to the flat floor. The old all-limb support/clearance reward weights
are zero: stance and swing now have separate normalization, and clearance is
scored only for the limb whose swing is scheduled. A narrow 8 mm error scale
replaces the old broad site-height reward. Peak-swing contact, hovering during
stance, and missing the landing window are no longer hidden by other limbs.
This is still phase-conditioned RL, not contact-triggered whole-body control.

```bash
uv run --project /Users/loopy/UniLab --no-sync --offline python -B rl/check.py --balanced
uv run --project /Users/loopy/UniLab --no-sync --offline python -B rl/train.py \
  --stage balanced-crawl algo.num_envs=32 algo.max_iterations=300
```

Use the same `evaluate.py CHECKPOINT --episodes 100` entry point. V3 reports
geometry gaps, fraction of peak samples with >=1.5 cm clearance and no contact,
and contact fraction in the first 10% of each limb's next stance phase. It also
pairs each sufficiently sampled peak window with its following landing window;
initial/final partial windows and post-reset episodes do not count as successes.
`balanced_lift_gate_passed` requires 95% episode survival and, for every limb,
70% clear peak samples, 90% landing contact, and 90% successful paired windows.
This is a low-lift screening gate, not proof of forward crawling, full-cycle
stability, precisely 2 cm clearance, or terrain readiness. Old site-rise metrics
remain for comparisons but must not be interpreted as geometry clearance.

Checks compare distance sensors with `mj_geomDistance` and a 5 cm root-height
translation in a separate test state; each limb's synthetic perfect swing must
outscore staying on the floor, and perpetual hovering loses stance/landing score.
These diagnostic state edits do not occur in training rollouts.

## Isolated foot-unloading V4 curricula

`G1SingleFoot23V4` uses the V3 physics/sensors and direct actions, but schedules
only one rear foot to lift and land. The other three limbs stay in stance;
the velocity command is zero, while the selected foot's phase remains active.
Left and right feet are separate policies selected by `env.target_foot` (2/3),
not a combined policy conditioned on a limb identifier. This avoids pretending
that a zero speed command disables a single-foot training exercise.

```bash
uv run --project /Users/loopy/UniLab --no-sync --offline python -B rl/check.py --foot left
uv run --project /Users/loopy/UniLab --no-sync --offline python -B rl/check.py --foot right
uv run --project /Users/loopy/UniLab --no-sync --offline python -B rl/train.py \
  --stage left-foot algo.num_envs=32 algo.max_iterations=600
uv run --project /Users/loopy/UniLab --no-sync --offline python -B rl/train.py \
  --stage right-foot algo.num_envs=32 algo.max_iterations=600
```

The four-second cycle and 2 cm geometric lift target remain unchanged. Compared
with V3, the foot course reduces velocity/height/posture reward pressure and
increases selected-swing and stance-contact rewards. It starts from scratch,
not from another task's checkpoint. No geometry, mass, force limit or actuator
gain changes are made. Whole-body actions remain available to learn weight
transfer; the three stance limbs are not mechanically fixed.

`evaluate.py` labels this stage `single-foot` and reports the selected foot.
`single_foot_lift_gate_passed` checks that foot's paired swing/landing windows
using the V3 clearance thresholds plus >=95% contact for each other limb over
the episode, >=95% episode survival, and <=10 cm p95 maximum horizontal drift
from the episode start. Unscheduled limbs have null swing
statistics, not fabricated zero/failing swing attempts. The four-limb balanced
gate is null for this task; passing a left-foot course is not evidence of
right-foot ability, forward crawling, or terrain readiness.

## Frozen component-model short step

The new `G1ComponentShortStepV1` task uses the audited 20mm-ball/component-hull
model and a native load-bearing reset, not the older V3/V4 reset/model assets.
It trains one complete left-hand placement with a kinematic soft reference and
23 joint-target residuals. Ordered unload/airborne/advance/reload events define
success; reward, static balance and partial unloading do not.

See `docs/RL_SHORT_STEP_RESIDUAL_PLAN.md` for the frozen protocol, implementation,
smoke results, limitations and explicitly bounded screen. Entry points:

```bash
PYTHONPATH=output/rl/solver_deps /Users/loopy/UniLab/.venv/bin/python rl/check_short_step.py
PYTHONPATH=output/rl/solver_deps /Users/loopy/UniLab/.venv/bin/python rl/train_short_step.py --stage smoke --seed 1
```

The MLX runner needs Metal access. Do not interpret a successful smoke test as
successful locomotion; the evaluated smoke policy has not completed a short step.

### Visualize the saved short-step evaluation

From the repository root on macOS:

```bash
/Users/loopy/UniLab/.venv/bin/mjpython rl/replay_short_step.py
```

This defaults to the audited 2026-10-19 21:08 evaluation's PPO trajectory at
half speed, not a fresh controller rollout. It verifies the bundle, model,
meshes and trajectory hashes without loading MLX or retraining. The current
PPO trial stops at 2.352 seconds on the 5 mm penetration guard; it has not
completed a step. Playback holds the final pose rather than hiding the stop.
Space pauses/resumes, R restarts, and the mouse rotates/zooms the camera.
Close the window to exit. Collision hull overlays are hidden by default.

Use `--case hold`, `--case known_unload_return` or `--case ik_zero_residual`
to compare the other saved physical trials. `--speed 1` uses real time;
`--report PATH` selects another compatible evaluation report.
`--check-only` validates the selected recording without opening a window
(ordinary Python works for this option).

## Termination-rule comparison

The termination-rule training comparison is documented in
`docs/RL_TERMINATION_ABLATION.md`. `compare_exploration_training.py` runs a
fixed 49,152-transition, two-seed comparison and evaluates every checkpoint
under both the unchanged strict task and a separate exploratory training task.
It requires Metal access. Exploratory training deliberately tolerates some
strict violations; its reported success is not strict acceptance or hardware
safety. Do not substitute its evaluation for `evaluate_short_step.py`.

## Unloading reward comparison

The reward-only unloading comparison is described in
`docs/RL_UNLOAD_OBJECTIVE_ABLATION.md`. `compare_unload_objective.py` runs one
384-transition smoke and two fresh 12,288-transition training seeds, then
uses the unchanged strict evaluator and native replay checker. It removes
alive/reference rewards and strengthens continuous unloading without changing
physics, actions, termination or the full-step success definition. Partial
unloading is not a successful step or a completed curriculum.

## Unloading learning curve

For the fixed 4x unloading learning-curve budget, see
`docs/RL_UNLOAD_LEARNING_CURVE.md`. `run_unload_curve.py` trains two fresh seeds
for 49,152 transitions each without changing rewards or termination, then
strictly evaluates all three preselected checkpoints. Episode telemetry is
separate from task success; partial episodes and endpoint dwell estimates
must not be reported as completed steps. `check_unload_curve_report.py`
verifies accounting, provenance and native replay of every evaluation.

## Terrain implementation

Slope and stair training are intentionally not registered yet. Do not train
them using the flat-ground reward or this task's floor-only contact sensors.
They need terrain-relative clearance, contact sensing against all terrain
geoms, height observations, spawn logic, and separate up/down evaluations.
See `docs/RL_TRAINING_PLAN.md` for the milestones and measured progress.
