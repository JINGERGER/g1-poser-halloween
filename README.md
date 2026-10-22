# G1 Poser Halloween — flat-ground crawl preview

A modified 23-actuator G1 crawling in a grassy Halloween cemetery, with pumpkins,
ghosts and three witches flying on broomsticks. The crawl controller runs live
MuJoCo physics; the scene decorations are non-colliding and do not change the
verified robot trajectory. The original MINK IK poser remains available separately.

## Crawl quick start (Apple Silicon macOS)

Install `uv`, clone this release/tag, then run from the repository root:

```bash
uv sync --extra crawl --python 3.13 --locked
uv run --extra crawl python rl/install_crawl_assets.py
uv run --extra crawl mjpython rl/run_crawl_controller.py --viewer
```

No UniLab installation or neighboring `g1_crawl` checkout is required. The installer
downloads the release's checksum-pinned model, initial state, reference policy and
test trajectory into ignored `output/rl/` directories. It refuses conflicting local
assets rather than overwriting research outputs. An offline archive can be installed
with `python rl/install_crawl_assets.py --archive /path/to/crawl-assets-v0.1.0-halloween.zip`.

Headless: `uv run --extra crawl python rl/run_crawl_controller.py`.
Add `--scene plain` to remove the cemetery background. Each run lasts 10 simulation
seconds and saves a timestamped report and trajectory under `output/rl/crawl_controller/`.
The graphics window has not been manually revalidated as part of release packaging.

**Scope:** flat-ground simulation preview, not a hardware controller or slope/stair
policy. Three nominal trials each advanced about 4.81 m in 10 seconds and passed
22 complete diagonal cycles. Actual peak clearance is about 4 cm at the hands and
3 cm at the feet, not the 8 cm residual request. This uses an upstream RL policy
plus deterministic control corrections and an explicit PD/contact profile; it is
not a newly trained policy or a pass under the original unmodified contact settings.

See [controller and validation](docs/RL_CRAWL_CONTROLLER.md),
[release notes](docs/RELEASE_v0.1.0-halloween.md), and
[third-party provenance](THIRD_PARTY_NOTICES.md).

![App screenshot](screenshot.png)

### Original IK poser quick start
```bash
uv sync
uv run main.py
```

Install `uv` first if it is not already available. The project uses an isolated
Python 3.11 environment; `uv sync` downloads the interpreter when needed.
The default installation is simulation-only and does not require the Unitree SDK.

To start with physics playback enabled:
```bash
uv run main.py --physics
```
Without `--physics`, the editor starts in IK mode. The flag initializes the
Physics checkbox; you can still switch modes in the UI. Physics playback bakes
the existing gait on startup and does not enable autonomous stair climbing.
The same flag works with `main-proc-anim.py`. Use `uv run main.py --help` to
see available startup options.

### macOS (Apple Silicon)

Run `uv run main.py` with ordinary Python, **not** `mjpython`. The launcher opens
the gait controls with Tk on the main thread and starts the MuJoCo viewer in a
separate `mjpython` process. This keeps both macOS GUI event loops on their own
main threads. Closing either window ends the application.

The viewer receives simulation state at up to 60 Hz; the IK loop retains its
200 Hz target. Camera changes and viewer edits are synchronized back to the
simulation. macOS uses a native number dialog and file picker, so Zenity is not
required. Linux retains the existing in-process viewer and Zenity dialogs.

`main.py` launches `main-proc-anim.py`, which contains the gait editor. You can
also run `uv run main-proc-anim.py` directly. If using a separately installed
Python instead of the uv-managed interpreter, ensure it includes Tkinter.

### What `main.py` provides
- **Live IK gait tuning**: place hands/feet with simple 2D pads, set lift heights, duty/phase, and symmetry; MINK solves IK at ~200 Hz.
- **Camera controls**: azimuth, elevation, distance.
- **Import/Export**: save/load compact gait configs to `output/gait_*.json`.
- **Animation export**: write 3-cycle `gait_animation.v1` JSON (`output/animation_*.json`) with `qpos` frames and site positions for offline playback (e.g., Isaac Sim).
- **Model**: uses `g1_description/scene_g1_targets.xml`.

### Crawl runtime versus research

The released runtime and verification tools live in `rl/`. They are independent
of the poser's `--physics` playback. Historical UniLab experiments and failed-search
outputs are not required by, or bundled into, this preview release.

### Original IK editor scene (not the crawl runtime)
The editor scene includes Halloween decorations and a solid ramp defined in
`g1_description/halloween.xml`. The ramp ascends along +X from x=1.30 to 3.05 m:
1.75 m horizontal run, 1.40 m width, 50 cm rise (about 15.9 degrees), and a landing.
The ramp and landing have collision enabled; pumpkins, ghosts, gravestones, and edge markings
are visual-only. Zoom out in the viewer to see the entire scene.
The pumpkins float with their bottoms 45 cm above the ground. The default gait
lift height is 8 cm for all four limbs, including after Reset.
The existing gait still targets a flat ground plane: adding a ramp does not
enable autonomous slope climbing or terrain-aware foot placement. IK playback
can pass through obstacles; physics mode alone does not provide a climbing policy.

### Notes on other scripts

- Older scripts have been moved to `scripts/`. They have not been re-tested after the move and may be outdated/broken. Use at your own risk.
- Some of those scripts (e.g., real control) depend on Unitree SDK2 + CycloneDDS.
  They are not installed by `uv sync`, and real-robot operation is not covered by
  the macOS simulation setup. After separately installing CycloneDDS and obtaining
  a compatible SDK checkout, install it explicitly, for example:
```bash
export CYCLONEDDS_HOME="$HOME/cyclonedds/install"
uv pip install -e ../unitree_sdk2_python
```
Subsequent `uv sync` calls remove packages outside the simulation lockfile;
install the SDK again if needed. Do not connect to a robot merely to test the GUI.

### Dev warning: Tkinter + MuJoCo threading
- Avoid `ttk.Entry`/`tk.Entry` with the MuJoCo viewer (can crash XCB). Use `ttk.Label` displays with variable traces instead.
- Defer `mujoco.*` calls that depend on the viewer until after the viewer is launched.
- On macOS, do not put Tk in an `mjpython` script: its Python thread is not the
  native macOS main thread. `platform_viewer.py` isolates the viewer instead.

### New reference
- `scripts/mink_g1_ik_reference.py`: concise, modern Mink IK example for G1 with mocap targets. Run:
```bash
uv run scripts/mink_g1_ik_reference.py
```
On macOS this standalone example has no Tk UI, so launch it with:
```bash
uv run mjpython scripts/mink_g1_ik_reference.py
```
Run these commands from the repository root so the example can find its model.
