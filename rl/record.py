from pathlib import Path
import argparse

import imageio.v2 as imageio
import mlx.core as mx
import mujoco
import numpy as np

from policy import load_policy


def main():
    parser = argparse.ArgumentParser(description="Record the first deterministic rollout, stopping on failure")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--seconds", type=float, default=20.0)
    parser.add_argument("--seed", type=int, default=1001)
    args = parser.parse_args()
    if args.seconds <= 0:
        parser.error("--seconds must be positive")
    env, policy = load_policy(args.checkpoint, 1, args.seed)
    output = args.checkpoint.with_name(f"{args.checkpoint.stem}_seed{args.seed}.gif")
    render_model = mujoco.MjModel.from_xml_path(env.cfg.scene.model_file)
    render_model.vis.global_.offwidth = 640
    render_model.vis.global_.offheight = 480
    render_data = mujoco.MjData(render_model)
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(render_model, camera)
    camera.distance = 2.3
    camera.azimuth = 135
    camera.elevation = -22
    try:
        state = env.init_state()
        with mujoco.Renderer(render_model, height=480, width=640) as renderer:
            with imageio.get_writer(output, mode="I", duration=env.cfg.ctrl_dt * 4000, loop=0) as writer:
                for step in range(min(int(args.seconds / env.cfg.ctrl_dt), env.cfg.max_episode_steps)):
                    snapshot = np.asarray(env.get_physics_state_snapshot()[0], dtype=np.float64)
                    if step % 4 == 0:
                        mujoco.mj_setState(render_model, render_data, snapshot, mujoco.mjtState.mjSTATE_FULLPHYSICS)
                        mujoco.mj_forward(render_model, render_data)
                        camera.lookat[:] = render_data.qpos[:3]
                        renderer.update_scene(render_data, camera)
                        pixels = renderer.render()
                        writer.append_data(pixels)
                        if step == 100:
                            imageio.imwrite(output.with_suffix(".png"), pixels)
                    state = env.step(np.asarray(policy.policy(mx.array(state.obs["obs"]))))
                    if state.terminated[0]:
                        print(f"Episode failed after {(step + 1) * env.cfg.ctrl_dt:.2f} seconds; not selecting a different rollout.")
                        break
        print(output)
    finally:
        env.close()


if __name__ == "__main__":
    main()
