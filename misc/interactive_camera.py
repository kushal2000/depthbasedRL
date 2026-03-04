#!/usr/bin/env python3
"""Interactive camera orientation finder using Flask.

Launches Isaac Lab with 1 env and provides a web UI with sliders
to adjust camera orientation in real-time, showing the depth image.

Usage:
    conda activate sapg_il
    python interactive_camera.py --enable_cameras --use_depth
    python interactive_camera.py --enable_cameras --use_depth --gpu 1

Current best config (set in sim_tool_real_cfg.py):
    rot=(0.5018, 0.3481, -0.5893, 0.5289)  # wxyz
    pos=(-0.11, 0.00, -0.03)
    # Equivalent: --rot_angle 37 --tilt_x -5 --tilt_y -10 --pos "-0.11,0,-0.03"
"""
import argparse
import os
import sys
import threading
import io
import base64
import time

os.environ.pop("DISPLAY", None)

# Parse gpu arg early
_pre_parser = argparse.ArgumentParser(add_help=False)
_pre_parser.add_argument("--gpu", type=int, default=0)
_pre_args, _ = _pre_parser.parse_known_args()

os.environ["CUDA_VISIBLE_DEVICES"] = str(_pre_args.gpu)
sys.argv.append(f"--/renderer/activeGpu={_pre_args.gpu}")
sys.argv.append("--/renderer/multiGpu/enabled=false")

import torch
torch.cuda.set_device(0)

# Global state shared between Flask and sim thread
state = {
    "rot": 0, "tx": 0, "ty": 0,
    "px": 0.0, "py": -0.05, "pz": -0.15,
    "depth_b64": "",
    "ready": False,
    "needs_update": True,
    "do_step": False,
}
state_lock = threading.Lock()

HTML_PAGE = """
<!DOCTYPE html>
<html>
<head>
<title>Camera Orientation Finder</title>
<style>
  body { font-family: monospace; background: #1a1a1a; color: #eee; margin: 20px; }
  .container { display: flex; gap: 30px; }
  .controls { min-width: 350px; }
  .image-box { text-align: center; }
  .image-box img { border: 2px solid #555; image-rendering: pixelated; width: 512px; height: 512px; }
  label { display: block; margin-top: 12px; font-size: 14px; }
  input[type=range] { width: 300px; }
  .val { color: #0f0; font-weight: bold; }
  button { margin-top: 15px; padding: 8px 20px; font-size: 14px; cursor: pointer;
           background: #333; color: #eee; border: 1px solid #666; margin-right: 10px; }
  button:hover { background: #555; }
  .config { margin-top: 20px; background: #222; padding: 10px; border: 1px solid #444;
            font-size: 12px; white-space: pre; }
  h2 { color: #8af; }
  hr { border-color: #444; }
</style>
</head>
<body>
<h2>Interactive Camera Orientation Finder</h2>
<div class="container">
  <div class="controls">
    <h3>Rotation (degrees)</h3>
    <label>rot_angle (Z): <span class="val" id="rot_val">0</span></label>
    <input type="range" id="rot" min="-180" max="180" step="1" value="0">

    <label>tilt_x (X): <span class="val" id="tx_val">0</span></label>
    <input type="range" id="tx" min="-45" max="45" step="1" value="0">

    <label>tilt_y (Y): <span class="val" id="ty_val">0</span></label>
    <input type="range" id="ty" min="-45" max="45" step="1" value="0">

    <hr>
    <h3>Position (meters)</h3>
    <label>pos_x: <span class="val" id="px_val">0.00</span></label>
    <input type="range" id="px" min="-0.20" max="0.20" step="0.01" value="0.00">

    <label>pos_y: <span class="val" id="py_val">-0.05</span></label>
    <input type="range" id="py" min="-0.20" max="0.20" step="0.01" value="-0.05">

    <label>pos_z: <span class="val" id="pz_val">-0.15</span></label>
    <input type="range" id="pz" min="-0.30" max="0.10" step="0.01" value="-0.15">

    <hr>
    <button onclick="stepSim()">Step Sim (random action)</button>
    <button onclick="copyConfig()">Copy Config</button>

    <div class="config" id="config_box">Adjust sliders to see config...</div>
  </div>
  <div class="image-box">
    <h3>64x64 (downsampled)</h3>
    <img id="depth_img_64" src="" alt="Waiting..." style="width:256px;height:256px;">
  </div>
  <div class="image-box">
    <h3>224x224 (rendered)</h3>
    <img id="depth_img_224" src="" alt="Waiting..." style="width:448px;height:448px;">
    <p id="status">Initializing...</p>
  </div>
</div>

<script>
const sliders = ['rot', 'tx', 'ty', 'px', 'py', 'pz'];
let debounceTimer = null;

sliders.forEach(name => {
  const el = document.getElementById(name);
  el.addEventListener('input', () => {
    document.getElementById(name + '_val').textContent = el.value;
    updateConfig();
    clearTimeout(debounceTimer);
    debounceTimer = setTimeout(sendUpdate, 200);
  });
});

function sendUpdate() {
  const params = {};
  sliders.forEach(name => { params[name] = parseFloat(document.getElementById(name).value); });
  fetch('/update', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(params)
  });
}

function stepSim() {
  fetch('/step', {method: 'POST'});
}

function updateConfig() {
  const rot = document.getElementById('rot').value;
  const tx = document.getElementById('tx').value;
  const ty = document.getElementById('ty').value;
  const px = document.getElementById('px').value;
  const py = document.getElementById('py').value;
  const pz = document.getElementById('pz').value;
  document.getElementById('config_box').textContent =
    `generate_gif args:\\n  --rot_angle ${rot} --tilt_x ${tx} --tilt_y ${ty}\\n  --pos "${px},${py},${pz}"\\n\\nConfig pos/rot will appear after first render.`;
}

function pollImage() {
  fetch('/image')
    .then(r => r.json())
    .then(data => {
      if (data.img64) {
        document.getElementById('depth_img_64').src = 'data:image/png;base64,' + data.img64;
        document.getElementById('depth_img_224').src = 'data:image/png;base64,' + data.img224;
        document.getElementById('status').textContent = 'Live';
        if (data.config) {
          document.getElementById('config_box').textContent = data.config;
        }
      }
      setTimeout(pollImage, 300);
    })
    .catch(() => setTimeout(pollImage, 1000));
}
pollImage();
</script>
</body>
</html>
"""


def run_flask(port):
    from flask import Flask, request, jsonify
    app = Flask(__name__)
    app.logger.disabled = True
    import logging
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.ERROR)

    @app.route('/')
    def index():
        return HTML_PAGE

    @app.route('/update', methods=['POST'])
    def update():
        data = request.json
        with state_lock:
            state["rot"] = data.get("rot", 0)
            state["tx"] = data.get("tx", 0)
            state["ty"] = data.get("ty", 0)
            state["px"] = data.get("px", 0.0)
            state["py"] = data.get("py", -0.05)
            state["pz"] = data.get("pz", -0.15)
            state["needs_update"] = True
        return jsonify({"ok": True})

    @app.route('/step', methods=['POST'])
    def step():
        with state_lock:
            state["do_step"] = True
        return jsonify({"ok": True})

    @app.route('/image')
    def image():
        with state_lock:
            return jsonify({"img64": state.get("depth_b64_64", ""), "img224": state.get("depth_b64_224", ""), "config": state.get("config_str", "")})

    app.run(host='0.0.0.0', port=port, threaded=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_depth", action="store_true")
    parser.add_argument("--headless", action="store_true", default=True)
    parser.add_argument("--enable_cameras", action="store_true")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--port", type=int, default=8012, help="Web UI port")
    args, _ = parser.parse_known_args()

    from isaaclab.app import AppLauncher
    app_launcher = AppLauncher(
        headless=args.headless,
        enable_cameras=args.enable_cameras or args.use_depth,
    )
    simulation_app = app_launcher.app

    import gymnasium
    import numpy as np
    from scipy.spatial.transform import Rotation as R
    from PIL import Image
    torch.set_float32_matmul_precision("high")

    import isaaclab_envs  # noqa: F401
    from isaaclab_envs.sim_tool_real_cfg import SimToolRealEnvCfg
    from rl.vec_env import IsaacLabVecEnv

    env_cfg = SimToolRealEnvCfg()
    env_cfg.sim.device = "cuda:0"
    env_cfg.scene.num_envs = 1
    env_cfg.use_depth_camera = True
    # Render at 224x224 for better visualization
    env_cfg.tiled_camera.width = 224
    env_cfg.tiled_camera.height = 224

    base_wxyz = env_cfg.tiled_camera.offset.rot
    base_pos = env_cfg.tiled_camera.offset.pos

    print(f"Base camera rot (wxyz): {base_wxyz}", flush=True)
    print(f"Base camera pos: {base_pos}", flush=True)

    with state_lock:
        state["px"] = base_pos[0]
        state["py"] = base_pos[1]
        state["pz"] = base_pos[2]

    env = gymnasium.make("SimToolReal-Direct-v0", cfg=env_cfg)
    vec_env = IsaacLabVecEnv(env)
    obs = vec_env.reset()

    if "depth" not in obs:
        print("ERROR: no depth!", flush=True)
        simulation_app.close()
        return

    # Start Flask in background thread
    flask_thread = threading.Thread(target=run_flask, args=(args.port,), daemon=True)
    flask_thread.start()
    print(f"\n{'='*60}", flush=True)
    print(f"  Web UI running at: http://localhost:{args.port}", flush=True)
    print(f"  Adjust sliders to change camera orientation", flush=True)
    print(f"{'='*60}\n", flush=True)

    def update_camera(rz, tx, ty, px, py, pz):
        """Update camera transform via USD."""
        base_rot = R.from_quat([base_wxyz[1], base_wxyz[2], base_wxyz[3], base_wxyz[0]])
        combined = base_rot * R.from_euler('z', rz, degrees=True) \
                             * R.from_euler('x', tx, degrees=True) \
                             * R.from_euler('y', ty, degrees=True)
        q = combined.as_quat()  # xyzw
        qwxyz = (q[3], q[0], q[1], q[2])

        from pxr import UsdGeom, Gf
        import omni.usd
        stage = omni.usd.get_context().get_stage()
        cam_prim = stage.GetPrimAtPath("/World/envs/env_0/Robot/iiwa14_link_6/DepthCamera")
        if cam_prim.IsValid():
            xformable = UsdGeom.Xformable(cam_prim)
            # Reuse existing xform ops to avoid precision mismatch
            ops = xformable.GetOrderedXformOps()
            if ops:
                for op in ops:
                    if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                        op.Set(Gf.Vec3d(px, py, pz))
                    elif op.GetOpType() == UsdGeom.XformOp.TypeOrient:
                        op.Set(Gf.Quatd(float(qwxyz[0]), float(qwxyz[1]), float(qwxyz[2]), float(qwxyz[3])))
            else:
                xformable.AddTranslateOp().Set(Gf.Vec3d(px, py, pz))
                xformable.AddOrientOp(precision=UsdGeom.XformOp.PrecisionFloat).Set(
                    Gf.Quatf(float(qwxyz[0]), float(qwxyz[1]), float(qwxyz[2]), float(qwxyz[3])))

        config_str = (
            f"rot=({qwxyz[0]:.4f}, {qwxyz[1]:.4f}, {qwxyz[2]:.4f}, {qwxyz[3]:.4f}),\n"
            f"pos=({px:.2f}, {py:.2f}, {pz:.2f}),\n\n"
            f"generate_gif args:\n"
            f"  --rot_angle {int(rz)} --tilt_x {int(tx)} --tilt_y {int(ty)}\n"
            f"  --pos \"{px},{py},{pz}\""
        )
        return config_str

    def get_depth_b64(obs_dict):
        """Convert depth obs to base64 PNGs: 64x64 (downsampled) and 224x224 (native render)."""
        if "depth" in obs_dict:
            depth = obs_dict["depth"][0, 0].cpu().numpy()
            img = (depth * 255).astype(np.uint8)
            pil_img = Image.fromarray(img)  # 224x224 native

            # 64x64 (downsampled from 224x224, what training would see)
            buf64 = io.BytesIO()
            pil_img.resize((64, 64), Image.BILINEAR).save(buf64, format='PNG')
            b64_64 = base64.b64encode(buf64.getvalue()).decode('utf-8')

            # 224x224 (native render resolution)
            buf224 = io.BytesIO()
            pil_img.save(buf224, format='PNG')
            b64_224 = base64.b64encode(buf224.getvalue()).decode('utf-8')

            return b64_64, b64_224
        return "", ""

    # Initial render
    actions = torch.zeros(1, vec_env.num_actions, device=vec_env.device)
    obs, _, _, _ = vec_env.step(actions)
    with state_lock:
        state["depth_b64_64"], state["depth_b64_224"] = get_depth_b64(obs)
        state["ready"] = True

    print("Running interactive loop. Press Ctrl+C to stop.", flush=True)
    try:
        while True:
            with state_lock:
                needs_update = state["needs_update"]
                do_step = state["do_step"]
                rz = state["rot"]
                tx = state["tx"]
                ty = state["ty"]
                px = state["px"]
                py = state["py"]
                pz = state["pz"]
                state["needs_update"] = False
                state["do_step"] = False

            if needs_update:
                config_str = update_camera(rz, tx, ty, px, py, pz)
                actions = torch.zeros(1, vec_env.num_actions, device=vec_env.device)
                obs, _, _, _ = vec_env.step(actions)
                b64_64, b64_224 = get_depth_b64(obs)
                with state_lock:
                    state["depth_b64_64"] = b64_64
                    state["depth_b64_224"] = b64_224
                    state["config_str"] = config_str

            if do_step:
                actions = 0.5 * torch.randn(1, vec_env.num_actions, device=vec_env.device)
                obs, _, _, _ = vec_env.step(actions)
                b64_64, b64_224 = get_depth_b64(obs)
                with state_lock:
                    state["depth_b64_64"] = b64_64
                    state["depth_b64_224"] = b64_224

            time.sleep(0.05)
    except KeyboardInterrupt:
        print("\nStopping...", flush=True)

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
