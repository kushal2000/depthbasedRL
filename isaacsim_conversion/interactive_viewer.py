from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any


def _json_script_payload(data: dict[str, Any]) -> str:
    # Avoid literal "</script>" inside the embedded JSON.
    return json.dumps(data, separators=(",", ":")).replace("</", "<\\/")


def create_pose_viewer_html(payload: dict[str, Any], *, title: str = "Isaac Sim rollout") -> str:
    payload_json = _json_script_payload(payload)
    safe_title = html.escape(title)
    return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{safe_title}</title>
  <style>
    html, body {{ margin: 0; width: 100%; height: 100%; overflow: hidden; background: #11151c; color: #e8eef8; font-family: sans-serif; }}
    #viewer {{ position: absolute; inset: 0 0 76px 0; }}
    #hud {{ position: absolute; left: 12px; top: 10px; padding: 8px 10px; background: rgba(0,0,0,0.55); border-radius: 8px; font-size: 13px; line-height: 1.35; }}
    #controls {{ position: absolute; left: 0; right: 0; bottom: 0; height: 76px; display: flex; gap: 14px; align-items: center; padding: 0 16px; box-sizing: border-box; background: rgba(4,6,10,0.92); }}
    #frame {{ flex: 1; }}
    button {{ background: #e8eef8; border: 0; border-radius: 7px; padding: 8px 12px; cursor: pointer; }}
    .legend {{ display: flex; gap: 10px; flex-wrap: wrap; }}
    .swatch {{ display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 4px; }}
  </style>
</head>
<body>
  <div id="viewer"></div>
  <div id="hud"></div>
  <div id="controls">
    <button id="play">Pause</button>
    <input id="frame" type="range" min="0" max="0" value="0" step="1" />
    <div id="readout"></div>
    <div class="legend">
      <span><span class="swatch" style="background:#56a3ff"></span>robot bodies</span>
      <span><span class="swatch" style="background:#ffad4d"></span>object</span>
      <span><span class="swatch" style="background:#44d17a"></span>goal</span>
      <span><span class="swatch" style="background:#9aa2ad"></span>table</span>
    </div>
  </div>
  <script id="rollout-data" type="application/json">{payload_json}</script>
  <script src="https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.min.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/three@0.160.0/examples/js/controls/OrbitControls.js"></script>
  <script>
    const data = JSON.parse(document.getElementById("rollout-data").textContent);
    const T = data.timestamps.length;
    const viewer = document.getElementById("viewer");
    const hud = document.getElementById("hud");
    const slider = document.getElementById("frame");
    const playButton = document.getElementById("play");
    const readout = document.getElementById("readout");
    slider.max = Math.max(T - 1, 0);

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x11151c);
    const camera = new THREE.PerspectiveCamera(45, viewer.clientWidth / viewer.clientHeight, 0.01, 50);
    camera.position.set(1.15, -1.65, 1.15);
    camera.up.set(0, 0, 1);
    const renderer = new THREE.WebGLRenderer({{ antialias: true }});
    renderer.setPixelRatio(window.devicePixelRatio || 1);
    renderer.setSize(viewer.clientWidth, viewer.clientHeight);
    viewer.appendChild(renderer.domElement);
    const controls = THREE.OrbitControls
      ? new THREE.OrbitControls(camera, renderer.domElement)
      : {{ target: new THREE.Vector3(0.0, 0.35, 0.55), update: () => {{}} }};
    controls.target.set(0.0, 0.35, 0.55);
    camera.lookAt(controls.target);
    controls.update();

    scene.add(new THREE.HemisphereLight(0xffffff, 0x334455, 1.7));
    const dirLight = new THREE.DirectionalLight(0xffffff, 1.3);
    dirLight.position.set(2, -3, 4);
    scene.add(dirLight);
    const grid = new THREE.GridHelper(2.0, 20, 0x385064, 0x25313f);
    grid.rotation.x = Math.PI / 2;
    scene.add(grid);

    const bodyGeom = new THREE.SphereGeometry(0.013, 12, 8);
    const bodyMat = new THREE.MeshStandardMaterial({{ color: 0x56a3ff, roughness: 0.45 }});
    const palmMat = new THREE.MeshStandardMaterial({{ color: 0xf4e45b, roughness: 0.45 }});
    const bodyMeshes = data.robot_body_names.map((name) => {{
      const mesh = new THREE.Mesh(bodyGeom, name.includes("palm") || name.includes("link_7") ? palmMat : bodyMat);
      scene.add(mesh);
      return mesh;
    }});

    const makeBox = (color, opacity, scale) => {{
      const mat = new THREE.MeshStandardMaterial({{ color, transparent: opacity < 1.0, opacity, roughness: 0.5 }});
      const mesh = new THREE.Mesh(new THREE.BoxGeometry(scale[0], scale[1], scale[2]), mat);
      scene.add(mesh);
      return mesh;
    }};
    const objectMesh = makeBox(0xffad4d, 0.92, [0.18, 0.045, 0.045]);
    const goalMesh = makeBox(0x44d17a, 0.38, [0.18, 0.045, 0.045]);
    const tableMesh = makeBox(0x9aa2ad, 0.42, [0.85, 0.65, 0.045]);

    const axes = new THREE.AxesHelper(0.12);
    scene.add(axes);

    function setPose(mesh, pose) {{
      mesh.position.set(pose[0], pose[1], pose[2]);
      mesh.quaternion.set(pose[3], pose[4], pose[5], pose[6]);
    }}

    function updateFrame(i) {{
      i = Math.max(0, Math.min(T - 1, i));
      slider.value = i;
      const bodies = data.robot_body_positions[i];
      for (let j = 0; j < bodyMeshes.length; j++) {{
        const p = bodies[j];
        bodyMeshes[j].position.set(p[0], p[1], p[2]);
      }}
      setPose(objectMesh, data.object_poses[i]);
      setPose(goalMesh, data.goal_poses[i]);
      setPose(tableMesh, data.table_poses[i]);
      axes.position.copy(objectMesh.position);
      axes.quaternion.copy(objectMesh.quaternion);
      readout.textContent = `frame ${{i + 1}}/${{T}}  t=${{data.timestamps[i].toFixed(2)}}s`;
      hud.innerHTML = `<b>${{data.title}}</b><br/>mode: ${{data.mode}}<br/>env: ${{data.env_id}}<br/>goal_idx: ${{data.goal_idx[i]}} / ${{data.num_goals}}<br/>kp_dist: ${{data.kp_dist[i].toFixed(4)}}`;
    }}

    let current = 0;
    let playing = true;
    slider.addEventListener("input", () => {{ current = Number(slider.value); updateFrame(current); }});
    playButton.addEventListener("click", () => {{
      playing = !playing;
      playButton.textContent = playing ? "Pause" : "Play";
    }});
    window.addEventListener("resize", () => {{
      camera.aspect = viewer.clientWidth / viewer.clientHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(viewer.clientWidth, viewer.clientHeight);
    }});

    function animate() {{
      requestAnimationFrame(animate);
      if (playing && T > 0) {{
        current = (current + 1) % T;
        updateFrame(current);
      }}
      controls.update();
      renderer.render(scene, camera);
    }}
    updateFrame(0);
    animate();
  </script>
</body>
</html>
"""


def write_pose_viewer_html(path: Path, payload: dict[str, Any], *, title: str) -> str:
    html_text = create_pose_viewer_html(payload, title=title)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html_text, encoding="utf-8")
    return html_text
