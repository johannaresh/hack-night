"""FastAPI backend for the DroneGym web app.

Serves the static canvas frontend in web/ and a small JSON API:

    GET  /api/meta      -> constants the frontend needs (max body rate, batteries)
    GET  /api/presets   -> {name: cfg} airframe presets
    GET  /api/models    -> {models: [...], default: "..."} checkpoints found
    GET  /api/runs      -> [run, ...] recorded flights (assets/replays + runs/)
    POST /api/run       -> run one live episode on a checkpoint, return the run

A live run is computed by the training-time dronegym.runner.EpisodeRunner (the
same env used in training) and returned whole; the browser animates it on a
canvas. The policy is loaded once per checkpoint (lru_cache).

    uvicorn server:app --port 8000        # dev
    uvicorn server:app --host 0.0.0.0 --port $PORT   # prod
"""

import asyncio
import functools
import glob
import math
import os
import threading
import time
from typing import Optional

import yaml
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from dronegym import runner

ROOT = os.path.dirname(os.path.abspath(__file__))
WEB_DIR = os.path.join(ROOT, "web")

DEFAULT_MODEL = "assets/model/best_model.zip"
MODEL_GLOBS = ("assets/model/*.zip", "models/*.zip", "checkpoints/*.zip",
               "runs/*/best_model.zip", "runs/*/ckpt/*.zip")
RUN_DIRS = ("assets/replays", "runs")

# gui.py: MAX_RATE_DEG = degrees(DroneConfig.max_body_rate default = 14.0 rad/s)
MAX_RATE_DEG = math.degrees(14.0)
BATTERY = {"1S (3.7V)": 3.7, "2S (7.4V)": 7.4, "3S (11.1V)": 11.1,
           "4S (14.8V)": 14.8, "6S (22.2V)": 22.2}


# --- data helpers ------------------------------------------------------------
def load_presets():
    out = {}
    for p in sorted(glob.glob(os.path.join(ROOT, "configs", "*.yaml"))):
        try:
            with open(p) as f:
                cfg = yaml.safe_load(f)
        except (yaml.YAMLError, OSError):
            continue
        if isinstance(cfg, dict) and "mass_g" in cfg:
            out[cfg.get("name", os.path.basename(p)[:-5])] = cfg
    return out


def scan_models():
    out = {}
    for pat in MODEL_GLOBS:
        for p in sorted(glob.glob(os.path.join(ROOT, pat))):
            out[os.path.relpath(p, ROOT).replace("\\", "/")] = p
    return out


def load_runs():
    seen, runs = set(), []
    for d in RUN_DIRS:
        for r in runner.load_runs(os.path.join(ROOT, d)):
            if r.get("_file") in seen:
                continue
            seen.add(r.get("_file"))
            runs.append(r)
    return runs


@functools.lru_cache(maxsize=8)
def get_policy(model_path):
    """SB3 checkpoint -> policy fn, cached per path so torch loads once."""
    return runner.load_policy(model_path)


# --- api ---------------------------------------------------------------------
app = FastAPI(title="DroneGym")


@app.get("/api/meta")
def api_meta():
    return {"maxRate": MAX_RATE_DEG, "battery": BATTERY, "defaultModel": DEFAULT_MODEL}


@app.get("/api/presets")
def api_presets():
    return load_presets()


@app.get("/api/models")
def api_models():
    models = list(scan_models())
    default = DEFAULT_MODEL if DEFAULT_MODEL in models else (models[0] if models else None)
    return {"models": models, "default": default}


@app.get("/api/runs")
def api_runs():
    return load_runs()


class RunReq(BaseModel):
    cfg: dict
    scenario: str = "static"
    target_speed: Optional[float] = None
    model: str = DEFAULT_MODEL


@app.post("/api/run")
def api_run(req: RunReq):
    path = scan_models().get(req.model)
    if not path or not os.path.isfile(path):
        raise HTTPException(404, f"checkpoint not found: {req.model}")
    try:
        policy = get_policy(path)
    except RuntimeError as e:
        raise HTTPException(400, str(e))
    intercept = req.scenario == "intercept"
    try:
        ep = runner.EpisodeRunner(
            req.cfg, policy, scenario="intercept" if intercept else "static",
            target_speed=req.target_speed if intercept else None,
            model_name=os.path.basename(req.model))
        ep.run()
    except (KeyError, ValueError, TypeError) as e:
        raise HTTPException(400, f"bad drone config: {e}")
    try:
        ep.save(os.path.join(ROOT, "runs"))     # so it shows in the replay browser
    except OSError:
        pass
    return ep.to_dict()


# --- live training (streamed over WebSocket) --------------------------------
# One training run at a time, in a background thread; per-rollout metrics are
# pushed to the browser so you can watch reward climb and the curriculum promote.
_train_lock = threading.Lock()
_train_seq = 0


def _train_thread(steps, scenario, emit, stop_event):
    """Run a short PPO training on freestyle_5inch, emitting metrics per rollout.

    Reuses train.py's env stack, model, and curriculum so this is the real
    training loop, just smaller (8 envs) and streamed. Saves the checkpoint into
    models/ so it immediately shows up in the checkpoint picker.
    """
    global _train_seq
    import torch
    import train as T
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.utils import safe_mean

    torch.set_num_threads(1)                         # train.py: ~+12% on this tiny net
    n_envs = 8
    venv = T.make_vec_env("freestyle_5inch", n_envs=n_envs, difficulty=0, seed=0)
    model = T.build_model(venv, n_envs, seed=0, tensorboard_log=None)
    curr = T.CurriculumCallback(eval_env=None)

    class Stream(BaseCallback):
        def __init__(self):
            super().__init__()
            self._t0 = None
            self._iter = 0

        def _on_training_start(self):
            self._t0 = time.perf_counter()
            emit({"type": "start", "total": int(steps), "n_envs": n_envs})

        def _on_step(self):
            return not stop_event.is_set()          # SB3 aborts when this is False

        def _on_rollout_end(self):
            self._iter += 1
            buf = self.model.ep_info_buffer
            rew = safe_mean([e["r"] for e in buf]) if buf else 0.0
            eplen = safe_mean([e["l"] for e in buf]) if buf else 0.0
            elapsed = time.perf_counter() - self._t0
            emit({"type": "metric", "iter": self._iter,
                  "steps": int(self.num_timesteps),
                  "fps": round(self.num_timesteps / max(elapsed, 1e-9)),
                  "reward": round(float(rew), 2), "eplen": round(float(eplen), 1),
                  "level": curr.level, "success": round(curr.success_rate, 3),
                  "elapsed": round(elapsed, 1)})

    model.learn(total_timesteps=int(steps), callback=[curr, Stream()])

    _train_seq += 1
    os.makedirs(os.path.join(ROOT, "models"), exist_ok=True)
    rel = f"models/trained_{int(steps / 1000)}k_{_train_seq}.zip"
    model.save(os.path.join(ROOT, rel))
    venv.close()
    get_policy.cache_clear()                        # new checkpoint is now loadable
    emit({"type": "done", "model": rel, "aborted": stop_event.is_set()})


@app.websocket("/ws/train")
async def ws_train(ws: WebSocket):
    """Client sends {steps, scenario}; server streams start/metric/done frames.
    Closing the socket aborts the run."""
    await ws.accept()
    try:
        params = await ws.receive_json()
    except Exception:
        await ws.close()
        return
    steps = max(4096, min(int(params.get("steps", 50000)), 300000))
    scenario = params.get("scenario", "static")

    if not _train_lock.acquire(blocking=False):
        await ws.send_json({"type": "error", "message": "a training run is already in progress"})
        await ws.close()
        return

    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()
    stop_event = threading.Event()

    def emit(m):
        loop.call_soon_threadsafe(queue.put_nowait, m)

    def run():
        try:
            _train_thread(steps, scenario, emit, stop_event)
        except Exception as e:                      # never leave the socket hanging
            emit({"type": "error", "message": f"{type(e).__name__}: {e}"})
        finally:
            emit({"type": "__end__"})
            _train_lock.release()

    threading.Thread(target=run, daemon=True).start()
    try:
        while True:
            m = await queue.get()
            if m.get("type") == "__end__":
                break
            await ws.send_json(m)
    except WebSocketDisconnect:
        pass
    finally:
        stop_event.set()                            # abort training if still running
        try:
            await ws.close()
        except RuntimeError:
            pass


# Static frontend, mounted last so the /api/* routes above win. html=True serves
# web/index.html at "/".
app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")
