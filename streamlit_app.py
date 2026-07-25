"""DroneGym — live web demo (Streamlit).

Configure a drone, pick a trained PPO checkpoint, fly it live against a target,
and browse recorded runs. The live flight is computed server-side by the exact
training-time EpisodeRunner (dronegym.runner) and then handed to Plotly, which
animates + scrubs it in the browser — so playback stays smooth even though
Streamlit re-runs the script on every interaction.

    streamlit run streamlit_app.py

Deploy: Streamlit Community Cloud, main file streamlit_app.py. The trained model
and seed replays are committed under assets/ so live inference works on the host.
"""

import os

import numpy as np
import streamlit as st

import webviz
from dronegym import runner

st.set_page_config(page_title="DroneGym — live RL demo", page_icon="🚁",
                   layout="wide")

ROOT = webviz.ROOT
DEFAULT_MODEL = "assets/model/best_model.zip"


# --- cached heavy resources --------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_policy(model_path):
    """SB3 checkpoint -> policy fn. Cached per path so torch loads once."""
    return runner.load_policy(model_path)


@st.cache_data(show_spinner=False)
def presets():
    return webviz.load_presets()


def all_runs():
    return webviz.load_runs()


# --- config from the sidebar -------------------------------------------------
def sidebar_config():
    st.sidebar.header("🚁 DRONE")
    mode = st.sidebar.radio("configure", ["Preset", "Custom"], horizontal=True,
                            label_visibility="collapsed")
    pdict = presets()
    names = list(pdict) or ["(none)"]
    default_i = names.index("freestyle_5inch") if "freestyle_5inch" in names else 0

    if mode == "Preset":
        name = st.sidebar.selectbox("airframe preset", names, index=default_i)
        cfg = dict(pdict[name])
    else:
        base = pdict.get("freestyle_5inch", next(iter(pdict.values()), {}))
        c1, c2 = st.sidebar.columns(2)
        cfg = {
            "name": "custom",
            "mass_g": c1.slider("mass (g)", 20, 3000, int(base.get("mass_g", 650))),
            "prop_diameter_in": c2.slider("prop (in)", 1.0, 12.0,
                                          float(base.get("prop_diameter_in", 5.1)), 0.1),
            "motor_kv": c1.slider("motor KV", 700, 26000, int(base.get("motor_kv", 1850))),
            "cam_angle_deg": c2.slider("cam uptilt (deg)", 0, 45,
                                       int(base.get("cam_angle_deg", 30))),
            "frame_size_mm": c1.slider("frame (mm)", 60, 500,
                                       int(base.get("frame_size_mm", 220))),
        }
        batt = c2.selectbox("battery", list(webviz.BATTERY),
                            index=list(webviz.BATTERY).index("6S (22.2V)"))
        cfg["battery_v"] = webviz.BATTERY[batt]

    st.sidebar.header("🎯 SCENARIO")
    scen = st.sidebar.radio("scenario", ["Static", "Intercept"], horizontal=True,
                            label_visibility="collapsed")
    tspeed = None
    if scen == "Intercept":
        tspeed = st.sidebar.slider("target speed (m/s)", 2.0, 10.0, 5.0, 0.5)

    st.sidebar.header("🧠 MODEL")
    models = webviz.scan_models()
    labels = list(models) or ["(no checkpoints found)"]
    idx = labels.index(DEFAULT_MODEL) if DEFAULT_MODEL in labels else 0
    mlabel = st.sidebar.selectbox("checkpoint", labels, index=idx)
    model_path = models.get(mlabel)

    return cfg, scen.lower(), tspeed, model_path, mlabel


def run_live(cfg, scenario, tspeed, model_path, mlabel):
    if not model_path or not os.path.isfile(model_path):
        st.sidebar.error("no valid checkpoint selected")
        return None
    try:
        policy = get_policy(model_path)
    except RuntimeError as e:
        st.sidebar.error(str(e))
        return None
    ep = runner.EpisodeRunner(
        cfg, policy, scenario=scenario,
        target_speed=tspeed if scenario == "intercept" else None,
        model_name=os.path.basename(mlabel))
    ep.run()
    return ep.to_dict()


# --- outcome styling ---------------------------------------------------------
OUTCOME_COLOR = {"hit": "#50ff78", "timeout": "#ffb84d", "lost": "#ff6a6a",
                 "crash": "#ff6a6a", "oob": "#ff6a6a", "tumble": "#ff6a6a",
                 "escaped": "#ffb84d"}


def metrics_row(run):
    outcome = run.get("outcome", "?")
    color = OUTCOME_COLOR.get(outcome, "#c8cdd7")
    c = st.columns([1.4, 1, 1, 1, 1])
    c[0].markdown(
        f"**Outcome**<br><span style='font-size:1.5rem;color:{color}'>"
        f"{outcome.upper()}</span>", unsafe_allow_html=True)
    tth = run.get("time_to_hit")
    c[1].metric("time to hit", f"{tth:.2f}s" if tth else "—")
    c[2].metric("closest", f"{run.get('closest_approach', 0):.2f} m")
    c[3].metric("reward", f"{run.get('total_reward', 0):.0f}")
    spd = run.get("target_speed") or 0
    c[4].metric("target", f"{spd:.1f} m/s" if spd else "static")


# --- main --------------------------------------------------------------------
def main():
    cfg, scenario, tspeed, model_path, mlabel = sidebar_config()

    if st.sidebar.button("🚀 RUN LIVE", type="primary", width="stretch"):
        with st.spinner("flying episode on the policy…"):
            run = run_live(cfg, scenario, tspeed, model_path, mlabel)
        if run is not None:
            st.session_state["run"] = run
            st.session_state["src"] = f"LIVE · {cfg.get('name', 'custom')} · {os.path.basename(mlabel)}"

    st.title("DroneGym — live RL flight")
    st.caption("A PPO policy trained in a NumPy quadrotor sim, flown live in your "
               "browser. Configure a drone on the left and hit **RUN LIVE**, or replay a recorded run below.")

    # default view: best recorded run, so the page is never empty
    runs = all_runs()
    if "run" not in st.session_state and runs:
        best = runner.rank_runs(runs, "Fastest hit")[0]
        st.session_state["run"] = best
        st.session_state["src"] = f"replay · {best.get('_file', '')}"

    run = st.session_state.get("run")
    if run:
        st.markdown(f"<span style='color:#7a8091'>showing: "
                    f"{st.session_state.get('src', '')}</span>",
                    unsafe_allow_html=True)
        metrics_row(run)
        st.plotly_chart(webviz.build_figure(run), width="stretch",
                        config={"displayModeBar": False})
    else:
        st.info("No runs yet — configure a drone and hit **RUN LIVE**.")

    # --- replay browser ---
    st.divider()
    st.subheader("📼 Recorded runs")
    if not runs:
        st.write("_none recorded yet_")
    else:
        rank = st.radio("rank by", ["Fastest hit", "Total reward"],
                        horizontal=True, label_visibility="collapsed")
        ranked = runner.rank_runs(runs, rank)
        labels = [webviz.run_label(r, i) for i, r in enumerate(ranked, 1)]
        pick = st.selectbox("run", labels, label_visibility="collapsed")
        if st.button("▶ Load replay", width="stretch"):
            chosen = ranked[labels.index(pick)]
            st.session_state["run"] = chosen
            st.session_state["src"] = f"replay · {chosen.get('_file', '')}"
            st.rerun()

    with st.expander("How this works"):
        st.markdown(
            "- **Live flight** is computed by `dronegym.runner.EpisodeRunner` — the "
            "same env used in training — then animated client-side by Plotly.\n"
            "- The policy is an SB3 PPO checkpoint (`assets/model/best_model.zip`), "
            "loaded once and cached.\n"
            "- **WORLD** is a 3D view (drag to orbit): cyan trail, red rotor arms, "
            "yellow nose, orange target. **FPV CAMERA** shows the target's bounding "
            "box in the drone's-eye frame.\n"
            "- The policy was trained on `freestyle_5inch`; other airframes show how "
            "it transfers (imperfectly) to different physics.")


if __name__ == "__main__":
    main()
