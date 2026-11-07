"""User-friendly Streamlit dashboard for NEOM Urban Agent Network (Python only)."""
import os
import time

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

load_dotenv()
from neom.model import CitySimulation, Config, PERFORMATIVES
from neom.llm import configured as llm_configured
from neom.prediction import trained_model

st.set_page_config(page_title="NEOM Urban Agent Network", page_icon="🏙️", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    .block-container {padding-top: 1.4rem; max-width: 1500px;}
    h1 {letter-spacing: -.035em; margin-bottom: .2rem;}
    h2, h3 {letter-spacing: -.02em;}
    [data-testid="stMetric"] {background: var(--secondary-background-color); border: 1px solid rgba(130,145,160,.18); border-radius: 14px; padding: 14px 16px;}
    [data-testid="stMetricLabel"] {font-weight: 600;}
    div[data-testid="stTabs"] button {font-weight: 650;}
    [data-testid="stSidebar"] {border-right: 1px solid rgba(130,145,160,.18);}
    .section-intro {opacity: .75; font-size: .92rem; margin-top: -.5rem; margin-bottom: 1rem;}
    .tiny-note {opacity: .70; font-size: .80rem;}
</style>
""", unsafe_allow_html=True)

SCENARIOS = {
    "⚡ EV charging & congestion": "charging",
    "♻️ Waste collection": "waste",
    "🛡️ Energy failure & recovery": "failure",
}
MODELS = ["openai/gpt-oss-20b", "llama-3.3-70b-versatile"]

if "scenario" not in st.session_state:
    st.session_state.scenario = "charging"
if "sim" not in st.session_state:
    st.session_state.sim = CitySimulation(st.session_state.scenario)
sim = st.session_state.sim


def reset_sim():
    config = Config(**vars(st.session_state.sim.config))
    st.session_state.sim = CitySimulation(st.session_state.scenario, config)


def run_steps(steps: int, delay: float = 0.0):
    current = st.session_state.sim
    steps = min(steps, max(0, current.config.max_steps - current.step_number))
    if steps <= 0:
        return
    with st.spinner("Agents are coordinating…"):
        bar = st.progress(0, text="Simulation running")
        for i in range(steps):
            current.step()
            bar.progress((i + 1) / steps, text=f"Step {current.step_number} / {current.config.max_steps}")
            if delay:
                time.sleep(delay)
        bar.empty()


with st.sidebar:
    st.markdown("### 🎛️ Control center")
    picked = st.selectbox("Choose scenario", list(SCENARIOS), index=list(SCENARIOS.values()).index(st.session_state.scenario))
    scenario = SCENARIOS[picked]
    if scenario != st.session_state.scenario:
        st.session_state.scenario = scenario
        st.session_state.sim = CitySimulation(scenario)
        st.rerun()
    sim = st.session_state.sim
    cfg = sim.config
    cfg.max_steps = st.slider("Maximum simulation steps", 5, 100, cfg.max_steps)
    delay = st.slider("Delay between steps (seconds)", 0.0, 1.5, 0.0, .05,
                      help="Use 0 for faster runs. API calls to Groq may still take time.")
    a, b = st.columns(2)
    if a.button("▶️ Run all", type="primary", use_container_width=True, disabled=sim.step_number >= cfg.max_steps):
        run_steps(cfg.max_steps - sim.step_number, delay)
        st.rerun()
    if b.button("⏭️ Step", use_container_width=True, disabled=sim.step_number >= cfg.max_steps):
        run_steps(1)
        st.rerun()
    c1, c2 = st.columns(2)
    if c1.button("▶️ Run 5", use_container_width=True, disabled=sim.step_number >= cfg.max_steps):
        run_steps(5, delay)
        st.rerun()
    if c2.button("↻ Reset", use_container_width=True):
        reset_sim()
        st.rerun()
    st.progress(min(sim.step_number / max(1, cfg.max_steps), 1.0), text=f"Progress: {sim.step_number} of {cfg.max_steps} steps")

    with st.expander("⚙️ City parameters", expanded=False):
        cfg.peak_threshold = st.slider("Grid threshold (%)", 40, 100, cfg.peak_threshold)
        cfg.congestion_limit = st.slider("Congestion threshold", 30, 100, cfg.congestion_limit)
        cfg.ev_arrivals = st.slider("EV arrivals / district / step", 0, 15, cfg.ev_arrivals)
        cfg.bin_fill_rate = st.slider("Bin filling (% / step)", 0, 15, cfg.bin_fill_rate)
        cfg.bin_threshold = st.slider("Collection threshold (%)", 40, 100, cfg.bin_threshold)
        cfg.randomness = st.slider("Random EV arrivals (±)", 0, 5, cfg.randomness)
        cfg.heartbeat_timeout = st.slider("Heartbeat timeout (steps)", 1, 10, cfg.heartbeat_timeout)
        cfg.max_delay_steps = st.slider("Max charging delay (steps)", 1, 12, cfg.max_delay_steps)

    with st.expander("🧠 AI configuration", expanded=True):
        cfg.llm_enabled = st.toggle("Groq LLM: energy proposals", value=cfg.llm_enabled)
        cfg.llm_model = st.selectbox("Groq model", MODELS, index=MODELS.index(cfg.llm_model) if cfg.llm_model in MODELS else 0)
        if llm_configured():
            st.success("Groq API key detected", icon="✅")
        else:
            st.warning("Groq API key missing; safe fallback enabled", icon="⚠️")
        cfg.ml_prediction_enabled = st.toggle("Random Forest: peak prediction", value=cfg.ml_prediction_enabled)
        cfg.forecast_horizon = st.slider("Forecast horizon (steps)", 1, 6, cfg.forecast_horizon)
        cfg.trend_window = st.slider("History window", 3, 10, max(3, cfg.trend_window))
        cfg.alert_threshold = st.slider("Peak alert threshold (EVs)", 5, 60, cfg.alert_threshold)

    with st.expander("🛠️ Simulate agent failures", expanded=False):
        st.caption("Try disabling an agent to test heartbeat detection and recovery.")
        for name in ("Energy", "Transportation", "Waste", "Peak Prediction"):
            f, r = st.columns(2)
            if f.button(f"Fail {name}", key=f"fail_{name}", use_container_width=True):
                sim.fail(name)
                st.rerun()
            if r.button(f"Restart {name}", key=f"restart_{name}", use_container_width=True):
                sim.restart(name)
                st.rerun()

    with st.expander("🔎 LangSmith tracing", expanded=False):
        tracing = os.getenv("LANGSMITH_TRACING", "false").lower() == "true" and bool(os.getenv("LANGSMITH_API_KEY"))
        st.success("Tracing configured") if tracing else st.info("Tracing not configured")
        st.caption("Traces appear in LangSmith after recorded calls; this indicator only checks local configuration.")
        st.link_button("Open LangSmith ↗", "https://smith.langchain.com/", use_container_width=True)

st.title("🏙️ NEOM Urban Agent Network")
st.caption("Smart city simulation  /  5 cooperating agents  /  Groq LLM + Random Forest + LangSmith")

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Simulation step", f"{sim.step_number} / {sim.config.max_steps}")
k2.metric("Highest grid load", f"{max(d.grid_load for d in sim.districts.values()):.1f}%")
k3.metric("Resolved conflicts", sim.completed_conflicts)
k4.metric("Safe mode", "🔴 ON" if sim.safe_mode else "🟢 OFF")
k5.metric("Active agents", f"{sum(a.state != 'FAILED' for a in sim.agents.values())} / 5")

if sim.last_llm_result.startswith("LLM fallback:"):
    st.warning(f"⚠️ Energy decision: {sim.last_llm_result}. Simulation continues with a safe heuristic.")
elif sim.last_llm_result.startswith("LLM proposal validated"):
    st.success(f"✅ Energy decision: {sim.last_llm_result}")
else:
    st.info(f"Energy decision source: {sim.last_llm_result}", icon="ℹ️")

main, agents_tab, ai_tab, messages_tab, settings_tab = st.tabs([
    "📊 Overview", "🤝 Agents", "🧠 AI & Forecasting", "💬 Messages & Events", "⚙️ Controls & Diagnostics"
])

with main:
    st.subheader("City health at a glance")
    st.markdown('<div class="section-intro">Live simulated conditions in three city districts. All values are synthetic.</div>', unsafe_allow_html=True)
    district_df = pd.DataFrame(sim.district_rows())
    if not district_df.empty:
        district_df = district_df.set_index("District")
        a, b = st.columns(2, gap="large")
        with a:
            st.markdown("**🚗 Road congestion**")
            st.bar_chart(district_df["Congestion"], height=225)
            st.markdown("**⚡ Grid load (%)**")
            st.bar_chart(district_df["Grid %"], height=225)
        with b:
            st.markdown("**🔋 Waiting EVs**")
            st.bar_chart(district_df["EV queue"], height=225)
            st.markdown("**♻️ Waste bin fill (%)**")
            st.bar_chart(district_df["Bin fill %"], height=225)
        with st.expander("See full district data", expanded=False):
            st.dataframe(district_df, use_container_width=True)
    st.subheader("Simulation history")
    if sim.history:
        history = pd.DataFrame(sim.history).set_index("step")
        choices = [k for k in ("max_grid_load", "max_congestion", "total_queue") if k in history]
        st.line_chart(history[choices], height=300)
        st.caption("Different indicators share one chart for trends; they have different units.")
    else:
        st.info("Start the simulation with **Step** or **Run 5** in the sidebar to display trends.")

with agents_tab:
    st.subheader("Live agent status")
    st.caption("Each agent senses a different service and communicates through ACL-like messages.")
    agents_df = pd.DataFrame(sim.agent_rows())
    if not agents_df.empty:
        st.dataframe(agents_df, use_container_width=True, hide_index=True, height=240)
    st.markdown("**Agent responsibilities**")
    card_items = [
        ("⚡ Energy", "Grid monitoring, charging schedules and Groq LLM negotiation proposals"),
        ("🚙 Transportation", "Traffic monitoring and alternative route proposals"),
        ("♻️ Waste", "Bin monitoring and truck dispatch"),
        ("🧭 Coordinator", "Contract Net negotiation, heartbeat monitoring and safe mode"),
        ("📈 Peak Prediction", "Random Forest queue forecasts and early-warning messages"),
    ]
    for i in range(0, len(card_items), 2):
        cols = st.columns(2)
        for col, (heading, description) in zip(cols, card_items[i:i+2]):
            with col:
                with st.container(border=True):
                    st.markdown(f"**{heading}**")
                    st.caption(description)

with ai_tab:
    st.subheader("AI decisions & peak forecasts")
    ai1, ai2 = st.columns(2)
    with ai1:
        with st.container(border=True):
            st.markdown("#### ⚡ Energy Agent — Groq")
            st.write("**Model:**", cfg.llm_model)
            st.write("**LLM setting:**", "Enabled" if cfg.llm_enabled else "Disabled")
            st.write("**API key:**", "Detected" if llm_configured() else "Missing")
            st.write("**Latest source:**")
            st.code(sim.last_llm_result, language=None)
            st.caption("A validated LLM proposal is evaluated by Coordinator; fallback means the heuristic handled the decision.")
    with ai2:
        with st.container(border=True):
            st.markdown("#### 📈 Peak Prediction — Random Forest")
            st.write("**Prediction mode:**", "Random Forest" if cfg.ml_prediction_enabled else "Trend baseline")
            st.write("**Horizon:**", f"{cfg.forecast_horizon} steps")
            st.write("**Alert threshold:**", f"{cfg.alert_threshold} waiting EVs")
            if cfg.ml_prediction_enabled:
                try:
                    _, mae = trained_model()
                    st.metric("Synthetic holdout MAE", f"{mae:.2f} EVs")
                except Exception as error:
                    st.warning(f"ML unavailable ({type(error).__name__}); the trend fallback remains available.")
            st.caption("The model is trained on generated simulation data, not operational NEOM data.")
    st.markdown("#### Predicted queue vs current queue")
    if sim.peak_predictions:
        forecast_df = pd.DataFrame([{"District": d, "Current EV queue": pred["current_queue"],
                                   "Predicted EV queue": pred["predicted_queue"], "Steps ahead": pred["horizon"],
                                   "Prediction source": pred["model"]} for d, pred in sim.peak_predictions.items()])
        st.bar_chart(forecast_df.set_index("District")[["Current EV queue", "Predicted EV queue"]], height=270)
        st.dataframe(forecast_df, use_container_width=True, hide_index=True)
    else:
        st.info("Run at least one step to view the Peak Prediction Agent's forecast.")
    st.caption("Peak Prediction issues warnings only; it does not directly change charging decisions.")

with messages_tab:
    st.subheader("Agent-to-agent communication")
    st.caption("ACL-like communication includes REQUEST, CFP, PROPOSE, ACCEPT and REFUSE.")
    messages = pd.DataFrame(sim.message_rows())
    f1, f2 = st.columns(2)
    with f1:
        chosen_types = st.multiselect("Message types", list(PERFORMATIVES), default=list(PERFORMATIVES))
    with f2:
        participants = sorted(set(messages["From"].astype(str)).union(set(messages["To"].astype(str)))) if not messages.empty else []
        chosen_agents = st.multiselect("Agents (sender or receiver)", participants, default=[])
    if not messages.empty:
        filtered = messages[messages["Performative"].isin(chosen_types)]
        if chosen_agents:
            filtered = filtered[filtered["From"].isin(chosen_agents) | filtered["To"].isin(chosen_agents)]
        st.caption(f"Showing {len(filtered)} of {len(messages)} messages")
        st.dataframe(filtered, use_container_width=True, hide_index=True, height=390)
        st.download_button("⬇️ Export filtered log (CSV)", filtered.to_csv(index=False).encode("utf-8-sig"),
                           file_name="neom_messages.csv", mime="text/csv")
    else:
        st.info("No messages yet. Run the simulation to view conversations between agents.")
    st.subheader("Coordinator decisions and events")
    if sim.events:
        st.text_area("Event history", "\n".join(sim.events[-120:]), height=215, label_visibility="collapsed")
    else:
        st.info("No events recorded yet.")

with settings_tab:
    st.subheader("Operations & diagnostics")
    st.caption("Change simulation parameters or inject failures using the sidebar.")
    x, y = st.columns(2)
    with x:
        st.markdown("**Current configuration**")
        st.dataframe(pd.DataFrame([{"Parameter": key, "Value": value} for key, value in vars(cfg).items()]),
                     use_container_width=True, hide_index=True, height=440)
    with y:
        st.markdown("**System status**")
        st.write("**Scenario:**", picked)
        st.write("**Safe mode:**", "ON" if sim.safe_mode else "OFF")
        st.write("**Failed agents:**", ", ".join(sorted(sim.failed_agents)) if sim.failed_agents else "None")
        tracing = os.getenv("LANGSMITH_TRACING", "false").lower() == "true" and bool(os.getenv("LANGSMITH_API_KEY"))
        st.write("**LangSmith configuration:**", "Enabled" if tracing else "Disabled")
        st.link_button("🔎 Inspect traces in LangSmith", "https://smith.langchain.com/")
        st.markdown("**Open negotiations and collection trucks**")
        st.json({"negotiations": sim.negotiations, "trucks": sim.trucks})

st.divider()
st.caption("Educational multi-agent demonstration • Simulated data only • No connection to live NEOM infrastructure • API keys are not displayed")
