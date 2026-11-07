"""Python-only, deterministic-capable NEOM multi-agent simulation core."""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from collections import deque
from typing import Any
import random
import os
from . import llm
from . import prediction

try:
    from langsmith import traceable
except ImportError:
    def traceable(*args, **kwargs):
        def wrap(fn): return fn
        return wrap

AGENTS = ("Energy", "Transportation", "Waste", "Coordinator", "Peak Prediction")
PERFORMATIVES = ("INFORM", "REQUEST", "CFP", "PROPOSE", "ACCEPT", "REFUSE")

@dataclass
class Config:
    peak_threshold: int = 80
    congestion_limit: int = 70
    bin_threshold: int = 85
    ev_arrivals: int = 3
    bin_fill_rate: int = 2
    randomness: int = 0
    heartbeat_timeout: int = 3
    negotiation_timeout: int = 2
    forecast_horizon: int = 3
    trend_window: int = 3
    alert_threshold: int = 15
    max_delay_steps: int = 6
    max_steps: int = 30
    ml_prediction_enabled: bool = True
    llm_enabled: bool = True
    llm_model: str = "openai/gpt-oss-20b"

@dataclass
class Message:
    step: int
    sender: str
    receiver: str
    performative: str
    conversation_id: str
    content: dict[str, Any]
    delivered: bool = False

@dataclass
class AgentStatus:
    name: str
    kind: str
    state: str = "ACTIVE"
    last_action: str = "Waiting"
    last_heartbeat: int = 0
    coordinator_view: str = "ACTIVE"

@dataclass
class District:
    name: str
    queue: int = 0
    parked: int = 0
    charging: int = 0
    grid_load: float = 50.0
    base_traffic: float = 35.0
    congestion: float = 35.0
    bin_fill: float = 20.0
    charge_paused_until: int = 0


class CitySimulation:
    """Message-driven, single-process ABM; reproducible with a fixed seed."""
    def __init__(self, scenario: str = "charging", config: Config | None = None, seed: int = 461):
        self.scenario = scenario
        self.config = config or Config()
        self.seed = seed
        self.rng = random.Random(seed)
        self.step_number = 0
        self.messages: list[Message] = []
        self.pending: list[Message] = []
        self.events: list[str] = []
        self.negotiations: dict[str, dict] = {}
        self.completed_conflicts = 0
        self.failed_agents: set[str] = set()
        self.safe_mode = False
        self.last_llm_result = "Not yet invoked"
        self.history: list[dict[str, Any]] = []
        self.queue_history = {f"D{i}": deque(maxlen=30) for i in range(1, 4)}
        self.districts = {f"D{i}": District(f"D{i}") for i in range(1, 4)}
        kinds = {"Energy": "Hybrid / LLM-enabled", "Transportation": "Hybrid", "Waste": "Reactive",
                 "Coordinator": "Deliberative", "Peak Prediction": "Deliberative / ML"}
        self.agents = {name: AgentStatus(name, kinds[name]) for name in AGENTS}
        self.trucks: list[dict] = []
        self.active_collection: set[str] = set()
        self.last_peak_alert: dict[str, int] = {}
        self.peak_predictions: dict[str, dict[str, Any]] = {}
        self._setup()

    def _setup(self):
        if self.scenario == "charging":
            d = self.districts["D2"]
            d.queue, d.grid_load, d.base_traffic = 20, 90, 55
            for q in (12, 15, 18): self.queue_history["D2"].append(q)
        elif self.scenario == "waste":
            self.districts["D3"].bin_fill = 92
            self.districts["D2"].base_traffic = 75
            self.districts["D1"].base_traffic = 30
            self.districts["D3"].base_traffic = 35
            self.config.ev_arrivals = 0
            self.config.bin_fill_rate = 0
        elif self.scenario == "failure":
            self.districts["D2"].base_traffic = 62
        else:
            raise ValueError("scenario must be charging, waste, or failure")
        self._recompute()
        self.events.append(f"Scenario '{self.scenario}' initialized (seed={self.seed}).")

    def send(self, sender: str, receiver: str, perf: str, content: dict, cid: str = ""):
        assert perf in PERFORMATIVES
        msg = Message(self.step_number, sender, receiver, perf, cid or f"msg-{len(self.messages)+1}", dict(content))
        self.messages.append(msg)
        self.pending.append(msg)
        return msg

    def event(self, text: str):
        self.events.append(f"Step {self.step_number}: {text}")

    def fail(self, name: str):
        if name not in self.agents or name == "Coordinator":
            return
        self.failed_agents.add(name)
        self.agents[name].state = "CRASHED"
        self.agents[name].last_action = "Failure injected"
        self.event(f"{name} crashed; waiting for heartbeat timeout.")

    def restart(self, name: str):
        if name in self.failed_agents:
            self.failed_agents.remove(name)
            self.agents[name].state = "ACTIVE"
            self.agents[name].last_heartbeat = self.step_number
            self.agents[name].coordinator_view = "ACTIVE"
            self.agents[name].last_action = "Recovered"
            if name == "Energy":
                self.safe_mode = False
                self.event("Energy restored; normal charging admissions resumed.")
            self.send(name, "Coordinator", "INFORM", {"topic": "heartbeat", "recovered": True})

    def _recompute(self):
        for d in self.districts.values():
            d.congestion = min(100.0, d.base_traffic + d.queue)

    def _deliver(self):
        ready, self.pending = self.pending, []
        for m in ready:
            if m.receiver in self.failed_agents:
                self.event(f"Dropped {m.performative} to crashed {m.receiver}.")
                continue
            m.delivered = True
            self._receive(m)

    def _receive(self, m: Message):
        if m.receiver == "Coordinator":
            if m.performative == "REQUEST" and m.content.get("action") == "delay_charging":
                district = m.content["district"]
                key = f"charging-{district}"
                if key not in self.negotiations and not self.safe_mode:
                    n = max(1, min(self.districts[district].queue, int(m.content.get("excess", 1)) + 2))
                    self.negotiations[key] = {"topic": "charging", "district": district, "created": self.step_number, "count": n}
                    self.send("Coordinator", "Energy", "CFP", {"topic": "charging", "district": district, "count": n,
                          "congestion": self.districts[district].congestion, "grid_load": self.districts[district].grid_load}, key)
                    self.event(f"Contract Net opened: {key}.")
            elif m.performative == "REQUEST" and m.content.get("action") == "route":
                bin_name = m.content["bin"]
                key = f"route-{bin_name}"
                if key not in self.negotiations and bin_name not in self.active_collection:
                    self.negotiations[key] = {"topic": "route", "district": bin_name, "created": self.step_number}
                    self.send("Coordinator", "Transportation", "CFP", {"topic": "route", "bin": bin_name}, key)
            elif m.performative == "PROPOSE":
                n = self.negotiations.get(m.conversation_id)
                if n is None:
                    self.send("Coordinator", m.sender, "REFUSE", {"reason": "No active negotiation"}, m.conversation_id)
                elif n["topic"] == "charging":
                    proposal = m.content
                    if proposal.get("count", 0) > 0 and 1 <= proposal.get("steps", 0) <= self.config.max_delay_steps:
                        self.send("Coordinator", "Energy", "ACCEPT", proposal, m.conversation_id)
                        self.send("Coordinator", "Transportation", "INFORM", {"topic": "charging_delayed", **proposal}, m.conversation_id)
                        self.completed_conflicts += 1
                        self.negotiations.pop(m.conversation_id, None)
                        self.event(f"Accepted charging delay for {proposal['count']} EVs in {proposal['district']}.")
                    else:
                        self.send("Coordinator", m.sender, "REFUSE", {"reason": "Invalid proposal"}, m.conversation_id)
                elif n["topic"] == "route":
                    route = m.content["road"]
                    self.send("Coordinator", "Transportation", "ACCEPT", {"road": route}, m.conversation_id)
                    self.send("Coordinator", "Waste", "INFORM", {"topic": "route_assigned", "bin": n["district"], "road": route}, m.conversation_id)
                    self.completed_conflicts += 1
                    self.negotiations.pop(m.conversation_id, None)
                    self.event(f"Route {route} assigned to {n['district']} collection.")
        elif m.receiver == "Energy" and m.performative == "CFP":
            decision = self._energy_offer(m.content)
            self.send("Energy", "Coordinator", "PROPOSE", decision, m.conversation_id)
            self.agents["Energy"].last_action = f"PROPOSE delay({decision['district']}, {decision['count']}, {decision['steps']})"
        elif m.receiver == "Energy" and m.performative == "ACCEPT":
            d = self.districts[m.content["district"]]
            count = min(d.queue, int(m.content["count"]))
            d.queue -= count
            d.parked += count
            d.charge_paused_until = self.step_number + int(m.content["steps"])
            d.grid_load = max(0, d.grid_load - 2 * count)
            self.agents["Energy"].last_action = f"delayCharging({d.name}, {count}, {m.content['steps']})"
        elif m.receiver == "Transportation" and m.performative == "CFP":
            road = min(self.districts, key=lambda x: self.districts[x].congestion)
            self.send("Transportation", "Coordinator", "PROPOSE", {"road": "R" + road[-1],
                      "congestion": self.districts[road].congestion}, m.conversation_id)
            self.agents["Transportation"].last_action = f"proposeRoute(R{road[-1]})"
        elif m.receiver == "Transportation" and m.content.get("topic") == "charging_delayed":
            self.agents["Transportation"].last_action = f"rerouteTraffic({m.content['district']})"
        elif m.receiver == "Waste" and m.content.get("topic") == "route_assigned":
            bin_name = m.content["bin"]
            self.active_collection.add(bin_name)
            self.trucks.append({"bin": bin_name, "road": m.content["road"], "arrival": self.step_number + 2})
            self.agents["Waste"].last_action = f"dispatchTruck({bin_name}, {m.content['road']})"

    @traceable(name="energy_delay_proposal", run_type="chain")
    def _energy_offer(self, request: dict) -> dict:
        """Generate an LLM proposal if configured, then enforce safety limits.

        Without credentials, or when a provider request fails, the simulation
        continues with a deterministic fallback and visibly reports its source.
        """
        district_name = request["district"]
        d = self.districts[district_name]
        available = max(0, min(int(request["count"]), d.queue))
        severity = max(float(request["grid_load"]) - self.config.peak_threshold, 0)
        fallback_steps = min(self.config.max_delay_steps, max(1, 2 + int(severity // 3)))
        fallback = {"district": district_name, "count": available,
                    "steps": fallback_steps, "policy": "bounded_heuristic"}
        self.last_llm_result = "Heuristic mode (no model call)"
        if not self.config.llm_enabled:
            self.last_llm_result = "LLM disabled in dashboard; heuristic used"
            return fallback
        if not llm.configured():
            self.last_llm_result = "GROQ_API_KEY missing; heuristic used"
            return fallback
        context = {"district": district_name, "queue": d.queue,
                   "requested_count": available, "grid_load": float(request["grid_load"]),
                   "peak_threshold": self.config.peak_threshold,
                   "congestion": float(request.get("congestion", d.congestion)),
                   "congestion_limit": self.config.congestion_limit,
                   "max_delay_steps": self.config.max_delay_steps}
        try:
            raw = llm.propose_charging_delay(context, self.config.llm_model)
            # Validate against *actual* live queue and original CFP request.
            # Never directly execute unbounded model-generated output.
            count = min(available, max(0, int(raw["count"])))
            steps = min(self.config.max_delay_steps, max(1, int(raw["steps"])))
            if count == 0 and available > 0:
                raise ValueError("LLM proposed no action for a positive relief request")
            offer = {"district": district_name, "count": count, "steps": steps,
                     "policy": "llm_validated", "reasoning": str(raw.get("reasoning", ""))[:240]}
            self.last_llm_result = f"LLM proposal validated ({self.config.llm_model})"
            self.event(self.last_llm_result)
            return offer
        except (Exception) as exc:
            # Networking failures should not block the city's other agents.
            self.last_llm_result = f"LLM fallback: {llm.explain_error(exc, self.config.llm_model)}"
            self.event(self.last_llm_result)
            return fallback

    def _heartbeat_and_failures(self):
        for name, agent in self.agents.items():
            if name not in self.failed_agents:
                agent.last_heartbeat = self.step_number
                if agent.coordinator_view != "ACTIVE":
                    agent.coordinator_view = "ACTIVE"
                    self.event(f"Coordinator detected recovery: {name}.")
            elif self.step_number - agent.last_heartbeat > self.config.heartbeat_timeout:
                if agent.coordinator_view != "FAILED":
                    agent.coordinator_view = "FAILED"
                    self.event(f"Coordinator detected failure: {name} (heartbeat timeout).")
                    for key, n in list(self.negotiations.items()):
                        if (name == "Energy" and n["topic"] == "charging") or (name == "Transportation" and n["topic"] == "route"):
                            del self.negotiations[key]
                            self.event(f"Cancelled negotiation {key} after agent failure.")
                    if name == "Energy":
                        self.safe_mode = True
                        self.event("Safe mode activated: EV admissions closed.")

    def _environment(self):
        arrivals = self.config.ev_arrivals
        if self.scenario == "failure" and self.step_number >= 4:
            arrivals = max(arrivals, 8)
        for d in self.districts.values():
            variance = self.rng.randint(-self.config.randomness, self.config.randomness)
            incoming = max(0, arrivals + variance)
            if not self.safe_mode:
                d.queue += incoming
            if d.charge_paused_until <= self.step_number and "Energy" not in self.failed_agents:
                admitted = min(3, d.queue)
                d.queue -= admitted
                d.charging = admitted
                d.grid_load = min(100, max(10, d.grid_load - 2 + admitted * 0.4))
            else:
                d.charging = 0
                d.grid_load = max(10, d.grid_load - 2)
            if self.safe_mode:
                d.queue = max(0, d.queue - 3)
            d.bin_fill = min(100, d.bin_fill + self.config.bin_fill_rate)
            self.queue_history[d.name].append(d.queue)
        self._recompute()
        for truck in list(self.trucks):
            if truck["arrival"] <= self.step_number:
                self.districts[truck["bin"]].bin_fill = 0
                self.active_collection.discard(truck["bin"])
                self.trucks.remove(truck)
                self.agents["Waste"].last_action = f"collectWaste({truck['bin']})"
                self.event(f"Collected waste in {truck['bin']} via {truck['road']}.")

    def _agents_act(self):
        if "Energy" not in self.failed_agents:
            maximum = max(d.grid_load for d in self.districts.values())
            self.send("Energy", "Coordinator", "INFORM", {"topic": "grid", "max_grid_load": round(maximum, 1)})
            if self.agents["Energy"].last_action == "Waiting":
                self.agents["Energy"].last_action = "monitorGrid()"
        if "Transportation" not in self.failed_agents:
            for d in self.districts.values():
                if d.congestion > self.config.congestion_limit and d.queue and not self.safe_mode:
                    self.send("Transportation", "Coordinator", "REQUEST", {"action": "delay_charging", "district": d.name,
                              "congestion": round(d.congestion, 1), "excess": int(d.congestion-self.config.congestion_limit)})
                    self.agents["Transportation"].last_action = f"requestChargingDelay({d.name})"
        if "Waste" not in self.failed_agents:
            for d in self.districts.values():
                if d.bin_fill >= self.config.bin_threshold and d.name not in self.active_collection:
                    if f"route-{d.name}" not in self.negotiations:
                        self.send("Waste", "Coordinator", "REQUEST", {"action": "route", "bin": d.name, "fill": d.bin_fill})
                        self.agents["Waste"].last_action = f"requestRoute({d.name})"
        if "Peak Prediction" not in self.failed_agents:
            for district, samples in self.queue_history.items():
                window = list(samples)[-max(3, self.config.trend_window):]
                if len(window) >= 3:
                    rise = (window[-1] - window[0]) / (len(window) - 1)
                    source = "trend heuristic"
                    predicted = max(0.0, window[-1] + rise * self.config.forecast_horizon)
                    if self.config.ml_prediction_enabled:
                        try:
                            predicted = prediction.predict_queue(window, self.config.forecast_horizon)
                            source = "Random Forest ML"
                        except Exception as exc:
                            self.event(f"Peak ML unavailable ({type(exc).__name__}); using trend fallback.")
                    self.peak_predictions[district] = {
                        "predicted_queue": round(predicted, 1),
                        "current_queue": window[-1],
                        "model": source,
                        "horizon": self.config.forecast_horizon,
                    }
                    self.agents["Peak Prediction"].last_action = f"predictPeak({district}, {source})"
                    if rise > 0 and predicted >= self.config.alert_threshold and self.step_number - self.last_peak_alert.get(district, -99) >= 3:
                        self.send("Peak Prediction", "all agents (log)", "INFORM", {"topic": "peak_alert", "district": district,
                                  "predicted_queue": round(predicted, 1), "source": source,
                                  "horizon": self.config.forecast_horizon})
                        self.last_peak_alert[district] = self.step_number
                        self.agents["Peak Prediction"].last_action = f"sendPeakAlert({district}, {source})"

    def _expire_negotiations(self):
        for key, n in list(self.negotiations.items()):
            if self.step_number - n["created"] > self.config.negotiation_timeout + 2:
                self.event(f"Negotiation {key} expired.")
                self.negotiations.pop(key, None)

    @traceable(name="neom_simulation_step", run_type="chain")
    def step(self):
        if self.step_number >= self.config.max_steps:
            return False
        self.step_number += 1
        if self.scenario == "failure" and self.step_number == 3:
            self.fail("Energy")
        if self.scenario == "failure" and self.step_number == 16:
            self.restart("Energy")
        self._environment()
        self._deliver()
        self._heartbeat_and_failures()
        self._expire_negotiations()
        self._agents_act()
        self.history.append({"step": self.step_number, "max_grid_load": round(max(d.grid_load for d in self.districts.values()), 1),
                            "max_congestion": round(max(d.congestion for d in self.districts.values()), 1),
                            "total_queue": sum(d.queue for d in self.districts.values()),
                            "open_conflicts": len(self.negotiations), "safe_mode": self.safe_mode})
        return True

    def run(self, steps: int | None = None):
        for _ in range(steps if steps is not None else self.config.max_steps - self.step_number):
            if not self.step(): break
        return self

    def agent_rows(self):
        return [{"Agent": a.name, "Type": a.kind, "State": a.state, "Coordinator view": a.coordinator_view,
                 "Last heartbeat": a.last_heartbeat, "Last action": a.last_action} for a in self.agents.values()]

    def message_rows(self):
        return [{"Step": m.step, "From": m.sender, "To": m.receiver, "Performative": m.performative,
                 "Conversation": m.conversation_id, "Content": str(m.content), "Delivered": m.delivered} for m in self.messages]

    def district_rows(self):
        return [{"District": d.name, "Grid %": round(d.grid_load, 1), "Congestion": round(d.congestion, 1),
                 "EV queue": d.queue, "Parked EVs": d.parked, "Bin fill %": round(d.bin_fill, 1)} for d in self.districts.values()]
