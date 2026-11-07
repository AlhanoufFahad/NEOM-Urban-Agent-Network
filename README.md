# NEOM-Urban-Agent-Network
AI-powered Python multi-agent system for smart city management, integrating Groq LLM, Random Forest, LangSmith, and Streamlit to coordinate energy, transportation, and waste services.
# NEOM Urban Agent Network — Python + Groq LLM + Predictive AI + LangSmith

A five-agent smart-city *simulation* for energy/EV charging, traffic, and waste collection.
The agents are **Energy**, **Transportation**, **Waste**, **Coordinator**, and **Peak Prediction**.
Energy uses a hosted **Groq LLM** (default: `openai/gpt-oss-20b`) through LangChain when configured, and Peak Prediction uses a local **Random Forest** model trained on synthetic data. Agents exchange ACL-like messages (`REQUEST`, `INFORM`, `CFP`, `PROPOSE`, `ACCEPT`, `REFUSE`); Coordinator negotiates using a Contract Net-style protocol and detects failed agent heartbeats.

**Important:** The model proposes a charging delay; the simulator ALWAYS validates bounds before implementing a proposal. Without API access, a safe deterministic heuristic takes over. The Random Forest forecasts are based on synthetic examples, not validated NEOM operational data. This version uses LangChain and LangSmith; **LangGraph is not implemented**.

## Installation

Requires Python 3.10+.

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
```

Copy `.env.example` to `.env`, then set **your real** `GROQ_API_KEY` from https://console.groq.com/keys. LangSmith keys are optional, obtained separately from https://smith.langchain.com. **Never upload `.env` to GitHub.**

## Run

```bash
python -m streamlit run app.py
```

Open http://localhost:8501. The sidebar offers three scenarios, Step / Run 5 / Run all / Reset, model selection, sliders, Random Forest switch, and Fail / Restart controls. The five tabs show city metrics, agent states, LLM & prediction information, ACL message logs, and settings/diagnostics.

## Scenario tests

1. EV charging versus congestion: REQUEST → CFP → PROPOSE → ACCEPT, with a validated charging-delay proposal from Groq when the key is set.
2. Full bin on busy road: Waste requests a route; Transportation proposes an uncongested road; truck collects waste.
3. Energy failure and rush hour: missing heartbeats trigger safe mode; the other agents continue; Energy recovers.

Run automated tests without calling Groq:

```bash
python -m pytest -q
```

## LLM & API behavior

- The Groq-hosted model is `openai/gpt-oss-20b` by default; `llama-3.3-70b-versatile` can be selected in the UI if available to your account.
- Uses LangChain `ChatGroq`, returning JSON with fields `count`, `steps`, `reasoning`.
- The Coordinator clamps the EV count to the pending CFP and actual queue, and caps delay at `max_delay_steps`.
- HTTP 5xx / timeout failures may be retried twice (3 attempts total). **HTTP 429 is not retried**; a 60-second cooldown avoids repeated immediate API calls, then the safe heuristic is used.
- Groq is a cloud provider, so **429 errors are still possible** under quota/rate limits. For demo use, consider limiting runs to the steps needed to show negotiation.
- LangSmith traces are available when `LANGSMITH_TRACING=true` and `LANGSMITH_API_KEY` is configured. Never commit secrets.

## Project structure

```text
app.py                     Streamlit dashboard
neom/model.py              Multi-agent simulation, ACL exchange, safety validation
neom/llm.py                Hosted Groq LLM adapter and error handling
neom/prediction.py         Random Forest queue forecasting
neom/scenarios.py           Scenario setup
tests/test_simulation.py   Simulation and mocked LLM tests
requirements.txt           Dependencies
.env.example               Configuration template
.gitignore                 Git secret/file exclusions
```

## Group submission

Upload the source files (not just a ZIP) to GitHub. Add all group members and the instructors **Dr. Eman** and **T. Manairh** as collaborators (using their actual GitHub usernames). Test installation from a fresh clone. Keep API keys in a local `.env` only.  


## Project Team
- Norah Mansour
- Raghad Almarhom
- Alhanouf Alsharafi
- Fatimah Almosbih
  
## Supervisors
- Dr. Eman
- T. Manairh
