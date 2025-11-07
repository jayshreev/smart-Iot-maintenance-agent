import os, json, asyncio
from typing import Annotated, List
from dotenv import load_dotenv
from pydantic import Field
import pandas as pd

# --- RAG (Azure AI Search) ---
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient

load_dotenv()

# ========= Config =========
# Foundry vars (printed for sanity; not used in Fix A execution path)
PROJECT_ENDPOINT = os.getenv("AZURE_AI_PROJECT_ENDPOINT")  # should look like .../api/projects/<id>
MODEL_DEPLOYMENT = os.getenv("AZURE_AI_MODEL_DEPLOYMENT_NAME", "gpt-4o")

# Azure AI Search (RAG)
AZURE_SEARCH_SERVICE = os.getenv("AZURE_SEARCH_SERVICE")
AZURE_SEARCH_API_KEY = os.getenv("AZURE_SEARCH_API_KEY")
AZURE_SEARCH_INDEX = os.getenv("AZURE_SEARCH_INDEX")

# Optional CSV
CSV_PATH = (os.getenv("CSV_PATH") or "").strip()

# Quick env sanity (print once so you know .env loaded)
print("Foundry Project Endpoint:", PROJECT_ENDPOINT)
print("Model Deployment:", MODEL_DEPLOYMENT)
print("Search svc/index:", AZURE_SEARCH_SERVICE, AZURE_SEARCH_INDEX)

# ========= Simple state (per device) =========
STATE = {}  # { deviceId: {history: [...], lastRisk: float} }

def update_state(deviceId: str, Temperature: float, Pressure: float, CPU_Usage: float, risk: float):
    if deviceId not in STATE:
        STATE[deviceId] = {"history": [], "lastRisk": 0.0}
    STATE[deviceId]["history"].append({"T": Temperature, "P": Pressure, "CPU": CPU_Usage, "risk": risk})
    STATE[deviceId]["history"] = STATE[deviceId]["history"][-100:]  # cap size
    STATE[deviceId]["lastRisk"] = risk

def read_last_state(deviceId: str):
    return STATE.get(deviceId, {"history": [], "lastRisk": 0.0})

# ========= Utilities =========
def compute_simple_risk(T: float, P: float, CPU: float) -> float:
    risk = 0.0
    if T > 85: risk += 0.4
    if P > 7.5: risk += 0.3
    if CPU > 90: risk += 0.2
    return round(min(1.0, max(0.0, risk)), 2)

def build_search_client() -> SearchClient:
    if not all([AZURE_SEARCH_SERVICE, AZURE_SEARCH_API_KEY, AZURE_SEARCH_INDEX]):
        raise RuntimeError("Missing AZURE_SEARCH_* in .env (service, key, index).")
    return SearchClient(
        endpoint=f"https://{AZURE_SEARCH_SERVICE}.search.windows.net",
        index_name=AZURE_SEARCH_INDEX,
        credential=AzureKeyCredential(AZURE_SEARCH_API_KEY)
    )

# ========= Tools (deterministic + RAG) =========

# 1) PREDICT — stateful heuristic with rationale
async def predict_failure_risk(
    deviceId: Annotated[str, Field(description="Device identifier")],
    Temperature: Annotated[float, Field(description="Temperature (C)")],
    Pressure: Annotated[float, Field(description="Pressure (units)")],
    CPU_Usage: Annotated[float, Field(description="CPU load (%)")],
) -> dict:
    risk = compute_simple_risk(Temperature, Pressure, CPU_Usage)
    prev = read_last_state(deviceId)
    update_state(deviceId, Temperature, Pressure, CPU_Usage, risk)
    rationale = (
        f"Heuristic thresholds → Prev risk={prev.get('lastRisk',0.0):.2f}, "
        f"Now={risk:.2f} (T={Temperature}°C, P={Pressure}, CPU={CPU_Usage}%)."
    )
    out = {
        "deviceId": deviceId,
        "risk": risk,
        "rationale": rationale,
        "failure_modes": ["overheating", "bearing wear", "misalignment"],
        "history_length": len(STATE[deviceId]['history']),
    }
    print("[TOOL] predict_failure_risk ->", out)
    return out

# 2) RETRIEVE — query your Azure AI Search index
def retrieve_docs(query: str, topK: int = 3) -> List[dict]:
    client = build_search_client()
    results = client.search(query, top=topK, query_type="simple")
    out = []
    for r in results:
        out.append({
            "title": r.get("title",""),
            "snippet": (r.get("content","") or r.get("chunk",""))[:500],
            "source": r.get("source",""),
            "score": r.get("@search.score", 0.0)
        })
    print("[TOOL] retrieve_docs ->", len(out), "docs")
    return out

# 3) PLAN — deterministic, readable plan (no SK, no model call)
def plan_repair_workflow(
    deviceId: Annotated[str, Field(description="Device identifier")],
    risk: Annotated[float, Field(description="Risk 0..1")],
    failure_modes: Annotated[List[str], Field(description="Failure modes")],
    evidence: Annotated[List[dict], Field(description="Docs from RAG")] = [],
) -> str:
    ev_lines = []
    for i, e in enumerate(evidence[:3], start=1):
        title = e.get("title") or f"Doc {i}"
        snippet = (e.get("snippet") or "")[:300].strip()
        src = e.get("source") or ""
        ev_lines.append(f"- {title}: {snippet}{(' ['+src+']') if src else ''}")

    steps = [
        "Apply Lockout/Tagout (LOTO) per site procedure; verify zero energy state.",
        "Visually inspect housing, inlet/outlet, mounts for looseness or damage.",
        "Measure bearing temperature and vibration; compare with baseline limits.",
        "Check shaft alignment and coupling; re-align if out of tolerance.",
        "Lubricate/replace bearings per OEM spec if play/noise detected.",
        "Clear cooling passages and filters; verify ventilation/airflow.",
        "Run controlled test (10–15 min); record temp/pressure/vibration trends.",
        "Update CMMS, attach readings, and trend risk for the next 7 days."
    ]
    parts = [
        "Bearing kit (per BOM)",
        "Coupling shims",
        "Approved grease/lubricant",
        "Cleaning kit / filters"
    ]
    safety = [
        "LOTO enforced; arc-flash PPE as required",
        "Eye/hand protection at all times",
        "Follow confined-space rules if applicable"
    ]
    eta = 3.5 if risk >= 0.7 else 2.0
    rollback = (
        "Restore original alignment and settings; reassemble with prior shims; "
        "remove LOTO only after verification and sign-off."
    )

    lines = [
        f"# Repair Plan for {deviceId}",
        f"Risk: {risk} | Failure modes: {', '.join(failure_modes)}",
        "## Evidence (summary)"
    ] + (ev_lines or ["- No external evidence referenced in this run."]) + [
        "## Steps"
    ] + [f"{i+1}. {s}" for i, s in enumerate(steps)] + [
        "## Parts/Materials"
    ] + [f"- {p}" for p in parts] + [
        "## Safety"
    ] + [f"- {s}" for s in safety] + [
        f"## ETA (hours)\n{eta}",
        f"## Rollback\n{rollback}"
    ]
    result = "\n".join(lines)
    print("[TOOL] plan_repair_workflow -> ok")
    return result

# 4) ALERT — minimal console alert
def send_alert(deviceId: str, risk: float) -> dict:
    res = {"alert": "TRIGGERED" if risk >= 0.7 else "NO_ALERT", "deviceId": deviceId, "risk": risk}
    print("[TOOL] send_alert ->", res)
    return res

# ========= Coordinator (deterministic orchestration) =========
async def run_coordinator(event: dict):
    deviceId = event["deviceId"]
    thr = float(event.get("alert_threshold", 0.7))
    topK = int(event.get("topK", 3))

    # 1) Predict
    pred = await predict_failure_risk(
        deviceId=deviceId,
        Temperature=event["Temperature"],
        Pressure=event["Pressure"],
        CPU_Usage=event["CPU_Usage"],
    )

    # 2) Retrieve evidence from your Search index (only if high risk)
    evidence = []
    if pred["risk"] >= thr:
        query = f"{' '.join(pred['failure_modes'])} pump maintenance pressure:{event['Pressure']} temperature:{event['Temperature']}"
        evidence = retrieve_docs(query, topK=topK)

    # 3) Plan (include evidence if any)
    plan_text = plan_repair_workflow(
        deviceId=deviceId,
        risk=pred["risk"],
        failure_modes=pred["failure_modes"],
        evidence=evidence,
    )

    # 4) Alert
    alert = send_alert(deviceId=deviceId, risk=pred["risk"])

    # 5) Compose final report (text only)
    lines = [
        f"# Smart Maintenance Report — {deviceId}",
        "## Risk & Rationale",
        f"- Risk score: **{pred['risk']}**",
        f"- Rationale: {pred['rationale']}",
        "## Failure Modes",
        *[f"- {m}" for m in pred["failure_modes"]],
        "## Evidence (summary)",
    ]
    if evidence:
        for e in evidence:
            src = f" [{e['source']}]" if e.get("source") else ""
            lines.append(f"- {e['title']}: {e['snippet']}{src}")
    else:
        lines.append("- No external evidence referenced in this run.")
    lines += [
        "## Repair Plan",
        plan_text,
        "## Alert Status",
        f"- {alert['alert']} (threshold: {thr})"
    ]
    report = "\n".join(lines)
    print(report)
    return report

# ========= CSV helper (optional) =========
def last_row_event(csv_path: str) -> dict:
    df = pd.read_csv(csv_path, parse_dates=["Timestamp"])
    df.sort_values(["DeviceID", "Timestamp"], inplace=True)
    last = df.groupby("DeviceID").tail(1).iloc[0]
    return {
        "deviceId": str(last["DeviceID"]),
        "Temperature": float(last["Temperature"]),
        "Pressure": float(last["Pressure"]),
        "CPU_Usage": float(last["CPU_Usage"]),
        "topK": 3,
        "alert_threshold": 0.7
    }

# ========= Main =========
if __name__ == "__main__":
    # Build event from CSV if provided, else use sample
    if CSV_PATH:
        evt = last_row_event(CSV_PATH)
    else:
        evt = {
            "deviceId": "pump-17",
            "Temperature": 92.0,
            "Pressure": 8.2,
            "CPU_Usage": 88.0,
            "topK": 3,
            "alert_threshold": 0.7
        }

    print("Event:", evt)
    print("\nRunning coordinator...\n")

    # Minimal sanity checks
    if not PROJECT_ENDPOINT or "projects/" not in PROJECT_ENDPOINT:
        print("[WARN] AZURE_AI_PROJECT_ENDPOINT is missing or not a Foundry project endpoint (must end with /projects/<id>).")
    if not all([AZURE_SEARCH_SERVICE, AZURE_SEARCH_API_KEY, AZURE_SEARCH_INDEX]):
        print("[WARN] AZURE_SEARCH_* values are missing—RAG retrieval will fail.")

    # Run
    print(asyncio.run(run_coordinator(evt)))
