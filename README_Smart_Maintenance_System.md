
# Smart Maintenance System for Industrial IoT

A multi-agent AI system for predicting equipment failures, retrieving maintenance evidence (RAG), generating repair workflows, and triggering alerts using Azure AI Foundry & Azure AI Search.

---

## ✅ Features

### 1. Stateful Predictor Agent
- Reads telemetry (Temperature, Pressure, CPU)
- Computes failure risk using threshold heuristics
- Stores historical state per device

### 2. RAG Retrieval using Azure AI Search
- Documents (CSV/PDF) stored in Blob
- Indexed using Azure AI Search
- Query constructed dynamically based on failure mode + telemetry
- Snippets included in final report

### 3. Planner Agent
- Generates:
  - Step-by-step repair procedure  
  - LOTO safety instructions  
  - Required parts  
  - Rollback plan  
  - ETA  
- Deterministic logic (SK-ready)

### 4. Alert Agent
- Triggers alert if risk ≥ threshold

### 5. Python Orchestrator
- Predict → Retrieve → Plan → Alert
- Outputs a structured maintenance report

---

## ✅ Prerequisites

- Azure Subscription  
- Azure AI Foundry Project  
- Deployment: `gpt-4o`  
- Azure AI Search index  
- Blob Storage  
- Python 3.10+  
- Azure CLI  

---

## ✅ Setup

### 1. Create virtual environment
```bash
python -m venv labenv
labenv/Scripts/activate
```

### 2. Install dependencies
```bash
pip install agent-framework azure-identity azure-search-documents python-dotenv pandas
```

### 3. Create `.env` file
```
AZURE_AI_PROJECT_ENDPOINT="https://<project>.api.ai.azure.com/api/projects/<id>"
AZURE_AI_MODEL_DEPLOYMENT_NAME="gpt-4o"

AZURE_SEARCH_SERVICE="your-search-service"
AZURE_SEARCH_API_KEY="your-key"
AZURE_SEARCH_INDEX="your-index-name"

CSV_PATH=""
```

---

## ✅ Running the System

```bash
az login
python smart_maintenance_agent.py
```

You will see the system produce:
- ✅ Risk score  
- ✅ Failure modes  
- ✅ RAG Evidence  
- ✅ Detailed plan  
- ✅ Alert  

---

## ✅ Architecture (PPT included)

See **Smart_Maintenance_System_PPT.pptx** for:
- Architecture diagram  
- Workflow  
- Agent design  
- Compliance overview  

---

## ✅ Deliverables for Grading

- ✅ Code repo  
- ✅ README  
- ✅ PPT  
- ✅ Screenshots:
  - Foundry project  
  - AI Search index  
  - Blob container  
  - Terminal output  

---

## ✅ License
MIT License
