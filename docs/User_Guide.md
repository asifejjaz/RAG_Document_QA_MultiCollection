# Eva Research AI — User Guide

This guide walks you through installing everything on a **bare-metal Windows PC** (or laptop), cloning the project, choosing how you want to run **LLMs** (local vs cloud), configuring **`.env`**, and running the application.

---

## Table of contents

1. [What you will need](#1-what-you-will-need)
2. [**Choose your configuration path** (read this first)](#2-choose-your-configuration-path-read-this-first)
3. [Install Git](#3-install-git)
4. [Install Python](#4-install-python)
5. [Install Visual Studio Code (optional)](#5-install-visual-studio-code-optional)
6. [Install Docker Desktop](#6-install-docker-desktop)
7. [Clone the repository](#7-clone-the-repository)
8. [Open the project in VS Code](#8-open-the-project-in-vs-code)
9. [Create the `.env` file — shared settings for everyone](#9-create-the-env-file--shared-settings-for-everyone)
10. [Path A — Local LLMs only (skip Azure & OpenAI.com)](#10-path-a--local-llms-only-skip-azure--openaicom)
11. [Path B — Azure OpenAI (Microsoft cloud)](#11-path-b--azure-openai-microsoft-cloud)
12. [Path C — OpenAI.com API (platform.openai.com)](#12-path-c--openaicom-api-platformopenaicom)
13. [Where to get API keys (quick reference)](#13-where-to-get-api-keys-quick-reference)
14. [Run the project with Docker](#14-run-the-project-with-docker)
15. [Open the application in your browser](#15-open-the-application-in-your-browser)
16. [Optional: Run scripts from your machine](#16-optional-run-scripts-from-your-machine)
17. [Troubleshooting](#17-troubleshooting)
18. [Quick reference](#quick-reference)

---

## 1. What you will need

- A **Windows 10/11** PC (this guide is written for Windows; steps are similar on macOS/Linux).
- **Administrator rights** to install software.
- An **internet connection** for downloads.
- **What you do *not* need for every path:**
  - **Path A (local only):** No Azure account and no OpenAI.com API key. You still need **Docker** (or local Qdrant + Ollama) and enough disk/RAM for models.
  - **Path B (Azure):** An **Azure subscription** with an **Azure OpenAI** resource.
  - **Path C (OpenAI.com):** An **OpenAI** account and **API key** from **https://platform.openai.com**.

---

## 2. Choose your configuration path (read this first)

Pick **one primary way** to run answers and embeddings. You can still switch models in the Streamlit **sidebar** after startup, as long as the matching keys are in `.env`.

| Path | Embeddings | Answers (chat) | Azure? | OpenAI.com key? | Ollama? |
|------|------------|----------------|--------|-----------------|---------|
| **A — Local only** | **BGE-M3** (local, downloads once) | **Ollama** (Qwen / Llama) | No | No | **Yes** (Docker or host) |
| **B — Azure** | **Azure** (`text-embedding-ada-002`) | **Azure GPT** (your deployment) | **Yes** | No | Optional (for extra models) |
| **C — OpenAI.com** | **OpenAI** (`text-embedding-3-small`, `text-embedding-ada-002`) | **OpenAI** (`gpt-4o-mini`, etc.) | No | **Yes** (`OPENAI_API_KEY` or `OPEN_AI_KEY`) | Optional |

In the Streamlit app sidebar, choose **Embedding / answer provider** first: **Azure OpenAI**, **OpenAI (API)** (only if a key is set), or **Local**. That controls which **Embedding model** and **Answer model** options appear.

**How to use this guide**

1. Complete **Sections 3–9** (install tools, clone repo, create `.env` with **shared** lines).
2. Then open **only the path that matches you:**
   - **Path A** → Section 10 (skip Sections 11–12).
   - **Path B** → Section 11 (add Azure variables; you can skip Section 12).
   - **Path C** → Section 12 (add OpenAI variables; you can skip Section 11 if you do not use Azure).
3. Continue with **Docker** (Section 14) and the **browser** (Section 15).

**Streamlit UI (after the app starts)**

- **Embedding model:** e.g. Azure Ada, BGE-M3, or OpenAI embeddings — must match how you **ingested** each collection (vector size 1536 vs 1024).
- **Answer model:** **Azure**, **OpenAI (…)****, or **Ollama: …** — pick the provider you configured.

**CLI note:** `scripts/answer_local.py` uses **Ollama only**. For Azure or OpenAI answers from the command line, use the **Streamlit app** or extend scripts separately.

---

## 3. Install Git

Git is used to clone the project from the repository.

### Steps

1. Go to **https://git-scm.com/download/win** and download the **Windows** installer.
2. Run the installer. You can keep the default options (e.g. “Git from the command line and also from 3rd-party software”).
3. When asked “Choosing the default editor”, you can choose **Notepad** or **Visual Studio Code** if you have it.
4. Complete the installation.
5. **Verify:** Open **Command Prompt** or **PowerShell** and run:
   ```cmd
   git --version
   ```
   You should see something like `git version 2.43.0.windows.1`.

---

## 4. Install Python

Python is required if you want to run the scripts (ingest, query, etc.) **directly on your machine** instead of only inside Docker. The project expects **Python 3.11 or newer**.

### Steps

1. Go to **https://www.python.org/downloads/** and download **Python 3.11** or **3.12** for Windows.
2. Run the installer.
3. **Important:** On the first screen, check **“Add python.exe to PATH”**, then click **“Install Now”**.
4. Complete the installation (you may need to allow it for your user only).
5. **Verify:** Open a **new** Command Prompt or PowerShell and run:
   ```cmd
   python --version
   ```
   You should see something like `Python 3.11.9` or `Python 3.12.x`.

If `python` is not found, Python was not added to PATH. Rerun the installer and ensure “Add python.exe to PATH” is checked.

---

## 5. Install Visual Studio Code (optional)

VS Code is optional but useful for editing the `.env` file and viewing the project.

### Steps

1. Go to **https://code.visualstudio.com/** and download **VS Code for Windows**.
2. Run the installer and accept the defaults (e.g. “Add to PATH”).
3. Launch **Visual Studio Code** from the Start menu.

---

## 6. Install Docker Desktop

Docker runs the application (Qdrant, Streamlit app, and optionally Ollama) in containers. On Windows you need **Docker Desktop**.

### Steps

1. Go to **https://www.docker.com/products/docker-desktop/** and download **Docker Desktop for Windows**.
2. Run the installer. If it asks to enable **WSL 2**, accept and follow the instructions (you may need to restart).
3. After installation, **restart your PC** if prompted.
4. Start **Docker Desktop** from the Start menu. Wait until it says “Docker Desktop is running” (green icon in the system tray).
5. **Verify:** Open **PowerShell** or **Command Prompt** and run:
   ```cmd
   docker --version
   docker compose version
   ```
   You should see version numbers for both.

If `docker` is not found, make sure Docker Desktop is running and that you opened a **new** terminal after installation.

---

## 7. Clone the repository

This step copies the project from the repository to a folder on your computer.

### Steps

1. Open **Command Prompt** or **PowerShell**.
2. Go to the folder where you want the project (e.g. your user folder or `C:\Projects`):
   ```cmd
   cd C:\Users\YourUsername\Documents
   ```
   (Replace `YourUsername` with your Windows username.)
3. Clone the repository (replace `<repo-url>` with the actual Git URL provided by your team):
   ```cmd
   git clone <repo-url>
   ```
4. Go into the project folder:
   ```cmd
   cd Eva_Rsearch_AI
   ```
5. **Verify:** List the contents:
   ```cmd
   dir
   ```
   You should see folders like `docs`, `frontend`, `scripts`, and files like `docker_compose.yaml`, `README.md`.

---

## 8. Open the project in VS Code

If you use VS Code:

1. Open **Visual Studio Code**.
2. Go to **File → Open Folder**.
3. Select the **Eva_Rsearch_AI** folder (the one that contains `docker_compose.yaml` and `README.md`).
4. Click **Select Folder**.

You will create and edit the `.env` file in this folder next.

---

## 9. Create the `.env` file — shared settings for everyone

Create a file named **`.env`** in the **project root** (same folder as `docker_compose.yaml`). It is not committed to Git because it contains secrets.

### Steps

1. Create **`.env`** (leading dot, no `.txt`).
2. Paste the **shared block** below into `.env` first.
3. Then add the block for **Path A**, **B**, or **C** (Sections 10–12). **You do not need to fill in Azure or OpenAI if you only use Path A.**

### Shared block (all paths)

```env
# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------
CHUNK_SIZE=1200
CHUNK_OVERLAP=150

# ---------------------------------------------------------------------------
# Vector database (Qdrant)
# ---------------------------------------------------------------------------
VECTOR_DB=qdrant
# Docker (containers use service names):
VECTOR_DB_URL=http://qdrant:6333
QDRANT_COLLECTION_PREFIX=rag_

# ---------------------------------------------------------------------------
# Ollama — used on Path A; optional on B/C for extra answer models
# ---------------------------------------------------------------------------
LOCAL_LLM_PROVIDER=ollama
LOCAL_LLM_MODEL_PRIMARY=qwen2.5:7b-instruct
LOCAL_LLM_MODEL_SECONDARY=llama3.1:8b-instruct
OLLAMA_BASE_URL=http://ollama:11434

# ---------------------------------------------------------------------------
# Optional: Serper or other integrations
# ---------------------------------------------------------------------------
# SERPER_API_KEY=
```

**Host vs Docker URLs:** If you run **Python scripts on your PC** while Qdrant/Ollama run in Docker, use **`http://localhost:6333`** and **`http://localhost:11434`** (or set **`QDRANT_URL`** / **`OLLAMA_BASE_URL`** in the terminal — see Section 16).

---

## 10. Path A — Local LLMs only (skip Azure & OpenAI.com)

Use this path if you **do not** have Azure OpenAI or an OpenAI.com API key. Everything runs **locally** or in Docker: **BGE-M3** for embeddings and **Ollama** for answers.

### Add these lines to `.env`

```env
# ---------------------------------------------------------------------------
# Path A — Local embeddings + local answers (no Azure / no OpenAI.com)
# ---------------------------------------------------------------------------
EMBED_MODEL=bge_m3
DEFAULT_LLM=ollama_qwen2.5
```

You may use **`DEFAULT_LLM=ollama_llama3.1`** instead; pull the matching Ollama model (Section 14).

### What to skip

- Do **not** need to fill **Azure** variables (Section 11).
- Do **not** need **OPENAI_API_KEY** (Section 12).

### Notes

- First run may **download BGE-M3** weights (can be large); ensure disk space and a stable network.
- **Ingest** and **query** must use the same embedding (**bge_m3** → Qdrant vectors are **1024** dimensions). Do not mix with collections built using Azure/OpenAI embeddings unless you know what you are doing.
- Pull Ollama models after Docker is up (Section 14).

---

## 11. Path B — Azure OpenAI (Microsoft cloud)

Use this path if your organization uses **Azure OpenAI**: same resource for **embeddings** (Ada) and **chat** (your GPT deployment).

### Add these lines to `.env`

```env
# ---------------------------------------------------------------------------
# Path B — Azure OpenAI (embeddings + chat)
# ---------------------------------------------------------------------------
EMBED_MODEL=azure_ada
DEFAULT_LLM=azure

AZURE_OPENAI_API_KEY=your_azure_api_key_here
AZURE_OPENAI_DEPLOYMENT_NAME=your_deployment_name_here
AZURE_OPENAI_API_VERSION=2024-12-01-preview
AZURE_OPENAI_ENDPOINT=https://your-resource-name.openai.azure.com/
AZURE_OPENAI_MODEL=gpt-4.1
```

Replace placeholders with values from the **Azure Portal** (see [Section 13](#13-where-to-get-api-keys-quick-reference)).

### Optional

- Add **Ollama** models and use the sidebar to switch to **Ollama** for some answers.
- You can skip **Section 12** entirely if you do not use OpenAI.com.

---

## 12. Path C — OpenAI.com API (platform.openai.com)

Use this path if you have an **OpenAI** account and API key from **https://platform.openai.com** (not Azure).

### Add these lines to `.env`

```env
# ---------------------------------------------------------------------------
# Path C — OpenAI.com API (embeddings + chat)
# ---------------------------------------------------------------------------
OPENAI_API_KEY=sk-your-openai-secret-key-here
# Alternative variable name (same effect as OPENAI_API_KEY):
# OPEN_AI_KEY=sk-your-openai-secret-key-here
OPENAI_CHAT_MODEL=gpt-4o-mini
OPENAI_EMBEDDING_MODEL=text-embedding-3-small

EMBED_MODEL=openai_small
DEFAULT_LLM=openai
```

- **`OPENAI_API_KEY`** or **`OPEN_AI_KEY`** — Create at **https://platform.openai.com/api-keys**. Use either variable name in `.env` (not both with different values unless you intend to override).
- **`OPENAI_CHAT_MODEL`** — e.g. `gpt-4o-mini`, `gpt-4o`, `gpt-3.5-turbo` (must be enabled for your account).
- **`OPENAI_EMBEDDING_MODEL`** — Default `text-embedding-3-small` (1536 dimensions). Ingested collections use this size.
- **`EMBED_MODEL=openai_small`** — Selects OpenAI embeddings in the app/registry.
- **`DEFAULT_LLM=openai`** — Default answer model in the UI is OpenAI.com chat.

### Mixing with other providers (advanced)

- You can set **`DEFAULT_LLM=ollama_qwen2.5`** and still keep **`OPENAI_API_KEY`** if you want OpenAI only for embeddings — use the **sidebar** to pick the answer model.
- **Azure** variables are **not** required for Path C unless you also want Azure in the sidebar.

---

## 13. Where to get API keys (quick reference)

### Azure OpenAI (Path B)

| Variable | Where to get it |
|----------|------------------|
| **AZURE_OPENAI_API_KEY** | Azure Portal → your **Azure OpenAI** resource → **Keys and Endpoint** → copy a key. |
| **AZURE_OPENAI_ENDPOINT** | Same page → **Endpoint** (must end with `/`). |
| **AZURE_OPENAI_DEPLOYMENT_NAME** | Azure OpenAI → **Deployments** → deployment name of your **chat** model (e.g. GPT-4). |
| **AZURE_OPENAI_API_VERSION** | Portal / docs for your resource; e.g. `2024-12-01-preview`. |
| **AZURE_OPENAI_MODEL** | Model name behind that deployment (e.g. `gpt-4.1`). |

**Short steps:** Sign in to **https://portal.azure.com** → open **Azure OpenAI** → **Keys and Endpoint** + **Model deployments**.

---

### OpenAI.com (Path C)

| Variable | Where to get it |
|----------|------------------|
| **OPENAI_API_KEY** or **OPEN_AI_KEY** | **https://platform.openai.com** → sign in → **API keys** → **Create new secret key**. |
| **OPENAI_CHAT_MODEL** | Product docs / model list; common default: **`gpt-4o-mini`**. |
| **OPENAI_EMBEDDING_MODEL** | Usually **`text-embedding-3-small`** (check OpenAI embedding docs). |

**Billing:** OpenAI.com API usage is billed to your OpenAI account; ensure billing is enabled if required.

---

### Chunking, Qdrant, Ollama

- **CHUNK_SIZE / CHUNK_OVERLAP** — No external account; defaults are fine.
- **VECTOR_DB_URL** — `http://qdrant:6333` in Docker; `http://localhost:6333` when running scripts on the host against Docker Qdrant.
- **OLLAMA_BASE_URL** — `http://ollama:11434` in Docker; `http://localhost:11434` on host.
- **Ollama models** — After containers are up: `docker exec -it rag-ollama ollama pull qwen2.5:7b-instruct` (and `llama3.1:8b` if needed).

---

### DEFAULT_LLM values (reference)

| Value | Meaning |
|-------|---------|
| **azure** | Default answers: Azure OpenAI chat (Path B). |
| **openai** | Default answers: OpenAI.com chat (Path C). |
| **ollama_qwen2.5** | Default answers: Ollama primary model. |
| **ollama_llama3.1** | Default answers: Ollama secondary model. |

---

## 14. Run the project with Docker

After your `.env` file is saved in the project root:

1. Open **PowerShell** or **Command Prompt** and go to the project folder:
   ```cmd
   cd C:\Users\YourUsername\Documents\Eva_Rsearch_AI
   ```
2. Build (first time or after Dockerfile changes):
   ```cmd
   docker compose -f docker_compose.yaml build
   ```
3. Start services:
   ```cmd
   docker compose -f docker_compose.yaml up -d
   ```
4. Check status:
   ```cmd
   docker compose -f docker_compose.yaml ps
   ```
   You should see **qdrant**, **rag-api**, and **ollama** (for Path A or optional Ollama) as **Up**.

5. **If you use Ollama (Path A or optional):** pull models once:
   ```cmd
   docker exec -it rag-ollama ollama pull qwen2.5:7b-instruct
   docker exec -it rag-ollama ollama pull llama3.1:8b
   ```

If a port is already in use (6333, 8501, 11434), stop the conflicting program or change ports in `docker_compose.yaml` (advanced).

---

## 15. Open the application in your browser

1. Open **http://localhost:8501**
2. In the **sidebar**:
- **Embedding / answer provider** — **Azure OpenAI**, **OpenAI (API)** (requires `OPENAI_API_KEY` or `OPEN_AI_KEY`), or **Local**.
- **Embedding model** — Filtered by provider; must match how documents were ingested (1536 vs 1024 dim).
- **Answer model** — Filtered by provider (**Azure** path: Azure + Ollama; **OpenAI** path: OpenAI + Ollama; **Local**: Ollama only).
3. Add PDFs/DOCX under **`data\<collection_name>\`** or use **Upload** in the app.
4. Ask a question in the chat box.

---

## 16. Optional: Run scripts from your machine

When scripts run on the **host**, point them at Qdrant/Ollama on **localhost**:

**PowerShell:**
```powershell
$env:PYTHONPATH = (Get-Location).Path
$env:QDRANT_URL = "http://localhost:6333"
$env:OLLAMA_BASE_URL = "http://localhost:11434"
$env:STATE_ROOT = "state"
python scripts/index_text.py --data-root ./data --collection hydrogen_books
```

**CMD:**
```cmd
set PYTHONPATH=%CD%
set QDRANT_URL=http://localhost:6333
set OLLAMA_BASE_URL=http://localhost:11434
set STATE_ROOT=state
python scripts/index_text.py --data-root ./data --collection hydrogen_books
```

`scripts/answer_local.py` calls **Ollama only** — use it when testing Path A (or Ollama answers).

---

## 17. Troubleshooting

| Problem | What to do |
|--------|-------------|
| **`git` / `python` / `docker` not found** | Install the tool (Sections 3–6), then open a **new** terminal. |
| **Connection refused** | Start Qdrant (and Ollama if needed). On host scripts, use **`localhost`** URLs (Section 16). |
| **Azure errors** | Check Path B variables (Section 11, 13). |
| **OpenAI API error** | Check **OPENAI_API_KEY** or **OPEN_AI_KEY** and **OPENAI_CHAT_MODEL**; verify billing/limits on platform.openai.com. |
| **“No Azure chat client configured”** | You selected **Azure** in the sidebar but Path A/C only — pick **OpenAI** or **Ollama**, or add Azure to `.env`. |
| **“OpenAI chat is not configured”** | Set **OPENAI_API_KEY** or **OPEN_AI_KEY**, choose **OpenAI (API)** provider and an OpenAI answer model, restart the app. |
| **OpenAI (API) missing from provider list** | Add a valid OpenAI.com API key to `.env`; the UI hides this provider if no key is set. |
| **Wrong embedding dimension / empty folders** | Collections are tied to vector size. Use the same **EMBED_MODEL** for ingest and query; or re-ingest with one embedding type. |
| **Ollama model not found** | Run `docker exec -it rag-ollama ollama pull <model>` (Section 14). |

---

## Quick reference

| Item | Value |
|------|--------|
| **Project root** | Folder with `docker_compose.yaml`, `.env` |
| **Start stack** | `docker compose -f docker_compose.yaml up -d` |
| **Stop stack** | `docker compose -f docker_compose.yaml down` |
| **App URL** | **http://localhost:8501** |
| **More detail** | **README.md**, **docs/M1_Requirements.md** |

---

*End of User Guide*
