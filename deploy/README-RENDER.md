# Deploying Eva Research AI on Render

This guide outlines how to deploy the decoupled React + FastAPI codebase of **Eva Research AI** using **Render Blueprints**.

## Deployment Blueprint (`render.yaml`)

The project includes a `render.yaml` file at the root. Render uses this blueprint to automatically spin up and wire together:
1. **`eva-research-ai-backend`**: A Python Web Service running FastAPI.
2. **`eva-research-ai-frontend`**: A Static Site building and hosting the Vite React app.

---

## Deployment Steps

### 1. Push to GitHub/GitLab
Ensure your codebase is pushed to a Git repository (such as GitHub, GitLab, or Bitbucket) linked to your Render account.

### 2. Launch the Blueprint
1. Log in to the **[Render Dashboard](https://dashboard.render.com)**.
2. Click **New +** and select **Blueprint**.
3. Select your repository containing the Eva Research AI project.
4. Render will parse the `render.yaml` file and prompt you for configuration parameters.

### 3. Configure Environment Variables
In the Render setup UI, configure the following secrets/variables for the **Backend Service**:

| Environment Variable | Description | Example / Value |
|----------------------|-------------|-----------------|
| `OPENAI_API_KEY` | Your OpenAI API secret key | `sk-proj-xxxx...` |
| `QDRANT_URL` | The HTTPS url of your Qdrant instance (e.g. Qdrant Cloud) | `https://xxxx.gcp.qdrant.io:6333` |
| `QDRANT_API_KEY` | (Optional) The API key for your Qdrant Cloud cluster | `xxxx` |
| `DEFAULT_LLM` | Default provider for generating answers | `openai` |
| `EMBED_MODEL` | Default embedding model | `openai_small` |

*Note: Render automatically compiles the React frontend using the backend's dynamic URL via the `VITE_API_URL` environment variable interpolation in `render.yaml`.*

---

## Technical Notes

### Decoupled Routing & CORS
- The backend is configured to accept CORS from all origins by default (`allow_origins=["*"]` in `backend/app/main.py`), ensuring that the Static React app can successfully communicate with the backend endpoint even when hosted on a different Render subdomain.
- In production, you can lock `allow_origins` to your frontend's specific Render URL (e.g., `https://eva-research-ai-frontend.onrender.com`).

### File Storage / State Persistence
- By default, Render Web Services run on ephemeral disk storage.
- Session histories are stored as local JSON files inside `./sessions`. 
- If you require persistent sessions and statistics across backend redeploys/restarts on Render, attach a **Persistent Disk** to the backend service in your Render dashboard:
  - **Mount Path**: `/app/sessions`
  - **Size**: `1 GB` (or larger depending on usage)
