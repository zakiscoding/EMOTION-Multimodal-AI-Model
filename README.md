# EMOTION-Multimodal-AI-Model
A full-stack **multimodal emotion / sentiment analysis** project:  
- **Model training + deployment code (Python)** in `multimodal/`
- A **web app (Next.js / T3 stack)** in `emotion_analyzer_saas/` that can authenticate users and call inference APIs

> Repo author: zakiscoding

---

## What’s inside

### 1) `multimodal/` (ML training + inference + deployment)
This folder contains the end-to-end pipeline for training and serving a multimodal model (video/audio/text style workflow) including:
- **Training scripts** (`multimodal/training/`)
- **Deployment + inference helpers** (`multimodal/deployment/`)
- **SageMaker training launcher** (`multimodal/train_sagemaker.py`)
- Saved evaluation outputs (example `metrics.json`) under `multimodal/deployment/model*/`  

Key files & folders:
- `multimodal/training/train.py` – main training entry
- `multimodal/training/meld_dataset.py` – dataset logic
- `multimodal/training/models.py` – model architecture(s)
- `multimodal/deployment/inference.py` – inference logic
- `multimodal/deployment/deploy_endpoint.py` – deployment helper
- `multimodal/deployment/models.py` – deployment-time model code
- `multimodal/deployment/requirements.txt` + `multimodal/training/requirements.txt` – dependencies

> Note: model weights (`*.pth`) are intentionally ignored by git. Add your own weights locally or load them from storage. :contentReference[oaicite:1]{index=1}

---

### 2) `emotion_analyzer_saas/` (SaaS UI + API)
A web app built with the **T3 Stack** (Next.js + NextAuth + Prisma + Tailwind, etc.). :contentReference[oaicite:2]{index=2}  
It includes:
- Auth pages (`/login`, `/signup`)
- API routes for inference + upload URL generation
- Client components for uploading and running inference

Key routes / code areas:
- `emotion_analyzer_saas/src/app/api/sentiment-inference/route.ts` – inference endpoint wiring :contentReference[oaicite:3]{index=3}
- `emotion_analyzer_saas/src/app/api/upload-url/route.ts` – upload URL endpoint wiring :contentReference[oaicite:4]{index=4}
- `emotion_analyzer_saas/src/components/client/UploadVideo.tsx` – upload UI :contentReference[oaicite:5]{index=5}
- `emotion_analyzer_saas/src/components/client/Inference.tsx` – inference UI :contentReference[oaicite:6]{index=6}

---

## Quickstart

### A) Run the SaaS app locally (`emotion_analyzer_saas/`)

#### 1) Install deps
```bash
cd emotion_analyzer_saas
npm install
2) Set environment variables
Copy the example env file and fill in your secrets:
cp .env.example .env
The repo currently expects these env vars (at minimum):
AUTH_SECRET
AUTH_DISCORD_ID
AUTH_DISCORD_SECRET
DATABASE_URL (defaults to sqlite) 
Tip: You can generate an auth secret via:
npx auth secret
3) Initialize Prisma + DB
npx prisma generate
npx prisma db push
4) Start dev server
npm run dev
B) Train / run ML locally (multimodal/)
Your training and deployment environments are separated in:
multimodal/training/requirements.txt
multimodal/deployment/requirements.txt 
Typical workflow:
Create a venv
Install dependencies from the appropriate requirements file
Run training from multimodal/training/train.py
Export weights to wherever your inference step expects them
Example (template):
cd multimodal
python -m venv .venv
source .venv/bin/activate

pip install -r training/requirements.txt
python training/train.py
Project structure (high level)
EMOTION-Multimodal-AI-Model/
├── emotion_analyzer_saas/        # Next.js SaaS app (UI + API routes)
│   ├── prisma/
│   └── src/
│       ├── app/api/
│       │   ├── sentiment-inference/
│       │   └── upload-url/
│       └── components/client/
└── multimodal/                   # Python ML pipeline (train + deploy)
    ├── training/
    └── deployment/
Notes
Large artifacts like *.pth, datasets, archives, and build output are gitignored by design. 
If you deploy the model behind an API (SageMaker endpoint, container, etc.), the SaaS layer can be pointed to that inference URL/route.
