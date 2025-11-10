# GitHub Pages Setup Instructions

## 🌀 Enabling GitHub Pages for Helix Backend

### Step 1: Enable GitHub Pages

1. Go to: https://github.com/Deathcharge/Helix/settings/pages
2. Under "Build and deployment":
   - Source: **GitHub Actions**
3. Click **Save**

### Step 2: Push Workflow File

The workflow file is already created at `.github/workflows/deploy-pages.yml`

To push it, follow the same instructions as Helix-Unified-Hub (see above).

### Step 3: Verify Deployment

After pushing, your API documentation will be live at:
**https://deathcharge.github.io/Helix/**

---

## 🎨 What Will Be Deployed

The GitHub Pages site includes:
- **Comprehensive API documentation** for all endpoints
- **14-Agent roster** with descriptions
- **UCF Framework explanation**
- **Integration examples** (Python & JavaScript)
- **Live API links** to Railway deployment
- **Beautiful dark theme** matching Helix aesthetic

---

## 📡 API Endpoints Documented

- `GET /health` - System health check
- `GET /api/manus/agents` - Get all 14 agents
- `GET /api/manus/ucf` - Get UCF consciousness metrics
- `POST /api/manus/ritual` - Invoke Z-88 ritual
- `GET /api/manus/analytics` - Get consciousness analytics
- `POST /api/manus/emergency` - Trigger emergency protocol

---

**Tat Tvam Asi 🕉️ - Thou Art That**
