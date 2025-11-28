# Hidden Connections - Project 4 (Maximum Features)

> *People who have never spoken can end up adjacent in semantic space purely from how they describe their inner worlds.*

An interactive web-based digital artwork that visualizes latent semantic relationships between anonymous participants based on their responses to psychological-style questions.

## Features

### Embedding Models (6 backends)
- **BGE** - BAAI/bge-large-en-v1.5 (excellent quality)
- **Sentence Transformers** - all-MiniLM-L6-v2 (fast, lightweight)
- **CLIP** - OpenAI CLIP models (multimodal)
- **Nomic** - nomic-embed-text-v1.5 (good quality, permissive)
- **OpenAI API** - text-embedding-3-large (requires API key)
- **Instructor** - Custom instruction embeddings

### Visualization
- **4 Viewing Modes**: Clusters, Social Energy, Decision Style, Region
- **Smooth Color Transitions**: Animated transitions when switching modes
- **Nearest-Neighbor Highlighting**: Toggle to see 5 closest semantic neighbors
- **Connection Lines**: Visual lines connecting neighbors on hover
- **Polished Glassmorphism UI**: Exhibition-ready aesthetic

### Interaction
- **Hover Panel**: Shows full responses and metadata
- **Touch Support**: Works on tablets
- **Keyboard Shortcuts**: Quick mode switching

## Quick Start

```bash
cd 4

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r pipeline/requirements.txt

# Option 1: Generate synthetic data
python pipeline/generate_synthetic.py -n 50 --stats

# Option 2: Use your own CSV data in data/responses_projective.csv

# Process with ML pipeline
python pipeline/process.py

# View visualization
cd web && python -m http.server 8000
# Open http://localhost:8000
```

## Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `1` | Clusters view (ML-detected) |
| `2` | Energy view (social energy) |
| `3` | Decision view (decision style) |
| `4` | Region view (geographic) |
| `N` | Toggle neighbor highlighting |
| `?` | Toggle help modal |
| `Esc` | Close modal |

## Embedding Configuration

Edit `pipeline/config.yaml`:

```yaml
# Fast & lightweight (default for testing)
embedding:
  type: "sentence-transformers"
  model: "all-MiniLM-L6-v2"

# Best open-source quality
embedding:
  type: "bge"
  model: "BAAI/bge-large-en-v1.5"

# CLIP (multimodal concepts)
embedding:
  type: "clip"
  model: "openai/clip-vit-large-patch14"

# OpenAI API (requires OPENAI_API_KEY)
embedding:
  type: "openai"
  model: "text-embedding-3-large"
```

## Data Schema

### Input CSV (data/responses_projective.csv)

| Field | Type | Description |
|-------|------|-------------|
| `id` | string | Unique participant ID |
| `q1_safe_place` | string | "Describe a place where you feel very safe" |
| `q2_stress` | string | "Think of a recent time you felt stressed" |
| `q3_understood` | string | "Describe a moment when you felt understood" |
| `q4_free_day` | string | "If you had a completely free day..." |
| `q5_one_word` | string | "One word to describe yourself, and why" |
| `q6_decision_style` | enum | Mostly rational / Mostly emotional / Depends |
| `q7_social_energy` | enum | Energised / Drained / Depends |
| `q8_region` | string | Geographic region |
| `nickname` | string | Optional display name |

### Output JSON (processed/points.json)

```json
{
  "id": "p_042",
  "text": "Q1 (safe place): ... \nQ2 (stress): ...",
  "x": 0.342,
  "y": -1.127,
  "cluster": 2,
  "decision_style": "Mostly rational",
  "social_energy": "Drained",
  "region": "Europe",
  "nickname": "quietwindow"
}
```

## Project Structure

```
4/
├── pipeline/
│   ├── config.yaml           # All configuration
│   ├── process.py            # ML pipeline (embeddings → UMAP → KMeans)
│   ├── generate_synthetic.py # Synthetic data generator
│   └── requirements.txt      # Python dependencies
├── data/
│   └── responses_projective.csv
├── processed/
│   └── points.json
├── web/
│   ├── index.html            # Frontend
│   ├── style.css             # Glassmorphism styles
│   ├── main.js               # Canvas visualization
│   └── points.json           # Copy for serving
└── README.md
```

## Color Palettes

### Cluster Colors (ML-detected)
- Cluster 0: `#4ECDC4` (teal)
- Cluster 1: `#FF6B6B` (coral)
- Cluster 2: `#C9B1FF` (lavender)
- Cluster 3: `#FFE66D` (warm yellow)
- Cluster 4: `#95E1D3` (mint)
- Cluster 5: `#F38181` (salmon)

### Social Energy
- Energised: `#FFD93D` (bright gold)
- Drained: `#6C5CE7` (deep violet)
- Depends: `#A8E6CF` (soft green)

### Decision Style
- Mostly rational: `#74B9FF` (sky blue)
- Mostly emotional: `#FD79A8` (pink)
- Depends: `#FFEAA7` (light gold)

## Features Inherited

| Source | Features |
|--------|----------|
| **P1** | Baseline pipeline, question formatting |
| **P2** | Synthetic data generator with semantic clusters |
| **P3** | Config-driven fields, sklearn normalization |
| **P4** | 6 embedding backends, 4 view modes, neighbor highlighting, smooth transitions, touch support |

## Success Metrics

Since this is an art piece:
- **Contemplative engagement**: Viewers spend time exploring
- **Moment of recognition**: Viewers find semantically similar points
- **Conversation starter**: Prompts discussion about algorithmic interpretation

---

*Project 4 · Maximum Features Edition*
