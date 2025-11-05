# PHANTOM: Pricing Heuristics Against Non-human Transaction Orchestration Mechanisms

[![Build PDF](https://github.com/velocitatem/PHANTOM/actions/workflows/latex.yml/badge.svg)](https://github.com/velocitatem/PHANTOM/actions/workflows/latex.yml)

> Bachelor's Thesis Project by Daniel Rösel, IE University Madrid (2025)
> Advisor: Alberto Martín Izquierdo

## Overview

PHANTOM is an academic research project investigating **pricing heuristics to protect e-commerce platforms from exploitation by LLM agents** in dynamic pricing environments. This project explores behavioral signature detection for agent identification and develops protection mechanisms against automated transaction orchestration.

**Research Focus Areas:**
- AI security in e-commerce systems
- Behavioral signature detection for autonomous agent identification
- Dynamic pricing protection mechanisms
- LLM agent behavior analysis and exploitation patterns

## Project Structure

This repository contains both the research thesis and a full-stack experimental platform:

```
PHANTOM/
├── paper/              # LaTeX Bachelor's Thesis
│   ├── src/           # LaTeX source files
│   ├── build/         # Compiled PDF output
│   └── concat_code.sh # Auto-concatenate code for appendix
│
├── docs/              # GitHub Pages academic project page
│   └── index.html     # Academic project showcase
│
├── web/               # Next.js research platform dashboard
│   └── src/           # Frontend application
│
├── backend/           # Python backend services
│   ├── provider/      # Data provider service
│   └── worker/        # Kafka consumer worker
│
├── experiments/       # Research experiments and data analysis
│   └── data_export.ipynb
│
└── docker/            # Infrastructure configurations
```

## Quick Start

### Prerequisites

- Docker & Docker Compose (for infrastructure)
- Node.js 18+ (for web application)
- Python 3.9+ (for backend services)
- LaTeX distribution (for building thesis paper)

### Infrastructure Setup

Start Kafka, Redis, and monitoring services:

```bash
docker-compose up -d
```

**Services:**
- Redis: `localhost:6379`
- Kafka: `localhost:9092`
- Zookeeper: `localhost:2181`
- Redpanda Console (Kafka UI): `http://localhost:8080`

### Web Application

```bash
cd web
npm install
npm run dev
```

Access at `http://localhost:3000`

### Backend Services

Install dependencies:
```bash
pip install -r requirements.txt
```

Run worker:
```bash
cd backend/worker
python main.py
```

## Building the Thesis Paper

### Using Make

```bash
make pdf        # Compile LaTeX to PDF
make watch      # Continuous compilation (live preview)
make clean      # Remove build artifacts
```

### Manual Build

```bash
cd paper/src
latexmk -pdf main.tex
```

**Output**: `paper/build/main.pdf`

### Automated CI/CD

The thesis PDF is automatically built via GitHub Actions on every push to `main` that affects `paper/**`. The compiled PDF artifact is available in the Actions tab.

## Technical Architecture

**Frontend**: Next.js 14, React 18, TypeScript, Tailwind CSS
**Backend**: Python, Kafka (event streaming)
**Infrastructure**: Redis (cache), Kafka + Zookeeper
**Monitoring**: Redpanda Console
**Data Analysis**: Jupyter Notebooks, Pandas, Matplotlib

### Event-Driven Architecture

The platform uses Kafka for real-time event streaming, enabling:
- Asynchronous task processing
- Scalable data collection
- Experiment tracking and analysis
- Behavioral pattern detection

## Research Experiments

Jupyter notebooks in `experiments/` contain:
- Data exploration and analysis
- Behavioral pattern visualizations
- Statistical analysis of agent behaviors
- Experiment result processing

Run experiments:
```bash
cd experiments
jupyter notebook data_export.ipynb
```

## Documentation

- **Academic Project Page**: Hosted on GitHub Pages at `/docs`
- **Thesis Paper**: Latest PDF available via GitHub Actions artifacts
- **Web App README**: See `web/README.md`
- **Backend READMEs**: See `backend/provider/README.md` and `backend/worker/README.md`

## Development Workflow

1. **Paper Development**: Edit LaTeX files in `paper/src/`, use `make watch` for live preview
2. **Web Development**: Standard Next.js workflow in `web/`
3. **Backend Development**: Python services in `backend/`
4. **Experiments**: Jupyter notebooks in `experiments/`

### Code in Thesis Appendix

The `paper/concat_code.sh` script automatically generates a LaTeX appendix containing all source code from:
- `backend/` (Python, JavaScript, Shell, YAML)
- `experiments/` (Analysis scripts)
- `docker/` (Infrastructure configs)
- `web/src/` (TypeScript/React components)

This runs automatically during PDF compilation.

## Contributing

This is an academic thesis project. For questions or collaboration inquiries, please open an issue.

## License

Academic research project - all rights reserved.

## Citation

```bibtex
@thesis{rosel2025phantom,
  title={Pricing Heuristics Against Non-human Transaction Orchestration Mechanisms},
  author={Rösel, Daniel},
  year={2025},
  school={IE University},
  address={Madrid, Spain},
  type={Bachelor's Thesis}
}
```

## Acknowledgments

Special thanks to Alberto Martín Izquierdo for academic supervision and guidance throughout this research project.
