# Contributing to PikoGPT - Team FunkyAI

This document outlines team roles, responsibilities, and workflow for the PikoGPT project.

## Team Roles

### Filipp - Data Engineer 🗂️

**Focus:** Data pipeline and model evaluation

| Responsibility | Description | Key Files |
|----------------|-------------|-----------|
| Data Quality | Monitor and improve data preprocessing | `src/data/preprocessing.py` |
| Benchmarks | Implement LAMBADA, HellaSwag, WinoGrande, OpenBookQA | `src/evaluation/` |
| Leaderboard | Prepare and submit leaderboard evaluations | Evaluation scripts |
| Results Analysis | Analyze training logs, create visualizations | `notebooks/` |
| Metrics | Implement perplexity, accuracy calculations | `src/evaluation/metrics.py` |

**Deliverables:**
- [ ] Benchmark implementation (Week 6-7)
- [ ] First leaderboard submission (Week 7)
- [ ] Results analysis notebook (Week 8)
- [ ] Poster draft (Week 10)
- [ ] Final poster (Week 11)
- [ ] Final evaluation report (Week 11)

---

### Roman - Platform Engineer ⚙️

**Focus:** Infrastructure and product experience

**Completed Work:** ✅
- Training pipeline (`src/training/stage.py`)
- Inference pipeline (`src/inference/stage.py`)
- Config system (`src/training/config.py`)
- CLI entry point (`main.py`)
- Checkpoint format design

| Responsibility | Description | Key Files |
|----------------|-------------|-----------|
| Chat Interface | Build Gradio/Streamlit demo | `src/interface/chat.py` |
| Documentation | README, docstrings, user guides | `README.md`, docstrings |
| Code Quality | Testing, error handling | `tests/` |

**Deliverables:**
- [ ] Chat interface prototype (Week 9)
- [ ] Polished demo (Week 10)
- [ ] Poster draft (Week 10)
- [ ] Final poster (Week 11)
- [ ] Tech report sections (Week 12)

---

### Arabella - ML Engineer 🧠

**Focus:** Model architecture and training optimization

**Completed Work:** ✅
- Architecture configurations (`configs/train_*.toml`)
- Training utilities (`src/training/utils.py`)
- LR warmup scheduling
- Architecture documentation

| Responsibility | Description | Key Files |
|----------------|-------------|-----------|
| Architecture | Design and tune model configs | `configs/` |
| Hyperparameters | Learning rate, batch size, warmup | Config files |
| Training Runs | Execute and monitor GPU training | Training scripts |
| Ablations | Test architectural variations | Experiment configs |
| Post-Training | Explore fine-tuning approaches | Post-training code |

**Deliverables:**
- [ ] GPU training run (Week 6)
- [ ] Hyperparameter tuning (Week 7)
- [ ] Ablation study (Week 8)
- [ ] Final model selection (Week 9)
- [ ] Poster draft (Week 10)
- [ ] Final poster (Week 11)
- [ ] Architecture section for report (Week 12)

---

## Work Distribution Summary

To ensure balanced workload given Roman's significant early contributions:

| Phase | Filipp | Roman | Arabella |
|-------|--------|-------|----------|
| Weeks 1-4 | EDA, preprocessing | Training + Inference code | Architecture research |
| Week 5 | Data verification | Code polish | Configs, warmup |
| Weeks 6-7 | **Benchmarks** | Light support | **GPU training** |
| Weeks 8-9 | **Evaluation** | **Chat interface** | Ablations, tuning |
| Weeks 10-11 | Results for poster | Combine results for poster | Model documentation |
| Week 12 | Report: Data section | Report: System section | Report: Model section |


---

## Workflow

### Git Branching Strategy

```
main (protected - always working)
  │
  └── dev (integration branch)
        │
        ├── feature/benchmarks        ← Filipp
        ├── feature/chat-interface    ← Roman
        └── feature/training-runs     ← Arabella
```

### Branch Rules

1. **Never push directly to `main`**
2. Create feature branches for all work: `feature/<description>`
3. Open PR to `dev` when ready for review
4. Get at least 1 approval before merging
5. Merge `dev` → `main` weekly after testing

### Commit Messages

Use conventional commits:
```
feat: Add LAMBADA benchmark implementation
fix: Resolve tokenizer padding issue
docs: Update README with training instructions
refactor: Simplify checkpoint loading
test: Add unit tests for data preprocessing
```

---

## Key Dates

| Date | Milestone |
|------|-----------|
| Week 5 | TA Check-in ✅ |
| Week 7 | First leaderboard submission |
| Week 9 | Chat interface working |
| May 11 | Poster submission |
| June 2 | Final submission (code + report) |

---

## File Ownership

Clear ownership helps avoid conflicts:

| Directory | Primary Owner | Reviewers |
|-----------|---------------|-----------|
| `src/data/` | Filipp | Arabella |
| `src/evaluation/` | Filipp | Roman |
| `src/training/` | Arabella | Roman |
| `src/inference/` | Roman | Arabella |
| `src/interface/` | Roman | Filipp |
| `configs/` | Arabella | Roman |
| `notebooks/` | Filipp | All |
| `main.py` | Roman | All |
| `README.md` | Roman | All |

---


## Decision Log

Track important decisions here:

| Date | Decision | Rationale | Decided By |
|------|----------|-----------|------------|
| Week 3 | Use 37M param model | Best use of 40M budget | All |
| Week 4 | 512 context length | Balance memory/capability | Arabella |
| Week 5 | Add LR warmup | Improves training stability | Arabella |
| | | | |

---

*Last updated: Week 5*
