# Option Pricing — HASTS 416/7 Group Work Project 1

## Group Members
> ***Will add group members here***

> **Stochastic Processes · Honours Level (Part IV)**  
> A full options-pricing pipeline in Python: stochastic-volatility calibration, exotic option pricing, interest-rate modelling, and Monte-Carlo simulation.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Team Structure](#team-structure)
- [Repository Layout](#repository-layout)
- [Notebook Guide](#notebook-guide)
- [Mathematical Models](#mathematical-models)
- [Quick Start](#quick-start)
- [Environment Setup](#environment-setup)
- [Data](#data)
- [Collaboration Workflow](#collaboration-workflow)
- [Submission Checklist](#submission-checklist)
- [References](#references)

---

## Project Overview

This project prices a range of derivative instruments on **SM Energy Company (SM)** — currently trading at **$232.90 USD** — using state-of-the-art stochastic-volatility and jump-diffusion models calibrated to live market option prices.

The work is organised into four steps:

| Step | Description | Responsible |
|------|-------------|-------------|
| **1** | Heston (1993) calibration (Lewis & Carr-Madan) + ATM Asian option pricing | All sub-groups |
| **2** | Bates (1996) calibration + Put option pricing (60–70-day horizon) | All sub-groups |
| **3** | CIR (1985) interest-rate model calibration + Euribor simulation | Full team |
| **4** | Final report assembly and code review | Full team |

---

## Team Structure

| Sub-group | Members | Primary Tasks |
|-----------|---------|---------------|
| **Sub-group 1** | Members 1 – 3 | Step 1a (Heston–Lewis), Step 2e (Bates–Carr-Madan) |
| **Sub-group 2** | Members 4 – 6 | Step 1b (Heston–Carr-Madan), Step 2f (Put pricing) |
| **Sub-group 3** | Members 7 – 10 | Step 1c (Asian option), Step 2d (Bates–Lewis) |
| **Full team** | Members 1 – 10 | Steps 3g, 3h, 4 (CIR, simulation, report) |

---

## Repository Layout

```
option-pricing/
│
├── datasets/
│   └── SM_options_data.xlsx          # Market option prices for SM Energy
│
├── notebooks/
│   ├── 00_project_overview.ipynb     # Setup, data exploration, common utilities
│   ├── 01_step1a_heston_lewis.ipynb  # Step 1a — Heston via Lewis (2001)
│   ├── 02_step1b_heston_carr_madan.ipynb  # Step 1b — Heston via Carr-Madan (1999)
│   ├── 03_step1c_asian_option.ipynb  # Step 1c — ATM Asian call, Monte-Carlo
│   ├── 04_step2d_bates_lewis.ipynb   # Step 2d — Bates via Lewis (2001)
│   ├── 05_step2e_bates_carr_madan.ipynb   # Step 2e — Bates via Carr-Madan (1999)
│   ├── 06_step2f_put_pricing.ipynb   # Step 2f — Put option pricing (70-day)
│   ├── 07_step3g_cir_calibration.ipynb    # Step 3g — CIR model calibration
│   └── 08_step3h_cir_simulation.ipynb     # Step 3h — Euribor MC simulation
│
├── reports/
│   └── final/                        # PDF report (answers only, no code)
│
├── environments.yml                  # Conda environment specification
├── requirements.txt                  # pip requirements
├── LICENSE
└── README.md
```

---

## Notebook Guide

### Step 1 — Short-maturity instruments (~15 days)

| Notebook | Task | Key output |
|----------|------|------------|
| `01_step1a_heston_lewis.ipynb` | Calibrate Heston (1993) using the **Lewis (2001)** single-integral formula. MSE minimisation via Differential Evolution + Nelder-Mead. | 5 calibrated parameters: κ, θ, σ, ρ, v₀ |
| `02_step1b_heston_carr_madan.ipynb` | Repeat calibration using the **Carr-Madan (1999)** FFT approach. Compare results with 01. | Parameter comparison table |
| `03_step1c_asian_option.ipynb` | Price a **20-day ATM Asian call** via Monte-Carlo under the calibrated Heston model. Apply 4% bank fee. | Fair value + client price |

### Step 2 — Medium-maturity instruments (60–70 days)

| Notebook | Task | Key output |
|----------|------|------------|
| `04_step2d_bates_lewis.ipynb` | Calibrate **Bates (1996)** model (Heston + Merton jumps) via Lewis. Target: 60-day options. | 8 calibrated parameters |
| `05_step2e_bates_carr_madan.ipynb` | Repeat Step 2d using Carr-Madan FFT. | Cross-validation of Bates parameters |
| `06_step2f_put_pricing.ipynb` | Price a **70-day put** at 95% moneyness (K = 0.95 × S₀) via Monte-Carlo. | Put fair value |

### Step 3 — Interest-rate modelling

| Notebook | Task | Key output |
|----------|------|------------|
| `07_step3g_cir_calibration.ipynb` | Build Euribor term structure (cubic spline interpolation). Calibrate **CIR (1985)** model. | κ, θ, σ for CIR; term-structure fit plot |
| `08_step3h_cir_simulation.ipynb` | 100,000 MC paths of 12-month Euribor over 1 year. Confidence intervals, expected rate, pricing impact. | Rate distribution plots & statistics |

---

## Mathematical Models

### Heston (1993) — Stochastic Volatility

The stock and its variance follow correlated SDEs under the risk-neutral measure:

```
dS_t  = r·S_t·dt + sqrt(v_t)·S_t·dW₁
dv_t  = κ(θ − v_t)·dt + σ·sqrt(v_t)·dW₂
dW₁·dW₂ = ρ·dt
```

Parameters: `κ` (mean-reversion speed), `θ` (long-run variance), `σ` (vol-of-vol), `ρ` (leverage correlation), `v₀` (initial variance).

**Feller condition:** `2κθ > σ²` ensures variance stays strictly positive.

### Bates (1996) — Heston + Jumps

Extends Heston with compound-Poisson jumps in the log-price:

```
dln(S_t) = (r − λμ_J)·dt + sqrt(v_t)·dW₁ + J·dN_t
```

Three additional parameters: `λ` (jump intensity), `μ_J` (mean jump size), `σ_J` (jump-size std dev).

### Lewis (2001) — Single-Integral Option Pricing

```
C = S₀ − (K·e^{−rT}/π) · ∫₀^∞  Re[e^{izk}·φ(z−i/2)] / (z²+¼) dz
```

where `k = ln(F/K)` (log-moneyness) and `φ` is the model characteristic function.

### Carr-Madan (1999) — FFT Option Pricing

Prices options via the dampened call price's Fourier transform, computed efficiently with the FFT algorithm.

### CIR (1985) — Short-Rate Model

```
dr_t = κ(θ − r_t)·dt + σ·sqrt(r_t)·dW_t
```

Ensures non-negative rates (Feller condition: `2κθ > σ²`).

---

## Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/MTW7503/Option-Pricing.git
cd option-pricing

# 2. Create the environment
conda env create -f environments.yml
conda activate option-pricing
# — OR —
pip install -r requirements.txt

# 3. Add the data file
# Download SM_options_data.xlsx from the assignment link
# and place it in the datasets/ folder

# 4. Launch Jupyter
jupyter notebook notebooks/
```

---

## Environment Setup

### Conda (recommended)

```bash
conda env create -f environments.yml
conda activate option-pricing
```

### pip

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Core dependencies

| Package | Purpose |
|---------|---------|
| `numpy` | Numerical arrays and linear algebra |
| `scipy` | Numerical integration (`quad`), FFT, and optimisers |
| `pandas` | Data loading and manipulation |
| `matplotlib` | All plots and figures |
| `openpyxl` | Reading the Excel option data |

---

## Data

- **Source:** SM Energy Company (SM) vanilla option chain — downloaded from the assignment link.
- **File:** `datasets/MScFE 622_Stochastic Modeling_GWP1_Option data.xlsx`
- **Content:** Strike, bid, ask, type (call/put), days to expiry.
- **Convention:** 1 year = **250 trading days** throughout all notebooks.

> **Important:** Do not commit the raw data file to a public repository if it contains proprietary market data. Add `datasets/*.xlsx` to `.gitignore` if necessary and share the file through a private channel with teammates.

---

## Collaboration Workflow

We follow a **branch-per-task** Git workflow:

```bash
# ── Starting a new task ───────────────────────────────────────────────────────
git checkout main
git pull origin main                         # always start from latest main

git checkout -b step1a-heston-lewis          # Sub-group 1 branch name
# ... work on notebook ...
git add notebooks/01_step1a_heston_lewis.ipynb
git commit -m "Step1a: Heston calibration via Lewis (2001) — initial draft"
git push origin step1a-heston-lewis

# ── Open a Pull Request on GitHub ────────────────────────────────────────────
# Title:   "[Step 1a] Heston–Lewis calibration"
# Body:    Brief description of what was done + calibrated parameter values
# Review:  At least one other sub-group member must approve before merge
```

### Branch naming conventions

| Branch | Responsible |
|--------|-------------|
| `step1a-heston-lewis` | Sub-group 1 |
| `step1b-heston-carr-madan` | Sub-group 2 |
| `step1c-asian-option` | Sub-group 3 |
| `step2d-bates-lewis` | Sub-group 3 |
| `step2e-bates-carr-madan` | Sub-group 1 |
| `step2f-put-pricing` | Sub-group 2 |
| `step3-cir` | Full team |
| `report/final` | Full team |

### Commit message format

```
[Step Xa] Short imperative description

- What changed / was added
- Any parameter values of note
- Any open questions for reviewers
```

### Pull Request checklist

Before requesting a review, confirm:

- [ ] Notebook runs top-to-bottom without errors (`Kernel → Restart & Run All`)
- [ ] All plots have axis labels, titles, and legends
- [ ] Calibrated parameter table is printed clearly
- [ ] Feller condition is reported (where applicable)
- [ ] No hardcoded absolute file paths (use relative paths from `datasets/`)
- [ ] Cells are numbered and the question number is stated in a comment or markdown cell

---

## Submission Checklist

| Item | Format | Notes |
|------|--------|-------|
| Final report | PDF (answers only, no code) | Use provided report template |
| Python code | Zipped folder: `.ipynb` + `.html` export | Notebook must be executable |
| Report submitted separately | PDF uploaded separately | Enables plagiarism detection |

### Generating the HTML export

```bash
jupyter nbconvert --to html notebooks/01_step1a_heston_lewis.ipynb
# Repeat for all notebooks, then zip the notebooks/ folder
```

---

## References

- Black, F. & Scholes, M. (1973). *The pricing of options and corporate liabilities.* Journal of Political Economy, 81(3), 637–654.
- Heston, S. L. (1993). *A closed-form solution for options with stochastic volatility with applications to bond and currency options.* Review of Financial Studies, 6(2), 327–343.
- Carr, P. & Madan, D. (1999). *Option valuation using the fast Fourier transform.* Journal of Computational Finance, 2(4), 61–73.
- Lewis, A. L. (2001). *A simple option formula for general jump-diffusion and other exponential Lévy processes.* SSRN Working Paper. https://ssrn.com/abstract=282110
- Bates, D. S. (1996). *Jumps and stochastic volatility: Exchange rate processes implicit in Deutsche Mark options.* Review of Financial Studies, 9(1), 69–107.
- Cox, J. C., Ingersoll, J. E. & Ross, S. A. (1985). *A theory of the term structure of interest rates.* Econometrica, 53(2), 385–408.

---

*HASTS 416/7 — Stochastic Processes | Group Work Project 1 | Honours Level (Part IV)*
