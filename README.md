# AI Agent for Emergency Department Pathway Simulation

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Status](https://img.shields.io/badge/Status-Completed-success)
![License](https://img.shields.io/badge/License-Academic-lightgrey)
![Framework](https://img.shields.io/badge/Simulation-SimPy-orange)
![ML](https://img.shields.io/badge/ML-RandomForest-green)

---

## Author 
**Sreeja Chowdary Tulluru**         
BSc Computer Science with Artificial Intelligence     
COMP3931 Individual Project | University of Leeds | 2025/26      

---

## Overview

This project investigates the use of Artificial Intelligence (AI) to simulate and optimise patient flow within an Emergency Department (ED). It integrates synthetic data generation, process mining, and discrete-event simulation (DES) to model realistic patient pathways from arrival through assessment to final disposition (discharge, admission, or transfer).

AI-based decision-support mechanisms are embedded within the discrete-event simulation (DES) model to improve clinical prioritisation and operational efficiency. The framework enables systematic evaluation of how AI-driven interventions influence key performance metrics such as waiting time, length of stay, and NHS 4-hour target compliance in resource-constrained healthcare environments.

---

## Objectives

- Develop a realistic simulation of ED patient flow using a fully synthetic dataset
- Design an AI agent for prioritisation and operational decision-making
- Integrate the AI agent into a discrete-event simulation (DES) framework
- Compare baseline and AI-driven system performance
- Evaluate improvements in waiting time, length of stay, and NHS compliance

---

## Project Structure

```bash
AI-AGENT-FOR-EMERGENCY-DEPARTMENT-PATHWAY-SIMULATION/
│
├── ER_PATIENTS_FLOW/              # Synthetic dataset generation
│   ├── src/
│   │   ├── generate_ed_cases.py
│   │   ├── extract_activity_gaps_from_mimic.py
│   │   ├── extract_ed_timing_from_mimic_iv_ed.py
│   │   └── ...
│   └── Synthetic_dataset/
│       └── data/
│           └── ed_cases.csv       # Generated ED cases
│
├── ED_SIMULATION/                 # Process mining, simulation, AI agents
│   ├── src/
│   │   ├── create_event_log.py
│   │   ├── process_discovery.py
│   │   ├── extract_ed_simulation_parameters.py
│   │   ├── ed_simulation.py              # Baseline DES
│   │   ├── ed_simulation_ai.py           # Rule-based AI
│   │   ├── ed_simulation_ml.py           # Hybrid (ML + rule-based)
│   │   ├── ai_agent.py
│   │   └── ...
│   │
│   ├── data/                     # Generated datasets & simulation outputs
│   └── figures/                  # Three-way comparison figures
│
├── Reference_mimic_iii/          # Reference schemas (not used directly)
├── requirements.txt
└── README.md
```

---

## Methodology

The project follows a structured pipeline:

### 1. Synthetic Dataset Generation
A hybrid dataset is generated using statistical distributions derived from:
- MIMIC-III (care unit transitions and inpatient pathways)
- MIMIC-IV-ED (ED timings and arrival patterns)

The dataset:
- preserves temporal consistency
- models repeated patient admissions (multi-admission patient behaviour)
- captures realistic ED flow dynamics

Output: 
```bash
ed_cases.csv
```
### 2. Event Log Construction
- Converts synthetic ED cases into event logs
- Format: `case_id`, `activity`, `timestamp`
- Enables process mining analysis

Output: 
```bash
event_log.csv
```
### 3. Process Mining
- Uses PM4Py to discover patient flow models
- Generates process trees and Directly-Follows Graphs (DFGs)
- Validates realism of the synthetic dataset
- Analyses transitions between ED stages

### 4. Discrete-Event Simulation (DES)
- Built using SimPy
- Simulated ED Pathway from process mining model:

```bash
Arrival → Assessment → Outcome Decision → Boarding (if required - only for non-discharge patients) → Departure
```
Key Design Features:
- Dataset-driven arrival schedule
- Combined assessment stage (triage + doctor proxy)
- Resource-constrained environment:
    - assessment bays (5)
    - boarding slots (7)

### 5. AI Agent Integration
Three simulation configurations are implemented:

1. **Baseline Model**
  - FIFO boarding queue
  - No prioritisation

2. **Rule-Based Agent**
  - Prioritises patients based on severity
  - Reduces waiting time for critical cases

| Severity | Priority | Boarding Time |
| -------- | -------- | ------------- |
| Critical | 1 (High) | 90 min        |
| High     | 2        | 110 min       |
| Medium   | 3        | 147 min       |
| Low      | 4 (Low)  | 147 min       |

3. **Hybrid (ML + Rule-based Agent)**
Adds ML-based POCT prediction at assessment stage:
- Random Forest classifier predicts POCT amenability
- Reduces assessment time for eligible patients
- Enables fast-track discharge for low/medium severity cases
- Combined with rule-based boarding prioritisation

### 6. Performance Evaluation

Simulation outputs are evaluated using the following key performance metrics:

- **Waiting Time**  
  Time spent waiting for initial assessment 

- **Boarding Time**  
  Time spent waiting for an inpatient ward/ICU after treatment decision (admitted/transferred patients) 

- **Total ED Length of Stay (LOS)**  
  Total time from patient arrival to ED departure  

- **NHS 4-Hour Target Compliance**  
  Percentage of patients discharged/admitted/transferred within 4 hours  

---

### Bottleneck Analysis

The baseline simulation identified two primary bottlenecks:

- **Assessment Stage Bottleneck**  
  Caused by limited assessment bays, resulting in queues forming. This leads to increased waiting times during peak arrival periods. 
  Exceeds NHS 15-minute benchmark 

- **Boarding Stage Bottleneck**  
  Caused by limited boarding slot capacity, preventing admitted patients from leaving the ED promptly. This results in prolonged boarding times and contributes significantly to overall ED congestion (exit block).  

The introduction of AI agents did not eliminate these bottlenecks completely, as they are structural constraints within the system. However, the AI interventions:
 
- Improved prioritisation for high-severity patients  
- Reduce initial assessment duration (ML agent)     
- Improved patient flow under constrained resources 

This demonstrates that while AI enhances decision-making and improves operational efficiency, system performance remains fundamentally constrained by structural capacity limitations. Therefore, optimal ED performance requires both intelligent decision-support and appropriate resource allocation.

---

## Data Sources and Licensing

This project uses:

- **MIMIC-III** (for care unit transitions and patient pathways)
- **MIMIC-IV-ED** (for ED-specific timing and arrival patterns)

Access to these datasets was obtained through the official PhysioNet credentialing process. The project complies with PhysioNet data usage policies and ethical guidelines for handling clinical data.

⚠️ Due to licensing and data protection restrictions, no raw MIMIC data is included in this repository. All datasets used in this project are fully synthetic and derived from statistical distributions.

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/sreejachowdaryt/ai-agent-for-emergency-department-pathway-simulation.git
cd ai-agent-for-emergency-department-pathway-simulation
```

### 2. Create virtual environment

```bash
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Install Graphviz (Required for process mining)

Download from:
https://graphviz.org/download/

Ensure it is added to system PATH.

---

## Usage Pipleine (Implementation Pipeline)

### Step 1: Extract distributions from MIMIC data 
The MIMIC-derived distributions have already been extracted and stored as CSV files in "ER_PATIENTS_FLOW/Synthetic_dataset/data/"

These scripts were used during development:
```bash
python ER_PATIENTS_FLOW/src/extract_ed_timing_from_mimic_iv_ed.py
python ER_PATIENTS_FLOW/src/extract_activity_gaps_from_mimic.py
```

Note: The original MIMIC databases are not included in this repository due to ethical considerations. Therefore, these extraction scripts cannot be rerun unless the user has authorised access to MIMIC-III and MIMIC-IV-ED. 

### Step 2: Generate synthetic dataset 
The synthetic dataset has already been generated from the extracted MIMIC distributions: 
- ed_cases.csv - 56,511 ED admission episodes 
- patients.csv - 50,000 synthetic patients
These files are located in "ER_PATIENTS_FLOW/Synthetic_dataset/data/"

The synthetic dataset was generated using:
```bash
python ER_PATIENTS_FLOW/src/generate_ed_cases.py
```

### Step 3: Create Event Log 

```bash
python ED_SIMULATION/src/create_event_log.py
```

### Step 4: Run process mining

```bash
python ED_SIMULATION/src/process_discovery.py
```

### Step 5: Extract simulation parameters

```bash
python ED_SIMULATION/src/extract_ed_simulation_parameters.py
```

### Step 6: Train ML model (POCT)

```bash
python ED_SIMULATION/src/train_poct_model.py
```

### Step 7: Run simulations

Baseline Simulation
```bash
python ED_SIMULATION/src/ed_simulation.py
```

Rule-Based AI Simulation
```bash
python ED_SIMULATION/src/ed_simulation_ai.py
```

Hybrid - ML + Rule-Based Simulation
```bash
python ED_SIMULATION/src/ed_simulation_ml.py
```

### Step 8: Compare Models
A complete Three-way simulation comparison between all the three models: baseline, rule-based and hybrid (ML+rule-based)

```bash
python ED_SIMULATION/src/compare_simulations.py
```

### Step 9: Generate baseline and comparison figures

```bash
python ED_SIMULATION/src/generate_baseline_plots.py
python ED_SIMULATION/src/generate_comparison_plots.py
```
---

## Reproducibility

All results presented in the dissertation can be reproduced by running the usage pipeline described above. Due to MIMIC data access restrictions, pre-generated distributions and datasets are included to ensure reproducibility without requiring direct access to restricted clinical databases.

---

## Key Features 

- Hybrid synthetic dataset generation (MIMIC-III + MIMIC-IV-ED)
- Dataset-driven simulation (realistic arrival patterns)
- Process Mining validation 
- Integration of AI agents into simulation (Rule-based and Hybrid)
- Performance comparison across multiple simulation strategies
- NHS compliance analysis

---

## Results 

- AI agents significantly improve prioritisation of critical patients
- Reduced assessment waiting time (ML agent)
- Reduction in waiting times for high-severity cases
- Improved NHS 4-hour target compliance
- Hybrid model achieves best overall system performance

These results highlight the effectiveness of AI-driven interventions in improving both system efficiency and clinical prioritisation under constrained resource conditions.

---

## Ethical Considerations 

- No real patient-identifiable data is used
- MIMIC datasets are referenced only for statistical distributions
- The system is designed as a decision-support tool, not a replacement for clinicians

---

## Project Status

This project has been completed as part of the COMP3931 Individual Project at the University of Leeds (2025/26). 

The implementation, evaluation, and documentation are finalised. Future improvements are mentioned in the Final Report submitted to the University.

---