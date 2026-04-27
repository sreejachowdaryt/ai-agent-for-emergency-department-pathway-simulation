# AI Agent for Emergency Department Pathway Simulation

---

## Author 
**Sreeja Chowdary Tulluru**         
BSc Computer Science with Artificial Intelligence     
COMP3931 Individual Project | University of Leeds | 2025/26      

---

## Overview

This project investigates how Artificial Intelligence (AI) can be used to simulate and optimise patient flow within an Emergency Department (ED). It combines synthetic data generation, process mining, and discrete-event simulation (DES) to model realistic ED pathways and evaluate the impact of AI-driven decision-making.

The system models the full patient pathway from arrival through assessment, treatment, and discharge or admission/transferred, and introduces AI-based decision-support mechanisms to improve system efficiency, prioritisation and overall operational performnace.

---

## Objectives

- Develop a realistic simulation of ED patient flow using fully synthetic dataset
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
│           └── ed_cases.csv
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
- ED Pathway Modelled:

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

This demonstrates that while AI can optimise flow and decision-making, underlying capacity limitations remain the dominant drivers of system congestion.

---

## Data Sources

This project uses:

- **MIMIC-III** (for care unit transitions and patient pathways)
- **MIMIC-IV-ED** (for ED-specific timing and arrival patterns)

⚠️ Due to licensing restrictions, MIMIC data is **not included** in this repository.

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

## Usage

### Step 1: Extract distributions from MIMIC data (Already extracted)

```bash
python ER_PATIENTS_FLOW/src/extract_ed_timing_from_mimic_iv_ed.py
python ER_PATIENTS_FLOW/src/extract_activity_gaps_from_mimic.py
```

### Step 2: Generate synthetic dataset (Already generated with 50,000 patients as MIMIC datasets are not included in the repository this csv file cannot be created from scratch, so move to Step 3)

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

Hybrid - ML + Rule-Baased Simulation
```bash
python ED_SIMULATION/src/ed_simulation_ml.py
```

### Step 8: Compare Models

```bash
python ED_SIMULATION/src/compare_simulations.py
```

### Step 9: Generate figures

```bash
python ED_SIMULATION/src/generate_baseline_plots.py
python ED_SIMULATION/src/generate_comparison_plots.py
```

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

---

## Ethical Considerations 

- No real patient-identifiable data is used
- MIMIC datasets are referenced only for statistical distributions
- The system is designed as a decision-support tool, not a replacement for clinicians

---

## License

This project is for academic purposes only.

---