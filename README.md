# AI Agent for Emergency Department Pathway Simulation

---

## Author 
Sreeja Chowdary Tulluru
Bsc Computer Science with Artificial Intelligence
COMP3931 Individual Project | University of Leeds | 2025/26 

---

## Overview

This project investigates how Artificial Intelligence (AI) can be used to simulate and optimise patient flow within an Emergency Department (ED). It combines synthetic data generation, process mining, and discrete-event simulation to model realistic ED pathways and evaluate the impact of AI-driven decision-making.

The system replicates the journey of patients from arrival through assessment, treatment, and discharge or admission, and introduces AI-based interventions to improve efficiency and prioritisation.

---

## Objectives

- Design a realistic simulation of ED patient flow using synthetic data
- Develop an AI agent for triage and prioritisation
- Integrate the AI agent into a discrete-event simulation
- Compare baseline and AI-driven system performance
- Evaluate improvements in waiting time, throughput, and resource utilisation

---

## Project Structure
AI-Agent-ED-Simulation/
│
├── ER_PATIENTS_FLOW/ # Data generation and process mining
│ ├── src/
│ │ ├── generate_ed_cases.py
│ │ ├── extract_activity_gaps_from_mimic.py
│ │ ├── extract_ed_timing_from_mimic_iv_ed.py
│ │ ├── create_event_log.py
│ │ ├── process_discovery.py
│ │ └── ...
│ └── Synthetic_dataset/
│ └── data/
│
├── ED_SIMULATION/ # Simulation and AI agents
│ ├── src/
│ │ ├── ed_simulation.py
│ │ ├── ed_simulation_ai.py
│ │ ├── ed_simulation_ml.py
│ │ ├── ai_agent.py
│ │ └── ...
│ └── data/
│
├── requirements.txt
├── requirements-lock.txt
└── README.md

---

## Methodology

The project follows a structured pipeline:

### 1. Synthetic Dataset Generation
- Based on statistical distributions derived from MIMIC-III and MIMIC-IV-ED
- Generates patient records and ED case pathways
- Preserves temporal consistency and realistic transitions

### 2. Event Log Construction
- Converts synthetic ED cases into event logs
- Format: `case_id`, `activity`, `timestamp`
- Enables process mining analysis

### 3. Process Mining
- Uses PM4Py to discover patient flow models
- Generates process trees and Directly-Follows Graphs (DFGs)
- Validates realism of the synthetic dataset

### 4. Discrete-Event Simulation (DES)
- Built using SimPy
- Models key ED stages:
  - Arrival
  - Triage
  - Doctor consultation
  - Boarding
  - Discharge or admission

### 5. AI Agent Integration
Two types of AI agents are implemented:

- **Rule-Based Agent**
  - Prioritises patients based on severity
  - Reduces waiting time for critical cases

- **Machine Learning Agent**
  - Predicts Point-of-Care Testing (POCT) needs
  - Enables fast-track pathways for suitable patients
  - Reduces doctor consultation time

### 6. Performance Evaluation
Simulation outputs are compared using:

- Waiting time (triage, doctor, boarding)
- Length of stay (LOS)
- Throughput
- NHS 4-hour target compliance

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

### Step 1: Extract distributions from MIMIC data

```bash
python ER_PATIENTS_FLOW/src/extract_ed_timing_from_mimic_iv_ed.py
python ER_PATIENTS_FLOW/src/extract_activity_gaps_from_mimic.py
```

### Step 2: Generate synthetic dataset

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

### Step 5: Run simulations

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

### Step 6: Compare results

```bash
python ED_SIMULATION/src/compare_simulations.py
```

---

## Key Features 

- Hybrid synthetic dataset generation (MIMIC-III + MIMIC-IV-ED)
- Realistic ED pathway modelling
- Integration of AI agents into simulation
- Process mining validation
- Performance comparison across multiple simulation strategies

---

## Results 

- AI agents significantly improve prioritisation of critical patients
- Reduction in waiting times for high-severity cases
- Improved NHS 4-hour target compliance
- Better resource utilisation under constrained conditions

---

## Ethical Considerations 

- No real patient-identifiable data is used
- MIMIC datasets are referenced only for statistical distributions
- The system is designed as a decision-support tool, not a replacement for clinicians

---

## License

This project is for academic purposes only.

---