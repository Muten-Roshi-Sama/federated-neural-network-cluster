# Federated Early Intrusion Detection System for Linux & Cloud Servers

## 1. Problem Statement
Linux servers and cloud infrastructures generate enormous volumes of log data (auth logs, syslog, auditd, web server logs, eBPF events). Security analysts are overwhelmed: real attacks are buried in millions of benign lines, and alert fatigue causes genuine threats to be missed. At the same time, organizations cannot pool their raw logs to train better detection models — logs are sensitive and often contain PII, credentials, and internal topology.

Federated Learning (FL) solves this: multiple servers/organizations train a shared detection model without sharing raw logs — only model updates are exchanged.

## 2. Project Goal
Build a Federated Early Intrusion Detection System that:
1. Monitors Linux/cloud server logs locally on each node.
2. Parses and scores log sequences for anomalous/suspicious behavior using a federated deep learning model.
3. Flags and ranks suspicious log events with a confidence score and explanation, pushing only high-priority alerts to a human analyst dashboard.
4. Assists — not replaces — the analyst: the system recommends a response (e.g., "block IP X", "inspect account Y") but requires human approval before enforcement.

> Scope note: Full autonomous blocking is explicitly out of scope for this 6-credit project. The system delivers prioritized, human-in-the-loop alerting, with the false-positive rate at the alerting threshold as a key evaluated metric. This keeps the project honest and matches real SOC practice.

## 3. Objectives
- O1: Build a log ingestion + parsing pipeline (raw log → structured event sequences).
- O2: Train an anomaly-detection model centrally (baseline) and in a federated setting (multi-client FedAvg simulation).
- O3: Evaluate detection performance (precision/recall/F1, false-positive rate at alert threshold) and compare federated vs. centralized.
- O4: Design a lightweight alerting/recommendation layer that surfaces ranked suspicious logs to an analyst.
- O5: Demonstrate feasibility on the Flemish Supercomputer (VSC) using GPU-accelerated federated simulation.
O6: Implement an explanation layer (SHAP global + LIME per-alert) that attributes each flagged log to its most influential features/tokens.

## 4. Candidate Datasets (to be selected)
|Dataset	|Why consider it	|What to verify in testing|
|-----------|-------------------|-------------------------|
|Security-Gym (v4)	|Labeled Linux server logs + eBPF kernel events; includes SSH brute-force and 30-day mixed-attack streams; designed for agent-controlled defenses.	|Download access, license, log format, label quality, size (~8.5 GB)|
|AIT-ADS	|Realistic Ubuntu client/server testbed; alerts from Wazuh & Suricata; small-company attack scenarios.	|Whether raw logs (not just alerts) are usable; alert schema|
|Linux-APT-Dataset-2024|	APT-focused, MITRE ATT&CK-mapped, auditd-based Linux logs.	|Auditd parsing effort, attack diversity, size|
|ADFA-LD	|Classic benchmark (system-call traces, Linux).	|Use as a benchmark reference point, not primary (older)|
|HDFS / BGL (via Loghub) (fallback)	|Widely used log-anomaly benchmarks; ensures a working baseline if the above fail.	|Quick smoke-test only|

Decision criteria: usable raw logs (not just pre-computed alerts) → labeled attacks → realistic Linux/cloud origin → manageable download/parse time → license clarity.


## 5. Candidate Toolkit (to be validated during the evaluation sprint)
|Layer	|Candidate	|Role|
|-------|------------|----|
|FL framework	|Flower (primary), TensorFlow Federated (alternative)	|Framework-agnostic, easy pip install, realistic client simulation|
|DL framework	|PyTorch or TensorFlow	|Model training|
|Log parsing	|Drain3 (primary), LogPai toolset	|Raw log → event templates|
|Models	        |DeepLog (LSTM), NeuralLog (Transformer), LogAnomaly — reference implementations available via the FedLAD testbed	|Sequence anomaly detection|
|Env	|Python venv on VSC (Hortense GPU nodes)	|Isolated, reproducible environment|
|Viz (optional)	|Streamlit / Plotly	|Analyst-facing alert dashboard demo|


## 6. Deliverables
- Reproducible codebase (data pipeline, federated training, alert ranking)
- Centralized vs. federated benchmark results on the chosen dataset
- False-positive-rate analysis at the analyst-alert threshold
- Short demo of the analyst-facing ranked alert output



## 7. xAI angle

Use Shap and lime to explain local and global scope.

-> The global model is an aggregate of many local models trained on non-IID data. A feature might be important on one server and irrelevant on another. 

measure whether explanations remain stable and useful when the model is federated rather than centralized, and if they degrade, implement a simple mitigation like a shared/global SHAP background set.

Can use already existing tools like : 
- Fed-SHAP uses federated fuzzy c-means clustering to construct a global background dataset for SHAP, so explanations remain consistent across clients with non-IID data.
- RE-FIDS integrates a SHAP-based interpretability layer into a robust federated IDS, explicitly framing xAI as a bridge between anomaly detection performance and "operational trust in decentralized security operations centers".
- Other frameworks like FLEX-IDS combine FL with post-hoc interpretability (SHAP and LIME) under adversarial conditions.






--- end