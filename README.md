# federated-neural-network-cluster

Federated deep learning for early intrusion detection on Linux and cloud server
clusters. A shared neural-network detector is trained across multiple nodes
without exchanging raw logs, and every flagged event is explained to a human
analyst.

Deep Learning course project (Master's, 6 credits).

Using the [Flemish Supercomputer Center (VSC)](https://docs.vscentrum.be/index.html).

---

## Overview

Each node in a server cluster (Linux hosts, cloud instances) produces a
continuous stream of logs: `auth.log`, `syslog`, `auditd`, web server logs, and
optionally eBPF kernel events. Real attacks are rare and buried in millions of
benign lines. Analysts suffer from alert fatigue, and organizations cannot pool
their logs to train a better detector because logs are sensitive — they contain
credentials, PII, and internal topology.

This project builds a **federated neural-network intrusion detector** for that
setting. A detection model is trained collaboratively across cluster nodes using
Federated Averaging: nodes share only model updates, never raw logs. The model
scores log sequences locally and surfaces **ranked, explained alerts** to a human
analyst, who decides whether to act.

The system assists the analyst — it does not replace them. Automated blocking is
deliberately out of scope (see *Limitations*).

---

## Problem

- Log volume per node is enormous; attacks are a tiny minority of events.
- Raw logs cannot be centralized for privacy and compliance reasons.
- Black-box alerts are hard to triage; analysts need to know *why* an event was
  flagged before they can act on it.

## Goal

1. **Detect** anomalous log sequences locally on each cluster node using a
   neural network.
2. **Collaborate** across nodes via federated training (FedAvg), without sharing
   raw logs.
3. **Explain** each high-priority alert (global model reliance + per-alert
   attribution) so the analyst can triage quickly.
4. **Recommend** a response (e.g., "block source IP", "inspect account") while
   leaving enforcement to the human.

---

## Approach

**Detection.** Neural sequence models trained to score log-event sequences as
normal or anomalous. Candidate architectures:

- **LSTM** (DeepLog-style) — the baseline workhorse for log-key sequences.
- **Transformer** (NeuralLog-style) — stronger, heavier; used for comparison.
- **Hybrid** (LogAnomaly-style) — sequence model + semantic log embeddings.

The final architecture is chosen empirically during the feasibility sprint.

**Federated training.** Multiple simulated cluster nodes, each holding a local
partition of the data (split by time or by source host), train locally and
exchange model updates with an aggregation server using FedAvg.

**Baseline.** Centralized training on the combined data, as the reference point
for every federated result.

**Explainability (xAI).**

- *Global*: SHAP summary showing which log features/tokens the model relies on.
- *Per-alert*: LIME attribution of the top contributing features/tokens for each
  flagged sequence.
- *Research question*: do SHAP explanations remain stable when the detector is
  trained federatedly rather than centrally, and does a shared background dataset
  improve consistency?

**Analyst output.** A ranked alert list where each entry carries: anomaly score,
LIME explanation, and a recommended action. Optional lightweight demo UI.

---

## Datasets

Candidates to be down-selected during the feasibility sprint.

| Dataset | Role | What to verify |
|---|---|---|
| **Security-Gym (v4)** | Primary candidate — labeled Linux server logs + eBPF events, SSH brute-force and mixed-attack streams | Access, license, log format, label quality, size |
| **AIT-ADS** | Ubuntu client/server testbed with Wazuh/Suricata alerts | Whether raw logs (not only alerts) are usable; alert schema |
| **Linux-APT-Dataset-2024** | APT-focused `auditd` logs, MITRE ATT&CK-mapped | Auditd parsing effort, attack diversity |
| **ADFA-LD** | Benchmark reference (Linux system-call traces) | Use as a comparison point only |
| **Loghub (HDFS / BGL)** | Fallback to guarantee a working baseline | Quick smoke test only |

**Selection criteria:** usable raw logs → labeled attacks → realistic Linux/cloud
origin → manageable download and parse time → clear license.

---

## Tech stack

| Layer | Choice |
|---|---|
| DL framework | PyTorch or TensorFlow — decision deferred to the feasibility sprint |
| FL framework | **Flower** (framework-agnostic, pip-installable, supports both) |
| Log parsing | Drain3 |
| Models | LSTM (DeepLog-style), Transformer (NeuralLog-style), optional hybrid |
| xAI | SHAP (global), LIME (per-alert) |
| Environment | Python virtual environment on VSC GPU nodes |
| Demo (optional) | Streamlit |

> Flower is chosen over TensorFlow Federated specifically because it is
> framework-agnostic, so the PyTorch-vs-TensorFlow decision can be made later
> without re-architecting the federated layer.

---

## Repository structure (planned)
```bash
├── data/ # download & preprocessing scripts
├── src/
│ ├── parsing/ # Drain3 log parsing -> structured sequences
│ ├── models/ # neural sequence models (LSTM / Transformer)
│ ├── federated/ # Flower client & server, FedAvg
│ ├── xai/ # SHAP & LIME wrappers
│ └── alerts/ # ranking & recommendation logic
├── notebooks/ # exploration & evaluation
├── scripts/ # VSC job scripts (GPU)
├── results/ # metrics & figures
└── README.md
```








## References

- DeepLog (Du et al.), NeuralLog, LogAnomaly — log anomaly detection baselines
- FedLAD — modular testbed for federated log anomaly detection
- Fed-SHAP, RE-FIDS, FLEX-IDS — federated xAI for intrusion detection
- Flower documentation; Drain3; SHAP; LIME












--- end