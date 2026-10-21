# federated-neural-network-cluster

Federated deep learning for early intrusion detection on Linux and cloud server
clusters.

Goal :
- A shared neural-network detector is trained across multiple nodes
- without exchanging raw logs, and every flagged event is explained to a human analyst.

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





## Notes

- use multiple datasets, study (https://doaj.org/article/20dcb8ffe1d9422390c12b1fb85205b6#1) found that combining multiple datasets reduced overfitting
- merging datasets might not be easy. but FL can be used advantegeously here, Treat multiple datasets as mutliple clients of the network. Letting the federated aggregate the results.


Data augmentation :
- enrich each client's local dataset, particularly for rare attack types, without sharing raw data.
- Federated WGANs (e.g., FedWID) (https://dl.acm.org/doi/10.1007/s42979-025-04531-y#1)
- LLM-Guided Augmentation (e.g., GAIA-FL)
- SMOTE / ADASYN
- Random Oversampling



Advantages of FL :
- privacy regulations and operational limitations that prevent organizations from pooling raw network logs.
- transmit only a few MBs of model weights instead of +50GBs of raw logs




Datasets comparison :


AIT Log Data Set V2.1 :
- https://zenodo.org/records/19483937#1
- Logs : Apache access and error logs, authentication logs, DNS logs, VPN logs, audit logs, Suricata logs, network traffic packet captures, horde logs, exim logs, syslog, and system monitoring logs.
- Attacks : Scans (nmap, WPScan, dirb), Webshell upload (CVE-2020-24186), Password cracking (John the Ripper), Privilege escalation, Remote command execution, Data exfiltration (DNSteal)
- Has enough benign traffic, already partitionned into 8 silos
- License : CC BY 4.0 (academic + commercial use.)


Entente Cross-silo :
- https://zenodo.org/records/16014433#1
- for GNNs





UNSW-NB15 : 
- j



TON_IoT :
- j



Mendeley Federated IoT/PC Traffic :
- Unsupervised learning : attack labels exist on testing set only.
- explicitly designed for FL experiments, fast prototyping only. 
- small, ~70k rows and tabular make it smoke test only



CAM-LDS :
- https://zenodo.org/records/18861762#1
- NO BENINGN LOGS. (hard to train classification without it).
- simulate real small linux entreprise (dmz, lan, repo server, dns, firewall, user workstation ...)
- logs (audit, authentication, appache access and erro, syslog, package management, cron, Zoneminder, FTP, Pupper, Docker, Nextcloud, mail ...)
- seven scenarios/attack chains : video server exploit, linuc mlware, lateral movement, network attack;network sniffing, attack on client, **Docker attack**.
- License ? tbd but must cite publication listed.


Dedale :
- https://entrepot.recherche.data.gouv.fr/dataset.xhtml?persistentId=doi:10.57745/Y5JLDG&version=2.0
- Very large (1TB) and very low proportion of attacks
- APT detection among logs
- 4 weeks of logs, apt attack lasting 8 days
- stealthy attack scenario, no brute forece or noisy attack types.
- License : Creative Commons Attribution 4.0 International License


---
Choice : 
- AIT as primary
- Mendeley as smoke-test

Model :
- LSTM : classic for log detection. See DeepLog, LogAnomaly. Simple to train.
- Transformers : LogBERT
- hybrid model that uses a transformer-based encoder for log embedding and a bidirectional LSTM with attention for sequence modeling (https://ieeexplore.ieee.org/document/11650311#1).



Multi-dataset Federated Learning :
- each dataset as separate client.
- study shows better performance (but pecif for IoT) : https://ora.ox.ac.uk/objects/uuid:c942fb20-d675-4deb-98a8-0e333e36681d
- must harmonize labels (e.g., map different attack names to a common family-level schema) and align features to their intersection set before training.

Federated Transfer Learning :
-  using a model pre-trained on one dataset (e.g., AIT) as a starting point and fine-tuning it on another.
- A two-stage LSTM pipeline is one such framework designed for this purpose.
- SOURCE ? Research on Federated Transfer Learning (FTL) for intrusion detection has shown that this approach can effectively improve detection accuracy and training efficiency. 






--- end