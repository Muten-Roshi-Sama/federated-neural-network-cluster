
# Federated — Future extension

Federated version of the AIT autoencoder, using Flower.

**Status: not started.** Only build this if there's time after `../ait/` is
complete. Will reuse:
- `../common/models.py` — same autoencoder architecture
- `../common/evaluation.py` — same metrics
- `../datasets/processed/ait/windows_testbed_*.npz` — same pre-processed data

The 8 AIT testbeds are the natural silos — one client per testbed.


structure


```bash
federated/
├── src/
│   ├── client.py                    # Flower NumPyClient (reuses common/models)
│   ├── server.py                    # FedAvg strategy
│   └── simulation.py                # start_simulation orchestration
├── batch-jobs/
│   └── 01_simulate.slurm
└── notebooks/
    └── 04-federated-results.ipynb
```

---




### Source files
(AIT Log Data Set V2.1, Published April 9, 2026)[https://zenodo.org/records/19483937]

Contains 8 testbeds (our 8 silos) for a total of 137.1 GB.

AIT-LDS V2.1 is a collection of 8 synthetic enterprise-network simulations ("testbeds"), each representing a small company network: mail server, file share, WordPress, VPN, firewall, intranet server, etc. Each testbed runs 4–6 days of simulated normal user behavior and then launches a chained multi-step attack. All logs from all hosts are collected, and a separate ground-truth layer labels exactly which log lines belong to which attack step


Download the files :
```bash
python download_ait_no_pcaps.py --silos fox
python download_ait_no_pcaps.py --silos harrison wheeler
python download_ait_no_pcaps.py --silos russellmitchell santos shaw wardbeck wilson

```




Directory Structure & How Labels Work
```bash
datasets/ait/fox/
├── dataset.yaml     # dataset global start and end time of simulation. 
├── environment      # source code testbed setup code — IGNORE
├── processing       # AIT authors' code that generated the labels — IGNORE
├── rules            # the labeling rules
├── ...
├── labels          # ground-truth labels, mirros /gather exactly 
├── gather          # raw logs
│   ├── attacker_0
│   ├── cloud_share
│   ├── ext_user_0
│   ├── ...
│   ├── inet-dns
│   ├── inet-firewall        # the DNS server
│   ├── internal_employee_0
│   ├── ...
│   ├── internal_share
│   ├── intranet_server     # the WordPress box that gets compromised
│   ├── monitoring
│   ├── remote_employee_0
│   ├── ...
│   ├── vpn
│   ├── webserver
│   ├── mail/, jonesmorgan_mail/, miller_mail/, taylorcruz_mail/, whittaker_mail/      #


```


Labels shows which lines of the audit are part of an attack : 
```json
{"line": 1860, "labels": ["attacker_change_user", "escalate"], "rules": {...}}
```

Our models need to train on normal/bening data only, so we will need to train them on lines that are **NOT** in the label files only.
```bash
train-set = everything-not-labeled  # (safe/benign)
eval-set = evrything-labeled        # (attack)
```






### Dev notes


- The dataset is made for federated learning, so combining the 8 silos into one to train a model won't be comparable to the federated one, combining would leak information in a way real federated IDS deployement would.
- Local-only : One silo, Train + test on it (AE, VAE and LSTM) on one silo, 
- Centralized : All 8 silos, .





--- end
