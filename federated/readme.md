
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





--- end
