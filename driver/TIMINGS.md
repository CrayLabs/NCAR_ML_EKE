# Benchmark for MOM6 OM4_025
The following timings were collected on horizon based on integrating OM4_025 for one day. In 'standard' mode, MOM6 was one on CascadeLake nodes with the database nodes having 18 Broadwell cores with P100 GPU. In colocated, MOM6 and the orchestrator are run on the same (Broadwell) nodes.

Times are the minimum, maximum, and average time over all PEs.

'Standard': 3 database nodes/2432 ranks
```
(SMARTREDIS run model)                0.514838     11.497466      7.12233
```

'Standard': 12 database nodes/2432 ranks
```
(SMARTREDIS run model)                0.258169      3.730217      2.447877

```

'Standard': 16 database nodes/2432 ranks
```
(SMARTREDIS run model)                0.179329      3.110898      1.945738
```


