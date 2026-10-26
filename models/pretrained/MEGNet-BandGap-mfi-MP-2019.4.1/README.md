---
library_name: matgl
tags:
- matgl
- materials-science
- graph-neural-network
---

## Introduction

This model is a multif-fidelity MEGNet band gap model for 89 elements of the periodic table. It contains the band gap
for most of materials with 4 fidelities ("0": PBE, "1": GLLB-SC, "2": HSE, "3": SCAN). This is essentially a retrained
version of the MEGNet band gap model originally implemented in tensorflow.


## Usage

```python
import matgl

model = matgl.load_model("materialyze/MEGNet-BandGap-mfi-MP-2019.4.1")
```

## Model Details

- Number of parameters: 394,000

## Metrics

| Split | MAE (eV) |
|---|---:|
| Train | 0.114 |
| Validation | 0.317 |

## Metadata

```json
{
  "dataset": "MP-Bandgap-mfi-2019.4.1",
}
```

# References

```txt
Chen, C.; Zuo, Y.; Ye, W.; Li, X.; Ong, S. P. Learning Properties of Ordered and Disordered Materials from
Multi-Fidelity Data. Nature Computational Science 2021, 1, 46–53. https://doi.org/10.1038/s43588-020-00002-x.
```
