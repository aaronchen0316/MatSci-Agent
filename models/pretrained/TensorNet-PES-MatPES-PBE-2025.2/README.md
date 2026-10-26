---
library_name: matgl
tags:
- matgl
- materials-science
- graph-neural-network
- machine-learning-interatomic-potential
- foundation-potential
- mlip
---

## Introduction

Pre-trained TensorNet foundation potential, i.e., universal machine learning interatomic potential trained on the MatPES PBE 2025.2 dataset.

## Potential

[matgl](https://github.com/materialsvirtuallab/matgl) `Potential` model (version 3).

## Usage

```python
import matgl

model = matgl.load_model("materialyze/TensorNet-PES-MatPES-PBE-2025.2")
```

## Stats
- Layers: 2
- Units: 128
- Test_MAE_energies: 32 meV/atom
- Test_MAE_forces: 119 meV/Å
- Test_MAE_stresses: 0.626 GPa

## Metadata

```json
{
  "dataset": "MatPES-PBE-2025.2"
}
```
