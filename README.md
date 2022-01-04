# Knowledge Graph Attention Network

This repository contains the implementation for the paper:

> Ryotaro Shimizu, Megumi Matsutani, Masayuki Goto (2021). An explainable recommendation framework based on an improved knowledge graph attention network with massive volumes of side information.
> Paper in Knowledge-Based Systems.
> https://www.sciencedirect.com/science/article/pii/S0950705121010959

## Citation
---
If you want to use our codes and datasets in your research, please cite:
```
@article{SHIMIZU2021107970,
  title = {An explainable recommendation framework based on an improved knowledge graph attention network with massive volumes of side information},
  journal = {Knowledge-Based Systems},
  pages = {107970},
  year = {2021},
  issn = {0950-7051},
  doi = {https://doi.org/10.1016/j.knosys.2021.107970},
  url = {https://www.sciencedirect.com/science/article/pii/S0950705121010959},
  author = {Ryotaro Shimizu and Megumi Matsutani and Masayuki Goto},
}
```

## How to run the code
---
run the kgat+(plsa-6) model
```bash
make setup
make run
```

run with another condition
```bash
poetry run python main.py --model_type {WIP}
```

### Argments
- model_type
WIP

## References
---
1. https://github.com/xiangwang1223/knowledge_graph_attention_network
2. https://github.com/LunaBlack/KGAT-pytorch
