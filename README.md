# knowledge-graph-makey
Codebase for exploring and analysis of knowledge graph creation by an LLM vs GNN

## For Running RGCN-TransE
python3 run.py --score_func transe --opn mult --gpu 0 --gamma 9 --hid_drop 0.2 --gcn_dim 100 --init_dim 100 --epoch 500 --batch 1 --num_base 5 --n_layer 1 --encoder rgcn --name repro

## For Running CompGCN-DistMult
python run.py --score_func distmult --opn mult --gpu 0 --epoch 500 --batch 256 --n_layer 2 --gcn_dim 100 --name repro
