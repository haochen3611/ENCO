#!/bin/sh

cd ../  # Go back to experiment direction
python run_exported_graphs_ns.py --graph_files ../causal_graphs/synthetic_graphs/graph_full_25_44.npz \
                                 --seed 43