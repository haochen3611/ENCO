#!/bin/sh

cd ../  # Go back to experiment direction
python run_exported_graphs_ns.py --graph_files ../causal_graphs/synthetic_graphs/*.npz \
                                 --seed 42