#!/usr/bin/env python3
"""
visualization_tools.py
------------------------
Production-ready script for generating visualizations.
Features:
- Error handling and logging.
- Functions to plot network graphs and ROC curves.
- Detailed inline documentation.
Usage:
    python src/visualization_tools.py --network results/wgcna/network_edgelist.csv --roc results/ml/roc_curve.csv --network_img results/figures/network.png --roc_img results/figures/roc_curve.png
"""

import os
import sys
import argparse
import logging
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

def parse_args():
    parser = argparse.ArgumentParser(description="Generate visualizations for network and ROC curve.")
    parser.add_argument("--network", required=True, help="Path to network edge list CSV file.")
    parser.add_argument("--roc", required=True, help="Path to ROC curve CSV file.")
    parser.add_argument("--network_img", required=True, help="Path to save the network plot image.")
    parser.add_argument("--roc_img", required=True, help="Path to save the ROC curve image.")
    return parser.parse_args()

def plot_network(edge_list_path, output_image):
    try:
        G = nx.read_edgelist(edge_list_path, delimiter=',')
        plt.figure(figsize=(10, 10))
        nx.draw_networkx(G, with_labels=True, node_size=500, node_color="skyblue", font_size=8)
        os.makedirs(os.path.dirname(output_image), exist_ok=True)
        plt.savefig(output_image)
        plt.close()
        logging.info(f"Network visualization saved to: {output_image}")
    except Exception as e:
        logging.error(f"Error in network visualization: {e}")
        sys.exit(1)

def plot_roc(performance_csv, output_image):
    try:
        data = pd.read_csv(performance_csv)
        plt.figure(figsize=(8, 6))
        plt.plot(data['fpr'], data['tpr'], marker='.', label='ROC Curve')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend()
        os.makedirs(os.path.dirname(output_image), exist_ok=True)
        plt.savefig(output_image)
        plt.close()
        logging.info(f"ROC curve saved to: {output_image}")
    except Exception as e:
        logging.error(f"Error in ROC plotting: {e}")
        sys.exit(1)

if __name__ == "__main__":
    args = parse_args()
    plot_network(args.network, args.network_img)
    plot_roc(args.roc, args.roc_img)
