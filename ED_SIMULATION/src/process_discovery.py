# src/create_event_log.py
"""
Perform process discovery on the synthetic ED event log using PM4Py.

This script:
- Loads the event log generated from the synthetic ED dataset
- Converts it into PM4Py format (case_id, activity, timestamp)
- Applies the Inductive Miner algorithm to discover the process model
- Generates a process tree representation of the ED workflow
- Constructs a Directly-Follows Graph (DFG) for behavioural analysis
- Saves visualisations of both models for interpretation and reporting

These outputs are used to validate that the simulated ED pathways
produce realistic and interpretable process structures.
"""

import pandas as pd

from pm4py.objects.conversion.log import converter as log_converter
from pm4py.algo.discovery.inductive import algorithm as inductive_miner

from pm4py.visualization.process_tree import visualizer as pt_visualizer
from pm4py.algo.discovery.dfg import algorithm as dfg_discovery
from pm4py.visualization.dfg import visualizer as dfg_visualizer


def discover_process(event_log_path):

    print("Loading event log...")

    df = pd.read_csv(event_log_path)

    # Rename columns to PM4Py standard
    df = df.rename(columns={
        "case_id": "case:concept:name",
        "activity": "concept:name",
        "timestamp": "time:timestamp"
    })

    # Convert timestamp to datetime
    df["time:timestamp"] = pd.to_datetime(df["time:timestamp"])

    # Convert dataframe → PM4Py event log
    event_log = log_converter.apply(df)

    print("Running Inductive Miner...")

    # Discover process tree
    process_tree = inductive_miner.apply(event_log)

    print("Process model discovered.")

    # -----------------------------
    # PROCESS TREE VISUALISATION
    # -----------------------------

    gviz_tree = pt_visualizer.apply(process_tree)

    # Save figure for dissertation
    pt_visualizer.save(gviz_tree, "../figures/process_tree.png")

    # Also open interactive view
    pt_visualizer.view(gviz_tree)

    print("Process tree saved to figures/process_tree.png")

    # -----------------------------
    # DIRECTLY-FOLLOWS GRAPH
    # -----------------------------

    print("Generating Directly-Follows Graph...")

    dfg = dfg_discovery.apply(event_log)

    gviz_dfg = dfg_visualizer.apply(dfg, log=event_log)

    dfg_visualizer.save(gviz_dfg, "../figures/dfg.png")

    dfg_visualizer.view(gviz_dfg)

    print("DFG saved to figures/dfg.png")

    return process_tree


if __name__ == "__main__":

    discover_process("../data/event_log.csv")