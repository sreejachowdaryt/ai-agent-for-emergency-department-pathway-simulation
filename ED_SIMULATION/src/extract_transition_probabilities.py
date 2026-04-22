# src/extract_transition_probabilities.py
"""
Extract transition probabilities from the ED event log for process analysis.

This script computes the probability of transitions between consecutive
activities in the event log, representing the observed patient flow
through the Emergency Department pathway.

Method:
- Events are ordered within each case using timestamps
- Consecutive activity pairs are extracted (e.g., A → B)
- Transition frequencies are counted across all cases
- Probabilities are computed as:
      P(next = B | current = A)

Output:
- A transition probability table with:
      from, to, count, total, probability

Use in project:
- Supports process mining analysis (e.g., Directly-Follows Graph)
- Provides empirical insight into pathway behaviour
- Can be used for validation of the synthetic dataset against expected flow patterns

Note:
- This script is analytical only and is not directly used by the simulation model
"""

import pandas as pd


def extract_transition_probabilities(event_log_path, output_path):
    """
    Extract transition probabilities between activities from the event log.

    The event log contains rows like:
        case_id | activity | timestamp

    Example:
        1000 | Arrival
        1000 | Initial Assessment
        1000 | Ward1 Transfer In

    We compute how often each activity is followed by another activity.
    """

    print("Loading event log...")

    # Load event log
    df = pd.read_csv(event_log_path)

    # Ensure events are ordered correctly within each case
    df = df.sort_values(["case_id", "timestamp"])

    transitions = []

    # ------------------------------------------
    # STEP 1: Extract activity transitions
    # ------------------------------------------

    # For each patient case
    for case_id, group in df.groupby("case_id"):

        # Get list of activities in order
        activities = group["activity"].tolist()

        # Create pairs of consecutive activities
        for i in range(len(activities) - 1):
            transitions.append((activities[i], activities[i + 1]))

    # Convert to DataFrame
    transitions_df = pd.DataFrame(transitions, columns=["from", "to"])

    # ------------------------------------------
    # STEP 2: Count transition frequencies
    # ------------------------------------------

    transition_counts = (
        transitions_df
        .groupby(["from", "to"])
        .size()
        .reset_index(name="count")
    )

    # ------------------------------------------
    # STEP 3: Compute total outgoing transitions
    # ------------------------------------------

    totals = (
        transition_counts
        .groupby("from")["count"]
        .sum()
        .reset_index(name="total")
    )

    # Merge totals with transition counts
    transition_probs = transition_counts.merge(totals, on="from")

    # ------------------------------------------
    # STEP 4: Compute probabilities
    # ------------------------------------------

    transition_probs["probability"] = (
        transition_probs["count"] / transition_probs["total"]
    )

    # Sort for readability
    transition_probs = transition_probs.sort_values(
        ["from", "probability"],
        ascending=[True, False]
    )

    # Save results
    transition_probs.to_csv(output_path, index=False)

    print("\nTransition probabilities:\n")
    print(transition_probs)

    print("\nSaved to:", output_path)


if __name__ == "__main__":

    extract_transition_probabilities(
        "../data/event_log.csv",
        "../data/transition_probabilities.csv"
    )