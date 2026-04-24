#!/usr/bin/env python
"""Extract env/success_once from TensorBoard logs and save to CSV."""

import os
import pandas as pd
from tensorboard.backend.event_processing import event_accumulator

def extract_scalar(logdir, tag='env/success_once'):
    """Extract scalar data from TensorBoard event files."""
    ea = event_accumulator.EventAccumulator(logdir)
    ea.Reload()

    # Check if tag exists
    if tag not in ea.Tags()['scalars']:
        print(f"Warning: Tag '{tag}' not found in {logdir}")
        print(f"Available tags: {ea.Tags()['scalars']}")
        return None

    # Extract data
    scalar_data = ea.Scalars(tag)
    df = pd.DataFrame(scalar_data)
    return df[['step', 'value']]

# Define log directories
log_dirs = {
    'hierarchical_dataflow_test': 'logs/20260319-16:40:13-maniskill_hierarchical_dataflow_test.yaml/tensorboard',
    'hier_noise10': 'logs/20260405-12:57:59-maniskill_hier_noise10/tensorboard',
    'hier_S6_both_10': 'logs/20260325-11:38:17-maniskill_hier_S6_both_10/tensorboard'
}

# Extract data from each directory
for name, logdir in log_dirs.items():
    print(f"\nProcessing {name}...")
    if not os.path.exists(logdir):
        print(f"  Directory not found: {logdir}")
        continue

    df = extract_scalar(logdir, 'env/success_once')
    if df is not None:
        output_file = f'{name}_success_once.csv'
        df.to_csv(output_file, index=False)
        print(f"  Saved to {output_file}")
        print(f"  Shape: {df.shape}, Steps: {df['step'].min()}-{df['step'].max()}")

print("\nDone!")
