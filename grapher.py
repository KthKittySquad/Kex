import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from collections import defaultdict

def load_labels(csv_path='all_labels.csv'):
    labels_df = pd.read_csv(csv_path)
    cpbeblist = []
    beblist = []
    
    for index, row in labels_df.iterrows():
        if row['Label'] == 1:
            cpbeblist.append(index)
        else:
            beblist.append(index)
            
    print(f"Found {len(cpbeblist)} subjects with CP")

    return cpbeblist, beblist

# Process state sequences for all output folders
def process_state_sequences(cpbeblist, beblist, output_folders):
    results = {}
    
    for folder in output_folders:
        sequences_path = os.path.join(folder, 'state_sequences')
        
        if not os.path.exists(sequences_path):
            print(f"Warning: {sequences_path} does not exist, skipping")
            continue
            
        # Get all CSV files in the state_sequences directory
        sequence_files = [f for f in os.listdir(sequences_path) if f.endswith('.csv')]

        beb_states = defaultdict(int)
        cpbeb_states = defaultdict(int)
        total_frames_beb = 0
        total_frames_cpbeb = 0
        
        index = 0
        for seq_file in sequence_files:
            subject_id = seq_file.split('.')[0]
   
            # Load the state sequence
            seq_path = os.path.join(sequences_path, seq_file)
            try:
                state_seq_df = pd.read_csv(seq_path, header=None)
                state_seq = state_seq_df.iloc[:, 0].values
                state_seq = state_seq[1:]
                
                if index in beblist:
                    for state in state_seq:
                        beb_states[state] += 1
                    total_frames_beb += len(state_seq)
                elif index in cpbeblist:
                    for state in state_seq:
                        cpbeb_states[state] += 1
                    total_frames_cpbeb += len(state_seq)
            except Exception as e:
                print(f"Error processing {seq_file}: {e}")
            
            index += 1
        
        # Calculate proportions
        if total_frames_beb > 0:
            beb_props = {k: v/total_frames_beb for k, v in beb_states.items()}
        else:
            beb_props = {}
            
        if total_frames_cpbeb > 0:
            cpbeb_props = {k: v/total_frames_cpbeb for k, v in cpbeb_states.items()}
        else:
            cpbeb_props = {}
        
        # Store results
        results[folder] = {
            'beb_props': beb_props,
            'cpbeb_props': cpbeb_props,
            'all_states': sorted(set(list(beb_states.keys()) + list(cpbeb_states.keys())), key=int),
            'beb_total': total_frames_beb,
            'cpbeb_total': total_frames_cpbeb
        }
    
    print("Processed")

    return results

# Create usage maps for each output folder
def create_usage_maps(results, output_dir='s7_figure_outputs'):
    os.makedirs(output_dir, exist_ok=True)
    
    for folder, data in results.items():
        folder_name = os.path.basename(folder)
        all_states = data['all_states']
        
        if not all_states:
            print(f"No states found for {folder}, skipping")
            continue
            
        filtered_states = []
        filtered_beb_usage = []
        filtered_cpbeb_usage = []
        filtered_state_indices = []
        
        min_val = 0.001

        # Filter states where either BEB or CPBEB usage is above threshold
        for i, state in enumerate(all_states):
            beb_data = data['beb_props'].get(state, 0)
            cpbeb_data = data['cpbeb_props'].get(state, 0)
            
            if (beb_data >= min_val) or (cpbeb_data >= min_val):
                filtered_states.append(state)
                filtered_beb_usage.append(beb_data)
                filtered_cpbeb_usage.append(cpbeb_data)
                filtered_state_indices.append(i)
        
        # Check if we have data to plot
        if not filtered_states:
            print(f"No states above threshold for {folder}, skipping")
            continue
        
        plt.figure(figsize=(12, 6))
        x = np.arange(len(filtered_states))
        width = 0.35
        
        plt.bar(x - width/2, filtered_beb_usage, width, label='Baby')
        plt.bar(x + width/2, filtered_cpbeb_usage, width, label='Baby with CP')
        
        plt.xlabel('State')
        plt.ylabel('Proportion of Time')
        plt.title(f'State Usage Map - {folder_name}')
        
        # Use the original state indices as x-tick labels
        plt.xticks(x, filtered_state_indices)
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        save_path = os.path.join(output_dir, f"usage_map_{folder_name}.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved usage map for {folder_name} to {save_path}")
        
        # Create a heatmap for state usage differences
        plt.figure(figsize=(12, 3))
        
        diff_usage = np.array(filtered_cpbeb_usage) - np.array(filtered_beb_usage)
        diff_reshaped = diff_usage.reshape(1, -1)
        
        # Create heatmap with filtered states
        sns.heatmap(diff_reshaped, cmap='coolwarm', center=0, 
                   xticklabels=filtered_state_indices, 
                   yticklabels=['Difference (Baby with CP - Baby)'],
                   annot=True, fmt='.3f', 
                   cbar_kws={'label': 'Difference in Usage Proportion'})
        
        plt.title(f'State Usage Difference - {folder_name}')
        plt.tight_layout()
        
        # Save the difference heatmap
        diff_save_path = os.path.join(output_dir, f"usage_diff_{folder_name}.png")
        plt.savefig(diff_save_path, dpi=300, bbox_inches='tight')
        print(f"Saved usage difference map for {folder_name} to {diff_save_path}")
        
        plt.close('all')


def main():
    cpbeblist, beblist = load_labels()
    
    # Define output folders to process
    output_folders = [
        'testing_output_2',
        'testing_output_4',
        'testing_output_8',
        'testing_output_16',
        'testing_output_32'
    ]

    output_dir='testing_figure_outputs'
    
    # Process state sequences
    results = process_state_sequences(cpbeblist, beblist, output_folders)
    
    # Create usage maps
    create_usage_maps(results, output_dir)

    print("Analysis complete!")

if __name__ == "__main__":
    main()