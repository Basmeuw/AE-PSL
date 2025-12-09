import os
import sys
import json
import itertools
import argparse
import copy
import traceback
from datetime import datetime
from argparse import Namespace


from utils.orchestrator_argument_utils import (
    build_base_argument_parser,
    expand_argument_parser_with_adapter_approach_parameters,
    expand_argument_parser_with_ae_pretraining_parameters,
    expand_argument_parser_with_distributed_learning_parameters,
    namespace_to_global_args_and_search_space,
    set_env_variables
)
from main_centralized_2_stage import run_2_stage


def setup_orchestrator_parser():
    """
    Constructs the parser by chaining the utility functions provided.
    Adds orchestrator-specific arguments.
    """
    parser = build_base_argument_parser()
    parser = expand_argument_parser_with_adapter_approach_parameters(parser)
    parser = expand_argument_parser_with_ae_pretraining_parameters(parser)
    # parser = expand_argument_parser_with_distributed_learning_parameters(parser)

    # Orchestrator specific controls
    parser.add_argument('--experiment_name', type=str, default='experiment',
                        help='Name/Tag for the experiment campaign folder.')
    parser.add_argument('--experiments_dir', type=str, default='experiment',
                        help='Name/Tag for the experiment campaign folder.')
    parser.add_argument('--resume_from_manifest', type=str, default=None,
                        help='Path to a manifest.json file to resume an interrupted campaign.')

    return parser


def generate_manifest(experiment_name, experiments_dir, global_args, search_space):
    """
    Creates the experiment directory and the manifest.json file containing all job configurations.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    campaign_folder_name = f"{timestamp}_{experiment_name}"

    # Define campaign directory (relative to this script: ../experiments/...)
    campaign_dir = os.path.join(experiments_dir, campaign_folder_name)

    os.makedirs(os.path.join(campaign_dir, "logs"), exist_ok=True)

    # Create Combinations
    # Sort keys to ensure deterministic ordering of parameters
    keys = sorted(search_space.keys())
    values = [search_space[k] for k in keys]
    combinations = list(itertools.product(*values))

    jobs = []
    for combo in combinations:
        job_params = dict(zip(keys, combo))
        jobs.append({
            "params": job_params
        })

    manifest = {
        "experiment_name": experiment_name,
        "created_at": timestamp,
        "global_args": global_args,
        "search_space": search_space,
        "jobs": jobs
    }

    manifest_path = os.path.join(campaign_dir, "manifest.json")
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=4)

    print(f"Initialized Campaign: {campaign_dir}")
    print(f"Generated {len(jobs)} jobs based on search space.")

    return campaign_dir, manifest


def get_job_filename(params):
    """
    Creates a unique, readable filename for a job based on its varying parameters.
    """
    if not params:
        return "run_default"

    # Mapping for shortening common long parameter names
    abbreviations = {
        'learning_rate': 'lr',
        'start_lr': 'lr',
        'batch_size': 'bs',
        'random_seed': 'seed',
        'dataset': 'ds',
        'split_layer': 'split',
        'ae_latent_dim': 'lat',
        'ae_type': 'ae'
    }

    parts = []
    for k in sorted(params.keys()):
        key_str = abbreviations.get(k, k)
        val_str = str(params[k])
        parts.append(f"{key_str}_{val_str}")

    return "run_" + "_".join(parts)


def run_orchestrator():
    parser = setup_orchestrator_parser()
    args = parser.parse_args()

    # 1. Setup Campaign (New or Resume)
    if args.resume_from_manifest:
        if not os.path.exists(args.resume_from_manifest):
            print(f"Error: Manifest file not found at {args.resume_from_manifest}")
            return

        print(f"Resuming campaign from: {args.resume_from_manifest}")
        with open(args.resume_from_manifest, 'r') as f:
            manifest = json.load(f)

        global_args = manifest['global_args']
        jobs = manifest['jobs']
        campaign_dir = os.path.dirname(os.path.abspath(args.resume_from_manifest))

    else:
        # Separate constant arguments from search space lists
        global_args, search_space = namespace_to_global_args_and_search_space(args)

        # Remove orchestrator-specific args from global_args to keep it clean
        global_args.pop('experiment_name', None)
        global_args.pop('experiments_dir', None)
        global_args.pop('resume_from_manifest', None)

        campaign_dir, manifest = generate_manifest(args.experiment_name, args.experiments_dir, global_args, search_space)
        jobs = manifest['jobs']

    # 2. Configure Environment
    # We use the global_args to set env variables (data dirs, etc.)
    set_env_variables(global_args)

    total_jobs = len(jobs)
    print(f"Starting Execution of {total_jobs} jobs...")

    # 3. Execution Loop
    for i, job in enumerate(jobs):
        job_params = job['params']

        # Merge global args with specific job params
        current_run_args = copy.deepcopy(global_args)
        current_run_args.update(job_params)

        # Determine output filename
        filename = get_job_filename(job_params)

        # We use absolute path for save_file_name to override the default behavior 
        # of saving into os.environ['MODEL_WEIGHTS_DIR']
        save_file_path = os.path.join("logs", filename)
        current_run_args['save_file_name'] = save_file_path
        os.environ['MODEL_WEIGHTS_DIR'] = campaign_dir

        # Check for existing result (Resume Logic)
        # Note: save_experiment_results appends '.json'
        expected_output_file = save_file_path + ".json"

        if os.path.exists(expected_output_file):
            print(f"[{i + 1}/{total_jobs}] SKIPPING: {filename} (Result exists)")
            continue

        print(f"[{i + 1}/{total_jobs}] RUNNING: {filename}")
        print(f"   Params: {job_params}")

        try:
            # Execute the training stage directly
            # This handles AE pre-training/loading internally
            run_2_stage(current_run_args, job_params)

        except KeyboardInterrupt:
            print("\nOrchestrator interrupted by user.")
            break
        except Exception as e:
            print(f"!!! JOB FAILED: {filename}")
            print(f"Error: {e}")
            traceback.print_exc()
            # We continue to the next job instead of crashing the whole campaign
            # continue

    print("Orchestrator finished.")


if __name__ == "__main__":
    run_orchestrator()