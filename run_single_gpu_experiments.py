#!/usr/bin/env python3
"""
Wrapper script to run experiments on a specific GPU.
This script is launched as a subprocess with CUDA_VISIBLE_DEVICES set.
"""
import sys
import os
import argparse
import logging
import pickle

# Set logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', type=int, required=True)
    parser.add_argument('--experiments_file', type=str, required=True)
    parser.add_argument('--args_file', type=str, required=True)
    parser.add_argument('--max_parallel', type=int, default=2, help='Maximum parallel experiments per GPU')
    args = parser.parse_args()
    
    # Set CUDA_VISIBLE_DEVICES before importing torch
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    logger.info(f"GPU {args.gpu_id} worker: Set CUDA_VISIBLE_DEVICES={args.gpu_id}")
    
    # Now import torch and other CUDA-related modules
    import torch
    import numpy as np
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    from generate_128_experiments import TurbulenceExperimentGenerator
    
    logger.info(f"GPU {args.gpu_id} worker: torch.cuda.is_available() = {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"GPU {args.gpu_id} worker: torch.cuda.device_count() = {torch.cuda.device_count()}")
    
    # Load experiments list
    with open(args.experiments_file, 'rb') as f:
        experiments = pickle.load(f)
    
    # Load args
    with open(args.args_file, 'rb') as f:
        main_args = pickle.load(f)
    
    logger.info(f"GPU {args.gpu_id} worker: Starting {len(experiments)} experiments with max_parallel={args.max_parallel}")
    
    # Use parallel execution within the GPU process
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    def run_single_experiment_wrapper(experiment_tuple):
        """Wrapper to run a single experiment and handle exceptions."""
        experiment_id, gpu_id, _ = experiment_tuple
        try:
            generator = TurbulenceExperimentGenerator(experiment_id, gpu_id, main_args)
            result = generator.run_simulation()
            logger.info(f"GPU {gpu_id} worker: Completed experiment {experiment_id}")
            return result
        except Exception as e:
            logger.error(f"GPU {gpu_id} worker: Exception in experiment {experiment_id}: {str(e)}")
            return {
                'experiment_id': experiment_id,
                'success': False,
                'error': str(e)
            }
    
    results = []
    
    # Run experiments in parallel using ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=args.max_parallel) as executor:
        # Submit all experiments
        future_to_experiment = {
            executor.submit(run_single_experiment_wrapper, exp): exp 
            for exp in experiments
        }
        
        # Collect results as they complete
        for future in as_completed(future_to_experiment):
            experiment = future_to_experiment[future]
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                experiment_id = experiment[0]
                logger.error(f"GPU {args.gpu_id} worker: Future exception in experiment {experiment_id}: {str(e)}")
                results.append({
                    'experiment_id': experiment_id,
                    'success': False,
                    'error': str(e)
                })
    
    # Save results
    results_file = args.experiments_file.replace('_experiments.pkl', '_results.pkl')
    with open(results_file, 'wb') as f:
        pickle.dump(results, f)
    
    logger.info(f"GPU {args.gpu_id} worker: Completed all experiments, results saved to {results_file}")

if __name__ == "__main__":
    main()

