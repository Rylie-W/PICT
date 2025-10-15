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
    
    logger.info(f"GPU {args.gpu_id} worker: Starting {len(experiments)} experiments")
    
    results = []
    for experiment_id, gpu_id, _ in experiments:
        try:
            generator = TurbulenceExperimentGenerator(experiment_id, gpu_id, main_args)
            result = generator.run_simulation()
            results.append(result)
            logger.info(f"GPU {gpu_id} worker: Completed experiment {experiment_id}")
        except Exception as e:
            logger.error(f"GPU {gpu_id} worker: Exception in experiment {experiment_id}: {str(e)}")
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

