#!/usr/bin/env python3
"""
Generate 128 turbulence experiments for training data generation.
Each experiment uses 2048x2048 grid, CFL=0.5, samples 166 points in [4.5, 25] time interval,
and downsamples to 512x512 using face-averaging approach.

Features:
- Multi-GPU parallel processing (automatically detects and uses all available GPUs)
- Customizable time step (dt) parameter
- Decaying turbulence (no forcing)
- Memory-efficient storage (only 166 data points per experiment)
- Uses Downsample_domain.py functions for downsampling
"""

import os
import sys
import argparse
import logging
import numpy as np
import torch
import multiprocessing as mp
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
import json
from typing import List, Tuple, Optional
import gc

# Set multiprocessing start method to 'spawn' for CUDA compatibility
# This must be done before any CUDA operations
try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    pass  # Already set

# Add project root to path
sys.path.append(str(Path(__file__).parent))

# PICT imports
import PISOtorch
import PISOtorch_simulation
import lib.data.shapes as shapes
from lib.util.logging import setup_run, get_logger, close_logging
from lib.util.GPU_info import get_available_GPU_id
from lib.util import domain_io
from Downsample_domain import downsample_domain

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TurbulenceExperimentGenerator:
    """Generate individual turbulence experiment with specified parameters."""
    
    def __init__(self, experiment_id: int, gpu_id: int, args):
        self.experiment_id = experiment_id
        self.gpu_id = gpu_id
        self.args = args
        self.dtype = torch.float32
        
        # Set GPU device (use torch.cuda.set_device instead of CUDA_VISIBLE_DEVICES in multiprocessing)
        if torch.cuda.is_available():
            torch.cuda.set_device(gpu_id)
            self.device = torch.device(f"cuda:{gpu_id}")
            logger.info(f"Experiment {experiment_id}: Using GPU {gpu_id} (cuda:{gpu_id})")
        else:
            self.device = torch.device("cpu")
            logger.warning(f"Experiment {experiment_id}: CUDA not available, using CPU")
        
        # Physical parameters
        self.domain_size = 2 * np.pi * self.args.domain_scale
        self.viscosity = self.args.viscosity
        self.max_velocity = self.args.max_velocity
        
        # Time parameters
        self.start_time = 4.5
        self.end_time = 25.0
        self.num_samples = 166
        self.cfl_target = 0.5
        
        # Grid parameters
        self.hr_resolution = 2048  # High resolution
        self.lr_resolution = 512    # Low resolution (downsampled)
    
    def create_domain(self, resolution: int) -> Tuple[PISOtorch.Domain, PISOtorch.Block]:
        """Create a 2D periodic domain for turbulence simulation."""
        
        # Create viscosity tensor on CPU (PISOtorch requirement)
        viscosity = torch.tensor([self.viscosity], dtype=self.dtype, device='cpu')
        
        # Create domain (will be moved to GPU internally by PISOtorch)
        domain = PISOtorch.Domain(
            2,  # 2D
            viscosity,
            name=f"TurbulenceDomain_{resolution}_{self.experiment_id}",
            device=self.device,
            dtype=self.dtype,
            passiveScalarChannels=0
        )
        
        # Create 2D regular grid coordinates
        grid = shapes.make_wall_refined_ortho_grid(
            resolution, resolution,
            corner_lower=(0, 0),
            corner_upper=(self.domain_size, self.domain_size),
            wall_refinement=[],  # No refinement for regular grid
            base=1.0,
            dtype=self.dtype
        )
        grid = grid.to(device=self.device)
        block = domain.CreateBlock(vertexCoordinates=grid, name=f"TurbulenceBlock_{resolution}_{self.experiment_id}")
        
        # Set all boundaries to periodic
        block.MakePeriodic("x")
        block.MakePeriodic("y")
        
        return domain, block
    
    def generate_initial_velocity(self, domain: PISOtorch.Domain, block: PISOtorch.Block) -> torch.Tensor:
        """Generate initial divergence-free turbulent velocity field."""
        
        # Get block size
        block_size = block.getSizes()
        shape = [1, 2, block_size.y, block_size.x]  # [1, 2, y, x]
        
        # Generate random velocity field using spectral method
        velocity = self._generate_divergence_free_field(shape)
        
        # Scale to desired maximum velocity
        velocity_magnitude = torch.sqrt(torch.sum(velocity**2, dim=1, keepdim=True))
        max_vel = torch.max(velocity_magnitude).item()
        
        if max_vel > 0:
            velocity = velocity * (self.max_velocity / max_vel)
        
        return velocity
    
    def _generate_divergence_free_field(self, shape: List[int]) -> torch.Tensor:
        """
        Generate divergence-free velocity field with LARGE VISIBLE VORTEX STRUCTURES.
        
        Key parameters for vortex size control:
        - integral_scale_factor: SMALLER = LARGER vortices (2.0 = very large, 6.0 = small)
        - dissipation_cutoff_strength: LARGER = CLEARER vortices (more small-scale suppression)
        - Re_lambda: SMALLER = LESS noise (fewer small-scale structures)
        """
        
        ny, nx = shape[2], shape[3]
        
        # Create wavenumber grids
        ky = torch.fft.fftfreq(ny, device=self.device)
        kx = torch.fft.fftfreq(nx, device=self.device)
        KY, KX = torch.meshgrid(ky, kx, indexing='ij')
        k_mag = torch.sqrt(KX**2 + KY**2)
        
        # ========== TUNABLE PARAMETERS FOR VORTEX SIZE ==========
        domain_size = max(ny, nx)
        
        # Parameter 1: Integral scale (MOST IMPORTANT for vortex size)
        # Smaller value = larger vortices
        # Recommended range: 2.0-6.0
        integral_scale_factor = 2.0  # OPTIMAL: large, visible vortices
        
        # Parameter 2: Taylor Reynolds number (controls small-scale turbulence)
        # Smaller value = less noise, clearer vortices
        # Recommended range: 20-60
        Re_lambda = 25.0  # OPTIMAL: minimal noise
        
        # Parameter 3: Dissipation cutoff strength
        # Larger value = more aggressive small-scale suppression
        # Recommended range: 2.0-5.0
        dissipation_cutoff_strength = 4.0  # OPTIMAL: clear vortex edges
        # ========================================================
        
        L_integral = domain_size / integral_scale_factor
        k0 = 1.0 / L_integral
        
        # Control dissipation scale
        eta_over_L = Re_lambda**(-3/4)
        k_eta = 1.0 / (eta_over_L * L_integral)
        
        # Create random streamfunction in Fourier space
        streamfunction_fft = torch.complex(
            torch.randn(ny, nx, device=self.device),
            torch.randn(ny, nx, device=self.device)
        )
        
        # Improved Von Karman spectrum for 2D turbulence
        k_over_k0 = k_mag / k0
        k_over_keta = k_mag / k_eta
        
        # Base Von Karman spectrum
        energy_spectrum = (k_over_k0**4) / (1 + k_over_k0**2)**(17/6)
        
        # Add exponential cutoff at dissipation scale (CRITICAL: suppresses small scales)
        energy_spectrum *= torch.exp(-dissipation_cutoff_strength * k_over_keta**2)
        
        # Normalize to ensure reasonable energy levels (CRITICAL: emphasizes large scales)
        k_peak_theory = k0 * (4.0/13.0)**(1/2)
        peak_mask = (k_mag >= k_peak_theory * 0.8) & (k_mag <= k_peak_theory * 1.2)
        if torch.any(peak_mask):
            energy_spectrum = energy_spectrum / torch.max(energy_spectrum[peak_mask])
        
        # Remove DC component
        energy_spectrum[k_mag < 1e-10] = 0
        
        # Apply smoothing near k=0 for smooth large-scale transitions
        k_smooth = k0 / 10.0
        smooth_factor = torch.tanh(k_mag / k_smooth)
        energy_spectrum *= smooth_factor
        
        # Apply spectrum
        streamfunction_fft *= torch.sqrt(energy_spectrum)
        
        # Compute velocity components from streamfunction: u = (-∂ψ/∂y, ∂ψ/∂x)
        ux_fft = -1j * (2*np.pi) * KY * streamfunction_fft
        uy_fft = 1j * (2*np.pi) * KX * streamfunction_fft
        
        # Convert back to physical space
        ux = torch.fft.ifftn(ux_fft).real
        uy = torch.fft.ifftn(uy_fft).real
        
        velocity = torch.stack([ux, uy], dim=0).unsqueeze(0)
        
        return velocity.to(dtype=self.dtype)
    
    def compute_timestep(self, velocity_field: torch.Tensor, resolution: int) -> float:
        """Compute timestep based on CFL condition."""
        
        # Grid spacing
        dx = self.domain_size / resolution
        
        # Maximum velocity
        velocity_magnitude = torch.sqrt(torch.sum(velocity_field**2, dim=1))
        max_velocity = torch.max(velocity_magnitude).item()
        
        # CFL-based timestep
        dt_cfl = self.cfl_target * dx / max_velocity if max_velocity > 0 else 1e-3
        
        # Viscous stability condition
        dt_viscous = 0.5 * dx**2 / self.viscosity if self.viscosity > 0 else 1e10
        
        # Take the most restrictive condition
        dt = min(dt_cfl, dt_viscous)
        
        # Apply safety factor
        dt_final = 0.8 * dt
        
        # Ensure reasonable bounds
        dt_final = max(1e-6, min(0.01, dt_final))
        
        return dt_final
    
    def run_simulation(self) -> dict:
        """Run single turbulence experiment and return results."""
        
        try:
            logger.info(f"Starting experiment {self.experiment_id} on GPU {self.gpu_id}")
            
            # Create output directory for this experiment
            exp_dir = Path(self.args.save_dir) / f"experiment_{self.experiment_id:03d}"
            exp_dir.mkdir(parents=True, exist_ok=True)
            
            # Create high-resolution domain
            hr_domain, hr_block = self.create_domain(self.hr_resolution)
            
            # Generate initial velocity
            initial_velocity = self.generate_initial_velocity(hr_domain, hr_block)
            hr_block.setVelocity(initial_velocity)
            hr_domain.PrepareSolve()
            
            # Compute timestep (use custom dt if provided)
            if self.args.dt is not None:
                dt = self.args.dt
                logger.info(f"Experiment {self.experiment_id}: Using custom dt = {dt:.2e}")
            else:
                dt = self.compute_timestep(initial_velocity, self.hr_resolution)
                logger.info(f"Experiment {self.experiment_id}: Computed dt = {dt:.2e}")
            
            # Create low-resolution domain for downsampling (create once, reuse)
            lr_domain, lr_block = self.create_domain(self.lr_resolution)
            
            # Save initial condition (t=0) in downsampled resolution
            logger.info(f"Experiment {self.experiment_id}: Saving initial condition (t=0)")
            downsample_domain(lr_domain, hr_domain, only_velocity=False)  # Save full domain at t=0
            initial_save_path = exp_dir / f"initial_condition_t_0.000"
            domain_io.save_domain(lr_domain, str(initial_save_path))
            logger.info(f"Experiment {self.experiment_id}: Saved initial condition to {initial_save_path}")
            
            # Create simulation
            sim = PISOtorch_simulation.Simulation(
                domain=hr_domain,
                time_step=dt,
                substeps="ADAPTIVE",
                corrector_steps=2,
                non_orthogonal=False,
                pressure_tol=1e-6,
                velocity_corrector="FD",
                adaptive_CFL=self.cfl_target,
                log_interval=1000,
                log_dir=None,
                stop_fn=lambda: False
            )
            
            # Time sampling points
            time_points = np.linspace(self.start_time, self.end_time, self.num_samples)
            current_time = 0.0
            sample_idx = 0
            
            # List to store saved file paths
            saved_files = [str(initial_save_path)]  # Include initial condition
            
            logger.info(f"Experiment {self.experiment_id}: Running simulation to t={self.end_time}")
            
            # Run simulation with time sampling
            while current_time < self.end_time and sample_idx < self.num_samples:
                # Check if we need to sample at this time
                if sample_idx < len(time_points) and current_time >= time_points[sample_idx]:
                    # Downsample current state immediately
                    downsample_domain(lr_domain, hr_domain, only_velocity=True)
                    
                    # Save downsampled domain using domain_io
                    save_path = exp_dir / f"timestep_{sample_idx:03d}_t_{current_time:.3f}"
                    domain_io.save_domain(lr_domain, str(save_path))
                    saved_files.append(str(save_path))
                    
                    sample_idx += 1
                    logger.info(f"Experiment {self.experiment_id}: Sampled and saved at t={current_time:.2f} ({sample_idx}/{self.num_samples})")
                
                # Run simulation step
                sim.run(iterations=1)
                current_time += dt
            
            # Ensure we have the right number of samples (sample final state if needed)
            while sample_idx < self.num_samples:
                downsample_domain(lr_domain, hr_domain, only_velocity=True)
                save_path = exp_dir / f"timestep_{sample_idx:03d}_t_{current_time:.3f}"
                domain_io.save_domain(lr_domain, str(save_path))
                saved_files.append(str(save_path))
                sample_idx += 1
                logger.info(f"Experiment {self.experiment_id}: Final sample {sample_idx}/{self.num_samples}")
            
            # Save metadata
            metadata = {
                'experiment_id': self.experiment_id,
                'gpu_id': self.gpu_id,
                'hr_resolution': self.hr_resolution,
                'lr_resolution': self.lr_resolution,
                'dt': float(dt),
                'time_points': time_points.tolist(),
                'domain_size': float(self.domain_size),
                'viscosity': float(self.viscosity),
                'max_velocity': float(self.max_velocity),
                'cfl_target': float(self.cfl_target),
                'num_samples': self.num_samples,
                'initial_condition_file': str(initial_save_path),
                'saved_files': saved_files,
                'success': True,
                'note': 'Initial condition (t=0) stored separately as initial_condition_t_0.000'
            }
            
            metadata_file = exp_dir / f"metadata_{self.experiment_id:03d}.json"
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"Experiment {self.experiment_id}: Completed successfully, {len(saved_files)} files saved")
            
            # Return lightweight result
            result = {
                'experiment_id': self.experiment_id,
                'gpu_id': self.gpu_id,
                'num_files_saved': len(saved_files),
                'experiment_dir': str(exp_dir),
                'success': True
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Experiment {self.experiment_id} failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return {
                'experiment_id': self.experiment_id,
                'gpu_id': self.gpu_id,
                'error': str(e),
                'success': False
            }
        finally:
            # Clean up GPU memory
            if 'hr_domain' in locals():
                del hr_domain
            if 'lr_domain' in locals():
                del lr_domain
            if 'sim' in locals():
                del sim
            torch.cuda.empty_cache()
            gc.collect()

def run_single_experiment(args_tuple):
    """Wrapper function for running single experiment in multiprocessing."""
    experiment_id, gpu_id, args = args_tuple
    generator = TurbulenceExperimentGenerator(experiment_id, gpu_id, args)
    return generator.run_simulation()

def detect_available_gpus() -> List[int]:
    """Detect all available GPUs using nvidia-smi or torch cuda device count."""
    available_gpus = []
    
    # Method 1: Use torch.cuda.device_count() (most reliable)
    try:
        if torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()
            available_gpus = list(range(num_gpus))
            return available_gpus
    except Exception as e:
        logger.warning(f"Failed to detect GPUs using torch: {e}")
    
    # Method 2: Fallback - try nvidia-smi
    try:
        import subprocess
        result = subprocess.run(['nvidia-smi', '-L'], 
                              capture_output=True, text=True, check=True)
        # Count lines starting with "GPU"
        gpu_lines = [line for line in result.stdout.split('\n') if line.startswith('GPU')]
        available_gpus = list(range(len(gpu_lines)))
        return available_gpus
    except Exception as e:
        logger.warning(f"Failed to detect GPUs using nvidia-smi: {e}")
    
    # Method 3: Last resort - assume single GPU if cuda is available
    if torch.cuda.is_available():
        return [0]
    
    return available_gpus

def save_experiment_data(experiment_data: dict, save_dir: Path):
    """Log experiment completion (data already saved during simulation)."""
    
    experiment_id = experiment_data['experiment_id']
    
    if not experiment_data.get('success', True):
        logger.error(f"Experiment {experiment_id} failed")
        return
    
    # Data is already saved during simulation using domain_io
    logger.info(f"Experiment {experiment_id} completed: {experiment_data['num_files_saved']} files saved to {experiment_data['experiment_dir']}")

def main():
    """Main function to generate 128 turbulence experiments."""
    
    # Ensure spawn method is set (redundant but safe)
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    
    parser = argparse.ArgumentParser(description='Generate 128 turbulence experiments')
    
    # Physical parameters
    parser.add_argument('--viscosity', type=float, default=1e-3, help='Kinematic viscosity')
    parser.add_argument('--max_velocity', type=float, default=4.2, help='Maximum velocity')
    parser.add_argument('--domain_scale', type=float, default=1.0, help='Domain scale factor')
    
    # Time parameters
    parser.add_argument('--dt', type=float, default=None, help='Custom time step (overrides CFL calculation)')
    parser.add_argument('--start_time', type=float, default=4.5, help='Start time for sampling')
    parser.add_argument('--end_time', type=float, default=25.0, help='End time for sampling')
    parser.add_argument('--num_samples', type=int, default=166, help='Number of time samples')
    
    # Output parameters
    parser.add_argument('--save_dir', type=str, default='./data/128_experiments', help='Output directory')
    parser.add_argument('--num_experiments', type=int, default=128, help='Number of experiments to generate')
    parser.add_argument('--max_workers', type=int, default=None, help='Maximum number of parallel workers')
    
    args = parser.parse_args()
    
    # Detect available GPUs
    available_gpus = detect_available_gpus()
    if not available_gpus:
        logger.error("No GPUs detected!")
        return
    
    logger.info(f"Detected GPUs: {available_gpus}")
    
    # Create output directory
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Prepare experiment arguments
    num_experiments = args.num_experiments
    num_gpus = len(available_gpus)
    
    # Distribute experiments across GPUs
    experiments_per_gpu = num_experiments // num_gpus
    remaining_experiments = num_experiments % num_gpus
    
    experiment_args = []
    experiment_id = 0
    
    for gpu_idx, gpu_id in enumerate(available_gpus):
        # Calculate number of experiments for this GPU
        gpu_experiments = experiments_per_gpu
        if gpu_idx < remaining_experiments:
            gpu_experiments += 1
        
        # Add experiments for this GPU
        for _ in range(gpu_experiments):
            experiment_args.append((experiment_id, gpu_id, args))
            experiment_id += 1
    
    logger.info(f"Distributing {num_experiments} experiments across {num_gpus} GPUs")
    for gpu_id in available_gpus:
        gpu_experiments = sum(1 for _, gid, _ in experiment_args if gid == gpu_id)
        logger.info(f"GPU {gpu_id}: {gpu_experiments} experiments")
    
    # Run experiments in parallel
    max_workers = args.max_workers or min(num_gpus, mp.cpu_count())
    logger.info(f"Running experiments with {max_workers} parallel workers")
    
    start_time = time.time()
    successful_experiments = 0
    failed_experiments = 0
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all experiments
        future_to_experiment = {
            executor.submit(run_single_experiment, exp_args): exp_args[0] 
            for exp_args in experiment_args
        }
        
        # Process completed experiments
        for future in as_completed(future_to_experiment):
            experiment_id = future_to_experiment[future]
            
            try:
                result = future.result()
                
                if result.get('success', True):
                    save_experiment_data(result, save_dir)
                    successful_experiments += 1
                    logger.info(f"Completed experiment {experiment_id} ({successful_experiments}/{num_experiments})")
                else:
                    failed_experiments += 1
                    logger.error(f"Failed experiment {experiment_id}")
                
            except Exception as e:
                failed_experiments += 1
                logger.error(f"Exception in experiment {experiment_id}: {str(e)}")
    
    # Summary
    total_time = time.time() - start_time
    logger.info(f"\n{'='*60}")
    logger.info(f"EXPERIMENT GENERATION COMPLETED!")
    logger.info(f"{'='*60}")
    logger.info(f"Total experiments: {num_experiments}")
    logger.info(f"Successful: {successful_experiments}")
    logger.info(f"Failed: {failed_experiments}")
    logger.info(f"Total time: {total_time:.2f} seconds")
    logger.info(f"Average time per experiment: {total_time/num_experiments:.2f} seconds")
    logger.info(f"Results saved to: {save_dir}")
    
    # Save summary
    summary = {
        'total_experiments': num_experiments,
        'successful_experiments': successful_experiments,
        'failed_experiments': failed_experiments,
        'total_time_seconds': total_time,
        'average_time_per_experiment': total_time / num_experiments,
        'gpus_used': available_gpus,
        'parameters': vars(args)
    }
    
    summary_file = save_dir / "generation_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"Summary saved to: {summary_file}")

if __name__ == "__main__":
    # This guard is required for multiprocessing with 'spawn' method
    main()
