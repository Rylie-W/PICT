import os
import argparse
import logging
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path

# PICT imports
import PISOtorch
import PISOtorch_simulation
import lib.data.shapes as shapes
from lib.util.logging import setup_run, get_logger, close_logging
from lib.util.GPU_info import get_available_GPU_id

# Set GPU
os.environ["CUDA_VISIBLE_DEVICES"] = str(get_available_GPU_id(active_mem_threshold=0.8, default=None))

assert torch.cuda.is_available()
cuda_device = torch.device("cuda")
cpu_device = torch.device("cpu")


class TurbulenceDataGenerator:
    def __init__(self, args):
        self.args = args
        self.dtype = torch.float32
        self.logger = logging.getLogger("TurbulenceGen")
        
    def create_domain(self, resolution):
        """Create a 3D periodic domain for turbulence simulation"""
        dims = 3 if self.args.dims == 3 else 2
        
        # Physical domain size
        domain_length = 2 * np.pi * self.args.domain_scale
        
        # Create viscosity tensor
        viscosity = torch.tensor([self.args.viscosity], dtype=self.dtype)
        
        # Create domain
        domain = PISOtorch.Domain(
            dims,
            viscosity,
            name=f"TurbulenceDomain_{resolution}",
            device=cuda_device,
            dtype=self.dtype,
            passiveScalarChannels=0
        )
        
        # Create block with specified resolution
        if dims == 3:
            size = [resolution, resolution, resolution]
        else:
            size = [resolution, resolution]
            
        block = domain.CreateBlockWithSize(size, name=f"TurbulenceBlock_{resolution}")
        
        # Set all boundaries to periodic
        for boundary_name in ["-x", "+x", "-y", "+y"] + (["-z", "+z"] if dims == 3 else []):
            block.setBoundary(boundary_name, PISOtorch.BoundaryType.PERIODIC)
        
        return domain, block
    
    def generate_initial_turbulence(self, domain, block):
        """Generate initial turbulent velocity field"""
        dims = domain.getSpatialDims()
        block_size = block.getSize()
        
        # Generate random velocity field
        if dims == 3:
            shape = [1, dims] + list(block_size.tolist())[::-1]  # [1, 3, z, y, x]
        else:
            shape = [1, dims] + list(block_size.tolist())[::-1]  # [1, 2, y, x]
        
        # Create Gaussian random field
        velocity = torch.randn(shape, dtype=self.dtype, device=cuda_device)
        
        # Apply spectral filtering to get realistic turbulence
        velocity = self._apply_spectral_filter(velocity, self.args.peak_wavenumber)
        
        # Scale to desired maximum velocity
        velocity_magnitude = torch.sqrt(torch.sum(velocity**2, dim=1, keepdim=True))
        max_vel = torch.max(velocity_magnitude)
        velocity = velocity * (self.args.max_velocity / max_vel)
        
        # Set velocity field
        block.setVelocity(velocity)
        
        return velocity
    
    def _apply_spectral_filter(self, velocity, peak_wavenumber):
        """Apply spectral filtering to create realistic turbulence spectrum"""
        # Convert to frequency domain
        velocity_fft = torch.fft.fftn(velocity, dim=list(range(2, velocity.ndim)))
        
        # Create wavenumber grid
        shape = velocity.shape[2:]
        if len(shape) == 3:  # 3D
            kz = torch.fft.fftfreq(shape[0], device=cuda_device)
            ky = torch.fft.fftfreq(shape[1], device=cuda_device) 
            kx = torch.fft.fftfreq(shape[2], device=cuda_device)
            KZ, KY, KX = torch.meshgrid(kz, ky, kx, indexing='ij')
            k_mag = torch.sqrt(KX**2 + KY**2 + KZ**2)
        else:  # 2D
            ky = torch.fft.fftfreq(shape[0], device=cuda_device)
            kx = torch.fft.fftfreq(shape[1], device=cuda_device)
            KY, KX = torch.meshgrid(ky, kx, indexing='ij')
            k_mag = torch.sqrt(KX**2 + KY**2)
        
        # Apply Kolmogorov-like spectrum filter
        # E(k) ~ k^(-5/3) for inertial range
        filter_func = torch.exp(-(k_mag / peak_wavenumber)**2) * (k_mag + 1e-8)**(-5/6)
        filter_func[k_mag == 0] = 0  # Remove DC component
        
        # Apply filter
        velocity_fft = velocity_fft * filter_func.unsqueeze(0).unsqueeze(0)
        
        # Convert back to physical space
        velocity_filtered = torch.fft.ifftn(velocity_fft, dim=list(range(2, velocity.ndim))).real
        
        return velocity_filtered
    
    def run_simulation(self, domain, resolution, steps, save_interval=50):
        """Run turbulence simulation and collect data"""
        self.logger.info(f"Running simulation at {resolution}^{domain.getSpatialDims()} resolution for {steps} steps")
        
        # Calculate time step based on CFL condition
        dx = (2 * np.pi * self.args.domain_scale) / resolution
        time_step = self.args.cfl_safety_factor * dx / self.args.max_velocity
        
        # Create simulation
        sim = PISOtorch_simulation.Simulation(
            domain=domain,
            time_step=time_step,
            substeps=1,
            corrector_steps=2,
            non_orthogonal=False,
            pressure_tol=1e-6,
            velocity_corrector="FD",
            log_interval=save_interval,
            log_dir=None,  # No intermediate logging
            stop_fn=lambda: False
        )
        
        # Storage for trajectory data
        trajectory_data = []
        
        # Run simulation and collect data
        for step in range(0, steps, save_interval):
            sim.run(iterations=save_interval)
            
            # Get current velocity field
            velocity = domain.getBlock(0).velocity.detach().cpu().numpy()
            trajectory_data.append(velocity.copy())
            
            if step % (save_interval * 10) == 0:
                self.logger.info(f"Completed {step}/{steps} steps at resolution {resolution}")
        
        return np.array(trajectory_data)
    
    def warmup_simulation(self, domain, resolution):
        """Run warmup simulation to reach statistical steady state"""
        warmup_steps = int(self.args.warmup_time / self.get_time_step(resolution))
        self.logger.info(f"Running warmup for {warmup_steps} steps at resolution {resolution}")
        
        dx = (2 * np.pi * self.args.domain_scale) / resolution
        time_step = self.args.cfl_safety_factor * dx / self.args.max_velocity
        
        sim = PISOtorch_simulation.Simulation(
            domain=domain,
            time_step=time_step,
            substeps=1,
            corrector_steps=2,
            non_orthogonal=False,
            pressure_tol=1e-6,
            velocity_corrector="FD",
            log_interval=max(warmup_steps // 10, 1),
            log_dir=None,
            stop_fn=lambda: False
        )
        
        sim.run(iterations=warmup_steps)
        self.logger.info(f"Warmup completed at resolution {resolution}")
    
    def get_time_step(self, resolution):
        """Calculate stable time step for given resolution"""
        dx = (2 * np.pi * self.args.domain_scale) / resolution
        return self.args.cfl_safety_factor * dx / self.args.max_velocity
    
    def downsample_velocity(self, velocity_hr, target_resolution, source_resolution):
        """Downsample high-resolution velocity to target resolution"""
        # Simple downsampling by taking every nth point
        factor = source_resolution // target_resolution
        
        if len(velocity_hr.shape) == 5:  # 3D: [1, 3, z, y, x]
            downsampled = velocity_hr[:, :, ::factor, ::factor, ::factor]
        else:  # 2D: [1, 2, y, x]
            downsampled = velocity_hr[:, :, ::factor, ::factor]
        
        return downsampled
    
    def generate_data(self):
        """Main data generation pipeline"""
        self.logger.info("Starting turbulence data generation with PICT")
        
        # Create high-resolution domain for initial conditions
        hr_domain, hr_block = self.create_domain(self.args.high_res)
        
        # Generate initial turbulent field
        initial_velocity = self.generate_initial_turbulence(hr_domain, hr_block)
        hr_domain.PrepareSolve()
        
        # Run warmup at high resolution
        self.warmup_simulation(hr_domain, self.args.high_res)
        
        # Get resolutions to generate
        resolution_list = []
        res = self.args.low_res
        while res <= self.args.high_res:
            resolution_list.append(res)
            res *= 2
        
        # Generate data for each resolution
        for resolution in resolution_list:
            self.logger.info(f"Generating data for resolution {resolution}")
            
            if resolution == self.args.high_res:
                # Use existing high-res domain
                domain = hr_domain
            else:
                # Create domain at target resolution
                domain, block = self.create_domain(resolution)
                
                # Downsample velocity from high-res
                hr_velocity = hr_domain.getBlock(0).velocity
                downsampled_velocity = self.downsample_velocity(
                    hr_velocity, resolution, self.args.high_res
                )
                block.setVelocity(downsampled_velocity)
                domain.PrepareSolve()
            
            # Run simulation and collect trajectory
            trajectory = self.run_simulation(
                domain, resolution, self.args.generate_steps, 
                save_interval=self.args.save_interval
            )
            
            # Save data
            self.save_trajectory_data(trajectory, resolution)
    
    def save_trajectory_data(self, trajectory, resolution):
        """Save trajectory data in numpy format"""
        save_dir = Path(self.args.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        data_file = save_dir / f"{self.args.save_file}_{resolution}x{resolution}_index_{self.args.save_index}.npz"
        
        self.logger.info(f"Saving training data to: {data_file}")
        
        # Extract velocity components
        if trajectory.shape[2] == 3:  # 3D
            u_data = trajectory[:, 0, 0, :, :, :]  # x-velocity
            v_data = trajectory[:, 0, 1, :, :, :]  # y-velocity  
            w_data = trajectory[:, 0, 2, :, :, :]  # z-velocity
            
            np.savez_compressed(
                data_file,
                u=u_data,
                v=v_data,
                w=w_data,
                resolution=resolution,
                steps=self.args.generate_steps,
                warmup_time=self.args.warmup_time,
                max_velocity=self.args.max_velocity,
                viscosity=self.args.viscosity,
                decay=self.args.decay,
                seed=self.args.seed,
                dims=3
            )
        else:  # 2D
            u_data = trajectory[:, 0, 0, :, :]  # x-velocity
            v_data = trajectory[:, 0, 1, :, :]  # y-velocity
            
            np.savez_compressed(
                data_file,
                u=u_data,
                v=v_data,
                resolution=resolution,
                steps=self.args.generate_steps,
                warmup_time=self.args.warmup_time,
                max_velocity=self.args.max_velocity,
                viscosity=self.args.viscosity,
                decay=self.args.decay,
                seed=self.args.seed,
                dims=2
            )
        
        self.logger.info(f"Saved trajectory shape: {trajectory.shape}")
        self.logger.info(f"Resolution: {resolution}x{resolution}, Steps: {self.args.generate_steps}")


def main():
    parser = argparse.ArgumentParser(description='Generate turbulence training data using PICT')
    
    # Simulation parameters
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--dims', type=int, default=2, choices=[2, 3], 
                       help='Spatial dimensions (2D or 3D)')
    parser.add_argument('--generate_steps', type=int, default=12200)
    parser.add_argument('--save_interval', type=int, default=50,
                       help='Save data every N simulation steps')
    parser.add_argument('--warmup_time', type=float, default=40.0)
    
    # Physical parameters
    parser.add_argument('--max_velocity', type=float, default=4.2)
    parser.add_argument('--cfl_safety_factor', type=float, default=0.5)
    parser.add_argument('--viscosity', type=float, default=1e-3)
    parser.add_argument('--peak_wavenumber', type=int, default=4)
    parser.add_argument('--domain_scale', type=float, default=1.0)
    parser.add_argument('--decay', action='store_true', default=True,
                       help='Generate decaying turbulence (no forcing)')
    
    # Resolution parameters
    parser.add_argument('--low_res', type=int, default=64)
    parser.add_argument('--high_res', type=int, default=512,  # Reduced from 2048 for PICT
                       help='Highest resolution (limited by GPU memory)')
    
    # Output parameters
    parser.add_argument('--save_dir', type=str, default='./data/pict_turbulence')
    parser.add_argument('--save_file', type=str, default="decaying_turbulence")
    parser.add_argument('--save_index', type=int, default=1)
    
    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("Main")
    
    logger.info(f"Starting PICT turbulence data generation")
    logger.info(f"Parameters: {vars(args)}")
    
    # Generate data
    generator = TurbulenceDataGenerator(args)
    generator.generate_data()
    
    logger.info("Data generation completed!")


if __name__ == "__main__":
    main() 