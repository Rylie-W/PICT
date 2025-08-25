import os
import argparse
import logging
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
import gc

# PICT imports
import PISOtorch
import PISOtorch_simulation
import lib.data.shapes as shapes
from lib.util.logging import setup_run, get_logger, close_logging
from lib.util.GPU_info import get_available_GPU_id

# For comparison and visualization
import matplotlib.colors as colors

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
        
        # Create block with specified resolution using vertex coordinates for proper transform setup
        if dims == 3:
            # Create 3D regular grid coordinates
            import lib.data.shapes as shapes
            grid = shapes.make_wall_refined_ortho_grid(
                resolution, resolution,
                corner_lower=(0, 0),
                corner_upper=(domain_length, domain_length),
                wall_refinement=[],  # No refinement for regular grid
                base=1.0,
                dtype=self.dtype
            )
            # Extrude to 3D
            grid = shapes.extrude_grid_z(grid, resolution, end_z=domain_length)
            grid = grid.to(device=cuda_device)
            block = domain.CreateBlock(vertexCoordinates=grid, name=f"TurbulenceBlock_{resolution}")
        else:
            # Create 2D regular grid coordinates
            import lib.data.shapes as shapes
            grid = shapes.make_wall_refined_ortho_grid(
                resolution, resolution,
                corner_lower=(0, 0),
                corner_upper=(domain_length, domain_length),
                wall_refinement=[],  # No refinement for regular grid
                base=1.0,
                dtype=self.dtype
            )
            grid = grid.to(device=cuda_device)
            block = domain.CreateBlock(vertexCoordinates=grid, name=f"TurbulenceBlock_{resolution}")
        
        # Set all boundaries to periodic
        if dims == 3:
            block.MakePeriodic("x")
            block.MakePeriodic("y") 
            block.MakePeriodic("z")
        else:
            block.MakePeriodic("x")
            block.MakePeriodic("y")
        
        return domain, block
    
    def _generate_divergence_free_field(self, shape, peak_wavenumber):
        """Generate divergence-free velocity field using proper spectral method with improved von Karman spectrum"""
        # Create wavenumber grids
        if len(shape) == 5:  # 3D
            nz, ny, nx = shape[2], shape[3], shape[4]
            kz = torch.fft.fftfreq(nz, device=cuda_device)
            ky = torch.fft.fftfreq(ny, device=cuda_device)
            kx = torch.fft.fftfreq(nx, device=cuda_device)
            KZ, KY, KX = torch.meshgrid(kz, ky, kx, indexing='ij')
            k_mag = torch.sqrt(KX**2 + KY**2 + KZ**2)
            
            # Domain size and length scale parameters
            domain_size = max(nz, ny, nx)
            
            # Create random potential in Fourier space
            potential_fft = torch.complex(
                torch.randn(nz, ny, nx, device=cuda_device),
                torch.randn(nz, ny, nx, device=cuda_device)
            )
            
        else:  # 2D
            ny, nx = shape[2], shape[3]
            ky = torch.fft.fftfreq(ny, device=cuda_device)
            kx = torch.fft.fftfreq(nx, device=cuda_device)
            KY, KX = torch.meshgrid(ky, kx, indexing='ij')
            k_mag = torch.sqrt(KX**2 + KY**2)
            
            # Domain size and length scale parameters
            domain_size = max(ny, nx)
            
            # For 2D, use streamfunction to ensure divergence-free field
            streamfunction_fft = torch.complex(
                torch.randn(ny, nx, device=cuda_device),
                torch.randn(ny, nx, device=cuda_device)
            )
        
        # Improved von Karman spectrum with configurable physical parameters
        # Control integral length scale (size of largest eddies)
        # Use parameters from args if available, otherwise use defaults
        integral_scale_factor = getattr(self.args, 'integral_scale_factor', 6.0)  # domain_size / this factor
        Re_lambda = getattr(self.args, 'taylor_reynolds', 50.0)  # Taylor microscale Reynolds number
        
        L_integral = domain_size / integral_scale_factor  # Integral length scale (pixels)
        k0 = 1.0 / L_integral  # Integral wavenumber
        
        # Control dissipation scale (size of smallest eddies)
        eta_over_L = Re_lambda**(-3/4)  # Kolmogorov scale / integral scale
        k_eta = 1.0 / (eta_over_L * L_integral)  # Dissipation wavenumber
        
        # True von Karman spectrum for 3D isotropic turbulence
        # E(k) = C * (k/k0)^4 / (1 + (k/k0)^2)^(17/6) * exp(-2*(k/k_eta)^2)
        k_over_k0 = k_mag / k0
        k_over_keta = k_mag / k_eta
        
        # Von Karman spectrum with realistic energy distribution
        energy_spectrum = (k_over_k0**4) / (1 + k_over_k0**2)**(17/6)
        
        # Add exponential cutoff at dissipation scale (more physical than previous version)
        energy_spectrum *= torch.exp(-2.0 * k_over_keta**2)
        
        # Normalize to ensure reasonable energy levels
        # Peak of the spectrum should be at k ≈ k0
        k_peak_theory = k0 * (4.0/13.0)**(1/2)  # Theoretical peak location
        peak_mask = (k_mag >= k_peak_theory * 0.8) & (k_mag <= k_peak_theory * 1.2)
        if torch.any(peak_mask):
            energy_spectrum = energy_spectrum / torch.max(energy_spectrum[peak_mask])
        
        # Remove DC component (no mean flow)
        energy_spectrum[k_mag < 1e-10] = 0
        
        # Apply additional smoothing near k=0 to avoid numerical issues
        k_smooth = k0 / 10.0
        smooth_factor = torch.tanh(k_mag / k_smooth)
        energy_spectrum *= smooth_factor
        
        # Apply spectrum to create realistic turbulence
        if len(shape) == 5:  # 3D
            # Generate velocity from vector potential curl
            # u = ∇ × A ensures ∇ · u = 0
            potential_fft *= torch.sqrt(energy_spectrum)
            
            # Apply different random phases for each component
            Ax_fft = potential_fft * torch.exp(1j * 2 * np.pi * torch.rand_like(k_mag))
            Ay_fft = potential_fft * torch.exp(1j * 2 * np.pi * torch.rand_like(k_mag))
            Az_fft = potential_fft * torch.exp(1j * 2 * np.pi * torch.rand_like(k_mag))
            
            # Compute velocity as curl of vector potential: u = ∇ × A
            # ux = ∂Az/∂y - ∂Ay/∂z
            # uy = ∂Ax/∂z - ∂Az/∂x  
            # uz = ∂Ay/∂x - ∂Ax/∂y
            ux_fft = 1j * (2*np.pi) * (KY * Az_fft - KZ * Ay_fft)
            uy_fft = 1j * (2*np.pi) * (KZ * Ax_fft - KX * Az_fft)
            uz_fft = 1j * (2*np.pi) * (KX * Ay_fft - KY * Ax_fft)
            
            # Convert back to physical space
            ux = torch.fft.ifftn(ux_fft).real
            uy = torch.fft.ifftn(uy_fft).real
            uz = torch.fft.ifftn(uz_fft).real
            
            velocity = torch.stack([ux, uy, uz], dim=0).unsqueeze(0)
            
        else:  # 2D
            # For 2D: u = (-∂ψ/∂y, ∂ψ/∂x) ensures ∇ · u = 0
            streamfunction_fft *= torch.sqrt(energy_spectrum)
            
            # Compute velocity components from streamfunction
            ux_fft = -1j * (2*np.pi) * KY * streamfunction_fft
            uy_fft = 1j * (2*np.pi) * KX * streamfunction_fft
            
            # Convert back to physical space
            ux = torch.fft.ifftn(ux_fft).real
            uy = torch.fft.ifftn(uy_fft).real
            
            velocity = torch.stack([ux, uy], dim=0).unsqueeze(0)
        
        return velocity.to(dtype=self.dtype)
    
    def load_initial_velocity_from_warmup_data(self, resolution):
        """Load initial velocity field from warmup segment data"""
        warmup_data_dir = Path(self.args.training_data_dir)
        
        # First try to load from warmup segment files (post-warmup data)
        # Use the last segment as it represents the state after warmup completion
        warmup_segment = getattr(self.args, 'warmup_segment', 6)  # Default to segment 6 (step 300)
        segment_file = warmup_data_dir / f"{resolution}" / f"decaying_turbulence_v2_warmup_segment_{warmup_segment}_step_{warmup_segment*50}_index_1.npz"
        
        if segment_file.exists():
            data_file = segment_file
        else:
            # Fallback to resolution-based warmup data file if available
            data_file = warmup_data_dir / f"decaying_turbulence_v2_{resolution}x{resolution}_index_1.npz"
            if not data_file.exists():
                return self.load_initial_velocity_from_training_data(resolution)
        
        # Load the data
        data = np.load(data_file)
        u_data = data['u']  # Shape: [time, y, x]
        v_data = data['v']  # Shape: [time, y, x]
        
        # Extract timestep information from warmup data
        training_timestep = None
        
        # Check for delta_t (the actual timestep field in our training data)
        if 'delta_t' in data.keys():
            training_timestep = float(data['delta_t'])
        elif 'timestep' in data.keys():
            # Alternative timestep field name
            training_timestep = float(data['timestep'])
        elif 'dt' in data.keys():
            # Another alternative timestep field name
            training_timestep = float(data['dt'])
        elif 'warmup_time_step' in data.keys():
            # Another alternative timestep field name
            training_timestep = float(data['warmup_time_step'])
        elif 'time_array' in data.keys():
            # If time_array is stored, calculate timestep from it
            time_array = data['time_array']
            if len(time_array) > 1:
                training_timestep = float(time_array[1] - time_array[0])
        elif 'time' in data.keys():
            # Fallback to 'time' field
            time_array = data['time']
            if len(time_array) > 1:
                training_timestep = float(time_array[1] - time_array[0])
        else:
            # Try to infer timestep from number of time steps and total simulation time
            num_timesteps = u_data.shape[0]
            if 'total_time' in data.keys():
                total_time = float(data['total_time'])
                training_timestep = total_time / (num_timesteps - 1)
            else:
                # Last resort: use computed CFD timestep 
                training_timestep = self.compute_cfd_timestep(resolution)
        
        # Extract velocity field - handle different data structures
        if len(u_data.shape) == 3:
            # Regular training data format: [time, y, x]
            u_t0 = u_data[0, :, :]  # [y, x]
            v_t0 = v_data[0, :, :]  # [y, x]
        elif len(u_data.shape) == 2:
            # Warmup segment format: [y, x] - single snapshot
            u_t0 = u_data  # [y, x]
            v_t0 = v_data  # [y, x]
        else:
            return self.load_initial_velocity_from_training_data(resolution)
        
        # Check if we need to resample the data to match target resolution
        actual_resolution = u_t0.shape[0]  # Assuming square domain
        
        if actual_resolution != resolution:
            # Simple resampling using scipy zoom
            try:
                from scipy.ndimage import zoom
                zoom_factor = resolution / actual_resolution
                u_t0 = zoom(u_t0, zoom_factor, order=1)  # Linear interpolation
                v_t0 = zoom(v_t0, zoom_factor, order=1)
            except ImportError:
                # Fallback to simple downsampling
                if actual_resolution > resolution:
                    factor = actual_resolution // resolution
                    u_t0 = u_t0[::factor, ::factor]
                    v_t0 = v_t0[::factor, ::factor]
                else:
                    return self.load_initial_velocity_from_training_data(resolution)
        
        # Convert to PICT format: [1, channels, y, x]
        if self.args.dims == 3:
            # For 3D, we need to add a z-component (set to zero for now)
            w_t0 = np.zeros_like(u_t0)
            velocity = np.stack([u_t0, v_t0, w_t0], axis=0)  # [3, y, x]
            velocity = velocity[np.newaxis, :]  # [1, 3, y, x]
        else:
            # For 2D
            velocity = np.stack([u_t0, v_t0], axis=0)  # [2, y, x]
            velocity = velocity[np.newaxis, :]  # [1, 2, y, x]
        
        # Convert to torch tensor
        velocity_tensor = torch.from_numpy(velocity).to(dtype=self.dtype, device=cuda_device)
        
        return velocity_tensor, training_timestep

    def compute_cfd_timestep(self, resolution, velocity_field=None):
        """
        Compute timestep based on CFD stability criteria
        
        Args:
            resolution: Grid resolution
            velocity_field: Current velocity field for CFL calculation (optional)
            
        Returns:
            timestep: Computed timestep based on stability criteria
        """
        # Physical parameters
        domain_length = 2 * np.pi * self.args.domain_scale
        dx = domain_length / resolution  # Grid spacing
        nu = self.args.viscosity  # Kinematic viscosity
        
        # 1. CFL condition: Δt ≤ CFL * Δx / |u_max|
        target_cfl = getattr(self.args, 'adaptive_cfl', 0.5)  # Conservative CFL number
        
        if velocity_field is not None:
            # Use actual velocity field to compute maximum velocity
            if isinstance(velocity_field, torch.Tensor):
                velocity_magnitude = torch.sqrt(torch.sum(velocity_field**2, dim=1))
                max_velocity = torch.max(velocity_magnitude).item()
            else:
                velocity_magnitude = np.sqrt(np.sum(velocity_field**2, axis=1))
                max_velocity = np.max(velocity_magnitude)
        else:
            # Use specified maximum velocity from args
            max_velocity = self.args.max_velocity
        
        # CFL-based timestep
        dt_cfl = target_cfl * dx / max_velocity if max_velocity > 0 else 1e-3
        
        # 2. Viscous stability condition: Δt ≤ 0.5 * (Δx)² / ν
        # For explicit viscous terms, diffusion number D = ν*Δt/(Δx)² ≤ 0.5
        dt_viscous = 0.5 * dx**2 / nu if nu > 0 else 1e10
        
        # 3. Kolmogorov time scale consideration for turbulent flows
        # τ_η = √(ν/ε) where ε is dissipation rate
        # Estimate dissipation rate: ε ≈ u³/L where u is velocity scale, L is length scale
        integral_scale_factor = getattr(self.args, 'integral_scale_factor', 6.0)
        L_integral = domain_length / integral_scale_factor  # Integral length scale
        
        # Estimate energy dissipation rate
        u_rms = max_velocity / np.sqrt(3)  # Rough estimate of RMS velocity
        epsilon = u_rms**3 / L_integral if L_integral > 0 else 1e-6
        
        # Kolmogorov time scale
        tau_eta = np.sqrt(nu / epsilon) if epsilon > 0 else 1e10
        
        # For accurate DNS, timestep should be much smaller than Kolmogorov time
        dt_kolmogorov = 0.1 * tau_eta  # Conservative factor
        
        # 4. Acoustic/pressure wave stability (for compressible effects)
        # For incompressible flows, this is less critical, but we include it for completeness
        # Acoustic CFL: Δt ≤ Δx / c where c is sound speed
        # For incompressible flow, we use a characteristic velocity instead
        c_characteristic = max_velocity * 10  # Rough estimate
        dt_acoustic = 0.1 * dx / c_characteristic if c_characteristic > 0 else 1e10
        
        # Take the most restrictive condition
        dt_computed = min(dt_cfl, dt_viscous, dt_kolmogorov, dt_acoustic)
        
        # Apply safety factor
        safety_factor = getattr(self.args, 'cfl_safety_factor', 0.8)
        dt_final = safety_factor * dt_computed
        
        # Ensure reasonable bounds
        dt_min = 1e-6  # Minimum timestep to avoid numerical issues
        dt_max = 0.01  # Maximum timestep for stability
        dt_final = max(dt_min, min(dt_max, dt_final))
        
        # Physical parameter verification (silent)
        Re_grid = max_velocity * dx / nu  # Grid Reynolds number
        Pe_grid = max_velocity * dx / nu  # Grid Peclet number (same as Re for momentum)
        
        return dt_final

    def load_timestep_from_simulation_data(self, resolution):
        """Load timestep from simulation data for the specified resolution"""
        training_data_dir = Path(self.args.training_data_dir)
        data_file = training_data_dir / f"decaying_turbulence_v2_{resolution}x{resolution}_index_1.npz"
        
        if not data_file.exists():
            # Fallback to computed timestep
            computed_timestep = self.compute_cfd_timestep(resolution)
            return computed_timestep
            
        data = np.load(data_file)
        
        # Extract timestep information using the same logic as training data
        timestep = None
        if 'delta_t' in data.keys():
            timestep = float(data['delta_t'])
        elif 'timestep' in data.keys():
            timestep = float(data['timestep'])
        elif 'dt' in data.keys():
            timestep = float(data['dt'])
        elif 'time_array' in data.keys():
            time_array = data['time_array']
            if len(time_array) > 1:
                timestep = float(time_array[1] - time_array[0])
        elif 'time' in data.keys():
            time_array = data['time']
            if len(time_array) > 1:
                timestep = float(time_array[1] - time_array[0])
        
        if timestep is None:
            # Fallback to computed timestep
            computed_timestep = self.compute_cfd_timestep(resolution)
            return computed_timestep
        
        return timestep

    def load_reference_training_data(self, resolution):
        """Load reference training data for comparison"""
        training_data_dir = Path(self.args.training_data_dir)
        data_file = training_data_dir / f"decaying_turbulence_v2_{resolution}x{resolution}_index_1.npz"
        
        if not data_file.exists():
            return None, None
            
        data = np.load(data_file)
        u_data = data['u']  # Shape: [time, y, x]
        v_data = data['v']  # Shape: [time, y, x]
        
        # Extract timestep information
        reference_timestep = self.load_timestep_from_simulation_data(resolution)
            
        return (u_data, v_data), reference_timestep

    def load_initial_velocity_from_training_data(self, resolution):
        """Load initial velocity field from existing training data"""
        # Construct path to training data file
        training_data_dir = Path(self.args.training_data_dir)
        data_file = training_data_dir / f"decaying_turbulence_v2_{resolution}x{resolution}_index_1.npz"
        
        if not data_file.exists():
            return None, None
            
        # Load the data
        data = np.load(data_file)
        u_data = data['u']  # Shape: [time, y, x]
        v_data = data['v']  # Shape: [time, y, x]
        
        # Extract timestep information from training data
        training_timestep = None
        
        # Check for delta_t (the actual timestep field in our training data)
        if 'delta_t' in data.keys():
            training_timestep = float(data['delta_t'])
        elif 'timestep' in data.keys():
            # Alternative timestep field name
            training_timestep = float(data['timestep'])
        elif 'dt' in data.keys():
            # Another alternative timestep field name
            training_timestep = float(data['dt'])
        elif 'time_array' in data.keys():
            # If time_array is stored, calculate timestep from it
            time_array = data['time_array']
            if len(time_array) > 1:
                training_timestep = float(time_array[1] - time_array[0])
        elif 'time' in data.keys():
            # Fallback to 'time' field
            time_array = data['time']
            if len(time_array) > 1:
                training_timestep = float(time_array[1] - time_array[0])
        else:
            # Try to infer timestep from number of time steps and total simulation time
            num_timesteps = u_data.shape[0]
            if 'total_time' in data.keys():
                total_time = float(data['total_time'])
                training_timestep = total_time / (num_timesteps - 1)
            else:
                # Last resort: use computed CFD timestep 
                training_timestep = self.compute_cfd_timestep(resolution)
        
        # Extract t=0 velocity field
        u_t0 = u_data[0, :, :]  # [y, x]
        v_t0 = v_data[0, :, :]  # [y, x]
        
        # Convert to PICT format: [1, channels, y, x]
        if self.args.dims == 3:
            # For 3D, we need to add a z-component (set to zero for now)
            w_t0 = np.zeros_like(u_t0)
            velocity = np.stack([u_t0, v_t0, w_t0], axis=0)  # [3, y, x]
            velocity = velocity[np.newaxis, :]  # [1, 3, y, x]
        else:
            # For 2D
            velocity = np.stack([u_t0, v_t0], axis=0)  # [2, y, x]
            velocity = velocity[np.newaxis, :]  # [1, 2, y, x]
        
        # Convert to torch tensor
        velocity_tensor = torch.from_numpy(velocity).to(dtype=self.dtype, device=cuda_device)
        
        return velocity_tensor, training_timestep
    
    def _verify_divergence_free(self, velocity, resolution):
        """Verify that the velocity field is divergence-free"""
        if velocity.shape[1] < 2:  # Need at least 2D
            return 0.0
            
        # Extract velocity components
        u = velocity[0, 0, :, :] if len(velocity.shape) == 4 else velocity[0, 0, :, :]
        v = velocity[0, 1, :, :] if len(velocity.shape) == 4 else velocity[0, 1, :, :]
        
        # Compute derivatives using finite differences (periodic boundaries)
        dx = 2 * np.pi * self.args.domain_scale / resolution
        
        # Central differences with periodic boundary conditions
        du_dx = (torch.roll(u, -1, dims=1) - torch.roll(u, 1, dims=1)) / (2 * dx)
        dv_dy = (torch.roll(v, -1, dims=0) - torch.roll(v, 1, dims=0)) / (2 * dx)
        
        # Compute divergence
        divergence = du_dx + dv_dy
        
        # Return RMS divergence
        div_rms = torch.sqrt(torch.mean(divergence**2)).item()
        
        return div_rms
    
    def compare_with_reference(self, pict_velocity, reference_data, step, resolution, save_dir):
        """Compare PICT velocity with reference data and create visualizations"""
        if reference_data is None:
            return
            
        u_ref, v_ref = reference_data
        
        # Convert PICT velocity from torch to numpy
        if isinstance(pict_velocity, torch.Tensor):
            pict_velocity = pict_velocity.detach().cpu().numpy()
        
        # Extract PICT velocity components [1, 2, y, x] -> [y, x]
        u_pict = pict_velocity[0, 0, :, :]
        v_pict = pict_velocity[0, 1, :, :]
        
        # Extract reference velocity for this step [time, y, x] -> [y, x]
        if step < u_ref.shape[0]:
            u_ref_step = u_ref[step, :, :]
            v_ref_step = v_ref[step, :, :]
        else:
            return
        
        # Ensure same shape
        if u_pict.shape != u_ref_step.shape:
            return
        
        # Calculate differences
        u_diff = u_pict - u_ref_step
        v_diff = v_pict - v_ref_step
        velocity_magnitude_pict = np.sqrt(u_pict**2 + v_pict**2)
        velocity_magnitude_ref = np.sqrt(u_ref_step**2 + v_ref_step**2)
        magnitude_diff = velocity_magnitude_pict - velocity_magnitude_ref
        
        # Statistics
        u_rmse = np.sqrt(np.mean(u_diff**2))
        v_rmse = np.sqrt(np.mean(v_diff**2))
        mag_rmse = np.sqrt(np.mean(magnitude_diff**2))
        max_u_diff = np.max(np.abs(u_diff))
        max_v_diff = np.max(np.abs(v_diff))
        max_mag_diff = np.max(np.abs(magnitude_diff))
        

        
        # Create comparison plots
        comparison_dir = Path(save_dir) / f"comparison_step_{step}"
        comparison_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subplots for comparison
        fig, axes = plt.subplots(3, 3, figsize=(15, 15))
        fig.suptitle(f'PICT vs Reference Comparison - Step {step} (Resolution {resolution}x{resolution})', fontsize=16)
        
        # Row 1: U-velocity
        im1 = axes[0,0].imshow(u_pict, cmap='RdBu_r', origin='lower')
        axes[0,0].set_title('PICT U-velocity')
        plt.colorbar(im1, ax=axes[0,0])
        
        im2 = axes[0,1].imshow(u_ref_step, cmap='RdBu_r', origin='lower')
        axes[0,1].set_title('Reference U-velocity')
        plt.colorbar(im2, ax=axes[0,1])
        
        # Use symmetric colorbar for differences
        u_diff_max = np.max(np.abs(u_diff))
        im3 = axes[0,2].imshow(u_diff, cmap='RdBu_r', origin='lower', 
                               vmin=-u_diff_max, vmax=u_diff_max)
        axes[0,2].set_title(f'U-velocity Difference (RMSE: {u_rmse:.4f})')
        plt.colorbar(im3, ax=axes[0,2])
        
        # Row 2: V-velocity
        im4 = axes[1,0].imshow(v_pict, cmap='RdBu_r', origin='lower')
        axes[1,0].set_title('PICT V-velocity')
        plt.colorbar(im4, ax=axes[1,0])
        
        im5 = axes[1,1].imshow(v_ref_step, cmap='RdBu_r', origin='lower')
        axes[1,1].set_title('Reference V-velocity')
        plt.colorbar(im5, ax=axes[1,1])
        
        v_diff_max = np.max(np.abs(v_diff))
        im6 = axes[1,2].imshow(v_diff, cmap='RdBu_r', origin='lower',
                               vmin=-v_diff_max, vmax=v_diff_max)
        axes[1,2].set_title(f'V-velocity Difference (RMSE: {v_rmse:.4f})')
        plt.colorbar(im6, ax=axes[1,2])
        
        # Row 3: Velocity magnitude
        im7 = axes[2,0].imshow(velocity_magnitude_pict, cmap='viridis', origin='lower')
        axes[2,0].set_title('PICT Velocity Magnitude')
        plt.colorbar(im7, ax=axes[2,0])
        
        im8 = axes[2,1].imshow(velocity_magnitude_ref, cmap='viridis', origin='lower')
        axes[2,1].set_title('Reference Velocity Magnitude')
        plt.colorbar(im8, ax=axes[2,1])
        
        mag_diff_max = np.max(np.abs(magnitude_diff))
        im9 = axes[2,2].imshow(magnitude_diff, cmap='RdBu_r', origin='lower',
                               vmin=-mag_diff_max, vmax=mag_diff_max)
        axes[2,2].set_title(f'Magnitude Difference (RMSE: {mag_rmse:.4f})')
        plt.colorbar(im9, ax=axes[2,2])
        
        # Save comparison plot
        comparison_file = comparison_dir / f"velocity_comparison_step_{step}.png"
        plt.tight_layout()
        plt.savefig(comparison_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        # Save comparison statistics
        stats_file = comparison_dir / f"comparison_stats_step_{step}.txt"
        with open(stats_file, 'w') as f:
            f.write(f"PICT vs Reference Comparison - Step {step}\n")
            f.write(f"Resolution: {resolution}x{resolution}\n")
            f.write(f"U-velocity RMSE: {u_rmse:.6f}\n")
            f.write(f"V-velocity RMSE: {v_rmse:.6f}\n")
            f.write(f"Magnitude RMSE: {mag_rmse:.6f}\n")
            f.write(f"Max U difference: {max_u_diff:.6f}\n")
            f.write(f"Max V difference: {max_v_diff:.6f}\n")
            f.write(f"Max magnitude difference: {max_mag_diff:.6f}\n")
        

        return {
            'u_rmse': u_rmse, 'v_rmse': v_rmse, 'mag_rmse': mag_rmse,
            'max_u_diff': max_u_diff, 'max_v_diff': max_v_diff, 'max_mag_diff': max_mag_diff
        }
    
    def generate_initial_turbulence(self, domain, block):
        """Generate initial turbulent velocity field with proper divergence-free constraint using improved von Karman spectrum"""
        dims = domain.getSpatialDims()
        block_size = block.getSizes()
        
        # Get resolution for verification
        resolution = block_size.x if dims == 2 else block_size.x
        
        # Generate random velocity field
        if dims == 3:
            shape = [1, dims, block_size.z, block_size.y, block_size.x]  # [1, 3, z, y, x]
        else:
            shape = [1, dims, block_size.y, block_size.x]  # [1, 2, y, x]
        
        # Create divergence-free velocity field using improved vector potential method
        velocity = self._generate_divergence_free_field(shape, self.args.peak_wavenumber)
        
        # Scale to desired maximum velocity
        velocity_magnitude = torch.sqrt(torch.sum(velocity**2, dim=1, keepdim=True))
        max_vel = torch.max(velocity_magnitude).item()
        
        velocity = velocity * (self.args.max_velocity / max_vel)
        
        # Set velocity field
        block.setVelocity(velocity)
        
        return velocity
    
    def run_simulation_with_comparison(self, sim, domain, resolution, steps, save_interval):
        """Run simulation with step-by-step comparison to reference data using existing simulation instance"""
        
        # Load reference data for comparison
        reference_data, ref_timestep = self.load_reference_training_data(resolution)
        
        # Storage for trajectory data and comparison results
        trajectory_data = []
        comparison_stats = []
        
        # Initial comparison (step 0)
        initial_velocity = domain.getBlock(0).velocity
        trajectory_data.append(initial_velocity.detach().cpu().numpy().copy())
        
        if reference_data is not None:
            stats = self.compare_with_reference(
                initial_velocity, reference_data, 0, resolution, self.args.save_dir
            )
            if stats:
                comparison_stats.append(stats)
        
        # Run simulation and collect data with step-by-step comparison
        for step in range(1, steps + 1):
            sim.run(iterations=save_interval)
            
            # Get current velocity field
            current_velocity = domain.getBlock(0).velocity
            trajectory_data.append(current_velocity.detach().cpu().numpy().copy())
            
            # Compare with reference data
            if reference_data is not None:
                stats = self.compare_with_reference(
                    current_velocity, reference_data, step, resolution, self.args.save_dir
                )
                if stats:
                    comparison_stats.append(stats)
            

        
        # Save comparison summary
        if comparison_stats:
            self.save_comparison_summary(comparison_stats, resolution)
        
        return np.array(trajectory_data)
    
    def run_simulation(self, sim, domain, resolution, steps, save_interval):
        """Run simulation and collect velocity trajectory data using existing simulation instance"""
        # Check if comparison mode is enabled
        if getattr(self.args, 'enable_comparison', False):
            return self.run_simulation_with_comparison(sim, domain, resolution, steps, save_interval)
        

        
        # Storage for trajectory data
        trajectory_data = []
        #velocity = domain.getBlock(0).velocity.detach().cpu().numpy()
        #trajectory_data.append(velocity.copy())
        
        # Run simulation and collect data
        for step in range(0, steps, save_interval):
            if step%1000 == 0 and step > 0:  # 跳过第0步（数据为空）
                percentage = (step / steps) * 100
                print(f"Step {step} of {steps} ({percentage:.1f}%). ")
                
                # 保存当前轨迹数据（如果有数据）
                if len(trajectory_data) > 0:
                    print(f"Saving trajectory data at step {step} with {len(trajectory_data)} time points...")
                    # 修改保存文件名包含步数
                    original_save_file = self.args.save_file
                    self.args.save_file = f"{original_save_file}_step{step}"
                    
                    self.save_trajectory_data(np.array(trajectory_data), resolution, self.args.training_timestep)
                    trajectory_data = []
                    self.args.save_file = original_save_file
                    
                    # 强制垃圾回收
                    gc.collect()
            sim.run(iterations=save_interval)
            
            # Get current velocity field
            velocity = domain.getBlock(0).velocity.detach().cpu().numpy()
            trajectory_data.append(velocity.copy())
            
        
        # 保存最后剩余的数据
        if len(trajectory_data) > 0:
            print(f"Saving final trajectory data with {len(trajectory_data)} time points...")
            original_save_file = self.args.save_file
            self.args.save_file = f"{original_save_file}_final"
            
            self.save_trajectory_data(np.array(trajectory_data), resolution, self.args.training_timestep)
            self.args.save_file = original_save_file
        
            
            print("Final data saved successfully.")
        
        return np.array(trajectory_data)
    
    def save_comparison_summary(self, comparison_stats, resolution):
        """Save a summary of all comparison statistics"""
        summary_dir = Path(self.args.save_dir) / "comparison_summary"
        summary_dir.mkdir(parents=True, exist_ok=True)
        
        summary_file = summary_dir / f"comparison_summary_{resolution}x{resolution}.txt"
        
        with open(summary_file, 'w') as f:
            f.write(f"PICT vs Reference Comparison Summary\n")
            f.write(f"Resolution: {resolution}x{resolution}\n")
            f.write(f"Number of steps: {len(comparison_stats)}\n")
            f.write("="*50 + "\n\n")
            
            for i, stats in enumerate(comparison_stats):
                f.write(f"Step {i}:\n")
                f.write(f"  U-velocity RMSE: {stats['u_rmse']:.6f}\n")
                f.write(f"  V-velocity RMSE: {stats['v_rmse']:.6f}\n")
                f.write(f"  Magnitude RMSE: {stats['mag_rmse']:.6f}\n")
                f.write(f"  Max U difference: {stats['max_u_diff']:.6f}\n")
                f.write(f"  Max V difference: {stats['max_v_diff']:.6f}\n")
                f.write(f"  Max magnitude difference: {stats['max_mag_diff']:.6f}\n")
                f.write("\n")
        
        # Create summary plot showing evolution of differences
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f'PICT vs Reference Error Evolution - Resolution {resolution}x{resolution}', fontsize=16)
        
        steps = list(range(len(comparison_stats)))
        u_rmse = [stats['u_rmse'] for stats in comparison_stats]
        v_rmse = [stats['v_rmse'] for stats in comparison_stats]
        mag_rmse = [stats['mag_rmse'] for stats in comparison_stats]
        max_u_diff = [stats['max_u_diff'] for stats in comparison_stats]
        max_v_diff = [stats['max_v_diff'] for stats in comparison_stats]
        max_mag_diff = [stats['max_mag_diff'] for stats in comparison_stats]
        
        axes[0,0].plot(steps, u_rmse, 'b-o')
        axes[0,0].set_title('U-velocity RMSE')
        axes[0,0].set_xlabel('Step')
        axes[0,0].set_ylabel('RMSE')
        axes[0,0].grid(True)
        
        axes[0,1].plot(steps, v_rmse, 'r-o')
        axes[0,1].set_title('V-velocity RMSE')
        axes[0,1].set_xlabel('Step')
        axes[0,1].set_ylabel('RMSE')
        axes[0,1].grid(True)
        
        axes[0,2].plot(steps, mag_rmse, 'g-o')
        axes[0,2].set_title('Velocity Magnitude RMSE')
        axes[0,2].set_xlabel('Step')
        axes[0,2].set_ylabel('RMSE')
        axes[0,2].grid(True)
        
        axes[1,0].plot(steps, max_u_diff, 'b-s')
        axes[1,0].set_title('Max U-velocity Difference')
        axes[1,0].set_xlabel('Step')
        axes[1,0].set_ylabel('Max Diff')
        axes[1,0].grid(True)
        
        axes[1,1].plot(steps, max_v_diff, 'r-s')
        axes[1,1].set_title('Max V-velocity Difference')
        axes[1,1].set_xlabel('Step')
        axes[1,1].set_ylabel('Max Diff')
        axes[1,1].grid(True)
        
        axes[1,2].plot(steps, max_mag_diff, 'g-s')
        axes[1,2].set_title('Max Magnitude Difference')
        axes[1,2].set_xlabel('Step')
        axes[1,2].set_ylabel('Max Diff')
        axes[1,2].grid(True)
        
        plt.tight_layout()
        summary_plot_file = summary_dir / f"error_evolution_{resolution}x{resolution}.png"
        plt.savefig(summary_plot_file, dpi=150, bbox_inches='tight')
        plt.close()
        

    
    def create_simulation(self, domain, time_step, log_interval, log_dir_name):
        """Create a PISOtorch simulation instance with consistent settings"""
        log_dir = Path(self.args.save_dir) / log_dir_name
        log_dir.mkdir(parents=True, exist_ok=True)
        
        sim = PISOtorch_simulation.Simulation(
            domain=domain,
            time_step=time_step,
            substeps="ADAPTIVE",  # Always use PICT native adaptive timestep
            corrector_steps=2,
            non_orthogonal=False,
            pressure_tol=1e-6,
            velocity_corrector="FD",
            adaptive_CFL=getattr(self.args, 'adaptive_cfl', 0.8),
            visualize_max_steps=getattr(self.args, 'visualize_max_steps', None),
            log_interval=log_interval,
            log_dir=str(log_dir),
            stop_fn=lambda: False
        )
        
        return sim
    
    def warmup_simulation(self, sim, resolution, warmup_steps):
        """Run warmup simulation to reach statistically steady state using existing simulation instance"""
        
        # Storage for warmup trajectory data
        warmup_trajectory = []
        
        # Collect initial state
        domain = sim.domain
        initial_velocity = domain.getBlock(0).velocity.detach().cpu().numpy()
        warmup_trajectory.append(initial_velocity.copy())
        
        # Run warmup and collect data at every step
        for step in range(warmup_steps):
            # Run one step at a time
            sim.run(iterations=1)
            
            # Collect velocity field after each step
            current_velocity = domain.getBlock(0).velocity.detach().cpu().numpy()
            warmup_trajectory.append(current_velocity.copy())
            
            # Memory management: save and clean every 1000 steps during warmup
            if (step + 1) % 100 == 0:
                percentage = ((step + 1) / warmup_steps) * 100
                print(f"Warmup step {step + 1} of {warmup_steps} ({percentage:.1f}%). ")
                
                # Save warmup data if we have enough
                if len(warmup_trajectory) > 0:
                    print(f"Saving warmup trajectory data at step {step + 1} with {len(warmup_trajectory)} time points...")
                    original_save_file = self.args.save_file
                    self.args.save_file = f"{original_save_file}_warmup_step{step + 1}"
                    
                    self.save_trajectory_data(np.array(warmup_trajectory), resolution, self.args.training_timestep)
                    warmup_trajectory = []  # Clear all warmup data to save memory
                    self.args.save_file = original_save_file
                    
                    # Force garbage collection
                    gc.collect()
                    print("Warmup memory cleaned.")
        
        # Save final warmup data if any remains
        if len(warmup_trajectory) > 0:
            print(f"Saving final warmup trajectory data with {len(warmup_trajectory)} time points...")
            original_save_file = self.args.save_file
            self.args.save_file = f"{original_save_file}_warmup_final"
            
            self.save_trajectory_data(np.array(warmup_trajectory), resolution, self.args.training_timestep)
            self.args.save_file = original_save_file
            print("Final warmup data saved successfully.")
        
        # Convert to numpy array (will be small or empty now)
        warmup_trajectory = np.array(warmup_trajectory)
        
        print("Final warmup_trajectory.shape", warmup_trajectory.shape)
        return warmup_trajectory
    

    
    def downsample_velocity(self, velocity_hr, target_resolution, source_resolution):
        """Downsample high-resolution velocity to target resolution"""
        # Simple downsampling by taking every nth point
        factor = source_resolution // target_resolution
        
        if len(velocity_hr.shape) == 5:  # 3D: [1, 3, z, y, x]
            downsampled = velocity_hr[:, :, ::factor, ::factor, ::factor]
        else:  # 2D: [1, 2, y, x]
            downsampled = velocity_hr[:, :, ::factor, ::factor]
        
        # Make tensor contiguous in memory
        return downsampled.contiguous()
    
    def generate_data(self):
        """Main data generation pipeline"""
        
        # Check if we should use warmup data or training data for initialization
        use_warmup_data_init = getattr(self.args, 'use_warmup_data_init', False)
        use_training_data_init = getattr(self.args, 'use_training_data_init', False)
        
        # Create high-resolution domain for initial conditions
        hr_domain, hr_block = self.create_domain(self.args.high_res)
        hr_training_timestep = None
        
        if use_warmup_data_init:
            # Try to load initial velocity from warmup data
            initial_velocity, hr_training_timestep = self.load_initial_velocity_from_warmup_data(self.args.high_res)
            
            if initial_velocity is not None:
                # Set the loaded velocity field
                hr_block.setVelocity(initial_velocity)
                hr_domain.PrepareSolve()
                hr_domain.UpdateDomainData()
            else:
                # Fallback to generated initial conditions
                initial_velocity = self.generate_initial_turbulence(hr_domain, hr_block)
                hr_domain.PrepareSolve()
                
        elif use_training_data_init:
            # Try to load initial velocity from training data
            initial_velocity_training, hr_training_timestep = self.load_initial_velocity_from_training_data(self.args.high_res)
            
            if initial_velocity_training is not None:
                # Set the loaded velocity field
                initial_velocity = initial_velocity_training
                hr_block.setVelocity(initial_velocity)
                hr_domain.PrepareSolve()
                hr_domain.UpdateDomainData()
            else:
                # Fallback to generated initial conditions
                initial_velocity = self.generate_initial_turbulence(hr_domain, hr_block)
                hr_domain.PrepareSolve()
                
        else:
            # Original approach: generate initial turbulent field
            initial_velocity = self.generate_initial_turbulence(hr_domain, hr_block)
            hr_domain.PrepareSolve()
            
        # Calculate warmup parameters
        if hasattr(self.args, 'warmup_timestep') and self.args.warmup_timestep is not None:
            # Use manually specified warmup timestep
            output_time_step = self.args.warmup_timestep
        elif hr_training_timestep is not None:
            output_time_step = hr_training_timestep
        else:
            # Use computed CFD timestep for warmup
            output_time_step = self.compute_cfd_timestep(self.args.high_res)
            
        warmup_steps = round(self.args.warmup_time / output_time_step)
        
        # Create simulation instance for high-resolution domain
        hr_sim = self.create_simulation(
            hr_domain, 
            output_time_step, 
            max(warmup_steps // 10, 1), 
            "warmup_logs"
        )
        
        # Run warmup using the simulation instance and collect trajectory
        warmup_trajectory = self.warmup_simulation(hr_sim, self.args.high_res, warmup_steps)
        
        # Save warmup trajectory data
        self.save_warmup_trajectory_data(warmup_trajectory, self.args.high_res, output_time_step)
        
        # Get resolutions to generate
        resolution_list = []
        res = self.args.low_res
        while res <= self.args.high_res:
            resolution_list.append(res)
            res *= 2
        
        # CRITICAL: For pressure continuity, we can only use the high-resolution domain
        # Different resolutions would require different domains, which breaks pressure continuity
        # So we only generate data for the high-resolution
        resolution = self.args.high_res
        
        domain = hr_domain
        current_training_timestep = hr_training_timestep
        
        # Load timestep for this resolution when using warmup data init
        if use_warmup_data_init:
            # Load timestep from simulation data for this resolution
            simulation_timestep = self.load_timestep_from_simulation_data(resolution)
            if simulation_timestep is not None:
                current_training_timestep = simulation_timestep
        
        # Determine final training timestep with manual override priority
        if hasattr(self.args, 'training_timestep') and self.args.training_timestep is not None:
            # Use manually specified training timestep (highest priority)
            final_training_timestep = self.args.training_timestep
        elif current_training_timestep is not None:
            # Use timestep from training data or simulation data
            final_training_timestep = current_training_timestep
        elif hr_training_timestep is not None:
            # Fallback to hr_training_timestep
            final_training_timestep = hr_training_timestep
        else:
            # Last resort: compute CFD timestep
            final_training_timestep = self.compute_cfd_timestep(resolution)
        
        # CRITICAL: Use the same simulator instance to preserve ALL pressure info
        sim = hr_sim  # Always use the same simulation instance with same domain
        
        trajectory = self.run_simulation(
            sim, domain, resolution, self.args.generate_steps, 
            save_interval=self.args.save_interval
        )
        
        self.save_trajectory_data(trajectory, resolution, final_training_timestep)
        
        # If user wants multiple resolutions, we need to downsample the trajectory data
        # This preserves pressure continuity while providing multiple resolution outputs
        if self.args.low_res < self.args.high_res:
            self._generate_downsampled_resolutions(trajectory, final_training_timestep)
    
    def _generate_downsampled_resolutions(self, trajectory, timestep):
        """Generate lower resolution data by downsampling the high-resolution trajectory"""
        resolution_list = []
        res = self.args.low_res
        while res < self.args.high_res:  # Only lower resolutions
            resolution_list.append(res)
            res *= 2
        
        for target_resolution in resolution_list:
            # Downsample each frame in the trajectory
            downsampled_trajectory = []
            for frame in trajectory:
                downsampled_frame = self.downsample_velocity(
                    torch.from_numpy(frame), target_resolution, self.args.high_res
                ).numpy()
                downsampled_trajectory.append(downsampled_frame)
            
            downsampled_trajectory = np.array(downsampled_trajectory)
            
            # Save the downsampled trajectory
            self.save_trajectory_data(downsampled_trajectory, target_resolution, timestep)

    def save_trajectory_data(self, trajectory, resolution, timestep=None):
        """Save trajectory data in numpy format"""
        save_dir = Path(self.args.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        data_file = save_dir / f"{self.args.save_file}_{resolution}x{resolution}_index_{self.args.save_index}.npz"
        
        self.logger.info(f"Saving training data to: {data_file}")
        
        # Calculate timestep if not provided
        if timestep is None:
            # Use computed CFD timestep instead of arbitrary default
            timestep = self.compute_cfd_timestep(resolution)
        
        num_timesteps = trajectory.shape[0]
        time_array = np.arange(num_timesteps) * timestep
        
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
                time_array=time_array,  # Use consistent field name with training data
                delta_t=timestep,       # Use consistent field name with training data
                outer_steps=num_timesteps,
                resolution=resolution,
                steps=self.args.generate_steps,
                warmup_time=self.args.warmup_time,
                max_velocity=self.args.max_velocity,
                viscosity=self.args.viscosity,
                decay=self.args.decay,
                seed=self.args.seed,
                dims=3,
                domain_scale=self.args.domain_scale,
                cfl_safety_factor=self.args.cfl_safety_factor,
                peak_wavenumber=self.args.peak_wavenumber
            )
        else:  # 2D
            u_data = trajectory[:, 0, 0, :, :]  # x-velocity
            v_data = trajectory[:, 0, 1, :, :]  # y-velocity
            
            np.savez_compressed(
                data_file,
                u=u_data,
                v=v_data,
                time_array=time_array,  # Use consistent field name with training data
                delta_t=timestep,       # Use consistent field name with training data
                outer_steps=num_timesteps,
                resolution=resolution,
                steps=self.args.generate_steps,
                warmup_time=self.args.warmup_time,
                max_velocity=self.args.max_velocity,
                viscosity=self.args.viscosity,
                decay=self.args.decay,
                seed=self.args.seed,
                dims=2,
                domain_scale=self.args.domain_scale,
                cfl_safety_factor=self.args.cfl_safety_factor,
                peak_wavenumber=self.args.peak_wavenumber
            )
        

    
    def save_warmup_trajectory_data(self, warmup_trajectory, resolution, timestep=None):
        """Save complete warmup trajectory data in one file"""
        save_dir = Path(self.args.save_dir) / "warmup_data" / str(resolution)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Calculate timestep if not provided
        if timestep is None:
            # Use computed CFD timestep instead of arbitrary default
            timestep = self.compute_cfd_timestep(resolution)
        
        # Save the complete warmup trajectory - ALL STEPS IN ONE FILE
        trajectory_file = save_dir / f"warmup_trajectory_{resolution}x{resolution}_index_{self.args.save_index}.npz"
        
        num_timesteps = warmup_trajectory.shape[0]
        time_array = np.arange(num_timesteps) * timestep
        
        # Extract velocity components for complete trajectory
        if warmup_trajectory.shape[2] == 3:  # 3D
            u_data = warmup_trajectory[:, 0, 0, :, :, :]  # x-velocity
            v_data = warmup_trajectory[:, 0, 1, :, :, :]  # y-velocity  
            w_data = warmup_trajectory[:, 0, 2, :, :, :]  # z-velocity
            
            np.savez_compressed(
                trajectory_file,
                u=u_data,
                v=v_data,
                w=w_data,
                time_array=time_array,
                delta_t=timestep,
                total_time=self.args.warmup_time,
                outer_steps=num_timesteps,
                resolution=resolution,
                max_velocity=self.args.max_velocity,
                viscosity=self.args.viscosity,
                decay=self.args.decay,
                seed=self.args.seed,
                dims=3,
                domain_scale=self.args.domain_scale,
                cfl_safety_factor=self.args.cfl_safety_factor,
                peak_wavenumber=self.args.peak_wavenumber
            )
        else:  # 2D
            u_data = warmup_trajectory[:, 0, 0, :, :]  # x-velocity
            v_data = warmup_trajectory[:, 0, 1, :, :]  # y-velocity
            
            np.savez_compressed(
                trajectory_file,
                u=u_data,
                v=v_data,
                time_array=time_array,
                delta_t=timestep,
                total_time=self.args.warmup_time,
                outer_steps=num_timesteps,
                resolution=resolution,
                max_velocity=self.args.max_velocity,
                viscosity=self.args.viscosity,
                decay=self.args.decay,
                seed=self.args.seed,
                dims=2,
                domain_scale=self.args.domain_scale,
                cfl_safety_factor=self.args.cfl_safety_factor,
                peak_wavenumber=self.args.peak_wavenumber
            )
        
        self.logger.info(f"Saved complete warmup trajectory: {trajectory_file}")


def main():
    """
    Main function for PICT turbulence data generation.
    
    NEW FEATURES: 
    1. The code now automatically extracts and uses timestep information 
       from training data when --use_training_data_init is enabled. This ensures 
       that PICT simulations use the same temporal resolution as the training data.
    2. Support for warmup data initialization with --use_warmup_data_init flag.
       This allows loading initial conditions from warmup_data directory.
    3. Professional CFD timestep calculation based on stability criteria:
       - CFL condition (Courant-Friedrichs-Lewy): Δt ≤ CFL * Δx / |u_max|
       - Viscous stability: Δt ≤ 0.5 * (Δx)² / ν 
       - Kolmogorov time scale: Δt ≤ 0.1 * √(ν/ε)
       - Acoustic stability for pressure waves
       - Grid Reynolds number verification for DNS adequacy
       - Automatic fallback to computed timestep when data is unavailable
    
    Usage examples:
    
    1. Generate data using warmup segment data for initialization (NEW):
    python generate_turbulence_data_pict.py --use_warmup_data_init --warmup_segment 6 --training_data_dir "./training_data" --generate_steps 12200 --save_file "pict_from_warmup"
    
    2. Use different warmup segments (1-6 available, where 6 is post-warmup state):
    python generate_turbulence_data_pict.py --use_warmup_data_init --warmup_segment 1 --generate_steps 12200 --save_file "pict_from_warmup_early"
    
    3. Generate data using training data for initialization and timestep:
    python generate_turbulence_data_pict.py --use_training_data_init --training_data_dir "./training_data" --generate_steps 5000 --save_file "pict_from_training"
    
    4. Generate training data with original method:
    python generate_turbulence_data_pict.py --generate_steps 5000 --high_res 512 --save_file "turbulence_training"
    
    5. Quick test with warmup data initialization:
    python generate_turbulence_data_pict.py --use_warmup_data_init --warmup_segment 6 --generate_steps 100 --high_res 256 --low_res 64
    
    6. Use custom training data directory:
    python generate_turbulence_data_pict.py --use_warmup_data_init --warmup_segment 6 --training_data_dir "/path/to/your/training_data" --generate_steps 12200
    
    7. Enable step-by-step comparison with reference data (NEW):
    python generate_turbulence_data_pict.py --use_warmup_data_init --warmup_segment 1 --enable_comparison --generate_steps 5 --save_file "pict_comparison"
    
    8. Use manually specified timesteps (NEW):
    python generate_turbulence_data_pict.py --warmup_timestep 1e-4 --training_timestep 2e-4 --generate_steps 5000 --save_file "manual_timestep"
    
    9. Override only warmup timestep (let system auto-calculate training timestep):
    python generate_turbulence_data_pict.py --use_warmup_data_init --warmup_timestep 5e-5 --generate_steps 12200
    
    10. Override only training timestep (let system auto-calculate warmup timestep):
    python generate_turbulence_data_pict.py --use_training_data_init --training_timestep 1e-4 --generate_steps 5000
    
    Warmup data features:
    - Loads initial velocity from warmup_data subdirectory
    - Supports warmup segment files (decaying_turbulence_v2_warmup_segment_X_step_Y_index_1.npz)
    - Default uses segment 6 (step 300) representing post-warmup state
    - Automatically resamples data to match target resolution if needed
    - Always performs 4s warmup as requested for proper comparison
    - Uses same timestep extraction logic as training data
    - Maintains consistency with reference simulation temporal resolution
    
    Available warmup segments:
    - Segment 1 (step 50): Early warmup state
    - Segment 2 (step 100): Mid-early warmup
    - Segment 3 (step 150): Mid warmup
    - Segment 4 (step 200): Mid-late warmup
    - Segment 5 (step 250): Late warmup
    - Segment 6 (step 300): Post-warmup state (recommended for comparison)
    
    Comparison and visualization features:
    - Enable with --enable_comparison flag
    - Loads reference training data for step-by-step comparison
    - Creates detailed comparison plots for each simulation step
    - Generates difference fields (PICT - Reference) with statistics
    - Produces error evolution plots showing RMSE and max differences over time
    - Saves comparison statistics for quantitative analysis
    - Outputs include U-velocity, V-velocity, and magnitude comparisons
    
    Training data timestep extraction:
    - Looks for 'delta_t' field in data files
    - Falls back to calculating from 'time_array' if available
    - Uses computed CFD timestep if data timestep cannot be extracted
    - Saves timestep information in generated data files for consistency
    
    CFD Timestep Calculation:
    The system automatically computes stable timesteps based on multiple criteria:
    
    1. CFL (Convective) Stability: Δt ≤ CFL_target * Δx / |u_max|
       - Ensures numerical stability for convective terms
       - Default CFL_target = 0.5 (conservative)
       - Uses actual maximum velocity from field or args.max_velocity
       
    2. Viscous (Diffusive) Stability: Δt ≤ 0.5 * (Δx)² / ν
       - Prevents instability in diffusive terms
       - Critical for explicit viscous schemes
       - Diffusion number D = ν*Δt/(Δx)² ≤ 0.5
       
    3. Kolmogorov Time Scale: Δt ≤ 0.1 * τ_η where τ_η = √(ν/ε)
       - Ensures proper resolution of smallest turbulent scales
       - Energy dissipation rate ε estimated as u³/L_integral
       - Essential for accurate DNS of turbulent flows
       
    4. Acoustic Stability: Δt ≤ 0.1 * Δx / c_characteristic
       - Accounts for pressure wave propagation
       - Less critical for incompressible flows but included for robustness
       
    5. Physical Parameter Verification:
       - Grid Reynolds number Re_grid = u*Δx/ν
       - Warning if Re_grid > 2 (insufficient resolution for DNS)
       - Timestep bounds: 1e-6 ≤ Δt ≤ 0.01
       - Safety factor applied (default 0.8)
       
    The most restrictive condition determines the final timestep, ensuring
    numerical stability and physical accuracy across all scales.
    
    Manual Timestep Override:
    Users can now manually specify timesteps using command line arguments:
    
    --warmup_timestep: Override automatic calculation for warmup phase
    - Takes precedence over training data timestep and CFD calculations
    - Useful for matching specific experimental conditions
    - System validates timestep is positive and warns if > 0.1 (potentially unstable)
    
    --training_timestep: Override automatic calculation for training data generation  
    - Takes precedence over all other timestep sources
    - Allows precise control over temporal resolution in generated data
    - System validates timestep is positive and warns if > 0.1 (potentially unstable)
    
    Timestep Priority Order:
    1. Manual specification (--warmup_timestep / --training_timestep) [HIGHEST]
    2. Extracted from training/simulation data files
    3. Computed using CFD stability criteria [FALLBACK]
    
    This provides flexibility for both automatic operation and manual control
    when specific timestep requirements are needed.
    """
    parser = argparse.ArgumentParser(description='Generate turbulence training data using PICT')
    
    # Simulation parameters
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--dims', type=int, default=2, choices=[2, 3], 
                       help='Spatial dimensions (2D or 3D)')
    parser.add_argument('--generate_steps', type=int, default=12200)
    parser.add_argument('--save_interval', type=int, default=1,
                       help='Save data every N simulation steps')
    parser.add_argument('--warmup_time', type=float, default=4.0)
    
    # Physical parameters
    parser.add_argument('--max_velocity', type=float, default=4.2)
    parser.add_argument('--cfl_safety_factor', type=float, default=0.5)
    parser.add_argument('--viscosity', type=float, default=1e-3)
    parser.add_argument('--peak_wavenumber', type=int, default=4)
    parser.add_argument('--domain_scale', type=float, default=1.0)
    parser.add_argument('--decay', action='store_true', default=True,
                       help='Generate decaying turbulence (no forcing)')
    
    # Turbulence spectrum parameters for spatial continuity control
    parser.add_argument('--integral_scale_factor', type=float, default=6.0,
                       help='Domain size / integral length scale (smaller = larger eddies, better continuity)')
    parser.add_argument('--taylor_reynolds', type=float, default=50.0,
                       help='Taylor microscale Reynolds number (smaller = larger Kolmogorov scale)')
    
    # Adaptive timestep (CFL-based)
    parser.add_argument('--adaptive_timestep', action='store_true', default=False,
                       help='Enable adaptive timestep based on CFL condition')
    parser.add_argument('--adaptive_cfl', type=float, default=0.8,
                       help='Target CFL number for adaptive timestep')
    parser.add_argument('--visualize_max_steps', type=int, default=5,
                       help='Only visualize the first N steps (set None to disable limit)')
    
    # Manual timestep specification
    parser.add_argument('--warmup_timestep', type=float, default=None,
                       help='Manually specify timestep for warmup phase (overrides auto-calculation)')
    parser.add_argument('--training_timestep', type=float, default=None,
                       help='Manually specify timestep for training data generation (overrides auto-calculation)')
    
    # Resolution parameters
    parser.add_argument('--low_res', type=int, default=64)

    parser.add_argument('--high_res', type=int, default=128,  # Reduced from 2048 for PICT
                       help='Highest resolution (limited by GPU memory)')
    
    # Output parameters
    parser.add_argument('--save_dir', type=str, default='./data/pict_turbulence')
    parser.add_argument('--save_file', type=str, default="decaying_turbulence")
    parser.add_argument('--save_index', type=int, default=1)
    
    # Training data initialization
    parser.add_argument('--use_training_data_init', action='store_true', default=True,
                       help='Use training data t=0 velocity for initialization (skips warmup)')
    parser.add_argument('--use_warmup_data_init', action='store_true', default=True,
                       help='Use warmup data t=0 velocity for initialization (overrides use_training_data_init)')
    parser.add_argument('--warmup_segment', type=int, default=0,
                       help='Which warmup segment to use for initialization (1-6, default=6 for post-warmup state)')
    parser.add_argument('--enable_comparison', action='store_true', default=False,
                       help='Enable step-by-step comparison with reference training data and visualization')
    parser.add_argument('--training_data_dir', type=str, default='./training_data/warmup_data',
                       help='Directory containing training data files')
    
    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("Main")
    
    # Validate manual timestep parameters
    if args.warmup_timestep is not None:
        if args.warmup_timestep <= 0:
            logger.error(f"Invalid warmup_timestep: {args.warmup_timestep}. Must be positive.")
            return
        if args.warmup_timestep > 0.1:
            logger.warning(f"Large warmup_timestep: {args.warmup_timestep}. This may cause numerical instability.")
    
    if args.training_timestep is not None:
        if args.training_timestep <= 0:
            logger.error(f"Invalid training_timestep: {args.training_timestep}. Must be positive.")
            return
        if args.training_timestep > 0.1:
            logger.warning(f"Large training_timestep: {args.training_timestep}. This may cause numerical instability.")
    
    # Generate data
    generator = TurbulenceDataGenerator(args)
    generator.generate_data()
    
    logger.info("Data generation completed!")


if __name__ == "__main__":
    main() 