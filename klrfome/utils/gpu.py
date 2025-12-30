"""GPU support and verification utilities."""

import jax
import jax.numpy as jnp


def check_gpu_available() -> bool:
    """
    Check if GPU is available for JAX.
    
    Returns:
        True if GPU is available
    """
    try:
        devices = jax.devices()
        return any(d.device_kind == 'gpu' for d in devices)
    except Exception:
        return False


def get_device_info() -> dict:
    """
    Get information about available JAX devices.
    
    Returns:
        Dictionary with device information
    """
    devices = jax.devices()
    info = {
        'devices': [str(d) for d in devices],
        'default_device': str(jax.devices()[0]),
        'gpu_available': check_gpu_available(),
        'device_count': len(devices),
    }
    
    if check_gpu_available():
        gpu_devices = [d for d in devices if d.device_kind == 'gpu']
        info['gpu_devices'] = [str(d) for d in gpu_devices]
        info['gpu_count'] = len(gpu_devices)
    
    return info


def test_gpu_kernel_computation():
    """
    Test that kernel computations work on GPU.
    
    Returns:
        True if test passes
    """
    try:
        from ..kernels.rbf import RBFKernel
        
        # Create test data
        X = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        Y = jnp.array([[1.0, 2.0], [5.0, 6.0]])
        
        # Compute kernel (should run on GPU if available)
        kernel = RBFKernel(sigma=1.0)
        K = kernel(X, Y)
        
        # Check result
        assert K.shape == (2, 2)
        return True
    except Exception as e:
        print(f"GPU test failed: {e}")
        return False

