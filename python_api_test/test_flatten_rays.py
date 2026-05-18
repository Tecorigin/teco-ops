#!/usr/bin/env python3
"""
Test script for flatten_rays function.

flatten_rays takes:
- rays: tensor of shape [N, 2] containing (offset, num_steps) for each ray
- N: number of rays
- M: total number of steps (sum of all num_steps)

It flattens the rays into a 1D array where each position is filled with the ray index.
"""

import torch
import torch_sdaa
import tecoops
import numpy as np


def test_flatten_rays():
    """Test flatten_rays function with simple example."""
    print("Testing flatten_rays...")
    
    # Create test data: 3 rays with different step counts
    # rays[i] = [offset, num_steps]
    # Ray 0: offset=0, num_steps=3
    # Ray 1: offset=3, num_steps=2  
    # Ray 2: offset=5, num_steps=4
    # Total steps M = 3 + 2 + 4 = 9
    rays_data = np.array([
        [0, 3],
        [3, 2],
        [5, 4]
    ], dtype=np.int32)
    
    N = 3  # number of rays
    M = 9  # total steps
    
    # Create tensors on SDAA
    rays = torch.from_numpy(rays_data).to('sdaa')
    res = torch.zeros(M, dtype=torch.int32, device='sdaa')
    
    print(f"Input rays (shape {rays.shape}):")
    print(rays.cpu().numpy())
    print(f"N={N}, M={M}")
    
    # Call flatten_rays
    tecoops.flatten_rays(rays, N, M, res)
    
    # Get result
    result = res.cpu().numpy()
    print(f"\nResult (shape {result.shape}):")
    print(result)
    
    # Expected result:
    # positions 0,1,2 should be 0 (ray 0)
    # positions 3,4 should be 1 (ray 1)
    # positions 5,6,7,8 should be 2 (ray 2)
    expected = np.array([0, 0, 0, 1, 1, 2, 2, 2, 2], dtype=np.int32)
    
    print(f"\nExpected:")
    print(expected)
    
    # Verify
    if np.array_equal(result, expected):
        print("\n✓ Test PASSED!")
        return True
    else:
        print("\n✗ Test FAILED!")
        print(f"Difference: {result - expected}")
        return False


def test_flatten_rays_random():
    """Test flatten_rays with random data."""
    print("\nTesting flatten_rays with random data...")
    
    np.random.seed(42)
    N = 10
    
    # Generate random num_steps for each ray
    num_steps_list = np.random.randint(1, 10, size=N)
    
    # Calculate offsets
    offsets = np.zeros(N, dtype=np.int32)
    offsets[1:] = np.cumsum(num_steps_list[:-1])
    
    # Create rays array
    rays_data = np.stack([offsets, num_steps_list], axis=1).astype(np.int32)
    M = int(num_steps_list.sum())
    
    print(f"Number of rays: {N}")
    print(f"Total steps: {M}")
    print(f"Rays data shape: {rays_data.shape}")
    
    # Create tensors on SDAA
    rays = torch.from_numpy(rays_data).to('sdaa')
    res = torch.zeros(M, dtype=torch.int32, device='sdaa')
    
    # Call flatten_rays
    tecoops.flatten_rays(rays, N, M, res)
    
    # Get result
    result = res.cpu().numpy()
    
    # Verify: check that each ray's segment has the correct ray index
    success = True
    for i in range(N):
        offset = int(offsets[i])
        num_steps = int(num_steps_list[i])
        segment = result[offset:offset + num_steps]
        
        if not np.all(segment == i):
            print(f"✗ Ray {i}: expected all {i}, got {segment}")
            success = False
    
    if success:
        print("✓ Random test PASSED!")
    else:
        print("✗ Random test FAILED!")
    
    return success


if __name__ == "__main__":
    print("=" * 60)
    print("flatten_rays Test Suite")
    print("=" * 60)
    
    # Check if SDAA is available
    if not torch.sdaa.is_available():
        print("Warning: SDAA is not available, tests may fail")
    
    # Run tests
    test1_passed = test_flatten_rays()
    test2_passed = test_flatten_rays_random()
    
    print("\n" + "=" * 60)
    if test1_passed and test2_passed:
        print("All tests PASSED!")
    else:
        print("Some tests FAILED!")
    print("=" * 60)
