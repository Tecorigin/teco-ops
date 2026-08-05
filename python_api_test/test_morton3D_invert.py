#!/usr/bin/env python3

import numpy as np
import torch
import torch_sdaa
import _bootstrap
import tecoops


def morton3d_invert_cpu(indices):
    def compact_bits(x):
        x = int(x)
        x &= 0x09249249
        x = (x ^ (x >> 2)) & 0x030C30C3
        x = (x ^ (x >> 4)) & 0x0300F00F
        x = (x ^ (x >> 8)) & 0xFF0000FF
        x = (x ^ (x >> 16)) & 0x000003FF
        return x

    coords = np.zeros((len(indices), 3), dtype=np.int32)
    for i, ind in enumerate(indices):
        ind = int(ind)
        coords[i, 0] = compact_bits(ind >> 0)
        coords[i, 1] = compact_bits(ind >> 1)
        coords[i, 2] = compact_bits(ind >> 2)
    return coords


def test_morton3d_invert_basic():
    print("Testing morton3D_invert...")

    indices_data = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=np.int32)
    N = len(indices_data)

    indices = torch.from_numpy(indices_data).to("sdaa")
    coords = torch.zeros((N, 3), dtype=torch.int32, device="sdaa")

    tecoops.morton3D_invert(indices, N, coords)

    result = coords.cpu().numpy()
    expected = morton3d_invert_cpu(indices_data)

    if np.array_equal(result, expected):
        print("\n✓ Basic test PASSED!")
        return True
    else:
        print("\n✗ Basic test FAILED!")
        print(result - expected)
        return False


def test_morton3d_invert_random():
    print("\nTesting morton3D_invert with random data...")

    np.random.seed(42)
    indices_data = np.random.randint(0, 1 << 20, size=128, dtype=np.int32)
    N = len(indices_data)

    indices = torch.from_numpy(indices_data).to("sdaa")
    coords = torch.zeros((N, 3), dtype=torch.int32, device="sdaa")

    tecoops.morton3D_invert(indices, N, coords)

    result = coords.cpu().numpy()
    expected = morton3d_invert_cpu(indices_data)

    if np.array_equal(result, expected):
        print("✓ Random test PASSED!")
        return True
    else:
        print("✗ Random test FAILED!")
        mismatch = np.where(result != expected)
        print("Mismatch indices:", mismatch)
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("morton3D_invert Test Suite")
    print("=" * 60)

    if not torch.sdaa.is_available():
        print("Warning: SDAA is not available, tests may fail")

    try:
        test1_passed = test_morton3d_invert_basic()
    except Exception as e:
        print(f"Basic test crashed: {e}")
        test1_passed = False

    try:
        test2_passed = test_morton3d_invert_random()
    except Exception as e:
        print(f"Random test crashed: {e}")
        test2_passed = False

    print("\n" + "=" * 60)
    if test1_passed and test2_passed:
        print("All tests PASSED!")
    else:
        print("Some tests FAILED!")
    print("=" * 60)
