# data_gen/generate_csi_cdla.py
"""
Generate CDL-A CSI dataset for LatentMIM training.

Output files (in --out-dir):
  train_data.npy   [70000, 2, 64, 64] float32   real/imag delay-azimuth
  test_data.npy    [10000, 2, 64, 64] float32
  train_paths.npy  [70000]            int32      dominant path count per sample
  test_paths.npy   [10000]            int32
"""

import os
import argparse
import numpy as np

os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"

import tensorflow as tf
from sionna.phy.channel.tr38901 import AntennaArray, CDL

CARRIER_FREQUENCY = 7.0e9
DELAY_SPREAD = 100e-9
SPEED = 30.0 / 3.6
N_DELAY = 64
N_FFT_ANGULAR = 64
MAX_DELAY = 1000e-9
DELAY_RESOLUTION = MAX_DELAY / N_DELAY
PATH_THRESHOLD_DB = -20.0
N_TOTAL = 80_000
N_TEST = 10_000
N_TRAIN = N_TOTAL - N_TEST


def build_arrays():
    common = dict(
        polarization='single',
        polarization_type='H',
        antenna_pattern='38.901',
        carrier_frequency=CARRIER_FREQUENCY,
    )
    bs_array = AntennaArray(num_rows=1, num_cols=32, **common)
    ut_array = AntennaArray(num_rows=1, num_cols=1,  **common)
    return bs_array, ut_array


def process_batch(a_tf, tau_tf):
    """
    a_tf:   [B, 1, 1, 1, 32, 23, 1] complex
    tau_tf: [B, 1, 1, 23]            float  (seconds)

    Returns:
      data:    [B, 2, N_DELAY, N_FFT_ANGULAR] float32
      n_paths: [B]                             int32
    """
    a   = a_tf[:, 0, 0, 0, :, :, 0].numpy()  # [B, 32, 23] complex
    tau = tau_tf[:, 0, 0, :].numpy()          # [B, 23] seconds
    assert a_tf.shape[-1] == 1, f"Expected num_time_steps=1, got {a_tf.shape[-1]}"
    B   = a.shape[0]

    data    = np.zeros((B, 2, N_DELAY, N_FFT_ANGULAR), dtype=np.float32)
    n_paths = np.zeros(B, dtype=np.int32)

    for b in range(B):
        h   = a[b].T                           # [23, 32] complex
        tau_b = tau[b]                         # [23]

        power = np.mean(np.abs(h) ** 2, axis=-1)
        max_p = power.max()
        if max_p > 0:
            threshold = max_p * 10 ** (PATH_THRESHOLD_DB / 10)
            n_paths[b] = int(np.sum(power >= threshold))
        else:
            print(f'Warning: sample {b} has zero channel power — skipping path count')

        H_angular = np.fft.fft(h, n=N_FFT_ANGULAR, axis=-1)  # [23, 64]

        H_grid = np.zeros((N_DELAY, N_FFT_ANGULAR), dtype=complex)
        for l in range(23):
            tap = int(round(tau_b[l] / DELAY_RESOLUTION))
            if 0 <= tap < N_DELAY:
                H_grid[tap] += H_angular[l]

        data[b, 0] = H_grid.real
        data[b, 1] = H_grid.imag

    return data, n_paths


def generate(num_samples, channel_model, batch_size=64):
    all_data    = np.zeros((num_samples, 2, N_DELAY, N_FFT_ANGULAR), dtype=np.float32)
    all_n_paths = np.zeros(num_samples, dtype=np.int32)
    generated = 0

    while generated < num_samples:
        bs = min(batch_size, num_samples - generated)
        a, tau = channel_model(batch_size=bs, num_time_steps=1,
                               sampling_frequency=1e9)
        data, n_paths = process_batch(a, tau)
        all_data[generated:generated + bs]    = data
        all_n_paths[generated:generated + bs] = n_paths
        generated += bs
        if generated % 5000 == 0 or generated == num_samples:
            print(f'  {generated}/{num_samples} samples generated')

    return all_data, all_n_paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out-dir',    default='data_gen/csi_cdla')
    parser.add_argument('--batch-size', type=int, default=64)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print('Building antenna arrays and CDL-A channel model...')
    bs_array, ut_array = build_arrays()
    channel_model = CDL(
        'A', DELAY_SPREAD, CARRIER_FREQUENCY,
        ut_array, bs_array,
        direction='downlink',
        min_speed=SPEED, max_speed=SPEED,
    )

    print(f'Generating {N_TOTAL} samples (batch_size={args.batch_size})...')
    all_data, all_n_paths = generate(N_TOTAL, channel_model, args.batch_size)

    rng = np.random.default_rng(42)
    idx = rng.permutation(N_TOTAL)
    all_data    = all_data[idx]
    all_n_paths = all_n_paths[idx]

    train_data, test_data   = all_data[:N_TRAIN],    all_data[N_TRAIN:]
    train_paths, test_paths = all_n_paths[:N_TRAIN], all_n_paths[N_TRAIN:]

    np.save(os.path.join(args.out_dir, 'train_data.npy'),  train_data)
    np.save(os.path.join(args.out_dir, 'test_data.npy'),   test_data)
    np.save(os.path.join(args.out_dir, 'train_paths.npy'), train_paths)
    np.save(os.path.join(args.out_dir, 'test_paths.npy'),  test_paths)

    print(f'train_data:  {train_data.shape}  n_paths mean={train_paths.mean():.1f}')
    print(f'test_data:   {test_data.shape}   n_paths mean={test_paths.mean():.1f}')
    print(f'Saved to {args.out_dir}')


if __name__ == '__main__':
    main()
