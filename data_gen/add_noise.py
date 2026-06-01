# data_gen/add_noise.py
"""
Add AWGN to clean CDL-A channel data at a target SNR.

Reads:
  <in-dir>/train_clean.npy  [N, 64, 64] complex64
  <in-dir>/test_clean.npy   [N, 64, 64] complex64

Writes to <out-dir>:
  train_data.npy  [N, 1, 64, 64] float32   20*log10(|H + noise|)
  test_data.npy   [N, 1, 64, 64] float32

SNR is defined relative to the peak bin power of each sample.
"""

import os
import argparse
import numpy as np


def add_awgn(clean, snr_db, rng):
    """
    clean:  [N, 64, 64] complex64
    Returns [N, 1, 64, 64] float32  dB magnitude
    """
    N = clean.shape[0]
    snr_linear = 10 ** (snr_db / 10)
    out = np.zeros((N, 1, 64, 64), dtype=np.float32)

    for n in range(N):
        H = clean[n]                             # [64, 64] complex
        peak_power = np.max(np.abs(H) ** 2)
        if peak_power > 0:
            sigma = np.sqrt(peak_power / (2 * snr_linear))
        else:
            sigma = 0.0
        noise = sigma * (rng.standard_normal(H.shape) + 1j * rng.standard_normal(H.shape))
        H_noisy = H + noise
        out[n, 0] = (20 * np.log10(np.abs(H_noisy) + 1e-10)).astype(np.float32)

    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--in-dir',   default='data_gen/csi_cdla',
                        help='Directory containing *_clean.npy files')
    parser.add_argument('--out-dir',  default=None,
                        help='Output directory (defaults to --in-dir)')
    parser.add_argument('--snr-db',   type=float, required=True,
                        help='Target SNR in dB (e.g. 30)')
    parser.add_argument('--seed',     type=int, default=0)
    args = parser.parse_args()

    out_dir = args.out_dir or args.in_dir
    os.makedirs(out_dir, exist_ok=True)

    rng = np.random.default_rng(args.seed)

    for split in ('train', 'test'):
        clean_path = os.path.join(args.in_dir, f'{split}_clean.npy')
        out_path   = os.path.join(out_dir,    f'{split}_data.npy')

        print(f'Loading {clean_path} ...')
        clean = np.load(clean_path)              # [N, 64, 64] complex64
        print(f'  shape={clean.shape}  dtype={clean.dtype}')

        print(f'  Adding AWGN at SNR={args.snr_db} dB ...')
        data = add_awgn(clean, args.snr_db, rng)

        np.save(out_path, data)
        print(f'  Saved {out_path}  shape={data.shape}  '
              f'min={data.min():.1f}  max={data.max():.1f}')

    print('Done.')


if __name__ == '__main__':
    main()
