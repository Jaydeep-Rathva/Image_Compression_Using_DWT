# DSP Project - Image Compression using DWT
# Based on: Madhankumar et al., IJCSMC, Vol.11, Jan 2022
# We implemented this in Python since we don't have MATLAB/Simulink

import argparse
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pywt


# loading the image from the given path, converting BGR to RGB since opencv loads as BGR by default
def load_image(path):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Image not found at: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def save_image(img_rgb, out_path):
    # converting back to BGR before saving
    cv2.imwrite(out_path, cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))


# applying 2 level DWT on a single channel (grayscale)
# level 1: splits into LL1, LH1, HL1, HH1
# level 2: splits LL1 further into LL2, LH2, HL2, HH2
def do_dwt(channel, wavelet='haar'):
    ch = channel.astype(np.float32)

    # first level decomposition
    ll1, (lh1, hl1, hh1) = pywt.dwt2(ch, wavelet)

    # second level decomposition on the LL subband
    ll2, (lh2, hl2, hh2) = pywt.dwt2(ll1, wavelet)

    return {
        'LL2': ll2,
        'LH2': lh2, 'HL2': hl2, 'HH2': hh2,   # level 2 details
        'LH1': lh1, 'HL1': hl1, 'HH1': hh1,   # level 1 details
    }


# hard thresholding - coefficients below threshold become 0
# this is the compression step, LL2 is kept as it is (most important part)
def apply_threshold(coeffs, thresh):
    result = {'LL2': coeffs['LL2']}  # keep approximation as it is
    for key in ('LH2', 'HL2', 'HH2', 'LH1', 'HL1', 'HH1'):
        result[key] = pywt.threshold(coeffs[key], thresh, mode='hard')
    return result


# reconstructing the image back from the coefficients using IDWT
def do_idwt(coeffs, wavelet='haar'):
    # reconstruct level 2 first
    ll1_rec = pywt.idwt2((coeffs['LL2'], (coeffs['LH2'], coeffs['HL2'], coeffs['HH2'])), wavelet)

    # then reconstruct level 1 to get back original size
    reconstructed = pywt.idwt2((ll1_rec, (coeffs['LH1'], coeffs['HL1'], coeffs['HH1'])), wavelet)

    return np.clip(reconstructed, 0, 255).astype(np.uint8)


# running the full compression pipeline on all 3 channels (R, G, B)
def compress(img_rgb, wavelet='haar', thresh=20.0):
    channels = cv2.split(img_rgb)
    all_coeffs = []
    reconstructed_channels = []

    for ch in channels:
        raw = do_dwt(ch, wavelet)
        thresholded = apply_threshold(raw, thresh)
        rec = do_idwt(thresholded, wavelet)
        all_coeffs.append(raw)
        reconstructed_channels.append(rec)

    comp_img = cv2.merge(reconstructed_channels)
    return comp_img, all_coeffs


# MSE = average of squared differences between original and compressed
def calc_mse(orig, comp):
    return float(np.mean((orig.astype(np.float64) - comp.astype(np.float64)) ** 2))


# PSNR tells us how good the compressed image is, higher is better
def calc_psnr(mse):
    if mse == 0:
        return float('inf')
    return 10.0 * np.log10((255.0 ** 2) / mse)


# compression ratio = original file size / compressed file size
def calc_compression_ratio(orig_path, comp_img, thresh):
    orig_size = os.path.getsize(orig_path)
    tmp = f'_tmp_{thresh}.png'
    save_image(comp_img, tmp)
    comp_size = os.path.getsize(tmp)
    os.remove(tmp)
    if comp_size == 0:
        return 0.0
    return orig_size / comp_size


# showing all results in one figure
def show_results(orig, comp, coeffs, mse, psnr, cr, thresh, wavelet):
    g = coeffs[1]  # using green channel for subband display (eye is most sensitive to green)
    diff = cv2.absdiff(orig, comp)

    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    fig.suptitle(
        f'Image Compression using 2-Level Haar DWT  |  Threshold={thresh}  |  '
        f'MSE={mse:.4f}  |  PSNR={psnr:.2f} dB  |  Compression Ratio={cr:.2f}x',
        fontsize=12
    )

    items = [
        (orig,              'Original Image',           None),
        (g['LL2'],          'LL (Level-2 Approx)',      'gray'),
        (np.abs(g['LH2']),  'LH (Level-2 Horizontal)',  'gray'),
        (np.abs(g['HL2']),  'HL (Level-2 Vertical)',    'gray'),
        (np.abs(g['HH2']),  'HH (Level-2 Diagonal)',    'gray'),
        (comp,              'Compressed / Reconstructed', None),
        (np.abs(g['LH1']),  'LH (Level-1 Horizontal)',  'gray'),
        (np.abs(g['HL1']),  'HL (Level-1 Vertical)',    'gray'),
        (np.abs(g['HH1']),  'HH (Level-1 Diagonal)',    'gray'),
        (diff,              'Difference |Orig - Comp|', 'hot'),
    ]

    for ax, (img, title, cmap) in zip(axes.flat, items):
        if img.ndim == 3:
            ax.imshow(img)
        else:
            ax.imshow(img, cmap=cmap)
        ax.set_title(title, fontsize=9)
        ax.axis('off')

    plt.tight_layout()
    plt.savefig('dwt_results.png', dpi=150, bbox_inches='tight')
    plt.show()


# plotting how PSNR and compression ratio change with different threshold values
def analyse_thresholds(img_path, wavelet='haar'):
    orig = load_image(img_path)
    thresholds = list(range(0, 201, 20))
    psnr_vals = []
    cr_vals = []

    for t in thresholds:
        comp, _ = compress(orig, wavelet, t)
        mse = calc_mse(orig, comp)
        psnr_vals.append(calc_psnr(mse) if mse != 0 else 100)
        cr_vals.append(calc_compression_ratio(img_path, comp, t))

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.set_xlabel('Threshold Value')
    ax1.set_ylabel('PSNR (dB)', color='steelblue')
    ax1.plot(thresholds, psnr_vals, 'o-', color='steelblue', label='PSNR (dB)')
    ax1.tick_params(axis='y', labelcolor='steelblue')

    ax2 = ax1.twinx()
    ax2.set_ylabel('Compression Ratio', color='darkorange')
    ax2.plot(thresholds, cr_vals, 's--', color='darkorange', label='Compression Ratio')
    ax2.tick_params(axis='y', labelcolor='darkorange')

    plt.title('PSNR and Compression Ratio vs. Threshold (2-Level Haar DWT)')
    fig.tight_layout()
    plt.savefig('threshold_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()


def parse_args():
    parser = argparse.ArgumentParser(description='DWT Image Compression')
    parser.add_argument('--image', type=str, default='input.jpg', help='path to input image')
    parser.add_argument('--wavelet', type=str, default='haar', help='wavelet type')
    parser.add_argument('--threshold', type=float, default=20.0, help='threshold value')
    parser.add_argument('--output', type=str, default='compressed_output.png', help='output image path')
    parser.add_argument('--analyse', action='store_true', help='plot threshold analysis')
    return parser.parse_args()


def main():
    args = parse_args()

    print(f'\nLoading: {args.image}')
    orig = load_image(args.image)
    print(f'Size: {orig.shape[1]} x {orig.shape[0]}  |  Channels: {orig.shape[2]}')

    print(f'Compressing with threshold = {args.threshold}...')
    comp, coeffs = compress(orig, wavelet=args.wavelet, thresh=args.threshold)

    mse = calc_mse(orig, comp)
    psnr = calc_psnr(mse)
    cr = calc_compression_ratio(args.image, comp, args.threshold)

    print('\n--- Results ---')
    print(f'  Wavelet   : {args.wavelet}')
    print(f'  Threshold : {args.threshold}')
    print(f'  MSE       : {mse:.6f}')
    print(f'  PSNR      : {psnr:.4f} dB')
    print(f'  Comp Ratio: {cr:.2f}x')
    print('----------------\n')

    save_image(comp, args.output)
    print(f'Saved compressed image as: {args.output}')

    show_results(orig, comp, coeffs, mse, psnr, cr, args.threshold, args.wavelet)

    if args.analyse:
        print('Running threshold analysis...')
        analyse_thresholds(args.image, wavelet=args.wavelet)


if __name__ == '__main__':
    main()
