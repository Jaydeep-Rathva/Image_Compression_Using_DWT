# Image Compression using Discrete Wavelet Transform (DWT)

A Python implementation of lossy image compression using the **2-level 2D Haar Discrete Wavelet Transform (DWT)**, based on the research paper cited below.

---

## 📄 Based On

> G. Madhankumar, K. Tarun, Dr. G. Ganesh Kumar,
> **"Image Compression Using Discrete Wavelet Transform in Simulink"**,
> *International Journal of Computer Science and Mobile Computing (IJCSMC)*,
> Vol. 11, Issue 1, January 2022, pp. 122–129.
> DOI: [10.47760/ijcsmc.2022.v11i01.015](https://doi.org/10.47760/ijcsmc.2022.v11i01.015)

---

## 📌 How It Works

1. Load a colour (RGB) image
2. Split into R, G, B channels
3. Apply **2-level 2D Haar DWT** to each channel → produces sub-bands: LL, LH, HL, HH (at both levels)
4. Apply **hard thresholding** to all detail sub-bands (LH, HL, HH) — small coefficients are zeroed out
5. Reconstruct the image using **2-level Inverse DWT (IDWT)**
6. Evaluate quality using **MSE** (Mean Square Error) and **PSNR** (Peak Signal-to-Noise Ratio)

---

## 🗂️ Project Structure

```
├── dwt_image_compression.py   # Main Python script
├── input.jpg                  # Your input image (add your own)
├── README.md                  # This file
```

---

## ⚙️ Requirements

Install dependencies using pip:

```bash
pip install numpy pywavelets opencv-python matplotlib
```

---

## 🚀 Usage

### Basic compression:
```bash
python dwt_image_compression.py --image input.jpg --threshold 20
```

### Save output to a custom file:
```bash
python dwt_image_compression.py --image input.jpg --threshold 20 --output result.png
```

### Also run threshold analysis (PSNR vs Threshold plot):
```bash
python dwt_image_compression.py --image input.jpg --threshold 20 --analyse
```

### All arguments:

| Argument      | Default       | Description                                      |
|---------------|---------------|--------------------------------------------------|
| `--image`     | `input.jpg`   | Path to input colour image                       |
| `--wavelet`   | `haar`        | Wavelet type (`haar`, `db1`, etc.)               |
| `--threshold` | `20.0`        | Hard-threshold value for detail coefficients     |
| `--output`    | `compressed_output.png` | Path to save compressed image        |
| `--analyse`   | *(flag)*      | Plot PSNR and Compression Ratio vs. Threshold    |

---

## 📊 Output

The script produces:
- **Console output**: MSE, PSNR (dB), and Compression Ratio
- **`dwt_results.png`**: Visual grid showing Original, all DWT sub-bands, Compressed, and Difference images
- **`threshold_analysis.png`** *(if `--analyse` is used)*: PSNR and Compression Ratio vs. Threshold plot
- **Compressed image** saved at the path specified by `--output`

### Example console output:
```
─────────────────────────────────────
  Wavelet          : haar
  Threshold        : 20.0
  MSE              : 0.2300
  PSNR             : 54.4000 dB
  Compression Ratio: 1.85x
─────────────────────────────────────
```

---

## 📚 Citation

If you use this code, please cite the original paper:

```
@article{madhankumar2022dwt,
  title     = {Image Compression Using Discrete Wavelet Transform in Simulink},
  author    = {Madhankumar, G. and Tarun, K. and Ganesh Kumar, G.},
  journal   = {International Journal of Computer Science and Mobile Computing},
  volume    = {11},
  number    = {1},
  pages     = {122--129},
  year      = {2022},
  doi       = {10.47760/ijcsmc.2022.v11i01.015}
}
```

---

## 📝 License

This project is for academic purposes only.
