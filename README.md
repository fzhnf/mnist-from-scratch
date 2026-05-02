# Klasifikasi Digit MNIST dari Nol

[![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/fzhnf/mnist-from-scratch/blob/main/mnist_from_scratch.py/wasm)

MNIST Classifier using Kelompok 4.
Klasifikasi digit tulisan tangan menggunakan Convolutional Neural Network (LeNet-5), dilatih dengan PyTorch dan dijalankan di browser via WebAssembly (molab/Pyodide).

## run locally using UV

```bash
uv sync
uv run marimo run mnist_from_scratch.py --no-sandbox
```

>![NOTE]
>Jika menjalankan notebook secara lokal, klik **Train locally** dan tunggu
>hingga pelatihan selesai.
>
>Jika `model_weights.npz` masih format lama, klik **Train locally** sekali
>untuk membuat bobot LeNet terbaru.
