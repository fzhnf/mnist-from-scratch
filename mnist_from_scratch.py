# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo>=0.23.4",
#     "matplotlib>=3.10.9",
#     "numpy",
#     "pandas>=3.0.0",
#     "anywidget>=0.9.0",
#     "traitlets",
# ]
# ///

import marimo

__generated_with = "0.23.4"
app = marimo.App(width="medium")

with app.setup:

    @app.cell(hide_code=True)
    def _():
        mo.md("""
        # Klasifikasi Digit MNIST Kelompok 4
        
        anggota:

        1. Azhar Rizqullah Fakhri Ismail 11221052 
        1. Pahril Dwi Saputra 11221056 
        1. Faiz Ahnaf Samudra Aziz 11221076 
        1. Ayu Nabila Andara Wati 11221084 

        **Link Live Preview Aplikasi (WASM):** https://molab.marimo.io/github/fzhnf/mnist-from-scratch/blob/main/mnist_from_scratch.py/wasm
        **Link Source code:** https://github.com/fzhnf/mnist-from-scratch

        ---
        """)
        return

    import marimo as mo
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np
    import anywidget
    import traitlets
    from types import SimpleNamespace
    import sys, io

    _REPO = "https://raw.githubusercontent.com/fzhnf/mnist-from-scratch/main/"
    EPOCHS = 10
    BATCH_SIZE = 256
    LR = 0.001
    OPTIMIZER = "Adam"
    LOSS_FN = "CrossEntropyLoss"

    if sys.platform == "emscripten":
        import urllib.request

        _d = np.load(
            io.BytesIO(urllib.request.urlopen(_REPO + "mnist_data.npz").read())
        )
        _emb = np.load(
            io.BytesIO(urllib.request.urlopen(_REPO + "embedding.npy").read())
        )
        _w = np.load(
            io.BytesIO(urllib.request.urlopen(_REPO + "model_weights.npz").read())
        )
    else:
        _d = np.load("mnist_data.npz")
        _emb = np.load("embedding.npy")
        _w = np.load("model_weights.npz")

    mnist = SimpleNamespace(
        data=_d["X"].astype(np.float32) / 255.0,
        attributes={"digits": _d["y"]},
        embedding=_emb,
        weights=_w,
    )


@app.class_definition
class DrawWidget(anywidget.AnyWidget):
    _esm = """
    function render({ model, el }) {
      const SIZE = 200;
      const canvas = document.createElement("canvas");
      canvas.width = SIZE; canvas.height = SIZE;
      canvas.style.cssText = "border:1px solid #ccc;cursor:crosshair;touch-action:none;display:block";
      const ctx = canvas.getContext("2d");
      ctx.fillStyle = "white"; ctx.fillRect(0, 0, SIZE, SIZE);

      let down = false;
      canvas.addEventListener("pointerdown", () => { down = true; });
      canvas.addEventListener("pointerup",   () => { down = false; push(); });
      canvas.addEventListener("pointermove", e => {
        if (!down) return;
        const r = canvas.getBoundingClientRect();
        ctx.fillStyle = "black";
        ctx.beginPath();
        ctx.arc(e.clientX - r.left, e.clientY - r.top, 8, 0, Math.PI * 2);
        ctx.fill();
      });

      function push() {
        const small = document.createElement("canvas");
        small.width = 28; small.height = 28;
        small.getContext("2d").drawImage(canvas, 0, 0, 28, 28);
        const d = small.getContext("2d").getImageData(0, 0, 28, 28).data;
        const out = [];
        for (let i = 0; i < d.length; i += 4)
          out.push((255 - d[i]) / 255);
        model.set("pixels", out);
        model.save_changes();
      }

      const btn = document.createElement("button");
      btn.textContent = "Clear";
      btn.style.marginTop = "4px";
      btn.onclick = () => {
        ctx.fillStyle = "white"; ctx.fillRect(0, 0, SIZE, SIZE);
        model.set("pixels", []); model.save_changes();
      };
      el.append(canvas, btn);
    }
    export default { render };
    """
    pixels = traitlets.List([]).tag(sync=True)


@app.cell(hide_code=True)
def _():
    mo.md("""
    ## Pendahuluan & Tujuan Eksperimen Awal

    Tujuan eksperimen awal ini adalah membangun sistem klasifikasi digit tulisan tangan menggunakan dataset MNIST. Masalah utama yang diselesaikan adalah
    mengklasifikasikan gambar digit 0–9 (28×28 piksel, grayscale) ke dalam 10 kelas yang tepat.

    Pipeline yang dibangun mencakup:

    1. Memuat dan melakukan pra-pemrosesan dataset MNIST,
    2. Melatih model **Multi-Layer Perceptron (MLP)** tiga lapisan dari awal menggunakan PyTorch,
    3. Menyediakan antarmuka gambar interaktif untuk menguji model secara langsung.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md("""
    ## Metodologi Data (Data Pipeline)

    ### Deskripsi Dataset

    | Atribut | Keterangan |
    |:--|:--|
    | **Sumber Data** | MNIST (*Mixed National Institute of Standards and Technology*), Yann LeCun et al. — diakses via `torchvision.datasets.MNIST` |
    | **Jumlah Data Total** | 70.000 gambar grayscale 28×28 piksel, label digit 0–9 |
    | **Pembagian Data** | 60.000 Training · 10.000 Testing (pembagian bawaan torchvision; test set digunakan sebagai set validasi) |
    | **Distribusi Kelas** | 10 kelas seimbang (*balanced*): ±6.000 sampel per kelas di set pelatihan |

    ### Pra-pemrosesan Data

    1. **Flatten**: gambar 28×28 piksel di-*flatten* menjadi vektor **784 dimensi**.
    2. **Normalisasi**: nilai piksel dari rentang \[0, 255\] (uint8) dinormalisasi ke \[0.0, 1.0\]
       (float32) dengan membagi 255.
    """)
    return


@app.cell(hide_code=True)
def _():
    _indices = [int(np.where(mnist.attributes["digits"] == d)[0][0]) for d in range(10)]
    _images = mnist.data.reshape(-1, 28, 28)[_indices]
    _fig, _axes = plt.subplots(1, 10, figsize=(12, 1.6))
    for _i, (_ax, _im) in enumerate(zip(_axes, _images)):
        _ax.imshow(_im, cmap="gray")
        _ax.set_title(str(_i), fontsize=9)
        _ax.axis("off")
    _fig.suptitle("Contoh satu gambar per kelas (digit 0–9)", fontsize=9, y=1.08)
    plt.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _():
    _counts = np.bincount(mnist.attributes["digits"][:60000].astype(int), minlength=10)
    _fig, _ax = plt.subplots(figsize=(7, 3))
    _ax.bar(range(10), _counts, color="steelblue")
    _ax.set_xticks(range(10))
    _ax.set_xlabel("Digit")
    _ax.set_ylabel("Jumlah Sampel")
    _ax.set_title("Distribusi Kelas — Training Set (60.000 sampel)")
    for _x, _c in enumerate(_counts):
        _ax.text(_x, _c + 60, str(_c), ha="center", fontsize=8)
    plt.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _():
    mo.md("""
    ## Arsitektur Baseline Model

    **Jenis Arsitektur:** Multi-Layer Perceptron (MLP)

    | Layer | Jenis | Dimensi Output | Aktivasi |
    |:--|:--|:--|:--|
    | Input | — | 784 | — |
    | Hidden 1 | `nn.Linear` | 256 | ReLU |
    | Hidden 2 | `nn.Linear` | 128 | ReLU |
    | Output | `nn.Linear` | 10 | — (Softmax implisit via `CrossEntropyLoss`) |

    ```
    Input (784)
        └─ Linear(784 → 256) + ReLU
               └─ Linear(256 → 128) + ReLU
                      └─ Linear(128 → 10)  →  argmax  →  Prediksi digit
    ```

    Bobot model hasil pelatihan disimpan ke `model_weights.npz` dan dimuat ulang sebagai array
    NumPy biasa.
    """)
    return


@app.cell
def _():
    get_weights, set_weights = mo.state(mnist.weights)
    return get_weights, set_weights


@app.cell(hide_code=True)
def _():
    mo.md(f"""
    ## Hasil Eksperimen Training

    ### Konfigurasi Hyperparameter (Baseline)

    | Hyperparameter | Nilai |
    |:--|:--|
    | **Epochs** | {EPOCHS} |
    | **Batch Size** | {BATCH_SIZE} |
    | **Learning Rate** | {LR} |
    | **Optimizer** | {OPTIMIZER} |
    | **Loss Function** | {LOSS_FN} |
    """)
    return


@app.cell(hide_code=True)
def _():
    train_btn = mo.ui.run_button(label="Train locally")
    if sys.platform == "emscripten":
        mo.md("_Berjalan di WASM — model pra-terlatih dimuat (akurasi ~97,5 %)._")
    else:
        train_btn
    return (train_btn,)


@app.cell
def _(set_weights, train_btn):
    mo.stop(sys.platform == "emscripten")
    mo.stop(not train_btn.value)
    try:
        import torch
        import torch.nn as nn
    except ImportError:
        mo.stop(
            True,
            mo.md(
                "_torch tidak terinstal — `pip install torch` untuk melatih secara lokal._"
            ),
        )

    _X_train = mnist.data[:60000]
    _y_train = mnist.attributes["digits"][:60000].astype(np.int64)
    _X_test = mnist.data[60000:]
    _y_test = mnist.attributes["digits"][60000:].astype(np.int64)
    _Xtr = torch.tensor(_X_train)
    _ytr = torch.tensor(_y_train)
    _Xte = torch.tensor(_X_test)
    _yte = torch.tensor(_y_test)
    _device = "cuda" if torch.cuda.is_available() else "cpu"
    _Xtr, _ytr = _Xtr.to(_device), _ytr.to(_device)
    _Xte, _yte = _Xte.to(_device), _yte.to(_device)

    _model = nn.Sequential(
        nn.Linear(784, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 10),
    ).to(_device)
    _optimizer_name = OPTIMIZER.strip().lower()
    if _optimizer_name == "adam":
        _opt = torch.optim.Adam(_model.parameters(), lr=LR)
    elif _optimizer_name == "sgd":
        _opt = torch.optim.SGD(_model.parameters(), lr=LR)
    elif _optimizer_name == "rmsprop":
        _opt = torch.optim.RMSprop(_model.parameters(), lr=LR)
    else:
        raise ValueError(f"Unsupported OPTIMIZER: {OPTIMIZER}")

    _loss_name = LOSS_FN.strip().lower().replace(" ", "")
    if _loss_name == "crossentropyloss":
        _loss_fn = nn.CrossEntropyLoss()
    else:
        raise ValueError(f"Unsupported LOSS_FN: {LOSS_FN}")

    _rows = []
    with mo.status.progress_bar(total=EPOCHS, title="Melatih model...") as _bar:
        for _epoch in range(EPOCHS):
            _model.train()
            _epoch_loss, _n_batches = 0.0, 0
            for _i in range(0, len(_Xtr), BATCH_SIZE):
                _xb, _yb = _Xtr[_i : _i + BATCH_SIZE], _ytr[_i : _i + BATCH_SIZE]
                _opt.zero_grad()
                _l = _loss_fn(_model(_xb), _yb)
                _l.backward()
                _opt.step()
                _epoch_loss += _l.item()
                _n_batches += 1
            _model.eval()
            with torch.no_grad():
                _val_loss = _loss_fn(_model(_Xte), _yte).item()
                _acc = (_model(_Xte).argmax(1) == _yte).float().mean().item()
            _rows.append(
                {
                    "Epoch": _epoch + 1,
                    "Training Loss": round(_epoch_loss / _n_batches, 4),
                    "Test Loss": round(_val_loss, 4),
                    "Test Accuracy": round(_acc, 4),
                }
            )
            _bar.update(title=f"epoch {_epoch + 1}/{EPOCHS}  acc={_acc:.4f}")

    _W1 = _model[0].weight.T.detach().cpu().numpy()
    _b1 = _model[0].bias.detach().cpu().numpy()
    _W2 = _model[2].weight.T.detach().cpu().numpy()
    _b2 = _model[2].bias.detach().cpu().numpy()
    _W3 = _model[4].weight.T.detach().cpu().numpy()
    _b3 = _model[4].bias.detach().cpu().numpy()
    set_weights({"W1": _W1, "b1": _b1, "W2": _W2, "b2": _b2, "W3": _W3, "b3": _b3})

    _df = pd.DataFrame(_rows)
    _fig, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(10, 3.5))
    _ax1.plot(_df["Epoch"], _df["Training Loss"], "b-o", label="Training Loss")
    _ax1.plot(_df["Epoch"], _df["Test Loss"], "r-o", label="Test Loss")
    _ax1.set_xlabel("Epoch")
    _ax1.set_ylabel("Loss")
    _ax1.set_title("Loss vs Epoch")
    _ax1.legend()
    _ax1.grid(True, alpha=0.3)
    _ax2.plot(_df["Epoch"], _df["Test Accuracy"], "g-o")
    _ax2.set_xlabel("Epoch")
    _ax2.set_ylabel("Accuracy")
    _ax2.set_title("Test Accuracy vs Epoch")
    _ax2.grid(True, alpha=0.3)
    plt.tight_layout()

    mo.vstack(
        [
            mo.md(
                f"**Training Loss Terakhir:** {_df['Training Loss'].iloc[-1]}  ·  "
                f"**Test Loss Terakhir:** {_df['Test Loss'].iloc[-1]}  ·  "
                f"**Test Accuracy:** {_df['Test Accuracy'].iloc[-1]:.4f}"
            ),
            mo.ui.table(_df),
            mo.as_html(_fig),
        ]
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md("""
    ## Visualisasi Ruang Fitur (Feature Space)

    Setiap titik mewakili satu digit dalam ruang 2D hasil **Minimum Distortion Embedding (MDE)**.
    **Seret untuk memilih klaster** dan lihat gambar-gambar di dalamnya.
    """)
    return


@app.cell
def _():
    embedding = mnist.embedding
    return (embedding,)


@app.cell
def _(embedding):
    _colors = [
        "#e41a1c",
        "#377eb8",
        "#4daf4a",
        "#984ea3",
        "#ff7f00",
        "#a65628",
        "#f781bf",
        "#999999",
        "#dede00",
        "#00bcd4",
    ]
    fig, _ax = plt.subplots(figsize=(8, 6))
    for _d in range(10):
        _m = mnist.attributes["digits"] == _d
        _ax.scatter(
            embedding[_m, 0],
            embedding[_m, 1],
            c=_colors[_d],
            s=1,
            alpha=0.3,
            label=str(_d),
        )
    _ax.legend(markerscale=5, title="Digit")
    ax = mo.ui.matplotlib(_ax)
    ax
    return (ax,)


@app.cell
def _(ax, embedding):
    mask = ax.value.get_mask(embedding[:, 0], embedding[:, 1])
    return (mask,)


@app.cell
def _(df, mask):
    table = mo.ui.table(df[mask])
    return (table,)


@app.cell(hide_code=True)
def _(mask, table):
    mo.stop(not mask.any())
    selected_images = (
        show_images(list(mask.nonzero()[0]))
        if not len(table.value)
        else show_images(list(table.value["index"]))
    )
    mo.md(
        f"""
        **Pratinjau gambar yang dipilih:**

        {mo.as_html(selected_images)}

        Data lengkap seleksi:

        {table}
        """
    )
    return


@app.function
def show_images(indices, max_images=10):
    indices = indices[:max_images]
    images = mnist.data.reshape((-1, 28, 28))[indices]
    fig, axes = plt.subplots(1, len(indices))
    fig.set_size_inches(12.5, 1.5)
    if len(indices) > 1:
        for im, ax in zip(images, axes.flat):
            ax.imshow(im, cmap="gray")
            ax.set_yticks([])
            ax.set_xticks([])
    else:
        axes.imshow(images[0], cmap="gray")
        axes.set_yticks([])
        axes.set_xticks([])
    plt.tight_layout()
    return fig


@app.cell
def _(embedding):
    indices = np.arange(mnist.data.shape[0])
    df = pd.DataFrame(
        {
            "index": indices,
            "x": embedding[:, 0],
            "y": embedding[:, 1],
            "digit": mnist.attributes["digits"][indices],
        }
    )
    return (df,)


@app.cell(hide_code=True)
def _():
    mo.md("""
    ## Demo Interaktif

    Gambar digit di kanvas di bawah — prediksi diperbarui setiap kali Anda melepas mouse.
    Setelah klik *Train locally* di atas, model yang baru terlatih langsung digunakan di sini.

    **Catatan mode Web Live (WASM)**

    1. Di mode web (WASM/`emscripten`), proses training PyTorch tidak dijalankan; aplikasi memuat bobot pra-latih dari `model_weights.npz`.
    2. Inferensi di kanvas memakai NumPy karena dependensi/eksekusi PyTorch di lingkungan WASM belum sefleksibel mode lokal.
    3. Akurasi web live bisa berbeda dari lokal karena web memakai snapshot bobot pra-latih, sedangkan lokal bisa melatih ulang (hasil dipengaruhi inisialisasi dan proses training).
    4. Input gambar dari kanvas bukan distribusi MNIST asli (ketebalan goresan, posisi, dan bentuk digit bisa berbeda), sehingga prediksi per sampel bisa lebih rendah dibanding evaluasi test set.
    5. Hasil NumPy dan PyTorch umumnya sangat dekat, tetapi perbedaan kecil numerik float dan alur pra-pemrosesan tetap bisa menggeser confidence/probabilitas.
    """)
    return


@app.cell
def _():
    canvas_ui = mo.ui.anywidget(DrawWidget())
    canvas_ui
    return (canvas_ui,)


@app.cell(hide_code=True)
def _(canvas_ui, get_weights):
    _px = canvas_ui.value.get("pixels", [])

    if not _px:
        _output = mo.md("_Gambar digit di atas untuk melihat prediksi._")
    else:
        _flat = np.array(_px, dtype=np.float32)
        _w = get_weights()

        def _relu(x):
            return np.maximum(0, x)

        def _softmax(x):
            e = np.exp(x - x.max())
            return e / e.sum()

        _h1 = _relu(_flat @ _w["W1"] + _w["b1"])
        _h2 = _relu(_h1 @ _w["W2"] + _w["b2"])
        _logits = _h2 @ _w["W3"] + _w["b3"]
        _probs = _softmax(_logits)
        _pred = int(_probs.argmax())

        _colors = ["steelblue"] * 10
        _colors[_pred] = "tomato"
        _fig, _ax = plt.subplots(figsize=(3, 4))
        _ax.barh(range(10), _probs, color=_colors)
        _ax.set_yticks(range(10))
        _ax.set_yticklabels([str(i) for i in range(10)])
        _ax.set_xlim(0, 1)
        _ax.set_xlabel("Confidence")
        _ax.set_title(f"Prediksi: {_pred}")
        _ax.invert_yaxis()
        plt.tight_layout()

        _output = mo.vstack(
            [
                mo.md(f"### Prediksi: **{_pred}**"),
                mo.as_html(_fig),
            ]
        )

    _output
    return


if __name__ == "__main__":
    app.run()
