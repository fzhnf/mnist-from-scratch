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
    import marimo as mo
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np
    import anywidget
    import traitlets
    from types import SimpleNamespace
    import sys, io

    _REPO = "https://raw.githubusercontent.com/fzhnf/mnist-from-scratch/main/"

    if sys.platform == "emscripten":
        import urllib.request
        _d = np.load(io.BytesIO(urllib.request.urlopen(_REPO + "mnist_data.npz").read()))
        _emb = np.load(io.BytesIO(urllib.request.urlopen(_REPO + "embedding.npy").read()))
        _w = np.load(io.BytesIO(urllib.request.urlopen(_REPO + "model_weights.npz").read()))
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
        model.set("pixels", out);   // 784 floats instead of 40,000
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
    # MNIST from Scratch
    """)
    return


@app.cell
def _():
    get_weights, set_weights = mo.state(mnist.weights)
    return get_weights, set_weights


@app.cell(hide_code=True)
def _():
    mo.md("""
    ## Training

    3-layer MLP — **784 → 256 → 128 → 10** — trained with Adam for 10 epochs.
    In WASM the model is pre-trained; click the button below to retrain locally.
    """)
    return


@app.cell(hide_code=True)
def _():
    train_btn = mo.ui.run_button(label="Train locally")
    if sys.platform == "emscripten":
        mo.md("_Running in WASM — pre-trained weights loaded (97.5 % test accuracy)._")
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
        mo.stop(True, mo.md("_torch not installed — `pip install torch` to enable training._"))

    _X_train = mnist.data[:60000]
    _y_train = mnist.attributes["digits"][:60000].astype(np.int64)
    _X_test  = mnist.data[60000:]
    _y_test  = mnist.attributes["digits"][60000:].astype(np.int64)
    _Xtr = torch.tensor(_X_train)
    _ytr = torch.tensor(_y_train)
    _Xte = torch.tensor(_X_test)
    _yte = torch.tensor(_y_test)
    _device = "cuda" if torch.cuda.is_available() else "cpu"
    _Xtr, _ytr = _Xtr.to(_device), _ytr.to(_device)
    _Xte, _yte = _Xte.to(_device), _yte.to(_device)

    _model = nn.Sequential(
        nn.Linear(784, 256), nn.ReLU(),
        nn.Linear(256, 128), nn.ReLU(),
        nn.Linear(128, 10),
    ).to(_device)
    _opt = torch.optim.Adam(_model.parameters(), lr=1e-3)
    _loss_fn = nn.CrossEntropyLoss()

    _rows = []
    with mo.status.progress_bar(total=10, title="Training") as _bar:
        for _epoch in range(10):
            _model.train()
            for _i in range(0, len(_Xtr), 256):
                _xb, _yb = _Xtr[_i:_i+256], _ytr[_i:_i+256]
                _opt.zero_grad()
                _loss_fn(_model(_xb), _yb).backward()
                _opt.step()
            _model.eval()
            with torch.no_grad():
                _acc = (_model(_Xte).argmax(1) == _yte).float().mean().item()
            _rows.append({"epoch": _epoch + 1, "test_accuracy": round(_acc, 4)})
            _bar.update(title=f"epoch {_epoch + 1}/10  acc={_acc:.4f}")

    _W1 = _model[0].weight.T.detach().cpu().numpy()
    _b1 = _model[0].bias.detach().cpu().numpy()
    _W2 = _model[2].weight.T.detach().cpu().numpy()
    _b2 = _model[2].bias.detach().cpu().numpy()
    _W3 = _model[4].weight.T.detach().cpu().numpy()
    _b3 = _model[4].bias.detach().cpu().numpy()
    set_weights({"W1": _W1, "b1": _b1, "W2": _W2, "b2": _b2, "W3": _W3, "b3": _b3})

    mo.ui.table(pd.DataFrame(_rows))
    return


@app.cell(hide_code=True)
def _():
    mo.md("""
    ## Feature Space

    Each point is a digit in 2D MDE space. **Drag to select a cluster.**
    """)
    return


@app.cell
def _():
    embedding = mnist.embedding
    return (embedding,)


@app.cell
def _(embedding):
    _colors = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00",
               "#a65628", "#f781bf", "#999999", "#dede00", "#00bcd4"]
    fig, _ax = plt.subplots(figsize=(8, 6))
    for _d in range(10):
        _m = mnist.attributes["digits"] == _d
        _ax.scatter(embedding[_m, 0], embedding[_m, 1],
                    c=_colors[_d], s=1, alpha=0.3, label=str(_d))
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
        **Here's a preview of the images you've selected**:

        {mo.as_html(selected_images)}

        Here's all the data you've selected.

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
    ## Draw a Digit

    Draw below — the prediction updates when you release the mouse.
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
        _output = mo.md("_Draw a digit above to see predictions._")
    else:
        _flat = np.array(_px, dtype=np.float32)  # 784 floats, already resized in JS
        _w = get_weights()

        def _relu(x): return np.maximum(0, x)
        def _softmax(x): e = np.exp(x - x.max()); return e / e.sum()

        _h1 = _relu(_flat @ _w["W1"] + _w["b1"])
        _h2 = _relu(_h1   @ _w["W2"] + _w["b2"])
        _logits = _h2     @ _w["W3"] + _w["b3"]
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
        _ax.set_title(f"Prediction: {_pred}")
        _ax.invert_yaxis()
        plt.tight_layout()

        _output = mo.vstack([
            mo.md(f"### Prediction: **{_pred}**"),
            mo.as_html(_fig),
        ])

    _output
    return


if __name__ == "__main__":
    app.run()
