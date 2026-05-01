# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo>=0.23.4",
#     "matplotlib>=3.10.9",
#     "numpy",
#     "pandas>=3.0.0",
#     "pillow>=12.2.0",
#     "pymde>=0.3.0",
#     "torch>=2.0.0",
#     "torchvision>=0.20.0",
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
    import pymde
    import torch
    import torchvision
    import numpy as np
    import anywidget
    import traitlets
    from PIL import Image
    from types import SimpleNamespace

    _ds_train = torchvision.datasets.MNIST(root="data/", train=True, download=True)
    _ds_test  = torchvision.datasets.MNIST(root="data/", train=False, download=True)
    mnist = SimpleNamespace(
        data=torch.cat([_ds_train.data, _ds_test.data]).float().reshape(-1, 784) / 255.0,
        attributes={"digits": torch.cat([_ds_train.targets, _ds_test.targets])},
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
        const d = ctx.getImageData(0, 0, SIZE, SIZE).data;
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
    # MNIST Explorer
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md("""
    ## Feature Space

    Each point is a digit in 2D MDE space. **Drag to select a cluster.**
    """)
    return


@app.function
@mo.persistent_cache
def compute_embedding(embedding_dim, constraint):
    mo.output.append(
        mo.md("Your embedding is being computed ... hang tight!").callout(kind="warn")
    )
    mde = pymde.preserve_neighbors(
        mnist.data,
        embedding_dim=embedding_dim,
        constraint=constraint,
        device="cuda" if torch.cuda.is_available() else "cpu",
        verbose=True,
    )
    X = mde.embed(verbose=True)
    mo.output.clear()
    return X


@app.cell
def _():
    embedding_dimension = 2
    constraint = pymde.Standardized()
    return constraint, embedding_dimension


@app.cell
def _(constraint, embedding_dimension):
    embedding = compute_embedding(embedding_dimension, constraint)
    return (embedding,)


@app.cell
def _(embedding):
    ax = pymde.plot(embedding, color_by=mnist.attributes["digits"])
    ax = mo.ui.matplotlib(ax)
    ax
    return (ax,)


@app.cell
def _(ax, embedding):
    mask = ax.value.get_mask(embedding[:, 0].cpu(), embedding[:, 1].cpu())
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
    indices = torch.arange(mnist.data.shape[0]).numpy()
    df = pd.DataFrame(
        {
            "index": indices,
            "x": embedding[:, 0].cpu().numpy(),
            "y": embedding[:, 1].cpu().numpy(),
            "digit": mnist.attributes["digits"][indices].cpu().numpy(),
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
    canvas_ui = DrawWidget()
    canvas_ui
    return (canvas_ui,)


@app.cell(hide_code=True)
def _(canvas_ui):
    _px = canvas_ui.pixels

    if not _px:
        _output = mo.md("_Draw a digit above to see predictions._")
    else:
        _arr = np.array(_px, dtype=np.float32).reshape(200, 200)
        _img = Image.fromarray((_arr * 255).astype(np.uint8), "L").resize(
            (28, 28), Image.Resampling.LANCZOS
        )
        _flat = np.array(_img, dtype=np.float32).ravel() / 255.0

        _X = mnist.data.float().reshape(-1, 784)[:10_000]
        _y = mnist.attributes["digits"][:10_000]
        _q = torch.tensor(_flat).unsqueeze(0)
        _, _idx = torch.topk(torch.cdist(_q, _X)[0], k=15, largest=False)
        _probs = torch.bincount(_y[_idx].long(), minlength=10).float() / 15
        _pred = int(_probs.argmax())

        _colors = ["steelblue"] * 10
        _colors[_pred] = "tomato"
        _fig, _ax = plt.subplots(figsize=(3, 4))
        _ax.barh(range(10), _probs.numpy(), color=_colors)
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
