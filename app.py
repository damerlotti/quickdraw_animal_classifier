"""Draw an animal and let the model guess it.

Run from the project's top folder:  python app.py
"""
import tkinter as tk

import numpy as np

from src.data import animals, load_data
from src.models import KNN, sq_dists
from src.pca import PCA

CANVAS_SIZE = 400       # drawing area in screen pixels
PREVIEW_SCALE = 5       # the 28x28 preview is shown at 5x size
N_COMPONENTS = 40
K = 20


# ---------- turning a drawing into a 28x28 image like the dataset ----------

def stamp(img, x, y, r):
    """Paint a filled circle of radius r at (x, y)."""
    h, w = img.shape
    x0, x1 = max(int(x - r), 0), min(int(x + r) + 2, w)
    y0, y1 = max(int(y - r), 0), min(int(y + r) + 2, h)
    if x0 >= x1 or y0 >= y1:
        return
    yy, xx = np.ogrid[y0:y1, x0:x1]
    img[y0:y1, x0:x1][(xx - x) ** 2 + (yy - y) ** 2 <= r * r] = 1.0


def strokes_to_image(strokes, size=28, scale=10, margin=0.1, radius=9):
    """Crop to the drawing, scale it to fill the frame, center it, shrink to 28x28."""
    strokes = [np.array(s, dtype=float) for s in strokes if len(s) > 0]
    if not strokes:
        return np.zeros(size * size, dtype=np.float32)

    # bounding box of all points
    pts = np.vstack(strokes)
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    extent = max((hi - lo).max(), 1.0)

    # draw on a 280x280 grid (10x the final size), drawing scaled to fit and centered
    big = size * scale
    box = big * (1 - 2 * margin)
    offset = (big - (hi - lo) / extent * box) / 2
    img = np.zeros((big, big), dtype=np.float32)
    for s in strokes:
        s = (s - lo) / extent * box + offset
        if len(s) == 1:
            s = np.vstack([s, s])
        for (xa, ya), (xb, yb) in zip(s[:-1], s[1:]):
            n = int(max(abs(xb - xa), abs(yb - ya)) / 2) + 1
            for t in np.linspace(0, 1, n + 1):
                stamp(img, xa + t * (xb - xa), ya + t * (yb - ya), radius)

    # average each 10x10 block into one pixel -> 28x28 with soft edges
    small = img.reshape(size, scale, size, scale).mean(axis=(1, 3))
    return (small / small.max()).reshape(-1).astype(np.float32)


# ---------- the app ----------

class DrawingApp:
    def __init__(self, root, pca, knn):
        self.pca, self.knn = pca, knn
        self.strokes = []

        root.title("Draw an animal")
        tk.Label(root, text="Draw one of: " + ", ".join(animals),
                 wraplength=CANVAS_SIZE + 160).pack(pady=5)

        row = tk.Frame(root)
        row.pack()
        self.canvas = tk.Canvas(row, width=CANVAS_SIZE, height=CANVAS_SIZE,
                                bg="white", cursor="pencil")
        self.canvas.pack(side=tk.LEFT, padx=5)

        side = tk.Frame(row)
        side.pack(side=tk.LEFT, padx=5, anchor="n")
        tk.Label(side, text="What the model sees:").pack()
        p = 28 * PREVIEW_SCALE
        self.preview = tk.Canvas(side, width=p, height=p, bg="white")
        self.preview.pack(pady=5)
        self.result = tk.Label(side, text="", font=("Arial", 13), justify=tk.LEFT)
        self.result.pack(pady=10, anchor="w")

        tk.Button(root, text="Clear", command=self.clear, width=12).pack(pady=8)

        self.canvas.bind("<ButtonPress-1>", self.start_stroke)
        self.canvas.bind("<B1-Motion>", self.draw)
        self.canvas.bind("<ButtonRelease-1>", lambda e: self.guess())

    def start_stroke(self, event):
        self.strokes.append([(event.x, event.y)])

    def draw(self, event):
        x0, y0 = self.strokes[-1][-1]
        self.canvas.create_line(x0, y0, event.x, event.y, width=6,
                                capstyle=tk.ROUND, smooth=True)
        self.strokes[-1].append((event.x, event.y))

    def clear(self):
        self.strokes = []
        self.canvas.delete("all")
        self.preview.delete("all")
        self.result.config(text="")

    def guess(self):
        img = strokes_to_image(self.strokes)
        self.show_preview(img)

        # same steps as in training: PCA, then the k nearest neighbours vote
        z = self.pca.transform(img[None, :])
        d = sq_dists(z, self.knn.X)[0]
        nearest = np.argsort(d)[:self.knn.k]
        votes = np.bincount(self.knn.y[nearest], minlength=len(animals))

        top = np.argsort(votes)[::-1][:3]
        lines = [f"{animals[c]}: {votes[c] / self.knn.k:.0%}" for c in top if votes[c] > 0]
        self.result.config(text="I think it's a...\n\n" + "\n".join(lines))

    def show_preview(self, img):
        self.preview.delete("all")
        s = PREVIEW_SCALE
        for i, v in enumerate(img):
            if v > 0.05:
                r, c = divmod(i, 28)
                gray = int(255 * (1 - v))
                color = f"#{gray:02x}{gray:02x}{gray:02x}"
                self.preview.create_rectangle(c * s, r * s, (c + 1) * s, (r + 1) * s,
                                              fill=color, outline="")


if __name__ == "__main__":
    print("Loading data and training the model...")
    X, y = load_data(animals)
    pca = PCA(N_COMPONENTS).fit(X)
    knn = KNN(k=K).fit(pca.transform(X), y)
    print("Ready! Draw in the window.")

    root = tk.Tk()
    DrawingApp(root, pca, knn)
    root.mainloop()