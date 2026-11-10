
import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parent
DATA, CORR = ROOT / "data", ROOT / "corr"


def load(path, reduce=4):
    im = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    full = im.size
    return np.asarray(im.reduce(reduce)), full


class Picker:
    def __init__(self, path1, path2, out):
        self.out = out
        self.pts = [[], []]
        self.artists = []
        self.saved = False
        self.fig, self.axes = plt.subplots(1, 2, figsize=(16, 6))
        for ax, p in zip(self.axes, (path1, path2)):
            im, (w, h) = load(p)
            ax.imshow(im, extent=(-0.5, w - 0.5, h - 0.5, -0.5))  # clicks come back in full-res pixels
            ax.set_title(Path(p).name)
            ax.set_xticks([]); ax.set_yticks([])
        self.colors = plt.cm.tab20(np.linspace(0, 1, 20))
        self.fig.canvas.mpl_connect("button_press_event", self.on_click)
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        self.status()

    @property
    def turn(self):  # 0 = expecting a click on the left image, 1 = right
        return 0 if len(self.pts[0]) == len(self.pts[1]) else 1

    def status(self):
        n = len(self.pts[1])
        self.fig.suptitle("%d pairs  |  click on the %s image  |  u: undo   enter: save   esc: quit"
                          % (n, "LEFT" if self.turn == 0 else "RIGHT"))
        self.fig.canvas.draw_idle()

    def on_click(self, ev):
        if ev.inaxes is None or self.fig.canvas.toolbar.mode or ev.button != 1:
            return
        side = list(self.axes).index(ev.inaxes)
        if side != self.turn:
            return
        i = len(self.pts[side])
        self.pts[side].append((ev.xdata, ev.ydata))
        c = self.colors[i % 20]
        dot = ev.inaxes.plot(ev.xdata, ev.ydata, "o", ms=7, mfc=c, mec="white", mew=1.2)[0]
        txt = ev.inaxes.annotate(str(i), (ev.xdata, ev.ydata), xytext=(5, 5), textcoords="offset points",
                                 color="white", fontweight="bold", bbox=dict(boxstyle="round,pad=0.15", fc=c, ec="none"))
        self.artists.append((dot, txt))
        print("pair %d %s: (%.1f, %.1f)" % (i, "L" if side == 0 else "R", ev.xdata, ev.ydata))
        self.status()

    def on_key(self, ev):
        if ev.key == "u" and self.artists:
            # the last click is on the right if lengths are equal (and >0), else on the left
            side = 1 if len(self.pts[0]) == len(self.pts[1]) else 0
            self.pts[side].pop()
            for a in self.artists.pop():
                a.remove()
            self.status()
        elif ev.key == "enter":
            n = min(len(self.pts[0]), len(self.pts[1]))
            if n < 4:
                print("need at least 4 complete pairs (have %d)" % n); return
            self.out.parent.mkdir(exist_ok=True)
            np.savez(self.out, im1_pts=np.array(self.pts[0][:n]), im2_pts=np.array(self.pts[1][:n]))
            self.saved = True
            print("saved %d pairs to %s" % (n, self.out))
            plt.close(self.fig)
        elif ev.key == "escape":
            plt.close(self.fig)


def pick(name1, name2):
    out = CORR / ("%s_%s.npz" % (Path(name1).stem, Path(name2).stem))
    p = Picker(DATA / name1, DATA / name2, out)
    plt.show()
    return p.saved


class RectPicker:
    """Click the 4 corners of a known rectangle in order: top-left, top-right, bottom-right, bottom-left."""
    LABELS = ["TL", "TR", "BR", "BL"]

    def __init__(self, path, out):
        self.out, self.pts, self.artists, self.saved = out, [], [], False
        self.fig, self.ax = plt.subplots(figsize=(14, 8))
        im, (w, h) = load(path)
        self.ax.imshow(im, extent=(-0.5, w - 0.5, h - 0.5, -0.5))
        self.ax.set_xticks([]); self.ax.set_yticks([])
        self.fig.canvas.mpl_connect("button_press_event", self.on_click)
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        self.status()

    def status(self):
        nxt = self.LABELS[len(self.pts)] if len(self.pts) < 4 else "done"
        self.fig.suptitle("rectangle corners: next = %s  |  u: undo   enter: save   esc: quit" % nxt)
        self.fig.canvas.draw_idle()

    def on_click(self, ev):
        if ev.inaxes is None or self.fig.canvas.toolbar.mode or ev.button != 1 or len(self.pts) >= 4:
            return
        self.pts.append((ev.xdata, ev.ydata))
        dot = self.ax.plot(ev.xdata, ev.ydata, "o", ms=7, mfc="yellow", mec="black")[0]
        txt = self.ax.annotate(self.LABELS[len(self.pts) - 1], (ev.xdata, ev.ydata), xytext=(5, 5),
                               textcoords="offset points", color="yellow", fontweight="bold")
        self.artists.append((dot, txt)); self.status()

    def on_key(self, ev):
        if ev.key == "u" and self.pts:
            self.pts.pop()
            for a in self.artists.pop():
                a.remove()
            self.status()
        elif ev.key == "enter" and len(self.pts) == 4:
            self.out.parent.mkdir(exist_ok=True)
            if self.out.exists():  # keep earlier picks
                import shutil, time
                (self.out.parent / "raw").mkdir(exist_ok=True)
                shutil.move(self.out, self.out.parent / "raw" / (self.out.stem + time.strftime("_%H%M%S.npz")))
            np.savez(self.out, corners=np.array(self.pts)); self.saved = True
            print("saved corners to %s" % self.out); plt.close(self.fig)
        elif ev.key == "escape":
            plt.close(self.fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="*", help="two image names in data/")
    ap.add_argument("--set", type=int, help="pick all consecutive pairs of a photo set (1 or 2)")
    ap.add_argument("--redo", action="store_true", help="re-pick pairs that already have a file")
    ap.add_argument("--rect", nargs="+", help="image name(s) to click 4 rectangle corners on, for rectification")
    a = ap.parse_args()
    if a.rect:
        for name in a.rect:
            p = RectPicker(DATA / name, CORR / ("rect_%s.npz" % Path(name).stem)); plt.show()
        return
    if a.set == 1:
        names = ["%d.JPG" % i for i in range(1, 8)]
    elif a.set == 2:
        names = ["2-%d.JPG" % i for i in range(1, 6)]
    else:
        names = a.images
    for n1, n2 in zip(names, names[1:]):
        out = CORR / ("%s_%s.npz" % (Path(n1).stem, Path(n2).stem))
        if out.exists() and not a.redo:
            print("skipping %s (exists)" % out.name); continue
        print("\n=== %s  <->  %s ===" % (n1, n2))
        if not pick(n1, n2):
            print("not saved; stopping"); break


if __name__ == "__main__":
    main()
