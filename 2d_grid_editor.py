import tkinter as tk
from tkinter import ttk, messagebox
import math

# ---------- Safe expression evaluation ----------

SAFE_FUNCS = {
    # builtins
    "abs": abs, "min": min, "max": max, "round": round,
    # math
    "sqrt": math.sqrt, "sin": math.sin, "cos": math.cos, "tan": math.tan,
    "asin": math.asin, "acos": math.acos, "atan": math.atan, "atan2": math.atan2,
    "sinh": math.sinh, "cosh": math.cosh, "tanh": math.tanh,
    "log": math.log, "log10": math.log10, "exp": math.exp,
    "floor": math.floor, "ceil": math.ceil, "fabs": math.fabs,
    "pi": math.pi, "e": math.e, "tau": math.tau,
    # helpers
    "sgn": lambda v: (v > 0) - (v < 0),
    "clamp": lambda v, lo, hi: lo if v < lo else (hi if v > hi else v),
}

def eval_expr(expr, x, y, w, h):
    """Evaluate expression with a small, safe namespace."""
    env = dict(SAFE_FUNCS)
    env.update({"x": x, "y": y, "w": w, "h": h})
    return eval(expr, {"__builtins__": {}}, env)

# ---------- Color mapping ----------

def lerp(a, b, t):
    return a + (b - a) * t

def val_to_color(v, vmin, vmax):
    """Map value to a perceptible gradient (purple -> teal -> yellow)."""
    if vmin == vmax:
        t = 0.5
    else:
        t = (v - vmin) / (vmax - vmin)
        t = 0.0 if t < 0 else (1.0 if t > 1 else t)
    # three-point blend: 0: purple (76,0,92), 0.5: teal (0,150,136), 1: yellow (255,214,0)
    if t < 0.5:
        t2 = t / 0.5
        r = int(lerp(76, 0, t2))
        g = int(lerp(0, 150, t2))
        b = int(lerp(92, 136, t2))
    else:
        t2 = (t - 0.5) / 0.5
        r = int(lerp(0, 255, t2))
        g = int(lerp(150, 214, t2))
        b = int(lerp(136, 0, t2))
    return f"#{r:02x}{g:02x}{b:02x}"

# ---------- App ----------

class GridEditor(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Tk Grid Function Visualizer")
        self.geometry("1000x720")

        # Controls
        top = ttk.Frame(self, padding=8)
        top.pack(side=tk.TOP, fill=tk.X)

        self.expr_var = tk.StringVar(value="(abs(x - w//2) + abs(y - h//2))")
        self.w_var = tk.IntVar(value=42)
        self.h_var = tk.IntVar(value=42)
        self.cell_var = tk.IntVar(value=24)
        self.show_text_var = tk.BooleanVar(value=True)

        ttk.Label(top, text="f(x,y) =").pack(side=tk.LEFT)
        expr_entry = ttk.Entry(top, textvariable=self.expr_var, width=60)
        expr_entry.pack(side=tk.LEFT, padx=(4, 10), fill=tk.X, expand=True)
        expr_entry.bind("<Return>", lambda e: self.render())

        ttk.Label(top, text="W").pack(side=tk.LEFT)
        ttk.Entry(top, textvariable=self.w_var, width=5).pack(side=tk.LEFT, padx=(2, 8))
        ttk.Label(top, text="H").pack(side=tk.LEFT)
        ttk.Entry(top, textvariable=self.h_var, width=5).pack(side=tk.LEFT, padx=(2, 8))
        ttk.Label(top, text="Cell").pack(side=tk.LEFT)
        ttk.Entry(top, textvariable=self.cell_var, width=4).pack(side=tk.LEFT, padx=(2, 8))

        ttk.Checkbutton(top, text="Show values", variable=self.show_text_var,
                        command=self.render).pack(side=tk.LEFT, padx=8)

        ttk.Button(top, text="Render (Enter)", command=self.render).pack(side=tk.LEFT, padx=6)

        # Canvas with scrollbars
        self.canvas_frame = ttk.Frame(self)
        self.canvas_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(self.canvas_frame, background="#202124", highlightthickness=0)
        self.hbar = ttk.Scrollbar(self.canvas_frame, orient=tk.HORIZONTAL, command=self.canvas.xview)
        self.vbar = ttk.Scrollbar(self.canvas_frame, orient=tk.VERTICAL, command=self.canvas.yview)
        self.canvas.configure(xscrollcommand=self.hbar.set, yscrollcommand=self.vbar.set)

        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.vbar.grid(row=0, column=1, sticky="ns")
        self.hbar.grid(row=1, column=0, sticky="ew")
        self.canvas_frame.rowconfigure(0, weight=1)
        self.canvas_frame.columnconfigure(0, weight=1)

        # Status
        self.status = ttk.Label(self, anchor="w")
        self.status.pack(side=tk.BOTTOM, fill=tk.X, padx=8, pady=4)

        # Events
        self.canvas.bind("<Button-1>", self.on_click)
        self.canvas.bind("<Motion>", self.on_motion)

        # Data cache
        self.values = []
        self.vmin = 0
        self.vmax = 1

        self.render()

    # ----- Rendering -----

    def compute_values(self):
        w = max(1, int(self.w_var.get()))
        h = max(1, int(self.h_var.get()))
        expr = self.expr_var.get().strip()
        vals = [[0 for _ in range(w)] for _ in range(h)]
        vmin, vmax = float("inf"), float("-inf")
        for y in range(h):
            for x in range(w):
                try:
                    v = eval_expr(expr, x, y, w, h)
                except Exception as e:
                    raise RuntimeError(f"Error at (x={x}, y={y}): {e}")
                # Coerce to number
                try:
                    v = float(v)
                except Exception:
                    raise RuntimeError(f"Expression must return a number; got {type(v)} at (x={x}, y={y})")
                vals[y][x] = v
                if v < vmin: vmin = v
                if v > vmax: vmax = v
        if vmin == float("inf"):
            vmin, vmax = 0.0, 1.0
        return vals, vmin, vmax

    def render(self):
        try:
            self.values, self.vmin, self.vmax = self.compute_values()
        except Exception as e:
            messagebox.showerror("Evaluation error", str(e))
            return

        h = len(self.values)
        w = len(self.values[0]) if h else 0
        cell = max(6, int(self.cell_var.get()))

        self.canvas.delete("all")
        total_w = w * cell
        total_h = h * cell
        self.canvas.config(scrollregion=(0, 0, total_w, total_h))

        show_text = self.show_text_var.get()
        font = ("TkDefaultFont", max(7, int(cell * 0.38)))

        for yy in range(h):
            y0 = yy * cell
            y1 = y0 + cell
            for xx in range(w):
                x0 = xx * cell
                x1 = x0 + cell
                v = self.values[yy][xx]
                fill = val_to_color(v, self.vmin, self.vmax)
                self.canvas.create_rectangle(x0, y0, x1, y1, fill=fill, width=1, outline="#1b1d20")
                if show_text and cell >= 14:
                    self.canvas.create_text((x0+x1)//2, (y0+y1)//2,
                                            text=f"{v:.0f}" if abs(v) >= 1 else f"{v:.2f}",
                                            font=font, fill="#f1f3f4")

        self.status.config(text=f"Rendered {w}×{h} | min={self.vmin:.4g} max={self.vmax:.4g}")

    # ----- Interaction -----

    def xy_from_event(self, event):
        cell = max(6, int(self.cell_var.get()))
        x = int(self.canvas.canvasx(event.x) // cell)
        y = int(self.canvas.canvasy(event.y) // cell)
        w = len(self.values[0]) if self.values else 0
        h = len(self.values)
        if 0 <= x < w and 0 <= y < h:
            return x, y
        return None

    def on_click(self, event):
        pos = self.xy_from_event(event)
        if pos is None: return
        x, y = pos
        v = self.values[y][x]
        self.status.config(text=f"(x={x}, y={y}) -> {v}")

    def on_motion(self, event):
        pos = self.xy_from_event(event)
        if pos is None:
            return
        x, y = pos
        v = self.values[y][x]
        self.status.config(text=f"Hover (x={x}, y={y}) -> {v}")

if __name__ == "__main__":
    app = GridEditor()

    # Helpful starter expressions you can try:
    #  - "(abs(x - w//2) + abs(y - h//2))"          # Manhattan distance to center
    #  - "((x - w/2)**2 + (y - h/2)**2)**0.5"        # Euclidean distance to center
    #  - "(x + y)"                                    # simple slope
    #  - "(min(abs(x), w-abs(x-w)) + min(abs(y), h-abs(y-h)))"  # torus Manhattan
    #  - "(sin(x/6)+cos(y/7))*10"
    app.mainloop()
