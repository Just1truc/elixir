import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import json
import math
import re

import ast

BASE_VARS = {"x", "y", "w", "h", "o", "cx", "cy"}

def extract_names(expr: str) -> set:
    """Return identifiers used in an expression (ignores literals)."""
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as e:
        raise RuntimeError(f"Syntax error: {e}")
    return {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}


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

IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

def make_base_env(x, y, w, h, o, cx, cy):
    env = dict(SAFE_FUNCS)
    env.update({"x": x, "y": y, "w": w, "h": h, "o": o, "cx": cx, "cy": cy})
    return env

def eval_with_env(expr, env):
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
        self.geometry("1180x760")

        # ---------- Top bar ----------
        top = ttk.Frame(self, padding=8)
        top.pack(side=tk.TOP, fill=tk.X)

        self.expr_var = tk.StringVar(value="abs(x) + abs(y)")
        self.w_var = tk.IntVar(value=42)
        self.h_var = tk.IntVar(value=42)
        self.cell_var = tk.IntVar(value=24)
        self.orientation = tk.IntVar(value=1)
        self.center_x = tk.IntVar(value=21)
        self.center_y = tk.IntVar(value=21)
        self.show_text_var = tk.BooleanVar(value=True)

        ttk.Label(top, text="f(x,y) =").pack(side=tk.LEFT)
        expr_entry = ttk.Entry(top, textvariable=self.expr_var, width=64)
        expr_entry.pack(side=tk.LEFT, padx=(4, 10), fill=tk.X, expand=True)
        expr_entry.bind("<Return>", lambda e: self.render())

        for label, var, width in [
            ("W", self.w_var, 6), ("H", self.h_var, 6),
            ("Orientation", self.orientation, 6),
            ("Pos_x", self.center_x, 6), ("Pos_y", self.center_y, 6),
            ("Cell", self.cell_var, 6)
        ]:
            ttk.Label(top, text=label).pack(side=tk.LEFT)
            ent = ttk.Entry(top, textvariable=var, width=width)
            ent.pack(side=tk.LEFT, padx=(2, 8))
            ent.bind("<Return>", lambda e: self.render())

        ttk.Checkbutton(top, text="Show values", variable=self.show_text_var,
                        command=self.render).pack(side=tk.LEFT, padx=8)

        ttk.Button(top, text="Render (Enter)", command=self.render).pack(side=tk.LEFT, padx=6)
        
        self.auto_center = tk.BooleanVar(value=True)
        ttk.Checkbutton(top, text="Auto-center", variable=self.auto_center).pack(side=tk.LEFT, padx=8)
        ttk.Button(top, text="Center", command=lambda: self.center_canvas()).pack(side=tk.LEFT, padx=4)

        # ---------- Toolbar ----------
        toolbar = ttk.Frame(self, padding=(8, 4))
        toolbar.pack(side=tk.TOP, fill=tk.X)

        ttk.Button(toolbar, text="+ Add field", command=self.add_param_row).pack(side=tk.LEFT)
        ttk.Button(toolbar, text="Save config", command=self.save_config).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Button(toolbar, text="Load config", command=self.load_config).pack(side=tk.LEFT, padx=(6, 0))
        ttk.Label(toolbar, text="Fields are expressions evaluated per cell. Order matters.").pack(side=tk.LEFT, padx=12)

        ttk.Button(toolbar, text="Export function", command=self.export_function).pack(side=tk.LEFT, padx=(8, 0))

        # ---------- Parameters panel ----------

        self.params_wrap = ttk.LabelFrame(self, text="Fields (expressions available inside f)", padding=8)
        self.params_wrap.pack(side=tk.TOP, fill=tk.X, padx=8, pady=(0, 8))
        self.params_container = ttk.Frame(self.params_wrap)
        self.params_container.pack(side=tk.TOP, fill=tk.X)
        self.param_rows = []  # dicts: {frame,name_var,type_var,value_var}
                
        self.show_fields = tk.BooleanVar(value=True)  # track visibility
        self.fields_toggle_btn = ttk.Button(toolbar, text="Hide variables", command=self._toggle_fields)
        self.fields_toggle_btn.pack(side=tk.LEFT, padx=(8, 0))

        # optional keyboard shortcut
        self.bind("<F2>", lambda e: self._toggle_fields())

        # Seed examples
        # self.add_param_row(("a", "float", "1.0"))
        # self.add_param_row(("freq", "float", "0.25"))
        # Examples showing dependency and toroidal helpers (optional)
        self.add_param_row(("tor_x", "int", "(abs(x) % (w//2)) if (abs(x) % w) < (w//2) else (w//2 - (abs(x) % (w//2)))"))
        self.add_param_row(("tor_y", "int", "(abs(y) % (h//2)) if (abs(y) % h) < (h//2) else (h//2 - (abs(y) % (h//2)))"))
        # self.add_param_row(("d", "float", "abs(tor_x) + abs(tor_y)"))

        # ---------- Canvas ----------
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

        # ---------- Status ----------
        self.status = ttk.Label(self, anchor="w")
        self.status.pack(side=tk.BOTTOM, fill=tk.X, padx=8, pady=4)

        # ---------- Events ----------
        self.canvas.bind("<Button-1>", self.on_click)
        self.canvas.bind("<Motion>", self.on_motion)
        # Mouse wheel + drag-to-pan
        self.canvas.bind("<Enter>", lambda e: self._bind_mousewheel())
        self.canvas.bind("<Leave>", lambda e: self._unbind_mousewheel())

        # Middle-button drag (press and move)
        self.canvas.bind("<ButtonPress-2>", self._pan_mark)
        self.canvas.bind("<B2-Motion>", self._pan_drag)
        # (on some mice, button 2 is the wheel click; on Macs with trackpad, two-finger drag + middle click emulation also works)

        self.bind("<Configure>", lambda e: (
            self._update_scrollregion_centered(
                max(1, len(self.values[0]) * max(6, int(self.cell_var.get()))) if self.values else 1,
                max(1, len(self.values)    * max(6, int(self.cell_var.get())))
            ),
            self.auto_center.get() and self.after_idle(self.center_canvas)
        ))

        
        # ---------- Data cache ----------
        self.values = []
        self.vmin = 0
        self.vmax = 1

        self.render()
        
    # def export_function(self):
    #     """Export a Python file containing grid_value(x, y) using current UI state."""
    #     try:
    #         # --- Gather + plan fields (uses your existing helpers) ---
    #         params = self._gather_params()
    #         plan   = self._plan_params(params)
    #         meta   = plan["meta"]

    #         # Current fixed params
    #         w  = max(1, int(self.w_var.get()))
    #         h  = max(1, int(self.h_var.get()))
    #         o  = int(self.orientation.get())
    #         cx = int(self.center_x.get())
    #         cy = int(self.center_y.get())
    #         f_expr = self.expr_var.get().strip()

    #         # --- Precompute constant fields once ---
    #         const_vals = {}
    #         base_for_const = make_base_env(0, 0, w, h, o, cx, cy)
    #         for name in plan["const_order"]:
    #             m = meta[name]
    #             val = eval_with_env(m["expr"], {**base_for_const, **const_vals})
    #             if m["type"] == "float":
    #                 val = float(val)
    #             elif m["type"] == "int":
    #                 val = int(val)
    #             elif m["type"] == "bool":
    #                 val = bool(val)
    #             const_vals[name] = val

    #         # --- Build source code ---
    #         lines = []
    #         lines.append("# Auto-generated by GridEditor — grid_value(x, y)\n")
    #         lines.append("from math import sqrt, sin, cos, tan, asin, acos, atan, atan2, sinh, cosh, tanh, log, log10, exp, floor, ceil, fabs, pi, e, tau\n")
    #         lines.append("\n")
    #         lines.append("def sgn(v):\n    return (v > 0) - (v < 0)\n\n")
    #         lines.append("def clamp(v, lo, hi):\n    return lo if v < lo else (hi if v > hi else v)\n\n")
    #         lines.append("def grid_value(x, y):\n")
    #         indent = "    "
    #         # Fixed grid params
    #         lines.append(f"{indent}# fixed grid parameters (baked from UI)\n")
    #         lines.append(f"{indent}w = {w}\n{indent}h = {h}\n{indent}o = {o}\n{indent}cx = {cx}\n{indent}cy = {cy}\n\n")
    #         # Relative coords (match the renderer)
    #         lines.append(f"{indent}# use coordinates relative to center, like the visualizer\n")
    #         lines.append(f"{indent}x = x - cx\n{indent}y = y - cy\n\n")
    #         # Constants
    #         if plan["const_order"]:
    #             lines.append(f"{indent}# constant fields (precomputed)\n")
    #             for name in plan["const_order"]:
    #                 lines.append(f"{indent}{name} = {repr(const_vals[name])}\n")
    #             lines.append("\n")
    #         # Per-cell fields
    #         if plan["cell_order"]:
    #             lines.append(f"{indent}# per-cell fields (evaluated in dependency order)\n")
    #             for name in plan["cell_order"]:
    #                 expr = meta[name]["expr"]
    #                 lines.append(f"{indent}{name} = {expr}\n")
    #             lines.append("\n")
    #         # Final expression
    #         lines.append(f"{indent}return float({f_expr})\n")

    #         src = "".join(lines)

    #         # --- Save file ---
    #         from tkinter import filedialog
    #         fpath = filedialog.asksaveasfilename(
    #             title="Export Python function",
    #             defaultextension=".py",
    #             filetypes=[("Python files", "*.py"), ("All files", "*.*")]
    #         )
    #         if not fpath:
    #             return
    #         with open(fpath, "w", encoding="utf-8") as f:
    #             f.write(src)
    #         self.status.config(text=f"Exported function to '{fpath}'")
    #     except Exception as e:
    #         from tkinter import messagebox
    #         messagebox.showerror("Export error", str(e))
    def export_function(self):
        """Export a parameterized Python function grid_value(x, y, o, h, w, cx, cy)."""
        try:
            # Plan fields (for correct dependency order)
            params = self._gather_params()
            plan   = self._plan_params(params)   # we only need plan["order"] and meta
            meta   = plan["meta"]
            order  = plan["order"]

            f_expr = self.expr_var.get().strip()

            # --- Build source code (no baking; fully parameterized) ---
            lines = []
            lines.append("# Auto-generated by GridEditor — do not edit by hand unless you know what you're doing.\n")
            lines.append("from math import sqrt, sin, cos, tan, asin, acos, atan, atan2, sinh, cosh, tanh, log, log10, exp, floor, ceil, fabs, pi, e, tau\n\n")
            lines.append("def sgn(v):\n    return (v > 0) - (v < 0)\n\n")
            lines.append("def clamp(v, lo, hi):\n    return lo if v < lo else (hi if v > hi else v)\n\n")
            lines.append("def grid_value(x, y, o, h, w, cx, cy):\n")
            indent = "    "

            # Match the app: use coords relative to center
            lines.append(f"{indent}# coordinates relative to center (match visualizer)\n")
            lines.append(f"{indent}x = x - cx\n{indent}y = y - cy\n\n")

            # Emit all fields in topo order (works even if some are “constants”)
            if order:
                lines.append(f"{indent}# fields evaluated in dependency order\n")
                for name in order:
                    expr = meta[name]["expr"]
                    lines.append(f"{indent}{name} = {expr}\n")
                lines.append("\n")

            # Final expression
            lines.append(f"{indent}return float({f_expr})\n")

            src = "".join(lines)

            # Save file
            from tkinter import filedialog
            fpath = filedialog.asksaveasfilename(
                title="Export Python function",
                defaultextension=".py",
                filetypes=[("Python files", "*.py"), ("All files", "*.*")]
            )
            if not fpath:
                return
            with open(fpath, "w", encoding="utf-8") as f:
                f.write(src)
            self.status.config(text=f"Exported function to '{fpath}'")
        except Exception as e:
            from tkinter import messagebox
            messagebox.showerror("Export error", str(e))


    def _set_fields_visible(self, visible: bool):
        """Show or hide the Fields panel without changing pack order."""
        if visible:
            # re-pack above the canvas frame to keep layout consistent
            self.params_wrap.pack(side=tk.TOP, fill=tk.X, padx=8, pady=(0, 8), before=self.canvas_frame)
            self.fields_toggle_btn.config(text="Hide variables")
            self.show_fields.set(True)
        else:
            self.params_wrap.pack_forget()
            self.fields_toggle_btn.config(text="Show variables")
            self.show_fields.set(False)

    def _toggle_fields(self):
        self._set_fields_visible(not self.show_fields.get())

        
    def center_canvas(self):
        """Center the canvas view both horizontally and vertically."""
        self.update_idletasks()
        sr = self.canvas.cget("scrollregion")
        if not sr:
            return
        try:
            x1, y1, x2, y2 = map(float, sr.split())
        except Exception:
            return

        total_w = x2 - x1
        total_h = y2 - y1
        view_w = max(1, self.canvas.winfo_width())
        view_h = max(1, self.canvas.winfo_height())

        max_x = max(0.0, total_w - view_w)
        max_y = max(0.0, total_h - view_h)

        fx = 0.0 if max_x == 0 else ((total_w - view_w) / 2.0) / max_x
        fy = 0.0 if max_y == 0 else ((total_h - view_h) / 2.0) / max_y

        self.canvas.xview_moveto(fx)
        self.canvas.yview_moveto(fy)


    def _update_scrollregion_centered(self, total_w, total_h):
        """Make scrollregion include symmetric margins so content can center."""
        self.update_idletasks()  # ensure canvas size is current
        view_w = max(1, self.canvas.winfo_width())
        view_h = max(1, self.canvas.winfo_height())

        pad_x = max(0, (view_w - total_w) // 2)
        pad_y = max(0, (view_h - total_h) // 2)

        # Negative x1/y1 gives us a left/top margin; symmetric on the other side.
        self.canvas.config(scrollregion=(-pad_x, -pad_y, total_w + pad_x, total_h + pad_y))

    # ----- Mouse wheel / pan / zoom -----
    
    def _bind_mousewheel(self):
        # Windows / macOS
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)
        # Linux (X11)
        self.canvas.bind_all("<Button-4>", self._on_linux_scroll)
        self.canvas.bind_all("<Button-5>", self._on_linux_scroll)
    
    def _unbind_mousewheel(self):
        self.canvas.unbind_all("<MouseWheel>")
        self.canvas.unbind_all("<Button-4>")
        self.canvas.unbind_all("<Button-5>")
    
    def _on_mousewheel(self, event):
        shift = bool(event.state & 0x0001)    # Shift for horizontal scroll
        ctrl  = bool(event.state & 0x0004)    # Ctrl for zoom
        if ctrl:
            # Zoom at cursor
            direction = 1 if event.delta > 0 else -1
            self._zoom_at_cursor(direction, event.x, event.y)
            return
        units = -1 if event.delta > 0 else 1
        if shift:
            self.canvas.xview_scroll(units, "units")
        else:
            self.canvas.yview_scroll(units, "units")
    
    def _on_linux_scroll(self, event):
        shift = bool(event.state & 0x0001)
        ctrl  = bool(event.state & 0x0004)
        if ctrl:
            direction = 1 if event.num == 4 else -1  # 4 up, 5 down
            self._zoom_at_cursor(direction, event.x, event.y)
            return
        units = -1 if event.num == 4 else 1
        if shift:
            self.canvas.xview_scroll(units, "units")
        else:
            self.canvas.yview_scroll(units, "units")
    
    def _pan_mark(self, event):
        self.canvas.scan_mark(event.x, event.y)
    
    def _pan_drag(self, event):
        self.canvas.scan_dragto(event.x, event.y, gain=1)
    
    def _zoom_at_cursor(self, direction, widget_x, widget_y):
        """
        Ctrl + wheel zoom. direction: +1 zoom in, -1 zoom out.
        Keeps the grid coordinate under the mouse fixed on screen.
        """
        old_cell = max(1, int(self.cell_var.get()))
        # Zoom factor; tweak if you want faster/slower zoom
        factor = 1.15 if direction > 0 else 1/1.15
        new_cell = int(round(old_cell * factor))
    
        # clamp (feel free to change limits)
        new_cell = max(4, min(120, new_cell))
        if new_cell == old_cell:
            return
    
        # Canvas coords under pointer before zoom
        before_cx = self.canvas.canvasx(widget_x)
        before_cy = self.canvas.canvasy(widget_y)
        # Grid coords under pointer
        gx = before_cx / old_cell
        gy = before_cy / old_cell
    
        # Apply zoom
        self.cell_var.set(new_cell)
        self.render()  # updates scrollregion
    
        # New canvas pixel coords of that same grid point
        after_px = gx * new_cell
        after_py = gy * new_cell
    
        # Desired top-left so that (after_px,after_py) appears at widget (widget_x,widget_y)
        target_left = after_px - widget_x
        target_top  = after_py - widget_y
    
        # Convert to fractions for xview_moveto/yview_moveto
        total_w = self.canvas.bbox("all")[2] if self.canvas.bbox("all") else 0
        total_h = self.canvas.bbox("all")[3] if self.canvas.bbox("all") else 0
        view_w  = max(1, self.canvas.winfo_width())
        view_h  = max(1, self.canvas.winfo_height())
    
        max_x = max(0, total_w - view_w)
        max_y = max(0, total_h - view_h)
    
        fx = 0.0 if max_x == 0 else min(1.0, max(0.0, target_left / max_x))
        fy = 0.0 if max_y == 0 else min(1.0, max(0.0, target_top  / max_y))
    
        self.canvas.xview_moveto(fx)
        self.canvas.yview_moveto(fy)
    


    # ----- Parameters (expression-capable fields) -----

    def add_param_row(self, preset=None):
        """Adds a row (Name | Type | Value[expression] | ✕)."""
        row_frame = ttk.Frame(self.params_container)
        row_frame.pack(side=tk.TOP, fill=tk.X, pady=2)

        name_var = tk.StringVar(value=(preset[0] if preset else "var"))
        type_var = tk.StringVar(value=(preset[1] if preset else "float"))
        value_var = tk.StringVar(value=(preset[2] if preset else "0"))

        ttk.Label(row_frame, text="Name").pack(side=tk.LEFT, padx=(0, 4))
        name_entry = ttk.Entry(row_frame, textvariable=name_var, width=12)
        name_entry.pack(side=tk.LEFT)
        name_entry.bind("<Return>", lambda e: self.render())
        
        self.auto_center = tk.BooleanVar(value=True)
        ttk.Label(row_frame, text="Type").pack(side=tk.LEFT, padx=(8, 4))
        type_box = ttk.Combobox(row_frame, textvariable=type_var, width=7, state="readonly",
                                values=["float", "int", "bool"])
        type_box.pack(side=tk.LEFT)
        type_box.bind("<<ComboboxSelected>>", lambda e: self.render())

        ttk.Label(row_frame, text="Value (expr)").pack(side=tk.LEFT, padx=(8, 4))
        value_entry = ttk.Entry(row_frame, textvariable=value_var, width=28)
        value_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        value_entry.bind("<Return>", lambda e: self.render())

        del_btn = ttk.Button(row_frame, text="✕", width=3,
                             command=lambda rf=row_frame: self.remove_param_row(rf))
        del_btn.pack(side=tk.LEFT, padx=(8, 0))

        self.param_rows.append({
            "frame": row_frame,
            "name_var": name_var,
            "type_var": type_var,
            "value_var": value_var
        })

    def remove_param_row(self, frame):
        for i, row in enumerate(self.param_rows):
            if row["frame"] == frame:
                row["frame"].destroy()
                del self.param_rows[i]
                break
        self.render()

    def _gather_params(self):
        """Read rows, validate names/types/exprs. Returns list of dicts."""
        used = set(SAFE_FUNCS.keys()) | BASE_VARS
        params = []
        for row in self.param_rows:
            name = row["name_var"].get().strip()
            typ  = row["type_var"].get()
            expr = row["value_var"].get().strip()
            if not name: raise RuntimeError("Field name cannot be empty.")
            if not IDENT_RE.match(name):
                raise RuntimeError(f"Invalid field name '{name}'.")
            if name in used:
                raise RuntimeError(f"Field name '{name}' clashes with a built-in.")
            if typ not in ("float", "int", "bool"):
                raise RuntimeError(f"Field '{name}': unknown type '{typ}'.")
            if not expr:
                raise RuntimeError(f"Field '{name}': expression cannot be empty.")
            params.append({"name": name, "type": typ, "expr": expr})
            used.add(name)
        return params

    def _plan_params(self, params):
        """
        Build dependency order, split into constants vs per-cell.
        Returns dict with:
          order : topo order of all names
          const_order : names evaluable once (no x/y and deps are constants)
          cell_order  : names to eval per cell
          meta : {name: dict(type, expr, deps, uses_base, is_const)}
        """
        name_set = {p["name"] for p in params}
        meta = {}
        # Build metadata
        for p in params:
            used = extract_names(p["expr"])
            deps = {n for n in used if n in name_set and n != p["name"]}
            uses_base = bool(used & BASE_VARS)
            meta[p["name"]] = {"type": p["type"], "expr": p["expr"], "deps": deps, "uses_base": uses_base, "is_const": False}

        # Topological sort
        indeg = {n: 0 for n in name_set}
        for n, m in meta.items():
            for d in m["deps"]:
                indeg[n] += 1
        from collections import deque
        q = deque([n for n in name_set if indeg[n] == 0])
        order = []
        while q:
            n = q.popleft()
            order.append(n)
            for k, m in meta.items():
                if n in m["deps"]:
                    indeg[k] -= 1
                    if indeg[k] == 0:
                        q.append(k)

        if len(order) != len(name_set):
            # cycle; try to show a hint
            cyclic = [n for n, d in indeg.items() if d > 0]
            raise RuntimeError(f"Field dependency cycle detected: {' -> '.join(cyclic)}")

        # Mark constants (no base vars AND all deps are constants)
        const_set = set()
        for n in order:
            m = meta[n]
            if not m["uses_base"] and all(meta[d]["is_const"] for d in m["deps"]):
                m["is_const"] = True
                const_set.add(n)

        const_order = [n for n in order if meta[n]["is_const"]]
        cell_order  = [n for n in order if not meta[n]["is_const"]]
        return {"order": order, "const_order": const_order, "cell_order": cell_order, "meta": meta}


    def validate_param_headers(self):
        """Validate names/types (not values) and return ordered descriptors."""
        used = set(SAFE_FUNCS.keys()) | {"x", "y", "w", "h", "o", "cx", "cy"}
        descs = []
        for row in self.param_rows:
            name = row["name_var"].get().strip()
            typ = row["type_var"].get()
            expr = row["value_var"].get().strip()

            if not name:
                raise RuntimeError("Field name cannot be empty.")
            if not IDENT_RE.match(name):
                raise RuntimeError(f"Invalid field name '{name}'. Use letters, digits, underscores; not starting with a digit.")
            if name in used:
                raise RuntimeError(f"Field name '{name}' clashes with a built-in or another name.")
            if typ not in ("float", "int", "bool"):
                raise RuntimeError(f"Field '{name}': unknown type '{typ}'")
            if not expr:
                raise RuntimeError(f"Field '{name}': expression cannot be empty.")

            used.add(name)
            descs.append((name, typ, expr))
        return descs

    # ----- Rendering -----

    # def compute_values(self):
    #     w = max(1, int(self.w_var.get()))
    #     h = max(1, int(self.h_var.get()))
    #     o = max(0, int(self.orientation.get()))

    #     cx = int(self.center_x.get())
    #     cy = int(self.center_y.get())
    #     f_expr = self.expr_var.get().strip()

    #     # Validate headers once
    #     param_descs = self.validate_param_headers()

    #     vals = [[0 for _ in range(w)] for _ in range(h)]
    #     vmin, vmax = float("inf"), float("-inf")

    #     for yy in range(h):
    #         for xx in range(w):
    #             # per-cell base coordinates relative to center
    #             x_val = xx - cx
    #             y_val = yy - cy

    #             # Build per-cell environment and evaluate fields in order
    #             env = make_base_env(x_val, y_val, w, h, o, cx, cy)

    #             for (pname, ptype, pexpr) in param_descs:
    #                 try:
    #                     pval = eval_with_env(pexpr, env)
    #                 except Exception as e:
    #                     raise RuntimeError(f"Field '{pname}' failed at (x={xx}, y={yy}): {e}")

    #                 # Cast to declared type
    #                 try:
    #                     if ptype == "float":
    #                         pval = float(pval)
    #                     elif ptype == "int":
    #                         pval = int(pval)
    #                     elif ptype == "bool":
    #                         # accept truthy/falsey from expression
    #                         pval = bool(pval)
    #                 except Exception as e:
    #                     raise RuntimeError(f"Field '{pname}' type '{ptype}' cast error at (x={xx}, y={yy}): {e}")

    #                 env[pname] = pval  # make available to subsequent fields and f(x,y)

    #             # Evaluate main expression with final env
    #             try:
    #                 v = eval_with_env(f_expr, env)
    #             except Exception as e:
    #                 raise RuntimeError(f"Error in f(x,y) at (x={xx}, y={yy}): {e}")

    #             try:
    #                 v = float(v)
    #             except Exception:
    #                 raise RuntimeError(f"f(x,y) must return a number; got {type(v)} at (x={xx}, y={yy})")

    #             vals[yy][xx] = v
    #             if v < vmin: vmin = v
    #             if v > vmax: vmax = v

    #     if vmin == float("inf"):
    #         vmin, vmax = 0.0, 1.0
    #     return vals, vmin, vmax
    def compute_values(self):
        w = max(1, int(self.w_var.get()))
        h = max(1, int(self.h_var.get()))
        o = max(0, int(self.orientation.get()))
        cx = int(self.center_x.get())
        cy = int(self.center_y.get())
        f_expr = self.expr_var.get().strip()

        # plan fields
        params = self._gather_params()
        plan = self._plan_params(params)
        meta = plan["meta"]

        # ---- Evaluate constant fields ONCE
        # keep ONLY constant field values here (no x/y/Safe funcs)
        const_vals = {}
        base_for_const = make_base_env(0, 0, w, h, o, cx, cy)
        for name in plan["const_order"]:
            m = meta[name]
            try:
                # constants can depend on earlier constants
                val = eval_with_env(m["expr"], {**base_for_const, **const_vals})
            except Exception as e:
                raise RuntimeError(f"Field '{name}' (constant) failed: {e}")
            try:
                if m["type"] == "float": val = float(val)
                elif m["type"] == "int":   val = int(val)
                elif m["type"] == "bool":  val = bool(val)
            except Exception as e:
                raise RuntimeError(f"Field '{name}' cast error: {e}")
            const_vals[name] = val  # only store the constant itself

        # ---- Per-cell evaluation
        vals = [[0 for _ in range(w)] for _ in range(h)]
        vmin, vmax = float("inf"), float("-inf")

        for yy in range(h):
            for xx in range(w):
                x_val = xx - cx
                y_val = yy - cy

                # fresh base env per cell, then add constants (won't overwrite x/y)
                env = make_base_env(x_val, y_val, w, h, o, cx, cy)
                env.update(const_vals)

                # evaluate per-cell fields in topo order
                for name in plan["cell_order"]:
                    m = meta[name]
                    try:
                        v = eval_with_env(m["expr"], env)
                    except Exception as e:
                        raise RuntimeError(f"Field '{name}' failed at (x={xx},y={yy}): {e}")
                    try:
                        if m["type"] == "float": v = float(v)
                        elif m["type"] == "int":   v = int(v)
                        elif m["type"] == "bool":  v = bool(v)
                    except Exception as e:
                        raise RuntimeError(f"Field '{name}' cast error at (x={xx},y={yy}): {e}")
                    env[name] = v

                # main function
                try:
                    out = eval_with_env(f_expr, env)
                except Exception as e:
                    raise RuntimeError(f"Error in f(x,y) at (x={xx}, y={yy}): {e}")
                try:
                    out = float(out)
                except Exception:
                    raise RuntimeError(f"f(x,y) must return a number; got {type(out)} at (x={xx}, y={yy})")

                vals[yy][xx] = out
                if out < vmin: vmin = out
                if out > vmax: vmax = out

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
        # self.canvas.config(scrollregion=(0, 0, total_w, total_h))
        self._update_scrollregion_centered(total_w, total_h)

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
        if self.auto_center.get():
            # Use 'after' to ensure geometry is updated before centering
            self.after_idle(self.center_canvas)


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

    # ----- Save / Load configuration -----

    def serialize(self):
        params = []
        for row in self.param_rows:
            params.append({
                "name": row["name_var"].get(),
                "type": row["type_var"].get(),
                "value": row["value_var"].get(),
            })
        return {
            "expr": self.expr_var.get(),
            "w": int(self.w_var.get()),
            "h": int(self.h_var.get()),
            "cell": int(self.cell_var.get()),
            "orientation": int(self.orientation.get()),
            "center_x": int(self.center_x.get()),
            "center_y": int(self.center_y.get()),
            "show_text": bool(self.show_text_var.get()),
            "params": params,
        }

    def apply_state(self, state):
        try:
            self.expr_var.set(state.get("expr", self.expr_var.get()))
            self.w_var.set(int(state.get("w", self.w_var.get())))
            self.h_var.set(int(state.get("h", self.h_var.get())))
            self.cell_var.set(int(state.get("cell", self.cell_var.get())))
            self.orientation.set(int(state.get("orientation", self.orientation.get())))
            self.center_x.set(int(state.get("center_x", self.center_x.get())))
            self.center_y.set(int(state.get("center_y", self.center_y.get())))
            self.show_text_var.set(bool(state.get("show_text", self.show_text_var.get())))

            for row in list(self.param_rows):
                row["frame"].destroy()
            self.param_rows.clear()

            for p in state.get("params", []):
                self.add_param_row((p.get("name","var"), p.get("type","float"), p.get("value","0")))
        except Exception as e:
            messagebox.showerror("Load error", f"Invalid configuration: {e}")
        self.render()

    def save_config(self):
        state = self.serialize()
        fpath = filedialog.asksaveasfilename(
            title="Save configuration",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if not fpath:
            return
        try:
            with open(fpath, "w", encoding="utf-8") as f:
                json.dump(state, f, indent=2)
            self.status.config(text=f"Saved configuration to '{fpath}'")
        except Exception as e:
            messagebox.showerror("Save error", str(e))

    def load_config(self):
        fpath = filedialog.askopenfilename(
            title="Load configuration",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if not fpath:
            return
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                state = json.load(f)
        except Exception as e:
            messagebox.showerror("Load error", str(e))
            return
        self.apply_state(state)

# ---------- Main ----------

if __name__ == "__main__":
    app = GridEditor()

    # Examples you can try (using the seeded fields a,freq):
    #   - Set fields:
    #       a = 8
    #       freq = 0.2
    #       waves = sin(x*freq) + cos(y*freq)
    #   - Then f(x,y):
    #       waves * a
    #
    # Toroidal helpers (add as fields if desired):
    #   tor_x = (abs(x) % (w//2)) if (abs(x) % w) < (w//2) else (w//2 - (abs(x) % (w//2)))
    #   tor_y = (abs(y) % (h//2)) if (abs(y) % h) < (h//2) else (h//2 - (abs(y) % (h//2)))
    #   f(x,y) = abs(tor_x) + abs(tor_y)
    app.mainloop()
