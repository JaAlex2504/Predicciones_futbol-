import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import messagebox, scrolledtext, ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


class InterpoladorUniversal:
    def __init__(self, ventana):
        self.ventana = ventana
        self.ventana.title(
            "Programa que realiza aproximación polinomial a través de interpolación"
        )
        self.ventana.geometry("1100x850")

        # Variables de estado
        self.coeficientes = None
        self.entradas_x = []
        self.entradas_y = []
        self.canvas_viz = None
        self.ax = None

        self.configurar_estilos()
        self.crear_interfaz()

    def configurar_estilos(self):
        style = ttk.Style()
        style.configure("TButton", padding=5, font=("Arial", 10))
        style.configure("Header.TLabel", font=("Arial", 12, "bold"))

    def crear_interfaz(self):
        # SECCIÓN SUPERIOR: CONFIGURACIÓN Y CONTROLES
        frame_top = ttk.Frame(self.ventana, padding=10)
        frame_top.pack(fill=tk.X)

        ttk.Label(frame_top, text="Número de Puntos (n):", style="Header.TLabel").pack(
            side=tk.LEFT, padx=5
        )
        self.entry_n = ttk.Entry(frame_top, width=8)
        self.entry_n.pack(side=tk.LEFT, padx=5)

        ttk.Button(frame_top, text="Generar Tabla", command=self.generar_tabla).pack(
            side=tk.LEFT, padx=10
        )
        ttk.Button(
            frame_top, text="Calcular Polinomio", command=self.ejecutar_calculo
        ).pack(side=tk.LEFT)

        ttk.Button(
            frame_top, text="Cargar Datos desde CSV", command=self.cargar_datos_csv
        ).pack(side=tk.LEFT, padx=10)

        # SECCIÓN CENTRAL: TABLA Y GRÁFICA
        self.frame_main = ttk.Frame(self.ventana, padding=10)
        self.frame_main.pack(fill=tk.BOTH, expand=True)

        # Sub-frame para la tabla (con scroll)
        self.frame_datos = ttk.LabelFrame(
            self.frame_main, text=" Datos Experimentales ", padding=10
        )
        self.frame_datos.pack(side=tk.LEFT, fill=tk.Y, padx=5)

        self.canvas_tabla = tk.Canvas(self.frame_datos, width=220)
        self.scrollbar = ttk.Scrollbar(
            self.frame_datos, orient="vertical", command=self.canvas_tabla.yview
        )
        self.scroll_frame = ttk.Frame(self.canvas_tabla)

        self.scroll_frame.bind(
            "<Configure>",
            lambda e: self.canvas_tabla.configure(
                scrollregion=self.canvas_tabla.bbox("all")
            ),
        )
        self.canvas_tabla.create_window((0, 0), window=self.scroll_frame, anchor="nw")
        self.canvas_tabla.configure(yscrollcommand=self.scrollbar.set)

        self.canvas_tabla.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Sub-frame para la gráfica
        self.frame_viz = ttk.LabelFrame(
            self.frame_main, text=" Visualización Matemática ", padding=10
        )
        self.frame_viz.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5)

        # --- SECCIÓN INFERIOR: RESULTADOS Y PREDICCIÓN ---
        frame_bottom = ttk.Frame(self.ventana, padding=10)
        frame_bottom.pack(fill=tk.X)

        # Ecuación y Coeficientes
        self.txt_output = scrolledtext.ScrolledText(
            frame_bottom, height=5, font=("Consolas", 10)
        )
        self.txt_output.pack(fill=tk.X, pady=5)

        # Herramienta de Predicción
        self.frame_pred = ttk.LabelFrame(
            frame_bottom, text=" Herramienta de Predicción (Evaluación) ", padding=10
        )
        self.frame_pred.pack(fill=tk.X)

        ttk.Label(self.frame_pred, text="Valor de X a predecir:").pack(
            side=tk.LEFT, padx=5
        )
        self.entry_x_eval = ttk.Entry(self.frame_pred, width=15)
        self.entry_x_eval.pack(side=tk.LEFT, padx=5)

        ttk.Button(
            self.frame_pred, text="Calcular Valor f(x)", command=self.predecir
        ).pack(side=tk.LEFT, padx=10)

        self.lbl_resultado = ttk.Label(
            self.frame_pred, text="Resultado: ---", font=("Arial", 11, "bold")
        )
        self.lbl_resultado.pack(side=tk.LEFT, padx=20)

    def cargar_datos_csv(self):
        ruta_archivos = tk.filedialog.askopenfilenames(
            title="Seleccionar archivos CSV",
            filetypes=[
                ("Archivos CSV", "*.csv"),
                ("Archivos de texto", "*.txt"),
                ("Todos los archivos", "*.*"),
            ],
        )
        if not ruta_archivos:
            return

        try:
            df = pd.read_csv(ruta_archivos[0])

            if len(df.columns) < 2:
                messagebox.showerror(
                    "Error", "El archivo debe contener al menos dos columnas."
                )
                return

            x_datos = df.iloc[:, 0].values
            y_datos = df.iloc[:, 1].values

            self.entry_n.delete(0, tk.END)
            self.entry_n.insert(0, str(len(x_datos)))

            self.generar_tabla()

            for i in range(len(x_datos)):
                self.entradas_x[i].delete(0, tk.END)
                self.entradas_x[i].insert(0, str(x_datos[i]))

                self.entradas_y[i].delete(0, tk.END)
                self.entradas_y[i].insert(0, str(y_datos[i]))

            messagebox.showinfo(
                "Éxito", "Datos cargados correctamente desde el archivo CSV."
            )

        except Exception as e:
            messagebox.showerror("Error", f"No se pudo cargar el archivo: {e}")

    def generar_tabla(self):
        for w in self.scroll_frame.winfo_children():
            w.destroy()
        try:
            n = int(self.entry_n.get())
            self.entradas_x, self.entradas_y = [], []

            ttk.Label(self.scroll_frame, text="X", font=("Arial", 10, "bold")).grid(
                row=0, column=0, pady=5
            )
            ttk.Label(self.scroll_frame, text="f(X)", font=("Arial", 10, "bold")).grid(
                row=0, column=1, pady=5
            )

            for i in range(n):
                ex = ttk.Entry(self.scroll_frame, width=10, justify="center")
                ey = ttk.Entry(self.scroll_frame, width=10, justify="center")
                ex.grid(row=i + 1, column=0, padx=5, pady=2)
                ey.grid(row=i + 1, column=1, padx=5, pady=2)
                self.entradas_x.append(ex)
                self.entradas_y.append(ey)
        except ValueError:
            messagebox.showerror("Error", "Ingrese un número entero válido para n.")

    def ejecutar_calculo(self):
        try:
            x_pts = [float(e.get()) for e in self.entradas_x]
            y_pts = [float(e.get()) for e in self.entradas_y]

            # Matriz de Vandermonde para ajuste exacto
            V = np.vander(x_pts, increasing=True)
            self.coeficientes = np.linalg.solve(V, y_pts)

            self.mostrar_resultados()
            self.graficar(x_pts, y_pts)
        except Exception as e:
            messagebox.showerror("Error", f"Error en el cálculo: {e}")

    def mostrar_resultados(self):
        self.txt_output.delete(1.0, tk.END)
        self.txt_output.insert(tk.END, "SISTEMA RESUELTO\n")
        self.txt_output.insert(tk.END, "-" * 50 + "\n")

        ecuacion = "P(x) = "
        for i, c in enumerate(self.coeficientes):
            termino = f"{c:+.6f}"
            if i > 0:
                termino += f"*x^{i}"
            ecuacion += termino + " "

        self.txt_output.insert(tk.END, ecuacion.replace("x^1 ", "x "))

    def predecir(self):
        if self.coeficientes is None:
            messagebox.showwarning("Atención", "Primero debe calcular el polinomio.")
            return
        try:
            x_val = float(self.entry_x_eval.get())
            y_val = sum(c * (x_val**i) for i, c in enumerate(self.coeficientes))
            self.lbl_resultado.config(text=f"f({x_val}) ≈ {y_val:.6f}")

            # Actualizar gráfica con el punto predicho
            self.ax.scatter(
                x_val, y_val, color="green", s=100, zorder=5, label="Predicción"
            )
            self.ax.legend()
            self.canvas_viz.draw()
        except ValueError:
            messagebox.showerror("Error", "Ingrese un valor numérico válido para X.")

    def graficar(self, x_pts, y_pts):
        for w in self.frame_viz.winfo_children():
            w.destroy()

        fig, self.ax = plt.subplots(figsize=(6, 4), dpi=100)

        # Generar curva suave
        margen = (max(x_pts) - min(x_pts)) * 0.1 if len(x_pts) > 1 else 1
        x_smooth = np.linspace(min(x_pts) - margen, max(x_pts) + margen, 200)
        y_smooth = sum(c * (x_smooth**i) for i, c in enumerate(self.coeficientes))

        self.ax.plot(
            x_smooth, y_smooth, "r-", label="Polinomio Interpolante", alpha=0.7
        )
        self.ax.scatter(
            x_pts,
            y_pts,
            color="blue",
            edgecolor="black",
            s=50,
            label="Puntos Originales",
        )

        self.ax.set_title("Interpolación Polinomial", fontsize=12)
        self.ax.grid(True, linestyle="--", alpha=0.6)
        self.ax.legend()

        self.canvas_viz = FigureCanvasTkAgg(fig, master=self.frame_viz)
        self.canvas_viz.draw()
        self.canvas_viz.get_tk_widget().pack(fill=tk.BOTH, expand=True)


if __name__ == "__main__":
    root = tk.Tk()
    app = InterpoladorUniversal(root)
    root.mainloop()
