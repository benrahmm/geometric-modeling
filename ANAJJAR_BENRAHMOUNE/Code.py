from tkinter import *
import numpy as np
import matplotlib.pyplot as plt

class HermiteSplineApp(Tk):
    def __init__(self):
        super().__init__()

        self.title("Hermite Spline Editor")
        self.geometry("1200x800")

        self.radius = 5
        self.points = []
        self.m0 = None
        self.mN = None
        self.current_mode = "add_points"  # Modes: "add_points", "draw_m0", "draw_mN"
        self.vector_start = None

        self.setup_canvas()
        self.setup_panel()

    def setup_canvas(self):
        self.canvas = Canvas(self, bd=2, cursor="plus", bg="#fbfbfb", width=1000, height=800)
        self.canvas.grid(column=0, padx=2, pady=2, rowspan=2, columnspan=2, sticky="nsew")

        self.canvas.bind('<Button-1>', self.handle_canvas_click)

    def setup_panel(self):
        self.panel = Frame(self, relief=RAISED, bg="#e1e1e1")
        self.panel.grid(row=0, column=2, padx=2, pady=2, rowspan=2, sticky="nswe")

        # Boutons pour les modes de dessin
        Button(self.panel, text="Ajouter des points", command=self.set_mode_add_points).pack()
        Button(self.panel, text="Dessiner m0", command=self.set_mode_draw_m0).pack()
        Button(self.panel, text="Dessiner mN", command=self.set_mode_draw_mN).pack()

        # Slider pour ajuster le paramètre c
        Label(self.panel, text="Paramètre c (tension): ").pack()
        self.c_slider = Scale(self.panel, from_=0, to=1, resolution=0.01, orient=HORIZONTAL)
        self.c_slider.set(0.5)
        self.c_slider.pack()


        Button(self.panel, text="Tracer la courbe", command=lambda:self.draw_curve_methode_1(False)).pack()
        Button(self.panel, text="Afficher la courbure", command=self.plot_curvature_methode_1).pack()

        Label(self.panel, text="Paramètre tension: ").pack()
        self.s_slider = Scale(self.panel, from_=-1, to=1, resolution=0.01, orient=HORIZONTAL)
        self.s_slider.set(0)
        self.s_slider.pack()

        Label(self.panel, text="Paramètre biais: ").pack()
        self.b_slider = Scale(self.panel, from_=-1, to=1, resolution=0.01, orient=HORIZONTAL)
        self.b_slider.set(0)
        self.b_slider.pack()

        Label(self.panel, text="Paramètre continuité: ").pack()
        self.t_slider = Scale(self.panel, from_=-1, to=1, resolution=0.01, orient=HORIZONTAL)
        self.t_slider.set(0)
        self.t_slider.pack()


        Button(self.panel, text="Tracer la courbe (autre méthode)", command=lambda:self.draw_curve_methode_2(False)).pack()
        Button(self.panel, text="Afficher la courbure (méthode 2)", command=self.plot_curvature_methode_2).pack()
        
        Button(self.panel, text="Tracer la courbe Lagrange", command=lambda:self.draw_lagrange_curve(False)).pack()
        Button(self.panel, text="Afficher la courbure Lagrange", command=self.plot_curvature_Lagrange).pack()

        Button(self.panel, text="Tracer la courbe C2", command=lambda:self.draw_c2_curve(False)).pack()

        Button(self.panel, text="Superposer les courbes", command=self.superposer).pack()


        Button(self.panel, text="Réinitialiser", command=self.reset_all).pack()


    def set_mode_add_points(self):
        self.current_mode = "add_points"

    def set_mode_draw_m0(self):
        self.current_mode = "draw_m0"

    def set_mode_draw_mN(self):
        self.current_mode = "draw_mN"

    def handle_canvas_click(self, event):
        if self.current_mode == "add_points":
            self.points.append((event.x, event.y))
            self.create_point(event.x, event.y, "red")
            if len(self.points) > 1:
                self.draw_polygon()

        elif self.current_mode in ["draw_m0", "draw_mN"]:
            if self.vector_start is None:
                self.vector_start = (event.x, event.y)
            else:
                vector_end = (event.x, event.y)
                self.draw_vector(self.vector_start, vector_end, color="purple" if self.current_mode == "draw_m0" else "orange")

                # Calcul des composantes du vecteur
                dx = vector_end[0] - self.vector_start[0]
                dy = vector_end[1] - self.vector_start[1]
                if self.current_mode == "draw_m0":
                    self.m0 = (dx, dy)
                else:
                    self.mN = (dx, dy)

                self.vector_start = None

    def create_point(self, x, y, color):
        self.canvas.create_oval(x - self.radius, y - self.radius, x + self.radius, y + self.radius,
                                outline=color, fill=color, tags="control_points")

    def draw_polygon(self):
        self.canvas.delete("control_polygon")
        for i in range(len(self.points) - 1):
            self.canvas.create_line(self.points[i][0], self.points[i][1],
                                    self.points[i + 1][0], self.points[i + 1][1],
                                    fill="blue", tags="control_polygon")

    def draw_vector(self, start, end, color):
        self.canvas.create_line(start[0], start[1], end[0], end[1], arrow=LAST, fill=color, tags="vector")

    def calculate_derivatives_methode_1(self):
        c = self.c_slider.get()

        if not self.m0:
            self.m0 = ((1 - c) * (self.points[1][0] - self.points[0][0]), (1 - c) * (self.points[1][1] - self.points[0][1]))
        if not self.mN:
            self.mN = ((1 - c) * (self.points[-1][0] - self.points[-2][0]), (1 - c) * (self.points[-1][1] - self.points[-2][1]))

        derivatives = [self.m0]
        for k in range(1, len(self.points) - 1):
            mk = (1 - c) * (np.array(self.points[k + 1]) - np.array(self.points[k - 1]))/2
            derivatives.append(mk)
        derivatives.append(self.mN)

        return derivatives

    def calculate_derivatives_methode_2(self):
        c = self.s_slider.get()
        t = self.t_slider.get()
        b = self.b_slider.get()

        if not self.m0:
            self.m0 = ((1-c)*(self.points[1][0] - self.points[0][0]), (1-c)*(self.points[1][1] - self.points[0][1]))
        if not self.mN:
            self.mN = ((1-c)*(self.points[-1][0] - self.points[-2][0]), (1-c)*(self.points[-1][1] - self.points[-2][1]))

        derivatives = [self.m0]
        for k in range(1, len(self.points) - 1):
            #Cherche une nouvelle méthode 
            mk = (1-c)*(1+b)*(1+t)*(np.array(self.points[k]) - np.array(self.points[k - 1]))/2 + (1-c)*(1-b)*(1-t)*(np.array(self.points[k + 1]) - np.array(self.points[k]))/2
            derivatives.append(mk)
        derivatives.append(self.mN)

        return derivatives

    def draw_curve_methode_1(self, superp):
        if len(self.points) < 2:
            print("Entrez au moins deux points.")
            return
        
        self.canvas.delete("curve")
        if not superp:
            self.canvas.delete("curve2")
            self.canvas.delete("C2_curve")
            self.canvas.delete("lagrange_curve")

        derivatives = self.calculate_derivatives_methode_1()

        t_values = np.linspace(0, 1, 100)
        n = len(self.points)

        for i in range(n - 1):
            p0, p1 = self.points[i], self.points[i + 1]
            m0, m1 = derivatives[i], derivatives[i + 1]

            x_curve, y_curve = [], []
            for t in t_values:
                h00 = 2 * t**3 - 3 * t**2 + 1
                h10 = t**3 - 2 * t**2 + t
                h01 = -2 * t**3 + 3 * t**2
                h11 = t**3 - t**2

                x = h00 * p0[0] + h10 * m0[0] + h01 * p1[0] + h11 * m1[0]
                y = h00 * p0[1] + h10 * m0[1] + h01 * p1[1] + h11 * m1[1]

                x_curve.append(x)
                y_curve.append(y)

            for j in range(len(x_curve) - 1):
                self.canvas.create_line(x_curve[j], y_curve[j], x_curve[j + 1], y_curve[j + 1],
                                        fill="green", width=2, tags="curve")

    def draw_curve_methode_2(self, superp):
        if len(self.points) < 2:
            print("Entrez au moins deux points.")
            return
            
        self.canvas.delete("curve2")
        if not superp:
            self.canvas.delete("curve")
            self.canvas.delete("C2_curve")
            self.canvas.delete("lagrange_curve")

        derivatives = self.calculate_derivatives_methode_2()

        t_values = np.linspace(0, 1, 100)
        n = len(self.points)

        for i in range(n - 1):
            p0, p1 = self.points[i], self.points[i + 1]
            m0, m1 = derivatives[i], derivatives[i + 1]

            x_curve, y_curve = [], []
            for t in t_values:
                h00 = 2 * t**3 - 3 * t**2 + 1
                h10 = t**3 - 2 * t**2 + t
                h01 = -2 * t**3 + 3 * t**2
                h11 = t**3 - t**2

                x = h00 * p0[0] + h10 * m0[0] + h01 * p1[0] + h11 * m1[0]
                y = h00 * p0[1] + h10 * m0[1] + h01 * p1[1] + h11 * m1[1]

                x_curve.append(x)
                y_curve.append(y)

            for j in range(len(x_curve) - 1):
                self.canvas.create_line(x_curve[j], y_curve[j], x_curve[j + 1], y_curve[j + 1],
                                        fill="blue", width=2, tags="curve2")

    def handle_drag_stop(self, event):
        pass

    def handle_drag(self, event):
        pass

    def reset_all(self):
        # Efface tout le contenu du canevas
        self.canvas.delete("all")
        # Réinitialise les variables internes
        self.points = []
        self.m0 = None
        self.mN = None
        self.vector_start = None
    # Ajoutez cette méthode dans la classe HermiteSplineApp
    def plot_curvature_methode_1(self):
        if len(self.points) < 2:
            print("Entrez au moins deux points.")
            return

        derivatives = self.calculate_derivatives_methode_1()
        t_values = np.linspace(0, 1, 100)
        curvatures = []
        n = len(self.points)

        for i in range(n - 1):
            p0, p1 = self.points[i], self.points[i + 1]
            m0, m1 = derivatives[i], derivatives[i + 1]

            for t in t_values:
                # Hermite basis functions and their derivatives
                h00 = 2 * t**3 - 3 * t**2 + 1
                h10 = t**3 - 2 * t**2 + t
                h01 = -2 * t**3 + 3 * t**2
                h11 = t**3 - t**2

                h00_prime = 6 * t**2 - 6 * t
                h10_prime = 3 * t**2 - 4 * t + 1
                h01_prime = -6 * t**2 + 6 * t
                h11_prime = 3 * t**2 - 2 * t

                h00_double_prime = 12 * t - 6
                h10_double_prime = 6 * t - 4
                h01_double_prime = -12 * t + 6
                h11_double_prime = 6 * t - 2

                # Derivatives of x and y
                x_prime = (h00_prime * p0[0] + h10_prime * m0[0] +
                        h01_prime * p1[0] + h11_prime * m1[0])
                y_prime = (h00_prime * p0[1] + h10_prime * m0[1] +
                        h01_prime * p1[1] + h11_prime * m1[1])

                x_double_prime = (h00_double_prime * p0[0] + h10_double_prime * m0[0] +
                                h01_double_prime * p1[0] + h11_double_prime * m1[0])
                y_double_prime = (h00_double_prime * p0[1] + h10_double_prime * m0[1] +
                                h01_double_prime * p1[1] + h11_double_prime * m1[1])

                # Calculate curvature
                numerator = abs(x_prime * y_double_prime - y_prime * x_double_prime)
                denominator = (x_prime**2 + y_prime**2)**1.5
                curvature = numerator / denominator if denominator != 0 else 0
                curvatures.append(curvature)

        # Plot the curvature
        plt.figure(figsize=(10, 5))
        plt.plot(np.linspace(0, n - 1, len(curvatures)), curvatures, label="Courbure")
        plt.title("Courbure de la Spline de Hermite (méthode 1)")
        plt.xlabel("u (paramètre)")
        plt.ylabel("Courbure κ(u)")
        plt.grid()
        plt.legend()
        plt.show()

    def plot_curvature_methode_2(self):
        if len(self.points) < 2:
            print("Entrez au moins deux points.")
            return

        derivatives = self.calculate_derivatives_methode_2()
        t_values = np.linspace(0, 1, 100)
        curvatures = []
        n = len(self.points)

        for i in range(n - 1):
            p0, p1 = self.points[i], self.points[i + 1]
            m0, m1 = derivatives[i], derivatives[i + 1]

            for t in t_values:
                # Hermite basis functions and their derivatives
                h00 = 2 * t**3 - 3 * t**2 + 1
                h10 = t**3 - 2 * t**2 + t
                h01 = -2 * t**3 + 3 * t**2
                h11 = t**3 - t**2

                h00_prime = 6 * t**2 - 6 * t
                h10_prime = 3 * t**2 - 4 * t + 1
                h01_prime = -6 * t**2 + 6 * t
                h11_prime = 3 * t**2 - 2 * t

                h00_double_prime = 12 * t - 6
                h10_double_prime = 6 * t - 4
                h01_double_prime = -12 * t + 6
                h11_double_prime = 6 * t - 2

                # Derivatives of x and y
                x_prime = (h00_prime * p0[0] + h10_prime * m0[0] +
                        h01_prime * p1[0] + h11_prime * m1[0])
                y_prime = (h00_prime * p0[1] + h10_prime * m0[1] +
                        h01_prime * p1[1] + h11_prime * m1[1])

                x_double_prime = (h00_double_prime * p0[0] + h10_double_prime * m0[0] +
                                h01_double_prime * p1[0] + h11_double_prime * m1[0])
                y_double_prime = (h00_double_prime * p0[1] + h10_double_prime * m0[1] +
                                h01_double_prime * p1[1] + h11_double_prime * m1[1])

                # Calculate curvature
                numerator = abs(x_prime * y_double_prime - y_prime * x_double_prime)
                denominator = (x_prime**2 + y_prime**2)**1.5
                curvature = numerator / denominator if denominator != 0 else 0
                curvatures.append(curvature)
        
        # Plot the curvature
        plt.figure(figsize=(10, 5))
        plt.plot(np.linspace(0, n - 1, len(curvatures)), curvatures, label="Courbure")
        plt.title("Courbure de la Spline de Hermite (méthode 2)")
        plt.xlabel("u (paramètre)")
        plt.ylabel("Courbure κ(u)")
        plt.grid()
        plt.legend()
        plt.show()


    def aitken_neville(self, x, points):
        """Implémente l'algorithme Aitken-Neville pour l'interpolation."""
        n = len(points)
        p = np.zeros((n, n))  # Matrice pour les calculs intermédiaires
        
        for i in range(n):
            p[i][0] = points[i][1]  # Initialiser avec les y-values
        
        for j in range(1, n):
            for i in range(n - j):
                xi, xj = points[i][0], points[i + j][0]
                p[i][j] = ((x - xi) * p[i + 1][j - 1] - (x - xj) * p[i][j - 1]) / (xj - xi)
        
        return p[0][n - 1]  # La valeur interpolée finale

    def draw_lagrange_curve(self, superp):
        self.canvas.delete("lagrange_curve")
        
        if not superp:
            self.canvas.delete("curve")
            self.canvas.delete("curve2")
            self.canvas.delete("C2_curve")


        """Trace la courbe interpolée basée sur les points saisis."""
        if len(self.points) < 2:
            print("Ajoutez au moins deux points pour tracer une interpolation.")
            return

        # Trier les points par x pour éviter des erreurs
        L = self.points.copy()
        L.sort(key=lambda pt: pt[0])
        
        x_values = np.linspace(L[0][0], L[-1][0], 500)
        y_values = [self.aitken_neville(x, L) for x in x_values]
        
        # Tracer la courbe
        for i in range(len(x_values) - 1):
            self.canvas.create_line(x_values[i], y_values[i], x_values[i + 1], y_values[i + 1],
                                    fill="red", width=2, tags="lagrange_curve")

    def interpolate(self, x_values):
        """
        Interpole les points en utilisant aitken_neville pour obtenir x(t), y(t).
        """
        x_points = [(p[0], p[0]) for p in self.points]  # x -> x
        y_points = [(p[0], p[1]) for p in self.points]  # x -> y

        x_interp = [self.aitken_neville(x, x_points) for x in x_values]
        y_interp = [self.aitken_neville(x, y_points) for x in x_values]

        return np.array(x_interp), np.array(y_interp)

    def calculate_curvature(self, x_values):
        """
        Calcule la courbure à partir des points interpolés.
        """
        x_interp, y_interp = self.interpolate(x_values)

        # Calcul des dérivées première et seconde
        x_prime = np.gradient(x_interp, x_values)
        y_prime = np.gradient(y_interp, x_values)
        x_double_prime = np.gradient(x_prime, x_values)
        y_double_prime = np.gradient(y_prime, x_values)

        # Calcul de la courbure
        numerator = np.abs(x_prime * y_double_prime - y_prime * x_double_prime)
        denominator = (x_prime**2 + y_prime**2)**1.5
        curvature = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator != 0)
        return curvature

    def plot_curvature_Lagrange(self):
        """
        Trace la courbure de l'interpolation de Lagrange.
        """
        L = self.points.copy()
        L.sort(key=lambda pt: pt[0])

        x_values = np.linspace(L[0][0], L[-1][0], 500)
        curvature = self.calculate_curvature(x_values)

        # Tracé de la courbure
        plt.figure(figsize=(10, 5))
        plt.plot(x_values, curvature, label="Courbure (Lagrange)")
        plt.title("Courbure de l'interpolation de Lagrange (Aitken-Neville)")
        plt.xlabel("x")
        plt.ylabel("Courbure κ(x)")
        plt.grid()
        plt.legend()
        plt.show()


    def calculate_spline_cubic(self, x, y):
        """
        Calcule une spline cubique naturelle.
        """
        n = len(x) - 1
        h = np.diff(x)
        alpha = [0] + [3 * (y[i + 1] - y[i]) / h[i] - 3 * (y[i] - y[i - 1]) / h[i - 1] for i in range(1, n)] + [0]

        # Construction de la matrice tridiagonale
        l = np.ones(n + 1)
        mu = np.zeros(n)
        z = np.zeros(n + 1)
        a = y.copy()

        l[0] = 1
        for i in range(1, n):
            l[i] = 2 * (x[i + 1] - x[i - 1]) - h[i - 1] * mu[i - 1]
            mu[i] = h[i] / l[i]
            z[i] = (alpha[i] - h[i - 1] * z[i - 1]) / l[i]
        l[n] = 1

        c = np.zeros(n + 1)
        b = np.zeros(n)
        d = np.zeros(n)

        for j in range(n - 1, -1, -1):
            c[j] = z[j] - mu[j] * c[j + 1]
            b[j] = (a[j + 1] - a[j]) / h[j] - h[j] * (c[j + 1] + 2 * c[j]) / 3
            d[j] = (c[j + 1] - c[j]) / (3 * h[j])

        return a[:-1], b, c[:-1], d

    def spline_cubic_evaluate(self, x, y, X):
        """
        Évalue la spline cubique sur les points donnés.
        """
        a, b, c, d = self.calculate_spline_cubic(x, y)
        n = len(x) - 1
        Y = np.zeros_like(X)

        for i in range(len(X)):
            for j in range(n):
                if x[j] <= X[i] <= x[j + 1]:
                    dx = X[i] - x[j]
                    Y[i] = a[j] + b[j] * dx + c[j] * dx**2 + d[j] * dx**3
                    break
        return Y

    def draw_c2_curve(self, superp):
        self.canvas.delete("C2_curve")

        if not superp:
            self.canvas.delete("curve")
            self.canvas.delete("curve2")
            self.canvas.delete("lagrange_curve")

        L = self.points.copy()
        L.sort(key=lambda pt: pt[0])

        x = [L[i][0] for i in range(len(L))]
        y = [L[i][1] for i in range(len(L))]

        X = np.linspace(min(x), max(x), 500)

        # Évaluer la spline cubique
        Y = self.spline_cubic_evaluate(x, y, X)
        # Tracé de la spline
        for i in range(len(X) - 1):
            self.canvas.create_line(X[i], Y[i], X[i + 1], Y[i + 1], fill="black", width=2, tags="C2_curve")

    def superposer(self):
        self.draw_curve_methode_1(True)
        self.draw_curve_methode_2(True)
        self.draw_lagrange_curve(True)
        self.draw_c2_curve(True)

if __name__ == "__main__":
    app = HermiteSplineApp()
    app.mainloop()

