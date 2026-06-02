import tkinter as tk
from tkinter import ttk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Create window
root = tk.Tk()
root.title("Derivative Learning for Kids")

# Graph area
fig, ax = plt.subplots()
canvas = FigureCanvasTkAgg(fig, master=root)
canvas.get_tk_widget().pack()

# Info labels (show steps + equation)
info = tk.Label(root, text="", font=("Arial", 11), justify="left")
info.pack(pady=10)

# Update function
def update(val):
    point = slider.get()

    # Equation: y = x^2
    x = np.linspace(-10, 10, 100)
    y = x**2

    # Step 1: original function
    y_value = point**2

    # Step 2: derivative
    slope = 2 * point

    # Step 3: tangent line equation
    tangent = slope * (x - point) + y_value

    # Draw graph
    ax.clear()
    ax.plot(x, y, label="y = x²")
    ax.plot(x, tangent, '--', label="tangent line")
    ax.scatter(point, y_value, color='red')

    ax.set_title(f"At x = {point:.2f}")
    ax.legend()
    canvas.draw()

    # Direction meaning
    if slope > 0:
        meaning = "Going UP ⬆️"
    elif slope < 0:
        meaning = "Going DOWN ⬇️"
    else:
        meaning = "Flat ➖"

    # Show full steps for kids
    info.config(text=
        f"STEP 1: Equation\n"
        f"y = x²\n\n"
        f"STEP 2: Put value x = {point:.1f}\n"
        f"y = {point:.1f}² = {y_value:.1f}\n\n"
        f"STEP 3: Derivative\n"
        f"dy/dx = 2x\n\n"
        f"STEP 4: Put x = {point:.1f}\n"
        f"slope = 2 × {point:.1f} = {slope:.1f}\n\n"
        f"STEP 5: Meaning\n"
        f"The curve is {meaning}\n\n"
        f"STEP 6: Tangent Line\n"
        f"y = {slope:.1f}(x - {point:.1f}) + {y_value:.1f}"
    )

# Slider
slider = ttk.Scale(root, from_=-10, to=10, command=update)
slider.set(2)
slider.pack()

# Initial call
update(0)

root.mainloop()