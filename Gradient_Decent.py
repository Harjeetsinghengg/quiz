import tkinter as tk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Window
root = tk.Tk()
root.title("AI Learning - Study vs Marks")
root.geometry("950x650")

# Graph
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8,4))
canvas = FigureCanvasTkAgg(fig, master=root)
canvas.get_tk_widget().pack()

# Info label (steps for kids)
info = tk.Label(root, text="", font=("Arial", 10), justify="left")
info.pack(pady=10)

# Data
x_data = np.array([1, 2, 3, 4, 5, 6])
y_data = np.array([30, 40, 50, 65, 75, 85])

# Model variables
weight = 0.0
learning_rate = 0.05
loss_history = []
step_count = 0

# Functions
def predict(x, w):
    return w * x

def compute_loss(w):
    y_pred = predict(x_data, w)
    return np.mean((y_data - y_pred)**2)

def compute_gradient(w):
    y_pred = predict(x_data, w)
    return -2 * np.mean(x_data * (y_data - y_pred))

# Training step
def train_step():
    global weight, step_count

    y_pred = predict(x_data, weight)
    loss = compute_loss(weight)
    grad = compute_gradient(weight)

    # Update rule
    new_weight = weight - learning_rate * grad

    loss_history.append(loss)
    step_count += 1

    # Plot data
    ax1.clear()
    ax1.scatter(x_data, y_data, color='blue', label="Real Data")
    ax1.plot(x_data, predict(x_data, new_weight), 'r', label="AI Line")
    ax1.set_title("Study Hours vs Marks")
    ax1.set_xlabel("Hours")
    ax1.set_ylabel("Marks")
    ax1.legend()

    # Plot loss
    ax2.clear()
    ax2.plot(loss_history, color='purple')
    ax2.set_title("Error Reducing")
    ax2.set_xlabel("Steps")
    ax2.set_ylabel("Error")

    canvas.draw()

    # Show simple steps (kid friendly)
    info.config(text=
        f"STEP {step_count}\n\n"
        f"1️⃣ Equation: Marks = Weight × Hours\n\n"
        f"2️⃣ Current Weight = {weight:.2f}\n\n"
        f"3️⃣ Predictions:\n"
        f"   Example: 1h → {weight:.2f} mark\n\n"
        f"4️⃣ Compare with real marks → find ERROR\n"
        f"   Error = {loss:.2f}\n\n"
        f"5️⃣ Derivative (Gradient) = {grad:.2f}\n"
        f"   👉 tells how to improve\n\n"
        f"6️⃣ Update Weight:\n"
        f"   new = {weight:.2f} - 0.05 × ({grad:.2f})\n"
        f"   new weight = {new_weight:.2f}\n\n"
        f"✅ AI improves step by step!"
    )

    # Apply update AFTER showing steps
    weight = new_weight

# Buttons
def next_step():
    train_step()

def complete_training():
    for _ in range(30):
        train_step()
        root.update()
        root.after(150)

def reset():
    global weight, loss_history, step_count
    weight = 0.0
    loss_history = []
    step_count = 0

    ax1.clear()
    ax2.clear()
    canvas.draw()

    info.config(text="Press 'Next Step' to start learning ✅")

# Button frame
frame = tk.Frame(root)
frame.pack(pady=10)

tk.Button(frame, text="Next Step ▶️", command=next_step).pack(side="left", padx=10)
tk.Button(frame, text="Complete ✅", command=complete_training).pack(side="left", padx=10)
tk.Button(frame, text="Reset 🔄", command=reset).pack(side="left", padx=10)

# Initial message
reset()

root.mainloop()