import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.title("🤖 AI Learning - Study vs Marks")

# Data (relatable for kids)
x_data = np.array([1, 2, 3, 4, 5, 6])
y_data = np.array([30, 40, 50, 65, 75, 85])

# Session state (to store values)
if "weight" not in st.session_state:
    st.session_state.weight = 0.0
    st.session_state.loss_history = []
    st.session_state.step = 0

learning_rate = 0.05

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
    w = st.session_state.weight

    loss = compute_loss(w)
    grad = compute_gradient(w)

    new_w = w - learning_rate * grad

    st.session_state.weight = new_w
    st.session_state.loss_history.append(loss)
    st.session_state.step += 1

# Buttons
col1, col2, col3 = st.columns(3)

if col1.button("▶️ Next Step"):
    train_step()

if col2.button("✅ Complete Training"):
    for _ in range(30):
        train_step()

if col3.button("🔄 Reset"):
    st.session_state.weight = 0.0
    st.session_state.loss_history = []
    st.session_state.step = 0

# Current values
w = st.session_state.weight
loss = compute_loss(w)

# Plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,4))

# Graph 1: Data + AI line
ax1.scatter(x_data, y_data, color='blue', label="Students data")
ax1.plot(x_data, predict(x_data, w), 'r', label="AI line")
ax1.set_title("Study vs Marks")
ax1.set_xlabel("Hours")
ax1.set_ylabel("Marks")
ax1.legend()

# Graph 2: Loss curve
ax2.plot(st.session_state.loss_history, color='purple')
ax2.set_title("Error decreasing")
ax2.set_xlabel("Steps")
ax2.set_ylabel("Error")

st.pyplot(fig)

# Step explanation
st.markdown(f"""
### 📚 Learning Steps

**Step:** {st.session_state.step}

**1️⃣ Equation:**  
Marks = weight × hours  

**2️⃣ Current weight:** {w:.2f}

**3️⃣ Error (loss):** {loss:.2f}

**4️⃣ Gradient (derivative):** {compute_gradient(w):.2f}  
👉 tells how to improve  

**5️⃣ Update:**  
new weight = old - learning_rate × gradient  

✅ AI is learning step by step!
""")
