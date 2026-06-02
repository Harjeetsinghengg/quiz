import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

st.title("🤖 AI Learning - Study vs Marks")

# Data
x_data = np.array([1, 2, 3, 4, 5, 6])
y_data = np.array([30, 40, 50, 65, 75, 85])

# Session state
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

# Values
w = st.session_state.weight
loss = compute_loss(w)
grad = compute_gradient(w)

# 📊 Graphs
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,4))

ax1.scatter(x_data, y_data, color='blue', label="Real data")
ax1.plot(x_data, predict(x_data, w), 'r', label="AI line")
ax1.set_title("Study vs Marks")
ax1.legend()

ax2.plot(st.session_state.loss_history, color='purple')
ax2.set_title("Error decreasing")

st.pyplot(fig)

# ✅ TABLE: Prediction Breakdown
st.subheader("📊 Step 1: Multiplication (Prediction)")

y_pred = predict(x_data, w)

table1 = pd.DataFrame({
    "Hours (x)": x_data,
    "Weight (w)": [w]*len(x_data),
    "Calculation (w × x)": [f"{w:.2f} × {x}" for x in x_data],
    "Predicted Marks (ŷ)": np.round(y_pred, 2),
    "Actual Marks (y)": y_data
})

st.dataframe(table1)

# ✅ TABLE: Error Calculation
st.subheader("📉 Step 2: Error (Difference)")

errors = y_data - y_pred

table2 = pd.DataFrame({
    "Hours": x_data,
    "Actual (y)": y_data,
    "Predicted (ŷ)": np.round(y_pred, 2),
    "Error (y - ŷ)": np.round(errors, 2),
    "Squared Error": np.round(errors**2, 2)
})

st.dataframe(table2)

# ✅ TABLE: Gradient Calculation
st.subheader("📐 Step 3: Derivative (Gradient)")

gradient_table = pd.DataFrame({
    "Hours (x)": x_data,
    "Error (y - ŷ)": np.round(errors, 2),
    "x × Error": np.round(x_data * errors, 2)
})

st.dataframe(gradient_table)

# ✅ FINAL FORMULA STEP
st.subheader("🧠 Step 4: Update Formula")

st.markdown(f"""
**Formula:**

new weight = old weight − learning_rate × gradient  

**Values:**

- old weight = {w:.3f}  
- learning rate = {learning_rate}  
- gradient = {grad:.3f}

✅ Calculation:

new weight = {w:.3f} - {learning_rate} × ({grad:.3f})  
new weight = {w - learning_rate * grad:.3f}
""")

# ✅ SIMPLE EXPLANATION
st.info(f"""
✅ STEP {st.session_state.step}

AI is learning like a student:

1. Multiply → predict marks  
2. Compare → find error  
3. Use derivative → understand mistake  
4. Update rule → improve  

👉 Better steps → less error → smarter AI 🎯
""")
