# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.1
# ---

# %%
import numpy as np
import matplotlib.pyplot as plt

# %%
#Problem setup & data
x_train = np.array([1.0, 2.0])      # sizes: 1000 sqft, 2000 sqft
y_train = np.array([300.0, 500.0])  # prices: $300k, $500k

# %%
print(f"x_train = {x_train}")
print(f"y_train = {y_train}")

# %%
#Number of training examples (m)
print(f"x_train.shape: {x_train.shape}")
m = x_train.shape[0]
print(f"Number of training examples is: {m}")

# %%
#We can also use len():
m_len = len(x_train)
print(f"Number of training examples using len(): {m_len}")

# %%
#Training example (x^(i), y^(i))
for i in range(m):
    x_i = x_train[i]
    y_i = y_train[i]
    print(f"(x^({i}), y^({i})) = ({x_i}, {y_i})")

# %%
#Plotting the data
plt.figure()
plt.scatter(x_train, y_train, marker='x', c='r')
plt.title("Housing Prices")
plt.ylabel('Price (in 1000s of dollars)')
plt.xlabel('Size (1000 sqft)')
plt.show()

# %%
#Model function ( Computes the prediction of a linear model.)
# Arguments:
# x -- Input data m examples, a numpy array of shape (m,)
# w,b -- Model parameters
# Returns:
# f_wb -- Model predictions for each x[i], a numpy array of shape (m,)
def compute_model_output(x: np.ndarray, w: float, b: float) -> np.ndarray:
    m_local = x.shape[0]
    f_wb = np.zeros(m_local)
    for i in range(m_local):
        f_wb[i] = w * x[i] + b
    return f_wb

# %%
#(w,b)=(100.0,100.0) -> we should see that the result in not a good fit
w = 100.0
b = 100.0
print(f"Candidate parameters -> w: {w}, b: {b}")
tmp_f_wb = compute_model_output(x_train, w, b)

# %%
#Plot model vs data
plt.figure()
plt.plot(x_train, tmp_f_wb, label='Our Prediction')  # model line
plt.scatter(x_train, y_train, marker='x', c='r', label='Actual Values')
plt.title("Housing Prices")
plt.ylabel('Price (in 1000s of dollars)')
plt.xlabel('Size (1000 sqft)')
plt.legend()
plt.show()

# %%
# Try to fit the data exactly.
# Let the points be (x1, y1) = (1, 300) and (x2, y2) = (2, 500).
# Slope w = (y2 - y1) / (x2 - x1) = (500 - 300) / (2 - 1) = 200
# Intercept b from y = w*x + b -> b = y1 - w*x1 = 300 - 200*1 = 100
w_fit = 200.0
b_fit = 100.0
print(f"Fitted parameters -> w: {w_fit}, b: {b_fit}")
f_wb_fit = compute_model_output(x_train, w_fit, b_fit)

# %%
#Plot the exact-fit model vs data
plt.figure()
plt.plot(x_train, f_wb_fit, label='Exact Fit (w=200, b=100)')
plt.scatter(x_train, y_train, marker='x', c='r', label='Actual Values')
plt.title("Housing Prices (Exact Fit)")
plt.ylabel('Price (in 1000s of dollars)')
plt.xlabel('Size (1000 sqft)')
plt.legend()
plt.show()

# %%
#Predict the price for a 1200 sqft house.
x_i = 1.2  # 1200 sqft
cost_1200sqft = w_fit * x_i + b_fit  # using the exact-fit parameters
print(f"Predicted price for 1200 sqft: ${cost_1200sqft:.0f} thousand dollars")

