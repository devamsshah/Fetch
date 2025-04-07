import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import torch

# Set the title and favicon that appear in the Browser's tab bar.
st.set_page_config(
    page_title='Fetch internship excercise',
    page_icon=':earth_americas:', # This is an emoji shortcode. Could be a URL too.
)

# -----------------------------------------------------------------------------
# Declare some useful functions.

@st.cache_data
def get_data():
    """Grab GDP data from a CSV file.

    This uses caching to avoid having to read the file every time. If we were
    reading from an HTTP endpoint instead of a file, it's a good idea to set
    a maximum age to the cache with the TTL argument: @st.cache_data(ttl='1d')
    """

    # Instead of a CSV on disk, you could read from an HTTP endpoint here too.
    DATA_FILENAME = Path(__file__).parent/'data/DailyDataReceiptCountPrediction.csv'
    df = pd.read_csv(DATA_FILENAME)
    df_int = df.apply(pd.to_numeric, errors='coerce').fillna(0).astype(int)
    data = df_int.iloc[:, 1].values.reshape(-1, 1) # convert to an array of daily Receipt Count
    dataTensor = torch.tensor(data, dtype=torch.float32) # convert to torch

    return dataTensor

dataTensor = get_data()

# -----------------------------------------------------------------------------
# Draw the actual page

# Set the title that appears at the top of the page.
'''
# :earth_americas: Fetch internship excercise

**Problem Statement:** Given an array of 365 integers, each corresponding to the number of receipts scanned each **day** in the year 2021, train an ML model to predict the number of receipts scanned for each **month** in 2022
'''

# Add some spacing
'''# Solution:
### Plotting the provided data:'''

min_day = int(0)
max_day = int(365)

from_day, to_day = st.slider(
    'Which day range are you interested in?',
    min_value=min_day,
    max_value=max_day,
    value=(min_day, max_day),
)

# Filter data based on selected date range
filtered_tensor = dataTensor[from_day:to_day]
filtered_np = filtered_tensor.numpy().flatten()

# Plotting
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(filtered_np, label='Daily Data Receipt Count')
ax.set_title('Daily Data Receipt Count Over Time')
ax.set_xlabel('Day Index')
ax.set_ylabel('Receipt Count')
ax.legend()
ax.grid(True)

st.pyplot(fig)

'''
## Thoughts about the data

The data clearly looks linear with some noise. I hypothesise that the data can be modeled by a simple Guassian Conditional Model. The data looks something like the following graph of dummy data:

'''
st.image("data/DummyGaussianConditionalModel.png", caption="Image to demonstrate the structure of the given data", use_container_width=True)

'''
## A Simple Gaussian Conditional Model

### Regression Model:
'''
st.latex(r'''
y = \underbrace{b + m x}_{f(\mathbf{w}, x)} + \epsilon,\quad \epsilon \sim \mathcal{N}(0, \sigma^2)''')


st.markdown(r'**Input:** $ x \in \mathbb{R} $, **Output:** $ y \in \mathbb{R} $')
st.markdown(r'**Model parameter:** $ \mathbf{w} = [b, m] $')

'''
### Conditional probability model:
'''
st.latex(r'''
y \sim \mathcal{N}(f(\mathbf{w}, x), \sigma^2)
''')
st.markdown(r'It only models the relationship between $ y $ and $ x $, but does not model the probability distribution of $ x $.')


'''### LEAST SQUARES REGRESSION

This kind of data is usually best modelled by Least Squares Regression. I have not shown the work for derivation, Maximum Likelihood Estimate or Optimization for the least squares here since it is a very basic model, especially when working with python. 
'''

'''## Testing the Hypothesis
Now, I will see how well least squares model works on the given data. During this, we can also look at what predictions are made based on the training data and how well it fits the testing. You can adjust the sliders to play around with the number of training and testing data points to use in the prediction. Default is the first 80% of data points for training, the rest are for testing. '''

# Data pre-processing
data = dataTensor.flatten().unsqueeze(1)
days = torch.arange(1, 366, dtype=data.dtype, device=data.device).unsqueeze(1)

# Slider for selecting training range
start_day, end_day = st.slider(
    "Select training range (days)",
    min_value=1,
    max_value=365,
    value=(1, int(0.8*365))
)

# Convert to indices 
start_idx = start_day - 1
end_idx = end_day

# Training subset
train_days = days[start_idx:end_idx]
train_data = data[start_idx:end_idx]

# Construct matrix A for Ax = b
ones = torch.ones_like(train_days)
A = torch.hstack((train_days, ones))

# Least squares solution using only training data
x = torch.linalg.lstsq(A, train_data).solution
m, b = x[0].item(), x[1].item()

# Full model prediction (days 1 to 730)
extended_days = torch.arange(1, 731, dtype=data.dtype, device=data.device).unsqueeze(1)
m = torch.tensor(m, dtype=extended_days.dtype, device=extended_days.device)
b = torch.tensor(b, dtype=extended_days.dtype, device=extended_days.device)
predicted_values = extended_days * m + b

# Plotting
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(
    torch.arange(start_day, end_day + 1).cpu().numpy(),
    data[start_idx:end_idx].cpu().numpy(),
    label="Training Data",
    color='blue'
)
if start_idx > 0:
    ax.plot(
        torch.arange(1, start_day).cpu().numpy(),
        data[:start_idx].cpu().numpy(),
        label="Other Data",
        color='green'
    )
if end_idx < 365:
    ax.plot(
        torch.arange(end_day + 1, 366).cpu().numpy(),
        data[end_idx:].cpu().numpy(),
        color='green'
    )
ax.plot(
    extended_days.cpu().numpy(),
    predicted_values.cpu().numpy(),
    label="Linear Model Prediction",
    color='orange',
    linestyle='--'
)

ax.axvline(x=365, color='gray', linestyle=':', label="End of 2021 Data")
ax.set_xlabel("Day")
ax.set_ylabel("Value")
ax.set_title("Interactive Least Squares Model")
ax.legend()
ax.grid(True)
st.pyplot(fig)

'''Seeing that the Least Squares Regression indeed models the data well, I perform further testing. This is to rule out the possibility that the first section of the data is not corelated to the second half and vice versa. To do this, I will split the data into blocks of the following size (view slider, default is 30 since we have approximately 30 days in each month) and assign each alternating block to training and the rest to testing data. This method approximately divides the dataset into halves of training and testing which will be used to validate predictions.
'''


data_flat = dataTensor.flatten()

# Slider to control block size
block_size = st.slider(
    "Select block size for alternating train/test split",
    min_value=5,
    max_value=50,
    value=30,
    step=1
)

# Initialize train and test tensor filled with NaNs
train_array = torch.full_like(data_flat, float('nan'))
test_array = torch.full_like(data_flat, float('nan'))

# Create alternating blocks
num_blocks = (len(data_flat) + block_size - 1) // block_size  # ceil division
for i in range(num_blocks):
    start = i * block_size
    end = min(start + block_size, len(data_flat))
    if i % 2 == 0:
        train_array[start:end] = data_flat[start:end]
    else:
        test_array[start:end] = data_flat[start:end]

# Alternate block-based masking
train_array = torch.full_like(data_flat, float('nan'))
test_array = torch.full_like(data_flat, float('nan'))

num_blocks = (len(data_flat) + block_size - 1) // block_size
for i in range(num_blocks):
    start = i * block_size
    end = min(start + block_size, len(data_flat))
    if i % 2 == 0:
        train_array[start:end] = data_flat[start:end]
    else:
        test_array[start:end] = data_flat[start:end]

# --- Least Squares Model on Training Only ---
train_mask = ~torch.isnan(train_array)
train_days = torch.arange(1, 366, dtype=train_array.dtype, device=train_array.device)[train_mask].unsqueeze(1)
train_targets = train_array[train_mask].unsqueeze(1)
train_ones = torch.ones_like(train_days)
A_train = torch.cat((train_days, train_ones), dim=1)

# Least squares
result = torch.linalg.lstsq(A_train, train_targets)
x = result.solution.squeeze()
m, b = x[0].item(), x[1].item()

# Prediction on full range (day 1 to 365)
all_days = torch.arange(1, 366, dtype=train_array.dtype, device=train_array.device).unsqueeze(1)
m_tensor = torch.tensor(m, dtype=all_days.dtype, device=all_days.device)
b_tensor = torch.tensor(b, dtype=all_days.dtype, device=all_days.device)
predicted_line = m_tensor * all_days + b_tensor

# --- Plot 1: Training and Testing Data with Regression Line ---
fig1, ax1 = plt.subplots(figsize=(12, 6))
ax1.plot(all_days.cpu().numpy(), train_array.cpu().numpy(), 'b-', label='Training')
ax1.plot(all_days.cpu().numpy(), test_array.cpu().numpy(), 'g-', label='Testing')
ax1.plot(all_days.cpu().numpy(), predicted_line.cpu().numpy(), 'r-', label='Regression line', linewidth=2)
ax1.set_xlabel("Day")
ax1.set_ylabel("Value")
ax1.set_title("Least Squares Fit with Alternating Block Training/Test Data")
ax1.legend()
ax1.grid(True)
st.pyplot(fig1)

# --- Monthly / Block-Level Comparison (Test Segments Only) ---
actual_sums = []
predicted_sums = []

num_segments = len(test_array) // block_size

for i in range(num_segments):
    start = i * block_size
    end = start + block_size
    segment = test_array[start:end]

    if not torch.isnan(segment).all():
        actual_sum = torch.nansum(segment)
        actual_sums.append(actual_sum)

        # Predict for this range
        days_segment = torch.arange(start + 1, end + 1, dtype=segment.dtype, device=segment.device).unsqueeze(1)
        predicted_segment = m_tensor * days_segment + b_tensor
        predicted_sum = predicted_segment.sum()
        predicted_sums.append(predicted_sum)

# Convert to tensors
actual_sums_tensor = torch.stack(actual_sums)
predicted_sums_tensor = torch.stack(predicted_sums)
differences = actual_sums_tensor - predicted_sums_tensor

# --- Plot 1:
st.subheader("30-Day Test Block Predictions:")
for i, (act, pred, diff) in enumerate(zip(actual_sums_tensor, predicted_sums_tensor, differences)):
    st.write(f"Segment {i+1}: Actual = {act:.2f}, Predicted = {pred:.2f}, Diff = {diff:.2f}")

window = torch.tensor([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31], device=dataTensor.device)
monthly_chunks = torch.split(data_flat, window.tolist())  # Splitting into monthly chunks
monthly_actual_sums = torch.stack([chunk.sum() for chunk in monthly_chunks])  # Monthly sums

# --- Plot 2: Actual vs Predicted Sums ---
fig2, ax2 = plt.subplots(figsize=(10, 5))
ax2.plot(actual_sums_tensor.cpu().numpy(), 'go-', label='Actual sums (test blocks)')
ax2.plot(predicted_sums_tensor.cpu().numpy(), 'ro-', label='Predicted sums (LS line)')
ax2.set_xlabel("Segment Index")
ax2.set_ylabel("30-Day Sum")
ax2.set_title("Comparison of Actual vs Predicted 30-Day Sums")
ax2.legend()
ax2.grid(True)
plt.tight_layout()
st.pyplot(fig2)



# Predict values for next year (days 366 to 730)
future_days = torch.arange(366, 731, dtype=torch.float32, device=train_array.device).unsqueeze(1)
future_predictions = m * future_days + b 

# Compute monthly predicted sums
start_indices = torch.cumsum(torch.cat((torch.tensor([0], device=window.device), window[:-1])), dim=0)
end_indices = start_indices + window
monthly_predicted_sums = torch.stack([
    future_predictions[start:end].sum() for start, end in zip(start_indices, end_indices)
])

'''
We can clearly see that the prediction model follows the groups of test data really well. Hence we can continue with this model and predict the number of receipts scanned for each month in 2022'''

# Print predicted monthly sums
st.subheader("Predicted Monthly Sums for 2022")

# --- Plot 1: Bar Plot for Monthly Predictions ---
fig1, ax1 = plt.subplots(figsize=(14, 8))
bars = ax1.bar(range(1, 13), monthly_predicted_sums.cpu())

for i, bar in enumerate(bars):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width() / 2, height + 5, f"{height:.2f}",
             ha='center', va='bottom', fontsize=10)

ax1.set_xlabel("Month", fontsize=14)
ax1.set_ylabel("Predicted Sum", fontsize=14)
ax1.set_title("Predicted Monthly Sums for 2022 (Using Least Squares Model)", fontsize=16)
ax1.set_xticks(range(1, 13))
ax1.set_xticklabels([
    'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
    'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'
], fontsize=12)
ax1.tick_params(axis='y', labelsize=12)
ax1.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
st.pyplot(fig1)

# --- Plot 2: Actual vs Predicted Monthly Sums ---
fig2, ax2 = plt.subplots(figsize=(10, 5))
ax2.plot(monthly_actual_sums.cpu(), 'bo-', label='Original Data sum')
ax2.plot(monthly_predicted_sums.cpu(), 'go-', label='Least Squares sum')
ax2.set_xticks(range(0, 12))
ax2.set_xticklabels([
    'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
    'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'
], fontsize=12)
ax2.set_xlabel("Months")
ax2.set_ylabel("30-day Sum")
ax2.set_title("Comparison of training/testing (2021) vs Prediction (2022) 30-day Sums")
ax2.legend()
ax2.grid(True)
plt.tight_layout()
st.pyplot(fig2)

# --- Plot 3: Average Per-Day Receipt Count Per Month ---
monthly_actual_avgs = monthly_actual_sums / window
monthly_predicted_avgs = monthly_predicted_sums / window

fig3, ax3 = plt.subplots(figsize=(10, 5))
ax3.plot(monthly_actual_avgs.cpu(), 'bo-', label='Original Data')
ax3.plot(monthly_predicted_avgs.cpu(), 'go-', label='Least Squares')
ax3.set_xticks(range(0, 12))
ax3.set_xticklabels([
    'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
    'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'
], fontsize=12)
ax3.set_xlabel("Months")
ax3.set_ylabel("Number of Receipts Scanned")
ax3.set_title("Comparison of Expected Daily Receipts: 2021 vs 2022 Prediction")
ax3.legend()
ax3.grid(True)
plt.tight_layout()
st.pyplot(fig3)

'''
### Observations 

At first glance, there seems to be **overfitting** in the model. The **lack of data may be responsiable for some overfitting**. But, we observe **dips and peaks at exactly the same positions** in both training and test data primarily because of the **variance in the number of days in each month**.

---
### Remarks
I do not model the noise (i.e. variance parameters of the gaussian) here because I am only supposed to predict based on each month. Moreover, summing over a sizable "chunk" of the data set (approximately 30 in this case) cancels out the noise and predicts the data fairly accurately. 

Some things like plotting and splitting into chunks might have had better performance with numpy or similar libraries, however, the instructions said that the excercise was being judged on skills in PyTorch or TensorFlow, hence, I chose to use PyTorch. 

---

### Additional Factors That May Contribute to Cost Prediction

Here are a few important considerations that I would include given more resources/data:

- **Time of the year**  
  Individuals typically tend to shop more during the end of the year/winter season due to events like Halloween, Thanksgiving, Christmas, etc concentrated around that time. However, this clearly doesn't seem to be the case. The growth of the company is steadily increasing over the months which brings me to the next point...

- **Growth of the company over the months**  
  As the company grows, it reaches more customers, leading to **more downloads and more receipts scanned per customer**.

- **Growth of the company over the year**  
  Similar logic applies here.  

- **External economic factors**  
  In 2021, **COVID-19 was at its peak** in the USA. The government provided **stimulus checks (up to $1400)** in April 2021.  
  This resulted in **higher spending** than usual for many households.  
  In contrast, there were **no stimulus checks in 2022**, possibly leading to **lower sales**.
  '''