#!/usr/bin/env python
# coding: utf-8

# ### Anamitra productivity on sales data 

# ### Imported Libraries

# In[1]:


import pandas as pd
import numpy as np # Statistical calculations
import matplotlib.pyplot as plt # Visualizing diagrams
import seaborn as sns  # Visualizing diagrams
get_ipython().system('pip install mplcursors')
get_ipython().system('pip install tensorflow')
import mplcursors


# ### Importing the data

# In[2]:


df = pd.read_csv('Anamitra.csv')

#Visualizing the data
df.head(5)


# ### Data Visualization

# ##### **Importance of features:-.**

# **Timestamp**: This indicates the date and time when the transaction occurred. In this dataset, the timestamp is in the format "DD-MM-YYYY".
# 
# **Design Number:** This is a unique identifier for each jewelry design. In this dataset, it appears as alphanumeric codes like "BBR-001".
# 
# **Category:** This specifies the category or type of jewelry. For example, "Baby Bracelet" indicates that the item belongs to the segment of jewelry designed for babies.
# 
# **Qty:** This denotes the quantity of the jewelry item being transacted or processed.
# 
# **Carat:** This refers to the purity or fineness of the metal used in the jewelry, typically measured in karats (kt).
# 
# **Gross Weight:** This indicates the total weight of the jewelry item, including any additional components such as stones or embellishments.
# 
# **Stone Weight:** If applicable, this represents the weight of any stones incorporated into the jewelry design.
# 
# **Net Weight:** This is the final weight of the jewelry item after deducting the weight of any additional components, such as stones or embellishments, from the gross weight.
# 
# **Dispatched:** This field likely indicates whether the item has been dispatched or shipped to the customer.
# 
# **Remarks:** This field may contain any additional notes or comments related to the transaction or item.
# 
# **Sales Amount:** This represents the monetary value or price associated with the sale lry item.
# elry item.

# - Check the data for thorough understanding do cleaning as required.
# - First five entries of data presented

# In[3]:


df.tail()


# - First Five rows of the Anamitra sales data

# In[4]:


df.tail(5)


# - Last Five rows of the Anamita sales data

# #### Checking for the type and information of the data pattern. 

# In[5]:


# Type of the data
df.info()


# Checking for the statistical values of the data

# In[6]:


df.describe()


# In[7]:


# for index, row in df.iterrows():
#     print("Date:", row['Timestamp'], "Sales Amount:", row['Sales Amount'], row['Dispatched'])


# Sales data for each day from 2021 to 2024

# ### Data Cleaning 

# In[8]:


df.isnull().sum()


# **All the columns which include the Nan values**

# #### Filling all the missing Data in category

# In[9]:


for index, row in df.iterrows():
    if row['Design Number'].startswith('SBK'):
        df.at[index, 'Category'] = 'Silver Baby kada'

print(df[df['Design Number'].str.startswith('SBK')]['Category'].unique())


# In[10]:


for index, row in df.iterrows():
    if row['Design Number'].startswith('AN'):
        df.at[index, 'Category'] = 'Anamitra bali'

print(df[df['Design Number'].str.startswith('AN')]['Category'].unique())


# In[11]:


for index, row in df.iterrows():
    if row['Design Number'].startswith('BK'):
        df.at[index, 'Category'] = 'Baby kada'

print(df[df['Design Number'].str.startswith('BK')]['Category'].unique())


# In[12]:


for index, row in df.iterrows():
    if row['Design Number'].startswith('SBBR'):
        df.at[index, 'Category'] = 'Silver Baby Bracelet'

print(df[df['Design Number'].str.startswith('SBBR')]['Category'].unique())


# In[13]:


for index, row in df.iterrows():
    if row['Design Number'].startswith('ER'):
        df.at[index, 'Category'] = 'Earings'

print(df[df['Design Number'].str.startswith('ER')]['Category'].unique())


# In[14]:


for index, row in df.iterrows():
    if row['Design Number'].startswith('ANB'):
        df.at[index, 'Category'] = 'Anamitra Bali'

print(df[df['Design Number'].str.startswith('ANB')]['Category'].unique())


# In[15]:


for index, row in df.iterrows():
    if row['Design Number'].startswith('KP'):
        df.at[index, 'Category'] = 'Kids Pendant'

print(df[df['Design Number'].str.startswith('KP')]['Category'].unique())


# In[16]:


for index, row in df.iterrows():
    if row['Design Number'].startswith('YBR'):
        df.at[index, 'Category'] = 'Youngsters bracelet'

print(df[df['Design Number'].str.startswith('YBR')]['Category'].unique())


# In[17]:


for index, row in df.iterrows():
    if row['Design Number'].startswith('SYBR'):
        df.at[index, 'Category'] = 'Silver youngster bracelet'

print(df[df['Design Number'].str.startswith('SYBR')]['Category'].unique())


# In[18]:


for index, row in df.iterrows():
    if row['Design Number'].startswith('CP'):
        df.at[index, 'Category'] = 'chain pendant'

print(df[df['Design Number'].str.startswith('CP')]['Category'].unique())


# In[19]:


for index, row in df.iterrows():
    if row['Design Number'].startswith('WCB'):
        df.at[index, 'Category'] = 'Watch chain bracelet'

print(df[df['Design Number'].str.startswith('WCB')]['Category'].unique())


# In[20]:


for index, row in df.iterrows():
    if row['Design Number'].startswith('BBR'):
        df.at[index, 'Category'] = 'Baby Bracelet'

print(df[df['Design Number'].str.startswith('BBR')]['Category'].unique())


# In[21]:


for index, row in df.iterrows():
    if row['Design Number'].startswith('BN'):
        df.at[index, 'Category'] = 'Baby Nazariya'

print(df[df['Design Number'].str.startswith('BN')]['Category'].unique())


# In[22]:


for index, row in df.iterrows():
    if row['Design Number'].startswith('KR'):
        df.at[index, 'Category'] = 'Kids ring'

print(df[df['Design Number'].str.startswith('KR')]['Category'].unique())


# In[23]:


for index, row in df.iterrows():
    if row['Design Number'].startswith('KP'):
        df.at[index, 'Category'] = 'Kids pendant'

print(df[df['Design Number'].str.startswith('KP')]['Category'].unique())


# In[24]:


for index, row in df.iterrows():
    if row['Design Number'].startswith('K CH'):
        df.at[index, 'Category'] = 'Kids chain'

print(df[df['Design Number'].str.startswith('K CH')]['Category'].unique())


# In[25]:


for index, row in df.iterrows():
    if row['Design Number'].startswith('EC'):
        df.at[index, 'Category'] = 'Elegance Chain'

print(df[df['Design Number'].str.startswith('EC')]['Category'].unique())


# In[26]:


df['Category'].fillna('bulk order', inplace=True)

# Update the 'Category' column to have consistent capitalization for "Baby Nazariya"
df.loc[df['Category'] == 'Baby Nazariya', 'Category'] = 'Baby Nazariya'

# Convert 'Category' column to lowercase for case-insensitive comparison
df['Category'] = df['Category'].str.lower()

print(df.head())


# In[27]:


df.isnull().sum()


# In[28]:


df.tail()


# In[29]:


missing_categories = df[df['Category'].isna()]

# Display the rows with missing categories
print("Rows with Missing Categories:")
print(missing_categories.to_string(index=False))


# - The above two categories are missing because it was a bulk order.

# In[30]:


df.isnull().sum()


# **Filled all Nan values with orignal data**

# ### Exploratary Data Analysis

# What is EDA exploratary data analysis (EDA) is used by data scientist to analyze and investigate datasets and summarize main characters often employing data visualization methods.

# In[31]:


df.isnull().sum()


# **There are no nan values.**

# In[32]:


import warnings


# In[33]:


df.info()


# ### **Sales trend over the time**

# In[34]:


# Plotting Sales Trends Over Time
plt.figure(figsize=(8, 6))
plt.plot(df['Timestamp'], df['Sales Amount'], marker='o', linestyle='-')
plt.title('Sales Trends Over Time')
plt.xlabel('Timestamp')
plt.ylabel('Sales Amount')
plt.xticks(rotation=45)
plt.grid(True)
plt.show()


# ### Correlation Matrix 
# - Relation is calculated between all features how they are linked with each other while the product being sold.

# In[35]:


import plotly.graph_objects as go

# Selecting only numeric columns
numeric_df = df.select_dtypes(include=[np.number])

# Create correlation matrix
corr_matrix = numeric_df.corr()

# Define heatmap
heatmap = go.Heatmap(
    z=corr_matrix.values,
    x=corr_matrix.columns,
    y=corr_matrix.columns,
    colorscale='RdBu',  # Choose a color scale
    zmin=-1, zmax=1,  # Set the range of values
    colorbar=dict(title='Correlation')
)

# Define layout
layout = go.Layout(
    title='Correlation Matrix',
    xaxis=dict(title='Features'),
    yaxis=dict(title='Features')
)

# Create figure
fig = go.Figure(data=[heatmap], layout=layout)

# Show plot
fig.show()


# ## Pending 

# ### The correlation is mentioned above with ----

# #### Checking the relationship betweeen Gross weight and Net weight

# In[36]:


get_ipython().system('pip install mplcursors')
import mplcursors


# In[37]:


plt.figure(figsize=(8, 6))
sns.lineplot(data=df, x='Gross Weight', y='Net Weight')
plt.title('Relationship between Gross Weight and Net Weight')
plt.xlabel('Gross Weight')
plt.ylabel('Net Weight')
plt.grid(True)
plt.show()


# In[38]:


df.columns


# In[39]:


import plotly.express as px


# In[40]:


categories = df['Category'].unique()

# Print all categories
print("All Categories:")
for category in categories:
    print(category)


# ### Sales completed category wise 

# # changes made 

# In[41]:


# # Convert 'Category' column to lowercase for case-insensitive comparison
# df = df['Category'].str.lower()

# # Group by the lowercase version of the category and calculate total sales amount
# category_sales = df.groupby('Category')['Sales Amount'].sum()

# # Print the total sales amount for each categorya
# print(category_sales)


# ### Sales distribution by category

# In[42]:


import plotly.express as px
import plotly.graph_objects as go


# In[43]:


import plotly.express as px

# Create a pie chart with Plotly
fig = px.pie(df, values='Sales Amount', names='Category', title='Sales Distribution by Category')

# Update layout for better visual appearance
fig.update_traces(textposition='inside', textinfo='percent+label')
fig.update_layout(
    legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
    title_x=0.5,
    title_y=0.1
)

fig.show()


# - **Baby Nazariya is the top-selling category with 42.9% percentage of the sales distribution, the second highest category is Anamitra bali sales done for 12.3% of the sales.** 
# 
# 
# - **Recommendations**
# - **Baby nazariya is the top selling caregory as it contributes the highest sales volume**

# In[44]:


df.tail()


# - **The average price of kurta button is 30420 Rupees and the product which is costing lowest is silver youngster bracelet**

# In[45]:


kurta_button_data = df[df['Category'] == 'kurta button']
print("Price of Kurta Button:")
print(kurta_button_data['Sales Amount'])


# #### Product Categories Distribution

# In[46]:


import pandas as pd
import plotly.graph_objs as go

df.drop_duplicates(inplace=True)
category_counts = df['Category'].value_counts()
categories = category_counts.index
counts = category_counts.values
fig = go.Figure(data=[
    go.Bar(x=categories, y=counts, hoverinfo='y', text=counts, textposition='auto')
])

fig.update_layout(
    title='Product Categories Distribution',
    xaxis=dict(title='Category', tickangle=45),
    yaxis=dict(title='Count'),
    hovermode='closest',  # Show hover info on nearest data point
    hoverlabel=dict(bgcolor="white", font_size=12),
)
fig.show()


# - Below are the varities in the carat 

# In[47]:


# Print unique values in the 'Carat' column
print(df['Carat'].unique())


# - At the starting of the website the company had different delivery methods and variable changes in category.

# - Total gross weight by dispatched and carat

# In[48]:


import seaborn as sns
import matplotlib.pyplot as plt
# Group the DataFrame by 'Dispatched' and 'Carat' and calculate total gross weight
total_gross_weight = df.groupby(['Dispatched', 'Carat'])['Gross Weight'].sum().reset_index()

# Print the first few rows to verify the calculation
print(total_gross_weight.head())

# Plotting
plt.figure(figsize=(6, 14))
sns.barplot(data=total_gross_weight, x='Dispatched', y='Gross Weight', hue='Carat', palette='viridis')
plt.title('Total Gross Weight by Dispatched and Carat', fontsize=16)
plt.xlabel('Dispatched', fontsize=14)
plt.ylabel('Total Gross Weight', fontsize=14)
plt.xticks(rotation=45, fontsize=12)
plt.yticks(fontsize=12)
plt.legend(title='Carat', fontsize=12)
plt.tight_layout()
plt.show()


# ##### Dispatch to customer sale has the highest amount of sale covered the total gross weight count for 22 karat sold out is 1414.30 but for 18 karat the gross weight sold out is 426.6 grams.

# In[49]:


print(df[['Timestamp', 'Sales Amount']]) 


# In[50]:


df.head()


# #### Converting timestamp column to date time column

# In[51]:


# Convert the 'Timestamp' column to datetime format
df['Timestamp'] = pd.to_datetime(df['Timestamp'])

daily_sales = df.groupby(df['Timestamp'].dt.date)['Sales Amount'].sum()

max_sales_date = daily_sales.idxmax()
max_sales_amount = daily_sales.max()

print("Date with the maximum sales amount:", max_sales_date)
print("Maximum sales amount on that date:", max_sales_amount)


# In[52]:


df.head()


# ##### The above is the date with the maximum sales happend on that particlular day

# In[53]:


df.tail()


# In[54]:


top_sales_dates = daily_sales.nlargest(5)  # Change 5 to the number of top dates you want to print

print("Top Date(s) with the highest sales amount(s):")
for date, amount in top_sales_dates.items():
    print("Date:", date, "Sales Amount:", amount)
import plotly.graph_objects as go

# Create bar plot
fig = go.Figure(data=[go.Bar(x=top_sales_dates.index, y=top_sales_dates.values,
                             marker_color='skyblue', marker_line_color='black', marker_line_width=1.5,
                             text=[f'₹{amount:.2f}' for amount in top_sales_dates.values], textposition='outside')])

# Update layout with dark theme and radiance effects
fig.update_layout(title='Top 5 sales maximum',
                  xaxis=dict(title='Dates', tickangle=-45, tickfont=dict(size=12)),
                  yaxis=dict(title='Total sales amount', tickfont=dict(size=12)),
                  plot_bgcolor='rgba(0, 0, 0, 0)',  # Transparent background
                  paper_bgcolor='rgba(0, 0, 0, 0)',  # Transparent background
                  font=dict(color='white'),  # Text color
                  hoverlabel=dict(font=dict(color='white')),  # Hover label text color
                  hovermode='x',
                  bargap=0.1,  # Gap between bars
                  uniformtext_minsize=8,  # Minimum text size
                  uniformtext_mode='hide',  # Hide text if it doesn't fit
                  template='plotly_dark'  # Dark theme template
                 )

# Show plot
fig.show()


# - **The above visualization is representing the top 5 dates where the maximum sales was Covered**

# In[55]:


# Group the DataFrame by 'Category' and get unique values for 'Design Number' within each group
category_models = df.groupby('Category')['Design Number'].unique()

# Print the models for each category
for category, models in category_models.items():
    print(f"Category: {category}")
    print("Models:", ", ".join(models))
    print()


# ### Sales which is covered in home or personal 

# In[56]:


df['Dispatched'].unique()


# ### Total sales is calculated in each category with total number of products sold.

# In[57]:


print("Data Types:")
print(df.dtypes)

personal_pickup_data = df[df['Dispatched'] == "Dispatch To Customer Sale-Personal"]
print("\nPersonal Pickup Data:")
print(personal_pickup_data)

dispatch_to_customer_sales = df[df['Dispatched'] == 'Dispatch To Customer Sale']
print("\n Dispatch To Customer Sales")
print(dispatch_to_customer_sales)

home_delivery_data = df[df['Dispatched'] == "Dispatch To Customer Sale-WebSite"]
print("\nHome Delivery Data:")
print(home_delivery_data)

personal_pickup_sales = personal_pickup_data['Sales Amount'].sum()
personal_pickup_qty = personal_pickup_data['Qty'].sum()

dispatch_to_customer_sales_sales = dispatch_to_customer_sales['Sales Amount'].sum()
dispatch_to_customer_sales_qty = dispatch_to_customer_sales['Qty'].sum()

home_delivery_sales = home_delivery_data['Sales Amount'].sum()
home_delivery_qty = home_delivery_data['Qty'].sum()

print("\nTotal Sales Amount for Personal Pickup:", personal_pickup_sales)
print("Total Quantity for Personal Pickup:", personal_pickup_qty)

print("\nTotal Sales Amount for Dispatch to Customer Sale:", dispatch_to_customer_sales_sales)
print("Total Quantity for Dispatch to Customer Sale:", dispatch_to_customer_sales_qty)

print("\nTotal Sales Amount for Home Delivery:", home_delivery_sales)
print("Total Quantity for Home Delivery:", home_delivery_qty)


# #### The total products sold and total amount recieved. 

# In[58]:


# Initialize variables for total sales amount and quantity
total_sales_amount = df['Sales Amount'].sum()
total_quantity = df['Qty'].sum()

print("Total Sales Amount:", total_sales_amount)
print("Total Quantity:", total_quantity)


# - The maximum sales is covered in the Home delivery which is 4658703

# In[59]:


get_ipython().system('pip install mplcursors')
import mplcursors


# ### Relationship between gross weight and Net weight

# ### All the categories from the data frame

# In[60]:


categories = df['Category'].unique()

# Print all categories
print("All Categories:")
for category in categories:
    print(category)


# In[61]:


import plotly.express as px
import plotly.graph_objects as go


# #### Monthly sales amount from 2021 March to March 2024

# In[62]:


import calendar
import pandas as pd
import plotly.graph_objects as go
from scipy.signal import find_peaks
# Assuming df is your DataFrame containing sales data

# Convert 'Timestamp' column to datetime
df['Timestamp'] = pd.to_datetime(df['Timestamp'])

# # Extract month and year from 'Timestamp'
df['Month'] = df['Timestamp'].dt.month
df['Year'] = df['Timestamp'].dt.year

# Group by year and month, and sum the sales amount
monthly_sales = df.groupby(['Year', 'Month'])['Sales Amount'].sum()

# Extend the x-axis labels to include the next four months
current_year = df['Year'].max()
current_month = df[df['Year'] == current_year]['Month'].max()
next_months = []
for i in range(1, 5):
    if current_month + i > 12:
        next_months.append((current_year + 1, current_month + i - 12))
    else:
        next_months.append((current_year, current_month + i))
next_labels = [f"{calendar.month_abbr[month]}-{year}" for (year, month) in next_months]

# Extend the monthly_sales DataFrame with next months
extended_index = monthly_sales.index.union(pd.MultiIndex.from_tuples(next_months))
monthly_sales = monthly_sales.reindex(extended_index, fill_value=0)

# Plot using Plotly
fig = go.Figure()

# Bar chart
fig.add_trace(go.Bar(
    x=monthly_sales.index.map(lambda x: f"{calendar.month_abbr[x[1]]}-{x[0]}"),
    y=monthly_sales.values,
    marker_color='skyblue',
    text=[f'₹{sales:,.2f}' for sales in monthly_sales.values],  # Include rupee symbol
    textposition='outside',
    name='Sales Amount',
    marker_line_color='black',  # Add marker line color
    marker_line_width=2,  # Add marker line width
    hoverinfo='x+y',  # Show both x and y information on hover
    hovertemplate='%{x}<br>Total Sales: ₹%{y:,.2f}',  # Customize hover template
))

# Find peaks and troughs in the sales data
peaks, _ = find_peaks(monthly_sales, height=0)
troughs, _ = find_peaks(-monthly_sales, height=0)

# Highlight peaks and troughs with markers
fig.add_trace(go.Scatter(
    x=monthly_sales.iloc[peaks].index.map(lambda x: f"{calendar.month_abbr[x[1]]}-{x[0]}"),
    y=monthly_sales.iloc[peaks].values,
    mode='markers',
    marker=dict(color='red', size=10),
    name='Peaks',
    hoverinfo='x+y',  # Show both x and y information on hover
    hovertemplate='%{x}<br>Peak Sales: ₹%{y:,.2f}',  # Customize hover template
))

fig.add_trace(go.Scatter(
    x=monthly_sales.iloc[troughs].index.map(lambda x: f"{calendar.month_abbr[x[1]]}-{x[0]}"),
    y=monthly_sales.iloc[troughs].values,
    mode='markers',
    marker=dict(color='green', symbol='triangle-down', size=60),
    name='Troughs',
    hoverinfo='x+y',  # Show both x and y information on hover
    hovertemplate='%{x}<br>Trough Sales: ₹%{y:,.2f}',  # Customize hover template
))

# Layout
fig.update_layout(
    title='Monthly Sales Amount',
    xaxis_title='Date',
    yaxis_title='Sales Amount (in ₹)',
    xaxis=dict(tickangle=45),
    yaxis=dict(showgrid=True, zeroline=False),
    showlegend=True,
    font=dict(size=16),
    width=1000,
    height=600,
)

fig.show()


# ### The above is month wise sales calculated and it is observed that the highest sales is covered in month of july 2022 which is 17 lakh. And also other peak sales which can be observed by visualizing the red triangle on barline.  

# In[63]:


# Define the DataFrame
monthly_sales_df = pd.DataFrame({'Date': monthly_sales.index.map(lambda x: f"{calendar.month_abbr[x[1]]}-{x[0]}"),
                                 'Sales Amount (in ₹)': monthly_sales.values})

# Apply high-tech styling
styled_df = (monthly_sales_df.style
              .set_properties(**{'text-align': 'center'})
              .set_table_styles([{'selector': 'th',
                                  'props': [('background-color', 'skyblue'),
                                            ('color', 'black'),
                                            ('font-size', '16px'),
                                            ('text-align', 'center')]}])
              .format({'Sales Amount (in ₹)': '₹{:,.2f}'.format}))

# Display the styled DataFrame
styled_df


# ### Libraries for model

# ### Model with streamlit install in it 

# In[64]:


# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import streamlit as st
import pickle
get_ipython().system('pip install streamlit-pandas-profiling')


# In[ ]:





# 
# ### **Data Pre-Processing for fitting the data to model for forecasting**

# In[65]:


import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# In[66]:


# Data Pre-Processing
df.drop_duplicates(inplace=True)


# ### Label Encoding 

# In[67]:


# Label Encoding
label_encoder = LabelEncoder()
df['Gross Weight'] = label_encoder.fit_transform(df['Gross Weight'])
df['Category'] = label_encoder.fit_transform(df['Category'])


# In[68]:


df.head()


# In[69]:


df.tail()


# In[70]:


# Splitting the data into features (X) and target (y)
X = df[['Month', 'Year', 'Gross Weight', 'Category']]
y = df['Sales Amount']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[71]:


# Scaling the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# In[72]:


get_ipython().system('pip install xgboost')


# In[73]:


from xgboost import XGBClassifier
import xgboost as xgb


# In[74]:


param_grid = {
    'n_estimators': [100, 150, 300],
    'max_depth': [None, 20, 25],
    'min_samples_split': [3, 6, 12],
    'min_samples_leaf': [2, 4, 5]
}

# Initialize the Random Forest regressor
rf = RandomForestRegressor(random_state=42)

# Initialize GridSearchCV
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)

# Fit the grid search to the data
grid_search.fit(X_train_scaled, y_train)

# Get the best parameters
best_params = grid_search.best_params_
print("Best Parameters:", best_params)

# Initialize the Random Forest regressor with the best parameters
best_rf = RandomForestRegressor(**best_params, random_state=42)

# Fit the model with the best parameters
best_rf.fit(X_train_scaled, y_train)

# Save the model and scaler
with open('best_rf_model.pkl', 'wb') as f:
    pickle.dump(best_rf, f)

with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# Predict on the training and testing data
y_pred_train_best = best_rf.predict(X_train_scaled)
y_pred_test_best = best_rf.predict(X_test_scaled)

# Evaluate the model
train_mse_best = mean_squared_error(y_train, y_pred_train_best)
test_mse_best = mean_squared_error(y_test, y_pred_test_best)
train_mae_best = mean_absolute_error(y_train, y_pred_train_best)
test_mae_best = mean_absolute_error(y_test, y_pred_test_best)
train_r2_best = r2_score(y_train, y_pred_train_best)
test_r2_best = r2_score(y_test, y_pred_test_best)

# Print the evaluation metrics
print("\nWith Best Parameters:")
print("Training MSE:", train_mse_best)
print("Testing MSE:", test_mse_best)
print("Training MAE:", train_mae_best)
print("Testing MAE:", test_mae_best)
print("Training R^2:", train_r2_best)
print("Testing R^2:", test_r2_best)


# ### Printing the quantity of training and testing set 

# In[75]:


# Print the quantity of each training and testing dataset
print("Training data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)

# Printing actual and predicted values for training and testing set
print("Training Set:")
print("Actual\t\tPredicted")
for actual, predicted in zip(y_train, y_pred_train_best):
    print(f"{actual}\t\t{predicted}")

print("\nTesting Set:")
print("Actual\t\tPredicted")
for actual, predicted in zip(y_test, y_pred_test_best):
    print(f"{actual}\t\t{predicted}")


# ### Streamlit

# In[76]:


## Streamlit Application
# Define the prediction function
def predict_sales(month, year, gross_weight, category):
    with open('best_rf_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    
    input_data = pd.DataFrame([[month, year, gross_weight, category]], columns=['Month', 'Year', 'Gross Weight', 'Category'])
    input_data_scaled = scaler.transform(input_data)
    prediction = model.predict(input_data_scaled)
    return prediction[0]

# Streamlit app
st.title('Sales Prediction Model')
st.write('Enter the necessary parameters to predict sales.')

month = st.number_input('Month', min_value=1, max_value=12, step=1)
year = st.number_input('Year', min_value=2021, max_value=2024, step=1)
gross_weight = st.number_input('Gross Weight', min_value=0.0, step=0.1)
category = st.number_input('Category', min_value=0, step=1)

if st.button('Predict'):
    prediction = predict_sales(month, year, gross_weight, category)
    st.write(f'The predicted sales value is: {prediction}')


# In[ ]:





# In[ ]:





# In[77]:


print("X_train_data first entries", X_train.head())
print("X-test_data first entries", X_test.head())


# ### Printing trainig and testing accuracy

# In[78]:


# Printing actual and predicted values for training set
print("Training Set:")
print("Actual\t\tPredicted")
for actual, predicted in zip(y_train, y_pred_train_best):
    print(f"{actual}\t\t{predicted}")

    
# Printing actual and predicted values for testing set
print("\nTesting Set:")
print("Actual\t\tPredicted")
for actual, predicted in zip(y_test, y_pred_test_best):
    print(f"{actual}\t\t{predicted}")


# In[79]:


# Scatter plot of actual vs. predicted values for testing set
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_test_best, color='blue', alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Actual vs. Predicted Values (Testing Set)')
plt.show()


# In[80]:


# Residual plot for testing set
plt.figure(figsize=(10, 6))
residuals = y_test - y_pred_test_best
sns.residplot(x=y_pred_test_best, y=residuals, lowess=True, color='green', scatter_kws={'alpha': 0.5})
plt.xlabel('Predicted')
plt.ylabel('Residuals')
plt.title('Residual Plot (Testing Set)')
plt.axhline(y=0, color='k', linestyle='--', linewidth=2)
plt.show()


# In[81]:


pip install streamlit


# In[82]:


import pickle 


# In[ ]:




