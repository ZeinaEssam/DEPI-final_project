#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns


# In[3]:


data = pd.read_csv("C:/Users/DELL/Desktop/data.csv")
print(data.head(1))


# In[4]:


data.info()


# data-preprocessing

# In[5]:


data.dropna(subset=['Order Quantity'], axis=0, inplace=True)
data.info()


# In[6]:


#  correct the column name by removing any leading or trailing spaces
data.columns = data.columns.str.strip()

#  proceed with the data cleaning
data['Cost Price'] = data['Cost Price'].replace('[\$,]', '', regex=True).astype(float)
data['Retail Price'] = data['Retail Price'].replace('[\$,]', '', regex=True).astype(float)
data['Profit Margin'] = data['Profit Margin'].replace('[\$,]', '', regex=True).astype(float)
data['Profit Margin'] = data['Profit Margin'].replace('[\$,]', '', regex=True).astype(float)
data['Sub Total'] = data['Sub Total'].replace('[\$,]', '', regex=True).astype(float)
data['Order Total'] = data['Order Total'].replace('[\$,]', '', regex=True).astype(float)
data['Shipping Cost'] = data[ 'Shipping Cost'].replace('[\$,]', '', regex=True).astype(float)
data['Total'] = data['Total'].replace('[\$,]', '', regex=True).astype(float)
data['Discount $'] = data['Discount $'].replace('[\$,]', '', regex=True).astype(float)


# In[7]:


missing_values=data.isnull()


for coulmns in missing_values.columns.values.tolist():
    print(coulmns)
    print(missing_values[coulmns].value_counts())
    print("")


# In[8]:


# simply drop whole row with NaN in "price" column
data.dropna(subset=["Address"], axis=0, inplace=True)

# reset index, because we droped two rows
data.reset_index(drop=True, inplace=True)


# In[9]:


data.info()


# In[10]:


data.head()


# In[11]:


data.describe()


# In[12]:


# Convert 'Order Date' and 'Ship Date' columns to datetime
data['Order Date'] = pd.to_datetime(data['Order Date'], format='%d-%m-%Y', errors='coerce')
data['Ship Date'] = pd.to_datetime(data['Ship Date'], format='%d-%m-%Y', errors='coerce')


# In[13]:


data.dtypes


# In[14]:


# Sales by Product Category
pivot_table = data.pivot_table(index='Product Category', values='Total', aggfunc='sum')
print(pivot_table)


# In[15]:



fig, axs = plt.subplots(2, 2, figsize=(16, 12))

# Sales Trend Plot 
sales_trend_monthly = data.groupby(pd.Grouper(key='Order Date', freq='M')).agg({'Total': 'sum'}).reset_index()
sns.lineplot(x='Order Date', y='Total', data=sales_trend_monthly, ax=axs[0, 0], marker='o', color='b')
axs[0, 0].set_title('Monthly Sales Trend Over Time')
axs[0, 0].set_xlabel('Month')
axs[0, 0].set_ylabel('Total Sales')
axs[0, 0].tick_params(axis='x', rotation=45)

# Sales by Product Category Plot
product_sales = data.groupby('Product Category')['Total'].sum().reset_index()
sns.barplot(x='Product Category', y='Total', data=product_sales, ax=axs[0, 1], palette='viridis')
axs[0, 1].set_title('Sales by Product Category')
axs[0, 1].set_xlabel('Product Category')
axs[0, 1].set_ylabel('Total Sales')
axs[0, 1].tick_params(axis='x', rotation=45)

# Sales by Region Plot 
region_sales = data.groupby('City')['Total'].sum().reset_index()
sns.barplot(x='City', y='Total', data=region_sales, ax=axs[1, 0], palette='coolwarm')
axs[1, 0].set_title('Sales by Region')
axs[1, 0].set_xlabel('City')
axs[1, 0].set_ylabel('Total Sales')
axs[1, 0].tick_params(axis='x', rotation=45)



# Sales by Customer Type (Pie Chart) 
customer_type_sales = data.groupby('Customer Type')['Total'].sum().reset_index()
axs[1, 1].pie(customer_type_sales['Total'], labels=customer_type_sales['Customer Type'], autopct='%1.1f%%', colors=sns.color_palette('pastel'))
axs[1, 1].set_title('Sales Distribution by Customer Type')





# Adjust layout
plt.tight_layout()
plt.show()


# In[17]:



sales_discount = data.groupby(pd.Grouper(key='Order Date', freq='M')).agg({'Total': 'sum', 'Discount $': 'sum'}).reset_index()

# Create a bar plot for Sales and a line plot for Discount
fig, ax1 = plt.subplots(figsize=(14, 6))  

# Plot Sales as bars
ax1.bar(sales_discount['Order Date'], sales_discount['Total'], color='b', alpha=0.6, label='Sales')
ax1.set_xlabel('Date')
ax1.set_ylabel('Sales', color='b')
ax1.tick_params('y', colors='b')

# Create a second y-axis for the Discount
ax2 = ax1.twinx()
ax2.plot(sales_discount['Order Date'], sales_discount['Discount $'], color='r', marker='o', label='Discount')
ax2.set_ylabel('Discount', color='r')
ax2.tick_params('y', colors='r')

# Set x-axis limits to start from October 2012
ax1.set_xlim(pd.Timestamp('2012-10-01'), sales_discount['Order Date'].max())


ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

fig.tight_layout()
plt.title('Sales vs. Discount Comparison')
plt.show()


# In[18]:


plt.figure(figsize=(14, 7))  # Increase figure size for better clarity
top_products = data.nlargest(10, 'Total')  # Get the top 10 products by 'Total'
top_products['Total'].plot(kind='bar')
plt.title('Top 10 Selling Products', fontsize=16)
plt.xlabel('Product Name', fontsize=14)
plt.ylabel('Total', fontsize=14)

# Set clear product names under the bars
plt.xticks(ticks=range(len(top_products)), 
           labels=top_products['Product Name'], 
           rotation=45, 
           ha='right',  # Align text to the right for clarity
           fontsize=12)  # Adjust font size

plt.tight_layout()  # Automatically adjust layout for better spacing
plt.show()


# In[19]:


data.head(5)


# In[20]:



sns.scatterplot(x='Retail Price', y='Total', data=data)
plt.title('Correlation Between Retail Price and Sales')
plt.xlabel('Retail Price')
plt.ylabel('Total')
plt.show()


# In[21]:


# Account Manager Performance: Evaluate the performance of account managers based on sales.
account_manager_sales = data.groupby('Account Manager')['Total'].sum().nlargest(10)

# Create the bar chart
plt.figure(figsize=(12, 6))
plt.bar(account_manager_sales.index, account_manager_sales)
plt.title('Top 10 Account Managers by Total Sales')
plt.xlabel('Account Manager')
plt.ylabel('Total Sales')

# Rotate x-axis labels for better readability
plt.xticks(rotation=45)

# Annotate the bars with sales values
for i, (account_manager, sales) in enumerate(account_manager_sales.items()):
    plt.annotate(f'{sales}', xy=(i, sales), ha='center', va='bottom')

plt.show()


# In[22]:


+0# Group by 'Order Priority' and calculate the average shipping cost
order_priority_shipping = data.groupby('Order Priority')['Shipping Cost'].mean()

# Plotting the result
order_priority_shipping.plot(kind='bar', color='skyblue', title='Average Shipping Cost by Order Priority')
plt.ylabel('Average Shipping Cost')
plt.show()


# In[24]:


.customer_category_sales = data.pivot_table(values='Total', 
                                           index='Customer Type', 
                                           columns='Product Category', 
                                           aggfunc='sum', 
                                           fill_value=0)

# Plot a heatmap to visualize the relationship between Customer Type and Product Category
plt.figure(figsize=(10, 6))
sns.heatmap(customer_category_sales, annot=True, cmap='Blues', fmt='.2f')

# Add title and labels
plt.title('Sales by Customer Type and Product Category', fontsize=14)
plt.xlabel('Product Category', fontsize=12)
plt.ylabel('Customer Segment', fontsize=12)
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:




