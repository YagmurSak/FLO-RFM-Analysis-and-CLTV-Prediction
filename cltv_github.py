### CLTV Prediction with BG-NBD and Gamma-Gamma ###

## Business Problem:
    #FLO Compony wants to determine a roadmap for sales and marketing activities.
# In order for the company to make a medium-long term plan, the potential value that existing
# customers will provide to the company in the future must be estimated. The dataset consists of
# information obtained from the past shopping behavior of customers who made their last purchases
# as OmniChannel (both online and offline shopping) in 2020 - 2021.


# VARIABLES
#master_id: Unique customer number
#order_channel: Which channel is used for the shopping platform (Android, ios, Desktop, Mobile, Offline)
#last_order_channel: The channel where the last shopping was made
#first_order_date: The customer's first shopping date
#last_order_date: The customer's last shopping date
#last_order_date_online: The customer's last shopping date on the online platform
#last_order_date_offline: The customer's last shopping date on the offline platform
#order_num_total_ever_online: The total number of shopping made by the customer on the online platform
#order_num_total_ever_offline: The total number of shopping made by the customer offline
#customer_value_total_ever_offline: The total amount paid by the customer for offline shopping
#customer_value_total_ever_online: The total amount paid by the customer for online shopping
#customer_value_total_ever_online: The total amount paid by the customer for online shopping
#interested_in_categories_12: The customer's last List of categories shopped in the last 12 months


import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot

from kural_Tabanlı_proje_odev import new_user

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter
from lifetimes.plotting import plot_period_transactions

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.float_format', lambda x: '%.4f' % x)
from sklearn.preprocessing import MinMaxScaler

### Data Preparation and First Insight

df_= pd.read_csv(r"C:\Users\Yagmu\OneDrive\Masaüstü\DATA SCIENCE BOOTCAMP\2-CRM Analytics\FLOCLTVPrediction\flo_data_20k.csv")
df = df_.copy()
df.describe().T
df.head()
df.isnull().sum()


# Threshold Value Determination

def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

# Outlier Suppression Function

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    # dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


# Outlier Suppression for Order Numbers and Customer Value

customer = ["order_num_total_ever_online","order_num_total_ever_offline","customer_value_total_ever_offline","customer_value_total_ever_online"]

for col in customer:
    replace_with_thresholds(df, col)

# Creating new variables for each customer's total purchase and value count

df["order_num_total_omnichannel"] = df["order_num_total_ever_online"] + df["order_num_total_ever_offline"]
df["customer_value_total_omnichannel"] = df["customer_value_total_ever_offline"] + df["customer_value_total_ever_online"]

# Better understanding data
plt.figure(figsize=(8, 6))
new_df = df.groupby("order_channel")["order_num_total_omnichannel"].mean()
new_df.plot(kind='bar', color=['blue', 'green', 'yellow', 'red'])
plt.title('Total Number of Transaction by Channels')
plt.xlabel('Order Channel')
plt.ylabel('Total number of Transaction')
plt.show()

plt.figure(figsize=(8, 6))
new_df = df.groupby("order_channel")["customer_value_total_omnichannel"].mean()
new_df.plot(kind='bar', color=['blue', 'green', 'yellow', 'red'])
plt.title('Total Customer Value by Channels')
plt.xlabel('Order Channel')
plt.ylabel('Total Value')
plt.show()


# Converting the type of variables expressing date to date

date_columns = df.columns[df.columns.str.contains("date")]
df[date_columns] = df[date_columns].apply(pd.to_datetime)

# We will take the analysis date as 2 days after the date of the last purchase in the data set.

print(df["last_order_date"].max())

today_date = dt.datetime(2021, 6, 1)

# Creating a new cltv dataframe containing customer_id, recency_cltv_weekly, T_weekly, frequency and monetary_cltv_avg values.
# # Monetary value will be expressed as average value per purchase, recency and tenure values will be expressed in weekly terms.

cltv_df = pd.DataFrame()
cltv_df["customer_id"] = df["master_id"]
cltv_df["recency_cltv_weekly"] = ( (df["last_order_date"]- df["first_order_date"]).dt.days) / 7
cltv_df["T_weekly"] = ((today_date - df["first_order_date"]).dt.days) / 7
cltv_df["frequency"] = df["order_num_total_omnichannel"]
cltv_df["monetary_cltv_avg"] = df["customer_value_total_omnichannel"] / df["order_num_total_omnichannel"]

plt.figure(figsize=(8, 6))
plt.hist(cltv_df['frequency'], bins=30, edgecolor='black')
plt.title('Purchase Frequency Distribution')
plt.xlabel('frequency')
plt.ylabel('Number of Customers')
plt.show()


plt.figure(figsize=(8, 6))
plt.hist(cltv_df['frequency'], bins=30, edgecolor='black')
plt.title('Monetary Distribution')
plt.xlabel('monetary_cltv_avg')
plt.ylabel('Number of Customers')
plt.show()


# Fitting the BG-NBD Model

bgf = BetaGeoFitter(penalizer_coef=0.001)

bgf.fit(cltv_df['frequency'],
        cltv_df['recency_cltv_weekly'],
        cltv_df['T_weekly'])


# Estimated purchases from customers within 3 months

cltv_df["exp_sales_3_month"] = bgf.predict(3*4,
                                              cltv_df['frequency'],
                                              cltv_df['recency_cltv_weekly'],
                                              cltv_df['T_weekly'])

# Estimated purchases from customers within 6 months

cltv_df["exp_sales_6_month"] = bgf.predict(6*4,
                                              cltv_df['frequency'],
                                              cltv_df['recency_cltv_weekly'],
                                              cltv_df['T_weekly'])

# Fitting the Gamma-Gamma Model

ggf = GammaGammaFitter(penalizer_coef=0.01)

cltv_df["frequency"] = cltv_df["frequency"].astype(int)

ggf.fit(cltv_df["frequency"], cltv_df["monetary_cltv_avg"])


# CLTV Prediction for 6 months

cltv_df["cltv"] = ggf.customer_lifetime_value(bgf,
                                   cltv_df['frequency'],
                                   cltv_df['recency_cltv_weekly'],
                                   cltv_df['T_weekly'],
                                   cltv_df['monetary_cltv_avg'],
                                   time=6,  # 6 months
                                   freq="W",  # Frequency of T
                                   discount_rate=0.01)

# Top 20 people with the highest cltv value

cltv_df.sort_values("cltv", ascending=False).head(20)["customer_id"]



# Creating Segments Based on CLTV

cltv_df["cltv_segment"] = pd.qcut(cltv_df["cltv"], 4, labels=["D", "C", "B", "A"])


# Number of Customers by CLTV Segment

cltv_segment_counts = cltv_df["cltv_segment"].value_counts()

plt.figure(figsize=(8, 6))
cltv_segment_counts.plot(kind='bar', color=['skyblue', 'orange', 'green', 'red'])
plt.title('Number of Customers by CLTV Segment')
plt.xlabel('CLTV Segment')
plt.ylabel('Number of Customers')
plt.show()

#Expected Purchases: 3 Month vs 6 Month


plt.figure(figsize=(8, 6))
plt.scatter(cltv_df['exp_sales_3_month'], cltv_df['exp_sales_6_month'], alpha=0.5, color='purple')
plt.title('Expected Purchases: 3 Month vs 6 Month')
plt.xlabel('Expected Purchases in 3 Months')
plt.ylabel('Expected Purchases in 6 Months')
plt.show()


# Average Monetary Value by CLTV Segment

plt.figure(figsize=(8, 6))
cltv_segment_monetary = cltv_df.groupby("cltv_segment")["monetary_cltv_avg"].mean()
cltv_segment_monetary.plot(kind='bar', color=['blue', 'green', 'yellow', 'red'])
plt.title('Average Monetary Value by CLTV Segment')
plt.xlabel('CLTV Segment')
plt.ylabel('Average Monetary Value')
plt.show()





