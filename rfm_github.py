import pandas as pd
import datetime as dt
from numba.core.runtime import rtsys

pd.set_option('display.max_columns', None)


#### Customer Segmentation with RFM #####

#### Business Problem:

# FLO wants to segment its customers and determine marketing strategies according to these segments.
# For this purpose, customer behaviors will be defined and groups will be created according to these behavior clusters.

# The dataset consists of information obtained from the past shopping behaviors of customers who made their last
# purchases as OmniChannel (both online and offline shopping) in 2020 - 2021.

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


### Data Preparation and First Insight

df_= pd.read_csv(r"C:\Users\Yagmu\PycharmProjects\pythonProject2\flo_data_20k.csv")
df = df_.copy()

df.head(10)
df.columns
df.describe().T
df.isnull().sum()
df.dtypes
df.info()
df.shape

# Creating new variables for each customer's total purchase and value count

df["order_num_total_omnichannel"] = df["order_num_total_ever_online"] + df["order_num_total_ever_offline"]
df["customer_value_total_omnichannel"] = df["customer_value_total_ever_offline"] + df["customer_value_total_ever_online"]

# Converting the type of variables expressing date to date

variable_contains_date = df.columns[df.columns.str.contains("date")]
df[variable_contains_date] = df[variable_contains_date].apply(pd.to_datetime)

# Looking at the distribution of the number of customers across shopping channels, the average number of items purchased, and the average spend.

df.groupby("order_channel").agg({"master_id":"count",
                                 "order_num_total_omnichannel" : "mean",
                                 "customer_value_total_omnichannel" : "mean"})


# Fuction for data preparation

def data_preparatory(dataframe):
    df["order_num_total_omnichannel"] = df["order_num_total_ever_online"] + df["order_num_total_ever_offline"]
    df["customer_value_total_omnichannel"] = df["customer_value_total_ever_offline"] + df["customer_value_total_ever_online"]

    df["first_order_date"] = pd.to_datetime(df["first_order_date"])
    df["last_order_date"] = pd.to_datetime(df["last_order_date"])
    df["last_order_date_online"] = pd.to_datetime(df["last_order_date_online"])
    df["last_order_date_offline"] = pd.to_datetime(df["last_order_date_offline"])

    return df


# We will take the analysis date as 2 days after the date of the last purchase in the data set.

df["last_order_date"].max()

today_date = dt.datetime(2021, 6, 1)
type(today_date)

# Defining new dataframe

rfm = df.groupby("master_id").agg({ "master_id" : lambda master_id : master_id,
                                    "last_order_date": lambda last_order_date: (today_date - last_order_date.max()).days,
                                    "order_num_total_omnichannel" : lambda order_num_total_omnichannel: order_num_total_omnichannel,
                                    "customer_value_total_omnichannel" : lambda  customer_value_total_omnichannel: customer_value_total_omnichannel  })

rfm.columns = ["customer_id", "recency", "frequency", "monetary"]


# Calculating RF and RFM Scores

rfm["recency_score"] = pd.qcut(rfm['recency'], 5, labels=[5, 4, 3, 2, 1])

rfm["frequency_score"] = pd.qcut(rfm['frequency'].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])

rfm["monetary_score"] = pd.qcut(rfm['monetary'], 5, labels=[1, 2, 3, 4, 5])

rfm["RF_SCORE"] = (rfm['recency_score'].astype(str) + rfm['frequency_score'].astype(str))

# Segmentation

seg_map = {
    r'[1-2][1-2]': 'hibernating',
    r'[1-2][3-4]': 'at_Risk',
    r'[1-2]5': 'cant_loose',
    r'3[1-2]': 'about_to_sleep',
    r'33': 'need_attention',
    r'[3-4][4-5]': 'loyal_customers',
    r'41': 'promising',
    r'51': 'new_customers',
    r'[4-5][2-3]': 'potential_loyalists',
    r'5[4-5]': 'champions'
}

rfm['segment'] = rfm['RF_SCORE'].replace(seg_map, regex=True)

# Application


# FLO is adding a new women's shoe brand to its portfolio. The product prices of the brand it is adding are above general customer preferences.
# For this reason, it is desired to contact customers with the profile that will be interested in the brand's promotion and product sales.
# It is planned that these customers will be loyal and shoppers from the women's category.

customer_segment_target = rfm[ rfm["segment"].isin(["loyal_customers"])]["customer_id"]

customer_target = df[df["master_id"].isin(customer_segment_target) & df["interested_in_categories_12"].str.contains("KADIN")]["master_id"]


# Saving the IDs of the customers with the appropriate profile to the csv file as discount_target_customer_ids.csv


customer_target.to_csv("discount_target_customer_ids.cvs", index=False)