######################################################
# Imports and settings
######################################################

import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter
from lifetimes.plotting import plot_period_transactions
pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)

######################################################
# Functions
######################################################
def check_df(dataframe,head=5):
    print("##Shape##")
    print(dataframe.shape)
    print("##Types##")
    print(dataframe.dtypes)
    print("##Head##")
    print(dataframe.head(head))
    print("##Tail##")
    print(dataframe.tail(head))
    print("##Missingentries##")
    print(dataframe.isnull().sum())
    print("##Quantiles##")
    print(dataframe.quantile([0,0.05,0.50,0.95,0.99,1]).T)
    print("##generalinformation##")
    print(dataframe.describe().T)

def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    # dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

def grab_col_names(dataframe,cat_th=10,car_th=20):
    """

    Verisetindekikategorik,numerikvekategorikfakatkardinaldeğişkenlerinisimleriniverir.
    Not:Kategorikdeğişkenleriniçerisinenumerikgörünümlükategorikdeğişkenlerdedahildir.

    Parameters
    ------
    dataframe:dataframe
    Değişkenisimlerialınmakistenilendataframe
    cat_th:int,optional
    numerikfakatkategorikolandeğişkenleriçinsınıfeşikdeğeri
    car_th:int,optinal
    kategorikfakatkardinaldeğişkenleriçinsınıfeşikdeğeri

    Returns
    ------
    cat_cols:list
    Kategorikdeğişkenlistesi
    num_cols:list
    Numerikdeğişkenlistesi
    cat_but_car:list
    Kategorikgörünümlükardinaldeğişkenlistesi

    Examples
    ------
    importseabornassns
    df=sns.load_dataset("iris")
    print(grab_col_names(df))


    Notes
    ------
    cat_cols+num_cols+cat_but_car=toplamdeğişkensayısı
    num_but_catcat_cols'uniçerisinde.
    Returnolan3listetoplamıtoplamdeğişkensayısınaeşittir:cat_cols+num_cols+cat_but_car=değişkensayısı

    """

    #cat_cols,cat_but_car
    cat_cols=[col for col in dataframe.columns if dataframe[col].dtypes=="O"]
    num_but_cat=[col for col in dataframe.columns if dataframe[col].nunique()<cat_th and
    dataframe[col].dtypes != "O"]
    cat_but_car=[col for col in dataframe.columns if dataframe[col].nunique()>car_th and
    dataframe[col].dtypes == "O"]
    cat_cols=cat_cols+num_but_cat
    cat_cols=[col for col in cat_cols if col not in cat_but_car]

    #num_cols
    num_cols=[col for col in dataframe.columns if dataframe[col].dtypes!="O"]
    num_cols=[col for col in num_cols if col not in num_but_cat]

    print(f"Observations:{dataframe.shape[0]}")
    print(f"Variables:{dataframe.shape[1]}")
    print(f'cat_cols:{len(cat_cols)}')
    print(f'num_cols:{len(num_cols)}')
    print(f'cat_but_car:{len(cat_but_car)}')
    print(f'num_but_cat:{len(num_but_cat)}')
    return cat_cols,num_cols,cat_but_car

######################################################
# Prepare the data
######################################################

df_ = pd.read_csv("FLO_RFM_Analizi/flo_data_20k.csv")
df = df_.copy()

check_df(df)

# Make a new variable for the total or orders for each client
df["order_num_total_ever"] = df["order_num_total_ever_online"] + df["order_num_total_ever_offline"]
df["total_Value_both"] = df["customer_value_total_ever_offline"] + df["customer_value_total_ever_online"]

# Transform date variables into datatime64 type:
df["first_order_date"] = df["first_order_date"].astype("datetime64[ns]")
df["last_order_date"] = df["last_order_date"].astype("datetime64[ns]")
df["last_order_date_online"] = df["last_order_date_online"].astype("datetime64[ns]")
df["last_order_date_offline"] = df["last_order_date_offline"].astype("datetime64[ns]")

# Take a look at the order channel:
gb = df.groupby("order_channel").agg({"order_num_total_ever": "sum",
                                      "total_Value_both": "sum",
                                      "master_id" : "count"})

gb.reset_index()

plt.hist(gb["order_num_total_ever"])
plt.hist(gb["total_Value_both"])
plt.show(block = True)

# Get our top 10 customers in terms of total value
df.sort_values("total_Value_both", ascending=False).head(10)

# Get our top 10 customers in terms of total orders
df.sort_values("order_num_total_ever", ascending=False).head(10)

cat_cols,num_cols,cat_but_car = grab_col_names(df)
for col in num_cols:
    replace_with_thresholds(df,col)
######################################################
# RFM
######################################################

# Take Today's date as 2 days after the last date of the data.
df["last_order_date_online"].max()
today_date = dt.datetime(2021, 6, 2)

# As frequency is the total orders a client has and monetary is the total spent money we can just
# rename our variables as they're the same as we want. we only have to calculate recency.
df["Recency"] = (today_date - df["last_order_date"]).dt.days

# Make an RFM dataframe
rfm = df[["Recency","order_num_total_ever","total_Value_both"]]
rfm.columns = ['Recency', 'Frequency', 'Monetary']

# Make RFM scores and save them in RF_SSCORE.
rfm["recency_score"] = pd.qcut(rfm['Recency'], 5, labels=[5, 4, 3, 2, 1])


rfm["frequency_score"] = pd.qcut(rfm['Frequency'].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])

rfm["monetary_score"] = pd.qcut(rfm['Monetary'], 5, labels=[1, 2, 3, 4, 5])

rfm["RF_SCORE"] = (rfm['recency_score'].astype(str) +
                    rfm['frequency_score'].astype(str))


# Make segments for our RF scores
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




rfm[["segment", "Recency", "Frequency", "Monetary"]].groupby("segment").mean()


df["segment"] = rfm["segment"]

######################################################
# Customer Lifetime Value
######################################################

df = df_.copy()

df["total_num_both"] = df["order_num_total_ever_online"] + df["order_num_total_ever_offline"]
df["total_Value_both"] = df["customer_value_total_ever_offline"] + df["customer_value_total_ever_online"]

# 5 convert to dates

df["first_order_date"] = df["first_order_date"].astype("datetime64[ns]")
df["last_order_date"] = df["last_order_date"].astype("datetime64[ns]")
df["last_order_date_online"] = df["last_order_date_online"].astype("datetime64[ns]")
df["last_order_date_offline"] = df["last_order_date_offline"].astype("datetime64[ns]")

cat_cols,num_cols,cat_but_car = grab_col_names(df)
for col in num_cols:
    replace_with_thresholds(df,col)


df["recency_CLTV_WEEKLY"] = (df["last_order_date"] - df["first_order_date"]).dt.days/7
df["T_weekly"] = (today_date - df["first_order_date"]).dt.days/7
df["frequency"] = df["total_num_both"]
df["monetary_cltv_avg"] = df["total_Value_both"]/df["total_num_both"]

cltv_c = df[["recency_CLTV_WEEKLY","T_weekly","frequency","monetary_cltv_avg"]]

######################################################
# BG/NBD
######################################################

bgf = BetaGeoFitter(penalizer_coef=0.001)
bgf.fit(cltv_c["frequency"],
        cltv_c["recency_CLTV_WEEKLY"],
        cltv_c["T_weekly"])

# Expected sales in the coming 3 months
df["exp_sales_3_month"] = bgf.predict(4 * 3,
                                               df['frequency'],
                                               df['recency_CLTV_WEEKLY'],
                                               df['T_weekly'])

# Expected sales in the coming 6 months
df["exp_sales_6_month"] = bgf.predict(4 * 6,
                                               df['frequency'],
                                               df['recency_CLTV_WEEKLY'],
                                               df['T_weekly'])

######################################################
# GAMMA GAMMA
######################################################

ggf = GammaGammaFitter(penalizer_coef=0.01)
ggf.fit(cltv_c["frequency"],cltv_c["monetary_cltv_avg"])


cltv = ggf.customer_lifetime_value(bgf,
                                cltv_c['frequency'],
                                cltv_c['recency_CLTV_WEEKLY'],
                                cltv_c['T_weekly'],
                                cltv_c['monetary_cltv_avg'],
                                time=6, # 6 months
                                freq="W",# T frequency info.
                                discount_rate=0.01)

df["cltv"] = cltv


df.sort_values(by="cltv", ascending=False).head(20)

df["segment"] = pd.qcut(df["cltv"],4,labels=("D","C","B","A"))

df.groupby("segment").agg(
    {"count", "mean", "sum"})