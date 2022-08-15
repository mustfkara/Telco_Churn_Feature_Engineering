
#############################################
# Feature Engineering of The Telcho_Churn
#############################################

##################
# Business Problem
##################

# Developing a machine learning model that can predict customers leaving the company is requested.
# In this project, necessary data analysis and feature engineering steps will be performed
# before model development.

####################
# About the Data Set
####################

# Telco churn data provided 7043 California customers with home phone and contains information about
# a fictitious telecom company that provides Internet services.
# It shows which customers have left, stayed or signed up for their service.

# CustomerId:
# Gender:
# SeniorCitizen: Whether the customer is old (1, 0)
# Partner: Whether the customer has a partner (Yes, No)
# Dependents: Whether the customer has dependents (Yes, No)
# tenure: Number of months the customer has stayed with the company
# PhoneService: Whether the customer has telephone service (Yes, No)
# MultipleLines: Whether the customer has more than one line (Yes, No, No phone service)
# InternetService: Customer's internet service provider (DSL, Fiber optic, No)
# OnlineSecurity: Whether the customer has online security (Yes, No, No Internet service)
# OnlineBackup: Whether the customer has an online backup (Yes, No, No Internet service)
# DeviceProtection: Whether the customer has device protection (Yes, No, No Internet service)
# TechSupport: Whether the customer received technical support (Yes, No, No Internet service)
# StreamingTV: Whether the customer has a TV broadcast (Yes, No, No Internet service)
# StreamingMovies: Whether the client is streaming movies (Yes, No, No Internet service)
# Contract: Customer's contract term (Month to month, One year, Two years)
# PaperlessBilling: Whether the customer has a paperless bill (Yes, No)
# PaymentMethod: Customer's payment method (Electronic check, Postal check,
#                                           Bank transfer (automatic), Credit card (automatic))
# MonthlyCharges: Amount charged from the customer on a monthly basis
# TotalCharges: Total amount charged from customer
# Churn:  (Yes, No)

############################
# Exploratory Data Analysis
############################

# 1- Necessary Libraries:

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
# !pip install missingno
import missingno as msno
from datetime import date
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler
from statsmodels.stats.proportion import proportions_ztest
from sklearn.ensemble import RandomForestClassifier

# 2- Customize DataFrame to display:

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)

# 3- Loading the Dataset:

def load():
    data = pd.read_csv("Projects/Telco_Churn_Feature_Enginnering/Telco-Customer-Churn.csv")
    return data


df = load()

# 4- Dataset Overview:

def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.describe([0, 0.05, 0.50, 0.95, 0.99, 1]).T)


check_df(df)

# The data set consists of 3 numerical, 18 object variables and 7043 observation units.
# While TotalCharges should be a numeric variable, it is an object type variable.
# We must make this transformation.

df.TotalCharges = pd.to_numeric(df.TotalCharges, errors='coerce')

# The dependent variable:

df["Churn"] = df["Churn"].apply(lambda x : 1 if x == "Yes" else 0)

df["Churn"].value_counts()

"""
0    5163
1    1869
"""
df.info()

# There are no missing observations in the data set.

# 5- Numeric and Categorical variables:

# Although the data set consists of 4 numerical variables, some variables may actually be categorical.
# There may actually be cardinal variables, as there are categorical variables and the number of unique
# classes is high.

TARGET = "Churn"
def grab_col_names(dataframe, target, cat_th=10, car_th=20):
    """

    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

    Parameters
    ------
        dataframe: dataframe
                Değişken isimleri alınmak istenilen dataframe
        cat_th: int, optional
                numerik fakat kategorik olan değişkenler için sınıf eşik değeri
        car_th: int, optinal
                kategorik fakat kardinal değişkenler için sınıf eşik değeri
        target: str
                Bağımlı(hedef) değişken

    Returns
    ------
        cat_cols: list
                Kategorik değişken listesi
        num_cols: list
                Numerik değişken listesi
        cat_but_car: list
                Kategorik görünümlü kardinal değişken listesi

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde.
        Return olan 3 liste toplamı toplam değişken sayısına eşittir: cat_cols + num_cols + cat_but_car = değişken sayısı

    """

    # cat_cols, cat_but_car
    cat_cols_mask = [col for col in dataframe.columns if dataframe[col].dtypes == "O"
                     and dataframe[col].nunique() < car_th and col != target]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].dtypes != "O"
                   and dataframe[col].nunique() < cat_th and col != target]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].dtypes == "O" and
                   col not in cat_cols_mask and col != target]

    cat_cols = cat_cols_mask + num_but_cat

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"
                and col not in num_but_cat and col != target]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car, num_but_cat

cat_cols, num_cols, cat_but_car ,num_but_cat = grab_col_names(df, TARGET, cat_th=20)

#Observations: 7043
#Variables: 21
#cat_cols: 16
#num_cols: 3
#cat_but_car: 1
#num_but_cat: 1

# num_cols
# ['tenure', 'MonthlyCharges', 'TotalCharges']

# cat_but_car
# ['customerID']

# cat_cols
# ['gender','Partner','Dependents','PhoneService','MultipleLines','InternetService','OnlineSecurity''OnlineBackup',
#  'DeviceProtection','TechSupport','StreamingTV','StreamingMovies','Contract','PaperlessBilling','PaymentMethod',
#  'Churn','SeniorCitizen']

# Analysis of numerical and categorical variables:

def num_variable_overview(dataframe, num_cols, plot=False):
    print(dataframe[num_cols].describe().T)
    for i in num_cols:
        print("---------------------------", i, "-----------------------------------")
        print("Missing Values : ", dataframe[num_cols].isnull().sum().sum())
        print("------------------------------------------------------------------------")

        if plot:
            print("--------------------------------Graph------------------------------")
            plt.hist(dataframe[i])
            plt.title("Distribution of Variable")
            plt.xlabel(i)
            plt.show(block=True)


num_variable_overview(df, num_cols, plot=True)

def cat_variable_overview(dataframe, cat_cols, plot=False):
    print(dataframe[cat_cols].describe().T)
    for i in cat_cols:
        print("---------------------------", i, "-----------------------------------")
        print("Missing Values : ", dataframe[cat_cols].isnull().sum().sum())
        print("------------------------------------------------------------------------")

        if plot:
            print("--------------------------------Graph------------------------------")
            plt.hist(dataframe[i])
            plt.title("Distribution of Variable")
            plt.xlabel(i)
            plt.show(block=True)

cat_variable_overview(df, cat_cols, plot=True)

# Average of numerical variables relative to the dependent variable:

def target_summary_with_cat(dataframe,target,categorical_col):
    print(pd.DataFrame({"CHURN_MEAN": dataframe.groupby(categorical_col)[target].mean()}))
    print("###################################")


for col in cat_cols:
    target_summary_with_cat(df,"Churn",col)


def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: "mean"}), end="\n\n\n")


for col in num_cols:
    target_summary_with_num(df, "Churn", col)

# 6- Outliers Analysis:

# Define an outlier thresholds for variables:

def outlier_thresholds(dataframe, col_name, q1=0.05, q3=0.95):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


# Check for outliers for variables:

def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False


for col in num_cols:
    print(col, check_outlier(df, col))

# tenure False
# MonthlyCharges False
# TotalCharges False

# 7- The Missing Values Analysis:

# Missing value and ratio analysis for variables:

def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]

    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")

    if na_name:
        return na_columns


na_cols = missing_values_table(df, True)

df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)

# 8- Correlation Analysis:

def correlated_cols(dataframe, plot=False):
    corr_matrix = dataframe.corr()

    if plot:
        import seaborn as sns
        import matplotlib.pyplot as plt
        sns.set(rc={'figure.figsize': (15, 15)})
        sns.heatmap(corr_matrix, cmap="RdBu")
        plt.show(block=True)
    return print(corr_matrix)


correlated_cols(df, plot=True)

"""
                SeniorCitizen  tenure  MonthlyCharges  TotalCharges  Churn
SeniorCitizen           1.000   0.017           0.220         0.102  0.151
tenure                  0.017   1.000           0.248         0.826 -0.352
MonthlyCharges          0.220   0.248           1.000         0.651  0.193
TotalCharges            0.102   0.826           0.651         1.000 -0.199
Churn                   0.151  -0.352           0.193        -0.199  1.000
"""
# There is a moderate positive high level of relationship between tenure and TotalCharges.(0.826)
# There is a moderate positive correlation between MonthlyCharges and TotalCharges. (0.651)

############################
# Future Engineering
############################
df_ = df.copy()
# 1- Necessary actions for missing and outliers values:


msno.bar(df_)
plt.show(block=True)


# 2- Creating New Features:

df_.loc[(df_['PaymentMethod'] == 'Bank transfer (automatic)'), "PAYMENT_METHOD"] = "AUTOMATIC"
df_.loc[(df_['PaymentMethod'] == 'Credit card (automatic)'), "PAYMENT_METHOD"] = "AUTOMATIC"
df_.loc[(df_['PaymentMethod'] == 'Electronic check'), "PAYMENT_METHOD"] = "CHECK"
df_.loc[(df_['PaymentMethod'] == 'Mailed check'), "PAYMENT_METHOD"] = "CHECK"

df_["MONTLY_CHARGE_INFO"] = pd.qcut(df_["MonthlyCharges"], 3, labels=["Low", "Medium", "High"])

df_["TENURE_YEAR"] = df_["tenure"] / 12
df_.head()

# 3- label Encoding and One-Hot Encoding:

def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe

binary_cols = [col for col in df_.columns if df_[col].dtype not in [int, float]
               and df_[col].nunique() == 2]

for col in binary_cols:
    df_ = label_encoder(df_, col)


def one_hot_encoder(dataframe, categorical_cols, drop_first=True):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe


ohe_cols = [col for col in df_.columns if 10 >= df_[col].nunique() > 2]

df_ = one_hot_encoder(df_, ohe_cols)

df_.head()

# 4- Standardization for numerical variables:

ss = StandardScaler()
df_[num_cols] = ss.fit_transform(df_[num_cols])

df_[num_cols].head()

df_.head()


############
# Modelling
############

y = df_[TARGET]
X = df_.drop([TARGET, "customerID"], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=17)

rf_model = RandomForestClassifier(random_state=46).fit(X_train, y_train)
y_pred = rf_model.predict(X_test)
accuracy_score(y_pred, y_test)

# 0.7851396119261713