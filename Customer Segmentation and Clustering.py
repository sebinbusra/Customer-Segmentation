
# pip install yellowbrick

import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer
from sklearn.preprocessing import StandardScaler

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)


############################################
# EDA ANALIZI
############################################
df = pd.read_csv("datasets/Mall_Customers.csv", index_col=0)
df.describe()

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
    print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)

check_df(df)



# SAYISAL DEĞİŞKEN ANALİZİ

num_cols = ["Age", "Annual Income (k$)", "Spending Score (1-100)"]
def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T, end="\n\n")

    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)

for col in num_cols:
    num_summary(df, col, plot=True)

# AYKIRI GÖZLEM ANALİZİ

def outlier_thresholds(dataframe, col_name, q1=0.25, q3=0.75):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def check_outlier(dataframe, col_name, q1=.25, q3=.75):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name, q1, q3)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

for col in num_cols:
    print(col, check_outlier(df, col, q1=0.1, q3=0.9))

# There is no outlier for these quartile interval.

# EKSİK GÖZLEM ANALİZİ

def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]
    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")
    if na_name:
        return na_columns

missing_values_table(df)
# No missing value.


# KORELASYON ANALİZİ

import numpy as np

def high_correlated_cols(dataframe, plot=False, corr_th=0.90):
    corr = dataframe.corr()
    cor_matrix = corr.abs()
    upper_triangle_matrix = cor_matrix.where(np.triu(np.ones(cor_matrix.shape), k=1).astype(np.bool))
    drop_list = [col for col in upper_triangle_matrix.columns if any(upper_triangle_matrix[col] > corr_th)]
    if plot:
        import seaborn as sns
        import matplotlib.pyplot as plt
        sns.set(rc={'figure.figsize': (15, 15)})
        sns.heatmap(corr, cmap="RdBu")
        plt.show()
    return drop_list

high_correlated_cols(df, plot=True, corr_th=0.90)

################################
# Optimum Küme Sayısının Belirlenmesi
################################

# Once income icin

df.columns
df.head()

scaler = StandardScaler()
scaled_features = scaler.fit_transform(df)
scaled_features[:5]


df.head()
# new_df = df.drop("Gender", axis=1, inplace=True)
kmeans = KMeans()
ssd1 = []
K = range(1, 30)

for k in K:
    kmeans = KMeans(n_clusters=k).fit(scaled_features[:,1].reshape(-1, 1))
    ssd1.append(kmeans.inertia_)

plt.plot(K, ssd1, "bx-")
plt.xlabel("Farklı K Değerlerine Karşılık SSE/SSR/SSD")
plt.title("Optimum Küme sayısı için Elbow Yöntemi")
plt.show()

# cluster sayimiz 6

################################
# Final Cluster'ların Oluşturulması
################################

kmeans = KMeans(n_clusters=6).fit(scaled_features[:,1].reshape(-1, 1))

kmeans.n_clusters
kmeans.cluster_centers_
kmeans.labels_

clusters_kmeans = kmeans.labels_
df["Income Cluster"] = clusters_kmeans

# Segmentleri analiz edelim

df.groupby("Income Cluster").agg(["count","mean"])

#  Income ve Spending Score icin
kmeans = KMeans()
ssd2 = []
K = range(1, 30)

for k in K:
    kmeans = KMeans(n_clusters=k).fit(scaled_features[:,1:3])
    ssd2.append(kmeans.inertia_)

plt.plot(K, ssd2, "bx-")
plt.xlabel("Farklı K Değerlerine Karşılık SSE/SSR/SSD")
plt.title("Optimum Küme sayısı için Elbow Yöntemi")
plt.show()

# cluster sayimiz 5

kmeans = KMeans(n_clusters=5).fit(scaled_features[:,1:3])

kmeans.n_clusters
kmeans.cluster_centers_
kmeans.labels_

clusters_kmeans = kmeans.labels_
df["Spending and Income Cluster"] = clusters_kmeans
df.head()

sns.scatterplot(data=df, x = "Annual Income (k$)", y = "Spending Score (1-100)", hue= "Spending and Income Cluster" , palette= "tab10", )

# Segmentleri analiz edelim

df.groupby("Income Cluster").agg(["count","mean"])

pd.crosstab(df["Spending and Income Cluster"], df["Gender"])

# Hedef grup cluster 2 olabilir, yuksek gelir ve harcama skoruna sahip
# Cluster 2'nin yuzde 54'u kadin. Musterileri favori urunlerini market kampanyalarina dahil ederek hedeflemeliyiz.
# Cluster 4'un geliri dusuk olmasina ragmen harcama skoru yuksek, ozel stratejiler uygulanmali.


# butun sayisal verilerle cluster
pd.get_dummies(df, drop_first=True)
df.head()

scaler = StandardScaler()
scaled_features = scaler.fit_transform(df)


kmeans = KMeans()
ssd3 = []
K = range(1, 30)

for k in K:
    kmeans = KMeans(n_clusters=k).fit(scaled_features)
    ssd3.append(kmeans.inertia_)

plt.plot(K, ssd3, "bx-")
plt.xlabel("Farklı K Değerlerine Karşılık SSE/SSR/SSD")
plt.title("Optimum Küme sayısı için Elbow Yöntemi")
plt.show()

# cluster sayimiz 4

kmeans = KMeans(n_clusters=4).fit(scaled_features)

kmeans.n_clusters
kmeans.cluster_centers_
kmeans.labels_

clusters_kmeans = kmeans.labels_
df["Final Cluster"] = clusters_kmeans
df.head()

df.to_csv("Customer Segmentation.csv")
