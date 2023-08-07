# -*- coding: utf-8 -*-

#Business understanding
#Business problem: Unforeseen demands due to the inappropriate machine learning algorithm
#Business objective: Maximize accurate prediction of demand and supply
#Business constraint:Miniminze risk/ Lack of historical data 
    
#Sucess criteria: Economic success criteria, ML success criteria, Business success criteria
#Data understanding
#import numpy, pandas for data handling
import numpy as np
import pandas as pd
#import walmart data from local system
walmart = pd.read_csv(r"C:/Users/ThinkPad/Downloads/360digimg/EDA/walmart_problem/Walmart.csv")
#info for different data types
walmart.info()
#There are total 9 columns and 6449 entries
#Out of 9 columns, 6 columns have numerical data, one has date data and two columns have categorical data
Data_types = pd.DataFrame({"Name of feature": ["Store", "Date", "Weekly_Sales", "Holiday_Flag", "Temperature", "Fuel_Price", "CPI", "Unemployment", "Company"],
                           "Description": ["Store number", "the week of sales", "sales for the given store", "special holiday week", "Temperature on the day", "Cost of fuel", "consumer price index", "unemployment rate", "Company name"],
                           "Data type": ["Discrete", "Discrete", "Continuous", "discrete", "Continuouos", "Continuous", "Continuous", "Ratio", "Nominal"],
                           "Relevance": ["Relevant", "Relevant", "Relevant", "Relevant", "Relevant", "Relevant", "Relevant", "Relevant", "Irrelevant"]
                           })

#we can drop company column
walmart1 = walmart.drop(["Company"], axis =1)


#Statistics
#First momentBusiness decision - mean, median, mode
#Second Business decion - std, var, range
#Store column is irrelevat here, can be dropped as well
walmart2 = walmart1.drop(["Store"], axis =1)
stat_W = walmart1.describe()
walmart2.Temperature.mode() #output 50.43
walmart2.Fuel_Price.mode() #output 3.638
walmart2.Unemployment.mode() #output 8.099

#In temprature, fuel price and CPI column, there seems to be outliers on the upper side as max value is too high

#Third moment business decision, skewness
walmart2.skew()
#Temprature, Fuel price, CPI, Unemployment has skewness greater than 1, positive skewness mean all have data shifted to right side
#weekly sales has skewness 0.666615, moderately skewed

#Fourth nmoment business decision, Kurtosis
walmart2.kurt()

#Temperature, Fuel_price and CPI has Kurtosis a lot greater than 3, It means wide tails and thin peak.
#Further, such higher values of Kurtosis suggest high probability of outlier in these columns.
#Weekly_sales and Unemployment has Kutosis between 0 and 3 which mean wide peaks and thin tails


#Graphical represenatation
#first import matplotlib and seaborn libraries
import matplotlib.pyplot as plt
import seaborn as sns

#Histogram
plt.hist(walmart2.Fuel_Price)
plt.hist(walmart2.CPI)
plt.hist(walmart2.Temperature)
#All three histogram contain a sinlge bar on left side of plot which again confirm the prsenece of outliers on higher side
#These outliers needed to be removed first to get more insights from plots.

plt.hist(walmart2.Unemployment)
#more data on left side than right side which means that Unemployment rate is less than median for most of cases.

plt.hist(walmart2.Weekly_Sales)
#data is clearly right skewed. This plot also shows that most of the sales have value 4,00,000.
#There are very few days, most probably during the holidays when we have sales value >30,00,000.

#Boxplot to check outliers
plt.boxplot(walmart2.Weekly_Sales)
#There are more than 20 outliers on the higher side of data. However, these outliers may refer to holidays.
#For other, we have blank boxplot because of empty cell or null values in data.

#Data Preprocessing
#1)Type casting
#Let's start with original file here. First check the data types
walmart.dtypes
#we can change store type to string

walmart.Store = walmart["Store"].astype("str")

#2)Duplicates handling
duplicate = walmart.duplicated() #return true for duplicates
sum(duplicate) #14 duplcates
walmartd = walmart.drop_duplicates() #remove all the duplicates

#3)Correlation coefficient
#Find correlation between different columns with heatmap
sns.heatmap(data = walmartd.corr(), annot = True)
#For all pairs, r<0.4, weak or no correlation 
#sns.pairplot(walmartd)

#4) check if variance is zero
walmartd.var()
#no zero variance but last column can be dropped as it is company name Wamlart
walmartd_1 = walmartd.drop(["Company"], axis =1)

#5) dummy variables
walmartd_2 = pd.get_dummies(walmartd_1, columns = ["Holiday_Flag"], drop_first = True)
#again check correlation to see if there is any correlation between weekly sales and holidays
sns.heatmap(data = walmartd_2.corr(), annot = True)
#no correlation found

#Using labelencoder for date columns
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder() #initalize
walmartd_2["Date1"] = labelencoder.fit_transform(walmartd_2['Date'])
#check for correlation
sns.heatmap(data = walmartd_2.corr(), annot = True) #no correlation
walmartd_3 = walmartd_2.drop(["Date1"], axis =1) #removed added dummy variable column

#6) outlier handling
walmartd_3.info()
walmartd_3.Holiday_Flag_Yes = walmartd_3["Holiday_Flag_Yes"].astype("str") #change type of holiday flag
#Find the outliers


#Temprature outliers
IQR_T = walmartd_3["Temperature"].quantile(0.75)-walmartd_3["Temperature"].quantile(0.25)
lower_T = walmartd_3["Temperature"].quantile(0.25) - IQR_T*1.5
upper_T = walmartd_3["Temperature"].quantile(0.75) + IQR_T*1.5


#flag outlier in dataset
outliers_T = np.where(walmartd_3["Temperature"] > upper_T, True, np.where(walmartd_3["Temperature"] < lower_T, True, False))
outliers_T.sum() #5 outliers

#replace outliers by upper and lower limits
walmartd_3["Temperature"]= np.where(walmartd_3["Temperature"] > upper_T, upper_T, np.where(walmartd_3["Temperature"] < lower_T, lower_T, walmartd_3["Temperature"]))
sns.boxplot(walmartd_3.Temperature)
###remove by winsorization but can't handle null values

#CPI outliers
IQR_C = walmartd_3["CPI"].quantile(0.75)-walmartd_3["CPI"].quantile(0.25)
lower_C = walmartd_3["CPI"].quantile(0.25) - IQR_C*1.5
upper_C = walmartd_3["CPI"].quantile(0.75) + IQR_C*1.5

#flag outlier in dataset
outliers_C = np.where(walmartd_3["CPI"] > upper_C, True, np.where(walmartd_3["CPI"] < lower_C, True, False))
outliers_C.sum() #2 outliers

walmartd_3["CPI"]= np.where(walmartd_3["CPI"] > upper_C, upper_C, np.where(walmartd_3["CPI"] < lower_C, lower_C, walmartd_3["CPI"]))
sns.boxplot(walmartd_3.CPI)
#Fuel price outliers
IQR_F = walmartd_3["Fuel_Price"].quantile(0.75)-walmartd_3["Fuel_Price"].quantile(0.25)
lower_F = walmartd_3["Fuel_Price"].quantile(0.25) - IQR_F*1.5
upper_F = walmartd_3["Fuel_Price"].quantile(0.75) + IQR_F*1.5

#flag outlier in dataset
outliers_F = np.where(walmartd_3["Fuel_Price"] > upper_F, True, np.where(walmartd_3["Fuel_Price"] < lower_F, True, False))
outliers_F.sum() #2 outliers

walmartd_3["Fuel_Price"]= np.where(walmartd_3["Fuel_Price"] > upper_F, upper_F, np.where(walmartd_3["Fuel_Price"] < lower_F, lower_F, walmartd_3["Fuel_Price"]))
sns.boxplot(walmartd_3.Fuel_Price)

#weekly sales outliers
IQR_W = walmartd_3["Weekly_Sales"].quantile(0.75)-walmartd_3["Weekly_Sales"].quantile(0.25)
lower_W = walmartd_3["Weekly_Sales"].quantile(0.25) - IQR_W*1.5
upper_W = walmartd_3["Weekly_Sales"].quantile(0.75) + IQR_W*1.5

#flag outlier in dataset
outliers_W = np.where((walmartd_3["Weekly_Sales"] > upper_W) & (walmartd_3["Holiday_Flag_Yes"] == "0"), True, np.where(walmartd_3["Weekly_Sales"] < lower_W, True, False))
outliers_W.sum() 
#34 outliers if we don't take into accoutn holidays
#remained 25 after removing outliers coresponding to holidays

walmartd_3["Weekly_Sales"]= np.where((walmartd_3["Weekly_Sales"] > upper_W) & 
                                     (walmartd_3["Holiday_Flag_Yes"] == "0"), upper_W, 
                                     np.where(walmartd_3["Weekly_Sales"] < lower_W, lower_W, walmartd_3["Weekly_Sales"]))

#we can see that most of the outliers for weekly_sales are on thanksgiving holidays, 26-11-2010 and 25-11-2011.
#we can't ignore these values as these give important information.

IQR_U = walmartd_3["Unemployment"].quantile(0.75)-walmartd_3["Unemployment"].quantile(0.25)
lower_U = walmartd_3["Unemployment"].quantile(0.25) - IQR_U*1.5
upper_U = walmartd_3["Unemployment"].quantile(0.75) + IQR_U*1.5
#In unemployment, we have 481 outliers, almost 7.45%, don't know waht to do with these
outliers_U = np.where(walmartd_3["Unemployment"] > upper_U, True, np.where(walmartd_3["Unemployment"] < lower_U, True, False))
outliers_U.sum()


#assign to new variable
walmart_new = walmartd_3

#7) Missing values handling
walmart_new.isna().sum() #gives number of missing values in each column
#Temprature: 3, Fuel_price: 2, CPI:4, Unemployment:2
walmart_new.isna().sum().sum() #Total missing values:11

from sklearn.impute import SimpleImputer
#replace by mean value
imp_m = SimpleImputer(missing_values= np.nan, strategy="mean")
walmart_new[["Temperature", "Fuel_Price", "CPI", "Unemployment"]] = imp_m.fit_transform(walmart_new[["Temperature", "Fuel_Price", "CPI", "Unemployment"]])

#using pd.dataFrame in the fit.transform was giving 14 null values at the end.
sns.boxplot(walmart_new.Temperature)
sns.boxplot(walmart_new.CPI)
#outliers are appearing in these two which are needed to be reomved
 
#Use Winsorizer with capping method IQR to remove outlier values in Tempaerature and Fuel_price
from feature_engine.outliers import Winsorizer
winsor_iqr = Winsorizer(capping_method = "iqr", tail ="both", fold = 1.5)
walmart_new[["Temperature", "CPI"]] = winsor_iqr.fit_transform(walmart_new[["Temperature", "CPI"]])

#again check boxplot for outliers
plt.boxplot(walmart_new.Temperature)
plt.boxplot(walmart_new.CPI)

#8) Transformation
import scipy.stats as stats
import pylab
#QQ plot to check if normal distribution or not
stats.probplot(walmart_new.Unemployment, dist = "norm", plot = pylab) 
#a kink in the upper values
stats.probplot(walmart_new.CPI, dist = "norm", plot = pylab) #no
stats.probplot(walmart_new.Temperature, dist = "norm", plot = pylab) #yes except few ponits
stats.probplot(walmart_new.Weekly_Sales, dist = "norm", plot = pylab) 
#a small kink, near to normal distribution
stats.probplot(walmart_new.Fuel_Price, dist = "norm", plot = pylab) #yes excpet lower and upper values


#normalize using boxcox 
#Temprature: normal distibution, boxcox is not making much difference

#Unemployment
f_dataU, f_lambdaU = stats.boxcox(walmart_new.Unemployment)
sns.kdeplot(f_dataU, label = "Normal Unemployment", color = "blue", shade = True)
stats.probplot(f_dataU, dist = "norm", plot = pylab) #QQ Plot
#log method is used here in boxcox

#CPI
f_dataC, f_lambdaC = stats.boxcox(walmart_new.CPI)
sns.kdeplot(f_dataC, label = "Normal CPI", color = "red", shade = True)
#made two distributions, implies two clusters in data, data is bimodal
#It is depending upon the store, some store have high CPI, means prices of goods are rising more rapidly than others
sns.kdeplot(walmart_new.CPI, label = "Original CPI", color = "red", shade = True)
#improved only second distribution a little  

#Weekly Sales
f_dataW, f_lambdaW = stats.boxcox(walmart_new.Weekly_Sales)
sns.kdeplot(f_dataW, label = "Normal Weekly Sales", color = "yellow", shade = True)
sns.kdeplot(walmart_new.Weekly_Sales, label = "Original Weekly Sales", color = "yellow", shade = True)
#multimodal distribution 
stats.probplot(f_dataW, dist = "norm", plot = pylab) 

#Fuel Price, no much change, no need to do
sns.kdeplot(walmart_new.Fuel_Price, label = "Original Fuel Price", color = "brown", shade = True)
#Bimodal distribution, fuel prices are around 2.6-2.7 for first year and then raised high in second year
plt.boxplot(walmart_new.Fuel_Price)
#replace with fitted data
walmart_new["Unemployment"] = f_dataU
walmart_new["CPI"] = f_dataC
walmart_new["Weekly_Sales"] = f_dataW

#9) Feature scaling
#scale around mean value
#from sklearn.preprocessing import StandardScaler
#scalar = StandardScaler() #intialize
#walmart_new[["Weekly_Sales", "CPI", "Fuel_Price"]] = scalar.fit_transform(walmart_new[["Weekly_Sales", "CPI", "Fuel_Price"]] )

#scale around min =0 and max = 1
from sklearn.preprocessing import MinMaxScaler
minmax = MinMaxScaler()
walmart_new[["Weekly_Sales", "CPI", "Fuel_Price", "Temperature", "Unemployment"]] = minmax.fit_transform(walmart_new[["Weekly_Sales", "CPI", "Fuel_Price", "Temperature", "Unemployment"]])

#10) Graphical representations
#Histograms
fig, axs = plt.subplots(1, 5, figsize=(15, 3))
axs[0].hist(walmart_new.Weekly_Sales, color = "green")
#plt.show()
axs[1].hist(walmart_new.Temperature)
#.show() #left skewed
axs[2].hist(walmart_new.CPI, color = "red")
#plt.show() #two spearate histograms
axs[3].hist(walmart_new.Unemployment)
#plt.show()
axs[4].hist(walmart_new.Fuel_Price, color = "green")

axs[0].set_title('Weekly Sales')
axs[1].set_title('Temprature')
axs[2].set_title('CPI')
axs[3].set_title('Unemployment')
axs[4].set_title('Fuel Price')
plt.show()

#Box plots
fig1, axs1 = plt.subplots(1, 4, figsize=(15, 3))
axs1[0].boxplot(walmart_new.Weekly_Sales)
#plt.show()
axs1[1].boxplot(walmart_new.Temperature) #again outlier appeared?
#plt.show()
axs1[2].boxplot(walmart_new.CPI)
#plt.show()
axs1[3].boxplot(walmart_new.Fuel_Price)

axs1[0].set_title('Weekly Sales')
axs1[1].set_title('Temprature')
axs1[2].set_title('CPI')
axs1[3].set_title('Fuel Price')
plt.show()


#density plots
columns_plot = ['Weekly_Sales', 'CPI', 'Temperature', 'Fuel_Price', 'Unemployment']

# Create a new DataFrame with columns to plot
walmart_plot = walmart_new[columns_plot]
fig, axs = plt.subplots(1, 5, figsize=(20, 5))

# Use Seaborn's kdeplot function to create density plots for each column
for i, col in enumerate(columns_plot):
    sns.kdeplot(walmart_plot[col], shade=True, ax=axs[i])
    axs[i].set_title(col)
# Show the plot
plt.show()
