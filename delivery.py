'''Simple Linear regression
# Simple linear regression is a regression model that estimates the relationship between one independent variable and a dependent variable using a straight line.

# CRISP-ML(Q) process model describes six phases:
# 
# - Business and Data Understanding
# - Data Preparation (Data Engineering)
# - Model Building (Machine Learning)
# - Model Evaluation and Tunning
# - Deployment
# - Monitoring and Maintenance

# **Problem Statement**

# Relationship between delivery time and sorting time
 
Business Objective - Maximize accurate prediction of delivery time
Business Constarint - Minimize the cost for solution
'''

#import necessary libraries
import pandas as pd
import numpy as np  # deals with numerical values  # for Mathematical calculations"
import matplotlib.pyplot as plt
import joblib
import pickle

from sqlalchemy import create_engine, text
from sklearn.model_selection import train_test_split
import statsmodels.formula.api as smf
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

delivery = pd.read_csv(r"D:/360digimg/ML/Simple Linear Regression/Simple LinearReg/SLR_Dataset/SLR_Dataset/delivery_time.csv")

#info for different data types
delivery.info()
#There are total 2 columns and 21 entries

#Data dictionary
X1 = ["Delivery Time", "Sorting Time"]
X2 = ["Time for delivery", "Time for sorting"]
X3 = [ "Numerical", "Numerical"]
X4 = ["Relevant", "Relevant"]

Data_types = pd.DataFrame({"Name of feature": X1,"Description": X2, "Data type": X3,"Relevance": X4})

#connect to SQL database
user = 'root' # user name
pw = 'swati@1234' # password
db = 'assignment_ML' # database
from urllib.parse import quote 
engine = create_engine(f"mysql+pymysql://{user}:%s@localhost/{db}" % quote(f'{pw}'))

delivery.to_sql('delivery',con = engine, if_exists = 'replace', index = False)

delivery.head(10)


#Data pre processing steps
#Check for duplicates
dupli = delivery.duplicated().sum()
print(dupli) #no dulicates

#Check for missing values
delivery.isnull().sum() #no missing values


#First and second moment business decision
delivery.describe()
#mean is approximately equal to median for sorting time

#Third moment business decision
delivery.skew()
#For delivery time, positively skewed, some outliers on right side of histribution

#Fourth moment business decision
delivery.kurt()
#For deliveryt time, Positive kurtosis indicates heavier tails and a more peaked distribution, outliers
#For sorting time, negative kurtosis suggests lighter tails and a flatter distribution

#EDA
#univariate analysis
# Create a histogram for the features
delivery.hist(bins=20, figsize=(14, 5))
plt.show()

#density distribution plot
delivery.plot(kind='density', subplots=True, layout=(1, 4), sharex=False, figsize=(12, 3))
plt.subplots_adjust(wspace=0.5)
plt.show()
#seems like normal distribution for both columns while delivery time has sharper peak

#Box plot for outliers
delivery.plot(kind ='box', subplots = True, layout =(2,5), sharex = False, sharey = False, figsize=(12, 8))
# Add space between subplots horizontally
plt.subplots_adjust(wspace=0.5)
plt.show()
#no outliers

#Bivariate analysis
#scatter plot b/s delivery time and sorting time
delivery.plot(kind='scatter', x='Delivery Time', y='Sorting Time', s=80, figsize=(12, 8))
plt.show()
#Positive correlation b/w two
#Let's find the coorelation value
delivery.corr() # 0.825997

# Rename the columns with underscores
delivery.columns = delivery.columns.str.replace(' ', '_')
delivery.sort_values(by = ['Sorting_Time'], axis = 0, inplace = True)

# Simple Linear Regression
# Fit the model
model = smf.ols('Delivery_Time ~ Sorting_Time', data=delivery).fit()
model.summary() #R-squared : 0.682

pred1 = model.predict(pd.DataFrame(delivery['Sorting_Time']))
pred1

# Regression Line
plt.scatter(delivery['Sorting_Time'], delivery['Delivery_Time'])
plt.plot(delivery['Sorting_Time'], pred1, "r")
plt.legend(['Observed data', 'Predicted line'])
plt.show()

# Error calculation (error = AV - PV)
res1 = delivery['Delivery_Time'] - pred1
print(np.mean(res1)) #-6.259*10^-15

res_sqr1 = res1 * res1
mse1 = np.mean(res_sqr1)
rmse1 = np.sqrt(mse1)
rmse1 #2.7916


# # Model Tuning with Transformations
# ## Log Transformation

plt.scatter(x = np.log(delivery['Sorting_Time']), y = delivery['Delivery_Time'], color = 'brown')
np.corrcoef(np.log(delivery['Sorting_Time']), delivery['Delivery_Time']) #correlation

model2 = smf.ols('Delivery_Time ~ np.log(Sorting_Time)', data = delivery).fit()
model2.summary() #R-squared : 0.695

pred2 = model2.predict(pd.DataFrame(delivery['Sorting_Time']))

# Regression Line
plt.scatter(np.log(delivery['Sorting_Time']), delivery['Delivery_Time'])
plt.plot(np.log(delivery['Sorting_Time']), pred2, "r")
plt.legend(['Observed data', 'Predicted line'])
plt.show()

# Error calculation
res2 = delivery['Delivery_Time'] - pred2
res_sqr2 = res2 * res2
mse2 = np.mean(res_sqr2)
rmse2 = np.sqrt(mse2)
rmse2 #2.733


# ## Exponential transformation

plt.scatter(x = delivery['Sorting_Time'], y = np.log(delivery['Delivery_Time']), color = 'orange')
np.corrcoef(delivery['Sorting_Time'], np.log(delivery['Delivery_Time'])) #correlation

model3 = smf.ols('np.log(Delivery_Time) ~ Sorting_Time', data = delivery).fit()
model3.summary() #0.711

pred3 = model3.predict(pd.DataFrame(delivery['Sorting_Time']))

# Regression Line
plt.scatter(delivery['Sorting_Time'], np.log(delivery['Delivery_Time']))
plt.plot(delivery['Sorting_Time'], pred3, "r")
plt.legend(['Observed data', 'Predicted line'])
plt.show()

pred3_sort = np.exp(pred3)
print(pred3_sort)

res3 = delivery['Delivery_Time'] - pred3_sort
res_sqr3 = res3 * res3
mse3 = np.mean(res_sqr3)
rmse3 = np.sqrt(mse3)
rmse3 #2.94


# ## Polynomial transformation 

X = pd.DataFrame(delivery['Sorting_Time'])

Y = pd.DataFrame(delivery['Delivery_Time'])


model4 = smf.ols('np.log(Delivery_Time) ~ Sorting_Time + I(Sorting_Time*Sorting_Time)', data = delivery).fit()
model4.summary() #R squared 0.765

pred4 = model4.predict(pd.DataFrame(delivery))
print(pred4)

plt.scatter(X['Sorting_Time'], np.log(Y['Delivery_Time']))
plt.plot(X['Sorting_Time'], pred4, color = 'red')
plt.plot(X['Sorting_Time'], pred3, color = 'green', label = 'linear')
plt.legend(['Transformed Data', 'Polynomial Regression Line', 'Linear Regression Line'])
plt.show()

pred4_sort = np.exp(pred4)
pred4_sort

# Error calculation
res4 = delivery['Delivery_Time'] - pred4_sort
res_sqr4 = res4 * res4
mse4 = np.mean(res_sqr4)
rmse4 = np.sqrt(mse4)
rmse4 #2.7999


# ### Choose the best model using RMSE
data = {"MODEL":pd.Series(["SLR", "Log model", "Exp model", "Poly model"]), "RMSE":pd.Series([rmse1, rmse2, rmse3, rmse4])}
table_rmse = pd.DataFrame(data)

table_rmse
#log model has least RMSE 2.733


# # Evaluate the best model
# Data Split
train, test = train_test_split(delivery, test_size = 0.2, random_state = 0)

plt.scatter(x = np.log(train['Sorting_Time']), y = train['Delivery_Time'], color = 'brown')
plt.figure(2)
plt.scatter(x= np.log(test['Sorting_Time']), y = test['Delivery_Time'])

# Fit the best model on train data
finalmodel = smf.ols('Delivery_Time ~ np.log(Sorting_Time)', data = train).fit()

# Prediction on train data
train_pred = finalmodel.predict(pd.DataFrame(train))

# Model Evaluation on train data
train_res = train['Delivery_Time'] - train_pred
train_sqrs = train_res * train_res
train_mse = np.mean(train_sqrs)
train_rmse = np.sqrt(train_mse)

train_rmse #3.02265

# Predict on test data
test_pred = finalmodel.predict(test)

# Model Evaluation on Test data
test_res = test['Delivery_Time'] - test_pred
test_sqrs = test_res * test_res
test_mse = np.mean(test_sqrs)
test_rmse = np.sqrt(test_mse)

test_rmse #1.48

#works better on test data rather than train data


