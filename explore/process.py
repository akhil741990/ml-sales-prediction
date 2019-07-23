'''
Created on 04-Jul-2019
@author: akhil
'''

import pandas as pd
import numpy as np
from oauthlib.oauth2.rfc6749 import catch_errors_and_unavailability

#load csv
train = pd.read_csv("../data/Train_BigMart.csv")
test = pd.read_csv("../data/Test_BigMart.csv")
train['source'] = 'train'
test['source'] = 'test'
data = pd.concat([train,test],ignore_index=True)
print("data shape",data.shape)
print("describe", data.describe())

print("Non Null\n", data.apply(lambda x: len(x.isnull())))
print("Unique\n", data.apply(lambda x: len(x.unique())))

#Filter categorical variables
categorical_columns = [x for x in data.dtypes.index if data.dtypes[x]=='object']
#Exclude ID cols and source:
categorical_columns = [x for x in categorical_columns if x not in ['Item_Identifier','Outlet_Identifier','source']]
#Print frequency of categories
for col in categorical_columns:
    print ('\nFrequency of Categories for varible %s'%col)
    print (data[col].value_counts())
    
    
    
    
 #Determine the average weight per item:
item_avg_weight = data.pivot_table(values='Item_Weight', index='Item_Identifier')

#Get a boolean variable specifying missing Item_Weight values
miss_bool = data['Item_Weight'].isnull() 

print(item_avg_weight)
#Impute data and check #missing values before and after imputation to confirm

print ('Orignal #missing: %d'% sum(miss_bool))
data.loc[miss_bool,'Item_Weight'] = data.loc[miss_bool,'Item_Identifier'].apply(lambda x: item_avg_weight.ix[x][0])
print ('Final #missing: %d'% sum(data['Item_Weight'].isnull()))   



            


from scipy.stats import mode

#Determing the mode for each
outlet_size_mode = data.pivot_table(values='Outlet_Size', columns='Outlet_Type',aggfunc=(lambda x: x.mode().iat[0]) )
print (outlet_size_mode)


#Get a boolean variable specifying missing Item_Weight values
miss_bool = data['Outlet_Size'].isnull() 

print('\nOrignal #missing: %d'% sum(miss_bool))
outlet_size_mode_t = np.transpose(outlet_size_mode)
data.loc[miss_bool,'Outlet_Size'] = data.loc[miss_bool,'Outlet_Type'].apply(lambda x: outlet_size_mode_t.ix[x][0])
print(sum(data['Outlet_Size'].isnull()))


#Feature Engineering
#Check the mean sales by type:
data.pivot_table(values='Item_Outlet_Sales',index='Outlet_Type')



#Determine average visibility of a product
visibility_avg = data.pivot_table(values='Item_Visibility', index='Item_Identifier')

miss_bool = (data['Item_Visibility'] == 0)
print( 'Number of 0 values initially: %d'%sum(miss_bool))

data.loc[miss_bool,'Item_Visibility'] = data.loc[miss_bool,'Item_Identifier'].apply(lambda x : visibility_avg.ix[x][0])

print('Number of 0 values after modification: %d'%sum(data['Item_Visibility'] == 0))



#Determine another variable with means ratio
data['Item_Visibility_MeanRatio'] = data.apply(lambda x: x['Item_Visibility']/visibility_avg.ix[x['Item_Identifier']][0], axis=1)
print(data['Item_Visibility_MeanRatio'].describe())




#Item type combine
data['Item_Type_Combined'] = data['Item_Identifier'].apply(lambda x: x[0:2]) #first two characters of Identifier
data['Item_Type_Combined'] = data['Item_Type_Combined'].map({'FD':'Food',
                                                             'NC':'Non-Consumable',
                                                             'DR':'Drinks'})
data['Item_Type_Combined'].value_counts()


#Change categories of low fat:
print ('Original Categories:')
print (data['Item_Fat_Content'].value_counts())
data['Item_Fat_Content'] = data['Item_Fat_Content'].replace({'LF':'Low Fat',
                                                             'reg':'Regular',
                                                             'low fat':'Low Fat'})
print (data['Item_Fat_Content'].value_counts())


#Years:
data['Outlet_Years'] = 2013 - data['Outlet_Establishment_Year']
data['Outlet_Years'].describe()



#Mark non-consumables as separate category in low_fat:
data.loc[data['Item_Type_Combined']=="Non-Consumable",'Item_Fat_Content'] = "Non-Edible"
data['Item_Fat_Content'].value_counts()


#Import library:
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
#New variable for outlet
data['Outlet'] = le.fit_transform(data['Outlet_Identifier'])


var_mod = ['Item_Fat_Content','Outlet_Location_Type','Outlet_Size','Item_Type_Combined','Outlet_Type','Outlet']
le = LabelEncoder()
for i in var_mod:
    print("Before",data[i])
    data[i] = le.fit_transform(data[i])
    print("After",data[i])


#One Hot Coding:
data = pd.get_dummies(data, columns=['Item_Fat_Content','Outlet_Location_Type','Outlet_Size','Outlet_Type',
                              'Item_Type_Combined','Outlet'])



data[['Item_Fat_Content_0','Item_Fat_Content_1','Item_Fat_Content_2']].head(10)

#Drop the columns which have been converted to different types:
data.drop(['Item_Type','Outlet_Establishment_Year'],axis=1,inplace=True)



#Divide into test and train:
train = data.loc[data['source']=="train"]
test = data.loc[data['source']=="test"]



#Drop unnecessary columns:
test.drop(['Item_Outlet_Sales','source'],axis=1,inplace=True)
train.drop(['source'],axis=1,inplace=True)

#Export files as modified versions:
train.to_csv("train_modified.csv",index=False)
test.to_csv("test_modified.csv",index=False)



'''Model training'''
''' 1  baseline model.'''
#Mean based:
mean_sales = train['Item_Outlet_Sales'].mean()

#Define a dataframe with IDs for submission:
base1 = test[['Item_Identifier','Outlet_Identifier']]
base1['Item_Outlet_Sales'] = mean_sales

#Export submission file
base1.to_csv("mean.csv",index=False)




''' generic function for training'''
#Define target and ID columns:
target = 'Item_Outlet_Sales'
IDcol = ['Item_Identifier','Outlet_Identifier']
from sklearn import metrics
from sklearn.model_selection import cross_val_score
def modelfit(alg1, dtrain, dtest, predictors, target, IDcol, filename):
    #Fit the algorithm on the data
    alg1.fit(dtrain[predictors], dtrain[target])
        
    #Predict training set:
    dtrain_predictions = alg1.predict(dtrain[predictors])

    #Perform cross-validation:
    #cv_score = cross_val_score(alg1, dtrain[predictors], dtrain[target], cv=20, scoring='mean_squared_error')
    cv_score = cross_val_score(alg1, dtrain[predictors], dtrain[target], cv=20)
    cv_score = np.sqrt(np.abs(cv_score))
    
    #Print model report:
    print("\nModel Report")
    print("RMSE : %.4g" % np.sqrt(metrics.mean_squared_error(dtrain[target].values, dtrain_predictions)))
    print("CV Score : Mean - %.4g | Std - %.4g | Min - %.4g | Max - %.4g" % (np.mean(cv_score),np.std(cv_score),np.min(cv_score),np.max(cv_score)))
    
    #Predict on testing data:
    dtest[target] = alg1.predict(dtest[predictors])
    
    #Export submission file:
    IDcol.append(target)
    submission = pd.DataFrame({ x: dtest[x] for x in IDcol})
    submission.to_csv(filename, index=False)






''' Linear Regression '''
from sklearn.linear_model import LinearRegression, Ridge, Lasso
import matplotlib.pyplot as plt    
    
predictors = [x for x in train.columns if x not in [target]+IDcol]    
alg1 = LinearRegression(normalize=True)
print("Data Predicted by Linear Regression model is witten to file :linear_reg.csv")
modelfit(alg1, train, test, predictors, target, IDcol, 'linear_reg.csv')
coef1 = pd.Series(alg1.coef_, predictors).sort_values()
coef1.plot(kind='bar', title='Model Coefficients')

plt.show()




'''Ridge Regression Model:'''
target = 'Item_Outlet_Sales'
IDcol = ['Item_Identifier','Outlet_Identifier']
predictors = [x for x in train.columns if x not in [target]+IDcol]
alg2 = Ridge(alpha=0.05,normalize=True)
modelfit(alg2, train, test, predictors, target, IDcol, 'ridge_regression.csv')
coef2 = pd.Series(alg2.coef_, predictors).sort_values()
coef2.plot(kind='bar', title='Model Coefficients')
plt.show()


