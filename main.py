import pandas as pd 
data=pd.read_csv("loan_data_1.csv")
print(data)

#step2. Data analysis
print(data.head()) #first 5 rows print ,head ma argument for how many rows
print(data.tail()) # last 5 rows print ,tail ma argument for how many rows
print(data.info()) # this function gives the information about the data like how many rows and columns are there and what type of data is there
print(data.isnull().sum()) #this function gives the information about the missing values in the data
print(data.duplicated().sum()) #this function gives the information about the duplicate values in the data
data=data.drop(["Loan_ID"],axis=1)  # drop the column axis=1 drop the column and axis=0 drop the row
print(data) 
print(data.isnull().sum()) #this function gives the information about the missing values in the data
data["Gender"]=data["Gender"].fillna(data["Gender"].mode()[0]) #handling missing values by mode
data["Married"]=data["Married"].fillna(data["Married"].mode()[0]) #handling missing values by mode
data["Dependents"]=data["Dependents"].fillna(data["Dependents"].mode()[0]) #handling missing values by mode
data["Education"]=data["Education"].fillna(data["Education"].mode()[0]) #handling missing values by mode
data["Self_Employed"]=data["Self_Employed"].fillna(data["Self_Employed"].mode()[0]) #handling missing values by mode
data["ApplicantIncome"]=data["ApplicantIncome"].fillna(data["ApplicantIncome"].mean()) #handling missing values by mean
data["CoapplicantIncome"]=data["CoapplicantIncome"].fillna(data["CoapplicantIncome"].mean()) #handling missing values by mean
data["LoanAmount"]=data["LoanAmount"].fillna(data["LoanAmount"].mean()) #handling missing values by mean
data["Loan_Amount_Term"]=data["Loan_Amount_Term"].fillna(data["Loan_Amount_Term"].mode()[0]) #handling missing values by mode
data["Credit_History"]=data["Credit_History"].fillna(data["Credit_History"].mode()[0]) #handling missing values by mode
# data["Property_Area"]=data["Property_Area"].fillna(data["Property_Area"].mean()) #handling missing values by mean
# data["Loan_Status"]=data["Loan_Status"].fillna(data["Loan_Status"].mean()) #handling missing values by mean
print(data.isnull().sum())

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
columns=['Gender','Married','Dependents','Education','Self_Employed','Property_Area','Loan_Status']
for col in columns:
    data[col]=le.fit_transform(data[col]) #transform colum to numeric , as we have to train model not possible with text

#split data into features and target variables
x=data.drop("Loan_Status",axis=1)
y=data["Loan_Status"]

#split data into training and testing sets
from sklearn.model_selection import train_test_split
#test_size=0.2 means 20% of the data will be used for testing and 80% for training
#train_size=0.8 means 80% of the data will be used for training and 20% for testing
#shuffle=True means the data will be shuffled before splitting
#random_state=42 means the data will be shuffled in a random state of 42
#if we not provide test_size and train_size then test_size will be 0.2 and train_size will be 0.8
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,train_size=0.8,shuffle=True,random_state=42)
#shape (rows,columns) of training and testing sets 
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

#model train
from sklearn.linear_model import LogisticRegression
model=LogisticRegression()
model.fit(x_train,y_train)

#prediction model
y_predict=model.predict(x_test)
print(y_predict)

#model evaluation
from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_predict)
print(accuracy)