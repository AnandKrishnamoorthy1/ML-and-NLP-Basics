# -*- coding: utf-8 -*-
"""
Created on Sun Apr  8 10:23:59 2018

@author: anand
"""

dict_temp= {'Name': 'Zara', 'Age': 7, 'Class': 'First'}
print ("dict['Name']: ", dict_temp['Name'])
print ("dict['Age']: ", dict_temp['Age'])

dict_temp = {'Name': 'Zara', 'Age': 7, 'Class': 'First'}
print ("dict['Alice']: ", dict_temp['Alice'])
dict_temp['Gender']='F'
dict_temp['Name']='Alice'
dict_temp['Age'] = 8

del dict_temp['Name'] # remove entry with key 'Name'
dict_temp.clear()     # remove all entries in dict
print(type(dict_temp))
del dict_temp



names=["Kohli","Dhoni","Aswin"]
def names_upper(names):
    return names.upper()
names_upper=list(map(names_upper,names))
print(names_upper)

new_names=list(map(lambda x:x.upper(),names))
print(new_names)

my_list=[2,4]
print(type(my_list))
my_list=list()
print(type(my_list))
typ_lst=(3,5)
print(type(typ_lst))
typ_lst=tuple()
print(type(typ_lst))

for number in range(1,20,2):
    my_list.append(number)

for number in range(10):
    print(number)
    
my_list=[]
for number in range(0,20):
    if number%2==0:
        my_list.append(number)
        
new_my_list=[number for number in range(0,20) if number%2==0]
print(new_my_list)

x=[2,3,4]

for val in range(len(x)):
    print(val*2)
my_list=[]    
for val in x:
    my_list.append(val*2)
    
#Numpy

import numpy as np
arry_val=[3,5,8,5]
val=np.array(arry_val)
print(val*2)
print(type(val))

import numpy as np
arry_val=[3,5,8,5]
val=np.array(arry_val)
print(val)


arry_val=[[5,8,5],[2,4,3]]
val=np.array(arry_val)
print(val)
print(val.shape)

n=np.arange(0,30,2)
print(n)

n.reshape(3,5)
print(n)
n.resize(3,5)
print(n)
print(n.T)
n=np.linspace(0,10,8)
print(n)

#Operations
x=np.array([5,3,6])
y=np.array([2,2,2])
print(x+y)
print(x*y)

y=3
z=np.array([y,y*2])
print(z)
print(z.T)
print(x.max())
print(x.mean())
print(x.std())

n=np.arange(0,30,2)
print(n[1:5])
print(n[-5:])
print(n[-5::2])


val=np.arange(36)
val.resize(6,6)
print(val)
print(val[2,3])
print(val[1:3,1:3])
print(val[val>28])
val[val>28]=30
val[val<5]=0
print(val)


#Iteration
new_test=np.arange(10)
new_test.resize(2,5)
print(new_test)

for row in new_test:
    print(row)
    
for i in range(len(new_test)):
    print(new_test[i])
    
for i,row in enumerate(new_test):
    print("i: ",i,"val: ",row)
       

#Pandas
import pandas as pd
#pd.Series?

birds=["Eagle","Parrot","Hen"]	
birds_Ser=pd.Series(birds)
print(birds_Ser)

val=[4,2,7]
val_ser=pd.Series(val)
print(val_ser)


birds=["Eagle","Parrot","Hen",None]	
birds_Ser=pd.Series(birds)
print(birds_Ser)

val=[4,2,7,None]
val_ser=pd.Series(val)
print(val_ser)


animals={"India":"Tiger","New Zealand":"Kiwi","Australia":"Kangaroo"}
#animals=["Tiger","Kiwi","Kangaroo"]
animals_ser=pd.Series(animals)
print(type(animals_ser.values))

animals=pd.Series(["Tiger","Kiwi","Kangaroo"],["India","New Zealand","Australia"])

print(animals)


#Series Querying
animals.iloc[2]
animals.loc["Australia"]

val=pd.Series(np.random.randint(0,1000,10000))
import time
import numpy as np

def lst_total():
    total=0
    for item in val:
        total+=item
 
def numpy_tot():
    total=np.sum(val)
    
import timeit
print(timeit.timeit(lst_total,number=100))
print(timeit.timeit(numpy_tot,number=100))


def lst_total():
    total=0
    for item in val:
        total+=item
        
    print(total)
    
lst_total()


animals={"India":"Tiger","New Zealand":"Kiwi","Australia":"Kangaroo"}
new_animals=pd.Series(["Ostritch"],index=["New Zealand"])
animals_ser=pd.Series(animals)
animals_ser=(animals_ser.append(new_animals))
animals_ser.index
animals_ser.values
animals_ser.name
print(animals_ser.loc["New Zealand"])
print(animals_ser.iloc[2])


#dataframes
india={"Gold":2,"Silver":5,"Bronze":4}
china={"Gold":20,"Silver":18,"Bronze":9}
Pakisthan={"Gold":1,"Silver":3,"Bronze":6}

olympics=pd.DataFrame([india,china,Pakisthan],index=["india","china","Pakisthan"])
print(olympics)

print(olympics.iloc[1])
print(olympics.loc["india"])
print(olympics.iloc[0])


import pandas as pd
purchase_1 = pd.Series({'Name': 'Arun',
                        'Item Purchased': 'Food',
                        'Cost': 22.50})
purchase_2 = pd.Series({'Name': 'Ron',
                        'Item Purchased': 'Biscuit',
                        'Cost': 25.0})
purchase_3 = pd.Series({'Name': 'Vinoth',
                        'Item Purchased': 'Perfume',
                        'Cost': 150.0})

df_purchase=pd.DataFrame([purchase_1,purchase_2,purchase_3])
    
df = pd.DataFrame([purchase_1, purchase_2, purchase_3], index=['Store 1', 'Store 1', 'Store 2'])

    
df[(df.index=='Store 1') | (df.index=='Store 2')]

df.loc['Store 2','Cost']
df.loc['Store 2',['Cost','Name']]

df.loc[:,['Cost','Name']]

#Dropping rows
df_temp=df.copy()
df_temp=df_temp.drop('Store 2')
del df_temp["Cost"]

df_temp["Age"]=14
df["Cost"]=df["Cost"]*2

for i in range(5):
    for j in range(5,10):
        print("i:",i,"j:",j)
        
df_meeting=pd.read_csv(r"D:\Work\DataScience\DS ppt\datascience notes\assignments\HalfHourParentTeacherConferenceSampleImportFile.csv",skiprows=1,index_col=0)
df_meeting=pd.read_csv("OneHourParentTeacherConference_Test.csv")
print(df_meeting)
df_meeting.rename(columns={"Location":"Room No"},inplace=True)
print(df_meeting)


import pandas as pd
#dataframes
india={"Gold":2,"Silver":5,"Bronze":4}
china={"Gold":20,"Silver":18,"Bronze":9}
pakisthan={"Gold":1,"Silver":3,"Bronze":6}
usa={"Gold":32,"Silver":35,"Bronze":19}
france={"Gold":15,"Silver":32,"Bronze":13}

olympics=pd.DataFrame([india,china,pakisthan,usa,france],index=["india","china","pakisthan","usa","france"])
print(olympics)


print(olympics['Silver']>10)
mask=olympics['Silver']>10
print(olympics[mask])
mask2=(olympics['Bronze']>10) & (olympics['Silver']>10)
print(olympics[mask2])


newolympics=olympics[["Gold","Silver"]]
olympics['Countries']=olympics.index

olympics=olympics.reset_index()
olympics=olympics.set_index(['Gold','Countries'])

print(olympics.iloc[0])

import pandas as pd
#Pandas Merge
a=[[1,"Ashwin","Chennai"],[2,"Raina","Chennai"],[3,"Steyn","Hydrabad"]]
b=[[2,"Raina","Chennai"],[4,"Kohli","Hydrabad"],[5,"Dhoni","Pune"]]

bowl=pd.DataFrame(a,columns=["ID","Name","Loc"])
bat=pd.DataFrame(b,columns=["ID","Name","Loc"])


val=pd.merge(bowl,bat,how='inner',left_index=True,right_index=True)
val=pd.merge(bowl,bat,how='inner',left_on="ID",right_on="ID")

val=pd.merge(bowl,bat,how='outer',left_on=["ID","Name","Loc"],right_on=["ID","Name","Loc"])
val=pd.merge(bowl,bat,how='left',left_on="ID",right_on="ID")
val=pd.merge(bowl,bat,how='right',left_on="ID",right_on="ID")


for i,row in val.groupby("Loc"):
    print(i)
    print(row)

raw_data = {
        'subject_id': ['1', '2', '3', '4', '5'],
        'first_name': ['Alex', 'Amy', 'Allen', 'Alice', 'Ayoung'], 
        'last_name': ['Anderson', 'Ackerman', 'Ali', 'Aoni', 'Atiches']}
df_a = pd.DataFrame(raw_data, columns = ['subject_id', 'first_name', 'last_name'])
print(df_a)

raw_data = {
        'subject_id': ['4', '5', '6', '7', '8'],
        'first_name': ['Billy', 'Brian', 'Bran', 'Bryce', 'Betty'], 
        'last_name': ['Bonder', 'Black', 'Balwner', 'Brice', 'Btisan']}
df_b = pd.DataFrame(raw_data, columns = ['subject_id', 'first_name', 'last_name'],index=[2,3,4,5,6])
df_b

raw_data = {
        'subject_id': ['1', '2', '3', '4', '5', '7', '8', '9', '10', '11'],
        'test_id': [51, 15, 15, 61, 16, 14, 15, 1, 61, 16]}
df_n = pd.DataFrame(raw_data, columns = ['subject_id','test_id'])

val=pd.merge(df_a, df_n, on='subject_id', how='left')