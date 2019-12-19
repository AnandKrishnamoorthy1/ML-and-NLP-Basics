# -*- coding: utf-8 -*-
"""
Created on Sun Apr 22 11:30:45 2018

@author: anand
"""

import sqlite3
conn=sqlite3.connect(r'D:\Work\DataScience\DS ppt\datascience notes\SQLDB\employee.db')
cur=conn.cursor()
cur.execute('select * from EMPLOYEE')

for rows in cur:
    print(list(rows))
#############################################################################
import sqlite3
import pandas as pd
conn=sqlite3.connect(r'D:\Work\DataScience\DS ppt\datascience notes\SQLDB\employee.db')
emp_set=pd.read_sql('select * from EMPLOYEE',conn)
#############################################################################
import sqlite3
conn=sqlite3.connect(r'D:\Work\DataScience\DS ppt\datascience notes\SQLDB\employee.db')
cur=conn.cursor()
cur.execute("INSERT INTO EMPLOYEE(EMP_ID,NAME,LOCATION,SALARY) VALUES(106,'KIRAN','CHENNAI',70000)")
conn.commit()


#############################################################################
a=[[1,"Ashwin","Chennai"],[2,"Raina","Chennai"],[3,"Steyn","Hydrabad"]]
b=[[2,"Raina","Chennai"],[4,"Kohli","Hydrabad"],[5,"Dhoni","Pune"]]

bowl=pd.DataFrame(a,columns=["ID","Name","Location"])
bat=pd.DataFrame(b,columns=["ID","Name","Location"])

conn=sqlite3.connect(r'D:\Work\DataScience\DS ppt\datascience notes\SQLDB\employee.db')
bowl.to_sql('BOWL',conn,if_exists='append',index=False)
bat.to_sql('BAT',conn,if_exists='append',index=False)
conn.commit()
#############################################################################

a=[[1,"AR Rahman"],[2,"Shreya Goshal"]]
b=[[2,"Raavan",1],[4,"PK",2],[5,"Slumdog",1]]
c=[[1,"Melody"],[2,"Rock"],[3,"Metal"]]
d=[[2,"Bheera",4.3,5,105,2,1],[2,"Khili Re",4.6,4.5,137,2,2],[3,"Char Kadham",4.1,5.2,122,4,3]]

Artist=pd.DataFrame(a,columns=["ID","Name"])
Album=pd.DataFrame(b,columns=["ID","Title","Artist_id"])
Genre=pd.DataFrame(c,columns=["ID","Name"])
Track=pd.DataFrame(d,columns=["ID","Title","Rating","Length","Count","Album_ID","Genre_ID"])

conn=sqlite3.connect(r'D:\Work\DataScience\DS ppt\datascience notes\SQLDB\Album.db')

Artist.to_sql('Artist',conn,if_exists='append',index=False)
Album.to_sql('Album',conn,if_exists='append',index=False)
Genre.to_sql('Genre',conn,if_exists='append',index=False)
Track.to_sql('Track',conn,if_exists='append',index=False)
conn.commit()
#############################################################################


import re
p = re.compile('[A-z ]+')
m=p.match("Hello World999 35")
print(m.group())

import re
p = re.compile('[A-z]+')
m=p.match("@c Hello World")
print(m.group())

import re
p = re.compile('[A-z ]+')
m=p.search("@67 Hello World999 35")
print(m.group())

import re    
p = re.compile('[0-9]+')
val=p.findall('12 drummers drumming, 11 pipers piping, 10 lords a-leaping')  
print(val)

import re    
p = re.compile('[0-9-]+')
val=p.findall('01-03-2018 INFO : Error occured while loading the file')  
print(val)

import re    
p = re.compile('[0-9]{2}[ ,.-][A-z]+[ ,.-][0-9]{2,4}')
val=p.findall('01-March-18 INFO : Error occured while loading the file')
print(val)
