# -*- coding: utf-8 -*-
"""
Created on Sat Apr 14 11:51:11 2018

@author: anand
"""


#Basic plots

import matplotlib.pyplot as plt
plt.plot(3,2)#-> nothing visible
plt.plot(5,3,'.')
#plt.figure()
plt.plot(2,3,'.')
ax=plt.gca()			#->(Get current axis, to change and edit the axis)
#also get access to figure usinf gcf function (Get current figure)
ax.axis([0,6,0,6])		#Setting x and y limits



#Scatter plot
#2-d plot

import numpy as np
import matplotlib.pyplot as plt
x=np.array([1,2,3,4])
y=x
plt.scatter(x,y)
y=np.array([4,1,7,3])
plt.figure()
plt.scatter(x,y)

import numpy as np
import matplotlib.pyplot as plt
x=np.array([1,2,3,4])
y=x
plt.scatter(x,y)
y=np.array([4,1,7,3])
plt.scatter(x,y)


import numpy as np
import matplotlib.pyplot as plt
x=np.array([1,2,3,4,5,6])
y=x
plt.figure()
colours=['blue']*(len(x)-1)
colours.extend(['red'])
print(colours)
plt.scatter(x,y,c=colours)

import numpy as np
import matplotlib.pyplot as plt
x=np.array([1,2,3,4,5,6])
y=x
plt.figure()
colours=['green']*(len(x)-1)
colours.extend(['red'])
plt.scatter(x,y,c=colours)
plt.xlabel("Stud Age")
plt.ylabel("Stud Height")
plt.title("Height of Students")
plt.legend()


import numpy as np
import matplotlib.pyplot as plt
x=np.array([1,2,3,4,5,6])
y=x
plt.figure()
colours=['yellow']*(len(x)-1)
colours.extend(['red'])
plt.scatter(x,y,c=colours,label='Height class 1')
z=x+1
plt.scatter(z,y,label='Height class 2')
plt.xlabel("X-Axis")
plt.ylabel("Y-Axis")
plt.title("Height of Students")
plt.legend()


#Line Plots

import numpy as np
import matplotlib.pyplot as plt
linear_data=np.array([1,2,3,4])
quard_data=linear_data**2
plt.figure()
plt.plot(linear_data,'-o')
plt.plot(quard_data,'-o')#No x-axis provided, takes index of numpy as x- axis

import numpy as np
import matplotlib.pyplot as plt

linear_data=np.array([1,2,3,4])
quard_data=linear_data**2

plt.figure()
plt.plot(linear_data,'-o',quard_data,'-o')
plt.xlabel("")
plt.ylabel("")
plt.title("")
plt.legend(['Linear', 'Quadratic'])
plt.gca().fill_between(range(len(linear_data)),linear_data,quard_data,facecolor='blue',alpha=.25)

#With date times
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
linear_data=np.array([1,2,3,4])
quard_data=linear_data**2
dates=np.arange('2017-01-01','2017-01-05',dtype='datetime64[D]')
ob_dates=list(map(pd.to_datetime,dates))
plt.plot(ob_dates,linear_data,'-o',ob_dates,quard_data,'-o')
x=plt.gca().xaxis

for item in x.get_ticklabels():
    item.set_rotation(45)

#Bar Charts

plt.figure()
data=[2,4,6,8]
xval=range(len(data))
plt.bar(xval,data,width=.5)
ax=plt.gca()
ax.axis([0,6,0,10])


plt.figure()
data=[2,4,6,8]
xval=range(len(data))
plt.bar(xval,data,width=.3)
ax=plt.gca()
ax.axis([-1,4,0,10])

new_data=[4,5,7,6]
new_xval=[]
for items in xval:
    new_xval.append(items+.3)    
plt.bar(new_xval,new_data,width=.3)



plt.figure()
data=[2,4,6,8]
xval=range(len(data))
plt.bar(xval,data,width=.5)
ax=plt.gca()
ax.axis([-1,4,0,15])

new_data=[4,5,7,6]  
plt.bar(xval,new_data,width=.5,bottom=data)


plt.figure()
data=[2,4,6,8]
xval=range(len(data))
plt.barh(xval,data,height=.5)
ax=plt.gca()
ax.axis([0,15,-1,5])

new_data=[4,5,7,6]  
plt.barh(xval,new_data,height=.5,left=data)