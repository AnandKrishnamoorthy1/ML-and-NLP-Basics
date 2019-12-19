# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

print("Hello World!")

i=1
if i:
    print("True")
    
    
def my_first_function():
	print('Hello world')    
my_first_function()  


def my_first_function(val):
	print('Hello world: ',val)    
my_first_function(val=4)

score=5
if score>10:
	print('score is greater than 10')
else:
	print('score less than 10')

if score>10
    {print(fdfd)
    }
    
count=1	
while count<5:
	print(count)
	count=count+1
    
for i in range(3,0,-1):
    print(i)

 
#get input
val=input('Enter the value: ')
print(val)


count=-1
print(count)
while count>-2:
	print(count)
	count=count-1
    
#Logical Operation		
x=5
print(type(x))
x='Hello'
print(type(x))

a=9
b=4
print(int(a/b))
print(a+b)
print(a-b)
print(a*b)
print(a**b)
print(abs(-5))
print(int(4.5))	

print('Hello\'s')
print("Hello World")
print('Hello \tWorld')
print(r'Hello \tworld')

name1='test'
name2='py'
print(name1+'\t'+name2)
print(name1.upper())
print(name1.count('t'))
print(name1.index('t'))
print(name1.find('x'))
print(name1.find('e'))

string='My name, is John. He is now, teaching'
print(string.split(','))


import math
x=5
y=10
p1=math.pi
print(round(p1))
print(math.log(x))
print(math.log10(y))
print(math.exp(x))
print(math.sqrt(y))
help(math.exp)


x=-2
y=0
print(bool(x))
print(bool(y))


Lists

x=list()
print(x)

y=[]
print(y)

x.append(4)
y.append([3,5])
x.append(y)
x.pop()#removes and returns last element by default(by index)
print(x)
x=[4]
y=[4]
x.extend([6])
y.append(6)
print(x)
print(y)

x=[2,3,5]
y=[3,4,7]
x.append(y)#creates list inside a list
x.extend(y)#adds multiple elements
x.remove(3)#removes by value
x.pop(2)
print(x)

x=[1,5,0]
print(min(x))
print(sorted(x))#sorted creates a copy

y=[6,8,2,7]
sorted(y)
print(y)
y.sort(reverse=True)
print(y)

x.reverse()


#similar to list(immutable)-Tuple
x=tuple()
print(x)
y=()
print(y)
print(tuple([1,2,[3,5]]))

y=(1,4)
print(y)


x=(1,)
print(x)
y=(1,4,4)
print(y)

print('Str' in x)
print(sorted(y))#convers to a list
print(y.count(4))
print(len(y))

a=[1,4,4]
p=len(a)
for val in range(len(a)):
    print(val)
    
    
my_list = [1,2,3,4,5]
print(type(my_list))
len(my_list)


x=[2,4,6,8]
print(x[0])
print(x[-1])
print(x[1:3])
print(x[1:])
print(x[:3])
    
    
