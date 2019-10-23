# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 01:40:10 2019

@author: anand
"""

###Text Operations. Splitting into sentences naively.

text="A line containing only whitespace, possibly with a comment, is known as a blank line and Python totally ignores it."
len(text)

text="A line containing only whitespace, possibly with a comment, is known as a blank line and Python totally ignores it."
tokens=text.split(' ')
print(len(tokens))
print((tokens))

new_tokens=[token for token in tokens if len(token)>=3]
print(new_tokens)

title_tokens=[token for token in tokens if token.istitle()]
print(title_tokens)

text=text.lower()


text="   A line containing only whitespace, possibly with a comment, is known as a blank line and Python totally ignores it.   "
tokens=text.split(' ')
print(tokens)

text="   A line containing only whitespace, possibly with a comment, is known as a blank line and Python totally ignores it.   "
text=text.strip()
tokens=text.split(' ')
print(tokens)

print(text.find('o'))
print(text.rfind('o'))
print(text.replace('o','O'))

###############################################################################
#Parsing Dates using regular expressions

import re
text="2018-02-11 03:33:08,358 [archive-weekly1_2428] INFO  [com.AbstractServiceCommand]: Calling command : "
date_pat='([0-9]{2,4}[-/][0-9]{2}[-/][0-9]{2})'
date_pat_comp=re.compile(date_pat)
date_pat_comp.findall(text)


import re
text="Oct 2017 [archive-weekly1_2428] INFO  [com.AbstractServiceCommand]: Calling command : "
date_pat='([Jan|Feb|Oct]+[ ][0-9]{2,4})'
date_pat_comp=re.compile(date_pat)
date_pat_comp.findall(text)

import pandas as pd

time_sentences = ["Monday: The doctor's appointment is at 2:45pm.", 
                  "Tuesday: The dentist's appointment is at 11:30 am.",
                  "Wednesday: At 7:00pm, there is a basketball game!",
                  "Thursday: Be back home by 11:15 pm at the latest.",
                  "Friday: Take the train at 08:10 am, arrive at 09:00am."]

df = pd.DataFrame(time_sentences, columns=['text'])
print(df)
print(df['text'].str.len())

time='[0-9]{1,2}[:][0-9]{1,2}'
df['text'].str.findall(time)