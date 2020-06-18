# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 17:42:08 2020

@author: Varun
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
movie=pd.read_csv("my_movies.csv")
movie
movie= movie.iloc[:,5:]

from mlxtend.frequent_patterns import apriori,association_rules

freq=apriori(movie,min_support=0.005, max_len=3,use_colnames=True)
freq.shape

freq.sort_values('support', ascending=False, inplace=True)
freq

plt.bar(x = list(range(1,11)),height = freq.support[1:11],color='rgmyk');plt.xticks(list(range(1,11)),freq.itemsets[1:11])
plt.xlabel('item-sets');plt.ylabel('support')
rules = association_rules(freq, metric="lift", min_threshold=1)
rules.head(20)
rules.sort_values('lift',ascending = False,inplace=True)
rules
###max lift=10, max support=0.6

### when max length =4

freqs=apriori(movie,min_support=0.005, max_len=4,use_colnames=True)
freqs.shape

freqs.sort_values('support', ascending=False, inplace=True)
freqs

plt.bar(x = list(range(1,11)),height = freqs.support[1:11],color='rgmyk');plt.xticks(list(range(1,11)),freqs.itemsets[1:11])
plt.xlabel('item-sets');plt.ylabel('support')
ruless = association_rules(freqs, metric="lift", min_threshold=1)
ruless.head(20)
ruless.sort_values('lift',ascending = False,inplace=True)
ruless


