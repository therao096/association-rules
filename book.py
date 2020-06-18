# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 23:00:55 2020

@author: Varun
"""

import pandas as pd
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import apriori,association_rules

books=pd.read_csv("book.csv")
freq_itemsets=apriori(books,min_support=0.005, max_len=3,use_colnames=True)
freq_itemsets.shape


freq_itemsets.sort_values('support',ascending = False,inplace=True)
plt.bar(x = list(range(1,11)),height = freq_itemsets.support[1:11],color='rgmyk');plt.xticks(list(range(1,11)),freq_itemsets.itemsets[1:11])
plt.xlabel('item-sets');plt.ylabel('support')

rules = association_rules(freq_itemsets, metric="lift", min_threshold=1)
rules.shape
rules.head(20)
rules.sort_values('lift',ascending = False,inplace=True)
rules
conf=rules[''].max()
conf
####1054x9, max lift=22.2973, max conf=1
###conf=rules.sort_values('confidence',ascending=False, inplace=False)
###conf
