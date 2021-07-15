import pandas as pd


df_1 = pd.DataFrame({'姓名': ["小明","小红","小刚"],
                   '年纪': [10,9,12],
                   '城市': ['上海','北京','深圳']})
print(df_1)

df_2 = pd.DataFrame({'零花钱': [50,200,600,400,80],
                   '城市': ['苏州','北京','上海','广州','重庆']})
print(df_2)

result = pd.merge(df_1, df_2, how='inner')
print(result)
