import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm 
import csv
from pingouin import read_dataset
from games.experimental_statistical_tests import (anova,
                                                  tukeys_hsd,
                                                  run_corrected_t_tests,
                                                  repeated_measures_anova)
                                                  



# Ensure reproducibility of the results
# np.random.seed(123)

# # Create three normally-distributed vectors.
# n = 20
# perf2010 = np.random.normal(loc=5, scale=0.8, size=n)     # Mean = 5, standard deviation = 0.8, size = 20
# perf2014 = np.random.normal(loc=4.8, scale=0.8, size=n)   # Mean = 4.8, ...
# perf2018 = np.random.normal(loc=3, scale=0.8, size=n)     # Mean = 3, ...

# # Concatenate in a long-format pandas DataFrame
# df = pd.DataFrame({'Memory': np.r_[perf2010, perf2014, perf2018],
#                    'Year': np.repeat(['2010', '2014', '2018'], n),
#                    'Subject': np.tile(np.arange(n), 3)})

# df['Sex'] = np.tile(np.repeat(['Men', 'Women'], 10), 3)
# print(df)


# one-way ANOVA and tukey's HSD
# df = pd.read_csv("https://raw.githubusercontent.com/researchpy/Data-sets/master/difficile.csv")
# df.drop('person', axis= 1, inplace= True)
# # Recoding value from numeric to string
# df['dose'].replace({1: 'placebo', 2: 'low', 3: 'high'}, inplace= True)
# # print(df)
# anova_table = anova("", "", df, "libido", "dose", None)
# # print(anova_table)
# tukeys_table = tukeys_hsd("", "", "libido", df, "dose")
# print(tukeys_table)

# two-way ANOVA and tukey's HSD
df = pd.read_csv("/Users/kdreyer/Desktop/ToothGrowth.csv")
df["dose"] = df["dose"].astype(str)
# print(df)
anova_table = anova("", "", df, "len", "supp", "dose")
print(anova_table)
tukeys_table_i, tukeys_tables = tukeys_hsd("", "", "len", df, ["supp", "dose"], ["supp", "dose"])
print(tukeys_table_i)
for table in tukeys_tables:
    print(table)