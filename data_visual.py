####################################################################################################################
            ################   VERİ GÖRSELLEŞTİRME:MATPLOTLIB&SEABORN  ################
################################################################################################################
##############################################################################################################
################################    MATPLOTLIB    ############################################################
##############################################################################################################
#      KATEGORİK DEĞİŞKEN:SÜTUN,GRAFİK,COUNTPLOT BAR
#      SAYISAL DEĞİŞKEN:HİST,BOXPLOT


#      KATEGORİK DEĞİŞKEN GÖRSELLEŞTİRME

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
pd.set_option('display.max_columns',None)
df=sns.load_dataset("titanic")
df.head()
df['sex'].value_counts().plot(kind='bar')
plt.show()

######################## SAYISAL DEĞİŞKEN GÖRSELLEŞTİRME ####################################################
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
pd.set_option('display.max_columns',None)
pd.set_option('display.max_columns',500)
df=sns.load_dataset("titanic")
df.head()
plt.hist(df["age"])
plt.show()
plt.boxplot(df["fare"])
plt.show()

#####################################   MATPLOTLIB'İN ÖZELLİKLERİ   ###############################################
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
pd.set_option('display.max_columns',None)
pd.set_option('display.width',500)

#######################################
# plot
#######################################
x=np.array([1,8])
y=np.array([0,150])
plt.plot(x,y)
plt.show()

plt.plot(x,y,'o')
plt.show()

x=np.array([2,4,6,8,10])
y=np.array([1,3,5,7,9])
plt.plot(x,y)
plt.show()
plt.plot(x,y,'o')
plt.show()

###########################################################################################################
# marker
#############################################################################################################
y=np.array([13,28,11,100])
plt.plot(y,marker='o')
plt.show()
plt.plot(y,marker='*')
plt.show()


markers=['o','*','.',',','x','X','+','P','s','D','d','p','H','h']

###########################################################################################
#   line
###################################################################################################
y=np.array([13,28,11,100])
plt.plot(y,linestyle='dashed')
plt.show()

y=np.array([13,28,11,100])
plt.plot(y)
plt.show()

y=np.array([13,28,11,100])
plt.plot(y,linestyle='dashdot')
plt.show()

y=np.array([13,28,11,100])
plt.plot(y,linestyle="dashdot",color="red")
plt.show()

################################################################################################################
################################ MULTIPLE LINES  ##############################################################
#################################################################################################################
x=np.array([23,18,31,10])
y=np.array([13,28,11,100])
plt.plot(x)
plt.plot(y)
plt.show()

##################################################################################################################
###################################  LABELS  ################################################################
##############################################################################################################

x=np.array([80,85,90,95,100,105,110,115,120,125])
y=np.array([240,250,260,270,280,290,300,310,320,330])
plt.plot(x,y)

# BAŞLIK
plt.title("Bu ana başlık")

# x eksenini isimlendirme

plt.xlabel("x ekseni isimlendirmesi")
plt.ylabel("y ekseni isimlendirmesi")
plt.grid()
plt.show()

####################################################################################################
#   SUBPLOTS
######################################################################################################
# PLOT1
x=np.array([80,85,90,95,100,105,110,115,120,125])
y=np.array([240,250,260,270,280,290,300,310,320,330])
plt.subplot(1,2,1)
plt.title("1")
plt.plot(x,y)

# plot2
x=np.array([8,8,9,9,10,15,11,15,12,15,12,15])
y=np.array([24,20,26,27,280,29,30,30,30,30])
plt.subplot(1,2,2)
plt.title("2")
plt.plot(x,y)
plt.show()

# plot 3
x=np.array([80,85,90,95,100,105,110,115,120,125])
y=np.array([240,250,260,270,280,290,300,310,320,330])
plt.subplot(1,3,3)
plt.title("3")
plt.plot(x,y)
plt.show()

# SEABORN İLE VERİ GÖRSELLEŞTİRME
################################################################################################################
#  SEABORN
################################################################################################################

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
df=sns.load_dataset("tips")
df.head()
df["sex"].value_counts()
sns.countplot(df["sex"],data=df)
plt.show()
df['sex'].value_counts().plot(kind='bar')
plt.show()


#################################################################################################################
#  SAYISAL DEĞİŞKEN GÖRSELLEŞTİRME
##################################################################################################################
sns.boxplot(x=df["total_bill"])
plt.show()

df["total_bill"].hist()
plt.show()

# GELİŞMİŞ FONKSİYONEL KEŞİFÇİ VERİ ANALİZİ

#(ADVANCED FUNCTIONAL EDA)

#1.GENEL BAKIŞ

#2.KATEGORİK DEĞİŞKEN ANALİZİ(ANALYSIS OF CATHEGORICAL VARIABLES)

#3.SAYISAL DEĞİŞKEN ANALİZİ(ANALYSIS OF NUMERICAL VARIABLES)

#4.HEDEF DEĞİŞKEN ANALİZİ(ANALYSIS OF TARGET VARIABLES)

#5.KORELASYON ANALİZİ(ANALYSIS OF CORRELATION)

################################################################################################################
############  GENEL RESİM  #############################################################################
############################################################################################################
import numpy as np
import  pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
pd.set_option('display.max_columns',None)
pd.set_option('display.width',500)
df=sns.load_dataset("titanic")

df.head()

df.tail()

df.shape

df.info()

df.columns

df.index

df.describe().T

df.isnull().values.any()

df.isnull().sum()

def check_df(dataframe,head=5):
    print("############################################SHAPE###########################################")
    print(dataframe.shape)

    print("######################################Types###################################################")
    print(dataframe.dtypes)

    print("#####################################Head########################################################")
    print(dataframe.head(head))

    print("###################################Tail#######################################################")
    print(dataframe.tail(head))

    print("#################################### NA ##################################################")
    print(dataframe.isnull().sum())

    print("################################### Quantiles ############################################")
    print(dataframe.describe([0,0.05,0.50,0.95,0.99]).T)


    check_df()
    df=sns.load_dataset("tips")

    check_df()

    check_df(df)
    df=sns.load_dataset("flights")

    check_df(df)

    #KATEGORİK DEĞİŞKEN ANALİZİ
    #(ANALYSIS OF CATEGORICAL VARIABLES)

    import numpy as np
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    pd.set_option('display.max_columns',None)
    pd.set_option('display.width',500)
    df=sns.load_dataset("titanic")
    df.head()
    df["embarked"].value_counts()
    df["sex"].unique()
    df["sex"].nunique()
    
