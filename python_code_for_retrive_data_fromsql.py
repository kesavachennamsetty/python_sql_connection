import matplotlib as mpl
import os
import pandas as pd
import numpy as np
from numpy import hstack
#from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
import scipy.io
import numpy as np
from numpy import genfromtxt
import itertools
np.random.seed(0)
#import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
#from matplotlib import pyplot as plt
import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=DeprecationWarning)
#from json.decoder import JSONDecodeError
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import metrics
from sklearn.metrics import r2_score
import json
mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.grid'] = False
from sklearn.model_selection import KFold
from sklearn import tree
#import pydotplus
from IPython.display import Image
from scipy.stats import pearsonr
from sklearn.utils import shuffle
import pickle
from sklearn.metrics import precision_recall_fscore_support
from pandas.io.json import json_normalize
###########################################################
import pymysql
import pymssql
import yaml
import pandas as pd
import traceback

def get_creds(db):
    creds = dict()
    with open('/home/shared/utils/creds.yaml') as file:
        creds = yaml.load(file)[db]
    return creds



def bankapp(query):
    # creds = dict()
    # with open('/home/shared/utils/creds.yaml') as file:
    #     creds = yaml.load(file,Loader=yaml.FullLoader)['bankapp']
    creds = get_creds('bankapp')
    try:
        conn = pymysql.connect(host=creds['host'],
                                port=creds['port'],
                                db=creds['database'],
                                user=creds['username'],
                                password=creds['password'])
        df_queried = pd.read_sql_query(query,con=conn)
    finally:
        conn.close()
    return df_queried


def iloans(query):
    # creds = dict()
    # with open('/home/shared/utils/creds.yaml') as file:
    #     creds = yaml.load(file,Loader=yaml.FullLoader)['iloans']
    creds = get_creds('iloans')
    try:
        conn = pymssql.connect(server=creds['server'],
                                port=creds['port'],
                                database=creds['database'],
                                user=creds['username'],
                                password=creds['password'])
        df_queried = pd.read_sql_query(query,con=conn)
    finally:
        conn.close()
    return df_queried






#############FlattenJsonFUnc
def flatten_json(y):
	out = {}
	def flatten(x, name=''):
		if type(x) is dict:
			for a in x:
				flatten(x[a], name + a + '_')
		elif type(x) is list:
			i = 0
			for a in x:
				flatten(a, name + str(i) + '_')
				i += 1
		else:
			out[name[:-1]] = x
	flatten(y)
	return out



def flatten_json_get_loc(y, attr_name):
    out = {}
    attr_addr = ''
    def flatten(x, name=''):
        if type(x) is dict:
            for a in x:
                flatten(x[a], name + a + '_')
        elif type(x) is list:
            i = 0
            for a in x:
                flatten(a, name + str(i) + '_')
                i += 1
        else:
            out[name[:-1]] = x
            if (name[:-1].endswith('_name') and x == attr_name):
                out['attr_addr'] = name[:-5]+str('value')

    flatten(y)
    return out['attr_addr']


def flatten_json_get_attr(y, attr_loc):
    out = {}
    attr_addr = ''
    def flatten(x, name=''):
        if type(x) is dict:
            for a in x:
                flatten(x[a], name + a + '_')
        elif type(x) is list:
            i = 0
            for a in x:
                flatten(a, name + str(i) + '_')
                i += 1
        else:
            out[name[:-1]] = x
            if (name[:-1] == attr_loc):
                out['attr'] = x
    flatten(y)
    return out['attr']


def flatten_json_fill_attr(FJ2, attribute):
	try:
		FJ2[attribute] =  flatten_json_get_attr(Loaded,flatten_json_get_loc(Loaded, attribute))
	except:
		FJ2[attribute] = "NA"
	return FJ2


#Month_num = 1
#Month_name = 'Jan'

pd.set_option('display.max_rows', None)



dataRibbit = iloans('''select LE.leadid, LE.timeadded, r.rawresponse, 
r.actioncode, r.NumericScore as score, rr.rawresponse as BLPresponse, 
LA.loanid, LA.TotalPrincipal, LA.OriginalPrincipal, LA.PaidPrincipal, 
LA.PaidFinanceFee, LA.PaidFeeCharges, 
(case when LA.IsFirstDefault='True' or (LA.IsFirstDefault is null and LS.isbad=1) then 1 else 0 end) as IsFirstDefault,
(Case when ISNULL(LA.LoanAge,0) > 0 then 1 when (LA.LoanStatusID) = 16 then 1 end) as Mature,
LA.loanstatus, LA.LoanStatusid, LS.IsOriginated, LS.IsGood, LS.IsBad 
from dbo.view_FCL_LeadAccepted LE 
left join [FreedomCashLenders].[dbo].[view_FCL_RibbitBVPlusReportData] r on LE.leadid = r.leadid 
left join [FreedomCashLenders].[dbo].[view_FCL_RibbitBLPlusReportData] rr on LE.leadid = rr.leadid 
left  join view_fcl_loan LA on LE.loanid = LA.loanid 
left join view_fcl_loanstatus LS on LA.LoanStatusid = LS.loanStatusId 
where LE.timeadded >= '2022-01-12' and  LE.timeadded < '2022-01-13' and rr.rawresponse is not null
and LE.Subscription not like '%Reloan%' 
and LE.subscription not like '%Admin%' 
and LE.subscription not like '%VIP-Holiday%'
 and LE.subscription not like '%Month Installment%'
and LE.subscription not like '%ReUp%';''')





###############(here dataRibbit = iloans('''write query here''')