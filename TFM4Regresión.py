{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {
    "azdata_cell_guid": "f0e72443-ffd0-4847-9a85-1c375a72bc4f"
   },
   "outputs": [],
   "source": [
    "import squarify\n",
    "import os\n",
    "    \n",
    "# Main libraries that we will use in this kernel\n",
    "import datetime\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "\n",
    "# # garbage collector: free some memory is needed\n",
    "import gc\n",
    "import matplotlib\n",
    "import matplotlib.pyplot as plt\n",
    "import plotly.express as px\n",
    "import seaborn as sns\n",
    "\n",
    "# statistical package and some useful functions to analyze our timeseries\n",
    "from statsmodels.tsa.stattools import pacf\n",
    "from statsmodels.graphics.tsaplots import plot_acf, plot_pacf\n",
    "from statsmodels.tsa.seasonal import seasonal_decompose\n",
    "import statsmodels.tsa.stattools as stattools\n",
    "\n",
    "import time\n",
    "\n",
    "from string import punctuation\n",
    "from sklearn.preprocessing import LabelEncoder\n",
    "from sklearn.linear_model import LinearRegression\n",
    "\n",
    "import warnings\n",
    "warnings.filterwarnings(\"ignore\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {
    "azdata_cell_guid": "10eb9938-bb1c-4a6f-b584-2907aa120466"
   },
   "outputs": [],
   "source": [
    "from sklearn import preprocessing"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {
    "azdata_cell_guid": "bc7f16db-3af4-4634-82d5-881a8786ce32"
   },
   "outputs": [],
   "source": [
    "# python core library for machine learning and data science\n",
    "from sklearn.pipeline import Pipeline\n",
    "from sklearn.preprocessing import StandardScaler, MinMaxScaler\n",
    "from sklearn.base import BaseEstimator, TransformerMixin\n",
    "from sklearn.impute import KNNImputer, SimpleImputer\n",
    "from sklearn.cluster import KMeans"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 540,
   "metadata": {
    "azdata_cell_guid": "12400a40-ad75-4e6a-9b1b-b3f585c9b8d3"
   },
   "outputs": [],
   "source": [
    "#full_3 = pd.read_pickle('./full_3.pickle')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {
    "azdata_cell_guid": "292a7689-642e-40df-a221-853a7efc034e"
   },
   "outputs": [],
   "source": [
    "full_3OHE = pd.read_pickle('./full_3OHE.pickle')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 112,
   "metadata": {
    "azdata_cell_guid": "22afde4a-8fe0-498a-adb8-c185a70c0a30"
   },
   "outputs": [],
   "source": [
    "#full_3.to_csv('full_3.csv') "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {
    "azdata_cell_guid": "6134d418-7fd4-4cc4-bce8-1ed1da91c720"
   },
   "outputs": [],
   "source": [
    "#full_3[\"Year\"] = full_3[\"date\"].dt.year"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {
    "azdata_cell_guid": "56a6ba8e-4c2a-43a9-b18b-07082a3c8ded"
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Productos</th>\n",
       "      <th>Ventas_cant</th>\n",
       "      <th>Precios</th>\n",
       "      <th>Familia_prod</th>\n",
       "      <th>Ingresos</th>\n",
       "      <th>entry_date</th>\n",
       "      <th>entry_channel</th>\n",
       "      <th>días_encartera</th>\n",
       "      <th>pk_cid</th>\n",
       "      <th>country_id</th>\n",
       "      <th>region_code</th>\n",
       "      <th>gender</th>\n",
       "      <th>age</th>\n",
       "      <th>deceased</th>\n",
       "      <th>date</th>\n",
       "      <th>Month</th>\n",
       "      <th>days_between</th>\n",
       "      <th>recurrencia</th>\n",
       "      <th>Year</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>22134</th>\n",
       "      <td>short_term_deposit</td>\n",
       "      <td>1.0</td>\n",
       "      <td>40</td>\n",
       "      <td>Inversión</td>\n",
       "      <td>40.0</td>\n",
       "      <td>2018-04-08</td>\n",
       "      <td>KHK</td>\n",
       "      <td>20.0</td>\n",
       "      <td>100296</td>\n",
       "      <td>ES</td>\n",
       "      <td>28.0</td>\n",
       "      <td>V</td>\n",
       "      <td>46</td>\n",
       "      <td>N</td>\n",
       "      <td>2018-04-28</td>\n",
       "      <td>4</td>\n",
       "      <td>0.0</td>\n",
       "      <td>1</td>\n",
       "      <td>2018</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>22135</th>\n",
       "      <td>short_term_deposit</td>\n",
       "      <td>1.0</td>\n",
       "      <td>40</td>\n",
       "      <td>Inversión</td>\n",
       "      <td>40.0</td>\n",
       "      <td>2018-04-08</td>\n",
       "      <td>KHK</td>\n",
       "      <td>50.0</td>\n",
       "      <td>100296</td>\n",
       "      <td>ES</td>\n",
       "      <td>28.0</td>\n",
       "      <td>V</td>\n",
       "      <td>46</td>\n",
       "      <td>N</td>\n",
       "      <td>2018-05-28</td>\n",
       "      <td>5</td>\n",
       "      <td>30.0</td>\n",
       "      <td>2</td>\n",
       "      <td>2018</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>22136</th>\n",
       "      <td>short_term_deposit</td>\n",
       "      <td>1.0</td>\n",
       "      <td>40</td>\n",
       "      <td>Inversión</td>\n",
       "      <td>40.0</td>\n",
       "      <td>2018-04-08</td>\n",
       "      <td>KHK</td>\n",
       "      <td>81.0</td>\n",
       "      <td>100296</td>\n",
       "      <td>ES</td>\n",
       "      <td>28.0</td>\n",
       "      <td>V</td>\n",
       "      <td>46</td>\n",
       "      <td>N</td>\n",
       "      <td>2018-06-28</td>\n",
       "      <td>6</td>\n",
       "      <td>31.0</td>\n",
       "      <td>3</td>\n",
       "      <td>2018</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>27826</th>\n",
       "      <td>short_term_deposit</td>\n",
       "      <td>1.0</td>\n",
       "      <td>40</td>\n",
       "      <td>Inversión</td>\n",
       "      <td>40.0</td>\n",
       "      <td>2015-01-30</td>\n",
       "      <td>KHM</td>\n",
       "      <td>1337.0</td>\n",
       "      <td>1003705</td>\n",
       "      <td>ES</td>\n",
       "      <td>28.0</td>\n",
       "      <td>H</td>\n",
       "      <td>33</td>\n",
       "      <td>N</td>\n",
       "      <td>2018-09-28</td>\n",
       "      <td>9</td>\n",
       "      <td>0.0</td>\n",
       "      <td>1</td>\n",
       "      <td>2018</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                Productos  Ventas_cant  Precios Familia_prod  Ingresos  \\\n",
       "22134  short_term_deposit          1.0       40    Inversión      40.0   \n",
       "22135  short_term_deposit          1.0       40    Inversión      40.0   \n",
       "22136  short_term_deposit          1.0       40    Inversión      40.0   \n",
       "27826  short_term_deposit          1.0       40    Inversión      40.0   \n",
       "\n",
       "      entry_date entry_channel  días_encartera   pk_cid country_id  \\\n",
       "22134 2018-04-08           KHK            20.0   100296         ES   \n",
       "22135 2018-04-08           KHK            50.0   100296         ES   \n",
       "22136 2018-04-08           KHK            81.0   100296         ES   \n",
       "27826 2015-01-30           KHM          1337.0  1003705         ES   \n",
       "\n",
       "       region_code gender  age deceased       date  Month  days_between  \\\n",
       "22134         28.0      V   46        N 2018-04-28      4           0.0   \n",
       "22135         28.0      V   46        N 2018-05-28      5          30.0   \n",
       "22136         28.0      V   46        N 2018-06-28      6          31.0   \n",
       "27826         28.0      H   33        N 2018-09-28      9           0.0   \n",
       "\n",
       "       recurrencia  Year  \n",
       "22134            1  2018  \n",
       "22135            2  2018  \n",
       "22136            3  2018  \n",
       "27826            1  2018  "
      ]
     },
     "execution_count": 6,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#full_3.head(4)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {
    "azdata_cell_guid": "a0c924ff-7225-489f-9934-b7316b13674f"
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(6254518, 19)"
      ]
     },
     "execution_count": 7,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#full_3.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {
    "azdata_cell_guid": "2ca69708-3018-4653-9a14-9239ebb10fa3"
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "1128353    149\n",
       "1190607    145\n",
       "1070525    142\n",
       "1053446    136\n",
       "1054409    136\n",
       "          ... \n",
       "1531265      1\n",
       "1461038      1\n",
       "1533505      1\n",
       "1549485      1\n",
       "1550590      1\n",
       "Name: pk_cid, Length: 350384, dtype: int64"
      ]
     },
     "execution_count": 8,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#full_3['pk_cid'].value_counts()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {
    "azdata_cell_guid": "df59c42a-b4b7-4d0c-93bf-1b1ba230e1f6"
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Productos</th>\n",
       "      <th>Familia_prod</th>\n",
       "      <th>entry_date</th>\n",
       "      <th>entry_channel</th>\n",
       "      <th>pk_cid</th>\n",
       "      <th>country_id</th>\n",
       "      <th>gender</th>\n",
       "      <th>deceased</th>\n",
       "      <th>date</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>22134</th>\n",
       "      <td>short_term_deposit</td>\n",
       "      <td>Inversión</td>\n",
       "      <td>2018-04-08</td>\n",
       "      <td>KHK</td>\n",
       "      <td>100296</td>\n",
       "      <td>ES</td>\n",
       "      <td>V</td>\n",
       "      <td>N</td>\n",
       "      <td>2018-04-28</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>22135</th>\n",
       "      <td>short_term_deposit</td>\n",
       "      <td>Inversión</td>\n",
       "      <td>2018-04-08</td>\n",
       "      <td>KHK</td>\n",
       "      <td>100296</td>\n",
       "      <td>ES</td>\n",
       "      <td>V</td>\n",
       "      <td>N</td>\n",
       "      <td>2018-05-28</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>22136</th>\n",
       "      <td>short_term_deposit</td>\n",
       "      <td>Inversión</td>\n",
       "      <td>2018-04-08</td>\n",
       "      <td>KHK</td>\n",
       "      <td>100296</td>\n",
       "      <td>ES</td>\n",
       "      <td>V</td>\n",
       "      <td>N</td>\n",
       "      <td>2018-06-28</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>27826</th>\n",
       "      <td>short_term_deposit</td>\n",
       "      <td>Inversión</td>\n",
       "      <td>2015-01-30</td>\n",
       "      <td>KHM</td>\n",
       "      <td>1003705</td>\n",
       "      <td>ES</td>\n",
       "      <td>H</td>\n",
       "      <td>N</td>\n",
       "      <td>2018-09-28</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                Productos Familia_prod entry_date entry_channel   pk_cid  \\\n",
       "22134  short_term_deposit    Inversión 2018-04-08           KHK   100296   \n",
       "22135  short_term_deposit    Inversión 2018-04-08           KHK   100296   \n",
       "22136  short_term_deposit    Inversión 2018-04-08           KHK   100296   \n",
       "27826  short_term_deposit    Inversión 2015-01-30           KHM  1003705   \n",
       "\n",
       "      country_id gender deceased       date  \n",
       "22134         ES      V        N 2018-04-28  \n",
       "22135         ES      V        N 2018-05-28  \n",
       "22136         ES      V        N 2018-06-28  \n",
       "27826         ES      H        N 2018-09-28  "
      ]
     },
     "execution_count": 9,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "'''categóricas = full_3.select_dtypes(exclude=np.number)\n",
    "categóricas.head(4)'''"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {
    "azdata_cell_guid": "612ec501-3763-4aa2-bafc-e5b0bb19c8b9"
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Ventas_cant</th>\n",
       "      <th>Precios</th>\n",
       "      <th>Ingresos</th>\n",
       "      <th>días_encartera</th>\n",
       "      <th>region_code</th>\n",
       "      <th>age</th>\n",
       "      <th>Month</th>\n",
       "      <th>days_between</th>\n",
       "      <th>recurrencia</th>\n",
       "      <th>Year</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>22134</th>\n",
       "      <td>1.0</td>\n",
       "      <td>40</td>\n",
       "      <td>40.0</td>\n",
       "      <td>20.0</td>\n",
       "      <td>28.0</td>\n",
       "      <td>46</td>\n",
       "      <td>4</td>\n",
       "      <td>0.0</td>\n",
       "      <td>1</td>\n",
       "      <td>2018</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>22135</th>\n",
       "      <td>1.0</td>\n",
       "      <td>40</td>\n",
       "      <td>40.0</td>\n",
       "      <td>50.0</td>\n",
       "      <td>28.0</td>\n",
       "      <td>46</td>\n",
       "      <td>5</td>\n",
       "      <td>30.0</td>\n",
       "      <td>2</td>\n",
       "      <td>2018</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "       Ventas_cant  Precios  Ingresos  días_encartera  region_code  age  \\\n",
       "22134          1.0       40      40.0            20.0         28.0   46   \n",
       "22135          1.0       40      40.0            50.0         28.0   46   \n",
       "\n",
       "       Month  days_between  recurrencia  Year  \n",
       "22134      4           0.0            1  2018  \n",
       "22135      5          30.0            2  2018  "
      ]
     },
     "execution_count": 10,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "'''numéricas = full_3.select_dtypes(include=np.number)\n",
    "numéricas.head(2)'''"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {
    "azdata_cell_guid": "01c3f924-7442-42e9-aec3-a5a2626d6dc2",
    "scrolled": true
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>nulls</th>\n",
       "      <th>unique</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>pk_cid</th>\n",
       "      <td>0</td>\n",
       "      <td>350384</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>entry_date</th>\n",
       "      <td>0</td>\n",
       "      <td>1411</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>entry_channel</th>\n",
       "      <td>0</td>\n",
       "      <td>60</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>country_id</th>\n",
       "      <td>0</td>\n",
       "      <td>37</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>date</th>\n",
       "      <td>0</td>\n",
       "      <td>17</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Productos</th>\n",
       "      <td>0</td>\n",
       "      <td>14</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Familia_prod</th>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>gender</th>\n",
       "      <td>0</td>\n",
       "      <td>2</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>deceased</th>\n",
       "      <td>0</td>\n",
       "      <td>2</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "               nulls  unique\n",
       "pk_cid             0  350384\n",
       "entry_date         0    1411\n",
       "entry_channel      0      60\n",
       "country_id         0      37\n",
       "date               0      17\n",
       "Productos          0      14\n",
       "Familia_prod       0       3\n",
       "gender             0       2\n",
       "deceased           0       2"
      ]
     },
     "execution_count": 11,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "'''unique_cat = categóricas.nunique()\n",
    "nulls_cat = categóricas.isnull().sum()\n",
    "nan_cat_df = pd.DataFrame({#'column': sample.columns,\n",
    "                        'nulls': nulls_cat,\n",
    "                        'unique': unique_cat\n",
    "})\n",
    "nan_cat_df.sort_values('unique', inplace=True, ascending=False)\n",
    "nan_cat_df.to_csv('nan_cat_df.csv')\n",
    "nan_cat_df'''"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {
    "azdata_cell_guid": "8da39fde-462a-48a9-adda-433b57fc959f"
   },
   "outputs": [],
   "source": [
    "'''def plot_cat_values(dataframe, column):\n",
    "\n",
    "    plt.figure(figsize=(15,8))\n",
    "\n",
    "    ax1 = plt.subplot(2,1,1) #una imagen con dos plots por fila, 1 plot por columna, el ax1 es el primer plot\n",
    "    #grafico 1 count\n",
    "    ax1 = sns.countplot(\n",
    "         dataframe[column],\n",
    "         order = list(dataframe[column].unique())\n",
    "\n",
    "        )\n",
    "\n",
    "   #grafico 3 leyenda \n",
    "\n",
    "    ax = sns.countplot(x=column, hue=\"pk_cid\", data=dataframe)\n",
    "    ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha=\"right\")\n",
    "    plt.tight_layout()\n",
    "    plt.show()'''"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {
    "azdata_cell_guid": "9b90f1de-7548-487a-91a1-65d8081321c7"
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "AhorroVista    5806772\n",
       "Inversión       376088\n",
       "Crédito          71658\n",
       "Name: Familia_prod, dtype: int64"
      ]
     },
     "execution_count": 13,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "'''full_3['Familia_prod'].value_counts()'''"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {
    "azdata_cell_guid": "1c917354-d6dc-42c1-9bbe-5b154ce2737d"
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Productos</th>\n",
       "      <th>Familia_prod</th>\n",
       "      <th>entry_date</th>\n",
       "      <th>entry_channel</th>\n",
       "      <th>pk_cid</th>\n",
       "      <th>country_id</th>\n",
       "      <th>gender</th>\n",
       "      <th>deceased</th>\n",
       "      <th>date</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>22134</th>\n",
       "      <td>short_term_deposit</td>\n",
       "      <td>Inversión</td>\n",
       "      <td>2018-04-08</td>\n",
       "      <td>KHK</td>\n",
       "      <td>100296</td>\n",
       "      <td>ES</td>\n",
       "      <td>V</td>\n",
       "      <td>N</td>\n",
       "      <td>2018-04-28</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>22135</th>\n",
       "      <td>short_term_deposit</td>\n",
       "      <td>Inversión</td>\n",
       "      <td>2018-04-08</td>\n",
       "      <td>KHK</td>\n",
       "      <td>100296</td>\n",
       "      <td>ES</td>\n",
       "      <td>V</td>\n",
       "      <td>N</td>\n",
       "      <td>2018-05-28</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>22136</th>\n",
       "      <td>short_term_deposit</td>\n",
       "      <td>Inversión</td>\n",
       "      <td>2018-04-08</td>\n",
       "      <td>KHK</td>\n",
       "      <td>100296</td>\n",
       "      <td>ES</td>\n",
       "      <td>V</td>\n",
       "      <td>N</td>\n",
       "      <td>2018-06-28</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>27826</th>\n",
       "      <td>short_term_deposit</td>\n",
       "      <td>Inversión</td>\n",
       "      <td>2015-01-30</td>\n",
       "      <td>KHM</td>\n",
       "      <td>1003705</td>\n",
       "      <td>ES</td>\n",
       "      <td>H</td>\n",
       "      <td>N</td>\n",
       "      <td>2018-09-28</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                Productos Familia_prod entry_date entry_channel   pk_cid  \\\n",
       "22134  short_term_deposit    Inversión 2018-04-08           KHK   100296   \n",
       "22135  short_term_deposit    Inversión 2018-04-08           KHK   100296   \n",
       "22136  short_term_deposit    Inversión 2018-04-08           KHK   100296   \n",
       "27826  short_term_deposit    Inversión 2015-01-30           KHM  1003705   \n",
       "\n",
       "      country_id gender deceased       date  \n",
       "22134         ES      V        N 2018-04-28  \n",
       "22135         ES      V        N 2018-05-28  \n",
       "22136         ES      V        N 2018-06-28  \n",
       "27826         ES      H        N 2018-09-28  "
      ]
     },
     "execution_count": 14,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "'''categóricas = full_3.select_dtypes(exclude=np.number)\n",
    "categóricas.head(4)'''"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {
    "azdata_cell_guid": "1aeffbba-b7b2-42a1-a636-31dcab6a8fb3"
   },
   "outputs": [],
   "source": [
    "'''#Agrupo las variables con menos de 61 observaciones únicas\n",
    "menos_61 = []\n",
    "mas_61 = []\n",
    "\n",
    "for i in categóricas:\n",
    "      \n",
    "        if full_3[i].nunique() < 61:\n",
    "            menos_61.append(i)\n",
    "            \n",
    "        else:\n",
    "            mas_61.append(i)'''"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {
    "azdata_cell_guid": "2730be9d-6657-4e9d-808b-26eec38d75be"
   },
   "outputs": [],
   "source": [
    "import plotly.express as px"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "metadata": {
    "azdata_cell_guid": "2e3a75dc-59c5-4dc9-9414-c7eada08b0cf"
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "'grafico_horizontal= full_3.groupby([\"recurrencia\",\"date\"])[\"Ingresos\"].sum().reset_index()\\ngrafico_horizontal\\n\\nevolucion_horizontal= px.bar(grafico_horizontal, x=\"date\", y=\"recurrencia\",                         color=\"recurrencia\", orientation=\"v\", color_discrete_sequence=px.colors.cyclical.mygbm)\\n\\nevolucion_horizontal.show()'"
      ]
     },
     "execution_count": 17,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "'''grafico_horizontal= full_3.groupby([\"recurrencia\",\"date\"])[\"Ingresos\"].sum().reset_index()\n",
    "grafico_horizontal\n",
    "\n",
    "evolucion_horizontal= px.bar(grafico_horizontal, x=\"date\", y=\"recurrencia\", \\\n",
    "                        color=\"recurrencia\", orientation=\"v\", color_discrete_sequence=px.colors.cyclical.mygbm)\n",
    "\n",
    "evolucion_horizontal.show()'''"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "metadata": {
    "azdata_cell_guid": "7ea44914-18a2-454d-a8ab-e9421f319103"
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "'grafico_horizontal= full_3.groupby([\\'Familia_prod\\',\"date\"])[\"Ingresos\"].sum().reset_index()\\ngrafico_horizontal\\n\\nevolucion_horizontal= px.bar(grafico_horizontal, x=\"date\", y=\"Ingresos\",                         color=\"Familia_prod\", orientation=\"v\", color_discrete_sequence=px.colors.cyclical.mygbm)\\n\\nevolucion_horizontal.show()'"
      ]
     },
     "execution_count": 18,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "'''grafico_horizontal= full_3.groupby(['Familia_prod',\"date\"])[\"Ingresos\"].sum().reset_index()\n",
    "grafico_horizontal\n",
    "\n",
    "evolucion_horizontal= px.bar(grafico_horizontal, x=\"date\", y=\"Ingresos\", \\\n",
    "                        color=\"Familia_prod\", orientation=\"v\", color_discrete_sequence=px.colors.cyclical.mygbm)\n",
    "\n",
    "evolucion_horizontal.show()'''"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "metadata": {
    "azdata_cell_guid": "c85e4ccb-6b7c-4b66-a876-a14509606889"
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "'grafico_horizontal= full_3.groupby([\\'entry_channel\\',\"date\"])[\"Ingresos\"].mean().reset_index()\\ngrafico_horizontal\\n\\nevolucion_horizontal= px.bar(grafico_horizontal, x=\"date\", y=\"Ingresos\",                         color=\"entry_channel\", orientation=\"v\", color_discrete_sequence=px.colors.cyclical.mygbm)\\n\\nevolucion_horizontal.show()'"
      ]
     },
     "execution_count": 19,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "'''grafico_horizontal= full_3.groupby(['entry_channel',\"date\"])[\"Ingresos\"].mean().reset_index()\n",
    "grafico_horizontal\n",
    "\n",
    "evolucion_horizontal= px.bar(grafico_horizontal, x=\"date\", y=\"Ingresos\", \\\n",
    "                        color=\"entry_channel\", orientation=\"v\", color_discrete_sequence=px.colors.cyclical.mygbm)\n",
    "\n",
    "evolucion_horizontal.show()'''"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "metadata": {
    "azdata_cell_guid": "1f009253-b07d-46f4-b7b6-498e8869ca98"
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "'grafico_horizontal= full_3.groupby([\\'gender\\',\"date\"])[\"Ingresos\"].sum().reset_index()\\ngrafico_horizontal\\n\\nevolucion_horizontal= px.bar(grafico_horizontal, x=\"date\", y=\"Ingresos\",                         color=\\'gender\\', orientation=\"v\", color_discrete_sequence=px.colors.cyclical.mygbm)\\n\\nevolucion_horizontal.show()'"
      ]
     },
     "execution_count": 20,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "\n",
    "'''grafico_horizontal= full_3.groupby(['gender',\"date\"])[\"Ingresos\"].sum().reset_index()\n",
    "grafico_horizontal\n",
    "\n",
    "evolucion_horizontal= px.bar(grafico_horizontal, x=\"date\", y=\"Ingresos\", \\\n",
    "                        color='gender', orientation=\"v\", color_discrete_sequence=px.colors.cyclical.mygbm)\n",
    "\n",
    "evolucion_horizontal.show()'''"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "metadata": {
    "azdata_cell_guid": "07b0ea3f-9441-4c13-b9ba-7715c9c6ddcd"
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "'full_3.info(verbose=True)'"
      ]
     },
     "execution_count": 21,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "'''full_3.info(verbose=True)'''"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "metadata": {
    "azdata_cell_guid": "a4349776-08f2-445a-9a93-14f7a8265951"
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "\"\\nfullinv=full_3[full_3['Familia_prod']=='Inversión']\\nfullahorro=full_3[full_3['Familia_prod']=='AhorroVista']\\nfullcredit=full_3[full_3['Familia_prod']=='Crédito']\""
      ]
     },
     "execution_count": 22,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "'''\n",
    "fullinv=full_3[full_3['Familia_prod']=='Inversión']\n",
    "fullahorro=full_3[full_3['Familia_prod']=='AhorroVista']\n",
    "fullcredit=full_3[full_3['Familia_prod']=='Crédito']'''"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "metadata": {
    "azdata_cell_guid": "00602340-d449-4f11-88de-35a72bd3ed73"
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "'grafico_horizontal= fullinv.groupby([\\'recurrencia\\',\"date\"])[\"Ingresos\"].count().reset_index()\\ngrafico_horizontal\\n\\nevolucion_horizontal= px.bar(grafico_horizontal, x=\"date\", y=\"Ingresos\",                         color=\\'recurrencia\\', orientation=\"v\", color_discrete_sequence=px.colors.cyclical.mygbm)\\n\\nevolucion_horizontal.show()'"
      ]
     },
     "execution_count": 23,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "'''grafico_horizontal= fullinv.groupby(['recurrencia',\"date\"])[\"Ingresos\"].count().reset_index()\n",
    "grafico_horizontal\n",
    "\n",
    "evolucion_horizontal= px.bar(grafico_horizontal, x=\"date\", y=\"Ingresos\", \\\n",
    "                        color='recurrencia', orientation=\"v\", color_discrete_sequence=px.colors.cyclical.mygbm)\n",
    "\n",
    "evolucion_horizontal.show()'''"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "metadata": {
    "azdata_cell_guid": "46bf6517-6c0c-4049-9ff9-1a7e2c96d3f1"
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "['Productos',\n",
       " 'Familia_prod',\n",
       " 'entry_channel',\n",
       " 'country_id',\n",
       " 'gender',\n",
       " 'deceased']"
      ]
     },
     "execution_count": 24,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "'''menos_61[0:6]'''"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "azdata_cell_guid": "d72bd3be-2147-4f55-b1b4-2630f9bbc087"
   },
   "source": [
    "##  OHE"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "metadata": {
    "azdata_cell_guid": "838b5310-5968-41b3-8ecf-ae217fde7376"
   },
   "outputs": [],
   "source": [
    "'''for i in menos_61[0:6]:\n",
    "   _dummy_dataset = pd.get_dummies(full_3[i], prefix=i)\n",
    "   full_3 = pd.concat([full_3,_dummy_dataset],axis=1)\n",
    "   full_3.drop([i],axis=1, inplace=True)'''"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "metadata": {
    "azdata_cell_guid": "cd00871c-d71d-4667-be7a-d0f4174bce0c",
    "scrolled": true
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(6254518, 131)"
      ]
     },
     "execution_count": 26,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "'''full_3.shape'''"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "metadata": {
    "azdata_cell_guid": "5cda1fbf-6c98-4c46-9bb1-a78b485d44dc"
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Ventas_cant  Precios  Ingresos  entry_date  días_encartera  pk_cid   region_code  age  date        Month  days_between  recurrencia  Year  Productos_credit_card  Productos_debit_card  Productos_em_account_p  Productos_em_acount  Productos_emc_account  Productos_funds  Productos_loans  Productos_long_term_deposit  Productos_mortgage  Productos_payroll  Productos_payroll_account  Productos_pension_plan  Productos_securities  Productos_short_term_deposit  Familia_prod_AhorroVista  Familia_prod_Crédito  Familia_prod_Inversión  entry_channel_  entry_channel_004  entry_channel_007  entry_channel_013  entry_channel_KAA  entry_channel_KAB  entry_channel_KAD  entry_channel_KAE  entry_channel_KAF  entry_channel_KAG  entry_channel_KAH  entry_channel_KAJ  entry_channel_KAK  entry_channel_KAM  entry_channel_KAQ  entry_channel_KAR  entry_channel_KAS  entry_channel_KAT  entry_channel_KAW  entry_channel_KAY  entry_channel_KAZ  entry_channel_KBE  entry_channel_KBG  entry_channel_KBH  entry_channel_KBO  entry_channel_KBW  entry_channel_KBZ  entry_channel_KCB  entry_channel_KCC  entry_channel_KCH  entry_channel_KCI  entry_channel_KCK  entry_channel_KCL  entry_channel_KDA  entry_channel_KDH  entry_channel_KDR  entry_channel_KDT  entry_channel_KEH  entry_channel_KEY  entry_channel_KFA  entry_channel_KFC  entry_channel_KFD  entry_channel_KFK  entry_channel_KFL  entry_channel_KFS  entry_channel_KGC  entry_channel_KGN  entry_channel_KGX  entry_channel_KHC  entry_channel_KHD  entry_channel_KHE  entry_channel_KHF  entry_channel_KHK  entry_channel_KHL  entry_channel_KHM  entry_channel_KHN  entry_channel_KHO  entry_channel_KHP  entry_channel_KHQ  entry_channel_RED  country_id_AR  country_id_AT  country_id_BE  country_id_BR  country_id_CA  country_id_CH  country_id_CI  country_id_CL  country_id_CM  country_id_CN  country_id_CO  country_id_DE  country_id_DO  country_id_DZ  country_id_ES  country_id_ET  country_id_FR  country_id_GA  country_id_GB  country_id_GT  country_id_IE  country_id_IT  country_id_LU  country_id_MA  country_id_MR  country_id_MX  country_id_NO  country_id_PE  country_id_PL  country_id_QA  country_id_RO  country_id_RU  country_id_SA  country_id_SE  country_id_SN  country_id_US  country_id_VE  gender_H  gender_V  deceased_N  deceased_S\n",
       "1.0          60       60.0      2019-05-11  17.0            1390124  10.0         49   2019-05-28  5      0.0           1            2019  1                      0                     0                       0                    0                      0                0                0                            0                   0                  0                          0                       0                     0                             0                         1                     0                       0               0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  1                  0                  0                  0                  0                  0                  0                  0                  0              0              0              0              0              0              0              0              0              0              0              0              0              0              1              0              0              0              0              0              0              0              0              0              0              0              0              0              0              0              0              0              0              0              0              0              0              0         1         1           0             1\n",
       "             10       10.0      2016-08-07  690.0           1158780  46.0         24   2018-06-28  6      31.0          6            2018  0                      0                     0                       1                    0                      0                0                0                            0                   0                  0                          0                       0                     0                             1                         0                     0                       0               0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  1                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0              0              0              0              0              0              0              0              0              0              0              0              0              0              1              0              0              0              0              0              0              0              0              0              0              0              0              0              0              0              0              0              0              0              0              0              0              1         0         1           0             1\n",
       "                                                            1158768  47.0         21   2018-06-28  6      31.0          6            2018  0                      0                     0                       1                    0                      0                0                0                            0                   0                  0                          0                       0                     0                             1                         0                     0                       0               0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  1                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0              0              0              0              0              0              0              0              0              0              0              0              0              0              1              0              0              0              0              0              0              0              0              0              0              0              0              0              0              0              0              0              0              0              0              0              0              1         0         1           0             1\n",
       "                                                            1158770  46.0         24   2018-06-28  6      31.0          6            2018  0                      0                     0                       1                    0                      0                0                0                            0                   0                  0                          0                       0                     0                             1                         0                     0                       0               0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  1                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0              0              0              0              0              0              0              0              0              0              0              0              0              0              1              0              0              0              0              0              0              0              0              0              0              0              0              0              0              0              0              0              0              0              0              0              0              1         0         1           0             1\n",
       "                                                            1158771  25.0         27   2018-06-28  6      31.0          6            2018  0                      0                     0                       1                    0                      0                0                0                            0                   0                  0                          0                       0                     0                             1                         0                     0                       0               0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  1                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0              0              0              0              0              0              0              0              0              0              0              0              0              0              1              0              0              0              0              0              0              0              0              0              0              0              0              0              0              0              0              0              0              0              0              0              0              1         0         1           0             1\n",
       "                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          ..\n",
       "                                2017-10-03  148.0           1321856  33.0         21   2018-02-28  2      31.0          2            2018  0                      0                     0                       1                    0                      0                0                0                            0                   0                  0                          0                       0                     0                             1                         0                     0                       0               0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  1                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0              0              0              0              0              0              0              0              0              0              0              0              0              0              1              0              0              0              0              0              0              0              0              0              0              0              0              0              0              0              0              0              0              0              0              0              0              1         0         1           0             1\n",
       "                                                            1321857  47.0         22   2018-02-28  2      31.0          2            2018  0                      0                     0                       1                    0                      0                0                0                            0                   0                  0                          0                       0                     0                             1                         0                     0                       0               0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  1                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0              0              0              0              0              0              0              0              0              0              0              0              0              0              1              0              0              0              0              0              0              0              0              0              0              0              0              0              0              0              0              0              0              0              0              0              0              0         1         1           0             1\n",
       "                                                            1321858  10.0         21   2018-02-28  2      31.0          2            2018  0                      0                     0                       1                    0                      0                0                0                            0                   0                  0                          0                       0                     0                             1                         0                     0                       0               0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  1                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0              0              0              0              0              0              0              0              0              0              0              0              0              0              1              0              0              0              0              0              0              0              0              0              0              0              0              0              0              0              0              0              0              0              0              0              0              1         0         1           0             1\n",
       "                                                            1321859  16.0         26   2018-02-28  2      31.0          2            2018  0                      0                     0                       1                    0                      0                0                0                            0                   0                  0                          0                       0                     0                             1                         0                     0                       0               0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  1                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0              0              0              0              0              0              0              0              0              0              0              0              0              0              1              0              0              0              0              0              0              0              0              0              0              0              0              0              0              0              0              0              0              0              0              0              0              0         1         1           0             1\n",
       "                                2015-01-02  1122.0          1000385  29.0         23   2018-01-28  1      0.0           1            2018  0                      0                     0                       1                    0                      0                0                0                            0                   0                  0                          0                       0                     0                             1                         0                     0                       0               0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0                  1                  0                  0                  0                  0                  0                  0                  0                  0                  0                  0              0              0              0              0              0              0              0              0              0              0              0              0              0              1              0              0              0              0              0              0              0              0              0              0              0              0              0              0              0              0              0              0              0              0              0              0              0         1         1           0             1\n",
       "Length: 6254518, dtype: int64"
      ]
     },
     "execution_count": 28,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "'''full_3.value_counts().T'''"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 138,
   "metadata": {
    "azdata_cell_guid": "645aa984-054b-4586-9984-38fe364d27f6"
   },
   "outputs": [],
   "source": [
    " #full_3.info(verbose=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "metadata": {
    "azdata_cell_guid": "863e7d85-ad74-4944-b48d-dd7946038ac0"
   },
   "outputs": [],
   "source": [
    "'''full_3.drop(['date','entry_date'],axis=1,inplace=True)'''"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "metadata": {
    "azdata_cell_guid": "ea3e5a45-2fb0-4881-ae61-8d80a5e07d18"
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "22134       0\n",
       "22135       0\n",
       "22136       0\n",
       "27826       0\n",
       "27827       0\n",
       "           ..\n",
       "89443855    0\n",
       "89443856    0\n",
       "89443857    0\n",
       "89443858    0\n",
       "89443859    0\n",
       "Name: Familia_prod_Crédito, Length: 6254518, dtype: uint8"
      ]
     },
     "execution_count": 31,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "'''full_3['Familia_prod_Crédito']'''"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 140,
   "metadata": {
    "azdata_cell_guid": "229797f1-e86b-4a33-8b83-139571551ac9"
   },
   "outputs": [],
   "source": [
    "#type('days_between')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 141,
   "metadata": {
    "azdata_cell_guid": "547a0fe8-9ac7-40f6-a083-f9185af7b526"
   },
   "outputs": [],
   "source": [
    "#(full_3.corr()).style.background_gradient(cmap=\"coolwarm\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 32,
   "metadata": {
    "azdata_cell_guid": "d823b65b-ad06-4f7e-bf4b-a1cd4591af7b"
   },
   "outputs": [],
   "source": [
    "'''full_3OHE=full_3\n",
    "pd.to_pickle(full_3OHE, './full_3OHE.pickle')'''"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {
    "azdata_cell_guid": "9ab31fa1-fdd6-49a1-a3de-db38bc3778f8"
   },
   "outputs": [],
   "source": [
    "'''full_4 = pd.read_pickle('./full_4.pickle')'''"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 145,
   "metadata": {
    "azdata_cell_guid": "7ca34afa-59f3-4746-8d59-8c7904e8b8ed"
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "\"confussion_matrix = pd.crosstab(full_4['recurrencia'], full_4['Ingresos'])\\nconfussion_matrix.head(15)\""
      ]
     },
     "execution_count": 145,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "'''confussion_matrix = pd.crosstab(full_4['recurrencia'], full_4['Ingresos'])\n",
    "confussion_matrix.head(15)'''"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 146,
   "metadata": {
    "azdata_cell_guid": "124f776f-4502-4722-b138-f4fb8e84db04"
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "\"confussion_matrix_1 = pd.crosstab(full_4['recurrencia'], full_4['Familia_prod_AhorroVista'].value_counts)\\nconfussion_matrix_1.head(15)\""
      ]
     },
     "execution_count": 146,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "'''confussion_matrix_1 = pd.crosstab(full_4['recurrencia'], full_4['Familia_prod_AhorroVista'].value_counts)\n",
    "confussion_matrix_1.head(15)'''"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 147,
   "metadata": {
    "azdata_cell_guid": "4c085e59-da03-4de4-ac6e-31b4b84d81e8"
   },
   "outputs": [],
   "source": [
    "'''#confussion_matrix_inversion = pd.crosstab(full_4['recurrencia'], full_4['Familia_prod_Inversión'].value_counts)\n",
    "#confussion_matrix_inversion.head(15)'''"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 148,
   "metadata": {
    "azdata_cell_guid": "4f819664-5a78-4be9-b025-855c1ea1c1e5"
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "\"full_kmeans = full_4.groupby('pk_cid').agg(\\n    Ingresos = ('Ingresos', 'sum'), \\n    Compras = ('pk_cid', 'count'),\\n    Age = ('age', 'mean'),\\n    días_encartera=('días_encartera','max'),\\n    days_between = ('days_between','mean'),               \\n    Productos_credit_card = ('Productos_credit_card' ,'sum'),     \\n    Productos_debit_card = ('Productos_debit_card','sum'),                \\n    Productos_em_account_p = ('Productos_em_account_p','sum'),                 \\n    Productos_em_acount = ('Productos_em_acount','sum'),             \\n    Productos_emc_account=('Productos_emc_account','sum'),                  \\n    Productos_funds=('Productos_funds','sum'),                       \\n    Productos_loans= ('Productos_loans','sum'),                           \\n    Productos_long_term_deposit= ('Productos_long_term_deposit','sum'),           \\n    Productos_mortgage= ('Productos_mortgage','sum'),                     \\n    Productos_payroll_account = ('Productos_payroll_account','sum'),             \\n    Productos_pension_plan   = ('Productos_pension_plan','sum'),              \\n    Productos_securities   = ('Productos_securities','sum'),                \\n    Productos_short_term_deposit   = ('Productos_short_term_deposit','sum'),        \\n    Familia_prod_AhorroVista = ('Familia_prod_AhorroVista','sum'),              \\n    Familia_prod_Crédito    = ('Familia_prod_Crédito','sum'),               \\n    Familia_prod_Inversión  = ('Familia_prod_Inversión','sum'),      \\n)\""
      ]
     },
     "execution_count": 148,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "'''full_kmeans = full_4.groupby('pk_cid').agg(\n",
    "    Ingresos = ('Ingresos', 'sum'), \n",
    "    Compras = ('pk_cid', 'count'),\n",
    "    Age = ('age', 'mean'),\n",
    "    días_encartera=('días_encartera','max'),\n",
    "    days_between = ('days_between','mean'),               \n",
    "    Productos_credit_card = ('Productos_credit_card' ,'sum'),     \n",
    "    Productos_debit_card = ('Productos_debit_card','sum'),                \n",
    "    Productos_em_account_p = ('Productos_em_account_p','sum'),                 \n",
    "    Productos_em_acount = ('Productos_em_acount','sum'),             \n",
    "    Productos_emc_account=('Productos_emc_account','sum'),                  \n",
    "    Productos_funds=('Productos_funds','sum'),                       \n",
    "    Productos_loans= ('Productos_loans','sum'),                           \n",
    "    Productos_long_term_deposit= ('Productos_long_term_deposit','sum'),           \n",
    "    Productos_mortgage= ('Productos_mortgage','sum'),                     \n",
    "    Productos_payroll_account = ('Productos_payroll_account','sum'),             \n",
    "    Productos_pension_plan   = ('Productos_pension_plan','sum'),              \n",
    "    Productos_securities   = ('Productos_securities','sum'),                \n",
    "    Productos_short_term_deposit   = ('Productos_short_term_deposit','sum'),        \n",
    "    Familia_prod_AhorroVista = ('Familia_prod_AhorroVista','sum'),              \n",
    "    Familia_prod_Crédito    = ('Familia_prod_Crédito','sum'),               \n",
    "    Familia_prod_Inversión  = ('Familia_prod_Inversión','sum'),      \n",
    ")'''"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 149,
   "metadata": {
    "azdata_cell_guid": "9273d8fc-d764-43b6-bc56-0de85d3f6124"
   },
   "outputs": [],
   "source": [
    "#full_kmeans.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 150,
   "metadata": {
    "azdata_cell_guid": "647d8899-95bc-450e-9ec7-f9ba36e2a27e"
   },
   "outputs": [],
   "source": [
    "#pd.to_pickle(full_kmeans, './full_kmeans.pickle')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 151,
   "metadata": {
    "azdata_cell_guid": "009caae1-fc78-41ec-917b-f5a4e67ee6af"
   },
   "outputs": [],
   "source": [
    "#full_4.set_index(\"pk_cid\", inplace = True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {},
   "outputs": [],
   "source": [
    "t = 'Productos_pension_plan'"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "metadata": {},
   "outputs": [],
   "source": [
    "full_3OHE.sort_values(by=[t],ascending=True,inplace=True) "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## DATASET PARA CLASIFICACIÓN BINARIA"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>count</th>\n",
       "      <th>unique</th>\n",
       "      <th>top</th>\n",
       "      <th>freq</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>pk_cid</th>\n",
       "      <td>6254518</td>\n",
       "      <td>350384</td>\n",
       "      <td>1128353</td>\n",
       "      <td>149</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "          count  unique      top freq\n",
       "pk_cid  6254518  350384  1128353  149"
      ]
     },
     "execution_count": 8,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "full_3OHE.select_dtypes(include=['object']).describe().T"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead tr th {\n",
       "        text-align: left;\n",
       "    }\n",
       "\n",
       "    .dataframe thead tr:last-of-type th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr>\n",
       "      <th></th>\n",
       "      <th>len</th>\n",
       "      <th>sum</th>\n",
       "      <th>mean</th>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th></th>\n",
       "      <th>Productos_pension_plan</th>\n",
       "      <th>Productos_pension_plan</th>\n",
       "      <th>Productos_pension_plan</th>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>pk_cid</th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>1387484</th>\n",
       "      <td>13</td>\n",
       "      <td>13</td>\n",
       "      <td>1.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1088564</th>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1247810</th>\n",
       "      <td>16</td>\n",
       "      <td>16</td>\n",
       "      <td>1.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1244964</th>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1410093</th>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>...</th>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1187517</th>\n",
       "      <td>17</td>\n",
       "      <td>0</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1187516</th>\n",
       "      <td>17</td>\n",
       "      <td>0</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1187515</th>\n",
       "      <td>17</td>\n",
       "      <td>0</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1187513</th>\n",
       "      <td>17</td>\n",
       "      <td>0</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>999892</th>\n",
       "      <td>72</td>\n",
       "      <td>0</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>350384 rows × 3 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "                           len                    sum                   mean\n",
       "        Productos_pension_plan Productos_pension_plan Productos_pension_plan\n",
       "pk_cid                                                                      \n",
       "1387484                     13                     13                    1.0\n",
       "1088564                      1                      1                    1.0\n",
       "1247810                     16                     16                    1.0\n",
       "1244964                      1                      1                    1.0\n",
       "1410093                      1                      1                    1.0\n",
       "...                        ...                    ...                    ...\n",
       "1187517                     17                      0                    0.0\n",
       "1187516                     17                      0                    0.0\n",
       "1187515                     17                      0                    0.0\n",
       "1187513                     17                      0                    0.0\n",
       "999892                      72                      0                    0.0\n",
       "\n",
       "[350384 rows x 3 columns]"
      ]
     },
     "execution_count": 9,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "full_3OHE.pivot_table(index='pk_cid', values=t, aggfunc=[len, sum, np.mean]).sort_values(by=[('mean',t)], ascending = False)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Aplicamos un Frequency Label PARA LA ÚNICA VBLE CATEGÓRICA QUE NOS QUEDA"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [],
   "source": [
    "customer_num = pd.DataFrame(full_3OHE['pk_cid'].value_counts(dropna = False))\n",
    "customer_num.columns = ['customer_count']\n",
    "customer_num['pk_cid'] = customer_num.index"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {
    "azdata_cell_guid": "1879dcf3-3eb2-4268-b020-aa2ebaa7c5ca"
   },
   "outputs": [],
   "source": [
    "df = full_3OHE.merge(customer_num, on = 'pk_cid')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 252,
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>count</th>\n",
       "      <th>unique</th>\n",
       "      <th>top</th>\n",
       "      <th>freq</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>pk_cid</th>\n",
       "      <td>6254518</td>\n",
       "      <td>350384</td>\n",
       "      <td>1128353</td>\n",
       "      <td>149</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "          count  unique      top freq\n",
       "pk_cid  6254518  350384  1128353  149"
      ]
     },
     "execution_count": 252,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.select_dtypes(include=['object']).describe().T"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead tr th {\n",
       "        text-align: left;\n",
       "    }\n",
       "\n",
       "    .dataframe thead tr:last-of-type th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th>len</th>\n",
       "      <th>sum</th>\n",
       "      <th>mean</th>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th>Productos_pension_plan</th>\n",
       "      <th>Productos_pension_plan</th>\n",
       "      <th>Productos_pension_plan</th>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Year</th>\n",
       "      <th>Month</th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th rowspan=\"12\" valign=\"top\">2018</th>\n",
       "      <th>1</th>\n",
       "      <td>296613</td>\n",
       "      <td>8835.0</td>\n",
       "      <td>0.029786</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>302458</td>\n",
       "      <td>9496.0</td>\n",
       "      <td>0.031396</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>308879</td>\n",
       "      <td>9882.0</td>\n",
       "      <td>0.031993</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>313499</td>\n",
       "      <td>10560.0</td>\n",
       "      <td>0.033684</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>5</th>\n",
       "      <td>316422</td>\n",
       "      <td>10148.0</td>\n",
       "      <td>0.032071</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>6</th>\n",
       "      <td>322287</td>\n",
       "      <td>11635.0</td>\n",
       "      <td>0.036101</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>7</th>\n",
       "      <td>337160</td>\n",
       "      <td>12779.0</td>\n",
       "      <td>0.037902</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>8</th>\n",
       "      <td>349012</td>\n",
       "      <td>12126.0</td>\n",
       "      <td>0.034744</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>9</th>\n",
       "      <td>369172</td>\n",
       "      <td>12624.0</td>\n",
       "      <td>0.034195</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>10</th>\n",
       "      <td>391335</td>\n",
       "      <td>13223.0</td>\n",
       "      <td>0.033789</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>11</th>\n",
       "      <td>402819</td>\n",
       "      <td>13818.0</td>\n",
       "      <td>0.034303</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>12</th>\n",
       "      <td>412142</td>\n",
       "      <td>15324.0</td>\n",
       "      <td>0.037181</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th rowspan=\"5\" valign=\"top\">2019</th>\n",
       "      <th>1</th>\n",
       "      <td>411289</td>\n",
       "      <td>12538.0</td>\n",
       "      <td>0.030485</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>421415</td>\n",
       "      <td>15265.0</td>\n",
       "      <td>0.036223</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>428653</td>\n",
       "      <td>15976.0</td>\n",
       "      <td>0.037270</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>432665</td>\n",
       "      <td>16220.0</td>\n",
       "      <td>0.037489</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>5</th>\n",
       "      <td>438698</td>\n",
       "      <td>17353.0</td>\n",
       "      <td>0.039556</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                              len                    sum  \\\n",
       "           Productos_pension_plan Productos_pension_plan   \n",
       "Year Month                                                 \n",
       "2018 1                     296613                 8835.0   \n",
       "     2                     302458                 9496.0   \n",
       "     3                     308879                 9882.0   \n",
       "     4                     313499                10560.0   \n",
       "     5                     316422                10148.0   \n",
       "     6                     322287                11635.0   \n",
       "     7                     337160                12779.0   \n",
       "     8                     349012                12126.0   \n",
       "     9                     369172                12624.0   \n",
       "     10                    391335                13223.0   \n",
       "     11                    402819                13818.0   \n",
       "     12                    412142                15324.0   \n",
       "2019 1                     411289                12538.0   \n",
       "     2                     421415                15265.0   \n",
       "     3                     428653                15976.0   \n",
       "     4                     432665                16220.0   \n",
       "     5                     438698                17353.0   \n",
       "\n",
       "                             mean  \n",
       "           Productos_pension_plan  \n",
       "Year Month                         \n",
       "2018 1                   0.029786  \n",
       "     2                   0.031396  \n",
       "     3                   0.031993  \n",
       "     4                   0.033684  \n",
       "     5                   0.032071  \n",
       "     6                   0.036101  \n",
       "     7                   0.037902  \n",
       "     8                   0.034744  \n",
       "     9                   0.034195  \n",
       "     10                  0.033789  \n",
       "     11                  0.034303  \n",
       "     12                  0.037181  \n",
       "2019 1                   0.030485  \n",
       "     2                   0.036223  \n",
       "     3                   0.037270  \n",
       "     4                   0.037489  \n",
       "     5                   0.039556  "
      ]
     },
     "execution_count": 12,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.pivot_table(index=['Year','Month'], values=t, aggfunc=[len, sum, np.mean])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {
    "azdata_cell_guid": "7a3aaab9-1caa-40b2-ad7a-84211c80e879"
   },
   "outputs": [],
   "source": [
    "df.drop(['Ingresos','Ventas_cant','pk_cid','Precios','Familia_prod_AhorroVista','Familia_prod_Inversión','Familia_prod_Crédito'], axis = 1, inplace = True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {},
   "outputs": [],
   "source": [
    "df.drop(['country_id_AR',  'country_id_AT' , 'country_id_BE' , 'country_id_BR'  ,'country_id_CA' , 'country_id_CH' , 'country_id_CI' , 'country_id_CL' , 'country_id_CM' , 'country_id_CN' , 'country_id_CO' , 'country_id_DE' , 'country_id_DO'  ,'country_id_DZ'  ,'country_id_ES' , 'country_id_ET' , 'country_id_FR'  ,'country_id_GA'  ,'country_id_GB' , 'country_id_GT' , 'country_id_IE'  ,'country_id_IT'  ,'country_id_LU'  ,'country_id_MA' , 'country_id_MR' , 'country_id_MX'  ,'country_id_NO' , 'country_id_PE' , 'country_id_PL' , 'country_id_QA',  'country_id_RO' , 'country_id_RU' ,'country_id_SA' , 'country_id_SE',  'country_id_SN' , 'country_id_US' , 'country_id_VE'], axis = 1, inplace = True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 267,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "\"Productos_credit_card = ('Productos_credit_card' ,'sum'),     \\n    Productos_debit_card = ('Productos_debit_card','sum'),                \\n    Productos_em_account_p = ('Productos_em_account_p','sum'),                 \\n    Productos_em_acount = ('Productos_em_acount','sum'),             \\n    Productos_emc_account=('Productos_emc_account','sum'),                  \\n    Productos_funds=('Productos_funds','sum'),                       \\n    Productos_loans= ('Productos_loans','sum'),                           \\n    Productos_long_term_deposit= ('Productos_long_term_deposit','sum'),           \\n    Productos_mortgage= ('Productos_mortgage','sum'),                     \\n    Productos_payroll_account = ('Productos_payroll_account','sum'),             \\n    Productos_pension_plan   = ('Productos_pension_plan','sum'),              \\n    Productos_securities   = ('Productos_securities','sum'), \""
      ]
     },
     "execution_count": 267,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "'''Productos_credit_card = ('Productos_credit_card' ,'sum'),     \n",
    "    Productos_debit_card = ('Productos_debit_card','sum'),                \n",
    "    Productos_em_account_p = ('Productos_em_account_p','sum'),                 \n",
    "    Productos_em_acount = ('Productos_em_acount','sum'),             \n",
    "    Productos_emc_account=('Productos_emc_account','sum'),                  \n",
    "    Productos_funds=('Productos_funds','sum'),                       \n",
    "    Productos_loans= ('Productos_loans','sum'),                           \n",
    "    Productos_long_term_deposit= ('Productos_long_term_deposit','sum'),           \n",
    "    Productos_mortgage= ('Productos_mortgage','sum'),                     \n",
    "    Productos_payroll_account = ('Productos_payroll_account','sum'),             \n",
    "    Productos_pension_plan   = ('Productos_pension_plan','sum'),              \n",
    "    Productos_securities   = ('Productos_securities','sum'), '''"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Regresión"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## TRABAJEMOS LA REGRESIÓN "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [],
   "source": [
    "full_3OHE.drop(['Ingresos','Ventas_cant','pk_cid','Precios','Familia_prod_AhorroVista','Familia_prod_Inversión','Familia_prod_Crédito'], axis = 1, inplace = True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [],
   "source": [
    "full_3OHE.drop(['country_id_AR',  'country_id_AT' , 'country_id_BE' , 'country_id_BR'  ,'country_id_CA' , 'country_id_CH' , 'country_id_CI' , 'country_id_CL' , 'country_id_CM' , 'country_id_CN' , 'country_id_CO' , 'country_id_DE' , 'country_id_DO'  ,'country_id_DZ'  ,'country_id_ES' , 'country_id_ET' , 'country_id_FR'  ,'country_id_GA'  ,'country_id_GB' , 'country_id_GT' , 'country_id_IE'  ,'country_id_IT'  ,'country_id_LU'  ,'country_id_MA' , 'country_id_MR' , 'country_id_MX'  ,'country_id_NO' , 'country_id_PE' , 'country_id_PL' , 'country_id_QA',  'country_id_RO' , 'country_id_RU' ,'country_id_SA' , 'country_id_SE',  'country_id_SN' , 'country_id_US' , 'country_id_VE'], axis = 1, inplace = True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "metadata": {},
   "outputs": [],
   "source": [
    "import xgboost as xgb\r\n",
    "from scipy import stats\r\n",
    "from datetime import datetime\r\n",
    "from sklearn import model_selection"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "metadata": {},
   "outputs": [],
   "source": [
    "df=full_3OHE "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "metadata": {},
   "outputs": [],
   "source": [
    "df_val = df.tail(1250904)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "metadata": {},
   "outputs": [],
   "source": [
    "df_dev = df.head(5003614)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "metadata": {},
   "outputs": [],
   "source": [
    "RANDOM_STATE = 42"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "metadata": {},
   "outputs": [],
   "source": [
    "TARGET = t"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "metadata": {},
   "outputs": [],
   "source": [
    "df_val_X = df_val.drop(TARGET, axis=1)\r\n",
    "df_val_y = df_val[[TARGET]]\r\n",
    "df_dev_X = df_dev.drop(TARGET, axis=1)\r\n",
    "df_dev_y = df_dev[[TARGET]]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "metadata": {},
   "outputs": [],
   "source": [
    "X_train, X_test, y_train, y_test = model_selection.train_test_split(\r\n",
    "    df_dev_X,\r\n",
    "    df_dev_y,\r\n",
    "    random_state = RANDOM_STATE,\r\n",
    "    test_size = 0.3\r\n",
    ")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "metadata": {},
   "outputs": [],
   "source": [
    "#Prueba de algoritmos \r\n",
    "first_model = xgb.XGBRegressor(random_state=42, n_estimators=100, max_depth=4)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Wall time: 1min 14s\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,\n",
       "             colsample_bynode=1, colsample_bytree=1, gamma=0, gpu_id=-1,\n",
       "             importance_type='gain', interaction_constraints='',\n",
       "             learning_rate=0.300000012, max_delta_step=0, max_depth=4,\n",
       "             min_child_weight=1, missing=nan, monotone_constraints='()',\n",
       "             n_estimators=100, n_jobs=12, num_parallel_tree=1, random_state=42,\n",
       "             reg_alpha=0, reg_lambda=1, scale_pos_weight=1, subsample=1,\n",
       "             tree_method='exact', validate_parameters=1, verbosity=None)"
      ]
     },
     "execution_count": 29,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "%%time\r\n",
    "first_model.fit(X_train, y_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "metadata": {},
   "outputs": [],
   "source": [
    "test_predictions = pd.DataFrame(first_model.predict(X_test), columns=['Prediction'], index=X_test.index)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Prediction</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>86056416</th>\n",
       "      <td>1.617256e-16</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>84363034</th>\n",
       "      <td>1.617256e-16</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>84559246</th>\n",
       "      <td>1.617256e-16</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>86318507</th>\n",
       "      <td>1.617256e-16</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>85294304</th>\n",
       "      <td>1.617256e-16</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>85458323</th>\n",
       "      <td>1.617256e-16</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>74818433</th>\n",
       "      <td>1.617256e-16</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>85832139</th>\n",
       "      <td>1.617256e-16</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>86156446</th>\n",
       "      <td>1.617256e-16</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>87584313</th>\n",
       "      <td>1.617256e-16</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "            Prediction\n",
       "86056416  1.617256e-16\n",
       "84363034  1.617256e-16\n",
       "84559246  1.617256e-16\n",
       "86318507  1.617256e-16\n",
       "85294304  1.617256e-16\n",
       "85458323  1.617256e-16\n",
       "74818433  1.617256e-16\n",
       "85832139  1.617256e-16\n",
       "86156446  1.617256e-16\n",
       "87584313  1.617256e-16"
      ]
     },
     "execution_count": 31,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "test_predictions.head(10)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 32,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Productos_pension_plan</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>86056416</th>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>84363034</th>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>84559246</th>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>86318507</th>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>85294304</th>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>85458323</th>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>74818433</th>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>85832139</th>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>86156446</th>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>87584313</th>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "          Productos_pension_plan\n",
       "86056416                       0\n",
       "84363034                       0\n",
       "84559246                       0\n",
       "86318507                       0\n",
       "85294304                       0\n",
       "85458323                       0\n",
       "74818433                       0\n",
       "85832139                       0\n",
       "86156446                       0\n",
       "87584313                       0"
      ]
     },
     "execution_count": 32,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "y_test.head(10)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 33,
   "metadata": {},
   "outputs": [],
   "source": [
    "results_df = y_test.join(test_predictions)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 36,
   "metadata": {},
   "outputs": [],
   "source": [
    "results_df.columns = ['Target', 'Prediction']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 43,
   "metadata": {},
   "outputs": [],
   "source": [
    "results_df['error'] = results_df['Target'] - results_df['Prediction']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 44,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Target</th>\n",
       "      <th>Prediction</th>\n",
       "      <th>error</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>86056416</th>\n",
       "      <td>0</td>\n",
       "      <td>1.617256e-16</td>\n",
       "      <td>-1.617256e-16</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>84363034</th>\n",
       "      <td>0</td>\n",
       "      <td>1.617256e-16</td>\n",
       "      <td>-1.617256e-16</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>84559246</th>\n",
       "      <td>0</td>\n",
       "      <td>1.617256e-16</td>\n",
       "      <td>-1.617256e-16</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>86318507</th>\n",
       "      <td>0</td>\n",
       "      <td>1.617256e-16</td>\n",
       "      <td>-1.617256e-16</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>85294304</th>\n",
       "      <td>0</td>\n",
       "      <td>1.617256e-16</td>\n",
       "      <td>-1.617256e-16</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "          Target    Prediction         error\n",
       "86056416       0  1.617256e-16 -1.617256e-16\n",
       "84363034       0  1.617256e-16 -1.617256e-16\n",
       "84559246       0  1.617256e-16 -1.617256e-16\n",
       "86318507       0  1.617256e-16 -1.617256e-16\n",
       "85294304       0  1.617256e-16 -1.617256e-16"
      ]
     },
     "execution_count": 44,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "results_df.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 45,
   "metadata": {},
   "outputs": [],
   "source": [
    "results_df['squared_error'] = results_df['error'] ** 2"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 46,
   "metadata": {},
   "outputs": [],
   "source": [
    "results_df['rooted_squared_error'] = np.sqrt(results_df['squared_error'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 47,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Target</th>\n",
       "      <th>Prediction</th>\n",
       "      <th>error</th>\n",
       "      <th>squared_error</th>\n",
       "      <th>rooted_squared_error</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>86056416</th>\n",
       "      <td>0</td>\n",
       "      <td>1.617256e-16</td>\n",
       "      <td>-1.617256e-16</td>\n",
       "      <td>2.615516e-32</td>\n",
       "      <td>1.617256e-16</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>60306838</th>\n",
       "      <td>0</td>\n",
       "      <td>1.617256e-16</td>\n",
       "      <td>-1.617256e-16</td>\n",
       "      <td>2.615516e-32</td>\n",
       "      <td>1.617256e-16</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>73952776</th>\n",
       "      <td>0</td>\n",
       "      <td>1.617256e-16</td>\n",
       "      <td>-1.617256e-16</td>\n",
       "      <td>2.615516e-32</td>\n",
       "      <td>1.617256e-16</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>88047596</th>\n",
       "      <td>0</td>\n",
       "      <td>1.617256e-16</td>\n",
       "      <td>-1.617256e-16</td>\n",
       "      <td>2.615516e-32</td>\n",
       "      <td>1.617256e-16</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>41908411</th>\n",
       "      <td>0</td>\n",
       "      <td>1.617256e-16</td>\n",
       "      <td>-1.617256e-16</td>\n",
       "      <td>2.615516e-32</td>\n",
       "      <td>1.617256e-16</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "          Target    Prediction         error  squared_error  \\\n",
       "86056416       0  1.617256e-16 -1.617256e-16   2.615516e-32   \n",
       "60306838       0  1.617256e-16 -1.617256e-16   2.615516e-32   \n",
       "73952776       0  1.617256e-16 -1.617256e-16   2.615516e-32   \n",
       "88047596       0  1.617256e-16 -1.617256e-16   2.615516e-32   \n",
       "41908411       0  1.617256e-16 -1.617256e-16   2.615516e-32   \n",
       "\n",
       "          rooted_squared_error  \n",
       "86056416          1.617256e-16  \n",
       "60306838          1.617256e-16  \n",
       "73952776          1.617256e-16  \n",
       "88047596          1.617256e-16  \n",
       "41908411          1.617256e-16  "
      ]
     },
     "execution_count": 47,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "results_df.sort_values(by='Target', ascending=False).head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 48,
   "metadata": {},
   "outputs": [],
   "source": [
    "mse = results_df['squared_error'].mean()\n",
    "rmse = results_df['rooted_squared_error'].mean()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 49,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "MSE: 2.5663998046742854e-32 - RMSE: 1.638227629741025e-16\n"
     ]
    }
   ],
   "source": [
    "print('MSE: {} - RMSE: {}'.format(mse, rmse))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 50,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<AxesSubplot:xlabel='rooted_squared_error', ylabel='Density'>"
      ]
     },
     "execution_count": 50,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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",
      "text/plain": [
       "<Figure size 1080x360 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.figure(figsize=(15,5))\n",
    "sns.distplot(\n",
    "    results_df['rooted_squared_error']\n",
    ")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 69,
   "metadata": {},
   "outputs": [
    {
     "ename": "KeyError",
     "evalue": "'rooted_squared_error'",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mKeyError\u001b[0m                                  Traceback (most recent call last)",
      "\u001b[1;32m~\\anaconda3\\lib\\site-packages\\pandas\\core\\indexes\\base.py\u001b[0m in \u001b[0;36mget_loc\u001b[1;34m(self, key, method, tolerance)\u001b[0m\n\u001b[0;32m   2894\u001b[0m             \u001b[1;32mtry\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m-> 2895\u001b[1;33m                 \u001b[1;32mreturn\u001b[0m \u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0m_engine\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mget_loc\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mcasted_key\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m   2896\u001b[0m             \u001b[1;32mexcept\u001b[0m \u001b[0mKeyError\u001b[0m \u001b[1;32mas\u001b[0m \u001b[0merr\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;32mpandas\\_libs\\index.pyx\u001b[0m in \u001b[0;36mpandas._libs.index.IndexEngine.get_loc\u001b[1;34m()\u001b[0m\n",
      "\u001b[1;32mpandas\\_libs\\index.pyx\u001b[0m in \u001b[0;36mpandas._libs.index.IndexEngine.get_loc\u001b[1;34m()\u001b[0m\n",
      "\u001b[1;32mpandas\\_libs\\hashtable_class_helper.pxi\u001b[0m in \u001b[0;36mpandas._libs.hashtable.PyObjectHashTable.get_item\u001b[1;34m()\u001b[0m\n",
      "\u001b[1;32mpandas\\_libs\\hashtable_class_helper.pxi\u001b[0m in \u001b[0;36mpandas._libs.hashtable.PyObjectHashTable.get_item\u001b[1;34m()\u001b[0m\n",
      "\u001b[1;31mKeyError\u001b[0m: 'rooted_squared_error'",
      "\nThe above exception was the direct cause of the following exception:\n",
      "\u001b[1;31mKeyError\u001b[0m                                  Traceback (most recent call last)",
      "\u001b[1;32m<ipython-input-69-f8101ba8aa4e>\u001b[0m in \u001b[0;36m<module>\u001b[1;34m\u001b[0m\n\u001b[0;32m      1\u001b[0m \u001b[0mplt\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mfigure\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mfigsize\u001b[0m\u001b[1;33m=\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;36m15\u001b[0m\u001b[1;33m,\u001b[0m\u001b[1;36m5\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m----> 2\u001b[1;33m \u001b[0msns\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mdistplot\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mresults_df\u001b[0m\u001b[1;33m[\u001b[0m\u001b[0mresults_df\u001b[0m\u001b[1;33m[\u001b[0m\u001b[1;34m'Target'\u001b[0m\u001b[1;33m]\u001b[0m \u001b[1;33m>\u001b[0m \u001b[1;36m0\u001b[0m\u001b[1;33m]\u001b[0m\u001b[1;33m[\u001b[0m\u001b[1;34m'rooted_squared_error'\u001b[0m\u001b[1;33m]\u001b[0m\u001b[1;33m,\u001b[0m\u001b[0mfit\u001b[0m\u001b[1;33m=\u001b[0m\u001b[0mstats\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mnorm\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m      3\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;32m~\\anaconda3\\lib\\site-packages\\pandas\\core\\frame.py\u001b[0m in \u001b[0;36m__getitem__\u001b[1;34m(self, key)\u001b[0m\n\u001b[0;32m   2900\u001b[0m             \u001b[1;32mif\u001b[0m \u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mcolumns\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mnlevels\u001b[0m \u001b[1;33m>\u001b[0m \u001b[1;36m1\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m   2901\u001b[0m                 \u001b[1;32mreturn\u001b[0m \u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0m_getitem_multilevel\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mkey\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m-> 2902\u001b[1;33m             \u001b[0mindexer\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mcolumns\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mget_loc\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mkey\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m   2903\u001b[0m             \u001b[1;32mif\u001b[0m \u001b[0mis_integer\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mindexer\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m   2904\u001b[0m                 \u001b[0mindexer\u001b[0m \u001b[1;33m=\u001b[0m \u001b[1;33m[\u001b[0m\u001b[0mindexer\u001b[0m\u001b[1;33m]\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;32m~\\anaconda3\\lib\\site-packages\\pandas\\core\\indexes\\base.py\u001b[0m in \u001b[0;36mget_loc\u001b[1;34m(self, key, method, tolerance)\u001b[0m\n\u001b[0;32m   2895\u001b[0m                 \u001b[1;32mreturn\u001b[0m \u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0m_engine\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mget_loc\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mcasted_key\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m   2896\u001b[0m             \u001b[1;32mexcept\u001b[0m \u001b[0mKeyError\u001b[0m \u001b[1;32mas\u001b[0m \u001b[0merr\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m-> 2897\u001b[1;33m                 \u001b[1;32mraise\u001b[0m \u001b[0mKeyError\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mkey\u001b[0m\u001b[1;33m)\u001b[0m \u001b[1;32mfrom\u001b[0m \u001b[0merr\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m   2898\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m   2899\u001b[0m         \u001b[1;32mif\u001b[0m \u001b[0mtolerance\u001b[0m \u001b[1;32mis\u001b[0m \u001b[1;32mnot\u001b[0m \u001b[1;32mNone\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;31mKeyError\u001b[0m: 'rooted_squared_error'"
     ]
    },
    {
     "data": {
      "text/plain": [
       "<Figure size 1080x360 with 0 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.figure(figsize=(15,5))\r\n",
    "sns.distplot(results_df[results_df['Target'] > 0]['rooted_squared_error'],fit=stats.norm)\r\n",
    "    "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 54,
   "metadata": {},
   "outputs": [
    {
     "ename": "ValueError",
     "evalue": "`dataset` input should have multiple elements.",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mValueError\u001b[0m                                Traceback (most recent call last)",
      "\u001b[1;32m<ipython-input-54-5b87d6e42338>\u001b[0m in \u001b[0;36m<module>\u001b[1;34m\u001b[0m\n\u001b[0;32m      1\u001b[0m \u001b[0mplt\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mfigure\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mfigsize\u001b[0m\u001b[1;33m=\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;36m15\u001b[0m\u001b[1;33m,\u001b[0m\u001b[1;36m5\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m----> 2\u001b[1;33m sns.distplot(\n\u001b[0m\u001b[0;32m      3\u001b[0m     \u001b[0mresults_df\u001b[0m\u001b[1;33m[\u001b[0m\u001b[0mresults_df\u001b[0m\u001b[1;33m[\u001b[0m\u001b[1;34m'Target'\u001b[0m\u001b[1;33m]\u001b[0m \u001b[1;33m>\u001b[0m \u001b[1;36m0\u001b[0m\u001b[1;33m]\u001b[0m\u001b[1;33m[\u001b[0m\u001b[1;34m'rooted_squared_error'\u001b[0m\u001b[1;33m]\u001b[0m\u001b[1;33m,\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m      4\u001b[0m     \u001b[0mfit\u001b[0m\u001b[1;33m=\u001b[0m\u001b[0mstats\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mnorm\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m      5\u001b[0m )\n",
      "\u001b[1;32m~\\anaconda3\\lib\\site-packages\\seaborn\\distributions.py\u001b[0m in \u001b[0;36mdistplot\u001b[1;34m(a, bins, hist, kde, rug, fit, hist_kws, kde_kws, rug_kws, fit_kws, color, vertical, norm_hist, axlabel, label, ax, x)\u001b[0m\n\u001b[0;32m   2637\u001b[0m         \u001b[0mcut\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mfit_kws\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mpop\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;34m\"cut\"\u001b[0m\u001b[1;33m,\u001b[0m \u001b[1;36m3\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m   2638\u001b[0m         \u001b[0mclip\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mfit_kws\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mpop\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;34m\"clip\"\u001b[0m\u001b[1;33m,\u001b[0m \u001b[1;33m(\u001b[0m\u001b[1;33m-\u001b[0m\u001b[0mnp\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0minf\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mnp\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0minf\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m-> 2639\u001b[1;33m         \u001b[0mbw\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mstats\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mgaussian_kde\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0ma\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mscotts_factor\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;33m)\u001b[0m \u001b[1;33m*\u001b[0m \u001b[0ma\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mstd\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mddof\u001b[0m\u001b[1;33m=\u001b[0m\u001b[1;36m1\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m   2640\u001b[0m         \u001b[0mx\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0m_kde_support\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0ma\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mbw\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mgridsize\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mcut\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mclip\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m   2641\u001b[0m         \u001b[0mparams\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mfit\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mfit\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0ma\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;32m~\\anaconda3\\lib\\site-packages\\scipy\\stats\\kde.py\u001b[0m in \u001b[0;36m__init__\u001b[1;34m(self, dataset, bw_method, weights)\u001b[0m\n\u001b[0;32m    191\u001b[0m         \u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mdataset\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0matleast_2d\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0masarray\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mdataset\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    192\u001b[0m         \u001b[1;32mif\u001b[0m \u001b[1;32mnot\u001b[0m \u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mdataset\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0msize\u001b[0m \u001b[1;33m>\u001b[0m \u001b[1;36m1\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m--> 193\u001b[1;33m             \u001b[1;32mraise\u001b[0m \u001b[0mValueError\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;34m\"`dataset` input should have multiple elements.\"\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m    194\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    195\u001b[0m         \u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0md\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mn\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mdataset\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mshape\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;31mValueError\u001b[0m: `dataset` input should have multiple elements."
     ]
    },
    {
     "data": {
      "image/png": "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",
      "text/plain": [
       "<Figure size 1080x360 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.figure(figsize=(15,5))\r\n",
    "sns.distplot(\r\n",
    "    results_df[results_df['Target'] > 0]['rooted_squared_error'],\r\n",
    "    fit=stats.norm\r\n",
    ")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "results_df['squared_error'] = results_df['squared_error'].apply(lambda x: np.log1p(x))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<AxesSubplot:xlabel='squared_error', ylabel='Density'>"
      ]
     },
     "execution_count": 134,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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q6oeBgAAgHfvHQvescce2/d40qRJVQ8DAADAu1fxPXjf+c538uyzzyZJnnnmmVxxxRW58sor88wzz1QrGwAAAPug4oJ3//33Z+zYsUmSb33rWznvvPNy/vnn54477qhWNgAAAPZBxQXvlVdeydChQ7Njx448++yzOeecc/Jbv/Vb2bhxYzXzAQAAUKG9/g7eG0aPHp1169bll7/8ZX7zN38zAwYMyCuvvJIBAyruiAAAAFRRxQVv5syZ+Yd/+IfU19fn6quvTpKsWrUqRx11VNXCAQAAULmKC96JJ56Y22677S3Lpk2blmnTpu33UAAAAOy7igte8uv78DZu3JidO3e+ZflHP/rR/RoKAACAfVdxwXv44YezbNmyDB48OIMGDepbXldXl5tvvrkq4QAAAKhcxQXvW9/6VubOnZsTTjihmnkAAAB4lyr+CsxyuZzjjjuumlkAAAB4DyoueL/3e7+Xb3/72ymXy9XMAwAAwLtU8SWa999/f1588cX8x3/8R4YNG/aWdUuXLt3vwQAAANg3FRe8q666qpo5AAAAeI8qLniTJk2qZg4AAADeo4oL3uuvv5577703jz76aF566aXceeed+Z//+Z8899xz+Z3f+Z1qZgQAAKACFX/Jyp133plf/vKXmT17durq6pIk48ePz/e+972qhQMAAKByFZ/Be/zxx7N48eIMHjy4r+CNGjUqW7durVo4AAAAKlfxGbz6+vrdfiJh+/btGT58+H4PBQAAwL6ruOBNmzYtN998czZt2pQk6erqyrJlyzJ9+vSqhQMAAKByFRe8iy66KGPHjs3VV1+dV155JbNnz06pVMof/dEfVTMfAAAAFar4Hrznn38+48aNy+///u+nXC7npJNOyuGHH17NbAAAAOyDvRa83t7eLF26NO3t7Rk9enRKpVK2bt2ae++9N6effnquuOKKvi9dAQAAoP/steC1tbVl7dq1WbBgQY466qi+5evXr89NN92U//zP/8zZZ59d1ZAAAADs3V7vwVuxYkX+7M/+7C3lLkmOOuqozJo1KytXrqxaOAAAACq314K3YcOGTJo0aY/rJk2alA0bNuz3UAAAAOy7vRa8crmcIUOG7HHdkCFDdvttPAAAAPrHXu/B27VrV5566qm3Xa/gAQAAvD/steCNHDkyS5cufdv1I0aM2K+BAAAAeHf2WvCWLFlSixwA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ExWpZL46OzuzZMmSvPjii6mrq0tzc3POPffcfkrMgWDNmjW5/fbbUy6Xc9ZZZ2XGjBlvWd/b25vbb789q1evzsEHH5xPf/rTmTBhQv+E5YCzt/lauXJl/v3f/z1JMnjw4Fx22WU54ogjah+UA9Le5usN69evz1//9V9nzpw5mTZtWm1DUjXO4FETra2tmTx5chYvXpzJkyentbX1bbd94IEHMm7cuNqF44BXyXwNHDgwF198cVpaWrJgwYI89NBD2bBhQ+3DckAol8tZtmxZrrvuurS0tOTRRx/dbV5Wr16d559/PosXL87ll1+ef/qnf+qntBxoKpmvsWPH5otf/GJuvPHG/MEf/EG+/vWv91NaDjSVzNcb29199905/vjjax+SqlLwqImOjo40NTUlSZqamtLR0bHH7bZs2ZJVq1blrLPOqmU8DnCVzFepVOo7uzJkyJCMGzcuW7durWlODhzr16/PYYcdlkMPPTT19fWZPn36bnP1xBNP5PTTT09dXV2OOeaYvPzyy+nq6uqnxBxIKpmviRMn9l2JcPTRR7/ld4PhnVQyX0ny3e9+NyeffHJGjBjRDympJgWPmti2bVvfj9iXSqVs3759j9vdcccdmTlzZurq6moZjwNcpfP1hk2bNuVnP/tZjjrqqFrE4wC0devWjB49uu/56NGjd/sPga1bt6ahoeEdt4E9qWS+3mz58uU54YQTahGNAqj08+vxxx/P2WefXet41IB78NhvvvSlL+XFF1/cbfknP/nJivb/4Q9/mJEjR2bChAl5+umn93M6DnTvdb7esHPnzixatCizZs3K0KFD91M6imZPXzD9f//jqZJtYE/2ZXaeeuqp/Nd//Vf+9m//ttqxKIhK5uuOO+7IH//xH2fAAOd6ikjBY7/5whe+8LbrRo4cma6urpRKpXR1de3xcoB169bliSeeyOrVq/Paa69lx44dWbx4cWbPnl3N2Bwg3ut8JUlPT08WLVqU0047LSeffHK1olIAo0ePfsslcVu2bOk7S/zmbTo7O99xG9iTSuYrSX7+85/ntttuy+c///kMHz68lhE5gFUyX//7v/+bm266KUmyffv2rF69OgMGDMhJJ51U06xUh9pOTUyZMiXt7e1Jkvb29kydOnW3bS666KLceuutWbJkST772c/mox/9qHJHRSqZr97e3tx6660ZN25czjvv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",
      "text/plain": [
       "<Figure size 1080x360 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.figure(figsize=(15,5))\n",
    "sns.distplot(\n",
    "    results_df[results_df['Target'] > 0]['squared_error'],\n",
    "    fit=stats.norm\n",
    ")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## PRUEBA ALGORITMOS"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "first_model = xgb.XGBRegressor(random_state=42, n_estimators=100, max_depth=4)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Wall time: 3min 40s\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,\n",
       "             colsample_bynode=1, colsample_bytree=1, gamma=0, gpu_id=-1,\n",
       "             importance_type='gain', interaction_constraints='',\n",
       "             learning_rate=0.300000012, max_delta_step=0, max_depth=4,\n",
       "             min_child_weight=1, missing=nan, monotone_constraints='()',\n",
       "             n_estimators=100, n_jobs=12, num_parallel_tree=1, random_state=42,\n",
       "             reg_alpha=0, reg_lambda=1, scale_pos_weight=1, subsample=1,\n",
       "             tree_method='exact', validate_parameters=1, verbosity=None)"
      ]
     },
     "execution_count": 105,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "%%time\n",
    "first_model.fit(X_train, y_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "test_predictions = pd.DataFrame(first_model.predict(X_test), columns=['Prediction'], index=X_test.index)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Prediction</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>48859303</th>\n",
       "      <td>1.307245e-07</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>87401118</th>\n",
       "      <td>1.307245e-07</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>87152798</th>\n",
       "      <td>1.307245e-07</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>29990359</th>\n",
       "      <td>6.931456e-01</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>68266553</th>\n",
       "      <td>1.307245e-07</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>62549693</th>\n",
       "      <td>1.307245e-07</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>85705458</th>\n",
       "      <td>1.307245e-07</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>69058558</th>\n",
       "      <td>1.307245e-07</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>47801467</th>\n",
       "      <td>1.307245e-07</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>84254051</th>\n",
       "      <td>1.307245e-07</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "            Prediction\n",
       "48859303  1.307245e-07\n",
       "87401118  1.307245e-07\n",
       "87152798  1.307245e-07\n",
       "29990359  6.931456e-01\n",
       "68266553  1.307245e-07\n",
       "62549693  1.307245e-07\n",
       "85705458  1.307245e-07\n",
       "69058558  1.307245e-07\n",
       "47801467  1.307245e-07\n",
       "84254051  1.307245e-07"
      ]
     },
     "execution_count": 107,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "test_predictions.head(10)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 55,
   "metadata": {},
   "outputs": [],
   "source": [
    "results_df = y_test.join(test_predictions)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 56,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Productos_pension_plan</th>\n",
       "      <th>Prediction</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>86056416</th>\n",
       "      <td>0</td>\n",
       "      <td>1.617256e-16</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>60306838</th>\n",
       "      <td>0</td>\n",
       "      <td>1.617256e-16</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>73952776</th>\n",
       "      <td>0</td>\n",
       "      <td>1.617256e-16</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>88047596</th>\n",
       "      <td>0</td>\n",
       "      <td>1.617256e-16</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>41908411</th>\n",
       "      <td>0</td>\n",
       "      <td>1.617256e-16</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>88184530</th>\n",
       "      <td>0</td>\n",
       "      <td>1.617256e-16</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>85558959</th>\n",
       "      <td>0</td>\n",
       "      <td>1.617256e-16</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>85619077</th>\n",
       "      <td>0</td>\n",
       "      <td>1.617256e-16</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>87261017</th>\n",
       "      <td>0</td>\n",
       "      <td>1.617256e-16</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>88172261</th>\n",
       "      <td>0</td>\n",
       "      <td>1.617256e-16</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>84511484</th>\n",
       "      <td>0</td>\n",
       "      <td>1.617256e-16</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>62816856</th>\n",
       "      <td>0</td>\n",
       "      <td>1.617256e-16</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>85498536</th>\n",
       "      <td>0</td>\n",
       "      <td>1.617256e-16</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>88555225</th>\n",
       "      <td>0</td>\n",
       "      <td>1.617256e-16</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>85503690</th>\n",
       "      <td>0</td>\n",
       "      <td>1.617256e-16</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>85466711</th>\n",
       "      <td>0</td>\n",
       "      <td>1.617256e-16</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>85900202</th>\n",
       "      <td>0</td>\n",
       "      <td>1.617256e-16</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>85930411</th>\n",
       "      <td>0</td>\n",
       "      <td>1.617256e-16</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>85431746</th>\n",
       "      <td>0</td>\n",
       "      <td>1.617256e-16</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "          Productos_pension_plan    Prediction\n",
       "86056416                       0  1.617256e-16\n",
       "60306838                       0  1.617256e-16\n",
       "73952776                       0  1.617256e-16\n",
       "88047596                       0  1.617256e-16\n",
       "41908411                       0  1.617256e-16\n",
       "88184530                       0  1.617256e-16\n",
       "85558959                       0  1.617256e-16\n",
       "85619077                       0  1.617256e-16\n",
       "87261017                       0  1.617256e-16\n",
       "88172261                       0  1.617256e-16\n",
       "84511484                       0  1.617256e-16\n",
       "62816856                       0  1.617256e-16\n",
       "85498536                       0  1.617256e-16\n",
       "88555225                       0  1.617256e-16\n",
       "85503690                       0  1.617256e-16\n",
       "85466711                       0  1.617256e-16\n",
       "85900202                       0  1.617256e-16\n",
       "85930411                       0  1.617256e-16\n",
       "85431746                       0  1.617256e-16"
      ]
     },
     "execution_count": 56,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "results_df.sort_values(by='Prediction', ascending=False).head(19)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 57,
   "metadata": {},
   "outputs": [],
   "source": [
    "results_df.columns = ['Target', 'Prediction']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 58,
   "metadata": {},
   "outputs": [],
   "source": [
    "results_df['error'] = results_df['Target'] - results_df['Prediction']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 59,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Target</th>\n",
       "      <th>Prediction</th>\n",
       "      <th>error</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>86056416</th>\n",
       "      <td>0</td>\n",
       "      <td>1.617256e-16</td>\n",
       "      <td>-1.617256e-16</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>84363034</th>\n",
       "      <td>0</td>\n",
       "      <td>1.617256e-16</td>\n",
       "      <td>-1.617256e-16</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>84559246</th>\n",
       "      <td>0</td>\n",
       "      <td>1.617256e-16</td>\n",
       "      <td>-1.617256e-16</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>86318507</th>\n",
       "      <td>0</td>\n",
       "      <td>1.617256e-16</td>\n",
       "      <td>-1.617256e-16</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>85294304</th>\n",
       "      <td>0</td>\n",
       "      <td>1.617256e-16</td>\n",
       "      <td>-1.617256e-16</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "          Target    Prediction         error\n",
       "86056416       0  1.617256e-16 -1.617256e-16\n",
       "84363034       0  1.617256e-16 -1.617256e-16\n",
       "84559246       0  1.617256e-16 -1.617256e-16\n",
       "86318507       0  1.617256e-16 -1.617256e-16\n",
       "85294304       0  1.617256e-16 -1.617256e-16"
      ]
     },
     "execution_count": 59,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "results_df.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Cerramos el modelo con el df de validación "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 60,
   "metadata": {},
   "outputs": [],
   "source": [
    "val_predictions = pd.DataFrame(first_model.predict(df_val_X), index=df_val_X.index, columns=['Prediction'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 61,
   "metadata": {},
   "outputs": [],
   "source": [
    "val_results_df = df_val_y.join(val_predictions)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 62,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Productos_pension_plan</th>\n",
       "      <th>Prediction</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>84833800</th>\n",
       "      <td>0</td>\n",
       "      <td>1.617256e-16</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>84833798</th>\n",
       "      <td>0</td>\n",
       "      <td>1.617256e-16</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>84833797</th>\n",
       "      <td>0</td>\n",
       "      <td>1.617256e-16</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>84833796</th>\n",
       "      <td>0</td>\n",
       "      <td>1.617256e-16</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>84833795</th>\n",
       "      <td>0</td>\n",
       "      <td>1.617256e-16</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "          Productos_pension_plan    Prediction\n",
       "84833800                       0  1.617256e-16\n",
       "84833798                       0  1.617256e-16\n",
       "84833797                       0  1.617256e-16\n",
       "84833796                       0  1.617256e-16\n",
       "84833795                       0  1.617256e-16"
      ]
     },
     "execution_count": 62,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "val_results_df.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 63,
   "metadata": {},
   "outputs": [],
   "source": [
    "val_results_df.columns = ['Target', 'Prediction']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 64,
   "metadata": {},
   "outputs": [],
   "source": [
    "val_results_df['error'] = val_results_df['Target'] - val_results_df['Prediction']\n",
    "val_results_df['squared_error'] = val_results_df['error'] ** 2\n",
    "val_results_df['rooted_squared_error'] = np.sqrt(val_results_df['squared_error'])\n",
    "mse = val_results_df['squared_error'].mean()\n",
    "rmse = val_results_df['rooted_squared_error'].mean()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 65,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "MSE: 0.17411567270755768 - RMSE: 0.17411567270755768\n"
     ]
    }
   ],
   "source": [
    "print('MSE: {} - RMSE: {}'.format(mse, rmse))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 66,
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<AxesSubplot:xlabel='Prediction', ylabel='Density'>"
      ]
     },
     "execution_count": 66,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAA3UAAAE9CAYAAACsmksIAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjMuMiwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy8vihELAAAACXBIWXMAAAsTAAALEwEAmpwYAAAVT0lEQVR4nO3df/BldX3f8dc7QOKvqCALUtCs024SGauQWYktTaIiBtEKnSYdSdRtx5RYjaNp2pTYTmra/kHajmM7TWLwx2SNUeMPFLQ0BtdfaeMPFiX+Ql0HrTIy7PqjghkrAd/943s28+36Zb93lz33y+fyeMzsnHvOPffe9+6c2d3n95x7b3V3AAAAGNMPbPUAAAAAHD1RBwAAMDBRBwAAMDBRBwAAMDBRBwAAMDBRBwAAMLDjt3qARZx88sm9ffv2rR4DAABgS1x//fVf6+5tG903RNRt3749e/fu3eoxAAAAtkRV/e+7u8/llwAAAAMTdQAAAAMTdQAAAAMTdQAAAAMTdQAAAAMTdQAAAAMTdQAAAAOb9XvqqupLSW5PcleSO7t7Z1WdlOSPk2xP8qUk/6i7vznnHAAAAKtqGWfqntTdZ3X3zmn9siR7untHkj3TOgAAAEdhKy6/vCjJ7un27iQXb8EMAAAAK2HuqOskf1pV11fVpdO2U7v7liSZlqfMPAMAAMDKmvU9dUnO7e6vVtUpSa6tqs8u+sApAi9Nkkc+8pFzzQfAfcQbPvLlo37sL/ykf4cAuPea9Uxdd391Wu5P8vYk5yS5tapOS5Jpuf9uHntFd+/s7p3btm2bc0wAAIBhzRZ1VfXAqvrhg7eTPDXJp5JcnWTXtNuuJFfNNQMAAMCqm/Pyy1OTvL2qDr7OG7r7T6rquiRvrqrnJflykp+fcQYAAICVNlvUdfdNSR63wfavJzlvrtcFAAC4L9mKrzQAAADgGBF1AAAAAxN1AAAAAxN1AAAAAxN1AAAAAxN1AAAAAxN1AAAAAxN1AAAAAxN1AAAAAxN1AAAAAxN1AAAAAxN1AAAAAxN1AAAAAxN1AAAAAxN1AAAAAxN1AAAAAxN1AAAAAxN1AAAAAxN1AAAAAxN1AAAAAxN1AAAAAxN1AAAAAxN1AAAAAxN1AAAAAxN1AAAAAxN1AAAAAxN1AAAAAxN1AAAAAxN1AAAAAxN1AAAAAxN1AAAAAxN1AAAAAxN1AAAAAxN1AAAAAxN1AAAAAxN1AAAAAxN1AAAAAxN1AAAAAxN1AAAAAxN1AAAAAxN1AAAAAxN1AAAAAxN1AAAAA5s96qrquKr6eFW9a1o/qaqurap90/LEuWcAAABYVcs4U/fiJDeuW78syZ7u3pFkz7QOAADAUZg16qrqjCRPT/LqdZsvSrJ7ur07ycVzzgAAALDK5j5T94okv57ke+u2ndrdtyTJtDxl5hkAAABW1mxRV1XPSLK/u68/ysdfWlV7q2rvgQMHjvF0AAAAq2HOM3XnJnlmVX0pyZuSPLmqXp/k1qo6LUmm5f6NHtzdV3T3zu7euW3bthnHBAAAGNdsUdfdv9HdZ3T39iTPSvLe7n52kquT7Jp225XkqrlmAAAAWHVb8T11lyc5v6r2JTl/WgcAAOAoHL+MF+nu9yd5/3T760nOW8brAgAArLqtOFMHAADAMSLqAAAABibqAAAABibqAAAABibqAAAABibqAAAABibqAAAABibqAAAABibqAAAABibqAAAABibqAAAABibqAAAABibqAAAABibqAAAABibqAAAABibqAAAABibqAAAABibqAAAABibqAAAABibqAAAABibqAAAABibqAAAABibqAAAABibqAAAABibqAAAABibqAAAABibqAAAABibqAAAABibqAAAABibqAAAABibqAAAABibqAAAABibqAAAABibqAAAABibqAAAABibqAAAABibqAAAABibqAAAABibqAAAABibqAAAABibqAAAABibqAAAABibqAAAABjZb1FXV/arqo1X1F1X16ar6rWn7SVV1bVXtm5YnzjUDAADAqpvzTN13kzy5ux+X5KwkF1TVE5JclmRPd+9IsmdaBwAA4CjMFnW95tvT6gnTr05yUZLd0/bdSS6eawYAAIBVN+t76qrquKq6Icn+JNd290eSnNrdtyTJtDzlbh57aVXtraq9Bw4cmHNMAACAYc0add19V3efleSMJOdU1WOO4LFXdPfO7t65bdu22WYEAAAY2VI+/bK7/0+S9ye5IMmtVXVakkzL/cuYAQAAYBUtFHVV9baqenpVLRyBVbWtqh463b5/kqck+WySq5PsmnbbleSqI5oYAACAv7ZopP1ekl9Isq+qLq+qH1/gMacleV9VfSLJdVl7T927klye5Pyq2pfk/GkdAACAo3D8Ijt193uS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",
      "text/plain": [
       "<Figure size 1080x360 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.figure(figsize=(15,5))\n",
    "sns.distplot(\n",
    "    val_results_df[val_results_df['Target'] > 0]['Prediction'],\n",
    "    fit=stats.norm\n",
    ")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 67,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(6254518, 85)"
      ]
     },
     "execution_count": 67,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "full_3OHE.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 68,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(6254518, 85)"
      ]
     },
     "execution_count": 68,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.shape"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# TRABAJEMOS LA CLASIFICACIÓN"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Para este caso, podemos por ejemplo validar con los últimos 6 meses de 2019 y utilizar 2018 y los 6 primeros de 2019 para realizar el entrenamiento del modelo (train y test). Para seleccionarlo, utilizaremos de nuevo el boolean indexing"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Trabajemos con Decition Three"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "metadata": {},
   "outputs": [],
   "source": [
    "\r\n",
    "\r\n",
    "TARGET1 = t\r\n",
    "\r\n",
    "\r\n",
    "\r\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "count    217802.0\n",
       "mean          1.0\n",
       "std           0.0\n",
       "min           1.0\n",
       "25%           1.0\n",
       "50%           1.0\n",
       "75%           1.0\n",
       "max           1.0\n",
       "Name: Productos_pension_plan, dtype: float64"
      ]
     },
     "execution_count": 20,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df[df[TARGET1]==1][TARGET1].describe()\n",
    "\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "count    6036716.0\n",
       "mean           0.0\n",
       "std            0.0\n",
       "min            0.0\n",
       "25%            0.0\n",
       "50%            0.0\n",
       "75%            0.0\n",
       "max            0.0\n",
       "Name: Productos_pension_plan, dtype: float64"
      ]
     },
     "execution_count": 21,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "\n",
    "df[df[TARGET1]==0][TARGET1].describe()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 333,
   "metadata": {},
   "outputs": [],
   "source": [
    "df['Compra'] = (df[TARGET1] == 1).astype(int)#esta variable es un indicador que me indica cuando alguien ha comprado, la creamos a partir del target. Cuando entrenemos el modelo esta variable no puede estar porque qle seria facil predecir.\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 334,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "hay un total de 6254518visitas, de las cuales 71658 acaban con compra, lo que representa un 1.1457%\n"
     ]
    }
   ],
   "source": [
    "count_transac= df['Compra'].count()\n",
    "sum_transac= df['Compra'].sum()\n",
    "mean_transac= df['Compra'].mean()\n",
    "print(f\"hay un total de {count_transac}visitas, de las cuales {sum_transac} acaban con compra, lo que representa un {round(mean_transac*100,4)}%\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "scrolled": false
   },
   "outputs": [],
   "source": [
    "'''# Distribución de la variable target al completo\n",
    "plt.figure(figsize=(15, 5))\n",
    "sns.distplot(\n",
    "    df['Compra']\n",
    ")'''"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "df.columns"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 364,
   "metadata": {},
   "outputs": [],
   "source": [
    "df.drop('Compra',axis=1,inplace=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "metadata": {},
   "outputs": [],
   "source": [
    "dev_df = df[(df['Year'] == 2018) | ((df['Year'] == 2019) & (df['Month'] < 4)) ] # development = train + test\n",
    "val_df = df[(df['Year'] == 2019) & (df['Month'] >= 4)] # validation"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 60,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(5383155, 84)"
      ]
     },
     "execution_count": 60,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "dev_df.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 62,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(871363, 84)"
      ]
     },
     "execution_count": 62,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "val_df.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "28.21991108891651"
      ]
     },
     "execution_count": 23,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "dev_df_1 = dev_df[dev_df[TARGET1]==1]\n",
    "dev_df_0 = dev_df[dev_df[TARGET1]==0]\n",
    "\n",
    "dev_df_0_shape = dev_df_0.shape[0]\n",
    "dev_df_0_shape\n",
    "\n",
    "dev_df_1_shape = dev_df_1.shape[0]\n",
    "dev_df_1_shape\n",
    "prop = dev_df_0_shape/dev_df_1_shape\n",
    "prop"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "metadata": {
    "azdata_cell_guid": "fe0ab95a-2889-4599-aa0d-49b9dc6b150d"
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "56.43982217783302"
      ]
     },
     "execution_count": 24,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "## Debemos rebalancear el dataset\r\n",
    "\r\n",
    "dev_df_1 = dev_df[dev_df[TARGET1]==1]\r\n",
    "dev_df_0 = dev_df[dev_df[TARGET1]==0]\r\n",
    "\r\n",
    "dev_df_0_shape = dev_df_0.shape[0]*2\r\n",
    "\r\n",
    "dev_df_1_shape = dev_df_1.shape[0]\r\n",
    "dev_df_1_shape\r\n",
    "\r\n",
    "prop_dev = dev_df_0_shape/dev_df_1_shape\r\n",
    "prop_dev\r\n",
    "\r\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "metadata": {},
   "outputs": [],
   "source": [
    "RANDOM_STATE = 42\r\n",
    "\r\n",
    "dev_df_0_sample = dev_df_0.sample(n=dev_df_0_shape,replace=True, random_state=RANDOM_STATE)\r\n",
    "\r\n",
    "\r\n",
    "\r\n",
    "dev_df_sample = pd.concat([dev_df_0,dev_df_1])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "metadata": {
    "azdata_cell_guid": "a22f7811-3145-414a-b107-369226302a76"
   },
   "outputs": [],
   "source": [
    "dev_df_X = dev_df.drop(TARGET1, axis = 1)\n",
    "dev_df_y = dev_df[[TARGET1]]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "metadata": {
    "azdata_cell_guid": "8d2f1b95-66bd-4c25-88ab-6b5d2ac9ca5c",
    "tags": []
   },
   "outputs": [],
   "source": [
    "val_df_X = val_df.drop(TARGET1, axis = 1)\n",
    "val_df_y = val_df[[TARGET1]]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "metadata": {
    "azdata_cell_guid": "c6ca6d62-df28-4364-9757-7f4b23c4968b"
   },
   "outputs": [],
   "source": [
    "from sklearn import model_selection\n",
    "from sklearn import metrics"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "metadata": {
    "azdata_cell_guid": "8b4613eb-6989-42fe-8cc7-21888f9f7b88"
   },
   "outputs": [],
   "source": [
    "X_train, X_test, y_train, y_test = model_selection.train_test_split(dev_df_X, dev_df_y, test_size=0.2, random_state=42)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    " ### Importar los scikits de modelización"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 266,
   "metadata": {
    "azdata_cell_guid": "f2b346fd-c44c-4504-86da-59fce1eb414a"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Collecting package metadata (current_repodata.json): ...working... done\n",
      "Solving environment: ...working... done\n",
      "\n",
      "# All requested packages already installed.\n",
      "\n"
     ]
    }
   ],
   "source": [
    "!conda install python-graphviz -y"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 267,
   "metadata": {
    "azdata_cell_guid": "f6def2ca-66b7-4e68-8b4b-4f930b89ef29"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Collecting package metadata (current_repodata.json): ...working... done\n",
      "Solving environment: ...working... done\n",
      "\n",
      "# All requested packages already installed.\n",
      "\n"
     ]
    }
   ],
   "source": [
    "!conda install pydot -y"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "metadata": {
    "azdata_cell_guid": "4e078a97-54c0-4544-87f4-e15583afc390"
   },
   "outputs": [],
   "source": [
    "from sklearn.tree import DecisionTreeClassifier\n",
    "from sklearn.tree import export_graphviz\n",
    "import graphviz"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "metadata": {
    "azdata_cell_guid": "eb325d69-02f7-418f-9e43-35f305662d98"
   },
   "outputs": [],
   "source": [
    "dt = DecisionTreeClassifier(\n",
    "\n",
    "                 #criterion=\"gini\",\n",
    "                 #splitter=\"best\",\n",
    "                 max_depth= 4, # número de preguntas que realiza el algoritmo\n",
    "                 #min_samples_split=2, # numero de obs minimas en cada particion\n",
    "                 min_samples_leaf=200,\n",
    "                 #min_weight_fraction_leaf=0.,\n",
    "                 #max_features=None, #numero de variables a evaluar en cada iteracion\n",
    "                 random_state=42\n",
    "                 #max_leaf_nodes=None,\n",
    "                 #min_impurity_decrease=0.,\n",
    "                 #min_impurity_split=None,\n",
    "                 #class_weight=None,\n",
    "                 #presort='deprecated'\n",
    ")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 32,
   "metadata": {
    "azdata_cell_guid": "87f3948f-b65c-43c5-8a1e-255eb6fb79f7"
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "DecisionTreeClassifier(max_depth=4, min_samples_leaf=200, random_state=42)"
      ]
     },
     "execution_count": 32,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "dt.fit(\n",
    "    X=X_train, \n",
    "    y=y_train, \n",
    "    # sample_weight=None, \n",
    "    # check_input=True, \n",
    "    # X_idx_sorted=None\n",
    ")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 33,
   "metadata": {
    "azdata_cell_guid": "b990c869-a10d-4d9a-9703-895d5e197c39"
   },
   "outputs": [],
   "source": [
    "dot_data = export_graphviz(\n",
    "                        decision_tree = dt,\n",
    "                        out_file=None,\n",
    "                        # max_depth=None,\n",
    "                        feature_names=X_test.columns,\n",
    "                        class_names=['Compra', 'No compra'],\n",
    "                        # label='all',\n",
    "                        filled=True,\n",
    "                        # leaves_parallel=False,\n",
    "                        impurity=True,\n",
    "                        # node_ids=False,\n",
    "                        proportion=True,\n",
    "                        rotate=True,\n",
    "                        rounded=True,\n",
    "                        # special_characters=False,\n",
    "                        precision=4,\n",
    "                        )"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 34,
   "metadata": {
    "azdata_cell_guid": "865e0fa5-00b5-4810-8b7a-48875f231658"
   },
   "outputs": [
    {
     "data": {
      "image/svg+xml": "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"no\"?>\r\n<!DOCTYPE svg PUBLIC \"-//W3C//DTD SVG 1.1//EN\"\r\n \"http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd\">\r\n<!-- Generated by graphviz version 2.38.0 (20140413.2041)\r\n -->\r\n<!-- Title: Tree Pages: 1 -->\r\n<svg width=\"1076pt\" height=\"850pt\"\r\n viewBox=\"0.00 0.00 1076.00 850.00\" xmlns=\"http://www.w3.org/2000/svg\" xmlns:xlink=\"http://www.w3.org/1999/xlink\">\r\n<g id=\"graph0\" class=\"graph\" transform=\"scale(1 1) rotate(0) translate(4 846)\">\r\n<title>Tree</title>\r\n<polygon fill=\"white\" stroke=\"none\" points=\"-4,4 -4,-846 1072,-846 1072,4 -4,4\"/>\r\n<!-- 0 -->\r\n<g id=\"node1\" class=\"node\"><title>0</title>\r\n<path fill=\"#e68540\" stroke=\"black\" d=\"M159,-441.5C159,-441.5 12,-441.5 12,-441.5 6,-441.5 0,-435.5 0,-429.5 0,-429.5 0,-370.5 0,-370.5 0,-364.5 6,-358.5 12,-358.5 12,-358.5 159,-358.5 159,-358.5 165,-358.5 171,-364.5 171,-370.5 171,-370.5 171,-429.5 171,-429.5 171,-435.5 165,-441.5 159,-441.5\"/>\r\n<text text-anchor=\"middle\" x=\"85.5\" y=\"-426.3\" font-family=\"Helvetica,sans-Serif\" font-size=\"14.00\">customer_count &lt;= 36.5</text>\r\n<text text-anchor=\"middle\" x=\"85.5\" y=\"-411.3\" font-family=\"Helvetica,sans-Serif\" font-size=\"14.00\">gini = 0.0662</text>\r\n<text text-anchor=\"middle\" x=\"85.5\" y=\"-396.3\" font-family=\"Helvetica,sans-Serif\" font-size=\"14.00\">samples = 100.0%</text>\r\n<text text-anchor=\"middle\" x=\"85.5\" y=\"-381.3\" font-family=\"Helvetica,sans-Serif\" font-size=\"14.00\">value = [0.9657, 0.0343]</text>\r\n<text text-anchor=\"middle\" x=\"85.5\" y=\"-366.3\" font-family=\"Helvetica,sans-Serif\" font-size=\"14.00\">class = Compra</text>\r\n</g>\r\n<!-- 1 -->\r\n<g id=\"node2\" class=\"node\"><title>1</title>\r\n<path fill=\"#e5823b\" stroke=\"black\" d=\"M400,-494.5C400,-494.5 219,-494.5 219,-494.5 213,-494.5 207,-488.5 207,-482.5 207,-482.5 207,-423.5 207,-423.5 207,-417.5 213,-411.5 219,-411.5 219,-411.5 400,-411.5 400,-411.5 406,-411.5 412,-417.5 412,-423.5 412,-423.5 412,-482.5 412,-482.5 412,-488.5 406,-494.5 400,-494.5\"/>\r\n<text text-anchor=\"middle\" x=\"309.5\" y=\"-479.3\" font-family=\"Helvetica,sans-Serif\" font-size=\"14.00\">Productos_em_acount &lt;= 0.5</text>\r\n<text text-anchor=\"middle\" x=\"309.5\" y=\"-464.3\" font-family=\"Helvetica,sans-Serif\" font-size=\"14.00\">gini = 0.0165</text>\r\n<text text-anchor=\"middle\" x=\"309.5\" y=\"-449.3\" font-family=\"Helvetica,sans-Serif\" font-size=\"14.00\">samples = 81.0%</text>\r\n<text text-anchor=\"middle\" x=\"309.5\" y=\"-434.3\" font-family=\"Helvetica,sans-Serif\" font-size=\"14.00\">value = [0.9917, 0.0083]</text>\r\n<text text-anchor=\"middle\" x=\"309.5\" y=\"-419.3\" font-family=\"Helvetica,sans-Serif\" font-size=\"14.00\">class = Compra</text>\r\n</g>\r\n<!-- 0&#45;&gt;1 -->\r\n<g id=\"edge1\" class=\"edge\"><title>0&#45;&gt;1</title>\r\n<path fill=\"none\" stroke=\"black\" d=\"M171.062,-420.188C179.52,-422.207 188.195,-424.278 196.875,-426.351\"/>\r\n<polygon fill=\"black\" stroke=\"black\" points=\"196.127,-429.77 206.666,-428.688 197.752,-422.962 196.127,-429.77\"/>\r\n<text text-anchor=\"middle\" x=\"185.367\" y=\"-438.078\" font-family=\"Helvetica,sans-Serif\" font-size=\"14.00\">True</text>\r\n</g>\r\n<!-- 10 -->\r\n<g id=\"node11\" class=\"node\"><title>10</title>\r\n<path fill=\"#e9965b\" stroke=\"black\" d=\"M382.5,-388.5C382.5,-388.5 236.5,-388.5 236.5,-388.5 230.5,-388.5 224.5,-382.5 224.5,-376.5 224.5,-376.5 224.5,-317.5 224.5,-317.5 224.5,-311.5 230.5,-305.5 236.5,-305.5 236.5,-305.5 382.5,-305.5 382.5,-305.5 388.5,-305.5 394.5,-311.5 394.5,-317.5 394.5,-317.5 394.5,-376.5 394.5,-376.5 394.5,-382.5 388.5,-388.5 382.5,-388.5\"/>\r\n<text text-anchor=\"middle\" x=\"309.5\" y=\"-373.3\" font-family=\"Helvetica,sans-Serif\" font-size=\"14.00\">recurrencia &lt;= 32.5</text>\r\n<text text-anchor=\"middle\" x=\"309.5\" y=\"-358.3\" font-family=\"Helvetica,sans-Serif\" font-size=\"14.00\">gini = 0.2483</text>\r\n<text text-anchor=\"middle\" x=\"309.5\" y=\"-343.3\" font-family=\"Helvetica,sans-Serif\" font-size=\"14.00\">samples = 19.0%</text>\r\n<text text-anchor=\"middle\" x=\"309.5\" y=\"-328.3\" font-family=\"Helvetica,sans-Serif\" font-size=\"14.00\">value = [0.8547, 0.1453]</text>\r\n<text text-anchor=\"middle\" x=\"309.5\" y=\"-313.3\" font-family=\"Helvetica,sans-Serif\" font-size=\"14.00\">class = Compra</text>\r\n</g>\r\n<!-- 0&#45;&gt;10 -->\r\n<g id=\"edge10\" class=\"edge\"><title>0&#45;&gt;10</title>\r\n<path fill=\"none\" stroke=\"black\" d=\"M171.062,-379.812C185.199,-376.437 199.944,-372.917 214.305,-369.488\"/>\r\n<polygon fill=\"black\" stroke=\"black\" points=\"215.259,-372.859 224.173,-367.132 213.634,-366.05 215.259,-372.859\"/>\r\n<text text-anchor=\"middle\" x=\"202.874\" y=\"-350.343\" font-family=\"Helvetica,sans-Serif\" font-size=\"14.00\">False</text>\r\n</g>\r\n<!-- 2 -->\r\n<g id=\"node3\" class=\"node\"><title>2</title>\r\n<path fill=\"#e68844\" stroke=\"black\" d=\"M609,-598.5C609,-598.5 463,-598.5 463,-598.5 457,-598.5 451,-592.5 451,-586.5 451,-586.5 451,-527.5 451,-527.5 451,-521.5 457,-515.5 463,-515.5 463,-515.5 609,-515.5 609,-515.5 615,-515.5 621,-521.5 621,-527.5 621,-527.5 621,-586.5 621,-586.5 621,-592.5 615,-598.5 609,-598.5\"/>\r\n<text text-anchor=\"middle\" x=\"536\" y=\"-583.3\" font-family=\"Helvetica,sans-Serif\" font-size=\"14.00\">days_between &lt;= &#45;14.0</text>\r\n<text text-anchor=\"middle\" x=\"536\" y=\"-568.3\" font-family=\"Helvetica,sans-Serif\" font-size=\"14.00\">gini = 0.0993</text>\r\n<text text-anchor=\"middle\" x=\"536\" y=\"-553.3\" font-family=\"Helvetica,sans-Serif\" font-size=\"14.00\">samples = 12.8%</text>\r\n<text text-anchor=\"middle\" x=\"536\" y=\"-538.3\" font-family=\"Helvetica,sans-Serif\" font-size=\"14.00\">value = [0.9476, 0.0524]</text>\r\n<text text-anchor=\"middle\" x=\"536\" y=\"-523.3\" font-family=\"Helvetica,sans-Serif\" font-size=\"14.00\">class = Compra</text>\r\n</g>\r\n<!-- 1&#45;&gt;2 -->\r\n<g id=\"edge2\" class=\"edge\"><title>1&#45;&gt;2</title>\r\n<path fill=\"none\" stroke=\"black\" d=\"M400.196,-494.552C413.791,-500.85 427.846,-507.361 441.519,-513.695\"/>\r\n<polygon fill=\"black\" stroke=\"black\" points=\"440.369,-517.019 450.914,-518.047 443.311,-510.668 440.369,-517.019\"/>\r\n</g>\r\n<!-- 9 -->\r\n<g id=\"node10\" class=\"node\"><title>9</title>\r\n<path fill=\"#e58139\" stroke=\"black\" d=\"M588,-487C588,-487 484,-487 484,-487 478,-487 472,-481 472,-475 472,-475 472,-431 472,-431 472,-425 478,-419 484,-419 484,-419 588,-419 588,-419 594,-419 600,-425 600,-431 600,-431 600,-475 600,-475 600,-481 594,-487 588,-487\"/>\r\n<text text-anchor=\"middle\" x=\"536\" y=\"-471.8\" font-family=\"Helvetica,sans-Serif\" font-size=\"14.00\">gini = 0.0</text>\r\n<text text-anchor=\"middle\" x=\"536\" y=\"-456.8\" font-family=\"Helvetica,sans-Serif\" font-size=\"14.00\">samples = 68.2%</text>\r\n<text text-anchor=\"middle\" x=\"536\" y=\"-441.8\" font-family=\"Helvetica,sans-Serif\" font-size=\"14.00\">value = [1.0, 0.0]</text>\r\n<text text-anchor=\"middle\" x=\"536\" y=\"-426.8\" font-family=\"Helvetica,sans-Serif\" font-size=\"14.00\">class = Compra</text>\r\n</g>\r\n<!-- 1&#45;&gt;9 -->\r\n<g id=\"edge9\" class=\"edge\"><title>1&#45;&gt;9</title>\r\n<path fill=\"none\" stroke=\"black\" d=\"M412.24,-453C428.895,-453 445.861,-453 461.654,-453\"/>\r\n<polygon fill=\"black\" stroke=\"black\" points=\"461.848,-456.5 471.848,-453 461.848,-449.5 461.848,-456.5\"/>\r\n</g>\r\n<!-- 3 -->\r\n<g id=\"node4\" class=\"node\"><title>3</title>\r\n<path fill=\"#efb184\" stroke=\"black\" d=\"M834,-763.5C834,-763.5 688,-763.5 688,-763.5 682,-763.5 676,-757.5 676,-751.5 676,-751.5 676,-692.5 676,-692.5 676,-686.5 682,-680.5 688,-680.5 688,-680.5 834,-680.5 834,-680.5 840,-680.5 846,-686.5 846,-692.5 846,-692.5 846,-751.5 846,-751.5 846,-757.5 840,-763.5 834,-763.5\"/>\r\n<text text-anchor=\"middle\" x=\"761\" y=\"-748.3\" font-family=\"Helvetica,sans-Serif\" font-size=\"14.00\">recurrencia &lt;= 8.5</text>\r\n<text text-anchor=\"middle\" x=\"761\" y=\"-733.3\" font-family=\"Helvetica,sans-Serif\" font-size=\"14.00\">gini = 0.3993</text>\r\n<text text-anchor=\"middle\" x=\"761\" y=\"-718.3\" font-family=\"Helvetica,sans-Serif\" font-size=\"14.00\">samples = 0.6%</text>\r\n<text text-anchor=\"middle\" x=\"761\" y=\"-703.3\" font-family=\"Helvetica,sans-Serif\" font-size=\"14.00\">value = [0.7244, 0.2756]</text>\r\n<text text-anchor=\"middle\" x=\"761\" y=\"-688.3\" font-family=\"Helvetica,sans-Serif\" font-size=\"14.00\">class = Compra</text>\r\n</g>\r\n<!-- 2&#45;&gt;3 -->\r\n<g id=\"edge3\" class=\"edge\"><title>2&#45;&gt;3</title>\r\n<path fill=\"none\" stroke=\"black\" d=\"M593.272,-598.636C624.436,-621.695 663.301,-650.452 695.666,-674.399\"/>\r\n<polygon fill=\"black\" stroke=\"black\" points=\"693.625,-677.242 703.746,-680.377 697.789,-671.615 693.625,-677.242\"/>\r\n</g>\r\n<!-- 6 -->\r\n<g id=\"node7\" class=\"node\"><title>6</title>\r\n<path fill=\"#e68642\" stroke=\"black\" d=\"M850,-598.5C850,-598.5 672,-598.5 672,-598.5 666,-598.5 660,-592.5 660,-586.5 660,-586.5 660,-527.5 660,-527.5 660,-521.5 666,-515.5 672,-515.5 672,-515.5 850,-515.5 850,-515.5 856,-515.5 862,-521.5 862,-527.5 862,-527.5 862,-586.5 862,-586.5 862,-592.5 856,-598.5 850,-598.5\"/>\r\n<text text-anchor=\"middle\" x=\"761\" y=\"-583.3\" font-family=\"Helvetica,sans-Serif\" font-size=\"14.00\">Productos_debit_card &lt;= 0.5</text>\r\n<text text-anchor=\"middle\" x=\"761\" y=\"-568.3\" font-family=\"Helvetica,sans-Serif\" font-size=\"14.00\">gini = 0.08</text>\r\n<text text-anchor=\"middle\" x=\"761\" y=\"-553.3\" font-family=\"Helvetica,sans-Serif\" font-size=\"14.00\">samples = 12.3%</text>\r\n<text text-anchor=\"middle\" x=\"761\" y=\"-538.3\" font-family=\"Helvetica,sans-Serif\" font-size=\"14.00\">value = [0.9582, 0.0418]</text>\r\n<text text-anchor=\"middle\" x=\"761\" y=\"-523.3\" font-family=\"Helvetica,sans-Serif\" font-size=\"14.00\">class = Compra</text>\r\n</g>\r\n<!-- 2&#45;&gt;6 -->\r\n<g id=\"edge6\" class=\"edge\"><title>2&#45;&gt;6</title>\r\n<path fill=\"none\" stroke=\"black\" d=\"M621.307,-557C630.546,-557 640.05,-557 649.544,-557\"/>\r\n<polygon fill=\"black\" stroke=\"black\" points=\"649.829,-560.5 659.829,-557 649.829,-553.5 649.829,-560.5\"/>\r\n</g>\r\n<!-- 4 -->\r\n<g id=\"node5\" class=\"node\"><title>4</title>\r\n<path fill=\"#fdf4ee\" stroke=\"black\" d=\"M1048,-842C1048,-842 918,-842 918,-842 912,-842 906,-836 906,-830 906,-830 906,-786 906,-786 906,-780 912,-774 918,-774 918,-774 1048,-774 1048,-774 1054,-774 1060,-780 1060,-786 1060,-786 1060,-830 1060,-830 1060,-836 1054,-842 1048,-842\"/>\r\n<text text-anchor=\"middle\" x=\"983\" y=\"-826.8\" font-family=\"Helvetica,sans-Serif\" font-size=\"14.00\">gini = 0.4989</text>\r\n<text text-anchor=\"middle\" x=\"983\" y=\"-811.8\" font-family=\"Helvetica,sans-Serif\" font-size=\"14.00\">samples = 0.3%</text>\r\n<text text-anchor=\"middle\" x=\"983\" y=\"-796.8\" font-family=\"Helvetica,sans-Serif\" font-size=\"14.00\">value = [0.523, 0.477]</text>\r\n<text text-anchor=\"middle\" x=\"983\" y=\"-781.8\" font-family=\"Helvetica,sans-Serif\" font-size=\"14.00\">class = Compra</text>\r\n</g>\r\n<!-- 3&#45;&gt;4 -->\r\n<g id=\"edge4\" class=\"edge\"><title>3&#45;&gt;4</title>\r\n<path fill=\"none\" stroke=\"black\" d=\"M846.114,-754.881C862.541,-761.302 879.785,-768.043 896.264,-774.485\"/>\r\n<polygon fill=\"black\" stroke=\"black\" points=\"895.278,-777.857 905.866,-778.239 897.827,-771.338 895.278,-777.857\"/>\r\n</g>\r\n<!-- 5 -->\r\n<g id=\"node6\" class=\"node\"><title>5</title>\r\n<path fill=\"#e78b49\" stroke=\"black\" d=\"M1056,-756C1056,-756 910,-756 910,-756 904,-756 898,-750 898,-744 898,-744 898,-700 898,-700 898,-694 904,-688 910,-688 910,-688 1056,-688 1056,-688 1062,-688 1068,-694 1068,-700 1068,-700 1068,-744 1068,-744 1068,-750 1062,-756 1056,-756\"/>\r\n<text text-anchor=\"middle\" x=\"983\" y=\"-740.8\" font-family=\"Helvetica,sans-Serif\" font-size=\"14.00\">gini = 0.1377</text>\r\n<text text-anchor=\"middle\" x=\"983\" y=\"-725.8\" font-family=\"Helvetica,sans-Serif\" font-size=\"14.00\">samples = 0.3%</text>\r\n<text text-anchor=\"middle\" x=\"983\" y=\"-710.8\" font-family=\"Helvetica,sans-Serif\" font-size=\"14.00\">value = [0.9256, 0.0744]</text>\r\n<text text-anchor=\"middle\" x=\"983\" y=\"-695.8\" font-family=\"Helvetica,sans-Serif\" font-size=\"14.00\">class = Compra</text>\r\n</g>\r\n<!-- 3&#45;&gt;5 -->\r\n<g id=\"edge5\" class=\"edge\"><title>3&#45;&gt;5</title>\r\n<path fill=\"none\" stroke=\"black\" d=\"M846.114,-722C859.78,-722 874.01,-722 887.896,-722\"/>\r\n<polygon fill=\"black\" stroke=\"black\" points=\"887.932,-725.5 897.932,-722 887.932,-718.5 887.932,-725.5\"/>\r\n</g>\r\n<!-- 7 -->\r\n<g id=\"node8\" class=\"node\"><title>7</title>\r\n<path fill=\"#e78b48\" stroke=\"black\" d=\"M1056,-670C1056,-670 910,-670 910,-670 904,-670 898,-664 898,-658 898,-658 898,-614 898,-614 898,-608 904,-602 910,-602 910,-602 1056,-602 1056,-602 1062,-602 1068,-608 1068,-614 1068,-614 1068,-658 1068,-658 1068,-664 1062,-670 1056,-670\"/>\r\n<text text-anchor=\"middle\" x=\"983\" y=\"-654.8\" font-family=\"Helvetica,sans-Serif\" font-size=\"14.00\">gini = 0.1312</text>\r\n<text text-anchor=\"middle\" x=\"983\" y=\"-639.8\" font-family=\"Helvetica,sans-Serif\" font-size=\"14.00\">samples = 7.3%</text>\r\n<text text-anchor=\"middle\" x=\"983\" y=\"-624.8\" font-family=\"Helvetica,sans-Serif\" font-size=\"14.00\">value = [0.9294, 0.0706]</text>\r\n<text text-anchor=\"middle\" x=\"983\" y=\"-609.8\" font-family=\"Helvetica,sans-Serif\" font-size=\"14.00\">class = Compra</text>\r\n</g>\r\n<!-- 6&#45;&gt;7 -->\r\n<g id=\"edge7\" class=\"edge\"><title>6&#45;&gt;7</title>\r\n<path fill=\"none\" stroke=\"black\" d=\"M862.022,-592.917C870.681,-596.027 879.424,-599.166 888.033,-602.257\"/>\r\n<polygon fill=\"black\" stroke=\"black\" points=\"887.089,-605.637 897.684,-605.723 889.455,-599.049 887.089,-605.637\"/>\r\n</g>\r\n<!-- 8 -->\r\n<g id=\"node9\" class=\"node\"><title>8</title>\r\n<path fill=\"#e58139\" stroke=\"black\" d=\"M1031.5,-584C1031.5,-584 934.5,-584 934.5,-584 928.5,-584 922.5,-578 922.5,-572 922.5,-572 922.5,-528 922.5,-528 922.5,-522 928.5,-516 934.5,-516 934.5,-516 1031.5,-516 1031.5,-516 1037.5,-516 1043.5,-522 1043.5,-528 1043.5,-528 1043.5,-572 1043.5,-572 1043.5,-578 1037.5,-584 1031.5,-584\"/>\r\n<text text-anchor=\"middle\" x=\"983\" y=\"-568.8\" font-family=\"Helvetica,sans-Serif\" font-size=\"14.00\">gini = 0.0</text>\r\n<text text-anchor=\"middle\" x=\"983\" y=\"-553.8\" font-family=\"Helvetica,sans-Serif\" font-size=\"14.00\">samples = 5.0%</text>\r\n<text text-anchor=\"middle\" x=\"983\" y=\"-538.8\" font-family=\"Helvetica,sans-Serif\" font-size=\"14.00\">value = [1.0, 0.0]</text>\r\n<text text-anchor=\"middle\" x=\"983\" y=\"-523.8\" font-family=\"Helvetica,sans-Serif\" font-size=\"14.00\">class = Compra</text>\r\n</g>\r\n<!-- 6&#45;&gt;8 -->\r\n<g id=\"edge8\" class=\"edge\"><title>6&#45;&gt;8</title>\r\n<path fill=\"none\" stroke=\"black\" d=\"M862.022,-553.817C878.968,-553.278 896.233,-552.729 912.19,-552.221\"/>\r\n<polygon fill=\"black\" stroke=\"black\" points=\"912.587,-555.71 922.47,-551.894 912.364,-548.714 912.587,-555.71\"/>\r\n</g>\r\n<!-- 11 -->\r\n<g id=\"node12\" class=\"node\"><title>11</title>\r\n<path fill=\"#eca26d\" stroke=\"black\" d=\"M612,-388.5C612,-388.5 460,-388.5 460,-388.5 454,-388.5 448,-382.5 448,-376.5 448,-376.5 448,-317.5 448,-317.5 448,-311.5 454,-305.5 460,-305.5 460,-305.5 612,-305.5 612,-305.5 618,-305.5 624,-311.5 624,-317.5 624,-317.5 624,-376.5 624,-376.5 624,-382.5 618,-388.5 612,-388.5\"/>\r\n<text text-anchor=\"middle\" x=\"536\" y=\"-373.3\" font-family=\"Helvetica,sans-Serif\" font-size=\"14.00\">Productos_payroll &lt;= 0.5</text>\r\n<text text-anchor=\"middle\" x=\"536\" y=\"-358.3\" font-family=\"Helvetica,sans-Serif\" font-size=\"14.00\">gini = 0.3294</text>\r\n<text text-anchor=\"middle\" x=\"536\" y=\"-343.3\" font-family=\"Helvetica,sans-Serif\" font-size=\"14.00\">samples = 11.0%</text>\r\n<text text-anchor=\"middle\" x=\"536\" y=\"-328.3\" font-family=\"Helvetica,sans-Serif\" font-size=\"14.00\">value = [0.792, 0.208]</text>\r\n<text text-anchor=\"middle\" x=\"536\" y=\"-313.3\" font-family=\"Helvetica,sans-Serif\" font-size=\"14.00\">class = Compra</text>\r\n</g>\r\n<!-- 10&#45;&gt;11 -->\r\n<g id=\"edge11\" class=\"edge\"><title>10&#45;&gt;11</title>\r\n<path fill=\"none\" stroke=\"black\" d=\"M394.734,-347C408.651,-347 423.181,-347 437.393,-347\"/>\r\n<polygon fill=\"black\" stroke=\"black\" points=\"437.671,-350.5 447.671,-347 437.671,-343.5 437.671,-350.5\"/>\r\n</g>\r\n<!-- 16 -->\r\n<g id=\"node17\" class=\"node\"><title>16</title>\r\n<path fill=\"#e78945\" stroke=\"black\" d=\"M609.5,-267.5C609.5,-267.5 462.5,-267.5 462.5,-267.5 456.5,-267.5 450.5,-261.5 450.5,-255.5 450.5,-255.5 450.5,-196.5 450.5,-196.5 450.5,-190.5 456.5,-184.5 462.5,-184.5 462.5,-184.5 609.5,-184.5 609.5,-184.5 615.5,-184.5 621.5,-190.5 621.5,-196.5 621.5,-196.5 621.5,-255.5 621.5,-255.5 621.5,-261.5 615.5,-267.5 609.5,-267.5\"/>\r\n<text text-anchor=\"middle\" x=\"536\" y=\"-252.3\" font-family=\"Helvetica,sans-Serif\" font-size=\"14.00\">customer_count &lt;= 70.5</text>\r\n<text text-anchor=\"middle\" x=\"536\" y=\"-237.3\" font-family=\"Helvetica,sans-Serif\" font-size=\"14.00\">gini = 0.1092</text>\r\n<text text-anchor=\"middle\" x=\"536\" y=\"-222.3\" font-family=\"Helvetica,sans-Serif\" font-size=\"14.00\">samples = 7.9%</text>\r\n<text text-anchor=\"middle\" x=\"536\" y=\"-207.3\" font-family=\"Helvetica,sans-Serif\" font-size=\"14.00\">value = [0.942, 0.058]</text>\r\n<text text-anchor=\"middle\" x=\"536\" y=\"-192.3\" font-family=\"Helvetica,sans-Serif\" font-size=\"14.00\">class = Compra</text>\r\n</g>\r\n<!-- 10&#45;&gt;16 -->\r\n<g id=\"edge16\" class=\"edge\"><title>10&#45;&gt;16</title>\r\n<path fill=\"none\" stroke=\"black\" d=\"M387.76,-305.359C407.537,-294.699 428.924,-283.172 449.106,-272.295\"/>\r\n<polygon fill=\"black\" stroke=\"black\" points=\"450.836,-275.339 457.978,-267.513 447.514,-269.177 450.836,-275.339\"/>\r\n</g>\r\n<!-- 12 -->\r\n<g id=\"node13\" class=\"node\"><title>12</title>\r\n<path fill=\"#efb083\" stroke=\"black\" d=\"M834.5,-469.5C834.5,-469.5 687.5,-469.5 687.5,-469.5 681.5,-469.5 675.5,-463.5 675.5,-457.5 675.5,-457.5 675.5,-398.5 675.5,-398.5 675.5,-392.5 681.5,-386.5 687.5,-386.5 687.5,-386.5 834.5,-386.5 834.5,-386.5 840.5,-386.5 846.5,-392.5 846.5,-398.5 846.5,-398.5 846.5,-457.5 846.5,-457.5 846.5,-463.5 840.5,-469.5 834.5,-469.5\"/>\r\n<text text-anchor=\"middle\" x=\"761\" y=\"-454.3\" font-family=\"Helvetica,sans-Serif\" font-size=\"14.00\">customer_count &lt;= 51.5</text>\r\n<text text-anchor=\"middle\" x=\"761\" y=\"-439.3\" font-family=\"Helvetica,sans-Serif\" font-size=\"14.00\">gini = 0.3953</text>\r\n<text text-anchor=\"middle\" x=\"761\" y=\"-424.3\" font-family=\"Helvetica,sans-Serif\" font-size=\"14.00\">samples = 8.5%</text>\r\n<text text-anchor=\"middle\" x=\"761\" y=\"-409.3\" font-family=\"Helvetica,sans-Serif\" font-size=\"14.00\">value = [0.7288, 0.2712]</text>\r\n<text text-anchor=\"middle\" x=\"761\" y=\"-394.3\" font-family=\"Helvetica,sans-Serif\" font-size=\"14.00\">class = Compra</text>\r\n</g>\r\n<!-- 11&#45;&gt;12 -->\r\n<g id=\"edge12\" class=\"edge\"><title>11&#45;&gt;12</title>\r\n<path fill=\"none\" stroke=\"black\" d=\"M624.176,-378.665C637.873,-383.64 652.085,-388.802 665.929,-393.831\"/>\r\n<polygon fill=\"black\" stroke=\"black\" points=\"664.85,-397.162 675.444,-397.287 667.239,-390.583 664.85,-397.162\"/>\r\n</g>\r\n<!-- 15 -->\r\n<g id=\"node16\" class=\"node\"><title>15</title>\r\n<path fill=\"#e58139\" stroke=\"black\" d=\"M809.5,-368C809.5,-368 712.5,-368 712.5,-368 706.5,-368 700.5,-362 700.5,-356 700.5,-356 700.5,-312 700.5,-312 700.5,-306 706.5,-300 712.5,-300 712.5,-300 809.5,-300 809.5,-300 815.5,-300 821.5,-306 821.5,-312 821.5,-312 821.5,-356 821.5,-356 821.5,-362 815.5,-368 809.5,-368\"/>\r\n<text text-anchor=\"middle\" x=\"761\" y=\"-352.8\" font-family=\"Helvetica,sans-Serif\" font-size=\"14.00\">gini = 0.0</text>\r\n<text text-anchor=\"middle\" x=\"761\" y=\"-337.8\" font-family=\"Helvetica,sans-Serif\" font-size=\"14.00\">samples = 2.6%</text>\r\n<text text-anchor=\"middle\" x=\"761\" y=\"-322.8\" font-family=\"Helvetica,sans-Serif\" font-size=\"14.00\">value = [1.0, 0.0]</text>\r\n<text text-anchor=\"middle\" x=\"761\" y=\"-307.8\" font-family=\"Helvetica,sans-Serif\" font-size=\"14.00\">class = Compra</text>\r\n</g>\r\n<!-- 11&#45;&gt;15 -->\r\n<g id=\"edge15\" class=\"edge\"><title>11&#45;&gt;15</title>\r\n<path fill=\"none\" stroke=\"black\" d=\"M624.176,-341.918C646.116,-340.639 669.378,-339.283 690.334,-338.061\"/>\r\n<polygon fill=\"black\" stroke=\"black\" points=\"690.641,-341.549 700.42,-337.473 690.234,-334.561 690.641,-341.549\"/>\r\n</g>\r\n<!-- 13 -->\r\n<g id=\"node14\" class=\"node\"><title>13</title>\r\n<path fill=\"#ea995f\" stroke=\"black\" d=\"M1056,-498C1056,-498 910,-498 910,-498 904,-498 898,-492 898,-486 898,-486 898,-442 898,-442 898,-436 904,-430 910,-430 910,-430 1056,-430 1056,-430 1062,-430 1068,-436 1068,-442 1068,-442 1068,-486 1068,-486 1068,-492 1062,-498 1056,-498\"/>\r\n<text text-anchor=\"middle\" x=\"983\" y=\"-482.8\" font-family=\"Helvetica,sans-Serif\" font-size=\"14.00\">gini = 0.2711</text>\r\n<text text-anchor=\"middle\" x=\"983\" y=\"-467.8\" font-family=\"Helvetica,sans-Serif\" font-size=\"14.00\">samples = 4.9%</text>\r\n<text text-anchor=\"middle\" x=\"983\" y=\"-452.8\" font-family=\"Helvetica,sans-Serif\" font-size=\"14.00\">value = [0.8383, 0.1617]</text>\r\n<text text-anchor=\"middle\" x=\"983\" y=\"-437.8\" font-family=\"Helvetica,sans-Serif\" font-size=\"14.00\">class = Compra</text>\r\n</g>\r\n<!-- 12&#45;&gt;13 -->\r\n<g id=\"edge13\" class=\"edge\"><title>12&#45;&gt;13</title>\r\n<path fill=\"none\" stroke=\"black\" d=\"M846.743,-441.867C860.054,-444.045 873.887,-446.309 887.403,-448.521\"/>\r\n<polygon fill=\"black\" stroke=\"black\" points=\"887.23,-452.039 897.664,-450.199 888.36,-445.13 887.23,-452.039\"/>\r\n</g>\r\n<!-- 14 -->\r\n<g id=\"node15\" class=\"node\"><title>14</title>\r\n<path fill=\"#f8ddca\" stroke=\"black\" d=\"M1048,-412C1048,-412 918,-412 918,-412 912,-412 906,-406 906,-400 906,-400 906,-356 906,-356 906,-350 912,-344 918,-344 918,-344 1048,-344 1048,-344 1054,-344 1060,-350 1060,-356 1060,-356 1060,-400 1060,-400 1060,-406 1054,-412 1048,-412\"/>\r\n<text text-anchor=\"middle\" x=\"983\" y=\"-396.8\" font-family=\"Helvetica,sans-Serif\" font-size=\"14.00\">gini = 0.4882</text>\r\n<text text-anchor=\"middle\" x=\"983\" y=\"-381.8\" font-family=\"Helvetica,sans-Serif\" font-size=\"14.00\">samples = 3.5%</text>\r\n<text text-anchor=\"middle\" x=\"983\" y=\"-366.8\" font-family=\"Helvetica,sans-Serif\" font-size=\"14.00\">value = [0.577, 0.423]</text>\r\n<text text-anchor=\"middle\" x=\"983\" y=\"-351.8\" font-family=\"Helvetica,sans-Serif\" font-size=\"14.00\">class = Compra</text>\r\n</g>\r\n<!-- 12&#45;&gt;14 -->\r\n<g id=\"edge14\" class=\"edge\"><title>12&#45;&gt;14</title>\r\n<path fill=\"none\" stroke=\"black\" d=\"M846.743,-408.74C862.882,-405.072 879.787,-401.23 895.966,-397.553\"/>\r\n<polygon fill=\"black\" stroke=\"black\" points=\"896.956,-400.918 905.932,-395.288 895.405,-394.092 896.956,-400.918\"/>\r\n</g>\r\n<!-- 17 -->\r\n<g id=\"node18\" class=\"node\"><title>17</title>\r\n<path fill=\"#e5823b\" stroke=\"black\" d=\"M834.5,-267.5C834.5,-267.5 687.5,-267.5 687.5,-267.5 681.5,-267.5 675.5,-261.5 675.5,-255.5 675.5,-255.5 675.5,-196.5 675.5,-196.5 675.5,-190.5 681.5,-184.5 687.5,-184.5 687.5,-184.5 834.5,-184.5 834.5,-184.5 840.5,-184.5 846.5,-190.5 846.5,-196.5 846.5,-196.5 846.5,-255.5 846.5,-255.5 846.5,-261.5 840.5,-267.5 834.5,-267.5\"/>\r\n<text text-anchor=\"middle\" x=\"761\" y=\"-252.3\" font-family=\"Helvetica,sans-Serif\" font-size=\"14.00\">customer_count &lt;= 68.5</text>\r\n<text text-anchor=\"middle\" x=\"761\" y=\"-237.3\" font-family=\"Helvetica,sans-Serif\" font-size=\"14.00\">gini = 0.02</text>\r\n<text text-anchor=\"middle\" x=\"761\" y=\"-222.3\" font-family=\"Helvetica,sans-Serif\" font-size=\"14.00\">samples = 4.9%</text>\r\n<text text-anchor=\"middle\" x=\"761\" y=\"-207.3\" font-family=\"Helvetica,sans-Serif\" font-size=\"14.00\">value = [0.9899, 0.0101]</text>\r\n<text text-anchor=\"middle\" x=\"761\" y=\"-192.3\" font-family=\"Helvetica,sans-Serif\" font-size=\"14.00\">class = Compra</text>\r\n</g>\r\n<!-- 16&#45;&gt;17 -->\r\n<g id=\"edge17\" class=\"edge\"><title>16&#45;&gt;17</title>\r\n<path fill=\"none\" stroke=\"black\" d=\"M621.625,-226C635.803,-226 650.598,-226 665.015,-226\"/>\r\n<polygon fill=\"black\" stroke=\"black\" points=\"665.433,-229.5 675.433,-226 665.433,-222.5 665.433,-229.5\"/>\r\n</g>\r\n<!-- 20 -->\r\n<g id=\"node21\" class=\"node\"><title>20</title>\r\n<path fill=\"#e99558\" stroke=\"black\" d=\"M834,-161.5C834,-161.5 688,-161.5 688,-161.5 682,-161.5 676,-155.5 676,-149.5 676,-149.5 676,-90.5 676,-90.5 676,-84.5 682,-78.5 688,-78.5 688,-78.5 834,-78.5 834,-78.5 840,-78.5 846,-84.5 846,-90.5 846,-90.5 846,-149.5 846,-149.5 846,-155.5 840,-161.5 834,-161.5\"/>\r\n<text text-anchor=\"middle\" x=\"761\" y=\"-146.3\" font-family=\"Helvetica,sans-Serif\" font-size=\"14.00\">recurrencia &lt;= 46.5</text>\r\n<text text-anchor=\"middle\" x=\"761\" y=\"-131.3\" font-family=\"Helvetica,sans-Serif\" font-size=\"14.00\">gini = 0.2339</text>\r\n<text text-anchor=\"middle\" x=\"761\" y=\"-116.3\" font-family=\"Helvetica,sans-Serif\" font-size=\"14.00\">samples = 3.0%</text>\r\n<text text-anchor=\"middle\" x=\"761\" y=\"-101.3\" font-family=\"Helvetica,sans-Serif\" font-size=\"14.00\">value = [0.8648, 0.1352]</text>\r\n<text text-anchor=\"middle\" x=\"761\" y=\"-86.3\" font-family=\"Helvetica,sans-Serif\" font-size=\"14.00\">class = Compra</text>\r\n</g>\r\n<!-- 16&#45;&gt;20 -->\r\n<g id=\"edge20\" class=\"edge\"><title>16&#45;&gt;20</title>\r\n<path fill=\"none\" stroke=\"black\" d=\"M621.625,-185.775C636.318,-178.791 651.673,-171.492 666.585,-164.404\"/>\r\n<polygon fill=\"black\" stroke=\"black\" points=\"668.273,-167.476 675.803,-160.022 665.268,-161.154 668.273,-167.476\"/>\r\n</g>\r\n<!-- 18 -->\r\n<g id=\"node19\" class=\"node\"><title>18</title>\r\n<path fill=\"#e5823b\" stroke=\"black\" d=\"M1056,-326C1056,-326 910,-326 910,-326 904,-326 898,-320 898,-314 898,-314 898,-270 898,-270 898,-264 904,-258 910,-258 910,-258 1056,-258 1056,-258 1062,-258 1068,-264 1068,-270 1068,-270 1068,-314 1068,-314 1068,-320 1062,-326 1056,-326\"/>\r\n<text text-anchor=\"middle\" x=\"983\" y=\"-310.8\" font-family=\"Helvetica,sans-Serif\" font-size=\"14.00\">gini = 0.0158</text>\r\n<text text-anchor=\"middle\" x=\"983\" y=\"-295.8\" font-family=\"Helvetica,sans-Serif\" font-size=\"14.00\">samples = 4.7%</text>\r\n<text text-anchor=\"middle\" x=\"983\" y=\"-280.8\" font-family=\"Helvetica,sans-Serif\" font-size=\"14.00\">value = [0.9921, 0.0079]</text>\r\n<text text-anchor=\"middle\" x=\"983\" y=\"-265.8\" font-family=\"Helvetica,sans-Serif\" font-size=\"14.00\">class = Compra</text>\r\n</g>\r\n<!-- 17&#45;&gt;18 -->\r\n<g id=\"edge18\" class=\"edge\"><title>17&#45;&gt;18</title>\r\n<path fill=\"none\" stroke=\"black\" d=\"M846.743,-251.423C860.185,-255.455 874.158,-259.647 887.801,-263.74\"/>\r\n<polygon fill=\"black\" stroke=\"black\" points=\"887.08,-267.178 897.664,-266.699 889.091,-260.473 887.08,-267.178\"/>\r\n</g>\r\n<!-- 19 -->\r\n<g id=\"node20\" class=\"node\"><title>19</title>\r\n<path fill=\"#e78945\" stroke=\"black\" d=\"M1056,-240C1056,-240 910,-240 910,-240 904,-240 898,-234 898,-228 898,-228 898,-184 898,-184 898,-178 904,-172 910,-172 910,-172 1056,-172 1056,-172 1062,-172 1068,-178 1068,-184 1068,-184 1068,-228 1068,-228 1068,-234 1062,-240 1056,-240\"/>\r\n<text text-anchor=\"middle\" x=\"983\" y=\"-224.8\" font-family=\"Helvetica,sans-Serif\" font-size=\"14.00\">gini = 0.1116</text>\r\n<text text-anchor=\"middle\" x=\"983\" y=\"-209.8\" font-family=\"Helvetica,sans-Serif\" font-size=\"14.00\">samples = 0.2%</text>\r\n<text text-anchor=\"middle\" x=\"983\" y=\"-194.8\" font-family=\"Helvetica,sans-Serif\" font-size=\"14.00\">value = [0.9407, 0.0593]</text>\r\n<text text-anchor=\"middle\" x=\"983\" y=\"-179.8\" font-family=\"Helvetica,sans-Serif\" font-size=\"14.00\">class = Compra</text>\r\n</g>\r\n<!-- 17&#45;&gt;19 -->\r\n<g id=\"edge19\" class=\"edge\"><title>17&#45;&gt;19</title>\r\n<path fill=\"none\" stroke=\"black\" d=\"M846.743,-218.296C860.054,-217.086 873.887,-215.828 887.403,-214.6\"/>\r\n<polygon fill=\"black\" stroke=\"black\" points=\"888.022,-218.058 897.664,-213.667 887.388,-211.087 888.022,-218.058\"/>\r\n</g>\r\n<!-- 21 -->\r\n<g id=\"node22\" class=\"node\"><title>21</title>\r\n<path fill=\"#f7d7c0\" stroke=\"black\" d=\"M1056,-154C1056,-154 910,-154 910,-154 904,-154 898,-148 898,-142 898,-142 898,-98 898,-98 898,-92 904,-86 910,-86 910,-86 1056,-86 1056,-86 1062,-86 1068,-92 1068,-98 1068,-98 1068,-142 1068,-142 1068,-148 1062,-154 1056,-154\"/>\r\n<text text-anchor=\"middle\" x=\"983\" y=\"-138.8\" font-family=\"Helvetica,sans-Serif\" font-size=\"14.00\">gini = 0.4822</text>\r\n<text text-anchor=\"middle\" x=\"983\" y=\"-123.8\" font-family=\"Helvetica,sans-Serif\" font-size=\"14.00\">samples = 0.8%</text>\r\n<text text-anchor=\"middle\" x=\"983\" y=\"-108.8\" font-family=\"Helvetica,sans-Serif\" font-size=\"14.00\">value = [0.5943, 0.4057]</text>\r\n<text text-anchor=\"middle\" x=\"983\" y=\"-93.8\" font-family=\"Helvetica,sans-Serif\" font-size=\"14.00\">class = Compra</text>\r\n</g>\r\n<!-- 20&#45;&gt;21 -->\r\n<g id=\"edge21\" class=\"edge\"><title>20&#45;&gt;21</title>\r\n<path fill=\"none\" stroke=\"black\" d=\"M846.114,-120C859.78,-120 874.01,-120 887.896,-120\"/>\r\n<polygon fill=\"black\" stroke=\"black\" points=\"887.932,-123.5 897.932,-120 887.932,-116.5 887.932,-123.5\"/>\r\n</g>\r\n<!-- 22 -->\r\n<g id=\"node23\" class=\"node\"><title>22</title>\r\n<path fill=\"#e68640\" stroke=\"black\" d=\"M1056,-68C1056,-68 910,-68 910,-68 904,-68 898,-62 898,-56 898,-56 898,-12 898,-12 898,-6 904,-0 910,-0 910,-0 1056,-0 1056,-0 1062,-0 1068,-6 1068,-12 1068,-12 1068,-56 1068,-56 1068,-62 1062,-68 1056,-68\"/>\r\n<text text-anchor=\"middle\" x=\"983\" y=\"-52.8\" font-family=\"Helvetica,sans-Serif\" font-size=\"14.00\">gini = 0.0688</text>\r\n<text text-anchor=\"middle\" x=\"983\" y=\"-37.8\" font-family=\"Helvetica,sans-Serif\" font-size=\"14.00\">samples = 2.2%</text>\r\n<text text-anchor=\"middle\" x=\"983\" y=\"-22.8\" font-family=\"Helvetica,sans-Serif\" font-size=\"14.00\">value = [0.9643, 0.0357]</text>\r\n<text text-anchor=\"middle\" x=\"983\" y=\"-7.8\" font-family=\"Helvetica,sans-Serif\" font-size=\"14.00\">class = Compra</text>\r\n</g>\r\n<!-- 20&#45;&gt;22 -->\r\n<g id=\"edge22\" class=\"edge\"><title>20&#45;&gt;22</title>\r\n<path fill=\"none\" stroke=\"black\" d=\"M846.114,-87.119C859.912,-81.7251 874.287,-76.1061 888.3,-70.6282\"/>\r\n<polygon fill=\"black\" stroke=\"black\" points=\"889.893,-73.7634 897.932,-66.8628 887.344,-67.2439 889.893,-73.7634\"/>\r\n</g>\r\n</g>\r\n</svg>\r\n",
      "text/plain": [
       "<graphviz.files.Source at 0x1fe22ae3790>"
      ]
     },
     "execution_count": 34,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "graph = graphviz.Source(dot_data)\n",
    "graph"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 35,
   "metadata": {
    "azdata_cell_guid": "943c4d97-8bb4-47ae-9f50-f4c66f3941f6"
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([0, 0, 0, ..., 0, 0, 0], dtype=uint8)"
      ]
     },
     "execution_count": 35,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "dt.predict(X_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 36,
   "metadata": {
    "azdata_cell_guid": "4264007c-d50a-4107-b745-8de25ff5cb90"
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[1.        , 0.        ],\n",
       "       [1.        , 0.        ],\n",
       "       [0.92942367, 0.07057633],\n",
       "       ...,\n",
       "       [1.        , 0.        ],\n",
       "       [1.        , 0.        ],\n",
       "       [0.92942367, 0.07057633]])"
      ]
     },
     "execution_count": 36,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "dt.predict_proba(X_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 49,
   "metadata": {
    "azdata_cell_guid": "862ad7cb-2e9a-419a-8586-8265f3608c43"
   },
   "outputs": [],
   "source": [
    "y_test_pred = pd.DataFrame(dt.predict(X_test), index= y_test.index, columns = ['Predice_compra'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 50,
   "metadata": {
    "azdata_cell_guid": "7d5d3147-10e4-47df-af3e-f9ad3c81d2e0"
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Predice_compra</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>1541005</th>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1605463</th>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4733672</th>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2547680</th>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3180461</th>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "         Predice_compra\n",
       "1541005               0\n",
       "1605463               0\n",
       "4733672               0\n",
       "2547680               0\n",
       "3180461               0"
      ]
     },
     "execution_count": 50,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "y_test_pred.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 52,
   "metadata": {
    "azdata_cell_guid": "edd944ec-606c-4e90-b2bf-8a9a3e616ad4"
   },
   "outputs": [],
   "source": [
    "results_df = y_test.join(y_test_pred)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 53,
   "metadata": {
    "azdata_cell_guid": "de16de6d-2013-49b0-978d-2e0333b7d689"
   },
   "outputs": [],
   "source": [
    "results_df['Success'] = (results_df['Predice_compra'] == results_df[TARGET1]).astype(int)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 54,
   "metadata": {
    "azdata_cell_guid": "f7979794-03e2-4b92-8565-10b5f147f95b"
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Productos_pension_plan</th>\n",
       "      <th>Predice_compra</th>\n",
       "      <th>Success</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>1541005</th>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1605463</th>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4733672</th>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2547680</th>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3180461</th>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "         Productos_pension_plan  Predice_compra  Success\n",
       "1541005                       0               0        1\n",
       "1605463                       0               0        1\n",
       "4733672                       0               0        1\n",
       "2547680                       0               0        1\n",
       "3180461                       0               0        1"
      ]
     },
     "execution_count": 54,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "results_df.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 55,
   "metadata": {
    "azdata_cell_guid": "0d216b2f-c7ba-4398-aef9-47639b59cbe4"
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.9660292152092964"
      ]
     },
     "execution_count": 55,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "results_df['Success'].mean()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 56,
   "metadata": {
    "azdata_cell_guid": "8a8d7ebc-7df1-4b12-8d98-52e61cd2f44b"
   },
   "outputs": [],
   "source": [
    "confussion_matrix = pd.crosstab(results_df[TARGET1], results_df['Predice_compra'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 57,
   "metadata": {
    "azdata_cell_guid": "2bc85b5c-66e3-4539-9e38-f7ffda1ae4ce"
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th>Predice_compra</th>\n",
       "      <th>0</th>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Productos_pension_plan</th>\n",
       "      <th></th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>1040057</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>36574</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "Predice_compra                0\n",
       "Productos_pension_plan         \n",
       "0                       1040057\n",
       "1                         36574"
      ]
     },
     "execution_count": 57,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "confussion_matrix"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 58,
   "metadata": {
    "azdata_cell_guid": "aad4d3d2-bac2-4ff9-a731-cb146e02c84a"
   },
   "outputs": [
    {
     "ename": "IndexError",
     "evalue": "single positional indexer is out-of-bounds",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mIndexError\u001b[0m                                Traceback (most recent call last)",
      "\u001b[1;32m<ipython-input-58-902ad10cbdf9>\u001b[0m in \u001b[0;36m<module>\u001b[1;34m\u001b[0m\n\u001b[1;32m----> 1\u001b[1;33m \u001b[0mTP\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mconfussion_matrix\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0miloc\u001b[0m\u001b[1;33m[\u001b[0m\u001b[1;36m1\u001b[0m\u001b[1;33m,\u001b[0m\u001b[1;36m1\u001b[0m\u001b[1;33m]\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m      2\u001b[0m \u001b[0mTN\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mconfussion_matrix\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0miloc\u001b[0m\u001b[1;33m[\u001b[0m\u001b[1;36m0\u001b[0m\u001b[1;33m,\u001b[0m\u001b[1;36m0\u001b[0m\u001b[1;33m]\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m      3\u001b[0m \u001b[0mFP\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mconfussion_matrix\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0miloc\u001b[0m\u001b[1;33m[\u001b[0m\u001b[1;36m0\u001b[0m\u001b[1;33m,\u001b[0m\u001b[1;36m1\u001b[0m\u001b[1;33m]\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m      4\u001b[0m \u001b[0mFN\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mconfussion_matrix\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0miloc\u001b[0m\u001b[1;33m[\u001b[0m\u001b[1;36m1\u001b[0m\u001b[1;33m,\u001b[0m\u001b[1;36m0\u001b[0m\u001b[1;33m]\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;32m~\\anaconda3\\lib\\site-packages\\pandas\\core\\indexing.py\u001b[0m in \u001b[0;36m__getitem__\u001b[1;34m(self, key)\u001b[0m\n\u001b[0;32m    871\u001b[0m                     \u001b[1;31m# AttributeError for IntervalTree get_value\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    872\u001b[0m                     \u001b[1;32mpass\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m--> 873\u001b[1;33m             \u001b[1;32mreturn\u001b[0m \u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0m_getitem_tuple\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mkey\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m    874\u001b[0m         \u001b[1;32melse\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    875\u001b[0m             \u001b[1;31m# we by definition only have the 0th axis\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;32m~\\anaconda3\\lib\\site-packages\\pandas\\core\\indexing.py\u001b[0m in \u001b[0;36m_getitem_tuple\u001b[1;34m(self, tup)\u001b[0m\n\u001b[0;32m   1441\u001b[0m     \u001b[1;32mdef\u001b[0m \u001b[0m_getitem_tuple\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mself\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mtup\u001b[0m\u001b[1;33m:\u001b[0m \u001b[0mTuple\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m   1442\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m-> 1443\u001b[1;33m         \u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0m_has_valid_tuple\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mtup\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m   1444\u001b[0m         \u001b[1;32mtry\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m   1445\u001b[0m             \u001b[1;32mreturn\u001b[0m \u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0m_getitem_lowerdim\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mtup\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;32m~\\anaconda3\\lib\\site-packages\\pandas\\core\\indexing.py\u001b[0m in \u001b[0;36m_has_valid_tuple\u001b[1;34m(self, key)\u001b[0m\n\u001b[0;32m    700\u001b[0m                 \u001b[1;32mraise\u001b[0m \u001b[0mIndexingError\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;34m\"Too many indexers\"\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    701\u001b[0m             \u001b[1;32mtry\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m--> 702\u001b[1;33m                 \u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0m_validate_key\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mk\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mi\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m    703\u001b[0m             \u001b[1;32mexcept\u001b[0m \u001b[0mValueError\u001b[0m \u001b[1;32mas\u001b[0m \u001b[0merr\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    704\u001b[0m                 raise ValueError(\n",
      "\u001b[1;32m~\\anaconda3\\lib\\site-packages\\pandas\\core\\indexing.py\u001b[0m in \u001b[0;36m_validate_key\u001b[1;34m(self, key, axis)\u001b[0m\n\u001b[0;32m   1350\u001b[0m             \u001b[1;32mreturn\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m   1351\u001b[0m         \u001b[1;32melif\u001b[0m \u001b[0mis_integer\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mkey\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m-> 1352\u001b[1;33m             \u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0m_validate_integer\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mkey\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0maxis\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m   1353\u001b[0m         \u001b[1;32melif\u001b[0m \u001b[0misinstance\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mkey\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mtuple\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m   1354\u001b[0m             \u001b[1;31m# a tuple should already have been caught by this point\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;32m~\\anaconda3\\lib\\site-packages\\pandas\\core\\indexing.py\u001b[0m in \u001b[0;36m_validate_integer\u001b[1;34m(self, key, axis)\u001b[0m\n\u001b[0;32m   1435\u001b[0m         \u001b[0mlen_axis\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mlen\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mobj\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0m_get_axis\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0maxis\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m   1436\u001b[0m         \u001b[1;32mif\u001b[0m \u001b[0mkey\u001b[0m \u001b[1;33m>=\u001b[0m \u001b[0mlen_axis\u001b[0m \u001b[1;32mor\u001b[0m \u001b[0mkey\u001b[0m \u001b[1;33m<\u001b[0m \u001b[1;33m-\u001b[0m\u001b[0mlen_axis\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m-> 1437\u001b[1;33m             \u001b[1;32mraise\u001b[0m \u001b[0mIndexError\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;34m\"single positional indexer is out-of-bounds\"\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m   1438\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m   1439\u001b[0m     \u001b[1;31m# -------------------------------------------------------------------\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;31mIndexError\u001b[0m: single positional indexer is out-of-bounds"
     ]
    }
   ],
   "source": [
    "TP = confussion_matrix.iloc[1,1]\n",
    "TN = confussion_matrix.iloc[0,0]\n",
    "FP = confussion_matrix.iloc[0,1]\n",
    "FN = confussion_matrix.iloc[1,0]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 46,
   "metadata": {
    "azdata_cell_guid": "23be7dd7-b0a0-4f0e-8d78-680fc570fd86"
   },
   "outputs": [
    {
     "ename": "NameError",
     "evalue": "name 'TP' is not defined",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mNameError\u001b[0m                                 Traceback (most recent call last)",
      "\u001b[1;32m<ipython-input-46-7e04f3739e47>\u001b[0m in \u001b[0;36m<module>\u001b[1;34m\u001b[0m\n\u001b[1;32m----> 1\u001b[1;33m \u001b[0mPrecision\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mTP\u001b[0m \u001b[1;33m/\u001b[0m \u001b[1;33m(\u001b[0m\u001b[0mTP\u001b[0m \u001b[1;33m+\u001b[0m \u001b[0mFP\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m      2\u001b[0m \u001b[0mRecall\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mTP\u001b[0m\u001b[1;33m/\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mTP\u001b[0m \u001b[1;33m+\u001b[0m \u001b[0mFN\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;31mNameError\u001b[0m: name 'TP' is not defined"
     ]
    }
   ],
   "source": [
    "Precision = TP / (TP + FP)\n",
    "Recall = TP/(TP + FN)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 234,
   "metadata": {
    "azdata_cell_guid": "e73bd8c8-e21c-4447-b408-59706faf73eb"
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.9813972410222271"
      ]
     },
     "execution_count": 234,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "Precision"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 235,
   "metadata": {
    "azdata_cell_guid": "0e852a35-c26e-4f68-abb3-f383ed7938b8"
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.7255427352764259"
      ]
     },
     "execution_count": 235,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "Recall"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 289,
   "metadata": {
    "azdata_cell_guid": "ddccb47d-7091-4ce5-81b4-1915c718739c"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Profundidad del árbol 1. accuracy en train: 0.9657136474799629 accuracy en test: 0.9660292152092964\n",
      "Profundidad del árbol 2. accuracy en train: 0.9657136474799629 accuracy en test: 0.9660292152092964\n",
      "Profundidad del árbol 3. accuracy en train: 0.9657136474799629 accuracy en test: 0.9660292152092964\n",
      "Profundidad del árbol 4. accuracy en train: 0.9657136474799629 accuracy en test: 0.9660292152092964\n"
     ]
    }
   ],
   "source": [
    "for i in range(1,5):\r\n",
    "    dt2 = DecisionTreeClassifier( max_depth= i, min_samples_leaf=50,random_state=42)\r\n",
    "    dt2.fit(X_train,y_train)\r\n",
    "    train_accuracy = dt2.score(X_train, y_train)\r\n",
    "    test_accuracy = dt2.score(X_test, y_test)\r\n",
    "    print('Profundidad del árbol {}. accuracy en train: {} accuracy en test: {}' .format(i,train_accuracy,test_accuracy))\r\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 60,
   "metadata": {
    "azdata_cell_guid": "ed2c648d-7180-4a0d-a8de-f6118d028d5d"
   },
   "outputs": [],
   "source": [
    "dt_final = DecisionTreeClassifier(max_depth= 4, min_samples_leaf=50,random_state=42)\r\n",
    "dt_final.fit(X_train,y_train)\r\n",
    "train_accuracy = dt_final.score(X_train, y_train)\r\n",
    "test_accuracy = dt_final.score(X_test, y_test)\r\n",
    "val_accuracy = dt_final.score(val_df_X, val_df_y)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 61,
   "metadata": {
    "azdata_cell_guid": "a8b86edc-ecc8-44fa-a76a-1660603e7cca"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      " acc en train: 0.9657136474799629 acc en test: 0.9660292152092964 y acc en val 0.9614707073860148\n"
     ]
    }
   ],
   "source": [
    "print(' acc en train: {} acc en test: {} y acc en val {}' .format(train_accuracy,test_accuracy, val_accuracy))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 62,
   "metadata": {
    "azdata_cell_guid": "e7066304-9877-4862-ae4a-4da1c380fc9c"
   },
   "outputs": [],
   "source": [
    "y_test_pred_proba = pd.DataFrame(dt.predict_proba(X_test)[:,1], index= y_test.index, columns = ['Scoring'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 63,
   "metadata": {
    "azdata_cell_guid": "d4ee1a9e-6a10-4f46-b537-261eb8dea063"
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Scoring</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>1541005</th>\n",
       "      <td>0.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1605463</th>\n",
       "      <td>0.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4733672</th>\n",
       "      <td>0.070576</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2547680</th>\n",
       "      <td>0.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3180461</th>\n",
       "      <td>0.070576</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>...</th>\n",
       "      <td>...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4783843</th>\n",
       "      <td>0.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2940980</th>\n",
       "      <td>0.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>5338820</th>\n",
       "      <td>0.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1160696</th>\n",
       "      <td>0.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>243117</th>\n",
       "      <td>0.070576</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>1076631 rows × 1 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "          Scoring\n",
       "1541005  0.000000\n",
       "1605463  0.000000\n",
       "4733672  0.070576\n",
       "2547680  0.000000\n",
       "3180461  0.070576\n",
       "...           ...\n",
       "4783843  0.000000\n",
       "2940980  0.000000\n",
       "5338820  0.000000\n",
       "1160696  0.000000\n",
       "243117   0.070576\n",
       "\n",
       "[1076631 rows x 1 columns]"
      ]
     },
     "execution_count": 63,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "y_test_pred_proba"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 64,
   "metadata": {
    "azdata_cell_guid": "fe57f6d2-e59a-4fa3-9937-d7859f1fa273"
   },
   "outputs": [],
   "source": [
    "results_df['Scoring'] = y_test_pred_proba"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 65,
   "metadata": {
    "azdata_cell_guid": "591f9a8a-e8b0-4a46-aaa9-d351721520ee"
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.9577177980434687"
      ]
     },
     "execution_count": 65,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "metrics.roc_auc_score(results_df[TARGET1], results_df['Scoring'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 66,
   "metadata": {
    "azdata_cell_guid": "f052ee2b-e442-425a-8fb0-0aa4aca97c4e"
   },
   "outputs": [],
   "source": [
    "fpr, tpr, _ = metrics.roc_curve(results_df[TARGET1], results_df['Scoring'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 67,
   "metadata": {
    "azdata_cell_guid": "58c1dde3-d048-412e-83fa-92fe362b921a"
   },
   "outputs": [
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAYIAAAEWCAYAAABrDZDcAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjMuMiwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy8vihELAAAACXBIWXMAAAsTAAALEwEAmpwYAAAsEklEQVR4nO3deXxVZZ7n8c+PLCQQQoCwQwgBRFFWAyiyI4atCrW1xLXK6npZdut01yw1VdMz3dU9NUvN9FbltLbj1DjVVTPVoBZYl0VRkU0RBGSRxWAAIQGUAAlL1pvkmT/uBWNMIIGce3Lv+b5fL17m3nPI/R1J7vc+z3nO75hzDhERCa5OfhcgIiL+UhCIiAScgkBEJOAUBCIiAacgEBEJOAWBiEjAKQhERAJOQSAJx8w+M7MqM7tkZp+b2a/MLKPJPlPM7F0zu2hm581spZmNarJPppn93MyOR79XUfRxdmyPSMRbCgJJVN9wzmUA44DxwL+7vMHM7gTeAn4PDACGAnuA980sL7pPKrAOuBWYB2QCU4CzwCSvijazZK++t0hLFASS0JxznwNriQTCZf8d+LVz7hfOuYvOuXPOuf8AbAX+MrrPE0AOcJ9z7oBzrsE5d9o591Pn3JrmXsvMbjWzt83snJl9YWZ/Fn3+V2b2nxrtN9PMSho9/szMfmRme4EKM/sPZvZak+/9CzN7Lvp1dzP732Z2ysxOmNl/MrOkG/s/JUGmIJCEZmaDgPlAUfRxFyKf7F9tZvdXgLnRr+8G3nTOXWrl63QD3gHeJDLKGE5kRNFaDwMLgSzgN8ACM8uMfu8k4FvAb6P7/hNQF32N8cA9wPfa8FoiX6EgkET1upldBIqB08BPos/3JPJzf6qZv3MKuDz/36uFfVqyCPjcOfe3zrnq6EhjWxv+/nPOuWLnXJVz7hjwEXBvdNtsoNI5t9XM+hIJth845yqcc6eBvweWtOG1RL5CQSCJ6l7nXDdgJnAzX77BlwENQP9m/k5/4Ez067Mt7NOSwcDh66o0orjJ498SGSUAPMKXo4EhQApwyszKzawc+J9Anxt4bQk4BYEkNOfcRuBXwN9EH1cAHwAPNrP7t/hyOucdoMDMurbypYqBYS1sqwC6NHrcr7lSmzx+FZgZndq6jy+DoBioAbKdc1nRP5nOuVtbWafI1ygIJAh+Dsw1s3HRxz8Gvm1mf2Jm3cysR/Rk7p3AX0X3+Q2RN93fmdnNZtbJzHqZ2Z+Z2YJmXmMV0M/MfmBmnaPfd3J0224ic/49zawf8INrFeycKwU2AP8HOOqcOxh9/hSRFU9/G13e2snMhpnZjDb+PxG5QkEgCS/6pvpr4M+jj98DCoD7iZwHOEbkpOtU59yn0X1qiJww/gR4G7gAfEhkiulrc//OuYtETjR/A/gc+BSYFd38GyLLUz8j8ia+rJWl/zZaw2+bPP8EkAocIDLV9Rptm8YS+QrTjWlERIJNIwIRkYBTEIiIBJyCQEQk4BQEIiIBF3cNrrKzs11ubq7fZYiIxJWdO3eecc71bm5b3AVBbm4uO3bs8LsMEZG4YmbHWtqmqSERkYBTEIiIBJyCQEQk4BQEIiIBpyAQEQk4z4LAzF42s9Nmtq+F7WZmz0VvCL7XzCZ4VYuIiLTMyxHBr4jc9Lsl84ER0T9PAf/oYS0iItICz64jcM5tMrPcq+yymMgNxB2w1cyyzKx/tN+6tKO6+gaKy6o4UnqJI6UVXKwO+12SiLSBa2jA1Vxi0i25TL+p2WvCboifF5QN5Ku35yuJPve1IDCzp4iMGsjJyYlJcfHGOce5ilqOnKm48oZ/uLSCo2cucfxcJeH6L9uNm/lYqIi0SU+r5K6Uo6RZHVuSv5lwQdDc21GzN0dwzr0EvASQn58f+BsoVNTUsfnTUg6XVnCktIIjZyJv/Oervvykn5rUidzsLozo042CW/uR1zuDvN5dycvuSlaXVB+rF5HWqKurY8OGDWzZspMuXbqwcOF93HLLLZ68lp9BUELkht+XDQJO+lRL3Nh5rIwfLNtF8bkqAPpmdiYvO4NFY/pfebMflp3BwB7pJHXSR3+ReLV06VIOHz7MuHHjuOeee0hPT/fstfwMghDwrJktBSYD53V+oGV19Q38j3eL+If1RfTvnsY/fXcStw/pQUbnuGsXJSItqKmpISkpieTkZKZOncqdd97JsGHDPH9dz95FzOyfgZlAtpmVAD8BUgCccy8Ca4AFQBFQCTzpVS3x7vjZSn6wbBcfHS/n/vED+avFt9ItLcXvskSkHRUVFbFq1SpGjx7NnDlziGWXZS9XDT18je0OeMar108Ezjl+99EJfvL7fXTqZDz38Hi+OXaA32WJSDuqqqpi7dq17Nmzh+zsbG666aaY16B5hQ6qqraef/PaHlbvPcWkoT35+4fGMTDLuzlCEYm9I0eOsHz5cqqqqpg2bRrTp08nOTn2b8sKgg7IOccPX9vDmo9P8cOCkTw9Y5hO/IokoK5du9KjRw8ee+wx+vXr51sdCoIOaNn2YlbtjYTAM7OG+12OiLQT5xx79uzh1KlTzJ8/n759+/Ld734X8/niHgVBB1N0+hJ/tfIAdw3vxR/N8H61gIjERllZGatWreLIkSPk5OQQDodJSUnxPQRAQdCh1NTV86dLd5GW0om/+9Y4Omk6SCTuNTQ0sH37dtatW4eZsWDBAvLz8ztEAFymIOhA/mZtIftPXuB/PZFP38w0v8sRkXZQWVnJ+vXrGTJkCIsWLaJ79+5+l/Q1CoIOYtOhUv7X5qM8fscQ5o7q63c5InID6uvr+fjjjxk7diwZGRl8//vfJysrq0ONAhpTEHQAZy7V8K9e2cOIPhn8+4Xe9BIRkdg4efIkoVCIL774goyMDIYPH06PHj38LuuqFAQ+c87xb1/by4XqML/5w0mkpST5XZKIXIdwOMzGjRvZsmULXbt25aGHHmL48PhY9acg8NmvPzjGu5+c5i+/MYpb+mf6XY6IXKdly5Zx+PBhxo8fzz333ENaWvyc51MQ+OiTzy/wn9ccZNbI3nx7Sq7f5YhIGzVtEjdlyhTy8vL8LqvNFAQ+qQ7X8yf/vIvMtBT++sGxHfYkkog079NPP2XVqlWMGTMm5k3i2puCwCf/Zc1BDn1xiX/67iSyMzr7XY6ItFJlZSVr165l79699O7dm5EjR/pd0g1TEPjgnQNf8OsPjvG9qUOZ4cFt50TEG4cPH2b58uVUV1czffp0pk2b5kuTuPYW/0cQZ764UM0PX9vDqP6Z/HBe/H+SEAmSbt260atXLxYuXEjfvolzvU8nvwsIkoYGx79+ZQ9V4Xqee3g8nZO1VFSkI3PO8dFHH7F69WoA+vTpw5NPPplQIQAaEcTUL987wntFZ/iv949meJ8Mv8sRkasoKytj5cqVHD16lNzc3A7VJK69KQhi5OOS8/z12kLm3dqPJRMH+12OiLSgoaGBbdu28e6779KpUycWLVrEhAkTEjIALlMQxEBFTR1/snQXvbp25md/MDqhf6BE4l1lZSUbN24kLy+PhQsXkpmZ+Bd6Kghi4D+uPMBnZyv47ffuIKtLqt/liEgT9fX17N27l3HjxpGRkcHTTz9N9+7dA/OhTUHgsdV7T7FsRzHPzBrGncN6+V2OiDRx4sQJQqEQp0+fJjMzk2HDhpGVleV3WTGlIPDQifIq/t3yvYwdnMUP7r7J73JEpJFwOMz69evZunUrGRkZLFmyhGHDgnlXQAWBR+obHP9y2W7qGxzPLRlHSpJW6op0JEuXLuXIkSNMmDCBuXPnxlWTuPamIPDIP24o4sOj5/i7b41lSK+ufpcjIkB1dTXJyckkJyczffp0pk6dytChQ/0uy3cKAg98dLyMv3/nUxaPG8B94wf6XY6IAIcOHbrSJO7uu+9myJAhfpfUYSgI2tnF6jB/unQX/bun8dN7bwvMqgORjqqiooI333yTffv20adPH265RXcBbEpB0M7+4vf7OVFWxatP30lmWorf5YgEWuMmcTNnzmTq1KkkJam1S1MKgna0YlcJK3ad4F/efRO3D+npdzkigdetWzeys7NZuHAhffr08bucDktLWdrJ8bOV/Pnr+5mY24NnZgVzCZqI35xz7Ny5k1WrVgFfNolTCFydRgTtIFzfwJ8s3YUZ/P1D40jWUlGRmDt37hwrV67ks88++0qTOLk2BUE7eG7dp+wuLucfHhnPoB5d/C5HJFAaGhrYunUr69evJykpiW984xuMHz9eCzXawNMgMLN5wC+AJOCXzrmfNdneHfi/QE60lr9xzv0fL2tqb/tOnOf59UU8cPsgFo0Z4Hc5IoFTWVnJ5s2bGTZsGAsWLAhEk7j25lkQmFkS8DwwFygBtptZyDl3oNFuzwAHnHPfMLPeQKGZ/T/nXK1XdbW3n646QM+uqfz5olF+lyISGHV1dezZs4cJEyaQkZHB97///UA1iWtvXo4IJgFFzrkjAGa2FFgMNA4CB3SzyL9eBnAOqPOwpnZ1+kI1246e44cFI+merrlIkVgoKSkhFApRWlpKVlZWIJvEtTcvg2AgUNzocQkwuck+/wCEgJNAN+Ah51xD029kZk8BTwHk5OR4Uuz12HCoFIBZI7UiQcRrtbW1V5rEZWZm8sgjjwS2SVx78zIImhujuSaPC4DdwGxgGPC2mW12zl34yl9y7iXgJYD8/Pym38M3GwtL6ZvZmVv6d/O7FJGEt2zZMo4cOUJ+fj533303nTt39rukhOFlEJQAje/JOIjIJ//GngR+5pxzQJGZHQVuBj70sK52UVffwOZPS5l3Wz/NS4p4pLq6mqSkJFJSUpg+fTrTp09XjyAPeLngfTswwsyGmlkqsITINFBjx4E5AGbWFxgJHPGwpnazq7icC9V1zNS0kIgnCgsLeeGFF9i4cSMAQ4YMUQh4xLMRgXOuzsyeBdYSWT76snNuv5k9Hd3+IvBT4Fdm9jGRqaQfOefOeFVTe9pYWEpSJ+Ou4dl+lyKSUCoqKnjjjTfYv38/ffv2ZdQorcjzmqfXETjn1gBrmjz3YqOvTwL3eFmDVzYcOs2EnCytFhJpR0VFRSxfvpza2lpmzZrFXXfdpSZxMaAri6/D6YvV7DtxgR8WjPS7FJGEkpmZSZ8+fVi4cCG9e/f2u5zAUFOc67DpUGT2asZN+kEVuRHOObZv387KlSuBSJO473znOwqBGNOI4DpsKDxN726duXWALmUXuV5nz54lFApx/Phx8vLyqKurIzlZb0l+0P/1NoosGz3D3FF9tWxU5Do0NDSwZcsWNmzYQEpKCosXL2bs2LH6ffKRgqCN9pSUc74qzMyRGrqKXI/Kykref/99RowYwYIFC+jWTRdk+k1B0EYbCkvpZDBtuIJApLXq6urYvXs3t99+OxkZGTz99NN0797d77IkSkHQRhsKSxmf04PuXbRsVKQ1iouLCYVCnDlzhp49e5KXl6cQ6GAUBG1QerGGj0+c51/PvcnvUkQ6vNraWt599122bdtG9+7defTRR8nLy/O7LGmGgqANNn8a6TaqthIi17Z06VKOHj3KxIkTmTNnjprEdWAKgjbYUFhKdkaqlo2KtKCqqork5GRSUlKYOXMmM2fO7FCt46V5CoJWqm9wbPq0lNk396FTJy1zE2nq4MGDrFmzhjFjxjB37lwFQBxRELTSnpJyyivDmhYSaeLSpUusWbOGgwcP0q9fP2677Ta/S5I2UhC00uVlo9NHqNuoyGWffvopy5cvJxwOM3v2bKZMmaImcXFIQdBKGwtPM25wFlldUv0uRaTDyMrKon///ixYsIDsbH1IildqOtcKZy/VsPfEeU0LSeA55/jwww8JhSL3mOrduzdPPPGEQiDOaUTQCps+LcU5dRuVYDtz5gyhUIji4mKGDRumJnEJRP+KrbChsJReXVMZPVBXQ0rw1NfXs2XLFjZu3KgmcQlKQXAN9Q2OTYdKmTlSy0YlmKqrq9myZQsjR45k/vz5ZGRk+F2StDMFwTV8fOI8ZZXqNirBUldXx65du8jPz6dr16780R/9EZmZupAyUSkIrmFD4WnMYNoIBYEEw/HjxwmFQpw9e5ZevXqRl5enEEhwCoJr2FBYythBWfTsqmWjkthqampYt24d27dvJysri8cee0xN4gJCQXAV5ypq2VNSzp/OGeF3KSKeW7ZsGUePHmXy5MnMnj2b1FR9+AkKBcFVbI4uG9X1A5KoGjeJmzVrFrNmzWLw4MF+lyUxpiC4ig2FpfTsmsoYLRuVBHTgwAHWrFnD2LFjmTt3rgIgwBQELWiILhudNiJby0YloVy8eJE1a9bwySef0L9/f0aPHu13SeIzBUELPj5xnrMVtVo2Kgnl0KFDrFixgrq6Ou6++27uvPNOOnVSp5mgUxC0YENhKWYwXctGJYH06NGDAQMGsGDBAnr16uV3OdJB6KNACzYcOs2Ygd3plaHb60n8amhoYOvWrfz+978HIk3iHn/8cYWAfIVGBM0oq6hlT3E5z87WslGJX6WlpYRCIUpKShgxYoSaxEmL9FPRjM1FZ2hw6PyAxKX6+nref/99Nm3aRGpqKvfddx+jR49WkzhpkadBYGbzgF8AScAvnXM/a2afmcDPgRTgjHNuhpc1tcaGwtP06JLC2EFZfpci0mbV1dVs3bqVm2++mfnz59O1a1e/S5IOzrMgMLMk4HlgLlACbDezkHPuQKN9soAXgHnOueNm5vuVW18uG+1NkpaNSpwIh8Ps2rWLiRMnXmkS161bN7/Lkjjh5YhgElDknDsCYGZLgcXAgUb7PAIsd84dB3DOnfawnlbZf/ICZy7V6iY0EjeOHTtGKBTi3LlzZGdnk5eXpxCQNvEyCAYCxY0elwCTm+xzE5BiZhuAbsAvnHO/bvqNzOwp4CmAnJwcT4q9bENhJIumKwikg6upqeGdd95hx44dZGVl8fjjj6tJnFwXL4OguXkV18zr3w7MAdKBD8xsq3Pu0Ff+knMvAS8B5OfnN/0e7WrDoVJGD+xO725aNiod29KlS/nss8+44447mDVrlprEyXXzMghKgMbNSwYBJ5vZ54xzrgKoMLNNwFjgED4or6xl1/Eynpk13I+XF7mmyspKUlJSSElJYfbs2ZgZgwYN8rssiXNeXlC2HRhhZkPNLBVYAoSa7PN7YJqZJZtZFyJTRwc9rOmqNn+qZaPSMTnn2LdvH88//zzr168HYPDgwQoBaReejQicc3Vm9iywlsjy0Zedc/vN7Ono9hedcwfN7E1gL9BAZInpPq9qupYNhaV0T09h3OAefpUg8jUXLlxgzZo1FBYWMmDAAMaOHet3SZJgPL2OwDm3BljT5LkXmzz+a+CvvayjtXYXlzFpaE8tG5UO49ChQyxfvpz6+nrmzp3LHXfcoSZx0u50ZXEjZytqmTIsze8yRK7o2bMngwcPZv78+fTs2dPvciRB6aNFVF19A+erwro3sfiqoaGBDz74gNdffx2A7OxsHn30UYWAeEojgqjyqjDOoSAQ35w+fZpQKMSJEyfUJE5iSj9lUecqagEFgcRefX097733Hps2bSItLY3777+f2267TU3iJGYUBFEKAvFLdXU127Zt49Zbb6WgoEBN4iTmFARRCgKJpXA4zM6dO5k0aZKaxInv2hwE0a6iS5xz/8+DenxzOQh6KQjEY0ePHmXlypWUlZXRp08fNYkT37UYBGaWCTxDpHlcCHgbeBb4N8BuICGDIKuLgkC8UV1dzdtvv81HH31Ejx49+Pa3v01ubq7fZYlcdUTwG6AM+AD4HvBDIBVY7Jzb7X1psXWuopZuacmkJmtFrXhj2bJlHDt2jClTpjBz5kxSUlL8LkkEuHoQ5DnnRgOY2S+BM0COc+5iTCqLsXMVtTo/IO2uoqKC1NRUUlJSmDNnDmbGwIED/S5L5CuuFgThy1845+rN7GiihgAoCKR9XW4S98YbbzBu3DjuueceNYiTDutqQTDWzC7w5X0F0hs9ds65TM+ri6FzFbX07672EnLjLly4wOrVqzl06BADBw5k3LhxfpckclUtBoFzLimWhfjtXEUttw5IqGwTHxQWFrJ8+XKccxQUFDBp0iQ1iZMO72qrhtKAp4HhRNpEv+ycq4tVYbHknONcpaaG5Mb16tWLnJwcFixYQI8eamcu8eFqH1X+CcgHPgYWAH8bk4p8UFFbT21dg4JA2qyhoYEtW7awYsUK4MsmcQoBiSdXO0cwqtGqof8NfBibkmKvTFcVy3X44osvCIVCnDx5kpEjR6pJnMSt1q4aqkvkBlhnFQTSBnV1dWzevJn33nuP9PR0HnjgAUaNGqUmcRK3rhYE46KrhCCyUihhVw2dq6gBFATSOjU1NezYsYPbbruNgoICunTp4ndJIjfkakGwxzk3PmaV+OhcRWTwoyCQltTW1rJz504mT558pUlcRkaG32WJtIurBYGLWRU+K69UnyFp2ZEjR1i5ciXl5eX069ePoUOHKgQkoVwtCPqY2b9qaaNz7u88qMcXZZW1JHUyMtN0ok++VF1dzVtvvcWuXbvo2bMn3/nOdxgyZIjfZYm0u6u98yUBGXx5ZXHCKqsMk5WeopN98hWXm8TdddddzJgxQ03iJGFdLQhOOef+Y8wq8VFZRS1ZXfRLLnDp0iVSU1NJTU1lzpw5dOrUiQEDBvhdloinrhYEgfl4XKarigPPOcfevXtZu3atmsRJ4FwtCObErAqflVeGGdxTSwCD6vz586xatYqioiIGDRrE+PGBWCwncsXVms6di2UhfiqrrGXMoO5+lyE++OSTT1ixYgXOOebNm8fEiRPVJE4CJ/DLZJxzlFWG6aGlo4HinMPMyM7OJjc3l/nz55OVleV3WSK+CPxHn6pwpOGcriEIhoaGBt57772vNIl7+OGHFQISaIEfEZRVRq4q7qFVQwnv888/JxQKcerUKW6++WY1iROJCvxvweXOoxoRJK66ujo2bdrE+++/T3p6Og8++CCjRo3yuyyRDiPwQVCuEUHCq6mpYefOnYwePZqCggLS09P9LkmkQ/H0HIGZzTOzQjMrMrMfX2W/iWZWb2YPeFlPc8qifYZ66DqChFJbW8uWLVtoaGiga9eu/PEf/zH33nuvQkCkGZ6NCMwsCXgemAuUANvNLOScO9DMfv8NWOtVLVdTdqXhnEYEieLw4cOsXLmS8+fP079/f4YOHUrXrl39Lkukw/JyamgSUOScOwJgZkuBxcCBJvv9C+B3wEQPa2lRWbQFdVa6RgTxrqqqirfeeovdu3fTq1cvnnzySXJycvwuS6TD8zIIBgLFjR6XAJMb72BmA4H7gNlcJQjM7CngKaDdf7HLKmvp1jmZ1OTAr6SNe8uWLeP48eNMnTqVGTNmaEWQSCt5+ZvSXK+ipvc4+DnwI+dc/dU6fzrnXgJeAsjPz2/X+ySUV9aS1VXTQvGqcZO4uXPnkpSURL9+/fwuSySueBkEJcDgRo8HASeb7JMPLI2GQDawwMzqnHOve1jXV+iq4vjknGPPnj1XmsQVFBQwcOBAv8sSiUteBsF2YISZDQVOAEuARxrv4JwbevlrM/sVsCqWIQDREYGCIK6Ul5ezatUqDh8+TE5ODrfffrvfJYnENc+CwDlXZ2bPElkNlAS87Jzbb2ZPR7e/6NVrt0VZZZjcbK0oiRcHDx5kxYoVmBnz589n4sSJuqGQyA3y9Gyac24NsKbJc80GgHPuO17W0pKyylpNDcWBy03i+vTpQ15eHvPmzVN/IJF2EuilMnX1DVysrtM1BB1YfX09mzdvZvny5QD06tWLJUuWKARE2lGg19eVV11uL6ERQUd06tQpQqEQn3/+ObfeequaxIl4JNC/VeW6qrhDCofDbNy4kS1bttC1a1ceeughbr75Zr/LEklYgQ6CcxUaEXRE4XCYXbt2MXbsWO655x71BxLxWKCD4HKfId243n81NTXs2LGDO++8ky5duvDMM8/QpYvuIy0SC4EOAk0NdQxFRUWsWrWK8+fPM3DgQHJzcxUCIjEU6CD48u5kGhH4obKykrfeeos9e/aQnZ3Nd7/7XQYPHnztvygi7SrgQVBLalInuqQm+V1KIL3yyisUFxczffp0pk2bphVBIj4J9G9eeUWYrC4pujI1hi5evEjnzp3VJE6kAwl0EOiq4thxzrF7927Wrl3L+PHj1SROpAMJdBCUV4Z1ojgGysrKWLVqFUeOHGHIkCHk5+f7XZKINBLoICirrGVY7wy/y0hojZvELVy4kNtvv11TcSIdTMCDIEwP3ZTGE42bxA0fPpyCggK6d+/ud1ki0ozANp1zzuleBB6or69n06ZNLF++HOccvXr14lvf+pZCQKQDC+yI4GJNHXUNjp4KgnZz8uRJQqEQX3zxBbfddhv19fVaEioSBwL7W1oe7TOkk8U3LhwOs2HDBj744AMyMjJYsmQJI0eO9LssEWmlwAbB5T5DWj5648LhMLt372b8+PHMnTuXtLQ0v0sSkTZQEOhk8XWpqalh+/btTJkyRU3iROJcYIOgvPLy1JBGBG116NAhVq9ezcWLFxk0aJCaxInEucAGgaaG2q6iooK1a9fy8ccf07t3bx588EEGDRrkd1kicoMCHARhzKB7uqaGWuuVV16hpKSEGTNmMG3aNJKS1KxPJBEENgjKK2vJTEshqZOucr2aCxcukJaWRmpqKgUFBSQnJ9OnTx+/yxKRdhTYICirDNNDS0db5Jzjo48+4u23377SJG7AgAF+lyUiHghsEOiq4padO3eOlStX8tlnn5Gbm8vEiRP9LklEPBTYIDhXUUufbp39LqPDOXDgACtWrCApKYlFixYxYcIENYkTSXCBDYLyyjAj+3Xzu4wO43KTuL59+3LTTTdRUFBAZmam32WJSAwEtumcbkoTUV9fz4YNG/jd7353pUncgw8+qBAQCZBAjghq6uqprK0P/MniEydOEAqFOH36NKNHj1aTOJGACuRvfdCvKg6Hw6xfv56tW7eSkZHBww8/zE033eR3WSLik0AGQdCvKg6Hw+zdu5cJEyYwd+5cOnfWSXORIPP0HIGZzTOzQjMrMrMfN7P9UTPbG/2zxczGelnPZWXRFtRBmhqqrq5m06ZNNDQ0XGkSt2jRIoWAiHg3IjCzJOB5YC5QAmw3s5Bz7kCj3Y4CM5xzZWY2H3gJmOxVTZeVR0cEQZkaKiwsZPXq1Vy6dImcnBxyc3NJT0/3uywR6SC8nBqaBBQ5544AmNlSYDFwJQicc1sa7b8ViEkHs7LoOYJEb0FdUVHBm2++yb59++jTpw9LlizR1cEi8jVeBsFAoLjR4xKu/mn/D4E3mttgZk8BTwHk5OTccGFBOUdwuUnczJkzmTp1qprEiUizvAyC5i5Hdc3uaDaLSBBMbW67c+4lItNG5OfnN/s92qK8spa0lE6kpSTeG2PjJnHz5s0jKSlJTeJE5Kq8DIISYHCjx4OAk013MrMxwC+B+c65sx7Wc8W5inDC3bTeOcfOnTuvNImbN28e/fv397ssEYkDXgbBdmCEmQ0FTgBLgEca72BmOcBy4HHn3CEPa/mKRGs4d/bsWVauXMmxY8cYOnQokyd7fr5dRBKIZ0HgnKszs2eBtUAS8LJzbr+ZPR3d/iLwF0Av4IVoY7M651y+VzVdVlZZmzAnivfv38/rr79OUlIS3/zmNxk3bpyaxIlIm3h6QZlzbg2wpslzLzb6+nvA97ysoTnllWH6Z8X38snLTeL69+/PyJEjKSgooFs3NdETkbYLZNO5SMO5+BwR1NXVsX79el577TWcc/Ts2ZMHHnhAISAi1y1wLSYaGhznq8JxuXS0pKSEUChEaWkpY8aMUZM4EWkXgXsXuVAdpsHF11XFtbW1vPvuu2zbto3MzEweeeQRRowY4XdZIpIgAhcEV64qjqOpobq6Ovbv38/EiROZM2eO+gOJSLsKYBDEx1XF1dXVbNu2jWnTpl1pEpeWluZ3WSKSgAIXBF82nOu4I4JPPvmE1atXU1FRQW5uLkOGDFEIiIhnAhcEl1tQ9+za8UYEly5d4o033uDAgQP07duXhx9+WE3iRMRzwQuCDtyC+tVXX+XEiRPMmjWLu+66S03iRCQmAhkESZ2MzLSOcejnz58nLS2Nzp07M2/ePJKTk+ndu7ffZYlIgHSMd8MYKqsMk5We4nsbBucc27dvZ926dWoSJyK+ClwQRBrO+Xui+MyZM6xcuZLjx4+Tl5fHHXfc4Ws9IhJsgQuCsgp/ryrev38/K1asICUlhcWLFzN27FjfRyciEmzBC4LKWgb16BLz123cJO6WW26hoKCAjIyMmNchItJU4JrOlVeGY3pVcV1dHevWrePVV1+90iTuD/7gDxQCItJhBHJE0CNG1xAUFxcTCoU4c+YMY8eOVZM4EemQAvWuVFVbT01dg+cni2tra1m3bh0ffvgh3bt359FHH2X48OGevqaIyPUKVBDEqs9QfX09Bw4cUJM4EYkLgQqCcxXeBUFVVRXbtm1j+vTppKenq0mciMSNQAVBuUctqA8cOMCaNWuorKxk6NChahInInElUEFwZWqonU4WX7x4kTfeeIODBw/Sr18/HnvsMfr169cu31tEJFYCFQTt3YL6tdde48SJE8yZM4cpU6bQqVPgVuOKSAIIVBBcvjtZVvr1jwjKy8tJT0+nc+fOzJ8/n+TkZLKzs9urRBGRmAtYENSS0TmZ1OS2f3J3zvHhhx+ybt06JkyYwLx58zQNJCIJIVBBUF4Zvq5poTNnzhAKhSguLmb48OFqEiciCSVQQVBWWdvmpaP79u3j9ddfJzU1lXvvvZcxY8aoSZyIJJSABUHrRwSXm8QNGDCAUaNGcc8996g/kIgkpEAtczlfWXvNW1SGw2HeeecdXnnllStN4u6//36FgIgkrECNCM5Xheme3vIhHzt2jJUrV3L27FnGjx9PQ0OD7hssIgkvMEHgnONCdR3d078+NVRTU8M777zDjh07yMrK4vHHHycvL8+HKkVEYi8wQVBRW099g2s2CBoaGigsLGTy5MnMnj2b1FT/7mAmIhJrgQmCC1WRi8ky0yJBUFlZybZt25gxY8aVJnHqEioiQeTpyWIzm2dmhWZWZGY/bma7mdlz0e17zWyCV7WcvxIEyezfv58XXniB9957j+LiYgCFgIgElmcjAjNLAp4H5gIlwHYzCznnDjTabT4wIvpnMvCP0f+2u/NVYdKp5bPt77C95Cj9+/dXkzgREbydGpoEFDnnjgCY2VJgMdA4CBYDv3bOOWCrmWWZWX/n3Kn2LuZCVZhZqUc4e6qKu+++mzvvvFNN4kRE8DYIBgLFjR6X8PVP+83tMxD4ShCY2VPAUwA5OTnXVUyvjFQ65UzgW3NHMnLIgOv6HiIiicjLIGiuD4O7jn1wzr0EvASQn5//te2tcfuQntz+vVnX81dFRBKal3MjJcDgRo8HASevYx8REfGQl0GwHRhhZkPNLBVYAoSa7BMCnoiuHroDOO/F+QEREWmZZ1NDzrk6M3sWWAskAS875/ab2dPR7S8Ca4AFQBFQCTzpVT0iItI8Ty8oc86tIfJm3/i5Fxt97YBnvKxBRESuTusnRUQCTkEgIhJwCgIRkYBTEIiIBJxFztfGDzMrBY5d51/PBs60YznxQMccDDrmYLiRYx7inOvd3Ia4C4IbYWY7nHP5ftcRSzrmYNAxB4NXx6ypIRGRgFMQiIgEXNCC4CW/C/CBjjkYdMzB4MkxB+ocgYiIfF3QRgQiItKEgkBEJOASMgjMbJ6ZFZpZkZn9uJntZmbPRbfvNbMJftTZnlpxzI9Gj3WvmW0xs7F+1NmernXMjfabaGb1ZvZALOvzQmuO2cxmmtluM9tvZhtjXWN7a8XPdnczW2lme6LHHNddjM3sZTM7bWb7Wtje/u9fzrmE+kOk5fVhIA9IBfYAo5rsswB4g8gd0u4AtvlddwyOeQrQI/r1/CAcc6P93iXSBfcBv+uOwb9zFpH7gudEH/fxu+4YHPOfAf8t+nVv4ByQ6nftN3DM04EJwL4Wtrf7+1cijggmAUXOuSPOuVpgKbC4yT6LgV+7iK1Alpn1j3Wh7eiax+yc2+KcK4s+3ErkbnDxrDX/zgD/AvgdcDqWxXmkNcf8CLDcOXccwDkX78fdmmN2QDczMyCDSBDUxbbM9uOc20TkGFrS7u9fiRgEA4HiRo9Los+1dZ940tbj+UMinyji2TWP2cwGAvcBL5IYWvPvfBPQw8w2mNlOM3siZtV5ozXH/A/ALURuc/sx8KfOuYbYlOeLdn//8vTGND6xZp5ruka2NfvEk1Yfj5nNIhIEUz2tyHutOeafAz9yztVHPizGvdYcczJwOzAHSAc+MLOtzrlDXhfnkdYccwGwG5gNDAPeNrPNzrkLHtfml3Z//0rEICgBBjd6PIjIJ4W27hNPWnU8ZjYG+CUw3zl3Nka1eaU1x5wPLI2GQDawwMzqnHOvx6TC9tfan+0zzrkKoMLMNgFjgXgNgtYc85PAz1xkAr3IzI4CNwMfxqbEmGv3969EnBraDowws6FmlgosAUJN9gkBT0TPvt8BnHfOnYp1oe3omsdsZjnAcuDxOP502Ng1j9k5N9Q5l+ucywVeA/44jkMAWvez/Xtgmpklm1kXYDJwMMZ1tqfWHPNxIiMgzKwvMBI4EtMqY6vd378SbkTgnKszs2eBtURWHLzsnNtvZk9Ht79IZAXJAqAIqCTyiSJutfKY/wLoBbwQ/YRc5+K4c2MrjzmhtOaYnXMHzexNYC/QAPzSOdfsMsR40Mp/558CvzKzj4lMm/zIORe37anN7J+BmUC2mZUAPwFSwLv3L7WYEBEJuEScGhIRkTZQEIiIBJyCQEQk4BQEIiIBpyAQEQk4BYFIK0U7mO5u9Cc32unzvJntMrODZvaT6L6Nn//EzP7G7/pFWpJw1xGIeKjKOTeu8RNmlgtsds4tMrOuwG4zWxXdfPn5dGCXma1wzr0f25JFrk0jApF2Em3rsJNIv5vGz1cR6YUTz40NJYEpCERaL73RtNCKphvNrBeR/vD7mzzfAxgBbIpNmSJto6khkdb72tRQ1DQz20WkpcPPoi0QZkaf30uk983PnHOfx6xSkTZQEIjcuM3OuUUtPW9mNwHvRc8R7I5xbSLXpKkhEY9Fu73+V+BHftci0hwFgUhsvAhMN7Ohfhci0pS6j4qIBJxGBCIiAacgEBEJOAWBiEjAKQhERAJOQSAiEnAKAhGRgFMQiIgE3P8HwkTQf16OGe8AAAAASUVORK5CYII=",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.clf()\n",
    "plt.plot(fpr, tpr)\n",
    "plt.plot([0, 1], [0, 1], color='gray', linestyle='--')\n",
    "plt.xlabel('FPR')\n",
    "plt.ylabel('TPR')\n",
    "plt.title('ROC curve')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Valoremos aplicar RandomForestClassifier"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 68,
   "metadata": {
    "azdata_cell_guid": "4bd20fc6-4411-40fc-be04-51023d77fe5c"
   },
   "outputs": [],
   "source": [
    "from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "azdata_cell_guid": "56f7b453-513f-4d12-900a-fc1679745bd2"
   },
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": 69,
   "metadata": {
    "azdata_cell_guid": "d17ae683-af99-4b45-a6c3-48bbcc8d509b"
   },
   "outputs": [],
   "source": [
    "rf = RandomForestClassifier(n_estimators= 4, max_depth= 4, random_state=42)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 70,
   "metadata": {
    "azdata_cell_guid": "c9c5aadf-4b3a-4c10-9db5-cea8c969bb33"
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "RandomForestClassifier(max_depth=4, n_estimators=4, random_state=42)"
      ]
     },
     "execution_count": 70,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "rf.fit(X_train,np.ravel(y_train))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 71,
   "metadata": {
    "azdata_cell_guid": "d3f1cad7-55e3-4f06-89dc-fe9aa6df6218"
   },
   "outputs": [],
   "source": [
    "tree_list = rf.estimators_"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 72,
   "metadata": {
    "azdata_cell_guid": "cd41ba25-2fa8-4ae8-869f-509a0097b7fc"
   },
   "outputs": [],
   "source": [
    "dot_data_rf_0 = export_graphviz(decision_tree=tree_list[0],\n",
    "                out_file=None,\n",
    "                feature_names= X_test.columns,\n",
    "                class_names=['Compra','No compra']\n",
    "                )"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 73,
   "metadata": {
    "azdata_cell_guid": "a07bafe8-5806-489e-bd29-8e87be4fa322"
   },
   "outputs": [
    {
     "data": {
      "image/svg+xml": "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"no\"?>\r\n<!DOCTYPE svg PUBLIC \"-//W3C//DTD SVG 1.1//EN\"\r\n \"http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd\">\r\n<!-- Generated by graphviz version 2.38.0 (20140413.2041)\r\n -->\r\n<!-- Title: Tree Pages: 1 -->\r\n<svg width=\"1853pt\" height=\"552pt\"\r\n viewBox=\"0.00 0.00 1853.00 552.00\" xmlns=\"http://www.w3.org/2000/svg\" xmlns:xlink=\"http://www.w3.org/1999/xlink\">\r\n<g id=\"graph0\" class=\"graph\" transform=\"scale(1 1) rotate(0) translate(4 548)\">\r\n<title>Tree</title>\r\n<polygon fill=\"white\" stroke=\"none\" points=\"-4,4 -4,-548 1849,-548 1849,4 -4,4\"/>\r\n<!-- 0 -->\r\n<g id=\"node1\" class=\"node\"><title>0</title>\r\n<polygon fill=\"none\" stroke=\"black\" points=\"1242.5,-544 1071.5,-544 1071.5,-461 1242.5,-461 1242.5,-544\"/>\r\n<text text-anchor=\"middle\" x=\"1157\" y=\"-528.8\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">entry_channel_KAT &lt;= 0.5</text>\r\n<text text-anchor=\"middle\" x=\"1157\" y=\"-513.8\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">gini = 0.066</text>\r\n<text text-anchor=\"middle\" x=\"1157\" y=\"-498.8\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">samples = 2722068</text>\r\n<text text-anchor=\"middle\" x=\"1157\" y=\"-483.8\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">value = [4159747, 146777]</text>\r\n<text text-anchor=\"middle\" x=\"1157\" y=\"-468.8\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">class = Compra</text>\r\n</g>\r\n<!-- 1 -->\r\n<g id=\"node2\" class=\"node\"><title>1</title>\r\n<polygon fill=\"none\" stroke=\"black\" points=\"1041,-425 871,-425 871,-342 1041,-342 1041,-425\"/>\r\n<text text-anchor=\"middle\" x=\"956\" y=\"-409.8\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">entry_channel_KFC &lt;= 0.5</text>\r\n<text text-anchor=\"middle\" x=\"956\" y=\"-394.8\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">gini = 0.061</text>\r\n<text text-anchor=\"middle\" x=\"956\" y=\"-379.8\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">samples = 2449816</text>\r\n<text text-anchor=\"middle\" x=\"956\" y=\"-364.8\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">value = [3754172, 121446]</text>\r\n<text text-anchor=\"middle\" x=\"956\" y=\"-349.8\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">class = Compra</text>\r\n</g>\r\n<!-- 0&#45;&gt;1 -->\r\n<g id=\"edge1\" class=\"edge\"><title>0&#45;&gt;1</title>\r\n<path fill=\"none\" stroke=\"black\" d=\"M1087.26,-460.907C1070.27,-451.016 1051.98,-440.368 1034.6,-430.254\"/>\r\n<polygon fill=\"black\" stroke=\"black\" points=\"1036.02,-427.027 1025.61,-425.021 1032.49,-433.076 1036.02,-427.027\"/>\r\n<text text-anchor=\"middle\" x=\"1032\" y=\"-445.492\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">True</text>\r\n</g>\r\n<!-- 16 -->\r\n<g id=\"node17\" class=\"node\"><title>16</title>\r\n<polygon fill=\"none\" stroke=\"black\" points=\"1462.5,-425 1277.5,-425 1277.5,-342 1462.5,-342 1462.5,-425\"/>\r\n<text text-anchor=\"middle\" x=\"1370\" y=\"-409.8\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">Productos_em_acount &lt;= 0.5</text>\r\n<text text-anchor=\"middle\" x=\"1370\" y=\"-394.8\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">gini = 0.111</text>\r\n<text text-anchor=\"middle\" x=\"1370\" y=\"-379.8\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">samples = 272252</text>\r\n<text text-anchor=\"middle\" x=\"1370\" y=\"-364.8\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">value = [405575, 25331]</text>\r\n<text text-anchor=\"middle\" x=\"1370\" y=\"-349.8\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">class = Compra</text>\r\n</g>\r\n<!-- 0&#45;&gt;16 -->\r\n<g id=\"edge16\" class=\"edge\"><title>0&#45;&gt;16</title>\r\n<path fill=\"none\" stroke=\"black\" d=\"M1230.9,-460.907C1249.16,-450.879 1268.83,-440.075 1287.46,-429.837\"/>\r\n<polygon fill=\"black\" stroke=\"black\" points=\"1289.15,-432.903 1296.23,-425.021 1285.78,-426.768 1289.15,-432.903\"/>\r\n<text text-anchor=\"middle\" x=\"1289.25\" y=\"-445.326\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">False</text>\r\n</g>\r\n<!-- 2 -->\r\n<g id=\"node3\" class=\"node\"><title>2</title>\r\n<polygon fill=\"none\" stroke=\"black\" points=\"572,-306 402,-306 402,-223 572,-223 572,-306\"/>\r\n<text text-anchor=\"middle\" x=\"487\" y=\"-290.8\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">entry_channel_RED &lt;= 0.5</text>\r\n<text text-anchor=\"middle\" x=\"487\" y=\"-275.8\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">gini = 0.04</text>\r\n<text text-anchor=\"middle\" x=\"487\" y=\"-260.8\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">samples = 1895574</text>\r\n<text text-anchor=\"middle\" x=\"487\" y=\"-245.8\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">value = [2937592, 60917]</text>\r\n<text text-anchor=\"middle\" x=\"487\" y=\"-230.8\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">class = Compra</text>\r\n</g>\r\n<!-- 1&#45;&gt;2 -->\r\n<g id=\"edge2\" class=\"edge\"><title>1&#45;&gt;2</title>\r\n<path fill=\"none\" stroke=\"black\" d=\"M870.796,-361.244C789.378,-340.933 666.858,-310.369 582.253,-289.262\"/>\r\n<polygon fill=\"black\" stroke=\"black\" points=\"582.936,-285.826 572.386,-286.801 581.242,-292.618 582.936,-285.826\"/>\r\n</g>\r\n<!-- 9 -->\r\n<g id=\"node10\" class=\"node\"><title>9</title>\r\n<polygon fill=\"none\" stroke=\"black\" points=\"1032,-306 880,-306 880,-223 1032,-223 1032,-306\"/>\r\n<text text-anchor=\"middle\" x=\"956\" y=\"-290.8\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">gender_V &lt;= 0.5</text>\r\n<text text-anchor=\"middle\" x=\"956\" y=\"-275.8\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">gini = 0.128</text>\r\n<text text-anchor=\"middle\" x=\"956\" y=\"-260.8\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">samples = 554242</text>\r\n<text text-anchor=\"middle\" x=\"956\" y=\"-245.8\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">value = [816580, 60529]</text>\r\n<text text-anchor=\"middle\" x=\"956\" y=\"-230.8\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">class = Compra</text>\r\n</g>\r\n<!-- 1&#45;&gt;9 -->\r\n<g id=\"edge9\" class=\"edge\"><title>1&#45;&gt;9</title>\r\n<path fill=\"none\" stroke=\"black\" d=\"M956,-341.907C956,-333.649 956,-324.864 956,-316.302\"/>\r\n<polygon fill=\"black\" stroke=\"black\" points=\"959.5,-316.021 956,-306.021 952.5,-316.021 959.5,-316.021\"/>\r\n</g>\r\n<!-- 3 -->\r\n<g id=\"node4\" class=\"node\"><title>3</title>\r\n<polygon fill=\"none\" stroke=\"black\" points=\"330,-187 160,-187 160,-104 330,-104 330,-187\"/>\r\n<text text-anchor=\"middle\" x=\"245\" y=\"-171.8\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">entry_channel_KHL &lt;= 0.5</text>\r\n<text text-anchor=\"middle\" x=\"245\" y=\"-156.8\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">gini = 0.037</text>\r\n<text text-anchor=\"middle\" x=\"245\" y=\"-141.8\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">samples = 1849424</text>\r\n<text text-anchor=\"middle\" x=\"245\" y=\"-126.8\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">value = [2870555, 54957]</text>\r\n<text text-anchor=\"middle\" x=\"245\" y=\"-111.8\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">class = Compra</text>\r\n</g>\r\n<!-- 2&#45;&gt;3 -->\r\n<g id=\"edge3\" class=\"edge\"><title>2&#45;&gt;3</title>\r\n<path fill=\"none\" stroke=\"black\" d=\"M403.039,-222.907C382.016,-212.743 359.342,-201.781 337.912,-191.42\"/>\r\n<polygon fill=\"black\" stroke=\"black\" points=\"339.339,-188.223 328.812,-187.021 336.292,-194.525 339.339,-188.223\"/>\r\n</g>\r\n<!-- 6 -->\r\n<g id=\"node7\" class=\"node\"><title>6</title>\r\n<polygon fill=\"none\" stroke=\"black\" points=\"556,-187 418,-187 418,-104 556,-104 556,-187\"/>\r\n<text text-anchor=\"middle\" x=\"487\" y=\"-171.8\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">Year &lt;= 2018.5</text>\r\n<text text-anchor=\"middle\" x=\"487\" y=\"-156.8\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">gini = 0.15</text>\r\n<text text-anchor=\"middle\" x=\"487\" y=\"-141.8\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">samples = 46150</text>\r\n<text text-anchor=\"middle\" x=\"487\" y=\"-126.8\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">value = [67037, 5960]</text>\r\n<text text-anchor=\"middle\" x=\"487\" y=\"-111.8\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">class = Compra</text>\r\n</g>\r\n<!-- 2&#45;&gt;6 -->\r\n<g id=\"edge6\" class=\"edge\"><title>2&#45;&gt;6</title>\r\n<path fill=\"none\" stroke=\"black\" d=\"M487,-222.907C487,-214.649 487,-205.864 487,-197.302\"/>\r\n<polygon fill=\"black\" stroke=\"black\" points=\"490.5,-197.021 487,-187.021 483.5,-197.021 490.5,-197.021\"/>\r\n</g>\r\n<!-- 4 -->\r\n<g id=\"node5\" class=\"node\"><title>4</title>\r\n<polygon fill=\"none\" stroke=\"black\" points=\"158,-68 0,-68 0,-0 158,-0 158,-68\"/>\r\n<text text-anchor=\"middle\" x=\"79\" y=\"-52.8\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">gini = 0.036</text>\r\n<text text-anchor=\"middle\" x=\"79\" y=\"-37.8\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">samples = 1825481</text>\r\n<text text-anchor=\"middle\" x=\"79\" y=\"-22.8\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">value = [2834813, 52915]</text>\r\n<text text-anchor=\"middle\" x=\"79\" y=\"-7.8\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">class = Compra</text>\r\n</g>\r\n<!-- 3&#45;&gt;4 -->\r\n<g id=\"edge4\" class=\"edge\"><title>3&#45;&gt;4</title>\r\n<path fill=\"none\" stroke=\"black\" d=\"M183.188,-103.726C168.316,-93.9161 152.471,-83.4644 137.744,-73.7496\"/>\r\n<polygon fill=\"black\" stroke=\"black\" points=\"139.492,-70.7099 129.217,-68.1252 135.637,-76.5532 139.492,-70.7099\"/>\r\n</g>\r\n<!-- 5 -->\r\n<g id=\"node6\" class=\"node\"><title>5</title>\r\n<polygon fill=\"none\" stroke=\"black\" points=\"314,-68 176,-68 176,-0 314,-0 314,-68\"/>\r\n<text text-anchor=\"middle\" x=\"245\" y=\"-52.8\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">gini = 0.102</text>\r\n<text text-anchor=\"middle\" x=\"245\" y=\"-37.8\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">samples = 23943</text>\r\n<text text-anchor=\"middle\" x=\"245\" y=\"-22.8\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">value = [35742, 2042]</text>\r\n<text text-anchor=\"middle\" x=\"245\" y=\"-7.8\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">class = Compra</text>\r\n</g>\r\n<!-- 3&#45;&gt;5 -->\r\n<g id=\"edge5\" class=\"edge\"><title>3&#45;&gt;5</title>\r\n<path fill=\"none\" stroke=\"black\" d=\"M245,-103.726C245,-95.5175 245,-86.8595 245,-78.56\"/>\r\n<polygon fill=\"black\" stroke=\"black\" points=\"248.5,-78.2996 245,-68.2996 241.5,-78.2996 248.5,-78.2996\"/>\r\n</g>\r\n<!-- 7 -->\r\n<g id=\"node8\" class=\"node\"><title>7</title>\r\n<polygon fill=\"none\" stroke=\"black\" points=\"470,-68 332,-68 332,-0 470,-0 470,-68\"/>\r\n<text text-anchor=\"middle\" x=\"401\" y=\"-52.8\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">gini = 0.149</text>\r\n<text text-anchor=\"middle\" x=\"401\" y=\"-37.8\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">samples = 33569</text>\r\n<text text-anchor=\"middle\" x=\"401\" y=\"-22.8\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">value = [48803, 4298]</text>\r\n<text text-anchor=\"middle\" x=\"401\" y=\"-7.8\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">class = Compra</text>\r\n</g>\r\n<!-- 6&#45;&gt;7 -->\r\n<g id=\"edge7\" class=\"edge\"><title>6&#45;&gt;7</title>\r\n<path fill=\"none\" stroke=\"black\" d=\"M454.977,-103.726C447.957,-94.7878 440.518,-85.3168 433.48,-76.3558\"/>\r\n<polygon fill=\"black\" stroke=\"black\" points=\"436.082,-74.0022 427.153,-68.2996 430.577,-78.3259 436.082,-74.0022\"/>\r\n</g>\r\n<!-- 8 -->\r\n<g id=\"node9\" class=\"node\"><title>8</title>\r\n<polygon fill=\"none\" stroke=\"black\" points=\"626,-68 488,-68 488,-0 626,-0 626,-68\"/>\r\n<text text-anchor=\"middle\" x=\"557\" y=\"-52.8\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">gini = 0.153</text>\r\n<text text-anchor=\"middle\" x=\"557\" y=\"-37.8\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">samples = 12581</text>\r\n<text text-anchor=\"middle\" x=\"557\" y=\"-22.8\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">value = [18234, 1662]</text>\r\n<text text-anchor=\"middle\" x=\"557\" y=\"-7.8\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">class = Compra</text>\r\n</g>\r\n<!-- 6&#45;&gt;8 -->\r\n<g id=\"edge8\" class=\"edge\"><title>6&#45;&gt;8</title>\r\n<path fill=\"none\" stroke=\"black\" d=\"M513.065,-103.726C518.663,-94.9703 524.587,-85.7032 530.211,-76.9051\"/>\r\n<polygon fill=\"black\" stroke=\"black\" points=\"533.275,-78.6103 535.713,-68.2996 527.377,-74.8399 533.275,-78.6103\"/>\r\n</g>\r\n<!-- 10 -->\r\n<g id=\"node11\" class=\"node\"><title>10</title>\r\n<polygon fill=\"none\" stroke=\"black\" points=\"938,-187 786,-187 786,-104 938,-104 938,-187\"/>\r\n<text text-anchor=\"middle\" x=\"862\" y=\"-171.8\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">region_code &lt;= 47.5</text>\r\n<text text-anchor=\"middle\" x=\"862\" y=\"-156.8\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">gini = 0.141</text>\r\n<text text-anchor=\"middle\" x=\"862\" y=\"-141.8\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">samples = 216678</text>\r\n<text text-anchor=\"middle\" x=\"862\" y=\"-126.8\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">value = [316782, 26180]</text>\r\n<text text-anchor=\"middle\" x=\"862\" y=\"-111.8\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">class = Compra</text>\r\n</g>\r\n<!-- 9&#45;&gt;10 -->\r\n<g id=\"edge10\" class=\"edge\"><title>9&#45;&gt;10</title>\r\n<path fill=\"none\" stroke=\"black\" d=\"M923.387,-222.907C916.169,-213.923 908.45,-204.315 901.006,-195.05\"/>\r\n<polygon fill=\"black\" stroke=\"black\" points=\"903.547,-192.624 894.555,-187.021 898.09,-197.009 903.547,-192.624\"/>\r\n</g>\r\n<!-- 13 -->\r\n<g id=\"node14\" class=\"node\"><title>13</title>\r\n<polygon fill=\"none\" stroke=\"black\" points=\"1141.5,-187 956.5,-187 956.5,-104 1141.5,-104 1141.5,-187\"/>\r\n<text text-anchor=\"middle\" x=\"1049\" y=\"-171.8\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">Productos_em_acount &lt;= 0.5</text>\r\n<text text-anchor=\"middle\" x=\"1049\" y=\"-156.8\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">gini = 0.12</text>\r\n<text text-anchor=\"middle\" x=\"1049\" y=\"-141.8\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">samples = 337564</text>\r\n<text text-anchor=\"middle\" x=\"1049\" y=\"-126.8\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">value = [499798, 34349]</text>\r\n<text text-anchor=\"middle\" x=\"1049\" y=\"-111.8\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">class = Compra</text>\r\n</g>\r\n<!-- 9&#45;&gt;13 -->\r\n<g id=\"edge13\" class=\"edge\"><title>9&#45;&gt;13</title>\r\n<path fill=\"none\" stroke=\"black\" d=\"M988.266,-222.907C995.407,-213.923 1003.04,-204.315 1010.41,-195.05\"/>\r\n<polygon fill=\"black\" stroke=\"black\" points=\"1013.31,-197.027 1016.79,-187.021 1007.83,-192.671 1013.31,-197.027\"/>\r\n</g>\r\n<!-- 11 -->\r\n<g id=\"node12\" class=\"node\"><title>11</title>\r\n<polygon fill=\"none\" stroke=\"black\" points=\"796,-68 644,-68 644,-0 796,-0 796,-68\"/>\r\n<text text-anchor=\"middle\" x=\"720\" y=\"-52.8\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">gini = 0.142</text>\r\n<text text-anchor=\"middle\" x=\"720\" y=\"-37.8\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">samples = 210321</text>\r\n<text text-anchor=\"middle\" x=\"720\" y=\"-22.8\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">value = [307283, 25650]</text>\r\n<text text-anchor=\"middle\" x=\"720\" y=\"-7.8\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">class = Compra</text>\r\n</g>\r\n<!-- 10&#45;&gt;11 -->\r\n<g id=\"edge11\" class=\"edge\"><title>10&#45;&gt;11</title>\r\n<path fill=\"none\" stroke=\"black\" d=\"M809.124,-103.726C796.705,-94.1494 783.493,-83.9611 771.151,-74.4438\"/>\r\n<polygon fill=\"black\" stroke=\"black\" points=\"773.239,-71.6345 763.183,-68.2996 768.965,-77.1778 773.239,-71.6345\"/>\r\n</g>\r\n<!-- 12 -->\r\n<g id=\"node13\" class=\"node\"><title>12</title>\r\n<polygon fill=\"none\" stroke=\"black\" points=\"939.5,-68 814.5,-68 814.5,-0 939.5,-0 939.5,-68\"/>\r\n<text text-anchor=\"middle\" x=\"877\" y=\"-52.8\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">gini = 0.1</text>\r\n<text text-anchor=\"middle\" x=\"877\" y=\"-37.8\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">samples = 6357</text>\r\n<text text-anchor=\"middle\" x=\"877\" y=\"-22.8\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">value = [9499, 530]</text>\r\n<text text-anchor=\"middle\" x=\"877\" y=\"-7.8\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">class = Compra</text>\r\n</g>\r\n<!-- 10&#45;&gt;12 -->\r\n<g id=\"edge12\" class=\"edge\"><title>10&#45;&gt;12</title>\r\n<path fill=\"none\" stroke=\"black\" d=\"M867.585,-103.726C868.722,-95.4263 869.922,-86.6671 871.071,-78.2834\"/>\r\n<polygon fill=\"black\" stroke=\"black\" points=\"874.549,-78.6821 872.438,-68.2996 867.614,-77.732 874.549,-78.6821\"/>\r\n</g>\r\n<!-- 14 -->\r\n<g id=\"node15\" class=\"node\"><title>14</title>\r\n<polygon fill=\"none\" stroke=\"black\" points=\"1110,-68 958,-68 958,-0 1110,-0 1110,-68\"/>\r\n<text text-anchor=\"middle\" x=\"1034\" y=\"-52.8\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">gini = 0.186</text>\r\n<text text-anchor=\"middle\" x=\"1034\" y=\"-37.8\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">samples = 208991</text>\r\n<text text-anchor=\"middle\" x=\"1034\" y=\"-22.8\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">value = [296633, 34349]</text>\r\n<text text-anchor=\"middle\" x=\"1034\" y=\"-7.8\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">class = Compra</text>\r\n</g>\r\n<!-- 13&#45;&gt;14 -->\r\n<g id=\"edge14\" class=\"edge\"><title>13&#45;&gt;14</title>\r\n<path fill=\"none\" stroke=\"black\" d=\"M1043.41,-103.726C1042.28,-95.4263 1041.08,-86.6671 1039.93,-78.2834\"/>\r\n<polygon fill=\"black\" stroke=\"black\" points=\"1043.39,-77.732 1038.56,-68.2996 1036.45,-78.6821 1043.39,-77.732\"/>\r\n</g>\r\n<!-- 15 -->\r\n<g id=\"node16\" class=\"node\"><title>15</title>\r\n<polygon fill=\"none\" stroke=\"black\" points=\"1253.5,-68 1128.5,-68 1128.5,-0 1253.5,-0 1253.5,-68\"/>\r\n<text text-anchor=\"middle\" x=\"1191\" y=\"-52.8\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">gini = 0.0</text>\r\n<text text-anchor=\"middle\" x=\"1191\" y=\"-37.8\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">samples = 128573</text>\r\n<text text-anchor=\"middle\" x=\"1191\" y=\"-22.8\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">value = [203165, 0]</text>\r\n<text text-anchor=\"middle\" x=\"1191\" y=\"-7.8\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">class = Compra</text>\r\n</g>\r\n<!-- 13&#45;&gt;15 -->\r\n<g id=\"edge15\" class=\"edge\"><title>13&#45;&gt;15</title>\r\n<path fill=\"none\" stroke=\"black\" d=\"M1101.88,-103.726C1114.29,-94.1494 1127.51,-83.9611 1139.85,-74.4438\"/>\r\n<polygon fill=\"black\" stroke=\"black\" points=\"1142.04,-77.1778 1147.82,-68.2996 1137.76,-71.6345 1142.04,-77.1778\"/>\r\n</g>\r\n<!-- 17 -->\r\n<g id=\"node18\" class=\"node\"><title>17</title>\r\n<polygon fill=\"none\" stroke=\"black\" points=\"1446,-306 1294,-306 1294,-223 1446,-223 1446,-306\"/>\r\n<text text-anchor=\"middle\" x=\"1370\" y=\"-290.8\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">gender_V &lt;= 0.5</text>\r\n<text text-anchor=\"middle\" x=\"1370\" y=\"-275.8\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">gini = 0.184</text>\r\n<text text-anchor=\"middle\" x=\"1370\" y=\"-260.8\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">samples = 156086</text>\r\n<text text-anchor=\"middle\" x=\"1370\" y=\"-245.8\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">value = [221730, 25331]</text>\r\n<text text-anchor=\"middle\" x=\"1370\" y=\"-230.8\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">class = Compra</text>\r\n</g>\r\n<!-- 16&#45;&gt;17 -->\r\n<g id=\"edge17\" class=\"edge\"><title>16&#45;&gt;17</title>\r\n<path fill=\"none\" stroke=\"black\" d=\"M1370,-341.907C1370,-333.649 1370,-324.864 1370,-316.302\"/>\r\n<polygon fill=\"black\" stroke=\"black\" points=\"1373.5,-316.021 1370,-306.021 1366.5,-316.021 1373.5,-316.021\"/>\r\n</g>\r\n<!-- 24 -->\r\n<g id=\"node25\" class=\"node\"><title>24</title>\r\n<polygon fill=\"none\" stroke=\"black\" points=\"1589.5,-298.5 1464.5,-298.5 1464.5,-230.5 1589.5,-230.5 1589.5,-298.5\"/>\r\n<text text-anchor=\"middle\" x=\"1527\" y=\"-283.3\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">gini = 0.0</text>\r\n<text text-anchor=\"middle\" x=\"1527\" y=\"-268.3\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">samples = 116166</text>\r\n<text text-anchor=\"middle\" x=\"1527\" y=\"-253.3\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">value = [183845, 0]</text>\r\n<text text-anchor=\"middle\" x=\"1527\" y=\"-238.3\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">class = Compra</text>\r\n</g>\r\n<!-- 16&#45;&gt;24 -->\r\n<g id=\"edge24\" class=\"edge\"><title>16&#45;&gt;24</title>\r\n<path fill=\"none\" stroke=\"black\" d=\"M1424.47,-341.907C1440.61,-329.88 1458.25,-316.735 1474.28,-304.791\"/>\r\n<polygon fill=\"black\" stroke=\"black\" points=\"1476.57,-307.449 1482.49,-298.667 1472.38,-301.836 1476.57,-307.449\"/>\r\n</g>\r\n<!-- 18 -->\r\n<g id=\"node19\" class=\"node\"><title>18</title>\r\n<polygon fill=\"none\" stroke=\"black\" points=\"1484.5,-187 1255.5,-187 1255.5,-104 1484.5,-104 1484.5,-187\"/>\r\n<text text-anchor=\"middle\" x=\"1370\" y=\"-171.8\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">Productos_long_term_deposit &lt;= 0.5</text>\r\n<text text-anchor=\"middle\" x=\"1370\" y=\"-156.8\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">gini = 0.192</text>\r\n<text text-anchor=\"middle\" x=\"1370\" y=\"-141.8\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">samples = 73230</text>\r\n<text text-anchor=\"middle\" x=\"1370\" y=\"-126.8\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">value = [103578, 12449]</text>\r\n<text text-anchor=\"middle\" x=\"1370\" y=\"-111.8\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">class = Compra</text>\r\n</g>\r\n<!-- 17&#45;&gt;18 -->\r\n<g id=\"edge18\" class=\"edge\"><title>17&#45;&gt;18</title>\r\n<path fill=\"none\" stroke=\"black\" d=\"M1370,-222.907C1370,-214.649 1370,-205.864 1370,-197.302\"/>\r\n<polygon fill=\"black\" stroke=\"black\" points=\"1373.5,-197.021 1370,-187.021 1366.5,-197.021 1373.5,-197.021\"/>\r\n</g>\r\n<!-- 21 -->\r\n<g id=\"node22\" class=\"node\"><title>21</title>\r\n<polygon fill=\"none\" stroke=\"black\" points=\"1743,-187 1529,-187 1529,-104 1743,-104 1743,-187\"/>\r\n<text text-anchor=\"middle\" x=\"1636\" y=\"-171.8\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">Productos_payroll_account &lt;= 0.5</text>\r\n<text text-anchor=\"middle\" x=\"1636\" y=\"-156.8\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">gini = 0.177</text>\r\n<text text-anchor=\"middle\" x=\"1636\" y=\"-141.8\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">samples = 82856</text>\r\n<text text-anchor=\"middle\" x=\"1636\" y=\"-126.8\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">value = [118152, 12882]</text>\r\n<text text-anchor=\"middle\" x=\"1636\" y=\"-111.8\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">class = Compra</text>\r\n</g>\r\n<!-- 17&#45;&gt;21 -->\r\n<g id=\"edge21\" class=\"edge\"><title>17&#45;&gt;21</title>\r\n<path fill=\"none\" stroke=\"black\" d=\"M1446.38,-226.909C1449.29,-225.58 1452.17,-224.273 1455,-223 1478.49,-212.447 1503.86,-201.466 1527.96,-191.228\"/>\r\n<polygon fill=\"black\" stroke=\"black\" points=\"1529.65,-194.317 1537.49,-187.193 1526.92,-187.871 1529.65,-194.317\"/>\r\n</g>\r\n<!-- 19 -->\r\n<g id=\"node20\" class=\"node\"><title>19</title>\r\n<polygon fill=\"none\" stroke=\"black\" points=\"1416.5,-68 1271.5,-68 1271.5,-0 1416.5,-0 1416.5,-68\"/>\r\n<text text-anchor=\"middle\" x=\"1344\" y=\"-52.8\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">gini = 0.204</text>\r\n<text text-anchor=\"middle\" x=\"1344\" y=\"-37.8\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">samples = 68126</text>\r\n<text text-anchor=\"middle\" x=\"1344\" y=\"-22.8\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">value = [95459, 12449]</text>\r\n<text text-anchor=\"middle\" x=\"1344\" y=\"-7.8\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">class = Compra</text>\r\n</g>\r\n<!-- 18&#45;&gt;19 -->\r\n<g id=\"edge19\" class=\"edge\"><title>18&#45;&gt;19</title>\r\n<path fill=\"none\" stroke=\"black\" d=\"M1360.32,-103.726C1358.35,-95.4263 1356.27,-86.6671 1354.28,-78.2834\"/>\r\n<polygon fill=\"black\" stroke=\"black\" points=\"1357.62,-77.2205 1351.91,-68.2996 1350.81,-78.8377 1357.62,-77.2205\"/>\r\n</g>\r\n<!-- 20 -->\r\n<g id=\"node21\" class=\"node\"><title>20</title>\r\n<polygon fill=\"none\" stroke=\"black\" points=\"1545.5,-68 1434.5,-68 1434.5,-0 1545.5,-0 1545.5,-68\"/>\r\n<text text-anchor=\"middle\" x=\"1490\" y=\"-52.8\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">gini = 0.0</text>\r\n<text text-anchor=\"middle\" x=\"1490\" y=\"-37.8\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">samples = 5104</text>\r\n<text text-anchor=\"middle\" x=\"1490\" y=\"-22.8\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">value = [8119, 0]</text>\r\n<text text-anchor=\"middle\" x=\"1490\" y=\"-7.8\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">class = Compra</text>\r\n</g>\r\n<!-- 18&#45;&gt;20 -->\r\n<g id=\"edge20\" class=\"edge\"><title>18&#45;&gt;20</title>\r\n<path fill=\"none\" stroke=\"black\" d=\"M1414.68,-103.726C1424.88,-94.423 1435.71,-84.5428 1445.88,-75.2612\"/>\r\n<polygon fill=\"black\" stroke=\"black\" points=\"1448.48,-77.6255 1453.51,-68.2996 1443.76,-72.4547 1448.48,-77.6255\"/>\r\n</g>\r\n<!-- 22 -->\r\n<g id=\"node23\" class=\"node\"><title>22</title>\r\n<polygon fill=\"none\" stroke=\"black\" points=\"1708.5,-68 1563.5,-68 1563.5,-0 1708.5,-0 1708.5,-68\"/>\r\n<text text-anchor=\"middle\" x=\"1636\" y=\"-52.8\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">gini = 0.204</text>\r\n<text text-anchor=\"middle\" x=\"1636\" y=\"-37.8\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">samples = 70832</text>\r\n<text text-anchor=\"middle\" x=\"1636\" y=\"-22.8\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">value = [99142, 12882]</text>\r\n<text text-anchor=\"middle\" x=\"1636\" y=\"-7.8\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">class = Compra</text>\r\n</g>\r\n<!-- 21&#45;&gt;22 -->\r\n<g id=\"edge22\" class=\"edge\"><title>21&#45;&gt;22</title>\r\n<path fill=\"none\" stroke=\"black\" d=\"M1636,-103.726C1636,-95.5175 1636,-86.8595 1636,-78.56\"/>\r\n<polygon fill=\"black\" stroke=\"black\" points=\"1639.5,-78.2996 1636,-68.2996 1632.5,-78.2996 1639.5,-78.2996\"/>\r\n</g>\r\n<!-- 23 -->\r\n<g id=\"node24\" class=\"node\"><title>23</title>\r\n<polygon fill=\"none\" stroke=\"black\" points=\"1845,-68 1727,-68 1727,-0 1845,-0 1845,-68\"/>\r\n<text text-anchor=\"middle\" x=\"1786\" y=\"-52.8\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">gini = 0.0</text>\r\n<text text-anchor=\"middle\" x=\"1786\" y=\"-37.8\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">samples = 12024</text>\r\n<text text-anchor=\"middle\" x=\"1786\" y=\"-22.8\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">value = [19010, 0]</text>\r\n<text text-anchor=\"middle\" x=\"1786\" y=\"-7.8\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">class = Compra</text>\r\n</g>\r\n<!-- 21&#45;&gt;23 -->\r\n<g id=\"edge23\" class=\"edge\"><title>21&#45;&gt;23</title>\r\n<path fill=\"none\" stroke=\"black\" d=\"M1691.85,-103.726C1704.97,-94.1494 1718.93,-83.9611 1731.97,-74.4438\"/>\r\n<polygon fill=\"black\" stroke=\"black\" points=\"1734.37,-77.0227 1740.38,-68.2996 1730.24,-71.3688 1734.37,-77.0227\"/>\r\n</g>\r\n</g>\r\n</svg>\r\n",
      "text/plain": [
       "<graphviz.files.Source at 0x1fe232c4880>"
      ]
     },
     "execution_count": 73,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "graphviz.Source(dot_data_rf_0)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 74,
   "metadata": {
    "azdata_cell_guid": "2e4c9a2b-4eba-4a4a-b7ab-ae2d54d5eb55"
   },
   "outputs": [],
   "source": [
    "dot_data_rf_1 = export_graphviz(decision_tree=tree_list[1],\n",
    "                out_file=None,\n",
    "                feature_names= X_test.columns,\n",
    "                class_names=['Compra','No compra']\n",
    "                )"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 75,
   "metadata": {
    "azdata_cell_guid": "abcdef67-d5aa-4bfc-9fb6-f5badec18d27"
   },
   "outputs": [
    {
     "data": {
      "image/svg+xml": "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"no\"?>\r\n<!DOCTYPE svg PUBLIC \"-//W3C//DTD SVG 1.1//EN\"\r\n \"http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd\">\r\n<!-- Generated by graphviz version 2.38.0 (20140413.2041)\r\n -->\r\n<!-- Title: Tree Pages: 1 -->\r\n<svg width=\"1014pt\" height=\"552pt\"\r\n viewBox=\"0.00 0.00 1013.50 552.00\" xmlns=\"http://www.w3.org/2000/svg\" xmlns:xlink=\"http://www.w3.org/1999/xlink\">\r\n<g id=\"graph0\" class=\"graph\" transform=\"scale(1 1) rotate(0) translate(4 548)\">\r\n<title>Tree</title>\r\n<polygon fill=\"white\" stroke=\"none\" points=\"-4,4 -4,-548 1009.5,-548 1009.5,4 -4,4\"/>\r\n<!-- 0 -->\r\n<g id=\"node1\" class=\"node\"><title>0</title>\r\n<polygon fill=\"none\" stroke=\"black\" points=\"757.5,-544 592.5,-544 592.5,-461 757.5,-461 757.5,-544\"/>\r\n<text text-anchor=\"middle\" x=\"675\" y=\"-528.8\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">recurrencia &lt;= 15.5</text>\r\n<text text-anchor=\"middle\" x=\"675\" y=\"-513.8\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">gini = 0.066</text>\r\n<text text-anchor=\"middle\" x=\"675\" y=\"-498.8\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">samples = 2722582</text>\r\n<text text-anchor=\"middle\" x=\"675\" y=\"-483.8\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">value = [4158815, 147709]</text>\r\n<text text-anchor=\"middle\" x=\"675\" y=\"-468.8\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">class = Compra</text>\r\n</g>\r\n<!-- 1 -->\r\n<g id=\"node2\" class=\"node\"><title>1</title>\r\n<polygon fill=\"none\" stroke=\"black\" points=\"666.5,-425 481.5,-425 481.5,-342 666.5,-342 666.5,-425\"/>\r\n<text text-anchor=\"middle\" x=\"574\" y=\"-409.8\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">Productos_em_acount &lt;= 0.5</text>\r\n<text text-anchor=\"middle\" x=\"574\" y=\"-394.8\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">gini = 0.031</text>\r\n<text text-anchor=\"middle\" x=\"574\" y=\"-379.8\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">samples = 2125102</text>\r\n<text text-anchor=\"middle\" x=\"574\" y=\"-364.8\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">value = [3308259, 53789]</text>\r\n<text text-anchor=\"middle\" x=\"574\" y=\"-349.8\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">class = Compra</text>\r\n</g>\r\n<!-- 0&#45;&gt;1 -->\r\n<g id=\"edge1\" class=\"edge\"><title>0&#45;&gt;1</title>\r\n<path fill=\"none\" stroke=\"black\" d=\"M639.959,-460.907C632.124,-451.832 623.741,-442.121 615.669,-432.769\"/>\r\n<polygon fill=\"black\" stroke=\"black\" points=\"618.163,-430.303 608.98,-425.021 612.865,-434.878 618.163,-430.303\"/>\r\n<text text-anchor=\"middle\" x=\"607.15\" y=\"-446.254\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">True</text>\r\n</g>\r\n<!-- 10 -->\r\n<g id=\"node11\" class=\"node\"><title>10</title>\r\n<polygon fill=\"none\" stroke=\"black\" points=\"869.5,-425 684.5,-425 684.5,-342 869.5,-342 869.5,-425\"/>\r\n<text text-anchor=\"middle\" x=\"777\" y=\"-409.8\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">Productos_em_acount &lt;= 0.5</text>\r\n<text text-anchor=\"middle\" x=\"777\" y=\"-394.8\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">gini = 0.179</text>\r\n<text text-anchor=\"middle\" x=\"777\" y=\"-379.8\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">samples = 597480</text>\r\n<text text-anchor=\"middle\" x=\"777\" y=\"-364.8\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">value = [850556, 93920]</text>\r\n<text text-anchor=\"middle\" x=\"777\" y=\"-349.8\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">class = Compra</text>\r\n</g>\r\n<!-- 0&#45;&gt;10 -->\r\n<g id=\"edge10\" class=\"edge\"><title>0&#45;&gt;10</title>\r\n<path fill=\"none\" stroke=\"black\" d=\"M710.388,-460.907C718.3,-451.832 726.766,-442.121 734.919,-432.769\"/>\r\n<polygon fill=\"black\" stroke=\"black\" points=\"737.741,-434.859 741.674,-425.021 732.465,-430.259 737.741,-434.859\"/>\r\n<text text-anchor=\"middle\" x=\"743.382\" y=\"-446.262\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">False</text>\r\n</g>\r\n<!-- 2 -->\r\n<g id=\"node3\" class=\"node\"><title>2</title>\r\n<polygon fill=\"none\" stroke=\"black\" points=\"497,-306 345,-306 345,-223 497,-223 497,-306\"/>\r\n<text text-anchor=\"middle\" x=\"421\" y=\"-290.8\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">gender_H &lt;= 0.5</text>\r\n<text text-anchor=\"middle\" x=\"421\" y=\"-275.8\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">gini = 0.136</text>\r\n<text text-anchor=\"middle\" x=\"421\" y=\"-260.8\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">samples = 461784</text>\r\n<text text-anchor=\"middle\" x=\"421\" y=\"-245.8\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">value = [676843, 53789]</text>\r\n<text text-anchor=\"middle\" x=\"421\" y=\"-230.8\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">class = Compra</text>\r\n</g>\r\n<!-- 1&#45;&gt;2 -->\r\n<g id=\"edge2\" class=\"edge\"><title>1&#45;&gt;2</title>\r\n<path fill=\"none\" stroke=\"black\" d=\"M520.917,-341.907C508.457,-332.379 495.078,-322.148 482.291,-312.37\"/>\r\n<polygon fill=\"black\" stroke=\"black\" points=\"484.058,-309.315 473.989,-306.021 479.806,-314.876 484.058,-309.315\"/>\r\n</g>\r\n<!-- 9 -->\r\n<g id=\"node10\" class=\"node\"><title>9</title>\r\n<polygon fill=\"none\" stroke=\"black\" points=\"646.5,-298.5 515.5,-298.5 515.5,-230.5 646.5,-230.5 646.5,-298.5\"/>\r\n<text text-anchor=\"middle\" x=\"581\" y=\"-283.3\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">gini = 0.0</text>\r\n<text text-anchor=\"middle\" x=\"581\" y=\"-268.3\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">samples = 1663318</text>\r\n<text text-anchor=\"middle\" x=\"581\" y=\"-253.3\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">value = [2631416, 0]</text>\r\n<text text-anchor=\"middle\" x=\"581\" y=\"-238.3\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">class = Compra</text>\r\n</g>\r\n<!-- 1&#45;&gt;9 -->\r\n<g id=\"edge9\" class=\"edge\"><title>1&#45;&gt;9</title>\r\n<path fill=\"none\" stroke=\"black\" d=\"M576.429,-341.907C577.069,-331.204 577.762,-319.615 578.411,-308.776\"/>\r\n<polygon fill=\"black\" stroke=\"black\" points=\"581.912,-308.858 579.016,-298.667 574.925,-308.44 581.912,-308.858\"/>\r\n</g>\r\n<!-- 3 -->\r\n<g id=\"node4\" class=\"node\"><title>3</title>\r\n<polygon fill=\"none\" stroke=\"black\" points=\"317,-187 165,-187 165,-104 317,-104 317,-187\"/>\r\n<text text-anchor=\"middle\" x=\"241\" y=\"-171.8\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">age &lt;= 34.5</text>\r\n<text text-anchor=\"middle\" x=\"241\" y=\"-156.8\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">gini = 0.127</text>\r\n<text text-anchor=\"middle\" x=\"241\" y=\"-141.8\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">samples = 259528</text>\r\n<text text-anchor=\"middle\" x=\"241\" y=\"-126.8\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">value = [382745, 27917]</text>\r\n<text text-anchor=\"middle\" x=\"241\" y=\"-111.8\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">class = Compra</text>\r\n</g>\r\n<!-- 2&#45;&gt;3 -->\r\n<g id=\"edge3\" class=\"edge\"><title>2&#45;&gt;3</title>\r\n<path fill=\"none\" stroke=\"black\" d=\"M358.55,-222.907C343.471,-213.106 327.251,-202.563 311.819,-192.533\"/>\r\n<polygon fill=\"black\" stroke=\"black\" points=\"313.632,-189.536 303.34,-187.021 309.817,-195.405 313.632,-189.536\"/>\r\n</g>\r\n<!-- 6 -->\r\n<g id=\"node7\" class=\"node\"><title>6</title>\r\n<polygon fill=\"none\" stroke=\"black\" points=\"523,-187 335,-187 335,-104 523,-104 523,-187\"/>\r\n<text text-anchor=\"middle\" x=\"429\" y=\"-171.8\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">Productos_credit_card &lt;= 0.5</text>\r\n<text text-anchor=\"middle\" x=\"429\" y=\"-156.8\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">gini = 0.149</text>\r\n<text text-anchor=\"middle\" x=\"429\" y=\"-141.8\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">samples = 202256</text>\r\n<text text-anchor=\"middle\" x=\"429\" y=\"-126.8\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">value = [294098, 25872]</text>\r\n<text text-anchor=\"middle\" x=\"429\" y=\"-111.8\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">class = Compra</text>\r\n</g>\r\n<!-- 2&#45;&gt;6 -->\r\n<g id=\"edge6\" class=\"edge\"><title>2&#45;&gt;6</title>\r\n<path fill=\"none\" stroke=\"black\" d=\"M423.776,-222.907C424.346,-214.558 424.954,-205.671 425.546,-197.02\"/>\r\n<polygon fill=\"black\" stroke=\"black\" points=\"429.039,-197.236 426.229,-187.021 422.055,-196.759 429.039,-197.236\"/>\r\n</g>\r\n<!-- 4 -->\r\n<g id=\"node5\" class=\"node\"><title>4</title>\r\n<polygon fill=\"none\" stroke=\"black\" points=\"152,-68 0,-68 0,-0 152,-0 152,-68\"/>\r\n<text text-anchor=\"middle\" x=\"76\" y=\"-52.8\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">gini = 0.176</text>\r\n<text text-anchor=\"middle\" x=\"76\" y=\"-37.8\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">samples = 102762</text>\r\n<text text-anchor=\"middle\" x=\"76\" y=\"-22.8\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">value = [146959, 15888]</text>\r\n<text text-anchor=\"middle\" x=\"76\" y=\"-7.8\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">class = Compra</text>\r\n</g>\r\n<!-- 3&#45;&gt;4 -->\r\n<g id=\"edge4\" class=\"edge\"><title>3&#45;&gt;4</title>\r\n<path fill=\"none\" stroke=\"black\" d=\"M179.56,-103.726C164.778,-93.9161 149.029,-83.4644 134.39,-73.7496\"/>\r\n<polygon fill=\"black\" stroke=\"black\" points=\"136.182,-70.7385 125.915,-68.1252 132.312,-76.571 136.182,-70.7385\"/>\r\n</g>\r\n<!-- 5 -->\r\n<g id=\"node6\" class=\"node\"><title>5</title>\r\n<polygon fill=\"none\" stroke=\"black\" points=\"322,-68 170,-68 170,-0 322,-0 322,-68\"/>\r\n<text text-anchor=\"middle\" x=\"246\" y=\"-52.8\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">gini = 0.092</text>\r\n<text text-anchor=\"middle\" x=\"246\" y=\"-37.8\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">samples = 156766</text>\r\n<text text-anchor=\"middle\" x=\"246\" y=\"-22.8\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">value = [235786, 12029]</text>\r\n<text text-anchor=\"middle\" x=\"246\" y=\"-7.8\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">class = Compra</text>\r\n</g>\r\n<!-- 3&#45;&gt;5 -->\r\n<g id=\"edge5\" class=\"edge\"><title>3&#45;&gt;5</title>\r\n<path fill=\"none\" stroke=\"black\" d=\"M242.862,-103.726C243.237,-95.5175 243.632,-86.8595 244.011,-78.56\"/>\r\n<polygon fill=\"black\" stroke=\"black\" points=\"247.52,-78.4489 244.479,-68.2996 240.527,-78.1295 247.52,-78.4489\"/>\r\n</g>\r\n<!-- 7 -->\r\n<g id=\"node8\" class=\"node\"><title>7</title>\r\n<polygon fill=\"none\" stroke=\"black\" points=\"501,-68 349,-68 349,-0 501,-0 501,-68\"/>\r\n<text text-anchor=\"middle\" x=\"425\" y=\"-52.8\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">gini = 0.156</text>\r\n<text text-anchor=\"middle\" x=\"425\" y=\"-37.8\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">samples = 191390</text>\r\n<text text-anchor=\"middle\" x=\"425\" y=\"-22.8\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">value = [276932, 25872]</text>\r\n<text text-anchor=\"middle\" x=\"425\" y=\"-7.8\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">class = Compra</text>\r\n</g>\r\n<!-- 6&#45;&gt;7 -->\r\n<g id=\"edge7\" class=\"edge\"><title>6&#45;&gt;7</title>\r\n<path fill=\"none\" stroke=\"black\" d=\"M427.511,-103.726C427.211,-95.5175 426.894,-86.8595 426.591,-78.56\"/>\r\n<polygon fill=\"black\" stroke=\"black\" points=\"430.079,-78.1651 426.216,-68.2996 423.084,-78.4207 430.079,-78.1651\"/>\r\n</g>\r\n<!-- 8 -->\r\n<g id=\"node9\" class=\"node\"><title>8</title>\r\n<polygon fill=\"none\" stroke=\"black\" points=\"637,-68 519,-68 519,-0 637,-0 637,-68\"/>\r\n<text text-anchor=\"middle\" x=\"578\" y=\"-52.8\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">gini = 0.0</text>\r\n<text text-anchor=\"middle\" x=\"578\" y=\"-37.8\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">samples = 10866</text>\r\n<text text-anchor=\"middle\" x=\"578\" y=\"-22.8\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">value = [17166, 0]</text>\r\n<text text-anchor=\"middle\" x=\"578\" y=\"-7.8\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">class = Compra</text>\r\n</g>\r\n<!-- 6&#45;&gt;8 -->\r\n<g id=\"edge8\" class=\"edge\"><title>6&#45;&gt;8</title>\r\n<path fill=\"none\" stroke=\"black\" d=\"M484.482,-103.726C497.514,-94.1494 511.377,-83.9611 524.328,-74.4438\"/>\r\n<polygon fill=\"black\" stroke=\"black\" points=\"526.703,-77.0418 532.688,-68.2996 522.558,-71.4012 526.703,-77.0418\"/>\r\n</g>\r\n<!-- 11 -->\r\n<g id=\"node12\" class=\"node\"><title>11</title>\r\n<polygon fill=\"none\" stroke=\"black\" points=\"862,-306 678,-306 678,-223 862,-223 862,-306\"/>\r\n<text text-anchor=\"middle\" x=\"770\" y=\"-290.8\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">Productos_debit_card &lt;= 0.5</text>\r\n<text text-anchor=\"middle\" x=\"770\" y=\"-275.8\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">gini = 0.285</text>\r\n<text text-anchor=\"middle\" x=\"770\" y=\"-260.8\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">samples = 344812</text>\r\n<text text-anchor=\"middle\" x=\"770\" y=\"-245.8\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">value = [450943, 93920]</text>\r\n<text text-anchor=\"middle\" x=\"770\" y=\"-230.8\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">class = Compra</text>\r\n</g>\r\n<!-- 10&#45;&gt;11 -->\r\n<g id=\"edge11\" class=\"edge\"><title>10&#45;&gt;11</title>\r\n<path fill=\"none\" stroke=\"black\" d=\"M774.571,-341.907C774.072,-333.558 773.54,-324.671 773.023,-316.02\"/>\r\n<polygon fill=\"black\" stroke=\"black\" points=\"776.515,-315.794 772.424,-306.021 769.528,-316.212 776.515,-315.794\"/>\r\n</g>\r\n<!-- 16 -->\r\n<g id=\"node17\" class=\"node\"><title>16</title>\r\n<polygon fill=\"none\" stroke=\"black\" points=\"1005.5,-298.5 880.5,-298.5 880.5,-230.5 1005.5,-230.5 1005.5,-298.5\"/>\r\n<text text-anchor=\"middle\" x=\"943\" y=\"-283.3\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">gini = 0.0</text>\r\n<text text-anchor=\"middle\" x=\"943\" y=\"-268.3\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">samples = 252668</text>\r\n<text text-anchor=\"middle\" x=\"943\" y=\"-253.3\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">value = [399613, 0]</text>\r\n<text text-anchor=\"middle\" x=\"943\" y=\"-238.3\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">class = Compra</text>\r\n</g>\r\n<!-- 10&#45;&gt;16 -->\r\n<g id=\"edge16\" class=\"edge\"><title>10&#45;&gt;16</title>\r\n<path fill=\"none\" stroke=\"black\" d=\"M834.593,-341.907C851.814,-329.769 870.651,-316.493 887.72,-304.462\"/>\r\n<polygon fill=\"black\" stroke=\"black\" points=\"889.785,-307.289 895.943,-298.667 885.752,-301.567 889.785,-307.289\"/>\r\n</g>\r\n<!-- 12 -->\r\n<g id=\"node13\" class=\"node\"><title>12</title>\r\n<polygon fill=\"none\" stroke=\"black\" points=\"825,-187 673,-187 673,-104 825,-104 825,-187\"/>\r\n<text text-anchor=\"middle\" x=\"749\" y=\"-171.8\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">recurrencia &lt;= 32.5</text>\r\n<text text-anchor=\"middle\" x=\"749\" y=\"-156.8\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">gini = 0.38</text>\r\n<text text-anchor=\"middle\" x=\"749\" y=\"-141.8\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">samples = 233533</text>\r\n<text text-anchor=\"middle\" x=\"749\" y=\"-126.8\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">value = [274797, 93920]</text>\r\n<text text-anchor=\"middle\" x=\"749\" y=\"-111.8\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">class = Compra</text>\r\n</g>\r\n<!-- 11&#45;&gt;12 -->\r\n<g id=\"edge12\" class=\"edge\"><title>11&#45;&gt;12</title>\r\n<path fill=\"none\" stroke=\"black\" d=\"M762.714,-222.907C761.216,-214.558 759.62,-205.671 758.068,-197.02\"/>\r\n<polygon fill=\"black\" stroke=\"black\" points=\"761.485,-196.245 756.273,-187.021 754.595,-197.482 761.485,-196.245\"/>\r\n</g>\r\n<!-- 15 -->\r\n<g id=\"node16\" class=\"node\"><title>15</title>\r\n<polygon fill=\"none\" stroke=\"black\" points=\"968.5,-179.5 843.5,-179.5 843.5,-111.5 968.5,-111.5 968.5,-179.5\"/>\r\n<text text-anchor=\"middle\" x=\"906\" y=\"-164.3\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">gini = 0.0</text>\r\n<text text-anchor=\"middle\" x=\"906\" y=\"-149.3\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">samples = 111279</text>\r\n<text text-anchor=\"middle\" x=\"906\" y=\"-134.3\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">value = [176146, 0]</text>\r\n<text text-anchor=\"middle\" x=\"906\" y=\"-119.3\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">class = Compra</text>\r\n</g>\r\n<!-- 11&#45;&gt;15 -->\r\n<g id=\"edge15\" class=\"edge\"><title>11&#45;&gt;15</title>\r\n<path fill=\"none\" stroke=\"black\" d=\"M817.185,-222.907C830.909,-211.101 845.885,-198.217 859.563,-186.45\"/>\r\n<polygon fill=\"black\" stroke=\"black\" points=\"862.149,-188.842 867.447,-179.667 857.584,-183.535 862.149,-188.842\"/>\r\n</g>\r\n<!-- 13 -->\r\n<g id=\"node14\" class=\"node\"><title>13</title>\r\n<polygon fill=\"none\" stroke=\"black\" points=\"818,-68 666,-68 666,-0 818,-0 818,-68\"/>\r\n<text text-anchor=\"middle\" x=\"742\" y=\"-52.8\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">gini = 0.448</text>\r\n<text text-anchor=\"middle\" x=\"742\" y=\"-37.8\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">samples = 138394</text>\r\n<text text-anchor=\"middle\" x=\"742\" y=\"-22.8\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">value = [144398, 74050]</text>\r\n<text text-anchor=\"middle\" x=\"742\" y=\"-7.8\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">class = Compra</text>\r\n</g>\r\n<!-- 12&#45;&gt;13 -->\r\n<g id=\"edge13\" class=\"edge\"><title>12&#45;&gt;13</title>\r\n<path fill=\"none\" stroke=\"black\" d=\"M746.393,-103.726C745.863,-95.4263 745.303,-86.6671 744.767,-78.2834\"/>\r\n<polygon fill=\"black\" stroke=\"black\" points=\"748.26,-78.0559 744.129,-68.2996 741.274,-78.5025 748.26,-78.0559\"/>\r\n</g>\r\n<!-- 14 -->\r\n<g id=\"node15\" class=\"node\"><title>14</title>\r\n<polygon fill=\"none\" stroke=\"black\" points=\"988,-68 836,-68 836,-0 988,-0 988,-68\"/>\r\n<text text-anchor=\"middle\" x=\"912\" y=\"-52.8\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">gini = 0.229</text>\r\n<text text-anchor=\"middle\" x=\"912\" y=\"-37.8\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">samples = 95139</text>\r\n<text text-anchor=\"middle\" x=\"912\" y=\"-22.8\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">value = [130399, 19870]</text>\r\n<text text-anchor=\"middle\" x=\"912\" y=\"-7.8\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">class = Compra</text>\r\n</g>\r\n<!-- 12&#45;&gt;14 -->\r\n<g id=\"edge14\" class=\"edge\"><title>12&#45;&gt;14</title>\r\n<path fill=\"none\" stroke=\"black\" d=\"M809.695,-103.726C824.298,-93.9161 839.857,-83.4644 854.318,-73.7496\"/>\r\n<polygon fill=\"black\" stroke=\"black\" points=\"856.341,-76.6069 862.69,-68.1252 852.438,-70.7963 856.341,-76.6069\"/>\r\n</g>\r\n</g>\r\n</svg>\r\n",
      "text/plain": [
       "<graphviz.files.Source at 0x1fe232c1640>"
      ]
     },
     "execution_count": 75,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "graphviz.Source(dot_data_rf_1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 76,
   "metadata": {
    "azdata_cell_guid": "03a664ca-f3eb-4d2f-80f7-f00848c6ffb7"
   },
   "outputs": [],
   "source": [
    "top_features = pd.Series(rf.feature_importances_, index = X_train.columns).sort_values(ascending = False).head(20)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 77,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Productos_em_acount            0.380135\n",
       "customer_count                 0.226547\n",
       "recurrencia                    0.124398\n",
       "Productos_debit_card           0.080845\n",
       "entry_channel_KFC              0.079647\n",
       "days_between                   0.027606\n",
       "entry_channel_KAT              0.024427\n",
       "entry_channel_RED              0.013931\n",
       "Productos_payroll_account      0.012505\n",
       "entry_channel_KHQ              0.005875\n",
       "Productos_long_term_deposit    0.004975\n",
       "age                            0.004237\n",
       "entry_channel_KFA              0.003848\n",
       "Productos_emc_account          0.002649\n",
       "Productos_payroll              0.002551\n",
       "entry_channel_KHL              0.002356\n",
       "gender_V                       0.001780\n",
       "Productos_credit_card          0.001098\n",
       "region_code                    0.000282\n",
       "gender_H                       0.000276\n",
       "dtype: float64"
      ]
     },
     "execution_count": 77,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "top_features"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 78,
   "metadata": {
    "azdata_cell_guid": "1281cfe6-1e1e-4c1e-a6f5-c8537a1f1cce"
   },
   "outputs": [],
   "source": [
    "y_score = pd.DataFrame(rf.predict_proba(X_test)[:,1], index= y_test.index, columns=['PredictScore'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 80,
   "metadata": {
    "azdata_cell_guid": "b86df0ea-a456-4230-bc1c-66336dcf9106"
   },
   "outputs": [],
   "source": [
    "rf_results_df = y_test.join(y_score)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 81,
   "metadata": {
    "azdata_cell_guid": "f3bf26e2-c47f-48bc-adb5-4d45895498fd"
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Productos_pension_plan</th>\n",
       "      <th>PredictScore</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>1541005</th>\n",
       "      <td>0</td>\n",
       "      <td>0.006537</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1605463</th>\n",
       "      <td>0</td>\n",
       "      <td>0.006537</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4733672</th>\n",
       "      <td>0</td>\n",
       "      <td>0.132065</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2547680</th>\n",
       "      <td>0</td>\n",
       "      <td>0.006537</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3180461</th>\n",
       "      <td>0</td>\n",
       "      <td>0.054576</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1611565</th>\n",
       "      <td>0</td>\n",
       "      <td>0.006537</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>5705683</th>\n",
       "      <td>0</td>\n",
       "      <td>0.006537</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3743435</th>\n",
       "      <td>0</td>\n",
       "      <td>0.006537</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1903762</th>\n",
       "      <td>0</td>\n",
       "      <td>0.042438</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1872194</th>\n",
       "      <td>0</td>\n",
       "      <td>0.006537</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "         Productos_pension_plan  PredictScore\n",
       "1541005                       0      0.006537\n",
       "1605463                       0      0.006537\n",
       "4733672                       0      0.132065\n",
       "2547680                       0      0.006537\n",
       "3180461                       0      0.054576\n",
       "1611565                       0      0.006537\n",
       "5705683                       0      0.006537\n",
       "3743435                       0      0.006537\n",
       "1903762                       0      0.042438\n",
       "1872194                       0      0.006537"
      ]
     },
     "execution_count": 81,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "rf_results_df.head(10)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 82,
   "metadata": {
    "azdata_cell_guid": "d91dcaf9-f374-48ad-9a36-e42669e27c5a"
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.933763285364321"
      ]
     },
     "execution_count": 82,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "metrics.roc_auc_score(rf_results_df[TARGET1], rf_results_df['PredictScore'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 83,
   "metadata": {
    "azdata_cell_guid": "571f53d5-dac8-4252-b8af-9f5db4d95831"
   },
   "outputs": [],
   "source": [
    "fpr, tpr, thre = metrics.roc_curve(rf_results_df[TARGET1], rf_results_df['PredictScore'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 84,
   "metadata": {
    "azdata_cell_guid": "c2eba321-4ca3-4201-a3b5-3777b74884a9"
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "[<matplotlib.lines.Line2D at 0x1fe422f3190>]"
      ]
     },
     "execution_count": 84,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.plot(fpr,tpr)\n",
    "plt.plot([0,1],[0,1], color = 'grey')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "azdata_cell_guid": "9b413f60-aa72-4563-88c5-1f23fdab9808"
   },
   "source": [
    "## Valoremos aplicar Gradient boosting"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 85,
   "metadata": {
    "azdata_cell_guid": "cf89cbf2-7801-451b-be4b-b65405121d1d"
   },
   "outputs": [],
   "source": [
    "import xgboost as xgb"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 86,
   "metadata": {
    "azdata_cell_guid": "11b03064-831d-4262-a0eb-022cb675708a"
   },
   "outputs": [],
   "source": [
    "xgb_model = xgb.XGBClassifier(max_depth = 5, n_estimators = 80, random_state = 42)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 87,
   "metadata": {
    "azdata_cell_guid": "f58c8bd4-f574-4dc1-bf9b-0472191b460e"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[10:31:25] WARNING: C:/Users/Administrator/workspace/xgboost-win64_release_1.4.0/src/learner.cc:1095: Starting in XGBoost 1.3.0, the default evaluation metric used with the objective 'binary:logistic' was changed from 'error' to 'logloss'. Explicitly set eval_metric if you'd like to restore the old behavior.\n"
     ]
    }
   ],
   "source": [
    "xgb_model.fit(X_train,y_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "azdata_cell_guid": "59215585-fd5e-4a06-8809-498b310858cb"
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.9983604965930476"
      ]
     },
     "execution_count": 148,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "metrics.roc_auc_score(y_test, xgb_model.predict_proba(X_test)[:,1])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "azdata_cell_guid": "17d137ca-f633-41a4-9382-e4eed35d1117"
   },
   "outputs": [],
   "source": [
    "fpr, tpr, thre = metrics.roc_curve(y_test, xgb_model.predict_proba(X_test)[:,1])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "azdata_cell_guid": "c053ff36-551a-48f4-843b-6a72a069b35d"
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "[<matplotlib.lines.Line2D at 0x21039802670>]"
      ]
     },
     "execution_count": 150,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.plot(fpr,tpr)\n",
    "plt.plot([0,1],[0,1], color = 'grey')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "azdata_cell_guid": "19cbfc4e-729a-4f13-8515-f011279e0691"
   },
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "azdata_cell_guid": "6cd04b55-09db-4c88-ab46-06004debb21b"
   },
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "azdata_cell_guid": "3123bc85-8ba7-413e-9e40-d37a90bdaa1b"
   },
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "azdata_cell_guid": "0283c188-f787-4564-8403-7cd2cca47ef2"
   },
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "azdata_cell_guid": "4b7f9341-1080-4444-b17f-e93bc4269769"
   },
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "azdata_cell_guid": "1f92b8a4-ae80-48ae-b189-b8685717ed6d"
   },
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "azdata_cell_guid": "ed084fc7-7227-4fca-a40b-02af6d46eac6"
   },
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "azdata_cell_guid": "5746858b-c4d6-4b35-8444-3cdbdacc72b3"
   },
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "azdata_cell_guid": "d4b5f6eb-cc6c-4182-a4d3-b287f94fabf6"
   },
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "azdata_cell_guid": "09141c62-5dd1-42b2-800c-5179a1463eb8"
   },
   "source": [
    "### NO SUPERVISADO - SEGMENTACIÓN"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 152,
   "metadata": {
    "azdata_cell_guid": "41b3724f-c128-43c6-ad16-2a4c0be6078f"
   },
   "outputs": [
    {
     "ename": "NameError",
     "evalue": "name 'full_kmeans' is not defined",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mNameError\u001b[0m                                 Traceback (most recent call last)",
      "\u001b[1;32m<ipython-input-152-9c4966fb2cb0>\u001b[0m in \u001b[0;36m<module>\u001b[1;34m\u001b[0m\n\u001b[0;32m      1\u001b[0m \u001b[0mstandard_scaler\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mStandardScaler\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m----> 2\u001b[1;33m \u001b[0mscaled_df\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mstandard_scaler\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mfit_transform\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mfull_kmeans\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m      3\u001b[0m \u001b[0mscaled_df\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mpd\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mDataFrame\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mscaled_df\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mindex\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mfull_kmeans\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mindex\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mcolumns\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mfull_kmeans\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mcolumns\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m      4\u001b[0m \u001b[0mscaled_df\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mshape\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;31mNameError\u001b[0m: name 'full_kmeans' is not defined"
     ]
    }
   ],
   "source": [
    "\n",
    "standard_scaler = StandardScaler()\n",
    "scaled_df = standard_scaler.fit_transform(full_kmeans)\n",
    "scaled_df = pd.DataFrame(scaled_df, index = full_kmeans.index, columns = full_kmeans.columns)\n",
    "scaled_df.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "azdata_cell_guid": "de4be44b-29ab-4157-8833-da597eda7a2d"
   },
   "outputs": [],
   "source": [
    "# import the function to compute cosine_similarity\n",
    "from sklearn.metrics.pairwise import cosine_similarity\n",
    "from sklearn.decomposition import PCA\n",
    "from sklearn.cluster import KMeans\n",
    "from sklearn.impute import SimpleImputer"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "azdata_cell_guid": "4351d410-3418-4b04-975d-5a2dd10b32ce"
   },
   "source": [
    "Reducción de la dimensionalidad con PCA "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "azdata_cell_guid": "41afa36a-a993-4735-8cca-dc90894cce8a"
   },
   "outputs": [],
   "source": [
    "CALCULATE_ELBOW = True\n",
    "\n",
    "if CALCULATE_ELBOW:\n",
    "    st = time.time()\n",
    "\n",
    "    sse = {}\n",
    "\n",
    "    for k in range(2, 15): #todo los clusteres que queramos ver en la curva\n",
    "\n",
    "        print(f\"Fitting pipe with {k} clusters\")\n",
    "        cluster_model = KMeans(n_clusters = k)\n",
    "        cluster_model.fit(scaled_df)\n",
    "\n",
    "        sse[k] = cluster_model.inertia_\n",
    "\n",
    "    et = time.time()\n",
    "    print(\"Elbow curve took {} minutes.\".format(round((et - st)/60), 2))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "azdata_cell_guid": "1046d9fb-d02d-46d4-945d-bbe86964f3d6"
   },
   "outputs": [],
   "source": [
    "if CALCULATE_ELBOW:\n",
    "    fig = plt.figure(figsize = (16, 8))\n",
    "    ax = fig.add_subplot()\n",
    "\n",
    "    x_values = list(sse.keys())\n",
    "    y_values = list(sse.values())\n",
    "\n",
    "    ax.plot(x_values, y_values, label = \"Inertia/dispersión de los clústers\")\n",
    "    fig.suptitle(\"Variación de la dispersión de los clústers en función de la k\", fontsize = 16);"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "azdata_cell_guid": "6c580fd7-3b03-4c87-9c13-142d0866e5e1"
   },
   "outputs": [],
   "source": [
    "kmeans = KMeans(n_clusters = 7)\n",
    "kmeans.fit(scaled_df)\n",
    "full_kmeans['cluster'] = kmeans.labels_"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "azdata_cell_guid": "268a7b64-740e-4b89-a1c4-79c239fdad34"
   },
   "outputs": [],
   "source": [
    "# fiteamos un modelo con k = 5 (que hemos sacado de la elbow curve anterior) \n",
    "# y con el dataframe escalado y sin outliers\n",
    "\n",
    "cluster_model = KMeans(n_clusters = 7)\n",
    "cluster_model.fit(scaled_df)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "azdata_cell_guid": "f52a2303-5ca1-4ef6-9b52-3bb54b538de0"
   },
   "outputs": [],
   "source": [
    "# generamos el dataframe escalado (con el scaler del paso anterior, entrado sin outliers) pero con todos los datos.\n",
    "# por tanto vamos a transformar incluso a los outliers pero con el scaler entrado sin ellos.\n",
    "# el motivo es porque los outliers pueden afectar mucho la media y la desviación utilizado para transformar.\n",
    "scaled_df_with_outliers = standard_scaler.transform(scaled_df)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "azdata_cell_guid": "4fbbfd73-7221-487a-9500-4de8ae9fbfcf"
   },
   "outputs": [],
   "source": [
    "# convertimos a dataframe\n",
    "scaled_df_with_outliers = pd.DataFrame(scaled_df_with_outliers, \n",
    "                                       index = scaled_df.index, \n",
    "                                       columns = scaled_df.columns)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "azdata_cell_guid": "47fd721c-2bbb-44c8-b841-aa0934cd1856"
   },
   "outputs": [],
   "source": [
    "scaled_df_with_outliers.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "azdata_cell_guid": "df69b23e-f39d-46dd-8d01-82b37627f484"
   },
   "outputs": [],
   "source": [
    "# calculamos el cluster de cada cliente, a partir del dataframe escalado y con outliers\n",
    "labels = cluster_model.predict(scaled_df_with_outliers)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "azdata_cell_guid": "ef125b83-8447-4326-9dd7-56db90c5ad4b"
   },
   "outputs": [],
   "source": [
    "scaled_df[\"cluster\"] = labels\n",
    "scaled_df.head(15).T"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "azdata_cell_guid": "d983d911-735e-4b41-a188-2153c4a4ac5f"
   },
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "azdata_cell_guid": "93b15f24-ec9f-44c2-b68d-d0c2bee7b507"
   },
   "outputs": [],
   "source": [
    "scaled_df.columns"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "azdata_cell_guid": "d59eaf35-1c95-43ce-9f2e-25b8d90dd6a2"
   },
   "outputs": [],
   "source": [
    "pd.to_pickle(scaled_df, './full_4.pickle')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "azdata_cell_guid": "23304a0b-3298-4db4-b7c9-bccc78eb8a8f"
   },
   "outputs": [],
   "source": [
    "# visualizamos nuestros grupos en base a las variables del modelo RFM, para ver que tal han quedado.\n",
    "selected_columns = ['Ingresos', 'días_encartera', 'Compras','Familia_prod_AhorroVista', 'Familia_prod_Crédito',\n",
    "       'Familia_prod_Inversión']\n",
    "\n",
    "sns.pairplot(scaled_df, vars = selected_columns, hue = 'cluster');"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "azdata_cell_guid": "85c2f52a-b663-49e9-8721-0d0854a910d4"
   },
   "outputs": [],
   "source": [
    "## Recomendación \"user based\""
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "azdata_cell_guid": "37b90ab8-1498-4570-923b-295e54d70391"
   },
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "celltoolbar": "Slideshow",
  "interpreter": {
   "hash": "02b54ae4fcef88d4936c518f12e4c1a4a3ccb83eb22d3060789450414c249266"
  },
  "kernelspec": {
   "display_name": "Python 3.8.5 64-bit ('base': conda)",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.5"
  },
  "varInspector": {
   "cols": {
    "lenName": 16,
    "lenType": 16,
    "lenVar": 40
   },
   "kernels_config": {
    "python": {
     "delete_cmd_postfix": "",
     "delete_cmd_prefix": "del ",
     "library": "var_list.py",
     "varRefreshCmd": "print(var_dic_list())"
    },
    "r": {
     "delete_cmd_postfix": ") ",
     "delete_cmd_prefix": "rm(",
     "library": "var_list.r",
     "varRefreshCmd": "cat(var_dic_list()) "
    }
   },
   "types_to_exclude": [
    "module",
    "function",
    "builtin_function_or_method",
    "instance",
    "_Feature"
   ],
   "window_display": false
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}