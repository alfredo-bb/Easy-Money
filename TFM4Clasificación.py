{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "source": [
    "import squarify\r\n",
    "import os\r\n",
    "    \r\n",
    "# Main libraries that we will use in this kernel\r\n",
    "import datetime\r\n",
    "import numpy as np\r\n",
    "import pandas as pd\r\n",
    "\r\n",
    "# # garbage collector: free some memory is needed\r\n",
    "import gc\r\n",
    "import matplotlib\r\n",
    "import matplotlib.pyplot as plt\r\n",
    "import plotly.express as px\r\n",
    "import seaborn as sns\r\n",
    "\r\n",
    "# statistical package and some useful functions to analyze our timeseries\r\n",
    "from statsmodels.tsa.stattools import pacf\r\n",
    "from statsmodels.graphics.tsaplots import plot_acf, plot_pacf\r\n",
    "from statsmodels.tsa.seasonal import seasonal_decompose\r\n",
    "import statsmodels.tsa.stattools as stattools\r\n",
    "\r\n",
    "import time\r\n",
    "\r\n",
    "from string import punctuation\r\n",
    "from sklearn.preprocessing import LabelEncoder\r\n",
    "from sklearn.linear_model import LinearRegression\r\n",
    "\r\n",
    "import warnings\r\n",
    "warnings.filterwarnings(\"ignore\")"
   ],
   "outputs": [],
   "metadata": {
    "azdata_cell_guid": "f0e72443-ffd0-4847-9a85-1c375a72bc4f"
   }
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "source": [
    "from sklearn import preprocessing"
   ],
   "outputs": [],
   "metadata": {
    "azdata_cell_guid": "10eb9938-bb1c-4a6f-b584-2907aa120466"
   }
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "source": [
    "# python core library for machine learning and data science\r\n",
    "from sklearn.pipeline import Pipeline\r\n",
    "from sklearn.preprocessing import StandardScaler, MinMaxScaler\r\n",
    "from sklearn.base import BaseEstimator, TransformerMixin\r\n",
    "from sklearn.impute import KNNImputer, SimpleImputer\r\n",
    "from sklearn.cluster import KMeans"
   ],
   "outputs": [],
   "metadata": {
    "azdata_cell_guid": "bc7f16db-3af4-4634-82d5-881a8786ce32"
   }
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "source": [
    "full_3OHE = pd.read_pickle('./full_3OHE.pickle')"
   ],
   "outputs": [],
   "metadata": {
    "azdata_cell_guid": "292a7689-642e-40df-a221-853a7efc034e"
   }
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "source": [
    "import plotly.express as px"
   ],
   "outputs": [],
   "metadata": {
    "azdata_cell_guid": "2730be9d-6657-4e9d-808b-26eec38d75be"
   }
  },
  {
   "cell_type": "markdown",
   "source": [
    "## DATASET PARA CLASIFICACIÓN BINARIA"
   ],
   "metadata": {}
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "source": [
    "full_3OHE.select_dtypes(include=['object']).describe().T"
   ],
   "outputs": [
    {
     "output_type": "execute_result",
     "data": {
      "text/plain": [
       "          count  unique      top freq\n",
       "pk_cid  6254518  350384  1128353  149"
      ],
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
      ]
     },
     "metadata": {},
     "execution_count": 6
    }
   ],
   "metadata": {
    "scrolled": true
   }
  },
  {
   "cell_type": "markdown",
   "source": [
    "### Aplicamos un Frequency Label PARA LA ÚNICA VBLE CATEGÓRICA QUE NOS QUEDA"
   ],
   "metadata": {}
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "source": [
    "customer_num = pd.DataFrame(full_3OHE['pk_cid'].value_counts(dropna = False))\r\n",
    "customer_num.columns = ['customer_count']\r\n",
    "customer_num['pk_cid'] = customer_num.index"
   ],
   "outputs": [],
   "metadata": {}
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "source": [
    "df = full_3OHE.merge(customer_num, on = 'pk_cid')"
   ],
   "outputs": [],
   "metadata": {
    "azdata_cell_guid": "1879dcf3-3eb2-4268-b020-aa2ebaa7c5ca"
   }
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "source": [
    "df.select_dtypes(include=['object']).describe().T"
   ],
   "outputs": [
    {
     "output_type": "execute_result",
     "data": {
      "text/plain": [
       "          count  unique      top freq\n",
       "pk_cid  6254518  350384  1128353  149"
      ],
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
      ]
     },
     "metadata": {},
     "execution_count": 9
    }
   ],
   "metadata": {
    "scrolled": true
   }
  },
  {
   "cell_type": "markdown",
   "source": [
    "## Definamos el target\r\n"
   ],
   "metadata": {}
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "source": [
    "Target = 'Productos_pension_plan'"
   ],
   "outputs": [],
   "metadata": {}
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "source": [
    "df.pivot_table(index=['Year','Month'], values=Target, aggfunc=[len, sum, np.mean])"
   ],
   "outputs": [
    {
     "output_type": "execute_result",
     "data": {
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
      ],
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
      ]
     },
     "metadata": {},
     "execution_count": 11
    }
   ],
   "metadata": {}
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "source": [
    "df.drop(['Ingresos','Ventas_cant','pk_cid','Precios','Familia_prod_AhorroVista','Familia_prod_Inversión','Familia_prod_Crédito'], axis = 1, inplace = True)"
   ],
   "outputs": [],
   "metadata": {
    "azdata_cell_guid": "7a3aaab9-1caa-40b2-ad7a-84211c80e879"
   }
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "source": [
    "df.drop(['country_id_AR',  'country_id_AT' , 'country_id_BE' , 'country_id_BR'  ,'country_id_CA' , 'country_id_CH' , 'country_id_CI' , 'country_id_CL' , 'country_id_CM' , 'country_id_CN' , 'country_id_CO' , 'country_id_DE' , 'country_id_DO'  ,'country_id_DZ'  ,'country_id_ES' , 'country_id_ET' , 'country_id_FR'  ,'country_id_GA'  ,'country_id_GB' , 'country_id_GT' , 'country_id_IE'  ,'country_id_IT'  ,'country_id_LU'  ,'country_id_MA' , 'country_id_MR' , 'country_id_MX'  ,'country_id_NO' , 'country_id_PE' , 'country_id_PL' , 'country_id_QA',  'country_id_RO' , 'country_id_RU' ,'country_id_SA' , 'country_id_SE',  'country_id_SN' , 'country_id_US' , 'country_id_VE'], axis = 1, inplace = True)"
   ],
   "outputs": [],
   "metadata": {}
  },
  {
   "cell_type": "markdown",
   "source": [
    "# TRABAJEMOS LA CLASIFICACIÓN"
   ],
   "metadata": {}
  },
  {
   "cell_type": "markdown",
   "source": [
    "#### Para este caso, podemos por ejemplo validar con los últimos 6 meses de 2019 y utilizar 2018 y los 6 primeros de 2019 para realizar el entrenamiento del modelo (train y test). Para seleccionarlo, utilizaremos de nuevo el boolean indexing"
   ],
   "metadata": {}
  },
  {
   "cell_type": "markdown",
   "source": [
    "### Trabajemos con Decition Three"
   ],
   "metadata": {}
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "source": [
    "df[df[Target]==1][Target].describe()\r\n",
    "\r\n",
    "\r\n"
   ],
   "outputs": [
    {
     "output_type": "execute_result",
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
     "metadata": {},
     "execution_count": 14
    }
   ],
   "metadata": {}
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "source": [
    "\r\n",
    "df[df[Target]==0][Target].describe()"
   ],
   "outputs": [
    {
     "output_type": "execute_result",
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
     "metadata": {},
     "execution_count": 15
    }
   ],
   "metadata": {}
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "source": [
    "df['Compra'] = (df[Target] == 1).astype(int)#esta variable es un indicador que me indica cuando alguien ha comprado, la creamos a partir del target. Cuando entrenemos el modelo esta variable no puede estar porque qle seria facil predecir.\r\n"
   ],
   "outputs": [],
   "metadata": {}
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "source": [
    "count_transac= df['Compra'].count()\r\n",
    "sum_transac= df['Compra'].sum()\r\n",
    "mean_transac= df['Compra'].mean()\r\n",
    "print(f\"hay un total de {count_transac}visitas, de las cuales {sum_transac} acaban con compra, lo que representa un {round(mean_transac*100,4)}%\")"
   ],
   "outputs": [
    {
     "output_type": "stream",
     "name": "stdout",
     "text": [
      "hay un total de 6254518visitas, de las cuales 217802 acaban con compra, lo que representa un 3.4823%\n"
     ]
    }
   ],
   "metadata": {}
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "source": [
    "df.drop('Compra',axis=1,inplace=True)"
   ],
   "outputs": [],
   "metadata": {}
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "source": [
    "dev_df = df[(df['Year'] == 2018) | ((df['Year'] == 2019) & (df['Month'] < 4)) ] # development = train + test\r\n",
    "val_df = df[(df['Year'] == 2019) & (df['Month'] >= 4)] # validation"
   ],
   "outputs": [],
   "metadata": {}
  },
  {
   "cell_type": "markdown",
   "source": [
    "Una vez realizada la selección de particiones, vamos a asignar los atributos y el target a las variables X e y, respectivamente.\\\r\n",
    "Para seleccionar, utilizaremos el método Drop sin el atributo inplace y haciendo asignación y la indexación directa (ojo con el doble claudator, usaremos un DataFrame... si usasemos el claudator simple sacaríamos una Serie)."
   ],
   "metadata": {}
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "source": [
    "dev_df.shape"
   ],
   "outputs": [
    {
     "output_type": "execute_result",
     "data": {
      "text/plain": [
       "(5383155, 86)"
      ]
     },
     "metadata": {},
     "execution_count": 20
    }
   ],
   "metadata": {}
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "source": [
    "val_df.shape"
   ],
   "outputs": [
    {
     "output_type": "execute_result",
     "data": {
      "text/plain": [
       "(871363, 86)"
      ]
     },
     "metadata": {},
     "execution_count": 21
    }
   ],
   "metadata": {}
  },
  {
   "cell_type": "markdown",
   "source": [
    "## Veamos el balanceo con respecto al target"
   ],
   "metadata": {}
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "source": [
    "dev_df_1 = dev_df[dev_df[Target]==1]\r\n",
    "dev_df_0 = dev_df[dev_df[Target]==0]\r\n",
    "\r\n",
    "dev_df_0_shape = dev_df_0.shape[0]\r\n",
    "dev_df_0_shape\r\n",
    "\r\n",
    "dev_df_1_shape = dev_df_1.shape[0]\r\n",
    "dev_df_1_shape\r\n",
    "prop = dev_df_0_shape/dev_df_1_shape\r\n",
    "prop"
   ],
   "outputs": [
    {
     "output_type": "execute_result",
     "data": {
      "text/plain": [
       "28.21991108891651"
      ]
     },
     "metadata": {},
     "execution_count": 22
    }
   ],
   "metadata": {}
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "source": [
    "dev_df_0_shape"
   ],
   "outputs": [
    {
     "output_type": "execute_result",
     "data": {
      "text/plain": [
       "5198926"
      ]
     },
     "metadata": {},
     "execution_count": 23
    }
   ],
   "metadata": {}
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "source": [
    "dev_df_1_shape"
   ],
   "outputs": [
    {
     "output_type": "execute_result",
     "data": {
      "text/plain": [
       "184229"
      ]
     },
     "metadata": {},
     "execution_count": 24
    }
   ],
   "metadata": {}
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "source": [
    "5198926/184229"
   ],
   "outputs": [
    {
     "output_type": "execute_result",
     "data": {
      "text/plain": [
       "28.21991108891651"
      ]
     },
     "metadata": {},
     "execution_count": 25
    }
   ],
   "metadata": {}
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "source": [
    "## Debemos rebalancear el dataset\r\n",
    "\r\n",
    "dev_df_1 = dev_df[dev_df[Target]==1]\r\n",
    "dev_df_0 = dev_df[dev_df[Target]==0]\r\n",
    "dev_df_0_shape = dev_df_0.shape[0]\r\n",
    "\r\n",
    "dev_df_1_shape = dev_df_1.shape[0]*28\r\n",
    "dev_df_1_shape\r\n",
    "\r\n",
    "prop_dev = dev_df_0_shape/dev_df_1_shape\r\n",
    "prop_dev\r\n"
   ],
   "outputs": [
    {
     "output_type": "execute_result",
     "data": {
      "text/plain": [
       "1.007853967461304"
      ]
     },
     "metadata": {},
     "execution_count": 26
    }
   ],
   "metadata": {
    "azdata_cell_guid": "fe0ab95a-2889-4599-aa0d-49b9dc6b150d"
   }
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "source": [
    "RANDOM_STATE = 42\r\n",
    "\r\n",
    "dev_df_0_sample = dev_df_0.sample(n=dev_df_1_shape, random_state=RANDOM_STATE)\r\n",
    "\r\n",
    "\r\n",
    "dev_df_sample = pd.concat([dev_df_1,dev_df_0_sample])"
   ],
   "outputs": [],
   "metadata": {}
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "source": [
    "dev_df_0_sample.shape[0]/dev_df_sample.shape[0]"
   ],
   "outputs": [
    {
     "output_type": "execute_result",
     "data": {
      "text/plain": [
       "0.9655172413793104"
      ]
     },
     "metadata": {},
     "execution_count": 28
    }
   ],
   "metadata": {}
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "source": [
    "dev_df =dev_df_sample"
   ],
   "outputs": [],
   "metadata": {}
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "source": [
    "dev_df_X = dev_df.drop(Target, axis = 1)\r\n",
    "dev_df_y = dev_df[[Target]]"
   ],
   "outputs": [],
   "metadata": {
    "azdata_cell_guid": "a22f7811-3145-414a-b107-369226302a76"
   }
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "source": [
    "val_df_X = val_df.drop(Target, axis = 1)\r\n",
    "val_df_y = val_df[[Target]]"
   ],
   "outputs": [],
   "metadata": {
    "azdata_cell_guid": "8d2f1b95-66bd-4c25-88ab-6b5d2ac9ca5c",
    "tags": []
   }
  },
  {
   "cell_type": "code",
   "execution_count": 32,
   "source": [
    "from sklearn import model_selection\r\n",
    "from sklearn import metrics"
   ],
   "outputs": [],
   "metadata": {
    "azdata_cell_guid": "c6ca6d62-df28-4364-9757-7f4b23c4968b"
   }
  },
  {
   "cell_type": "code",
   "execution_count": 33,
   "source": [
    "X_train, X_test, y_train, y_test = model_selection.train_test_split(dev_df_X, dev_df_y, test_size=0.2, random_state=42)"
   ],
   "outputs": [],
   "metadata": {
    "azdata_cell_guid": "8b4613eb-6989-42fe-8cc7-21888f9f7b88"
   }
  },
  {
   "cell_type": "code",
   "execution_count": 34,
   "source": [
    "y_train.describe().T.head()"
   ],
   "outputs": [
    {
     "output_type": "execute_result",
     "data": {
      "text/plain": [
       "                            count      mean       std  min  25%  50%  75%  max\n",
       "Productos_pension_plan  4274112.0  0.034466  0.182423  0.0  0.0  0.0  0.0  1.0"
      ],
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
       "      <th>mean</th>\n",
       "      <th>std</th>\n",
       "      <th>min</th>\n",
       "      <th>25%</th>\n",
       "      <th>50%</th>\n",
       "      <th>75%</th>\n",
       "      <th>max</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>Productos_pension_plan</th>\n",
       "      <td>4274112.0</td>\n",
       "      <td>0.034466</td>\n",
       "      <td>0.182423</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>1.0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ]
     },
     "metadata": {},
     "execution_count": 34
    }
   ],
   "metadata": {}
  },
  {
   "cell_type": "code",
   "execution_count": 35,
   "source": [
    "y_test.describe().T.head()"
   ],
   "outputs": [
    {
     "output_type": "execute_result",
     "data": {
      "text/plain": [
       "                            count      mean       std  min  25%  50%  75%  max\n",
       "Productos_pension_plan  1068529.0  0.034549  0.182636  0.0  0.0  0.0  0.0  1.0"
      ],
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
       "      <th>mean</th>\n",
       "      <th>std</th>\n",
       "      <th>min</th>\n",
       "      <th>25%</th>\n",
       "      <th>50%</th>\n",
       "      <th>75%</th>\n",
       "      <th>max</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>Productos_pension_plan</th>\n",
       "      <td>1068529.0</td>\n",
       "      <td>0.034549</td>\n",
       "      <td>0.182636</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>1.0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ]
     },
     "metadata": {},
     "execution_count": 35
    }
   ],
   "metadata": {}
  },
  {
   "cell_type": "markdown",
   "source": [
    " ##  MODEL SELECTION\r\n",
    " ### Importar los scikits de modelización\r\n"
   ],
   "metadata": {}
  },
  {
   "cell_type": "code",
   "execution_count": 36,
   "source": [
    "!conda install python-graphviz -y"
   ],
   "outputs": [
    {
     "output_type": "stream",
     "name": "stdout",
     "text": [
      "Collecting package metadata (current_repodata.json): ...working... done\n",
      "Solving environment: ...working... done\n",
      "\n",
      "# All requested packages already installed.\n",
      "\n"
     ]
    },
    {
     "output_type": "stream",
     "name": "stderr",
     "text": [
      "Did not find path entry C:\\Users\\Guica\\anaconda3\\bin\n"
     ]
    }
   ],
   "metadata": {
    "azdata_cell_guid": "f2b346fd-c44c-4504-86da-59fce1eb414a"
   }
  },
  {
   "cell_type": "code",
   "execution_count": 37,
   "source": [
    "!conda install pydot -y"
   ],
   "outputs": [
    {
     "output_type": "stream",
     "name": "stdout",
     "text": [
      "Collecting package metadata (current_repodata.json): ...working... done\n",
      "Solving environment: ...working... done\n",
      "\n",
      "# All requested packages already installed.\n",
      "\n"
     ]
    },
    {
     "output_type": "stream",
     "name": "stderr",
     "text": [
      "Did not find path entry C:\\Users\\Guica\\anaconda3\\bin\n"
     ]
    }
   ],
   "metadata": {
    "azdata_cell_guid": "f6def2ca-66b7-4e68-8b4b-4f930b89ef29"
   }
  },
  {
   "cell_type": "code",
   "execution_count": 38,
   "source": [
    "from sklearn.tree import DecisionTreeClassifier\r\n",
    "from sklearn.tree import export_graphviz\r\n",
    "import graphviz"
   ],
   "outputs": [],
   "metadata": {
    "azdata_cell_guid": "4e078a97-54c0-4544-87f4-e15583afc390"
   }
  },
  {
   "cell_type": "markdown",
   "source": [
    "### Decision Three"
   ],
   "metadata": {}
  },
  {
   "cell_type": "code",
   "execution_count": 39,
   "source": [
    "dt = DecisionTreeClassifier(\r\n",
    "\r\n",
    "                 #criterion=\"gini\",\r\n",
    "                 #splitter=\"best\",\r\n",
    "                 max_depth= 4, # número de preguntas que realiza el algoritmo\r\n",
    "                 #min_samples_split=2, # numero de obs minimas en cada particion\r\n",
    "                 min_samples_leaf=50,\r\n",
    "                 #min_weight_fraction_leaf=0.,\r\n",
    "                 #max_features=None, #numero de variables a evaluar en cada iteracion\r\n",
    "                 random_state=42\r\n",
    "                 #max_leaf_nodes=None,\r\n",
    "                 #min_impurity_decrease=0.,\r\n",
    "                 #min_impurity_split=None,\r\n",
    "                 #class_weight=None,\r\n",
    "                 #presort='deprecated'\r\n",
    ")"
   ],
   "outputs": [],
   "metadata": {
    "azdata_cell_guid": "eb325d69-02f7-418f-9e43-35f305662d98"
   }
  },
  {
   "cell_type": "code",
   "execution_count": 40,
   "source": [
    "dt.fit(\r\n",
    "    X=X_train, \r\n",
    "    y=y_train, \r\n",
    "    # sample_weight=None, \r\n",
    "    # check_input=True, \r\n",
    "    # X_idx_sorted=None\r\n",
    ")"
   ],
   "outputs": [
    {
     "output_type": "execute_result",
     "data": {
      "text/plain": [
       "DecisionTreeClassifier(max_depth=4, min_samples_leaf=50, random_state=42)"
      ]
     },
     "metadata": {},
     "execution_count": 40
    }
   ],
   "metadata": {
    "azdata_cell_guid": "87f3948f-b65c-43c5-8a1e-255eb6fb79f7"
   }
  },
  {
   "cell_type": "code",
   "execution_count": 41,
   "source": [
    "dot_data = export_graphviz(\r\n",
    "                        decision_tree = dt,\r\n",
    "                        out_file=None,\r\n",
    "                        # max_depth=None,\r\n",
    "                        feature_names=X_test.columns,\r\n",
    "                        class_names=['Compra','No_compra'],\r\n",
    "                        # label='all',\r\n",
    "                        filled=True,\r\n",
    "                        # leaves_parallel=False,\r\n",
    "                        impurity=True,\r\n",
    "                        # node_ids=False,\r\n",
    "                        proportion=True,\r\n",
    "                        rotate=True,\r\n",
    "                        rounded=True,\r\n",
    "                        # special_characters=False,\r\n",
    "                        precision=4,\r\n",
    "                        )"
   ],
   "outputs": [],
   "metadata": {
    "azdata_cell_guid": "b990c869-a10d-4d9a-9703-895d5e197c39"
   }
  },
  {
   "cell_type": "code",
   "execution_count": 42,
   "source": [
    "graph = graphviz.Source(dot_data)\r\n",
    "graph"
   ],
   "outputs": [
    {
     "output_type": "execute_result",
     "data": {
      "text/plain": [
       "<graphviz.files.Source at 0x2bb20e61a90>"
      ],
      "image/svg+xml": "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"no\"?>\r\n<!DOCTYPE svg PUBLIC \"-//W3C//DTD SVG 1.1//EN\"\r\n \"http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd\">\r\n<!-- Generated by graphviz version 2.38.0 (20140413.2041)\r\n -->\r\n<!-- Title: Tree Pages: 1 -->\r\n<svg width=\"1076pt\" height=\"850pt\"\r\n viewBox=\"0.00 0.00 1076.00 850.00\" xmlns=\"http://www.w3.org/2000/svg\" xmlns:xlink=\"http://www.w3.org/1999/xlink\">\r\n<g id=\"graph0\" class=\"graph\" transform=\"scale(1 1) rotate(0) translate(4 846)\">\r\n<title>Tree</title>\r\n<polygon fill=\"white\" stroke=\"none\" points=\"-4,4 -4,-846 1072,-846 1072,4 -4,4\"/>\r\n<!-- 0 -->\r\n<g id=\"node1\" class=\"node\"><title>0</title>\r\n<path fill=\"#e68540\" stroke=\"black\" d=\"M159,-441.5C159,-441.5 12,-441.5 12,-441.5 6,-441.5 0,-435.5 0,-429.5 0,-429.5 0,-370.5 0,-370.5 0,-364.5 6,-358.5 12,-358.5 12,-358.5 159,-358.5 159,-358.5 165,-358.5 171,-364.5 171,-370.5 171,-370.5 171,-429.5 171,-429.5 171,-435.5 165,-441.5 159,-441.5\"/>\r\n<text text-anchor=\"middle\" x=\"85.5\" y=\"-426.3\" font-family=\"Helvetica,sans-Serif\" font-size=\"14.00\">customer_count &lt;= 36.5</text>\r\n<text text-anchor=\"middle\" x=\"85.5\" y=\"-411.3\" font-family=\"Helvetica,sans-Serif\" font-size=\"14.00\">gini = 0.0666</text>\r\n<text text-anchor=\"middle\" x=\"85.5\" y=\"-396.3\" font-family=\"Helvetica,sans-Serif\" font-size=\"14.00\">samples = 100.0%</text>\r\n<text text-anchor=\"middle\" x=\"85.5\" y=\"-381.3\" font-family=\"Helvetica,sans-Serif\" font-size=\"14.00\">value = [0.9655, 0.0345]</text>\r\n<text text-anchor=\"middle\" x=\"85.5\" y=\"-366.3\" font-family=\"Helvetica,sans-Serif\" font-size=\"14.00\">class = Compra</text>\r\n</g>\r\n<!-- 1 -->\r\n<g id=\"node2\" class=\"node\"><title>1</title>\r\n<path fill=\"#e5823b\" stroke=\"black\" d=\"M400,-494.5C400,-494.5 219,-494.5 219,-494.5 213,-494.5 207,-488.5 207,-482.5 207,-482.5 207,-423.5 207,-423.5 207,-417.5 213,-411.5 219,-411.5 219,-411.5 400,-411.5 400,-411.5 406,-411.5 412,-417.5 412,-423.5 412,-423.5 412,-482.5 412,-482.5 412,-488.5 406,-494.5 400,-494.5\"/>\r\n<text text-anchor=\"middle\" x=\"309.5\" y=\"-479.3\" font-family=\"Helvetica,sans-Serif\" font-size=\"14.00\">Productos_em_acount &lt;= 0.5</text>\r\n<text text-anchor=\"middle\" x=\"309.5\" y=\"-464.3\" font-family=\"Helvetica,sans-Serif\" font-size=\"14.00\">gini = 0.0165</text>\r\n<text text-anchor=\"middle\" x=\"309.5\" y=\"-449.3\" font-family=\"Helvetica,sans-Serif\" font-size=\"14.00\">samples = 81.0%</text>\r\n<text text-anchor=\"middle\" x=\"309.5\" y=\"-434.3\" font-family=\"Helvetica,sans-Serif\" font-size=\"14.00\">value = [0.9917, 0.0083]</text>\r\n<text text-anchor=\"middle\" x=\"309.5\" y=\"-419.3\" font-family=\"Helvetica,sans-Serif\" font-size=\"14.00\">class = Compra</text>\r\n</g>\r\n<!-- 0&#45;&gt;1 -->\r\n<g id=\"edge1\" class=\"edge\"><title>0&#45;&gt;1</title>\r\n<path fill=\"none\" stroke=\"black\" d=\"M171.062,-420.188C179.52,-422.207 188.195,-424.278 196.875,-426.351\"/>\r\n<polygon fill=\"black\" stroke=\"black\" points=\"196.127,-429.77 206.666,-428.688 197.752,-422.962 196.127,-429.77\"/>\r\n<text text-anchor=\"middle\" x=\"185.367\" y=\"-438.078\" font-family=\"Helvetica,sans-Serif\" font-size=\"14.00\">True</text>\r\n</g>\r\n<!-- 10 -->\r\n<g id=\"node11\" class=\"node\"><title>10</title>\r\n<path fill=\"#e9975b\" stroke=\"black\" d=\"M382.5,-388.5C382.5,-388.5 236.5,-388.5 236.5,-388.5 230.5,-388.5 224.5,-382.5 224.5,-376.5 224.5,-376.5 224.5,-317.5 224.5,-317.5 224.5,-311.5 230.5,-305.5 236.5,-305.5 236.5,-305.5 382.5,-305.5 382.5,-305.5 388.5,-305.5 394.5,-311.5 394.5,-317.5 394.5,-317.5 394.5,-376.5 394.5,-376.5 394.5,-382.5 388.5,-388.5 382.5,-388.5\"/>\r\n<text text-anchor=\"middle\" x=\"309.5\" y=\"-373.3\" font-family=\"Helvetica,sans-Serif\" font-size=\"14.00\">recurrencia &lt;= 32.5</text>\r\n<text text-anchor=\"middle\" x=\"309.5\" y=\"-358.3\" font-family=\"Helvetica,sans-Serif\" font-size=\"14.00\">gini = 0.2496</text>\r\n<text text-anchor=\"middle\" x=\"309.5\" y=\"-343.3\" font-family=\"Helvetica,sans-Serif\" font-size=\"14.00\">samples = 19.0%</text>\r\n<text text-anchor=\"middle\" x=\"309.5\" y=\"-328.3\" font-family=\"Helvetica,sans-Serif\" font-size=\"14.00\">value = [0.8539, 0.1461]</text>\r\n<text text-anchor=\"middle\" x=\"309.5\" y=\"-313.3\" font-family=\"Helvetica,sans-Serif\" font-size=\"14.00\">class = Compra</text>\r\n</g>\r\n<!-- 0&#45;&gt;10 -->\r\n<g id=\"edge10\" class=\"edge\"><title>0&#45;&gt;10</title>\r\n<path fill=\"none\" stroke=\"black\" d=\"M171.062,-379.812C185.199,-376.437 199.944,-372.917 214.305,-369.488\"/>\r\n<polygon fill=\"black\" stroke=\"black\" points=\"215.259,-372.859 224.173,-367.132 213.634,-366.05 215.259,-372.859\"/>\r\n<text text-anchor=\"middle\" x=\"202.874\" y=\"-350.343\" font-family=\"Helvetica,sans-Serif\" font-size=\"14.00\">False</text>\r\n</g>\r\n<!-- 2 -->\r\n<g id=\"node3\" class=\"node\"><title>2</title>\r\n<path fill=\"#e68844\" stroke=\"black\" d=\"M609,-598.5C609,-598.5 463,-598.5 463,-598.5 457,-598.5 451,-592.5 451,-586.5 451,-586.5 451,-527.5 451,-527.5 451,-521.5 457,-515.5 463,-515.5 463,-515.5 609,-515.5 609,-515.5 615,-515.5 621,-521.5 621,-527.5 621,-527.5 621,-586.5 621,-586.5 621,-592.5 615,-598.5 609,-598.5\"/>\r\n<text text-anchor=\"middle\" x=\"536\" y=\"-583.3\" font-family=\"Helvetica,sans-Serif\" font-size=\"14.00\">days_between &lt;= &#45;14.0</text>\r\n<text text-anchor=\"middle\" x=\"536\" y=\"-568.3\" font-family=\"Helvetica,sans-Serif\" font-size=\"14.00\">gini = 0.0996</text>\r\n<text text-anchor=\"middle\" x=\"536\" y=\"-553.3\" font-family=\"Helvetica,sans-Serif\" font-size=\"14.00\">samples = 12.8%</text>\r\n<text text-anchor=\"middle\" x=\"536\" y=\"-538.3\" font-family=\"Helvetica,sans-Serif\" font-size=\"14.00\">value = [0.9474, 0.0526]</text>\r\n<text text-anchor=\"middle\" x=\"536\" y=\"-523.3\" font-family=\"Helvetica,sans-Serif\" font-size=\"14.00\">class = Compra</text>\r\n</g>\r\n<!-- 1&#45;&gt;2 -->\r\n<g id=\"edge2\" class=\"edge\"><title>1&#45;&gt;2</title>\r\n<path fill=\"none\" stroke=\"black\" d=\"M400.196,-494.552C413.791,-500.85 427.846,-507.361 441.519,-513.695\"/>\r\n<polygon fill=\"black\" stroke=\"black\" points=\"440.369,-517.019 450.914,-518.047 443.311,-510.668 440.369,-517.019\"/>\r\n</g>\r\n<!-- 9 -->\r\n<g id=\"node10\" class=\"node\"><title>9</title>\r\n<path fill=\"#e58139\" stroke=\"black\" d=\"M588,-487C588,-487 484,-487 484,-487 478,-487 472,-481 472,-475 472,-475 472,-431 472,-431 472,-425 478,-419 484,-419 484,-419 588,-419 588,-419 594,-419 600,-425 600,-431 600,-431 600,-475 600,-475 600,-481 594,-487 588,-487\"/>\r\n<text text-anchor=\"middle\" x=\"536\" y=\"-471.8\" font-family=\"Helvetica,sans-Serif\" font-size=\"14.00\">gini = 0.0</text>\r\n<text text-anchor=\"middle\" x=\"536\" y=\"-456.8\" font-family=\"Helvetica,sans-Serif\" font-size=\"14.00\">samples = 68.2%</text>\r\n<text text-anchor=\"middle\" x=\"536\" y=\"-441.8\" font-family=\"Helvetica,sans-Serif\" font-size=\"14.00\">value = [1.0, 0.0]</text>\r\n<text text-anchor=\"middle\" x=\"536\" y=\"-426.8\" font-family=\"Helvetica,sans-Serif\" font-size=\"14.00\">class = Compra</text>\r\n</g>\r\n<!-- 1&#45;&gt;9 -->\r\n<g id=\"edge9\" class=\"edge\"><title>1&#45;&gt;9</title>\r\n<path fill=\"none\" stroke=\"black\" d=\"M412.24,-453C428.895,-453 445.861,-453 461.654,-453\"/>\r\n<polygon fill=\"black\" stroke=\"black\" points=\"461.848,-456.5 471.848,-453 461.848,-449.5 461.848,-456.5\"/>\r\n</g>\r\n<!-- 3 -->\r\n<g id=\"node4\" class=\"node\"><title>3</title>\r\n<path fill=\"#efb184\" stroke=\"black\" d=\"M834,-763.5C834,-763.5 688,-763.5 688,-763.5 682,-763.5 676,-757.5 676,-751.5 676,-751.5 676,-692.5 676,-692.5 676,-686.5 682,-680.5 688,-680.5 688,-680.5 834,-680.5 834,-680.5 840,-680.5 846,-686.5 846,-692.5 846,-692.5 846,-751.5 846,-751.5 846,-757.5 840,-763.5 834,-763.5\"/>\r\n<text text-anchor=\"middle\" x=\"761\" y=\"-748.3\" font-family=\"Helvetica,sans-Serif\" font-size=\"14.00\">recurrencia &lt;= 8.5</text>\r\n<text text-anchor=\"middle\" x=\"761\" y=\"-733.3\" font-family=\"Helvetica,sans-Serif\" font-size=\"14.00\">gini = 0.3994</text>\r\n<text text-anchor=\"middle\" x=\"761\" y=\"-718.3\" font-family=\"Helvetica,sans-Serif\" font-size=\"14.00\">samples = 0.6%</text>\r\n<text text-anchor=\"middle\" x=\"761\" y=\"-703.3\" font-family=\"Helvetica,sans-Serif\" font-size=\"14.00\">value = [0.7243, 0.2757]</text>\r\n<text text-anchor=\"middle\" x=\"761\" y=\"-688.3\" font-family=\"Helvetica,sans-Serif\" font-size=\"14.00\">class = Compra</text>\r\n</g>\r\n<!-- 2&#45;&gt;3 -->\r\n<g id=\"edge3\" class=\"edge\"><title>2&#45;&gt;3</title>\r\n<path fill=\"none\" stroke=\"black\" d=\"M593.272,-598.636C624.436,-621.695 663.301,-650.452 695.666,-674.399\"/>\r\n<polygon fill=\"black\" stroke=\"black\" points=\"693.625,-677.242 703.746,-680.377 697.789,-671.615 693.625,-677.242\"/>\r\n</g>\r\n<!-- 6 -->\r\n<g id=\"node7\" class=\"node\"><title>6</title>\r\n<path fill=\"#e68742\" stroke=\"black\" d=\"M850,-598.5C850,-598.5 672,-598.5 672,-598.5 666,-598.5 660,-592.5 660,-586.5 660,-586.5 660,-527.5 660,-527.5 660,-521.5 666,-515.5 672,-515.5 672,-515.5 850,-515.5 850,-515.5 856,-515.5 862,-521.5 862,-527.5 862,-527.5 862,-586.5 862,-586.5 862,-592.5 856,-598.5 850,-598.5\"/>\r\n<text text-anchor=\"middle\" x=\"761\" y=\"-583.3\" font-family=\"Helvetica,sans-Serif\" font-size=\"14.00\">Productos_debit_card &lt;= 0.5</text>\r\n<text text-anchor=\"middle\" x=\"761\" y=\"-568.3\" font-family=\"Helvetica,sans-Serif\" font-size=\"14.00\">gini = 0.0803</text>\r\n<text text-anchor=\"middle\" x=\"761\" y=\"-553.3\" font-family=\"Helvetica,sans-Serif\" font-size=\"14.00\">samples = 12.3%</text>\r\n<text text-anchor=\"middle\" x=\"761\" y=\"-538.3\" font-family=\"Helvetica,sans-Serif\" font-size=\"14.00\">value = [0.9581, 0.0419]</text>\r\n<text text-anchor=\"middle\" x=\"761\" y=\"-523.3\" font-family=\"Helvetica,sans-Serif\" font-size=\"14.00\">class = Compra</text>\r\n</g>\r\n<!-- 2&#45;&gt;6 -->\r\n<g id=\"edge6\" class=\"edge\"><title>2&#45;&gt;6</title>\r\n<path fill=\"none\" stroke=\"black\" d=\"M621.307,-557C630.546,-557 640.05,-557 649.544,-557\"/>\r\n<polygon fill=\"black\" stroke=\"black\" points=\"649.829,-560.5 659.829,-557 649.829,-553.5 649.829,-560.5\"/>\r\n</g>\r\n<!-- 4 -->\r\n<g id=\"node5\" class=\"node\"><title>4</title>\r\n<path fill=\"#fcf2ea\" stroke=\"black\" d=\"M1056,-842C1056,-842 910,-842 910,-842 904,-842 898,-836 898,-830 898,-830 898,-786 898,-786 898,-780 904,-774 910,-774 910,-774 1056,-774 1056,-774 1062,-774 1068,-780 1068,-786 1068,-786 1068,-830 1068,-830 1068,-836 1062,-842 1056,-842\"/>\r\n<text text-anchor=\"middle\" x=\"983\" y=\"-826.8\" font-family=\"Helvetica,sans-Serif\" font-size=\"14.00\">gini = 0.4984</text>\r\n<text text-anchor=\"middle\" x=\"983\" y=\"-811.8\" font-family=\"Helvetica,sans-Serif\" font-size=\"14.00\">samples = 0.3%</text>\r\n<text text-anchor=\"middle\" x=\"983\" y=\"-796.8\" font-family=\"Helvetica,sans-Serif\" font-size=\"14.00\">value = [0.5282, 0.4718]</text>\r\n<text text-anchor=\"middle\" x=\"983\" y=\"-781.8\" font-family=\"Helvetica,sans-Serif\" font-size=\"14.00\">class = Compra</text>\r\n</g>\r\n<!-- 3&#45;&gt;4 -->\r\n<g id=\"edge4\" class=\"edge\"><title>3&#45;&gt;4</title>\r\n<path fill=\"none\" stroke=\"black\" d=\"M846.114,-754.881C859.912,-760.275 874.287,-765.894 888.3,-771.372\"/>\r\n<polygon fill=\"black\" stroke=\"black\" points=\"887.344,-774.756 897.932,-775.137 889.893,-768.237 887.344,-774.756\"/>\r\n</g>\r\n<!-- 5 -->\r\n<g id=\"node6\" class=\"node\"><title>5</title>\r\n<path fill=\"#e78c4a\" stroke=\"black\" d=\"M1056,-756C1056,-756 910,-756 910,-756 904,-756 898,-750 898,-744 898,-744 898,-700 898,-700 898,-694 904,-688 910,-688 910,-688 1056,-688 1056,-688 1062,-688 1068,-694 1068,-700 1068,-700 1068,-744 1068,-744 1068,-750 1062,-756 1056,-756\"/>\r\n<text text-anchor=\"middle\" x=\"983\" y=\"-740.8\" font-family=\"Helvetica,sans-Serif\" font-size=\"14.00\">gini = 0.143</text>\r\n<text text-anchor=\"middle\" x=\"983\" y=\"-725.8\" font-family=\"Helvetica,sans-Serif\" font-size=\"14.00\">samples = 0.3%</text>\r\n<text text-anchor=\"middle\" x=\"983\" y=\"-710.8\" font-family=\"Helvetica,sans-Serif\" font-size=\"14.00\">value = [0.9225, 0.0775]</text>\r\n<text text-anchor=\"middle\" x=\"983\" y=\"-695.8\" font-family=\"Helvetica,sans-Serif\" font-size=\"14.00\">class = Compra</text>\r\n</g>\r\n<!-- 3&#45;&gt;5 -->\r\n<g id=\"edge5\" class=\"edge\"><title>3&#45;&gt;5</title>\r\n<path fill=\"none\" stroke=\"black\" d=\"M846.114,-722C859.78,-722 874.01,-722 887.896,-722\"/>\r\n<polygon fill=\"black\" stroke=\"black\" points=\"887.932,-725.5 897.932,-722 887.932,-718.5 887.932,-725.5\"/>\r\n</g>\r\n<!-- 7 -->\r\n<g id=\"node8\" class=\"node\"><title>7</title>\r\n<path fill=\"#e78b48\" stroke=\"black\" d=\"M1056,-670C1056,-670 910,-670 910,-670 904,-670 898,-664 898,-658 898,-658 898,-614 898,-614 898,-608 904,-602 910,-602 910,-602 1056,-602 1056,-602 1062,-602 1068,-608 1068,-614 1068,-614 1068,-658 1068,-658 1068,-664 1062,-670 1056,-670\"/>\r\n<text text-anchor=\"middle\" x=\"983\" y=\"-654.8\" font-family=\"Helvetica,sans-Serif\" font-size=\"14.00\">gini = 0.1317</text>\r\n<text text-anchor=\"middle\" x=\"983\" y=\"-639.8\" font-family=\"Helvetica,sans-Serif\" font-size=\"14.00\">samples = 7.3%</text>\r\n<text text-anchor=\"middle\" x=\"983\" y=\"-624.8\" font-family=\"Helvetica,sans-Serif\" font-size=\"14.00\">value = [0.9291, 0.0709]</text>\r\n<text text-anchor=\"middle\" x=\"983\" y=\"-609.8\" font-family=\"Helvetica,sans-Serif\" font-size=\"14.00\">class = Compra</text>\r\n</g>\r\n<!-- 6&#45;&gt;7 -->\r\n<g id=\"edge7\" class=\"edge\"><title>6&#45;&gt;7</title>\r\n<path fill=\"none\" stroke=\"black\" d=\"M862.022,-592.917C870.681,-596.027 879.424,-599.166 888.033,-602.257\"/>\r\n<polygon fill=\"black\" stroke=\"black\" points=\"887.089,-605.637 897.684,-605.723 889.455,-599.049 887.089,-605.637\"/>\r\n</g>\r\n<!-- 8 -->\r\n<g id=\"node9\" class=\"node\"><title>8</title>\r\n<path fill=\"#e58139\" stroke=\"black\" d=\"M1031.5,-584C1031.5,-584 934.5,-584 934.5,-584 928.5,-584 922.5,-578 922.5,-572 922.5,-572 922.5,-528 922.5,-528 922.5,-522 928.5,-516 934.5,-516 934.5,-516 1031.5,-516 1031.5,-516 1037.5,-516 1043.5,-522 1043.5,-528 1043.5,-528 1043.5,-572 1043.5,-572 1043.5,-578 1037.5,-584 1031.5,-584\"/>\r\n<text text-anchor=\"middle\" x=\"983\" y=\"-568.8\" font-family=\"Helvetica,sans-Serif\" font-size=\"14.00\">gini = 0.0</text>\r\n<text text-anchor=\"middle\" x=\"983\" y=\"-553.8\" font-family=\"Helvetica,sans-Serif\" font-size=\"14.00\">samples = 5.0%</text>\r\n<text text-anchor=\"middle\" x=\"983\" y=\"-538.8\" font-family=\"Helvetica,sans-Serif\" font-size=\"14.00\">value = [1.0, 0.0]</text>\r\n<text text-anchor=\"middle\" x=\"983\" y=\"-523.8\" font-family=\"Helvetica,sans-Serif\" font-size=\"14.00\">class = Compra</text>\r\n</g>\r\n<!-- 6&#45;&gt;8 -->\r\n<g id=\"edge8\" class=\"edge\"><title>6&#45;&gt;8</title>\r\n<path fill=\"none\" stroke=\"black\" d=\"M862.022,-553.817C878.968,-553.278 896.233,-552.729 912.19,-552.221\"/>\r\n<polygon fill=\"black\" stroke=\"black\" points=\"912.587,-555.71 922.47,-551.894 912.364,-548.714 912.587,-555.71\"/>\r\n</g>\r\n<!-- 11 -->\r\n<g id=\"node12\" class=\"node\"><title>11</title>\r\n<path fill=\"#eca26d\" stroke=\"black\" d=\"M612,-388.5C612,-388.5 460,-388.5 460,-388.5 454,-388.5 448,-382.5 448,-376.5 448,-376.5 448,-317.5 448,-317.5 448,-311.5 454,-305.5 460,-305.5 460,-305.5 612,-305.5 612,-305.5 618,-305.5 624,-311.5 624,-317.5 624,-317.5 624,-376.5 624,-376.5 624,-382.5 618,-388.5 612,-388.5\"/>\r\n<text text-anchor=\"middle\" x=\"536\" y=\"-373.3\" font-family=\"Helvetica,sans-Serif\" font-size=\"14.00\">Productos_payroll &lt;= 0.5</text>\r\n<text text-anchor=\"middle\" x=\"536\" y=\"-358.3\" font-family=\"Helvetica,sans-Serif\" font-size=\"14.00\">gini = 0.3307</text>\r\n<text text-anchor=\"middle\" x=\"536\" y=\"-343.3\" font-family=\"Helvetica,sans-Serif\" font-size=\"14.00\">samples = 11.0%</text>\r\n<text text-anchor=\"middle\" x=\"536\" y=\"-328.3\" font-family=\"Helvetica,sans-Serif\" font-size=\"14.00\">value = [0.791, 0.209]</text>\r\n<text text-anchor=\"middle\" x=\"536\" y=\"-313.3\" font-family=\"Helvetica,sans-Serif\" font-size=\"14.00\">class = Compra</text>\r\n</g>\r\n<!-- 10&#45;&gt;11 -->\r\n<g id=\"edge11\" class=\"edge\"><title>10&#45;&gt;11</title>\r\n<path fill=\"none\" stroke=\"black\" d=\"M394.734,-347C408.651,-347 423.181,-347 437.393,-347\"/>\r\n<polygon fill=\"black\" stroke=\"black\" points=\"437.671,-350.5 447.671,-347 437.671,-343.5 437.671,-350.5\"/>\r\n</g>\r\n<!-- 16 -->\r\n<g id=\"node17\" class=\"node\"><title>16</title>\r\n<path fill=\"#e78945\" stroke=\"black\" d=\"M609.5,-267.5C609.5,-267.5 462.5,-267.5 462.5,-267.5 456.5,-267.5 450.5,-261.5 450.5,-255.5 450.5,-255.5 450.5,-196.5 450.5,-196.5 450.5,-190.5 456.5,-184.5 462.5,-184.5 462.5,-184.5 609.5,-184.5 609.5,-184.5 615.5,-184.5 621.5,-190.5 621.5,-196.5 621.5,-196.5 621.5,-255.5 621.5,-255.5 621.5,-261.5 615.5,-267.5 609.5,-267.5\"/>\r\n<text text-anchor=\"middle\" x=\"536\" y=\"-252.3\" font-family=\"Helvetica,sans-Serif\" font-size=\"14.00\">customer_count &lt;= 70.5</text>\r\n<text text-anchor=\"middle\" x=\"536\" y=\"-237.3\" font-family=\"Helvetica,sans-Serif\" font-size=\"14.00\">gini = 0.1105</text>\r\n<text text-anchor=\"middle\" x=\"536\" y=\"-222.3\" font-family=\"Helvetica,sans-Serif\" font-size=\"14.00\">samples = 7.9%</text>\r\n<text text-anchor=\"middle\" x=\"536\" y=\"-207.3\" font-family=\"Helvetica,sans-Serif\" font-size=\"14.00\">value = [0.9413, 0.0587]</text>\r\n<text text-anchor=\"middle\" x=\"536\" y=\"-192.3\" font-family=\"Helvetica,sans-Serif\" font-size=\"14.00\">class = Compra</text>\r\n</g>\r\n<!-- 10&#45;&gt;16 -->\r\n<g id=\"edge16\" class=\"edge\"><title>10&#45;&gt;16</title>\r\n<path fill=\"none\" stroke=\"black\" d=\"M387.76,-305.359C407.537,-294.699 428.924,-283.172 449.106,-272.295\"/>\r\n<polygon fill=\"black\" stroke=\"black\" points=\"450.836,-275.339 457.978,-267.513 447.514,-269.177 450.836,-275.339\"/>\r\n</g>\r\n<!-- 12 -->\r\n<g id=\"node13\" class=\"node\"><title>12</title>\r\n<path fill=\"#efb083\" stroke=\"black\" d=\"M834.5,-469.5C834.5,-469.5 687.5,-469.5 687.5,-469.5 681.5,-469.5 675.5,-463.5 675.5,-457.5 675.5,-457.5 675.5,-398.5 675.5,-398.5 675.5,-392.5 681.5,-386.5 687.5,-386.5 687.5,-386.5 834.5,-386.5 834.5,-386.5 840.5,-386.5 846.5,-392.5 846.5,-398.5 846.5,-398.5 846.5,-457.5 846.5,-457.5 846.5,-463.5 840.5,-469.5 834.5,-469.5\"/>\r\n<text text-anchor=\"middle\" x=\"761\" y=\"-454.3\" font-family=\"Helvetica,sans-Serif\" font-size=\"14.00\">customer_count &lt;= 51.5</text>\r\n<text text-anchor=\"middle\" x=\"761\" y=\"-439.3\" font-family=\"Helvetica,sans-Serif\" font-size=\"14.00\">gini = 0.3964</text>\r\n<text text-anchor=\"middle\" x=\"761\" y=\"-424.3\" font-family=\"Helvetica,sans-Serif\" font-size=\"14.00\">samples = 8.5%</text>\r\n<text text-anchor=\"middle\" x=\"761\" y=\"-409.3\" font-family=\"Helvetica,sans-Serif\" font-size=\"14.00\">value = [0.7276, 0.2724]</text>\r\n<text text-anchor=\"middle\" x=\"761\" y=\"-394.3\" font-family=\"Helvetica,sans-Serif\" font-size=\"14.00\">class = Compra</text>\r\n</g>\r\n<!-- 11&#45;&gt;12 -->\r\n<g id=\"edge12\" class=\"edge\"><title>11&#45;&gt;12</title>\r\n<path fill=\"none\" stroke=\"black\" d=\"M624.176,-378.665C637.873,-383.64 652.085,-388.802 665.929,-393.831\"/>\r\n<polygon fill=\"black\" stroke=\"black\" points=\"664.85,-397.162 675.444,-397.287 667.239,-390.583 664.85,-397.162\"/>\r\n</g>\r\n<!-- 15 -->\r\n<g id=\"node16\" class=\"node\"><title>15</title>\r\n<path fill=\"#e58139\" stroke=\"black\" d=\"M809.5,-368C809.5,-368 712.5,-368 712.5,-368 706.5,-368 700.5,-362 700.5,-356 700.5,-356 700.5,-312 700.5,-312 700.5,-306 706.5,-300 712.5,-300 712.5,-300 809.5,-300 809.5,-300 815.5,-300 821.5,-306 821.5,-312 821.5,-312 821.5,-356 821.5,-356 821.5,-362 815.5,-368 809.5,-368\"/>\r\n<text text-anchor=\"middle\" x=\"761\" y=\"-352.8\" font-family=\"Helvetica,sans-Serif\" font-size=\"14.00\">gini = 0.0</text>\r\n<text text-anchor=\"middle\" x=\"761\" y=\"-337.8\" font-family=\"Helvetica,sans-Serif\" font-size=\"14.00\">samples = 2.6%</text>\r\n<text text-anchor=\"middle\" x=\"761\" y=\"-322.8\" font-family=\"Helvetica,sans-Serif\" font-size=\"14.00\">value = [1.0, 0.0]</text>\r\n<text text-anchor=\"middle\" x=\"761\" y=\"-307.8\" font-family=\"Helvetica,sans-Serif\" font-size=\"14.00\">class = Compra</text>\r\n</g>\r\n<!-- 11&#45;&gt;15 -->\r\n<g id=\"edge15\" class=\"edge\"><title>11&#45;&gt;15</title>\r\n<path fill=\"none\" stroke=\"black\" d=\"M624.176,-341.918C646.116,-340.639 669.378,-339.283 690.334,-338.061\"/>\r\n<polygon fill=\"black\" stroke=\"black\" points=\"690.641,-341.549 700.42,-337.473 690.234,-334.561 690.641,-341.549\"/>\r\n</g>\r\n<!-- 13 -->\r\n<g id=\"node14\" class=\"node\"><title>13</title>\r\n<path fill=\"#ea995f\" stroke=\"black\" d=\"M1048,-498C1048,-498 918,-498 918,-498 912,-498 906,-492 906,-486 906,-486 906,-442 906,-442 906,-436 912,-430 918,-430 918,-430 1048,-430 1048,-430 1054,-430 1060,-436 1060,-442 1060,-442 1060,-486 1060,-486 1060,-492 1054,-498 1048,-498\"/>\r\n<text text-anchor=\"middle\" x=\"983\" y=\"-482.8\" font-family=\"Helvetica,sans-Serif\" font-size=\"14.00\">gini = 0.2715</text>\r\n<text text-anchor=\"middle\" x=\"983\" y=\"-467.8\" font-family=\"Helvetica,sans-Serif\" font-size=\"14.00\">samples = 4.9%</text>\r\n<text text-anchor=\"middle\" x=\"983\" y=\"-452.8\" font-family=\"Helvetica,sans-Serif\" font-size=\"14.00\">value = [0.838, 0.162]</text>\r\n<text text-anchor=\"middle\" x=\"983\" y=\"-437.8\" font-family=\"Helvetica,sans-Serif\" font-size=\"14.00\">class = Compra</text>\r\n</g>\r\n<!-- 12&#45;&gt;13 -->\r\n<g id=\"edge13\" class=\"edge\"><title>12&#45;&gt;13</title>\r\n<path fill=\"none\" stroke=\"black\" d=\"M846.743,-441.867C862.882,-444.508 879.787,-447.274 895.966,-449.922\"/>\r\n<polygon fill=\"black\" stroke=\"black\" points=\"895.498,-453.392 905.932,-451.552 896.628,-446.483 895.498,-453.392\"/>\r\n</g>\r\n<!-- 14 -->\r\n<g id=\"node15\" class=\"node\"><title>14</title>\r\n<path fill=\"#f8decb\" stroke=\"black\" d=\"M1048,-412C1048,-412 918,-412 918,-412 912,-412 906,-406 906,-400 906,-400 906,-356 906,-356 906,-350 912,-344 918,-344 918,-344 1048,-344 1048,-344 1054,-344 1060,-350 1060,-356 1060,-356 1060,-400 1060,-400 1060,-406 1054,-412 1048,-412\"/>\r\n<text text-anchor=\"middle\" x=\"983\" y=\"-396.8\" font-family=\"Helvetica,sans-Serif\" font-size=\"14.00\">gini = 0.4888</text>\r\n<text text-anchor=\"middle\" x=\"983\" y=\"-381.8\" font-family=\"Helvetica,sans-Serif\" font-size=\"14.00\">samples = 3.6%</text>\r\n<text text-anchor=\"middle\" x=\"983\" y=\"-366.8\" font-family=\"Helvetica,sans-Serif\" font-size=\"14.00\">value = [0.575, 0.425]</text>\r\n<text text-anchor=\"middle\" x=\"983\" y=\"-351.8\" font-family=\"Helvetica,sans-Serif\" font-size=\"14.00\">class = Compra</text>\r\n</g>\r\n<!-- 12&#45;&gt;14 -->\r\n<g id=\"edge14\" class=\"edge\"><title>12&#45;&gt;14</title>\r\n<path fill=\"none\" stroke=\"black\" d=\"M846.743,-408.74C862.882,-405.072 879.787,-401.23 895.966,-397.553\"/>\r\n<polygon fill=\"black\" stroke=\"black\" points=\"896.956,-400.918 905.932,-395.288 895.405,-394.092 896.956,-400.918\"/>\r\n</g>\r\n<!-- 17 -->\r\n<g id=\"node18\" class=\"node\"><title>17</title>\r\n<path fill=\"#e5823b\" stroke=\"black\" d=\"M834.5,-267.5C834.5,-267.5 687.5,-267.5 687.5,-267.5 681.5,-267.5 675.5,-261.5 675.5,-255.5 675.5,-255.5 675.5,-196.5 675.5,-196.5 675.5,-190.5 681.5,-184.5 687.5,-184.5 687.5,-184.5 834.5,-184.5 834.5,-184.5 840.5,-184.5 846.5,-190.5 846.5,-196.5 846.5,-196.5 846.5,-255.5 846.5,-255.5 846.5,-261.5 840.5,-267.5 834.5,-267.5\"/>\r\n<text text-anchor=\"middle\" x=\"761\" y=\"-252.3\" font-family=\"Helvetica,sans-Serif\" font-size=\"14.00\">customer_count &lt;= 68.5</text>\r\n<text text-anchor=\"middle\" x=\"761\" y=\"-237.3\" font-family=\"Helvetica,sans-Serif\" font-size=\"14.00\">gini = 0.0206</text>\r\n<text text-anchor=\"middle\" x=\"761\" y=\"-222.3\" font-family=\"Helvetica,sans-Serif\" font-size=\"14.00\">samples = 4.9%</text>\r\n<text text-anchor=\"middle\" x=\"761\" y=\"-207.3\" font-family=\"Helvetica,sans-Serif\" font-size=\"14.00\">value = [0.9896, 0.0104]</text>\r\n<text text-anchor=\"middle\" x=\"761\" y=\"-192.3\" font-family=\"Helvetica,sans-Serif\" font-size=\"14.00\">class = Compra</text>\r\n</g>\r\n<!-- 16&#45;&gt;17 -->\r\n<g id=\"edge17\" class=\"edge\"><title>16&#45;&gt;17</title>\r\n<path fill=\"none\" stroke=\"black\" d=\"M621.625,-226C635.803,-226 650.598,-226 665.015,-226\"/>\r\n<polygon fill=\"black\" stroke=\"black\" points=\"665.433,-229.5 675.433,-226 665.433,-222.5 665.433,-229.5\"/>\r\n</g>\r\n<!-- 20 -->\r\n<g id=\"node21\" class=\"node\"><title>20</title>\r\n<path fill=\"#e99558\" stroke=\"black\" d=\"M834,-161.5C834,-161.5 688,-161.5 688,-161.5 682,-161.5 676,-155.5 676,-149.5 676,-149.5 676,-90.5 676,-90.5 676,-84.5 682,-78.5 688,-78.5 688,-78.5 834,-78.5 834,-78.5 840,-78.5 846,-84.5 846,-90.5 846,-90.5 846,-149.5 846,-149.5 846,-155.5 840,-161.5 834,-161.5\"/>\r\n<text text-anchor=\"middle\" x=\"761\" y=\"-146.3\" font-family=\"Helvetica,sans-Serif\" font-size=\"14.00\">recurrencia &lt;= 47.5</text>\r\n<text text-anchor=\"middle\" x=\"761\" y=\"-131.3\" font-family=\"Helvetica,sans-Serif\" font-size=\"14.00\">gini = 0.2359</text>\r\n<text text-anchor=\"middle\" x=\"761\" y=\"-116.3\" font-family=\"Helvetica,sans-Serif\" font-size=\"14.00\">samples = 3.0%</text>\r\n<text text-anchor=\"middle\" x=\"761\" y=\"-101.3\" font-family=\"Helvetica,sans-Serif\" font-size=\"14.00\">value = [0.8634, 0.1366]</text>\r\n<text text-anchor=\"middle\" x=\"761\" y=\"-86.3\" font-family=\"Helvetica,sans-Serif\" font-size=\"14.00\">class = Compra</text>\r\n</g>\r\n<!-- 16&#45;&gt;20 -->\r\n<g id=\"edge20\" class=\"edge\"><title>16&#45;&gt;20</title>\r\n<path fill=\"none\" stroke=\"black\" d=\"M621.625,-185.775C636.318,-178.791 651.673,-171.492 666.585,-164.404\"/>\r\n<polygon fill=\"black\" stroke=\"black\" points=\"668.273,-167.476 675.803,-160.022 665.268,-161.154 668.273,-167.476\"/>\r\n</g>\r\n<!-- 18 -->\r\n<g id=\"node19\" class=\"node\"><title>18</title>\r\n<path fill=\"#e5823b\" stroke=\"black\" d=\"M1056,-326C1056,-326 910,-326 910,-326 904,-326 898,-320 898,-314 898,-314 898,-270 898,-270 898,-264 904,-258 910,-258 910,-258 1056,-258 1056,-258 1062,-258 1068,-264 1068,-270 1068,-270 1068,-314 1068,-314 1068,-320 1062,-326 1056,-326\"/>\r\n<text text-anchor=\"middle\" x=\"983\" y=\"-310.8\" font-family=\"Helvetica,sans-Serif\" font-size=\"14.00\">gini = 0.0161</text>\r\n<text text-anchor=\"middle\" x=\"983\" y=\"-295.8\" font-family=\"Helvetica,sans-Serif\" font-size=\"14.00\">samples = 4.7%</text>\r\n<text text-anchor=\"middle\" x=\"983\" y=\"-280.8\" font-family=\"Helvetica,sans-Serif\" font-size=\"14.00\">value = [0.9919, 0.0081]</text>\r\n<text text-anchor=\"middle\" x=\"983\" y=\"-265.8\" font-family=\"Helvetica,sans-Serif\" font-size=\"14.00\">class = Compra</text>\r\n</g>\r\n<!-- 17&#45;&gt;18 -->\r\n<g id=\"edge18\" class=\"edge\"><title>17&#45;&gt;18</title>\r\n<path fill=\"none\" stroke=\"black\" d=\"M846.743,-251.423C860.185,-255.455 874.158,-259.647 887.801,-263.74\"/>\r\n<polygon fill=\"black\" stroke=\"black\" points=\"887.08,-267.178 897.664,-266.699 889.091,-260.473 887.08,-267.178\"/>\r\n</g>\r\n<!-- 19 -->\r\n<g id=\"node20\" class=\"node\"><title>19</title>\r\n<path fill=\"#e78946\" stroke=\"black\" d=\"M1056,-240C1056,-240 910,-240 910,-240 904,-240 898,-234 898,-228 898,-228 898,-184 898,-184 898,-178 904,-172 910,-172 910,-172 1056,-172 1056,-172 1062,-172 1068,-178 1068,-184 1068,-184 1068,-228 1068,-228 1068,-234 1062,-240 1056,-240\"/>\r\n<text text-anchor=\"middle\" x=\"983\" y=\"-224.8\" font-family=\"Helvetica,sans-Serif\" font-size=\"14.00\">gini = 0.1149</text>\r\n<text text-anchor=\"middle\" x=\"983\" y=\"-209.8\" font-family=\"Helvetica,sans-Serif\" font-size=\"14.00\">samples = 0.2%</text>\r\n<text text-anchor=\"middle\" x=\"983\" y=\"-194.8\" font-family=\"Helvetica,sans-Serif\" font-size=\"14.00\">value = [0.9388, 0.0612]</text>\r\n<text text-anchor=\"middle\" x=\"983\" y=\"-179.8\" font-family=\"Helvetica,sans-Serif\" font-size=\"14.00\">class = Compra</text>\r\n</g>\r\n<!-- 17&#45;&gt;19 -->\r\n<g id=\"edge19\" class=\"edge\"><title>17&#45;&gt;19</title>\r\n<path fill=\"none\" stroke=\"black\" d=\"M846.743,-218.296C860.054,-217.086 873.887,-215.828 887.403,-214.6\"/>\r\n<polygon fill=\"black\" stroke=\"black\" points=\"888.022,-218.058 897.664,-213.667 887.388,-211.087 888.022,-218.058\"/>\r\n</g>\r\n<!-- 21 -->\r\n<g id=\"node22\" class=\"node\"><title>21</title>\r\n<path fill=\"#f6d3ba\" stroke=\"black\" d=\"M1056,-154C1056,-154 910,-154 910,-154 904,-154 898,-148 898,-142 898,-142 898,-98 898,-98 898,-92 904,-86 910,-86 910,-86 1056,-86 1056,-86 1062,-86 1068,-92 1068,-98 1068,-98 1068,-142 1068,-142 1068,-148 1062,-154 1056,-154\"/>\r\n<text text-anchor=\"middle\" x=\"983\" y=\"-138.8\" font-family=\"Helvetica,sans-Serif\" font-size=\"14.00\">gini = 0.4781</text>\r\n<text text-anchor=\"middle\" x=\"983\" y=\"-123.8\" font-family=\"Helvetica,sans-Serif\" font-size=\"14.00\">samples = 0.9%</text>\r\n<text text-anchor=\"middle\" x=\"983\" y=\"-108.8\" font-family=\"Helvetica,sans-Serif\" font-size=\"14.00\">value = [0.6046, 0.3954]</text>\r\n<text text-anchor=\"middle\" x=\"983\" y=\"-93.8\" font-family=\"Helvetica,sans-Serif\" font-size=\"14.00\">class = Compra</text>\r\n</g>\r\n<!-- 20&#45;&gt;21 -->\r\n<g id=\"edge21\" class=\"edge\"><title>20&#45;&gt;21</title>\r\n<path fill=\"none\" stroke=\"black\" d=\"M846.114,-120C859.78,-120 874.01,-120 887.896,-120\"/>\r\n<polygon fill=\"black\" stroke=\"black\" points=\"887.932,-123.5 897.932,-120 887.932,-116.5 887.932,-123.5\"/>\r\n</g>\r\n<!-- 22 -->\r\n<g id=\"node23\" class=\"node\"><title>22</title>\r\n<path fill=\"#e6853f\" stroke=\"black\" d=\"M1056,-68C1056,-68 910,-68 910,-68 904,-68 898,-62 898,-56 898,-56 898,-12 898,-12 898,-6 904,-0 910,-0 910,-0 1056,-0 1056,-0 1062,-0 1068,-6 1068,-12 1068,-12 1068,-56 1068,-56 1068,-62 1062,-68 1056,-68\"/>\r\n<text text-anchor=\"middle\" x=\"983\" y=\"-52.8\" font-family=\"Helvetica,sans-Serif\" font-size=\"14.00\">gini = 0.0611</text>\r\n<text text-anchor=\"middle\" x=\"983\" y=\"-37.8\" font-family=\"Helvetica,sans-Serif\" font-size=\"14.00\">samples = 2.2%</text>\r\n<text text-anchor=\"middle\" x=\"983\" y=\"-22.8\" font-family=\"Helvetica,sans-Serif\" font-size=\"14.00\">value = [0.9685, 0.0315]</text>\r\n<text text-anchor=\"middle\" x=\"983\" y=\"-7.8\" font-family=\"Helvetica,sans-Serif\" font-size=\"14.00\">class = Compra</text>\r\n</g>\r\n<!-- 20&#45;&gt;22 -->\r\n<g id=\"edge22\" class=\"edge\"><title>20&#45;&gt;22</title>\r\n<path fill=\"none\" stroke=\"black\" d=\"M846.114,-87.119C859.912,-81.7251 874.287,-76.1061 888.3,-70.6282\"/>\r\n<polygon fill=\"black\" stroke=\"black\" points=\"889.893,-73.7634 897.932,-66.8628 887.344,-67.2439 889.893,-73.7634\"/>\r\n</g>\r\n</g>\r\n</svg>\r\n"
     },
     "metadata": {},
     "execution_count": 42
    }
   ],
   "metadata": {
    "azdata_cell_guid": "865e0fa5-00b5-4810-8b7a-48875f231658"
   }
  },
  {
   "cell_type": "markdown",
   "source": [
    "# EVALUACIÓN RESULTADOS"
   ],
   "metadata": {}
  },
  {
   "cell_type": "code",
   "execution_count": 43,
   "source": [
    "y_test_pred = pd.DataFrame(dt.predict(X_test), index=y_test.index, columns=['Predice_compra'])"
   ],
   "outputs": [],
   "metadata": {}
  },
  {
   "cell_type": "code",
   "execution_count": 44,
   "source": [
    "y_test.head()"
   ],
   "outputs": [
    {
     "output_type": "execute_result",
     "data": {
      "text/plain": [
       "         Productos_pension_plan\n",
       "4227490                       0\n",
       "1868500                       0\n",
       "1584741                       1\n",
       "944015                        0\n",
       "337619                        0"
      ],
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
       "      <th>4227490</th>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1868500</th>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1584741</th>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>944015</th>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>337619</th>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ]
     },
     "metadata": {},
     "execution_count": 44
    }
   ],
   "metadata": {}
  },
  {
   "cell_type": "code",
   "execution_count": 45,
   "source": [
    "y_test_pred.head()"
   ],
   "outputs": [
    {
     "output_type": "execute_result",
     "data": {
      "text/plain": [
       "         Predice_compra\n",
       "4227490               0\n",
       "1868500               0\n",
       "1584741               0\n",
       "944015                0\n",
       "337619                0"
      ],
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
       "      <th>4227490</th>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1868500</th>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1584741</th>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>944015</th>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>337619</th>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ]
     },
     "metadata": {},
     "execution_count": 45
    }
   ],
   "metadata": {}
  },
  {
   "cell_type": "code",
   "execution_count": 46,
   "source": [
    "y_test.shape"
   ],
   "outputs": [
    {
     "output_type": "execute_result",
     "data": {
      "text/plain": [
       "(1068529, 1)"
      ]
     },
     "metadata": {},
     "execution_count": 46
    }
   ],
   "metadata": {}
  },
  {
   "cell_type": "code",
   "execution_count": 47,
   "source": [
    "y_test_pred.shape"
   ],
   "outputs": [
    {
     "output_type": "execute_result",
     "data": {
      "text/plain": [
       "(1068529, 1)"
      ]
     },
     "metadata": {},
     "execution_count": 47
    }
   ],
   "metadata": {}
  },
  {
   "cell_type": "code",
   "execution_count": 48,
   "source": [
    "results_df = y_test.join(y_test_pred,how= 'inner')   \r\n"
   ],
   "outputs": [],
   "metadata": {}
  },
  {
   "cell_type": "code",
   "execution_count": 49,
   "source": [
    "results_df['Predice_compra']"
   ],
   "outputs": [
    {
     "output_type": "execute_result",
     "data": {
      "text/plain": [
       "4227490    0\n",
       "1868500    0\n",
       "1584741    0\n",
       "944015     0\n",
       "337619     0\n",
       "          ..\n",
       "2571554    0\n",
       "1642996    0\n",
       "5479260    0\n",
       "3311356    0\n",
       "4189143    0\n",
       "Name: Predice_compra, Length: 1068529, dtype: uint8"
      ]
     },
     "metadata": {},
     "execution_count": 49
    }
   ],
   "metadata": {}
  },
  {
   "cell_type": "code",
   "execution_count": 50,
   "source": [
    "results_df[Target]"
   ],
   "outputs": [
    {
     "output_type": "execute_result",
     "data": {
      "text/plain": [
       "4227490    0\n",
       "1868500    0\n",
       "1584741    1\n",
       "944015     0\n",
       "337619     0\n",
       "          ..\n",
       "2571554    0\n",
       "1642996    0\n",
       "5479260    0\n",
       "3311356    0\n",
       "4189143    0\n",
       "Name: Productos_pension_plan, Length: 1068529, dtype: uint8"
      ]
     },
     "metadata": {},
     "execution_count": 50
    }
   ],
   "metadata": {}
  },
  {
   "cell_type": "code",
   "execution_count": 51,
   "source": [
    "results_df['Success'] = (results_df[Target] == results_df['Predice_compra']).astype(int)"
   ],
   "outputs": [],
   "metadata": {
    "azdata_cell_guid": "de16de6d-2013-49b0-978d-2e0333b7d689"
   }
  },
  {
   "cell_type": "code",
   "execution_count": 52,
   "source": [
    "results_df.head(20)"
   ],
   "outputs": [
    {
     "output_type": "execute_result",
     "data": {
      "text/plain": [
       "         Productos_pension_plan  Predice_compra  Success\n",
       "4227490                       0               0        1\n",
       "1868500                       0               0        1\n",
       "1584741                       1               0        0\n",
       "944015                        0               0        1\n",
       "337619                        0               0        1\n",
       "3162861                       0               0        1\n",
       "5407903                       0               0        1\n",
       "2060447                       0               0        1\n",
       "1816798                       0               0        1\n",
       "4824208                       0               0        1\n",
       "5553705                       0               0        1\n",
       "2768607                       0               0        1\n",
       "4061995                       0               0        1\n",
       "2424551                       0               0        1\n",
       "2042411                       0               0        1\n",
       "4947109                       0               0        1\n",
       "1348542                       0               0        1\n",
       "5469026                       0               0        1\n",
       "4545910                       0               0        1\n",
       "5310079                       0               0        1"
      ],
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
       "      <th>4227490</th>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1868500</th>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1584741</th>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>944015</th>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>337619</th>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3162861</th>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>5407903</th>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2060447</th>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1816798</th>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4824208</th>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>5553705</th>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2768607</th>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4061995</th>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2424551</th>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2042411</th>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4947109</th>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1348542</th>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>5469026</th>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4545910</th>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>5310079</th>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ]
     },
     "metadata": {},
     "execution_count": 52
    }
   ],
   "metadata": {
    "azdata_cell_guid": "f7979794-03e2-4b92-8565-10b5f147f95b"
   }
  },
  {
   "cell_type": "code",
   "execution_count": 53,
   "source": [
    "results_df['Success'].mean()"
   ],
   "outputs": [
    {
     "output_type": "execute_result",
     "data": {
      "text/plain": [
       "0.9654506335345133"
      ]
     },
     "metadata": {},
     "execution_count": 53
    }
   ],
   "metadata": {
    "azdata_cell_guid": "0d216b2f-c7ba-4398-aef9-47639b59cbe4"
   }
  },
  {
   "cell_type": "code",
   "execution_count": 54,
   "source": [
    "results_df['Success'].sum()"
   ],
   "outputs": [
    {
     "output_type": "execute_result",
     "data": {
      "text/plain": [
       "1031612"
      ]
     },
     "metadata": {},
     "execution_count": 54
    }
   ],
   "metadata": {}
  },
  {
   "cell_type": "code",
   "execution_count": 55,
   "source": [
    "print('Hemos acertado {} registros de un total de {}, por tanto el Accuracy es {}.'.format(results_df['Success'].sum(), results_df['Success'].count(), results_df['Success'].mean()))"
   ],
   "outputs": [
    {
     "output_type": "stream",
     "name": "stdout",
     "text": [
      "Hemos acertado 1031612 registros de un total de 1068529, por tanto el Accuracy es 0.9654506335345133.\n"
     ]
    }
   ],
   "metadata": {}
  },
  {
   "cell_type": "markdown",
   "source": [
    "## matriz de confusión"
   ],
   "metadata": {}
  },
  {
   "cell_type": "code",
   "execution_count": 56,
   "source": [
    "confussion_matrix = pd.crosstab(results_df[Target],results_df['Predice_compra'])"
   ],
   "outputs": [],
   "metadata": {
    "azdata_cell_guid": "8a8d7ebc-7df1-4b12-8d98-52e61cd2f44b"
   }
  },
  {
   "cell_type": "code",
   "execution_count": 57,
   "source": [
    "confussion_matrix"
   ],
   "outputs": [
    {
     "output_type": "execute_result",
     "data": {
      "text/plain": [
       "Predice_compra                0\n",
       "Productos_pension_plan         \n",
       "0                       1031612\n",
       "1                         36917"
      ],
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
       "      <td>1031612</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>36917</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ]
     },
     "metadata": {},
     "execution_count": 57
    }
   ],
   "metadata": {
    "azdata_cell_guid": "2bc85b5c-66e3-4539-9e38-f7ffda1ae4ce"
   }
  },
  {
   "cell_type": "code",
   "execution_count": 58,
   "source": [
    "TP = confussion_matrix.iloc[1,1]\r\n",
    "TN = confussion_matrix.iloc[0,0]\r\n",
    "FP = confussion_matrix.iloc[0,1]\r\n",
    "FN = confussion_matrix.iloc[1,0]"
   ],
   "outputs": [
    {
     "output_type": "error",
     "ename": "IndexError",
     "evalue": "single positional indexer is out-of-bounds",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mIndexError\u001b[0m                                Traceback (most recent call last)",
      "\u001b[1;32m<ipython-input-58-6a5337ea87aa>\u001b[0m in \u001b[0;36m<module>\u001b[1;34m\u001b[0m\n\u001b[1;32m----> 1\u001b[1;33m \u001b[0mTP\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mconfussion_matrix\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0miloc\u001b[0m\u001b[1;33m[\u001b[0m\u001b[1;36m1\u001b[0m\u001b[1;33m,\u001b[0m\u001b[1;36m1\u001b[0m\u001b[1;33m]\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m      2\u001b[0m \u001b[0mTN\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mconfussion_matrix\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0miloc\u001b[0m\u001b[1;33m[\u001b[0m\u001b[1;36m0\u001b[0m\u001b[1;33m,\u001b[0m\u001b[1;36m0\u001b[0m\u001b[1;33m]\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m      3\u001b[0m \u001b[0mFP\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mconfussion_matrix\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0miloc\u001b[0m\u001b[1;33m[\u001b[0m\u001b[1;36m0\u001b[0m\u001b[1;33m,\u001b[0m\u001b[1;36m1\u001b[0m\u001b[1;33m]\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m      4\u001b[0m \u001b[0mFN\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mconfussion_matrix\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0miloc\u001b[0m\u001b[1;33m[\u001b[0m\u001b[1;36m1\u001b[0m\u001b[1;33m,\u001b[0m\u001b[1;36m0\u001b[0m\u001b[1;33m]\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;32m~\\anaconda3\\lib\\site-packages\\pandas\\core\\indexing.py\u001b[0m in \u001b[0;36m__getitem__\u001b[1;34m(self, key)\u001b[0m\n\u001b[0;32m    871\u001b[0m                     \u001b[1;31m# AttributeError for IntervalTree get_value\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    872\u001b[0m                     \u001b[1;32mpass\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m--> 873\u001b[1;33m             \u001b[1;32mreturn\u001b[0m \u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0m_getitem_tuple\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mkey\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m    874\u001b[0m         \u001b[1;32melse\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    875\u001b[0m             \u001b[1;31m# we by definition only have the 0th axis\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;32m~\\anaconda3\\lib\\site-packages\\pandas\\core\\indexing.py\u001b[0m in \u001b[0;36m_getitem_tuple\u001b[1;34m(self, tup)\u001b[0m\n\u001b[0;32m   1441\u001b[0m     \u001b[1;32mdef\u001b[0m \u001b[0m_getitem_tuple\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mself\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mtup\u001b[0m\u001b[1;33m:\u001b[0m \u001b[0mTuple\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m   1442\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m-> 1443\u001b[1;33m         \u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0m_has_valid_tuple\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mtup\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m   1444\u001b[0m         \u001b[1;32mtry\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m   1445\u001b[0m             \u001b[1;32mreturn\u001b[0m \u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0m_getitem_lowerdim\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mtup\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;32m~\\anaconda3\\lib\\site-packages\\pandas\\core\\indexing.py\u001b[0m in \u001b[0;36m_has_valid_tuple\u001b[1;34m(self, key)\u001b[0m\n\u001b[0;32m    700\u001b[0m                 \u001b[1;32mraise\u001b[0m \u001b[0mIndexingError\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;34m\"Too many indexers\"\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    701\u001b[0m             \u001b[1;32mtry\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m--> 702\u001b[1;33m                 \u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0m_validate_key\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mk\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mi\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m    703\u001b[0m             \u001b[1;32mexcept\u001b[0m \u001b[0mValueError\u001b[0m \u001b[1;32mas\u001b[0m \u001b[0merr\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    704\u001b[0m                 raise ValueError(\n",
      "\u001b[1;32m~\\anaconda3\\lib\\site-packages\\pandas\\core\\indexing.py\u001b[0m in \u001b[0;36m_validate_key\u001b[1;34m(self, key, axis)\u001b[0m\n\u001b[0;32m   1350\u001b[0m             \u001b[1;32mreturn\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m   1351\u001b[0m         \u001b[1;32melif\u001b[0m \u001b[0mis_integer\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mkey\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m-> 1352\u001b[1;33m             \u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0m_validate_integer\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mkey\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0maxis\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m   1353\u001b[0m         \u001b[1;32melif\u001b[0m \u001b[0misinstance\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mkey\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mtuple\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m   1354\u001b[0m             \u001b[1;31m# a tuple should already have been caught by this point\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;32m~\\anaconda3\\lib\\site-packages\\pandas\\core\\indexing.py\u001b[0m in \u001b[0;36m_validate_integer\u001b[1;34m(self, key, axis)\u001b[0m\n\u001b[0;32m   1435\u001b[0m         \u001b[0mlen_axis\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mlen\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mobj\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0m_get_axis\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0maxis\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m   1436\u001b[0m         \u001b[1;32mif\u001b[0m \u001b[0mkey\u001b[0m \u001b[1;33m>=\u001b[0m \u001b[0mlen_axis\u001b[0m \u001b[1;32mor\u001b[0m \u001b[0mkey\u001b[0m \u001b[1;33m<\u001b[0m \u001b[1;33m-\u001b[0m\u001b[0mlen_axis\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m-> 1437\u001b[1;33m             \u001b[1;32mraise\u001b[0m \u001b[0mIndexError\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;34m\"single positional indexer is out-of-bounds\"\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m   1438\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m   1439\u001b[0m     \u001b[1;31m# -------------------------------------------------------------------\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;31mIndexError\u001b[0m: single positional indexer is out-of-bounds"
     ]
    }
   ],
   "metadata": {
    "azdata_cell_guid": "aad4d3d2-bac2-4ff9-a731-cb146e02c84a"
   }
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "source": [
    "accuracy = (TP + TN) / (TP + TN + FP + FN)\r\n",
    "accuracy"
   ],
   "outputs": [
    {
     "output_type": "execute_result",
     "data": {
      "text/plain": [
       "0.9633007678719493"
      ]
     },
     "metadata": {},
     "execution_count": 135
    }
   ],
   "metadata": {}
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "source": [
    "Precision = TP / (TP + FP)\r\n",
    "Recall = TP/(TP + FN)"
   ],
   "outputs": [],
   "metadata": {
    "azdata_cell_guid": "23be7dd7-b0a0-4f0e-8d78-680fc570fd86"
   }
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "source": [
    "Precision"
   ],
   "outputs": [
    {
     "output_type": "execute_result",
     "data": {
      "text/plain": [
       "0.931525047505339"
      ]
     },
     "metadata": {},
     "execution_count": 137
    }
   ],
   "metadata": {
    "azdata_cell_guid": "e73bd8c8-e21c-4447-b408-59706faf73eb"
   }
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "source": [
    "Recall"
   ],
   "outputs": [
    {
     "output_type": "execute_result",
     "data": {
      "text/plain": [
       "1.0"
      ]
     },
     "metadata": {},
     "execution_count": 138
    }
   ],
   "metadata": {
    "azdata_cell_guid": "0e852a35-c26e-4f68-abb3-f383ed7938b8"
   }
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "source": [
    "for i in range(1,5):\r\n",
    "    dt2 = DecisionTreeClassifier( max_depth= i, min_samples_leaf=50,random_state=42)\r\n",
    "    dt2.fit(X_train,y_train)\r\n",
    "    train_accuracy = dt2.score(X_train, y_train)\r\n",
    "    test_accuracy = dt2.score(X_test, y_test)\r\n",
    "    print('Profundidad del árbol {}. accuracy en train: {} accuracy en test: {}' .format(i,train_accuracy,test_accuracy))\r\n"
   ],
   "outputs": [
    {
     "output_type": "stream",
     "name": "stdout",
     "text": [
      "Profundidad del árbol 1. accuracy en train: 0.8706198853599625 accuracy en test: 0.8716698511121526\n",
      "Profundidad del árbol 2. accuracy en train: 0.9174312340026677 accuracy en test: 0.9183820613576553\n",
      "Profundidad del árbol 3. accuracy en train: 0.9448384945383755 accuracy en test: 0.9453116550704784\n",
      "Profundidad del árbol 4. accuracy en train: 0.9632286672194383 accuracy en test: 0.9633007678719493\n"
     ]
    }
   ],
   "metadata": {
    "azdata_cell_guid": "ddccb47d-7091-4ce5-81b4-1915c718739c"
   }
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "source": [
    "dt_final = DecisionTreeClassifier(max_depth= 4, min_samples_leaf=50,random_state=42)\r\n",
    "dt_final.fit(X_train,y_train)\r\n",
    "train_accuracy = dt_final.score(X_train, y_train)\r\n",
    "test_accuracy = dt_final.score(X_test, y_test)\r\n",
    "val_accuracy = dt_final.score(val_df_X, val_df_y)"
   ],
   "outputs": [],
   "metadata": {
    "azdata_cell_guid": "ed2c648d-7180-4a0d-a8de-f6118d028d5d"
   }
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "source": [
    "print(' acc en train: {} acc en test: {} y acc en val {}' .format(train_accuracy,test_accuracy, val_accuracy))"
   ],
   "outputs": [
    {
     "output_type": "stream",
     "name": "stdout",
     "text": [
      " acc en train: 0.9632286672194383 acc en test: 0.9633007678719493 y acc en val 0.9312238412693676\n"
     ]
    }
   ],
   "metadata": {
    "azdata_cell_guid": "a8b86edc-ecc8-44fa-a76a-1660603e7cca"
   }
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "source": [
    "y_test_pred_proba = pd.DataFrame(dt.predict_proba(X_test)[:,1], index= y_test.index, columns = ['Scoring'])"
   ],
   "outputs": [],
   "metadata": {
    "azdata_cell_guid": "e7066304-9877-4862-ae4a-4da1c380fc9c"
   }
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "source": [
    "y_test_pred_proba"
   ],
   "outputs": [
    {
     "output_type": "execute_result",
     "data": {
      "text/plain": [
       "          Scoring\n",
       "1889574  0.000000\n",
       "676242   0.931519\n",
       "3606281  0.000000\n",
       "4351315  0.000000\n",
       "3075124  0.000000\n",
       "...           ...\n",
       "1073435  0.000000\n",
       "4487474  0.000000\n",
       "968449   0.931519\n",
       "1331837  0.931519\n",
       "1113727  0.931519\n",
       "\n",
       "[110956 rows x 1 columns]"
      ],
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
       "      <th>1889574</th>\n",
       "      <td>0.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>676242</th>\n",
       "      <td>0.931519</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3606281</th>\n",
       "      <td>0.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4351315</th>\n",
       "      <td>0.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3075124</th>\n",
       "      <td>0.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>...</th>\n",
       "      <td>...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1073435</th>\n",
       "      <td>0.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4487474</th>\n",
       "      <td>0.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>968449</th>\n",
       "      <td>0.931519</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1331837</th>\n",
       "      <td>0.931519</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1113727</th>\n",
       "      <td>0.931519</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>110956 rows × 1 columns</p>\n",
       "</div>"
      ]
     },
     "metadata": {},
     "execution_count": 143
    }
   ],
   "metadata": {
    "azdata_cell_guid": "d4ee1a9e-6a10-4f46-b537-261eb8dea063"
   }
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "source": [
    "results_df['Scoring'] = y_test_pred_proba"
   ],
   "outputs": [],
   "metadata": {
    "azdata_cell_guid": "fe57f6d2-e59a-4fa3-9937-d7859f1fa273"
   }
  },
  {
   "cell_type": "markdown",
   "source": [
    "### Usando el módulo metrics "
   ],
   "metadata": {}
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "source": [
    "metrics.roc_auc_score(results_df[Target], results_df['Scoring'])"
   ],
   "outputs": [
    {
     "output_type": "execute_result",
     "data": {
      "text/plain": [
       "0.963355591152067"
      ]
     },
     "metadata": {},
     "execution_count": 145
    }
   ],
   "metadata": {
    "azdata_cell_guid": "591f9a8a-e8b0-4a46-aaa9-d351721520ee"
   }
  },
  {
   "cell_type": "markdown",
   "source": [
    "### Usando el modelo"
   ],
   "metadata": {}
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "source": [
    "dt.score(X_test, y_test)"
   ],
   "outputs": [
    {
     "output_type": "execute_result",
     "data": {
      "text/plain": [
       "0.9633007678719493"
      ]
     },
     "metadata": {},
     "execution_count": 146
    }
   ],
   "metadata": {}
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "source": [],
   "outputs": [],
   "metadata": {}
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "source": [],
   "outputs": [],
   "metadata": {}
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "source": [
    "fpr, tpr, _ = metrics.roc_curve(results_df[Target], results_df['Scoring'])"
   ],
   "outputs": [],
   "metadata": {
    "azdata_cell_guid": "f052ee2b-e442-425a-8fb0-0aa4aca97c4e"
   }
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "source": [
    "plt.clf()\r\n",
    "plt.plot(fpr, tpr)\r\n",
    "plt.plot([0, 1], [0, 1], color='gray', linestyle='--')\r\n",
    "plt.xlabel('FPR')\r\n",
    "plt.ylabel('TPR')\r\n",
    "plt.title('ROC curve')\r\n",
    "plt.show()"
   ],
   "outputs": [
    {
     "output_type": "display_data",
     "data": {
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ],
      "image/png": "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"
     },
     "metadata": {
      "needs_background": "light"
     }
    }
   ],
   "metadata": {
    "azdata_cell_guid": "58c1dde3-d048-412e-83fa-92fe362b921a"
   }
  },
  {
   "cell_type": "markdown",
   "source": [
    "## Valoremos aplicar RandomForestClassifier"
   ],
   "metadata": {}
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "source": [
    "from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier"
   ],
   "outputs": [],
   "metadata": {
    "azdata_cell_guid": "4bd20fc6-4411-40fc-be04-51023d77fe5c"
   }
  },
  {
   "cell_type": "markdown",
   "source": [],
   "metadata": {
    "azdata_cell_guid": "56f7b453-513f-4d12-900a-fc1679745bd2"
   }
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "source": [
    "rf = RandomForestClassifier(n_estimators= 4, max_depth= 4, random_state=42)"
   ],
   "outputs": [],
   "metadata": {
    "azdata_cell_guid": "d17ae683-af99-4b45-a6c3-48bbcc8d509b"
   }
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "source": [
    "rf.fit(X_train,np.ravel(y_train))"
   ],
   "outputs": [
    {
     "output_type": "execute_result",
     "data": {
      "text/plain": [
       "RandomForestClassifier(max_depth=4, n_estimators=4, random_state=42)"
      ]
     },
     "metadata": {},
     "execution_count": 151
    }
   ],
   "metadata": {
    "azdata_cell_guid": "c9c5aadf-4b3a-4c10-9db5-cea8c969bb33"
   }
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "source": [
    "tree_list = rf.estimators_"
   ],
   "outputs": [],
   "metadata": {
    "azdata_cell_guid": "d3f1cad7-55e3-4f06-89dc-fe9aa6df6218"
   }
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "source": [
    "dot_data_rf_0 = export_graphviz(decision_tree=tree_list[0],\r\n",
    "                out_file=None,\r\n",
    "                feature_names= X_test.columns,\r\n",
    "                class_names=['Compra','No compra']\r\n",
    "                )"
   ],
   "outputs": [],
   "metadata": {
    "azdata_cell_guid": "cd41ba25-2fa8-4ae8-869f-509a0097b7fc"
   }
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "source": [
    "graphviz.Source(dot_data_rf_0)"
   ],
   "outputs": [
    {
     "output_type": "execute_result",
     "data": {
      "text/plain": [
       "<graphviz.files.Source at 0x2717878a910>"
      ],
      "image/svg+xml": "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"no\"?>\r\n<!DOCTYPE svg PUBLIC \"-//W3C//DTD SVG 1.1//EN\"\r\n \"http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd\">\r\n<!-- Generated by graphviz version 2.38.0 (20140413.2041)\r\n -->\r\n<!-- Title: Tree Pages: 1 -->\r\n<svg width=\"891pt\" height=\"552pt\"\r\n viewBox=\"0.00 0.00 890.50 552.00\" xmlns=\"http://www.w3.org/2000/svg\" xmlns:xlink=\"http://www.w3.org/1999/xlink\">\r\n<g id=\"graph0\" class=\"graph\" transform=\"scale(1 1) rotate(0) translate(4 548)\">\r\n<title>Tree</title>\r\n<polygon fill=\"white\" stroke=\"none\" points=\"-4,4 -4,-548 886.5,-548 886.5,4 -4,4\"/>\r\n<!-- 0 -->\r\n<g id=\"node1\" class=\"node\"><title>0</title>\r\n<polygon fill=\"none\" stroke=\"black\" points=\"741,-544 543,-544 543,-461 741,-461 741,-544\"/>\r\n<text text-anchor=\"middle\" x=\"642\" y=\"-528.8\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">Productos_pension_plan &lt;= 0.5</text>\r\n<text text-anchor=\"middle\" x=\"642\" y=\"-513.8\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">gini = 0.5</text>\r\n<text text-anchor=\"middle\" x=\"642\" y=\"-498.8\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">samples = 280251</text>\r\n<text text-anchor=\"middle\" x=\"642\" y=\"-483.8\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">value = [222338, 221486]</text>\r\n<text text-anchor=\"middle\" x=\"642\" y=\"-468.8\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">class = Compra</text>\r\n</g>\r\n<!-- 1 -->\r\n<g id=\"node2\" class=\"node\"><title>1</title>\r\n<polygon fill=\"none\" stroke=\"black\" points=\"648,-425 478,-425 478,-342 648,-342 648,-425\"/>\r\n<text text-anchor=\"middle\" x=\"563\" y=\"-409.8\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">entry_channel_KFC &lt;= 0.5</text>\r\n<text text-anchor=\"middle\" x=\"563\" y=\"-394.8\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">gini = 0.5</text>\r\n<text text-anchor=\"middle\" x=\"563\" y=\"-379.8\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">samples = 275043</text>\r\n<text text-anchor=\"middle\" x=\"563\" y=\"-364.8\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">value = [213994, 221486]</text>\r\n<text text-anchor=\"middle\" x=\"563\" y=\"-349.8\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">class = No compra</text>\r\n</g>\r\n<!-- 0&#45;&gt;1 -->\r\n<g id=\"edge1\" class=\"edge\"><title>0&#45;&gt;1</title>\r\n<path fill=\"none\" stroke=\"black\" d=\"M614.591,-460.907C608.586,-452.014 602.169,-442.509 595.971,-433.331\"/>\r\n<polygon fill=\"black\" stroke=\"black\" points=\"598.857,-431.35 590.36,-425.021 593.056,-435.267 598.857,-431.35\"/>\r\n<text text-anchor=\"middle\" x=\"585.602\" y=\"-445.864\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">True</text>\r\n</g>\r\n<!-- 14 -->\r\n<g id=\"node15\" class=\"node\"><title>14</title>\r\n<polygon fill=\"none\" stroke=\"black\" points=\"777.5,-417.5 666.5,-417.5 666.5,-349.5 777.5,-349.5 777.5,-417.5\"/>\r\n<text text-anchor=\"middle\" x=\"722\" y=\"-402.3\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">gini = 0.0</text>\r\n<text text-anchor=\"middle\" x=\"722\" y=\"-387.3\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">samples = 5208</text>\r\n<text text-anchor=\"middle\" x=\"722\" y=\"-372.3\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">value = [8344, 0]</text>\r\n<text text-anchor=\"middle\" x=\"722\" y=\"-357.3\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">class = Compra</text>\r\n</g>\r\n<!-- 0&#45;&gt;14 -->\r\n<g id=\"edge14\" class=\"edge\"><title>0&#45;&gt;14</title>\r\n<path fill=\"none\" stroke=\"black\" d=\"M669.756,-460.907C677.451,-449.652 685.817,-437.418 693.551,-426.106\"/>\r\n<polygon fill=\"black\" stroke=\"black\" points=\"696.567,-427.897 699.322,-417.667 690.788,-423.946 696.567,-427.897\"/>\r\n<text text-anchor=\"middle\" x=\"703.936\" y=\"-438.537\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">False</text>\r\n</g>\r\n<!-- 2 -->\r\n<g id=\"node3\" class=\"node\"><title>2</title>\r\n<polygon fill=\"none\" stroke=\"black\" points=\"540,-306 370,-306 370,-223 540,-223 540,-306\"/>\r\n<text text-anchor=\"middle\" x=\"455\" y=\"-290.8\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">entry_channel_RED &lt;= 0.5</text>\r\n<text text-anchor=\"middle\" x=\"455\" y=\"-275.8\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">gini = 0.49</text>\r\n<text text-anchor=\"middle\" x=\"455\" y=\"-260.8\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">samples = 193485</text>\r\n<text text-anchor=\"middle\" x=\"455\" y=\"-245.8\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">value = [174432, 131810]</text>\r\n<text text-anchor=\"middle\" x=\"455\" y=\"-230.8\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">class = Compra</text>\r\n</g>\r\n<!-- 1&#45;&gt;2 -->\r\n<g id=\"edge2\" class=\"edge\"><title>1&#45;&gt;2</title>\r\n<path fill=\"none\" stroke=\"black\" d=\"M525.53,-341.907C517.069,-332.742 508.009,-322.927 499.298,-313.489\"/>\r\n<polygon fill=\"black\" stroke=\"black\" points=\"501.758,-310.995 492.404,-306.021 496.615,-315.743 501.758,-310.995\"/>\r\n</g>\r\n<!-- 9 -->\r\n<g id=\"node10\" class=\"node\"><title>9</title>\r\n<polygon fill=\"none\" stroke=\"black\" points=\"764,-306 580,-306 580,-223 764,-223 764,-306\"/>\r\n<text text-anchor=\"middle\" x=\"672\" y=\"-290.8\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">Productos_debit_card &lt;= 0.5</text>\r\n<text text-anchor=\"middle\" x=\"672\" y=\"-275.8\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">gini = 0.425</text>\r\n<text text-anchor=\"middle\" x=\"672\" y=\"-260.8\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">samples = 81558</text>\r\n<text text-anchor=\"middle\" x=\"672\" y=\"-245.8\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">value = [39562, 89676]</text>\r\n<text text-anchor=\"middle\" x=\"672\" y=\"-230.8\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">class = No compra</text>\r\n</g>\r\n<!-- 1&#45;&gt;9 -->\r\n<g id=\"edge9\" class=\"edge\"><title>1&#45;&gt;9</title>\r\n<path fill=\"none\" stroke=\"black\" d=\"M600.817,-341.907C609.356,-332.742 618.5,-322.927 627.292,-313.489\"/>\r\n<polygon fill=\"black\" stroke=\"black\" points=\"629.994,-315.723 634.25,-306.021 624.872,-310.952 629.994,-315.723\"/>\r\n</g>\r\n<!-- 3 -->\r\n<g id=\"node4\" class=\"node\"><title>3</title>\r\n<polygon fill=\"none\" stroke=\"black\" points=\"327,-187 157,-187 157,-104 327,-104 327,-187\"/>\r\n<text text-anchor=\"middle\" x=\"242\" y=\"-171.8\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">entry_channel_KFA &lt;= 0.5</text>\r\n<text text-anchor=\"middle\" x=\"242\" y=\"-156.8\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">gini = 0.487</text>\r\n<text text-anchor=\"middle\" x=\"242\" y=\"-141.8\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">samples = 185888</text>\r\n<text text-anchor=\"middle\" x=\"242\" y=\"-126.8\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">value = [171327, 122974]</text>\r\n<text text-anchor=\"middle\" x=\"242\" y=\"-111.8\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">class = Compra</text>\r\n</g>\r\n<!-- 2&#45;&gt;3 -->\r\n<g id=\"edge3\" class=\"edge\"><title>2&#45;&gt;3</title>\r\n<path fill=\"none\" stroke=\"black\" d=\"M381.101,-222.907C362.845,-212.879 343.174,-202.075 324.536,-191.837\"/>\r\n<polygon fill=\"black\" stroke=\"black\" points=\"326.219,-188.768 315.769,-187.021 322.848,-194.903 326.219,-188.768\"/>\r\n</g>\r\n<!-- 6 -->\r\n<g id=\"node7\" class=\"node\"><title>6</title>\r\n<polygon fill=\"none\" stroke=\"black\" points=\"543,-187 367,-187 367,-104 543,-104 543,-187\"/>\r\n<text text-anchor=\"middle\" x=\"455\" y=\"-171.8\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">Productos_securities &lt;= 0.5</text>\r\n<text text-anchor=\"middle\" x=\"455\" y=\"-156.8\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">gini = 0.385</text>\r\n<text text-anchor=\"middle\" x=\"455\" y=\"-141.8\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">samples = 7597</text>\r\n<text text-anchor=\"middle\" x=\"455\" y=\"-126.8\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">value = [3105, 8836]</text>\r\n<text text-anchor=\"middle\" x=\"455\" y=\"-111.8\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">class = No compra</text>\r\n</g>\r\n<!-- 2&#45;&gt;6 -->\r\n<g id=\"edge6\" class=\"edge\"><title>2&#45;&gt;6</title>\r\n<path fill=\"none\" stroke=\"black\" d=\"M455,-222.907C455,-214.649 455,-205.864 455,-197.302\"/>\r\n<polygon fill=\"black\" stroke=\"black\" points=\"458.5,-197.021 455,-187.021 451.5,-197.021 458.5,-197.021\"/>\r\n</g>\r\n<!-- 4 -->\r\n<g id=\"node5\" class=\"node\"><title>4</title>\r\n<polygon fill=\"none\" stroke=\"black\" points=\"158,-68 0,-68 0,-0 158,-0 158,-68\"/>\r\n<text text-anchor=\"middle\" x=\"79\" y=\"-52.8\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">gini = 0.482</text>\r\n<text text-anchor=\"middle\" x=\"79\" y=\"-37.8\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">samples = 177139</text>\r\n<text text-anchor=\"middle\" x=\"79\" y=\"-22.8\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">value = [167107, 113334]</text>\r\n<text text-anchor=\"middle\" x=\"79\" y=\"-7.8\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">class = Compra</text>\r\n</g>\r\n<!-- 3&#45;&gt;4 -->\r\n<g id=\"edge4\" class=\"edge\"><title>3&#45;&gt;4</title>\r\n<path fill=\"none\" stroke=\"black\" d=\"M181.305,-103.726C166.702,-93.9161 151.143,-83.4644 136.682,-73.7496\"/>\r\n<polygon fill=\"black\" stroke=\"black\" points=\"138.562,-70.7963 128.31,-68.1252 134.659,-76.6069 138.562,-70.7963\"/>\r\n</g>\r\n<!-- 5 -->\r\n<g id=\"node6\" class=\"node\"><title>5</title>\r\n<polygon fill=\"none\" stroke=\"black\" points=\"307.5,-68 176.5,-68 176.5,-0 307.5,-0 307.5,-68\"/>\r\n<text text-anchor=\"middle\" x=\"242\" y=\"-52.8\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">gini = 0.424</text>\r\n<text text-anchor=\"middle\" x=\"242\" y=\"-37.8\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">samples = 8749</text>\r\n<text text-anchor=\"middle\" x=\"242\" y=\"-22.8\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">value = [4220, 9640]</text>\r\n<text text-anchor=\"middle\" x=\"242\" y=\"-7.8\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">class = No compra</text>\r\n</g>\r\n<!-- 3&#45;&gt;5 -->\r\n<g id=\"edge5\" class=\"edge\"><title>3&#45;&gt;5</title>\r\n<path fill=\"none\" stroke=\"black\" d=\"M242,-103.726C242,-95.5175 242,-86.8595 242,-78.56\"/>\r\n<polygon fill=\"black\" stroke=\"black\" points=\"245.5,-78.2996 242,-68.2996 238.5,-78.2996 245.5,-78.2996\"/>\r\n</g>\r\n<!-- 7 -->\r\n<g id=\"node8\" class=\"node\"><title>7</title>\r\n<polygon fill=\"none\" stroke=\"black\" points=\"456.5,-68 325.5,-68 325.5,-0 456.5,-0 456.5,-68\"/>\r\n<text text-anchor=\"middle\" x=\"391\" y=\"-52.8\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">gini = 0.384</text>\r\n<text text-anchor=\"middle\" x=\"391\" y=\"-37.8\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">samples = 7589</text>\r\n<text text-anchor=\"middle\" x=\"391\" y=\"-22.8\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">value = [3088, 8836]</text>\r\n<text text-anchor=\"middle\" x=\"391\" y=\"-7.8\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">class = No compra</text>\r\n</g>\r\n<!-- 6&#45;&gt;7 -->\r\n<g id=\"edge7\" class=\"edge\"><title>6&#45;&gt;7</title>\r\n<path fill=\"none\" stroke=\"black\" d=\"M431.169,-103.726C426.104,-95.0615 420.748,-85.8962 415.653,-77.1802\"/>\r\n<polygon fill=\"black\" stroke=\"black\" points=\"418.531,-75.167 410.463,-68.2996 412.487,-78.6992 418.531,-75.167\"/>\r\n</g>\r\n<!-- 8 -->\r\n<g id=\"node9\" class=\"node\"><title>8</title>\r\n<polygon fill=\"none\" stroke=\"black\" points=\"581,-68 475,-68 475,-0 581,-0 581,-68\"/>\r\n<text text-anchor=\"middle\" x=\"528\" y=\"-52.8\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">gini = 0.0</text>\r\n<text text-anchor=\"middle\" x=\"528\" y=\"-37.8\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">samples = 8</text>\r\n<text text-anchor=\"middle\" x=\"528\" y=\"-22.8\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">value = [17, 0]</text>\r\n<text text-anchor=\"middle\" x=\"528\" y=\"-7.8\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">class = Compra</text>\r\n</g>\r\n<!-- 6&#45;&gt;8 -->\r\n<g id=\"edge8\" class=\"edge\"><title>6&#45;&gt;8</title>\r\n<path fill=\"none\" stroke=\"black\" d=\"M482.183,-103.726C488.081,-94.879 494.327,-85.51 500.246,-76.6303\"/>\r\n<polygon fill=\"black\" stroke=\"black\" points=\"503.165,-78.5616 505.8,-68.2996 497.341,-74.6787 503.165,-78.5616\"/>\r\n</g>\r\n<!-- 10 -->\r\n<g id=\"node11\" class=\"node\"><title>10</title>\r\n<polygon fill=\"none\" stroke=\"black\" points=\"753,-187 591,-187 591,-104 753,-104 753,-187\"/>\r\n<text text-anchor=\"middle\" x=\"672\" y=\"-171.8\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">Productos_payroll &lt;= 0.5</text>\r\n<text text-anchor=\"middle\" x=\"672\" y=\"-156.8\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">gini = 0.384</text>\r\n<text text-anchor=\"middle\" x=\"672\" y=\"-141.8\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">samples = 76378</text>\r\n<text text-anchor=\"middle\" x=\"672\" y=\"-126.8\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">value = [31295, 89676]</text>\r\n<text text-anchor=\"middle\" x=\"672\" y=\"-111.8\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">class = No compra</text>\r\n</g>\r\n<!-- 9&#45;&gt;10 -->\r\n<g id=\"edge10\" class=\"edge\"><title>9&#45;&gt;10</title>\r\n<path fill=\"none\" stroke=\"black\" d=\"M672,-222.907C672,-214.649 672,-205.864 672,-197.302\"/>\r\n<polygon fill=\"black\" stroke=\"black\" points=\"675.5,-197.021 672,-187.021 668.5,-197.021 675.5,-197.021\"/>\r\n</g>\r\n<!-- 13 -->\r\n<g id=\"node14\" class=\"node\"><title>13</title>\r\n<polygon fill=\"none\" stroke=\"black\" points=\"882.5,-179.5 771.5,-179.5 771.5,-111.5 882.5,-111.5 882.5,-179.5\"/>\r\n<text text-anchor=\"middle\" x=\"827\" y=\"-164.3\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">gini = 0.0</text>\r\n<text text-anchor=\"middle\" x=\"827\" y=\"-149.3\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">samples = 5180</text>\r\n<text text-anchor=\"middle\" x=\"827\" y=\"-134.3\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">value = [8267, 0]</text>\r\n<text text-anchor=\"middle\" x=\"827\" y=\"-119.3\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">class = Compra</text>\r\n</g>\r\n<!-- 9&#45;&gt;13 -->\r\n<g id=\"edge13\" class=\"edge\"><title>9&#45;&gt;13</title>\r\n<path fill=\"none\" stroke=\"black\" d=\"M725.776,-222.907C741.71,-210.88 759.125,-197.735 774.948,-185.791\"/>\r\n<polygon fill=\"black\" stroke=\"black\" points=\"777.188,-188.485 783.061,-179.667 772.971,-182.898 777.188,-188.485\"/>\r\n</g>\r\n<!-- 11 -->\r\n<g id=\"node12\" class=\"node\"><title>11</title>\r\n<polygon fill=\"none\" stroke=\"black\" points=\"744.5,-68 599.5,-68 599.5,-0 744.5,-0 744.5,-68\"/>\r\n<text text-anchor=\"middle\" x=\"672\" y=\"-52.8\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">gini = 0.363</text>\r\n<text text-anchor=\"middle\" x=\"672\" y=\"-37.8\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">samples = 74358</text>\r\n<text text-anchor=\"middle\" x=\"672\" y=\"-22.8\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">value = [28026, 89676]</text>\r\n<text text-anchor=\"middle\" x=\"672\" y=\"-7.8\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">class = No compra</text>\r\n</g>\r\n<!-- 10&#45;&gt;11 -->\r\n<g id=\"edge11\" class=\"edge\"><title>10&#45;&gt;11</title>\r\n<path fill=\"none\" stroke=\"black\" d=\"M672,-103.726C672,-95.5175 672,-86.8595 672,-78.56\"/>\r\n<polygon fill=\"black\" stroke=\"black\" points=\"675.5,-78.2996 672,-68.2996 668.5,-78.2996 675.5,-78.2996\"/>\r\n</g>\r\n<!-- 12 -->\r\n<g id=\"node13\" class=\"node\"><title>12</title>\r\n<polygon fill=\"none\" stroke=\"black\" points=\"873.5,-68 762.5,-68 762.5,-0 873.5,-0 873.5,-68\"/>\r\n<text text-anchor=\"middle\" x=\"818\" y=\"-52.8\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">gini = 0.0</text>\r\n<text text-anchor=\"middle\" x=\"818\" y=\"-37.8\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">samples = 2020</text>\r\n<text text-anchor=\"middle\" x=\"818\" y=\"-22.8\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">value = [3269, 0]</text>\r\n<text text-anchor=\"middle\" x=\"818\" y=\"-7.8\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">class = Compra</text>\r\n</g>\r\n<!-- 10&#45;&gt;12 -->\r\n<g id=\"edge12\" class=\"edge\"><title>10&#45;&gt;12</title>\r\n<path fill=\"none\" stroke=\"black\" d=\"M726.365,-103.726C739.134,-94.1494 752.719,-83.9611 765.408,-74.4438\"/>\r\n<polygon fill=\"black\" stroke=\"black\" points=\"767.701,-77.0996 773.601,-68.2996 763.501,-71.4996 767.701,-77.0996\"/>\r\n</g>\r\n</g>\r\n</svg>\r\n"
     },
     "metadata": {},
     "execution_count": 154
    }
   ],
   "metadata": {
    "azdata_cell_guid": "a07bafe8-5806-489e-bd29-8e87be4fa322"
   }
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "source": [
    "dot_data_rf_1 = export_graphviz(decision_tree=tree_list[1],\r\n",
    "                out_file=None,\r\n",
    "                feature_names= X_test.columns,\r\n",
    "                class_names=['Compra','No compra']\r\n",
    "                )"
   ],
   "outputs": [],
   "metadata": {
    "azdata_cell_guid": "2e4c9a2b-4eba-4a4a-b7ab-ae2d54d5eb55"
   }
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "source": [
    "graphviz.Source(dot_data_rf_1)"
   ],
   "outputs": [
    {
     "output_type": "execute_result",
     "data": {
      "text/plain": [
       "<graphviz.files.Source at 0x27178b9d8e0>"
      ],
      "image/svg+xml": "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"no\"?>\r\n<!DOCTYPE svg PUBLIC \"-//W3C//DTD SVG 1.1//EN\"\r\n \"http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd\">\r\n<!-- Generated by graphviz version 2.38.0 (20140413.2041)\r\n -->\r\n<!-- Title: Tree Pages: 1 -->\r\n<svg width=\"998pt\" height=\"552pt\"\r\n viewBox=\"0.00 0.00 998.00 552.00\" xmlns=\"http://www.w3.org/2000/svg\" xmlns:xlink=\"http://www.w3.org/1999/xlink\">\r\n<g id=\"graph0\" class=\"graph\" transform=\"scale(1 1) rotate(0) translate(4 548)\">\r\n<title>Tree</title>\r\n<polygon fill=\"white\" stroke=\"none\" points=\"-4,4 -4,-548 994,-548 994,4 -4,4\"/>\r\n<!-- 0 -->\r\n<g id=\"node1\" class=\"node\"><title>0</title>\r\n<polygon fill=\"none\" stroke=\"black\" points=\"590,-544 420,-544 420,-461 590,-461 590,-544\"/>\r\n<text text-anchor=\"middle\" x=\"505\" y=\"-528.8\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">entry_channel_KFA &lt;= 0.5</text>\r\n<text text-anchor=\"middle\" x=\"505\" y=\"-513.8\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">gini = 0.5</text>\r\n<text text-anchor=\"middle\" x=\"505\" y=\"-498.8\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">samples = 280513</text>\r\n<text text-anchor=\"middle\" x=\"505\" y=\"-483.8\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">value = [222051, 221773]</text>\r\n<text text-anchor=\"middle\" x=\"505\" y=\"-468.8\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">class = Compra</text>\r\n</g>\r\n<!-- 1 -->\r\n<g id=\"node2\" class=\"node\"><title>1</title>\r\n<polygon fill=\"none\" stroke=\"black\" points=\"509.5,-425 324.5,-425 324.5,-342 509.5,-342 509.5,-425\"/>\r\n<text text-anchor=\"middle\" x=\"417\" y=\"-409.8\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">Productos_em_acount &lt;= 0.5</text>\r\n<text text-anchor=\"middle\" x=\"417\" y=\"-394.8\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">gini = 0.5</text>\r\n<text text-anchor=\"middle\" x=\"417\" y=\"-379.8\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">samples = 271467</text>\r\n<text text-anchor=\"middle\" x=\"417\" y=\"-364.8\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">value = [217435, 212108]</text>\r\n<text text-anchor=\"middle\" x=\"417\" y=\"-349.8\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">class = Compra</text>\r\n</g>\r\n<!-- 0&#45;&gt;1 -->\r\n<g id=\"edge1\" class=\"edge\"><title>0&#45;&gt;1</title>\r\n<path fill=\"none\" stroke=\"black\" d=\"M474.469,-460.907C467.711,-451.923 460.485,-442.315 453.516,-433.05\"/>\r\n<polygon fill=\"black\" stroke=\"black\" points=\"456.285,-430.909 447.477,-425.021 450.691,-435.116 456.285,-430.909\"/>\r\n<text text-anchor=\"middle\" x=\"443.975\" y=\"-446.074\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">True</text>\r\n</g>\r\n<!-- 8 -->\r\n<g id=\"node9\" class=\"node\"><title>8</title>\r\n<polygon fill=\"none\" stroke=\"black\" points=\"658.5,-425 527.5,-425 527.5,-342 658.5,-342 658.5,-425\"/>\r\n<text text-anchor=\"middle\" x=\"593\" y=\"-409.8\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">recurrencia &lt;= 15.5</text>\r\n<text text-anchor=\"middle\" x=\"593\" y=\"-394.8\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">gini = 0.438</text>\r\n<text text-anchor=\"middle\" x=\"593\" y=\"-379.8\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">samples = 9046</text>\r\n<text text-anchor=\"middle\" x=\"593\" y=\"-364.8\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">value = [4616, 9665]</text>\r\n<text text-anchor=\"middle\" x=\"593\" y=\"-349.8\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">class = No compra</text>\r\n</g>\r\n<!-- 0&#45;&gt;8 -->\r\n<g id=\"edge8\" class=\"edge\"><title>0&#45;&gt;8</title>\r\n<path fill=\"none\" stroke=\"black\" d=\"M535.531,-460.907C542.289,-451.923 549.515,-442.315 556.484,-433.05\"/>\r\n<polygon fill=\"black\" stroke=\"black\" points=\"559.309,-435.116 562.523,-425.021 553.715,-430.909 559.309,-435.116\"/>\r\n<text text-anchor=\"middle\" x=\"566.025\" y=\"-446.074\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">False</text>\r\n</g>\r\n<!-- 2 -->\r\n<g id=\"node3\" class=\"node\"><title>2</title>\r\n<polygon fill=\"none\" stroke=\"black\" points=\"340.5,-306 141.5,-306 141.5,-223 340.5,-223 340.5,-306\"/>\r\n<text text-anchor=\"middle\" x=\"241\" y=\"-290.8\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">Productos_emc_account &lt;= 0.5</text>\r\n<text text-anchor=\"middle\" x=\"241\" y=\"-275.8\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">gini = 0.325</text>\r\n<text text-anchor=\"middle\" x=\"241\" y=\"-260.8\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">samples = 168413</text>\r\n<text text-anchor=\"middle\" x=\"241\" y=\"-245.8\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">value = [54454, 212108]</text>\r\n<text text-anchor=\"middle\" x=\"241\" y=\"-230.8\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">class = No compra</text>\r\n</g>\r\n<!-- 1&#45;&gt;2 -->\r\n<g id=\"edge2\" class=\"edge\"><title>1&#45;&gt;2</title>\r\n<path fill=\"none\" stroke=\"black\" d=\"M355.938,-341.907C341.331,-332.197 325.628,-321.758 310.665,-311.811\"/>\r\n<polygon fill=\"black\" stroke=\"black\" points=\"312.22,-308.642 301.954,-306.021 308.345,-314.472 312.22,-308.642\"/>\r\n</g>\r\n<!-- 7 -->\r\n<g id=\"node8\" class=\"node\"><title>7</title>\r\n<polygon fill=\"none\" stroke=\"black\" points=\"483.5,-298.5 358.5,-298.5 358.5,-230.5 483.5,-230.5 483.5,-298.5\"/>\r\n<text text-anchor=\"middle\" x=\"421\" y=\"-283.3\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">gini = 0.0</text>\r\n<text text-anchor=\"middle\" x=\"421\" y=\"-268.3\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">samples = 103054</text>\r\n<text text-anchor=\"middle\" x=\"421\" y=\"-253.3\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">value = [162981, 0]</text>\r\n<text text-anchor=\"middle\" x=\"421\" y=\"-238.3\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">class = Compra</text>\r\n</g>\r\n<!-- 1&#45;&gt;7 -->\r\n<g id=\"edge7\" class=\"edge\"><title>1&#45;&gt;7</title>\r\n<path fill=\"none\" stroke=\"black\" d=\"M418.388,-341.907C418.754,-331.204 419.15,-319.615 419.52,-308.776\"/>\r\n<polygon fill=\"black\" stroke=\"black\" points=\"423.022,-308.781 419.866,-298.667 416.026,-308.541 423.022,-308.781\"/>\r\n</g>\r\n<!-- 3 -->\r\n<g id=\"node4\" class=\"node\"><title>3</title>\r\n<polygon fill=\"none\" stroke=\"black\" points=\"203,-187 15,-187 15,-104 203,-104 203,-187\"/>\r\n<text text-anchor=\"middle\" x=\"109\" y=\"-171.8\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">Productos_credit_card &lt;= 0.5</text>\r\n<text text-anchor=\"middle\" x=\"109\" y=\"-156.8\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">gini = 0.281</text>\r\n<text text-anchor=\"middle\" x=\"109\" y=\"-141.8\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">samples = 161241</text>\r\n<text text-anchor=\"middle\" x=\"109\" y=\"-126.8\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">value = [43112, 212108]</text>\r\n<text text-anchor=\"middle\" x=\"109\" y=\"-111.8\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">class = No compra</text>\r\n</g>\r\n<!-- 2&#45;&gt;3 -->\r\n<g id=\"edge3\" class=\"edge\"><title>2&#45;&gt;3</title>\r\n<path fill=\"none\" stroke=\"black\" d=\"M195.203,-222.907C184.658,-213.56 173.35,-203.538 162.509,-193.929\"/>\r\n<polygon fill=\"black\" stroke=\"black\" points=\"164.521,-191.035 154.716,-187.021 159.878,-196.273 164.521,-191.035\"/>\r\n</g>\r\n<!-- 6 -->\r\n<g id=\"node7\" class=\"node\"><title>6</title>\r\n<polygon fill=\"none\" stroke=\"black\" points=\"339,-179.5 221,-179.5 221,-111.5 339,-111.5 339,-179.5\"/>\r\n<text text-anchor=\"middle\" x=\"280\" y=\"-164.3\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">gini = 0.0</text>\r\n<text text-anchor=\"middle\" x=\"280\" y=\"-149.3\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">samples = 7172</text>\r\n<text text-anchor=\"middle\" x=\"280\" y=\"-134.3\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">value = [11342, 0]</text>\r\n<text text-anchor=\"middle\" x=\"280\" y=\"-119.3\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">class = Compra</text>\r\n</g>\r\n<!-- 2&#45;&gt;6 -->\r\n<g id=\"edge6\" class=\"edge\"><title>2&#45;&gt;6</title>\r\n<path fill=\"none\" stroke=\"black\" d=\"M254.531,-222.907C258.135,-212.094 262.041,-200.376 265.686,-189.441\"/>\r\n<polygon fill=\"black\" stroke=\"black\" points=\"269.102,-190.261 268.944,-179.667 262.462,-188.047 269.102,-190.261\"/>\r\n</g>\r\n<!-- 4 -->\r\n<g id=\"node5\" class=\"node\"><title>4</title>\r\n<polygon fill=\"none\" stroke=\"black\" points=\"152,-68 0,-68 0,-0 152,-0 152,-68\"/>\r\n<text text-anchor=\"middle\" x=\"76\" y=\"-52.8\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">gini = 0.27</text>\r\n<text text-anchor=\"middle\" x=\"76\" y=\"-37.8\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">samples = 159671</text>\r\n<text text-anchor=\"middle\" x=\"76\" y=\"-22.8\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">value = [40595, 212108]</text>\r\n<text text-anchor=\"middle\" x=\"76\" y=\"-7.8\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">class = No compra</text>\r\n</g>\r\n<!-- 3&#45;&gt;4 -->\r\n<g id=\"edge4\" class=\"edge\"><title>3&#45;&gt;4</title>\r\n<path fill=\"none\" stroke=\"black\" d=\"M96.712,-103.726C94.1832,-95.3351 91.5129,-86.4745 88.9611,-78.0072\"/>\r\n<polygon fill=\"black\" stroke=\"black\" points=\"92.2722,-76.8643 86.0355,-68.2996 85.5699,-78.8842 92.2722,-76.8643\"/>\r\n</g>\r\n<!-- 5 -->\r\n<g id=\"node6\" class=\"node\"><title>5</title>\r\n<polygon fill=\"none\" stroke=\"black\" points=\"281.5,-68 170.5,-68 170.5,-0 281.5,-0 281.5,-68\"/>\r\n<text text-anchor=\"middle\" x=\"226\" y=\"-52.8\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">gini = 0.0</text>\r\n<text text-anchor=\"middle\" x=\"226\" y=\"-37.8\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">samples = 1570</text>\r\n<text text-anchor=\"middle\" x=\"226\" y=\"-22.8\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">value = [2517, 0]</text>\r\n<text text-anchor=\"middle\" x=\"226\" y=\"-7.8\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">class = Compra</text>\r\n</g>\r\n<!-- 3&#45;&gt;5 -->\r\n<g id=\"edge5\" class=\"edge\"><title>3&#45;&gt;5</title>\r\n<path fill=\"none\" stroke=\"black\" d=\"M152.566,-103.726C162.507,-94.423 173.064,-84.5428 182.981,-75.2612\"/>\r\n<polygon fill=\"black\" stroke=\"black\" points=\"185.51,-77.6882 190.42,-68.2996 180.727,-72.5774 185.51,-77.6882\"/>\r\n</g>\r\n<!-- 9 -->\r\n<g id=\"node10\" class=\"node\"><title>9</title>\r\n<polygon fill=\"none\" stroke=\"black\" points=\"671,-306 509,-306 509,-223 671,-223 671,-306\"/>\r\n<text text-anchor=\"middle\" x=\"590\" y=\"-290.8\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">Productos_payroll &lt;= 0.5</text>\r\n<text text-anchor=\"middle\" x=\"590\" y=\"-275.8\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">gini = 0.473</text>\r\n<text text-anchor=\"middle\" x=\"590\" y=\"-260.8\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">samples = 2403</text>\r\n<text text-anchor=\"middle\" x=\"590\" y=\"-245.8\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">value = [2339, 1463]</text>\r\n<text text-anchor=\"middle\" x=\"590\" y=\"-230.8\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">class = Compra</text>\r\n</g>\r\n<!-- 8&#45;&gt;9 -->\r\n<g id=\"edge9\" class=\"edge\"><title>8&#45;&gt;9</title>\r\n<path fill=\"none\" stroke=\"black\" d=\"M591.959,-341.907C591.745,-333.558 591.517,-324.671 591.295,-316.02\"/>\r\n<polygon fill=\"black\" stroke=\"black\" points=\"594.794,-315.928 591.039,-306.021 587.797,-316.107 594.794,-315.928\"/>\r\n</g>\r\n<!-- 14 -->\r\n<g id=\"node15\" class=\"node\"><title>14</title>\r\n<polygon fill=\"none\" stroke=\"black\" points=\"820.5,-306 689.5,-306 689.5,-223 820.5,-223 820.5,-306\"/>\r\n<text text-anchor=\"middle\" x=\"755\" y=\"-290.8\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">deceased_S &lt;= 0.5</text>\r\n<text text-anchor=\"middle\" x=\"755\" y=\"-275.8\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">gini = 0.34</text>\r\n<text text-anchor=\"middle\" x=\"755\" y=\"-260.8\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">samples = 6643</text>\r\n<text text-anchor=\"middle\" x=\"755\" y=\"-245.8\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">value = [2277, 8202]</text>\r\n<text text-anchor=\"middle\" x=\"755\" y=\"-230.8\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">class = No compra</text>\r\n</g>\r\n<!-- 8&#45;&gt;14 -->\r\n<g id=\"edge14\" class=\"edge\"><title>8&#45;&gt;14</title>\r\n<path fill=\"none\" stroke=\"black\" d=\"M649.205,-341.907C662.525,-332.288 676.834,-321.953 690.49,-312.09\"/>\r\n<polygon fill=\"black\" stroke=\"black\" points=\"692.837,-314.713 698.894,-306.021 688.738,-309.038 692.837,-314.713\"/>\r\n</g>\r\n<!-- 10 -->\r\n<g id=\"node11\" class=\"node\"><title>10</title>\r\n<polygon fill=\"none\" stroke=\"black\" points=\"519.5,-187 388.5,-187 388.5,-104 519.5,-104 519.5,-187\"/>\r\n<text text-anchor=\"middle\" x=\"454\" y=\"-171.8\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">age &lt;= 34.5</text>\r\n<text text-anchor=\"middle\" x=\"454\" y=\"-156.8\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">gini = 0.485</text>\r\n<text text-anchor=\"middle\" x=\"454\" y=\"-141.8\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">samples = 2245</text>\r\n<text text-anchor=\"middle\" x=\"454\" y=\"-126.8\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">value = [2082, 1463]</text>\r\n<text text-anchor=\"middle\" x=\"454\" y=\"-111.8\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">class = Compra</text>\r\n</g>\r\n<!-- 9&#45;&gt;10 -->\r\n<g id=\"edge10\" class=\"edge\"><title>9&#45;&gt;10</title>\r\n<path fill=\"none\" stroke=\"black\" d=\"M542.815,-222.907C531.845,-213.469 520.074,-203.343 508.806,-193.649\"/>\r\n<polygon fill=\"black\" stroke=\"black\" points=\"510.964,-190.889 501.101,-187.021 506.399,-196.196 510.964,-190.889\"/>\r\n</g>\r\n<!-- 13 -->\r\n<g id=\"node14\" class=\"node\"><title>13</title>\r\n<polygon fill=\"none\" stroke=\"black\" points=\"644,-179.5 538,-179.5 538,-111.5 644,-111.5 644,-179.5\"/>\r\n<text text-anchor=\"middle\" x=\"591\" y=\"-164.3\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">gini = 0.0</text>\r\n<text text-anchor=\"middle\" x=\"591\" y=\"-149.3\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">samples = 158</text>\r\n<text text-anchor=\"middle\" x=\"591\" y=\"-134.3\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">value = [257, 0]</text>\r\n<text text-anchor=\"middle\" x=\"591\" y=\"-119.3\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">class = Compra</text>\r\n</g>\r\n<!-- 9&#45;&gt;13 -->\r\n<g id=\"edge13\" class=\"edge\"><title>9&#45;&gt;13</title>\r\n<path fill=\"none\" stroke=\"black\" d=\"M590.347,-222.907C590.438,-212.204 590.537,-200.615 590.63,-189.776\"/>\r\n<polygon fill=\"black\" stroke=\"black\" points=\"594.131,-189.697 590.717,-179.667 587.131,-189.637 594.131,-189.697\"/>\r\n</g>\r\n<!-- 11 -->\r\n<g id=\"node12\" class=\"node\"><title>11</title>\r\n<polygon fill=\"none\" stroke=\"black\" points=\"470,-68 346,-68 346,-0 470,-0 470,-68\"/>\r\n<text text-anchor=\"middle\" x=\"408\" y=\"-52.8\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">gini = 0.485</text>\r\n<text text-anchor=\"middle\" x=\"408\" y=\"-37.8\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">samples = 766</text>\r\n<text text-anchor=\"middle\" x=\"408\" y=\"-22.8\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">value = [492, 698]</text>\r\n<text text-anchor=\"middle\" x=\"408\" y=\"-7.8\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">class = No compra</text>\r\n</g>\r\n<!-- 10&#45;&gt;11 -->\r\n<g id=\"edge11\" class=\"edge\"><title>10&#45;&gt;11</title>\r\n<path fill=\"none\" stroke=\"black\" d=\"M436.871,-103.726C433.308,-95.2439 429.543,-86.2819 425.951,-77.7312\"/>\r\n<polygon fill=\"black\" stroke=\"black\" points=\"429.089,-76.1635 421.989,-68.2996 422.635,-78.8747 429.089,-76.1635\"/>\r\n</g>\r\n<!-- 12 -->\r\n<g id=\"node13\" class=\"node\"><title>12</title>\r\n<polygon fill=\"none\" stroke=\"black\" points=\"613.5,-68 488.5,-68 488.5,-0 613.5,-0 613.5,-68\"/>\r\n<text text-anchor=\"middle\" x=\"551\" y=\"-52.8\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">gini = 0.439</text>\r\n<text text-anchor=\"middle\" x=\"551\" y=\"-37.8\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">samples = 1479</text>\r\n<text text-anchor=\"middle\" x=\"551\" y=\"-22.8\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">value = [1590, 765]</text>\r\n<text text-anchor=\"middle\" x=\"551\" y=\"-7.8\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">class = Compra</text>\r\n</g>\r\n<!-- 10&#45;&gt;12 -->\r\n<g id=\"edge12\" class=\"edge\"><title>10&#45;&gt;12</title>\r\n<path fill=\"none\" stroke=\"black\" d=\"M490.119,-103.726C498.199,-94.6054 506.77,-84.93 514.851,-75.8078\"/>\r\n<polygon fill=\"black\" stroke=\"black\" points=\"517.491,-78.1058 521.502,-68.2996 512.251,-73.4642 517.491,-78.1058\"/>\r\n</g>\r\n<!-- 15 -->\r\n<g id=\"node16\" class=\"node\"><title>15</title>\r\n<polygon fill=\"none\" stroke=\"black\" points=\"847.5,-187 662.5,-187 662.5,-104 847.5,-104 847.5,-187\"/>\r\n<text text-anchor=\"middle\" x=\"755\" y=\"-171.8\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">Productos_em_acount &lt;= 0.5</text>\r\n<text text-anchor=\"middle\" x=\"755\" y=\"-156.8\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">gini = 0.34</text>\r\n<text text-anchor=\"middle\" x=\"755\" y=\"-141.8\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">samples = 6641</text>\r\n<text text-anchor=\"middle\" x=\"755\" y=\"-126.8\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">value = [2277, 8200]</text>\r\n<text text-anchor=\"middle\" x=\"755\" y=\"-111.8\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">class = No compra</text>\r\n</g>\r\n<!-- 14&#45;&gt;15 -->\r\n<g id=\"edge15\" class=\"edge\"><title>14&#45;&gt;15</title>\r\n<path fill=\"none\" stroke=\"black\" d=\"M755,-222.907C755,-214.649 755,-205.864 755,-197.302\"/>\r\n<polygon fill=\"black\" stroke=\"black\" points=\"758.5,-197.021 755,-187.021 751.5,-197.021 758.5,-197.021\"/>\r\n</g>\r\n<!-- 18 -->\r\n<g id=\"node19\" class=\"node\"><title>18</title>\r\n<polygon fill=\"none\" stroke=\"black\" points=\"990,-179.5 866,-179.5 866,-111.5 990,-111.5 990,-179.5\"/>\r\n<text text-anchor=\"middle\" x=\"928\" y=\"-164.3\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">gini = 0.0</text>\r\n<text text-anchor=\"middle\" x=\"928\" y=\"-149.3\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">samples = 2</text>\r\n<text text-anchor=\"middle\" x=\"928\" y=\"-134.3\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">value = [0, 2]</text>\r\n<text text-anchor=\"middle\" x=\"928\" y=\"-119.3\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">class = No compra</text>\r\n</g>\r\n<!-- 14&#45;&gt;18 -->\r\n<g id=\"edge18\" class=\"edge\"><title>14&#45;&gt;18</title>\r\n<path fill=\"none\" stroke=\"black\" d=\"M815.021,-222.907C832.969,-210.769 852.6,-197.493 870.389,-185.462\"/>\r\n<polygon fill=\"black\" stroke=\"black\" points=\"872.636,-188.168 878.958,-179.667 868.714,-182.37 872.636,-188.168\"/>\r\n</g>\r\n<!-- 16 -->\r\n<g id=\"node17\" class=\"node\"><title>16</title>\r\n<polygon fill=\"none\" stroke=\"black\" points=\"791.5,-68 660.5,-68 660.5,-0 791.5,-0 791.5,-68\"/>\r\n<text text-anchor=\"middle\" x=\"726\" y=\"-52.8\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">gini = 0.241</text>\r\n<text text-anchor=\"middle\" x=\"726\" y=\"-37.8\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">samples = 6054</text>\r\n<text text-anchor=\"middle\" x=\"726\" y=\"-22.8\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">value = [1339, 8200]</text>\r\n<text text-anchor=\"middle\" x=\"726\" y=\"-7.8\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">class = No compra</text>\r\n</g>\r\n<!-- 15&#45;&gt;16 -->\r\n<g id=\"edge16\" class=\"edge\"><title>15&#45;&gt;16</title>\r\n<path fill=\"none\" stroke=\"black\" d=\"M744.201,-103.726C741.979,-95.3351 739.633,-86.4745 737.39,-78.0072\"/>\r\n<polygon fill=\"black\" stroke=\"black\" points=\"740.763,-77.0703 734.819,-68.2996 733.996,-78.8624 740.763,-77.0703\"/>\r\n</g>\r\n<!-- 17 -->\r\n<g id=\"node18\" class=\"node\"><title>17</title>\r\n<polygon fill=\"none\" stroke=\"black\" points=\"916,-68 810,-68 810,-0 916,-0 916,-68\"/>\r\n<text text-anchor=\"middle\" x=\"863\" y=\"-52.8\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">gini = 0.0</text>\r\n<text text-anchor=\"middle\" x=\"863\" y=\"-37.8\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">samples = 587</text>\r\n<text text-anchor=\"middle\" x=\"863\" y=\"-22.8\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">value = [938, 0]</text>\r\n<text text-anchor=\"middle\" x=\"863\" y=\"-7.8\" font-family=\"Times New Roman,serif\" font-size=\"14.00\">class = Compra</text>\r\n</g>\r\n<!-- 15&#45;&gt;17 -->\r\n<g id=\"edge17\" class=\"edge\"><title>15&#45;&gt;17</title>\r\n<path fill=\"none\" stroke=\"black\" d=\"M795.215,-103.726C804.301,-94.5142 813.945,-84.7364 823.021,-75.5343\"/>\r\n<polygon fill=\"black\" stroke=\"black\" points=\"825.626,-77.877 830.157,-68.2996 820.643,-72.9615 825.626,-77.877\"/>\r\n</g>\r\n</g>\r\n</svg>\r\n"
     },
     "metadata": {},
     "execution_count": 156
    }
   ],
   "metadata": {
    "azdata_cell_guid": "abcdef67-d5aa-4bfc-9fb6-f5badec18d27"
   }
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "source": [
    "top_features = pd.Series(rf.feature_importances_, index = X_train.columns).sort_values(ascending = False).head(20)"
   ],
   "outputs": [],
   "metadata": {
    "azdata_cell_guid": "03a664ca-f3eb-4d2f-80f7-f00848c6ffb7"
   }
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "source": [
    "top_features"
   ],
   "outputs": [
    {
     "output_type": "execute_result",
     "data": {
      "text/plain": [
       "Productos_em_acount            5.163729e-01\n",
       "entry_channel_KFC              1.756524e-01\n",
       "Productos_debit_card           6.545047e-02\n",
       "age                            5.031594e-02\n",
       "recurrencia                    4.208394e-02\n",
       "Productos_pension_plan         3.142743e-02\n",
       "Productos_payroll              2.753445e-02\n",
       "Productos_emc_account          2.589529e-02\n",
       "entry_channel_KFA              1.818614e-02\n",
       "entry_channel_RED              1.766954e-02\n",
       "Productos_long_term_deposit    1.310871e-02\n",
       "days_between                   6.699834e-03\n",
       "Productos_credit_card          5.858398e-03\n",
       "entry_channel_KHD              2.174584e-03\n",
       "días_encartera                 1.224128e-03\n",
       "Month                          1.696969e-04\n",
       "Productos_securities           1.383307e-04\n",
       "gender_V                       3.195453e-05\n",
       "entry_channel_KHN              5.510790e-06\n",
       "deceased_S                     3.151417e-07\n",
       "dtype: float64"
      ]
     },
     "metadata": {},
     "execution_count": 158
    }
   ],
   "metadata": {}
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "source": [
    "y_score = pd.DataFrame(rf.predict_proba(X_test)[:,1], index= y_test.index, columns=['PredictScore'])"
   ],
   "outputs": [],
   "metadata": {
    "azdata_cell_guid": "1281cfe6-1e1e-4c1e-a6f5-c8537a1f1cce"
   }
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "source": [
    "rf_results_df = y_test.join(y_score)"
   ],
   "outputs": [],
   "metadata": {
    "azdata_cell_guid": "b86df0ea-a456-4230-bc1c-66336dcf9106"
   }
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "source": [
    "rf_results_df.sort_values(by='PredictScore', ascending=False)"
   ],
   "outputs": [
    {
     "output_type": "execute_result",
     "data": {
      "text/plain": [
       "         Productos_payroll_account  PredictScore\n",
       "910363                           1      0.836641\n",
       "1284616                          1      0.836641\n",
       "868214                           1      0.836641\n",
       "883329                           1      0.836641\n",
       "1149894                          1      0.836641\n",
       "...                            ...           ...\n",
       "3742546                          0      0.144805\n",
       "3264155                          0      0.144805\n",
       "3333994                          0      0.144805\n",
       "2814294                          0      0.144805\n",
       "2784414                          0      0.144805\n",
       "\n",
       "[110956 rows x 2 columns]"
      ],
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
       "      <th>Productos_payroll_account</th>\n",
       "      <th>PredictScore</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>910363</th>\n",
       "      <td>1</td>\n",
       "      <td>0.836641</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1284616</th>\n",
       "      <td>1</td>\n",
       "      <td>0.836641</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>868214</th>\n",
       "      <td>1</td>\n",
       "      <td>0.836641</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>883329</th>\n",
       "      <td>1</td>\n",
       "      <td>0.836641</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1149894</th>\n",
       "      <td>1</td>\n",
       "      <td>0.836641</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>...</th>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3742546</th>\n",
       "      <td>0</td>\n",
       "      <td>0.144805</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3264155</th>\n",
       "      <td>0</td>\n",
       "      <td>0.144805</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3333994</th>\n",
       "      <td>0</td>\n",
       "      <td>0.144805</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2814294</th>\n",
       "      <td>0</td>\n",
       "      <td>0.144805</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2784414</th>\n",
       "      <td>0</td>\n",
       "      <td>0.144805</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>110956 rows × 2 columns</p>\n",
       "</div>"
      ]
     },
     "metadata": {},
     "execution_count": 161
    }
   ],
   "metadata": {
    "azdata_cell_guid": "f3bf26e2-c47f-48bc-adb5-4d45895498fd"
   }
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "source": [
    "metrics.roc_auc_score(rf_results_df[Target], rf_results_df['PredictScore'])"
   ],
   "outputs": [
    {
     "output_type": "execute_result",
     "data": {
      "text/plain": [
       "0.9596318519355371"
      ]
     },
     "metadata": {},
     "execution_count": 163
    }
   ],
   "metadata": {
    "azdata_cell_guid": "d91dcaf9-f374-48ad-9a36-e42669e27c5a"
   }
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "source": [
    "fpr, tpr, thre = metrics.roc_curve(rf_results_df[Target], rf_results_df['PredictScore'])"
   ],
   "outputs": [],
   "metadata": {
    "azdata_cell_guid": "571f53d5-dac8-4252-b8af-9f5db4d95831"
   }
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "source": [
    "plt.plot(fpr,tpr)\r\n",
    "plt.plot([0,1],[0,1], color = 'grey')"
   ],
   "outputs": [
    {
     "output_type": "execute_result",
     "data": {
      "text/plain": [
       "[<matplotlib.lines.Line2D at 0x2717fe1c9d0>]"
      ]
     },
     "metadata": {},
     "execution_count": 165
    },
    {
     "output_type": "display_data",
     "data": {
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ],
      "image/png": "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"
     },
     "metadata": {
      "needs_background": "light"
     }
    }
   ],
   "metadata": {
    "azdata_cell_guid": "c2eba321-4ca3-4201-a3b5-3777b74884a9"
   }
  },
  {
   "cell_type": "markdown",
   "source": [
    "## Valoremos aplicar Gradient boosting"
   ],
   "metadata": {
    "azdata_cell_guid": "9b413f60-aa72-4563-88c5-1f23fdab9808"
   }
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "source": [
    "import xgboost as xgb"
   ],
   "outputs": [],
   "metadata": {
    "azdata_cell_guid": "cf89cbf2-7801-451b-be4b-b65405121d1d"
   }
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "source": [
    "xgb_model = xgb.XGBClassifier(max_depth = 5, n_estimators = 80, random_state = 42)"
   ],
   "outputs": [],
   "metadata": {
    "azdata_cell_guid": "11b03064-831d-4262-a0eb-022cb675708a"
   }
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "source": [
    "xgb_model.fit(X_train,y_train)"
   ],
   "outputs": [
    {
     "output_type": "stream",
     "name": "stdout",
     "text": [
      "[18:40:59] WARNING: C:/Users/Administrator/workspace/xgboost-win64_release_1.4.0/src/learner.cc:1095: Starting in XGBoost 1.3.0, the default evaluation metric used with the objective 'binary:logistic' was changed from 'error' to 'logloss'. Explicitly set eval_metric if you'd like to restore the old behavior.\n"
     ]
    },
    {
     "output_type": "execute_result",
     "data": {
      "text/plain": [
       "XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,\n",
       "              colsample_bynode=1, colsample_bytree=1, gamma=0, gpu_id=-1,\n",
       "              importance_type='gain', interaction_constraints='',\n",
       "              learning_rate=0.300000012, max_delta_step=0, max_depth=5,\n",
       "              min_child_weight=1, missing=nan, monotone_constraints='()',\n",
       "              n_estimators=80, n_jobs=12, num_parallel_tree=1, random_state=42,\n",
       "              reg_alpha=0, reg_lambda=1, scale_pos_weight=1, subsample=1,\n",
       "              tree_method='exact', validate_parameters=1, verbosity=None)"
      ]
     },
     "metadata": {},
     "execution_count": 168
    }
   ],
   "metadata": {
    "azdata_cell_guid": "f58c8bd4-f574-4dc1-bf9b-0472191b460e"
   }
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "source": [
    "metrics.roc_auc_score(y_test, xgb_model.predict_proba(X_test)[:,1])"
   ],
   "outputs": [
    {
     "output_type": "execute_result",
     "data": {
      "text/plain": [
       "1.0000000000000002"
      ]
     },
     "metadata": {},
     "execution_count": 169
    }
   ],
   "metadata": {
    "azdata_cell_guid": "59215585-fd5e-4a06-8809-498b310858cb"
   }
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "source": [
    "fpr, tpr, thre = metrics.roc_curve(y_test, xgb_model.predict_proba(X_test)[:,1])"
   ],
   "outputs": [],
   "metadata": {
    "azdata_cell_guid": "17d137ca-f633-41a4-9382-e4eed35d1117"
   }
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "source": [
    "plt.plot(fpr,tpr)\n",
    "plt.plot([0,1],[0,1], color = 'grey')"
   ],
   "outputs": [
    {
     "output_type": "execute_result",
     "data": {
      "text/plain": [
       "[<matplotlib.lines.Line2D at 0x2704220d1c0>]"
      ]
     },
     "metadata": {},
     "execution_count": 171
    },
    {
     "output_type": "display_data",
     "data": {
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ],
      "image/png": "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"
     },
     "metadata": {
      "needs_background": "light"
     }
    }
   ],
   "metadata": {
    "azdata_cell_guid": "c053ff36-551a-48f4-843b-6a72a069b35d"
   }
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "source": [],
   "outputs": [],
   "metadata": {
    "azdata_cell_guid": "6cd04b55-09db-4c88-ab46-06004debb21b"
   }
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "source": [],
   "outputs": [],
   "metadata": {
    "azdata_cell_guid": "3123bc85-8ba7-413e-9e40-d37a90bdaa1b"
   }
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "source": [],
   "outputs": [],
   "metadata": {
    "azdata_cell_guid": "0283c188-f787-4564-8403-7cd2cca47ef2"
   }
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "source": [],
   "outputs": [],
   "metadata": {
    "azdata_cell_guid": "4b7f9341-1080-4444-b17f-e93bc4269769"
   }
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "source": [],
   "outputs": [],
   "metadata": {
    "azdata_cell_guid": "1f92b8a4-ae80-48ae-b189-b8685717ed6d"
   }
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "source": [],
   "outputs": [],
   "metadata": {
    "azdata_cell_guid": "ed084fc7-7227-4fca-a40b-02af6d46eac6"
   }
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "source": [],
   "outputs": [],
   "metadata": {
    "azdata_cell_guid": "5746858b-c4d6-4b35-8444-3cdbdacc72b3"
   }
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "source": [],
   "outputs": [],
   "metadata": {
    "azdata_cell_guid": "d4b5f6eb-cc6c-4182-a4d3-b287f94fabf6"
   }
  },
  {
   "cell_type": "markdown",
   "source": [
    "### NO SUPERVISADO - SEGMENTACIÓN"
   ],
   "metadata": {
    "azdata_cell_guid": "09141c62-5dd1-42b2-800c-5179a1463eb8"
   }
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "source": [
    "\n",
    "standard_scaler = StandardScaler()\n",
    "scaled_df = standard_scaler.fit_transform(full_kmeans)\n",
    "scaled_df = pd.DataFrame(scaled_df, index = full_kmeans.index, columns = full_kmeans.columns)\n",
    "scaled_df.shape"
   ],
   "outputs": [
    {
     "output_type": "error",
     "ename": "NameError",
     "evalue": "name 'full_kmeans' is not defined",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mNameError\u001b[0m                                 Traceback (most recent call last)",
      "\u001b[1;32m<ipython-input-152-9c4966fb2cb0>\u001b[0m in \u001b[0;36m<module>\u001b[1;34m\u001b[0m\n\u001b[0;32m      1\u001b[0m \u001b[0mstandard_scaler\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mStandardScaler\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m----> 2\u001b[1;33m \u001b[0mscaled_df\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mstandard_scaler\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mfit_transform\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mfull_kmeans\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m      3\u001b[0m \u001b[0mscaled_df\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mpd\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mDataFrame\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mscaled_df\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mindex\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mfull_kmeans\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mindex\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mcolumns\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mfull_kmeans\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mcolumns\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m      4\u001b[0m \u001b[0mscaled_df\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mshape\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;31mNameError\u001b[0m: name 'full_kmeans' is not defined"
     ]
    }
   ],
   "metadata": {
    "azdata_cell_guid": "41b3724f-c128-43c6-ad16-2a4c0be6078f"
   }
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "source": [
    "# import the function to compute cosine_similarity\n",
    "from sklearn.metrics.pairwise import cosine_similarity\n",
    "from sklearn.decomposition import PCA\n",
    "from sklearn.cluster import KMeans\n",
    "from sklearn.impute import SimpleImputer"
   ],
   "outputs": [],
   "metadata": {
    "azdata_cell_guid": "de4be44b-29ab-4157-8833-da597eda7a2d"
   }
  },
  {
   "cell_type": "markdown",
   "source": [
    "Reducción de la dimensionalidad con PCA "
   ],
   "metadata": {
    "azdata_cell_guid": "4351d410-3418-4b04-975d-5a2dd10b32ce"
   }
  },
  {
   "cell_type": "code",
   "execution_count": null,
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
   ],
   "outputs": [],
   "metadata": {
    "azdata_cell_guid": "41afa36a-a993-4735-8cca-dc90894cce8a"
   }
  },
  {
   "cell_type": "code",
   "execution_count": null,
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
   ],
   "outputs": [],
   "metadata": {
    "azdata_cell_guid": "1046d9fb-d02d-46d4-945d-bbe86964f3d6"
   }
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "source": [
    "kmeans = KMeans(n_clusters = 7)\n",
    "kmeans.fit(scaled_df)\n",
    "full_kmeans['cluster'] = kmeans.labels_"
   ],
   "outputs": [],
   "metadata": {
    "azdata_cell_guid": "6c580fd7-3b03-4c87-9c13-142d0866e5e1"
   }
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "source": [
    "# fiteamos un modelo con k = 5 (que hemos sacado de la elbow curve anterior) \n",
    "# y con el dataframe escalado y sin outliers\n",
    "\n",
    "cluster_model = KMeans(n_clusters = 7)\n",
    "cluster_model.fit(scaled_df)"
   ],
   "outputs": [],
   "metadata": {
    "azdata_cell_guid": "268a7b64-740e-4b89-a1c4-79c239fdad34"
   }
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "source": [
    "# generamos el dataframe escalado (con el scaler del paso anterior, entrado sin outliers) pero con todos los datos.\n",
    "# por tanto vamos a transformar incluso a los outliers pero con el scaler entrado sin ellos.\n",
    "# el motivo es porque los outliers pueden afectar mucho la media y la desviación utilizado para transformar.\n",
    "scaled_df_with_outliers = standard_scaler.transform(scaled_df)"
   ],
   "outputs": [],
   "metadata": {
    "azdata_cell_guid": "f52a2303-5ca1-4ef6-9b52-3bb54b538de0"
   }
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "source": [
    "# convertimos a dataframe\n",
    "scaled_df_with_outliers = pd.DataFrame(scaled_df_with_outliers, \n",
    "                                       index = scaled_df.index, \n",
    "                                       columns = scaled_df.columns)"
   ],
   "outputs": [],
   "metadata": {
    "azdata_cell_guid": "4fbbfd73-7221-487a-9500-4de8ae9fbfcf"
   }
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "source": [
    "scaled_df_with_outliers.shape"
   ],
   "outputs": [],
   "metadata": {
    "azdata_cell_guid": "47fd721c-2bbb-44c8-b841-aa0934cd1856"
   }
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "source": [
    "# calculamos el cluster de cada cliente, a partir del dataframe escalado y con outliers\n",
    "labels = cluster_model.predict(scaled_df_with_outliers)"
   ],
   "outputs": [],
   "metadata": {
    "azdata_cell_guid": "df69b23e-f39d-46dd-8d01-82b37627f484"
   }
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "source": [
    "scaled_df[\"cluster\"] = labels\n",
    "scaled_df.head(15).T"
   ],
   "outputs": [],
   "metadata": {
    "azdata_cell_guid": "ef125b83-8447-4326-9dd7-56db90c5ad4b"
   }
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "source": [],
   "outputs": [],
   "metadata": {
    "azdata_cell_guid": "d983d911-735e-4b41-a188-2153c4a4ac5f"
   }
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "source": [
    "scaled_df.columns"
   ],
   "outputs": [],
   "metadata": {
    "azdata_cell_guid": "93b15f24-ec9f-44c2-b68d-d0c2bee7b507"
   }
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "source": [
    "pd.to_pickle(scaled_df, './full_4.pickle')"
   ],
   "outputs": [],
   "metadata": {
    "azdata_cell_guid": "d59eaf35-1c95-43ce-9f2e-25b8d90dd6a2"
   }
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "source": [
    "# visualizamos nuestros grupos en base a las variables del modelo RFM, para ver que tal han quedado.\n",
    "selected_columns = ['Ingresos', 'días_encartera', 'Compras','Familia_prod_AhorroVista', 'Familia_prod_Crédito',\n",
    "       'Familia_prod_Inversión']\n",
    "\n",
    "sns.pairplot(scaled_df, vars = selected_columns, hue = 'cluster');"
   ],
   "outputs": [],
   "metadata": {
    "azdata_cell_guid": "23304a0b-3298-4db4-b7c9-bccc78eb8a8f"
   }
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "source": [
    "## Recomendación \"user based\""
   ],
   "outputs": [],
   "metadata": {
    "azdata_cell_guid": "85c2f52a-b663-49e9-8721-0d0854a910d4"
   }
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "source": [],
   "outputs": [],
   "metadata": {
    "azdata_cell_guid": "37b90ab8-1498-4570-923b-295e54d70391"
   }
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