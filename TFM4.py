{
    "metadata": {
        "kernelspec": {
            "name": "python3",
            "display_name": "Python 3.8.5 64-bit ('base': conda)"
        },
        "language_info": {
            "name": "python",
            "version": "3.8.5",
            "mimetype": "text/x-python",
            "codemirror_mode": {
                "name": "ipython",
                "version": 3
            },
            "pygments_lexer": "ipython3",
            "nbconvert_exporter": "python",
            "file_extension": ".py"
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
        },
        "interpreter": {
            "hash": "02b54ae4fcef88d4936c518f12e4c1a4a3ccb83eb22d3060789450414c249266"
        }
    },
    "nbformat_minor": 2,
    "nbformat": 4,
    "cells": [
        {
            "cell_type": "code",
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
            ],
            "metadata": {
                "azdata_cell_guid": "f0e72443-ffd0-4847-9a85-1c375a72bc4f"
            },
            "outputs": [],
            "execution_count": 25
        },
        {
            "cell_type": "code",
            "source": [
                "# python core library for machine learning and data science\n",
                "from sklearn.pipeline import Pipeline\n",
                "from sklearn.preprocessing import StandardScaler, MinMaxScaler\n",
                "from sklearn.base import BaseEstimator, TransformerMixin\n",
                "from sklearn.impute import KNNImputer, SimpleImputer\n",
                "from sklearn.cluster import KMeans"
            ],
            "metadata": {
                "azdata_cell_guid": "bc7f16db-3af4-4634-82d5-881a8786ce32"
            },
            "outputs": [],
            "execution_count": 26
        },
        {
            "cell_type": "code",
            "source": [
                "full_3 = pd.read_pickle('./full_3.pickle')"
            ],
            "metadata": {
                "azdata_cell_guid": "22afde4a-8fe0-498a-adb8-c185a70c0a30"
            },
            "outputs": [],
            "execution_count": 27
        },
        {
            "cell_type": "code",
            "execution_count": 28,
            "metadata": {},
            "outputs": [
                {
                    "output_type": "execute_result",
                    "data": {
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
                            "       recurrencia  \n",
                            "22134            1  \n",
                            "22135            2  \n",
                            "22136            3  \n",
                            "27826            1  "
                        ],
                        "text/html": "<div>\n<style scoped>\n    .dataframe tbody tr th:only-of-type {\n        vertical-align: middle;\n    }\n\n    .dataframe tbody tr th {\n        vertical-align: top;\n    }\n\n    .dataframe thead th {\n        text-align: right;\n    }\n</style>\n<table border=\"1\" class=\"dataframe\">\n  <thead>\n    <tr style=\"text-align: right;\">\n      <th></th>\n      <th>Productos</th>\n      <th>Ventas_cant</th>\n      <th>Precios</th>\n      <th>Familia_prod</th>\n      <th>Ingresos</th>\n      <th>entry_date</th>\n      <th>entry_channel</th>\n      <th>días_encartera</th>\n      <th>pk_cid</th>\n      <th>country_id</th>\n      <th>region_code</th>\n      <th>gender</th>\n      <th>age</th>\n      <th>deceased</th>\n      <th>date</th>\n      <th>Month</th>\n      <th>days_between</th>\n      <th>recurrencia</th>\n    </tr>\n  </thead>\n  <tbody>\n    <tr>\n      <th>22134</th>\n      <td>short_term_deposit</td>\n      <td>1.0</td>\n      <td>40</td>\n      <td>Inversión</td>\n      <td>40.0</td>\n      <td>2018-04-08</td>\n      <td>KHK</td>\n      <td>20.0</td>\n      <td>100296</td>\n      <td>ES</td>\n      <td>28.0</td>\n      <td>V</td>\n      <td>46</td>\n      <td>N</td>\n      <td>2018-04-28</td>\n      <td>4</td>\n      <td>0.0</td>\n      <td>1</td>\n    </tr>\n    <tr>\n      <th>22135</th>\n      <td>short_term_deposit</td>\n      <td>1.0</td>\n      <td>40</td>\n      <td>Inversión</td>\n      <td>40.0</td>\n      <td>2018-04-08</td>\n      <td>KHK</td>\n      <td>50.0</td>\n      <td>100296</td>\n      <td>ES</td>\n      <td>28.0</td>\n      <td>V</td>\n      <td>46</td>\n      <td>N</td>\n      <td>2018-05-28</td>\n      <td>5</td>\n      <td>30.0</td>\n      <td>2</td>\n    </tr>\n    <tr>\n      <th>22136</th>\n      <td>short_term_deposit</td>\n      <td>1.0</td>\n      <td>40</td>\n      <td>Inversión</td>\n      <td>40.0</td>\n      <td>2018-04-08</td>\n      <td>KHK</td>\n      <td>81.0</td>\n      <td>100296</td>\n      <td>ES</td>\n      <td>28.0</td>\n      <td>V</td>\n      <td>46</td>\n      <td>N</td>\n      <td>2018-06-28</td>\n      <td>6</td>\n      <td>31.0</td>\n      <td>3</td>\n    </tr>\n    <tr>\n      <th>27826</th>\n      <td>short_term_deposit</td>\n      <td>1.0</td>\n      <td>40</td>\n      <td>Inversión</td>\n      <td>40.0</td>\n      <td>2015-01-30</td>\n      <td>KHM</td>\n      <td>1337.0</td>\n      <td>1003705</td>\n      <td>ES</td>\n      <td>28.0</td>\n      <td>H</td>\n      <td>33</td>\n      <td>N</td>\n      <td>2018-09-28</td>\n      <td>9</td>\n      <td>0.0</td>\n      <td>1</td>\n    </tr>\n  </tbody>\n</table>\n</div>"
                    },
                    "metadata": {},
                    "execution_count": 28
                }
            ],
            "source": [
                "full_3.head(4)"
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
            "source": [
                "categóricas = full_3.select_dtypes(exclude=np.number)\n",
                "categóricas.head(4)"
            ],
            "metadata": {
                "azdata_cell_guid": "df59c42a-b4b7-4d0c-93bf-1b1ba230e1f6"
            },
            "outputs": [
                {
                    "output_type": "execute_result",
                    "data": {
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
                        ],
                        "text/html": "<div>\n<style scoped>\n    .dataframe tbody tr th:only-of-type {\n        vertical-align: middle;\n    }\n\n    .dataframe tbody tr th {\n        vertical-align: top;\n    }\n\n    .dataframe thead th {\n        text-align: right;\n    }\n</style>\n<table border=\"1\" class=\"dataframe\">\n  <thead>\n    <tr style=\"text-align: right;\">\n      <th></th>\n      <th>Productos</th>\n      <th>Familia_prod</th>\n      <th>entry_date</th>\n      <th>entry_channel</th>\n      <th>pk_cid</th>\n      <th>country_id</th>\n      <th>gender</th>\n      <th>deceased</th>\n      <th>date</th>\n    </tr>\n  </thead>\n  <tbody>\n    <tr>\n      <th>22134</th>\n      <td>short_term_deposit</td>\n      <td>Inversión</td>\n      <td>2018-04-08</td>\n      <td>KHK</td>\n      <td>100296</td>\n      <td>ES</td>\n      <td>V</td>\n      <td>N</td>\n      <td>2018-04-28</td>\n    </tr>\n    <tr>\n      <th>22135</th>\n      <td>short_term_deposit</td>\n      <td>Inversión</td>\n      <td>2018-04-08</td>\n      <td>KHK</td>\n      <td>100296</td>\n      <td>ES</td>\n      <td>V</td>\n      <td>N</td>\n      <td>2018-05-28</td>\n    </tr>\n    <tr>\n      <th>22136</th>\n      <td>short_term_deposit</td>\n      <td>Inversión</td>\n      <td>2018-04-08</td>\n      <td>KHK</td>\n      <td>100296</td>\n      <td>ES</td>\n      <td>V</td>\n      <td>N</td>\n      <td>2018-06-28</td>\n    </tr>\n    <tr>\n      <th>27826</th>\n      <td>short_term_deposit</td>\n      <td>Inversión</td>\n      <td>2015-01-30</td>\n      <td>KHM</td>\n      <td>1003705</td>\n      <td>ES</td>\n      <td>H</td>\n      <td>N</td>\n      <td>2018-09-28</td>\n    </tr>\n  </tbody>\n</table>\n</div>"
                    },
                    "metadata": {},
                    "execution_count": 29
                }
            ],
            "execution_count": 29
        },
        {
            "cell_type": "code",
            "source": [
                "numéricas = full_3.select_dtypes(include=np.number)\n",
                "numéricas.head(2)"
            ],
            "metadata": {
                "azdata_cell_guid": "612ec501-3763-4aa2-bafc-e5b0bb19c8b9"
            },
            "outputs": [
                {
                    "output_type": "execute_result",
                    "data": {
                        "text/plain": [
                            "       Ventas_cant  Precios  Ingresos  días_encartera  region_code  age  \\\n",
                            "22134          1.0       40      40.0            20.0         28.0   46   \n",
                            "22135          1.0       40      40.0            50.0         28.0   46   \n",
                            "\n",
                            "       Month  days_between  recurrencia  \n",
                            "22134      4           0.0            1  \n",
                            "22135      5          30.0            2  "
                        ],
                        "text/html": "<div>\n<style scoped>\n    .dataframe tbody tr th:only-of-type {\n        vertical-align: middle;\n    }\n\n    .dataframe tbody tr th {\n        vertical-align: top;\n    }\n\n    .dataframe thead th {\n        text-align: right;\n    }\n</style>\n<table border=\"1\" class=\"dataframe\">\n  <thead>\n    <tr style=\"text-align: right;\">\n      <th></th>\n      <th>Ventas_cant</th>\n      <th>Precios</th>\n      <th>Ingresos</th>\n      <th>días_encartera</th>\n      <th>region_code</th>\n      <th>age</th>\n      <th>Month</th>\n      <th>days_between</th>\n      <th>recurrencia</th>\n    </tr>\n  </thead>\n  <tbody>\n    <tr>\n      <th>22134</th>\n      <td>1.0</td>\n      <td>40</td>\n      <td>40.0</td>\n      <td>20.0</td>\n      <td>28.0</td>\n      <td>46</td>\n      <td>4</td>\n      <td>0.0</td>\n      <td>1</td>\n    </tr>\n    <tr>\n      <th>22135</th>\n      <td>1.0</td>\n      <td>40</td>\n      <td>40.0</td>\n      <td>50.0</td>\n      <td>28.0</td>\n      <td>46</td>\n      <td>5</td>\n      <td>30.0</td>\n      <td>2</td>\n    </tr>\n  </tbody>\n</table>\n</div>"
                    },
                    "metadata": {},
                    "execution_count": 30
                }
            ],
            "execution_count": 30
        },
        {
            "cell_type": "code",
            "source": [
                "\n",
                "unique_cat = categóricas.nunique()\n",
                "nulls_cat = categóricas.isnull().sum()\n",
                "nan_cat_df = pd.DataFrame({#'column': sample.columns,\n",
                "                        'nulls': nulls_cat,\n",
                "                        'unique': unique_cat\n",
                "})\n",
                "nan_cat_df.sort_values('unique', inplace=True, ascending=False)\n",
                "nan_cat_df.to_csv('nan_cat_df.csv')\n",
                "nan_cat_df"
            ],
            "metadata": {
                "scrolled": true,
                "azdata_cell_guid": "01c3f924-7442-42e9-aec3-a5a2626d6dc2"
            },
            "outputs": [
                {
                    "output_type": "execute_result",
                    "data": {
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
                        ],
                        "text/html": "<div>\n<style scoped>\n    .dataframe tbody tr th:only-of-type {\n        vertical-align: middle;\n    }\n\n    .dataframe tbody tr th {\n        vertical-align: top;\n    }\n\n    .dataframe thead th {\n        text-align: right;\n    }\n</style>\n<table border=\"1\" class=\"dataframe\">\n  <thead>\n    <tr style=\"text-align: right;\">\n      <th></th>\n      <th>nulls</th>\n      <th>unique</th>\n    </tr>\n  </thead>\n  <tbody>\n    <tr>\n      <th>pk_cid</th>\n      <td>0</td>\n      <td>350384</td>\n    </tr>\n    <tr>\n      <th>entry_date</th>\n      <td>0</td>\n      <td>1411</td>\n    </tr>\n    <tr>\n      <th>entry_channel</th>\n      <td>0</td>\n      <td>60</td>\n    </tr>\n    <tr>\n      <th>country_id</th>\n      <td>0</td>\n      <td>37</td>\n    </tr>\n    <tr>\n      <th>date</th>\n      <td>0</td>\n      <td>17</td>\n    </tr>\n    <tr>\n      <th>Productos</th>\n      <td>0</td>\n      <td>14</td>\n    </tr>\n    <tr>\n      <th>Familia_prod</th>\n      <td>0</td>\n      <td>3</td>\n    </tr>\n    <tr>\n      <th>gender</th>\n      <td>0</td>\n      <td>2</td>\n    </tr>\n    <tr>\n      <th>deceased</th>\n      <td>0</td>\n      <td>2</td>\n    </tr>\n  </tbody>\n</table>\n</div>"
                    },
                    "metadata": {},
                    "execution_count": 31
                }
            ],
            "execution_count": 31
        },
        {
            "cell_type": "code",
            "source": [
                "def plot_cat_values(dataframe, column):\n",
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
                "    plt.show()"
            ],
            "metadata": {
                "azdata_cell_guid": "8da39fde-462a-48a9-adda-433b57fc959f"
            },
            "outputs": [],
            "execution_count": 32
        },
        {
            "cell_type": "code",
            "source": [
                "full_3['Familia_prod'].value_counts()"
            ],
            "metadata": {
                "azdata_cell_guid": "9b90f1de-7548-487a-91a1-65d8081321c7"
            },
            "outputs": [
                {
                    "output_type": "execute_result",
                    "data": {
                        "text/plain": [
                            "AhorroVista    5806772\n",
                            "Inversión       376088\n",
                            "Crédito          71658\n",
                            "Name: Familia_prod, dtype: int64"
                        ]
                    },
                    "metadata": {},
                    "execution_count": 33
                }
            ],
            "execution_count": 33
        },
        {
            "cell_type": "code",
            "source": [
                "#Agrupo las variables con menos de 61 observaciones únicas\n",
                "menos_61 = []\n",
                "mas_61 = []\n",
                "\n",
                "for i in categóricas:\n",
                "      \n",
                "        if full_3[i].nunique() < 61:\n",
                "            menos_61.append(i)\n",
                "            \n",
                "        else:\n",
                "            mas_61.append(i)"
            ],
            "metadata": {
                "azdata_cell_guid": "1aeffbba-b7b2-42a1-a636-31dcab6a8fb3"
            },
            "outputs": [],
            "execution_count": 34
        },
        {
            "cell_type": "code",
            "source": [
                "import plotly.express as px"
            ],
            "metadata": {
                "azdata_cell_guid": "2730be9d-6657-4e9d-808b-26eec38d75be"
            },
            "outputs": [],
            "execution_count": 35
        },
        {
            "cell_type": "code",
            "source": [
                "grafico_horizontal= full_3.groupby([\"Productos\",\"date\"])[\"Ingresos\"].sum().reset_index()\n",
                "grafico_horizontal\n",
                "\n",
                "evolucion_horizontal= px.bar(grafico_horizontal, x=\"date\", y=\"Ingresos\", \\\n",
                "                        color=\"Productos\", orientation=\"v\", color_discrete_sequence=px.colors.cyclical.mygbm)\n",
                "\n",
                "evolucion_horizontal.show()"
            ],
            "metadata": {
                "azdata_cell_guid": "2e3a75dc-59c5-4dc9-9414-c7eada08b0cf"
            },
            "outputs": [
                {
                    "output_type": "display_data",
                    "data": {
                        "application/vnd.plotly.v1+json": {
                            "config": {
                                "plotlyServerURL": "https://plot.ly"
                            },
                            "data": [
                                {
                                    "alignmentgroup": "True",
                                    "hovertemplate": "Productos=credit_card<br>date=%{x}<br>Ingresos=%{y}<extra></extra>",
                                    "legendgroup": "credit_card",
                                    "marker": {
                                        "color": "#ef55f1"
                                    },
                                    "name": "credit_card",
                                    "offsetgroup": "credit_card",
                                    "orientation": "v",
                                    "showlegend": true,
                                    "textposition": "auto",
                                    "type": "bar",
                                    "x": [
                                        "2018-01-28T00:00:00",
                                        "2018-02-28T00:00:00",
                                        "2018-03-28T00:00:00",
                                        "2018-04-28T00:00:00",
                                        "2018-05-28T00:00:00",
                                        "2018-06-28T00:00:00",
                                        "2018-07-28T00:00:00",
                                        "2018-08-28T00:00:00",
                                        "2018-09-28T00:00:00",
                                        "2018-10-28T00:00:00",
                                        "2018-11-28T00:00:00",
                                        "2018-12-28T00:00:00",
                                        "2019-01-28T00:00:00",
                                        "2019-02-28T00:00:00",
                                        "2019-03-28T00:00:00",
                                        "2019-04-28T00:00:00",
                                        "2019-05-28T00:00:00"
                                    ],
                                    "xaxis": "x",
                                    "y": [
                                        199500,
                                        196920,
                                        210660,
                                        223500,
                                        227820,
                                        240120,
                                        248220,
                                        252360,
                                        257340,
                                        263100,
                                        270720,
                                        272160,
                                        269580,
                                        272520,
                                        274560,
                                        284820,
                                        288060
                                    ],
                                    "yaxis": "y"
                                },
                                {
                                    "alignmentgroup": "True",
                                    "hovertemplate": "Productos=debit_card<br>date=%{x}<br>Ingresos=%{y}<extra></extra>",
                                    "legendgroup": "debit_card",
                                    "marker": {
                                        "color": "#fb84ce"
                                    },
                                    "name": "debit_card",
                                    "offsetgroup": "debit_card",
                                    "orientation": "v",
                                    "showlegend": true,
                                    "textposition": "auto",
                                    "type": "bar",
                                    "x": [
                                        "2018-01-28T00:00:00",
                                        "2018-02-28T00:00:00",
                                        "2018-03-28T00:00:00",
                                        "2018-04-28T00:00:00",
                                        "2018-05-28T00:00:00",
                                        "2018-06-28T00:00:00",
                                        "2018-07-28T00:00:00",
                                        "2018-08-28T00:00:00",
                                        "2018-09-28T00:00:00",
                                        "2018-10-28T00:00:00",
                                        "2018-11-28T00:00:00",
                                        "2018-12-28T00:00:00",
                                        "2019-01-28T00:00:00",
                                        "2019-02-28T00:00:00",
                                        "2019-03-28T00:00:00",
                                        "2019-04-28T00:00:00",
                                        "2019-05-28T00:00:00"
                                    ],
                                    "xaxis": "x",
                                    "y": [
                                        246960,
                                        254480,
                                        270460,
                                        277640,
                                        279110,
                                        288280,
                                        295780,
                                        292050,
                                        316840,
                                        343720,
                                        354660,
                                        374300,
                                        373500,
                                        393990,
                                        413900,
                                        421400,
                                        432610
                                    ],
                                    "yaxis": "y"
                                },
                                {
                                    "alignmentgroup": "True",
                                    "hovertemplate": "Productos=em_account_p<br>date=%{x}<br>Ingresos=%{y}<extra></extra>",
                                    "legendgroup": "em_account_p",
                                    "marker": {
                                        "color": "#fbafa1"
                                    },
                                    "name": "em_account_p",
                                    "offsetgroup": "em_account_p",
                                    "orientation": "v",
                                    "showlegend": true,
                                    "textposition": "auto",
                                    "type": "bar",
                                    "x": [
                                        "2018-01-28T00:00:00",
                                        "2018-02-28T00:00:00",
                                        "2018-03-28T00:00:00",
                                        "2018-04-28T00:00:00",
                                        "2018-05-28T00:00:00",
                                        "2018-06-28T00:00:00",
                                        "2018-07-28T00:00:00",
                                        "2018-08-28T00:00:00",
                                        "2018-09-28T00:00:00",
                                        "2018-10-28T00:00:00",
                                        "2018-11-28T00:00:00",
                                        "2018-12-28T00:00:00",
                                        "2019-01-28T00:00:00",
                                        "2019-02-28T00:00:00",
                                        "2019-03-28T00:00:00",
                                        "2019-04-28T00:00:00",
                                        "2019-05-28T00:00:00"
                                    ],
                                    "xaxis": "x",
                                    "y": [
                                        20,
                                        20,
                                        20,
                                        20,
                                        20,
                                        20,
                                        20,
                                        20,
                                        20,
                                        20,
                                        20,
                                        20,
                                        20,
                                        20,
                                        20,
                                        20,
                                        20
                                    ],
                                    "yaxis": "y"
                                },
                                {
                                    "alignmentgroup": "True",
                                    "hovertemplate": "Productos=em_acount<br>date=%{x}<br>Ingresos=%{y}<extra></extra>",
                                    "legendgroup": "em_acount",
                                    "marker": {
                                        "color": "#fcd471"
                                    },
                                    "name": "em_acount",
                                    "offsetgroup": "em_acount",
                                    "orientation": "v",
                                    "showlegend": true,
                                    "textposition": "auto",
                                    "type": "bar",
                                    "x": [
                                        "2018-01-28T00:00:00",
                                        "2018-02-28T00:00:00",
                                        "2018-03-28T00:00:00",
                                        "2018-04-28T00:00:00",
                                        "2018-05-28T00:00:00",
                                        "2018-06-28T00:00:00",
                                        "2018-07-28T00:00:00",
                                        "2018-08-28T00:00:00",
                                        "2018-09-28T00:00:00",
                                        "2018-10-28T00:00:00",
                                        "2018-11-28T00:00:00",
                                        "2018-12-28T00:00:00",
                                        "2019-01-28T00:00:00",
                                        "2019-02-28T00:00:00",
                                        "2019-03-28T00:00:00",
                                        "2019-04-28T00:00:00",
                                        "2019-05-28T00:00:00"
                                    ],
                                    "xaxis": "x",
                                    "y": [
                                        2152930,
                                        2170980,
                                        2186830,
                                        2198160,
                                        2212910,
                                        2243280,
                                        2343240,
                                        2459800,
                                        2612100,
                                        2777070,
                                        2849000,
                                        2889280,
                                        2917860,
                                        2929880,
                                        2950460,
                                        2958440,
                                        2963800
                                    ],
                                    "yaxis": "y"
                                },
                                {
                                    "alignmentgroup": "True",
                                    "hovertemplate": "Productos=emc_account<br>date=%{x}<br>Ingresos=%{y}<extra></extra>",
                                    "legendgroup": "emc_account",
                                    "marker": {
                                        "color": "#f0ed35"
                                    },
                                    "name": "emc_account",
                                    "offsetgroup": "emc_account",
                                    "orientation": "v",
                                    "showlegend": true,
                                    "textposition": "auto",
                                    "type": "bar",
                                    "x": [
                                        "2018-01-28T00:00:00",
                                        "2018-02-28T00:00:00",
                                        "2018-03-28T00:00:00",
                                        "2018-04-28T00:00:00",
                                        "2018-05-28T00:00:00",
                                        "2018-06-28T00:00:00",
                                        "2018-07-28T00:00:00",
                                        "2018-08-28T00:00:00",
                                        "2018-09-28T00:00:00",
                                        "2018-10-28T00:00:00",
                                        "2018-11-28T00:00:00",
                                        "2018-12-28T00:00:00",
                                        "2019-01-28T00:00:00",
                                        "2019-02-28T00:00:00",
                                        "2019-03-28T00:00:00",
                                        "2019-04-28T00:00:00",
                                        "2019-05-28T00:00:00"
                                    ],
                                    "xaxis": "x",
                                    "y": [
                                        153200,
                                        158270,
                                        164280,
                                        169470,
                                        175690,
                                        179500,
                                        181850,
                                        183330,
                                        186180,
                                        188440,
                                        194940,
                                        200430,
                                        209210,
                                        217960,
                                        224800,
                                        234550,
                                        247510
                                    ],
                                    "yaxis": "y"
                                },
                                {
                                    "alignmentgroup": "True",
                                    "hovertemplate": "Productos=funds<br>date=%{x}<br>Ingresos=%{y}<extra></extra>",
                                    "legendgroup": "funds",
                                    "marker": {
                                        "color": "#c6e516"
                                    },
                                    "name": "funds",
                                    "offsetgroup": "funds",
                                    "orientation": "v",
                                    "showlegend": true,
                                    "textposition": "auto",
                                    "type": "bar",
                                    "x": [
                                        "2018-01-28T00:00:00",
                                        "2018-02-28T00:00:00",
                                        "2018-03-28T00:00:00",
                                        "2018-04-28T00:00:00",
                                        "2018-05-28T00:00:00",
                                        "2018-06-28T00:00:00",
                                        "2018-07-28T00:00:00",
                                        "2018-08-28T00:00:00",
                                        "2018-09-28T00:00:00",
                                        "2018-10-28T00:00:00",
                                        "2018-11-28T00:00:00",
                                        "2018-12-28T00:00:00",
                                        "2019-01-28T00:00:00",
                                        "2019-02-28T00:00:00",
                                        "2019-03-28T00:00:00",
                                        "2019-04-28T00:00:00",
                                        "2019-05-28T00:00:00"
                                    ],
                                    "xaxis": "x",
                                    "y": [
                                        31440,
                                        34880,
                                        38760,
                                        43360,
                                        45800,
                                        46360,
                                        46880,
                                        48760,
                                        49480,
                                        49960,
                                        51280,
                                        52880,
                                        53200,
                                        52640,
                                        52880,
                                        52800,
                                        52600
                                    ],
                                    "yaxis": "y"
                                },
                                {
                                    "alignmentgroup": "True",
                                    "hovertemplate": "Productos=loans<br>date=%{x}<br>Ingresos=%{y}<extra></extra>",
                                    "legendgroup": "loans",
                                    "marker": {
                                        "color": "#96d310"
                                    },
                                    "name": "loans",
                                    "offsetgroup": "loans",
                                    "orientation": "v",
                                    "showlegend": true,
                                    "textposition": "auto",
                                    "type": "bar",
                                    "x": [
                                        "2018-01-28T00:00:00",
                                        "2018-02-28T00:00:00",
                                        "2018-03-28T00:00:00",
                                        "2018-04-28T00:00:00",
                                        "2018-05-28T00:00:00",
                                        "2018-06-28T00:00:00",
                                        "2018-07-28T00:00:00",
                                        "2018-08-28T00:00:00",
                                        "2018-09-28T00:00:00",
                                        "2018-10-28T00:00:00",
                                        "2018-11-28T00:00:00",
                                        "2018-12-28T00:00:00",
                                        "2019-01-28T00:00:00",
                                        "2019-02-28T00:00:00",
                                        "2019-03-28T00:00:00",
                                        "2019-04-28T00:00:00",
                                        "2019-05-28T00:00:00"
                                    ],
                                    "xaxis": "x",
                                    "y": [
                                        1140,
                                        1140,
                                        1380,
                                        1440,
                                        1620,
                                        1620,
                                        1680,
                                        1740,
                                        1860,
                                        1980,
                                        1920,
                                        1800,
                                        1680,
                                        1740,
                                        1740,
                                        1800,
                                        1800
                                    ],
                                    "yaxis": "y"
                                },
                                {
                                    "alignmentgroup": "True",
                                    "hovertemplate": "Productos=long_term_deposit<br>date=%{x}<br>Ingresos=%{y}<extra></extra>",
                                    "legendgroup": "long_term_deposit",
                                    "marker": {
                                        "color": "#61c10b"
                                    },
                                    "name": "long_term_deposit",
                                    "offsetgroup": "long_term_deposit",
                                    "orientation": "v",
                                    "showlegend": true,
                                    "textposition": "auto",
                                    "type": "bar",
                                    "x": [
                                        "2018-01-28T00:00:00",
                                        "2018-02-28T00:00:00",
                                        "2018-03-28T00:00:00",
                                        "2018-04-28T00:00:00",
                                        "2018-05-28T00:00:00",
                                        "2018-06-28T00:00:00",
                                        "2018-07-28T00:00:00",
                                        "2018-08-28T00:00:00",
                                        "2018-09-28T00:00:00",
                                        "2018-10-28T00:00:00",
                                        "2018-11-28T00:00:00",
                                        "2018-12-28T00:00:00",
                                        "2019-01-28T00:00:00",
                                        "2019-02-28T00:00:00",
                                        "2019-03-28T00:00:00",
                                        "2019-04-28T00:00:00",
                                        "2019-05-28T00:00:00"
                                    ],
                                    "xaxis": "x",
                                    "y": [
                                        195360,
                                        198240,
                                        198600,
                                        204280,
                                        214240,
                                        223520,
                                        231840,
                                        236920,
                                        246320,
                                        255200,
                                        257880,
                                        269960,
                                        266360,
                                        266320,
                                        261480,
                                        254720,
                                        245160
                                    ],
                                    "yaxis": "y"
                                },
                                {
                                    "alignmentgroup": "True",
                                    "hovertemplate": "Productos=mortgage<br>date=%{x}<br>Ingresos=%{y}<extra></extra>",
                                    "legendgroup": "mortgage",
                                    "marker": {
                                        "color": "#31ac28"
                                    },
                                    "name": "mortgage",
                                    "offsetgroup": "mortgage",
                                    "orientation": "v",
                                    "showlegend": true,
                                    "textposition": "auto",
                                    "type": "bar",
                                    "x": [
                                        "2018-01-28T00:00:00",
                                        "2018-02-28T00:00:00",
                                        "2018-03-28T00:00:00",
                                        "2018-04-28T00:00:00",
                                        "2018-05-28T00:00:00",
                                        "2018-06-28T00:00:00",
                                        "2018-07-28T00:00:00",
                                        "2018-08-28T00:00:00",
                                        "2018-09-28T00:00:00",
                                        "2018-10-28T00:00:00",
                                        "2018-11-28T00:00:00",
                                        "2018-12-28T00:00:00",
                                        "2019-01-28T00:00:00",
                                        "2019-02-28T00:00:00",
                                        "2019-03-28T00:00:00",
                                        "2019-04-28T00:00:00",
                                        "2019-05-28T00:00:00"
                                    ],
                                    "xaxis": "x",
                                    "y": [
                                        900,
                                        900,
                                        1020,
                                        1020,
                                        1080,
                                        1140,
                                        1200,
                                        1200,
                                        1200,
                                        1140,
                                        1140,
                                        1140,
                                        1200,
                                        1200,
                                        1200,
                                        1380,
                                        1380
                                    ],
                                    "yaxis": "y"
                                },
                                {
                                    "alignmentgroup": "True",
                                    "hovertemplate": "Productos=payroll<br>date=%{x}<br>Ingresos=%{y}<extra></extra>",
                                    "legendgroup": "payroll",
                                    "marker": {
                                        "color": "#439064"
                                    },
                                    "name": "payroll",
                                    "offsetgroup": "payroll",
                                    "orientation": "v",
                                    "showlegend": true,
                                    "textposition": "auto",
                                    "type": "bar",
                                    "x": [
                                        "2018-01-28T00:00:00",
                                        "2018-02-28T00:00:00",
                                        "2018-03-28T00:00:00",
                                        "2018-04-28T00:00:00",
                                        "2018-05-28T00:00:00",
                                        "2018-06-28T00:00:00",
                                        "2018-07-28T00:00:00",
                                        "2018-08-28T00:00:00",
                                        "2018-09-28T00:00:00",
                                        "2018-10-28T00:00:00",
                                        "2018-11-28T00:00:00",
                                        "2018-12-28T00:00:00",
                                        "2019-01-28T00:00:00",
                                        "2019-02-28T00:00:00",
                                        "2019-03-28T00:00:00",
                                        "2019-04-28T00:00:00",
                                        "2019-05-28T00:00:00"
                                    ],
                                    "xaxis": "x",
                                    "y": [
                                        81450,
                                        88890,
                                        97350,
                                        99050,
                                        99940,
                                        109460,
                                        118880,
                                        113830,
                                        118500,
                                        124540,
                                        130320,
                                        144520,
                                        120580,
                                        143780,
                                        151140,
                                        152310,
                                        163330
                                    ],
                                    "yaxis": "y"
                                },
                                {
                                    "alignmentgroup": "True",
                                    "hovertemplate": "Productos=payroll_account<br>date=%{x}<br>Ingresos=%{y}<extra></extra>",
                                    "legendgroup": "payroll_account",
                                    "marker": {
                                        "color": "#3d719a"
                                    },
                                    "name": "payroll_account",
                                    "offsetgroup": "payroll_account",
                                    "orientation": "v",
                                    "showlegend": true,
                                    "textposition": "auto",
                                    "type": "bar",
                                    "x": [
                                        "2018-01-28T00:00:00",
                                        "2018-02-28T00:00:00",
                                        "2018-03-28T00:00:00",
                                        "2018-04-28T00:00:00",
                                        "2018-05-28T00:00:00",
                                        "2018-06-28T00:00:00",
                                        "2018-07-28T00:00:00",
                                        "2018-08-28T00:00:00",
                                        "2018-09-28T00:00:00",
                                        "2018-10-28T00:00:00",
                                        "2018-11-28T00:00:00",
                                        "2018-12-28T00:00:00",
                                        "2019-01-28T00:00:00",
                                        "2019-02-28T00:00:00",
                                        "2019-03-28T00:00:00",
                                        "2019-04-28T00:00:00",
                                        "2019-05-28T00:00:00"
                                    ],
                                    "xaxis": "x",
                                    "y": [
                                        134780,
                                        142460,
                                        149890,
                                        158250,
                                        166970,
                                        155400,
                                        169160,
                                        181180,
                                        188620,
                                        199450,
                                        214370,
                                        213590,
                                        223240,
                                        232640,
                                        243900,
                                        255010,
                                        265290
                                    ],
                                    "yaxis": "y"
                                },
                                {
                                    "alignmentgroup": "True",
                                    "hovertemplate": "Productos=pension_plan<br>date=%{x}<br>Ingresos=%{y}<extra></extra>",
                                    "legendgroup": "pension_plan",
                                    "marker": {
                                        "color": "#284ec8"
                                    },
                                    "name": "pension_plan",
                                    "offsetgroup": "pension_plan",
                                    "orientation": "v",
                                    "showlegend": true,
                                    "textposition": "auto",
                                    "type": "bar",
                                    "x": [
                                        "2018-01-28T00:00:00",
                                        "2018-02-28T00:00:00",
                                        "2018-03-28T00:00:00",
                                        "2018-04-28T00:00:00",
                                        "2018-05-28T00:00:00",
                                        "2018-06-28T00:00:00",
                                        "2018-07-28T00:00:00",
                                        "2018-08-28T00:00:00",
                                        "2018-09-28T00:00:00",
                                        "2018-10-28T00:00:00",
                                        "2018-11-28T00:00:00",
                                        "2018-12-28T00:00:00",
                                        "2019-01-28T00:00:00",
                                        "2019-02-28T00:00:00",
                                        "2019-03-28T00:00:00",
                                        "2019-04-28T00:00:00",
                                        "2019-05-28T00:00:00"
                                    ],
                                    "xaxis": "x",
                                    "y": [
                                        353400,
                                        379840,
                                        395280,
                                        422400,
                                        405920,
                                        465400,
                                        511160,
                                        485040,
                                        504960,
                                        528920,
                                        552720,
                                        612960,
                                        501520,
                                        610600,
                                        639040,
                                        648800,
                                        694120
                                    ],
                                    "yaxis": "y"
                                },
                                {
                                    "alignmentgroup": "True",
                                    "hovertemplate": "Productos=securities<br>date=%{x}<br>Ingresos=%{y}<extra></extra>",
                                    "legendgroup": "securities",
                                    "marker": {
                                        "color": "#2e21ea"
                                    },
                                    "name": "securities",
                                    "offsetgroup": "securities",
                                    "orientation": "v",
                                    "showlegend": true,
                                    "textposition": "auto",
                                    "type": "bar",
                                    "x": [
                                        "2018-01-28T00:00:00",
                                        "2018-02-28T00:00:00",
                                        "2018-03-28T00:00:00",
                                        "2018-04-28T00:00:00",
                                        "2018-05-28T00:00:00",
                                        "2018-06-28T00:00:00",
                                        "2018-07-28T00:00:00",
                                        "2018-08-28T00:00:00",
                                        "2018-09-28T00:00:00",
                                        "2018-10-28T00:00:00",
                                        "2018-11-28T00:00:00",
                                        "2018-12-28T00:00:00",
                                        "2019-01-28T00:00:00",
                                        "2019-02-28T00:00:00",
                                        "2019-03-28T00:00:00",
                                        "2019-04-28T00:00:00",
                                        "2019-05-28T00:00:00"
                                    ],
                                    "xaxis": "x",
                                    "y": [
                                        37280,
                                        38600,
                                        38600,
                                        39320,
                                        40480,
                                        40760,
                                        42120,
                                        45320,
                                        48040,
                                        53920,
                                        54400,
                                        56800,
                                        66200,
                                        69880,
                                        70760,
                                        71280,
                                        71560
                                    ],
                                    "yaxis": "y"
                                },
                                {
                                    "alignmentgroup": "True",
                                    "hovertemplate": "Productos=short_term_deposit<br>date=%{x}<br>Ingresos=%{y}<extra></extra>",
                                    "legendgroup": "short_term_deposit",
                                    "marker": {
                                        "color": "#6324f5"
                                    },
                                    "name": "short_term_deposit",
                                    "offsetgroup": "short_term_deposit",
                                    "orientation": "v",
                                    "showlegend": true,
                                    "textposition": "auto",
                                    "type": "bar",
                                    "x": [
                                        "2018-01-28T00:00:00",
                                        "2018-02-28T00:00:00",
                                        "2018-03-28T00:00:00",
                                        "2018-04-28T00:00:00",
                                        "2018-05-28T00:00:00",
                                        "2018-06-28T00:00:00",
                                        "2018-07-28T00:00:00",
                                        "2018-08-28T00:00:00",
                                        "2018-09-28T00:00:00",
                                        "2018-10-28T00:00:00",
                                        "2018-11-28T00:00:00",
                                        "2018-12-28T00:00:00",
                                        "2019-01-28T00:00:00",
                                        "2019-02-28T00:00:00",
                                        "2019-03-28T00:00:00",
                                        "2019-04-28T00:00:00",
                                        "2019-05-28T00:00:00"
                                    ],
                                    "xaxis": "x",
                                    "y": [
                                        35320,
                                        53720,
                                        66560,
                                        69600,
                                        58200,
                                        49760,
                                        51280,
                                        53400,
                                        55440,
                                        54960,
                                        40720,
                                        21120,
                                        5000,
                                        440,
                                        80,
                                        80,
                                        80
                                    ],
                                    "yaxis": "y"
                                }
                            ],
                            "layout": {
                                "barmode": "relative",
                                "legend": {
                                    "title": {
                                        "text": "Productos"
                                    },
                                    "tracegroupgap": 0
                                },
                                "margin": {
                                    "t": 60
                                },
                                "template": {
                                    "data": {
                                        "bar": [
                                            {
                                                "error_x": {
                                                    "color": "#2a3f5f"
                                                },
                                                "error_y": {
                                                    "color": "#2a3f5f"
                                                },
                                                "marker": {
                                                    "line": {
                                                        "color": "#E5ECF6",
                                                        "width": 0.5
                                                    }
                                                },
                                                "type": "bar"
                                            }
                                        ],
                                        "barpolar": [
                                            {
                                                "marker": {
                                                    "line": {
                                                        "color": "#E5ECF6",
                                                        "width": 0.5
                                                    }
                                                },
                                                "type": "barpolar"
                                            }
                                        ],
                                        "carpet": [
                                            {
                                                "aaxis": {
                                                    "endlinecolor": "#2a3f5f",
                                                    "gridcolor": "white",
                                                    "linecolor": "white",
                                                    "minorgridcolor": "white",
                                                    "startlinecolor": "#2a3f5f"
                                                },
                                                "baxis": {
                                                    "endlinecolor": "#2a3f5f",
                                                    "gridcolor": "white",
                                                    "linecolor": "white",
                                                    "minorgridcolor": "white",
                                                    "startlinecolor": "#2a3f5f"
                                                },
                                                "type": "carpet"
                                            }
                                        ],
                                        "choropleth": [
                                            {
                                                "colorbar": {
                                                    "outlinewidth": 0,
                                                    "ticks": ""
                                                },
                                                "type": "choropleth"
                                            }
                                        ],
                                        "contour": [
                                            {
                                                "colorbar": {
                                                    "outlinewidth": 0,
                                                    "ticks": ""
                                                },
                                                "colorscale": [
                                                    [
                                                        0,
                                                        "#0d0887"
                                                    ],
                                                    [
                                                        0.1111111111111111,
                                                        "#46039f"
                                                    ],
                                                    [
                                                        0.2222222222222222,
                                                        "#7201a8"
                                                    ],
                                                    [
                                                        0.3333333333333333,
                                                        "#9c179e"
                                                    ],
                                                    [
                                                        0.4444444444444444,
                                                        "#bd3786"
                                                    ],
                                                    [
                                                        0.5555555555555556,
                                                        "#d8576b"
                                                    ],
                                                    [
                                                        0.6666666666666666,
                                                        "#ed7953"
                                                    ],
                                                    [
                                                        0.7777777777777778,
                                                        "#fb9f3a"
                                                    ],
                                                    [
                                                        0.8888888888888888,
                                                        "#fdca26"
                                                    ],
                                                    [
                                                        1,
                                                        "#f0f921"
                                                    ]
                                                ],
                                                "type": "contour"
                                            }
                                        ],
                                        "contourcarpet": [
                                            {
                                                "colorbar": {
                                                    "outlinewidth": 0,
                                                    "ticks": ""
                                                },
                                                "type": "contourcarpet"
                                            }
                                        ],
                                        "heatmap": [
                                            {
                                                "colorbar": {
                                                    "outlinewidth": 0,
                                                    "ticks": ""
                                                },
                                                "colorscale": [
                                                    [
                                                        0,
                                                        "#0d0887"
                                                    ],
                                                    [
                                                        0.1111111111111111,
                                                        "#46039f"
                                                    ],
                                                    [
                                                        0.2222222222222222,
                                                        "#7201a8"
                                                    ],
                                                    [
                                                        0.3333333333333333,
                                                        "#9c179e"
                                                    ],
                                                    [
                                                        0.4444444444444444,
                                                        "#bd3786"
                                                    ],
                                                    [
                                                        0.5555555555555556,
                                                        "#d8576b"
                                                    ],
                                                    [
                                                        0.6666666666666666,
                                                        "#ed7953"
                                                    ],
                                                    [
                                                        0.7777777777777778,
                                                        "#fb9f3a"
                                                    ],
                                                    [
                                                        0.8888888888888888,
                                                        "#fdca26"
                                                    ],
                                                    [
                                                        1,
                                                        "#f0f921"
                                                    ]
                                                ],
                                                "type": "heatmap"
                                            }
                                        ],
                                        "heatmapgl": [
                                            {
                                                "colorbar": {
                                                    "outlinewidth": 0,
                                                    "ticks": ""
                                                },
                                                "colorscale": [
                                                    [
                                                        0,
                                                        "#0d0887"
                                                    ],
                                                    [
                                                        0.1111111111111111,
                                                        "#46039f"
                                                    ],
                                                    [
                                                        0.2222222222222222,
                                                        "#7201a8"
                                                    ],
                                                    [
                                                        0.3333333333333333,
                                                        "#9c179e"
                                                    ],
                                                    [
                                                        0.4444444444444444,
                                                        "#bd3786"
                                                    ],
                                                    [
                                                        0.5555555555555556,
                                                        "#d8576b"
                                                    ],
                                                    [
                                                        0.6666666666666666,
                                                        "#ed7953"
                                                    ],
                                                    [
                                                        0.7777777777777778,
                                                        "#fb9f3a"
                                                    ],
                                                    [
                                                        0.8888888888888888,
                                                        "#fdca26"
                                                    ],
                                                    [
                                                        1,
                                                        "#f0f921"
                                                    ]
                                                ],
                                                "type": "heatmapgl"
                                            }
                                        ],
                                        "histogram": [
                                            {
                                                "marker": {
                                                    "colorbar": {
                                                        "outlinewidth": 0,
                                                        "ticks": ""
                                                    }
                                                },
                                                "type": "histogram"
                                            }
                                        ],
                                        "histogram2d": [
                                            {
                                                "colorbar": {
                                                    "outlinewidth": 0,
                                                    "ticks": ""
                                                },
                                                "colorscale": [
                                                    [
                                                        0,
                                                        "#0d0887"
                                                    ],
                                                    [
                                                        0.1111111111111111,
                                                        "#46039f"
                                                    ],
                                                    [
                                                        0.2222222222222222,
                                                        "#7201a8"
                                                    ],
                                                    [
                                                        0.3333333333333333,
                                                        "#9c179e"
                                                    ],
                                                    [
                                                        0.4444444444444444,
                                                        "#bd3786"
                                                    ],
                                                    [
                                                        0.5555555555555556,
                                                        "#d8576b"
                                                    ],
                                                    [
                                                        0.6666666666666666,
                                                        "#ed7953"
                                                    ],
                                                    [
                                                        0.7777777777777778,
                                                        "#fb9f3a"
                                                    ],
                                                    [
                                                        0.8888888888888888,
                                                        "#fdca26"
                                                    ],
                                                    [
                                                        1,
                                                        "#f0f921"
                                                    ]
                                                ],
                                                "type": "histogram2d"
                                            }
                                        ],
                                        "histogram2dcontour": [
                                            {
                                                "colorbar": {
                                                    "outlinewidth": 0,
                                                    "ticks": ""
                                                },
                                                "colorscale": [
                                                    [
                                                        0,
                                                        "#0d0887"
                                                    ],
                                                    [
                                                        0.1111111111111111,
                                                        "#46039f"
                                                    ],
                                                    [
                                                        0.2222222222222222,
                                                        "#7201a8"
                                                    ],
                                                    [
                                                        0.3333333333333333,
                                                        "#9c179e"
                                                    ],
                                                    [
                                                        0.4444444444444444,
                                                        "#bd3786"
                                                    ],
                                                    [
                                                        0.5555555555555556,
                                                        "#d8576b"
                                                    ],
                                                    [
                                                        0.6666666666666666,
                                                        "#ed7953"
                                                    ],
                                                    [
                                                        0.7777777777777778,
                                                        "#fb9f3a"
                                                    ],
                                                    [
                                                        0.8888888888888888,
                                                        "#fdca26"
                                                    ],
                                                    [
                                                        1,
                                                        "#f0f921"
                                                    ]
                                                ],
                                                "type": "histogram2dcontour"
                                            }
                                        ],
                                        "mesh3d": [
                                            {
                                                "colorbar": {
                                                    "outlinewidth": 0,
                                                    "ticks": ""
                                                },
                                                "type": "mesh3d"
                                            }
                                        ],
                                        "parcoords": [
                                            {
                                                "line": {
                                                    "colorbar": {
                                                        "outlinewidth": 0,
                                                        "ticks": ""
                                                    }
                                                },
                                                "type": "parcoords"
                                            }
                                        ],
                                        "pie": [
                                            {
                                                "automargin": true,
                                                "type": "pie"
                                            }
                                        ],
                                        "scatter": [
                                            {
                                                "marker": {
                                                    "colorbar": {
                                                        "outlinewidth": 0,
                                                        "ticks": ""
                                                    }
                                                },
                                                "type": "scatter"
                                            }
                                        ],
                                        "scatter3d": [
                                            {
                                                "line": {
                                                    "colorbar": {
                                                        "outlinewidth": 0,
                                                        "ticks": ""
                                                    }
                                                },
                                                "marker": {
                                                    "colorbar": {
                                                        "outlinewidth": 0,
                                                        "ticks": ""
                                                    }
                                                },
                                                "type": "scatter3d"
                                            }
                                        ],
                                        "scattercarpet": [
                                            {
                                                "marker": {
                                                    "colorbar": {
                                                        "outlinewidth": 0,
                                                        "ticks": ""
                                                    }
                                                },
                                                "type": "scattercarpet"
                                            }
                                        ],
                                        "scattergeo": [
                                            {
                                                "marker": {
                                                    "colorbar": {
                                                        "outlinewidth": 0,
                                                        "ticks": ""
                                                    }
                                                },
                                                "type": "scattergeo"
                                            }
                                        ],
                                        "scattergl": [
                                            {
                                                "marker": {
                                                    "colorbar": {
                                                        "outlinewidth": 0,
                                                        "ticks": ""
                                                    }
                                                },
                                                "type": "scattergl"
                                            }
                                        ],
                                        "scattermapbox": [
                                            {
                                                "marker": {
                                                    "colorbar": {
                                                        "outlinewidth": 0,
                                                        "ticks": ""
                                                    }
                                                },
                                                "type": "scattermapbox"
                                            }
                                        ],
                                        "scatterpolar": [
                                            {
                                                "marker": {
                                                    "colorbar": {
                                                        "outlinewidth": 0,
                                                        "ticks": ""
                                                    }
                                                },
                                                "type": "scatterpolar"
                                            }
                                        ],
                                        "scatterpolargl": [
                                            {
                                                "marker": {
                                                    "colorbar": {
                                                        "outlinewidth": 0,
                                                        "ticks": ""
                                                    }
                                                },
                                                "type": "scatterpolargl"
                                            }
                                        ],
                                        "scatterternary": [
                                            {
                                                "marker": {
                                                    "colorbar": {
                                                        "outlinewidth": 0,
                                                        "ticks": ""
                                                    }
                                                },
                                                "type": "scatterternary"
                                            }
                                        ],
                                        "surface": [
                                            {
                                                "colorbar": {
                                                    "outlinewidth": 0,
                                                    "ticks": ""
                                                },
                                                "colorscale": [
                                                    [
                                                        0,
                                                        "#0d0887"
                                                    ],
                                                    [
                                                        0.1111111111111111,
                                                        "#46039f"
                                                    ],
                                                    [
                                                        0.2222222222222222,
                                                        "#7201a8"
                                                    ],
                                                    [
                                                        0.3333333333333333,
                                                        "#9c179e"
                                                    ],
                                                    [
                                                        0.4444444444444444,
                                                        "#bd3786"
                                                    ],
                                                    [
                                                        0.5555555555555556,
                                                        "#d8576b"
                                                    ],
                                                    [
                                                        0.6666666666666666,
                                                        "#ed7953"
                                                    ],
                                                    [
                                                        0.7777777777777778,
                                                        "#fb9f3a"
                                                    ],
                                                    [
                                                        0.8888888888888888,
                                                        "#fdca26"
                                                    ],
                                                    [
                                                        1,
                                                        "#f0f921"
                                                    ]
                                                ],
                                                "type": "surface"
                                            }
                                        ],
                                        "table": [
                                            {
                                                "cells": {
                                                    "fill": {
                                                        "color": "#EBF0F8"
                                                    },
                                                    "line": {
                                                        "color": "white"
                                                    }
                                                },
                                                "header": {
                                                    "fill": {
                                                        "color": "#C8D4E3"
                                                    },
                                                    "line": {
                                                        "color": "white"
                                                    }
                                                },
                                                "type": "table"
                                            }
                                        ]
                                    },
                                    "layout": {
                                        "annotationdefaults": {
                                            "arrowcolor": "#2a3f5f",
                                            "arrowhead": 0,
                                            "arrowwidth": 1
                                        },
                                        "autotypenumbers": "strict",
                                        "coloraxis": {
                                            "colorbar": {
                                                "outlinewidth": 0,
                                                "ticks": ""
                                            }
                                        },
                                        "colorscale": {
                                            "diverging": [
                                                [
                                                    0,
                                                    "#8e0152"
                                                ],
                                                [
                                                    0.1,
                                                    "#c51b7d"
                                                ],
                                                [
                                                    0.2,
                                                    "#de77ae"
                                                ],
                                                [
                                                    0.3,
                                                    "#f1b6da"
                                                ],
                                                [
                                                    0.4,
                                                    "#fde0ef"
                                                ],
                                                [
                                                    0.5,
                                                    "#f7f7f7"
                                                ],
                                                [
                                                    0.6,
                                                    "#e6f5d0"
                                                ],
                                                [
                                                    0.7,
                                                    "#b8e186"
                                                ],
                                                [
                                                    0.8,
                                                    "#7fbc41"
                                                ],
                                                [
                                                    0.9,
                                                    "#4d9221"
                                                ],
                                                [
                                                    1,
                                                    "#276419"
                                                ]
                                            ],
                                            "sequential": [
                                                [
                                                    0,
                                                    "#0d0887"
                                                ],
                                                [
                                                    0.1111111111111111,
                                                    "#46039f"
                                                ],
                                                [
                                                    0.2222222222222222,
                                                    "#7201a8"
                                                ],
                                                [
                                                    0.3333333333333333,
                                                    "#9c179e"
                                                ],
                                                [
                                                    0.4444444444444444,
                                                    "#bd3786"
                                                ],
                                                [
                                                    0.5555555555555556,
                                                    "#d8576b"
                                                ],
                                                [
                                                    0.6666666666666666,
                                                    "#ed7953"
                                                ],
                                                [
                                                    0.7777777777777778,
                                                    "#fb9f3a"
                                                ],
                                                [
                                                    0.8888888888888888,
                                                    "#fdca26"
                                                ],
                                                [
                                                    1,
                                                    "#f0f921"
                                                ]
                                            ],
                                            "sequentialminus": [
                                                [
                                                    0,
                                                    "#0d0887"
                                                ],
                                                [
                                                    0.1111111111111111,
                                                    "#46039f"
                                                ],
                                                [
                                                    0.2222222222222222,
                                                    "#7201a8"
                                                ],
                                                [
                                                    0.3333333333333333,
                                                    "#9c179e"
                                                ],
                                                [
                                                    0.4444444444444444,
                                                    "#bd3786"
                                                ],
                                                [
                                                    0.5555555555555556,
                                                    "#d8576b"
                                                ],
                                                [
                                                    0.6666666666666666,
                                                    "#ed7953"
                                                ],
                                                [
                                                    0.7777777777777778,
                                                    "#fb9f3a"
                                                ],
                                                [
                                                    0.8888888888888888,
                                                    "#fdca26"
                                                ],
                                                [
                                                    1,
                                                    "#f0f921"
                                                ]
                                            ]
                                        },
                                        "colorway": [
                                            "#636efa",
                                            "#EF553B",
                                            "#00cc96",
                                            "#ab63fa",
                                            "#FFA15A",
                                            "#19d3f3",
                                            "#FF6692",
                                            "#B6E880",
                                            "#FF97FF",
                                            "#FECB52"
                                        ],
                                        "font": {
                                            "color": "#2a3f5f"
                                        },
                                        "geo": {
                                            "bgcolor": "white",
                                            "lakecolor": "white",
                                            "landcolor": "#E5ECF6",
                                            "showlakes": true,
                                            "showland": true,
                                            "subunitcolor": "white"
                                        },
                                        "hoverlabel": {
                                            "align": "left"
                                        },
                                        "hovermode": "closest",
                                        "mapbox": {
                                            "style": "light"
                                        },
                                        "paper_bgcolor": "white",
                                        "plot_bgcolor": "#E5ECF6",
                                        "polar": {
                                            "angularaxis": {
                                                "gridcolor": "white",
                                                "linecolor": "white",
                                                "ticks": ""
                                            },
                                            "bgcolor": "#E5ECF6",
                                            "radialaxis": {
                                                "gridcolor": "white",
                                                "linecolor": "white",
                                                "ticks": ""
                                            }
                                        },
                                        "scene": {
                                            "xaxis": {
                                                "backgroundcolor": "#E5ECF6",
                                                "gridcolor": "white",
                                                "gridwidth": 2,
                                                "linecolor": "white",
                                                "showbackground": true,
                                                "ticks": "",
                                                "zerolinecolor": "white"
                                            },
                                            "yaxis": {
                                                "backgroundcolor": "#E5ECF6",
                                                "gridcolor": "white",
                                                "gridwidth": 2,
                                                "linecolor": "white",
                                                "showbackground": true,
                                                "ticks": "",
                                                "zerolinecolor": "white"
                                            },
                                            "zaxis": {
                                                "backgroundcolor": "#E5ECF6",
                                                "gridcolor": "white",
                                                "gridwidth": 2,
                                                "linecolor": "white",
                                                "showbackground": true,
                                                "ticks": "",
                                                "zerolinecolor": "white"
                                            }
                                        },
                                        "shapedefaults": {
                                            "line": {
                                                "color": "#2a3f5f"
                                            }
                                        },
                                        "ternary": {
                                            "aaxis": {
                                                "gridcolor": "white",
                                                "linecolor": "white",
                                                "ticks": ""
                                            },
                                            "baxis": {
                                                "gridcolor": "white",
                                                "linecolor": "white",
                                                "ticks": ""
                                            },
                                            "bgcolor": "#E5ECF6",
                                            "caxis": {
                                                "gridcolor": "white",
                                                "linecolor": "white",
                                                "ticks": ""
                                            }
                                        },
                                        "title": {
                                            "x": 0.05
                                        },
                                        "xaxis": {
                                            "automargin": true,
                                            "gridcolor": "white",
                                            "linecolor": "white",
                                            "ticks": "",
                                            "title": {
                                                "standoff": 15
                                            },
                                            "zerolinecolor": "white",
                                            "zerolinewidth": 2
                                        },
                                        "yaxis": {
                                            "automargin": true,
                                            "gridcolor": "white",
                                            "linecolor": "white",
                                            "ticks": "",
                                            "title": {
                                                "standoff": 15
                                            },
                                            "zerolinecolor": "white",
                                            "zerolinewidth": 2
                                        }
                                    }
                                },
                                "xaxis": {
                                    "anchor": "y",
                                    "domain": [
                                        0,
                                        1
                                    ],
                                    "title": {
                                        "text": "date"
                                    }
                                },
                                "yaxis": {
                                    "anchor": "x",
                                    "domain": [
                                        0,
                                        1
                                    ],
                                    "title": {
                                        "text": "Ingresos"
                                    }
                                }
                            }
                        }
                    },
                    "metadata": {}
                }
            ],
            "execution_count": 36
        },
        {
            "cell_type": "code",
            "execution_count": 37,
            "metadata": {},
            "outputs": [
                {
                    "output_type": "display_data",
                    "data": {
                        "application/vnd.plotly.v1+json": {
                            "config": {
                                "plotlyServerURL": "https://plot.ly"
                            },
                            "data": [
                                {
                                    "alignmentgroup": "True",
                                    "hovertemplate": "Familia_prod=AhorroVista<br>date=%{x}<br>Ingresos=%{y}<extra></extra>",
                                    "legendgroup": "AhorroVista",
                                    "marker": {
                                        "color": "#ef55f1"
                                    },
                                    "name": "AhorroVista",
                                    "offsetgroup": "AhorroVista",
                                    "orientation": "v",
                                    "showlegend": true,
                                    "textposition": "auto",
                                    "type": "bar",
                                    "x": [
                                        "2018-01-28T00:00:00",
                                        "2018-02-28T00:00:00",
                                        "2018-03-28T00:00:00",
                                        "2018-04-28T00:00:00",
                                        "2018-05-28T00:00:00",
                                        "2018-06-28T00:00:00",
                                        "2018-07-28T00:00:00",
                                        "2018-08-28T00:00:00",
                                        "2018-09-28T00:00:00",
                                        "2018-10-28T00:00:00",
                                        "2018-11-28T00:00:00",
                                        "2018-12-28T00:00:00",
                                        "2019-01-28T00:00:00",
                                        "2019-02-28T00:00:00",
                                        "2019-03-28T00:00:00",
                                        "2019-04-28T00:00:00",
                                        "2019-05-28T00:00:00"
                                    ],
                                    "xaxis": "x",
                                    "y": [
                                        2769340,
                                        2815100,
                                        2868830,
                                        2902590,
                                        2934640,
                                        2975940,
                                        3108930,
                                        3230210,
                                        3422260,
                                        3633240,
                                        3743310,
                                        3822140,
                                        3844410,
                                        3918270,
                                        3984220,
                                        4021730,
                                        4072560
                                    ],
                                    "yaxis": "y"
                                },
                                {
                                    "alignmentgroup": "True",
                                    "hovertemplate": "Familia_prod=Crédito<br>date=%{x}<br>Ingresos=%{y}<extra></extra>",
                                    "legendgroup": "Crédito",
                                    "marker": {
                                        "color": "#fb84ce"
                                    },
                                    "name": "Crédito",
                                    "offsetgroup": "Crédito",
                                    "orientation": "v",
                                    "showlegend": true,
                                    "textposition": "auto",
                                    "type": "bar",
                                    "x": [
                                        "2018-01-28T00:00:00",
                                        "2018-02-28T00:00:00",
                                        "2018-03-28T00:00:00",
                                        "2018-04-28T00:00:00",
                                        "2018-05-28T00:00:00",
                                        "2018-06-28T00:00:00",
                                        "2018-07-28T00:00:00",
                                        "2018-08-28T00:00:00",
                                        "2018-09-28T00:00:00",
                                        "2018-10-28T00:00:00",
                                        "2018-11-28T00:00:00",
                                        "2018-12-28T00:00:00",
                                        "2019-01-28T00:00:00",
                                        "2019-02-28T00:00:00",
                                        "2019-03-28T00:00:00",
                                        "2019-04-28T00:00:00",
                                        "2019-05-28T00:00:00"
                                    ],
                                    "xaxis": "x",
                                    "y": [
                                        201540,
                                        198960,
                                        213060,
                                        225960,
                                        230520,
                                        242880,
                                        251100,
                                        255300,
                                        260400,
                                        266220,
                                        273780,
                                        275100,
                                        272460,
                                        275460,
                                        277500,
                                        288000,
                                        291240
                                    ],
                                    "yaxis": "y"
                                },
                                {
                                    "alignmentgroup": "True",
                                    "hovertemplate": "Familia_prod=Inversión<br>date=%{x}<br>Ingresos=%{y}<extra></extra>",
                                    "legendgroup": "Inversión",
                                    "marker": {
                                        "color": "#fbafa1"
                                    },
                                    "name": "Inversión",
                                    "offsetgroup": "Inversión",
                                    "orientation": "v",
                                    "showlegend": true,
                                    "textposition": "auto",
                                    "type": "bar",
                                    "x": [
                                        "2018-01-28T00:00:00",
                                        "2018-02-28T00:00:00",
                                        "2018-03-28T00:00:00",
                                        "2018-04-28T00:00:00",
                                        "2018-05-28T00:00:00",
                                        "2018-06-28T00:00:00",
                                        "2018-07-28T00:00:00",
                                        "2018-08-28T00:00:00",
                                        "2018-09-28T00:00:00",
                                        "2018-10-28T00:00:00",
                                        "2018-11-28T00:00:00",
                                        "2018-12-28T00:00:00",
                                        "2019-01-28T00:00:00",
                                        "2019-02-28T00:00:00",
                                        "2019-03-28T00:00:00",
                                        "2019-04-28T00:00:00",
                                        "2019-05-28T00:00:00"
                                    ],
                                    "xaxis": "x",
                                    "y": [
                                        652800,
                                        705280,
                                        737800,
                                        778960,
                                        764640,
                                        825800,
                                        883280,
                                        869440,
                                        904240,
                                        942960,
                                        957000,
                                        1013720,
                                        892280,
                                        999880,
                                        1024240,
                                        1027680,
                                        1063520
                                    ],
                                    "yaxis": "y"
                                }
                            ],
                            "layout": {
                                "barmode": "relative",
                                "legend": {
                                    "title": {
                                        "text": "Familia_prod"
                                    },
                                    "tracegroupgap": 0
                                },
                                "margin": {
                                    "t": 60
                                },
                                "template": {
                                    "data": {
                                        "bar": [
                                            {
                                                "error_x": {
                                                    "color": "#2a3f5f"
                                                },
                                                "error_y": {
                                                    "color": "#2a3f5f"
                                                },
                                                "marker": {
                                                    "line": {
                                                        "color": "#E5ECF6",
                                                        "width": 0.5
                                                    }
                                                },
                                                "type": "bar"
                                            }
                                        ],
                                        "barpolar": [
                                            {
                                                "marker": {
                                                    "line": {
                                                        "color": "#E5ECF6",
                                                        "width": 0.5
                                                    }
                                                },
                                                "type": "barpolar"
                                            }
                                        ],
                                        "carpet": [
                                            {
                                                "aaxis": {
                                                    "endlinecolor": "#2a3f5f",
                                                    "gridcolor": "white",
                                                    "linecolor": "white",
                                                    "minorgridcolor": "white",
                                                    "startlinecolor": "#2a3f5f"
                                                },
                                                "baxis": {
                                                    "endlinecolor": "#2a3f5f",
                                                    "gridcolor": "white",
                                                    "linecolor": "white",
                                                    "minorgridcolor": "white",
                                                    "startlinecolor": "#2a3f5f"
                                                },
                                                "type": "carpet"
                                            }
                                        ],
                                        "choropleth": [
                                            {
                                                "colorbar": {
                                                    "outlinewidth": 0,
                                                    "ticks": ""
                                                },
                                                "type": "choropleth"
                                            }
                                        ],
                                        "contour": [
                                            {
                                                "colorbar": {
                                                    "outlinewidth": 0,
                                                    "ticks": ""
                                                },
                                                "colorscale": [
                                                    [
                                                        0,
                                                        "#0d0887"
                                                    ],
                                                    [
                                                        0.1111111111111111,
                                                        "#46039f"
                                                    ],
                                                    [
                                                        0.2222222222222222,
                                                        "#7201a8"
                                                    ],
                                                    [
                                                        0.3333333333333333,
                                                        "#9c179e"
                                                    ],
                                                    [
                                                        0.4444444444444444,
                                                        "#bd3786"
                                                    ],
                                                    [
                                                        0.5555555555555556,
                                                        "#d8576b"
                                                    ],
                                                    [
                                                        0.6666666666666666,
                                                        "#ed7953"
                                                    ],
                                                    [
                                                        0.7777777777777778,
                                                        "#fb9f3a"
                                                    ],
                                                    [
                                                        0.8888888888888888,
                                                        "#fdca26"
                                                    ],
                                                    [
                                                        1,
                                                        "#f0f921"
                                                    ]
                                                ],
                                                "type": "contour"
                                            }
                                        ],
                                        "contourcarpet": [
                                            {
                                                "colorbar": {
                                                    "outlinewidth": 0,
                                                    "ticks": ""
                                                },
                                                "type": "contourcarpet"
                                            }
                                        ],
                                        "heatmap": [
                                            {
                                                "colorbar": {
                                                    "outlinewidth": 0,
                                                    "ticks": ""
                                                },
                                                "colorscale": [
                                                    [
                                                        0,
                                                        "#0d0887"
                                                    ],
                                                    [
                                                        0.1111111111111111,
                                                        "#46039f"
                                                    ],
                                                    [
                                                        0.2222222222222222,
                                                        "#7201a8"
                                                    ],
                                                    [
                                                        0.3333333333333333,
                                                        "#9c179e"
                                                    ],
                                                    [
                                                        0.4444444444444444,
                                                        "#bd3786"
                                                    ],
                                                    [
                                                        0.5555555555555556,
                                                        "#d8576b"
                                                    ],
                                                    [
                                                        0.6666666666666666,
                                                        "#ed7953"
                                                    ],
                                                    [
                                                        0.7777777777777778,
                                                        "#fb9f3a"
                                                    ],
                                                    [
                                                        0.8888888888888888,
                                                        "#fdca26"
                                                    ],
                                                    [
                                                        1,
                                                        "#f0f921"
                                                    ]
                                                ],
                                                "type": "heatmap"
                                            }
                                        ],
                                        "heatmapgl": [
                                            {
                                                "colorbar": {
                                                    "outlinewidth": 0,
                                                    "ticks": ""
                                                },
                                                "colorscale": [
                                                    [
                                                        0,
                                                        "#0d0887"
                                                    ],
                                                    [
                                                        0.1111111111111111,
                                                        "#46039f"
                                                    ],
                                                    [
                                                        0.2222222222222222,
                                                        "#7201a8"
                                                    ],
                                                    [
                                                        0.3333333333333333,
                                                        "#9c179e"
                                                    ],
                                                    [
                                                        0.4444444444444444,
                                                        "#bd3786"
                                                    ],
                                                    [
                                                        0.5555555555555556,
                                                        "#d8576b"
                                                    ],
                                                    [
                                                        0.6666666666666666,
                                                        "#ed7953"
                                                    ],
                                                    [
                                                        0.7777777777777778,
                                                        "#fb9f3a"
                                                    ],
                                                    [
                                                        0.8888888888888888,
                                                        "#fdca26"
                                                    ],
                                                    [
                                                        1,
                                                        "#f0f921"
                                                    ]
                                                ],
                                                "type": "heatmapgl"
                                            }
                                        ],
                                        "histogram": [
                                            {
                                                "marker": {
                                                    "colorbar": {
                                                        "outlinewidth": 0,
                                                        "ticks": ""
                                                    }
                                                },
                                                "type": "histogram"
                                            }
                                        ],
                                        "histogram2d": [
                                            {
                                                "colorbar": {
                                                    "outlinewidth": 0,
                                                    "ticks": ""
                                                },
                                                "colorscale": [
                                                    [
                                                        0,
                                                        "#0d0887"
                                                    ],
                                                    [
                                                        0.1111111111111111,
                                                        "#46039f"
                                                    ],
                                                    [
                                                        0.2222222222222222,
                                                        "#7201a8"
                                                    ],
                                                    [
                                                        0.3333333333333333,
                                                        "#9c179e"
                                                    ],
                                                    [
                                                        0.4444444444444444,
                                                        "#bd3786"
                                                    ],
                                                    [
                                                        0.5555555555555556,
                                                        "#d8576b"
                                                    ],
                                                    [
                                                        0.6666666666666666,
                                                        "#ed7953"
                                                    ],
                                                    [
                                                        0.7777777777777778,
                                                        "#fb9f3a"
                                                    ],
                                                    [
                                                        0.8888888888888888,
                                                        "#fdca26"
                                                    ],
                                                    [
                                                        1,
                                                        "#f0f921"
                                                    ]
                                                ],
                                                "type": "histogram2d"
                                            }
                                        ],
                                        "histogram2dcontour": [
                                            {
                                                "colorbar": {
                                                    "outlinewidth": 0,
                                                    "ticks": ""
                                                },
                                                "colorscale": [
                                                    [
                                                        0,
                                                        "#0d0887"
                                                    ],
                                                    [
                                                        0.1111111111111111,
                                                        "#46039f"
                                                    ],
                                                    [
                                                        0.2222222222222222,
                                                        "#7201a8"
                                                    ],
                                                    [
                                                        0.3333333333333333,
                                                        "#9c179e"
                                                    ],
                                                    [
                                                        0.4444444444444444,
                                                        "#bd3786"
                                                    ],
                                                    [
                                                        0.5555555555555556,
                                                        "#d8576b"
                                                    ],
                                                    [
                                                        0.6666666666666666,
                                                        "#ed7953"
                                                    ],
                                                    [
                                                        0.7777777777777778,
                                                        "#fb9f3a"
                                                    ],
                                                    [
                                                        0.8888888888888888,
                                                        "#fdca26"
                                                    ],
                                                    [
                                                        1,
                                                        "#f0f921"
                                                    ]
                                                ],
                                                "type": "histogram2dcontour"
                                            }
                                        ],
                                        "mesh3d": [
                                            {
                                                "colorbar": {
                                                    "outlinewidth": 0,
                                                    "ticks": ""
                                                },
                                                "type": "mesh3d"
                                            }
                                        ],
                                        "parcoords": [
                                            {
                                                "line": {
                                                    "colorbar": {
                                                        "outlinewidth": 0,
                                                        "ticks": ""
                                                    }
                                                },
                                                "type": "parcoords"
                                            }
                                        ],
                                        "pie": [
                                            {
                                                "automargin": true,
                                                "type": "pie"
                                            }
                                        ],
                                        "scatter": [
                                            {
                                                "marker": {
                                                    "colorbar": {
                                                        "outlinewidth": 0,
                                                        "ticks": ""
                                                    }
                                                },
                                                "type": "scatter"
                                            }
                                        ],
                                        "scatter3d": [
                                            {
                                                "line": {
                                                    "colorbar": {
                                                        "outlinewidth": 0,
                                                        "ticks": ""
                                                    }
                                                },
                                                "marker": {
                                                    "colorbar": {
                                                        "outlinewidth": 0,
                                                        "ticks": ""
                                                    }
                                                },
                                                "type": "scatter3d"
                                            }
                                        ],
                                        "scattercarpet": [
                                            {
                                                "marker": {
                                                    "colorbar": {
                                                        "outlinewidth": 0,
                                                        "ticks": ""
                                                    }
                                                },
                                                "type": "scattercarpet"
                                            }
                                        ],
                                        "scattergeo": [
                                            {
                                                "marker": {
                                                    "colorbar": {
                                                        "outlinewidth": 0,
                                                        "ticks": ""
                                                    }
                                                },
                                                "type": "scattergeo"
                                            }
                                        ],
                                        "scattergl": [
                                            {
                                                "marker": {
                                                    "colorbar": {
                                                        "outlinewidth": 0,
                                                        "ticks": ""
                                                    }
                                                },
                                                "type": "scattergl"
                                            }
                                        ],
                                        "scattermapbox": [
                                            {
                                                "marker": {
                                                    "colorbar": {
                                                        "outlinewidth": 0,
                                                        "ticks": ""
                                                    }
                                                },
                                                "type": "scattermapbox"
                                            }
                                        ],
                                        "scatterpolar": [
                                            {
                                                "marker": {
                                                    "colorbar": {
                                                        "outlinewidth": 0,
                                                        "ticks": ""
                                                    }
                                                },
                                                "type": "scatterpolar"
                                            }
                                        ],
                                        "scatterpolargl": [
                                            {
                                                "marker": {
                                                    "colorbar": {
                                                        "outlinewidth": 0,
                                                        "ticks": ""
                                                    }
                                                },
                                                "type": "scatterpolargl"
                                            }
                                        ],
                                        "scatterternary": [
                                            {
                                                "marker": {
                                                    "colorbar": {
                                                        "outlinewidth": 0,
                                                        "ticks": ""
                                                    }
                                                },
                                                "type": "scatterternary"
                                            }
                                        ],
                                        "surface": [
                                            {
                                                "colorbar": {
                                                    "outlinewidth": 0,
                                                    "ticks": ""
                                                },
                                                "colorscale": [
                                                    [
                                                        0,
                                                        "#0d0887"
                                                    ],
                                                    [
                                                        0.1111111111111111,
                                                        "#46039f"
                                                    ],
                                                    [
                                                        0.2222222222222222,
                                                        "#7201a8"
                                                    ],
                                                    [
                                                        0.3333333333333333,
                                                        "#9c179e"
                                                    ],
                                                    [
                                                        0.4444444444444444,
                                                        "#bd3786"
                                                    ],
                                                    [
                                                        0.5555555555555556,
                                                        "#d8576b"
                                                    ],
                                                    [
                                                        0.6666666666666666,
                                                        "#ed7953"
                                                    ],
                                                    [
                                                        0.7777777777777778,
                                                        "#fb9f3a"
                                                    ],
                                                    [
                                                        0.8888888888888888,
                                                        "#fdca26"
                                                    ],
                                                    [
                                                        1,
                                                        "#f0f921"
                                                    ]
                                                ],
                                                "type": "surface"
                                            }
                                        ],
                                        "table": [
                                            {
                                                "cells": {
                                                    "fill": {
                                                        "color": "#EBF0F8"
                                                    },
                                                    "line": {
                                                        "color": "white"
                                                    }
                                                },
                                                "header": {
                                                    "fill": {
                                                        "color": "#C8D4E3"
                                                    },
                                                    "line": {
                                                        "color": "white"
                                                    }
                                                },
                                                "type": "table"
                                            }
                                        ]
                                    },
                                    "layout": {
                                        "annotationdefaults": {
                                            "arrowcolor": "#2a3f5f",
                                            "arrowhead": 0,
                                            "arrowwidth": 1
                                        },
                                        "autotypenumbers": "strict",
                                        "coloraxis": {
                                            "colorbar": {
                                                "outlinewidth": 0,
                                                "ticks": ""
                                            }
                                        },
                                        "colorscale": {
                                            "diverging": [
                                                [
                                                    0,
                                                    "#8e0152"
                                                ],
                                                [
                                                    0.1,
                                                    "#c51b7d"
                                                ],
                                                [
                                                    0.2,
                                                    "#de77ae"
                                                ],
                                                [
                                                    0.3,
                                                    "#f1b6da"
                                                ],
                                                [
                                                    0.4,
                                                    "#fde0ef"
                                                ],
                                                [
                                                    0.5,
                                                    "#f7f7f7"
                                                ],
                                                [
                                                    0.6,
                                                    "#e6f5d0"
                                                ],
                                                [
                                                    0.7,
                                                    "#b8e186"
                                                ],
                                                [
                                                    0.8,
                                                    "#7fbc41"
                                                ],
                                                [
                                                    0.9,
                                                    "#4d9221"
                                                ],
                                                [
                                                    1,
                                                    "#276419"
                                                ]
                                            ],
                                            "sequential": [
                                                [
                                                    0,
                                                    "#0d0887"
                                                ],
                                                [
                                                    0.1111111111111111,
                                                    "#46039f"
                                                ],
                                                [
                                                    0.2222222222222222,
                                                    "#7201a8"
                                                ],
                                                [
                                                    0.3333333333333333,
                                                    "#9c179e"
                                                ],
                                                [
                                                    0.4444444444444444,
                                                    "#bd3786"
                                                ],
                                                [
                                                    0.5555555555555556,
                                                    "#d8576b"
                                                ],
                                                [
                                                    0.6666666666666666,
                                                    "#ed7953"
                                                ],
                                                [
                                                    0.7777777777777778,
                                                    "#fb9f3a"
                                                ],
                                                [
                                                    0.8888888888888888,
                                                    "#fdca26"
                                                ],
                                                [
                                                    1,
                                                    "#f0f921"
                                                ]
                                            ],
                                            "sequentialminus": [
                                                [
                                                    0,
                                                    "#0d0887"
                                                ],
                                                [
                                                    0.1111111111111111,
                                                    "#46039f"
                                                ],
                                                [
                                                    0.2222222222222222,
                                                    "#7201a8"
                                                ],
                                                [
                                                    0.3333333333333333,
                                                    "#9c179e"
                                                ],
                                                [
                                                    0.4444444444444444,
                                                    "#bd3786"
                                                ],
                                                [
                                                    0.5555555555555556,
                                                    "#d8576b"
                                                ],
                                                [
                                                    0.6666666666666666,
                                                    "#ed7953"
                                                ],
                                                [
                                                    0.7777777777777778,
                                                    "#fb9f3a"
                                                ],
                                                [
                                                    0.8888888888888888,
                                                    "#fdca26"
                                                ],
                                                [
                                                    1,
                                                    "#f0f921"
                                                ]
                                            ]
                                        },
                                        "colorway": [
                                            "#636efa",
                                            "#EF553B",
                                            "#00cc96",
                                            "#ab63fa",
                                            "#FFA15A",
                                            "#19d3f3",
                                            "#FF6692",
                                            "#B6E880",
                                            "#FF97FF",
                                            "#FECB52"
                                        ],
                                        "font": {
                                            "color": "#2a3f5f"
                                        },
                                        "geo": {
                                            "bgcolor": "white",
                                            "lakecolor": "white",
                                            "landcolor": "#E5ECF6",
                                            "showlakes": true,
                                            "showland": true,
                                            "subunitcolor": "white"
                                        },
                                        "hoverlabel": {
                                            "align": "left"
                                        },
                                        "hovermode": "closest",
                                        "mapbox": {
                                            "style": "light"
                                        },
                                        "paper_bgcolor": "white",
                                        "plot_bgcolor": "#E5ECF6",
                                        "polar": {
                                            "angularaxis": {
                                                "gridcolor": "white",
                                                "linecolor": "white",
                                                "ticks": ""
                                            },
                                            "bgcolor": "#E5ECF6",
                                            "radialaxis": {
                                                "gridcolor": "white",
                                                "linecolor": "white",
                                                "ticks": ""
                                            }
                                        },
                                        "scene": {
                                            "xaxis": {
                                                "backgroundcolor": "#E5ECF6",
                                                "gridcolor": "white",
                                                "gridwidth": 2,
                                                "linecolor": "white",
                                                "showbackground": true,
                                                "ticks": "",
                                                "zerolinecolor": "white"
                                            },
                                            "yaxis": {
                                                "backgroundcolor": "#E5ECF6",
                                                "gridcolor": "white",
                                                "gridwidth": 2,
                                                "linecolor": "white",
                                                "showbackground": true,
                                                "ticks": "",
                                                "zerolinecolor": "white"
                                            },
                                            "zaxis": {
                                                "backgroundcolor": "#E5ECF6",
                                                "gridcolor": "white",
                                                "gridwidth": 2,
                                                "linecolor": "white",
                                                "showbackground": true,
                                                "ticks": "",
                                                "zerolinecolor": "white"
                                            }
                                        },
                                        "shapedefaults": {
                                            "line": {
                                                "color": "#2a3f5f"
                                            }
                                        },
                                        "ternary": {
                                            "aaxis": {
                                                "gridcolor": "white",
                                                "linecolor": "white",
                                                "ticks": ""
                                            },
                                            "baxis": {
                                                "gridcolor": "white",
                                                "linecolor": "white",
                                                "ticks": ""
                                            },
                                            "bgcolor": "#E5ECF6",
                                            "caxis": {
                                                "gridcolor": "white",
                                                "linecolor": "white",
                                                "ticks": ""
                                            }
                                        },
                                        "title": {
                                            "x": 0.05
                                        },
                                        "xaxis": {
                                            "automargin": true,
                                            "gridcolor": "white",
                                            "linecolor": "white",
                                            "ticks": "",
                                            "title": {
                                                "standoff": 15
                                            },
                                            "zerolinecolor": "white",
                                            "zerolinewidth": 2
                                        },
                                        "yaxis": {
                                            "automargin": true,
                                            "gridcolor": "white",
                                            "linecolor": "white",
                                            "ticks": "",
                                            "title": {
                                                "standoff": 15
                                            },
                                            "zerolinecolor": "white",
                                            "zerolinewidth": 2
                                        }
                                    }
                                },
                                "xaxis": {
                                    "anchor": "y",
                                    "domain": [
                                        0,
                                        1
                                    ],
                                    "title": {
                                        "text": "date"
                                    }
                                },
                                "yaxis": {
                                    "anchor": "x",
                                    "domain": [
                                        0,
                                        1
                                    ],
                                    "title": {
                                        "text": "Ingresos"
                                    }
                                }
                            }
                        }
                    },
                    "metadata": {}
                }
            ],
            "source": [
                "grafico_horizontal= full_3.groupby(['Familia_prod',\"date\"])[\"Ingresos\"].sum().reset_index()\n",
                "grafico_horizontal\n",
                "\n",
                "evolucion_horizontal= px.bar(grafico_horizontal, x=\"date\", y=\"Ingresos\", \\\n",
                "                        color=\"Familia_prod\", orientation=\"v\", color_discrete_sequence=px.colors.cyclical.mygbm)\n",
                "\n",
                "evolucion_horizontal.show()"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": 38,
            "metadata": {},
            "outputs": [
                {
                    "output_type": "display_data",
                    "data": {
                        "application/vnd.plotly.v1+json": {
                            "config": {
                                "plotlyServerURL": "https://plot.ly"
                            },
                            "data": [
                                {
                                    "alignmentgroup": "True",
                                    "hovertemplate": "entry_channel=<br>date=%{x}<br>Ingresos=%{y}<extra></extra>",
                                    "legendgroup": "",
                                    "marker": {
                                        "color": "#ef55f1"
                                    },
                                    "name": "",
                                    "offsetgroup": "",
                                    "orientation": "v",
                                    "showlegend": false,
                                    "textposition": "auto",
                                    "type": "bar",
                                    "x": [
                                        "2018-01-28T00:00:00",
                                        "2018-02-28T00:00:00",
                                        "2018-03-28T00:00:00",
                                        "2018-04-28T00:00:00",
                                        "2018-05-28T00:00:00",
                                        "2018-06-28T00:00:00",
                                        "2018-07-28T00:00:00",
                                        "2018-08-28T00:00:00",
                                        "2018-09-28T00:00:00",
                                        "2018-10-28T00:00:00",
                                        "2018-11-28T00:00:00",
                                        "2018-12-28T00:00:00",
                                        "2019-01-28T00:00:00",
                                        "2019-02-28T00:00:00",
                                        "2019-03-28T00:00:00",
                                        "2019-04-28T00:00:00",
                                        "2019-05-28T00:00:00"
                                    ],
                                    "xaxis": "x",
                                    "y": [
                                        300,
                                        270,
                                        270,
                                        210,
                                        150,
                                        60,
                                        123910,
                                        116350,
                                        168360,
                                        168690,
                                        88120,
                                        36670,
                                        39910,
                                        28720,
                                        24770,
                                        19790,
                                        17670
                                    ],
                                    "yaxis": "y"
                                },
                                {
                                    "alignmentgroup": "True",
                                    "hovertemplate": "entry_channel=004<br>date=%{x}<br>Ingresos=%{y}<extra></extra>",
                                    "legendgroup": "004",
                                    "marker": {
                                        "color": "#fb84ce"
                                    },
                                    "name": "004",
                                    "offsetgroup": "004",
                                    "orientation": "v",
                                    "showlegend": true,
                                    "textposition": "auto",
                                    "type": "bar",
                                    "x": [
                                        "2018-01-28T00:00:00",
                                        "2018-02-28T00:00:00",
                                        "2018-03-28T00:00:00",
                                        "2018-04-28T00:00:00",
                                        "2018-05-28T00:00:00",
                                        "2018-06-28T00:00:00",
                                        "2018-07-28T00:00:00",
                                        "2018-08-28T00:00:00",
                                        "2018-09-28T00:00:00",
                                        "2018-10-28T00:00:00",
                                        "2018-11-28T00:00:00",
                                        "2018-12-28T00:00:00",
                                        "2019-01-28T00:00:00",
                                        "2019-02-28T00:00:00",
                                        "2019-03-28T00:00:00",
                                        "2019-04-28T00:00:00",
                                        "2019-05-28T00:00:00"
                                    ],
                                    "xaxis": "x",
                                    "y": [
                                        100,
                                        100,
                                        60,
                                        100,
                                        60,
                                        100,
                                        100,
                                        100,
                                        100,
                                        100,
                                        100,
                                        100,
                                        60,
                                        100,
                                        100,
                                        100,
                                        100
                                    ],
                                    "yaxis": "y"
                                },
                                {
                                    "alignmentgroup": "True",
                                    "hovertemplate": "entry_channel=007<br>date=%{x}<br>Ingresos=%{y}<extra></extra>",
                                    "legendgroup": "007",
                                    "marker": {
                                        "color": "#fbafa1"
                                    },
                                    "name": "007",
                                    "offsetgroup": "007",
                                    "orientation": "v",
                                    "showlegend": true,
                                    "textposition": "auto",
                                    "type": "bar",
                                    "x": [
                                        "2018-01-28T00:00:00",
                                        "2018-02-28T00:00:00",
                                        "2018-03-28T00:00:00",
                                        "2018-04-28T00:00:00",
                                        "2018-05-28T00:00:00",
                                        "2018-06-28T00:00:00",
                                        "2018-07-28T00:00:00",
                                        "2018-08-28T00:00:00",
                                        "2018-09-28T00:00:00",
                                        "2018-10-28T00:00:00",
                                        "2018-11-28T00:00:00",
                                        "2018-12-28T00:00:00",
                                        "2019-01-28T00:00:00",
                                        "2019-02-28T00:00:00",
                                        "2019-03-28T00:00:00",
                                        "2019-04-28T00:00:00",
                                        "2019-05-28T00:00:00"
                                    ],
                                    "xaxis": "x",
                                    "y": [
                                        1620,
                                        1610,
                                        1710,
                                        1770,
                                        1680,
                                        1750,
                                        1790,
                                        1890,
                                        2120,
                                        2080,
                                        2270,
                                        2180,
                                        2150,
                                        2200,
                                        2130,
                                        2220,
                                        2270
                                    ],
                                    "yaxis": "y"
                                },
                                {
                                    "alignmentgroup": "True",
                                    "hovertemplate": "entry_channel=013<br>date=%{x}<br>Ingresos=%{y}<extra></extra>",
                                    "legendgroup": "013",
                                    "marker": {
                                        "color": "#fcd471"
                                    },
                                    "name": "013",
                                    "offsetgroup": "013",
                                    "orientation": "v",
                                    "showlegend": true,
                                    "textposition": "auto",
                                    "type": "bar",
                                    "x": [
                                        "2018-01-28T00:00:00",
                                        "2018-02-28T00:00:00",
                                        "2018-03-28T00:00:00",
                                        "2018-04-28T00:00:00",
                                        "2018-05-28T00:00:00",
                                        "2018-06-28T00:00:00",
                                        "2018-07-28T00:00:00",
                                        "2018-08-28T00:00:00",
                                        "2018-09-28T00:00:00",
                                        "2018-10-28T00:00:00",
                                        "2018-11-28T00:00:00",
                                        "2018-12-28T00:00:00",
                                        "2019-01-28T00:00:00",
                                        "2019-02-28T00:00:00",
                                        "2019-03-28T00:00:00",
                                        "2019-04-28T00:00:00",
                                        "2019-05-28T00:00:00"
                                    ],
                                    "xaxis": "x",
                                    "y": [
                                        620,
                                        600,
                                        720,
                                        660,
                                        650,
                                        630,
                                        720,
                                        760,
                                        900,
                                        910,
                                        960,
                                        990,
                                        890,
                                        970,
                                        960,
                                        1110,
                                        1060
                                    ],
                                    "yaxis": "y"
                                },
                                {
                                    "alignmentgroup": "True",
                                    "hovertemplate": "entry_channel=KAA<br>date=%{x}<br>Ingresos=%{y}<extra></extra>",
                                    "legendgroup": "KAA",
                                    "marker": {
                                        "color": "#f0ed35"
                                    },
                                    "name": "KAA",
                                    "offsetgroup": "KAA",
                                    "orientation": "v",
                                    "showlegend": true,
                                    "textposition": "auto",
                                    "type": "bar",
                                    "x": [
                                        "2018-01-28T00:00:00",
                                        "2018-02-28T00:00:00",
                                        "2018-03-28T00:00:00",
                                        "2018-04-28T00:00:00",
                                        "2018-05-28T00:00:00",
                                        "2018-06-28T00:00:00",
                                        "2018-07-28T00:00:00",
                                        "2018-08-28T00:00:00",
                                        "2018-09-28T00:00:00",
                                        "2018-10-28T00:00:00",
                                        "2018-11-28T00:00:00",
                                        "2018-12-28T00:00:00",
                                        "2019-01-28T00:00:00",
                                        "2019-02-28T00:00:00",
                                        "2019-03-28T00:00:00",
                                        "2019-04-28T00:00:00",
                                        "2019-05-28T00:00:00"
                                    ],
                                    "xaxis": "x",
                                    "y": [
                                        260,
                                        260,
                                        260,
                                        260,
                                        260,
                                        260,
                                        220,
                                        280,
                                        130,
                                        200,
                                        240,
                                        190,
                                        190,
                                        140,
                                        190,
                                        250,
                                        270
                                    ],
                                    "yaxis": "y"
                                },
                                {
                                    "alignmentgroup": "True",
                                    "hovertemplate": "entry_channel=KAB<br>date=%{x}<br>Ingresos=%{y}<extra></extra>",
                                    "legendgroup": "KAB",
                                    "marker": {
                                        "color": "#c6e516"
                                    },
                                    "name": "KAB",
                                    "offsetgroup": "KAB",
                                    "orientation": "v",
                                    "showlegend": true,
                                    "textposition": "auto",
                                    "type": "bar",
                                    "x": [
                                        "2018-01-28T00:00:00",
                                        "2018-02-28T00:00:00",
                                        "2018-03-28T00:00:00",
                                        "2018-04-28T00:00:00",
                                        "2018-05-28T00:00:00",
                                        "2018-06-28T00:00:00",
                                        "2018-07-28T00:00:00",
                                        "2018-08-28T00:00:00",
                                        "2018-09-28T00:00:00",
                                        "2018-10-28T00:00:00",
                                        "2018-11-28T00:00:00",
                                        "2018-12-28T00:00:00",
                                        "2019-01-28T00:00:00",
                                        "2019-02-28T00:00:00",
                                        "2019-05-28T00:00:00"
                                    ],
                                    "xaxis": "x",
                                    "y": [
                                        120,
                                        120,
                                        60,
                                        120,
                                        60,
                                        60,
                                        60,
                                        60,
                                        120,
                                        120,
                                        60,
                                        50,
                                        120,
                                        60,
                                        60
                                    ],
                                    "yaxis": "y"
                                },
                                {
                                    "alignmentgroup": "True",
                                    "hovertemplate": "entry_channel=KAD<br>date=%{x}<br>Ingresos=%{y}<extra></extra>",
                                    "legendgroup": "KAD",
                                    "marker": {
                                        "color": "#96d310"
                                    },
                                    "name": "KAD",
                                    "offsetgroup": "KAD",
                                    "orientation": "v",
                                    "showlegend": true,
                                    "textposition": "auto",
                                    "type": "bar",
                                    "x": [
                                        "2018-01-28T00:00:00",
                                        "2018-02-28T00:00:00",
                                        "2018-03-28T00:00:00",
                                        "2018-04-28T00:00:00",
                                        "2018-05-28T00:00:00",
                                        "2018-06-28T00:00:00",
                                        "2018-07-28T00:00:00",
                                        "2018-08-28T00:00:00",
                                        "2018-09-28T00:00:00",
                                        "2018-10-28T00:00:00",
                                        "2018-11-28T00:00:00",
                                        "2018-12-28T00:00:00",
                                        "2019-01-28T00:00:00",
                                        "2019-02-28T00:00:00",
                                        "2019-03-28T00:00:00",
                                        "2019-04-28T00:00:00",
                                        "2019-05-28T00:00:00"
                                    ],
                                    "xaxis": "x",
                                    "y": [
                                        130,
                                        130,
                                        130,
                                        130,
                                        130,
                                        130,
                                        130,
                                        130,
                                        130,
                                        130,
                                        130,
                                        130,
                                        130,
                                        130,
                                        130,
                                        130,
                                        130
                                    ],
                                    "yaxis": "y"
                                },
                                {
                                    "alignmentgroup": "True",
                                    "hovertemplate": "entry_channel=KAE<br>date=%{x}<br>Ingresos=%{y}<extra></extra>",
                                    "legendgroup": "KAE",
                                    "marker": {
                                        "color": "#61c10b"
                                    },
                                    "name": "KAE",
                                    "offsetgroup": "KAE",
                                    "orientation": "v",
                                    "showlegend": true,
                                    "textposition": "auto",
                                    "type": "bar",
                                    "x": [
                                        "2018-01-28T00:00:00",
                                        "2018-02-28T00:00:00",
                                        "2018-03-28T00:00:00",
                                        "2018-04-28T00:00:00",
                                        "2018-05-28T00:00:00",
                                        "2018-06-28T00:00:00",
                                        "2018-07-28T00:00:00",
                                        "2018-08-28T00:00:00",
                                        "2018-09-28T00:00:00",
                                        "2018-10-28T00:00:00",
                                        "2018-11-28T00:00:00",
                                        "2018-12-28T00:00:00",
                                        "2019-01-28T00:00:00",
                                        "2019-02-28T00:00:00",
                                        "2019-03-28T00:00:00",
                                        "2019-04-28T00:00:00",
                                        "2019-05-28T00:00:00"
                                    ],
                                    "xaxis": "x",
                                    "y": [
                                        80,
                                        80,
                                        130,
                                        190,
                                        140,
                                        190,
                                        130,
                                        180,
                                        190,
                                        140,
                                        140,
                                        190,
                                        90,
                                        140,
                                        80,
                                        30,
                                        90
                                    ],
                                    "yaxis": "y"
                                },
                                {
                                    "alignmentgroup": "True",
                                    "hovertemplate": "entry_channel=KAF<br>date=%{x}<br>Ingresos=%{y}<extra></extra>",
                                    "legendgroup": "KAF",
                                    "marker": {
                                        "color": "#31ac28"
                                    },
                                    "name": "KAF",
                                    "offsetgroup": "KAF",
                                    "orientation": "v",
                                    "showlegend": true,
                                    "textposition": "auto",
                                    "type": "bar",
                                    "x": [
                                        "2018-01-28T00:00:00",
                                        "2018-02-28T00:00:00",
                                        "2018-03-28T00:00:00",
                                        "2018-04-28T00:00:00",
                                        "2018-05-28T00:00:00",
                                        "2018-06-28T00:00:00",
                                        "2018-07-28T00:00:00",
                                        "2018-08-28T00:00:00",
                                        "2018-09-28T00:00:00",
                                        "2018-10-28T00:00:00",
                                        "2018-11-28T00:00:00",
                                        "2018-12-28T00:00:00",
                                        "2019-01-28T00:00:00",
                                        "2019-02-28T00:00:00",
                                        "2019-03-28T00:00:00",
                                        "2019-04-28T00:00:00",
                                        "2019-05-28T00:00:00"
                                    ],
                                    "xaxis": "x",
                                    "y": [
                                        290,
                                        290,
                                        300,
                                        260,
                                        260,
                                        260,
                                        200,
                                        160,
                                        160,
                                        160,
                                        160,
                                        220,
                                        160,
                                        110,
                                        220,
                                        330,
                                        270
                                    ],
                                    "yaxis": "y"
                                },
                                {
                                    "alignmentgroup": "True",
                                    "hovertemplate": "entry_channel=KAG<br>date=%{x}<br>Ingresos=%{y}<extra></extra>",
                                    "legendgroup": "KAG",
                                    "marker": {
                                        "color": "#439064"
                                    },
                                    "name": "KAG",
                                    "offsetgroup": "KAG",
                                    "orientation": "v",
                                    "showlegend": true,
                                    "textposition": "auto",
                                    "type": "bar",
                                    "x": [
                                        "2018-01-28T00:00:00",
                                        "2018-02-28T00:00:00",
                                        "2018-03-28T00:00:00",
                                        "2018-04-28T00:00:00",
                                        "2018-05-28T00:00:00",
                                        "2018-06-28T00:00:00",
                                        "2018-07-28T00:00:00",
                                        "2018-08-28T00:00:00",
                                        "2018-09-28T00:00:00",
                                        "2018-10-28T00:00:00",
                                        "2018-11-28T00:00:00",
                                        "2018-12-28T00:00:00",
                                        "2019-01-28T00:00:00",
                                        "2019-02-28T00:00:00",
                                        "2019-03-28T00:00:00",
                                        "2019-04-28T00:00:00",
                                        "2019-05-28T00:00:00"
                                    ],
                                    "xaxis": "x",
                                    "y": [
                                        380,
                                        370,
                                        370,
                                        370,
                                        310,
                                        350,
                                        410,
                                        320,
                                        370,
                                        400,
                                        500,
                                        460,
                                        460,
                                        470,
                                        360,
                                        430,
                                        420
                                    ],
                                    "yaxis": "y"
                                },
                                {
                                    "alignmentgroup": "True",
                                    "hovertemplate": "entry_channel=KAH<br>date=%{x}<br>Ingresos=%{y}<extra></extra>",
                                    "legendgroup": "KAH",
                                    "marker": {
                                        "color": "#3d719a"
                                    },
                                    "name": "KAH",
                                    "offsetgroup": "KAH",
                                    "orientation": "v",
                                    "showlegend": true,
                                    "textposition": "auto",
                                    "type": "bar",
                                    "x": [
                                        "2018-01-28T00:00:00",
                                        "2018-02-28T00:00:00",
                                        "2018-03-28T00:00:00",
                                        "2018-04-28T00:00:00",
                                        "2018-05-28T00:00:00",
                                        "2018-06-28T00:00:00",
                                        "2018-07-28T00:00:00",
                                        "2018-08-28T00:00:00",
                                        "2018-09-28T00:00:00",
                                        "2018-10-28T00:00:00",
                                        "2018-11-28T00:00:00",
                                        "2018-12-28T00:00:00",
                                        "2019-01-28T00:00:00",
                                        "2019-02-28T00:00:00",
                                        "2019-03-28T00:00:00",
                                        "2019-04-28T00:00:00",
                                        "2019-05-28T00:00:00"
                                    ],
                                    "xaxis": "x",
                                    "y": [
                                        70,
                                        70,
                                        10,
                                        10,
                                        10,
                                        10,
                                        10,
                                        10,
                                        10,
                                        10,
                                        10,
                                        10,
                                        10,
                                        10,
                                        10,
                                        10,
                                        10
                                    ],
                                    "yaxis": "y"
                                },
                                {
                                    "alignmentgroup": "True",
                                    "hovertemplate": "entry_channel=KAJ<br>date=%{x}<br>Ingresos=%{y}<extra></extra>",
                                    "legendgroup": "KAJ",
                                    "marker": {
                                        "color": "#284ec8"
                                    },
                                    "name": "KAJ",
                                    "offsetgroup": "KAJ",
                                    "orientation": "v",
                                    "showlegend": true,
                                    "textposition": "auto",
                                    "type": "bar",
                                    "x": [
                                        "2018-01-28T00:00:00",
                                        "2018-02-28T00:00:00",
                                        "2018-03-28T00:00:00",
                                        "2018-04-28T00:00:00",
                                        "2018-05-28T00:00:00",
                                        "2018-06-28T00:00:00",
                                        "2018-07-28T00:00:00",
                                        "2018-08-28T00:00:00",
                                        "2018-09-28T00:00:00",
                                        "2018-10-28T00:00:00",
                                        "2018-11-28T00:00:00",
                                        "2018-12-28T00:00:00",
                                        "2019-01-28T00:00:00",
                                        "2019-02-28T00:00:00",
                                        "2019-03-28T00:00:00",
                                        "2019-04-28T00:00:00",
                                        "2019-05-28T00:00:00"
                                    ],
                                    "xaxis": "x",
                                    "y": [
                                        40,
                                        40,
                                        40,
                                        40,
                                        40,
                                        40,
                                        40,
                                        40,
                                        40,
                                        40,
                                        40,
                                        40,
                                        40,
                                        40,
                                        40,
                                        40,
                                        40
                                    ],
                                    "yaxis": "y"
                                },
                                {
                                    "alignmentgroup": "True",
                                    "hovertemplate": "entry_channel=KAK<br>date=%{x}<br>Ingresos=%{y}<extra></extra>",
                                    "legendgroup": "KAK",
                                    "marker": {
                                        "color": "#2e21ea"
                                    },
                                    "name": "KAK",
                                    "offsetgroup": "KAK",
                                    "orientation": "v",
                                    "showlegend": true,
                                    "textposition": "auto",
                                    "type": "bar",
                                    "x": [
                                        "2018-01-28T00:00:00",
                                        "2018-02-28T00:00:00",
                                        "2018-03-28T00:00:00",
                                        "2018-04-28T00:00:00",
                                        "2018-05-28T00:00:00",
                                        "2018-06-28T00:00:00",
                                        "2018-07-28T00:00:00",
                                        "2018-08-28T00:00:00",
                                        "2018-09-28T00:00:00",
                                        "2018-10-28T00:00:00",
                                        "2018-11-28T00:00:00",
                                        "2018-12-28T00:00:00",
                                        "2019-01-28T00:00:00",
                                        "2019-02-28T00:00:00",
                                        "2019-03-28T00:00:00",
                                        "2019-04-28T00:00:00",
                                        "2019-05-28T00:00:00"
                                    ],
                                    "xaxis": "x",
                                    "y": [
                                        20,
                                        20,
                                        20,
                                        20,
                                        30,
                                        30,
                                        40,
                                        30,
                                        30,
                                        30,
                                        30,
                                        30,
                                        30,
                                        30,
                                        30,
                                        30,
                                        30
                                    ],
                                    "yaxis": "y"
                                },
                                {
                                    "alignmentgroup": "True",
                                    "hovertemplate": "entry_channel=KAM<br>date=%{x}<br>Ingresos=%{y}<extra></extra>",
                                    "legendgroup": "KAM",
                                    "marker": {
                                        "color": "#6324f5"
                                    },
                                    "name": "KAM",
                                    "offsetgroup": "KAM",
                                    "orientation": "v",
                                    "showlegend": true,
                                    "textposition": "auto",
                                    "type": "bar",
                                    "x": [
                                        "2018-01-28T00:00:00",
                                        "2018-02-28T00:00:00",
                                        "2018-03-28T00:00:00",
                                        "2018-04-28T00:00:00",
                                        "2018-05-28T00:00:00",
                                        "2018-06-28T00:00:00",
                                        "2018-07-28T00:00:00",
                                        "2018-08-28T00:00:00",
                                        "2018-09-28T00:00:00",
                                        "2018-10-28T00:00:00",
                                        "2018-11-28T00:00:00",
                                        "2018-12-28T00:00:00",
                                        "2019-01-28T00:00:00",
                                        "2019-02-28T00:00:00",
                                        "2019-03-28T00:00:00",
                                        "2019-04-28T00:00:00",
                                        "2019-05-28T00:00:00"
                                    ],
                                    "xaxis": "x",
                                    "y": [
                                        10,
                                        10,
                                        50,
                                        50,
                                        50,
                                        50,
                                        50,
                                        50,
                                        50,
                                        50,
                                        50,
                                        50,
                                        50,
                                        50,
                                        50,
                                        10,
                                        10
                                    ],
                                    "yaxis": "y"
                                },
                                {
                                    "alignmentgroup": "True",
                                    "hovertemplate": "entry_channel=KAQ<br>date=%{x}<br>Ingresos=%{y}<extra></extra>",
                                    "legendgroup": "KAQ",
                                    "marker": {
                                        "color": "#9139fa"
                                    },
                                    "name": "KAQ",
                                    "offsetgroup": "KAQ",
                                    "orientation": "v",
                                    "showlegend": true,
                                    "textposition": "auto",
                                    "type": "bar",
                                    "x": [
                                        "2018-01-28T00:00:00",
                                        "2018-02-28T00:00:00",
                                        "2018-03-28T00:00:00",
                                        "2018-04-28T00:00:00",
                                        "2018-05-28T00:00:00",
                                        "2018-06-28T00:00:00",
                                        "2018-07-28T00:00:00",
                                        "2018-08-28T00:00:00",
                                        "2018-09-28T00:00:00",
                                        "2018-10-28T00:00:00",
                                        "2018-11-28T00:00:00",
                                        "2018-12-28T00:00:00",
                                        "2019-01-28T00:00:00",
                                        "2019-02-28T00:00:00",
                                        "2019-03-28T00:00:00",
                                        "2019-04-28T00:00:00",
                                        "2019-05-28T00:00:00"
                                    ],
                                    "xaxis": "x",
                                    "y": [
                                        40,
                                        80,
                                        80,
                                        80,
                                        80,
                                        80,
                                        40,
                                        100,
                                        100,
                                        100,
                                        100,
                                        100,
                                        100,
                                        100,
                                        100,
                                        100,
                                        100
                                    ],
                                    "yaxis": "y"
                                },
                                {
                                    "alignmentgroup": "True",
                                    "hovertemplate": "entry_channel=KAR<br>date=%{x}<br>Ingresos=%{y}<extra></extra>",
                                    "legendgroup": "KAR",
                                    "marker": {
                                        "color": "#c543fa"
                                    },
                                    "name": "KAR",
                                    "offsetgroup": "KAR",
                                    "orientation": "v",
                                    "showlegend": true,
                                    "textposition": "auto",
                                    "type": "bar",
                                    "x": [
                                        "2018-01-28T00:00:00",
                                        "2018-02-28T00:00:00",
                                        "2018-03-28T00:00:00",
                                        "2018-04-28T00:00:00",
                                        "2018-05-28T00:00:00",
                                        "2018-06-28T00:00:00",
                                        "2018-07-28T00:00:00",
                                        "2018-08-28T00:00:00",
                                        "2018-09-28T00:00:00",
                                        "2018-10-28T00:00:00",
                                        "2018-11-28T00:00:00",
                                        "2018-12-28T00:00:00",
                                        "2019-01-28T00:00:00",
                                        "2019-02-28T00:00:00",
                                        "2019-03-28T00:00:00",
                                        "2019-04-28T00:00:00",
                                        "2019-05-28T00:00:00"
                                    ],
                                    "xaxis": "x",
                                    "y": [
                                        180,
                                        180,
                                        140,
                                        140,
                                        140,
                                        100,
                                        100,
                                        60,
                                        60,
                                        60,
                                        60,
                                        60,
                                        60,
                                        50,
                                        50,
                                        50,
                                        50
                                    ],
                                    "yaxis": "y"
                                },
                                {
                                    "alignmentgroup": "True",
                                    "hovertemplate": "entry_channel=KAS<br>date=%{x}<br>Ingresos=%{y}<extra></extra>",
                                    "legendgroup": "KAS",
                                    "marker": {
                                        "color": "#ef55f1"
                                    },
                                    "name": "KAS",
                                    "offsetgroup": "KAS",
                                    "orientation": "v",
                                    "showlegend": true,
                                    "textposition": "auto",
                                    "type": "bar",
                                    "x": [
                                        "2018-01-28T00:00:00",
                                        "2018-02-28T00:00:00",
                                        "2018-03-28T00:00:00",
                                        "2018-04-28T00:00:00",
                                        "2018-05-28T00:00:00",
                                        "2018-06-28T00:00:00",
                                        "2018-07-28T00:00:00",
                                        "2018-08-28T00:00:00",
                                        "2018-09-28T00:00:00",
                                        "2018-10-28T00:00:00",
                                        "2018-11-28T00:00:00",
                                        "2018-12-28T00:00:00",
                                        "2019-01-28T00:00:00",
                                        "2019-02-28T00:00:00",
                                        "2019-03-28T00:00:00",
                                        "2019-04-28T00:00:00",
                                        "2019-05-28T00:00:00"
                                    ],
                                    "xaxis": "x",
                                    "y": [
                                        460,
                                        510,
                                        360,
                                        470,
                                        420,
                                        530,
                                        480,
                                        530,
                                        510,
                                        610,
                                        610,
                                        550,
                                        420,
                                        450,
                                        470,
                                        590,
                                        640
                                    ],
                                    "yaxis": "y"
                                },
                                {
                                    "alignmentgroup": "True",
                                    "hovertemplate": "entry_channel=KAT<br>date=%{x}<br>Ingresos=%{y}<extra></extra>",
                                    "legendgroup": "KAT",
                                    "marker": {
                                        "color": "#ef55f1"
                                    },
                                    "name": "KAT",
                                    "offsetgroup": "KAT",
                                    "orientation": "v",
                                    "showlegend": true,
                                    "textposition": "auto",
                                    "type": "bar",
                                    "x": [
                                        "2018-01-28T00:00:00",
                                        "2018-02-28T00:00:00",
                                        "2018-03-28T00:00:00",
                                        "2018-04-28T00:00:00",
                                        "2018-05-28T00:00:00",
                                        "2018-06-28T00:00:00",
                                        "2018-07-28T00:00:00",
                                        "2018-08-28T00:00:00",
                                        "2018-09-28T00:00:00",
                                        "2018-10-28T00:00:00",
                                        "2018-11-28T00:00:00",
                                        "2018-12-28T00:00:00",
                                        "2019-01-28T00:00:00",
                                        "2019-02-28T00:00:00",
                                        "2019-03-28T00:00:00",
                                        "2019-04-28T00:00:00",
                                        "2019-05-28T00:00:00"
                                    ],
                                    "xaxis": "x",
                                    "y": [
                                        503400,
                                        513770,
                                        522630,
                                        535000,
                                        527980,
                                        543120,
                                        550200,
                                        540570,
                                        544360,
                                        550880,
                                        550900,
                                        557580,
                                        530360,
                                        549080,
                                        549740,
                                        545150,
                                        548860
                                    ],
                                    "yaxis": "y"
                                },
                                {
                                    "alignmentgroup": "True",
                                    "hovertemplate": "entry_channel=KAW<br>date=%{x}<br>Ingresos=%{y}<extra></extra>",
                                    "legendgroup": "KAW",
                                    "marker": {
                                        "color": "#fb84ce"
                                    },
                                    "name": "KAW",
                                    "offsetgroup": "KAW",
                                    "orientation": "v",
                                    "showlegend": true,
                                    "textposition": "auto",
                                    "type": "bar",
                                    "x": [
                                        "2018-01-28T00:00:00",
                                        "2018-02-28T00:00:00",
                                        "2018-03-28T00:00:00",
                                        "2018-04-28T00:00:00",
                                        "2018-05-28T00:00:00",
                                        "2018-06-28T00:00:00",
                                        "2018-07-28T00:00:00",
                                        "2018-08-28T00:00:00",
                                        "2018-09-28T00:00:00",
                                        "2018-10-28T00:00:00",
                                        "2018-11-28T00:00:00",
                                        "2018-12-28T00:00:00",
                                        "2019-01-28T00:00:00",
                                        "2019-02-28T00:00:00",
                                        "2019-03-28T00:00:00",
                                        "2019-04-28T00:00:00",
                                        "2019-05-28T00:00:00"
                                    ],
                                    "xaxis": "x",
                                    "y": [
                                        100,
                                        40,
                                        40,
                                        40,
                                        40,
                                        40,
                                        30,
                                        30,
                                        30,
                                        90,
                                        40,
                                        40,
                                        40,
                                        30,
                                        20,
                                        20,
                                        20
                                    ],
                                    "yaxis": "y"
                                },
                                {
                                    "alignmentgroup": "True",
                                    "hovertemplate": "entry_channel=KAY<br>date=%{x}<br>Ingresos=%{y}<extra></extra>",
                                    "legendgroup": "KAY",
                                    "marker": {
                                        "color": "#fbafa1"
                                    },
                                    "name": "KAY",
                                    "offsetgroup": "KAY",
                                    "orientation": "v",
                                    "showlegend": true,
                                    "textposition": "auto",
                                    "type": "bar",
                                    "x": [
                                        "2018-01-28T00:00:00",
                                        "2018-02-28T00:00:00",
                                        "2018-03-28T00:00:00",
                                        "2018-04-28T00:00:00",
                                        "2018-05-28T00:00:00",
                                        "2018-06-28T00:00:00",
                                        "2018-07-28T00:00:00",
                                        "2018-08-28T00:00:00",
                                        "2018-09-28T00:00:00",
                                        "2018-10-28T00:00:00",
                                        "2018-11-28T00:00:00",
                                        "2018-12-28T00:00:00",
                                        "2019-01-28T00:00:00",
                                        "2019-02-28T00:00:00",
                                        "2019-03-28T00:00:00",
                                        "2019-04-28T00:00:00",
                                        "2019-05-28T00:00:00"
                                    ],
                                    "xaxis": "x",
                                    "y": [
                                        280,
                                        280,
                                        280,
                                        280,
                                        280,
                                        230,
                                        290,
                                        240,
                                        240,
                                        210,
                                        110,
                                        110,
                                        130,
                                        120,
                                        120,
                                        110,
                                        120
                                    ],
                                    "yaxis": "y"
                                },
                                {
                                    "alignmentgroup": "True",
                                    "hovertemplate": "entry_channel=KAZ<br>date=%{x}<br>Ingresos=%{y}<extra></extra>",
                                    "legendgroup": "KAZ",
                                    "marker": {
                                        "color": "#fcd471"
                                    },
                                    "name": "KAZ",
                                    "offsetgroup": "KAZ",
                                    "orientation": "v",
                                    "showlegend": true,
                                    "textposition": "auto",
                                    "type": "bar",
                                    "x": [
                                        "2018-01-28T00:00:00",
                                        "2018-02-28T00:00:00",
                                        "2018-03-28T00:00:00",
                                        "2018-04-28T00:00:00",
                                        "2018-05-28T00:00:00",
                                        "2018-06-28T00:00:00",
                                        "2018-07-28T00:00:00",
                                        "2018-08-28T00:00:00",
                                        "2018-09-28T00:00:00",
                                        "2018-10-28T00:00:00",
                                        "2018-11-28T00:00:00",
                                        "2018-12-28T00:00:00",
                                        "2019-01-28T00:00:00",
                                        "2019-02-28T00:00:00",
                                        "2019-03-28T00:00:00",
                                        "2019-04-28T00:00:00",
                                        "2019-05-28T00:00:00"
                                    ],
                                    "xaxis": "x",
                                    "y": [
                                        9420,
                                        9370,
                                        9680,
                                        9350,
                                        9210,
                                        9250,
                                        9180,
                                        9270,
                                        8990,
                                        9100,
                                        9030,
                                        9330,
                                        9100,
                                        9420,
                                        9130,
                                        9260,
                                        9040
                                    ],
                                    "yaxis": "y"
                                },
                                {
                                    "alignmentgroup": "True",
                                    "hovertemplate": "entry_channel=KBE<br>date=%{x}<br>Ingresos=%{y}<extra></extra>",
                                    "legendgroup": "KBE",
                                    "marker": {
                                        "color": "#f0ed35"
                                    },
                                    "name": "KBE",
                                    "offsetgroup": "KBE",
                                    "orientation": "v",
                                    "showlegend": true,
                                    "textposition": "auto",
                                    "type": "bar",
                                    "x": [
                                        "2018-01-28T00:00:00",
                                        "2018-02-28T00:00:00",
                                        "2018-03-28T00:00:00",
                                        "2018-04-28T00:00:00",
                                        "2018-05-28T00:00:00",
                                        "2018-06-28T00:00:00",
                                        "2018-07-28T00:00:00",
                                        "2018-08-28T00:00:00",
                                        "2018-09-28T00:00:00",
                                        "2018-10-28T00:00:00",
                                        "2018-11-28T00:00:00",
                                        "2018-12-28T00:00:00",
                                        "2019-01-28T00:00:00",
                                        "2019-02-28T00:00:00",
                                        "2019-03-28T00:00:00",
                                        "2019-04-28T00:00:00",
                                        "2019-05-28T00:00:00"
                                    ],
                                    "xaxis": "x",
                                    "y": [
                                        70,
                                        70,
                                        70,
                                        70,
                                        70,
                                        70,
                                        70,
                                        70,
                                        70,
                                        70,
                                        70,
                                        70,
                                        70,
                                        70,
                                        70,
                                        70,
                                        70
                                    ],
                                    "yaxis": "y"
                                },
                                {
                                    "alignmentgroup": "True",
                                    "hovertemplate": "entry_channel=KBG<br>date=%{x}<br>Ingresos=%{y}<extra></extra>",
                                    "legendgroup": "KBG",
                                    "marker": {
                                        "color": "#c6e516"
                                    },
                                    "name": "KBG",
                                    "offsetgroup": "KBG",
                                    "orientation": "v",
                                    "showlegend": true,
                                    "textposition": "auto",
                                    "type": "bar",
                                    "x": [
                                        "2018-01-28T00:00:00",
                                        "2018-02-28T00:00:00",
                                        "2018-03-28T00:00:00",
                                        "2018-04-28T00:00:00",
                                        "2018-05-28T00:00:00",
                                        "2018-06-28T00:00:00",
                                        "2018-07-28T00:00:00",
                                        "2018-08-28T00:00:00",
                                        "2018-09-28T00:00:00",
                                        "2018-10-28T00:00:00",
                                        "2018-11-28T00:00:00",
                                        "2018-12-28T00:00:00",
                                        "2019-01-28T00:00:00",
                                        "2019-02-28T00:00:00",
                                        "2019-03-28T00:00:00",
                                        "2019-04-28T00:00:00",
                                        "2019-05-28T00:00:00"
                                    ],
                                    "xaxis": "x",
                                    "y": [
                                        2200,
                                        2450,
                                        2370,
                                        2370,
                                        2390,
                                        2290,
                                        2350,
                                        2420,
                                        2330,
                                        2390,
                                        2400,
                                        2390,
                                        2020,
                                        2150,
                                        2120,
                                        2220,
                                        2190
                                    ],
                                    "yaxis": "y"
                                },
                                {
                                    "alignmentgroup": "True",
                                    "hovertemplate": "entry_channel=KBH<br>date=%{x}<br>Ingresos=%{y}<extra></extra>",
                                    "legendgroup": "KBH",
                                    "marker": {
                                        "color": "#96d310"
                                    },
                                    "name": "KBH",
                                    "offsetgroup": "KBH",
                                    "orientation": "v",
                                    "showlegend": true,
                                    "textposition": "auto",
                                    "type": "bar",
                                    "x": [
                                        "2018-01-28T00:00:00",
                                        "2018-02-28T00:00:00",
                                        "2018-03-28T00:00:00",
                                        "2018-04-28T00:00:00",
                                        "2018-05-28T00:00:00",
                                        "2018-06-28T00:00:00",
                                        "2018-07-28T00:00:00",
                                        "2018-08-28T00:00:00",
                                        "2018-09-28T00:00:00",
                                        "2018-10-28T00:00:00",
                                        "2018-11-28T00:00:00",
                                        "2018-12-28T00:00:00",
                                        "2019-01-28T00:00:00",
                                        "2019-02-28T00:00:00",
                                        "2019-03-28T00:00:00",
                                        "2019-04-28T00:00:00",
                                        "2019-05-28T00:00:00"
                                    ],
                                    "xaxis": "x",
                                    "y": [
                                        40,
                                        40,
                                        40,
                                        40,
                                        40,
                                        40,
                                        40,
                                        40,
                                        40,
                                        40,
                                        40,
                                        40,
                                        40,
                                        40,
                                        40,
                                        40,
                                        40
                                    ],
                                    "yaxis": "y"
                                },
                                {
                                    "alignmentgroup": "True",
                                    "hovertemplate": "entry_channel=KBO<br>date=%{x}<br>Ingresos=%{y}<extra></extra>",
                                    "legendgroup": "KBO",
                                    "marker": {
                                        "color": "#61c10b"
                                    },
                                    "name": "KBO",
                                    "offsetgroup": "KBO",
                                    "orientation": "v",
                                    "showlegend": true,
                                    "textposition": "auto",
                                    "type": "bar",
                                    "x": [
                                        "2018-01-28T00:00:00",
                                        "2018-02-28T00:00:00",
                                        "2018-03-28T00:00:00",
                                        "2018-04-28T00:00:00",
                                        "2018-05-28T00:00:00",
                                        "2018-06-28T00:00:00",
                                        "2018-07-28T00:00:00",
                                        "2018-08-28T00:00:00",
                                        "2018-09-28T00:00:00",
                                        "2018-10-28T00:00:00",
                                        "2018-11-28T00:00:00",
                                        "2018-12-28T00:00:00",
                                        "2019-01-28T00:00:00",
                                        "2019-02-28T00:00:00",
                                        "2019-03-28T00:00:00",
                                        "2019-04-28T00:00:00",
                                        "2019-05-28T00:00:00"
                                    ],
                                    "xaxis": "x",
                                    "y": [
                                        60,
                                        60,
                                        60,
                                        70,
                                        70,
                                        70,
                                        70,
                                        70,
                                        70,
                                        20,
                                        10,
                                        20,
                                        20,
                                        20,
                                        20,
                                        20,
                                        20
                                    ],
                                    "yaxis": "y"
                                },
                                {
                                    "alignmentgroup": "True",
                                    "hovertemplate": "entry_channel=KBW<br>date=%{x}<br>Ingresos=%{y}<extra></extra>",
                                    "legendgroup": "KBW",
                                    "marker": {
                                        "color": "#31ac28"
                                    },
                                    "name": "KBW",
                                    "offsetgroup": "KBW",
                                    "orientation": "v",
                                    "showlegend": true,
                                    "textposition": "auto",
                                    "type": "bar",
                                    "x": [
                                        "2018-06-28T00:00:00",
                                        "2018-07-28T00:00:00",
                                        "2018-08-28T00:00:00",
                                        "2018-09-28T00:00:00",
                                        "2018-10-28T00:00:00",
                                        "2018-11-28T00:00:00",
                                        "2018-12-28T00:00:00",
                                        "2019-01-28T00:00:00",
                                        "2019-02-28T00:00:00",
                                        "2019-03-28T00:00:00",
                                        "2019-04-28T00:00:00",
                                        "2019-05-28T00:00:00"
                                    ],
                                    "xaxis": "x",
                                    "y": [
                                        40,
                                        40,
                                        40,
                                        40,
                                        40,
                                        40,
                                        40,
                                        40,
                                        40,
                                        40,
                                        40,
                                        40
                                    ],
                                    "yaxis": "y"
                                },
                                {
                                    "alignmentgroup": "True",
                                    "hovertemplate": "entry_channel=KBZ<br>date=%{x}<br>Ingresos=%{y}<extra></extra>",
                                    "legendgroup": "KBZ",
                                    "marker": {
                                        "color": "#439064"
                                    },
                                    "name": "KBZ",
                                    "offsetgroup": "KBZ",
                                    "orientation": "v",
                                    "showlegend": true,
                                    "textposition": "auto",
                                    "type": "bar",
                                    "x": [
                                        "2018-01-28T00:00:00",
                                        "2018-02-28T00:00:00",
                                        "2018-03-28T00:00:00",
                                        "2018-04-28T00:00:00",
                                        "2018-05-28T00:00:00",
                                        "2018-06-28T00:00:00",
                                        "2018-07-28T00:00:00",
                                        "2018-08-28T00:00:00",
                                        "2018-09-28T00:00:00",
                                        "2018-10-28T00:00:00",
                                        "2018-11-28T00:00:00",
                                        "2018-12-28T00:00:00",
                                        "2019-01-28T00:00:00",
                                        "2019-02-28T00:00:00",
                                        "2019-03-28T00:00:00",
                                        "2019-04-28T00:00:00",
                                        "2019-05-28T00:00:00"
                                    ],
                                    "xaxis": "x",
                                    "y": [
                                        170,
                                        220,
                                        320,
                                        330,
                                        320,
                                        260,
                                        260,
                                        260,
                                        270,
                                        280,
                                        270,
                                        270,
                                        270,
                                        270,
                                        280,
                                        270,
                                        270
                                    ],
                                    "yaxis": "y"
                                },
                                {
                                    "alignmentgroup": "True",
                                    "hovertemplate": "entry_channel=KCB<br>date=%{x}<br>Ingresos=%{y}<extra></extra>",
                                    "legendgroup": "KCB",
                                    "marker": {
                                        "color": "#3d719a"
                                    },
                                    "name": "KCB",
                                    "offsetgroup": "KCB",
                                    "orientation": "v",
                                    "showlegend": true,
                                    "textposition": "auto",
                                    "type": "bar",
                                    "x": [
                                        "2018-01-28T00:00:00",
                                        "2018-02-28T00:00:00",
                                        "2018-03-28T00:00:00",
                                        "2018-04-28T00:00:00",
                                        "2018-05-28T00:00:00",
                                        "2018-06-28T00:00:00",
                                        "2018-07-28T00:00:00",
                                        "2018-08-28T00:00:00",
                                        "2018-09-28T00:00:00",
                                        "2018-10-28T00:00:00",
                                        "2018-11-28T00:00:00",
                                        "2018-12-28T00:00:00",
                                        "2019-01-28T00:00:00",
                                        "2019-02-28T00:00:00",
                                        "2019-03-28T00:00:00",
                                        "2019-04-28T00:00:00",
                                        "2019-05-28T00:00:00"
                                    ],
                                    "xaxis": "x",
                                    "y": [
                                        100,
                                        100,
                                        100,
                                        160,
                                        160,
                                        50,
                                        50,
                                        50,
                                        110,
                                        110,
                                        110,
                                        80,
                                        80,
                                        20,
                                        80,
                                        80,
                                        80
                                    ],
                                    "yaxis": "y"
                                },
                                {
                                    "alignmentgroup": "True",
                                    "hovertemplate": "entry_channel=KCC<br>date=%{x}<br>Ingresos=%{y}<extra></extra>",
                                    "legendgroup": "KCC",
                                    "marker": {
                                        "color": "#284ec8"
                                    },
                                    "name": "KCC",
                                    "offsetgroup": "KCC",
                                    "orientation": "v",
                                    "showlegend": true,
                                    "textposition": "auto",
                                    "type": "bar",
                                    "x": [
                                        "2018-01-28T00:00:00",
                                        "2018-02-28T00:00:00",
                                        "2018-03-28T00:00:00",
                                        "2018-04-28T00:00:00",
                                        "2018-05-28T00:00:00",
                                        "2018-06-28T00:00:00",
                                        "2018-07-28T00:00:00",
                                        "2018-08-28T00:00:00",
                                        "2018-09-28T00:00:00",
                                        "2018-10-28T00:00:00",
                                        "2018-11-28T00:00:00",
                                        "2018-12-28T00:00:00",
                                        "2019-01-28T00:00:00",
                                        "2019-02-28T00:00:00",
                                        "2019-03-28T00:00:00",
                                        "2019-04-28T00:00:00",
                                        "2019-05-28T00:00:00"
                                    ],
                                    "xaxis": "x",
                                    "y": [
                                        90,
                                        40,
                                        90,
                                        90,
                                        90,
                                        90,
                                        90,
                                        90,
                                        90,
                                        90,
                                        100,
                                        140,
                                        100,
                                        100,
                                        140,
                                        140,
                                        140
                                    ],
                                    "yaxis": "y"
                                },
                                {
                                    "alignmentgroup": "True",
                                    "hovertemplate": "entry_channel=KCH<br>date=%{x}<br>Ingresos=%{y}<extra></extra>",
                                    "legendgroup": "KCH",
                                    "marker": {
                                        "color": "#2e21ea"
                                    },
                                    "name": "KCH",
                                    "offsetgroup": "KCH",
                                    "orientation": "v",
                                    "showlegend": true,
                                    "textposition": "auto",
                                    "type": "bar",
                                    "x": [
                                        "2018-01-28T00:00:00",
                                        "2018-02-28T00:00:00",
                                        "2018-03-28T00:00:00",
                                        "2018-04-28T00:00:00",
                                        "2018-05-28T00:00:00",
                                        "2018-06-28T00:00:00",
                                        "2018-07-28T00:00:00",
                                        "2018-08-28T00:00:00",
                                        "2018-09-28T00:00:00",
                                        "2018-10-28T00:00:00",
                                        "2018-11-28T00:00:00",
                                        "2018-12-28T00:00:00",
                                        "2019-01-28T00:00:00",
                                        "2019-02-28T00:00:00",
                                        "2019-03-28T00:00:00",
                                        "2019-04-28T00:00:00",
                                        "2019-05-28T00:00:00"
                                    ],
                                    "xaxis": "x",
                                    "y": [
                                        70,
                                        10,
                                        10,
                                        10,
                                        130,
                                        10,
                                        10,
                                        10,
                                        10,
                                        70,
                                        70,
                                        70,
                                        70,
                                        70,
                                        70,
                                        70,
                                        70
                                    ],
                                    "yaxis": "y"
                                },
                                {
                                    "alignmentgroup": "True",
                                    "hovertemplate": "entry_channel=KCI<br>date=%{x}<br>Ingresos=%{y}<extra></extra>",
                                    "legendgroup": "KCI",
                                    "marker": {
                                        "color": "#6324f5"
                                    },
                                    "name": "KCI",
                                    "offsetgroup": "KCI",
                                    "orientation": "v",
                                    "showlegend": true,
                                    "textposition": "auto",
                                    "type": "bar",
                                    "x": [
                                        "2019-01-28T00:00:00",
                                        "2019-02-28T00:00:00",
                                        "2019-03-28T00:00:00",
                                        "2019-04-28T00:00:00",
                                        "2019-05-28T00:00:00"
                                    ],
                                    "xaxis": "x",
                                    "y": [
                                        10,
                                        20,
                                        30,
                                        30,
                                        30
                                    ],
                                    "yaxis": "y"
                                },
                                {
                                    "alignmentgroup": "True",
                                    "hovertemplate": "entry_channel=KCK<br>date=%{x}<br>Ingresos=%{y}<extra></extra>",
                                    "legendgroup": "KCK",
                                    "marker": {
                                        "color": "#9139fa"
                                    },
                                    "name": "KCK",
                                    "offsetgroup": "KCK",
                                    "orientation": "v",
                                    "showlegend": true,
                                    "textposition": "auto",
                                    "type": "bar",
                                    "x": [
                                        "2018-08-28T00:00:00",
                                        "2018-09-28T00:00:00",
                                        "2018-10-28T00:00:00",
                                        "2018-11-28T00:00:00",
                                        "2018-12-28T00:00:00",
                                        "2019-01-28T00:00:00",
                                        "2019-02-28T00:00:00",
                                        "2019-03-28T00:00:00",
                                        "2019-04-28T00:00:00",
                                        "2019-05-28T00:00:00"
                                    ],
                                    "xaxis": "x",
                                    "y": [
                                        40,
                                        40,
                                        40,
                                        40,
                                        40,
                                        40,
                                        40,
                                        40,
                                        40,
                                        40
                                    ],
                                    "yaxis": "y"
                                },
                                {
                                    "alignmentgroup": "True",
                                    "hovertemplate": "entry_channel=KCL<br>date=%{x}<br>Ingresos=%{y}<extra></extra>",
                                    "legendgroup": "KCL",
                                    "marker": {
                                        "color": "#c543fa"
                                    },
                                    "name": "KCL",
                                    "offsetgroup": "KCL",
                                    "orientation": "v",
                                    "showlegend": true,
                                    "textposition": "auto",
                                    "type": "bar",
                                    "x": [
                                        "2018-01-28T00:00:00",
                                        "2018-02-28T00:00:00",
                                        "2018-03-28T00:00:00",
                                        "2018-04-28T00:00:00",
                                        "2018-05-28T00:00:00",
                                        "2018-06-28T00:00:00",
                                        "2018-07-28T00:00:00",
                                        "2018-08-28T00:00:00",
                                        "2018-09-28T00:00:00",
                                        "2018-10-28T00:00:00",
                                        "2018-11-28T00:00:00",
                                        "2018-12-28T00:00:00",
                                        "2019-01-28T00:00:00",
                                        "2019-02-28T00:00:00",
                                        "2019-03-28T00:00:00",
                                        "2019-04-28T00:00:00",
                                        "2019-05-28T00:00:00"
                                    ],
                                    "xaxis": "x",
                                    "y": [
                                        10,
                                        10,
                                        10,
                                        10,
                                        10,
                                        10,
                                        10,
                                        10,
                                        10,
                                        10,
                                        10,
                                        10,
                                        10,
                                        10,
                                        10,
                                        10,
                                        10
                                    ],
                                    "yaxis": "y"
                                },
                                {
                                    "alignmentgroup": "True",
                                    "hovertemplate": "entry_channel=KDA<br>date=%{x}<br>Ingresos=%{y}<extra></extra>",
                                    "legendgroup": "KDA",
                                    "marker": {
                                        "color": "#ef55f1"
                                    },
                                    "name": "KDA",
                                    "offsetgroup": "KDA",
                                    "orientation": "v",
                                    "showlegend": true,
                                    "textposition": "auto",
                                    "type": "bar",
                                    "x": [
                                        "2018-07-28T00:00:00",
                                        "2018-08-28T00:00:00"
                                    ],
                                    "xaxis": "x",
                                    "y": [
                                        10,
                                        10
                                    ],
                                    "yaxis": "y"
                                },
                                {
                                    "alignmentgroup": "True",
                                    "hovertemplate": "entry_channel=KDH<br>date=%{x}<br>Ingresos=%{y}<extra></extra>",
                                    "legendgroup": "KDH",
                                    "marker": {
                                        "color": "#ef55f1"
                                    },
                                    "name": "KDH",
                                    "offsetgroup": "KDH",
                                    "orientation": "v",
                                    "showlegend": true,
                                    "textposition": "auto",
                                    "type": "bar",
                                    "x": [
                                        "2018-01-28T00:00:00",
                                        "2018-02-28T00:00:00",
                                        "2018-03-28T00:00:00",
                                        "2018-04-28T00:00:00",
                                        "2018-05-28T00:00:00",
                                        "2018-06-28T00:00:00",
                                        "2018-07-28T00:00:00",
                                        "2018-08-28T00:00:00",
                                        "2018-09-28T00:00:00",
                                        "2018-10-28T00:00:00",
                                        "2018-11-28T00:00:00",
                                        "2018-12-28T00:00:00",
                                        "2019-01-28T00:00:00",
                                        "2019-02-28T00:00:00",
                                        "2019-03-28T00:00:00",
                                        "2019-04-28T00:00:00",
                                        "2019-05-28T00:00:00"
                                    ],
                                    "xaxis": "x",
                                    "y": [
                                        30,
                                        30,
                                        30,
                                        30,
                                        30,
                                        30,
                                        30,
                                        30,
                                        30,
                                        30,
                                        30,
                                        30,
                                        30,
                                        30,
                                        30,
                                        30,
                                        30
                                    ],
                                    "yaxis": "y"
                                },
                                {
                                    "alignmentgroup": "True",
                                    "hovertemplate": "entry_channel=KDR<br>date=%{x}<br>Ingresos=%{y}<extra></extra>",
                                    "legendgroup": "KDR",
                                    "marker": {
                                        "color": "#fb84ce"
                                    },
                                    "name": "KDR",
                                    "offsetgroup": "KDR",
                                    "orientation": "v",
                                    "showlegend": true,
                                    "textposition": "auto",
                                    "type": "bar",
                                    "x": [
                                        "2018-03-28T00:00:00",
                                        "2018-04-28T00:00:00",
                                        "2018-05-28T00:00:00",
                                        "2018-06-28T00:00:00",
                                        "2018-07-28T00:00:00",
                                        "2018-08-28T00:00:00",
                                        "2018-09-28T00:00:00",
                                        "2018-11-28T00:00:00",
                                        "2018-12-28T00:00:00",
                                        "2019-02-28T00:00:00"
                                    ],
                                    "xaxis": "x",
                                    "y": [
                                        60,
                                        60,
                                        60,
                                        60,
                                        60,
                                        60,
                                        60,
                                        60,
                                        60,
                                        60
                                    ],
                                    "yaxis": "y"
                                },
                                {
                                    "alignmentgroup": "True",
                                    "hovertemplate": "entry_channel=KDT<br>date=%{x}<br>Ingresos=%{y}<extra></extra>",
                                    "legendgroup": "KDT",
                                    "marker": {
                                        "color": "#fbafa1"
                                    },
                                    "name": "KDT",
                                    "offsetgroup": "KDT",
                                    "orientation": "v",
                                    "showlegend": true,
                                    "textposition": "auto",
                                    "type": "bar",
                                    "x": [
                                        "2018-01-28T00:00:00",
                                        "2018-02-28T00:00:00",
                                        "2018-03-28T00:00:00",
                                        "2018-04-28T00:00:00",
                                        "2018-05-28T00:00:00",
                                        "2018-06-28T00:00:00",
                                        "2018-07-28T00:00:00",
                                        "2018-08-28T00:00:00",
                                        "2018-09-28T00:00:00",
                                        "2018-10-28T00:00:00"
                                    ],
                                    "xaxis": "x",
                                    "y": [
                                        20,
                                        20,
                                        80,
                                        80,
                                        80,
                                        80,
                                        10,
                                        20,
                                        10,
                                        10
                                    ],
                                    "yaxis": "y"
                                },
                                {
                                    "alignmentgroup": "True",
                                    "hovertemplate": "entry_channel=KEH<br>date=%{x}<br>Ingresos=%{y}<extra></extra>",
                                    "legendgroup": "KEH",
                                    "marker": {
                                        "color": "#fcd471"
                                    },
                                    "name": "KEH",
                                    "offsetgroup": "KEH",
                                    "orientation": "v",
                                    "showlegend": true,
                                    "textposition": "auto",
                                    "type": "bar",
                                    "x": [
                                        "2018-01-28T00:00:00",
                                        "2018-02-28T00:00:00",
                                        "2018-03-28T00:00:00",
                                        "2018-04-28T00:00:00",
                                        "2018-05-28T00:00:00",
                                        "2018-06-28T00:00:00",
                                        "2018-07-28T00:00:00",
                                        "2018-08-28T00:00:00",
                                        "2018-09-28T00:00:00",
                                        "2018-10-28T00:00:00",
                                        "2018-11-28T00:00:00",
                                        "2018-12-28T00:00:00",
                                        "2019-01-28T00:00:00",
                                        "2019-02-28T00:00:00",
                                        "2019-03-28T00:00:00",
                                        "2019-04-28T00:00:00",
                                        "2019-05-28T00:00:00"
                                    ],
                                    "xaxis": "x",
                                    "y": [
                                        400,
                                        400,
                                        450,
                                        450,
                                        450,
                                        390,
                                        440,
                                        440,
                                        440,
                                        390,
                                        390,
                                        390,
                                        380,
                                        390,
                                        400,
                                        390,
                                        400
                                    ],
                                    "yaxis": "y"
                                },
                                {
                                    "alignmentgroup": "True",
                                    "hovertemplate": "entry_channel=KEY<br>date=%{x}<br>Ingresos=%{y}<extra></extra>",
                                    "legendgroup": "KEY",
                                    "marker": {
                                        "color": "#f0ed35"
                                    },
                                    "name": "KEY",
                                    "offsetgroup": "KEY",
                                    "orientation": "v",
                                    "showlegend": true,
                                    "textposition": "auto",
                                    "type": "bar",
                                    "x": [
                                        "2018-01-28T00:00:00",
                                        "2018-02-28T00:00:00",
                                        "2018-04-28T00:00:00",
                                        "2018-05-28T00:00:00",
                                        "2018-06-28T00:00:00",
                                        "2018-07-28T00:00:00",
                                        "2018-08-28T00:00:00",
                                        "2018-09-28T00:00:00",
                                        "2018-10-28T00:00:00",
                                        "2018-11-28T00:00:00",
                                        "2018-12-28T00:00:00",
                                        "2019-01-28T00:00:00",
                                        "2019-02-28T00:00:00",
                                        "2019-03-28T00:00:00",
                                        "2019-04-28T00:00:00"
                                    ],
                                    "xaxis": "x",
                                    "y": [
                                        40,
                                        40,
                                        40,
                                        40,
                                        40,
                                        40,
                                        40,
                                        40,
                                        40,
                                        40,
                                        40,
                                        40,
                                        40,
                                        40,
                                        40
                                    ],
                                    "yaxis": "y"
                                },
                                {
                                    "alignmentgroup": "True",
                                    "hovertemplate": "entry_channel=KFA<br>date=%{x}<br>Ingresos=%{y}<extra></extra>",
                                    "legendgroup": "KFA",
                                    "marker": {
                                        "color": "#c6e516"
                                    },
                                    "name": "KFA",
                                    "offsetgroup": "KFA",
                                    "orientation": "v",
                                    "showlegend": true,
                                    "textposition": "auto",
                                    "type": "bar",
                                    "x": [
                                        "2018-01-28T00:00:00",
                                        "2018-02-28T00:00:00",
                                        "2018-03-28T00:00:00",
                                        "2018-04-28T00:00:00",
                                        "2018-05-28T00:00:00",
                                        "2018-06-28T00:00:00",
                                        "2018-07-28T00:00:00",
                                        "2018-08-28T00:00:00",
                                        "2018-09-28T00:00:00",
                                        "2018-10-28T00:00:00",
                                        "2018-11-28T00:00:00",
                                        "2018-12-28T00:00:00",
                                        "2019-01-28T00:00:00",
                                        "2019-02-28T00:00:00",
                                        "2019-03-28T00:00:00",
                                        "2019-04-28T00:00:00",
                                        "2019-05-28T00:00:00"
                                    ],
                                    "xaxis": "x",
                                    "y": [
                                        119390,
                                        120480,
                                        122380,
                                        128470,
                                        128090,
                                        133840,
                                        136520,
                                        136690,
                                        138070,
                                        139130,
                                        140470,
                                        143140,
                                        136580,
                                        140280,
                                        140450,
                                        139140,
                                        139550
                                    ],
                                    "yaxis": "y"
                                },
                                {
                                    "alignmentgroup": "True",
                                    "hovertemplate": "entry_channel=KFC<br>date=%{x}<br>Ingresos=%{y}<extra></extra>",
                                    "legendgroup": "KFC",
                                    "marker": {
                                        "color": "#96d310"
                                    },
                                    "name": "KFC",
                                    "offsetgroup": "KFC",
                                    "orientation": "v",
                                    "showlegend": true,
                                    "textposition": "auto",
                                    "type": "bar",
                                    "x": [
                                        "2018-01-28T00:00:00",
                                        "2018-02-28T00:00:00",
                                        "2018-03-28T00:00:00",
                                        "2018-04-28T00:00:00",
                                        "2018-05-28T00:00:00",
                                        "2018-06-28T00:00:00",
                                        "2018-07-28T00:00:00",
                                        "2018-08-28T00:00:00",
                                        "2018-09-28T00:00:00",
                                        "2018-10-28T00:00:00",
                                        "2018-11-28T00:00:00",
                                        "2018-12-28T00:00:00",
                                        "2019-01-28T00:00:00",
                                        "2019-02-28T00:00:00",
                                        "2019-03-28T00:00:00",
                                        "2019-04-28T00:00:00",
                                        "2019-05-28T00:00:00"
                                    ],
                                    "xaxis": "x",
                                    "y": [
                                        1057430,
                                        1066960,
                                        1082710,
                                        1093740,
                                        1083490,
                                        1111710,
                                        1127050,
                                        1101070,
                                        1113170,
                                        1120700,
                                        1115870,
                                        1131310,
                                        1070090,
                                        1107040,
                                        1106860,
                                        1104630,
                                        1109280
                                    ],
                                    "yaxis": "y"
                                },
                                {
                                    "alignmentgroup": "True",
                                    "hovertemplate": "entry_channel=KFD<br>date=%{x}<br>Ingresos=%{y}<extra></extra>",
                                    "legendgroup": "KFD",
                                    "marker": {
                                        "color": "#61c10b"
                                    },
                                    "name": "KFD",
                                    "offsetgroup": "KFD",
                                    "orientation": "v",
                                    "showlegend": true,
                                    "textposition": "auto",
                                    "type": "bar",
                                    "x": [
                                        "2018-01-28T00:00:00",
                                        "2018-02-28T00:00:00",
                                        "2018-03-28T00:00:00",
                                        "2018-04-28T00:00:00",
                                        "2018-05-28T00:00:00",
                                        "2018-06-28T00:00:00",
                                        "2018-07-28T00:00:00",
                                        "2018-08-28T00:00:00",
                                        "2018-09-28T00:00:00",
                                        "2018-10-28T00:00:00",
                                        "2018-11-28T00:00:00",
                                        "2018-12-28T00:00:00",
                                        "2019-01-28T00:00:00",
                                        "2019-02-28T00:00:00",
                                        "2019-03-28T00:00:00",
                                        "2019-04-28T00:00:00",
                                        "2019-05-28T00:00:00"
                                    ],
                                    "xaxis": "x",
                                    "y": [
                                        60,
                                        60,
                                        120,
                                        60,
                                        60,
                                        60,
                                        60,
                                        90,
                                        120,
                                        120,
                                        120,
                                        120,
                                        130,
                                        160,
                                        160,
                                        160,
                                        160
                                    ],
                                    "yaxis": "y"
                                },
                                {
                                    "alignmentgroup": "True",
                                    "hovertemplate": "entry_channel=KFK<br>date=%{x}<br>Ingresos=%{y}<extra></extra>",
                                    "legendgroup": "KFK",
                                    "marker": {
                                        "color": "#31ac28"
                                    },
                                    "name": "KFK",
                                    "offsetgroup": "KFK",
                                    "orientation": "v",
                                    "showlegend": true,
                                    "textposition": "auto",
                                    "type": "bar",
                                    "x": [
                                        "2018-01-28T00:00:00",
                                        "2018-02-28T00:00:00",
                                        "2018-03-28T00:00:00",
                                        "2018-04-28T00:00:00",
                                        "2018-05-28T00:00:00",
                                        "2018-06-28T00:00:00",
                                        "2018-07-28T00:00:00",
                                        "2018-08-28T00:00:00",
                                        "2018-09-28T00:00:00",
                                        "2018-10-28T00:00:00",
                                        "2018-11-28T00:00:00",
                                        "2018-12-28T00:00:00",
                                        "2019-01-28T00:00:00",
                                        "2019-02-28T00:00:00",
                                        "2019-03-28T00:00:00",
                                        "2019-04-28T00:00:00",
                                        "2019-05-28T00:00:00"
                                    ],
                                    "xaxis": "x",
                                    "y": [
                                        10,
                                        10,
                                        10,
                                        10,
                                        10,
                                        10,
                                        10,
                                        10,
                                        10,
                                        10,
                                        10,
                                        10,
                                        10,
                                        10,
                                        10,
                                        10,
                                        10
                                    ],
                                    "yaxis": "y"
                                },
                                {
                                    "alignmentgroup": "True",
                                    "hovertemplate": "entry_channel=KFL<br>date=%{x}<br>Ingresos=%{y}<extra></extra>",
                                    "legendgroup": "KFL",
                                    "marker": {
                                        "color": "#439064"
                                    },
                                    "name": "KFL",
                                    "offsetgroup": "KFL",
                                    "orientation": "v",
                                    "showlegend": true,
                                    "textposition": "auto",
                                    "type": "bar",
                                    "x": [
                                        "2018-10-28T00:00:00",
                                        "2018-11-28T00:00:00",
                                        "2018-12-28T00:00:00",
                                        "2019-01-28T00:00:00",
                                        "2019-02-28T00:00:00",
                                        "2019-03-28T00:00:00",
                                        "2019-04-28T00:00:00",
                                        "2019-05-28T00:00:00"
                                    ],
                                    "xaxis": "x",
                                    "y": [
                                        40,
                                        40,
                                        40,
                                        80,
                                        80,
                                        80,
                                        80,
                                        80
                                    ],
                                    "yaxis": "y"
                                },
                                {
                                    "alignmentgroup": "True",
                                    "hovertemplate": "entry_channel=KFS<br>date=%{x}<br>Ingresos=%{y}<extra></extra>",
                                    "legendgroup": "KFS",
                                    "marker": {
                                        "color": "#3d719a"
                                    },
                                    "name": "KFS",
                                    "offsetgroup": "KFS",
                                    "orientation": "v",
                                    "showlegend": true,
                                    "textposition": "auto",
                                    "type": "bar",
                                    "x": [
                                        "2018-01-28T00:00:00",
                                        "2018-02-28T00:00:00",
                                        "2018-03-28T00:00:00",
                                        "2018-04-28T00:00:00",
                                        "2018-05-28T00:00:00",
                                        "2018-06-28T00:00:00",
                                        "2018-07-28T00:00:00",
                                        "2018-08-28T00:00:00",
                                        "2018-09-28T00:00:00",
                                        "2018-10-28T00:00:00",
                                        "2018-11-28T00:00:00",
                                        "2018-12-28T00:00:00",
                                        "2019-01-28T00:00:00",
                                        "2019-02-28T00:00:00",
                                        "2019-03-28T00:00:00",
                                        "2019-04-28T00:00:00",
                                        "2019-05-28T00:00:00"
                                    ],
                                    "xaxis": "x",
                                    "y": [
                                        20,
                                        20,
                                        20,
                                        20,
                                        20,
                                        20,
                                        20,
                                        20,
                                        20,
                                        20,
                                        20,
                                        20,
                                        20,
                                        20,
                                        20,
                                        20,
                                        20
                                    ],
                                    "yaxis": "y"
                                },
                                {
                                    "alignmentgroup": "True",
                                    "hovertemplate": "entry_channel=KGC<br>date=%{x}<br>Ingresos=%{y}<extra></extra>",
                                    "legendgroup": "KGC",
                                    "marker": {
                                        "color": "#284ec8"
                                    },
                                    "name": "KGC",
                                    "offsetgroup": "KGC",
                                    "orientation": "v",
                                    "showlegend": true,
                                    "textposition": "auto",
                                    "type": "bar",
                                    "x": [
                                        "2018-01-28T00:00:00",
                                        "2018-02-28T00:00:00",
                                        "2018-03-28T00:00:00",
                                        "2018-04-28T00:00:00",
                                        "2018-05-28T00:00:00",
                                        "2018-06-28T00:00:00",
                                        "2018-07-28T00:00:00",
                                        "2018-08-28T00:00:00",
                                        "2018-09-28T00:00:00",
                                        "2018-10-28T00:00:00",
                                        "2018-11-28T00:00:00",
                                        "2018-12-28T00:00:00",
                                        "2019-01-28T00:00:00",
                                        "2019-02-28T00:00:00",
                                        "2019-03-28T00:00:00",
                                        "2019-04-28T00:00:00",
                                        "2019-05-28T00:00:00"
                                    ],
                                    "xaxis": "x",
                                    "y": [
                                        10,
                                        10,
                                        10,
                                        10,
                                        10,
                                        10,
                                        10,
                                        10,
                                        10,
                                        10,
                                        10,
                                        10,
                                        10,
                                        10,
                                        10,
                                        10,
                                        10
                                    ],
                                    "yaxis": "y"
                                },
                                {
                                    "alignmentgroup": "True",
                                    "hovertemplate": "entry_channel=KGN<br>date=%{x}<br>Ingresos=%{y}<extra></extra>",
                                    "legendgroup": "KGN",
                                    "marker": {
                                        "color": "#2e21ea"
                                    },
                                    "name": "KGN",
                                    "offsetgroup": "KGN",
                                    "orientation": "v",
                                    "showlegend": true,
                                    "textposition": "auto",
                                    "type": "bar",
                                    "x": [
                                        "2018-01-28T00:00:00",
                                        "2018-02-28T00:00:00",
                                        "2018-03-28T00:00:00",
                                        "2018-04-28T00:00:00",
                                        "2018-05-28T00:00:00",
                                        "2018-06-28T00:00:00",
                                        "2018-07-28T00:00:00",
                                        "2018-08-28T00:00:00",
                                        "2018-09-28T00:00:00",
                                        "2018-10-28T00:00:00",
                                        "2018-11-28T00:00:00",
                                        "2018-12-28T00:00:00",
                                        "2019-01-28T00:00:00",
                                        "2019-02-28T00:00:00",
                                        "2019-03-28T00:00:00",
                                        "2019-04-28T00:00:00",
                                        "2019-05-28T00:00:00"
                                    ],
                                    "xaxis": "x",
                                    "y": [
                                        20,
                                        20,
                                        20,
                                        20,
                                        20,
                                        20,
                                        20,
                                        20,
                                        20,
                                        20,
                                        20,
                                        20,
                                        20,
                                        20,
                                        20,
                                        20,
                                        10
                                    ],
                                    "yaxis": "y"
                                },
                                {
                                    "alignmentgroup": "True",
                                    "hovertemplate": "entry_channel=KGX<br>date=%{x}<br>Ingresos=%{y}<extra></extra>",
                                    "legendgroup": "KGX",
                                    "marker": {
                                        "color": "#6324f5"
                                    },
                                    "name": "KGX",
                                    "offsetgroup": "KGX",
                                    "orientation": "v",
                                    "showlegend": true,
                                    "textposition": "auto",
                                    "type": "bar",
                                    "x": [
                                        "2018-01-28T00:00:00",
                                        "2018-02-28T00:00:00",
                                        "2018-03-28T00:00:00",
                                        "2018-04-28T00:00:00",
                                        "2018-05-28T00:00:00",
                                        "2018-06-28T00:00:00"
                                    ],
                                    "xaxis": "x",
                                    "y": [
                                        10,
                                        10,
                                        20,
                                        10,
                                        10,
                                        10
                                    ],
                                    "yaxis": "y"
                                },
                                {
                                    "alignmentgroup": "True",
                                    "hovertemplate": "entry_channel=KHC<br>date=%{x}<br>Ingresos=%{y}<extra></extra>",
                                    "legendgroup": "KHC",
                                    "marker": {
                                        "color": "#9139fa"
                                    },
                                    "name": "KHC",
                                    "offsetgroup": "KHC",
                                    "orientation": "v",
                                    "showlegend": true,
                                    "textposition": "auto",
                                    "type": "bar",
                                    "x": [
                                        "2018-01-28T00:00:00",
                                        "2018-02-28T00:00:00",
                                        "2018-03-28T00:00:00",
                                        "2018-04-28T00:00:00",
                                        "2018-05-28T00:00:00",
                                        "2018-06-28T00:00:00",
                                        "2018-07-28T00:00:00",
                                        "2018-08-28T00:00:00",
                                        "2018-09-28T00:00:00",
                                        "2018-10-28T00:00:00",
                                        "2018-11-28T00:00:00",
                                        "2018-12-28T00:00:00",
                                        "2019-01-28T00:00:00",
                                        "2019-02-28T00:00:00",
                                        "2019-03-28T00:00:00",
                                        "2019-04-28T00:00:00",
                                        "2019-05-28T00:00:00"
                                    ],
                                    "xaxis": "x",
                                    "y": [
                                        7880,
                                        8020,
                                        7920,
                                        8170,
                                        7900,
                                        8240,
                                        8360,
                                        7880,
                                        7770,
                                        8150,
                                        8200,
                                        8140,
                                        7810,
                                        7710,
                                        7650,
                                        7650,
                                        7740
                                    ],
                                    "yaxis": "y"
                                },
                                {
                                    "alignmentgroup": "True",
                                    "hovertemplate": "entry_channel=KHD<br>date=%{x}<br>Ingresos=%{y}<extra></extra>",
                                    "legendgroup": "KHD",
                                    "marker": {
                                        "color": "#c543fa"
                                    },
                                    "name": "KHD",
                                    "offsetgroup": "KHD",
                                    "orientation": "v",
                                    "showlegend": true,
                                    "textposition": "auto",
                                    "type": "bar",
                                    "x": [
                                        "2018-01-28T00:00:00",
                                        "2018-02-28T00:00:00",
                                        "2018-03-28T00:00:00",
                                        "2018-04-28T00:00:00",
                                        "2018-05-28T00:00:00",
                                        "2018-06-28T00:00:00",
                                        "2018-07-28T00:00:00",
                                        "2018-08-28T00:00:00",
                                        "2018-09-28T00:00:00",
                                        "2018-10-28T00:00:00",
                                        "2018-11-28T00:00:00",
                                        "2018-12-28T00:00:00",
                                        "2019-01-28T00:00:00",
                                        "2019-02-28T00:00:00",
                                        "2019-03-28T00:00:00",
                                        "2019-04-28T00:00:00",
                                        "2019-05-28T00:00:00"
                                    ],
                                    "xaxis": "x",
                                    "y": [
                                        39790,
                                        39820,
                                        39880,
                                        40070,
                                        39940,
                                        40140,
                                        40950,
                                        40070,
                                        40370,
                                        40000,
                                        40210,
                                        40610,
                                        39770,
                                        39970,
                                        40230,
                                        40080,
                                        40440
                                    ],
                                    "yaxis": "y"
                                },
                                {
                                    "alignmentgroup": "True",
                                    "hovertemplate": "entry_channel=KHE<br>date=%{x}<br>Ingresos=%{y}<extra></extra>",
                                    "legendgroup": "KHE",
                                    "marker": {
                                        "color": "#ef55f1"
                                    },
                                    "name": "KHE",
                                    "offsetgroup": "KHE",
                                    "orientation": "v",
                                    "showlegend": true,
                                    "textposition": "auto",
                                    "type": "bar",
                                    "x": [
                                        "2018-01-28T00:00:00",
                                        "2018-02-28T00:00:00",
                                        "2018-03-28T00:00:00",
                                        "2018-04-28T00:00:00",
                                        "2018-05-28T00:00:00",
                                        "2018-06-28T00:00:00",
                                        "2018-07-28T00:00:00",
                                        "2018-08-28T00:00:00",
                                        "2018-09-28T00:00:00",
                                        "2018-10-28T00:00:00",
                                        "2018-11-28T00:00:00",
                                        "2018-12-28T00:00:00",
                                        "2019-01-28T00:00:00",
                                        "2019-02-28T00:00:00",
                                        "2019-03-28T00:00:00",
                                        "2019-04-28T00:00:00",
                                        "2019-05-28T00:00:00"
                                    ],
                                    "xaxis": "x",
                                    "y": [
                                        1713680,
                                        1720080,
                                        1727050,
                                        1729130,
                                        1728830,
                                        1739130,
                                        1748510,
                                        1738590,
                                        1741890,
                                        1750940,
                                        1754250,
                                        1769690,
                                        1739260,
                                        1758710,
                                        1764520,
                                        1763200,
                                        1769380
                                    ],
                                    "yaxis": "y"
                                },
                                {
                                    "alignmentgroup": "True",
                                    "hovertemplate": "entry_channel=KHF<br>date=%{x}<br>Ingresos=%{y}<extra></extra>",
                                    "legendgroup": "KHF",
                                    "marker": {
                                        "color": "#ef55f1"
                                    },
                                    "name": "KHF",
                                    "offsetgroup": "KHF",
                                    "orientation": "v",
                                    "showlegend": true,
                                    "textposition": "auto",
                                    "type": "bar",
                                    "x": [
                                        "2018-01-28T00:00:00",
                                        "2018-02-28T00:00:00",
                                        "2018-03-28T00:00:00",
                                        "2018-04-28T00:00:00",
                                        "2018-05-28T00:00:00",
                                        "2018-06-28T00:00:00",
                                        "2018-07-28T00:00:00",
                                        "2018-08-28T00:00:00",
                                        "2018-09-28T00:00:00",
                                        "2018-10-28T00:00:00",
                                        "2018-11-28T00:00:00",
                                        "2018-12-28T00:00:00",
                                        "2019-01-28T00:00:00",
                                        "2019-02-28T00:00:00",
                                        "2019-03-28T00:00:00",
                                        "2019-04-28T00:00:00",
                                        "2019-05-28T00:00:00"
                                    ],
                                    "xaxis": "x",
                                    "y": [
                                        9600,
                                        10320,
                                        11040,
                                        12110,
                                        11460,
                                        12430,
                                        12030,
                                        11790,
                                        11670,
                                        11660,
                                        11500,
                                        11910,
                                        11470,
                                        11730,
                                        12140,
                                        11960,
                                        12120
                                    ],
                                    "yaxis": "y"
                                },
                                {
                                    "alignmentgroup": "True",
                                    "hovertemplate": "entry_channel=KHK<br>date=%{x}<br>Ingresos=%{y}<extra></extra>",
                                    "legendgroup": "KHK",
                                    "marker": {
                                        "color": "#fb84ce"
                                    },
                                    "name": "KHK",
                                    "offsetgroup": "KHK",
                                    "orientation": "v",
                                    "showlegend": true,
                                    "textposition": "auto",
                                    "type": "bar",
                                    "x": [
                                        "2018-01-28T00:00:00",
                                        "2018-02-28T00:00:00",
                                        "2018-03-28T00:00:00",
                                        "2018-04-28T00:00:00",
                                        "2018-05-28T00:00:00",
                                        "2018-06-28T00:00:00",
                                        "2018-07-28T00:00:00",
                                        "2018-08-28T00:00:00",
                                        "2018-09-28T00:00:00",
                                        "2018-10-28T00:00:00",
                                        "2018-11-28T00:00:00",
                                        "2018-12-28T00:00:00",
                                        "2019-01-28T00:00:00",
                                        "2019-02-28T00:00:00",
                                        "2019-03-28T00:00:00",
                                        "2019-04-28T00:00:00",
                                        "2019-05-28T00:00:00"
                                    ],
                                    "xaxis": "x",
                                    "y": [
                                        92030,
                                        139260,
                                        181860,
                                        213000,
                                        227550,
                                        234620,
                                        240350,
                                        235340,
                                        237200,
                                        243250,
                                        245050,
                                        252510,
                                        241890,
                                        250920,
                                        253060,
                                        252150,
                                        252190
                                    ],
                                    "yaxis": "y"
                                },
                                {
                                    "alignmentgroup": "True",
                                    "hovertemplate": "entry_channel=KHL<br>date=%{x}<br>Ingresos=%{y}<extra></extra>",
                                    "legendgroup": "KHL",
                                    "marker": {
                                        "color": "#fbafa1"
                                    },
                                    "name": "KHL",
                                    "offsetgroup": "KHL",
                                    "orientation": "v",
                                    "showlegend": true,
                                    "textposition": "auto",
                                    "type": "bar",
                                    "x": [
                                        "2018-01-28T00:00:00",
                                        "2018-02-28T00:00:00",
                                        "2018-03-28T00:00:00",
                                        "2018-04-28T00:00:00",
                                        "2018-05-28T00:00:00",
                                        "2018-06-28T00:00:00",
                                        "2018-07-28T00:00:00",
                                        "2018-08-28T00:00:00",
                                        "2018-09-28T00:00:00",
                                        "2018-10-28T00:00:00",
                                        "2018-11-28T00:00:00",
                                        "2018-12-28T00:00:00",
                                        "2019-01-28T00:00:00",
                                        "2019-02-28T00:00:00",
                                        "2019-03-28T00:00:00",
                                        "2019-04-28T00:00:00",
                                        "2019-05-28T00:00:00"
                                    ],
                                    "xaxis": "x",
                                    "y": [
                                        7340,
                                        20730,
                                        38380,
                                        46570,
                                        48250,
                                        49270,
                                        51410,
                                        50480,
                                        51960,
                                        52830,
                                        52690,
                                        54300,
                                        53360,
                                        54810,
                                        54350,
                                        53310,
                                        53850
                                    ],
                                    "yaxis": "y"
                                },
                                {
                                    "alignmentgroup": "True",
                                    "hovertemplate": "entry_channel=KHM<br>date=%{x}<br>Ingresos=%{y}<extra></extra>",
                                    "legendgroup": "KHM",
                                    "marker": {
                                        "color": "#fcd471"
                                    },
                                    "name": "KHM",
                                    "offsetgroup": "KHM",
                                    "orientation": "v",
                                    "showlegend": true,
                                    "textposition": "auto",
                                    "type": "bar",
                                    "x": [
                                        "2018-01-28T00:00:00",
                                        "2018-02-28T00:00:00",
                                        "2018-03-28T00:00:00",
                                        "2018-04-28T00:00:00",
                                        "2018-05-28T00:00:00",
                                        "2018-06-28T00:00:00",
                                        "2018-07-28T00:00:00",
                                        "2018-08-28T00:00:00",
                                        "2018-09-28T00:00:00",
                                        "2018-10-28T00:00:00",
                                        "2018-11-28T00:00:00",
                                        "2018-12-28T00:00:00",
                                        "2019-01-28T00:00:00",
                                        "2019-02-28T00:00:00",
                                        "2019-03-28T00:00:00",
                                        "2019-04-28T00:00:00",
                                        "2019-05-28T00:00:00"
                                    ],
                                    "xaxis": "x",
                                    "y": [
                                        280,
                                        150,
                                        30,
                                        920,
                                        6740,
                                        27070,
                                        43900,
                                        76960,
                                        111170,
                                        149550,
                                        186940,
                                        227350,
                                        242970,
                                        299430,
                                        349060,
                                        389050,
                                        429300
                                    ],
                                    "yaxis": "y"
                                },
                                {
                                    "alignmentgroup": "True",
                                    "hovertemplate": "entry_channel=KHN<br>date=%{x}<br>Ingresos=%{y}<extra></extra>",
                                    "legendgroup": "KHN",
                                    "marker": {
                                        "color": "#f0ed35"
                                    },
                                    "name": "KHN",
                                    "offsetgroup": "KHN",
                                    "orientation": "v",
                                    "showlegend": true,
                                    "textposition": "auto",
                                    "type": "bar",
                                    "x": [
                                        "2018-01-28T00:00:00",
                                        "2018-02-28T00:00:00",
                                        "2018-03-28T00:00:00",
                                        "2018-04-28T00:00:00",
                                        "2018-05-28T00:00:00",
                                        "2018-06-28T00:00:00",
                                        "2018-07-28T00:00:00",
                                        "2018-08-28T00:00:00",
                                        "2018-09-28T00:00:00",
                                        "2018-10-28T00:00:00",
                                        "2018-11-28T00:00:00",
                                        "2018-12-28T00:00:00",
                                        "2019-01-28T00:00:00",
                                        "2019-02-28T00:00:00",
                                        "2019-03-28T00:00:00",
                                        "2019-04-28T00:00:00",
                                        "2019-05-28T00:00:00"
                                    ],
                                    "xaxis": "x",
                                    "y": [
                                        110,
                                        60,
                                        690,
                                        9390,
                                        26300,
                                        46810,
                                        57680,
                                        76300,
                                        98300,
                                        123690,
                                        147260,
                                        167660,
                                        180850,
                                        217020,
                                        241400,
                                        261520,
                                        285770
                                    ],
                                    "yaxis": "y"
                                },
                                {
                                    "alignmentgroup": "True",
                                    "hovertemplate": "entry_channel=KHO<br>date=%{x}<br>Ingresos=%{y}<extra></extra>",
                                    "legendgroup": "KHO",
                                    "marker": {
                                        "color": "#c6e516"
                                    },
                                    "name": "KHO",
                                    "offsetgroup": "KHO",
                                    "orientation": "v",
                                    "showlegend": true,
                                    "textposition": "auto",
                                    "type": "bar",
                                    "x": [
                                        "2018-03-28T00:00:00",
                                        "2018-04-28T00:00:00",
                                        "2018-05-28T00:00:00",
                                        "2018-06-28T00:00:00",
                                        "2018-07-28T00:00:00",
                                        "2018-08-28T00:00:00",
                                        "2018-09-28T00:00:00",
                                        "2018-10-28T00:00:00",
                                        "2018-11-28T00:00:00",
                                        "2018-12-28T00:00:00",
                                        "2019-01-28T00:00:00",
                                        "2019-02-28T00:00:00",
                                        "2019-03-28T00:00:00",
                                        "2019-04-28T00:00:00",
                                        "2019-05-28T00:00:00"
                                    ],
                                    "xaxis": "x",
                                    "y": [
                                        410,
                                        1230,
                                        2080,
                                        3600,
                                        4480,
                                        5940,
                                        7970,
                                        10080,
                                        11910,
                                        15000,
                                        16490,
                                        19530,
                                        22050,
                                        24540,
                                        26830
                                    ],
                                    "yaxis": "y"
                                },
                                {
                                    "alignmentgroup": "True",
                                    "hovertemplate": "entry_channel=KHP<br>date=%{x}<br>Ingresos=%{y}<extra></extra>",
                                    "legendgroup": "KHP",
                                    "marker": {
                                        "color": "#96d310"
                                    },
                                    "name": "KHP",
                                    "offsetgroup": "KHP",
                                    "orientation": "v",
                                    "showlegend": true,
                                    "textposition": "auto",
                                    "type": "bar",
                                    "x": [
                                        "2018-08-28T00:00:00",
                                        "2018-09-28T00:00:00",
                                        "2018-10-28T00:00:00",
                                        "2018-11-28T00:00:00",
                                        "2018-12-28T00:00:00",
                                        "2019-01-28T00:00:00",
                                        "2019-02-28T00:00:00",
                                        "2019-03-28T00:00:00",
                                        "2019-04-28T00:00:00",
                                        "2019-05-28T00:00:00"
                                    ],
                                    "xaxis": "x",
                                    "y": [
                                        160,
                                        260,
                                        530,
                                        690,
                                        800,
                                        770,
                                        820,
                                        820,
                                        810,
                                        810
                                    ],
                                    "yaxis": "y"
                                },
                                {
                                    "alignmentgroup": "True",
                                    "hovertemplate": "entry_channel=KHQ<br>date=%{x}<br>Ingresos=%{y}<extra></extra>",
                                    "legendgroup": "KHQ",
                                    "marker": {
                                        "color": "#61c10b"
                                    },
                                    "name": "KHQ",
                                    "offsetgroup": "KHQ",
                                    "orientation": "v",
                                    "showlegend": true,
                                    "textposition": "auto",
                                    "type": "bar",
                                    "x": [
                                        "2018-03-28T00:00:00",
                                        "2018-04-28T00:00:00",
                                        "2018-05-28T00:00:00",
                                        "2018-06-28T00:00:00",
                                        "2018-07-28T00:00:00",
                                        "2018-08-28T00:00:00",
                                        "2018-09-28T00:00:00",
                                        "2018-10-28T00:00:00",
                                        "2018-11-28T00:00:00",
                                        "2018-12-28T00:00:00",
                                        "2019-01-28T00:00:00",
                                        "2019-02-28T00:00:00",
                                        "2019-03-28T00:00:00",
                                        "2019-04-28T00:00:00",
                                        "2019-05-28T00:00:00"
                                    ],
                                    "xaxis": "x",
                                    "y": [
                                        10,
                                        10,
                                        10,
                                        10,
                                        10,
                                        116280,
                                        206480,
                                        354700,
                                        496590,
                                        562580,
                                        568350,
                                        569340,
                                        572090,
                                        573080,
                                        574410
                                    ],
                                    "yaxis": "y"
                                },
                                {
                                    "alignmentgroup": "True",
                                    "hovertemplate": "entry_channel=RED<br>date=%{x}<br>Ingresos=%{y}<extra></extra>",
                                    "legendgroup": "RED",
                                    "marker": {
                                        "color": "#31ac28"
                                    },
                                    "name": "RED",
                                    "offsetgroup": "RED",
                                    "orientation": "v",
                                    "showlegend": true,
                                    "textposition": "auto",
                                    "type": "bar",
                                    "x": [
                                        "2018-01-28T00:00:00",
                                        "2018-02-28T00:00:00",
                                        "2018-03-28T00:00:00",
                                        "2018-04-28T00:00:00",
                                        "2018-05-28T00:00:00",
                                        "2018-06-28T00:00:00",
                                        "2018-07-28T00:00:00",
                                        "2018-08-28T00:00:00",
                                        "2018-09-28T00:00:00",
                                        "2018-10-28T00:00:00",
                                        "2018-11-28T00:00:00",
                                        "2018-12-28T00:00:00",
                                        "2019-01-28T00:00:00",
                                        "2019-02-28T00:00:00",
                                        "2019-03-28T00:00:00",
                                        "2019-04-28T00:00:00",
                                        "2019-05-28T00:00:00"
                                    ],
                                    "xaxis": "x",
                                    "y": [
                                        54800,
                                        61640,
                                        66100,
                                        71210,
                                        73110,
                                        76820,
                                        80230,
                                        82390,
                                        89780,
                                        99250,
                                        104900,
                                        112970,
                                        111500,
                                        120210,
                                        128870,
                                        132740,
                                        140670
                                    ],
                                    "yaxis": "y"
                                }
                            ],
                            "layout": {
                                "barmode": "relative",
                                "legend": {
                                    "title": {
                                        "text": "entry_channel"
                                    },
                                    "tracegroupgap": 0
                                },
                                "margin": {
                                    "t": 60
                                },
                                "template": {
                                    "data": {
                                        "bar": [
                                            {
                                                "error_x": {
                                                    "color": "#2a3f5f"
                                                },
                                                "error_y": {
                                                    "color": "#2a3f5f"
                                                },
                                                "marker": {
                                                    "line": {
                                                        "color": "#E5ECF6",
                                                        "width": 0.5
                                                    }
                                                },
                                                "type": "bar"
                                            }
                                        ],
                                        "barpolar": [
                                            {
                                                "marker": {
                                                    "line": {
                                                        "color": "#E5ECF6",
                                                        "width": 0.5
                                                    }
                                                },
                                                "type": "barpolar"
                                            }
                                        ],
                                        "carpet": [
                                            {
                                                "aaxis": {
                                                    "endlinecolor": "#2a3f5f",
                                                    "gridcolor": "white",
                                                    "linecolor": "white",
                                                    "minorgridcolor": "white",
                                                    "startlinecolor": "#2a3f5f"
                                                },
                                                "baxis": {
                                                    "endlinecolor": "#2a3f5f",
                                                    "gridcolor": "white",
                                                    "linecolor": "white",
                                                    "minorgridcolor": "white",
                                                    "startlinecolor": "#2a3f5f"
                                                },
                                                "type": "carpet"
                                            }
                                        ],
                                        "choropleth": [
                                            {
                                                "colorbar": {
                                                    "outlinewidth": 0,
                                                    "ticks": ""
                                                },
                                                "type": "choropleth"
                                            }
                                        ],
                                        "contour": [
                                            {
                                                "colorbar": {
                                                    "outlinewidth": 0,
                                                    "ticks": ""
                                                },
                                                "colorscale": [
                                                    [
                                                        0,
                                                        "#0d0887"
                                                    ],
                                                    [
                                                        0.1111111111111111,
                                                        "#46039f"
                                                    ],
                                                    [
                                                        0.2222222222222222,
                                                        "#7201a8"
                                                    ],
                                                    [
                                                        0.3333333333333333,
                                                        "#9c179e"
                                                    ],
                                                    [
                                                        0.4444444444444444,
                                                        "#bd3786"
                                                    ],
                                                    [
                                                        0.5555555555555556,
                                                        "#d8576b"
                                                    ],
                                                    [
                                                        0.6666666666666666,
                                                        "#ed7953"
                                                    ],
                                                    [
                                                        0.7777777777777778,
                                                        "#fb9f3a"
                                                    ],
                                                    [
                                                        0.8888888888888888,
                                                        "#fdca26"
                                                    ],
                                                    [
                                                        1,
                                                        "#f0f921"
                                                    ]
                                                ],
                                                "type": "contour"
                                            }
                                        ],
                                        "contourcarpet": [
                                            {
                                                "colorbar": {
                                                    "outlinewidth": 0,
                                                    "ticks": ""
                                                },
                                                "type": "contourcarpet"
                                            }
                                        ],
                                        "heatmap": [
                                            {
                                                "colorbar": {
                                                    "outlinewidth": 0,
                                                    "ticks": ""
                                                },
                                                "colorscale": [
                                                    [
                                                        0,
                                                        "#0d0887"
                                                    ],
                                                    [
                                                        0.1111111111111111,
                                                        "#46039f"
                                                    ],
                                                    [
                                                        0.2222222222222222,
                                                        "#7201a8"
                                                    ],
                                                    [
                                                        0.3333333333333333,
                                                        "#9c179e"
                                                    ],
                                                    [
                                                        0.4444444444444444,
                                                        "#bd3786"
                                                    ],
                                                    [
                                                        0.5555555555555556,
                                                        "#d8576b"
                                                    ],
                                                    [
                                                        0.6666666666666666,
                                                        "#ed7953"
                                                    ],
                                                    [
                                                        0.7777777777777778,
                                                        "#fb9f3a"
                                                    ],
                                                    [
                                                        0.8888888888888888,
                                                        "#fdca26"
                                                    ],
                                                    [
                                                        1,
                                                        "#f0f921"
                                                    ]
                                                ],
                                                "type": "heatmap"
                                            }
                                        ],
                                        "heatmapgl": [
                                            {
                                                "colorbar": {
                                                    "outlinewidth": 0,
                                                    "ticks": ""
                                                },
                                                "colorscale": [
                                                    [
                                                        0,
                                                        "#0d0887"
                                                    ],
                                                    [
                                                        0.1111111111111111,
                                                        "#46039f"
                                                    ],
                                                    [
                                                        0.2222222222222222,
                                                        "#7201a8"
                                                    ],
                                                    [
                                                        0.3333333333333333,
                                                        "#9c179e"
                                                    ],
                                                    [
                                                        0.4444444444444444,
                                                        "#bd3786"
                                                    ],
                                                    [
                                                        0.5555555555555556,
                                                        "#d8576b"
                                                    ],
                                                    [
                                                        0.6666666666666666,
                                                        "#ed7953"
                                                    ],
                                                    [
                                                        0.7777777777777778,
                                                        "#fb9f3a"
                                                    ],
                                                    [
                                                        0.8888888888888888,
                                                        "#fdca26"
                                                    ],
                                                    [
                                                        1,
                                                        "#f0f921"
                                                    ]
                                                ],
                                                "type": "heatmapgl"
                                            }
                                        ],
                                        "histogram": [
                                            {
                                                "marker": {
                                                    "colorbar": {
                                                        "outlinewidth": 0,
                                                        "ticks": ""
                                                    }
                                                },
                                                "type": "histogram"
                                            }
                                        ],
                                        "histogram2d": [
                                            {
                                                "colorbar": {
                                                    "outlinewidth": 0,
                                                    "ticks": ""
                                                },
                                                "colorscale": [
                                                    [
                                                        0,
                                                        "#0d0887"
                                                    ],
                                                    [
                                                        0.1111111111111111,
                                                        "#46039f"
                                                    ],
                                                    [
                                                        0.2222222222222222,
                                                        "#7201a8"
                                                    ],
                                                    [
                                                        0.3333333333333333,
                                                        "#9c179e"
                                                    ],
                                                    [
                                                        0.4444444444444444,
                                                        "#bd3786"
                                                    ],
                                                    [
                                                        0.5555555555555556,
                                                        "#d8576b"
                                                    ],
                                                    [
                                                        0.6666666666666666,
                                                        "#ed7953"
                                                    ],
                                                    [
                                                        0.7777777777777778,
                                                        "#fb9f3a"
                                                    ],
                                                    [
                                                        0.8888888888888888,
                                                        "#fdca26"
                                                    ],
                                                    [
                                                        1,
                                                        "#f0f921"
                                                    ]
                                                ],
                                                "type": "histogram2d"
                                            }
                                        ],
                                        "histogram2dcontour": [
                                            {
                                                "colorbar": {
                                                    "outlinewidth": 0,
                                                    "ticks": ""
                                                },
                                                "colorscale": [
                                                    [
                                                        0,
                                                        "#0d0887"
                                                    ],
                                                    [
                                                        0.1111111111111111,
                                                        "#46039f"
                                                    ],
                                                    [
                                                        0.2222222222222222,
                                                        "#7201a8"
                                                    ],
                                                    [
                                                        0.3333333333333333,
                                                        "#9c179e"
                                                    ],
                                                    [
                                                        0.4444444444444444,
                                                        "#bd3786"
                                                    ],
                                                    [
                                                        0.5555555555555556,
                                                        "#d8576b"
                                                    ],
                                                    [
                                                        0.6666666666666666,
                                                        "#ed7953"
                                                    ],
                                                    [
                                                        0.7777777777777778,
                                                        "#fb9f3a"
                                                    ],
                                                    [
                                                        0.8888888888888888,
                                                        "#fdca26"
                                                    ],
                                                    [
                                                        1,
                                                        "#f0f921"
                                                    ]
                                                ],
                                                "type": "histogram2dcontour"
                                            }
                                        ],
                                        "mesh3d": [
                                            {
                                                "colorbar": {
                                                    "outlinewidth": 0,
                                                    "ticks": ""
                                                },
                                                "type": "mesh3d"
                                            }
                                        ],
                                        "parcoords": [
                                            {
                                                "line": {
                                                    "colorbar": {
                                                        "outlinewidth": 0,
                                                        "ticks": ""
                                                    }
                                                },
                                                "type": "parcoords"
                                            }
                                        ],
                                        "pie": [
                                            {
                                                "automargin": true,
                                                "type": "pie"
                                            }
                                        ],
                                        "scatter": [
                                            {
                                                "marker": {
                                                    "colorbar": {
                                                        "outlinewidth": 0,
                                                        "ticks": ""
                                                    }
                                                },
                                                "type": "scatter"
                                            }
                                        ],
                                        "scatter3d": [
                                            {
                                                "line": {
                                                    "colorbar": {
                                                        "outlinewidth": 0,
                                                        "ticks": ""
                                                    }
                                                },
                                                "marker": {
                                                    "colorbar": {
                                                        "outlinewidth": 0,
                                                        "ticks": ""
                                                    }
                                                },
                                                "type": "scatter3d"
                                            }
                                        ],
                                        "scattercarpet": [
                                            {
                                                "marker": {
                                                    "colorbar": {
                                                        "outlinewidth": 0,
                                                        "ticks": ""
                                                    }
                                                },
                                                "type": "scattercarpet"
                                            }
                                        ],
                                        "scattergeo": [
                                            {
                                                "marker": {
                                                    "colorbar": {
                                                        "outlinewidth": 0,
                                                        "ticks": ""
                                                    }
                                                },
                                                "type": "scattergeo"
                                            }
                                        ],
                                        "scattergl": [
                                            {
                                                "marker": {
                                                    "colorbar": {
                                                        "outlinewidth": 0,
                                                        "ticks": ""
                                                    }
                                                },
                                                "type": "scattergl"
                                            }
                                        ],
                                        "scattermapbox": [
                                            {
                                                "marker": {
                                                    "colorbar": {
                                                        "outlinewidth": 0,
                                                        "ticks": ""
                                                    }
                                                },
                                                "type": "scattermapbox"
                                            }
                                        ],
                                        "scatterpolar": [
                                            {
                                                "marker": {
                                                    "colorbar": {
                                                        "outlinewidth": 0,
                                                        "ticks": ""
                                                    }
                                                },
                                                "type": "scatterpolar"
                                            }
                                        ],
                                        "scatterpolargl": [
                                            {
                                                "marker": {
                                                    "colorbar": {
                                                        "outlinewidth": 0,
                                                        "ticks": ""
                                                    }
                                                },
                                                "type": "scatterpolargl"
                                            }
                                        ],
                                        "scatterternary": [
                                            {
                                                "marker": {
                                                    "colorbar": {
                                                        "outlinewidth": 0,
                                                        "ticks": ""
                                                    }
                                                },
                                                "type": "scatterternary"
                                            }
                                        ],
                                        "surface": [
                                            {
                                                "colorbar": {
                                                    "outlinewidth": 0,
                                                    "ticks": ""
                                                },
                                                "colorscale": [
                                                    [
                                                        0,
                                                        "#0d0887"
                                                    ],
                                                    [
                                                        0.1111111111111111,
                                                        "#46039f"
                                                    ],
                                                    [
                                                        0.2222222222222222,
                                                        "#7201a8"
                                                    ],
                                                    [
                                                        0.3333333333333333,
                                                        "#9c179e"
                                                    ],
                                                    [
                                                        0.4444444444444444,
                                                        "#bd3786"
                                                    ],
                                                    [
                                                        0.5555555555555556,
                                                        "#d8576b"
                                                    ],
                                                    [
                                                        0.6666666666666666,
                                                        "#ed7953"
                                                    ],
                                                    [
                                                        0.7777777777777778,
                                                        "#fb9f3a"
                                                    ],
                                                    [
                                                        0.8888888888888888,
                                                        "#fdca26"
                                                    ],
                                                    [
                                                        1,
                                                        "#f0f921"
                                                    ]
                                                ],
                                                "type": "surface"
                                            }
                                        ],
                                        "table": [
                                            {
                                                "cells": {
                                                    "fill": {
                                                        "color": "#EBF0F8"
                                                    },
                                                    "line": {
                                                        "color": "white"
                                                    }
                                                },
                                                "header": {
                                                    "fill": {
                                                        "color": "#C8D4E3"
                                                    },
                                                    "line": {
                                                        "color": "white"
                                                    }
                                                },
                                                "type": "table"
                                            }
                                        ]
                                    },
                                    "layout": {
                                        "annotationdefaults": {
                                            "arrowcolor": "#2a3f5f",
                                            "arrowhead": 0,
                                            "arrowwidth": 1
                                        },
                                        "autotypenumbers": "strict",
                                        "coloraxis": {
                                            "colorbar": {
                                                "outlinewidth": 0,
                                                "ticks": ""
                                            }
                                        },
                                        "colorscale": {
                                            "diverging": [
                                                [
                                                    0,
                                                    "#8e0152"
                                                ],
                                                [
                                                    0.1,
                                                    "#c51b7d"
                                                ],
                                                [
                                                    0.2,
                                                    "#de77ae"
                                                ],
                                                [
                                                    0.3,
                                                    "#f1b6da"
                                                ],
                                                [
                                                    0.4,
                                                    "#fde0ef"
                                                ],
                                                [
                                                    0.5,
                                                    "#f7f7f7"
                                                ],
                                                [
                                                    0.6,
                                                    "#e6f5d0"
                                                ],
                                                [
                                                    0.7,
                                                    "#b8e186"
                                                ],
                                                [
                                                    0.8,
                                                    "#7fbc41"
                                                ],
                                                [
                                                    0.9,
                                                    "#4d9221"
                                                ],
                                                [
                                                    1,
                                                    "#276419"
                                                ]
                                            ],
                                            "sequential": [
                                                [
                                                    0,
                                                    "#0d0887"
                                                ],
                                                [
                                                    0.1111111111111111,
                                                    "#46039f"
                                                ],
                                                [
                                                    0.2222222222222222,
                                                    "#7201a8"
                                                ],
                                                [
                                                    0.3333333333333333,
                                                    "#9c179e"
                                                ],
                                                [
                                                    0.4444444444444444,
                                                    "#bd3786"
                                                ],
                                                [
                                                    0.5555555555555556,
                                                    "#d8576b"
                                                ],
                                                [
                                                    0.6666666666666666,
                                                    "#ed7953"
                                                ],
                                                [
                                                    0.7777777777777778,
                                                    "#fb9f3a"
                                                ],
                                                [
                                                    0.8888888888888888,
                                                    "#fdca26"
                                                ],
                                                [
                                                    1,
                                                    "#f0f921"
                                                ]
                                            ],
                                            "sequentialminus": [
                                                [
                                                    0,
                                                    "#0d0887"
                                                ],
                                                [
                                                    0.1111111111111111,
                                                    "#46039f"
                                                ],
                                                [
                                                    0.2222222222222222,
                                                    "#7201a8"
                                                ],
                                                [
                                                    0.3333333333333333,
                                                    "#9c179e"
                                                ],
                                                [
                                                    0.4444444444444444,
                                                    "#bd3786"
                                                ],
                                                [
                                                    0.5555555555555556,
                                                    "#d8576b"
                                                ],
                                                [
                                                    0.6666666666666666,
                                                    "#ed7953"
                                                ],
                                                [
                                                    0.7777777777777778,
                                                    "#fb9f3a"
                                                ],
                                                [
                                                    0.8888888888888888,
                                                    "#fdca26"
                                                ],
                                                [
                                                    1,
                                                    "#f0f921"
                                                ]
                                            ]
                                        },
                                        "colorway": [
                                            "#636efa",
                                            "#EF553B",
                                            "#00cc96",
                                            "#ab63fa",
                                            "#FFA15A",
                                            "#19d3f3",
                                            "#FF6692",
                                            "#B6E880",
                                            "#FF97FF",
                                            "#FECB52"
                                        ],
                                        "font": {
                                            "color": "#2a3f5f"
                                        },
                                        "geo": {
                                            "bgcolor": "white",
                                            "lakecolor": "white",
                                            "landcolor": "#E5ECF6",
                                            "showlakes": true,
                                            "showland": true,
                                            "subunitcolor": "white"
                                        },
                                        "hoverlabel": {
                                            "align": "left"
                                        },
                                        "hovermode": "closest",
                                        "mapbox": {
                                            "style": "light"
                                        },
                                        "paper_bgcolor": "white",
                                        "plot_bgcolor": "#E5ECF6",
                                        "polar": {
                                            "angularaxis": {
                                                "gridcolor": "white",
                                                "linecolor": "white",
                                                "ticks": ""
                                            },
                                            "bgcolor": "#E5ECF6",
                                            "radialaxis": {
                                                "gridcolor": "white",
                                                "linecolor": "white",
                                                "ticks": ""
                                            }
                                        },
                                        "scene": {
                                            "xaxis": {
                                                "backgroundcolor": "#E5ECF6",
                                                "gridcolor": "white",
                                                "gridwidth": 2,
                                                "linecolor": "white",
                                                "showbackground": true,
                                                "ticks": "",
                                                "zerolinecolor": "white"
                                            },
                                            "yaxis": {
                                                "backgroundcolor": "#E5ECF6",
                                                "gridcolor": "white",
                                                "gridwidth": 2,
                                                "linecolor": "white",
                                                "showbackground": true,
                                                "ticks": "",
                                                "zerolinecolor": "white"
                                            },
                                            "zaxis": {
                                                "backgroundcolor": "#E5ECF6",
                                                "gridcolor": "white",
                                                "gridwidth": 2,
                                                "linecolor": "white",
                                                "showbackground": true,
                                                "ticks": "",
                                                "zerolinecolor": "white"
                                            }
                                        },
                                        "shapedefaults": {
                                            "line": {
                                                "color": "#2a3f5f"
                                            }
                                        },
                                        "ternary": {
                                            "aaxis": {
                                                "gridcolor": "white",
                                                "linecolor": "white",
                                                "ticks": ""
                                            },
                                            "baxis": {
                                                "gridcolor": "white",
                                                "linecolor": "white",
                                                "ticks": ""
                                            },
                                            "bgcolor": "#E5ECF6",
                                            "caxis": {
                                                "gridcolor": "white",
                                                "linecolor": "white",
                                                "ticks": ""
                                            }
                                        },
                                        "title": {
                                            "x": 0.05
                                        },
                                        "xaxis": {
                                            "automargin": true,
                                            "gridcolor": "white",
                                            "linecolor": "white",
                                            "ticks": "",
                                            "title": {
                                                "standoff": 15
                                            },
                                            "zerolinecolor": "white",
                                            "zerolinewidth": 2
                                        },
                                        "yaxis": {
                                            "automargin": true,
                                            "gridcolor": "white",
                                            "linecolor": "white",
                                            "ticks": "",
                                            "title": {
                                                "standoff": 15
                                            },
                                            "zerolinecolor": "white",
                                            "zerolinewidth": 2
                                        }
                                    }
                                },
                                "xaxis": {
                                    "anchor": "y",
                                    "domain": [
                                        0,
                                        1
                                    ],
                                    "title": {
                                        "text": "date"
                                    }
                                },
                                "yaxis": {
                                    "anchor": "x",
                                    "domain": [
                                        0,
                                        1
                                    ],
                                    "title": {
                                        "text": "Ingresos"
                                    }
                                }
                            }
                        }
                    },
                    "metadata": {}
                }
            ],
            "source": [
                "grafico_horizontal= full_3.groupby(['entry_channel',\"date\"])[\"Ingresos\"].sum().reset_index()\n",
                "grafico_horizontal\n",
                "\n",
                "evolucion_horizontal= px.bar(grafico_horizontal, x=\"date\", y=\"Ingresos\", \\\n",
                "                        color=\"entry_channel\", orientation=\"v\", color_discrete_sequence=px.colors.cyclical.mygbm)\n",
                "\n",
                "evolucion_horizontal.show()"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": 39,
            "metadata": {},
            "outputs": [
                {
                    "output_type": "display_data",
                    "data": {
                        "application/vnd.plotly.v1+json": {
                            "config": {
                                "plotlyServerURL": "https://plot.ly"
                            },
                            "data": [
                                {
                                    "alignmentgroup": "True",
                                    "hovertemplate": "gender=H<br>date=%{x}<br>Ingresos=%{y}<extra></extra>",
                                    "legendgroup": "H",
                                    "marker": {
                                        "color": "#ef55f1"
                                    },
                                    "name": "H",
                                    "offsetgroup": "H",
                                    "orientation": "v",
                                    "showlegend": true,
                                    "textposition": "auto",
                                    "type": "bar",
                                    "x": [
                                        "2018-01-28T00:00:00",
                                        "2018-02-28T00:00:00",
                                        "2018-03-28T00:00:00",
                                        "2018-04-28T00:00:00",
                                        "2018-05-28T00:00:00",
                                        "2018-06-28T00:00:00",
                                        "2018-07-28T00:00:00",
                                        "2018-08-28T00:00:00",
                                        "2018-09-28T00:00:00",
                                        "2018-10-28T00:00:00",
                                        "2018-11-28T00:00:00",
                                        "2018-12-28T00:00:00",
                                        "2019-01-28T00:00:00",
                                        "2019-02-28T00:00:00",
                                        "2019-03-28T00:00:00",
                                        "2019-04-28T00:00:00",
                                        "2019-05-28T00:00:00"
                                    ],
                                    "xaxis": "x",
                                    "y": [
                                        1779260,
                                        1816440,
                                        1856860,
                                        1897510,
                                        1904740,
                                        1956820,
                                        2055060,
                                        2112380,
                                        2230560,
                                        2364100,
                                        2433160,
                                        2498460,
                                        2455730,
                                        2531000,
                                        2573500,
                                        2593490,
                                        2638330
                                    ],
                                    "yaxis": "y"
                                },
                                {
                                    "alignmentgroup": "True",
                                    "hovertemplate": "gender=V<br>date=%{x}<br>Ingresos=%{y}<extra></extra>",
                                    "legendgroup": "V",
                                    "marker": {
                                        "color": "#fb84ce"
                                    },
                                    "name": "V",
                                    "offsetgroup": "V",
                                    "orientation": "v",
                                    "showlegend": true,
                                    "textposition": "auto",
                                    "type": "bar",
                                    "x": [
                                        "2018-01-28T00:00:00",
                                        "2018-02-28T00:00:00",
                                        "2018-03-28T00:00:00",
                                        "2018-04-28T00:00:00",
                                        "2018-05-28T00:00:00",
                                        "2018-06-28T00:00:00",
                                        "2018-07-28T00:00:00",
                                        "2018-08-28T00:00:00",
                                        "2018-09-28T00:00:00",
                                        "2018-10-28T00:00:00",
                                        "2018-11-28T00:00:00",
                                        "2018-12-28T00:00:00",
                                        "2019-01-28T00:00:00",
                                        "2019-02-28T00:00:00",
                                        "2019-03-28T00:00:00",
                                        "2019-04-28T00:00:00",
                                        "2019-05-28T00:00:00"
                                    ],
                                    "xaxis": "x",
                                    "y": [
                                        1844420,
                                        1902900,
                                        1962830,
                                        2010000,
                                        2025060,
                                        2087800,
                                        2188250,
                                        2242570,
                                        2356340,
                                        2478320,
                                        2540930,
                                        2612500,
                                        2553420,
                                        2662610,
                                        2712460,
                                        2743920,
                                        2788990
                                    ],
                                    "yaxis": "y"
                                }
                            ],
                            "layout": {
                                "barmode": "relative",
                                "legend": {
                                    "title": {
                                        "text": "gender"
                                    },
                                    "tracegroupgap": 0
                                },
                                "margin": {
                                    "t": 60
                                },
                                "template": {
                                    "data": {
                                        "bar": [
                                            {
                                                "error_x": {
                                                    "color": "#2a3f5f"
                                                },
                                                "error_y": {
                                                    "color": "#2a3f5f"
                                                },
                                                "marker": {
                                                    "line": {
                                                        "color": "#E5ECF6",
                                                        "width": 0.5
                                                    }
                                                },
                                                "type": "bar"
                                            }
                                        ],
                                        "barpolar": [
                                            {
                                                "marker": {
                                                    "line": {
                                                        "color": "#E5ECF6",
                                                        "width": 0.5
                                                    }
                                                },
                                                "type": "barpolar"
                                            }
                                        ],
                                        "carpet": [
                                            {
                                                "aaxis": {
                                                    "endlinecolor": "#2a3f5f",
                                                    "gridcolor": "white",
                                                    "linecolor": "white",
                                                    "minorgridcolor": "white",
                                                    "startlinecolor": "#2a3f5f"
                                                },
                                                "baxis": {
                                                    "endlinecolor": "#2a3f5f",
                                                    "gridcolor": "white",
                                                    "linecolor": "white",
                                                    "minorgridcolor": "white",
                                                    "startlinecolor": "#2a3f5f"
                                                },
                                                "type": "carpet"
                                            }
                                        ],
                                        "choropleth": [
                                            {
                                                "colorbar": {
                                                    "outlinewidth": 0,
                                                    "ticks": ""
                                                },
                                                "type": "choropleth"
                                            }
                                        ],
                                        "contour": [
                                            {
                                                "colorbar": {
                                                    "outlinewidth": 0,
                                                    "ticks": ""
                                                },
                                                "colorscale": [
                                                    [
                                                        0,
                                                        "#0d0887"
                                                    ],
                                                    [
                                                        0.1111111111111111,
                                                        "#46039f"
                                                    ],
                                                    [
                                                        0.2222222222222222,
                                                        "#7201a8"
                                                    ],
                                                    [
                                                        0.3333333333333333,
                                                        "#9c179e"
                                                    ],
                                                    [
                                                        0.4444444444444444,
                                                        "#bd3786"
                                                    ],
                                                    [
                                                        0.5555555555555556,
                                                        "#d8576b"
                                                    ],
                                                    [
                                                        0.6666666666666666,
                                                        "#ed7953"
                                                    ],
                                                    [
                                                        0.7777777777777778,
                                                        "#fb9f3a"
                                                    ],
                                                    [
                                                        0.8888888888888888,
                                                        "#fdca26"
                                                    ],
                                                    [
                                                        1,
                                                        "#f0f921"
                                                    ]
                                                ],
                                                "type": "contour"
                                            }
                                        ],
                                        "contourcarpet": [
                                            {
                                                "colorbar": {
                                                    "outlinewidth": 0,
                                                    "ticks": ""
                                                },
                                                "type": "contourcarpet"
                                            }
                                        ],
                                        "heatmap": [
                                            {
                                                "colorbar": {
                                                    "outlinewidth": 0,
                                                    "ticks": ""
                                                },
                                                "colorscale": [
                                                    [
                                                        0,
                                                        "#0d0887"
                                                    ],
                                                    [
                                                        0.1111111111111111,
                                                        "#46039f"
                                                    ],
                                                    [
                                                        0.2222222222222222,
                                                        "#7201a8"
                                                    ],
                                                    [
                                                        0.3333333333333333,
                                                        "#9c179e"
                                                    ],
                                                    [
                                                        0.4444444444444444,
                                                        "#bd3786"
                                                    ],
                                                    [
                                                        0.5555555555555556,
                                                        "#d8576b"
                                                    ],
                                                    [
                                                        0.6666666666666666,
                                                        "#ed7953"
                                                    ],
                                                    [
                                                        0.7777777777777778,
                                                        "#fb9f3a"
                                                    ],
                                                    [
                                                        0.8888888888888888,
                                                        "#fdca26"
                                                    ],
                                                    [
                                                        1,
                                                        "#f0f921"
                                                    ]
                                                ],
                                                "type": "heatmap"
                                            }
                                        ],
                                        "heatmapgl": [
                                            {
                                                "colorbar": {
                                                    "outlinewidth": 0,
                                                    "ticks": ""
                                                },
                                                "colorscale": [
                                                    [
                                                        0,
                                                        "#0d0887"
                                                    ],
                                                    [
                                                        0.1111111111111111,
                                                        "#46039f"
                                                    ],
                                                    [
                                                        0.2222222222222222,
                                                        "#7201a8"
                                                    ],
                                                    [
                                                        0.3333333333333333,
                                                        "#9c179e"
                                                    ],
                                                    [
                                                        0.4444444444444444,
                                                        "#bd3786"
                                                    ],
                                                    [
                                                        0.5555555555555556,
                                                        "#d8576b"
                                                    ],
                                                    [
                                                        0.6666666666666666,
                                                        "#ed7953"
                                                    ],
                                                    [
                                                        0.7777777777777778,
                                                        "#fb9f3a"
                                                    ],
                                                    [
                                                        0.8888888888888888,
                                                        "#fdca26"
                                                    ],
                                                    [
                                                        1,
                                                        "#f0f921"
                                                    ]
                                                ],
                                                "type": "heatmapgl"
                                            }
                                        ],
                                        "histogram": [
                                            {
                                                "marker": {
                                                    "colorbar": {
                                                        "outlinewidth": 0,
                                                        "ticks": ""
                                                    }
                                                },
                                                "type": "histogram"
                                            }
                                        ],
                                        "histogram2d": [
                                            {
                                                "colorbar": {
                                                    "outlinewidth": 0,
                                                    "ticks": ""
                                                },
                                                "colorscale": [
                                                    [
                                                        0,
                                                        "#0d0887"
                                                    ],
                                                    [
                                                        0.1111111111111111,
                                                        "#46039f"
                                                    ],
                                                    [
                                                        0.2222222222222222,
                                                        "#7201a8"
                                                    ],
                                                    [
                                                        0.3333333333333333,
                                                        "#9c179e"
                                                    ],
                                                    [
                                                        0.4444444444444444,
                                                        "#bd3786"
                                                    ],
                                                    [
                                                        0.5555555555555556,
                                                        "#d8576b"
                                                    ],
                                                    [
                                                        0.6666666666666666,
                                                        "#ed7953"
                                                    ],
                                                    [
                                                        0.7777777777777778,
                                                        "#fb9f3a"
                                                    ],
                                                    [
                                                        0.8888888888888888,
                                                        "#fdca26"
                                                    ],
                                                    [
                                                        1,
                                                        "#f0f921"
                                                    ]
                                                ],
                                                "type": "histogram2d"
                                            }
                                        ],
                                        "histogram2dcontour": [
                                            {
                                                "colorbar": {
                                                    "outlinewidth": 0,
                                                    "ticks": ""
                                                },
                                                "colorscale": [
                                                    [
                                                        0,
                                                        "#0d0887"
                                                    ],
                                                    [
                                                        0.1111111111111111,
                                                        "#46039f"
                                                    ],
                                                    [
                                                        0.2222222222222222,
                                                        "#7201a8"
                                                    ],
                                                    [
                                                        0.3333333333333333,
                                                        "#9c179e"
                                                    ],
                                                    [
                                                        0.4444444444444444,
                                                        "#bd3786"
                                                    ],
                                                    [
                                                        0.5555555555555556,
                                                        "#d8576b"
                                                    ],
                                                    [
                                                        0.6666666666666666,
                                                        "#ed7953"
                                                    ],
                                                    [
                                                        0.7777777777777778,
                                                        "#fb9f3a"
                                                    ],
                                                    [
                                                        0.8888888888888888,
                                                        "#fdca26"
                                                    ],
                                                    [
                                                        1,
                                                        "#f0f921"
                                                    ]
                                                ],
                                                "type": "histogram2dcontour"
                                            }
                                        ],
                                        "mesh3d": [
                                            {
                                                "colorbar": {
                                                    "outlinewidth": 0,
                                                    "ticks": ""
                                                },
                                                "type": "mesh3d"
                                            }
                                        ],
                                        "parcoords": [
                                            {
                                                "line": {
                                                    "colorbar": {
                                                        "outlinewidth": 0,
                                                        "ticks": ""
                                                    }
                                                },
                                                "type": "parcoords"
                                            }
                                        ],
                                        "pie": [
                                            {
                                                "automargin": true,
                                                "type": "pie"
                                            }
                                        ],
                                        "scatter": [
                                            {
                                                "marker": {
                                                    "colorbar": {
                                                        "outlinewidth": 0,
                                                        "ticks": ""
                                                    }
                                                },
                                                "type": "scatter"
                                            }
                                        ],
                                        "scatter3d": [
                                            {
                                                "line": {
                                                    "colorbar": {
                                                        "outlinewidth": 0,
                                                        "ticks": ""
                                                    }
                                                },
                                                "marker": {
                                                    "colorbar": {
                                                        "outlinewidth": 0,
                                                        "ticks": ""
                                                    }
                                                },
                                                "type": "scatter3d"
                                            }
                                        ],
                                        "scattercarpet": [
                                            {
                                                "marker": {
                                                    "colorbar": {
                                                        "outlinewidth": 0,
                                                        "ticks": ""
                                                    }
                                                },
                                                "type": "scattercarpet"
                                            }
                                        ],
                                        "scattergeo": [
                                            {
                                                "marker": {
                                                    "colorbar": {
                                                        "outlinewidth": 0,
                                                        "ticks": ""
                                                    }
                                                },
                                                "type": "scattergeo"
                                            }
                                        ],
                                        "scattergl": [
                                            {
                                                "marker": {
                                                    "colorbar": {
                                                        "outlinewidth": 0,
                                                        "ticks": ""
                                                    }
                                                },
                                                "type": "scattergl"
                                            }
                                        ],
                                        "scattermapbox": [
                                            {
                                                "marker": {
                                                    "colorbar": {
                                                        "outlinewidth": 0,
                                                        "ticks": ""
                                                    }
                                                },
                                                "type": "scattermapbox"
                                            }
                                        ],
                                        "scatterpolar": [
                                            {
                                                "marker": {
                                                    "colorbar": {
                                                        "outlinewidth": 0,
                                                        "ticks": ""
                                                    }
                                                },
                                                "type": "scatterpolar"
                                            }
                                        ],
                                        "scatterpolargl": [
                                            {
                                                "marker": {
                                                    "colorbar": {
                                                        "outlinewidth": 0,
                                                        "ticks": ""
                                                    }
                                                },
                                                "type": "scatterpolargl"
                                            }
                                        ],
                                        "scatterternary": [
                                            {
                                                "marker": {
                                                    "colorbar": {
                                                        "outlinewidth": 0,
                                                        "ticks": ""
                                                    }
                                                },
                                                "type": "scatterternary"
                                            }
                                        ],
                                        "surface": [
                                            {
                                                "colorbar": {
                                                    "outlinewidth": 0,
                                                    "ticks": ""
                                                },
                                                "colorscale": [
                                                    [
                                                        0,
                                                        "#0d0887"
                                                    ],
                                                    [
                                                        0.1111111111111111,
                                                        "#46039f"
                                                    ],
                                                    [
                                                        0.2222222222222222,
                                                        "#7201a8"
                                                    ],
                                                    [
                                                        0.3333333333333333,
                                                        "#9c179e"
                                                    ],
                                                    [
                                                        0.4444444444444444,
                                                        "#bd3786"
                                                    ],
                                                    [
                                                        0.5555555555555556,
                                                        "#d8576b"
                                                    ],
                                                    [
                                                        0.6666666666666666,
                                                        "#ed7953"
                                                    ],
                                                    [
                                                        0.7777777777777778,
                                                        "#fb9f3a"
                                                    ],
                                                    [
                                                        0.8888888888888888,
                                                        "#fdca26"
                                                    ],
                                                    [
                                                        1,
                                                        "#f0f921"
                                                    ]
                                                ],
                                                "type": "surface"
                                            }
                                        ],
                                        "table": [
                                            {
                                                "cells": {
                                                    "fill": {
                                                        "color": "#EBF0F8"
                                                    },
                                                    "line": {
                                                        "color": "white"
                                                    }
                                                },
                                                "header": {
                                                    "fill": {
                                                        "color": "#C8D4E3"
                                                    },
                                                    "line": {
                                                        "color": "white"
                                                    }
                                                },
                                                "type": "table"
                                            }
                                        ]
                                    },
                                    "layout": {
                                        "annotationdefaults": {
                                            "arrowcolor": "#2a3f5f",
                                            "arrowhead": 0,
                                            "arrowwidth": 1
                                        },
                                        "autotypenumbers": "strict",
                                        "coloraxis": {
                                            "colorbar": {
                                                "outlinewidth": 0,
                                                "ticks": ""
                                            }
                                        },
                                        "colorscale": {
                                            "diverging": [
                                                [
                                                    0,
                                                    "#8e0152"
                                                ],
                                                [
                                                    0.1,
                                                    "#c51b7d"
                                                ],
                                                [
                                                    0.2,
                                                    "#de77ae"
                                                ],
                                                [
                                                    0.3,
                                                    "#f1b6da"
                                                ],
                                                [
                                                    0.4,
                                                    "#fde0ef"
                                                ],
                                                [
                                                    0.5,
                                                    "#f7f7f7"
                                                ],
                                                [
                                                    0.6,
                                                    "#e6f5d0"
                                                ],
                                                [
                                                    0.7,
                                                    "#b8e186"
                                                ],
                                                [
                                                    0.8,
                                                    "#7fbc41"
                                                ],
                                                [
                                                    0.9,
                                                    "#4d9221"
                                                ],
                                                [
                                                    1,
                                                    "#276419"
                                                ]
                                            ],
                                            "sequential": [
                                                [
                                                    0,
                                                    "#0d0887"
                                                ],
                                                [
                                                    0.1111111111111111,
                                                    "#46039f"
                                                ],
                                                [
                                                    0.2222222222222222,
                                                    "#7201a8"
                                                ],
                                                [
                                                    0.3333333333333333,
                                                    "#9c179e"
                                                ],
                                                [
                                                    0.4444444444444444,
                                                    "#bd3786"
                                                ],
                                                [
                                                    0.5555555555555556,
                                                    "#d8576b"
                                                ],
                                                [
                                                    0.6666666666666666,
                                                    "#ed7953"
                                                ],
                                                [
                                                    0.7777777777777778,
                                                    "#fb9f3a"
                                                ],
                                                [
                                                    0.8888888888888888,
                                                    "#fdca26"
                                                ],
                                                [
                                                    1,
                                                    "#f0f921"
                                                ]
                                            ],
                                            "sequentialminus": [
                                                [
                                                    0,
                                                    "#0d0887"
                                                ],
                                                [
                                                    0.1111111111111111,
                                                    "#46039f"
                                                ],
                                                [
                                                    0.2222222222222222,
                                                    "#7201a8"
                                                ],
                                                [
                                                    0.3333333333333333,
                                                    "#9c179e"
                                                ],
                                                [
                                                    0.4444444444444444,
                                                    "#bd3786"
                                                ],
                                                [
                                                    0.5555555555555556,
                                                    "#d8576b"
                                                ],
                                                [
                                                    0.6666666666666666,
                                                    "#ed7953"
                                                ],
                                                [
                                                    0.7777777777777778,
                                                    "#fb9f3a"
                                                ],
                                                [
                                                    0.8888888888888888,
                                                    "#fdca26"
                                                ],
                                                [
                                                    1,
                                                    "#f0f921"
                                                ]
                                            ]
                                        },
                                        "colorway": [
                                            "#636efa",
                                            "#EF553B",
                                            "#00cc96",
                                            "#ab63fa",
                                            "#FFA15A",
                                            "#19d3f3",
                                            "#FF6692",
                                            "#B6E880",
                                            "#FF97FF",
                                            "#FECB52"
                                        ],
                                        "font": {
                                            "color": "#2a3f5f"
                                        },
                                        "geo": {
                                            "bgcolor": "white",
                                            "lakecolor": "white",
                                            "landcolor": "#E5ECF6",
                                            "showlakes": true,
                                            "showland": true,
                                            "subunitcolor": "white"
                                        },
                                        "hoverlabel": {
                                            "align": "left"
                                        },
                                        "hovermode": "closest",
                                        "mapbox": {
                                            "style": "light"
                                        },
                                        "paper_bgcolor": "white",
                                        "plot_bgcolor": "#E5ECF6",
                                        "polar": {
                                            "angularaxis": {
                                                "gridcolor": "white",
                                                "linecolor": "white",
                                                "ticks": ""
                                            },
                                            "bgcolor": "#E5ECF6",
                                            "radialaxis": {
                                                "gridcolor": "white",
                                                "linecolor": "white",
                                                "ticks": ""
                                            }
                                        },
                                        "scene": {
                                            "xaxis": {
                                                "backgroundcolor": "#E5ECF6",
                                                "gridcolor": "white",
                                                "gridwidth": 2,
                                                "linecolor": "white",
                                                "showbackground": true,
                                                "ticks": "",
                                                "zerolinecolor": "white"
                                            },
                                            "yaxis": {
                                                "backgroundcolor": "#E5ECF6",
                                                "gridcolor": "white",
                                                "gridwidth": 2,
                                                "linecolor": "white",
                                                "showbackground": true,
                                                "ticks": "",
                                                "zerolinecolor": "white"
                                            },
                                            "zaxis": {
                                                "backgroundcolor": "#E5ECF6",
                                                "gridcolor": "white",
                                                "gridwidth": 2,
                                                "linecolor": "white",
                                                "showbackground": true,
                                                "ticks": "",
                                                "zerolinecolor": "white"
                                            }
                                        },
                                        "shapedefaults": {
                                            "line": {
                                                "color": "#2a3f5f"
                                            }
                                        },
                                        "ternary": {
                                            "aaxis": {
                                                "gridcolor": "white",
                                                "linecolor": "white",
                                                "ticks": ""
                                            },
                                            "baxis": {
                                                "gridcolor": "white",
                                                "linecolor": "white",
                                                "ticks": ""
                                            },
                                            "bgcolor": "#E5ECF6",
                                            "caxis": {
                                                "gridcolor": "white",
                                                "linecolor": "white",
                                                "ticks": ""
                                            }
                                        },
                                        "title": {
                                            "x": 0.05
                                        },
                                        "xaxis": {
                                            "automargin": true,
                                            "gridcolor": "white",
                                            "linecolor": "white",
                                            "ticks": "",
                                            "title": {
                                                "standoff": 15
                                            },
                                            "zerolinecolor": "white",
                                            "zerolinewidth": 2
                                        },
                                        "yaxis": {
                                            "automargin": true,
                                            "gridcolor": "white",
                                            "linecolor": "white",
                                            "ticks": "",
                                            "title": {
                                                "standoff": 15
                                            },
                                            "zerolinecolor": "white",
                                            "zerolinewidth": 2
                                        }
                                    }
                                },
                                "xaxis": {
                                    "anchor": "y",
                                    "domain": [
                                        0,
                                        1
                                    ],
                                    "title": {
                                        "text": "date"
                                    }
                                },
                                "yaxis": {
                                    "anchor": "x",
                                    "domain": [
                                        0,
                                        1
                                    ],
                                    "title": {
                                        "text": "Ingresos"
                                    }
                                }
                            }
                        }
                    },
                    "metadata": {}
                }
            ],
            "source": [
                "\n",
                "grafico_horizontal= full_3.groupby(['gender',\"date\"])[\"Ingresos\"].sum().reset_index()\n",
                "grafico_horizontal\n",
                "\n",
                "evolucion_horizontal= px.bar(grafico_horizontal, x=\"date\", y=\"Ingresos\", \\\n",
                "                        color='gender', orientation=\"v\", color_discrete_sequence=px.colors.cyclical.mygbm)\n",
                "\n",
                "evolucion_horizontal.show()"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": 40,
            "metadata": {},
            "outputs": [
                {
                    "output_type": "display_data",
                    "data": {
                        "application/vnd.plotly.v1+json": {
                            "config": {
                                "plotlyServerURL": "https://plot.ly"
                            },
                            "data": [
                                {
                                    "alignmentgroup": "True",
                                    "hovertemplate": "country_id=AR<br>date=%{x}<br>Ingresos=%{y}<extra></extra>",
                                    "legendgroup": "AR",
                                    "marker": {
                                        "color": "#ef55f1"
                                    },
                                    "name": "AR",
                                    "offsetgroup": "AR",
                                    "orientation": "v",
                                    "showlegend": true,
                                    "textposition": "auto",
                                    "type": "bar",
                                    "x": [
                                        "2018-01-28T00:00:00",
                                        "2018-02-28T00:00:00",
                                        "2018-03-28T00:00:00",
                                        "2018-04-28T00:00:00",
                                        "2018-05-28T00:00:00",
                                        "2018-06-28T00:00:00",
                                        "2018-07-28T00:00:00",
                                        "2018-08-28T00:00:00",
                                        "2018-09-28T00:00:00",
                                        "2018-10-28T00:00:00",
                                        "2018-11-28T00:00:00",
                                        "2018-12-28T00:00:00",
                                        "2019-01-28T00:00:00",
                                        "2019-02-28T00:00:00",
                                        "2019-03-28T00:00:00",
                                        "2019-04-28T00:00:00",
                                        "2019-05-28T00:00:00"
                                    ],
                                    "xaxis": "x",
                                    "y": [
                                        20,
                                        20,
                                        20,
                                        20,
                                        20,
                                        20,
                                        20,
                                        20,
                                        20,
                                        20,
                                        20,
                                        30,
                                        30,
                                        30,
                                        40,
                                        40,
                                        40
                                    ],
                                    "yaxis": "y"
                                },
                                {
                                    "alignmentgroup": "True",
                                    "hovertemplate": "country_id=AT<br>date=%{x}<br>Ingresos=%{y}<extra></extra>",
                                    "legendgroup": "AT",
                                    "marker": {
                                        "color": "#fb84ce"
                                    },
                                    "name": "AT",
                                    "offsetgroup": "AT",
                                    "orientation": "v",
                                    "showlegend": true,
                                    "textposition": "auto",
                                    "type": "bar",
                                    "x": [
                                        "2018-01-28T00:00:00",
                                        "2018-02-28T00:00:00",
                                        "2018-03-28T00:00:00",
                                        "2018-04-28T00:00:00",
                                        "2018-05-28T00:00:00",
                                        "2018-06-28T00:00:00",
                                        "2018-07-28T00:00:00",
                                        "2018-08-28T00:00:00",
                                        "2018-09-28T00:00:00",
                                        "2018-10-28T00:00:00",
                                        "2018-11-28T00:00:00",
                                        "2018-12-28T00:00:00",
                                        "2019-01-28T00:00:00",
                                        "2019-02-28T00:00:00",
                                        "2019-03-28T00:00:00",
                                        "2019-04-28T00:00:00",
                                        "2019-05-28T00:00:00"
                                    ],
                                    "xaxis": "x",
                                    "y": [
                                        160,
                                        110,
                                        110,
                                        170,
                                        100,
                                        110,
                                        100,
                                        110,
                                        100,
                                        100,
                                        100,
                                        100,
                                        110,
                                        110,
                                        110,
                                        110,
                                        110
                                    ],
                                    "yaxis": "y"
                                },
                                {
                                    "alignmentgroup": "True",
                                    "hovertemplate": "country_id=BE<br>date=%{x}<br>Ingresos=%{y}<extra></extra>",
                                    "legendgroup": "BE",
                                    "marker": {
                                        "color": "#fbafa1"
                                    },
                                    "name": "BE",
                                    "offsetgroup": "BE",
                                    "orientation": "v",
                                    "showlegend": true,
                                    "textposition": "auto",
                                    "type": "bar",
                                    "x": [
                                        "2018-01-28T00:00:00",
                                        "2018-02-28T00:00:00",
                                        "2018-03-28T00:00:00",
                                        "2018-04-28T00:00:00",
                                        "2018-05-28T00:00:00",
                                        "2018-06-28T00:00:00",
                                        "2018-07-28T00:00:00",
                                        "2018-08-28T00:00:00",
                                        "2018-09-28T00:00:00",
                                        "2018-10-28T00:00:00",
                                        "2018-11-28T00:00:00",
                                        "2018-12-28T00:00:00",
                                        "2019-01-28T00:00:00",
                                        "2019-02-28T00:00:00",
                                        "2019-03-28T00:00:00",
                                        "2019-04-28T00:00:00",
                                        "2019-05-28T00:00:00"
                                    ],
                                    "xaxis": "x",
                                    "y": [
                                        30,
                                        30,
                                        30,
                                        30,
                                        90,
                                        30,
                                        30,
                                        30,
                                        90,
                                        90,
                                        30,
                                        90,
                                        90,
                                        30,
                                        100,
                                        100,
                                        40
                                    ],
                                    "yaxis": "y"
                                },
                                {
                                    "alignmentgroup": "True",
                                    "hovertemplate": "country_id=BR<br>date=%{x}<br>Ingresos=%{y}<extra></extra>",
                                    "legendgroup": "BR",
                                    "marker": {
                                        "color": "#fcd471"
                                    },
                                    "name": "BR",
                                    "offsetgroup": "BR",
                                    "orientation": "v",
                                    "showlegend": true,
                                    "textposition": "auto",
                                    "type": "bar",
                                    "x": [
                                        "2018-01-28T00:00:00",
                                        "2018-02-28T00:00:00",
                                        "2018-03-28T00:00:00",
                                        "2018-04-28T00:00:00",
                                        "2018-05-28T00:00:00",
                                        "2018-06-28T00:00:00",
                                        "2018-07-28T00:00:00",
                                        "2018-08-28T00:00:00",
                                        "2018-09-28T00:00:00",
                                        "2018-10-28T00:00:00",
                                        "2018-11-28T00:00:00",
                                        "2018-12-28T00:00:00",
                                        "2019-01-28T00:00:00",
                                        "2019-02-28T00:00:00",
                                        "2019-03-28T00:00:00",
                                        "2019-04-28T00:00:00",
                                        "2019-05-28T00:00:00"
                                    ],
                                    "xaxis": "x",
                                    "y": [
                                        30,
                                        30,
                                        40,
                                        40,
                                        40,
                                        40,
                                        40,
                                        40,
                                        40,
                                        40,
                                        40,
                                        40,
                                        40,
                                        40,
                                        40,
                                        40,
                                        40
                                    ],
                                    "yaxis": "y"
                                },
                                {
                                    "alignmentgroup": "True",
                                    "hovertemplate": "country_id=CA<br>date=%{x}<br>Ingresos=%{y}<extra></extra>",
                                    "legendgroup": "CA",
                                    "marker": {
                                        "color": "#f0ed35"
                                    },
                                    "name": "CA",
                                    "offsetgroup": "CA",
                                    "orientation": "v",
                                    "showlegend": true,
                                    "textposition": "auto",
                                    "type": "bar",
                                    "x": [
                                        "2018-01-28T00:00:00",
                                        "2018-02-28T00:00:00",
                                        "2018-03-28T00:00:00",
                                        "2018-04-28T00:00:00",
                                        "2018-05-28T00:00:00",
                                        "2018-06-28T00:00:00",
                                        "2018-07-28T00:00:00",
                                        "2018-08-28T00:00:00",
                                        "2018-09-28T00:00:00",
                                        "2018-10-28T00:00:00",
                                        "2018-11-28T00:00:00",
                                        "2018-12-28T00:00:00",
                                        "2019-02-28T00:00:00",
                                        "2019-03-28T00:00:00",
                                        "2019-04-28T00:00:00",
                                        "2019-05-28T00:00:00"
                                    ],
                                    "xaxis": "x",
                                    "y": [
                                        70,
                                        70,
                                        70,
                                        110,
                                        110,
                                        110,
                                        110,
                                        110,
                                        110,
                                        110,
                                        110,
                                        110,
                                        100,
                                        100,
                                        100,
                                        100
                                    ],
                                    "yaxis": "y"
                                },
                                {
                                    "alignmentgroup": "True",
                                    "hovertemplate": "country_id=CH<br>date=%{x}<br>Ingresos=%{y}<extra></extra>",
                                    "legendgroup": "CH",
                                    "marker": {
                                        "color": "#c6e516"
                                    },
                                    "name": "CH",
                                    "offsetgroup": "CH",
                                    "orientation": "v",
                                    "showlegend": true,
                                    "textposition": "auto",
                                    "type": "bar",
                                    "x": [
                                        "2018-01-28T00:00:00",
                                        "2018-02-28T00:00:00",
                                        "2018-03-28T00:00:00",
                                        "2018-04-28T00:00:00",
                                        "2018-05-28T00:00:00",
                                        "2018-06-28T00:00:00",
                                        "2018-07-28T00:00:00",
                                        "2018-08-28T00:00:00",
                                        "2018-09-28T00:00:00",
                                        "2018-10-28T00:00:00",
                                        "2018-11-28T00:00:00",
                                        "2018-12-28T00:00:00",
                                        "2019-01-28T00:00:00",
                                        "2019-02-28T00:00:00",
                                        "2019-03-28T00:00:00",
                                        "2019-04-28T00:00:00",
                                        "2019-05-28T00:00:00"
                                    ],
                                    "xaxis": "x",
                                    "y": [
                                        280,
                                        270,
                                        300,
                                        290,
                                        250,
                                        260,
                                        230,
                                        220,
                                        280,
                                        280,
                                        290,
                                        290,
                                        290,
                                        430,
                                        450,
                                        410,
                                        400
                                    ],
                                    "yaxis": "y"
                                },
                                {
                                    "alignmentgroup": "True",
                                    "hovertemplate": "country_id=CI<br>date=%{x}<br>Ingresos=%{y}<extra></extra>",
                                    "legendgroup": "CI",
                                    "marker": {
                                        "color": "#96d310"
                                    },
                                    "name": "CI",
                                    "offsetgroup": "CI",
                                    "orientation": "v",
                                    "showlegend": true,
                                    "textposition": "auto",
                                    "type": "bar",
                                    "x": [
                                        "2018-01-28T00:00:00",
                                        "2018-02-28T00:00:00",
                                        "2018-03-28T00:00:00",
                                        "2018-04-28T00:00:00",
                                        "2018-05-28T00:00:00",
                                        "2018-06-28T00:00:00",
                                        "2018-07-28T00:00:00",
                                        "2018-08-28T00:00:00",
                                        "2018-09-28T00:00:00",
                                        "2018-10-28T00:00:00",
                                        "2018-11-28T00:00:00",
                                        "2018-12-28T00:00:00",
                                        "2019-01-28T00:00:00",
                                        "2019-02-28T00:00:00",
                                        "2019-03-28T00:00:00",
                                        "2019-04-28T00:00:00",
                                        "2019-05-28T00:00:00"
                                    ],
                                    "xaxis": "x",
                                    "y": [
                                        40,
                                        10,
                                        10,
                                        40,
                                        40,
                                        40,
                                        40,
                                        40,
                                        40,
                                        40,
                                        40,
                                        40,
                                        40,
                                        40,
                                        40,
                                        50,
                                        20
                                    ],
                                    "yaxis": "y"
                                },
                                {
                                    "alignmentgroup": "True",
                                    "hovertemplate": "country_id=CL<br>date=%{x}<br>Ingresos=%{y}<extra></extra>",
                                    "legendgroup": "CL",
                                    "marker": {
                                        "color": "#61c10b"
                                    },
                                    "name": "CL",
                                    "offsetgroup": "CL",
                                    "orientation": "v",
                                    "showlegend": true,
                                    "textposition": "auto",
                                    "type": "bar",
                                    "x": [
                                        "2018-01-28T00:00:00",
                                        "2018-02-28T00:00:00",
                                        "2018-03-28T00:00:00",
                                        "2018-04-28T00:00:00",
                                        "2018-05-28T00:00:00",
                                        "2018-06-28T00:00:00",
                                        "2018-07-28T00:00:00",
                                        "2018-08-28T00:00:00",
                                        "2018-09-28T00:00:00",
                                        "2018-10-28T00:00:00",
                                        "2018-11-28T00:00:00",
                                        "2018-12-28T00:00:00",
                                        "2019-01-28T00:00:00",
                                        "2019-02-28T00:00:00",
                                        "2019-03-28T00:00:00",
                                        "2019-04-28T00:00:00",
                                        "2019-05-28T00:00:00"
                                    ],
                                    "xaxis": "x",
                                    "y": [
                                        60,
                                        70,
                                        80,
                                        80,
                                        70,
                                        70,
                                        60,
                                        60,
                                        60,
                                        60,
                                        70,
                                        70,
                                        20,
                                        20,
                                        20,
                                        80,
                                        100
                                    ],
                                    "yaxis": "y"
                                },
                                {
                                    "alignmentgroup": "True",
                                    "hovertemplate": "country_id=CM<br>date=%{x}<br>Ingresos=%{y}<extra></extra>",
                                    "legendgroup": "CM",
                                    "marker": {
                                        "color": "#31ac28"
                                    },
                                    "name": "CM",
                                    "offsetgroup": "CM",
                                    "orientation": "v",
                                    "showlegend": true,
                                    "textposition": "auto",
                                    "type": "bar",
                                    "x": [
                                        "2018-01-28T00:00:00",
                                        "2018-02-28T00:00:00",
                                        "2018-03-28T00:00:00",
                                        "2018-04-28T00:00:00",
                                        "2018-05-28T00:00:00",
                                        "2018-06-28T00:00:00",
                                        "2018-07-28T00:00:00",
                                        "2018-08-28T00:00:00",
                                        "2018-09-28T00:00:00",
                                        "2018-10-28T00:00:00",
                                        "2018-11-28T00:00:00",
                                        "2018-12-28T00:00:00",
                                        "2019-01-28T00:00:00",
                                        "2019-02-28T00:00:00",
                                        "2019-03-28T00:00:00",
                                        "2019-04-28T00:00:00",
                                        "2019-05-28T00:00:00"
                                    ],
                                    "xaxis": "x",
                                    "y": [
                                        60,
                                        120,
                                        120,
                                        120,
                                        60,
                                        60,
                                        60,
                                        60,
                                        60,
                                        60,
                                        60,
                                        60,
                                        60,
                                        120,
                                        60,
                                        10,
                                        70
                                    ],
                                    "yaxis": "y"
                                },
                                {
                                    "alignmentgroup": "True",
                                    "hovertemplate": "country_id=CN<br>date=%{x}<br>Ingresos=%{y}<extra></extra>",
                                    "legendgroup": "CN",
                                    "marker": {
                                        "color": "#439064"
                                    },
                                    "name": "CN",
                                    "offsetgroup": "CN",
                                    "orientation": "v",
                                    "showlegend": true,
                                    "textposition": "auto",
                                    "type": "bar",
                                    "x": [
                                        "2018-01-28T00:00:00",
                                        "2018-02-28T00:00:00",
                                        "2018-03-28T00:00:00",
                                        "2018-04-28T00:00:00",
                                        "2018-05-28T00:00:00",
                                        "2018-06-28T00:00:00",
                                        "2018-07-28T00:00:00",
                                        "2018-08-28T00:00:00",
                                        "2018-09-28T00:00:00",
                                        "2018-10-28T00:00:00",
                                        "2018-11-28T00:00:00",
                                        "2018-12-28T00:00:00",
                                        "2019-01-28T00:00:00",
                                        "2019-02-28T00:00:00",
                                        "2019-03-28T00:00:00",
                                        "2019-04-28T00:00:00",
                                        "2019-05-28T00:00:00"
                                    ],
                                    "xaxis": "x",
                                    "y": [
                                        60,
                                        60,
                                        60,
                                        60,
                                        60,
                                        60,
                                        70,
                                        70,
                                        70,
                                        70,
                                        20,
                                        20,
                                        20,
                                        20,
                                        20,
                                        20,
                                        20
                                    ],
                                    "yaxis": "y"
                                },
                                {
                                    "alignmentgroup": "True",
                                    "hovertemplate": "country_id=CO<br>date=%{x}<br>Ingresos=%{y}<extra></extra>",
                                    "legendgroup": "CO",
                                    "marker": {
                                        "color": "#3d719a"
                                    },
                                    "name": "CO",
                                    "offsetgroup": "CO",
                                    "orientation": "v",
                                    "showlegend": true,
                                    "textposition": "auto",
                                    "type": "bar",
                                    "x": [
                                        "2018-01-28T00:00:00",
                                        "2018-02-28T00:00:00",
                                        "2018-03-28T00:00:00",
                                        "2018-04-28T00:00:00",
                                        "2018-05-28T00:00:00",
                                        "2018-06-28T00:00:00",
                                        "2018-07-28T00:00:00",
                                        "2018-08-28T00:00:00",
                                        "2018-09-28T00:00:00",
                                        "2018-10-28T00:00:00",
                                        "2018-11-28T00:00:00",
                                        "2018-12-28T00:00:00",
                                        "2019-01-28T00:00:00",
                                        "2019-02-28T00:00:00",
                                        "2019-03-28T00:00:00",
                                        "2019-04-28T00:00:00",
                                        "2019-05-28T00:00:00"
                                    ],
                                    "xaxis": "x",
                                    "y": [
                                        10,
                                        10,
                                        10,
                                        10,
                                        10,
                                        10,
                                        10,
                                        10,
                                        10,
                                        10,
                                        10,
                                        10,
                                        10,
                                        10,
                                        10,
                                        10,
                                        10
                                    ],
                                    "yaxis": "y"
                                },
                                {
                                    "alignmentgroup": "True",
                                    "hovertemplate": "country_id=DE<br>date=%{x}<br>Ingresos=%{y}<extra></extra>",
                                    "legendgroup": "DE",
                                    "marker": {
                                        "color": "#284ec8"
                                    },
                                    "name": "DE",
                                    "offsetgroup": "DE",
                                    "orientation": "v",
                                    "showlegend": true,
                                    "textposition": "auto",
                                    "type": "bar",
                                    "x": [
                                        "2018-01-28T00:00:00",
                                        "2018-02-28T00:00:00",
                                        "2018-03-28T00:00:00",
                                        "2018-04-28T00:00:00",
                                        "2018-05-28T00:00:00",
                                        "2018-06-28T00:00:00",
                                        "2018-07-28T00:00:00",
                                        "2018-08-28T00:00:00",
                                        "2018-09-28T00:00:00",
                                        "2018-10-28T00:00:00",
                                        "2018-11-28T00:00:00",
                                        "2018-12-28T00:00:00",
                                        "2019-01-28T00:00:00",
                                        "2019-02-28T00:00:00",
                                        "2019-03-28T00:00:00",
                                        "2019-04-28T00:00:00",
                                        "2019-05-28T00:00:00"
                                    ],
                                    "xaxis": "x",
                                    "y": [
                                        280,
                                        280,
                                        300,
                                        290,
                                        290,
                                        260,
                                        250,
                                        260,
                                        260,
                                        260,
                                        250,
                                        250,
                                        260,
                                        280,
                                        320,
                                        320,
                                        320
                                    ],
                                    "yaxis": "y"
                                },
                                {
                                    "alignmentgroup": "True",
                                    "hovertemplate": "country_id=DO<br>date=%{x}<br>Ingresos=%{y}<extra></extra>",
                                    "legendgroup": "DO",
                                    "marker": {
                                        "color": "#2e21ea"
                                    },
                                    "name": "DO",
                                    "offsetgroup": "DO",
                                    "orientation": "v",
                                    "showlegend": true,
                                    "textposition": "auto",
                                    "type": "bar",
                                    "x": [
                                        "2018-01-28T00:00:00",
                                        "2018-02-28T00:00:00",
                                        "2018-03-28T00:00:00",
                                        "2018-04-28T00:00:00",
                                        "2018-05-28T00:00:00",
                                        "2018-06-28T00:00:00",
                                        "2018-07-28T00:00:00",
                                        "2018-08-28T00:00:00",
                                        "2018-09-28T00:00:00",
                                        "2018-10-28T00:00:00",
                                        "2018-11-28T00:00:00",
                                        "2018-12-28T00:00:00",
                                        "2019-01-28T00:00:00",
                                        "2019-02-28T00:00:00",
                                        "2019-03-28T00:00:00",
                                        "2019-04-28T00:00:00",
                                        "2019-05-28T00:00:00"
                                    ],
                                    "xaxis": "x",
                                    "y": [
                                        20,
                                        20,
                                        20,
                                        20,
                                        20,
                                        20,
                                        20,
                                        20,
                                        20,
                                        20,
                                        20,
                                        20,
                                        20,
                                        20,
                                        20,
                                        10,
                                        10
                                    ],
                                    "yaxis": "y"
                                },
                                {
                                    "alignmentgroup": "True",
                                    "hovertemplate": "country_id=DZ<br>date=%{x}<br>Ingresos=%{y}<extra></extra>",
                                    "legendgroup": "DZ",
                                    "marker": {
                                        "color": "#6324f5"
                                    },
                                    "name": "DZ",
                                    "offsetgroup": "DZ",
                                    "orientation": "v",
                                    "showlegend": true,
                                    "textposition": "auto",
                                    "type": "bar",
                                    "x": [
                                        "2018-11-28T00:00:00",
                                        "2018-12-28T00:00:00",
                                        "2019-01-28T00:00:00",
                                        "2019-02-28T00:00:00",
                                        "2019-03-28T00:00:00",
                                        "2019-04-28T00:00:00",
                                        "2019-05-28T00:00:00"
                                    ],
                                    "xaxis": "x",
                                    "y": [
                                        10,
                                        10,
                                        10,
                                        10,
                                        10,
                                        10,
                                        10
                                    ],
                                    "yaxis": "y"
                                },
                                {
                                    "alignmentgroup": "True",
                                    "hovertemplate": "country_id=ES<br>date=%{x}<br>Ingresos=%{y}<extra></extra>",
                                    "legendgroup": "ES",
                                    "marker": {
                                        "color": "#9139fa"
                                    },
                                    "name": "ES",
                                    "offsetgroup": "ES",
                                    "orientation": "v",
                                    "showlegend": true,
                                    "textposition": "auto",
                                    "type": "bar",
                                    "x": [
                                        "2018-01-28T00:00:00",
                                        "2018-02-28T00:00:00",
                                        "2018-03-28T00:00:00",
                                        "2018-04-28T00:00:00",
                                        "2018-05-28T00:00:00",
                                        "2018-06-28T00:00:00",
                                        "2018-07-28T00:00:00",
                                        "2018-08-28T00:00:00",
                                        "2018-09-28T00:00:00",
                                        "2018-10-28T00:00:00",
                                        "2018-11-28T00:00:00",
                                        "2018-12-28T00:00:00",
                                        "2019-01-28T00:00:00",
                                        "2019-02-28T00:00:00",
                                        "2019-03-28T00:00:00",
                                        "2019-04-28T00:00:00",
                                        "2019-05-28T00:00:00"
                                    ],
                                    "xaxis": "x",
                                    "y": [
                                        3621250,
                                        3716940,
                                        3817220,
                                        3904820,
                                        3927320,
                                        4042200,
                                        4240680,
                                        4352210,
                                        4584130,
                                        4839800,
                                        4971480,
                                        5108200,
                                        5006450,
                                        5190690,
                                        5282950,
                                        5334450,
                                        5424270
                                    ],
                                    "yaxis": "y"
                                },
                                {
                                    "alignmentgroup": "True",
                                    "hovertemplate": "country_id=ET<br>date=%{x}<br>Ingresos=%{y}<extra></extra>",
                                    "legendgroup": "ET",
                                    "marker": {
                                        "color": "#c543fa"
                                    },
                                    "name": "ET",
                                    "offsetgroup": "ET",
                                    "orientation": "v",
                                    "showlegend": true,
                                    "textposition": "auto",
                                    "type": "bar",
                                    "x": [
                                        "2018-01-28T00:00:00",
                                        "2018-02-28T00:00:00",
                                        "2018-03-28T00:00:00",
                                        "2018-04-28T00:00:00",
                                        "2018-05-28T00:00:00",
                                        "2018-06-28T00:00:00",
                                        "2018-07-28T00:00:00",
                                        "2018-08-28T00:00:00",
                                        "2018-09-28T00:00:00",
                                        "2018-10-28T00:00:00",
                                        "2018-11-28T00:00:00",
                                        "2018-12-28T00:00:00",
                                        "2019-01-28T00:00:00",
                                        "2019-02-28T00:00:00",
                                        "2019-03-28T00:00:00",
                                        "2019-04-28T00:00:00",
                                        "2019-05-28T00:00:00"
                                    ],
                                    "xaxis": "x",
                                    "y": [
                                        70,
                                        10,
                                        10,
                                        70,
                                        10,
                                        10,
                                        70,
                                        70,
                                        70,
                                        10,
                                        10,
                                        10,
                                        20,
                                        10,
                                        10,
                                        10,
                                        10
                                    ],
                                    "yaxis": "y"
                                },
                                {
                                    "alignmentgroup": "True",
                                    "hovertemplate": "country_id=FR<br>date=%{x}<br>Ingresos=%{y}<extra></extra>",
                                    "legendgroup": "FR",
                                    "marker": {
                                        "color": "#ef55f1"
                                    },
                                    "name": "FR",
                                    "offsetgroup": "FR",
                                    "orientation": "v",
                                    "showlegend": true,
                                    "textposition": "auto",
                                    "type": "bar",
                                    "x": [
                                        "2018-01-28T00:00:00",
                                        "2018-02-28T00:00:00",
                                        "2018-03-28T00:00:00",
                                        "2018-04-28T00:00:00",
                                        "2018-05-28T00:00:00",
                                        "2018-06-28T00:00:00",
                                        "2018-07-28T00:00:00",
                                        "2018-08-28T00:00:00",
                                        "2018-09-28T00:00:00",
                                        "2018-10-28T00:00:00",
                                        "2018-11-28T00:00:00",
                                        "2018-12-28T00:00:00",
                                        "2019-01-28T00:00:00",
                                        "2019-02-28T00:00:00",
                                        "2019-03-28T00:00:00",
                                        "2019-04-28T00:00:00",
                                        "2019-05-28T00:00:00"
                                    ],
                                    "xaxis": "x",
                                    "y": [
                                        390,
                                        380,
                                        390,
                                        330,
                                        330,
                                        290,
                                        320,
                                        380,
                                        400,
                                        420,
                                        420,
                                        410,
                                        410,
                                        300,
                                        360,
                                        290,
                                        440
                                    ],
                                    "yaxis": "y"
                                },
                                {
                                    "alignmentgroup": "True",
                                    "hovertemplate": "country_id=GA<br>date=%{x}<br>Ingresos=%{y}<extra></extra>",
                                    "legendgroup": "GA",
                                    "marker": {
                                        "color": "#ef55f1"
                                    },
                                    "name": "GA",
                                    "offsetgroup": "GA",
                                    "orientation": "v",
                                    "showlegend": true,
                                    "textposition": "auto",
                                    "type": "bar",
                                    "x": [
                                        "2018-01-28T00:00:00",
                                        "2018-02-28T00:00:00",
                                        "2018-03-28T00:00:00",
                                        "2018-04-28T00:00:00",
                                        "2018-05-28T00:00:00",
                                        "2018-06-28T00:00:00",
                                        "2018-07-28T00:00:00",
                                        "2018-08-28T00:00:00",
                                        "2018-09-28T00:00:00",
                                        "2018-10-28T00:00:00",
                                        "2018-11-28T00:00:00",
                                        "2018-12-28T00:00:00",
                                        "2019-01-28T00:00:00",
                                        "2019-02-28T00:00:00",
                                        "2019-03-28T00:00:00",
                                        "2019-04-28T00:00:00",
                                        "2019-05-28T00:00:00"
                                    ],
                                    "xaxis": "x",
                                    "y": [
                                        10,
                                        10,
                                        10,
                                        10,
                                        10,
                                        10,
                                        10,
                                        10,
                                        10,
                                        10,
                                        10,
                                        10,
                                        10,
                                        10,
                                        10,
                                        10,
                                        10
                                    ],
                                    "yaxis": "y"
                                },
                                {
                                    "alignmentgroup": "True",
                                    "hovertemplate": "country_id=GB<br>date=%{x}<br>Ingresos=%{y}<extra></extra>",
                                    "legendgroup": "GB",
                                    "marker": {
                                        "color": "#fb84ce"
                                    },
                                    "name": "GB",
                                    "offsetgroup": "GB",
                                    "orientation": "v",
                                    "showlegend": true,
                                    "textposition": "auto",
                                    "type": "bar",
                                    "x": [
                                        "2018-01-28T00:00:00",
                                        "2018-02-28T00:00:00",
                                        "2018-03-28T00:00:00",
                                        "2018-04-28T00:00:00",
                                        "2018-05-28T00:00:00",
                                        "2018-06-28T00:00:00",
                                        "2018-07-28T00:00:00",
                                        "2018-08-28T00:00:00",
                                        "2018-09-28T00:00:00",
                                        "2018-10-28T00:00:00",
                                        "2018-11-28T00:00:00",
                                        "2018-12-28T00:00:00",
                                        "2019-01-28T00:00:00",
                                        "2019-02-28T00:00:00",
                                        "2019-03-28T00:00:00",
                                        "2019-04-28T00:00:00",
                                        "2019-05-28T00:00:00"
                                    ],
                                    "xaxis": "x",
                                    "y": [
                                        380,
                                        370,
                                        380,
                                        430,
                                        450,
                                        560,
                                        660,
                                        610,
                                        540,
                                        470,
                                        470,
                                        550,
                                        560,
                                        570,
                                        520,
                                        570,
                                        490
                                    ],
                                    "yaxis": "y"
                                },
                                {
                                    "alignmentgroup": "True",
                                    "hovertemplate": "country_id=GT<br>date=%{x}<br>Ingresos=%{y}<extra></extra>",
                                    "legendgroup": "GT",
                                    "marker": {
                                        "color": "#fbafa1"
                                    },
                                    "name": "GT",
                                    "offsetgroup": "GT",
                                    "orientation": "v",
                                    "showlegend": true,
                                    "textposition": "auto",
                                    "type": "bar",
                                    "x": [
                                        "2018-01-28T00:00:00",
                                        "2018-02-28T00:00:00",
                                        "2018-03-28T00:00:00",
                                        "2018-04-28T00:00:00",
                                        "2018-05-28T00:00:00",
                                        "2018-06-28T00:00:00",
                                        "2018-07-28T00:00:00",
                                        "2018-08-28T00:00:00",
                                        "2018-09-28T00:00:00",
                                        "2018-10-28T00:00:00",
                                        "2018-11-28T00:00:00",
                                        "2018-12-28T00:00:00",
                                        "2019-01-28T00:00:00",
                                        "2019-02-28T00:00:00",
                                        "2019-03-28T00:00:00",
                                        "2019-04-28T00:00:00",
                                        "2019-05-28T00:00:00"
                                    ],
                                    "xaxis": "x",
                                    "y": [
                                        10,
                                        10,
                                        10,
                                        10,
                                        10,
                                        10,
                                        10,
                                        10,
                                        10,
                                        10,
                                        10,
                                        10,
                                        10,
                                        10,
                                        10,
                                        10,
                                        10
                                    ],
                                    "yaxis": "y"
                                },
                                {
                                    "alignmentgroup": "True",
                                    "hovertemplate": "country_id=IE<br>date=%{x}<br>Ingresos=%{y}<extra></extra>",
                                    "legendgroup": "IE",
                                    "marker": {
                                        "color": "#fcd471"
                                    },
                                    "name": "IE",
                                    "offsetgroup": "IE",
                                    "orientation": "v",
                                    "showlegend": true,
                                    "textposition": "auto",
                                    "type": "bar",
                                    "x": [
                                        "2018-01-28T00:00:00",
                                        "2018-02-28T00:00:00",
                                        "2018-03-28T00:00:00",
                                        "2018-04-28T00:00:00",
                                        "2018-05-28T00:00:00",
                                        "2018-06-28T00:00:00",
                                        "2018-07-28T00:00:00",
                                        "2018-08-28T00:00:00",
                                        "2018-09-28T00:00:00",
                                        "2018-10-28T00:00:00",
                                        "2018-11-28T00:00:00",
                                        "2018-12-28T00:00:00",
                                        "2019-01-28T00:00:00",
                                        "2019-02-28T00:00:00",
                                        "2019-03-28T00:00:00",
                                        "2019-04-28T00:00:00",
                                        "2019-05-28T00:00:00"
                                    ],
                                    "xaxis": "x",
                                    "y": [
                                        50,
                                        50,
                                        50,
                                        50,
                                        60,
                                        50,
                                        60,
                                        50,
                                        60,
                                        50,
                                        60,
                                        50,
                                        60,
                                        50,
                                        60,
                                        50,
                                        60
                                    ],
                                    "yaxis": "y"
                                },
                                {
                                    "alignmentgroup": "True",
                                    "hovertemplate": "country_id=IT<br>date=%{x}<br>Ingresos=%{y}<extra></extra>",
                                    "legendgroup": "IT",
                                    "marker": {
                                        "color": "#f0ed35"
                                    },
                                    "name": "IT",
                                    "offsetgroup": "IT",
                                    "orientation": "v",
                                    "showlegend": true,
                                    "textposition": "auto",
                                    "type": "bar",
                                    "x": [
                                        "2018-01-28T00:00:00",
                                        "2018-02-28T00:00:00",
                                        "2018-03-28T00:00:00",
                                        "2018-04-28T00:00:00",
                                        "2018-05-28T00:00:00",
                                        "2018-06-28T00:00:00",
                                        "2018-07-28T00:00:00",
                                        "2018-08-28T00:00:00",
                                        "2018-09-28T00:00:00",
                                        "2018-10-28T00:00:00",
                                        "2018-11-28T00:00:00",
                                        "2018-12-28T00:00:00",
                                        "2019-01-28T00:00:00",
                                        "2019-02-28T00:00:00",
                                        "2019-03-28T00:00:00",
                                        "2019-04-28T00:00:00",
                                        "2019-05-28T00:00:00"
                                    ],
                                    "xaxis": "x",
                                    "y": [
                                        20,
                                        20,
                                        20,
                                        20,
                                        20,
                                        20,
                                        20,
                                        20,
                                        20,
                                        20,
                                        20,
                                        20,
                                        20,
                                        20,
                                        20,
                                        20,
                                        20
                                    ],
                                    "yaxis": "y"
                                },
                                {
                                    "alignmentgroup": "True",
                                    "hovertemplate": "country_id=LU<br>date=%{x}<br>Ingresos=%{y}<extra></extra>",
                                    "legendgroup": "LU",
                                    "marker": {
                                        "color": "#c6e516"
                                    },
                                    "name": "LU",
                                    "offsetgroup": "LU",
                                    "orientation": "v",
                                    "showlegend": true,
                                    "textposition": "auto",
                                    "type": "bar",
                                    "x": [
                                        "2018-01-28T00:00:00",
                                        "2018-02-28T00:00:00",
                                        "2018-03-28T00:00:00",
                                        "2018-04-28T00:00:00",
                                        "2018-05-28T00:00:00",
                                        "2018-06-28T00:00:00",
                                        "2018-07-28T00:00:00",
                                        "2018-08-28T00:00:00",
                                        "2018-09-28T00:00:00",
                                        "2018-10-28T00:00:00",
                                        "2018-11-28T00:00:00",
                                        "2018-12-28T00:00:00",
                                        "2019-01-28T00:00:00",
                                        "2019-02-28T00:00:00",
                                        "2019-03-28T00:00:00",
                                        "2019-04-28T00:00:00",
                                        "2019-05-28T00:00:00"
                                    ],
                                    "xaxis": "x",
                                    "y": [
                                        20,
                                        20,
                                        20,
                                        20,
                                        20,
                                        10,
                                        10,
                                        10,
                                        20,
                                        10,
                                        10,
                                        20,
                                        10,
                                        20,
                                        10,
                                        20,
                                        20
                                    ],
                                    "yaxis": "y"
                                },
                                {
                                    "alignmentgroup": "True",
                                    "hovertemplate": "country_id=MA<br>date=%{x}<br>Ingresos=%{y}<extra></extra>",
                                    "legendgroup": "MA",
                                    "marker": {
                                        "color": "#96d310"
                                    },
                                    "name": "MA",
                                    "offsetgroup": "MA",
                                    "orientation": "v",
                                    "showlegend": true,
                                    "textposition": "auto",
                                    "type": "bar",
                                    "x": [
                                        "2018-01-28T00:00:00",
                                        "2018-02-28T00:00:00",
                                        "2018-03-28T00:00:00",
                                        "2018-04-28T00:00:00",
                                        "2018-05-28T00:00:00",
                                        "2018-06-28T00:00:00",
                                        "2018-07-28T00:00:00",
                                        "2018-08-28T00:00:00",
                                        "2018-09-28T00:00:00",
                                        "2018-10-28T00:00:00",
                                        "2018-11-28T00:00:00",
                                        "2018-12-28T00:00:00",
                                        "2019-01-28T00:00:00",
                                        "2019-02-28T00:00:00",
                                        "2019-03-28T00:00:00",
                                        "2019-04-28T00:00:00",
                                        "2019-05-28T00:00:00"
                                    ],
                                    "xaxis": "x",
                                    "y": [
                                        20,
                                        20,
                                        20,
                                        20,
                                        20,
                                        20,
                                        20,
                                        20,
                                        20,
                                        20,
                                        20,
                                        20,
                                        20,
                                        20,
                                        20,
                                        20,
                                        20
                                    ],
                                    "yaxis": "y"
                                },
                                {
                                    "alignmentgroup": "True",
                                    "hovertemplate": "country_id=MR<br>date=%{x}<br>Ingresos=%{y}<extra></extra>",
                                    "legendgroup": "MR",
                                    "marker": {
                                        "color": "#61c10b"
                                    },
                                    "name": "MR",
                                    "offsetgroup": "MR",
                                    "orientation": "v",
                                    "showlegend": true,
                                    "textposition": "auto",
                                    "type": "bar",
                                    "x": [
                                        "2018-01-28T00:00:00",
                                        "2018-02-28T00:00:00",
                                        "2018-03-28T00:00:00",
                                        "2018-04-28T00:00:00",
                                        "2018-05-28T00:00:00",
                                        "2018-06-28T00:00:00",
                                        "2018-07-28T00:00:00",
                                        "2018-08-28T00:00:00",
                                        "2018-09-28T00:00:00",
                                        "2018-10-28T00:00:00",
                                        "2018-11-28T00:00:00",
                                        "2018-12-28T00:00:00",
                                        "2019-01-28T00:00:00",
                                        "2019-02-28T00:00:00",
                                        "2019-03-28T00:00:00",
                                        "2019-04-28T00:00:00",
                                        "2019-05-28T00:00:00"
                                    ],
                                    "xaxis": "x",
                                    "y": [
                                        10,
                                        10,
                                        10,
                                        10,
                                        10,
                                        10,
                                        10,
                                        10,
                                        10,
                                        10,
                                        10,
                                        10,
                                        10,
                                        10,
                                        10,
                                        10,
                                        10
                                    ],
                                    "yaxis": "y"
                                },
                                {
                                    "alignmentgroup": "True",
                                    "hovertemplate": "country_id=MX<br>date=%{x}<br>Ingresos=%{y}<extra></extra>",
                                    "legendgroup": "MX",
                                    "marker": {
                                        "color": "#31ac28"
                                    },
                                    "name": "MX",
                                    "offsetgroup": "MX",
                                    "orientation": "v",
                                    "showlegend": true,
                                    "textposition": "auto",
                                    "type": "bar",
                                    "x": [
                                        "2018-01-28T00:00:00",
                                        "2018-02-28T00:00:00",
                                        "2018-03-28T00:00:00",
                                        "2018-04-28T00:00:00",
                                        "2018-05-28T00:00:00",
                                        "2018-06-28T00:00:00",
                                        "2018-07-28T00:00:00",
                                        "2018-08-28T00:00:00",
                                        "2018-09-28T00:00:00",
                                        "2018-10-28T00:00:00",
                                        "2018-11-28T00:00:00",
                                        "2018-12-28T00:00:00",
                                        "2019-01-28T00:00:00",
                                        "2019-02-28T00:00:00",
                                        "2019-03-28T00:00:00",
                                        "2019-04-28T00:00:00",
                                        "2019-05-28T00:00:00"
                                    ],
                                    "xaxis": "x",
                                    "y": [
                                        30,
                                        30,
                                        30,
                                        30,
                                        30,
                                        30,
                                        30,
                                        70,
                                        70,
                                        70,
                                        70,
                                        70,
                                        70,
                                        70,
                                        70,
                                        80,
                                        80
                                    ],
                                    "yaxis": "y"
                                },
                                {
                                    "alignmentgroup": "True",
                                    "hovertemplate": "country_id=NO<br>date=%{x}<br>Ingresos=%{y}<extra></extra>",
                                    "legendgroup": "NO",
                                    "marker": {
                                        "color": "#439064"
                                    },
                                    "name": "NO",
                                    "offsetgroup": "NO",
                                    "orientation": "v",
                                    "showlegend": true,
                                    "textposition": "auto",
                                    "type": "bar",
                                    "x": [
                                        "2018-01-28T00:00:00",
                                        "2018-02-28T00:00:00",
                                        "2018-03-28T00:00:00",
                                        "2018-04-28T00:00:00",
                                        "2018-05-28T00:00:00",
                                        "2018-06-28T00:00:00",
                                        "2018-07-28T00:00:00",
                                        "2018-08-28T00:00:00",
                                        "2018-09-28T00:00:00",
                                        "2018-10-28T00:00:00",
                                        "2018-11-28T00:00:00",
                                        "2018-12-28T00:00:00",
                                        "2019-01-28T00:00:00",
                                        "2019-02-28T00:00:00",
                                        "2019-03-28T00:00:00",
                                        "2019-04-28T00:00:00",
                                        "2019-05-28T00:00:00"
                                    ],
                                    "xaxis": "x",
                                    "y": [
                                        20,
                                        20,
                                        20,
                                        20,
                                        20,
                                        20,
                                        20,
                                        20,
                                        10,
                                        10,
                                        20,
                                        20,
                                        20,
                                        20,
                                        20,
                                        20,
                                        20
                                    ],
                                    "yaxis": "y"
                                },
                                {
                                    "alignmentgroup": "True",
                                    "hovertemplate": "country_id=PE<br>date=%{x}<br>Ingresos=%{y}<extra></extra>",
                                    "legendgroup": "PE",
                                    "marker": {
                                        "color": "#3d719a"
                                    },
                                    "name": "PE",
                                    "offsetgroup": "PE",
                                    "orientation": "v",
                                    "showlegend": true,
                                    "textposition": "auto",
                                    "type": "bar",
                                    "x": [
                                        "2019-02-28T00:00:00",
                                        "2019-03-28T00:00:00",
                                        "2019-04-28T00:00:00",
                                        "2019-05-28T00:00:00"
                                    ],
                                    "xaxis": "x",
                                    "y": [
                                        40,
                                        40,
                                        40,
                                        40
                                    ],
                                    "yaxis": "y"
                                },
                                {
                                    "alignmentgroup": "True",
                                    "hovertemplate": "country_id=PL<br>date=%{x}<br>Ingresos=%{y}<extra></extra>",
                                    "legendgroup": "PL",
                                    "marker": {
                                        "color": "#284ec8"
                                    },
                                    "name": "PL",
                                    "offsetgroup": "PL",
                                    "orientation": "v",
                                    "showlegend": true,
                                    "textposition": "auto",
                                    "type": "bar",
                                    "x": [
                                        "2018-01-28T00:00:00",
                                        "2018-02-28T00:00:00",
                                        "2018-03-28T00:00:00",
                                        "2018-04-28T00:00:00",
                                        "2018-05-28T00:00:00",
                                        "2018-06-28T00:00:00",
                                        "2018-07-28T00:00:00",
                                        "2018-08-28T00:00:00",
                                        "2018-09-28T00:00:00",
                                        "2018-10-28T00:00:00",
                                        "2018-11-28T00:00:00",
                                        "2018-12-28T00:00:00",
                                        "2019-01-28T00:00:00",
                                        "2019-02-28T00:00:00",
                                        "2019-03-28T00:00:00",
                                        "2019-04-28T00:00:00",
                                        "2019-05-28T00:00:00"
                                    ],
                                    "xaxis": "x",
                                    "y": [
                                        20,
                                        20,
                                        60,
                                        60,
                                        60,
                                        60,
                                        60,
                                        60,
                                        60,
                                        60,
                                        60,
                                        70,
                                        70,
                                        70,
                                        70,
                                        20,
                                        20
                                    ],
                                    "yaxis": "y"
                                },
                                {
                                    "alignmentgroup": "True",
                                    "hovertemplate": "country_id=QA<br>date=%{x}<br>Ingresos=%{y}<extra></extra>",
                                    "legendgroup": "QA",
                                    "marker": {
                                        "color": "#2e21ea"
                                    },
                                    "name": "QA",
                                    "offsetgroup": "QA",
                                    "orientation": "v",
                                    "showlegend": true,
                                    "textposition": "auto",
                                    "type": "bar",
                                    "x": [
                                        "2018-01-28T00:00:00",
                                        "2018-02-28T00:00:00",
                                        "2018-03-28T00:00:00",
                                        "2018-04-28T00:00:00",
                                        "2018-05-28T00:00:00",
                                        "2018-06-28T00:00:00",
                                        "2018-07-28T00:00:00",
                                        "2018-08-28T00:00:00",
                                        "2018-09-28T00:00:00",
                                        "2018-10-28T00:00:00",
                                        "2018-11-28T00:00:00",
                                        "2018-12-28T00:00:00",
                                        "2019-01-28T00:00:00",
                                        "2019-02-28T00:00:00",
                                        "2019-03-28T00:00:00",
                                        "2019-04-28T00:00:00",
                                        "2019-05-28T00:00:00"
                                    ],
                                    "xaxis": "x",
                                    "y": [
                                        20,
                                        20,
                                        20,
                                        20,
                                        20,
                                        20,
                                        30,
                                        30,
                                        30,
                                        20,
                                        20,
                                        20,
                                        30,
                                        30,
                                        20,
                                        20,
                                        20
                                    ],
                                    "yaxis": "y"
                                },
                                {
                                    "alignmentgroup": "True",
                                    "hovertemplate": "country_id=RO<br>date=%{x}<br>Ingresos=%{y}<extra></extra>",
                                    "legendgroup": "RO",
                                    "marker": {
                                        "color": "#6324f5"
                                    },
                                    "name": "RO",
                                    "offsetgroup": "RO",
                                    "orientation": "v",
                                    "showlegend": true,
                                    "textposition": "auto",
                                    "type": "bar",
                                    "x": [
                                        "2018-09-28T00:00:00",
                                        "2018-10-28T00:00:00",
                                        "2018-11-28T00:00:00",
                                        "2018-12-28T00:00:00",
                                        "2019-01-28T00:00:00",
                                        "2019-02-28T00:00:00",
                                        "2019-03-28T00:00:00",
                                        "2019-04-28T00:00:00",
                                        "2019-05-28T00:00:00"
                                    ],
                                    "xaxis": "x",
                                    "y": [
                                        60,
                                        60,
                                        60,
                                        60,
                                        60,
                                        60,
                                        60,
                                        60,
                                        120
                                    ],
                                    "yaxis": "y"
                                },
                                {
                                    "alignmentgroup": "True",
                                    "hovertemplate": "country_id=RU<br>date=%{x}<br>Ingresos=%{y}<extra></extra>",
                                    "legendgroup": "RU",
                                    "marker": {
                                        "color": "#9139fa"
                                    },
                                    "name": "RU",
                                    "offsetgroup": "RU",
                                    "orientation": "v",
                                    "showlegend": true,
                                    "textposition": "auto",
                                    "type": "bar",
                                    "x": [
                                        "2018-01-28T00:00:00",
                                        "2018-02-28T00:00:00",
                                        "2018-03-28T00:00:00",
                                        "2018-04-28T00:00:00",
                                        "2018-05-28T00:00:00",
                                        "2018-06-28T00:00:00",
                                        "2018-07-28T00:00:00",
                                        "2018-08-28T00:00:00",
                                        "2018-09-28T00:00:00",
                                        "2018-10-28T00:00:00",
                                        "2018-11-28T00:00:00",
                                        "2018-12-28T00:00:00",
                                        "2019-01-28T00:00:00",
                                        "2019-02-28T00:00:00",
                                        "2019-03-28T00:00:00",
                                        "2019-04-28T00:00:00",
                                        "2019-05-28T00:00:00"
                                    ],
                                    "xaxis": "x",
                                    "y": [
                                        10,
                                        10,
                                        10,
                                        10,
                                        10,
                                        10,
                                        10,
                                        10,
                                        10,
                                        10,
                                        10,
                                        10,
                                        10,
                                        10,
                                        10,
                                        10,
                                        10
                                    ],
                                    "yaxis": "y"
                                },
                                {
                                    "alignmentgroup": "True",
                                    "hovertemplate": "country_id=SA<br>date=%{x}<br>Ingresos=%{y}<extra></extra>",
                                    "legendgroup": "SA",
                                    "marker": {
                                        "color": "#c543fa"
                                    },
                                    "name": "SA",
                                    "offsetgroup": "SA",
                                    "orientation": "v",
                                    "showlegend": true,
                                    "textposition": "auto",
                                    "type": "bar",
                                    "x": [
                                        "2018-01-28T00:00:00",
                                        "2018-02-28T00:00:00",
                                        "2018-03-28T00:00:00",
                                        "2018-04-28T00:00:00",
                                        "2018-05-28T00:00:00",
                                        "2018-06-28T00:00:00",
                                        "2018-07-28T00:00:00",
                                        "2018-08-28T00:00:00",
                                        "2018-09-28T00:00:00",
                                        "2018-10-28T00:00:00",
                                        "2018-11-28T00:00:00",
                                        "2018-12-28T00:00:00",
                                        "2019-01-28T00:00:00",
                                        "2019-02-28T00:00:00",
                                        "2019-03-28T00:00:00",
                                        "2019-04-28T00:00:00",
                                        "2019-05-28T00:00:00"
                                    ],
                                    "xaxis": "x",
                                    "y": [
                                        50,
                                        60,
                                        60,
                                        60,
                                        60,
                                        20,
                                        20,
                                        20,
                                        20,
                                        10,
                                        10,
                                        10,
                                        10,
                                        10,
                                        10,
                                        50,
                                        10
                                    ],
                                    "yaxis": "y"
                                },
                                {
                                    "alignmentgroup": "True",
                                    "hovertemplate": "country_id=SE<br>date=%{x}<br>Ingresos=%{y}<extra></extra>",
                                    "legendgroup": "SE",
                                    "marker": {
                                        "color": "#ef55f1"
                                    },
                                    "name": "SE",
                                    "offsetgroup": "SE",
                                    "orientation": "v",
                                    "showlegend": true,
                                    "textposition": "auto",
                                    "type": "bar",
                                    "x": [
                                        "2019-02-28T00:00:00",
                                        "2019-03-28T00:00:00",
                                        "2019-04-28T00:00:00",
                                        "2019-05-28T00:00:00"
                                    ],
                                    "xaxis": "x",
                                    "y": [
                                        10,
                                        10,
                                        10,
                                        10
                                    ],
                                    "yaxis": "y"
                                },
                                {
                                    "alignmentgroup": "True",
                                    "hovertemplate": "country_id=SN<br>date=%{x}<br>Ingresos=%{y}<extra></extra>",
                                    "legendgroup": "SN",
                                    "marker": {
                                        "color": "#ef55f1"
                                    },
                                    "name": "SN",
                                    "offsetgroup": "SN",
                                    "orientation": "v",
                                    "showlegend": true,
                                    "textposition": "auto",
                                    "type": "bar",
                                    "x": [
                                        "2018-01-28T00:00:00",
                                        "2018-02-28T00:00:00",
                                        "2018-03-28T00:00:00",
                                        "2018-04-28T00:00:00",
                                        "2018-05-28T00:00:00",
                                        "2018-06-28T00:00:00",
                                        "2018-07-28T00:00:00",
                                        "2018-08-28T00:00:00",
                                        "2018-09-28T00:00:00",
                                        "2018-10-28T00:00:00",
                                        "2018-11-28T00:00:00",
                                        "2018-12-28T00:00:00",
                                        "2019-01-28T00:00:00",
                                        "2019-02-28T00:00:00",
                                        "2019-03-28T00:00:00",
                                        "2019-04-28T00:00:00",
                                        "2019-05-28T00:00:00"
                                    ],
                                    "xaxis": "x",
                                    "y": [
                                        10,
                                        10,
                                        10,
                                        10,
                                        10,
                                        10,
                                        10,
                                        10,
                                        10,
                                        10,
                                        10,
                                        10,
                                        10,
                                        10,
                                        10,
                                        10,
                                        10
                                    ],
                                    "yaxis": "y"
                                },
                                {
                                    "alignmentgroup": "True",
                                    "hovertemplate": "country_id=US<br>date=%{x}<br>Ingresos=%{y}<extra></extra>",
                                    "legendgroup": "US",
                                    "marker": {
                                        "color": "#fb84ce"
                                    },
                                    "name": "US",
                                    "offsetgroup": "US",
                                    "orientation": "v",
                                    "showlegend": true,
                                    "textposition": "auto",
                                    "type": "bar",
                                    "x": [
                                        "2018-01-28T00:00:00",
                                        "2018-02-28T00:00:00",
                                        "2018-03-28T00:00:00",
                                        "2018-04-28T00:00:00",
                                        "2018-05-28T00:00:00",
                                        "2018-06-28T00:00:00",
                                        "2018-07-28T00:00:00",
                                        "2018-08-28T00:00:00",
                                        "2018-09-28T00:00:00",
                                        "2018-10-28T00:00:00",
                                        "2018-11-28T00:00:00",
                                        "2018-12-28T00:00:00",
                                        "2019-01-28T00:00:00",
                                        "2019-02-28T00:00:00",
                                        "2019-03-28T00:00:00",
                                        "2019-04-28T00:00:00",
                                        "2019-05-28T00:00:00"
                                    ],
                                    "xaxis": "x",
                                    "y": [
                                        130,
                                        190,
                                        130,
                                        190,
                                        130,
                                        130,
                                        180,
                                        240,
                                        140,
                                        140,
                                        200,
                                        200,
                                        250,
                                        270,
                                        280,
                                        280,
                                        290
                                    ],
                                    "yaxis": "y"
                                },
                                {
                                    "alignmentgroup": "True",
                                    "hovertemplate": "country_id=VE<br>date=%{x}<br>Ingresos=%{y}<extra></extra>",
                                    "legendgroup": "VE",
                                    "marker": {
                                        "color": "#fbafa1"
                                    },
                                    "name": "VE",
                                    "offsetgroup": "VE",
                                    "orientation": "v",
                                    "showlegend": true,
                                    "textposition": "auto",
                                    "type": "bar",
                                    "x": [
                                        "2018-01-28T00:00:00",
                                        "2018-02-28T00:00:00",
                                        "2018-03-28T00:00:00",
                                        "2018-04-28T00:00:00",
                                        "2018-05-28T00:00:00",
                                        "2018-06-28T00:00:00",
                                        "2018-07-28T00:00:00",
                                        "2018-08-28T00:00:00",
                                        "2018-09-28T00:00:00",
                                        "2018-10-28T00:00:00",
                                        "2018-11-28T00:00:00",
                                        "2018-12-28T00:00:00",
                                        "2019-01-28T00:00:00",
                                        "2019-02-28T00:00:00",
                                        "2019-03-28T00:00:00",
                                        "2019-04-28T00:00:00",
                                        "2019-05-28T00:00:00"
                                    ],
                                    "xaxis": "x",
                                    "y": [
                                        40,
                                        40,
                                        40,
                                        40,
                                        40,
                                        40,
                                        40,
                                        40,
                                        40,
                                        40,
                                        40,
                                        40,
                                        40,
                                        40,
                                        40,
                                        40,
                                        40
                                    ],
                                    "yaxis": "y"
                                }
                            ],
                            "layout": {
                                "barmode": "relative",
                                "legend": {
                                    "title": {
                                        "text": "country_id"
                                    },
                                    "tracegroupgap": 0
                                },
                                "margin": {
                                    "t": 60
                                },
                                "template": {
                                    "data": {
                                        "bar": [
                                            {
                                                "error_x": {
                                                    "color": "#2a3f5f"
                                                },
                                                "error_y": {
                                                    "color": "#2a3f5f"
                                                },
                                                "marker": {
                                                    "line": {
                                                        "color": "#E5ECF6",
                                                        "width": 0.5
                                                    }
                                                },
                                                "type": "bar"
                                            }
                                        ],
                                        "barpolar": [
                                            {
                                                "marker": {
                                                    "line": {
                                                        "color": "#E5ECF6",
                                                        "width": 0.5
                                                    }
                                                },
                                                "type": "barpolar"
                                            }
                                        ],
                                        "carpet": [
                                            {
                                                "aaxis": {
                                                    "endlinecolor": "#2a3f5f",
                                                    "gridcolor": "white",
                                                    "linecolor": "white",
                                                    "minorgridcolor": "white",
                                                    "startlinecolor": "#2a3f5f"
                                                },
                                                "baxis": {
                                                    "endlinecolor": "#2a3f5f",
                                                    "gridcolor": "white",
                                                    "linecolor": "white",
                                                    "minorgridcolor": "white",
                                                    "startlinecolor": "#2a3f5f"
                                                },
                                                "type": "carpet"
                                            }
                                        ],
                                        "choropleth": [
                                            {
                                                "colorbar": {
                                                    "outlinewidth": 0,
                                                    "ticks": ""
                                                },
                                                "type": "choropleth"
                                            }
                                        ],
                                        "contour": [
                                            {
                                                "colorbar": {
                                                    "outlinewidth": 0,
                                                    "ticks": ""
                                                },
                                                "colorscale": [
                                                    [
                                                        0,
                                                        "#0d0887"
                                                    ],
                                                    [
                                                        0.1111111111111111,
                                                        "#46039f"
                                                    ],
                                                    [
                                                        0.2222222222222222,
                                                        "#7201a8"
                                                    ],
                                                    [
                                                        0.3333333333333333,
                                                        "#9c179e"
                                                    ],
                                                    [
                                                        0.4444444444444444,
                                                        "#bd3786"
                                                    ],
                                                    [
                                                        0.5555555555555556,
                                                        "#d8576b"
                                                    ],
                                                    [
                                                        0.6666666666666666,
                                                        "#ed7953"
                                                    ],
                                                    [
                                                        0.7777777777777778,
                                                        "#fb9f3a"
                                                    ],
                                                    [
                                                        0.8888888888888888,
                                                        "#fdca26"
                                                    ],
                                                    [
                                                        1,
                                                        "#f0f921"
                                                    ]
                                                ],
                                                "type": "contour"
                                            }
                                        ],
                                        "contourcarpet": [
                                            {
                                                "colorbar": {
                                                    "outlinewidth": 0,
                                                    "ticks": ""
                                                },
                                                "type": "contourcarpet"
                                            }
                                        ],
                                        "heatmap": [
                                            {
                                                "colorbar": {
                                                    "outlinewidth": 0,
                                                    "ticks": ""
                                                },
                                                "colorscale": [
                                                    [
                                                        0,
                                                        "#0d0887"
                                                    ],
                                                    [
                                                        0.1111111111111111,
                                                        "#46039f"
                                                    ],
                                                    [
                                                        0.2222222222222222,
                                                        "#7201a8"
                                                    ],
                                                    [
                                                        0.3333333333333333,
                                                        "#9c179e"
                                                    ],
                                                    [
                                                        0.4444444444444444,
                                                        "#bd3786"
                                                    ],
                                                    [
                                                        0.5555555555555556,
                                                        "#d8576b"
                                                    ],
                                                    [
                                                        0.6666666666666666,
                                                        "#ed7953"
                                                    ],
                                                    [
                                                        0.7777777777777778,
                                                        "#fb9f3a"
                                                    ],
                                                    [
                                                        0.8888888888888888,
                                                        "#fdca26"
                                                    ],
                                                    [
                                                        1,
                                                        "#f0f921"
                                                    ]
                                                ],
                                                "type": "heatmap"
                                            }
                                        ],
                                        "heatmapgl": [
                                            {
                                                "colorbar": {
                                                    "outlinewidth": 0,
                                                    "ticks": ""
                                                },
                                                "colorscale": [
                                                    [
                                                        0,
                                                        "#0d0887"
                                                    ],
                                                    [
                                                        0.1111111111111111,
                                                        "#46039f"
                                                    ],
                                                    [
                                                        0.2222222222222222,
                                                        "#7201a8"
                                                    ],
                                                    [
                                                        0.3333333333333333,
                                                        "#9c179e"
                                                    ],
                                                    [
                                                        0.4444444444444444,
                                                        "#bd3786"
                                                    ],
                                                    [
                                                        0.5555555555555556,
                                                        "#d8576b"
                                                    ],
                                                    [
                                                        0.6666666666666666,
                                                        "#ed7953"
                                                    ],
                                                    [
                                                        0.7777777777777778,
                                                        "#fb9f3a"
                                                    ],
                                                    [
                                                        0.8888888888888888,
                                                        "#fdca26"
                                                    ],
                                                    [
                                                        1,
                                                        "#f0f921"
                                                    ]
                                                ],
                                                "type": "heatmapgl"
                                            }
                                        ],
                                        "histogram": [
                                            {
                                                "marker": {
                                                    "colorbar": {
                                                        "outlinewidth": 0,
                                                        "ticks": ""
                                                    }
                                                },
                                                "type": "histogram"
                                            }
                                        ],
                                        "histogram2d": [
                                            {
                                                "colorbar": {
                                                    "outlinewidth": 0,
                                                    "ticks": ""
                                                },
                                                "colorscale": [
                                                    [
                                                        0,
                                                        "#0d0887"
                                                    ],
                                                    [
                                                        0.1111111111111111,
                                                        "#46039f"
                                                    ],
                                                    [
                                                        0.2222222222222222,
                                                        "#7201a8"
                                                    ],
                                                    [
                                                        0.3333333333333333,
                                                        "#9c179e"
                                                    ],
                                                    [
                                                        0.4444444444444444,
                                                        "#bd3786"
                                                    ],
                                                    [
                                                        0.5555555555555556,
                                                        "#d8576b"
                                                    ],
                                                    [
                                                        0.6666666666666666,
                                                        "#ed7953"
                                                    ],
                                                    [
                                                        0.7777777777777778,
                                                        "#fb9f3a"
                                                    ],
                                                    [
                                                        0.8888888888888888,
                                                        "#fdca26"
                                                    ],
                                                    [
                                                        1,
                                                        "#f0f921"
                                                    ]
                                                ],
                                                "type": "histogram2d"
                                            }
                                        ],
                                        "histogram2dcontour": [
                                            {
                                                "colorbar": {
                                                    "outlinewidth": 0,
                                                    "ticks": ""
                                                },
                                                "colorscale": [
                                                    [
                                                        0,
                                                        "#0d0887"
                                                    ],
                                                    [
                                                        0.1111111111111111,
                                                        "#46039f"
                                                    ],
                                                    [
                                                        0.2222222222222222,
                                                        "#7201a8"
                                                    ],
                                                    [
                                                        0.3333333333333333,
                                                        "#9c179e"
                                                    ],
                                                    [
                                                        0.4444444444444444,
                                                        "#bd3786"
                                                    ],
                                                    [
                                                        0.5555555555555556,
                                                        "#d8576b"
                                                    ],
                                                    [
                                                        0.6666666666666666,
                                                        "#ed7953"
                                                    ],
                                                    [
                                                        0.7777777777777778,
                                                        "#fb9f3a"
                                                    ],
                                                    [
                                                        0.8888888888888888,
                                                        "#fdca26"
                                                    ],
                                                    [
                                                        1,
                                                        "#f0f921"
                                                    ]
                                                ],
                                                "type": "histogram2dcontour"
                                            }
                                        ],
                                        "mesh3d": [
                                            {
                                                "colorbar": {
                                                    "outlinewidth": 0,
                                                    "ticks": ""
                                                },
                                                "type": "mesh3d"
                                            }
                                        ],
                                        "parcoords": [
                                            {
                                                "line": {
                                                    "colorbar": {
                                                        "outlinewidth": 0,
                                                        "ticks": ""
                                                    }
                                                },
                                                "type": "parcoords"
                                            }
                                        ],
                                        "pie": [
                                            {
                                                "automargin": true,
                                                "type": "pie"
                                            }
                                        ],
                                        "scatter": [
                                            {
                                                "marker": {
                                                    "colorbar": {
                                                        "outlinewidth": 0,
                                                        "ticks": ""
                                                    }
                                                },
                                                "type": "scatter"
                                            }
                                        ],
                                        "scatter3d": [
                                            {
                                                "line": {
                                                    "colorbar": {
                                                        "outlinewidth": 0,
                                                        "ticks": ""
                                                    }
                                                },
                                                "marker": {
                                                    "colorbar": {
                                                        "outlinewidth": 0,
                                                        "ticks": ""
                                                    }
                                                },
                                                "type": "scatter3d"
                                            }
                                        ],
                                        "scattercarpet": [
                                            {
                                                "marker": {
                                                    "colorbar": {
                                                        "outlinewidth": 0,
                                                        "ticks": ""
                                                    }
                                                },
                                                "type": "scattercarpet"
                                            }
                                        ],
                                        "scattergeo": [
                                            {
                                                "marker": {
                                                    "colorbar": {
                                                        "outlinewidth": 0,
                                                        "ticks": ""
                                                    }
                                                },
                                                "type": "scattergeo"
                                            }
                                        ],
                                        "scattergl": [
                                            {
                                                "marker": {
                                                    "colorbar": {
                                                        "outlinewidth": 0,
                                                        "ticks": ""
                                                    }
                                                },
                                                "type": "scattergl"
                                            }
                                        ],
                                        "scattermapbox": [
                                            {
                                                "marker": {
                                                    "colorbar": {
                                                        "outlinewidth": 0,
                                                        "ticks": ""
                                                    }
                                                },
                                                "type": "scattermapbox"
                                            }
                                        ],
                                        "scatterpolar": [
                                            {
                                                "marker": {
                                                    "colorbar": {
                                                        "outlinewidth": 0,
                                                        "ticks": ""
                                                    }
                                                },
                                                "type": "scatterpolar"
                                            }
                                        ],
                                        "scatterpolargl": [
                                            {
                                                "marker": {
                                                    "colorbar": {
                                                        "outlinewidth": 0,
                                                        "ticks": ""
                                                    }
                                                },
                                                "type": "scatterpolargl"
                                            }
                                        ],
                                        "scatterternary": [
                                            {
                                                "marker": {
                                                    "colorbar": {
                                                        "outlinewidth": 0,
                                                        "ticks": ""
                                                    }
                                                },
                                                "type": "scatterternary"
                                            }
                                        ],
                                        "surface": [
                                            {
                                                "colorbar": {
                                                    "outlinewidth": 0,
                                                    "ticks": ""
                                                },
                                                "colorscale": [
                                                    [
                                                        0,
                                                        "#0d0887"
                                                    ],
                                                    [
                                                        0.1111111111111111,
                                                        "#46039f"
                                                    ],
                                                    [
                                                        0.2222222222222222,
                                                        "#7201a8"
                                                    ],
                                                    [
                                                        0.3333333333333333,
                                                        "#9c179e"
                                                    ],
                                                    [
                                                        0.4444444444444444,
                                                        "#bd3786"
                                                    ],
                                                    [
                                                        0.5555555555555556,
                                                        "#d8576b"
                                                    ],
                                                    [
                                                        0.6666666666666666,
                                                        "#ed7953"
                                                    ],
                                                    [
                                                        0.7777777777777778,
                                                        "#fb9f3a"
                                                    ],
                                                    [
                                                        0.8888888888888888,
                                                        "#fdca26"
                                                    ],
                                                    [
                                                        1,
                                                        "#f0f921"
                                                    ]
                                                ],
                                                "type": "surface"
                                            }
                                        ],
                                        "table": [
                                            {
                                                "cells": {
                                                    "fill": {
                                                        "color": "#EBF0F8"
                                                    },
                                                    "line": {
                                                        "color": "white"
                                                    }
                                                },
                                                "header": {
                                                    "fill": {
                                                        "color": "#C8D4E3"
                                                    },
                                                    "line": {
                                                        "color": "white"
                                                    }
                                                },
                                                "type": "table"
                                            }
                                        ]
                                    },
                                    "layout": {
                                        "annotationdefaults": {
                                            "arrowcolor": "#2a3f5f",
                                            "arrowhead": 0,
                                            "arrowwidth": 1
                                        },
                                        "autotypenumbers": "strict",
                                        "coloraxis": {
                                            "colorbar": {
                                                "outlinewidth": 0,
                                                "ticks": ""
                                            }
                                        },
                                        "colorscale": {
                                            "diverging": [
                                                [
                                                    0,
                                                    "#8e0152"
                                                ],
                                                [
                                                    0.1,
                                                    "#c51b7d"
                                                ],
                                                [
                                                    0.2,
                                                    "#de77ae"
                                                ],
                                                [
                                                    0.3,
                                                    "#f1b6da"
                                                ],
                                                [
                                                    0.4,
                                                    "#fde0ef"
                                                ],
                                                [
                                                    0.5,
                                                    "#f7f7f7"
                                                ],
                                                [
                                                    0.6,
                                                    "#e6f5d0"
                                                ],
                                                [
                                                    0.7,
                                                    "#b8e186"
                                                ],
                                                [
                                                    0.8,
                                                    "#7fbc41"
                                                ],
                                                [
                                                    0.9,
                                                    "#4d9221"
                                                ],
                                                [
                                                    1,
                                                    "#276419"
                                                ]
                                            ],
                                            "sequential": [
                                                [
                                                    0,
                                                    "#0d0887"
                                                ],
                                                [
                                                    0.1111111111111111,
                                                    "#46039f"
                                                ],
                                                [
                                                    0.2222222222222222,
                                                    "#7201a8"
                                                ],
                                                [
                                                    0.3333333333333333,
                                                    "#9c179e"
                                                ],
                                                [
                                                    0.4444444444444444,
                                                    "#bd3786"
                                                ],
                                                [
                                                    0.5555555555555556,
                                                    "#d8576b"
                                                ],
                                                [
                                                    0.6666666666666666,
                                                    "#ed7953"
                                                ],
                                                [
                                                    0.7777777777777778,
                                                    "#fb9f3a"
                                                ],
                                                [
                                                    0.8888888888888888,
                                                    "#fdca26"
                                                ],
                                                [
                                                    1,
                                                    "#f0f921"
                                                ]
                                            ],
                                            "sequentialminus": [
                                                [
                                                    0,
                                                    "#0d0887"
                                                ],
                                                [
                                                    0.1111111111111111,
                                                    "#46039f"
                                                ],
                                                [
                                                    0.2222222222222222,
                                                    "#7201a8"
                                                ],
                                                [
                                                    0.3333333333333333,
                                                    "#9c179e"
                                                ],
                                                [
                                                    0.4444444444444444,
                                                    "#bd3786"
                                                ],
                                                [
                                                    0.5555555555555556,
                                                    "#d8576b"
                                                ],
                                                [
                                                    0.6666666666666666,
                                                    "#ed7953"
                                                ],
                                                [
                                                    0.7777777777777778,
                                                    "#fb9f3a"
                                                ],
                                                [
                                                    0.8888888888888888,
                                                    "#fdca26"
                                                ],
                                                [
                                                    1,
                                                    "#f0f921"
                                                ]
                                            ]
                                        },
                                        "colorway": [
                                            "#636efa",
                                            "#EF553B",
                                            "#00cc96",
                                            "#ab63fa",
                                            "#FFA15A",
                                            "#19d3f3",
                                            "#FF6692",
                                            "#B6E880",
                                            "#FF97FF",
                                            "#FECB52"
                                        ],
                                        "font": {
                                            "color": "#2a3f5f"
                                        },
                                        "geo": {
                                            "bgcolor": "white",
                                            "lakecolor": "white",
                                            "landcolor": "#E5ECF6",
                                            "showlakes": true,
                                            "showland": true,
                                            "subunitcolor": "white"
                                        },
                                        "hoverlabel": {
                                            "align": "left"
                                        },
                                        "hovermode": "closest",
                                        "mapbox": {
                                            "style": "light"
                                        },
                                        "paper_bgcolor": "white",
                                        "plot_bgcolor": "#E5ECF6",
                                        "polar": {
                                            "angularaxis": {
                                                "gridcolor": "white",
                                                "linecolor": "white",
                                                "ticks": ""
                                            },
                                            "bgcolor": "#E5ECF6",
                                            "radialaxis": {
                                                "gridcolor": "white",
                                                "linecolor": "white",
                                                "ticks": ""
                                            }
                                        },
                                        "scene": {
                                            "xaxis": {
                                                "backgroundcolor": "#E5ECF6",
                                                "gridcolor": "white",
                                                "gridwidth": 2,
                                                "linecolor": "white",
                                                "showbackground": true,
                                                "ticks": "",
                                                "zerolinecolor": "white"
                                            },
                                            "yaxis": {
                                                "backgroundcolor": "#E5ECF6",
                                                "gridcolor": "white",
                                                "gridwidth": 2,
                                                "linecolor": "white",
                                                "showbackground": true,
                                                "ticks": "",
                                                "zerolinecolor": "white"
                                            },
                                            "zaxis": {
                                                "backgroundcolor": "#E5ECF6",
                                                "gridcolor": "white",
                                                "gridwidth": 2,
                                                "linecolor": "white",
                                                "showbackground": true,
                                                "ticks": "",
                                                "zerolinecolor": "white"
                                            }
                                        },
                                        "shapedefaults": {
                                            "line": {
                                                "color": "#2a3f5f"
                                            }
                                        },
                                        "ternary": {
                                            "aaxis": {
                                                "gridcolor": "white",
                                                "linecolor": "white",
                                                "ticks": ""
                                            },
                                            "baxis": {
                                                "gridcolor": "white",
                                                "linecolor": "white",
                                                "ticks": ""
                                            },
                                            "bgcolor": "#E5ECF6",
                                            "caxis": {
                                                "gridcolor": "white",
                                                "linecolor": "white",
                                                "ticks": ""
                                            }
                                        },
                                        "title": {
                                            "x": 0.05
                                        },
                                        "xaxis": {
                                            "automargin": true,
                                            "gridcolor": "white",
                                            "linecolor": "white",
                                            "ticks": "",
                                            "title": {
                                                "standoff": 15
                                            },
                                            "zerolinecolor": "white",
                                            "zerolinewidth": 2
                                        },
                                        "yaxis": {
                                            "automargin": true,
                                            "gridcolor": "white",
                                            "linecolor": "white",
                                            "ticks": "",
                                            "title": {
                                                "standoff": 15
                                            },
                                            "zerolinecolor": "white",
                                            "zerolinewidth": 2
                                        }
                                    }
                                },
                                "xaxis": {
                                    "anchor": "y",
                                    "domain": [
                                        0,
                                        1
                                    ],
                                    "title": {
                                        "text": "date"
                                    }
                                },
                                "yaxis": {
                                    "anchor": "x",
                                    "domain": [
                                        0,
                                        1
                                    ],
                                    "title": {
                                        "text": "Ingresos"
                                    }
                                }
                            }
                        }
                    },
                    "metadata": {}
                }
            ],
            "source": [
                "\t\n",
                "    grafico_horizontal= full_3.groupby(['country_id',\"date\"])[\"Ingresos\"].sum().reset_index()\n",
                "grafico_horizontal\n",
                "\n",
                "evolucion_horizontal= px.bar(grafico_horizontal, x=\"date\", y=\"Ingresos\", \\\n",
                "                        color='country_id', orientation=\"v\", color_discrete_sequence=px.colors.cyclical.mygbm)\n",
                "\n",
                "evolucion_horizontal.show()"
            ]
        },
        {
            "cell_type": "code",
            "source": [
                "menos_61[0:6]"
            ],
            "metadata": {
                "azdata_cell_guid": "46bf6517-6c0c-4049-9ff9-1a7e2c96d3f1"
            },
            "outputs": [
                {
                    "output_type": "execute_result",
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
                    "metadata": {},
                    "execution_count": 41
                }
            ],
            "execution_count": 41
        },
        {
            "cell_type": "code",
            "source": [
                "for i in menos_61[0:6]:\n",
                "   _dummy_dataset = pd.get_dummies(full_3[i], prefix=i)\n",
                "   full_bomselk = pd.concat([full_3,_dummy_dataset],axis=1)\n",
                "   full_bomselk.drop([i],axis=1, inplace=True)"
            ],
            "metadata": {
                "azdata_cell_guid": "cdb873b2-bbe7-418f-9da5-3692d7ce7564"
            },
            "outputs": [],
            "execution_count": 42
        },
        {
            "cell_type": "code",
            "source": [
                " full_bomselk.shape"
            ],
            "metadata": {
                "azdata_cell_guid": "d72bd3be-2147-4f55-b1b4-2630f9bbc087"
            },
            "outputs": [
                {
                    "output_type": "execute_result",
                    "data": {
                        "text/plain": [
                            "(6254518, 19)"
                        ]
                    },
                    "metadata": {},
                    "execution_count": 43
                }
            ],
            "execution_count": 43
        },
        {
            "cell_type": "code",
            "execution_count": 44,
            "metadata": {},
            "outputs": [],
            "source": [
                "for i in menos_61[0:6]:\n",
                "   _dummy_dataset = pd.get_dummies(full_3[i], prefix=i)\n",
                "   full_3 = pd.concat([full_3,_dummy_dataset],axis=1)\n",
                "   full_3.drop([i],axis=1, inplace=True)"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": 45,
            "metadata": {},
            "outputs": [
                {
                    "output_type": "execute_result",
                    "data": {
                        "text/plain": [
                            "(6254518, 130)"
                        ]
                    },
                    "metadata": {},
                    "execution_count": 45
                }
            ],
            "source": [
                " full_3.shape"
            ]
        },
        {
            "cell_type": "code",
            "source": [
                " full_3.info(verbose=True)"
            ],
            "metadata": {
                "azdata_cell_guid": "645aa984-054b-4586-9984-38fe364d27f6"
            },
            "outputs": [
                {
                    "output_type": "stream",
                    "name": "stdout",
                    "text": [
                        "<class 'pandas.core.frame.DataFrame'>\nInt64Index: 6254518 entries, 22134 to 89443859\nData columns (total 130 columns):\n #   Column                        Dtype         \n---  ------                        -----         \n 0   Ventas_cant                   float64       \n 1   Precios                       int64         \n 2   Ingresos                      float64       \n 3   entry_date                    datetime64[ns]\n 4   días_encartera                float64       \n 5   pk_cid                        object        \n 6   region_code                   float64       \n 7   age                           int64         \n 8   date                          datetime64[ns]\n 9   Month                         int64         \n 10  days_between                  float64       \n 11  recurrencia                   int64         \n 12  Productos_credit_card         uint8         \n 13  Productos_debit_card          uint8         \n 14  Productos_em_account_p        uint8         \n 15  Productos_em_acount           uint8         \n 16  Productos_emc_account         uint8         \n 17  Productos_funds               uint8         \n 18  Productos_loans               uint8         \n 19  Productos_long_term_deposit   uint8         \n 20  Productos_mortgage            uint8         \n 21  Productos_payroll             uint8         \n 22  Productos_payroll_account     uint8         \n 23  Productos_pension_plan        uint8         \n 24  Productos_securities          uint8         \n 25  Productos_short_term_deposit  uint8         \n 26  Familia_prod_AhorroVista      uint8         \n 27  Familia_prod_Crédito          uint8         \n 28  Familia_prod_Inversión        uint8         \n 29  entry_channel_                uint8         \n 30  entry_channel_004             uint8         \n 31  entry_channel_007             uint8         \n 32  entry_channel_013             uint8         \n 33  entry_channel_KAA             uint8         \n 34  entry_channel_KAB             uint8         \n 35  entry_channel_KAD             uint8         \n 36  entry_channel_KAE             uint8         \n 37  entry_channel_KAF             uint8         \n 38  entry_channel_KAG             uint8         \n 39  entry_channel_KAH             uint8         \n 40  entry_channel_KAJ             uint8         \n 41  entry_channel_KAK             uint8         \n 42  entry_channel_KAM             uint8         \n 43  entry_channel_KAQ             uint8         \n 44  entry_channel_KAR             uint8         \n 45  entry_channel_KAS             uint8         \n 46  entry_channel_KAT             uint8         \n 47  entry_channel_KAW             uint8         \n 48  entry_channel_KAY             uint8         \n 49  entry_channel_KAZ             uint8         \n 50  entry_channel_KBE             uint8         \n 51  entry_channel_KBG             uint8         \n 52  entry_channel_KBH             uint8         \n 53  entry_channel_KBO             uint8         \n 54  entry_channel_KBW             uint8         \n 55  entry_channel_KBZ             uint8         \n 56  entry_channel_KCB             uint8         \n 57  entry_channel_KCC             uint8         \n 58  entry_channel_KCH             uint8         \n 59  entry_channel_KCI             uint8         \n 60  entry_channel_KCK             uint8         \n 61  entry_channel_KCL             uint8         \n 62  entry_channel_KDA             uint8         \n 63  entry_channel_KDH             uint8         \n 64  entry_channel_KDR             uint8         \n 65  entry_channel_KDT             uint8         \n 66  entry_channel_KEH             uint8         \n 67  entry_channel_KEY             uint8         \n 68  entry_channel_KFA             uint8         \n 69  entry_channel_KFC             uint8         \n 70  entry_channel_KFD             uint8         \n 71  entry_channel_KFK             uint8         \n 72  entry_channel_KFL             uint8         \n 73  entry_channel_KFS             uint8         \n 74  entry_channel_KGC             uint8         \n 75  entry_channel_KGN             uint8         \n 76  entry_channel_KGX             uint8         \n 77  entry_channel_KHC             uint8         \n 78  entry_channel_KHD             uint8         \n 79  entry_channel_KHE             uint8         \n 80  entry_channel_KHF             uint8         \n 81  entry_channel_KHK             uint8         \n 82  entry_channel_KHL             uint8         \n 83  entry_channel_KHM             uint8         \n 84  entry_channel_KHN             uint8         \n 85  entry_channel_KHO             uint8         \n 86  entry_channel_KHP             uint8         \n 87  entry_channel_KHQ             uint8         \n 88  entry_channel_RED             uint8         \n 89  country_id_AR                 uint8         \n 90  country_id_AT                 uint8         \n 91  country_id_BE                 uint8         \n 92  country_id_BR                 uint8         \n 93  country_id_CA                 uint8         \n 94  country_id_CH                 uint8         \n 95  country_id_CI                 uint8         \n 96  country_id_CL                 uint8         \n 97  country_id_CM                 uint8         \n 98  country_id_CN                 uint8         \n 99  country_id_CO                 uint8         \n 100 country_id_DE                 uint8         \n 101 country_id_DO                 uint8         \n 102 country_id_DZ                 uint8         \n 103 country_id_ES                 uint8         \n 104 country_id_ET                 uint8         \n 105 country_id_FR                 uint8         \n 106 country_id_GA                 uint8         \n 107 country_id_GB                 uint8         \n 108 country_id_GT                 uint8         \n 109 country_id_IE                 uint8         \n 110 country_id_IT                 uint8         \n 111 country_id_LU                 uint8         \n 112 country_id_MA                 uint8         \n 113 country_id_MR                 uint8         \n 114 country_id_MX                 uint8         \n 115 country_id_NO                 uint8         \n 116 country_id_PE                 uint8         \n 117 country_id_PL                 uint8         \n 118 country_id_QA                 uint8         \n 119 country_id_RO                 uint8         \n 120 country_id_RU                 uint8         \n 121 country_id_SA                 uint8         \n 122 country_id_SE                 uint8         \n 123 country_id_SN                 uint8         \n 124 country_id_US                 uint8         \n 125 country_id_VE                 uint8         \n 126 gender_H                      uint8         \n 127 gender_V                      uint8         \n 128 deceased_N                    uint8         \n 129 deceased_S                    uint8         \ndtypes: datetime64[ns](2), float64(5), int64(4), object(1), uint8(118)\nmemory usage: 1.3+ GB\n"
                    ]
                }
            ],
            "execution_count": 46
        },
        {
            "cell_type": "code",
            "source": [
                " full_3.drop(['date','entry_date'],axis=1,inplace=True)"
            ],
            "metadata": {
                "azdata_cell_guid": "863e7d85-ad74-4944-b48d-dd7946038ac0"
            },
            "outputs": [],
            "execution_count": 47
        },
        {
            "cell_type": "code",
            "source": [
                "type('days_between')"
            ],
            "metadata": {
                "azdata_cell_guid": "229797f1-e86b-4a33-8b83-139571551ac9"
            },
            "outputs": [
                {
                    "output_type": "execute_result",
                    "data": {
                        "text/plain": [
                            "str"
                        ]
                    },
                    "metadata": {},
                    "execution_count": 48
                }
            ],
            "execution_count": 48
        },
        {
            "cell_type": "code",
            "source": [
                "(full_3.corr()).style.background_gradient(cmap=\"coolwarm\")"
            ],
            "metadata": {
                "azdata_cell_guid": "547a0fe8-9ac7-40f6-a083-f9185af7b526"
            },
            "outputs": [
                {
                    "output_type": "execute_result",
                    "data": {
                        "text/plain": [
                            "<pandas.io.formats.style.Styler at 0x23cd8537070>"
                        ],
                    },
                    "metadata": {},
                    "execution_count": 49
                }
            ],
            "execution_count": 49
        },
        {
            "cell_type": "code",
            "source": [
                "full_4=full_3\n",
                "pd.to_pickle(full_4, './full_4.pickle')"
            ],
            "metadata": {
                "azdata_cell_guid": "d823b65b-ad06-4f7e-bf4b-a1cd4591af7b"
            },
            "outputs": [],
            "execution_count": 50
        },
        {
            "cell_type": "code",
            "source": [
                "full_4.head()"
            ],
            "metadata": {
                "azdata_cell_guid": "9273d8fc-d764-43b6-bc56-0de85d3f6124"
            },
            "outputs": [
                {
                    "output_type": "execute_result",
                    "data": {
                        "text/plain": [
                            "       Ventas_cant  Precios  Ingresos  días_encartera   pk_cid  region_code  \\\n",
                            "22134          1.0       40      40.0            20.0   100296         28.0   \n",
                            "22135          1.0       40      40.0            50.0   100296         28.0   \n",
                            "22136          1.0       40      40.0            81.0   100296         28.0   \n",
                            "27826          1.0       40      40.0          1337.0  1003705         28.0   \n",
                            "27827          1.0       40      40.0          1367.0  1003705         28.0   \n",
                            "\n",
                            "       age  Month  days_between  recurrencia  ...  country_id_RU  \\\n",
                            "22134   46      4           0.0            1  ...              0   \n",
                            "22135   46      5          30.0            2  ...              0   \n",
                            "22136   46      6          31.0            3  ...              0   \n",
                            "27826   33      9           0.0            1  ...              0   \n",
                            "27827   33     10          30.0            2  ...              0   \n",
                            "\n",
                            "       country_id_SA  country_id_SE  country_id_SN  country_id_US  \\\n",
                            "22134              0              0              0              0   \n",
                            "22135              0              0              0              0   \n",
                            "22136              0              0              0              0   \n",
                            "27826              0              0              0              0   \n",
                            "27827              0              0              0              0   \n",
                            "\n",
                            "       country_id_VE  gender_H  gender_V  deceased_N  deceased_S  \n",
                            "22134              0         0         1           1           0  \n",
                            "22135              0         0         1           1           0  \n",
                            "22136              0         0         1           1           0  \n",
                            "27826              0         1         0           1           0  \n",
                            "27827              0         1         0           1           0  \n",
                            "\n",
                            "[5 rows x 128 columns]"
                        ],
                        "text/html": "<div>\n<style scoped>\n    .dataframe tbody tr th:only-of-type {\n        vertical-align: middle;\n    }\n\n    .dataframe tbody tr th {\n        vertical-align: top;\n    }\n\n    .dataframe thead th {\n        text-align: right;\n    }\n</style>\n<table border=\"1\" class=\"dataframe\">\n  <thead>\n    <tr style=\"text-align: right;\">\n      <th></th>\n      <th>Ventas_cant</th>\n      <th>Precios</th>\n      <th>Ingresos</th>\n      <th>días_encartera</th>\n      <th>pk_cid</th>\n      <th>region_code</th>\n      <th>age</th>\n      <th>Month</th>\n      <th>days_between</th>\n      <th>recurrencia</th>\n      <th>...</th>\n      <th>country_id_RU</th>\n      <th>country_id_SA</th>\n      <th>country_id_SE</th>\n      <th>country_id_SN</th>\n      <th>country_id_US</th>\n      <th>country_id_VE</th>\n      <th>gender_H</th>\n      <th>gender_V</th>\n      <th>deceased_N</th>\n      <th>deceased_S</th>\n    </tr>\n  </thead>\n  <tbody>\n    <tr>\n      <th>22134</th>\n      <td>1.0</td>\n      <td>40</td>\n      <td>40.0</td>\n      <td>20.0</td>\n      <td>100296</td>\n      <td>28.0</td>\n      <td>46</td>\n      <td>4</td>\n      <td>0.0</td>\n      <td>1</td>\n      <td>...</td>\n      <td>0</td>\n      <td>0</td>\n      <td>0</td>\n      <td>0</td>\n      <td>0</td>\n      <td>0</td>\n      <td>0</td>\n      <td>1</td>\n      <td>1</td>\n      <td>0</td>\n    </tr>\n    <tr>\n      <th>22135</th>\n      <td>1.0</td>\n      <td>40</td>\n      <td>40.0</td>\n      <td>50.0</td>\n      <td>100296</td>\n      <td>28.0</td>\n      <td>46</td>\n      <td>5</td>\n      <td>30.0</td>\n      <td>2</td>\n      <td>...</td>\n      <td>0</td>\n      <td>0</td>\n      <td>0</td>\n      <td>0</td>\n      <td>0</td>\n      <td>0</td>\n      <td>0</td>\n      <td>1</td>\n      <td>1</td>\n      <td>0</td>\n    </tr>\n    <tr>\n      <th>22136</th>\n      <td>1.0</td>\n      <td>40</td>\n      <td>40.0</td>\n      <td>81.0</td>\n      <td>100296</td>\n      <td>28.0</td>\n      <td>46</td>\n      <td>6</td>\n      <td>31.0</td>\n      <td>3</td>\n      <td>...</td>\n      <td>0</td>\n      <td>0</td>\n      <td>0</td>\n      <td>0</td>\n      <td>0</td>\n      <td>0</td>\n      <td>0</td>\n      <td>1</td>\n      <td>1</td>\n      <td>0</td>\n    </tr>\n    <tr>\n      <th>27826</th>\n      <td>1.0</td>\n      <td>40</td>\n      <td>40.0</td>\n      <td>1337.0</td>\n      <td>1003705</td>\n      <td>28.0</td>\n      <td>33</td>\n      <td>9</td>\n      <td>0.0</td>\n      <td>1</td>\n      <td>...</td>\n      <td>0</td>\n      <td>0</td>\n      <td>0</td>\n      <td>0</td>\n      <td>0</td>\n      <td>0</td>\n      <td>1</td>\n      <td>0</td>\n      <td>1</td>\n      <td>0</td>\n    </tr>\n    <tr>\n      <th>27827</th>\n      <td>1.0</td>\n      <td>40</td>\n      <td>40.0</td>\n      <td>1367.0</td>\n      <td>1003705</td>\n      <td>28.0</td>\n      <td>33</td>\n      <td>10</td>\n      <td>30.0</td>\n      <td>2</td>\n      <td>...</td>\n      <td>0</td>\n      <td>0</td>\n      <td>0</td>\n      <td>0</td>\n      <td>0</td>\n      <td>0</td>\n      <td>1</td>\n      <td>0</td>\n      <td>1</td>\n      <td>0</td>\n    </tr>\n  </tbody>\n</table>\n<p>5 rows × 128 columns</p>\n</div>"
                    },
                    "metadata": {},
                    "execution_count": 51
                }
            ],
            "execution_count": 51
        },
        {
            "cell_type": "code",
            "execution_count": 52,
            "metadata": {},
            "outputs": [],
            "source": [
                "full_4.set_index(\"pk_cid\", inplace = True)"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": 53,
            "metadata": {},
            "outputs": [
                {
                    "output_type": "execute_result",
                    "data": {
                        "text/plain": [
                            "         Ventas_cant  Precios  Ingresos  días_encartera  region_code  age  \\\n",
                            "pk_cid                                                                      \n",
                            "100296           1.0       40      40.0            20.0         28.0   46   \n",
                            "100296           1.0       40      40.0            50.0         28.0   46   \n",
                            "100296           1.0       40      40.0            81.0         28.0   46   \n",
                            "1003705          1.0       40      40.0          1337.0         28.0   33   \n",
                            "1003705          1.0       40      40.0          1367.0         28.0   33   \n",
                            "\n",
                            "         Month  days_between  recurrencia  Productos_credit_card  ...  \\\n",
                            "pk_cid                                                            ...   \n",
                            "100296       4           0.0            1                      0  ...   \n",
                            "100296       5          30.0            2                      0  ...   \n",
                            "100296       6          31.0            3                      0  ...   \n",
                            "1003705      9           0.0            1                      0  ...   \n",
                            "1003705     10          30.0            2                      0  ...   \n",
                            "\n",
                            "         country_id_RU  country_id_SA  country_id_SE  country_id_SN  \\\n",
                            "pk_cid                                                                \n",
                            "100296               0              0              0              0   \n",
                            "100296               0              0              0              0   \n",
                            "100296               0              0              0              0   \n",
                            "1003705              0              0              0              0   \n",
                            "1003705              0              0              0              0   \n",
                            "\n",
                            "         country_id_US  country_id_VE  gender_H  gender_V  deceased_N  \\\n",
                            "pk_cid                                                                  \n",
                            "100296               0              0         0         1           1   \n",
                            "100296               0              0         0         1           1   \n",
                            "100296               0              0         0         1           1   \n",
                            "1003705              0              0         1         0           1   \n",
                            "1003705              0              0         1         0           1   \n",
                            "\n",
                            "         deceased_S  \n",
                            "pk_cid               \n",
                            "100296            0  \n",
                            "100296            0  \n",
                            "100296            0  \n",
                            "1003705           0  \n",
                            "1003705           0  \n",
                            "\n",
                            "[5 rows x 127 columns]"
                        ],
                        "text/html": "<div>\n<style scoped>\n    .dataframe tbody tr th:only-of-type {\n        vertical-align: middle;\n    }\n\n    .dataframe tbody tr th {\n        vertical-align: top;\n    }\n\n    .dataframe thead th {\n        text-align: right;\n    }\n</style>\n<table border=\"1\" class=\"dataframe\">\n  <thead>\n    <tr style=\"text-align: right;\">\n      <th></th>\n      <th>Ventas_cant</th>\n      <th>Precios</th>\n      <th>Ingresos</th>\n      <th>días_encartera</th>\n      <th>region_code</th>\n      <th>age</th>\n      <th>Month</th>\n      <th>days_between</th>\n      <th>recurrencia</th>\n      <th>Productos_credit_card</th>\n      <th>...</th>\n      <th>country_id_RU</th>\n      <th>country_id_SA</th>\n      <th>country_id_SE</th>\n      <th>country_id_SN</th>\n      <th>country_id_US</th>\n      <th>country_id_VE</th>\n      <th>gender_H</th>\n      <th>gender_V</th>\n      <th>deceased_N</th>\n      <th>deceased_S</th>\n    </tr>\n    <tr>\n      <th>pk_cid</th>\n      <th></th>\n      <th></th>\n      <th></th>\n      <th></th>\n      <th></th>\n      <th></th>\n      <th></th>\n      <th></th>\n      <th></th>\n      <th></th>\n      <th></th>\n      <th></th>\n      <th></th>\n      <th></th>\n      <th></th>\n      <th></th>\n      <th></th>\n      <th></th>\n      <th></th>\n      <th></th>\n      <th></th>\n    </tr>\n  </thead>\n  <tbody>\n    <tr>\n      <th>100296</th>\n      <td>1.0</td>\n      <td>40</td>\n      <td>40.0</td>\n      <td>20.0</td>\n      <td>28.0</td>\n      <td>46</td>\n      <td>4</td>\n      <td>0.0</td>\n      <td>1</td>\n      <td>0</td>\n      <td>...</td>\n      <td>0</td>\n      <td>0</td>\n      <td>0</td>\n      <td>0</td>\n      <td>0</td>\n      <td>0</td>\n      <td>0</td>\n      <td>1</td>\n      <td>1</td>\n      <td>0</td>\n    </tr>\n    <tr>\n      <th>100296</th>\n      <td>1.0</td>\n      <td>40</td>\n      <td>40.0</td>\n      <td>50.0</td>\n      <td>28.0</td>\n      <td>46</td>\n      <td>5</td>\n      <td>30.0</td>\n      <td>2</td>\n      <td>0</td>\n      <td>...</td>\n      <td>0</td>\n      <td>0</td>\n      <td>0</td>\n      <td>0</td>\n      <td>0</td>\n      <td>0</td>\n      <td>0</td>\n      <td>1</td>\n      <td>1</td>\n      <td>0</td>\n    </tr>\n    <tr>\n      <th>100296</th>\n      <td>1.0</td>\n      <td>40</td>\n      <td>40.0</td>\n      <td>81.0</td>\n      <td>28.0</td>\n      <td>46</td>\n      <td>6</td>\n      <td>31.0</td>\n      <td>3</td>\n      <td>0</td>\n      <td>...</td>\n      <td>0</td>\n      <td>0</td>\n      <td>0</td>\n      <td>0</td>\n      <td>0</td>\n      <td>0</td>\n      <td>0</td>\n      <td>1</td>\n      <td>1</td>\n      <td>0</td>\n    </tr>\n    <tr>\n      <th>1003705</th>\n      <td>1.0</td>\n      <td>40</td>\n      <td>40.0</td>\n      <td>1337.0</td>\n      <td>28.0</td>\n      <td>33</td>\n      <td>9</td>\n      <td>0.0</td>\n      <td>1</td>\n      <td>0</td>\n      <td>...</td>\n      <td>0</td>\n      <td>0</td>\n      <td>0</td>\n      <td>0</td>\n      <td>0</td>\n      <td>0</td>\n      <td>1</td>\n      <td>0</td>\n      <td>1</td>\n      <td>0</td>\n    </tr>\n    <tr>\n      <th>1003705</th>\n      <td>1.0</td>\n      <td>40</td>\n      <td>40.0</td>\n      <td>1367.0</td>\n      <td>28.0</td>\n      <td>33</td>\n      <td>10</td>\n      <td>30.0</td>\n      <td>2</td>\n      <td>0</td>\n      <td>...</td>\n      <td>0</td>\n      <td>0</td>\n      <td>0</td>\n      <td>0</td>\n      <td>0</td>\n      <td>0</td>\n      <td>1</td>\n      <td>0</td>\n      <td>1</td>\n      <td>0</td>\n    </tr>\n  </tbody>\n</table>\n<p>5 rows × 127 columns</p>\n</div>"
                    },
                    "metadata": {},
                    "execution_count": 53
                }
            ],
            "source": [
                "full_4.head()"
            ]
        },
        {
            "cell_type": "code",
            "source": [
                "\n",
                "standard_scaler = StandardScaler()\n",
                "scaled_df = standard_scaler.fit_transform(full_4)\n",
                "scaled_df = pd.DataFrame(scaled_df, index = full_4.index, columns = full_4.columns)\n",
                "scaled_df.shape"
            ],
            "metadata": {
                "azdata_cell_guid": "41b3724f-c128-43c6-ad16-2a4c0be6078f"
            },
            "outputs": [
                {
                    "output_type": "execute_result",
                    "data": {
                        "text/plain": [
                            "(6254518, 127)"
                        ]
                    },
                    "metadata": {},
                    "execution_count": 54
                }
            ],
            "execution_count": 54
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
            "execution_count": 55,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Los dos pasos anteriores se pueden hacer del tirón usando fit_transform\n",
                "knn_imputer = KNNImputer()\n",
                "df_imputed = knn_imputer.fit_transform(full_4)"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": 56,
            "metadata": {},
            "outputs": [
                {
                    "output_type": "execute_result",
                    "data": {
                        "text/plain": [
                            "         Ventas_cant  Precios  Ingresos  días_encartera  region_code   age  \\\n",
                            "pk_cid                                                                       \n",
                            "100296           1.0     40.0      40.0            20.0         28.0  46.0   \n",
                            "100296           1.0     40.0      40.0            50.0         28.0  46.0   \n",
                            "100296           1.0     40.0      40.0            81.0         28.0  46.0   \n",
                            "1003705          1.0     40.0      40.0          1337.0         28.0  33.0   \n",
                            "1003705          1.0     40.0      40.0          1367.0         28.0  33.0   \n",
                            "...              ...      ...       ...             ...          ...   ...   \n",
                            "999892           1.0     10.0      10.0           672.0         28.0  37.0   \n",
                            "999892           1.0     10.0      10.0           703.0         28.0  37.0   \n",
                            "999892           1.0     10.0      10.0           731.0         28.0  38.0   \n",
                            "999892           1.0     10.0      10.0           762.0         28.0  38.0   \n",
                            "999892           1.0     10.0      10.0           792.0         28.0  38.0   \n",
                            "\n",
                            "         Month  days_between  recurrencia  Productos_credit_card  ...  \\\n",
                            "pk_cid                                                            ...   \n",
                            "100296     4.0           0.0          1.0                    0.0  ...   \n",
                            "100296     5.0          30.0          2.0                    0.0  ...   \n",
                            "100296     6.0          31.0          3.0                    0.0  ...   \n",
                            "1003705    9.0           0.0          1.0                    0.0  ...   \n",
                            "1003705   10.0          30.0          2.0                    0.0  ...   \n",
                            "...        ...           ...          ...                    ...  ...   \n",
                            "999892     1.0          31.0         68.0                    0.0  ...   \n",
                            "999892     2.0          31.0         69.0                    0.0  ...   \n",
                            "999892     3.0          28.0         70.0                    0.0  ...   \n",
                            "999892     4.0          31.0         71.0                    0.0  ...   \n",
                            "999892     5.0          30.0         72.0                    0.0  ...   \n",
                            "\n",
                            "         country_id_RU  country_id_SA  country_id_SE  country_id_SN  \\\n",
                            "pk_cid                                                                \n",
                            "100296             0.0            0.0            0.0            0.0   \n",
                            "100296             0.0            0.0            0.0            0.0   \n",
                            "100296             0.0            0.0            0.0            0.0   \n",
                            "1003705            0.0            0.0            0.0            0.0   \n",
                            "1003705            0.0            0.0            0.0            0.0   \n",
                            "...                ...            ...            ...            ...   \n",
                            "999892             0.0            0.0            0.0            0.0   \n",
                            "999892             0.0            0.0            0.0            0.0   \n",
                            "999892             0.0            0.0            0.0            0.0   \n",
                            "999892             0.0            0.0            0.0            0.0   \n",
                            "999892             0.0            0.0            0.0            0.0   \n",
                            "\n",
                            "         country_id_US  country_id_VE  gender_H  gender_V  deceased_N  \\\n",
                            "pk_cid                                                                  \n",
                            "100296             0.0            0.0       0.0       1.0         1.0   \n",
                            "100296             0.0            0.0       0.0       1.0         1.0   \n",
                            "100296             0.0            0.0       0.0       1.0         1.0   \n",
                            "1003705            0.0            0.0       1.0       0.0         1.0   \n",
                            "1003705            0.0            0.0       1.0       0.0         1.0   \n",
                            "...                ...            ...       ...       ...         ...   \n",
                            "999892             0.0            0.0       1.0       0.0         1.0   \n",
                            "999892             0.0            0.0       1.0       0.0         1.0   \n",
                            "999892             0.0            0.0       1.0       0.0         1.0   \n",
                            "999892             0.0            0.0       1.0       0.0         1.0   \n",
                            "999892             0.0            0.0       1.0       0.0         1.0   \n",
                            "\n",
                            "         deceased_S  \n",
                            "pk_cid               \n",
                            "100296          0.0  \n",
                            "100296          0.0  \n",
                            "100296          0.0  \n",
                            "1003705         0.0  \n",
                            "1003705         0.0  \n",
                            "...             ...  \n",
                            "999892          0.0  \n",
                            "999892          0.0  \n",
                            "999892          0.0  \n",
                            "999892          0.0  \n",
                            "999892          0.0  \n",
                            "\n",
                            "[6254518 rows x 127 columns]"
                        ],
                        "text/html": "<div>\n<style scoped>\n    .dataframe tbody tr th:only-of-type {\n        vertical-align: middle;\n    }\n\n    .dataframe tbody tr th {\n        vertical-align: top;\n    }\n\n    .dataframe thead th {\n        text-align: right;\n    }\n</style>\n<table border=\"1\" class=\"dataframe\">\n  <thead>\n    <tr style=\"text-align: right;\">\n      <th></th>\n      <th>Ventas_cant</th>\n      <th>Precios</th>\n      <th>Ingresos</th>\n      <th>días_encartera</th>\n      <th>region_code</th>\n      <th>age</th>\n      <th>Month</th>\n      <th>days_between</th>\n      <th>recurrencia</th>\n      <th>Productos_credit_card</th>\n      <th>...</th>\n      <th>country_id_RU</th>\n      <th>country_id_SA</th>\n      <th>country_id_SE</th>\n      <th>country_id_SN</th>\n      <th>country_id_US</th>\n      <th>country_id_VE</th>\n      <th>gender_H</th>\n      <th>gender_V</th>\n      <th>deceased_N</th>\n      <th>deceased_S</th>\n    </tr>\n    <tr>\n      <th>pk_cid</th>\n      <th></th>\n      <th></th>\n      <th></th>\n      <th></th>\n      <th></th>\n      <th></th>\n      <th></th>\n      <th></th>\n      <th></th>\n      <th></th>\n      <th></th>\n      <th></th>\n      <th></th>\n      <th></th>\n      <th></th>\n      <th></th>\n      <th></th>\n      <th></th>\n      <th></th>\n      <th></th>\n      <th></th>\n    </tr>\n  </thead>\n  <tbody>\n    <tr>\n      <th>100296</th>\n      <td>1.0</td>\n      <td>40.0</td>\n      <td>40.0</td>\n      <td>20.0</td>\n      <td>28.0</td>\n      <td>46.0</td>\n      <td>4.0</td>\n      <td>0.0</td>\n      <td>1.0</td>\n      <td>0.0</td>\n      <td>...</td>\n      <td>0.0</td>\n      <td>0.0</td>\n      <td>0.0</td>\n      <td>0.0</td>\n      <td>0.0</td>\n      <td>0.0</td>\n      <td>0.0</td>\n      <td>1.0</td>\n      <td>1.0</td>\n      <td>0.0</td>\n    </tr>\n    <tr>\n      <th>100296</th>\n      <td>1.0</td>\n      <td>40.0</td>\n      <td>40.0</td>\n      <td>50.0</td>\n      <td>28.0</td>\n      <td>46.0</td>\n      <td>5.0</td>\n      <td>30.0</td>\n      <td>2.0</td>\n      <td>0.0</td>\n      <td>...</td>\n      <td>0.0</td>\n      <td>0.0</td>\n      <td>0.0</td>\n      <td>0.0</td>\n      <td>0.0</td>\n      <td>0.0</td>\n      <td>0.0</td>\n      <td>1.0</td>\n      <td>1.0</td>\n      <td>0.0</td>\n    </tr>\n    <tr>\n      <th>100296</th>\n      <td>1.0</td>\n      <td>40.0</td>\n      <td>40.0</td>\n      <td>81.0</td>\n      <td>28.0</td>\n      <td>46.0</td>\n      <td>6.0</td>\n      <td>31.0</td>\n      <td>3.0</td>\n      <td>0.0</td>\n      <td>...</td>\n      <td>0.0</td>\n      <td>0.0</td>\n      <td>0.0</td>\n      <td>0.0</td>\n      <td>0.0</td>\n      <td>0.0</td>\n      <td>0.0</td>\n      <td>1.0</td>\n      <td>1.0</td>\n      <td>0.0</td>\n    </tr>\n    <tr>\n      <th>1003705</th>\n      <td>1.0</td>\n      <td>40.0</td>\n      <td>40.0</td>\n      <td>1337.0</td>\n      <td>28.0</td>\n      <td>33.0</td>\n      <td>9.0</td>\n      <td>0.0</td>\n      <td>1.0</td>\n      <td>0.0</td>\n      <td>...</td>\n      <td>0.0</td>\n      <td>0.0</td>\n      <td>0.0</td>\n      <td>0.0</td>\n      <td>0.0</td>\n      <td>0.0</td>\n      <td>1.0</td>\n      <td>0.0</td>\n      <td>1.0</td>\n      <td>0.0</td>\n    </tr>\n    <tr>\n      <th>1003705</th>\n      <td>1.0</td>\n      <td>40.0</td>\n      <td>40.0</td>\n      <td>1367.0</td>\n      <td>28.0</td>\n      <td>33.0</td>\n      <td>10.0</td>\n      <td>30.0</td>\n      <td>2.0</td>\n      <td>0.0</td>\n      <td>...</td>\n      <td>0.0</td>\n      <td>0.0</td>\n      <td>0.0</td>\n      <td>0.0</td>\n      <td>0.0</td>\n      <td>0.0</td>\n      <td>1.0</td>\n      <td>0.0</td>\n      <td>1.0</td>\n      <td>0.0</td>\n    </tr>\n    <tr>\n      <th>...</th>\n      <td>...</td>\n      <td>...</td>\n      <td>...</td>\n      <td>...</td>\n      <td>...</td>\n      <td>...</td>\n      <td>...</td>\n      <td>...</td>\n      <td>...</td>\n      <td>...</td>\n      <td>...</td>\n      <td>...</td>\n      <td>...</td>\n      <td>...</td>\n      <td>...</td>\n      <td>...</td>\n      <td>...</td>\n      <td>...</td>\n      <td>...</td>\n      <td>...</td>\n      <td>...</td>\n    </tr>\n    <tr>\n      <th>999892</th>\n      <td>1.0</td>\n      <td>10.0</td>\n      <td>10.0</td>\n      <td>672.0</td>\n      <td>28.0</td>\n      <td>37.0</td>\n      <td>1.0</td>\n      <td>31.0</td>\n      <td>68.0</td>\n      <td>0.0</td>\n      <td>...</td>\n      <td>0.0</td>\n      <td>0.0</td>\n      <td>0.0</td>\n      <td>0.0</td>\n      <td>0.0</td>\n      <td>0.0</td>\n      <td>1.0</td>\n      <td>0.0</td>\n      <td>1.0</td>\n      <td>0.0</td>\n    </tr>\n    <tr>\n      <th>999892</th>\n      <td>1.0</td>\n      <td>10.0</td>\n      <td>10.0</td>\n      <td>703.0</td>\n      <td>28.0</td>\n      <td>37.0</td>\n      <td>2.0</td>\n      <td>31.0</td>\n      <td>69.0</td>\n      <td>0.0</td>\n      <td>...</td>\n      <td>0.0</td>\n      <td>0.0</td>\n      <td>0.0</td>\n      <td>0.0</td>\n      <td>0.0</td>\n      <td>0.0</td>\n      <td>1.0</td>\n      <td>0.0</td>\n      <td>1.0</td>\n      <td>0.0</td>\n    </tr>\n    <tr>\n      <th>999892</th>\n      <td>1.0</td>\n      <td>10.0</td>\n      <td>10.0</td>\n      <td>731.0</td>\n      <td>28.0</td>\n      <td>38.0</td>\n      <td>3.0</td>\n      <td>28.0</td>\n      <td>70.0</td>\n      <td>0.0</td>\n      <td>...</td>\n      <td>0.0</td>\n      <td>0.0</td>\n      <td>0.0</td>\n      <td>0.0</td>\n      <td>0.0</td>\n      <td>0.0</td>\n      <td>1.0</td>\n      <td>0.0</td>\n      <td>1.0</td>\n      <td>0.0</td>\n    </tr>\n    <tr>\n      <th>999892</th>\n      <td>1.0</td>\n      <td>10.0</td>\n      <td>10.0</td>\n      <td>762.0</td>\n      <td>28.0</td>\n      <td>38.0</td>\n      <td>4.0</td>\n      <td>31.0</td>\n      <td>71.0</td>\n      <td>0.0</td>\n      <td>...</td>\n      <td>0.0</td>\n      <td>0.0</td>\n      <td>0.0</td>\n      <td>0.0</td>\n      <td>0.0</td>\n      <td>0.0</td>\n      <td>1.0</td>\n      <td>0.0</td>\n      <td>1.0</td>\n      <td>0.0</td>\n    </tr>\n    <tr>\n      <th>999892</th>\n      <td>1.0</td>\n      <td>10.0</td>\n      <td>10.0</td>\n      <td>792.0</td>\n      <td>28.0</td>\n      <td>38.0</td>\n      <td>5.0</td>\n      <td>30.0</td>\n      <td>72.0</td>\n      <td>0.0</td>\n      <td>...</td>\n      <td>0.0</td>\n      <td>0.0</td>\n      <td>0.0</td>\n      <td>0.0</td>\n      <td>0.0</td>\n      <td>0.0</td>\n      <td>1.0</td>\n      <td>0.0</td>\n      <td>1.0</td>\n      <td>0.0</td>\n    </tr>\n  </tbody>\n</table>\n<p>6254518 rows × 127 columns</p>\n</div>"
                    },
                    "metadata": {},
                    "execution_count": 56
                }
            ],
            "source": [
                "# Convertimos nuestro df_imputed de numpy array a dataframe\n",
                "df_imputed = pd.DataFrame(df_imputed, index = full_4.index, columns = full_4.columns)\n",
                "df_imputed"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": 57,
            "metadata": {},
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
            "source": [
                "Reducción de la dimensionalidad con PCA "
            ],
            "cell_type": "markdown",
            "metadata": {}
        },
        {
            "cell_type": "code",
            "execution_count": 58,
            "metadata": {},
            "outputs": [
                {
                    "output_type": "stream",
                    "name": "stdout",
                    "text": [
                        "Total PCA took 1.83 minutes\n"
                    ]
                }
            ],
            "source": [
                "st = time.time()\n",
                "pca = PCA(n_components = 30)\n",
                "pca.fit(full_4)\n",
                "pca_samples = pca.transform(full_4)\n",
                "et = time.time()\n",
                "print(\"Total PCA took {} minutes\".format(round((et - st)/60, 2)))"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": 59,
            "metadata": {},
            "outputs": [
                {
                    "output_type": "execute_result",
                    "data": {
                        "text/plain": [
                            "(6254518, 127)"
                        ]
                    },
                    "metadata": {},
                    "execution_count": 59
                }
            ],
            "source": [
                "full_4.shape"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": 60,
            "metadata": {},
            "outputs": [
                {
                    "output_type": "execute_result",
                    "data": {
                        "text/plain": [
                            "array([9.68636771e-01, 2.70255752e-02, 1.38346109e-03, 1.16697292e-03,\n",
                            "       1.03753333e-03, 6.65136249e-04, 7.48945143e-05, 3.20675765e-06,\n",
                            "       1.57665954e-06, 9.37573736e-07, 8.20175717e-07, 4.57436750e-07,\n",
                            "       3.82399663e-07, 3.27005805e-07, 3.14564362e-07, 2.90565695e-07,\n",
                            "       2.23590897e-07, 2.22162903e-07, 1.63752999e-07, 1.35805345e-07,\n",
                            "       1.28244871e-07, 1.16835308e-07, 8.19117875e-08, 6.61380201e-08,\n",
                            "       5.63204444e-08, 3.79466795e-08, 2.16219965e-08, 2.00330613e-08,\n",
                            "       1.68027159e-08, 1.23472791e-08])"
                        ]
                    },
                    "metadata": {},
                    "execution_count": 60
                }
            ],
            "source": [
                "pca.explained_variance_ratio_"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": 61,
            "metadata": {},
            "outputs": [
                {
                    "output_type": "execute_result",
                    "data": {
                        "text/plain": [
                            "102.3"
                        ]
                    },
                    "metadata": {},
                    "execution_count": 61
                }
            ],
            "source": [
                "3069/30"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": 62,
            "metadata": {},
            "outputs": [
                {
                    "output_type": "execute_result",
                    "data": {
                        "text/plain": [
                            "'40%'"
                        ]
                    },
                    "metadata": {},
                    "execution_count": 62
                }
            ],
            "source": [
                "\"40%\""
            ]
        },
        {
            "cell_type": "code",
            "execution_count": 63,
            "metadata": {},
            "outputs": [
                {
                    "output_type": "display_data",
                    "data": {
                        "text/plain": "<Figure size 432x288 with 1 Axes>",
                        "image/svg+xml": "<?xml version=\"1.0\" encoding=\"utf-8\" standalone=\"no\"?>\r\n<!DOCTYPE svg PUBLIC \"-//W3C//DTD SVG 1.1//EN\"\r\n  \"http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd\">\r\n<!-- Created with matplotlib (https://matplotlib.org/) -->\r\n<svg height=\"262.19625pt\" version=\"1.1\" viewBox=\"0 0 400.145866 262.19625\" width=\"400.145866pt\" xmlns=\"http://www.w3.org/2000/svg\" xmlns:xlink=\"http://www.w3.org/1999/xlink\">\r\n <metadata>\r\n  <rdf:RDF xmlns:cc=\"http://creativecommons.org/ns#\" xmlns:dc=\"http://purl.org/dc/elements/1.1/\" xmlns:rdf=\"http://www.w3.org/1999/02/22-rdf-syntax-ns#\">\r\n   <cc:Work>\r\n    <dc:type rdf:resource=\"http://purl.org/dc/dcmitype/StillImage\"/>\r\n    <dc:date>2021-06-24T09:17:41.689527</dc:date>\r\n    <dc:format>image/svg+xml</dc:format>\r\n    <dc:creator>\r\n     <cc:Agent>\r\n      <dc:title>Matplotlib v3.3.2, https://matplotlib.org/</dc:title>\r\n     </cc:Agent>\r\n    </dc:creator>\r\n   </cc:Work>\r\n  </rdf:RDF>\r\n </metadata>\r\n <defs>\r\n  <style type=\"text/css\">*{stroke-linecap:butt;stroke-linejoin:round;}</style>\r\n </defs>\r\n <g id=\"figure_1\">\r\n  <g id=\"patch_1\">\r\n   <path d=\"M 0 262.19625 \r\nL 400.145866 262.19625 \r\nL 400.145866 0 \r\nL 0 0 \r\nz\r\n\" style=\"fill:none;\"/>\r\n  </g>\r\n  <g id=\"axes_1\">\r\n   <g id=\"patch_2\">\r\n    <path d=\"M 56.50625 224.64 \r\nL 391.30625 224.64 \r\nL 391.30625 7.2 \r\nL 56.50625 7.2 \r\nz\r\n\" style=\"fill:#ffffff;\"/>\r\n   </g>\r\n   <g id=\"matplotlib.axis_1\">\r\n    <g id=\"xtick_1\">\r\n     <g id=\"line2d_1\">\r\n      <defs>\r\n       <path d=\"M 0 0 \r\nL 0 3.5 \r\n\" id=\"m4b250dab11\" style=\"stroke:#000000;stroke-width:0.8;\"/>\r\n      </defs>\r\n      <g>\r\n       <use style=\"stroke:#000000;stroke-width:0.8;\" x=\"71.724432\" xlink:href=\"#m4b250dab11\" y=\"224.64\"/>\r\n      </g>\r\n     </g>\r\n     <g id=\"text_1\">\r\n      <!-- 0 -->\r\n      <g transform=\"translate(68.543182 239.238438)scale(0.1 -0.1)\">\r\n       <defs>\r\n        <path d=\"M 31.78125 66.40625 \r\nQ 24.171875 66.40625 20.328125 58.90625 \r\nQ 16.5 51.421875 16.5 36.375 \r\nQ 16.5 21.390625 20.328125 13.890625 \r\nQ 24.171875 6.390625 31.78125 6.390625 \r\nQ 39.453125 6.390625 43.28125 13.890625 \r\nQ 47.125 21.390625 47.125 36.375 \r\nQ 47.125 51.421875 43.28125 58.90625 \r\nQ 39.453125 66.40625 31.78125 66.40625 \r\nz\r\nM 31.78125 74.21875 \r\nQ 44.046875 74.21875 50.515625 64.515625 \r\nQ 56.984375 54.828125 56.984375 36.375 \r\nQ 56.984375 17.96875 50.515625 8.265625 \r\nQ 44.046875 -1.421875 31.78125 -1.421875 \r\nQ 19.53125 -1.421875 13.0625 8.265625 \r\nQ 6.59375 17.96875 6.59375 36.375 \r\nQ 6.59375 54.828125 13.0625 64.515625 \r\nQ 19.53125 74.21875 31.78125 74.21875 \r\nz\r\n\" id=\"DejaVuSans-48\"/>\r\n       </defs>\r\n       <use xlink:href=\"#DejaVuSans-48\"/>\r\n      </g>\r\n     </g>\r\n    </g>\r\n    <g id=\"xtick_2\">\r\n     <g id=\"line2d_2\">\r\n      <g>\r\n       <use style=\"stroke:#000000;stroke-width:0.8;\" x=\"124.200921\" xlink:href=\"#m4b250dab11\" y=\"224.64\"/>\r\n      </g>\r\n     </g>\r\n     <g id=\"text_2\">\r\n      <!-- 5 -->\r\n      <g transform=\"translate(121.019671 239.238438)scale(0.1 -0.1)\">\r\n       <defs>\r\n        <path d=\"M 10.796875 72.90625 \r\nL 49.515625 72.90625 \r\nL 49.515625 64.59375 \r\nL 19.828125 64.59375 \r\nL 19.828125 46.734375 \r\nQ 21.96875 47.46875 24.109375 47.828125 \r\nQ 26.265625 48.1875 28.421875 48.1875 \r\nQ 40.625 48.1875 47.75 41.5 \r\nQ 54.890625 34.8125 54.890625 23.390625 \r\nQ 54.890625 11.625 47.5625 5.09375 \r\nQ 40.234375 -1.421875 26.90625 -1.421875 \r\nQ 22.3125 -1.421875 17.546875 -0.640625 \r\nQ 12.796875 0.140625 7.71875 1.703125 \r\nL 7.71875 11.625 \r\nQ 12.109375 9.234375 16.796875 8.0625 \r\nQ 21.484375 6.890625 26.703125 6.890625 \r\nQ 35.15625 6.890625 40.078125 11.328125 \r\nQ 45.015625 15.765625 45.015625 23.390625 \r\nQ 45.015625 31 40.078125 35.4375 \r\nQ 35.15625 39.890625 26.703125 39.890625 \r\nQ 22.75 39.890625 18.8125 39.015625 \r\nQ 14.890625 38.140625 10.796875 36.28125 \r\nz\r\n\" id=\"DejaVuSans-53\"/>\r\n       </defs>\r\n       <use xlink:href=\"#DejaVuSans-53\"/>\r\n      </g>\r\n     </g>\r\n    </g>\r\n    <g id=\"xtick_3\">\r\n     <g id=\"line2d_3\">\r\n      <g>\r\n       <use style=\"stroke:#000000;stroke-width:0.8;\" x=\"176.67741\" xlink:href=\"#m4b250dab11\" y=\"224.64\"/>\r\n      </g>\r\n     </g>\r\n     <g id=\"text_3\">\r\n      <!-- 10 -->\r\n      <g transform=\"translate(170.31491 239.238438)scale(0.1 -0.1)\">\r\n       <defs>\r\n        <path d=\"M 12.40625 8.296875 \r\nL 28.515625 8.296875 \r\nL 28.515625 63.921875 \r\nL 10.984375 60.40625 \r\nL 10.984375 69.390625 \r\nL 28.421875 72.90625 \r\nL 38.28125 72.90625 \r\nL 38.28125 8.296875 \r\nL 54.390625 8.296875 \r\nL 54.390625 0 \r\nL 12.40625 0 \r\nz\r\n\" id=\"DejaVuSans-49\"/>\r\n       </defs>\r\n       <use xlink:href=\"#DejaVuSans-49\"/>\r\n       <use x=\"63.623047\" xlink:href=\"#DejaVuSans-48\"/>\r\n      </g>\r\n     </g>\r\n    </g>\r\n    <g id=\"xtick_4\">\r\n     <g id=\"line2d_4\">\r\n      <g>\r\n       <use style=\"stroke:#000000;stroke-width:0.8;\" x=\"229.153899\" xlink:href=\"#m4b250dab11\" y=\"224.64\"/>\r\n      </g>\r\n     </g>\r\n     <g id=\"text_4\">\r\n      <!-- 15 -->\r\n      <g transform=\"translate(222.791399 239.238438)scale(0.1 -0.1)\">\r\n       <use xlink:href=\"#DejaVuSans-49\"/>\r\n       <use x=\"63.623047\" xlink:href=\"#DejaVuSans-53\"/>\r\n      </g>\r\n     </g>\r\n    </g>\r\n    <g id=\"xtick_5\">\r\n     <g id=\"line2d_5\">\r\n      <g>\r\n       <use style=\"stroke:#000000;stroke-width:0.8;\" x=\"281.630388\" xlink:href=\"#m4b250dab11\" y=\"224.64\"/>\r\n      </g>\r\n     </g>\r\n     <g id=\"text_5\">\r\n      <!-- 20 -->\r\n      <g transform=\"translate(275.267888 239.238438)scale(0.1 -0.1)\">\r\n       <defs>\r\n        <path d=\"M 19.1875 8.296875 \r\nL 53.609375 8.296875 \r\nL 53.609375 0 \r\nL 7.328125 0 \r\nL 7.328125 8.296875 \r\nQ 12.9375 14.109375 22.625 23.890625 \r\nQ 32.328125 33.6875 34.8125 36.53125 \r\nQ 39.546875 41.84375 41.421875 45.53125 \r\nQ 43.3125 49.21875 43.3125 52.78125 \r\nQ 43.3125 58.59375 39.234375 62.25 \r\nQ 35.15625 65.921875 28.609375 65.921875 \r\nQ 23.96875 65.921875 18.8125 64.3125 \r\nQ 13.671875 62.703125 7.8125 59.421875 \r\nL 7.8125 69.390625 \r\nQ 13.765625 71.78125 18.9375 73 \r\nQ 24.125 74.21875 28.421875 74.21875 \r\nQ 39.75 74.21875 46.484375 68.546875 \r\nQ 53.21875 62.890625 53.21875 53.421875 \r\nQ 53.21875 48.921875 51.53125 44.890625 \r\nQ 49.859375 40.875 45.40625 35.40625 \r\nQ 44.1875 33.984375 37.640625 27.21875 \r\nQ 31.109375 20.453125 19.1875 8.296875 \r\nz\r\n\" id=\"DejaVuSans-50\"/>\r\n       </defs>\r\n       <use xlink:href=\"#DejaVuSans-50\"/>\r\n       <use x=\"63.623047\" xlink:href=\"#DejaVuSans-48\"/>\r\n      </g>\r\n     </g>\r\n    </g>\r\n    <g id=\"xtick_6\">\r\n     <g id=\"line2d_6\">\r\n      <g>\r\n       <use style=\"stroke:#000000;stroke-width:0.8;\" x=\"334.106877\" xlink:href=\"#m4b250dab11\" y=\"224.64\"/>\r\n      </g>\r\n     </g>\r\n     <g id=\"text_6\">\r\n      <!-- 25 -->\r\n      <g transform=\"translate(327.744377 239.238438)scale(0.1 -0.1)\">\r\n       <use xlink:href=\"#DejaVuSans-50\"/>\r\n       <use x=\"63.623047\" xlink:href=\"#DejaVuSans-53\"/>\r\n      </g>\r\n     </g>\r\n    </g>\r\n    <g id=\"xtick_7\">\r\n     <g id=\"line2d_7\">\r\n      <g>\r\n       <use style=\"stroke:#000000;stroke-width:0.8;\" x=\"386.583366\" xlink:href=\"#m4b250dab11\" y=\"224.64\"/>\r\n      </g>\r\n     </g>\r\n     <g id=\"text_7\">\r\n      <!-- 30 -->\r\n      <g transform=\"translate(380.220866 239.238438)scale(0.1 -0.1)\">\r\n       <defs>\r\n        <path d=\"M 40.578125 39.3125 \r\nQ 47.65625 37.796875 51.625 33 \r\nQ 55.609375 28.21875 55.609375 21.1875 \r\nQ 55.609375 10.40625 48.1875 4.484375 \r\nQ 40.765625 -1.421875 27.09375 -1.421875 \r\nQ 22.515625 -1.421875 17.65625 -0.515625 \r\nQ 12.796875 0.390625 7.625 2.203125 \r\nL 7.625 11.71875 \r\nQ 11.71875 9.328125 16.59375 8.109375 \r\nQ 21.484375 6.890625 26.8125 6.890625 \r\nQ 36.078125 6.890625 40.9375 10.546875 \r\nQ 45.796875 14.203125 45.796875 21.1875 \r\nQ 45.796875 27.640625 41.28125 31.265625 \r\nQ 36.765625 34.90625 28.71875 34.90625 \r\nL 20.21875 34.90625 \r\nL 20.21875 43.015625 \r\nL 29.109375 43.015625 \r\nQ 36.375 43.015625 40.234375 45.921875 \r\nQ 44.09375 48.828125 44.09375 54.296875 \r\nQ 44.09375 59.90625 40.109375 62.90625 \r\nQ 36.140625 65.921875 28.71875 65.921875 \r\nQ 24.65625 65.921875 20.015625 65.03125 \r\nQ 15.375 64.15625 9.8125 62.3125 \r\nL 9.8125 71.09375 \r\nQ 15.4375 72.65625 20.34375 73.4375 \r\nQ 25.25 74.21875 29.59375 74.21875 \r\nQ 40.828125 74.21875 47.359375 69.109375 \r\nQ 53.90625 64.015625 53.90625 55.328125 \r\nQ 53.90625 49.265625 50.4375 45.09375 \r\nQ 46.96875 40.921875 40.578125 39.3125 \r\nz\r\n\" id=\"DejaVuSans-51\"/>\r\n       </defs>\r\n       <use xlink:href=\"#DejaVuSans-51\"/>\r\n       <use x=\"63.623047\" xlink:href=\"#DejaVuSans-48\"/>\r\n      </g>\r\n     </g>\r\n    </g>\r\n    <g id=\"text_8\">\r\n     <!-- Nr. PC -->\r\n     <g transform=\"translate(208.882031 252.916563)scale(0.1 -0.1)\">\r\n      <defs>\r\n       <path d=\"M 9.8125 72.90625 \r\nL 23.09375 72.90625 \r\nL 55.421875 11.921875 \r\nL 55.421875 72.90625 \r\nL 64.984375 72.90625 \r\nL 64.984375 0 \r\nL 51.703125 0 \r\nL 19.390625 60.984375 \r\nL 19.390625 0 \r\nL 9.8125 0 \r\nz\r\n\" id=\"DejaVuSans-78\"/>\r\n       <path d=\"M 41.109375 46.296875 \r\nQ 39.59375 47.171875 37.8125 47.578125 \r\nQ 36.03125 48 33.890625 48 \r\nQ 26.265625 48 22.1875 43.046875 \r\nQ 18.109375 38.09375 18.109375 28.8125 \r\nL 18.109375 0 \r\nL 9.078125 0 \r\nL 9.078125 54.6875 \r\nL 18.109375 54.6875 \r\nL 18.109375 46.1875 \r\nQ 20.953125 51.171875 25.484375 53.578125 \r\nQ 30.03125 56 36.53125 56 \r\nQ 37.453125 56 38.578125 55.875 \r\nQ 39.703125 55.765625 41.0625 55.515625 \r\nz\r\n\" id=\"DejaVuSans-114\"/>\r\n       <path d=\"M 10.6875 12.40625 \r\nL 21 12.40625 \r\nL 21 0 \r\nL 10.6875 0 \r\nz\r\n\" id=\"DejaVuSans-46\"/>\r\n       <path id=\"DejaVuSans-32\"/>\r\n       <path d=\"M 19.671875 64.796875 \r\nL 19.671875 37.40625 \r\nL 32.078125 37.40625 \r\nQ 38.96875 37.40625 42.71875 40.96875 \r\nQ 46.484375 44.53125 46.484375 51.125 \r\nQ 46.484375 57.671875 42.71875 61.234375 \r\nQ 38.96875 64.796875 32.078125 64.796875 \r\nz\r\nM 9.8125 72.90625 \r\nL 32.078125 72.90625 \r\nQ 44.34375 72.90625 50.609375 67.359375 \r\nQ 56.890625 61.8125 56.890625 51.125 \r\nQ 56.890625 40.328125 50.609375 34.8125 \r\nQ 44.34375 29.296875 32.078125 29.296875 \r\nL 19.671875 29.296875 \r\nL 19.671875 0 \r\nL 9.8125 0 \r\nz\r\n\" id=\"DejaVuSans-80\"/>\r\n       <path d=\"M 64.40625 67.28125 \r\nL 64.40625 56.890625 \r\nQ 59.421875 61.53125 53.78125 63.8125 \r\nQ 48.140625 66.109375 41.796875 66.109375 \r\nQ 29.296875 66.109375 22.65625 58.46875 \r\nQ 16.015625 50.828125 16.015625 36.375 \r\nQ 16.015625 21.96875 22.65625 14.328125 \r\nQ 29.296875 6.6875 41.796875 6.6875 \r\nQ 48.140625 6.6875 53.78125 8.984375 \r\nQ 59.421875 11.28125 64.40625 15.921875 \r\nL 64.40625 5.609375 \r\nQ 59.234375 2.09375 53.4375 0.328125 \r\nQ 47.65625 -1.421875 41.21875 -1.421875 \r\nQ 24.65625 -1.421875 15.125 8.703125 \r\nQ 5.609375 18.84375 5.609375 36.375 \r\nQ 5.609375 53.953125 15.125 64.078125 \r\nQ 24.65625 74.21875 41.21875 74.21875 \r\nQ 47.75 74.21875 53.53125 72.484375 \r\nQ 59.328125 70.75 64.40625 67.28125 \r\nz\r\n\" id=\"DejaVuSans-67\"/>\r\n      </defs>\r\n      <use xlink:href=\"#DejaVuSans-78\"/>\r\n      <use x=\"74.804688\" xlink:href=\"#DejaVuSans-114\"/>\r\n      <use x=\"106.792969\" xlink:href=\"#DejaVuSans-46\"/>\r\n      <use x=\"138.580078\" xlink:href=\"#DejaVuSans-32\"/>\r\n      <use x=\"170.367188\" xlink:href=\"#DejaVuSans-80\"/>\r\n      <use x=\"230.669922\" xlink:href=\"#DejaVuSans-67\"/>\r\n     </g>\r\n    </g>\r\n   </g>\r\n   <g id=\"matplotlib.axis_2\">\r\n    <g id=\"ytick_1\">\r\n     <g id=\"line2d_8\">\r\n      <defs>\r\n       <path d=\"M 0 0 \r\nL -3.5 0 \r\n\" id=\"m91f5325c8b\" style=\"stroke:#000000;stroke-width:0.8;\"/>\r\n      </defs>\r\n      <g>\r\n       <use style=\"stroke:#000000;stroke-width:0.8;\" x=\"56.50625\" xlink:href=\"#m91f5325c8b\" y=\"206.164339\"/>\r\n      </g>\r\n     </g>\r\n     <g id=\"text_9\">\r\n      <!-- 0.970 -->\r\n      <g transform=\"translate(20.878125 209.963558)scale(0.1 -0.1)\">\r\n       <defs>\r\n        <path d=\"M 10.984375 1.515625 \r\nL 10.984375 10.5 \r\nQ 14.703125 8.734375 18.5 7.8125 \r\nQ 22.3125 6.890625 25.984375 6.890625 \r\nQ 35.75 6.890625 40.890625 13.453125 \r\nQ 46.046875 20.015625 46.78125 33.40625 \r\nQ 43.953125 29.203125 39.59375 26.953125 \r\nQ 35.25 24.703125 29.984375 24.703125 \r\nQ 19.046875 24.703125 12.671875 31.3125 \r\nQ 6.296875 37.9375 6.296875 49.421875 \r\nQ 6.296875 60.640625 12.9375 67.421875 \r\nQ 19.578125 74.21875 30.609375 74.21875 \r\nQ 43.265625 74.21875 49.921875 64.515625 \r\nQ 56.59375 54.828125 56.59375 36.375 \r\nQ 56.59375 19.140625 48.40625 8.859375 \r\nQ 40.234375 -1.421875 26.421875 -1.421875 \r\nQ 22.703125 -1.421875 18.890625 -0.6875 \r\nQ 15.09375 0.046875 10.984375 1.515625 \r\nz\r\nM 30.609375 32.421875 \r\nQ 37.25 32.421875 41.125 36.953125 \r\nQ 45.015625 41.5 45.015625 49.421875 \r\nQ 45.015625 57.28125 41.125 61.84375 \r\nQ 37.25 66.40625 30.609375 66.40625 \r\nQ 23.96875 66.40625 20.09375 61.84375 \r\nQ 16.21875 57.28125 16.21875 49.421875 \r\nQ 16.21875 41.5 20.09375 36.953125 \r\nQ 23.96875 32.421875 30.609375 32.421875 \r\nz\r\n\" id=\"DejaVuSans-57\"/>\r\n        <path d=\"M 8.203125 72.90625 \r\nL 55.078125 72.90625 \r\nL 55.078125 68.703125 \r\nL 28.609375 0 \r\nL 18.3125 0 \r\nL 43.21875 64.59375 \r\nL 8.203125 64.59375 \r\nz\r\n\" id=\"DejaVuSans-55\"/>\r\n       </defs>\r\n       <use xlink:href=\"#DejaVuSans-48\"/>\r\n       <use x=\"63.623047\" xlink:href=\"#DejaVuSans-46\"/>\r\n       <use x=\"95.410156\" xlink:href=\"#DejaVuSans-57\"/>\r\n       <use x=\"159.033203\" xlink:href=\"#DejaVuSans-55\"/>\r\n       <use x=\"222.65625\" xlink:href=\"#DejaVuSans-48\"/>\r\n      </g>\r\n     </g>\r\n    </g>\r\n    <g id=\"ytick_2\">\r\n     <g id=\"line2d_9\">\r\n      <g>\r\n       <use style=\"stroke:#000000;stroke-width:0.8;\" x=\"56.50625\" xlink:href=\"#m91f5325c8b\" y=\"174.650847\"/>\r\n      </g>\r\n     </g>\r\n     <g id=\"text_10\">\r\n      <!-- 0.975 -->\r\n      <g transform=\"translate(20.878125 178.450066)scale(0.1 -0.1)\">\r\n       <use xlink:href=\"#DejaVuSans-48\"/>\r\n       <use x=\"63.623047\" xlink:href=\"#DejaVuSans-46\"/>\r\n       <use x=\"95.410156\" xlink:href=\"#DejaVuSans-57\"/>\r\n       <use x=\"159.033203\" xlink:href=\"#DejaVuSans-55\"/>\r\n       <use x=\"222.65625\" xlink:href=\"#DejaVuSans-53\"/>\r\n      </g>\r\n     </g>\r\n    </g>\r\n    <g id=\"ytick_3\">\r\n     <g id=\"line2d_10\">\r\n      <g>\r\n       <use style=\"stroke:#000000;stroke-width:0.8;\" x=\"56.50625\" xlink:href=\"#m91f5325c8b\" y=\"143.137355\"/>\r\n      </g>\r\n     </g>\r\n     <g id=\"text_11\">\r\n      <!-- 0.980 -->\r\n      <g transform=\"translate(20.878125 146.936574)scale(0.1 -0.1)\">\r\n       <defs>\r\n        <path d=\"M 31.78125 34.625 \r\nQ 24.75 34.625 20.71875 30.859375 \r\nQ 16.703125 27.09375 16.703125 20.515625 \r\nQ 16.703125 13.921875 20.71875 10.15625 \r\nQ 24.75 6.390625 31.78125 6.390625 \r\nQ 38.8125 6.390625 42.859375 10.171875 \r\nQ 46.921875 13.96875 46.921875 20.515625 \r\nQ 46.921875 27.09375 42.890625 30.859375 \r\nQ 38.875 34.625 31.78125 34.625 \r\nz\r\nM 21.921875 38.8125 \r\nQ 15.578125 40.375 12.03125 44.71875 \r\nQ 8.5 49.078125 8.5 55.328125 \r\nQ 8.5 64.0625 14.71875 69.140625 \r\nQ 20.953125 74.21875 31.78125 74.21875 \r\nQ 42.671875 74.21875 48.875 69.140625 \r\nQ 55.078125 64.0625 55.078125 55.328125 \r\nQ 55.078125 49.078125 51.53125 44.71875 \r\nQ 48 40.375 41.703125 38.8125 \r\nQ 48.828125 37.15625 52.796875 32.3125 \r\nQ 56.78125 27.484375 56.78125 20.515625 \r\nQ 56.78125 9.90625 50.3125 4.234375 \r\nQ 43.84375 -1.421875 31.78125 -1.421875 \r\nQ 19.734375 -1.421875 13.25 4.234375 \r\nQ 6.78125 9.90625 6.78125 20.515625 \r\nQ 6.78125 27.484375 10.78125 32.3125 \r\nQ 14.796875 37.15625 21.921875 38.8125 \r\nz\r\nM 18.3125 54.390625 \r\nQ 18.3125 48.734375 21.84375 45.5625 \r\nQ 25.390625 42.390625 31.78125 42.390625 \r\nQ 38.140625 42.390625 41.71875 45.5625 \r\nQ 45.3125 48.734375 45.3125 54.390625 \r\nQ 45.3125 60.0625 41.71875 63.234375 \r\nQ 38.140625 66.40625 31.78125 66.40625 \r\nQ 25.390625 66.40625 21.84375 63.234375 \r\nQ 18.3125 60.0625 18.3125 54.390625 \r\nz\r\n\" id=\"DejaVuSans-56\"/>\r\n       </defs>\r\n       <use xlink:href=\"#DejaVuSans-48\"/>\r\n       <use x=\"63.623047\" xlink:href=\"#DejaVuSans-46\"/>\r\n       <use x=\"95.410156\" xlink:href=\"#DejaVuSans-57\"/>\r\n       <use x=\"159.033203\" xlink:href=\"#DejaVuSans-56\"/>\r\n       <use x=\"222.65625\" xlink:href=\"#DejaVuSans-48\"/>\r\n      </g>\r\n     </g>\r\n    </g>\r\n    <g id=\"ytick_4\">\r\n     <g id=\"line2d_11\">\r\n      <g>\r\n       <use style=\"stroke:#000000;stroke-width:0.8;\" x=\"56.50625\" xlink:href=\"#m91f5325c8b\" y=\"111.623863\"/>\r\n      </g>\r\n     </g>\r\n     <g id=\"text_12\">\r\n      <!-- 0.985 -->\r\n      <g transform=\"translate(20.878125 115.423082)scale(0.1 -0.1)\">\r\n       <use xlink:href=\"#DejaVuSans-48\"/>\r\n       <use x=\"63.623047\" xlink:href=\"#DejaVuSans-46\"/>\r\n       <use x=\"95.410156\" xlink:href=\"#DejaVuSans-57\"/>\r\n       <use x=\"159.033203\" xlink:href=\"#DejaVuSans-56\"/>\r\n       <use x=\"222.65625\" xlink:href=\"#DejaVuSans-53\"/>\r\n      </g>\r\n     </g>\r\n    </g>\r\n    <g id=\"ytick_5\">\r\n     <g id=\"line2d_12\">\r\n      <g>\r\n       <use style=\"stroke:#000000;stroke-width:0.8;\" x=\"56.50625\" xlink:href=\"#m91f5325c8b\" y=\"80.110371\"/>\r\n      </g>\r\n     </g>\r\n     <g id=\"text_13\">\r\n      <!-- 0.990 -->\r\n      <g transform=\"translate(20.878125 83.90959)scale(0.1 -0.1)\">\r\n       <use xlink:href=\"#DejaVuSans-48\"/>\r\n       <use x=\"63.623047\" xlink:href=\"#DejaVuSans-46\"/>\r\n       <use x=\"95.410156\" xlink:href=\"#DejaVuSans-57\"/>\r\n       <use x=\"159.033203\" xlink:href=\"#DejaVuSans-57\"/>\r\n       <use x=\"222.65625\" xlink:href=\"#DejaVuSans-48\"/>\r\n      </g>\r\n     </g>\r\n    </g>\r\n    <g id=\"ytick_6\">\r\n     <g id=\"line2d_13\">\r\n      <g>\r\n       <use style=\"stroke:#000000;stroke-width:0.8;\" x=\"56.50625\" xlink:href=\"#m91f5325c8b\" y=\"48.596879\"/>\r\n      </g>\r\n     </g>\r\n     <g id=\"text_14\">\r\n      <!-- 0.995 -->\r\n      <g transform=\"translate(20.878125 52.396098)scale(0.1 -0.1)\">\r\n       <use xlink:href=\"#DejaVuSans-48\"/>\r\n       <use x=\"63.623047\" xlink:href=\"#DejaVuSans-46\"/>\r\n       <use x=\"95.410156\" xlink:href=\"#DejaVuSans-57\"/>\r\n       <use x=\"159.033203\" xlink:href=\"#DejaVuSans-57\"/>\r\n       <use x=\"222.65625\" xlink:href=\"#DejaVuSans-53\"/>\r\n      </g>\r\n     </g>\r\n    </g>\r\n    <g id=\"ytick_7\">\r\n     <g id=\"line2d_14\">\r\n      <g>\r\n       <use style=\"stroke:#000000;stroke-width:0.8;\" x=\"56.50625\" xlink:href=\"#m91f5325c8b\" y=\"17.083387\"/>\r\n      </g>\r\n     </g>\r\n     <g id=\"text_15\">\r\n      <!-- 1.000 -->\r\n      <g transform=\"translate(20.878125 20.882606)scale(0.1 -0.1)\">\r\n       <use xlink:href=\"#DejaVuSans-49\"/>\r\n       <use x=\"63.623047\" xlink:href=\"#DejaVuSans-46\"/>\r\n       <use x=\"95.410156\" xlink:href=\"#DejaVuSans-48\"/>\r\n       <use x=\"159.033203\" xlink:href=\"#DejaVuSans-48\"/>\r\n       <use x=\"222.65625\" xlink:href=\"#DejaVuSans-48\"/>\r\n      </g>\r\n     </g>\r\n    </g>\r\n    <g id=\"text_16\">\r\n     <!-- Cumulative explained variance -->\r\n     <g transform=\"translate(14.798438 193.546563)rotate(-90)scale(0.1 -0.1)\">\r\n      <defs>\r\n       <path d=\"M 8.5 21.578125 \r\nL 8.5 54.6875 \r\nL 17.484375 54.6875 \r\nL 17.484375 21.921875 \r\nQ 17.484375 14.15625 20.5 10.265625 \r\nQ 23.53125 6.390625 29.59375 6.390625 \r\nQ 36.859375 6.390625 41.078125 11.03125 \r\nQ 45.3125 15.671875 45.3125 23.6875 \r\nL 45.3125 54.6875 \r\nL 54.296875 54.6875 \r\nL 54.296875 0 \r\nL 45.3125 0 \r\nL 45.3125 8.40625 \r\nQ 42.046875 3.421875 37.71875 1 \r\nQ 33.40625 -1.421875 27.6875 -1.421875 \r\nQ 18.265625 -1.421875 13.375 4.4375 \r\nQ 8.5 10.296875 8.5 21.578125 \r\nz\r\nM 31.109375 56 \r\nz\r\n\" id=\"DejaVuSans-117\"/>\r\n       <path d=\"M 52 44.1875 \r\nQ 55.375 50.25 60.0625 53.125 \r\nQ 64.75 56 71.09375 56 \r\nQ 79.640625 56 84.28125 50.015625 \r\nQ 88.921875 44.046875 88.921875 33.015625 \r\nL 88.921875 0 \r\nL 79.890625 0 \r\nL 79.890625 32.71875 \r\nQ 79.890625 40.578125 77.09375 44.375 \r\nQ 74.3125 48.1875 68.609375 48.1875 \r\nQ 61.625 48.1875 57.5625 43.546875 \r\nQ 53.515625 38.921875 53.515625 30.90625 \r\nL 53.515625 0 \r\nL 44.484375 0 \r\nL 44.484375 32.71875 \r\nQ 44.484375 40.625 41.703125 44.40625 \r\nQ 38.921875 48.1875 33.109375 48.1875 \r\nQ 26.21875 48.1875 22.15625 43.53125 \r\nQ 18.109375 38.875 18.109375 30.90625 \r\nL 18.109375 0 \r\nL 9.078125 0 \r\nL 9.078125 54.6875 \r\nL 18.109375 54.6875 \r\nL 18.109375 46.1875 \r\nQ 21.1875 51.21875 25.484375 53.609375 \r\nQ 29.78125 56 35.6875 56 \r\nQ 41.65625 56 45.828125 52.96875 \r\nQ 50 49.953125 52 44.1875 \r\nz\r\n\" id=\"DejaVuSans-109\"/>\r\n       <path d=\"M 9.421875 75.984375 \r\nL 18.40625 75.984375 \r\nL 18.40625 0 \r\nL 9.421875 0 \r\nz\r\n\" id=\"DejaVuSans-108\"/>\r\n       <path d=\"M 34.28125 27.484375 \r\nQ 23.390625 27.484375 19.1875 25 \r\nQ 14.984375 22.515625 14.984375 16.5 \r\nQ 14.984375 11.71875 18.140625 8.90625 \r\nQ 21.296875 6.109375 26.703125 6.109375 \r\nQ 34.1875 6.109375 38.703125 11.40625 \r\nQ 43.21875 16.703125 43.21875 25.484375 \r\nL 43.21875 27.484375 \r\nz\r\nM 52.203125 31.203125 \r\nL 52.203125 0 \r\nL 43.21875 0 \r\nL 43.21875 8.296875 \r\nQ 40.140625 3.328125 35.546875 0.953125 \r\nQ 30.953125 -1.421875 24.3125 -1.421875 \r\nQ 15.921875 -1.421875 10.953125 3.296875 \r\nQ 6 8.015625 6 15.921875 \r\nQ 6 25.140625 12.171875 29.828125 \r\nQ 18.359375 34.515625 30.609375 34.515625 \r\nL 43.21875 34.515625 \r\nL 43.21875 35.40625 \r\nQ 43.21875 41.609375 39.140625 45 \r\nQ 35.0625 48.390625 27.6875 48.390625 \r\nQ 23 48.390625 18.546875 47.265625 \r\nQ 14.109375 46.140625 10.015625 43.890625 \r\nL 10.015625 52.203125 \r\nQ 14.9375 54.109375 19.578125 55.046875 \r\nQ 24.21875 56 28.609375 56 \r\nQ 40.484375 56 46.34375 49.84375 \r\nQ 52.203125 43.703125 52.203125 31.203125 \r\nz\r\n\" id=\"DejaVuSans-97\"/>\r\n       <path d=\"M 18.3125 70.21875 \r\nL 18.3125 54.6875 \r\nL 36.8125 54.6875 \r\nL 36.8125 47.703125 \r\nL 18.3125 47.703125 \r\nL 18.3125 18.015625 \r\nQ 18.3125 11.328125 20.140625 9.421875 \r\nQ 21.96875 7.515625 27.59375 7.515625 \r\nL 36.8125 7.515625 \r\nL 36.8125 0 \r\nL 27.59375 0 \r\nQ 17.1875 0 13.234375 3.875 \r\nQ 9.28125 7.765625 9.28125 18.015625 \r\nL 9.28125 47.703125 \r\nL 2.6875 47.703125 \r\nL 2.6875 54.6875 \r\nL 9.28125 54.6875 \r\nL 9.28125 70.21875 \r\nz\r\n\" id=\"DejaVuSans-116\"/>\r\n       <path d=\"M 9.421875 54.6875 \r\nL 18.40625 54.6875 \r\nL 18.40625 0 \r\nL 9.421875 0 \r\nz\r\nM 9.421875 75.984375 \r\nL 18.40625 75.984375 \r\nL 18.40625 64.59375 \r\nL 9.421875 64.59375 \r\nz\r\n\" id=\"DejaVuSans-105\"/>\r\n       <path d=\"M 2.984375 54.6875 \r\nL 12.5 54.6875 \r\nL 29.59375 8.796875 \r\nL 46.6875 54.6875 \r\nL 56.203125 54.6875 \r\nL 35.6875 0 \r\nL 23.484375 0 \r\nz\r\n\" id=\"DejaVuSans-118\"/>\r\n       <path d=\"M 56.203125 29.59375 \r\nL 56.203125 25.203125 \r\nL 14.890625 25.203125 \r\nQ 15.484375 15.921875 20.484375 11.0625 \r\nQ 25.484375 6.203125 34.421875 6.203125 \r\nQ 39.59375 6.203125 44.453125 7.46875 \r\nQ 49.3125 8.734375 54.109375 11.28125 \r\nL 54.109375 2.78125 \r\nQ 49.265625 0.734375 44.1875 -0.34375 \r\nQ 39.109375 -1.421875 33.890625 -1.421875 \r\nQ 20.796875 -1.421875 13.15625 6.1875 \r\nQ 5.515625 13.8125 5.515625 26.8125 \r\nQ 5.515625 40.234375 12.765625 48.109375 \r\nQ 20.015625 56 32.328125 56 \r\nQ 43.359375 56 49.78125 48.890625 \r\nQ 56.203125 41.796875 56.203125 29.59375 \r\nz\r\nM 47.21875 32.234375 \r\nQ 47.125 39.59375 43.09375 43.984375 \r\nQ 39.0625 48.390625 32.421875 48.390625 \r\nQ 24.90625 48.390625 20.390625 44.140625 \r\nQ 15.875 39.890625 15.1875 32.171875 \r\nz\r\n\" id=\"DejaVuSans-101\"/>\r\n       <path d=\"M 54.890625 54.6875 \r\nL 35.109375 28.078125 \r\nL 55.90625 0 \r\nL 45.3125 0 \r\nL 29.390625 21.484375 \r\nL 13.484375 0 \r\nL 2.875 0 \r\nL 24.125 28.609375 \r\nL 4.6875 54.6875 \r\nL 15.28125 54.6875 \r\nL 29.78125 35.203125 \r\nL 44.28125 54.6875 \r\nz\r\n\" id=\"DejaVuSans-120\"/>\r\n       <path d=\"M 18.109375 8.203125 \r\nL 18.109375 -20.796875 \r\nL 9.078125 -20.796875 \r\nL 9.078125 54.6875 \r\nL 18.109375 54.6875 \r\nL 18.109375 46.390625 \r\nQ 20.953125 51.265625 25.265625 53.625 \r\nQ 29.59375 56 35.59375 56 \r\nQ 45.5625 56 51.78125 48.09375 \r\nQ 58.015625 40.1875 58.015625 27.296875 \r\nQ 58.015625 14.40625 51.78125 6.484375 \r\nQ 45.5625 -1.421875 35.59375 -1.421875 \r\nQ 29.59375 -1.421875 25.265625 0.953125 \r\nQ 20.953125 3.328125 18.109375 8.203125 \r\nz\r\nM 48.6875 27.296875 \r\nQ 48.6875 37.203125 44.609375 42.84375 \r\nQ 40.53125 48.484375 33.40625 48.484375 \r\nQ 26.265625 48.484375 22.1875 42.84375 \r\nQ 18.109375 37.203125 18.109375 27.296875 \r\nQ 18.109375 17.390625 22.1875 11.75 \r\nQ 26.265625 6.109375 33.40625 6.109375 \r\nQ 40.53125 6.109375 44.609375 11.75 \r\nQ 48.6875 17.390625 48.6875 27.296875 \r\nz\r\n\" id=\"DejaVuSans-112\"/>\r\n       <path d=\"M 54.890625 33.015625 \r\nL 54.890625 0 \r\nL 45.90625 0 \r\nL 45.90625 32.71875 \r\nQ 45.90625 40.484375 42.875 44.328125 \r\nQ 39.84375 48.1875 33.796875 48.1875 \r\nQ 26.515625 48.1875 22.3125 43.546875 \r\nQ 18.109375 38.921875 18.109375 30.90625 \r\nL 18.109375 0 \r\nL 9.078125 0 \r\nL 9.078125 54.6875 \r\nL 18.109375 54.6875 \r\nL 18.109375 46.1875 \r\nQ 21.34375 51.125 25.703125 53.5625 \r\nQ 30.078125 56 35.796875 56 \r\nQ 45.21875 56 50.046875 50.171875 \r\nQ 54.890625 44.34375 54.890625 33.015625 \r\nz\r\n\" id=\"DejaVuSans-110\"/>\r\n       <path d=\"M 45.40625 46.390625 \r\nL 45.40625 75.984375 \r\nL 54.390625 75.984375 \r\nL 54.390625 0 \r\nL 45.40625 0 \r\nL 45.40625 8.203125 \r\nQ 42.578125 3.328125 38.25 0.953125 \r\nQ 33.9375 -1.421875 27.875 -1.421875 \r\nQ 17.96875 -1.421875 11.734375 6.484375 \r\nQ 5.515625 14.40625 5.515625 27.296875 \r\nQ 5.515625 40.1875 11.734375 48.09375 \r\nQ 17.96875 56 27.875 56 \r\nQ 33.9375 56 38.25 53.625 \r\nQ 42.578125 51.265625 45.40625 46.390625 \r\nz\r\nM 14.796875 27.296875 \r\nQ 14.796875 17.390625 18.875 11.75 \r\nQ 22.953125 6.109375 30.078125 6.109375 \r\nQ 37.203125 6.109375 41.296875 11.75 \r\nQ 45.40625 17.390625 45.40625 27.296875 \r\nQ 45.40625 37.203125 41.296875 42.84375 \r\nQ 37.203125 48.484375 30.078125 48.484375 \r\nQ 22.953125 48.484375 18.875 42.84375 \r\nQ 14.796875 37.203125 14.796875 27.296875 \r\nz\r\n\" id=\"DejaVuSans-100\"/>\r\n       <path d=\"M 48.78125 52.59375 \r\nL 48.78125 44.1875 \r\nQ 44.96875 46.296875 41.140625 47.34375 \r\nQ 37.3125 48.390625 33.40625 48.390625 \r\nQ 24.65625 48.390625 19.8125 42.84375 \r\nQ 14.984375 37.3125 14.984375 27.296875 \r\nQ 14.984375 17.28125 19.8125 11.734375 \r\nQ 24.65625 6.203125 33.40625 6.203125 \r\nQ 37.3125 6.203125 41.140625 7.25 \r\nQ 44.96875 8.296875 48.78125 10.40625 \r\nL 48.78125 2.09375 \r\nQ 45.015625 0.34375 40.984375 -0.53125 \r\nQ 36.96875 -1.421875 32.421875 -1.421875 \r\nQ 20.0625 -1.421875 12.78125 6.34375 \r\nQ 5.515625 14.109375 5.515625 27.296875 \r\nQ 5.515625 40.671875 12.859375 48.328125 \r\nQ 20.21875 56 33.015625 56 \r\nQ 37.15625 56 41.109375 55.140625 \r\nQ 45.0625 54.296875 48.78125 52.59375 \r\nz\r\n\" id=\"DejaVuSans-99\"/>\r\n      </defs>\r\n      <use xlink:href=\"#DejaVuSans-67\"/>\r\n      <use x=\"69.824219\" xlink:href=\"#DejaVuSans-117\"/>\r\n      <use x=\"133.203125\" xlink:href=\"#DejaVuSans-109\"/>\r\n      <use x=\"230.615234\" xlink:href=\"#DejaVuSans-117\"/>\r\n      <use x=\"293.994141\" xlink:href=\"#DejaVuSans-108\"/>\r\n      <use x=\"321.777344\" xlink:href=\"#DejaVuSans-97\"/>\r\n      <use x=\"383.056641\" xlink:href=\"#DejaVuSans-116\"/>\r\n      <use x=\"422.265625\" xlink:href=\"#DejaVuSans-105\"/>\r\n      <use x=\"450.048828\" xlink:href=\"#DejaVuSans-118\"/>\r\n      <use x=\"509.228516\" xlink:href=\"#DejaVuSans-101\"/>\r\n      <use x=\"570.751953\" xlink:href=\"#DejaVuSans-32\"/>\r\n      <use x=\"602.539062\" xlink:href=\"#DejaVuSans-101\"/>\r\n      <use x=\"662.3125\" xlink:href=\"#DejaVuSans-120\"/>\r\n      <use x=\"721.492188\" xlink:href=\"#DejaVuSans-112\"/>\r\n      <use x=\"784.96875\" xlink:href=\"#DejaVuSans-108\"/>\r\n      <use x=\"812.751953\" xlink:href=\"#DejaVuSans-97\"/>\r\n      <use x=\"874.03125\" xlink:href=\"#DejaVuSans-105\"/>\r\n      <use x=\"901.814453\" xlink:href=\"#DejaVuSans-110\"/>\r\n      <use x=\"965.193359\" xlink:href=\"#DejaVuSans-101\"/>\r\n      <use x=\"1026.716797\" xlink:href=\"#DejaVuSans-100\"/>\r\n      <use x=\"1090.193359\" xlink:href=\"#DejaVuSans-32\"/>\r\n      <use x=\"1121.980469\" xlink:href=\"#DejaVuSans-118\"/>\r\n      <use x=\"1181.160156\" xlink:href=\"#DejaVuSans-97\"/>\r\n      <use x=\"1242.439453\" xlink:href=\"#DejaVuSans-114\"/>\r\n      <use x=\"1283.552734\" xlink:href=\"#DejaVuSans-105\"/>\r\n      <use x=\"1311.335938\" xlink:href=\"#DejaVuSans-97\"/>\r\n      <use x=\"1372.615234\" xlink:href=\"#DejaVuSans-110\"/>\r\n      <use x=\"1435.994141\" xlink:href=\"#DejaVuSans-99\"/>\r\n      <use x=\"1490.974609\" xlink:href=\"#DejaVuSans-101\"/>\r\n     </g>\r\n    </g>\r\n   </g>\r\n   <g id=\"line2d_15\">\r\n    <path clip-path=\"url(#p7aa4d2edb9)\" d=\"M 71.724432 214.756364 \r\nL 82.21973 44.422314 \r\nL 92.715027 35.702776 \r\nL 103.210325 28.347698 \r\nL 113.705623 21.808438 \r\nL 124.200921 17.616285 \r\nL 134.696219 17.144247 \r\nL 145.191516 17.124036 \r\nL 155.686814 17.114099 \r\nL 166.182112 17.10819 \r\nL 176.67741 17.10302 \r\nL 187.172708 17.100137 \r\nL 197.668005 17.097727 \r\nL 208.163303 17.095666 \r\nL 218.658601 17.093683 \r\nL 229.153899 17.091852 \r\nL 239.649197 17.090443 \r\nL 250.144495 17.089043 \r\nL 260.639792 17.08801 \r\nL 271.13509 17.087155 \r\nL 281.630388 17.086346 \r\nL 292.125686 17.08561 \r\nL 302.620984 17.085094 \r\nL 313.116281 17.084677 \r\nL 323.611579 17.084322 \r\nL 334.106877 17.084083 \r\nL 344.602175 17.083946 \r\nL 355.097473 17.08382 \r\nL 365.59277 17.083714 \r\nL 376.088068 17.083636 \r\n\" style=\"fill:none;stroke:#1f77b4;stroke-linecap:square;stroke-width:1.5;\"/>\r\n   </g>\r\n   <g id=\"patch_3\">\r\n    <path d=\"M 56.50625 224.64 \r\nL 56.50625 7.2 \r\n\" style=\"fill:none;stroke:#000000;stroke-linecap:square;stroke-linejoin:miter;stroke-width:0.8;\"/>\r\n   </g>\r\n   <g id=\"patch_4\">\r\n    <path d=\"M 391.30625 224.64 \r\nL 391.30625 7.2 \r\n\" style=\"fill:none;stroke:#000000;stroke-linecap:square;stroke-linejoin:miter;stroke-width:0.8;\"/>\r\n   </g>\r\n   <g id=\"patch_5\">\r\n    <path d=\"M 56.50625 224.64 \r\nL 391.30625 224.64 \r\n\" style=\"fill:none;stroke:#000000;stroke-linecap:square;stroke-linejoin:miter;stroke-width:0.8;\"/>\r\n   </g>\r\n   <g id=\"patch_6\">\r\n    <path d=\"M 56.50625 7.2 \r\nL 391.30625 7.2 \r\n\" style=\"fill:none;stroke:#000000;stroke-linecap:square;stroke-linejoin:miter;stroke-width:0.8;\"/>\r\n   </g>\r\n  </g>\r\n </g>\r\n <defs>\r\n  <clipPath id=\"p7aa4d2edb9\">\r\n   <rect height=\"217.44\" width=\"334.8\" x=\"56.50625\" y=\"7.2\"/>\r\n  </clipPath>\r\n </defs>\r\n</svg>\r\n",
                        "image/png": "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\n"
                    },
                    "metadata": {
                        "needs_background": "light"
                    }
                }
            ],
            "source": [
                "plt.plot(np.cumsum(pca.explained_variance_ratio_))\n",
                "plt.xlabel('Nr. PC')\n",
                "plt.ylabel('Cumulative explained variance');"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": 64,
            "metadata": {},
            "outputs": [],
            "source": [
                "pca_df = pd.DataFrame(pca_samples[:, 0:20])"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": 65,
            "metadata": {},
            "outputs": [
                {
                    "output_type": "execute_result",
                    "data": {
                        "text/plain": [
                            "                 0          1          2         3          4          5   \\\n",
                            "0       -631.879677   9.523199  21.837547  5.212761 -35.847065  -6.902606   \n",
                            "1       -601.372436 -19.953069  22.922992  5.338221 -35.122278  -7.322895   \n",
                            "2       -570.351502 -20.427363  23.311628  5.346803 -34.646132  -7.677534   \n",
                            "3        684.845244  31.053664   3.110651  0.226361 -42.052455 -10.507853   \n",
                            "4        715.352486   1.577395   4.196097  0.351820 -41.327669 -10.928142   \n",
                            "...             ...        ...        ...       ...        ...        ...   \n",
                            "6254513   21.141777  -9.575407  34.777646  2.695565  36.789093 -20.302645   \n",
                            "6254514   52.145948  -9.050202  35.142514  2.700199  37.256869 -20.655168   \n",
                            "6254515   80.101881  -5.561851  36.138450  2.839678  37.598512 -20.304637   \n",
                            "6254516  111.156340  -8.035142  36.574623  2.856154  38.091399 -20.663507   \n",
                            "6254517  141.143943  -6.526957  36.923248  2.859255  38.556709 -21.018223   \n",
                            "\n",
                            "               6         7         8         9         10        11        12  \\\n",
                            "0       -1.480611 -0.577582  0.207337  0.336142 -0.036688 -0.053773  0.359493   \n",
                            "1       -0.731045 -0.578528  0.209608  0.332390 -0.034153 -0.054948  0.360766   \n",
                            "2        0.255310 -0.576605  0.217183  0.328611 -0.028147 -0.052861  0.368501   \n",
                            "3        3.757013  0.654786 -0.540264  0.114627  0.368271  0.038455  0.479739   \n",
                            "4        4.506580  0.653840 -0.537993  0.110874  0.370806  0.037280  0.481013   \n",
                            "...           ...       ...       ...       ...       ...       ...       ...   \n",
                            "6254513 -5.281492  0.933052  0.732636 -0.416351  0.284201  0.026163  0.324067   \n",
                            "6254514 -4.286979  0.935075  0.740402 -0.420126  0.290317  0.028359  0.332014   \n",
                            "6254515 -3.264657  0.944130  0.769230 -0.422100  0.291306  0.032738  0.344135   \n",
                            "6254516 -2.294617  0.945850  0.776422 -0.425885  0.297093  0.034607  0.351444   \n",
                            "6254517 -1.292168  0.948044  0.784619 -0.429538  0.303021  0.036816  0.359307   \n",
                            "\n",
                            "               13        14        15        16        17        18        19  \n",
                            "0       -0.016235 -0.342887  0.573760 -0.155327 -0.492288 -0.135032  0.174302  \n",
                            "1       -0.016844 -0.346471  0.576328 -0.155267 -0.489648 -0.134625  0.173080  \n",
                            "2       -0.015748 -0.345764  0.577934 -0.153327 -0.487261 -0.136426  0.172022  \n",
                            "3        0.003430 -0.308732  0.202518 -0.086179  0.825972 -0.522792  0.134562  \n",
                            "4        0.002822 -0.312316  0.205086 -0.086119  0.828611 -0.522384  0.133340  \n",
                            "...           ...       ...       ...       ...       ...       ...       ...  \n",
                            "6254513  0.116012  0.059629  0.254434  0.208963  0.353994  0.387869  0.482290  \n",
                            "6254514  0.117168  0.060485  0.256004  0.210969  0.356370  0.385992  0.481239  \n",
                            "6254515  0.118387  0.064214  0.251736  0.214386  0.356446  0.382334  0.478122  \n",
                            "6254516  0.119363  0.064621  0.253413  0.216194  0.358855  0.380684  0.477049  \n",
                            "6254517  0.120617  0.065676  0.254883  0.218297  0.361162  0.378753  0.476057  \n",
                            "\n",
                            "[6254518 rows x 20 columns]"
                        ],
                        "text/html": "<div>\n<style scoped>\n    .dataframe tbody tr th:only-of-type {\n        vertical-align: middle;\n    }\n\n    .dataframe tbody tr th {\n        vertical-align: top;\n    }\n\n    .dataframe thead th {\n        text-align: right;\n    }\n</style>\n<table border=\"1\" class=\"dataframe\">\n  <thead>\n    <tr style=\"text-align: right;\">\n      <th></th>\n      <th>0</th>\n      <th>1</th>\n      <th>2</th>\n      <th>3</th>\n      <th>4</th>\n      <th>5</th>\n      <th>6</th>\n      <th>7</th>\n      <th>8</th>\n      <th>9</th>\n      <th>10</th>\n      <th>11</th>\n      <th>12</th>\n      <th>13</th>\n      <th>14</th>\n      <th>15</th>\n      <th>16</th>\n      <th>17</th>\n      <th>18</th>\n      <th>19</th>\n    </tr>\n  </thead>\n  <tbody>\n    <tr>\n      <th>0</th>\n      <td>-631.879677</td>\n      <td>9.523199</td>\n      <td>21.837547</td>\n      <td>5.212761</td>\n      <td>-35.847065</td>\n      <td>-6.902606</td>\n      <td>-1.480611</td>\n      <td>-0.577582</td>\n      <td>0.207337</td>\n      <td>0.336142</td>\n      <td>-0.036688</td>\n      <td>-0.053773</td>\n      <td>0.359493</td>\n      <td>-0.016235</td>\n      <td>-0.342887</td>\n      <td>0.573760</td>\n      <td>-0.155327</td>\n      <td>-0.492288</td>\n      <td>-0.135032</td>\n      <td>0.174302</td>\n    </tr>\n    <tr>\n      <th>1</th>\n      <td>-601.372436</td>\n      <td>-19.953069</td>\n      <td>22.922992</td>\n      <td>5.338221</td>\n      <td>-35.122278</td>\n      <td>-7.322895</td>\n      <td>-0.731045</td>\n      <td>-0.578528</td>\n      <td>0.209608</td>\n      <td>0.332390</td>\n      <td>-0.034153</td>\n      <td>-0.054948</td>\n      <td>0.360766</td>\n      <td>-0.016844</td>\n      <td>-0.346471</td>\n      <td>0.576328</td>\n      <td>-0.155267</td>\n      <td>-0.489648</td>\n      <td>-0.134625</td>\n      <td>0.173080</td>\n    </tr>\n    <tr>\n      <th>2</th>\n      <td>-570.351502</td>\n      <td>-20.427363</td>\n      <td>23.311628</td>\n      <td>5.346803</td>\n      <td>-34.646132</td>\n      <td>-7.677534</td>\n      <td>0.255310</td>\n      <td>-0.576605</td>\n      <td>0.217183</td>\n      <td>0.328611</td>\n      <td>-0.028147</td>\n      <td>-0.052861</td>\n      <td>0.368501</td>\n      <td>-0.015748</td>\n      <td>-0.345764</td>\n      <td>0.577934</td>\n      <td>-0.153327</td>\n      <td>-0.487261</td>\n      <td>-0.136426</td>\n      <td>0.172022</td>\n    </tr>\n    <tr>\n      <th>3</th>\n      <td>684.845244</td>\n      <td>31.053664</td>\n      <td>3.110651</td>\n      <td>0.226361</td>\n      <td>-42.052455</td>\n      <td>-10.507853</td>\n      <td>3.757013</td>\n      <td>0.654786</td>\n      <td>-0.540264</td>\n      <td>0.114627</td>\n      <td>0.368271</td>\n      <td>0.038455</td>\n      <td>0.479739</td>\n      <td>0.003430</td>\n      <td>-0.308732</td>\n      <td>0.202518</td>\n      <td>-0.086179</td>\n      <td>0.825972</td>\n      <td>-0.522792</td>\n      <td>0.134562</td>\n    </tr>\n    <tr>\n      <th>4</th>\n      <td>715.352486</td>\n      <td>1.577395</td>\n      <td>4.196097</td>\n      <td>0.351820</td>\n      <td>-41.327669</td>\n      <td>-10.928142</td>\n      <td>4.506580</td>\n      <td>0.653840</td>\n      <td>-0.537993</td>\n      <td>0.110874</td>\n      <td>0.370806</td>\n      <td>0.037280</td>\n      <td>0.481013</td>\n      <td>0.002822</td>\n      <td>-0.312316</td>\n      <td>0.205086</td>\n      <td>-0.086119</td>\n      <td>0.828611</td>\n      <td>-0.522384</td>\n      <td>0.133340</td>\n    </tr>\n    <tr>\n      <th>...</th>\n      <td>...</td>\n      <td>...</td>\n      <td>...</td>\n      <td>...</td>\n      <td>...</td>\n      <td>...</td>\n      <td>...</td>\n      <td>...</td>\n      <td>...</td>\n      <td>...</td>\n      <td>...</td>\n      <td>...</td>\n      <td>...</td>\n      <td>...</td>\n      <td>...</td>\n      <td>...</td>\n      <td>...</td>\n      <td>...</td>\n      <td>...</td>\n      <td>...</td>\n    </tr>\n    <tr>\n      <th>6254513</th>\n      <td>21.141777</td>\n      <td>-9.575407</td>\n      <td>34.777646</td>\n      <td>2.695565</td>\n      <td>36.789093</td>\n      <td>-20.302645</td>\n      <td>-5.281492</td>\n      <td>0.933052</td>\n      <td>0.732636</td>\n      <td>-0.416351</td>\n      <td>0.284201</td>\n      <td>0.026163</td>\n      <td>0.324067</td>\n      <td>0.116012</td>\n      <td>0.059629</td>\n      <td>0.254434</td>\n      <td>0.208963</td>\n      <td>0.353994</td>\n      <td>0.387869</td>\n      <td>0.482290</td>\n    </tr>\n    <tr>\n      <th>6254514</th>\n      <td>52.145948</td>\n      <td>-9.050202</td>\n      <td>35.142514</td>\n      <td>2.700199</td>\n      <td>37.256869</td>\n      <td>-20.655168</td>\n      <td>-4.286979</td>\n      <td>0.935075</td>\n      <td>0.740402</td>\n      <td>-0.420126</td>\n      <td>0.290317</td>\n      <td>0.028359</td>\n      <td>0.332014</td>\n      <td>0.117168</td>\n      <td>0.060485</td>\n      <td>0.256004</td>\n      <td>0.210969</td>\n      <td>0.356370</td>\n      <td>0.385992</td>\n      <td>0.481239</td>\n    </tr>\n    <tr>\n      <th>6254515</th>\n      <td>80.101881</td>\n      <td>-5.561851</td>\n      <td>36.138450</td>\n      <td>2.839678</td>\n      <td>37.598512</td>\n      <td>-20.304637</td>\n      <td>-3.264657</td>\n      <td>0.944130</td>\n      <td>0.769230</td>\n      <td>-0.422100</td>\n      <td>0.291306</td>\n      <td>0.032738</td>\n      <td>0.344135</td>\n      <td>0.118387</td>\n      <td>0.064214</td>\n      <td>0.251736</td>\n      <td>0.214386</td>\n      <td>0.356446</td>\n      <td>0.382334</td>\n      <td>0.478122</td>\n    </tr>\n    <tr>\n      <th>6254516</th>\n      <td>111.156340</td>\n      <td>-8.035142</td>\n      <td>36.574623</td>\n      <td>2.856154</td>\n      <td>38.091399</td>\n      <td>-20.663507</td>\n      <td>-2.294617</td>\n      <td>0.945850</td>\n      <td>0.776422</td>\n      <td>-0.425885</td>\n      <td>0.297093</td>\n      <td>0.034607</td>\n      <td>0.351444</td>\n      <td>0.119363</td>\n      <td>0.064621</td>\n      <td>0.253413</td>\n      <td>0.216194</td>\n      <td>0.358855</td>\n      <td>0.380684</td>\n      <td>0.477049</td>\n    </tr>\n    <tr>\n      <th>6254517</th>\n      <td>141.143943</td>\n      <td>-6.526957</td>\n      <td>36.923248</td>\n      <td>2.859255</td>\n      <td>38.556709</td>\n      <td>-21.018223</td>\n      <td>-1.292168</td>\n      <td>0.948044</td>\n      <td>0.784619</td>\n      <td>-0.429538</td>\n      <td>0.303021</td>\n      <td>0.036816</td>\n      <td>0.359307</td>\n      <td>0.120617</td>\n      <td>0.065676</td>\n      <td>0.254883</td>\n      <td>0.218297</td>\n      <td>0.361162</td>\n      <td>0.378753</td>\n      <td>0.476057</td>\n    </tr>\n  </tbody>\n</table>\n<p>6254518 rows × 20 columns</p>\n</div>"
                    },
                    "metadata": {},
                    "execution_count": 65
                }
            ],
            "source": [
                "pca_df\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": 66,
            "metadata": {},
            "outputs": [
                {
                    "output_type": "stream",
                    "name": "stdout",
                    "text": [
                        "Fitting pipe with 2 clusters\n",
                        "Fitting pipe with 3 clusters\n",
                        "Fitting pipe with 4 clusters\n",
                        "Fitting pipe with 5 clusters\n",
                        "Fitting pipe with 6 clusters\n"
                    ]
                },
                {
                    "output_type": "error",
                    "ename": "KeyboardInterrupt",
                    "evalue": "",
                    "traceback": [
                        "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
                        "\u001b[1;31mKeyboardInterrupt\u001b[0m                         Traceback (most recent call last)",
                        "\u001b[1;32m<ipython-input-66-9d1a9632ddbd>\u001b[0m in \u001b[0;36m<module>\u001b[1;34m\u001b[0m\n\u001b[0;32m     10\u001b[0m         \u001b[0mprint\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;34mf\"Fitting pipe with {k} clusters\"\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m     11\u001b[0m         \u001b[0mcluster_model\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mKMeans\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mn_clusters\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mk\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m---> 12\u001b[1;33m         \u001b[0mcluster_model\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mfit\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mpca_df\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m     13\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m     14\u001b[0m         \u001b[0msse\u001b[0m\u001b[1;33m[\u001b[0m\u001b[0mk\u001b[0m\u001b[1;33m]\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mcluster_model\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0minertia_\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
                        "\u001b[1;32m~\\anaconda3\\lib\\site-packages\\sklearn\\cluster\\_kmeans.py\u001b[0m in \u001b[0;36mfit\u001b[1;34m(self, X, y, sample_weight)\u001b[0m\n\u001b[0;32m   1066\u001b[0m         \u001b[1;32mfor\u001b[0m \u001b[0mseed\u001b[0m \u001b[1;32min\u001b[0m \u001b[0mseeds\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m   1067\u001b[0m             \u001b[1;31m# run a k-means once\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m-> 1068\u001b[1;33m             labels, inertia, centers, n_iter_ = kmeans_single(\n\u001b[0m\u001b[0;32m   1069\u001b[0m                 \u001b[0mX\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0msample_weight\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mn_clusters\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mmax_iter\u001b[0m\u001b[1;33m=\u001b[0m\u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mmax_iter\u001b[0m\u001b[1;33m,\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m   1070\u001b[0m                 \u001b[0minit\u001b[0m\u001b[1;33m=\u001b[0m\u001b[0minit\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mverbose\u001b[0m\u001b[1;33m=\u001b[0m\u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mverbose\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mtol\u001b[0m\u001b[1;33m=\u001b[0m\u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0m_tol\u001b[0m\u001b[1;33m,\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
                        "\u001b[1;32m~\\anaconda3\\lib\\site-packages\\sklearn\\cluster\\_kmeans.py\u001b[0m in \u001b[0;36m_kmeans_single_elkan\u001b[1;34m(X, sample_weight, n_clusters, max_iter, init, verbose, x_squared_norms, random_state, tol, n_threads)\u001b[0m\n\u001b[0;32m    426\u001b[0m         \u001b[1;31m# compute new pairwise distances between centers and closest other\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    427\u001b[0m         \u001b[1;31m# center of each center for next iterations\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m--> 428\u001b[1;33m         \u001b[0mcenter_half_distances\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0meuclidean_distances\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mcenters_new\u001b[0m\u001b[1;33m)\u001b[0m \u001b[1;33m/\u001b[0m \u001b[1;36m2\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m    429\u001b[0m         distance_next_center = np.partition(\n\u001b[0;32m    430\u001b[0m             np.asarray(center_half_distances), kth=1, axis=0)[1]\n",
                        "\u001b[1;31mKeyboardInterrupt\u001b[0m: "
                    ]
                }
            ],
            "source": [
                "CALCULATE_ELBOW = True\n",
                "\n",
                "if CALCULATE_ELBOW:\n",
                "    st = time.time()\n",
                "\n",
                "    sse = {}\n",
                "\n",
                "    for k in range(2, 15):\n",
                "\n",
                "        print(f\"Fitting pipe with {k} clusters\")\n",
                "        cluster_model = KMeans(n_clusters = k)\n",
                "        cluster_model.fit(pca_df)\n",
                "\n",
                "        sse[k] = cluster_model.inertia_\n",
                "\n",
                "    et = time.time()\n",
                "    print(\"Elbow curve took {} minutes.\".format(round((et - st)/60), 2))"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": 101,
            "metadata": {},
            "outputs": [
                {
                    "output_type": "display_data",
                    "data": {
                        "text/plain": "<Figure size 1152x576 with 1 Axes>",
                        "image/svg+xml": "<?xml version=\"1.0\" encoding=\"utf-8\" standalone=\"no\"?>\r\n<!DOCTYPE svg PUBLIC \"-//W3C//DTD SVG 1.1//EN\"\r\n  \"http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd\">\r\n<!-- Created with matplotlib (https://matplotlib.org/) -->\r\n<svg height=\"523.558125pt\" version=\"1.1\" viewBox=\"0 0 930.103125 523.558125\" width=\"930.103125pt\" xmlns=\"http://www.w3.org/2000/svg\" xmlns:xlink=\"http://www.w3.org/1999/xlink\">\r\n <metadata>\r\n  <rdf:RDF xmlns:cc=\"http://creativecommons.org/ns#\" xmlns:dc=\"http://purl.org/dc/elements/1.1/\" xmlns:rdf=\"http://www.w3.org/1999/02/22-rdf-syntax-ns#\">\r\n   <cc:Work>\r\n    <dc:type rdf:resource=\"http://purl.org/dc/dcmitype/StillImage\"/>\r\n    <dc:date>2021-06-23T23:24:37.942947</dc:date>\r\n    <dc:format>image/svg+xml</dc:format>\r\n    <dc:creator>\r\n     <cc:Agent>\r\n      <dc:title>Matplotlib v3.3.2, https://matplotlib.org/</dc:title>\r\n     </cc:Agent>\r\n    </dc:creator>\r\n   </cc:Work>\r\n  </rdf:RDF>\r\n </metadata>\r\n <defs>\r\n  <style type=\"text/css\">*{stroke-linecap:butt;stroke-linejoin:round;}</style>\r\n </defs>\r\n <g id=\"figure_1\">\r\n  <g id=\"patch_1\">\r\n   <path d=\"M 0 523.558125 \r\nL 930.103125 523.558125 \r\nL 930.103125 0 \r\nL 0 0 \r\nz\r\n\" style=\"fill:none;\"/>\r\n  </g>\r\n  <g id=\"axes_1\">\r\n   <g id=\"patch_2\">\r\n    <path d=\"M 30.103125 499.68 \r\nL 922.903125 499.68 \r\nL 922.903125 64.8 \r\nL 30.103125 64.8 \r\nz\r\n\" style=\"fill:#e5e5e5;\"/>\r\n   </g>\r\n   <g id=\"matplotlib.axis_1\">\r\n    <g id=\"xtick_1\">\r\n     <g id=\"line2d_1\">\r\n      <path clip-path=\"url(#p070f028ef1)\" d=\"M 70.684943 499.68 \r\nL 70.684943 64.8 \r\n\" style=\"fill:none;stroke:#ffffff;stroke-linecap:square;stroke-width:0.8;\"/>\r\n     </g>\r\n     <g id=\"line2d_2\">\r\n      <defs>\r\n       <path d=\"M 0 0 \r\nL 0 3.5 \r\n\" id=\"m41ac6bf7bb\" style=\"stroke:#555555;stroke-width:0.8;\"/>\r\n      </defs>\r\n      <g>\r\n       <use style=\"fill:#555555;stroke:#555555;stroke-width:0.8;\" x=\"70.684943\" xlink:href=\"#m41ac6bf7bb\" y=\"499.68\"/>\r\n      </g>\r\n     </g>\r\n     <g id=\"text_1\">\r\n      <!-- 2 -->\r\n      <g style=\"fill:#555555;\" transform=\"translate(67.503693 514.278437)scale(0.1 -0.1)\">\r\n       <defs>\r\n        <path d=\"M 19.1875 8.296875 \r\nL 53.609375 8.296875 \r\nL 53.609375 0 \r\nL 7.328125 0 \r\nL 7.328125 8.296875 \r\nQ 12.9375 14.109375 22.625 23.890625 \r\nQ 32.328125 33.6875 34.8125 36.53125 \r\nQ 39.546875 41.84375 41.421875 45.53125 \r\nQ 43.3125 49.21875 43.3125 52.78125 \r\nQ 43.3125 58.59375 39.234375 62.25 \r\nQ 35.15625 65.921875 28.609375 65.921875 \r\nQ 23.96875 65.921875 18.8125 64.3125 \r\nQ 13.671875 62.703125 7.8125 59.421875 \r\nL 7.8125 69.390625 \r\nQ 13.765625 71.78125 18.9375 73 \r\nQ 24.125 74.21875 28.421875 74.21875 \r\nQ 39.75 74.21875 46.484375 68.546875 \r\nQ 53.21875 62.890625 53.21875 53.421875 \r\nQ 53.21875 48.921875 51.53125 44.890625 \r\nQ 49.859375 40.875 45.40625 35.40625 \r\nQ 44.1875 33.984375 37.640625 27.21875 \r\nQ 31.109375 20.453125 19.1875 8.296875 \r\nz\r\n\" id=\"DejaVuSans-50\"/>\r\n       </defs>\r\n       <use xlink:href=\"#DejaVuSans-50\"/>\r\n      </g>\r\n     </g>\r\n    </g>\r\n    <g id=\"xtick_2\">\r\n     <g id=\"line2d_3\">\r\n      <path clip-path=\"url(#p070f028ef1)\" d=\"M 205.95767 499.68 \r\nL 205.95767 64.8 \r\n\" style=\"fill:none;stroke:#ffffff;stroke-linecap:square;stroke-width:0.8;\"/>\r\n     </g>\r\n     <g id=\"line2d_4\">\r\n      <g>\r\n       <use style=\"fill:#555555;stroke:#555555;stroke-width:0.8;\" x=\"205.95767\" xlink:href=\"#m41ac6bf7bb\" y=\"499.68\"/>\r\n      </g>\r\n     </g>\r\n     <g id=\"text_2\">\r\n      <!-- 4 -->\r\n      <g style=\"fill:#555555;\" transform=\"translate(202.77642 514.278437)scale(0.1 -0.1)\">\r\n       <defs>\r\n        <path d=\"M 37.796875 64.3125 \r\nL 12.890625 25.390625 \r\nL 37.796875 25.390625 \r\nz\r\nM 35.203125 72.90625 \r\nL 47.609375 72.90625 \r\nL 47.609375 25.390625 \r\nL 58.015625 25.390625 \r\nL 58.015625 17.1875 \r\nL 47.609375 17.1875 \r\nL 47.609375 0 \r\nL 37.796875 0 \r\nL 37.796875 17.1875 \r\nL 4.890625 17.1875 \r\nL 4.890625 26.703125 \r\nz\r\n\" id=\"DejaVuSans-52\"/>\r\n       </defs>\r\n       <use xlink:href=\"#DejaVuSans-52\"/>\r\n      </g>\r\n     </g>\r\n    </g>\r\n    <g id=\"xtick_3\">\r\n     <g id=\"line2d_5\">\r\n      <path clip-path=\"url(#p070f028ef1)\" d=\"M 341.230398 499.68 \r\nL 341.230398 64.8 \r\n\" style=\"fill:none;stroke:#ffffff;stroke-linecap:square;stroke-width:0.8;\"/>\r\n     </g>\r\n     <g id=\"line2d_6\">\r\n      <g>\r\n       <use style=\"fill:#555555;stroke:#555555;stroke-width:0.8;\" x=\"341.230398\" xlink:href=\"#m41ac6bf7bb\" y=\"499.68\"/>\r\n      </g>\r\n     </g>\r\n     <g id=\"text_3\">\r\n      <!-- 6 -->\r\n      <g style=\"fill:#555555;\" transform=\"translate(338.049148 514.278437)scale(0.1 -0.1)\">\r\n       <defs>\r\n        <path d=\"M 33.015625 40.375 \r\nQ 26.375 40.375 22.484375 35.828125 \r\nQ 18.609375 31.296875 18.609375 23.390625 \r\nQ 18.609375 15.53125 22.484375 10.953125 \r\nQ 26.375 6.390625 33.015625 6.390625 \r\nQ 39.65625 6.390625 43.53125 10.953125 \r\nQ 47.40625 15.53125 47.40625 23.390625 \r\nQ 47.40625 31.296875 43.53125 35.828125 \r\nQ 39.65625 40.375 33.015625 40.375 \r\nz\r\nM 52.59375 71.296875 \r\nL 52.59375 62.3125 \r\nQ 48.875 64.0625 45.09375 64.984375 \r\nQ 41.3125 65.921875 37.59375 65.921875 \r\nQ 27.828125 65.921875 22.671875 59.328125 \r\nQ 17.53125 52.734375 16.796875 39.40625 \r\nQ 19.671875 43.65625 24.015625 45.921875 \r\nQ 28.375 48.1875 33.59375 48.1875 \r\nQ 44.578125 48.1875 50.953125 41.515625 \r\nQ 57.328125 34.859375 57.328125 23.390625 \r\nQ 57.328125 12.15625 50.6875 5.359375 \r\nQ 44.046875 -1.421875 33.015625 -1.421875 \r\nQ 20.359375 -1.421875 13.671875 8.265625 \r\nQ 6.984375 17.96875 6.984375 36.375 \r\nQ 6.984375 53.65625 15.1875 63.9375 \r\nQ 23.390625 74.21875 37.203125 74.21875 \r\nQ 40.921875 74.21875 44.703125 73.484375 \r\nQ 48.484375 72.75 52.59375 71.296875 \r\nz\r\n\" id=\"DejaVuSans-54\"/>\r\n       </defs>\r\n       <use xlink:href=\"#DejaVuSans-54\"/>\r\n      </g>\r\n     </g>\r\n    </g>\r\n    <g id=\"xtick_4\">\r\n     <g id=\"line2d_7\">\r\n      <path clip-path=\"url(#p070f028ef1)\" d=\"M 476.503125 499.68 \r\nL 476.503125 64.8 \r\n\" style=\"fill:none;stroke:#ffffff;stroke-linecap:square;stroke-width:0.8;\"/>\r\n     </g>\r\n     <g id=\"line2d_8\">\r\n      <g>\r\n       <use style=\"fill:#555555;stroke:#555555;stroke-width:0.8;\" x=\"476.503125\" xlink:href=\"#m41ac6bf7bb\" y=\"499.68\"/>\r\n      </g>\r\n     </g>\r\n     <g id=\"text_4\">\r\n      <!-- 8 -->\r\n      <g style=\"fill:#555555;\" transform=\"translate(473.321875 514.278437)scale(0.1 -0.1)\">\r\n       <defs>\r\n        <path d=\"M 31.78125 34.625 \r\nQ 24.75 34.625 20.71875 30.859375 \r\nQ 16.703125 27.09375 16.703125 20.515625 \r\nQ 16.703125 13.921875 20.71875 10.15625 \r\nQ 24.75 6.390625 31.78125 6.390625 \r\nQ 38.8125 6.390625 42.859375 10.171875 \r\nQ 46.921875 13.96875 46.921875 20.515625 \r\nQ 46.921875 27.09375 42.890625 30.859375 \r\nQ 38.875 34.625 31.78125 34.625 \r\nz\r\nM 21.921875 38.8125 \r\nQ 15.578125 40.375 12.03125 44.71875 \r\nQ 8.5 49.078125 8.5 55.328125 \r\nQ 8.5 64.0625 14.71875 69.140625 \r\nQ 20.953125 74.21875 31.78125 74.21875 \r\nQ 42.671875 74.21875 48.875 69.140625 \r\nQ 55.078125 64.0625 55.078125 55.328125 \r\nQ 55.078125 49.078125 51.53125 44.71875 \r\nQ 48 40.375 41.703125 38.8125 \r\nQ 48.828125 37.15625 52.796875 32.3125 \r\nQ 56.78125 27.484375 56.78125 20.515625 \r\nQ 56.78125 9.90625 50.3125 4.234375 \r\nQ 43.84375 -1.421875 31.78125 -1.421875 \r\nQ 19.734375 -1.421875 13.25 4.234375 \r\nQ 6.78125 9.90625 6.78125 20.515625 \r\nQ 6.78125 27.484375 10.78125 32.3125 \r\nQ 14.796875 37.15625 21.921875 38.8125 \r\nz\r\nM 18.3125 54.390625 \r\nQ 18.3125 48.734375 21.84375 45.5625 \r\nQ 25.390625 42.390625 31.78125 42.390625 \r\nQ 38.140625 42.390625 41.71875 45.5625 \r\nQ 45.3125 48.734375 45.3125 54.390625 \r\nQ 45.3125 60.0625 41.71875 63.234375 \r\nQ 38.140625 66.40625 31.78125 66.40625 \r\nQ 25.390625 66.40625 21.84375 63.234375 \r\nQ 18.3125 60.0625 18.3125 54.390625 \r\nz\r\n\" id=\"DejaVuSans-56\"/>\r\n       </defs>\r\n       <use xlink:href=\"#DejaVuSans-56\"/>\r\n      </g>\r\n     </g>\r\n    </g>\r\n    <g id=\"xtick_5\">\r\n     <g id=\"line2d_9\">\r\n      <path clip-path=\"url(#p070f028ef1)\" d=\"M 611.775852 499.68 \r\nL 611.775852 64.8 \r\n\" style=\"fill:none;stroke:#ffffff;stroke-linecap:square;stroke-width:0.8;\"/>\r\n     </g>\r\n     <g id=\"line2d_10\">\r\n      <g>\r\n       <use style=\"fill:#555555;stroke:#555555;stroke-width:0.8;\" x=\"611.775852\" xlink:href=\"#m41ac6bf7bb\" y=\"499.68\"/>\r\n      </g>\r\n     </g>\r\n     <g id=\"text_5\">\r\n      <!-- 10 -->\r\n      <g style=\"fill:#555555;\" transform=\"translate(605.413352 514.278437)scale(0.1 -0.1)\">\r\n       <defs>\r\n        <path d=\"M 12.40625 8.296875 \r\nL 28.515625 8.296875 \r\nL 28.515625 63.921875 \r\nL 10.984375 60.40625 \r\nL 10.984375 69.390625 \r\nL 28.421875 72.90625 \r\nL 38.28125 72.90625 \r\nL 38.28125 8.296875 \r\nL 54.390625 8.296875 \r\nL 54.390625 0 \r\nL 12.40625 0 \r\nz\r\n\" id=\"DejaVuSans-49\"/>\r\n        <path d=\"M 31.78125 66.40625 \r\nQ 24.171875 66.40625 20.328125 58.90625 \r\nQ 16.5 51.421875 16.5 36.375 \r\nQ 16.5 21.390625 20.328125 13.890625 \r\nQ 24.171875 6.390625 31.78125 6.390625 \r\nQ 39.453125 6.390625 43.28125 13.890625 \r\nQ 47.125 21.390625 47.125 36.375 \r\nQ 47.125 51.421875 43.28125 58.90625 \r\nQ 39.453125 66.40625 31.78125 66.40625 \r\nz\r\nM 31.78125 74.21875 \r\nQ 44.046875 74.21875 50.515625 64.515625 \r\nQ 56.984375 54.828125 56.984375 36.375 \r\nQ 56.984375 17.96875 50.515625 8.265625 \r\nQ 44.046875 -1.421875 31.78125 -1.421875 \r\nQ 19.53125 -1.421875 13.0625 8.265625 \r\nQ 6.59375 17.96875 6.59375 36.375 \r\nQ 6.59375 54.828125 13.0625 64.515625 \r\nQ 19.53125 74.21875 31.78125 74.21875 \r\nz\r\n\" id=\"DejaVuSans-48\"/>\r\n       </defs>\r\n       <use xlink:href=\"#DejaVuSans-49\"/>\r\n       <use x=\"63.623047\" xlink:href=\"#DejaVuSans-48\"/>\r\n      </g>\r\n     </g>\r\n    </g>\r\n    <g id=\"xtick_6\">\r\n     <g id=\"line2d_11\">\r\n      <path clip-path=\"url(#p070f028ef1)\" d=\"M 747.04858 499.68 \r\nL 747.04858 64.8 \r\n\" style=\"fill:none;stroke:#ffffff;stroke-linecap:square;stroke-width:0.8;\"/>\r\n     </g>\r\n     <g id=\"line2d_12\">\r\n      <g>\r\n       <use style=\"fill:#555555;stroke:#555555;stroke-width:0.8;\" x=\"747.04858\" xlink:href=\"#m41ac6bf7bb\" y=\"499.68\"/>\r\n      </g>\r\n     </g>\r\n     <g id=\"text_6\">\r\n      <!-- 12 -->\r\n      <g style=\"fill:#555555;\" transform=\"translate(740.68608 514.278437)scale(0.1 -0.1)\">\r\n       <use xlink:href=\"#DejaVuSans-49\"/>\r\n       <use x=\"63.623047\" xlink:href=\"#DejaVuSans-50\"/>\r\n      </g>\r\n     </g>\r\n    </g>\r\n    <g id=\"xtick_7\">\r\n     <g id=\"line2d_13\">\r\n      <path clip-path=\"url(#p070f028ef1)\" d=\"M 882.321307 499.68 \r\nL 882.321307 64.8 \r\n\" style=\"fill:none;stroke:#ffffff;stroke-linecap:square;stroke-width:0.8;\"/>\r\n     </g>\r\n     <g id=\"line2d_14\">\r\n      <g>\r\n       <use style=\"fill:#555555;stroke:#555555;stroke-width:0.8;\" x=\"882.321307\" xlink:href=\"#m41ac6bf7bb\" y=\"499.68\"/>\r\n      </g>\r\n     </g>\r\n     <g id=\"text_7\">\r\n      <!-- 14 -->\r\n      <g style=\"fill:#555555;\" transform=\"translate(875.958807 514.278437)scale(0.1 -0.1)\">\r\n       <use xlink:href=\"#DejaVuSans-49\"/>\r\n       <use x=\"63.623047\" xlink:href=\"#DejaVuSans-52\"/>\r\n      </g>\r\n     </g>\r\n    </g>\r\n   </g>\r\n   <g id=\"matplotlib.axis_2\">\r\n    <g id=\"ytick_1\">\r\n     <g id=\"line2d_15\">\r\n      <path clip-path=\"url(#p070f028ef1)\" d=\"M 30.103125 430.869955 \r\nL 922.903125 430.869955 \r\n\" style=\"fill:none;stroke:#ffffff;stroke-linecap:square;stroke-width:0.8;\"/>\r\n     </g>\r\n     <g id=\"line2d_16\">\r\n      <defs>\r\n       <path d=\"M 0 0 \r\nL -3.5 0 \r\n\" id=\"m56473b1964\" style=\"stroke:#555555;stroke-width:0.8;\"/>\r\n      </defs>\r\n      <g>\r\n       <use style=\"fill:#555555;stroke:#555555;stroke-width:0.8;\" x=\"30.103125\" xlink:href=\"#m56473b1964\" y=\"430.869955\"/>\r\n      </g>\r\n     </g>\r\n     <g id=\"text_8\">\r\n      <!-- 0.5 -->\r\n      <g style=\"fill:#555555;\" transform=\"translate(7.2 434.669174)scale(0.1 -0.1)\">\r\n       <defs>\r\n        <path d=\"M 10.6875 12.40625 \r\nL 21 12.40625 \r\nL 21 0 \r\nL 10.6875 0 \r\nz\r\n\" id=\"DejaVuSans-46\"/>\r\n        <path d=\"M 10.796875 72.90625 \r\nL 49.515625 72.90625 \r\nL 49.515625 64.59375 \r\nL 19.828125 64.59375 \r\nL 19.828125 46.734375 \r\nQ 21.96875 47.46875 24.109375 47.828125 \r\nQ 26.265625 48.1875 28.421875 48.1875 \r\nQ 40.625 48.1875 47.75 41.5 \r\nQ 54.890625 34.8125 54.890625 23.390625 \r\nQ 54.890625 11.625 47.5625 5.09375 \r\nQ 40.234375 -1.421875 26.90625 -1.421875 \r\nQ 22.3125 -1.421875 17.546875 -0.640625 \r\nQ 12.796875 0.140625 7.71875 1.703125 \r\nL 7.71875 11.625 \r\nQ 12.109375 9.234375 16.796875 8.0625 \r\nQ 21.484375 6.890625 26.703125 6.890625 \r\nQ 35.15625 6.890625 40.078125 11.328125 \r\nQ 45.015625 15.765625 45.015625 23.390625 \r\nQ 45.015625 31 40.078125 35.4375 \r\nQ 35.15625 39.890625 26.703125 39.890625 \r\nQ 22.75 39.890625 18.8125 39.015625 \r\nQ 14.890625 38.140625 10.796875 36.28125 \r\nz\r\n\" id=\"DejaVuSans-53\"/>\r\n       </defs>\r\n       <use xlink:href=\"#DejaVuSans-48\"/>\r\n       <use x=\"63.623047\" xlink:href=\"#DejaVuSans-46\"/>\r\n       <use x=\"95.410156\" xlink:href=\"#DejaVuSans-53\"/>\r\n      </g>\r\n     </g>\r\n    </g>\r\n    <g id=\"ytick_2\">\r\n     <g id=\"line2d_17\">\r\n      <path clip-path=\"url(#p070f028ef1)\" d=\"M 30.103125 354.355139 \r\nL 922.903125 354.355139 \r\n\" style=\"fill:none;stroke:#ffffff;stroke-linecap:square;stroke-width:0.8;\"/>\r\n     </g>\r\n     <g id=\"line2d_18\">\r\n      <g>\r\n       <use style=\"fill:#555555;stroke:#555555;stroke-width:0.8;\" x=\"30.103125\" xlink:href=\"#m56473b1964\" y=\"354.355139\"/>\r\n      </g>\r\n     </g>\r\n     <g id=\"text_9\">\r\n      <!-- 1.0 -->\r\n      <g style=\"fill:#555555;\" transform=\"translate(7.2 358.154358)scale(0.1 -0.1)\">\r\n       <use xlink:href=\"#DejaVuSans-49\"/>\r\n       <use x=\"63.623047\" xlink:href=\"#DejaVuSans-46\"/>\r\n       <use x=\"95.410156\" xlink:href=\"#DejaVuSans-48\"/>\r\n      </g>\r\n     </g>\r\n    </g>\r\n    <g id=\"ytick_3\">\r\n     <g id=\"line2d_19\">\r\n      <path clip-path=\"url(#p070f028ef1)\" d=\"M 30.103125 277.840324 \r\nL 922.903125 277.840324 \r\n\" style=\"fill:none;stroke:#ffffff;stroke-linecap:square;stroke-width:0.8;\"/>\r\n     </g>\r\n     <g id=\"line2d_20\">\r\n      <g>\r\n       <use style=\"fill:#555555;stroke:#555555;stroke-width:0.8;\" x=\"30.103125\" xlink:href=\"#m56473b1964\" y=\"277.840324\"/>\r\n      </g>\r\n     </g>\r\n     <g id=\"text_10\">\r\n      <!-- 1.5 -->\r\n      <g style=\"fill:#555555;\" transform=\"translate(7.2 281.639542)scale(0.1 -0.1)\">\r\n       <use xlink:href=\"#DejaVuSans-49\"/>\r\n       <use x=\"63.623047\" xlink:href=\"#DejaVuSans-46\"/>\r\n       <use x=\"95.410156\" xlink:href=\"#DejaVuSans-53\"/>\r\n      </g>\r\n     </g>\r\n    </g>\r\n    <g id=\"ytick_4\">\r\n     <g id=\"line2d_21\">\r\n      <path clip-path=\"url(#p070f028ef1)\" d=\"M 30.103125 201.325508 \r\nL 922.903125 201.325508 \r\n\" style=\"fill:none;stroke:#ffffff;stroke-linecap:square;stroke-width:0.8;\"/>\r\n     </g>\r\n     <g id=\"line2d_22\">\r\n      <g>\r\n       <use style=\"fill:#555555;stroke:#555555;stroke-width:0.8;\" x=\"30.103125\" xlink:href=\"#m56473b1964\" y=\"201.325508\"/>\r\n      </g>\r\n     </g>\r\n     <g id=\"text_11\">\r\n      <!-- 2.0 -->\r\n      <g style=\"fill:#555555;\" transform=\"translate(7.2 205.124727)scale(0.1 -0.1)\">\r\n       <use xlink:href=\"#DejaVuSans-50\"/>\r\n       <use x=\"63.623047\" xlink:href=\"#DejaVuSans-46\"/>\r\n       <use x=\"95.410156\" xlink:href=\"#DejaVuSans-48\"/>\r\n      </g>\r\n     </g>\r\n    </g>\r\n    <g id=\"ytick_5\">\r\n     <g id=\"line2d_23\">\r\n      <path clip-path=\"url(#p070f028ef1)\" d=\"M 30.103125 124.810692 \r\nL 922.903125 124.810692 \r\n\" style=\"fill:none;stroke:#ffffff;stroke-linecap:square;stroke-width:0.8;\"/>\r\n     </g>\r\n     <g id=\"line2d_24\">\r\n      <g>\r\n       <use style=\"fill:#555555;stroke:#555555;stroke-width:0.8;\" x=\"30.103125\" xlink:href=\"#m56473b1964\" y=\"124.810692\"/>\r\n      </g>\r\n     </g>\r\n     <g id=\"text_12\">\r\n      <!-- 2.5 -->\r\n      <g style=\"fill:#555555;\" transform=\"translate(7.2 128.609911)scale(0.1 -0.1)\">\r\n       <use xlink:href=\"#DejaVuSans-50\"/>\r\n       <use x=\"63.623047\" xlink:href=\"#DejaVuSans-46\"/>\r\n       <use x=\"95.410156\" xlink:href=\"#DejaVuSans-53\"/>\r\n      </g>\r\n     </g>\r\n    </g>\r\n    <g id=\"text_13\">\r\n     <!-- 1e11 -->\r\n     <g style=\"fill:#555555;\" transform=\"translate(30.103125 61.8)scale(0.1 -0.1)\">\r\n      <defs>\r\n       <path d=\"M 56.203125 29.59375 \r\nL 56.203125 25.203125 \r\nL 14.890625 25.203125 \r\nQ 15.484375 15.921875 20.484375 11.0625 \r\nQ 25.484375 6.203125 34.421875 6.203125 \r\nQ 39.59375 6.203125 44.453125 7.46875 \r\nQ 49.3125 8.734375 54.109375 11.28125 \r\nL 54.109375 2.78125 \r\nQ 49.265625 0.734375 44.1875 -0.34375 \r\nQ 39.109375 -1.421875 33.890625 -1.421875 \r\nQ 20.796875 -1.421875 13.15625 6.1875 \r\nQ 5.515625 13.8125 5.515625 26.8125 \r\nQ 5.515625 40.234375 12.765625 48.109375 \r\nQ 20.015625 56 32.328125 56 \r\nQ 43.359375 56 49.78125 48.890625 \r\nQ 56.203125 41.796875 56.203125 29.59375 \r\nz\r\nM 47.21875 32.234375 \r\nQ 47.125 39.59375 43.09375 43.984375 \r\nQ 39.0625 48.390625 32.421875 48.390625 \r\nQ 24.90625 48.390625 20.390625 44.140625 \r\nQ 15.875 39.890625 15.1875 32.171875 \r\nz\r\n\" id=\"DejaVuSans-101\"/>\r\n      </defs>\r\n      <use xlink:href=\"#DejaVuSans-49\"/>\r\n      <use x=\"63.623047\" xlink:href=\"#DejaVuSans-101\"/>\r\n      <use x=\"125.146484\" xlink:href=\"#DejaVuSans-49\"/>\r\n      <use x=\"188.769531\" xlink:href=\"#DejaVuSans-49\"/>\r\n     </g>\r\n    </g>\r\n   </g>\r\n   <g id=\"line2d_25\">\r\n    <path clip-path=\"url(#p070f028ef1)\" d=\"M 70.684943 84.567273 \r\nL 138.321307 282.725222 \r\nL 205.95767 366.299472 \r\nL 273.594034 395.348733 \r\nL 341.230398 415.237487 \r\nL 408.866761 433.57954 \r\nL 476.503125 445.441305 \r\nL 544.139489 457.005722 \r\nL 611.775852 464.770164 \r\nL 679.412216 470.027588 \r\nL 747.04858 473.901494 \r\nL 814.684943 477.009277 \r\nL 882.321307 479.912727 \r\n\" style=\"fill:none;stroke:#e24a33;stroke-linecap:square;stroke-width:1.5;\"/>\r\n   </g>\r\n   <g id=\"patch_3\">\r\n    <path d=\"M 30.103125 499.68 \r\nL 30.103125 64.8 \r\n\" style=\"fill:none;stroke:#ffffff;stroke-linecap:square;stroke-linejoin:miter;\"/>\r\n   </g>\r\n   <g id=\"patch_4\">\r\n    <path d=\"M 922.903125 499.68 \r\nL 922.903125 64.8 \r\n\" style=\"fill:none;stroke:#ffffff;stroke-linecap:square;stroke-linejoin:miter;\"/>\r\n   </g>\r\n   <g id=\"patch_5\">\r\n    <path d=\"M 30.103125 499.68 \r\nL 922.903125 499.68 \r\n\" style=\"fill:none;stroke:#ffffff;stroke-linecap:square;stroke-linejoin:miter;\"/>\r\n   </g>\r\n   <g id=\"patch_6\">\r\n    <path d=\"M 30.103125 64.8 \r\nL 922.903125 64.8 \r\n\" style=\"fill:none;stroke:#ffffff;stroke-linecap:square;stroke-linejoin:miter;\"/>\r\n   </g>\r\n  </g>\r\n  <g id=\"text_14\">\r\n   <!-- Variación de la dispersión de los clústers en función de la k -->\r\n   <g transform=\"translate(225.186875 19.9975)scale(0.16 -0.16)\">\r\n    <defs>\r\n     <path d=\"M 28.609375 0 \r\nL 0.78125 72.90625 \r\nL 11.078125 72.90625 \r\nL 34.1875 11.53125 \r\nL 57.328125 72.90625 \r\nL 67.578125 72.90625 \r\nL 39.796875 0 \r\nz\r\n\" id=\"DejaVuSans-86\"/>\r\n     <path d=\"M 34.28125 27.484375 \r\nQ 23.390625 27.484375 19.1875 25 \r\nQ 14.984375 22.515625 14.984375 16.5 \r\nQ 14.984375 11.71875 18.140625 8.90625 \r\nQ 21.296875 6.109375 26.703125 6.109375 \r\nQ 34.1875 6.109375 38.703125 11.40625 \r\nQ 43.21875 16.703125 43.21875 25.484375 \r\nL 43.21875 27.484375 \r\nz\r\nM 52.203125 31.203125 \r\nL 52.203125 0 \r\nL 43.21875 0 \r\nL 43.21875 8.296875 \r\nQ 40.140625 3.328125 35.546875 0.953125 \r\nQ 30.953125 -1.421875 24.3125 -1.421875 \r\nQ 15.921875 -1.421875 10.953125 3.296875 \r\nQ 6 8.015625 6 15.921875 \r\nQ 6 25.140625 12.171875 29.828125 \r\nQ 18.359375 34.515625 30.609375 34.515625 \r\nL 43.21875 34.515625 \r\nL 43.21875 35.40625 \r\nQ 43.21875 41.609375 39.140625 45 \r\nQ 35.0625 48.390625 27.6875 48.390625 \r\nQ 23 48.390625 18.546875 47.265625 \r\nQ 14.109375 46.140625 10.015625 43.890625 \r\nL 10.015625 52.203125 \r\nQ 14.9375 54.109375 19.578125 55.046875 \r\nQ 24.21875 56 28.609375 56 \r\nQ 40.484375 56 46.34375 49.84375 \r\nQ 52.203125 43.703125 52.203125 31.203125 \r\nz\r\n\" id=\"DejaVuSans-97\"/>\r\n     <path d=\"M 41.109375 46.296875 \r\nQ 39.59375 47.171875 37.8125 47.578125 \r\nQ 36.03125 48 33.890625 48 \r\nQ 26.265625 48 22.1875 43.046875 \r\nQ 18.109375 38.09375 18.109375 28.8125 \r\nL 18.109375 0 \r\nL 9.078125 0 \r\nL 9.078125 54.6875 \r\nL 18.109375 54.6875 \r\nL 18.109375 46.1875 \r\nQ 20.953125 51.171875 25.484375 53.578125 \r\nQ 30.03125 56 36.53125 56 \r\nQ 37.453125 56 38.578125 55.875 \r\nQ 39.703125 55.765625 41.0625 55.515625 \r\nz\r\n\" id=\"DejaVuSans-114\"/>\r\n     <path d=\"M 9.421875 54.6875 \r\nL 18.40625 54.6875 \r\nL 18.40625 0 \r\nL 9.421875 0 \r\nz\r\nM 9.421875 75.984375 \r\nL 18.40625 75.984375 \r\nL 18.40625 64.59375 \r\nL 9.421875 64.59375 \r\nz\r\n\" id=\"DejaVuSans-105\"/>\r\n     <path d=\"M 48.78125 52.59375 \r\nL 48.78125 44.1875 \r\nQ 44.96875 46.296875 41.140625 47.34375 \r\nQ 37.3125 48.390625 33.40625 48.390625 \r\nQ 24.65625 48.390625 19.8125 42.84375 \r\nQ 14.984375 37.3125 14.984375 27.296875 \r\nQ 14.984375 17.28125 19.8125 11.734375 \r\nQ 24.65625 6.203125 33.40625 6.203125 \r\nQ 37.3125 6.203125 41.140625 7.25 \r\nQ 44.96875 8.296875 48.78125 10.40625 \r\nL 48.78125 2.09375 \r\nQ 45.015625 0.34375 40.984375 -0.53125 \r\nQ 36.96875 -1.421875 32.421875 -1.421875 \r\nQ 20.0625 -1.421875 12.78125 6.34375 \r\nQ 5.515625 14.109375 5.515625 27.296875 \r\nQ 5.515625 40.671875 12.859375 48.328125 \r\nQ 20.21875 56 33.015625 56 \r\nQ 37.15625 56 41.109375 55.140625 \r\nQ 45.0625 54.296875 48.78125 52.59375 \r\nz\r\n\" id=\"DejaVuSans-99\"/>\r\n     <path d=\"M 30.609375 48.390625 \r\nQ 23.390625 48.390625 19.1875 42.75 \r\nQ 14.984375 37.109375 14.984375 27.296875 \r\nQ 14.984375 17.484375 19.15625 11.84375 \r\nQ 23.34375 6.203125 30.609375 6.203125 \r\nQ 37.796875 6.203125 41.984375 11.859375 \r\nQ 46.1875 17.53125 46.1875 27.296875 \r\nQ 46.1875 37.015625 41.984375 42.703125 \r\nQ 37.796875 48.390625 30.609375 48.390625 \r\nz\r\nM 30.609375 56 \r\nQ 42.328125 56 49.015625 48.375 \r\nQ 55.71875 40.765625 55.71875 27.296875 \r\nQ 55.71875 13.875 49.015625 6.21875 \r\nQ 42.328125 -1.421875 30.609375 -1.421875 \r\nQ 18.84375 -1.421875 12.171875 6.21875 \r\nQ 5.515625 13.875 5.515625 27.296875 \r\nQ 5.515625 40.765625 12.171875 48.375 \r\nQ 18.84375 56 30.609375 56 \r\nz\r\nM 37.40625 79.984375 \r\nL 47.125 79.984375 \r\nL 31.203125 61.625 \r\nL 23.734375 61.625 \r\nz\r\n\" id=\"DejaVuSans-243\"/>\r\n     <path d=\"M 54.890625 33.015625 \r\nL 54.890625 0 \r\nL 45.90625 0 \r\nL 45.90625 32.71875 \r\nQ 45.90625 40.484375 42.875 44.328125 \r\nQ 39.84375 48.1875 33.796875 48.1875 \r\nQ 26.515625 48.1875 22.3125 43.546875 \r\nQ 18.109375 38.921875 18.109375 30.90625 \r\nL 18.109375 0 \r\nL 9.078125 0 \r\nL 9.078125 54.6875 \r\nL 18.109375 54.6875 \r\nL 18.109375 46.1875 \r\nQ 21.34375 51.125 25.703125 53.5625 \r\nQ 30.078125 56 35.796875 56 \r\nQ 45.21875 56 50.046875 50.171875 \r\nQ 54.890625 44.34375 54.890625 33.015625 \r\nz\r\n\" id=\"DejaVuSans-110\"/>\r\n     <path id=\"DejaVuSans-32\"/>\r\n     <path d=\"M 45.40625 46.390625 \r\nL 45.40625 75.984375 \r\nL 54.390625 75.984375 \r\nL 54.390625 0 \r\nL 45.40625 0 \r\nL 45.40625 8.203125 \r\nQ 42.578125 3.328125 38.25 0.953125 \r\nQ 33.9375 -1.421875 27.875 -1.421875 \r\nQ 17.96875 -1.421875 11.734375 6.484375 \r\nQ 5.515625 14.40625 5.515625 27.296875 \r\nQ 5.515625 40.1875 11.734375 48.09375 \r\nQ 17.96875 56 27.875 56 \r\nQ 33.9375 56 38.25 53.625 \r\nQ 42.578125 51.265625 45.40625 46.390625 \r\nz\r\nM 14.796875 27.296875 \r\nQ 14.796875 17.390625 18.875 11.75 \r\nQ 22.953125 6.109375 30.078125 6.109375 \r\nQ 37.203125 6.109375 41.296875 11.75 \r\nQ 45.40625 17.390625 45.40625 27.296875 \r\nQ 45.40625 37.203125 41.296875 42.84375 \r\nQ 37.203125 48.484375 30.078125 48.484375 \r\nQ 22.953125 48.484375 18.875 42.84375 \r\nQ 14.796875 37.203125 14.796875 27.296875 \r\nz\r\n\" id=\"DejaVuSans-100\"/>\r\n     <path d=\"M 9.421875 75.984375 \r\nL 18.40625 75.984375 \r\nL 18.40625 0 \r\nL 9.421875 0 \r\nz\r\n\" id=\"DejaVuSans-108\"/>\r\n     <path d=\"M 44.28125 53.078125 \r\nL 44.28125 44.578125 \r\nQ 40.484375 46.53125 36.375 47.5 \r\nQ 32.28125 48.484375 27.875 48.484375 \r\nQ 21.1875 48.484375 17.84375 46.4375 \r\nQ 14.5 44.390625 14.5 40.28125 \r\nQ 14.5 37.15625 16.890625 35.375 \r\nQ 19.28125 33.59375 26.515625 31.984375 \r\nL 29.59375 31.296875 \r\nQ 39.15625 29.25 43.1875 25.515625 \r\nQ 47.21875 21.78125 47.21875 15.09375 \r\nQ 47.21875 7.46875 41.1875 3.015625 \r\nQ 35.15625 -1.421875 24.609375 -1.421875 \r\nQ 20.21875 -1.421875 15.453125 -0.5625 \r\nQ 10.6875 0.296875 5.421875 2 \r\nL 5.421875 11.28125 \r\nQ 10.40625 8.6875 15.234375 7.390625 \r\nQ 20.0625 6.109375 24.8125 6.109375 \r\nQ 31.15625 6.109375 34.5625 8.28125 \r\nQ 37.984375 10.453125 37.984375 14.40625 \r\nQ 37.984375 18.0625 35.515625 20.015625 \r\nQ 33.0625 21.96875 24.703125 23.78125 \r\nL 21.578125 24.515625 \r\nQ 13.234375 26.265625 9.515625 29.90625 \r\nQ 5.8125 33.546875 5.8125 39.890625 \r\nQ 5.8125 47.609375 11.28125 51.796875 \r\nQ 16.75 56 26.8125 56 \r\nQ 31.78125 56 36.171875 55.265625 \r\nQ 40.578125 54.546875 44.28125 53.078125 \r\nz\r\n\" id=\"DejaVuSans-115\"/>\r\n     <path d=\"M 18.109375 8.203125 \r\nL 18.109375 -20.796875 \r\nL 9.078125 -20.796875 \r\nL 9.078125 54.6875 \r\nL 18.109375 54.6875 \r\nL 18.109375 46.390625 \r\nQ 20.953125 51.265625 25.265625 53.625 \r\nQ 29.59375 56 35.59375 56 \r\nQ 45.5625 56 51.78125 48.09375 \r\nQ 58.015625 40.1875 58.015625 27.296875 \r\nQ 58.015625 14.40625 51.78125 6.484375 \r\nQ 45.5625 -1.421875 35.59375 -1.421875 \r\nQ 29.59375 -1.421875 25.265625 0.953125 \r\nQ 20.953125 3.328125 18.109375 8.203125 \r\nz\r\nM 48.6875 27.296875 \r\nQ 48.6875 37.203125 44.609375 42.84375 \r\nQ 40.53125 48.484375 33.40625 48.484375 \r\nQ 26.265625 48.484375 22.1875 42.84375 \r\nQ 18.109375 37.203125 18.109375 27.296875 \r\nQ 18.109375 17.390625 22.1875 11.75 \r\nQ 26.265625 6.109375 33.40625 6.109375 \r\nQ 40.53125 6.109375 44.609375 11.75 \r\nQ 48.6875 17.390625 48.6875 27.296875 \r\nz\r\n\" id=\"DejaVuSans-112\"/>\r\n     <path d=\"M 30.609375 48.390625 \r\nQ 23.390625 48.390625 19.1875 42.75 \r\nQ 14.984375 37.109375 14.984375 27.296875 \r\nQ 14.984375 17.484375 19.15625 11.84375 \r\nQ 23.34375 6.203125 30.609375 6.203125 \r\nQ 37.796875 6.203125 41.984375 11.859375 \r\nQ 46.1875 17.53125 46.1875 27.296875 \r\nQ 46.1875 37.015625 41.984375 42.703125 \r\nQ 37.796875 48.390625 30.609375 48.390625 \r\nz\r\nM 30.609375 56 \r\nQ 42.328125 56 49.015625 48.375 \r\nQ 55.71875 40.765625 55.71875 27.296875 \r\nQ 55.71875 13.875 49.015625 6.21875 \r\nQ 42.328125 -1.421875 30.609375 -1.421875 \r\nQ 18.84375 -1.421875 12.171875 6.21875 \r\nQ 5.515625 13.875 5.515625 27.296875 \r\nQ 5.515625 40.765625 12.171875 48.375 \r\nQ 18.84375 56 30.609375 56 \r\nz\r\n\" id=\"DejaVuSans-111\"/>\r\n     <path d=\"M 8.5 21.578125 \r\nL 8.5 54.6875 \r\nL 17.484375 54.6875 \r\nL 17.484375 21.921875 \r\nQ 17.484375 14.15625 20.5 10.265625 \r\nQ 23.53125 6.390625 29.59375 6.390625 \r\nQ 36.859375 6.390625 41.078125 11.03125 \r\nQ 45.3125 15.671875 45.3125 23.6875 \r\nL 45.3125 54.6875 \r\nL 54.296875 54.6875 \r\nL 54.296875 0 \r\nL 45.3125 0 \r\nL 45.3125 8.40625 \r\nQ 42.046875 3.421875 37.71875 1 \r\nQ 33.40625 -1.421875 27.6875 -1.421875 \r\nQ 18.265625 -1.421875 13.375 4.4375 \r\nQ 8.5 10.296875 8.5 21.578125 \r\nz\r\nM 31.109375 56 \r\nz\r\nM 37.796875 79.984375 \r\nL 47.515625 79.984375 \r\nL 31.59375 61.625 \r\nL 24.125 61.625 \r\nz\r\n\" id=\"DejaVuSans-250\"/>\r\n     <path d=\"M 18.3125 70.21875 \r\nL 18.3125 54.6875 \r\nL 36.8125 54.6875 \r\nL 36.8125 47.703125 \r\nL 18.3125 47.703125 \r\nL 18.3125 18.015625 \r\nQ 18.3125 11.328125 20.140625 9.421875 \r\nQ 21.96875 7.515625 27.59375 7.515625 \r\nL 36.8125 7.515625 \r\nL 36.8125 0 \r\nL 27.59375 0 \r\nQ 17.1875 0 13.234375 3.875 \r\nQ 9.28125 7.765625 9.28125 18.015625 \r\nL 9.28125 47.703125 \r\nL 2.6875 47.703125 \r\nL 2.6875 54.6875 \r\nL 9.28125 54.6875 \r\nL 9.28125 70.21875 \r\nz\r\n\" id=\"DejaVuSans-116\"/>\r\n     <path d=\"M 37.109375 75.984375 \r\nL 37.109375 68.5 \r\nL 28.515625 68.5 \r\nQ 23.6875 68.5 21.796875 66.546875 \r\nQ 19.921875 64.59375 19.921875 59.515625 \r\nL 19.921875 54.6875 \r\nL 34.71875 54.6875 \r\nL 34.71875 47.703125 \r\nL 19.921875 47.703125 \r\nL 19.921875 0 \r\nL 10.890625 0 \r\nL 10.890625 47.703125 \r\nL 2.296875 47.703125 \r\nL 2.296875 54.6875 \r\nL 10.890625 54.6875 \r\nL 10.890625 58.5 \r\nQ 10.890625 67.625 15.140625 71.796875 \r\nQ 19.390625 75.984375 28.609375 75.984375 \r\nz\r\n\" id=\"DejaVuSans-102\"/>\r\n     <path d=\"M 8.5 21.578125 \r\nL 8.5 54.6875 \r\nL 17.484375 54.6875 \r\nL 17.484375 21.921875 \r\nQ 17.484375 14.15625 20.5 10.265625 \r\nQ 23.53125 6.390625 29.59375 6.390625 \r\nQ 36.859375 6.390625 41.078125 11.03125 \r\nQ 45.3125 15.671875 45.3125 23.6875 \r\nL 45.3125 54.6875 \r\nL 54.296875 54.6875 \r\nL 54.296875 0 \r\nL 45.3125 0 \r\nL 45.3125 8.40625 \r\nQ 42.046875 3.421875 37.71875 1 \r\nQ 33.40625 -1.421875 27.6875 -1.421875 \r\nQ 18.265625 -1.421875 13.375 4.4375 \r\nQ 8.5 10.296875 8.5 21.578125 \r\nz\r\nM 31.109375 56 \r\nz\r\n\" id=\"DejaVuSans-117\"/>\r\n     <path d=\"M 9.078125 75.984375 \r\nL 18.109375 75.984375 \r\nL 18.109375 31.109375 \r\nL 44.921875 54.6875 \r\nL 56.390625 54.6875 \r\nL 27.390625 29.109375 \r\nL 57.625 0 \r\nL 45.90625 0 \r\nL 18.109375 26.703125 \r\nL 18.109375 0 \r\nL 9.078125 0 \r\nz\r\n\" id=\"DejaVuSans-107\"/>\r\n    </defs>\r\n    <use xlink:href=\"#DejaVuSans-86\"/>\r\n    <use x=\"60.658203\" xlink:href=\"#DejaVuSans-97\"/>\r\n    <use x=\"121.9375\" xlink:href=\"#DejaVuSans-114\"/>\r\n    <use x=\"163.050781\" xlink:href=\"#DejaVuSans-105\"/>\r\n    <use x=\"190.833984\" xlink:href=\"#DejaVuSans-97\"/>\r\n    <use x=\"252.113281\" xlink:href=\"#DejaVuSans-99\"/>\r\n    <use x=\"307.09375\" xlink:href=\"#DejaVuSans-105\"/>\r\n    <use x=\"334.876953\" xlink:href=\"#DejaVuSans-243\"/>\r\n    <use x=\"396.058594\" xlink:href=\"#DejaVuSans-110\"/>\r\n    <use x=\"459.4375\" xlink:href=\"#DejaVuSans-32\"/>\r\n    <use x=\"491.224609\" xlink:href=\"#DejaVuSans-100\"/>\r\n    <use x=\"554.701172\" xlink:href=\"#DejaVuSans-101\"/>\r\n    <use x=\"616.224609\" xlink:href=\"#DejaVuSans-32\"/>\r\n    <use x=\"648.011719\" xlink:href=\"#DejaVuSans-108\"/>\r\n    <use x=\"675.794922\" xlink:href=\"#DejaVuSans-97\"/>\r\n    <use x=\"737.074219\" xlink:href=\"#DejaVuSans-32\"/>\r\n    <use x=\"768.861328\" xlink:href=\"#DejaVuSans-100\"/>\r\n    <use x=\"832.337891\" xlink:href=\"#DejaVuSans-105\"/>\r\n    <use x=\"860.121094\" xlink:href=\"#DejaVuSans-115\"/>\r\n    <use x=\"912.220703\" xlink:href=\"#DejaVuSans-112\"/>\r\n    <use x=\"975.697266\" xlink:href=\"#DejaVuSans-101\"/>\r\n    <use x=\"1037.220703\" xlink:href=\"#DejaVuSans-114\"/>\r\n    <use x=\"1078.333984\" xlink:href=\"#DejaVuSans-115\"/>\r\n    <use x=\"1130.433594\" xlink:href=\"#DejaVuSans-105\"/>\r\n    <use x=\"1158.216797\" xlink:href=\"#DejaVuSans-243\"/>\r\n    <use x=\"1219.398438\" xlink:href=\"#DejaVuSans-110\"/>\r\n    <use x=\"1282.777344\" xlink:href=\"#DejaVuSans-32\"/>\r\n    <use x=\"1314.564453\" xlink:href=\"#DejaVuSans-100\"/>\r\n    <use x=\"1378.041016\" xlink:href=\"#DejaVuSans-101\"/>\r\n    <use x=\"1439.564453\" xlink:href=\"#DejaVuSans-32\"/>\r\n    <use x=\"1471.351562\" xlink:href=\"#DejaVuSans-108\"/>\r\n    <use x=\"1499.134766\" xlink:href=\"#DejaVuSans-111\"/>\r\n    <use x=\"1560.316406\" xlink:href=\"#DejaVuSans-115\"/>\r\n    <use x=\"1612.416016\" xlink:href=\"#DejaVuSans-32\"/>\r\n    <use x=\"1644.203125\" xlink:href=\"#DejaVuSans-99\"/>\r\n    <use x=\"1699.183594\" xlink:href=\"#DejaVuSans-108\"/>\r\n    <use x=\"1726.966797\" xlink:href=\"#DejaVuSans-250\"/>\r\n    <use x=\"1790.345703\" xlink:href=\"#DejaVuSans-115\"/>\r\n    <use x=\"1842.445312\" xlink:href=\"#DejaVuSans-116\"/>\r\n    <use x=\"1881.654297\" xlink:href=\"#DejaVuSans-101\"/>\r\n    <use x=\"1943.177734\" xlink:href=\"#DejaVuSans-114\"/>\r\n    <use x=\"1984.291016\" xlink:href=\"#DejaVuSans-115\"/>\r\n    <use x=\"2036.390625\" xlink:href=\"#DejaVuSans-32\"/>\r\n    <use x=\"2068.177734\" xlink:href=\"#DejaVuSans-101\"/>\r\n    <use x=\"2129.701172\" xlink:href=\"#DejaVuSans-110\"/>\r\n    <use x=\"2193.080078\" xlink:href=\"#DejaVuSans-32\"/>\r\n    <use x=\"2224.867188\" xlink:href=\"#DejaVuSans-102\"/>\r\n    <use x=\"2260.072266\" xlink:href=\"#DejaVuSans-117\"/>\r\n    <use x=\"2323.451172\" xlink:href=\"#DejaVuSans-110\"/>\r\n    <use x=\"2386.830078\" xlink:href=\"#DejaVuSans-99\"/>\r\n    <use x=\"2441.810547\" xlink:href=\"#DejaVuSans-105\"/>\r\n    <use x=\"2469.59375\" xlink:href=\"#DejaVuSans-243\"/>\r\n    <use x=\"2530.775391\" xlink:href=\"#DejaVuSans-110\"/>\r\n    <use x=\"2594.154297\" xlink:href=\"#DejaVuSans-32\"/>\r\n    <use x=\"2625.941406\" xlink:href=\"#DejaVuSans-100\"/>\r\n    <use x=\"2689.417969\" xlink:href=\"#DejaVuSans-101\"/>\r\n    <use x=\"2750.941406\" xlink:href=\"#DejaVuSans-32\"/>\r\n    <use x=\"2782.728516\" xlink:href=\"#DejaVuSans-108\"/>\r\n    <use x=\"2810.511719\" xlink:href=\"#DejaVuSans-97\"/>\r\n    <use x=\"2871.791016\" xlink:href=\"#DejaVuSans-32\"/>\r\n    <use x=\"2903.578125\" xlink:href=\"#DejaVuSans-107\"/>\r\n   </g>\r\n  </g>\r\n </g>\r\n <defs>\r\n  <clipPath id=\"p070f028ef1\">\r\n   <rect height=\"434.88\" width=\"892.8\" x=\"30.103125\" y=\"64.8\"/>\r\n  </clipPath>\r\n </defs>\r\n</svg>\r\n",
                        "image/png": "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\n"
                    },
                    "metadata": {
                        "needs_background": "light"
                    }
                }
            ],
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
            "execution_count": 113,
            "metadata": {},
            "outputs": [],
            "source": [
                "kmeans = KMeans(n_clusters = 4)\n",
                "kmeans.fit(pca_df)\n",
                "pca_df['cluster'] = kmeans.labels_"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": 114,
            "metadata": {},
            "outputs": [
                {
                    "output_type": "execute_result",
                    "data": {
                        "text/plain": [
                            "KMeans(n_clusters=4)"
                        ]
                    },
                    "metadata": {},
                    "execution_count": 114
                }
            ],
            "source": [
                "# fiteamos un modelo con k = 5 (que hemos sacado de la elbow curve anterior) \n",
                "# y con el dataframe escalado y sin outliers\n",
                "\n",
                "cluster_model = KMeans(n_clusters = 4)\n",
                "cluster_model.fit(scaled_df)"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": 116,
            "metadata": {},
            "outputs": [],
            "source": [
                "# generamos el dataframe escalado (con el scaler del paso anterior, entrado sin outliers) pero con todos los datos.\n",
                "# por tanto vamos a transformar incluso a los outliers pero con el scaler entrado sin ellos.\n",
                "# el motivo es porque los outliers pueden afectar mucho la media y la desviación utilizado para transformar.\n",
                "scaled_df_with_outliers = standard_scaler.transform(full_4)"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": 117,
            "metadata": {},
            "outputs": [],
            "source": [
                "# convertimos a dataframe\n",
                "scaled_df_with_outliers = pd.DataFrame(scaled_df_with_outliers, \n",
                "                                       index = full_4.index, \n",
                "                                       columns = full_4.columns)"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": 118,
            "metadata": {},
            "outputs": [
                {
                    "output_type": "execute_result",
                    "data": {
                        "text/plain": [
                            "(6254518, 128)"
                        ]
                    },
                    "metadata": {},
                    "execution_count": 118
                }
            ],
            "source": [
                "scaled_df_with_outliers.shape"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": 119,
            "metadata": {},
            "outputs": [],
            "source": [
                "# calculamos el cluster de cada cliente, a partir del dataframe escalado y con outliers\n",
                "labels = cluster_model.predict(scaled_df_with_outliers)"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": 120,
            "metadata": {},
            "outputs": [
                {
                    "output_type": "execute_result",
                    "data": {
                        "text/plain": [
                            "         Ventas_cant  Precios  Ingresos  días_encartera  region_code  age  \\\n",
                            "pk_cid                                                                      \n",
                            "100296           1.0       40      40.0            20.0         28.0   46   \n",
                            "100296           1.0       40      40.0            50.0         28.0   46   \n",
                            "100296           1.0       40      40.0            81.0         28.0   46   \n",
                            "1003705          1.0       40      40.0          1337.0         28.0   33   \n",
                            "1003705          1.0       40      40.0          1367.0         28.0   33   \n",
                            "1003705          1.0       40      40.0          1398.0         28.0   33   \n",
                            "1004643          1.0       40      40.0          1115.0          3.0   30   \n",
                            "\n",
                            "         Month  days_between  recurrencia  Productos_credit_card  ...  \\\n",
                            "pk_cid                                                            ...   \n",
                            "100296       4           0.0            1                      0  ...   \n",
                            "100296       5          30.0            2                      0  ...   \n",
                            "100296       6          31.0            3                      0  ...   \n",
                            "1003705      9           0.0            1                      0  ...   \n",
                            "1003705     10          30.0            2                      0  ...   \n",
                            "1003705     11          31.0            3                      0  ...   \n",
                            "1004643      2           0.0            1                      0  ...   \n",
                            "\n",
                            "         country_id_SA  country_id_SE  country_id_SN  country_id_US  \\\n",
                            "pk_cid                                                                \n",
                            "100296               0              0              0              0   \n",
                            "100296               0              0              0              0   \n",
                            "100296               0              0              0              0   \n",
                            "1003705              0              0              0              0   \n",
                            "1003705              0              0              0              0   \n",
                            "1003705              0              0              0              0   \n",
                            "1004643              0              0              0              0   \n",
                            "\n",
                            "         country_id_VE  gender_H  gender_V  deceased_N  deceased_S  cluster  \n",
                            "pk_cid                                                                       \n",
                            "100296               0         0         1           1           0        0  \n",
                            "100296               0         0         1           1           0        0  \n",
                            "100296               0         0         1           1           0        0  \n",
                            "1003705              0         1         0           1           0        0  \n",
                            "1003705              0         1         0           1           0        0  \n",
                            "1003705              0         1         0           1           0        0  \n",
                            "1004643              0         0         1           1           0        0  \n",
                            "\n",
                            "[7 rows x 128 columns]"
                        ],
                        "text/html": "<div>\n<style scoped>\n    .dataframe tbody tr th:only-of-type {\n        vertical-align: middle;\n    }\n\n    .dataframe tbody tr th {\n        vertical-align: top;\n    }\n\n    .dataframe thead th {\n        text-align: right;\n    }\n</style>\n<table border=\"1\" class=\"dataframe\">\n  <thead>\n    <tr style=\"text-align: right;\">\n      <th></th>\n      <th>Ventas_cant</th>\n      <th>Precios</th>\n      <th>Ingresos</th>\n      <th>días_encartera</th>\n      <th>region_code</th>\n      <th>age</th>\n      <th>Month</th>\n      <th>days_between</th>\n      <th>recurrencia</th>\n      <th>Productos_credit_card</th>\n      <th>...</th>\n      <th>country_id_SA</th>\n      <th>country_id_SE</th>\n      <th>country_id_SN</th>\n      <th>country_id_US</th>\n      <th>country_id_VE</th>\n      <th>gender_H</th>\n      <th>gender_V</th>\n      <th>deceased_N</th>\n      <th>deceased_S</th>\n      <th>cluster</th>\n    </tr>\n    <tr>\n      <th>pk_cid</th>\n      <th></th>\n      <th></th>\n      <th></th>\n      <th></th>\n      <th></th>\n      <th></th>\n      <th></th>\n      <th></th>\n      <th></th>\n      <th></th>\n      <th></th>\n      <th></th>\n      <th></th>\n      <th></th>\n      <th></th>\n      <th></th>\n      <th></th>\n      <th></th>\n      <th></th>\n      <th></th>\n      <th></th>\n    </tr>\n  </thead>\n  <tbody>\n    <tr>\n      <th>100296</th>\n      <td>1.0</td>\n      <td>40</td>\n      <td>40.0</td>\n      <td>20.0</td>\n      <td>28.0</td>\n      <td>46</td>\n      <td>4</td>\n      <td>0.0</td>\n      <td>1</td>\n      <td>0</td>\n      <td>...</td>\n      <td>0</td>\n      <td>0</td>\n      <td>0</td>\n      <td>0</td>\n      <td>0</td>\n      <td>0</td>\n      <td>1</td>\n      <td>1</td>\n      <td>0</td>\n      <td>0</td>\n    </tr>\n    <tr>\n      <th>100296</th>\n      <td>1.0</td>\n      <td>40</td>\n      <td>40.0</td>\n      <td>50.0</td>\n      <td>28.0</td>\n      <td>46</td>\n      <td>5</td>\n      <td>30.0</td>\n      <td>2</td>\n      <td>0</td>\n      <td>...</td>\n      <td>0</td>\n      <td>0</td>\n      <td>0</td>\n      <td>0</td>\n      <td>0</td>\n      <td>0</td>\n      <td>1</td>\n      <td>1</td>\n      <td>0</td>\n      <td>0</td>\n    </tr>\n    <tr>\n      <th>100296</th>\n      <td>1.0</td>\n      <td>40</td>\n      <td>40.0</td>\n      <td>81.0</td>\n      <td>28.0</td>\n      <td>46</td>\n      <td>6</td>\n      <td>31.0</td>\n      <td>3</td>\n      <td>0</td>\n      <td>...</td>\n      <td>0</td>\n      <td>0</td>\n      <td>0</td>\n      <td>0</td>\n      <td>0</td>\n      <td>0</td>\n      <td>1</td>\n      <td>1</td>\n      <td>0</td>\n      <td>0</td>\n    </tr>\n    <tr>\n      <th>1003705</th>\n      <td>1.0</td>\n      <td>40</td>\n      <td>40.0</td>\n      <td>1337.0</td>\n      <td>28.0</td>\n      <td>33</td>\n      <td>9</td>\n      <td>0.0</td>\n      <td>1</td>\n      <td>0</td>\n      <td>...</td>\n      <td>0</td>\n      <td>0</td>\n      <td>0</td>\n      <td>0</td>\n      <td>0</td>\n      <td>1</td>\n      <td>0</td>\n      <td>1</td>\n      <td>0</td>\n      <td>0</td>\n    </tr>\n    <tr>\n      <th>1003705</th>\n      <td>1.0</td>\n      <td>40</td>\n      <td>40.0</td>\n      <td>1367.0</td>\n      <td>28.0</td>\n      <td>33</td>\n      <td>10</td>\n      <td>30.0</td>\n      <td>2</td>\n      <td>0</td>\n      <td>...</td>\n      <td>0</td>\n      <td>0</td>\n      <td>0</td>\n      <td>0</td>\n      <td>0</td>\n      <td>1</td>\n      <td>0</td>\n      <td>1</td>\n      <td>0</td>\n      <td>0</td>\n    </tr>\n    <tr>\n      <th>1003705</th>\n      <td>1.0</td>\n      <td>40</td>\n      <td>40.0</td>\n      <td>1398.0</td>\n      <td>28.0</td>\n      <td>33</td>\n      <td>11</td>\n      <td>31.0</td>\n      <td>3</td>\n      <td>0</td>\n      <td>...</td>\n      <td>0</td>\n      <td>0</td>\n      <td>0</td>\n      <td>0</td>\n      <td>0</td>\n      <td>1</td>\n      <td>0</td>\n      <td>1</td>\n      <td>0</td>\n      <td>0</td>\n    </tr>\n    <tr>\n      <th>1004643</th>\n      <td>1.0</td>\n      <td>40</td>\n      <td>40.0</td>\n      <td>1115.0</td>\n      <td>3.0</td>\n      <td>30</td>\n      <td>2</td>\n      <td>0.0</td>\n      <td>1</td>\n      <td>0</td>\n      <td>...</td>\n      <td>0</td>\n      <td>0</td>\n      <td>0</td>\n      <td>0</td>\n      <td>0</td>\n      <td>0</td>\n      <td>1</td>\n      <td>1</td>\n      <td>0</td>\n      <td>0</td>\n    </tr>\n  </tbody>\n</table>\n<p>7 rows × 128 columns</p>\n</div>"
                    },
                    "metadata": {},
                    "execution_count": 120
                }
            ],
            "source": [
                "full_4[\"cluster\"] = labels\n",
                "full_4.head(7)"
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
            "execution_count": 122,
            "metadata": {},
            "outputs": [
                {
                    "output_type": "execute_result",
                    "data": {
                        "text/plain": [
                            "Index(['Ventas_cant', 'Precios', 'Ingresos', 'días_encartera', 'region_code',\n",
                            "       'age', 'Month', 'days_between', 'recurrencia', 'Productos_credit_card',\n",
                            "       ...\n",
                            "       'country_id_SA', 'country_id_SE', 'country_id_SN', 'country_id_US',\n",
                            "       'country_id_VE', 'gender_H', 'gender_V', 'deceased_N', 'deceased_S',\n",
                            "       'cluster'],\n",
                            "      dtype='object', length=128)"
                        ]
                    },
                    "metadata": {},
                    "execution_count": 122
                }
            ],
            "source": [
                "full_4.columns"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": 123,
            "metadata": {},
            "outputs": [
                {
                    "output_type": "stream",
                    "name": "stdout",
                    "text": [
                        "Error in callback <function flush_figures at 0x0000024E11712EE0> (for post_execute):\n"
                    ]
                },
                {
                    "output_type": "error",
                    "ename": "KeyboardInterrupt",
                    "evalue": "",
                    "traceback": [
                        "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
                        "\u001b[1;31mKeyboardInterrupt\u001b[0m                         Traceback (most recent call last)",
                        "\u001b[1;32m~\\anaconda3\\lib\\site-packages\\ipykernel\\pylab\\backend_inline.py\u001b[0m in \u001b[0;36mflush_figures\u001b[1;34m()\u001b[0m\n\u001b[0;32m    119\u001b[0m         \u001b[1;31m# ignore the tracking, just draw and close all figures\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    120\u001b[0m         \u001b[1;32mtry\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m--> 121\u001b[1;33m             \u001b[1;32mreturn\u001b[0m \u001b[0mshow\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;32mTrue\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m    122\u001b[0m         \u001b[1;32mexcept\u001b[0m \u001b[0mException\u001b[0m \u001b[1;32mas\u001b[0m \u001b[0me\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    123\u001b[0m             \u001b[1;31m# safely show traceback if in IPython, else raise\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
                        "\u001b[1;32m~\\anaconda3\\lib\\site-packages\\ipykernel\\pylab\\backend_inline.py\u001b[0m in \u001b[0;36mshow\u001b[1;34m(close, block)\u001b[0m\n\u001b[0;32m     39\u001b[0m     \u001b[1;32mtry\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m     40\u001b[0m         \u001b[1;32mfor\u001b[0m \u001b[0mfigure_manager\u001b[0m \u001b[1;32min\u001b[0m \u001b[0mGcf\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mget_all_fig_managers\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m---> 41\u001b[1;33m             display(\n\u001b[0m\u001b[0;32m     42\u001b[0m                 \u001b[0mfigure_manager\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mcanvas\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mfigure\u001b[0m\u001b[1;33m,\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m     43\u001b[0m                 \u001b[0mmetadata\u001b[0m\u001b[1;33m=\u001b[0m\u001b[0m_fetch_figure_metadata\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mfigure_manager\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mcanvas\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mfigure\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
                        "\u001b[1;32m~\\anaconda3\\lib\\site-packages\\IPython\\core\\display.py\u001b[0m in \u001b[0;36mdisplay\u001b[1;34m(include, exclude, metadata, transient, display_id, *objs, **kwargs)\u001b[0m\n\u001b[0;32m    311\u001b[0m             \u001b[0mpublish_display_data\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mdata\u001b[0m\u001b[1;33m=\u001b[0m\u001b[0mobj\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mmetadata\u001b[0m\u001b[1;33m=\u001b[0m\u001b[0mmetadata\u001b[0m\u001b[1;33m,\u001b[0m \u001b[1;33m**\u001b[0m\u001b[0mkwargs\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    312\u001b[0m         \u001b[1;32melse\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m--> 313\u001b[1;33m             \u001b[0mformat_dict\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mmd_dict\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mformat\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mobj\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0minclude\u001b[0m\u001b[1;33m=\u001b[0m\u001b[0minclude\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mexclude\u001b[0m\u001b[1;33m=\u001b[0m\u001b[0mexclude\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m    314\u001b[0m             \u001b[1;32mif\u001b[0m \u001b[1;32mnot\u001b[0m \u001b[0mformat_dict\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    315\u001b[0m                 \u001b[1;31m# nothing to display (e.g. _ipython_display_ took over)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
                        "\u001b[1;32m~\\anaconda3\\lib\\site-packages\\IPython\\core\\formatters.py\u001b[0m in \u001b[0;36mformat\u001b[1;34m(self, obj, include, exclude)\u001b[0m\n\u001b[0;32m    178\u001b[0m             \u001b[0mmd\u001b[0m \u001b[1;33m=\u001b[0m \u001b[1;32mNone\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    179\u001b[0m             \u001b[1;32mtry\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m--> 180\u001b[1;33m                 \u001b[0mdata\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mformatter\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mobj\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m    181\u001b[0m             \u001b[1;32mexcept\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    182\u001b[0m                 \u001b[1;31m# FIXME: log the exception\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
                        "\u001b[1;32m<decorator-gen-2>\u001b[0m in \u001b[0;36m__call__\u001b[1;34m(self, obj)\u001b[0m\n",
                        "\u001b[1;32m~\\anaconda3\\lib\\site-packages\\IPython\\core\\formatters.py\u001b[0m in \u001b[0;36mcatch_format_error\u001b[1;34m(method, self, *args, **kwargs)\u001b[0m\n\u001b[0;32m    222\u001b[0m     \u001b[1;34m\"\"\"show traceback on failed format call\"\"\"\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    223\u001b[0m     \u001b[1;32mtry\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m--> 224\u001b[1;33m         \u001b[0mr\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mmethod\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mself\u001b[0m\u001b[1;33m,\u001b[0m \u001b[1;33m*\u001b[0m\u001b[0margs\u001b[0m\u001b[1;33m,\u001b[0m \u001b[1;33m**\u001b[0m\u001b[0mkwargs\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m    225\u001b[0m     \u001b[1;32mexcept\u001b[0m \u001b[0mNotImplementedError\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    226\u001b[0m         \u001b[1;31m# don't warn on NotImplementedErrors\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
                        "\u001b[1;32m~\\anaconda3\\lib\\site-packages\\IPython\\core\\formatters.py\u001b[0m in \u001b[0;36m__call__\u001b[1;34m(self, obj)\u001b[0m\n\u001b[0;32m    339\u001b[0m                 \u001b[1;32mpass\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    340\u001b[0m             \u001b[1;32melse\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m--> 341\u001b[1;33m                 \u001b[1;32mreturn\u001b[0m \u001b[0mprinter\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mobj\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m    342\u001b[0m             \u001b[1;31m# Finally look for special method names\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    343\u001b[0m             \u001b[0mmethod\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mget_real_method\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mobj\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mprint_method\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
                        "\u001b[1;32m~\\anaconda3\\lib\\site-packages\\IPython\\core\\pylabtools.py\u001b[0m in \u001b[0;36m<lambda>\u001b[1;34m(fig)\u001b[0m\n\u001b[0;32m    252\u001b[0m         \u001b[0mjpg_formatter\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mfor_type\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mFigure\u001b[0m\u001b[1;33m,\u001b[0m \u001b[1;32mlambda\u001b[0m \u001b[0mfig\u001b[0m\u001b[1;33m:\u001b[0m \u001b[0mprint_figure\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mfig\u001b[0m\u001b[1;33m,\u001b[0m \u001b[1;34m'jpg'\u001b[0m\u001b[1;33m,\u001b[0m \u001b[1;33m**\u001b[0m\u001b[0mkwargs\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    253\u001b[0m     \u001b[1;32mif\u001b[0m \u001b[1;34m'svg'\u001b[0m \u001b[1;32min\u001b[0m \u001b[0mformats\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m--> 254\u001b[1;33m         \u001b[0msvg_formatter\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mfor_type\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mFigure\u001b[0m\u001b[1;33m,\u001b[0m \u001b[1;32mlambda\u001b[0m \u001b[0mfig\u001b[0m\u001b[1;33m:\u001b[0m \u001b[0mprint_figure\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mfig\u001b[0m\u001b[1;33m,\u001b[0m \u001b[1;34m'svg'\u001b[0m\u001b[1;33m,\u001b[0m \u001b[1;33m**\u001b[0m\u001b[0mkwargs\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m    255\u001b[0m     \u001b[1;32mif\u001b[0m \u001b[1;34m'pdf'\u001b[0m \u001b[1;32min\u001b[0m \u001b[0mformats\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    256\u001b[0m         \u001b[0mpdf_formatter\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mfor_type\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mFigure\u001b[0m\u001b[1;33m,\u001b[0m \u001b[1;32mlambda\u001b[0m \u001b[0mfig\u001b[0m\u001b[1;33m:\u001b[0m \u001b[0mprint_figure\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mfig\u001b[0m\u001b[1;33m,\u001b[0m \u001b[1;34m'pdf'\u001b[0m\u001b[1;33m,\u001b[0m \u001b[1;33m**\u001b[0m\u001b[0mkwargs\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
                        "\u001b[1;32m~\\anaconda3\\lib\\site-packages\\IPython\\core\\pylabtools.py\u001b[0m in \u001b[0;36mprint_figure\u001b[1;34m(fig, fmt, bbox_inches, **kwargs)\u001b[0m\n\u001b[0;32m    130\u001b[0m         \u001b[0mFigureCanvasBase\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mfig\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    131\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m--> 132\u001b[1;33m     \u001b[0mfig\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mcanvas\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mprint_figure\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mbytes_io\u001b[0m\u001b[1;33m,\u001b[0m \u001b[1;33m**\u001b[0m\u001b[0mkw\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m    133\u001b[0m     \u001b[0mdata\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mbytes_io\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mgetvalue\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    134\u001b[0m     \u001b[1;32mif\u001b[0m \u001b[0mfmt\u001b[0m \u001b[1;33m==\u001b[0m \u001b[1;34m'svg'\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
                        "\u001b[1;32m~\\anaconda3\\lib\\site-packages\\matplotlib\\backend_bases.py\u001b[0m in \u001b[0;36mprint_figure\u001b[1;34m(self, filename, dpi, facecolor, edgecolor, orientation, format, bbox_inches, pad_inches, bbox_extra_artists, backend, **kwargs)\u001b[0m\n\u001b[0;32m   2208\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m   2209\u001b[0m             \u001b[1;32mtry\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m-> 2210\u001b[1;33m                 result = print_method(\n\u001b[0m\u001b[0;32m   2211\u001b[0m                     \u001b[0mfilename\u001b[0m\u001b[1;33m,\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m   2212\u001b[0m                     \u001b[0mdpi\u001b[0m\u001b[1;33m=\u001b[0m\u001b[0mdpi\u001b[0m\u001b[1;33m,\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
                        "\u001b[1;32m~\\anaconda3\\lib\\site-packages\\matplotlib\\backends\\backend_svg.py\u001b[0m in \u001b[0;36mprint_svg\u001b[1;34m(self, filename, *args, **kwargs)\u001b[0m\n\u001b[0;32m   1327\u001b[0m                 \u001b[0mdetach\u001b[0m \u001b[1;33m=\u001b[0m \u001b[1;32mTrue\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m   1328\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m-> 1329\u001b[1;33m             \u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0m_print_svg\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mfilename\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mfh\u001b[0m\u001b[1;33m,\u001b[0m \u001b[1;33m**\u001b[0m\u001b[0mkwargs\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m   1330\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m   1331\u001b[0m             \u001b[1;31m# Detach underlying stream from wrapper so that it remains open in\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
                        "\u001b[1;32m~\\anaconda3\\lib\\site-packages\\matplotlib\\backend_bases.py\u001b[0m in \u001b[0;36mwrapper\u001b[1;34m(*args, **kwargs)\u001b[0m\n\u001b[0;32m   1637\u001b[0m             \u001b[0mkwargs\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mpop\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0marg\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m   1638\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m-> 1639\u001b[1;33m         \u001b[1;32mreturn\u001b[0m \u001b[0mfunc\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;33m*\u001b[0m\u001b[0margs\u001b[0m\u001b[1;33m,\u001b[0m \u001b[1;33m**\u001b[0m\u001b[0mkwargs\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m   1640\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m   1641\u001b[0m     \u001b[1;32mreturn\u001b[0m \u001b[0mwrapper\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
                        "\u001b[1;32m~\\anaconda3\\lib\\site-packages\\matplotlib\\backends\\backend_svg.py\u001b[0m in \u001b[0;36m_print_svg\u001b[1;34m(self, filename, fh, dpi, bbox_inches_restore, metadata)\u001b[0m\n\u001b[0;32m   1351\u001b[0m             bbox_inches_restore=bbox_inches_restore)\n\u001b[0;32m   1352\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m-> 1353\u001b[1;33m         \u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mfigure\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mdraw\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mrenderer\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m   1354\u001b[0m         \u001b[0mrenderer\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mfinalize\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m   1355\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n",
                        "\u001b[1;32m~\\anaconda3\\lib\\site-packages\\matplotlib\\artist.py\u001b[0m in \u001b[0;36mdraw_wrapper\u001b[1;34m(artist, renderer, *args, **kwargs)\u001b[0m\n\u001b[0;32m     39\u001b[0m                 \u001b[0mrenderer\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mstart_filter\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m     40\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m---> 41\u001b[1;33m             \u001b[1;32mreturn\u001b[0m \u001b[0mdraw\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0martist\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mrenderer\u001b[0m\u001b[1;33m,\u001b[0m \u001b[1;33m*\u001b[0m\u001b[0margs\u001b[0m\u001b[1;33m,\u001b[0m \u001b[1;33m**\u001b[0m\u001b[0mkwargs\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m     42\u001b[0m         \u001b[1;32mfinally\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m     43\u001b[0m             \u001b[1;32mif\u001b[0m \u001b[0martist\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mget_agg_filter\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;33m)\u001b[0m \u001b[1;32mis\u001b[0m \u001b[1;32mnot\u001b[0m \u001b[1;32mNone\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
                        "\u001b[1;32m~\\anaconda3\\lib\\site-packages\\matplotlib\\figure.py\u001b[0m in \u001b[0;36mdraw\u001b[1;34m(self, renderer)\u001b[0m\n\u001b[0;32m   1861\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m   1862\u001b[0m             \u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mpatch\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mdraw\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mrenderer\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m-> 1863\u001b[1;33m             mimage._draw_list_compositing_images(\n\u001b[0m\u001b[0;32m   1864\u001b[0m                 renderer, self, artists, self.suppressComposite)\n\u001b[0;32m   1865\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n",
                        "\u001b[1;32m~\\anaconda3\\lib\\site-packages\\matplotlib\\image.py\u001b[0m in \u001b[0;36m_draw_list_compositing_images\u001b[1;34m(renderer, parent, artists, suppress_composite)\u001b[0m\n\u001b[0;32m    129\u001b[0m     \u001b[1;32mif\u001b[0m \u001b[0mnot_composite\u001b[0m \u001b[1;32mor\u001b[0m \u001b[1;32mnot\u001b[0m \u001b[0mhas_images\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    130\u001b[0m         \u001b[1;32mfor\u001b[0m \u001b[0ma\u001b[0m \u001b[1;32min\u001b[0m \u001b[0martists\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m--> 131\u001b[1;33m             \u001b[0ma\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mdraw\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mrenderer\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m    132\u001b[0m     \u001b[1;32melse\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    133\u001b[0m         \u001b[1;31m# Composite any adjacent images together\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
                        "\u001b[1;32m~\\anaconda3\\lib\\site-packages\\matplotlib\\artist.py\u001b[0m in \u001b[0;36mdraw_wrapper\u001b[1;34m(artist, renderer, *args, **kwargs)\u001b[0m\n\u001b[0;32m     39\u001b[0m                 \u001b[0mrenderer\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mstart_filter\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m     40\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m---> 41\u001b[1;33m             \u001b[1;32mreturn\u001b[0m \u001b[0mdraw\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0martist\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mrenderer\u001b[0m\u001b[1;33m,\u001b[0m \u001b[1;33m*\u001b[0m\u001b[0margs\u001b[0m\u001b[1;33m,\u001b[0m \u001b[1;33m**\u001b[0m\u001b[0mkwargs\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m     42\u001b[0m         \u001b[1;32mfinally\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m     43\u001b[0m             \u001b[1;32mif\u001b[0m \u001b[0martist\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mget_agg_filter\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;33m)\u001b[0m \u001b[1;32mis\u001b[0m \u001b[1;32mnot\u001b[0m \u001b[1;32mNone\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
                        "\u001b[1;32m~\\anaconda3\\lib\\site-packages\\matplotlib\\cbook\\deprecation.py\u001b[0m in \u001b[0;36mwrapper\u001b[1;34m(*inner_args, **inner_kwargs)\u001b[0m\n\u001b[0;32m    409\u001b[0m                          \u001b[1;32melse\u001b[0m \u001b[0mdeprecation_addendum\u001b[0m\u001b[1;33m,\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    410\u001b[0m                 **kwargs)\n\u001b[1;32m--> 411\u001b[1;33m         \u001b[1;32mreturn\u001b[0m \u001b[0mfunc\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;33m*\u001b[0m\u001b[0minner_args\u001b[0m\u001b[1;33m,\u001b[0m \u001b[1;33m**\u001b[0m\u001b[0minner_kwargs\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m    412\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    413\u001b[0m     \u001b[1;32mreturn\u001b[0m \u001b[0mwrapper\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
                        "\u001b[1;32m~\\anaconda3\\lib\\site-packages\\matplotlib\\axes\\_base.py\u001b[0m in \u001b[0;36mdraw\u001b[1;34m(self, renderer, inframe)\u001b[0m\n\u001b[0;32m   2745\u001b[0m             \u001b[0mrenderer\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mstop_rasterizing\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m   2746\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m-> 2747\u001b[1;33m         \u001b[0mmimage\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0m_draw_list_compositing_images\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mrenderer\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mself\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0martists\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m   2748\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m   2749\u001b[0m         \u001b[0mrenderer\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mclose_group\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;34m'axes'\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
                        "\u001b[1;32m~\\anaconda3\\lib\\site-packages\\matplotlib\\image.py\u001b[0m in \u001b[0;36m_draw_list_compositing_images\u001b[1;34m(renderer, parent, artists, suppress_composite)\u001b[0m\n\u001b[0;32m    129\u001b[0m     \u001b[1;32mif\u001b[0m \u001b[0mnot_composite\u001b[0m \u001b[1;32mor\u001b[0m \u001b[1;32mnot\u001b[0m \u001b[0mhas_images\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    130\u001b[0m         \u001b[1;32mfor\u001b[0m \u001b[0ma\u001b[0m \u001b[1;32min\u001b[0m \u001b[0martists\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m--> 131\u001b[1;33m             \u001b[0ma\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mdraw\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mrenderer\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m    132\u001b[0m     \u001b[1;32melse\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    133\u001b[0m         \u001b[1;31m# Composite any adjacent images together\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
                        "\u001b[1;32m~\\anaconda3\\lib\\site-packages\\matplotlib\\artist.py\u001b[0m in \u001b[0;36mdraw_wrapper\u001b[1;34m(artist, renderer, *args, **kwargs)\u001b[0m\n\u001b[0;32m     39\u001b[0m                 \u001b[0mrenderer\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mstart_filter\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m     40\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m---> 41\u001b[1;33m             \u001b[1;32mreturn\u001b[0m \u001b[0mdraw\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0martist\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mrenderer\u001b[0m\u001b[1;33m,\u001b[0m \u001b[1;33m*\u001b[0m\u001b[0margs\u001b[0m\u001b[1;33m,\u001b[0m \u001b[1;33m**\u001b[0m\u001b[0mkwargs\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m     42\u001b[0m         \u001b[1;32mfinally\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m     43\u001b[0m             \u001b[1;32mif\u001b[0m \u001b[0martist\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mget_agg_filter\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;33m)\u001b[0m \u001b[1;32mis\u001b[0m \u001b[1;32mnot\u001b[0m \u001b[1;32mNone\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
                        "\u001b[1;32m~\\anaconda3\\lib\\site-packages\\matplotlib\\collections.py\u001b[0m in \u001b[0;36mdraw\u001b[1;34m(self, renderer)\u001b[0m\n\u001b[0;32m    929\u001b[0m     \u001b[1;32mdef\u001b[0m \u001b[0mdraw\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mself\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mrenderer\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    930\u001b[0m         \u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mset_sizes\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0m_sizes\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mfigure\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mdpi\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m--> 931\u001b[1;33m         \u001b[0mCollection\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mdraw\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mself\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mrenderer\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m    932\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    933\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n",
                        "\u001b[1;32m~\\anaconda3\\lib\\site-packages\\matplotlib\\artist.py\u001b[0m in \u001b[0;36mdraw_wrapper\u001b[1;34m(artist, renderer, *args, **kwargs)\u001b[0m\n\u001b[0;32m     39\u001b[0m                 \u001b[0mrenderer\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mstart_filter\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m     40\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m---> 41\u001b[1;33m             \u001b[1;32mreturn\u001b[0m \u001b[0mdraw\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0martist\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mrenderer\u001b[0m\u001b[1;33m,\u001b[0m \u001b[1;33m*\u001b[0m\u001b[0margs\u001b[0m\u001b[1;33m,\u001b[0m \u001b[1;33m**\u001b[0m\u001b[0mkwargs\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m     42\u001b[0m         \u001b[1;32mfinally\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m     43\u001b[0m             \u001b[1;32mif\u001b[0m \u001b[0martist\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mget_agg_filter\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;33m)\u001b[0m \u001b[1;32mis\u001b[0m \u001b[1;32mnot\u001b[0m \u001b[1;32mNone\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
                        "\u001b[1;32m~\\anaconda3\\lib\\site-packages\\matplotlib\\collections.py\u001b[0m in \u001b[0;36mdraw\u001b[1;34m(self, renderer)\u001b[0m\n\u001b[0;32m    404\u001b[0m                 mpath.Path(offsets), transOffset, tuple(facecolors[0]))\n\u001b[0;32m    405\u001b[0m         \u001b[1;32melse\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m--> 406\u001b[1;33m             renderer.draw_path_collection(\n\u001b[0m\u001b[0;32m    407\u001b[0m                 \u001b[0mgc\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mtransform\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mfrozen\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mpaths\u001b[0m\u001b[1;33m,\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    408\u001b[0m                 \u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mget_transforms\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0moffsets\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mtransOffset\u001b[0m\u001b[1;33m,\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
                        "\u001b[1;32m~\\anaconda3\\lib\\site-packages\\matplotlib\\backends\\backend_svg.py\u001b[0m in \u001b[0;36mdraw_path_collection\u001b[1;34m(self, gc, master_transform, paths, all_transforms, offsets, offsetTrans, facecolors, edgecolors, linewidths, linestyles, antialiaseds, urls, offset_position)\u001b[0m\n\u001b[0;32m    765\u001b[0m                 \u001b[1;34m'x'\u001b[0m\u001b[1;33m:\u001b[0m \u001b[0mshort_float_fmt\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mxo\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m,\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    766\u001b[0m                 \u001b[1;34m'y'\u001b[0m\u001b[1;33m:\u001b[0m \u001b[0mshort_float_fmt\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mheight\u001b[0m \u001b[1;33m-\u001b[0m \u001b[0myo\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m,\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m--> 767\u001b[1;33m                 \u001b[1;34m'style'\u001b[0m\u001b[1;33m:\u001b[0m \u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0m_get_style\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mgc0\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mrgbFace\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m    768\u001b[0m                 }\n\u001b[0;32m    769\u001b[0m             \u001b[0mwriter\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0melement\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;34m'use'\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mattrib\u001b[0m\u001b[1;33m=\u001b[0m\u001b[0mattrib\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
                        "\u001b[1;32m~\\anaconda3\\lib\\site-packages\\matplotlib\\backends\\backend_svg.py\u001b[0m in \u001b[0;36m_get_style\u001b[1;34m(self, gc, rgbFace)\u001b[0m\n\u001b[0;32m    576\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    577\u001b[0m     \u001b[1;32mdef\u001b[0m \u001b[0m_get_style\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mself\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mgc\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mrgbFace\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m--> 578\u001b[1;33m         \u001b[1;32mreturn\u001b[0m \u001b[0mgenerate_css\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0m_get_style_dict\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mgc\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mrgbFace\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m    579\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    580\u001b[0m     \u001b[1;32mdef\u001b[0m \u001b[0m_get_clip\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mself\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mgc\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
                        "\u001b[1;32m~\\anaconda3\\lib\\site-packages\\matplotlib\\backends\\backend_svg.py\u001b[0m in \u001b[0;36m_get_style_dict\u001b[1;34m(self, gc, rgbFace)\u001b[0m\n\u001b[0;32m    554\u001b[0m             \u001b[0mattrib\u001b[0m\u001b[1;33m[\u001b[0m\u001b[1;34m'opacity'\u001b[0m\u001b[1;33m]\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mshort_float_fmt\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mgc\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mget_alpha\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    555\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m--> 556\u001b[1;33m         \u001b[0moffset\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mseq\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mgc\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mget_dashes\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m    557\u001b[0m         \u001b[1;32mif\u001b[0m \u001b[0mseq\u001b[0m \u001b[1;32mis\u001b[0m \u001b[1;32mnot\u001b[0m \u001b[1;32mNone\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    558\u001b[0m             attrib['stroke-dasharray'] = ','.join(\n",
                        "\u001b[1;31mKeyboardInterrupt\u001b[0m: "
                    ]
                }
            ],
            "source": [
                "# visualizamos nuestros grupos en base a las variables del modelo RFM, para ver que tal han quedado.\n",
                "selected_columns = ['Ingresos', 'días_encartera', 'age']\n",
                "\n",
                "sns.pairplot(full_4, vars = selected_columns, hue = 'cluster');"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": null,
            "metadata": {},
            "outputs": [],
            "source": [
                "## Recomendación \"user based\""
            ]
        },
        {
            "cell_type": "code",
            "execution_count": 83,
            "metadata": {},
            "outputs": [],
            "source": [
                "def top_users(user, full_4):\n",
                "    '''\n",
                "    This function prints the top 10 similar users based on cosine similarity.\n",
                "    '''\n",
                "    \n",
                "    if user not in full_4.columns:\n",
                "        return('No data available on user {}'.format(user))\n",
                "    \n",
                "    print('Most Similar Users:\\n')\n",
                "    \n",
                "    sim_users = full_4.sort_values(by = user, ascending=False).index[1:11]\n",
                "    sim_values = full_4.sort_values(by = user, ascending=False).loc[:,user].tolist()[1:11]\n",
                "    \n",
                "    for user, sim in zip(sim_users, sim_values):\n",
                "        print('User #{0}, Similarity value: {1:.2f}'.format(user, sim)) "
            ]
        },
        {
            "cell_type": "code",
            "execution_count": 84,
            "metadata": {},
            "outputs": [],
            "source": [
                "def compare_2_users(user1, user2, full_4, nr_animes):\n",
                "    '''\n",
                "    Returns a DataFrame with top 10 animes by 2 similar users (based on cosine similarity).\n",
                "    '''\n",
                "\n",
                "    top_10_user_1 = full_4[full_4.index == user1].melt().sort_values(\"value\", ascending = False)[:nr_animes]\n",
                "    top_10_user_1.columns = [\"name_user_{}\".format(user1), \"rating_user_{}\".format(user1)]\n",
                "    top_10_user_1 = top_10_user_1.reset_index(drop = True)\n",
                "\n",
                "    top_10_user_2 = full_4[full_4.index == user2].melt().sort_values(\"value\", ascending = False)[:nr_animes]\n",
                "    top_10_user_2.columns = [\"name_user_{}\".format(user2), \"rating_user_{}\".format(user2)]\n",
                "    top_10_user_2 = top_10_user_2.reset_index(drop = True)\n",
                "\n",
                "    combined_2_users = pd.concat([top_10_user_1, top_10_user_2], axis = 1, join = \"inner\")\n",
                "    \n",
                "    return combined_2_users"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": 86,
            "metadata": {},
            "outputs": [
                {
                    "output_type": "error",
                    "ename": "NameError",
                    "evalue": "name 'user_sim_df' is not defined",
                    "traceback": [
                        "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
                        "\u001b[1;31mNameError\u001b[0m                                 Traceback (most recent call last)",
                        "\u001b[1;32m<ipython-input-86-fc9960555ac4>\u001b[0m in \u001b[0;36m<module>\u001b[1;34m\u001b[0m\n\u001b[0;32m      3\u001b[0m \u001b[0muser2\u001b[0m \u001b[1;33m=\u001b[0m \u001b[1;36m2390\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m      4\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m----> 5\u001b[1;33m \u001b[0mtop_users\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0muser1\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0muser_sim_df\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m      6\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m      7\u001b[0m \u001b[0mcombined_2_users\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mcompare_2_users\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0muser1\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0muser2\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mfull_4\u001b[0m\u001b[1;33m,\u001b[0m \u001b[1;36m10\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
                        "\u001b[1;31mNameError\u001b[0m: name 'user_sim_df' is not defined"
                    ]
                }
            ],
            "source": [
                "user1 = 20\n",
                "\n",
                "user2 = 2390\n",
                "\n",
                "top_users(user1, user_sim_df)\n",
                "\n",
                "combined_2_users = compare_2_users(user1, user2, full_4, 10)\n",
                "combined_2_users"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": null,
            "metadata": {},
            "outputs": [],
            "source": []
        }
    ]
}