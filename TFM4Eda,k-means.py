{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 5,
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
   "execution_count": 6,
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
   "execution_count": 7,
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
   "cell_type": "markdown",
   "source": [
    "## EDA"
   ],
   "metadata": {}
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "source": [
    "full_3 = pd.read_pickle('./full_3.pickle')"
   ],
   "outputs": [],
   "metadata": {
    "azdata_cell_guid": "12400a40-ad75-4e6a-9b1b-b3f585c9b8d3"
   }
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "source": [
    "full_3.to_csv(\"full_3.csv\", index = False)"
   ],
   "outputs": [],
   "metadata": {}
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "source": [
    "Full_clientesnuevos = full_3[full_3['recurrencia']==1]"
   ],
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
   "execution_count": 14,
   "source": [
    "full_3[full_3['recurrencia']==1]['Ingresos'].sum()"
   ],
   "outputs": [
    {
     "output_type": "execute_result",
     "data": {
      "text/plain": [
       "4369300.0"
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
    "full_3[full_3['recurrencia']>1]['Ingresos'].sum()"
   ],
   "outputs": [
    {
     "output_type": "execute_result",
     "data": {
      "text/plain": [
       "73041420.0"
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
    "(full_3[full_3['recurrencia']==1]['Ingresos'].sum())/(full_3[full_3['recurrencia']>1]['Ingresos'].sum())"
   ],
   "outputs": [
    {
     "output_type": "execute_result",
     "data": {
      "text/plain": [
       "0.059819483246629106"
      ]
     },
     "metadata": {},
     "execution_count": 16
    }
   ],
   "metadata": {}
  },
  {
   "cell_type": "code",
   "execution_count": 32,
   "source": [
    "(full_3[full_3['recurrencia']>1]['Ingresos'].sum())/(full_3[full_3['recurrencia']==1]['Ingresos'].sum())"
   ],
   "outputs": [
    {
     "output_type": "execute_result",
     "data": {
      "text/plain": [
       "16.716961527018057"
      ]
     },
     "metadata": {},
     "execution_count": 32
    }
   ],
   "metadata": {}
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "source": [
    "Prod_recurrencia_nuevos = Full_clientesnuevos.groupby(['Productos'])[\"Ingresos\"].sum().reset_index()\r\n",
    "Prod_recurrencia_nuevos"
   ],
   "outputs": [
    {
     "output_type": "execute_result",
     "data": {
      "text/plain": [
       "             Productos   Ingresos\n",
       "0          credit_card   428820.0\n",
       "1           debit_card   346190.0\n",
       "2            em_acount  2517870.0\n",
       "3          emc_account   147860.0\n",
       "4                funds    62320.0\n",
       "5                loans     2640.0\n",
       "6    long_term_deposit   284440.0\n",
       "7             mortgage     1500.0\n",
       "8              payroll   218040.0\n",
       "9      payroll_account    33500.0\n",
       "10        pension_plan    40840.0\n",
       "11          securities    72320.0\n",
       "12  short_term_deposit   212960.0"
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
       "      <th>Productos</th>\n",
       "      <th>Ingresos</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>credit_card</td>\n",
       "      <td>428820.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>debit_card</td>\n",
       "      <td>346190.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>em_acount</td>\n",
       "      <td>2517870.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>emc_account</td>\n",
       "      <td>147860.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>funds</td>\n",
       "      <td>62320.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>5</th>\n",
       "      <td>loans</td>\n",
       "      <td>2640.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>6</th>\n",
       "      <td>long_term_deposit</td>\n",
       "      <td>284440.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>7</th>\n",
       "      <td>mortgage</td>\n",
       "      <td>1500.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>8</th>\n",
       "      <td>payroll</td>\n",
       "      <td>218040.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>9</th>\n",
       "      <td>payroll_account</td>\n",
       "      <td>33500.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>10</th>\n",
       "      <td>pension_plan</td>\n",
       "      <td>40840.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>11</th>\n",
       "      <td>securities</td>\n",
       "      <td>72320.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>12</th>\n",
       "      <td>short_term_deposit</td>\n",
       "      <td>212960.0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ]
     },
     "metadata": {},
     "execution_count": 31
    }
   ],
   "metadata": {}
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "source": [
    "Prod_recurrencia_nuevos = Full_clientesnuevos.groupby(['Productos'])[\"Ingresos\"].sum().reset_index()\r\n",
    "Prod_recurrencia_nuevos\r\n",
    "Prod_recurrencia_nuevos = px.bar(Prod_recurrencia_nuevos, x=\"Productos\", y=\"Ingresos\", \\\r\n",
    "                        color=\"Productos\", orientation=\"v\", color_discrete_sequence=px.colors.cyclical.mygbm)\r\n",
    "\r\n",
    "Prod_recurrencia_nuevos.show()\r\n"
   ],
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
         "hovertemplate": "Productos=%{x}<br>Ingresos=%{y}<extra></extra>",
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
          "credit_card"
         ],
         "xaxis": "x",
         "y": [
          428820
         ],
         "yaxis": "y"
        },
        {
         "alignmentgroup": "True",
         "hovertemplate": "Productos=%{x}<br>Ingresos=%{y}<extra></extra>",
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
          "debit_card"
         ],
         "xaxis": "x",
         "y": [
          346190
         ],
         "yaxis": "y"
        },
        {
         "alignmentgroup": "True",
         "hovertemplate": "Productos=%{x}<br>Ingresos=%{y}<extra></extra>",
         "legendgroup": "em_acount",
         "marker": {
          "color": "#fbafa1"
         },
         "name": "em_acount",
         "offsetgroup": "em_acount",
         "orientation": "v",
         "showlegend": true,
         "textposition": "auto",
         "type": "bar",
         "x": [
          "em_acount"
         ],
         "xaxis": "x",
         "y": [
          2517870
         ],
         "yaxis": "y"
        },
        {
         "alignmentgroup": "True",
         "hovertemplate": "Productos=%{x}<br>Ingresos=%{y}<extra></extra>",
         "legendgroup": "emc_account",
         "marker": {
          "color": "#fcd471"
         },
         "name": "emc_account",
         "offsetgroup": "emc_account",
         "orientation": "v",
         "showlegend": true,
         "textposition": "auto",
         "type": "bar",
         "x": [
          "emc_account"
         ],
         "xaxis": "x",
         "y": [
          147860
         ],
         "yaxis": "y"
        },
        {
         "alignmentgroup": "True",
         "hovertemplate": "Productos=%{x}<br>Ingresos=%{y}<extra></extra>",
         "legendgroup": "funds",
         "marker": {
          "color": "#f0ed35"
         },
         "name": "funds",
         "offsetgroup": "funds",
         "orientation": "v",
         "showlegend": true,
         "textposition": "auto",
         "type": "bar",
         "x": [
          "funds"
         ],
         "xaxis": "x",
         "y": [
          62320
         ],
         "yaxis": "y"
        },
        {
         "alignmentgroup": "True",
         "hovertemplate": "Productos=%{x}<br>Ingresos=%{y}<extra></extra>",
         "legendgroup": "loans",
         "marker": {
          "color": "#c6e516"
         },
         "name": "loans",
         "offsetgroup": "loans",
         "orientation": "v",
         "showlegend": true,
         "textposition": "auto",
         "type": "bar",
         "x": [
          "loans"
         ],
         "xaxis": "x",
         "y": [
          2640
         ],
         "yaxis": "y"
        },
        {
         "alignmentgroup": "True",
         "hovertemplate": "Productos=%{x}<br>Ingresos=%{y}<extra></extra>",
         "legendgroup": "long_term_deposit",
         "marker": {
          "color": "#96d310"
         },
         "name": "long_term_deposit",
         "offsetgroup": "long_term_deposit",
         "orientation": "v",
         "showlegend": true,
         "textposition": "auto",
         "type": "bar",
         "x": [
          "long_term_deposit"
         ],
         "xaxis": "x",
         "y": [
          284440
         ],
         "yaxis": "y"
        },
        {
         "alignmentgroup": "True",
         "hovertemplate": "Productos=%{x}<br>Ingresos=%{y}<extra></extra>",
         "legendgroup": "mortgage",
         "marker": {
          "color": "#61c10b"
         },
         "name": "mortgage",
         "offsetgroup": "mortgage",
         "orientation": "v",
         "showlegend": true,
         "textposition": "auto",
         "type": "bar",
         "x": [
          "mortgage"
         ],
         "xaxis": "x",
         "y": [
          1500
         ],
         "yaxis": "y"
        },
        {
         "alignmentgroup": "True",
         "hovertemplate": "Productos=%{x}<br>Ingresos=%{y}<extra></extra>",
         "legendgroup": "payroll",
         "marker": {
          "color": "#31ac28"
         },
         "name": "payroll",
         "offsetgroup": "payroll",
         "orientation": "v",
         "showlegend": true,
         "textposition": "auto",
         "type": "bar",
         "x": [
          "payroll"
         ],
         "xaxis": "x",
         "y": [
          218040
         ],
         "yaxis": "y"
        },
        {
         "alignmentgroup": "True",
         "hovertemplate": "Productos=%{x}<br>Ingresos=%{y}<extra></extra>",
         "legendgroup": "payroll_account",
         "marker": {
          "color": "#439064"
         },
         "name": "payroll_account",
         "offsetgroup": "payroll_account",
         "orientation": "v",
         "showlegend": true,
         "textposition": "auto",
         "type": "bar",
         "x": [
          "payroll_account"
         ],
         "xaxis": "x",
         "y": [
          33500
         ],
         "yaxis": "y"
        },
        {
         "alignmentgroup": "True",
         "hovertemplate": "Productos=%{x}<br>Ingresos=%{y}<extra></extra>",
         "legendgroup": "pension_plan",
         "marker": {
          "color": "#3d719a"
         },
         "name": "pension_plan",
         "offsetgroup": "pension_plan",
         "orientation": "v",
         "showlegend": true,
         "textposition": "auto",
         "type": "bar",
         "x": [
          "pension_plan"
         ],
         "xaxis": "x",
         "y": [
          40840
         ],
         "yaxis": "y"
        },
        {
         "alignmentgroup": "True",
         "hovertemplate": "Productos=%{x}<br>Ingresos=%{y}<extra></extra>",
         "legendgroup": "securities",
         "marker": {
          "color": "#284ec8"
         },
         "name": "securities",
         "offsetgroup": "securities",
         "orientation": "v",
         "showlegend": true,
         "textposition": "auto",
         "type": "bar",
         "x": [
          "securities"
         ],
         "xaxis": "x",
         "y": [
          72320
         ],
         "yaxis": "y"
        },
        {
         "alignmentgroup": "True",
         "hovertemplate": "Productos=%{x}<br>Ingresos=%{y}<extra></extra>",
         "legendgroup": "short_term_deposit",
         "marker": {
          "color": "#2e21ea"
         },
         "name": "short_term_deposit",
         "offsetgroup": "short_term_deposit",
         "orientation": "v",
         "showlegend": true,
         "textposition": "auto",
         "type": "bar",
         "x": [
          "short_term_deposit"
         ],
         "xaxis": "x",
         "y": [
          212960
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
         "categoryarray": [
          "credit_card",
          "debit_card",
          "em_acount",
          "emc_account",
          "funds",
          "loans",
          "long_term_deposit",
          "mortgage",
          "payroll",
          "payroll_account",
          "pension_plan",
          "securities",
          "short_term_deposit"
         ],
         "categoryorder": "array",
         "domain": [
          0,
          1
         ],
         "title": {
          "text": "Productos"
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
   "metadata": {}
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "source": [
    "Prod_recurrencia= full_3.groupby(['Productos'])[\"recurrencia\"].mean().reset_index()\r\n",
    "Prod_recurrencia"
   ],
   "outputs": [
    {
     "output_type": "execute_result",
     "data": {
      "text/plain": [
       "             Productos  recurrencia\n",
       "0          credit_card     8.831033\n",
       "1           debit_card    24.365115\n",
       "2         em_account_p    34.500000\n",
       "3            em_acount    11.000248\n",
       "4          emc_account    20.107169\n",
       "5                funds     7.840937\n",
       "6                loans     7.117521\n",
       "7    long_term_deposit     8.713968\n",
       "8             mortgage     8.160494\n",
       "9              payroll     9.501339\n",
       "10     payroll_account    27.217680\n",
       "11        pension_plan    20.516373\n",
       "12          securities     8.997425\n",
       "13  short_term_deposit     2.036638"
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
       "      <th>Productos</th>\n",
       "      <th>recurrencia</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>credit_card</td>\n",
       "      <td>8.831033</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>debit_card</td>\n",
       "      <td>24.365115</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>em_account_p</td>\n",
       "      <td>34.500000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>em_acount</td>\n",
       "      <td>11.000248</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>emc_account</td>\n",
       "      <td>20.107169</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>5</th>\n",
       "      <td>funds</td>\n",
       "      <td>7.840937</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>6</th>\n",
       "      <td>loans</td>\n",
       "      <td>7.117521</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>7</th>\n",
       "      <td>long_term_deposit</td>\n",
       "      <td>8.713968</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>8</th>\n",
       "      <td>mortgage</td>\n",
       "      <td>8.160494</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>9</th>\n",
       "      <td>payroll</td>\n",
       "      <td>9.501339</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>10</th>\n",
       "      <td>payroll_account</td>\n",
       "      <td>27.217680</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>11</th>\n",
       "      <td>pension_plan</td>\n",
       "      <td>20.516373</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>12</th>\n",
       "      <td>securities</td>\n",
       "      <td>8.997425</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>13</th>\n",
       "      <td>short_term_deposit</td>\n",
       "      <td>2.036638</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ]
     },
     "metadata": {},
     "execution_count": 12
    }
   ],
   "metadata": {}
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "source": [
    "Prod_recurrencia= full_3.groupby(['Productos'])[\"recurrencia\"].mean().reset_index()\r\n",
    "Prod_recurrencia\r\n",
    "Prod_recurrencia = px.bar(Prod_recurrencia, x=\"Productos\", y=\"recurrencia\", \\\r\n",
    "                        color=\"Productos\", orientation=\"v\", color_discrete_sequence=px.colors.cyclical.mygbm)\r\n",
    "\r\n",
    "\r\n",
    "\r\n",
    "Prod_recurrencia.show()\r\n"
   ],
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
         "hovertemplate": "Productos=%{x}<br>recurrencia=%{y}<extra></extra>",
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
          "credit_card"
         ],
         "xaxis": "x",
         "y": [
          8.831033217621991
         ],
         "yaxis": "y"
        },
        {
         "alignmentgroup": "True",
         "hovertemplate": "Productos=%{x}<br>recurrencia=%{y}<extra></extra>",
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
          "debit_card"
         ],
         "xaxis": "x",
         "y": [
          24.36511489107729
         ],
         "yaxis": "y"
        },
        {
         "alignmentgroup": "True",
         "hovertemplate": "Productos=%{x}<br>recurrencia=%{y}<extra></extra>",
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
          "em_account_p"
         ],
         "xaxis": "x",
         "y": [
          34.5
         ],
         "yaxis": "y"
        },
        {
         "alignmentgroup": "True",
         "hovertemplate": "Productos=%{x}<br>recurrencia=%{y}<extra></extra>",
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
          "em_acount"
         ],
         "xaxis": "x",
         "y": [
          11.000248082778855
         ],
         "yaxis": "y"
        },
        {
         "alignmentgroup": "True",
         "hovertemplate": "Productos=%{x}<br>recurrencia=%{y}<extra></extra>",
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
          "emc_account"
         ],
         "xaxis": "x",
         "y": [
          20.107168744896182
         ],
         "yaxis": "y"
        },
        {
         "alignmentgroup": "True",
         "hovertemplate": "Productos=%{x}<br>recurrencia=%{y}<extra></extra>",
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
          "funds"
         ],
         "xaxis": "x",
         "y": [
          7.840937360067665
         ],
         "yaxis": "y"
        },
        {
         "alignmentgroup": "True",
         "hovertemplate": "Productos=%{x}<br>recurrencia=%{y}<extra></extra>",
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
          "loans"
         ],
         "xaxis": "x",
         "y": [
          7.117521367521367
         ],
         "yaxis": "y"
        },
        {
         "alignmentgroup": "True",
         "hovertemplate": "Productos=%{x}<br>recurrencia=%{y}<extra></extra>",
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
          "long_term_deposit"
         ],
         "xaxis": "x",
         "y": [
          8.71396781243791
         ],
         "yaxis": "y"
        },
        {
         "alignmentgroup": "True",
         "hovertemplate": "Productos=%{x}<br>recurrencia=%{y}<extra></extra>",
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
          "mortgage"
         ],
         "xaxis": "x",
         "y": [
          8.160493827160494
         ],
         "yaxis": "y"
        },
        {
         "alignmentgroup": "True",
         "hovertemplate": "Productos=%{x}<br>recurrencia=%{y}<extra></extra>",
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
          "payroll"
         ],
         "xaxis": "x",
         "y": [
          9.501338762895616
         ],
         "yaxis": "y"
        },
        {
         "alignmentgroup": "True",
         "hovertemplate": "Productos=%{x}<br>recurrencia=%{y}<extra></extra>",
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
          "payroll_account"
         ],
         "xaxis": "x",
         "y": [
          27.21767955801105
         ],
         "yaxis": "y"
        },
        {
         "alignmentgroup": "True",
         "hovertemplate": "Productos=%{x}<br>recurrencia=%{y}<extra></extra>",
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
          "pension_plan"
         ],
         "xaxis": "x",
         "y": [
          20.51637266875419
         ],
         "yaxis": "y"
        },
        {
         "alignmentgroup": "True",
         "hovertemplate": "Productos=%{x}<br>recurrencia=%{y}<extra></extra>",
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
          "securities"
         ],
         "xaxis": "x",
         "y": [
          8.99742466000994
         ],
         "yaxis": "y"
        },
        {
         "alignmentgroup": "True",
         "hovertemplate": "Productos=%{x}<br>recurrencia=%{y}<extra></extra>",
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
          "short_term_deposit"
         ],
         "xaxis": "x",
         "y": [
          2.03663765103287
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
         "categoryarray": [
          "credit_card",
          "debit_card",
          "em_account_p",
          "em_acount",
          "emc_account",
          "funds",
          "loans",
          "long_term_deposit",
          "mortgage",
          "payroll",
          "payroll_account",
          "pension_plan",
          "securities",
          "short_term_deposit"
         ],
         "categoryorder": "array",
         "domain": [
          0,
          1
         ],
         "title": {
          "text": "Productos"
         }
        },
        "yaxis": {
         "anchor": "x",
         "domain": [
          0,
          1
         ],
         "title": {
          "text": "recurrencia"
         }
        }
       }
      }
     },
     "metadata": {}
    }
   ],
   "metadata": {}
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "source": [
    "Prod_ventaq= full_3.groupby(['Productos'])[\"Ventas_cant\"].sum().reset_index()\r\n",
    "Prod_ventaq"
   ],
   "outputs": [
    {
     "output_type": "execute_result",
     "data": {
      "text/plain": [
       "             Productos  Ventas_cant\n",
       "0          credit_card      70866.0\n",
       "1           debit_card     562968.0\n",
       "2         em_account_p         34.0\n",
       "3            em_acount    4381602.0\n",
       "4          emc_account     326961.0\n",
       "5                funds      20099.0\n",
       "6                loans        468.0\n",
       "7    long_term_deposit     100660.0\n",
       "8             mortgage        324.0\n",
       "9              payroll     205787.0\n",
       "10     payroll_account     329420.0\n",
       "11        pension_plan     217802.0\n",
       "12          securities      22133.0\n",
       "13  short_term_deposit      15394.0"
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
       "      <th>Productos</th>\n",
       "      <th>Ventas_cant</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>credit_card</td>\n",
       "      <td>70866.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>debit_card</td>\n",
       "      <td>562968.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>em_account_p</td>\n",
       "      <td>34.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>em_acount</td>\n",
       "      <td>4381602.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>emc_account</td>\n",
       "      <td>326961.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>5</th>\n",
       "      <td>funds</td>\n",
       "      <td>20099.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>6</th>\n",
       "      <td>loans</td>\n",
       "      <td>468.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>7</th>\n",
       "      <td>long_term_deposit</td>\n",
       "      <td>100660.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>8</th>\n",
       "      <td>mortgage</td>\n",
       "      <td>324.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>9</th>\n",
       "      <td>payroll</td>\n",
       "      <td>205787.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>10</th>\n",
       "      <td>payroll_account</td>\n",
       "      <td>329420.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>11</th>\n",
       "      <td>pension_plan</td>\n",
       "      <td>217802.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>12</th>\n",
       "      <td>securities</td>\n",
       "      <td>22133.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>13</th>\n",
       "      <td>short_term_deposit</td>\n",
       "      <td>15394.0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ]
     },
     "metadata": {},
     "execution_count": 21
    }
   ],
   "metadata": {}
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "source": [
    "Prod_ingresos= full_3.groupby(['Productos'])[\"Ingresos\"].sum().reset_index()\r\n",
    "Prod_ingresos"
   ],
   "outputs": [
    {
     "output_type": "execute_result",
     "data": {
      "text/plain": [
       "             Productos    Ingresos\n",
       "0          credit_card   4251960.0\n",
       "1           debit_card   5629680.0\n",
       "2         em_account_p       340.0\n",
       "3            em_acount  43816020.0\n",
       "4          emc_account   3269610.0\n",
       "5                funds    803960.0\n",
       "6                loans     28080.0\n",
       "7    long_term_deposit   4026400.0\n",
       "8             mortgage     19440.0\n",
       "9              payroll   2057870.0\n",
       "10     payroll_account   3294200.0\n",
       "11        pension_plan   8712080.0\n",
       "12          securities    885320.0\n",
       "13  short_term_deposit    615760.0"
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
       "      <th>Productos</th>\n",
       "      <th>Ingresos</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>credit_card</td>\n",
       "      <td>4251960.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>debit_card</td>\n",
       "      <td>5629680.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>em_account_p</td>\n",
       "      <td>340.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>em_acount</td>\n",
       "      <td>43816020.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>emc_account</td>\n",
       "      <td>3269610.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>5</th>\n",
       "      <td>funds</td>\n",
       "      <td>803960.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>6</th>\n",
       "      <td>loans</td>\n",
       "      <td>28080.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>7</th>\n",
       "      <td>long_term_deposit</td>\n",
       "      <td>4026400.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>8</th>\n",
       "      <td>mortgage</td>\n",
       "      <td>19440.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>9</th>\n",
       "      <td>payroll</td>\n",
       "      <td>2057870.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>10</th>\n",
       "      <td>payroll_account</td>\n",
       "      <td>3294200.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>11</th>\n",
       "      <td>pension_plan</td>\n",
       "      <td>8712080.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>12</th>\n",
       "      <td>securities</td>\n",
       "      <td>885320.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>13</th>\n",
       "      <td>short_term_deposit</td>\n",
       "      <td>615760.0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
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
   "execution_count": 6,
   "source": [
    "full_3.head()"
   ],
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
       "27827  short_term_deposit          1.0       40    Inversión      40.0   \n",
       "\n",
       "      entry_date entry_channel  días_encartera   pk_cid country_id  \\\n",
       "22134 2018-04-08           KHK            20.0   100296         ES   \n",
       "22135 2018-04-08           KHK            50.0   100296         ES   \n",
       "22136 2018-04-08           KHK            81.0   100296         ES   \n",
       "27826 2015-01-30           KHM          1337.0  1003705         ES   \n",
       "27827 2015-01-30           KHM          1367.0  1003705         ES   \n",
       "\n",
       "       region_code gender  age deceased       date  Month  days_between  \\\n",
       "22134         28.0      V   46        N 2018-04-28      4           0.0   \n",
       "22135         28.0      V   46        N 2018-05-28      5          30.0   \n",
       "22136         28.0      V   46        N 2018-06-28      6          31.0   \n",
       "27826         28.0      H   33        N 2018-09-28      9           0.0   \n",
       "27827         28.0      H   33        N 2018-10-28     10          30.0   \n",
       "\n",
       "       recurrencia  \n",
       "22134            1  \n",
       "22135            2  \n",
       "22136            3  \n",
       "27826            1  \n",
       "27827            2  "
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
       "    </tr>\n",
       "    <tr>\n",
       "      <th>27827</th>\n",
       "      <td>short_term_deposit</td>\n",
       "      <td>1.0</td>\n",
       "      <td>40</td>\n",
       "      <td>Inversión</td>\n",
       "      <td>40.0</td>\n",
       "      <td>2015-01-30</td>\n",
       "      <td>KHM</td>\n",
       "      <td>1367.0</td>\n",
       "      <td>1003705</td>\n",
       "      <td>ES</td>\n",
       "      <td>28.0</td>\n",
       "      <td>H</td>\n",
       "      <td>33</td>\n",
       "      <td>N</td>\n",
       "      <td>2018-10-28</td>\n",
       "      <td>10</td>\n",
       "      <td>30.0</td>\n",
       "      <td>2</td>\n",
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
    "azdata_cell_guid": "22afde4a-8fe0-498a-adb8-c185a70c0a30"
   }
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "source": [
    "full_3[\"Year\"] = full_3[\"date\"].dt.year"
   ],
   "outputs": [],
   "metadata": {
    "azdata_cell_guid": "6134d418-7fd4-4cc4-bce8-1ed1da91c720"
   }
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "source": [
    "full_3['pk_cid'].value_counts()"
   ],
   "outputs": [
    {
     "output_type": "execute_result",
     "data": {
      "text/plain": [
       "1128353    149\n",
       "1190607    145\n",
       "1070525    142\n",
       "1133500    136\n",
       "1071910    136\n",
       "          ... \n",
       "1549173      1\n",
       "1548495      1\n",
       "1489571      1\n",
       "1535751      1\n",
       "1542539      1\n",
       "Name: pk_cid, Length: 350384, dtype: int64"
      ]
     },
     "metadata": {},
     "execution_count": 8
    }
   ],
   "metadata": {
    "azdata_cell_guid": "2ca69708-3018-4653-9a14-9239ebb10fa3"
   }
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "source": [
    "def plot_cat_values(dataframe, column):\r\n",
    "\r\n",
    "    plt.figure(figsize=(15,8))\r\n",
    "\r\n",
    "    ax1 = plt.subplot(2,1,1) #una imagen con dos plots por fila, 1 plot por columna, el ax1 es el primer plot\r\n",
    "    #grafico 1 count\r\n",
    "    ax1 = sns.countplot(\r\n",
    "         dataframe[column],\r\n",
    "         order = list(dataframe[column].unique())\r\n",
    "\r\n",
    "        )\r\n",
    "\r\n",
    "   #grafico 3 leyenda \r\n",
    "\r\n",
    "    ax = sns.countplot(x=column, hue=\"pk_cid\", data=dataframe)\r\n",
    "    ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha=\"right\")\r\n",
    "    plt.tight_layout()\r\n",
    "    plt.show()"
   ],
   "outputs": [],
   "metadata": {
    "azdata_cell_guid": "8da39fde-462a-48a9-adda-433b57fc959f"
   }
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "source": [
    "full_3['Familia_prod'].value_counts()"
   ],
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
     "execution_count": 6
    }
   ],
   "metadata": {
    "azdata_cell_guid": "9b90f1de-7548-487a-91a1-65d8081321c7"
   }
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "source": [
    "import plotly.express as px"
   ],
   "outputs": [],
   "metadata": {
    "azdata_cell_guid": "2730be9d-6657-4e9d-808b-26eec38d75be"
   }
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "source": [
    "grafico_horizontal= full_3.groupby([\"Productos\",\"date\"])[\"Ingresos\"].sum().reset_index()\r\n",
    "grafico_horizontal\r\n",
    "\r\n",
    "evolucion_horizontal= px.bar(grafico_horizontal, x=\"date\", y=\"Ingresos\", \\\r\n",
    "                        color=\"Productos\", orientation=\"v\", color_discrete_sequence=px.colors.cyclical.mygbm)\r\n",
    "\r\n",
    "evolucion_horizontal.show()"
   ],
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
   "metadata": {
    "azdata_cell_guid": "2e3a75dc-59c5-4dc9-9414-c7eada08b0cf"
   }
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "source": [
    "client_pivot_[\"venta_perc_cliente\"] = (client_pivot_[\"venta_por_cliente\"]/client_pivot_[\"venta_por_cliente\"].sum()).cumsum()\r\n",
    "client_pivot_.head()"
   ],
   "outputs": [
    {
     "output_type": "error",
     "ename": "NameError",
     "evalue": "name 'client_pivot_' is not defined",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mNameError\u001b[0m                                 Traceback (most recent call last)",
      "\u001b[1;32m<ipython-input-27-140435d3e9ce>\u001b[0m in \u001b[0;36m<module>\u001b[1;34m\u001b[0m\n\u001b[1;32m----> 1\u001b[1;33m \u001b[0mclient_pivot_\u001b[0m\u001b[1;33m[\u001b[0m\u001b[1;34m\"venta_perc_cliente\"\u001b[0m\u001b[1;33m]\u001b[0m \u001b[1;33m=\u001b[0m \u001b[1;33m(\u001b[0m\u001b[0mclient_pivot_\u001b[0m\u001b[1;33m[\u001b[0m\u001b[1;34m\"venta_por_cliente\"\u001b[0m\u001b[1;33m]\u001b[0m\u001b[1;33m/\u001b[0m\u001b[0mclient_pivot_\u001b[0m\u001b[1;33m[\u001b[0m\u001b[1;34m\"venta_por_cliente\"\u001b[0m\u001b[1;33m]\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0msum\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mcumsum\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m      2\u001b[0m \u001b[0mclient_pivot_\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mhead\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;31mNameError\u001b[0m: name 'client_pivot_' is not defined"
     ]
    }
   ],
   "metadata": {}
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "source": [
    "grafico_horizontal= full_3.groupby(['Familia_prod',\"date\"])[\"Ingresos\"].sum().reset_index()\r\n",
    "grafico_horizontal\r\n",
    "\r\n",
    "evolucion_horizontal= px.bar(grafico_horizontal, x=\"date\", y=\"Ingresos\", \\\r\n",
    "                        color=\"Familia_prod\", orientation=\"v\", color_discrete_sequence=px.colors.cyclical.mygbm)\r\n",
    "\r\n",
    "evolucion_horizontal.show()"
   ],
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
   "metadata": {
    "azdata_cell_guid": "7ea44914-18a2-454d-a8ab-e9421f319103"
   }
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "source": [
    "grafico_horizontal= full_3.groupby(['entry_channel',\"date\"])[\"Ingresos\"].mean().reset_index()\r\n",
    "grafico_horizontal\r\n",
    "\r\n",
    "evolucion_horizontal= px.bar(grafico_horizontal, x=\"date\", y=\"Ingresos\", \\\r\n",
    "                        color=\"entry_channel\", orientation=\"v\", color_discrete_sequence=px.colors.cyclical.mygbm)\r\n",
    "\r\n",
    "evolucion_horizontal.show()"
   ],
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
          15.789473684210526,
          15,
          15,
          17.5,
          16.666666666666668,
          10,
          10.669020148097124,
          10.630424851530378,
          10.469498165536969,
          10.403330249768732,
          10.384162149422579,
          11.108754922750682,
          10.480567226890756,
          10.740463724756918,
          10.455888560574081,
          10.867655134541462,
          10.455621301775148
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
          25,
          25,
          20,
          25,
          20,
          25,
          25,
          25,
          25,
          25,
          25,
          25,
          20,
          25,
          25,
          25,
          25
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
          23.47826086956522,
          23.676470588235293,
          23.75,
          23.6,
          23.333333333333332,
          23.64864864864865,
          23.866666666666667,
          23.924050632911392,
          23.820224719101123,
          23.636363636363637,
          23.645833333333332,
          22.94736842105263,
          22.63157894736842,
          22.68041237113402,
          22.1875,
          22.88659793814433,
          22.92929292929293
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
          16.756756756756758,
          16.216216216216218,
          18.46153846153846,
          18.333333333333332,
          17.56756756756757,
          18,
          18.46153846153846,
          20,
          20.454545454545453,
          20.22222222222222,
          20.425531914893618,
          19.8,
          19.347826086956523,
          20.208333333333332,
          19.2,
          19.821428571428573,
          18.928571428571427
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
          28.88888888888889,
          28.88888888888889,
          28.88888888888889,
          28.88888888888889,
          28.88888888888889,
          28.88888888888889,
          27.5,
          31.11111111111111,
          26,
          28.571428571428573,
          30,
          23.75,
          23.75,
          23.333333333333332,
          23.75,
          27.77777777777778,
          24.545454545454547
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
          60,
          60,
          60,
          60,
          60,
          60,
          60,
          60,
          60,
          60,
          60,
          25,
          60,
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
          18.571428571428573,
          18.571428571428573,
          18.571428571428573,
          18.571428571428573,
          18.571428571428573,
          18.571428571428573,
          18.571428571428573,
          18.571428571428573,
          18.571428571428573,
          18.571428571428573,
          18.571428571428573,
          18.571428571428573,
          18.571428571428573,
          18.571428571428573,
          18.571428571428573,
          18.571428571428573,
          18.571428571428573
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
          26.666666666666668,
          26.666666666666668,
          26,
          23.75,
          23.333333333333332,
          23.75,
          18.571428571428573,
          25.714285714285715,
          23.75,
          23.333333333333332,
          23.333333333333332,
          23.75,
          22.5,
          23.333333333333332,
          16,
          10,
          22.5
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
          24.166666666666668,
          24.166666666666668,
          27.272727272727273,
          26,
          26,
          26,
          22.22222222222222,
          20,
          20,
          20,
          20,
          24.444444444444443,
          20,
          18.333333333333332,
          20,
          23.571428571428573,
          20.76923076923077
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
          27.142857142857142,
          24.666666666666668,
          24.666666666666668,
          24.666666666666668,
          22.142857142857142,
          23.333333333333332,
          25.625,
          21.333333333333332,
          24.666666666666668,
          22.22222222222222,
          25,
          21.904761904761905,
          21.904761904761905,
          21.363636363636363,
          18.94736842105263,
          20.476190476190474,
          21
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
          35,
          35,
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
          25,
          25,
          25,
          25,
          25,
          25,
          25,
          25,
          25,
          25,
          25,
          25,
          25,
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
          40,
          40,
          40,
          40,
          40,
          40,
          50,
          50,
          50,
          50,
          50,
          50,
          50,
          50,
          50,
          50
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
          20,
          20,
          17.5,
          17.5,
          17.5,
          14.285714285714286,
          14.285714285714286,
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
          27.058823529411764,
          30,
          24,
          26.11111111111111,
          26.25,
          27.894736842105264,
          25.263157894736842,
          27.894736842105264,
          26.842105263157894,
          29.047619047619047,
          29.047619047619047,
          27.5,
          23.333333333333332,
          25,
          23.5,
          24.583333333333332,
          26.666666666666668
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
          14.810238305383937,
          14.84369582803652,
          14.80412429538566,
          14.964197807115687,
          14.81051361889534,
          15.018665486823549,
          15.095064336470136,
          15.051790388149469,
          15.047129391602398,
          15.081033727551468,
          15.101839414457633,
          15.146279846793252,
          14.820734944809278,
          15.014492753623188,
          14.992772792974609,
          14.94544357933984,
          14.964283766835706
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
          20,
          10,
          10,
          10,
          10,
          10,
          10,
          10,
          10,
          22.5,
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
          35,
          35,
          35,
          35,
          35,
          28.75,
          32.22222222222222,
          26.666666666666668,
          26.666666666666668,
          23.333333333333332,
          36.666666666666664,
          36.666666666666664,
          26,
          30,
          30,
          36.666666666666664,
          30
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
          16.382608695652173,
          16.239168110918545,
          16.490630323679728,
          16.14853195164076,
          15.934256055363322,
          15.975820379965457,
          16.19047619047619,
          16.010362694300518,
          15.99644128113879,
          16.163410301953817,
          15.897887323943662,
          16.1139896373057,
          16.134751773049647,
          16.102564102564102,
          15.989492119089316,
          16.13240418118467,
          15.887521968365554
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
          17.5,
          17.5,
          17.5,
          17.5,
          17.5,
          17.5,
          17.5,
          17.5,
          17.5,
          17.5,
          17.5,
          17.5,
          17.5,
          17.5,
          17.5,
          17.5,
          17.5
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
          13.580246913580247,
          14.16184971098266,
          13.941176470588236,
          14.107142857142858,
          14.311377245508982,
          14.049079754601227,
          14.156626506024097,
          14.404761904761905,
          13.952095808383234,
          14.058823529411764,
          14.035087719298245,
          14.142011834319527,
          13.202614379084967,
          13.522012578616351,
          13.333333333333334,
          13.788819875776397,
          13.602484472049689
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
          20,
          20,
          20,
          17.5,
          17.5,
          17.5,
          17.5,
          17.5,
          17.5,
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
          18.88888888888889,
          20,
          24.615384615384617,
          23.571428571428573,
          24.615384615384617,
          21.666666666666668,
          21.666666666666668,
          21.666666666666668,
          20.76923076923077,
          20,
          20.76923076923077,
          20.76923076923077,
          20.76923076923077,
          20.76923076923077,
          20,
          20.76923076923077,
          20.76923076923077
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
          50,
          50,
          50,
          32,
          32,
          25,
          25,
          25,
          36.666666666666664,
          36.666666666666664,
          36.666666666666664,
          26.666666666666668,
          26.666666666666668,
          10,
          26.666666666666668,
          26.666666666666668,
          26.666666666666668
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
          15,
          10,
          15,
          15,
          15,
          15,
          15,
          15,
          15,
          15,
          14.285714285714286,
          17.5,
          14.285714285714286,
          14.285714285714286,
          17.5,
          17.5,
          17.5
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
          35,
          10,
          10,
          10,
          43.333333333333336,
          10,
          10,
          10,
          10,
          35,
          35,
          35,
          35,
          35,
          35,
          35,
          35
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
          10,
          10,
          10,
          10
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
          10,
          10,
          26.666666666666668,
          26.666666666666668,
          26.666666666666668,
          26.666666666666668,
          10,
          10,
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
          10,
          10,
          10.714285714285714,
          10.714285714285714,
          10.714285714285714,
          10,
          10.731707317073171,
          10.731707317073171,
          10.731707317073171,
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
          17.480234260614935,
          17.292952490311468,
          17.04694247109625,
          17.328028055031023,
          17.131202353885246,
          17.31659981886402,
          17.382225617519737,
          17.390585241730278,
          17.328062248995984,
          17.219059405940595,
          17.280108254397835,
          17.333494792928068,
          16.966459627329193,
          17.11566617862372,
          16.98512516628371,
          16.93318729463308,
          16.880367727107778
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
          14.840081397796645,
          14.839292916649281,
          14.865651559046038,
          14.972894535100208,
          14.918145644301863,
          15.101677647218638,
          15.184032549241506,
          15.105498545793777,
          15.125414424696995,
          15.154834347532116,
          15.15263029249613,
          15.21416371925389,
          14.86112268422076,
          15.073458328227332,
          15.01397140609316,
          15.002037157757497,
          15.011976777231943
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
          20,
          20,
          30,
          20,
          20,
          20,
          20,
          22.5,
          30,
          30,
          30,
          30,
          26,
          32,
          32,
          32,
          32
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
          10,
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
          15.54240631163708,
          15.512572533849129,
          15.652173913043478,
          15.833333333333334,
          15.705765407554672,
          15.81573896353167,
          16.076923076923077,
          15.57312252964427,
          15.447316103379721,
          15.794573643410853,
          15.984405458089668,
          15.8984375,
          15.588822355289421,
          15.575757575757576,
          15.48582995951417,
          15.517241379310345,
          15.668016194331983
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
          10.179073931951907,
          10.186748529035558,
          10.212548015364916,
          10.221938775510203,
          10.214833759590793,
          10.25811397904421,
          10.369713851608003,
          10.26646169613118,
          10.319529652351738,
          10.227563283047814,
          10.273377618804293,
          10.330704655303993,
          10.24735892811131,
          10.275064267352185,
          10.304815573770492,
          10.284834488067744,
          10.35595390524968
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
          10.2567663007697,
          10.278277393024242,
          10.296055180965666,
          10.311590572968846,
          10.318477800259032,
          10.366959351919741,
          10.418277910517128,
          10.402931937172776,
          10.404996147160547,
          10.410302450161423,
          10.430292291958999,
          10.475319494018551,
          10.38934817929848,
          10.459111156044269,
          10.47690298064363,
          10.48444161666855,
          10.51543695955784
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
          10.526315789473685,
          10.817610062893081,
          10.791788856304985,
          11.254646840148698,
          10.852272727272727,
          11.341240875912408,
          11.201117318435754,
          11.112158341187559,
          11.040681173131505,
          11.01038715769594,
          10.921177587844255,
          11.068773234200744,
          10.861742424242424,
          11.024436090225564,
          11.230342275670676,
          11.104921077065924,
          11.211840888066606
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
          14.069714110992203,
          14.918050348152116,
          15.233707488691573,
          15.043435270852461,
          14.463230153181211,
          14.246159451089927,
          14.241275108135332,
          14.08968448781656,
          14.07381037142518,
          14.236802060166218,
          14.324545507686912,
          14.396237172177878,
          14.070734686754697,
          14.265734265734265,
          14.301215032495055,
          14.274796195652174,
          14.226321430586111
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
          11.896272285251216,
          13.647136273864385,
          14.526873580620743,
          15.159505208333334,
          14.777947932618684,
          14.356060606060606,
          14.376398210290828,
          14.316505955757233,
          14.449388209121246,
          14.55772940203913,
          14.583448657625242,
          14.747419880499729,
          14.683544303797468,
          14.825534216932649,
          14.736984815618221,
          14.617493830545653,
          14.577693557119654
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
          20,
          21.428571428571427,
          10,
          10.69767441860465,
          11.252086811352253,
          13.15994166261546,
          15.018816284639069,
          15.03125,
          14.82859810590903,
          14.406126577401022,
          13.875157722853114,
          13.607254010055064,
          12.953564002772298,
          12.984259138805776,
          12.941569034554353,
          12.915809043224222,
          12.954133977066988
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
          10,
          10,
          10.454545454545455,
          13.20675105485232,
          14.530386740331492,
          15.453945196434466,
          16.17045135968601,
          15.728715728715729,
          15.47544080604534,
          15.18786836935167,
          14.70541242260835,
          14.270150651119245,
          13.757036360870227,
          13.759827542480345,
          13.634566506636544,
          13.5798109876415,
          13.611336032388664
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
          14.137931034482758,
          16.4,
          16.25,
          16.438356164383563,
          16.173285198555956,
          16.454293628808863,
          16.004016064257026,
          16.205787781350484,
          16.13821138211382,
          16.268980477223426,
          16.08780487804878,
          15.711987127916332,
          15.682788051209103,
          15.502210991787745,
          15.296465222348917
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
          10,
          10,
          10,
          10.454545454545455,
          10.81081081081081,
          10.405405405405405,
          10.789473684210526,
          10.789473684210526,
          10.8,
          10.8
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
          10.023273855702096,
          10.037431335375043,
          10.049866832889442,
          10.072206559438573,
          10.117981367576705,
          10.104179629860086,
          10.12538014191965,
          10.147397921174926,
          10.16874567489398,
          10.191621866184063
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
          14.029697900665642,
          14.248728617660657,
          14.245689655172415,
          14.49124949124949,
          14.448616600790514,
          14.66310364573392,
          14.893261555596807,
          14.77847533632287,
          14.898771988051775,
          14.999244370560676,
          14.915398834067965,
          14.88209722039257,
          14.529580401355226,
          14.674072265625,
          14.744851258581235,
          14.602860286028603,
          14.657705532979056
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
   "metadata": {
    "azdata_cell_guid": "c85e4ccb-6b7c-4b66-a876-a14509606889"
   }
  },
  {
   "cell_type": "code",
   "execution_count": 33,
   "source": [
    "\r\n",
    "grafico_horizontal= full_3.groupby(['days_between',\"date\"])[\"Ingresos\"].sum().reset_index()\r\n",
    "grafico_horizontal\r\n",
    "\r\n",
    "evolucion_horizontal= px.bar(grafico_horizontal, x=\"date\", y=\"Ingresos\", \\\r\n",
    "                        color='days_between', orientation=\"v\", color_discrete_sequence=px.colors.cyclical.mygbm)\r\n",
    "\r\n",
    "evolucion_horizontal.show()"
   ],
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
         "hovertemplate": "date=%{x}<br>Ingresos=%{y}<br>days_between=%{marker.color}<extra></extra>",
         "legendgroup": "",
         "marker": {
          "color": [
           -485,
           -455,
           -454,
           -426,
           -424,
           -424,
           -396,
           -396,
           -395,
           -393,
           -365,
           -365,
           -365,
           -365,
           -365,
           -337,
           -335,
           -334,
           -334,
           -334,
           -334,
           -306,
           -306,
           -304,
           -304,
           -304,
           -304,
           -303,
           -276,
           -275,
           -275,
           -274,
           -273,
           -273,
           -273,
           -273,
           -245,
           -245,
           -245,
           -244,
           -243,
           -243,
           -243,
           -242,
           -242,
           -215,
           -214,
           -214,
           -214,
           -214,
           -212,
           -212,
           -212,
           -212,
           -212,
           -184,
           -184,
           -184,
           -184,
           -183,
           -183,
           -182,
           -181,
           -181,
           -181,
           -181,
           -153,
           -153,
           -153,
           -153,
           -153,
           -153,
           -153,
           -151,
           -151,
           -151,
           -151,
           -150,
           -123,
           -123,
           -123,
           -122,
           -122,
           -122,
           -122,
           -122,
           -121,
           -120,
           -120,
           -120,
           -120,
           -92,
           -92,
           -92,
           -92,
           -92,
           -92,
           -92,
           -91,
           -91,
           -90,
           -90,
           -90,
           -89,
           -89,
           -62,
           -62,
           -61,
           -61,
           -61,
           -61,
           -61,
           -61,
           -61,
           -61,
           -61,
           -59,
           -59,
           -59,
           -59,
           -31,
           -31,
           -31,
           -31,
           -31,
           -31,
           -31,
           -31,
           -31,
           -30,
           -30,
           -30,
           -30,
           -30,
           -28,
           -28,
           0,
           0,
           0,
           0,
           0,
           0,
           0,
           0,
           0,
           0,
           0,
           0,
           0,
           0,
           0,
           0,
           0,
           28,
           28,
           30,
           30,
           30,
           30,
           30,
           31,
           31,
           31,
           31,
           31,
           31,
           31,
           31,
           31,
           59,
           59,
           59,
           59,
           61,
           61,
           61,
           61,
           61,
           61,
           61,
           61,
           61,
           62,
           62,
           89,
           89,
           90,
           90,
           90,
           91,
           91,
           92,
           92,
           92,
           92,
           92,
           92,
           92,
           120,
           120,
           120,
           120,
           121,
           122,
           122,
           122,
           122,
           122,
           123,
           123,
           123,
           150,
           151,
           151,
           151,
           151,
           153,
           153,
           153,
           153,
           153,
           153,
           153,
           181,
           181,
           181,
           181,
           182,
           183,
           183,
           184,
           184,
           184,
           184,
           212,
           212,
           212,
           212,
           212,
           214,
           214,
           214,
           214,
           215,
           242,
           242,
           243,
           243,
           243,
           244,
           245,
           245,
           245,
           273,
           273,
           273,
           273,
           274,
           275,
           275,
           276,
           303,
           304,
           304,
           304,
           304,
           306,
           306,
           334,
           334,
           334,
           334,
           335,
           337,
           365,
           365,
           365,
           365,
           365,
           393,
           395,
           396,
           396,
           424,
           424,
           426,
           454,
           455,
           485
          ],
          "coloraxis": "coloraxis"
         },
         "name": "",
         "offsetgroup": "",
         "orientation": "v",
         "showlegend": false,
         "textposition": "auto",
         "type": "bar",
         "x": [
          "2018-01-28T00:00:00",
          "2018-01-28T00:00:00",
          "2018-02-28T00:00:00",
          "2018-03-28T00:00:00",
          "2018-01-28T00:00:00",
          "2018-02-28T00:00:00",
          "2018-01-28T00:00:00",
          "2018-03-28T00:00:00",
          "2018-04-28T00:00:00",
          "2018-02-28T00:00:00",
          "2018-01-28T00:00:00",
          "2018-02-28T00:00:00",
          "2018-03-28T00:00:00",
          "2018-04-28T00:00:00",
          "2018-05-28T00:00:00",
          "2018-03-28T00:00:00",
          "2018-05-28T00:00:00",
          "2018-01-28T00:00:00",
          "2018-02-28T00:00:00",
          "2018-04-28T00:00:00",
          "2018-06-28T00:00:00",
          "2018-03-28T00:00:00",
          "2018-04-28T00:00:00",
          "2018-01-28T00:00:00",
          "2018-05-28T00:00:00",
          "2018-06-28T00:00:00",
          "2018-07-28T00:00:00",
          "2018-02-28T00:00:00",
          "2018-05-28T00:00:00",
          "2018-03-28T00:00:00",
          "2018-04-28T00:00:00",
          "2018-07-28T00:00:00",
          "2018-01-28T00:00:00",
          "2018-02-28T00:00:00",
          "2018-06-28T00:00:00",
          "2018-08-28T00:00:00",
          "2018-03-28T00:00:00",
          "2018-05-28T00:00:00",
          "2018-06-28T00:00:00",
          "2018-04-28T00:00:00",
          "2018-01-28T00:00:00",
          "2018-07-28T00:00:00",
          "2018-08-28T00:00:00",
          "2018-02-28T00:00:00",
          "2018-09-28T00:00:00",
          "2018-07-28T00:00:00",
          "2018-03-28T00:00:00",
          "2018-04-28T00:00:00",
          "2018-05-28T00:00:00",
          "2018-06-28T00:00:00",
          "2018-01-28T00:00:00",
          "2018-02-28T00:00:00",
          "2018-08-28T00:00:00",
          "2018-09-28T00:00:00",
          "2018-10-28T00:00:00",
          "2018-03-28T00:00:00",
          "2018-05-28T00:00:00",
          "2018-07-28T00:00:00",
          "2018-08-28T00:00:00",
          "2018-04-28T00:00:00",
          "2018-06-28T00:00:00",
          "2018-10-28T00:00:00",
          "2018-01-28T00:00:00",
          "2018-02-28T00:00:00",
          "2018-09-28T00:00:00",
          "2018-11-28T00:00:00",
          "2018-03-28T00:00:00",
          "2018-04-28T00:00:00",
          "2018-05-28T00:00:00",
          "2018-06-28T00:00:00",
          "2018-07-28T00:00:00",
          "2018-08-28T00:00:00",
          "2018-09-28T00:00:00",
          "2018-01-28T00:00:00",
          "2018-10-28T00:00:00",
          "2018-11-28T00:00:00",
          "2018-12-28T00:00:00",
          "2018-02-28T00:00:00",
          "2018-05-28T00:00:00",
          "2018-07-28T00:00:00",
          "2018-10-28T00:00:00",
          "2018-03-28T00:00:00",
          "2018-04-28T00:00:00",
          "2018-06-28T00:00:00",
          "2018-08-28T00:00:00",
          "2018-09-28T00:00:00",
          "2018-12-28T00:00:00",
          "2018-01-28T00:00:00",
          "2018-02-28T00:00:00",
          "2018-11-28T00:00:00",
          "2019-01-28T00:00:00",
          "2018-03-28T00:00:00",
          "2018-05-28T00:00:00",
          "2018-06-28T00:00:00",
          "2018-07-28T00:00:00",
          "2018-08-28T00:00:00",
          "2018-10-28T00:00:00",
          "2018-11-28T00:00:00",
          "2018-04-28T00:00:00",
          "2018-09-28T00:00:00",
          "2018-01-28T00:00:00",
          "2018-12-28T00:00:00",
          "2019-01-28T00:00:00",
          "2018-02-28T00:00:00",
          "2019-02-28T00:00:00",
          "2018-07-28T00:00:00",
          "2018-12-28T00:00:00",
          "2018-03-28T00:00:00",
          "2018-04-28T00:00:00",
          "2018-05-28T00:00:00",
          "2018-06-28T00:00:00",
          "2018-08-28T00:00:00",
          "2018-09-28T00:00:00",
          "2018-10-28T00:00:00",
          "2018-11-28T00:00:00",
          "2019-03-28T00:00:00",
          "2018-01-28T00:00:00",
          "2018-02-28T00:00:00",
          "2019-01-28T00:00:00",
          "2019-02-28T00:00:00",
          "2018-01-28T00:00:00",
          "2018-03-28T00:00:00",
          "2018-05-28T00:00:00",
          "2018-07-28T00:00:00",
          "2018-08-28T00:00:00",
          "2018-10-28T00:00:00",
          "2018-12-28T00:00:00",
          "2019-01-28T00:00:00",
          "2019-03-28T00:00:00",
          "2018-04-28T00:00:00",
          "2018-06-28T00:00:00",
          "2018-09-28T00:00:00",
          "2018-11-28T00:00:00",
          "2019-04-28T00:00:00",
          "2018-02-28T00:00:00",
          "2019-02-28T00:00:00",
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
          "2019-05-28T00:00:00",
          "2018-03-28T00:00:00",
          "2019-03-28T00:00:00",
          "2018-05-28T00:00:00",
          "2018-07-28T00:00:00",
          "2018-10-28T00:00:00",
          "2018-12-28T00:00:00",
          "2019-05-28T00:00:00",
          "2018-02-28T00:00:00",
          "2018-04-28T00:00:00",
          "2018-06-28T00:00:00",
          "2018-08-28T00:00:00",
          "2018-09-28T00:00:00",
          "2018-11-28T00:00:00",
          "2019-01-28T00:00:00",
          "2019-02-28T00:00:00",
          "2019-04-28T00:00:00",
          "2018-03-28T00:00:00",
          "2018-04-28T00:00:00",
          "2019-03-28T00:00:00",
          "2019-04-28T00:00:00",
          "2018-05-28T00:00:00",
          "2018-06-28T00:00:00",
          "2018-07-28T00:00:00",
          "2018-08-28T00:00:00",
          "2018-10-28T00:00:00",
          "2018-11-28T00:00:00",
          "2018-12-28T00:00:00",
          "2019-01-28T00:00:00",
          "2019-05-28T00:00:00",
          "2018-09-28T00:00:00",
          "2019-02-28T00:00:00",
          "2018-05-28T00:00:00",
          "2019-05-28T00:00:00",
          "2018-04-28T00:00:00",
          "2019-03-28T00:00:00",
          "2019-04-28T00:00:00",
          "2018-07-28T00:00:00",
          "2018-12-28T00:00:00",
          "2018-06-28T00:00:00",
          "2018-08-28T00:00:00",
          "2018-09-28T00:00:00",
          "2018-10-28T00:00:00",
          "2018-11-28T00:00:00",
          "2019-01-28T00:00:00",
          "2019-02-28T00:00:00",
          "2018-05-28T00:00:00",
          "2018-06-28T00:00:00",
          "2019-03-28T00:00:00",
          "2019-05-28T00:00:00",
          "2019-04-28T00:00:00",
          "2018-07-28T00:00:00",
          "2018-08-28T00:00:00",
          "2018-10-28T00:00:00",
          "2018-12-28T00:00:00",
          "2019-01-28T00:00:00",
          "2018-09-28T00:00:00",
          "2018-11-28T00:00:00",
          "2019-02-28T00:00:00",
          "2018-07-28T00:00:00",
          "2018-06-28T00:00:00",
          "2019-03-28T00:00:00",
          "2019-04-28T00:00:00",
          "2019-05-28T00:00:00",
          "2018-08-28T00:00:00",
          "2018-09-28T00:00:00",
          "2018-10-28T00:00:00",
          "2018-11-28T00:00:00",
          "2018-12-28T00:00:00",
          "2019-01-28T00:00:00",
          "2019-02-28T00:00:00",
          "2018-07-28T00:00:00",
          "2018-08-28T00:00:00",
          "2019-03-28T00:00:00",
          "2019-05-28T00:00:00",
          "2019-04-28T00:00:00",
          "2018-10-28T00:00:00",
          "2018-12-28T00:00:00",
          "2018-09-28T00:00:00",
          "2018-11-28T00:00:00",
          "2019-01-28T00:00:00",
          "2019-02-28T00:00:00",
          "2018-08-28T00:00:00",
          "2018-09-28T00:00:00",
          "2019-03-28T00:00:00",
          "2019-04-28T00:00:00",
          "2019-05-28T00:00:00",
          "2018-10-28T00:00:00",
          "2018-11-28T00:00:00",
          "2018-12-28T00:00:00",
          "2019-01-28T00:00:00",
          "2019-02-28T00:00:00",
          "2018-10-28T00:00:00",
          "2019-05-28T00:00:00",
          "2018-09-28T00:00:00",
          "2019-03-28T00:00:00",
          "2019-04-28T00:00:00",
          "2018-12-28T00:00:00",
          "2018-11-28T00:00:00",
          "2019-01-28T00:00:00",
          "2019-02-28T00:00:00",
          "2018-10-28T00:00:00",
          "2018-11-28T00:00:00",
          "2019-03-28T00:00:00",
          "2019-05-28T00:00:00",
          "2019-04-28T00:00:00",
          "2018-12-28T00:00:00",
          "2019-01-28T00:00:00",
          "2019-02-28T00:00:00",
          "2018-12-28T00:00:00",
          "2018-11-28T00:00:00",
          "2019-03-28T00:00:00",
          "2019-04-28T00:00:00",
          "2019-05-28T00:00:00",
          "2019-01-28T00:00:00",
          "2019-02-28T00:00:00",
          "2018-12-28T00:00:00",
          "2019-01-28T00:00:00",
          "2019-03-28T00:00:00",
          "2019-05-28T00:00:00",
          "2019-04-28T00:00:00",
          "2019-02-28T00:00:00",
          "2019-01-28T00:00:00",
          "2019-02-28T00:00:00",
          "2019-03-28T00:00:00",
          "2019-04-28T00:00:00",
          "2019-05-28T00:00:00",
          "2019-03-28T00:00:00",
          "2019-05-28T00:00:00",
          "2019-02-28T00:00:00",
          "2019-04-28T00:00:00",
          "2019-03-28T00:00:00",
          "2019-04-28T00:00:00",
          "2019-05-28T00:00:00",
          "2019-05-28T00:00:00",
          "2019-04-28T00:00:00",
          "2019-05-28T00:00:00"
         ],
         "xaxis": "x",
         "y": [
          809870,
          51270,
          89020,
          58400,
          30410,
          6330,
          27640,
          3640,
          50570,
          4080,
          24390,
          3470,
          2860,
          4070,
          50430,
          2780,
          4100,
          24840,
          3450,
          3240,
          60640,
          2280,
          1370,
          29590,
          2040,
          4400,
          61240,
          3720,
          1160,
          2850,
          1750,
          5500,
          24180,
          4110,
          3340,
          52070,
          2700,
          1970,
          1930,
          2300,
          22530,
          3540,
          4590,
          2790,
          58710,
          3110,
          1660,
          2740,
          1880,
          1800,
          19000,
          2210,
          2750,
          4050,
          72480,
          2820,
          2470,
          3990,
          2470,
          2910,
          2900,
          5920,
          19790,
          2570,
          3310,
          73890,
          1770,
          3080,
          2020,
          2690,
          4950,
          3170,
          3420,
          21750,
          3600,
          6910,
          81690,
          3280,
          1810,
          3280,
          3280,
          2770,
          1960,
          2730,
          2660,
          2610,
          7790,
          21460,
          4380,
          3540,
          64490,
          3770,
          2820,
          4000,
          5230,
          2980,
          3450,
          3620,
          2840,
          2730,
          15940,
          5220,
          5180,
          5410,
          68580,
          8450,
          4630,
          3600,
          3130,
          2820,
          4380,
          4450,
          3650,
          4180,
          4090,
          66300,
          15660,
          2900,
          2610,
          5410,
          18290,
          2430,
          2770,
          6280,
          7580,
          3770,
          4480,
          3650,
          5990,
          2380,
          3990,
          4060,
          4670,
          59690,
          3460,
          3310,
          2447070,
          125810,
          111430,
          88660,
          84780,
          89130,
          189430,
          196360,
          227350,
          254950,
          154530,
          102610,
          88070,
          87140,
          91740,
          85050,
          149540,
          3547820,
          4976530,
          3692160,
          3841350,
          4353500,
          4757910,
          5116740,
          3452350,
          3658700,
          3733180,
          3991190,
          4139730,
          4601480,
          4736370,
          4772550,
          5056450,
          66110,
          60760,
          79560,
          72250,
          53450,
          98230,
          60950,
          47200,
          70330,
          65030,
          76590,
          47390,
          92600,
          88910,
          201050,
          15020,
          21020,
          17050,
          26720,
          22530,
          19670,
          21460,
          16380,
          16420,
          22040,
          26530,
          17390,
          19850,
          18310,
          8100,
          8860,
          10500,
          11160,
          12080,
          10920,
          8270,
          15190,
          11730,
          11010,
          9780,
          12480,
          10240,
          6420,
          6040,
          7040,
          7350,
          8600,
          5630,
          5640,
          8620,
          9540,
          9970,
          5310,
          6550,
          9000,
          3730,
          5920,
          5690,
          5210,
          5370,
          6700,
          4420,
          5790,
          11060,
          4180,
          3430,
          3760,
          3390,
          4130,
          4310,
          4890,
          3420,
          5100,
          4240,
          4130,
          3620,
          4200,
          2730,
          2650,
          2900,
          4700,
          2680,
          2070,
          3120,
          2740,
          3160,
          2740,
          2320,
          2610,
          3460,
          2420,
          2220,
          3730,
          1870,
          1820,
          1690,
          2490,
          1820,
          2310,
          3190,
          1710,
          1390,
          2260,
          2000,
          1400,
          1900,
          2140,
          1610,
          1520,
          2550,
          860,
          1480,
          970,
          580,
          1200,
          520,
          950,
          620,
          850,
          790
         ],
         "yaxis": "y"
        }
       ],
       "layout": {
        "barmode": "relative",
        "coloraxis": {
         "colorbar": {
          "title": {
           "text": "days_between"
          }
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
         ]
        },
        "legend": {
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
   "metadata": {
    "azdata_cell_guid": "1f009253-b07d-46f4-b7b6-498e8869ca98"
   }
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "source": [
    "\r\n",
    "fullinv=full_3[full_3['Familia_prod']=='Inversión']\r\n",
    "fullahorro=full_3[full_3['Familia_prod']=='AhorroVista']\r\n",
    "fullcredit=full_3[full_3['Familia_prod']=='Crédito']"
   ],
   "outputs": [],
   "metadata": {
    "azdata_cell_guid": "a4349776-08f2-445a-9a93-14f7a8265951"
   }
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "source": [
    "grafico_horizontal= fullinv.groupby(['Productos',\"date\"])[\"recurrencia\"].mean().reset_index()\r\n",
    "grafico_horizontal\r\n",
    "\r\n",
    "evolucion_horizontal= px.bar(grafico_horizontal, x=\"date\", y=\"Ingresos\", \\\r\n",
    "                        color='Productos', orientation=\"v\", color_discrete_sequence=px.colors.cyclical.mygbm)\r\n",
    "\r\n",
    "evolucion_horizontal.show()"
   ],
   "outputs": [
    {
     "output_type": "error",
     "ename": "ValueError",
     "evalue": "Value of 'y' is not the name of a column in 'data_frame'. Expected one of ['Productos', 'date', 'recurrencia'] but received: Ingresos",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mValueError\u001b[0m                                Traceback (most recent call last)",
      "\u001b[1;32m<ipython-input-16-c27d1ffbb084>\u001b[0m in \u001b[0;36m<module>\u001b[1;34m\u001b[0m\n\u001b[0;32m      2\u001b[0m \u001b[0mgrafico_horizontal\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m      3\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m----> 4\u001b[1;33m evolucion_horizontal= px.bar(grafico_horizontal, x=\"date\", y=\"Ingresos\", \\\n\u001b[0m\u001b[0;32m      5\u001b[0m                         color='Productos', orientation=\"v\", color_discrete_sequence=px.colors.cyclical.mygbm)\n\u001b[0;32m      6\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;32m~\\anaconda3\\lib\\site-packages\\plotly\\express\\_chart_types.py\u001b[0m in \u001b[0;36mbar\u001b[1;34m(data_frame, x, y, color, facet_row, facet_col, facet_col_wrap, facet_row_spacing, facet_col_spacing, hover_name, hover_data, custom_data, text, base, error_x, error_x_minus, error_y, error_y_minus, animation_frame, animation_group, category_orders, labels, color_discrete_sequence, color_discrete_map, color_continuous_scale, range_color, color_continuous_midpoint, opacity, orientation, barmode, log_x, log_y, range_x, range_y, title, template, width, height)\u001b[0m\n\u001b[0;32m    348\u001b[0m     \u001b[0mmark\u001b[0m\u001b[1;33m.\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    349\u001b[0m     \"\"\"\n\u001b[1;32m--> 350\u001b[1;33m     return make_figure(\n\u001b[0m\u001b[0;32m    351\u001b[0m         \u001b[0margs\u001b[0m\u001b[1;33m=\u001b[0m\u001b[0mlocals\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m,\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    352\u001b[0m         \u001b[0mconstructor\u001b[0m\u001b[1;33m=\u001b[0m\u001b[0mgo\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mBar\u001b[0m\u001b[1;33m,\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;32m~\\anaconda3\\lib\\site-packages\\plotly\\express\\_core.py\u001b[0m in \u001b[0;36mmake_figure\u001b[1;34m(args, constructor, trace_patch, layout_patch)\u001b[0m\n\u001b[0;32m   1859\u001b[0m     \u001b[0mapply_default_cascade\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0margs\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m   1860\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m-> 1861\u001b[1;33m     \u001b[0margs\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mbuild_dataframe\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0margs\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mconstructor\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m   1862\u001b[0m     \u001b[1;32mif\u001b[0m \u001b[0mconstructor\u001b[0m \u001b[1;32min\u001b[0m \u001b[1;33m[\u001b[0m\u001b[0mgo\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mTreemap\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mgo\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mSunburst\u001b[0m\u001b[1;33m]\u001b[0m \u001b[1;32mand\u001b[0m \u001b[0margs\u001b[0m\u001b[1;33m[\u001b[0m\u001b[1;34m\"path\"\u001b[0m\u001b[1;33m]\u001b[0m \u001b[1;32mis\u001b[0m \u001b[1;32mnot\u001b[0m \u001b[1;32mNone\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m   1863\u001b[0m         \u001b[0margs\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mprocess_dataframe_hierarchy\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0margs\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;32m~\\anaconda3\\lib\\site-packages\\plotly\\express\\_core.py\u001b[0m in \u001b[0;36mbuild_dataframe\u001b[1;34m(args, constructor)\u001b[0m\n\u001b[0;32m   1375\u001b[0m     \u001b[1;31m# now that things have been prepped, we do the systematic rewriting of `args`\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m   1376\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m-> 1377\u001b[1;33m     df_output, wide_id_vars = process_args_into_dataframe(\n\u001b[0m\u001b[0;32m   1378\u001b[0m         \u001b[0margs\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mwide_mode\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mvar_name\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mvalue_name\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m   1379\u001b[0m     )\n",
      "\u001b[1;32m~\\anaconda3\\lib\\site-packages\\plotly\\express\\_core.py\u001b[0m in \u001b[0;36mprocess_args_into_dataframe\u001b[1;34m(args, wide_mode, var_name, value_name)\u001b[0m\n\u001b[0;32m   1181\u001b[0m                         \u001b[1;32mif\u001b[0m \u001b[0margument\u001b[0m \u001b[1;33m==\u001b[0m \u001b[1;34m\"index\"\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m   1182\u001b[0m                             \u001b[0merr_msg\u001b[0m \u001b[1;33m+=\u001b[0m \u001b[1;34m\"\\n To use the index, pass it in directly as `df.index`.\"\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m-> 1183\u001b[1;33m                         \u001b[1;32mraise\u001b[0m \u001b[0mValueError\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0merr_msg\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m   1184\u001b[0m                 \u001b[1;32melif\u001b[0m \u001b[0mlength\u001b[0m \u001b[1;32mand\u001b[0m \u001b[0mlen\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mdf_input\u001b[0m\u001b[1;33m[\u001b[0m\u001b[0margument\u001b[0m\u001b[1;33m]\u001b[0m\u001b[1;33m)\u001b[0m \u001b[1;33m!=\u001b[0m \u001b[0mlength\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m   1185\u001b[0m                     raise ValueError(\n",
      "\u001b[1;31mValueError\u001b[0m: Value of 'y' is not the name of a column in 'data_frame'. Expected one of ['Productos', 'date', 'recurrencia'] but received: Ingresos"
     ]
    }
   ],
   "metadata": {
    "azdata_cell_guid": "00602340-d449-4f11-88de-35a72bd3ed73"
   }
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "source": [
    " #full_3.info(verbose=True)"
   ],
   "outputs": [],
   "metadata": {
    "azdata_cell_guid": "645aa984-054b-4586-9984-38fe364d27f6"
   }
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "source": [
    "'''full_3.drop(['date','entry_date'],axis=1,inplace=True)'''"
   ],
   "outputs": [
    {
     "output_type": "execute_result",
     "data": {
      "text/plain": [
       "\"full_3.drop(['date','entry_date'],axis=1,inplace=True)\""
      ]
     },
     "metadata": {},
     "execution_count": 19
    }
   ],
   "metadata": {
    "azdata_cell_guid": "863e7d85-ad74-4944-b48d-dd7946038ac0"
   }
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "source": [
    "'''full_3['Familia_prod_Crédito']'''"
   ],
   "outputs": [
    {
     "output_type": "execute_result",
     "data": {
      "text/plain": [
       "\"full_3['Familia_prod_Crédito']\""
      ]
     },
     "metadata": {},
     "execution_count": 20
    }
   ],
   "metadata": {
    "azdata_cell_guid": "ea3e5a45-2fb0-4881-ae61-8d80a5e07d18"
   }
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "source": [
    "#type('days_between')"
   ],
   "outputs": [],
   "metadata": {
    "azdata_cell_guid": "229797f1-e86b-4a33-8b83-139571551ac9"
   }
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "source": [
    "#(full_3.corr()).style.background_gradient(cmap=\"coolwarm\")"
   ],
   "outputs": [],
   "metadata": {
    "azdata_cell_guid": "547a0fe8-9ac7-40f6-a083-f9185af7b526"
   }
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "source": [
    "'''full_3OHE=full_3\r\n",
    "pd.to_pickle(full_3OHE, './full_3OHE.pickle')'''"
   ],
   "outputs": [
    {
     "output_type": "execute_result",
     "data": {
      "text/plain": [
       "\"full_3OHE=full_3\\npd.to_pickle(full_3OHE, './full_3OHE.pickle')\""
      ]
     },
     "metadata": {},
     "execution_count": 23
    }
   ],
   "metadata": {
    "azdata_cell_guid": "d823b65b-ad06-4f7e-bf4b-a1cd4591af7b"
   }
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "source": [
    "full_4 = pd.read_pickle('./full_4.pickle')"
   ],
   "outputs": [],
   "metadata": {
    "azdata_cell_guid": "9ab31fa1-fdd6-49a1-a3de-db38bc3778f8"
   }
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "source": [
    "full_4.T"
   ],
   "outputs": [
    {
     "output_type": "execute_result",
     "data": {
      "text/plain": [
       "pk_cid                         1000028   1000113   1000157   1000162  \\\n",
       "Ingresos                      0.419979 -0.696691 -0.003585 -0.234620   \n",
       "Compras                       1.224240 -1.119262  0.335325 -0.149537   \n",
       "Age                           1.085781  1.898254  1.153487  0.491472   \n",
       "días_encartera                0.301552  0.685468  0.965934 -0.606514   \n",
       "days_between                 -0.758598 -1.438310 -0.301998  0.579700   \n",
       "Productos_credit_card        -0.128798 -0.128798 -0.128798 -0.128798   \n",
       "Productos_debit_card          3.606298 -0.376418  1.263524 -0.376418   \n",
       "Productos_em_account_p       -0.002389 -0.002389 -0.002389 -0.002389   \n",
       "Productos_em_acount           0.602336 -1.810554  0.429987  0.602336   \n",
       "Productos_emc_account        -0.257948  0.294906 -0.257948 -0.257948   \n",
       "Productos_funds              -0.063353 -0.063353 -0.063353 -0.063353   \n",
       "Productos_loans              -0.010046 -0.010046 -0.010046 -0.010046   \n",
       "Productos_long_term_deposit  -0.150412 -0.150412 -0.150412 -0.150412   \n",
       "Productos_mortgage           -0.007769 -0.007769 -0.007769 -0.007769   \n",
       "Productos_payroll_account    -0.275955 -0.275955 -0.275955 -0.275955   \n",
       "Productos_pension_plan       -0.237815 -0.237815 -0.237815 -0.237815   \n",
       "Productos_securities         -0.067806 -0.067806 -0.067806 -0.067806   \n",
       "Productos_short_term_deposit -0.120429 -0.120429 -0.120429 -0.120429   \n",
       "Familia_prod_AhorroVista      1.730929 -1.324754  0.571877 -0.060333   \n",
       "Familia_prod_Crédito         -0.128604 -0.128604 -0.128604 -0.128604   \n",
       "Familia_prod_Inversión       -0.286732 -0.286732 -0.286732 -0.286732   \n",
       "cluster                       5.000000  5.000000  5.000000  5.000000   \n",
       "\n",
       "pk_cid                          1000217   1000306   1000385   1000386  \\\n",
       "Ingresos                       2.229755  2.191249 -0.196115 -0.196115   \n",
       "Compras                        0.092894  2.759638 -0.068727 -0.068727   \n",
       "Age                            1.070735 -0.565205 -0.555095 -0.555095   \n",
       "días_encartera                 0.616501  0.432589  1.984347  1.984347   \n",
       "days_between                  -0.013150 -1.284019  0.594809  0.594809   \n",
       "Productos_credit_card          1.144838 -0.128798 -0.128798 -0.128798   \n",
       "Productos_debit_card          -0.376418  3.606298 -0.376418 -0.376418   \n",
       "Productos_em_account_p        -0.002389 -0.002389 -0.002389 -0.002389   \n",
       "Productos_em_acount           -2.155253 -2.155253  0.774685  0.774685   \n",
       "Productos_emc_account         -0.257948 -0.257948 -0.257948 -0.257948   \n",
       "Productos_funds               -0.063353 -0.063353 -0.063353 -0.063353   \n",
       "Productos_loans               -0.010046 -0.010046 -0.010046 -0.010046   \n",
       "Productos_long_term_deposit   -0.150412 -0.150412 -0.150412 -0.150412   \n",
       "Productos_mortgage            -0.007769 -0.007769 -0.007769 -0.007769   \n",
       "Productos_payroll_account     -0.275955  4.713824 -0.275955 -0.275955   \n",
       "Productos_pension_plan        -0.237815  3.205397 -0.237815 -0.237815   \n",
       "Productos_securities          18.180365 -0.067806 -0.067806 -0.067806   \n",
       "Productos_short_term_deposit  -0.120429 -0.120429 -0.120429 -0.120429   \n",
       "Familia_prod_AhorroVista      -1.746227  2.784613  0.045035  0.045035   \n",
       "Familia_prod_Crédito           1.129054 -0.128604 -0.128604 -0.128604   \n",
       "Familia_prod_Inversión         4.254561  2.117482 -0.286732 -0.286732   \n",
       "cluster                        5.000000  5.000000  5.000000  5.000000   \n",
       "\n",
       "pk_cid                         1000387   1000388  ...    998250    998252  \\\n",
       "Ingresos                     -0.196115 -0.196115  ...  2.075732 -0.234620   \n",
       "Compras                      -0.068727 -0.068727  ...  2.274775 -0.149537   \n",
       "Age                          -0.555095 -0.321443  ...  0.833274  0.931092   \n",
       "días_encartera                1.984347  1.984347  ...  1.910782  1.919978   \n",
       "days_between                  0.594809  0.594809  ... -1.875413  0.769271   \n",
       "Productos_credit_card        -0.128798 -0.128798  ...  3.692111 -0.128798   \n",
       "Productos_debit_card         -0.376418 -0.376418  ...  1.029247 -0.376418   \n",
       "Productos_em_account_p       -0.002389 -0.002389  ... -0.002389 -0.002389   \n",
       "Productos_em_acount           0.774685  0.774685  ... -2.155253 -2.155253   \n",
       "Productos_emc_account        -0.257948 -0.257948  ...  4.441318  4.164890   \n",
       "Productos_funds              -0.063353 -0.063353  ... -0.063353 -0.063353   \n",
       "Productos_loans              -0.010046 -0.010046  ... -0.010046 -0.010046   \n",
       "Productos_long_term_deposit  -0.150412 -0.150412  ... -0.150412 -0.150412   \n",
       "Productos_mortgage           -0.007769 -0.007769  ... -0.007769 -0.007769   \n",
       "Productos_payroll_account    -0.275955 -0.275955  ...  4.713824 -0.275955   \n",
       "Productos_pension_plan       -0.237815 -0.237815  ... -0.237815 -0.237815   \n",
       "Productos_securities         -0.067806 -0.067806  ... -0.067806 -0.067806   \n",
       "Productos_short_term_deposit -0.120429 -0.120429  ... -0.120429 -0.120429   \n",
       "Familia_prod_AhorroVista      0.045035  0.045035  ...  2.468508 -0.060333   \n",
       "Familia_prod_Crédito         -0.128604 -0.128604  ...  3.644370 -0.128604   \n",
       "Familia_prod_Inversión       -0.286732 -0.286732  ... -0.286732 -0.286732   \n",
       "cluster                       5.000000  5.000000  ...  5.000000  5.000000   \n",
       "\n",
       "pk_cid                          998325     998330     998771    998837  \\\n",
       "Ingresos                     -0.196115   3.076884   3.076884 -0.350138   \n",
       "Compras                      -0.068727  -0.068727  -0.068727 -0.391968   \n",
       "Age                           0.267556   0.759199   0.364911  0.504203   \n",
       "días_encartera                0.710756   1.763653   1.708479 -0.822610   \n",
       "days_between                  0.594809   0.594809   0.594809  0.550536   \n",
       "Productos_credit_card        -0.128798  10.697111  10.697111 -0.128798   \n",
       "Productos_debit_card         -0.376418  -0.376418  -0.376418 -0.376418   \n",
       "Productos_em_account_p       -0.002389  -0.002389  -0.002389 -0.002389   \n",
       "Productos_em_acount           0.774685  -2.155253  -2.155253  0.085288   \n",
       "Productos_emc_account        -0.257948  -0.257948  -0.257948 -0.257948   \n",
       "Productos_funds              -0.063353  -0.063353  -0.063353 -0.063353   \n",
       "Productos_loans              -0.010046  -0.010046  -0.010046 -0.010046   \n",
       "Productos_long_term_deposit  -0.150412  -0.150412  -0.150412 -0.150412   \n",
       "Productos_mortgage           -0.007769  -0.007769  -0.007769 -0.007769   \n",
       "Productos_payroll_account    -0.275955  -0.275955  -0.275955 -0.275955   \n",
       "Productos_pension_plan       -0.237815  -0.237815  -0.237815 -0.237815   \n",
       "Productos_securities         -0.067806  -0.067806  -0.067806 -0.067806   \n",
       "Productos_short_term_deposit -0.120429  -0.120429  -0.120429 -0.120429   \n",
       "Familia_prod_AhorroVista      0.045035  -1.746227  -1.746227 -0.376439   \n",
       "Familia_prod_Crédito         -0.128604  10.561487  10.561487 -0.128604   \n",
       "Familia_prod_Inversión       -0.286732  -0.286732  -0.286732 -0.286732   \n",
       "cluster                       5.000000   5.000000   5.000000  5.000000   \n",
       "\n",
       "pk_cid                          998859    999825     999835    999892  \n",
       "Ingresos                      4.501601 -0.196115   3.808496  2.383779  \n",
       "Compras                       5.183950 -0.068727   1.062619  4.375846  \n",
       "Age                           1.076791  0.768935   0.112415  0.588016  \n",
       "días_encartera                1.701582  0.306150   1.037200  0.110743  \n",
       "days_between                 -2.123807  0.594809  -2.010375 -1.537512  \n",
       "Productos_credit_card         1.781657 -0.128798  -0.128798 -0.128798  \n",
       "Productos_debit_card          3.606298 -0.376418  -0.376418  3.606298  \n",
       "Productos_em_account_p       -0.002389 -0.002389  -0.002389 -0.002389  \n",
       "Productos_em_acount          -2.155253  0.774685  -1.982903  0.774685  \n",
       "Productos_emc_account         4.441318 -0.257948  -0.257948  4.441318  \n",
       "Productos_funds              -0.063353 -0.063353  15.398697 -0.063353  \n",
       "Productos_loans              -0.010046 -0.010046  -0.010046 -0.010046  \n",
       "Productos_long_term_deposit  -0.150412 -0.150412   7.179482  1.943844  \n",
       "Productos_mortgage           -0.007769 -0.007769  -0.007769 -0.007769  \n",
       "Productos_payroll_account     4.713824 -0.275955  -0.275955  4.713824  \n",
       "Productos_pension_plan        5.118293 -0.237815  -0.237815 -0.237815  \n",
       "Productos_securities         -0.067806 -0.067806  -0.067806 -0.067806  \n",
       "Productos_short_term_deposit -0.120429 -0.120429   5.361739 -0.120429  \n",
       "Familia_prod_AhorroVista      5.102717  0.045035  -1.640859  5.418822  \n",
       "Familia_prod_Crédito          1.757883 -0.128604  -0.128604 -0.128604  \n",
       "Familia_prod_Inversión        3.453156 -0.286732   7.727314  0.781808  \n",
       "cluster                       5.000000  5.000000   4.000000  5.000000  \n",
       "\n",
       "[22 rows x 350384 columns]"
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
       "      <th>pk_cid</th>\n",
       "      <th>1000028</th>\n",
       "      <th>1000113</th>\n",
       "      <th>1000157</th>\n",
       "      <th>1000162</th>\n",
       "      <th>1000217</th>\n",
       "      <th>1000306</th>\n",
       "      <th>1000385</th>\n",
       "      <th>1000386</th>\n",
       "      <th>1000387</th>\n",
       "      <th>1000388</th>\n",
       "      <th>...</th>\n",
       "      <th>998250</th>\n",
       "      <th>998252</th>\n",
       "      <th>998325</th>\n",
       "      <th>998330</th>\n",
       "      <th>998771</th>\n",
       "      <th>998837</th>\n",
       "      <th>998859</th>\n",
       "      <th>999825</th>\n",
       "      <th>999835</th>\n",
       "      <th>999892</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>Ingresos</th>\n",
       "      <td>0.419979</td>\n",
       "      <td>-0.696691</td>\n",
       "      <td>-0.003585</td>\n",
       "      <td>-0.234620</td>\n",
       "      <td>2.229755</td>\n",
       "      <td>2.191249</td>\n",
       "      <td>-0.196115</td>\n",
       "      <td>-0.196115</td>\n",
       "      <td>-0.196115</td>\n",
       "      <td>-0.196115</td>\n",
       "      <td>...</td>\n",
       "      <td>2.075732</td>\n",
       "      <td>-0.234620</td>\n",
       "      <td>-0.196115</td>\n",
       "      <td>3.076884</td>\n",
       "      <td>3.076884</td>\n",
       "      <td>-0.350138</td>\n",
       "      <td>4.501601</td>\n",
       "      <td>-0.196115</td>\n",
       "      <td>3.808496</td>\n",
       "      <td>2.383779</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Compras</th>\n",
       "      <td>1.224240</td>\n",
       "      <td>-1.119262</td>\n",
       "      <td>0.335325</td>\n",
       "      <td>-0.149537</td>\n",
       "      <td>0.092894</td>\n",
       "      <td>2.759638</td>\n",
       "      <td>-0.068727</td>\n",
       "      <td>-0.068727</td>\n",
       "      <td>-0.068727</td>\n",
       "      <td>-0.068727</td>\n",
       "      <td>...</td>\n",
       "      <td>2.274775</td>\n",
       "      <td>-0.149537</td>\n",
       "      <td>-0.068727</td>\n",
       "      <td>-0.068727</td>\n",
       "      <td>-0.068727</td>\n",
       "      <td>-0.391968</td>\n",
       "      <td>5.183950</td>\n",
       "      <td>-0.068727</td>\n",
       "      <td>1.062619</td>\n",
       "      <td>4.375846</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Age</th>\n",
       "      <td>1.085781</td>\n",
       "      <td>1.898254</td>\n",
       "      <td>1.153487</td>\n",
       "      <td>0.491472</td>\n",
       "      <td>1.070735</td>\n",
       "      <td>-0.565205</td>\n",
       "      <td>-0.555095</td>\n",
       "      <td>-0.555095</td>\n",
       "      <td>-0.555095</td>\n",
       "      <td>-0.321443</td>\n",
       "      <td>...</td>\n",
       "      <td>0.833274</td>\n",
       "      <td>0.931092</td>\n",
       "      <td>0.267556</td>\n",
       "      <td>0.759199</td>\n",
       "      <td>0.364911</td>\n",
       "      <td>0.504203</td>\n",
       "      <td>1.076791</td>\n",
       "      <td>0.768935</td>\n",
       "      <td>0.112415</td>\n",
       "      <td>0.588016</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>días_encartera</th>\n",
       "      <td>0.301552</td>\n",
       "      <td>0.685468</td>\n",
       "      <td>0.965934</td>\n",
       "      <td>-0.606514</td>\n",
       "      <td>0.616501</td>\n",
       "      <td>0.432589</td>\n",
       "      <td>1.984347</td>\n",
       "      <td>1.984347</td>\n",
       "      <td>1.984347</td>\n",
       "      <td>1.984347</td>\n",
       "      <td>...</td>\n",
       "      <td>1.910782</td>\n",
       "      <td>1.919978</td>\n",
       "      <td>0.710756</td>\n",
       "      <td>1.763653</td>\n",
       "      <td>1.708479</td>\n",
       "      <td>-0.822610</td>\n",
       "      <td>1.701582</td>\n",
       "      <td>0.306150</td>\n",
       "      <td>1.037200</td>\n",
       "      <td>0.110743</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>days_between</th>\n",
       "      <td>-0.758598</td>\n",
       "      <td>-1.438310</td>\n",
       "      <td>-0.301998</td>\n",
       "      <td>0.579700</td>\n",
       "      <td>-0.013150</td>\n",
       "      <td>-1.284019</td>\n",
       "      <td>0.594809</td>\n",
       "      <td>0.594809</td>\n",
       "      <td>0.594809</td>\n",
       "      <td>0.594809</td>\n",
       "      <td>...</td>\n",
       "      <td>-1.875413</td>\n",
       "      <td>0.769271</td>\n",
       "      <td>0.594809</td>\n",
       "      <td>0.594809</td>\n",
       "      <td>0.594809</td>\n",
       "      <td>0.550536</td>\n",
       "      <td>-2.123807</td>\n",
       "      <td>0.594809</td>\n",
       "      <td>-2.010375</td>\n",
       "      <td>-1.537512</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Productos_credit_card</th>\n",
       "      <td>-0.128798</td>\n",
       "      <td>-0.128798</td>\n",
       "      <td>-0.128798</td>\n",
       "      <td>-0.128798</td>\n",
       "      <td>1.144838</td>\n",
       "      <td>-0.128798</td>\n",
       "      <td>-0.128798</td>\n",
       "      <td>-0.128798</td>\n",
       "      <td>-0.128798</td>\n",
       "      <td>-0.128798</td>\n",
       "      <td>...</td>\n",
       "      <td>3.692111</td>\n",
       "      <td>-0.128798</td>\n",
       "      <td>-0.128798</td>\n",
       "      <td>10.697111</td>\n",
       "      <td>10.697111</td>\n",
       "      <td>-0.128798</td>\n",
       "      <td>1.781657</td>\n",
       "      <td>-0.128798</td>\n",
       "      <td>-0.128798</td>\n",
       "      <td>-0.128798</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Productos_debit_card</th>\n",
       "      <td>3.606298</td>\n",
       "      <td>-0.376418</td>\n",
       "      <td>1.263524</td>\n",
       "      <td>-0.376418</td>\n",
       "      <td>-0.376418</td>\n",
       "      <td>3.606298</td>\n",
       "      <td>-0.376418</td>\n",
       "      <td>-0.376418</td>\n",
       "      <td>-0.376418</td>\n",
       "      <td>-0.376418</td>\n",
       "      <td>...</td>\n",
       "      <td>1.029247</td>\n",
       "      <td>-0.376418</td>\n",
       "      <td>-0.376418</td>\n",
       "      <td>-0.376418</td>\n",
       "      <td>-0.376418</td>\n",
       "      <td>-0.376418</td>\n",
       "      <td>3.606298</td>\n",
       "      <td>-0.376418</td>\n",
       "      <td>-0.376418</td>\n",
       "      <td>3.606298</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Productos_em_account_p</th>\n",
       "      <td>-0.002389</td>\n",
       "      <td>-0.002389</td>\n",
       "      <td>-0.002389</td>\n",
       "      <td>-0.002389</td>\n",
       "      <td>-0.002389</td>\n",
       "      <td>-0.002389</td>\n",
       "      <td>-0.002389</td>\n",
       "      <td>-0.002389</td>\n",
       "      <td>-0.002389</td>\n",
       "      <td>-0.002389</td>\n",
       "      <td>...</td>\n",
       "      <td>-0.002389</td>\n",
       "      <td>-0.002389</td>\n",
       "      <td>-0.002389</td>\n",
       "      <td>-0.002389</td>\n",
       "      <td>-0.002389</td>\n",
       "      <td>-0.002389</td>\n",
       "      <td>-0.002389</td>\n",
       "      <td>-0.002389</td>\n",
       "      <td>-0.002389</td>\n",
       "      <td>-0.002389</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Productos_em_acount</th>\n",
       "      <td>0.602336</td>\n",
       "      <td>-1.810554</td>\n",
       "      <td>0.429987</td>\n",
       "      <td>0.602336</td>\n",
       "      <td>-2.155253</td>\n",
       "      <td>-2.155253</td>\n",
       "      <td>0.774685</td>\n",
       "      <td>0.774685</td>\n",
       "      <td>0.774685</td>\n",
       "      <td>0.774685</td>\n",
       "      <td>...</td>\n",
       "      <td>-2.155253</td>\n",
       "      <td>-2.155253</td>\n",
       "      <td>0.774685</td>\n",
       "      <td>-2.155253</td>\n",
       "      <td>-2.155253</td>\n",
       "      <td>0.085288</td>\n",
       "      <td>-2.155253</td>\n",
       "      <td>0.774685</td>\n",
       "      <td>-1.982903</td>\n",
       "      <td>0.774685</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Productos_emc_account</th>\n",
       "      <td>-0.257948</td>\n",
       "      <td>0.294906</td>\n",
       "      <td>-0.257948</td>\n",
       "      <td>-0.257948</td>\n",
       "      <td>-0.257948</td>\n",
       "      <td>-0.257948</td>\n",
       "      <td>-0.257948</td>\n",
       "      <td>-0.257948</td>\n",
       "      <td>-0.257948</td>\n",
       "      <td>-0.257948</td>\n",
       "      <td>...</td>\n",
       "      <td>4.441318</td>\n",
       "      <td>4.164890</td>\n",
       "      <td>-0.257948</td>\n",
       "      <td>-0.257948</td>\n",
       "      <td>-0.257948</td>\n",
       "      <td>-0.257948</td>\n",
       "      <td>4.441318</td>\n",
       "      <td>-0.257948</td>\n",
       "      <td>-0.257948</td>\n",
       "      <td>4.441318</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Productos_funds</th>\n",
       "      <td>-0.063353</td>\n",
       "      <td>-0.063353</td>\n",
       "      <td>-0.063353</td>\n",
       "      <td>-0.063353</td>\n",
       "      <td>-0.063353</td>\n",
       "      <td>-0.063353</td>\n",
       "      <td>-0.063353</td>\n",
       "      <td>-0.063353</td>\n",
       "      <td>-0.063353</td>\n",
       "      <td>-0.063353</td>\n",
       "      <td>...</td>\n",
       "      <td>-0.063353</td>\n",
       "      <td>-0.063353</td>\n",
       "      <td>-0.063353</td>\n",
       "      <td>-0.063353</td>\n",
       "      <td>-0.063353</td>\n",
       "      <td>-0.063353</td>\n",
       "      <td>-0.063353</td>\n",
       "      <td>-0.063353</td>\n",
       "      <td>15.398697</td>\n",
       "      <td>-0.063353</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Productos_loans</th>\n",
       "      <td>-0.010046</td>\n",
       "      <td>-0.010046</td>\n",
       "      <td>-0.010046</td>\n",
       "      <td>-0.010046</td>\n",
       "      <td>-0.010046</td>\n",
       "      <td>-0.010046</td>\n",
       "      <td>-0.010046</td>\n",
       "      <td>-0.010046</td>\n",
       "      <td>-0.010046</td>\n",
       "      <td>-0.010046</td>\n",
       "      <td>...</td>\n",
       "      <td>-0.010046</td>\n",
       "      <td>-0.010046</td>\n",
       "      <td>-0.010046</td>\n",
       "      <td>-0.010046</td>\n",
       "      <td>-0.010046</td>\n",
       "      <td>-0.010046</td>\n",
       "      <td>-0.010046</td>\n",
       "      <td>-0.010046</td>\n",
       "      <td>-0.010046</td>\n",
       "      <td>-0.010046</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Productos_long_term_deposit</th>\n",
       "      <td>-0.150412</td>\n",
       "      <td>-0.150412</td>\n",
       "      <td>-0.150412</td>\n",
       "      <td>-0.150412</td>\n",
       "      <td>-0.150412</td>\n",
       "      <td>-0.150412</td>\n",
       "      <td>-0.150412</td>\n",
       "      <td>-0.150412</td>\n",
       "      <td>-0.150412</td>\n",
       "      <td>-0.150412</td>\n",
       "      <td>...</td>\n",
       "      <td>-0.150412</td>\n",
       "      <td>-0.150412</td>\n",
       "      <td>-0.150412</td>\n",
       "      <td>-0.150412</td>\n",
       "      <td>-0.150412</td>\n",
       "      <td>-0.150412</td>\n",
       "      <td>-0.150412</td>\n",
       "      <td>-0.150412</td>\n",
       "      <td>7.179482</td>\n",
       "      <td>1.943844</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Productos_mortgage</th>\n",
       "      <td>-0.007769</td>\n",
       "      <td>-0.007769</td>\n",
       "      <td>-0.007769</td>\n",
       "      <td>-0.007769</td>\n",
       "      <td>-0.007769</td>\n",
       "      <td>-0.007769</td>\n",
       "      <td>-0.007769</td>\n",
       "      <td>-0.007769</td>\n",
       "      <td>-0.007769</td>\n",
       "      <td>-0.007769</td>\n",
       "      <td>...</td>\n",
       "      <td>-0.007769</td>\n",
       "      <td>-0.007769</td>\n",
       "      <td>-0.007769</td>\n",
       "      <td>-0.007769</td>\n",
       "      <td>-0.007769</td>\n",
       "      <td>-0.007769</td>\n",
       "      <td>-0.007769</td>\n",
       "      <td>-0.007769</td>\n",
       "      <td>-0.007769</td>\n",
       "      <td>-0.007769</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Productos_payroll_account</th>\n",
       "      <td>-0.275955</td>\n",
       "      <td>-0.275955</td>\n",
       "      <td>-0.275955</td>\n",
       "      <td>-0.275955</td>\n",
       "      <td>-0.275955</td>\n",
       "      <td>4.713824</td>\n",
       "      <td>-0.275955</td>\n",
       "      <td>-0.275955</td>\n",
       "      <td>-0.275955</td>\n",
       "      <td>-0.275955</td>\n",
       "      <td>...</td>\n",
       "      <td>4.713824</td>\n",
       "      <td>-0.275955</td>\n",
       "      <td>-0.275955</td>\n",
       "      <td>-0.275955</td>\n",
       "      <td>-0.275955</td>\n",
       "      <td>-0.275955</td>\n",
       "      <td>4.713824</td>\n",
       "      <td>-0.275955</td>\n",
       "      <td>-0.275955</td>\n",
       "      <td>4.713824</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Productos_pension_plan</th>\n",
       "      <td>-0.237815</td>\n",
       "      <td>-0.237815</td>\n",
       "      <td>-0.237815</td>\n",
       "      <td>-0.237815</td>\n",
       "      <td>-0.237815</td>\n",
       "      <td>3.205397</td>\n",
       "      <td>-0.237815</td>\n",
       "      <td>-0.237815</td>\n",
       "      <td>-0.237815</td>\n",
       "      <td>-0.237815</td>\n",
       "      <td>...</td>\n",
       "      <td>-0.237815</td>\n",
       "      <td>-0.237815</td>\n",
       "      <td>-0.237815</td>\n",
       "      <td>-0.237815</td>\n",
       "      <td>-0.237815</td>\n",
       "      <td>-0.237815</td>\n",
       "      <td>5.118293</td>\n",
       "      <td>-0.237815</td>\n",
       "      <td>-0.237815</td>\n",
       "      <td>-0.237815</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Productos_securities</th>\n",
       "      <td>-0.067806</td>\n",
       "      <td>-0.067806</td>\n",
       "      <td>-0.067806</td>\n",
       "      <td>-0.067806</td>\n",
       "      <td>18.180365</td>\n",
       "      <td>-0.067806</td>\n",
       "      <td>-0.067806</td>\n",
       "      <td>-0.067806</td>\n",
       "      <td>-0.067806</td>\n",
       "      <td>-0.067806</td>\n",
       "      <td>...</td>\n",
       "      <td>-0.067806</td>\n",
       "      <td>-0.067806</td>\n",
       "      <td>-0.067806</td>\n",
       "      <td>-0.067806</td>\n",
       "      <td>-0.067806</td>\n",
       "      <td>-0.067806</td>\n",
       "      <td>-0.067806</td>\n",
       "      <td>-0.067806</td>\n",
       "      <td>-0.067806</td>\n",
       "      <td>-0.067806</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Productos_short_term_deposit</th>\n",
       "      <td>-0.120429</td>\n",
       "      <td>-0.120429</td>\n",
       "      <td>-0.120429</td>\n",
       "      <td>-0.120429</td>\n",
       "      <td>-0.120429</td>\n",
       "      <td>-0.120429</td>\n",
       "      <td>-0.120429</td>\n",
       "      <td>-0.120429</td>\n",
       "      <td>-0.120429</td>\n",
       "      <td>-0.120429</td>\n",
       "      <td>...</td>\n",
       "      <td>-0.120429</td>\n",
       "      <td>-0.120429</td>\n",
       "      <td>-0.120429</td>\n",
       "      <td>-0.120429</td>\n",
       "      <td>-0.120429</td>\n",
       "      <td>-0.120429</td>\n",
       "      <td>-0.120429</td>\n",
       "      <td>-0.120429</td>\n",
       "      <td>5.361739</td>\n",
       "      <td>-0.120429</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Familia_prod_AhorroVista</th>\n",
       "      <td>1.730929</td>\n",
       "      <td>-1.324754</td>\n",
       "      <td>0.571877</td>\n",
       "      <td>-0.060333</td>\n",
       "      <td>-1.746227</td>\n",
       "      <td>2.784613</td>\n",
       "      <td>0.045035</td>\n",
       "      <td>0.045035</td>\n",
       "      <td>0.045035</td>\n",
       "      <td>0.045035</td>\n",
       "      <td>...</td>\n",
       "      <td>2.468508</td>\n",
       "      <td>-0.060333</td>\n",
       "      <td>0.045035</td>\n",
       "      <td>-1.746227</td>\n",
       "      <td>-1.746227</td>\n",
       "      <td>-0.376439</td>\n",
       "      <td>5.102717</td>\n",
       "      <td>0.045035</td>\n",
       "      <td>-1.640859</td>\n",
       "      <td>5.418822</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Familia_prod_Crédito</th>\n",
       "      <td>-0.128604</td>\n",
       "      <td>-0.128604</td>\n",
       "      <td>-0.128604</td>\n",
       "      <td>-0.128604</td>\n",
       "      <td>1.129054</td>\n",
       "      <td>-0.128604</td>\n",
       "      <td>-0.128604</td>\n",
       "      <td>-0.128604</td>\n",
       "      <td>-0.128604</td>\n",
       "      <td>-0.128604</td>\n",
       "      <td>...</td>\n",
       "      <td>3.644370</td>\n",
       "      <td>-0.128604</td>\n",
       "      <td>-0.128604</td>\n",
       "      <td>10.561487</td>\n",
       "      <td>10.561487</td>\n",
       "      <td>-0.128604</td>\n",
       "      <td>1.757883</td>\n",
       "      <td>-0.128604</td>\n",
       "      <td>-0.128604</td>\n",
       "      <td>-0.128604</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Familia_prod_Inversión</th>\n",
       "      <td>-0.286732</td>\n",
       "      <td>-0.286732</td>\n",
       "      <td>-0.286732</td>\n",
       "      <td>-0.286732</td>\n",
       "      <td>4.254561</td>\n",
       "      <td>2.117482</td>\n",
       "      <td>-0.286732</td>\n",
       "      <td>-0.286732</td>\n",
       "      <td>-0.286732</td>\n",
       "      <td>-0.286732</td>\n",
       "      <td>...</td>\n",
       "      <td>-0.286732</td>\n",
       "      <td>-0.286732</td>\n",
       "      <td>-0.286732</td>\n",
       "      <td>-0.286732</td>\n",
       "      <td>-0.286732</td>\n",
       "      <td>-0.286732</td>\n",
       "      <td>3.453156</td>\n",
       "      <td>-0.286732</td>\n",
       "      <td>7.727314</td>\n",
       "      <td>0.781808</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>cluster</th>\n",
       "      <td>5.000000</td>\n",
       "      <td>5.000000</td>\n",
       "      <td>5.000000</td>\n",
       "      <td>5.000000</td>\n",
       "      <td>5.000000</td>\n",
       "      <td>5.000000</td>\n",
       "      <td>5.000000</td>\n",
       "      <td>5.000000</td>\n",
       "      <td>5.000000</td>\n",
       "      <td>5.000000</td>\n",
       "      <td>...</td>\n",
       "      <td>5.000000</td>\n",
       "      <td>5.000000</td>\n",
       "      <td>5.000000</td>\n",
       "      <td>5.000000</td>\n",
       "      <td>5.000000</td>\n",
       "      <td>5.000000</td>\n",
       "      <td>5.000000</td>\n",
       "      <td>5.000000</td>\n",
       "      <td>4.000000</td>\n",
       "      <td>5.000000</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>22 rows × 350384 columns</p>\n",
       "</div>"
      ]
     },
     "metadata": {},
     "execution_count": 30
    }
   ],
   "metadata": {}
  },
  {
   "cell_type": "code",
   "execution_count": 42,
   "source": [
    "confussion_matrix = pd.crosstab(full_3['recurrencia'], full_3['Familia_prod'])\r\n",
    "confussion_matrix.head(15)"
   ],
   "outputs": [
    {
     "output_type": "execute_result",
     "data": {
      "text/plain": [
       "Familia_prod  AhorroVista  Crédito  Inversión\n",
       "recurrencia                                  \n",
       "1                  326346     7216      16822\n",
       "2                  318930     6374      19633\n",
       "3                  316815     5655      18083\n",
       "4                  314258     5205      16271\n",
       "5                  311715     4785      15057\n",
       "6                  306526     4411      16004\n",
       "7                  302762     4123      14840\n",
       "8                  292084     3836      15567\n",
       "9                  276022     3575      14243\n",
       "10                 260106     3353      14350\n",
       "11                 248438     3108      13135\n",
       "12                 237303     2852      13307\n",
       "13                 235807     2601      12053\n",
       "14                 233146     2376      11923\n",
       "15                 231566     2095      10856"
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
       "      <th>Familia_prod</th>\n",
       "      <th>AhorroVista</th>\n",
       "      <th>Crédito</th>\n",
       "      <th>Inversión</th>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>recurrencia</th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>326346</td>\n",
       "      <td>7216</td>\n",
       "      <td>16822</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>318930</td>\n",
       "      <td>6374</td>\n",
       "      <td>19633</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>316815</td>\n",
       "      <td>5655</td>\n",
       "      <td>18083</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>314258</td>\n",
       "      <td>5205</td>\n",
       "      <td>16271</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>5</th>\n",
       "      <td>311715</td>\n",
       "      <td>4785</td>\n",
       "      <td>15057</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>6</th>\n",
       "      <td>306526</td>\n",
       "      <td>4411</td>\n",
       "      <td>16004</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>7</th>\n",
       "      <td>302762</td>\n",
       "      <td>4123</td>\n",
       "      <td>14840</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>8</th>\n",
       "      <td>292084</td>\n",
       "      <td>3836</td>\n",
       "      <td>15567</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>9</th>\n",
       "      <td>276022</td>\n",
       "      <td>3575</td>\n",
       "      <td>14243</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>10</th>\n",
       "      <td>260106</td>\n",
       "      <td>3353</td>\n",
       "      <td>14350</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>11</th>\n",
       "      <td>248438</td>\n",
       "      <td>3108</td>\n",
       "      <td>13135</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>12</th>\n",
       "      <td>237303</td>\n",
       "      <td>2852</td>\n",
       "      <td>13307</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>13</th>\n",
       "      <td>235807</td>\n",
       "      <td>2601</td>\n",
       "      <td>12053</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>14</th>\n",
       "      <td>233146</td>\n",
       "      <td>2376</td>\n",
       "      <td>11923</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>15</th>\n",
       "      <td>231566</td>\n",
       "      <td>2095</td>\n",
       "      <td>10856</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ]
     },
     "metadata": {},
     "execution_count": 42
    }
   ],
   "metadata": {
    "azdata_cell_guid": "7ca34afa-59f3-4746-8d59-8c7904e8b8ed"
   }
  },
  {
   "cell_type": "code",
   "execution_count": 43,
   "source": [
    "confussion_matrix_1 = pd.crosstab(full_3['pk_cid'], full_3['Familia_prod'].value_counts)\r\n",
    "confussion_matrix_1.head(15)"
   ],
   "outputs": [
    {
     "output_type": "execute_result",
     "data": {
      "text/plain": [
       "col_0    <bound method IndexOpsMixin.value_counts of 22134         Inversión\\n22135         Inversión\\n22136         Inversión\\n27826         Inversión\\n27827         Inversión\\n               ...     \\n89443855    AhorroVista\\n89443856    AhorroVista\\n89443857    AhorroVista\\n89443858    AhorroVista\\n89443859    AhorroVista\\nName: Familia_prod, Length: 6254518, dtype: object>\n",
       "pk_cid                                                                                                                                                                                                                                                                                                                                                                                     \n",
       "1000028                                                 33                                                                                                                                                                                                                                                                                                                                 \n",
       "1000113                                                  4                                                                                                                                                                                                                                                                                                                                 \n",
       "1000157                                                 22                                                                                                                                                                                                                                                                                                                                 \n",
       "1000162                                                 16                                                                                                                                                                                                                                                                                                                                 \n",
       "1000217                                                 19                                                                                                                                                                                                                                                                                                                                 \n",
       "1000306                                                 52                                                                                                                                                                                                                                                                                                                                 \n",
       "1000385                                                 17                                                                                                                                                                                                                                                                                                                                 \n",
       "1000386                                                 17                                                                                                                                                                                                                                                                                                                                 \n",
       "1000387                                                 17                                                                                                                                                                                                                                                                                                                                 \n",
       "1000388                                                 17                                                                                                                                                                                                                                                                                                                                 \n",
       "1000389                                                 17                                                                                                                                                                                                                                                                                                                                 \n",
       "1000390                                                 17                                                                                                                                                                                                                                                                                                                                 \n",
       "1000391                                                 31                                                                                                                                                                                                                                                                                                                                 \n",
       "1000393                                                 17                                                                                                                                                                                                                                                                                                                                 \n",
       "1000394                                                 17                                                                                                                                                                                                                                                                                                                                 "
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
       "      <th>col_0</th>\n",
       "      <th>&lt;bound method IndexOpsMixin.value_counts of 22134         Inversión\\n22135         Inversión\\n22136         Inversión\\n27826         Inversión\\n27827         Inversión\\n               ...     \\n89443855    AhorroVista\\n89443856    AhorroVista\\n89443857    AhorroVista\\n89443858    AhorroVista\\n89443859    AhorroVista\\nName: Familia_prod, Length: 6254518, dtype: object&gt;</th>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>pk_cid</th>\n",
       "      <th></th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>1000028</th>\n",
       "      <td>33</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1000113</th>\n",
       "      <td>4</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1000157</th>\n",
       "      <td>22</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1000162</th>\n",
       "      <td>16</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1000217</th>\n",
       "      <td>19</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1000306</th>\n",
       "      <td>52</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1000385</th>\n",
       "      <td>17</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1000386</th>\n",
       "      <td>17</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1000387</th>\n",
       "      <td>17</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1000388</th>\n",
       "      <td>17</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1000389</th>\n",
       "      <td>17</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1000390</th>\n",
       "      <td>17</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1000391</th>\n",
       "      <td>31</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1000393</th>\n",
       "      <td>17</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1000394</th>\n",
       "      <td>17</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ]
     },
     "metadata": {},
     "execution_count": 43
    }
   ],
   "metadata": {
    "azdata_cell_guid": "124f776f-4502-4722-b138-f4fb8e84db04"
   }
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "source": [
    "confussion_matrix_inversion = pd.crosstab(full_4['recurrencia'], full_4['Familia_prod_Inversión'].value_counts)\r\n",
    "confussion_matrix_inversion.head(15)"
   ],
   "outputs": [
    {
     "output_type": "error",
     "ename": "KeyError",
     "evalue": "'recurrencia'",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mKeyError\u001b[0m                                  Traceback (most recent call last)",
      "\u001b[1;32m~\\anaconda3\\lib\\site-packages\\pandas\\core\\indexes\\base.py\u001b[0m in \u001b[0;36mget_loc\u001b[1;34m(self, key, method, tolerance)\u001b[0m\n\u001b[0;32m   2894\u001b[0m             \u001b[1;32mtry\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m-> 2895\u001b[1;33m                 \u001b[1;32mreturn\u001b[0m \u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0m_engine\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mget_loc\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mcasted_key\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m   2896\u001b[0m             \u001b[1;32mexcept\u001b[0m \u001b[0mKeyError\u001b[0m \u001b[1;32mas\u001b[0m \u001b[0merr\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;32mpandas\\_libs\\index.pyx\u001b[0m in \u001b[0;36mpandas._libs.index.IndexEngine.get_loc\u001b[1;34m()\u001b[0m\n",
      "\u001b[1;32mpandas\\_libs\\index.pyx\u001b[0m in \u001b[0;36mpandas._libs.index.IndexEngine.get_loc\u001b[1;34m()\u001b[0m\n",
      "\u001b[1;32mpandas\\_libs\\hashtable_class_helper.pxi\u001b[0m in \u001b[0;36mpandas._libs.hashtable.PyObjectHashTable.get_item\u001b[1;34m()\u001b[0m\n",
      "\u001b[1;32mpandas\\_libs\\hashtable_class_helper.pxi\u001b[0m in \u001b[0;36mpandas._libs.hashtable.PyObjectHashTable.get_item\u001b[1;34m()\u001b[0m\n",
      "\u001b[1;31mKeyError\u001b[0m: 'recurrencia'",
      "\nThe above exception was the direct cause of the following exception:\n",
      "\u001b[1;31mKeyError\u001b[0m                                  Traceback (most recent call last)",
      "\u001b[1;32m<ipython-input-23-a9f99f107583>\u001b[0m in \u001b[0;36m<module>\u001b[1;34m\u001b[0m\n\u001b[1;32m----> 1\u001b[1;33m \u001b[0mconfussion_matrix_inversion\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mpd\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mcrosstab\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mfull_4\u001b[0m\u001b[1;33m[\u001b[0m\u001b[1;34m'recurrencia'\u001b[0m\u001b[1;33m]\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mfull_4\u001b[0m\u001b[1;33m[\u001b[0m\u001b[1;34m'Familia_prod_Inversión'\u001b[0m\u001b[1;33m]\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mvalue_counts\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m      2\u001b[0m \u001b[0mconfussion_matrix_inversion\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mhead\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;36m15\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;32m~\\anaconda3\\lib\\site-packages\\pandas\\core\\frame.py\u001b[0m in \u001b[0;36m__getitem__\u001b[1;34m(self, key)\u001b[0m\n\u001b[0;32m   2900\u001b[0m             \u001b[1;32mif\u001b[0m \u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mcolumns\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mnlevels\u001b[0m \u001b[1;33m>\u001b[0m \u001b[1;36m1\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m   2901\u001b[0m                 \u001b[1;32mreturn\u001b[0m \u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0m_getitem_multilevel\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mkey\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m-> 2902\u001b[1;33m             \u001b[0mindexer\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mcolumns\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mget_loc\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mkey\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m   2903\u001b[0m             \u001b[1;32mif\u001b[0m \u001b[0mis_integer\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mindexer\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m   2904\u001b[0m                 \u001b[0mindexer\u001b[0m \u001b[1;33m=\u001b[0m \u001b[1;33m[\u001b[0m\u001b[0mindexer\u001b[0m\u001b[1;33m]\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;32m~\\anaconda3\\lib\\site-packages\\pandas\\core\\indexes\\base.py\u001b[0m in \u001b[0;36mget_loc\u001b[1;34m(self, key, method, tolerance)\u001b[0m\n\u001b[0;32m   2895\u001b[0m                 \u001b[1;32mreturn\u001b[0m \u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0m_engine\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mget_loc\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mcasted_key\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m   2896\u001b[0m             \u001b[1;32mexcept\u001b[0m \u001b[0mKeyError\u001b[0m \u001b[1;32mas\u001b[0m \u001b[0merr\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m-> 2897\u001b[1;33m                 \u001b[1;32mraise\u001b[0m \u001b[0mKeyError\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mkey\u001b[0m\u001b[1;33m)\u001b[0m \u001b[1;32mfrom\u001b[0m \u001b[0merr\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m   2898\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m   2899\u001b[0m         \u001b[1;32mif\u001b[0m \u001b[0mtolerance\u001b[0m \u001b[1;32mis\u001b[0m \u001b[1;32mnot\u001b[0m \u001b[1;32mNone\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;31mKeyError\u001b[0m: 'recurrencia'"
     ]
    }
   ],
   "metadata": {
    "azdata_cell_guid": "4c085e59-da03-4de4-ac6e-31b4b84d81e8"
   }
  },
  {
   "cell_type": "markdown",
   "source": [
    "## SEGMENTACIÓN NO SUPERVISADA"
   ],
   "metadata": {}
  },
  {
   "cell_type": "markdown",
   "source": [],
   "metadata": {}
  },
  {
   "cell_type": "code",
   "execution_count": 33,
   "source": [
    "full_kmeans = full_4.groupby('pk_cid').agg(\r\n",
    "    Age = ('Age', 'mean'),\r\n",
    "    días_encartera=('días_encartera','max'),\r\n",
    "    days_between = ('days_between','mean'),               \r\n",
    "    Productos_credit_card = ('Productos_credit_card' ,'sum'),     \r\n",
    "    Productos_debit_card = ('Productos_debit_card','sum'),                \r\n",
    "    Productos_em_account_p = ('Productos_em_account_p','sum'),                 \r\n",
    "    Productos_em_acount = ('Productos_em_acount','sum'),             \r\n",
    "    Productos_emc_account=('Productos_emc_account','sum'),                  \r\n",
    "    Productos_funds=('Productos_funds','sum'),                       \r\n",
    "    Productos_loans= ('Productos_loans','sum'),                           \r\n",
    "    Productos_long_term_deposit= ('Productos_long_term_deposit','sum'),           \r\n",
    "    Productos_mortgage= ('Productos_mortgage','sum'),                     \r\n",
    "    Productos_payroll_account = ('Productos_payroll_account','sum'),             \r\n",
    "    Productos_pension_plan   = ('Productos_pension_plan','sum'),              \r\n",
    "    Productos_securities   = ('Productos_securities','sum'),                \r\n",
    "    Productos_short_term_deposit   = ('Productos_short_term_deposit','sum'),        \r\n",
    ")"
   ],
   "outputs": [],
   "metadata": {
    "azdata_cell_guid": "4f819664-5a78-4be9-b025-855c1ea1c1e5"
   }
  },
  {
   "cell_type": "code",
   "execution_count": 34,
   "source": [
    "full_kmeans.head()"
   ],
   "outputs": [
    {
     "output_type": "execute_result",
     "data": {
      "text/plain": [
       "              Age  días_encartera  days_between  Productos_credit_card  \\\n",
       "pk_cid                                                                   \n",
       "1000028  1.085781        0.301552     -0.758598              -0.128798   \n",
       "1000113  1.898254        0.685468     -1.438310              -0.128798   \n",
       "1000157  1.153487        0.965934     -0.301998              -0.128798   \n",
       "1000162  0.491472       -0.606514      0.579700              -0.128798   \n",
       "1000217  1.070735        0.616501     -0.013150               1.144838   \n",
       "\n",
       "         Productos_debit_card  Productos_em_account_p  Productos_em_acount  \\\n",
       "pk_cid                                                                       \n",
       "1000028              3.606298               -0.002389             0.602336   \n",
       "1000113             -0.376418               -0.002389            -1.810554   \n",
       "1000157              1.263524               -0.002389             0.429987   \n",
       "1000162             -0.376418               -0.002389             0.602336   \n",
       "1000217             -0.376418               -0.002389            -2.155253   \n",
       "\n",
       "         Productos_emc_account  Productos_funds  Productos_loans  \\\n",
       "pk_cid                                                             \n",
       "1000028              -0.257948        -0.063353        -0.010046   \n",
       "1000113               0.294906        -0.063353        -0.010046   \n",
       "1000157              -0.257948        -0.063353        -0.010046   \n",
       "1000162              -0.257948        -0.063353        -0.010046   \n",
       "1000217              -0.257948        -0.063353        -0.010046   \n",
       "\n",
       "         Productos_long_term_deposit  Productos_mortgage  \\\n",
       "pk_cid                                                     \n",
       "1000028                    -0.150412           -0.007769   \n",
       "1000113                    -0.150412           -0.007769   \n",
       "1000157                    -0.150412           -0.007769   \n",
       "1000162                    -0.150412           -0.007769   \n",
       "1000217                    -0.150412           -0.007769   \n",
       "\n",
       "         Productos_payroll_account  Productos_pension_plan  \\\n",
       "pk_cid                                                       \n",
       "1000028                  -0.275955               -0.237815   \n",
       "1000113                  -0.275955               -0.237815   \n",
       "1000157                  -0.275955               -0.237815   \n",
       "1000162                  -0.275955               -0.237815   \n",
       "1000217                  -0.275955               -0.237815   \n",
       "\n",
       "         Productos_securities  Productos_short_term_deposit  \n",
       "pk_cid                                                       \n",
       "1000028             -0.067806                     -0.120429  \n",
       "1000113             -0.067806                     -0.120429  \n",
       "1000157             -0.067806                     -0.120429  \n",
       "1000162             -0.067806                     -0.120429  \n",
       "1000217             18.180365                     -0.120429  "
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
       "      <th>Age</th>\n",
       "      <th>días_encartera</th>\n",
       "      <th>days_between</th>\n",
       "      <th>Productos_credit_card</th>\n",
       "      <th>Productos_debit_card</th>\n",
       "      <th>Productos_em_account_p</th>\n",
       "      <th>Productos_em_acount</th>\n",
       "      <th>Productos_emc_account</th>\n",
       "      <th>Productos_funds</th>\n",
       "      <th>Productos_loans</th>\n",
       "      <th>Productos_long_term_deposit</th>\n",
       "      <th>Productos_mortgage</th>\n",
       "      <th>Productos_payroll_account</th>\n",
       "      <th>Productos_pension_plan</th>\n",
       "      <th>Productos_securities</th>\n",
       "      <th>Productos_short_term_deposit</th>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>pk_cid</th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>1000028</th>\n",
       "      <td>1.085781</td>\n",
       "      <td>0.301552</td>\n",
       "      <td>-0.758598</td>\n",
       "      <td>-0.128798</td>\n",
       "      <td>3.606298</td>\n",
       "      <td>-0.002389</td>\n",
       "      <td>0.602336</td>\n",
       "      <td>-0.257948</td>\n",
       "      <td>-0.063353</td>\n",
       "      <td>-0.010046</td>\n",
       "      <td>-0.150412</td>\n",
       "      <td>-0.007769</td>\n",
       "      <td>-0.275955</td>\n",
       "      <td>-0.237815</td>\n",
       "      <td>-0.067806</td>\n",
       "      <td>-0.120429</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1000113</th>\n",
       "      <td>1.898254</td>\n",
       "      <td>0.685468</td>\n",
       "      <td>-1.438310</td>\n",
       "      <td>-0.128798</td>\n",
       "      <td>-0.376418</td>\n",
       "      <td>-0.002389</td>\n",
       "      <td>-1.810554</td>\n",
       "      <td>0.294906</td>\n",
       "      <td>-0.063353</td>\n",
       "      <td>-0.010046</td>\n",
       "      <td>-0.150412</td>\n",
       "      <td>-0.007769</td>\n",
       "      <td>-0.275955</td>\n",
       "      <td>-0.237815</td>\n",
       "      <td>-0.067806</td>\n",
       "      <td>-0.120429</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1000157</th>\n",
       "      <td>1.153487</td>\n",
       "      <td>0.965934</td>\n",
       "      <td>-0.301998</td>\n",
       "      <td>-0.128798</td>\n",
       "      <td>1.263524</td>\n",
       "      <td>-0.002389</td>\n",
       "      <td>0.429987</td>\n",
       "      <td>-0.257948</td>\n",
       "      <td>-0.063353</td>\n",
       "      <td>-0.010046</td>\n",
       "      <td>-0.150412</td>\n",
       "      <td>-0.007769</td>\n",
       "      <td>-0.275955</td>\n",
       "      <td>-0.237815</td>\n",
       "      <td>-0.067806</td>\n",
       "      <td>-0.120429</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1000162</th>\n",
       "      <td>0.491472</td>\n",
       "      <td>-0.606514</td>\n",
       "      <td>0.579700</td>\n",
       "      <td>-0.128798</td>\n",
       "      <td>-0.376418</td>\n",
       "      <td>-0.002389</td>\n",
       "      <td>0.602336</td>\n",
       "      <td>-0.257948</td>\n",
       "      <td>-0.063353</td>\n",
       "      <td>-0.010046</td>\n",
       "      <td>-0.150412</td>\n",
       "      <td>-0.007769</td>\n",
       "      <td>-0.275955</td>\n",
       "      <td>-0.237815</td>\n",
       "      <td>-0.067806</td>\n",
       "      <td>-0.120429</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1000217</th>\n",
       "      <td>1.070735</td>\n",
       "      <td>0.616501</td>\n",
       "      <td>-0.013150</td>\n",
       "      <td>1.144838</td>\n",
       "      <td>-0.376418</td>\n",
       "      <td>-0.002389</td>\n",
       "      <td>-2.155253</td>\n",
       "      <td>-0.257948</td>\n",
       "      <td>-0.063353</td>\n",
       "      <td>-0.010046</td>\n",
       "      <td>-0.150412</td>\n",
       "      <td>-0.007769</td>\n",
       "      <td>-0.275955</td>\n",
       "      <td>-0.237815</td>\n",
       "      <td>18.180365</td>\n",
       "      <td>-0.120429</td>\n",
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
   "metadata": {
    "azdata_cell_guid": "9273d8fc-d764-43b6-bc56-0de85d3f6124"
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
   "execution_count": 35,
   "source": [
    "\r\n",
    "standard_scaler = StandardScaler()\r\n",
    "scaled_df = standard_scaler.fit_transform(full_kmeans)\r\n",
    "scaled_df = pd.DataFrame(scaled_df, index = full_kmeans.index, columns = full_kmeans.columns)\r\n",
    "scaled_df.shape"
   ],
   "outputs": [
    {
     "output_type": "execute_result",
     "data": {
      "text/plain": [
       "(350384, 16)"
      ]
     },
     "metadata": {},
     "execution_count": 35
    }
   ],
   "metadata": {
    "azdata_cell_guid": "41b3724f-c128-43c6-ad16-2a4c0be6078f"
   }
  },
  {
   "cell_type": "code",
   "execution_count": 36,
   "source": [
    "# import the function to compute cosine_similarity\r\n",
    "from sklearn.metrics.pairwise import cosine_similarity\r\n",
    "from sklearn.decomposition import PCA\r\n",
    "from sklearn.cluster import KMeans\r\n",
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
   "execution_count": 37,
   "source": [
    "CALCULATE_ELBOW = True\r\n",
    "\r\n",
    "if CALCULATE_ELBOW:\r\n",
    "    st = time.time()\r\n",
    "\r\n",
    "    sse = {}\r\n",
    "\r\n",
    "    for k in range(2, 15): #todo los clusteres que queramos ver en la curva\r\n",
    "\r\n",
    "        print(f\"Fitting pipe with {k} clusters\")\r\n",
    "        cluster_model = KMeans(n_clusters = k)\r\n",
    "        cluster_model.fit(scaled_df)\r\n",
    "\r\n",
    "        sse[k] = cluster_model.inertia_\r\n",
    "\r\n",
    "    et = time.time()\r\n",
    "    print(\"Elbow curve took {} minutes.\".format(round((et - st)/60), 2))"
   ],
   "outputs": [
    {
     "output_type": "stream",
     "name": "stdout",
     "text": [
      "Fitting pipe with 2 clusters\n",
      "Fitting pipe with 3 clusters\n",
      "Fitting pipe with 4 clusters\n",
      "Fitting pipe with 5 clusters\n",
      "Fitting pipe with 6 clusters\n",
      "Fitting pipe with 7 clusters\n",
      "Fitting pipe with 8 clusters\n",
      "Fitting pipe with 9 clusters\n",
      "Fitting pipe with 10 clusters\n",
      "Fitting pipe with 11 clusters\n",
      "Fitting pipe with 12 clusters\n",
      "Fitting pipe with 13 clusters\n",
      "Fitting pipe with 14 clusters\n",
      "Elbow curve took 1 minutes.\n"
     ]
    }
   ],
   "metadata": {
    "azdata_cell_guid": "41afa36a-a993-4735-8cca-dc90894cce8a"
   }
  },
  {
   "cell_type": "code",
   "execution_count": 38,
   "source": [
    "if CALCULATE_ELBOW:\r\n",
    "    fig = plt.figure(figsize = (16, 8))\r\n",
    "    ax = fig.add_subplot()\r\n",
    "\r\n",
    "    x_values = list(sse.keys())\r\n",
    "    y_values = list(sse.values())\r\n",
    "\r\n",
    "    ax.plot(x_values, y_values, label = \"Inertia/dispersión de los clústers\")\r\n",
    "    fig.suptitle(\"Variación de la dispersión de los clústers en función de la k\", fontsize = 16);"
   ],
   "outputs": [
    {
     "output_type": "display_data",
     "data": {
      "text/plain": [
       "<Figure size 1152x576 with 1 Axes>"
      ],
      "image/png": "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"
     },
     "metadata": {
      "needs_background": "light"
     }
    }
   ],
   "metadata": {
    "azdata_cell_guid": "1046d9fb-d02d-46d4-945d-bbe86964f3d6"
   }
  },
  {
   "cell_type": "code",
   "execution_count": 39,
   "source": [
    "clust = 6"
   ],
   "outputs": [],
   "metadata": {}
  },
  {
   "cell_type": "code",
   "execution_count": 40,
   "source": [
    "kmeans = KMeans(n_clusters = clust)\r\n",
    "kmeans.fit(scaled_df)\r\n",
    "full_kmeans['cluster'] = kmeans.labels_"
   ],
   "outputs": [],
   "metadata": {
    "azdata_cell_guid": "6c580fd7-3b03-4c87-9c13-142d0866e5e1"
   }
  },
  {
   "cell_type": "code",
   "execution_count": 41,
   "source": [
    "# fiteamos un modelo con k = clust (que hemos sacado de la elbow curve anterior) \r\n",
    "# y con el dataframe escalado y sin outliers\r\n",
    "\r\n",
    "cluster_model = KMeans(n_clusters = clust)\r\n",
    "cluster_model.fit(scaled_df)"
   ],
   "outputs": [
    {
     "output_type": "execute_result",
     "data": {
      "text/plain": [
       "KMeans(n_clusters=6)"
      ]
     },
     "metadata": {},
     "execution_count": 41
    }
   ],
   "metadata": {
    "azdata_cell_guid": "268a7b64-740e-4b89-a1c4-79c239fdad34"
   }
  },
  {
   "cell_type": "code",
   "execution_count": 42,
   "source": [
    "# generamos el dataframe escalado (con el scaler del paso anterior, entrado sin outliers) pero con todos los datos.\r\n",
    "# por tanto vamos a transformar incluso a los outliers pero con el scaler entrado sin ellos.\r\n",
    "# el motivo es porque los outliers pueden afectar mucho la media y la desviación utilizado para transformar.\r\n",
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
    "# convertimos a dataframe\r\n",
    "scaled_df_with_outliers = pd.DataFrame(scaled_df_with_outliers, \r\n",
    "                                       index = scaled_df.index, \r\n",
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
   "outputs": [
    {
     "output_type": "execute_result",
     "data": {
      "text/plain": [
       "(350384, 16)"
      ]
     },
     "metadata": {},
     "execution_count": 58
    }
   ],
   "metadata": {
    "azdata_cell_guid": "47fd721c-2bbb-44c8-b841-aa0934cd1856"
   }
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "source": [
    "# calculamos el cluster de cada cliente, a partir del dataframe escalado y con outliers\r\n",
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
    "scaled_df[\"cluster\"] = labels\r\n",
    "scaled_df.head(15).T"
   ],
   "outputs": [
    {
     "output_type": "execute_result",
     "data": {
      "text/plain": [
       "pk_cid                         1000028   1000113   1000157   1000162  \\\n",
       "Age                           1.085781  1.898254  1.153487  0.491472   \n",
       "días_encartera                0.301552  0.685468  0.965934 -0.606514   \n",
       "days_between                 -0.758598 -1.438310 -0.301998  0.579700   \n",
       "Productos_credit_card        -0.128798 -0.128798 -0.128798 -0.128798   \n",
       "Productos_debit_card          3.606298 -0.376418  1.263524 -0.376418   \n",
       "Productos_em_account_p       -0.002389 -0.002389 -0.002389 -0.002389   \n",
       "Productos_em_acount           0.602336 -1.810554  0.429987  0.602336   \n",
       "Productos_emc_account        -0.257948  0.294906 -0.257948 -0.257948   \n",
       "Productos_funds              -0.063353 -0.063353 -0.063353 -0.063353   \n",
       "Productos_loans              -0.010046 -0.010046 -0.010046 -0.010046   \n",
       "Productos_long_term_deposit  -0.150412 -0.150412 -0.150412 -0.150412   \n",
       "Productos_mortgage           -0.007769 -0.007769 -0.007769 -0.007769   \n",
       "Productos_payroll_account    -0.275955 -0.275955 -0.275955 -0.275955   \n",
       "Productos_pension_plan       -0.237815 -0.237815 -0.237815 -0.237815   \n",
       "Productos_securities         -0.067806 -0.067806 -0.067806 -0.067806   \n",
       "Productos_short_term_deposit -0.120429 -0.120429 -0.120429 -0.120429   \n",
       "cluster                       1.000000  1.000000  1.000000  1.000000   \n",
       "\n",
       "pk_cid                          1000217   1000306   1000385   1000386  \\\n",
       "Age                            1.070735 -0.565205 -0.555095 -0.555095   \n",
       "días_encartera                 0.616501  0.432589  1.984347  1.984347   \n",
       "days_between                  -0.013150 -1.284019  0.594809  0.594809   \n",
       "Productos_credit_card          1.144838 -0.128798 -0.128798 -0.128798   \n",
       "Productos_debit_card          -0.376418  3.606298 -0.376418 -0.376418   \n",
       "Productos_em_account_p        -0.002389 -0.002389 -0.002389 -0.002389   \n",
       "Productos_em_acount           -2.155253 -2.155253  0.774685  0.774685   \n",
       "Productos_emc_account         -0.257948 -0.257948 -0.257948 -0.257948   \n",
       "Productos_funds               -0.063353 -0.063353 -0.063353 -0.063353   \n",
       "Productos_loans               -0.010046 -0.010046 -0.010046 -0.010046   \n",
       "Productos_long_term_deposit   -0.150412 -0.150412 -0.150412 -0.150412   \n",
       "Productos_mortgage            -0.007769 -0.007769 -0.007769 -0.007769   \n",
       "Productos_payroll_account     -0.275955  4.713824 -0.275955 -0.275955   \n",
       "Productos_pension_plan        -0.237815  3.205397 -0.237815 -0.237815   \n",
       "Productos_securities          18.180365 -0.067806 -0.067806 -0.067806   \n",
       "Productos_short_term_deposit  -0.120429 -0.120429 -0.120429 -0.120429   \n",
       "cluster                        1.000000  1.000000  1.000000  1.000000   \n",
       "\n",
       "pk_cid                         1000387   1000388   1000389   1000390  \\\n",
       "Age                          -0.555095 -0.321443 -0.555095 -0.555095   \n",
       "días_encartera                1.984347  1.984347  1.984347  1.984347   \n",
       "days_between                  0.594809  0.594809  0.594809  0.594809   \n",
       "Productos_credit_card        -0.128798 -0.128798 -0.128798 -0.128798   \n",
       "Productos_debit_card         -0.376418 -0.376418 -0.376418 -0.376418   \n",
       "Productos_em_account_p       -0.002389 -0.002389 -0.002389 -0.002389   \n",
       "Productos_em_acount           0.774685  0.774685  0.774685  0.774685   \n",
       "Productos_emc_account        -0.257948 -0.257948 -0.257948 -0.257948   \n",
       "Productos_funds              -0.063353 -0.063353 -0.063353 -0.063353   \n",
       "Productos_loans              -0.010046 -0.010046 -0.010046 -0.010046   \n",
       "Productos_long_term_deposit  -0.150412 -0.150412 -0.150412 -0.150412   \n",
       "Productos_mortgage           -0.007769 -0.007769 -0.007769 -0.007769   \n",
       "Productos_payroll_account    -0.275955 -0.275955 -0.275955 -0.275955   \n",
       "Productos_pension_plan       -0.237815 -0.237815 -0.237815 -0.237815   \n",
       "Productos_securities         -0.067806 -0.067806 -0.067806 -0.067806   \n",
       "Productos_short_term_deposit -0.120429 -0.120429 -0.120429 -0.120429   \n",
       "cluster                       1.000000  1.000000  1.000000  1.000000   \n",
       "\n",
       "pk_cid                         1000391   1000393   1000394  \n",
       "Age                           0.782439 -0.555095 -0.555095  \n",
       "días_encartera                1.984347  1.984347  1.984347  \n",
       "days_between                 -0.763667  0.594809  0.594809  \n",
       "Productos_credit_card        -0.128798 -0.128798 -0.128798  \n",
       "Productos_debit_card          1.966356 -0.376418 -0.376418  \n",
       "Productos_em_account_p       -0.002389 -0.002389 -0.002389  \n",
       "Productos_em_acount          -0.776458  0.774685  0.774685  \n",
       "Productos_emc_account        -0.257948 -0.257948 -0.257948  \n",
       "Productos_funds              -0.063353 -0.063353 -0.063353  \n",
       "Productos_loans              -0.010046 -0.010046 -0.010046  \n",
       "Productos_long_term_deposit  -0.150412 -0.150412 -0.150412  \n",
       "Productos_mortgage           -0.007769 -0.007769 -0.007769  \n",
       "Productos_payroll_account     2.365693 -0.275955 -0.275955  \n",
       "Productos_pension_plan        0.527343 -0.237815 -0.237815  \n",
       "Productos_securities         -0.067806 -0.067806 -0.067806  \n",
       "Productos_short_term_deposit -0.120429 -0.120429 -0.120429  \n",
       "cluster                       1.000000  1.000000  1.000000  "
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
       "      <th>pk_cid</th>\n",
       "      <th>1000028</th>\n",
       "      <th>1000113</th>\n",
       "      <th>1000157</th>\n",
       "      <th>1000162</th>\n",
       "      <th>1000217</th>\n",
       "      <th>1000306</th>\n",
       "      <th>1000385</th>\n",
       "      <th>1000386</th>\n",
       "      <th>1000387</th>\n",
       "      <th>1000388</th>\n",
       "      <th>1000389</th>\n",
       "      <th>1000390</th>\n",
       "      <th>1000391</th>\n",
       "      <th>1000393</th>\n",
       "      <th>1000394</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>Age</th>\n",
       "      <td>1.085781</td>\n",
       "      <td>1.898254</td>\n",
       "      <td>1.153487</td>\n",
       "      <td>0.491472</td>\n",
       "      <td>1.070735</td>\n",
       "      <td>-0.565205</td>\n",
       "      <td>-0.555095</td>\n",
       "      <td>-0.555095</td>\n",
       "      <td>-0.555095</td>\n",
       "      <td>-0.321443</td>\n",
       "      <td>-0.555095</td>\n",
       "      <td>-0.555095</td>\n",
       "      <td>0.782439</td>\n",
       "      <td>-0.555095</td>\n",
       "      <td>-0.555095</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>días_encartera</th>\n",
       "      <td>0.301552</td>\n",
       "      <td>0.685468</td>\n",
       "      <td>0.965934</td>\n",
       "      <td>-0.606514</td>\n",
       "      <td>0.616501</td>\n",
       "      <td>0.432589</td>\n",
       "      <td>1.984347</td>\n",
       "      <td>1.984347</td>\n",
       "      <td>1.984347</td>\n",
       "      <td>1.984347</td>\n",
       "      <td>1.984347</td>\n",
       "      <td>1.984347</td>\n",
       "      <td>1.984347</td>\n",
       "      <td>1.984347</td>\n",
       "      <td>1.984347</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>days_between</th>\n",
       "      <td>-0.758598</td>\n",
       "      <td>-1.438310</td>\n",
       "      <td>-0.301998</td>\n",
       "      <td>0.579700</td>\n",
       "      <td>-0.013150</td>\n",
       "      <td>-1.284019</td>\n",
       "      <td>0.594809</td>\n",
       "      <td>0.594809</td>\n",
       "      <td>0.594809</td>\n",
       "      <td>0.594809</td>\n",
       "      <td>0.594809</td>\n",
       "      <td>0.594809</td>\n",
       "      <td>-0.763667</td>\n",
       "      <td>0.594809</td>\n",
       "      <td>0.594809</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Productos_credit_card</th>\n",
       "      <td>-0.128798</td>\n",
       "      <td>-0.128798</td>\n",
       "      <td>-0.128798</td>\n",
       "      <td>-0.128798</td>\n",
       "      <td>1.144838</td>\n",
       "      <td>-0.128798</td>\n",
       "      <td>-0.128798</td>\n",
       "      <td>-0.128798</td>\n",
       "      <td>-0.128798</td>\n",
       "      <td>-0.128798</td>\n",
       "      <td>-0.128798</td>\n",
       "      <td>-0.128798</td>\n",
       "      <td>-0.128798</td>\n",
       "      <td>-0.128798</td>\n",
       "      <td>-0.128798</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Productos_debit_card</th>\n",
       "      <td>3.606298</td>\n",
       "      <td>-0.376418</td>\n",
       "      <td>1.263524</td>\n",
       "      <td>-0.376418</td>\n",
       "      <td>-0.376418</td>\n",
       "      <td>3.606298</td>\n",
       "      <td>-0.376418</td>\n",
       "      <td>-0.376418</td>\n",
       "      <td>-0.376418</td>\n",
       "      <td>-0.376418</td>\n",
       "      <td>-0.376418</td>\n",
       "      <td>-0.376418</td>\n",
       "      <td>1.966356</td>\n",
       "      <td>-0.376418</td>\n",
       "      <td>-0.376418</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Productos_em_account_p</th>\n",
       "      <td>-0.002389</td>\n",
       "      <td>-0.002389</td>\n",
       "      <td>-0.002389</td>\n",
       "      <td>-0.002389</td>\n",
       "      <td>-0.002389</td>\n",
       "      <td>-0.002389</td>\n",
       "      <td>-0.002389</td>\n",
       "      <td>-0.002389</td>\n",
       "      <td>-0.002389</td>\n",
       "      <td>-0.002389</td>\n",
       "      <td>-0.002389</td>\n",
       "      <td>-0.002389</td>\n",
       "      <td>-0.002389</td>\n",
       "      <td>-0.002389</td>\n",
       "      <td>-0.002389</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Productos_em_acount</th>\n",
       "      <td>0.602336</td>\n",
       "      <td>-1.810554</td>\n",
       "      <td>0.429987</td>\n",
       "      <td>0.602336</td>\n",
       "      <td>-2.155253</td>\n",
       "      <td>-2.155253</td>\n",
       "      <td>0.774685</td>\n",
       "      <td>0.774685</td>\n",
       "      <td>0.774685</td>\n",
       "      <td>0.774685</td>\n",
       "      <td>0.774685</td>\n",
       "      <td>0.774685</td>\n",
       "      <td>-0.776458</td>\n",
       "      <td>0.774685</td>\n",
       "      <td>0.774685</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Productos_emc_account</th>\n",
       "      <td>-0.257948</td>\n",
       "      <td>0.294906</td>\n",
       "      <td>-0.257948</td>\n",
       "      <td>-0.257948</td>\n",
       "      <td>-0.257948</td>\n",
       "      <td>-0.257948</td>\n",
       "      <td>-0.257948</td>\n",
       "      <td>-0.257948</td>\n",
       "      <td>-0.257948</td>\n",
       "      <td>-0.257948</td>\n",
       "      <td>-0.257948</td>\n",
       "      <td>-0.257948</td>\n",
       "      <td>-0.257948</td>\n",
       "      <td>-0.257948</td>\n",
       "      <td>-0.257948</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Productos_funds</th>\n",
       "      <td>-0.063353</td>\n",
       "      <td>-0.063353</td>\n",
       "      <td>-0.063353</td>\n",
       "      <td>-0.063353</td>\n",
       "      <td>-0.063353</td>\n",
       "      <td>-0.063353</td>\n",
       "      <td>-0.063353</td>\n",
       "      <td>-0.063353</td>\n",
       "      <td>-0.063353</td>\n",
       "      <td>-0.063353</td>\n",
       "      <td>-0.063353</td>\n",
       "      <td>-0.063353</td>\n",
       "      <td>-0.063353</td>\n",
       "      <td>-0.063353</td>\n",
       "      <td>-0.063353</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Productos_loans</th>\n",
       "      <td>-0.010046</td>\n",
       "      <td>-0.010046</td>\n",
       "      <td>-0.010046</td>\n",
       "      <td>-0.010046</td>\n",
       "      <td>-0.010046</td>\n",
       "      <td>-0.010046</td>\n",
       "      <td>-0.010046</td>\n",
       "      <td>-0.010046</td>\n",
       "      <td>-0.010046</td>\n",
       "      <td>-0.010046</td>\n",
       "      <td>-0.010046</td>\n",
       "      <td>-0.010046</td>\n",
       "      <td>-0.010046</td>\n",
       "      <td>-0.010046</td>\n",
       "      <td>-0.010046</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Productos_long_term_deposit</th>\n",
       "      <td>-0.150412</td>\n",
       "      <td>-0.150412</td>\n",
       "      <td>-0.150412</td>\n",
       "      <td>-0.150412</td>\n",
       "      <td>-0.150412</td>\n",
       "      <td>-0.150412</td>\n",
       "      <td>-0.150412</td>\n",
       "      <td>-0.150412</td>\n",
       "      <td>-0.150412</td>\n",
       "      <td>-0.150412</td>\n",
       "      <td>-0.150412</td>\n",
       "      <td>-0.150412</td>\n",
       "      <td>-0.150412</td>\n",
       "      <td>-0.150412</td>\n",
       "      <td>-0.150412</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Productos_mortgage</th>\n",
       "      <td>-0.007769</td>\n",
       "      <td>-0.007769</td>\n",
       "      <td>-0.007769</td>\n",
       "      <td>-0.007769</td>\n",
       "      <td>-0.007769</td>\n",
       "      <td>-0.007769</td>\n",
       "      <td>-0.007769</td>\n",
       "      <td>-0.007769</td>\n",
       "      <td>-0.007769</td>\n",
       "      <td>-0.007769</td>\n",
       "      <td>-0.007769</td>\n",
       "      <td>-0.007769</td>\n",
       "      <td>-0.007769</td>\n",
       "      <td>-0.007769</td>\n",
       "      <td>-0.007769</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Productos_payroll_account</th>\n",
       "      <td>-0.275955</td>\n",
       "      <td>-0.275955</td>\n",
       "      <td>-0.275955</td>\n",
       "      <td>-0.275955</td>\n",
       "      <td>-0.275955</td>\n",
       "      <td>4.713824</td>\n",
       "      <td>-0.275955</td>\n",
       "      <td>-0.275955</td>\n",
       "      <td>-0.275955</td>\n",
       "      <td>-0.275955</td>\n",
       "      <td>-0.275955</td>\n",
       "      <td>-0.275955</td>\n",
       "      <td>2.365693</td>\n",
       "      <td>-0.275955</td>\n",
       "      <td>-0.275955</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Productos_pension_plan</th>\n",
       "      <td>-0.237815</td>\n",
       "      <td>-0.237815</td>\n",
       "      <td>-0.237815</td>\n",
       "      <td>-0.237815</td>\n",
       "      <td>-0.237815</td>\n",
       "      <td>3.205397</td>\n",
       "      <td>-0.237815</td>\n",
       "      <td>-0.237815</td>\n",
       "      <td>-0.237815</td>\n",
       "      <td>-0.237815</td>\n",
       "      <td>-0.237815</td>\n",
       "      <td>-0.237815</td>\n",
       "      <td>0.527343</td>\n",
       "      <td>-0.237815</td>\n",
       "      <td>-0.237815</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Productos_securities</th>\n",
       "      <td>-0.067806</td>\n",
       "      <td>-0.067806</td>\n",
       "      <td>-0.067806</td>\n",
       "      <td>-0.067806</td>\n",
       "      <td>18.180365</td>\n",
       "      <td>-0.067806</td>\n",
       "      <td>-0.067806</td>\n",
       "      <td>-0.067806</td>\n",
       "      <td>-0.067806</td>\n",
       "      <td>-0.067806</td>\n",
       "      <td>-0.067806</td>\n",
       "      <td>-0.067806</td>\n",
       "      <td>-0.067806</td>\n",
       "      <td>-0.067806</td>\n",
       "      <td>-0.067806</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Productos_short_term_deposit</th>\n",
       "      <td>-0.120429</td>\n",
       "      <td>-0.120429</td>\n",
       "      <td>-0.120429</td>\n",
       "      <td>-0.120429</td>\n",
       "      <td>-0.120429</td>\n",
       "      <td>-0.120429</td>\n",
       "      <td>-0.120429</td>\n",
       "      <td>-0.120429</td>\n",
       "      <td>-0.120429</td>\n",
       "      <td>-0.120429</td>\n",
       "      <td>-0.120429</td>\n",
       "      <td>-0.120429</td>\n",
       "      <td>-0.120429</td>\n",
       "      <td>-0.120429</td>\n",
       "      <td>-0.120429</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>cluster</th>\n",
       "      <td>1.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>1.000000</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ]
     },
     "metadata": {},
     "execution_count": 60
    }
   ],
   "metadata": {
    "azdata_cell_guid": "ef125b83-8447-4326-9dd7-56db90c5ad4b"
   }
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "source": [
    "scaled_df.columns"
   ],
   "outputs": [
    {
     "output_type": "execute_result",
     "data": {
      "text/plain": [
       "Index(['Age', 'días_encartera', 'days_between', 'Productos_credit_card',\n",
       "       'Productos_debit_card', 'Productos_em_account_p', 'Productos_em_acount',\n",
       "       'Productos_emc_account', 'Productos_funds', 'Productos_loans',\n",
       "       'Productos_long_term_deposit', 'Productos_mortgage',\n",
       "       'Productos_payroll_account', 'Productos_pension_plan',\n",
       "       'Productos_securities', 'Productos_short_term_deposit', 'cluster'],\n",
       "      dtype='object')"
      ]
     },
     "metadata": {},
     "execution_count": 61
    }
   ],
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
    "# visualizamos nuestros grupos en base a las variables del modelo RFM, para ver que tal han quedado.\r\n",
    "selected_columns = ['días_encartera', 'Age','Productos_payroll_account', 'Productos_mortgage',\r\n",
    "       'Productos_short_term_deposit']\r\n",
    "\r\n",
    "sns.pairplot(scaled_df, vars = selected_columns, hue = 'cluster');"
   ],
   "outputs": [
    {
     "output_type": "display_data",
     "data": {
      "text/plain": [
       "<Figure size 942.375x900 with 30 Axes>"
      ],
      "image/png": "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z6dRquM2WzS2C3j1JqnhXlRTo7S2Tb3eCAVhJFgjeXjghaUIWm6L5uJyaJnVsciSCPngLUH7s0OHLUm4H0T73wgE7bYHFWKjORQpaZYL2B/sIRAEqgBFIRCibdMl6CQWO29+8FT73+3XcfCWg8dCHNomBJ3tnc0P7euIIdiDfT54pTj8nqP7Dv8vHn5x+2uQgUQGAUEgCXyJDCR+EIQ+ZIqCAPylJfYWl2hIFSkELmoYuAuB8mB/QiABzw+/0VgihhEPS617RwfIRKVjEREREREREREREc8MIaX89FedYIQQu8Dycc/9ia//RPf/+af/47En8TlG0iDZfVBKG/jB53q/logR749UI08ya9+5S+D4h7b96f/332x9uPKd6x/zlj0p5U980j4/6fz8HHQDe4+5j+dJNP7nw7M6Px/won5PD3iRx/8ijv1ZnJ8v2vcSjffp81nH/Inn5+c4N1/E7+hxiI736fOp184vCi98kPosEEK8I6V8+XmP40nzRT0u+GId24t+LNH4vxy86N/Tizz+F3nsT5MX7XuJxvv0edZjfhG/o8chOt6IJ8mjNUNGRERERERERERERERERDwFoiA1IiIiIiIiIiIiIiIi4sQQBamfjb/1vAfwlPiiHhd8sY7tRT+WaPxfDl707+lFHv+LPPanyYv2vUTjffo86zG/iN/R4xAdb8QTI+pJjYiIiIiIiIiIiIiIiDgxRJnUiIiIiIiIiIiIiIiIiBNDFKRGREREREREREREREREnBhe+CD1J37iJyQQ/UQ/z+PnU4nOz+jnOf58KtH5Gf08x59PJTo/o5/n+POJROdm9PMcf740vPBB6t7el8kzOOJFIzo/I04y0fkZcZKJzs+Ik0p0bkZEPH1e+CA1IiIiIiIiIiIiIiIi4otDFKRGREREREREREREREREnBi05z2AiC8GMgjwWy1QVLSY+byHExEREfGxSCnxmy0QAs2KPe/hRDwFvGYTADUWQwjxnEcT8WXDt22k56PGTISqPu/hRES8kERBasRj49YbVBYWqa+sopomubOnsfp6owtzRETEicNrNqktrVC9dw+hamRPzWENDqDq+vMeWsQTwLcd6mtrlG/fBSA9M01ydBjVjBZPI54+Mgho7uxSvHodv9UiMTxEenYaPZF43kOLiHjhiMp9XzCk7+PbDlKeDIEvKSXVe/epL6+A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"
     },
     "metadata": {
      "needs_background": "light"
     }
    }
   ],
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