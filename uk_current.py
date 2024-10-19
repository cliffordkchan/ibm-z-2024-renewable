import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from matplotlib.ticker import AutoMinorLocator


def Custom_Ax(ax, ylabel_name='', ylabel_size=16, xlabel_name='', xlabel_size=16):
    """
    Customizando Axis Matplotlib.

    Parameters
    ----------
    ax : TYPE
        DESCRIPTION.
    ylabel_name : TYPE, optional
        DESCRIPTION. The default is ''.
    ylabel_size : TYPE, optional
        DESCRIPTION. The default is 16.
    xlabel_name : TYPE, optional
        DESCRIPTION. The default is ''.
    xlabel_size : TYPE, optional
        DESCRIPTION. The default is 16.

        Returns
    -------
    ax : TYPE
        DESCRIPTION.

    """
    ax.set_ylabel(ylabel_name, fontdict={'fontsize': ylabel_size})
    ax.set_xlabel(xlabel_name, fontdict={'fontsize': xlabel_size})
    ax.grid(color='black', linestyle='-', linewidth=1, alpha=.5)
    ax.grid(which='minor', axis="x", color="black", alpha=.2, linewidth=.5, linestyle="--")
    ax.grid(which='minor', axis="y", color="black", alpha=.2, linewidth=.5, linestyle="--")
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.grid(visible=True, which='major', axis="x", color="black", alpha=1, linewidth=.5, linestyle="-")
    ax.grid(visible=True, which='major', axis="y", color="black", alpha=1, linewidth=.5, linestyle="-")

    return ax


file_path = 'World Energy Consumption/World Energy Consumption.csv'
df = pd.read_csv(file_path, encoding='gbk')

# 打印列名，检查是否存在与代码中的列名一致的列
print(df.columns)

# 去除列名中的空格
df.columns = df.columns.str.strip()

# 确保 'year' 列是整数类型
df['year'] = df['year'].astype(int)
df['hydro_electricity_%'] = (df['hydro_electricity'] / df['electricity_generation'])
df['nuclear_electricity_%'] = df['nuclear_electricity'] / df['electricity_generation']
df['solar_electricity_%'] = df['solar_electricity'] / df['electricity_generation']
df['wind_electricity_%'] = df['wind_electricity'] / df['electricity_generation']
df['renewables_electricity_%'] = df['renewables_electricity'] / df['electricity_generation']
df['other_renewable_electricity_%'] = df['other_renewable_electricity'] / df['electricity_generation']
df['other_renewable_exc_biofuel_electricity_%'] = df['other_renewable_exc_biofuel_electricity'] / df['electricity_generation']

df['sum_renewable_%'] = df['hydro_electricity_%'] + df['nuclear_electricity_%'] + df['solar_electricity_%'] + df['wind_electricity_%']

# 检查列名是否正确，如果存在需要修正的列名，修改代码中的列名
fig, ax = plt.subplots(figsize=(16, 8))
ax.set(xticks=df.year[(df['year'] >= 1990) & (df['year'] <= 2022)].unique())

# 设置 x, y 标签的字体大小
ax.set_xlabel('Year', fontsize=12)
ax.set_ylabel('%', fontsize=12)

df[(df.country == 'United Kingdom') & (df['year'] >= 1990) & (df['year'] <= 2022)].plot(
    kind='line',
    x='year',
    y=['hydro_electricity_%', 'nuclear_electricity_%', 'solar_electricity_%', 'wind_electricity_%', 'sum_renewable_%'],
    ax=ax,
    title='Electricity - United Kingdom',
    grid=True
)

ax.legend(bbox_to_anchor=(1.0, 1.0))
plt.tight_layout()
plt.show()

# 过滤数据：仅保留 'United Kingdom' 且 'year' 在 1990 到 2022 之间的数据
filtered_df = df[(df.country == 'United Kingdom') & (df['year'] >= 1990) & (df['year'] <= 2022)]

# 选择需要保存的列
filtered_df = filtered_df[['year', 'hydro_electricity_%', 'nuclear_electricity_%', 'solar_electricity_%', 'wind_electricity_%', 'sum_renewable_%']]

# 保存到新的 CSV 文件
filtered_df.to_csv('uk_electricity_1990_2022.csv', index=False)

print("Data saved to 'uk_electricity_1990_2022.csv'")

