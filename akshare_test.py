import akshare as ak
import warnings
warnings.filterwarnings('ignore')


# 上证
stock_info_sh_name_code_df = ak.stock_info_sh_name_code(symbol="主板A股")
# 深证
stock_info_sz_name_code_df = ak.stock_info_sz_name_code(symbol="A股列表")

for index, row in stock_info_sh_name_code_df.iterrows():
    # print(row['证券代码'])
    try:
        stock_industry_change_cninfo_df = ak.stock_industry_change_cninfo(
            symbol=row['证券代码'],
            start_date="20091227",
            end_date="20220708",
        )
    except KeyError:
        # 个别证券缺少 records 字段，跳过
        continue
    # 过滤申银万国行业分类，避免对空数组求真值
    target_series = stock_industry_change_cninfo_df.loc[
        stock_industry_change_cninfo_df['分类标准'] == '申银万国行业分类标准',
        '行业大类',
    ].dropna()

    if target_series.empty:
        continue

    industry_value = target_series.iloc[0]
    if isinstance(industry_value, str) and '芯片' in industry_value:
        print(row['证券代码'])
