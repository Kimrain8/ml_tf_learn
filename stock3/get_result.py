import sys
from pathlib import Path
from datetime import datetime
import re
import pandas as pd
import numpy as np
from IPython import embed

from stock3.check_data import main as chkData
from stock3.features_v2 import run_m3_v2 as calcFeatures
from stock3.train_v2 import main as trainLSTM
from stock3.train_lgb import run_lgb as trainLGB
from stock3.predict_v2 import run_predict_v2 as predLSTM
from stock3.predict_lgb import run_predict_lgb as predLGB
from stock3.baseline import run_baselines
from stock3.backtest_v2 import run_backtest_v2
# 用于执行每日过程，然后获取结果

def get_latest_file(directory: str):
    pattern = re.compile(r"^pred_lstm_(\d{8})\.csv$")
    best_file, best_date = None, None
    for p in Path(directory).iterdir():
        m = pattern.match(p.name)
        if not m or not p.is_file():
            continue
        try:
            d = datetime.strptime(m.group(1), "%Y%m%d")
        except ValueError:
            continue  # 跳过非法日期，如 20261340
        if best_date is None or d > best_date:
            best_date, best_file = d, p
    return best_date.strftime("%Y%m%d"), best_file

def process_data(mods='1111'):
    """计算整个模型, 共4位, 第1位: 获取最新数据; 第2位: 计算特征; 第3位: 训练数据; 第4位: 预测结果; """
    if mods[0] == '1':
        # print("chkData()")
        chkData()
    if mods[1] == '1':
        # print("calcFeatures()")
        calcFeatures()
    if mods[2] == '1':
        # print("trainLSTM()")
        trainLSTM()
        trainLGB()
    if mods[3] == '1':
        # print("predLSTM(select_ratio=0.05)")
        predLSTM(select_ratio=0.05)
        predLGB(select_ratio=0.05)
    # run_baselines()
    # run_backtest_v2(select_ratio=0.05)


def get_result(dd=None,topn=100,debug_flag=False):
    """dd: 日期格式yyyymmdd; topn: 取结果的前多少的数据; """
    if dd is None:
        dd,f = get_latest_file("D:/pyprj/cv2/data/processed")
    if dd is None:
        print("没有可使用的文件！")
        return None
    df = pd.read_csv(f"D:/pyprj/cv2/data/processed/pred_lstm_{dd}.csv")
    df2 = pd.read_csv(f"D:/pyprj/cv2/data/processed/pred_lgb_{dd}.csv")
    df['symbol'] = df['symbol'].astype(str).str.zfill(6)
    df2['symbol'] = df2['symbol'].astype(str).str.zfill(6)
    df3 = df.merge(df2,on = ['symbol'],how='inner')
    rst = df3[(df3['rank_x'] < topn) & (df3['rank_y'] < topn)]

    # p = f"D:/pyprj/cv2/data/top{topn}_{dd}.xlsx"
    # rst.to_excel(p)
    if debug_flag == True:
        embed()
    return rst

if __name__ == "__main__":
    # print("全部参数列表：", sys.argv)
    if 'process' in sys.argv:
        process_data()
    if 'debug' in sys.argv:
        df = get_result(debug_flag=True)
    else:
        df = get_result()
        print(df.head(10))