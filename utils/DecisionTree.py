import numpy as np
import pandas as pd
import warnings
from sklearn.feature_selection import mutual_info_classif


class DTree():
    @staticmethod
    def myEntropy(series, base=2):
        """计算边缘熵 H(series)"""
        probs = series.value_counts(normalize=True)
        return -np.sum(probs * np.log(probs) / np.log(base)) #np.log是以e为底，为了以2为底，所以只是进行了换底
    @staticmethod
    def myJointEntropy(series1, series2, base=2):
        """计算联合熵 H(series1, series2)"""
        joint = pd.Series(list(zip(series1, series2)))
        probs = joint.value_counts(normalize=True)
        return -np.sum(probs * np.log(probs) / np.log(base))
    @staticmethod
    def myConditionalEntropy(target, feature, base=2):
        """计算条件熵 H(target | feature) = H(target, feature) - H(feature)"""
        return DTree.myJointEntropy(target, feature, base) - DTree.myEntropy(feature, base)
    @staticmethod
    def myFeatureEntropy(df, target_col, bins=10, base=2, numeric_bins='auto'):
        """
        分析每个特征相对于目标列的信息量。
        
        参数
        ----------
        df : DataFrame
            输入数据。
        target_col : str
            目标列（类别标签）。
        bins : int, 默认 10
            数值特征分箱数量。
        base : float, 默认 2
            对数底。
        numeric_bins : 'auto' 或 int
            如果 'auto'，则对数值特征先判断唯一值数量，若 > bins 则分箱。
        
        返回
        -------
        DataFrame : 
            包含每个特征（不包括目标）的：
            - 边缘熵 H(feature)
            - 联合熵 H(feature, target)
            - 条件熵 H(feature | target)
            - 信息增益 IG = H(target) - H(feature | target)
            按信息增益降序排列（即越重要越靠前）。
        """
        target_series = df[target_col]
        H_target = DTree.myEntropy(target_series, base)
        
        results = []
        for col in df.columns:
            if col == target_col:
                continue
            feat = df[col]
            # 处理数值特征
            if pd.api.types.is_numeric_dtype(feat) and feat.nunique() > bins:
                if numeric_bins == 'auto':
                    # 自适应：如果唯一值数量 > bins，则分箱
                    feat_binned = pd.cut(feat, bins=bins, labels=False, include_lowest=True).fillna(-1).astype(int)
                else:
                    feat_binned = feat
            else:
                feat_binned = feat
            
            # 计算各项指标
            H_f = DTree.myEntropy(feat_binned, base)
            H_joint = DTree.myJointEntropy(feat_binned, target_series, base)
            H_c = H_joint - H_f  # H(feature | target) = H(feature,target) - H(target)
            # 注意：条件熵也可以直接用 conditional_entropy(feat_binned, target_series) 但需要先计算 H(target)
            # 这里使用联合熵减边缘熵 H(feature) 来算，更稳定。
            # 但信息增益 = H(target) - H(target | feature)  ≠ H(target) - H(feature | target)
            # 正确公式：IG = H(target) - H(target | feature)
            # 而 H(target | feature) = H(target, feature) - H(feature)
            # 所以 IG = H(target) - (H_joint - H_f)
            IG = H_target - (H_joint - H_f)
            
            results.append({
                'feature': col,
                'H(feature)': H_f,
                'H(feature,target)': H_joint,
                'H(feature|target)': H_c,
                'Information Gain': IG
            })
        
        result_df = pd.DataFrame(results)
        result_df = result_df.sort_values('Information Gain', ascending=False).reset_index(drop=True)
        return result_df
    @staticmethod
    def mySelectFeatures(df, target_col, max_features=None, bins=10, base=2):
        """
        使用贪心前向选择（逐步选择）特征。
        
        每次选择当前信息增益最高的特征，并可以继续添加更多特征（但注意冗余问题）。
        如果 max_features 为 None，则返回所有特征按重要性排序的列表。
        若指定数量，则返回前 max_features 个。
        """
        analysis = DTree.myFeatureEntropy(df, target_col, bins, base)
        selected = analysis['feature'].tolist()
        if max_features is not None:
            selected = selected[:max_features]
        return selected
    
    @staticmethod
    def myCalcMutual(df,col=None):
        """基于 k 近邻（k-NN）的非参数方法，来估计每个特征与离散目标变量之间的互信息（Mutual Information, MI）"""
        rst = dict()
        if isinstance(df,pd.DataFrame):
            if col is None:
                X = df.iloc[:,:-1]
                y = df.iloc[:,-1]
            else:
                X = df[[i for i in df.columns if i != col ]]
                y = df[col]
            mi = mutual_info_classif(X, y, random_state=42)
            for i,v in enumerate(X.columns):
                rst[v] = mi[i]
        else:
            if col is None:
                X = df[:,:-1]
                y = df[:,-1]
            else:
                X = df[:,[i for i in range(df.shape[1]) if i != col ]]
                y = df[:,col]
            mi = mutual_info_classif(X, y, random_state=42)
            for i,v in enumerate(mi):
                rst[f'col{i}'] = v
        return rst

if __name__=="__main__":
    from sklearn.datasets import load_iris
    import pandas as pd
    # import DTree
    # 加载数据
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['species'] = iris.target   # 目标列，值为 0,1,2
    
    # 1. 分析所有特征
    analysis_df = DTree.myFeatureEntropy(df, target_col='species', bins=10)
    print(analysis_df)
    
    # 2. 按信息增益排序的特征名（最重要的排前面）
    best_features = analysis_df['feature'].tolist()
    print("特征重要性排序:", best_features)
    
    # 3. 逐步选择前2个特征
    selected = DTree.mySelectFeatures(df, target_col='species', max_features=2)
    print("选出的前2个特征:", selected)
    