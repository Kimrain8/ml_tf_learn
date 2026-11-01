import numpy as np
from unittest.mock import MagicMock
from ts_neural import updata_wb


def test_updata_wb():
    # 模拟 fead_forward 函数
    fead_forward = MagicMock()
    fead_forward.return_value = (np.array([[0.1, 0.2], [0.3, 0.4]]), 
                                  [np.array([[0.1, 0.2], [0.3, 0.4]]), 
                                   np.array([[0.5, 0.6], [0.7, 0.8]])], 
                                  np.array([[0.9, 0.1], [0.2, 0.3]]))
    
    # 模拟损失函数及其导数
    def loss_fun(output, labs):
        return np.mean((output - labs) ** 2)
    
    def de_loss_fun(output, labs):
        return 2 * (output - labs) / output.shape[0]
    
    # 初始化网络层
    layers = [
        {"w": np.array([[0.1, 0.2], [0.3, 0.4]]), "b": np.array([[0.1], [0.2]]), "de_act_fun": lambda z, h: np.ones_like(z)},
        {"w": np.array([[0.5, 0.6], [0.7, 0.8]]), "b": np.array([[0.3], [0.4]]), "de_act_fun": lambda z, h: np.ones_like(z)}
    ]
    
    # 输入数据和标签
    datas = np.array([[0.1, 0.2], [0.3, 0.4]])
    labs = np.array([[0.5, 0.6], [0.7, 0.8]])
    
    # 调用被测方法
    updated_layers, loss = updata_wb(datas, labs, layers, loss_fun, de_loss_fun, alpha=0.01)
    
    # 验证更新后的权重和偏置
    expected_w1 = np.array([[0.099, 0.198], [0.297, 0.396]])
    expected_b1 = np.array([[0.099], [0.198]])
    expected_w2 = np.array([[0.499, 0.598], [0.697, 0.796]])
    expected_b2 = np.array([[0.299], [0.398]])
    
    assert np.allclose(updated_layers[0]["w"], expected_w1)
    assert np.allclose(updated_layers[0]["b"], expected_b1)
    assert np.allclose(updated_layers[1]["w"], expected_w2)
    assert np.allclose(updated_layers[1]["b"], expected_b2)
    
    # 验证损失
    expected_loss = loss_fun(fead_forward.return_value[2], labs)
    assert np.allclose(loss, expected_loss)