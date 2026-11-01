import sys
import os
import inspect

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch


print("=== 1. config.py 检查 ===")
from config import Config
c = Config()
for f in ['seed', 'patience', 'weight_decay', 'n_patches', 'total_tokens',
          'n_entity_tokens', 'check_model_ready']:
    print(f"  Config.{f}: {'OK' if hasattr(c, f) else '❌ 缺失'}")
print(f"  n_patches={c.n_patches}, total_tokens={c.total_tokens}  (期望 12 / 21)")

print("\n=== 2. tokenizers.py 检查 ===")
from tokenizers import StockDataTokenizer
mro_names = [cls.__name__ for cls in StockDataTokenizer.__mro__]
print(f"  继承 nn.Module: {'OK' if 'Module' in mro_names else '❌'}")
tok_src = inspect.getsource(StockDataTokenizer)
print(f"  有 pad_token 参数: {'OK' if 'self.pad_token' in tok_src else '❌'}")
print(f"  forward 返回 tuple: {'OK' if 'return tokens,' in tok_src else '❌'}")
print(f"  CLS 为可学习 Parameter: {'OK' if 'self.cls_token = nn.Parameter' in tok_src else '❌'}")

print("\n=== 3. model.py 检查 ===")
import model
model_src = inspect.getsource(model.StockTransformer.forward)
print(f"  传了 src_key_padding_mask: {'OK' if 'src_key_padding_mask' in model_src else '❌'}")

print("\n=== 4. data_pipeline.py 检查 ===")
import data_pipeline
print(f"  有 _load_csv: {'OK' if hasattr(data_pipeline, '_load_csv') else '❌'}")
print(f"  有 _build_competitor_map: {'OK' if hasattr(data_pipeline, '_build_competitor_map') else '❌'}")
print(f"  有 _pick_industry_name: {'OK' if hasattr(data_pipeline, '_pick_industry_name') else '❌'}")
dp_src = inspect.getsource(data_pipeline._load_csv)
print(f"  .1 后缀列重命名处理: {'OK' if '申万行业二级名称' in dp_src else '❌'}")
print(f"  三级→二级→一级扩展: {'OK' if 'sw2_groups' in inspect.getsource(data_pipeline._build_competitor_map) else '❌'}")

print("\n=== 5. business_classifier.py 检查 ===")
import business_classifier
print(f"  PRODUCT_KEYWORDS 类别数: {len(business_classifier.PRODUCT_KEYWORDS)}")
print(f"  有 classify_main_business: {'OK' if hasattr(business_classifier, 'classify_main_business') else '❌'}")
print(f"  有 tokenize_for_tfidf: {'OK' if hasattr(business_classifier, 'tokenize_for_tfidf') else '❌'}")
print(f"  有 NUM_PRODUCT_CLASSES: {'OK' if hasattr(business_classifier, 'NUM_PRODUCT_CLASSES') else '❌'}")

print("\n=== 6. train.py 检查 ===")
import train
train_src = inspect.getsource(train)
for name in ['argparse', 'set_seed', 'class_weights_from_returns', 'evaluate',
             'build_stock_dataset', 'ReduceLROnPlateau', 'clip_grad_norm_']:
    print(f"  {name}: {'OK' if name in train_src else '❌'}")
print(f"  __main__ 调 main(): {'OK' if 'if __name__ ==' in train_src and 'main()' in train_src else '❌'}")

print("\n=== 7. dataset.py 检查 ===")
import dataset
ds_src = inspect.getsource(dataset)
print(f"  用 torch.as_tensor: {'OK' if 'as_tensor' in ds_src else '❌'}")
print(f"  trend 用 numpy 算: {'OK' if 'np.ones' in ds_src else '❌'}")

print("\n=== 8. inference.py 检查 ===")
import inference
print(f"  有 load_model: {'OK' if hasattr(inference, 'load_model') else '❌'}")
print(f"  有 industry_name_to_id: {'OK' if hasattr(inference, 'industry_name_to_id') else '❌'}")
print(f"  有 product_id_from_business: {'OK' if hasattr(inference, 'product_id_from_business') else '❌'}")
inf_src = inspect.getsource(inference)
print(f"  torch.load 带 map_location: {'OK' if 'map_location' in inf_src else '❌'}")

print("\n=== 9. 依赖检查 ===")
for mod in ['jieba', 'sklearn', 'pandas', 'numpy']:
    try:
        __import__(mod)
        print(f"  {mod}: OK")
    except ImportError as e:
        print(f"  {mod}: ❌ {e}")

print("\n=== 全部检查完成 ===")

# 训练（默认 val_days=0，全部训练）：
# C:/Users/kim/.virtualenvs/cv2-AFZlAyVW/Scripts/python.exe stock2/train.py
# 推理——用训练好的模型跑另一批数据（DB）：
# C:/Users/kim/.virtualenvs/cv2-AFZlAyVW/Scripts/python.exe stock2/predict.py --ckpt checkpoints/best.pt  --sql "select ... from ... where qq.trade_date >= '2026-10-29'"
# C:/Users/kim/.virtualenvs/cv2-AFZlAyVW/Scripts/python.exe stock2/predict.py --ckpt checkpoints/best.pt --csv other_data.csv