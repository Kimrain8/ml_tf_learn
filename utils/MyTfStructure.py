class MyTfAnalyzer:
    """Transformer 分析工具类"""
    
    def __init__(self, model):
        self.model = model
        self.analysis_results = {}
    
    def analyze_layers(self):
        """分析所有层"""
        results = {}
        
        # 编码器
        results['encoder'] = self._analyze_module(
            self.model.encoder, 
            "Encoder", 
            depth=0
        )
        
        # 解码器
        results['decoder'] = self._analyze_module(
            self.model.decoder, 
            "Decoder", 
            depth=0
        )
        
        self.analysis_results = results
        return results
    
    def _analyze_module(self, module, name, depth):
        """递归分析模块"""
        analysis = {
            'name': name,
            'type': module.__class__.__name__,
            'depth': depth,
            'parameters': {},
            'children': [],
            'total_params': 0
        }
        
        # 统计参数
        for param_name, param in module.named_parameters():
            if param_name.startswith('weight') or param_name.startswith('bias'):
                param_count = param.numel()
                analysis['parameters'][param_name] = {
                    'shape': tuple(param.shape),
                    'count': param_count,
                    'trainable': param.requires_grad
                }
                analysis['total_params'] += param_count
        
        # 递归分析子模块
        for child_name, child in module.named_children():
            child_analysis = self._analyze_module(
                child, 
                f"{name}.{child_name}", 
                depth + 1
            )
            analysis['children'].append(child_analysis)
            analysis['total_params'] += child_analysis['total_params']
        
        return analysis
    
    def print_summary(self):
        """打印汇总信息"""
        if not self.analysis_results:
            self.analyze_layers()
        
        print("\n" + "="*60)
        print("TRANSFORMER 结构分析报告")
        print("="*60)
        
        self._print_module_summary(self.analysis_results['encoder'], 0)
        self._print_module_summary(self.analysis_results['decoder'], 0)
        
        total_params = (
            self.analysis_results['encoder']['total_params'] + 
            self.analysis_results['decoder']['total_params']
        )
        print(f"\n总参数数量: {total_params:,}")
    
    def _print_module_summary(self, module_info, indent):
        """打印模块信息"""
        indent_str = "  " * indent
        print(f"\n{indent_str}{module_info['name']} ({module_info['type']})")
        print(f"{indent_str}  总参数: {module_info['total_params']:,}")
        
        # 打印主要参数
        if module_info['parameters']:
            for param_name, param_info in module_info['parameters'].items():
                print(f"{indent_str}  {param_name}: {param_info['shape']} "
                      f"({param_info['count']:,} params)")
        
        # 递归打印子模块
        for child in module_info['children']:
            self._print_module_summary(child, indent + 2)

if __name__ == "__main__":
    # 使用分析工具
    analyzer = MyTfAnalyzer(transformer)
    analyzer.print_summary()