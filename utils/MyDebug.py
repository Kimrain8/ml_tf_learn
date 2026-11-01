import sys
from functools import wraps
class MyDiagnose:
    @staticmethod
    def watch_class(*var_names):
        """类装饰器：监控类中所有方法的指定局部变量"""
        def trace(frame, event, arg):
            if event == 'line':
                # 只关心目标变量
                for name in var_names:
                    if name in frame.f_locals:
                        print(f"[{frame.f_code.co_name}] {name} = {frame.f_locals[name]}")
            return trace
    
        def decorator(cls):
            for attr_name, attr_value in cls.__dict__.items():
                if callable(attr_value) and not attr_name.startswith('__'):
                    original = attr_value
    
                    @wraps(original)
                    def wrapper(self, *args, _original=original, **kwargs):
                        old_trace = sys.gettrace()
                        sys.settrace(trace)
                        try:
                            return _original(self, *args, **kwargs)
                        finally:
                            sys.settrace(old_trace)
    
                    setattr(cls, attr_name, wrapper)
            return cls
        return decorator
    
    @staticmethod
    def watch_func(*var_names):
        """监控函数中的指定的局部变量"""
        def trace(frame, event, arg):
            if event == 'line':
                for name in var_names:
                    if name in frame.f_locals:
                        print(f"[{frame.f_code.co_name}] {name} = {frame.f_locals[name]}")
            return trace
    
        def decorator(func):
            def wrapper(*args, **kwargs):
                old_trace = sys.gettrace()
                sys.settrace(trace)
                try:
                    return func(*args, **kwargs)
                finally:
                    sys.settrace(old_trace)
            return wrapper
        return decorator
    
class MyMonitored:
    """描述符：为每个属性单独监控"""
    def __set_name__(self, owner, name):
        self.name = name

    def __get__(self, instance, owner):
        return instance.__dict__.get(self.name)

    def __set__(self, instance, value):
        old = instance.__dict__.get(self.name, 'N/A')
        print(f"[{type(instance).__name__}] {self.name}: {old} -> {value}")
        instance.__dict__[self.name] = value
    

# 使用示例
if __name__ == '__main__':
    @MyDiagnose.watch_class('x', 'y')
    class Calculator:
        def compute(self):
            x = 10
            y = x * 2
            x = y + 5
            return x
    
        def another(self):
            x = 100
            y = 200
            return x + y


    @MyDiagnose.watch_func('x','y')
    def calculate():
        x = 10
        y = x * 2
        x = y + 5
        return x
    
    # calc = Calculator()
    # calc.compute()

    class Calculator2:
        """用于测试 使用MyMonitored 类来监控类的成员变量"""
        x = Monitored()
        y = Monitored()
    
        def compute(self):
            self.x = 10
            self.x = self.x * 2
            self.y = self.x + 5

    
    