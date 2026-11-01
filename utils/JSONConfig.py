import json 
import os 
import sys
from cryptography.fernet import Fernet 
from collections.abc import Iterable

class ConfigHandler:
    def __init__(self,config_file,encryption_key=None):
        """
        初始化参数
        :config_file：配置文件路径
        :encryption_key：加密密钥，如果没有提供则生成一个新的密钥
        """
        self.config_file = config_file
        self.encryption_key = os.environ.get("mydbkey") #encryption_key or Fernet.generate_key()
        self.cipher = Fernet(self.encryption_key) 
        self.load_config()
    def get_decrypt(self,kys):
        """
        功能：对指定kys路径的值进行解密
        kys：指定的路径
        """
        if isinstance(kys,Iterable) and isinstance(self.config_data,dict):
            td = self.config_data 
            for ky in kys:
                if isinstance(td[ky],dict):
                    td = td[ky]
                elif isinstance(td[ky],str): 
                    break
            return self.cipher.decrypt(td[ky].encode()).decode() 
        else:
            print(f"非可迭代对象")
            return None
    @staticmethod
    def get_decrypt_str(encry_str):
        encryption_key = os.environ.get("mydbkey")
        cipher = Fernet(encryption_key) 
        return cipher.decrypt(encry_str.encode()).decode() 
    @staticmethod
    def set_encrypt_str(scr_str):
        encryption_key = os.environ.get("mydbkey")
        cipher = Fernet(encryption_key) 
        return cipher.encrypt(scr_str.encode()).decode()
    
    def set_encrypt(self,kys):
        """
        功能：对指定kys路径的值进行加密
        kys：指定的路径
        """
        if isinstance(kys,Iterable) and isinstance(self.config_data,dict):
            td = self.config_data 
            for ky in kys:
                if isinstance(td[ky],dict):
                    td=td[ky]
                elif isinstance(td[ky],str):
                    break
            td[ky]=self.cipher.encrypt(td[ky].encode()).decode()
            return 0
        else:
            print(f"加密找不到对象！")
            return -1
    def upd_config(self,config_data,*kys):
        """
        保存json参数文件，并对需要加密的参数进行加密
        :param  config_data：要保存的配置数据
        kys：可选，保存到的路径，list，
        """
        if len(kys) == 0 and isinstance(config_data,dict):
            for ky in config_data:
                self.config_data[ky] = config_data[ky] 
            return 0

        if len(kys) >= 1:
            td = self.config_data 
            for ky in kys:
                if ky not in td.keys():
                    td[ky]={}
                else:
                    td = td[ky]
            td[ky] = config_data 
            return 0
        else:
            self.config_data[kys] = config_data 
            return 0
    def save_config(self):
        """
        直接保存self.config_data到文件中
        """
        with open(self.config_file,'w') as file:
            json.dump(self.config_data,file,indent=4) 
        return 0
    def load_config(self):
        """
        读取json参数文件，并对已经加密的参数进行解密，解密部分已经移除。 
        """
        try:
            with open(self.config_file,'r') as file:
                self.config_data = json.load(file)
                #if isinstance(self.config_data,dict):
                #get_decrypt(self.config_data,self.cipher) 
                return 0
        except FileNotFoundError:
            self.config_data = {}
            return -1
        except Exception as e:
            print(f"读取配置文件时发生错误：{e}")
            self.config_data = None 
            return -2

    def get_value_from_key(self,*keys):
        def get_decrypt(data,cipher):
            for key in data:
                if isinstance(data[key], dict): 
                    get_decrypt(data[key],cipher)
                elif key in ['password','api_key']:
                    data[key]=cipher.decrypt(data[key].encode()).decode()
        def get_nested_value(data,cipher,*keys):
            if isinstance(data,dict):
                key=keys[0] 
                if key in data:
                    if len(keys)== 1:
                        get_decrypt(data,cipher)
                        return data[key] 
                    else:
                        return get_nested_value(data[key],cipher, *keys[1:])
            return None
        with open(self.config_file,'r') as file: 
            self.config_data = json.load(file)
            return get_nested_value(self.config_data,self.cipher,*keys)

if __name__ == "__main__":
    def setup_iadb():
        """
        测试代码
        """
        project_root = "D:/pyprj/cv2" 
        sys.path.insert(0,project_root)
        sys.path.insert(0,os.path.join(project_root,"utils"))
        
        config_file='config.json'
        handler = ConfigHandler(config_file)
        
        #初始化配置数据
        config_data ={"ginger2":{
            'username':'root',
            'password':'kim789',
            'host':'127.0.0.1',
            'port':'3308',
            'dbname':'ginger2',
            'api_key':'api456',
            'maxconnection':5 
        } 
        }
        handler.upd_config(config_data)
        handler.set_encrypt(('ginger2','password'))
        #保存配置数据
        handler.save_config()
        #print(f"config saved to {config_file}")
        #读取配置数据
        ret = handler.load_config()
        #print(self.config_data)

        print(handler.get_decrypt(("ginger2","password"))) 

    def setup_item():
        """
        测试代码 
        """

        project_root="D:/pyprj/cv2" 
        sys.path.insert(0,project_root)
        sys.path.insert(0,os.path.join(project_root,"utils")) 
        config_file ='config.json'
        handler = ConfigHandler(config_file)
        
        #初始化配置数据
        config_data ={"ITEM":{
            'username':'datagov_read',
            'password':'s5m%d%Eg@7@e$dG7',
            'host':'192.168.97.48',
            'port':'3306',
            'dbname':'mmd_prod',
            'maxconnection':5
        }
        }
        #保存配置数据
        handler.upd_config(config_data)
        handler.set_encrypt(('ITEM','password'))
        handler.save_config()
        #print(f"Config saved to [config_file}")
    setup_iadb()