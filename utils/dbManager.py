# 创建数据库引擎管理模块（db_manager.py）
import os
import pandas as pd
import numpy as np
import mysql.connector
from mysql.connector import Error,errorcode
from queue import Queue, Empty
from contextlib import contextmanager
from pathlib import Path
# 注意：确保 JSONConfig 能正确导入，若有路径问题需调整 sys.path
from utils.JSONConfig import ConfigHandler

class MySQLHelper:
    _instance = None  # 单例模式
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(MySQLHelper, cls).__new__(cls)
        return cls._instance
    
    def __init__(self, config_path=r'D:\pyprj\cv2\utils\config.json', db='ginger2'):
        self.config_path = config_path
        # 避免重复初始化
        if getattr(self, 'db', None) != db:
            self.db = db
            cfghandle = ConfigHandler(self.config_path)
            cfg = cfghandle.config_data[self.db]
            cfg['password'] = cfghandle.get_decrypt((self.db, 'password'))
            self.host = cfg['host']
            self.port = cfg['port']
            self.user = cfg['username']
            self.password = cfg['password']
            self.database = cfg['dbname']
            self.max_connections = cfg['maxconnection']
            self.connections = Queue(maxsize=self.max_connections)
            self._initialize_connections()
    
    # def _initialize_connections(self):
    #     # 初始化3个连接（不超过最大连接数）
    #     init_count = min(3, self.max_connections)
    #     for _ in range(init_count):
    #         self._create_connection()

    def _initialize_connections(self):
        init_count = min(3, self.max_connections)
        for _ in range(init_count):
            conn = self._create_connection()
            if conn:
                self.connections.put(conn)
            else:
                print('初始化连接失败')
                raise

    def _create_connection(self):
        """创建数据库连接，失败重试最多 3 次，彻底失败返回 None"""
        attempts = 3
        for i in range(attempts):
            try:
                connection = mysql.connector.connect(
                    host=self.host,
                    port=self.port,
                    auth_plugin='mysql_native_password',
                    user=self.user,
                    password=self.password,
                    database=self.database,
                    # charset='utf8mb4',
                    autocommit=False  # 保持显式事务控制，但确保在连接管理器中处理
                )
                return connection

            except Error as err:
                print(f"[创建连接失败 第{i+1}/{attempts}次] 原因: {err}")
                if i + 1 >= attempts:
                    print(f"❌ 已达到最大重试次数 {attempts}，放弃创建连接")
                    return None

        return None

    
    def _get_connection(self):
        try:
            # 从队列拿连接
            connection = self.connections.get(timeout=2)
            # 检查是否存活
            if not connection.is_connected():
                print("连接失效，尝试重新连接...")
                # 这里只尝试重新连接一次，失败就扔掉
                new_conn = self._create_connection()
                if new_conn:
                    return new_conn
                else:
                    raise Exception("无法重新建立连接")
            return connection
        except Empty:
            # 队列空了 → 尝试新建连接（最多3次）
            print("连接池空，尝试新建连接...")
            new_conn = self._create_connection()
            if new_conn:
                return new_conn
            else:
                raise Exception("❌ 连接池耗尽，且无法创建新连接（已达最大重试次数）")

    
    def _release_connection(self, connection):
        """ 释放连接回队列 """
        if connection.is_connected():
            # 【关键修复1】连接放回队列前，确保事务已提交/回滚，释放锁
            try:
                if not connection.autocommit:
                    connection.rollback()  # 未提交的事务统一回滚，避免锁遗留
            except Exception as e:
                print(f"Warning: Failed to rollback on release: {e}")
                
            if self.connections.qsize() < self.max_connections:
                self.connections.put(connection)
            else:
                connection.close()
                print("关闭多余连接")
        else:
            # 连接失效则重建
            # self._create_connection()
            print("Connection is dead, skip recreating in release")
    
    @contextmanager
    def connection_manager(self):
        connection = self._get_connection()
        try:
            yield connection
            # 【关键修复3】正常操作后显式提交（即使是查询，提交也无害，确保事务结束）
            if not connection.autocommit:
                connection.commit()
        except Exception as e:
            # 出错时回滚，避免事务挂起
            if connection.is_connected() and not connection.autocommit:
                connection.rollback()
            print(f"Error during database operation: {e}")
            raise
        finally:
            self._release_connection(connection)
    
    def close_all(self):
        """ 关闭所有连接 """
        temp_connections = []
        while not self.connections.empty():
            # connection = self.connections.get()
            # connection.close()
            try:
                connection = self.connections.get_nowait()
                if connection.is_connected():
                    try:
                        if not connection.autocommit:
                            connection.rollback()
                    except:
                        pass
                    connection.close()
                temp_connections.append(connection)
            except Empty:
                break
        print("所有连接已关闭")
    
    def insert_single(self, table, data):
        """
        table：对应数据库的表名
        data：字典数据，其中key对应的是table中的列
        """
        # 插入单条记录
        with self.connection_manager() as connection:
            cursor = None
            try:
                cursor = connection.cursor()
                placeholders = ','.join(['%s'] * len(data))
                columns = ','.join(data.keys())
                sql = f"INSERT INTO {table} ({columns}) VALUES ({placeholders})"
                cursor.execute(sql, list(data.values()))
                connection.commit()
                print("单条插入成功")
            except Exception as e:
                if connection:
                    connection.rollback()
                print(f"❌ 批量插入失败，已回滚。错误信息: {str(e)}")
                raise
            finally:
                # 无论成功失败，确保关闭cursor
                if cursor:
                    cursor.close()
    
    def bulkInsert(self, table, df):
        """ 批量插入DataFrame """
        
        with self.connection_manager() as connection:
            cursor = None
            try:
                cursor = connection.cursor()
                placeholders = ','.join(['%s'] * len(df.columns))
                columns = ','.join(df.columns)
                sql = f"INSERT INTO {table} ({columns}) VALUES ({placeholders})"
                data = [tuple(x) for x in df.values]
                cursor.executemany(sql, data)
                connection.commit()
                print(f"✅ 批量插入成功，共{len(data)}条")
                # cursor.close()
            except Exception as e:
                if connection:
                    connection.rollback()
                print(f"❌ 批量插入失败，已回滚。错误信息: {str(e)}")
                raise  # 可选：是否向上抛出异常，根据业务需求决定
            finally:
                # 无论成功失败，确保关闭cursor
                if cursor:
                    cursor.close()
    def bulkInsUpd(self, df, table):
        """
        插入或更新（有唯一索引冲突就更新，无冲突就插入）
        依赖：表必须设置 主键 / 唯一索引
        性能：executemany 批量执行，比循环快100倍
        """
        if df.empty:
            print("✅ DataFrame 为空，无需操作")
            return
        df = df.replace({np.nan: None})
        # 列和占位符
        columns = list(df.columns)
        col_str = ", ".join(columns)
        placeholders = ", ".join(["%s"] * len(columns))
    
        # 生成 ON DUPLICATE KEY UPDATE 子句
        update_clause = ", ".join([f"{col} = VALUES({col})" for col in columns])
    
        # 最终SQL（MySQL 专用，高性能）
        sql = f"""
            INSERT INTO {table} ({col_str})
            VALUES ({placeholders})
            ON DUPLICATE KEY UPDATE {update_clause}
        """
    
        # 批量参数
        params = df.to_numpy().tolist()
    
        # 执行
        with self.connection_manager() as conn:
            cursor = conn.cursor()
            cursor.executemany(sql, params)
            conn.commit()
            affected = cursor.rowcount
            cursor.close()
            print(f"✅ insert_or_update 执行完成 | 影响行数：{affected} | 总数据：{len(df)}")
            
    def query(self, sql, dct=True):
        """ 执行查询，返回列表（字典/元组格式）"""
        with self.connection_manager() as connection:
            try:
                cursor = connection.cursor(dictionary=dct)
                cursor.execute(sql)
                result = cursor.fetchall()
                cursor.close()
                return result
            except Exception as e:
                print(f"❌ 批量插入失败，已回滚。错误信息: {str(e)}")
                raise
            finally:
                # 无论成功失败，确保关闭cursor
                if cursor:
                    cursor.close()
    
    def query2df(self, sql):
        """ 查询并返回DataFrame（用pandas内置方法）"""
        with self.connection_manager() as connection:
            try:
                df = pd.read_sql(sql, connection)
                return df
            except Exception as e:
                print(f"❌ 数据读取失败。错误信息: {str(e)}")
                raise
    
    def qry2df(self, sql):
        """ 手动查询并返回DataFrame（兼容低版本pandas）"""
        with self.connection_manager() as connection:
            try:
                cursor = connection.cursor()
                cursor.execute(sql)
                columns = [desc[0] for desc in cursor.description]
                data = cursor.fetchall()
                cursor.close()
                # 修复：Dataframe改为DataFrame
                df = pd.DataFrame(data, columns=columns)
                return df
            except Exception as e:
                print(f"❌ 数据读取失败。错误信息: {str(e)}")
                raise
            finally:
                # 无论成功失败，确保关闭cursor
                if cursor:
                    cursor.close()
    
    def bulk_update(self, table, df, key):
        """
        批量一条条更新DataFrame数据
        table: 更新的表名
        key:  唯一索引列，可以是 字符串(单列) 或 list(多列联合主键)
              例如 key='id' 或 key=['user_id', 'trade_date']
        """
        with self.connection_manager() as connection:
            cursor = connection.cursor()
            update_count = 0
    
            # 统一转成列表格式（兼容单列/多列）
            if isinstance(key, str):
                key_columns = [key]
            else:
                key_columns = list(key)
    
            for index, row in df.iterrows():
                # ================== 生成 SET 子句（排除主键）==================
                set_columns = [col for col in df.columns if col not in key_columns]
                set_clause = ','.join([f"{col} = %s" for col in set_columns])
    
                # ================== 生成 WHERE 子句（多列用 AND 连接）==================
                where_clause = ' AND '.join([f"{k} = %s" for k in key_columns])
    
                # ================== 拼接最终 SQL ==================
                sql = f"UPDATE {table} SET {set_clause} WHERE {where_clause}"
    
                # ================== 参数顺序：先赋值列，再主键列 ==================
                values = [row[col] for col in set_columns] + [row[col] for col in key_columns]
    
                cursor.execute(sql, values)
                update_count += 1
    
            connection.commit()
            print(f"批量更新成功，共 {update_count} 条")
            cursor.close()

    def bulkUpd(self, table, df, key):
        """
        高性能批量更新（使用 executemany，比循环快几十倍）
        table: 表名
        key:  主键/唯一键，可以是 str 或 list（多列联合主键）
        """
        # 统一转为列表（兼容单列/多列）
        if isinstance(key, str):
            key_cols = [key]
        else:
            key_cols = list(key)
    
        # 要更新的字段（排除主键）
        update_cols = [col for col in df.columns if col not in key_cols]
        if not update_cols:
            print("没有需要更新的字段")
            return
    
        # 拼接 SQL
        set_clause = ", ".join([f"{col} = %s" for col in update_cols])
        where_clause = " AND ".join([f"{col} = %s" for col in key_cols])
        sql = f"UPDATE {table} SET {set_clause} WHERE {where_clause}"
    
        # 构造参数列表：[更新字段...] + [主键字段...]
        param_order = update_cols + key_cols
        params = df[param_order].to_numpy().tolist()  # 超快
    
        # executemany 批量执行（核心：只发一次请求到 MySQL）
        with self.connection_manager() as conn:
            cursor = conn.cursor()
            cursor.executemany(sql, params)
            conn.commit()
            rowcount = cursor.rowcount
            cursor.close()
            print(f"✅ 批量更新完成 | 影响行数：{rowcount} | 总条数：{len(df)}")

    
    def delete(self, table, condition):
        """ 删除数据（注意condition需自行确保安全，避免SQL注入）"""
        with self.connection_manager() as connection:
            cursor = connection.cursor()
            sql = f"DELETE FROM {table} WHERE {condition}"
            cursor.execute(sql)
            connection.commit()
            print(f"删除成功，影响行数：{cursor.rowcount}")
            cursor.close()

    def delFromDf(self, df, table):
        """
        根据 DataFrame 的 header 和值作为条件批量删除数据
        :param df: pandas DataFrame，列名=表字段名，每行是一组删除条件
        :param table: 数据库表名
        """
        if df.empty:
            print("✅ DataFrame 为空，无需删除")
            return

        # 拼接 WHERE 条件：列1=%s AND 列2=%s AND ...
        where_conditions = " AND ".join([f"{col} = %s" for col in df.columns])
        sql = f"DELETE FROM {table} WHERE {where_conditions}"

        # 转换为参数列表（批量执行格式）
        params = df.to_numpy().tolist()

        # 执行批量删除
        with self.connection_manager() as conn:
            cursor = conn.cursor()
            try:
                cursor.executemany(sql, params)
                conn.commit()
                print(f"✅ 批量删除完成 | 表：{table} | 条件列：{list(df.columns)} | 影响行数：{cursor.rowcount}")
            except Exception as e:
                conn.rollback()
                print(f"❌ 批量删除失败，已回滚 | 错误信息：{str(e)}")
                raise
            finally:
                cursor.close()
    
    def truncate(self, table):
        # 清空表（慎用，不可恢复）
        self.close_all()
        with self.connection_manager() as connection:
            cursor = connection.cursor()
            sql = f"TRUNCATE TABLE {table}"
            cursor.execute(sql)
            connection.commit()
            print(f"表{table}已清空")
            cursor.close()
        # 重新初始化连接
        self._initialize_connections()

    def lookUpColName(self,df):
        """把字段映射为可以理解的短备注
        df: 需要转换为中文标题的df"""
        sql = f"""select distinct cols.COLUMN_NAME, \
            SUBSTRING_INDEX(cols.COLUMN_COMMENT,'~',1) as COLUMN_COMMENT \
            from information_schema.`COLUMNS` as cols \
            where cols.TABLE_SCHEMA = 'jhd'"""
        cols = self.qry2df(sql)
        cols["COLUMN_COMMENT"] = cols["COLUMN_COMMENT"].apply(lambda x: x.decode('utf-8') if isinstance(x, bytes) else x)
        colmap = cols['COLUMN_COMMENT']
        colmap.index = cols['COLUMN_NAME']
        return df.columns.map(colmap)


        
        

# -------------------------- 主程序测试代码（仅直接运行时执行） --------------------------
if __name__ == "__main__":
    # 初始化助手
    db = MySQLHelper()
    
    # 测试单条插入
    data = {'id': 9001, '1v5': '测试1'}
    db.insert_single('ts_import', data)
    
    # 测试批量插入
    df = pd.DataFrame({'id': [9002, 9003], '1v5': ['测试2', '测试3']})  # 修复字典语法
    db.bulk_insert('ts_import', df)
    
    # 测试查询
    result = db.query('SELECT * FROM ts_import where minid is null', True)
    print("查询结果：", result)
    
    # 测试批量更新
    df_update = pd.DataFrame({
        'id': [1, 2],
        'col1': ['new_value1', 'new_value3'],
        'col2': ['new_value2', 'new_value4']  # 修复co12为col2
    })
    db.bulk_update('my_table', df_update, key='id')