# 创建数据库引擎管理模块（db_manager.py）
import os
import json
import pandas as pd
import numpy as np
import mysql.connector
from mysql.connector import Error, errorcode
from queue import Queue, Empty
from contextlib import contextmanager
from pathlib import Path
# 注意：确保 JSONConfig 能正确导入，若有路径问题需调整 sys.path
from utils.JSONConfig import ConfigHandler

class MySQLHelper:
    # 【改动】不再是单个实例，而是实例缓存池：key=db名称，value=MySQLHelper实例
    # 待修正：数据库密码不能放到实例中。
    _instance_pool = {}

    def __new__(cls, *args, **kwargs):
        """
        改造单例：按db名称隔离实例
        使用方式不变：db = MySQLHelper(db='jhd')
        同一个db多次调用，返回同一个连接池实例；不同db返回各自独立实例
        """
        # 获取传入的db参数，默认ginger2
        db_name = kwargs.get("db", "ginger2")
        if db_name not in cls._instance_pool:
            # 不存在，则新建实例放入池
            new_obj = super().__new__(cls)
            cls._instance_pool[db_name] = new_obj
        return cls._instance_pool[db_name]

    def __init__(self, config_path=r'D:\pyprj\cv2\utils\config.json', db='ginger2'):
        self.config_path = config_path
        self.db = db
        # 标记是否已经初始化过，防止重复初始化（关键）
        if hasattr(self, "_initialized") and self._initialized:
            return

        # 首次初始化才加载配置、创建连接队列
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
        # JSON 字段清单：配置里没有 → None（本库无 JSON 字段）
        # 用 frozenset：O(1) 查找、不可变、内存友好
        js = cfg.get('jsfields')
        self.jsfields = frozenset(js) if js else None
        self._initialize_connections()
        # 设置初始化标记，后续__init__直接跳过
        self._initialized = True

    def _initialize_connections(self):
        """初始化连接池，最多填充到 min(3, max_connections) 个。
        修复：put 前检查 qsize，避免队列已满时 put 阻塞（truncate 场景）"""
        target = min(3, self.max_connections)
        while self.connections.qsize() < target:
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
            # 连接失效则丢弃
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
        """ 关闭当前db实例下所有连接 """
        while not self.connections.empty():
            try:
                connection = self.connections.get_nowait()
                if connection.is_connected():
                    try:
                        if not connection.autocommit:
                            connection.rollback()
                    except:
                        pass
                    connection.close()
            except Empty:
                break
        print(f"[{self.db}] 所有连接已关闭")

    @classmethod
    def close_all_pool(cls):
        """【新增类方法】关闭全部db池所有连接，程序退出时调用清理资源"""
        for db_name, helper in cls._instance_pool.items():
            helper.close_all()
        cls._instance_pool.clear()
        print("✅ 全部数据库连接池已清空")

    # ==================== JSON 字段处理工具 ====================
    #
    # 设计要点：
    #   1) self.jsfields 为 frozenset 或 None；
    #   2) 每个入口方法第一句 `if not self.jsfields: return` 短路，
    #      配置里没写 jsfields 的库 → 与改造前完全一致，零额外开销；
    #   3) 写入：dict/list → JSON 字符串；已是字符串则透传（兼容旧代码）；
    #   4) 读取：JSON 字符串 → dict/list；解析失败原样返回（兼容脏数据）。
    #

    def _json_serialize_value(self, v):
        """单值序列化：dict/list/tuple → JSON 字符串；其它原样返回"""
        if v is None:
            return None
        if isinstance(v, str):
            # 用户自己已经 dumps 过 → 透传，不二次编码
            return v
        if isinstance(v, (dict, list, tuple)):
            try:
                return json.dumps(v, ensure_ascii=False, default=str)
            except (TypeError, ValueError) as e:
                print(f"Warning: JSON 序列化失败 {v} → {e}")
                return v
        return v

    def _json_deserialize_value(self, v):
        """单值反序列化：JSON 字符串 → dict/list；失败原样返回"""
        if v is None:
            return None
        if isinstance(v, (dict, list)):
            return v
        # MySQL 部分驱动会把 JSON 列返回为 bytes
        if isinstance(v, (bytes, bytearray)):
            try:
                v = v.decode('utf-8')
            except Exception:
                return v
        if isinstance(v, str):
            s = v.strip()
            if not s or s[0] not in '{["':
                return v
            try:
                return json.loads(s)
            except (ValueError, TypeError):
                return v
        return v

    def _serialize_record(self, data: dict) -> dict:
        """对 dict 数据中属于 jsfields 的字段做序列化（返回新 dict，不改入参）"""
        if not self.jsfields or not isinstance(data, dict):
            return data
        # 只有命中列才做 dumps，未命中列保持原引用
        hit = self.jsfields & data.keys()
        if not hit:
            return data
        return {k: (self._json_serialize_value(v) if k in hit else v)
                for k, v in data.items()}

    def _serialize_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """对 DataFrame 中属于 jsfields 的列做序列化（未命中直接返回原对象）"""
        if not self.jsfields or df is None or df.empty:
            return df
        js_cols = [c for c in df.columns if c in self.jsfields]
        if not js_cols:
            return df
        df = df.copy()
        for c in js_cols:
            df[c] = df[c].apply(self._json_serialize_value)
        return df

    def _deserialize_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """对 DataFrame 中属于 jsfields 的列做反序列化（未命中直接返回原对象）"""
        if not self.jsfields or df is None or df.empty:
            return df
        js_cols = [c for c in df.columns if c in self.jsfields]
        if not js_cols:
            return df
        df = df.copy()
        for c in js_cols:
            df[c] = df[c].apply(self._json_deserialize_value)
        return df

    def insert_single(self, table, data):
        """
        table：对应数据库的表名
        data：字典数据，其中key对应的是table中的列
        """
        # 插入单条记录
        data = self._serialize_record(data)
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
                print(f"❌ 单条插入失败，已回滚。错误信息: {str(e)}")
                raise
            finally:
                # 无论成功失败，确保关闭cursor
                if cursor:
                    cursor.close()
        return 1

    def bulkInsert(self, table, df):
        """ 批量插入DataFrame """
        df = self._serialize_df(df)
        with self.connection_manager() as connection:
            cursor = None
            try:
                cursor = connection.cursor()
                placeholders = ','.join(['%s'] * len(df.columns))
                columns = ','.join(df.columns)
                sql = f"INSERT INTO {table} ({columns}) VALUES ({placeholders})"
                # 性能优化：to_numpy().tolist() 比 [tuple(x) for x in df.values] 快
                data = df.to_numpy().tolist()
                cursor.executemany(sql, data)
                affected = cursor.rowcount
                connection.commit()
                print(f"✅ 批量插入成功，共{len(data)}条")
            except Exception as e:
                if connection:
                    connection.rollback()
                print(f"❌ 批量插入失败，已回滚。错误信息: {str(e)}")
                raise  # 可选：是否向上抛出异常，根据业务需求决定
            finally:
                # 无论成功失败，确保关闭cursor
                if cursor:
                    cursor.close()
        return affected

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
        df = self._serialize_df(df)
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
        return affected

    def bulk_update(self, table, df, key):
        """
        批量一条条更新DataFrame数据
        table: 更新的表名
        key:  唯一索引列，可以是 字符串(单列) 或 list(多列联合主键)
              例如 key='id' 或 key=['user_id', 'trade_date']
        """
        df = self._serialize_df(df)
        with self.connection_manager() as connection:
            cursor = connection.cursor()
            update_count = 0

            # 统一转成列表格式（兼容单列/多列）
            if isinstance(key, str):
                key_columns = [key]
            else:
                key_columns = list(key)

            # ================== 循环外预计算 SQL 片段（性能优化）==================
            set_columns = [col for col in df.columns if col not in key_columns]
            set_clause = ','.join([f"{col} = %s" for col in set_columns])
            where_clause = ' AND '.join([f"{k} = %s" for k in key_columns])
            sql = f"UPDATE {table} SET {set_clause} WHERE {where_clause}"

            for _, row in df.iterrows():
                # 参数顺序：先赋值列，再主键列
                values = [row[col] for col in set_columns] + [row[col] for col in key_columns]
                cursor.execute(sql, values)
                update_count += 1

            connection.commit()
            print(f"批量更新成功，共 {update_count} 条")
            cursor.close()
            return update_count

    def bulkUpd(self, table, df, key):
        """
        高性能批量更新（使用 executemany，比循环快几十倍）
        table: 表名
        key:  主键/唯一键，可以是 str 或 list（多列联合主键）
        """
        # 统一转为列表（兼容单列/多列）
        df = self._serialize_df(df)
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
        return rowcount

    def delete(self, table, condition):
        """ 删除数据（注意condition需自行确保安全，避免SQL注入）"""
        with self.connection_manager() as connection:
            cursor = connection.cursor()
            sql = f"DELETE FROM {table} WHERE {condition}"
            cursor.execute(sql)
            connection.commit()
            cnt = cursor.rowcount
            print(f"删除成功，影响行数：{cnt}")
            cursor.close()
        return cnt

    def delFromDf(self, df, table):
        """
        根据 DataFrame 的 header 和值作为条件批量删除数据
        :param df: pandas DataFrame，列名=表字段名，每行是一组删除条件
        :param table: 数据库表名
        """
        if df.empty:
            print("✅ DataFrame 为空，无需删除")
            return
        df = self._serialize_df(df)
        # 拼接 WHERE 条件：列1=%s AND 列2=%s AND ...
        where_conditions = " AND ".join([f"{col} = %s" for col in df.columns])
        sql = f"DELETE FROM {table} WHERE {where_conditions}"
        # 转换为参数列表（批量执行格式）
        params = df.to_numpy().tolist()
        with self.connection_manager() as conn:
            cursor = conn.cursor()
            try:
                cursor.executemany(sql, params)
                conn.commit()
                cnt = cursor.rowcount
                print(f"✅ 批量删除完成 | 表：{table} | 条件列：{list(df.columns)} | 影响行数：{cnt}")
            except Exception as e:
                conn.rollback()
                print(f"❌ 批量删除失败，已回滚 | 错误信息：{str(e)}")
                raise
            finally:
                cursor.close()
        return cnt

    def execute_raw(self, sql, params=None):
        """执行任意写 SQL（UPDATE / DELETE / 复杂 INSERT），返回影响行数"""
        with self.connection_manager() as conn:
            cursor = conn.cursor()
            try:
                cursor.execute(sql, params or ())
                conn.commit()
                return cursor.rowcount
            except Exception as e:
                conn.rollback()
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
        # 关闭刚释放回队列的连接，再重新初始化，避免 qsize 累加导致 put 阻塞
        self.close_all()
        self._initialize_connections()
        return 0

    def query(self, sql, dct=True):
        """ 执行查询，返回列表（字典/元组格式）；jsfields 命中时自动反序列化 """
        with self.connection_manager() as connection:
            cursor = None
            try:
                cursor = connection.cursor(dictionary=dct)
                cursor.execute(sql)
                result = cursor.fetchall()

                # self.jsfields 为 None → 一句短路，完全跳过
                if self.jsfields and result:
                    cols_list = [d[0] for d in cursor.description]
                    js_idx = [i for i, c in enumerate(cols_list) if c in self.jsfields]

                    # 本次结果里没有 JSON 列 → 也不做任何逐行处理
                    if js_idx:
                        if dct:
                            hit = {cols_list[i] for i in js_idx}
                            for r in result:
                                for k in hit:
                                    if k in r:
                                        r[k] = self._json_deserialize_value(r[k])
                        else:
                            new_result = []
                            for r in result:
                                r = list(r)
                                for i in js_idx:
                                    r[i] = self._json_deserialize_value(r[i])
                                new_result.append(tuple(r))
                            result = new_result

                return result
            except Exception as e:
                print(f"❌ 查询失败，错误信息: {str(e)}")
                raise
            finally:
                if cursor:
                    cursor.close()

    def query2df(self, sql):
        """ 查询并返回DataFrame（用pandas内置方法）；jsfields 命中时自动反序列化 """
        with self.connection_manager() as connection:
            try:
                df = pd.read_sql(sql, connection)
                df = self._deserialize_df(df)   # ← JSON 字段自动解码（未命中直接返回原 df）
                return df
            except Exception as e:
                print(f"❌ 数据读取失败。错误信息: {str(e)}")
                raise

    def qry2df(self, sql):
        """ 手动查询并返回DataFrame（兼容低版本pandas）；jsfields 命中时自动反序列化 """
        with self.connection_manager() as connection:
            cursor = None
            try:
                cursor = connection.cursor()
                cursor.execute(sql)
                columns = [desc[0] for desc in cursor.description]
                data = cursor.fetchall()
                df = pd.DataFrame(data, columns=columns)
                df = self._deserialize_df(df)   # ← JSON 字段自动解码
                return df
            except Exception as e:
                print(f"❌ 数据读取失败。错误信息: {str(e)}")
                raise
            finally:
                if cursor:
                    cursor.close()

    def lookUpColName(self, df):
        """把字段映射为可以理解的短备注
        df: 需要转换为中文标题的df"""
        sql = (
            "select distinct cols.COLUMN_NAME, "
            "SUBSTRING_INDEX(cols.COLUMN_COMMENT,'~',1) as COLUMN_COMMENT "
            "from information_schema.`COLUMNS` as cols "
            f"where cols.TABLE_SCHEMA = '{self.database}'"
        )
        cols = self.qry2df(sql)
        cols["COLUMN_COMMENT"] = cols["COLUMN_COMMENT"].apply(
            lambda x: x.decode('utf-8') if isinstance(x, bytes) else x
        )
        colmap = cols['COLUMN_COMMENT']
        colmap.index = cols['COLUMN_NAME']
        return df.columns.map(colmap)

# -------------------------- 主程序测试代码（仅直接运行时执行） --------------------------
if __name__ == "__main__":
    # 测试：两个不同db，各自独立连接池
    db_jhd = MySQLHelper(db='jhd')
    db_ginger = MySQLHelper(db='ginger2')

    # 测试单条插入（jhd 库配置了 jsfields=['slct_reason']，会自动 JSON 编码）
    data = {'id': 9001, 'slct_reason': {'code': 3, 'msg': 'invalid'}}
    db_jhd.insert_single('ts_import', data)

    # 测试批量插入
    df = pd.DataFrame({
        'id': [9002, 9003],
        'slct_reason': [['a', 'b'], {'k': 1}],
    })
    db_jhd.bulkInsert('ts_import', df)

    # 测试批量更新
    df_update = pd.DataFrame({
        'id': [1, 2],
        'col1': ['new_value1', 'new_value3'],
        'col2': ['new_value2', 'new_value4']
    })
    db_jhd.bulk_update('my_table', df_update, key='id')
    
    #下面测试json字段
    # 方式 1：直接走 SQL（推荐，高效）
    rows = db_jhd.query(
        "SELECT * FROM tst_json WHERE tst_js->>'$.key1' LIKE 'Y%'",
        dct=True
    )
    
    with db_jhd.connection_manager() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE tst_json
            SET tst_js = JSON_SET(tst_js, '$.key1', CONCAT('Y', tst_js->>'$.key1'))
            WHERE id = %s AND tst_js->>'$.key1' IS NOT NULL
        """, (12345,))
        conn.commit()
        print(f"影响行数: {cursor.rowcount}")
        cursor.close()

    df = db_jhd.qry2df(sql="select * from my_choice_stock")
    df['slct_reason'] = df['slct_reason'].str['市盈率']
    df['slct_reason'] = df['slct_reason'].apply(lambda d: d.get('key1') if isinstance(d, dict) else None)
    
    df = pd.DataFrame({
        'id':     [1, 2, 3],
        'name':   ['A', 'B', 'C'],
        'tst_js': [
            {'key1': 'abc', 'key2': 100},           # dict → 自动 dumps
            {'key1': 'def', 'tags': ['x', 'y']},
            {'key1': 'ghi', 'nested': {'a': 1}},
        ],
    })

    db.bulkInsUpd(df, 'tst_json')
    
    
