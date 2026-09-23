# -*- coding: utf-8 -*-
"""
scheduler.py —— 支持实例池、异步、依赖（OR/AND 分组）、事件驱动、热重载

用法:
    python scheduler.py <配置文件路径>

命令:
    update     重新加载配置文件
    cancel all 取消所有任务和调度计划，退出
    其它输入无效

start_at:
    null / 省略 : 立即
    0           : 立即（now + 0s）
    1           : now + 1s
    "08:30"     : 今天的 08:30（已过则顺延到明天）
    ISO 字符串   : 绝对时间

interval:
    null / ""   : 只在 start_at 调度一次
    0           : 事件驱动（依赖成功后立即执行）
    "ss_5" "mm_2" "hh_1" "D_10" "M_2" "Y_1"   正数间隔（从 start_at 网格推进）
    "M_-1"      : 下个月最后一天
    "Y_-1"      : 下一年最后一天

deps (顶层 OR，组内 AND):
    []                     无依赖
    ["a","b"]              a 成功 或 b 成功
    [["a","b"], "c"]       (a 且 b) 或 c
    [["a","b"],["c","d"]]  (a 且 b) 或 (c 且 d)
"""
from __future__ import annotations

import asyncio
import importlib
import inspect
import json
import logging
import sys
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional

log = logging.getLogger("scheduler")

_GLOBAL_SCHEDULER: Optional["Scheduler"] = None
EVENT = "__EVENT__"     # interval 的特殊值：事件驱动


def get_scheduler() -> "Scheduler":
    if _GLOBAL_SCHEDULER is None:
        raise RuntimeError("调度器尚未启动")
    return _GLOBAL_SCHEDULER


# =========================================================================== #
# 日志
# =========================================================================== #
def setup_logging(log_file: str = "scheduler.log"):
    logger = logging.getLogger("scheduler")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    return logger


# =========================================================================== #
# 日历 / 时间工具
# =========================================================================== #
def _days_in_month(y: int, m: int) -> int:
    if m == 2:
        return 29 if (y % 4 == 0 and (y % 100 != 0 or y % 400 == 0)) else 28
    return (31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)[m - 1]


def _add_months(dt: datetime, n: int) -> datetime:
    total = (dt.year * 12 + (dt.month - 1)) + n
    y = total // 12
    m = total % 12 + 1
    d = min(dt.day, _days_in_month(y, m))
    return dt.replace(year=y, month=m, day=d)


def add_interval(dt: datetime, spec: str) -> Optional[datetime]:
    if spec is None or spec == "":
        return None
    try:
        unit, n_str = spec.split("_", 1)
        n = int(n_str)
    except Exception:
        raise ValueError(f"非法 interval: {spec!r}（示例: ss_5 / mm_2 / D_1 / M_-1 / Y_-1）")

    if n == -1:
        if unit == "M":
            nxt = _add_months(dt.replace(day=1), 1)
            return nxt.replace(day=_days_in_month(nxt.year, nxt.month))
        if unit == "Y":
            return dt.replace(year=dt.year + 1, month=12, day=31)
        raise ValueError(f"负数间隔只支持 M_-1 / Y_-1，收到: {spec!r}")
    if n < -1:
        raise ValueError(f"不支持小于 -1 的负数间隔: {spec!r}")
    if n == 0:
        raise ValueError(f"间隔数量必须 != 0（事件驱动用 interval=0 表达）: {spec!r}")

    if unit == "ss":
        return dt + timedelta(seconds=n)
    if unit == "mm":
        return dt + timedelta(minutes=n)
    if unit == "hh":
        return dt + timedelta(hours=n)
    if unit == "D":
        return dt + timedelta(days=n)
    if unit == "M":
        return _add_months(dt, n)
    if unit == "Y":
        y = dt.year + n
        d = min(dt.day, _days_in_month(y, dt.month))
        return dt.replace(year=y, day=d)
    raise ValueError(f"未知 interval 单位: {unit!r}")


def advance_to_after(base: datetime, interval: str, after: datetime) -> datetime:
    """从 base 起用 interval 推进，返回第一个 > after 的点。"""
    if base > after:
        return base
    try:
        unit, n_str = interval.split("_", 1)
        n = int(n_str)
    except Exception:
        raise ValueError(f"非法 interval: {interval!r}")

    sec = None
    if n > 0:
        if unit == "ss":
            sec = n
        elif unit == "mm":
            sec = n * 60
        elif unit == "hh":
            sec = n * 3600
        elif unit == "D":
            sec = n * 86400
    if sec is not None:
        gap = (after - base).total_seconds()
        steps = int(gap // sec) + 1
        return base + timedelta(seconds=steps * sec)

    cur = base
    for _ in range(100000):
        cur = add_interval(cur, interval)
        if cur is None or cur > after:
            return cur
    return cur


def parse_start_at(s) -> datetime:
    """None/"" -> now, 数字 -> now + N 秒, 'HH:MM[:SS]' -> 今天/明天该时刻, ISO -> 绝对时间"""
    now = datetime.now()
    if s is None or s == "":
        return now
    if isinstance(s, datetime):
        return s
    if isinstance(s, bool):
        raise ValueError(f"start_at 不能是 bool: {s!r}")
    if isinstance(s, (int, float)):
        return now + timedelta(seconds=float(s))

    s = str(s).strip()
    if ":" in s and "-" not in s and "T" not in s:
        parts = [int(p) for p in s.split(":")]
        while len(parts) < 3:
            parts.append(0)
        h, m, sec = parts[:3]
        t = now.replace(hour=h, minute=m, second=sec, microsecond=0)
        if t <= now:
            t += timedelta(days=1)
        return t
    return datetime.fromisoformat(s)


def _normalize_interval(raw) -> Optional[str]:
    """
    统一为：None（只跑一次）/ EVENT（事件驱动）/ "ss_5" 等（时间间隔）
    """
    if raw is None:
        return None
    if isinstance(raw, bool):
        raise ValueError(f"interval 不能是 bool: {raw!r}")
    if isinstance(raw, (int, float)):
        n = int(raw)
        if n < 0:
            raise ValueError(f"interval 数字必须 >= 0: {raw!r}")
        return EVENT if n == 0 else f"ss_{n}"
    s = str(raw).strip()
    if s == "":
        return None
    if s == "0":
        return EVENT
    add_interval(datetime.now(), s)   # 格式校验
    return s


def _parse_deps(raw) -> List[List[str]]:
    """
    统一为 List[List[str]]：外层 OR，内层 AND。
    ["a","b"] -> [["a"],["b"]]
    [["a","b"],"c"] -> [["a","b"],["c"]]
    """
    if not raw:
        return []
    groups: List[List[str]] = []
    for item in raw:
        if isinstance(item, (list, tuple)):
            groups.append([str(x) for x in item])
        else:
            groups.append([str(item)])
    return groups


# =========================================================================== #
# 类 / 函数加载
# =========================================================================== #
def _load_class(path: str) -> type:
    if ":" in path:
        mod_name, cls_name = path.split(":", 1)
    else:
        mod_name, cls_name = path.rsplit(".", 1)
    mod = importlib.import_module(mod_name)
    cls = getattr(mod, cls_name)
    if not isinstance(cls, type):
        raise TypeError(f"{path} 不是类")
    return cls


def _load_module_attr(path: str) -> Any:
    if ":" in path:
        mod_name, attr_path = path.split(":", 1)
    else:
        mod_name, attr_path = path.rsplit(".", 1)
    mod = importlib.import_module(mod_name)
    obj: Any = mod
    for part in attr_path.split("."):
        obj = getattr(obj, part)
    return obj


# =========================================================================== #
# 实例池
# =========================================================================== #
@dataclass
class InstanceSpec:
    name: str
    class_path: str
    args: Dict[str, Any] = field(default_factory=dict)
    lifetime: str = "call"      # call / batch / permanent


class InstancePool:
    def __init__(self):
        self._specs: Dict[str, InstanceSpec] = {}
        self._permanent: Dict[str, Any] = {}

    def update_specs(self, specs: Dict[str, InstanceSpec]) -> None:
        for name, new_spec in specs.items():
            old = self._specs.get(name)
            if old is None:
                continue
            if (old.class_path != new_spec.class_path
                    or old.args != new_spec.args
                    or old.lifetime != new_spec.lifetime):
                self._destroy_permanent(name)
                log.info("实例 [%s] 配置变更，销毁旧实例", name)
        for name in list(self._permanent.keys()):
            if name not in specs:
                self._destroy_permanent(name)
        self._specs = dict(specs)

    def has(self, name: str) -> bool:
        return name in self._specs

    def get_spec(self, name: str) -> Optional[InstanceSpec]:
        return self._specs.get(name)

    def get(self, name: str, batch_ctx: Dict[str, Any]) -> Any:
        spec = self._specs[name]
        if spec.lifetime == "permanent":
            if name not in self._permanent:
                self._permanent[name] = self._create(spec)
                log.info("创建 permanent 实例 [%s] = %s", name, spec.class_path)
            return self._permanent[name]
        if spec.lifetime == "batch":
            if name not in batch_ctx:
                batch_ctx[name] = self._create(spec)
                log.info("创建 batch 实例 [%s] = %s", name, spec.class_path)
            return batch_ctx[name]
        return self._create(spec)

    def close_one(self, name: str, inst: Any) -> None:
        self._close(name, inst)

    def release_batch(self, batch_ctx: Dict[str, Any]) -> None:
        if not batch_ctx:
            return
        for name, inst in batch_ctx.items():
            self._close(name, inst)
            log.info("释放 batch 实例 [%s]", name)
        batch_ctx.clear()

    def shutdown(self) -> None:
        for name in list(self._permanent.keys()):
            self._destroy_permanent(name)

    def _destroy_permanent(self, name: str) -> None:
        inst = self._permanent.pop(name, None)
        if inst is not None:
            self._close(name, inst)

    def _create(self, spec: InstanceSpec) -> Any:
        cls = _load_class(spec.class_path)
        return cls(**spec.args)

    def _close(self, name: str, inst: Any) -> None:
        for method in ("close", "disconnect", "shutdown", "stop"):
            fn = getattr(inst, method, None)
            if callable(fn):
                try:
                    fn()
                except Exception:
                    log.exception("关闭实例 [%s] 出错", name)
                return
        exit_fn = getattr(inst, "__exit__", None)
        if callable(exit_fn):
            try:
                exit_fn(None, None, None)
            except Exception:
                log.exception("__exit__ [%s] 出错", name)


# =========================================================================== #
# 任务
# =========================================================================== #
@dataclass
class Task:
    name: str
    func_spec: str
    interval: Optional[str]            # None=仅一次, EVENT=事件驱动, "ss_5"...
    next_run: Optional[datetime]       # EVENT 或已完成的 once 为 None
    params: Dict[str, Any] = field(default_factory=dict)
    deps_groups: List[List[str]] = field(default_factory=list)  # OR of AND
    validate: int = 1
    timeout: Optional[float] = None
    start_at_raw: Any = None

    last_status: str = "pending"       # pending / ok / failed / skipped
    last_run: Optional[datetime] = None
    last_error: Optional[str] = None
    run_count: int = 0
    fail_count: int = 0
    running: bool = False
    pending_remove: bool = False

    @property
    def is_event(self) -> bool:
        return self.interval == EVENT

    @property
    def is_once(self) -> bool:
        return self.interval is None


# =========================================================================== #
# 调度器
# =========================================================================== #
class Scheduler:
    def __init__(self, name: str = "scheduler"):
        self.name = name
        self._tasks: Dict[str, Task] = {}
        self._cv = threading.Condition()
        self._thread: Optional[threading.Thread] = None
        self._stop = False
        self._config_path: Optional[str] = None
        self._reload_flag = False

        self._instances = InstancePool()

        self._async_loop: Optional[asyncio.AbstractEventLoop] = None
        self._async_thread: Optional[threading.Thread] = None
        self._async_lock = threading.Lock()

    # ------------------------------------------------------------------ #
    # 配置加载 / 热重载
    # ------------------------------------------------------------------ #
    def load_config(self, path: str) -> None:
        self._config_path = path
        self._reload_config()

    def request_reload(self) -> None:
        with self._cv:
            self._reload_flag = True
            self._cv.notify_all()

    def _reload_config(self) -> None:
        if not self._config_path:
            return
        try:
            with open(self._config_path, encoding="utf-8") as f:
                cfg = json.load(f)
        except Exception:
            log.exception("读取配置失败: %s", self._config_path)
            return

        # ---- instances ----
        new_specs: Dict[str, InstanceSpec] = {}
        for name, spec in cfg.get("instances", {}).items():
            try:
                new_specs[name] = InstanceSpec(
                    name=name,
                    class_path=spec["class"],
                    args=dict(spec.get("args", {})),
                    lifetime=spec.get("lifetime", "call"),
                )
            except Exception:
                log.exception("解析实例 [%s] 失败", name)
        self._instances.update_specs(new_specs)

        # ---- tasks ----
        specs = {t["name"]: t for t in cfg.get("tasks", [])}
        with self._cv:
            old_names = set(self._tasks.keys())
            new_names = set(specs.keys())

            for name in old_names - new_names:
                t = self._tasks[name]
                if t.running:
                    log.warning("[%s] 正在运行，标记 pending_remove", name)
                    t.pending_remove = True
                else:
                    del self._tasks[name]
                    log.info("[%s] 已移除", name)

            for name, spec in specs.items():
                if name in self._tasks:
                    self._update_task(self._tasks[name], spec)
                else:
                    try:
                        self._tasks[name] = self._build_task(spec)
                        t = self._tasks[name]
                        nr = t.next_run.strftime("%Y-%m-%d %H:%M:%S") if t.next_run else "事件驱动/仅一次"
                        log.info("[%s] 新增 validate=%s interval=%s next=%s deps=%s",
                                 name, t.validate, t.interval or "(仅一次)", nr,
                                 t.deps_groups or "-")
                    except Exception:
                        log.exception("[%s] 创建失败", name)

            self._cv.notify_all()

    def _build_task(self, spec: Dict[str, Any]) -> Task:
        name = spec["name"]
        func_spec = spec["func"]
        if ":" not in func_spec:
            raise ValueError(f"[{name}] func 必须形如 'module:function' 或 'instance:method'")

        interval = _normalize_interval(spec.get("interval"))
        deps_groups = _parse_deps(spec.get("deps", []))
        start_raw = spec.get("start_at")

        if interval == EVENT:
            next_run = None                       # 事件驱动，不参与时间调度
            if not deps_groups:
                log.warning("[%s] interval=0（事件驱动）但未声明 deps，永远不会触发", name)
        elif interval is None:
            next_run = parse_start_at(start_raw)  # 只跑一次，等 start_at 到
        else:
            start = parse_start_at(start_raw)
            next_run = advance_to_after(start, interval, datetime.now())

        return Task(
            name=name,
            func_spec=func_spec,
            interval=interval,
            next_run=next_run,
            params=dict(spec.get("params", {})),
            deps_groups=deps_groups,
            validate=int(spec.get("validate", 1)),
            timeout=spec.get("timeout"),
            start_at_raw=start_raw,
        )

    def _update_task(self, t: Task, spec: Dict[str, Any]) -> None:
        note = "（任务正在运行，本次不打断）" if t.running else ""

        new_func_spec = spec["func"]
        if new_func_spec != t.func_spec:
            t.func_spec = new_func_spec
            log.info("[%s] func 更新为 %s %s", t.name, new_func_spec, note)

        new_params = dict(spec.get("params", {}))
        if new_params != t.params:
            t.params = new_params
            log.info("[%s] 参数更新 %s", t.name, note)

        new_deps = _parse_deps(spec.get("deps", []))
        if new_deps != t.deps_groups:
            t.deps_groups = new_deps
            log.info("[%s] 依赖更新 %s", t.name, note)

        new_validate = int(spec.get("validate", 1))
        was = t.validate
        t.validate = new_validate
        if new_validate != was:
            log.info("[%s] validate 更新为 %s %s", t.name, new_validate, note)

        t.timeout = spec.get("timeout")

        new_interval = _normalize_interval(spec.get("interval"))
        interval_changed = new_interval != t.interval
        if interval_changed:
            t.interval = new_interval
            log.info("[%s] interval 更新为 %r %s", t.name, new_interval, note)

        new_start_raw = spec.get("start_at")
        start_changed = str(new_start_raw) != str(t.start_at_raw)
        if start_changed:
            t.start_at_raw = new_start_raw
            if t.interval == EVENT:
                t.next_run = None
            elif t.interval is None:
                t.next_run = parse_start_at(new_start_raw)
            else:
                t.next_run = advance_to_after(parse_start_at(new_start_raw),
                                              t.interval, datetime.now())
            log.info("[%s] start_at 更新 next=%s %s", t.name,
                     t.next_run.strftime("%Y-%m-%d %H:%M:%S") if t.next_run else "None", note)
        elif interval_changed or (t.validate == 1 and was == 0):
            if t.interval == EVENT:
                t.next_run = None
            elif t.interval is None:
                pass
            elif t.next_run is not None:
                t.next_run = advance_to_after(t.next_run, t.interval, datetime.now())

    # ------------------------------------------------------------------ #
    # 动态参数维护
    # ------------------------------------------------------------------ #
    def update_params(self, name: str, **kwargs) -> None:
        with self._cv:
            if name in self._tasks:
                self._tasks[name].params.update(kwargs)
                self._cv.notify_all()
                log.info("动态更新 [%s] 参数: %s", name, kwargs)

    def set_params(self, name: str, params: Dict[str, Any]) -> None:
        with self._cv:
            if name in self._tasks:
                self._tasks[name].params = dict(params)
                self._cv.notify_all()

    # ------------------------------------------------------------------ #
    # 异步事件循环
    # ------------------------------------------------------------------ #
    def _ensure_async_loop(self) -> asyncio.AbstractEventLoop:
        with self._async_lock:
            if self._async_loop is not None and self._async_loop.is_running():
                return self._async_loop
            loop = asyncio.new_event_loop()

            def runner():
                asyncio.set_event_loop(loop)
                log.info("asyncio 事件循环启动")
                loop.run_forever()
                try:
                    pending = asyncio.all_tasks(loop)
                    for t in pending:
                        t.cancel()
                    if pending:
                        loop.run_until_complete(
                            asyncio.gather(*pending, return_exceptions=True))
                except Exception:
                    pass
                loop.close()
                log.info("asyncio 事件循环关闭")

            self._async_loop = loop
            self._async_thread = threading.Thread(
                target=runner, name=f"{self.name}-async", daemon=True)
            self._async_thread.start()
            return loop

    def _stop_async_loop(self):
        with self._async_lock:
            loop = self._async_loop
            self._async_loop = None
        if loop and loop.is_running():
            loop.call_soon_threadsafe(loop.stop)
        if self._async_thread:
            self._async_thread.join(timeout=3)

    # ------------------------------------------------------------------ #
    # 启停
    # ------------------------------------------------------------------ #
    def start(self) -> "Scheduler":
        if self._thread and self._thread.is_alive():
            return self
        self._stop = False
        self._thread = threading.Thread(target=self._loop, name=self.name, daemon=True)
        self._thread.start()
        return self

    def stop(self, timeout: float = 3) -> None:
        with self._cv:
            running = [t.name for t in self._tasks.values() if t.running]
            if running:
                log.warning("正在运行的任务: %s（最多等 %.0fs）", ", ".join(running), timeout)
            self._stop = True
            self._cv.notify_all()
        if self._thread:
            self._thread.join(timeout)
        self._stop_async_loop()
        self._instances.shutdown()
        log.info("调度器已停止")

    # ------------------------------------------------------------------ #
    # 依赖检查
    # ------------------------------------------------------------------ #
    def _deps_blocked_reason(self, task: Task) -> Optional[str]:
        """
        返回 None 表示依赖满足（可以执行）。
        返回字符串表示被阻塞的原因。
        规则：
          - 外层 OR：任一 AND 组满足即可
          - AND 组内：所有依赖都必须 last_status == "ok" 且不在 running
          - 依赖不存在 / 无效 / pending / failed / skipped / running 都视为不满足
        """
        if not task.deps_groups:
            return None

        reasons: List[str] = []
        for group in task.deps_groups:
            group_ok = True
            group_why: List[str] = []
            for dep_name in group:
                dep = self._tasks.get(dep_name)
                if dep is None:
                    group_ok = False
                    group_why.append(f"{dep_name}(未注册)")
                    break
                if not dep.validate:
                    group_ok = False
                    group_why.append(f"{dep_name}(已禁用)")
                    break
                if dep.running:
                    group_ok = False
                    group_why.append(f"{dep_name}(运行中)")
                    break
                if dep.last_status != "ok":
                    group_ok = False
                    group_why.append(f"{dep_name}({dep.last_status})")
                    break
            if group_ok:
                return None
            reasons.append("(" + "+".join(group_why) + ")")
        return " or ".join(reasons)

    # ------------------------------------------------------------------ #
    # 拓扑排序：保证同批内被依赖者先执行
    # ------------------------------------------------------------------ #
    def _topo_sort(self, tasks: List[Task]) -> List[Task]:
        by_name = {t.name: t for t in tasks}
        visited: set = set()
        in_stack: set = set()
        result: List[Task] = []

        def visit(t: Task):
            if t.name in visited:
                return
            if t.name in in_stack:
                log.warning("检测到循环依赖，跳过: %s", t.name)
                return
            in_stack.add(t.name)
            for group in t.deps_groups:
                for dep_name in group:
                    dep = by_name.get(dep_name)
                    if dep is not None:
                        visit(dep)
            in_stack.discard(t.name)
            visited.add(t.name)
            result.append(t)

        for t in tasks:
            visit(t)
        return result

    # ------------------------------------------------------------------ #
    # 主循环
    # ------------------------------------------------------------------ #
    def _loop(self):
        log.info("调度线程启动")
        while True:
            # ---- 重载 ----
            do_reload = False
            with self._cv:
                if self._stop:
                    break
                if self._reload_flag:
                    self._reload_flag = False
                    do_reload = True
            if do_reload:
                try:
                    self._reload_config()
                except Exception:
                    log.exception("重载配置失败")

            # ---- 找时间到期的任务（不含事件驱动）----
            with self._cv:
                if self._stop:
                    break
                now = datetime.now()
                due: List[Task] = []
                earliest: Optional[datetime] = None
                for t in self._tasks.values():
                    if not t.validate or t.running or t.pending_remove:
                        continue
                    if t.interval == EVENT:
                        continue
                    if t.next_run is None:
                        continue
                    if t.next_run <= now:
                        due.append(t)
                    elif earliest is None or t.next_run < earliest:
                        earliest = t.next_run

                if not due:
                    if earliest is None:
                        self._cv.wait(timeout=1.0)
                    else:
                        timeout = max(0.05, (earliest - now).total_seconds())
                        self._cv.wait(timeout=min(timeout, 1.0))
                    continue
                due.sort(key=lambda x: x.next_run)

            # ---- 执行批次 ----
            self._run_batch(due)

        log.info("调度线程退出")

    def _run_batch(self, due: List[Task]) -> None:
        ordered = self._topo_sort(due)
        queue: List[Task] = list(ordered)
        executed: set = set()
        batch_ctx: Dict[str, Any] = {}

        try:
            while queue:
                task = queue.pop(0)
                if task.name in executed:
                    continue
                with self._cv:
                    if not task.validate or task.running or task.pending_remove:
                        continue
                executed.add(task.name)
                self._run_task(task, batch_ctx)

                if task.pending_remove:
                    with self._cv:
                        self._tasks.pop(task.name, None)
                    log.info("[%s] 已运行完毕，按配置删除", task.name)
                    continue

                # 事件驱动触发：只有成功才触发下游
                if task.last_status == "ok":
                    for t in list(self._tasks.values()):
                        if not t.is_event:
                            continue
                        if t.name in executed:
                            continue
                        if not t.validate or t.running or t.pending_remove:
                            continue
                        if self._deps_blocked_reason(t) is None:
                            queue.append(t)
                            log.info("[%s] 依赖满足，事件驱动触发", t.name)
        finally:
            try:
                self._instances.release_batch(batch_ctx)
            except Exception:
                log.exception("释放 batch 实例失败")

    # ------------------------------------------------------------------ #
    # 函数解析
    # ------------------------------------------------------------------ #
    def _resolve_task_func(self, task: Task, batch_ctx: Dict[str, Any]):
        spec = task.func_spec
        if ":" not in spec:
            raise ValueError(f"[{task.name}] 非法 func: {spec!r}")
        prefix, name = spec.split(":", 1)

        if self._instances.has(prefix):
            spec_obj = self._instances.get_spec(prefix)
            inst = self._instances.get(prefix, batch_ctx)
            fn = getattr(inst, name, None)
            if fn is None:
                raise AttributeError(f"实例 [{prefix}] 没有属性 {name!r}")
            cleanup = (inst, prefix) if spec_obj.lifetime == "call" else None
            return fn, cleanup

        fn = _load_module_attr(spec)
        return fn, None

    # ------------------------------------------------------------------ #
    # 执行
    # ------------------------------------------------------------------ #
    def _run_task(self, task: Task, batch_ctx: Dict[str, Any]) -> None:
        scheduled_time = task.next_run
        task.running = True
        task.last_run = datetime.now()
        cleanup = None

        try:
            blocked = self._deps_blocked_reason(task)
            if blocked:
                task.last_status = "skipped"
                log.warning("[%s] 跳过执行，依赖未满足: %s", task.name, blocked)
            else:
                try:
                    func, cleanup = self._resolve_task_func(task, batch_ctx)
                except Exception as e:
                    task.last_status = "failed"
                    task.fail_count += 1
                    task.last_error = repr(e)
                    log.exception("[%s] 解析 func 失败: %s", task.name, task.func_spec)
                else:
                    log.info("[%s] 开始执行 func=%s params=%s",
                             task.name, task.func_spec, task.params)
                    try:
                        self._execute(task, func)
                    except Exception as e:
                        task.last_status = "failed"
                        task.fail_count += 1
                        task.last_error = repr(e)
                        log.exception("[%s] 执行失败", task.name)
                    else:
                        task.last_status = "ok"
                        task.run_count += 1
                        task.last_error = None
                        log.info("[%s] 执行成功", task.name)
        finally:
            if cleanup is not None:
                inst, iname = cleanup
                try:
                    self._instances.close_one(iname, inst)
                except Exception:
                    log.exception("关闭 call 实例 [%s] 失败", iname)

            task.running = False
            # next_run 推进
            if task.interval == EVENT:
                task.next_run = None
            elif task.interval is None:
                task.next_run = None           # 只跑一次
            else:
                now = datetime.now()
                task.next_run = advance_to_after(scheduled_time, task.interval, now)
            with self._cv:
                self._cv.notify_all()

    def _execute(self, task: Task, func: Callable) -> None:
        if inspect.iscoroutinefunction(func):
            loop = self._ensure_async_loop()
            coro = func(**task.params)
            future = asyncio.run_coroutine_threadsafe(coro, loop)
            try:
                if task.timeout:
                    future.result(timeout=task.timeout)
                else:
                    future.result()
            except asyncio.TimeoutError:
                future.cancel()
                raise TimeoutError(f"异步任务 {task.name} 超时 {task.timeout}s")
        else:
            func(**task.params)

    # ------------------------------------------------------------------ #
    # 状态
    # ------------------------------------------------------------------ #
    def status(self) -> Dict[str, dict]:
        now = datetime.now()
        with self._cv:
            return {
                name: {
                    "validate": t.validate,
                    "func": t.func_spec,
                    "interval": (t.interval if t.interval else
                                 ("事件驱动" if t.interval == EVENT else "(仅一次)")),
                    "next_run": (t.next_run.strftime("%Y-%m-%d %H:%M:%S")
                                 if t.next_run else "无"),
                    "in_seconds": (round((t.next_run - now).total_seconds(), 1)
                                   if t.next_run else None),
                    "running": t.running,
                    "last_status": t.last_status,
                    "last_error": t.last_error,
                    "ok": t.run_count,
                    "fail": t.fail_count,
                    "deps_groups": [list(g) for g in t.deps_groups],
                    "params": dict(t.params),
                }
                for name, t in self._tasks.items()
            }


# =========================================================================== #
# 主入口
# =========================================================================== #
def sch_main():
    global _GLOBAL_SCHEDULER

    if len(sys.argv) < 2:
        print("用法: python scheduler.py <配置文件路径>")
        sys.exit(1)

    config_path = sys.argv[1]
    setup_logging()

    sch = Scheduler()
    _GLOBAL_SCHEDULER = sch

    try:
        sch.load_config(config_path)
    except Exception:
        log.exception("首次加载配置失败")
        sys.exit(1)

    sch.start()

    print()
    print("=" * 60)
    print(f"调度器已启动 | 配置文件: {config_path}")
    print("命令:")
    print("  update     重新加载配置文件")
    print("  cancel all 取消所有任务和调度计划并退出")
    print("=" * 60)
    print()

    try:
        while True:
            try:
                line = input("scheduler> ").strip()
            except EOFError:
                print("\n输入已关闭，退出...")
                break

            if line == "update":
                sch.request_reload()
                print("已通知调度器重载配置")
            elif line == "cancel all":
                print("正在取消所有任务和调度计划...")
                break
            elif line == "":
                continue
            else:
                print("无效命令。可用: update / cancel all")
    except KeyboardInterrupt:
        print("\n收到中断信号，退出...")
    finally:
        sch.stop(timeout=3)
        print("调度器已退出")


if __name__ == "__main__":
    sch_main()
