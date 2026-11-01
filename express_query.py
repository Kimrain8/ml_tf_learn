import json
import xlwings as xw
import requests


class Query:
    def __init__(self, appcode, delivery_status=True, is_sign=True, expName=True, expSite=None, expPhone=None,
                 courier=None, courierPhone=None, updateTime=None, takeTime=True):
        self.appcode = appcode

        # 运输状态码，0：快递收件(揽件)1.在途中 2.正在派件 3.已签收 4.派送失败 5.疑难件 6.退件签收
        self.delivery_status = delivery_status

        self.is_sign = is_sign  # 是否签收
        self.expName = expName  # 快递公司名称
        self.expSite = expSite  # 快递公司网站
        self.expPhone = expPhone  # 快递公司官方电话(不是快递员电话)
        self.courier = courier  # 快递员姓名
        self.courierPhone = courierPhone  # 快递员电话
        self.updateTime = updateTime  # 快递轨迹最新更新时间
        self.takeTime = takeTime  # 发货到收货耗时
        self.status_ = [
            "status状态码含义",
            "0:正常查询",
            "201:快递单号错误",
            "203:快递公司不存在",
            "204:快递公司识别失败",
            "205:没有信息",
            "207:该单号被限制"]
        self.delivery_status_ = [
            "delivery_status状态码含义",
            "0：快递收件(揽件)",
            "1.在途中",
            "2.正在派件",
            "3.已签收",
            "4.派送失败",
            "5.疑难件",
            "6.退件签收"]
        self.exp_code_ = [
            "常见快递公司代码缩写：",
            "顺丰快递：SFEXPRESS",
            "中通快递：ZTO",
            "申通快递：STO",
            "圆通快递：YTO",
            "韵达快递：YUNDA",
            "京东快递：JD",
            "德邦：DEPPON"]

    def query_(self, number, company=None, sf_phone=None):
        # number是快递单号，必须填写。company是快递公司的缩写，可以不填，但是准确率会受到影响
        # 如果是顺丰，还需要提供 收件人或者寄件人 手机号码后四位
        host = 'https://wuliu.market.alicloudapi.com'
        path = '/kdi'
        appcode = self.appcode  # 开通服务后 买家中心-查看AppCode

        if company == "SFEXPRESS":
            query_info = f'no={number}:{sf_phone}&type={company}'
        elif company:
            query_info = f'no={number}&type={company}'
        else:
            query_info = f'no={number}'

        url = host + path + '?' + query_info  # 准备URL
        header = {"Authorization": 'APPCODE ' + appcode}  # 准备headers
        try:
            res = requests.get(url, headers=header)
        except requests.ConnectionError or requests.HTTPError:
            return None, "URL错误、HTTP错误或者链接错误"
        httpStatusCode = res.status_code

        if httpStatusCode == 200:
            # print("正常请求计费(其他均不计费)")
            return self.parser(httpStatusCode, json.loads(res.text))

        else:
            httpReason = res.headers['X-Ca-Error-Message']
            if httpStatusCode == 400 and httpReason == 'Invalid Param Location':
                return httpStatusCode, "参数错误"
            elif httpStatusCode == 400 and httpReason == 'Invalid AppCode':
                return httpStatusCode, "AppCode错误"
            elif httpStatusCode == 400 and httpReason == 'Invalid Url':
                return httpStatusCode, "请求的 Method、Path 或者环境错误"
            elif httpStatusCode == 403 and httpReason == 'Unauthorized':
                return httpStatusCode, "服务未被授权（或URL和Path不正确）"
            elif httpStatusCode == 403 and httpReason == 'Quota Exhausted':
                return httpStatusCode, "套餐包次数用完"
            elif httpStatusCode == 403 and httpReason == 'Api Market Subscription quota exhausted':
                return httpStatusCode, "套餐包次数用完，请续购套餐"
            elif httpStatusCode == 500:
                return httpStatusCode, "API网关错误"
            else:
                return "参数名错误 或 其他错误"

    def parser(self, httpStatusCode, value):
        if httpStatusCode == 200:
            retval = []
            if self.delivery_status:
                res = value["result"]["deliverystatus"]
                if res == "0":
                    retval.append("快递收件(揽件)")
                elif res == "1":
                    retval.append("在途中")
                elif res == "2":
                    retval.append("正在派件")
                elif res == "3":
                    retval.append("已签收")
                elif res == "4":
                    retval.append("派送失败")
                elif res == "5":
                    retval.append("疑难件")
                elif res == "6":
                    retval.append("退件签收")

            if self.is_sign:
                res = value["result"]["issign"]
                if res == "1":
                    retval.append("已签收")
                else:
                    retval.append("未签收")
            if self.expName:
                res = value["result"]["expName"]
                retval.append(res)
            if self.expSite:
                res = value["result"]["expSite"]
                retval.append(res)
            if self.expPhone:
                res = value["result"]["expPhone"]
                retval.append(res)
            if self.courier:
                res = value["result"]["courier"]
                retval.append(res)
            if self.courierPhone:
                res = value["result"]["courierPhone"]
                retval.append(res)
            if self.updateTime:
                res = value["result"]["updateTime"]
                retval.append(res)
            if self.takeTime:
                res = value["result"]["takeTime"]
                retval.append(res)
            return retval


@xw.func()
def query(number, company=None, sf_phone=None):
    q = Query("----在此处填写你的appkey----")  # 在此处填写你的appkey
    return q.query_(number, company, sf_phone)


@xw.func()
def queryTitle():
    q = Query("")
    retval = []
    if q.delivery_status:
        retval.append("快递运输状态")
    if q.is_sign:
        retval.append("签收状态")
    if q.expName:
        retval.append("快递公司名称")
    if q.expSite:
        retval.append("快递公司官网")
    if q.expPhone:
        retval.append("快递公司电话")
    if q.courier:
        retval.append("快递员姓名")
    if q.courierPhone:
        retval.append("快递员电话")
    if q.updateTime:
        retval.append("信息最后更新时间")
    if q.takeTime:
        retval.append("快递运送耗时")
    return retval

@xw.func()
def hello(name):
    return f"hello {name}"
