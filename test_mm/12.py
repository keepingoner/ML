# -*- coding:utf-8 -*-
import json
import urllib2
import datetime

year = int(datetime.datetime.now().strftime("%Y"))
all_days = [1, 3, 5, 7, 8, 10, 12]
loss_days = [4, 6, 9, 11]

# 构造月份
month = [x for x in range(1, 13)]


def query_month(month):
    """
    根据月份返回天数
    :param month:
    :return: days
    """
    if i in all_days:
        days = [x for x in range(1, 32)]

    elif i in loss_days:
        days = [x for x in range(1, 31)]

    elif i == 2:
        # 判断是平年还是闰年

        if ((year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)):

            days = [x for x in range(1, 30)]

        else:
            days = [x for x in range(1, 29)]

    return days


def is_work(date):
    """
    判断是否是工作日
    :param date:
    :return: True or False
    """

    api_url = "http://api.goseek.cn/Tools/holiday?date=" + date

    vop_url_request = urllib2.Request(api_url)

    vop_response = urllib2.urlopen(vop_url_request)

    vop_data = json.loads(vop_response.read())

    print(vop_data)

    if vop_data["data"] == 0:
        return True
    return False


# 根据月份构造天数
for i in month:
    days = query_month(i)

    # 构造年月日
    if len(str(i)) == 1:
        i = "0" + str(i)
        year_month = str(year) + i
    else:
        year_month = str(year) + str(i)

    # 构造年月日
    for j in days:

        if len(str(j)) == 1:
            j = "0" + str(j)
            year_month_day = year_month + j
        else:
            year_month_day = year_month + str(j)

        yes = is_work(year_month_day)
        print(yes)
