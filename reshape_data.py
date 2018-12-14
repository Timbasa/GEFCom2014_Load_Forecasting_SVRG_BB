import numpy as np
import pandas as pd
from datetime import datetime, timedelta


def reshape_data(data, flag):
    baseDates = [datetime(2005, 1, 1, 1, 00), datetime(2010, 10, 12, 1, 00), datetime(2010, 11, 12, 1, 00),
                 datetime(2010, 12, 12, 1, 00), datetime(2011, 1, 12, 1, 00), datetime(2011, 2, 12, 1, 00),
                 datetime(2011, 3, 12, 1, 00), datetime(2011, 4, 12, 1, 00), datetime(2011, 5, 12, 1, 00),
                 datetime(2011, 6, 12, 1, 00), datetime(2011, 7, 12, 1, 00), datetime(2011, 8, 12, 1, 00),
                 datetime(2011, 9, 12, 1, 00), datetime(2011, 10, 12, 1, 00), datetime(2011, 11, 12, 1, 00)]
    baseDate = baseDates[flag]
    trend = np.arange(0, len(data), dtype=np.intc)
    date = list(map(lambda x: baseDate + timedelta(hours=np.asscalar(x)), trend))
    weekday = list(map(lambda x: x.weekday(), date))
    month = list(map(lambda x: x.month, date))
    hour = list(map(lambda  x: x.hour, date))

    dayofweek = np.zeros((len(data), 7))
    for i, w in enumerate(weekday):
        dayofweek[i, w] = 1

    dayofmonth = np.zeros((len(data), 12))
    for i, w in enumerate(month):
        dayofmonth[i, w-1] = 1

    dayofhour = np.zeros((len(data), 24))
    for i, w in enumerate(hour):
        dayofhour[i, w-1] = 1

    reshapedData = np.zeros((data.shape[0], 20))
    for i in range(reshapedData.shape[0]):
        reshapedData[i][0] = data[i][2]
        # reshapedData[i][1:25] = dayofhour[i]
        # reshapedData[i][25:32] = dayofweek[i]
        # reshapedData[i][32:44] = dayofmonth[i]
        reshapedData[i][1:8] = dayofweek[i]
        reshapedData[i][8:20] = dayofmonth[i]

    return reshapedData