from gwosc.datasets import event_gps
from gwpy.timeseries import TimeSeries
import sys

data = TimeSeries.fetch_open_data(sys.argv[1], sys.argv[2], sys.argv[3])

data.write('data.txt')


