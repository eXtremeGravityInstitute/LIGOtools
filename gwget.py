from gwosc.datasets import event_gps
from gwpy.timeseries import TimeSeries
import sys

data = TimeSeries.get(sys.argv[1], sys.argv[2], sys.argv[3])

data.write('data.txt')


