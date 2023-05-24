import time
import os
import logging
from logging import Logger
from typing import Optional
from prettytable import PrettyTable

try:
    # Available in Python >= 3.2
    from contextlib import ContextDecorator
except ImportError:
    import functools

    class ContextDecorator(object):  # type: ignore[no-redef]

        def __enter__(self):
            raise NotImplementedError

        def __exit__(self, exc_type, exc_val, exc_tb):
            raise NotImplementedError

        def __call__(self, func):
            @functools.wraps(func)
            def wrapped(*args, **kwargs):
                with self:
                    return func(*args, **kwargs)

            return wrapped

class Metric(ContextDecorator):
    def __init__(self, name: Optional[str] = None, logger: Optional[Logger] = None, timer_name: Optional[str] = None):
        self.times = {}
        self.counts = {}
        self.starts = {}
        self.sets = {}
        self._name = name
        if name is None and timer_name is None:
            self._name = "pid_{}_time_{}".format(os.getpid(), time.strftime("%Y%m%d%H%M%S", time.localtime()))
        self._logger = logger
        if timer_name is not None:
            self._timer_name = timer_name

    def __enter__(self):
        self.start(self._timer_name)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop(self._timer_name)

    def start(self, name):
        self.starts[name] = time.time()

    def stop(self, name):
        if name not in self.times:
            self.times[name] = []
            self.counts[name] = 0
        self.times[name].append(time.time() - self.starts[name])
        self.counts[name] += 1

    def set(self, name, value):
        if name not in self.sets:
            self.sets[name] = []
            self.counts[name] = 0
        self.sets[name].append(value)
        self.counts[name] += 1
         
    def _get_count(self, name):
        if name in self.counts:
            return self.counts[name]
        else:
            return 0
    
    def _average(self, name):
        if name in self.times and self.counts[name] > 0:
            return sum(self.times[name]) / self.counts[name]
        else:
            return 0

    def _max_time(self, name):
        if name in self.times and self.counts[name] > 0:
            return max(self.times[name])
        else:
            return 0

    def _min_time(self, name):
        if name in self.times and self.counts[name] > 0:
            return min(self.times[name])
        else:
            return 0

    def print_metrics(self, output_file=None):
        if output_file:
            with open(output_file, 'w') as f:
                self._print_metrics(f)
        else:
            self._print_metrics()
        if self._logger:
            self._log_metrics()
    
    def logger_print_metrics(self):
        self._log_metrics()
        
    def _print_metrics(self, output_file=None):
        table = PrettyTable()
        table.field_names = ["Name", "Count", "Average", "Max", "Min", "Category"]
        table.float_format = ".4"
        table.align["Name"] = "l"
        for name in self.times.keys():
            count = self._get_count(name)
            avg = self._average(name)
            max_time = self._max_time(name)
            min_time = self._min_time(name)
            table.add_row([name, count, avg, max_time, min_time, "Time"])
        for name in self.sets.keys():
            count = self._get_count(name)
            avg = sum(self.sets[name]) / count
            max_value = max(self.sets[name])
            min_value = min(self.sets[name])
            table.add_row([name, count, avg, max_value, min_value, "Value"])
        table.title = "Metrics for {}".format(self._name)
        if output_file:
            with open(output_file, 'w') as f:
                f.write(str(table))
        else:
            print(table)

    def _log_metrics(self, log: Optional[Logger] = None):
        if log is None:
            log = self._logger
        header = ["Name", "Count", "Average", "Max", "Min", "Category"]
        table = PrettyTable(header)
        table.float_format = ".4"
        table.align["Name"] = "l"
        for name in self.times.keys():
            count = self._get_count(name)
            avg = self._average(name)
            max_time = self._max_time(name)
            min_time = self._min_time(name)
            table.add_row([name, count, avg, max_time, min_time, "Time"])
        for name in self.sets.keys():
            count = self._get_count(name)
            avg = sum(self.sets[name]) / count
            max_value = max(self.sets[name])
            min_value = min(self.sets[name])
            table.add_row([name, count, avg, max_value, min_value, "Value"])
        table.title = "Metrics for {}".format(self._name)
        log.info("\n" + str(table))