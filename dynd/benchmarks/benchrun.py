"""
Benchrun is a Python script for defining and running performance benchmarks.
It allows you to run a benchmark for different versions of the code and for
different values of an input parameter, and automatically generates tables
that compare the results. 

A benchmark is defined by creating a subclass of Benchmark.
The subclass should define a method run() that executes the code
to be timed and returns the elapsed time in seconds (as a float),
or None if the benchmark should be skipped.

This file was originally taken from https://code.google.com/p/benchrun/ under the MIT License,
but has been modified since.
"""

from __future__ import print_function
import math

import sys
if sys.platform=='win32':
    from time import clock
else:
    from time import time as clock

# http://aspn.activestate.com/ASPN/Cookbook/Python/Recipe/302478
def combinations(*seqin):
    def rloop(seqin,comb):
        if seqin:
            for item in seqin[0]:
                newcomb = comb + [item]
                for item in rloop(seqin[1:],newcomb):   
                    yield item
        else:
            yield comb
    return rloop(seqin,[])

def _mean(n = 10):
    def wrap(func):
        def wrapper(*args, **kwds):
            results = [func(*args, **kwds) for i in range(n)]
            return math.fsum(results) / n
        return wrapper
    return wrap

def mean(callable_or_value):
    if callable(callable_or_value):
        return _mean()(callable_or_value)

    return _mean(callable_or_value)

def _median(n = 10):
    def wrap(func):
        def wrapper(*args, **kwds):
            results = sorted(func(*args, **kwds) for i in range(n))
            i = n // 2
            if n % 2 == 1:
                return results[i]
            return (results[i - 1] + results[i]) / 2.0    
        return wrapper

    return wrap

def median(callable_or_value):
    if callable(callable_or_value):
        return _median()(callable_or_value)

    return _median(callable_or_value)

class Benchmark:
    sort_by = []
    reference = None

    def __init__(self):
        self.pnames = []
        self.pvalues = []
        self.results = []
        self.results_dict = {}
        for pname in self.parameters:
            value = getattr(self, pname)
            self.pnames.append(pname)
            self.pvalues.append(value)
        self.pcombos = list(combinations(*self.pvalues))
        if self.reference:
            self.reference_param = self.reference[0]
            self.reference_value = self.reference[1]

    def time_all(self):
        """Run benchmark for all versions and parameters."""
        for params in self.pcombos:
            args = dict(zip(self.pnames, params))
            t = self.run(**args)
            self.results.append(tuple(params) + (t,))
            self.results_dict[tuple(params)] = t

    def sort_results(self):
        sort_keys = []
        for name in self.sort_by:
            sort_keys += [self.pnames.index(name)]
        for i, name in enumerate(self.pnames):
            if i not in sort_keys:
                sort_keys += [i]
        def key(v):
            return list(v[i] for i in sort_keys)
        self.results.sort(key=key)

    def get_factor(self, pvalues, time):
        if not self.reference or not time:
            return None
        pvalues = list(pvalues)
        i = self.pnames.index(self.reference_param)
        if pvalues[i] == self.reference_value:
            return None
        else:
            pvalues[i] = self.reference_value
        ref = self.results_dict[tuple(pvalues)]
        if ref == None:
            return None
        return ref / time

    def print_result(self):
        """Run benchmark for all versions and parameters and print results
        in tabular form to the standard output."""
        self.time_all()
        self.sort_results()

        print("=" * 78)
        print()
        print(self.__class__.__name__)
        print(self.__doc__, "\n")

        colwidth = 15
        reftimes = {}

        ts = "seconds"
        if self.reference:
            ts += " (x faster than " + (str(self.reference_value)) + ")"
        print("  ", "   ".join([str(r).ljust(colwidth) for r in self.pnames + [ts]]))
        print("-"*79)

        rows = []
        for vals in self.results:
            pvalues =  vals[:-1]
            time = vals[-1]
            if time == None:
                stime = "(n/a)"
            else:
                stime = "%.8f" % time
                factor = self.get_factor(pvalues, time)
                if factor != None:
                    stime += ("  (%.2f)" % factor)
            vals = pvalues + (stime,)
            row = [str(val).ljust(colwidth) for val in vals]
            print("  ", "   ".join(row))
        print()

    def plot_result(self, loglog = False):
        import matplotlib
        import matplotlib.pyplot

        self.time_all()
        self.sort_results()

        if loglog:
            from matplotlib.pyplot import loglog as plot
        else:
            from matplotlib.pyplot import plot

        plot(*zip(*self.results), label = self.__class__.__name__, marker = "o", linestyle = '--', linewidth = 2)
        matplotlib.pyplot.xlabel(self.pnames[0])
        matplotlib.pyplot.ylabel("seconds")

        matplotlib.pyplot.legend(loc = 2, markerscale = 0)
