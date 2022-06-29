"""Settings
"""
from . import logging
import matplotlib.pyplot as plt
from time import time

verbosity = 3
"""Verbosity level (0=errors, 1=warnings, 2=info, 3=hints)
"""

logfile = ""
"""Name of logfile. By default is set to '' and writes to standard output."""

plot_prefix = "scFates_"
"""Global prefix that is appended to figure filenames.
"""

plot_suffix = ""
"""Global suffix that is appended to figure filenames.
"""


def _set_start_time():
    return time()


_start = _set_start_time()
"""Time when the settings module is first imported."""

_previous_time = _start
"""Variable for timing program parts."""

_previous_memory_usage = -1
"""Stores the previous memory usage."""


def set_figure_pubready():
    plt.rcParams["svg.fonttype"] = "none"
    plt.rcParams["font.family"] = ["sans-serif"]
    plt.rcParams["font.sans-serif"] = ["Arial"]
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["savefig.bbox"] = "tight"
