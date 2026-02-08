try:
    from importlib_metadata import version
except:
    from importlib.metadata import version
__version__ = version(__name__)
del version

import matplotlib.pyplot as plt

plt.rcParams["figure.max_open_warning"] = 100

from anndata import AnnData

from . import pp
from . import tl
from . import pl
from . import datasets
from . import get
from . import settings
from .settings import set_figure_pubready
