.. module:: scFates
.. automodule:: scFates
   :noindex:


API
===

Import scFates as::

   import scFates as scf

Tree inference
--------------

.. autosummary::
    :toctree: .

    tl.tree
    tl.cleanup
    tl.root
    tl.roots
    
Pseudotime analysis
------------------- 

.. autosummary::
    :toctree: .

    tl.pseudotime
    tl.refine_pseudotime
    tl.test_association
    tl.fit
    tl.cluster
    
Bifurcation analysis
--------------------

**Branch specific feature extraction and classification**

.. autosummary::
    :toctree: .

    tl.test_fork
    tl.branch_specific
    tl.activation
    
**Correlation analysis**

.. autosummary::
    :toctree: .

    tl.slide_cells
    tl.slide_cors
    tl.synchro_path

Plot
----

.. autosummary::
    :toctree: .
    
    pl.tree