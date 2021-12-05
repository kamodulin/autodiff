=============
API reference
=============

This page gives an overview of all public objects, functions, and methods of our automatic differentiation library.

.. note::
   Our automatic differentiation package's default behavior uses forward mode.

   You can import this as follows:

   .. highlight:: python
   >>> import autodiff as ad
   
   If you would like to use reverse mode, please explicitly import it as:
   
   .. highlight:: python
   >>> import autodiff.reverse as ad

.. toctree::
   :maxdepth: 2
   
   forward/index
   reverse/index