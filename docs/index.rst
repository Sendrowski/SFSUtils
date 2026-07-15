.. _introduction:

Introduction
============
``sfsutils`` is a Python toolkit for deriving site-frequency spectra (SFS) from raw variant data. It provides a configurable VCF-to-SFS parser together with the annotations and filters commonly required to prepare population-genetic data for downstream analysis. The package is object-oriented, thoroughly documented, and designed so that spectra obtained from different datasets are directly comparable.

Motivation
----------
The SFS condenses population genetic variation by quantifying the number of alleles observed at each frequency, and is the input to a wide range of population-genetic methods. Obtaining a spectrum from real data is rarely a single step: sites must be polarised against an ancestral state, stratified into meaningful categories (for example by degeneracy or synonymy), and filtered to remove sites that violate modelling assumptions. Because comparisons across species, populations, or genomic regions are only meaningful when the spectra are derived in a consistent manner, ``sfsutils`` collects these operations behind a single, reproducible parsing interface.

How it works
------------
``sfsutils`` reads variants directly from VCF files and counts derived-allele frequencies into one or more spectra. Rather than depending on pre-annotated input, it can derive the required site-level information itself. Ancestral alleles may be taken from an existing ``AA`` info tag, or inferred from one or more outgroups either by maximum parsimony or under a maximum-likelihood substitution model; site degeneracy and synonymy are computed directly from FASTA and GFF references. These annotations in turn drive on-the-fly stratification, for instance to contrast putatively neutral and selected sites. A collection of filters excludes sites that violate downstream modelling assumptions, including non-biallelic and non-SNV sites, sites outside coding sequences, CpG sites, and sites affected by GC-biased gene conversion or by deviant outgroups. The resulting spectra are represented by the :class:`~sfsutils.spectrum.Spectrum` and :class:`~sfsutils.spectrum.Spectra` classes, which support folding, polarising, resampling, and visualisation.

Features
--------

``sfsutils`` offers a range of features for preparing site-frequency spectra.

**Parsing**: streamlining the extraction of spectra from raw variant data:

- Built-in VCF-to-SFS parser, with support for versatile stratification
- Stratification by degeneracy, synonymy, base transition or transversion type, ancestral base, genomic context, contig, and more
- Utilities to determine the number of mutational target sites when monomorphic sites are not present in the provided VCF file
- Serialization of spectra and parser configurations
- Object-oriented and customizable design

**Annotation**: adding the site-level information required to build meaningful spectra:

- Site-degeneracy annotation from FASTA and GFF references
- Ancestral-allele annotation, either by maximum parsimony or by a maximum-likelihood model using one or more outgroups
- Synonymy annotation of coding variants

**Filtration**: removing sites that violate downstream modelling assumptions:

- Filtering of non-biallelic and non-SNV sites
- Restriction to coding sequences
- Removal of sites affected by GC-biased gene conversion or by deviant outgroups

**Visualization**: plotting utilities for inspecting spectra:

- Visualization of individual and stratified spectra

Contents
--------

.. toctree::
   :caption: Python Reference

   reference/Python/installation
   reference/Python/parser
   reference/Python/spectra

.. toctree::
   :caption: API Reference
   :maxdepth: 1

   modules/spectrum
   modules/spectra
   modules/parser
   modules/annotation
   modules/filtration

.. toctree::
   :caption: Miscellaneous
   :maxdepth: 1

   modules/changelog

References
----------
.. bibliography::
   :style: plain
