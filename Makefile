# Top-level project Makefile. Standard targets:
#   make test        fast tier only (the pytest.ini default: -m "not slow"); what CI runs
#   make test-full   the entire suite incl. the slow tier (end-to-end VCF parsing + est-sfs)
#   make docs        rebuild the HTML docs from scratch (clean + html)
#   make clean       remove the built docs
#
# The slow tier (marked `slow` in the tests) is the end-to-end VCF parsing/annotation and
# est-sfs binary comparisons; many of its tests also skip when their large fixtures are
# absent. Tests are meant to run in the `sfsutils-dev` conda env (see envs/dev.yaml).

PYTEST ?= pytest

# Always run tests in parallel via pytest-xdist. Override e.g. `make test XDIST="-n 4"`
# or disable with `make test XDIST=""` for a serial run (useful when debugging).
XDIST ?= -n auto

.PHONY: help test test-full docs clean

help:
	@echo "Targets:"
	@echo "  make test       # fast tier (default: -m 'not slow'); what CI runs"
	@echo "  make test-full  # entire suite incl. the slow tier"
	@echo "  make docs       # rebuild HTML docs from scratch (clean + html)"
	@echo "  make clean      # remove the built docs"

test:
	$(PYTEST) $(XDIST)

test-full:
	$(PYTEST) $(XDIST) -m "slow or not slow"

# --- docs (Sphinx + myst-nb; notebooks are not executed, nb_execution_mode='off') ---
docs:
	$(MAKE) -C docs clean
	$(MAKE) -C docs html
	@echo "Docs built -> docs/_build/html/index.html"

clean:
	$(MAKE) -C docs clean
