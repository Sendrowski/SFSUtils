import os

from sfsutils import Settings

# When running under pytest-xdist, each worker is a separate process and xdist already
# parallelizes across cores. Disable sfsutils' own multiprocessing (Settings.parallelize)
# so the annotation/parsing pools don't oversubscribe cores on top of the xdist workers.
# xdist sets PYTEST_XDIST_WORKER (e.g. "gw0") in every worker process.
if os.environ.get("PYTEST_XDIST_WORKER"):
    Settings.parallelize = False
