"""Helpers for fetching and building KGLite graphs from the Sodir
(Norwegian Offshore Directorate) FactMaps REST API at
https://factmaps.sodir.no/api/rest/services/DataService.

KGLite is an independent project — not affiliated with Sodir or the
Norwegian Offshore Directorate. The dataset structure and licensing
are defined by upstream; this module only handles the cache + build
lifecycle on the client side.

Public API:
    open(workdir, ...)       -> KnowledgeGraph    # full lifecycle
    fetch_all(workdir, ...)  -> dict              # CSVs only

Layout managed under ``workdir``:

    workdir/
        sodir_index.json          # fetch manifest (per-dataset row count, timestamps)
        csv/                      # cached CSVs, flat
            field.csv
            wellbore.csv
            ...
        graph/                    # disk graph dir built from the CSVs
            sodir_source.json     # build-time snapshot
            disk_graph_meta.json
            ...
"""

from .wrapper import fetch_all, open, remove_complement

__all__ = ["open", "fetch_all", "remove_complement"]
