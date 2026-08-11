"""Data platform: FastF1 ingest, schema contracts, cleaning, and the Parquet store.

Downstream modules read Parquet from ``store``. Nothing outside this package should
import ``fastf1`` -- if the FastF1 API changes, only ``load.py`` should need to.
"""
