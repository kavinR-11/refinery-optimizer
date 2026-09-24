"""
Backend REST API for SIH 26119 Indigenous Solver.
Built with FastAPI. Exposes solve, job polling, telemetry, benchmarks, and refinery planning endpoints.
"""

from .app import app
