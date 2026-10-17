"""Examples for the Multi-Agent Tracking Environment."""

# Run the following command to show demos:
#   python3 -m examples.<name>

# Keep this package initializer intentionally lightweight.  Importing every
# example here made a targeted launcher such as ``examples.facmac.camera``
# depend on all optional RLlib agents (including unrelated A3C APIs).  Python
# loads requested submodules on demand, so eager imports are unnecessary and
# make the training entry points fragile across the supported Ray variants.
