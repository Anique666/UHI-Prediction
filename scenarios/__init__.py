"""UHI mitigation scenarios.

This package is intentionally independent of the frozen model/evaluator: a
scenario transformer only edits an input tensor, and the frozen model + existing
evaluation code consume it unchanged. Future mitigation modules can reuse this
infrastructure.
"""