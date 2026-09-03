"""Real-data acquisition adapters for Multi-Factor Ranking ML.

Every module here produces the same `production.normalization.Raw*Record`
shapes the CLI's `--raw-root` JSON files already use -- acquisition is
kept as a separate, swappable layer in front of that existing contract,
never a second data model. Nothing here computes a feature, label,
score, or trading decision; it only turns a provider's real response
into a provider-neutral raw record.

Every network-calling function takes an injectable client (a small
Protocol), so unit tests never require real network access -- only a
dedicated, `network`-marked test exercises the real HTTP/yfinance calls.
"""
