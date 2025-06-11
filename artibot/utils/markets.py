"""Market-related helpers."""

from __future__ import annotations

import re
from typing import List


def generate_candidates(symbol: str) -> List[str]:
    """Return possible market symbol permutations."""
    parts = [p for p in re.split(r"[/:]", symbol) if p]
    cands: set[str] = set()
    if len(parts) == 2:
        base, quote = parts
        cands.update(
            {
                f"{base}/{quote}",
                f"{base}{quote}",
                f"{base}:{quote}",
                f"{base}/USDT",
                f"{base}USDT",
                f"{base}/{quote}:{quote}",
                f"{base}/{quote}:USDT",
                f"{base}{quote}:{quote}",
                f"{base}USDT:USDT",
            }
        )
    elif len(parts) == 1:
        token = parts[0]
        cands.update(
            {
                f"{token}/USD",
                f"{token}USD",
                f"{token}/USD:USD",
                f"{token}USDT",
                token,
            }
        )
    else:
        cands.add(symbol)
    return list(cands)
