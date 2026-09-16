"""Fixed shared three-dimensional channels for contextual relation routing.

These triples are implementation channels, not token-pair or fact IDs.  A
contextual hidden state chooses among the same bank at every position; no
sentence, word, or word pair allocates a new matrix.
"""

from typing import List, Optional, Tuple


Triple = Tuple[int, int, int]
DEFAULT_RELATION_CANDIDATE_COUNT = 1500
MAX_RELATION_CANDIDATE_COUNT = 1500


def build_context_space_triples(
    hidden_size: int,
    candidate_count: Optional[int] = None,
) -> List[Triple]:
    """Build a fixed, reusable bank of coordinate triples.

    The first ``ceil(hidden_size / 3)`` triples cover every hidden coordinate
    without dropping the final remainder.  ``candidate_count`` may add more
    distinct, deterministic triples to increase shared relation capacity.  The
    extra candidates are still implementation channels: they are not keyed by
    words, token pairs, facts, or sentences.

    With hidden size 256 and ``candidate_count=1500`` this returns 1500
    reusable candidates.  The first 86 provide complete coordinate coverage;
    the remaining candidates provide additional learned route/Q/K capacity.
    """
    if hidden_size < 3:
        raise ValueError("hidden_size must be at least 3")
    minimum_count = (hidden_size + 2) // 3
    count = minimum_count if candidate_count is None else int(candidate_count)
    if count < minimum_count:
        raise ValueError(
            "candidate_count must be at least the number of triples needed "
            "to cover every hidden coordinate"
        )
    if count > MAX_RELATION_CANDIDATE_COUNT:
        raise ValueError(
            f"candidate_count={count} exceeds the project cap of "
            f"{MAX_RELATION_CANDIDATE_COUNT}"
        )
    capacity = hidden_size * (hidden_size - 1) * (hidden_size - 2) // 6
    if count > capacity:
        raise ValueError(
            f"candidate_count={count} exceeds the {capacity} unique triples "
            f"available for hidden_size={hidden_size}"
        )

    triples = [
        tuple(
            sorted(
                (
                    (3 * index) % hidden_size,
                    (3 * index + 1) % hidden_size,
                    (3 * index + 2) % hidden_size,
                )
            )
        )
        for index in range(count)
    ]

    # ``count`` initially means the complete-coverage minimum.  Add the
    # remaining candidates in deterministic lexicographic order.  The scan is
    # bounded by the requested bank size, not by a materialized C(n, 3) list.
    triples = triples[:minimum_count]
    seen = set(triples)
    if len(triples) < count:
        for first in range(hidden_size - 2):
            for second in range(first + 1, hidden_size - 1):
                for third in range(second + 1, hidden_size):
                    triple = (first, second, third)
                    if triple in seen:
                        continue
                    triples.append(triple)
                    seen.add(triple)
                    if len(triples) == count:
                        break
                if len(triples) == count:
                    break
            if len(triples) == count:
                break

    if len(set(triples)) != len(triples):
        raise ValueError("context space construction produced duplicate triples")
    covered = {dimension for triple in triples for dimension in triple}
    if covered != set(range(hidden_size)):
        raise ValueError("context space bank does not cover every hidden dimension")
    return triples  # type: ignore[return-value]
