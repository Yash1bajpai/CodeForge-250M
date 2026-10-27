"""Real input-ID budgets: whole rows, one pass, no padding or skipped rows."""


def input_token_budget(config, explicit=None):
    """A stage's data allocation is authoritative, not the all-stage run target."""
    value = explicit if explicit is not None else config.get('data', {}).get('target_tokens')
    if value is None:
        return None  # Legacy configs may remain step-bounded.
    if type(value) is not int or value <= 0:
        raise ValueError('Real input-token budget must be a positive integer')
    return value


def budgeted_prefix(dataset, indices, consumed, budget):
    """Select the longest fitting ordered prefix; never skip or truncate a row.

    Counts real IDs in x=seq[:-1]: min(valid_len, seq_length-1). A padded
    row's final real ID is still an input even though its next target is ignored.
    Returns (indices, stopped_for_budget). Metadata-only; no token shards read.
    """
    if type(consumed) is not int or consumed < 0:
        raise ValueError('Invalid cumulative real input-token counter')
    if budget is None:
        return indices, False
    if type(budget) is not int or budget <= 0 or consumed > budget:
        raise ValueError('Invalid real input-token budget/counter')
    remaining = budget - consumed
    for end, index in enumerate(indices):
        count = dataset.input_tokens(index)
        if count > remaining:
            return indices[:end], True
        remaining -= count
    return indices, False
