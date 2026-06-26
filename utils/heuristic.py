"""Heuristic policies used by the single-agent baseline and as baselines in reports."""

import numpy as np


def order_up_to_policy(state, firm_id, target_inventory=120, max_order=20):
    """Order-Up-To heuristic: order enough to bring inventory back to ``target_inventory``.

    Reads the inventory dimension from the observation (index 2) and clamps
    the resulting order to ``[0, max_order]``.
    """
    inventory = np.asarray(state).reshape(-1)[2]
    order = max(0, target_inventory - inventory)
    order = min(order, max_order)
    return int(order)


def random_policy(max_order=20):
    """Uniformly sample an order quantity in [0, max_order]."""
    return np.random.randint(0, max_order + 1)