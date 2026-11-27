from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from sglang.srt.environ import envs
from sglang.srt.mem_cache.base_prefix_cache import (
    BasePrefixCache,
    MatchPrefixParams,
    zero_match_result,
)
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.runtime_context import get_disagg
from sglang.srt.utils import get_bool_env_var

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.managers.scheduler_components.schedule_events import (
        SchedulingEngineView,
    )


logger = logging.getLogger(__name__)
_ROUTING_KEY_POLICY_DEBUG_LOG = get_bool_env_var("SGLANG_ROUTING_KEY_POLICY_DEBUG_LOG")


@dataclass
class PolicyData:
    tree_cache: BasePrefixCache
    routing_key_counts: Counter[str] = field(default_factory=Counter)


def initialize_policy(engine_view: SchedulingEngineView) -> PolicyData:
    return PolicyData(tree_cache=engine_view.tree_cache)


def on_sort(engine_view: SchedulingEngineView, policy_data: PolicyData) -> None:
    policy_data.routing_key_counts = Counter(
        req.routing_key for req in engine_view.running_reqs if req.routing_key
    )


def rank(waiting_queue: list[Req], policy_data: PolicyData) -> None:
    """Prefer routing keys that are most common in the current running batch."""
    _match_prefixes_for_load(waiting_queue, policy_data.tree_cache)
    counts = policy_data.routing_key_counts
    if _ROUTING_KEY_POLICY_DEBUG_LOG:
        logger.info(
            "routing_key_counts=%s, waiting_keys_before=%s",
            dict(counts),
            [req.routing_key for req in waiting_queue],
        )
    if not counts:
        return

    def sort_key(req: Req):
        key = req.routing_key
        if key and key in counts:
            return (0, -counts[key], key)
        return (1, 0, key or "")

    waiting_queue.sort(key=sort_key)
    if _ROUTING_KEY_POLICY_DEBUG_LOG:
        logger.info("waiting_keys_after=%s", [req.routing_key for req in waiting_queue])


def _match_prefixes_for_load(
    waiting_queue: list[Req], tree_cache: BasePrefixCache
) -> None:
    if (
        tree_cache.supports_fast_match_prefix()
        and get_disagg().disaggregation_mode != "decode"
    ):
        for req in waiting_queue:
            _match_prefix_for_req(tree_cache, req)


def _match_prefix_for_req(tree_cache: BasePrefixCache, req: Req):
    token_ids = req.origin_input_ids + req.output_ids
    # Recompute the SWA tail because its per-request ring is not content-stable.
    reprefill_tail = tree_cache.swa_reprefill_tail_tokens()
    key_limit = max(0, len(token_ids) - reprefill_tail) if reprefill_tail else None
    match_result = tree_cache.match_prefix(
        MatchPrefixParams(
            key=RadixKey(
                token_ids=token_ids,
                extra_key=req.extra_key,
                limit=key_limit,
                cache_salt=req.cache_salt,
            ),
            cow_mamba=False,
            req=req,
        )
    )
    if envs.SGLANG_RADIX_FORCE_MISS.get():
        match_result = zero_match_result(
            tree_cache, match_result, extra_key=req.extra_key
        )
    (
        req.prefix_indices,
        req.last_node,
        req.last_host_node,
        req.best_match_node,
        req.host_hit_length,
        req.swa_host_hit_length,
        req.mamba_host_hit_length,
    ) = (
        match_result.device_indices,
        match_result.last_device_node,
        match_result.last_host_node,
        match_result.best_match_node,
        match_result.host_hit_length,
        match_result.swa_host_hit_length,
        match_result.mamba_host_hit_length,
    )
    max_len = req._compute_max_prefix_len(len(token_ids))
    req.num_matched_prefix_tokens = min(
        len(req.prefix_indices) + req.host_hit_length, max_len
    )
    req.swa_branching_seqlen = match_result.swa_branching_seqlen
    if match_result.mamba_branching_seqlen is not None:
        req.mamba_branching_seqlen = match_result.mamba_branching_seqlen
    if match_result.cache_protected_len is not None:
        req.kv.cache_protected_len = match_result.cache_protected_len
    return match_result
