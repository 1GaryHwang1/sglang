from __future__ import annotations

import logging
import os
from array import array
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

from sglang.srt.environ import envs
from sglang.srt.mem_cache.base_prefix_cache import (
    BasePrefixCache,
    InsertParams,
    MatchPrefixParams,
    zero_match_result,
)
from sglang.srt.mem_cache.radix_cache import RadixCache, RadixKey

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.managers.scheduler_components.schedule_events import (
        SchedulingEngineView,
    )


IN_BATCH_PREFIX_CACHING_CHECK_THRESHOLD = int(
    os.environ.get("IN_BATCH_PREFIX_CACHING_CHECK_THRESHOLD", "32")
)
IN_BATCH_PREFIX_CACHING_DEPRIORITIZE_THRESHOLD = int(
    os.environ.get("IN_BATCH_PREFIX_CACHING_DEPRIORITIZE_THRESHOLD", "32")
)


logger = logging.getLogger(__name__)


@dataclass
class PolicyData:
    tree_cache: BasePrefixCache
    rank_calls: int = 0
    waiting_queue_radix_tree: RadixCache = field(
        default_factory=RadixCache.create_simulated,
    )


def initialize_policy(engine_view: SchedulingEngineView) -> PolicyData:
    return PolicyData(tree_cache=engine_view.tree_cache)


def _prefill_work(req: Req) -> int:
    return max(
        1,
        len(req.origin_input_ids) + len(req.output_ids) - req.num_matched_prefix_tokens,
    )


def rank(waiting_queue: list[Req], policy_data: PolicyData) -> None:
    """Prefer short uncached prefills while deferring duplicate prefixes."""
    if getattr(policy_data.tree_cache, "disable", True):
        raise ValueError("Experimental shortest-prefill-first requires prefix caching")

    temporary_deprioritized = _compute_prefix_matches(waiting_queue, policy_data)
    waiting_queue.sort(
        key=lambda req: (
            req.rid in temporary_deprioritized,
            _prefill_work(req),
            req.time_stats.wait_queue_entry_time,
        )
    )
    policy_data.rank_calls += 1
    if waiting_queue and policy_data.rank_calls % 128 == 1:
        logger.info(
            "Experimental shortest-prefill-first: queue=%d shortest_uncached=%d",
            len(waiting_queue),
            _prefill_work(waiting_queue[0]),
        )


def _match_prefix_for_req(
    tree_cache: BasePrefixCache,
    req: Req,
    token_ids: array | None = None,
):
    if token_ids is None:
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


def _compute_prefix_matches(
    waiting_queue: list[Req],
    policy_data: PolicyData,
) -> set[str]:
    temporary_deprioritized: set[str] = set()
    queue_tree = policy_data.waiting_queue_radix_tree
    queue_tree.reset()

    for req in waiting_queue:
        prefix_ids = req.origin_input_ids + req.output_ids
        _match_prefix_for_req(policy_data.tree_cache, req, prefix_ids)

        # Defer duplicate cold prefixes so one request can populate the cache.
        if len(req.prefix_indices) <= IN_BATCH_PREFIX_CACHING_CHECK_THRESHOLD:
            match_result = queue_tree.match_prefix(
                MatchPrefixParams(
                    key=RadixKey(
                        token_ids=prefix_ids,
                        extra_key=req.extra_key,
                        cache_salt=req.cache_salt,
                    )
                )
            )
            if envs.SGLANG_RADIX_FORCE_MISS.get():
                match_result = zero_match_result(
                    queue_tree, match_result, extra_key=req.extra_key
                )
            if (
                len(match_result.device_indices)
                >= IN_BATCH_PREFIX_CACHING_DEPRIORITIZE_THRESHOLD
            ):
                temporary_deprioritized.add(req.rid)
            else:
                queue_tree.insert(
                    InsertParams(
                        key=RadixKey(
                            token_ids=prefix_ids,
                            extra_key=req.extra_key,
                            cache_salt=req.cache_salt,
                        ),
                        value=torch.empty(len(prefix_ids), dtype=torch.bool),
                    )
                )
    return temporary_deprioritized
