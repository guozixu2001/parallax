"""
MLX-LM backend implementation of high level executor
"""

import time
from typing import Any, Dict, List, Optional, Tuple

import mlx.core as mx

from parallax.server.cache_manager import CacheManager
from parallax.server.cache.speculative_cache_manager import SpeculativeCacheManager
from parallax.server.executor.base_executor import BaseExecutor
from parallax.server.request import (
    InitialRequest,
    IntermediateRequest,
    Request,
    RequestStatus,
)
from parallax.server.sampling.sampler import SamplingBatchInfo
from parallax.server.shard_loader import MLXModelLoader
from parallax.utils.utils import (
    combine_padding_and_causal_masks,
    create_causal_mask,
    get_device_dtype,
    get_layer_types,
    pad_inputs,
)
from parallax_utils.logging_config import get_logger

logger = get_logger(__name__)


class MLXExecutor(BaseExecutor):
    def __init__(
        self,
        # Model Configs
        model_repo: str,
        start_layer: int,
        end_layer: int,
        dtype: str = "float16",
        # Device override
        device: Optional[str] = None,
        use_hfcache: bool = False,
        # Scheduler Configs
        max_batch_size: Optional[int] = 8,
        max_sequence_length: Optional[int] = None,
        max_tokens_in_kv_pool: Optional[int] = None,
        # Controlling perfill / decode ratio
        max_num_tokens_per_batch: int = 1024,
        prefill_priority: int = 0,
        micro_batch_ratio: int = 2,
        scheduler_wait_ms: int = 500,
        request_timeout_s: Optional[int] = 600,
        # Metrics Configs
        layer_latency_update_every: int = 4096,
        # KV Cache Configs
        kv_block_size: int = 64,
        kv_cache_memory_fraction: float = 0.8,
        enable_prefix_cache: Optional[bool] = False,
        # Communication Configs
        # P2P Communication Configs
        send_to_peer_addr: Optional[str] = None,
        recv_from_peer_addr: Optional[str] = None,
        # IPC Communication Configs
        executor_input_ipc_addr: Optional[str] = None,
        executor_output_ipc_addr: Optional[str] = None,
        # GPU Specialized Configs
        attention_backend: Optional[str] = "flashinfer",
        moe_runner_backend: Optional[str] = "auto",
        enable_lora: Optional[bool] = False,
        max_lora_rank: Optional[int] = None,
        lora_target_modules: Optional[List[str]] = None,
        lora_paths: Optional[List[str]] = None,
        max_loras_per_batch: Optional[int] = None,
        max_loaded_loras: Optional[int] = None,
        lora_eviction_policy: Optional[str] = "lru",
        lora_backend: Optional[str] = "triton",
        max_lora_chunk_size: Optional[int] = 128,
        # Tensor Parallel Configs
        tp_rank: Optional[int] = 0,
        tp_size: Optional[int] = 1,
        nccl_port: Optional[int] = 4000,
        # Data Parallel Configs (not used in MLX, but accepted for compatibility)
        enable_dp_attention: Optional[bool] = False,
        dp_rank: Optional[int] = 0,
        dp_size: Optional[int] = 1,
        # Optional shared state for layer reallocation detection (when running in subprocess)
        shared_state: Optional[dict] = None,
        # Weight Refit
        enable_weight_refit: Optional[bool] = False,
        # Pipe communication
        conn: Optional[Any] = None,
        # Speculative Decoding Configs
        draft_model: Optional[str] = None,
        draft_start_layer: Optional[int] = 0,
        draft_end_layer: Optional[int] = None,
        num_draft_tokens: Optional[int] = 3,
    ):
        logger.debug(
            f"Initializing MLX sharded model loader for repo={model_repo}, layers=[{start_layer}, {end_layer})"
        )
        self.shard_loader = MLXModelLoader(
            model_repo,
            start_layer=start_layer,
            end_layer=end_layer,
            use_hfcache=use_hfcache,
        )
        t0 = time.time()
        self.model_shard, self.config, self.tokenizer = self.shard_loader.load()

        adapters = lora_paths[0] if lora_paths else None
        if adapters:
            logger.debug(f"mlx adapters is: {adapters}")
            self.model_shard = self.shard_loader.load_lora(self.model_shard, adapters)

        logger.debug(
            f"MLX sharded model loaded in {(time.time() - t0) * 1000:.1f} ms; num_layers={self.config.get('num_hidden_layers')}"
        )

        # TODO: Duplicate code to BaseExecutor since num_shard_layers and dtype are needed for initializing kv cache
        self.num_shard_layers = end_layer - start_layer
        self.dtype = get_device_dtype(dtype, device)
        logger.debug(
            f"Executor dtype set to {dtype} (resolved={self.dtype}); shard_layers={self.num_shard_layers}"
        )

        # Calculate feature dimensions for kv cache
        num_key_value_heads = self.config.get("num_key_value_heads")
        head_dim = self.config.get("head_dim") or self.config.get("hidden_size") // self.config.get(
            "num_attention_heads"
        )
        qk_nope_head_dim = self.config.get("qk_nope_head_dim", None)
        qk_rope_head_dim = self.config.get("qk_rope_head_dim", None)
        if qk_nope_head_dim is not None and qk_rope_head_dim is not None:
            logger.debug(
                f"qk_nope_head_dim={qk_nope_head_dim}, qk_rope_head_dim={qk_rope_head_dim}"
            )
            head_dim = qk_nope_head_dim + qk_rope_head_dim

        v_head_dim = self.config.get("v_head_dim", None)
        linear_key_head_dim = self.config.get("linear_key_head_dim", None)
        linear_value_head_dim = self.config.get("linear_value_head_dim", None)
        linear_conv_kernel_dim = self.config.get("linear_conv_kernel_dim", None)
        linear_num_key_heads = self.config.get("linear_num_key_heads", None)
        linear_num_value_heads = self.config.get("linear_num_value_heads", None)
        key_dim, value_dim, conv_dim = None, None, None
        if linear_key_head_dim is not None and linear_num_key_heads is not None:
            key_dim = linear_key_head_dim * linear_num_key_heads
        if linear_value_head_dim is not None and linear_num_value_heads is not None:
            value_dim = linear_value_head_dim * linear_num_value_heads
        if key_dim is not None and value_dim is not None:
            conv_dim = key_dim * 2 + value_dim

        index_head_dim = self.config.get("index_head_dim", None)
        index_n_heads = self.config.get("index_n_heads", None)

        layer_types = get_layer_types(self.config, start_layer, end_layer)
        logger.debug(f"layer_types: {layer_types}")
        time.sleep(5)

        sliding_window = self.config.get("sliding_window", None)
        use_sliding_window = self.config.get("use_sliding_window", None)
        if use_sliding_window is False:
            sliding_window = None

        # Validate and adjust block size for Metal backend
        supported_block_sizes = [8, 16, 32, 64]
        if kv_block_size not in supported_block_sizes:
            nearest_block_size = min(supported_block_sizes, key=lambda x: abs(x - kv_block_size))
            logger.warning(
                f"Block size {kv_block_size} is not supported for MLX Metal backend. "
                f"Supported block sizes are {supported_block_sizes}. "
                f"Automatically adjusting to supported block size: {nearest_block_size}"
            )
            kv_block_size = nearest_block_size

        logger.debug(
            "Initializing CacheManager (mlx) with block_size=%d, layers=%d",
            kv_block_size,
            self.num_shard_layers,
        )

        # === Speculative Decoding Setup ===
        # TODO(guozixu): temp recalculate is_first_peer
        # Calculate is_first_peer and is_last_peer early (needed for draft model loading)
        # These will be set again in super().__init__(), but we need them now
        self.is_first_peer = start_layer == 0
        self.is_last_peer = end_layer == self.config.get("num_hidden_layers")

        self.enable_speculative = draft_model is not None
        self.num_draft_tokens = num_draft_tokens
        self.draft_model = draft_model
        self.draft_start_layer = draft_start_layer
        self.draft_end_layer = draft_end_layer
        self.draft_model_shard = None
        self.draft_cache_manager = None
        self.draft_generator = None

        if self.enable_speculative:
            logger.info("Speculative decoding enabled")

            # Use SpeculativeCacheManager (ContinuousKVCache) for target model
            self.cache_manager = SpeculativeCacheManager(
                num_layers=self.num_shard_layers,
                max_seq_len=max_sequence_length or 2048,
                num_kv_heads=num_key_value_heads,
                head_dim=head_dim,
                dtype=self.dtype,
            )

            # Set target model to use pure SDPA (no paged attention)
            for layer in self.model_shard.layers:
                if hasattr(layer, 'self_attn'):
                    layer.self_attn.use_paged_attention = False

            # Load draft model (only on first peer)
            if self.is_first_peer and draft_model is not None:
                logger.info(f"Loading draft model from {draft_model}")

                # Load full draft model (all layers)
                self.draft_shard_loader = MLXModelLoader(
                    draft_model,
                    start_layer=draft_start_layer,
                    end_layer=draft_end_layer,  # Load all layers
                    use_hfcache=use_hfcache,
                )
                self.draft_model_shard, draft_config, _ = self.draft_shard_loader.load()

                # Set draft model to use SDPA
                for layer in self.draft_model_shard.layers:
                    if hasattr(layer, 'self_attn'):
                        layer.self_attn.use_paged_attention = False

                # Create draft cache manager
                draft_num_layers = draft_config.get('num_hidden_layers')
                draft_num_kv_heads = draft_config.get('num_key_value_heads')
                draft_head_dim = draft_config.get('head_dim') or (
                    draft_config.get('hidden_size') // draft_config.get('num_attention_heads')
                )

                self.draft_cache_manager = SpeculativeCacheManager(
                    num_layers=draft_num_layers,
                    max_seq_len=max_sequence_length or 2048,
                    num_kv_heads=draft_num_kv_heads,
                    head_dim=draft_head_dim,
                    dtype=self.dtype,
                )

                # Create draft generator
                from parallax.server.speculative.draft_generator import DraftGenerator

                self.draft_generator = DraftGenerator(
                    draft_model=self.draft_model_shard,
                    cache_manager=self.draft_cache_manager,
                    max_draft_tokens=self.num_draft_tokens,
                )

        else:
            # Use standard PagedCacheManager
            self.cache_manager = CacheManager(
                num_layers=self.num_shard_layers,
                num_kv_heads=num_key_value_heads,
                head_dim=head_dim,
                dtype=self.dtype,
                block_size=kv_block_size,
                cache_memory_fraction=kv_cache_memory_fraction,
                head_dim_v=v_head_dim,
                index_head_dim=index_head_dim,
                index_n_heads=index_n_heads,
                layer_types=layer_types,
                max_num_seqs=max_batch_size // micro_batch_ratio,
                conv_dim=conv_dim,
                conv_kernel_size=linear_conv_kernel_dim,
                linear_k_dim=linear_key_head_dim,
                linear_v_dim=linear_value_head_dim,
                linear_num_k_heads=linear_num_key_heads,
                linear_num_v_heads=linear_num_value_heads,
                enable_prefix_cache=enable_prefix_cache,
                sliding_window=sliding_window,
            )
        super().__init__(
            start_layer=start_layer,
            end_layer=end_layer,
            dtype=dtype,
            device=device,
            max_batch_size=max_batch_size,
            max_sequence_length=max_sequence_length,
            max_num_tokens_per_batch=max_num_tokens_per_batch,
            prefill_priority=prefill_priority,
            micro_batch_ratio=micro_batch_ratio,
            scheduler_wait_ms=scheduler_wait_ms,
            request_timeout_s=request_timeout_s,
            layer_latency_update_every=layer_latency_update_every,
            send_to_peer_addr=send_to_peer_addr,
            recv_from_peer_addr=recv_from_peer_addr,
            executor_input_ipc_addr=executor_input_ipc_addr,
            executor_output_ipc_addr=executor_output_ipc_addr,
            tp_rank=tp_rank,
            tp_size=tp_size,
            shared_state=shared_state,
            enable_weight_refit=enable_weight_refit,
            conn=conn,
        )

        try:
            mx.set_wired_limit(mx.metal.device_info()["max_recommended_working_set_size"])
        except Exception:
            logger.warning(f"Using mlx without metal backend.")

        # Prefix Cache Manager
        self.enable_prefix_cache = enable_prefix_cache
        # self.prefix_cache = RadixCache(
        #     num_kv_heads=num_key_value_heads,
        #     head_dim=head_dim,
        #     head_dim_v=v_head_dim,
        #     num_layers=self.num_shard_layers,
        #     dtype=self.dtype,
        #     page_size=1,
        # )

        logger.debug(
            f"CacheManager ready; wired_limit set; prefix_cache={'on' if self.enable_prefix_cache else 'off'}"
        )

    def handle_input_requests(self, requests: List[Request]):
        """Update requests states and status in scheduler and cache manager."""
        if not requests:
            return
        if self.is_first_peer:
            # First peer can receive InitialRequests from the client RPC,
            # or IntermediateRequests from the last peer.
            for req in requests:
                if isinstance(req, InitialRequest):
                    self.scheduler.enque_request(req)
                elif isinstance(req, IntermediateRequest):
                    original_req = self.scheduler.get_running_request(req.request_id)
                    if original_req is None:
                        logger.warning(
                            f"Received IntermediateRequest {req.request_id}. "
                            "But no corresponding request found in scheduler. "
                            "It might have been cancelled or finished."
                        )
                        continue
                    if not self.cache_manager.has_request(req.request_id):
                        logger.warning(
                            f"Received IntermediateRequest {req.request_id}. "
                            "But no corresponding request found in cache manager. "
                            "It might have been cancelled or finished."
                        )
                        continue

                    # === Speculative Decoding: Handle Verification Results ===
                    if original_req.is_speculative and req.num_accepted_tokens > 0:
                        # TODO(guozixu): change to read num_accepted_tokens and next_token_id direct from req
                        # Parse verification results: [num_accepted, bonus_token]
                        # These are encoded in req.hidden_states as shape (1, 2) or (2,)
                        if req.hidden_states.ndim == 2:
                            # Shape is (1, 2)
                            num_accepted = int(req.hidden_states[0, 0])
                            bonus_token = int(req.hidden_states[0, 1])
                        else:
                            # Shape is (2,) - flattened
                            num_accepted = int(req.hidden_states[0])
                            bonus_token = int(req.hidden_states[1])

                        logger.debug(
                            f"Speculative result for {req.request_id}: "
                            f"accepted={num_accepted}, bonus={bonus_token}"
                        )

                        # Calculate original length BEFORE modifying output_ids
                        # This is the cache length before accepting any draft tokens
                        original_length = original_req.total_length
                        num_draft_tokens = len(original_req.draft_token_ids) if original_req.draft_token_ids else 0

                        # Accept draft tokens
                        accepted_tokens = []
                        if original_req.draft_token_ids:
                            accepted_tokens = original_req.draft_token_ids[:num_accepted]
                            original_req.output_ids.extend(accepted_tokens)
                            logger.debug(
                                f"Accepted {len(accepted_tokens)} draft tokens: {accepted_tokens}"
                            )

                        # Add bonus token
                        original_req.output_ids.append(bonus_token)
                        logger.debug(f"Added bonus token: {bonus_token}")

                        # Rollback KV cache (remove rejected draft tokens)
                        if isinstance(self.cache_manager, SpeculativeCacheManager):
                            # Cache currently contains: [prompt, old_tokens, ALL_draft_tokens]
                            # We want: [prompt, old_tokens, accepted_draft_tokens]
                            # Note: bonus_token is NOT in cache yet, it will be added in next iteration
                            # Target length = original_length + num_accepted
                            new_length = original_length + num_accepted

                            # Rollback target cache to new_length (removes rejected tokens)
                            self.cache_manager.rollback_to_position(
                                original_req.request_id, new_length
                            )
                            logger.debug(
                                f"Rolled back target cache for {req.request_id} "
                                f"from {original_length + num_draft_tokens} to {new_length}"
                            )

                            # Rollback draft cache: matches mlx-lm's trim formula
                            # trim_amount = max(num_draft - num_accept - 1, 0)
                            # draft_new_length = original_length + num_draft - trim_amount
                            #                  = original_length + num_draft - max(num_draft - num_accept - 1, 0)
                            #                  = original_length + min(num_accept + 1, num_draft)
                            # Simplified: original_length + num_accepted (when num_accepted < num_draft)
                            #             original_length + num_draft - 1 (when all accepted)
                            if isinstance(self.draft_cache_manager, SpeculativeCacheManager):
                                if num_accepted == num_draft_tokens:
                                    # All tokens accepted: keep all but the last one
                                    draft_new_length = original_length + num_draft_tokens - 1
                                else:
                                    # Some rejected: keep only accepted tokens
                                    draft_new_length = original_length + num_accepted

                                self.draft_cache_manager.rollback_to_position(
                                    original_req.request_id, draft_new_length
                                )
                                logger.debug(
                                    f"Rolled back draft cache for {req.request_id} to {draft_new_length}"
                                )

                        # Clear draft tokens
                        original_req.draft_token_ids = None
                        original_req.num_draft_tokens = 0

                        # Update status back to SPECULATIVE
                        original_req.update_status(RequestStatus.SPECULATIVE)
                        logger.debug(f"Request {req.request_id} status changed to SPECULATIVE")

                        # Send all generated tokens to HTTP server
                        if self.tp_rank == 0:
                            # Send accepted draft tokens + bonus token
                            tokens_to_send = accepted_tokens + [bonus_token]

                            for token_id in tokens_to_send:
                                req_dict = {
                                    "prompt_tokens": len(req.input_ids),
                                    "next_token_id": token_id,
                                    "rid": req.request_id,
                                }
                                if original_req.status == RequestStatus.FINISHED_EOS:
                                    req_dict["eos"] = True
                                if original_req.status == RequestStatus.FINISHED_MAX_LENGTH:
                                    req_dict["length"] = True
                                if original_req.status == RequestStatus.FINISHED_ABORT:
                                    req_dict["abort"] = True

                                if self.enable_weight_refit:
                                    req_dict["weight_version"] = self.weight_version
                                if hasattr(self, "send_to_ipc_socket"):
                                    self.send_to_ipc_socket.send_pyobj(req_dict)

                            logger.debug(
                                f"Sent {len(tokens_to_send)} tokens to HTTP server for {req.request_id}"
                            )

                    elif not req.abort and req.next_token_id is not None:
                        # Normal decoding path
                        # Pass enable_speculative flag for proper status transition
                        original_req.commit_new_token(
                            req.next_token_id,
                            enable_speculative=self.enable_speculative
                        )

                    if len(req.routing_table) > 0:
                        original_req.routing_table = req.routing_table

                    # Check for termination.
                    if req.abort:
                        original_req.abort = True

                    if self.scheduler.check_and_update_request_status(original_req):
                        self.cache_manager.release_request(original_req.request_id)
                        logger.debug(
                            f"Released resources for finished request {req.request_id}, "
                            f"memory usage: {mx.get_active_memory() / 1024**3 :.3f} GB"
                        )
                        if not self.is_last_peer and not req.abort:
                            self.finished_batch.append(req)
                    else:
                        self.scheduler.enque_request(original_req)

                    # detokenize and send to http server
                    # Skip if we already sent tokens in speculative handling above
                    if self.tp_rank == 0 and not (original_req.is_speculative and req.num_accepted_tokens > 0):
                        # Only send token if it's valid
                        token_to_send = req.next_token_id if req.next_token_id is not None else -1
                        req_dict = {
                            "prompt_tokens": len(req.input_ids),
                            "next_token_id": token_to_send,
                            "rid": req.request_id,
                        }
                        if original_req.status == RequestStatus.FINISHED_EOS:
                            req_dict["eos"] = True
                        if original_req.status == RequestStatus.FINISHED_MAX_LENGTH:
                            req_dict["length"] = True
                        if original_req.status == RequestStatus.FINISHED_ABORT:
                            req_dict["abort"] = True

                        # Add prob value for the sampled token (if requested and available)
                        if original_req.return_probs and req.token_prob is not None:
                            req_dict["probs"] = req.token_prob
                        if self.enable_weight_refit:
                            req_dict["weight_version"] = self.weight_version
                        if hasattr(self, "send_to_ipc_socket"):
                            self.send_to_ipc_socket.send_pyobj(req_dict)
                else:
                    raise TypeError(f"First peer received unexpected request type: {type(req)}")

        else:
            # Intermediate and Last peers receive IntermediateRequests from the previous peer.
            for req in requests:
                assert isinstance(
                    req, IntermediateRequest
                ), "Non-first peers must receive IntermediateRequests."
                if req.is_finished or req.hidden_states is None:
                    if self.enable_prefix_cache:
                        keys, values = self.cache_manager.gather_kv_cache(req.request_id)
                        self.prefix_cache.cache_finished_request(req, keys, values)
                        self.prefix_cache.evict_request(req.request_id)

                    self.cache_manager.release_request(req.request_id)
                    logger.debug(
                        f"Released resources for finished request {req.request_id}, "
                        f"memory usage: {mx.get_active_memory() / 1024**3 :.3f} GB"
                    )
                    self.scheduler.evict_request(req.request_id)
                    if not self.is_last_peer and not req.abort:
                        self.finished_batch.append(req)
                else:
                    # This is an active request, add it to the scheduler queue to be processed.
                    self.scheduler.enque_request(req)

    def process_batch(self, prepared_inputs: Dict[str, Any], return_decoded_tokens: bool = True):
        """Process a batch of requests in MLX."""
        # Run model and get updated cache
        # Note: Paged Attention writes KV cache in-place within the model (via reshape_and_cache).
        # The returned 'hidden_states' is what we need.
        # The returned cache tuple (_, _) is ignored/unused here.
        logger.debug(f"prefix_cache is {'on' if self.enable_prefix_cache else 'off'}")
        logger.debug(f"prefix_lens: {prepared_inputs.get('prefix_lens')}")
        hidden_states = self.model_shard(
            h_or_tokens=prepared_inputs["h_or_tokens"],
            cache=prepared_inputs["cache"],
            mask=prepared_inputs.get("mask"),
            block_tables=prepared_inputs.get("block_tables"),
            context_lengths=prepared_inputs.get("context_lengths"),
            slot_mapping=prepared_inputs.get("slot_mapping"),
            state_slot_mapping=prepared_inputs.get("state_slot_mapping"),
            prefix_lens=prepared_inputs.get("prefix_lens"),  # For RoPE offset in prefix cache
        )

        logger.debug(
            f"Processing batch with {len(prepared_inputs['requests'])} requests, "
            f"request status: {prepared_inputs['requests'][0].status}, "
            f"hidden_states shape: {hidden_states.shape}"
        )

        lengths = mx.zeros((len(prepared_inputs["requests"]),), dtype=mx.int32)
        requests = prepared_inputs["requests"]
        for i, req in enumerate(requests):
            if req.is_prefill:
                # Use actual_processed_lengths if available (for prefix cache case),
                # otherwise use context_lengths (total_length)
                if "actual_processed_lengths" in prepared_inputs:
                    lengths[i] = prepared_inputs.get("actual_processed_lengths")[i]
                else:
                    lengths[i] = prepared_inputs.get("context_lengths")[i]
            elif req.is_decoding:
                lengths[i] = 1
            else:
                continue

        # Note: With PagedAttention, we don't need to explicitly update requests with new K/V
        # because they are written in-place to the global cache.
        # self.cache_manager.update_requests(...) is REMOVED.

        # Update prefix cache: insert full blocks after prefill
        if self.enable_prefix_cache:
            for req in requests:
                if req.is_prefill:
                    # Insert all full blocks from this prefill into the prefix cache
                    self.cache_manager.insert_full_blocks_to_cache(req.request_id)

        # === Speculative Decoding: Last Peer Verification ===
        is_speculative = any(req.is_speculative for req in requests) if requests else False

        if is_speculative and self.is_last_peer and return_decoded_tokens:
            # TODO(guozixu): using sampling_info
            sampling_info = SamplingBatchInfo.from_reqs(requests)
            # Last peer: verify draft tokens and generate bonus token
            return self._verify_and_sample(hidden_states, requests)

        # Process last peer: need additional sampling + detokenization
        if return_decoded_tokens:
            sampling_info = SamplingBatchInfo.from_reqs(requests)

            # For MLX, hidden_states at last shard is already logits (after lm_head)
            # hidden_states shape: [batch_size, seq_len, vocab_size]
            token_ids = mx.array(
                self.model_shard.logits_to_tokens(hidden_states, lengths, sampling_info)
            )

            needs_probs = any(
                (isinstance(req, InitialRequest) and req.return_probs)
                or (isinstance(req, IntermediateRequest) and req.return_probs)
                for req in requests
            )

            token_probs = None
            if needs_probs:
                # Extract probability values for sampled tokens
                try:
                    # Get last position logits for each request
                    batch_probs = []
                    for i, req in enumerate(requests):
                        if lengths[i] > 0:
                            # Get logit at last position
                            last_idx = int(lengths[i]) - 1
                            last_logits = hidden_states[i, last_idx, :]  # [vocab_size]
                            probs = last_logits / sampling_info.temperatures.reshape(-1, 1)
                            probs[:] = mx.softmax(probs, axis=-1)
                            # logit_value = float(last_logits[token_id])
                            # batch_logits.append(logit_value)
                            # Extract probability for the sampled token
                            token_id = int(token_ids[i])
                            batch_probs.append(float(probs[i, token_id]))

                    token_probs = batch_probs if batch_probs else None
                except Exception as e:
                    logger.debug(f"Failed to extract token probs: {e}")
                    token_probs = None

            # Return dict with token_ids and optional probs
            return {"hidden_states": token_ids, "probs": token_probs}

        # Intermediate peer: return hidden states without probs
        return {"hidden_states": hidden_states, "probs": None}

    def _verify_and_sample(self, hidden_states: mx.array, requests: List[Request]) -> Dict[str, Any]:
        """
        Verify draft tokens and generate bonus token (on last peer).

        Args:
            hidden_states: (batch, K+1, vocab_size) - Target model logits
            requests: List of speculative requests

        Returns:
            Dict with [num_accepted, bonus_token] for each request
        """
        from parallax.server.speculative.verifier import verify_draft_tokens

        logger.debug(f"Last peer: Verifying {len(requests)} speculative requests")

        results = []

        for i, req in enumerate(requests):
            draft_tokens = req.draft_token_ids
            if draft_tokens is None:
                logger.error(f"No draft tokens for request {req.request_id}")
                # Should not happen, but handle gracefully
                results.append(mx.array([0, 0], dtype=mx.int32))
                continue

            # hidden_states[i] shape: (K+1, vocab_size)
            logits = hidden_states[i]

            try:
                # Verify draft tokens and generate bonus token
                num_accepted, bonus_token = verify_draft_tokens(logits, draft_tokens)

                # Update request with verification results
                req.num_accepted_tokens = num_accepted

                logger.debug(
                    f"Request {req.request_id}: "
                    f"accepted {num_accepted}/{len(draft_tokens)} draft tokens, "
                    f"bonus_token={bonus_token}"
                )

                # Encode results: [num_accepted, bonus_token]
                results.append(mx.array([num_accepted, bonus_token], dtype=mx.int32))

            except Exception as e:
                logger.exception(f"Error verifying request {req.request_id}: {e}")
                # Return default values
                results.append(mx.array([0, 0], dtype=mx.int32))

        # Stack results: (batch, 2) - [num_accepted, bonus_token]
        results_tensor = mx.stack(results, axis=0)

        return {"hidden_states": results_tensor, "probs": None}

    def _release_request(self, rid: str):
        """Release per-request resources in MLX."""
        try:
            if hasattr(self, "cache_manager") and self.cache_manager is not None:
                self.cache_manager.release_request(rid)
        except Exception:
            pass

    def _gen_token_id_from_hidden(self, hidden_states) -> Tuple[int, Any]:
        """
        Extract token ID from hidden states (last peer only).

        For normal decoding: hidden_states contains a single token ID
        For speculative decoding: hidden_states contains [num_accepted, bonus_token]

        Returns token_id, hidden_states
        """
        if hidden_states.dtype == mx.uint32:
            # Normal decoding: single token ID
            next_token_id = int(hidden_states[0])
            hidden_states = hidden_states.astype(mx.int32)
        else:
            # For speculative decoding verification results: [num_accepted, bonus_token]
            # Extract bonus_token (second element) as next_token_id
            flattened = hidden_states.flatten()
            if flattened.size >= 2:
                # Speculative verification: extract bonus_token (second element)
                next_token_id = int(flattened[1])
            else:
                # Regular decode: single token
                next_token_id = int(flattened[0])
        return next_token_id, hidden_states

    # TODO(guozixu): check correctness  
    def _prepare_next_single_request(
        self, request: Request, hidden_states: Any, token_prob: Optional[float] = None
    ) -> Request:
        """
        Override to handle speculative decoding status transitions.

        For speculative decoding:
        - PREFILLING -> SPECULATIVE (after prefill completes)
        - SPECULATIVE -> SPECULATIVE (continue speculative decoding)
        """
        # Import here to avoid circular dependency
        from parallax.server.request import IntermediateRequest

        # This peer is the last peer or a single node.
        if self.is_last_peer and self.is_first_peer:
            assert isinstance(
                request, (InitialRequest, IntermediateRequest)
            ), "Invalid request type for decoding."

            next_token_id, hidden_states = self._gen_token_id_from_hidden(hidden_states)

            # Determine next status based on current status and speculative mode
            if request.is_prefill and self.enable_speculative:
                next_status = RequestStatus.SPECULATIVE
            elif request.is_speculative:
                next_status = RequestStatus.SPECULATIVE
            else:
                next_status = RequestStatus.DECODING

            return IntermediateRequest(
                request_id=request.request_id,
                status=next_status,
                # TODO(guozixu): in speculative decoding, 
                # the current_position should be total_length + num_draft_tokens
                # but `current_position` seems not used anywhere
                current_position=request.total_length + 1,
                input_ids=request.input_ids,
                hidden_states=hidden_states,
                next_token_id=next_token_id,
                routing_table=request.routing_table,
                lora_path=request.lora_path,
                token_prob=token_prob,
                draft_token_ids=getattr(request, 'draft_token_ids', None),
                num_accepted_tokens=getattr(request, 'num_accepted_tokens', 0),
            )
        if self.is_last_peer:
            # Last peer decodes a token and sends it back to the first peer.
            # The token is wrapped in an IntermediateRequest.
            assert isinstance(
                request, IntermediateRequest
            ), "Last peer must receive an IntermediateRequest."

            next_token_id, hidden_states = self._gen_token_id_from_hidden(hidden_states)

            # Determine next status based on current status and speculative mode
            if request.is_prefill and self.enable_speculative:
                next_status = RequestStatus.SPECULATIVE
            elif request.is_speculative:
                next_status = RequestStatus.SPECULATIVE
            else:
                next_status = RequestStatus.DECODING

            return IntermediateRequest(
                request_id=request.request_id,
                status=next_status,
                current_position=request.total_length,
                input_ids=request.input_ids,
                hidden_states=hidden_states,
                next_token_id=next_token_id,
                routing_table=request.routing_table,
                lora_path=request.lora_path,
                token_prob=token_prob,
                draft_token_ids=getattr(request, 'draft_token_ids', None),
                num_accepted_tokens=getattr(request, 'num_accepted_tokens', 0),
            )

        # For first peer (not last) and intermediate peers, use parent class logic
        return super()._prepare_next_single_request(request, hidden_states, token_prob)

    def _prepare_speculative_batch(self, batched_requests: List[Request]) -> Optional[Dict[str, Any]]:
        """
        Prepare speculative decoding batch.

        First peer: Generate K draft tokens using draft model
        Pipeline peers: Prepare for verification using target model

        Args:
            batched_requests: List of speculative requests

        Returns:
            Prepared batch dict, or None if no valid requests
        """
        if not batched_requests:
            return None

        # === First Peer: Draft Phase ===
        if self.is_first_peer and self.enable_speculative:
            logger.debug(f"First peer: Generating draft tokens for {len(batched_requests)} requests")

            draft_tokens_list = []
            valid_requests = []

            for req in batched_requests:
                assert req.is_speculative, f"Request {req.request_id} should be SPECULATIVE"

                # Allocate draft cache if needed
                if not self.draft_cache_manager.has_request(req.request_id):
                    success = self.draft_cache_manager.allocate_request(
                        req.request_id, req.total_length
                    )
                    if not success:
                        logger.warning(f"Failed to allocate draft cache for {req.request_id}")
                        continue

                    # Prefill draft cache with the same tokens
                    # TODO(guozixu): prefill should be a standalone function in **/speculative
                    logger.debug(f"Prefilling draft cache for {req.request_id} with {len(req.input_ids)} tokens")
                    prefill_input = mx.array([req.input_ids])

                    # Create causal mask for prefill
                    from parallax.utils.utils import create_causal_mask
                    prefill_mask = create_causal_mask(len(req.input_ids), len(req.input_ids), self.dtype)

                    # Get draft layer caches
                    draft_layer_caches = self.draft_cache_manager.request_caches[req.request_id]

                    # Debug: Check cache types
                    logger.warning(f"Draft layer caches type: {type(draft_layer_caches)}, length: {len(draft_layer_caches)}")
                    if len(draft_layer_caches) > 0:
                        logger.warning(f"First layer cache type: {type(draft_layer_caches[0])}")

                    # Run draft model prefill
                    _ = self.draft_model_shard(
                        prefill_input,
                        cache=draft_layer_caches,
                        mask=prefill_mask,
                        block_tables=None,
                        context_lengths=None,
                        slot_mapping=None,
                    )

                    # Update draft cache length
                    for layer_cache in draft_layer_caches:
                        layer_cache.current_length = len(req.input_ids)

                # Generate K draft tokens using DraftGenerator
                try:
                    draft_ids = self.draft_generator.generate_draft_tokens(
                        request_id=req.request_id,
                        last_token=req.output_ids[-1],
                        num_tokens=self.num_draft_tokens,
                    )

                    # Attach to request
                    req.draft_token_ids = draft_ids
                    draft_tokens_list.append(draft_ids)
                    valid_requests.append(req)

                    logger.debug(
                        f"Generated {len(draft_ids)} draft tokens for request {req.request_id}"
                    )

                except Exception as e:
                    logger.exception(f"Error generating draft tokens for {req.request_id}: {e}")
                    continue

            if not valid_requests:
                logger.warning("No valid requests after draft generation")
                return None

            # Prepare for target model verification
            return self._prepare_verify_batch(valid_requests, draft_tokens_list)

        # === Pipeline Peers: Verify Phase ===
        else:
            logger.debug(f"Pipeline peer: Preparing verification for {len(batched_requests)} requests")
            return self._prepare_verify_batch(batched_requests, None)

    def _prepare_verify_batch(
        self,
        batched_requests: List[Request],
        draft_tokens_list: Optional[List[List[int]]] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Prepare batch for target model verification.

        Args:
            batched_requests: List of speculative requests
            draft_tokens_list: List of draft token IDs (only on first peer)

        Returns:
            Prepared batch dict for target model
        """
        if not batched_requests:
            return None

        h_or_tokens_list = []
        valid_requests = []

        for i, req in enumerate(batched_requests):
            # Get draft tokens
            if draft_tokens_list is not None:
                draft_tokens = draft_tokens_list[i]
            else:
                draft_tokens = req.draft_token_ids

            if draft_tokens is None:
                logger.warning(f"No draft tokens for request {req.request_id}")
                continue

            # Allocate target cache
            if not self.cache_manager.has_request(req.request_id):
                total_len = req.total_length + len(draft_tokens) + 1  # +1 for bonus token
                success = self.cache_manager.allocate_request(req.request_id, total_len)
                if not success:
                    logger.error(f"Failed to allocate target cache for {req.request_id}")
                    continue

            # Prepare input
            if self.is_first_peer:
                # Input: [last_token] + K draft_tokens
                # Total length: K + 1 tokens
                input_tokens = [req.output_ids[-1]] + draft_tokens
                h_or_tokens_list.append(input_tokens)
            else:
                # Intermediate peers: receive hidden states from previous peer
                h_or_tokens_list.append(req.hidden_states)

            # Pre-allocate slots for K+1 tokens (K draft + 1 bonus)
            # Note: In ContinuousKVCache mode, append_slot is a no-op
            # The actual KV will be appended during model forward
            for _ in range(len(draft_tokens) + 1):
                self.cache_manager.append_slot(req.request_id)

            valid_requests.append(req)

        if not valid_requests:
            logger.warning("No valid requests after verify batch preparation")
            return None

        # Pad inputs (if first peer)
        if self.is_first_peer:
            from parallax.server.executor.mlx_executor import pad_inputs, create_causal_mask, combine_padding_and_causal_masks

            padded_inputs, padding_mask = pad_inputs(
                self.pad_token_id, h_or_tokens_list, self.dtype
            )

            # Create causal mask for K+1 tokens
            # For speculative verification, we need to attend to all cached tokens + new tokens
            max_len = max(len(tokens) for tokens in h_or_tokens_list)

            # Get current cache length (number of tokens already in cache)
            request_id = valid_requests[0].request_id
            cache_len = self.cache_manager.get_context_length(request_id)

            # Total sequence length: cached tokens + new tokens
            total_len = cache_len + max_len

            # Create causal mask: new tokens can attend to all previous + current tokens
            # Shape: (max_len, total_len)
            causal_mask = create_causal_mask(max_len, total_len, self.dtype)

            # Expand padding mask to cover full sequence length
            # padding_mask is (B, 1, 1, max_len), we need (B, 1, 1, total_len)
            # The cached tokens are all valid (not padded), so we prepend ones
            if cache_len > 0:
                batch_size = padding_mask.shape[0]
                # Create mask for cached tokens (all valid)
                cache_mask = mx.ones((batch_size, 1, 1, cache_len), dtype=padding_mask.dtype)
                # Concatenate: [cached_tokens_mask, new_tokens_mask]
                padding_mask = mx.concatenate([cache_mask, padding_mask], axis=3)

            mask = combine_padding_and_causal_masks(padding_mask, causal_mask, self.dtype)
        else:
            # Concatenate hidden states for intermediate peers
            # h_or_tokens_list contains hidden_states arrays
            padded_inputs = mx.concatenate(h_or_tokens_list, axis=0)
            batch_size = len(valid_requests)
            # Reshape to (batch, seq_len, hidden_dim)
            padded_inputs = padded_inputs.reshape(batch_size, -1, self.config.get('hidden_size'))
            mask = None

        # Prepare caches for model forward
        # For single request, get cache objects (not tuples)
        request_id = valid_requests[0].request_id if len(valid_requests) == 1 else None
        if request_id:
            # Get list of ContinuousKVCache objects for each layer
            layer_caches = self.cache_manager.request_caches.get(request_id)
            cache_for_model = layer_caches

            # Get context lengths for RoPE positioning
            # This is critical for correct position embeddings
            cache_len = self.cache_manager.get_context_length(request_id)
            context_lengths_tensor = mx.array([cache_len], dtype=mx.int32)
        else:
            # Batch mode - need to handle multiple requests
            # For now, return None and let model handle it
            cache_for_model = None
            context_lengths_tensor = None

        return {
            "h_or_tokens": padded_inputs,
            "cache": cache_for_model,
            "mask": mask,
            "requests": valid_requests,
            "block_tables": None,  # Not used in speculative mode
            "context_lengths": context_lengths_tensor,  # Needed for cache management
            "slot_mapping": None,  # Not used in speculative mode
            "prefix_lens": mx.array([cache_len], dtype=mx.int32) if request_id else None,  # RoPE starting position
        }

    def _prepare_prefill_batch(self, batched_requests: List[Request]) -> Dict[str, Any]:
        """Prepares inputs for ShardedModel from a batch of prefill requests."""
        batch_size = len(batched_requests)
        if batch_size == 0:
            return None

        h_or_tokens_list = []
        block_tables_list = []
        context_lengths_list = []
        prefix_lens_list = []  # Track matched prefix lengths for each request
        actual_processed_lengths_list = []  # Track actual processed token lengths for each request

        for req in batched_requests:
            assert req.is_prefill, f"Request {req.request_id} is not a prefill request."

            # Allocate Paged KV blocks with prefix cache support
            # For first peer, pass input_ids for prefix matching
            token_ids = None
            if self.is_first_peer and self.enable_prefix_cache:
                token_ids = req.input_ids

            success, matched_tokens = self.cache_manager.allocate_request(
                req.request_id, req.total_length, token_ids=token_ids
            )
            if not success:
                raise RuntimeError(f"OOM during prefill allocation for {req.request_id}")

            prefix_lens_list.append(matched_tokens)

            if self.is_first_peer:
                if matched_tokens > 0 and self.enable_prefix_cache:
                    # Skip the prefix tokens that are already cached
                    # But we must keep at least the last token to generate next token logits
                    new_tokens = req.input_ids[matched_tokens:]
                    if len(new_tokens) == 0:
                        # All tokens cached - keep the last token and adjust prefix_len
                        new_tokens = req.input_ids[-1:]
                        prefix_lens_list[-1] = matched_tokens - 1
                        actual_processed_lengths_list.append(1)
                        logger.debug(
                            f"Request {req.request_id}: Full cache hit, keeping last token for logits"
                        )
                    else:
                        actual_processed_lengths_list.append(len(new_tokens))
                        logger.debug(
                            f"Request {req.request_id}: Skipping {matched_tokens} cached tokens, "
                            f"processing {len(new_tokens)} new tokens"
                        )
                    h_or_tokens_list.append(new_tokens)
                else:
                    h_or_tokens_list.append(req.input_ids)
                    actual_processed_lengths_list.append(len(req.input_ids))
            else:
                if matched_tokens > 0 and self.enable_prefix_cache:
                    # Skip the prefix hidden states that correspond to cached tokens
                    new_hidden = req.hidden_states[matched_tokens:]
                    if new_hidden.shape[0] == 0:
                        # All tokens cached - keep the last hidden state
                        new_hidden = req.hidden_states[-1:]
                        prefix_lens_list[-1] = matched_tokens - 1
                        actual_processed_lengths_list.append(1)
                    else:
                        actual_processed_lengths_list.append(new_hidden.shape[0])
                    h_or_tokens_list.append(new_hidden)
                else:
                    h_or_tokens_list.append(req.hidden_states)
                    actual_processed_lengths_list.append(req.hidden_states.shape[0])

            # Collect block_tables only for paged cache
            if not self.enable_speculative:
                block_table = self.cache_manager.get_block_table(req.request_id)
                block_tables_list.append(block_table)

            # For prefill, context length after this step will be total_length
            context_lengths_list.append(req.total_length)

        if self.is_first_peer:
            padded_inputs, padding_mask = pad_inputs(
                self.pad_token_id, h_or_tokens_list, self.dtype
            )
        else:
            padded_inputs, padding_mask = pad_inputs(0, h_or_tokens_list, self.dtype)

        # Generate slot_mapping for prefill (only for NEW tokens, starting from prefix_len)
        # Only for paged cache - skip for speculative decoding (continuous cache)
        if self.enable_speculative:
            # Speculative decoding uses ContinuousKVCache - no paged cache logic
            slot_mapping_tensor = None
            block_tables_tensor = None
            context_lengths_tensor = mx.array(context_lengths_list, dtype=mx.int32)
        else:
            # Paged cache logic
            max_len = padded_inputs.shape[1]
            slot_mapping_flat = []

            for i, req in enumerate(batched_requests):
                block_table = block_tables_list[i]
                prefix_len = prefix_lens_list[i]
                total_len = req.total_length
                new_tokens_len = total_len - prefix_len

                for seq_idx in range(max_len):
                    if seq_idx < new_tokens_len:
                        # Valid new token - map to position after prefix
                        actual_pos = prefix_len + seq_idx
                        block_idx = actual_pos // self.cache_manager.block_size
                        block_offset = actual_pos % self.cache_manager.block_size
                        physical_block = block_table[block_idx]
                        slot = physical_block * self.cache_manager.block_size + block_offset
                        slot_mapping_flat.append(slot)
                    else:
                        # Padding token
                        # Map to -1. The kernel should ignore this.
                        slot_mapping_flat.append(-1)

            slot_mapping_tensor = mx.array(slot_mapping_flat, dtype=mx.int64)

            # Pad block tables
            max_blocks = max(len(bt) for bt in block_tables_list)
            padded_block_tables = []
            for bt in block_tables_list:
                padded_block_tables.append(bt + [0] * (max_blocks - len(bt)))

            block_tables_tensor = mx.array(padded_block_tables, dtype=mx.int32)
            context_lengths_tensor = mx.array(context_lengths_list, dtype=mx.int32)

        # Create mask for standard attention (used during Prefill computation)
        causal_mask = create_causal_mask(padded_inputs.shape[1], padded_inputs.shape[1], self.dtype)
        mask = combine_padding_and_causal_masks(padding_mask, causal_mask, self.dtype)

        # Prepare state slot mapping if needed
        state_slot_mapping = None
        if hasattr(self.cache_manager, 'needs_slots') and self.cache_manager.needs_slots:
            req_ids = [r.request_id for r in batched_requests]
            slots = [self.cache_manager.get_slot(rid) for rid in req_ids]
            state_slot_mapping = mx.array(slots, dtype=mx.int32)

        # Convert prefix_lens to tensor for models that need RoPE offset adjustment
        prefix_lens_tensor = mx.array(prefix_lens_list, dtype=mx.int32)
        # Convert actual_processed_lengths to tensor for correct logit selection
        actual_processed_lengths_tensor = mx.array(actual_processed_lengths_list, dtype=mx.int32)

        ret = {
            "h_or_tokens": padded_inputs,
            "cache": self.cache_manager.get_caches(),
            "mask": mask,
            "requests": batched_requests,
            "block_tables": block_tables_tensor,
            "context_lengths": context_lengths_tensor,
            "slot_mapping": slot_mapping_tensor,
            "state_slot_mapping": state_slot_mapping,
            "prefix_lens": prefix_lens_tensor,  # For RoPE offset calculation
            "actual_processed_lengths": actual_processed_lengths_tensor,  # For correct logit selection
        }
        logger.debug(f"Prepared MLX prefill batch (size={batch_size})")
        return ret

    def _prepare_decode_batch(self, batched_requests: List[Request]) -> Optional[Dict[str, Any]]:
        """Prepares inputs for ShardedModel from a batch of decode requests."""
        batch_size = len(batched_requests)
        if batch_size == 0:
            return None

        h_or_tokens_list = []
        block_tables_list = []
        context_lengths_list = []
        valid_requests = []

        for req in batched_requests:
            assert req.is_decoding, f"Request {req.request_id} is not a decode request."

            # Allocate slot for new token
            success = self.cache_manager.append_slot(req.request_id)
            if not success:
                logger.error(
                    f"OOM during decode for {req.request_id}. Aborting request and notifying other nodes."
                )
                req.update_status(RequestStatus.FINISHED_ABORT)
                self.cache_manager.free_request(req.request_id)
                self.scheduler.evict_request(req.request_id)
                # Add to finished_batch to trigger abort notification
                self.finished_batch.append(req)

                # If this is First Peer, we must also notify HTTP Server immediately
                if self.is_first_peer and self.tp_rank == 0:
                    req_dict = {
                        "prompt_tokens": req.prompt_len,
                        "next_token_id": (
                            req.output_ids[-1] if req.output_ids else -1
                        ),  # Best effort to return last token
                        "rid": req.request_id,
                        "abort": True,
                    }
                    if hasattr(self, "send_to_ipc_socket"):
                        self.send_to_ipc_socket.send_pyobj(req_dict)

                continue

            # Allocation successful, proceed with batch preparation
            valid_requests.append(req)

            if self.is_first_peer:
                # First peer input is the last generated token
                h_or_tokens_list.append([req.output_ids[-1]])
            else:
                h_or_tokens_list.append(req.hidden_states)

            # Update token_ids for prefix cache (if enabled)
            if self.enable_prefix_cache and self.is_first_peer:
                # Add the new token to the request's token_ids
                self.cache_manager.update_request_tokens(req.request_id, [req.output_ids[-1]])

            # Allocate slot for new token
            # Note: append_slot will automatically insert full blocks to prefix cache
            success = self.cache_manager.append_slot(req.request_id)
            if not success:
                raise RuntimeError(f"OOM during decode for {req.request_id}")

            block_table = self.cache_manager.get_block_table(req.request_id)
            block_tables_list.append(block_table)
            context_lengths_list.append(self.cache_manager.get_context_length(req.request_id))

        # Check if we have any valid requests left
        if not valid_requests:
            return None

        batch_size = len(valid_requests)

        if isinstance(h_or_tokens_list[0], list):
            # First peer case: h_or_tokens_list is list of list of ints [[token_id], ...]
            padded_inputs = mx.array(h_or_tokens_list, dtype=mx.int32)  # (Batch, 1)
        else:
            # Intermediate peer case: h_or_tokens_ list is list of mx.arrays (1, D)
            padded_inputs = mx.concatenate(h_or_tokens_list, axis=0)  # (Batch, D)
            padded_inputs = padded_inputs.reshape(batch_size, 1, -1)  # (Batch, 1, D)

        # Pad block tables
        max_blocks = max(len(bt) for bt in block_tables_list)
        padded_block_tables = []
        for bt in block_tables_list:
            padded_block_tables.append(bt + [0] * (max_blocks - len(bt)))

        block_tables_tensor = mx.array(padded_block_tables, dtype=mx.int32)
        context_lengths_tensor = mx.array(context_lengths_list, dtype=mx.int32)

        # Prepare state slot mapping if needed
        state_slot_mapping = None
        if self.cache_manager.needs_slots:
            req_ids = [r.request_id for r in valid_requests]
            slots = [self.cache_manager.get_slot(rid) for rid in req_ids]
            state_slot_mapping = mx.array(slots, dtype=mx.int32)

        ret = {
            "h_or_tokens": padded_inputs,
            "cache": self.cache_manager.get_caches(),
            "mask": None,
            "requests": valid_requests,
            "block_tables": block_tables_tensor,
            "context_lengths": context_lengths_tensor,
            "slot_mapping": None,
            "state_slot_mapping": state_slot_mapping,
        }
        logger.debug(f"Prepared MLX decode batch (size={batch_size})")
        return ret
