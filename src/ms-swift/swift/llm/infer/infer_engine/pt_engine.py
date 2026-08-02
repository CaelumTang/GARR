# Copyright (c) Alibaba, Inc. and its affiliates.
# Modified for GARR in 2026; see src/ms-swift/GARR_MODIFICATIONS.md.
import asyncio
import hashlib
import inspect
import pickle
import time
from copy import deepcopy
from queue import Queue
from threading import Thread
from typing import Any, AsyncIterator, Dict, Iterator, List, Literal, Optional, Union

import json
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import GenerationConfig, LogitsProcessorList
from transformers.utils import is_torch_npu_available

from swift.llm import InferRequest, Template, TemplateMeta, get_model_tokenizer, safe_snapshot_download, to_device
from swift.plugin import Metric
from swift.tuners import Swift
from ..protocol import (ChatCompletionResponse, ChatCompletionResponseChoice, ChatCompletionResponseStreamChoice,
                        ChatCompletionStreamResponse, ChatMessage, DeltaMessage, EmbeddingResponse,
                        EmbeddingResponseData, RequestConfig, random_uuid)
from .infer_engine import InferEngine
from .utils import AdapterRequest, InferStreamer, LogitsStreamer, TokensIteratorStreamer, prepare_generation_config


class _GenerationConfig(GenerationConfig):

    def __repr__(self) -> str:
        parameters = inspect.signature(self.to_json_string).parameters
        kwargs = {}
        if 'ignore_metadata' in parameters:
            kwargs['ignore_metadata'] = True
        gen_kwargs = json.loads(self.to_json_string(**kwargs))
        gen_kwargs.pop('transformers_version', None)
        return f'GenerationConfig({gen_kwargs})'


class PtEngine(InferEngine):
    """PyTorch inference engine."""

    _GARR_DEFAULT_ID_KEY = 'video_id'
    _GARR_TITLE_MARKER = 'Video-Title:'
    _GARR_CAT_MARKER = 'Video-Category:'
    _GARR_TOPIC_MARKER = 'Video-Topic:'
    _GARR_DESC_MARKER = 'Video-Description:'
    _GARR_PRED_MARKER = 'Please predict'

    def __init__(
            self,
            model_id_or_path: str,
            torch_dtype: Optional[torch.dtype] = None,
            *,
            adapters: List[str] = None,
            max_batch_size: int = 1,  # 0/1: no limit
            model_type: Optional[str] = None,
            use_hf: Optional[bool] = None,
            revision: Optional[str] = None,
            hub_token: Optional[str] = None,
            load_model: bool = True,
            # model kwargs
            attn_impl: Optional[str] = None,
            device_map: Optional[Union[str, Dict[str, Any]]] = None,
            task_type: Optional[str] = None,
            quantization_config=None,
            model_kwargs: Optional[Dict[str, Any]] = None,
            template: Optional[Template] = None,
            **kwargs):
        self.model, self.processor = get_model_tokenizer(
            model_id_or_path,
            torch_dtype,
            load_model=load_model,
            model_type=model_type,
            download_model=True,
            use_hf=use_hf,
            hub_token=hub_token,
            revision=revision,
            device_map=device_map,
            quantization_config=quantization_config,
            attn_impl=attn_impl,
            task_type=task_type,
            model_kwargs=model_kwargs,
            **kwargs)
        self.max_batch_size = max_batch_size
        if isinstance(adapters, str):
            adapters = [adapters]
        self.adapters = adapters or []
        for adapter in self.adapters:
            self._add_adapter(safe_snapshot_download(adapter, use_hf=use_hf, hub_token=hub_token))
        self._post_init(template)

    def _post_init(self, template=None):
        super()._post_init(template)
        self.engine = self.model  # dummy
        self.generation_config = self.model.generation_config
        self._queue = Queue()
        self._task_pool = {}
        self._task_thread = None

    def _start_infer_worker(self):
        self._task_thread = Thread(target=self._infer_worker, daemon=True)
        self._task_thread.start()

    def _fetch_infer_requests(self):
        while not self._queue.empty():
            infer_request, kwargs, queue = self._queue.get()
            template = kwargs['template']
            info = hashlib.sha256(pickle.dumps((kwargs['request_config'], template
                                                and template.template_meta))).hexdigest()
            if info not in self._task_pool:
                self._task_pool[info] = kwargs, []
            self._task_pool[info][1].append((infer_request, queue))
        if len(self._task_pool) == 0:
            return
        key, (kwargs, data) = next(iter(self._task_pool.items()))
        max_batch_size = self.max_batch_size
        if max_batch_size <= 0:
            max_batch_size = len(data)
        data, remain_data = data[:max_batch_size], data[max_batch_size:]
        if remain_data:
            self._task_pool[key] = kwargs, remain_data
        else:
            self._task_pool.pop(key)
        kwargs = kwargs.copy()
        kwargs['infer_requests'] = [d[0] for d in data]
        queue_list = [d[1] for d in data]
        return kwargs, queue_list

    def _infer_worker(self):
        while True:
            time.sleep(0.01)
            item = self._fetch_infer_requests()
            if item is not None:
                kwargs, queue_list = item
                request_config = kwargs['request_config']
                res_list_or_gen = self._infer(**kwargs)
                if request_config.stream:
                    finished = False
                    while not finished:
                        try:
                            res_list = next(res_list_or_gen)
                        except StopIteration:
                            finished = True
                            res_list = [None] * len(queue_list)
                        for (queue, loop), res in zip(queue_list, res_list):
                            asyncio.run_coroutine_threadsafe(queue.put(res), loop)
                else:
                    for (queue, loop), res in zip(queue_list, res_list_or_gen):
                        asyncio.run_coroutine_threadsafe(queue.put(res), loop)

    def _add_adapter(self, adapter_path: str, adapter_name: Optional[str] = None) -> None:
        self.model = Swift.from_pretrained(self.model, adapter_path, adapter_name)

    @classmethod
    def from_model_template(cls, model, template=None, *, max_batch_size: int = 1):
        self = super().__new__(cls)
        self.model = model
        self.processor = template.processor
        self.max_batch_size = max_batch_size
        self._post_init(template)
        return self

    def _prepare_generation_config(self, request_config: RequestConfig) -> _GenerationConfig:
        generation_config = prepare_generation_config(self.generation_config, request_config, self.tokenizer)
        generation_config.return_dict_in_generate = True
        if request_config.logprobs:
            generation_config.output_logits = True
        generation_config.num_return_sequences = request_config.n
        return _GenerationConfig(**generation_config.to_dict())

    def _add_stop_words(self, generation_config: _GenerationConfig, request_config: RequestConfig,
                        template_meta: TemplateMeta) -> None:
        stop_words = (request_config.stop or []) + template_meta.stop_words
        generation_config.stop_words = self._get_stop_words(stop_words)

    @staticmethod
    def preprocess_logits(batched_logits: Optional[List[torch.Tensor]], batched_generate_ids: torch.Tensor,
                          top_logprobs: Optional[int]):
        top_logprobs = top_logprobs or 1
        batch_size = batched_generate_ids.shape[0]
        if batched_logits is None:
            return None
        batched_logprobs = []
        for i in range(batch_size):
            logprobs_list = []
            generate_ids = batched_generate_ids[i]
            for j, logits in enumerate(batched_logits):
                token = generate_ids[j].item()
                logprobs = torch.log_softmax(logits[i], -1)
                tokens = [token] + logprobs.argsort(descending=True, dim=-1)[:top_logprobs].tolist()
                logprobs_list.append({token: logprobs[token].item() for token in tokens})
            batched_logprobs.append(logprobs_list)
        return batched_logprobs

    @staticmethod
    def _update_batched_logprobs(batched_logprobs: List[torch.Tensor], logits_streamer: Optional[LogitsStreamer],
                                 generate_ids: torch.Tensor, top_logprobs: int) -> None:
        seq_len = generate_ids.shape[1] - len(batched_logprobs[0])
        if logits_streamer is None or seq_len == 0:
            return

        res = []
        for i in range(seq_len):
            res.append(logits_streamer.queue.get())
        new_batched_logprobs = PtEngine.preprocess_logits(res, generate_ids[:, -seq_len:], top_logprobs)
        for logprobs, new_logprobs in zip(batched_logprobs, new_batched_logprobs):
            logprobs += new_logprobs

    def _infer_stream(self, template: Template, inputs: Dict[str, Any], *, generation_config: GenerationConfig,
                      adapter_request: Optional[AdapterRequest], request_config: RequestConfig,
                      **kwargs) -> Iterator[List[Optional[ChatCompletionStreamResponse]]]:

        if generation_config.num_beams != 1:
            error_msg = 'Streaming generation does not support beam search.'
            raise ValueError(error_msg)
        streamer = TokensIteratorStreamer()
        generate_kwargs = {
            'generation_config': generation_config,
            'streamer': streamer,
            **inputs,
        }
        adapter_names = self._get_adapter_names(adapter_request)
        if adapter_names is not None:
            generate_kwargs['adapter_names'] = adapter_names
        num_prompt_tokens = self._get_num_tokens(inputs)

        logits_streamer = None
        if generation_config.output_logits:
            generate_kwargs['logits_processor'] = LogitsProcessorList([LogitsStreamer()])

        def _model_generate(**kwargs):
            if is_torch_npu_available():
                torch.npu.set_device(self.model.device)
            template.generate(self.model, **kwargs)

        generate_kwargs = template.prepare_generate_kwargs(generate_kwargs, model=self.model)
        thread = Thread(target=_model_generate, kwargs=generate_kwargs)
        thread.start()
        batch_size = inputs['attention_mask'].shape[0]
        all_is_finished = False
        is_finished = [False] * batch_size
        infer_streamers = [InferStreamer(template) for _ in range(batch_size)]
        request_id_list = [f'chatcmpl-{random_uuid()}' for _ in range(batch_size)]
        token_idxs = [0] * batch_size

        raw_batched_generate_ids = None  # or torch.Tensor: [batch_size, seq_len]
        batched_logprobs = [[] for _ in range(batch_size)]
        while not all_is_finished:
            try:
                batched_tokens = next(streamer)
                if batched_tokens.ndim == 1:
                    batched_tokens = batched_tokens[:, None]

                raw_batched_generate_ids = torch.concat(
                    [batched_tokens]
                    if raw_batched_generate_ids is None else [raw_batched_generate_ids, batched_tokens],
                    dim=1)
            except StopIteration:
                all_is_finished = True

            batched_generate_ids = template.get_generate_ids(raw_batched_generate_ids, num_prompt_tokens)
            self._update_batched_logprobs(batched_logprobs, logits_streamer, batched_generate_ids,
                                          request_config.top_logprobs)

            res = []
            for i in range(batched_generate_ids.shape[0]):
                if is_finished[i]:
                    res.append(None)
                    continue
                generate_ids = batched_generate_ids[i]

                # ignore pad_token
                masks = generate_ids != self.tokenizer.pad_token_id
                generate_ids = generate_ids[masks].tolist()
                logprobs_list = None
                if batched_logprobs[i]:
                    logprobs_list = [logprobs for m, logprobs in zip(masks, batched_logprobs[i]) if m.item()]

                is_finished[i] = (
                    all_is_finished or is_finished[i]
                    or len(generate_ids) > 0 and generate_ids[-1] == self.tokenizer.pad_token_id)
                delta_text = infer_streamers[i].get_printable_text(generate_ids, is_finished[i])
                if not delta_text and not is_finished[i]:
                    res.append(None)
                    continue
                logprobs = self._get_logprobs(logprobs_list, generate_ids[token_idxs[i]:], request_config.top_logprobs)
                token_idxs[i] = len(generate_ids)

                usage_info = self._get_usage_info(num_prompt_tokens, len(generate_ids))
                toolcall = None
                if is_finished[i]:
                    toolcall = self._get_toolcall(template.decode(generate_ids), template)
                finish_reason = self._get_finish_reason(generation_config.max_new_tokens, usage_info.completion_tokens,
                                                        is_finished[i])

                choices = [
                    ChatCompletionResponseStreamChoice(
                        index=0,
                        delta=DeltaMessage(role='assistant', content=delta_text, tool_calls=toolcall),
                        finish_reason=finish_reason,
                        logprobs=logprobs)
                ]
                res.append(
                    ChatCompletionStreamResponse(
                        model=self.model_name, choices=choices, usage=usage_info, id=request_id_list[i]))
            if any(res):
                yield res

    def _get_adapter_names(self, adapter_request: Optional[AdapterRequest]) -> Optional[List[str]]:
        # Qwen2.5-VL PEFT LoRA is incompatible with adapter_names when inputs are flattened
        # to token sequences (len(adapter_names) must match seq_len). Use set_adapter instead.
        model_meta = getattr(self.model, 'model_meta', None) or getattr(getattr(self.model, 'model', None), 'model_meta', None)
        is_qwen2_5_vl = bool(getattr(model_meta, 'model_type', '') == 'qwen2_5_vl')
        if adapter_request is None:
            if self._adapters_pool:
                if is_qwen2_5_vl:
                    return None
                return ['__base__']
            return
        adapter_name = adapter_request.name
        if adapter_name not in self._adapters_pool:
            self._adapters_pool[adapter_name] = adapter_request
            self._add_adapter(adapter_request.path, adapter_name)
            if is_qwen2_5_vl and hasattr(self.model, 'set_adapter'):
                self.model.set_adapter(adapter_name)
        if is_qwen2_5_vl:
            return None
        return [adapter_name]

    @staticmethod
    def _broadcast_adapter_names(adapter_names: Optional[List[str]], batch_size: int) -> Optional[List[str]]:
        """Broadcast adapter_names to match batch_size.

        PEFT LoRA expects len(adapter_names) == batch_size when adapter_names is provided.
        We allow a single adapter name to be broadcast across the whole batch.
        """
        if adapter_names is None:
            return None
        if not isinstance(adapter_names, list) or any(not isinstance(x, str) for x in adapter_names):
            raise ValueError(f'adapter_names must be List[str], got {type(adapter_names)}: {adapter_names!r}')
        bs = int(batch_size)
        if bs <= 0:
            raise ValueError(f'batch_size must be > 0, got {batch_size!r}')
        if len(adapter_names) == bs:
            return adapter_names
        if len(adapter_names) == 1:
            return adapter_names * bs
        raise ValueError(
            f'Length of adapter_names should be the same as the number of inputs, '
            f'but got {len(adapter_names)} and {bs} respectively.'
        )

    def _infer_forward(self, template: Template, inputs: Dict[str, Any], adapter_request: Optional[AdapterRequest],
                       request_config: RequestConfig, **kwargs):
        call_kwargs = {}
        top_logprobs = request_config.top_logprobs or 20
        adapter_names = self._get_adapter_names(adapter_request)
        if adapter_names is not None:
            if 'attention_mask' in inputs and torch.is_tensor(inputs['attention_mask']):
                bs = int(inputs['attention_mask'].shape[0])
            elif 'input_ids' in inputs and torch.is_tensor(inputs['input_ids']):
                bs = int(inputs['input_ids'].shape[0])
            else:
                raise RuntimeError('Cannot infer batch_size for adapter_names broadcasting (missing attention_mask/input_ids).')
            call_kwargs['adapter_names'] = self._broadcast_adapter_names(adapter_names, bs)
        num_prompt_tokens = self._get_num_tokens(inputs)
        inputs.pop('labels', None)
        output = self.model(**inputs, **call_kwargs)
        if hasattr(output, 'logits'):
            logits = output.logits
        elif 'last_hidden_state' in output:
            # embeddings
            logits = output['last_hidden_state']
        if template.task_type == 'seq_cls':
            preds, logprobs = template.decode_seq_cls(logits, top_logprobs)
        elif template.task_type == 'prm':
            preds = template.decode_prm(inputs['input_ids'], logits)
            logprobs = [None] * len(preds)
        elif template.task_type == 'embedding':
            preds = logits
            logprobs = [None] * len(preds)
        else:
            raise ValueError(f'Unsupported task_type: {template.task_type}')

        res = []
        for i, pred in enumerate(preds):
            usage_info = self._get_usage_info(num_prompt_tokens, 1)
            if template.task_type == 'embedding':
                res.append(
                    EmbeddingResponse(
                        model=self.model_name,
                        usage=usage_info,
                        data=[EmbeddingResponseData(embedding=pred.to(torch.float32).cpu().numpy().tolist())]))
            else:
                choices = [
                    ChatCompletionResponseChoice(
                        index=0,
                        message=ChatMessage(role='assistant', content=pred, tool_calls=None),
                        finish_reason='stop',
                        logprobs=logprobs[i])
                ]
                res.append(ChatCompletionResponse(model=self.model_name, choices=choices, usage=usage_info))
        return res

    def _infer_full(self, template: Template, inputs: Dict[str, Any], *, generation_config: GenerationConfig,
                    adapter_request: Optional[AdapterRequest], request_config: RequestConfig,
                    template_inputs) -> List[ChatCompletionResponse]:
        # bos_token TODO: encoder-decoder
        generate_kwargs = {'generation_config': generation_config, **inputs}
        adapter_names = self._get_adapter_names(adapter_request)
        if adapter_names is not None:
            if 'attention_mask' not in inputs or not torch.is_tensor(inputs['attention_mask']):
                raise RuntimeError('GARR generation requires attention_mask for adapter_names broadcasting.')
            bs = int(inputs['attention_mask'].shape[0])
            generate_kwargs['adapter_names'] = self._broadcast_adapter_names(adapter_names, bs)
        num_prompt_tokens = self._get_num_tokens(inputs)
        generate_kwargs = template.prepare_generate_kwargs(generate_kwargs, model=self.model)
        output = dict(template.generate(self.model, **generate_kwargs))
        output.pop('past_key_values', None)
        batched_generate_ids = output['sequences']
        batched_generate_ids = template.get_generate_ids(batched_generate_ids, num_prompt_tokens)
        template.debug_logger({'generate_ids': batched_generate_ids})  # debug
        batched_logprobs = self.preprocess_logits(
            output.get('logits'), batched_generate_ids, request_config.top_logprobs)

        res = []
        num_return_sequences = generation_config.num_return_sequences
        for i in range(inputs['attention_mask'].shape[0]):
            choices = []
            usage_info = self._get_usage_info(num_prompt_tokens, 0)
            for j in range(num_return_sequences):
                batched_index = i * num_return_sequences + j
                generate_ids = batched_generate_ids[batched_index]

                # ignore pad_token
                masks = generate_ids != self.tokenizer.pad_token_id
                generate_ids = generate_ids[masks].tolist()
                logprobs_list = None
                if batched_logprobs is not None:
                    logprobs_list = [
                        logprobs for m, logprobs in zip(masks, batched_logprobs[batched_index]) if m.item()
                    ]

                logprobs = self._get_logprobs(logprobs_list, generate_ids, request_config.top_logprobs)
                usage_info = self._update_usage_info(usage_info, len(generate_ids))
                response = template.decode(generate_ids, template_inputs=template_inputs[i])
                finish_reason = self._get_finish_reason(generation_config.max_new_tokens, len(generate_ids), True)
                toolcall = self._get_toolcall(response, template)
                token_ids = template.skip_stop_tokens(generate_ids) if request_config.return_details else None
                choices.append(
                    ChatCompletionResponseChoice(
                        index=j,
                        message=ChatMessage(role='assistant', content=response, tool_calls=toolcall),
                        finish_reason=finish_reason,
                        logprobs=logprobs,
                        token_ids=token_ids))
            prompt_token_ids = None
            images_size = None
            if request_config.return_details:
                if 'input_ids' in inputs:
                    non_pad_indices = (inputs['input_ids'][i] != self.tokenizer.pad_token_id).nonzero()
                    if non_pad_indices.numel() > 0:
                        idx = non_pad_indices.min().item()
                        prompt_token_ids = inputs['input_ids'][i][idx:].tolist()
                if all(isinstance(image, Image.Image) for image in template_inputs[i].images):
                    images_size = [image.size for image in template_inputs[i].images]
            res.append(
                ChatCompletionResponse(
                    model=self.model_name,
                    choices=choices,
                    usage=usage_info,
                    prompt_token_ids=prompt_token_ids,
                    images_size=images_size))
        return res

    @staticmethod
    def _garr_find_subseq(seq: List[int], pat: List[int], start: int = 0) -> int:
        if not pat:
            return -1
        L = len(seq)
        P = len(pat)
        i = max(0, int(start))
        while i + P <= L:
            if seq[i:i + P] == pat:
                return i
            i += 1
        return -1

    @classmethod
    def _garr_build_masks_from_input_ids(cls, input_ids: torch.Tensor, tokenizer) -> tuple[torch.Tensor, torch.Tensor]:
        """Build (vision_mask, text_mask) from input_ids.

        Returns float masks shaped [B, T] on the same device as input_ids.
        """
        if not torch.is_tensor(input_ids) or input_ids.dim() != 2:
            raise RuntimeError(f'GARR mask requires input_ids as [B,T] tensor, got: {type(input_ids)} {getattr(input_ids, "shape", None)}')

        img_id = tokenizer.convert_tokens_to_ids('<IMG_CONTEXT>')
        unk_id = getattr(tokenizer, 'unk_token_id', None)
        if img_id is None or (unk_id is not None and img_id == unk_id):
            img_id = tokenizer.convert_tokens_to_ids('<|image_pad|>')
        if img_id is None or (unk_id is not None and img_id == unk_id):
            img_id = getattr(tokenizer, 'image_token_id', None)
        if img_id is None or (unk_id is not None and img_id == unk_id):
            img_id = getattr(getattr(tokenizer, 'tokenizer', None), 'image_token_id', None)
        if img_id is None or int(img_id) < 0:
            raise RuntimeError('IMG_CONTEXT token id not found in tokenizer')

        title_ids = tokenizer(cls._GARR_TITLE_MARKER, add_special_tokens=False).input_ids
        cat_ids = tokenizer(cls._GARR_CAT_MARKER, add_special_tokens=False).input_ids
        topic_ids = tokenizer(cls._GARR_TOPIC_MARKER, add_special_tokens=False).input_ids
        desc_ids = tokenizer(cls._GARR_DESC_MARKER, add_special_tokens=False).input_ids
        pred_ids = tokenizer(cls._GARR_PRED_MARKER, add_special_tokens=False).input_ids
        if not title_ids or not pred_ids:
            raise RuntimeError('Failed to tokenize one of the GARR prompt markers for mask building.')
        if not cat_ids and not topic_ids:
            raise RuntimeError('Failed to tokenize one of the GARR prompt markers for mask building.')

        vision_mask = (input_ids == int(img_id)).to(dtype=torch.float32)

        ids_list = input_ids.detach().cpu().tolist()
        tm_list: List[List[float]] = []
        text_mark_total = 0
        for seq in ids_list:
            Tlen = len(seq)
            tm = [0.0] * Tlen

            s_topic = cls._garr_find_subseq(seq, topic_ids, 0) if topic_ids else -1
            s_title_from_topic = cls._garr_find_subseq(seq, title_ids, (s_topic + len(topic_ids)) if s_topic != -1 else 0)
            s_desc = cls._garr_find_subseq(seq, desc_ids, (s_title_from_topic + len(title_ids)) if s_title_from_topic != -1 else 0) if desc_ids else -1
            s_pred_from_desc = cls._garr_find_subseq(seq, pred_ids, (s_desc + len(desc_ids)) if s_desc != -1 else 0)

            used_topic_style = (s_topic != -1 and s_title_from_topic != -1 and s_desc != -1 and s_pred_from_desc != -1)
            if used_topic_style:
                if s_title_from_topic > s_topic:
                    for i in range(s_topic + len(topic_ids), s_title_from_topic):
                        if 0 <= i < Tlen:
                            tm[i] = 1.0
                if s_desc > s_title_from_topic:
                    for i in range(s_title_from_topic + len(title_ids), s_desc):
                        if 0 <= i < Tlen:
                            tm[i] = 1.0
                if s_pred_from_desc > s_desc:
                    for i in range(s_desc + len(desc_ids), s_pred_from_desc):
                        if 0 <= i < Tlen:
                            tm[i] = 1.0
            else:
                s1 = cls._garr_find_subseq(seq, title_ids, 0)
                s2 = cls._garr_find_subseq(seq, cat_ids, (s1 + len(title_ids)) if s1 != -1 else 0) if cat_ids else -1
                s3 = cls._garr_find_subseq(seq, pred_ids, (s2 + len(cat_ids)) if s2 != -1 else 0)
                if s1 != -1 and s2 != -1 and s2 > s1:
                    a = s1 + len(title_ids)
                    b = s2
                    for i in range(a, b):
                        if 0 <= i < Tlen:
                            tm[i] = 1.0
                if s2 != -1 and s3 != -1 and s3 > s2:
                    a = s2 + len(cat_ids)
                    b = s3
                    for i in range(a, b):
                        if 0 <= i < Tlen:
                            tm[i] = 1.0
            text_mark_total += int(sum(tm))
            tm_list.append(tm)

        if text_mark_total <= 0:
            raise RuntimeError('GARR text mask building failed: prompt markers not found in input_ids.')

        text_mask = torch.tensor(tm_list, device=input_ids.device, dtype=torch.float32)
        if text_mask.shape != vision_mask.shape:
            raise RuntimeError(f'GARR mask shape mismatch: vision={tuple(vision_mask.shape)} text={tuple(text_mask.shape)}')
        if float(vision_mask.sum().item()) <= 0.0:
            raise RuntimeError('GARR vision mask building failed: no <IMG_CONTEXT> tokens found in input_ids.')
        return vision_mask, text_mask

    @staticmethod
    def _garr_pool_last_hidden(last_hidden: torch.Tensor, vision_mask: torch.Tensor,
                              text_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Mean-pool final-layer hidden states over mask positions."""
        if not (torch.is_tensor(last_hidden) and last_hidden.dim() == 3):
            raise RuntimeError(f'last_hidden must be [B,T,C], got {type(last_hidden)} {getattr(last_hidden, "shape", None)}')
        if not (torch.is_tensor(vision_mask) and torch.is_tensor(text_mask)):
            raise RuntimeError('vision_mask/text_mask must be tensors.')
        if vision_mask.shape != text_mask.shape or vision_mask.shape[:2] != last_hidden.shape[:2]:
            raise RuntimeError(
                f'shape mismatch: last_hidden={tuple(last_hidden.shape)} vision_mask={tuple(vision_mask.shape)} text_mask={tuple(text_mask.shape)}'
            )
        vm = vision_mask.to(dtype=last_hidden.dtype, device=last_hidden.device)
        tm = text_mask.to(dtype=last_hidden.dtype, device=last_hidden.device)
        v_cnt = vm.sum(dim=1, keepdim=True).clamp_min(1.0)
        t_cnt = tm.sum(dim=1, keepdim=True).clamp_min(1.0)
        v_emb = (last_hidden * vm.unsqueeze(-1)).sum(dim=1) / v_cnt
        t_emb = (last_hidden * tm.unsqueeze(-1)).sum(dim=1) / t_cnt
        return v_emb, t_emb

    def infer_garr(
        self,
        infer_requests: List[Union[InferRequest, Dict[str, Any]]],
        *,
        mode: Literal['score', 'score_emb'] = 'score',
        id_key: str = _GARR_DEFAULT_ID_KEY,
        request_config: Optional[RequestConfig] = None,
        template: Optional[Template] = None,
        adapter_request: Optional[AdapterRequest] = None,
        use_tqdm: bool = True,
    ) -> List[Dict[str, Any]]:
        """Run GARR inference and return rows aligned with the input order."""
        if request_config is None:
            request_config = RequestConfig()
        if request_config.stream:
            raise RuntimeError('infer_garr does not support stream=True.')
        if mode not in ('score', 'score_emb'):
            raise RuntimeError(f'Invalid mode={mode!r}. Expected one of: score, score_emb')

        max_batch_size = self.max_batch_size
        if max_batch_size <= 0:
            max_batch_size = len(infer_requests)

        out: List[Dict[str, Any]] = []
        prog = tqdm(total=len(infer_requests), dynamic_ncols=True, disable=not use_tqdm)
        i = 0
        while i < len(infer_requests):
            batch = infer_requests[i:i + max_batch_size]
            out.extend(
                self.infer_garr_batch(
                    batch,
                    request_config=request_config,
                    template=template,
                    adapter_request=adapter_request,
                    mode=mode,
                    id_key=id_key,
                ))
            i += max_batch_size
            prog.update(len(batch))
        prog.close()
        return out

    @torch.inference_mode()
    def infer_garr_batch(
        self,
        infer_requests: List[Union[InferRequest, Dict[str, Any]]],
        *,
        request_config: RequestConfig,
        template: Optional[Template],
        adapter_request: Optional[AdapterRequest],
        mode: Literal['score', 'score_emb'],
        id_key: str,
    ) -> List[Dict[str, Any]]:
        self.model.eval()
        request_config = deepcopy(request_config)
        if template is None:
            template = self.default_template
        if template.use_model:
            template.model = self.model
        if self.model_info.task_type == 'causal_lm':
            template.set_mode('pt')

        batched_inputs, error_list = self._batch_encode(
            infer_requests, template=template, strict=getattr(self, 'strict', True))
        if error_list:
            raise RuntimeError(f'Batch encode failed: {error_list[0]}')
        if len(batched_inputs) == 0:
            return []

        template_inputs = [inputs.pop('template_inputs') for inputs in batched_inputs]
        inputs_pre = to_device(template.data_collator(batched_inputs), self.model.device)
        if 'input_ids' not in inputs_pre:
            raise RuntimeError('GARR inference requires input_ids prior to pre_forward_hook.')
        input_ids_pre = inputs_pre['input_ids']

        ids: List[str] = []
        gts: List[str] = []
        for req in infer_requests:
            if isinstance(req, InferRequest):
                obj = req.objects or {}
                if id_key not in obj:
                    raise RuntimeError(f'InferRequest.objects missing required id_key={id_key!r}')
                ids.append(str(obj[id_key]))
                gts.append(str(obj.get('ground_truth', '')))
            elif isinstance(req, dict):
                obj = req.get('objects', {}) or {}
                if id_key in obj:
                    ids.append(str(obj[id_key]))
                elif id_key in req:
                    ids.append(str(req[id_key]))
                else:
                    raise RuntimeError(f'InferRequest dict missing id_key={id_key!r} (expected in objects or top-level).')
                if 'ground_truth' in obj:
                    gts.append(str(obj.get('ground_truth', '')))
                elif 'ground_truth' in req:
                    gts.append(str(req.get('ground_truth', '')))
                else:
                    gts.append('')
            else:
                raise RuntimeError(f'Unsupported infer_request type: {type(req)}')

        vision_mask, text_mask = self._garr_build_masks_from_input_ids(input_ids_pre, self.tokenizer)

        inputs = inputs_pre
        if self.model.model_meta.is_multimodal:
            _, inputs = template.pre_forward_hook(self.model, None, inputs)

        v_vecs: Optional[np.ndarray] = None
        t_vecs: Optional[np.ndarray] = None
        if mode == 'score_emb':
            call_kwargs = {}
            adapter_names = self._get_adapter_names(adapter_request)
            if adapter_names is not None:
                bs = int(input_ids_pre.shape[0])
                call_kwargs['adapter_names'] = self._broadcast_adapter_names(adapter_names, bs)
            inputs_forward = dict(inputs)
            inputs_forward.pop('labels', None)
            inputs_forward['return_dict'] = True
            inputs_forward['output_hidden_states'] = True
            inputs_forward['use_cache'] = False
            out = self.model(**inputs_forward, **call_kwargs)
            hs = None
            if hasattr(out, 'hidden_states') and out.hidden_states is not None:
                hs = out.hidden_states
            elif isinstance(out, dict) and 'hidden_states' in out:
                hs = out['hidden_states']
            if hs is None or len(hs) == 0:
                raise RuntimeError('GARR embedding export requires outputs.hidden_states (set output_hidden_states=True).')
            last_hidden = hs[-1]
            v_emb, t_emb = self._garr_pool_last_hidden(last_hidden, vision_mask, text_mask)
            v_vecs = v_emb.detach().to(torch.float32).cpu().numpy()
            t_vecs = t_emb.detach().to(torch.float32).cpu().numpy()

        self.set_default_max_tokens(request_config, inputs)
        generation_config = self._prepare_generation_config(request_config)
        self._add_stop_words(generation_config, request_config, template.template_meta)
        resp_list = self._infer_full(
            template=template,
            inputs=inputs,
            generation_config=generation_config,
            adapter_request=adapter_request,
            request_config=request_config,
            template_inputs=template_inputs,
        )
        results: List[Dict[str, Any]] = []
        for idx, (vid, resp) in enumerate(zip(ids, resp_list)):
            gen_text = resp.choices[0].message.content
            ground_truth = gts[idx] if idx < len(gts) else ''
            row: Dict[str, Any] = {
                id_key: vid,
                'gen_text': gen_text,
                'ground_truth': ground_truth,
            }
            if mode == 'score_emb':
                assert v_vecs is not None and t_vecs is not None
                row['vision_emb'] = v_vecs[idx]
                row['text_emb'] = t_vecs[idx]
            results.append(row)
        return results

    async def infer_async(
        self,
        infer_request: InferRequest,
        request_config: Optional[RequestConfig] = None,
        *,
        template: Optional[Template] = None,
        adapter_request: Optional[AdapterRequest] = None,
        pre_infer_hook=None,
    ) -> Union[ChatCompletionResponse, AsyncIterator[ChatCompletionStreamResponse]]:
        if request_config is None:
            request_config = RequestConfig()
        queue = asyncio.Queue()
        self._queue.put((infer_request, {
            'request_config': request_config,
            'template': template,
            'adapter_request': adapter_request,
            'pre_infer_hook': pre_infer_hook
        }, (queue, asyncio.get_event_loop())))
        await asyncio.sleep(0)
        if self._task_thread is None:
            self._start_infer_worker()
        if request_config.stream:

            async def _gen_wrapper():
                while True:
                    item = await queue.get()
                    await asyncio.sleep(0)
                    if item is None:
                        break
                    yield item

            return _gen_wrapper()
        else:
            return await queue.get()

    # Ensure `template._post_encode` has no gradient.
    @torch.inference_mode()
    def _infer(
        self,
        infer_requests: List[InferRequest],
        request_config: RequestConfig,
        *,
        template: Optional[Template] = None,
        adapter_request: Optional[AdapterRequest] = None,
        pre_infer_hook=None,
    ) -> Union[List[ChatCompletionResponse], Iterator[List[Optional[ChatCompletionStreamResponse]]]]:
        self.model.eval()
        request_config = deepcopy(request_config)
        if template is None:
            template = self.default_template
        if template.use_model:
            template.model = self.model

        if self.model_info.task_type == 'causal_lm':
            template.set_mode('pt')

        batched_inputs, error_list = self._batch_encode(
            infer_requests, template=template, strict=getattr(self, 'strict', True))
        if len(batched_inputs) > 0:
            template_inputs = [inputs.pop('template_inputs') for inputs in batched_inputs]
            inputs = to_device(template.data_collator(batched_inputs), self.model.device)
            template.debug_logger(inputs)  # debug
            if self.model.model_meta.is_multimodal:
                _, inputs = template.pre_forward_hook(self.model, None, inputs)
            if self.model_info.task_type == 'causal_lm':
                self.set_default_max_tokens(request_config, inputs)
                generation_config = self._prepare_generation_config(request_config)
                self._add_stop_words(generation_config, request_config, template.template_meta)
            else:
                generation_config = request_config

            kwargs = {
                'template': template,
                'inputs': inputs,
                'generation_config': generation_config,
                'adapter_request': adapter_request,
                'request_config': request_config,
                'template_inputs': template_inputs,
            }
            if pre_infer_hook:
                kwargs = pre_infer_hook(kwargs)
        else:
            kwargs = {}
        if request_config.stream:

            def _gen_wrapper():
                if len(kwargs) > 0:
                    for res in self._infer_stream(**kwargs):
                        yield self._add_error_list(res, error_list)
                else:
                    yield self._add_error_list([], error_list)

            return _gen_wrapper()
        else:
            if len(kwargs) > 0:
                infer_func = self._infer_forward if template.task_type in {'seq_cls', 'prm', 'embedding'
                                                                           } else self._infer_full
                res = infer_func(**kwargs)
            else:
                res = []
            return self._add_error_list(res, error_list)

    def infer(
        self,
        infer_requests: List[InferRequest],
        request_config: Optional[RequestConfig] = None,
        metrics: Optional[List[Metric]] = None,
        *,
        template: Optional[Template] = None,
        use_tqdm: Optional[bool] = None,
        adapter_request: Optional[AdapterRequest] = None
    ) -> List[Union[ChatCompletionResponse, Iterator[ChatCompletionStreamResponse]]]:
        if request_config is None:
            request_config = RequestConfig()
        if request_config.stream:
            return super().infer(
                infer_requests,
                request_config,
                metrics,
                template=template,
                use_tqdm=use_tqdm,
                adapter_request=adapter_request)
        # Has higher stability than calling super().infer
        if use_tqdm is None:
            use_tqdm = not request_config.stream and len(infer_requests) > 1
        prog_bar = tqdm(total=len(infer_requests), dynamic_ncols=True, disable=not use_tqdm)
        # If self.max_batch_size <= 0, then process all infer_requests at once.
        max_batch_size = self.max_batch_size
        if max_batch_size <= 0:
            max_batch_size = len(infer_requests)
        res = []
        i = 0
        while i < len(infer_requests):
            infer_requests_samples = infer_requests[i:i + max_batch_size]
            res += self._infer(
                infer_requests_samples, request_config, template=template, adapter_request=adapter_request)
            i += max_batch_size
            prog_bar.update(len(infer_requests_samples))
        prog_bar.close()
        self._update_metrics(res, metrics)
        return res
