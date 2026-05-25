import os
import sys
import json
import shutil
import asyncio
import multiprocessing
from dataclasses import asdict, dataclass, field
from datetime import datetime
from functools import partial
from typing import Callable, Dict, List, Optional, Type, Union, cast
from transformers import AutoModel, AutoTokenizer
import torch
import tiktoken


from ._llm import (
    LLMConfig,
    openai_config,
    azure_openai_config,
    ollama_config
)
from ._op import (
    chunking_by_video_segments,
    extract_entities,
    get_chunks,
    videorag_query,
    videorag_query_multiple_choice,
)
from .pipeline.EBR_RAG import EBR_RAG_answer
from ._storage import (
    JsonKVStorage,
    NanoVectorDBStorage,
    NanoVectorDBVideoSegmentStorage,
    NetworkXStorage,
)
from ._utils import (
    EmbeddingFunc,
    compute_mdhash_id,
    limit_async_func_call,
    wrap_embedding_func_with_attrs,
    convert_response_to_json,
    always_get_an_event_loop,
    logger,
)
from .base import (
    BaseGraphStorage,
    BaseKVStorage,
    BaseVectorStorage,
    StorageNameSpace,
    QueryParam,
)
from ._videoutil import(
    split_video,
    speech_to_text,
    segment_caption,
    merge_segment_information,
    saving_video_segments,
)


@dataclass
class VideoRAG:
    working_dir: str = field(
        default_factory=lambda: f"./videorag_cache_{datetime.now().strftime('%Y-%m-%d-%H:%M:%S')}"
    )
    
    # video
    threads_for_split: int = 10
    video_segment_length: int = 30 # seconds
    rough_num_frames_per_segment: int = 5 # frames
    fine_num_frames_per_segment: int = 15 # frames
    video_output_format: str = "mp4"
    audio_output_format: str = "mp3"
    video_embedding_batch_num: int = 2
    segment_retrieval_top_k: int = 8 # Ablation: Tweak baseline to 8 (slightly larger/equal to EBR-RAG max cap)
    video_embedding_dim: int = 1024
    
    # query
    retrieval_topk_chunks: int = 8 # Ablation: Tweak baseline to 8
    query_better_than_threshold: float = 0.2
    
    # graph mode
    enable_local: bool = True
    enable_naive_rag: bool = True

    # text chunking
    chunk_func: Callable[
        [
            list[list[int]],
            List[str],
            tiktoken.Encoding,
            Optional[int],
        ],
        List[Dict[str, Union[str, int]]],
    ] = chunking_by_video_segments
    chunk_token_size: int = 1200
    # chunk_overlap_token_size: int = 100
    tiktoken_model_name: str = "gpt-4o"

    # entity extraction
    entity_extract_max_gleaning: int = 1
    entity_summary_to_max_tokens: int = 500

    # Change to your LLM provider
    llm: LLMConfig = field(default_factory=lambda: openai_config)
    
    # entity extraction
    entity_extraction_func: callable = extract_entities
    
    # storage
    key_string_value_json_storage_cls: Type[BaseKVStorage] = JsonKVStorage
    vector_db_storage_cls: Type[BaseVectorStorage] = NanoVectorDBStorage
    vs_vector_db_storage_cls: Type[BaseVectorStorage] = NanoVectorDBVideoSegmentStorage
    vector_db_storage_cls_kwargs: dict = field(default_factory=dict)
    graph_storage_cls: Type[BaseGraphStorage] = NetworkXStorage
    enable_llm_cache: bool = True
    use_tm_graph: bool = False # Ablation flag for TM Graph RAG

    # extension
    always_create_working_dir: bool = True
    addon_params: dict = field(default_factory=dict)
    convert_response_to_json_func: callable = convert_response_to_json

    def load_caption_model(self, debug=False):
        # caption model
        if not debug:
            model_path = os.path.abspath("./MiniCPM-V-2_6-int4")
            if not os.path.exists(model_path):
                model_path = "openbmb/MiniCPM-V-2_6-int4"
            self.caption_model = AutoModel.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map="cuda")
            self.caption_tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            self.caption_model.eval()
        else:
            self.caption_model = None
            self.caption_tokenizer = None
    
    def __post_init__(self):
        _print_config = ",\n  ".join([f"{k} = {v}" for k, v in asdict(self).items()])
        logger.debug(f"VideoRAG init with param:\n\n  {_print_config}\n")
        
        if not os.path.exists(self.working_dir) and self.always_create_working_dir:
            logger.info(f"Creating working directory {self.working_dir}")
            os.makedirs(self.working_dir)

        self.video_path_db = self.key_string_value_json_storage_cls(
            namespace="video_path", global_config=asdict(self)
        )
        
        self.video_segments = self.key_string_value_json_storage_cls(
            namespace="video_segments", global_config=asdict(self)
        )

        ns_suffix = "_tm" if self.use_tm_graph else ""

        self.text_chunks = self.key_string_value_json_storage_cls(
            namespace=f"text_chunks{ns_suffix}", global_config=asdict(self)
        )

        self.llm_response_cache = (
            self.key_string_value_json_storage_cls(
                namespace=f"llm_response_cache{ns_suffix}", global_config=asdict(self)
            )
            if self.enable_llm_cache
            else None
        )

        self.chunk_entity_relation_graph = self.graph_storage_cls(
            namespace=f"chunk_entity_relation{ns_suffix}", global_config=asdict(self)
        )

        self.embedding_func = limit_async_func_call(self.llm.embedding_func_max_async)(wrap_embedding_func_with_attrs(
                embedding_dim = self.llm.embedding_dim,
                max_token_size = self.llm.embedding_max_token_size,
                model_name = self.llm.embedding_model_name)(self.llm.embedding_func))
        self.entities_vdb = (
            self.vector_db_storage_cls(
                namespace=f"entities{ns_suffix}",
                global_config=asdict(self),
                embedding_func=self.embedding_func,
                meta_fields={"entity_name"},
            )
            if self.enable_local
            else None
        )
        self.chunks_vdb = (
            self.vector_db_storage_cls(
                namespace=f"chunks{ns_suffix}",
                global_config=asdict(self),
                embedding_func=self.embedding_func,
            )
            if self.enable_naive_rag
            else None
        )
        
        self.video_segment_feature_vdb = (
            self.vs_vector_db_storage_cls(
                namespace="video_segment_feature",
                global_config=asdict(self),
                embedding_func=None, # we code the embedding process inside the insert() function.
            )
        )

        self.llm.best_model_func = limit_async_func_call(self.llm.best_model_max_async)(
            partial(self.llm.best_model_func, hashing_kv=self.llm_response_cache)
        )
        self.llm.cheap_model_func = limit_async_func_call(self.llm.cheap_model_max_async)(
            partial(self.llm.cheap_model_func, hashing_kv=self.llm_response_cache)
        )
        
        if self.use_tm_graph:
            from ._op import extract_entities_tm
            self.entity_extraction_func = extract_entities_tm

    def insert_video(self, video_path_list=None):
        loop = always_get_an_event_loop()
        for video_path in video_path_list:
            # Step0: check the existence
            video_name = os.path.basename(video_path).split('.')[0]
            existing_data = self.video_segments._data.get(video_name, {})
            
            # check the completeness
            all_done = False
            if existing_data:
                all_done = all(v.get("content") is not None and "Caption:\nNone" not in v.get("content") for v in existing_data.values())
            
            if all_done:
                logger.info(f"Find the fully processed video named {os.path.basename(video_path)} in storage and skip it.")
                continue
            
            loop.run_until_complete(self.video_path_db.upsert(
                {video_name: video_path}
            ))
            
            # Check if visual features already exist in VDB
            has_features = False
            try:
                # NanoVectorDB stores data in .data (list) and mapping in ._map
                if f"{video_name}_0" in self.video_segment_feature_vdb._client._map:
                    has_features = True
            except (AttributeError, KeyError):
                pass

            # Step1: split the videos (SKIP if has_features and has_captions)
            # We still need segment_index2name and segment_times_info for subsequent steps
            # if we are skipping, we reconstruct them from existing_data
            if has_features and existing_data:
                logger.info(f"Find visual features for {video_name} in storage. Skipping Step 1-6.")
                segment_index2name = {index: f"dummy_{index}" for index in existing_data.keys()}
                segment_times_info = {index: v.get("time") for index, v in existing_data.items()}
            else:
                segment_index2name, segment_times_info = split_video(
                    video_path, 
                    self.working_dir, 
                    self.video_segment_length,
                    self.rough_num_frames_per_segment,
                    self.audio_output_format,
                )

            
            # Step2: obtain transcript with whisper (skip if already exists in existing_data)
            has_transcripts = existing_data and all(v.get("transcript") is not None for v in existing_data.values())
            if not has_transcripts:
                transcripts = speech_to_text(
                    video_name, 
                    self.working_dir, 
                    segment_index2name,
                    self.audio_output_format
                )
                # save temporary information for whisper
                partial_segments = merge_segment_information(
                    segment_index2name,
                    segment_times_info,
                    transcripts,
                    {index: "None" for index in segment_index2name}, # Temporary NO caption
                )
                loop.run_until_complete(self.video_segments.upsert({video_name: partial_segments}))
                loop.run_until_complete(self._save_video_segments())
            else:
                logger.info(f"Find transcripts for {video_name} in storage and skip whisper.")
                transcripts = {index: v.get("transcript") for index, v in existing_data.items()}
            
            # Step3: saving video segments **as well as** obtain caption with vision language model
            manager = multiprocessing.Manager()
            captions = manager.dict()
            error_queue = manager.Queue()
            
            has_captions = existing_data and all(v.get("content") is not None and "Caption:\nNone" not in v.get("content") for v in existing_data.values())
            
            if not has_captions:
                process_saving_video_segments = multiprocessing.Process(
                    target=saving_video_segments,
                    args=(
                        video_name,
                        video_path,
                        self.working_dir,
                        segment_index2name,
                        segment_times_info,
                        error_queue,
                        self.video_output_format,
                    )
                )
                
                process_segment_caption = multiprocessing.Process(
                    target=segment_caption,
                    args=(
                        video_name,
                        video_path,
                        segment_index2name,
                        transcripts,
                        segment_times_info,
                        captions,
                        error_queue,
                    )
                )
                
                process_saving_video_segments.start()
                process_segment_caption.start()
                
                # Monitor processes
                import time
                while process_saving_video_segments.is_alive() and process_segment_caption.is_alive():
                    time.sleep(1)
                
                # if one died, check for error and terminate other
                if not process_segment_caption.is_alive() and process_segment_caption.exitcode != 0:
                    process_saving_video_segments.terminate()
                if not process_saving_video_segments.is_alive() and process_saving_video_segments.exitcode != 0:
                    process_segment_caption.terminate()

                process_saving_video_segments.join()
                process_segment_caption.join()
                
                # if raise error in this two, stop the processing
                error_messages = []
                while not error_queue.empty():
                    error_messages.append(error_queue.get())
                
                if error_messages:
                    for error_message in error_messages:
                        with open('error_log_videorag.txt', 'a', encoding='utf-8') as log_file:
                            log_file.write(f"Video Name:{video_name} Error processing:\n{error_message}\n\n")
                    raise RuntimeError("\n".join(error_messages))
            else:
                logger.info(f"Find captions for {video_name} in storage and skip vlm.")
                captions = {index: v.get("content").split("Caption:\n")[1].split("\nTranscript:")[0] for index, v in existing_data.items()}
            
            # Step4: insert video segments information
            segments_information = merge_segment_information(
                segment_index2name,
                segment_times_info,
                transcripts,
                captions,
            )
            manager.shutdown()
            loop.run_until_complete(self.video_segments.upsert(
                {video_name: segments_information}
            ))
            
            # Step5: encode video segment features
            if not has_features:
                loop.run_until_complete(self.video_segment_feature_vdb.upsert(
                    video_name,
                    segment_index2name,
                    self.video_output_format,
                ))
            else:
                logger.info(f"Visual features for {video_name} already exist. Skipping encoding.")

            
            # Step6: delete the cache file
            video_segment_cache_path = os.path.join(self.working_dir, '_cache', video_name)
            if os.path.exists(video_segment_cache_path):
                shutil.rmtree(video_segment_cache_path)
            
            # Step 7: saving current video information
            loop.run_until_complete(self._save_video_segments())
        
        loop.run_until_complete(self.ainsert(self.video_segments._data))

    def query(self, query: str, param: QueryParam = QueryParam()):
        loop = always_get_an_event_loop()
        return loop.run_until_complete(self.aquery(query, param))

    async def aquery(self, query: str, param: QueryParam = QueryParam()):
        if param.mode == "EBR_RAG":
            response = await EBR_RAG_answer(self, query, param)
            # For backward compatibility with official eval scripts, return string by default
            if isinstance(response, dict) and not param.return_detailed:
                return response.get("answer", "")
        elif param.mode == "videorag":
            response = await videorag_query(
                query,
                self.entities_vdb,
                self.text_chunks,
                self.chunks_vdb,
                self.video_path_db,
                self.video_segments,
                self.video_segment_feature_vdb,
                self.chunk_entity_relation_graph,
                self.caption_model, 
                self.caption_tokenizer,
                param,
                asdict(self),
            )
        elif param.mode == "videorag_multiple_choice":
            response = await videorag_query_multiple_choice(
                query,
                self.entities_vdb,
                self.text_chunks,
                self.chunks_vdb,
                self.video_path_db,
                self.video_segments,
                self.video_segment_feature_vdb,
                self.chunk_entity_relation_graph,
                self.caption_model, 
                self.caption_tokenizer,
                param,
                asdict(self),
            )
        else:
            raise ValueError(f"Unknown mode {param.mode}")
        await self._query_done()
        return response

    async def ainsert(self, new_video_segment):
        await self._insert_start()
        try:
            # ---------- chunking
            inserting_chunks = get_chunks(
                new_videos=new_video_segment,
                chunk_func=self.chunk_func,
                max_token_size=self.chunk_token_size,
            )
            _add_chunk_keys = await self.text_chunks.filter_keys(
                list(inserting_chunks.keys())
            )
            inserting_chunks = {
                k: v for k, v in inserting_chunks.items() if k in _add_chunk_keys
            }
            if not len(inserting_chunks):
                logger.warning(f"All chunks are already in the storage")
                return
            logger.info(f"[New Chunks] inserting {len(inserting_chunks)} chunks")
            if self.enable_naive_rag:
                logger.info("Insert chunks for naive RAG")
                await self.chunks_vdb.upsert(inserting_chunks)

            # TODO: no incremental update for communities now, so just drop all
            # await self.community_reports.drop()

            # ---------- extract/summary entity and upsert to graph
            logger.info("[Entity Extraction]...")
            maybe_new_kg, _, _ = await self.entity_extraction_func(
                inserting_chunks,
                knowledge_graph_inst=self.chunk_entity_relation_graph,
                entity_vdb=self.entities_vdb,
                global_config=asdict(self),
            )
            if maybe_new_kg is None:
                raise RuntimeError("Entity extraction returned None. Halting ingestion.")
            self.chunk_entity_relation_graph = maybe_new_kg
            # ---------- commit upsertings and indexing
            await self.text_chunks.upsert(inserting_chunks)

        finally:
            await self._insert_done()

    async def _insert_start(self):
        tasks = []
        for storage_inst in [
            self.chunk_entity_relation_graph,
        ]:
            if storage_inst is None:
                continue
            tasks.append(cast(StorageNameSpace, storage_inst).index_start_callback())
        await asyncio.gather(*tasks)

    async def _save_video_segments(self):
        tasks = []
        for storage_inst in [
            self.video_segment_feature_vdb,
            self.video_segments,
            self.video_path_db,
        ]:
            if storage_inst is None:
                continue
            tasks.append(cast(StorageNameSpace, storage_inst).index_done_callback())
        await asyncio.gather(*tasks)
    
    async def _insert_done(self):
        tasks = []
        for storage_inst in [
            self.text_chunks,
            self.llm_response_cache,
            self.entities_vdb,
            self.chunks_vdb,
            self.chunk_entity_relation_graph,
            self.video_segment_feature_vdb,
            self.video_segments,
            self.video_path_db,
        ]:
            if storage_inst is None:
                continue
            tasks.append(cast(StorageNameSpace, storage_inst).index_done_callback())
        await asyncio.gather(*tasks)

    async def _query_done(self):
        tasks = []
        for storage_inst in [self.llm_response_cache]:
            if storage_inst is None:
                continue
            tasks.append(cast(StorageNameSpace, storage_inst).index_done_callback())
        await asyncio.gather(*tasks)
