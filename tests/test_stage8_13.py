from __future__ import annotations
import copy
import numpy as np
from src.generation.llm import CallableLLM
from src.generation.prompts import build_rag_prompt, format_context
from src.pipelines.closed_book import ClosedBookPipeline
from src.pipelines.dense_rag import DenseRAGPipeline
from src.pipelines.hybrid_rag import HybridRAGPipeline
from src.pipelines.reranked_rag import RerankedRAGPipeline
from src.pipelines.settings import ExperimentSettings
from src.retrieval.base import BaseRetriever
from src.retrieval.fusion import ReciprocalRankFusion
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.query_processor import process_query
from src.retrieval.reranker import CrossEncoderReranker
from src.retrieval.sparse import BM25Retriever

QUESTION={"financebench_id":"q1","question":"What was revenue?","answer":"$100 million"}
CHUNKS=[
    {"chunk_id":"c1","doc_name":"doc","page":1,"text":"Revenue was $100 million.","rank":1,"score":0.9},
    {"chunk_id":"c2","doc_name":"doc","page":2,"text":"Other text.","rank":2,"score":0.8},
]
class StaticRetriever(BaseRetriever):
    def __init__(self,rows): self.rows=rows; self.requested_k=None
    def retrieve(self,query,k=5): self.requested_k=k; return [dict(x) for x in self.rows[:k]]
class DiagnosticRetriever(StaticRetriever):
    def __init__(self,rows,processed_query):
        super().__init__(rows); self.processed_query=processed_query
    def retrieve_with_diagnostics(self,query,k=5):
        return self.retrieve(query,k), {"processed_retrieval_query":self.processed_query}
class FakeCE:
    def __init__(self): self.pairs=[]
    def predict(self,pairs,**kwargs):
        self.pairs=list(pairs)
        return np.array([0.2,0.9])
class FakeBM25:
    corpus_size=3
    def get_scores(self,tokens): return np.array([100.0,1.0,90.0])
class WindowTokenizer:
    is_fast=True
    model_max_length=12
    def num_special_tokens_to_add(self,pair=False): return 3 if pair else 2
    def __call__(self,text,text_pair=None,add_special_tokens=True,truncation=False,return_offsets_mapping=False):
        def spans(value):
            import re
            return [(m.group(0),m.start(),m.end()) for m in re.finditer(r"\S+",str(value))]
        left=spans(text)
        if text_pair is not None:
            right=spans(text_pair)
            extra=self.num_special_tokens_to_add(pair=True) if add_special_tokens else 0
            return {"input_ids": list(range(len(left)+len(right)+extra))}
        result={"input_ids": list(range(len(left)))}
        if return_offsets_mapping:
            result["offset_mapping"]=[(start,end) for _,start,end in left]
        return result
class WindowCE:
    tokenizer=WindowTokenizer()
    max_length=12
    def predict(self,pairs,**kwargs):
        return np.array([5.0 if "target amount" in doc else 0.5 for _,doc in pairs])
def fake_llm(system,user): return "$100 million"

def check_common(result,arch):
    assert result["question_id"]=="q1"
    assert result["architecture"]==arch
    assert result["generated_answer"]=="$100 million"
    assert result["generation_latency_ms"]>=0
    assert result["total_latency_ms"]>=0

def test_p0():
    r=ClosedBookPipeline(CallableLLM(fake_llm)).answer(QUESTION)
    check_common(r,"P0")
    assert r["retrieved_chunks"]==[]
    assert r["retrieval_latency_ms"] is None

def test_p1():
    retriever=StaticRetriever(CHUNKS)
    r=DenseRAGPipeline(
        CallableLLM(fake_llm),retriever,top_k=1,candidate_k=2
    ).answer(QUESTION)
    check_common(r,"P1")
    assert retriever.requested_k==2
    assert len(r["retrieved_chunks"])==1
    assert r["candidate_count_before_selection"]==2
    assert r["retrieval_latency_ms"]>=0

def test_p2():
    retriever=StaticRetriever(CHUNKS)
    r=HybridRAGPipeline(
        CallableLLM(fake_llm),retriever,candidate_k=2,final_k=1
    ).answer(QUESTION)
    check_common(r,"P2")
    assert retriever.requested_k==2
    assert len(r["retrieved_chunks"])==1
    assert r["candidate_count_before_selection"]==2

def test_p3():
    original=copy.deepcopy(CHUNKS)
    retriever=DiagnosticRetriever(CHUNKS,"revenue sales turnover")
    cross_encoder=FakeCE()
    r=RerankedRAGPipeline(
        CallableLLM(fake_llm),retriever,
        CrossEncoderReranker(cross_encoder),candidate_k=2,final_k=1
    ).answer(QUESTION)
    check_common(r,"P3")
    assert CHUNKS==original
    assert r["candidate_count_before_reranking"]==2
    assert r["retrieved_chunks"][0]["chunk_id"]=="c2"
    assert r["reranking_latency_ms"]>=0
    assert {query for query,_ in cross_encoder.pairs}=={"revenue sales turnover"}
    assert r["retrieval_diagnostics"]["reranker_query"]=="revenue sales turnover"

def test_hybrid_diagnostics_include_processed_retrieval_query():
    dense=StaticRetriever(CHUNKS)
    sparse=StaticRetriever(CHUNKS)
    hybrid=HybridRetriever(
        dense,sparse,ReciprocalRankFusion(k=60),dense_k=2,sparse_k=2
    )
    _,diagnostics=hybrid.retrieve_with_diagnostics(
        "What was 3M's PP&E balance sheet amount in 2018?",k=2
    )
    processed=diagnostics["processed_retrieval_query"]
    assert "property plant and equipment net" in processed
    assert "consolidated balance sheet" in processed
    assert processed.endswith("2018")

def test_query_processor_extracts_plain_year_and_collapsed_company():
    processed=process_query(
        "What was American Express revenue in 2022?",
        companies=["AMERICANEXPRESS","3M"],
    )
    assert processed.company=="americanexpress"
    assert processed.fiscal_year==2022
    assert process_query("Compare 2022 with 2021 for 3M.",companies=["3M"]).fiscal_year is None

def test_query_processor_removes_task_boilerplate_and_preserves_source_hint():
    processed=process_query(
        "Assume that you are a public equities analyst. Answer the following "
        "question by primarily using information that is shown in the balance "
        "sheet: what is FY2018 net PPNE for 3M? Answer in USD billions.",
        companies=["3M"],
    )
    assert "public equities analyst" not in processed.retrieval_query
    assert processed.retrieval_query.startswith("balance sheet:")
    assert "property plant and equipment net" in processed.retrieval_query
    assert "total assets liabilities equity" not in processed.retrieval_query

def test_query_processor_expands_capital_intensity_metrics():
    processed=process_query(
        "Is 3M a capital-intensive business based on FY2022 data?",
        companies=["3M"],
    )
    for term in ("capital expenditures","revenue","total assets","return on assets"):
        assert term in processed.retrieval_query

def test_bm25_filters_company_and_unambiguous_year():
    rows=[
        {"chunk_id":"wrong_year","doc_name":"3M_2023_10K","page":1,"text":"2023", "retrieval_text":"2023", "company":"3M","fiscal_year":2023},
        {"chunk_id":"right","doc_name":"3M_2022_10K","page":1,"text":"2022", "retrieval_text":"2022", "company":"3M","fiscal_year":2022},
        {"chunk_id":"wrong_company","doc_name":"ACME_2022_10K","page":1,"text":"2022", "retrieval_text":"2022", "company":"ACME","fiscal_year":2022},
    ]
    retriever=BM25Retriever(FakeBM25(),rows)
    results=retriever.retrieve("What was the 3M amount in 2022?",k=3)
    assert [row["chunk_id"] for row in results]==["right"]

def test_reranker_scores_overlapping_windows_for_long_candidates():
    docs=[
        {"chunk_id":"long","doc_name":"doc","page":1,"text":"filler " * 20 + "target amount"},
        {"chunk_id":"short","doc_name":"doc","page":2,"text":"pension assumptions"},
    ]
    results=CrossEncoderReranker(WindowCE(),window_overlap=2).rerank("find target",docs,top_k=2)
    assert results[0]["chunk_id"]=="long"
    assert results[0]["reranker_input_truncated"] is True
    assert results[0]["reranker_window_count"]>1
    assert "target amount" in results[0]["reranker_best_text"]

def test_same_rag_prompt_is_deterministic():
    assert build_rag_prompt(QUESTION["question"],CHUNKS)==build_rag_prompt(QUESTION["question"],CHUNKS)

def test_generation_uses_canonical_chunk_not_reranker_window():
    chunk={**CHUNKS[0],"reranker_best_text":"window-only text"}
    context=format_context([chunk])
    assert "Revenue was $100 million." in context
    assert "window-only text" not in context

def test_frozen_architecture_candidate_budgets():
    settings=ExperimentSettings()
    p1=settings.result_metadata("P1")
    p2=settings.result_metadata("P2")
    p3=settings.result_metadata("P3")
    assert p1["dense_candidate_k"]==5
    assert (p2["dense_candidate_k"],p2["sparse_candidate_k"],p2["fusion_candidate_k"])==(10,10,5)
    assert (p3["dense_candidate_k"],p3["sparse_candidate_k"],p3["fusion_candidate_k"])==(20,20,20)
    assert p1["final_context_k"]==p2["final_context_k"]==p3["final_context_k"]==5
    assert p2["rrf_k"]==p3["rrf_k"]==60
