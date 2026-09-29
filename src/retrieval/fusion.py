from __future__ import annotations
from collections import defaultdict
from typing import Any, Sequence

class ReciprocalRankFusion:
    def __init__(self, k: int = 60) -> None:
        if k < 0:
            raise ValueError("RRF k must be >= 0.")
        self.k = int(k)

    def fuse(self, *result_sets: Sequence[dict[str, Any]], source_names: Sequence[str] | None = None) -> list[dict[str, Any]]:
        if not result_sets:
            return []
        if source_names is None:
            source_names = [f"source_{i}" for i in range(len(result_sets))]
        if len(source_names) != len(result_sets):
            raise ValueError("source_names must match result_sets.")
        rrf_scores = defaultdict(float)
        docs, ranks, scores = {}, defaultdict(dict), defaultdict(dict)
        for source, results in zip(source_names, result_sets):
            seen = set()
            for pos, result in enumerate(results, start=1):
                chunk_id = str(result["chunk_id"])
                if chunk_id in seen:
                    raise ValueError(f"Duplicate chunk {chunk_id} in {source}.")
                seen.add(chunk_id)
                rank = int(result.get("rank", pos))
                if rank <= 0:
                    raise ValueError("Ranks must be positive.")
                rrf_scores[chunk_id] += 1.0 / (self.k + rank)
                ranks[chunk_id][source] = rank
                if "score" in result:
                    scores[chunk_id][source] = float(result["score"])
                docs.setdefault(chunk_id, dict(result))
        ranked_ids = sorted(rrf_scores, key=lambda cid: (-rrf_scores[cid], min(ranks[cid].values()), cid))
        output = []
        for fused_rank, cid in enumerate(ranked_ids, start=1):
            result = dict(docs[cid])
            result.update({
                "rank": fused_rank, "score": float(rrf_scores[cid]),
                "rrf_score": float(rrf_scores[cid]), "retrieval_method": "rrf",
                "component_ranks": dict(ranks[cid]), "component_scores": dict(scores[cid]),
            })
            if "dense" in ranks[cid]: result["dense_rank"] = ranks[cid]["dense"]
            if "sparse" in ranks[cid]: result["sparse_rank"] = ranks[cid]["sparse"]
            if "dense" in scores[cid]: result["dense_score"] = scores[cid]["dense"]
            if "sparse" in scores[cid]: result["sparse_score"] = scores[cid]["sparse"]
            output.append(result)
        return output
