"""Analysis Agent public contracts, loaded lazily to avoid NLP import cycles."""

__all__ = ["AnalysisAgent", "AnalysisRequest", "AnalysisResponse"]


def __getattr__(name: str):
    if name == "AnalysisAgent":
        from agents.analysis.agent import AnalysisAgent

        return AnalysisAgent
    if name in {"AnalysisRequest", "AnalysisResponse"}:
        from agents.analysis.schemas import AnalysisRequest, AnalysisResponse

        return {"AnalysisRequest": AnalysisRequest, "AnalysisResponse": AnalysisResponse}[name]
    raise AttributeError(name)
