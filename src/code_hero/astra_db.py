"""AstraDB handling for the strategic framework."""

from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, Any, Optional, List
from pydantic import BaseModel

from .state import Status
from .tools import astra_retriever

router = APIRouter()

class CollectionResponse(BaseModel):
    """Response model for collection operations."""
    collections: List[str]
    status: str
    message: str

@router.get("/collections", response_model=CollectionResponse)
async def list_collections() -> CollectionResponse:
    """List all available collections."""
    try:
        collections = await astra_retriever.list_collections()
        return CollectionResponse(
            collections=collections,
            status="success",
            message=f"Found {len(collections)} collections"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={"status": "error", "message": f"Failed to list collections: {str(e)}"}
        )

class SearchRequest(BaseModel):
    """Request model for search operations."""
    query: str
    collection: str = "strategy_book"
    limit: int = 5

class SearchResponse(BaseModel):
    """Response model for search operations."""
    results: List[Dict[str, Any]]
    status: str
    message: str

@router.post("/search", response_model=SearchResponse)
async def search_documents(request: SearchRequest) -> SearchResponse:
    """Search documents in a collection."""
    try:
        results = await astra_retriever.search(
            query=request.query,
            collection_name=request.collection,
            k=request.limit
        )
        return SearchResponse(
            results=results,
            status="success",
            message=f"Found {len(results)} documents"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={"status": "error", "message": f"Search failed: {str(e)}"}
        )

@router.get("/health")
async def health_check() -> Dict[str, Any]:
    """Check AstraDB connection health."""
    try:
        collections = await astra_retriever.list_collections()
        return {
            "status": "healthy",
            "collections_available": len(collections),
            "initialized": astra_retriever._initialized,
            "message": "AstraDB connection is healthy"
        }
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail={
                "status": "unhealthy",
                "error": str(e),
                "message": "AstraDB connection is not healthy"
            }
        )

# Export router
__all__ = ["router"]
