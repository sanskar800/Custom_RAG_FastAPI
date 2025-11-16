from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any

from app.services.redis_memory import redis_memory_service
from app.services.rag import rag_service
from app.services.booking_engine import booking_engine

router = APIRouter()

class ChatRequest(BaseModel):
    """Request model for chat endpoint."""
    session_id: str = Field(..., description="Unique session identifier")
    message: str = Field(..., description="User's message")
    top_k: Optional[int] = Field(5, description="Number of chunks to retrieve for RAG")

class ChatResponse(BaseModel):
    """Response model for chat endpoint."""
    session_id: str
    user_message: str
    assistant_message: str
    is_booking_flow: bool
    booking_complete: bool
    booking_data: Optional[Dict[str, Any]] = None

@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Conversational RAG endpoint with integrated booking flow.
    
    This endpoint handles:
    1. Multi-turn conversations with Redis memory
    2. Semantic search and RAG-based responses
    3. Booking intent detection
    4. Interview booking flow (name → email → date → time)
    5. Saving booking information to MongoDB
    
    The booking flow is state-driven and integrated within this endpoint.
    
    Args:
        request: Chat request with session_id and message
        
    Returns:
        Chat response with assistant's message and booking status
    """
    try:
        session_id = request.session_id
        user_message = request.message.strip()
        
        if not user_message:
            raise HTTPException(
                status_code=400,
                detail="Message cannot be empty"
            )
        
        # Add user message to history
        redis_memory_service.add_message(session_id, "user", user_message)
        
        # Check if currently in booking flow
        booking_state = redis_memory_service.get_booking_state(session_id)
        in_booking_flow = bool(booking_state)
        
        # Handle booking flow
        if in_booking_flow:
            try:
                # Process booking input
                response_message, booking_data, is_complete = booking_engine.process_booking_input(
                    session_id=session_id,
                    user_input=user_message,
                    current_state=booking_state
                )
                
                if is_complete:
                    # Booking completed
                    redis_memory_service.clear_booking_state(session_id)
                    redis_memory_service.add_message(session_id, "assistant", response_message)
                    
                    return ChatResponse(
                        session_id=session_id,
                        user_message=user_message,
                        assistant_message=response_message,
                        is_booking_flow=True,
                        booking_complete=True,
                        booking_data=booking_data
                    )
                else:
                    # Continue booking flow
                    redis_memory_service.set_booking_state(session_id, booking_state)
                    redis_memory_service.add_message(session_id, "assistant", response_message)
                    
                    return ChatResponse(
                        session_id=session_id,
                        user_message=user_message,
                        assistant_message=response_message,
                        is_booking_flow=True,
                        booking_complete=False,
                        booking_data=None
                    )
            except Exception as e:
                # If booking fails, clear state and return error
                redis_memory_service.clear_booking_state(session_id)
                error_msg = f"Booking process error: {str(e)}. Please start over."
                redis_memory_service.add_message(session_id, "assistant", error_msg)
                return ChatResponse(
                    session_id=session_id,
                    user_message=user_message,
                    assistant_message=error_msg,
                    is_booking_flow=False,
                    booking_complete=False,
                    booking_data=None
                )
        
        # Check for booking intent
        try:
            if booking_engine.detect_booking_intent(user_message):
                # Start booking flow
                response_message = booking_engine.start_booking()
                
                # Initialize booking state
                initial_state = {"step": "name"}
                redis_memory_service.set_booking_state(session_id, initial_state)
                redis_memory_service.add_message(session_id, "assistant", response_message)
                
                return ChatResponse(
                    session_id=session_id,
                    user_message=user_message,
                    assistant_message=response_message,
                    is_booking_flow=True,
                    booking_complete=False,
                    booking_data=None
                )
        except Exception as e:
            print(f"Booking intent detection error: {e}")
            # Continue to regular RAG if intent detection fails
        
        # Regular RAG flow
        try:
            answer = rag_service.process_query(
                session_id=session_id,
                query=user_message,
                top_k=request.top_k
            )
        except Exception as e:
            answer = f"I encountered an error while processing your question: {str(e)}"
        
        # Add assistant response to history
        redis_memory_service.add_message(session_id, "assistant", answer)
        
        return ChatResponse(
            session_id=session_id,
            user_message=user_message,
            assistant_message=answer,
            is_booking_flow=False,
            booking_complete=False,
            booking_data=None
        )
    
    except HTTPException:
        raise
    except Exception as e:
        # Log the full error for debugging
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Chat processing failed: {str(e)}"
        )

@router.get("/chat/history/{session_id}")
async def get_chat_history(session_id: str, limit: int = 50):
    """
    Retrieve conversation history for a session.
    
    Args:
        session_id: Session identifier
        limit: Maximum number of messages to retrieve
        
    Returns:
        List of conversation messages
    """
    try:
        history = redis_memory_service.get_history(session_id, limit=limit)
        return {
            "session_id": session_id,
            "message_count": len(history),
            "messages": history
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve history: {str(e)}"
        )

@router.delete("/chat/session/{session_id}")
async def clear_session(session_id: str):
    """
    Clear all data for a session (conversation history and booking state).
    
    Args:
        session_id: Session identifier
        
    Returns:
        Success confirmation
    """
    try:
        redis_memory_service.clear_session(session_id)
        return {
            "success": True,
            "message": f"Session {session_id} cleared successfully"
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to clear session: {str(e)}"
        )