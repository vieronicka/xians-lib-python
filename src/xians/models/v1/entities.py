"""Entity models for Xians SDK v1."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, field_validator

from ...constants.v1.core import MessageRole, WorkflowType


class LLMMessage(BaseModel):
    """
    Represents a single message in an LLM conversation.

    Attributes:
        role: The role of the message sender (system, user, assistant, etc.).
        content: The text content of the message.
        name: Optional name identifier for the message sender.
        metadata: Additional metadata for the message.
    """

    role: MessageRole = Field(description="Role of the message sender")
    content: str = Field(description="Text content of the message")
    name: str | None = Field(default=None, description="Optional name identifier")
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata",
    )

    model_config = {"frozen": False}


class LLMResponse(BaseModel):
    """
    Represents a response from an LLM provider.

    Attributes:
        text: The generated text response.
        model: The model identifier that generated the response.
        finish_reason: The reason the generation finished (e.g., 'stop', 'length').
        usage: Token usage statistics for the request.
        metadata: Additional provider-specific metadata.
    """

    text: str = Field(description="Generated text response")
    model: str = Field(description="Model identifier")
    finish_reason: str = Field(description="Reason generation finished")
    usage: dict[str, int] = Field(
        default_factory=dict,
        description="Token usage statistics",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata",
    )

    model_config = {"frozen": False}


class AgentDefinition(BaseModel):
    """
    Defines an agent in the Xians platform.

    Attributes:
        name: Unique name for the agent (must be non-empty).
        description: Optional description of the agent's purpose.
        system_scoped: Whether the agent is system-scoped or user-scoped.
        metadata: Additional metadata for the agent.
        agent_key: Generated unique key (name-based if not provided).
        version: Agent version for tracking updates.
        hash: Content hash for idempotent uploads.
    """

    name: str = Field(min_length=1, description="Unique agent name")
    description: str | None = Field(default=None, description="Agent description")
    system_scoped: bool = Field(
        default=False,
        description="Whether agent is system-scoped",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata",
    )
    agent_key: str | None = Field(default=None, description="Generated unique key")
    version: str = Field(default="1.0.0", description="Agent version")
    hash: str | None = Field(default=None, description="Content hash for idempotency")

    model_config = {"frozen": False}

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Validate agent name is non-empty after stripping whitespace."""
        if not v.strip():
            raise ValueError("Agent name cannot be empty or whitespace")
        return v


class WorkflowDefinition(BaseModel):
    """
    Defines a workflow configuration.

    Attributes:
        workflow_type: The type of workflow (Conversational, TaskBased, etc.).
        name: Name for the workflow instance.
        workers: Number of worker instances (must be >= 1).
        metadata: Additional workflow configuration metadata.
        agent_key: Associated agent key.
        task_queue: Temporal task queue name.
        activity_name: Name of the activity to invoke.
        version: Workflow version.
        hash: Content hash for idempotency.
    """

    workflow_type: WorkflowType = Field(description="Type of workflow")
    name: str = Field(description="Workflow instance name")
    workers: int = Field(default=1, description="Number of worker instances", ge=1)
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional workflow configuration",
    )
    agent_key: str | None = Field(default=None, description="Associated agent key")
    task_queue: str | None = Field(default=None, description="Temporal task queue name")
    activity_name: str = Field(
        default="execute_agent_activity",
        description="Activity function name",
    )
    version: str = Field(default="1.0.0", description="Workflow version")
    hash: str | None = Field(default=None, description="Content hash for idempotency")

    model_config = {"frozen": False}


class ChatMessageContext(BaseModel):
    """
    Context provided to chat message handlers.

    Attributes:
        message: The incoming message.
        conversation_id: Unique identifier for the conversation.
        user_id: Identifier for the user sending the message.
        metadata: Additional context metadata.
    """

    message: LLMMessage = Field(description="The incoming message")
    conversation_id: str = Field(description="Conversation identifier")
    user_id: str | None = Field(default=None, description="User identifier")
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional context",
    )

    model_config = {"frozen": False}


class AgentRequest(BaseModel):
    """
    Represents a request to execute an agent activity.

    This is the primary input model for agent execution activities.
    It encapsulates all necessary context for the agent to process a request.

    Attributes:
        agent_key: Unique identifier for the agent.
        conversation_id: Optional conversation identifier for stateful sessions.
        message: The input message (string or structured dict).
        metadata: Additional context metadata.
        tenant_id: Optional tenant identifier for multi-tenant scenarios.
        system_scoped: Whether the agent is system-scoped or user-scoped.
        idempotency_key: Optional key for idempotent request processing.
        timestamp: Request timestamp (defaults to current UTC time).
    """

    agent_key: str = Field(min_length=1, description="Unique agent identifier")
    conversation_id: str | None = Field(
        default=None,
        description="Optional conversation identifier",
    )
    message: str | dict[str, Any] = Field(description="Input message")
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional context metadata",
    )
    tenant_id: str | None = Field(default=None, description="Tenant identifier")
    system_scoped: bool = Field(
        default=False,
        description="Whether agent is system-scoped",
    )
    idempotency_key: str | None = Field(
        default=None,
        description="Idempotency key for request processing",
    )
    timestamp: datetime | None = Field(
        default=None,
        description="Request timestamp (set in workflow with workflow.now() or in activity)",
    )

    model_config = {"frozen": False}

    @field_validator("agent_key")
    @classmethod
    def validate_agent_key(cls, v: str) -> str:
        """Validate agent key is non-empty after stripping whitespace."""
        if not v.strip():
            raise ValueError("Agent key cannot be empty or whitespace")
        return v


class AgentResponse(BaseModel):
    """
    Represents a response from an agent activity execution.

    This is the primary output model for agent execution activities.
    It provides flexibility for different response formats.

    Attributes:
        text: Optional text response from the agent.
        payload: Optional structured payload (e.g., tool results, state).
        raw: Optional raw response from underlying framework.
        usage: Optional token/resource usage statistics.
        model: Optional model identifier used for generation.
        metadata: Additional response metadata.
    """

    text: str | None = Field(default=None, description="Text response")
    payload: dict[str, Any] | None = Field(
        default=None,
        description="Structured response payload",
    )
    raw: Any | None = Field(default=None, description="Raw response data")
    usage: dict[str, int] | None = Field(
        default=None,
        description="Token/resource usage statistics",
    )
    model: str | None = Field(default=None, description="Model identifier")
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional response metadata",
    )

    model_config = {"frozen": False}


__all__ = [
    "LLMMessage",
    "LLMResponse",
    "AgentDefinition",
    "WorkflowDefinition",
    "ChatMessageContext",
    "AgentRequest",
    "AgentResponse",
]
