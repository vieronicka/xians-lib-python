"""Pydantic models representing the canonical Xians Server REST API contracts.
"""

from typing import Any

from pydantic import BaseModel, Field, field_validator


class ParameterDefinition(BaseModel):
    """
    Parameter definition model.

    Attributes:
        name: Parameter name (required).
        type: Parameter type string, e.g., 'string', 'integer' (required).
    """

    name: str = Field(
        min_length=1,
        description="Parameter name (required, non-empty)",
        serialization_alias="name",
    )
    type: str = Field(
        min_length=1,
        description="Parameter type (required, non-empty)",
        serialization_alias="type",
    )

    model_config = {"populate_by_name": True}

    @field_validator("name", "type", mode="before")
    @classmethod
    def validate_non_empty_strings(cls, v: Any) -> str:
        """Ensure required string fields are non-empty."""
        if isinstance(v, str):
            v = v.strip()
            if not v:
                raise ValueError("Field cannot be empty or whitespace")
        return v


class ActivityDefinitionRequest(BaseModel):
    """
    Activity definition for flow definitions.

    Attributes:
        activityName: Name of the activity (required, non-empty).
        agentToolNames: List of agent tool names (optional).
        knowledgeIds: List of knowledge IDs (required key, can be empty array).
        parameterDefinitions: List of parameter definitions (required key, can be empty array).
    """

    activity_name: str = Field(
        min_length=1,
        description="Activity name (required, non-empty)",
        serialization_alias="activityName",
    )
    agent_tool_names: list[str] | None = Field(
        default=None,
        description="List of agent tool names (optional)",
        serialization_alias="agentToolNames",
    )
    knowledge_ids: list[str] = Field(
        default_factory=list,
        description="List of knowledge IDs (required key, can be empty)",
        serialization_alias="knowledgeIds",
    )
    parameter_definitions: list[ParameterDefinition] = Field(
        default_factory=list,
        description="List of parameter definitions (required key, can be empty)",
        serialization_alias="parameterDefinitions",
    )

    model_config = {"populate_by_name": True}

    @field_validator("activity_name", mode="before")
    @classmethod
    def validate_activity_name(cls, v: Any) -> str:
        """Ensure activityName is non-empty."""
        if isinstance(v, str):
            v = v.strip()
            if not v:
                raise ValueError("activityName cannot be empty or whitespace")
        return v


class FlowDefinitionRequest(BaseModel):
    """
    Flow definition request for POST /api/agent/definitions.

    This is the canonical model for uploading agent/workflow definitions.

    Attributes:
        agent: Agent name (required, non-empty).
        workflowType: Workflow type identifier (required, non-empty).
        name: Definition name (optional).
        source: Definition source (optional).
        activityDefinitions: List of activity definitions (required key, must have >= 1).
        parameterDefinitions: List of top-level parameter definitions (required key, must have >= 1).
        systemScoped: Whether system-scoped (optional, defaults to false).
        onboardingJson: Onboarding JSON string (optional).
    """

    agent: str = Field(
        min_length=1,
        description="Agent name (required, non-empty)",
        serialization_alias="agent",
    )
    workflow_type: str = Field(
        min_length=1,
        description="Workflow type identifier (required, non-empty)",
        serialization_alias="workflowType",
    )
    name: str | None = Field(
        default=None,
        description="Definition name (optional)",
        serialization_alias="name",
    )
    source: str | None = Field(
        default=None,
        description="Definition source (optional)",
        serialization_alias="source",
    )
    activity_definitions: list[ActivityDefinitionRequest] = Field(
        default_factory=list,
        description="Activity definitions (can be empty for built-in Conversational)",
        serialization_alias="activityDefinitions",
    )
    parameter_definitions: list[ParameterDefinition] = Field(
        default_factory=list,
        description="Top-level parameter definitions (can be empty)",
        serialization_alias="parameterDefinitions",
    )
    system_scoped: bool = Field(
        default=False,
        description="Whether system-scoped (optional)",
        serialization_alias="systemScoped",
    )
    activable: bool = Field(
        default=True,
        description="Whether workflow can be started by the server (optional)",
        serialization_alias="activable",
    )
    onboarding_json: str | None = Field(
        default=None,
        description="Onboarding JSON string (optional)",
        serialization_alias="onboardingJson",
    )

    model_config = {"populate_by_name": True}

    @field_validator("agent", "workflow_type", mode="before")
    @classmethod
    def validate_required_non_empty_strings(cls, v: Any) -> str:
        """Ensure required string fields are non-empty."""
        if isinstance(v, str):
            v = v.strip()
            if not v:
                raise ValueError("Field cannot be empty or whitespace")
        return v


class CreateAgentRequest(BaseModel):
    """
    Request model for POST /api/agent/definitions/agent (create agent in platform).

    Attributes:
        agentName: Display name (required); must match name used in workflowType.
        systemScoped: false = tenant-scoped (Deployed Agents), true = template (system-wide).
        onboardingJson, description, summary, version, author, category: optional.
    """

    agent_name: str = Field(
        min_length=1,
        description="Agent display name (required)",
        serialization_alias="agentName",
    )
    system_scoped: bool = Field(
        default=False,
        description="true = template; false = tenant-scoped",
        serialization_alias="systemScoped",
    )
    onboarding_json: str | None = Field(default=None, serialization_alias="onboardingJson")
    description: str | None = Field(default=None, serialization_alias="description")
    summary: str | None = Field(default=None, serialization_alias="summary")
    version: str | None = Field(default=None, serialization_alias="version")
    author: str | None = Field(default=None, serialization_alias="author")
    category: str | None = Field(default=None, serialization_alias="category")

    model_config = {"populate_by_name": True}


class ChatOrDataRequest(BaseModel):
    """
    Request model for outbound chat, data, and webhook endpoints.

    Used for:
    - POST /api/agent/conversation/outbound/chat
    - POST /api/agent/conversation/outbound/data
    - POST /api/agent/conversation/outbound/webhook

    Attributes:
        participantId: Participant identifier (required, non-empty).
        workflowId: Workflow ID (optional).
        workflowType: Workflow type (optional).
        scope: Scope identifier (optional).
        text: Text content (optional).
        data: Arbitrary JSON data (optional).
        requestId: Request ID for tracking (optional).
        origin: Origin identifier (optional).
    """

    participant_id: str = Field(
        min_length=1,
        description="Participant ID (required, non-empty)",
        serialization_alias="participantId",
    )
    workflow_id: str | None = Field(
        default=None,
        description="Workflow ID (optional)",
        serialization_alias="workflowId",
    )
    workflow_type: str | None = Field(
        default=None,
        description="Workflow type (optional)",
        serialization_alias="workflowType",
    )
    scope: str | None = Field(
        default=None,
        description="Scope (optional)",
        serialization_alias="scope",
    )
    text: str | None = Field(
        default=None,
        description="Text content (optional)",
        serialization_alias="text",
    )
    data: Any = Field(
        default=None,
        description="Arbitrary JSON data (optional)",
        serialization_alias="data",
    )
    request_id: str | None = Field(
        default=None,
        description="Request ID for tracking (optional)",
        serialization_alias="requestId",
    )
    thread_id: str | None = Field(
        default=None,
        description="Thread ID (links reply to same conversation; from signal payload)",
        serialization_alias="threadId",
    )
    origin: str | None = Field(
        default=None,
        description="Origin (optional)",
        serialization_alias="origin",
    )

    model_config = {"populate_by_name": True}

    @field_validator("participant_id", mode="before")
    @classmethod
    def validate_participant_id(cls, v: Any) -> str:
        """Ensure participantId is non-empty."""
        if isinstance(v, str):
            v = v.strip()
            if not v:
                raise ValueError("participantId cannot be empty or whitespace")
        return v


class HandoffRequest(BaseModel):
    """
    Request model for handoff endpoint.

    Used for:
    - POST /api/agent/conversation/outbound/handoff

    Attributes:
        participantId: Participant identifier (required, non-empty).
        workflowId: Workflow ID (optional).
        workflowType: Workflow type (optional).
        scope: Scope (optional).
        text: Text content (optional).
        data: Arbitrary JSON data (optional).
        requestId: Request ID for tracking (optional).
        origin: Origin (optional).
        target: Handoff target/agent (optional).
        reason: Reason for handoff (optional).
    """

    participant_id: str = Field(
        min_length=1,
        description="Participant ID (required, non-empty)",
        serialization_alias="participantId",
    )
    workflow_id: str | None = Field(
        default=None,
        description="Workflow ID (optional)",
        serialization_alias="workflowId",
    )
    workflow_type: str | None = Field(
        default=None,
        description="Workflow type (optional)",
        serialization_alias="workflowType",
    )
    scope: str | None = Field(
        default=None,
        description="Scope (optional)",
        serialization_alias="scope",
    )
    text: str | None = Field(
        default=None,
        description="Text content (optional)",
        serialization_alias="text",
    )
    data: Any = Field(
        default=None,
        description="Arbitrary JSON data (optional)",
        serialization_alias="data",
    )
    request_id: str | None = Field(
        default=None,
        description="Request ID for tracking (optional)",
        serialization_alias="requestId",
    )
    origin: str | None = Field(
        default=None,
        description="Origin (optional)",
        serialization_alias="origin",
    )
    target: str | None = Field(
        default=None,
        description="Handoff target (optional)",
        serialization_alias="target",
    )
    reason: str | None = Field(
        default=None,
        description="Handoff reason (optional)",
        serialization_alias="reason",
    )

    model_config = {"populate_by_name": True}

    @field_validator("participant_id", mode="before")
    @classmethod
    def validate_participant_id(cls, v: Any) -> str:
        """Ensure participantId is non-empty."""
        if isinstance(v, str):
            v = v.strip()
            if not v:
                raise ValueError("participantId cannot be empty or whitespace")
        return v


class UsageReportRequest(BaseModel):
    """
    Request model for usage reporting endpoint.

    Used for:
    - POST /api/agent/usage/report

    Attributes:
        model: LLM model name (optional).
        workflowId: Workflow ID (optional).
        requestId: Request ID for tracking (optional).
        source: Usage source identifier (optional).
        promptTokens: Number of prompt tokens (required, >= 0).
        completionTokens: Number of completion tokens (required, >= 0).
        totalTokens: Total tokens (required, >= 0).
        messageCount: Message count (required, >= 0).
        responseTimeMs: Response time in milliseconds (optional, >= 0).
        metadata: Additional metadata (optional).
    """

    model: str | None = Field(
        default=None,
        description="LLM model name (optional)",
        serialization_alias="model",
    )
    workflow_id: str | None = Field(
        default=None,
        description="Workflow ID (optional)",
        serialization_alias="workflowId",
    )
    request_id: str | None = Field(
        default=None,
        description="Request ID (optional)",
        serialization_alias="requestId",
    )
    source: str | None = Field(
        default=None,
        description="Usage source (optional)",
        serialization_alias="source",
    )
    prompt_tokens: int = Field(
        ge=0,
        description="Prompt tokens (required, >= 0)",
        serialization_alias="promptTokens",
    )
    completion_tokens: int = Field(
        ge=0,
        description="Completion tokens (required, >= 0)",
        serialization_alias="completionTokens",
    )
    total_tokens: int = Field(
        ge=0,
        description="Total tokens (required, >= 0)",
        serialization_alias="totalTokens",
    )
    message_count: int = Field(
        ge=0,
        description="Message count (required, >= 0)",
        serialization_alias="messageCount",
    )
    response_time_ms: int | None = Field(
        default=None,
        ge=0,
        description="Response time in ms (optional, >= 0)",
        serialization_alias="responseTimeMs",
    )
    metadata: dict[str, str] | None = Field(
        default=None,
        description="Additional metadata (optional)",
        serialization_alias="metadata",
    )

    model_config = {"populate_by_name": True}

    @field_validator("prompt_tokens", "completion_tokens", "total_tokens", "message_count", mode="before")
    @classmethod
    def validate_non_negative_integers(cls, v: Any) -> int:
        """Ensure token and count fields are non-negative integers."""
        if not isinstance(v, int):
            raise ValueError(f"Expected integer, got {type(v).__name__}")
        if v < 0:
            raise ValueError(f"Value must be >= 0, got {v}")
        return v


__all__ = [
    "ParameterDefinition",
    "ActivityDefinitionRequest",
    "FlowDefinitionRequest",
    "CreateAgentRequest",
    "ChatOrDataRequest",
    "HandoffRequest",
    "UsageReportRequest",
]

