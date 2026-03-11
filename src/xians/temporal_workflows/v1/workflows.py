"""Built-in Temporal workflows for Xians SDK v1."""

import logging
import os
from datetime import timedelta
from typing import Any, Type

from temporalio import workflow
from temporalio.common import RetryPolicy

from ...models.v1.entities import AgentRequest, AgentResponse
from .failure_unwrap import unwrap_temporal_failure

logger = logging.getLogger(__name__)


# Shared retry policy configuration
_DEFAULT_RETRY_POLICY = RetryPolicy(
    initial_interval=timedelta(seconds=1),
    maximum_interval=timedelta(seconds=30),
    maximum_attempts=3,
    non_retryable_error_types=["ValueError", "ValidationError"],
)

_DEFAULT_ACTIVITY_TIMEOUT = timedelta(minutes=5)


def _create_error_response(error: Exception) -> AgentResponse:
    error_details = unwrap_temporal_failure(error)

    root_type = error_details["root_error_type"]
    root_message = error_details["root_error_message"]

    error_text = f"Agent execution failed ({root_type}): {root_message}"
    metadata = {
        "is_error": True,
        "error": root_message,
        "error_type": root_type,
        "error_details": error_details,
    }

    return AgentResponse(
        text=error_text,
        metadata=metadata,
    )


@workflow.defn
class InvokeAgentWorkflow:

    @workflow.run
    async def run(self, request: AgentRequest) -> AgentResponse:
        workflow.logger.info(
            f"InvokeAgentWorkflow started for agent: {request.agent_key}, "
            f"conversation: {request.conversation_id}"
        )

        activity_name = workflow.memo_value("activity_name", default="execute_agent_activity")

        try:
            response = await workflow.execute_activity(
                activity_name,
                request,
                start_to_close_timeout=_DEFAULT_ACTIVITY_TIMEOUT,
                retry_policy=_DEFAULT_RETRY_POLICY,
            )

            workflow.logger.info(f"InvokeAgentWorkflow completed for agent: {request.agent_key}")
            return response

        except Exception as e:
            workflow.logger.error(
                f"InvokeAgentWorkflow failed for agent {request.agent_key}: {str(e)}"
            )
            return _create_error_response(e)


@workflow.defn
class ConversationWorkflow:
    """Base conversation workflow. Subclasses (e.g. server-aligned) may set _DEFAULT_ACTIVITY_NAME."""

    _DEFAULT_ACTIVITY_NAME = "execute_agent_activity"

    def __init__(self) -> None:
        self._messages: list[dict[str, Any]] = []
        self._session_metadata: dict[str, Any] = {}
        self._activity_name = "execute_agent_activity"

    def _record_message(
        self, message: str, metadata: dict[str, Any], direction: str
    ) -> None:
        self._messages.append(
            {
                "message": message,
                "metadata": metadata,
                "timestamp": workflow.now().isoformat(),
                "direction": direction,
            }
        )

    def _create_agent_request(
        self, message: str, metadata: dict[str, Any]
    ) -> AgentRequest:
        # If workflow was started via SignalWithStart, _session_metadata may not be set yet
        wf_type = metadata.get("workflowType") or metadata.get("workflow_type") or ""
        agent_key = self._session_metadata.get("agent_key") or (wf_type.split(":", 1)[0] if ":" in wf_type else wf_type or "unknown")
        conv_id = (
            self._session_metadata.get("conversation_id")
            or metadata.get("threadId")
            or metadata.get("ThreadId")
            or metadata.get("thread_id")
            or "default"
        )
        if "agent_key" not in self._session_metadata:
            self._session_metadata["agent_key"] = agent_key
        if "conversation_id" not in self._session_metadata:
            self._session_metadata["conversation_id"] = conv_id
        return AgentRequest(
            agent_key=agent_key,
            conversation_id=conv_id,
            message=message,
            metadata=metadata,
            timestamp=workflow.now(),
        )

    @workflow.run
    async def run(
        self, agent_key: str = "", conversation_id: str = ""
    ) -> dict[str, Any]:
        """Run the workflow. Server may start via SignalWithStart with no input, so args are optional."""
        workflow.logger.info(
            f"ConversationWorkflow started for agent: {agent_key or '(from signal)'}, "
            f"conversation: {conversation_id or '(from signal)'}"
        )

        self._session_metadata = {
            "agent_key": agent_key or "",
            "conversation_id": conversation_id or "",
            "started_at": workflow.now().isoformat(),
        }

        self._activity_name = workflow.memo_value(
            "activity_name",
            default=getattr(self.__class__, "_DEFAULT_ACTIVITY_NAME", "execute_agent_activity"),
        )

        await workflow.wait_condition(lambda: False)

        return {
            "conversation_id": self._session_metadata.get("conversation_id", conversation_id or ""),
            "message_count": len(self._messages),
            "metadata": self._session_metadata,
        }

    @workflow.signal
    async def inbound_message(self, message: str, metadata: dict[str, Any] | None = None) -> None:
        workflow.logger.info("Received inbound_message signal for conversation workflow")
        metadata = metadata or {}
        self._record_message(message, metadata, "inbound")
        request = self._create_agent_request(message, metadata)
        try:
            response = await workflow.execute_activity(
                self._activity_name,
                request,
                start_to_close_timeout=_DEFAULT_ACTIVITY_TIMEOUT,
                retry_policy=_DEFAULT_RETRY_POLICY,
            )
            self._record_message(
                response.text or str(response.payload),
                response.metadata,
                "outbound",
            )
        except Exception as e:
            workflow.logger.error("Activity failed in inbound_message: %s", e)
            raise

    @workflow.signal
    async def HandleInboundChatOrData(
        self, message_or_payload: str | dict[str, Any], metadata: dict[str, Any] | None = None
    ) -> None:
        """Handle server Manager UI inbound chat: run activity (activity sends outbound reply).
        Accepts either (message: str, metadata: dict) or a single payload dict with text/message and participantId etc.
        Server may send a nested payload (e.g. under "payload" or "Payload"), or a list [message, meta].
        """
        workflow.logger.info("Received HandleInboundChatOrData signal")
        # Some servers send signal as [message, payload_dict]
        if isinstance(message_or_payload, (list, tuple)) and len(message_or_payload) >= 2:
            _msg = message_or_payload[0]
            _meta = (
                message_or_payload[1]
                if isinstance(message_or_payload[1], dict)
                else (metadata or {})
            )
            message_or_payload = _msg
            metadata = _meta
        if isinstance(message_or_payload, dict):
            payload = message_or_payload
            workflow.logger.info(
                "HandleInboundChatOrData payload (top-level) keys: %s",
                list(payload.keys()),
            )
            # Unwrap nested payload from server (e.g. { "payload": { "threadId": "...", "text": "..." } })
            if "payload" in payload and isinstance(payload["payload"], dict):
                payload = payload["payload"]
                workflow.logger.info("Unwrapped inner payload keys: %s", list(payload.keys()))
            elif "Payload" in payload and isinstance(payload["Payload"], dict):
                payload = payload["Payload"]
                workflow.logger.info("Unwrapped inner Payload keys: %s", list(payload.keys()))
            message = str(payload.get("text") or payload.get("message") or payload.get("Text") or "")
            meta = {k: v for k, v in payload.items() if k not in ("text", "message", "Text")}
        else:
            message = str(message_or_payload)
            meta = metadata or {}
            workflow.logger.info(
                "HandleInboundChatOrData message (str), metadata keys: %s",
                list(meta.keys()) if meta else [],
            )
        self._record_message(message, meta, "inbound")
        request = self._create_agent_request(message, meta)
        try:
            response = await workflow.execute_activity(
                self._activity_name,
                request,
                start_to_close_timeout=_DEFAULT_ACTIVITY_TIMEOUT,
                retry_policy=_DEFAULT_RETRY_POLICY,
            )
            self._record_message(
                response.text or str(response.payload),
                response.metadata,
                "outbound",
            )
        except Exception as e:
            workflow.logger.error("Activity failed in HandleInboundChatOrData: %s", e)
            raise


    @workflow.update
    async def request_response(
        self, message: str, metadata: dict[str, Any] | None = None
    ) -> AgentResponse:
        workflow.logger.info(f"Processing request_response for conversation workflow")

        metadata = metadata or {}
        self._record_message(message, metadata, "inbound")

        request = self._create_agent_request(message, metadata)

        try:
            response = await workflow.execute_activity(
                self._activity_name,
                request,
                start_to_close_timeout=_DEFAULT_ACTIVITY_TIMEOUT,
                retry_policy=_DEFAULT_RETRY_POLICY,
            )

            self._record_message(
                response.text or str(response.payload),
                response.metadata,
                "outbound"
            )

            return response

        except Exception as e:
            workflow.logger.error(f"Agent activity failed: {str(e)}")
            return _create_error_response(e)

    @workflow.query
    def get_session_state(self) -> dict[str, Any]:
        return {
            "metadata": self._session_metadata,
            "message_count": len(self._messages),
            "last_message_at": (
                self._messages[-1]["timestamp"] if self._messages else None
            ),
        }

    @workflow.query
    def get_message_history(self) -> list[dict[str, Any]]:
        return self._messages


def create_server_conversation_workflow_class(workflow_type_name: str) -> Type[ConversationWorkflow]:
    """
    Create a ConversationWorkflow class registered with the server's workflow type.
    Use when integrating with Xians Server Manager UI so the server can route signals.
    workflow_type_name must be {AgentName}:{FlowName} (e.g. 'My Conversational Agent:Conversational').
    Temporal requires the workflow class to be module-level (no "<locals>" in __qualname__).
    """
    # Sanitize for a valid Python identifier
    safe_suffix = (
        workflow_type_name.replace(" ", "_")
        .replace(":", "_")
        .replace("-", "_")
        .replace(".", "_")
    )
    safe_suffix = "".join(c if (c.isalnum() or c == "_") else "_" for c in safe_suffix)
    class_name = f"_ServerConversationWorkflow_{safe_suffix}"

    module_globals = globals()
    if class_name in module_globals:
        return module_globals[class_name]

    name_repr = repr(workflow_type_name)
    # Server workflow history may schedule activity "ProcessAndSendMessage"; use that as default
    # so replay matches and we avoid nondeterminism (worker registers an alias for that name).
    default_activity_repr = repr("ProcessAndSendMessage")
    # Define class at module level via exec so __qualname__ does not contain "<locals>"
    exec(
        f"""
@workflow.defn(name={name_repr})
class {class_name}(ConversationWorkflow):
    _DEFAULT_ACTIVITY_NAME = {default_activity_repr}

    @workflow.run
    async def run(self, agent_key: str = "", conversation_id: str = "") -> dict[str, Any]:
        return await super().run(agent_key, conversation_id)

    @workflow.signal
    async def inbound_message(self, message: str, metadata: dict[str, Any] | None = None) -> None:
        await super().inbound_message(message, metadata)

    @workflow.signal
    async def HandleInboundChatOrData(
        self, message_or_payload: str | dict[str, Any], metadata: dict[str, Any] | None = None
    ) -> None:
        await super().HandleInboundChatOrData(message_or_payload, metadata)

    @workflow.update
    async def request_response(
        self, message: str, metadata: dict[str, Any] | None = None
    ) -> AgentResponse:
        return await super().request_response(message, metadata)

    @workflow.query
    def get_session_state(self) -> dict[str, Any]:
        return super().get_session_state()

    @workflow.query
    def get_message_history(self) -> list[dict[str, Any]]:
        return super().get_message_history()
""",
        module_globals,
    )
    return module_globals[class_name]


# When the Temporal worker sandbox re-imports this module, it uses a fresh sys.modules
# and never ran create_server_conversation_workflow_class. Create the class at import time
# when the env var is set so the sandbox can resolve the workflow by name.
_ENV_WORKFLOW_TYPE = os.environ.get("XIANS_SERVER_CONVERSATIONAL_WORKFLOW_TYPE")
if _ENV_WORKFLOW_TYPE:
    create_server_conversation_workflow_class(_ENV_WORKFLOW_TYPE)


__all__ = ["InvokeAgentWorkflow", "ConversationWorkflow", "create_server_conversation_workflow_class"]

