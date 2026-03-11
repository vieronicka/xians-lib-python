"""Tests for Xians Server API contracts and client methods."""

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from pydantic import ValidationError

from xians.exceptions.v1.errors import XiansServerError
from xians.interfaces.v1.xians_client import XiansServerClient
from xians.models.v1.configs import XiansServerConfig
from xians.models.v1.server_contracts import (
    ActivityDefinitionRequest,
    ChatOrDataRequest,
    FlowDefinitionRequest,
    HandoffRequest,
    ParameterDefinition,
    UsageReportRequest,
)
from pydantic import SecretStr


@pytest.fixture
def sample_server_config() -> XiansServerConfig:
    """Provide a sample server configuration."""
    return XiansServerConfig(
        server_url="https://api.xians.ai",
        auth_mode="bearer_cert",
        bearer_cert_base64=SecretStr("eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0"),
    )


@pytest.fixture
async def xians_client(sample_server_config: XiansServerConfig) -> XiansServerClient:
    """Provide a XiansServerClient instance."""
    client = XiansServerClient(sample_server_config)
    yield client
    await client.close()


# ========== Server Contract Models Tests ==========


class TestParameterDefinition:
    """Test ParameterDefinition model."""

    def test_valid_parameter_definition(self) -> None:
        """Test creating a valid parameter definition."""
        param = ParameterDefinition(
            name="input_text",
            type="string",
        )
        assert param.name == "input_text"
        assert param.type == "string"

    def test_parameter_definition_camelcase_serialization(self) -> None:
        """Test camelCase serialization."""
        param = ParameterDefinition(
            name="my_param",
            type="integer",
        )
        data = param.model_dump(by_alias=True)
        assert data["name"] == "my_param"
        assert data["type"] == "integer"

    def test_parameter_definition_empty_name_validation(self) -> None:
        """Test that empty name raises validation error."""
        with pytest.raises(ValidationError) as exc_info:
            ParameterDefinition(name="", type="string")
        assert "empty" in str(exc_info.value).lower()

    def test_parameter_definition_empty_type_validation(self) -> None:
        """Test that empty type raises validation error."""
        with pytest.raises(ValidationError) as exc_info:
            ParameterDefinition(name="param", type="")
        assert "empty" in str(exc_info.value).lower()


class TestActivityDefinitionRequest:
    """Test ActivityDefinitionRequest model."""

    def test_minimal_activity_definition(self) -> None:
        """Test minimal valid activity definition."""
        activity = ActivityDefinitionRequest(
            activity_name="process_text",
        )
        assert activity.activity_name == "process_text"
        assert activity.knowledge_ids == []
        assert activity.parameter_definitions == []
        assert activity.agent_tool_names is None

    def test_activity_definition_with_parameters(self) -> None:
        """Test activity definition with parameters."""
        param = ParameterDefinition(name="input", type="string")
        activity = ActivityDefinitionRequest(
            activity_name="process",
            parameter_definitions=[param],
            knowledge_ids=["kb1", "kb2"],
        )
        assert len(activity.parameter_definitions) == 1
        assert len(activity.knowledge_ids) == 2

    def test_activity_definition_camelcase_serialization(self) -> None:
        """Test camelCase serialization of activity definition."""
        activity = ActivityDefinitionRequest(
            activity_name="my_activity",
            agent_tool_names=["tool1", "tool2"],
        )
        data = activity.model_dump(by_alias=True)
        assert data["activityName"] == "my_activity"
        assert data["agentToolNames"] == ["tool1", "tool2"]
        assert data["knowledgeIds"] == []
        assert data["parameterDefinitions"] == []

    def test_activity_definition_empty_name_validation(self) -> None:
        """Test that empty activity name raises validation error."""
        with pytest.raises(ValidationError) as exc_info:
            ActivityDefinitionRequest(activity_name="")
        assert "empty" in str(exc_info.value).lower()


class TestFlowDefinitionRequest:
    """Test FlowDefinitionRequest model."""

    def test_minimal_valid_flow_definition(self) -> None:
        """Test minimal valid flow definition."""
        param = ParameterDefinition(name="param1", type="string")
        activity = ActivityDefinitionRequest(activity_name="activity1")

        flow = FlowDefinitionRequest(
            agent="MyAgent",
            workflow_type="Conversational",
            activity_definitions=[activity],
            parameter_definitions=[param],
        )

        assert flow.agent == "MyAgent"
        assert flow.workflow_type == "Conversational"
        assert len(flow.activity_definitions) == 1
        assert len(flow.parameter_definitions) == 1
        assert flow.system_scoped is False

    def test_flow_definition_camelcase_serialization(self) -> None:
        """Test camelCase serialization."""
        param = ParameterDefinition(name="p1", type="string")
        activity = ActivityDefinitionRequest(activity_name="a1")

        flow = FlowDefinitionRequest(
            agent="Agent1",
            workflow_type="Type1",
            activity_definitions=[activity],
            parameter_definitions=[param],
            system_scoped=True,
            name="FlowName",
        )

        data = flow.model_dump(by_alias=True, exclude_none=True)
        assert data["agent"] == "Agent1"
        assert data["workflowType"] == "Type1"
        assert data["systemScoped"] is True
        assert data["name"] == "FlowName"
        assert "activityDefinitions" in data
        assert "parameterDefinitions" in data

    def test_flow_definition_missing_agent_validation(self) -> None:
        """Test that empty agent raises validation error."""
        param = ParameterDefinition(name="p1", type="string")
        activity = ActivityDefinitionRequest(activity_name="a1")

        with pytest.raises(ValidationError) as exc_info:
            FlowDefinitionRequest(
                agent="",
                workflow_type="Type1",
                activity_definitions=[activity],
                parameter_definitions=[param],
            )
        assert "empty" in str(exc_info.value).lower()

    def test_flow_definition_empty_activities_allowed(self) -> None:
        """Empty activity_definitions is allowed for built-in Conversational flow."""
        param = ParameterDefinition(name="p1", type="string")
        req = FlowDefinitionRequest(
            agent="Agent1",
            workflow_type="Type1",
            activity_definitions=[],
            parameter_definitions=[param],
        )
        assert req.activity_definitions == []

    def test_flow_definition_empty_parameters_allowed(self) -> None:
        """Empty parameter_definitions is allowed for built-in Conversational flow."""
        activity = ActivityDefinitionRequest(activity_name="a1")
        req = FlowDefinitionRequest(
            agent="Agent1",
            workflow_type="Type1",
            activity_definitions=[activity],
            parameter_definitions=[],
        )
        assert req.parameter_definitions == []


class TestChatOrDataRequest:
    """Test ChatOrDataRequest model."""

    def test_minimal_chat_request(self) -> None:
        """Test minimal valid chat request."""
        req = ChatOrDataRequest(participant_id="user123")
        assert req.participant_id == "user123"
        assert req.text is None
        assert req.data is None

    def test_chat_request_with_text_and_data(self) -> None:
        """Test chat request with text and data."""
        req = ChatOrDataRequest(
            participant_id="user123",
            text="Hello",
            data={"key": "value"},
        )
        assert req.participant_id == "user123"
        assert req.text == "Hello"
        assert req.data == {"key": "value"}

    def test_chat_request_camelcase_serialization(self) -> None:
        """Test camelCase serialization."""
        req = ChatOrDataRequest(
            participant_id="user1",
            workflow_id="wf1",
            workflow_type="Conversational",
            request_id="req1",
        )
        data = req.model_dump(by_alias=True, exclude_none=True)
        assert data["participantId"] == "user1"
        assert data["workflowId"] == "wf1"
        assert data["workflowType"] == "Conversational"
        assert data["requestId"] == "req1"

    def test_chat_request_empty_participant_validation(self) -> None:
        """Test that empty participant ID raises validation error."""
        with pytest.raises(ValidationError) as exc_info:
            ChatOrDataRequest(participant_id="")
        assert "empty" in str(exc_info.value).lower()


class TestHandoffRequest:
    """Test HandoffRequest model."""

    def test_minimal_handoff_request(self) -> None:
        """Test minimal valid handoff request."""
        req = HandoffRequest(participant_id="user123")
        assert req.participant_id == "user123"
        assert req.target is None

    def test_handoff_request_with_target_and_reason(self) -> None:
        """Test handoff request with target and reason."""
        req = HandoffRequest(
            participant_id="user123",
            target="agent_name",
            reason="User requested handoff",
        )
        assert req.participant_id == "user123"
        assert req.target == "agent_name"
        assert req.reason == "User requested handoff"

    def test_handoff_request_camelcase_serialization(self) -> None:
        """Test camelCase serialization."""
        req = HandoffRequest(
            participant_id="user1",
            workflow_id="wf1",
            request_id="req1",
            target="human",
            reason="Escalation",
        )
        data = req.model_dump(by_alias=True, exclude_none=True)
        assert data["participantId"] == "user1"
        assert data["workflowId"] == "wf1"
        assert data["requestId"] == "req1"
        assert data["target"] == "human"
        assert data["reason"] == "Escalation"


class TestUsageReportRequest:
    """Test UsageReportRequest model."""

    def test_minimal_usage_report(self) -> None:
        """Test minimal valid usage report."""
        req = UsageReportRequest(
            prompt_tokens=10,
            completion_tokens=20,
            total_tokens=30,
            message_count=1,
        )
        assert req.prompt_tokens == 10
        assert req.completion_tokens == 20
        assert req.total_tokens == 30
        assert req.message_count == 1

    def test_usage_report_with_optional_fields(self) -> None:
        """Test usage report with optional fields."""
        req = UsageReportRequest(
            model="gpt-4",
            workflow_id="wf1",
            request_id="req1",
            source="openai",
            prompt_tokens=10,
            completion_tokens=20,
            total_tokens=30,
            message_count=1,
            response_time_ms=500,
            metadata={"version": "1.0"},
        )
        assert req.model == "gpt-4"
        assert req.workflow_id == "wf1"
        assert req.response_time_ms == 500

    def test_usage_report_camelcase_serialization(self) -> None:
        """Test camelCase serialization."""
        req = UsageReportRequest(
            prompt_tokens=10,
            completion_tokens=20,
            total_tokens=30,
            message_count=1,
            response_time_ms=500,
        )
        data = req.model_dump(by_alias=True, exclude_none=True)
        assert data["promptTokens"] == 10
        assert data["completionTokens"] == 20
        assert data["totalTokens"] == 30
        assert data["messageCount"] == 1
        assert data["responseTimeMs"] == 500

    def test_usage_report_negative_tokens_validation(self) -> None:
        """Test that negative token counts raise validation error."""
        with pytest.raises(ValidationError) as exc_info:
            UsageReportRequest(
                prompt_tokens=-1,
                completion_tokens=20,
                total_tokens=30,
                message_count=1,
            )
        assert "must be >= 0" in str(exc_info.value)

    def test_usage_report_invalid_token_type_validation(self) -> None:
        """Test that non-integer token counts raise validation error."""
        with pytest.raises(ValidationError) as exc_info:
            UsageReportRequest(
                prompt_tokens="10",
                completion_tokens=20,
                total_tokens=30,
                message_count=1,
            )
        assert "Expected integer" in str(exc_info.value)


# ========== Client Method Tests ==========


class TestXiansServerClientDefinitions:
    """Test flow definition upload methods."""

    @pytest.mark.asyncio
    async def test_upload_flow_definition_success(
        self, xians_client: XiansServerClient
    ) -> None:
        """Test successful flow definition upload."""
        param = ParameterDefinition(name="p1", type="string")
        activity = ActivityDefinitionRequest(activity_name="a1")
        flow_def = FlowDefinitionRequest(
            agent="MyAgent",
            workflow_type="Conversational",
            activity_definitions=[activity],
            parameter_definitions=[param],
        )

        # Mock the _request method
        mock_response = MagicMock()
        mock_response.json.return_value = {"status": "success"}
        xians_client._request = AsyncMock(return_value=mock_response)

        result = await xians_client.upload_flow_definition(flow_def)

        assert result == {"status": "success"}
        xians_client._request.assert_called_once()
        call_args = xians_client._request.call_args
        assert call_args[0][0] == "POST"
        assert call_args[0][1] == "/api/agent/definitions"

    @pytest.mark.asyncio
    async def test_upload_flow_definition_400_error(
        self, xians_client: XiansServerClient
    ) -> None:
        """Test flow definition upload with 400 Bad Request."""
        param = ParameterDefinition(name="p1", type="string")
        activity = ActivityDefinitionRequest(activity_name="a1")
        flow_def = FlowDefinitionRequest(
            agent="MyAgent",
            workflow_type="Conversational",
            activity_definitions=[activity],
            parameter_definitions=[param],
        )

        # Mock _request to raise XiansServerError
        error = XiansServerError(
            "Bad Request",
            status_code=400,
            response_body="Invalid payload",
            method="POST",
            url="/api/agent/definitions",
        )
        xians_client._request = AsyncMock(side_effect=error)

        with pytest.raises(XiansServerError) as exc_info:
            await xians_client.upload_flow_definition(flow_def)

        assert exc_info.value.status_code == 400


class TestXiansServerClientConversation:
    """Test conversation outbound endpoints."""

    @pytest.mark.asyncio
    async def test_send_outbound_chat(self, xians_client: XiansServerClient) -> None:
        """Test sending outbound chat."""
        req = ChatOrDataRequest(
            participant_id="user1",
            text="Hello",
        )

        mock_response = MagicMock()
        mock_response.json.return_value = {"status": "sent"}
        xians_client._request = AsyncMock(return_value=mock_response)

        result = await xians_client.send_outbound_chat(req)

        assert result == {"status": "sent"}
        call_args = xians_client._request.call_args
        assert call_args[0][1] == "/api/agent/conversation/outbound/chat"

    @pytest.mark.asyncio
    async def test_send_outbound_data(self, xians_client: XiansServerClient) -> None:
        """Test sending outbound data."""
        req = ChatOrDataRequest(
            participant_id="user1",
            data={"key": "value"},
        )

        mock_response = MagicMock()
        mock_response.json.return_value = {"status": "sent"}
        xians_client._request = AsyncMock(return_value=mock_response)

        result = await xians_client.send_outbound_data(req)

        assert result == {"status": "sent"}
        call_args = xians_client._request.call_args
        assert call_args[0][1] == "/api/agent/conversation/outbound/data"

    @pytest.mark.asyncio
    async def test_send_outbound_webhook(self, xians_client: XiansServerClient) -> None:
        """Test sending outbound webhook."""
        req = ChatOrDataRequest(
            participant_id="user1",
            data={"webhook": "payload"},
        )

        mock_response = MagicMock()
        mock_response.json.return_value = {"status": "sent"}
        xians_client._request = AsyncMock(return_value=mock_response)

        result = await xians_client.send_outbound_webhook(req)

        assert result == {"status": "sent"}
        call_args = xians_client._request.call_args
        assert call_args[0][1] == "/api/agent/conversation/outbound/webhook"

    @pytest.mark.asyncio
    async def test_send_handoff(self, xians_client: XiansServerClient) -> None:
        """Test sending handoff."""
        req = HandoffRequest(
            participant_id="user1",
            target="human_agent",
            reason="Escalation needed",
        )

        mock_response = MagicMock()
        mock_response.json.return_value = {"status": "handoff_initiated"}
        xians_client._request = AsyncMock(return_value=mock_response)

        result = await xians_client.send_handoff(req)

        assert result == {"status": "handoff_initiated"}
        call_args = xians_client._request.call_args
        assert call_args[0][1] == "/api/agent/conversation/outbound/handoff"


class TestXiansServerClientUsage:
    """Test usage reporting endpoint."""

    @pytest.mark.asyncio
    async def test_report_usage_success(self, xians_client: XiansServerClient) -> None:
        """Test successful usage reporting."""
        req = UsageReportRequest(
            model="gpt-4",
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
            message_count=1,
        )

        mock_response = MagicMock()
        mock_response.json.return_value = {"status": "recorded"}
        xians_client._request = AsyncMock(return_value=mock_response)

        result = await xians_client.report_usage(req)

        assert result == {"status": "recorded"}
        call_args = xians_client._request.call_args
        assert call_args[0][1] == "/api/agent/usage/report"

    @pytest.mark.asyncio
    async def test_report_usage_with_metadata(self, xians_client: XiansServerClient) -> None:
        """Test usage reporting with metadata."""
        req = UsageReportRequest(
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
            message_count=1,
            metadata={"region": "us-east-1"},
        )

        mock_response = MagicMock()
        mock_response.json.return_value = {"status": "recorded"}
        xians_client._request = AsyncMock(return_value=mock_response)

        result = await xians_client.report_usage(req)

        assert result == {"status": "recorded"}


class TestXiansServerClientKnowledge:
    """Test knowledge endpoints."""

    @pytest.mark.asyncio
    async def test_get_latest_knowledge(self, xians_client: XiansServerClient) -> None:
        """Test getting latest knowledge."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"knowledge": "data"}
        xians_client._request = AsyncMock(return_value=mock_response)

        result = await xians_client.get_latest_knowledge("kb_name", "agent1")

        assert result == {"knowledge": "data"}
        call_args = xians_client._request.call_args
        assert call_args[0][1] == "/api/agent/knowledge/latest"
        assert call_args[1]["params"] == {"name": "kb_name", "agent": "agent1"}

    @pytest.mark.asyncio
    async def test_list_knowledge(self, xians_client: XiansServerClient) -> None:
        """Test listing knowledge."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"items": []}
        xians_client._request = AsyncMock(return_value=mock_response)

        result = await xians_client.list_knowledge("agent1")

        assert result == {"items": []}
        call_args = xians_client._request.call_args
        assert call_args[0][1] == "/api/agent/knowledge/list"

    @pytest.mark.asyncio
    async def test_create_knowledge(self, xians_client: XiansServerClient) -> None:
        """Test creating knowledge."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"id": "kb1"}
        xians_client._request = AsyncMock(return_value=mock_response)

        result = await xians_client.create_knowledge(
            "kb_name", "agent1", "text", "content"
        )

        assert result == {"id": "kb1"}
        call_args = xians_client._request.call_args
        assert call_args[0][1] == "/api/agent/knowledge"

    @pytest.mark.asyncio
    async def test_delete_knowledge(self, xians_client: XiansServerClient) -> None:
        """Test deleting knowledge."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"status": "deleted"}
        xians_client._request = AsyncMock(return_value=mock_response)

        result = await xians_client.delete_knowledge("kb_name", "agent1")

        assert result == {"status": "deleted"}
        call_args = xians_client._request.call_args
        assert call_args[0][1] == "/api/agent/knowledge"


class TestXiansServerClientDocuments:
    """Test document endpoints."""

    @pytest.mark.asyncio
    async def test_save_document(self, xians_client: XiansServerClient) -> None:
        """Test saving a document."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"id": "doc1"}
        xians_client._request = AsyncMock(return_value=mock_response)

        doc = {"title": "Test"}
        result = await xians_client.save_document(doc)

        assert result == {"id": "doc1"}
        call_args = xians_client._request.call_args
        assert call_args[0][1] == "/api/agent/documents/save"

    @pytest.mark.asyncio
    async def test_get_document(self, xians_client: XiansServerClient) -> None:
        """Test getting a document by ID."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"id": "doc1", "title": "Test"}
        xians_client._request = AsyncMock(return_value=mock_response)

        result = await xians_client.get_document("doc1")

        assert result == {"id": "doc1", "title": "Test"}
        call_args = xians_client._request.call_args
        assert call_args[0][1] == "/api/agent/documents/get"

    @pytest.mark.asyncio
    async def test_get_document_by_key(self, xians_client: XiansServerClient) -> None:
        """Test getting a document by key."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"id": "doc1"}
        xians_client._request = AsyncMock(return_value=mock_response)

        result = await xians_client.get_document_by_key("user", "user123")

        assert result == {"id": "doc1"}
        call_args = xians_client._request.call_args
        assert call_args[0][1] == "/api/agent/documents/get-by-key"

    @pytest.mark.asyncio
    async def test_query_documents(self, xians_client: XiansServerClient) -> None:
        """Test querying documents."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"results": []}
        xians_client._request = AsyncMock(return_value=mock_response)

        query = {"field": "value"}
        result = await xians_client.query_documents(query)

        assert result == {"results": []}
        call_args = xians_client._request.call_args
        assert call_args[0][1] == "/api/agent/documents/query"

    @pytest.mark.asyncio
    async def test_delete_document(self, xians_client: XiansServerClient) -> None:
        """Test deleting a document."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"status": "deleted"}
        xians_client._request = AsyncMock(return_value=mock_response)

        result = await xians_client.delete_document("doc1")

        assert result == {"status": "deleted"}
        call_args = xians_client._request.call_args
        assert call_args[0][1] == "/api/agent/documents/delete"

    @pytest.mark.asyncio
    async def test_delete_many_documents(self, xians_client: XiansServerClient) -> None:
        """Test deleting multiple documents."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"status": "deleted"}
        xians_client._request = AsyncMock(return_value=mock_response)

        result = await xians_client.delete_many_documents(["doc1", "doc2"])

        assert result == {"status": "deleted"}
        call_args = xians_client._request.call_args
        assert call_args[0][1] == "/api/agent/documents/delete-many"

    @pytest.mark.asyncio
    async def test_document_exists(self, xians_client: XiansServerClient) -> None:
        """Test checking if document exists."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"exists": True}
        xians_client._request = AsyncMock(return_value=mock_response)

        result = await xians_client.document_exists("doc1")

        assert result == {"exists": True}
        call_args = xians_client._request.call_args
        assert call_args[0][1] == "/api/agent/documents/exists"

