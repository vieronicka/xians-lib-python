"""Xians Server HTTP client implementation for SDK v1.
"""

import json
import logging
from pathlib import Path
from typing import Any

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from ...exceptions.v1.errors import XiansServerError
from ...models.v1.configs import XiansServerConfig
from ...models.v1.entities import AgentDefinition, WorkflowDefinition
from ...models.v1.server_contracts import (
    ChatOrDataRequest,
    CreateAgentRequest,
    FlowDefinitionRequest,
    HandoffRequest,
    UsageReportRequest,
)
from ...utils.v1.hashing import compute_hash
from ...utils.v1.payload_builder import build_workflow_definition_payload

logger = logging.getLogger(__name__)


class XiansServerClient:
    """
    Async HTTP client for Xians Server integration.
    """

    def __init__(
        self,
        config: XiansServerConfig,
        cache_dir: Path | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.config = config
        self.cache_dir = cache_dir or Path.home() / ".xians" / "cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache_file = self.cache_dir / "uploaded_definitions.json"
        self._uploaded_hashes: dict[str, str] = self._load_cache()

        logger.debug(
            "Initializing XiansServerClient",
            extra={"server_url": str(config.server_url)},
        )

        self._client = httpx.AsyncClient(
            base_url=str(config.server_url),
            timeout=config.timeout_seconds,
            verify=config.verify_ssl,
            event_hooks={"request": [self._inject_headers]},
            transport=transport,
        )

    async def _inject_headers(self, request: httpx.Request) -> None:
        if self.config.auth_mode == "bearer_cert" and self.config.bearer_cert_base64:
            request.headers.setdefault(
                "Authorization",
                f"Bearer {self.config.bearer_cert_base64.get_secret_value()}",
            )
        elif self.config.auth_mode == "x_api_key" and self.config.x_api_key:
            request.headers.setdefault("X-API-Key", self.config.x_api_key.get_secret_value())

        if "/api/agent/" in request.url.path and self.config.tenant_id:
            request.headers.setdefault("X-Tenant-Id", self.config.tenant_id)

    async def _request(
        self,
        method: str,
        url: str,
        **kwargs: Any,
    ) -> httpx.Response:
        try:
            response = await self._client.request(method, url, **kwargs)
            response.raise_for_status()
            return response
        except httpx.HTTPStatusError as e:
            error_msg = f"HTTP {e.response.status_code}: {method} {url}"
            raise XiansServerError(
                error_msg,
                status_code=e.response.status_code,
                response_body=e.response.text,
                method=method,
                url=url,
                cause=e,
            ) from e
        except httpx.RequestError as e:
            raise XiansServerError(
                f"Request error: {str(e)}",
                method=method,
                url=url,
                cause=e,
            ) from e

    def _load_cache(self) -> dict[str, str]:
        if self._cache_file.exists():
            try:
                with open(self._cache_file, "r") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load cache file: {e}")
        return {}

    def _save_cache(self) -> None:
        try:
            with open(self._cache_file, "w") as f:
                json.dump(self._uploaded_hashes, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save cache file: {e}")

    @retry(
        retry=retry_if_exception_type((httpx.TimeoutException, httpx.NetworkError)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
    )
    async def fetch_temporal_settings(self) -> dict[str, Any]:
        try:
            response = await self._request("GET", "/api/agent/settings/flowserver")
            data = response.json()
            logger.info("Successfully fetched Temporal settings from Xians Server")
            return data
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 401:
                error_msg = (
                    "Authentication failed (401 Unauthorized). "
                    "Please verify that your server credentials are valid and configured."
                )
                logger.error(error_msg)
            else:
                error_msg = f"Failed to fetch Temporal settings: {e.response.status_code}"

            raise XiansServerError(
                error_msg,
                status_code=e.response.status_code,
                response_body=e.response.text,
                cause=e,
            )
        except Exception as e:
            raise XiansServerError(
                f"Unexpected error fetching Temporal settings: {str(e)}",
                cause=e,
            )

    async def create_agent(self, request: CreateAgentRequest) -> dict[str, Any]:
        """Create agent in platform (POST /api/agent/definitions/agent). Agent appears in Manager UI."""
        try:
            payload = request.model_dump(by_alias=True, exclude_none=True)
            logger.debug("Creating agent: %s", request.agent_name)
            response = await self._request(
                "POST",
                "/api/agent/definitions/agent",
                json=payload,
            )
            result = response.json()
            logger.info("Successfully created agent: %s", request.agent_name)
            return result
        except XiansServerError as e:
            if e.status_code == 400:
                logger.error("Bad Request (400) creating agent. Response: %s", e.response_body)
            raise

    async def upload_flow_definition(
        self,
        definition: FlowDefinitionRequest,
    ) -> dict[str, Any]:
        try:
            payload = definition.model_dump(by_alias=True, exclude_none=True)
            logger.debug(
                f"Uploading flow definition for agent='{definition.agent}', "
                f"workflowType='{definition.workflow_type}'"
            )

            response = await self._request(
                "POST",
                "/api/agent/definitions",
                json=payload,
            )
            result = response.json()
            logger.info(
                f"Successfully uploaded flow definition: agent='{definition.agent}', "
                f"workflowType='{definition.workflow_type}'"
            )
            return result
        except XiansServerError as e:
            if e.status_code == 400:
                logger.error(
                    f"Bad Request (400) uploading flow definition. "
                    f"Server rejected the payload. "
                    f"Agent: {definition.agent}, WorkflowType: {definition.workflow_type}. "
                    f"Response: {e.response_body}"
                )
            raise


    async def send_outbound_chat(self, request: ChatOrDataRequest) -> dict[str, Any]:
        payload = request.model_dump(by_alias=True, exclude_none=True)
        logger.debug(f"Sending outbound chat to participant: {request.participant_id}")

        response = await self._request(
            "POST",
            "/api/agent/conversation/outbound/chat",
            json=payload,
        )
        return response.json()

    async def send_outbound_data(self, request: ChatOrDataRequest) -> dict[str, Any]:
        payload = request.model_dump(by_alias=True, exclude_none=True)
        logger.debug(f"Sending outbound data to participant: {request.participant_id}")

        response = await self._request(
            "POST",
            "/api/agent/conversation/outbound/data",
            json=payload,
        )
        return response.json()

    async def send_outbound_webhook(self, request: ChatOrDataRequest) -> dict[str, Any]:
        payload = request.model_dump(by_alias=True, exclude_none=True)
        logger.debug(f"Sending outbound webhook to participant: {request.participant_id}")

        response = await self._request(
            "POST",
            "/api/agent/conversation/outbound/webhook",
            json=payload,
        )
        return response.json()

    async def send_handoff(self, request: HandoffRequest) -> dict[str, Any]:
        payload = request.model_dump(by_alias=True, exclude_none=True)
        logger.debug(
            f"Sending handoff from participant {request.participant_id} to {request.target}"
        )

        response = await self._request(
            "POST",
            "/api/agent/conversation/outbound/handoff",
            json=payload,
        )
        return response.json()



    async def report_usage(self, request: UsageReportRequest) -> dict[str, Any]:
        if (
            request.prompt_tokens == 0
            and request.completion_tokens == 0
            and request.total_tokens == 0
            and request.message_count == 0
        ):
            logger.warning(
                "Usage report with all zero counts submitted. "
                "At least one counter should be > 0."
            )

        payload = request.model_dump(by_alias=True, exclude_none=True)
        logger.debug(
            f"Reporting usage: promptTokens={request.prompt_tokens}, "
            f"completionTokens={request.completion_tokens}, "
            f"messageCount={request.message_count}"
        )

        response = await self._request(
            "POST",
            "/api/agent/usage/report",
            json=payload,
        )
        return response.json()


    async def get_latest_knowledge(self, name: str, agent: str) -> dict[str, Any]:
        logger.debug(f"Fetching latest knowledge: name={name}, agent={agent}")

        response = await self._request(
            "GET",
            "/api/agent/knowledge/latest",
            params={"name": name, "agent": agent},
        )
        return response.json()

    async def list_knowledge(self, agent: str) -> dict[str, Any]:
        logger.debug(f"Listing knowledge for agent: {agent}")

        response = await self._request(
            "GET",
            "/api/agent/knowledge/list",
            params={"agent": agent},
        )
        return response.json()

    async def create_knowledge(
        self,
        name: str,
        agent: str,
        type: str,
        content: str,
    ) -> dict[str, Any]:
        payload = {
            "name": name,
            "agent": agent,
            "type": type,
            "content": content,
        }
        logger.debug(f"Creating knowledge: name={name}, agent={agent}, type={type}")

        response = await self._request(
            "POST",
            "/api/agent/knowledge",
            json=payload,
        )
        return response.json()

    async def delete_knowledge(self, name: str, agent: str) -> dict[str, Any]:
        logger.debug(f"Deleting knowledge: name={name}, agent={agent}")

        response = await self._request(
            "DELETE",
            "/api/agent/knowledge",
            params={"name": name, "agent": agent},
        )
        return response.json()


    async def save_document(
        self,
        document: dict[str, Any],
        options: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload = {
            "document": document,
        }
        if options is not None:
            payload["options"] = options

        logger.debug("Saving document")

        response = await self._request(
            "POST",
            "/api/agent/documents/save",
            json=payload,
        )
        return response.json()

    async def update_document(
        self,
        document: dict[str, Any],
        options: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload = {
            "document": document,
        }
        if options is not None:
            payload["options"] = options

        logger.debug("Updating document")

        response = await self._request(
            "POST",
            "/api/agent/documents/update",
            json=payload,
        )
        return response.json()

    async def get_document(self, id: str) -> dict[str, Any]:
        payload = {"id": id}
        logger.debug(f"Getting document: id={id}")

        response = await self._request(
            "POST",
            "/api/agent/documents/get",
            json=payload,
        )
        return response.json()

    async def get_document_by_key(self, type: str, key: str) -> dict[str, Any]:
        payload = {
            "type": type,
            "key": key,
        }
        logger.debug(f"Getting document by key: type={type}, key={key}")

        response = await self._request(
            "POST",
            "/api/agent/documents/get-by-key",
            json=payload,
        )
        return response.json()

    async def query_documents(
        self,
        query: dict[str, Any],
        content_type: str | None = None,
    ) -> dict[str, Any]:
        payload = {"query": query}
        if content_type is not None:
            payload["contentType"] = content_type

        logger.debug("Querying documents")

        response = await self._request(
            "POST",
            "/api/agent/documents/query",
            json=payload,
        )
        return response.json()

    async def delete_document(self, id: str) -> dict[str, Any]:
        payload = {"id": id}
        logger.debug(f"Deleting document: id={id}")

        response = await self._request(
            "POST",
            "/api/agent/documents/delete",
            json=payload,
        )
        return response.json()

    async def delete_many_documents(self, ids: list[str]) -> dict[str, Any]:
        payload = {"ids": ids}
        logger.debug(f"Deleting {len(ids)} documents")

        response = await self._request(
            "POST",
            "/api/agent/documents/delete-many",
            json=payload,
        )
        return response.json()

    async def document_exists(self, id: str) -> dict[str, Any]:
        payload = {"id": id}
        logger.debug(f"Checking if document exists: id={id}")

        response = await self._request(
            "POST",
            "/api/agent/documents/exists",
            json=payload,
        )
        return response.json()

    async def close(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> "XiansServerClient":
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.close()


__all__ = ["XiansServerClient"]

