"""Main platform facade for Xians SDK v1."""

import logging
import os
from typing import Callable

from temporalio.client import Client

from ...configs.v1.logging import configure_logging
from ...constants.v1.core import WorkflowType
from ...exceptions.v1.errors import ConfigurationError, TemporalError
from ...middleware.v1 import (
    initialize_middleware,
)
from ...models.v1.configs import TemporalConfig, XiansOptions, XiansServerConfig
from ...models.v1.configs import TemporalTLSConfig
from ...models.v1.entities import AgentDefinition, WorkflowDefinition
from ...temporal_workflows.v1.worker_runner import (
    WorkerHost,
    WorkerRegistry,
    build_server_task_queue_name,
    SERVER_CONVERSATIONAL_ACTIVITY_ALIAS,
)
from ...temporal_workflows.v1.workflows import (
    ConversationWorkflow,
    InvokeAgentWorkflow,
    create_server_conversation_workflow_class,
)
from ..v1.agent_client import AgentClient
from ..v1.xians_client import XiansServerClient

logger = logging.getLogger(__name__)


class AgentRegistration:
    """
    Represents a registered agent with workflow configuration.
    """

    def __init__(
        self,
        platform: "XiansPlatform",
        definition: AgentDefinition,
    ) -> None:
        """
        Initialize agent registration.
        """
        self.platform = platform
        self.definition = definition
        self.workflows: list[WorkflowDefinition] = []

    def _define_workflow(
        self,
        workflow_type: WorkflowType,
        workflow_class: type,
        name: str,
        workers: int,
        activity_func: Callable | None,
    ) -> WorkflowDefinition:
        """
        Internal helper to define a workflow for this agent.
        """
        workflow_def = WorkflowDefinition(
            workflow_type=workflow_type,
            name=name,
            workers=workers,
            agent_key=self.definition.agent_key or self.definition.name,
            activity_name=activity_func.__name__ if activity_func else "execute_agent_activity",
        )

        self.workflows.append(workflow_def)

        if activity_func:
            agent_key = workflow_def.agent_key
            tenant_id = self.platform.options.tenant_id
            task_queue_override = None

            if (
                self.platform._use_server_task_queue
                and workflow_type == WorkflowType.CONVERSATIONAL
            ):
                workflow_type_str = f"{self.definition.name}:{name}"
                # So the Temporal worker sandbox can resolve the workflow when it re-imports the module
                os.environ["XIANS_SERVER_CONVERSATIONAL_WORKFLOW_TYPE"] = workflow_type_str
                workflow_class = create_server_conversation_workflow_class(workflow_type_str)
                task_queue_override = build_server_task_queue_name(
                    workflow_type_str,
                    tenant_id=tenant_id,
                    system_scoped=self.definition.system_scoped,
                )

            self.platform._worker_registry.register(
                agent_key=agent_key,
                workflow_name=workflow_def.name,
                workflow_class=workflow_class,
                activity_func=activity_func,
                tenant_id=tenant_id,
                system_scoped=self.definition.system_scoped,
                workers=workers,
                task_queue_override=task_queue_override,
                activity_alias_names=(
                    [SERVER_CONVERSATIONAL_ACTIVITY_ALIAS]
                    if task_queue_override is not None
                    else None
                ),
            )

        logger.info(f"Defined {workflow_type} workflow '{name}' for agent '{self.definition.name}'")
        return workflow_def

    def define_invoke_workflow(
        self,
        name: str = "InvokeAgent",
        workers: int = 1,
        activity_func: Callable | None = None,
    ) -> WorkflowDefinition:
        """
        Define a task-based (invoke) workflow for this agent.
        """
        from ...constants.v1.core import WorkflowType

        return self._define_workflow(
            workflow_type=WorkflowType.TASK_BASED,
            workflow_class=InvokeAgentWorkflow,
            name=name,
            workers=workers,
            activity_func=activity_func,
        )

    def define_conversation_workflow(
        self,
        name: str = "Conversation",
        workers: int = 1,
        activity_func: Callable | None = None,
    ) -> WorkflowDefinition:
        """
        Define a conversational (long-running) workflow for this agent.
        """
        from ...constants.v1.core import WorkflowType

        return self._define_workflow(
            workflow_type=WorkflowType.CONVERSATIONAL,
            workflow_class=ConversationWorkflow,
            name=name,
            workers=workers,
            activity_func=activity_func,
        )


class AgentRegistry:
    """
    Registry for managing agent definitions and workflows.
    """

    def __init__(self, platform: "XiansPlatform") -> None:
        self.platform = platform
        self._agents: dict[str, AgentRegistration] = {}

    def register(
        self,
        name: str,
        description: str | None = None,
        system_scoped: bool = False,
    ) -> AgentRegistration:
        definition = AgentDefinition(
            name=name,
            description=description,
            system_scoped=system_scoped,
            agent_key=name,  # Use name as key for now
        )

        agent_reg = AgentRegistration(self.platform, definition)
        self._agents[name] = agent_reg

        logger.info(f"Registered agent: {name} (system_scoped={system_scoped})")
        return agent_reg

    def get(self, name: str) -> AgentRegistration | None:
        return self._agents.get(name)

    def all(self) -> list[AgentRegistration]:
        return list(self._agents.values())


class XiansPlatform:
    """
    Main entry point for Xians SDK.
    """

    def __init__(
        self,
        options: XiansOptions,
        xians_client: XiansServerClient,
        temporal_config: TemporalConfig,
    ) -> None:
        """
        Initialize platform (use XiansPlatform.initialize() instead).
        """
        self.options = options
        self.xians_client = xians_client
        self.temporal_config = temporal_config
        self._use_server_task_queue = False

        self._worker_host: WorkerHost | None = None
        self._worker_registry = WorkerRegistry()
        self._temporal_client: Client | None = None

        self.agents = AgentRegistry(self)

    @classmethod
    async def initialize(cls, options: XiansOptions) -> "XiansPlatform":
        """
        Initialize the Xians platform.
        """
        initialize_middleware()
        logger.info("Exception handler middleware initialized")

        configure_logging(
            log_level=options.log_level,
            enable_structured=options.enable_structured_logging,
        )

        logger.info("Initializing Xians Platform...")

        server_config = XiansServerConfig(
            server_url=options.server_url,
            auth_mode=options.server_auth_mode,
            api_key=options.server_api_key,
            x_api_key=options.server_x_api_key,
            tenant_id=options.tenant_id,
        )
        xians_client = XiansServerClient(server_config)

        temporal_config = options.temporal
        fetched_temporal_from_server = temporal_config is None
        if temporal_config is None:
            logger.info("Fetching Temporal settings from Xians Server...")
            try:
                settings = await xians_client.fetch_temporal_settings()
                temporal_config = cls._build_temporal_config_from_settings(settings)
                logger.info(f"Using Temporal at {temporal_config.address}")
                # Debug TLS summary without sensitive contents
                tls = temporal_config.tls
                tls_enabled = bool(tls and (tls.enabled or any([
                    tls.root_ca_pem,
                    tls.root_ca_path,
                    tls.client_cert_pem,
                    tls.client_cert_path,
                    tls.client_key_pem,
                    tls.client_key_path,
                ])))
                root_ca_provided = bool(tls and (tls.root_ca_pem or tls.root_ca_path))
                mtls_provided = bool(tls and ((tls.client_cert_pem or tls.client_cert_path) and (tls.client_key_pem or tls.client_key_path)))
                domain_override = tls.domain if tls else None
                logger.debug(
                    f"Temporal settings: namespace={temporal_config.namespace}, tls_enabled={tls_enabled}, "
                    f"root_ca_provided={root_ca_provided}, mtls_provided={mtls_provided}, domain_override={domain_override}"
                )
            except Exception as e:
                raise ConfigurationError(
                    "Failed to fetch Temporal settings from Xians Server",
                    cause=e,
                )
        else:
            logger.warning(
                "Using a custom Temporal config. For Manager UI and agent registration (Quick Start steps 1-5), "
                "the worker must use the same cluster as the server—pass temporal=None to use flowserver settings."
            )

        platform = cls(options, xians_client, temporal_config)
        platform._use_server_task_queue = fetched_temporal_from_server

        logger.info("Xians Platform initialized successfully")
        return platform

    async def run_all(self) -> None:
        """
        Start all registered workers and run until stopped.
        """
        logger.info("Starting Xians Platform workers...")

        try:

            self._worker_host = WorkerHost(self.temporal_config)
            await self._worker_host.connect()

            self._temporal_client = self._worker_host.client

            await self._upload_definitions()

            task_queues = self._worker_registry.get_all_task_queues()

            if not task_queues:
                logger.warning("No workflows registered. Nothing to run.")
                return

            for task_queue in task_queues:
                workflows = self._worker_registry.get_workflows_for_queue(task_queue)
                activities = self._worker_registry.get_activities_for_queue(task_queue)
                worker_count = self._worker_registry.get_worker_count(task_queue)

                for workflow in workflows:
                    self._worker_host.register_workflow(workflow)
                for activity in activities:
                    self._worker_host.register_activity(activity)

                await self._worker_host.start_workers(
                    task_queues=[task_queue],
                    workflows=workflows,
                    activities=activities,
                    workers_per_queue=worker_count,
                )

            logger.info(f"Started workers for {len(task_queues)} task queues")

            await self._worker_host.run_until_stopped()

        except KeyboardInterrupt:
            logger.info("Received shutdown signal (Ctrl+C)")
            raise

        except Exception as e:
            logger.error(f"Failed to run workers: {str(e)}", exc_info=True)
            raise TemporalError(
                f"Worker execution failed: {str(e)}",
                cause=e,
            )

        finally:
            logger.info("Cleaning up resources...")
            if self._worker_host:
                try:
                    await self._worker_host.shutdown()
                    logger.info("Workers shut down successfully")
                except Exception as cleanup_error:
                    logger.warning(f"Error during worker cleanup: {cleanup_error}")

            try:
                await self.xians_client.close()
                logger.info("Xians client closed successfully")
            except Exception as cleanup_error:
                logger.warning(f"Error closing Xians client: {cleanup_error}")

    async def _upload_definitions(self) -> None:
        """Upload agent and workflow definitions to Xians Server."""
        from ...models.v1.server_contracts import (
            CreateAgentRequest,
            FlowDefinitionRequest,
        )

        logger.info("Uploading definitions to Xians Server...")

        for agent_reg in self.agents.all():
            try:
                agent_key = agent_reg.definition.agent_key or agent_reg.definition.name
                agent_reg.definition.agent_key = agent_key

                # Create agent in platform first (POST /api/agent/definitions/agent)
                try:
                    await self.xians_client.create_agent(
                        CreateAgentRequest(
                            agent_name=agent_key,
                            system_scoped=agent_reg.definition.system_scoped,
                            description=getattr(
                                agent_reg.definition, "description", None
                            ),
                            summary=getattr(agent_reg.definition, "summary", None),
                        )
                    )
                    logger.debug(f"Created agent: {agent_key}")
                except Exception as agent_error:
                    logger.warning(
                        f"Create agent for {agent_key} failed (may already exist): {agent_error}"
                    )

                for workflow_def in agent_reg.workflows:
                    workflow_def.agent_key = agent_key
                    try:
                        # Server expects workflowType = "{AgentName}:{FlowName}"
                        workflow_type_str = f"{agent_key}:{workflow_def.name}"
                        # Built-in Conversational: empty activity/parameter defs allowed
                        flow_def = FlowDefinitionRequest(
                            agent=agent_key,
                            workflow_type=workflow_type_str,
                            name=workflow_def.name,
                            source=getattr(workflow_def, "source", None) or "",
                            activity_definitions=getattr(
                                workflow_def, "activity_definitions", None
                            )
                            or [],
                            parameter_definitions=getattr(
                                workflow_def, "parameter_definitions", None
                            )
                            or [],
                            system_scoped=agent_reg.definition.system_scoped,
                            activable=True,
                        )

                        await self.xians_client.upload_flow_definition(flow_def)
                        logger.debug(f"Uploaded workflow definition: {workflow_def.name}")
                    except Exception as wf_error:
                        logger.warning(
                            f"Failed to upload workflow definition for {workflow_def.name}: {wf_error}"
                        )

            except Exception as e:
                logger.warning(
                    f"Failed to upload definition for agent {agent_reg.definition.name}: {e}",
                    exc_info=True,
                )

        logger.info("Definitions uploaded successfully")

    def client(self) -> AgentClient:
        """
        Get an agent client for invoking workflows.
        """
        if not self._temporal_client:
            raise ConfigurationError(
                "Temporal client not initialized. Call run_all() first or connect manually."
            )

        return AgentClient(self._temporal_client)

    async def connect_temporal(self) -> None:
        """
        Connect to Temporal without starting workers.

        Useful for client-only mode (invoking workflows without hosting workers).
        """
        self._worker_host = WorkerHost(self.temporal_config)
        await self._worker_host.connect()
        self._temporal_client = self._worker_host.client
        logger.info("Connected to Temporal (client mode)")

    async def shutdown(self) -> None:
        """Shutdown platform and cleanup resources."""
        logger.info("Shutting down Xians Platform...")

        errors = []

        try:
            if self._worker_host:
                try:
                    await self._worker_host.shutdown()
                    logger.info("Workers shut down successfully")
                except Exception as e:
                    errors.append(f"Worker shutdown error: {e}")
                    logger.error(f"Error shutting down workers: {e}", exc_info=True)
        finally:
            try:
                await self.xians_client.close()
                logger.info("Xians client closed successfully")
            except Exception as e:
                errors.append(f"Xians client close error: {e}")
                logger.error(f"Error closing Xians client: {e}", exc_info=True)

        if errors:
            logger.warning(f"Shutdown completed with {len(errors)} error(s)")
        else:
            logger.info("Xians Platform shutdown complete")

    @staticmethod
    def _build_temporal_config_from_settings(settings: dict[str, object]) -> TemporalConfig:
        """Build TemporalConfig from server settings with .NET-aligned field names."""


        server_url_override = os.getenv("TEMPORAL_SERVER_URL")
        flow_server_url = str(server_url_override or settings.get("flowServerUrl") or "")
        if not flow_server_url:
            raise ConfigurationError("flowServerUrl missing from Temporal settings")

        flow_server_url = flow_server_url.strip()

        if "://" in flow_server_url:
            _, flow_server_url = flow_server_url.split("://", 1)
        host: str
        port: int

        if ":" in flow_server_url:
            parts = flow_server_url.rsplit(":", 1)
            host = parts[0]
            try:
                port = int(parts[1])
            except (ValueError, IndexError):
                logger.warning(f"Invalid port in flowServerUrl '{flow_server_url}', using default 7233")
                port = 7233
        else:
            host = flow_server_url
            port = 7233

        host = host.strip().rstrip("/")

        if not host:
            raise ConfigurationError(f"Invalid flowServerUrl: '{settings.get('flowServerUrl')}' - could not extract hostname")

        logger.debug(f"Parsed Temporal address: host='{host}', port={port}")

        namespace = str(settings.get("flowServerNamespace") or settings.get("namespace") or "default")

        flow_server_cert_b64 = settings.get("flowServerCertBase64")
        flow_server_key_b64 = settings.get("flowServerPrivateKeyBase64")
        flow_server_ca_pem = settings.get("flowServerRootCaPem")
        flow_server_domain = settings.get("flowServerDomainOverride") or settings.get("flowServerSniDomain")

        tls_enabled = bool(flow_server_cert_b64 or flow_server_key_b64 or flow_server_ca_pem)

        tls_cfg = None
        if tls_enabled:
            tls_cfg = TemporalTLSConfig(
                enabled=True,
                root_ca_pem=str(flow_server_ca_pem) if flow_server_ca_pem else None,
                client_cert_pem=str(flow_server_cert_b64) if flow_server_cert_b64 else None,
                client_key_pem=str(flow_server_key_b64) if flow_server_key_b64 else None,
                domain=str(flow_server_domain) if flow_server_domain else None,
                pem_is_base64=bool(flow_server_cert_b64 or flow_server_key_b64),
            )

        return TemporalConfig(
            address=f"{host}:{port}",
            namespace=namespace,
            task_queue=str(settings.get("task_queue") or "xians-agents"),
            tls=tls_cfg,
            host=host,
            port=port,
            tls_enabled=tls_enabled,
            server_root_ca_cert_base64=flow_server_cert_b64 if flow_server_ca_pem is None else None,
            client_cert_base64=flow_server_cert_b64,
            client_private_key_base64=flow_server_key_b64,
        )


__all__ = [
    "XiansPlatform",
    "AgentRegistry",
    "AgentRegistration",
]

