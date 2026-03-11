"""Temporal worker host and runner for Xians SDK v1."""

import asyncio
import logging
from typing import Any, Callable

from temporalio.client import Client, TLSConfig
from temporalio.worker import Worker
from temporalio import activity

from ...exceptions.v1.errors import TemporalError
from ...models.v1.configs import TemporalConfig, TemporalTLSConfig
from .tls_utils import resolve_cert_bytes, TLSMaterialError

logger = logging.getLogger(__name__)

# Activity name used by Xians Server / .NET when starting conversational workflows.
# Workflow history may contain "ScheduleActivity ProcessAndSendMessage"; the Python worker
# must register an activity under this name to avoid nondeterminism errors on replay.
SERVER_CONVERSATIONAL_ACTIVITY_ALIAS = "ProcessAndSendMessage"


def _make_activity_alias(activity_func: Callable[..., Any], alias_name: str) -> Callable[..., Any]:
    """Return an activity-decorated callable that delegates to activity_func under alias_name.
    Temporal requires @activity.defn; use name= so the worker accepts workflow history that
    schedules the activity by the alias name (e.g. ProcessAndSendMessage).
    """
    @activity.defn(name=alias_name)
    async def _alias(*args: Any, **kwargs: Any) -> Any:
        return await activity_func(*args, **kwargs)
    return _alias


def build_task_queue_name(
    agent_key: str,
    workflow_name: str,
    tenant_id: str | None = None,
    system_scoped: bool = False,
) -> str:
    """
    Build a deterministic task queue name for Temporal routing.
    """
    scope = "system" if system_scoped else "user"
    tenant_part = tenant_id if tenant_id else "default"

    return f"xians-{tenant_part}-{scope}-{agent_key}-{workflow_name}"


def build_server_task_queue_name(
    workflow_type: str,
    tenant_id: str | None = None,
    system_scoped: bool = False,
) -> str:
    """
    Build task queue name to match Xians Server (parity with .NET).
    Tenant-scoped: {tenantId}:{workflowType} (e.g. default:My Conversational Agent:Conversational).
    System-scoped: {workflowType} only.
    """
    if system_scoped:
        return workflow_type
    tenant_part = tenant_id if tenant_id else "default"
    return f"{tenant_part}:{workflow_type}"


class WorkerHost:
    """
    Manages Temporal worker lifecycle.
    """

    def __init__(self, temporal_config: TemporalConfig) -> None:
        self.config = temporal_config
        self.client: Client | None = None
        self.workers: list[Worker] = []
        self._workflows: list[type] = []
        self._activities: list[Callable] = []

    def _build_temporal_tls_config(self, tls_cfg: TemporalTLSConfig | None) -> TLSConfig | None:
        """Build Temporal TLSConfig from SDK TLS configuration.

        Returns None if TLS config is not provided.
        """
        if not tls_cfg:
            return None
        # Determine if TLS should be enabled
        any_material = any(
            [
                tls_cfg.root_ca_pem,
                tls_cfg.root_ca_path,
                tls_cfg.client_cert_pem,
                tls_cfg.client_cert_path,
                tls_cfg.client_key_pem,
                tls_cfg.client_key_path,
            ]
        )
        if not tls_cfg.enabled and not any_material:
            return None

        server_root = resolve_cert_bytes(tls_cfg.root_ca_pem, tls_cfg.root_ca_path, tls_cfg.pem_is_base64)
        client_cert = resolve_cert_bytes(tls_cfg.client_cert_pem, tls_cfg.client_cert_path, tls_cfg.pem_is_base64)
        client_key = resolve_cert_bytes(tls_cfg.client_key_pem, tls_cfg.client_key_path, tls_cfg.pem_is_base64)

        # Validate mTLS pair
        if (client_cert and not client_key) or (client_key and not client_cert):
            raise TemporalError(
                "mTLS requires both client cert and private key. Provide client_cert_* and client_key_*."
            )

        # Construct TLSConfig
        return TLSConfig(
            server_root_ca_cert=server_root,
            client_cert=client_cert,
            client_private_key=client_key,
            domain=tls_cfg.domain,
        )

    async def connect(self) -> None:
        try:
            target = self.config.address
            tls_config = self._build_temporal_tls_config(self.config.tls)

            self.client = await Client.connect(
                target,
                namespace=self.config.namespace,
                tls=tls_config,
            )

            logger.info(
                f"Connected to Temporal server at {target}, namespace: {self.config.namespace}"
            )

        except TLSMaterialError as e:
            raise TemporalError(
                f"Failed to load TLS materials: {e}"
            ) from e
        except Exception as e:
            # Actionable hints based on common TLS errors
            tls = self.config.tls
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

            msg = (
                f"Failed to connect to Temporal {self.config.address} (namespace={self.config.namespace}). "
                f"TLS enabled={tls_enabled}, root_ca_provided={root_ca_provided}, mtls_provided={mtls_provided}, "
                f"domain_override={domain_override}. Underlying error: {e}"
            )
            err_str = str(e)
            if "UnknownIssuer" in err_str or "CERTIFICATE_VERIFY_FAILED" in err_str:
                msg += (
                    " Hint: For private CA, provide TemporalConfig.tls.root_ca_pem or root_ca_path. "
                    "If certificate hostname mismatch, set TemporalConfig.tls.domain. "
                    "If server requires mTLS, set client_cert_* and client_key_*."
                )
            elif "hostname" in err_str or "SNI" in err_str:
                msg += " Hint: Set TemporalConfig.tls.domain to the server certificate hostname."
            elif "dns error" in err_str.lower() or "nodename nor servname" in err_str:
                msg += (
                    f" Hint: DNS resolution failed for '{self.config.address}'. "
                    "Verify the hostname is correct and resolvable. "
                    "Check your network connection and DNS settings. "
                    "For Temporal Cloud, ensure the address is in format: 'namespace.accountid.tmprl.cloud:7233'. "
                    "If the address is behind a proxy or VPN, ensure proper network configuration."
                )

            raise TemporalError(msg, cause=e)

    def register_workflow(self, workflow_class: type) -> None:
        self._workflows.append(workflow_class)
        logger.debug(f"Registered workflow: {workflow_class.__name__}")

    def register_activity(self, activity_func: Callable) -> None:
        self._activities.append(activity_func)
        logger.debug(f"Registered activity: {activity_func.__name__}")

    async def start_worker(
        self,
        task_queue: str,
        workflows: list[type] | None = None,
        activities: list[Callable] | None = None,
        max_concurrent_activities: int = 10,
    ) -> Worker:
        if not self.client:
            raise TemporalError("Must call connect() before starting workers")

        try:
            worker = Worker(
                self.client,
                task_queue=task_queue,
                workflows=workflows or self._workflows,
                activities=activities or self._activities,
                max_concurrent_activities=max_concurrent_activities,
            )

            asyncio.create_task(worker.run())

            self.workers.append(worker)
            logger.info(
                f"Started worker on task queue: {task_queue} "
                f"with {len(workflows or self._workflows)} workflows, "
                f"{len(activities or self._activities)} activities"
            )

            return worker

        except Exception as e:
            raise TemporalError(
                f"Failed to start worker on task queue {task_queue}: {str(e)}",
                cause=e,
            )

    async def start_workers(
        self,
        task_queues: list[str],
        workflows: list[type] | None = None,
        activities: list[Callable] | None = None,
        workers_per_queue: int = 1,
    ) -> None:
        # Temporal does not allow multiple workers with overlapping task types on the same
        # namespace/task queue/build ID within the same process. Enforce a single worker per queue.
        if workers_per_queue > 1:
            logger.warning(
                "Multiple workers per task queue in the same process are not supported. "
                "Creating a single worker per queue. Use max_concurrent_activities to scale activity concurrency, "
                "or run additional workers in separate processes."
            )
        for task_queue in task_queues:
            # Avoid duplicate worker creation for the same queue
            existing = [w for w in self.workers if w.task_queue == task_queue]
            if existing:
                logger.debug(f"Worker for task queue '{task_queue}' already exists; skipping creation.")
                continue
            await self.start_worker(
                task_queue=task_queue,
                workflows=workflows,
                activities=activities,
            )
            logger.debug(f"Started worker 1/1 for queue {task_queue}")

    async def shutdown(self) -> None:
        logger.info(f"Shutting down {len(self.workers)} workers...")

        for worker in self.workers:
            try:
                await worker.shutdown()
            except Exception as e:
                logger.warning(f"Error shutting down worker: {e}")

        self.workers.clear()
        logger.info("All workers shut down")

    async def run_until_stopped(self) -> None:
        if not self.workers:
            raise TemporalError("No workers started. Call start_worker() first.")

        logger.info(f"Running {len(self.workers)} workers. Press Ctrl+C to stop.")

        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            logger.info("Received cancellation signal")
        finally:
            await self.shutdown()


class WorkerRegistry:

    def __init__(self) -> None:
        """Initialize empty registry."""
        self._registrations: dict[str, dict[str, Any]] = {}

    def register(
        self,
        agent_key: str,
        workflow_name: str,
        workflow_class: type,
        activity_func: Callable,
        tenant_id: str | None = None,
        system_scoped: bool = False,
        workers: int = 1,
        task_queue_override: str | None = None,
        activity_alias_names: list[str] | None = None,
    ) -> str:
        task_queue = (
            task_queue_override
            if task_queue_override is not None
            else build_task_queue_name(
                agent_key=agent_key,
                workflow_name=workflow_name,
                tenant_id=tenant_id,
                system_scoped=system_scoped,
            )
        )

        self._registrations[task_queue] = {
            "agent_key": agent_key,
            "workflow_name": workflow_name,
            "workflow_class": workflow_class,
            "activity_func": activity_func,
            "workers": workers,
            "activity_alias_names": activity_alias_names or [],
        }

        logger.debug(f"Registered {workflow_name} for {agent_key} on queue {task_queue}")
        return task_queue

    def get_all_task_queues(self) -> list[str]:
        return list(self._registrations.keys())

    def get_workflows_for_queue(self, task_queue: str) -> list[type]:
        reg = self._registrations.get(task_queue)
        return [reg["workflow_class"]] if reg else []

    def get_activities_for_queue(self, task_queue: str) -> list[Callable]:
        reg = self._registrations.get(task_queue)
        if not reg:
            return []
        activities = [reg["activity_func"]]
        for alias_name in reg.get("activity_alias_names") or []:
            activities.append(_make_activity_alias(reg["activity_func"], alias_name))
        return activities

    def get_worker_count(self, task_queue: str) -> int:
        reg = self._registrations.get(task_queue)
        return reg["workers"] if reg else 1


__all__ = [
    "WorkerHost",
    "WorkerRegistry",
    "build_task_queue_name",
    "build_server_task_queue_name",
    "SERVER_CONVERSATIONAL_ACTIVITY_ALIAS",
]

