"""Read boundary for ``GET /queries/{query_id}/evaluation`` (Milestone F,
section 9).

Purely a reader over ``EvaluationResultRecord`` (Milestone C's
evaluation-run schema, see app.models.evaluation) - this milestone does
not add any evaluation *computation*; the metrics it can return are
whatever a real evaluation run (Phase 8/evaluation harness, out of scope
here) already wrote. An empty result is reported as "not run", never
fabricated.
"""

from __future__ import annotations

import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.evaluation import EvaluationResultRecord


class EvaluationRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def get_metrics_for_query(
        self, query_id: uuid.UUID
    ) -> list[EvaluationResultRecord]:
        stmt = select(EvaluationResultRecord).where(
            EvaluationResultRecord.query_id == query_id
        )
        result = await self._session.execute(stmt)
        return list(result.scalars().all())
