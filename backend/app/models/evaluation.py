import uuid

from sqlalchemy import Float, ForeignKey, Index, String
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base
from app.models.mixins import TimestampMixin


class EvaluationResultRecord(TimestampMixin, Base):
    """One measured metric value from one evaluation run (section 29-32).

    Deliberately narrow (one row per metric, not one wide row per run)
    so a run can report an arbitrary, evolving set of metrics
    (P@5, faithfulness, hallucination_rate, ...) without a schema change
    every time a new metric is added, and so retrieval-config comparisons
    (section 32: "run and compare at least three chunk/retrieval
    configurations") are just a filter on ``chunking_config``.
    """

    __tablename__ = "evaluation_results"

    id: Mapped[uuid.UUID] = mapped_column(
        PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    run_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    benchmark_item_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    query_id: Mapped[uuid.UUID | None] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("queries.id", ondelete="SET NULL"),
        nullable=True,
    )
    metric_name: Mapped[str] = mapped_column(String(64), nullable=False)
    metric_value: Mapped[float] = mapped_column(Float, nullable=False)
    chunking_config: Mapped[str | None] = mapped_column(String(64), nullable=True)
    mlflow_run_id: Mapped[str | None] = mapped_column(String(64), nullable=True)

    __table_args__ = (
        Index("ix_evaluation_results_run_metric", "run_id", "metric_name"),
    )

    def __repr__(self) -> str:  # pragma: no cover - debugging aid only
        return (
            f"EvaluationResultRecord(run_id={self.run_id!r}, "
            f"metric_name={self.metric_name!r}, value={self.metric_value!r})"
        )
