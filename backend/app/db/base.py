"""Shared SQLAlchemy 2.x declarative base.

Every ORM model in ``app.models`` inherits from this ``Base`` so they all
share one ``MetaData`` object - that's what lets Alembic autogenerate a
single, consistent migration for the whole schema, and what lets
``app.models`` be imported once (see ``app/models/__init__.py``) to fully
populate ``Base.metadata`` before Alembic (or a test) inspects it.

A naming convention is set on the metadata so constraint/index names are
deterministic and readable (``ix_document_chunks_document_id`` rather than
a driver-generated ``document_chunks_document_id_idx1`` or similar) - this
matters for Alembic migrations, which reference constraints by name when
altering or dropping them.
"""

from sqlalchemy import MetaData
from sqlalchemy.orm import DeclarativeBase

NAMING_CONVENTION = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}


class Base(DeclarativeBase):
    metadata = MetaData(naming_convention=NAMING_CONVENTION)
