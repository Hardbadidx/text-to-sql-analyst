"""Database schema introspection with sensitive column denylist filtering."""

from typing import Any

from pydantic import BaseModel, Field
from sqlalchemy import Engine, inspect, text


class ColumnInfo(BaseModel):
    """Metadata describing a single database column."""

    name: str
    type_name: str
    nullable: bool = True
    primary_key: bool = False
    sample_values: list[Any] = Field(default_factory=list)


class ForeignKeyInfo(BaseModel):
    """Metadata describing a foreign key relationship."""

    constrained_columns: list[str]
    referred_table: str
    referred_columns: list[str]


class TableInfo(BaseModel):
    """Metadata describing a single database table."""

    name: str
    columns: list[ColumnInfo] = Field(default_factory=list)
    foreign_keys: list[ForeignKeyInfo] = Field(default_factory=list)
    row_count: int | None = None

    @property
    def column_names(self) -> set[str]:
        return {col.name.lower() for col in self.columns}


class DatabaseSchema(BaseModel):
    """Complete metadata for an introspected database."""

    tables: dict[str, TableInfo] = Field(default_factory=dict)
    denylist: list[str] = Field(default_factory=list)
    reference_date: str = "2024-12-31"

    @property
    def table_names(self) -> set[str]:
        return {t.lower() for t in self.tables}

    def get_allowed_columns(self) -> dict[str, set[str]]:
        """Return a mapping of table name (lowercase) to allowed column names (lowercase)."""
        return {tbl_name.lower(): tbl_info.column_names for tbl_name, tbl_info in self.tables.items()}

    def is_column_denylisted(self, column_name: str) -> bool:
        """Check if a column name matches any entry in the denylist."""
        col_clean = column_name.lower().strip().strip("`\"'[]")
        return any(col_clean == blocked.lower().strip() for blocked in self.denylist)

    def to_prompt_context(self) -> str:
        """Format the introspected schema as markdown documentation for the LLM prompt.

        Critically, denylisted columns and their sample values are completely omitted.
        """
        lines: list[str] = [
            "# Database Schema (Clinic SaaS)\n",
            f"**Fixed Reference Date (CURRENT_DATE)**: `{self.reference_date}`",
            "*(Important: Use this fixed date as 'today' or 'current_date' for relative calculations like 'this year', 'last month', etc.)*\n",
        ]

        for table_name, table in self.tables.items():
            lines.append(f"## Table: `{table_name}`")
            if table.row_count is not None:
                lines.append(f"- Estimated Rows: {table.row_count:,}")
            lines.append("- Columns:")
            for col in table.columns:
                pk_str = " [PRIMARY KEY]" if col.primary_key else ""
                null_str = " NULL" if col.nullable else " NOT NULL"

                if col.sample_values:
                    # Format samples safely
                    formatted_samples = []
                    for v in col.sample_values:
                        if isinstance(v, str):
                            formatted_samples.append(f"'{v}'")
                        elif v is None:
                            continue
                        else:
                            formatted_samples.append(str(v))
                    samples_str = f" | Samples: [{', '.join(formatted_samples[:3])}]"
                else:
                    samples_str = ""

                lines.append(f"  - `{col.name}` ({col.type_name}{null_str}{pk_str}){samples_str}")

            if table.foreign_keys:
                lines.append("- Foreign Keys:")
                for fk in table.foreign_keys:
                    local_cols = ", ".join(fk.constrained_columns)
                    ref_cols = ", ".join(fk.referred_columns)
                    lines.append(f"  - ({local_cols}) -> `{fk.referred_table}`({ref_cols})")
            lines.append("")

        return "\n".join(lines)


def introspect_schema(
    engine: Engine,
    denylist: list[str] | None = None,
    sample_limit: int = 3,
) -> DatabaseSchema:
    """Introspect all tables, columns, foreign keys, and safe sample values from the database.

    Denylisted columns (e.g. phone, email) are strictly excluded from both the schema
    and sample queries, ensuring zero leak of sensitive PII.
    """
    if denylist is None:
        denylist = ["phone", "email", "address", "ssn", "social_security"]

    denylist_lower = {col.lower().strip() for col in denylist}
    inspector = inspect(engine)
    table_names = inspector.get_table_names()

    # Determine reference date from data if appointments table exists
    reference_date = "2024-12-31"
    if "appointments" in table_names:
        with engine.connect() as conn:
            try:
                max_dt = conn.execute(text("SELECT MAX(scheduled_at) FROM appointments")).scalar()
                if max_dt:
                    reference_date = str(max_dt)[:10]
            except Exception:
                pass

    schema = DatabaseSchema(denylist=list(denylist_lower), reference_date=reference_date)

    with engine.connect() as conn:
        for table_name in table_names:
            columns_raw = inspector.get_columns(table_name)
            pk_constraint = inspector.get_pk_constraint(table_name)
            pk_cols = set(pk_constraint.get("constrained_columns", []) if pk_constraint else [])
            fks_raw = inspector.get_foreign_keys(table_name)

            # Get row count
            try:
                count_res = conn.execute(text(f'SELECT COUNT(*) FROM "{table_name}"')).scalar()
                row_count = int(count_res) if count_res is not None else 0
            except Exception:
                row_count = None

            table_columns: list[ColumnInfo] = []

            for col in columns_raw:
                col_name = col["name"]
                # Skip denylisted column completely
                if col_name.lower() in denylist_lower:
                    continue

                col_type = str(col["type"])
                is_pk = col_name in pk_cols
                nullable = bool(col.get("nullable", True))

                # Fetch sample values strictly for non-denylisted columns
                sample_values: list[Any] = []
                try:
                    query = text(
                        f'SELECT DISTINCT "{col_name}" FROM "{table_name}" '
                        f'WHERE "{col_name}" IS NOT NULL LIMIT {sample_limit}'
                    )
                    rows = conn.execute(query).fetchall()
                    sample_values = [r[0] for r in rows if r[0] is not None]
                except Exception:
                    sample_values = []

                table_columns.append(
                    ColumnInfo(
                        name=col_name,
                        type_name=col_type,
                        nullable=nullable,
                        primary_key=is_pk,
                        sample_values=sample_values,
                    )
                )

            foreign_keys: list[ForeignKeyInfo] = []
            for fk in fks_raw:
                foreign_keys.append(
                    ForeignKeyInfo(
                        constrained_columns=fk.get("constrained_columns", []),
                        referred_table=fk.get("referred_table", ""),
                        referred_columns=fk.get("referred_columns", []),
                    )
                )

            schema.tables[table_name] = TableInfo(
                name=table_name,
                columns=table_columns,
                foreign_keys=foreign_keys,
                row_count=row_count,
            )

    return schema
