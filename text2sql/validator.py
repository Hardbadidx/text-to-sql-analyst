"""AST-based SQL security validator and query rewriter using sqlglot."""

import sqlglot
from sqlglot import exp


class SecurityValidationError(Exception):
    """Raised when an input SQL query violates security or schema policies."""

    pass


class ValidatedSQL:
    """Type-safe, tamper-resistant wrapper representing SQL that has passed validation."""

    def __init__(self, sql: str):
        self._sql = sql

    @property
    def sql(self) -> str:
        return self._sql

    def __str__(self) -> str:
        return self._sql

    def __repr__(self) -> str:
        return f"ValidatedSQL({self._sql!r})"


class ValidationResult:
    """Outcome of SQL validation and rewriting."""

    def __init__(
        self,
        is_valid: bool,
        sanitized_sql: str | None = None,
        error_message: str | None = None,
        expanded_stars: bool = False,
        clamped_limit: bool = False,
        rewrote_dates: bool = False,
    ):
        self.is_valid = is_valid
        self.sanitized_sql = sanitized_sql
        self.error_message = error_message
        self.expanded_stars = expanded_stars
        self.clamped_limit = clamped_limit
        self.rewrote_dates = rewrote_dates
        self.validated_sql = ValidatedSQL(sanitized_sql) if is_valid and sanitized_sql else None

    def __repr__(self) -> str:
        return (
            f"ValidationResult(is_valid={self.is_valid}, "
            f"sanitized_sql={self.sanitized_sql!r}, "
            f"error_message={self.error_message!r})"
        )


class SQLValidator:
    """Security validator that enforces read-only semantics, table/column allowlists,

    function allowlists, sensitive column denylists, safe wildcard expansions,
    relative date rewrites to fixed reference date, and row LIMIT constraints.
    """

    FORBIDDEN_EXPRESSIONS: tuple[type[exp.Expression], ...] = (
        exp.Insert,
        exp.Update,
        exp.Delete,
        exp.Drop,
        exp.Alter,
        exp.Create,
        exp.Command,
        exp.Pragma,
        exp.Transaction,
        exp.Commit,
        exp.Rollback,
        exp.Merge,
    )

    # Strict function allowlist: only known-safe analytical, scalar, and window functions permitted
    ALLOWED_FUNCTIONS: set[str] = {
        # Aggregates
        "count",
        "sum",
        "avg",
        "min",
        "max",
        "total",
        "group_concat",
        # Math & String
        "round",
        "abs",
        "lower",
        "upper",
        "length",
        "substr",
        "substring",
        "trim",
        "ltrim",
        "rtrim",
        "instr",
        "replace",
        "concat",
        "concat_ws",
        "coalesce",
        "nullif",
        "ifnull",
        "iif",
        "if",
        # Date & Time
        "strftime",
        "date",
        "time",
        "datetime",
        "julianday",
        "time_to_str",
        "str_to_time",
        "timetostr",
        # Casting & Collation & Subqueries
        "cast",
        "collate",
        "exists",
        # Window functions
        "row_number",
        "rownumber",
        "rank",
        "dense_rank",
        "denserank",
        "lag",
        "lead",
        "ntile",
        "first_value",
        "firstvalue",
        "last_value",
        "lastvalue",
    }

    # Forbidden system / table-valued tables
    FORBIDDEN_TABLE_PREFIXES: tuple[str, ...] = (
        "sqlite_",
        "pragma_",
        "json_",
        "generate_series",
    )

    # AST nodes that inherit from exp.Func in sqlglot but represent logical operators,
    # conditional control flow, or internal transpilation helpers
    EXCLUDED_FUNC_TYPES: tuple[type[exp.Expression], ...] = (
        exp.Connector,
        exp.And,
        exp.Or,
        exp.Not,
        exp.Case,
        exp.If,
        exp.Concat,
        exp.TsOrDsToTimestamp,
        exp.TsOrDsToDate,
        exp.TsOrDsToDateStr,
        exp.TsOrDsToDatetime,
        exp.TsOrDsToTime,
        exp.TsOrDiToDi,
        exp.TsOrDsAdd,
        exp.TsOrDsDiff,
    )

    def __init__(
        self,
        allowed_tables: set[str],
        allowed_columns: dict[str, set[str]],
        denylist_columns: list[str] | None = None,
        max_query_limit: int = 1000,
        reference_date: str = "2024-12-31",
        dialect: str = "sqlite",
    ):
        self.allowed_tables = {t.lower() for t in allowed_tables}
        self.allowed_columns = {
            tbl.lower(): {c.lower() for c in cols}
            for tbl, cols in allowed_columns.items()
        }
        self.denylist = [c.lower().strip() for c in (denylist_columns or [])]
        self.max_query_limit = max_query_limit
        self.reference_date = reference_date
        self.dialect = dialect

        # Identify tables that contain denylisted columns
        self.tables_with_denylisted_cols: dict[str, list[str]] = {}
        for tbl in self.allowed_columns:
            self.tables_with_denylisted_cols[tbl] = [
                d for d in self.denylist if d in ("phone", "email", "address", "ssn", "social_security")
            ]

    def is_denylisted(self, col_name: str) -> bool:
        """Check if column name matches any denylisted sensitive attribute."""
        col_clean = col_name.lower().strip().strip("`\"'[]")
        # Handle possible table.col notation (e.g. patients.phone)
        if "." in col_clean:
            col_clean = col_clean.split(".")[-1].strip("`\"'[]")
        return any(col_clean == blocked for blocked in self.denylist)

    def validate(
        self,
        raw_sql: str,
        expand_stars: bool = True,
        reject_stars_on_sensitive: bool = False,
    ) -> ValidationResult:
        """Validate raw SQL string against security constraints and return sanitized SQL.

        Args:
            raw_sql: Untrusted SQL query string.
            expand_stars: If True, expand SELECT * / alias.* to non-denylisted columns.
            reject_stars_on_sensitive: If True, reject SELECT * on tables with sensitive columns
                                       instead of expanding.

        Returns:
            ValidationResult with sanitized SQL or error message.
        """
        if not raw_sql or not raw_sql.strip():
            return ValidationResult(is_valid=False, error_message="Empty query.")

        cleaned = raw_sql.strip().rstrip(";")

        # 1. Parse AST with sqlglot and verify single statement
        try:
            parsed = sqlglot.parse(cleaned, read=self.dialect)
            statements = [s for s in parsed if s is not None]
        except Exception as e:
            return ValidationResult(
                is_valid=False,
                error_message=f"SQL syntax parsing failed: {e}",
            )

        if not statements:
            return ValidationResult(is_valid=False, error_message="No valid SQL statements found.")

        if len(statements) > 1:
            return ValidationResult(
                is_valid=False,
                error_message="Multiple SQL statements detected. Only a single SELECT query is permitted.",
            )

        ast = statements[0]

        # 2. Check root statement type: must be Select or Union (with optional With / CTE)
        if not isinstance(ast, (exp.Select, exp.Union)):
            return ValidationResult(
                is_valid=False,
                error_message=f"Disallowed statement type: {type(ast).__name__}. Only SELECT queries are permitted.",
            )

        # 3. Check for any forbidden destructive / administrative expressions in entire tree
        for forbidden_cls in self.FORBIDDEN_EXPRESSIONS:
            if list(ast.find_all(forbidden_cls)):
                return ValidationResult(
                    is_valid=False,
                    error_message=f"Disallowed operation detected: {forbidden_cls.__name__}.",
                )

        # 4. Check for NATURAL JOINs (implicit joins forbidden)
        for join in ast.find_all(exp.Join):
            join_method = str(join.args.get("method") or "").upper()
            if "NATURAL" in join_method or join.args.get("natural"):
                return ValidationResult(
                    is_valid=False,
                    error_message="NATURAL JOIN is forbidden. Explicit JOIN ... ON conditions are required.",
                )
            # Check JOIN ... USING(col) for denylisted columns
            using_clause = join.args.get("using")
            if using_clause:
                for using_col in using_clause:
                    col_name = str(using_col).lower().strip().strip("`\"'[]")
                    if self.is_denylisted(col_name):
                        return ValidationResult(
                            is_valid=False,
                            error_message=(
                                f"Access to sensitive denylisted column '{col_name}' in JOIN USING is strictly forbidden."
                            ),
                        )

        # 5. Date rewrites: replace 'now', CURRENT_DATE, CURRENT_TIME, CURRENT_TIMESTAMP
        # with fixed reference date (2024-12-31) so queries never evaluate against the live wall-clock
        ast, rewrote_dates = self._rewrite_relative_dates(ast)

        # 6. Function allowlist enforcement: reject load_extension, zeroblob, randomblob, and unauthorized functions
        for func_node in ast.find_all(exp.Func):
            # Skip operators, predicates, CASE/WHEN, and transpiler helpers that inherit from exp.Func
            if isinstance(func_node, self.EXCLUDED_FUNC_TYPES):
                continue

            if isinstance(func_node, exp.Anonymous):
                func_name = func_node.name.lower().strip()
            else:
                key_name = getattr(func_node, "key", "").lower().strip()
                sql_name = func_node.sql_name().lower().strip() if hasattr(func_node, "sql_name") else ""
                func_name = key_name or sql_name

            is_allowed = (
                func_name in self.ALLOWED_FUNCTIONS
                or getattr(func_node, "key", "").lower().strip() in self.ALLOWED_FUNCTIONS
                or (hasattr(func_node, "sql_name") and func_node.sql_name().lower().strip() in self.ALLOWED_FUNCTIONS)
            )
            if not is_allowed:
                return ValidationResult(
                    is_valid=False,
                    error_message=f"Function '{func_name}' is not allowed.",
                )

        # 6. Extract CTE definitions and subquery aliases to treat as valid virtual tables
        cte_names: set[str] = set()
        for with_clause in ast.find_all(exp.With):
            for cte in with_clause.expressions:
                if cte.alias:
                    cte_alias = cte.alias.lower()
                    # Check if CTE shadows an authorized physical table name
                    if cte_alias in self.allowed_tables:
                        return ValidationResult(
                            is_valid=False,
                            error_message=f"CTE alias '{cte.alias}' illegally shadows authorized table name '{cte.alias}'.",
                        )
                    cte_names.add(cte_alias)

        # Collect subquery aliases
        for subquery in ast.find_all(exp.Subquery):
            if subquery.alias:
                cte_names.add(subquery.alias.lower())

        # 7. Validate table references
        table_aliases: dict[str, str] = {}  # alias -> real_table_name
        query_tables: set[str] = set()

        for tbl_node in ast.find_all(exp.Table):
            tbl_name = tbl_node.name.lower()
            if not tbl_name:
                return ValidationResult(
                    is_valid=False,
                    error_message="Encountered invalid table expression without identifier.",
                )

            # Check forbidden table prefixes (sqlite_*, pragma_*, json_*, etc.)
            for prefix in self.FORBIDDEN_TABLE_PREFIXES:
                if tbl_name.startswith(prefix):
                    return ValidationResult(
                        is_valid=False,
                        error_message=f"Access to system or table-valued function '{tbl_name}' is forbidden.",
                    )

            # If it's a CTE or virtual alias, allow it
            if tbl_name in cte_names:
                continue

            if tbl_name not in self.allowed_tables:
                return ValidationResult(
                    is_valid=False,
                    error_message=f"Access to unknown or unauthorized table '{tbl_name}' is forbidden.",
                )

            query_tables.add(tbl_name)
            alias = tbl_node.alias.lower() if tbl_node.alias else tbl_name
            table_aliases[alias] = tbl_name
            table_aliases[tbl_name] = tbl_name

        # 8. Check explicit column references for denylist violations
        projected_aliases: set[str] = {
            a.alias.lower().strip("`\"'[]") for a in ast.find_all(exp.Alias) if a.alias
        }

        for col_node in ast.find_all(exp.Column):
            col_name = col_node.name.lower()
            if col_name == "*":
                continue

            # Strip quoting, brackets, backticks
            clean_col = col_name.strip("`\"'[]")

            if self.is_denylisted(clean_col):
                return ValidationResult(
                    is_valid=False,
                    error_message=(
                        f"Access to sensitive denylisted column '{clean_col}' is strictly forbidden."
                    ),
                )

            # Skip checking against base table columns if column is a projected alias (e.g. in ORDER BY, HAVING, or CTE)
            if clean_col in projected_aliases:
                continue

            # Check column allowlist if table can be determined
            tbl_ref = col_node.table.lower() if col_node.table else None
            real_tbl = table_aliases.get(tbl_ref) if tbl_ref else None

            if not real_tbl and len(query_tables) == 1:
                real_tbl = next(iter(query_tables))

            if real_tbl and real_tbl in self.allowed_columns:
                allowed_cols = self.allowed_columns[real_tbl]
                if clean_col not in allowed_cols:
                    return ValidationResult(
                        is_valid=False,
                        error_message=(
                            f"Unknown column '{clean_col}' on table '{real_tbl}'."
                        ),
                    )

        # 9. Wildcard (SELECT *) handling across tables with denylisted columns
        expanded_stars = False
        try:
            ast, expanded_stars = self._handle_star_selections(
                ast,
                table_aliases=table_aliases,
                query_tables=query_tables,
                expand_stars=expand_stars,
                reject_stars_on_sensitive=reject_stars_on_sensitive,
            )
        except SecurityValidationError as sve:
            return ValidationResult(is_valid=False, error_message=str(sve))

        # 11. LIMIT enforcement: add default or clamp excessive LIMIT
        clamped_limit = False
        limit_node = ast.args.get("limit")
        if limit_node is None:
            ast = ast.limit(self.max_query_limit)
            clamped_limit = True
        else:
            limit_val_expr = limit_node.expression
            if isinstance(limit_val_expr, exp.Literal) and limit_val_expr.is_int:
                val = int(limit_val_expr.this)
                if val > self.max_query_limit:
                    limit_node.set("expression", exp.Literal.number(self.max_query_limit))
                    clamped_limit = True
            elif not isinstance(limit_val_expr, exp.Literal):
                # Non-literal or complex limit, clamp to safe limit
                limit_node.set("expression", exp.Literal.number(self.max_query_limit))
                clamped_limit = True

        sanitized_sql = ast.sql(dialect=self.dialect)
        return ValidationResult(
            is_valid=True,
            sanitized_sql=sanitized_sql,
            expanded_stars=expanded_stars,
            clamped_limit=clamped_limit,
            rewrote_dates=rewrote_dates,
        )

    def _rewrite_relative_dates(self, ast: exp.Expression) -> tuple[exp.Expression, bool]:
        """Rewrite 'now', CURRENT_DATE, CURRENT_TIME, CURRENT_TIMESTAMP to fixed reference date."""
        rewrote = False
        ref_date = self.reference_date
        ref_timestamp = f"{ref_date} 23:59:59"

        for node in list(ast.walk()):
            if isinstance(node, exp.CurrentDate):
                node.replace(exp.Literal.string(ref_date))
                rewrote = True
            elif isinstance(node, exp.CurrentTimestamp):
                node.replace(exp.Literal.string(ref_timestamp))
                rewrote = True
            elif isinstance(node, exp.CurrentTime):
                node.replace(exp.Literal.string("23:59:59"))
                rewrote = True
            elif isinstance(node, exp.Literal) and str(node.this).lower() == "now":
                parent = node.parent
                if parent and isinstance(parent, exp.Anonymous) and parent.name.lower() in ("datetime", "julianday", "strftime"):
                    node.replace(exp.Literal.string(ref_timestamp))
                else:
                    node.replace(exp.Literal.string(ref_date))
                rewrote = True

        return ast, rewrote

    def _handle_star_selections(
        self,
        ast: exp.Expression,
        table_aliases: dict[str, str],
        query_tables: set[str],
        expand_stars: bool,
        reject_stars_on_sensitive: bool,
    ) -> tuple[exp.Expression, bool]:
        """Inspect and either safely expand or reject SELECT * or alias.* on sensitive tables."""
        expanded_any = False

        # Find all Select expressions (including in CTEs and subqueries)
        for select_stmt in ast.find_all(exp.Select):
            new_select_expressions: list[exp.Expression] = []
            modified = False

            # Local table mappings for this select statement
            local_tables: set[str] = set()
            local_aliases: dict[str, str] = {}

            for tbl_node in select_stmt.find_all(exp.Table):
                t_name = tbl_node.name.lower()
                alias = tbl_node.alias.lower() if tbl_node.alias else t_name
                real_t = table_aliases.get(t_name, t_name)
                local_tables.add(real_t)
                local_aliases[alias] = real_t
                local_aliases[t_name] = real_t

            if not local_tables and query_tables:
                local_tables = query_tables

            for expr in select_stmt.expressions:
                # Case 1: Unqualified Star `SELECT *`
                if isinstance(expr, exp.Star):
                    # Check if any local table has denylisted columns
                    has_sensitive = any(
                        tbl in ("patients",) or any(self.is_denylisted(d) for d in self.denylist)
                        for tbl in local_tables
                    )
                    if has_sensitive:
                        if reject_stars_on_sensitive or not expand_stars:
                            raise SecurityValidationError(
                                "Wildcard 'SELECT *' is blocked because queried tables contain sensitive columns. "
                                "Explicitly specify the allowed columns."
                            )
                        if not local_tables:
                            raise SecurityValidationError(
                                "Wildcard 'SELECT *' cannot be expanded because target table could not be resolved."
                            )
                        # Expand star to allowed columns of all local tables
                        for tbl in sorted(local_tables):
                            if tbl in self.allowed_columns:
                                for col_name in sorted(self.allowed_columns[tbl]):
                                    new_select_expressions.append(
                                        exp.Column(
                                            this=exp.to_identifier(col_name),
                                            table=exp.to_identifier(tbl) if len(local_tables) > 1 else None,
                                        )
                                    )
                        modified = True
                        expanded_any = True
                    else:
                        new_select_expressions.append(expr)

                # Case 2: Qualified Star `SELECT p.*`
                elif isinstance(expr, exp.Column) and isinstance(expr.this, exp.Star):
                    alias = expr.table.lower() if expr.table else ""
                    real_tbl = local_aliases.get(alias, table_aliases.get(alias))

                    if not real_tbl:
                        raise SecurityValidationError(
                            f"Unable to resolve wildcard '{alias}.*' to an authorized table schema."
                        )

                    if real_tbl in ("patients",) or any(self.is_denylisted(d) for d in self.denylist):
                        if reject_stars_on_sensitive or not expand_stars:
                            raise SecurityValidationError(
                                f"Wildcard '{alias}.*' is blocked because table '{real_tbl}' contains "
                                f"sensitive columns. Explicitly specify the allowed columns."
                            )
                        if real_tbl in self.allowed_columns:
                            for col_name in sorted(self.allowed_columns[real_tbl]):
                                new_select_expressions.append(
                                    exp.Column(
                                        this=exp.to_identifier(col_name),
                                        table=exp.to_identifier(alias) if alias else None,
                                    )
                                )
                            modified = True
                            expanded_any = True
                        else:
                            new_select_expressions.append(expr)
                    else:
                        new_select_expressions.append(expr)
                else:
                    new_select_expressions.append(expr)

            if modified:
                select_stmt.set("expressions", new_select_expressions)

        return ast, expanded_any
