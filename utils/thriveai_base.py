from vanna.legacy.base import VannaBase

from utils.quick_logger import get_logger

logger = get_logger(__name__)


_REDSHIFT_CHEATSHEET = (
    "===Dialect Notes (Redshift) \n"
    "Redshift differs from PostgreSQL in several places. When you emit SQL:\n"
    " - Use LISTAGG(col, ', ') WITHIN GROUP (ORDER BY ...) instead of STRING_AGG(... ORDER BY ...).\n"
    " - Redshift has no CORR() aggregate; compute correlation manually with AVG/STDDEV if needed.\n"
    " - Do not use array_position(), array_agg(), or other Postgres array helpers.\n"
    " - SUBSTRING(col FROM x FOR y) is portable; prefer it over SUBSTR variants.\n"
    " - EXTRACT(YEAR FROM col), DATE_TRUNC, and CAST are all supported.\n"
    " - Identifier quoting uses double quotes; string literals use single quotes.\n"
)


def dialect_cheatsheet(dialect: str | None) -> str:
    """Return dialect-specific guidance for the system prompt.

    Empty string when the active dialect doesn't need extra rules
    (e.g. the default PostgreSQL path, which matches the bulk of our
    training examples).
    """
    if (dialect or "").lower() == "redshift":
        return _REDSHIFT_CHEATSHEET
    return ""


class ThriveAI_Base(VannaBase):
    """Base class for ThriveAI"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.schema = self.config.get("schema", "public")
        self.dialect = self.config.get("dialect", getattr(self, "dialect", "postgresql"))

    def log(self, message: str, title: str = "Info"):
        """Override the deault log method, which is print, to use the logger."""
        logger.debug("%s: %s", title, message)

    def get_sql_prompt(
        self,
        initial_prompt: str,
        question: str,
        question_sql_list: list,
        ddl_list: list,
        doc_list: list,
        **kwargs,
    ):
        """
        Example:
        ```python
        vn.get_sql_prompt(
            question="What are the top 10 customers by sales?",
            question_sql_list=[{"question": "What are the top 10 customers by sales?", "sql": "SELECT * FROM customers ORDER BY sales DESC LIMIT 10"}],
            ddl_list=["CREATE TABLE customers (id INT, name TEXT, sales DECIMAL)"],
            doc_list=["The customers table contains information about customers and their sales."],
        )

        ```

        This method is used to generate a prompt for the LLM to generate SQL.

        Args:
            question (str): The question to generate SQL for.
            question_sql_list (list): A list of questions and their corresponding SQL statements.
            ddl_list (list): A list of DDL statements.
            doc_list (list): A list of documentation.

        Returns:
            any: The prompt for the LLM to generate SQL.
        """

        if initial_prompt is None:
            initial_prompt = (
                f"You are a {self.dialect} expert. "
                + "Please help to generate a SQL query to answer the question. Your response should ONLY be based on the given context and follow the response guidelines and format instructions. "
            )

        initial_prompt = self.add_ddl_to_prompt(initial_prompt, ddl_list, max_tokens=self.max_tokens)

        if self.static_documentation != "":
            doc_list.append(self.static_documentation)

        initial_prompt = self.add_documentation_to_prompt(initial_prompt, doc_list, max_tokens=self.max_tokens)

        initial_prompt += (
            "===Response Guidelines \n"
            "1. If the provided context is sufficient, please generate a valid SQL query without any explanations for the question. \n"
            "2. If the provided context is almost sufficient but requires knowledge of a specific string in a particular column, please generate an intermediate SQL query to find the distinct strings in that column. Prepend the query with a comment saying intermediate_sql \n"
            "3. If the provided context is insufficient, please explain why it can't be generated. \n"
            "4. Please use the most relevant table(s). \n"
            "5. If the question has been asked and answered before, please repeat the answer exactly as it was given before. \n"
            f"6. Ensure that the output SQL is {self.dialect}-compliant and executable, and free of syntax errors. \n"
            f"7. Make sure to fully qualify table names with the schema name {self.schema}, even if not specified in the ddl. Eg. SELECT * FROM dw.customers;\n"
        )

        initial_prompt += dialect_cheatsheet(self.dialect)

        initial_prompt += "=== Example Question/SQL Pairs \n"

        message_log = [self.system_message(initial_prompt)]

        for example in question_sql_list:
            if example is None:
                print("example is None")
            else:
                if example is not None and "question" in example and "sql" in example:
                    message_log.append(self.user_message(example["question"]))
                    message_log.append(self.assistant_message(example["sql"]))

        message_log.append(self.user_message(question))

        return message_log
