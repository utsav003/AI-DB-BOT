
import os
import pyodbc
from pathlib import Path
from typing import Dict, List
from dotenv import load_dotenv

from llama_index.llms.openai import OpenAI
from llama_index.core.schema import TextNode
from llama_index.core.llms import ChatResponse
from llama_index.core.retrievers import SQLRetriever
from llama_index.llms.azure_openai import AzureOpenAI
from sqlalchemy import (create_engine,MetaData,text)
from sqlalchemy.exc import DBAPIError,ProgrammingError
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from llama_index.core.prompts.default_prompts import DEFAULT_TEXT_TO_SQL_PROMPT
from llama_index.core.objects import (SQLTableNodeMapping,ObjectIndex,SQLTableSchema,)
from llama_index.core import SQLDatabase, VectorStoreIndex, PromptTemplate, VectorStoreIndex, load_index_from_storage, StorageContext, Settings
from llama_index.core.query_pipeline import (QueryPipeline as QP, InputComponent, FnComponent)

print(pyodbc.drivers())

load_dotenv()

system_prompt= "You are DB analyst and your job is to give precise information from database"

############ Pipeline with Azure AI #######################

llm = AzureOpenAI(
    model="gpt-35-turbo-16k",
    deployment_name="QueryModel",
    api_key=os.environ.get("api_key"),
    azure_endpoint=os.environ.get("azure_endpoint"),
    api_version=os.environ.get("api_version"),
    system_prompt=system_prompt
)

embed_model = AzureOpenAIEmbedding(
    model="text-embedding-ada-002",
    deployment_name="IndexingModel",
    api_key=os.environ.get("api_key"),
    azure_endpoint=os.environ.get("azure_endpoint"),
    api_version=os.environ.get("api_version"),
)

Settings.llm = llm
Settings.embed_model = embed_model

############ Pipeline with OpenAI ############

# llm=OpenAI(model="gpt-3.5-turbo",api_key=os.environ.get("OPENAI_API_KEY"))

engine = create_engine("postgresql+psycopg2://admin:test@localhost/sample")
metadata_obj = MetaData()
metadata_obj.reflect(engine)
sql_database = SQLDatabase(engine)

table_node_mapping = SQLTableNodeMapping(sql_database)

def create_tableschema():
    context_data = (
    """This table product gives information regarding the report_time, product_name, sub_product,\
    'market_name', 'value', 'unit \
     The user will query regarding all the columns."""

)
    table_schema_objs = []

    data_table = ["product"]

    for table_name, ctx_data in zip(data_table, context_data):
        table_schema_objs.append(SQLTableSchema(table_name=table_name , context_str=ctx_data))

    obj_index = ObjectIndex.from_objects(
        table_schema_objs,
        table_node_mapping,
        VectorStoreIndex,)
    
    return obj_index
obj_index= create_tableschema()
obj_retriever = obj_index.as_retriever(similarity_top_k=3)

print('.........LOADING DATA..............')

sql_retriever = SQLRetriever(sql_database)

def parse_response_to_sql(response: ChatResponse) -> str:
    """Parse response to SQL."""
    response = response.message.content
    sql_query_start = response.find("SQLQuery:")
    if sql_query_start != -1:
        response = response[sql_query_start:]
        if response.startswith("SQLQuery:"):
            response = response[len("SQLQuery:") :]
    sql_result_start = response.find("SQLResult:")
    if sql_result_start != -1:
        response = response[:sql_result_start]
    return response.strip().strip("```").strip()


sql_parser_component = FnComponent(fn=parse_response_to_sql)

text2sql_prompt = DEFAULT_TEXT_TO_SQL_PROMPT.partial_format(
    dialect=engine.dialect.name
)
print(text2sql_prompt.template)

response_synthesis_prompt_str = (
    "Given an input question, synthesize a response from the query results.\n"
    "Query: {query_str}\n"
    "SQL: {sql_query}\n"
    "SQL Response: {context_str}\n"
    "Response: "
)
response_synthesis_prompt = PromptTemplate(
    response_synthesis_prompt_str,
)

def index_all_tables(
    sql_database: SQLDatabase, table_index_dir: str = "table_index_dir"
) -> Dict[str, VectorStoreIndex]:
    """Index all tables."""
    if not Path(table_index_dir).exists():
        os.makedirs(table_index_dir)

    vector_index_dict = {}
    engine = sql_database.engine

    data_table = ["product"]

    for table_name in data_table:
        print(f"Indexing rows in table: {table_name}")
        if not os.path.exists(f"{table_index_dir}/{table_name}"):
            with engine.connect() as conn:
                cursor = conn.execute(text(f'SELECT * FROM "{table_name}" LIMIT 500'))
                result = cursor.fetchall()
                row_tups = []
                for row in result:
                    row_tups.append(tuple(row))

            nodes = [TextNode(text=str(t)) for t in row_tups]

            index = VectorStoreIndex(nodes)

            index.set_index_id("vector_index")
            index.storage_context.persist(f"{table_index_dir}/{table_name}")
        else:
            # rebuild storage context
            storage_context = StorageContext.from_defaults(
                persist_dir=f"{table_index_dir}/{table_name}"
            )
            # load index
            index = load_index_from_storage(
                storage_context, index_id="vector_index"
            )
        vector_index_dict[table_name] = index

    return vector_index_dict


vector_index_dict = index_all_tables(sql_database)
sql_retriever = SQLRetriever(sql_database)


def get_table_context_and_rows_str(
    query_str: str, table_schema_objs: List[SQLTableSchema]
):
    """Get table context string."""
    context_strs = []
    for table_schema_obj in table_schema_objs:
        # first append table info + additional context
        table_info = sql_database.get_single_table_info(
            table_schema_obj.table_name
        )
        if table_schema_obj.context_str:
            table_opt_context = " The table description is: "
            table_opt_context += table_schema_obj.context_str
            table_info += table_opt_context

        # also lookup vector index to return relevant table rows
        vector_retriever = vector_index_dict[
            table_schema_obj.table_name
        ].as_retriever(similarity_top_k=5)
        relevant_nodes = vector_retriever.retrieve(query_str)
        if len(relevant_nodes) > 0:
            table_row_context = "\nHere are some relevant example rows (values in the same order as columns above)\n"
            for node in relevant_nodes:
                table_row_context += str(node.get_content()) + "\n"
            table_info += table_row_context

        context_strs.append(table_info)
    return "\n\n".join(context_strs)


table_parser_component = FnComponent(fn=get_table_context_and_rows_str)

qp = QP(
    modules={
        "input": InputComponent(),
        "table_retriever": obj_retriever,
        "table_output_parser": table_parser_component,
        "text2sql_prompt": text2sql_prompt,
        "text2sql_llm": llm,
        "sql_output_parser": sql_parser_component,
        "sql_retriever": sql_retriever,
        "response_synthesis_prompt": response_synthesis_prompt,
        "response_synthesis_llm": llm,
    },
    verbose=True,
)

qp.add_link("input", "table_retriever")
qp.add_link("input", "table_output_parser", dest_key="query_str")
qp.add_link(
    "table_retriever", "table_output_parser", dest_key="table_schema_objs"
)
qp.add_link("input", "text2sql_prompt", dest_key="query_str")
qp.add_link("table_output_parser", "text2sql_prompt", dest_key="schema")
qp.add_chain(
    ["text2sql_prompt", "text2sql_llm", "sql_output_parser", "sql_retriever"]
)
qp.add_link(
    "sql_output_parser", "response_synthesis_prompt", dest_key="sql_query"
)
qp.add_link(
    "sql_retriever", "response_synthesis_prompt", dest_key="context_str"
)
qp.add_link("input", "response_synthesis_prompt", dest_key="query_str")
qp.add_link("response_synthesis_prompt", "response_synthesis_llm")

    
def pipeline_qa_chain(user_input):
    print('........GOT INPUT QA CHAIN.........')
    try:
        response = qp.run(query=user_input)
        print("Response:", response)
        return response.message.content

    except (DBAPIError, ProgrammingError,NotImplementedError,KeyError) as e:
        return "Please ask only database-related questions."