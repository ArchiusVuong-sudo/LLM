import os
import chromadb
from dotenv import load_dotenv
from neo4j import GraphDatabase
from langchain_chroma import Chroma
from pydantic import BaseModel, Field
from langchain_openai import OpenAIEmbeddings
from langchain.indexes import SQLRecordManager, index
from langchain_core.prompts import ChatPromptTemplate
from chromadb.config import DEFAULT_TENANT, DEFAULT_DATABASE, Settings
from langchain_community.graphs import Neo4jGraph
from langchain_experimental.graph_transformers import LLMGraphTransformer

load_dotenv()

class VectorStore():
    def __init__(self, 
        host            = "localhost",
        port            = 8000,
        ssl             = False,
        headers         = None,
        settings        = Settings(),
        tenant          = DEFAULT_TENANT,
        database        = DEFAULT_DATABASE,
        collection_name = "chroma",
        embedding       = OpenAIEmbeddings(model="text-embedding-3-large")
    ):
        self.client = chromadb.HttpClient(
            host        = host,
            port        = port,
            ssl         = ssl,
            headers     = headers,
            settings    = settings,
            tenant      = tenant,
            database    = database
        )

        self.vector_store = Chroma(
            collection_name     = collection_name,
            embedding_function  = embedding
        )

        self.record_manager = SQLRecordManager(
            "chroma", db_url="sqlite:///record_manager_cache.sql"
        )

        self.record_manager.create_schema()
    
    def add_doc(self, docs):
        self.vector_store.add_documents(docs)
        self._clear()
        index(docs, self.record_manager, self.vector_store, cleanup="full", source_id_key="source")

    def get_retriever(self, search_type="mmr", k=5, fetch_k=10):
        return self.vector_store.as_retriever(
            search_type     = search_type, 
            search_kwargs   = {"k": k, "fetch_k": fetch_k}
        )
    
    def _clear(self):
        index([], self.record_manager, self.vector_store, cleanup="full", source_id_key="source")


class GraphStore():

    class Entities(BaseModel):
        """Identifying information about entities."""

        names: list[str] = Field(
            ...,
            description="All the person, organization, or business entities that "
            "appear in the text",
        )

    def __init__(self, llm):
        self.llm = llm
        self.graph = Neo4jGraph()
        self.llm_transformer = LLMGraphTransformer(llm=llm)
        self.driver = GraphDatabase.driver(
            uri = os.environ["NEO4J_URI"],
            auth = (
                os.environ["NEO4J_USERNAME"],
                os.environ["NEO4J_PASSWORD"]
            )
        )
    
    def add_doc(self, docs):
        graph_documents = self.llm_transformer.convert_to_graph_documents(docs)
        self.graph.add_graph_documents(
            graph_documents,
            baseEntityLabel=True,
            include_source=True
        )
        self.create_index()
    
    @staticmethod
    def create_fulltext_index(tx):
        query = '''
        CREATE FULLTEXT INDEX `fulltext_entity_id` 
        FOR (n:__Entity__) 
        ON EACH [n.id];
        '''
        tx.run(query)

    def create_index(self):
        try:
            with self.driver.session() as session:
                session.execute_write(self.create_fulltext_index)
        except Exception as e:
            print(f"- Error: {e}")
    
    def graph_retriever(self, question: str) -> str:
        """
        Collects the neighborhood of entities mentioned
        in the question
        """
        result = ""
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are extracting organization and person entities from the text.",
                ),
                (
                    "human",
                    "Use the given format to extract information from the following "
                    "input: {question}",
                ),
            ]
        )


        entity_chain = self.llm.with_structured_output(self.Entities)
        entities = entity_chain.invoke(question)

        for entity in entities.names:
            response = self.graph.query(
                """CALL db.index.fulltext.queryNodes('fulltext_entity_id', $query, {limit:2})
                YIELD node,score
                CALL {
                WITH node
                MATCH (node)-[r:!MENTIONS]->(neighbor)
                RETURN node.id + ' - ' + type(r) + ' -> ' + neighbor.id AS output
                UNION ALL
                WITH node
                MATCH (node)<-[r:!MENTIONS]-(neighbor)
                RETURN neighbor.id + ' - ' + type(r) + ' -> ' +  node.id AS output
                }
                RETURN output LIMIT 50
                """,
                {"query": entity},
            )
            result += "\n".join([el['output'] for el in response])

        return result
