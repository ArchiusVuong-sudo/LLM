import operator
from langchain import hub
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langchain_text_splitters import CharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain.chains.combine_documents.reduce import acollapse_docs, split_list_of_docs
from langgraph.constants import Send
from langgraph.graph import END, START, StateGraph

from typing import Annotated, List, Literal, TypedDict, Dict, Any


class DocumentSummarizer:
    """Class to summarize documents using map-reduce approach with LangChain LLM."""
    
    token_max: int = 20000
    
    def __init__(self, 
            llm: ChatOpenAI,
            docs: list[Document], 
            chunk_size: int = 10000, 
            chunk_overlap: int = 1000
    ):
        """Initialize the DocumentSummarizer with model and document loader."""
        self.llm = llm
        self.text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
            chunk_size = chunk_size, 
            chunk_overlap = chunk_overlap
        )        

        self.split_docs = self.text_splitter.split_documents(docs)
        
        # Define map and reduce chains
        self.map_prompt: str = hub.pull("rlm/map-prompt")
        self.map_chain = self.map_prompt | self.llm | StrOutputParser()

        self.reduce_prompt: str = hub.pull("rlm/reduce-prompt")
        self.reduce_chain = self.reduce_prompt | self.llm | StrOutputParser()
        
        self.graph = self._construct_graph()

    class OverallState(TypedDict):
        """State of the main graph containing document contents, summaries, and final summary."""
        contents: List[str]
        summaries: Annotated[list, operator.add]
        collapsed_summaries: List[Document]
        final_summary: str

    class SummaryState(TypedDict):
        """State of the node to generate summaries for individual documents."""
        content: str

    def length_function(self, documents: List[Document]) -> int:
        """Calculate the total token count of document contents."""
        return sum(self.llm.get_num_tokens(doc.page_content) for doc in documents)

    async def generate_summary(self, state: SummaryState) -> Dict[str, Any]:
        """Generate a summary for a given document content."""
        response = await self.map_chain.ainvoke(state["content"])
        return {"summaries": [response]}

    def map_summaries(self, state: OverallState) -> List[Send]:
        """Map documents to summary generation nodes in the graph."""
        return [
            Send("generate_summary", {"content": content}) for content in state["contents"]
        ]

    def collect_summaries(self, state: OverallState) -> Dict[str, List[Document]]:
        """Collect summaries into collapsed summary documents."""
        return {
            "collapsed_summaries": [Document(summary) for summary in state["summaries"]]
        }

    async def collapse_summaries(self, state: OverallState) -> Dict[str, List[Document]]:
        """Collapse summaries to reduce token count if necessary."""
        doc_lists = split_list_of_docs(state["collapsed_summaries"], self.length_function, self.token_max)
        results = []
        for doc_list in doc_lists:
            results.append(await acollapse_docs(doc_list, self.reduce_chain.ainvoke))
        return {"collapsed_summaries": results}

    def should_collapse(self, state: OverallState) -> Literal["collapse_summaries", "generate_final_summary"]:
        """Determine whether to collapse summaries or generate final summary."""
        num_tokens = self.length_function(state["collapsed_summaries"])
        if num_tokens > self.token_max:
            return "collapse_summaries"
        return "generate_final_summary"

    async def generate_final_summary(self, state: OverallState) -> Dict[str, str]:
        """Generate the final consolidated summary."""
        response = await self.reduce_chain.ainvoke(state["collapsed_summaries"])
        return {"final_summary": response}

    def _construct_graph(self) -> StateGraph:
        """Construct the state graph for document summarization."""
        graph = StateGraph(self.OverallState)
        graph.add_node("generate_summary", self.generate_summary)
        graph.add_node("collect_summaries", self.collect_summaries)
        graph.add_node("collapse_summaries", self.collapse_summaries)
        graph.add_node("generate_final_summary", self.generate_final_summary)

        # Define graph edges
        graph.add_conditional_edges(START, self.map_summaries, ["generate_summary"])
        graph.add_edge("generate_summary", "collect_summaries")
        graph.add_conditional_edges("collect_summaries", self.should_collapse)
        graph.add_conditional_edges("collapse_summaries", self.should_collapse)
        graph.add_edge("generate_final_summary", END)
        
        return graph

    def compile_app(self):
        """Compile and return the application graph."""
        return self.graph.compile()


if __name__ == "__main__":
    pass
