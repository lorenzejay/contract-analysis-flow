import os
from pydantic import BaseModel

from crewai.flow import Flow, listen, start

from contract_analysis_flow_workshop.pre_process_service import (
    ContractProcessingService,
)
from crewai_tools import WeaviateVectorSearchTool
from crewai import Agent

class Report(BaseModel):
    report: str
    source_citations: list[str]

class ContractAnalysisState(BaseModel):
    metadata_filters: str = ""
    query: str = ""
    contract_analysis: str = ""
    report: Report = Report(report="", source_citations=[])

class ContractAnalysisFlow(Flow[ContractAnalysisState]):
    inputs: dict = {"query": "how are warranties defined in digitalcinemadestination?"}

    @start()
    def pre_process_documents(self):
        print("Pre-processing documents with query", self.state.query)
        service = ContractProcessingService()
        with service:
            service.process_documents(
                folder_path="knowledge/contracts/",
            )
        self.state.query = self.inputs["query"]
        return {"query": self.state.query}

    vector_search_tool = WeaviateVectorSearchTool(
        collection_name="Contracts_business_latest_6",
        weaviate_cluster_url=os.getenv("WEAVIATE_URL"),
        weaviate_api_key=os.getenv("WEAVIATE_API_KEY"),
    )

    @listen(pre_process_documents)
    def generate_contract_analysis(self):
        print("Generating contract analysis", self.state.query)
        agent = Agent(
            role="Retrieval Agent",
            goal="Retrieve the contract analysis for the given query",
            backstory="You are a contract analysis agent that retrieves the most relevant contracts based on the given query. Be sure to include the source citations such as the file name, sections and page numbers of the contract. When you use the vector search tool, pass the argument 'query' as the query to search for.",
            tools=[self.vector_search_tool],
            verbose=True,
            llm="gpt-4o",
        )
        print("contract analysis query", self.state.query)
        query = f"Retrieve the most relevant contracts for the given query: {self.state.query}. The output should show all the retrieved data. Do not summarize the retrieved data."
        result = agent.kickoff(query)
        print("Contract analysis generated", result.raw)
        self.state.contract_analysis = result.raw
        return result.raw

    @listen(generate_contract_analysis)
    def generate_report(self):
        print("Generating report", self.state.contract_analysis)
        agent = Agent(
            role="Report Generation Agent",
            goal="Generate a report based on the contract analysis. Be sure to include the source citations such as the file name, sections and page numbers of the contract.",
            backstory="You are a report generation agent that generates a report based on the contract analysis.",
            verbose=True,
        )
        query = f"Generate a report based on the contract analysis: {self.state.contract_analysis}."
        result = agent.kickoff(query, response_format=Report)
        parsed_result = result.pydantic
        print("--------------------------------")
        print("Report generated")
        print("--------------------------------")
        print(parsed_result.report)
        print("--------------------------------")
        print("Source citations")
        print("--------------------------------")
        print(parsed_result.source_citations)
        print("--------------------------------")

        self.state.report = parsed_result
        return {
            "report": parsed_result.report,
            "source_citations": parsed_result.source_citations,
        }


def kickoff():
    contract_analysis_flow = ContractAnalysisFlow()
    contract_analysis_flow.kickoff(
        inputs={"query": "how are warranties defined in digitalcinemadestination?"}
    )

def plot():
    contract_analysis_flow = ContractAnalysisFlow()
    contract_analysis_flow.plot()

if __name__ == "__main__":
    kickoff()
