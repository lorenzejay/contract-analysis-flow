from crewai.utilities.events.base_event_listener import BaseEventListener
from crewai.utilities.events.flow_events import MethodExecutionFinishedEvent
from crewai.flow import Flow
from opik.evaluation.metrics import Hallucination


# docs: https://www.comet.com/docs/opik/cookbook/evaluate_hallucination_metric
class EvalListener(BaseEventListener):
    def setup_listeners(self, crewai_event_bus):
        @crewai_event_bus.on(MethodExecutionFinishedEvent)
        def on_flow_finished(source: Flow, event: MethodExecutionFinishedEvent):
            print(f"Running hallucination metric for {event.method_name}")
            if event.method_name == "generate_report":
                # adding opik here
                hallucination_metric = Hallucination(
                    model="gpt-4o",
                )
                hallucination_score = hallucination_metric.score(
                    source.state.contract_analysis,
                    event.result,
                )
                print(f"Hallucination score: {hallucination_score}")
