from __future__ import annotations

from mteb.abstasks import AbsTask
from mteb.abstasks.aggregated_task import AbsTaskAggregate, AggregateTaskMetadata
from mteb.tasks.RTEB import (
    RTEBAILACasedocsTask,
    RTEBAILAStatutesTask,
    RTEBAPPSTask,
    RTEBChatDoctor_HealthCareMagicTask,
    RTEBConvFinQATask,
    RTEBCOVID_QATask,
    RTEBDialogsumGermanTask,
    RTEBDS1000Task,
    RTEBFinanceBenchTask,
    RTEBFinQATask,
    RTEBFiQAPersonalFinanceTask,
    RTEBFrenchBoolQTask,
    RTEBFrenchOpenFiscalTextsTask,
    RTEBFrenchTriviaQAWikicontextTask,
    RTEBGermanLegalSentencesTask,
    RTEBGithubTask,
    RTEBHC3FinanceTask,
    RTEBHealthCareGermanTask,
    RTEBHumanEvalTask,
    RTEBJapaneseCoNaLaTask,
    RTEBJapanLawTask,
    RTEBLegalQuADTask,
    RTEBLegalSummarizationTask,
    RTEBMBPPTask,
    RTEBTAT_QATask,
    RTEBWikiSQLTask,
)

task_list_rteb: list[AbsTask] = [
    RTEBAILACasedocsTask(),
    RTEBAILAStatutesTask(),
    RTEBAPPSTask(),
    RTEBChatDoctor_HealthCareMagicTask(),
    RTEBConvFinQATask(),
    RTEBCOVID_QATask(),
    RTEBDialogsumGermanTask(),
    RTEBDS1000Task(),
    RTEBFinanceBenchTask(),
    RTEBFinQATask(),
    RTEBFiQAPersonalFinanceTask(),
    RTEBFrenchBoolQTask(),
    RTEBFrenchOpenFiscalTextsTask(),
    RTEBFrenchTriviaQAWikicontextTask(),
    RTEBGermanLegalSentencesTask(),
    RTEBGithubTask(),
    RTEBHC3FinanceTask(),
    RTEBHealthCareGermanTask(),
    RTEBHumanEvalTask(),
    RTEBJapaneseCoNaLaTask(),
    RTEBJapanLawTask(),
    RTEBLegalQuADTask(),
    RTEBLegalSummarizationTask(),
    RTEBMBPPTask(),
    RTEBTAT_QATask(),
    RTEBWikiSQLTask(),
]


class RTEBAggregatedTask(AbsTaskAggregate):
    metadata = AggregateTaskMetadata(
        name="RTEBAggregatedTask",
        description="Aggregated task for all RTEB tasks",
        reference=None,
        tasks=task_list_rteb,
        main_score="average_score",
        type="Aggregated",
        eval_splits=["test"],
        bibtex_citation=None,
    )
