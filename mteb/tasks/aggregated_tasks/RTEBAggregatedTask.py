from __future__ import annotations

from mteb.abstasks import AbsTask
from mteb.abstasks.aggregated_task import AbsTaskAggregate, AggregateTaskMetadata
from mteb.tasks.RTEB.RTEBAILACasedocsTask import RTEBAILACasedocs
from mteb.tasks.RTEB.RTEBAILAStatutesTask import RTEBAILAStatutes
from mteb.tasks.RTEB.RTEBAPPSTask import RTEBAPPS
from mteb.tasks.RTEB.RTEBLegalQuADTask import RTEBLegalQuAD

# RTEBChatDoctor_HealthCareMagicTask,
# RTEBConvFinQATask,
# RTEBCOVID_QATask,
# RTEBDialogsumGermanTask,
# RTEBDS1000Task,
# RTEBFinanceBenchTask,
# RTEBFinQATask,
# RTEBFiQAPersonalFinanceTask,
# RTEBFrenchBoolQTask,
# RTEBFrenchOpenFiscalTextsTask,
# RTEBFrenchTriviaQAWikicontextTask,
# RTEBGermanLegalSentencesTask,
# RTEBGithubTask,
# RTEBHC3FinanceTask,
# RTEBHealthCareGermanTask,
# RTEBHumanEvalTask,
# RTEBJapaneseCoNaLaTask,
# RTEBJapanLawTask,
# RTEBLegalSummarizationTask,
# RTEBMBPPTask,
# RTEBTAT_QATask,
# RTEBWikiSQLTask,


task_list_rteb: list[AbsTask] = [
    RTEBAILACasedocs(),
    RTEBAILAStatutes(),
    RTEBAPPS(),
    RTEBLegalQuAD(),
    # RTEBChatDoctor_HealthCareMagic(),
    # RTEBConvFinQA(),
    # RTEBCOVID_QA(),
    # RTEBDialogsumGerman(),
    # RTEBDS1000(),
    # RTEBFinanceBench(),
    # RTEBFinQA(),
    # RTEBFiQAPersonalFinance(),
    # RTEBFrenchBoolQ(),
    # RTEBFrenchOpenFiscalTexts(),
    # RTEBFrenchTriviaQAWikicontext(),
    # RTEBGermanLegalSentences(),
    # RTEBGithub(),
    # RTEBHC3Finance(),
    # RTEBHealthCareGerman(),
    # RTEBHumanEval(),
    # RTEBJapaneseCoNaLa(),
    # RTEBJapanLaw(),
    # RTEBLegalSummarization(),
    # RTEBMBPP(),
    # RTEBTAT_QA(),
    # RTEBWikiSQL(),
]


class RTEBAggregatedTask(AbsTaskAggregate):
    metadata = AggregateTaskMetadata(
        name="RTEBAggregatedTask",
        description="Aggregated task for all RTEB tasks",
        reference=None,
        tasks=task_list_rteb,
        main_score="average_score",
        type="RTEB",
        eval_splits=["test"],
        bibtex_citation=None,
    )
