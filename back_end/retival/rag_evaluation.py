from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from back_end.retival.retrieval_service import answer_question

OUTPUT_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "rag_evaluation_results.json"
)


TEST_CASES = [
    {
        "id": 1,
        "question": "What are the normal office hours at the Institute?",
        "expected_keywords": ["9:00", "5:45"],
    },
    {
        "id": 2,
        "question": "How much grace time is allowed when an employee arrives late?",
        "expected_keywords": ["15 minutes"],
    },
    {
        "id": 3,
        "question": "What happens if an employee reports late for the third time in a month?",
        "expected_keywords": ["half", "casual leave"],
    },
    {
        "id": 4,
        "question": "What happens if the employee has no Casual Leave balance when penalized for late attendance?",
        "expected_keywords": ["earned leave"],
    },
    {
        "id": 5,
        "question": "What are the specified lunch-break hours?",
        "expected_keywords": ["1:00", "1:45"],
    },
    {
        "id": 6,
        "question": "How should employees record their attendance?",
        "expected_keywords": ["biometric"],
    },
    {
        "id": 7,
        "question": "Who determines working hours for shift-based staff?",
        "expected_keywords": ["head of department"],
    },
    {
        "id": 8,
        "question": "What can happen in cases of unauthorized absence?",
        "expected_keywords": ["break in service"],
    },
    {
        "id": 9,
        "question": "How many days of maternity leave are allowed for pregnancy?",
        "expected_keywords": ["180 days"],
    },
    {
        "id": 10,
        "question": "Is maternity leave available to an employee who already has two surviving children?",
        "expected_keywords": ["less than two surviving children"],
    },
    {
        "id": 11,
        "question": "How much leave is available for miscarriage or abortion?",
        "expected_keywords": ["45 days"],
    },
    {
        "id": 12,
        "question": "Is maternity leave deducted from the employee's leave account?",
        "expected_keywords": ["not debited"],
    },
    {
        "id": 13,
        "question": "Is maternity leave granted on full pay?",
        "expected_keywords": ["full pay"],
    },
    {
        "id": 14,
        "question": "Can maternity leave be combined with another type of leave?",
        "expected_keywords": ["combined"],
    },
    {
        "id": 15,
        "question": "What documents must an employee submit after maternity leave?",
        "expected_keywords": [
            "discharge certificate",
            "birth certificate",
        ],
    },
    {
        "id": 16,
        "question": "How many days of paternity leave can an eligible employee receive?",
        "expected_keywords": ["15 days"],
    },
    {
        "id": 17,
        "question": "When can paternity leave be taken relative to the child's birth?",
        "expected_keywords": [
            "15 days before",
            "six months",
        ],
    },
    {
        "id": 18,
        "question": "Is paternity leave debited from the employee's leave account?",
        "expected_keywords": ["not debited"],
    },
    {
        "id": 19,
        "question": "Can paternity leave be combined with Casual Leave?",
        "expected_keywords": ["except casual leave"],
    },
    {
        "id": 20,
        "question": "What documents are required when applying for paternity leave?",
        "expected_keywords": [
            "discharge certificate",
            "birth certificate",
        ],
    },
    {
        "id": 21,
        "question": "What is the recruitment process?",
        "expected_keywords": [
            "manpower requisition",
            "advertisement",
            "shortlisting",
            "interview",
        ],
    },
    {
        "id": 22,
        "question": "What happens after manpower requisition is approved?",
        "expected_keywords": [
            "recruitment process",
            "HR department",
        ],
    },
    {
        "id": 23,
        "question": "Who shortlists candidates after applications are received?",
        "expected_keywords": [
            "concerned department",
            "eligibility criteria",
        ],
    },
    {
        "id": 24,
        "question": "What methods may be used during the interview and selection process?",
        "expected_keywords": [
            "skill test",
            "personal interview",
        ],
    },
    {
        "id": 25,
        "question": "What travel reimbursement is available to Manager and above candidates?",
        "expected_keywords": [
            "economy air travel",
        ],
    },
    {
        "id": 26,
        "question": "What medical examination is required before final selection?",
        "expected_keywords": [
            "fitness certificate",
            "civil hospital",
        ],
    },
    {
        "id": 27,
        "question": "What documents are checked during joining?",
        "expected_keywords": [
            "mark sheets",
            "birth certificate",
            "identity proof",
        ],
    },
    {
        "id": 28,
        "question": "When is monthly salary payable?",
        "expected_keywords": [
            "last working day",
        ],
    },
    {
        "id": 29,
        "question": "When does the annual performance appraisal process start?",
        "expected_keywords": ["June"],
    },
    {
        "id": 30,
        "question": "What minimum service is required for performance review?",
        "expected_keywords": ["six months"],
    },
    {
        "id": 31,
        "question": "What period does the annual performance evaluation cover?",
        "expected_keywords": [
            "1st July",
            "30th June",
        ],
    },
    {
        "id": 32,
        "question": "Who writes the appraiser evaluation?",
        "expected_keywords": [
            "reporting manager",
        ],
    },
    {
        "id": 33,
        "question": "How long must a manager have supervised an employee for appraisal purposes?",
        "expected_keywords": [
            "three months",
        ],
    },
    {
        "id": 34,
        "question": "How many justifications must an HOD give for a performance rating?",
        "expected_keywords": ["three"],
    },
    {
        "id": 35,
        "question": "What is the grievance process?",
        "expected_keywords": [
            "supervisor",
            "grievance manager",
            "committee",
        ],
    },
    {
        "id": 36,
        "question": "What is the first step for an employee submitting a grievance?",
        "expected_keywords": [
            "supervisor",
            "head of department",
        ],
    },
    {
        "id": 37,
        "question": "Who is the final authority on grievance matters?",
        "expected_keywords": ["director"],
    },
    {
        "id": 38,
        "question": "How quickly should the first grievance handling stage ideally be completed?",
        "expected_keywords": ["two weeks"],
    },
    {
        "id": 39,
        "question": "How often does the grievance committee report to the Director?",
        "expected_keywords": ["quarterly"],
    },
    {
        "id": 40,
        "question": "What is the retirement age for staff?",
        "expected_keywords": ["60 years"],
    },
    {
        "id": 41,
        "question": "Can an employee engage in outside business without permission?",
        "expected_keywords": [
            "written permission",
            "director",
        ],
    },
    {
        "id": 42,
        "question": "What proof of age can employees provide?",
        "expected_keywords": [
            "birth certificate",
            "school leaving certificate",
            "passport",
        ],
    },
    {
        "id": 43,
        "question": "Can non-faculty employees be transferred between departments?",
        "expected_keywords": [
            "transferred",
            "department",
        ],
    },
    {
        "id": 44,
        "question": "What is the policy on confidential Institute information?",
        "expected_keywords": [
            "disclose",
            "prior approval",
        ],
    },
    {
        "id": 45,
        "question": "How many days of Casual Leave are allowed in a calendar year?",
        "expected_keywords": ["eight days"],
    },
    {
        "id": 46,
        "question": "How far in advance should planned leave of more than two days be requested?",
        "expected_keywords": ["10 days"],
    },
    {
        "id": 47,
        "question": "Can leave be claimed as a matter of right?",
        "expected_keywords": ["cannot be claimed"],
    },
    {
        "id": 48,
        "question": "What happens if leave is extended without proper sanction?",
        "expected_keywords": [
            "half pay leave",
            "extra ordinary leave",
        ],
    },
    {
        "id": 49,
        "question": "What is Leave Travel Concession eligibility based on?",
        "expected_keywords": [
            "one year",
            "continuous service",
        ],
    },
    {
        "id": 50,
        "question": "What is this HR Policy Manual about?",
        "expected_keywords": [
            "HR policies",
            "procedures",
            "employees",
        ],
    },
]


def normalize_text(value: Any) -> str:
    return str(value or "").lower().strip()


def keyword_score(
    answer: str,
    expected_keywords: list[str],
) -> tuple[int, int, float]:

    answer_lower = normalize_text(answer)

    found = 0

    for keyword in expected_keywords:
        if normalize_text(keyword) in answer_lower:
            found += 1

    total = len(expected_keywords)

    ratio = (
        found / total
        if total
        else 0.0
    )

    return found, total, ratio


def source_summary(
    sources: list[dict[str, Any]],
) -> list[dict[str, Any]]:

    output = []

    for source in sources:
        metadata = source.get("metadata", source)

        output.append(
            {
                "id": source.get("id"),
                "score": source.get("score"),
                "page_number": metadata.get("page_number"),
                "parent_title": metadata.get("parent_title"),
                "section_title": metadata.get("section_title"),
                "child_text": (
                    metadata.get("child_text")
                    or ""
                )[:1000],
            }
        )

    return output


def main() -> None:

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results = []

    total_questions = len(TEST_CASES)

    print("=" * 100)
    print("RAG EVALUATION")
    print("=" * 100)
    print(f"Questions: {total_questions}")
    print()

    passed = 0
    source_hits = 0
    total_latency = 0.0

    for index, case in enumerate(
        TEST_CASES,
        start=1,
    ):

        question = case["question"]

        print(
            f"[{index}/{total_questions}] "
            f"{question}"
        )

        started = time.perf_counter()

        try:
            response = answer_question(
                question,
                top_k=5,
            )

            latency = (
                time.perf_counter()
                - started
            )

            total_latency += latency

            answer = response.get(
                "answer",
                "",
            )

            sources = response.get(
                "sources",
                [],
            ) or []

            found, expected, ratio = keyword_score(
                answer,
                case["expected_keywords"],
            )

            keyword_pass = (
                ratio >= 0.5
            )

            if keyword_pass:
                passed += 1

            if sources:
                source_hits += 1

            record = {
                "id": case["id"],
                "question": question,
                "expected_keywords":
                    case["expected_keywords"],

                "answer": answer,

                "keyword_matches": found,
                "keyword_total": expected,
                "keyword_score": round(
                    ratio,
                    4,
                ),
                "keyword_pass": keyword_pass,

                "source_count": len(sources),
                "sources": source_summary(
                    sources
                ),

                "latency_seconds": round(
                    latency,
                    3,
                ),

                "db_proof": response.get(
                    "db_proof",
                    {},
                ),

                "retrieval_debug": response.get(
                    "retrieval_debug",
                    {},
                ),

                "logs": response.get(
                    "logs",
                    [],
                ),
            }

            results.append(record)

            print(
                f"  answer_score={ratio:.2f}"
                f" | sources={len(sources)}"
                f" | latency={latency:.2f}s"
            )

        except Exception as exc:

            latency = (
                time.perf_counter()
                - started
            )

            results.append(
                {
                    "id": case["id"],
                    "question": question,
                    "expected_keywords":
                        case["expected_keywords"],
                    "error": str(exc),
                    "keyword_pass": False,
                    "source_count": 0,
                    "latency_seconds": round(
                        latency,
                        3,
                    ),
                }
            )

            print(
                f"  ERROR: {exc}"
            )

    answer_accuracy = (
        passed / total_questions
        if total_questions
        else 0.0
    )

    retrieval_hit_rate = (
        source_hits / total_questions
        if total_questions
        else 0.0
    )

    average_latency = (
        total_latency / total_questions
        if total_questions
        else 0.0
    )

    summary = {
        "total_questions": total_questions,

        "keyword_passed": passed,

        "keyword_answer_accuracy":
            round(
                answer_accuracy,
                4,
            ),

        "questions_with_sources":
            source_hits,

        "retrieval_source_hit_rate":
            round(
                retrieval_hit_rate,
                4,
            ),

        "average_latency_seconds":
            round(
                average_latency,
                3,
            ),
    }

    output = {
        "summary": summary,
        "results": results,
    }

    OUTPUT_PATH.write_text(
        json.dumps(
            output,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 100)
    print("SUMMARY")
    print("=" * 100)

    print(
        f"Total questions: "
        f"{total_questions}"
    )

    print(
        f"Keyword answer accuracy: "
        f"{answer_accuracy:.2%}"
    )

    print(
        f"Questions with retrieved sources: "
        f"{source_hits}/{total_questions}"
    )

    print(
        f"Retrieval source hit rate: "
        f"{retrieval_hit_rate:.2%}"
    )

    print(
        f"Average latency: "
        f"{average_latency:.2f}s"
    )

    print()
    print(
        f"Saved results to:"
        f"\n{OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()