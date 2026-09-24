from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from back_end.retival.retrieval_service import answer_question, query_policy_store


EVAL_DATA = [
    {
        "question": "What are the normal office hours at the Institute?",
        "ground_truth": "9:00 AM to 5:45 PM, Monday through Friday.",
    },
    {
        "question": "How much grace time is allowed when an employee arrives late?",
        "ground_truth": "15 minutes for unforeseen delays in the morning, although frequent use of the grace period may not be condoned.",
    },
    {
        "question": "What happens if an employee reports late for the third time in a month?",
        "ground_truth": "Half a day of Casual Leave is debited.",
    },
    {
        "question": "What happens if the employee has no Casual Leave balance when penalized for late attendance?",
        "ground_truth": "Earned Leave is deducted.",
    },
    {
        "question": "What are the specified lunch-break hours?",
        "ground_truth": "1:00 PM to 1:45 PM.",
    },
    {
        "question": "How should employees record their attendance?",
        "ground_truth": "Staff should punch their presence using the biometric terminal nearest their department.",
    },
    {
        "question": "Who determines working hours for shift-based staff?",
        "ground_truth": "The Head of Department decides them according to the nature of work, and different shift timings should be shared with HR by the first day of every month.",
    },
    {
        "question": "What can happen in cases of unauthorized absence?",
        "ground_truth": "The HOD must inform HR; if unauthorized absence is not condoned, it can result in a break in service.",
    },
    {
        "question": "How many days of maternity leave are allowed for pregnancy?",
        "ground_truth": "180 days.",
    },
    {
        "question": "Is maternity leave available to an employee who already has two surviving children?",
        "ground_truth": "No. The 180-day pregnancy maternity leave provision is admissible only to employees with fewer than two surviving children.",
    },
    {
        "question": "How much leave is available for miscarriage or abortion?",
        "ground_truth": "45 days during the entire service, supported by a certificate from a Registered Medical Practitioner.",
    },
    {
        "question": "Is maternity leave deducted from the employee's leave account?",
        "ground_truth": "No. It is not debited to the leave account.",
    },
    {
        "question": "Is maternity leave granted on full pay?",
        "ground_truth": "Yes, maternity leave is granted on full pay.",
    },
    {
        "question": "Can maternity leave be combined with another type of leave?",
        "ground_truth": "Yes. It may be combined with leave of another kind.",
    },
    {
        "question": "What documents must an employee submit after maternity leave?",
        "ground_truth": "Hospital discharge certificate and a copy of the child's birth certificate.",
    },
    {
        "question": "How many days of paternity leave can an eligible employee receive?",
        "ground_truth": "15 days.",
    },
    {
        "question": "When can paternity leave be taken relative to the child's birth?",
        "ground_truth": "Up to 15 days before childbirth or up to six months from the date of delivery.",
    },
    {
        "question": "Is paternity leave debited from the employee's leave account?",
        "ground_truth": "No.",
    },
    {
        "question": "Can paternity leave be combined with Casual Leave?",
        "ground_truth": "No. It may be combined with other leave, except Casual Leave.",
    },
    {
        "question": "What documents are required when applying for paternity leave?",
        "ground_truth": "The wife's hospital discharge certificate and a copy of the baby's birth certificate.",
    },
    {
        "question": "How much child adoption leave can an eligible female employee receive?",
        "ground_truth": "180 days immediately after valid adoption for an eligible female employee with fewer than two surviving children.",
    },
    {
        "question": "What age must the adopted child be for child adoption leave?",
        "ground_truth": "Below one year.",
    },
    {
        "question": "Can child adoption leave be combined with another kind of leave?",
        "ground_truth": "Yes, with leave of another kind.",
    },
    {
        "question": "What is the minimum service required for an employee to be considered for performance review?",
        "ground_truth": "Six months.",
    },
    {
        "question": "When does the annual performance appraisal process normally start?",
        "ground_truth": "Every year during June, or when an employee completes their term as a probationer or otherwise.",
    },
    {
        "question": "What period does the annual performance evaluation cover for permanent employees?",
        "ground_truth": "Annually from 1 July through 30 June.",
    },
    {
        "question": "Who normally writes the appraiser evaluation?",
        "ground_truth": "The reporting manager immediately superior to the employee writes the appraiser evaluation, and the next higher authority reviews it.",
    },
    {
        "question": "What is expected from an employee during a performance review meeting?",
        "ground_truth": "The appraisee is required to make an honest self-appraisal before discussing performance with the supervisor.",
    },
    {
        "question": "How many justifications must an HOD provide for a performance rating?",
        "ground_truth": "At least three justifications for the rating.",
    },
    {
        "question": "Who is the appointing authority for administrative staff in Pay Level 11 and above?",
        "ground_truth": "The Director.",
    },
    {
        "question": "Who is the appointing authority for administrative staff in Pay Levels 6 through 10?",
        "ground_truth": "Chief Administrative Officer, under authority delegated by the Director.",
    },
    {
        "question": "Who is the appointing authority for administrative staff in Pay Level 5 and below?",
        "ground_truth": "Associate Vice President – HR, under authority sub-delegated by the Director.",
    },
    {
        "question": "What type of appointment is initially used for positions covered by the appointment policy?",
        "ground_truth": "Tenure Based Scaled Contract.",
    },
    {
        "question": "Are employees on Tenure Based Scaled Contract entitled to leave similar to permanent employees?",
        "ground_truth": "Yes. TBSC employees are given leave as provided to permanent employees and are also eligible for benefits such as LTC and dispensary facilities.",
    },
    {
        "question": "What medical examination is required before final selection?",
        "ground_truth": "Medical checkup may be performed by the Institute's doctor, or the candidate can provide a fitness certificate from a Civil Hospital; the Institute doctor may require further tests.",
    },
    {
        "question": "What documents are verified during the joining procedure?",
        "ground_truth": "Educational/professional mark sheets and passing certificates, birth certificate, address proof, previous employer relieving letter and latest salary slip, medical fitness certificate, specified government-issued photo ID, and photographs.",
    },
    {
        "question": "When does an employee's monthly salary become due and payable?",
        "ground_truth": "On the last working day of each month.",
    },
    {
        "question": "What methods may be used during the interview and selection process?",
        "ground_truth": "The structure may include a skill test, personal interview and/or group discussion, debate, or quiz.",
    },
    {
        "question": "What travel reimbursement is available to candidates interviewing for Manager and above positions?",
        "ground_truth": "Economy air travel fare for the shortest distance is reimbursed for Manager and above positions.",
    },
    {
        "question": "What are the relocation reimbursement limits for Groups A, B, and C?",
        "ground_truth": "Group A up to ₹40,000; Group B up to ₹30,000; Group C up to ₹20,000, or actual cost if lower, subject to the stated conditions.",
    },
    {
        "question": "What happens to relocation reimbursement if an employee leaves the Institute within one year?",
        "ground_truth": "The entire relocation amount paid is recovered.",
    },
    {
        "question": "What can employees do through the ESS leave-management system?",
        "ground_truth": "Apply for leave, view leave history, cancel applied leave, and check leave balances; leave actions go to the supervisor for approval.",
    },
    {
        "question": "What is the first step an employee should take when submitting a grievance?",
        "ground_truth": "Submit the grievance in writing to their supervisor or Head of Department.",
    },
    {
        "question": "Who is the final authority on staff and Managers' grievance matters?",
        "ground_truth": "The Director.",
    },
    {
        "question": "How long are Category A, Category B, and Category C HR records retained?",
        "ground_truth": "Category A records are kept permanently, Category B for five years, and Category C for two years.",
    },
    {
        "question": "What is annual leave?",
        "ground_truth": "Annual leave is the yearly leave entitlement described in the policy; exact eligibility and rules depend on the section governing leave balances and encashment.",
    },
    {
        "question": "What is the difference between Casual Leave and Earned Leave?",
        "ground_truth": "Casual Leave is a short-term leave category usually used for short absences and is penalized differently when not available, while Earned Leave is a leave category that can be deducted or encashed under the policy conditions described in the leave rules.",
    },
    {
        "question": "Is the policy clear about leave encashment?",
        "ground_truth": "The policy gives specific conditions for encashment, including the number of days, eligibility, and minimum balance requirements; it does not permit unrestricted encashment.",
    },
]


def evaluate_rag() -> list[dict[str, Any]]:
    report = []
    for item in EVAL_DATA:
        question = item["question"]
        matches = query_policy_store(question, top_k=5)
        result = answer_question(question, top_k=5)
        report.append(
            {
                "question": question,
                "ground_truth": item["ground_truth"],
                "retrieved_matches": len(matches),
                "retrieved_sources": matches,
                "answer": result.get("answer"),
                "db_proof": result.get("db_proof", {}),
            }
        )
    return report


if __name__ == "__main__":
    data = evaluate_rag()
    output_path = Path(__file__).resolve().parent.parent.parent / "artifacts" / "rag_evaluation.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved evaluation report to: {output_path}")
