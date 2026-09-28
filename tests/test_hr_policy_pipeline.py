import unittest

from back_end.ingestion.hr_policy_pipeline import (
    extract_chapter_title,
    build_parent_child_chunks,
    build_pinecone_records,
)
from back_end.retival.retrieval_service import _should_refuse_authority_question


class HRPolicyPipelineTests(unittest.TestCase):
    def test_extract_chapter_title(self):
        text = "CHAPTER 3 RECRUITMENT POLICY\n\nMANPOWER REQUISITION"
        self.assertEqual(extract_chapter_title(text), "RECRUITMENT POLICY")

    def test_extract_chapter_title_with_section_number(self):
        text = "CHAPTER 5\nCONTRACTS AND SERVICES"
        self.assertEqual(extract_chapter_title(text), "CONTRACTS AND SERVICES")

    def test_build_parent_child_chunks(self):
        sections = [
            {
                "title": "CHAPTER 3 - RECRUITMENT POLICY",
                "content": "Manpower requisition. The company shall advertise open positions. Interview panels will evaluate candidates.",
                "page_number": 12,
            },
            {
                "title": "CHAPTER 3 - RECRUITMENT POLICY",
                "content": "The selection process includes written tests and document verification. Final approval is by the department head.",
                "page_number": 13,
            },
        ]

        chunks = build_parent_child_chunks(sections, child_token_target=30, parent_token_target=80)
        self.assertTrue(len(chunks) >= 2)
        self.assertTrue(any(chunk["parent_title"] == "RECRUITMENT POLICY" for chunk in chunks))
        self.assertTrue(all("child_text" in chunk for chunk in chunks))

    def test_build_pinecone_records(self):
        chunks = [
            {
                "chunk_id": "chunk_1",
                "parent_title": "RECRUITMENT POLICY",
                "section_title": "CHAPTER 3 - RECRUITMENT POLICY",
                "page_number": 12,
                "child_text": "The company shall advertise open positions.",
            }
        ]

        vectors = build_pinecone_records(chunks, target_dimension=512)
        self.assertEqual(vectors[0]["id"], "chunk_1")
        self.assertIn("values", vectors[0])
        self.assertIn("metadata", vectors[0])
        self.assertEqual(vectors[0]["metadata"]["page_number"], 12)
        self.assertEqual(len(vectors[0]["values"]), 512)

    def test_logical_parent_child_metadata_for_sections(self):
        sections = [
            {
                "chapter": "Leave and Attendance",
                "section": "5.1 Casual Leave",
                "title": "CHAPTER 6 - LEAVE AND ATTENDANCE",
                "content": "Casual leave is admissible for 8 days in a calendar year. Employees may apply in advance.\n\nThe leave is credited at the start of the year.",
                "page_number": 69,
            }
        ]

        chunks = build_parent_child_chunks(sections, child_token_target=20, parent_token_target=80)
        self.assertTrue(chunks)
        self.assertIn("parent_id", chunks[0])
        self.assertIn("document_id", chunks[0])
        self.assertEqual(chunks[0]["chapter"], "Leave and Attendance")
        self.assertEqual(chunks[0]["section"], "5.1 Casual Leave")
        self.assertEqual(chunks[0]["page_start"], 69)
        self.assertEqual(chunks[0]["page_end"], 69)

    def test_refuse_vague_authority_question_without_explicit_policy_evidence(self):
        question = "if i sick leave whome i need to ask"
        matches = [
            {
                "metadata": {
                    "child_text": "The leave policy covers annual leave, maternity leave, and sick leave benefits."
                }
            }
        ]
        self.assertTrue(_should_refuse_authority_question(question, matches))

    def test_accept_explicit_authority_instruction_when_policy_names_the_contact(self):
        question = "if i sick leave whome i need to ask"
        matches = [
            {
                "metadata": {
                    "child_text": "Employees should contact the Director for sick leave approval and follow up with the reporting officer."
                }
            }
        ]
        self.assertFalse(_should_refuse_authority_question(question, matches))


if __name__ == "__main__":
    unittest.main()
