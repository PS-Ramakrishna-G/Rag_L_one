import unittest

from back_end.ingestion.hr_policy_pipeline import (
    extract_chapter_title,
    build_parent_child_chunks,
    build_pinecone_records,
)


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


if __name__ == "__main__":
    unittest.main()
