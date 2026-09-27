"""Tests for the career planner. No network and no database required.

The most valuable case here is `test_recomputed_gpa_matches_stored_final_gpa`. The
dataset documents its derived columns as agreeing with their sources by
construction, so rebuilding a GPA by walking the transcript forward and comparing
it to the stored `final_gpa` checks the whole as-of-term accumulation at once. If
that drifts, every feature the models see is wrong.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from model import actions, data, datasets, skills, state as state_module, terms


class TermTest(unittest.TestCase):
    def test_round_trip(self) -> None:
        for label in ("Spring 2015", "Summer 2020", "Fall 2026"):
            self.assertEqual(terms.term_label(terms.term_index(label)), label)

    def test_ordering_within_year(self) -> None:
        spring = terms.term_index("Spring 2024")
        summer = terms.term_index("Summer 2024")
        fall = terms.term_index("Fall 2024")
        self.assertLess(spring, summer)
        self.assertLess(summer, fall)
        self.assertLess(fall, terms.term_index("Spring 2025"))

    def test_next_term_skips_summer(self) -> None:
        after_spring = terms.next_term(terms.term_index("Spring 2026"))
        self.assertEqual(terms.term_label(after_spring), "Fall 2026")

    def test_date_lands_in_right_term(self) -> None:
        self.assertEqual(terms.term_label(terms.index_from_date("2021-01-15")), "Spring 2021")
        self.assertEqual(terms.term_label(terms.index_from_date("2021-09-01")), "Fall 2021")


class SkillTest(unittest.TestCase):
    def test_shared_vocabulary(self) -> None:
        # The course catalog and the employment table are documented as drawing on
        # one vocabulary of 119 skills; that is what makes a transcript and a job
        # posting directly comparable.
        self.assertEqual(skills.size(), 119)

    def test_role_skills_are_a_subset_of_course_skills(self) -> None:
        catalog = set()
        for value in data.load("course_catalog")["skill_tags"]:
            catalog.update(data.split_pipe(value))
        roles = set()
        for value in data.load("employment_history")["role_skill_tags"]:
            roles.update(data.split_pipe(value))
        self.assertTrue(roles <= catalog)

    def test_internship_profile_is_stronger_than_a_club(self) -> None:
        internship = skills.experience_profile("Cybersecurity Intern", "Internship")
        club = skills.experience_profile("Bitcamp", "Hackathon")
        self.assertGreater(internship.sum(), club.sum())

    def test_coverage_and_missing_are_complementary(self) -> None:
        target = skills.role_profile("Software Engineering", "Entry")
        held = np.zeros(skills.size(), dtype=np.float32)
        self.assertEqual(skills.coverage(held, target), 0.0)
        self.assertTrue(skills.missing(held, target))
        self.assertEqual(skills.coverage(target, target), 1.0)
        self.assertEqual(skills.missing(target, target), [])


class ActionSpaceTest(unittest.TestCase):
    def test_every_action_is_nameable(self) -> None:
        for action in actions.action_space():
            self.assertTrue(action.position)
            self.assertTrue(action.action_id)

    def test_prerequisites_block_a_freshman(self) -> None:
        # CMSC202 requires CMSC201, so it must not be offered to someone who has
        # not taken it.
        advanced = actions.lookup("Course|CMSC202")
        assert advanced is not None
        freshman = state_module.State(
            major="Computer Science", term_index=terms.term_index("Fall 2026")
        )
        self.assertFalse(actions.feasible(advanced, freshman))
        freshman.completed_courses.add("CMSC201")
        self.assertTrue(actions.feasible(advanced, freshman))

    def test_transfer_students_are_not_blocked_by_absent_prerequisites(self) -> None:
        # A transfer student's earlier coursework has no transcript rows at all, so
        # enforcing the chain against it would lock them out permanently.
        advanced = actions.lookup("Course|CMSC202")
        assert advanced is not None
        transfer = state_module.State(
            major="Computer Science",
            entry_type="Transfer",
            term_index=terms.term_index("Fall 2026"),
        )
        self.assertTrue(actions.feasible(advanced, transfer))

    def test_completed_courses_are_not_recommended_again(self) -> None:
        intro = actions.lookup("Course|CMSC201")
        assert intro is not None
        student = state_module.State(
            major="Computer Science", term_index=terms.term_index("Fall 2026")
        )
        self.assertTrue(actions.feasible(intro, student))
        student.completed_courses.add("CMSC201")
        self.assertFalse(actions.feasible(intro, student))

    def test_roles_require_graduation(self) -> None:
        role = next(action for action in actions.action_space() if action.is_role)
        student = state_module.State(term_index=terms.term_index("Fall 2026"))
        self.assertFalse(actions.feasible(role, student))

    def test_courses_respect_terms_offered(self) -> None:
        for action in actions.action_space():
            if not action.is_course or not action.terms_offered:
                continue
            if "Spring" in action.terms_offered or not action.prerequisites:
                continue
            student = state_module.State(
                major="Computer Science",
                entry_type="Transfer",
                term_index=terms.term_index("Spring 2027"),
            )
            self.assertFalse(actions.feasible(action, student))
            break


class StateTest(unittest.TestCase):
    def test_recomputed_gpa_matches_stored_final_gpa(self) -> None:
        alumni = data.load("alumni").set_index("campus_id")
        timelines = datasets.timelines()
        checked = 0
        for campus_id in list(timelines)[:400]:
            timeline = timelines[campus_id]
            if not timeline.is_alumnus or timeline.entry_type == "Transfer":
                continue
            final = datasets.final_academic_state(timeline)
            expected = float(alumni.loc[campus_id, "final_gpa"])
            self.assertIsNotNone(final.gpa)
            self.assertAlmostEqual(final.gpa, expected, places=1)
            checked += 1
        self.assertGreater(checked, 20)

    def test_skills_never_decrease_through_a_transition(self) -> None:
        student = state_module.State(
            major="Computer Science", term_index=terms.term_index("Fall 2026")
        )
        for action in list(actions.candidates(student))[:40]:
            after = state_module.transition(student, action)
            self.assertTrue(np.all(after.skill_vector >= student.skill_vector))

    def test_transition_advances_the_term_and_leaves_the_original_alone(self) -> None:
        student = state_module.State(
            major="Computer Science", term_index=terms.term_index("Fall 2026")
        )
        before = student.skill_vector.copy()
        action = actions.lookup("Course|CMSC201")
        assert action is not None
        after = state_module.transition(student, action)
        self.assertGreater(after.term_index, student.term_index)
        self.assertIn("CMSC201", after.completed_courses)
        self.assertNotIn("CMSC201", student.completed_courses)
        np.testing.assert_array_equal(student.skill_vector, before)

    def test_features_carry_no_end_of_degree_columns(self) -> None:
        # These describe a graduate, not a student mid-degree. Letting one reach the
        # feature matrix would leak the answer and make the model useless in
        # practice while looking excellent in evaluation.
        leaks = {
            "final_gpa",
            "time_to_degree_years",
            "total_credits_earned",
            "months_to_first_job",
            "first_job_found_via",
            "first_job_title",
            "first_job_annual_salary_usd",
            "graduation_year",
            "graduation_term",
        }
        features = set(state_module.state_features(state_module.State()))
        self.assertEqual(features & leaks, set())

    def test_from_request_honours_reported_gpa_and_credits(self) -> None:
        student = state_module.from_request(
            major="Computer Science",
            track="Data Science",
            gpa=3.25,
            credits_earned=75,
            completed_courses=["CMSC201", "Discrete Structures"],
            experiences=[],
            current_term="Fall 2026",
            entry_term="Fall 2024",
        )
        self.assertAlmostEqual(student.gpa, 3.25, places=2)
        self.assertEqual(student.credits_earned, 75)
        self.assertEqual(student.class_level, "Junior")
        # Given as a title rather than an id, and still resolved.
        self.assertIn("CMSC203", student.completed_courses)


class OutcomeLabelTest(unittest.TestCase):
    def test_unknown_outcomes_are_excluded(self) -> None:
        alumni = data.load("alumni")
        unknown = set(
            alumni.loc[alumni["first_destination"] == "No Response", "campus_id"]
        )
        self.assertTrue(unknown)
        self.assertEqual(unknown & set(datasets.outcomes()), set())

    def test_non_employment_outcomes_are_kept_as_labels(self) -> None:
        families = {record["outcome_family"] for record in datasets.outcomes().values()}
        self.assertIn("Continuing Education", families)
        self.assertIn("Still Seeking", families)

    def test_salary_is_deflated_to_2026_dollars(self) -> None:
        index = data.wage_index()
        self.assertAlmostEqual(index[2026], 1.0, places=6)
        self.assertLess(index[2015], 1.0)
        nominal = 80000
        self.assertGreater(data.to_real_dollars(nominal, 2015), nominal)
        self.assertAlmostEqual(data.to_real_dollars(nominal, 2026), nominal, places=2)

    def test_splits_do_not_share_people(self) -> None:
        frame = datasets.outcome_dataset()
        per_person = frame.groupby("campus_id")["split"].nunique()
        self.assertEqual(int(per_person.max()), 1)


class PlannerTest(unittest.TestCase):
    """Only runs when the models have been trained."""

    @classmethod
    def setUpClass(cls) -> None:
        from model.train_next_step import MODEL_PATH as next_step_path
        from model.train_outcome import MODEL_PATH as outcome_path

        if not next_step_path.exists() or not outcome_path.exists():
            raise unittest.SkipTest(
                "train the models first: python -m model.train_outcome "
                "&& python -m model.train_next_step"
            )
        from model.planner import Planner

        cls.planner = Planner(explain=False)
        cls.student = state_module.from_request(
            major="Computer Science",
            track="Cybersecurity",
            gpa=3.1,
            credits_earned=68,
            completed_courses=["CMSC201", "CMSC202", "CMSC203", "CMSC341", "MATH151"],
            experiences=[],
            current_term="Fall 2026",
            entry_term="Fall 2024",
        )
        cls.goal = state_module.Goal(
            job_family="Cybersecurity",
            industry="Financial Services",
            salary_usd=115000,
            seniority="Senior",
        )

    def test_recommendations_are_feasible_and_named(self) -> None:
        steps = self.planner.recommend(self.student, self.goal, depth=1)
        self.assertTrue(steps)
        for step in steps:
            action = actions.lookup(step.action_id)
            assert action is not None
            self.assertTrue(actions.feasible(action, self.student))
            self.assertTrue(step.position)
            low, median, high = step.projected_salary
            self.assertLessEqual(low, median)
            self.assertLessEqual(median, high)

    def test_recommendations_are_not_all_the_same_kind(self) -> None:
        steps = self.planner.recommend(self.student, self.goal, depth=1)
        self.assertGreater(len({step.action_type for step in steps}), 1)

    def test_a_goal_pulls_recommendations_toward_itself(self) -> None:
        security = self.planner.recommend(self.student, self.goal, depth=1)
        analytics = self.planner.recommend(
            self.student, state_module.Goal(job_family="Data & Analytics"), depth=1
        )
        self.assertNotEqual(
            {step.action_id for step in security},
            {step.action_id for step in analytics},
        )

    def test_depth_produces_a_branching_plan(self) -> None:
        steps = self.planner.recommend(self.student, self.goal, depth=2)
        self.assertTrue(any(step.next_steps for step in steps))

    def test_evidence_points_at_real_people(self) -> None:
        outcomes = datasets.outcomes()
        steps = self.planner.recommend(self.student, self.goal, depth=1)
        supported = 0
        for step in steps:
            for campus_id in step.evidence.example_campus_ids:
                self.assertIn(campus_id, outcomes)
                supported += 1
        self.assertGreater(supported, 0)

    def test_stated_industry_is_respected_when_available(self) -> None:
        steps = self.planner.recommend(self.student, self.goal, depth=1)
        with_industry = [step for step in steps if step.industry]
        for step in with_industry:
            self.assertEqual(step.industry, self.goal.industry)


class GoalResolutionTest(unittest.TestCase):
    def test_free_text_maps_onto_families_that_exist(self) -> None:
        from api import resolve_goal

        cases = {
            "Machine Learning Engineer": "Machine Learning & AI",
            "ML engineer": "Machine Learning & AI",
            "senior cybersecurity analyst": "Cybersecurity",
            "cloud engineer": "Infrastructure & Cloud",
            "help desk": "IT Support & Operations",
            "health informatics": "Health IT",
        }
        for text, expected in cases.items():
            self.assertEqual(resolve_goal(text).job_family, expected, msg=text)

    def test_seniority_is_read_from_the_title(self) -> None:
        from api import resolve_goal

        self.assertEqual(resolve_goal("senior data analyst").seniority, "Senior")
        self.assertEqual(resolve_goal("director of data").seniority, "Director")
        self.assertEqual(resolve_goal("data analyst").seniority, "Entry")

    def test_industry_is_matched_loosely_but_not_invented(self) -> None:
        from api import resolve_goal

        self.assertEqual(
            resolve_goal("data analyst", "healthcare").industry,
            "Healthcare & Life Sciences",
        )
        self.assertIsNone(resolve_goal("data analyst", "zzzz nonexistent").industry)


if __name__ == "__main__":
    unittest.main(verbosity=2)
