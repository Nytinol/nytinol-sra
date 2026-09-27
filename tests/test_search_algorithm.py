import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import search_algorithm
import tiger_db
from search_algorithm import Achievement, CurrentAchievements


def student(
    name: str,
    age: int,
    major: str,
    experiences: list[Achievement],
) -> CurrentAchievements:
    return CurrentAchievements(name=name, age=age, major=major, experiences=experiences)


NONE = student("Maya Ortiz", 19, "Computer Science", [])

MID = student(
    "Jordan Hale",
    21,
    "Information Systems",
    [
        Achievement("Internship", "IT Support Intern", "Software Products", "Summer 2025", 1, 15),
        Achievement("Student Organization", "Product Club", "Not Applicable", "Fall 2025", 2, 4),
        Achievement("Certification", "SQL Associate", "Not Applicable", "Spring 2025", 1, None),
    ],
)

HIGH = student(
    "Priya Shah",
    23,
    "Cybersecurity",
    [
        Achievement("Internship", "Security Operations Intern", "Cybersecurity Services", "Summer 2024", 2, 40),
        Achievement("Internship", "SOC Intern", "Federal Government", "Fall 2024", 2, 35),
        Achievement("Co-op", "Network Defense Co-op", "Defense & Aerospace", "Spring 2025", 2, 40),
        Achievement("Undergraduate Research", "Security Lab Assistant", "Not Applicable", "Fall 2023", 3, 12),
        Achievement("Hackathon", "Bitcamp", "Not Applicable", "Spring 2024", 1, 36),
        Achievement("Hackathon", "HopHacks", "Not Applicable", "Fall 2024", 1, 30),
        Achievement("Certification", "Security Analyst Associate", "Not Applicable", "Summer 2023", 1, None),
        Achievement("Certification", "Network Associate", "Not Applicable", "Fall 2023", 1, None),
        Achievement("Campus Job", "Help Desk Consultant", "Not Applicable", "Fall 2022", 4, 12),
        Achievement("Student Organization", "Cybersecurity Club", "Not Applicable", "Fall 2022", 6, 5),
    ],
)


class ExampleStudentTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.maya_title = tiger_db.get_job_vector("Data Analyst")
        cls.maya_industry = tiger_db.get_industry_vector("Insurance")
        cls.jordan_title = tiger_db.get_job_vector("Software Engineer I")
        cls.jordan_industry = tiger_db.get_industry_vector("Software Products")
        cls.priya_title = tiger_db.get_job_vector("SOC Analyst")
        cls.priya_industry = tiger_db.get_industry_vector("Cybersecurity Services")

    def _steps(
        self,
        person: CurrentAchievements,
        job_title: list[float],
        industry: list[float],
        gpa: float,
        classes: list[str],
        salary: float,
    ) -> list[str]:
        return search_algorithm.search_algorithm(
            job_title,
            person,
            gpa,
            classes,
            person.major,
            salary,
            industry,
        )

    def _print_steps(self, person: CurrentAchievements, goal: str, steps: list[str]) -> None:
        print(
            f"\n{person.name}, age {person.age}, {person.major}, {len(person.experiences)} experiences. Goal: {goal}",
            file=sys.stderr,
        )
        for number, step in enumerate(steps, start=1):
            print(f"  {number}. {step}", file=sys.stderr)

    def test_three_students_with_different_experience_and_majors(self) -> None:
        maya = self._steps(NONE, self.maya_title, self.maya_industry, 3.6, ["CMSC201"], 75000)
        jordan = self._steps(MID, self.jordan_title, self.jordan_industry, 3.1, ["IS300"], 85000)
        priya = self._steps(HIGH, self.priya_title, self.priya_industry, 2.8, ["CMSC201", "CMSC202"], 90000)
        self._print_steps(NONE, "Data Analyst in Insurance at $75,000", maya)
        self._print_steps(MID, "Software Engineer I in Software Products at $85,000", jordan)
        self._print_steps(HIGH, "SOC Analyst in Cybersecurity Services at $90,000", priya)
        self.assertEqual(len({NONE.major, MID.major, HIGH.major}), 3)
        for steps in (maya, jordan, priya):
            self.assertEqual(len(steps), 3)
        self.assertNotEqual(maya, jordan)
        self.assertNotEqual(jordan, priya)
        self.assertNotEqual(maya, priya)
        self.assertFalse(any("CMSC201" in step for step in maya))
        self.assertFalse(any("IS300" in step for step in jordan))
        for steps in (maya, jordan, priya):
            self.assertFalse(any(step.startswith("Project") or step.startswith("Build ") for step in steps))


if __name__ == "__main__":
    unittest.main()
