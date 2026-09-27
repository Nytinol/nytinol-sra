-- Exploring the dataset in SQL.
--
-- One of several ways in -- see explore_python.py for the same ideas in
-- pandas, and quickstart.py for a version with no dependencies at all.
-- Nothing about this dataset requires SQL.
--
-- ---------------------------------------------------------------------
-- Loading
-- ---------------------------------------------------------------------
--
-- DuckDB reads the CSVs directly, with types inferred and no import step:
--
--     SELECT * FROM 'data/alumni.csv' LIMIT 5;
--
-- SQLite needs an import first. From the repository root:
--
--     sqlite3 campus.db
--     .mode csv
--     .import data/students_current.csv    students_current
--     .import data/alumni.csv              alumni
--     .import data/transcripts.csv         transcripts
--     .import data/employment_history.csv  employment_history
--     .import data/student_experience.csv  student_experience
--     .import data/course_catalog.csv      course_catalog
--
-- SQLite stores every column as text, so CAST before doing arithmetic.
-- The queries below are written to work in either.


-- ---------------------------------------------------------------------
-- 1. Courses that gate the most other courses
--    One file. The prerequisite graph is the curriculum's real structure.
-- ---------------------------------------------------------------------

-- DuckDB:
SELECT unnest(str_split(prerequisite_ids, '|')) AS gate, COUNT(*) AS unlocks
FROM course_catalog
WHERE prerequisite_ids <> 'Not Applicable'
GROUP BY gate
ORDER BY unlocks DESC
LIMIT 10;

-- SQLite, which has no split function -- a recursive CTE does the work:
WITH RECURSIVE split(gate, rest) AS (
  SELECT '', prerequisite_ids || '|'
    FROM course_catalog WHERE prerequisite_ids <> 'Not Applicable'
  UNION ALL
  SELECT substr(rest, 1, instr(rest, '|') - 1),
         substr(rest, instr(rest, '|') + 1)
    FROM split WHERE rest <> ''
)
SELECT gate, COUNT(*) AS unlocks
FROM split WHERE gate <> ''
GROUP BY gate ORDER BY unlocks DESC LIMIT 10;


-- ---------------------------------------------------------------------
-- 2. Which courses are hardest
--    W and IP carry no grade points and are excluded from GPA.
-- ---------------------------------------------------------------------

SELECT course_id, course_title,
       ROUND(AVG(CAST(grade_points AS REAL)), 2) AS avg_points,
       COUNT(*) AS attempts
FROM transcripts
WHERE grade NOT IN ('W', 'IP')
GROUP BY course_id, course_title
HAVING COUNT(*) >= 200
ORDER BY avg_points
LIMIT 10;


-- ---------------------------------------------------------------------
-- 3. Recompute a person's GPA from their transcript
--    This matches the stored cumulative_gpa, which is a good way to
--    confirm you have understood the file.
-- ---------------------------------------------------------------------

SELECT t.campus_id,
       ROUND(SUM(CAST(t.grade_points AS REAL) * t.credits_attempted)
             / SUM(t.credits_attempted), 2) AS recomputed,
       a.final_gpa AS stored
FROM transcripts t
JOIN alumni a ON a.campus_id = t.campus_id
WHERE t.grade NOT IN ('W', 'IP')
GROUP BY t.campus_id, a.final_gpa
LIMIT 10;


-- ---------------------------------------------------------------------
-- 4. What each graduating class reported
--    One file. Note that 'No Response' is a real category, not a null.
-- ---------------------------------------------------------------------

SELECT graduation_year, first_destination, COUNT(*) AS n
FROM alumni
GROUP BY graduation_year, first_destination
ORDER BY graduation_year, n DESC;


-- ---------------------------------------------------------------------
-- 5. The career-pathway graph
--    Sort by person and date, pair each role with the next one. These
--    are the edges; job_title values are the nodes.
-- ---------------------------------------------------------------------

WITH steps AS (
  SELECT campus_id, job_title,
         LEAD(job_title) OVER (PARTITION BY campus_id ORDER BY start_date)
           AS next_role
  FROM employment_history
)
SELECT job_title, next_role, COUNT(*) AS weight
FROM steps
WHERE next_role IS NOT NULL
GROUP BY job_title, next_role
ORDER BY weight DESC
LIMIT 20;


-- ---------------------------------------------------------------------
-- 6. Salary in real terms
--    cost_of_living_index is inline (100 = national average), so this
--    needs no join. Whether it is the right adjustment is your call.
-- ---------------------------------------------------------------------

SELECT region,
       COUNT(*) AS spells,
       CAST(AVG(CAST(annual_salary_usd AS REAL)) AS INT) AS nominal,
       CAST(AVG(CAST(annual_salary_usd AS REAL)
                / (CAST(cost_of_living_index AS REAL) / 100.0)) AS INT) AS adjusted
FROM employment_history
GROUP BY region
HAVING COUNT(*) >= 50
ORDER BY nominal DESC;


-- ---------------------------------------------------------------------
-- 7. Turning the activity log into per-person features
--    One row per person, ready to join to anything else.
-- ---------------------------------------------------------------------

SELECT campus_id,
       SUM(CASE WHEN experience_type IN ('Internship', 'Co-op')
                THEN 1 ELSE 0 END) AS internships,
       SUM(CASE WHEN experience_type = 'Certification'
                THEN 1 ELSE 0 END) AS certifications,
       SUM(CASE WHEN experience_type = 'Student Organization'
                THEN 1 ELSE 0 END) AS organizations,
       COUNT(*) AS total_activities
FROM student_experience
GROUP BY campus_id
LIMIT 20;


-- ---------------------------------------------------------------------
-- 8. Everyone as one population
--    A campus_id is in students_current OR alumni, never both. Union the
--    columns they share to treat the whole campus as one table.
-- ---------------------------------------------------------------------

SELECT campus_id, major, track, entry_type, residency,
       'Current student' AS status
FROM students_current
UNION ALL
SELECT campus_id, major, track, entry_type, residency,
       'Graduate' AS status
FROM alumni;
