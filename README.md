# NYTINOL SRA

This repository implements a career-planning recommender that compares a student's current profile to alumni outcomes and proposes the next best academic and experiential steps toward a target job and industry. The project pulls together a synthetic student/alumni dataset, semantic embeddings for job titles and industries, a scoring model based on historical outcomes, and a small FastAPI service that exposes recommendations.

The project is built around the HackUMBC 2026 data bundle in `hackumbc-2026-main/`, with the recommender logic running on top of cleaned tables and vector embeddings stored in a Timescale/Postgres-backed database.

## What the project is trying to do

The core problem is: given a student profile, a target title, an industry, and an expected salary, what should they do next to move closer to that outcome?

The system answers that by:

- loading the historical alumni population from a synthetic career dataset
- comparing the student’s target title and industry against alumni job vectors and industry vectors
- ranking alumni who are closest to the student’s goal in title and industry similarity
- scoring the selected alumni using salary gap, title match, and industry match
- pulling the closest matching campus records for those alumni
- evaluating the student’s completed classes and prior experience categories
- recommending a short list of high-impact next steps: a course to take, an experience to pursue, and/or a GPA improvement path

The output is not a hard-coded recommendation engine for one degree; it is a general template for student advising based on outcome patterns in historical data.

## Repository structure

```text
.
├── README.md
├── api.py                     # recommendation service logic
├── main.py                    # FastAPI app entrypoint
├── search_algorithm.py        # ranking and recommendation logic
├── tiger_db.py               # database access and vector lookup
├── upload_to_tiger.py        # uploads the CSVs into a Timescale database
├── Feature-engineering/      # dataset cleanup and embedding generation scripts
│   ├── Alumni_cleanup.py
│   ├── employment_history_cleanup.py
│   ├── industry_vector_embedding_logic.py
│   ├── job_vector_embedding_logic.py
│   ├── student_experience_cleanup.py
│   ├── transcipt_cleanup.py
│   └── ...
├── hackumbc-2026-main/
│   ├── README.md
│   ├── data/
│   └── examples/
├── tests/
│   └── test_search_algorithm.py
└── .env / tiger-cloud-nytinoltiger-credentials.env  # environment config (not always committed)
```

## Data and domain model

The system depends on the synthetic dataset stored under `hackumbc-2026-main/data/`.

The important files are:

- `alumni.csv`: graduate records, outcome data, degree metadata, and first-job information
- `students_current.csv`: currently enrolled students
- `transcripts.csv`: course attempts, grades, credits, and GPA-related fields
- `student_experience.csv`: internships, clubs, research, certifications, hackathons, and campus jobs
- `employment_history.csv`: job history over time
- `course_catalog.csv`: course metadata including prerequisites and skill tags

The dataset is intentionally designed so that a `campus_id` identifies a person, and the same person can be tracked across multiple tables:

- students and alumni are separate populations
- transcripts and course catalog connect via `course_id`
- student experience and alumni records connect through `campus_id`
- employment history tells the story of the trajectory after graduation

The repository also includes a top-level dataset README at `hackumbc-2026-main/README.md`, which explains the schema and assumptions for the generated data.

## How the data gets cleaned and prepared

The `Feature-engineering/` directory contains the data preparation scripts that normalize the raw CSVs before the recommender is used.

### Alumni cleanup

`Feature-engineering/Alumni_cleanup.py` filters the alumni population to remove records with missing or non-actionable first-job outcomes and trims out master’s-degree records from the relevant sample. This keeps the recommender focused on the population whose early career paths are most useful for comparison.

### Student experience cleanup

`Feature-engineering/student_experience_cleanup.py` strips columns and rows that are not useful for the adviser logic, such as `is_paid` and `hours_per_week` in some cases, and removes entries with `Not Applicable` role labels.

### Transcript cleanup

`Feature-engineering/transcipt_cleanup.py` normalizes transcript data and ensures the course records are usable for grade-based and prerequisite-based recommendation logic.

### Employment history cleanup

`Feature-engineering/employment_history_cleanup.py` prepares employment records for later analysis and metric extraction, ensuring the historical job flow is consistent enough to compare against paths in the alumni data.

### Embedding generation

The project generates semantic embeddings for job titles and industries using Gemini embeddings. These are built in:

- `Feature-engineering/job_vector_embedding_logic.py`
- `Feature-engineering/industry_vector_embedding_logic.py`

The job-title logic builds a lookup table of unique first-job titles from the alumni dataset, embeds each title with Gemini, normalizes the vectors, and writes them to a CSV (`hackumbc-2026-main/data/job_title_embeddings.csv`).

The industry embedding logic does the same for industry names such as `Software Products`, `Insurance`, or `Cybersecurity Services`. The data is converted into a consistent embedding lookup that can be used later for similarity comparisons.

## Database layer and vector store

`upload_to_tiger.py` and `tiger_db.py` are the bridge between the curated CSVs and the application logic.

### upload_to_tiger.py

This script reads each CSV in `hackumbc-2026-main/data/` and loads it into the database using `asyncpg` and `pandas`.

It does three main things:

1. infers a database column type from each CSV column
2. creates a table in the remote database with matching schema
3. copies each row from the CSV into a database table

The script is designed to load the main dataset tables such as:

- alumni
- transcripts
- student_experience
- course_catalog
- employment_history
- students_current

This is the operational layer that makes the cleaned source records available to the recommender at runtime.

### tiger_db.py

This is the main repository access layer for the application.

It uses a Timescale/Postgres service configured via environment variables and provides functions such as:

- `get_job_vector(job_title)`
- `get_industry_vector(industry)`
- `get_all_job_vectors()`
- `load_scoring_data()`
- `load_campus_records(campus_ids)`
- `vectors_for_user(uid, job_title, industry, embed)`

The database layer stores:

- `job_vectors`: user-goal job title embeddings
- `industry_vectors`: user-goal industry embeddings
- `users`: a mapping of `uid` to the job/industry embedding positions used for a student
- the raw dataset tables loaded from the CSVs

The runtime also loads the historical alumni dataset and job/industry lookup embeddings from the database, with a caching layer so the expensive data fetch is not repeated unnecessarily.

This service is central to the entire system: it turns raw data tables and semantic vectors into a queryable scoring model.

## Recommendation algorithm

The actual recommendation logic lives in `search_algorithm.py`.

At a high level, it does the following:

1. loads the alumni set and the job/industry vector catalog from the database
2. embeds the target job title and industry for the current user
3. computes cosine similarity between the target vectors and the alumni first-job vectors
4. ranks alumni by a composite score made from:
   - job title similarity
   - industry similarity
   - salary proximity to the requested target salary
5. selects the closest alumni records
6. loads the campus records for those selected alumni
7. computes weighted candidate recommendations based on:
   - classes not yet completed
   - course popularity among similar alumni
   - experience gaps and opportunity buckets
   - GPA improvement patterns
   - the student’s current experience mix

### Scoring model

The system uses a weighted score formula:

- `TITLE_WEIGHT = 0.50`
- `INDUSTRY_WEIGHT = 0.30`
- `SALARY_WEIGHT = 0.20`

This means the model weights title match most heavily, then industry, then salary closeness. The salary score is normalized using the relative distance between the alumni’s first-job salary and the user’s target salary.

### Experience buckets

The recommender groups experiences into buckets such as:

- internship
- research
- club
- hackathon
- certification
- campus_job

This is important because it determines which kinds of experience are currently underrepresented for the student. The algorithm estimates the student’s existing investment by experience type and then boosts underrepresented categories so recommendations feel personalized rather than generic.

### Course logic

The next-step recommendation system has a course module that:

- gathers transcripts for the selected alumni
- looks at courses they took and the grades they earned
- finds courses that are near the top of the similarity graph
- excludes courses the user already completed
- prioritizes courses most associated with similar alumni outcomes

This is implemented in `_next_steps()` in `search_algorithm.py`.

### Experience logic

For each relevant experience type, the algorithm:

- removes any experience the user already has
- scores similar experiences from the selected alumni
- weights them by how much the student is currently missing that category
- returns the most promising experiences in ranked order

This helps the recommendations include actions like internships, research opportunities, leadership roles, or technical certifications rather than only course suggestions.

## API surface

The application exposes a REST endpoint via FastAPI.

### `main.py`

This file defines an HTTP server and a single `/search` route. The request body is a `SearchRequest` model with these fields:

- `uid`
- `name`
- `age`
- `major`
- `gpa`
- `classes`
- `salary`
- `job_title`
- `industry`
- `experiences`

Each experience entry contains:

- `experience_type`
- `experience_name`
- `industry`
- `term`
- `terms_participated`
- `hours_per_week` (optional)

The route sends the student profile to `CareerSearch.recommend()` and returns a list of recommendation strings.

### `api.py`

This file contains the application logic that orchestrates the recommendation flow:

- calls `tiger_db.vectors_for_user()` for the target job title and industry
- creates the `CurrentAchievements` profile from the student's experience list
- passes the student’s profile plus vector inputs into `search_algorithm.search_algorithm()`
- returns the final recommendation steps

It also loads and uses Gemini embeddings through `client.models.embed_content()` for generic job and industry text that is not already present in the catalog.

## Runtime flow

A normal request follows this path:

1. `main.py` receives a POST to `/search`
2. The payload is validated by Pydantic
3. `CareerSearch.recommend()` calls `tiger_db.vectors_for_user()`
4. The job title and industry are converted into vectors using the database or Gemini embedding fallback
5. The current profile is converted into a `CurrentAchievements` object and passed to the search algorithm
6. The algorithm loads alumni scoring data and the relevant campus records
7. The closest alumni are selected and the next-step recommendations are generated
8. A list of human-readable steps is returned to the API caller

In effect, the service is a planner that compares a student to a historical population of similar outcomes and suggests what to pursue next.

## Example recommendation behavior

The repository includes a test suite in `tests/test_search_algorithm.py` with example students representing different experience levels.

The tests validate that:

- the system returns exactly three recommendations for each student
- recommendations differ across profiles
- the system avoids repeating classes already completed
- the generated suggestions are not generic project-generation output

The example students include cases like:

- a student with almost no prior experience
- a student with moderate internship and club involvement
- a student with a fuller background including research, hackathons, internships, and certifications

This test set is useful because it demonstrates the model’s behavior across different stages of a student’s journey.

## Environment and setup

The project expects environment variables to be present, especially:

- `GEMINI_API_KEY` for embedding generation
- `TIMESCALE_SERVICE_URL` for the Timescale/Postgres connection string

The repo contains helper patterns like:

- `.env` for local secrets
- `tiger-cloud-nytinoltiger-credentials.env` for database credentials

The scripts intentionally load these files with a small helper function and call `os.environ.setdefault(...)` so configuration can be supplied without hard-coding secrets in the code.

## Typical local workflow

A developer or analyst typically works through the system in this order:

1. Prepare or verify raw CSV data in `hackumbc-2026-main/data/`
2. Run the cleanup scripts in `Feature-engineering/`
3. Generate embeddings for job titles and industries
4. Upload the CSV tables to the remote database via `upload_to_tiger.py`
5. Start the FastAPI app with `main.py`
6. Submit a POST request to `/search` with a student profile and target job/industry
7. Review the returned recommendations and tune the algorithm weights or filters as needed

This is a data-driven advising pipeline, not a single offline script: it is built to support iterative experimentation over real outcome data.

## Integration points

Important integration components include:

- `FastAPI` server for API access
- `Pydantic` request models for typed schema validation
- `asyncpg` for direct database access and bulk transfers
- `pandas` for CSV cleaning and inspection
- `numpy` for vector and cosine math
- `google.genai` for embedding generation via Gemini

This combination keeps the project flexible enough for both analytical work and live recommendation serving.

## Design notes

A few important architectural choices shape the project:

- The system separates raw dataset ingestion from runtime logic.
- The scoring model is built around historical alumni similarity rather than direct rule-based mapping.
- Vector embeddings are used to measure semantic similarity between goals and historical outcomes.
- Recommendations are generated from a mix of course history, experience patterns, and GPA trends instead of only one signal.
- The code intentionally keeps the database and the recommendation logic distinct so it can evolve without reworking the whole data pipeline.

## Limitations and operational considerations

The project is intentionally built against synthetic data. That means:

- it is best suited for experimenting with career-planning logic and advising heuristics
- it is useful for building a prototype or academic demo, not a production hiring recommendation system
- the model quality depends heavily on the quality and consistency of the generated dataset and the embeddings
- environment configuration and remote database connectivity are required for the live workflow

The dataset also contains placeholder values such as `Not Applicable`, which the code handles on a case-by-case basis with filtering and normalization.

## Summary

This repository combines academic data, semantic search, and a recommender model to answer a practical advising question: what should a student do next to become more likely to reach a chosen career outcome?

The project spans data cleaning, embedding generation, database loading, scoring, and API serving, making it a complete workflow rather than a single script. It is designed as a prototype career-advisor system that learns from historical alumni outcomes and turns those patterns into actionable next steps for present students.
