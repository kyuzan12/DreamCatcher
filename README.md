# DreamCatcher

Sleep-disorder screening for university students (ITTE105b Final Term Requirement).
DreamCatcher estimates whether a student's answers about sleep, lifestyle and health look most like **Insomnia**,
**Sleep Apnea** or **No Sleep Disorder**, and gives screening advice. It is an educational screening aid, not a diagnosis.

- **Data:** BD-Uni Sleep Disorder Dataset, 619 students, 12 predictors, 3 classes
- **Models compared:** Random Forest, KNN and SVM; **KNN** was selected (k = 25)
- **Deployment:** FastAPI service packaged with Docker, with Swagger UI and a web page

## Project folder

```
DreamCatcher_Final_Project.ipynb   notebook: data preparation, EDA, models, evaluation, saving the model
DreamCatcher_Documentation.docx    written report
dreamcatcher_utils.py              cleaning, feature engineering and decision rules (used by the notebook)
dreamcatcher_app/                  the deployed application
  app/dataset/                     BD-Uni. Sleep Disorder Dataset.xlsx
  app/models/                      dreamcatcher_model.joblib, label_encoder.pkl, model_card.json
  app/static/index.html            web page
  app/dreamcatcher_utils.py        copy of the shared module (the notebook refreshes it)
  app/server.py                    FastAPI service
  Dockerfile, requirements.txt
```

## What you need

- **Docker Desktop**, to run the application
- **Anaconda**, only if you want to run the notebook or the app without Docker

## Run the application (Docker)

Start Docker Desktop, open a terminal inside `dreamcatcher_app`, then run:

```bash
docker build -t dreamcatcher_img .
docker run --name dreamcatcher -p 8000:8000 dreamcatcher_img
```

Wait for `Uvicorn running on http://0.0.0.0:8000`, then open:

| Address | What it is |
| --- | --- |
| http://localhost:8000 | connection test, shows `{"message":"DreamCatcher API"}` |
| http://localhost:8000/web/ | the web application |
| http://localhost:8000/docs | Swagger UI, for testing the API |

Stop it with `Ctrl+C`. To run it again later, remove the old container first: `docker rm -f dreamcatcher`.

**Without Docker** (Anaconda Prompt, inside `dreamcatcher_app`):

```bash
pip install -r requirements.txt
python -m uvicorn app.server:app --port 8000
```

## Test it yourself

### Web page (http://localhost:8000/web/)

1. Fill in the form (a name is optional; every other field is required).
2. Click **+ Add person** to screen up to 20 people at once.
3. Click **Run screening**, then **Details** to see the probability of each class.

### Swagger UI (http://localhost:8000/docs)

Open an endpoint, click **Try it out**, then **Execute**. A valid example is already filled in for every POST endpoint.

| Endpoint | Input |
| --- | --- |
| `POST /predict` | one student, answers written out |
| `POST /predict-batch` | up to 20 students, answers written out (used by the web page) |
| `POST /predict-features` | up to 20 students, each as a row of 11 numbers |

### Sample students

| # | Student | Expected result |
| --- | --- | --- |
| 1 | Sleeps less than 4 hours, very poor quality, high stress | Sleep Apnea 56%, Higher likelihood |
| 2 | Sleeps 6-7 hours, poor quality, moderate stress | Insomnia 64%, Higher likelihood |
| 3 | Sleeps 8 hours or more, good quality, low stress | No Sleep Disorder 79%, Lower likelihood (Insomnia ruled out) |
| 4 | Sleeps 6-7 hours, average quality, very high stress | Sleep Apnea 36%, tied with No Sleep Disorder 36%, Possible |

Paste one of these into `POST /predict`:

```json
{"gender": "Male", "age": 21, "sleep_duration": "Less than 4 hours", "sleep_quality": "Very Poor", "stress_level": "High", "physical_activity": 20, "bmi_category": "Overweight", "blood_pressure": "130/85", "heart_rate": 88, "daily_steps": "Less than 4,999"}
```

```json
{"gender": "Female", "age": 22, "sleep_duration": "6-7 hours", "sleep_quality": "Poor", "stress_level": "Moderate", "physical_activity": 30, "bmi_category": "Normal", "blood_pressure": "118/76", "heart_rate": 82, "daily_steps": "5,000-7,999"}
```

```json
{"gender": "Female", "age": 22, "sleep_duration": "8 hours or more", "sleep_quality": "Good", "stress_level": "Low", "physical_activity": 60, "bmi_category": "Normal", "blood_pressure": "120/80", "heart_rate": 72, "daily_steps": "8,000-11,999"}
```

```json
{"gender": "Male", "age": 22, "sleep_duration": "6-7 hours", "sleep_quality": "Average", "stress_level": "Very High", "physical_activity": 104, "bmi_category": "Normal", "blood_pressure": "125/77", "heart_rate": 80, "daily_steps": "5,000-7,999"}
```

All four students in `POST /predict-features`:

```json
{"features": [[1, 21, 0, 0, 3, 20, 2, 130, 85, 88, 0], [0, 22, 2, 1, 2, 30, 1, 118, 76, 82, 1], [0, 22, 3, 3, 1, 60, 1, 120, 80, 72, 2], [1, 22, 2, 2, 4, 104, 1, 125, 77, 80, 1]]}
```

The 11 numbers, in order:

| # | Value | Codes |
| --- | --- | --- |
| 1 | gender | 0 = Female, 1 = Male |
| 2 | age | years |
| 3 | sleep duration | 0 = Less than 4 hours, 1 = 4-5, 2 = 6-7, 3 = 8 hours or more |
| 4 | sleep quality | 0 = Very Poor, 1 = Poor, 2 = Average, 3 = Good, 4 = Excellent |
| 5 | stress level | 0 = Very Low, 1 = Low, 2 = Moderate, 3 = High, 4 = Very High |
| 6 | physical activity | minutes per day |
| 7 | BMI category | 0 = Underweight, 1 = Normal, 2 = Overweight, 3 = Obese |
| 8 | systolic blood pressure | mmHg |
| 9 | diastolic blood pressure | mmHg (lower than systolic) |
| 10 | heart rate | beats per minute |
| 11 | daily steps | 0 = Less than 4,999, 1 = 5,000-7,999, 2 = 8,000-11,999, 3 = 12,000-15,999, 4 = More than 16,000 |

To see input validation, send `"sleep_quality": "Terrible"` or a blood pressure of `"80/120"`: the API answers with code 422 and names the wrong field.

**From PowerShell instead of Swagger** (while the container is running):

```powershell
Invoke-RestMethod -Method Post -Uri http://localhost:8000/predict-features -ContentType "application/json" -Body '{"features": [[1, 21, 0, 0, 3, 20, 2, 130, 85, 88, 0]]}'
```

## Reading a result

- **Most likely pattern** and **confidence**: the class with the highest probability. Below 50% means the model is unsure.
- **Screening advice**: based on the probability of *any* sleep disorder: 0.69 or more = Higher likelihood,
  0.44 to 0.69 = Possible, below 0.44 = Lower likelihood.
- **Two decision rules**: a tie between a disorder and No Sleep Disorder is reported as the disorder; Insomnia is
  ruled out for students who sleep 8 hours or more with good or excellent quality.

## Re-train the model (notebook)

Open `DreamCatcher_Final_Project.ipynb` in Jupyter (Anaconda) from this folder and run **Kernel > Restart & Run All**
(about 5 minutes). The first cell installs missing packages. Section 8 saves the new model straight into
`dreamcatcher_app/app/models/`. After re-training, rebuild the Docker image so the container uses the new model.

## Common problems

| Message | Fix |
| --- | --- |
| `Conflict. The container name "/dreamcatcher" is already in use` | `docker rm -f dreamcatcher`, then `docker run` again |
| `port is already allocated` | another app or container uses port 8000: stop it, or run with `-p 8001:8000` and open http://localhost:8001 |
| Changes to the code or model do not show | rebuild: `docker build -t dreamcatcher_img .`, remove the old container, run again |
| `error during connect` / Docker not found | start Docker Desktop and wait until it says it is running |
| Web page says it cannot reach the service | open it from the running server (http://localhost:8000/web/), not by double-clicking `index.html` |
