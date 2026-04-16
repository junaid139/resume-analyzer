from flask import Flask, render_template, request
import os
from utils.extractor import extract_text
from utils.analyzer import (calculate_score, get_skill_gap,
                             extract_name_email, get_improvements,
                             check_education, check_experience)
from utils.db import get_connection

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    job_title = request.form['job_title']
    job_description = request.form['job_description']
    resume_files = request.files.getlist('resumes')

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        "INSERT INTO job_descriptions (job_title, job_description) VALUES (%s, %s)",
        (job_title, job_description)
    )
    job_id = cursor.lastrowid

    candidates = []

    for resume_file in resume_files:
        filename = resume_file.filename
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        resume_file.save(file_path)

        resume_text = extract_text(file_path)
        score = calculate_score(resume_text, job_description)
        matched_skills, missing_skills = get_skill_gap(resume_text, job_description)

        extracted_name, extracted_email = extract_name_email(resume_text)
        candidate_name = extracted_name if extracted_name else os.path.splitext(filename)[0].replace("_", " ").title()

        has_education = check_education(resume_text)
        exp_score = check_experience(resume_text)
        improvements = get_improvements(score, matched_skills, missing_skills, has_education, exp_score)

        cursor.execute(
            "INSERT INTO resumes (candidate_name, filename) VALUES (%s, %s)",
            (candidate_name, filename)
        )
        resume_id = cursor.lastrowid

        cursor.execute(
            "INSERT INTO results (resume_id, job_id, score, matched_skills, missing_skills) VALUES (%s, %s, %s, %s, %s)",
            (resume_id, job_id, score, ", ".join(matched_skills), ", ".join(missing_skills))
        )

        candidates.append({
            "name": candidate_name,
            "email": extracted_email,
            "filename": filename,
            "score": score,
            "matched": matched_skills,
            "missing": missing_skills,
            "improvements": improvements
        })

    conn.commit()
    cursor.close()
    conn.close()

    candidates = sorted(candidates, key=lambda x: x['score'], reverse=True)

    return render_template('result.html',
        job_title=job_title,
        candidates=candidates
    )

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)