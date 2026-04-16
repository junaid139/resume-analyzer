import nltk
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

from nltk.corpus import stopwords

SKILLS_DB = [
    # Programming Languages
    "python", "java", "javascript", "typescript", "c", "c++", "c#", "r", "go",
    "rust", "kotlin", "swift", "scala", "perl", "ruby", "php", "dart", "lua",
    "matlab", "bash", "shell", "powershell", "groovy", "haskell", "elixir",
    "clojure", "fortran", "cobol", "assembly", "vba", "objective-c",

    # Web Frontend
    "html", "css", "html5", "css3", "bootstrap", "tailwind", "sass", "less",
    "react", "angular", "vue", "nextjs", "nuxtjs", "svelte", "jquery",
    "redux", "webpack", "babel", "vite", "gatsby", "ember", "backbone",

    # Web Backend
    "flask", "django", "fastapi", "spring boot", "spring", "express",
    "nodejs", "nestjs", "laravel", "symfony", "rails", "ruby on rails",
    "asp.net", ".net", "struts", "hibernate", "jsp", "servlet",

    # Databases
    "mysql", "postgresql", "sqlite", "mongodb", "redis", "cassandra",
    "oracle", "sql server", "mariadb", "dynamodb", "firebase",
    "elasticsearch", "couchdb", "neo4j", "influxdb", "supabase",

    # SQL Skills
    "sql", "nosql", "stored procedures", "triggers", "indexing",
    "query optimization", "database design", "normalization", "joins",
    "transactions", "jdbc", "orm", "sequelize", "prisma", "sqlalchemy",

    # Cloud & DevOps
    "aws", "azure", "google cloud", "gcp", "docker", "kubernetes",
    "terraform", "ansible", "jenkins", "gitlab ci", "github actions",
    "circleci", "travis ci", "helm", "nginx", "apache", "linux",
    "ubuntu", "centos", "shell scripting", "ci/cd", "devops",
    "cloudformation", "lambda", "ec2", "s3", "rds", "heroku",
    "vercel", "netlify", "digitalocean", "openshift",

    # Data Science & ML
    "machine learning", "deep learning", "data science", "data analysis",
    "artificial intelligence", "ai", "nlp", "natural language processing",
    "computer vision", "opencv", "tensorflow", "keras", "pytorch",
    "scikit-learn", "pandas", "numpy", "matplotlib", "seaborn",
    "plotly", "scipy", "statsmodels", "xgboost", "lightgbm",
    "random forest", "neural network", "cnn", "rnn", "lstm",
    "transformer", "bert", "gpt", "reinforcement learning",
    "data visualization", "feature engineering", "model deployment",
    "mlops", "data mining", "big data", "hadoop", "spark",
    "hive", "kafka", "airflow", "tableau", "power bi", "excel",

    # Mobile Development
    "android", "ios", "react native", "flutter", "swift", "kotlin",
    "xamarin", "ionic", "cordova", "mobile development",

    # API & Integration
    "rest api", "restful", "graphql", "soap", "api development",
    "microservices", "webhooks", "oauth", "jwt", "xml", "json",
    "grpc", "websocket", "postman", "swagger", "openapi",

    # Testing
    "unit testing", "integration testing", "selenium", "pytest",
    "junit", "testng", "mocha", "jest", "cypress", "playwright",
    "test driven development", "tdd", "bdd", "cucumber", "appium",

    # Tools & Version Control
    "git", "github", "gitlab", "bitbucket", "jira", "confluence",
    "trello", "slack", "maven", "gradle", "npm", "yarn", "pip",
    "virtualenv", "conda", "jupyter", "vs code", "intellij",
    "eclipse", "android studio", "xcode", "figma", "adobe xd",

    # Security
    "cybersecurity", "ethical hacking", "penetration testing",
    "network security", "cryptography", "ssl", "tls", "firewall",
    "vulnerability assessment", "owasp", "siem", "soc",
    "identity management", "zero trust", "cloud security",

    # Networking
    "networking", "tcp/ip", "dns", "http", "https", "ftp",
    "ssh", "vpn", "load balancing", "cdn", "proxy",

    # Project Management & Soft Skills
    "agile", "scrum", "kanban", "waterfall", "sdlc", "project management",
    "product management", "leadership", "communication", "teamwork",
    "problem solving", "critical thinking", "time management",
    "adaptability", "creativity", "collaboration", "presentation",
    "documentation", "technical writing", "code review", "mentoring",

    # Business & Analytics
    "business analysis", "requirements gathering", "stakeholder management",
    "data analytics", "reporting", "kpi", "crm", "erp", "sap",
    "salesforce", "hubspot", "google analytics", "seo", "digital marketing",

    # Embedded & Hardware
    "embedded systems", "arduino", "raspberry pi", "iot",
    "internet of things", "rtos", "firmware", "plc", "fpga",
    "microcontroller", "circuit design", "pcb design",

    # Blockchain
    "blockchain", "solidity", "ethereum", "web3", "smart contracts",
    "cryptocurrency", "nft", "defi", "hyperledger",

    # Game Development
    "unity", "unreal engine", "game development", "opengl", "directx",
    "blender", "3d modeling",

    # Education Keywords
    "bca", "bsc", "mca", "msc", "b.tech", "m.tech", "bachelor",
    "master", "phd", "computer science", "information technology",
    "software engineering", "data engineering"
]

EDUCATION_KEYWORDS = [
    "bachelor", "master", "phd", "bca", "bsc", "mca", "msc", "b.tech",
    "m.tech", "degree", "engineering", "computer science", "information technology"
]

EXPERIENCE_KEYWORDS = [
    "experience", "worked", "developed", "built", "designed", "managed",
    "led", "implemented", "created", "maintained", "years"
]

def extract_name_email(text):
    email = ""
    name = ""

    # Extract email using regex
    email_match = re.findall(r'[\w\.-]+@[\w\.-]+\.\w+', text)
    if email_match:
        email = email_match[0]

    # Extract name using spaCy NER
    doc = nlp(text[:500])  # Only check first 500 chars for speed
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            name = ent.text
            break

    return name, email

def clean_text(text):
    stop_words = set(stopwords.words('english'))
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    tokens = text.split()
    cleaned = [word for word in tokens if word not in stop_words]
    return " ".join(cleaned)

def extract_skills(text):
    text_lower = text.lower()
    return [skill for skill in SKILLS_DB if skill in text_lower]

def check_education(text):
    text_lower = text.lower()
    for keyword in EDUCATION_KEYWORDS:
        if keyword in text_lower:
            return True
    return False

def check_experience(text):
    text_lower = text.lower()
    count = sum(1 for keyword in EXPERIENCE_KEYWORDS if keyword in text_lower)
    return min(count / len(EXPERIENCE_KEYWORDS), 1.0)

def calculate_score(resume_text, job_text):
    resume_skills = set(extract_skills(resume_text))
    job_skills = set(extract_skills(job_text))

    if len(job_skills) > 0:
        skill_score = len(resume_skills & job_skills) / len(job_skills)
    else:
        vectorizer = TfidfVectorizer()
        vectors = vectorizer.fit_transform([clean_text(resume_text), clean_text(job_text)])
        skill_score = cosine_similarity(vectors[0], vectors[1])[0][0]

    edu_score = 1.0 if check_education(resume_text) else 0.0
    exp_score = check_experience(resume_text)

    final_score = (skill_score * 0.60) + (edu_score * 0.20) + (exp_score * 0.20)
    return round(final_score * 100, 2)

def get_skill_gap(resume_text, job_text):
    resume_skills = extract_skills(resume_text)
    job_skills = extract_skills(job_text)
    matched = [skill for skill in job_skills if skill in resume_skills]
    missing = [skill for skill in job_skills if skill not in resume_skills]
    return matched, missing

def get_improvements(score, matched, missing, has_education, exp_score):
    tips = []

    if missing:
        tips.append(f"Learn these missing skills: {', '.join(missing[:5])}")
    if not has_education:
        tips.append("Add your education details (degree, college, year) to the resume")
    if exp_score < 0.3:
        tips.append("Add more experience details — projects, internships, or work history")
    if score < 40:
        tips.append("Resume needs major improvement — skills don't match this job well")
    elif score < 70:
        tips.append("Good base! Focus on adding the missing skills to improve your score")
    else:
        tips.append("Strong match! Keep your resume updated with latest skills")

    return tips