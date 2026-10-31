#!/usr/bin/env python3
"""Generate a synthetic resume and job-description corpus (PDF only) for local RAG experiments.

The 35 original candidates keep their role, years, skills and headline achievement; every
resume is expanded with contact details, a dated job history and adjacent skills. New
candidates are spread across strong / partial / weak fits for each job description, plus
distractor roles, so ranking has something to separate. Output is deterministic (seeded).
"""

from __future__ import annotations

import json
import random
import textwrap
from pathlib import Path


ROOT = Path(__file__).parent
RESUMES = ROOT / "resumes"
JOBS = ROOT / "job_descriptions"

PEOPLE = [
    ("aisha_khan", "Aisha Khan", "Data Scientist", "6", "Python, SQL, Machine Learning, TensorFlow, AWS", "MSc Data Science", "Built demand forecasting models that improved inventory planning by 18%."),
    ("benjamin_lee", "Benjamin Lee", "Backend Engineer", "8", "Python, Django, PostgreSQL, Redis, Docker", "BSc Computer Science", "Designed payment APIs serving 2 million monthly transactions."),
    ("carla_mendes", "Carla Mendes", "Product Manager", "7", "Product Strategy, Agile, SQL, A/B Testing, Figma", "MBA", "Launched a self-service onboarding flow that raised activation by 22%."),
    ("deepak_sharma", "Deepak Sharma", "Machine Learning Engineer", "5", "Python, PyTorch, MLOps, Kubernetes, MLflow", "MTech Artificial Intelligence", "Deployed computer-vision models with monitored production pipelines."),
    ("elena_petrova", "Elena Petrova", "Frontend Engineer", "5", "TypeScript, React, Next.js, Jest, Accessibility", "BSc Software Engineering", "Created accessible analytics dashboards used by enterprise customers."),
    ("farah_ali", "Farah Ali", "Data Engineer", "6", "Python, Spark, Airflow, Snowflake, dbt", "BSc Information Systems", "Built batch and streaming pipelines for customer event data."),
    ("gabriel_santos", "Gabriel Santos", "DevOps Engineer", "9", "AWS, Terraform, Kubernetes, CI/CD, Linux", "BSc Computer Engineering", "Standardized infrastructure-as-code across 40 cloud services."),
    ("hana_kim", "Hana Kim", "UX Researcher", "5", "User Research, Usability Testing, Figma, Surveys, Analytics", "MA Human-Computer Interaction", "Led research that simplified checkout for a retail platform."),
    ("isaac_brown", "Isaac Brown", "Cybersecurity Analyst", "6", "SIEM, Incident Response, Python, Splunk, Risk Assessment", "BSc Cybersecurity", "Reduced incident triage time through automated alert enrichment."),
    ("jia_wei", "Jia Wei", "Business Analyst", "4", "SQL, Tableau, Excel, Requirements Analysis, Agile", "BBA Analytics", "Translated operational reporting needs into executive dashboards."),
    ("kofi_mensah", "Kofi Mensah", "Cloud Architect", "10", "Azure, AWS, Terraform, Kubernetes, Architecture", "MSc Information Technology", "Migrated a regulated workload to a resilient multi-cloud design."),
    ("lena_fischer", "Lena Fischer", "NLP Engineer", "4", "Python, NLP, Transformers, Hugging Face, FastAPI", "MSc Computational Linguistics", "Built a document-classification service for support operations."),
    ("marcus_johnson", "Marcus Johnson", "Mobile Engineer", "6", "Kotlin, Android, Jetpack Compose, Firebase, REST APIs", "BSc Computer Science", "Delivered offline-capable field-service applications."),
    ("nora_ahmed", "Nora Ahmed", "QA Automation Engineer", "5", "Python, Selenium, Playwright, API Testing, CI/CD", "BSc Software Engineering", "Expanded regression coverage and shortened release verification."),
    ("oliver_smith", "Oliver Smith", "Financial Analyst", "7", "Financial Modeling, Excel, SQL, Forecasting, Power BI", "BCom Finance", "Created rolling forecasts used in quarterly business reviews."),
    ("priya_patel", "Priya Patel", "Full Stack Engineer", "7", "JavaScript, React, Node.js, MongoDB, AWS", "BSc Computer Science", "Built customer portals and maintained secure API integrations."),
    ("quentin_dubois", "Quentin Dubois", "Data Scientist", "3", "Python, R, Statistics, SQL, Tableau", "MSc Statistics", "Analyzed retention experiments and presented statistical recommendations."),
    ("rosa_garcia", "Rosa Garcia", "HR Operations Specialist", "6", "HRIS, Recruiting, Onboarding, Excel, Policy Compliance", "BA Human Resources", "Coordinated onboarding and reporting for a distributed workforce."),
    ("samuel_okoro", "Samuel Okoro", "Site Reliability Engineer", "7", "Go, Kubernetes, Prometheus, Grafana, AWS", "BSc Computer Engineering", "Improved service reliability with actionable SLOs and runbooks."),
    ("tanya_wilson", "Tanya Wilson", "Marketing Analyst", "5", "Google Analytics, SQL, Tableau, Campaign Analysis, SEO", "BA Marketing", "Measured multi-channel campaigns and identified growth opportunities."),
    ("umar_hassan", "Umar Hassan", "Java Engineer", "8", "Java, Spring Boot, Kafka, PostgreSQL, Microservices", "BSc Software Engineering", "Modernized order-processing services with event-driven architecture."),
    ("victoria_nguyen", "Victoria Nguyen", "Data Engineer", "4", "Python, SQL, BigQuery, Airflow, Looker", "BSc Data Analytics", "Automated marketing data ingestion and quality checks."),
    ("william_chen", "William Chen", "AI Product Manager", "6", "Product Management, LLMs, SQL, Roadmaps, Experimentation", "MBA", "Defined AI-assisted workflows for a knowledge-management product."),
    ("xenia_ivanova", "Xenia Ivanova", "Platform Engineer", "6", "Python, Kubernetes, Terraform, GitHub Actions, AWS", "MSc Computer Science", "Created self-service deployment templates for application teams."),
    ("yusuf_abdi", "Yusuf Abdi", "Network Engineer", "8", "Cisco, TCP/IP, Firewalls, VPN, Network Monitoring", "BSc Network Engineering", "Maintained secure branch connectivity and network observability."),
    ("zoe_martin", "Zoe Martin", "Content Designer", "4", "UX Writing, Content Strategy, Figma, Accessibility, Research", "BA Communications", "Improved in-product guidance for complex account setup."),
    ("aarav_mehta", "Aarav Mehta", "Machine Learning Engineer", "7", "Python, TensorFlow, Spark, AWS, SageMaker", "MSc Machine Learning", "Productionized fraud-detection models with scalable feature pipelines."),
    ("bianca_rossi", "Bianca Rossi", "Frontend Engineer", "3", "TypeScript, React, CSS, Storybook, Cypress", "BSc Digital Media", "Developed reusable interface components and visual test coverage."),
    ("chinedu_eze", "Chinedu Eze", "Backend Engineer", "5", "Node.js, TypeScript, PostgreSQL, Docker, REST APIs", "BSc Computer Science", "Implemented secure partner APIs and asynchronous processing workers."),
    ("diana_ortiz", "Diana Ortiz", "Operations Manager", "9", "Operations, Process Improvement, Excel, Vendor Management, KPIs", "BA Business Administration", "Reduced fulfillment delays through cross-functional process redesign."),
    ("ethan_miller", "Ethan Miller", "Data Analyst", "3", "SQL, Python, Power BI, Excel, Statistics", "BSc Economics", "Built recurring performance reports for sales leadership."),
    ("fatima_zahra", "Fatima Zahra", "Security Engineer", "5", "Python, Cloud Security, IAM, AWS, Threat Modeling", "MSc Cybersecurity", "Automated cloud access reviews and security configuration checks."),
    ("george_adams", "George Adams", "Solutions Architect", "11", "AWS, APIs, Microservices, Stakeholder Management, Java", "BSc Information Technology", "Designed integration patterns for large enterprise clients."),
    ("helen_cho", "Helen Cho", "QA Engineer", "4", "Manual Testing, Playwright, JavaScript, Jira, API Testing", "BSc Information Systems", "Created end-to-end test plans for web application releases."),
    ("imran_siddiqui", "Imran Siddiqui", "Analytics Engineer", "5", "SQL, dbt, Snowflake, Python, Tableau", "BSc Computer Science", "Modeled trusted business metrics for self-service reporting."),
]

# Role profiles: the JD's required skills, adjacent skills, a focus phrase for the summary,
# achievement bullets, and degrees. Roles without a profile fall back to GENERIC_ACHIEVEMENTS.
ROLE_PROFILES = {
    "Data Scientist": (
        ["Python", "SQL", "Machine Learning", "TensorFlow", "AWS"],
        ["Statistics", "scikit-learn", "Pandas", "A/B Testing", "PyTorch", "Spark", "Tableau"],
        "building predictive models and turning experiments into product decisions",
        [
            "Built a churn-prediction model that let retention teams target the top 5% of at-risk accounts.",
            "Designed A/B test analyses for pricing changes and presented results to product leadership.",
            "Trained TensorFlow demand models and deployed batch scoring jobs on AWS.",
            "Created feature pipelines in SQL and Python that cut model refresh time from days to hours.",
            "Mentored two junior analysts on experiment design and model validation.",
            "Replaced spreadsheet forecasts with gradient-boosted models, reducing error by 12%.",
        ],
        ["MSc Data Science", "MSc Statistics", "BSc Mathematics"],
    ),
    "Machine Learning Engineer": (
        ["Python", "PyTorch", "MLOps", "Kubernetes", "MLflow"],
        ["Docker", "AWS", "SageMaker", "Kubeflow", "Feature Stores", "TensorFlow", "Airflow"],
        "taking machine-learning models from notebooks to reliable production services",
        [
            "Built a model-serving platform on Kubernetes handling 3,000 predictions per second.",
            "Introduced MLflow experiment tracking and a model registry across four teams.",
            "Automated retraining pipelines with drift monitoring and alerting.",
            "Optimized PyTorch inference with quantization, halving GPU costs.",
            "Containerized training workloads with Docker and standardized CI checks for models.",
            "Partnered with data scientists to productionize recommendation models.",
        ],
        ["MSc Machine Learning", "MTech Artificial Intelligence", "BSc Computer Science"],
    ),
    "Backend Engineer": (
        ["Python", "Django", "PostgreSQL", "Docker", "REST APIs"],
        ["Redis", "Celery", "FastAPI", "AWS", "Kafka", "GraphQL", "CI/CD"],
        "designing reliable APIs and data models for SaaS products",
        [
            "Designed REST APIs in Django serving 40 enterprise customers with 99.95% uptime.",
            "Tuned PostgreSQL queries and indexes, cutting p95 latency by 60%.",
            "Introduced Docker-based local environments and CI pipelines for six services.",
            "Built asynchronous job processing with Celery and Redis for report generation.",
            "Led the migration of a monolith module into a separately deployed service.",
            "Wrote integration tests that reduced production regressions by a third.",
        ],
        ["BSc Computer Science", "BSc Software Engineering", "BEng Computer Engineering"],
    ),
    "Data Engineer": (
        ["Python", "SQL", "Airflow", "Snowflake", "dbt"],
        ["Spark", "Kafka", "BigQuery", "AWS", "Terraform", "Data Quality", "Databricks"],
        "building scalable data pipelines and trusted analytical datasets",
        [
            "Built Airflow DAGs orchestrating 120 daily ingestion jobs into Snowflake.",
            "Modeled curated marts with dbt, adding tests and documentation for every model.",
            "Migrated legacy ETL scripts to Python and SQL with automated data-quality checks.",
            "Implemented incremental loads that cut warehouse compute costs by 30%.",
            "Set up streaming ingestion for clickstream events with Kafka.",
            "Defined data contracts with source teams to stop breaking schema changes.",
        ],
        ["BSc Information Systems", "BSc Computer Science", "BSc Data Analytics"],
    ),
    "Frontend Engineer": (
        ["TypeScript", "React", "Accessibility", "Testing"],
        ["Next.js", "Jest", "Cypress", "Storybook", "CSS", "GraphQL", "Performance Optimization"],
        "creating fast, accessible web experiences with design and product partners",
        [
            "Rebuilt the checkout UI in React and TypeScript, improving conversion by 9%.",
            "Led a WCAG 2.1 AA accessibility audit and fixed 140 issues across the app.",
            "Created a Storybook component library adopted by five product teams.",
            "Raised frontend test coverage from 35% to 80% with Jest and Cypress.",
            "Reduced bundle size by 40% through code splitting and lazy loading.",
            "Partnered with designers to prototype and validate new onboarding flows.",
        ],
        ["BSc Software Engineering", "BSc Digital Media", "BSc Computer Science"],
    ),
    "Cloud Security Engineer": (
        ["Python", "AWS", "IAM", "Cloud Security", "Threat Modeling"],
        ["Terraform", "SIEM", "Kubernetes", "Incident Response", "Compliance", "Zero Trust"],
        "automating cloud security controls and strengthening identity governance",
        [
            "Automated IAM least-privilege reviews across 60 AWS accounts with Python.",
            "Ran threat-modeling workshops for new services before launch.",
            "Built guardrails in Terraform that blocked public storage buckets by default.",
            "Led incident response for credential-exposure events and wrote postmortems.",
            "Mapped cloud controls to SOC 2 requirements and prepared audit evidence.",
            "Integrated cloud audit logs into the SIEM with tuned detection rules.",
        ],
        ["MSc Cybersecurity", "BSc Cybersecurity", "BSc Computer Science"],
    ),
    "Security Engineer": "Cloud Security Engineer",
    "DevOps Engineer": (
        ["AWS", "Terraform", "Kubernetes", "CI/CD", "Linux"],
        ["Docker", "Ansible", "GitHub Actions", "Prometheus", "Python", "Bash", "Helm"],
        "automating infrastructure and delivery pipelines",
        [
            "Codified AWS infrastructure in Terraform modules reused by 15 teams.",
            "Built CI/CD pipelines that took deployments from weekly to multiple times a day.",
            "Operated production Kubernetes clusters and Helm-based release workflows.",
            "Hardened Linux base images and automated patching.",
            "Cut cloud spend by 25% with rightsizing and scheduled scaling.",
            "Wrote runbooks and on-call tooling that reduced mean time to recovery.",
        ],
        ["BSc Computer Engineering", "BSc Computer Science", "BSc Information Technology"],
    ),
    "Full Stack Engineer": (
        ["JavaScript", "React", "Node.js", "PostgreSQL", "AWS"],
        ["TypeScript", "MongoDB", "GraphQL", "Docker", "Next.js", "Redis"],
        "shipping end-to-end product features across web and API layers",
        [
            "Delivered a customer portal end to end with React, Node.js and PostgreSQL.",
            "Built GraphQL APIs that replaced a dozen bespoke REST endpoints.",
            "Deployed services on AWS with infrastructure templates and autoscaling.",
            "Implemented role-based access control across frontend and backend.",
            "Improved page load time by 45% through caching and query optimization.",
            "Owned on-call for the web platform and drove fixes for top incidents.",
        ],
        ["BSc Computer Science", "BSc Software Engineering", "BSc Information Technology"],
    ),
    "NLP Engineer": (
        ["Python", "NLP", "Transformers", "Hugging Face", "PyTorch"],
        ["LLMs", "RAG", "FastAPI", "spaCy", "Vector Databases", "LangChain", "MLOps"],
        "building language-understanding systems and LLM-powered features",
        [
            "Fine-tuned Hugging Face transformer models for support-ticket classification.",
            "Built a retrieval-augmented generation service over 200,000 internal documents.",
            "Created evaluation sets and metrics to track LLM answer quality.",
            "Served NLP models behind FastAPI with batching and caching.",
            "Built named-entity extraction pipelines for contracts using spaCy.",
            "Reduced annotation cost with active-learning sampling.",
        ],
        ["MSc Computational Linguistics", "MSc Artificial Intelligence", "BSc Computer Science"],
    ),
    "Analytics Engineer": (
        ["SQL", "dbt", "Snowflake", "Python", "Data Modeling"],
        ["Looker", "Tableau", "Airflow", "BigQuery", "Git", "Data Quality"],
        "modeling trusted business metrics for self-service analytics",
        [
            "Built a dbt project of 150 tested models powering company KPIs.",
            "Designed dimensional models in Snowflake for finance and growth teams.",
            "Defined a metrics layer that ended conflicting revenue numbers across dashboards.",
            "Automated data-quality alerts for late and anomalous loads.",
            "Trained analysts on SQL style, Git workflows and code review.",
            "Built Looker explores that cut ad-hoc data requests by half.",
        ],
        ["BSc Computer Science", "BSc Economics", "BSc Data Analytics"],
    ),
    "Product Manager": (
        ["Product Strategy", "Roadmaps", "Agile", "SQL", "A/B Testing"],
        ["Figma", "Stakeholder Management", "User Research", "Analytics", "Jira", "Pricing"],
        "turning customer insight into prioritized product roadmaps",
        [
            "Owned the roadmap for an onboarding product used by 300,000 monthly users.",
            "Ran A/B tests on activation flows and shipped the winning variants.",
            "Wrote SQL to size opportunities and track feature adoption.",
            "Led Agile planning for two squads of engineers and designers.",
            "Interviewed customers and translated findings into product requirements.",
            "Aligned sales, support and engineering on a quarterly launch plan.",
        ],
        ["MBA", "BSc Computer Science", "BA Economics"],
    ),
    "Mobile Engineer": (
        ["Kotlin", "Android", "Jetpack Compose", "REST APIs", "Testing"],
        ["Swift", "iOS", "Firebase", "Flutter", "CI/CD", "Offline Sync"],
        "building polished, reliable mobile apps",
        [
            "Rebuilt the Android app UI in Jetpack Compose, cutting crash rate by 40%.",
            "Implemented offline sync for field workers with conflict resolution.",
            "Integrated REST APIs with retry and caching layers.",
            "Set up mobile CI with automated UI tests and staged rollouts.",
            "Shipped push notifications and deep links with Firebase.",
            "Improved app start time by 35% through profiling and lazy initialization.",
        ],
        ["BSc Computer Science", "BSc Software Engineering", "BEng Computer Engineering"],
    ),
    "Site Reliability Engineer": (
        ["Go", "Kubernetes", "Prometheus", "Grafana", "AWS"],
        ["Terraform", "Linux", "Incident Response", "Python", "SLOs", "Chaos Engineering"],
        "keeping production services reliable and observable",
        [
            "Defined SLOs and error budgets for 20 customer-facing services.",
            "Built Prometheus and Grafana dashboards with actionable alerts.",
            "Wrote Go tooling that automated failover drills.",
            "Led incident reviews and tracked remediation to completion.",
            "Reduced alert noise by 70% by consolidating and tuning rules.",
            "Ran capacity planning for peak seasonal traffic.",
        ],
        ["BSc Computer Engineering", "BSc Computer Science"],
    ),
    "QA Automation Engineer": (
        ["Python", "Selenium", "Playwright", "API Testing", "CI/CD"],
        ["Pytest", "Postman", "Performance Testing", "JavaScript", "Jira"],
        "building automated test suites that speed up releases",
        [
            "Built a Playwright suite of 600 end-to-end tests running on every merge.",
            "Automated API contract tests with Python and Pytest.",
            "Cut release verification from two days to three hours.",
            "Introduced flaky-test tracking and quarantine in CI.",
            "Ran load tests ahead of major launches and reported bottlenecks.",
            "Coached developers on writing testable code.",
        ],
        ["BSc Software Engineering", "BSc Information Systems"],
    ),
    "QA Engineer": "QA Automation Engineer",
    "Data Analyst": (
        ["SQL", "Python", "Power BI", "Excel", "Statistics"],
        ["Tableau", "Looker", "Pandas", "A/B Testing", "Google Analytics"],
        "turning business questions into clear reporting and analysis",
        [
            "Built recurring Power BI reports for sales and operations leadership.",
            "Automated weekly Excel reporting with Python, saving ten hours a week.",
            "Analyzed funnel drop-off and recommended changes that lifted signups.",
            "Wrote SQL queries and views used across finance and marketing.",
            "Presented quarterly insights to executives.",
            "Cleaned and reconciled data from three CRM systems.",
        ],
        ["BSc Economics", "BSc Statistics", "BBA Analytics"],
    ),
    "Java Engineer": (
        ["Java", "Spring Boot", "Kafka", "PostgreSQL", "Microservices"],
        ["Docker", "Kubernetes", "AWS", "REST APIs", "Redis", "gRPC"],
        "building event-driven backend services on the JVM",
        [
            "Built Spring Boot microservices processing 5 million orders a month.",
            "Introduced Kafka event streams to decouple order and inventory services.",
            "Optimized PostgreSQL schemas for high-write workloads.",
            "Led the upgrade from Java 11 to Java 21 across 30 services.",
            "Added distributed tracing to diagnose cross-service latency.",
            "Reviewed code and mentored three junior engineers.",
        ],
        ["BSc Software Engineering", "BSc Computer Science"],
    ),
    "Technical Recruiter": (
        ["Recruiting", "Sourcing", "ATS", "Interviewing", "Stakeholder Management"],
        ["LinkedIn Recruiter", "Employer Branding", "Excel", "Onboarding"],
        "hiring engineering and data talent",
        [
            "Hired 45 engineers in a year while keeping offer acceptance above 85%.",
            "Built structured interview loops with hiring managers.",
            "Sourced passive candidates for hard-to-fill data roles.",
            "Tracked funnel metrics in the ATS and reported weekly to leadership.",
        ],
        ["BA Human Resources", "BA Psychology"],
    ),
    "Technical Writer": (
        ["Technical Writing", "Markdown", "API Documentation", "Git", "Information Architecture"],
        ["Docs-as-Code", "OpenAPI", "UX Writing", "Python"],
        "making complex developer products easy to learn",
        [
            "Rewrote API reference docs from OpenAPI specs.",
            "Moved documentation to a docs-as-code workflow in Git.",
            "Created tutorials that reduced onboarding support tickets by 30%.",
            "Defined a style guide adopted across engineering.",
        ],
        ["BA English", "BA Communications"],
    ),
    "Customer Success Manager": (
        ["Customer Success", "Account Management", "Onboarding", "CRM", "Renewals"],
        ["SQL", "Salesforce", "Training", "Excel"],
        "growing and retaining enterprise customer accounts",
        [
            "Managed a portfolio of 40 enterprise accounts with 110% net retention.",
            "Ran onboarding programs that shortened time to first value.",
            "Partnered with product to prioritize top customer requests.",
            "Built health-score reporting in the CRM.",
        ],
        ["BA Business Administration", "BA Communications"],
    ),
    "Sales Engineer": (
        ["Solution Design", "Product Demos", "APIs", "Python", "Stakeholder Management"],
        ["AWS", "SQL", "Proof of Concept", "Salesforce"],
        "bridging customer needs and technical solutions",
        [
            "Ran technical demos and proofs of concept for enterprise prospects.",
            "Wrote Python integration samples for customer APIs.",
            "Supported 3 million dollars of closed revenue in a year.",
            "Captured product feedback from the field for engineering.",
        ],
        ["BSc Computer Science", "BEng Electrical Engineering"],
    ),
    "Project Manager": (
        ["Project Management", "Agile", "Scrum", "Jira", "Risk Management"],
        ["Stakeholder Management", "Budgeting", "Confluence", "Excel"],
        "delivering cross-functional projects on time and on budget",
        [
            "Delivered a platform migration across five teams on schedule.",
            "Ran Scrum ceremonies and tracked delivery metrics in Jira.",
            "Maintained risk registers and escalated blockers early.",
            "Managed budgets and vendor contracts for IT projects.",
        ],
        ["BA Business Administration", "BSc Information Systems"],
    ),
}

GENERIC_ACHIEVEMENTS = [
    "Partnered with cross-functional stakeholders to prioritize and deliver quarterly goals.",
    "Documented processes and mentored newer team members.",
    "Presented results to leadership and turned feedback into follow-up improvements.",
]

COMPANIES = [
    "Brightlane Systems", "Copperleaf Labs", "Harborview Health", "Northgate Retail", "Bluepeak Logistics",
    "Silverline Bank", "Meadowlark Media", "Ironwood Insurance", "Tidewater Energy", "Cobalt Commerce",
    "Riverstone Analytics", "Pinecrest Software", "Summit Freight", "Lattice Learning", "Orchard Telecom",
]
CITIES = ["Austin, TX", "Seattle, WA", "Toronto, ON", "London, UK", "Berlin, Germany", "Bengaluru, India",
          "Remote", "New York, NY", "Chicago, IL", "Dublin, Ireland", "Singapore", "Denver, CO"]
UNIVERSITIES = ["Westbrook University", "Lakeside Institute of Technology", "Cedar Valley University",
                "Northern Coast University", "Eastfield College"]

FIRST_NAMES = [
    "Adele", "Bilal", "Camila", "Dmitri", "Esther", "Femi", "Grace", "Hiro", "Ines", "Jonah", "Kavya", "Liam",
    "Mei", "Nikhil", "Olga", "Pablo", "Rania", "Stefan", "Thandi", "Uma", "Viktor", "Wanjiru", "Yara", "Zane",
    "Amara", "Bruno", "Chloe", "Daniel", "Emeka", "Freya", "Gustavo", "Hannah", "Ivan", "Julia", "Kenji",
    "Leila", "Mateo", "Naomi", "Omar", "Paula", "Rafael", "Sofia", "Tomas", "Valeria", "Yasmin", "Anika",
    "Beatriz", "Carmen", "Dario", "Elif", "Fiona", "Goran", "Helga", "Isla", "Joao", "Kiran", "Lucia", "Marek",
    "Nadia", "Oscar", "Petra", "Ravi", "Selin", "Tariq", "Ursula", "Vera", "Willem", "Ximena", "Yusra", "Zara",
    "Boris", "Dev", "Erik", "Farid", "Gemma",
]
LAST_NAMES = [
    "Achebe", "Bianchi", "Castillo", "Dimitrov", "Eriksen", "Fernandes", "Gupta", "Hoffmann", "Ibrahim",
    "Jansen", "Kowalski", "Larsen", "Moreau", "Nakamura", "Osei", "Popescu", "Quinn", "Reyes", "Sato",
    "Tanaka", "Usman", "Varga", "Whitfield", "Yilmaz", "Zhou", "Alvarez", "Brennan", "Chowdhury", "Delgado",
]

# Roles for new candidates: four per JD role (strong, partial, partial, weak) plus distractors.
JD_ROLES = [
    "Data Scientist", "Machine Learning Engineer", "Backend Engineer", "Data Engineer", "Frontend Engineer",
    "Cloud Security Engineer", "DevOps Engineer", "Full Stack Engineer", "NLP Engineer", "Analytics Engineer",
    "Product Manager", "Mobile Engineer",
]
DISTRACTOR_ROLES = [
    "Site Reliability Engineer", "QA Automation Engineer", "Data Analyst", "Java Engineer",
    "Site Reliability Engineer", "QA Automation Engineer", "Data Analyst", "Java Engineer",
    "Technical Recruiter", "Technical Writer", "Customer Success Manager", "Sales Engineer", "Project Manager",
    "Technical Recruiter", "Customer Success Manager", "Sales Engineer", "Project Manager",
]
TIERS = ["strong", "partial", "partial", "weak"]

JOBS_DATA = [
    ("data_scientist", "Data Scientist", 5, "Statistics, A/B Testing, PyTorch",
     ["Develop forecasting and classification models.", "Partner with product teams on experiments.",
      "Communicate results to non-technical stakeholders."]),
    ("ml_engineer", "Machine Learning Engineer", 5, "Docker, AWS, Feature Stores",
     ["Build, deploy and monitor machine-learning services in production.",
      "Own model registries, retraining and drift monitoring.", "Partner with data scientists on model handoff."]),
    ("backend_engineer", "Backend Engineer", 5, "Redis, Celery, AWS",
     ["Design reliable APIs, data models and integrations for a growing SaaS platform.",
      "Improve performance and observability of core services.", "Review code and mentor engineers."]),
    ("data_engineer", "Data Engineer", 4, "Spark, Kafka, Data Quality",
     ["Own scalable data pipelines and orchestration.", "Build tested, documented analytical datasets.",
      "Work with analysts and source teams on data contracts."]),
    ("frontend_engineer", "Frontend Engineer", 3, "Next.js, Storybook, Performance Optimization",
     ["Create fast, accessible customer experiences.", "Maintain a shared component library.",
      "Partner closely with design and product."]),
    ("cloud_security_engineer", "Cloud Security Engineer", 5, "Terraform, SIEM, Compliance",
     ["Automate cloud security controls and guardrails.", "Conduct risk reviews and threat modeling.",
      "Strengthen identity governance and incident response."]),
    ("devops_engineer", "DevOps Engineer", 4, "Helm, Prometheus, Python",
     ["Codify and operate cloud infrastructure.", "Build and maintain CI/CD pipelines.",
      "Improve reliability, cost and security of the platform."]),
    ("full_stack_engineer", "Full Stack Engineer", 4, "TypeScript, GraphQL, Docker",
     ["Ship product features end to end across frontend and backend.", "Design APIs and database schemas.",
      "Participate in on-call and production support."]),
    ("nlp_engineer", "NLP Engineer", 3, "LLMs, RAG, Vector Databases",
     ["Build language-understanding and LLM-powered features.", "Create evaluation datasets and quality metrics.",
      "Serve models efficiently in production."]),
    ("analytics_engineer", "Analytics Engineer", 3, "Looker, Airflow, Git",
     ["Model trusted business metrics in the warehouse.", "Maintain tests and documentation for data models.",
      "Enable self-service analytics for business teams."]),
    ("product_manager", "Product Manager", 5, "Figma, User Research, Pricing",
     ["Own the roadmap for a customer-facing product area.", "Run experiments and measure feature adoption.",
      "Align engineering, design and go-to-market teams."]),
    ("mobile_engineer", "Mobile Engineer", 4, "Firebase, Offline Sync, CI/CD",
     ["Build and maintain the Android app.", "Integrate backend APIs with robust error handling.",
      "Improve app performance, stability and release automation."]),
]


def profile(role: str):
    value = ROLE_PROFILES.get(role)
    return ROLE_PROFILES[value] if isinstance(value, str) else value


def new_people(rng: random.Random, taken: set[str]) -> list[tuple[str, str, str, str, str, str, str, str]]:
    """Generate (id, name, role, years, skills, education, achievement, tier) for new candidates."""
    roles = [(role, TIERS[i]) for role in JD_ROLES for i in range(4)] + [(role, "distractor") for role in DISTRACTOR_ROLES]
    names = [(f, l) for f in FIRST_NAMES for l in LAST_NAMES]
    rng.shuffle(names)
    people = []
    for role, tier in roles:
        first, last = names.pop()
        while first in taken:  # unique first names keep "why did <first name> rank higher" unambiguous
            first, last = names.pop()
        taken.add(first)
        required, extras, _, achievements, degrees = profile(role)
        if tier == "strong":
            skills = required + rng.sample(extras, 2)
            years = rng.randint(6, 12)
        elif tier == "partial":
            skills = rng.sample(required, len(required) - rng.randint(1, 2)) + rng.sample(extras, 3)
            years = rng.randint(3, 8)
        elif tier == "weak":
            skills = rng.sample(required, 2) + rng.sample(extras, 3)
            years = rng.randint(1, 4)
        else:
            skills = required + rng.sample(extras, 1)
            years = rng.randint(2, 11)
        identifier = f"{first}_{last}".lower()
        people.append((identifier, f"{first} {last}", role, str(years), ", ".join(skills),
                       rng.choice(degrees), rng.choice(achievements), tier))
    return people


def resume_sections(rng: random.Random, role: str, years: int, skills: str, education: str,
                    achievement: str) -> list[tuple[str, str]]:
    """Professional Summary / Skills / Experience / Education for one candidate."""
    prof = profile(role)
    focus = prof[2] if prof else "delivering measurable outcomes in collaborative teams"
    bullets = [b for b in (prof[3] if prof else GENERIC_ACHIEVEMENTS) if b != achievement]
    rng.shuffle(bullets)
    if prof and len(skills.split(", ")) < 7:
        adjacent = [s for s in prof[1] if s not in skills]
        skills = ", ".join(skills.split(", ") + rng.sample(adjacent, min(2, len(adjacent))))

    seniority = "Lead" if years >= 10 else "Senior" if years >= 6 else "" if years >= 3 else "Junior"
    position_count = 1 if years <= 3 else 2 if years <= 7 else 3
    splits = sorted(rng.sample(range(1, years), position_count - 1)) if position_count > 1 else []
    bounds = [0, *splits, years]  # year offsets from career start; the last span is the current job
    companies = rng.sample(COMPANIES, position_count)
    end_year = 2026
    career_start = end_year - years
    experience = []
    for index in range(position_count):  # most recent first
        start, stop = bounds[position_count - 1 - index], bounds[position_count - index]
        title = f"{seniority} {role}".strip() if index == 0 else (role if index == 1 else f"Associate {role}")
        period = f"{career_start + start} - {'Present' if index == 0 else career_start + stop}"
        experience.append(f"{title} | {companies[index]} | {period}")
        picks = [achievement] + bullets[:2] if index == 0 else bullets[2 * index:2 * index + 2]
        experience.extend(f"- {bullet}" for bullet in picks or GENERIC_ACHIEVEMENTS[:1])

    return [
        ("Professional Summary", f"{seniority + ' ' if seniority else ''}{role} with {years} years of experience {focus}."),
        ("Skills", skills),
        ("Experience", "\n".join(experience)),
        ("Education", f"{education}, {rng.choice(UNIVERSITIES)} ({end_year - years - rng.randint(0, 2)})"),
    ]


def jd_sections(title: str, years: int, required: str, nice: str, responsibilities: list[str],
                company: str, location: str) -> list[tuple[str, str]]:
    return [
        ("Role Overview", f"{company} is hiring {'an' if title[0] in 'AEIO' or title.startswith('NLP') else 'a'} {title} to join a cross-functional team. Location: {location}."),
        ("Minimum Experience", f"{years}+ years"),
        ("Required Skills", required),
        ("Nice-to-Have Skills", nice),
        ("Responsibilities", "\n".join(f"- {item}" for item in responsibilities)),
    ]


def escape_pdf(value: str) -> str:
    return value.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def write_pdf(path: Path, title: str, sections: list[tuple[str, str]], subtitle: str = "") -> Path:
    """Write a multi-page, word-wrapped PDF with bold headings using only the standard library."""
    lines: list[tuple[str, int, str]] = [("F2", 18, title)]
    if subtitle:
        lines.append(("F1", 10, subtitle))
    for heading, body in sections:
        lines += [("F1", 6, ""), ("F2", 13, heading)]
        for paragraph in body.splitlines():
            indent = "  " if paragraph.startswith("- ") else ""
            for number, wrapped in enumerate(textwrap.wrap(paragraph, 85, break_on_hyphens=False) or [""]):
                lines.append(("F1", 11, (indent if number else "") + wrapped))

    pages: list[list[str]] = [[]]
    y = 750
    for font, size, text in lines:
        y -= size + 5
        if y < 60:
            pages.append([])
            y = 750 - size - 5
        if text:
            pages[-1].append(f"BT /{font} {size} Tf 1 0 0 1 72 {y} Tm ({escape_pdf(text)}) Tj ET")

    objects = [b"<< /Type /Catalog /Pages 2 0 R >>", b"", b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
               b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold >>"]
    kids = []
    for page in pages:
        stream = "\n".join(page).encode("ascii")
        objects.append(b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream")
        objects.append(
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 3 0 R /F2 4 0 R >> >> "
            b"/Contents " + str(len(objects)).encode() + b" 0 R >>"
        )
        kids.append(f"{len(objects)} 0 R")
    objects[1] = f"<< /Type /Pages /Kids [{' '.join(kids)}] /Count {len(kids)} >>".encode()

    output = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = [0]
    for number, obj in enumerate(objects, start=1):
        offsets.append(len(output))
        output.extend(f"{number} 0 obj\n".encode())
        output.extend(obj)
        output.extend(b"\nendobj\n")
    xref_offset = len(output)
    output.extend(f"xref\n0 {len(objects) + 1}\n".encode())
    output.extend(b"0000000000 65535 f \n")
    output.extend(b"".join(f"{offset:010d} 00000 n \n".encode() for offset in offsets[1:]))
    output.extend(f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref_offset}\n%%EOF\n".encode())
    path.write_bytes(output)
    return path


def generate() -> dict[str, object]:
    RESUMES.mkdir(parents=True, exist_ok=True)
    JOBS.mkdir(parents=True, exist_ok=True)
    rng = random.Random(42)
    manifest: dict[str, object] = {
        "source": "Synthetic documents generated for RAGProfileMatching educational use.",
        "formats": [".pdf"],
        "resumes": [],
        "job_descriptions": [],
    }

    existing = [(*person, "original") for person in PEOPLE]
    people = existing + new_people(rng, {name.split()[0] for _, name, *_ in PEOPLE})
    for number, (identifier, name, role, years, skills, education, achievement, tier) in enumerate(people):
        sections = resume_sections(rng, role, int(years), skills, education, achievement)
        contact = f"{identifier.replace('_', '.')}@example.com | +1-555-01{number:02d} | {rng.choice(CITIES)}"
        pdf = write_pdf(RESUMES / f"{identifier}.pdf", name, sections, contact)
        manifest["resumes"].append({
            "id": identifier, "name": name, "role": role, "years_experience": int(years),
            "skills": sections[1][1].split(", "), "tier": tier, "pdf": str(pdf.relative_to(ROOT)),
        })

    for identifier, title, years, nice, responsibilities in JOBS_DATA:
        required = ", ".join(profile(title)[0])
        company, location = rng.choice(COMPANIES), rng.choice(CITIES)
        pdf = write_pdf(JOBS / f"{identifier}.pdf", title,
                        jd_sections(title, years, required, nice, responsibilities, company, location))
        manifest["job_descriptions"].append({
            "id": identifier, "title": title, "min_experience_years": years, "required_skills": required.split(", "),
            "nice_to_have_skills": nice.split(", "), "pdf": str(pdf.relative_to(ROOT)),
        })

    (ROOT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


if __name__ == "__main__":
    output = generate()
    print(f"Generated {len(output['resumes'])} resumes and {len(output['job_descriptions'])} job descriptions as PDF.")
