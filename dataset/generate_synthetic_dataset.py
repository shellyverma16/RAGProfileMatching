#!/usr/bin/env python3
"""Generate a synthetic resume and job-description corpus for local RAG experiments."""

from __future__ import annotations

import json
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

JOBS_DATA = [
    ("data_scientist", "Data Scientist", "5+ years", "Python, SQL, Machine Learning, TensorFlow, AWS", "Develop forecasting and classification models; partner with product teams and communicate experiment results."),
    ("ml_engineer", "Machine Learning Engineer", "5+ years", "Python, PyTorch, MLOps, Kubernetes, MLflow", "Build, deploy, and monitor reliable machine-learning services in production."),
    ("backend_engineer", "Backend Engineer", "5+ years", "Python, Django, PostgreSQL, Docker, REST APIs", "Design reliable APIs, data models, and integrations for a growing SaaS platform."),
    ("data_engineer", "Data Engineer", "4+ years", "Python, SQL, Airflow, Snowflake, dbt", "Own scalable data pipelines, testing, and curated analytical datasets."),
    ("frontend_engineer", "Frontend Engineer", "3+ years", "TypeScript, React, Accessibility, Testing", "Create fast, accessible customer experiences in partnership with design and product."),
    ("cloud_security_engineer", "Cloud Security Engineer", "5+ years", "Python, AWS, IAM, Cloud Security, Threat Modeling", "Automate cloud security controls, conduct risk reviews, and strengthen identity governance."),
]


def escape_rtf(value: str) -> str:
    return value.replace("\\", "\\\\").replace("{", "\\{").replace("}", "\\}")


def write_rtf(path: Path, title: str, sections: list[tuple[str, str]]) -> None:
    content = [r"{\rtf1\ansi\deff0", r"{\fonttbl{\f0 Arial;}}", r"\fs24"]
    content.append(r"\b\fs32 " + escape_rtf(title) + r"\b0\fs24\par")
    for heading, body in sections:
        content.append(r"\par\b " + escape_rtf(heading) + r"\b0\par")
        content.append(escape_rtf(body) + r"\par")
    content.append("}")
    path.write_text("\n".join(content), encoding="ascii")


def escape_pdf(value: str) -> str:
    return value.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def write_pdf(path: Path, title: str, sections: list[tuple[str, str]]) -> Path:
    lines = [title, ""]
    for heading, body in sections:
        lines.append(heading)
        lines.extend(body.splitlines())
        lines.append("")
    text_commands = ["BT", "/F1 18 Tf", "72 750 Td"]
    for index, line in enumerate(lines):
        if index:
            text_commands.append("0 -20 Td")
            text_commands.append("/F1 11 Tf")
        text_commands.append(f"({escape_pdf(line)}) Tj")
    text_commands.append("ET")
    stream = "\n".join(text_commands).encode("ascii")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>",
        b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
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
    manifest: dict[str, object] = {
        "source": "Synthetic documents generated for RAGProfileMatching educational use.",
        "formats": [".rtf", ".pdf"],
        "resumes": [],
        "job_descriptions": [],
    }

    for identifier, name, role, years, skills, education, achievement in PEOPLE:
        rtf = RESUMES / f"{identifier}.rtf"
        write_rtf(
            rtf,
            name,
            [
                ("Professional Summary", f"{role} with {years} years of experience delivering measurable outcomes in collaborative teams."),
                ("Skills", skills),
                ("Experience", f"{role} | {years} years\n{achievement}"),
                ("Education", education),
            ],
        )
        write_pdf(
            rtf.with_suffix(".pdf"),
            name,
            [
                ("Professional Summary", f"{role} with {years} years of experience delivering measurable outcomes in collaborative teams."),
                ("Skills", skills),
                ("Experience", f"{role} | {years} years\n{achievement}"),
                ("Education", education),
            ],
        )
        manifest["resumes"].append(
            {"id": identifier, "name": name, "role": role, "years_experience": int(years), "rtf": str(rtf.relative_to(ROOT)), "pdf": str(rtf.with_suffix(".pdf").relative_to(ROOT))}
        )

    for identifier, title, experience, requirements, responsibilities in JOBS_DATA:
        rtf = JOBS / f"{identifier}.rtf"
        write_rtf(
            rtf,
            title,
            [
                ("Role Overview", f"We are hiring a {title} to join a cross-functional team."),
                ("Minimum Experience", experience),
                ("Required Skills", requirements),
                ("Responsibilities", responsibilities),
            ],
        )
        write_pdf(
            rtf.with_suffix(".pdf"),
            title,
            [
                ("Role Overview", f"We are hiring a {title} to join a cross-functional team."),
                ("Minimum Experience", experience),
                ("Required Skills", requirements),
                ("Responsibilities", responsibilities),
            ],
        )
        manifest["job_descriptions"].append(
            {"id": identifier, "title": title, "rtf": str(rtf.relative_to(ROOT)), "pdf": str(rtf.with_suffix(".pdf").relative_to(ROOT))}
        )

    (ROOT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


if __name__ == "__main__":
    output = generate()
    print(f"Generated {len(output['resumes'])} resumes and {len(output['job_descriptions'])} job descriptions in RTF and PDF.")
