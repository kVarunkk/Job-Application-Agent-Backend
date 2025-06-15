---
title: JOB-APPLICATION-AGENT
description: Automate your job applications on WorkAtAStartup using Python, LangChain, LangGraph, Playwright and OpenAI.
---

# JOB-APPLICATION-AGENT

A Python agent that automates job applications on [WorkAtAStartup](https://www.workatastartup.com/).  
It logs in, fetches job links, extracts job descriptions, matches them to your resume, generates tailored cover letters, and can even auto-apply for you.

### Note:

You will need:

- An OpenAI API key
- A PDF version of your resume
- Python 3.8+
- A Y Combinator account [Signup](https://account.ycombinator.com/?continue=https%3A%2F%2Fwww.workatastartup.com%2F)
- Completed Profile on the Y Combinator Job Portal: [WorkAtAStartup](https://www.workatastartup.com/)

## Setup Instructions

### 1. Clone the Repository

```sh
git clone <repo-url>
cd JOB-APPLICATION-AGENT
```

### 2. Create and Activate a Virtual Environment

**Windows:**

```sh
python -m venv venv
venv\Scripts\activate
```

**macOS/Linux:**

```sh
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```sh
pip install -r requirements.txt
```

### 4. Add Your OpenAI API Key

Create a `.env` file in the root directory with the following content:

```
OPENAI_API_KEY=your_openai_api_key_here
```

Replace `your_openai_api_key_here` with your actual OpenAI API key.

### 5. Add Your Resume

Upload a PDF version of your resume to the project root directory.  
Remember the file name (e.g., `resume.pdf`).

---

## Running the Script

Run the following command:

```sh
python app.py
```

You will be prompted to enter:

- **Your Y Combinator (WorkAtAStartup) username**
- **Your Y Combinator (WorkAtAStartup) password** (input is hidden)
- **The job portal filter URL** (e.g., copy from [WorkAtAStartup](https://www.workatastartup.com/companies?...))
- **The name of your resume PDF file** (e.g., `resume.pdf`)
- **The number of jobs you want that you want the agent to scan** (e.g., 5)

---

## What Happens Next?

1. The agent logs in to WorkAtAStartup using your credentials.
2. It fetches job links based on your filter URL.
3. It extracts job descriptions and compares them to your resume.
4. For good matches, it generates a tailored cover letter.
5. For each job, you will be shown the cover letter and asked if you want to apply (`y/n`).
6. If you choose to apply, the agent will submit your application automatically.

---

## Notes

- Make sure your `.env` file and resume PDF are in the project root.
- Do not share your credentials or API key with anyone.
- This script is for educational and personal use only.

---

## Troubleshooting

- If you encounter errors, check that all dependencies are installed and your API key is correct.
- Ensure your resume PDF is not corrupted and is readable by the script.

---

## Next Steps

- Setup Agent memory.
- Setup Agent Scheduling.
- Build a UI.

---

## License

MIT License

---

**Happy job hunting!**
