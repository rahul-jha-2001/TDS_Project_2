
import os
import subprocess
import dotenv
dotenv.load_dotenv()

# Predefined EVALUATOR URL
EVALUATOR_LINK = "https://raw.githubusercontent.com/sanand0/tools-in-data-science-public/tds-2024-t3/project2/evaluate.py"

# Prompt user for the API Token and Submission URL
AIPROXY_TOKEN = os.environ.get("AIPROXY_TOKEN")
SUBMISSION_LINK = "https://raw.githubusercontent.com/rahul-jha-2001/TDS_Project_2/refs/heads/main/autolysis.py"

# Set environment variables
os.environ["AIPROXY_TOKEN"] = AIPROXY_TOKEN
os.environ["SUBMISSION"] = SUBMISSION_LINK
os.environ["EVALUATOR"] = EVALUATOR_LINK

# Construct the command
command = f"uv run {os.environ['EVALUATOR']} {os.environ['SUBMISSION']}"

# Run the command using subprocess and capture output
result = subprocess.run(command, shell=True, capture_output=True, text=True)

# Print the stdout and stderr (output and error logs)
print("STDOUT:", result.stdout)
print("STDERR:", result.stderr)
print("Return Code:", result.returncode)
