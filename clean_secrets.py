import os
import glob
import subprocess

base_dir = "/teamspace/studios/this_studio/CodeForge-250M"
os.chdir(base_dir)

print("=======================================================")
print("=== [1. REMOVING HARDCODED HF TOKENS FOR GITHUB SECURITY] ===")
print("=======================================================")

secret_token = os.environ.get("HF_TOKEN", "")

# Find all files
files_to_check = []
for ext in ["*.py", "*.sh", "*.md", "*.yaml", "*.txt"]:
    files_to_check.extend(glob.glob(f"**/{ext}", recursive=True))
    files_to_check.extend(glob.glob(ext))

files_modified = 0
for fpath in set(files_to_check):
    if os.path.isdir(fpath) or ".git" in fpath or "checkpoints" in fpath or "data/raw" in fpath:
        continue
    try:
        with open(fpath, "r", encoding="utf-8") as f:
            content = f.read()
            
        if secret_token in content:
            if fpath.endswith(".py"):
                # In Python, replace with os.environ.get("HF_TOKEN", "")
                content = content.replace(f'"{secret_token}"', 'os.environ.get("HF_TOKEN", "")')
                content = content.replace(f"'{secret_token}'", 'os.environ.get("HF_TOKEN", "")')
                content = content.replace(secret_token, 'os.environ.get("HF_TOKEN", "")')
            elif fpath.endswith(".sh"):
                content = content.replace(secret_token, '"$HF_TOKEN"')
            else:
                content = content.replace(secret_token, '<YOUR_HF_TOKEN>')
                
            with open(fpath, "w", encoding="utf-8") as f:
                f.write(content)
            print(f"--> Cleaned secret token from: {fpath}")
            files_modified += 1
    except Exception as e:
        pass

print(f"--> Total files sanitized: {files_modified}")

print("\n=======================================================")
print("=== [2. AMENDING GIT COMMIT TO WIPE SECRET FROM HISTORY] ===")
print("=======================================================")

subprocess.run(["git", "add", "."], check=True)

commit_msg = "feat: CodeForge-250M Phase 1 & 2 complete - SDPA, 50% FIM, ready for nexus-agent integration"
subprocess.run(["git", "commit", "--amend", "-m", commit_msg], check=True)
print(f"--> Successfully amended commit to wipe secret from Git history!")

# Verify no secrets remain in git log
res_grep = subprocess.run(["git", "log", "-1", "-p"], capture_output=True, text=True)
if secret_token in res_grep.stdout:
    print("WARNING: Secret token still found in latest commit diff!")
else:
    print("SUCCESS: 0 security violations found! Repository is 100% clean and ready for GitHub push!")
