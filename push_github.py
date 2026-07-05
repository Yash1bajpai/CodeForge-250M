import os
import subprocess
import re

base_dir = "/teamspace/studios/this_studio/CodeForge-250M"
os.chdir(base_dir)

print("=======================================================")
print("=== [1. RENAMING DEVMIND TO NEXUS-AGENT IN DOCS] ===")
print("=======================================================")

# Update README.md
readme_path = os.path.join(base_dir, "README.md")
if os.path.exists(readme_path):
    with open(readme_path, "r", encoding="utf-8") as f:
        content = f.read()
    content = content.replace("DevMind / CodeForge-250M", "Nexus-Agent / CodeForge-250M")
    content = content.replace("DevMind (CodeForge-250M)", "Nexus-Agent (CodeForge-250M)")
    content = content.replace("DevMind", "Nexus-Agent")
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(content)
    print("--> Updated README.md with Nexus-Agent branding!")

# Update CONTEXT_HANDOFF.md
handoff_path = os.path.join(base_dir, "docs", "CONTEXT_HANDOFF.md")
if os.path.exists(handoff_path):
    with open(handoff_path, "r", encoding="utf-8") as f:
        content = f.read()
    content = content.replace("DevMind", "Nexus-Agent")
    with open(handoff_path, "w", encoding="utf-8") as f:
        f.write(content)
    print("--> Updated CONTEXT_HANDOFF.md with Nexus-Agent branding!")

print("\n=======================================================")
print("=== [2. GITHUB CONFIGURATION & PUSH AUTOMATION] ===")
print("=======================================================")

# Ensure git is initialized
if not os.path.exists(".git"):
    subprocess.run(["git", "init"], check=True)
    print("--> Initialized Git repository.")

# Set git user info if not set
subprocess.run(["git", "config", "user.name", "Yash Bajpai"], check=False)
subprocess.run(["git", "config", "user.email", "yash1bajpai@gmail.com"], check=False)

# Check remote
res_remote = subprocess.run(["git", "remote", "-v"], capture_output=True, text=True)
print("Current Git Remotes:\n", res_remote.stdout if res_remote.stdout else "No remote set yet.")

# If no remote named origin is set, let's check or set a default GitHub repo
if "origin" not in res_remote.stdout:
    # We will check if there is an existing repo or try to add one
    default_repo = "https://github.com/Yash1bajpai/CodeForge-250M.git"
    subprocess.run(["git", "remote", "add", "origin", default_repo], check=False)
    print(f"--> Added default Git remote origin: {default_repo}")

# Stage all files (respecting .gitignore)
subprocess.run(["git", "add", "."], check=True)
print("--> Staged all code, configs, scripts, and documentation.")

# Check if there is anything to commit
res_status = subprocess.run(["git", "status"], capture_output=True, text=True)
print("Git Status Summary:\n", res_status.stdout)

# Commit changes
commit_msg = "feat: Nexus-Agent Phase 1 & 2 complete - SDPA, 50% FIM, MinHash dedup, 2M token training verified"
res_commit = subprocess.run(["git", "commit", "-m", commit_msg], capture_output=True, text=True)
if res_commit.returncode == 0:
    print(f"--> Created commit: {commit_msg}")
else:
    print("--> Commit output:", res_commit.stdout or res_commit.stderr)

# Try pushing to origin main or master
print("--> Attempting to push to GitHub remote origin...")
res_push = subprocess.run(["git", "push", "-u", "origin", "main"], capture_output=True, text=True)
if res_push.returncode != 0:
    res_push = subprocess.run(["git", "push", "-u", "origin", "master"], capture_output=True, text=True)

print("PUSH STDOUT:", res_push.stdout)
print("PUSH STDERR:", res_push.stderr)

if res_push.returncode == 0:
    print("\nSUCCESS: All code and progress successfully pushed to GitHub!")
else:
    print("\n[Note on Push Authentication] Git push requires GitHub authentication/credential on the studio.")
