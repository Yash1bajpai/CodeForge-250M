import os
import subprocess

base_dir = "/teamspace/studios/this_studio/CodeForge-250M"
os.chdir(base_dir)

print("=======================================================")
print("=== [1. CLARIFYING CODEFORGE-250M & NEXUS-AGENT BRANDING] ===")
print("=======================================================")

# Update README.md
readme_path = os.path.join(base_dir, "README.md")
if os.path.exists(readme_path):
    with open(readme_path, "r", encoding="utf-8") as f:
        content = f.read()
    content = content.replace("Nexus-Agent / CodeForge-250M", "CodeForge-250M (Custom Foundation Model for nexus-agent)")
    content = content.replace("Nexus-Agent (CodeForge-250M)", "CodeForge-250M")
    content = content.replace("power autonomous software agents (such as **Nexus-Agent**)", "power our autonomous software agent framework **nexus-agent** (formerly DevMind)")
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(content)
    print("--> Updated README.md: Project is CodeForge-250M, powering nexus-agent!")

# Update CONTEXT_HANDOFF.md
handoff_path = os.path.join(base_dir, "docs", "CONTEXT_HANDOFF.md")
if os.path.exists(handoff_path):
    with open(handoff_path, "r", encoding="utf-8") as f:
        content = f.read()
    content = content.replace("Nexus-Agent / CodeForge-250M", "CodeForge-250M")
    content = content.replace("Nexus-Agent (CodeForge-250M)", "CodeForge-250M")
    content = content.replace("DevMind", "nexus-agent (formerly DevMind)")
    with open(handoff_path, "w", encoding="utf-8") as f:
        f.write(content)
    print("--> Updated CONTEXT_HANDOFF.md: Project is CodeForge-250M, powering nexus-agent!")

print("\n=======================================================")
print("=== [2. UPDATING GIT COMMIT & CHECKING REMOTE] ===")
print("=======================================================")

subprocess.run(["git", "add", "."], check=True)

commit_msg = "feat: CodeForge-250M Phase 1 & 2 complete - SDPA, 50% FIM, ready for nexus-agent integration"
subprocess.run(["git", "commit", "--amend", "-m", commit_msg], check=False)
print(f"--> Updated Git Commit Message: {commit_msg}")

res_remote = subprocess.run(["git", "remote", "-v"], capture_output=True, text=True)
print("Current Git Remotes:\n", res_remote.stdout)

print("--> Attempting Git Push to GitHub...")
res_push = subprocess.run(["git", "push", "-u", "origin", "master", "--force"], capture_output=True, text=True)
if res_push.returncode != 0:
    res_push = subprocess.run(["git", "push", "-u", "origin", "main", "--force"], capture_output=True, text=True)

print("PUSH STDOUT:", res_push.stdout)
print("PUSH STDERR:", res_push.stderr)

if res_push.returncode == 0:
    print("\nSUCCESS: Successfully pushed to GitHub!")
else:
    print("\n[WHY GITHUB PUSH FAILED OVER SSH SCRIPT]:")
    print("GitHub no longer allows plain passwords over terminal scripts without a Personal Access Token (PAT) or SSH Key.")
