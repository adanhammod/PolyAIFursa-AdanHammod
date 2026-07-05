---
name: git-ship-to-dev
description: Use this skill when asked to ship the current branch. Stages and commits all changes, merges into dev, pushes dev, waits for the deploy-dev CI job to pass, then opens a PR from dev to main.

# Ship Current Branch to Dev

Follow these steps in order. Check the output of every command before moving to the next step. If any command fails unexpectedly, stop immediately and show the exact error message.

---

## Step 1 — Get the current branch

```bash
git rev-parse --abbrev-ref HEAD
```

Save the output as `<current-branch>`. This is the branch you will merge into dev.

---

## Step 2 — Stage all changes

```bash
git add -A
```

---

## Step 3 — Inspect what is staged

```bash
git diff --staged
```

Read the diff carefully. Summarise what changed (added files, modified logic, removed code). You will use this summary to write the commit message.

If the diff is empty (nothing staged), tell the user there is nothing to commit and stop here.

---

## Step 4 — Commit with a conventional commit message

Generate a commit message from the staged diff. The message must explain the **purpose** of the change, not just list which files were modified.

Follow the existing commit style in this repository:
- Descriptive verb phrases: "Refactor X to do Y", "Fix Z in W", "Add Q for R"
- Use conventional commit prefixes when the type is clear: `feat:`, `fix:`, `chore:`, `refactor:`, `test:`, `ci:`
- Keep the subject line under 72 characters

Examples of good messages:
```
feat(yolo): add S3 upload step before running inference
fix(ci): allow Docker Scout to comment on pull requests
chore(git): ship current branch through dev
refactor(agent): simplify tool-calling loop
```

Run the commit:
```bash
git commit -m "<your generated message>"
```

---

## Step 5 — Switch to dev and pull the latest

```bash
git checkout dev
git pull origin dev
```

---

## Step 6 — Merge the current branch into dev

```bash
git merge <current-branch>
```

### If there are merge conflicts

Do **not** auto-resolve conflicts. Never use `git merge -X ours` or `git merge -X theirs`.

Run:
```bash
git status
```

List every file shown under "both modified" or "unmerged". Then stop and say:

> There are merge conflicts in the following files:
> - [list the files]
>
> Please resolve these conflicts manually, then let me know and I will continue.

Wait for the user to resolve them. Do not proceed until they confirm.

After the user resolves conflicts, continue to Step 7.

---

## Step 7 — Verify the working tree is clean before pushing

```bash
git status
```

The output must show **"nothing to commit, working tree clean"** (or equivalent).

- If it shows untracked or modified files that are not part of the merge, check with the user before proceeding.
- If it shows unmerged paths, the conflicts are not fully resolved — stop and tell the user.

If the merge required a merge commit and git is waiting for it:
```bash
git commit
```
(This completes the merge commit with the default message.)

---

## Step 8 — Push dev

```bash
git push origin dev
```

Never use `--force` or `--force-with-lease` on dev or main.

---

## Step 9 — Save the pushed commit SHA

```bash
git rev-parse HEAD
```

Save this output as `<pushed-sha>`. You will use it to identify the correct CI run.

---

## Step 10 — Find the matching CI run

The `deploy.yaml` workflow triggers on every push to dev. Wait a few seconds for GitHub to register the run, then:

```bash
gh run list --branch dev --limit 5 --json databaseId,headSha,status,conclusion,displayTitle
```

Find the entry whose `headSha` matches `<pushed-sha>`. Save its `databaseId` as `<run-id>`.

If no matching run appears yet, wait 10 seconds and retry once.

---

## Step 11 — Watch the CI run

```bash
gh run watch <run-id> --exit-status
```

This streams live output and exits with a non-zero code if the run fails or is cancelled.

---

## Step 12 — If CI failed or was cancelled

```bash
gh run view <run-id> --log-failed
```

Show the failed job logs to the user, then stop. Do **not** open a PR.

---

## Step 13 — If CI passed, check for an existing PR

```bash
gh pr list --base main --head dev --state open
```

If a PR already exists, show its URL to the user. You are done — do not open a duplicate.

---

## Step 14 — Open a PR from dev to main

If no open PR exists:

```bash
gh pr create --base main --head dev --fill
```

Show the PR URL to the user.

> Note: Opening this PR triggers the `test.yaml` workflow, which runs pytest. You can monitor test results with:
> ```bash
> gh pr checks
> ```

---

## Summary of safety rules

| Rule | Detail |
|------|--------|
| No force push | Never use `--force` on dev or main |
| No auto-resolve | Never use `git merge -X ours` or `git merge -X theirs` |
| Stop on conflict | Run `git status`, list conflicted files, wait for the user |
| Verify before push | `git status` must be clean before `git push` |
| Check output | Read every command's output before the next step |
| Stop on error | Show the exact error; do not guess or retry blindly |
| PR only after CI | Only open the PR if the deploy-dev run concluded as `success` |
---
