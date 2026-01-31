# Instructions to Push N_grams Folder to GitHub

## Repository Information
- **Repository URL**: https://github.com/ICS-NLP/Prosit1.git
- **Organization**: ICS-NLP
- **Repository Name**: Prosit1

## Quick Start (Automated Script)

1. **Run the automated script**:
   ```bash
   cd "/Users/macpro/Desktop/Masters ICS/Natural Language Processing/section_b_ngram/N_grams"
   ./push_to_github.sh
   ```

   The script will:
   - Find or clone the Prosit1 repository
   - Copy the N_grams folder to it
   - Add, commit, and push to GitHub

## Manual Method

### Step 1: Clone or Navigate to Repository

If you haven't cloned the repository yet:
```bash
cd ~/Desktop  # or any directory you prefer
git clone https://github.com/ICS-NLP/Prosit1.git
cd Prosit1
```

If you already have it cloned, navigate to it:
```bash
cd /path/to/Prosit1
```

### Step 2: Copy N_grams Folder

```bash
# Copy the N_grams folder to the repository
cp -r "/Users/macpro/Desktop/Masters ICS/Natural Language Processing/section_b_ngram/N_grams" .
```

### Step 3: Add and Commit

```bash
# Check status
git status

# Add the N_grams folder
git add N_grams/

# Commit with a message
git commit -m "Add N-gram language model implementation for Akan language"
```

### Step 4: Push to GitHub

```bash
# Push to main branch (or master if that's your default)
git push origin main
# OR
git push origin master
```

## Authentication

Since this is a GitHub organization repository, you may need to authenticate:

### Option 1: SSH (Recommended)
```bash
# Set remote URL to use SSH
git remote set-url origin git@github.com:ICS-NLP/Prosit1.git
git push origin main
```

### Option 2: Personal Access Token (HTTPS)
```bash
# When prompted for password, use your personal access token
# Create one at: https://github.com/settings/tokens
git push origin main
```

### Option 3: GitHub CLI
```bash
gh auth login
git push origin main
```

## Verify Push

After pushing, verify the files are on GitHub:
1. Go to https://github.com/ICS-NLP/Prosit1
2. Check that the `N_grams` folder appears
3. Verify all files are present:
   - Code files (preprocessing.py, ngram_model.py, etc.)
   - Models folder (akan_best.pkl, akan_best_best_2gram.pkl)
   - Output files (output.txt, output2.txt, output3.txt)
   - README.md

## Troubleshooting

### Issue: "Permission denied" or "Authentication failed"
**Solution**: Set up SSH keys or use a personal access token
```bash
# Generate SSH key (if you don't have one)
ssh-keygen -t ed25519 -C "your_email@example.com"

# Add to GitHub: Settings → SSH and GPG keys → New SSH key
# Then use SSH URL:
git remote set-url origin git@github.com:ICS-NLP/Prosit1.git
```

### Issue: "Repository not found"
**Solution**: Make sure you have access to the ICS-NLP organization repository. Contact the repository owner if needed.

### Issue: "Branch 'main' does not exist"
**Solution**: The repository might use 'master' instead:
```bash
git push origin master
```

### Issue: "Updates were rejected"
**Solution**: Pull latest changes first:
```bash
git pull origin main --rebase
git push origin main
```

## What's Being Pushed

The N_grams folder contains:
- ✅ All code files (preprocessing.py, ngram_model.py, smoothing.py, etc.)
- ✅ Trained models (akan_best.pkl, akan_best_best_2gram.pkl)
- ✅ Output files (output.txt, output2.txt, output3.txt)
- ✅ README.md with complete documentation
- ✅ requirements.txt

## After Pushing

1. Verify the push was successful on GitHub
2. Update the main repository README.md if needed to reference the N_grams folder
3. Share the repository link with your team members

## Need Help?

If you encounter issues:
1. Check your git configuration: `git config --list`
2. Verify your GitHub access: `gh auth status` (if using GitHub CLI)
3. Check repository permissions in the ICS-NLP organization settings
