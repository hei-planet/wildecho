# Create the WildEcho repository in the GitHub organization

The commands below assume the organization is `heiplanet`. Change `ORG` if the organization uses a different GitHub name.

## Install and authenticate GitHub CLI

### Arch Linux

```bash
sudo pacman -S --needed github-cli
gh auth login
```

### macOS

```bash
brew install gh
gh auth login
```

Choose:

1. GitHub.com
2. HTTPS
3. Login with a web browser

Verify the login:

```bash
gh auth status
```

## Create a fresh repository

A fresh Git history is recommended because an older development copy contained a token. Do not push the old `.git` history.

```bash
cd ~/Projects/wildecho

rm -rf .git
git init -b main
git add .
git commit -m "Initial release: WildEcho v0.5.0"

ORG="heiplanet"
gh repo create "$ORG/wildecho" \
  --private \
  --description "Fast local analysis of AudioMoth recordings with BirdNET, speech detection, and speaker diarization." \
  --source=. \
  --remote=origin \
  --push
```

Open the repository:

```bash
gh repo view --web
```

To make it public later:

```bash
gh repo edit "$ORG/wildecho" --visibility public
```
