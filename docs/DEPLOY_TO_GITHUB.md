# Deploy MSBiGate-Ace to GitHub

Recommended public repository:

`https://github.com/xingleyun/MSBiGate-Ace`

## Create the repository

1. On GitHub, create a new **public** repository named `MSBiGate-Ace`.
2. Do **not** initialize it with a README, license, or `.gitignore`, because these files are already included.
3. Extract this release archive, open a terminal inside the `MSBiGate-Ace` folder, and run:

```bash
git init
git add .
git commit -m "Initial public release of MSBiGate-Ace"
git branch -M main
git remote add origin https://github.com/xingleyun/MSBiGate-Ace.git
git push -u origin main
```

## GitHub CLI alternative

```bash
gh repo create xingleyun/MSBiGate-Ace --public --source=. --remote=origin --push
```

## Publication note

This package intentionally does **not** include pretrained `.pt` checkpoints. It provides the benchmark data, source code, species-specific final configurations, and hyperparameter-search spaces needed to retrain the nine species-specific models.

If pretrained checkpoints are added in a future release, state that explicitly in the README and manuscript data/code availability statement.

## Suggested tagged release

After checking the public repository, freeze the version used for the manuscript:

```bash
git tag -a v1.0.0 -m "MSBiGate-Ace v1.0.0"
git push origin v1.0.0
```
