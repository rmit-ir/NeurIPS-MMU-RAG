# GitHub Actions Workflow for AWS ECR

This workflow builds and pushes the Docker image to AWS ECR automatically.

## Setup Instructions

### 1. Configure GitHub Secrets

You need to add the following secrets to your GitHub repository:

1. Go to your GitHub repository
2. Navigate to Settings → Secrets and variables → Actions
3. Add the following repository secrets:

- `AWS_ACCESS_KEY_ID`: Your AWS access key ID
- `AWS_SECRET_ACCESS_KEY`: Your AWS secret access key

### 2. Workflow Triggers

The workflow will run automatically on:

- Push to `main` or `master` branch
- Pull requests to `main` or `master` branch
- Manual trigger via GitHub Actions UI

### 3. Image Tagging Strategy

The workflow uses the following tagging strategy:

- `latest`: For pushes to the default branch
- `<branch-name>`: For pushes to feature branches
- `<branch-name>-<sha>`: For all commits
- `pr-<number>`: For pull requests

### 4. ECR Repository Details

- **Registry**: 970547356481.dkr.ecr.us-east-1.amazonaws.com
- **Repository**: neurips2025text/rmit-adms_ir
- **Region**: us-east-1
- **Exposed Port**: 5025

### 5. Manual Docker Commands (for reference)

If you need to build and push manually:

```bash
# Login to ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 970547356481.dkr.ecr.us-east-1.amazonaws.com

# Build the image
docker build -t neurips2025text/rmit-adms_ir .

# Tag the image
docker tag neurips2025text/rmit-adms_ir:latest 970547356481.dkr.ecr.us-east-1.amazonaws.com/neurips2025text/rmit-adms_ir:latest

# Push the image
docker push 970547356481.dkr.ecr.us-east-1.amazonaws.com/neurips2025text/rmit-adms_ir:latest
```

### 6. Team Information

- **Team ID**: f97a22bb-ef2b-4eda-a275-c451d474ef17
- **Team Name**: RMIT-ADMS IR

## Troubleshooting

- Ensure AWS credentials are correctly set in GitHub secrets
- Verify ECR repository exists and permissions are correct
- Check workflow logs in GitHub Actions tab for detailed error messages
