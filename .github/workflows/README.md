# GitHub Actions Workflow for AWS ECR

Simple workflow that builds and pushes a Docker image to AWS ECR for x64 architecture.

## Setup Instructions

### 1. Configure GitHub Secrets

Add the following secrets to your GitHub repository:

1. Go to Settings → Secrets and variables → Actions
2. Add these repository secrets:
   - `AWS_ACCESS_KEY_ID`: Your AWS access key ID
   - `AWS_SECRET_ACCESS_KEY`: Your AWS secret access key

### 2. Workflow Details

- **Triggers**: Push to `main`/`master` branch or manual trigger
- **Platform**: Linux/AMD64 (x64) only
- **Tag**: Always uses `latest`
- **Registry**: 970547356481.dkr.ecr.us-east-1.amazonaws.com
- **Repository**: neurips2025text/rmit-adms_ir
- **Port**: 5025
- More details in the email, search for "MMU-RAGent text-2-text track"

### 3. Manual Build (if needed)

```bash
# Login to ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 970547356481.dkr.ecr.us-east-1.amazonaws.com

# Build and push
docker build --platform linux/amd64 -t 970547356481.dkr.ecr.us-east-1.amazonaws.com/neurips2025text/rmit-adms_ir:latest .
docker push 970547356481.dkr.ecr.us-east-1.amazonaws.com/neurips2025text/rmit-adms_ir:latest
```

### 4. Team Information

- **Team ID**: f97a22bb-ef2b-4eda-a275-c451d474ef17
- **Team Name**: RMIT-ADMS IR
