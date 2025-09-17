# EC2 Management Role

## Quick Deploy

First make sure you have the AWS CLI configured with your credentials.

Deploy the EC2 management role using CloudFormation:

```bash
aws cloudformation deploy \
  --template-file cloud/rag_server_proxy/ec2-management-role.yaml \
  --stack-name mmu-rag-ec2-management-role-stack \
  --capabilities CAPABILITY_NAMED_IAM \
  --region us-west-2
```

## Attach Role to EC2 Instance

After deployment, attach the role to your EC2 instance:

```bash
# Attach to existing EC2 instance
aws ec2 associate-iam-instance-profile \
    --instance-id YOUR_INSTANCE_ID \
    --iam-instance-profile Name=rmit-workload-ec2-management-profile \
    --region us-west-2
```

## Usage

Once attached, the EC2 instance can start/stop other instances in any region:

```bash
# Start an instance
aws ec2 start-instances --instance-ids i-xxxxxxxxx --region ap-southeast-2

# Stop an instance
aws ec2 stop-instances --instance-ids i-xxxxxxxxx --region us-east-1

# List instances
aws ec2 describe-instances --region eu-west-1
