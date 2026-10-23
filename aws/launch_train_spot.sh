#!/usr/bin/env bash
# Stage 4/8: launch the full pretrain on an EC2 Spot g5.xlarge.
# Prereqs (operator fills in): BUCKET, KEY_NAME, SECURITY_GROUP, AMI (Ubuntu 22.04 + CUDA driver),
# an instance profile (IAM role) attached with aws/iam-policy.json, repo + data on S3.
# The instance syncs data to local NVMe first (never trains off S3), trains resumably,
# and syncs checkpoints to S3 every --ckpt-interval steps.
set -euo pipefail

BUCKET=TODO-BUCKET
REGION=TODO-REGION
AMI=TODO-AMI-ID          # Ubuntu 22.04 with NVIDIA driver (e.g. from AWS Marketplace deep learning AMI)
KEY_NAME=TODO-KEY
IAM_PROFILE=TODO-INSTANCE-PROFILE
INSTANCE_TYPE=g5.xlarge  # A10G 24GB; right-sized for ~110M params, see docs/stage4-sizing.md

# Spot request (interruptions tolerated by resumable checkpoints synced to S3)
SPOT_JSON=$(aws ec2 request-spot-instances \
  --region "$REGION" \
  --spot-price "0.60" \
  --instance-count 1 \
  --type "one-time" \
  --launch-specification "{
    \"ImageId\": \"$AMI\",
    \"InstanceType\": \"$INSTANCE_TYPE\",
    \"KeyName\": \"$KEY_NAME\",
    \"IamInstanceProfile\": {\"Name\": \"$IAM_PROFILE\"},
    \"SecurityGroupIds\": [\"TODO-SECURITY-GROUP\"],
    \"BlockDeviceMappings\": [{
      \"DeviceName\": \"/dev/sda1\",
      \"Ebs\": {\"VolumeSize\": 100, \"VolumeType\": \"gp3\"}
    }],
    \"UserData\": \"$(base64 -w0 aws/userdata_train.sh)\"
  }")
echo "$SPOT_JSON"
echo "Watch: aws ec2 describe-spot-instance-requests --region $REGION"
echo "Then train.py resumes automatically from checkpoints/last.pt in S3 if present."
