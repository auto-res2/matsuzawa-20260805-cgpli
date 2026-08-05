#!/usr/bin/env bash
set -euo pipefail

# Cloud GPU Runner Manager for GitHub Actions
#
# Launches an on-demand GPU instance on the configured cloud provider
# and registers it as a GitHub Actions self-hosted runner.
# Instance teardown is handled by the on-instance watchdog
# (auto-shutdown after job completes or max lifetime exceeded).
#
# Usage:
#   cloud-gpu-runner.sh start
#
# Required env vars (common):
#   CLOUD_PROVIDER        — "aws" or "gcp"
#   GH_PERSONAL_ACCESS_TOKEN — GitHub PAT (repo scope)
#   GITHUB_REPOSITORY     — owner/repo (set by GitHub Actions)
#   GITHUB_RUN_ID         — workflow run ID (set by GitHub Actions)
#
# Required env vars (AWS):
#   AWS_ACCESS_KEY_ID     — AWS credentials (via configure-aws-credentials action)
#   AWS_SECRET_ACCESS_KEY — AWS credentials (via configure-aws-credentials action)
#   All other AWS settings (region, AMI, subnet, SG) are auto-resolved.
#
# Optional env vars (AWS):
#   GPU_INSTANCE_TYPE (default: g4dn.xlarge)
#   GPU_INSTANCE_TYPE_FALLBACKS (default: g5.xlarge,g6.xlarge,g4dn.2xlarge,g5.2xlarge)
#   AWS_REGION (preferred region; falls back to other regions if unavailable)
#   AWS_REGION_FALLBACKS (comma-separated fallback regions)
#   EC2_AMI_ID (auto-resolved from SSM: Deep Learning AMI Ubuntu 22.04)
#   EC2_SUBNET_ID (falls back to default VPC subnet)
#   EC2_SECURITY_GROUP_ID (falls back to default security group)
#   EC2_IAM_INSTANCE_PROFILE
#
# Required env vars (GCP):
#   GCP_PROJECT, GCP_ZONE, GCP_MACHINE_IMAGE
#   GCP_ACCELERATOR_TYPE, GCP_ACCELERATOR_COUNT (optional, default: 1)
#   GCP_SUBNET (optional)

COMMAND="${1:-}"
RUNNER_LABEL="gpu-${GITHUB_RUN_ID:-$(date +%s)}-${RANDOM}"
RUNNER_NAME="github-runner-${GITHUB_RUN_ID:-$(date +%s)}-${RANDOM}"
MAX_WAIT_SECONDS=600
POLL_INTERVAL=10
MAX_INSTANCE_HOURS="${MAX_INSTANCE_HOURS:-120}"
RUNNER_PROCESS_POLL=60

log() {
  echo "$(date -u '+%Y-%m-%d %H:%M:%S UTC') [cloud-gpu-runner] $*" >&2
}

warn() {
  local msg="$(date -u '+%Y-%m-%d %H:%M:%S UTC') [cloud-gpu-runner] WARNING: $*"
  echo "::warning::${msg}"
  echo "${msg}" >&2
}

get_runner_version() {
  local response version
  response=$(curl -s https://api.github.com/repos/actions/runner/releases/latest)
  version=$(echo "$response" | jq -r '.tag_name' | sed 's/^v//')

  if [[ -z "$version" || "$version" == "null" ]]; then
    log "ERROR: Failed to fetch latest runner version"
    log "API response: $(echo "$response" | jq -r '.message // "unknown error"')"
    return 1
  fi
  echo "$version"
}

# ============================================================
# GitHub Runner helpers (provider-agnostic)
# ============================================================

get_registration_token() {
  local response token
  response=$(curl -s -X POST \
    -H "Authorization: token ${GH_PERSONAL_ACCESS_TOKEN}" \
    -H "Accept: application/vnd.github+json" \
    "https://api.github.com/repos/${GITHUB_REPOSITORY}/actions/runners/registration-token")
  token=$(echo "$response" | jq -r '.token')

  if [[ -z "$token" || "$token" == "null" ]]; then
    log "ERROR: Failed to get registration token"
    log "API response: $(echo "$response" | jq -r '.message // "unknown error"')"
    return 1
  fi
  echo "$token"
}

generate_user_data() {
  local reg_token="$1"
  local runner_version="$2"
  cat <<USERDATA
#!/bin/bash
set -e
set +x
exec > /var/log/runner-setup.log 2>&1
echo "=== Runner setup started at \$(date -u) ==="

useradd -m -s /bin/bash runner || true
RUNNER_DIR="/home/runner/actions-runner"
mkdir -p "\$RUNNER_DIR"
cd "\$RUNNER_DIR"

curl -sL "https://github.com/actions/runner/releases/download/v${runner_version}/actions-runner-linux-x64-${runner_version}.tar.gz" -o runner.tar.gz
tar xzf runner.tar.gz
rm runner.tar.gz
./bin/installdependencies.sh

chown -R runner:runner "\$RUNNER_DIR"
sudo -u runner env ACTIONS_RUNNER_INPUT_TOKEN="${reg_token}" ./config.sh \\
  --url "https://github.com/${GITHUB_REPOSITORY}" \\
  --name "${RUNNER_NAME}" \\
  --labels "${RUNNER_LABEL}" \\
  --unattended \\
  --ephemeral \\
  --replace

./svc.sh install runner
./svc.sh start
echo "=== Runner setup completed at \$(date -u) ==="

# --- Watchdog: process monitor + absolute timeout ---
_WD_MAX=\$(( ${MAX_INSTANCE_HOURS} * 3600 ))
_WD_POLL=${RUNNER_PROCESS_POLL}

cat > /usr/local/bin/instance-watchdog.sh <<'WATCHDOG'
#!/bin/bash
BOOT_TIME=\$(date +%s)

while true; do
  sleep __POLL__

  # Check 1: Runner process exited (ephemeral job completed)
  if ! pgrep -f 'Runner.Listener' > /dev/null 2>&1; then
    echo "\$(date -u) [watchdog] Runner process not found. Job completed. Shutting down."
    shutdown -h now
    exit 0
  fi

  # Check 2: Absolute max lifetime exceeded
  ELAPSED=\$(( \$(date +%s) - BOOT_TIME ))
  if [[ \$ELAPSED -ge __MAX__ ]]; then
    echo "\$(date -u) [watchdog] Max lifetime (__MAX__s) exceeded. Shutting down."
    shutdown -h now
    exit 0
  fi
done
WATCHDOG
sed -i "s/__MAX__/\${_WD_MAX}/g" /usr/local/bin/instance-watchdog.sh
sed -i "s/__POLL__/\${_WD_POLL}/g" /usr/local/bin/instance-watchdog.sh
chmod +x /usr/local/bin/instance-watchdog.sh
nohup /usr/local/bin/instance-watchdog.sh >> /var/log/instance-watchdog.log 2>&1 &
echo "=== Watchdog started ==="
USERDATA
}

wait_for_runner_online() {
  local elapsed=0
  log "Waiting for runner '${RUNNER_NAME}' to come online (max ${MAX_WAIT_SECONDS}s)..."

  while [[ $elapsed -lt $MAX_WAIT_SECONDS ]]; do
    local status
    status=$(curl -s \
      -H "Authorization: token ${GH_PERSONAL_ACCESS_TOKEN}" \
      -H "Accept: application/vnd.github+json" \
      "https://api.github.com/repos/${GITHUB_REPOSITORY}/actions/runners" \
      | jq -r ".runners[] | select(.name == \"${RUNNER_NAME}\") | .status" 2>/dev/null || echo "")

    if [[ "$status" == "online" ]]; then
      log "Runner '${RUNNER_NAME}' is online"
      return 0
    fi

    log "Runner not online yet (status: ${status:-not found}), waiting ${POLL_INTERVAL}s... (${elapsed}/${MAX_WAIT_SECONDS}s)"
    sleep "$POLL_INTERVAL"
    elapsed=$((elapsed + POLL_INTERVAL))
  done

  log "ERROR: Runner did not come online within ${MAX_WAIT_SECONDS}s"
  return 1
}

# ============================================================
# AWS provider
# ============================================================

aws_resolve_ami() {
  if [[ -n "${EC2_AMI_ID:-}" && "${AWS_DEFAULT_REGION}" == "${_PRIMARY_REGION:-}" ]]; then
    echo "$EC2_AMI_ID"
    return
  fi

  local ssm_path="/aws/service/deeplearning/ami/x86_64/oss-nvidia-driver-gpu-pytorch-2.7-ubuntu-22.04/latest/ami-id"
  local ami_id
  ami_id=$(aws ssm get-parameter \
    --region "${AWS_DEFAULT_REGION}" \
    --name "$ssm_path" \
    --query 'Parameter.Value' \
    --output text 2>/dev/null || echo "")

  if [[ -z "$ami_id" || "$ami_id" == "None" ]]; then
    log "ERROR: Failed to resolve AMI from SSM ($ssm_path)"
    return 1
  fi
  log "Resolved AMI from SSM: ${ami_id}"
  echo "$ami_id"
}

aws_resolve_subnets() {
  if [[ -n "${EC2_SUBNET_ID:-}" && "${AWS_DEFAULT_REGION}" == "${_PRIMARY_REGION:-}" ]]; then
    echo "$EC2_SUBNET_ID"
    return
  fi

  local subnets
  subnets=$(aws ec2 describe-subnets \
    --filters "Name=default-for-az,Values=true" \
    --query 'Subnets[*].SubnetId' \
    --output text 2>/dev/null || echo "")

  if [[ -z "$subnets" || "$subnets" == "None" ]]; then
    log "ERROR: No default subnets found"
    return 1
  fi
  log "Found default subnets: ${subnets}"
  echo "$subnets"
}

aws_resolve_security_group() {
  if [[ -n "${EC2_SECURITY_GROUP_ID:-}" && "${AWS_DEFAULT_REGION}" == "${_PRIMARY_REGION:-}" ]]; then
    echo "$EC2_SECURITY_GROUP_ID"
    return
  fi

  local sg_id
  sg_id=$(aws ec2 describe-security-groups \
    --filters "Name=group-name,Values=default" \
    --query 'SecurityGroups[0].GroupId' \
    --output text 2>/dev/null || echo "")

  if [[ -z "$sg_id" || "$sg_id" == "None" ]]; then
    log "ERROR: No default security group found"
    return 1
  fi
  log "Using default security group: ${sg_id}"
  echo "$sg_id"
}

# Try launching in a single region. Outputs instance_id on success.
aws_try_region() {
  local region="$1"
  shift
  local -a type_list=("$@")

  log "=== Trying region: ${region} ==="
  export AWS_DEFAULT_REGION="$region"
  export AWS_REGION="$region"

  local ami_id subnets sg_id
  ami_id=$(aws_resolve_ami) || { log "Skipping ${region}: failed to resolve AMI"; return 1; }
  subnets=$(aws_resolve_subnets) || { log "Skipping ${region}: no subnets found"; return 1; }
  sg_id=$(aws_resolve_security_group) || { log "Skipping ${region}: no security group found"; return 1; }

  local user_data
  user_data=$(generate_user_data "$reg_token" "$runner_version" | base64 -w 0)

  local iam_flag=""
  if [[ -n "${EC2_IAM_INSTANCE_PROFILE:-}" ]]; then
    iam_flag="--iam-instance-profile Name=${EC2_IAM_INSTANCE_PROFILE}"
  fi

  local launch_err_file
  launch_err_file=$(mktemp)

  local instance_id=""
  for instance_type in "${type_list[@]}"; do
    for subnet_id in $subnets; do
      log "Trying ${instance_type} in ${region} / ${subnet_id}..."

      instance_id=$(aws ec2 run-instances \
        --region "${region}" \
        --image-id "${ami_id}" \
        --instance-type "${instance_type}" \
        --subnet-id "${subnet_id}" \
        --security-group-ids "${sg_id}" \
        --user-data "$user_data" \
        --instance-initiated-shutdown-behavior terminate \
        --metadata-options "HttpTokens=required,HttpEndpoint=enabled" \
        ${iam_flag} \
        --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=${RUNNER_NAME}},{Key=github-actions,Value=true},{Key=github-run-id,Value=${GITHUB_RUN_ID:-unknown}}]" \
        --query 'Instances[0].InstanceId' \
        --output text 2>"$launch_err_file" || true)

      if [[ -n "$instance_id" && "$instance_id" != "None" ]]; then
        log "Successfully launched ${instance_type} in ${region}"
        rm -f "$launch_err_file"
        _LAUNCHED_INSTANCE_ID="$instance_id"
        return 0
      fi
      local launch_err
      launch_err=$(cat "$launch_err_file")
      if grep -qE "VcpuLimitExceeded|PendingVerification" "$launch_err_file"; then
        _REGION_FAIL_REASON=$(grep -oE "VcpuLimitExceeded|PendingVerification" "$launch_err_file" | head -1)
        warn "${region}: ${_REGION_FAIL_REASON} (skipping region)"
        rm -f "$launch_err_file"
        return 1
      elif grep -qE "Unsupported|InvalidAMIID\.NotFound|InsufficientInstanceCapacity|InvalidSubnetID\.NotFound" "$launch_err_file"; then
        _REGION_FAIL_REASON="InsufficientInstanceCapacity"
        warn "${instance_type} not available in ${region}/${subnet_id} (skipping)"
      else
        _REGION_FAIL_REASON="UnknownError"
        log "FAILED ${instance_type} in ${region}/${subnet_id}: ${launch_err}"
      fi
      instance_id=""
    done
  done

  rm -f "$launch_err_file"
  return 1
}

aws_start() {
  local instance_types_str="${GPU_INSTANCE_TYPE:-g4dn.xlarge}"
  local fallback_types="${GPU_INSTANCE_TYPE_FALLBACKS:-g5.xlarge,g6.xlarge,g4dn.2xlarge,g5.2xlarge}"
  local primary_region="${AWS_REGION:-us-east-1}"
  export _PRIMARY_REGION="$primary_region"
  local fallback_regions="${AWS_REGION_FALLBACKS:-us-east-1,us-west-2,us-east-2,eu-west-1,ap-northeast-1,ap-northeast-3}"

  # Build instance type list
  local all_types="${instance_types_str},${fallback_types}"
  local IFS_ORIG="$IFS"
  IFS=',' read -ra type_list <<< "$all_types"
  IFS="$IFS_ORIG"

  # Build region list (primary first, then fallbacks excluding primary)
  local all_regions="${primary_region}"
  IFS=',' read -ra fallback_region_list <<< "$fallback_regions"
  IFS="$IFS_ORIG"
  for r in "${fallback_region_list[@]}"; do
    r=$(echo "$r" | tr -d ' ')
    if [[ "$r" != "$primary_region" ]]; then
      all_regions="${all_regions},${r}"
    fi
  done
  IFS=',' read -ra region_list <<< "$all_regions"
  IFS="$IFS_ORIG"

  # Get registration token and runner version (region-independent)
  reg_token=$(get_registration_token) || return 1
  runner_version=$(get_runner_version) || return 1
  log "Got runner registration token (runner v${runner_version})"
  log "Instance type priority: ${all_types}"
  log "Region priority: ${all_regions}"

  local instance_id=""
  local launched_region=""
  _LAUNCHED_INSTANCE_ID=""
  local region_summary=""
  for region in "${region_list[@]}"; do
    _REGION_FAIL_REASON=""
    if aws_try_region "$region" "${type_list[@]}"; then
      instance_id="$_LAUNCHED_INSTANCE_ID"
      launched_region="$region"
      break
    fi
    region_summary+=" ${region}:${_REGION_FAIL_REASON:-Unknown}"
  done

  if [[ -z "$instance_id" ]]; then
    log "ERROR: Failed to launch EC2 instance. Region summary:${region_summary}"
    return 1
  fi

  log "EC2 instance launched: ${instance_id} in ${launched_region}"
  log "Waiting for instance to enter running state..."
  aws ec2 wait instance-running --region "${launched_region}" --instance-ids "$instance_id"
  log "Instance is running"

  echo "${instance_id}"
}

aws_stop() {
  local instance_id="$1"
  local region="${INSTANCE_REGION:-${AWS_REGION:-us-east-1}}"
  log "Terminating AWS EC2 instance: ${instance_id} in ${region}"
  aws ec2 terminate-instances --region "${region}" --instance-ids "$instance_id" > /dev/null
  log "Terminate request sent"
}

# ============================================================
# GCP provider (NOT YET TESTED)
# ============================================================

gcp_start() {
  local machine_type="${GPU_INSTANCE_TYPE:-n1-standard-4}"
  local accelerator_type="${GCP_ACCELERATOR_TYPE:-nvidia-tesla-t4}"
  local accelerator_count="${GCP_ACCELERATOR_COUNT:-1}"
  local reg_token runner_version
  reg_token=$(get_registration_token) || return 1
  runner_version=$(get_runner_version) || return 1
  log "Got runner registration token (runner v${runner_version})"

  local user_data
  user_data=$(generate_user_data "$reg_token" "$runner_version")

  local instance_name
  instance_name=$(echo "${RUNNER_NAME}" | tr '[:upper:]' '[:lower:]' | sed 's/[^a-z0-9-]/-/g' | head -c 63)

  log "Launching GCP instance (type: ${machine_type}, GPU: ${accelerator_type} x${accelerator_count})..."

  local subnet_flag=""
  if [[ -n "${GCP_SUBNET:-}" ]]; then
    subnet_flag="--subnet=${GCP_SUBNET}"
  fi

  gcloud compute instances create "${instance_name}" \
    --project="${GCP_PROJECT}" \
    --zone="${GCP_ZONE}" \
    --machine-type="${machine_type}" \
    --accelerator="type=${accelerator_type},count=${accelerator_count}" \
    --maintenance-policy=TERMINATE \
    --source-machine-image="${GCP_MACHINE_IMAGE}" \
    ${subnet_flag} \
    --metadata="startup-script=${user_data}" \
    --labels="github-actions=true,github-run-id=${GITHUB_RUN_ID:-unknown}" \
    --format="value(name)" \
    --quiet

  log "GCP instance launched: ${instance_name}"

  # Wait for instance to be running
  gcloud compute instances describe "${instance_name}" \
    --project="${GCP_PROJECT}" \
    --zone="${GCP_ZONE}" \
    --format="value(status)" > /dev/null

  echo "${instance_name}"
}

gcp_stop() {
  local instance_name="$1"
  log "Deleting GCP instance: ${instance_name}"
  gcloud compute instances delete "${instance_name}" \
    --project="${GCP_PROJECT}" \
    --zone="${GCP_ZONE}" \
    --quiet
  log "Delete request sent"
}

# ============================================================
# Commands
# ============================================================

cmd_start() {
  local provider="${CLOUD_PROVIDER:?CLOUD_PROVIDER is required (aws or gcp)}"
  local instance_id

  case "$provider" in
    aws) instance_id=$(aws_start) ;;
    gcp) instance_id=$(gcp_start) ;;
    *)
      log "ERROR: Unsupported CLOUD_PROVIDER: ${provider}"
      return 1
      ;;
  esac

  if ! wait_for_runner_online; then
    log "ERROR: Runner failed to come online. Dumping instance console output..."
    case "$provider" in
      aws)
        # Retrieve serial console output for debugging before termination
        local console_output
        console_output=$(aws ec2 get-console-output \
          --region "${AWS_DEFAULT_REGION}" \
          --instance-id "$instance_id" \
          --output text 2>/dev/null || echo "(failed to retrieve console output)")
        log "=== BEGIN CONSOLE OUTPUT ==="
        echo "$console_output" >&2
        log "=== END CONSOLE OUTPUT ==="
        aws_stop "$instance_id"
        ;;
      gcp)
        local console_output
        console_output=$(gcloud compute instances get-serial-port-output "$instance_id" \
          --project="${GCP_PROJECT}" \
          --zone="${GCP_ZONE}" 2>/dev/null || echo "(failed to retrieve console output)")
        log "=== BEGIN CONSOLE OUTPUT ==="
        echo "$console_output" >&2
        log "=== END CONSOLE OUTPUT ==="
        gcp_stop "$instance_id"
        ;;
    esac
    return 1
  fi

  echo "runner-label=${RUNNER_LABEL}" >> "$GITHUB_OUTPUT"

  log "GPU runner ready: provider=${provider}, instance=${instance_id}, label=${RUNNER_LABEL}"
}

# ============================================================
# Main
# ============================================================

case "$COMMAND" in
  start) cmd_start ;;
  *)
    echo "Usage: $0 {start}" >&2
    exit 1
    ;;
esac
