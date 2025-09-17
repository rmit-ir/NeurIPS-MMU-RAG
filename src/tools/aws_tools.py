"""
AWS Tools

Common AWS operations for managing EC2 instances.
"""

import json
import subprocess
import time
from typing import Dict, List, Optional, TypedDict


def get_aws_cli_version() -> str:
    try:
        result = subprocess.run(
            ["aws", "--version"],
            capture_output=True,
            text=True,
            check=True
        )
        # AWS CLI version output format: "aws-cli/2.x.x ..." or "aws-cli/1.x.x ..."
        version_line = result.stdout.strip() or result.stderr.strip()
        if "aws-cli/2." in version_line:
            return "2"
        elif "aws-cli/1." in version_line:
            return "1"
        else:
            # Default to version 2 behavior if we can't determine
            return "2"
    except (subprocess.CalledProcessError, FileNotFoundError):
        raise RuntimeError("AWS CLI is not installed or not accessible")


# Detect AWS CLI version at module load time
try:
    AWS_CLI_VERSION = get_aws_cli_version()
    USE_NO_CLI_PAGER = AWS_CLI_VERSION == "2"
except RuntimeError:
    # Default to version 2 behavior if detection fails
    AWS_CLI_VERSION = "2"
    USE_NO_CLI_PAGER = True


def no_pager(base_command: List[str]) -> List[str]:
    command = base_command.copy()
    if USE_NO_CLI_PAGER:
        command.append("--no-cli-pager")
    return command


class InstanceState(TypedDict):
    """EC2 Instance State information"""
    Code: int  # 80
    Name: str  # 'stopped'


class InstanceData(TypedDict):
    """EC2 Instance data structure"""
    InstanceId: str
    InstanceType: str
    State: InstanceState
    PublicIpAddress: Optional[str]
    PrivateIpAddress: str
    LaunchTime: str
    # Add other fields as needed


def run_aws_command(command: List[str]) -> Dict:
    """
    Run an AWS CLI command and return the JSON response.

    Args:
        command: List of command arguments

    Returns:
        Dict containing the JSON response from AWS CLI

    Raises:
        RuntimeError: If the command fails
    """
    command = no_pager(command)
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=True
        )
        return json.loads(result.stdout)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"AWS command failed: {e.stderr}")
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Failed to parse AWS response as JSON: {e}")


def describe_instance(instance_id: str, region: str) -> InstanceData:
    """
    Describe an EC2 instance.

    Args:
        instance_id: The EC2 instance ID
        region: The AWS region

    Returns:
        InstanceData containing instance information

    Raises:
        RuntimeError: If the command fails or instance not found
    """
    command = [
        "aws", "ec2", "describe-instances",
        "--instance-ids", instance_id,
        "--region", region,
        "--output", "json",
        "--no-cli-pager"
    ]

    response = run_aws_command(command)

    if not response.get('Reservations'):
        raise RuntimeError(
            f"Instance {instance_id} not found in region {region}")

    instances = response['Reservations'][0]['Instances']
    if not instances:
        raise RuntimeError(f"No instances found for ID {instance_id}")

    return instances[0]


def start_instance(instance_id: str, region: str) -> Dict:
    """
    Start an EC2 instance.

    Args:
        instance_id: The EC2 instance ID
        region: The AWS region

    Returns:
        Dict containing the start response

    Raises:
        RuntimeError: If the command fails
    """
    command = [
        "aws", "ec2", "start-instances",
        "--instance-ids", instance_id,
        "--region", region,
        "--output", "json",
        "--no-cli-pager"
    ]

    return run_aws_command(command)


def stop_instance(instance_id: str, region: str) -> Dict:
    """
    Stop an EC2 instance.

    Args:
        instance_id: The EC2 instance ID
        region: The AWS region

    Returns:
        Dict containing the stop response

    Raises:
        RuntimeError: If the command fails
    """
    command = [
        "aws", "ec2", "stop-instances",
        "--instance-ids", instance_id,
        "--region", region,
        "--output", "json",
        "--no-cli-pager"
    ]

    return run_aws_command(command)


def get_instance_state(instance_id: str, region: str) -> str:
    """
    'running', 'stopped', 'pending'
    """
    instance_data = describe_instance(instance_id, region)
    return instance_data['State']['Name']


def get_instance_public_ip(instance_id: str, region: str) -> Optional[str]:
    """
    Get the public IP address of an EC2 instance.

    Args:
        instance_id: The EC2 instance ID
        region: The AWS region

    Returns:
        Public IP address as string, or None if no public IP is assigned

    Raises:
        RuntimeError: If the command fails
    """
    instance_data = describe_instance(instance_id, region)
    return instance_data.get('PublicIpAddress')


def wait_for_instance_state(
    instance_id: str,
    region: str,
    target_states: List[str],
    timeout: int = 60 * 30,
    poll_interval: int = 5
) -> str:
    """
    Wait for an EC2 instance to reach a target state.

    Args:
        instance_id: The EC2 instance ID
        region: The AWS region
        target_states: Target state(s) to wait for (string or list of strings)
        timeout: Maximum time to wait in seconds (default: 1800, i.e. 30 minutes)
        poll_interval: Time between status checks in seconds (default: 5)

    Returns:
        The final state that was reached

    Raises:
        TimeoutError: If timeout is reached before target state
        RuntimeError: If AWS command fails
    """
    start_time = time.time()

    while time.time() - start_time < timeout:
        current_state = get_instance_state(instance_id, region)
        print(f"Instance {instance_id} current state: {current_state}")

        if current_state in target_states:
            return current_state

        time.sleep(poll_interval)

    raise TimeoutError(
        f"Timeout waiting for instance {instance_id} to reach state(s) {target_states}. "
        f"Current state: {get_instance_state(instance_id, region)}"
    )


def is_instance_actionable(state: str) -> bool:
    """
    Check if an instance state allows for start/stop actions.

    Args:
        state: The instance state

    Returns:
        True if the instance can be started or stopped, False otherwise
    """
    actionable_states = ['running', 'stopped', 'terminated']
    return state in actionable_states


def wait_for_actionable_state(
    instance_id: str,
    region: str,
    timeout: int = 300,
    poll_interval: int = 5
) -> str:
    """
    Wait for an EC2 instance to reach an actionable state.

    Args:
        instance_id: The EC2 instance ID
        region: The AWS region
        timeout: Maximum time to wait in seconds (default: 300)
        poll_interval: Time between status checks in seconds (default: 5)

    Returns:
        The actionable state that was reached

    Raises:
        TimeoutError: If timeout is reached before actionable state
        RuntimeError: If AWS command fails
    """
    actionable_states = ['running', 'stopped', 'terminated']
    return wait_for_instance_state(
        instance_id, region, actionable_states, timeout, poll_interval
    )
