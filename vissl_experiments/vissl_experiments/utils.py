import sys
import os
from os.path import splitext, isdir
import subprocess
import uuid

import boto3
from rastervision.pipeline.file_system.utils import (
    sync_from_dir, download_if_needed, file_exists, make_dir, unzip, get_local_path)
from rastervision.aws_s3.s3_file_system import S3FileSystem


# From https://stackoverflow.com/questions/4417546/constantly-print-subprocess-output-while-process-is-running
def _execute(cmd):
    popen = subprocess.Popen(cmd, stdout=subprocess.PIPE, universal_newlines=True)
    for stdout_line in iter(popen.stdout.readline, ""):
        yield stdout_line
    popen.stdout.close()
    return_code = popen.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, cmd)

def execute(cmd):
    for line in _execute(cmd):
        print(line, end='')

def batch_submit(command, job_prefix, attempts=3):
    job_def = os.environ['JOB_DEF']
    job_queue = os.environ['JOB_QUEUE']
    client = boto3.client('batch')
    job_name = f'{job_prefix}-{uuid.uuid4()}'

    kwargs = {
        'jobName': job_name,
        'jobQueue': job_queue,
        'jobDefinition': job_def,
        'containerOverrides': {
            'command': command
        },
        'retryStrategy': {
            'attempts': attempts
        }
    }

    job_id = client.submit_job(**kwargs)['jobId']
    msg = 'submitted job with jobName={} and jobId={}'.format(job_name, job_id)
    print(command)
    print(msg)
    return job_id

def run_remote_if_needed(args, extra_args, s3_file_args, job_prefix):
    if args.aws_batch:
        uri_dict = {}
        for file_arg in s3_file_args:
            uri_dict[file_arg] = getattr(args, file_arg)
        check_s3_exists(**uri_dict)
        command = sys.argv
        command.remove('--aws-batch')
        command.insert(0, 'python')
        batch_submit(command, job_prefix)
        exit()

def check_s3_exists(**uris):
    for name, uri in uris.items():
        if uri is not None and not (S3FileSystem.matches_uri(uri, 'r') and file_exists(uri)):
            raise Exception(f'{name}: {uri} does not exist on S3.')

def get_file(uri, cache_dir):
    path = get_local_path(uri, cache_dir)
    if file_exists(path):
        print(f'Using cached file in {path}')
    else:
        path = download_if_needed(uri, cache_dir)
    return path

def open_zip_file(zip_uri, cache_dir):
    zip_path = get_file(zip_uri, cache_dir)
    zip_dir = splitext(get_local_path(zip_uri, cache_dir))[0]
    if isdir(zip_dir):
        print(f'Using cached data in {zip_dir}')
    else:
        unzip(zip_path, zip_dir)
    return zip_dir
