import os
import logging
import boto3
import botocore

DEFAULT_BUCKET_NAME = 'ai-distillery'
def download_file_from_s3(key, output_path, bucket_name=DEFAULT_BUCKET_NAME):
    logger = logging.getLogger()
    s3 = boto3.resource('s3',
                        aws_access_key_id=os.environ['AWS_ACCESS_KEY_ID'],
                        aws_secret_access_key=os.environ['AWS_SECRET_ACCESS_KEY_ID'])

    try:
        logger.info('Attempting to download file from S3 at {} and saving file to local path: {}'.format(
            key, output_path
        ))
        s3.Bucket(bucket_name).download_file(key, output_path)
    except botocore.exceptions.ClientError as e:
        if e.response['Error']['Code'] == "404":
            print('404: The object at: {} does not exist.'.format(key))
            raise

if __name__ == '__main__':
    pass  # if local do below
    # download_file_from_s3('101x101maze1.png', 'data/paper_embeddings/')
