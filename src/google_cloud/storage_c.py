from google.cloud import storage
from google.oauth2 import service_account
import os

class GCS:
    def __init__(self):
        self.KEY_PATH="src/google_cloud/config/deepbuger-d6edb63d6840.json"
        self.project_id = 'fake-id-403700'
        # buket name
        self.bucket_name = 'deep_fake_dataset'

        self.credentials=service_account.Credentials.from_service_account_file(self.KEY_PATH)


        self.client = storage.Client(credentials=self.credentials,
                                        project=self.credentials.project_id)

    def create_bucket(bucket_name):
        # storage class
        storage_class='STANDARD'
        # bucket location
        location='asia-northeast3'
        # ACL means
        predefined_acl='public-read'
        # ACL object means
        predefined_default_object_acl='public-read'

        bucket = self.client.bucket(bucket_name)
        bucket.storage_class=storage_class
        bucket = self.client.create_bucket(
            bucket,
            location=location,
            predefined_acl=predefined_acl,
            predefined_default_object_acl=predefined_default_object_acl,
        )
        return bucket

    def upload_to_bucket(self,folder_name,blob_name,file_path):
        bucket=self.client.bucket(self.bucket_name)
        blob=bucket.blob((folder_name+blob_name))
        if type(file_path)==type(bytes()):
            blob.upload_from_string(file_path,content_type='video/mp4')
        elif type(file_path)==type(str()):
            blob.upload_from_filename(file_path,content_type='video/mp4')
        blob.make_public()

        url = blob.public_url
        return url

    def select_to_bucket(client,bucket_name,blob_name,file_path):
        pass

# file_path='/Users/hyeok/Desktop/Development/Python/google_cloud/video/jennie_df.avi'
# url=upload_to_bucket(client,bucket_name,file_path.split('/')[-1][:-3],file_path)
# print(url)

# bucket=create_bucket(bucket_name=bucket_name)
# bucket.id

