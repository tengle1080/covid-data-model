import boto3
import os
import logging

from libs.enums import Intervention
from libs import validate_results
from libs import build_dod_dataset

_logger = logging.getLogger(__name__)


class DatasetDeployer(object):
    def __init__(self, key="filename.csv", body="a random data", output_dir="."):
        self.s3 = boto3.client("s3")
        # Supplied by ENV on AWS
        # BUCKET_NAME format is s3://{BUCKET_NAME}
        self.bucket_name = os.environ.get("BUCKET_NAME")
        self.key = key
        self.body = body
        self.output_dir = output_dir

    def _persist_to_s3(self):
        """Persists specific data onto an s3 bucket.
        This method assumes versioned is handled on the bucket itself.
        """
        print("persisting {} to s3".format(self.key))

        response = self.s3.put_object(
            Bucket=self.bucket_name, Key=self.key, Body=self.body, ACL="public-read"
        )
        return response

    def _persist_to_local(self):
        """Persists specific data onto an s3 bucket.
        This method assumes versioned is handled on the bucket itself.
        """
        print("persisting {} to local".format(self.key))

        with open(os.path.join(self.output_dir, self.key), "wb") as f:
            # hack to allow the local writer to take either bytes or a string
            # note this assumes that all strings are given in utf-8 and not,
            # like, ASCII
            f.write(
                self.body.encode("UTF-8") if isinstance(self.body, str) else self.body
            )

        pass

    def persist(self):
        if self.bucket_name:
            self._persist_to_s3()
        else:
            self._persist_to_local()
        return


def upload_csv(key_name, csv, output_dir):
    blob = {
        "key": f"{key_name}.csv",
        "body": csv,
        "output_dir": output_dir,
    }
    obj = DatasetDeployer(**blob)
    obj.persist()
    _logger.info(f"Generated csv for {key_name}")


def deploy_shapefiles(output_dir, key, shp_bytes, sxh_bytes, dbf_bytes):
    DatasetDeployer(
        key=f"{key}.shp", body=shp_bytes.getvalue(), output_dir=output_dir
    ).persist()
    DatasetDeployer(
        key=f"{key}.shx", body=sxh_bytes.getvalue(), output_dir=output_dir
    ).persist()
    DatasetDeployer(
        key=f"{key}.dbf", body=dbf_bytes.getvalue(), output_dir=output_dir
    ).persist()
