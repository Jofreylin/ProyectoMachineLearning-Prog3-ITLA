from google.cloud import automl
import os
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "automl-vision-api-2e7deeee7f7f.json"

"""""
from google_auth_oauthlib import flow

# TODO: Uncomment the line below to set the `launch_browser` variable.
launch_browser = True
#
# The `launch_browser` boolean variable indicates if a local server is used
# as the callback URL in the auth flow. A value of `True` is recommended,
# but a local server does not work if accessing the application remotely,
# such as over SSH or from a remote Jupyter notebook.

appflow = flow.InstalledAppFlow.from_client_secrets_file(
    'client_secret_780908075204-t0l40kme33up86okpun6hvf425pcbqqo.apps.googleusercontent.com.json',
    scopes=['https://www.googleapis.com/auth/cloud-platform'])

if launch_browser:
    appflow.run_local_server()
else:
    appflow.run_console()

credentials = appflow.credentials

def implicit():
    from google.cloud import storage

    # If you don't specify credentials when constructing the client, the
    # client library will look for credentials in the environment.
    storage_client = storage.Client()

    # Make an authenticated API request
    buckets = list(storage_client.list_buckets())
    print(buckets)
"""""
# TODO(developer): Uncomment and set the following variables
project_id = "automl-vision-api-283118"
model_id = "ICN3862999963772387328"
file_path = "Mano3.jpeg"

prediction_client = automl.PredictionServiceClient()

# Get the full path of the model.
model_full_id = prediction_client.model_path(
    project_id, "us-central1", model_id
)

# Read the file.
with open(file_path, "rb") as content_file:
    content = content_file.read()

image = automl.types.Image(image_bytes=content)
payload = automl.types.ExamplePayload(image=image)

# params is additional domain-specific parameters.
# score_threshold is used to filter the result
# https://cloud.google.com/automl/docs/reference/rpc/google.cloud.automl.v1#predictrequest
params = {"score_threshold": "0.8"}

response = prediction_client.predict(model_full_id, payload, params)
print("Prediction results:")
for result in response.payload:
    print("Predicted class name: {}".format(result.display_name))
    print("Predicted class score: {}".format(result.classification.score))