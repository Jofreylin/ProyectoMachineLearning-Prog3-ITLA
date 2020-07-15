from google.cloud import automl
import os
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "automl-vision-api-4aecf082f068.json"

# TODO(developer): Uncomment and set the following variables
project_id = "automl-vision-api-283118"
model_id = "ICN3862999963772387328"
file_path = "Mano2.png"

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

if len(response.payload)==0:
    print("No se pudo identificar el modelo")
else:
    for result in response.payload:
            print("Predicted class name: {}".format(result.display_name))
            print("Predicted class score: {}".format(result.classification.score))