import weaviate
import json

import weaviate

# Correct way to pass additional config
client = weaviate.connect_to_local(
    port=8080, 
    additional_config={
        "client": {
            "timeout": 60  # Timeout duration (optional, in seconds)
        },
        "init": {
            "skip_init_checks": True  # Skip gRPC health check
        }
    }
)

class_name = "WorldWarChunk"
limit = 1000
offset = 0
all_objects = []

while True:
    response = client.collections.get(class_name).query.fetch_objects(
        limit=limit,
        offset=offset,
        return_vector=True
    )
    
    objs = response.objects
    if not objs:
        break  # no more data

    all_objects.extend(objs)
    offset += limit

# Save all vectors to file
with open("embeddings.json", "w") as f:
    json.dump(all_objects, f, indent=2)
