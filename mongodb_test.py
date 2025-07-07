from pymongo.mongo_client import MongoClient

# It is recommended to use a more secure way to handle credentials in production,
# such as environment variables or a configuration file.
uri = "mongodb+srv://AISpec:pads123@ai-spec.deyll.mongodb.net/"

# Create a new client and connect to the server
client = MongoClient(uri)

# Select the database and collection
db = client['motorcycle_db']
collection = db['motorcycles']

try:
    # Define the motorcycle data to be inserted
    motorcycle_data = {
        "plate_number": "ABC1234",
        "region": "Metro Manila",
        "blacklisted": True,
        "expired": False,
        "violations": True
    }

    # Insert the document into the collection
    result = collection.insert_one(motorcycle_data)
    print(f"Motorcycle information added to MongoDB with document id: {result.inserted_id}")

except Exception as e:
    print(f"An error occurred: {e}")

finally:
    # It is good practice to close the connection when you are done
    client.close()