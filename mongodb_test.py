from pymongo.mongo_client import MongoClient

uri = "mongodb+srv://AISpec:pads123@ai-spec.deyll.mongodb.net/"

client = MongoClient(uri)

db = client['motorcycle_db']  
collection = db['motorcycles'] 


try:

    motorcycle_data = {
        "plate_number": "ABC1234",
        "region": "Metro Manila",
        "blacklisted": True,  
        "expired": False,  
        "violations": True  
    }

    collection.insert_one(motorcycle_data)
    print("Motorcycle information added to MongoDB!")

except Exception as e:
    print("Error:", e)  