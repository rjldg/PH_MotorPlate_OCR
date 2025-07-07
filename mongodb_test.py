# motorcycle_dal.py
# This script provides a Data Access Layer (DAL) for interacting with
# a 'motorcycles' collection in a MongoDB database.

from pymongo import MongoClient
from pymongo.errors import DuplicateKeyError

class MotorcycleDAL:
    """
    A Data Access Layer for handling CRUD operations for motorcycle documents
    in a MongoDB collection. It ensures a unique index on 'plate_number'.
    """

    def __init__(self, uri: str, db_name: str = 'motorcycle_db', collection_name: str = 'motorcycles'):
        """
        Initializes the connection to the MongoDB database and ensures the
        unique index on 'plate_number' exists.

        Args:
            uri (str): The MongoDB connection string.
            db_name (str): The name of the database.
            collection_name (str): The name of the collection.
        """
        try:
            self.client = MongoClient(uri)
            self.db = self.client[db_name]
            self.collection = self.db[collection_name]
            self._create_unique_index()
            print("Successfully connected to the database.")
        except Exception as e:
            print(f"Error: Could not connect to the database. {e}")
            self.client = None
            self.collection = None

    def _create_unique_index(self):
        """
        (Private) Creates a unique index on the 'plate_number' field to prevent duplicates.
        This method is idempotent; it won't re-create the index if it already exists.
        """
        if self.collection is not None:
            try:
                self.collection.create_index("plate_number", unique=True)
                print("Unique index on 'plate_number' is ensured.")
            except Exception as e:
                print(f"Error creating unique index: {e}")

    def insert_motorcycle(self, plate_number: str, region: str, blacklisted: bool = False, expired: bool = False, violations: bool = False) -> bool:
        """
        1. Inserts a new motorcycle document into the collection.

        Args:
            plate_number (str): The unique plate number of the motorcycle.
            region (str): The region where the motorcycle is registered.
            blacklisted (bool): Status if the motorcycle is blacklisted.
            expired (bool): Status if the registration is expired.
            violations (bool): Status if the motorcycle has violations.

        Returns:
            bool: True if insertion was successful, False otherwise.
        """
        if self.collection is None:
            return False
        try:
            motorcycle_doc = {
                "plate_number": plate_number,
                "region": region,
                "blacklisted": blacklisted,
                "expired": expired,
                "violations": violations
            }
            self.collection.insert_one(motorcycle_doc)
            print(f"Successfully inserted motorcycle with plate: {plate_number}")
            return True
        except DuplicateKeyError:
            print(f"Error: A motorcycle with plate number '{plate_number}' already exists.")
            return False
        except Exception as e:
            print(f"An unexpected error occurred during insertion: {e}")
            return False

    def _update_status_flag(self, plate_number: str, field_to_update: str, status: bool) -> bool:
        """
        (Private) Generic helper to update a single boolean status field for a motorcycle.
        """
        if self.collection is None:
            return False
        try:
            result = self.collection.update_one(
                {"plate_number": plate_number},
                {"$set": {field_to_update: status}}
            )
            if result.matched_count > 0:
                print(f"Successfully updated '{field_to_update}' for plate {plate_number} to {status}.")
                return True
            else:
                print(f"Error: No motorcycle found with plate number '{plate_number}'.")
                return False
        except Exception as e:
            print(f"An unexpected error occurred during update: {e}")
            return False

    def update_blacklisted_status(self, plate_number: str) -> bool:
        """2. Sets the 'blacklisted' status to True for a given plate number."""
        return self._update_status_flag(plate_number, "blacklisted", True)

    def update_expired_status(self, plate_number: str) -> bool:
        """3. Sets the 'expired' status to True for a given plate number."""
        return self._update_status_flag(plate_number, "expired", True)

    def update_violations_status(self, plate_number: str) -> bool:
        """4. Sets the 'violations' status to True for a given plate number."""
        return self._update_status_flag(plate_number, "violations", True)

    def clear_all_statuses(self, plate_number: str) -> bool:
        """
        5. Clears all statuses (blacklisted, expired, violations) to False
           for a given plate number.
        """
        if self.collection is None:
            return False
        try:
            statuses_to_clear = {
                "blacklisted": False,
                "expired": False,
                "violations": False
            }
            result = self.collection.update_one(
                {"plate_number": plate_number},
                {"$set": statuses_to_clear}
            )
            if result.matched_count > 0:
                print(f"Successfully cleared all statuses for plate: {plate_number}.")
                return True
            else:
                print(f"Error: No motorcycle found with plate number '{plate_number}'.")
                return False
        except Exception as e:
            print(f"An unexpected error occurred while clearing statuses: {e}")
            return False

    def delete_motorcycle(self, plate_number: str) -> bool:
        """6. Deletes a motorcycle document from the collection by its plate number."""
        if self.collection is None:
            return False
        try:
            result = self.collection.delete_one({"plate_number": plate_number})
            if result.deleted_count > 0:
                print(f"Successfully deleted motorcycle with plate: {plate_number}")
                return True
            else:
                print(f"Error: No motorcycle found with plate number '{plate_number}' to delete.")
                return False
        except Exception as e:
            print(f"An unexpected error occurred during deletion: {e}")
            return False
            
    def find_motorcycle(self, plate_number: str):
        """Finds and returns a single motorcycle document by its plate number."""
        if self.collection is None:
            return None
        return self.collection.find_one({"plate_number": plate_number})

    def close_connection(self):
        """Closes the connection to the database."""
        if self.client:
            self.client.close()
            print("Database connection closed.")


# --- Main execution block to demonstrate the DAL's functionality ---
if __name__ == "__main__":
    # IMPORTANT: Replace with your actual connection string.
    # It is recommended to use environment variables for credentials in production.
    URI = "mongodb+srv://AISpec:pads123@ai-spec.deyll.mongodb.net/"
    
    # Instantiate the Data Access Layer
    dal = MotorcycleDAL(URI)

    if dal.collection is not None:
        try:
            print("\n--- 1. Testing Insertions ---")
            dal.insert_motorcycle("PH1234", "Metro Manila")
            dal.insert_motorcycle("PH5678", "Calabarzon", violations=True)
            # This second attempt to insert PH1234 will fail due to the unique index
            dal.insert_motorcycle("PH1234", "Bicol") 

            print("\n--- 2. Testing Status Updates ---")
            print("Before update:", dal.find_motorcycle("PH1234"))
            dal.update_blacklisted_status("PH1234")
            dal.update_expired_status("PH1234")
            print("After update:", dal.find_motorcycle("PH1234"))

            print("\n--- 3. Testing Clearing Statuses ---")
            dal.clear_all_statuses("PH1234")
            print("After clearing:", dal.find_motorcycle("PH1234"))
            
            print("\n--- 4. Testing Deletion ---")
            dal.delete_motorcycle("PH5678")
            print("After deletion, searching for PH5678:", dal.find_motorcycle("PH5678"))
            # Trying to delete a non-existent document
            dal.delete_motorcycle("FAKE999")

        finally:
            # Ensure the connection is closed
            dal.close_connection()
