from mongodb_test import MotorcycleDAL

dal = MotorcycleDAL("mongodb+srv://AISpec:pads123@ai-spec.deyll.mongodb.net/")
dal.insert_motorcycle("PCR8345", "REGION X")