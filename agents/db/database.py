#db/database.py

from motor.motor_asyncio import AsyncIOMotorClient

MONGO_URI = "mongodb://localhost:27017"
client = AsyncIOMotorClient(MONGO_URI)
db = client.agentic_ai 


# Collections
public_docs_collection = db.public_documents
