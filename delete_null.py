from pymongo import MongoClient

MONGO_URI = "####"  
db_name = "phishing"  
collection_name = "phishing" 

client = MongoClient(MONGO_URI)
db = client[db_name]
collection = db[collection_name]

# Delete documents where "html_content" is null
def delete_null_html_content():
    query = {"html_content": None}
    result = collection.delete_many(query)  
    print(f"Deleted {result.deleted_count} documents where 'html_content' is null.")

if __name__ == "__main__":
    delete_null_html_content()
